"""Isolated float64 LETKF timing alternative; not a production replacement.

Prepare once with the *actual* production operators, then call
``hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw_field)``.
The plan freezes the observation lattice, block/center ordering, localization,
inflation, ensemble size and default-dtype identity arithmetic at preparation.
Rebuild it if any of those change. Preparation is separate from update timing.

Raw fields may contain NaNs outside the fixed sensor lattice. Selected sensors
must remain finite: missing sensors are errors, never silently remasked. On this
contract gathering equals production's dense operator after its caller's
``nan_to_num(nan=1e10)``. State and observations must be float64 on plan.device.
No H or global R is retained/constructed. General ``torch.linalg.eig`` and its
real-part convention are preserved, with an explicit serial production-arithmetic
fallback for degenerate eigenspaces. Solver failures propagate, never switch eig.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class _BlockChunk:
    ids: tuple[int, ...]
    obs_counts: tuple[int, ...]
    sensors: torch.Tensor
    precision: torch.Tensor
    blocks: torch.Tensor
    valid_blocks: torch.Tensor


@dataclass(frozen=True)
class PreparedLETKFPlan:
    shape: tuple[int, int, int]
    num_ens: int
    sensor_indices: torch.Tensor
    prior: torch.Tensor
    chunks: tuple[_BlockChunk, ...]
    block_chunk: int

    @property
    def device(self) -> torch.device:
        return self.sensor_indices.device

    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(i for chunk in self.chunks for i in chunk.ids)

    def select_blocks(self, block_ids) -> PreparedLETKFPlan:
        """Slice once by original zero-based production block IDs (not points).

        Selection must be unique and drawn from this plan. Empty shards are
        allowed. Outputs are zero outside these blocks, suitable for SUM reduce.
        """
        ids = tuple(block_ids)
        if len(set(ids)) != len(ids) or not set(ids).issubset(self.block_ids):
            raise ValueError("block_ids must be unique IDs present in this plan")
        selected, chunks = set(ids), []
        for c in self.chunks:
            rows = [j for j, i in enumerate(c.ids) if i in selected]
            if rows:
                chunks.append(_BlockChunk(
                    tuple(c.ids[j] for j in rows), tuple(c.obs_counts[j] for j in rows),
                    c.sensors[rows], c.precision[rows], c.blocks[rows], c.valid_blocks[rows],
                ))
        return replace(self, chunks=tuple(chunks))

    def to(self, device: str | torch.device) -> PreparedLETKFPlan:
        """Move only packed operators; prepare/slice CPU before calling this."""
        chunks = tuple(_BlockChunk(c.ids, c.obs_counts, c.sensors.to(device),
                                   c.precision.to(device), c.blocks.to(device),
                                   c.valid_blocks.to(device)) for c in self.chunks)
        return replace(self, sensor_indices=self.sensor_indices.to(device),
                       prior=self.prior.to(device), chunks=chunks)


@torch.no_grad()
def prepare_letkf_plan(
    *, cfg, obs_matrix: torch.Tensor, dist_xy: torch.Tensor,
    dist_z: torch.Tensor, decay_factors: torch.Tensor,
    block_idx: list[torch.Tensor], center_idx: list[torch.Tensor],
    block_chunk: int = 32,
) -> PreparedLETKFPlan:
    """Validate static production setup and pack at most block_chunk blocks.

    Supplied H must be a float64, finite, one-hot selection operator. Sensor
    order is preserved (including any duplicated rows). Blocks must partition
    the state exactly; supplied centers are used, not geometrically recomputed.
    Ragged observation counts and irregular edge blocks are zero padded only
    inside each chunk. Padding contributes zero precision and is never written.
    Singleton cylinders are rejected: production's squeezed indices fail there.
    """
    shape = (cfg.nz, cfg.ny, cfg.nx)
    nstate = math.prod(shape)
    if min(shape) < 1 or cfg.num_ens < 2:
        raise ValueError("Positive grid dimensions and at least two members required")
    if not isinstance(block_chunk, int) or block_chunk < 1:
        raise ValueError("block_chunk must be a positive integer")
    if any(not math.isfinite(v) or v <= 0 for v in (
        cfg.obs_noise_std, cfg.inflation_factor,
    )):
        raise ValueError("obs_noise_std and inflation_factor must be finite/positive")
    if any(not math.isfinite(v) or v < 0 for v in (
        cfg.cylinder_radius, cfg.cylinder_height,
    )):
        raise ValueError("Cylinder dimensions must be finite/nonnegative")
    if obs_matrix.ndim != 2 or obs_matrix.shape[1] != nstate:
        raise ValueError("Observation matrix shape does not match configured grid")
    if obs_matrix.dtype != torch.float64 or not obs_matrix.shape[0]:
        raise ValueError("Observation matrix must be nonempty and float64")
    expected = (obs_matrix.shape[0], nstate)
    if any(t.shape != expected for t in (dist_xy, dist_z, decay_factors)):
        raise ValueError("Distance/decay shapes must equal observation matrix shape")
    if any(t.dtype != torch.float64 for t in (dist_xy, dist_z, decay_factors)):
        raise ValueError("Distance/decay tensors must be float64")
    if any(t.device.type != "cpu" for t in (
        obs_matrix, dist_xy, dist_z, decay_factors, *block_idx, *center_idx,
    )):
        raise ValueError("Prepare static geometry on CPU, then use plan.to(device)")
    if not block_idx or len(block_idx) != len(center_idx):
        raise ValueError("Nonempty, equally sized block and center lists required")
    device = torch.device("cpu")

    # Inspect H in bounded row slices; do not allocate another global dense H.
    sensor_parts = []
    for rows in obs_matrix.split(64):
        indices = rows.argmax(dim=1)
        values = rows.gather(1, indices[:, None])
        if not (torch.isfinite(rows).all() and (values == 1).all()
                and (torch.count_nonzero(rows, dim=1) == 1).all()):
            raise ValueError("Only finite one-hot observation selection is supported")
        sensor_parts.append(indices.to(device))
    sensors = torch.cat(sensor_parts)

    blocks, centers = [], []
    for b, c in zip(block_idx, center_idx):
        if b.ndim != 1 or b.numel() == 0 or b.dtype != torch.long:
            raise ValueError("Each block must be a nonempty 1D int64 tensor")
        if c.numel() != 1 or c.dtype != torch.long:
            raise ValueError("Each center must be a scalar int64 tensor")
        b, c = b.detach().cpu().clone(), int(c.item())
        if b.min() < 0 or b.max() >= nstate or not (b == c).any():
            raise ValueError("Block index out of range or center not in its block")
        blocks.append(b)
        centers.append(c)
    flat = torch.cat(blocks)
    if flat.numel() != nstate or not torch.equal(
        flat.sort().values, torch.arange(nstate)
    ):
        raise ValueError("Blocks must partition every state index exactly once")

    chunks = []
    # Preserve R=obs_noise_std, NOT obs_noise_std**2, and decay**2 on R^-1.
    inverse_r = torch.tensor(cfg.obs_noise_std, dtype=torch.float64).reciprocal()
    for start in range(0, len(blocks), block_chunk):
        local_obs, local_precision = [], []
        bs = blocks[start:start + block_chunk]
        for c in centers[start:start + block_chunk]:
            xy, z = dist_xy[:, c], dist_z[:, c]
            decay = decay_factors[:, c]
            if any(not torch.isfinite(t).all() for t in (xy, z, decay)):
                raise ValueError(f"Nonfinite static geometry/decay at center {c}")
            chosen = torch.nonzero(
                (xy <= cfg.cylinder_radius) & (z <= cfg.cylinder_height / 2),
                as_tuple=True,
            )[0]
            if chosen.numel() == 0:
                raise ValueError(f"No observations within cylinder at center {c}")
            if chosen.numel() == 1:
                raise ValueError("Singleton cylinder unsupported by production squeeze")
            d = decay[chosen].to(device=device, dtype=torch.float64)
            local_obs.append(chosen.to(device))
            local_precision.append(inverse_r.to(device) * (d * d))
        nobs = max(t.numel() for t in local_obs)
        npoints = max(b.numel() for b in bs)
        packed_obs = torch.zeros((len(bs), nobs), dtype=torch.long, device=device)
        precision = torch.zeros((len(bs), nobs), dtype=torch.float64, device=device)
        packed_blocks = torch.zeros((len(bs), npoints), dtype=torch.long, device=device)
        valid = torch.zeros_like(packed_blocks, dtype=torch.bool)
        for i, (b, o, p) in enumerate(zip(bs, local_obs, local_precision)):
            packed_obs[i, :o.numel()] = o
            precision[i, :o.numel()] = p
            packed_blocks[i, :b.numel()] = b.to(device)
            valid[i, :b.numel()] = True
        chunks.append(_BlockChunk(
            tuple(range(start, start + len(bs))), tuple(o.numel() for o in local_obs),
            packed_obs, precision, packed_blocks, valid,
        ))

    # Production creates eye in the default dtype *before* scaling. Retain that
    # rounding even when the state is float64; do not silently improve it here.
    prior = (cfg.num_ens - 1) * torch.eye(cfg.num_ens).to(device)
    prior = prior / cfg.inflation_factor
    return PreparedLETKFPlan(
        shape, cfg.num_ens, sensors, prior, tuple(chunks), block_chunk,
    )


@torch.no_grad()
def hr_assimilate_batched(
    *, plan: PreparedLETKFPlan, hr_pv: torch.Tensor,
    gt_added_noise: torch.Tensor, fallback: str = "production",
    diagnostics: dict | None = None,
) -> torch.Tensor:
    """Return member-wise (M,Z,Y,X) analysis using prepared batched general eig.

    CPU float64 equivalence tests use rtol=1e-10, atol=1e-11 against actual
    production, not an independently reimplemented oracle. No GPU equivalence
    or speedup is implied. Repeated eigenvalues/complex eigenvectors trigger a
    production-arithmetic, serial general-eig fallback (local dense R only).
    Pass fallback="error" to disallow it. Fallback IDs are written to an explicit
    diagnostics dict, or warned when no dict is supplied. No eigh/jitter/clipping.
    """
    if fallback not in ("production", "error"):
        raise ValueError("fallback must be 'production' or 'error'")
    if tuple(hr_pv.shape) != (plan.num_ens, *plan.shape):
        raise ValueError("State shape/ensemble size does not match prepared plan")
    if tuple(gt_added_noise.shape) != plan.shape:
        raise ValueError("Observed field shape does not match prepared plan")
    for t in (hr_pv, gt_added_noise):
        if t.dtype != torch.float64 or t.device != plan.device:
            raise ValueError("State and observations must be float64 on plan.device")
    state = hr_pv.reshape(plan.num_ens, -1)
    obs = gt_added_noise.reshape(-1)[plan.sensor_indices]
    if not torch.isfinite(state).all() or not torch.isfinite(obs).all():
        raise ValueError("Nonfinite state or selected sensor; fixed mask cannot change")
    mean = state.mean(dim=0, keepdim=True)
    anomalies = state - mean
    obs_mean = mean[0, plan.sensor_indices]
    obs_anomalies = anomalies[:, plan.sensor_indices].t()
    innovation = obs - obs_mean
    output = torch.zeros_like(state)  # Outside selected blocks: additive identity.
    fallback_ids = []

    for chunk in plan.chunks:
        y = obs_anomalies[chunk.sensors]  # (blocks, local obs, M)
        c = y.transpose(1, 2) * chunk.precision[:, None, :]
        system = plan.prior + c.bmm(y)
        eigenvalues, eigenvectors = torch.linalg.eig(system)
        values, vectors = eigenvalues.real, eigenvectors.real
        if not torch.isfinite(values).all() or (values <= 0).any():
            raise FloatingPointError("Invalid LETKF eigenvalues; no fallback applied")
        gaps = values.sort(dim=1).values.diff(dim=1).abs()
        tolerance = 128 * torch.finfo(values.dtype).eps * values.abs().amax(dim=1).clamp_min(1)
        risky = (gaps <= tolerance[:, None]).any(dim=1)
        risky |= (eigenvectors.imag != 0).any(dim=(1, 2))
        risky |= (eigenvalues.imag != 0).any(dim=1)
        bad = torch.where(risky)[0].tolist()
        if bad and fallback == "error":
            raise RuntimeError("Degenerate/complex general eig requires production fallback")
        fallback_ids.extend(chunk.ids[i] for i in bad)
        good = torch.where(~risky)[0]
        vectors, values, c = vectors[good], values[good], c[good]
        vectors_inv = torch.linalg.inv(vectors)
        reciprocal = values.reciprocal()
        pa = (vectors * reciprocal[:, None, :]).bmm(vectors_inv)
        wa = (plan.num_ens - 1) ** 0.5 * (
            vectors * reciprocal.sqrt()[:, None, :]
        ).bmm(vectors_inv)
        weights_mean = pa.bmm(c).bmm(innovation[chunk.sensors[good], None])
        weights = wa + weights_mean
        blocks = chunk.blocks[good]
        x = anomalies[:, blocks].permute(1, 0, 2)
        analysis = mean[0, blocks][:, None, :] + weights.transpose(1, 2).bmm(x)
        valid = chunk.valid_blocks[good].reshape(-1)
        output[:, blocks.reshape(-1)[valid]] = (
            analysis.permute(1, 0, 2).reshape(plan.num_ens, -1)[:, valid]
        )
        for i in bad:
            sensors = chunk.sensors[i, :chunk.obs_counts[i]]
            y = obs_anomalies[sensors]
            c = y.t().mm(torch.diag(chunk.precision[i, :sensors.numel()]))
            ev, v = torch.linalg.eig(plan.prior + c.mm(y))
            v, ev = v.real, ev.real
            inverse = torch.linalg.inv(v)
            reciprocal = torch.diag(1.0 / ev)
            pa = v.mm(reciprocal).mm(inverse)
            wa = (plan.num_ens - 1) ** 0.5 * v.mm(reciprocal.sqrt()).mm(inverse)
            wa = wa + pa.mm(c).mm(innovation[sensors, None])
            b = chunk.blocks[i, chunk.valid_blocks[i]]
            output[:, b] = mean[:, b] + wa.t().mm(anomalies[:, b])
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite LETKF analysis; no fallback applied")
    if diagnostics is not None:
        diagnostics.update(blocks=len(plan.block_ids), production_fallback_ids=fallback_ids,
                           production_fallback_blocks=len(fallback_ids))
    elif fallback_ids:
        warnings.warn(f"Production-arithmetic eig fallback for block IDs {fallback_ids}",
                      RuntimeWarning, stacklevel=2)
    return output.reshape_as(hr_pv)
