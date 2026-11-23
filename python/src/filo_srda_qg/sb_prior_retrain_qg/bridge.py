from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class ContinuousQuadraticBridgeConfig:
    epsilon: float = 0.2
    noise_exponent: int = 1
    interior_strata: int = 30

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.noise_exponent < 1:
            raise ValueError("noise_exponent must be a positive integer")
        if self.interior_strata <= 0:
            raise ValueError("interior_strata must be positive")


@dataclasses.dataclass(frozen=True)
class TimeRole:
    kind: str
    stratum: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"endpoint_zero", "endpoint_one", "interior"}:
            raise ValueError(f"Unknown time role {self.kind!r}")
        if self.kind == "interior" and self.stratum is None:
            raise ValueError("Interior roles require a stratum")
        if self.kind != "interior" and self.stratum is not None:
            raise ValueError("Endpoint roles cannot have a stratum")


def _group_seed(seed: int, epoch: int, optimizer_group: int) -> int:
    modulus = 2**63 - 1
    return int(
        (int(seed) + 1_000_003 * int(epoch) + 10_000_019 * int(optimizer_group))
        % modulus
    )


class StratifiedTimeAssignment:
    """Deterministic global-batch time roles for DDP gradient accumulation."""

    def __init__(
        self,
        *,
        world_size: int,
        accumulation_steps: int,
        interior_strata: int = 30,
        seed: int = 42,
    ) -> None:
        self.world_size = int(world_size)
        self.accumulation_steps = int(accumulation_steps)
        self.interior_strata = int(interior_strata)
        self.seed = int(seed)
        self.mode = "continuous_stratified"
        self.dense_grid_intervals: int | None = None
        if self.world_size <= 0 or self.accumulation_steps <= 0:
            raise ValueError("world_size and accumulation_steps must be positive")
        expected = self.interior_strata + 2
        if self.global_batch_size != expected:
            raise ValueError(
                "Stratified schedule requires global batch interior_strata + 2: "
                f"got {self.global_batch_size}, expected {expected}"
            )

    @property
    def global_batch_size(self) -> int:
        return self.world_size * self.accumulation_steps

    def roles(self, *, epoch: int, optimizer_group: int) -> list[TimeRole]:
        roles = [TimeRole("endpoint_zero"), TimeRole("endpoint_one")]
        roles.extend(TimeRole("interior", index) for index in range(self.interior_strata))
        generator = torch.Generator(device="cpu").manual_seed(
            _group_seed(self.seed, epoch, optimizer_group)
        )
        order = torch.randperm(len(roles), generator=generator).tolist()
        return [roles[index] for index in order]

    def role(
        self,
        *,
        epoch: int,
        optimizer_group: int,
        rank: int,
        accumulation_index: int,
    ) -> TimeRole:
        if not (0 <= rank < self.world_size):
            raise ValueError("rank is outside world size")
        if not (0 <= accumulation_index < self.accumulation_steps):
            raise ValueError("accumulation index is outside its group")
        # Interleave ranks at each microstep. This is stable across process launch order.
        slot = accumulation_index * self.world_size + rank
        return self.roles(epoch=epoch, optimizer_group=optimizer_group)[slot]

    def time(
        self,
        *,
        epoch: int,
        optimizer_group: int,
        rank: int,
        accumulation_index: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, TimeRole]:
        role = self.role(
            epoch=epoch,
            optimizer_group=optimizer_group,
            rank=rank,
            accumulation_index=accumulation_index,
        )
        if role.kind == "endpoint_zero":
            value = 0.0
        elif role.kind == "endpoint_one":
            value = 1.0
        else:
            assert role.stratum is not None
            generator = torch.Generator(device="cpu").manual_seed(
                _group_seed(self.seed + 97_409 * (role.stratum + 1), epoch, optimizer_group)
            )
            jitter = float(torch.rand((), generator=generator).item())
            value = (role.stratum + jitter) / float(self.interior_strata)
            # torch.rand is [0,1), but protect the contract from future backend changes.
            value = min(max(value, math.nextafter(0.0, 1.0)), math.nextafter(1.0, 0.0))
        return torch.tensor([value], device=device, dtype=dtype), role

    def deterministic_validation_time(
        self, sample_index: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, TimeRole]:
        if sample_index < 0:
            raise ValueError("sample_index must be nonnegative")
        group, slot = divmod(sample_index, self.global_batch_size)
        accumulation_index, rank = divmod(slot, self.world_size)
        return self.time(
            epoch=0,
            optimizer_group=group,
            rank=rank,
            accumulation_index=accumulation_index,
            device=device,
            dtype=dtype,
        )


def make_time_assignment(
    *,
    config: dict,
    world_size: int,
) -> StratifiedTimeAssignment:
    time_config = config.get("time_sampling", {})
    mode = str(time_config.get("mode", "continuous_stratified"))
    common = dict(
        world_size=world_size,
        accumulation_steps=int(config["train"]["gradient_accumulation_steps"]),
        interior_strata=int(config["bridge"]["interior_strata"]),
        seed=int(config["train"]["seed"]),
    )
    if mode == "continuous_stratified":
        return StratifiedTimeAssignment(**common)
    raise ValueError(f"Unknown time_sampling.mode: {mode!r}")


def generalized_bare_interval_variance(
    t0: float, t1: float, epsilon: float, noise_exponent: int
) -> float:
    if epsilon <= 0.0 or noise_exponent < 1 or not (0.0 <= t0 < t1 <= 1.0):
        raise ValueError((t0, t1, epsilon, noise_exponent))
    power = 2 * int(noise_exponent) + 1
    return (epsilon**2 / power) * (
        (1.0 - t0) ** power - (1.0 - t1) ** power
    )


class ContinuousQuadraticBridge:
    def __init__(self, config: ContinuousQuadraticBridgeConfig):
        self.config = config

    @staticmethod
    def _wide(t: torch.Tensor, ndim: int) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError("t must have shape (batch,)")
        return t.reshape((t.shape[0],) + (1,) * (ndim - 1))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.config.epsilon * (1.0 - t).pow(self.config.noise_exponent)

    def interpolate(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        if source.shape != target.shape or source.shape != noise.shape:
            raise ValueError("source, target and noise shapes must match")
        tw = self._wide(t.to(source), source.ndim)
        sigma = self.config.epsilon * (1.0 - tw).pow(self.config.noise_exponent)
        return (1.0 - tw) * source + tw.square() * target + sigma * torch.sqrt(tw) * noise

    def target_drift(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
        *,
        marginalize_t1: bool = True,
    ) -> torch.Tensor:
        if source.shape != target.shape or source.shape != noise.shape:
            raise ValueError("source, target and noise shapes must match")
        tw = self._wide(t.to(source), source.ndim)
        p = self.config.noise_exponent
        stochastic = (
            -p
            * self.config.epsilon
            * (1.0 - tw).pow(p - 1)
            * torch.sqrt(tw)
            * noise
        )
        drift = -source + 2.0 * tw * target + stochastic
        if marginalize_t1:
            endpoint = (t == 1.0).reshape((t.shape[0],) + (1,) * (source.ndim - 1))
            drift = torch.where(endpoint, -source + 2.0 * target, drift)
        return drift

    def loss_at_time(
        self,
        model: torch.nn.Module,
        *,
        source: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        state = self.interpolate(source, target, t, noise)
        truth = self.target_drift(source, target, t, noise, marginalize_t1=True)
        return F.mse_loss(model(state, source, t), truth)

    def interval_variance(self, t0: float, t1: float) -> float:
        return generalized_bare_interval_variance(
            t0, t1, self.config.epsilon, self.config.noise_exponent
        )
