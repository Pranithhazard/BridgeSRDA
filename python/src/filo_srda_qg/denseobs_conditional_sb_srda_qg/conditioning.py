from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


DENSE_OBSERVATION_SLOT_INDICES = (0, 1)
DENSE_OBSERVATION_FRAME_INDICES = (0, 4)


@dataclass(frozen=True)
class _InterpolationPlan:
    rows: tuple[int, ...]
    cols: tuple[int, ...]
    y0: torch.Tensor
    y1: torch.Tensor
    wy: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    wx: torch.Tensor


_PLAN_CACHE: dict[tuple[object, ...], _InterpolationPlan] = {}


def clear_interpolation_cache() -> None:
    _PLAN_CACHE.clear()


def _validate_raw_contract(
    source: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    if source.ndim != 5 or source.shape[1] != 9:
        raise ValueError(f"source must have shape (B,9,Z,Y,X), got {tuple(source.shape)}")
    expected = (source.shape[0], 3, 1, *source.shape[-3:])
    if tuple(observations.shape) != expected or tuple(masks.shape) != expected:
        raise ValueError(
            "observations and masks must both have shape "
            f"{expected}, got {tuple(observations.shape)} and {tuple(masks.shape)}"
        )
    if not torch.isfinite(source).all() or not torch.isfinite(observations).all():
        raise ValueError("source and observations must be finite")
    if not torch.isfinite(masks).all():
        raise ValueError("masks must be finite")


def _cartesian_axes(mask: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return Cartesian sensor axes or reject a non-Cartesian 3-D mask."""
    # Production observations use the same horizontal grid on every z layer.
    binary = mask > 0.5
    if not torch.equal(binary, binary[0:1].expand_as(binary)):
        raise ValueError("dense-observation interpolation requires the same mask on every z layer")
    plane = binary[0]
    rows_t = torch.nonzero(plane.any(dim=1), as_tuple=False).flatten()
    cols_t = torch.nonzero(plane.any(dim=0), as_tuple=False).flatten()
    if rows_t.numel() == 0 or cols_t.numel() == 0:
        return (), ()
    expected = torch.zeros_like(plane)
    expected[rows_t[:, None], cols_t[None, :]] = True
    if not torch.equal(plane, expected):
        raise ValueError("production dense-observation path requires a Cartesian mask")
    return (
        tuple(int(value) for value in rows_t.cpu().tolist()),
        tuple(int(value) for value in cols_t.cpu().tolist()),
    )


def _brackets_nonperiodic(
    coordinates: tuple[int, ...], size: int
) -> tuple[list[int], list[int], list[float]]:
    if not coordinates:
        raise ValueError("empty coordinate axis")
    lo: list[int] = []
    hi: list[int] = []
    weights: list[float] = []
    for point in range(size):
        if point <= coordinates[0]:
            lo.append(0)
            hi.append(0)
            weights.append(0.0)
            continue
        if point >= coordinates[-1]:
            last = len(coordinates) - 1
            lo.append(last)
            hi.append(last)
            weights.append(0.0)
            continue
        upper = next(index for index, value in enumerate(coordinates) if value >= point)
        lower = upper - 1
        span = coordinates[upper] - coordinates[lower]
        lo.append(lower)
        hi.append(upper)
        weights.append((point - coordinates[lower]) / span)
    return lo, hi, weights


def _brackets_periodic(
    coordinates: tuple[int, ...], size: int
) -> tuple[list[int], list[int], list[float]]:
    if not coordinates:
        raise ValueError("empty coordinate axis")
    lo: list[int] = []
    hi: list[int] = []
    weights: list[float] = []
    extended = (coordinates[-1] - size, *coordinates, coordinates[0] + size)
    for point in range(size):
        upper_ext = next(index for index, value in enumerate(extended) if value >= point)
        lower_ext = upper_ext - 1
        left = extended[lower_ext]
        right = extended[upper_ext]
        lo.append((lower_ext - 1) % len(coordinates))
        hi.append((upper_ext - 1) % len(coordinates))
        weights.append(0.0 if right == left else (point - left) / (right - left))
    return lo, hi, weights


def _plan(
    *,
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    y_size: int,
    x_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _InterpolationPlan:
    key = (
        y_size,
        x_size,
        rows,
        cols,
        device.type,
        device.index,
        dtype,
    )
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached
    y0, y1, wy = _brackets_nonperiodic(rows, y_size)
    x0, x1, wx = _brackets_periodic(cols, x_size)
    cached = _InterpolationPlan(
        rows=rows,
        cols=cols,
        y0=torch.tensor(y0, device=device, dtype=torch.long),
        y1=torch.tensor(y1, device=device, dtype=torch.long),
        wy=torch.tensor(wy, device=device, dtype=dtype).view(1, y_size, 1),
        x0=torch.tensor(x0, device=device, dtype=torch.long),
        x1=torch.tensor(x1, device=device, dtype=torch.long),
        wx=torch.tensor(wx, device=device, dtype=dtype).view(1, 1, x_size),
    )
    _PLAN_CACHE[key] = cached
    return cached


def _periodic_bilinear(
    observation: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    z_size, y_size, x_size = observation.shape
    rows, cols = _cartesian_axes(mask)
    if not rows:
        raise ValueError("missing observation slot must be handled by source fallback")
    plan = _plan(
        rows=rows,
        cols=cols,
        y_size=y_size,
        x_size=x_size,
        device=observation.device,
        dtype=observation.dtype,
    )
    row_index = torch.tensor(rows, device=observation.device, dtype=torch.long)
    col_index = torch.tensor(cols, device=observation.device, dtype=torch.long)
    coarse = observation.index_select(1, row_index).index_select(2, col_index)
    left = coarse.index_select(2, plan.x0)
    right = coarse.index_select(2, plan.x1)
    along_x = left + plan.wx * (right - left)
    lower = along_x.index_select(1, plan.y0)
    upper = along_x.index_select(1, plan.y1)
    dense = lower + plan.wy * (upper - lower)
    # This overwrite makes the sensor preservation contract exact, even in low precision.
    return torch.where(mask > 0.5, observation, dense)


def _audit_bicubic(
    observation: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """DiffSRDA-like dense bicubic field for audit comparison only.

    This deliberately uses standard bicubic image interpolation and border
    extension. It is not the production method because it is not periodic in x
    and can overshoot the normalized range.
    """
    rows, cols = _cartesian_axes(mask)
    if not rows:
        raise ValueError("missing observation slot must be handled by source fallback")
    row_index = torch.tensor(rows, device=observation.device, dtype=torch.long)
    col_index = torch.tensor(cols, device=observation.device, dtype=torch.long)
    coarse = observation.index_select(1, row_index).index_select(2, col_index)
    dense = F.interpolate(
        coarse[:, None],
        size=observation.shape[-2:],
        mode="bicubic",
        align_corners=True,
    )[:, 0]
    return torch.where(mask > 0.5, observation, dense)


def build_dense_observation_condition(
    source: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    method: Literal["periodic_bilinear", "diffsrda_bicubic"] = "periodic_bilinear",
) -> torch.Tensor:
    """Densify past/current sparse observations into two HR-grid fields.

    Observation noise and normalization are already applied by the dataset.
    A completely missing selected slot falls back to the corresponding lifted
    LR source frame, which is the neutral no-new-information condition.
    """
    _validate_raw_contract(source, observations, masks)
    interpolation = {
        "periodic_bilinear": _periodic_bilinear,
        "diffsrda_bicubic": _audit_bicubic,
    }.get(method)
    if interpolation is None:
        raise ValueError(f"unknown dense-observation interpolation method: {method}")

    dense_slots: list[torch.Tensor] = []
    for batch_index in range(source.shape[0]):
        per_sample: list[torch.Tensor] = []
        for slot, frame in zip(
            DENSE_OBSERVATION_SLOT_INDICES, DENSE_OBSERVATION_FRAME_INDICES
        ):
            observation = observations[batch_index, slot, 0]
            mask = masks[batch_index, slot, 0]
            if not bool((mask > 0.5).any()):
                per_sample.append(source[batch_index, frame])
            else:
                per_sample.append(interpolation(observation, mask))
        dense_slots.append(torch.stack(per_sample, dim=0))
    dense = torch.stack(dense_slots, dim=0)
    expected = (source.shape[0], 2, *source.shape[-3:])
    if tuple(dense.shape) != expected or not torch.isfinite(dense).all():
        raise RuntimeError(f"invalid dense observation result: {tuple(dense.shape)}")
    return dense


__all__ = [
    "DENSE_OBSERVATION_FRAME_INDICES",
    "DENSE_OBSERVATION_SLOT_INDICES",
    "build_dense_observation_condition",
    "clear_interpolation_cache",
]
