from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


HR_SIZE = (16, 64, 128)
LR_ANCHOR_INDICES = (0, 4, 8)


def spatial_lift_lr(
    lr: torch.Tensor,
    *,
    hr_size: tuple[int, int, int] = HR_SIZE,
) -> torch.Tensor:
    """Nearest-exact lift from (B,3,1,4,16,32) to the HR spatial grid."""

    if lr.ndim != 6:
        raise ValueError(f"Expected LR (B,T,C,Z,Y,X), got {tuple(lr.shape)}")
    b, t, c, z, y, x = lr.shape
    lifted = F.interpolate(
        lr.reshape(b * t, c, z, y, x),
        size=hr_size,
        mode="nearest-exact",
    )
    return lifted.reshape(b, t, c, *hr_size)


def temporal_lift_lr(
    lr_hr: torch.Tensor,
    *,
    target_length: int = 9,
    anchor_indices: Iterable[int] = LR_ANCHOR_INDICES,
) -> torch.Tensor:
    """Piecewise-linear interpolation of three LR frames onto nine HR times."""

    if lr_hr.ndim != 6 or lr_hr.shape[1] != 3:
        raise ValueError(
            f"Expected lifted LR (B,3,C,Z,Y,X), got {tuple(lr_hr.shape)}"
        )
    anchors = tuple(int(i) for i in anchor_indices)
    if anchors != (0, 4, 8) or target_length != 9:
        raise ValueError(
            "DirectSI MVP requires three anchors at target indices (0,4,8)"
        )
    first, middle, last = lr_hr[:, 0], lr_hr[:, 1], lr_hr[:, 2]
    frames = []
    for index in range(target_length):
        if index <= anchors[1]:
            weight = float(index - anchors[0]) / float(anchors[1] - anchors[0])
            frame = (1.0 - weight) * first + weight * middle
        else:
            weight = float(index - anchors[1]) / float(anchors[2] - anchors[1])
            frame = (1.0 - weight) * middle + weight * last
        frames.append(frame)
    return torch.stack(frames, dim=1)


def build_source_mean(
    source_lr: torch.Tensor,
    *,
    hr_size: tuple[int, int, int] = HR_SIZE,
) -> torch.Tensor:
    return temporal_lift_lr(spatial_lift_lr(source_lr, hr_size=hr_size))


def sample_source_from_mean(
    source_mean: torch.Tensor,
    sigma: torch.Tensor,
    *,
    alpha: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if source_mean.ndim != 6:
        raise ValueError(
            f"Expected source mean (B,9,1,Z,Y,X), got {tuple(source_mean.shape)}"
        )
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    sigma = sigma.to(device=source_mean.device, dtype=source_mean.dtype)
    noise = torch.randn(
        source_mean.shape,
        device=source_mean.device,
        dtype=source_mean.dtype,
        generator=generator,
    )
    return source_mean + float(alpha) * sigma * noise


def sample_source(
    source_lr: torch.Tensor,
    sigma: torch.Tensor,
    *,
    alpha: float,
    generator: torch.Generator | None = None,
    hr_size: tuple[int, int, int] = HR_SIZE,
) -> torch.Tensor:
    return sample_source_from_mean(
        build_source_mean(source_lr, hr_size=hr_size),
        sigma,
        alpha=alpha,
        generator=generator,
    )


def fold_pv(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 6 or x.shape[2] != 1:
        raise ValueError(f"Expected (B,T,1,Z,Y,X), got {tuple(x.shape)}")
    return x[:, :, 0]


def unfold_pv(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected (B,T,Z,Y,X), got {tuple(x.shape)}")
    return x[:, :, None]

