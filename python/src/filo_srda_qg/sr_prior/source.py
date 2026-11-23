from __future__ import annotations

import torch

from src.filo_srda_qg.shared_srda.source import (
    HR_SIZE,
    LR_ANCHOR_INDICES,
    spatial_lift_lr,
    temporal_lift_lr,
)


def build_bridge_source(condition_lr: torch.Tensor) -> torch.Tensor:
    """Lift mixed/unmixed LR to the nine-channel HR-grid point-mass source."""

    lifted = temporal_lift_lr(
        spatial_lift_lr(condition_lr, hr_size=HR_SIZE),
        target_length=9,
        anchor_indices=LR_ANCHOR_INDICES,
    )
    if lifted.shape[2] != 1:
        raise ValueError(f"Expected singleton PV channel, got {tuple(lifted.shape)}")
    return lifted[:, :, 0]


def fold_hr_target(hr: torch.Tensor) -> torch.Tensor:
    if hr.ndim != 6 or tuple(hr.shape[1:3]) != (9, 1):
        raise ValueError(f"Expected HR (B,9,1,Z,Y,X), got {tuple(hr.shape)}")
    return hr[:, :, 0]
