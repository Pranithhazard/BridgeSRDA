from __future__ import annotations

import numpy as np
import torch


def normalized_windows_to_physical(
    values: torch.Tensor,
    *,
    pv_min: float,
    pv_max: float,
) -> torch.Tensor:
    """Convert (...,T,Z,64,128) normalized windows and restore y=65 boundary."""

    physical = values * float(pv_max - pv_min) + float(pv_min)
    zeros = torch.zeros(
        *physical.shape[:-2],
        1,
        physical.shape[-1],
        dtype=physical.dtype,
        device=physical.device,
    )
    return torch.cat([physical, zeros], dim=-2)
