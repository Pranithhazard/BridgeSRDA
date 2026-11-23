from __future__ import annotations

import torch


OBSERVATION_FRAME_INDICES = (0, 4, 8)


def _fold_observation_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim != 6 or tuple(value.shape[1:3]) != (3, 1):
        raise ValueError(
            f"{name} must have shape (B,3,1,Z,Y,X), got {tuple(value.shape)}"
        )
    return value[:, :, 0]


def build_observation_condition(
    source: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Return three masked innovations followed by three explicit mask channels."""
    if source.ndim != 5 or source.shape[1] != 9:
        raise ValueError(f"source must have shape (B,9,Z,Y,X), got {tuple(source.shape)}")
    obs = _fold_observation_tensor(observations, name="observations")
    mask = _fold_observation_tensor(masks, name="masks")
    if obs.shape != mask.shape or obs.shape[0] != source.shape[0]:
        raise ValueError("observations, masks, and source batch/spatial shapes must agree")
    if tuple(obs.shape[-3:]) != tuple(source.shape[-3:]):
        raise ValueError("observation and source spatial shapes must agree")
    source_at_observation_times = source[:, OBSERVATION_FRAME_INDICES]
    innovation = mask * (obs - source_at_observation_times)
    return torch.cat([innovation, mask], dim=1)
