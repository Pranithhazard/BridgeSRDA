from __future__ import annotations

import torch
import torch.nn.functional as F

from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge

from .conditioning import build_dense_observation_condition


def denseobs_conditional_bridge_loss_at_time(
    bridge: ContinuousQuadraticBridge,
    model: torch.nn.Module,
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    interpolation_method: str = "periodic_bilinear",
) -> torch.Tensor:
    dense_observations = build_dense_observation_condition(
        source, observations, masks, method=interpolation_method
    )
    state = bridge.interpolate(source, target, t, noise)
    truth = bridge.target_drift(source, target, t, noise, marginalize_t1=True)
    prediction = model(state, source, dense_observations, t)
    return F.mse_loss(prediction, truth)


__all__ = ["denseobs_conditional_bridge_loss_at_time"]
