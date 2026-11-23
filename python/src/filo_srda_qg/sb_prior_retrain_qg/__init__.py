"""Continuous-time Schrödinger-bridge prior retraining (Experiment 071B)."""

from .bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    StratifiedTimeAssignment,
    generalized_bare_interval_variance,
    make_time_assignment,
)
from .ema import EMA
from .sampler import BareHeunResult, sample_bare_heun_exact_noise

__all__ = [
    "BareHeunResult",
    "ContinuousQuadraticBridge",
    "ContinuousQuadraticBridgeConfig",
    "EMA",
    "StratifiedTimeAssignment",
    "generalized_bare_interval_variance",
    "make_time_assignment",
    "sample_bare_heun_exact_noise",
]
