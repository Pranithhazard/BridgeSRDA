"""Dense-observation conditional Schrödinger-bridge SRDA (experiment 073A)."""

from .bridge import denseobs_conditional_bridge_loss_at_time
from .conditioning import (
    DENSE_OBSERVATION_FRAME_INDICES,
    DENSE_OBSERVATION_SLOT_INDICES,
    build_dense_observation_condition,
)
from .model import (
    DenseObsConditionalSBUNet3D,
    denseobs_model_signature,
    make_denseobs_conditional_sb_model,
)
from .sampler import sample_denseobs_conditional_bare_heun_exact_noise

__all__ = [
    "DENSE_OBSERVATION_FRAME_INDICES",
    "DENSE_OBSERVATION_SLOT_INDICES",
    "DenseObsConditionalSBUNet3D",
    "build_dense_observation_condition",
    "denseobs_model_signature",
    "denseobs_conditional_bridge_loss_at_time",
    "make_denseobs_conditional_sb_model",
    "sample_denseobs_conditional_bare_heun_exact_noise",
]
