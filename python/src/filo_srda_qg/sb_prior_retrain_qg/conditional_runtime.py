from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable


MASKED_INNOVATION_KIND = "masked_normalized_innovation_plus_mask"
DENSE_OBSERVATION_KIND = "dense_interpolated_observation_fields"


@dataclass(frozen=True)
class ConditionalRuntime:
    kind: str
    make_dataset: Callable[..., Any]
    make_model: Callable[..., Any]
    loss_at_time: Callable[..., Any]
    evaluate_panel: Callable[..., Any]
    audit_time: Callable[..., Any]
    sample: Callable[..., Any]
    load_training_checkpoint: Callable[..., Any]
    model_signature: str
    observation_contract: dict[str, Any]
    description: str


def conditioning_kind(config: dict[str, Any]) -> str | None:
    conditioning = config.get("conditioning", {})
    if not bool(conditioning.get("enabled", False)):
        return None
    return str(conditioning.get("kind", MASKED_INNOVATION_KIND))


def get_conditional_runtime(config: dict[str, Any]) -> ConditionalRuntime | None:
    kind = conditioning_kind(config)
    if kind is None:
        return None

    from src.filo_srda_qg.conditional_sb_srda_qg.data import (
        make_conditional_sb_dataset,
    )

    if kind == MASKED_INNOVATION_KIND:
        from src.filo_srda_qg.conditional_sb_srda_qg.bridge import (
            conditional_bridge_loss_at_time,
        )
        from src.filo_srda_qg.conditional_sb_srda_qg.checkpoint import (
            load_training_checkpoint,
        )
        from src.filo_srda_qg.conditional_sb_srda_qg.model import (
            make_conditional_sb_model,
        )
        from src.filo_srda_qg.conditional_sb_srda_qg.sampler import (
            sample_conditional_bare_heun_exact_noise,
        )
        from src.filo_srda_qg.conditional_sb_srda_qg.validation import (
            audit_conditional_time,
            evaluate_conditional_panel,
        )

        return ConditionalRuntime(
            kind=kind,
            make_dataset=make_conditional_sb_dataset,
            make_model=make_conditional_sb_model,
            loss_at_time=conditional_bridge_loss_at_time,
            evaluate_panel=evaluate_conditional_panel,
            audit_time=audit_conditional_time,
            sample=sample_conditional_bare_heun_exact_noise,
            load_training_checkpoint=load_training_checkpoint,
            model_signature="ConditionalSBUNet3D:(9+9+3innovation+3mask)->9",
            observation_contract={
                "condition": MASKED_INNOVATION_KIND,
                "network_observation_frames": [0, 4, 8],
                "raw_slots_retained": [0, 1, 2],
                "future_slot_omitted_from_network": False,
            },
            description="072 conditional",
        )

    if kind == DENSE_OBSERVATION_KIND:
        from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.bridge import (
            denseobs_conditional_bridge_loss_at_time,
        )
        from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.checkpoint import (
            load_training_checkpoint,
        )
        from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.model import (
            denseobs_model_signature,
            make_denseobs_conditional_sb_model,
        )
        from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler import (
            sample_denseobs_conditional_bare_heun_exact_noise,
        )
        from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.validation import (
            audit_denseobs_conditional_time,
            evaluate_denseobs_conditional_panel,
        )

        interpolation = str(
            config.get("conditioning", {}).get(
                "interpolation_method", "periodic_bilinear"
            )
        )
        return ConditionalRuntime(
            kind=kind,
            make_dataset=make_conditional_sb_dataset,
            make_model=make_denseobs_conditional_sb_model,
            loss_at_time=partial(
                denseobs_conditional_bridge_loss_at_time,
                interpolation_method=interpolation,
            ),
            evaluate_panel=partial(
                evaluate_denseobs_conditional_panel,
                interpolation_method=interpolation,
            ),
            audit_time=partial(
                audit_denseobs_conditional_time,
                interpolation_method=interpolation,
            ),
            sample=partial(
                sample_denseobs_conditional_bare_heun_exact_noise,
                interpolation_method=interpolation,
            ),
            load_training_checkpoint=load_training_checkpoint,
            model_signature=denseobs_model_signature(config["model"]),
            observation_contract={
                "condition": DENSE_OBSERVATION_KIND,
                "interpolation_method": interpolation,
                "network_observation_frames": [0, 4],
                "raw_slots_retained": [0, 1, 2],
                "future_slot_omitted_from_network": True,
                "missing_slot_fallback": "corresponding_lifted_lr_source_frame",
            },
            description=(
                "073B dense-observation conditional with periodic-x padding"
                if bool(config["model"].get("periodic_padding_x", False))
                else "073A dense-observation conditional"
            ),
        )

    raise ValueError(f"unsupported conditioning.kind: {kind}")


__all__ = [
    "ConditionalRuntime",
    "DENSE_OBSERVATION_KIND",
    "MASKED_INNOVATION_KIND",
    "conditioning_kind",
    "get_conditional_runtime",
]
