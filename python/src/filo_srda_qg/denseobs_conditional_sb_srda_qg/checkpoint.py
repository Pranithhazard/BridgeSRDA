from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.filo_srda_qg.sb_prior_retrain_qg.ema import EMA

from .model import denseobs_model_signature, make_denseobs_conditional_sb_model


DEFAULT_WEIGHT_ORDER = (
    "best_panel_crps_ema.pt",
    "best_panel_clean_ema.pt",
    "best_panel_crps_raw.pt",
    "best_panel_clean_raw.pt",
    "best_loss_raw.pt",
)


def select_weights(checkpoint_dir: Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if path.is_absolute():
            return path
        resolved = path.resolve()
        return resolved if resolved.exists() else checkpoint_dir / path
    for name in DEFAULT_WEIGHT_ORDER:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No 073A evaluation weights in {checkpoint_dir}")


def load_eval_model(
    *,
    config: dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device,
    explicit: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = select_weights(checkpoint_dir, explicit)
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model_state_dict"] if "model_state_dict" in payload else payload
    metadata = dict(payload.get("metadata", {})) if isinstance(payload, dict) else {}
    model = make_denseobs_conditional_sb_model(config["model"])
    model.load_state_dict(state)
    metadata.update(
        weights_path=str(path), weights_kind=metadata.get("weights_kind", "raw")
    )
    return model.to(device).float().eval(), metadata


def load_training_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
    weights_kind: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if weights_kind not in {"raw", "ema"}:
        raise ValueError("weights_kind must be raw or ema")
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model = make_denseobs_conditional_sb_model(config["model"]).to(device).float().eval()
    if "ema_state_dict" not in payload and "metadata" in payload:
        artifact_kind = str(payload["metadata"].get("weights_kind", "raw"))
        if artifact_kind != weights_kind:
            raise ValueError(
                f"Standalone artifact contains {artifact_kind} weights, not {weights_kind}"
            )
        model.load_state_dict(payload["model_state_dict"])
        return model, {
            "checkpoint_path": str(path),
            "checkpoint_epoch": int(payload["metadata"].get("selected_epoch", 0)),
            "weights_kind": artifact_kind,
            "weights_source": "standalone_model_state_dict",
            "run_name": config["output"]["run_name"],
            "parameter_count": payload["metadata"].get(
                "parameter_count", sum(p.numel() for p in model.parameters())
            ),
            "model_signature": payload["metadata"].get(
                "model_signature",
                denseobs_model_signature(config["model"]),
            ),
            "observation_contract": payload["metadata"].get(
                "observation_contract"
            ),
        }
    model.load_state_dict(payload["model_state_dict"])
    source = "model_state_dict"
    if weights_kind == "ema":
        ema = EMA(model, decay=float(config["train"]["ema_decay"]))
        ema.load_state_dict(payload["ema_state_dict"])
        ema.copy_to(model)
        source = "ema_state_dict"
    return model, {
        "checkpoint_path": str(path),
        "checkpoint_epoch": int(payload["epoch"]) + 1,
        "weights_kind": weights_kind,
        "weights_source": source,
        "run_name": payload.get("resolved_config", {}).get("output", {}).get("run_name"),
        "parameter_count": payload.get("parameter_count"),
        "model_signature": payload.get("model_signature"),
        "observation_contract": payload.get("observation_contract"),
    }


__all__ = ["load_eval_model", "load_training_checkpoint", "select_weights"]
