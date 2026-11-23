from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.filo_srda_qg.sb_prior_retrain_qg.ema import EMA
from src.filo_srda_qg.sr_prior.model import make_sb_prior_model


DEFAULT_WEIGHT_ORDER = (
    "best_panel_clean_ema.pt",
    "best_panel_clean_raw.pt",
    "best_panel_crps_ema.pt",
    "best_panel_crps_raw.pt",
    "best_loss_raw.pt",
)


def select_weights(checkpoint_dir: Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if path.is_absolute():
            return path
        # Accept paths relative to either the current working directory or the
        # run checkpoint directory.  Do not infer a repository depth from the
        # checkpoint layout; that made custom work roots resolve incorrectly.
        cwd_candidate = path.resolve()
        return cwd_candidate if cwd_candidate.exists() else checkpoint_dir / path
    for name in DEFAULT_WEIGHT_ORDER:
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No evaluation weights in {checkpoint_dir}")


def load_eval_model(
    *,
    config: dict[str, Any],
    checkpoint_dir: Path,
    device: torch.device,
    explicit: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    path = select_weights(checkpoint_dir, explicit)
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
        metadata = dict(payload.get("metadata", {}))
    else:
        state = payload
        metadata = {}
    model = make_sb_prior_model(config["model"])
    model.load_state_dict(state)
    metadata.update(weights_path=str(path), weights_kind=metadata.get("weights_kind", "raw"))
    return model.to(device).float().eval(), metadata


def load_model_from_training_checkpoint(
    *,
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: torch.device,
    weights_kind: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load raw or EMA weights from a full 071B training checkpoint.

    The normal ``load_eval_model`` path loads standalone weight artifacts such
    as ``best_panel_clean_ema.pt``.  Narrow-jet checkpoint sweeps need to load
    archived full checkpoints, e.g. ``checkpoint_epoch0400.pth``, and select
    either the raw model state or the EMA shadow state explicitly.
    """

    if weights_kind not in {"raw", "ema"}:
        raise ValueError(f"weights_kind must be raw or ema, got {weights_kind!r}")
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise KeyError(f"Full checkpoint lacks model_state_dict: {path}")

    model = make_sb_prior_model(config["model"]).to(device).float().eval()
    model.load_state_dict(payload["model_state_dict"])
    weights_source = "model_state_dict"
    if weights_kind == "ema":
        if "ema_state_dict" not in payload:
            raise KeyError(f"Full checkpoint lacks ema_state_dict: {path}")
        ema = EMA(model, decay=float(config["train"].get("ema_decay", 0.999)))
        ema.load_state_dict(payload["ema_state_dict"])
        ema.copy_to(model)
        weights_source = "ema_state_dict"

    epoch_value = payload.get("epoch")
    # Training checkpoints store zero-based epoch internally.
    checkpoint_epoch = int(epoch_value) + 1 if epoch_value is not None else None
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_epoch": checkpoint_epoch,
        "weights_kind": weights_kind,
        "weights_source": weights_source,
        "run_name": payload.get("resolved_config", {}).get("output", {}).get("run_name"),
        "parameter_count": payload.get("parameter_count"),
        "model_signature": payload.get("model_signature"),
    }
    return model, metadata
