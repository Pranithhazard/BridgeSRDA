from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state)

