from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from src.four_dim_srda.config.config_loader import load_config
from src.four_dim_srda.models.model_maker import make_model


class NotsuAnchorWrapper(nn.Module):
    """Frozen wrapper around reproduced Notsu UNetMaxVitVer01."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        weights_path: str | Path,
        model_name: str = "UNetMaxVitVer01",
        map_location: str | torch.device = "cpu",
    ):
        super().__init__()
        self.config_path = str(config_path)
        self.weights_path = str(weights_path)
        self.model_name = model_name
        cfg = load_config(model_name=model_name, config_path=self.config_path)
        self.model = make_model(cfg)
        state: Any = torch.load(self.weights_path, map_location=map_location)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, lr: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(lr, obs)

