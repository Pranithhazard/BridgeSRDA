from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.filo_srda_qg.sr_prior.model import (
    SBPriorUNet3D,
    SBPriorUNet3DConfig,
)

from .conditioning import build_observation_condition


class ConditionalSBUNet3D(SBPriorUNet3D):
    """The 071B U-Net with six additive observation-conditioning channels."""

    def __init__(self, cfg: SBPriorUNet3DConfig):
        if cfg.input_channels != 24 or cfg.out_channels != 9:
            raise ValueError("ConditionalSBUNet3D requires 24 input and 9 output channels")
        base_cfg = copy.deepcopy(cfg)
        base_cfg.input_channels = 18
        super().__init__(base_cfg)
        self.cfg = copy.deepcopy(cfg)
        self.input_conv = nn.Conv3d(
            24, cfg.base_width, kernel_size=3, padding=1, bias=True
        )

    def forward(
        self,
        bridge_state: torch.Tensor,
        source: torch.Tensor,
        observations: torch.Tensor,
        masks: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if bridge_state.shape != source.shape or bridge_state.ndim != 5:
            raise ValueError("bridge_state and source must share (B,9,Z,Y,X)")
        if bridge_state.shape[1] != 9 or t.shape != (bridge_state.shape[0],):
            raise ValueError("invalid channel or time shape")
        observation_condition = build_observation_condition(source, observations, masks)
        emb = self.time_mlp(t)
        x = self.input_conv(
            torch.cat([bridge_state, source, observation_condition], dim=1)
        )
        skips = [x]
        from src.filo_srda_qg.sr_prior.model import SBResBlock3D

        for layer in self.encoder:
            if isinstance(layer, SBResBlock3D):
                x = self._block(layer, x, emb)
            else:
                x = layer(x)
            skips.append(x)
        for layer in self.middle:
            x = self._block(layer, x, emb)
        for layer in self.decoder:
            if isinstance(layer, SBResBlock3D):
                x = self._block(layer, torch.cat([x, skips.pop()], dim=1), emb)
            else:
                x = layer(x)
        if skips:
            raise RuntimeError("forward skip mismatch")
        return self.output_conv(F.silu(self.output_norm(x)))


def make_conditional_sb_model(config: dict[str, Any]) -> ConditionalSBUNet3D:
    model = ConditionalSBUNet3D(SBPriorUNet3DConfig.from_dict(config))
    count = sum(parameter.numel() for parameter in model.parameters())
    if count >= int(config.get("max_parameter_count", 100_000_000)):
        raise ValueError(f"conditional SB model exceeds parameter limit: {count:,}")
    return model
