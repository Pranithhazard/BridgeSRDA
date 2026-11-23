from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.filo_srda_qg.sr_prior.model import (
    SBPriorUNet3D,
    SBPriorUNet3DConfig,
    SBResBlock3D,
)
from src.filo_srda_qg.diffusion_srda_qg.model import _spatial_conv3d


class DenseObsConditionalSBUNet3D(SBPriorUNet3D):
    """The 072 3-D U-Net conditioned on two dense observation fields."""

    def __init__(self, cfg: SBPriorUNet3DConfig):
        if cfg.input_channels != 20 or cfg.out_channels != 9:
            raise ValueError(
                "DenseObsConditionalSBUNet3D requires 20 input and 9 output channels"
            )
        base_cfg = copy.deepcopy(cfg)
        base_cfg.input_channels = 18
        super().__init__(base_cfg)
        self.cfg = copy.deepcopy(cfg)
        self.input_conv = _spatial_conv3d(
            20,
            cfg.base_width,
            kernel_size=3,
            padding=1,
            bias=True,
            periodic_padding_x=cfg.periodic_padding_x,
        )

    def forward(
        self,
        bridge_state: torch.Tensor,
        source: torch.Tensor,
        dense_observations: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        if bridge_state.shape != source.shape or bridge_state.ndim != 5:
            raise ValueError("bridge_state and source must share (B,9,Z,Y,X)")
        expected_dense = (bridge_state.shape[0], 2, *bridge_state.shape[-3:])
        if tuple(dense_observations.shape) != expected_dense:
            raise ValueError(
                f"dense_observations must have shape {expected_dense}, "
                f"got {tuple(dense_observations.shape)}"
            )
        if bridge_state.shape[1] != 9 or tau.shape != (bridge_state.shape[0],):
            raise ValueError("invalid bridge channel or bridge-time shape")
        embedding = self.time_mlp(tau)
        x = self.input_conv(
            torch.cat([bridge_state, source, dense_observations], dim=1)
        )
        skips = [x]
        for layer in self.encoder:
            if isinstance(layer, SBResBlock3D):
                x = self._block(layer, x, embedding)
            else:
                x = layer(x)
            skips.append(x)
        for layer in self.middle:
            x = self._block(layer, x, embedding)
        for layer in self.decoder:
            if isinstance(layer, SBResBlock3D):
                x = self._block(layer, torch.cat([x, skips.pop()], dim=1), embedding)
            else:
                x = layer(x)
        if skips:
            raise RuntimeError("forward skip mismatch")
        return self.output_conv(F.silu(self.output_norm(x)))


def make_denseobs_conditional_sb_model(
    config: dict[str, Any],
) -> DenseObsConditionalSBUNet3D:
    model = DenseObsConditionalSBUNet3D(SBPriorUNet3DConfig.from_dict(config))
    count = sum(parameter.numel() for parameter in model.parameters())
    limit = int(config.get("max_parameter_count", 100_000_000))
    if count >= limit:
        raise ValueError(f"dense-observation SB model exceeds parameter limit: {count:,}")
    return model


def denseobs_model_signature(config: dict[str, Any]) -> str:
    if bool(config.get("periodic_padding_x", False)):
        return (
            "DenseObsConditionalSBUNet3D_PeriodicX:"
            "(9+9+2denseobs)->9[padding_x=circular,padding_yz=zeros]"
        )
    return "DenseObsConditionalSBUNet3D:(9+9+2denseobs)->9"


__all__ = [
    "DenseObsConditionalSBUNet3D",
    "denseobs_model_signature",
    "make_denseobs_conditional_sb_model",
]
