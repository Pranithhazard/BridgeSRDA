from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.filo_srda_qg.shared_srda.source import spatial_lift_lr
from src.four_dim_srda.models.neural_nets.block.voxel_shuffle import VoxelShuffle
from src.four_dim_srda.models.neural_nets.u_net_maxvit_v1 import MaxViTBlocks


def _group_norm(channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = min(channels, requested_groups)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = t.float()[:, None] * frequencies[None]
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2:
            embedding = nn.functional.pad(embedding, (0, 1))
        return embedding


class TimeFiLMResBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        time_embed_dim: int,
        norm_groups: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.norm1 = _group_norm(in_channels, norm_groups)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=bias
        )
        self.time_film = nn.Linear(time_embed_dim, 2 * out_channels)
        self.norm2 = _group_norm(out_channels, norm_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=bias
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(nn.functional.silu(self.norm1(x)))
        scale, shift = self.time_film(time_embedding).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None, None])
        h = h + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(nn.functional.silu(h)))
        return (h + residual) / math.sqrt(2.0)


class SiLUVoxelShuffleBlock(nn.Module):
    def __init__(self, channels: int, *, upscale_factor: int, bias: bool):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels * upscale_factor**3,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.shuffle = VoxelShuffle(factor=upscale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(nn.functional.silu(self.conv(x)))


@dataclasses.dataclass
class DirectSIUNetMaxViTConfig:
    model_name: str = "DirectSIUNetMaxViT"
    input_channels: int = 15
    out_channels: int = 9
    width: int = 128
    num_res_blocks: int = 3
    num_maxvit_blocks: int = 6
    norm_groups: int = 32
    time_embed_dim: int = 256
    residual_dropout: float = 0.0
    maxvit_dropout: float = 0.1
    bias: bool = False
    n_head: int = 32
    expansion_rate: int = 4
    shrink_rate: float = 0.25
    grid_window_size_h: int = 2
    grid_window_size_w: int = 4
    use_downsample: bool = False
    activation_checkpointing: bool = False
    hr_z_size: int = 16
    hr_y_size: int = 64
    hr_x_size: int = 128

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectSIUNetMaxViTConfig":
        return cls(**data)


class DirectSIUNetMaxViT(nn.Module):
    def __init__(self, cfg: DirectSIUNetMaxViTConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.input_channels != 15 or cfg.out_channels != 9:
            raise ValueError("DirectSI MVP requires 15 input and 9 output channels")
        if cfg.hr_z_size % 4 or cfg.hr_y_size % 4 or cfg.hr_x_size % 4:
            raise ValueError("HR dimensions must be divisible by four")

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cfg.time_embed_dim),
            nn.Linear(cfg.time_embed_dim, 4 * cfg.time_embed_dim),
            nn.SiLU(),
            nn.Linear(4 * cfg.time_embed_dim, cfg.time_embed_dim),
        )
        self.input_proj = nn.Conv3d(
            cfg.input_channels, cfg.width, kernel_size=3, padding=1, bias=cfg.bias
        )
        self.full_block = self._block(cfg.width, cfg.width)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = nn.ModuleList(
            [self._block(cfg.width, cfg.width) for _ in range(cfg.num_res_blocks)]
        )
        self.enc2 = nn.ModuleList(
            [self._block(cfg.width, cfg.width) for _ in range(cfg.num_res_blocks)]
        )
        bottleneck_z = cfg.hr_z_size // 4
        self.maxvit = MaxViTBlocks(
            emb_dim=cfg.width * bottleneck_z,
            expansion_rate=cfg.expansion_rate,
            shrink_rate=cfg.shrink_rate,
            bias=cfg.bias,
            n_head=cfg.n_head,
            grid_window_size_h=cfg.grid_window_size_h,
            grid_window_size_w=cfg.grid_window_size_w,
            use_downsample=cfg.use_downsample,
            dropout=cfg.maxvit_dropout,
            num_block=cfg.num_maxvit_blocks,
        )
        self.up1 = SiLUVoxelShuffleBlock(
            cfg.width, bias=cfg.bias, upscale_factor=2
        )
        self.dec1 = nn.ModuleList(
            [
                self._block(
                    2 * cfg.width if index == 0 else cfg.width,
                    cfg.width,
                )
                for index in range(cfg.num_res_blocks)
            ]
        )
        self.up2 = SiLUVoxelShuffleBlock(
            cfg.width, bias=cfg.bias, upscale_factor=2
        )
        self.dec2 = nn.ModuleList(
            [
                self._block(
                    2 * cfg.width if index == 0 else cfg.width,
                    cfg.width,
                )
                for index in range(cfg.num_res_blocks)
            ]
        )
        self.output_norm = _group_norm(cfg.width, cfg.norm_groups)
        self.output = nn.Conv3d(
            cfg.width, cfg.out_channels, kernel_size=3, padding=1, bias=True
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _block(self, in_channels: int, out_channels: int) -> TimeFiLMResBlock3D:
        return TimeFiLMResBlock3D(
            in_channels,
            out_channels,
            time_embed_dim=self.cfg.time_embed_dim,
            norm_groups=self.cfg.norm_groups,
            dropout=self.cfg.residual_dropout,
            bias=self.cfg.bias,
        )

    def _run_block(
        self,
        block: TimeFiLMResBlock3D,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(
                block,
                x,
                time_embedding,
                use_reentrant=False,
            )
        return block(x, time_embedding)

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        for block in blocks:
            x = self._run_block(block, x, time_embedding)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        condition_lr: torch.Tensor,
        obs: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if x_t.ndim != 5 or x_t.shape[1] != 9:
            raise ValueError(f"Expected x_t (B,9,Z,Y,X), got {tuple(x_t.shape)}")
        if condition_lr.ndim != 6 or tuple(condition_lr.shape[1:3]) != (3, 1):
            raise ValueError(
                f"Expected condition_lr (B,3,1,Z,Y,X), got {tuple(condition_lr.shape)}"
            )
        if obs.ndim != 6 or tuple(obs.shape[1:3]) != (3, 1):
            raise ValueError(f"Expected obs (B,3,1,Z,Y,X), got {tuple(obs.shape)}")
        condition_hr = spatial_lift_lr(
            condition_lr, hr_size=tuple(x_t.shape[-3:])
        )[:, :, 0]
        obs_folded = obs[:, :, 0]
        x = torch.cat([x_t, condition_hr, obs_folded], dim=1)
        if x.shape[1] != self.cfg.input_channels:
            raise RuntimeError(f"Combined input has {x.shape[1]} channels")
        time_embedding = self.time_mlp(t)

        x = self.input_proj(x)
        full = self._run_block(self.full_block, x, time_embedding)
        enc1 = self._run_blocks(self.enc1, self.pool(full), time_embedding)
        bottleneck = self._run_blocks(
            self.enc2, self.pool(enc1), time_embedding
        )

        b, c, z, y, x_size = bottleneck.shape
        bottleneck_2d = bottleneck.reshape(b, c * z, y, x_size)
        if self.cfg.activation_checkpointing and self.training:
            bottleneck_2d = checkpoint(
                self.maxvit, bottleneck_2d, use_reentrant=False
            )
        else:
            bottleneck_2d = self.maxvit(bottleneck_2d)
        bottleneck = bottleneck_2d.reshape(b, c, z, y, x_size)

        decoded = self.up1(bottleneck)
        decoded = self._run_blocks(
            self.dec1, torch.cat([decoded, enc1], dim=1), time_embedding
        )
        decoded = self.up2(decoded)
        decoded = self._run_blocks(
            self.dec2, torch.cat([decoded, full], dim=1), time_embedding
        )
        return self.output(nn.functional.silu(self.output_norm(decoded)))
