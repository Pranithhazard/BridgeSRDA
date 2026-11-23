from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.filo_srda_qg.diffusion_srda_qg.model import (
    DiffSRDADownsample3D,
    DiffSRDAUpsample3D,
    _spatial_conv3d,
    _group_norm,
)


class SBTimeEmbedding(nn.Module):
    """Sinusoidal bridge-time embedding used by the urban reference model."""

    def __init__(self, dim: int, max_period: float):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        arguments = t[:, None].float() * frequencies[None]
        output = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        return F.pad(output, (0, self.dim - output.shape[1]))


class SBSelfAttention3D(nn.Module):
    """3D form of the urban reference's zero-projection attention block."""

    def __init__(self, channels: int, *, n_head: int, norm_groups: int):
        super().__init__()
        if channels % n_head:
            raise ValueError("attention channels must be divisible by heads")
        self.channels = channels
        self.n_head = n_head
        self.norm = _group_norm(channels, norm_groups)
        self.qkv = nn.Conv3d(channels, 3 * channels, 1, bias=True)
        self.output = nn.Conv3d(channels, channels, 1, bias=True)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, z, y, x_size = x.shape
        head_dim = channels // self.n_head
        tokens = z * y * x_size
        qkv = self.qkv(self.norm(x)).reshape(batch, self.n_head, 3 * head_dim, tokens)
        query, key, value = qkv.chunk(3, dim=2)
        scale = 1.0 / math.sqrt(math.sqrt(head_dim))
        weights = torch.einsum("bhct,bhcs->bhts", query * scale, key * scale)
        weights = torch.softmax(weights.float(), dim=-1).to(value.dtype)
        output = torch.einsum("bhts,bhcs->bhct", weights, value).reshape(
            batch, channels, z, y, x_size
        )
        return x + self.output(output)


class SBResBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        embed_dim: int,
        norm_groups: int,
        dropout: float,
        attention: bool,
        attention_heads: int,
        periodic_padding_x: bool = False,
    ):
        super().__init__()
        self.norm1 = _group_norm(in_channels, norm_groups)
        self.conv1 = _spatial_conv3d(
            in_channels,
            out_channels,
            3,
            padding=1,
            bias=True,
            periodic_padding_x=periodic_padding_x,
        )
        self.embed = nn.Linear(embed_dim, 2 * out_channels, bias=True)
        self.norm2 = _group_norm(out_channels, norm_groups)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2 = _spatial_conv3d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=True,
            periodic_padding_x=periodic_padding_x,
        )
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=True)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.attention = (
            SBSelfAttention3D(
                out_channels, n_head=attention_heads, norm_groups=norm_groups
            )
            if attention
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.embed(embedding).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return self.attention(h + self.skip(x))


@dataclasses.dataclass
class SBPriorUNet3DConfig:
    input_channels: int = 18
    out_channels: int = 9
    base_width: int = 32
    channel_multipliers: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 4, 8, 8]
    )
    num_res_blocks: int = 3
    norm_groups: int = 32
    dropout: float = 0.2
    attention_head_dim: int = 16
    max_period: float = 10.0
    activation_checkpointing: bool = True
    hr_z_size: int = 16
    hr_y_size: int = 64
    hr_x_size: int = 128
    periodic_padding_x: bool = False
    max_parameter_count: int = 100_000_000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SBPriorUNet3DConfig":
        return cls(**data)


class SBPriorUNet3D(nn.Module):
    def __init__(self, cfg: SBPriorUNet3DConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.input_channels != 18 or cfg.out_channels != 9:
            raise ValueError("SBPriorUNet3D requires 18 input and 9 output channels")
        if len(cfg.channel_multipliers) != 5:
            raise ValueError("five multiscale levels are required")
        divisor = 2 ** (len(cfg.channel_multipliers) - 1)
        if any(size % divisor for size in (cfg.hr_z_size, cfg.hr_y_size, cfg.hr_x_size)):
            raise ValueError("HR dimensions must be divisible by 16")
        embed_dim = 4 * cfg.base_width
        self.time_mlp = nn.Sequential(
            SBTimeEmbedding(cfg.base_width, cfg.max_period),
            nn.Linear(cfg.base_width, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.input_conv = _spatial_conv3d(
            18,
            cfg.base_width,
            3,
            padding=1,
            bias=True,
            periodic_padding_x=cfg.periodic_padding_x,
        )
        self.encoder = nn.ModuleList()
        skip_channels = [cfg.base_width]
        channels = cfg.base_width
        deepest = len(cfg.channel_multipliers) - 1
        for level, multiplier in enumerate(cfg.channel_multipliers):
            out_channels = cfg.base_width * multiplier
            heads = max(1, out_channels // cfg.attention_head_dim)
            for _ in range(cfg.num_res_blocks):
                self.encoder.append(
                    SBResBlock3D(
                        channels,
                        out_channels,
                        embed_dim=embed_dim,
                        norm_groups=cfg.norm_groups,
                        dropout=cfg.dropout,
                        attention=level == deepest,
                        attention_heads=heads,
                        periodic_padding_x=cfg.periodic_padding_x,
                    )
                )
                channels = out_channels
                skip_channels.append(channels)
            if level != deepest:
                self.encoder.append(
                    DiffSRDADownsample3D(
                        channels,
                        periodic_padding_x=cfg.periodic_padding_x,
                    )
                )
                skip_channels.append(channels)
        deep_heads = max(1, channels // cfg.attention_head_dim)
        self.middle = nn.ModuleList(
            [
                SBResBlock3D(
                    channels,
                    channels,
                    embed_dim=embed_dim,
                    norm_groups=cfg.norm_groups,
                    dropout=cfg.dropout,
                    attention=True,
                    attention_heads=deep_heads,
                    periodic_padding_x=cfg.periodic_padding_x,
                ),
                SBResBlock3D(
                    channels,
                    channels,
                    embed_dim=embed_dim,
                    norm_groups=cfg.norm_groups,
                    dropout=cfg.dropout,
                    attention=False,
                    attention_heads=deep_heads,
                    periodic_padding_x=cfg.periodic_padding_x,
                ),
            ]
        )
        self.decoder = nn.ModuleList()
        for level in reversed(range(len(cfg.channel_multipliers))):
            out_channels = cfg.base_width * cfg.channel_multipliers[level]
            heads = max(1, out_channels // cfg.attention_head_dim)
            for _ in range(cfg.num_res_blocks + 1):
                self.decoder.append(
                    SBResBlock3D(
                        channels + skip_channels.pop(),
                        out_channels,
                        embed_dim=embed_dim,
                        norm_groups=cfg.norm_groups,
                        dropout=cfg.dropout,
                        attention=level == deepest,
                        attention_heads=heads,
                        periodic_padding_x=cfg.periodic_padding_x,
                    )
                )
                channels = out_channels
            if level > 0:
                self.decoder.append(
                    DiffSRDAUpsample3D(
                        channels,
                        periodic_padding_x=cfg.periodic_padding_x,
                    )
                )
        if skip_channels:
            raise RuntimeError("skip construction mismatch")
        self.output_norm = _group_norm(channels, cfg.norm_groups)
        self.output_conv = _spatial_conv3d(
            channels,
            9,
            3,
            padding=1,
            bias=True,
            periodic_padding_x=cfg.periodic_padding_x,
        )
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def _block(self, block: SBResBlock3D, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(block, x, emb, use_reentrant=False)
        return block(x, emb)

    def forward(
        self, bridge_state: torch.Tensor, source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        if bridge_state.shape != source.shape or bridge_state.ndim != 5:
            raise ValueError("bridge_state and source must share (B,9,Z,Y,X)")
        if bridge_state.shape[1] != 9 or t.shape != (bridge_state.shape[0],):
            raise ValueError("invalid channel or time shape")
        emb = self.time_mlp(t)
        x = self.input_conv(torch.cat([bridge_state, source], dim=1))
        skips = [x]
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


def make_sb_prior_model(config: dict[str, Any]) -> SBPriorUNet3D:
    model = SBPriorUNet3D(SBPriorUNet3DConfig.from_dict(config))
    count = sum(parameter.numel() for parameter in model.parameters())
    if count >= int(config.get("max_parameter_count", 100_000_000)):
        raise ValueError(f"SB prior model exceeds parameter limit: {count:,}")
    return model
