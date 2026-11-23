from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from torch.utils.checkpoint import checkpoint

from src.filo_srda_qg.shared_srda.model import (
    DirectSIUNetMaxViT,
    DirectSIUNetMaxViTConfig,
)
from src.filo_srda_qg.shared_srda.source import spatial_lift_lr


@dataclasses.dataclass
class DiffusionUNetMaxViTConfig:
    model_name: str = "DiffusionUNetMaxViT"
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
    output_init_std: float = 0.02

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiffusionUNetMaxViTConfig":
        return cls(**data)


class DiffusionUNetMaxViT(DirectSIUNetMaxViT):
    """Notsu-capacity 3D denoiser with DiffSRDA noise-level conditioning."""

    def __init__(self, cfg: DiffusionUNetMaxViTConfig):
        direct_cfg = DirectSIUNetMaxViTConfig(
            **{
                field.name: getattr(cfg, field.name)
                for field in dataclasses.fields(DirectSIUNetMaxViTConfig)
                if field.name != "model_name"
            }
        )
        super().__init__(direct_cfg)
        self.diffusion_cfg = cfg
        nn.init.normal_(self.output.weight, mean=0.0, std=cfg.output_init_std)
        if self.output.bias is not None:
            nn.init.zeros_(self.output.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        condition_lr: torch.Tensor,
        obs: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(x_noisy, condition_lr, obs, noise_level)


def _group_norm(channels: int, requested_groups: int) -> nn.GroupNorm:
    groups = min(channels, requested_groups)
    while groups > 1 and channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class PeriodicXConv3d(nn.Conv3d):
    """Conv3d with circular padding only along the periodic x dimension.

    PyTorch orders 3-D convolution padding as (z, y, x), whereas ``F.pad``
    receives the reverse flattened order.  We therefore apply x padding
    explicitly with ``mode="circular"`` and retain ordinary zero padding in
    z and y inside ``nn.Conv3d``.  Parameters and state-dict keys are exactly
    those of ``nn.Conv3d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if padding_mode != "zeros":
            raise ValueError(
                "PeriodicXConv3d manages x padding explicitly and requires "
                "padding_mode='zeros' for the unchanged y/z dimensions"
            )
        padding_z, padding_y, padding_x = _triple(padding)
        self.periodic_padding_x = int(padding_x)
        self.requested_padding = (
            int(padding_z),
            int(padding_y),
            int(padding_x),
        )
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(padding_z, padding_y, 0),
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.periodic_padding_x:
            input = F.pad(
                input,
                (self.periodic_padding_x, self.periodic_padding_x, 0, 0, 0, 0),
                mode="circular",
            )
        return super().forward(input)


def _spatial_conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    *,
    periodic_padding_x: bool,
    **kwargs: Any,
) -> nn.Conv3d:
    conv_type = PeriodicXConv3d if periodic_padding_x else nn.Conv3d
    return conv_type(in_channels, out_channels, kernel_size, **kwargs)


class DiffSRDANoiseLevelEmbedding(nn.Module):
    """Exact WaveGrad-style noise-level embedding used by DiffSRDA."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level: torch.Tensor) -> torch.Tensor:
        count = self.dim // 2
        step = (
            torch.arange(
                count,
                dtype=noise_level.dtype,
                device=noise_level.device,
            )
            / count
        )
        encoding = noise_level[:, None] * torch.exp(
            -math.log(1.0e4) * step[None]
        )
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1
        )
        if encoding.shape[1] < self.dim:
            encoding = F.pad(encoding, (0, self.dim - encoding.shape[1]))
        return encoding


class DiffSRDAResBlock3D(nn.Module):
    """3D port of DiffSRDA's additive noise-conditioned residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        noise_embed_dim: int,
        norm_groups: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = _group_norm(in_channels, norm_groups)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.noise_projection = nn.Linear(noise_embed_dim, out_channels)
        self.norm2 = _group_norm(out_channels, norm_groups)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.residual = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, noise_embedding: torch.Tensor
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.noise_projection(noise_embedding)[
            :, :, None, None, None
        ]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.residual(x)


class DiffSRDASelfAttention3D(nn.Module):
    """Full voxel attention, used only at the 1x4x8 deepest resolution."""

    def __init__(
        self, channels: int, *, n_head: int, norm_groups: int
    ):
        super().__init__()
        if channels % n_head:
            raise ValueError(
                f"Attention channels {channels} must divide n_head {n_head}"
            )
        self.channels = channels
        self.n_head = n_head
        self.norm = _group_norm(channels, norm_groups)
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1, bias=False)
        self.output = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, z, y, x_size = x.shape
        head_dim = channels // self.n_head
        tokens = z * y * x_size
        qkv = self.qkv(self.norm(x)).reshape(
            batch, self.n_head, 3 * head_dim, tokens
        )
        query, key, value = qkv.chunk(3, dim=2)
        attention = torch.softmax(
            torch.einsum("bhct,bhcs->bhts", query, key)
            / math.sqrt(channels),
            dim=-1,
        )
        output = torch.einsum(
            "bhts,bhcs->bhct", attention, value
        ).reshape(batch, channels, z, y, x_size)
        return self.output(output) + x


class DiffSRDAResBlockWithAttention3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        noise_embed_dim: int,
        norm_groups: int,
        dropout: float,
        with_attention: bool,
        attention_heads: int,
    ):
        super().__init__()
        self.residual = DiffSRDAResBlock3D(
            in_channels,
            out_channels,
            noise_embed_dim=noise_embed_dim,
            norm_groups=norm_groups,
            dropout=dropout,
        )
        self.attention = (
            DiffSRDASelfAttention3D(
                out_channels,
                n_head=attention_heads,
                norm_groups=norm_groups,
            )
            if with_attention
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, noise_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.attention(self.residual(x, noise_embedding))


class DiffSRDADownsample3D(nn.Module):
    def __init__(self, channels: int, *, periodic_padding_x: bool = False):
        super().__init__()
        self.conv = _spatial_conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            periodic_padding_x=periodic_padding_x,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DiffSRDAUpsample3D(nn.Module):
    def __init__(self, channels: int, *, periodic_padding_x: bool = False):
        super().__init__()
        self.conv = _spatial_conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            periodic_padding_x=periodic_padding_x,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


@dataclasses.dataclass
class DiffSRDAUNet3DConfig:
    model_name: str = "DiffSRDAUNet3D"
    input_channels: int = 15
    out_channels: int = 9
    base_width: int = 32
    channel_multipliers: list[int] = dataclasses.field(
        default_factory=lambda: [1, 2, 4, 8, 8]
    )
    num_res_blocks: int = 3
    norm_groups: int = 32
    dropout: float = 0.5
    attention_heads: int = 1
    activation_checkpointing: bool = True
    hr_z_size: int = 16
    hr_y_size: int = 64
    hr_x_size: int = 128
    max_parameter_count: int = 100_000_000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DiffSRDAUNet3DConfig":
        return cls(**data)


class DiffSRDAUNet3D(nn.Module):
    """Faithful five-level 3D port of the successful DiffSRDA U-Net."""

    def __init__(self, cfg: DiffSRDAUNet3DConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.input_channels != 15 or cfg.out_channels != 9:
            raise ValueError("DiffSRDAUNet3D requires 15 input and 9 output channels")
        if len(cfg.channel_multipliers) != 5:
            raise ValueError("DiffSRDAUNet3D requires five channel multipliers")
        divisor = 2 ** (len(cfg.channel_multipliers) - 1)
        for size in (cfg.hr_z_size, cfg.hr_y_size, cfg.hr_x_size):
            if size % divisor:
                raise ValueError(
                    f"HR dimensions must be divisible by {divisor}"
                )

        noise_dim = cfg.base_width
        self.noise_mlp = nn.Sequential(
            DiffSRDANoiseLevelEmbedding(noise_dim),
            nn.Linear(noise_dim, 4 * noise_dim),
            nn.SiLU(),
            nn.Linear(4 * noise_dim, noise_dim),
        )
        self.input_conv = nn.Conv3d(
            cfg.input_channels, cfg.base_width, kernel_size=3, padding=1
        )

        self.encoder = nn.ModuleList()
        skip_channels = [cfg.base_width]
        current_channels = cfg.base_width
        deepest_level = len(cfg.channel_multipliers) - 1
        for level, multiplier in enumerate(cfg.channel_multipliers):
            out_channels = cfg.base_width * multiplier
            for _ in range(cfg.num_res_blocks):
                self.encoder.append(
                    DiffSRDAResBlockWithAttention3D(
                        current_channels,
                        out_channels,
                        noise_embed_dim=noise_dim,
                        norm_groups=cfg.norm_groups,
                        dropout=cfg.dropout,
                        with_attention=level == deepest_level,
                        attention_heads=cfg.attention_heads,
                    )
                )
                current_channels = out_channels
                skip_channels.append(current_channels)
            if level != deepest_level:
                self.encoder.append(DiffSRDADownsample3D(current_channels))
                skip_channels.append(current_channels)

        self.middle = nn.ModuleList(
            [
                DiffSRDAResBlockWithAttention3D(
                    current_channels,
                    current_channels,
                    noise_embed_dim=noise_dim,
                    norm_groups=cfg.norm_groups,
                    dropout=cfg.dropout,
                    with_attention=True,
                    attention_heads=cfg.attention_heads,
                ),
                DiffSRDAResBlockWithAttention3D(
                    current_channels,
                    current_channels,
                    noise_embed_dim=noise_dim,
                    norm_groups=cfg.norm_groups,
                    dropout=cfg.dropout,
                    with_attention=False,
                    attention_heads=cfg.attention_heads,
                ),
            ]
        )

        self.decoder = nn.ModuleList()
        for level in reversed(range(len(cfg.channel_multipliers))):
            out_channels = cfg.base_width * cfg.channel_multipliers[level]
            for _ in range(cfg.num_res_blocks + 1):
                skip_channels_at_level = skip_channels.pop()
                self.decoder.append(
                    DiffSRDAResBlockWithAttention3D(
                        current_channels + skip_channels_at_level,
                        out_channels,
                        noise_embed_dim=noise_dim,
                        norm_groups=cfg.norm_groups,
                        dropout=cfg.dropout,
                        with_attention=level == deepest_level,
                        attention_heads=cfg.attention_heads,
                    )
                )
                current_channels = out_channels
            if level > 0:
                self.decoder.append(DiffSRDAUpsample3D(current_channels))
        if skip_channels:
            raise RuntimeError("DiffSRDAUNet3D skip-channel construction mismatch")

        self.output_norm = _group_norm(current_channels, cfg.norm_groups)
        self.output_conv = nn.Conv3d(
            current_channels, cfg.out_channels, kernel_size=3, padding=1
        )
        self.apply(self._orthogonal_initialize)

    @staticmethod
    def _orthogonal_initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _run_conditioned(
        self,
        block: DiffSRDAResBlockWithAttention3D,
        x: torch.Tensor,
        noise_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.activation_checkpointing and self.training:
            return checkpoint(
                block, x, noise_embedding, use_reentrant=False
            )
        return block(x, noise_embedding)

    def forward(
        self,
        x_noisy: torch.Tensor,
        condition_lr: torch.Tensor,
        obs: torch.Tensor,
        noise_level: torch.Tensor,
    ) -> torch.Tensor:
        if x_noisy.ndim != 5 or x_noisy.shape[1] != 9:
            raise ValueError(
                f"Expected x_noisy (B,9,Z,Y,X), got {tuple(x_noisy.shape)}"
            )
        condition_hr = spatial_lift_lr(
            condition_lr, hr_size=tuple(x_noisy.shape[-3:])
        )[:, :, 0]
        if obs.ndim != 6 or tuple(obs.shape[1:3]) != (3, 1):
            raise ValueError(
                f"Expected obs (B,3,1,Z,Y,X), got {tuple(obs.shape)}"
            )
        # Match DiffSRDA exactly: all conditions precede the noisy target.
        x = torch.cat([condition_hr, obs[:, :, 0], x_noisy], dim=1)
        if x.shape[1] != self.cfg.input_channels:
            raise RuntimeError(f"Combined input has {x.shape[1]} channels")
        noise_embedding = self.noise_mlp(noise_level)
        x = self.input_conv(x)
        skips = [x]
        for layer in self.encoder:
            if isinstance(layer, DiffSRDAResBlockWithAttention3D):
                x = self._run_conditioned(layer, x, noise_embedding)
            else:
                x = layer(x)
            skips.append(x)
        for layer in self.middle:
            x = self._run_conditioned(layer, x, noise_embedding)
        for layer in self.decoder:
            if isinstance(layer, DiffSRDAResBlockWithAttention3D):
                x = torch.cat([x, skips.pop()], dim=1)
                x = self._run_conditioned(layer, x, noise_embedding)
            else:
                x = layer(x)
        if skips:
            raise RuntimeError("DiffSRDAUNet3D forward skip mismatch")
        return self.output_conv(F.silu(self.output_norm(x)))


def make_diffusion_model(
    config: dict[str, Any],
) -> DiffusionUNetMaxViT | DiffSRDAUNet3D:
    model_name = str(config.get("model_name", "DiffusionUNetMaxViT"))
    if model_name == "DiffusionUNetMaxViT":
        cfg = DiffusionUNetMaxViTConfig.from_dict(config)
        model: DiffusionUNetMaxViT | DiffSRDAUNet3D = DiffusionUNetMaxViT(cfg)
        parameter_limit = 100_000_000
    elif model_name == "DiffSRDAUNet3D":
        cfg3d = DiffSRDAUNet3DConfig.from_dict(config)
        model = DiffSRDAUNet3D(cfg3d)
        parameter_limit = cfg3d.max_parameter_count
    else:
        raise ValueError(f"Unsupported diffusion model: {model_name}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= parameter_limit:
        raise ValueError(
            "Production diffusion model exceeds parameter limit "
            f"{parameter_limit:,}: {parameter_count:,}"
        )
    return model
