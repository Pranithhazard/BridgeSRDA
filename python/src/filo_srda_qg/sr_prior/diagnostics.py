from __future__ import annotations

import torch


def horizontal_gradient_variance(values: torch.Tensor) -> torch.Tensor:
    dx = torch.diff(values, dim=-1)
    dy = torch.diff(values, dim=-2)
    return 0.5 * (dx.var(unbiased=False) + dy.var(unbiased=False))


def temporal_increment_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted_increment = torch.diff(prediction, dim=1)
    target_increment = torch.diff(target, dim=1)
    return (predicted_increment - target_increment).square().mean().sqrt()


def horizontal_power_spectrum(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotropic index-wavenumber spectrum averaged over leading dimensions."""

    if values.ndim < 2:
        raise ValueError("values must have horizontal y/x dimensions")
    ny, nx = values.shape[-2:]
    transformed = torch.fft.rfft2(values.float(), dim=(-2, -1), norm="ortho")
    power = transformed.abs().square().reshape(-1, ny, nx // 2 + 1).mean(0)
    ky = torch.fft.fftfreq(ny, device=values.device) * ny
    kx = torch.fft.rfftfreq(nx, device=values.device) * nx
    radius = torch.sqrt(ky[:, None].square() + kx[None].square()).round().long()
    max_bin = min(ny // 2, nx // 2)
    spectrum = torch.zeros(max_bin + 1, device=values.device)
    counts = torch.zeros_like(spectrum)
    valid = radius <= max_bin
    spectrum.scatter_add_(0, radius[valid], power[valid])
    counts.scatter_add_(0, radius[valid], torch.ones_like(power[valid]))
    return torch.arange(max_bin + 1, device=values.device), spectrum / counts.clamp_min(1)
