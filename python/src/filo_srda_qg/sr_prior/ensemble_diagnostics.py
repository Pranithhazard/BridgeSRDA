from __future__ import annotations

import math

import torch

from src.filo_srda_qg.sr_prior.diagnostics import (
    horizontal_power_spectrum,
)


def laplacian_variance(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-2] < 3 or values.shape[-1] < 3:
        raise ValueError("horizontal dimensions must be at least 3")
    center = values[..., 1:-1, 1:-1]
    laplacian = (
        values[..., 1:-1, 2:]
        + values[..., 1:-1, :-2]
        + values[..., 2:, 1:-1]
        + values[..., :-2, 1:-1]
        - 4.0 * center
    )
    return laplacian.float().var(unbiased=False)


def horizontal_spectrum_mode_counts(
    ny: int, nx: int, *, device: torch.device
) -> torch.Tensor:
    if ny <= 1 or nx <= 1:
        raise ValueError("horizontal dimensions must exceed one")
    ky = torch.fft.fftfreq(ny, device=device) * ny
    kx = torch.fft.rfftfreq(nx, device=device) * nx
    radius = torch.sqrt(ky[:, None].square() + kx[None, :].square()).round().long()
    max_bin = min(ny // 2, nx // 2)
    counts = torch.zeros(max_bin + 1, device=device)
    valid = radius <= max_bin
    multiplicity_x = torch.full_like(kx, 2.0)
    multiplicity_x[0] = 1.0
    if nx % 2 == 0:
        multiplicity_x[-1] = 1.0
    multiplicity = multiplicity_x[None, :].expand(ny, -1)
    counts.scatter_add_(0, radius[valid], multiplicity[valid])
    return counts


def high_wavenumber_fraction(
    spectrum: torch.Tensor, mode_counts: torch.Tensor
) -> torch.Tensor:
    if spectrum.ndim != 1 or spectrum.numel() < 2:
        raise ValueError("spectrum must be a one-dimensional nontrivial tensor")
    if mode_counts.shape != spectrum.shape or torch.any(mode_counts < 0):
        raise ValueError("mode_counts must be nonnegative and match spectrum")
    first = int(math.ceil(0.75 * (spectrum.numel() - 1)))
    shell_energy = spectrum * mode_counts.to(spectrum)
    return shell_energy[first:].sum() / shell_energy.sum().clamp_min(
        torch.finfo(spectrum.dtype).tiny
    )


def horizontal_high_wavenumber_fraction(values: torch.Tensor) -> torch.Tensor:
    """Energy fraction for normalized radial wavenumbers 0.75 <= k/k_N <= 1."""

    if values.ndim < 2:
        raise ValueError("values must have horizontal y/x dimensions")
    ny, nx = values.shape[-2:]
    if ny <= 1 or nx <= 1:
        raise ValueError("horizontal dimensions must exceed one")
    transformed = torch.fft.rfft2(values.float(), dim=(-2, -1), norm="ortho")
    power = transformed.abs().square().reshape(-1, ny, nx // 2 + 1).mean(dim=0)
    ky = torch.fft.fftfreq(ny, device=values.device) * ny
    kx = torch.fft.rfftfreq(nx, device=values.device) * nx
    radius = torch.sqrt(
        (ky[:, None] / float(ny // 2)).square()
        + (kx[None, :] / float(nx // 2)).square()
    )
    multiplicity_x = torch.full_like(kx, 2.0)
    multiplicity_x[0] = 1.0
    if nx % 2 == 0:
        multiplicity_x[-1] = 1.0
    weighted_power = power * multiplicity_x[None, :]
    represented = radius <= 1.0
    high = represented & (radius >= 0.75)
    return weighted_power[high].sum() / weighted_power[represented].sum().clamp_min(
        torch.finfo(weighted_power.dtype).tiny
    )


def perturbation_spectra(
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if samples.ndim < 6 or samples.shape[1] < 2:
        raise ValueError("samples must have shape (B,M,T,Z,Y,X) with M>=2")
    mean = samples.mean(dim=1)
    perturbations = samples - mean[:, None]
    wave, mean_spectrum = horizontal_power_spectrum(mean)
    _, member_spectrum = horizontal_power_spectrum(samples)
    _, perturbation_spectrum = horizontal_power_spectrum(perturbations)
    return wave, mean_spectrum, member_spectrum, perturbation_spectrum


def radial_perturbation_correlation(
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero-padded radial autocorrelation and its first 1/e crossing."""

    if samples.ndim < 6 or samples.shape[1] < 2:
        raise ValueError("samples must have shape (B,M,T,Z,Y,X) with M>=2")
    perturbations = (samples - samples.mean(dim=1, keepdim=True)).float()
    ny, nx = perturbations.shape[-2:]
    flat = perturbations.reshape(-1, ny, nx)
    transformed = torch.fft.rfft2(flat, s=(2 * ny, 2 * nx), norm="backward")
    covariance = torch.fft.irfft2(
        transformed.abs().square(), s=(2 * ny, 2 * nx), norm="backward"
    )[..., :ny, :nx]
    lag_y = torch.arange(ny, device=samples.device)
    lag_x = torch.arange(nx, device=samples.device)
    pair_counts = (ny - lag_y)[:, None] * (nx - lag_x)[None, :]
    covariance = (covariance / pair_counts.clamp_min(1)).mean(dim=0)
    covariance = covariance / covariance[0, 0].clamp_min(torch.finfo(covariance.dtype).tiny)
    radius = torch.sqrt(lag_y[:, None].square() + lag_x[None, :].square())
    radius_bin = radius.round().long()
    max_bin = min(ny - 1, nx - 1)
    radial = torch.zeros(max_bin + 1, device=samples.device)
    counts = torch.zeros_like(radial)
    valid = radius_bin <= max_bin
    radial.scatter_add_(0, radius_bin[valid], covariance[valid])
    counts.scatter_add_(0, radius_bin[valid], torch.ones_like(covariance[valid]))
    radial = radial / counts.clamp_min(1)
    below = torch.nonzero(radial <= math.exp(-1.0), as_tuple=False)
    length = (
        below[0, 0].to(radial.dtype)
        if below.numel()
        else torch.tensor(float(max_bin), device=samples.device)
    )
    return torch.arange(max_bin + 1, device=samples.device), radial, length
