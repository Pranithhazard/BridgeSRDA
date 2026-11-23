from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EnsembleDiagnostics:
    mean: torch.Tensor
    spread: torch.Tensor
    metrics: dict[str, float]
    rank_histogram: torch.Tensor


def fixed_subensemble_indices(
    *, ensemble_size: int, subset_size: int, seed: int = 42
) -> np.ndarray:
    if not 1 < subset_size <= ensemble_size:
        raise ValueError(
            f"subset_size must be in [2, {ensemble_size}], got {subset_size}"
        )
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(ensemble_size, size=subset_size, replace=False).astype(np.int64)
    )


def _linear_quantile_from_sorted(samples: torch.Tensor, q: float) -> torch.Tensor:
    ensemble_size = samples.shape[0]
    position = q * float(ensemble_size - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    weight = position - lower_index
    if lower_index == upper_index:
        return samples[lower_index]
    return samples[lower_index].lerp(samples[upper_index], weight)


@torch.no_grad()
def ensemble_diagnostics(
    samples: torch.Tensor,
    target: torch.Tensor,
) -> EnsembleDiagnostics:
    """Compute physical-space ensemble diagnostics with a single ensemble sort.

    Args:
        samples: ``(M, ...)`` ensemble tensor.
        target: ``(...)`` verifying state.
    """

    if samples.ndim != target.ndim + 1:
        raise ValueError(
            f"Expected samples (M,...) and target (...), got {samples.shape}, {target.shape}"
        )
    if tuple(samples.shape[1:]) != tuple(target.shape):
        raise ValueError(
            f"Sample state shape {samples.shape[1:]} != target shape {target.shape}"
        )
    if samples.shape[0] < 2:
        raise ValueError("At least two ensemble members are required")
    if not torch.isfinite(samples).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite sample or target passed to ensemble diagnostics")

    samples = samples.float()
    target = target.to(device=samples.device, dtype=samples.dtype)
    ensemble_size = samples.shape[0]
    mean = samples.mean(dim=0)
    error = mean - target
    rmse = error.square().mean().sqrt()
    spread = samples.std(dim=0, unbiased=True)
    spread_rms = spread.square().mean().sqrt()

    sorted_samples = samples.sort(dim=0).values
    coefficients = (
        2 * torch.arange(1, ensemble_size + 1, device=samples.device)
        - ensemble_size
        - 1
    ).to(samples.dtype)
    coefficient_shape = (ensemble_size,) + (1,) * target.ndim
    skill_term = (samples - target.unsqueeze(0)).abs().mean(dim=0)
    spread_term = (
        sorted_samples * coefficients.view(coefficient_shape)
    ).sum(dim=0) / float(ensemble_size**2)

    metrics = {
        "physical_mae": error.abs().mean().item(),
        "physical_rmse": rmse.item(),
        "physical_spread_rms": spread_rms.item(),
        "spread_rmse_ratio": (spread_rms / rmse.clamp_min(1.0e-12)).item(),
        "physical_crps": (skill_term - spread_term).mean().item(),
    }
    for level in (0.50, 0.80, 0.90):
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q
        lower = _linear_quantile_from_sorted(sorted_samples, lower_q)
        upper = _linear_quantile_from_sorted(sorted_samples, upper_q)
        name = int(level * 100)
        metrics[f"coverage_{name}"] = (
            ((target >= lower) & (target <= upper)).float().mean().item()
        )
        metrics[f"physical_interval_width_{name}"] = (
            upper - lower
        ).mean().item()

    ranks = (samples < target.unsqueeze(0)).sum(dim=0).reshape(-1)
    rank_histogram = torch.bincount(ranks, minlength=ensemble_size + 1)
    return EnsembleDiagnostics(
        mean=mean,
        spread=spread,
        metrics=metrics,
        rank_histogram=rank_histogram,
    )


def hr_truth_from_uhr(uhr: torch.Tensor, *, nz: int, ny: int, nx: int) -> torch.Tensor:
    """Reproduce the established UHR-to-HR average-pooling truth definition."""

    if uhr.ndim != 3:
        raise ValueError(f"Expected UHR field (Z,Y,X), got {tuple(uhr.shape)}")
    kernel_size = uhr.shape[-1] // nx
    if kernel_size <= 0 or uhr.shape[-1] % nx:
        raise ValueError(f"UHR x-size {uhr.shape[-1]} is incompatible with HR x-size {nx}")
    pooled = F.avg_pool3d(uhr[:, 1:, :].unsqueeze(0), kernel_size=kernel_size)[0]
    expected = (nz, ny - 1, nx)
    if tuple(pooled.shape) != expected:
        raise ValueError(f"Pooled truth shape {tuple(pooled.shape)} != {expected}")
    output = torch.zeros((nz, ny, nx), dtype=pooled.dtype, device=pooled.device)
    output[:, 1:, :] = pooled
    return output
