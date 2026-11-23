from __future__ import annotations

import torch


def ensemble_crps(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean empirical CRPS without constructing the MxM pairwise tensor."""

    if samples.ndim != target.ndim + 1:
        raise ValueError("samples must have one ensemble dimension after batch")
    if samples.shape[0] != target.shape[0] or samples.shape[2:] != target.shape[1:]:
        raise ValueError("sample and target shapes are incompatible")
    ensemble_size = samples.shape[1]
    term_skill = (samples - target[:, None]).abs().mean(dim=1)
    sorted_samples = samples.sort(dim=1).values
    coefficients = (
        2 * torch.arange(1, ensemble_size + 1, device=samples.device)
        - ensemble_size
        - 1
    ).to(samples.dtype)
    coefficient_shape = (1, ensemble_size) + (1,) * (samples.ndim - 2)
    term_spread = (
        sorted_samples * coefficients.view(coefficient_shape)
    ).sum(dim=1) / float(ensemble_size**2)
    return (term_skill - term_spread).mean()


def rank_histogram_counts(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rank target before ties, producing M+1 deterministic histogram bins."""

    ranks = (samples < target[:, None]).sum(dim=1).reshape(-1)
    return torch.bincount(ranks, minlength=samples.shape[1] + 1)


def ensemble_summary(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    pv_min: float,
    pv_max: float,
) -> dict[str, float]:
    scale = float(pv_max - pv_min)
    mean = samples.mean(dim=1)
    error = mean - target
    rmse = torch.sqrt(error.pow(2).mean())
    spread_field = samples.std(dim=1, unbiased=True)
    spread_rms = torch.sqrt(spread_field.pow(2).mean())
    output = {
        "normalized_mae": error.abs().mean().item(),
        "normalized_rmse": rmse.item(),
        "physical_mae": (error.abs().mean() * scale).item(),
        "physical_rmse": (rmse * scale).item(),
        "normalized_spread_rms": spread_rms.item(),
        "physical_spread_rms": (spread_rms * scale).item(),
        "spread_rmse_ratio": (spread_rms / rmse.clamp_min(1.0e-12)).item(),
        "normalized_crps": ensemble_crps(samples, target).item(),
    }
    output["physical_crps"] = output["normalized_crps"] * scale
    for level in (0.50, 0.80, 0.90):
        lower_q = (1.0 - level) / 2.0
        upper_q = 1.0 - lower_q
        lower = samples.quantile(lower_q, dim=1)
        upper = samples.quantile(upper_q, dim=1)
        name = int(level * 100)
        output[f"coverage_{name}"] = (
            ((target >= lower) & (target <= upper)).float().mean().item()
        )
        output[f"normalized_interval_width_{name}"] = (
            (upper - lower).mean().item()
        )
        output[f"physical_interval_width_{name}"] = (
            output[f"normalized_interval_width_{name}"] * scale
        )
    return output

