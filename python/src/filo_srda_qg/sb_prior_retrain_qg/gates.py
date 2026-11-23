from __future__ import annotations

import dataclasses
from typing import Mapping


@dataclasses.dataclass(frozen=True)
class CleanGateResult:
    passed: bool
    checks: dict[str, bool]
    values: dict[str, float]


def _relative(lower: float, upper: float) -> float:
    return abs(upper - lower) / max(abs(lower), 1.0e-30)


def evaluate_clean_gate(
    *,
    nfe20: Mapping[str, float],
    nfe50: Mapping[str, float],
    midpoint_grid_ratio: float,
    thresholds: Mapping[str, float],
) -> CleanGateResult:
    pairs = {
        "mae_change": ("physical_mae", "mae_relative_change"),
        "rmse_change": ("physical_rmse", "rmse_relative_change"),
        "crps_change": ("physical_crps", "crps_relative_change"),
        "spread_change": ("physical_spread_rms", "spread_relative_change"),
        "spread_ratio_change": ("spread_rmse_ratio", "spread_ratio_relative_change"),
        "high_k_change": ("perturbation_high_k_fraction", "high_k_relative_change"),
    }
    values = {
        name: _relative(float(nfe20[key]), float(nfe50[key]))
        for name, (key, _) in pairs.items()
    }
    values.update(
        midpoint_grid_ratio=float(midpoint_grid_ratio),
        coverage_change=abs(float(nfe50["coverage_90"]) - float(nfe20["coverage_90"])),
        high_k=float(nfe50["perturbation_high_k_fraction"]),
        correlation_length=float(nfe50["perturbation_correlation_length"]),
        gradient_ratio=float(nfe50["member_gradient_variance"])
        / max(float(nfe50["target_gradient_variance"]), 1.0e-30),
        laplacian_ratio=float(nfe50["member_laplacian_variance"])
        / max(float(nfe50["target_laplacian_variance"]), 1.0e-30),
        spread_rmse_ratio=float(nfe50["spread_rmse_ratio"]),
    )
    checks = {
        "midpoint_grid": values["midpoint_grid_ratio"] <= thresholds["midpoint_grid_ratio"],
        "mae_convergence": values["mae_change"] <= thresholds["mae_relative_change"],
        "rmse_convergence": values["rmse_change"] <= thresholds["rmse_relative_change"],
        "crps_convergence": values["crps_change"] <= thresholds["crps_relative_change"],
        "spread_convergence": values["spread_change"] <= thresholds["spread_relative_change"],
        "spread_ratio_convergence": values["spread_ratio_change"] <= thresholds["spread_ratio_relative_change"],
        "coverage_convergence": values["coverage_change"] <= thresholds["coverage_absolute_change"],
        "high_k_convergence": values["high_k_change"] <= thresholds["high_k_relative_change"],
        "high_k": values["high_k"] <= thresholds["high_k_max"],
        "correlation_length": values["correlation_length"] >= thresholds["correlation_length_min"],
        "gradient_ratio": values["gradient_ratio"] <= thresholds["gradient_ratio_max"],
        "laplacian_ratio": values["laplacian_ratio"] <= thresholds["laplacian_ratio_max"],
        "spread_ratio": thresholds["spread_ratio_min"]
        <= values["spread_rmse_ratio"]
        <= thresholds["spread_ratio_max"],
    }
    return CleanGateResult(passed=all(checks.values()), checks=checks, values=values)
