from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.filo_srda_qg.window_data import parse_hr_filename
from src.filo_srda_qg.sr_prior.diagnostics import (
    horizontal_gradient_variance,
    temporal_increment_rmse,
)
from src.filo_srda_qg.sr_prior.metrics import ensemble_summary
from src.filo_srda_qg.sr_prior.source import (
    build_bridge_source,
    fold_hr_target,
)
from src.filo_srda_qg.sr_prior.ensemble_diagnostics import (
    horizontal_high_wavenumber_fraction,
    laplacian_variance,
    perturbation_spectra,
    radial_perturbation_correlation,
)

from .bridge import ContinuousQuadraticBridge
from .sampler import sample_bare_heun_exact_noise


def build_validation_manifest(
    dataset: Any, starts: Iterable[int], output_path: Path | None = None
) -> list[dict[str, Any]]:
    parsed = [parse_hr_filename(path) for path in dataset.all_hr_paths]
    manifest: list[dict[str, Any]] = []
    for start in starts:
        candidates = sorted(
            (
                (index, item)
                for index, item in enumerate(parsed)
                if int(item["start"]) == int(start)
            ),
            key=lambda pair: (
                int(pair[1]["seed"]), int(pair[1]["member"]), str(pair[1]["hr_path"])
            ),
        )
        if not candidates:
            raise RuntimeError(f"No validation window has exact start={start}")
        index, metadata = candidates[0]
        manifest.append(
            {
                "dataset_index": index,
                "seed": int(metadata["seed"]),
                "member": int(metadata["member"]),
                "start": int(metadata["start"]),
                "end": int(metadata["end"]),
                "hr_path": str(metadata["hr_path"]),
            }
        )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def _sample_from_manifest(dataset: Any, entry: dict[str, Any]) -> dict[str, Any]:
    original = dataset.active_indices
    try:
        dataset.active_indices = [int(entry["dataset_index"])]
        return dataset[0]
    finally:
        dataset.active_indices = original


def panel_metrics(
    samples: torch.Tensor,
    target: torch.Tensor,
    *,
    pv_min: float,
    pv_max: float,
) -> dict[str, float]:
    values = ensemble_summary(samples, target, pv_min=pv_min, pv_max=pv_max)
    mean = samples.mean(dim=1)
    perturbations = samples - mean[:, None]
    # Physical horizontal Laplacian on the model grid: periodic x and
    # nearest-edge extension in nonperiodic y. The model tensor omits the
    # restored y=0 analysis boundary, but retains the physical HR spacing.
    def horizontal_laplacian(field: torch.Tensor) -> torch.Tensor:
        dx = 2.0 * torch.pi / float(field.shape[-1])
        dy = torch.pi / float(field.shape[-2])
        left = torch.roll(field, shifts=1, dims=-1)
        right = torch.roll(field, shifts=-1, dims=-1)
        below = torch.cat((field[..., :1, :], field[..., :-1, :]), dim=-2)
        above = torch.cat((field[..., 1:, :], field[..., -1:, :]), dim=-2)
        d2x = (left - 2.0 * field + right) / (dx * dx)
        d2y = (below - 2.0 * field + above) / (dy * dy)
        return d2x + d2y

    laplacian_error = horizontal_laplacian(mean) - horizontal_laplacian(target)
    normalized_laplacian_rmse = laplacian_error.float().square().mean().sqrt()
    spread_field = samples.float().std(dim=1, unbiased=samples.shape[1] > 1)
    spread_q95 = torch.quantile(spread_field, 0.95)
    spread_q999 = torch.quantile(spread_field, 0.999)
    _, _, _, perturbation_spectrum = perturbation_spectra(samples)
    lag, correlation, correlation_length = radial_perturbation_correlation(samples)
    del lag, correlation
    values.update(
        perturbation_high_k_fraction=float(
            horizontal_high_wavenumber_fraction(perturbations).item()
        ),
        perturbation_correlation_length=float(correlation_length.item()),
        member_gradient_variance=float(horizontal_gradient_variance(samples).item()),
        perturbation_gradient_variance=float(
            horizontal_gradient_variance(perturbations).item()
        ),
        target_gradient_variance=float(horizontal_gradient_variance(target).item()),
        member_laplacian_variance=float(laplacian_variance(samples).item()),
        perturbation_laplacian_variance=float(laplacian_variance(perturbations).item()),
        target_laplacian_variance=float(laplacian_variance(target).item()),
        temporal_increment_rmse=float(temporal_increment_rmse(mean, target).item()),
        perturbation_spectrum_energy=float(perturbation_spectrum.sum().item()),
        normalized_laplacian_error_rmse=float(normalized_laplacian_rmse.item()),
        physical_laplacian_error_rmse=float(
            normalized_laplacian_rmse.item() * (pv_max - pv_min)
        ),
        spread_hotspot_q999_over_q95=float(
            (spread_q999 / spread_q95.clamp_min(1.0e-12)).item()
        ),
    )
    return values


@torch.no_grad()
def evaluate_panel(
    *,
    model: torch.nn.Module,
    dataset: Any,
    manifest: list[dict[str, Any]],
    bridge: ContinuousQuadraticBridge,
    nfe_budget: int,
    ensemble_size: int,
    member_chunk_size: int,
    brownian_base_steps: int,
    sampling_seed: int,
    pv_min: float,
    pv_max: float,
    output_dir: Path,
    weights_kind: str,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_window: list[dict[str, float]] = []
    total_seconds = 0.0
    peak = 0
    for window_index, entry in enumerate(manifest):
        sample = _sample_from_manifest(dataset, entry)
        source = build_bridge_source(sample["condition_lr"][None].to(next(model.parameters()).device).float())
        target = fold_hr_target(sample["hr"][None].to(source.device).float())
        result = sample_bare_heun_exact_noise(
            model=model,
            source=source,
            bridge=bridge,
            ensemble_size=ensemble_size,
            nfe_budget=nfe_budget,
            member_chunk_size=member_chunk_size,
            sampling_seed=sampling_seed + 1009 * int(entry["start"]),
            brownian_base_steps=brownian_base_steps,
        )
        metrics = panel_metrics(result.samples, target, pv_min=pv_min, pv_max=pv_max)
        metrics.update(
            window=float(window_index),
            seed=float(entry["seed"]),
            member=float(entry["member"]),
            start=float(entry["start"]),
        )
        per_window.append(metrics)
        total_seconds += result.elapsed_seconds
        peak = max(peak, result.peak_gpu_bytes)
        mean = result.samples.mean(dim=1)
        spread = result.samples.std(dim=1, unbiased=True)
        np.savez_compressed(
            output_dir / f"start{int(entry['start']):04d}.npz",
            source=source.cpu().numpy().astype(np.float32),
            target=target.cpu().numpy().astype(np.float32),
            samples=result.samples.cpu().numpy().astype(np.float32),
            ensemble_mean=mean.cpu().numpy().astype(np.float32),
            ensemble_spread=spread.cpu().numpy().astype(np.float32),
            metadata_json=np.array(json.dumps(entry)),
        )
    keys = [
        key
        for key in per_window[0]
        if key not in {"window", "seed", "member", "start"}
    ]
    summary = {
        key: float(np.mean([row[key] for row in per_window])) for key in keys
    }
    summary.update(
        nfe=float(nfe_budget),
        ensemble_size=float(ensemble_size),
        windows=float(len(per_window)),
        sampling_seconds=float(total_seconds),
        peak_gpu_bytes=float(peak),
    )
    import pandas as pd

    pd.DataFrame(per_window).to_csv(output_dir / "per_window_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps({"weights_kind": weights_kind, **summary}, indent=2)
    )
    return summary


@torch.no_grad()
def audit_time_conditioning(
    *,
    model: torch.nn.Module,
    dataset: Any,
    manifest: list[dict[str, Any]],
    bridge: ContinuousQuadraticBridge,
    noise_repeats: int,
    seed: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    grid = [index / 10.0 for index in range(10)]
    midpoints = [(index + 0.5) / 10.0 for index in range(10)]
    random_generator = torch.Generator(device="cpu").manual_seed(seed + 88_001)
    random_times = torch.rand(10, generator=random_generator).tolist()
    families = {
        "grid": grid,
        "midpoint": midpoints,
        "random": random_times,
        "endpoint_zero": [0.0],
        "endpoint_one": [1.0],
    }
    rows: list[dict[str, float | str]] = []
    for window, entry in enumerate(manifest):
        sample = _sample_from_manifest(dataset, entry)
        source = build_bridge_source(sample["condition_lr"][None].to(device).float())
        target = fold_hr_target(sample["hr"][None].to(device).float())
        for repeat in range(noise_repeats):
            generator = torch.Generator(device=device).manual_seed(
                seed + 1009 * window + 17 * repeat
            )
            noise = torch.randn(source.shape, device=device, generator=generator)
            for family, times in families.items():
                for value in times:
                    t = torch.tensor([value], device=device)
                    state = bridge.interpolate(source, target, t, noise)
                    truth = bridge.target_drift(
                        source, target, t, noise, marginalize_t1=True
                    )
                    prediction = model(state, source, t)
                    rows.append(
                        {
                            "family": family,
                            "time": value,
                            "mse": float((prediction - truth).square().mean().item()),
                        }
                    )
    def family_mean(name: str) -> float:
        return float(np.mean([row["mse"] for row in rows if row["family"] == name]))

    grid_mean = family_mean("grid")
    midpoint_mean = family_mean("midpoint")
    return {
        "grid_mean_excluding_t1": grid_mean,
        "midpoint_mean": midpoint_mean,
        "random_mean": family_mean("random"),
        "t0_loss": family_mean("endpoint_zero"),
        "t1_conditional_mean_loss": family_mean("endpoint_one"),
        "midpoint_grid_ratio": midpoint_mean / max(grid_mean, 1.0e-30),
        "rows": rows,
    }


def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model)
