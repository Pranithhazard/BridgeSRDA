from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.filo_srda_qg.sr_prior.source import (
    build_bridge_source,
    fold_hr_target,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge
from src.filo_srda_qg.sb_prior_retrain_qg.validation import (
    build_validation_manifest,
    clone_model,
    panel_metrics,
)

from .bridge import denseobs_conditional_bridge_loss_at_time
from .conditioning import (
    build_dense_observation_condition,
)
from .sampler import sample_denseobs_conditional_bare_heun_exact_noise


def _sample_from_manifest(
    dataset: Any, entry: dict[str, Any], *, observation_seed: int
) -> dict[str, Any]:
    original = dataset.active_indices
    try:
        dataset.active_indices = [int(entry["dataset_index"])]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(observation_seed))
            return dataset[0]
    finally:
        dataset.active_indices = original


def _method(config_method: str | None) -> str:
    return config_method or "periodic_bilinear"


@torch.no_grad()
def evaluate_denseobs_conditional_panel(
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
    observation_seed: int,
    pv_min: float,
    pv_max: float,
    output_dir: Path,
    weights_kind: str,
    interpolation_method: str = "periodic_bilinear",
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    total_seconds = 0.0
    peak = 0
    for window_index, entry in enumerate(manifest):
        sample = _sample_from_manifest(
            dataset,
            entry,
            observation_seed=observation_seed + 1009 * int(entry["start"]),
        )
        device = next(model.parameters()).device
        source = build_bridge_source(sample["condition_lr"][None].to(device).float())
        target = fold_hr_target(sample["hr"][None].to(device).float())
        observations = sample["obs"][None].to(device).float()
        masks = sample["mask"][None].to(device).float()
        dense = build_dense_observation_condition(
            source, observations, masks, method=_method(interpolation_method)
        )
        result = sample_denseobs_conditional_bare_heun_exact_noise(
            model=model,
            source=source,
            observations=observations,
            masks=masks,
            bridge=bridge,
            ensemble_size=ensemble_size,
            nfe_budget=nfe_budget,
            member_chunk_size=member_chunk_size,
            sampling_seed=sampling_seed + 1009 * int(entry["start"]),
            brownian_base_steps=brownian_base_steps,
            interpolation_method=_method(interpolation_method),
        )
        metrics = panel_metrics(result.samples, target, pv_min=pv_min, pv_max=pv_max)
        metrics.update(
            window=float(window_index),
            seed=float(entry["seed"]),
            member=float(entry["member"]),
            start=float(entry["start"]),
        )
        rows.append(metrics)
        total_seconds += result.elapsed_seconds
        peak = max(peak, result.peak_gpu_bytes)
        np.savez_compressed(
            output_dir / f"start{int(entry['start']):04d}.npz",
            source=source.cpu().numpy().astype(np.float32),
            target=target.cpu().numpy().astype(np.float32),
            observations=observations.cpu().numpy().astype(np.float32),
            masks=masks.cpu().numpy().astype(np.float32),
            dense_observations=dense.cpu().numpy().astype(np.float32),
            samples=result.samples.cpu().numpy().astype(np.float32),
            ensemble_mean=result.samples.mean(dim=1).cpu().numpy().astype(np.float32),
            ensemble_spread=(
                result.samples.std(dim=1, unbiased=ensemble_size > 1)
                .cpu()
                .numpy()
                .astype(np.float32)
            ),
            metadata_json=np.array(json.dumps(entry)),
        )
    if not rows:
        raise ValueError("empty validation manifest")
    keys = [key for key in rows[0] if key not in {"window", "seed", "member", "start"}]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary.update(
        nfe=float(nfe_budget),
        ensemble_size=float(ensemble_size),
        windows=float(len(rows)),
        sampling_seconds=float(total_seconds),
        peak_gpu_bytes=float(peak),
    )
    pd.DataFrame(rows).to_csv(output_dir / "per_window_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "weights_kind": weights_kind,
                "interpolation_method": interpolation_method,
                **summary,
            },
            indent=2,
        )
    )
    return summary


@torch.no_grad()
def audit_denseobs_conditional_time(
    *,
    model: torch.nn.Module,
    dataset: Any,
    manifest: list[dict[str, Any]],
    bridge: ContinuousQuadraticBridge,
    noise_repeats: int,
    seed: int,
    observation_seed: int,
    interpolation_method: str = "periodic_bilinear",
) -> dict[str, Any]:
    device = next(model.parameters()).device
    random_generator = torch.Generator(device="cpu").manual_seed(seed + 88_001)
    families = {
        "grid": [index / 10.0 for index in range(10)],
        "midpoint": [(index + 0.5) / 10.0 for index in range(10)],
        "random": torch.rand(10, generator=random_generator).tolist(),
        "endpoint_zero": [0.0],
        "endpoint_one": [1.0],
    }
    rows: list[dict[str, float | str]] = []
    for window, entry in enumerate(manifest):
        sample = _sample_from_manifest(
            dataset, entry, observation_seed=observation_seed + 1009 * window
        )
        source = build_bridge_source(sample["condition_lr"][None].to(device).float())
        target = fold_hr_target(sample["hr"][None].to(device).float())
        observations = sample["obs"][None].to(device).float()
        masks = sample["mask"][None].to(device).float()
        dense = build_dense_observation_condition(
            source, observations, masks, method=interpolation_method
        )
        for repeat in range(noise_repeats):
            noise_generator = torch.Generator(device=device).manual_seed(
                seed + 1009 * window + 17 * repeat
            )
            noise = torch.randn(source.shape, device=device, generator=noise_generator)
            for family, times in families.items():
                for value in times:
                    tau = torch.tensor([value], device=device)
                    state = bridge.interpolate(source, target, tau, noise)
                    truth = bridge.target_drift(
                        source, target, tau, noise, marginalize_t1=True
                    )
                    prediction = model(state, source, dense, tau)
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




# Shared trainer expects these generic callable names from the selected runtime.
evaluate_conditional_panel = evaluate_denseobs_conditional_panel
audit_conditional_time = audit_denseobs_conditional_time
conditional_bridge_loss_at_time = denseobs_conditional_bridge_loss_at_time

__all__ = [
    "audit_conditional_time",
    "audit_denseobs_conditional_time",
    "build_validation_manifest",
    "clone_model",
    "conditional_bridge_loss_at_time",
    "evaluate_conditional_panel",
    "evaluate_denseobs_conditional_panel",
]
