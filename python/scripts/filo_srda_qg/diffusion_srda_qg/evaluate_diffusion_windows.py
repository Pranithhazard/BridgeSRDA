from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.four_dim_srda.models.frozen_baseline import NotsuAnchorWrapper
from src.filo_srda_qg.diffusion_srda_qg.data import (
    diffusion_conditions_to_zero_one,
    diffusion_values_to_zero_one,
    make_diffusion_dataset,
    make_diffusion_loader,
)
from src.filo_srda_qg.diffusion_srda_qg.diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
)
from src.filo_srda_qg.diffusion_srda_qg.metrics import (
    ensemble_summary,
    rank_histogram_counts,
)
from src.filo_srda_qg.diffusion_srda_qg.model import make_diffusion_model
from src.filo_srda_qg.shared_srda.source import build_source_mean, fold_pv
from src.four_dim_srda.utils.random_seed_helper import set_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--protocol", choices=["matched_clean", "training_matched"], required=True
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--eta", type=float)
    parser.add_argument(
        "--weights-kind",
        choices=["primary", "reconstruction", "noise"],
        default="primary",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve(root: pathlib.Path, path: str | pathlib.Path) -> pathlib.Path:
    value = pathlib.Path(path)
    return value if value.is_absolute() else root / value


def load_weights(model: torch.nn.Module, path: pathlib.Path, device: torch.device) -> None:
    state: Any = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def resolved_checkpoint_config(
    configured: dict[str, Any],
    checkpoint_path: pathlib.Path,
    device: torch.device,
) -> dict[str, Any]:
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        resolved = checkpoint.get("resolved_config") if isinstance(checkpoint, dict) else None
        if isinstance(resolved, dict):
            return resolved
    return configured


def metadata_rows(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return [{} for _ in range(batch_size)]
    rows: list[dict[str, Any]] = []
    for index in range(batch_size):
        row: dict[str, Any] = {}
        for key, value in metadata.items():
            item = value[index] if hasattr(value, "__len__") and not isinstance(value, str) else value
            row[key] = item.item() if hasattr(item, "item") else item
        rows.append(row)
    return rows


def deterministic_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prefix: str,
    scale: float,
) -> dict[str, float]:
    error = prediction - target
    rmse = torch.sqrt(error.square().mean()).item()
    return {
        f"{prefix}_normalized_mae": error.abs().mean().item(),
        f"{prefix}_normalized_rmse": rmse,
        f"{prefix}_physical_mae": error.abs().mean().item() * scale,
        f"{prefix}_physical_rmse": rmse * scale,
    }


def horizontal_power(values: torch.Tensor) -> torch.Tensor:
    power = torch.fft.rfft2(values, dim=(-2, -1), norm="ortho").abs().square()
    return power.mean(dim=tuple(range(power.ndim - 2)))


def gradient_variance(values: torch.Tensor) -> float:
    dx = values[..., 1:] - values[..., :-1]
    dy = values[..., 1:, :] - values[..., :-1, :]
    return 0.5 * (dx.var().item() + dy.var().item())


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_path)
    root = pathlib.Path(args.root_dir).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = int(config["eval"].get("seed", 42))
    set_seeds(seed, use_deterministic=True)
    ensemble_size = int(args.ensemble_size or config["eval"]["ensemble_size"])
    steps = int(args.num_steps or config["eval"]["num_steps"])
    eta = float(args.eta if args.eta is not None else config["eval"]["eta"])
    if ensemble_size < 2:
        raise ValueError("Independent-window probabilistic evaluation requires M >= 2")
    chunk_size = int(config["eval"].get("member_chunk_size", 1))
    run_name = str(config["output"]["run_name"])
    suffix = (
        f"weights_{args.weights_kind}_steps{steps}_eta{eta:g}_m{ensemble_size}"
    )
    work_root = resolve(root, config["output"]["work_root"])
    output_dir = work_root / "metrics" / run_name / args.protocol / suffix
    samples_dir = work_root / "samples" / run_name / args.protocol / suffix
    occupied = [
        path
        for path in (output_dir, samples_dir)
        if path.exists() and any(path.iterdir())
    ]
    if occupied and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {occupied}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = work_root / "checkpoints" / run_name
    checkpoint_config = resolved_checkpoint_config(
        config, checkpoint_dir / "checkpoint.pth", device
    )
    dataset_config = dict(checkpoint_config["dataset"])
    if args.protocol == "matched_clean":
        dataset_config["fixed_obs_mask_index"] = 0
        dataset_config["obs_noise_std"] = 0.0
    else:
        dataset_config["fixed_obs_mask_index"] = None
        dataset_config["obs_noise_std"] = 0.022
    dataset = make_diffusion_dataset(
        base_config=dataset_config,
        root_dir=root,
        split="test",
        samples_per_epoch=None,
    )
    loader, _ = make_diffusion_loader(
        dataset=dataset,
        split="test",
        batch_size=int(config["eval"].get("batch_size", 1)),
        num_workers=0,
        seed=seed,
    )
    model = make_diffusion_model(
        checkpoint_config["model"]
    ).to(device)
    weights_name = {
        "primary": "best_weights.pth",
        "reconstruction": "best_reconstruction_weights.pth",
        "noise": "best_noise_weights.pth",
    }[args.weights_kind]
    weights_path = checkpoint_dir / weights_name
    if not weights_path.exists():
        raise FileNotFoundError(f"Requested weights do not exist: {weights_path}")
    load_weights(model, weights_path, device)
    model.eval()
    diffusion = GaussianDiffusion(
        DiffusionConfig.from_dict(checkpoint_config["diffusion"])
    ).to(device)
    anchor = NotsuAnchorWrapper(
        config_path=resolve(root, config["anchor"]["config_path"]),
        weights_path=resolve(root, config["anchor"]["weights_path"]),
        model_name=str(config["anchor"].get("model_name", "UNetMaxVitVer01")),
        map_location=device,
    ).to(device)
    anchor.eval()

    metric_dataset_config = checkpoint_config["dataset"]
    data_scaling = str(metric_dataset_config.get("data_scaling", "zero_one"))
    scale = float(
        metric_dataset_config["pv_max"] - metric_dataset_config["pv_min"]
    )
    save_indices = set(int(value) for value in config["eval"].get("save_batch_indices", [0]))
    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    rank_counts = torch.zeros(ensemble_size + 1, dtype=torch.long)
    time_depth_error = {
        model_name: {
            "absolute": torch.zeros(9, 16, dtype=torch.float64),
            "squared": torch.zeros(9, 16, dtype=torch.float64),
        }
        for model_name in ["lifted_lr", "notsu", "diffusion_mean"]
    }
    time_depth_count = 0
    spectra = {
        name: torch.zeros(64, 65, dtype=torch.float64)
        for name in ["target", "lifted_lr", "notsu", "diffusion_mean", "diffusion_members"]
    }
    spectra_count = 0
    saved: dict[str, list[np.ndarray]] = {
        key: [] for key in ["hr", "lifted_lr", "notsu", "ensemble", "condition_lr", "obs"]
    }
    saved_metadata: list[str] = []
    saved_batch_indices: list[int] = []
    total_seconds = 0.0
    total_model_calls = 0
    peak_gpu_bytes = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            condition_lr = batch["condition_lr"].to(device)
            obs = batch["obs"].to(device)
            mask = batch["mask"].to(device)
            target = diffusion_values_to_zero_one(
                fold_pv(batch["hr"]), data_scaling=data_scaling
            ).cpu()
            lifted_lr = diffusion_values_to_zero_one(
                fold_pv(build_source_mean(batch["source_lr"].to(device))),
                data_scaling=data_scaling,
            ).cpu()
            anchor_lr, _ = diffusion_conditions_to_zero_one(
                condition_lr, obs, mask, data_scaling=data_scaling
            )
            anchor_obs = batch.get("sparse_obs_zero_one")
            if anchor_obs is None:
                _, anchor_obs = diffusion_conditions_to_zero_one(
                    condition_lr, obs, mask, data_scaling=data_scaling
                )
            else:
                anchor_obs = anchor_obs.to(device)
            notsu = fold_pv(anchor(anchor_lr, anchor_obs)).cpu()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result = diffusion.sample(
                model,
                condition_lr=condition_lr,
                obs=obs,
                ensemble_size=ensemble_size,
                steps=steps,
                eta=eta,
                chunk_size=chunk_size,
                generator=torch.Generator(device=device).manual_seed(
                    seed + 10_000 * batch_index
                ),
            )
            samples = diffusion_values_to_zero_one(
                result.samples, data_scaling=data_scaling
            )
            ensemble_mean = samples.mean(dim=1)
            total_seconds += result.elapsed_seconds
            total_model_calls += result.nfe * (
                (ensemble_size + chunk_size - 1) // chunk_size
            )
            if device.type == "cuda":
                peak_gpu_bytes = max(
                    peak_gpu_bytes, torch.cuda.max_memory_allocated(device)
                )
            metadata = metadata_rows(batch.get("metadata"), target.shape[0])
            for sample_index in range(target.shape[0]):
                rows.append(
                    {
                        "batch": batch_index,
                        **metadata[sample_index],
                        **deterministic_metrics(
                            lifted_lr[sample_index],
                            target[sample_index],
                            prefix="lifted_lr",
                            scale=scale,
                        ),
                        **deterministic_metrics(
                            notsu[sample_index],
                            target[sample_index],
                            prefix="notsu",
                            scale=scale,
                        ),
                        **ensemble_summary(
                            samples[sample_index : sample_index + 1],
                            target[sample_index : sample_index + 1],
                            pv_min=float(metric_dataset_config["pv_min"]),
                            pv_max=float(metric_dataset_config["pv_max"]),
                        ),
                        "reverse_steps": steps,
                        "nfe": result.nfe,
                        "eta": eta,
                        "seconds_per_window": result.elapsed_seconds / target.shape[0],
                    }
                )
            for model_name, prediction in {
                "lifted_lr": lifted_lr,
                "notsu": notsu,
                "diffusion_mean": ensemble_mean,
            }.items():
                error = prediction - target
                time_depth_error[model_name]["absolute"] += (
                    error.abs().mean(dim=(3, 4)).sum(dim=0).double()
                )
                time_depth_error[model_name]["squared"] += (
                    error.square().mean(dim=(3, 4)).sum(dim=0).double()
                )
            time_depth_count += target.shape[0]
            rank_counts += rank_histogram_counts(samples, target)
            for name, values in {
                "target": target,
                "lifted_lr": lifted_lr,
                "notsu": notsu,
                "diffusion_mean": ensemble_mean,
                "diffusion_members": samples,
            }.items():
                spectra[name] += horizontal_power(values).double()
            spectra_count += 1
            diagnostic_rows.append(
                {
                    "batch": batch_index,
                    "target_gradient_variance": gradient_variance(target),
                    "lifted_lr_gradient_variance": gradient_variance(lifted_lr),
                    "notsu_gradient_variance": gradient_variance(notsu),
                    "diffusion_mean_gradient_variance": gradient_variance(ensemble_mean),
                    "diffusion_member_gradient_variance": gradient_variance(samples),
                    "target_temporal_increment_rms": torch.sqrt(
                        (target[:, 1:] - target[:, :-1]).square().mean()
                    ).item(),
                    "diffusion_mean_temporal_increment_rms": torch.sqrt(
                        (ensemble_mean[:, 1:] - ensemble_mean[:, :-1]).square().mean()
                    ).item(),
                    "diffusion_member_temporal_increment_rms": torch.sqrt(
                        (samples[:, :, 1:] - samples[:, :, :-1]).square().mean()
                    ).item(),
                }
            )

            if batch_index in save_indices:
                for key, values in {
                    "hr": target,
                    "lifted_lr": lifted_lr,
                    "notsu": notsu,
                    "ensemble": samples,
                    "condition_lr": anchor_lr.cpu(),
                    "obs": anchor_obs.cpu(),
                }.items():
                    saved[key].append(values.numpy())
                saved_metadata.extend(json.dumps(row, default=str) for row in metadata)
                saved_batch_indices.extend([batch_index] * target.shape[0])
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_window_metrics.csv", index=False)
    frame.mean(numeric_only=True).to_frame("value").to_csv(
        output_dir / "metrics_summary.csv"
    )
    if "seed" in frame:
        frame.groupby("seed").mean(numeric_only=True).to_csv(
            output_dir / "metrics_by_seed.csv"
        )
    if "notsu_physical_rmse" in frame and len(frame) >= 10:
        deciles = frame.copy()
        deciles["notsu_error_decile"] = pd.qcut(
            deciles["notsu_physical_rmse"], q=10, labels=False, duplicates="drop"
        )
        deciles.groupby("notsu_error_decile").mean(numeric_only=True).to_csv(
            output_dir / "metrics_by_notsu_error_decile.csv"
        )
    pd.DataFrame(diagnostic_rows).to_csv(
        output_dir / "physical_diagnostics.csv", index=False
    )
    np.savez_compressed(
        output_dir / "horizontal_power_spectra.npz",
        **{key: (value / max(spectra_count, 1)).numpy() for key, value in spectra.items()},
    )
    pd.DataFrame(
        [
            {
                "model": model_name,
                "target_time": target_time,
                "z": z,
                "normalized_mae": (
                    values["absolute"] / max(time_depth_count, 1)
                )[target_time, z].item(),
                "normalized_rmse": torch.sqrt(
                    values["squared"] / max(time_depth_count, 1)
                )[target_time, z].item(),
                "physical_mae": (
                    values["absolute"] / max(time_depth_count, 1)
                )[target_time, z].item()
                * scale,
                "physical_rmse": torch.sqrt(
                    values["squared"] / max(time_depth_count, 1)
                )[target_time, z].item()
                * scale,
            }
            for model_name, values in time_depth_error.items()
            for target_time in range(9)
            for z in range(16)
        ]
    ).to_csv(output_dir / "metrics_by_time_depth.csv", index=False)
    pd.DataFrame(
        {"rank": range(ensemble_size + 1), "count": rank_counts.numpy()}
    ).to_csv(output_dir / "rank_histogram.csv", index=False)
    (output_dir / "runtime.json").write_text(
        json.dumps(
            {
                "total_sampling_seconds": total_seconds,
                "reverse_steps": steps,
                "nfe_per_member": steps,
                "actual_model_calls": total_model_calls,
                "eta": eta,
                "ensemble_size": ensemble_size,
                "weights_kind": args.weights_kind,
                "weights_path": str(weights_path),
                "peak_gpu_bytes": peak_gpu_bytes,
            },
            indent=2,
        )
    )
    if saved["hr"]:
        np.savez_compressed(
            samples_dir / "selected_windows.npz",
            **{key: np.concatenate(values, axis=0) for key, values in saved.items()},
            metadata_json=np.asarray(saved_metadata),
            batch_indices=np.asarray(saved_batch_indices, dtype=np.int64),
        )
    print(f"wrote {output_dir}")
    print(f"wrote {samples_dir}")


if __name__ == "__main__":
    main()
