from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import (
    DEFAULT_CFD_CONFIG,
    DEFAULT_LETKF_CONFIG,
    DEFAULT_SRDA_CONFIG,
    load_letkf_matched_observations,
    make_filter,
    make_initial_hr_pv,
    resolve,
)
from src.filo_srda_qg.sb_prior_retrain_qg.cycling_schedule import analysis_indices
from src.filo_srda_qg.shared_srda.metrics import (
    ensemble_summary,
    rank_histogram_counts,
)
from src.four_dim_srda.config.config_loader import load_config
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.four_dim_srda.da.letkf_block_processing import (
    LETKFBlockProcessingConfig,
    calc_decay_factors_of_localization,
    calc_distances_from_obs_grid_points,
    hr_assimilate_with_existing_data,
    make_splitted_grid_and_its_center_indices_array,
)
from src.four_dim_srda.utils.enkf_helper import get_obs_matrix
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from src.four_dim_srda.utils.sr_da_helper import get_uhr_and_hr_pvs
from src.qg_model.qg_model import QGModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--baseline-kind",
        choices=["true_lr_letkf"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=9991)
    parser.add_argument("--ensemble-size", type=int, default=30)
    parser.add_argument("--config-cfd-path", default=DEFAULT_CFD_CONFIG)
    parser.add_argument("--config-srda-path", default=DEFAULT_SRDA_CONFIG)
    parser.add_argument("--config-letkf-path", default=DEFAULT_LETKF_CONFIG)
    parser.add_argument("--obs-noise-std", type=float, default=0.022)
    parser.add_argument(
        "--lr-obs-grid-mode",
        choices=["physical_matched"],
        default="physical_matched",
        help="physical_matched maps HR ogx08/ogy08 to LR ogx02/ogy02.",
    )
    parser.add_argument("--selected-analysis-indices", nargs="+", type=int, default=[0, 49, 99, 149, 198])
    parser.add_argument("--max-analyses", type=int, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def lift_lr_to_hr(lr_state: torch.Tensor, *, hr_shape: tuple[int, int, int]) -> torch.Tensor:
    if lr_state.ndim != 4:
        raise ValueError("lr_state must have shape (M,Z,Y,X)")
    return F.interpolate(
        lr_state[:, None],
        size=hr_shape,
        mode="trilinear",
        align_corners=True,
    )[:, 0]


def integrate_lr_from_lr_state(
    *,
    lr_state: torch.Tensor,
    num_integrate_steps: int,
    cfg_cfd: CFDConfig,
) -> torch.Tensor:
    lr_model = QGModel(cfg_cfd.lr_base_config, show_input_cfg_info=False)
    lr_model.initialize_pv(grid_pv=lr_state)
    forecast = [lr_model.get_grid_pv()]
    steps_per_loop = int(cfg_cfd.time_config.output_lr_dt / cfg_cfd.time_config.lr_dt)
    for _ in range(num_integrate_steps):
        lr_model.integrate_n_steps(
            dt_per_step=cfg_cfd.time_config.lr_dt,
            n_steps=steps_per_loop,
        )
        forecast.append(lr_model.get_grid_pv())
    return torch.stack(forecast, dim=1)


def make_observed_grid(
    truth: torch.Tensor,
    *,
    interval_x: int,
    interval_y: int,
    noise_std: float,
    seed: int,
) -> torch.Tensor:
    if truth.ndim != 4:
        raise ValueError("truth must have shape (T,Z,Y,X)")
    generator = torch.Generator(device=truth.device)
    generator.manual_seed(int(seed))
    mask = torch.zeros_like(truth[0])
    mask[:, 0:: int(interval_y), 0:: int(interval_x)] = 1.0
    noise = torch.randn(
        truth.shape, device=truth.device, dtype=truth.dtype, generator=generator
    ) * float(noise_std)
    missing = torch.full_like(truth, float("nan"))
    return torch.where(mask[None] > 0, truth + noise, missing)


def make_grid_letkf_state(
    *,
    cfg_base: Any,
    cfg_letkf: LETKFBlockProcessingConfig,
    observations: torch.Tensor,
) -> dict[str, Any]:
    model = QGModel(cfg_base, show_input_cfg_info=False)
    x_coords, y_coords, z_coords = model.get_grids()
    flattened_x = x_coords.reshape(-1)
    flattened_y = y_coords.reshape(-1)
    flattened_z = z_coords.reshape(-1)
    block_idx, center_idx = make_splitted_grid_and_its_center_indices_array(cfg_letkf)
    obs_matrix = get_obs_matrix(
        obs=observations[0],
        nz=cfg_letkf.nz,
        ny=cfg_letkf.ny,
        nx=cfg_letkf.nx,
        device="cpu",
    )
    dist_xy, dist_z = calc_distances_from_obs_grid_points(
        obs_matrix=obs_matrix,
        flattend_x_coords=flattened_x,
        flattend_y_coords=flattened_y,
        flattend_z_coords=flattened_z,
        cfg=cfg_letkf,
    )
    decay_factors = calc_decay_factors_of_localization(
        dist_xy=dist_xy,
        dist_z=dist_z,
        obs_matrix=obs_matrix,
        cfg=cfg_letkf,
    )
    return {
        "obs_matrix": obs_matrix,
        "dist_xy": dist_xy,
        "dist_z": dist_z,
        "decay_factors": decay_factors,
        "block_idx": block_idx,
        "center_idx": center_idx,
    }


def assimilate(
    *,
    members: torch.Tensor,
    observations_at_cycle: torch.Tensor,
    state: dict[str, Any],
    cfg_letkf: LETKFBlockProcessingConfig,
    device: torch.device,
) -> torch.Tensor:
    letkf_dtype = state["obs_matrix"].dtype
    return hr_assimilate_with_existing_data(
        hr_pv=members.to(device=device, dtype=letkf_dtype),
        gt_added_noise=torch.nan_to_num(
            observations_at_cycle.to(device=device, dtype=letkf_dtype), nan=1.0e10
        ),
        obs_matrix=state["obs_matrix"].to(device=device, dtype=letkf_dtype),
        dist_xy=state["dist_xy"].to(device=device, dtype=letkf_dtype),
        dist_z=state["dist_z"].to(device=device, dtype=letkf_dtype),
        decay_factors=state["decay_factors"].to(device=device, dtype=letkf_dtype),
        block_idx=state["block_idx"],
        center_idx=state["center_idx"],
        cfg=cfg_letkf,
    ).float()


def write_outputs(
    *,
    temporary: pathlib.Path,
    seed: int,
    mean_analysis: torch.Tensor,
    spread_analysis: torch.Tensor,
    selected_members: torch.Tensor | None,
    selected_indices: list[int],
    rank_histogram: torch.Tensor,
    online_metrics: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> None:
    temporary.mkdir(parents=True, exist_ok=False)
    mean_np = mean_analysis.cpu().numpy().astype(np.float32)
    spread_np = spread_analysis.cpu().numpy().astype(np.float32)
    selected_np = (
        selected_members.cpu().numpy().astype(np.float32)
        if selected_members is not None
        else np.empty((0,), dtype=np.float32)
    )
    selected_indices_np = np.asarray(selected_indices, dtype=np.int64)
    rank_np = rank_histogram.cpu().numpy().astype(np.int64)

    np.save(temporary / "mean_analysis.npy", mean_np)
    np.save(temporary / "spread_analysis.npy", spread_np)
    np.save(temporary / "letkf_mean_analysis.npy", mean_np)
    np.save(temporary / "letkf_spread_analysis.npy", spread_np)
    np.save(temporary / "selected_members.npy", selected_np)
    np.save(temporary / "selected_member_analysis_indices.npy", selected_indices_np)
    np.save(temporary / "rank_histogram.npy", rank_np)
    np.save(temporary / "letkf_rank_histogram.npy", rank_np)
    pd.DataFrame(online_metrics).to_csv(temporary / "online_metrics.csv", index=False)
    pd.DataFrame(online_metrics).to_csv(temporary / "metrics.csv", index=False)
    (temporary / "runtime.json").write_text(json.dumps(runtime, indent=2, default=str))
    np.savez_compressed(
        temporary / f"UHR_seed_{seed:05}.npz",
        letkf_mean_analysis=mean_np,
        letkf_spread_analysis=spread_np,
        selected_members=selected_np,
        selected_member_analysis_indices=selected_indices_np,
        letkf_rank_histogram=rank_np,
        online_metrics=np.asarray(online_metrics, dtype=object),
        metadata=np.asarray(runtime, dtype=object),
    )
    (temporary / "_COMPLETE").write_text("complete\n")


def run(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root_dir).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = int(args.seed)
    ensemble = int(args.ensemble_size)
    cfg_srda = load_config(
        model_name="UNetMaxVitVer01",
        config_path=str(resolve(root, args.config_srda_path)),
    )
    cfg_base = CFDConfig.load(resolve(root, args.config_cfd_path))
    cfg = copy.deepcopy(cfg_base)
    cfg.lr_base_config.n_batch = ensemble
    cfg.hr_base_config.n_batch = ensemble
    cfg.lr_base_config.device = str(device)
    cfg.hr_base_config.device = str(device)
    cfg.uhr_base_config.device = str(device)

    assimilation_period = cfg.da_config.segment_length - cfg.da_config.forecast_span - 1
    forecast_span = cfg.da_config.forecast_span
    num_times = cfg_srda.dataset_config.max_start_time_index + assimilation_period + forecast_span
    output_scale_factor = int(cfg.time_config.output_lr_dt // cfg.time_config.output_hr_dt)
    expected_indices = analysis_indices(
        num_times=num_times, assimilation_period=assimilation_period
    )
    if args.max_analyses is not None:
        expected_indices = expected_indices[: int(args.max_analyses)]
    selected_lookup = set(int(v) for v in args.selected_analysis_indices)
    low_pass_filter = make_filter(cfg)
    uhr_root = root / "data/four_dim_srda/handover/cfd_simulation/qg_model/uhr_pv_narrow_jet"
    _, hr_pvs = get_uhr_and_hr_pvs(
        result_dir_path=str(uhr_root / f"seed{seed:05}"),
        i_seed_uhr=seed,
        num_times=num_times,
        cfg=cfg_base,
    )
    hr_pvs = hr_pvs.to(device=device).float()
    _, observation_source_path = load_letkf_matched_observations(
        root=root,
        seed=seed,
        reference=hr_pvs,
        srda_config_path=resolve(root, args.config_srda_path),
    )

    lr_truth = low_pass_filter.apply(hr_grid_data=hr_pvs).float()
    lr_obs_interval_x = max(1, int(cfg_srda.dataset_config.obs_grid_interval_x) // 4)
    lr_obs_interval_y = max(1, int(cfg_srda.dataset_config.obs_grid_interval_y) // 4)
    lr_observations = make_observed_grid(
        lr_truth,
        interval_x=lr_obs_interval_x,
        interval_y=lr_obs_interval_y,
        noise_std=float(args.obs_noise_std),
        seed=seed * 1009 + 6113,
    )

    cfg_letkf_hr = LETKFBlockProcessingConfig.load(resolve(root, args.config_letkf_path))
    cfg_letkf_hr.num_ens = ensemble
    cfg_letkf_hr.data_dir = str(root / "data/four_dim_srda/handover/cfd_simulation/qg_model")
    cfg_letkf_lr = copy.deepcopy(cfg_letkf_hr)
    cfg_letkf_lr.nx = int(cfg.lr_base_config.nx)
    cfg_letkf_lr.ny = int(cfg.lr_base_config.ny)
    cfg_letkf_lr.nz = int(cfg.lr_base_config.nz)
    cfg_letkf_lr.obs_grid_interval_x = int(lr_obs_interval_x)
    cfg_letkf_lr.obs_grid_interval_y = int(lr_obs_interval_y)
    cfg_letkf_lr.block_size = min(int(cfg_letkf_hr.block_size), int(cfg.lr_base_config.nx), int(cfg.lr_base_config.ny))
    lr_letkf_state = make_grid_letkf_state(
        cfg_base=cfg.lr_base_config,
        cfg_letkf=cfg_letkf_lr,
        observations=lr_observations,
    )

    final_dir = (
        root
        / "work/experiments/072_conditional_sb_srda_qg/evaluation/narrow_jet"
        / "lr_letkf_native"
        / f"seed{seed:05d}"
        / f"m{ensemble}"
    )
    temporary = final_dir.with_name(final_dir.name + ".partial")
    if final_dir.exists():
        if args.skip_existing and (final_dir / "_COMPLETE").exists():
            print(f"skip completed output: {final_dir}")
            return 0
        if not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {final_dir}")
        shutil.rmtree(final_dir)
    if temporary.exists():
        if not args.allow_overwrite:
            raise FileExistsError(f"Stale partial output exists: {temporary}")
        shutil.rmtree(temporary)

    initialization_seed = 555 * seed
    set_seeds(initialization_seed, use_deterministic=True)
    initial_hr = make_initial_hr_pv(copy.deepcopy(cfg)).to(device=device).float()
    if initial_hr.shape[0] != ensemble:
        raise RuntimeError(
            "Initial HR ensemble has unexpected size: "
            f"got {initial_hr.shape[0]}, expected {ensemble}"
        )
    initial_hr_spread_rms = float(initial_hr.std(dim=0, unbiased=True).square().mean().sqrt().item())
    lr_state = low_pass_filter.apply(hr_grid_data=initial_hr).float()
    initial_lr_spread_rms = float(lr_state.std(dim=0, unbiased=True).square().mean().sqrt().item())

    mean_analysis: list[torch.Tensor] = []
    spread_analysis: list[torch.Tensor] = []
    selected_members: list[torch.Tensor] = []
    selected_indices: list[int] = []
    online_metrics: list[dict[str, Any]] = []
    rank_counts = torch.zeros(ensemble + 1, dtype=torch.long)
    lr_forecast_seconds = 0.0
    lifting_seconds = 0.0
    letkf_analysis_seconds = 0.0
    started = time.perf_counter()

    for cycle_index in tqdm(range(num_times), desc=f"{args.baseline_kind} seed {seed}"):
        if cycle_index == 0 or cycle_index % assimilation_period:
            continue
        if len(mean_analysis) >= len(expected_indices):
            break
        integrate_steps = (assimilation_period + forecast_span) // output_scale_factor
        before = time.perf_counter()
        lr_forecast = integrate_lr_from_lr_state(
            lr_state=lr_state,
            num_integrate_steps=integrate_steps,
            cfg_cfd=cfg,
        )
        lr_forecast_seconds += time.perf_counter() - before
        lr_forecast_current = lr_forecast[:, assimilation_period // output_scale_factor]

        before = time.perf_counter()
        lr_analysis = assimilate(
            members=lr_forecast_current,
            observations_at_cycle=lr_observations[cycle_index],
            state=lr_letkf_state,
            cfg_letkf=cfg_letkf_lr,
            device=device,
        )
        letkf_analysis_seconds += time.perf_counter() - before
        lr_state = lr_analysis
        before = time.perf_counter()
        analysis_members = lift_lr_to_hr(
            lr_analysis,
            hr_shape=tuple(hr_pvs.shape[-3:]),
        )
        lifting_seconds += time.perf_counter() - before
        grid_space = "lr"

        truth = hr_pvs[cycle_index].float()
        analysis_number = len(mean_analysis)
        metrics = ensemble_summary(
            analysis_members[None].float(), truth[None], pv_min=0.0, pv_max=1.0
        )
        online_metrics.append(
            {
                "analysis_number": analysis_number,
                "cycle_index": int(cycle_index),
                **metrics,
            }
        )
        rank_counts += rank_histogram_counts(
            analysis_members[None].float(), truth[None].float()
        ).cpu()
        mean_analysis.append(analysis_members.mean(dim=0))
        spread_analysis.append(analysis_members.std(dim=0, unbiased=True))
        if analysis_number in selected_lookup:
            selected_indices.append(analysis_number)
            selected_members.append(analysis_members)

    elapsed = time.perf_counter() - started
    if len(mean_analysis) != len(expected_indices):
        raise RuntimeError(f"Saved {len(mean_analysis)} analyses, expected {len(expected_indices)}")
    runtime = {
        "seed": seed,
        "baseline_kind": args.baseline_kind,
        "grid_space": grid_space,
        "total_wall_time_seconds": elapsed,
        "lr_forecast_seconds": lr_forecast_seconds,
        "lifting_seconds": lifting_seconds,
        "letkf_analysis_seconds": letkf_analysis_seconds,
        "seconds_per_analysis": elapsed / float(len(mean_analysis)),
        "ensemble_size": ensemble,
        "analysis_indices": expected_indices,
        "selected_analysis_indices": selected_indices,
        "initialization": "canonical_perturbed_hr_jet_then_low_pass_to_lr",
        "initialization_seed": initialization_seed,
        "initial_hr_spread_rms": initial_hr_spread_rms,
        "initial_lr_spread_rms": initial_lr_spread_rms,
        "lr_obs_grid_mode": args.lr_obs_grid_mode,
        "lr_obs_grid_interval": [int(lr_obs_interval_x), int(lr_obs_interval_y)],
        "hr_obs_grid_interval": [
            int(cfg_srda.dataset_config.obs_grid_interval_x),
            int(cfg_srda.dataset_config.obs_grid_interval_y),
        ],
        "observation_noise_std": float(args.obs_noise_std),
        "observation_source_path": str(observation_source_path),
        "config_cfd_path": str(resolve(root, args.config_cfd_path)),
        "config_srda_path": str(resolve(root, args.config_srda_path)),
        "config_letkf_path": str(resolve(root, args.config_letkf_path)),
    }
    write_outputs(
        temporary=temporary,
        seed=seed,
        mean_analysis=torch.stack(mean_analysis),
        spread_analysis=torch.stack(spread_analysis),
        selected_members=torch.stack(selected_members) if selected_members else None,
        selected_indices=selected_indices,
        rank_histogram=rank_counts,
        online_metrics=online_metrics,
        runtime=runtime,
    )
    temporary.rename(final_dir)
    print(f"wrote {final_dir} analyses={len(mean_analysis)} elapsed={elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
