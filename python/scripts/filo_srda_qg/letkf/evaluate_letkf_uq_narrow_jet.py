from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from src.filo_srda_qg.letkf.letkf_uq import (
    ensemble_diagnostics,
    fixed_subensemble_indices,
    hr_truth_from_uhr,
)
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
from src.qg_model.jet_maker import (
    make_jet_pv_with_linear_profile,
    make_jet_pv_with_tanh_profile,
    make_jet_velocity_with_sech_squared_and_sigmoid_profile,
    make_jet_velocity_with_sech_squared_profile,
)
from src.qg_model.qg_model import QGModel
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--config-cfd-path", type=pathlib.Path, required=True)
    parser.add_argument("--config-letkf-path", type=pathlib.Path, required=True)
    parser.add_argument("--config-srda-path", type=pathlib.Path, required=True)
    parser.add_argument("--model-name", default="UNetMaxVitVer01")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path(
            "work/experiments/072_conditional_sb_srda_qg/evaluation/narrow_jet/hr_letkf_m30_native"
        ),
    )
    parser.add_argument("--matched-ensemble-size", type=int, default=30)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument(
        "--selected-analysis-indices",
        nargs="+",
        type=int,
        default=[0, 49, 99, 149, 198],
    )
    parser.add_argument("--canonical-mean-tolerance", type=float, default=5.0e-5)
    parser.add_argument(
        "--skip-canonical-check",
        action="store_true",
        help=(
            "Run a new native LETKF ensemble without requiring an existing "
            "canonical all_letkf_fcst.npy archive. Use this for new ensemble "
            "sizes such as M30."
        ),
    )
    parser.add_argument(
        "--canonical-strict-analysis-count",
        type=int,
        default=1,
        help=(
            "Require archive agreement only for this many initial analyses. "
            "Later differences are recorded but are not a hard failure because "
            "roundoff grows under the chaotic forecast dynamics."
        ),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--observation-scenario", choices=["canonical_ogx08_ogy08", "sparse_ogx16_ogy16"], default="canonical_ogx08_ogy08")
    parser.add_argument("--observation-scenario-seed", type=int, default=9117)
    parser.add_argument("--max-analyses", type=int, default=None)
    return parser.parse_args()


def resolve(root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    return path if path.is_absolute() else root / path


def initial_ensemble(cfg_cfd: CFDConfig) -> torch.Tensor:
    if cfg_cfd.jet_profile == "tanh":
        return make_jet_pv_with_tanh_profile(cfg_cfd.hr_base_config)
    if cfg_cfd.jet_profile == "linear":
        return make_jet_pv_with_linear_profile(cfg_cfd.hr_base_config)
    if cfg_cfd.jet_profile == "sech-squared":
        velocity = make_jet_velocity_with_sech_squared_profile(
            cfg_cfd.hr_base_config, use_shift_y=False, use_narrow_jet=True
        )[None]
    elif cfg_cfd.jet_profile == "sech_squared_and_sigmoid":
        velocity = make_jet_velocity_with_sech_squared_and_sigmoid_profile(
            cfg_cfd.hr_base_config, use_shift_y=False, use_narrow_jet=True
        )[None]
    else:
        raise ValueError(f"Unsupported jet profile: {cfg_cfd.jet_profile}")
    noise = cfg_cfd.hr_base_config.noise_amplitude * torch.randn(
        size=(cfg_cfd.hr_base_config.n_batch,) + velocity.shape[-3:]
    )
    model = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=False)
    model.initialize(grid_u=velocity, grid_pv_noise=noise)
    return model.get_grid_pv()


def canonical_paths(
    *, root: pathlib.Path, config_cfd: pathlib.Path, config_letkf: pathlib.Path
) -> pathlib.Path:
    match = re.search(r"(na\d+e[+-]\d+)", config_cfd.stem)
    if match is None:
        raise ValueError(f"Cannot infer noise-amplitude tag from {config_cfd}")
    experiment_name = config_letkf.parts[-3]
    run_name = f"{match.group(1)}_{config_letkf.stem}"
    return (
        root
        / "python/results/four_dim_srda"
        / experiment_name
        / "letkf/perform_letkf_hr_using_uhr/use_narrow_jet"
        / run_name
    )


def canonical_seed_index(canonical_dir: pathlib.Path, seed: int) -> int:
    elapsed_path = canonical_dir / "all_elapsed_time.csv"
    if not elapsed_path.exists():
        raise FileNotFoundError(elapsed_path)
    elapsed = pd.read_csv(elapsed_path)
    matches = np.flatnonzero(elapsed["I_SEED_UHR"].to_numpy() == seed)
    if len(matches) != 1:
        raise ValueError(f"Seed {seed} is not uniquely represented in {elapsed_path}")
    return int(matches[0])


def main() -> int:
    args = parse_args()
    root = args.root_dir.resolve()
    config_cfd_path = resolve(root, args.config_cfd_path)
    config_letkf_path = resolve(root, args.config_letkf_path)
    config_srda_path = resolve(root, args.config_srda_path)
    output_root = resolve(root, args.output_root)
    final_dir = output_root / f"seed{args.seed:05d}"
    partial_dir = output_root / f"seed{args.seed:05d}.partial"

    if final_dir.exists():
        if args.skip_existing and (final_dir / "_COMPLETE").exists():
            print(f"skip completed LETKF-UQ seed: {final_dir}")
            return 0
        if not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {final_dir}")
        shutil.rmtree(final_dir)
    if partial_dir.exists():
        if not args.allow_overwrite:
            raise FileExistsError(
                f"Stale partial output exists: {partial_dir}; use --allow-overwrite to restart"
            )
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=False)

    cfg_letkf = LETKFBlockProcessingConfig.load(config_letkf_path)
    if args.observation_scenario != "canonical_ogx08_ogy08" and not args.skip_canonical_check:
        raise ValueError("New observation layouts cannot be checked against a canonical-layout archive")
    cfg_cfd = CFDConfig.load(config_cfd_path)
    native_size = int(cfg_letkf.num_ens)
    if cfg_cfd.hr_base_config.n_batch != native_size:
        raise ValueError(
            f"HR batch {cfg_cfd.hr_base_config.n_batch} != LETKF ensemble {native_size}"
        )
    if not cfg_cfd.seed_config.uhr_seed_start <= args.seed <= cfg_cfd.seed_config.uhr_seed_end:
        raise ValueError(
            f"Seed {args.seed} outside CFD shard range "
            f"[{cfg_cfd.seed_config.uhr_seed_start}, {cfg_cfd.seed_config.uhr_seed_end}]"
        )
    real_dtype = torch.float64 if cfg_cfd.hr_base_config.precision == "double" else torch.float32
    num_times = int(cfg_cfd.time_config.end_time / cfg_cfd.time_config.output_hr_dt)
    assimilation_period = int(cfg_cfd.da_config.assimilation_interval)
    analysis_cycles = list(range(assimilation_period, num_times, assimilation_period))
    if args.max_analyses is not None:
        if args.max_analyses < 1:
            raise ValueError("max_analyses must be positive")
        analysis_cycles = analysis_cycles[:args.max_analyses]
    selected_indices = list(args.selected_analysis_indices)
    if args.canonical_strict_analysis_count < 1:
        raise ValueError("canonical_strict_analysis_count must be at least one")
    if args.canonical_strict_analysis_count > len(analysis_cycles):
        raise ValueError(
            "canonical_strict_analysis_count exceeds the number of analyses"
        )
    if not all(0 <= index < len(analysis_cycles) for index in selected_indices):
        raise ValueError(f"Invalid selected analysis indices: {selected_indices}")
    subset_indices = fixed_subensemble_indices(
        ensemble_size=native_size,
        subset_size=args.matched_ensemble_size,
        seed=args.subset_seed,
    )

    experiment_name = config_letkf_path.parts[-3]
    srda_run_name = config_srda_path.stem
    observation_path = (
        root
        / "python/results/four_dim_srda"
        / experiment_name
        / "srda"
        / args.model_name
        / "use_narrow_jet"
        / srda_run_name
        / f"UHR_seed_{args.seed:05d}.npz"
    )
    if not observation_path.exists():
        raise FileNotFoundError(observation_path)
    with np.load(observation_path, allow_pickle=False) as archive:
        hr_obs = torch.from_numpy(np.asarray(archive["hr_obs"]))
    if tuple(hr_obs.shape) != (
        num_times,
        cfg_cfd.hr_base_config.nz,
        cfg_cfd.hr_base_config.ny,
        cfg_cfd.hr_base_config.nx,
    ):
        raise ValueError(f"Unexpected observation shape: {tuple(hr_obs.shape)}")

    uhr_path = (
        root
        / "data/four_dim_srda"
        / experiment_name
        / "cfd_simulation/qg_model/uhr_pv_narrow_jet"
        / f"seed{args.seed:05d}"
        / f"seed{args.seed:05d}_start000_end{num_times:03d}_uhr_pv.npy"
    )
    if not uhr_path.exists():
        raise FileNotFoundError(uhr_path)
    uhr_truth = np.load(uhr_path, mmap_mode="r")
    observation_seed = None
    if args.observation_scenario != "canonical_ogx08_ogy08":
        # Reuse 073A's exact truth, full noise-array shape, CUDA RNG and mask.
        # Do not use the historical sparse LETKF config, which also retunes
        # cylinder_height. Only layout is changed from the M30 baseline.
        from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import (
            observation_scenario_spec, deterministic_scenario_seed,
            synthetic_observations_from_truth,
        )
        scenario = observation_scenario_spec(args.observation_scenario, canonical_x=8, canonical_y=8)
        observation_seed = deterministic_scenario_seed(seed=args.seed, scenario=scenario.name, base_seed=args.observation_scenario_seed)
        truth = torch.stack([hr_truth_from_uhr(torch.from_numpy(np.array(uhr_truth[i], copy=True)), nz=cfg_cfd.hr_base_config.nz, ny=cfg_cfd.hr_base_config.ny, nx=cfg_cfd.hr_base_config.nx) for i in range(num_times)]).to(device=cfg_cfd.hr_base_config.device, dtype=torch.float32)
        hr_obs = synthetic_observations_from_truth(truth=truth, scenario=scenario, obs_noise_std=float(cfg_letkf.obs_noise_std), seed=observation_seed).cpu()
        del truth
        cfg_letkf.obs_grid_interval_x = scenario.obs_grid_interval_x
        cfg_letkf.obs_grid_interval_y = scenario.obs_grid_interval_y
        observation_path = pathlib.Path(f"generated:{scenario.name}:seed{observation_seed}")
    observation_hash = hashlib.sha256(torch.nan_to_num(hr_obs.float(), nan=1.0e10).contiguous().numpy().tobytes()).hexdigest()

    canonical_mean_path: pathlib.Path | None = None
    canonical_mean = None
    canonical_index: int | None = None
    if not args.skip_canonical_check:
        canonical_dir = canonical_paths(
            root=root,
            config_cfd=config_cfd_path,
            config_letkf=config_letkf_path,
        )
        canonical_mean_path = canonical_dir / "all_letkf_fcst.npy"
        if not canonical_mean_path.exists():
            raise FileNotFoundError(canonical_mean_path)
        canonical_mean = np.load(canonical_mean_path, mmap_mode="r")
        canonical_index = canonical_seed_index(canonical_dir, args.seed)

    state_shape = (
        cfg_cfd.hr_base_config.nz,
        cfg_cfd.hr_base_config.ny,
        cfg_cfd.hr_base_config.nx,
    )
    analysis_shape = (len(analysis_cycles),) + state_shape
    mean_output = np.lib.format.open_memmap(
        partial_dir / "letkf_mean_analysis.npy", mode="w+", dtype=np.float32, shape=analysis_shape
    )
    spread_native_output = np.lib.format.open_memmap(
        partial_dir / f"letkf_spread_analysis_ne{native_size}.npy",
        mode="w+",
        dtype=np.float32,
        shape=analysis_shape,
    )
    spread_matched_output = np.lib.format.open_memmap(
        partial_dir / f"letkf_spread_analysis_ne{args.matched_ensemble_size}.npy",
        mode="w+",
        dtype=np.float32,
        shape=analysis_shape,
    )
    selected_output = np.lib.format.open_memmap(
        partial_dir / f"selected_members_ne{args.matched_ensemble_size}.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(selected_indices), args.matched_ensemble_size) + state_shape,
    )

    model = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=True)
    x_coords, y_coords, z_coords = model.get_grids()
    flattened_x = x_coords.reshape(-1)
    flattened_y = y_coords.reshape(-1)
    flattened_z = z_coords.reshape(-1)
    block_idx, center_idx = make_splitted_grid_and_its_center_indices_array(cfg_letkf)
    obs_matrix = get_obs_matrix(
        obs=hr_obs[assimilation_period],
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
    # Match the canonical evaluator: construct deterministic operators first,
    # then reset RNG immediately before drawing the initial ensemble perturbations.
    set_seeds(seed=(args.seed + 1) * 457, use_deterministic=True)
    model.initialize_pv(initial_ensemble(cfg_cfd))

    device = torch.device(cfg_cfd.hr_base_config.device)
    subset_on_device = torch.as_tensor(subset_indices, device=device, dtype=torch.long)
    rows_native: list[dict[str, float | int]] = []
    rows_matched: list[dict[str, float | int]] = []
    rank_native = torch.zeros(native_size + 1, dtype=torch.long)
    rank_matched = torch.zeros(args.matched_ensemble_size + 1, dtype=torch.long)
    selected_lookup = {analysis: slot for slot, analysis in enumerate(selected_indices)}
    maximum_mean_difference = 0.0
    maximum_mean_rmse_difference = 0.0
    first_mean_difference: float | None = None
    first_mean_rmse_difference: float | None = None
    canonical_tolerance_exceedance_count = 0
    reported_post_verification_drift = False
    started = time.perf_counter()

    loop_stop = analysis_cycles[-1] + 1 if args.max_analyses is not None else num_times
    for cycle_index in tqdm(range(loop_stop), desc=f"LETKF-UQ seed {args.seed}"):
        if cycle_index > 0 and cycle_index % assimilation_period == 0:
            gt_added_noise = torch.nan_to_num(
                hr_obs[cycle_index].to(real_dtype), nan=1.0e10
            )
            analysis_all = hr_assimilate_with_existing_data(
                hr_pv=model.get_grid_pv().to(device),
                gt_added_noise=gt_added_noise.to(device),
                obs_matrix=obs_matrix.to(device),
                dist_xy=dist_xy.to(device),
                dist_z=dist_z.to(device),
                decay_factors=decay_factors.to(device),
                block_idx=block_idx,
                center_idx=center_idx,
                cfg=cfg_letkf,
            )
            analysis_number = len(rows_native)
            expected_cycle = analysis_cycles[analysis_number]
            if cycle_index != expected_cycle:
                raise RuntimeError(f"Analysis cycle {cycle_index} != expected {expected_cycle}")
            truth_uhr = torch.from_numpy(np.array(uhr_truth[cycle_index], copy=True))
            truth_hr = hr_truth_from_uhr(
                truth_uhr,
                nz=cfg_cfd.hr_base_config.nz,
                ny=cfg_cfd.hr_base_config.ny,
                nx=cfg_cfd.hr_base_config.nx,
            ).to(device=device, dtype=torch.float32)
            samples_native = analysis_all.float()
            diagnostics_native = ensemble_diagnostics(samples_native, truth_hr)
            samples_matched = samples_native.index_select(0, subset_on_device)
            diagnostics_matched = ensemble_diagnostics(samples_matched, truth_hr)

            # The canonical evaluator averages the double-precision ensemble
            # before casting its saved trajectory to float32.
            canonical_precision_mean = analysis_all.mean(dim=0).float()
            mean_np = canonical_precision_mean.cpu().numpy().astype(
                np.float32, copy=False
            )
            mean_output[analysis_number] = mean_np
            spread_native_output[analysis_number] = (
                diagnostics_native.spread.cpu().numpy().astype(np.float32, copy=False)
            )
            spread_matched_output[analysis_number] = (
                diagnostics_matched.spread.cpu().numpy().astype(np.float32, copy=False)
            )
            if analysis_number in selected_lookup:
                selected_output[selected_lookup[analysis_number]] = (
                    samples_matched.cpu().numpy().astype(np.float32, copy=False)
                )

            if canonical_mean is None or canonical_index is None:
                mean_difference = float("nan")
                mean_rmse_difference = float("nan")
            else:
                canonical_slice = np.asarray(
                    canonical_mean[canonical_index, cycle_index], dtype=np.float32
                )
                canonical_delta = mean_np - canonical_slice
                mean_difference = float(np.max(np.abs(canonical_delta)))
                canonical_delta_64 = canonical_delta.astype(np.float64, copy=False)
                mean_rmse_difference = float(
                    np.sqrt(np.mean(canonical_delta_64 * canonical_delta_64))
                )
                if first_mean_difference is None:
                    first_mean_difference = mean_difference
                    first_mean_rmse_difference = mean_rmse_difference
                maximum_mean_difference = max(maximum_mean_difference, mean_difference)
                maximum_mean_rmse_difference = max(
                    maximum_mean_rmse_difference, mean_rmse_difference
                )
                tolerance_exceeded = mean_difference > args.canonical_mean_tolerance
                canonical_tolerance_exceedance_count += int(tolerance_exceeded)
                if (
                    analysis_number < args.canonical_strict_analysis_count
                    and tolerance_exceeded
                ):
                    raise RuntimeError(
                        f"Initial canonical LETKF mean mismatch at cycle {cycle_index}: "
                        f"{mean_difference} > {args.canonical_mean_tolerance}"
                    )
                if (
                    analysis_number >= args.canonical_strict_analysis_count
                    and tolerance_exceeded
                    and not reported_post_verification_drift
                ):
                    print(
                        "warning: canonical LETKF pointwise tolerance was exceeded "
                        f"after initial verification at cycle {cycle_index}: "
                        f"max_abs={mean_difference:.8g}, rmse={mean_rmse_difference:.8g}; "
                        "continuing and recording archive divergence diagnostics"
                    )
                    reported_post_verification_drift = True

            rows_native.append(
                {
                    "seed": args.seed,
                    "analysis_number": analysis_number,
                    "cycle_index": cycle_index,
                    "canonical_mean_max_abs_difference": mean_difference,
                    "canonical_mean_rmse_difference": mean_rmse_difference,
                    **diagnostics_native.metrics,
                }
            )
            rows_matched.append(
                {
                    "seed": args.seed,
                    "analysis_number": analysis_number,
                    "cycle_index": cycle_index,
                    "canonical_mean_max_abs_difference": mean_difference,
                    "canonical_mean_rmse_difference": mean_rmse_difference,
                    **diagnostics_matched.metrics,
                }
            )
            rank_native += diagnostics_native.rank_histogram.cpu()
            rank_matched += diagnostics_matched.rank_histogram.cpu()
            model.initialize_pv(grid_pv=analysis_all)
            del (
                diagnostics_native,
                diagnostics_matched,
                samples_native,
                samples_matched,
                canonical_precision_mean,
                truth_hr,
                analysis_all,
            )

        n_steps = int(cfg_cfd.time_config.output_hr_dt / cfg_cfd.time_config.hr_dt)
        model.integrate_n_steps(dt_per_step=cfg_cfd.time_config.hr_dt, n_steps=n_steps)

    elapsed_seconds = time.perf_counter() - started
    if len(rows_native) != len(analysis_cycles) or len(rows_matched) != len(analysis_cycles):
        raise RuntimeError(
            f"Saved {len(rows_native)} native and {len(rows_matched)} matched analyses; "
            f"expected {len(analysis_cycles)}"
        )
    for output in (mean_output, spread_native_output, spread_matched_output, selected_output):
        output.flush()
    del mean_output, spread_native_output, spread_matched_output, selected_output

    pd.DataFrame(rows_native).to_csv(
        partial_dir / f"metrics_ne{native_size}.csv", index=False
    )
    pd.DataFrame(rows_matched).to_csv(
        partial_dir / f"metrics_ne{args.matched_ensemble_size}.csv", index=False
    )
    np.save(partial_dir / f"rank_histogram_ne{native_size}.npy", rank_native.numpy())
    np.save(
        partial_dir / f"rank_histogram_ne{args.matched_ensemble_size}.npy",
        rank_matched.numpy(),
    )
    np.save(
        partial_dir / "selected_member_analysis_indices.npy",
        np.asarray(selected_indices, dtype=np.int64),
    )
    metadata = {
        "seed": args.seed,
        "native_ensemble_size": native_size,
        "matched_ensemble_size": args.matched_ensemble_size,
        "matched_member_indices": subset_indices.tolist(),
        "subset_seed": args.subset_seed,
        "analysis_cycles": analysis_cycles,
        "selected_analysis_indices": selected_indices,
        "observation_path": str(observation_path),
        "canonical_mean_path": str(canonical_mean_path) if canonical_mean_path else None,
        "canonical_initial_verification_passed": not args.skip_canonical_check,
        "canonical_check_skipped": bool(args.skip_canonical_check),
        "canonical_strict_analysis_count": args.canonical_strict_analysis_count,
        "canonical_first_mean_max_abs_difference": first_mean_difference,
        "canonical_first_mean_rmse_difference": first_mean_rmse_difference,
        "canonical_mean_max_abs_difference": maximum_mean_difference,
        "canonical_mean_max_rmse_difference": maximum_mean_rmse_difference,
        "canonical_mean_tolerance": args.canonical_mean_tolerance,
        "canonical_tolerance_exceedance_count": canonical_tolerance_exceedance_count,
        "canonical_post_verification_differences_are_diagnostic": True,
        "elapsed_seconds": elapsed_seconds,
        "config_cfd_path": str(config_cfd_path),
        "config_letkf_path": str(config_letkf_path),
        "config_srda_path": str(config_srda_path),
        "observation_scenario": args.observation_scenario,
        "observation_scenario_seed": observation_seed,
        "observation_nan_filled_float32_sha256": observation_hash,
        "resolved_letkf_parameters": {name: getattr(cfg_letkf, name) for name in ["num_ens", "cylinder_height", "cylinder_radius", "inflation_factor", "localization_radius", "block_size", "obs_grid_interval_x", "obs_grid_interval_y", "obs_noise_std"]},
        "max_analyses": args.max_analyses,
    }
    (partial_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (partial_dir / "_COMPLETE").write_text("complete\n")
    partial_dir.rename(final_dir)
    print(
        f"wrote {final_dir} analyses={len(analysis_cycles)} "
        f"elapsed={elapsed_seconds:.3f}s canonical_max_abs={maximum_mean_difference:.8g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
