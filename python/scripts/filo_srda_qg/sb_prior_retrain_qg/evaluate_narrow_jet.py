from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import time
import zlib
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.filo_srda_qg.sb_prior_retrain_qg.cycling_schedule import cycling_schedule
from src.filo_srda_qg.shared_srda.metrics import (
    ensemble_summary,
    rank_histogram_counts,
)
from src.filo_srda_qg.shared_srda.narrow_jet import (
    normalized_windows_to_physical,
)
from src.filo_srda_qg.sr_prior.source import build_bridge_source
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.conditional_runtime import (
    get_conditional_runtime,
)
from src.filo_srda_qg.sb_prior_retrain_qg.checkpoint import (
    load_model_from_training_checkpoint,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config as load_sb_config
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import (
    sample_bare_heun_exact_noise,
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
from src.four_dim_srda.utils.sr_da_helper import (
    get_uhr_and_hr_pvs,
    initialize_and_itegrate_lr_cfd_model_for_forecast,
    make_preprocessed_lr_for_forecast,
    make_preprocessed_obs_for_forecast,
)
from src.qg_model.jet_maker import (
    make_jet_pv_with_linear_profile,
    make_jet_pv_with_tanh_profile,
    make_jet_velocity_with_sech_squared_and_sigmoid_profile,
    make_jet_velocity_with_sech_squared_profile,
)
from src.qg_model.low_pass_filter import LowPassFilter
from src.qg_model.qg_model import QGModel


DEFAULT_CFD_CONFIG = (
    "python/configs/four_dim_srda/handover/cfd_simulation/qg_model/"
    "gpu_evaluation_config.yml"
)
DEFAULT_SRDA_CONFIG = (
    "python/configs/four_dim_srda/handover/perform_4D_SRDA/UNetMaxVitVer01/"
    "bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04.yml"
)
DEFAULT_LETKF_CONFIG = (
    "python/configs/four_dim_srda/handover/perform_letkf_hr_using_uhr/"
    "letkf_cfg_ogx08_ogy08_ne100_ch16e-04_cr6e+00_if12e-01_lr57e-01_bs6.yml"
)


def conditional_mode(config: dict[str, Any]) -> bool:
    return bool(config.get("conditioning", {}).get("enabled", False))


def make_mask_from_history(
    *,
    history: list[torch.Tensor],
    assimilation_period: int,
    forecast_span: int,
    interval: int,
) -> torch.Tensor:
    obs = torch.stack(history[-(assimilation_period + 1) :], dim=0)
    dummy = torch.full_like(obs[0], fill_value=torch.nan)
    obs = torch.cat(
        [obs, torch.broadcast_to(dummy, size=(forecast_span,) + dummy.shape)], dim=0
    )
    obs = obs.permute(1, 0, 2, 3, 4).contiguous()
    return torch.isfinite(obs).float()[:, ::interval, None, :, :-1, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--mode", choices=["sr_only", "sr_letkf"], required=True)
    parser.add_argument("--seed", type=int, default=9991)
    parser.add_argument("--ensemble-size", type=int, default=30)
    parser.add_argument("--nfe-budget", type=int, choices=[10, 20, 50], required=True)
    parser.add_argument("--weights-kind", choices=["raw", "ema"], required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-label", default="ep0400")
    parser.add_argument("--member-chunk-size", type=int, default=1)
    parser.add_argument("--brownian-base-steps", type=int, default=600)
    parser.add_argument("--sampling-seed", type=int, default=7042)
    parser.add_argument("--random-seed-multiplier", type=int, default=555)
    parser.add_argument(
        "--selected-analysis-indices",
        nargs="+",
        type=int,
        default=[0, 49, 99, 149, 198],
    )
    parser.add_argument("--config-cfd-path", default=DEFAULT_CFD_CONFIG)
    parser.add_argument("--config-srda-path", default=DEFAULT_SRDA_CONFIG)
    parser.add_argument("--config-letkf-path", default=DEFAULT_LETKF_CONFIG)
    parser.add_argument("--obs-noise-std", type=float, default=0.022)
    parser.add_argument(
        "--observation-scenario",
        default="canonical_ogx08_ogy08",
        choices=[
            "canonical_ogx08_ogy08",
            "sparse_ogx16_ogy16",
        ],
        help="Inference-time observation layout perturbation for conditional 072 evaluation.",
    )
    parser.add_argument(
        "--observation-scenario-seed",
        type=int,
        default=9117,
        help="Deterministic seed used by observation-layout robustness scenarios.",
    )
    parser.add_argument("--save-nowcast-metrics", action="store_true")
    parser.add_argument("--cycling-stride", type=int, choices=[1, 2], default=1,
                        help="2: feed terminal nowcast back and issue every second analysis; direct conditional mode only")
    parser.add_argument("--max-analyses", type=int, default=None, help="Bounded preflight; writes to a separate pilot leaf")
    parser.add_argument(
        "--nowcast-leads",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Future leads scored from the 9-frame SB/SR window.",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve(root: pathlib.Path, path: str | pathlib.Path) -> pathlib.Path:
    value = pathlib.Path(path)
    return value if value.is_absolute() else root / value


def make_initial_hr_pv(cfg: CFDConfig) -> torch.Tensor:
    if cfg.jet_profile == "tanh":
        return make_jet_pv_with_tanh_profile(cfg.hr_base_config)
    if cfg.jet_profile == "linear":
        return make_jet_pv_with_linear_profile(cfg.hr_base_config)
    if cfg.jet_profile == "sech-squared":
        velocity = make_jet_velocity_with_sech_squared_profile(
            cfg.hr_base_config, use_shift_y=False, use_narrow_jet=True
        )[None]
    elif cfg.jet_profile == "sech_squared_and_sigmoid":
        velocity = make_jet_velocity_with_sech_squared_and_sigmoid_profile(
            cfg.hr_base_config, use_shift_y=False, use_narrow_jet=True
        )[None]
    else:
        raise ValueError(f"Unsupported jet profile: {cfg.jet_profile}")
    noise = cfg.hr_base_config.noise_amplitude * torch.randn(
        size=(cfg.hr_base_config.n_batch,) + velocity.shape[-3:],
        device=velocity.device,
        dtype=velocity.dtype,
    )
    model = QGModel(cfg.hr_base_config, show_input_cfg_info=False)
    model.initialize(grid_u=velocity, grid_pv_noise=noise)
    return model.get_grid_pv()


def make_filter(cfg: CFDConfig) -> LowPassFilter:
    return LowPassFilter(
        nx_lr=cfg.lr_base_config.nx,
        ny_lr=cfg.lr_base_config.ny,
        nz_lr=cfg.lr_base_config.nz,
        nx_hr=cfg.hr_base_config.nx,
        ny_hr=cfg.hr_base_config.ny,
        nz_hr=cfg.hr_base_config.nz,
        dtype=torch.complex128
        if cfg.hr_base_config.precision == "double"
        else torch.complex64,
        device=cfg.hr_base_config.device,
    )


def load_letkf_matched_observations(
    *,
    root: pathlib.Path,
    seed: int,
    reference: torch.Tensor,
    srda_config_path: pathlib.Path,
    model_name: str = "UNetMaxVitVer01",
) -> tuple[torch.Tensor, pathlib.Path]:
    path = (
        root
        / "python/results/four_dim_srda/handover/srda"
        / model_name
        / "use_narrow_jet"
        / srda_config_path.stem
        / f"UHR_seed_{seed:05}.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        observations = torch.from_numpy(np.asarray(archive["hr_obs"]))
    observations = observations.to(device=reference.device, dtype=reference.dtype)
    if tuple(observations.shape) != tuple(reference.shape):
        raise ValueError(
            f"LETKF observation shape {tuple(observations.shape)} != truth shape {tuple(reference.shape)}"
        )
    return observations, path


@dataclass(frozen=True)
class ObservationScenario:
    name: str
    obs_grid_interval_x: int | None = None
    obs_grid_interval_y: int | None = None
    init_index_x: int = 0
    init_index_y: int = 0
    dropout_rate: float = 0.0
    slot_policy: str = "all_available"
    description: str = ""


def observation_scenario_spec(
    name: str, *, canonical_x: int, canonical_y: int
) -> ObservationScenario:
    if name == "canonical_ogx08_ogy08":
        return ObservationScenario(
            name=name,
            obs_grid_interval_x=canonical_x,
            obs_grid_interval_y=canonical_y,
            description="Canonical fixed Notsu/LETKF-matched ogx08/ogy08 observations.",
        )
    if name == "sparse_ogx16_ogy16":
        return ObservationScenario(
            name=name,
            obs_grid_interval_x=16,
            obs_grid_interval_y=16,
            description="Synthetic sparse observations at ogx16/ogy16.",
        )
    raise ValueError(f"Unsupported observation scenario: {name}")


def deterministic_scenario_seed(seed: int, scenario: str, base_seed: int) -> int:
    scenario_hash = zlib.crc32(scenario.encode("utf-8")) & 0xFFFFFFFF
    return int((int(base_seed) + 1009 * int(seed) + scenario_hash) % (2**31 - 1))


def make_grid_observation_mask(
    *,
    shape: tuple[int, int, int],
    interval_x: int,
    interval_y: int,
    init_index_x: int = 0,
    init_index_y: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    nz, ny, nx = shape
    mask = torch.zeros((nz, ny, nx), device=device, dtype=dtype)
    mask[:, int(init_index_y) :: int(interval_y), int(init_index_x) :: int(interval_x)] = 1.0
    return mask


def synthetic_observations_from_truth(
    *,
    truth: torch.Tensor,
    scenario: ObservationScenario,
    obs_noise_std: float,
    seed: int,
) -> torch.Tensor:
    if scenario.obs_grid_interval_x is None or scenario.obs_grid_interval_y is None:
        raise ValueError(f"Scenario {scenario.name} does not define a grid interval")
    mask = make_grid_observation_mask(
        shape=tuple(truth.shape[-3:]),
        interval_x=int(scenario.obs_grid_interval_x),
        interval_y=int(scenario.obs_grid_interval_y),
        init_index_x=int(scenario.init_index_x),
        init_index_y=int(scenario.init_index_y),
        device=truth.device,
        dtype=truth.dtype,
    )
    generator = torch.Generator(device=truth.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(
        truth.shape, device=truth.device, dtype=truth.dtype, generator=generator
    ) * float(obs_noise_std)
    observed = truth + noise
    missing = torch.full_like(truth, float("nan"))
    return torch.where(mask[None] > 0, observed, missing)


def apply_observation_scenario_to_raw(
    *,
    observations: torch.Tensor,
    hr_pvs: torch.Tensor,
    scenario: ObservationScenario,
    obs_noise_std: float,
    seed: int,
) -> torch.Tensor:
    if scenario.name == "sparse_ogx16_ogy16":
        return synthetic_observations_from_truth(
            truth=hr_pvs,
            scenario=scenario,
            obs_noise_std=obs_noise_std,
            seed=seed,
        )
    return observations


def prepare_letkf(
    *,
    cfg_cfd: CFDConfig,
    cfg_letkf: LETKFBlockProcessingConfig,
    observations: torch.Tensor,
) -> dict[str, Any]:
    model = QGModel(cfg_cfd.hr_base_config, show_input_cfg_info=False)
    x_coords, y_coords, z_coords = model.get_grids()
    flattened_x = x_coords.reshape(-1)
    flattened_y = y_coords.reshape(-1)
    flattened_z = z_coords.reshape(-1)
    block_idx, center_idx = make_splitted_grid_and_its_center_indices_array(cfg_letkf)
    obs_matrix = get_obs_matrix(
        obs=observations[cfg_cfd.da_config.assimilation_interval],
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


def save_compatibility_npz(
    *,
    output_dir: pathlib.Path,
    seed: int,
    mean_key: str,
    mean_values: np.ndarray,
    spread_key: str,
    spread_values: np.ndarray,
    selected_members: np.ndarray,
    selected_indices: np.ndarray,
    rank_histogram: np.ndarray,
    metrics: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    np.savez_compressed(
        output_dir / f"UHR_seed_{seed:05}.npz",
        **{
            mean_key: mean_values.astype(np.float32, copy=False),
            spread_key: spread_values.astype(np.float32, copy=False),
            "sb_mean_analysis": mean_values.astype(np.float32, copy=False),
            "sb_spread_analysis": spread_values.astype(np.float32, copy=False),
            "selected_members": selected_members.astype(np.float32, copy=False),
            "selected_member_analysis_indices": selected_indices.astype(np.int64, copy=False),
            "rank_histogram": rank_histogram.astype(np.int64, copy=False),
            "online_metrics": np.asarray(metrics, dtype=object),
            "metadata": np.asarray(metadata, dtype=object),
        },
    )


def write_outputs(
    *,
    temporary: pathlib.Path,
    seed: int,
    mode: str,
    mean_analysis: torch.Tensor,
    spread_analysis: torch.Tensor,
    selected_members: torch.Tensor | None,
    selected_indices: list[int],
    rank_histogram: torch.Tensor,
    online_metrics: list[dict[str, Any]],
    runtime: dict[str, Any],
    forecast_mean: torch.Tensor | None = None,
    forecast_spread: torch.Tensor | None = None,
    forecast_metrics: list[dict[str, Any]] | None = None,
    nowcast_mean: torch.Tensor | None = None,
    nowcast_spread: torch.Tensor | None = None,
    nowcast_rank_histograms: torch.Tensor | None = None,
    nowcast_metrics: list[dict[str, Any]] | None = None,
    runtime_nowcast: dict[str, Any] | None = None,
    observation_scenario: dict[str, Any] | None = None,
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

    if mode == "sr_letkf":
        if forecast_mean is None or forecast_spread is None or forecast_metrics is None:
            raise ValueError("SR-LETKF output requires forecast diagnostics")
        forecast_mean_np = forecast_mean.cpu().numpy().astype(np.float32)
        forecast_spread_np = forecast_spread.cpu().numpy().astype(np.float32)
        np.save(temporary / "forecast_mean_analysis.npy", forecast_mean_np)
        np.save(temporary / "forecast_spread_analysis.npy", forecast_spread_np)
        np.save(temporary / "analysis_mean_analysis.npy", mean_np)
        np.save(temporary / "analysis_spread_analysis.npy", spread_np)
        pd.DataFrame(forecast_metrics).to_csv(
            temporary / "online_forecast_metrics.csv", index=False
        )
        pd.DataFrame(online_metrics).to_csv(
            temporary / "online_analysis_metrics.csv", index=False
        )
        compatibility_mean_key = "analysis_mean_analysis"
        compatibility_spread_key = "analysis_spread_analysis"
    else:
        compatibility_mean_key = "sb_mean_analysis"
        compatibility_spread_key = "sb_spread_analysis"

    np.save(temporary / "sb_mean_analysis.npy", mean_np)
    np.save(temporary / "sb_spread_analysis.npy", spread_np)
    np.save(temporary / "selected_members.npy", selected_np)
    np.save(temporary / "selected_member_analysis_indices.npy", selected_indices_np)
    np.save(temporary / "rank_histogram.npy", rank_np)
    pd.DataFrame(online_metrics).to_csv(temporary / "online_metrics.csv", index=False)
    if nowcast_mean is not None:
        np.save(
            temporary / "nowcast_mean_leads.npy",
            nowcast_mean.cpu().numpy().astype(np.float32),
        )
    if nowcast_spread is not None:
        np.save(
            temporary / "nowcast_spread_leads.npy",
            nowcast_spread.cpu().numpy().astype(np.float32),
        )
    if nowcast_rank_histograms is not None:
        np.save(
            temporary / "nowcast_rank_histograms.npy",
            nowcast_rank_histograms.cpu().numpy().astype(np.int64),
        )
    if nowcast_metrics is not None:
        pd.DataFrame(nowcast_metrics).to_csv(
            temporary / "online_nowcast_metrics.csv", index=False
        )
    if runtime_nowcast is not None:
        (temporary / "runtime_nowcast.json").write_text(
            json.dumps(runtime_nowcast, indent=2, default=str)
        )
    if observation_scenario is not None:
        (temporary / "observation_scenario.json").write_text(
            json.dumps(observation_scenario, indent=2, default=str)
        )
    (temporary / "runtime.json").write_text(json.dumps(runtime, indent=2, default=str))
    save_compatibility_npz(
        output_dir=temporary,
        seed=seed,
        mean_key=compatibility_mean_key,
        mean_values=mean_np,
        spread_key=compatibility_spread_key,
        spread_values=spread_np,
        selected_members=selected_np,
        selected_indices=selected_indices_np,
        rank_histogram=rank_np,
        metrics=online_metrics,
        metadata=runtime,
    )
    (temporary / "_COMPLETE").write_text("complete\n")


def run_cycle(args: argparse.Namespace) -> int:
    root = pathlib.Path(args.root_dir).resolve()
    config = load_sb_config(resolve(root, args.config_path))
    conditional_runtime = get_conditional_runtime(config)
    stride = int(getattr(args, "cycling_stride", 1))
    max_analyses = getattr(args, "max_analyses", None)
    if stride == 2 and (conditional_runtime is None or args.mode != "sr_only"):
        raise ValueError("Terminal-nowcast cycling is restricted to direct conditional SB-SRDA")
    if max_analyses is not None and max_analyses < 1:
        raise ValueError("max_analyses must be positive")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_srda = load_config(
        model_name="UNetMaxVitVer01",
        config_path=str(resolve(root, args.config_srda_path)),
    )
    cfg_base = CFDConfig.load(resolve(root, args.config_cfd_path))
    cfg = copy.deepcopy(cfg_base)
    cfg.lr_base_config.n_batch = int(args.ensemble_size)
    cfg.hr_base_config.n_batch = int(args.ensemble_size)
    cfg.lr_base_config.device = str(device)
    cfg.hr_base_config.device = str(device)
    cfg.uhr_base_config.device = str(device)

    if conditional_runtime is not None:
        model, checkpoint_metadata = conditional_runtime.load_training_checkpoint(
            config=config,
            checkpoint_path=resolve(root, args.checkpoint_path),
            device=device,
            weights_kind=args.weights_kind,
        )
    else:
        model, checkpoint_metadata = load_model_from_training_checkpoint(
            config=config,
            checkpoint_path=resolve(root, args.checkpoint_path),
            device=device,
            weights_kind=args.weights_kind,
        )
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(**config["bridge"])
    )

    seed = int(args.seed)
    ensemble = int(args.ensemble_size)
    assimilation_period = cfg.da_config.segment_length - cfg.da_config.forecast_span - 1
    forecast_span = cfg.da_config.forecast_span
    num_times = cfg_srda.dataset_config.max_start_time_index + assimilation_period + forecast_span
    output_scale_factor = int(cfg.time_config.output_lr_dt // cfg.time_config.output_hr_dt)
    expected_indices, feedback_frame = cycling_schedule(
        num_times=num_times, assimilation_period=assimilation_period,
        forecast_span=forecast_span, stride=stride,
    )
    if max_analyses is not None:
        expected_indices = expected_indices[:max_analyses]
    scheduled_centers = set(expected_indices)
    selected_lookup = set(int(value) for value in args.selected_analysis_indices)
    dataset_view = SimpleNamespace(cfg=cfg_srda.dataset_config)
    low_pass_filter = make_filter(cfg)
    uhr_root = root / "data/four_dim_srda/handover/cfd_simulation/qg_model/uhr_pv_narrow_jet"

    output_root = (
        root
        / str(config["output"]["work_root"])
    )
    scenario = observation_scenario_spec(
        args.observation_scenario,
        canonical_x=int(config.get("dataset", {}).get("obs_grid_interval_x", 8)),
        canonical_y=int(config.get("dataset", {}).get("obs_grid_interval_y", 8)),
    )
    write_robustness_output = conditional_mode(config) and scenario.name != "canonical_ogx08_ogy08"
    if write_robustness_output:
        output_root = (
            output_root
            / "evaluation/narrow_jet_observation_robustness"
            / str(config["output"]["run_name"])
            / scenario.name
        )
    else:
        output_root = output_root / "evaluation/narrow_jet" / str(config["output"]["run_name"])
    leaf = f"{args.weights_kind}_{args.checkpoint_label}_nfe{args.nfe_budget}_m{ensemble}"
    if stride != 1:
        leaf += f"_stride{stride}"
    if max_analyses is not None:
        leaf += f"_pilot{max_analyses}"
    if conditional_mode(config):
        mode_dir = "conditional_member_wise"
    else:
        mode_dir = "prior_member_wise" if args.mode == "sr_only" else "sr_letkf_member_wise"
    final_dir = output_root / mode_dir / f"seed{seed:05d}" / leaf
    temporary = final_dir.with_name(final_dir.name + ".partial")
    if final_dir.exists():
        complete_for_request = (final_dir / "_COMPLETE").exists() and (
            not args.save_nowcast_metrics
            or (
                (final_dir / "online_nowcast_metrics.csv").exists()
                and (final_dir / "nowcast_mean_leads.npy").exists()
                and (final_dir / "nowcast_spread_leads.npy").exists()
            )
        )
        if args.skip_existing and complete_for_request:
            print(f"skip completed output: {final_dir}")
            return 0
        if not args.allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {final_dir}")
        shutil.rmtree(final_dir)
    if temporary.exists():
        if not args.allow_overwrite:
            raise FileExistsError(
                f"Stale partial output exists: {temporary}; use --allow-overwrite"
            )
        shutil.rmtree(temporary)

    set_seeds(args.random_seed_multiplier * seed, use_deterministic=True)
    initial = make_initial_hr_pv(copy.deepcopy(cfg_base)).to(device=device).float()
    state = initial.repeat(ensemble, 1, 1, 1)
    _, hr_pvs = get_uhr_and_hr_pvs(
        result_dir_path=str(uhr_root / f"seed{seed:05}"),
        i_seed_uhr=seed,
        num_times=num_times,
        cfg=cfg_base,
    )
    hr_pvs = hr_pvs.to(device=device).float()

    observations, observation_source_path = load_letkf_matched_observations(
        root=root,
        seed=seed,
        reference=hr_pvs,
        srda_config_path=resolve(root, args.config_srda_path),
    )
    scenario_seed = deterministic_scenario_seed(
        seed=seed, scenario=scenario.name, base_seed=int(args.observation_scenario_seed)
    )
    observations = apply_observation_scenario_to_raw(
        observations=observations,
        hr_pvs=hr_pvs,
        scenario=scenario,
        obs_noise_std=float(args.obs_noise_std),
        seed=scenario_seed,
    )

    letkf_state: dict[str, Any] | None = None
    cfg_letkf: LETKFBlockProcessingConfig | None = None
    if args.mode == "sr_letkf":
        cfg_letkf = LETKFBlockProcessingConfig.load(resolve(root, args.config_letkf_path))
        cfg_letkf.num_ens = ensemble
        cfg_letkf.data_dir = str(root / "data/four_dim_srda/handover/cfd_simulation/qg_model")
        letkf_state = prepare_letkf(
            cfg_cfd=cfg,
            cfg_letkf=cfg_letkf,
            observations=observations,
        )

    mean_analysis: list[torch.Tensor] = []
    spread_analysis: list[torch.Tensor] = []
    forecast_mean_analysis: list[torch.Tensor] = []
    forecast_spread_analysis: list[torch.Tensor] = []
    selected_members: list[torch.Tensor] = []
    selected_member_indices: list[int] = []
    online_metrics: list[dict[str, Any]] = []
    online_forecast_metrics: list[dict[str, Any]] = []
    rank_counts = torch.zeros(ensemble + 1, dtype=torch.long)
    nowcast_means_by_analysis: list[torch.Tensor] = []
    nowcast_spreads_by_analysis: list[torch.Tensor] = []
    nowcast_metrics: list[dict[str, Any]] = []
    nowcast_rank_counts = torch.zeros(
        len(args.nowcast_leads), ensemble + 1, dtype=torch.long
    )
    sb_sampling_seconds = 0.0
    letkf_analysis_seconds = 0.0
    total_drift_nfe = 0
    peak_gpu_bytes = 0
    started = time.perf_counter()
    observation_history: list[torch.Tensor] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    description = (
        conditional_runtime.description
        if conditional_runtime is not None
        else f"071B {args.mode}"
    )
    loop_stop = expected_indices[-1] + 1 if max_analyses is not None else num_times
    for cycle_index in tqdm(range(loop_stop), desc=f"{description} seed {seed}"):
        observed = observations[cycle_index]
        if cycle_index % assimilation_period == 0:
            observation_history.append(observed[None].repeat(ensemble, 1, 1, 1))
        else:
            observation_history.append(
                torch.full(
                    (ensemble,) + tuple(observed.shape),
                    float("nan"),
                    device=observed.device,
                    dtype=observed.dtype,
                )
            )
        if cycle_index not in scheduled_centers:
            continue
        integrate_steps = (assimilation_period + forecast_span) // output_scale_factor
        lr_forecast = initialize_and_itegrate_lr_cfd_model_for_forecast(
            last_hr_pv0=state,
            num_integrate_steps=integrate_steps,
            low_pass_filter=low_pass_filter,
            cfg_cfd=cfg,
        )
        lr = make_preprocessed_lr_for_forecast(
            lr_forecast=lr_forecast, test_dataset=dataset_view, cfg_cfd=cfg
        )
        source = build_bridge_source(lr.to(device=device).float())
        before_sampling = time.perf_counter()
        if conditional_runtime is not None:
            processed_observations = make_preprocessed_obs_for_forecast(
                hr_obs=observation_history,
                assimilation_period=assimilation_period,
                forecast_span=forecast_span,
                test_dataset=dataset_view,
                cfg_cfd=cfg,
            ).to(device=device, dtype=source.dtype)
            observation_masks = make_mask_from_history(
                history=observation_history,
                assimilation_period=assimilation_period,
                forecast_span=forecast_span,
                interval=int(dataset_view.cfg.lr_and_obs_time_interval),
            ).to(device=device, dtype=source.dtype)
            result = conditional_runtime.sample(
                model=model, source=source,
                observations=processed_observations, masks=observation_masks,
                bridge=bridge, ensemble_size=1,
                nfe_budget=int(args.nfe_budget),
                member_chunk_size=int(args.member_chunk_size),
                sampling_seed=int(args.sampling_seed) + seed * 1000 + cycle_index,
                brownian_base_steps=int(args.brownian_base_steps),
            )
        else:
            result = sample_bare_heun_exact_noise(
                model=model,
                source=source,
                bridge=bridge,
                ensemble_size=1,
                nfe_budget=int(args.nfe_budget),
                member_chunk_size=int(args.member_chunk_size),
                sampling_seed=int(args.sampling_seed) + seed * 1000 + cycle_index,
                brownian_base_steps=int(args.brownian_base_steps),
            )
        sb_sampling_seconds += time.perf_counter() - before_sampling
        total_drift_nfe += int(result.actual_nfe) * ensemble
        peak_gpu_bytes = max(peak_gpu_bytes, int(result.peak_gpu_bytes))
        normalized = result.samples[:, 0]
        physical_windows = normalized_windows_to_physical(
            normalized,
            pv_min=float(cfg_srda.dataset_config.pv_min),
            pv_max=float(cfg_srda.dataset_config.pv_max),
        )
        forecast_members = physical_windows[:, assimilation_period].float()
        analysis_number = len(mean_analysis)
        truth = hr_pvs[cycle_index].float()
        forecast_metrics = ensemble_summary(
            forecast_members[None].float(), truth[None], pv_min=0.0, pv_max=1.0
        )
        online_forecast_metrics.append(
            {
                "analysis_number": analysis_number,
                "cycle_index": cycle_index,
                **forecast_metrics,
            }
        )
        forecast_mean_analysis.append(forecast_members.mean(dim=0))
        forecast_spread_analysis.append(forecast_members.std(dim=0, unbiased=True))

        if args.mode == "sr_letkf":
            assert letkf_state is not None and cfg_letkf is not None
            before_letkf = time.perf_counter()
            # The canonical LETKF geometry/observation operators are float64,
            # while SB inference intentionally runs in float32.  LETKF uses
            # matrix products between the operator, observations, and forecast
            # state, so all three must enter assimilation with one dtype.
            letkf_dtype = letkf_state["obs_matrix"].dtype
            analysis_members = hr_assimilate_with_existing_data(
                hr_pv=forecast_members.to(device=device, dtype=letkf_dtype),
                gt_added_noise=torch.nan_to_num(
                    observations[cycle_index].to(device=device, dtype=letkf_dtype),
                    nan=1.0e10,
                ),
                obs_matrix=letkf_state["obs_matrix"].to(device=device, dtype=letkf_dtype),
                dist_xy=letkf_state["dist_xy"].to(device=device, dtype=letkf_dtype),
                dist_z=letkf_state["dist_z"].to(device=device, dtype=letkf_dtype),
                decay_factors=letkf_state["decay_factors"].to(
                    device=device, dtype=letkf_dtype
                ),
                block_idx=letkf_state["block_idx"],
                center_idx=letkf_state["center_idx"],
                cfg=cfg_letkf,
            ).float()
            letkf_analysis_seconds += time.perf_counter() - before_letkf
        else:
            analysis_members = forecast_members

        if args.save_nowcast_metrics:
            lead_means: list[torch.Tensor] = []
            lead_spreads: list[torch.Tensor] = []
            for lead_pos, lead in enumerate(args.nowcast_leads):
                frame_index = assimilation_period + int(lead)
                target_index = cycle_index + int(lead)
                if (
                    frame_index < physical_windows.shape[1]
                    and 0 <= target_index < hr_pvs.shape[0]
                ):
                    lead_members = (
                        analysis_members
                        if int(lead) == 0
                        else physical_windows[:, frame_index].float()
                    )
                    lead_truth = hr_pvs[target_index].float()
                    lead_summary = ensemble_summary(
                        lead_members[None].float(),
                        lead_truth[None],
                        pv_min=0.0,
                        pv_max=1.0,
                    )
                    nowcast_metrics.append(
                        {
                            "analysis_number": analysis_number,
                            "cycle_index": cycle_index,
                            "lead": int(lead),
                            "target_cycle_index": int(target_index),
                            **lead_summary,
                        }
                    )
                    nowcast_rank_counts[lead_pos] += rank_histogram_counts(
                        lead_members[None].float(),
                        lead_truth.to(device=lead_members.device)[None].float(),
                    ).cpu()
                    lead_means.append(lead_members.mean(dim=0))
                    lead_spreads.append(lead_members.std(dim=0, unbiased=True))
                else:
                    nan_field = torch.full_like(hr_pvs[0], float("nan"))
                    lead_means.append(nan_field)
                    lead_spreads.append(nan_field)
            nowcast_means_by_analysis.append(torch.stack(lead_means, dim=0))
            nowcast_spreads_by_analysis.append(torch.stack(lead_spreads, dim=0))

        state = (analysis_members if stride == 1 else physical_windows[:, feedback_frame].float()).clone()
        if not torch.isfinite(state).all():
            raise FloatingPointError("Nonfinite cycling feedback")
        analysis_metrics = ensemble_summary(
            analysis_members[None].float(), truth[None], pv_min=0.0, pv_max=1.0
        )
        online_metrics.append(
            {
                "analysis_number": analysis_number,
                "cycle_index": cycle_index,
                **analysis_metrics,
            }
        )
        rank_counts += rank_histogram_counts(
            analysis_members[None].float(),
            truth.to(device=analysis_members.device)[None].float(),
        ).cpu()
        mean_analysis.append(analysis_members.mean(dim=0))
        spread_analysis.append(analysis_members.std(dim=0, unbiased=True))
        if analysis_number in selected_lookup:
            selected_member_indices.append(analysis_number)
            selected_members.append(analysis_members)

    elapsed = time.perf_counter() - started
    if len(mean_analysis) != len(expected_indices):
        raise RuntimeError(
            f"Saved {len(mean_analysis)} analyses, expected {len(expected_indices)}"
        )
    mean_tensor = torch.stack(mean_analysis)
    spread_tensor = torch.stack(spread_analysis)
    selected_tensor = torch.stack(selected_members) if selected_members else None
    forecast_mean_tensor = torch.stack(forecast_mean_analysis)
    forecast_spread_tensor = torch.stack(forecast_spread_analysis)
    peak_gpu = max(
        peak_gpu_bytes,
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    )
    runtime = {
        "seed": seed,
        "mode": args.mode,
        "conditional_observation_model": bool(conditional_runtime is not None),
        "conditioning_kind": (
            conditional_runtime.kind if conditional_runtime is not None else None
        ),
        "conditioning_contract": (
            copy.deepcopy(config.get("conditioning", {}))
            if conditional_mode(config)
            else None
        ),
        "observation_noise_std": (
            float(config["dataset"]["obs_noise_std"])
            if conditional_mode(config)
            else None
        ),
        "observation_grid_interval": (
            [
                int(config["dataset"]["obs_grid_interval_x"]),
                int(config["dataset"]["obs_grid_interval_y"]),
            ]
            if conditional_mode(config)
            else None
        ),
        "total_wall_time_seconds": elapsed,
        "sb_sampling_seconds": sb_sampling_seconds,
        "letkf_analysis_seconds": letkf_analysis_seconds if args.mode == "sr_letkf" else None,
        "seconds_per_analysis": elapsed / float(len(mean_analysis)),
        "nfe_per_member": int(args.nfe_budget),
        "total_drift_nfe": int(total_drift_nfe),
        "ensemble_size": ensemble,
        "analysis_indices": expected_indices,
        "cycling_stride": stride,
        "analysis_interval_physical": stride * assimilation_period * float(cfg.time_config.output_hr_dt),
        "feedback_frame": feedback_frame,
        "feedback_policy": "center_analysis" if stride == 1 else "terminal_nowcast",
        "window_anchor_policy": "previous feedback time, plus 1 and 2 physical units; window length unchanged",
        "nominal_verified_horizon_physical": expected_indices[-1] * float(cfg.time_config.output_hr_dt),
        "seconds_per_verified_physical_unit": elapsed / (expected_indices[-1] * float(cfg.time_config.output_hr_dt)),
        "max_analyses": max_analyses,
        "selected_analysis_indices": selected_member_indices,
        "peak_gpu_bytes": peak_gpu,
        "checkpoint_provenance": checkpoint_metadata,
        "checkpoint_label": args.checkpoint_label,
        "weights_kind": args.weights_kind,
        "sampler": "bare_heun_exact_noise",
        "brownian_base_steps": int(args.brownian_base_steps),
        "sampling_seed": int(args.sampling_seed),
        "observation_source_path": str(observation_source_path),
        "observation_scenario": asdict(scenario),
        "observation_scenario_seed": int(scenario_seed),
        "config_path": str(resolve(root, args.config_path)),
        "config_cfd_path": str(resolve(root, args.config_cfd_path)),
        "config_srda_path": str(resolve(root, args.config_srda_path)),
        "config_letkf_path": str(resolve(root, args.config_letkf_path))
        if args.mode == "sr_letkf"
        else None,
    }
    nowcast_mean_tensor = (
        torch.stack(nowcast_means_by_analysis)
        if args.save_nowcast_metrics
        else None
    )
    nowcast_spread_tensor = (
        torch.stack(nowcast_spreads_by_analysis)
        if args.save_nowcast_metrics
        else None
    )
    runtime_nowcast = (
        {
            "seed": seed,
            "mode": args.mode,
            "leads": [int(lead) for lead in args.nowcast_leads],
            "lead_frame_mapping": {
                int(lead): int(assimilation_period + int(lead))
                for lead in args.nowcast_leads
            },
            "num_metric_rows": len(nowcast_metrics),
            "array_shape": list(nowcast_mean_tensor.shape)
            if nowcast_mean_tensor is not None
            else None,
            "lead0_should_match_online_metrics": True,
        }
        if args.save_nowcast_metrics
        else None
    )
    write_outputs(
        temporary=temporary,
        seed=seed,
        mode=args.mode,
        mean_analysis=mean_tensor,
        spread_analysis=spread_tensor,
        selected_members=selected_tensor,
        selected_indices=selected_member_indices,
        rank_histogram=rank_counts,
        online_metrics=online_metrics,
        runtime=runtime,
        forecast_mean=forecast_mean_tensor if args.mode == "sr_letkf" else None,
        forecast_spread=forecast_spread_tensor if args.mode == "sr_letkf" else None,
        forecast_metrics=online_forecast_metrics if args.mode == "sr_letkf" else None,
        nowcast_mean=nowcast_mean_tensor,
        nowcast_spread=nowcast_spread_tensor,
        nowcast_rank_histograms=nowcast_rank_counts
        if args.save_nowcast_metrics
        else None,
        nowcast_metrics=nowcast_metrics if args.save_nowcast_metrics else None,
        runtime_nowcast=runtime_nowcast,
        observation_scenario=asdict(scenario) if write_robustness_output else None,
    )
    temporary.rename(final_dir)
    print(f"wrote {final_dir} analyses={len(mean_analysis)} elapsed={elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cycle(parse_args()))
