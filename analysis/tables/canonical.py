"""Build the canonical HR-grid tables from completed manuscript runs or metric caches."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.four_dim_srda.utils.sr_da_helper import downsample_uhr_to_hr_pv
from src.four_dim_srda.utils.ssim import MSSIM

HR = (16, 65, 128)
UHR = (32, 129, 256)
CYCLES = np.arange(4, 800, 4)
SEEDS = tuple(range(9990, 10000))
LAYERS = {
    "all_layers": list(range(16)),
    "bottom5": list(range(5)),
    "middle5": [6, 7, 8, 9, 10],
    "top5": list(range(11, 16)),
}
KEYS = ["model", "seed", "cycle_index"]
POINT_KEYS = [*KEYS, "layer_group"]
POINT_FILES = {
    "maer": "point_metrics_hr_grid.csv",
    "mssim_loss": "mssim_metrics_hr_grid.csv",
    "laplacian_rmse": "laplacian_omega_metrics_hr_grid.csv",
}
ONLINE = {
    "physical_rmse": "rmse",
    "physical_spread_rms": "spread",
    "spread_rmse_ratio": "spread_rmse_ratio",
    "physical_crps": "physical_crps",
    "coverage_90": "coverage_90",
}
FOUR_DIM = "Notsu / 4D-SRDA"
CASES = {
    "073A EMA ep2000 NFE20 M30": "BridgeSRDA-D",
    "072 EMA ep1200 NFE20 M30": "BridgeSRDA-I",
    "HR-LETKF M30": "HR-LETKF M30",
    "HR-LETKF M10": "HR-LETKF M10",
    "LR-LETKF M30": "LR-LETKF M30",
    "071B EMA SR-LETKF ep0900 NFE20 M30": "SR-LETKF",
    FOUR_DIM: "4D-SRDA",
}
CHECKPOINTS = {"BridgeSRDA-D": "ep2000", "BridgeSRDA-I": "ep1200", "SR-LETKF": "ep0900"}
WORK = Path("work/experiments")
DATA = Path("data/four_dim_srda/handover/cfd_simulation/qg_model")
SRDA_STEM = "bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04"
SRDA_RESULTS = Path("python/results/four_dim_srda/handover/srda/UNetMaxVitVer01/use_narrow_jet") / SRDA_STEM


def evaluation_spec(model, seed):
    leaf = f"seed{seed:05d}"
    innovation = WORK / "072_conditional_sb_srda_qg/evaluation/narrow_jet"
    if model.startswith("HR-LETKF"):
        members = int(model.rsplit("M", 1)[1])
        return (innovation / f"hr_letkf_m{members}_native" / leaf,
                "letkf_mean_analysis.npy", f"metrics_ne{members}.csv",
                f"rank_histogram_ne{members}.npy", "metadata.json", members)
    if model == "LR-LETKF M30":
        return (innovation / "lr_letkf_native" / leaf / "m30", "mean_analysis.npy",
                "online_metrics.csv", "rank_histogram.npy", "runtime.json", 30)
    if CASES[model] == "SR-LETKF":
        base = WORK / "071b_sb_prior_retrain_qg/evaluation/narrow_jet"
        run = "SBPriorUNet3D_quad_p2_eps02582_continuous_b32_ema999_2gpu"
        return (base / run / "sr_letkf_member_wise" / leaf / "ema_ep0900_nfe20_m30",
                "analysis_mean_analysis.npy", "online_analysis_metrics.csv",
                "rank_histogram.npy", "runtime.json", 30)
    if CASES[model] == "BridgeSRDA-D":
        base = WORK / "073a_denseobs_conditional_sb_srda_qg/evaluation/narrow_jet"
        run = "DenseObsConditionalSBUNet3D_Scratch_quad_p2_eps02582_ogx08_ogy08_mixup_b32_ema999"
        checkpoint = "ep2000"
    else:
        base = innovation
        run = "ConditionalSBUNet3D_Scratch_quad_p2_eps02582_ogx08_ogy08_mixup_b32_ema999"
        checkpoint = "ep1200"
    return (base / run / "conditional_member_wise" / leaf / f"ema_{checkpoint}_nfe20_m30",
            "sb_mean_analysis.npy", "online_metrics.csv", "rank_histogram.npy", "runtime.json", 30)


def hr_field(field, *, deterministic_archive=False):
    array = np.asarray(field, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite PV field")
    # The historical 4D-SRDA verifier uses this projection even on native HR
    # output. At unit pooling ratio it still restores y=0 as exact zero.
    if array.shape == UHR or (deterministic_archive and array.shape == HR):
        return downsample_uhr_to_hr_pv(torch.from_numpy(array), hr_shape=HR).numpy()
    if array.shape != HR:
        raise ValueError(f"Unexpected PV shape {array.shape}")
    return array


def laplacian(field):
    x = np.roll(field, 1, axis=-1) + np.roll(field, -1, axis=-1) - 2.0 * field
    y_prev = np.concatenate([field[..., :1, :], field[..., :-1, :]], axis=-2)
    y_next = np.concatenate([field[..., 1:, :], field[..., -1:, :]], axis=-2)
    y = y_prev + y_next - 2.0 * field
    return x + y


def score_fields(pred, truth, metric, device):
    def normalized(field):
        return torch.from_numpy(np.clip((field + 2.88) / 5.8, 0.0, 1.0))[None, None].to(device)

    truth_image, prediction_image = normalized(truth), normalized(pred)
    layer_ssim = metric(img1=truth_image, img2=prediction_image)[0].cpu().numpy()
    error = np.abs(pred - truth)
    ref = np.abs(truth)
    lap_error = (laplacian(pred) - laplacian(truth)) ** 2
    for group, layers in LAYERS.items():
        yield (
            group,
            float(error[layers].mean() / (ref[layers].mean() + 1e-12)),
            float(1.0 - layer_ssim[layers].mean()),
            float(np.sqrt(lap_error[layers].mean())),
        )


def require_index(frame, keys, expected, metrics):
    actual = pd.MultiIndex.from_frame(frame[keys])
    if not actual.is_unique or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"Unexpected or incomplete support for {metrics}")
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError(f"Nonfinite values in {metrics}")


def with_time(frame):
    return frame.assign(analysis_index=frame.cycle_index // 4 - 1,
                        time_index=frame.cycle_index / 4.0)


def cached_tables(cache, seeds):
    def read(name):
        frame = pd.read_csv(cache / name)
        return frame[frame.model.isin(CASES) & frame.seed.isin(seeds)]

    point = None
    expected = pd.MultiIndex.from_product([CASES, seeds, CYCLES, LAYERS])
    for metric, name in POINT_FILES.items():
        frame = read(name)
        require_index(frame, POINT_KEYS, expected, [metric])
        frame = frame[[*POINT_KEYS, metric]]
        point = frame if point is None else point.merge(frame, on=POINT_KEYS, validate="one_to_one")
    uq = read("uq_metrics.csv")[[*KEYS, *ONLINE.values()]]
    runtime = read("runtime_summary.csv")
    runtime = runtime.assign(seconds_per_analysis=runtime.seconds_per_analysis.fillna(
        runtime.total_wall_time_seconds / len(CYCLES)))
    return point, uq, runtime[["model", "seed", "seconds_per_analysis"]]


@torch.inference_mode()
def score_runs(root, seeds, device):
    metric = MSSIM(window_3d_size=(5, 11, 11), sigma_3d=(0.7, 1.5, 1.5),
                   value_magnitude=1.0, use_gaussian=False).to(device)
    point, uq, runtimes, ranks = [], [], [], []
    deterministic_times = pd.read_csv(root / SRDA_RESULTS / "all_elapsed_time.csv")
    deterministic_times = deterministic_times.set_index("I_SEED_UHR", verify_integrity=True)
    for seed in seeds:
        leaf = f"seed{seed:05d}"
        truth_path = root / DATA / "uhr_pv_narrow_jet" / leaf / f"{leaf}_start000_end800_uhr_pv.npy"
        truth_native = np.load(truth_path, mmap_mode="r", allow_pickle=False)
        if truth_native.shape not in ((800, *UHR), (801, *UHR)):
            raise ValueError(f"{truth_path}: unexpected shape {truth_native.shape}")
        truth = np.stack([hr_field(truth_native[cycle]) for cycle in CYCLES])
        for model in CASES:
            deterministic = model == FOUR_DIM
            if deterministic:
                with np.load(root / SRDA_RESULTS / f"UHR_seed_{seed:05d}.npz", allow_pickle=False) as archive:
                    means = archive["srda_forecast"]
                if means.shape[0] not in (800, 801):
                    raise ValueError(f"{model} seed {seed}: unexpected archive size {means.shape}")
                seconds = deterministic_times.loc[seed, "time_second"] / len(CYCLES)
            else:
                relative, mean_name, online_name, rank_name, metadata_name, members = evaluation_spec(model, seed)
                folder = root / relative
                if not (folder / "_COMPLETE").is_file():
                    raise FileNotFoundError(folder / "_COMPLETE")
                means = np.load(folder / mean_name, mmap_mode="r", allow_pickle=False)
                if means.shape != (len(CYCLES), *HR):
                    raise ValueError(f"{folder}: unexpected mean shape {means.shape}")
                online = pd.read_csv(folder / online_name)
                if not np.array_equal(online.cycle_index.to_numpy(), CYCLES):
                    raise ValueError(f"{folder}: unexpected cycle indices")
                uq.append(online[["cycle_index", *ONLINE]].rename(columns=ONLINE).assign(model=model, seed=seed))
                metadata = json.loads((folder / metadata_name).read_text())
                if metadata["seed"] != seed or metadata.get("native_ensemble_size", metadata.get("ensemble_size")) != members:
                    raise ValueError(f"{folder}: mismatched seed or ensemble size")
                name = CASES[model]
                if name in CHECKPOINTS:
                    expected = (CHECKPOINTS[name], "ema", 20, "bare_heun_exact_noise")
                    actual = tuple(metadata.get(key) for key in
                                   ("checkpoint_label", "weights_kind", "nfe_per_member", "sampler"))
                    if actual != expected:
                        raise ValueError(f"{folder}: unexpected checkpoint or sampler settings {actual}")
                if name == "LR-LETKF M30" and metadata.get("baseline_kind") != "true_lr_letkf":
                    raise ValueError(f"{folder}: unexpected LR baseline")
                seconds = metadata.get("seconds_per_analysis")
                if seconds is None:
                    seconds = metadata["elapsed_seconds"] / len(CYCLES)
                counts = np.load(folder / rank_name, allow_pickle=False).reshape(-1)
                if (counts.shape != (members + 1,) or not np.issubdtype(counts.dtype, np.integer)
                        or (counts < 0).any() or counts.sum() != len(CYCLES) * np.prod(HR)):
                    raise ValueError(f"{folder}: unexpected rank support")
                ranks.extend((model, seed, rank, int(count)) for rank, count in enumerate(counts))
            runtimes.append((model, seed, seconds))
            for idx, cycle in enumerate(CYCLES):
                pred = hr_field(means[cycle if deterministic else idx], deterministic_archive=deterministic)
                point.extend((model, seed, cycle, *row) for row in score_fields(pred, truth[idx], metric, device))
    return (
        pd.DataFrame(point, columns=[*POINT_KEYS, *POINT_FILES]),
        pd.concat(uq, ignore_index=True),
        pd.DataFrame(runtimes, columns=["model", "seed", "seconds_per_analysis"]),
        pd.DataFrame(ranks, columns=["model", "seed", "rank", "count"]),
    )


def write_tables(output, point, uq, runtime, seeds, ranks):
    require_index(point, POINT_KEYS, pd.MultiIndex.from_product([CASES, seeds, CYCLES, LAYERS]), list(POINT_FILES))
    require_index(uq, KEYS, pd.MultiIndex.from_product(
        [[model for model in CASES if model != FOUR_DIM], seeds, CYCLES]), list(ONLINE.values()))
    require_index(runtime, KEYS[:2], pd.MultiIndex.from_product([CASES, seeds]), ["seconds_per_analysis"])
    per_seed = (point.loc[point.layer_group.eq("all_layers")]
                .groupby(["model", "seed"])[list(POINT_FILES)].mean())
    per_seed = per_seed.join(uq.groupby(["model", "seed"])[list(ONLINE.values())].mean())
    per_seed = per_seed.join(runtime.set_index(["model", "seed"]))
    summary = per_seed.groupby("model").mean()
    output.mkdir(parents=True, exist_ok=False)
    for metric, filename in POINT_FILES.items():
        with_time(point[[*POINT_KEYS, metric]]).to_csv(output / filename, index=False)
    with_time(uq).to_csv(output / "uq_metrics.csv", index=False)
    runtime.to_csv(output / "runtime_summary.csv", index=False)
    per_seed.rename(index=CASES, level="model").to_csv(output / "per_seed_summary.csv")
    summary.rename(index=CASES).sort_values("maer").to_csv(output / "canonical_summary.csv")
    if ranks is not None:
        ranks.assign(model=ranks.model.map(CASES)).to_csv(output / "rank_counts.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--root", type=Path, help="Repository or run root with completed evaluator outputs")
    inputs.add_argument("--cache", type=Path, help="Reassemble verified scalar presentation CSVs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="MSSIM device for rebuilding field scores")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=SEEDS)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"Output already exists: {args.output}")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seed list contains duplicates")
    if args.cache:
        tables, ranks = cached_tables(args.cache, args.seeds), None
    else:
        *tables, ranks = score_runs(args.root, args.seeds, args.device)
    write_tables(args.output, *tables, args.seeds, ranks)
    (args.output / "verification.json").write_text(json.dumps({
        "seeds": list(args.seeds), "cycles_per_seed": len(CYCLES), "verification_grid": HR,
        "source": "scalar presentation cache" if args.cache else "evaluator outputs",
        "mssim_device": None if args.cache else args.device,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
