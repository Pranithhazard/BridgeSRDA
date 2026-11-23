from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.four_dim_srda.utils.sr_da_helper import downsample_uhr_to_hr_pv
from src.four_dim_srda.utils.ssim import MSSIM, _ssim_3D


PV_MIN = -2.88
PV_MAX = 2.92
PV_RANGE = PV_MAX - PV_MIN
RUN_NAME = (
    "DenseObsConditionalSBUNet3D_Scratch_quad_p2_eps02582_"
    "ogx08_ogy08_mixup_b32_ema999"
)
CASES = {
    "NFE20/M30": "ema_ep2000_nfe20_m30",
    "NFE20/M10": "ema_ep2000_nfe20_m10",
    "NFE10/M30": "ema_ep2000_nfe10_m30",
    "NFE10/M10": "ema_ep2000_nfe10_m10",
}
COLORS = {
    "NFE20/M30": "#C51B7D",
    "NFE20/M10": "#7A0177",
    "NFE10/M30": "#2C7FB8",
    "NFE10/M10": "#41B6C4",
}
ONLINE_METRICS = (
    "physical_mae",
    "physical_rmse",
    "physical_crps",
    "physical_spread_rms",
    "spread_rmse_ratio",
    "coverage_50",
    "coverage_80",
    "coverage_90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the completed 073A reduced-cost narrow-jet runs."
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument(
        "--evaluation-root",
        help="Optional stable snapshot of the conditional_member_wise directory.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(9990, 10000)))
    parser.add_argument("--panel-seed", type=int, default=9991)
    parser.add_argument("--panel-cycle", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def evaluation_root(root: Path) -> Path:
    return (
        root
        / "work/experiments/073a_denseobs_conditional_sb_srda_qg/evaluation/"
        "narrow_jet"
        / RUN_NAME
        / "conditional_member_wise"
    )


def truth_path(root: Path, seed: int) -> Path:
    return (
        root
        / "data/four_dim_srda/handover/cfd_simulation/qg_model/"
        "uhr_pv_narrow_jet"
        / f"seed{seed:05d}"
        / f"seed{seed:05d}_start000_end800_uhr_pv.npy"
    )


def leaf_path(
    root: Path, seed: int, leaf: str, override: Path | None = None
) -> Path:
    path = (override or evaluation_root(root)) / f"seed{seed:05d}" / leaf
    if not (path / "_COMPLETE").exists():
        raise FileNotFoundError(f"Incomplete or missing evaluation leaf: {path}")
    return path


def hr_reference(uhr: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    value = torch.from_numpy(np.array(uhr, dtype=np.float32, copy=True))
    return (
        downsample_uhr_to_hr_pv(value, hr_shape=shape)
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


def horizontal_laplacian(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    x_term = np.roll(value, 1, axis=-1) + np.roll(value, -1, axis=-1) - 2 * value
    y_previous = np.concatenate([value[..., :1, :], value[..., :-1, :]], axis=-2)
    y_following = np.concatenate([value[..., 1:, :], value[..., -1:, :]], axis=-2)
    return x_term + y_previous + y_following - 2 * value


def load_case(
    root: Path, seed: int, leaf: str, evaluation_override: Path | None = None
) -> tuple[Path, np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    path = leaf_path(root, seed, leaf, evaluation_override)
    mean = np.load(path / "sb_mean_analysis.npy", mmap_mode="r")
    spread = np.load(path / "sb_spread_analysis.npy", mmap_mode="r")
    online = pd.read_csv(path / "online_metrics.csv")
    runtime = json.loads((path / "runtime.json").read_text())
    if len(mean) != 199 or len(spread) != 199 or len(online) != 199:
        raise RuntimeError(
            f"Expected 199 analyses in {path}; got mean={len(mean)}, "
            f"spread={len(spread)}, metrics={len(online)}"
        )
    return path, mean, spread, online, runtime


def compute_spatial_metrics(
    *,
    root: Path,
    seed: int,
    mean: np.ndarray,
    online: pd.DataFrame,
    mssim: MSSIM,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    truth = np.load(truth_path(root, seed), mmap_mode="r")
    rows: list[dict[str, float | int]] = []
    for start in range(0, len(mean), batch_size):
        stop = min(start + batch_size, len(mean))
        predictions = np.asarray(mean[start:stop], dtype=np.float32)
        references = np.stack(
            [
                hr_reference(
                    np.asarray(truth[int(online.iloc[index].cycle_index)]),
                    tuple(predictions[index - start].shape),
                )
                for index in range(start, stop)
            ]
        )
        difference = predictions - references
        lap_difference = horizontal_laplacian(predictions) - horizontal_laplacian(
            references
        )
        prediction_normalized = np.clip((predictions - PV_MIN) / PV_RANGE, 0, 1)
        reference_normalized = np.clip((references - PV_MIN) / PV_RANGE, 0, 1)
        prediction_tensor = torch.from_numpy(prediction_normalized)[:, None].to(device)
        reference_tensor = torch.from_numpy(reference_normalized)[:, None].to(device)
        mssim._set_window(reference_image=reference_tensor)
        with torch.no_grad():
            ssim_map = _ssim_3D(
                img1=reference_tensor,
                img2=prediction_tensor,
                window=mssim.window,
                window_3d_size=mssim.window_3d_size,
                channel=1,
                value_magnitude=mssim.value_magnitude,
            )
            mssim_losses = 1.0 - ssim_map.mean(dim=(1, 2, 3, 4))
        for offset, index in enumerate(range(start, stop)):
            rows.append(
                {
                    "seed": seed,
                    "analysis_index": index,
                    "cycle_index": int(online.iloc[index].cycle_index),
                    "hr_maer": float(
                        np.mean(np.abs(difference[offset]))
                        / max(float(np.mean(np.abs(references[offset]))), 1.0e-12)
                    ),
                    "laplacian_pv_rmse": float(
                        np.sqrt(np.mean(lap_difference[offset] ** 2))
                    ),
                    "mssim_loss": float(mssim_losses[offset].item()),
                }
            )
    frame = pd.DataFrame(rows)
    return frame, float(frame.mssim_loss.mean())


def plot_bars(summary: pd.DataFrame, metrics: list[tuple[str, str]], output: Path) -> None:
    cases = list(CASES)
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.3))
    axes = np.atleast_1d(axes)
    for axis, (column, title) in zip(axes, metrics, strict=True):
        values = [float(summary.loc[case, column]) for case in cases]
        bars = axis.bar(cases, values, color=[COLORS[case] for case in cases])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.4g", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cost_skill(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for case in CASES:
        row = summary.loc[case]
        for axis, metric, label in (
            (axes[0], "hr_maer", "HR-grid MAER"),
            (axes[1], "physical_crps", "Physical CRPS"),
        ):
            axis.scatter(
                row.seconds_per_analysis,
                row[metric],
                s=85,
                color=COLORS[case],
                label=case,
                zorder=3,
            )
            axis.annotate(
                case,
                (row.seconds_per_analysis, row[metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
            axis.set_xlabel("Seconds per analysis")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
    figure.suptitle("073A cost-skill trade-off (10 seeds, 199 cycles each)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rank_histograms(rank_histograms: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharey=True)
    for axis, case in zip(axes.flat, CASES, strict=True):
        counts = rank_histograms[case].astype(np.float64)
        relative_to_uniform = counts / counts.sum() * len(counts)
        positions = np.linspace(0.0, 1.0, len(counts))
        axis.bar(positions, relative_to_uniform, width=0.82 / max(len(counts) - 1, 1), color=COLORS[case])
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(case)
        axis.set_xlabel("Normalized rank")
        axis.set_ylabel("Frequency / uniform frequency")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("073A rank histograms")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cycle_metrics(cycles: pd.DataFrame, output: Path) -> None:
    metrics = (
        ("hr_maer", "HR-grid MAER"),
        ("laplacian_pv_rmse", "Laplacian-PV RMSE"),
        ("physical_crps", "Physical CRPS"),
        ("coverage_90", "90% coverage"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    means = cycles.groupby(["case", "cycle_index"], as_index=False).mean(numeric_only=True)
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        for case in CASES:
            selection = means[means.case == case]
            axis.plot(
                selection.cycle_index,
                selection[metric],
                label=case,
                color=COLORS[case],
                linewidth=1.4,
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Analysis cycle")
    axes[1, 1].set_xlabel("Analysis cycle")
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle("073A reduced-cost metrics through the cycling evaluation")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_focused_skill_curves(
    *,
    root: Path,
    cycles: pd.DataFrame,
    output_dir: Path,
) -> None:
    presentation = (
        root
        / "work/experiments/comparative_analysis_sb_srda_qg/metrics/presentation"
    )
    specifications = (
        ("hr_maer", "point_metrics_hr_grid.csv", "maer", "HR-grid MAER", "hr_maer_curves.png"),
        ("mssim_loss", "mssim_metrics_hr_grid.csv", "mssim_loss", "MSSIM loss", "hr_mssim_curves.png"),
        (
            "laplacian_pv_rmse",
            "laplacian_omega_metrics_hr_grid.csv",
            "laplacian_rmse",
            "Laplacian-PV RMSE",
            "hr_laplacian_rmse_curves.png",
        ),
    )
    reduced = cycles.copy()
    reduced["time_index"] = reduced.cycle_index / 4.0
    reference_color = "#238B45"
    for reduced_metric, reference_file, reference_metric, title, filename in specifications:
        reference = pd.read_csv(presentation / reference_file)
        reference = reference[
            (reference.model == "HR-LETKF M30")
            & (reference.layer_group == "all_layers")
        ]
        reference = reference.groupby("time_index", as_index=False)[reference_metric].mean()
        reduced_mean = reduced.groupby(["case", "time_index"], as_index=False)[
            reduced_metric
        ].mean()
        figure, axis = plt.subplots(figsize=(11.5, 5.4))
        axis.plot(
            reference.time_index,
            reference[reference_metric],
            label="HR-LETKF M30",
            color=reference_color,
            linewidth=2.7,
            zorder=5,
        )
        for case in CASES:
            selection = reduced_mean[reduced_mean.case == case]
            linewidth = 2.6 if case == "NFE20/M30" else 1.8
            axis.plot(
                selection.time_index,
                selection[reduced_metric],
                label=f"073A {case}",
                color=COLORS[case],
                linewidth=linewidth,
                alpha=0.95,
            )
        axis.set_xlabel("Assimilation time index")
        axis.set_ylabel(title)
        axis.set_title(f"073A reduced-cost comparison: {title}")
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def plot_flowfields(
    *,
    root: Path,
    seed: int,
    cycle: int,
    output_dir: Path,
    evaluation_override: Path | None,
) -> None:
    layer_specs = ((15, "top"), (8, "middle"), (0, "bottom"))
    truth = np.load(truth_path(root, seed), mmap_mode="r")
    case_fields: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    cycle_index: int | None = None
    reference: np.ndarray | None = None
    for case, leaf in CASES.items():
        _, mean, spread, online, _ = load_case(
            root, seed, leaf, evaluation_override
        )
        matches = np.flatnonzero(online.cycle_index.to_numpy() == cycle)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one cycle {cycle} row for {case}; found {len(matches)}")
        index = int(matches[0])
        if cycle_index is None:
            cycle_index = index
            reference = hr_reference(np.asarray(truth[cycle]), tuple(mean[index].shape))
        case_fields[case] = (
            np.asarray(mean[index], dtype=np.float32),
            np.asarray(spread[index], dtype=np.float32),
        )
    assert reference is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 5, figsize=(19, 8.2), constrained_layout=True)
    for row, (z_index, layer) in enumerate(layer_specs):
        values = [reference[z_index]] + [case_fields[case][0][z_index] for case in CASES]
        vmin = min(float(value.min()) for value in values)
        vmax = max(float(value.max()) for value in values)
        for column, (axis, value) in enumerate(zip(axes[row], values, strict=True)):
            image = axis.imshow(value, origin="lower", cmap="seismic", vmin=vmin, vmax=vmax)
            axis.set_title(("HR truth" if column == 0 else list(CASES)[column - 1]) + f"\n{layer}, z={z_index}")
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(image, ax=axes[row].tolist(), shrink=0.75)
    figure.suptitle(f"073A PV fields: seed {seed}, cycle {cycle}")
    figure.savefig(output_dir / f"seed{seed:05d}_cycle{cycle:04d}_pv.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    for field_name in ("error", "spread"):
        figure, axes = plt.subplots(3, 4, figsize=(15.5, 8.2), constrained_layout=True)
        for row, (z_index, layer) in enumerate(layer_specs):
            if field_name == "error":
                values = [
                    np.abs(case_fields[case][0][z_index] - reference[z_index])
                    for case in CASES
                ]
                cmap, vmin, vmax = "Greys", 0.0, max(float(value.max()) for value in values)
            else:
                values = [case_fields[case][1][z_index] for case in CASES]
                cmap, vmin, vmax = "viridis", 0.0, max(float(value.max()) for value in values)
            for axis, case, value in zip(axes[row], CASES, values, strict=True):
                image = axis.imshow(value, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_title(f"{case}\n{layer}, z={z_index}")
                axis.set_xticks([])
                axis.set_yticks([])
            figure.colorbar(image, ax=axes[row].tolist(), shrink=0.75)
        title = "absolute PV error" if field_name == "error" else "ensemble spread"
        figure.suptitle(f"073A {title}: seed {seed}, cycle {cycle}")
        figure.savefig(
            output_dir / f"seed{seed:05d}_cycle{cycle:04d}_{field_name}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(figure)

    figure, axes = plt.subplots(3, 5, figsize=(19, 8.2), constrained_layout=True)
    for row, (z_index, layer) in enumerate(layer_specs):
        values = [horizontal_laplacian(reference)[z_index]] + [
            horizontal_laplacian(case_fields[case][0])[z_index] for case in CASES
        ]
        limit = max(float(np.max(np.abs(value))) for value in values)
        for column, (axis, value) in enumerate(zip(axes[row], values, strict=True)):
            image = axis.imshow(
                value,
                origin="lower",
                cmap="PuOr_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_title(
                ("HR truth" if column == 0 else list(CASES)[column - 1])
                + f"\n{layer}, z={z_index}"
            )
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(image, ax=axes[row].tolist(), shrink=0.75)
    figure.suptitle(f"073A Laplacian-PV fields: seed {seed}, cycle {cycle}")
    figure.savefig(
        output_dir / f"seed{seed:05d}_cycle{cycle:04d}_laplacian_pv.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)
    for z_index, layer in layer_specs:
        values = [horizontal_laplacian(reference)[z_index]] + [
            horizontal_laplacian(case_fields[case][0])[z_index] for case in CASES
        ]
        limit = max(float(np.max(np.abs(value))) for value in values)
        figure, axes = plt.subplots(1, 5, figsize=(19, 3.3), constrained_layout=True)
        for column, (axis, value) in enumerate(zip(axes, values, strict=True)):
            image = axis.imshow(
                value,
                origin="lower",
                cmap="PuOr_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_title("HR truth" if column == 0 else list(CASES)[column - 1])
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(image, ax=axes.tolist(), shrink=0.78)
        figure.suptitle(
            f"073A Laplacian-PV: seed {seed}, cycle {cycle}, {layer} z={z_index}"
        )
        figure.savefig(
            output_dir
            / f"seed{seed:05d}_cycle{cycle:04d}_laplacian_pv_{layer}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    evaluation_override = (
        Path(args.evaluation_root).resolve() if args.evaluation_root else None
    )
    metrics_dir = (
        root
        / "work/experiments/comparative_analysis_sb_srda_qg/metrics/presentation/"
        "073a_reduced_cost"
    )
    figures_dir = (
        root
        / "work/experiments/comparative_analysis_sb_srda_qg/figures/presentation/"
        "073a_reduced_cost"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    previous_runtime: dict[tuple[str, int], tuple[float, float]] = {}
    previous_per_seed = metrics_dir / "per_seed_summary.csv"
    if previous_per_seed.exists():
        previous = pd.read_csv(previous_per_seed)
        for item in previous.itertuples(index=False):
            previous_runtime[(str(item.case), int(item.seed))] = (
                float(item.seconds_per_analysis),
                float(item.peak_gpu_bytes),
            )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mssim = MSSIM(
        window_3d_size=(5, 11, 11),
        sigma_3d=(0.7, 1.5, 1.5),
        value_magnitude=1.0,
        use_gaussian=False,
    ).to(device)

    per_seed_rows: list[dict[str, object]] = []
    per_cycle_frames: list[pd.DataFrame] = []
    rank_histograms: dict[str, np.ndarray] = {}
    for case, leaf in CASES.items():
        case_ranks: np.ndarray | None = None
        for seed in args.seeds:
            path, mean, _, online, runtime = load_case(
                root, seed, leaf, evaluation_override
            )
            spatial, mssim_loss = compute_spatial_metrics(
                root=root,
                seed=seed,
                mean=mean,
                online=online,
                mssim=mssim,
                device=device,
                batch_size=args.batch_size,
            )
            cycle_frame = spatial.merge(
                online[["cycle_index", *ONLINE_METRICS]],
                on="cycle_index",
                validate="one_to_one",
            )
            cycle_frame.insert(0, "case", case)
            per_cycle_frames.append(cycle_frame)
            row: dict[str, object] = {
                "case": case,
                "leaf": leaf,
                "seed": seed,
                "analyses": len(online),
                "hr_maer": float(spatial.hr_maer.mean()),
                "mssim_loss": mssim_loss,
                "laplacian_pv_rmse": float(spatial.laplacian_pv_rmse.mean()),
                "seconds_per_analysis": previous_runtime.get(
                    (case, seed),
                    (float(runtime["seconds_per_analysis"]), float("nan")),
                )[0],
                "peak_gpu_bytes": previous_runtime.get(
                    (case, seed),
                    (float("nan"), float(runtime.get("peak_gpu_bytes", np.nan))),
                )[1],
            }
            row.update({metric: float(online[metric].mean()) for metric in ONLINE_METRICS})
            per_seed_rows.append(row)
            ranks = np.asarray(np.load(path / "rank_histogram.npy"), dtype=np.int64)
            case_ranks = ranks.copy() if case_ranks is None else case_ranks + ranks
        assert case_ranks is not None
        rank_histograms[case] = case_ranks

    per_seed = pd.DataFrame(per_seed_rows)
    per_cycle = pd.concat(per_cycle_frames, ignore_index=True)
    per_seed.to_csv(metrics_dir / "per_seed_summary.csv", index=False)
    per_cycle.to_csv(metrics_dir / "per_cycle_summary.csv", index=False)
    summary = per_seed.groupby("case", sort=False).mean(numeric_only=True).loc[list(CASES)]
    summary.to_csv(metrics_dir / "aggregate_summary.csv")
    np.savez_compressed(
        metrics_dir / "rank_histograms.npz",
        **{case.replace("/", "_").lower(): values for case, values in rank_histograms.items()},
    )

    baseline = summary.loc["NFE20/M30"]
    relative = summary.copy()
    for metric in ("hr_maer", "mssim_loss", "laplacian_pv_rmse", "physical_rmse", "physical_crps"):
        relative[f"{metric}_relative_change_percent"] = (
            (summary[metric] / baseline[metric]) - 1.0
        ) * 100.0
    relative["coverage90_change_percentage_points"] = (
        summary.coverage_90 - baseline.coverage_90
    ) * 100.0
    relative["runtime_speedup_vs_nfe20_m30"] = (
        baseline.seconds_per_analysis / summary.seconds_per_analysis
    )
    relative.to_csv(metrics_dir / "relative_to_nfe20_m30.csv")

    metadata = {
        "model": "073A EMA epoch 2000",
        "seeds": args.seeds,
        "analyses_per_seed": 199,
        "truth_grid": "HR",
        "truth_construction": "drop UHR y=0, avg_pool3d factor 2, restore HR y=0",
        "cases": CASES,
        "panel": {"seed": args.panel_seed, "cycle": args.panel_cycle},
        "nfe_semantics": {"NFE20": "10 Heun intervals", "NFE10": "5 Heun intervals"},
        "runtime_source": "preserved from the completed analysis-only reduced-cost evaluations when available",
    }
    (metrics_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plot_bars(
        summary,
        [
            ("hr_maer", "HR-grid MAER"),
            ("mssim_loss", "MSSIM loss"),
            ("laplacian_pv_rmse", "Laplacian-PV RMSE"),
        ],
        figures_dir / "point_metrics.png",
    )
    plot_bars(
        summary,
        [
            ("physical_crps", "Physical CRPS"),
            ("spread_rmse_ratio", "Spread / RMSE"),
            ("coverage_90", "90% coverage"),
        ],
        figures_dir / "uq_metrics.png",
    )
    plot_bars(
        summary,
        [("seconds_per_analysis", "Seconds per analysis")],
        figures_dir / "runtime.png",
    )
    plot_cost_skill(summary, figures_dir / "cost_skill_pareto.png")
    plot_rank_histograms(rank_histograms, figures_dir / "rank_histograms.png")
    plot_cycle_metrics(per_cycle, figures_dir / "metrics_by_cycle.png")
    plot_focused_skill_curves(root=root, cycles=per_cycle, output_dir=figures_dir)
    plot_flowfields(
        root=root,
        seed=args.panel_seed,
        cycle=args.panel_cycle,
        output_dir=figures_dir / "flowfields",
        evaluation_override=evaluation_override,
    )
    print(summary.to_string())
    print(f"wrote {metrics_dir}")
    print(f"wrote {figures_dir}")


if __name__ == "__main__":
    main()
