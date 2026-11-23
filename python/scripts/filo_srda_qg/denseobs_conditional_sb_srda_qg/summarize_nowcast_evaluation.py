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


RUN_NAME = (
    "DenseObsConditionalSBUNet3D_Scratch_quad_p2_eps02582_"
    "ogx08_ogy08_mixup_b32_ema999"
)
LEAF = "ema_ep2000_nfe20_m10"
LABEL = "073A EMA ep2000 NFE20/M10"
PV_MIN = -2.88
PV_MAX = 2.92
PV_RANGE = PV_MAX - PV_MIN
LAYER_GROUPS = {
    "all_layers": tuple(range(16)),
    "bottom5": tuple(range(0, 5)),
    "middle5": tuple(range(6, 11)),
    "top5": tuple(range(11, 16)),
}
LEAD_COLORS = {
    0: "#2C7BB6",
    1: "#00A6CA",
    2: "#00CCBC",
    3: "#F9D057",
    4: "#D7191C",
}
ONLINE_COLUMNS = (
    "physical_mae",
    "physical_rmse",
    "physical_spread_rms",
    "spread_rmse_ratio",
    "physical_crps",
    "coverage_50",
    "coverage_80",
    "coverage_90",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed 073A NFE20/M10 narrow-jet nowcasts."
    )
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(9990, 10000)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--panel-seed", type=int, default=9991)
    parser.add_argument(
        "--panel-cycles", nargs="+", type=int, default=[100, 200, 300, 500, 792]
    )
    return parser.parse_args()


def evaluation_root(root: Path) -> Path:
    return (
        root
        / "work/experiments/073a_denseobs_conditional_sb_srda_qg/evaluation/"
        "narrow_jet"
        / RUN_NAME
        / "conditional_member_wise"
    )


def leaf_path(root: Path, seed: int) -> Path:
    path = evaluation_root(root) / f"seed{seed:05d}" / LEAF
    required = (
        "_COMPLETE",
        "online_nowcast_metrics.csv",
        "nowcast_mean_leads.npy",
        "nowcast_spread_leads.npy",
        "nowcast_rank_histograms.npy",
    )
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete nowcast leaf {path}: missing {missing}")
    return path


def truth_path(root: Path, seed: int) -> Path:
    return (
        root
        / "data/four_dim_srda/handover/cfd_simulation/qg_model/"
        "uhr_pv_narrow_jet"
        / f"seed{seed:05d}"
        / f"seed{seed:05d}_start000_end800_uhr_pv.npy"
    )


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


def compute_point_metrics(
    *,
    root: Path,
    seed: int,
    online: pd.DataFrame,
    means: np.ndarray,
    mssim: MSSIM,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    truth = np.load(truth_path(root, seed), mmap_mode="r")
    rows: list[dict[str, float | int | str]] = []
    for start in range(0, len(online), batch_size):
        batch = online.iloc[start : start + batch_size]
        predictions = np.stack(
            [
                np.asarray(
                    means[int(item.analysis_number), int(item.lead)],
                    dtype=np.float32,
                )
                for item in batch.itertuples(index=False)
            ]
        )
        references = np.stack(
            [
                hr_reference(
                    np.asarray(truth[int(item.target_cycle_index)]),
                    tuple(predictions[0].shape),
                )
                for item in batch.itertuples(index=False)
            ]
        )
        difference = predictions - references
        laplacian_difference = horizontal_laplacian(predictions) - horizontal_laplacian(
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
            ).mean(dim=(1, 3, 4))
        for offset, item in enumerate(batch.itertuples(index=False)):
            for layer_group, layers in LAYER_GROUPS.items():
                indices = list(layers)
                expected = references[offset, indices]
                delta = difference[offset, indices]
                rows.append(
                    {
                        "model": LABEL,
                        "seed": seed,
                        "lead": int(item.lead),
                        "analysis_index": int(item.analysis_number),
                        "cycle_index": int(item.cycle_index),
                        "target_cycle_index": int(item.target_cycle_index),
                        "time_index": float(item.target_cycle_index) / 4.0,
                        "layer_group": layer_group,
                        "hr_maer": float(
                            np.mean(np.abs(delta))
                            / max(float(np.mean(np.abs(expected))), 1.0e-12)
                        ),
                        "mssim_loss": float(
                            1.0 - ssim_map[offset, indices].mean().item()
                        ),
                        "laplacian_pv_rmse": float(
                            np.sqrt(
                                np.mean(laplacian_difference[offset, indices] ** 2)
                            )
                        ),
                    }
                )
    return pd.DataFrame(rows)


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_metric_by_cycle(
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.0))
    for lead in range(5):
        selection = frame[frame.lead == lead]
        selection = selection.groupby("target_cycle_index", as_index=False)[metric].mean()
        axis.plot(
            selection.target_cycle_index,
            selection[metric],
            label=f"lead {lead}",
            color=LEAD_COLORS[lead],
            linewidth=1.7,
        )
    axis.set_xlabel("Target QG cycle index")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{LABEL}: {ylabel}")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    figure.tight_layout()
    save_figure(figure, output)


def plot_summary_by_lead(summary: pd.DataFrame, output: Path) -> None:
    specifications = (
        ("hr_maer", "HR-grid MAER"),
        ("mssim_loss", "MSSIM loss"),
        ("laplacian_pv_rmse", "Laplacian-PV RMSE"),
        ("physical_crps", "Physical CRPS"),
        ("spread_rmse_ratio", "Spread / RMSE"),
        ("coverage_90", "90% coverage"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for axis, (metric, ylabel) in zip(axes.flat, specifications, strict=True):
        axis.plot(
            summary.index,
            summary[metric],
            marker="o",
            color="#C51B7D",
            linewidth=2.0,
        )
        axis.set_xlabel("Nowcast lead")
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(5))
        axis.grid(alpha=0.25)
    figure.suptitle(f"{LABEL}: mean metrics versus lead")
    figure.tight_layout()
    save_figure(figure, output)


def plot_rank_histograms(rank_counts: np.ndarray, output: Path) -> None:
    figure, axes = plt.subplots(1, 5, figsize=(17, 3.5), sharey=True)
    for lead, axis in enumerate(axes):
        counts = rank_counts[lead].astype(np.float64)
        relative = counts / counts.sum() * len(counts)
        axis.bar(np.arange(len(counts)), relative, color=LEAD_COLORS[lead])
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"lead {lead}")
        axis.set_xlabel("Rank")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Frequency / uniform")
    figure.suptitle(f"{LABEL}: rank histograms")
    figure.tight_layout()
    save_figure(figure, output)


def plot_flowfield_comparisons(
    *,
    root: Path,
    seed: int,
    cycles: list[int],
    output_dir: Path,
) -> None:
    path = leaf_path(root, seed)
    online = pd.read_csv(path / "online_nowcast_metrics.csv")
    means = np.load(path / "nowcast_mean_leads.npy", mmap_mode="r")
    spreads = np.load(path / "nowcast_spread_leads.npy", mmap_mode="r")
    truth = np.load(truth_path(root, seed), mmap_mode="r")
    layers = ((15, "top"), (8, "middle"), (0, "bottom"))
    for cycle in cycles:
        selections: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for lead in (0, 4):
            match = online[(online.cycle_index == cycle) & (online.lead == lead)]
            if len(match) != 1:
                raise RuntimeError(
                    f"Expected one row for seed={seed}, cycle={cycle}, lead={lead}; "
                    f"found {len(match)}"
                )
            item = match.iloc[0]
            index = int(item.analysis_number)
            target = int(item.target_cycle_index)
            prediction = np.asarray(means[index, lead], dtype=np.float32)
            spread = np.asarray(spreads[index, lead], dtype=np.float32)
            reference = hr_reference(np.asarray(truth[target]), tuple(prediction.shape))
            selections[lead] = (prediction, spread, reference)

        for field in ("pv", "error", "spread", "laplacian_pv"):
            columns = 3 if field != "spread" else 2
            figure, axes = plt.subplots(
                3, columns, figsize=(5.1 * columns, 8.1), constrained_layout=True
            )
            axes = np.asarray(axes).reshape(3, columns)
            for row, (z_index, layer_name) in enumerate(layers):
                lead0, spread0, truth0 = selections[0]
                lead4, spread4, truth4 = selections[4]
                if field == "pv":
                    values = [truth0[z_index], lead0[z_index], lead4[z_index]]
                    titles = ["HR truth (lead 0)", "lead 0", "lead 4"]
                    cmap = "seismic"
                    limit = max(float(np.max(np.abs(value))) for value in values)
                    vmin, vmax = -limit, limit
                elif field == "error":
                    values = [
                        np.abs(lead0[z_index] - truth0[z_index]),
                        np.abs(lead4[z_index] - truth4[z_index]),
                        np.abs(truth4[z_index] - truth0[z_index]),
                    ]
                    titles = ["lead-0 absolute error", "lead-4 absolute error", "truth evolution"]
                    cmap = "Greys"
                    vmin, vmax = 0.0, max(float(value.max()) for value in values)
                elif field == "spread":
                    values = [spread0[z_index], spread4[z_index]]
                    titles = ["lead-0 spread", "lead-4 spread"]
                    cmap = "viridis"
                    vmin, vmax = 0.0, max(float(value.max()) for value in values)
                else:
                    values = [
                        horizontal_laplacian(truth0)[z_index],
                        horizontal_laplacian(lead0)[z_index],
                        horizontal_laplacian(lead4)[z_index],
                    ]
                    titles = ["HR Laplacian (lead 0)", "lead-0 Laplacian", "lead-4 Laplacian"]
                    cmap = "PuOr_r"
                    limit = max(float(np.max(np.abs(value))) for value in values)
                    vmin, vmax = -limit, limit
                for column, (value, title) in enumerate(zip(values, titles, strict=True)):
                    image = axes[row, column].imshow(
                        value, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
                    )
                    axes[row, column].set_title(f"{title}\n{layer_name}, z={z_index}")
                    axes[row, column].set_xticks([])
                    axes[row, column].set_yticks([])
                figure.colorbar(image, ax=axes[row].tolist(), shrink=0.75)
            figure.suptitle(
                f"{LABEL}: seed {seed}, analysis cycle {cycle}, {field.replace('_', ' ')}"
            )
            save_figure(
                figure,
                output_dir / f"seed{seed:05d}_cycle{cycle:04d}_{field}.png",
            )


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).resolve()
    metrics_dir = (
        root
        / "work/experiments/comparative_analysis_sb_srda_qg/metrics/presentation/"
        "073a_nowcast_nfe20_m10"
    )
    figures_dir = (
        root
        / "work/experiments/comparative_analysis_sb_srda_qg/figures/presentation/"
        "073a_nowcast_nfe20_m10"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mssim = MSSIM(
        window_3d_size=(5, 11, 11),
        sigma_3d=(0.7, 1.5, 1.5),
        value_magnitude=1.0,
        use_gaussian=False,
    ).to(device)

    online_frames: list[pd.DataFrame] = []
    point_frames: list[pd.DataFrame] = []
    rank_counts: np.ndarray | None = None
    for seed in args.seeds:
        path = leaf_path(root, seed)
        online = pd.read_csv(path / "online_nowcast_metrics.csv")
        expected_rows = 4 * 199 + 198
        if len(online) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} nowcast rows in {path}; got {len(online)}")
        online.insert(0, "seed", seed)
        online_frames.append(online)
        means = np.load(path / "nowcast_mean_leads.npy", mmap_mode="r")
        point_frames.append(
            compute_point_metrics(
                root=root,
                seed=seed,
                online=online,
                means=means,
                mssim=mssim,
                device=device,
                batch_size=args.batch_size,
            )
        )
        ranks = np.asarray(np.load(path / "nowcast_rank_histograms.npy"), dtype=np.int64)
        rank_counts = ranks.copy() if rank_counts is None else rank_counts + ranks

    online_all = pd.concat(online_frames, ignore_index=True)
    point_all = pd.concat(point_frames, ignore_index=True)
    online_all.to_csv(metrics_dir / "online_nowcast_metrics_all.csv", index=False)
    point_all.to_csv(metrics_dir / "point_metrics_hr_grid.csv", index=False)
    assert rank_counts is not None
    np.save(metrics_dir / "rank_histograms.npy", rank_counts)

    online_seed = online_all.groupby(["lead", "seed"], as_index=False).mean(numeric_only=True)
    online_summary = online_seed.groupby("lead", sort=True).mean(numeric_only=True)
    point_all_layers = point_all[point_all.layer_group == "all_layers"]
    point_seed = point_all_layers.groupby(["lead", "seed"], as_index=False)[
        ["hr_maer", "mssim_loss", "laplacian_pv_rmse"]
    ].mean()
    point_summary = point_seed.groupby("lead", sort=True).mean(numeric_only=True)
    summary = online_summary.join(
        point_summary[["hr_maer", "mssim_loss", "laplacian_pv_rmse"]]
    )
    summary.to_csv(metrics_dir / "summary_by_lead.csv")

    metadata = {
        "model": LABEL,
        "leaf": LEAF,
        "seeds": args.seeds,
        "analysis_counts_by_lead": {
            str(int(lead)): int(count)
            for lead, count in online_all.groupby("lead").size().items()
        },
        "truth_grid": "HR",
        "truth_construction": "drop UHR y=0, avg_pool3d factor 2, restore HR y=0",
        "nowcast_mapping": {"0": 4, "1": 5, "2": 6, "3": 7, "4": 8},
        "point_metrics": ["hr_maer", "mssim_loss", "laplacian_pv_rmse"],
        "uq_metrics": list(ONLINE_COLUMNS),
    }
    (metrics_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for metric, ylabel, filename in (
        ("hr_maer", "HR-grid MAER", "nowcast_hr_maer_all_layers_by_lead.png"),
        ("mssim_loss", "MSSIM loss", "nowcast_hr_mssim_all_layers_by_lead.png"),
        (
            "laplacian_pv_rmse",
            "Laplacian-PV RMSE",
            "nowcast_laplacian_pv_rmse_all_layers_by_lead.png",
        ),
    ):
        plot_metric_by_cycle(
            point_all_layers,
            metric=metric,
            ylabel=ylabel,
            output=figures_dir / filename,
        )
    for metric, ylabel, filename in (
        ("physical_mae", "Physical MAE", "nowcast_physical_mae_by_lead.png"),
        ("physical_rmse", "Physical RMSE", "nowcast_physical_rmse_by_lead.png"),
        ("physical_crps", "Physical CRPS", "nowcast_physical_crps_by_lead.png"),
        (
            "spread_rmse_ratio",
            "Spread / RMSE",
            "nowcast_spread_rmse_ratio_by_lead.png",
        ),
        ("coverage_90", "90% coverage", "nowcast_coverage90_by_lead.png"),
    ):
        plot_metric_by_cycle(
            online_all,
            metric=metric,
            ylabel=ylabel,
            output=figures_dir / filename,
        )
    plot_summary_by_lead(summary, figures_dir / "nowcast_mean_metrics_vs_lead.png")
    plot_rank_histograms(rank_counts, figures_dir / "nowcast_rank_histograms.png")
    plot_flowfield_comparisons(
        root=root,
        seed=args.panel_seed,
        cycles=args.panel_cycles,
        output_dir=figures_dir / "flowfields",
    )
    print(summary.to_string())
    print(f"wrote {metrics_dir}")
    print(f"wrote {figures_dir}")


if __name__ == "__main__":
    main()
