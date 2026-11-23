from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.conditioning import (
    DENSE_OBSERVATION_FRAME_INDICES,
    build_dense_observation_condition,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.data import (
    make_conditional_sb_dataset,
)
from src.filo_srda_qg.sr_prior.source import (
    build_bridge_source,
    fold_hr_target,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


LAYER_GROUPS = {
    "all": tuple(range(16)),
    "bottom": (0,),
    "middle": (8,),
    "top": (15,),
}


def laplacian_xy(field: torch.Tensor) -> torch.Tensor:
    dx = torch.roll(field, 1, dims=-1) - 2 * field + torch.roll(field, -1, dims=-1)
    padded = torch.nn.functional.pad(field, (0, 0, 1, 1), mode="replicate")
    dy = padded[..., :-2, :] - 2 * padded[..., 1:-1, :] + padded[..., 2:, :]
    return dx + dy


def gradient_statistics(
    field: torch.Tensor,
) -> tuple[float, float, float, float]:
    gx = (torch.roll(field, -1, dims=-1) - field).abs()
    gy = (field[..., 1:, :] - field[..., :-1, :]).abs()
    gradients = torch.cat([gx.flatten(), gy.flatten()])
    seam = (field[..., 0] - field[..., -1]).abs().mean()
    global_interior = gx[..., :-1].mean().clamp_min(1.0e-12)
    # A seam-adjacent edge on at least one side belongs to the same periodic
    # piecewise-linear segment as x=-1 -> x=0. Which side matches depends on
    # the Cartesian sensor-grid origin. Comparing with both local neighbors
    # tests periodic continuation without confusing a valid sharp structure
    # crossing the seam with an interpolation discontinuity.
    adjacent_before = (field[..., -1] - field[..., -2]).abs().mean()
    adjacent_after = (field[..., 1] - field[..., 0]).abs().mean()
    adjacent = torch.maximum(adjacent_before, adjacent_after)
    scale = torch.maximum(seam, adjacent)
    if float(scale) <= 1.0e-10:
        local_ratio = 0.0
    else:
        local_ratio = float(seam / adjacent.clamp_min(1.0e-12))
    return (
        local_ratio,
        float(seam / global_interior),
        float(gradients.max()),
        float(torch.quantile(gradients, 0.99)),
    )


def periodic_seam_reproduction_error(
    field: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Check the seam against the periodic linear segment that brackets it."""
    if field.shape != mask.shape:
        raise ValueError(
            f"field/mask shape mismatch: {tuple(field.shape)} vs {tuple(mask.shape)}"
        )
    x_size = int(field.shape[-1])
    active_x = (mask > 0.5).any(dim=tuple(range(mask.ndim - 1)))
    sensor_x = torch.nonzero(active_x, as_tuple=False).flatten()
    if sensor_x.numel() < 2:
        raise ValueError("periodic seam audit requires at least two sensor x-columns")

    first_x = int(sensor_x[0])
    last_x = int(sensor_x[-1])
    right_x = first_x + x_size
    span = right_x - last_x
    if span <= 0:
        raise ValueError("invalid periodic sensor bracketing interval")

    left_value = field[..., last_x]
    right_value = field[..., first_x]

    def expected_at(unwrapped_x: int) -> torch.Tensor:
        weight = float(unwrapped_x - last_x) / float(span)
        return left_value * (1.0 - weight) + right_value * weight

    error_last = (field[..., -1] - expected_at(x_size - 1)).abs().max()
    error_zero = (field[..., 0] - expected_at(x_size)).abs().max()
    return float(torch.maximum(error_last, error_zero))


def save_panel(
    *,
    sparse: torch.Tensor,
    dense: torch.Tensor,
    truth: torch.Tensor,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    error = dense - truth
    sparse_numpy = sparse.cpu().numpy()
    dense_numpy = dense.cpu().numpy()
    truth_numpy = truth.cpu().numpy()
    error_numpy = error.cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), constrained_layout=True)
    entries = [
        ("sparse observations", sparse_numpy, "seismic", -0.3, 1.3),
        ("dense interpolation", dense_numpy, "seismic", 0, 1),
        ("HR truth", truth_numpy, "seismic", 0, 1),
        ("error", error_numpy, "coolwarm", -0.5, 0.5),
    ]
    for axis, (title, values, cmap, vmin, vmax) in zip(axes, entries):
        image = axis.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_axis_off()
        fig.colorbar(image, ax=axis, fraction=0.046)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--windows", type=int)
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    config = load_config(args.config_path)
    audit_config = config.get("interpolation_audit", {})
    windows = int(args.windows or audit_config.get("windows", 96))
    torch.manual_seed(int(audit_config.get("seed", 7301)))
    dataset = make_conditional_sb_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="valid",
        use_mixup=False,
        samples_per_epoch=windows,
        deterministic_observations=True,
    )
    pv_range = float(config["dataset"]["pv_max"] - config["dataset"]["pv_min"])
    output_dir = root / config["output"]["work_root"] / "audits" / "interpolation"
    panel_dir = output_dir / "panels"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    methods = ("periodic_bilinear", "diffsrda_bicubic")
    panel_windows = {0, windows // 2, windows - 1}
    for window_index in range(windows):
        sample = dataset[window_index]
        source = build_bridge_source(sample["condition_lr"][None]).float()
        target = fold_hr_target(sample["hr"][None]).float()
        observations = sample["obs"][None].float()
        masks = sample["mask"][None].float()
        for method in methods:
            dense = build_dense_observation_condition(
                source, observations, masks, method=method
            )
            for slot, frame in enumerate(DENSE_OBSERVATION_FRAME_INDICES):
                truth = target[:, frame]
                active = masks[:, slot, 0] > 0.5
                sensor_error = float(
                    (dense[:, slot][active] - observations[:, slot, 0][active])
                    .abs()
                    .max()
                )
                for group, layers in LAYER_GROUPS.items():
                    selection = torch.tensor(layers, dtype=torch.long)
                    prediction = dense[:, slot].index_select(1, selection)
                    reference = truth.index_select(1, selection)
                    selected_mask = masks[:, slot, 0].index_select(1, selection)
                    difference = prediction - reference
                    (
                        seam_ratio,
                        seam_global_ratio,
                        gradient_max,
                        gradient_p99,
                    ) = gradient_statistics(prediction)
                    rows.append(
                        {
                            "window": window_index,
                            "method": method,
                            "slot": slot,
                            "target_frame": frame,
                            "layer_group": group,
                            "physical_mae": float(difference.abs().mean() * pv_range),
                            "normalized_rmse": float(difference.square().mean().sqrt()),
                            "physical_rmse": float(
                                difference.square().mean().sqrt() * pv_range
                            ),
                            "laplacian_pv_rmse": float(
                                (laplacian_xy(prediction) - laplacian_xy(reference))
                                .square()
                                .mean()
                                .sqrt()
                                * pv_range
                            ),
                            "sensor_site_max_abs_error": sensor_error,
                            "nonfinite_fraction": float(
                                (~torch.isfinite(prediction)).float().mean()
                            ),
                            "normalized_overshoot_fraction": float(
                                ((prediction < 0) | (prediction > 1)).float().mean()
                            ),
                            "periodic_seam_reproduction_error": (
                                periodic_seam_reproduction_error(
                                    prediction, selected_mask
                                )
                            ),
                            "periodic_seam_neighbor_gradient_ratio_diagnostic": (
                                seam_ratio
                            ),
                            "periodic_seam_global_mean_ratio": seam_global_ratio,
                            "maximum_gradient": gradient_max,
                            "p99_gradient": gradient_p99,
                        }
                    )
                if window_index in panel_windows:
                    for layer_name, layer_index in (
                        ("bottom", 0),
                        ("middle", 8),
                        ("top", 15),
                    ):
                        sparse = torch.full_like(observations[0, slot, 0, layer_index], torch.nan)
                        active_plane = masks[0, slot, 0, layer_index] > 0.5
                        sparse[active_plane] = observations[0, slot, 0, layer_index][active_plane]
                        save_panel(
                            sparse=sparse,
                            dense=dense[0, slot, layer_index],
                            truth=truth[0, layer_index],
                            output=(
                                panel_dir
                                / method
                                / f"window{window_index:03d}_slot{slot}_{layer_name}.png"
                            ),
                        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "interpolation_metrics.csv", index=False)
    summary = (
        frame.groupby(["method", "slot", "layer_group"], as_index=False)
        .mean(numeric_only=True)
        .to_dict(orient="records")
    )
    (output_dir / "interpolation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    production = frame[frame.method == "periodic_bilinear"]
    checks = {
        "all_outputs_finite": bool(production.nonfinite_fraction.max() == 0),
        "sensor_site_max_abs_error": float(production.sensor_site_max_abs_error.max()),
        "normalized_overshoot_fraction": float(
            production.normalized_overshoot_fraction.max()
        ),
        "periodic_seam_reproduction_error": float(
            production.periodic_seam_reproduction_error.max()
        ),
        "periodic_seam_neighbor_gradient_ratio_diagnostic": float(
            production.periodic_seam_neighbor_gradient_ratio_diagnostic.max()
        ),
        "periodic_seam_global_mean_ratio_diagnostic": float(
            production.periodic_seam_global_mean_ratio.max()
        ),
        "expected_rows": int(len(production)),
    }
    passed = (
        checks["all_outputs_finite"]
        and checks["sensor_site_max_abs_error"]
        <= float(audit_config.get("sensor_tolerance", 1.0e-6))
        and checks["normalized_overshoot_fraction"] == 0
        and checks["periodic_seam_reproduction_error"]
        <= float(audit_config.get("seam_reproduction_tolerance", 1.0e-6))
        and checks["expected_rows"] == windows * 2 * len(LAYER_GROUPS)
    )
    verdict = {
        "production_method": "periodic_bilinear",
        "audit_only_method": "diffsrda_bicubic",
        "windows": windows,
        "checks": checks,
        "passed": passed,
    }
    (output_dir / "interpolation_verdict.json").write_text(
        json.dumps(verdict, indent=2)
    )
    print(json.dumps(verdict, indent=2))
    if not passed:
        raise RuntimeError("073A interpolation audit failed")


if __name__ == "__main__":
    main()
