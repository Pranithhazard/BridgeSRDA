"""CPU-only point-metric postprocessing of completed sparse-layout analyses.

Never performs inference or CFD integration. All four cases use the official
PyTorch MSSIM implementation on CPU, rather than mixing numerical backends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.summarize_nowcast_evaluation import (
    LAYER_GROUPS, PV_MIN, PV_RANGE, RUN_NAME, horizontal_laplacian,
    hr_reference, truth_path,
)
from src.four_dim_srda.utils.ssim import MSSIM, _ssim_3D


CASES = ["hr_letkf_canonical", "hr_letkf_sparse", "073a_canonical", "073a_sparse"]


def uniform_ssim_layers(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Official zero-padded 5x11x11 uniform SSIM, one value per z layer."""
    metric = MSSIM(window_3d_size=(5, 11, 11), sigma_3d=(.7, 1.5, 1.5),
                   value_magnitude=1., use_gaussian=False)
    x = torch.from_numpy(np.clip((prediction - PV_MIN) / PV_RANGE, 0, 1))[None, None]
    y = torch.from_numpy(np.clip((truth - PV_MIN) / PV_RANGE, 0, 1))[None, None]
    metric._set_window(reference_image=x)
    with torch.no_grad():
        return metric(y, x)[0].numpy()


def case_directory(root, seed, case):
    experiment = root / "work/experiments/073a_denseobs_conditional_sb_srda_qg"
    if case == "hr_letkf_canonical":
        return root / "work/experiments/072_conditional_sb_srda_qg/evaluation/narrow_jet/hr_letkf_m30_native" / f"seed{seed:05d}"
    if case == "hr_letkf_sparse":
        return experiment / "evaluation/paper_followup_20261005/hr_letkf_m30/sparse_ogx16_ogy16" / f"seed{seed:05d}"
    if case == "073a_canonical":
        return experiment / "evaluation/narrow_jet" / RUN_NAME / "conditional_member_wise" / f"seed{seed:05d}" / "ema_ep2000_nfe20_m30"
    return (experiment / "evaluation/narrow_jet_observation_robustness" / RUN_NAME
            / "sparse_ogx16_ogy16/conditional_member_wise" / f"seed{seed:05d}"
            / "ema_ep2000_nfe20_m30")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(9990, 10000)))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root, out = args.root_dir.resolve(), args.output_dir.resolve()
    stages = [160, 500, 792]
    canonical = root / "work/experiments/072_conditional_sb_srda_qg/evaluation/narrow_jet/hr_letkf_m30_native/seed09991"
    means = np.load(canonical / "letkf_mean_analysis.npy", mmap_mode="r")
    truth = np.load(truth_path(root, 9991), mmap_mode="r")
    reference_cache = pd.read_csv(root / "work/experiments/comparative_analysis_sb_srda_qg/metrics/presentation/mssim_metrics_hr_grid.csv")
    preflight = []
    for cycle in stages:
        ai = cycle // 4 - 1
        prediction = np.asarray(means[ai], dtype=np.float32)
        reference = hr_reference(truth[cycle], tuple(prediction.shape))
        fast = uniform_ssim_layers(prediction, reference)
        differences = {}
        for group, layers in LAYER_GROUPS.items():
            cached = reference_cache[(reference_cache.model.eq("HR-LETKF M30"))
                                     & reference_cache.seed.eq(9991)
                                     & reference_cache.cycle_index.eq(cycle)
                                     & reference_cache.layer_group.eq(group)]
            if len(cached) != 1:
                raise ValueError(f"Missing canonical cached MSSIM at {cycle}/{group}")
            differences[group] = float(1 - fast[list(layers)].mean() - cached.mssim_loss.iloc[0])
        # Preserve these differences; do not silently call CPU and old GPU
        # caches bitwise equivalent. Recompute every comparison case on CPU.
        preflight.append({"cycle_index": cycle, "cpu_minus_existing_cache": differences})
    print("PREFLIGHT", json.dumps(preflight), flush=True)
    if args.preflight_only:
        for case in CASES:
            path = case_directory(root, 9991, case)
            if not (path / "_COMPLETE").exists():
                raise FileNotFoundError(path)
            print("CASE", case, path, flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    source_contracts = []
    fields = {"cycle_indices": np.array(stages)}
    for seed in args.seeds:
        uhr = np.load(truth_path(root, seed), mmap_mode="r")
        for case in CASES:
            path = case_directory(root, seed, case)
            if not (path / "_COMPLETE").is_file():
                raise FileNotFoundError(path / "_COMPLETE")
            is_letkf = case.startswith("hr_letkf")
            mean_name = "letkf_mean_analysis.npy" if is_letkf else "sb_mean_analysis.npy"
            spread_name = "letkf_spread_analysis_ne30.npy" if is_letkf else "sb_spread_analysis.npy"
            online = pd.read_csv(path / ("metrics_ne30.csv" if is_letkf else "online_metrics.csv"))
            means = np.load(path / mean_name, mmap_mode="r")
            spreads = np.load(path / spread_name, mmap_mode="r")
            if online.cycle_index.tolist() != list(range(4, 800, 4)) or means.shape != (199, 16, 65, 128):
                raise ValueError(f"Wrong analysis support: {path}")
            rows = []
            for item in online.itertuples(index=False):
                ai, cycle = int(item.analysis_number), int(item.cycle_index)
                pred = np.asarray(means[ai], dtype=np.float32)
                ref = hr_reference(uhr[cycle], tuple(pred.shape))
                error = pred - ref
                mae, rmse = float(np.abs(error).mean()), float(np.sqrt(np.square(error).mean()))
                if not (np.isclose(mae, item.physical_mae, atol=2e-6, rtol=1e-5)
                        and np.isclose(rmse, item.physical_rmse, atol=2e-6, rtol=1e-5)):
                    raise ValueError(f"HR truth/units mismatch: {case}/{seed}/{cycle}")
                ss = uniform_ssim_layers(pred, ref)
                lap_error = horizontal_laplacian(pred) - horizontal_laplacian(ref)
                for group, layers in LAYER_GROUPS.items():
                    ix = list(layers)
                    rows.append({"case": case, "seed": seed, "analysis_index": ai,
                                 "cycle_index": cycle, "physical_time": cycle * .25,
                                 "layer_group": group,
                                 "hr_maer": float(np.abs(error[ix]).mean() / max(float(np.abs(ref[ix]).mean()), 1e-12)),
                                 "mssim_loss": float(1 - ss[ix].mean()),
                                 "laplacian_pv_rmse": float(np.sqrt(np.square(lap_error[ix]).mean()))})
                if seed == 9991 and cycle in stages:
                    for name, value in [("truth", ref), (case + "_mean", pred),
                                        (case + "_spread", np.asarray(spreads[ai]))]:
                        fields.setdefault(name, []).append(value.copy())
            pd.DataFrame(rows).to_csv(out / f"{case}_seed{seed:05d}.csv", index=False)
            source_contracts.append({"case": case, "seed": seed, "source": str(path)})
            print("FINISHED", case, seed, len(rows), flush=True)
    # The shared truth was appended once per case; retain one ordered copy.
    if "truth" in fields:
        fields["truth"] = fields["truth"][:3]
    np.savez_compressed(out / "selected_fields_seed09991.npz",
                        **{k: np.stack(v) if isinstance(v, list) else v for k, v in fields.items()})
    point = pd.concat([pd.read_csv(out / f"{case}_seed{seed:05d}.csv")
                       for seed in args.seeds for case in CASES], ignore_index=True)
    point.to_csv(out / "point_metrics_hr_grid.csv", index=False)
    point.groupby(["case", "layer_group"])[["hr_maer", "mssim_loss", "laplacian_pv_rmse"]].mean().reset_index().to_csv(out / "point_summary.csv", index=False)
    (out / "contract.json").write_text(json.dumps({"seeds": args.seeds, "analyses_per_seed": 199,
        "truth": "shared boundary-aware UHR-to-HR conversion; all-layer MAE/RMSE checked against online metrics",
        "mssim": "same clipped [-2.88,2.92] normalization, zero-padded uniform 5x11x11 kernel; official PyTorch MSSIM on CPU for all four cases",
        "mssim_cache_crosscheck": preflight,
        "mssim_comparison_note": "New subsection uses four consistently CPU-recomputed cases; original primary cache and tables remain unchanged. Recorded cache discrepancies are not attributed to an established cause.",
        "laplacian": "unit-grid periodic x, nearest-edge y; no physical-grid scaling",
        "field_seed": 9991, "field_times": [40, 125, 198], "sources": source_contracts,
        "execution": "CPU postprocessing only; no inference, CFD or training"}, indent=2) + "\n")
    (out / "_COMPLETE").write_text("complete\n")
    print(point.groupby(["case", "layer_group"])[["hr_maer", "mssim_loss", "laplacian_pv_rmse"]].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
