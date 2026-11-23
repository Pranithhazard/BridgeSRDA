"""Server-side postprocessing only: paired stride-2 versus normal M10 cycling.

No inference or physical integration. Match all delivered integer times,
including previously issued lead-4 nowcasts at skipped analysis times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.summarize_nowcast_evaluation import (
    ONLINE_COLUMNS, RUN_NAME, compute_point_metrics, hr_reference,
    truth_path,
)
from src.four_dim_srda.utils.ssim import MSSIM


def delivered_rows(online: pd.DataFrame) -> pd.DataFrame:
    rows = online[online.lead.isin([0, 4])].copy().sort_values("target_cycle_index")
    expected = np.arange(4, 800, 4)
    if not np.array_equal(rows.target_cycle_index.to_numpy(), expected):
        raise ValueError("Stride delivery must cover each of the 199 integer verification times exactly once")
    if not np.array_equal(rows.loc[rows.lead.eq(0), "cycle_index"], np.arange(4, 800, 8)):
        raise ValueError("Wrong stride-2 analysis centers")
    if not (rows.target_cycle_index == rows.cycle_index + rows.lead).all():
        raise ValueError("Issued-nowcast time alignment failure")
    return rows.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--seeds", nargs="+", type=int, default=[9990, 9991])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use a spare server GPU for the MSSIM postprocessing")
    root, out = args.root_dir.resolve(), args.output_dir
    if (out / "_COMPLETE").exists():
        raise FileExistsError(f"Already complete: {out}")
    out.mkdir(parents=True, exist_ok=True)
    eval_root = root / "work/experiments/073a_denseobs_conditional_sb_srda_qg/evaluation/narrow_jet" / RUN_NAME / "conditional_member_wise"
    cache = root / "work/experiments/comparative_analysis_sb_srda_qg/metrics/presentation"
    base_points = pd.read_csv(cache / "073a_reduced_cost/per_cycle_summary.csv")
    old_lead_points = pd.read_csv(cache / "073a_nowcast_nfe20_m10/point_metrics_hr_grid.csv")
    old_lead_points = old_lead_points[old_lead_points.layer_group.eq("all_layers")]
    device = torch.device("cuda:0")
    mssim = MSSIM(window_3d_size=(5, 11, 11), sigma_3d=(.7, 1.5, 1.5), value_magnitude=1., use_gaussian=False).to(device)
    frames, runtimes, field_payload, contracts = [], [], {}, []
    for seed in args.seeds:
        base = eval_root / f"seed{seed:05d}/ema_ep2000_nfe20_m10"
        stride = eval_root / f"seed{seed:05d}/ema_ep2000_nfe20_m10_stride2"
        for path in [base, stride]:
            for filename in ["_COMPLETE", "runtime.json", "online_metrics.csv", "online_nowcast_metrics.csv", "nowcast_mean_leads.npy", "nowcast_spread_leads.npy"]:
                if not (path / filename).is_file():
                    raise FileNotFoundError(path / filename)
        rb, rs = [json.loads((p / "runtime.json").read_text()) for p in [base, stride]]
        for key in ["ensemble_size", "nfe_per_member", "weights_kind", "checkpoint_label", "sampling_seed", "brownian_base_steps", "observation_scenario_seed", "observation_source_path"]:
            if rb[key] != rs[key]:
                raise ValueError(f"Unmatched {key} for seed {seed}")
        if rb["analysis_indices"] != list(range(4, 800, 4)) or rs["analysis_indices"] != list(range(4, 800, 8)):
            raise ValueError("Unexpected center schedule")
        if rs["feedback_frame"] != 8 or rs["feedback_policy"] != "terminal_nowcast":
            raise ValueError("Wrong stride feedback")
        baseline_online = pd.read_csv(base / "online_metrics.csv")
        base_now = pd.read_csv(base / "online_nowcast_metrics.csv")
        stride_now = pd.read_csv(stride / "online_nowcast_metrics.csv")
        means = np.load(stride / "nowcast_mean_leads.npy", mmap_mode="r")
        spreads = np.load(stride / "nowcast_spread_leads.npy", mmap_mode="r")
        delivery = delivered_rows(stride_now)
        # Recompute only the new fields; reuse corrected, matching baseline caches.
        points = compute_point_metrics(root=root, seed=seed, online=delivery, means=means, mssim=mssim, device=device, batch_size=args.batch_size)
        points = points[points.layer_group.eq("all_layers")]
        stride_metrics = delivery.merge(points[["analysis_index", "lead", "hr_maer", "mssim_loss", "laplacian_pv_rmse"]], left_on=["analysis_number", "lead"], right_on=["analysis_index", "lead"], validate="one_to_one")
        stride_metrics = stride_metrics.assign(case="stride2_delivery", seed=seed, physical_time=lambda d: d.target_cycle_index * .25)
        bpoints = base_points[base_points.case.eq("NFE20/M10") & base_points.seed.eq(seed)][["cycle_index", "hr_maer", "mssim_loss", "laplacian_pv_rmse"]]
        baseline_metrics = baseline_online.merge(bpoints, on="cycle_index", validate="one_to_one").assign(case="baseline_analysis", seed=seed, lead=0, target_cycle_index=lambda d: d.cycle_index, physical_time=lambda d: d.cycle_index * .25)
        if len(baseline_metrics) != 199 or len(stride_metrics) != 199:
            raise ValueError("Incomplete baseline/delivery support")
        # Same information-age control: baseline's previously issued terminal
        # nowcast, not its updated analysis using the new observation.
        old_control = base_now[base_now.cycle_index.isin(rs["analysis_indices"]) & base_now.lead.eq(4)].copy()
        cached = old_lead_points[old_lead_points.seed.eq(seed) & old_lead_points.lead.eq(4)][["cycle_index", "lead", "hr_maer", "mssim_loss", "laplacian_pv_rmse"]]
        old_control = old_control.merge(cached, on=["cycle_index", "lead"], validate="one_to_one").assign(case="baseline_issued_lead4", seed=seed, physical_time=lambda d: d.target_cycle_index * .25)
        if len(old_control) != 99:
            raise ValueError("Incomplete baseline issued-nowcast control")
        frames.extend([baseline_metrics, stride_metrics, old_control])
        # Online MAE/RMSE must reproduce saved fields and the shared HR reference.
        uhr = np.load(truth_path(root, seed), mmap_mode="r")
        for item in delivery.itertuples(index=False):
            prediction = np.asarray(means[item.analysis_number, item.lead], dtype=np.float32)
            target = hr_reference(uhr[item.target_cycle_index], tuple(prediction.shape))
            err = prediction - target
            if not np.isclose(np.abs(err).mean(), item.physical_mae, atol=2e-6, rtol=1e-5) or not np.isclose(np.sqrt(np.square(err).mean()), item.physical_rmse, atol=2e-6, rtol=1e-5):
                raise ValueError(f"Online truth/units mismatch: seed={seed}, target={item.target_cycle_index}")
        for name, runtime in [("baseline_analysis", rb), ("stride2_delivery", rs)]:
            runtimes.append({"seed": seed, "case": name, "model_calls": len(runtime["analysis_indices"]), "verified_integer_times": 199, "verified_horizon": 199., "total_wall_seconds": runtime["total_wall_time_seconds"], "seconds_per_verified_physical_unit": runtime["total_wall_time_seconds"] / 199., "seconds_per_call": runtime["seconds_per_analysis"], "sb_sampling_seconds": runtime["sb_sampling_seconds"]})
        contracts.append({"seed": seed, "baseline_source": str(base), "stride_source": str(stride), "baseline_runtime": rb, "stride_runtime": rs})
        if seed == 9991:
            targets = np.array([160, 500, 792])
            bm = np.load(base / "sb_mean_analysis.npy", mmap_mode="r")
            bs = np.load(base / "sb_spread_analysis.npy", mmap_mode="r")
            base_next = np.load(base / "nowcast_mean_leads.npy", mmap_mode="r")
            base_next_spread = np.load(base / "nowcast_spread_leads.npy", mmap_mode="r")
            refs, bmean, bspread, smean, sspread, nmean, nspread, centers, leads = [], [], [], [], [], [], [], [], []
            for target_i in targets:
                item = delivery.loc[delivery.target_cycle_index.eq(target_i)].iloc[0]
                ai, lead, issued = int(item.analysis_number), int(item.lead), int(item.cycle_index)
                ref = hr_reference(uhr[target_i], tuple(bm.shape[-3:]))
                bi = rb["analysis_indices"].index(int(target_i))
                issued_bi = rb["analysis_indices"].index(issued)
                refs.append(ref); bmean.append(bm[bi]); bspread.append(bs[bi])
                smean.append(means[ai, lead]); sspread.append(spreads[ai, lead])
                nmean.append(base_next[issued_bi, lead]); nspread.append(base_next_spread[issued_bi, lead])
                centers.append(issued); leads.append(lead)
            field_payload = {"truth": np.stack(refs), "baseline_mean": np.stack(bmean), "baseline_spread": np.stack(bspread), "stride_mean": np.stack(smean), "stride_spread": np.stack(sspread), "baseline_age_matched_mean": np.stack(nmean), "baseline_age_matched_spread": np.stack(nspread), "target_indices": targets, "stride_issued_indices": np.array(centers), "stride_leads": np.array(leads)}
    metrics = pd.concat(frames, ignore_index=True)
    metric_cols = ["hr_maer", "mssim_loss", "laplacian_pv_rmse", *ONLINE_COLUMNS]
    metrics.to_csv(out / "per_time_metrics.csv", index=False)
    views = [metrics[metrics.case.eq("baseline_analysis")].assign(comparison="all_integer_times"), metrics[metrics.case.eq("stride2_delivery")].assign(comparison="all_integer_times")]
    for comparison, select in [("shared_analysis_centers", lambda d: d.target_cycle_index.mod(8).eq(4)), ("skipped_analysis_times", lambda d: d.target_cycle_index.mod(8).eq(0))]:
        view = metrics[metrics.case.isin(["baseline_analysis", "stride2_delivery"])].copy()
        views.append(view[select(view)].assign(comparison=comparison))
    control = metrics[metrics.case.eq("baseline_issued_lead4") | (metrics.case.eq("stride2_delivery") & metrics.lead.eq(4))]
    views.append(control.assign(comparison="same_age_issued_nowcasts"))
    views = pd.concat(views, ignore_index=True)
    per_seed = views.groupby(["comparison", "case", "seed"])[metric_cols].mean().reset_index()
    counts = views.groupby(["comparison", "case", "seed"]).size().rename("samples").reset_index()
    per_seed = per_seed.merge(counts, on=["comparison", "case", "seed"], validate="one_to_one")
    per_seed.to_csv(out / "per_seed_summary.csv", index=False)
    per_seed.groupby(["comparison", "case"])[metric_cols + ["samples"]].mean().reset_index().to_csv(out / "aggregate_summary.csv", index=False)
    pd.DataFrame(runtimes).to_csv(out / "runtime_per_seed.csv", index=False)
    np.savez_compressed(out / "selected_fields_seed09991.npz", **field_payload)
    (out / "contract.json").write_text(json.dumps({"seeds": args.seeds, "ensemble_size": 10, "nfe": 20, "checkpoint": "073A terminal EMA ep2000", "scope": "two-seed pilot; not ten-seed primary result", "delivery": "199 identical target times; stride lead0 at odd integer times, previously issued lead4 at even times", "fairness": "baseline analysis uses a newer observation at skipped times; same-age baseline lead4 control separates this information-age difference from recursive cycling", "rank": "no paired rank histogram: baseline pooled ranks cannot be restricted to odd centers from saved moments", "runtime": "separately recorded runs, not guaranteed occupancy-matched; elapsed divided by verified horizon 199", "field_times": [40,125,198], "sources": contracts}, indent=2) + "\n")
    (out / "_COMPLETE").write_text("complete\n")
    print(per_seed.to_string(index=False))
    print("COMPLETE", out)


if __name__ == "__main__":
    main()
