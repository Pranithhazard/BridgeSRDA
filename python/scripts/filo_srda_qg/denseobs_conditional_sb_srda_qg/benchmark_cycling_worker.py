"""Run one condition of the Appendix F cycling benchmark.

Independent single-GPU workers write isolated mean/spread/member arrays and
canonical point/UQ diagnostics. Core timings exclude verification and I/O.
Compilation/warmup, setup, core cycling, and evaluation wall costs are separate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg import benchmark_inference_timing as h
from src.filo_srda_qg.letkf.letkf_uq import ensemble_diagnostics, hr_truth_from_uhr
from src.four_dim_srda.utils.ssim import MSSIM

SELECTED = (0, 24, 49, 74, 99, 124, 149, 174, 192, 198)


def physical_centers(max_analyses=199):
    if not 1 <= max_analyses <= 199:
        raise ValueError("Expected 1..199 normal-cycling analyses")
    return list(range(4, 4 * max_analyses + 1, 4))


def validate_reference_shape(shape):
    if tuple(shape[1:]) != (32, 129, 256) or shape[0] not in (800, 801):
        raise ValueError(f"Wrong reference trajectory shape {shape}")


def horizontal_laplacian(value):
    x = torch.roll(value, 1, -1) + torch.roll(value, -1, -1) - 2 * value
    previous = torch.cat([value[..., :1, :], value[..., :-1, :]], dim=-2)
    following = torch.cat([value[..., 1:, :], value[..., -1:, :]], dim=-2)
    return x + previous + following - 2 * value


def point_metrics(mean, truth, ssim):
    x = ((mean + 2.88) / 5.80).clamp(0, 1)[None, None]
    y = ((truth + 2.88) / 5.80).clamp(0, 1)[None, None]
    if not hasattr(ssim, "window") or ssim.window.device != x.device:
        ssim._set_window(reference_image=x)
    similarity = ssim(y, x).mean()
    return dict(hr_maer=float((mean - truth).abs().mean() / truth.abs().mean().clamp_min(1e-12)),
                mssim_loss=float(1 - similarity),
                laplacian_pv_rmse=float((horizontal_laplacian(mean) - horizontal_laplacian(truth)).square().mean().sqrt()))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    torch.cuda.set_device(device)
    h.configure(args.threads, conv_tf32=True)
    if args.seed != 9991:
        raise ValueError("Appendix F uses seed 9991")
    centers = physical_centers(args.max_analyses)
    selected = [i for i in SELECTED if i < len(centers)]
    observation_path = root / "python/results/four_dim_srda/handover/srda/UNetMaxVitVer01/use_narrow_jet" / Path(h.SRDA_CONFIG).stem / f"UHR_seed_{args.seed:05d}.npz"
    truth_path = root / f"data/four_dim_srda/handover/cfd_simulation/qg_model/uhr_pv_narrow_jet/seed{args.seed:05d}/seed{args.seed:05d}_start000_end800_uhr_pv.npy"
    for gate in ("interpolation", "data", "preflight", "overfit", "ddp_smoke", "regression_tests"):
        if not (root / f"work/experiments/073a_denseobs_conditional_sb_srda_qg/gates/{gate}.passed").is_file():
            raise RuntimeError(f"Missing production gate {gate}")
    with np.load(observation_path, allow_pickle=False) as archive:
        observations = torch.from_numpy(np.array(archive["hr_obs"], dtype=np.float32, copy=True))
    if tuple(observations.shape) != (800, 16, 65, 128):
        raise ValueError(f"Unexpected observation shape {observations.shape}")
    observation_hash = __import__("hashlib").sha256(torch.nan_to_num(observations, nan=1e10).contiguous().numpy().tobytes()).hexdigest()
    assets = json.loads((root / "manifests/assets.json").read_text())
    timing_inputs = assets["timing_inputs"]
    if observation_hash != timing_inputs["observation_float32_nanfilled_sha256"]:
        raise ValueError("Appendix F requires the recorded canonical observation stream")
    if args.method == "bridge":
        checkpoint = assets["checkpoint_assets"]["bridge_d"]
        if h.digest(root / checkpoint["path"]) != checkpoint["historical_sha256"]:
            raise ValueError("Appendix F requires the recorded BridgeSRDA-D checkpoint")
    truth_array = np.load(truth_path, mmap_mode="r")
    # Numerical archives may retain the terminal t=200 field (801 frames).
    # Production verification uses indices 0..799, with analyses ending at796.
    validate_reference_shape(truth_array.shape)
    setup_start = time.perf_counter()
    if args.method == "bridge":
        from src.filo_srda_qg.timing_benchmark_qg.cycling_bridge import CompiledBridgeCycler
        cycler = CompiledBridgeCycler(root=root, seed=args.seed, device=device,
                                     observations=observations, ensemble_size=args.ensemble_size)
    else:
        if args.ensemble_size != 30:
            raise ValueError("The LETKF benchmark requires a native 30-member ensemble")
        from src.filo_srda_qg.timing_benchmark_qg.cycling_letkf import OptimizedLETKFCycler
        cycler = OptimizedLETKFCycler(root=root, seed=args.seed, device=device, observations=observations)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    ssim = MSSIM(window_3d_size=(5, 11, 11), sigma_3d=(.7, 1.5, 1.5),
                 value_magnitude=1., use_gaussian=False).to(device)
    shape = (len(centers), *cycler.shape)
    means = np.lib.format.open_memmap(output / "mean_analysis.npy", mode="w+", dtype=np.float32, shape=shape)
    spreads = np.lib.format.open_memmap(output / "spread_analysis.npy", mode="w+", dtype=np.float32, shape=shape)
    members = np.lib.format.open_memmap(output / "selected_members.npy", mode="w+", dtype=np.float32,
                                      shape=(len(selected), args.ensemble_size, *cycler.shape))
    selected_slot = {index: slot for slot, index in enumerate(selected)}
    np.save(output / "selected_member_analysis_indices.npy", np.asarray(selected, dtype=np.int64))
    np.save(output / "analysis_cycle_indices.npy", np.asarray(centers, dtype=np.int64))
    sources = [Path(__file__), *cycler.source_paths,
               root / "python/src/filo_srda_qg/timing_benchmark_qg/cycling_bridge.py",
               root / "python/src/filo_srda_qg/timing_benchmark_qg/cycling_letkf.py",
               root / "python/src/filo_srda_qg/timing_benchmark_qg/inference_network.py",
               root / "python/src/filo_srda_qg/timing_benchmark_qg/bridge_sampler.py",
               root / "python/src/filo_srda_qg/timing_benchmark_qg/letkf_kernels.py"]
    manifest = dict(args=vars(args), source_sha256={str(p.relative_to(root)): h.digest(p) for p in sources},
                    observation_path=str(observation_path), observation_float32_nanfilled_sha256=observation_hash,
                    truth_path=str(truth_path), truth_shape=list(truth_array.shape),
                    backend_settings=h.backend_settings(), torch_version=torch.__version__,
                    gpu=torch.cuda.get_device_name(device), gpu_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    setup_seconds=setup_seconds, analysis_indices=centers, selected_analysis_indices=selected,
                    feedback="member-wise current analysis j=4; normal stride-one cycling",
                    timing_scope="Synchronized forecast+conditions+update+feedback, excluding truth scoring and disk I/O. Setup and initial warmup separate. No unused HR forecast after final analysis.",
                    precision="Bridge: BF16 autocast network, FP32 parameters/states/noise/Heun; LETKF: production FP64 numerical evolution and analysis",
                    observation_scenario="canonical_ogx08_ogy08", nfe=20 if args.method == "bridge" else None)
    save_json(output / "manifest.json", manifest)
    all_rows, timing_rows, layer_rows = [], [], []
    ranks = torch.zeros(args.ensemble_size + 1, dtype=torch.long)
    layer_ranks = {name: torch.zeros_like(ranks) for name in ("top", "middle", "bottom")}
    with torch.inference_mode():
        before = time.perf_counter()
        warm_samples, warm_timing, warm_diagnostics = cycler.step(4)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - before
        if not torch.isfinite(warm_samples).all():
            raise FloatingPointError("Nonfinite warmup")
        # LETKF first-analysis check proves initialization/operator compatibility.
        archive_check = None
        if args.method == "letkf":
            archived = root / f"work/experiments/072_conditional_sb_srda_qg/evaluation/narrow_jet/hr_letkf_m30_native/seed{args.seed:05d}/letkf_mean_analysis.npy"
            expected = torch.from_numpy(np.array(np.load(archived, mmap_mode="r")[0], copy=True))
            difference = float((warm_samples.mean(0).float().cpu() - expected).abs().max())
            archive_check = dict(first_mean_max_abs=difference, tolerance=5e-5, archive=str(archived))
            if difference > 5e-5:
                raise RuntimeError(f"LETKF initial archive check failed {archive_check}")
        del warm_samples
        cycler.reset()
        save_json(output / "warmup.json", dict(seconds=warmup_seconds, components=warm_timing,
                                               diagnostics=warm_diagnostics, archive_check=archive_check,
                                               reset_to_exact_initial_state=True))
        torch.cuda.reset_peak_memory_stats(device)
        evaluation_start = time.perf_counter()
        for number, cycle_index in enumerate(centers):
            cycle_start = time.perf_counter()
            samples, timing, details = cycler.step(cycle_index)
            torch.cuda.synchronize()
            diag_start = time.perf_counter()
            target = hr_truth_from_uhr(torch.from_numpy(np.array(truth_array[cycle_index], copy=True)),
                                      nz=16, ny=65, nx=128).to(device=device, dtype=torch.float32)
            diagnostic = ensemble_diagnostics(samples, target)
            mean = samples.mean(0).float()  # Preserve LETKF's FP64 ensemble reduction.
            row = dict(analysis_number=number, cycle_index=cycle_index, physical_time=cycle_index * .25,
                       **diagnostic.metrics, **point_metrics(mean, target, ssim))
            if not all(np.isfinite(v) for v in row.values()):
                raise FloatingPointError(f"Nonfinite verification at cycle {cycle_index}")
            all_rows.append(row)
            ranks += diagnostic.rank_histogram.cpu()
            for name, z in (("top", 15), ("middle", 8), ("bottom", 0)):
                local = ensemble_diagnostics(samples[:, z], target[z])
                layer_rows.append(dict(analysis_number=number, cycle_index=cycle_index, layer=name,
                                       z_index=z, **local.metrics))
                layer_ranks[name] += local.rank_histogram.cpu()
            torch.cuda.synchronize()
            diagnostics_seconds = time.perf_counter() - diag_start
            io_start = time.perf_counter()
            means[number] = mean.cpu().numpy()
            spreads[number] = diagnostic.spread.cpu().numpy()
            if number in selected_slot:
                members[selected_slot[number]] = samples.float().cpu().numpy()
            timing_rows.append(dict(analysis_number=number, cycle_index=cycle_index, **timing,
                                    diagnostics_seconds=diagnostics_seconds,
                                    io_seconds=time.perf_counter() - io_start,
                                    cycle_wall_seconds=time.perf_counter() - cycle_start,
                                    fallback_blocks=len(details.get("fallback_block_ids", []))))
            save_json(output / "progress.json", dict(completed=number + 1, expected=len(centers),
                      physical_time=cycle_index * .25, last_core_seconds=timing["core_seconds"],
                      mean_core_seconds=statistics.mean(r["core_seconds"] for r in timing_rows),
                      last_metrics=row, details=details))
            if number < 3 or (number + 1) % 10 == 0 or number == len(centers) - 1:
                print(json.dumps(dict(completed=number + 1, physical_time=cycle_index * .25,
                     core_seconds=timing["core_seconds"], rmse=row["physical_rmse"],
                     coverage90=row["coverage_90"])), flush=True)
            del samples, diagnostic, target, mean, local
        torch.cuda.synchronize()
        evaluation_wall = time.perf_counter() - evaluation_start
    for array in (means, spreads, members):
        array.flush()
    write_csv(output / "metrics.csv", all_rows)
    write_csv(output / "layer_metrics.csv", layer_rows)
    write_csv(output / "cycle_timings.csv", timing_rows)
    np.save(output / "rank_histogram.npy", ranks.numpy())
    np.savez(output / "layer_rank_histograms.npz", **{k: v.numpy() for k, v in layer_ranks.items()})
    summary = dict(method=args.method, ensemble_size=args.ensemble_size, seed=args.seed,
                   analyses=len(centers), setup_seconds=setup_seconds, warmup_seconds=warmup_seconds,
                   cycling_wall_seconds=evaluation_wall, cycling_wall_seconds_per_analysis=evaluation_wall / len(centers),
                   mean_core_seconds=statistics.mean(r["core_seconds"] for r in timing_rows),
                   median_core_seconds=statistics.median(r["core_seconds"] for r in timing_rows),
                   component_mean_seconds={k: statistics.mean(r[k] for r in timing_rows) for k in
                                          ("forecast_seconds", "conditioning_seconds", "analysis_seconds", "postprocess_seconds", "diagnostics_seconds", "io_seconds")},
                   metrics={k: statistics.mean(r[k] for r in all_rows) for k in all_rows[0]
                            if k not in ("analysis_number", "cycle_index", "physical_time")},
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
                   fallback_blocks=sum(r["fallback_blocks"] for r in timing_rows),
                   status="completed one-seed recursive diagnostic; no automatic production promotion")
    save_json(output / "summary.json", summary)
    (output / "_COMPLETE").write_text("complete\n")
    print(json.dumps(summary), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("bridge", "letkf"), required=True)
    p.add_argument("--ensemble-size", type=int, choices=(5, 10, 30), required=True)
    p.add_argument("--seed", type=int, default=9991)
    p.add_argument("--max-analyses", type=int, default=199)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    try:
        run(parsed)
    except Exception:
        failure = Path(parsed.output) / "FAILED.txt"
        if failure.parent.exists() and not (failure.parent / "_COMPLETE").exists():
            failure.write_text(traceback.format_exc())
        raise
