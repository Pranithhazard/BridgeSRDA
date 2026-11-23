"""Run the four Appendix F conditions with a two-cycle check before full cycling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

CASES = (("hr_letkf_m30_optimized", 0, "letkf", 30),
         ("bridge_compiled_bf16_nfe20_m30", 1, "bridge", 30),
         ("bridge_compiled_bf16_nfe20_m10", 2, "bridge", 10),
         ("bridge_compiled_bf16_nfe20_m5", 3, "bridge", 5))
OWNED_WORKERS = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def identities(root):
    from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.benchmark_inference_timing import digest
    manifest = root / "manifests/assets.json"
    assets = json.loads(manifest.read_text())
    checkpoint = assets["checkpoint_assets"]["bridge_d"]
    config_root = root / "python/configs"
    runtime_root = root / "python/src"
    script_root = root / "python/scripts"
    paths = [
        *config_root.rglob("*.yml"),
        *runtime_root.rglob("*.py"),
        *script_root.rglob("*.py"),
        manifest,
        root / checkpoint["path"],
    ]
    hashes = {str(path.relative_to(root)): digest(path) for path in sorted(paths)}
    if hashes[checkpoint["path"]] != checkpoint["historical_sha256"]:
        raise ValueError("Appendix F requires the recorded BridgeSRDA-D checkpoint")
    return hashes


def assert_idle():
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=15).strip()
    if processes:
        raise RuntimeError(f"GPUs not idle before stage: {processes}")


def stage(root, output, name, analyses, original_identities):
    if identities(root) != original_identities:
        raise RuntimeError("Source identities changed between smoke and full stages")
    assert_idle()
    workers = []
    started = time.monotonic()
    stage_dir = output / name
    stage_dir.mkdir()
    worker = root / "python/scripts/filo_srda_qg/denseobs_conditional_sb_srda_qg/benchmark_cycling_worker.py"
    for case, gpu, method, members in CASES:
        env = dict(os.environ)
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(root / "python"),
                   PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                   OPENBLAS_NUM_THREADS="8", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                   TORCHINDUCTOR_COMPILE_THREADS="1",
                   TORCHINDUCTOR_CACHE_DIR=str(output / "cache" / case / "inductor"),
                   TRITON_CACHE_DIR=str(output / "cache" / case / "triton"))
        command = [sys.executable, str(worker),
                   "--root", str(root), "--output", str(stage_dir / case), "--method", method,
                   "--ensemble-size", str(members), "--seed", "9991", "--max-analyses", str(analyses)]
        log = (stage_dir / f"{case}.log").open("x")
        process = subprocess.Popen(command, env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        OWNED_WORKERS.append(process)
        workers.append(dict(case=case, gpu=gpu, command=command, process=process, log=log))
    deadline = started + (600 if analyses == 2 else 90 * 60)
    while any(w["process"].poll() is None for w in workers):
        records = [dict(case=w["case"], gpu=w["gpu"], pid=w["process"].pid,
                        returncode=w["process"].poll(), command=w["command"]) for w in workers]
        save(stage_dir / "processes.json", records)
        if time.monotonic() > deadline:
            for w in workers:
                if w["process"].poll() is None:
                    os.killpg(w["process"].pid, signal.SIGTERM)
            time.sleep(5)
            for w in workers:
                if w["process"].poll() is None:
                    os.killpg(w["process"].pid, signal.SIGKILL)
            raise TimeoutError(f"Owned {name} stage exceeded time cap")
        time.sleep(5)
    save(stage_dir / "processes.json", [dict(case=w["case"], gpu=w["gpu"], pid=w["process"].pid,
                                            returncode=w["process"].returncode, command=w["command"]) for w in workers])
    results = {}
    for w in workers:
        w["log"].close()
        case_dir = stage_dir / w["case"]
        if w["process"].returncode != 0 or not (case_dir / "_COMPLETE").is_file():
            results[w["case"]] = dict(status="failed", returncode=w["process"].returncode)
        else:
            summary = json.loads((case_dir / "summary.json").read_text())
            if summary["analyses"] != analyses:
                raise RuntimeError("Incomplete analysis count despite completion marker")
            results[w["case"]] = summary
    save(stage_dir / "results.json", results)
    if any(r.get("status") == "failed" for r in results.values()):
        raise RuntimeError(f"{name} failed; inspect retained logs/results")
    manifests = [json.loads((stage_dir / case / "manifest.json").read_text()) for case, *_ in CASES]
    if len({m["observation_float32_nanfilled_sha256"] for m in manifests}) != 1:
        raise RuntimeError("Observation streams differ across methods")
    save(stage_dir / "stage_complete.json", dict(wall_seconds=time.monotonic() - started,
                                                 observation_stream_matched=True, analyses_per_case=analyses))
    return results


def main(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "controller.json", dict(pid=os.getpid(), started_unix=time.time(),
                                         requested_seed=9991, cases=CASES, normal_cycling=True,
                                         shared_host_resources="Four simultaneous single-GPU jobs share CPU and filesystem; core and evaluation timing kept separate."))
    source_ids = identities(root)
    save(output / "source_manifest.json", source_ids)
    stage(root, output, "smoke", 2, source_ids)
    save(output / "SMOKE_PASSED.json", dict(all_four_cases_passed=True, next="full199"))
    results = stage(root, output, "full", 199, source_ids)
    save(output / "summary.json", results)
    (output / "_COMPLETE").write_text("all four full recursive runs complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Controller received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main(args)
    except Exception:
        for process in OWNED_WORKERS:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        directory = Path(args.output)
        if directory.is_dir():
            save(directory / "SUITE_FAILED.json", dict(error=traceback.format_exc()))
        raise
