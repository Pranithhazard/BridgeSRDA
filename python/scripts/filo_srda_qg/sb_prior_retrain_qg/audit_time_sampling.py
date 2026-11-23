from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    make_time_assignment,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-path")
    args = parser.parse_args()
    config = load_config(args.config_path)
    assignment = make_time_assignment(config=config, world_size=args.world_size)
    roles = assignment.roles(epoch=0, optimizer_group=0)
    counts = Counter(role.kind for role in roles)
    times = []
    for accumulation_index in range(assignment.accumulation_steps):
        for rank in range(args.world_size):
            t, role = assignment.time(
                epoch=0,
                optimizer_group=0,
                rank=rank,
                accumulation_index=accumulation_index,
                device=torch.device("cpu"),
            )
            times.append((float(t.item()), role.kind, role.stratum))
    expected_counts = {
        "endpoint_zero": 1,
        "endpoint_one": 1,
        "interior": int(config["bridge"]["interior_strata"]),
    }
    if counts != expected_counts:
        raise RuntimeError(f"Invalid global role counts: {counts}")
    interior = [value for value, kind, _ in times if kind == "interior"]
    if not all(0.0 < value < 1.0 for value in interior):
        raise RuntimeError("Interior times escaped (0,1)")
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(**config["bridge"])
    )
    source = torch.randn(1, 9, 2, 3, 4)
    target = torch.randn_like(source)
    noisy = torch.randn_like(source)
    t1 = torch.ones(1)
    endpoint = bridge.target_drift(source, target, t1, noisy)
    expected = -source + 2.0 * target
    endpoint_error = float((endpoint - expected).abs().max().item())
    if endpoint_error != 0.0:
        raise RuntimeError("t=1 target contains stochastic noise")
    payload = {
        "global_batch_size": assignment.global_batch_size,
        "mode": assignment.mode,
        "dense_grid_intervals": assignment.dense_grid_intervals,
        "counts": dict(counts),
        "interior_min": min(interior),
        "interior_max": max(interior),
        "unique_interior_strata": len({stratum for _, kind, stratum in times if kind == "interior"}),
        "t1_conditional_mean_max_abs_error": endpoint_error,
        "roles": [
            {"time": value, "kind": kind, "stratum": stratum}
            for value, kind, stratum in times
        ],
    }
    print(json.dumps(payload, indent=2))
    if args.output_path:
        path = Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
