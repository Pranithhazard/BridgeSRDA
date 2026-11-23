from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.checkpoint import (
    load_eval_model,
    load_training_checkpoint,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.data import (
    make_conditional_sb_dataset,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.validation import (
    build_validation_manifest,
    evaluate_denseobs_conditional_panel,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--weights-kind", choices=("raw", "ema"))
    parser.add_argument("--nfe-budget", type=int, default=20)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    config = load_config(args.config_path)
    work_root = root / config["output"]["work_root"]
    run_name = config["output"]["run_name"]
    checkpoint_dir = work_root / "checkpoints" / run_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.weights_kind is not None and args.checkpoint_path is None:
        parser.error("--weights-kind requires --checkpoint-path")
    if args.checkpoint_path is not None and args.weights_kind is not None:
        model, provenance = load_training_checkpoint(
            config=config,
            checkpoint_path=args.checkpoint_path,
            device=device,
            weights_kind=args.weights_kind,
        )
        label = (
            f"{args.weights_kind}_ep"
            f"{int(provenance['checkpoint_epoch']):04d}"
        )
    else:
        model, provenance = load_eval_model(
            config=config,
            checkpoint_dir=checkpoint_dir,
            device=device,
            explicit=args.checkpoint_path,
        )
        label = Path(
            str(provenance.get("weights_path", args.checkpoint_path))
        ).stem
    dataset = make_conditional_sb_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="valid",
        use_mixup=False,
        samples_per_epoch=None,
        deterministic_observations=True,
    )
    dataset.active_indices = list(range(len(dataset.all_hr_paths)))
    manifest = build_validation_manifest(
        dataset, config["validation_panel"]["starts"], None
    )
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
    output = (
        work_root
        / "metrics"
        / run_name
        / label
        / f"nfe{args.nfe_budget}_m{args.ensemble_size}"
    )
    if output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{output} exists; pass --allow-overwrite")
    output.mkdir(parents=True, exist_ok=True)
    common = dict(
        model=model,
        dataset=dataset,
        manifest=manifest,
        bridge=bridge,
        nfe_budget=args.nfe_budget,
        ensemble_size=args.ensemble_size,
        member_chunk_size=int(config["validation_panel"]["member_chunk_size"]),
        brownian_base_steps=int(config["validation_panel"]["brownian_base_steps"]),
        sampling_seed=int(config["validation_panel"]["sampling_seed"]),
        observation_seed=int(config["validation_panel"]["observation_seed"]),
        pv_min=float(config["dataset"]["pv_min"]),
        pv_max=float(config["dataset"]["pv_max"]),
        output_dir=output,
        weights_kind=str(provenance.get("weights_kind", label)),
        interpolation_method=str(config["conditioning"]["interpolation_method"]),
    )
    summary = evaluate_denseobs_conditional_panel(**common)

    (output / "metadata.json").write_text(
        json.dumps({"provenance": provenance, "summary": summary}, indent=2, default=str)
    )
    pd.DataFrame([summary]).to_csv(output / "metrics_summary.csv", index=False)
    (output / "_COMPLETE").write_text("ok\n")
    print(json.dumps({"output_dir": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
