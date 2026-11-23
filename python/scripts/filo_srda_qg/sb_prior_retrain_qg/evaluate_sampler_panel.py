from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.filo_srda_qg.sr_prior.data import make_sb_dataset
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.checkpoint import load_eval_model
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config
from src.filo_srda_qg.sb_prior_retrain_qg.validation import (
    build_validation_manifest,
    evaluate_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--nfe-budget", type=int, default=20)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--member-chunk-size", type=int)
    parser.add_argument("--brownian-base-steps", type=int)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--selection-label", default="validation_manifest")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    config = load_config(args.config_path)
    work_root = root / config["output"]["work_root"]
    run_name = str(config["output"]["run_name"])
    checkpoint_dir = work_root / "checkpoints" / run_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, provenance = load_eval_model(
        config=config,
        checkpoint_dir=checkpoint_dir,
        device=device,
        explicit=args.checkpoint_path,
    )
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(**config["bridge"])
    )
    dataset = make_sb_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="valid",
        use_mixup=False,
        samples_per_epoch=None,
    )
    dataset.active_indices = list(range(len(dataset.all_hr_paths)))
    manifest = build_validation_manifest(
        dataset,
        config["validation_panel"]["starts"],
        None,
    )
    weight_name = Path(str(provenance["weights_path"])).stem
    output_dir = (
        work_root
        / "metrics"
        / run_name
        / weight_name
        / args.selection_label
        / f"nfe{args.nfe_budget}_m{args.ensemble_size}"
    )
    if output_dir.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --allow-overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_panel(
        model=model,
        dataset=dataset,
        manifest=manifest,
        bridge=bridge,
        nfe_budget=args.nfe_budget,
        ensemble_size=args.ensemble_size,
        member_chunk_size=args.member_chunk_size
        or int(config["validation_panel"]["member_chunk_size"]),
        brownian_base_steps=args.brownian_base_steps
        or int(config["validation_panel"]["brownian_base_steps"]),
        sampling_seed=args.sampling_seed
        or int(config["validation_panel"]["sampling_seed"]),
        pv_min=float(config["dataset"]["pv_min"]),
        pv_max=float(config["dataset"]["pv_max"]),
        output_dir=output_dir,
        weights_kind=str(provenance.get("weights_kind", weight_name)),
    )
    metadata = {
        "config_path": str(Path(args.config_path).resolve()),
        "run_name": run_name,
        "provenance": provenance,
        "summary": summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    pd.DataFrame([summary]).to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "_COMPLETE").write_text("ok\n")
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
