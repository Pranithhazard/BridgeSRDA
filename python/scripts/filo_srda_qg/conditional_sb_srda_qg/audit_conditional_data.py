from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.filo_srda_qg.conditional_sb_srda_qg.conditioning import (
    build_observation_condition,
)
from src.filo_srda_qg.conditional_sb_srda_qg.data import make_conditional_sb_dataset
from src.filo_srda_qg.sr_prior.source import build_bridge_source
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    args = parser.parse_args()
    config = load_config(args.config_path)
    dataset = make_conditional_sb_dataset(
        base_config=config["dataset"], root_dir=Path(args.root_dir).resolve(),
        split="train", use_mixup=True, samples_per_epoch=1,
    )
    sample = dataset[0]
    source = build_bridge_source(sample["condition_lr"][None])
    condition = build_observation_condition(
        source, sample["obs"][None], sample["mask"][None]
    )
    report = {
        "source_shape": list(source.shape),
        "observation_shape": list(sample["obs"].shape),
        "mask_shape": list(sample["mask"].shape),
        "condition_shape": list(condition.shape),
        "observed_fraction": float(sample["mask"].mean()),
        "future_mask_nonzero": int(torch.count_nonzero(sample["mask"][2])),
        "masked_innovation_nonzero": int(torch.count_nonzero(condition[:, :3])),
        "mask_channels_match": bool(torch.equal(condition[:, 3:], sample["mask"][None, :, 0])),
        "obs_mask_index": int(sample["metadata"]["obs_mask_index"]),
    }
    print(json.dumps(report, indent=2))
    if report["condition_shape"] != [1, 6, 16, 64, 128]:
        raise AssertionError("invalid observation condition shape")
    if report["future_mask_nonzero"] != 0 or not report["mask_channels_match"]:
        raise AssertionError("future/mask contract failed")


if __name__ == "__main__":
    main()
