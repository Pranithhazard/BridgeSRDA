from __future__ import annotations

import argparse
import copy
import json
import pathlib
import random

import numpy as np
import torch
import yaml

from src.filo_srda_qg.diffusion_srda_qg.data import (
    diffusion_values_to_zero_one,
    make_diffusion_dataset,
)


def reset(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    root = pathlib.Path(args.root_dir).resolve()
    with open(args.config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    train = make_diffusion_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="train",
        samples_per_epoch=8,
    )
    no_mix_config = copy.deepcopy(config["dataset"])
    no_mix_config["use_mixup"] = False
    train_no_mix = make_diffusion_dataset(
        base_config=no_mix_config,
        root_dir=root,
        split="train",
        samples_per_epoch=8,
    )
    valid = make_diffusion_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="valid",
        samples_per_epoch=8,
    )
    reset(1234)
    train_sample = train[0]
    reset(1234)
    train_no_mix_sample = train_no_mix[0]
    reset(1234)
    valid_first = valid[0]
    reset(1234)
    valid_second = valid[0]
    observation_hr = train_sample["hr"][
        :: int(config["dataset"]["lr_and_obs_time_interval"])
    ]
    if observation_hr.shape != train_sample["obs"].shape:
        raise AssertionError(
            "Observation/HR time alignment mismatch: "
            f"{tuple(train_sample['obs'].shape)} != {tuple(observation_hr.shape)}"
        )
    observed = train_sample["mask"] > 0
    sparse_obs = train_sample.get("sparse_obs_zero_one", train_sample["obs"])
    fill_mode = str(
        config["dataset"].get(
            "observation_fill_mode", "sparse_sentinel"
        )
    )
    scaling = str(config["dataset"].get("data_scaling", "zero_one"))
    paired_observation_hr = (
        diffusion_values_to_zero_one(
            observation_hr, data_scaling=scaling
        )
        if "sparse_obs_zero_one" in train_sample
        else observation_hr
    )
    missing_value = float(
        config["dataset"].get("diffusion_missing_value", 0.0)
    )
    report = {
        "train_shapes": {
            key: list(train_sample[key].shape)
            for key in ["condition_lr", "source_lr", "obs", "hr"]
        },
        "train_condition_differs_from_source": bool(
            not torch.equal(train_sample["condition_lr"], train_sample["source_lr"])
        ),
        "mixup_preserves_paired_source_and_hr": bool(
            torch.equal(train_sample["source_lr"], train_no_mix_sample["source_lr"])
            and torch.equal(train_sample["hr"], train_no_mix_sample["hr"])
        ),
        "observations_are_paired_with_hr": bool(
            (
                sparse_obs[train_sample["mask"] > 0]
                - paired_observation_hr[observed]
            )
            .abs()
            .max()
            .item()
            < 0.05
        ),
        "valid_condition_equals_source": bool(
            torch.equal(valid_first["condition_lr"], valid_first["source_lr"])
        ),
        "valid_repeat_is_deterministic": bool(
            all(
                torch.equal(valid_first[key], valid_second[key])
                for key in ["condition_lr", "source_lr", "obs", "hr"]
            )
        ),
        "future_observation_slot_is_empty": bool(
            torch.count_nonzero(train_sample["mask"][-1]).item() == 0
            and torch.all(train_sample["obs"][-1] == missing_value).item()
        ),
        "data_scaling": scaling,
        "diffusion_missing_value": missing_value,
        "train_hr_min": float(train_sample["hr"].min().item()),
        "train_hr_max": float(train_sample["hr"].max().item()),
        "missing_observations_are_sentinel": bool(
            (
                torch.all(train_sample["obs"][~observed] == missing_value).item()
                if fill_mode == "sparse_sentinel"
                else True
            )
        ),
        "observation_fill_mode": fill_mode,
        "filled_observed_slots_are_dense": bool(
            torch.all(train_sample["obs"][:-1] != missing_value).item()
            if fill_mode == "bicubic_horizontal"
            else True
        ),
    }
    if not report["valid_condition_equals_source"]:
        raise AssertionError("Validation LR must be unmixed")
    if not report["train_condition_differs_from_source"]:
        raise AssertionError("Training mixup did not change condition_lr")
    if not report["mixup_preserves_paired_source_and_hr"]:
        raise AssertionError("Training mixup changed paired source LR or HR")
    if not report["observations_are_paired_with_hr"]:
        raise AssertionError("Training observations are not paired with HR")
    if not report["valid_repeat_is_deterministic"]:
        raise AssertionError("Validation sample is not deterministic after RNG reset")
    if not report["future_observation_slot_is_empty"]:
        raise AssertionError("Future observation slot is not empty")
    if not report["missing_observations_are_sentinel"]:
        raise AssertionError("Missing observations do not use the configured sentinel")
    if not report["filled_observed_slots_are_dense"]:
        raise AssertionError("Bicubic observation filling left sentinel holes")
    output = json.dumps(report, indent=2)
    print(output)
    if args.output_path:
        path = pathlib.Path(args.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)


if __name__ == "__main__":
    main()
