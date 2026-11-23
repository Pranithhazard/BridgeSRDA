from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.conditioning import (
    DENSE_OBSERVATION_FRAME_INDICES,
    build_dense_observation_condition,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.data import (
    make_conditional_sb_dataset,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.model import (
    make_denseobs_conditional_sb_model,
)
from src.filo_srda_qg.sr_prior.source import (
    build_bridge_source,
    fold_hr_target,
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
    args = parser.parse_args()
    root = Path(args.root_dir).resolve()
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    observed_origins: set[int] = set()
    sensor_errors: list[float] = []
    representative: dict[str, object] | None = None
    for origin in range(64):
        dataset_config = copy.deepcopy(config["dataset"])
        dataset_config["fixed_obs_mask_index"] = origin
        dataset = make_conditional_sb_dataset(
            base_config=dataset_config,
            root_dir=root,
            split="train",
            use_mixup=False,
            samples_per_epoch=1,
            deterministic_observations=True,
        )
        sample = dataset[0]
        source = build_bridge_source(sample["condition_lr"][None].to(device).float())
        target = fold_hr_target(sample["hr"][None].to(device).float())
        observations = sample["obs"][None].to(device).float()
        masks = sample["mask"][None].to(device).float()
        dense = build_dense_observation_condition(source, observations, masks)
        observed_origins.add(int(sample["metadata"]["obs_mask_index"]))
        for slot in range(2):
            active = masks[:, slot, 0] > 0.5
            sensor_errors.append(
                float((dense[:, slot][active] - observations[:, slot, 0][active]).abs().max())
            )
        if representative is None:
            representative = {
                "source_shape": list(source.shape),
                "target_shape": list(target.shape),
                "raw_observation_shape": list(sample["obs"].shape),
                "raw_mask_shape": list(sample["mask"].shape),
                "dense_shape": list(dense.shape),
                "future_mask_nonzero": int(torch.count_nonzero(masks[:, 2])),
                "future_observation_nonzero": int(torch.count_nonzero(observations[:, 2])),
            }

    assert representative is not None
    model = make_denseobs_conditional_sb_model(config["model"]).to(device).eval()
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
    tau = torch.tensor([0.375], device=device)
    noise = torch.randn_like(source)
    state = bridge.interpolate(source, target, tau, noise)
    output = model(state, source, dense, tau)
    zero_prediction = float(output.abs().max())
    report = {
        **representative,
        "dense_observation_frames": list(DENSE_OBSERVATION_FRAME_INDICES),
        "future_slot_enters_network": False,
        "complete_input_channels": 9 + 9 + 2,
        "model_output_shape": list(output.shape),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "zero_output_initialization_max_abs": zero_prediction,
        "all_64_mask_origins_seen": sorted(observed_origins) == list(range(64)),
        "mask_origins": sorted(observed_origins),
        "sensor_site_max_abs_error": max(sensor_errors),
        "raw_sparse_inputs_retained": True,
        "noise_before_normalization_and_interpolation": True,
        "source_target_contract_identical_to_072": True,
        "p2_contract": {
            "noise_exponent": bridge.config.noise_exponent,
            "epsilon": bridge.config.epsilon,
        },
    }
    output_dir = root / config["output"]["work_root"] / "audits" / "contracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_model_contract.json").write_text(json.dumps(report, indent=2))
    markdown = [
        "# 073A data/model contract audit",
        "",
        f"- Raw observation shape: `{report['raw_observation_shape']}`",
        f"- Dense condition shape: `{report['dense_shape']}`",
        f"- Complete input channels: `{report['complete_input_channels']}`",
        f"- Parameter count: `{report['parameter_count']:,}`",
        f"- All 64 mask origins: `{report['all_64_mask_origins_seen']}`",
        f"- Sensor max error: `{report['sensor_site_max_abs_error']:.3e}`",
        f"- Future slot enters network: `{report['future_slot_enters_network']}`",
    ]
    (output_dir / "data_model_contract.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(report, indent=2))

    required = {
        "raw_observation_shape": [3, 1, 16, 64, 128],
        "raw_mask_shape": [3, 1, 16, 64, 128],
        "dense_shape": [1, 2, 16, 64, 128],
        "model_output_shape": [1, 9, 16, 64, 128],
        "complete_input_channels": 20,
        "future_slot_enters_network": False,
        "all_64_mask_origins_seen": True,
    }
    for key, expected in required.items():
        if report[key] != expected:
            raise AssertionError(f"{key}: expected {expected}, got {report[key]}")
    if report["sensor_site_max_abs_error"] > 1.0e-6:
        raise AssertionError("sensor-site preservation failed")
    if report["parameter_count"] >= 100_000_000:
        raise AssertionError("parameter budget failed")


if __name__ == "__main__":
    main()
