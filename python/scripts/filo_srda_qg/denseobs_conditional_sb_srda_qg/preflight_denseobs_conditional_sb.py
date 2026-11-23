from __future__ import annotations

import argparse
import json

import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.bridge import (
    denseobs_conditional_bridge_loss_at_time,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.conditioning import (
    build_dense_observation_condition,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.model import (
    make_denseobs_conditional_sb_model,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler import (
    sample_denseobs_conditional_bare_heun_exact_noise,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def regular_masks(batch: int, device: torch.device) -> torch.Tensor:
    masks = torch.zeros(batch, 3, 1, 16, 64, 128, device=device)
    for index in range(batch):
        oy, ox = index % 8, (3 * index) % 8
        masks[index, :2, 0, :, oy::8, ox::8] = 1
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bf16 = (
        device.type == "cuda"
        and config["train"]["precision"] == "bfloat16"
        and torch.cuda.is_bf16_supported()
    )
    model = make_denseobs_conditional_sb_model(config["model"]).to(device).train()
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
    source = torch.rand(1, 9, 16, 64, 128, device=device)
    target = torch.rand_like(source)
    masks = regular_masks(1, device)
    observations = masks * torch.rand_like(masks)
    dense = build_dense_observation_condition(source, observations, masks)
    tau = torch.tensor([0.4375], device=device)
    noise = torch.randn_like(source)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        loss = denseobs_conditional_bridge_loss_at_time(
            bridge,
            model,
            source=source,
            target=target,
            observations=observations,
            masks=masks,
            t=tau,
            noise=noise,
        )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("missing or non-finite gradients")
    model.eval()
    result = sample_denseobs_conditional_bare_heun_exact_noise(
        model=model,
        source=source,
        observations=observations,
        masks=masks,
        bridge=bridge,
        ensemble_size=2,
        nfe_budget=20,
        member_chunk_size=1,
        sampling_seed=42,
        brownian_base_steps=100,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "device": str(device),
        "bf16": bf16,
        "parameter_count": count,
        "expected_parameter_count": 90_832_745,
        "dense_shape": list(dense.shape),
        "loss": float(loss),
        "sample_shape": list(result.samples.shape),
        "actual_nfe": result.actual_nfe,
        "finite": bool(torch.isfinite(result.samples).all()),
    }
    print(json.dumps(report, indent=2))
    if count != 90_832_745:
        raise AssertionError(f"unexpected model parameter count: {count:,}")
    if report["dense_shape"] != [1, 2, 16, 64, 128]:
        raise AssertionError("invalid dense-observation shape")
    if report["sample_shape"] != [1, 2, 9, 16, 64, 128]:
        raise AssertionError("invalid NFE20 sample shape")


if __name__ == "__main__":
    main()
