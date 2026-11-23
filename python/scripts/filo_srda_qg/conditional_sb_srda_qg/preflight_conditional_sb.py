from __future__ import annotations

import argparse

import torch

from src.filo_srda_qg.conditional_sb_srda_qg.bridge import conditional_bridge_loss_at_time
from src.filo_srda_qg.conditional_sb_srda_qg.model import make_conditional_sb_model
from src.filo_srda_qg.conditional_sb_srda_qg.sampler import (
    sample_conditional_bare_heun_exact_noise,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and config["train"]["precision"] == "bfloat16"
    if bf16 and not torch.cuda.is_bf16_supported():
        bf16 = False
    model = make_conditional_sb_model(config["model"]).to(device).train()
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
    source = torch.rand(1, 9, 16, 64, 128, device=device)
    target = torch.rand_like(source)
    observations = torch.rand(1, 3, 1, 16, 64, 128, device=device)
    masks = (torch.rand_like(observations) > 0.9).float()
    masks[:, 2] = 0.0
    observations[:, 2] = 0.0
    t = torch.tensor([0.4375], device=device)
    noise = torch.randn_like(source)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        loss = conditional_bridge_loss_at_time(
            bridge, model, source=source, target=target,
            observations=observations, masks=masks, t=t, noise=noise,
        )
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("missing or non-finite gradients")
    model.eval()
    result = sample_conditional_bare_heun_exact_noise(
        model=model, source=source, observations=observations, masks=masks,
        bridge=bridge, ensemble_size=2, nfe_budget=4, member_chunk_size=1,
        sampling_seed=42, brownian_base_steps=20,
    )
    count = sum(p.numel() for p in model.parameters())
    print(f"device={device} bf16={bf16} parameter_count={count:,}")
    print(f"loss={loss.item():.8g} sample_shape={tuple(result.samples.shape)}")
    print(f"nfe={result.actual_nfe} peak_gpu_bytes={result.peak_gpu_bytes}")


if __name__ == "__main__":
    main()
