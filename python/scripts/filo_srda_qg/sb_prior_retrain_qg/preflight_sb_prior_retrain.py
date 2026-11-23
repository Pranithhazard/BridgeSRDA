from __future__ import annotations

import argparse

import torch

from src.filo_srda_qg.sr_prior.model import make_sb_prior_model
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config
from src.filo_srda_qg.sb_prior_retrain_qg.ema import EMA
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import sample_bare_heun_exact_noise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = str(config["train"]["precision"])
    bf16 = device.type == "cuda" and precision == "bfloat16"
    if bf16 and not torch.cuda.is_bf16_supported():
        print("BF16 requested but unsupported; preflight falling back to full FP32")
        bf16 = False
    model = make_sb_prior_model(config["model"]).to(device).train()
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(**config["bridge"])
    )
    source = torch.rand(1, 9, 16, 64, 128, device=device)
    target = torch.rand_like(source)
    noise = torch.randn_like(source)
    t = torch.tensor([0.4375], device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        loss = bridge.loss_at_time(
            model, source=source, target=target, t=t, noise=noise
        )
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise FloatingPointError("Missing or non-finite full-model gradients")
    gradient_norm = torch.sqrt(sum(value.float().square().sum() for value in gradients))
    ema = EMA(model, decay=float(config["train"]["ema_decay"]))
    ema.update(model)
    model.eval()
    result = sample_bare_heun_exact_noise(
        model=model,
        source=source,
        bridge=bridge,
        ensemble_size=2,
        nfe_budget=4,
        member_chunk_size=1,
        sampling_seed=42,
        brownian_base_steps=100,
    )
    if not torch.isfinite(result.samples).all():
        raise FloatingPointError("Non-finite sampler output")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} bf16={bf16} parameter_count={parameter_count:,}")
    print(f"loss={loss.item():.8g} gradient_norm={gradient_norm.item():.8g}")
    print(
        f"sample_shape={tuple(result.samples.shape)} nfe={result.actual_nfe} "
        f"diversity={(result.samples[:, 0] - result.samples[:, 1]).square().mean().sqrt().item():.8g}"
    )
    if device.type == "cuda":
        print(f"peak_gpu_bytes={torch.cuda.max_memory_allocated(device)}")


if __name__ == "__main__":
    main()
