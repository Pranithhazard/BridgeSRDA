from __future__ import annotations

import argparse
import pathlib

import torch
import yaml
from src.filo_srda_qg.diffusion_srda_qg.diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
)
from src.filo_srda_qg.diffusion_srda_qg.model import make_diffusion_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--n-head", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(pathlib.Path(args.config_path).read_text())
    if args.activation_checkpointing:
        config["model"]["activation_checkpointing"] = True
    if args.model_width is not None:
        width_key = (
            "base_width"
            if config["model"].get("model_name") == "DiffSRDAUNet3D"
            else "width"
        )
        config["model"][width_key] = args.model_width
    if args.n_head is not None:
        config["model"]["n_head"] = args.n_head
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = make_diffusion_model(config["model"]).to(device).train()
    diffusion = GaussianDiffusion(DiffusionConfig.from_dict(config["diffusion"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    batch = int(config["preflight"]["batch_size"])
    target = torch.randn(batch, 9, 16, 64, 128, device=device)
    lr = torch.randn(batch, 3, 1, 4, 16, 32, device=device)
    obs = torch.randn(batch, 3, 1, 16, 64, 128, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss = diffusion.training_loss(
        model, target=target, condition_lr=lr, obs=obs
    )
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["train"]["gradient_clip_norm"])
    )
    gradient_count = sum(
        parameter.grad is not None for parameter in model.parameters()
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameters_finite = all(
        torch.isfinite(parameter).all().item()
        for parameter in model.parameters()
    )
    print(f"device={device} parameter_count={sum(p.numel() for p in model.parameters()):,}")
    print(
        f"loss={loss.item():.8g} gradient_tensors={gradient_count} "
        f"gradient_norm={gradient_norm.item():.8g} optimizer_step=True"
    )
    print(
        "finite="
        f"{torch.isfinite(loss).item() and torch.isfinite(gradient_norm).item() and parameters_finite}"
    )
    if device.type == "cuda":
        print(f"peak_gpu_bytes={torch.cuda.max_memory_allocated(device)}")
        print(f"peak_gpu_reserved_bytes={torch.cuda.max_memory_reserved(device)}")


if __name__ == "__main__":
    main()
