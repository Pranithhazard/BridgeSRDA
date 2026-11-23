from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.filo_srda_qg.sr_prior.data import make_sb_dataset
from src.filo_srda_qg.sr_prior.model import make_sb_prior_model
from src.filo_srda_qg.sr_prior.source import build_bridge_source, fold_hr_target
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    make_time_assignment,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import sample_bare_heun_exact_noise


def group_loss(model, bridge, assignment, source, target, group: int) -> torch.Tensor:
    total = source.new_zeros(())
    for slot in range(assignment.global_batch_size):
        t, role = assignment.time(
            epoch=0,
            optimizer_group=group,
            rank=0,
            accumulation_index=slot,
            device=source.device,
        )
        if role.kind == "interior":
            generator = torch.Generator(device=source.device).manual_seed(42 + 1009 * group + slot)
            noise = torch.randn(source.shape, device=source.device, generator=generator)
        else:
            noise = torch.zeros_like(source)
        total = total + bridge.loss_at_time(
            model, source=source, target=target, t=t, noise=noise
        )
    return total / assignment.global_batch_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    args = parser.parse_args()
    config = load_config(args.config_path)
    root = Path(args.root_dir).resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = make_sb_dataset(
        base_config=config["dataset"], root_dir=root, split="train",
        use_mixup=False, samples_per_epoch=1,
    )
    sample = dataset[0]
    source = build_bridge_source(sample["condition_lr"][None].to(device).float())
    target = fold_hr_target(sample["hr"][None].to(device).float())
    model = make_sb_prior_model(config["model"]).to(device)
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(**config["bridge"])
    )
    config = dict(config)
    config["train"] = dict(config["train"])
    config["train"]["gradient_accumulation_steps"] = int(config["bridge"]["interior_strata"]) + 2
    assignment = make_time_assignment(config=config, world_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["overfit"]["lr"]))
    model.eval()
    with torch.no_grad():
        initial = float(group_loss(model, bridge, assignment, source, target, 0).item())
    model.train()
    steps = int(config["overfit"]["steps"])
    for step in range(steps):
        value = group_loss(model, bridge, assignment, source, target, step)
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        print(f"step={step + 1} loss={value.item():.8g}")
    model.eval()
    with torch.no_grad():
        final = float(group_loss(model, bridge, assignment, source, target, 0).item())
        result = sample_bare_heun_exact_noise(
            model=model, source=source, bridge=bridge, ensemble_size=2,
            nfe_budget=4, member_chunk_size=1, sampling_seed=42,
            brownian_base_steps=100,
        )
    ratio = final / max(initial, 1.0e-12)
    diversity = float((result.samples[:, 0] - result.samples[:, 1]).square().mean().sqrt().item())
    print(f"initial={initial:.8g} final={final:.8g} ratio={ratio:.8g}")
    print(f"sample_shape={tuple(result.samples.shape)} nfe={result.actual_nfe} diversity={diversity:.8g}")
    if ratio >= float(config["overfit"]["max_final_ratio"]):
        raise RuntimeError("tiny continuous-time overfit criterion failed")
    if not torch.isfinite(result.samples).all() or diversity <= 0.0:
        raise RuntimeError("sampling sanity failed")


if __name__ == "__main__":
    main()
