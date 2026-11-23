from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.filo_srda_qg.conditional_sb_srda_qg.bridge import conditional_bridge_loss_at_time
from src.filo_srda_qg.conditional_sb_srda_qg.data import make_conditional_sb_dataset
from src.filo_srda_qg.conditional_sb_srda_qg.model import make_conditional_sb_model
from src.filo_srda_qg.conditional_sb_srda_qg.sampler import sample_conditional_bare_heun_exact_noise
from src.filo_srda_qg.sr_prior.source import build_bridge_source, fold_hr_target
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    make_time_assignment,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def group_loss(model, bridge, assignment, source, target, observations, masks, group: int):
    """Average one complete optimizer-group time contract.

    This mirrors the 071B prior overfit check: one exact t=0 role, one
    deterministic t=1 role and all interior strata are optimized together.
    Testing isolated roles is too noisy for this conditional model and can fail
    even when the actual DDP objective is well formed.
    """
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
            generator = torch.Generator(device=source.device).manual_seed(
                42 + 1009 * group + slot
            )
            noise = torch.randn(source.shape, device=source.device, generator=generator)
        else:
            noise = torch.zeros_like(source)
        total = total + conditional_bridge_loss_at_time(
            bridge,
            model,
            source=source,
            target=target,
            observations=observations,
            masks=masks,
            t=t,
            noise=noise,
        )
    return total / assignment.global_batch_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    args = parser.parse_args()
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = make_conditional_sb_dataset(
        base_config=config["dataset"], root_dir=Path(args.root_dir).resolve(),
        split="train", use_mixup=False, samples_per_epoch=1,
        deterministic_observations=True,
    )
    sample = dataset[0]
    source = build_bridge_source(sample["condition_lr"][None].to(device).float())
    target = fold_hr_target(sample["hr"][None].to(device).float())
    observations = sample["obs"][None].to(device).float()
    masks = sample["mask"][None].to(device).float()
    model = make_conditional_sb_model(config["model"]).to(device)
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
    local_config = dict(config)
    local_config["train"] = dict(config["train"])
    local_config["train"]["gradient_accumulation_steps"] = 32
    assignment = make_time_assignment(config=local_config, world_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["overfit"]["lr"]))
    model.eval()
    with torch.no_grad():
        initial = float(
            group_loss(model, bridge, assignment, source, target, observations, masks, 0)
        )
    model.train()
    for step in range(int(config["overfit"]["steps"])):
        loss = group_loss(
            model, bridge, assignment, source, target, observations, masks, step
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 5 == 0:
            print(f"step={step + 1} loss={loss.item():.8g}")
    model.eval()
    with torch.no_grad():
        final = float(
            group_loss(model, bridge, assignment, source, target, observations, masks, 0)
        )
        result = sample_conditional_bare_heun_exact_noise(
            model=model, source=source, observations=observations, masks=masks,
            bridge=bridge, ensemble_size=2, nfe_budget=4, member_chunk_size=1,
            sampling_seed=42, brownian_base_steps=20,
        )
    ratio = final / max(initial, 1e-12)
    diversity = float((result.samples[:, 0] - result.samples[:, 1]).square().mean().sqrt())
    print(f"initial={initial:.8g} final={final:.8g} ratio={ratio:.8g}")
    print(f"sample_shape={tuple(result.samples.shape)} diversity={diversity:.8g}")
    if ratio >= float(config["overfit"]["max_final_ratio"]):
        raise RuntimeError("tiny conditional overfit criterion failed")


if __name__ == "__main__":
    main()
