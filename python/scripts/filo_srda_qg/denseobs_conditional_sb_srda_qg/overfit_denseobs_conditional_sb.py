from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.bridge import (
    denseobs_conditional_bridge_loss_at_time,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.conditioning import (
    build_dense_observation_condition,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.data import (
    make_conditional_sb_dataset,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.model import (
    make_denseobs_conditional_sb_model,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler import (
    sample_denseobs_conditional_bare_heun_exact_noise,
)
from src.filo_srda_qg.sr_prior.source import (
    build_bridge_source,
    fold_hr_target,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    make_time_assignment,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def group_loss(
    model: torch.nn.Module,
    bridge: ContinuousQuadraticBridge,
    assignment,
    source: torch.Tensor,
    target: torch.Tensor,
    observations: torch.Tensor,
    masks: torch.Tensor,
    group: int,
) -> torch.Tensor:
    total = source.new_zeros(())
    for slot in range(assignment.global_batch_size):
        tau, role = assignment.time(
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
        total = total + denseobs_conditional_bridge_loss_at_time(
            bridge,
            model,
            source=source,
            target=target,
            observations=observations,
            masks=masks,
            t=tau,
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
        base_config=config["dataset"],
        root_dir=Path(args.root_dir).resolve(),
        split="train",
        use_mixup=False,
        samples_per_epoch=2,
        deterministic_observations=True,
    )
    samples = [dataset[index] for index in range(2)]
    source = build_bridge_source(samples[0]["condition_lr"][None].to(device).float())
    target = fold_hr_target(samples[0]["hr"][None].to(device).float())
    observations = samples[0]["obs"][None].to(device).float()
    masks = samples[0]["mask"][None].to(device).float()
    other_observations = samples[1]["obs"][None].to(device).float()
    other_masks = samples[1]["mask"][None].to(device).float()
    model = make_denseobs_conditional_sb_model(config["model"]).to(device)
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
    local_config = dict(config)
    local_config["train"] = dict(config["train"])
    local_config["train"]["gradient_accumulation_steps"] = 32
    assignment = make_time_assignment(config=local_config, world_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["overfit"]["lr"]))
    model.eval()
    with torch.no_grad():
        initial = float(
            group_loss(
                model,
                bridge,
                assignment,
                source,
                target,
                observations,
                masks,
                0,
            )
        )
    model.train()
    for step in range(int(config["overfit"]["steps"])):
        loss = group_loss(
            model,
            bridge,
            assignment,
            source,
            target,
            observations,
            masks,
            step,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["train"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite overfit gradient")
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={loss.item():.8g}")
    model.eval()
    with torch.no_grad():
        final = float(
            group_loss(
                model,
                bridge,
                assignment,
                source,
                target,
                observations,
                masks,
                0,
            )
        )
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
        tau = torch.tensor([0.5], device=device)
        noise = torch.randn_like(source)
        state = bridge.interpolate(source, target, tau, noise)
        correct_dense = build_dense_observation_condition(source, observations, masks)
        shuffled_dense = build_dense_observation_condition(
            source, other_observations, other_masks
        )
        missing_observations = observations.clone()
        missing_masks = masks.clone()
        missing_observations[:, :2] = 0
        missing_masks[:, :2] = 0
        fallback_dense = build_dense_observation_condition(
            source, missing_observations, missing_masks
        )
        correct_prediction = model(state, source, correct_dense, tau)
        shuffled_prediction = model(state, source, shuffled_dense, tau)
        fallback_prediction = model(state, source, fallback_dense, tau)
    ratio = final / max(initial, 1.0e-12)
    diversity = float(
        (result.samples[:, 0] - result.samples[:, 1]).square().mean().sqrt()
    )
    shuffled_change = float(
        (correct_prediction - shuffled_prediction).square().mean().sqrt()
    )
    fallback_change = float(
        (correct_prediction - fallback_prediction).square().mean().sqrt()
    )
    print(f"initial={initial:.8g} final={final:.8g} ratio={ratio:.8g}")
    print(
        f"sample_shape={tuple(result.samples.shape)} diversity={diversity:.8g} "
        f"shuffled_change={shuffled_change:.8g} fallback_change={fallback_change:.8g}"
    )
    if ratio > float(config["overfit"]["max_final_ratio"]):
        raise RuntimeError("tiny dense-observation conditional overfit criterion failed")
    if not torch.isfinite(result.samples).all() or diversity <= 0:
        raise RuntimeError("invalid overfit sample/diversity")
    threshold = float(config["overfit"].get("observation_change_min_rms", 1.0e-6))
    if max(shuffled_change, fallback_change) <= threshold:
        raise RuntimeError("tiny model did not measurably use dense observations")


if __name__ == "__main__":
    main()
