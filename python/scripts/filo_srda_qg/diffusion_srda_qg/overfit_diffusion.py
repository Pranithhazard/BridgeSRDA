from __future__ import annotations

import argparse
import pathlib

import torch
import yaml
from src.filo_srda_qg.diffusion_srda_qg.data import make_diffusion_dataset
from src.filo_srda_qg.diffusion_srda_qg.diffusion import DiffusionConfig, GaussianDiffusion
from src.filo_srda_qg.diffusion_srda_qg.model import make_diffusion_model
from src.filo_srda_qg.shared_srda.source import fold_pv
from src.four_dim_srda.utils.random_seed_helper import set_seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--root-dir", default=".")
    args = parser.parse_args()
    config = yaml.safe_load(pathlib.Path(args.config_path).read_text())
    root = pathlib.Path(args.root_dir).resolve()
    set_seeds(int(config["train"]["seed"]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = make_diffusion_dataset(
        base_config=config["dataset"],
        root_dir=root,
        split="valid",
        samples_per_epoch=1,
    )
    sample = dataset[0]
    target = fold_pv(sample["hr"][None]).to(device)
    lr = sample["condition_lr"][None].to(device)
    obs = sample["obs"][None].to(device)
    model = make_diffusion_model(config["model"]).to(device)
    diffusion = GaussianDiffusion(DiffusionConfig.from_dict(config["diffusion"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["overfit"]["lr"]), weight_decay=0.0
    )
    generator = torch.Generator(device=device).manual_seed(4242)
    gamma = diffusion.sample_noise_level(1, device=device, generator=generator)
    noise = torch.randn(target.shape, device=device, generator=generator)

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            return diffusion.training_loss(
                model,
                target=target,
                condition_lr=lr,
                obs=obs,
                noise_level=gamma,
                noise=noise,
            ).item()

    initial = evaluate()
    model.train()
    steps = int(config["overfit"]["steps"])
    for step in range(steps):
        loss = diffusion.training_loss(
            model,
            target=target,
            condition_lr=lr,
            obs=obs,
            noise_level=gamma,
            noise=noise,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == steps:
            print(f"step={step + 1} loss={loss.item():.8g}")
    final = evaluate()
    ratio = final / max(initial, 1e-12)
    print(f"initial_eval={initial:.8g} final_eval={final:.8g} ratio={ratio:.8g}")
    if ratio > float(config["overfit"]["max_final_ratio"]):
        raise RuntimeError("Tiny diffusion overfit acceptance criterion failed")
    model.eval()
    sample = diffusion.sample(
        model,
        condition_lr=lr,
        obs=obs,
        ensemble_size=2,
        steps=min(20, diffusion.num_timesteps),
        eta=1.0,
        chunk_size=1,
        generator=torch.Generator(device=device).manual_seed(999),
    )
    if not torch.isfinite(sample.samples).all():
        raise FloatingPointError("Tiny conditional sampling produced non-finite values")
    diversity = (sample.samples[:, 0] - sample.samples[:, 1]).square().mean().sqrt()
    print(
        f"sample_shape={tuple(sample.samples.shape)} nfe={sample.nfe} "
        f"member_rms_difference={diversity.item():.8g}"
    )
    if diversity.item() == 0.0:
        raise RuntimeError("Tiny conditional sampling produced identical members")


if __name__ == "__main__":
    main()
