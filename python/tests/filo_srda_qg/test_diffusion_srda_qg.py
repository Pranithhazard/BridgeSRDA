from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml

from src.filo_srda_qg.diffusion_srda_qg.diffusion import (
    DiffusionConfig,
    GaussianDiffusion,
)
from src.filo_srda_qg.diffusion_srda_qg.data import (
    DiffusionWindowDataset,
    diffusion_conditions_to_zero_one,
    diffusion_values_to_zero_one,
    prepare_diffusion_observations,
    zero_one_conditions_to_diffusion,
)
from src.filo_srda_qg.diffusion_srda_qg.model import make_diffusion_model
from src.filo_srda_qg.diffusion_srda_qg.narrow_jet import (
    sample_memberwise_in_chunks,
)
from src.filo_srda_qg.shared_srda.source import spatial_lift_lr


class ExactNoiseModel(nn.Module):
    def __init__(self, noise: torch.Tensor):
        super().__init__()
        self.register_buffer("noise", noise)

    def forward(self, x, lr, obs, noise_level):
        return self.noise.expand_as(x)


class ZeroModel(nn.Module):
    def forward(self, x, lr, obs, noise_level):
        return torch.zeros_like(x)


class OneSampleDataset:
    def __init__(self):
        self.cfg = SimpleNamespace()
        self.epoch = None
        mask = torch.tensor([[[[[1.0, 0.0]]]]])
        self.sample = {
            "condition_lr": torch.tensor([[[[[[0.25]]]]]]),
            "source_lr": torch.tensor([[[[[[0.75]]]]]]),
            "lr": torch.tensor([[[[[[0.25]]]]]]),
            "obs": torch.tensor([[[[[0.75, 0.0]]]]]),
            "mask": mask,
            "hr": torch.tensor([[[[[[0.0, 1.0]]]]]]),
            "metadata": {},
        }

    def __len__(self):
        return 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        assert index == 0
        return self.sample

    def get_by_path(self, path):
        assert str(path) == "sample.npy"
        return self.sample


def test_fp32_reconstruction_config_locked_defaults() -> None:
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "python/configs/filo_srda_qg/diffusion_srda_qg/"
        "diffusion_w144_fp32_reconsel.yml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["dataset"]["data_scaling"] == "minus_one_one"
    assert config["dataset"]["diffusion_missing_value"] == -1.1
    assert config["model"]["width"] == 144
    assert config["model"]["num_maxvit_blocks"] == 6
    assert config["train"]["amp"] is False
    assert config["train"]["samples_per_epoch"] == 3600
    assert config["train"]["num_epochs"] == 400
    assert config["train"]["use_recon_loss"] is True
    assert config["train"]["checkpoint_selection"] == "reconstruction_mae"
    assert config["train"]["validation_panel_interval"] == 5
    assert config["train"]["validation_panel_size"] == 8
    assert config["train"]["early_stopping_patience"] == 20


def test_diffsrda3d_config_locked_defaults() -> None:
    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "python/configs/filo_srda_qg/diffusion_srda_qg/"
        "diffusion_diffsrda3d_b32_fp32_filledobs.yml"
    )
    config = yaml.safe_load(path.read_text())
    assert config["dataset"]["data_scaling"] == "minus_one_one"
    assert config["dataset"]["diffusion_missing_value"] == -1.1
    assert config["dataset"]["observation_fill_mode"] == "bicubic_horizontal"
    assert config["model"]["model_name"] == "DiffSRDAUNet3D"
    assert config["model"]["base_width"] == 32
    assert config["model"]["channel_multipliers"] == [1, 2, 4, 8, 8]
    assert config["model"]["num_res_blocks"] == 3
    assert config["model"]["dropout"] == 0.5
    assert config["model"]["attention_heads"] == 1
    assert config["model"]["activation_checkpointing"] is True
    assert config["train"]["amp"] is False
    assert config["train"]["checkpoint_selection"] == "reconstruction_mae"
    assert config["train"]["reconstruction_selection_metric"] == "sample_l1"


def test_minus_one_one_scaling_and_missing_observation_sentinel() -> None:
    base = OneSampleDataset()
    dataset = DiffusionWindowDataset(
        base, data_scaling="minus_one_one", diffusion_missing_value=-1.1
    )
    sample = dataset[0]
    assert torch.equal(sample["condition_lr"], torch.tensor([[[[[[-0.5]]]]]]))
    assert torch.equal(sample["source_lr"], torch.tensor([[[[[[0.5]]]]]]))
    assert torch.equal(sample["hr"], torch.tensor([[[[[[-1.0, 1.0]]]]]]))
    assert torch.equal(sample["obs"], torch.tensor([[[[[0.5, -1.1]]]]]))
    assert torch.equal(sample["lr"], sample["condition_lr"])
    assert torch.equal(sample["mask"], base.sample["mask"])
    dataset.set_epoch(7)
    assert base.epoch == 7
    assert torch.equal(dataset.get_by_path("sample.npy")["obs"], sample["obs"])

    zero_one_lr, zero_one_obs = diffusion_conditions_to_zero_one(
        sample["condition_lr"],
        sample["obs"],
        sample["mask"],
        data_scaling="minus_one_one",
    )
    assert torch.equal(zero_one_lr, base.sample["condition_lr"])
    assert torch.equal(zero_one_obs, base.sample["obs"])
    assert torch.equal(
        diffusion_values_to_zero_one(
            sample["hr"], data_scaling="minus_one_one"
        ),
        base.sample["hr"],
    )
    diffusion_lr, diffusion_obs = zero_one_conditions_to_diffusion(
        base.sample["condition_lr"],
        base.sample["obs"],
        data_scaling="minus_one_one",
        diffusion_missing_value=-1.1,
        observation_mask=base.sample["mask"] > 0,
    )
    assert torch.equal(diffusion_lr, sample["condition_lr"])
    assert torch.equal(diffusion_obs, sample["obs"])


def test_bicubic_observation_fill_and_future_sentinel() -> None:
    obs = torch.zeros(3, 1, 2, 8, 16)
    mask = torch.zeros_like(obs)
    mask[:2, :, :, 1::4, 2::4] = 1.0
    obs[0, 0, 0][mask[0, 0, 0] > 0] = 0.25
    obs[0, 0, 1][mask[0, 0, 1] > 0] = 0.75
    obs[1, 0, 0][mask[1, 0, 0] > 0] = 0.50
    obs[1, 0, 1][mask[1, 0, 1] > 0] = 1.00
    filled = prepare_diffusion_observations(
        obs,
        mask,
        data_scaling="minus_one_one",
        diffusion_missing_value=-1.1,
        observation_fill_mode="bicubic_horizontal",
    )
    assert filled.shape == obs.shape
    assert torch.allclose(filled[0, 0, 0], torch.full((8, 16), -0.5))
    assert torch.allclose(filled[0, 0, 1], torch.full((8, 16), 0.5))
    assert torch.allclose(
        filled[1, 0, 0], torch.zeros(8, 16), atol=1.0e-6
    )
    assert torch.allclose(
        filled[1, 0, 1], torch.ones(8, 16), atol=1.0e-6
    )
    assert torch.all(filled[2] == -1.1)


def test_nearest_exact_lift_shape_and_alignment() -> None:
    lr = torch.arange(4 * 16 * 32, dtype=torch.float32).reshape(1, 1, 1, 4, 16, 32)
    lifted = spatial_lift_lr(lr)
    assert lifted.shape == (1, 1, 1, 16, 64, 128)
    assert torch.equal(lifted[..., ::4, ::4, ::4], lr)


def test_schedule_and_forward_process() -> None:
    diffusion = GaussianDiffusion(
        DiffusionConfig(num_train_timesteps=1000, beta_start=1e-4, beta_end=2e-2)
    )
    assert torch.isclose(diffusion.betas[0], torch.tensor(1e-4))
    assert torch.isclose(diffusion.betas[-1], torch.tensor(2e-2))
    assert torch.all(diffusion.alphas_cumprod[1:] < diffusion.alphas_cumprod[:-1])
    target = torch.ones(2, 9, 2, 3, 4)
    noise = torch.zeros_like(target)
    gamma = torch.tensor([0.25, 0.75])
    noisy = diffusion.q_sample(target, gamma, noise)
    assert torch.allclose(noisy[0], torch.full_like(noisy[0], 0.25))
    assert torch.allclose(noisy[1], torch.full_like(noisy[1], 0.75))
    sampled = diffusion.sample_noise_level(
        2048,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.all(sampled <= 1.0)
    assert torch.all(sampled >= diffusion.gamma[-1])
    assert any(
        torch.all(sampled <= diffusion.gamma_boundaries[index - 1])
        and torch.all(sampled >= diffusion.gamma_boundaries[index])
        for index in range(1, diffusion.num_timesteps + 1)
    )


def test_forward_process_empirical_moments() -> None:
    target = torch.full((200_000,), 0.4)
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(3))
    gamma = torch.full((target.shape[0],), 0.6)
    noisy = GaussianDiffusion.q_sample(target, gamma, noise)
    assert noisy.mean().item() == pytest.approx(0.24, abs=5e-3)
    assert noisy.var(unbiased=False).item() == pytest.approx(0.64, abs=1e-2)


def test_exact_noise_zero_loss_and_x0_reconstruction() -> None:
    diffusion = GaussianDiffusion(DiffusionConfig(num_train_timesteps=10))
    target = torch.randn(1, 9, 2, 3, 4)
    noise = torch.randn_like(target)
    gamma = torch.tensor([0.6])
    model = ExactNoiseModel(noise)
    lr = torch.randn(1, 3, 1, 4, 16, 32)
    obs = torch.randn(1, 3, 1, 2, 3, 4)
    loss = diffusion.training_loss(
        model,
        target=target,
        condition_lr=lr,
        obs=obs,
        noise_level=gamma,
        noise=noise,
    )
    assert loss.item() == 0.0
    noisy = diffusion.q_sample(target, gamma, noise)
    reconstructed = diffusion.predict_x0(noisy, gamma, noise)
    assert torch.allclose(reconstructed, target, atol=1e-6)


def test_sampler_reproducibility_diversity_and_nfe() -> None:
    diffusion = GaussianDiffusion(DiffusionConfig(num_train_timesteps=20))
    model = ZeroModel()
    lr = torch.zeros(1, 3, 1, 4, 2, 2)
    obs = torch.zeros(1, 3, 1, 2, 2, 2)
    first = diffusion.sample(
        model,
        condition_lr=lr,
        obs=obs,
        ensemble_size=2,
        steps=4,
        eta=1.0,
        chunk_size=1,
        generator=torch.Generator().manual_seed(7),
    )
    second = diffusion.sample(
        model,
        condition_lr=lr,
        obs=obs,
        ensemble_size=2,
        steps=4,
        eta=1.0,
        chunk_size=1,
        generator=torch.Generator().manual_seed(7),
    )
    assert first.nfe == 4
    assert torch.equal(first.samples, second.samples)
    assert not torch.equal(first.samples[:, 0], first.samples[:, 1])

    deterministic_first = diffusion.sample(
        model,
        condition_lr=lr,
        obs=obs,
        ensemble_size=1,
        steps=4,
        eta=0.0,
        chunk_size=1,
        generator=torch.Generator().manual_seed(12),
    )
    deterministic_second = diffusion.sample(
        model,
        condition_lr=lr,
        obs=obs,
        ensemble_size=1,
        steps=4,
        eta=0.0,
        chunk_size=1,
        generator=torch.Generator().manual_seed(12),
    )
    assert torch.equal(deterministic_first.samples, deterministic_second.samples)

    third = diffusion.sample(
        model,
        condition_lr=lr,
        obs=obs,
        ensemble_size=2,
        steps=4,
        eta=1.0,
        chunk_size=1,
        generator=torch.Generator().manual_seed(8),
    )
    assert not torch.equal(first.samples, third.samples)


def test_memberwise_sampling_preserves_parent_count_and_order() -> None:
    diffusion = GaussianDiffusion(DiffusionConfig(num_train_timesteps=10))
    condition_lr = torch.stack(
        [
            torch.full((3, 1, 1, 2, 2), float(index))
            for index in range(3)
        ]
    )
    obs = torch.zeros(3, 3, 1, 2, 2, 2)
    result = sample_memberwise_in_chunks(
        diffusion=diffusion,
        model=ZeroModel(),
        condition_lr=condition_lr,
        obs=obs,
        steps=3,
        eta=0.0,
        parent_chunk_size=1,
        generator=torch.Generator().manual_seed(22),
    )
    assert result.samples.shape == (3, 1, 9, 2, 2, 2)
    assert result.nfe == 3
    manual_generator = torch.Generator().manual_seed(22)
    manual = []
    for parent in range(3):
        child = diffusion.sample(
            ZeroModel(),
            condition_lr=condition_lr[parent : parent + 1],
            obs=obs[parent : parent + 1],
            ensemble_size=1,
            steps=3,
            eta=0.0,
            chunk_size=1,
            generator=manual_generator,
        )
        manual.append(child.samples[0, 0])
    assert torch.equal(result.samples[:, 0], torch.stack(manual))


def test_respacing_preserves_selected_alpha_and_one_step_formula() -> None:
    diffusion = GaussianDiffusion(DiffusionConfig(num_train_timesteps=20))
    indices = diffusion.respaced_indices(5)
    assert len(indices) == 5
    assert indices[0].item() == 0
    assert indices[-1].item() == 19
    selected = diffusion.alphas_cumprod[indices]
    assert torch.equal(selected, diffusion.alphas_cumprod.index_select(0, indices))

    lr = torch.zeros(1, 3, 1, 1, 2, 2)
    obs = torch.zeros(1, 3, 1, 2, 2, 2)
    seed = 71
    initial = torch.randn(
        (1, 9, 2, 2, 2), generator=torch.Generator().manual_seed(seed)
    )
    result = diffusion.sample(
        ZeroModel(),
        condition_lr=lr,
        obs=obs,
        ensemble_size=1,
        steps=1,
        eta=1.0,
        chunk_size=1,
        generator=torch.Generator().manual_seed(seed),
    )
    gamma = torch.sqrt(diffusion.alphas_cumprod[0])
    assert torch.allclose(result.samples[0, 0], initial[0] / gamma)
    assert result.nfe == 1


def test_diffusion_model_shape_gradients_and_initialization() -> None:
    model = make_diffusion_model(
        {
            "model_name": "DiffusionUNetMaxViT",
            "input_channels": 15,
            "out_channels": 9,
            "width": 16,
            "num_res_blocks": 1,
            "num_maxvit_blocks": 1,
            "norm_groups": 8,
            "time_embed_dim": 32,
            "residual_dropout": 0.0,
            "maxvit_dropout": 0.0,
            "bias": False,
            "n_head": 4,
            "expansion_rate": 2,
            "shrink_rate": 0.5,
            "grid_window_size_h": 2,
            "grid_window_size_w": 2,
            "use_downsample": False,
            "activation_checkpointing": False,
            "hr_z_size": 4,
            "hr_y_size": 8,
            "hr_x_size": 8,
            "output_init_std": 0.02,
        }
    )
    x = torch.randn(1, 9, 4, 8, 8)
    lr = torch.randn(1, 3, 1, 1, 2, 2)
    obs = torch.randn(1, 3, 1, 4, 8, 8)
    output = model(x, lr, obs, torch.tensor([0.5]))
    assert output.shape == x.shape
    assert model.output.weight.std().item() > 0.0
    output.square().mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_production_diffusion_model_parameter_limit() -> None:
    model = make_diffusion_model(
        {
            "model_name": "DiffusionUNetMaxViT",
            "width": 128,
            "n_head": 32,
        }
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 58_951_305
    assert count < 100_000_000
    assert model.output.weight.std().item() == pytest.approx(0.02, rel=0.02)


def test_fp32_reconstruction_model_parameter_limit() -> None:
    model = make_diffusion_model(
        {
            "model_name": "DiffusionUNetMaxViT",
            "width": 144,
            "n_head": 32,
            "num_maxvit_blocks": 6,
        }
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == 74_308_297
    assert count < 100_000_000


def test_diffsrda3d_model_shape_initialization_and_parameter_limit() -> None:
    model = make_diffusion_model(
        {
            "model_name": "DiffSRDAUNet3D",
            "input_channels": 15,
            "out_channels": 9,
            "base_width": 8,
            "channel_multipliers": [1, 2, 2, 4, 4],
            "num_res_blocks": 1,
            "norm_groups": 8,
            "dropout": 0.1,
            "attention_heads": 1,
            "activation_checkpointing": False,
            "hr_z_size": 16,
            "hr_y_size": 16,
            "hr_x_size": 16,
            "max_parameter_count": 100_000_000,
        }
    )
    captured = {}

    def capture_input(_module, inputs):
        captured["combined"] = inputs[0].detach().clone()

    handle = model.input_conv.register_forward_pre_hook(capture_input)
    noisy = torch.full((1, 9, 16, 16, 16), 3.0)
    lr = torch.full((1, 3, 1, 4, 4, 4), 1.0)
    obs = torch.full((1, 3, 1, 16, 16, 16), 2.0)
    output = model(noisy, lr, obs, torch.tensor([0.5]))
    handle.remove()
    assert output.shape == noisy.shape
    assert torch.all(captured["combined"][:, :3] == 1.0)
    assert torch.all(captured["combined"][:, 3:6] == 2.0)
    assert torch.all(captured["combined"][:, 6:] == 3.0)
    output.square().mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    root = Path(__file__).resolve().parents[3]
    config = yaml.safe_load(
        (
            root
            / "python/configs/filo_srda_qg/diffusion_srda_qg/"
            "diffusion_diffsrda3d_b32_fp32_filledobs.yml"
        ).read_text()
    )
    production = make_diffusion_model(config["model"])
    count = sum(parameter.numel() for parameter in production.parameters())
    assert count == 89_535_497
    assert count < 100_000_000
