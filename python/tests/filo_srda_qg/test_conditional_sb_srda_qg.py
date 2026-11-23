from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.filo_srda_qg.conditional_sb_srda_qg.bridge import conditional_bridge_loss_at_time
from src.filo_srda_qg.conditional_sb_srda_qg.conditioning import build_observation_condition
from src.filo_srda_qg.conditional_sb_srda_qg.model import make_conditional_sb_model
from src.filo_srda_qg.conditional_sb_srda_qg.sampler import (
    sample_conditional_bare_heun_exact_noise,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def test_observation_condition_has_exact_slots_masks_and_shape() -> None:
    source = torch.zeros(1, 9, 2, 3, 4)
    source[:, 0] = 1.0
    source[:, 4] = 2.0
    source[:, 8] = 3.0
    obs = torch.zeros(1, 3, 1, 2, 3, 4)
    mask = torch.zeros_like(obs)
    obs[:, 0] = 4.0
    obs[:, 1] = 8.0
    mask[:, 0, :, :, 1, 2] = 1.0
    mask[:, 1, :, :, 0, 1] = 1.0
    condition = build_observation_condition(source, obs, mask)
    assert condition.shape == (1, 6, 2, 3, 4)
    assert torch.equal(condition[:, 3:], mask[:, :, 0])
    assert torch.all(condition[:, 0, :, 1, 2] == 3.0)
    assert torch.all(condition[:, 1, :, 0, 1] == 6.0)
    assert torch.count_nonzero(condition[:, 2]) == 0


def test_unobserved_values_cannot_change_condition() -> None:
    source = torch.rand(1, 9, 2, 3, 4)
    obs = torch.rand(1, 3, 1, 2, 3, 4)
    mask = torch.zeros_like(obs)
    first = build_observation_condition(source, obs, mask)
    second = build_observation_condition(source, obs + 1000.0, mask)
    assert torch.equal(first, second)


def test_p2_conditional_loss_is_zero_for_exact_drift() -> None:
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(
            epsilon=0.2581988897471611, noise_exponent=2
        )
    )
    source = torch.randn(1, 9, 2, 3, 4)
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    obs = torch.randn(1, 3, 1, 2, 3, 4)
    mask = torch.ones_like(obs)
    t = torch.tensor([0.37])
    truth = bridge.target_drift(source, target, t, noise)

    class Exact(torch.nn.Module):
        def forward(self, state, source_arg, obs_arg, mask_arg, time):
            assert state.shape == source_arg.shape
            assert obs_arg.shape == mask_arg.shape
            assert torch.equal(time, t)
            return truth

    assert conditional_bridge_loss_at_time(
        bridge, Exact(), source=source, target=target,
        observations=obs, masks=mask, t=t, noise=noise,
    ) == 0


class ConditionalConstantDrift(torch.nn.Module):
    def forward(self, state, source, observations, masks, t):
        assert observations.shape[0] == state.shape[0]
        assert masks.shape == observations.shape
        return torch.full_like(state, 0.1)


def test_conditional_sampler_is_reproducible_diverse_and_chunk_invariant() -> None:
    source = torch.zeros(2, 9, 2, 3, 4)
    obs = torch.zeros(2, 3, 1, 2, 3, 4)
    mask = torch.ones_like(obs)
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(
            epsilon=0.2581988897471611, noise_exponent=2
        )
    )
    common = dict(
        model=ConditionalConstantDrift(), source=source,
        observations=obs, masks=mask, bridge=bridge,
        ensemble_size=3, nfe_budget=10, sampling_seed=123,
        brownian_base_steps=10,
    )
    first = sample_conditional_bare_heun_exact_noise(
        member_chunk_size=1, **common
    )
    second = sample_conditional_bare_heun_exact_noise(
        member_chunk_size=4, **common
    )
    assert first.actual_nfe == 10
    assert torch.equal(first.samples, second.samples)
    assert not torch.equal(first.samples[:, 0], first.samples[:, 1])


def test_config_locks_scratch_p2_four_gpu_contract() -> None:
    config = load_config(
        Path(
            "python/configs/filo_srda_qg/conditional_sb_srda_qg/"
            "conditional_sb_p2_b32.yml"
        )
    )
    assert config["conditioning"]["enabled"] is True
    assert config["model"]["input_channels"] == 24
    assert config["bridge"]["noise_exponent"] == 2
    assert config["dataset"]["obs_noise_std"] == pytest.approx(0.022)
    assert config["dataset"]["fixed_obs_mask_index"] is None
    assert 4 * config["train"]["gradient_accumulation_steps"] == 32


def test_small_conditional_model_has_24_to_9_signature_and_zero_output() -> None:
    config = {
        "input_channels": 24,
        "out_channels": 9,
        "base_width": 32,
        "channel_multipliers": [1, 1, 1, 1, 1],
        "num_res_blocks": 1,
        "norm_groups": 32,
        "dropout": 0.0,
        "attention_head_dim": 16,
        "max_period": 10.0,
        "activation_checkpointing": False,
        "hr_z_size": 16,
        "hr_y_size": 16,
        "hr_x_size": 16,
        "max_parameter_count": 100_000_000,
    }
    model = make_conditional_sb_model(config).eval()
    assert model.input_conv.in_channels == 24
    assert model.output_conv.out_channels == 9
    source = torch.rand(2, 9, 16, 16, 16)
    obs = torch.rand(2, 3, 1, 16, 16, 16)
    mask = torch.ones_like(obs)
    with torch.no_grad():
        output = model(source, source, obs, mask, torch.tensor([0.5, 0.5]))
    assert output.shape == source.shape
    assert torch.count_nonzero(output) == 0
