from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
    StratifiedTimeAssignment,
    generalized_bare_interval_variance,
    make_time_assignment,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config
from src.filo_srda_qg.sb_prior_retrain_qg.ema import EMA
from src.filo_srda_qg.sb_prior_retrain_qg.gates import evaluate_clean_gate
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import sample_bare_heun_exact_noise


class ConstantDrift(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(
        self, state: torch.Tensor, source: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return torch.full_like(state, self.value)


def test_stratified_global_batch_has_required_endpoint_and_interior_roles() -> None:
    assignment = StratifiedTimeAssignment(
        world_size=4, accumulation_steps=8, interior_strata=30, seed=42
    )
    roles = assignment.roles(epoch=3, optimizer_group=11)
    assert sum(role.kind == "endpoint_zero" for role in roles) == 1
    assert sum(role.kind == "endpoint_one" for role in roles) == 1
    interiors = [role.stratum for role in roles if role.kind == "interior"]
    assert sorted(interiors) == list(range(30))
    times = [
        assignment.time(
            epoch=3,
            optimizer_group=11,
            rank=rank,
            accumulation_index=acc,
            device=torch.device("cpu"),
        )[0].item()
        for acc in range(8)
        for rank in range(4)
    ]
    assert times.count(0.0) == 1
    assert times.count(1.0) == 1
    assert all(value in {0.0, 1.0} or 0.0 < value < 1.0 for value in times)


def test_stratified_schedule_rejects_wrong_effective_batch() -> None:
    with pytest.raises(ValueError, match="interior_strata"):
        StratifiedTimeAssignment(world_size=4, accumulation_steps=3, interior_strata=30)


def test_endpoint_targets_are_deterministic_conditional_means() -> None:
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(epsilon=0.2, noise_exponent=1)
    )
    source = torch.randn(2, 9, 2, 3, 4)
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    t0 = torch.zeros(2)
    t1 = torch.ones(2)
    assert torch.allclose(
        bridge.interpolate(source, target, t0, noise),
        source,
    )
    assert torch.allclose(
        bridge.target_drift(source, target, t0, noise),
        -source,
    )
    assert torch.allclose(
        bridge.interpolate(source, target, t1, noise),
        target,
    )
    assert torch.allclose(
        bridge.target_drift(source, target, t1, noise),
        -source + 2.0 * target,
    )


def test_generalized_interval_variance_integrates_and_adds() -> None:
    epsilon = 0.2
    total_p1 = generalized_bare_interval_variance(0.0, 1.0, epsilon, 1)
    parts_p1 = sum(
        generalized_bare_interval_variance(i / 10, (i + 1) / 10, epsilon, 1)
        for i in range(10)
    )
    assert total_p1 == pytest.approx(epsilon**2 / 3.0)
    assert parts_p1 == pytest.approx(total_p1)
    epsilon_p2 = 0.2 * math.sqrt(5.0 / 3.0)
    assert generalized_bare_interval_variance(0.0, 1.0, epsilon_p2, 2) == pytest.approx(
        epsilon**2 / 3.0
    )


def test_continuous_bridge_loss_zero_for_exact_drift() -> None:
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(epsilon=0.2, noise_exponent=1)
    )
    source = torch.randn(1, 9, 2, 3, 4)
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    t = torch.tensor([0.37])
    truth = bridge.target_drift(source, target, t, noise)

    class Exact(torch.nn.Module):
        def forward(self, state, source_arg, time):
            assert state.shape == source_arg.shape
            assert torch.equal(time, t)
            return truth

    assert bridge.loss_at_time(Exact(), source=source, target=target, t=t, noise=noise) == 0


def test_bare_heun_exact_noise_reports_nfe_reproducible_and_diverse() -> None:
    source = torch.zeros(1, 9, 2, 3, 4)
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(epsilon=0.2, noise_exponent=1)
    )
    kwargs = dict(
        model=ConstantDrift(0.1),
        source=source,
        bridge=bridge,
        ensemble_size=3,
        nfe_budget=10,
        member_chunk_size=1,
        sampling_seed=123,
        brownian_base_steps=10,
    )
    first = sample_bare_heun_exact_noise(**kwargs)
    second = sample_bare_heun_exact_noise(**kwargs)
    assert first.actual_nfe == 10
    assert first.num_intervals == 5
    assert torch.equal(first.samples, second.samples)
    assert not torch.equal(first.samples[:, 0], first.samples[:, 1])


def test_ema_updates_and_copies_fp32_shadow() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    target = torch.nn.Linear(2, 1, bias=False)
    ema.copy_to(target)
    assert target.weight.dtype == model.weight.dtype
    assert target.weight.mean().item() == pytest.approx(2.0)


def test_clean_gate_passes_and_fails_expected_conditions() -> None:
    nfe20 = {
        "physical_mae": 1.0,
        "physical_rmse": 1.0,
        "physical_crps": 1.0,
        "physical_spread_rms": 1.0,
        "spread_rmse_ratio": 1.0,
        "coverage_90": 0.8,
        "perturbation_high_k_fraction": 0.1,
        "perturbation_correlation_length": 3.0,
        "member_gradient_variance": 1.0,
        "target_gradient_variance": 1.0,
        "member_laplacian_variance": 1.0,
        "target_laplacian_variance": 1.0,
    }
    nfe50 = {**nfe20, "physical_crps": 0.98}
    thresholds = {
        "midpoint_grid_ratio": 1.25,
        "mae_relative_change": 0.05,
        "rmse_relative_change": 0.05,
        "crps_relative_change": 0.05,
        "spread_relative_change": 0.05,
        "spread_ratio_relative_change": 0.05,
        "coverage_absolute_change": 0.03,
        "high_k_relative_change": 0.10,
        "high_k_max": 0.15,
        "correlation_length_min": 2.0,
        "gradient_ratio_max": 1.5,
        "laplacian_ratio_max": 3.0,
        "spread_ratio_min": 0.5,
        "spread_ratio_max": 1.5,
    }
    assert evaluate_clean_gate(
        nfe20=nfe20,
        nfe50=nfe50,
        midpoint_grid_ratio=1.0,
        thresholds=thresholds,
    ).passed
    bad = {**nfe50, "perturbation_high_k_fraction": 0.4}
    result = evaluate_clean_gate(
        nfe20=nfe20,
        nfe50=bad,
        midpoint_grid_ratio=1.0,
        thresholds=thresholds,
    )
    assert not result.passed
    assert not result.checks["high_k"]


def test_config_inheritance_for_p2_variant() -> None:
    path = Path("python/configs/filo_srda_qg/sb_prior_retrain_qg/sb_prior_retrain_p2_b32.yml")
    config = load_config(path)
    assert config["bridge"]["noise_exponent"] == 2
    assert config["bridge"]["epsilon"] == pytest.approx(0.2581988897)
    assert config["train"]["gradient_accumulation_steps"] == 8
    assert config["train"]["ema_decay"] == pytest.approx(0.999)
    assert config["time_sampling"]["mode"] == "continuous_stratified"


def test_config_inheritance_for_two_gpu_variants() -> None:
    p2_path = Path(
        "python/configs/filo_srda_qg/sb_prior_retrain_qg/"
        "sb_prior_retrain_p2_b32_2gpu.yml"
    )
    p2 = load_config(p2_path)
    assert p2["bridge"]["noise_exponent"] == 2
    assert p2["train"]["gradient_accumulation_steps"] == 16
    assert p2["train"]["ema_decay"] == pytest.approx(0.999)
    assert p2["time_sampling"]["mode"] == "continuous_stratified"
    assert "ema999_2gpu" in p2["output"]["run_name"]
