"""CPU contract checks against the actual official dense-observation sampler."""

import math

import pytest
import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler import (
    sample_denseobs_conditional_bare_heun_exact_noise as official_sample,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.sampler import _trajectory_seed
from src.filo_srda_qg.timing_benchmark_qg import bridge_sampler as sidecar


class TinyDrift(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.times = []

    def forward(self, state, source, dense, tau):
        assert not torch.is_grad_enabled()
        self.times.append(tau.clone())
        return (0.1 * state + 0.03 * source + 0.02 * dense[:, :1]
                + 0.01 * tau[:, None, None, None, None])


def inputs(members=3):
    source = torch.linspace(-0.3, 0.4, members * 9 * 3 * 4).reshape(members, 9, 1, 3, 4)
    observations = torch.zeros(members, 3, 1, 1, 3, 4)
    masks = torch.zeros_like(observations)
    masks[:, :2, :, :, ::2, ::2] = 1
    observations.copy_(masks * 0.2)
    return dict(model=TinyDrift().eval(), source=source, observations=observations,
                masks=masks, sampling_seed=7342,
                bridge=ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(
                    epsilon=0.2581988897471611, noise_exponent=2)))


def test_base600_official_chunk_and_global_shard_equivalence():
    args = inputs()
    expected = official_sample(**args, ensemble_size=1, nfe_budget=20,
                              member_chunk_size=1, brownian_base_steps=600).samples[:, 0]
    for chunk in (1, 2, 3, 30):
        actual, nfe = sidecar.sample_bridge_heun(**args, member_chunk_size=chunk)
        assert nfe == 20
        assert torch.equal(actual, expected)
        assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
    shards = []
    for start, stop in ((0, 1), (1, 3)):
        shard = {**args, **{k: args[k][start:stop] for k in ("source", "observations", "masks")}}
        actual, _ = sidecar.sample_bridge_heun(**shard, global_member_offset=start,
                                             member_chunk_size=2)
        shards.append(actual)
    assert torch.equal(torch.cat(shards), expected)
    wrong, _ = sidecar.sample_bridge_heun(**shard, global_member_offset=0)
    assert not torch.equal(wrong, expected[1:])


def test_supplied_increments_match_official_integration_and_densify_once(monkeypatch):
    import src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler as official

    args = inputs()
    increments = torch.linspace(-0.02, 0.03, 10 * args["source"].numel()).reshape(
        10, *args["source"].shape)

    def injected(**kwargs):
        # The oracle runs all members together; the true production Heun loop
        # consumes a known path independently of either random-number method.
        return increments[kwargs["interval_index"]]

    monkeypatch.setattr(official, "_exact_increment", injected)
    expected = official_sample(**args, ensemble_size=1, nfe_budget=20,
                              member_chunk_size=3).samples[:, 0]
    calls = []
    original_dense = sidecar.build_dense_observation_condition

    def counted(*a, **kw):
        calls.append(1)
        return original_dense(*a, **kw)

    monkeypatch.setattr(sidecar, "build_dense_observation_condition", counted)
    for chunk in (1, 2, 3):
        args["model"].times.clear()
        actual, nfe = sidecar.sample_bridge_heun(**args, increments=increments,
                                               member_chunk_size=chunk)
        assert torch.equal(actual, expected)
        assert nfe == 20
        assert len(args["model"].times) == 20 * math.ceil(3 / chunk)
        assert args["model"].times[0].eq(0).all()
        assert args["model"].times[-1].eq(1).all()
    assert len(calls) == 3


@pytest.mark.parametrize("override,match", [
    ({"member_chunk_size": 0}, "member_chunk_size"),
    ({"member_chunk_size": 31}, "member_chunk_size"),
    ({"global_member_offset": -1}, "global_member_offset"),
    ({"nfe_budget": 19}, "nfe_budget"),
    ({"nfe_budget": 14}, "dividing 600"),
    ({"increments": torch.zeros(9, 3, 9, 1, 3, 4)}, "increments must have shape"),
    ({"increments": torch.full((10, 3, 9, 1, 3, 4), float("nan"))}, "increments must be finite"),
])
def test_invalid_contract_fails_explicitly(override, match):
    with pytest.raises(ValueError, match=match):
        sidecar.sample_bridge_heun(**{**inputs(), **override})


def test_training_precision_and_nonfinite_outputs_rejected():
    args = inputs()
    args["model"].train()
    with pytest.raises(ValueError, match="eval mode"):
        sidecar.sample_bridge_heun(**args)
    args["model"].eval()
    with pytest.raises(ValueError, match="FP32"):
        sidecar.sample_bridge_heun(**{**args, "source": args["source"].double()})
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            sidecar.sample_bridge_heun(**args)

    class BadDrift(torch.nn.Module):
        def forward(self, state, *unused):
            return torch.full_like(state, float("nan"))

    with pytest.raises(ValueError, match="nonfinite sampled output"):
        sidecar.sample_bridge_heun(**{**args, "model": BadDrift().eval()})
