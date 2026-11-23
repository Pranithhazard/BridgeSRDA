from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.bridge import (
    denseobs_conditional_bridge_loss_at_time,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.checkpoint import (
    load_training_checkpoint,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.conditioning import (
    build_dense_observation_condition,
    clear_interpolation_cache,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.model import (
    make_denseobs_conditional_sb_model,
)
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler import (
    sample_denseobs_conditional_bare_heun_exact_noise,
)
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import (
    ContinuousQuadraticBridge,
    ContinuousQuadraticBridgeConfig,
)
from src.filo_srda_qg.sb_prior_retrain_qg.conditional_runtime import (
    DENSE_OBSERVATION_KIND,
    MASKED_INNOVATION_KIND,
    get_conditional_runtime,
)
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config


def raw_tensors(
    *, batch: int = 1, z: int = 2, y: int = 9, x: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.rand(batch, 9, z, y, x)
    observations = torch.zeros(batch, 3, 1, z, y, x)
    masks = torch.zeros_like(observations)
    for index in range(batch):
        oy, ox = index % 4, (index + 1) % 4
        masks[index, :2, 0, :, oy::4, ox::4] = 1
    observations = masks * torch.rand_like(observations)
    return source, observations, masks


def test_dense_observations_have_two_slots_and_preserve_sensors_exactly() -> None:
    source, observations, masks = raw_tensors(batch=2)
    dense = build_dense_observation_condition(source, observations, masks)
    assert dense.shape == (2, 2, 2, 9, 16)
    for batch in range(2):
        for slot in range(2):
            active = masks[batch, slot, 0].bool()
            assert torch.equal(dense[batch, slot][active], observations[batch, slot, 0][active])


def test_all_64_canonical_origins_are_supported() -> None:
    clear_interpolation_cache()
    source = torch.rand(1, 9, 1, 64, 128)
    for origin in range(64):
        oy, ox = divmod(origin, 8)
        observations = torch.zeros(1, 3, 1, 1, 64, 128)
        masks = torch.zeros_like(observations)
        masks[:, :2, 0, :, oy::8, ox::8] = 1
        observations = masks * torch.rand_like(observations)
        dense = build_dense_observation_condition(source, observations, masks)
        assert dense.shape == (1, 2, 1, 64, 128)
        assert torch.isfinite(dense).all()


@pytest.mark.parametrize("interval", [4, 8, 16])
def test_canonical_dense_and_sparse_cartesian_layouts(interval: int) -> None:
    source = torch.rand(1, 9, 2, 64, 128)
    observations = torch.zeros(1, 3, 1, 2, 64, 128)
    masks = torch.zeros_like(observations)
    masks[:, :2, 0, :, 1::interval, 2::interval] = 1
    observations = masks * torch.rand_like(observations)
    dense = build_dense_observation_condition(source, observations, masks)
    assert dense.shape == (1, 2, 2, 64, 128)
    assert torch.isfinite(dense).all()
    for slot in range(2):
        active = masks[:, slot, 0].bool()
        assert torch.equal(dense[:, slot][active], observations[:, slot, 0][active])


def test_periodic_x_and_nonperiodic_y_edge_extension() -> None:
    source = torch.zeros(1, 9, 1, 6, 8)
    observations = torch.zeros(1, 3, 1, 1, 6, 8)
    masks = torch.zeros_like(observations)
    masks[:, :2, 0, :, 2::2, 1::4] = 1
    # Constant-by-row sensor values make edge behavior unambiguous.
    observations[:, :2, 0, :, 2, 1] = 2
    observations[:, :2, 0, :, 2, 5] = 2
    observations[:, :2, 0, :, 4, 1] = 4
    observations[:, :2, 0, :, 4, 5] = 4
    dense = build_dense_observation_condition(source, observations, masks)
    assert torch.allclose(dense[:, :, :, 0], torch.full_like(dense[:, :, :, 0], 2))
    assert torch.allclose(dense[:, :, :, 5], torch.full_like(dense[:, :, :, 5], 4))
    # A constant sensor row must remain continuous through the periodic seam.
    assert torch.allclose(dense[..., 0], dense[..., -1])


def test_non_cartesian_mask_is_rejected() -> None:
    source, observations, masks = raw_tensors()
    masks[:, 0, 0, :, 0, 0] = 1
    with pytest.raises(ValueError, match="Cartesian"):
        build_dense_observation_condition(source, observations, masks)


def test_missing_slot_falls_back_to_corresponding_source_frame() -> None:
    source, observations, masks = raw_tensors()
    observations[:, 0] = 0
    masks[:, 0] = 0
    dense = build_dense_observation_condition(source, observations, masks)
    assert torch.equal(dense[:, 0], source[:, 0])
    assert not torch.equal(dense[:, 1], source[:, 4])


def test_future_slot_does_not_affect_dense_condition() -> None:
    source, observations, masks = raw_tensors()
    first = build_dense_observation_condition(source, observations, masks)
    observations[:, 2] = 999
    masks[:, 2] = 1
    second = build_dense_observation_condition(source, observations, masks)
    assert torch.equal(first, second)


def test_p2_denseobs_loss_is_zero_for_exact_implemented_drift() -> None:
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(
            epsilon=0.2581988897471611, noise_exponent=2
        )
    )
    source, observations, masks = raw_tensors()
    target = torch.randn_like(source)
    noise = torch.randn_like(source)
    tau = torch.tensor([0.37])
    truth = bridge.target_drift(source, target, tau, noise, marginalize_t1=True)

    class Exact(torch.nn.Module):
        def forward(self, state, source_arg, dense_arg, time):
            assert state.shape == source_arg.shape
            assert dense_arg.shape[1] == 2
            assert torch.equal(time, tau)
            return truth

    loss = denseobs_conditional_bridge_loss_at_time(
        bridge,
        Exact(),
        source=source,
        target=target,
        observations=observations,
        masks=masks,
        t=tau,
        noise=noise,
    )
    assert loss == 0


class DenseConstantDrift(torch.nn.Module):
    def forward(self, state, source, dense_observations, tau):
        assert dense_observations.shape[0] == state.shape[0]
        assert dense_observations.shape[1] == 2
        return torch.full_like(state, 0.1)


def test_sampler_is_reproducible_diverse_nfe_exact_and_chunk_invariant() -> None:
    source, observations, masks = raw_tensors(batch=2)
    bridge = ContinuousQuadraticBridge(
        ContinuousQuadraticBridgeConfig(
            epsilon=0.2581988897471611, noise_exponent=2
        )
    )
    common = dict(
        model=DenseConstantDrift(),
        source=source,
        observations=observations,
        masks=masks,
        bridge=bridge,
        ensemble_size=3,
        nfe_budget=10,
        sampling_seed=123,
        brownian_base_steps=10,
    )
    first = sample_denseobs_conditional_bare_heun_exact_noise(
        member_chunk_size=1, **common
    )
    second = sample_denseobs_conditional_bare_heun_exact_noise(
        member_chunk_size=4, **common
    )
    assert first.actual_nfe == 10
    assert torch.equal(first.samples, second.samples)
    assert not torch.equal(first.samples[:, 0], first.samples[:, 1])


def small_model_config() -> dict[str, object]:
    return {
        "input_channels": 20,
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


def test_small_model_has_20_to_9_signature_and_zero_output() -> None:
    model = make_denseobs_conditional_sb_model(small_model_config()).eval()
    assert model.input_conv.in_channels == 20
    assert model.output_conv.out_channels == 9
    source = torch.rand(2, 9, 16, 16, 16)
    dense = torch.rand(2, 2, 16, 16, 16)
    with torch.no_grad():
        output = model(source, source, dense, torch.tensor([0.5, 0.5]))
    assert output.shape == source.shape
    assert torch.count_nonzero(output) == 0


def test_config_and_runtime_dispatch_lock_073a_without_regressing_072() -> None:
    config = load_config(
        Path(
            "python/configs/filo_srda_qg/denseobs_conditional_sb_srda_qg/"
            "denseobs_conditional_sb_p2_b32.yml"
        )
    )
    assert config["conditioning"]["kind"] == DENSE_OBSERVATION_KIND
    assert config["model"]["input_channels"] == 20
    assert config["train"]["num_epochs"] == 2000
    assert config["train"]["early_stopping_min_epoch"] == 1200
    assert config["bridge"]["noise_exponent"] == 2
    runtime = get_conditional_runtime(config)
    assert runtime is not None
    assert runtime.model_signature == "DenseObsConditionalSBUNet3D:(9+9+2denseobs)->9"

    config_072 = load_config(
        Path(
            "python/configs/filo_srda_qg/conditional_sb_srda_qg/"
            "conditional_sb_p2_b32.yml"
        )
    )
    runtime_072 = get_conditional_runtime(config_072)
    assert runtime_072 is not None
    assert runtime_072.kind == MASKED_INNOVATION_KIND
    assert runtime_072.model_signature.startswith("ConditionalSBUNet3D")


def test_standalone_checkpoint_preserves_073a_contract(tmp_path: Path) -> None:
    config = load_config(
        Path(
            "python/configs/filo_srda_qg/denseobs_conditional_sb_srda_qg/"
            "denseobs_conditional_sb_p2_b32.yml"
        )
    )
    config["model"] = small_model_config()
    model = make_denseobs_conditional_sb_model(config["model"])
    contract = {
        "condition": DENSE_OBSERVATION_KIND,
        "interpolation_method": "periodic_bilinear",
        "network_observation_frames": [0, 4],
        "future_slot_omitted_from_network": True,
    }
    path = tmp_path / "best_panel_crps_ema.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": {
                "selected_epoch": 1200,
                "weights_kind": "ema",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "model_signature": "DenseObsConditionalSBUNet3D:(9+9+2denseobs)->9",
                "observation_contract": contract,
            },
        },
        path,
    )
    loaded, metadata = load_training_checkpoint(
        config=config,
        checkpoint_path=path,
        device=torch.device("cpu"),
        weights_kind="ema",
    )
    assert loaded.input_conv.in_channels == 20
    assert metadata["checkpoint_epoch"] == 1200
    assert metadata["weights_kind"] == "ema"
    assert metadata["model_signature"] == "DenseObsConditionalSBUNet3D:(9+9+2denseobs)->9"
    assert metadata["observation_contract"] == contract
