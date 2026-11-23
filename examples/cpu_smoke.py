"""Exercise the solver, conditional samplers and scores on generated CPU data."""

from dataclasses import replace
from pathlib import Path

import torch

from src.filo_srda_qg.conditional_sb_srda_qg.model import make_conditional_sb_model
from src.filo_srda_qg.conditional_sb_srda_qg.sampler import sample_conditional_bare_heun_exact_noise
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.model import make_denseobs_conditional_sb_model
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.sampler import sample_denseobs_conditional_bare_heun_exact_noise
from src.filo_srda_qg.shared_srda.source import spatial_lift_lr, temporal_lift_lr
from src.filo_srda_qg.letkf.letkf_uq import ensemble_diagnostics
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge, ContinuousQuadraticBridgeConfig
from src.filo_srda_qg.sb_prior_retrain_qg.cycling_schedule import cycling_schedule
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.qg_model.jet_maker import make_jet_velocity_with_sech_squared_and_sigmoid_profile
from src.qg_model.qg_model import QGModel


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    path = root / "python/configs/four_dim_srda/handover/cfd_simulation/qg_model/gpu_make_data_config.yml"
    config = CFDConfig.load(path)
    tiny = replace(config.lr_base_config, nx=8, ny=9, n_batch=1, device="cpu")
    qg = QGModel(tiny, show_input_cfg_info=False)
    velocity = make_jet_velocity_with_sech_squared_and_sigmoid_profile(tiny)
    qg.initialize(velocity)
    anchors = [qg.get_grid_pv()]
    for _ in range(2):
        qg.integrate_n_steps(dt_per_step=config.time_config.lr_dt, n_steps=1)
        anchors.append(qg.get_grid_pv())
    lr = torch.stack(anchors)[None, :, None, :, :-1, :].float()
    source = temporal_lift_lr(spatial_lift_lr(lr, hr_size=(16, 16, 16)))[:, :, 0]
    sparse = source[:, (0, 4, 8), None]
    mask = torch.zeros_like(sparse)
    mask[:, :2, :, :, ::4, ::4] = 1
    observations = sparse * mask
    bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(
        epsilon=0.2581988897471611, noise_exponent=2, interior_strata=30))
    model_config = dict(out_channels=9, base_width=32, channel_multipliers=[1] * 5,
                        num_res_blocks=1, norm_groups=8, dropout=0.0, attention_head_dim=16,
                        max_period=10.0, activation_checkpointing=False, hr_z_size=16,
                        hr_y_size=16, hr_x_size=16, max_parameter_count=100_000_000)
    variants = (
        ("D", 20, make_denseobs_conditional_sb_model, sample_denseobs_conditional_bare_heun_exact_noise),
        ("I", 24, make_conditional_sb_model, sample_conditional_bare_heun_exact_noise),
    )
    for name, channels, build_model, sample in variants:
        result = sample(model=build_model(dict(model_config, input_channels=channels)).eval(),
                        source=source, observations=observations, masks=mask, bridge=bridge,
                        ensemble_size=2, nfe_budget=2, member_chunk_size=2, sampling_seed=7042)
        assert result.samples.shape == (1, 2, 9, 16, 16, 16)
        assert result.actual_nfe == 2 and result.num_intervals == 1
        assert torch.isfinite(result.samples).all()
        members = result.samples[0, :, 4]
        summary = ensemble_diagnostics(members, source[0, 4])
        assert summary.rank_histogram.sum().item() == source[0, 4].numel()
        print(f"BridgeSRDA-{name}: generated CPU window {tuple(result.samples.shape)}, NFE={result.actual_nfe}")
    assert cycling_schedule(num_times=12, assimilation_period=4, forecast_span=4) == ([4, 8], 4)
    assert cycling_schedule(num_times=12, assimilation_period=4, forecast_span=4, stride=2) == ([4], 8)


if __name__ == "__main__":
    main()
