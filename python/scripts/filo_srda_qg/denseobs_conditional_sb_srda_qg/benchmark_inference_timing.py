"""Forecast setup and synchronized timing used by the recursive benchmark."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

RUN = "DenseObsConditionalSBUNet3D_Scratch_quad_p2_eps02582_ogx08_ogy08_mixup_b32_ema999"


SB_CONFIG = "python/configs/filo_srda_qg/denseobs_conditional_sb_srda_qg/denseobs_conditional_sb_p2_b32.yml"


CFD_CONFIG = "python/configs/four_dim_srda/handover/cfd_simulation/qg_model/gpu_evaluation_config.yml"


SRDA_CONFIG = "python/configs/four_dim_srda/handover/perform_4D_SRDA/UNetMaxVitVer01/bea2_bed2_dspe360_nsls100_ogx08_ogy08_n3drb3_nmb6_bias0_bs12_lr1e-04.yml"


LETKF_CONFIG = "python/configs/four_dim_srda/handover/perform_letkf_hr_using_uhr/letkf_cfg_ogx08_ogy08_ne030_ch16e-04_cr6e+00_if12e-01_lr57e-01_bs6.yml"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def configure(threads, *, conv_tf32=False):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = conv_tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore", message="This method.*initialize_pv.*")


def backend_settings():
    """Record effective policy, rather than inferring arithmetic from dtype."""
    return dict(
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cpu_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        cudnn_version=torch.backends.cudnn.version(),
        cuda_version=torch.version.cuda,
        nvidia_tf32_override=os.environ.get("NVIDIA_TF32_OVERRIDE"),
    )


class BridgeWindow:
    def __init__(self, args, window, device, member_start=0, member_stop=30):
        from src.four_dim_srda.config.experiment_config import CFDConfig
        from src.four_dim_srda.config.config_loader import load_config
        from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import make_filter
        self.cfg = CFDConfig.load(Path(args.root) / CFD_CONFIG)
        self.device = device
        self.member_start, self.member_stop = member_start, member_stop
        for config in (self.cfg.hr_base_config, self.cfg.lr_base_config, self.cfg.uhr_base_config):
            config.device = str(device)
        self.cfg.hr_base_config.n_batch = self.cfg.lr_base_config.n_batch = member_stop - member_start
        srda = load_config(model_name="UNetMaxVitVer01", config_path=str(Path(args.root) / SRDA_CONFIG))
        self.dataset = SimpleNamespace(cfg=srda.dataset_config)
        self.filter = make_filter(self.cfg)
        self.state = window["sb_state"][member_start:member_stop].to(device=device, dtype=torch.float32)
        self.raw = window["raw_history"].to(device=device, dtype=torch.float32)
        self.seed = window["sampling_seed"]

    def forecast(self):
        from src.four_dim_srda.utils.sr_da_helper import initialize_and_itegrate_lr_cfd_model_for_forecast
        return initialize_and_itegrate_lr_cfd_model_for_forecast(
            last_hr_pv0=self.state, num_integrate_steps=2,
            low_pass_filter=self.filter, cfg_cfd=self.cfg)

    def conditions(self, lr):
        from src.four_dim_srda.utils.sr_da_helper import make_preprocessed_lr_for_forecast, make_preprocessed_obs_for_forecast
        from src.filo_srda_qg.sr_prior.source import build_bridge_source
        from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import make_mask_from_history
        count = self.member_stop - self.member_start
        history = [self.raw[i][None].repeat(count, 1, 1, 1) if i in (0, 4)
                   else torch.full_like(self.state, float("nan")) for i in range(5)]
        preprocessed = make_preprocessed_lr_for_forecast(lr_forecast=lr, test_dataset=self.dataset, cfg_cfd=self.cfg)
        source = build_bridge_source(preprocessed.to(self.device).float())
        obs = make_preprocessed_obs_for_forecast(hr_obs=history, assimilation_period=4, forecast_span=4,
                                               test_dataset=self.dataset, cfg_cfd=self.cfg).to(self.device, dtype=source.dtype)
        masks = make_mask_from_history(history=history, assimilation_period=4, forecast_span=4, interval=4).to(self.device)
        return source, obs, masks


def prepare_letkf_operators(args, window):
    from src.four_dim_srda.da.letkf_block_processing import LETKFBlockProcessingConfig, make_splitted_grid_and_its_center_indices_array, calc_distances_from_obs_grid_points, calc_decay_factors_of_localization
    from src.four_dim_srda.utils.enkf_helper import get_obs_matrix
    from src.four_dim_srda.config.experiment_config import CFDConfig
    cfg = LETKFBlockProcessingConfig.load(Path(args.root) / LETKF_CONFIG)
    cfd = CFDConfig.load(Path(args.root) / CFD_CONFIG)
    # QGModel.get_grids has no dependence on the evolving state.
    from src.qg_model.qg_model import QGModel
    cfd.hr_base_config.device = "cpu"
    geometry = QGModel(cfd.hr_base_config, show_input_cfg_info=False)
    x, y, z = geometry.get_grids()
    obs_matrix = get_obs_matrix(obs=window["raw_history"][4], nz=cfg.nz, ny=cfg.ny, nx=cfg.nx, device="cpu")
    dist_xy, dist_z = calc_distances_from_obs_grid_points(obs_matrix=obs_matrix, flattend_x_coords=x.reshape(-1),
                                                       flattend_y_coords=y.reshape(-1), flattend_z_coords=z.reshape(-1), cfg=cfg)
    decay = calc_decay_factors_of_localization(dist_xy=dist_xy, dist_z=dist_z, obs_matrix=obs_matrix, cfg=cfg)
    blocks, centers = make_splitted_grid_and_its_center_indices_array(cfg)
    return dict(cfg=cfg, obs_matrix=obs_matrix, dist_xy=dist_xy, dist_z=dist_z, decay_factors=decay,
                block_idx=blocks, center_idx=centers)


def timed(function, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    torch.cuda.synchronize(device)
    return value, time.perf_counter() - started
