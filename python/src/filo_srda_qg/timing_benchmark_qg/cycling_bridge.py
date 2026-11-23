"""Isolated recursive Bridge pilot using the screened compiled-BF16 policy."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg import benchmark_inference_timing as h
from scripts.filo_srda_qg.sb_prior_retrain_qg.evaluate_narrow_jet import make_initial_hr_pv
from src.filo_srda_qg.denseobs_conditional_sb_srda_qg.checkpoint import load_training_checkpoint
from src.filo_srda_qg.shared_srda.narrow_jet import normalized_windows_to_physical
from src.filo_srda_qg.sb_prior_retrain_qg.bridge import ContinuousQuadraticBridge, ContinuousQuadraticBridgeConfig
from src.filo_srda_qg.sb_prior_retrain_qg.config import load_config
from src.filo_srda_qg.timing_benchmark_qg.bridge_sampler import sample_bridge_heun
from src.filo_srda_qg.timing_benchmark_qg.inference_network import InferenceDrift
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.four_dim_srda.utils.random_seed_helper import set_seeds


class CompiledBridgeCycler:
    def __init__(self, *, root: Path, seed: int, device, observations, ensemble_size: int):
        if ensemble_size not in (5, 10, 30):
            raise ValueError("This pilot is restricted to M5/M10/M30")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 CUDA support is required; no fallback")
        self.device = torch.device(device)
        self.ensemble_size = ensemble_size
        self.seed = seed
        self.observations = observations
        self.shape = (16, 65, 128)
        self.checkpoint = root / "work/experiments/073a_denseobs_conditional_sb_srda_qg/checkpoints" / h.RUN / "checkpoint_epoch2000.pth"
        self.source_paths = [root / h.CFD_CONFIG, root / h.SB_CONFIG, root / h.SRDA_CONFIG, self.checkpoint]
        config = load_config(root / h.SB_CONFIG)
        if config["conditioning"]["interpolation_method"] != "periodic_bilinear":
            raise ValueError("Unexpected observation interpolation")
        raw_model, self.checkpoint_metadata = load_training_checkpoint(
            config=config, checkpoint_path=self.checkpoint, device=self.device, weights_kind="ema")
        if self.checkpoint_metadata["checkpoint_epoch"] != 2000:
            raise RuntimeError("Wrong EMA checkpoint epoch")
        self.model = InferenceDrift(raw_model, bf16=True, compile_network=True)
        self.bridge = ContinuousQuadraticBridge(ContinuousQuadraticBridgeConfig(**config["bridge"]))
        # Exactly the production evaluator's single initial draw, before M repeats.
        cfg_base = CFDConfig.load(root / h.CFD_CONFIG)
        set_seeds(555 * seed, use_deterministic=True)
        initial = make_initial_hr_pv(copy.deepcopy(cfg_base)).to(device=self.device).float()
        if tuple(initial.shape) != (1, *self.shape):
            raise ValueError(f"Unexpected initial Bridge state {initial.shape}")
        self.initial = initial.repeat(ensemble_size, 1, 1, 1)
        window = dict(sb_state=self.initial, raw_history=observations[:5], sampling_seed=0)
        self.context = h.BridgeWindow(SimpleNamespace(root=root), window, self.device, member_stop=ensemble_size)
        self.reset()

    def reset(self):
        self.context.state = self.initial.clone()
        self.previous_cycle = 0

    def step(self, cycle_index: int):
        if cycle_index != self.previous_cycle + 4:
            raise ValueError("Expected consecutive normal-cycling centers, stride=4 output indices")
        self.context.seed = 7042 + self.seed * 1000 + cycle_index
        forecast, forecast_s = h.timed(self.context.forecast, self.device)
        def conditions_with_history():
            self.context.raw = self.observations[cycle_index - 4:cycle_index + 1].to(self.device, dtype=torch.float32)
            return self.context.conditions(forecast)
        conditions, condition_s = h.timed(conditions_with_history, self.device)
        source, obs, masks = conditions
        def sample():
            value, nfe = sample_bridge_heun(
                model=self.model, source=source, observations=obs, masks=masks,
                bridge=self.bridge, sampling_seed=self.context.seed,
                member_chunk_size=self.ensemble_size, nfe_budget=20)
            expected_shape = (self.ensemble_size, 9, self.shape[0], self.shape[1] - 1, self.shape[2])
            if nfe != 20 or value.dtype != torch.float32 or tuple(value.shape) != expected_shape:
                raise RuntimeError("Changed NFE, ensemble, window, or integration precision")
            return value
        windows, sampling_s = h.timed(sample, self.device)
        def feedback():
            current = normalized_windows_to_physical(windows[:, 4], pv_min=-2.88, pv_max=2.92).float()
            if not torch.isfinite(current).all():
                raise FloatingPointError("Nonfinite Bridge feedback")
            self.context.state = current.clone()
            return current
        samples, post_s = h.timed(feedback, self.device)
        self.previous_cycle = cycle_index
        timing = dict(forecast_seconds=forecast_s, conditioning_seconds=condition_s,
                      analysis_seconds=sampling_s, postprocess_seconds=post_s,
                      core_seconds=forecast_s + condition_s + sampling_s + post_s)
        return samples, timing, dict(nfe_per_member=20, total_member_nfe=20 * self.ensemble_size,
                                    feedback_frame=4, cycling_stride=1)
