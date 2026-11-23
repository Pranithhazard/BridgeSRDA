"""Isolated native-M30 recursive pilot; no scoring, archives or output writes.

The caller owns observations and must not mutate them during cycling. Geometry
is prepared once from frame 4 on CPU, using the frozen-window benchmark helper.
Initialization instead uses the *native* M30 CFD config and production draw.
Neither setup nor reset changes the global default dtype. In particular, the
LETKF plan retains production's default-dtype identity-matrix rounding.

Timing includes the QG getter's GPU -> CPU -> GPU round trip in forecast,
current raw-observation transfer in analysis, and member-wise initialize_pv in
postprocessing. Setup/reset, input validation and caller diagnostics/I/O are
outside core_seconds. No forecast follows the final analysis at frame 796.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.benchmark_inference_timing import (
    CFD_CONFIG as GEOMETRY_CFD_CONFIG,
    LETKF_CONFIG,
    prepare_letkf_operators,
)
from scripts.filo_srda_qg.letkf.evaluate_letkf_uq_narrow_jet import (
    initial_ensemble,
)
from src.filo_srda_qg.timing_benchmark_qg.letkf_kernels import (
    hr_assimilate_batched,
    prepare_letkf_plan,
)
from src.four_dim_srda.config.experiment_config import CFDConfig
from src.four_dim_srda.utils.random_seed_helper import set_seeds
from src.qg_model.qg_model import QGModel


CFD_CONFIG = (
    "python/configs/four_dim_srda/handover/perform_letkf_hr_using_uhr/"
    "cfd_cfg_ne030_na3e-03_uhr_seed9990_9992.yml"
)
BENCHMARK_SOURCE = (
    "python/scripts/filo_srda_qg/denseobs_conditional_sb_srda_qg/"
    "benchmark_inference_timing.py"
)
INITIALIZATION_SOURCE = (
    "python/scripts/filo_srda_qg/letkf/"
    "evaluate_letkf_uq_narrow_jet.py"
)


class OptimizedLETKFCycler:
    """Sequential step(4), step(8), ..., step(796), with replayable warmup.

    step returns (FP64 member analyses, timing dict, kernel diagnostics dict).
    reset restores the original production draw, not a new random draw, and
    clears the sequential counter and any failed-step state. Initial/archive
    equivalence and all truth-based checks belong to the caller.
    """

    shape = (16, 65, 128)
    ensemble_size = 30
    block_chunk = 256
    first_cycle_index = 4
    final_cycle_index = 796

    @torch.no_grad()
    def __init__(self, root: Path, seed: int, device, observations: torch.Tensor):
        self.root = Path(root).resolve()
        self.seed = seed
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("FP64 cycler requires an explicit CPU or CUDA device")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.default_dtype = torch.get_default_dtype()
        self.source_paths = [self.root / p for p in (
            CFD_CONFIG, LETKF_CONFIG, GEOMETRY_CFD_CONFIG,
            BENCHMARK_SOURCE, INITIALIZATION_SOURCE,
            "python/src/filo_srda_qg/timing_benchmark_qg/cycling_letkf.py",
            "python/src/filo_srda_qg/timing_benchmark_qg/letkf_kernels.py",
            "python/src/four_dim_srda/da/letkf_block_processing.py",
            "python/src/four_dim_srda/utils/enkf_helper.py",
            "python/src/four_dim_srda/utils/random_seed_helper.py",
            "python/src/qg_model/qg_model.py",
            "python/src/qg_model/jet_maker.py",
        )]
        if (not isinstance(observations, torch.Tensor)
                or observations.ndim != 4
                or tuple(observations.shape[1:]) != self.shape
                or observations.shape[0] <= self.final_cycle_index
                or not observations.is_floating_point()):
            raise ValueError("observations must be floating (T,16,65,128), T >= 797")
        self.observations = observations.detach()
        canonical = self.observations[self.first_cycle_index].cpu()
        self._sensor_mask = ~torch.isnan(canonical)
        if not self._sensor_mask.any() or not torch.isfinite(canonical[self._sensor_mask]).all():
            raise ValueError("Frame 4 must contain finite sensors and NaNs elsewhere")

        self.cfg = CFDConfig.load(self.root / CFD_CONFIG)
        hr, clock, da = self.cfg.hr_base_config, self.cfg.time_config, self.cfg.da_config
        if (hr.nz, hr.ny, hr.nx) != self.shape or hr.n_batch != self.ensemble_size:
            raise ValueError("Native CFD grid/batch must be (30,16,65,128)")
        if hr.precision != "double":
            raise ValueError("Native CFD precision must remain double")
        if (isinstance(seed, bool) or not isinstance(seed, int) or seed != 9991
                or not self.cfg.seed_config.uhr_seed_start <= seed <= self.cfg.seed_config.uhr_seed_end):
            raise ValueError("This authorized one-seed pilot requires seed 9991")
        if (clock.start_time != 0 or clock.end_time != 200
                or clock.output_hr_dt != 0.25 or da.assimilation_interval != 4
                or da.assimilation_dt != 1.0):
            raise ValueError("Pilot requires physical times 0..200 with frame spacing 0.25 and stride 4")
        if not math.isfinite(clock.hr_dt) or clock.hr_dt <= 0:
            raise ValueError("hr_dt must be finite and positive")
        self.steps_per_output = round(clock.output_hr_dt / clock.hr_dt)
        if (self.steps_per_output < 1
                or not math.isclose(self.steps_per_output * clock.hr_dt, 0.25, rel_tol=0, abs_tol=1e-12)
                or 4 * self.steps_per_output != round(1.0 / clock.hr_dt)):
            raise ValueError("Four output increments must equal exactly one physical unit")
        hr.device = str(self.device)
        self.model = QGModel(hr, show_input_cfg_info=False)

        # The benchmark helper constructs geometry on CPU, independently of
        # evolving QG state. Do not retain its large global dense operators.
        operators = prepare_letkf_operators(
            SimpleNamespace(root=self.root), {"raw_history": {4: canonical}},
        )
        cfg_letkf = operators["cfg"]
        if ((cfg_letkf.nz, cfg_letkf.ny, cfg_letkf.nx) != self.shape
                or cfg_letkf.num_ens != self.ensemble_size):
            raise ValueError("LETKF config must describe the native M30 HR grid")
        self.plan = prepare_letkf_plan(**operators, block_chunk=self.block_chunk).to(self.device)
        del operators

        # Match production ordering: all deterministic setup first; no work
        # between this seed reset and the production initial-ensemble draw.
        set_seeds(seed=(seed + 1) * 457, use_deterministic=True)
        initial = initial_ensemble(self.cfg)
        self._validate_ensemble(initial, "initial ensemble", check_device=False)
        self._initial_ensemble = initial.detach().cpu().clone()
        self.reset()

    @property
    def initial_ensemble(self) -> torch.Tensor:
        """A CPU FP64 copy for optional caller-side initialization checks."""
        return self._initial_ensemble.clone()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _validate_ensemble(self, value, label, *, check_device=True):
        if (not isinstance(value, torch.Tensor)
                or tuple(value.shape) != (self.ensemble_size, *self.shape)
                or value.dtype != torch.float64):
            raise ValueError(f"{label} must be FP64 (30,16,65,128)")
        if check_device and value.device != self.device:
            raise ValueError(f"{label} is not on the requested device")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite {label}")

    @torch.no_grad()
    def reset(self):
        """Discard warmup/failed state and restore the exact original draw."""
        self._needs_reset = True
        self.model.initialize_pv(grid_pv=self._initial_ensemble.clone())
        self._synchronize()
        self.next_cycle_index = self.first_cycle_index
        self._needs_reset = False

    @torch.no_grad()
    def step(self, cycle_index: int):
        if self._needs_reset:
            raise RuntimeError("Previous step/reset failed; call reset() before cycling")
        if (isinstance(cycle_index, bool) or not isinstance(cycle_index, int)
                or cycle_index != self.next_cycle_index
                or cycle_index > self.final_cycle_index):
            raise ValueError(f"Expected sequential cycle {self.next_cycle_index} within 4..796")
        if torch.get_default_dtype() != self.default_dtype:
            raise RuntimeError("Global default dtype changed since plan preparation")
        raw = self.observations[cycle_index]
        # Missing/new sensors are errors, not a request to rebuild the geometry.
        mask = ~torch.isnan(raw)
        if not torch.equal(mask.cpu(), self._sensor_mask) or not torch.isfinite(raw[mask]).all():
            raise ValueError(f"Nonfinite sensor or changed observation mask at frame {cycle_index}")

        diagnostics = {}
        self._synchronize()
        started = time.perf_counter()
        self._needs_reset = True
        # Preserve production's four output-boundary calls (100 steps each).
        for _ in range(4):
            self.model.integrate_n_steps(
                dt_per_step=self.cfg.time_config.hr_dt, n_steps=self.steps_per_output,
            )
        forecast = self.model.get_grid_pv().to(device=self.device)
        self._validate_ensemble(forecast, "forecast")
        self._synchronize()
        forecast_done = time.perf_counter()
        analysis_all = hr_assimilate_batched(
            plan=self.plan, hr_pv=forecast,
            gt_added_noise=raw.to(device=self.device, dtype=torch.float64),
            fallback="production", diagnostics=diagnostics,
        )
        diagnostics["fallback_block_ids"] = list(diagnostics["production_fallback_ids"])
        self._synchronize()
        analysis_done = time.perf_counter()
        self._validate_ensemble(analysis_all, "analysis")
        self.model.initialize_pv(grid_pv=analysis_all)
        self._synchronize()
        finished = time.perf_counter()
        self.next_cycle_index += 4
        self._needs_reset = False
        timing = dict(
            forecast_seconds=forecast_done - started,
            conditioning_seconds=0.0,
            analysis_seconds=analysis_done - forecast_done,
            postprocess_seconds=finished - analysis_done,
            core_seconds=finished - started,
        )
        return analysis_all, timing, diagnostics
