"""CPU/mock contracts only: never launch native-grid CFD or remote/GPU work."""

import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.filo_srda_qg.timing_benchmark_qg import cycling_letkf as cycling
from src.four_dim_srda.da.letkf_block_processing import LETKFBlockProcessingConfig


ROOT = Path(__file__).resolve().parents[3]
SMALL_SHAPE = (1, 2, 3)


@pytest.fixture
def mocked(monkeypatch):
    """Real configs, but tiny fields and explicitly mocked numerical work."""
    events = []
    cfg = cycling.CFDConfig.load(ROOT / cycling.CFD_CONFIG)
    cfg.hr_base_config.nz, cfg.hr_base_config.ny, cfg.hr_base_config.nx = SMALL_SHAPE
    cfg_letkf = LETKFBlockProcessingConfig.load(ROOT / cycling.LETKF_CONFIG)
    cfg_letkf.nz, cfg_letkf.ny, cfg_letkf.nx = SMALL_SHAPE
    monkeypatch.setattr(cycling.OptimizedLETKFCycler, "shape", SMALL_SHAPE)
    monkeypatch.setattr(cycling.CFDConfig, "load", lambda path: copy.deepcopy(cfg))

    class Model:
        def __init__(self, config, **kwargs):
            events.append(("model", config.n_batch, config.device, config.precision))
            self.integrations = []
            self.initializations = []
            self.getters = 0

        def initialize_pv(self, grid_pv):
            self.state = grid_pv.clone()
            self.initializations.append(grid_pv.clone())
            self.t = 0.0
            events.append(("feedback",))

        def integrate_n_steps(self, *, dt_per_step, n_steps):
            self.integrations.append((dt_per_step, n_steps))
            self.state += n_steps * dt_per_step
            self.t += n_steps * dt_per_step

        def get_grid_pv(self):
            self.getters += 1
            events.append(("getter", self.t))
            return self.state.detach().cpu().clone()

    def operators(args, window):
        events.append(("geometry", args.root, window["raw_history"][4].clone()))
        assert window["raw_history"][4].device.type == "cpu"
        return {"cfg": cfg_letkf}

    class Plan:
        def to(self, device):
            self.device = device
            return self

    def prepare(**kwargs):
        events.append(("plan", kwargs["block_chunk"], torch.get_default_dtype()))
        return Plan()

    def seed(*, seed, use_deterministic):
        events.append(("seed", seed, use_deterministic))
        torch.manual_seed(seed)

    def initial(config):
        events.append(("draw", config.hr_base_config.n_batch, torch.get_default_dtype()))
        return torch.randn((config.hr_base_config.n_batch, *SMALL_SHAPE)).double()

    def assimilate(*, plan, hr_pv, gt_added_noise, fallback, diagnostics):
        assert hr_pv.dtype == gt_added_noise.dtype == torch.float64
        assert torch.isnan(gt_added_noise[..., 0, 0]).all()  # Still raw, no sentinel.
        assert fallback == "production"
        events.append(("analysis", hr_pv.clone(), gt_added_noise.clone()))
        diagnostics.update(blocks=3, production_fallback_ids=[0, 2], production_fallback_blocks=2)
        # Distinct member values make ensemble-mean feedback detectable.
        return hr_pv + torch.nan_to_num(gt_added_noise, nan=0)

    monkeypatch.setattr(cycling, "QGModel", Model)
    monkeypatch.setattr(cycling, "prepare_letkf_operators", operators)
    monkeypatch.setattr(cycling, "prepare_letkf_plan", prepare)
    monkeypatch.setattr(cycling, "set_seeds", seed)
    monkeypatch.setattr(cycling, "initial_ensemble", initial)
    monkeypatch.setattr(cycling, "hr_assimilate_batched", assimilate)
    # Fully independent fields, so tests can alter exactly one cycle.
    observations = torch.ones((800, *SMALL_SHAPE), dtype=torch.float32)
    observations[..., 0, 0] = float("nan")
    observations[8, ..., 0, 1] = 7

    def make(**kwargs):
        return cycling.OptimizedLETKFCycler(
            root=ROOT, seed=kwargs.pop("seed", 9991), device="cpu",
            observations=kwargs.pop("observations", observations), **kwargs,
        )

    return SimpleNamespace(make=make, events=events, observations=observations, cfg=cfg)


def test_native_config_and_geometry_contract():
    assert cycling.OptimizedLETKFCycler.shape == (16, 65, 128)
    assert cycling.OptimizedLETKFCycler.ensemble_size == 30
    native = cycling.CFDConfig.load(ROOT / cycling.CFD_CONFIG)
    geometry = cycling.CFDConfig.load(ROOT / cycling.GEOMETRY_CFD_CONFIG)
    tuned = LETKFBlockProcessingConfig.load(ROOT / cycling.LETKF_CONFIG)
    assert native.hr_base_config.n_batch == tuned.num_ens == 30
    assert native.hr_base_config.precision == "double"
    assert native.hr_base_config.noise_amplitude == 0.003
    assert native.jet_profile == "sech_squared_and_sigmoid"
    assert (native.seed_config.uhr_seed_start, native.seed_config.uhr_seed_end) == (9990, 9992)
    assert (tuned.obs_noise_std, tuned.inflation_factor, tuned.localization_radius) == (0.022, 1.2, 5.7)
    # The reused benchmark geometry must not import its different draw settings.
    a, b = map(dataclasses.asdict, (native.hr_base_config, geometry.hr_base_config))
    for field in ("n_batch", "noise_amplitude", "device"):
        a.pop(field)
        b.pop(field)
    assert a == b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_setup_once_seed_order_default_dtype_and_paths(mocked, dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        cycler = mocked.make()
        assert torch.get_default_dtype() == dtype
        names = [e[0] for e in mocked.events]
        assert names == ["model", "geometry", "plan", "seed", "draw", "feedback"]
        assert mocked.events[0][1:] == (30, "cpu", "double")
        assert mocked.events[2][1:] == (256, dtype)
        assert mocked.events[3][1:] == ((9991 + 1) * 457, True)
        assert mocked.events[4][1:] == (30, dtype)
        assert all(isinstance(p, Path) and p.is_absolute() and p.is_file() for p in cycler.source_paths)
        assert {ROOT / cycling.CFD_CONFIG, ROOT / cycling.LETKF_CONFIG,
                ROOT / cycling.GEOMETRY_CFD_CONFIG}.issubset(cycler.source_paths)
        cycler.step(4)
        cycler.reset()
        cycler.step(4)
        assert [e[0] for e in mocked.events].count("geometry") == 1
        assert [e[0] for e in mocked.events].count("plan") == 1
        assert [e[0] for e in mocked.events].count("draw") == 1
    finally:
        torch.set_default_dtype(previous)


def test_memberwise_recursion_current_observation_and_timing(mocked, monkeypatch):
    cycler = mocked.make()
    clock = iter([10.0, 12.0, 15.0, 19.0])
    monkeypatch.setattr(cycling.time, "perf_counter", lambda: next(clock))
    analysis, timing, diagnostics = cycler.step(4)
    expected = cycler.initial_ensemble + 1 + torch.nan_to_num(mocked.observations[4].double(), nan=0)
    torch.testing.assert_close(analysis, expected, rtol=0, atol=0)
    assert analysis.dtype == torch.float64
    torch.testing.assert_close(cycler.model.state, analysis, rtol=0, atol=0)
    assert cycler.model.integrations == [(0.0025, 100)] * 4
    assert cycler.model.getters == 1
    assert timing == dict(forecast_seconds=2.0, conditioning_seconds=0.0,
                          analysis_seconds=3.0, postprocess_seconds=4.0, core_seconds=9.0)
    assert diagnostics["production_fallback_blocks"] == 2
    assert diagnostics["fallback_block_ids"] == [0, 2]
    assert cycler.next_cycle_index == 8
    clock = iter([20.0, 21.0, 22.0, 23.0])
    second, _, _ = cycler.step(8)
    torch.testing.assert_close(
        second, analysis + 1 + torch.nan_to_num(mocked.observations[8].double(), nan=0),
        rtol=0, atol=0,
    )
    assert cycler.model.integrations == [(0.0025, 100)] * 8


def test_warmup_reset_exact_initial_and_counter_without_redraw(mocked):
    cycler = mocked.make()
    original = cycler.initial_ensemble
    warmup, _, _ = cycler.step(4)
    # Neither the returned analysis nor the exposed copy can damage reset data.
    warmup_copy = warmup.clone()
    warmup.zero_()
    cycler.initial_ensemble.zero_()
    cycler.reset()
    assert cycler.next_cycle_index == 4 and cycler.model.t == 0
    torch.testing.assert_close(cycler.model.state, original, rtol=0, atol=0)
    scored, _, _ = cycler.step(4)
    torch.testing.assert_close(scored, warmup_copy, rtol=0, atol=0)
    assert [e[0] for e in mocked.events].count("seed") == 1


def test_exactly_199_analyses_and_no_unused_tail_forecast(mocked):
    cycler = mocked.make()
    for index in range(4, 797, 4):
        _, timing, _ = cycler.step(index)
        assert all(value >= 0 for value in timing.values())
    assert len(cycler.model.integrations) == 199 * 4
    assert cycler.model.getters == 199
    assert len(cycler.model.initializations) == 200
    assert cycler.next_cycle_index == 800
    with pytest.raises(ValueError, match="within 4..796"):
        cycler.step(800)
    assert len(cycler.model.integrations) == 796


@pytest.mark.parametrize("index", [0, 8, -4, 5, 4.0, True, 800])
def test_reject_nonsequential_indices_without_forecast(mocked, index):
    cycler = mocked.make()
    with pytest.raises(ValueError, match="sequential"):
        cycler.step(index)
    assert not cycler.model.integrations


@pytest.mark.parametrize("seed", [9990, 9992, 9993, True, 9991.0])
def test_reject_unauthorized_seeds(mocked, seed):
    with pytest.raises(ValueError, match="requires seed 9991"):
        mocked.make(seed=seed)


@pytest.mark.parametrize("change", ["missing", "extra", "infinite"])
def test_reject_changed_or_nonfinite_sensors_before_forecast(mocked, change):
    cycler = mocked.make()
    if change == "extra":
        mocked.observations[4, ..., 0, 0] = 0
    else:
        mocked.observations[4, ..., 0, 1] = float("nan" if change == "missing" else "inf")
    with pytest.raises(ValueError, match="sensor|mask"):
        cycler.step(4)
    assert not cycler.model.integrations


@pytest.mark.parametrize("bad", ["shape", "short", "integer", "infinite", "all_missing"])
def test_reject_invalid_observation_input(mocked, bad):
    obs = mocked.observations.clone()
    if bad == "shape":
        obs = obs[..., :1]
    elif bad == "short":
        obs = obs[:796]
    elif bad == "integer":
        obs = torch.zeros_like(obs, dtype=torch.int64)
    elif bad == "infinite":
        obs[4, ..., 0, 1] = float("inf")
    else:
        obs[4] = float("nan")
    with pytest.raises(ValueError, match="observations|Frame 4"):
        mocked.make(observations=obs)


@pytest.mark.parametrize("bad", ["members", "precision", "interval", "dt"])
def test_reject_config_contract_changes(mocked, bad):
    if bad == "members":
        mocked.cfg.hr_base_config.n_batch = 1
    elif bad == "precision":
        mocked.cfg.hr_base_config.precision = "single"
    elif bad == "interval":
        mocked.cfg.da_config.assimilation_interval = 2
    else:
        mocked.cfg.time_config.hr_dt = 0.003
    with pytest.raises(ValueError):
        mocked.make()


@pytest.mark.parametrize("bad", ["dtype", "shape", "nonfinite", "solver"])
def test_failed_analysis_requires_reset_and_never_feedbacks(mocked, monkeypatch, bad):
    cycler = mocked.make()
    original = cycling.hr_assimilate_batched

    def fail(**kwargs):
        if bad == "solver":
            raise RuntimeError("solver failed")
        value = original(**kwargs)
        if bad == "dtype":
            return value.float()
        if bad == "shape":
            return value[:1]
        value[0, 0, 0, 0] = float("nan")
        return value

    monkeypatch.setattr(cycling, "hr_assimilate_batched", fail)
    with pytest.raises((ValueError, FloatingPointError, RuntimeError)):
        cycler.step(4)
    assert len(cycler.model.initializations) == 1
    with pytest.raises(RuntimeError, match="reset"):
        cycler.step(4)
    assert len(cycler.model.integrations) == 4
    monkeypatch.setattr(cycling, "hr_assimilate_batched", original)
    cycler.reset()
    cycler.step(4)


def test_default_dtype_change_is_not_silently_accepted(mocked):
    cycler = mocked.make()
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64 if original == torch.float32 else torch.float32)
        with pytest.raises(RuntimeError, match="default dtype changed"):
            cycler.step(4)
        assert not cycler.model.integrations
    finally:
        torch.set_default_dtype(original)


def test_cpu_never_synchronizes_cuda_and_cuda_targets_exact_device(mocked, monkeypatch):
    cycler = mocked.make()
    synchronizations = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronizations.append(device))
    cycler.step(4)
    assert not synchronizations
    # Device-only helper exercise: no CUDA tensors, kernels or actual sync.
    cycler.device = torch.device("cuda:3")
    cycler._synchronize()
    assert synchronizations == [torch.device("cuda:3")]


def test_explicit_sync_at_every_core_boundary(mocked, monkeypatch):
    cycler = mocked.make()
    boundaries = []
    monkeypatch.setattr(cycler, "_synchronize", lambda: boundaries.append(len(cycler.model.integrations)))
    cycler.step(4)
    assert boundaries == [0, 4, 4, 4]
