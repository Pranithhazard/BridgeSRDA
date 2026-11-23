import numpy as np
import pytest
import torch
from types import SimpleNamespace

from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg import benchmark_cycling_worker as worker
from src.filo_srda_qg.timing_benchmark_qg import cycling_bridge as bridge_module
from scripts.filo_srda_qg.denseobs_conditional_sb_srda_qg.summarize_reduced_cost_evaluation import horizontal_laplacian as reference_laplacian
from src.four_dim_srda.utils.ssim import MSSIM


def test_centers_match_199_analyses_and_snapshots():
    centers = worker.physical_centers()
    assert len(centers) == 199 and centers[0] == 4 and centers[-1] == 796
    assert [centers[i] * .25 for i in (49, 124, 192)] == [50, 125, 193]
    assert worker.physical_centers(2) == [4, 8]
    for n in (0, 200):
        with pytest.raises(ValueError):
            worker.physical_centers(n)


def test_reference_allows_saved_terminal_field_without_changing_centers():
    worker.validate_reference_shape((800, 32, 129, 256))
    worker.validate_reference_shape((801, 32, 129, 256))
    for shape in ((799, 32, 129, 256), (801, 16, 65, 128)):
        with pytest.raises(ValueError):
            worker.validate_reference_shape(shape)


def test_laplacian_matches_canonical_and_identical_point_scores():
    torch.manual_seed(2)
    value = torch.randn(5, 12, 12)
    np.testing.assert_allclose(worker.horizontal_laplacian(value).numpy(), reference_laplacian(value.numpy()), atol=2e-6)
    ssim = MSSIM(window_3d_size=(5, 11, 11), use_gaussian=False)
    metrics = worker.point_metrics(value, value, ssim)
    assert metrics["hr_maer"] == 0 and metrics["laplacian_pv_rmse"] == 0
    assert abs(metrics["mssim_loss"]) < 1e-6


def test_bridge_feedback_is_current_member_frame_and_sequential(monkeypatch):
    # Exercise real adapter stepping on CPU with an isolated tiny sampler.
    cycler = bridge_module.CompiledBridgeCycler.__new__(bridge_module.CompiledBridgeCycler)
    cycler.device = torch.device("cpu")
    cycler.ensemble_size = 5
    cycler.seed = 9991
    cycler.shape = (2, 4, 3)
    cycler.observations = torch.zeros(12, *cycler.shape)
    cycler.initial = torch.zeros(5, *cycler.shape)
    cycler.context = SimpleNamespace(state=cycler.initial, forecast=lambda: cycler.context.state.clone(),
                                     conditions=lambda forecast: (forecast, forecast, forecast))
    cycler.model = None
    cycler.bridge = None
    seen = []
    def sample(**kwargs):
        seen.append(kwargs)
        out = torch.zeros(5, 9, 2, 3, 3)
        out[:, 4] = torch.arange(5)[:, None, None, None] * .1 + .5
        out[:, 8] = 99
        return out, 20
    monkeypatch.setattr(bridge_module, "sample_bridge_heun", sample)
    monkeypatch.setattr(bridge_module.h, "timed", lambda function, device: (function(), .01))
    cycler.reset()
    samples, timing, details = cycler.step(4)
    assert samples.shape == (5, 2, 4, 3)
    assert torch.equal(cycler.context.state, samples)
    assert samples.max() < 4 and samples[0, 0, 0, 0] != samples[1, 0, 0, 0]
    assert seen[0]["sampling_seed"] == 7042 + 9991 * 1000 + 4
    assert seen[0]["member_chunk_size"] == 5 and seen[0]["nfe_budget"] == 20
    assert details["feedback_frame"] == 4 and timing["core_seconds"] == .04
    with pytest.raises(ValueError):
        cycler.step(12)
    cycler.step(8)
    cycler.reset()
    assert cycler.previous_cycle == 0 and torch.equal(cycler.context.state, cycler.initial)
