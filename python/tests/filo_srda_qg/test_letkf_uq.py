from __future__ import annotations

import numpy as np
import torch
from src.filo_srda_qg.shared_srda.metrics import (
    ensemble_summary,
    rank_histogram_counts,
)
from src.filo_srda_qg.letkf.letkf_uq import (
    ensemble_diagnostics,
    fixed_subensemble_indices,
    hr_truth_from_uhr,
)


def test_fixed_subensemble_indices_are_reproducible_unique_and_sorted() -> None:
    first = fixed_subensemble_indices(ensemble_size=100, subset_size=30, seed=42)
    second = fixed_subensemble_indices(ensemble_size=100, subset_size=30, seed=42)
    assert np.array_equal(first, second)
    assert first.shape == (30,)
    assert np.array_equal(first, np.sort(first))
    assert len(np.unique(first)) == 30
    assert first.min() >= 0 and first.max() < 100


def test_letkf_diagnostics_match_existing_joint_sfm_metrics() -> None:
    generator = torch.Generator().manual_seed(17)
    samples = torch.randn(7, 2, 3, generator=generator)
    target = torch.randn(2, 3, generator=generator)
    actual = ensemble_diagnostics(samples, target)
    expected = ensemble_summary(
        samples.unsqueeze(0),
        target.unsqueeze(0),
        pv_min=0.0,
        pv_max=1.0,
    )
    for key in (
        "physical_mae",
        "physical_rmse",
        "physical_spread_rms",
        "spread_rmse_ratio",
        "physical_crps",
        "coverage_50",
        "coverage_80",
        "coverage_90",
        "physical_interval_width_50",
        "physical_interval_width_80",
        "physical_interval_width_90",
    ):
        assert np.isclose(actual.metrics[key], expected[key], rtol=1e-6, atol=1e-7)
    expected_ranks = rank_histogram_counts(samples.unsqueeze(0), target.unsqueeze(0))
    assert torch.equal(actual.rank_histogram, expected_ranks)
    assert torch.allclose(actual.mean, samples.mean(dim=0))
    assert torch.allclose(actual.spread, samples.std(dim=0, unbiased=True))


def test_hr_truth_from_uhr_matches_established_pooling_contract() -> None:
    uhr = torch.arange(4 * 5 * 6, dtype=torch.float64).reshape(4, 5, 6)
    actual = hr_truth_from_uhr(uhr, nz=2, ny=3, nx=3)
    expected_interior = torch.nn.functional.avg_pool3d(
        uhr[:, 1:, :].unsqueeze(0), kernel_size=2
    )[0]
    assert actual.shape == (2, 3, 3)
    assert torch.count_nonzero(actual[:, 0]) == 0
    assert torch.equal(actual[:, 1:], expected_interior)
