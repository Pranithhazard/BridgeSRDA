"""CPU float64 equivalence to the actual, unchanged production entrypoint."""

from dataclasses import replace

import pytest
import torch
from src.filo_srda_qg.timing_benchmark_qg.letkf_kernels import (
    hr_assimilate_batched,
    prepare_letkf_plan,
)
from src.four_dim_srda.da.letkf_block_processing import (
    LETKFBlockProcessingConfig,
    calc_decay_factors_of_localization,
    calc_distances_from_obs_grid_points,
    hr_assimilate_with_existing_data,
    make_splitted_grid_and_its_center_indices_array,
)


RTOL, ATOL = 1e-10, 1e-11


def _case(members=7, shape=(2, 5, 7), block_size=3, obs_interval=2):
    nz, ny, nx = shape
    cfg = LETKFBlockProcessingConfig(
        data_dir="dummy", train_valid_test_ratios=[1, 1, 1],
        train_valid_test_kinds="test", num_simulation=1,
        nx=nx, ny=ny, nz=nz, num_ens=members,
        obs_grid_interval_x=obs_interval, obs_grid_interval_y=obs_interval,
        obs_noise_std=0.037, cylinder_radius=2.0, cylinder_height=2.0,
        inflation_factor=1.13, localization_radius=1.7, block_size=block_size,
    )
    generator = torch.Generator().manual_seed(28)
    state = torch.randn((members, nz, ny, nx), generator=generator,
                        dtype=torch.float64) * 0.1
    raw = torch.full(shape, float("nan"), dtype=torch.float64)
    observed = torch.randn(shape, generator=generator, dtype=torch.float64) * 0.2
    raw[:, ::obs_interval, ::obs_interval] = observed[:, ::obs_interval, ::obs_interval]
    # Same isnan-based mask and flattened lattice order as get_obs_matrix.
    sensors = torch.where(~torch.isnan(raw.reshape(-1)))[0]
    h = torch.zeros((sensors.numel(), raw.numel()), dtype=torch.float64)
    h[torch.arange(sensors.numel()), sensors] = 1
    z, y, x = torch.meshgrid(
        *(torch.arange(n, dtype=torch.float64) for n in shape), indexing="ij",
    )
    xy, dz = calc_distances_from_obs_grid_points(
        obs_matrix=h, flattend_x_coords=x.reshape(-1),
        flattend_y_coords=y.reshape(-1), flattend_z_coords=z.reshape(-1), cfg=cfg,
    )
    decay = calc_decay_factors_of_localization(
        dist_xy=xy, dist_z=dz, obs_matrix=h, cfg=cfg,
    )
    blocks, centers = make_splitted_grid_and_its_center_indices_array(cfg)
    setup = dict(cfg=cfg, obs_matrix=h, dist_xy=xy, dist_z=dz,
                 decay_factors=decay, block_idx=blocks, center_idx=centers)
    return setup, state, raw


def _production(setup, state, raw):
    # This preprocessing is performed by the actual production cycling caller.
    return hr_assimilate_with_existing_data(
        **setup, hr_pv=state, gt_added_noise=torch.nan_to_num(raw, nan=1e10),
    )


@pytest.mark.parametrize("members", [3, 7, 30])
@pytest.mark.parametrize("block_chunk", [1, 4, 100])
def test_actual_production_equivalence_ragged_blocks(members, block_chunk):
    setup, state, raw = _case(members)
    state_before, raw_before = state.clone(), raw.clone()
    expected = _production(setup, state, raw)
    plan = prepare_letkf_plan(**setup, block_chunk=block_chunk)
    diagnostics = {}
    actual = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw,
                                   diagnostics=diagnostics)
    if members == 3:
        assert diagnostics["production_fallback_blocks"] == 0
    assert actual.dtype == torch.float64 and actual.shape == state.shape
    assert len({b.numel() for b in setup["block_idx"]}) == 4  # 9, 6, 3, 2
    assert torch.equal(state, state_before)
    torch.testing.assert_close(raw, raw_before, equal_nan=True, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    # All edge and boundary points are compared, not only an interior crop.
    assert not torch.equal(actual[:, :, 0, :], state[:, :, 0, :])


def test_alternative_small_3d_grid_and_noncontiguous_state():
    setup, state, raw = _case(members=5, shape=(3, 4, 5), block_size=2)
    state = state.transpose(-1, -2).contiguous().transpose(-1, -2)
    assert not state.is_contiguous()
    plan = prepare_letkf_plan(**setup, block_chunk=5)
    torch.testing.assert_close(
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw),
        _production(setup, state, raw), rtol=RTOL, atol=ATOL,
    )


def test_m30_batched_path_without_degenerate_fallback():
    setup, state, raw = _case(members=30, obs_interval=1)
    setup["cfg"] = replace(setup["cfg"], cylinder_radius=10.0)
    plan = prepare_letkf_plan(**setup, block_chunk=4)
    diagnostics = {}
    actual = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw,
                                   fallback="error", diagnostics=diagnostics)
    assert diagnostics["production_fallback_blocks"] == 0
    torch.testing.assert_close(actual, _production(setup, state, raw),
                               rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_preserves_default_dtype_identity_rounding(default_dtype):
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        setup, state, raw = _case()
        plan = prepare_letkf_plan(**setup)
        assert plan.prior.dtype == default_dtype
        torch.testing.assert_close(
            hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw),
            _production(setup, state, raw), rtol=RTOL, atol=ATOL,
        )
    finally:
        torch.set_default_dtype(original)


def test_observation_order_and_duplicate_sensor_rows_are_not_changed():
    setup, state, raw = _case()
    order = torch.cat((torch.arange(setup["obs_matrix"].shape[0]).flip(0),
                       torch.tensor([0])))
    for key in ("obs_matrix", "dist_xy", "dist_z", "decay_factors"):
        setup[key] = setup[key][order]
    plan = prepare_letkf_plan(**setup, block_chunk=4)
    torch.testing.assert_close(
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw),
        _production(setup, state, raw), rtol=RTOL, atol=ATOL,
    )


def test_exact_cylinder_ties_and_supplied_center_indices():
    setup, state, raw = _case()
    # Deliberately use the first point rather than production's flattened median.
    setup["center_idx"] = [b[0] for b in setup["block_idx"]]
    plan = prepare_letkf_plan(**setup, block_chunk=4)
    expected_indices = []
    for c in setup["center_idx"]:
        chosen = torch.where((setup["dist_xy"][:, c] <= 2.0)
                             & (setup["dist_z"][:, c] <= 1.0))[0]
        expected_indices.append(chosen)
    packed_indices = []
    for chunk in plan.chunks:
        for sensors, precision in zip(chunk.sensors, chunk.precision):
            packed_indices.append(sensors[precision > 0])
    assert any((setup["dist_xy"][:, c] == 2.0).any()
               for c in setup["center_idx"])
    assert (setup["dist_z"] == 1.0).any()
    for actual, expected in zip(packed_indices, expected_indices):
        assert torch.equal(actual, expected)
    torch.testing.assert_close(
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw),
        _production(setup, state, raw), rtol=RTOL, atol=ATOL,
    )


def test_covariance_and_localization_conventions_are_consequential():
    setup, state, raw = _case()
    plan = prepare_letkf_plan(**setup)
    actual = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw)
    torch.testing.assert_close(actual, _production(setup, state, raw),
                               rtol=RTOL, atol=ATOL)
    squared_r = dict(setup, cfg=replace(setup["cfg"], obs_noise_std=0.037**2))
    unsquared_decay = dict(setup, decay_factors=setup["decay_factors"].sqrt())
    for wrong in (squared_r, unsquared_decay):
        assert (actual - _production(wrong, state, raw)).abs().max() > 1e-4


def test_plan_reuse_tracks_member_perturbations_not_just_ensemble_mean():
    setup, state, raw = _case()
    plan = prepare_letkf_plan(**setup, block_chunk=3)
    first = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw)
    perturbed = state.clone()
    perturbed[0] += 0.07
    perturbed[1] -= 0.07  # Same mean but different ensemble anomalies.
    second = hr_assimilate_batched(plan=plan, hr_pv=perturbed, gt_added_noise=raw)
    torch.testing.assert_close(second, _production(setup, perturbed, raw),
                               rtol=RTOL, atol=ATOL)
    assert (first - second).abs().max() > 1e-3
    with pytest.raises(ValueError, match="ensemble size"):
        hr_assimilate_batched(plan=plan, hr_pv=state[:-1], gt_added_noise=raw)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_missing_or_nonfinite_selected_sensor_is_not_silently_skipped(bad_value):
    setup, state, raw = _case()
    plan = prepare_letkf_plan(**setup)
    raw.reshape(-1)[plan.sensor_indices[0]] = bad_value
    with pytest.raises(ValueError, match="selected sensor"):
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw)


def test_rejects_nonfinite_state_and_float32_without_coercion():
    setup, state, raw = _case()
    plan = prepare_letkf_plan(**setup)
    with pytest.raises(ValueError, match="float64"):
        hr_assimilate_batched(plan=plan, hr_pv=state.float(), gt_added_noise=raw)
    state[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite state"):
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw)


@pytest.mark.parametrize("problem", ["chunk", "operator", "empty_cylinder", "singleton",
                                    "partition", "covariance", "geometry"])
def test_invalid_preparation_is_explicit(problem):
    setup, _, _ = _case()
    if problem == "chunk":
        setup["block_chunk"] = 0
    elif problem == "operator":
        setup["obs_matrix"][0, 0] = 0.5
    elif problem == "empty_cylinder":
        setup["dist_xy"][:] = 1e6
    elif problem == "singleton":
        setup["dist_xy"][:] = 1e6
        setup["dist_xy"][0, :] = 0.0
    elif problem == "partition":
        setup["block_idx"] = setup["block_idx"][:-1]
        setup["center_idx"] = setup["center_idx"][:-1]
    elif problem == "covariance":
        setup["cfg"] = replace(setup["cfg"], obs_noise_std=0)
    else:
        setup["decay_factors"][0, setup["center_idx"][0]] = float("nan")
    with pytest.raises(ValueError):
        prepare_letkf_plan(**setup)


def test_eig_failure_propagates_without_hidden_solver_fallback(monkeypatch):
    setup, state, raw = _case()
    plan = prepare_letkf_plan(**setup)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate batched eig failure")

    monkeypatch.setattr(torch.linalg, "eig", fail)
    with pytest.raises(RuntimeError, match="deliberate batched eig failure"):
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw)


def test_four_disjoint_prepared_shards_sum_to_full_analysis_with_zero_exteriors():
    setup, state, raw = _case(members=30)
    plan = prepare_letkf_plan(**setup, block_chunk=5)
    full = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw,
                                 diagnostics={})
    combined = torch.zeros_like(full)
    for rank in range(4):
        ids = list(range(rank, len(setup["block_idx"]), 4))
        shard = plan.select_blocks(ids).to("cpu")
        assert shard.block_ids == tuple(ids)
        diagnostics = {}
        result = hr_assimilate_batched(plan=shard, hr_pv=state,
                                       gt_added_noise=raw, diagnostics=diagnostics)
        points = torch.cat([setup["block_idx"][i] for i in ids])
        outside = torch.ones(raw.numel(), dtype=torch.bool)
        outside[points] = False
        assert torch.count_nonzero(result.reshape(30, -1)[:, outside]) == 0
        assert diagnostics["blocks"] == len(ids)
        assert set(diagnostics["production_fallback_ids"]).issubset(ids)
        combined += result
    assert plan.block_ids == tuple(range(len(setup["block_idx"])))
    torch.testing.assert_close(combined, full, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(combined, _production(setup, state, raw),
                               rtol=RTOL, atol=ATOL)
    empty = plan.select_blocks([])
    assert torch.count_nonzero(hr_assimilate_batched(
        plan=empty, hr_pv=state, gt_added_noise=raw,
    )) == 0


@pytest.mark.parametrize("ids", [[0, 0], [-1], [100]])
def test_invalid_shard_selection_fails(ids):
    setup, _, _ = _case()
    plan = prepare_letkf_plan(**setup)
    with pytest.raises(ValueError, match="unique IDs"):
        plan.select_blocks(ids)


def test_degenerate_eig_fallback_is_explicitly_reported_or_disallowed():
    setup, state, raw = _case(members=30)
    plan = prepare_letkf_plan(**setup)
    with pytest.raises(RuntimeError, match="requires production fallback"):
        hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw,
                              fallback="error")
    with pytest.warns(RuntimeWarning, match="Production-arithmetic eig fallback"):
        warned = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw)
    diagnostics = {}
    reported = hr_assimilate_batched(plan=plan, hr_pv=state, gt_added_noise=raw,
                                     diagnostics=diagnostics)
    assert diagnostics["production_fallback_blocks"] == len(plan.block_ids)
    torch.testing.assert_close(warned, reported, rtol=0, atol=0)
