import pytest

from src.filo_srda_qg.sb_prior_retrain_qg.cycling_schedule import (
    cycling_schedule, next_window_start,
)


def test_primary_schedule_is_unchanged():
    centers, feedback = cycling_schedule(num_times=800, assimilation_period=4, forecast_span=4)
    assert centers == list(range(4, 800, 4)) and len(centers) == 199
    assert feedback == 4


def test_terminal_nowcast_is_next_window_start_not_next_analysis():
    centers, feedback = cycling_schedule(num_times=800, assimilation_period=4, forecast_span=4, stride=2)
    assert centers == list(range(4, 800, 8)) and len(centers) == 100
    assert feedback == 8
    for first, second in zip(centers, centers[1:]):
        assert next_window_start(center=first, assimilation_period=4, forecast_span=4, stride=2) == second - 4
        assert first + 4 < second  # feedback has timestamp 2, not 3, after first t=1 call


def test_stride_two_rejects_incompatible_horizon():
    with pytest.raises(ValueError):
        cycling_schedule(num_times=800, assimilation_period=4, forecast_span=3, stride=2)
