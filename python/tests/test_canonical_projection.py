import numpy as np

from analysis.tables.canonical import HR, hr_field


def test_historical_4d_srda_projection_on_native_hr_grid():
    field = np.ones(HR, dtype=np.float32)
    projected = hr_field(field, deterministic_archive=True)
    assert np.all(projected[:, 0, :] == 0)
    np.testing.assert_array_equal(projected[:, 1:, :], field[:, 1:, :])
    np.testing.assert_array_equal(hr_field(field), field)
