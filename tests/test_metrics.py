import numpy as np

from cke.metrics import all_metrics, global_mean, nrmse_global, nrmse_spatial, nrmse_total

LAT = np.linspace(-87.5, 87.5, 8)


def test_perfect_prediction_zero():
    y = np.random.rand(5, 8, 12) + 1
    m = all_metrics(y, y.copy(), LAT)
    assert all(abs(v) < 1e-12 for v in m.values())


def test_total_is_spatial_plus_5_global():
    rng = np.random.default_rng(0)
    y = rng.random((6, 8, 12)) + 1
    p = y + rng.normal(0, 0.1, y.shape)
    assert np.isclose(nrmse_total(y, p, LAT), nrmse_spatial(y, p, LAT) + 5 * nrmse_global(y, p, LAT))


def test_constant_offset():
    y = np.ones((4, 8, 12)) * 2.0
    p = y + 0.5
    assert np.isclose(nrmse_global(y, p, LAT), 0.25)
    assert np.isclose(nrmse_spatial(y, p, LAT), 0.25)
    assert np.allclose(global_mean(y, LAT), 2.0)
