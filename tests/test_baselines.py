import numpy as np
import torch

from cke.baselines import LocalMLP, PatternScaling
from cke.data import ClimateBenchDataset


def test_pattern_scaling_recovers_linear_pattern():
    rng = np.random.default_rng(1)
    T, H, W = 40, 6, 8
    cum_co2 = np.linspace(0, 4000, T)
    tg = 5e-4 * cum_co2
    a, b = rng.normal(size=(H, W)), rng.normal(size=(H, W))
    y = a[None] + b[None] * tg[:, None, None]
    m = PatternScaling().fit(np.stack([cum_co2, np.ones(T)], 1), y, tg)
    assert np.allclose(m.a, a, atol=1e-6) and np.allclose(m.b, b, atol=1e-6)
    assert np.allclose(m.predict(np.stack([cum_co2, np.ones(T)], 1)), y, atol=1e-5)


def test_local_mlp_forward_shape():
    T, H, W = 3, 6, 8
    ds = ClimateBenchDataset(np.random.rand(T, 2), np.random.rand(T, 2, H, W), np.random.rand(T, 2, H, W))
    x, y = ds[0]
    assert x.shape == (4, H, W) and y.shape == (2, H, W)
    model = LocalMLP(4, 2, hidden=(8, 8))
    assert model(ds.x).shape == (T, 2, H, W)
    assert sum(p.numel() for p in model.parameters()) < 1000
