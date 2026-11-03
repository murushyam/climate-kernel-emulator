import numpy as np
import torch

from cke.baselines import LocalMLP
from cke.utils.seed import seed_everything


def _draw():
    seed_everything(123)
    m = LocalMLP(4, 2, hidden=(8,))
    return np.random.rand(3), torch.rand(3), next(m.parameters()).detach().clone()


def test_seeding_is_deterministic():
    a, b = _draw(), _draw()
    assert np.array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
