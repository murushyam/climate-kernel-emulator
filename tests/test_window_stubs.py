from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from cke.data import select_years, stack_scenarios
from test_data import _write

CONF = Path(__file__).resolve().parents[1] / "configs"


def test_select_years(tmp_path):
    _write(tmp_path, "a", T=5)
    d = stack_scenarios(tmp_path, ["a"])
    assert list(d["years"]) == [0, 1, 2, 3, 4]
    w = select_years(d, 2, 4)
    assert list(w["years"]) == [2, 3, 4] and w["maps"].shape[0] == 3 and w["targets"]["tas"].shape[0] == 3
    with pytest.raises(ValueError):
        select_years(d, 2080, 2100)


def test_default_eval_window_and_layers_disabled():
    cfg = OmegaConf.load(CONF / "config.yaml")
    assert list(cfg.eval.window) == [2080, 2100]
    for g, f in [("temporal", "timestral"), ("residual", "diffusion"), ("whatif", "placeholder")]:
        c = OmegaConf.load(CONF / g / f"{f}.yaml")
        assert c.enabled is False
    assert OmegaConf.load(CONF / "temporal" / "timestral.yaml").device == "cpu"


def test_stubs_raise():
    from cke import residual_diffusion, temporal, whatif
    with pytest.raises(NotImplementedError):
        temporal.forecast_series(np.zeros(3), None, 1, None)
    with pytest.raises(NotImplementedError):
        residual_diffusion.fit_residual()
    with pytest.raises(NotImplementedError):
        whatif.apply_action(None, None)
