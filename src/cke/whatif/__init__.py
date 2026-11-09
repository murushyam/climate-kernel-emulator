"""Layer 5: Action-conditioned counterfactual 'what-if' perturbation interface.

Reference:
- Wang et al. (2026), "Earth System World Model for What-If Simulations: A Case Study
  for Terrestrial Ecosystems", arXiv:2609.08855, doi:10.48550/arXiv.2609.08855.
"""
from __future__ import annotations

import copy
import numpy as np

from cke.data import GLOBAL_VARS, MAP_VARS


def _region_mask(lat: np.ndarray | None, lon: np.ndarray | None, action: dict, shape_hw: tuple[int, int]) -> np.ndarray:
    H, W = shape_hw
    if lat is None or lon is None:
        return np.ones((H, W), dtype=bool)
    lat_arr = np.asarray(lat, dtype=float)[:, None]
    lon_arr = (np.asarray(lon, dtype=float)[None, :]) % 360.0
    lat_range = action.get("lat_range", [-90.0, 90.0])
    lon_range = action.get("lon_range", [0.0, 360.0])
    m_lat = (lat_arr >= float(lat_range[0])) & (lat_arr <= float(lat_range[1]))
    lo, hi = float(lon_range[0]) % 360.0, float(lon_range[1]) % 360.0
    if lo <= hi:
        m_lon = (lon_arr >= lo) & (lon_arr <= hi)
    else:
        m_lon = (lon_arr >= lo) | (lon_arr <= hi)
    return np.broadcast_to(m_lat & m_lon, (H, W))


def apply_action(inputs, action, lat: np.ndarray | None = None, lon: np.ndarray | None = None):
    """Apply an action-conditioned counterfactual perturbation to ClimateBench forcing inputs.

    Raises NotImplementedError when `inputs is None` or `action is None`
    (preserving the placeholder guard tested in `tests/test_window_stubs.py`).

    Supported `inputs`:
      - Stacked split dict with keys `"global"` `(T, 2)` and `"maps"` `(T, 2, H, W)`
      - 4D array/tensor `(T, C, H, W)` where channels are `[CO2, CH4, SO2, BC]`

    Supported `action` dict (or list of dicts):
      - `{"type": "forcing_scale", "var": "SO2", "factor": 0.5, "lat_range": [5, 40], "lon_range": [65, 145]}`
      - `{"type": "forcing_offset", "var": "CO2", "delta": -100.0}`
    """
    if inputs is None or action is None:
        raise NotImplementedError("What-if layer requires valid inputs and action specification")

    actions = [action] if isinstance(action, dict) else list(action)
    if not actions:
        raise NotImplementedError("What-if layer received an empty action list")

    channel_order = GLOBAL_VARS + MAP_VARS

    if isinstance(inputs, dict) and "global" in inputs and "maps" in inputs:
        out = copy.deepcopy(inputs)
        lat_use = out.get("lat") if lat is None else lat
        lon_use = out.get("lon") if lon is None else lon
        H, W = out["maps"].shape[-2:]
        for act in actions:
            atype = act.get("type", "forcing_scale")
            var = act["var"]
            if var in GLOBAL_VARS:
                idx = GLOBAL_VARS.index(var)
                if atype == "forcing_scale":
                    out["global"][:, idx] *= float(act["factor"])
                elif atype == "forcing_offset":
                    out["global"][:, idx] += float(act["delta"])
                else:
                    raise ValueError(f"Unknown action type: {atype}")
            elif var in MAP_VARS:
                idx = MAP_VARS.index(var)
                mask = _region_mask(lat_use, lon_use, act, (H, W))
                if atype == "forcing_scale":
                    out["maps"][:, idx] = np.where(mask[None], out["maps"][:, idx] * float(act["factor"]), out["maps"][:, idx])
                elif atype == "forcing_offset":
                    out["maps"][:, idx] = np.where(mask[None], out["maps"][:, idx] + float(act["delta"]), out["maps"][:, idx])
                else:
                    raise ValueError(f"Unknown action type: {atype}")
            else:
                raise ValueError(f"Unknown forcing variable: {var}")
        return out

    arr = np.array(inputs, dtype=np.float32, copy=True)
    H, W = arr.shape[-2:]
    for act in actions:
        atype = act.get("type", "forcing_scale")
        var = act["var"]
        idx = channel_order.index(var)
        mask = _region_mask(lat, lon, act, (H, W))
        if atype == "forcing_scale":
            arr[:, idx] = np.where(mask[None], arr[:, idx] * float(act["factor"]), arr[:, idx])
        elif atype == "forcing_offset":
            arr[:, idx] = np.where(mask[None], arr[:, idx] + float(act["delta"]), arr[:, idx])
        else:
            raise ValueError(f"Unknown action type: {atype}")
    return arr
