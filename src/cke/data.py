"""ClimateBench loading: forcings -> annual tas/pr (ensemble mean), coarsening, normalisation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

GLOBAL_VARS = ["CO2", "CH4"]
MAP_VARS = ["SO2", "BC"]


def _coarsen(da: xr.DataArray, factor: int) -> xr.DataArray:
    if factor <= 1:
        return da
    dims = {d: factor for d in ("latitude", "longitude", "lat", "lon") if d in da.dims}
    return da.coarsen(dims, boundary="trim").mean()


def _std_dims(da: xr.DataArray) -> xr.DataArray:
    ren = {k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in da.dims}
    return da.rename(ren)


def load_inputs(root, scenario: str, coarsen: int = 1) -> xr.Dataset:
    ds = xr.open_dataset(Path(root) / f"inputs_{scenario}.nc")
    out = {v: ds[v] for v in GLOBAL_VARS}
    for v in MAP_VARS:
        out[v] = _coarsen(_std_dims(ds[v]), coarsen)
    return xr.Dataset(out)


def load_outputs(root, scenario: str, variables=("tas", "pr"), coarsen: int = 1,
                 pr_scale: float = 86400.0) -> xr.Dataset:
    ds = xr.open_dataset(Path(root) / f"outputs_{scenario}.nc")
    out = {}
    for v in variables:
        da = _std_dims(ds[v])
        if "member" in da.dims:
            da = da.mean("member")
        if v == "pr":
            da = da * pr_scale
        out[v] = _coarsen(da, coarsen)
    return xr.Dataset(out)


def eof_reduce(maps: np.ndarray, n: int, basis: np.ndarray | None = None):
    """maps (time, lat, lon) -> (time, n) PCs; returns (pcs, basis). Fit basis on training data."""
    flat = maps.reshape(maps.shape[0], -1)
    if basis is None:
        _, _, vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
        basis = vt[:n]
    return flat @ basis.T, basis


@dataclass
class Normaliser:
    mean: dict
    std: dict

    @classmethod
    def fit(cls, arrays: dict):
        return cls({k: float(np.mean(a)) for k, a in arrays.items()},
                   {k: float(np.std(a) + 1e-8) for k, a in arrays.items()})

    def apply(self, k, a):
        return (a - self.mean[k]) / self.std[k]

    def invert(self, k, a):
        return a * self.std[k] + self.mean[k]


def stack_scenarios(root, scenarios, variables=("tas", "pr"), coarsen=1, pr_scale=86400.0):
    """Return dict of numpy arrays concatenated over scenarios along time, plus lat/lon."""
    X_g, X_m, Y, yrs, scen = [], [], {v: [] for v in variables}, [], []
    lat = lon = None
    for s in scenarios:
        xi = load_inputs(root, s, coarsen)
        yo = load_outputs(root, s, variables, coarsen, pr_scale)
        n = min(xi.sizes["time"], yo.sizes["time"])
        X_g.append(np.stack([xi[v].values[:n] for v in GLOBAL_VARS], -1))
        X_m.append(np.stack([xi[v].values[:n] for v in MAP_VARS], 1))
        for v in variables:
            Y[v].append(yo[v].values[:n])
        lat, lon = yo["latitude"].values, yo["longitude"].values
        yrs.append(_years(yo, n))
        scen += [s] * n
    return {
        "global": np.concatenate(X_g).astype(np.float32),      # (T, 2)
        "maps": np.nan_to_num(np.concatenate(X_m)).astype(np.float32),  # (T, 2, H, W)
        "targets": {v: np.concatenate(Y[v]).astype(np.float32) for v in variables},  # (T, H, W)
        "lat": lat, "lon": lon,
        "years": np.concatenate(yrs).astype(int),            # (T,)
        "scenario": np.array(scen),                          # (T,)
    }


def _years(ds: xr.Dataset, n: int) -> np.ndarray:
    t = ds["time"].values[:n]
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[Y]").astype(int) + 1970
    if t.dtype == object:  # cftime
        return np.array([x.year for x in t])
    return np.asarray(t).astype(int)


def select_years(split: dict, start: int | None, end: int | None) -> dict:
    """Restrict a stacked split to start <= year <= end (inclusive). None = open bound."""
    y = split["years"]
    m = np.ones_like(y, dtype=bool)
    if start is not None:
        m &= y >= start
    if end is not None:
        m &= y <= end
    if not m.any():
        raise ValueError(f"no years in window [{start}, {end}]; available {y.min()}-{y.max()}")
    out = dict(split)
    out["_full_global"] = split["global"]
    out["_full_scenario"] = split["scenario"]
    out["_window_mask"] = m
    out["global"], out["maps"] = split["global"][m], split["maps"][m]
    out["targets"] = {k: v[m] for k, v in split["targets"].items()}
    out["years"], out["scenario"] = y[m], split["scenario"][m]
    return out


class ClimateBenchDataset(Dataset):
    """One sample = one year: inputs (C, H, W) [global forcings broadcast + local maps], target (V, H, W)."""

    def __init__(self, global_f: np.ndarray, maps: np.ndarray, targets: np.ndarray):
        T, _, H, W = maps.shape
        g = np.broadcast_to(global_f[:, :, None, None], (T, global_f.shape[1], H, W))
        self.x = torch.from_numpy(np.concatenate([g, maps], 1).astype(np.float32))
        self.y = torch.from_numpy(targets.astype(np.float32))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]
