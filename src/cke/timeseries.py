"""Step 1: extract area-weighted annual aggregate series (global, land/ocean, latitude bands).

    python -m cke.timeseries                       # uses configs/config.yaml
    python -m cke.timeseries timeseries.format=netcdf
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cke.data import GLOBAL_VARS, MAP_VARS, load_inputs, load_outputs, _years


def area_weights(lat: np.ndarray, nlon: int) -> np.ndarray:
    """(H, W) cos(lat) weights."""
    return np.repeat(np.cos(np.deg2rad(np.asarray(lat, float)))[:, None], nlon, 1)


def weighted_mean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x (T, H, W), w (H, W) >= 0 -> (T,). NaNs in x are ignored."""
    m = np.isfinite(x)
    ww = np.where(m, w[None], 0.0)
    return np.nansum(np.where(m, x, 0.0) * ww, axis=(-2, -1)) / ww.sum(axis=(-2, -1))


def region_weights(lat, lon, lat_bands=(), land_frac=None) -> dict:
    lat = np.asarray(lat, float)
    base = area_weights(lat, len(lon))
    regs = {"global": base}
    if land_frac is not None:
        lf = np.clip(np.asarray(land_frac, float), 0, 1)
        regs["land"], regs["ocean"] = base * lf, base * (1 - lf)
    for lo, hi in lat_bands:
        mask = ((lat >= lo) & (lat <= hi)).astype(float)[:, None]
        regs[f"lat_{lo}_{hi}"] = base * mask
    return regs


def extract_scenario(root, scenario, variables=("tas", "pr"), lat_bands=(), land_frac=None,
                     coarsen=1, pr_scale=86400.0) -> pd.DataFrame:
    """One row per year: year, scenario, forcings (global), <region>_<var> aggregates."""
    xi = load_inputs(root, scenario, coarsen)
    yo = load_outputs(root, scenario, variables, coarsen, pr_scale)
    n = min(xi.sizes["time"], yo.sizes["time"])
    lat, lon = yo["latitude"].values, yo["longitude"].values
    regs = region_weights(lat, lon, lat_bands, land_frac)
    df = pd.DataFrame({"year": _years(yo, n), "scenario": scenario})
    for v in GLOBAL_VARS:
        df[v] = xi[v].values[:n]
    w = area_weights(xi["SO2"]["latitude"].values, xi["SO2"].sizes["longitude"])
    for v in MAP_VARS:  # global-mean aerosol emissions as covariates
        df[f"{v}_global"] = weighted_mean(np.nan_to_num(xi[v].values[:n]), w)
    for v in variables:
        a = yo[v].values[:n]
        for r, rw in regs.items():
            df[f"{r}_{v}"] = weighted_mean(a, rw)
    return df


def extract_all(root, scenarios, out_dir, fmt="csv", **kw) -> Path:
    df = pd.concat([extract_scenario(root, s, **kw) for s in scenarios], ignore_index=True)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        path = out / "aggregate_series.csv"
        df.to_csv(path, index=False)
    elif fmt == "netcdf":
        path = out / "aggregate_series.nc"
        df.set_index(["scenario", "year"]).to_xarray().to_netcdf(path)
    else:
        raise ValueError(fmt)
    return path


def _main():
    import hydra
    from omegaconf import DictConfig

    from cke.train import CONFIG_PATH

    @hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
    def run(cfg: DictConfig):
        d, t = cfg.data, cfg.timeseries
        lf = None
        if t.land_mask:
            ds = xr.open_dataset(t.land_mask)
            lf = ds[list(ds.data_vars)[0]].values
        p = extract_all(d.root, list(d.train_scenarios) + list(d.test_scenarios), t.out_dir, t.format,
                        variables=tuple(d.variables), lat_bands=[tuple(b) for b in t.lat_bands],
                        land_frac=lf, coarsen=d.coarsen, pr_scale=d.pr_scale)
        print(f"wrote {p}")

    run()


if __name__ == "__main__":
    _main()
