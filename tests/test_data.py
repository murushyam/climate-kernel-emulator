import numpy as np
import xarray as xr

from cke.data import load_inputs, load_outputs, stack_scenarios


def _write(tmp, scen, T=5, H=8, W=12):
    t = np.arange(T)
    lat, lon = np.linspace(-87.5, 87.5, H), np.linspace(0, 357.5, W)
    xr.Dataset({
        "CO2": ("time", np.linspace(0, 100, T)), "CH4": ("time", np.ones(T)),
        "SO2": (("time", "lat", "lon"), np.random.rand(T, H, W)),
        "BC": (("time", "lat", "lon"), np.random.rand(T, H, W)),
    }, coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(tmp / f"inputs_{scen}.nc")
    xr.Dataset({
        "tas": (("member", "time", "lat", "lon"), np.random.rand(3, T, H, W)),
        "pr": (("member", "time", "lat", "lon"), np.random.rand(3, T, H, W) * 1e-5),
    }, coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(tmp / f"outputs_{scen}.nc")


def test_load_and_coarsen(tmp_path):
    _write(tmp_path, "ssp245")
    x = load_inputs(tmp_path, "ssp245", coarsen=2)
    y = load_outputs(tmp_path, "ssp245", coarsen=2)
    assert x["SO2"].shape == (5, 4, 6)
    assert y["tas"].shape == (5, 4, 6)  # member averaged


def test_stack(tmp_path):
    for s in ["a", "b"]:
        _write(tmp_path, s)
    d = stack_scenarios(tmp_path, ["a", "b"], coarsen=1)
    assert d["global"].shape == (10, 2) and d["maps"].shape == (10, 2, 8, 12)
    assert d["targets"]["pr"].max() < 1.0  # scaled to mm/day
