import numpy as np
import pandas as pd
import xarray as xr

from cke.timeseries import area_weights, extract_all, region_weights, weighted_mean
from test_data import _write


def test_weighted_mean_constant():
    lat = np.linspace(-87.5, 87.5, 8)
    x = np.full((3, 8, 12), 2.5)
    assert np.allclose(weighted_mean(x, area_weights(lat, 12)), 2.5)


def test_regions_partition():
    lat, lon = np.linspace(-87.5, 87.5, 8), np.arange(12)
    lf = np.zeros((8, 12)); lf[:, :6] = 1.0
    r = region_weights(lat, lon, [(-90, 0), (0, 90)], lf)
    assert np.allclose(r["land"] + r["ocean"], r["global"])
    assert np.allclose(r["lat_-90_0"] + r["lat_0_90"], r["global"])
    x = np.zeros((1, 8, 12)); x[:, :, :6] = 1.0  # 1 on land
    assert np.isclose(weighted_mean(x, r["land"])[0], 1.0)
    assert np.isclose(weighted_mean(x, r["ocean"])[0], 0.0)


def test_extract_csv_and_netcdf(tmp_path):
    for s in ["a", "b"]:
        _write(tmp_path, s)
    p = extract_all(tmp_path, ["a", "b"], tmp_path / "ts", "csv", lat_bands=[(-30, 30)])
    df = pd.read_csv(p)
    assert len(df) == 10 and {"global_tas", "global_pr", "lat_-30_30_tas", "SO2_global", "CO2"} <= set(df.columns)
    p2 = extract_all(tmp_path, ["a", "b"], tmp_path / "ts", "netcdf")
    ds = xr.open_dataset(p2)
    assert ds["global_tas"].shape == (2, 5)
