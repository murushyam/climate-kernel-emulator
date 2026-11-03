"""ClimateBench NRMSE metrics (Watson-Parris et al., 2022).

Arrays are (time, lat, lon). Normaliser is the mean absolute global-mean truth.
NRMSE_s : RMSE of the time-mean field (latitude-weighted).
NRMSE_g : RMSE of the global-mean time series.
NRMSE_t = NRMSE_s + 5 * NRMSE_g
"""
import numpy as np


def lat_weights(lat: np.ndarray) -> np.ndarray:
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))
    return w / w.mean()


def global_mean(x: np.ndarray, lat: np.ndarray) -> np.ndarray:
    w = lat_weights(lat)[:, None]
    return (x * w).mean(axis=(-2, -1))


def _normaliser(truth, lat):
    return np.abs(global_mean(truth, lat).mean())


def nrmse_spatial(truth, pred, lat):
    w = lat_weights(lat)[:, None]
    diff2 = (truth.mean(0) - pred.mean(0)) ** 2
    return float(np.sqrt((diff2 * w).mean()) / _normaliser(truth, lat))


def nrmse_global(truth, pred, lat):
    d = global_mean(truth, lat) - global_mean(pred, lat)
    return float(np.sqrt((d ** 2).mean()) / _normaliser(truth, lat))


def nrmse_total(truth, pred, lat, alpha: float = 5.0):
    return nrmse_spatial(truth, pred, lat) + alpha * nrmse_global(truth, pred, lat)


def all_metrics(truth, pred, lat):
    s, g = nrmse_spatial(truth, pred, lat), nrmse_global(truth, pred, lat)
    return {"nrmse_spatial": s, "nrmse_global": g, "nrmse_total": s + 5 * g}
