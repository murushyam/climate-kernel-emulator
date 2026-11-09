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


def pinball_loss(truth: np.ndarray, pred_quantiles: np.ndarray, quantiles: np.ndarray) -> float:
    """Quantile (pinball) loss: truth (...,), pred_quantiles (..., Q), quantiles (Q,) in (0, 1)."""
    q = np.asarray(quantiles, dtype=float)
    err = np.asarray(truth, dtype=float)[..., None] - np.asarray(pred_quantiles, dtype=float)
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.mean(loss))


def crps_from_quantiles(truth: np.ndarray, pred_quantiles: np.ndarray, quantiles: np.ndarray) -> float:
    """Approximate Continuous Ranked Probability Score via 2 * integral_0^1 pinball(q) dq."""
    q = np.asarray(quantiles, dtype=float)
    err = np.asarray(truth, dtype=float)[..., None] - np.asarray(pred_quantiles, dtype=float)
    axes = tuple(range(err.ndim - 1))
    pinball_per_q = np.mean(np.maximum(q * err, (q - 1.0) * err), axis=axes) if axes else np.maximum(q * err, (q - 1.0) * err)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(2.0 * trapz_fn(pinball_per_q, q))

