"""Layer 3: temporal / probabilistic module (Timestral, CPU).

Reference:
- Petralia et al. / Mistral AI (2026), "Timestral: Time Series Foundation Models with
  Decoupled Horizon Forecasting", arXiv:2610.37585.
- Package: `pip install git+https://github.com/mistralai/timestral.git`
- Checkpoints: `mistralai/Timestral-5M` (21 MB), `mistralai/Timestral-28M` (111 MB).
- Verified covariate API: `forecaster.forecast(context, prediction_length=H, known_covariates=known)`
  where `context` has shape `(B, V, T)` and `known_covariates` has shape `(B, C_known, T + H)`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIMESTRAL_DEFAULT_QUANTILES = np.array(
    [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
     0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99],
    dtype=np.float32,
)


def load_forecaster(checkpoint: str = "mistralai/Timestral-5M", device: str = "cpu"):
    try:
        from timestral import TimestralForecaster
    except ImportError as e:  # optional dependency
        raise ImportError("pip install git+https://github.com/mistralai/timestral.git") from e
    return TimestralForecaster.from_pretrained(checkpoint, device=device)


def prepare_timestral_tensors(
    df: pd.DataFrame,
    context_scenarios: list[str] | tuple[str, ...] = ("historical",),
    test_scenario: str = "ssp245",
    target_series: list[str] | tuple[str, ...] = ("global_tas", "global_pr"),
    covariates: list[str] | tuple[str, ...] = ("CO2", "CH4", "SO2_global", "BC_global"),
) -> dict[str, np.ndarray]:
    """Build (1, V, T_ctx) context and (1, C_known, T_ctx + T_pred) known_covariates from aggregate CSV/DataFrame."""
    ctx_df = df[df["scenario"].isin(context_scenarios)].sort_values("year")
    te_df = df[df["scenario"] == test_scenario].sort_values("year")
    if ctx_df.empty or te_df.empty:
        raise ValueError(f"Missing scenarios in DataFrame: context={context_scenarios}, test={test_scenario}")

    ctx_targets = np.stack([ctx_df[col].to_numpy(dtype=np.float32) for col in target_series], axis=0)[None]  # (1, V, T_ctx)
    te_targets = np.stack([te_df[col].to_numpy(dtype=np.float32) for col in target_series], axis=0)[None]   # (1, V, T_pred)

    full_df = pd.concat([ctx_df, te_df], ignore_index=True)
    known_cov = np.stack([full_df[col].to_numpy(dtype=np.float32) for col in covariates], axis=0)[None]     # (1, C, T_ctx + T_pred)

    return {
        "context": ctx_targets,
        "known_covariates": known_cov,
        "target_truth": te_targets,
        "test_years": te_df["year"].to_numpy(dtype=int),
    }


def forecast_series(
    context: np.ndarray,
    covariates: np.ndarray | None,
    prediction_length: int,
    cfg,
    forecaster=None,
) -> dict[str, np.ndarray]:
    """Run Timestral zero-shot quantile forecasting with forcings as known-future covariates.

    Raises NotImplementedError when `cfg is None` or `cfg.get('enabled') is False`
    (preserving the Step 1 stub contract tested in `tests/test_window_stubs.py`).
    """
    if cfg is None or not (cfg.get("enabled", False) if hasattr(cfg, "get") else getattr(cfg, "enabled", False)):
        raise NotImplementedError("Temporal module is disabled or unconfigured (set temporal.enabled=true)")

    ctx = np.asarray(context, dtype=np.float32)
    if ctx.ndim == 1:
        ctx = ctx[None, None, :]
    elif ctx.ndim == 2:
        ctx = ctx[None, :, :]

    B, V, T_ctx = ctx.shape
    H = int(prediction_length)

    known_cov = None
    if covariates is not None:
        cov = np.asarray(covariates, dtype=np.float32)
        if cov.ndim == 2:
            cov = cov[None, :, :]
        if cov.shape[-1] != T_ctx + H:
            raise ValueError(f"known_covariates time dimension must be T_ctx + prediction_length ({T_ctx + H}), got {cov.shape[-1]}")
        known_cov = cov

    if forecaster is None:
        ckpt = cfg.get("checkpoint", "mistralai/Timestral-5M") if hasattr(cfg, "get") else getattr(cfg, "checkpoint", "mistralai/Timestral-5M")
        dev = cfg.get("device", "cpu") if hasattr(cfg, "get") else getattr(cfg, "device", "cpu")
        forecaster = load_forecaster(checkpoint=ckpt, device=dev)

    kwargs = {"prediction_length": H}
    if known_cov is not None:
        kwargs["known_covariates"] = known_cov

    res = forecaster.forecast(ctx, **kwargs)
    quantiles = np.asarray(getattr(res, "quantiles", res), dtype=np.float32)  # (B, V, H, Q)
    median = np.asarray(getattr(res, "median", quantiles[..., quantiles.shape[-1] // 2]), dtype=np.float32)

    return {
        "quantiles": quantiles,
        "median": median,
        "quantile_levels": TIMESTRAL_DEFAULT_QUANTILES,
    }
