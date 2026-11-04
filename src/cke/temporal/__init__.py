"""Layer 3: temporal / probabilistic module (Timestral, CPU). STUB - Step 3.

Verified 2026-11-04: `pip install git+https://github.com/mistralai/timestral.git`;
`from timestral import TimestralForecaster`; checkpoints mistralai/Timestral-5M, -28M (also 123M, 209M).
forecast(context (B,V,T), prediction_length) -> .quantiles (B,V,H,21), .median.
TODO: confirm the exact keyword for known-future covariates in the package README before wiring it in.
"""
from __future__ import annotations

import numpy as np


def load_forecaster(checkpoint: str = "mistralai/Timestral-5M", device: str = "cpu"):
    try:
        from timestral import TimestralForecaster
    except ImportError as e:  # optional dependency
        raise ImportError("pip install git+https://github.com/mistralai/timestral.git") from e
    return TimestralForecaster.from_pretrained(checkpoint, device=device)


def forecast_series(context: np.ndarray, covariates: np.ndarray | None, prediction_length: int, cfg):
    """TODO(Step 3): run Timestral on aggregate series with forcings as known-future covariates."""
    raise NotImplementedError("Temporal module is a Step 3 stub")
