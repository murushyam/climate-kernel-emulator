"""Layer 2: Explainable AI & mechanistic regime fidelity checks (Suri & Sonnewald, 2026)."""
from cke.interp.fidelity import (
    define_mechanistic_regimes,
    integrated_gradients_shap,
    lrp_local_mlp,
    verify_regime_fidelity,
)

__all__ = [
    "define_mechanistic_regimes",
    "integrated_gradients_shap",
    "lrp_local_mlp",
    "verify_regime_fidelity",
]

