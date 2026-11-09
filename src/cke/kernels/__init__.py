"""Layer 1: Interpretable nonlocal integration kernels (Ferretti et al., 2026)."""
from cke.kernels.model import KernelEmulator
from cke.kernels.parametric import (
    SpatialKernelLayer,
    TemporalExponentialKernel,
    build_temporal_history,
    pairwise_spherical_offsets,
    spherical_coord_features,
)

__all__ = [
    "KernelEmulator",
    "SpatialKernelLayer",
    "TemporalExponentialKernel",
    "build_temporal_history",
    "pairwise_spherical_offsets",
    "spherical_coord_features",
]

