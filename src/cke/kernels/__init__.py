"""Layer 1: Interpretable nonlocal integration kernels (Ferretti et al., 2026)."""
from cke.kernels.model import KernelEmulator, load_checkpoint, save_checkpoint
from cke.kernels.parametric import (
    SpatialKernelLayer,
    TemporalExponentialKernel,
    build_temporal_history,
    pairwise_spherical_offsets,
    spherical_coord_features,
)

__all__ = [
    "KernelEmulator",
    "load_checkpoint",
    "save_checkpoint",
    "SpatialKernelLayer",
    "TemporalExponentialKernel",
    "build_temporal_history",
    "pairwise_spherical_offsets",
    "spherical_coord_features",
]

