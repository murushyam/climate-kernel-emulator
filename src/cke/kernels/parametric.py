"""Parametric nonlocal integration kernels on the sphere and backward time axis.

References:
- Ferretti et al. (2026), "Data-Driven Integration Kernels for Interpretable
  Nonlocal Operator Learning", Environmental Data Science, doi:10.1017/eds.2026.10056
  (arXiv:2603.10305), Appendix A.
- Bouabid, Sejdinovic, & Watson-Parris (2024), "FaIRGP: A Bayesian Energy Balance
  Model for Surface Temperatures Emulation", ICLR 2024 (arXiv:2306.04393).
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EARTH_RADIUS_KM = 6371.0
KernelFamily = Literal["gaussian", "mog", "tophat", "exponential"]


def spherical_coord_features(lat: np.ndarray, lon: np.ndarray, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return static (4, H, W) tensor: [sin(lat), cos(lat), sin(lon), cos(lon)]."""
    phi = np.deg2rad(np.asarray(lat, dtype=np.float32))[:, None]
    lam = np.deg2rad(np.asarray(lon, dtype=np.float32))[None, :]
    H, W = phi.shape[0], lam.shape[1]
    feats = np.stack([
        np.broadcast_to(np.sin(phi), (H, W)),
        np.broadcast_to(np.cos(phi), (H, W)),
        np.broadcast_to(np.sin(lam), (H, W)),
        np.broadcast_to(np.cos(lam), (H, W)),
    ], axis=0)
    return torch.as_tensor(feats, dtype=dtype, device=device)


def pairwise_spherical_offsets(lat: np.ndarray, lon: np.ndarray, device=None, dtype=torch.float32):
    """Precompute pairwise coordinate differences between target (H, W) and source (H, W) cells.

    Returns:
        dphi: (H*W, H*W) in radians, phi_source - phi_target in [-pi, pi]
        dlam: (H*W, H*W) in radians, wrapped periodic longitude difference in [-pi, pi]
        cos_phi_src: (1, H*W) area quadrature factor cos(phi_source)
        cos_phi_tgt: (H*W, 1) cos(phi_target)
    """
    phi = np.deg2rad(np.asarray(lat, dtype=np.float32))
    lam = np.deg2rad(np.asarray(lon, dtype=np.float32))
    phi_grid, lam_grid = np.meshgrid(phi, lam, indexing="ij")
    phi_flat = phi_grid.reshape(-1)
    lam_flat = lam_grid.reshape(-1)

    dphi = phi_flat[None, :] - phi_flat[:, None]
    dlam_raw = lam_flat[None, :] - lam_flat[:, None]
    dlam = (dlam_raw + np.pi) % (2.0 * np.pi) - np.pi
    cos_src = np.clip(np.cos(phi_flat), 1e-6, None)[None, :]
    cos_tgt = np.clip(np.cos(phi_flat), 1e-6, None)[:, None]

    return (
        torch.as_tensor(dphi, dtype=dtype, device=device),
        torch.as_tensor(dlam, dtype=dtype, device=device),
        torch.as_tensor(cos_src, dtype=dtype, device=device),
        torch.as_tensor(cos_tgt, dtype=dtype, device=device),
    )


class SpatialKernelLayer(nn.Module):
    """Parametric nonlocal integration kernel layer over spherical (lat, lon) grids.

    Following Ferretti et al. (2026), each input map channel c in {0..C-1} is integrated
    against K learnable continuous parametric kernels k_{c,k}(x, x'), normalized over the
    sphere with cos(lat) quadrature weights so each kernel sums to 1.

    Supported families:
      - 'gaussian': single anisotropic or isotropic Gaussian with learnable center shift (mu_lat, mu_lon)
      - 'mog': Mixture of M Gaussians with softmax component weights
      - 'tophat': differentiable double-sigmoid interval/annular window
      - 'exponential': monotonic exponential decay exp(-r / tau)
    """

    def __init__(
        self,
        in_channels: int,
        n_kernels: int,
        lat: np.ndarray,
        lon: np.ndarray,
        family: KernelFamily = "gaussian",
        anisotropic: bool = True,
        n_components: int = 2,
        sharpness: float = 0.05,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_kernels = int(n_kernels)
        self.family: KernelFamily = family
        self.anisotropic = bool(anisotropic)
        self.n_components = int(n_components)
        self.sharpness = float(sharpness)

        self.H = len(lat)
        self.W = len(lon)
        dphi, dlam, cos_src, cos_tgt = pairwise_spherical_offsets(lat, lon)
        self.register_buffer("dphi", dphi, persistent=False)
        self.register_buffer("dlam", dlam, persistent=False)
        self.register_buffer("cos_src", cos_src, persistent=False)
        self.register_buffer("cos_tgt", cos_tgt, persistent=False)

        C, K = self.in_channels, self.n_kernels
        init_scales = torch.linspace(math.radians(8.0), math.radians(40.0), K).unsqueeze(0).repeat(C, 1)

        if family == "gaussian":
            self.mu_lat = nn.Parameter(torch.zeros(C, K))
            self.mu_lon = nn.Parameter(torch.zeros(C, K))
            self.log_sigma_lat = nn.Parameter(torch.log(init_scales.clone()))
            if anisotropic:
                self.log_sigma_lon = nn.Parameter(torch.log(init_scales.clone()))
        elif family == "mog":
            M = self.n_components
            self.mu_lat = nn.Parameter(torch.zeros(C, K, M))
            self.mu_lon = nn.Parameter(torch.linspace(-0.2, 0.2, M).view(1, 1, M).repeat(C, K, 1))
            s_init = init_scales.unsqueeze(-1).repeat(1, 1, M)
            self.log_sigma_lat = nn.Parameter(torch.log(s_init.clone()))
            self.log_sigma_lon = nn.Parameter(torch.log(s_init.clone()))
            self.logit_weights = nn.Parameter(torch.zeros(C, K, M))
        elif family == "tophat":
            self.mu_lat = nn.Parameter(torch.zeros(C, K))
            self.mu_lon = nn.Parameter(torch.zeros(C, K))
            self.inner_r = nn.Parameter(torch.zeros(C, K))
            self.log_width = nn.Parameter(torch.log(init_scales.clone()))
        elif family == "exponential":
            self.mu_lat = nn.Parameter(torch.zeros(C, K))
            self.mu_lon = nn.Parameter(torch.zeros(C, K))
            self.log_tau = nn.Parameter(torch.log(init_scales.clone()))
        else:
            raise ValueError(f"Unknown kernel family: {family}")

    @property
    def out_channels(self) -> int:
        return self.in_channels * self.n_kernels

    def _shifted_distances(self, mu_lat: torch.Tensor, mu_lon: torch.Tensor):
        dp = self.dphi.unsqueeze(0).unsqueeze(0) - mu_lat.unsqueeze(-1).unsqueeze(-1)
        dl_raw = self.dlam.unsqueeze(0).unsqueeze(0) - mu_lon.unsqueeze(-1).unsqueeze(-1)
        dl = torch.remainder(dl_raw + math.pi, 2.0 * math.pi) - math.pi
        return dp, dl

    def compute_weights(self) -> torch.Tensor:
        """Return area-normalized integration weights of shape (C, K, N_tgt, N_src)."""
        if self.family == "gaussian":
            mu_lat = torch.tanh(self.mu_lat) * (math.pi / 2.0)
            mu_lon = torch.tanh(self.mu_lon) * math.pi
            dp, dl = self._shifted_distances(mu_lat, mu_lon)
            sig_lat = torch.clamp(torch.exp(self.log_sigma_lat), 1e-2, math.pi).unsqueeze(-1).unsqueeze(-1)
            if self.anisotropic:
                sig_lon = torch.clamp(torch.exp(self.log_sigma_lon), 1e-2, math.pi).unsqueeze(-1).unsqueeze(-1)
                dist2 = (dp / sig_lat) ** 2 + (dl / sig_lon) ** 2
            else:
                r2 = dp ** 2 + (self.cos_tgt.unsqueeze(0).unsqueeze(0) * dl) ** 2
                dist2 = r2 / (sig_lat ** 2)
            raw = torch.exp(-0.5 * dist2)

        elif self.family == "mog":
            mu_lat = torch.tanh(self.mu_lat) * (math.pi / 2.0)
            mu_lon = torch.tanh(self.mu_lon) * math.pi
            dp = self.dphi.view(1, 1, 1, *self.dphi.shape) - mu_lat.unsqueeze(-1).unsqueeze(-1)
            dl_raw = self.dlam.view(1, 1, 1, *self.dlam.shape) - mu_lon.unsqueeze(-1).unsqueeze(-1)
            dl = torch.remainder(dl_raw + math.pi, 2.0 * math.pi) - math.pi
            sig_lat = torch.clamp(torch.exp(self.log_sigma_lat), 1e-2, math.pi).unsqueeze(-1).unsqueeze(-1)
            sig_lon = torch.clamp(torch.exp(self.log_sigma_lon), 1e-2, math.pi).unsqueeze(-1).unsqueeze(-1)
            comp = torch.exp(-0.5 * ((dp / sig_lat) ** 2 + (dl / sig_lon) ** 2))
            w = F.softmax(self.logit_weights, dim=-1).unsqueeze(-1).unsqueeze(-1)
            raw = (w * comp).sum(dim=2)

        elif self.family == "tophat":
            mu_lat = torch.tanh(self.mu_lat) * (math.pi / 2.0)
            mu_lon = torch.tanh(self.mu_lon) * math.pi
            dp, dl = self._shifted_distances(mu_lat, mu_lon)
            r = torch.sqrt(dp ** 2 + (self.cos_tgt.unsqueeze(0).unsqueeze(0) * dl) ** 2 + 1e-8)
            a = F.softplus(self.inner_r).unsqueeze(-1).unsqueeze(-1)
            width = torch.clamp(torch.exp(self.log_width), 1e-2, math.pi).unsqueeze(-1).unsqueeze(-1)
            b = a + width
            eps = self.sharpness
            raw = torch.sigmoid((r - a) / eps) * torch.sigmoid((b - r) / eps)

        elif self.family == "exponential":
            mu_lat = torch.tanh(self.mu_lat) * (math.pi / 2.0)
            mu_lon = torch.tanh(self.mu_lon) * math.pi
            dp, dl = self._shifted_distances(mu_lat, mu_lon)
            r = torch.sqrt(dp ** 2 + (self.cos_tgt.unsqueeze(0).unsqueeze(0) * dl) ** 2 + 1e-8)
            tau = torch.clamp(torch.exp(self.log_tau), 1e-3, math.pi).unsqueeze(-1).unsqueeze(-1)
            raw = torch.exp(-r / tau)

        weighted = raw * self.cos_src.unsqueeze(0).unsqueeze(0)
        norm = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weighted / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate spatial maps x of shape (B, C, H, W) -> (B, C * K, H, W)."""
        B, C, H, W = x.shape
        assert C == self.in_channels and H == self.H and W == self.W
        W_mat = self.compute_weights()
        x_flat = x.reshape(B, C, H * W)
        z = torch.einsum("bcs,ckts->bckt", x_flat, W_mat)
        return z.reshape(B, C * self.n_kernels, H, W)

    @torch.no_grad()
    def kernel_footprints(self, lat_idx: int | None = None, lon_idx: int | None = None) -> np.ndarray:
        """Return normalized (C, K, H, W) spatial kernel footprints for a target grid cell."""
        if lat_idx is None:
            lat_idx = self.H // 2
        if lon_idx is None:
            lon_idx = self.W // 2
        tgt_flat = lat_idx * self.W + lon_idx
        W_mat = self.compute_weights()[:, :, tgt_flat, :]
        return W_mat.reshape(self.in_channels, self.n_kernels, self.H, self.W).detach().cpu().numpy()

    @torch.no_grad()
    def export_parameters(self) -> dict:
        """Export human-readable physical kernel parameters (degrees and kilometers)."""
        rad2deg = 180.0 / math.pi
        rad2km = EARTH_RADIUS_KM
        info: dict = {"family": self.family, "in_channels": self.in_channels, "n_kernels": self.n_kernels}
        if self.family == "gaussian":
            mu_lat = (torch.tanh(self.mu_lat) * 90.0).cpu().tolist()
            mu_lon = (torch.tanh(self.mu_lon) * 180.0).cpu().tolist()
            sig_lat = torch.clamp(torch.exp(self.log_sigma_lat), 1e-2, math.pi).cpu()
            info["mu_lat_deg"] = mu_lat
            info["mu_lon_deg"] = mu_lon
            info["sigma_lat_deg"] = (sig_lat * rad2deg).tolist()
            info["sigma_lat_km"] = (sig_lat * rad2km).tolist()
            if self.anisotropic:
                sig_lon = torch.clamp(torch.exp(self.log_sigma_lon), 1e-2, math.pi).cpu()
                info["sigma_lon_deg"] = (sig_lon * rad2deg).tolist()
                info["sigma_lon_km"] = (sig_lon * rad2km).tolist()
        elif self.family == "exponential":
            tau = torch.clamp(torch.exp(self.log_tau), 1e-3, math.pi).cpu()
            info["tau_deg"] = (tau * rad2deg).tolist()
            info["tau_km"] = (tau * rad2km).tolist()
        elif self.family == "tophat":
            width = torch.clamp(torch.exp(self.log_width), 1e-2, math.pi).cpu()
            info["width_deg"] = (width * rad2deg).tolist()
            info["width_km"] = (width * rad2km).tolist()
        elif self.family == "mog":
            w = F.softmax(self.logit_weights, dim=-1).cpu().tolist()
            info["mixture_weights"] = w
        return info


def build_temporal_history(
    global_f: np.ndarray,
    scenarios: np.ndarray | list[str] | None = None,
    window: int = 10,
) -> np.ndarray:
    """Construct causal lookback windows (T, F, L) for global forcings without cross-scenario leakage."""
    T, F_dim = global_f.shape
    L = max(1, int(window))
    out = np.zeros((T, F_dim, L), dtype=np.float32)
    scen = np.zeros(T, dtype=int) if scenarios is None else np.asarray(scenarios)
    start_idx = 0
    for t in range(T):
        if t > 0 and scen[t] != scen[t - 1]:
            start_idx = t
        for lag in range(L):
            src_t = max(start_idx, t - lag)
            out[t, :, lag] = global_f[src_t]
    return out


class TemporalExponentialKernel(nn.Module):
    """Causal 1D exponential memory kernel over forcing history (Ferretti et al., 2026; Bouabid et al., 2024)."""

    def __init__(self, in_features: int, n_kernels: int = 2, window: int = 10):
        super().__init__()
        self.in_features = int(in_features)
        self.n_kernels = int(n_kernels)
        self.window = int(window)
        lags = torch.arange(self.window, dtype=torch.float32).view(1, 1, self.window)
        self.register_buffer("lags", lags, persistent=False)
        init_tau = torch.logspace(math.log10(1.5), math.log10(15.0), self.n_kernels)
        self.log_tau = nn.Parameter(torch.log(init_tau).unsqueeze(0).repeat(self.in_features, 1))

    @property
    def out_features(self) -> int:
        return self.in_features * self.n_kernels

    def compute_weights(self) -> torch.Tensor:
        """Return normalized causal weights of shape (F, M, L)."""
        tau = torch.clamp(torch.exp(self.log_tau), 0.25, 100.0).unsqueeze(-1)
        raw = torch.exp(-self.lags / tau)
        return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """x_hist: (B, F, L) -> (B, F * M)."""
        B, F_dim, L = x_hist.shape
        assert F_dim == self.in_features and L == self.window
        w = self.compute_weights()
        out = torch.einsum("bfl,fml->bfm", x_hist, w)
        return out.reshape(B, F_dim * self.n_kernels)

    @torch.no_grad()
    def export_parameters(self) -> dict:
        tau = torch.clamp(torch.exp(self.log_tau), 0.25, 100.0).cpu().tolist()
        return {"window_years": self.window, "n_kernels": self.n_kernels, "tau_years": tau}
