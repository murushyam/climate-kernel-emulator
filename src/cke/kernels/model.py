"""Complete nonlocal-kernel + local-MLP climate emulator (Layer 1, Step 2)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cke.baselines.local_mlp import LocalMLP
from cke.kernels.parametric import (
    KernelFamily,
    SpatialKernelLayer,
    TemporalExponentialKernel,
    spherical_coord_features,
)


class KernelEmulator(nn.Module):
    """Nonlocal-kernel climate emulator following Ferretti et al. (2026)."""

    def __init__(
        self,
        n_global: int,
        n_maps: int,
        out_channels: int,
        lat: np.ndarray,
        lon: np.ndarray,
        n_spatial_kernels: int = 4,
        family: KernelFamily = "gaussian",
        anisotropic: bool = True,
        n_components: int = 2,
        n_temporal_kernels: int = 0,
        temporal_window: int = 10,
        include_local_maps: bool = True,
        coord_features: bool = True,
        hidden: tuple[int, ...] = (32, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_global = int(n_global)
        self.n_maps = int(n_maps)
        self.out_channels = int(out_channels)
        self.include_local_maps = bool(include_local_maps)
        self.coord_features = bool(coord_features)
        self.H = len(lat)
        self.W = len(lon)

        self.spatial_kernels = SpatialKernelLayer(
            in_channels=n_maps,
            n_kernels=n_spatial_kernels,
            lat=lat,
            lon=lon,
            family=family,
            anisotropic=anisotropic,
            n_components=n_components,
        )

        self.temporal_kernels: TemporalExponentialKernel | None = None
        temporal_channels = 0
        if n_temporal_kernels > 0 and temporal_window > 1:
            self.temporal_kernels = TemporalExponentialKernel(
                in_features=n_global,
                n_kernels=n_temporal_kernels,
                window=temporal_window,
            )
            temporal_channels = self.temporal_kernels.out_features

        if coord_features:
            coords = spherical_coord_features(lat, lon)
            self.register_buffer("coords", coords, persistent=False)
            coord_channels = 4
        else:
            self.register_buffer("coords", None, persistent=False)
            coord_channels = 0

        local_map_channels = n_maps if include_local_maps else 0
        total_in_channels = (
            n_global
            + temporal_channels
            + local_map_channels
            + self.spatial_kernels.out_channels
            + coord_channels
        )
        self.total_in_channels = total_in_channels
        self.local_predictor = LocalMLP(
            in_channels=total_in_channels,
            out_channels=out_channels,
            hidden=tuple(hidden),
            dropout=dropout,
        )
        self.register_buffer("anchor_a", torch.zeros(1, out_channels, self.H, self.W), persistent=True)
        self.register_buffer("anchor_b", torch.zeros(1, out_channels, n_global, self.H, self.W), persistent=True)
        self.has_anchor = False

    @torch.no_grad()
    def fit_spatial_anchor(self, g_tr: np.ndarray | torch.Tensor, y_tr: np.ndarray | torch.Tensor):
        """Fit closed-form per-gridcell linear baseline y ~ a(x) + b(x)^T g on training split.

        Allows the nonlocal kernel + local MLP pathway to focus on nonlocal aerosol forcing,
        ocean thermal lag, and nonlinear feedbacks around the stationary geographic pattern.
        """
        g_np = g_tr.detach().cpu().numpy() if isinstance(g_tr, torch.Tensor) else np.asarray(g_tr, dtype=np.float32)
        y_np = y_tr.detach().cpu().numpy() if isinstance(y_tr, torch.Tensor) else np.asarray(y_tr, dtype=np.float32)
        T, F_dim = g_np.shape
        _, V, H, W = y_np.shape
        A = np.concatenate([np.ones((T, 1), dtype=np.float32), g_np], axis=1)  # (T, 1 + F)
        # Ridge-stabilized least squares
        ata = A.T @ A + 1e-2 * np.eye(1 + F_dim, dtype=np.float32)
        coef = np.linalg.solve(ata, A.T @ y_np.reshape(T, -1)).reshape(1 + F_dim, V, H, W)
        dev = self.anchor_a.device
        self.anchor_a.copy_(torch.as_tensor(coef[0:1], dtype=torch.float32, device=dev))
        self.anchor_b.copy_(torch.as_tensor(np.transpose(coef[1:], (1, 0, 2, 3))[None], dtype=torch.float32, device=dev))
        self.has_anchor = True
        # Initialize final 1x1 conv of local_predictor near zero so training starts from the physical anchor
        last_conv = self.local_predictor.net[-1]
        if isinstance(last_conv, nn.Conv2d):
            nn.init.normal_(last_conv.weight, mean=0.0, std=1e-2)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def assemble_features(
        self,
        x: torch.Tensor,
        g_hist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble local + nonlocal feature stack of shape (B, C_total, H, W)."""
        B, C, H, W = x.shape
        assert C == self.n_global + self.n_maps
        g_local = x[:, : self.n_global]
        maps = x[:, self.n_global :]

        feats = [g_local]
        if self.temporal_kernels is not None:
            if g_hist is None:
                g_vec = g_local[:, :, 0, 0].unsqueeze(-1).expand(B, self.n_global, self.temporal_kernels.window)
            else:
                g_vec = g_hist
            g_mem = self.temporal_kernels(g_vec)
            feats.append(g_mem[:, :, None, None].expand(B, g_mem.shape[1], H, W))

        if self.include_local_maps:
            feats.append(maps)

        z_spatial = self.spatial_kernels(maps)
        feats.append(z_spatial)

        if self.coord_features and self.coords is not None:
            feats.append(self.coords.unsqueeze(0).expand(B, -1, H, W))

        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor, g_hist: torch.Tensor | None = None) -> torch.Tensor:
        """Predict target fields (B, out_channels, H, W)."""
        feat_stack = self.assemble_features(x, g_hist=g_hist)
        out = self.local_predictor(feat_stack)
        if self.has_anchor:
            g_local = x[:, None, : self.n_global]  # (B, 1, F, H, W)
            base = self.anchor_a + (self.anchor_b * g_local).sum(dim=2)
            out = base + out
        return out

    @torch.no_grad()
    def export_kernel_summary(self) -> dict:
        out = {"spatial": self.spatial_kernels.export_parameters()}
        if self.temporal_kernels is not None:
            out["temporal"] = self.temporal_kernels.export_parameters()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        anchor = int(self.anchor_a.numel() + self.anchor_b.numel()) if self.has_anchor else 0
        out["trainable_parameters"] = trainable
        out["anchor_coefficients"] = anchor
        out["total_parameters"] = trainable + anchor
        return out


def save_checkpoint(model: "KernelEmulator", path) -> None:
    """Save everything needed to restore a fitted KernelEmulator.

    Includes the network weights, the fitted closed-form anchor buffers (in the state dict),
    whether the anchor is active, and the input normalisation statistics.
    """
    stats = getattr(model, "input_stats", None)
    if stats is not None:
        stats = {k: (torch.as_tensor(np.asarray(v)) if not isinstance(v, torch.Tensor) else v.detach().cpu())
                 for k, v in stats.items()}
    torch.save({"state_dict": model.state_dict(), "has_anchor": bool(model.has_anchor),
                "input_stats": stats}, path)


def load_checkpoint(model: "KernelEmulator", path, map_location="cpu") -> "KernelEmulator":
    """Restore a checkpoint written by `save_checkpoint` into an identically configured model."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.has_anchor = bool(ckpt["has_anchor"])
    if ckpt.get("input_stats") is not None:
        model.input_stats = {k: v.numpy() for k, v in ckpt["input_stats"].items()}
    return model
