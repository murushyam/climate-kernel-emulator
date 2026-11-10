"""Layer 4: residual operator-diffusion correction on top of the kernel prior.

The interpretable kernel model (Layer 1) supplies the coarse prior y_kernel. A small
residual-conditioned model learns only the residual Delta = y_true - y_kernel, and the
final output is y_kernel + Delta_hat (residual operator-diffusion idea, Park et al., 2026,
npj Artificial Intelligence). Two stages, in order:

1. ``ResidualHead``: deterministic lightweight residual CNN (default, ``mode: head``).
2. ``ResidualDenoiser``: tiny residual diffusion (``mode: diffusion``), used only if the
   residual left after stage 1 is still large and spatially structured.

References:
- Park et al. (2026), residual operator-diffusion, npj Artificial Intelligence.
- Kingston et al. (2026), "IPSL-AID: Generative Diffusion Models for Climate Downscaling
  from Global to Regional Scales", arXiv:2604.03275, doi:10.48550/arXiv.2604.03275.
- Mardani et al. (2024), "Residual Corrective Diffusion Modeling for Km-Scale Atmospheric
  Downscaling", Communications Earth & Environment, 5, 124, doi:10.1038/s43247-024-01291-0.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ResidualDenoiser(nn.Module):
    """Compact circular-padded 2D denoiser network for climate emulator residuals (< 5e4 params)."""

    def __init__(self, target_channels: int, cond_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)
        self.hidden_channels = int(hidden_channels)

        in_ch = target_channels + cond_channels + 2  # +2 for [sin(pi*t), cos(pi*t)]
        self.conv_in = nn.Conv2d(in_ch, hidden_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.conv_mid = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular")
        self.conv_out = nn.Conv2d(hidden_channels, target_channels, kernel_size=1)

    def forward(self, r_noisy: torch.Tensor, cond: torch.Tensor, t_frac: torch.Tensor) -> torch.Tensor:
        """Predict noise epsilon given noisy residual r_noisy (B, V, H, W), conditioning (B, C, H, W), and t_frac (B,) in [0, 1]."""
        B, _, H, W = r_noisy.shape
        t_ang = t_frac.view(B, 1, 1, 1) * math.pi
        t_emb = torch.cat([torch.sin(t_ang).expand(B, 1, H, W), torch.cos(t_ang).expand(B, 1, H, W)], dim=1)
        x = torch.cat([r_noisy, cond, t_emb], dim=1)
        h = F.silu(self.conv_in(x))
        h = h + F.silu(self.conv_mid(h))
        return self.conv_out(h)


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


class ResidualHead(nn.Module):
    """Deterministic residual corrector: Delta_hat = f(kernel prediction, forcings)."""

    def __init__(self, target_channels: int, cond_channels: int, hidden_channels: int = 16):
        super().__init__()
        self.target_channels = int(target_channels)
        self.cond_channels = int(cond_channels)
        self.hidden_channels = int(hidden_channels)
        self.conv_in = nn.Conv2d(cond_channels, hidden_channels, 3, padding=1, padding_mode="circular")
        self.conv_mid = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode="circular")
        self.conv_out = nn.Conv2d(hidden_channels, target_channels, 1)
        nn.init.zeros_(self.conv_out.weight)  # start as the identity correction (Delta_hat = 0)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv_in(cond))
        h = h + F.silu(self.conv_mid(h))
        return self.conv_out(h)


def fit_residual_head(
    residuals: np.ndarray | torch.Tensor | None = None,
    cond: np.ndarray | torch.Tensor | None = None,
    cfg=None,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> ResidualHead:
    """Fit the deterministic residual head on Delta = y_true - y_kernel (training scenarios only)."""
    if residuals is None or cond is None or cfg is None or not _cfg_get(cfg, "enabled", False):
        raise NotImplementedError("Residual correction is disabled (set residual.enabled=true)")
    r_t = torch.as_tensor(residuals, dtype=torch.float32, device=device)
    c_t = torch.as_tensor(cond, dtype=torch.float32, device=device)
    model = ResidualHead(r_t.shape[1], c_t.shape[1], int(_cfg_get(cfg, "channels", 16))).to(device)
    if sum(p.numel() for p in model.parameters()) > float(_cfg_get(cfg, "max_params", 2.0e6)):
        raise ValueError("ResidualHead exceeds configured max_params budget")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(epochs):
        loss = F.mse_loss(model(c_t), r_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def apply_residual_correction(y_kernel: np.ndarray, model: nn.Module, cond, steps: int = 25, seed: int = 42) -> np.ndarray:
    """Final output = kernel prediction + residual correction (head or diffusion sample)."""
    if isinstance(model, ResidualHead):
        dev = next(model.parameters()).device
        delta = model(torch.as_tensor(cond, dtype=torch.float32, device=dev)).cpu().numpy()
    else:
        delta = sample_residual(model, cond, steps=steps, seed=seed)
    return np.asarray(y_kernel, dtype=np.float32) + delta


def cosine_alpha_bar(steps: int, device=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (betas, alphas, alpha_bars) for a compact DDPM schedule."""
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float32, device=device) / float(steps)
    f = torch.cos((t + 0.008) / 1.008 * (math.pi / 2.0)) ** 2
    alpha_bars = f / f[0]
    betas = torch.clamp(1.0 - (alpha_bars[1:] / alpha_bars[:-1]), 1e-4, 0.999)
    alphas = 1.0 - betas
    return betas, alphas, alpha_bars[1:]


def fit_residual(
    residuals: np.ndarray | torch.Tensor | None = None,
    cond: np.ndarray | torch.Tensor | None = None,
    cfg=None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> ResidualDenoiser:
    """Fit a lightweight corrective diffusion model on kernel-model residuals r = y - y_hat.

    Raises NotImplementedError when called with no arguments or when `cfg.enabled` is False
    (preserving the stub contract tested in `tests/test_window_stubs.py`).
    """
    enabled = cfg.get("enabled", False) if hasattr(cfg, "get") else getattr(cfg, "enabled", False)
    if residuals is None or cond is None or cfg is None or not enabled:
        raise NotImplementedError("Residual diffusion is optional and disabled (set residual.enabled=true)")

    r_t = torch.as_tensor(residuals, dtype=torch.float32, device=device)
    c_t = torch.as_tensor(cond, dtype=torch.float32, device=device)
    channels = int(cfg.get("channels", 16) if hasattr(cfg, "get") else getattr(cfg, "channels", 16))
    steps = int(cfg.get("steps", 50) if hasattr(cfg, "get") else getattr(cfg, "steps", 50))
    max_params = float(cfg.get("max_params", 2.0e6) if hasattr(cfg, "get") else getattr(cfg, "max_params", 2.0e6))

    model = ResidualDenoiser(r_t.shape[1], c_t.shape[1], hidden_channels=channels).to(device)
    if sum(p.numel() for p in model.parameters()) > max_params:
        raise ValueError("ResidualDenoiser exceeds configured max_params budget")

    _, _, alpha_bars = cosine_alpha_bar(steps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    B = r_t.shape[0]

    model.train()
    for _ in range(epochs):
        t_idx = torch.randint(0, steps, (B,), device=device)
        ab = alpha_bars[t_idx].view(B, 1, 1, 1)
        noise = torch.randn_like(r_t)
        r_noisy = torch.sqrt(ab) * r_t + torch.sqrt(1.0 - ab) * noise
        t_frac = (t_idx.float() + 0.5) / float(steps)
        pred_noise = model(r_noisy, c_t, t_frac)
        loss = F.mse_loss(pred_noise, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model


@torch.no_grad()
def sample_residual(
    model: ResidualDenoiser,
    cond: np.ndarray | torch.Tensor,
    steps: int = 25,
    seed: int = 42,
) -> np.ndarray:
    """Sample corrective residual field (B, V, H, W) conditioned on coarse prediction/forcings."""
    dev = next(model.parameters()).device
    c_t = torch.as_tensor(cond, dtype=torch.float32, device=dev)
    B, _, H, W = c_t.shape
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = torch.randn((B, model.target_channels, H, W), generator=gen, device=dev)
    betas, alphas, alpha_bars = cosine_alpha_bar(steps, device=dev)

    for i in range(steps - 1, -1, -1):
        t_frac = torch.full((B,), (i + 0.5) / float(steps), device=dev)
        eps_pred = model(r, c_t, t_frac)
        alpha = alphas[i]
        ab = alpha_bars[i]
        mean = (r - ((1.0 - alpha) / torch.sqrt(1.0 - ab)) * eps_pred) / torch.sqrt(alpha)
        if i > 0:
            noise = torch.randn(r.shape, generator=gen, device=dev)
            r = mean + torch.sqrt(betas[i]) * noise
        else:
            r = mean
    return r.cpu().numpy()
