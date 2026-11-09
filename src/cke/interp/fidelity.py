"""Layer 2: Physics-grounded fidelity verification (Suri & Sonnewald, 2026).

References:
- Suri, S., & Sonnewald, M. (2026). "Trusting Machine Learning With Physics:
  A Fidelity Verification Framework for Complex Systems." Journal of Geophysical
  Research: Machine Learning and Computation, 3(6), e2025JH001211, doi:10.1029/2025JH001211.
- Suri, S., Ringel, K., & Sonnewald, M. (2026). "OceanCBM: A Concept Bottleneck Model
  for Mechanistic Interpretability in Ocean Forecasting." arXiv:2605.12639.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from cke.baselines.local_mlp import LocalMLP
from cke.metrics import lat_weights


@torch.no_grad()
def lrp_local_mlp(
    mlp: LocalMLP,
    feat_stack: torch.Tensor,
    target_idx: int = 0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Exact Layer-wise Relevance Propagation (LRP-epsilon rule) through LocalMLP.

    Propagates output relevance for `target_idx` backward through the 1x1 Conv2d and ReLU
    layers of `LocalMLP` to the input feature channels `feat_stack` of shape (B, C_in, H, W).
    """
    conv_layers: list[nn.Conv2d] = []
    activations: list[torch.Tensor] = [feat_stack]
    h = feat_stack
    for mod in mlp.net:
        if isinstance(mod, nn.Conv2d):
            conv_layers.append(mod)
            h = mod(h)
        elif isinstance(mod, nn.ReLU):
            h = torch.relu(h)
            activations.append(h)
        # Dropout is identity in eval mode

    # Final output before selecting target_idx: h is (B, V, H, W)
    R = torch.zeros_like(h)
    R[:, target_idx] = h[:, target_idx]

    # Propagate backward through each 1x1 Conv2d layer
    for layer_idx in range(len(conv_layers) - 1, -1, -1):
        conv = conv_layers[layer_idx]
        a_in = activations[layer_idx]  # (B, C_in, H, W)
        w = conv.weight[:, :, 0, 0]     # (C_out, C_in)
        z = torch.einsum("bchw,oc->bohw", a_in, w)
        if conv.bias is not None:
            z = z + conv.bias.view(1, -1, 1, 1)
        denom = z + eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
        s = R / denom  # (B, C_out, H, W)
        c_back = torch.einsum("bohw,oc->bchw", s, w)
        R = a_in * c_back

    return R


def integrated_gradients_shap(
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor | None = None,
    g_hist: torch.Tensor | None = None,
    g_hist_baseline: torch.Tensor | None = None,
    target_idx: int = 0,
    steps: int = 16,
) -> torch.Tensor:
    """Compute Integrated-Gradients / Expected-Gradients SHAP attributions on feature stack.

    If `model` is a `KernelEmulator`, attributes relevance to its assembled local + nonlocal
    feature stack `assemble_features(x, g_hist)` of shape (B, C_total, H, W). Otherwise
    attributes directly to `x`.
    """
    model.eval()
    has_assemble = hasattr(model, "assemble_features") and hasattr(model, "local_predictor")
    with torch.no_grad():
        if has_assemble:
            f_in = model.assemble_features(x, g_hist=g_hist)
            if baseline is None:
                f_base = torch.zeros_like(f_in)
            else:
                f_base = model.assemble_features(baseline, g_hist=g_hist_baseline)
        else:
            f_in = x
            f_base = torch.zeros_like(x) if baseline is None else baseline

    diff = f_in - f_base
    total_grad = torch.zeros_like(f_in)
    alphas = torch.linspace(0.5 / steps, 1.0 - 0.5 / steps, steps, device=f_in.device)

    predictor = model.local_predictor if has_assemble else model
    for alpha in alphas:
        f_step = (f_base + alpha * diff).detach().requires_grad_(True)
        out = predictor(f_step)[:, target_idx].sum()
        grad = torch.autograd.grad(out, f_step, create_graph=False)[0]
        total_grad = total_grad + grad.detach()

    return diff * (total_grad / float(steps))


def define_mechanistic_regimes(
    lat: np.ndarray,
    lon: np.ndarray,
    so2_mean: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Partition the (H, W) spherical grid into 4 mechanistic dynamical regimes (Suri & Sonnewald, 2026).

    Returns boolean masks of shape (H, W) for:
      - 'aerosol_hubs': high anthropogenic SO2 source regions (or South/East Asian & European industrial belt)
      - 'arctic_amplification': northern high-latitude cryosphere feedback zone (lat >= 60N)
      - 'tropical_itcz': deep convective precipitation band (|lat| <= 15)
      - 'southern_ocean': high thermal-inertia circumpolar ocean band (lat <= -45S)
    """
    lat_arr = np.asarray(lat, dtype=float)[:, None]
    lon_arr = np.asarray(lon, dtype=float)[None, :]
    H, W = lat_arr.shape[0], lon_arr.shape[1]
    lon_360 = lon_arr % 360.0

    if so2_mean is not None and so2_mean.shape == (H, W) and np.nanmax(so2_mean) > np.nanmin(so2_mean):
        thresh = np.nanpercentile(so2_mean, 85.0)
        aerosol_mask = so2_mean >= thresh
    else:
        aerosol_mask = (lat_arr >= 10.0) & (lat_arr <= 55.0) & (lon_360 >= 65.0) & (lon_360 <= 145.0)

    return {
        "aerosol_hubs": np.broadcast_to(aerosol_mask, (H, W)).copy(),
        "arctic_amplification": np.broadcast_to(lat_arr >= 60.0, (H, W)).copy(),
        "tropical_itcz": np.broadcast_to(np.abs(lat_arr) <= 15.0, (H, W)).copy(),
        "southern_ocean": np.broadcast_to(lat_arr <= -45.0, (H, W)).copy(),
    }


def verify_regime_fidelity(
    shap_attr: np.ndarray | torch.Tensor,
    lrp_attr: np.ndarray | torch.Tensor,
    regimes: dict[str, np.ndarray],
    lat: np.ndarray,
) -> dict[str, dict]:
    """Compare SHAP and LRP attributions across mechanistic regimes (Suri & Sonnewald, 2026).

    Args:
        shap_attr: (T, C, H, W) or (C, H, W) attribution tensor/array
        lrp_attr:  (T, C, H, W) or (C, H, W) attribution tensor/array
        regimes:   dict of (H, W) boolean masks from `define_mechanistic_regimes`
        lat:       (H,) latitude array in degrees

    Returns:
        Dict keyed by regime name with:
          - 'cosine_similarity': agreement between area-weighted mean |SHAP| and |LRP| channel vectors
          - 'shap_channel_importance': normalized channel importance in that regime
          - 'lrp_channel_importance': normalized channel importance in that regime
          - 'dominant_channel_agree': bool whether top attributed channel matches between SHAP and LRP
    """
    s_np = shap_attr.detach().cpu().numpy() if isinstance(shap_attr, torch.Tensor) else np.asarray(shap_attr)
    l_np = lrp_attr.detach().cpu().numpy() if isinstance(lrp_attr, torch.Tensor) else np.asarray(lrp_attr)
    if s_np.ndim == 4:
        s_np = np.abs(s_np).mean(axis=0)
        l_np = np.abs(l_np).mean(axis=0)
    else:
        s_np = np.abs(s_np)
        l_np = np.abs(l_np)

    w_lat = lat_weights(lat)[:, None]  # (H, 1)
    report: dict[str, dict] = {}

    for name, mask in regimes.items():
        if not np.any(mask):
            continue
        w_reg = np.where(mask, w_lat, 0.0)
        w_sum = max(float(w_reg.sum()), 1e-12)
        s_vec = (s_np * w_reg[None]).sum(axis=(-2, -1)) / w_sum  # (C,)
        l_vec = (l_np * w_reg[None]).sum(axis=(-2, -1)) / w_sum  # (C,)

        s_norm = s_vec / max(float(np.linalg.norm(s_vec)), 1e-12)
        l_norm = l_vec / max(float(np.linalg.norm(l_vec)), 1e-12)
        cos_sim = float(np.dot(s_norm, l_norm))

        s_share = (s_vec / max(float(s_vec.sum()), 1e-12)).tolist()
        l_share = (l_vec / max(float(l_vec.sum()), 1e-12)).tolist()

        report[name] = {
            "cosine_similarity": cos_sim,
            "shap_channel_importance": s_share,
            "lrp_channel_importance": l_share,
            "dominant_channel_agree": bool(int(np.argmax(s_vec)) == int(np.argmax(l_vec))),
        }

    return report

