"""Unit and integration tests for Layers 1-5:
- Layer 1: SpatialKernelLayer (all 4 Ferretti et al., 2026 families), TemporalExponentialKernel, KernelEmulator, train.run_kernel_mlp
- Layer 2: Suri & Sonnewald (2026) LRP-epsilon, Integrated Gradients SHAP, and mechanistic regime fidelity verification
- Layer 3: Timestral known_covariates preparation, quantile forecasting, and CRPS/pinball metrics
- Layer 4: ResidualDenoiser and DDPM training/sampling
- Layer 5: Action-conditioned regional and global counterfactual forcing perturbations
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from cke.baselines import LocalMLP
from cke.data import ClimateBenchDataset, stack_scenarios
from cke.interp import (
    define_mechanistic_regimes,
    integrated_gradients_shap,
    lrp_local_mlp,
    verify_regime_fidelity,
)
from cke.kernels import (
    KernelEmulator,
    load_checkpoint,
    save_checkpoint,
    SpatialKernelLayer,
    TemporalExponentialKernel,
    build_temporal_history,
)
from cke.metrics import crps_from_quantiles, pinball_loss
from cke.residual_diffusion import fit_residual, sample_residual
from cke.temporal import TIMESTRAL_DEFAULT_QUANTILES, forecast_series, prepare_timestral_tensors
from cke.train import run_kernel_mlp
from cke.whatif import apply_action
from test_data import _write


@pytest.mark.parametrize("family", ["gaussian", "mog", "tophat", "exponential"])
def test_spatial_kernel_families_normalization_and_constant_preservation(family):
    lat = np.linspace(-80.0, 80.0, 6)
    lon = np.linspace(0.0, 300.0, 8)
    layer = SpatialKernelLayer(in_channels=2, n_kernels=3, lat=lat, lon=lon, family=family, anisotropic=True)

    W = layer.compute_weights()
    assert W.shape == (2, 3, 48, 48)
    # Weights must be non-negative and sum to 1 over source cells
    assert torch.all(W >= 0.0)
    assert torch.allclose(W.sum(dim=-1), torch.ones(2, 3, 48), atol=1e-5)

    # Constant spatial map integrated with normalized kernels must equal the same constant
    x_const = torch.full((2, 2, 6, 8), 3.75)
    z = layer(x_const)
    assert z.shape == (2, 6, 6, 8)
    assert torch.allclose(z, torch.full_like(z, 3.75), atol=1e-5)

    fp = layer.kernel_footprints(lat_idx=3, lon_idx=4)
    assert fp.shape == (2, 3, 6, 8)
    info = layer.export_parameters()
    assert info["family"] == family and info["n_kernels"] == 3


def test_temporal_history_respects_scenario_boundaries():
    # Two scenarios of length 3 each: scenario 'a' has values 10, 20, 30; scenario 'b' has 100, 200, 300
    g = np.array([[10.0], [20.0], [30.0], [100.0], [200.0], [300.0]], dtype=np.float32)
    scen = np.array(["a", "a", "a", "b", "b", "b"])
    hist = build_temporal_history(g, scen, window=4)
    assert hist.shape == (6, 1, 4)
    # At index 3 (first year of scenario 'b'), all lags must clamp to index 3 (100.0) and never leak 30.0 from 'a'
    assert np.allclose(hist[3, 0], [100.0, 100.0, 100.0, 100.0])
    assert np.allclose(hist[4, 0], [200.0, 100.0, 100.0, 100.0])

    tk = TemporalExponentialKernel(in_features=1, n_kernels=2, window=4)
    out = tk(torch.from_numpy(hist))
    assert out.shape == (6, 2)
    assert np.isclose(out[3, 0].item(), 100.0, atol=1e-5)


def test_kernel_emulator_forward_and_parameter_budget():
    T, H, W = 4, 8, 12
    lat = np.linspace(-85.0, 85.0, H)
    lon = np.linspace(0.0, 330.0, W)
    ds = ClimateBenchDataset(np.random.rand(T, 2), np.random.rand(T, 2, H, W), np.random.rand(T, 2, H, W))
    g_hist = torch.rand(T, 2, 6)

    model = KernelEmulator(
        n_global=2,
        n_maps=2,
        out_channels=2,
        lat=lat,
        lon=lon,
        n_spatial_kernels=4,
        family="gaussian",
        n_temporal_kernels=2,
        temporal_window=6,
        hidden=(32, 32),
    )
    y_hat = model(ds.x, g_hist=g_hist)
    assert y_hat.shape == (T, 2, H, W)
    summary = model.export_kernel_summary()
    assert summary["trainable_parameters"] < 5000
    assert summary["total_parameters"] == summary["trainable_parameters"] + summary["anchor_coefficients"]


def test_run_kernel_mlp_end_to_end(tmp_path):
    _write(tmp_path, "historical", T=6, H=8, W=12)
    _write(tmp_path, "ssp245", T=6, H=8, W=12)
    tr = stack_scenarios(tmp_path, ["historical"], coarsen=2)
    te = stack_scenarios(tmp_path, ["ssp245"], coarsen=2)
    cfg = OmegaConf.create({
        "seed": 42,
        "device": "cpu",
        "data": {"variables": ["tas", "pr"]},
        "model": {
            "name": "kernel_mlp",
            "family": "gaussian",
            "n_spatial_kernels": 3,
            "anisotropic": True,
            "n_temporal_kernels": 2,
            "temporal_window": 4,
            "hidden": [16, 16],
            "dropout": 0.0,
        },
        "train": {"epochs": 4, "batch_size": 3, "lr": 1e-2, "weight_decay": 1e-5},
    })
    res, preds, model = run_kernel_mlp(cfg, tr, te)
    assert set(res.keys()) == {"tas", "pr"}
    assert preds["tas"].shape == (6, 4, 6)
    assert np.isfinite(res["tas"]["nrmse_total"])


def test_suri_sonnewald_lrp_shap_and_mechanistic_regimes():
    B, C, H, W = 3, 6, 12, 18
    lat = np.linspace(-85.0, 85.0, H)
    lon = np.linspace(0.0, 340.0, W)
    torch.manual_seed(0)

    # Zero-bias MLP satisfies exact LRP conservation: sum_c R_c == f(x)
    mlp = LocalMLP(in_channels=C, out_channels=2, hidden=(16, 16))
    for m in mlp.net:
        if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    mlp.eval()

    feat = torch.rand(B, C, H, W) + 0.1
    with torch.no_grad():
        pred = mlp(feat)[:, 0]  # (B, H, W)
    lrp_r = lrp_local_mlp(mlp, feat, target_idx=0, eps=1e-7)
    assert lrp_r.shape == (B, C, H, W)
    assert torch.allclose(lrp_r.sum(dim=1), pred, rtol=1e-3, atol=1e-4)

    shap_r = integrated_gradients_shap(mlp, feat, baseline=torch.zeros_like(feat), target_idx=0, steps=12)
    assert shap_r.shape == (B, C, H, W)
    assert torch.allclose(shap_r.sum(dim=1), pred, rtol=2e-2, atol=2e-3)

    regimes = define_mechanistic_regimes(lat, lon)
    assert set(regimes.keys()) == {"aerosol_hubs", "arctic_amplification", "tropical_itcz", "southern_ocean"}
    report = verify_regime_fidelity(shap_r, lrp_r, regimes, lat)
    for reg_stats in report.values():
        assert reg_stats["cosine_similarity"] > 0.85


def test_timestral_temporal_module_and_crps():
    years_hist = np.arange(1995, 2015)
    years_test = np.arange(2015, 2025)
    df_hist = pd.DataFrame({
        "year": years_hist,
        "scenario": "historical",
        "CO2": np.linspace(360, 400, len(years_hist)),
        "CH4": np.ones(len(years_hist)),
        "SO2_global": np.ones(len(years_hist)) * 0.5,
        "BC_global": np.ones(len(years_hist)) * 0.2,
        "global_tas": np.linspace(0.5, 1.0, len(years_hist)),
        "global_pr": np.linspace(2.8, 2.9, len(years_hist)),
    })
    df_test = pd.DataFrame({
        "year": years_test,
        "scenario": "ssp245",
        "CO2": np.linspace(402, 430, len(years_test)),
        "CH4": np.ones(len(years_test)) * 1.1,
        "SO2_global": np.ones(len(years_test)) * 0.4,
        "BC_global": np.ones(len(years_test)) * 0.15,
        "global_tas": np.linspace(1.05, 1.35, len(years_test)),
        "global_pr": np.linspace(2.91, 2.98, len(years_test)),
    })
    df = pd.concat([df_hist, df_test], ignore_index=True)
    tensors = prepare_timestral_tensors(df)
    assert tensors["context"].shape == (1, 2, 20)
    assert tensors["known_covariates"].shape == (1, 4, 30)

    class DummyTimestral:
        def forecast(self, context, prediction_length, known_covariates=None):
            assert known_covariates is not None and known_covariates.shape == (1, 4, 30)
            B, V, _ = context.shape
            Q = len(TIMESTRAL_DEFAULT_QUANTILES)
            base = tensors["target_truth"][..., None]  # (1, 2, 10, 1)
            offsets = (TIMESTRAL_DEFAULT_QUANTILES - 0.5)[None, None, None, :] * 0.2
            q_pred = base + offsets
            return SimpleNamespace(quantiles=q_pred, median=q_pred[..., Q // 2])

    out = forecast_series(
        tensors["context"],
        tensors["known_covariates"],
        prediction_length=10,
        cfg={"enabled": True},
        forecaster=DummyTimestral(),
    )
    assert out["quantiles"].shape == (1, 2, 10, 21)
    crps = crps_from_quantiles(tensors["target_truth"], out["quantiles"], out["quantile_levels"])
    p_loss = pinball_loss(tensors["target_truth"], out["quantiles"], out["quantile_levels"])
    assert 0.0 <= crps < 0.1 and 0.0 <= p_loss < 0.1


def test_residual_diffusion_and_whatif_enabled():
    rng = np.random.default_rng(7)
    residuals = rng.normal(0, 0.1, size=(6, 2, 8, 12)).astype(np.float32)
    cond = rng.normal(0, 1.0, size=(6, 4, 8, 12)).astype(np.float32)
    cfg = {"enabled": True, "channels": 8, "steps": 10, "max_params": 50000}
    denoiser = fit_residual(residuals, cond, cfg=cfg, epochs=3, lr=1e-2)
    sampled = sample_residual(denoiser, cond[:2], steps=5)
    assert sampled.shape == (2, 2, 8, 12) and np.all(np.isfinite(sampled))

    split = {
        "global": np.ones((4, 2), dtype=np.float32) * 100.0,
        "maps": np.ones((4, 2, 8, 12), dtype=np.float32) * 2.0,
        "lat": np.linspace(-80, 80, 8),
        "lon": np.linspace(0, 330, 12),
    }
    perturbed = apply_action(
        split,
        [
            {"type": "forcing_scale", "var": "SO2", "factor": 0.0, "lat_range": [0, 60], "lon_range": [60, 150]},
            {"type": "forcing_offset", "var": "CO2", "delta": -50.0},
        ],
    )
    assert np.allclose(perturbed["global"][:, 0], 50.0)
    assert perturbed["maps"][:, 0].min() == 0.0 and perturbed["maps"][:, 0].max() == 2.0



def test_kernel_checkpoint_round_trip_restores_anchor_and_stats(tmp_path):
    T, H, W = 5, 8, 12
    lat = np.linspace(-85.0, 85.0, H)
    lon = np.linspace(0.0, 330.0, W)
    kw = dict(n_global=2, n_maps=2, out_channels=2, lat=lat, lon=lon, n_spatial_kernels=3,
              family="gaussian", n_temporal_kernels=2, temporal_window=4, hidden=(16, 16))
    torch.manual_seed(0)
    model = KernelEmulator(**kw)
    g = np.random.rand(T, 2).astype(np.float32)
    y = np.random.rand(T, 2, H, W).astype(np.float32)
    model.fit_spatial_anchor(g, y)
    model.input_stats = {"g_mean": np.ones(2, dtype=np.float32), "g_std": np.full(2, 2.0, dtype=np.float32)}
    model.eval()
    ds = ClimateBenchDataset(g, np.random.rand(T, 2, H, W), y)
    g_hist = torch.rand(T, 2, 4)
    with torch.no_grad():
        ref = model(ds.x, g_hist=g_hist)
    save_checkpoint(model, tmp_path / "model.pt")

    torch.manual_seed(123)
    fresh = KernelEmulator(**kw)
    load_checkpoint(fresh, tmp_path / "model.pt")
    fresh.eval()
    assert fresh.has_anchor
    assert np.allclose(fresh.input_stats["g_std"], 2.0)
    with torch.no_grad():
        out = fresh(ds.x, g_hist=g_hist)
    assert torch.allclose(ref, out, atol=1e-6)
    summary = fresh.export_kernel_summary()
    assert summary["anchor_coefficients"] == 2 * H * W * (1 + 2)


def test_residual_head_kernel_plus_delta():
    from cke.residual_diffusion import ResidualHead, apply_residual_correction, fit_residual_head
    rng = np.random.default_rng(3)
    cond = rng.normal(0, 1.0, size=(8, 3, 8, 12)).astype(np.float32)
    delta = (0.5 * cond[:, :2]).astype(np.float32)  # learnable structured residual
    y_kernel = rng.normal(0, 1.0, size=(8, 2, 8, 12)).astype(np.float32)
    cfg = {"enabled": True, "mode": "head", "channels": 8, "max_params": 50000}
    head = fit_residual_head(delta, cond, cfg=cfg, epochs=150)
    assert isinstance(head, ResidualHead)
    zero = ResidualHead(2, 3, 8)
    assert np.allclose(apply_residual_correction(y_kernel, zero, cond), y_kernel)  # starts as identity
    y_final = apply_residual_correction(y_kernel, head, cond)
    y_true = y_kernel + delta
    assert np.mean((y_final - y_true) ** 2) < 0.5 * np.mean(delta ** 2)
    with pytest.raises(NotImplementedError):
        fit_residual_head(delta, cond, cfg={"enabled": False})
