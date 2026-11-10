"""Generate publication-quality plots and quantitative outcomes for climate-kernel-emulator.

Produces 5 figures and a complete JSON/Markdown outcomes summary:
  1. fig1_nrmse_ablation.png: ClimateBench NRMSE_s, NRMSE_g, NRMSE_t across baselines & 4 kernel families
  2. fig2_spatial_maps_ssp245.png: Held-out SSP245 (2080-2100) truth, predictions, and spatial bias maps
  3. fig3_learned_kernel_footprints.png: Learned spatial footprints (Ferretti et al., 2026) & temporal memory decay
  4. fig4_suri_sonnewald_fidelity.png: Integrated-gradients vs. LRP-epsilon attributions of the local predictor across 4 mechanistic regimes (Suri & Sonnewald, 2026)
  5. fig5_whatif_and_trajectories.png: Global-mean trajectories (with quantile fan) & Asian SO2 clean-air what-if counterfactual
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
import xarray as xr

from cke.baselines import LocalMLP, PatternScaling
from cke.data import ClimateBenchDataset, Normaliser, select_years, stack_scenarios
from cke.interp import (
    define_mechanistic_regimes,
    integrated_gradients_shap,
    lrp_local_mlp,
    verify_regime_fidelity,
)
from cke.kernels import KernelEmulator, build_temporal_history
from cke.metrics import all_metrics, crps_from_quantiles, global_mean, lat_weights, pinball_loss
from cke.residual_diffusion import fit_residual, sample_residual
from cke.train import run_kernel_mlp, run_local_mlp, run_pattern_scaling
from cke.utils.seed import seed_everything
from cke.whatif import apply_action


def find_real_climatebench_root(base: Path) -> Path | None:
    """Return directory containing inputs_historical.nc if available."""
    for p in [base] + list(base.glob("**")):
        if p.is_dir() and (p / "inputs_historical.nc").exists() and (p / "inputs_ssp245.nc").exists():
            return p
    return None


def generate_physically_structured_surrogate(out_dir: Path, H: int = 48, W: int = 72):
    """Create a physically structured ClimateBench-format dataset when real NetCDFs are not yet extracted.

    Embeds:
      - Realistic polar amplification (Arctic warming ~2.2x global mean)
      - Nonlocal upwind aerosol cooling & hydrological suppression (SO2 plumes in East/South Asia & Europe
        advected ~15-25 deg eastward and diffused across synoptic ~1500 km scales)
      - Multi-year ocean thermal inertia lag (tau ~ 6 years)
      - ITCZ convective precipitation shift under warming
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    lat = np.linspace(-88.125, 88.125, H)
    lon = np.linspace(0.0, 355.0, W)
    phi = np.deg2rad(lat)[:, None]
    lam = np.deg2rad(lon)[None, :]

    # Spatial patterns: polar amplification + land/ocean contrast proxy
    polar_amp = 0.75 + 0.85 * (np.sin(phi) ** 2) + 0.45 * np.maximum(0.0, np.sin(phi))
    land_proxy = 1.0 + 0.25 * np.cos(2.0 * lam) * np.cos(phi)
    tas_pattern = polar_amp * land_proxy  # (H, W)

    # ITCZ + midlatitude storm track pattern for precipitation
    pr_pattern = 1.8 * np.exp(-((lat[:, None] / 14.0) ** 2)) + 0.7 * np.exp(-(((np.abs(lat[:, None]) - 48.0) / 12.0) ** 2))
    pr_pattern = np.broadcast_to(pr_pattern, (H, W)) * (1.0 + 0.15 * np.sin(lam))

    # Construct true nonlocal advection-diffusion operator on (H, W) for aerosol response
    phi_g, lam_g = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    p_flat, l_flat = phi_g.reshape(-1), lam_g.reshape(-1)
    dp = p_flat[None, :] - p_flat[:, None]
    dl = (l_flat[None, :] - l_flat[:, None] + np.pi) % (2 * np.pi) - np.pi
    # Downwind eastward advection shift (-0.28 rad ~ 16 deg) and synoptic width (0.32 rad ~ 2000 km)
    W_nonlocal = np.exp(-0.5 * ((dp / 0.25) ** 2 + ((dl + 0.28) / 0.35) ** 2)) * np.clip(np.cos(p_flat)[None, :], 1e-4, None)
    W_nonlocal /= W_nonlocal.sum(axis=-1, keepdims=True)

    # Industrial emission hubs (South/East Asia + Europe + Eastern North America)
    asia_plume = np.exp(-(((lat[:, None] - 28.0) / 12.0) ** 2) - (((lon[None, :] - 110.0) / 22.0) ** 2))
    eur_plume = np.exp(-(((lat[:, None] - 48.0) / 10.0) ** 2) - (((lon[None, :] - 15.0) / 18.0) ** 2))
    nam_plume = np.exp(-(((lat[:, None] - 38.0) / 10.0) ** 2) - (((lon[None, :] - 275.0) / 18.0) ** 2))

    scenarios_spec = {
        "historical": (np.arange(1950, 2015), 200.0, 1200.0, 0.8, 1.2, 1.0),
        "ssp126":     (np.arange(2015, 2101), 1250.0, 1800.0, 0.5, 0.15, 0.75),
        "ssp245":     (np.arange(2015, 2101), 1250.0, 2650.0, 0.6, 0.25, 0.90),
        "ssp370":     (np.arange(2015, 2101), 1250.0, 3800.0, 0.8, 0.75, 1.25),
        "ssp585":     (np.arange(2015, 2101), 1250.0, 4800.0, 0.7, 0.20, 1.15),
        "hist-GHG":   (np.arange(1950, 2015), 200.0, 1200.0, 0.0, 0.0, 1.0),
        "hist-aer":   (np.arange(1950, 2015), 0.0,   0.0,    0.8, 1.2, 0.0),
    }

    for scen, (years, co2_start, co2_end, aer_start, aer_end, ch4_scale) in scenarios_spec.items():
        T = len(years)
        tau_frac = np.linspace(0.0, 1.0, T)
        co2 = co2_start + (co2_end - co2_start) * (tau_frac ** 1.15)
        ch4 = 0.35 * ch4_scale * (1.0 + 0.8 * np.sin(tau_frac * np.pi * 0.85))
        aer_env = aer_start + (aer_end - aer_start) * np.sin(tau_frac * np.pi * 0.75 + 0.2)

        so2 = np.zeros((T, H, W), dtype=np.float32)
        bc = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            so2[t] = aer_env[t] * (1.3 * asia_plume + 0.7 * eur_plume + 0.6 * nam_plume) + rng.normal(0, 0.02, (H, W))
            bc[t] = 0.45 * aer_env[t] * (1.5 * asia_plume + 0.4 * eur_plume) + rng.normal(0, 0.01, (H, W))
        so2 = np.clip(so2, 0.0, None)
        bc = np.clip(bc, 0.0, None)

        # Causal thermal memory of CO2 + CH4 (tau = 5 years)
        eff_ghg = np.zeros(T, dtype=np.float32)
        w_exp = np.exp(-np.arange(8) / 4.5)
        w_exp /= w_exp.sum()
        raw_ghg = 5.2e-4 * co2 + 0.42 * ch4
        for t in range(T):
            acc = 0.0
            for lag in range(8):
                acc += w_exp[lag] * raw_ghg[max(0, t - lag)]
            eff_ghg[t] = acc

        # Nonlocal aerosol field via W_nonlocal
        so2_nonlocal = (so2.reshape(T, -1) @ W_nonlocal.T).reshape(T, H, W)
        bc_nonlocal = (bc.reshape(T, -1) @ W_nonlocal.T).reshape(T, H, W)

        # True forced response: nonlocal aerosol cooling + nonlinear saturation + GHG warming
        tas_mean = (
            eff_ghg[:, None, None] * tas_pattern[None]
            - 0.85 * np.tanh(1.2 * so2_nonlocal)
            + 0.30 * bc_nonlocal
            - 0.15 * so2
        )
        pr_mean = (
            0.12 * eff_ghg[:, None, None] * pr_pattern[None]
            - 0.45 * so2_nonlocal * pr_pattern[None]
            - 0.18 * bc_nonlocal
            + 0.05 * np.tanh(tas_mean)
        )

        # 3 ensemble members with internal variability
        tas_ens = tas_mean[None] + rng.normal(0, 0.04, (3, T, H, W)).astype(np.float32)
        pr_ens = (pr_mean[None] + rng.normal(0, 0.02, (3, T, H, W))).astype(np.float32) / 86400.0

        xr.Dataset(
            {
                "CO2": ("time", co2.astype(np.float32)),
                "CH4": ("time", ch4.astype(np.float32)),
                "SO2": (("time", "latitude", "longitude"), so2),
                "BC": (("time", "latitude", "longitude"), bc),
            },
            coords={"time": years, "latitude": lat, "longitude": lon},
        ).to_netcdf(out_dir / f"inputs_{scen}.nc")

        xr.Dataset(
            {
                "tas": (("member", "time", "latitude", "longitude"), tas_ens),
                "pr": (("member", "time", "latitude", "longitude"), pr_ens),
            },
            coords={"member": [0, 1, 2], "time": years, "latitude": lat, "longitude": lon},
        ).to_netcdf(out_dir / f"outputs_{scen}.nc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/climatebench")
    ap.add_argument("--out-dir", default="outputs/benchmark_suite")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--coarsen", type=int, default=2)
    ap.add_argument("--allow-surrogate", action="store_true",
                    help="If real ClimateBench data is missing, fall back to a synthetic surrogate (smoke test only; results are not ClimateBench scores).")
    args = ap.parse_args()

    seed_everything(42)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    real_root = find_real_climatebench_root(Path(args.data_root))
    if real_root is not None:
        data_root = real_root
        coarsen = args.coarsen
        data_source = f"ClimateBench v1.0.0 ({real_root})"
    elif not args.allow_surrogate:
        raise SystemExit(
            f"[cke] Real ClimateBench data not found under {args.data_root!r}. "
            "Download it with scripts/download_climatebench.py, or pass --allow-surrogate "
            "to run a synthetic smoke test (its outputs are NOT ClimateBench scores)."
        )
    else:
        print("[cke] WARNING: real ClimateBench data not found; using a SYNTHETIC surrogate. Results are not ClimateBench scores.")
        data_root = out_dir / "surrogate_climatebench"
        generate_physically_structured_surrogate(data_root, H=24, W=36)
        coarsen = 1
        data_source = "Physically Structured ClimateBench Protocol Surrogate (24x36 grid, 7 scenarios)"

    print(f"[cke] Running benchmark suite on: {data_source}")
    train_scenarios = ["historical", "ssp126", "ssp370", "ssp585", "hist-GHG", "hist-aer"]
    test_scenarios = ["ssp245"]
    variables = ["tas", "pr"]

    tr = stack_scenarios(data_root, train_scenarios, variables=variables, coarsen=coarsen)
    te_full = stack_scenarios(data_root, test_scenarios, variables=variables, coarsen=coarsen)
    te = select_years(te_full, 2080, 2100)

    results: dict[str, dict] = {}
    predictions: dict[str, dict] = {}

    # 1. Pattern Scaling Baseline
    base_cfg = OmegaConf.create({
        "seed": 42,
        "device": "cpu",
        "data": {"variables": variables, "coarsen": coarsen},
        "train": {"epochs": args.epochs, "batch_size": 256, "lr": 5e-3, "weight_decay": 1e-4},
    })
    res_ps, pred_ps = run_pattern_scaling(base_cfg, tr, te)
    results["PatternScaling"] = res_ps
    predictions["PatternScaling"] = pred_ps

    # 2. Local MLP Baseline
    cfg_lmlp = OmegaConf.merge(base_cfg, {"model": {"name": "local_mlp", "hidden": [32, 32], "dropout": 0.0}})
    res_lmlp, pred_lmlp, model_lmlp = run_local_mlp(cfg_lmlp, tr, te)
    results["LocalMLP"] = res_lmlp
    predictions["LocalMLP"] = pred_lmlp

    # 3. Nonlocal Kernel Emulator across all 4 Ferretti et al. (2026) families
    kernel_models: dict[str, KernelEmulator] = {}
    kernel_summaries: dict[str, dict] = {}
    for fam in ["gaussian", "mog", "tophat", "exponential"]:
        seed_everything(42)
        cfg_k = OmegaConf.merge(
            base_cfg,
            {
                "model": {
                    "name": f"kernel_{fam}",
                    "family": fam,
                    "n_spatial_kernels": 4,
                    "anisotropic": True,
                    "n_components": 2,
                    "n_temporal_kernels": 2,
                    "temporal_window": 8,
                    "include_local_maps": True,
                    "coord_features": True,
                    "hidden": [32, 32],
                    "dropout": 0.0,
                }
            },
        )
        label = f"Kernel-{fam.capitalize()}"
        res_k, pred_k, mod_k = run_kernel_mlp(cfg_k, tr, te)
        results[label] = res_k
        predictions[label] = pred_k
        kernel_models[fam] = mod_k
        kernel_summaries[fam] = mod_k.export_kernel_summary()

    best_k_model = kernel_models["gaussian"]

    # 4. Layer 2: Suri & Sonnewald (2026) Fidelity Audit on Held-Out SSP245 (2080-2100)
    nx = Normaliser.fit({"g": tr["global"], "m": tr["maps"]})
    st = best_k_model.input_stats
    g_te_norm = ((te["global"] - st["g_mean"]) / st["g_std"]).astype(np.float32)
    m_te_norm = ((te["maps"] - st["m_mean"]) / st["m_std"]).astype(np.float32)
    y_dummy = np.zeros(( len(te["years"]), 2, len(te["lat"]), len(te["lon"]) ), dtype=np.float32)
    ds_te = ClimateBenchDataset(g_te_norm, m_te_norm, y_dummy)
    g_full_k = ((te_full["global"] - st["g_mean"]) / st["g_std"]).astype(np.float32)
    gh_full_te = build_temporal_history(g_full_k, te_full["scenario"], window=8)
    gh_te = torch.from_numpy(gh_full_te[te["_window_mask"]])

    with torch.no_grad():
        feat_stack = best_k_model.assemble_features(ds_te.x, g_hist=gh_te)
    lrp_tas = lrp_local_mlp(best_k_model.local_predictor, feat_stack, target_idx=0)
    shap_tas = integrated_gradients_shap(best_k_model, ds_te.x, g_hist=gh_te, target_idx=0, steps=16)

    so2_mean_map = te["maps"][:, 0].mean(axis=0)
    regimes = define_mechanistic_regimes(te["lat"], te["lon"], so2_mean=so2_mean_map)
    fidelity_report = verify_regime_fidelity(shap_tas, lrp_tas, regimes, te["lat"])

    # 5. Layer 5: Action-Conditioned What-If Counterfactual (100% South/East Asian SO2 Phaseout on SSP245)
    te_clean_air = apply_action(
        te,
        {"type": "forcing_scale", "var": "SO2", "factor": 0.0, "lat_range": [5.0, 50.0], "lon_range": [65.0, 145.0]},
    )
    ny = Normaliser.fit(tr["targets"])
    g_clean_norm = ((te_clean_air["global"] - st["g_mean"]) / st["g_std"]).astype(np.float32)
    m_clean_norm = ((te_clean_air["maps"] - st["m_mean"]) / st["m_std"]).astype(np.float32)
    ds_clean = ClimateBenchDataset(g_clean_norm, m_clean_norm, y_dummy)
    with torch.no_grad():
        p_clean_raw = best_k_model(ds_clean.x, g_hist=gh_te).cpu().numpy()
    pred_clean = {v: ny.invert(v, p_clean_raw[:, i]) for i, v in enumerate(variables)}
    delta_tas_clean = pred_clean["tas"].mean(0) - predictions["Kernel-Gaussian"]["tas"].mean(0)
    delta_pr_clean = pred_clean["pr"].mean(0) - predictions["Kernel-Gaussian"]["pr"].mean(0)

    # Save outcomes JSON
    outcomes = {
        "data_source": data_source,
        "grid_shape": [int(len(te["lat"])), int(len(te["lon"]))],
        "scoring_window": [int(te["years"].min()), int(te["years"].max())],
        "metrics_ssp245_2080_2100": results,
        "kernel_parameters": kernel_summaries,
        "suri_sonnewald_fidelity": fidelity_report,
        "whatif_asian_so2_phaseout": {
            "global_mean_delta_tas_K": float(global_mean(delta_tas_clean[None], te["lat"])[0]),
            "asian_hub_max_delta_tas_K": float(delta_tas_clean.max()),
            "asian_hub_max_delta_pr_mm_day": float(delta_pr_clean.max()),
        },
    }
    (out_dir / "outcomes.json").write_text(json.dumps(outcomes, indent=2))

    # =========================================================================
    # PLOT 1: NRMSE Ablation Bar Chart (tas and pr)
    # =========================================================================
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), dpi=200)
    model_names = list(results.keys())
    x_pos = np.arange(len(model_names))
    width = 0.25
    colors = ["#2b5c8f", "#3690c0", "#d95f02"]

    for ax_idx, (var, title) in enumerate([("tas", "Surface Temperature (tas)"), ("pr", "Precipitation (pr)")]):
        ax = axes[ax_idx]
        s_vals = [results[m][var]["nrmse_spatial"] for m in model_names]
        g_vals = [results[m][var]["nrmse_global"] for m in model_names]
        t_vals = [results[m][var]["nrmse_total"] for m in model_names]

        b1 = ax.bar(x_pos - width, s_vals, width, label=r"$\mathrm{NRMSE}_s$ (Spatial)", color=colors[0], alpha=0.9)
        b2 = ax.bar(x_pos, g_vals, width, label=r"$\mathrm{NRMSE}_g$ (Global)", color=colors[1], alpha=0.9)
        b3 = ax.bar(x_pos + width, t_vals, width, label=r"$\mathrm{NRMSE}_t = \mathrm{NRMSE}_s + 5\,\mathrm{NRMSE}_g$", color=colors[2], alpha=0.9)

        for bar in b3:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=22, ha="right")
        ax.set_title(f"{title} — Held-Out SSP245 (2080–2100)")
        ax.set_ylabel("Normalized RMSE (lower is better)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle("ClimateBench v1.0 Protocol Ablation: Baselines vs. Parametric Nonlocal Kernels (Ferretti et al., 2026)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_nrmse_ablation.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # PLOT 2: Spatial Maps & Error Bias on Held-Out SSP245 (2080-2100)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), dpi=200)
    lon_g, lat_g = np.meshgrid(te["lon"], te["lat"])

    tas_true = te["targets"]["tas"].mean(0)
    tas_lmlp_err = predictions["LocalMLP"]["tas"].mean(0) - tas_true
    tas_kern_err = predictions["Kernel-Gaussian"]["tas"].mean(0) - tas_true
    vmax_t = max(float(np.abs(tas_lmlp_err).max()), float(np.abs(tas_kern_err).max()), 0.1)

    im00 = axes[0, 0].pcolormesh(lon_g, lat_g, tas_true, cmap="YlOrRd", shading="auto")
    axes[0, 0].set_title("Truth: SSP245 Mean tas (2080–2100) [K]")
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im01 = axes[0, 1].pcolormesh(lon_g, lat_g, tas_lmlp_err, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t, shading="auto")
    axes[0, 1].set_title(f"LocalMLP Bias (NRMSE_s={results['LocalMLP']['tas']['nrmse_spatial']:.3f})")
    plt.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im02 = axes[0, 2].pcolormesh(lon_g, lat_g, tas_kern_err, cmap="RdBu_r", vmin=-vmax_t, vmax=vmax_t, shading="auto")
    axes[0, 2].set_title(f"Kernel-Gaussian Bias (NRMSE_s={results['Kernel-Gaussian']['tas']['nrmse_spatial']:.3f})")
    plt.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.04)

    pr_true = te["targets"]["pr"].mean(0)
    pr_lmlp_err = predictions["LocalMLP"]["pr"].mean(0) - pr_true
    pr_kern_err = predictions["Kernel-Gaussian"]["pr"].mean(0) - pr_true
    vmax_p = max(float(np.abs(pr_lmlp_err).max()), float(np.abs(pr_kern_err).max()), 0.05)

    im10 = axes[1, 0].pcolormesh(lon_g, lat_g, pr_true, cmap="BrBG", shading="auto")
    axes[1, 0].set_title("Truth: SSP245 Mean pr (2080–2100) [mm/day]")
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im11 = axes[1, 1].pcolormesh(lon_g, lat_g, pr_lmlp_err, cmap="BrBG", vmin=-vmax_p, vmax=vmax_p, shading="auto")
    axes[1, 1].set_title(f"LocalMLP Bias (NRMSE_s={results['LocalMLP']['pr']['nrmse_spatial']:.3f})")
    plt.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im12 = axes[1, 2].pcolormesh(lon_g, lat_g, pr_kern_err, cmap="BrBG", vmin=-vmax_p, vmax=vmax_p, shading="auto")
    axes[1, 2].set_title(f"Kernel-Gaussian Bias (NRMSE_s={results['Kernel-Gaussian']['pr']['nrmse_spatial']:.3f})")
    plt.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")

    fig.suptitle("Spatial Prediction Skill on Held-Out SSP245 (2080–2100): LocalMLP vs. Nonlocal Kernel Emulator",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_spatial_maps_ssp245.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # PLOT 3: Learned Spatial Kernel Footprints & Temporal Memory Decay
    # =========================================================================
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), dpi=200)
    # Target cell in East/South Asia industrial belt
    lat_idx = int(np.argmin(np.abs(te["lat"] - 28.0)))
    lon_idx = int(np.argmin(np.abs(te["lon"] - 110.0)))

    for idx, fam in enumerate(["gaussian", "mog", "tophat", "exponential"]):
        ax = axes[idx]
        fp = kernel_models[fam].spatial_kernels.kernel_footprints(lat_idx=lat_idx, lon_idx=lon_idx)
        # Plot first SO2 nonlocal kernel footprint
        im = ax.pcolormesh(lon_g, lat_g, fp[0, 0], cmap="magma", shading="auto")
        ax.plot(te["lon"][lon_idx], te["lat"][lat_idx], marker="*", color="cyan", markersize=10, label="Target Cell")
        ax.set_title(f"{fam.capitalize()} Kernel ($k_{{SO_2, 1}}$)")
        ax.set_xlabel("Lon (°E)")
        if idx == 0:
            ax.set_ylabel("Lat (°N)")
        ax.legend(loc="lower left", fontsize=7.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 5: Temporal Exponential Memory Kernel weights (Bouabid et al., 2024; Ferretti et al., 2026)
    ax_t = axes[4]
    tk = best_k_model.temporal_kernels
    w_temp = tk.compute_weights().detach().cpu().numpy()  # (F=2, M=2, L=8)
    lags = np.arange(tk.window)
    taus = tk.export_parameters()["tau_years"]
    ax_t.plot(lags, w_temp[0, 0], "-o", color="#1b9e77", label=f"CO2 Fast ($\\tau={taus[0][0]:.1f}$ yr)")
    ax_t.plot(lags, w_temp[0, 1], "-s", color="#d95f02", label=f"CO2 Slow ($\\tau={taus[0][1]:.1f}$ yr)")
    ax_t.plot(lags, w_temp[1, 0], "--^", color="#7570b3", label=f"CH4 Fast ($\\tau={taus[1][0]:.1f}$ yr)")
    ax_t.set_title("Learned Temporal Memory $w(\\Delta t)$")
    ax_t.set_xlabel("Lookback Lag $\\Delta t$ (years)")
    ax_t.set_ylabel("Normalized Kernel Weight")
    ax_t.grid(True, linestyle="--", alpha=0.4)
    ax_t.legend(fontsize=7.5)

    fig.suptitle("Interpretable Nonlocal Spatial Footprints & Causal Thermal Memory Kernels (Ferretti et al., 2026)",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_learned_kernel_footprints.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # PLOT 4: Suri & Sonnewald (2026) Fidelity Verification across 4 Regimes
    # =========================================================================
    # Group assembled feature channels into 5 physical groups:
    # [0:2]=Global Instant (CO2, CH4), [2:6]=Global Memory (tau_1, tau_2),
    # [6:8]=Local Maps (SO2, BC), [8:16]=Nonlocal Spatial Kernels (SO2/BC), [16:20]=Spherical Coords
    group_names = ["Global Instant\n(CO2, CH4)", "Thermal Memory\n(CO2/CH4 Lag)", "Local Aerosol\n(SO2, BC)", "Nonlocal Kernels\n(SO2, BC)", "Spherical\nCoords"]
    group_slices = [slice(0, 2), slice(2, 6), slice(6, 8), slice(8, 16), slice(16, 20)]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), dpi=200, sharey=True)
    regime_titles = {
        "aerosol_hubs": "Aerosol Emission Hubs",
        "arctic_amplification": "Arctic Amplification",
        "tropical_itcz": "Tropical ITCZ Belt",
        "southern_ocean": "Southern Ocean",
    }

    for idx, (r_key, r_title) in enumerate(regime_titles.items()):
        ax = axes[idx]
        r_info = fidelity_report[r_key]
        s_full = np.array(r_info["shap_channel_importance"])
        l_full = np.array(r_info["lrp_channel_importance"])
        s_grp = [float(s_full[sl].sum()) for sl in group_slices]
        l_grp = [float(l_full[sl].sum()) for sl in group_slices]

        y_pos = np.arange(len(group_names))
        ax.barh(y_pos - 0.18, s_grp, height=0.35, color="#2b5c8f", label="Integrated Gradients")
        ax.barh(y_pos + 0.18, l_grp, height=0.35, color="#e7298a", label="LRP-$\\varepsilon$")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(group_names)
        ax.invert_yaxis()
        ax.set_xlabel("Normalized Attribution Share")
        ax.set_title(f"{r_title}\n(Cosine Sim = {r_info['cosine_similarity']:.3f})")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Suri & Sonnewald (2026) Fidelity Verification: SHAP vs. LRP-ε Across Mechanistic Regimes (SSP245)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_suri_sonnewald_fidelity.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # PLOT 5: Full SSP245 Trajectory + Action-Conditioned What-If Counterfactual
    # =========================================================================
    # Predict full SSP245 trajectory (2015-2100)
    g_full_lmlp = nx.apply("g", te_full["global"])
    m_full_lmlp = nx.apply("m", te_full["maps"])
    m_full_k = ((te_full["maps"] - st["m_mean"]) / st["m_std"]).astype(np.float32)
    y_full_dummy = np.zeros((len(te_full["years"]), 2, len(te_full["lat"]), len(te_full["lon"])), dtype=np.float32)
    ds_full_k = ClimateBenchDataset(g_full_k, m_full_k, y_full_dummy)
    ds_full_lmlp = ClimateBenchDataset(g_full_lmlp, m_full_lmlp, y_full_dummy)
    gh_full = torch.from_numpy(gh_full_te)
    with torch.no_grad():
        p_full_kern = best_k_model(ds_full_k.x, g_hist=gh_full).cpu().numpy()
        p_full_lmlp = model_lmlp(ds_full_lmlp.x).cpu().numpy()

    tas_full_true = global_mean(te_full["targets"]["tas"], te_full["lat"])
    tas_full_kern = global_mean(ny.invert("tas", p_full_kern[:, 0]), te_full["lat"])
    tas_full_lmlp = global_mean(ny.invert("tas", p_full_lmlp[:, 0]), te_full["lat"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=200)
    yrs = te_full["years"]
    axes[0].plot(yrs, tas_full_true, "k-", lw=2.2, label="Truth (SSP245)")
    axes[0].plot(yrs, tas_full_lmlp, "--", color="#e41a1c", lw=1.8, label="LocalMLP")
    axes[0].plot(yrs, tas_full_kern, "-", color="#1b9e77", lw=2.0, label="Kernel-Gaussian")
    # NOTE: no predictive uncertainty is estimated yet, so no band is drawn (a fixed +/-0.06 K band was removed).
    axes[0].axvspan(2080, 2100, color="gold", alpha=0.2, label="Scoring Window (2080–2100)")
    axes[0].set_title("Global-Mean tas Trajectory (SSP245)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Global Mean tas Anomaly [K]")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=8, loc="upper left")

    im_w1 = axes[1].pcolormesh(lon_g, lat_g, delta_tas_clean, cmap="Reds", shading="auto")
    axes[1].set_title("What-If: Asian SO2 Phaseout $\\Delta\\mathrm{tas}$ [K]\n(Unmasked Aerosol Warming)")
    axes[1].set_xlabel("Longitude (°E)")
    axes[1].set_ylabel("Latitude (°N)")
    plt.colorbar(im_w1, ax=axes[1], fraction=0.046, pad=0.04)

    im_w2 = axes[2].pcolormesh(lon_g, lat_g, delta_pr_clean, cmap="BrBG", shading="auto")
    axes[2].set_title("What-If: Asian SO2 Phaseout $\\Delta\\mathrm{pr}$ [mm/day]\n(Monsoon Hydrological Recovery)")
    axes[2].set_xlabel("Longitude (°E)")
    axes[2].set_ylabel("Latitude (°N)")
    plt.colorbar(im_w2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Trajectory Projection & Action-Conditioned What-If Counterfactual (Wang et al., 2026)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_whatif_and_trajectories.png", bbox_inches="tight")
    plt.close(fig)

    print(f"[cke] Wrote outcomes.json and 5 figures to {fig_dir}")


if __name__ == "__main__":
    main()

