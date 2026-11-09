---
name: cke-benchmark
description: >-
  Run, evaluate, and audit the climate-kernel-emulator on the ClimateBench v1.0
  protocol. Use this skill when training pattern_scaling, local_mlp, or kernel_mlp
  models, inspecting learned nonlocal spatial/temporal integration kernels,
  running Suri-Sonnewald SHAP/LRP fidelity verification across mechanistic regimes,
  or running zero-shot probabilistic forecasts with Timestral.
---

# Climate Kernel Emulator (`cke`) Workflow Skill

## 1. Environment & Test Verification

1. Run the unit and integration test suite before and after any model changes:
   ```bash
   pytest -q tests/
   ```
2. Verify that all five layered modules (`kernels`, `interp`, `temporal`, `residual_diffusion`, `whatif`) pass synthetic-data tests within `< 5s`.

## 2. Training & Benchmarking (`PatternScaling` -> `LocalMLP` -> `KernelEmulator`)

1. Download ClimateBench v1.0.0 (`doi:10.5281/zenodo.7064308`) if not present:
   ```bash
   python scripts/download_climatebench.py --out data/climatebench
   ```
2. Train the hierarchy of models on the coarsened ($48 \times 72$) grid:
   ```bash
   python -m cke.train model=pattern_scaling
   python -m cke.train model=local_mlp data.coarsen=2 train.epochs=50
   python -m cke.train model=kernel_mlp model.family=gaussian data.coarsen=2 train.epochs=50
   ```
3. Compare $\text{NRMSE}_s$, $\text{NRMSE}_g$, and $\text{NRMSE}_t$ on held-out `ssp245` (`2080–2100`):
   ```bash
   python -m cke.evaluate outputs/*/metrics.json
   ```

## 3. Inspecting Learned Nonlocal Kernels & Fidelity Verification

1. Inspect `outputs/kernel_mlp/kernels.json` for learned spatial length scales (`sigma_lat_km`, `sigma_lon_km`), upwind advection shifts (`mu_lat_deg`, `mu_lon_deg`), and causal ocean thermal memory timescales (`tau_years`).
2. Run the Suri & Sonnewald (2026) fidelity audit via `cke.interp`:
   - Compute `lrp_local_mlp` and `integrated_gradients_shap` on the assembled feature stack.
   - Partition into the 4 mechanistic regimes via `define_mechanistic_regimes`.
   - Verify attribution agreement via `verify_regime_fidelity`.

## 4. Literature & Equations Reference

For mathematical formulations and BibTeX citations of the underlying literature (Watson-Parris et al., 2022; Ferretti et al., 2026; Suri & Sonnewald, 2026; Petralia et al., 2026; Kingston et al., 2026; Wang et al., 2026; Bouabid et al., 2024), read [literature.md](./references/literature.md).

