# Workspace Rules: `climate-kernel-emulator`

## 1. Hardware & Memory Ceiling
- Every model, training loop, and diagnostic script in this repository **must** execute within a **2 GB GPU VRAM** and **32 GB system RAM** budget (or on CPU).
- Keep default spatial resolution at `data.coarsen=2` ($48 \times 72$ grid) and keep parameter counts for `LocalMLP` and `KernelEmulator` under $5 \times 10^4$ parameters (and $< 2 \times 10^6$ for any optional residual denoiser).

## 2. ClimateBench Protocol & Zero Data Leakage
- **Training scenarios**: `historical`, `ssp126`, `ssp370`, `ssp585`, `hist-GHG`, `hist-aer`.
- **Held-out test scenario**: `ssp245`, scored strictly on the `2080–2100` window (`eval.window: [2080, 2100]`) unless full-trajectory diagnostics are explicitly requested.
- **Normalisation & EOF fitting**: Always fit `Normaliser` and `eof_reduce` bases strictly on training scenarios (`tr`), never on `ssp245`.
- **Scenario boundary integrity**: When constructing causal temporal lookback windows (`build_temporal_history` in `src/cke/kernels/parametric.py`), always pass `split["scenario"]` so lag indices clamp at the start of each scenario and never leak across concatenated scenarios.

## 3. Evaluation Metrics & Physical Fidelity
- Report all three latitude-weighted ClimateBench metrics from `src/cke/metrics.py`:
  - $\text{NRMSE}_s$ (spatial pattern error)
  - $\text{NRMSE}_g$ (global-mean trajectory error)
  - $\text{NRMSE}_t = \text{NRMSE}_s + 5\,\text{NRMSE}_g$ (total score)
- When evaluating `KernelEmulator`, export human-interpretable kernel length scales and advection shifts (`kernels.json`) and verify mechanistic attribution consistency across `aerosol_hubs`, `arctic_amplification`, `tropical_itcz`, and `southern_ocean` using `src/cke/interp/fidelity.py` (Suri & Sonnewald, 2026).

## 4. Optional Module Stubs & Config Defaults
- Keep `enabled: false` by default in `configs/temporal/timestral.yaml`, `configs/residual/diffusion.yaml`, and `configs/whatif/placeholder.yaml`.
- Preserve `NotImplementedError` guards when `temporal.forecast_series`, `residual_diffusion.fit_residual`, or `whatif.apply_action` are called with `None` or disabled configs so `tests/test_window_stubs.py` always passes.

