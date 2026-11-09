# Project plan (definitive, 9 Nov 2026)

**Goal.** A lightweight, interpretable emulator of the global forced climate response (annual-mean `tas` and `pr`, extremes optional) under greenhouse-gas and aerosol forcings, evaluated on the ClimateBench protocol. Hard budget: 2 GB GPU + 32 GB RAM. Deliverables: a clean, reproducible codebase and a high-impact paper.

## Layered architecture (priority order)
| # | Layer | Source of idea | Role | Code | Status |
|---|---|---|---|---|---|
| 1 | Interpretable nonlocal kernels | Ferretti et al. (2026, *EDS*, doi:10.1017/eds.2026.10056) | Parametric integration kernels aggregate spatial (and short temporal) context into a small local nonlinear predictor. **Primary contribution, implement fully first.** | `cke/kernels/` (Step 2) | Implemented & benchmarked (`gaussian`, `mog`, `tophat`, `exponential` + `TemporalExponentialKernel`) |
| 2 | Fidelity verification | Suri & Sonnewald (2026, *JGR-MLC*, doi:10.1029/2025JH001211) | SHAP / LRP-$\varepsilon$ plus mechanistic-regime analysis, in-distribution and held-out scenarios | `cke/interp/` (Step 4) | Implemented & verified (`lrp_local_mlp`, `integrated_gradients_shap`, 4 regimes, $\cos\theta=0.856\text{--}0.952$) |
| 3 | Temporal / probabilistic module | Timestral (Petralia et al., 2026, arXiv:2610.37585) | Small checkpoints (5M / 28M, CPU) on global-mean and regional aggregate series, forcings as known-future covariates (`known_covariates`), quantile forecasts | `cke/temporal/`, `configs/temporal/timestral.yaml` | Implemented (disabled by default; enable via `temporal.enabled=true`) |
| 4 | Fine-scale residual (optional) | IPSL-AID (Kingston et al., 2026) / CorrDiff (Mardani et al., 2024) | Small residual diffusion, **only** if structured residual error stays large after Layer 1 | `cke/residual_diffusion/`, `configs/residual/diffusion.yaml` | Implemented (disabled by default; enable via `residual.enabled=true`) |
| 5 | Action-conditioned what-if (future) | Earth System World Model (Wang et al., 2026, arXiv:2609.08855) | Perturb land cover, fire or forcing pathway and re-run. Not needed for paper 1 | `cke/whatif/`, `configs/whatif/placeholder.yaml` | Implemented & demonstrated (`apply_action` regional/global perturbations) |

## Execution plan
| Step | Weeks | Content | Status |
|---|---|---|---|
| 1 | 1–2 | ClimateBench → annual `tas`, `pr` on a coarse grid; local-MLP and pattern-scaling baselines; Hydra configs, fixed seeds, pytest, CI; **extract global-mean / regional aggregate series** (`python -m cke.timeseries`) | Done |
| 2 | 2–3 | Kernel layer: parametric nonlocal kernels; hierarchy local baseline → kernel + local predictor; stay within 2 GB GPU | Done ($1{,}822\text{--}1{,}878$ params) |
| 3 | 1–2 | ClimateBench NRMSE (spatial, global, total = spatial + 5 × global) on held-out ssp245, **2080–2100 window**; physical diagnostics; Timestral (CPU) on global-mean series with forcings as covariates, compare probabilistic skill (CRPS / quantile loss) | Done (`Kernel-MoG` `pr` $\text{NRMSE}_t = 2.619$; `Kernel-Tophat` `tas` $\text{NRMSE}_s = 0.067$) |
| 4 | 1–2 | Visualise learned kernels; Suri–Sonnewald SHAP/LRP fidelity checks; check that ensembles improve reliability; optional residual diffusion | Done (`docs/figures/fig3_learned_kernel_footprints.png`, `fig4_suri_sonnewald_fidelity.png`) |
| 5 | 2–3 | Paper (JAMES / GMD / QJRMS / Environmental Data Science); open code + configs + reproducibility note; short what-if demonstration or discussion; update CV, site, GitHub, LinkedIn | Code, figures & `docs/outcomes.json` complete |

## Timestral facts (checked 9 Nov 2026)
- Code: `pip install git+https://github.com/mistralai/timestral.git` (Apache-2.0; Python ≥ 3.10, torch ≥ 2.4). Import: `from timestral import TimestralForecaster`.
- Weights (Hugging Face, Modified MIT): `mistralai/Timestral-5M` (21 MB), `mistralai/Timestral-28M` (111 MB); also 123M and 209M (too big for our CPU budget).
- Input `(series, variates, time)`, NaN = missing; outputs 21 quantiles (0.01…0.99). Supports past-only and known-future covariates via `known_covariates` of shape `(B, C_known, T_context + prediction_length)`.
- Caveat: annual ClimateBench series are short (~165 historical + 86 scenario years) relative to the model's 32-step patches. Treat Timestral as a zero-shot probabilistic baseline, not the main model.
