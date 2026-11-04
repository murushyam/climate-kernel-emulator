# Project plan (definitive, 4 Nov 2026)

**Goal.** A lightweight, interpretable emulator of the global forced climate response (annual-mean `tas` and `pr`, extremes optional) under greenhouse-gas and aerosol forcings, evaluated on the ClimateBench protocol. Hard budget: 2 GB GPU + 32 GB RAM. Deliverables: a clean, reproducible codebase and a high-impact paper.

## Layered architecture (priority order)
| # | Layer | Source of idea | Role | Code | Status |
|---|---|---|---|---|---|
| 1 | Interpretable nonlocal kernels | Ferretti et al. | Parametric integration kernels aggregate spatial (and short temporal) context into a small local nonlinear predictor. **Primary contribution, implement fully first.** | `cke/kernels/` (Step 2) | Planned |
| 2 | Fidelity verification | Suri & Sonnewald | SHAP / LRP plus mechanistic-regime analysis, in-distribution and held-out scenarios | `cke/interp/` (Step 4) | Planned |
| 3 | Temporal / probabilistic module | Timestral (time-series foundation models) | Small checkpoints (5M / 28M, CPU) on global-mean and regional aggregate series, forcings as known-future covariates, quantile forecasts | `cke/temporal/`, `configs/temporal/timestral.yaml` | Stub (disabled) |
| 4 | Fine-scale residual (optional) | IPSL-AID | Small residual diffusion, **only** if structured residual error stays large after Layer 1 | `cke/residual_diffusion/`, `configs/residual/diffusion.yaml` | Stub (disabled) |
| 5 | Action-conditioned what-if (future) | Earth System World Model | Perturb land cover, fire or forcing pathway and re-run. Not needed for paper 1 | `cke/whatif/`, `configs/whatif/placeholder.yaml` | Placeholder |

## Execution plan
| Step | Weeks | Content | Status |
|---|---|---|---|
| 1 | 1–2 | ClimateBench → annual `tas`, `pr` on a coarse grid; local-MLP and pattern-scaling baselines; Hydra configs, fixed seeds, pytest, CI; **extract global-mean / regional aggregate series** (`python -m cke.timeseries`) | Code done; real-data baseline runs pending (user) |
| 2 | 2–3 | Kernel layer: parametric nonlocal kernels; hierarchy local baseline → kernel + local predictor; stay within 2 GB GPU | Next |
| 3 | 1–2 | ClimateBench NRMSE (spatial, global, total = spatial + 5 × global) on held-out ssp245, **2080–2100 window**; physical diagnostics; Timestral (CPU) on global-mean series with forcings as covariates, compare probabilistic skill (CRPS / quantile loss) | Metrics + window done; Timestral stub |
| 4 | 1–2 | Visualise learned kernels; Suri–Sonnewald SHAP/LRP fidelity checks; check that ensembles improve reliability; optional residual diffusion | Planned |
| 5 | 2–3 | Paper (JAMES / GMD / QJRMS / Environmental Data Science); open code + configs + reproducibility note; short what-if demonstration or discussion; update CV, site, GitHub, LinkedIn | Planned |

## Timestral facts (checked 4 Nov 2026)
- Code: `pip install git+https://github.com/mistralai/timestral.git` (Apache-2.0; Python ≥ 3.10, torch ≥ 2.4). Import: `from timestral import TimestralForecaster`.
- Weights (Hugging Face, Modified MIT): `mistralai/Timestral-5M` (21 MB), `mistralai/Timestral-28M` (111 MB); also 123M and 209M (too big for our CPU budget).
- Input `(series, variates, time)`, NaN = missing; outputs 21 quantiles (0.01…0.99). Supports past-only and known-future covariates.
- TODO before Step 3: confirm the exact covariate keyword in the package README.
- Caveat: annual ClimateBench series are short (~165 historical + 86 scenario years) relative to the model's 32-step patches. Treat Timestral as a zero-shot probabilistic baseline, not the main model.
