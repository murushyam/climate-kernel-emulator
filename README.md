# climate-kernel-emulator

A lightweight, interpretable emulator of the **global forced climate response**: annual-mean surface temperature (`tas`) and precipitation (`pr`), with extremes as an option, under greenhouse-gas and aerosol forcings. It is evaluated on the **ClimateBench** protocol (Watson-Parris et al., 2022, *JAMES*, doi:10.1029/2021MS002954).

Author: M.S. Shyam Sunder ([ORCID 0000-0002-9964-6199](https://orcid.org/0000-0002-9964-6199)). MIT licence.

## Architecture (priority order)
1. **Interpretable nonlocal kernels** (after Ferretti et al.): parametric integration kernels aggregate spatial (and short temporal) context into a small local nonlinear predictor. Core contribution.
2. **Fidelity verification** (after Suri & Sonnewald): SHAP/LRP plus mechanistic-regime checks, in-distribution and on held-out scenarios.
3. **Temporal / probabilistic module**: small Timestral checkpoints (5M / 28M, CPU) on global-mean and regional series, forcings as known-future covariates, quantile forecasts. *Stub.*
4. **Optional residual diffusion** (IPSL-AID inspired): only if structured residual error remains. *Stub, disabled.*
5. **Action-conditioned what-if** (Earth System World Model inspired): future work, not paper 1. *Placeholder.*

Full plan and status: [`docs/PLAN.md`](docs/PLAN.md).

## Status
| Step | Content | Status |
|---|---|---|
| 1 | Data, pattern-scaling + local-MLP baselines, Hydra, seeds, tests, CI, aggregate time series | Code done; real-data baseline runs pending |
| 2 | Kernel layer + local predictor | Next |
| 3 | NRMSE on held-out ssp245 (2080–2100), physical diagnostics, Timestral comparison | Metrics done; Timestral stub |
| 4 | Kernel visualisation, fidelity checks, optional residual | Planned |
| 5 | Paper, reproducibility note, what-if section, profile update | Planned |

## Hardware constraints
Everything must run on a **2 GB GPU with 32 GB RAM**, or on CPU:
- Coarse grid via `data.coarsen` (2 turns the native 96×144 grid into 48×72).
- Small models (the local MLP has hidden width 32).
- Annual means only, ensemble-averaged, so the full training set fits in RAM.

## Setup
```bash
conda create -n cke python=3.11 -y && conda activate cke
# CPU:  pip install torch --index-url https://download.pytorch.org/whl/cpu
# GPU:  pip install torch            (pick the CUDA build for your driver)
pip install -e .[dev]            # add ,interp for SHAP
pytest -q

# Data: ClimateBench v1.0.0, Zenodo record 7064308 (~0.9 GB, md5-checked, skips files already extracted)
python scripts/download_climatebench.py --dry-run
python scripts/download_climatebench.py --out data/climatebench
```
If the archives extract into a subfolder, set `data.root` to the folder that holds `inputs_*.nc` and `outputs_*.nc`.

## Run
```bash
python -m cke.train model=pattern_scaling
python -m cke.train model=local_mlp data.coarsen=2 train.epochs=50
python -m cke.train model=local_mlp seed=1 device=cpu
python -m cke.evaluate outputs/*/metrics.json
python -m cke.timeseries                          # -> data/timeseries/aggregate_series.csv
python -m cke.timeseries timeseries.format=netcdf timeseries.land_mask=path/to/landfrac.nc
```
- Train scenarios: historical, ssp126, ssp370, ssp585, hist-GHG, hist-aer.
- Test scenario: ssp245.
- Scoring window: 2080–2100 (`eval.window`; set `eval.window=null` for all years).
- Metrics (latitude-weighted): NRMSE spatial, NRMSE global, and NRMSE total = spatial + 5 × global.

## Layout
```
configs/            Hydra: config.yaml, data/, model/, train/, temporal/, residual/, whatif/
docs/PLAN.md        definitive plan
scripts/            download_climatebench.py
src/cke/            data.py, metrics.py, train.py, evaluate.py, timeseries.py, baselines/, utils/,
                    temporal/ (stub), residual_diffusion/ (stub), whatif/ (placeholder)
tests/              synthetic-data tests
```

## Citation
See `CITATION.cff`. Please also cite ClimateBench (doi:10.5281/zenodo.7064308).
