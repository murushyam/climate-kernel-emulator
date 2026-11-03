# climate-kernel-emulator

A lightweight, interpretable emulator of the **global forced climate response**: annual-mean surface temperature (`tas`) and precipitation (`pr`), with extremes as an option, under greenhouse-gas and aerosol forcings. It is evaluated on the **ClimateBench** protocol (Watson-Parris et al., 2022, *JAMES*, doi:10.1029/2021MS002954).

Author: M.S. Shyam Sunder ([ORCID 0000-0002-9964-6199](https://orcid.org/0000-0002-9964-6199)). MIT licence.

## Three pieces
1. **Interpretable nonlocal kernels** (after Ferretti et al., QJRMS): parametric integration kernels gather spatial context and short temporal context, then feed a small local nonlinear predictor. This is the core scientific contribution.
2. **Fidelity verification** (after Suri & Sonnewald): SHAP/LRP plus regime and mechanistic checks, to show the model uses real physical drivers rather than shortcuts, including on held-out future scenarios.
3. **Optional probabilistic residual** (inspired by IPSL-AID): a small residual diffusion or ensemble stage, added only where the kernel model leaves structured error.

## Plan and status
| Step | Content | Status |
|---|---|---|
| 1 | Data pipeline, plus pattern-scaling and local-MLP baselines, NRMSE metrics, configs, tests and CI | **In progress** (skeleton done; real-data baseline runs pending) |
| 2 | Kernel layer: parametric nonlocal kernels plus a local predictor | Planned |
| 3 | Experimental hierarchy, ClimateBench NRMSE and physical diagnostics | Planned |
| 4 | Kernel visualisation, fidelity checks (in and out of distribution), optional residual stage | Planned |
| 5 | Paper (JAMES / GMD / QJRMS / EDS) and reproducibility release | Planned |

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
```
- Train scenarios: historical, ssp126, ssp370, ssp585, hist-GHG, hist-aer.
- Test scenario: ssp245.
- Metrics (latitude-weighted): NRMSE spatial, NRMSE global, and NRMSE total = spatial + 5 × global.

## Layout
```
configs/            Hydra: config.yaml, data/, model/, train/
scripts/            download_climatebench.py
src/cke/            data.py, metrics.py, train.py, evaluate.py, baselines/, utils/seed.py
tests/              synthetic-data tests
```

## Citation
See `CITATION.cff`. Please also cite ClimateBench (doi:10.5281/zenodo.7064308).
