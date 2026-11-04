"""Hydra entrypoint:  python -m cke.train model=local_mlp  |  model=pattern_scaling"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from cke.baselines import LocalMLP, PatternScaling
from cke.data import ClimateBenchDataset, Normaliser, select_years, stack_scenarios
from cke.metrics import all_metrics, global_mean
from cke.utils.seed import seed_everything

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def _device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_splits(cfg):
    d = cfg.data
    kw = dict(variables=list(d.variables), coarsen=d.coarsen, pr_scale=d.pr_scale)
    tr = stack_scenarios(d.root, d.train_scenarios, **kw)
    te = stack_scenarios(d.root, d.test_scenarios, **kw)
    w = cfg.get("eval", {}).get("window", None)
    if w is not None:  # ClimateBench protocol: score held-out scenario on 2080-2100 only
        te = select_years(te, w[0], w[1])
    return tr, te


def run_pattern_scaling(cfg, tr, te):
    res, preds = {}, {}
    tg = global_mean(tr["targets"]["tas"], tr["lat"])
    for v in cfg.data.variables:
        m = PatternScaling().fit(tr["global"], tr["targets"][v], tg)
        preds[v] = m.predict(te["global"])
        res[v] = all_metrics(te["targets"][v], preds[v], te["lat"])
    return res, preds


def run_local_mlp(cfg, tr, te):
    variables = list(cfg.data.variables)
    xs = {"g": tr["global"], "m": tr["maps"]}
    nx = Normaliser.fit(xs)
    ny = Normaliser.fit(tr["targets"])

    def prep(split):
        y = np.stack([ny.apply(v, split["targets"][v]) for v in variables], 1)
        return ClimateBenchDataset(nx.apply("g", split["global"]), nx.apply("m", split["maps"]), y)

    ds_tr, ds_te = prep(tr), prep(te)
    dev = _device(cfg.device)
    model = LocalMLP(ds_tr.x.shape[1], len(variables), tuple(cfg.model.hidden), cfg.model.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loader = torch.utils.data.DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(cfg.seed))
    for ep in range(cfg.train.epochs):
        tot = 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(x)
        if ep % 10 == 0 or ep == cfg.train.epochs - 1:
            print(f"epoch {ep:3d}  train_mse {tot / len(ds_tr):.4f}")
    model.eval()
    with torch.no_grad():
        p = model(ds_te.x.to(dev)).cpu().numpy()
    preds = {v: ny.invert(v, p[:, i]) for i, v in enumerate(variables)}
    res = {v: all_metrics(te["targets"][v], preds[v], te["lat"]) for v in variables}
    return res, preds, model


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    tr, te = load_splits(cfg)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    if cfg.model.name == "pattern_scaling":
        res, _ = run_pattern_scaling(cfg, tr, te)
    elif cfg.model.name == "local_mlp":
        res, _, model = run_local_mlp(cfg, tr, te)
        torch.save(model.state_dict(), out / "model.pt")
    else:
        raise ValueError(cfg.model.name)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
