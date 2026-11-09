"""Hydra entrypoint:  python -m cke.train model=local_mlp  |  model=pattern_scaling  |  model=kernel_mlp"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from cke.baselines import LocalMLP, PatternScaling
from cke.data import ClimateBenchDataset, Normaliser, select_years, stack_scenarios
from cke.kernels import KernelEmulator, build_temporal_history
from cke.metrics import all_metrics, global_mean, lat_weights
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


def run_kernel_mlp(cfg, tr, te):
    variables = list(cfg.data.variables)
    g_mean = tr["global"].mean(axis=0, keepdims=True)
    g_std = tr["global"].std(axis=0, keepdims=True) + 1e-8
    m_mean = tr["maps"].mean(axis=(0, 2, 3), keepdims=True)
    m_std = tr["maps"].std(axis=(0, 2, 3), keepdims=True) + 1e-8
    ny = Normaliser.fit(tr["targets"])
    win_L = int(cfg.model.get("temporal_window", 10))

    def prep(split):
        g_norm = ((split["global"] - g_mean) / g_std).astype(np.float32)
        m_norm = ((split["maps"] - m_mean) / m_std).astype(np.float32)
        y = np.stack([ny.apply(v, split["targets"][v]) for v in variables], 1).astype(np.float32)
        ds = ClimateBenchDataset(g_norm, m_norm, y)
        if "_full_global" in split and "_window_mask" in split:
            g_full_norm = ((split["_full_global"] - g_mean) / g_std).astype(np.float32)
            gh_full = build_temporal_history(g_full_norm, split.get("_full_scenario"), window=win_L)
            gh = gh_full[split["_window_mask"]]
        else:
            gh = build_temporal_history(g_norm, split.get("scenario"), window=win_L)
        g_hist = torch.from_numpy(gh)
        return torch.utils.data.TensorDataset(ds.x, g_hist, ds.y), g_norm, y

    (ds_tr, g_tr_norm, y_tr_norm), (ds_te, _, _) = prep(tr), prep(te)
    dev = _device(cfg.device)
    mcfg = cfg.model
    model = KernelEmulator(
        n_global=tr["global"].shape[1],
        n_maps=tr["maps"].shape[1],
        out_channels=len(variables),
        lat=tr["lat"],
        lon=tr["lon"],
        n_spatial_kernels=mcfg.get("n_spatial_kernels", 4),
        family=mcfg.get("family", "gaussian"),
        anisotropic=mcfg.get("anisotropic", True),
        n_components=mcfg.get("n_components", 2),
        n_temporal_kernels=mcfg.get("n_temporal_kernels", 2),
        temporal_window=win_L,
        include_local_maps=mcfg.get("include_local_maps", True),
        coord_features=mcfg.get("coord_features", True),
        hidden=tuple(mcfg.get("hidden", [32, 32])),
        dropout=mcfg.get("dropout", 0.0),
    ).to(dev)
    if mcfg.get("use_spatial_anchor", True):
        model.fit_spatial_anchor(g_tr_norm, y_tr_norm)
    model.input_stats = {"g_mean": g_mean, "g_std": g_std, "m_mean": m_mean, "m_std": m_std}
    w_lat = torch.as_tensor(lat_weights(tr["lat"]), dtype=torch.float32, device=dev).view(1, 1, -1, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    loader = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    for ep in range(cfg.train.epochs):
        tot = 0.0
        for x, gh, y in loader:
            x, gh, y = x.to(dev), gh.to(dev), y.to(dev)
            pred = model(x, g_hist=gh)
            diff = pred - y
            loss_grid = ((diff ** 2) * w_lat).mean()
            loss_glob = ((diff * w_lat).mean(dim=(-2, -1)) ** 2).mean()
            loss = loss_grid + 2.0 * loss_glob
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        if ep % 10 == 0 or ep == cfg.train.epochs - 1:
            print(f"epoch {ep:3d}  train_wmse {tot / len(ds_tr):.4f}")
    model.eval()
    with torch.no_grad():
        x_te, gh_te, _ = ds_te.tensors
        p = model(x_te.to(dev), g_hist=gh_te.to(dev)).cpu().numpy()
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
    elif cfg.model.name == "kernel_mlp":
        res, _, model = run_kernel_mlp(cfg, tr, te)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "kernels.json").write_text(json.dumps(model.export_kernel_summary(), indent=2))
    else:
        raise ValueError(cfg.model.name)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
