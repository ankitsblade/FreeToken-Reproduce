"""Shared training / offline-evaluation helpers for the runtime-q study."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paths import ROOT, study_dir
RQ = study_dir("runtime_q")
sys.path.insert(0, str(ROOT / "experiments/runtime_q/instrumentation"))
from policy_models import CostModel, linear_design, mlp_design, tree_design  # noqa: E402
from schema import GROUPS  # noqa: E402

MAX_Q = 4


def load_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_pickle(RQ / "data" / "clean" / "dataset.pkl")
    report = json.loads((RQ / "data" / "clean" / "feature_report.json").read_text())
    return df, report


def group_features(report: dict, groups: list[str]) -> list[str]:
    out: list[str] = []
    for g in groups:
        out += [f for f in report["groups_kept"].get(g, []) if f not in out]
    return out


def training_frames(df: pd.DataFrame, target: str, split: str = "split_run"):
    d = df[df["randomized"] & df[target].notna()]
    return d[d[split] == "train"], d[d[split] == "val"], d[d[split] == "test"]


def _xqm(d: pd.DataFrame, feats: list[str]):
    return (d[feats].to_numpy(dtype=np.float64), d["n_fetch"].to_numpy(dtype=np.float64),
            d["m"].to_numpy(dtype=np.float64))


def fit(kind: str, tr: pd.DataFrame, va: pd.DataFrame, feats: list[str], target: str,
        seed: int = 0, **hp) -> dict:
    x, q, m = _xqm(tr, feats)
    y = tr[target].to_numpy(dtype=np.float64)
    mean = x.mean(0) if len(feats) else np.zeros(0)
    std = x.std(0) if len(feats) else np.ones(0)
    std = np.where(std < 1e-9, 1.0, std)
    art = {"kind": None, "features": feats, "x_mean": mean.tolist(), "x_std": std.tolist(),
           "target": target, "n_train": int(len(tr)), "hp": hp}
    if kind in ("linear", "ridge"):
        from sklearn.linear_model import Ridge
        alpha = hp.get("alpha", 1e-6 if kind == "linear" else 10.0)
        model = Ridge(alpha=alpha).fit(linear_design((x - mean) / std, q, m), y)
        art.update(kind="linear", w=model.coef_.tolist(), b=float(model.intercept_), label=kind)
    elif kind in ("tree", "gbm"):
        d = tree_design(x, q, m)
        if kind == "tree":
            from sklearn.tree import DecisionTreeRegressor
            est = [DecisionTreeRegressor(max_depth=hp.get("max_depth", 7),
                                         min_samples_leaf=hp.get("min_leaf", 100),
                                         random_state=seed).fit(d, y)]
            base, lr = 0.0, 1.0
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            g = GradientBoostingRegressor(n_estimators=hp.get("n_estimators", 80),
                                          max_depth=hp.get("max_depth", 3),
                                          learning_rate=hp.get("lr", 0.1), subsample=0.5,
                                          min_samples_leaf=50, random_state=seed).fit(d, y)
            est, base, lr = [e[0] for e in g.estimators_], float(g.init_.constant_[0][0]), g.learning_rate
        trees = []
        for e in est:
            t = e.tree_
            trees.append({"feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                          "left": t.children_left.tolist(), "right": t.children_right.tolist(),
                          "value": t.value[:, 0, 0].tolist()})
        art.update(kind="trees", base=base, lr=lr, trees=trees, label=kind)
    elif kind == "mlp":
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(4)
        hidden = hp.get("hidden", [64, 64])
        y_mean, y_std = float(y.mean()), float(y.std() or 1.0)
        dt = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
        xt, yt = dt(mlp_design((x - mean) / std, q, m)), dt((y - y_mean) / y_std)
        xv, qv, mv = _xqm(va, feats)
        xvt = dt(mlp_design((xv - mean) / std, qv, mv))
        yvt = dt((va[target].to_numpy(dtype=np.float64) - y_mean) / y_std)
        dims = [xt.shape[1], *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.GELU(approximate="tanh"))
        net = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=hp.get("lr", 2e-3), weight_decay=1e-5)
        best, best_state, bad = float("inf"), None, 0
        gen = torch.Generator().manual_seed(seed)
        for _ in range(hp.get("epochs", 60)):
            perm = torch.randperm(len(xt), generator=gen)
            for i in range(0, len(xt), 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad()
                torch.nn.functional.mse_loss(net(xt[idx])[:, 0], yt[idx]).backward()
                opt.step()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(net(xvt)[:, 0], yvt).item() if len(xvt) else 0.0
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 6:
                    break
        if best_state:
            net.load_state_dict(best_state)
        lin = [l for l in net if isinstance(l, torch.nn.Linear)]
        art.update(kind="mlp", y_mean=y_mean, y_std=y_std, hidden=hidden, label="mlp",
                   layers=[[l.weight.detach().numpy().tolist(), l.bias.detach().numpy().tolist()]
                           for l in lin])
    else:
        raise ValueError(kind)
    return art


def choose_q(model: CostModel, d: pd.DataFrame) -> np.ndarray:
    x = d[model.features].to_numpy(dtype=np.float64)
    m = d["m"].to_numpy(dtype=np.float64)
    preds = np.full((len(d), MAX_Q + 1), np.inf)
    for q in range(MAX_Q + 1):
        ok = m >= q
        if ok.any():
            preds[ok, q] = model.predict(x[ok], np.full(int(ok.sum()), float(q)), m[ok])
    return preds.argmin(1)


def matched_value(d: pd.DataFrame, q_pi: np.ndarray, target: str) -> tuple[float, int]:
    """Self-normalized IPS value of a policy on UNIFORM-random logged rows.

    Unbiased for 'take pi's action now, then the logging policy afterwards' -- a one-step
    deviation value, not the value of running pi for a whole decode."""
    hit = d["n_fetch"].to_numpy() == q_pi
    w = (d["m"].to_numpy() + 1.0) * hit
    return float((w * d[target].to_numpy()).sum() / max(w.sum(), 1e-9)), int(hit.sum())


def evaluate(art: dict, te: pd.DataFrame, uniform: pd.DataFrame, target: str) -> dict:
    model = CostModel(art)
    x, q, m = _xqm(te, model.features)
    y = te[target].to_numpy(dtype=np.float64)
    pred = model.predict(x, q, m)
    err = pred - y
    q_pi = choose_q(model, uniform)
    v, n = matched_value(uniform, q_pi, target)
    xv = uniform[model.features].to_numpy(dtype=np.float64)[:2000]
    mv = uniform["m"].to_numpy()[:2000]
    t0 = time.perf_counter()
    for i in range(len(xv)):
        model.choose(xv[i], int(mv[i]))
    us = 1e6 * (time.perf_counter() - t0) / max(1, len(xv))
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "ips_value": v, "ips_matches": n, "mean_q": float(q_pi.mean()),
            "frac_q0": float((q_pi == 0).mean()), "infer_us": us, "n_params": model.n_params()}


def save_artifact(art: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art))
