"""Numpy-only cost models L_hat(x, q) shared by training (offline) and the runtime hook.

A model artifact is a JSON dict:
  kind      "linear" | "mlp" | "trees"
  features  ordered list of pre-action feature names (x)
  x_mean, x_std   standardization of x (linear / mlp)
  ...       kind-specific parameters
Every model scores a candidate action q for a context x; the policy takes the argmin over
the valid actions q in {0..m}.

Design matrix for one (x, q):
  linear:  [z, qn, qn^2, qf, qn * z]        z = standardized x, qn = q/4, qf = q/max(m,1)
  mlp:     [z, qn, qf]
  trees:   [x, q, qf]
The q-by-feature interactions are what let a *linear* model pick a context-dependent q;
without them the argmin over q would be the same corner for every context.
"""

from __future__ import annotations

import json
import math

import numpy as np

Q_SCALE = 4.0


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def linear_design(z: np.ndarray, q: np.ndarray, m: np.ndarray) -> np.ndarray:
    """z: (n, d) standardized, q/m: (n,). Returns (n, 2d + 3)."""
    qn = (q / Q_SCALE)[:, None]
    qf = (q / np.maximum(m, 1.0))[:, None]
    return np.concatenate([z, qn, qn * qn, qf, qn * z], axis=1)


def mlp_design(z: np.ndarray, q: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.concatenate([z, (q / Q_SCALE)[:, None], (q / np.maximum(m, 1.0))[:, None]], axis=1)


def tree_design(x: np.ndarray, q: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.concatenate([x, q[:, None], (q / np.maximum(m, 1.0))[:, None]], axis=1)


def gelu(a: np.ndarray) -> np.ndarray:
    # tanh form, identical to torch.nn.GELU(approximate="tanh")
    return 0.5 * a * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (a + 0.044715 * a ** 3)))


class CostModel:
    def __init__(self, art: dict) -> None:
        self.art = art
        self.kind = art["kind"]
        self.features = list(art["features"])
        self.mean = np.asarray(art.get("x_mean", [0.0] * len(self.features)), dtype=np.float64)
        self.std = np.asarray(art.get("x_std", [1.0] * len(self.features)), dtype=np.float64)
        if self.kind == "linear":
            self.w = np.asarray(art["w"], dtype=np.float64)
            self.b = float(art["b"])
        elif self.kind == "mlp":
            self.layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64))
                           for w, b in art["layers"]]
            self.y_mean, self.y_std = float(art["y_mean"]), float(art["y_std"])
        elif self.kind == "trees":
            self.base = float(art["base"])
            self.lr = float(art["lr"])
            self.trees = [tuple(np.asarray(t[k]) for k in ("feature", "threshold", "left", "right", "value"))
                          for t in art["trees"]]
        else:
            raise ValueError(self.kind)

    @classmethod
    def load(cls, path: str) -> "CostModel":
        with open(path) as f:
            return cls(json.load(f))

    def n_params(self) -> int:
        if self.kind == "linear":
            return int(self.w.size + 1)
        if self.kind == "mlp":
            return int(sum(w.size + b.size for w, b in self.layers))
        return int(sum(t[0].size for t in self.trees))  # nodes

    def predict(self, x: np.ndarray, q: np.ndarray, m: np.ndarray) -> np.ndarray:
        """x: (n, d) raw features in self.features order."""
        if self.kind == "linear":
            return linear_design(standardize(x, self.mean, self.std), q, m) @ self.w + self.b
        if self.kind == "mlp":
            a = mlp_design(standardize(x, self.mean, self.std), q, m)
            for i, (w, b) in enumerate(self.layers):
                a = a @ w.T + b
                if i < len(self.layers) - 1:
                    a = gelu(a)
            return a[:, 0] * self.y_std + self.y_mean
        d = tree_design(x, q, m)
        out = np.full(d.shape[0], self.base)
        rows = np.arange(d.shape[0])
        for feature, threshold, left, right, value in self.trees:
            node = np.zeros(d.shape[0], dtype=np.int64)
            for _ in range(64):
                f = feature[node]
                leaf = f < 0
                if leaf.all():
                    break
                go_left = d[rows, np.maximum(f, 0)] <= threshold[node]
                node = np.where(leaf, node, np.where(go_left, left[node], right[node]))
            out += self.lr * value[node]
        return out

    def choose(self, xvec: np.ndarray, m: int) -> int:
        """argmin_q L_hat(x, q) over the valid actions 0..m (ties -> smaller q)."""
        if m <= 0:
            return 0
        q = np.arange(m + 1, dtype=np.float64)
        x = np.repeat(xvec[None, :], m + 1, axis=0)
        return int(np.argmin(self.predict(x, q, np.full(m + 1, float(m)))))
