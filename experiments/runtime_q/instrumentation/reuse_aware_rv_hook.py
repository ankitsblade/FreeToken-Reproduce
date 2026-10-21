"""Runtime-variable-aware Reuse-Aware (Policy C) on the lean v2 harness.

Subclasses reuse_aware_v2_hook.ReuseAwareV2; the original ReuseAware rule (Policy B), paper q*
(fraction / fraction_min1) and built-in q* stay selectable by their own spec kinds and are not
changed. This hook adds spec kind "reuse_aware_rv":

    fetch candidates in decreasing p while   r_j * T_j * v  >=  C(q + 1) - C(q)

    C(j) = max(a + b (h + j) + j t_x,  [m > j] (c0 + (m - j) t_c))     (ms, concurrent branches)
    v    = mean over the observed miss-count distribution of C(m,0) - C(m-1,0) with one more
           GPU-resident expert (ms one future hit saves)

Four independently switchable feature groups (spec flags, default all on):

  exec   measured CPU (c0, t_c) and GPU compute (a, b) costs from the branch timers;
         off: t_c = S / B_H and a = b = 0 (the paper's bandwidth model, GPU compute free)
  xfer   measured PCIe transfer t_x (pcie timer); off: t_x = S / B_P
  cache  cache state and eviction: a fill into a free slot stays (free slots left after it) /
         (measured fills per token) + (measured residency at eviction) tokens instead of 32,
         and r_j = p_j - p_victim(j) (reuse lost by evicting the predicted LRU victim);
         off: T = 32 for a free slot, r_j = p_j
  hist   recent expert-reuse history: p from exponentially decayed routing counts (half-life
         8 tokens) instead of the flat 32-token window; off: the original window

With all four off the rule is the original ReuseAware decision exactly: C becomes
(S / B_H) * max(j / rho, m - j) with rho = B_P / B_H, v = S / B_H, i.e. the original dimensionless
rule scaled by a positive constant (checked by experiments/runtime_q/check_policy_equivalence.py).
Measured costs come from exponentially forgetting least-squares fits (lambda 0.998 per layer
sample) that fall back to the bandwidth priors until 8 samples at 2 distinct x exist. Every input
is observed before the decision; nothing uses future routing.
"""

from __future__ import annotations

import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reuse_aware_v2_hook import (  # noqa: E402
    LEAN_COLUMNS, PRIOR_N, T_FREE, BranchFit, ReuseAwareV2,
)
from runtime_q_hook import Policy  # noqa: E402

LAMBDA = 0.998               # per-sample forgetting of the cost fits
HALF_LIFE_TOK = 8.0          # decayed routing counts (hist group)
GAMMA = 0.5 ** (1.0 / HALF_LIFE_TOK)
T_MAX = 1024.0               # cap on residency estimates (tokens)
RV_COLUMNS = LEAN_COLUMNS + ["q_rv", "ins_rate", "res_age", "tx_ms", "gemm_a", "gemm_b", "c0_rv", "tc_rv"]


class DecayFit(BranchFit):
    """BranchFit with exponential forgetting of old samples."""

    def __init__(self, slope_prior: float, lam: float = LAMBDA) -> None:
        super().__init__(slope_prior)
        self.lam = lam

    def add(self, x: float, y: float) -> None:
        lam = self.lam
        self.n *= lam
        self.sx *= lam
        self.sxx *= lam
        self.sy *= lam
        self.sxy *= lam
        super().add(x, y)


class ReuseRV(Policy):
    name = "reuse_aware_rv"

    def __init__(self, spec: dict, top_k: int, num_experts: int) -> None:
        self.exec = bool(spec.get("exec", True))
        self.xfer = bool(spec.get("xfer", True))
        self.cache = bool(spec.get("cache", True))
        self.hist = bool(spec.get("hist", True))
        self.top_k = top_k
        self.prior = top_k / num_experts

    def costs(self, x: dict) -> tuple[float, float, float, float, float]:
        f, pr = x["_rv_fit"], x["_rv_prior"]
        a, b, c0, tc = (f["a"], f["b"], f["c0"], f["tc"]) if self.exec else (0.0, 0.0, 0.0, pr["tc"])
        tx = f["tx"] if self.xfer else pr["tx"]
        return a, b, c0, tc, tx

    def decide(self, m, x, layer):
        if m <= 0:
            return 0
        a, b, c0, tc, tx = self.costs(x)
        k = self.top_k

        def C(mm, hh, j):
            return max(a + b * (hh + j) + j * tx, c0 + (mm - j) * tc if mm > j else 0.0)

        mc = x["_mcount"]
        tot = sum(mc[1:])
        if tot:
            v = sum(n * (C(mm, k - mm, 0) - C(mm - 1, k - mm + 1, 0)) for mm, n in enumerate(mc) if mm) / tot
        else:
            v = C(1, k - 1, 0) - C(0, k, 0)
        h = int(x["n_hits"])
        pm = x["_p_hist"] if self.hist else x["_p_miss"]
        order = sorted(range(m), key=lambda i: -pm[i])
        p = [pm[i] for i in order]
        pv = x["_pv_hist"] if self.hist else x["_victim_p"]
        free, age = x["free_slots"], x["lru_age_tok"]
        q = 0
        while q < m:
            if free > q:
                t = ((free - q - 1) / max(x["_rv_ins_rate"], 0.05) + x["_rv_res_age"]) if self.cache else T_FREE
            else:
                t = age
            r = p[q] - (pv[q] if self.cache and q < len(pv) else 0.0)
            if r * min(t, T_MAX) * v < C(m, h, q + 1) - C(m, h, q):
                break
            q += 1
        return q


class ReuseAwareRV(ReuseAwareV2):
    def __init__(self, cache) -> None:
        super().__init__(cache)
        self.lean_columns = RV_COLUMNS

    def _reset_state(self) -> None:
        super()._reset_state()
        self.rv_prior = {"tx": self.expert_bytes / (self.bp * 1e6), "tc": self.expert_bytes / (self.bh * 1e6)}
        self.fx = DecayFit(self.rv_prior["tx"])     # pcie ms vs fills
        self.fg = DecayFit(0.0)                     # gpu gemm ms vs GPU-resident experts used
        self.fc = DecayFit(self.rv_prior["tc"])     # cpu task ms vs CPU experts
        self.rv_ins = deque(maxlen=32)
        self.rv_tok_ins = 0
        self.rv_last_tok = -1
        self.rv_res_age = None
        # decayed routing counts per layer: expert -> (value, tok); window mass -> (value, tok)
        self.rv_cnt = [dict() for _ in range(self.L)]
        self.rv_mass = [(0.0, -1) for _ in range(self.L)]
        self._rv_lines()

    def _rv_lines(self) -> None:
        x0, tx = self.fx.line()
        a, b = self.fg.line() if self.fg.n >= PRIOR_N and len(self.fg.xs) >= 2 else (0.0, 0.0)
        c0, tc = self.fc.line()
        self.rv_fit = {"tx": tx, "a": a, "b": max(b, 0.0), "c0": c0, "tc": tc}

    def _p_decayed(self, layer: int, e: int) -> float:
        val, t = self.rv_cnt[layer].get(e, (0.0, self.tok))
        mval, mt = self.rv_mass[layer]
        tok = self.tok
        cnt = val * GAMMA ** (tok - t) if val else 0.0
        mass = mval * GAMMA ** (tok - mt) if mt >= 0 else 0.0
        return (cnt + self.prior * PRIOR_N) / (mass + PRIOR_N)

    def _build(self, spec: dict, seed: int) -> Policy:
        if spec["kind"] == "reuse_aware_rv":
            return ReuseRV(spec, self.top_k, self.E)
        return super()._build(spec, seed)

    def _lean_features(self, layer, uniq, missing, slots) -> dict:
        x = super()._lean_features(layer, uniq, missing, slots)
        if self.tok != self.rv_last_tok:  # token boundary: close the previous token's fill count
            if self.rv_last_tok >= 0:
                self.rv_ins.append(self.rv_tok_ins)
            self.rv_tok_ins, self.rv_last_tok = 0, self.tok
        E = self.E
        pv = []
        for g in x["_victims"]:
            if g < 0:
                pv.append(0.0)
            else:
                vl, ve = divmod(g, E)
                pv.append(self._p_decayed(vl, ve))
        x.update({
            "_rv_fit": self.rv_fit, "_rv_prior": self.rv_prior, "_mcount": list(self.m_count),
            "_rv_ins_rate": sum(self.rv_ins) / len(self.rv_ins) if self.rv_ins else 0.0,
            "_rv_res_age": self.rv_res_age if self.rv_res_age is not None else T_FREE,
            "_p_hist": [self._p_decayed(layer, e) for e in missing], "_pv_hist": pv,
        })
        return x

    def _on_resize(self) -> None:
        super()._on_resize()
        self.rv_res_age = None

    def _after_layer(self, layer, m, n_fetch, n_cpu, n_hits, pcie, gemm, cpu_ms, evicted, ages) -> None:
        if n_fetch:
            self.fx.add(float(n_fetch), pcie)
        self.fg.add(float(n_hits + n_fetch), gemm)
        if n_cpu > 0 and cpu_ms == cpu_ms:
            self.fc.add(float(n_cpu), cpu_ms)
        self.rv_tok_ins += n_fetch
        for a in ages:
            self.rv_res_age = float(a) if self.rv_res_age is None else 0.95 * self.rv_res_age + 0.05 * a
        # decayed routing counts, updated after the decision (past-only for later tokens)
        tok, cnt = self.tok, self.rv_cnt[layer]
        for e in self.cur["uniq"]:
            val, t = cnt.get(e, (0.0, tok))
            cnt[e] = (val * GAMMA ** (tok - t) + 1.0, tok)
        mval, mt = self.rv_mass[layer]
        self.rv_mass[layer] = ((mval * GAMMA ** (tok - mt) if mt >= 0 else 0.0) + 1.0, tok)
        self._rv_lines()

    def _extra_row(self, m: int, x: dict, layer: int) -> dict:
        f = self.rv_fit
        return {"q_rv": ReuseRV({}, self.top_k, self.E).decide(m, x, layer),
                "ins_rate": round(x["_rv_ins_rate"], 4), "res_age": round(x["_rv_res_age"], 3),
                "tx_ms": round(f["tx"], 5), "gemm_a": round(f["a"], 5), "gemm_b": round(f["b"], 5),
                "c0_rv": round(f["c0"], 5), "tc_rv": round(f["tc"], 5)}


def build(cache) -> ReuseAwareRV:
    return ReuseAwareRV(cache)
