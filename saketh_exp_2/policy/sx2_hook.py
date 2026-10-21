"""saketh_exp_2 policy hook: paper q*, built-in q*, original Reuse-Aware (B) and B + V1/V2/V3.

Loaded by the engine through FREETOKEN_MOE_RUNTIME_Q_MODULE (experiment-hooks.patch). It
subclasses the frozen lean harness ``experiments/runtime_q/instrumentation/reuse_aware_v2_hook``
and leaves every frozen file unchanged. Policies by control-file spec kind:

  fraction_min1   A   paper q* (Sec. 3.2 Eq. 4, at least one fill), FreeToken balance rounding
  fraction        A0  the same without the minimum fill (diagnostic)
  native          E   built-in q*: the engine kernel with its own profile fraction
  reuse_aware     B   runtime_q_hook.ReuseAware, the original rule, unchanged
  reuse_aware_v   B + V1/V2/V3 switches (reuse_aware_v2_hook.ReuseVariant; all off == B)

Changes relative to the frozen lean harness, all policy-independent:
  * Victim prediction reproduces the engine kernel exactly: sequential argmin(usage) over all
    slots, slots owned by this layer's routed experts excluded, ties -> lowest slot index. The
    frozen harness used argpartition, which picks arbitrary slots among usage ties.
  * Every decision is checked against the kernel's real evictions, and per-episode counters
    (victim matches, V1 fit fallbacks, V1 cost-prediction error, V2 calibration samples, branch
    times) are appended to sx2_stats_<pid>.jsonl every 32 tokens for every episode.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_FROZEN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "runtime_q", "instrumentation"
)
sys.path.insert(0, os.path.abspath(_FROZEN))
from reuse_aware_v2_hook import PRIOR_N, LEAN_COLUMNS, BranchFit, ReuseAwareV2  # noqa: E402

STATS_EVERY_TOK = 32
INT64_MAX = np.iinfo(np.int64).max
SX2_COLUMNS = LEAN_COLUMNS + ["victim_match", "fit_g_ok", "fit_c_ok", "cost_pred_ms", "cost_meas_ms", "cal_n"]


def engine_victims(usage: np.ndarray, own_slots: list, k: int) -> np.ndarray:
    """Slots the hybrid ensure kernel evicts for k fills, in eviction order.

    Kernel (offload_kernels._ensure_experts_hybrid_kernel, phase 2): owner-active slots get
    usage INT64_MAX, then k times victim = argmin(u) (Triton tie_break_left -> lowest index)
    and u[victim] = INT64_MAX. That is the k smallest (usage, slot) pairs.
    """
    cap = usage.shape[0]
    k = min(k, cap)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    u = usage.astype(np.int64, copy=True)
    if own_slots:
        u[own_slots] = INT64_MAX
    if k < cap:
        kth = np.partition(u, k - 1)[k - 1]
        less = np.flatnonzero(u < kth)
        idx = np.concatenate([less, np.flatnonzero(u == kth)[: k - less.size]])
    else:
        idx = np.arange(cap)
    return idx[np.lexsort((idx, u[idx]))][:k]


def fit_ok(fit: BranchFit) -> bool:
    """True when BranchFit.line() uses the fitted line rather than its bandwidth prior."""
    if fit.n < PRIOR_N or len(fit.xs) < 2:
        return False
    den = fit.n * fit.sxx - fit.sx * fit.sx
    return den > 0 and (fit.n * fit.sxy - fit.sx * fit.sy) / den > 0


class Sx2Hook(ReuseAwareV2):
    def __init__(self, cache) -> None:
        super().__init__(cache)
        self.lean_columns = SX2_COLUMNS
        self.stats_path = None
        root = os.environ.get("EXPERIMENT_OUTPUT_ROOT")
        if root:
            self.stats_path = os.path.join(root, "runtime_q", "results", f"sx2_stats_{os.getpid()}.jsonl")
        self.stats_key = None
        self.stats: dict = {}

    # -- state -----------------------------------------------------------------
    def _reset_state(self) -> None:
        # A new episode resets alpha and the fits: write the old episode's last row first.
        if getattr(self, "stats", None):
            self._flush_stats()
            self.stats = {}
        super()._reset_state()
        self.cal_n = 0

    def _check_control(self) -> None:
        super()._check_control()
        key = (self.ctrl.get("episode"), self.ctrl.get("phase"))
        if key != self.stats_key:
            self._flush_stats()
            self.stats_key = key
            self.stats = {k: 0.0 for k in (
                "decisions", "m", "q", "n_fetch", "hits", "unique", "n_cpu", "evictions",
                "victim_checks", "victim_matches", "fit_g_prior", "fit_c_prior",
                "cost_abs_err_ms", "cost_meas_ms", "cost_n", "decide_us",
                "pcie_ms", "gemm_ms", "cpu_ms", "cpu_n", "tokens")}

    def _new_token(self) -> None:
        super()._new_token()
        if self.stats:
            self.stats["tokens"] += 1
            if self.tok % STATS_EVERY_TOK == 0:
                self._flush_stats()

    def _flush_stats(self) -> None:
        if not self.stats_path or not self.stats or self.stats_key is None:
            return
        ctl = self.ctrl
        row = {"episode": self.stats_key[0], "phase": self.stats_key[1],
               "policy": ctl.get("policy_name"), "cache_rate": ctl.get("cache_rate"),
               "problem": ctl.get("problem"), "seed": ctl.get("seed"), "cap": self.cap,
               "alpha": (self.hits_done + PRIOR_N) / (self.pred_done + PRIOR_N),
               "hits_done": self.hits_done, "pred_done": self.pred_done, "cal_n": self.cal_n,
               "fit_g": list(self.fit_g.line()), "fit_c": list(self.fit_c.line()),
               "v_hit": self.v_hit, "time": time.time(), **self.stats}
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        with open(self.stats_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # -- features --------------------------------------------------------------
    def _lean_features(self, layer: int, uniq: list, missing: list, slots: list) -> dict:
        # Same features as ReuseAwareV2._lean_features; only the victim prediction differs.
        c = self.cache
        state = torch.cat((c.id_of_slot.long(), c.usage, c.step.view(1))).cpu().numpy()
        cap = self.cap
        ids_before, usage, step = state[:cap], state[cap:2 * cap], int(state[-1])
        self._ids_before = ids_before
        occupied = ids_before >= 0
        occ = int(occupied.sum())
        n = len(self.routed[layer])
        f32 = self.freq32[layer]
        miss_f32 = [f32.get(e, 0) for e in missing]
        m = len(missing)
        own = [slots[e] for e in uniq if slots[e] >= 0]
        vslots = engine_victims(usage, own, m)
        victims, victim_p = [], []
        E, prior = self.E, self.prior
        for g in ids_before[vslots].tolist():
            victims.append(g)
            if g < 0:
                victim_p.append(0.0)
            else:
                vl, ve = divmod(g, E)
                victim_p.append((self.freq32[vl].get(ve, 0) + prior * PRIOR_N)
                                / (len(self.routed[vl]) + PRIOR_N))
        return {
            "m": float(m), "n_hits": float(len(uniq) - m), "n_unique": float(len(uniq)),
            "free_slots": float(cap - occ), "occ": float(occ),
            "lru_age_tok": (step - int(usage[occupied].min())) / self.L if occ else 0.0,
            "_miss_f32": miss_f32, "_window": float(n),
            "_p_miss": [(f + self.prior * PRIOR_N) / (n + PRIOR_N) for f in miss_f32],
            "_victims": victims, "_victim_p": victim_p,
            "_alpha": (self.hits_done + PRIOR_N) / (self.pred_done + PRIOR_N),
            "_cost": self.cost, "_vhit": self.v_hit,
        }

    # -- per-layer outcome -----------------------------------------------------
    def _after_layer(self, layer, m, n_fetch, n_cpu, n_hits, pcie, gemm, cpu_ms, evicted, ages) -> None:
        cur, s = self.cur, self.stats
        x = cur["x"]
        self.cal_n += len(ages)
        match = ""
        if n_fetch:
            pred = x["_victims"][:n_fetch]
            match = int(sorted(g for g in pred if g >= 0) == sorted(evicted)
                        and sum(1 for g in pred if g < 0) == n_fetch - len(evicted))
        # V1 cost model evaluated at the realised q (every arm, so the error is comparable).
        g0, tp, c0, tc = x["_cost"]
        pred_ms = max(g0 + n_fetch * tp, c0 + n_cpu * tc if n_cpu > 0 else 0.0)
        meas_ms = max(pcie + gemm, cpu_ms if cpu_ms == cpu_ms else 0.0)
        cur["sx2"] = {"victim_match": match, "fit_g_ok": int(fit_ok(self.fit_g)),
                      "fit_c_ok": int(fit_ok(self.fit_c)), "cost_pred_ms": round(pred_ms, 5),
                      "cost_meas_ms": round(meas_ms, 5), "cal_n": self.cal_n}
        if not s:
            return
        s["decisions"] += 1
        s["m"] += m
        s["q"] += cur["q"]
        s["n_fetch"] += n_fetch
        s["hits"] += n_hits
        s["unique"] += x["n_unique"]
        s["n_cpu"] += n_cpu
        s["evictions"] += len(evicted)
        if n_fetch:
            s["victim_checks"] += 1
            s["victim_matches"] += match
        s["fit_g_prior"] += 0 if fit_ok(self.fit_g) else 1
        s["fit_c_prior"] += 0 if fit_ok(self.fit_c) else 1
        if m:
            s["cost_abs_err_ms"] += abs(pred_ms - meas_ms)
            s["cost_meas_ms"] += meas_ms
            s["cost_n"] += 1
        s["decide_us"] += 1e6 * self.decide_s
        s["pcie_ms"] += pcie
        s["gemm_ms"] += gemm
        if cpu_ms == cpu_ms and n_cpu > 0:
            s["cpu_ms"] += cpu_ms
            s["cpu_n"] += 1

    def _extra_row(self, m: int, x: dict, layer: int) -> dict:
        return dict(self.cur.get("sx2", {}))


def build(cache) -> Sx2Hook:
    return Sx2Hook(cache)
