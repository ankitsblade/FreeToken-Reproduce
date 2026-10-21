"""Policy-comparison hook for q_expswithreuse_aware: frozen decision rules + validation counters.

Loaded by FREETOKEN_MOE_RUNTIME_Q_MODULE. Subclasses the frozen
experiments/runtime_q/instrumentation/reuse_aware_v2_hook.ReuseAwareV2 (itself a subclass of
runtime_q_hook.RuntimeQ) and changes no decision. Every decision comes from a frozen class:

  fraction_min1   Policy A, paper q* with the minimum fill   reuse_aware_v2_hook.FractionMin1
  fraction        A0 diagnostic, paper q* without minimum    runtime_q_hook.Fraction
  native          E diagnostic, engine built-in q*           engine kernel (hook returns None)
  reuse_aware     Policy B, original Reuse-Aware              runtime_q_hook.ReuseAware
  reuse_aware_v   Policy C, B + V1/V2/V3 switches             reuse_aware_v2_hook.ReuseVariant

One input differs from the frozen harness: the predicted eviction victims (used by V3 only) are
chosen exactly as the engine chooses them; the frozen prediction breaks usage ties arbitrarily at
the k-th boundary (docs/AUDIT.md, finding F7). Everything else this file adds is logging, computed
in every lean episode whatever the policy, so the harness cost stays equal across arms:

  V1  sample counts and prior-fallback state of both branch fits at decision time, and the cost
      model's prediction C(m, q) for the q actually taken (compared offline with the measured
      copy-start -> layer-end time)
  V2  number of finished residencies behind alpha, and the realised / predicted totals
  V3  misses of experts this episode had cached and evicted earlier (reuse lost to eviction),
      with tokens since that eviction; cold misses are the rest
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "experiments", "runtime_q", "instrumentation"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from reuse_aware_v2_hook import LEAN_COLUMNS, PRIOR_N, ReuseAwareV2  # noqa: E402

QX_COLUMNS = LEAN_COLUMNS + [
    "fit_g_n", "fit_c_n", "fit_g_prior", "fit_c_prior", "c_pred_ms",
    "calib_n", "hits_done", "pred_done", "n_remiss", "remiss_tok", "victim_fix",
]


def branch_cost(cost: tuple, m: int, q: int) -> float:
    """V1 cost model C(m, q) = max(g0 + q tp, [m > q] (c0 + (m - q) tc)), in ms."""
    g0, tp, c0, tc = cost
    return max(g0 + q * tp, c0 + (m - q) * tc if q < m else 0.0)


def engine_victim_slots(u: np.ndarray, k: int) -> np.ndarray:
    """The k slots the hybrid kernel evicts, in order: smallest usage first, ties to the lower
    slot (repeated argmin over usage). O(cap): partition, then break the boundary tie by index."""
    if k >= len(u):
        idx = np.arange(len(u))
    else:
        thr = np.partition(u, k - 1)[k - 1]
        lo = np.flatnonzero(u < thr)
        idx = np.concatenate([lo, np.flatnonzero(u == thr)[:k - len(lo)]])
    return idx[np.lexsort((idx, u[idx]))][:k]


class QExpHook(ReuseAwareV2):
    def __init__(self, cache) -> None:
        super().__init__(cache)
        self.lean_columns = QX_COLUMNS

    def _reset_state(self) -> None:
        super()._reset_state()
        self.calib_n = 0
        self.evicted_tok: dict[int, int] = {}  # gid -> token of its latest eviction
        self.qx: dict = {}

    def _on_resize(self) -> None:
        super()._on_resize()
        self.evicted_tok.clear()

    def _lean_features(self, layer, uniq, missing, slots) -> dict:
        # Same features as ReuseAwareV2._lean_features (one device->host copy of slot ids, usage
        # and step) except the predicted victims: the frozen version takes np.argpartition's k
        # smallest usage stamps, which picks an arbitrary slot when the k-th stamp is tied, while
        # the engine evicts ties in slot order. Fills of one kernel call share a stamp, so such
        # ties are common. The frozen prediction is kept for the mismatch count only.
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
        victims, victim_p, frozen = [], [], []
        k = min(m, cap)
        if k:
            u = usage.copy()
            own = [slots[e] for e in uniq if slots[e] >= 0]
            u[own] = np.iinfo(np.int64).max
            idx = engine_victim_slots(u, k)
            old = np.argpartition(u, k - 1)[:k] if k < cap else np.arange(cap)
            frozen = ids_before[old[np.lexsort((old, u[old]))][:k]].tolist()
            E, prior = self.E, self.prior
            for g in ids_before[idx].tolist():
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
            "_victim_fix": sum(a != b for a, b in zip(victims, frozen)),
            "_alpha": (self.hits_done + PRIOR_N) / (self.pred_done + PRIOR_N),
            "_cost": self.cost, "_vhit": self.v_hit,
            # fit state as the decision sees it (both fits are updated only after this layer)
            "_fit_state": (int(self.fit_g.n), int(self.fit_c.n),
                           int(self.fit_g.line() == self.fit_g.prior),
                           int(self.fit_c.line() == self.fit_c.prior)),
        }

    def _after_layer(self, layer, m, n_fetch, n_cpu, n_hits, pcie, gemm, cpu_ms, evicted, ages) -> None:
        E, tok = self.E, self.tok
        remiss, dist = 0, []
        for e in self.cur["missing"]:  # counted before this layer's own evictions are recorded
            t = self.evicted_tok.get(layer * E + e)
            if t is not None:
                remiss += 1
                dist.append(tok - t)
        for g in evicted:
            self.evicted_tok[g] = tok
        self.calib_n += len(ages)
        x = self.cur["x"]
        g_n, c_n, g_prior, c_prior = x["_fit_state"]
        self.qx = {
            "fit_g_n": g_n, "fit_c_n": c_n, "fit_g_prior": g_prior, "fit_c_prior": c_prior,
            "c_pred_ms": round(branch_cost(x["_cost"], m, self.cur["q"]), 5) if m else "",
            "calib_n": self.calib_n, "hits_done": round(self.hits_done, 3),
            "pred_done": round(self.pred_done, 3), "n_remiss": remiss,
            "remiss_tok": ";".join(map(str, dist)), "victim_fix": x["_victim_fix"],
        }

    def _extra_row(self, m: int, x: dict, layer: int) -> dict:
        return self.qx


def build(cache) -> QExpHook:
    return QExpHook(cache)
