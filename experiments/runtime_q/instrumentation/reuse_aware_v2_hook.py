"""Reuse-aware v2 pilot hook: a lean harness plus single-correction variants of ReuseAware.

Subclasses runtime_q_hook.RuntimeQ and leaves it (and its ReuseAware rule) untouched. The
control file selects the harness per episode:

  features="study" (default)  the parent's begin/finish_layer, feature groups and CSV schema,
                              i.e. exactly the harness of the completed reuse_aware study.
  features="lean"             only what the reuse rules need: routing ids, this layer's slot
                              row, one copy of the slot ids / usage / step, the 32-token
                              routing windows, the predicted LRU victims, the online reuse
                              calibration and the online branch-cost fit. Every lean episode
                              computes all of these whatever its policy, so harness cost is
                              the same across lean arms.

Variants (kind "reuse_aware_v"; each switches on one Stage-A correction, defaults = original):
  cost="runtime"     V1: fetch/hit costs in ms from an online fit of the two branch timers,
                         C(m, j) = max(g0 + j*tp, [m > j]*(c0 + (m - j)*tc)).
  reuse="calibrated" V2: R = alpha * p * T_res, alpha = (realised hits + 8) / (predicted
                         p*T_res + 8) over residencies already evicted in this episode.
  evict=true         V3: R = (p_e - p_victim) * T_res, victim = predicted LRU slot.
Decisions use only past measurements and the cache state before the fetch kernel runs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime_q_hook import Fraction, Policy, ReuseAware, RuntimeQ, round_balance  # noqa: E402
from schema import COLUMNS, GROUPS  # noqa: E402

PRIOR_N = 8.0  # pseudo-count convention shared with ReuseAware's p estimate
T_FREE = 32.0
LEAN_COLUMNS = [
    "episode", "phase", "policy", "workload", "cache_rate", "problem", "seed", "tok", "layer_id",
    "token_wall_ms", "m", "n_hits", "q", "q_native", "n_fetch", "wall_ms", "wall_net_ms",
    "decide_us", "pre_ms", "pcie_ms", "gemm_ms", "exposed_ms", "cpu_task_ms", "free_slots",
    "lru_age_tok", "occ", "n_evict", "alpha", "v_hit_ms", "g0_ms", "tp_ms", "c0_ms", "tc_ms",
    "q_orig", "q_v1", "q_v2", "q_v3", "routed_ids", "missing_ids", "fetched_ids", "evicted_gids",
    "victim_gids", "p_miss", "pred_fetched",
]


class ReuseVariant(Policy):
    """ReuseAware with optional runtime cost (V1), calibrated reuse (V2), eviction loss (V3).

    Fetch candidates in decreasing p while  scale * (p - p_victim) * T_res * v_hit >= dC(m, q),
    with dC in CPU-expert units (paper) or ms (runtime) and v_hit the matching value of one
    future hit (1 CPU expert, or its expected ms saving).
    """

    def __init__(self, spec: dict, rho: float, top_k: int, num_experts: int) -> None:
        self.cost = spec.get("cost", "paper")
        self.reuse = spec.get("reuse", "window")
        self.evict = bool(spec.get("evict", False))
        self.rho = float(spec.get("rho", rho))
        self.prior = top_k / num_experts

    def decide(self, m, x, layer):
        if m <= 0:
            return 0
        n = x["_window"]
        p = sorted(((f + self.prior * PRIOR_N) / (n + PRIOR_N) for f in x["_miss_f32"]), reverse=True)
        free, age = x["free_slots"], x["lru_age_tok"]
        scale = x["_alpha"] if self.reuse == "calibrated" else 1.0
        if self.cost == "runtime":
            g0, tp, c0, tc = x["_cost"]
            v = x["_vhit"]

            def cost(j):
                return max(g0 + j * tp, c0 + (m - j) * tc if j < m else 0.0)
        else:
            v, rho = 1.0, self.rho

            def cost(j):
                return max(j / rho, m - j)
        pv = x["_victim_p"]
        q = 0
        while q < m:
            r = p[q] - (pv[q] if self.evict else 0.0)
            if scale * r * (T_FREE if free > q else age) * v < cost(q + 1) - cost(q):
                break
            q += 1
        return q


class FractionMin1(Fraction):
    """Paper q* with the paper's minimum fill (\u00a73.2: "rounds q* to an integer ... It always
    retains at least one fill"): FreeToken's balance rounding of fraction * m, raised to 1
    whenever there is a miss. The engine and the `fraction` policy omit the minimum."""

    def decide(self, m, x, layer):
        return max(1, round_balance(self.fraction, m)) if m > 0 else 0


class BranchFit:
    """Least-squares line y = a + b*x over this episode's samples of one branch timer; the
    benchbw prior (a=0, b=prior) until 8 samples at >= 2 distinct x give a positive slope."""

    def __init__(self, slope_prior: float) -> None:
        self.prior = (0.0, slope_prior)
        self.n = self.sx = self.sxx = self.sy = self.sxy = 0.0
        self.xs: set = set()

    def add(self, x: float, y: float) -> None:
        self.n += 1
        self.sx += x
        self.sxx += x * x
        self.sy += y
        self.sxy += x * y
        if len(self.xs) < 2:
            self.xs.add(x)

    def line(self) -> tuple[float, float]:
        if self.n < PRIOR_N or len(self.xs) < 2:
            return self.prior
        den = self.n * self.sxx - self.sx * self.sx
        b = (self.n * self.sxy - self.sx * self.sy) / den if den > 0 else 0.0
        if b <= 0:
            return self.prior
        return max(0.0, (self.sy - b * self.sx) / self.n), b


class ReuseAwareV2(RuntimeQ):
    def __init__(self, cache) -> None:
        super().__init__(cache)
        self.lean = False
        self.cur_lean = False
        self.log_key = None
        self.lean_columns = LEAN_COLUMNS  # subclasses may extend (see _extra_row)

    # -- state -----------------------------------------------------------------
    def _reset_state(self) -> None:
        # Capacity is re-read here, not cached from _init: a runtime rebuild
        # (offload_cache.rebuild) reallocates the slot cache at a new size.
        self.cap = self.cache.cache_size
        super()._reset_state()
        self.top_k = getattr(self.executor, "top_k", 4)
        self.prior = self.top_k / self.E
        self.resident: dict[int, list] = {}  # gid -> [predicted p*T_res, realised hits, tok]
        self.hits_done = self.pred_done = 0.0
        self.fit_g = BranchFit(self.expert_bytes / (self.bp * 1e6))  # pcie + gemm vs n_fetch
        self.fit_c = BranchFit(self.expert_bytes / (self.bh * 1e6))  # cpu task vs n_cpu
        self.m_count = [0] * (self.top_k + 1)
        self._refit()

    def _refit(self) -> None:
        g0, tp = self.fit_g.line()
        c0, tc = self.fit_c.line()
        self.cost = (g0, tp, c0, tc)

        def c(n):
            return max(g0, c0 + n * tc if n > 0 else 0.0)
        tot = sum(self.m_count[1:])
        if tot:
            self.v_hit = sum(k * (c(m) - c(m - 1)) for m, k in enumerate(self.m_count) if m) / tot
        else:
            self.v_hit = c(1) - c(0)

    # -- control (parent's semantics, plus a per-mode log schema) --------------------
    def _check_control(self) -> None:
        try:
            mtime = os.stat(self.ctrl_path).st_mtime_ns
        except FileNotFoundError:
            return
        if mtime == self.ctrl_mtime:
            return
        self.ctrl_mtime = mtime
        with open(self.ctrl_path) as f:
            new = json.load(f)
        new_episode = new.get("episode") != self.ctrl.get("episode")
        self.ctrl = new
        self.groups = set(GROUPS) if new.get("groups", "full") == "full" else set(new["groups"])
        self.timing = bool(new.get("timing", True))
        self.lean = new.get("features", "study") == "lean"
        if new_episode:
            self.cache.reset()
            self._reset_state()
            self.policy = self._build(new["policy"], int(new.get("seed", 0)))
            if isinstance(self.policy, Fraction) and self.policy.fraction < 0:
                self.policy.fraction = self.native_frac
        key = (new.get("log"), self.lean)
        if key != self.log_key:
            if self.out_file:
                self.out_file.close()
            self.log_key, self.out_path = key, new.get("log")
            self.out_file = self.writer = None
            if self.out_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
                fresh = not os.path.exists(self.out_path) or os.path.getsize(self.out_path) == 0
                self.out_file = open(self.out_path, "a", newline="")
                self.writer = csv.DictWriter(self.out_file, fieldnames=self.lean_columns if self.lean else COLUMNS,
                                             extrasaction="ignore")
                if fresh:
                    self.writer.writeheader()

    def _on_resize(self) -> None:
        """The slot cache was rebuilt mid-episode (idle-only, so between requests): adopt
        the new capacity. The rebuild emptied every slot, so residency records are dropped
        unscored; measured branch costs and routing windows stay valid."""
        self.cap = self.cache.cache_size
        self.resident.clear()
        self.occ = 0

    def _after_layer(self, layer, m, n_fetch, n_cpu, n_hits, pcie, gemm, cpu_ms, evicted, ages) -> None:
        """Per-layer measured sample after the branch timers are read (subclass extension)."""

    def _extra_row(self, m: int, x: dict, layer: int) -> dict:
        """Extra decision-log fields for subclasses that extend lean_columns."""
        return {}

    def _build(self, spec: dict, seed: int) -> Policy:
        if spec["kind"] == "fraction_min1":
            return FractionMin1(float(spec["fraction"]))
        if spec["kind"] == "reuse_aware_v":
            return ReuseVariant(spec, self.bp / self.bh, self.top_k, self.E)
        return super()._build(spec, seed)

    # -- DecodeTrace interface -------------------------------------------------
    def begin(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        if not self.in_layer:
            return
        if layer_id == 0:
            self._check_control()  # settle the mode before dispatching this token
            if self.cache.cache_size != self.cap:
                self._on_resize()
        self.cur_lean = self.lean
        if not self.lean:
            return super().begin(layer_id, expert_ids)
        tb = time.perf_counter()
        if layer_id == 0:
            self._new_token()
        c = self.cache
        ids = expert_ids.reshape(-1).tolist()
        uniq = list(dict.fromkeys(ids))
        slots = c.slot_for_id[layer_id].tolist()
        missing = [e for e in uniq if slots[e] < 0]
        m = len(missing)
        x = self._lean_features(layer_id, uniq, missing, slots)
        q = self.policy.decide(m, x, layer_id) if self.policy else None
        q_native = round_balance(self.native_frac, m) if self.native_frac > 0 else min(m, self.native_max)
        if q is None:
            c.hybrid_max_fetch, c.hybrid_fetch_fraction = self.native_max, self.native_frac
            q = q_native
        else:
            q = max(0, min(m, int(q)))
            c.hybrid_max_fetch, c.hybrid_fetch_fraction = q, 0.0
        task = self.timed_tasks.get(layer_id)
        if task is None and self.timing:
            task = self.executor._task_for(layer_id, expert_ids.shape[0])
            self.executor._ext.set_task_timing(task, True)
            self.timed_tasks[layer_id] = task
        self.cur = {"layer": layer_id, "m": m, "q": q, "q_native": q_native, "x": x,
                    "uniq": uniq, "missing": missing, "task": task}
        self.decide_s = time.perf_counter() - tb

    def _lean_features(self, layer: int, uniq: list, missing: list, slots: list) -> dict:
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
        # predicted victims: the engine's argmin(usage) over slots not owned by this layer's
        # routed experts (free slots have usage 0, so they go first); ties -> lower slot
        victims, victim_p = [], []
        k = min(m, cap)
        if k:
            u = usage.copy()
            own = [slots[e] for e in uniq if slots[e] >= 0]
            u[own] = np.iinfo(np.int64).max
            idx = np.argpartition(u, k - 1)[:k] if k < cap else np.arange(cap)
            idx = idx[np.lexsort((idx, u[idx]))][:k]
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
            "_alpha": (self.hits_done + PRIOR_N) / (self.pred_done + PRIOR_N),
            "_cost": self.cost, "_vhit": self.v_hit,
        }

    def finish_layer(self) -> None:
        if not self.in_layer:
            return
        if not self.cur_lean:
            return super().finish_layer()
        self.in_layer = False
        c, cur, ev = self.cache, self.cur, self.ev
        ev["end"].record()
        torch.cuda.synchronize(c.device)
        wall = 1000 * (time.perf_counter() - self.t0)
        layer, m, x, missing = cur["layer"], cur["m"], cur["x"], cur["missing"]
        n_fetch = int(c.num_indices.item())
        pcie = ev["pcie_start"].elapsed_time(ev["pcie_end"])
        gemm = ev["gpu_start"].elapsed_time(ev["gpu_end"])
        pre = ev["start"].elapsed_time(ev["pcie_start"])
        exposed = ev["gpu_end"].elapsed_time(ev["end"])
        try:
            cpu_ms = self.executor._ext.task_timing_ms(cur["task"]) if cur["task"] else float("nan")
        except RuntimeError:
            cpu_ms = float("nan")
        fetched, evicted = [], []
        if n_fetch:
            post = c.slot_for_id[layer].tolist()
            fetched = [e for e in missing if post[e] >= 0]
            before = self._ids_before
            after = c.id_of_slot.cpu().numpy()
            changed = np.nonzero(after != before)[0]
            evicted = [int(before[i]) for i in changed if before[i] >= 0]
        n_cpu = m - n_fetch

        # residency bookkeeping for the reuse calibration (past-only)
        E, tok, res = self.E, self.tok, self.resident
        for e in cur["uniq"]:
            rec = res.get(layer * E + e)
            if rec is not None and rec[2] < tok:
                rec[1] += 1
        pred = []
        if fetched:
            pm = dict(zip(missing, x["_p_miss"]))
            rank = sorted(missing, key=lambda e: -pm[e])
            free, age = x["free_slots"], x["lru_age_tok"]
            for e in fetched:
                j = rank.index(e)
                r = pm[e] * (T_FREE if free > j else age)
                res[layer * E + e] = [r, 0, tok]
                pred.append(round(r, 3))
        ages = []
        for g in evicted:
            rec = res.pop(g, None)
            if rec is not None:
                self.hits_done += rec[1]
                self.pred_done += rec[0]
                ages.append(tok - rec[2])

        # branch-cost samples (lagged: they inform later decisions only)
        self.fit_g.add(float(n_fetch), pcie + gemm)
        if n_cpu > 0 and cpu_ms == cpu_ms:
            self.fit_c.add(float(n_cpu), cpu_ms)
        if m:
            self.m_count[min(m, self.top_k)] += 1
        self._refit()
        self._after_layer(layer, m, n_fetch, n_cpu, int(x["n_hits"]), pcie, gemm, cpu_ms, evicted, ages)

        routed = self.routed[layer]
        if len(routed) == routed.maxlen:
            self.freq32[layer].subtract(routed[0])
        routed.append(cur["uniq"])
        self.freq32[layer].update(cur["uniq"])
        self.policy.observe(layer, wall)

        if self.writer is not None:
            shadow = {}
            for name, spec in (("q_orig", None), ("q_v1", {"cost": "runtime"}),
                               ("q_v2", {"reuse": "calibrated"}), ("q_v3", {"evict": True})):
                pol = ReuseAware({}, self.bp / self.bh, self.top_k, self.E) if spec is None \
                    else ReuseVariant(spec, self.bp / self.bh, self.top_k, self.E)
                shadow[name] = pol.decide(m, x, layer)
            ctl, g0, tp, c0, tc = self.ctrl, *x["_cost"]
            row = {
                "episode": ctl.get("episode"), "phase": ctl.get("phase"),
                "policy": ctl.get("policy_name"), "workload": ctl.get("workload"),
                "cache_rate": ctl.get("cache_rate"), "problem": ctl.get("problem"),
                "seed": ctl.get("seed"), "tok": self.tok, "layer_id": layer,
                "token_wall_ms": round(self.token_wall_ms, 4) if layer == 0 else "",
                "m": m, "n_hits": int(x["n_hits"]), "q": cur["q"], "q_native": cur["q_native"],
                "n_fetch": n_fetch, "wall_ms": round(wall, 4),
                "wall_net_ms": round(wall - 1000 * self.decide_s, 4),
                "decide_us": round(1e6 * self.decide_s, 1), "pre_ms": round(pre, 4),
                "pcie_ms": round(pcie, 4), "gemm_ms": round(gemm, 4), "exposed_ms": round(exposed, 4),
                "cpu_task_ms": round(cpu_ms, 4) if cpu_ms == cpu_ms else "",
                "free_slots": int(x["free_slots"]), "lru_age_tok": round(x["lru_age_tok"], 3),
                "occ": int(x["occ"]), "n_evict": len(evicted), "alpha": round(x["_alpha"], 4),
                "v_hit_ms": round(x["_vhit"], 5), "g0_ms": round(g0, 5), "tp_ms": round(tp, 5),
                "c0_ms": round(c0, 5), "tc_ms": round(tc, 5), **shadow,
                "routed_ids": ";".join(map(str, cur["uniq"])),
                "missing_ids": ";".join(map(str, missing)),
                "fetched_ids": ";".join(map(str, fetched)),
                "evicted_gids": ";".join(map(str, evicted)),
                "victim_gids": ";".join(map(str, x["_victims"])),
                "p_miss": ";".join(f"{v:.4f}" for v in x["_p_miss"]),
                "pred_fetched": ";".join(map(str, pred)),
            }
            row.update(self._extra_row(m, x, layer))
            self.writer.writerow(row)
            if layer == self.L - 1:
                self.out_file.flush()


def build(cache) -> ReuseAwareV2:
    return ReuseAwareV2(cache)
