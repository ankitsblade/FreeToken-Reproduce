"""Host-side runtime q policy + feature logger for FreeToken's hybrid decode.

Loaded only when FREETOKEN_MOE_RUNTIME_Q_MODULE points at this file; it duck-types
freetoken.moe.decode_trace.DecodeTrace (start_layer / begin / finish / finish_layer), so
no other engine code changes. Eager decode only (--no-graph): the decision needs the routed
ids and the cache state on the host.

Decision point (begin, called inside ensure_experts_hybrid before the fetch kernel):
  routed experts known -> cache state read -> missing set M, m = |M| -> features x
  -> policy picks q in {0..m} -> cache.hybrid_max_fetch = q for this one call.
All features are computed here, before the action. Outcomes are measured in finish_layer.

A JSON control file (FREETOKEN_MOE_RUNTIME_Q_CONTROL) selects the policy and delimits
episodes; a new episode resets the expert cache so several policies can be measured in one
server process from the same cold state.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, deque

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_models import CostModel, linear_design  # noqa: E402
from schema import COLUMNS, GROUPS, HIST_SIGNALS, HIST_WINDOWS, PRE_ACTION  # noqa: E402

Q16 = 1 << 16
REC_CAP = 64.0


def round_balance(fraction: float, m: int) -> int:
    """FreeToken's fraction -> q rounding (offload_kernels.py), reproduced exactly."""
    if m <= 0:
        return 0
    frac = min(Q16, max(0, round(fraction * Q16)))
    lo = (m * frac) >> 16
    cost = lambda f: max(f * (Q16 - frac), (m - f) * frac)  # noqa: E731
    return max(0, min(m, lo if cost(lo) <= cost(lo + 1) else lo + 1))


# --------------------------------------------------------------------------- policies
class Policy:
    name = "base"

    def decide(self, m: int, x: dict, layer: int) -> int | None:
        raise NotImplementedError

    def observe(self, layer: int, wall_ms: float) -> None:
        pass


class Native(Policy):
    """Leave the engine's own fetch fraction untouched (reference q*)."""

    def decide(self, m, x, layer):
        return None


class Fraction(Policy):
    def __init__(self, fraction: float) -> None:
        self.fraction = fraction

    def decide(self, m, x, layer):
        return round_balance(self.fraction, m)


class Explore(Policy):
    """With prob eps a uniform-random valid q, else the base fraction (eps=1 -> uniform)."""

    def __init__(self, fraction: float, eps: float, seed: int) -> None:
        self.fraction, self.eps, self.rng = fraction, eps, random.Random(seed)

    def decide(self, m, x, layer):
        if m and self.rng.random() < self.eps:
            return self.rng.randint(0, m)
        return round_balance(self.fraction, m)


class Table(Policy):
    """q as a function of m only (table[m]), with optional eps-uniform exploration."""

    def __init__(self, table: list, eps: float, seed: int) -> None:
        self.table, self.eps, self.rng = [int(v) for v in table], eps, random.Random(seed)

    def decide(self, m, x, layer):
        if m and self.eps and self.rng.random() < self.eps:
            return self.rng.randint(0, m)
        return min(m, self.table[min(m, len(self.table) - 1)])


class ModelPolicy(Policy):
    def __init__(self, path: str) -> None:
        self.model = CostModel.load(path)
        self.features = self.model.features

    def decide(self, m, x, layer):
        if m <= 0:
            return 0
        return self.model.choose(np.array([x[f] for f in self.features], dtype=np.float64), m)


class Rule(Policy):
    """Interpretable cache-aware rule: fetch a missing expert only if it looks reusable.

    q = #missing experts routed at least `min_freq` times in the last 32 tokens of this
    layer, capped at `max_q`; if the LRU victim is younger than `min_victim_age` tokens the
    cache is thrashing and nothing is fetched.
    """

    def __init__(self, min_freq: float, max_q: int, min_victim_age: float) -> None:
        self.min_freq, self.max_q, self.min_victim_age = min_freq, max_q, min_victim_age

    def decide(self, m, x, layer):
        if m <= 0:
            return 0
        if x["free_slots"] <= 0 and x["lru_age_tok"] < self.min_victim_age:
            return 0
        return int(min(m, self.max_q, x["_n_miss_freq_ge"].get(self.min_freq, 0)))


class ReuseAware(Policy):
    """Analytic reuse-aware rule (dimensionless; nests q*).

    Marginal exposed cost of fetching one more expert, in CPU-expert units, from the same
    two-branch overlap model q* uses (rho = B_P / B_H = t_cpu / t_pcie per expert):
        delta(m, q) = max((q + 1) / rho, m - q - 1) - max(q / rho, m - q)
    delta <= 0 means the fetch is fully hidden behind the CPU branch (this is q*'s regime).
    Expected reuses of a fetched expert while it stays resident:
        R = p * T_res,  p = recent routing frequency of that expert in this layer,
        T_res = `t_free` tokens if the cache has a free slot, else the LRU victim's age.
    Fetch candidates in decreasing R while  R >= kappa * delta(m, q).
    """

    def __init__(self, spec: dict, rho: float, top_k: int, num_experts: int) -> None:
        self.kappa = float(spec.get("kappa", 1.0))
        self.t_free = float(spec.get("t_free", 32.0))
        self.rho = float(spec.get("rho", rho))
        self.per_expert = bool(spec.get("per_expert", True))
        self.use_cache = bool(spec.get("use_cache", True))
        self.online = bool(spec.get("rho_online", False))
        self.prior, self.prior_n = top_k / num_experts, float(spec.get("prior_n", 8.0))

    def decide(self, m, x, layer):
        if m <= 0:
            return 0
        n = x["_window"]
        if self.per_expert:
            p = sorted(((f + self.prior * self.prior_n) / (n + self.prior_n) for f in x["_miss_f32"]),
                       reverse=True)
        else:
            p = [self.prior] * m
        free, age = (x["free_slots"], x["lru_age_tok"]) if self.use_cache else (0.0, self.t_free)
        rho = self.rho
        if self.online and x["cpu_ms_per_exp_ewma"] > 0 and x["pcie_ms_per_exp_ewma"] > 0:
            rho = x["cpu_ms_per_exp_ewma"] / x["pcie_ms_per_exp_ewma"]  # measured t_cpu / t_pcie
        self_rho, self.rho = self.rho, rho
        q = 0
        while q < m:
            delta = max((q + 1) / self.rho, m - q - 1) - max(q / self.rho, m - q)
            t_res = self.t_free if free > q else age
            if p[q] * t_res < self.kappa * delta:
                break
            q += 1
        self.rho = self_rho
        return q


class EpsWrap(Policy):
    """Base policy with eps-uniform exploration (on-policy data for retraining)."""

    def __init__(self, base: Policy, eps: float, seed: int) -> None:
        self.base, self.eps, self.rng = base, eps, random.Random(seed)

    def decide(self, m, x, layer):
        q = self.base.decide(m, x, layer)
        if m and self.rng.random() < self.eps:
            return self.rng.randint(0, m)
        return q

    def observe(self, layer, wall_ms):
        self.base.observe(layer, wall_ms)


class LinBandit(Policy):
    """Shared-parameter linear contextual bandit over phi(x, q); LinUCB or Thompson.

    Cost = wall time of this layer now plus its next `horizon` visits (delayed update), so
    a fetch can be credited for the cache hits it causes. horizon=0 is the myopic bandit.
    Variable action set: only q in {0..m} is ever scored.
    """

    def __init__(self, spec: dict, seed: int) -> None:
        self.features = list(spec["features"])
        self.mean = np.asarray(spec["x_mean"], dtype=np.float64)
        self.std = np.asarray(spec["x_std"], dtype=np.float64)
        self.mode = spec.get("mode", "ucb")
        self.alpha = float(spec.get("alpha", 1.0))
        self.horizon = int(spec.get("horizon", 8))
        self.scale = float(spec.get("cost_scale", 1.0))
        d = 2 * len(self.features) + 4  # linear_design + bias
        lam = float(spec.get("lambda", 1.0))
        self.a_inv = np.eye(d) / lam
        theta0 = np.asarray(spec["theta0"], dtype=np.float64) if spec.get("theta0") else np.zeros(d)
        self.b = lam * theta0
        self.theta = theta0.copy()
        self.rng = np.random.default_rng(seed)
        self.pending: dict[int, deque] = {}

    def decide(self, m, x, layer):
        if m <= 0:
            self._chosen = None
            return 0
        z = (np.array([x[f] for f in self.features], dtype=np.float64) - self.mean) / self.std
        q = np.arange(m + 1, dtype=np.float64)
        phi = linear_design(np.repeat(z[None, :], m + 1, axis=0), q, np.full(m + 1, float(m)))
        phi = np.concatenate([phi, np.ones((m + 1, 1))], axis=1)
        if self.mode == "ts":
            chol = np.linalg.cholesky(self.a_inv + 1e-9 * np.eye(len(self.theta)))
            theta = self.theta + self.alpha * chol @ self.rng.standard_normal(len(self.theta))
            score = phi @ theta
        else:
            bonus = np.sqrt(np.einsum("ij,jk,ik->i", phi, self.a_inv, phi))
            score = phi @ self.theta - self.alpha * bonus
        k = int(np.argmin(score))
        self._chosen = phi[k]
        return k

    def observe(self, layer, wall_ms):
        queue = self.pending.setdefault(layer, deque())
        for item in queue:
            item[1] += wall_ms
            item[2] += 1
        queue.append([self._chosen, wall_ms, 0])
        while queue and queue[0][2] >= self.horizon:
            phi, cost, _ = queue.popleft()
            if phi is None:
                continue
            av = self.a_inv @ phi
            self.a_inv -= np.outer(av, av) / (1.0 + phi @ av)
            self.b += phi * (cost / self.scale)
            self.theta = self.a_inv @ self.b


def make_policy(spec: dict, seed: int) -> Policy:
    kind = spec["kind"]
    if kind == "native":
        return Native()
    if kind == "fraction":
        return Fraction(float(spec["fraction"]))
    if kind == "explore":
        return Explore(float(spec.get("fraction", 0.0)), float(spec.get("eps", 1.0)), seed)
    if kind == "table":
        return Table(spec["table"], float(spec.get("eps", 0.0)), seed)
    if kind == "model":
        return ModelPolicy(spec["path"])
    if kind == "rule":
        return Rule(float(spec["min_freq"]), int(spec.get("max_q", 4)),
                    float(spec.get("min_victim_age", 0.0)))
    if kind == "bandit":
        return LinBandit(spec, seed)
    raise ValueError(f"unknown policy kind {kind!r}")


# --------------------------------------------------------------------------- hook
class RuntimeQ:
    def __init__(self, cache) -> None:
        self.cache = cache
        self.ctrl_path = os.environ["FREETOKEN_MOE_RUNTIME_Q_CONTROL"]
        self.bp = float(os.environ.get("FREETOKEN_MOE_RUNTIME_Q_BP", "6.71"))
        self.bh = float(os.environ.get("FREETOKEN_MOE_RUNTIME_Q_BH", "35.67"))
        self.ready = False
        self.in_layer = False
        self.ctrl_mtime = None
        self.ctrl: dict = {}
        self.policy: Policy | None = None
        self.writer = None
        self.out_file = None
        self.out_path = None
        self.rows: list[dict] = []

    # -- setup -----------------------------------------------------------------
    def _init(self) -> None:
        c = self.cache
        self.L, self.E, self.cap = c.num_layers, c.num_experts, c.cache_size
        self.native_max, self.native_frac = c.hybrid_max_fetch, c.hybrid_fetch_fraction
        self.expert_bytes = sum(src[0][0].numel() * src[0].element_size() for src, _ in c.banks)
        self.expert_mb = self.expert_bytes / 1e6
        self.ev = {k: torch.cuda.Event(enable_timing=True)
                   for k in ("start", "pcie_start", "pcie_end", "gpu_start", "gpu_end", "end")}
        c._layer_replay_events = self.ev  # recorded by layers/moe.py::_decode_hybrid
        self.executor = c.cpu_executor
        self.cpu_threads = float(getattr(self.executor, "num_threads", 0) or os.cpu_count() or 0)
        self.timed_tasks: dict[int, int] = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.nvml_h = pynvml.nvmlDeviceGetHandleByIndex(c.device.index or 0)
        except Exception:
            self.nvml = None
        self.cpu_prev = self._cpu_counters()
        self.sys = {"cpu_util": 0.0, "gpu_util": 0.0, "gpu_mem_util": 0.0, "load1": 0.0,
                    "gpu_free_gb": 0.0, "host_avail_gb": 0.0}
        self.groups = set(GROUPS)
        self.timing = True
        self._ids_before = None
        if self.policy is None:
            self.policy = Native()
        self._reset_state()
        self.ready = True

    def _reset_state(self) -> None:
        L = self.L
        self.tok = -1
        self.t_tok = None
        self.token_wall_ms = 0.0
        self.hist = [deque(maxlen=max(HIST_WINDOWS)) for _ in range(L)]
        self.routed = [deque(maxlen=32) for _ in range(L)]
        self.freq32 = [Counter() for _ in range(L)]
        self.last_seen = [dict() for _ in range(L)]
        self.ewma = {"cpu_ms_per_exp": [0.0] * L, "gemm": [0.0] * L, "hit": [0.0] * L}
        self.pcie_per_exp_ewma = 0.0
        self.bp_recent = self.bp
        self.bh_recent = self.bh
        self.ins_hist = deque(maxlen=32)
        self.evi_hist = deque(maxlen=32)
        self.tok_ins = self.tok_evi = 0
        self.occ = 0

    @staticmethod
    def _cpu_counters() -> tuple[int, int]:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
        return sum(v), v[3] + v[4]

    def _sample_system(self) -> None:
        cur = self._cpu_counters()
        tot, idle = cur[0] - self.cpu_prev[0], cur[1] - self.cpu_prev[1]
        self.cpu_prev = cur
        s = self.sys
        if tot > 0:
            s["cpu_util"] = 100.0 * (tot - idle) / tot
        with open("/proc/loadavg") as f:
            s["load1"] = float(f.read().split()[0])
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    s["host_avail_gb"] = int(line.split()[1]) / 1048576
                    break
        if self.nvml is not None:
            u = self.nvml.nvmlDeviceGetUtilizationRates(self.nvml_h)
            mem = self.nvml.nvmlDeviceGetMemoryInfo(self.nvml_h)
            s["gpu_util"], s["gpu_mem_util"] = float(u.gpu), float(u.memory)
            s["gpu_free_gb"] = mem.free / 2**30

    # -- control ---------------------------------------------------------------
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
        if new_episode:
            self.cache.reset()  # same cold expert cache for every episode
            self._reset_state()
            self.policy = self._build(new["policy"], int(new.get("seed", 0)))
            if isinstance(self.policy, Fraction) and self.policy.fraction < 0:
                self.policy.fraction = self.native_frac  # q* recomputed on the host
        if new.get("log") != self.out_path:
            if self.out_file:
                self.out_file.close()
            self.out_path = new.get("log")
            self.out_file = self.writer = None
            if self.out_path:
                os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
                fresh = not os.path.exists(self.out_path) or os.path.getsize(self.out_path) == 0
                self.out_file = open(self.out_path, "a", newline="")
                self.writer = csv.DictWriter(self.out_file, fieldnames=COLUMNS,
                                             extrasaction="ignore")
                if fresh:
                    self.writer.writeheader()

    def _build(self, spec: dict, seed: int) -> Policy:
        if spec["kind"] == "eps_wrap":
            return EpsWrap(self._build(spec["base"], seed), float(spec["eps"]), seed + 1)
        if spec["kind"] == "reuse_aware":
            return ReuseAware(spec, self.bp / self.bh, getattr(self.executor, "top_k", 4), self.E)
        return make_policy(spec, seed)

    def _new_token(self) -> None:
        self._check_control()
        now = time.perf_counter()
        self.token_wall_ms = 1000 * (now - self.t_tok) if self.t_tok is not None else 0.0
        self.t_tok = now
        self.tok += 1
        self.ins_hist.append(self.tok_ins)
        self.evi_hist.append(self.tok_evi)
        self.tok_ins = self.tok_evi = 0
        if {"G8_util", "G9_memory"} & self.groups:
            self._sample_system()

    # -- DecodeTrace interface -------------------------------------------------
    def start_layer(self) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("runtime q hook requires eager decode (--no-graph)")
        if not self.ready:
            self._init()
        torch.cuda.synchronize(self.cache.device)
        self.in_layer = True
        self.t0 = time.perf_counter()
        self.ev["start"].record()

    def begin(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        if not self.in_layer:
            return
        tb = time.perf_counter()
        if layer_id == 0:
            self._new_token()
        c = self.cache
        ids = expert_ids.reshape(-1).tolist()
        uniq = list(dict.fromkeys(ids))
        slots = c.slot_for_id[layer_id].tolist()
        missing = [e for e in uniq if slots[e] < 0]
        m = len(missing)
        x = self._features(layer_id, len(ids), uniq, missing, slots)
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
                    "uniq": uniq, "missing": missing, "slots": slots, "task": task}
        self.decide_s = time.perf_counter() - tb

    def finish(self) -> None:
        pass

    def finish_layer(self) -> None:
        if not self.in_layer:
            return
        self.in_layer = False
        c, cur, ev = self.cache, self.cur, self.ev
        ev["end"].record()
        torch.cuda.synchronize(c.device)
        wall = 1000 * (time.perf_counter() - self.t0)
        layer, m, x = cur["layer"], cur["m"], cur["x"]
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
            fetched = [e for e in cur["missing"] if post[e] >= 0]
            if self._ids_before is not None:
                after = c.id_of_slot.tolist()
                evicted = [b for b, a in zip(self._ids_before, after) if b != a and b >= 0]
            else:
                evicted = [0] * max(0, self.occ + n_fetch - self.cap)
        n_evict = len(evicted)
        self.occ += n_fetch - n_evict
        self.tok_ins += n_fetch
        self.tok_evi += n_evict
        n_cpu = m - n_fetch
        hits = x["n_hits"]

        # lagged measurements become features of later decisions
        a = 0.2
        if n_cpu > 0 and cpu_ms == cpu_ms:
            per = cpu_ms / n_cpu
            e = self.ewma["cpu_ms_per_exp"]
            e[layer] = per if e[layer] == 0 else (1 - a) * e[layer] + a * per
            self.bh_recent = (1 - a) * self.bh_recent + a * (n_cpu * self.expert_bytes / (cpu_ms * 1e6))
        if n_fetch > 0 and pcie > 0:
            per = pcie / n_fetch
            self.pcie_per_exp_ewma = per if self.pcie_per_exp_ewma == 0 else (1 - a) * self.pcie_per_exp_ewma + a * per
            self.bp_recent = (1 - a) * self.bp_recent + a * (n_fetch * self.expert_bytes / (pcie * 1e6))
        g = self.ewma["gemm"]
        g[layer] = gemm if g[layer] == 0 else (1 - a) * g[layer] + a * gemm
        h = self.ewma["hit"]
        h[layer] = (1 - a) * h[layer] + a * (hits / max(1, x["n_unique"]))
        cpu_val = cpu_ms if cpu_ms == cpu_ms else 0.0
        self.hist[layer].append({"m": m, "hits": hits, "q": n_fetch, "wall": wall,
                                 "cpu": cpu_val, "gpu": pcie + gemm, "pcie": pcie,
                                 "gemm": gemm, "exposed": exposed,
                                 "h2d_mb": n_fetch * self.expert_mb})
        routed = self.routed[layer]
        if len(routed) == routed.maxlen:
            self.freq32[layer].subtract(routed[0])
        routed.append(cur["uniq"])
        self.freq32[layer].update(cur["uniq"])
        for e in cur["uniq"]:
            self.last_seen[layer][e] = self.tok

        self.policy.observe(layer, wall)
        if self.writer is not None:
            row = {k: v for k, v in x.items() if not k.startswith("_")}
            ctl = self.ctrl
            row.update({
                "episode": ctl.get("episode"), "phase": ctl.get("phase"),
                "policy": ctl.get("policy_name"), "workload": ctl.get("workload"),
                "cache_rate": ctl.get("cache_rate"), "problem": ctl.get("problem"),
                "seed": ctl.get("seed"), "t_abs": f"{self.t0:.6f}",
                "token_wall_ms": round(self.token_wall_ms, 4) if layer == 0 else "",
                "q": cur["q"], "q_native": cur["q_native"], "n_fetch": n_fetch,
                "wall_ms": round(wall, 4), "wall_net_ms": round(wall - 1000 * self.decide_s, 4),
                "cpu_task_ms": round(cpu_ms, 4) if cpu_ms == cpu_ms else "",
                "pcie_ms": round(pcie, 4), "gemm_ms": round(gemm, 4), "pre_ms": round(pre, 4),
                "exposed_ms": round(exposed, 4), "decide_us": round(1e6 * self.decide_s, 1),
                "n_evict": n_evict, "occ_post": self.occ,
                "routed_ids": ";".join(map(str, cur["uniq"])),
                "missing_ids": ";".join(map(str, cur["missing"])),
                "fetched_ids": ";".join(map(str, fetched)),
                "evicted_gids": ";".join(map(str, evicted)),
            })
            for k, v in row.items():
                if isinstance(v, float):
                    row[k] = round(v, 5)
            self.writer.writerow(row)
            if layer == self.L - 1:
                self.out_file.flush()

    # -- features --------------------------------------------------------------
    def _features(self, layer: int, top_k: int, uniq: list, missing: list, slots: list) -> dict:
        c, G = self.cache, self.groups
        m, n_uniq = len(missing), len(uniq)
        hits = n_uniq - m
        x = dict.fromkeys(PRE_ACTION, 0.0)
        x.update({"m": float(m), "bp_gbps": self.bp, "bh_gbps": self.bh,
                  "layer_id": float(layer), "tok": float(self.tok), "top_k": float(top_k),
                  "n_unique": float(n_uniq), "n_hits": float(hits),
                  "miss_ratio": m / max(1, n_uniq)})
        hist = self.hist[layer]
        prev = hist[-1] if hist else None
        if "G2_cpu" in G:
            x["cpu_ms_per_exp_ewma"] = self.ewma["cpu_ms_per_exp"][layer]
            x["cpu_ms_prev"] = prev["cpu"] if prev else 0.0
            x["bh_recent_gbps"] = self.bh_recent
            x["cpu_threads"] = self.cpu_threads
        if "G3_gpu" in G:
            x["gemm_ms_prev"] = prev["gemm"] if prev else 0.0
            x["gemm_ms_ewma"] = self.ewma["gemm"][layer]
        if "G4_transfer" in G:
            x["pcie_ms_per_exp_ewma"] = self.pcie_per_exp_ewma
            x["bp_recent_gbps"] = self.bp_recent
            x["expert_mb"] = self.expert_mb
            x["q_prev"] = float(prev["q"]) if prev else 0.0
            x["h2d_mb_prev"] = prev["h2d_mb"] if prev else 0.0
        if "G5_overlap" in G and prev:
            tc, tg = prev["cpu"], prev["gpu"]
            x["tcpu_prev"], x["tgpu_prev"] = tc, tg
            x["imbalance_prev"] = abs(tc - tg) / max(tc, tg, 1e-6)
            x["exposed_cpu_prev"] = prev["exposed"]
            x["wall_prev"] = prev["wall"]
            x["tcpu_over_tgpu_prev"] = min(50.0, tc / max(tg, 1e-3))
        self._ids_before = None
        if "G6_cache" in G:
            ids_before = c.id_of_slot.tolist()
            self._ids_before = ids_before
            usage = c.usage.tolist()
            step = int(c.step.item())
            occupied = [u for u, i in zip(usage, ids_before) if i >= 0]
            occ = len(occupied)
            self.occ = occ
            ages = [(step - u) / self.L for u in occupied]
            x["occ"], x["cap"] = float(occ), float(self.cap)
            x["free_slots"] = float(self.cap - occ)
            x["occ_frac"] = occ / self.cap
            x["layer_resident"] = float(sum(1 for s in slots if s >= 0))
            x["ins_w32"] = float(sum(self.ins_hist))
            x["evi_w32"] = float(sum(self.evi_hist))
            x["churn_w32"] = x["evi_w32"] / max(1.0, float(self.cap))
            x["lru_age_tok"] = max(ages) if ages else 0.0
            x["mean_age_tok"] = sum(ages) / len(ages) if ages else 0.0
            x["hit_ewma"] = self.ewma["hit"][layer]
        ge = {}
        if "G7_reuse" in G:
            f32, seen, routed = self.freq32[layer], self.last_seen[layer], self.routed[layer]
            recs = [min(REC_CAP, self.tok - seen[e]) if e in seen else REC_CAP for e in missing]
            fr32 = [f32.get(e, 0) for e in missing]
            last8 = list(routed)[-8:]
            c8 = Counter(e for t in last8 for e in t)
            fr8 = [c8.get(e, 0) for e in missing]
            if missing:
                x["miss_rec_min"], x["miss_rec_mean"] = min(recs), sum(recs) / m
                x["miss_freq32_mean"], x["miss_freq32_max"] = sum(fr32) / m, float(max(fr32))
                x["miss_freq8_mean"] = sum(fr8) / m
                x["n_miss_seen8"] = float(sum(1 for v in fr8 if v > 0))
            x["distinct8"] = float(len(c8))
            live = [v for v in f32.values() if v > 0]
            x["distinct32"] = float(len(live))
            tot = sum(live)
            if tot:
                x["entropy32"] = -sum(v / tot * math.log(v / tot) for v in live)
                x["top4_share32"] = sum(sorted(live, reverse=True)[:4]) / tot
            for thr in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
                ge[thr] = sum(1 for v in fr32 if v >= thr)
        x["_n_miss_freq_ge"] = ge
        x["_miss_f32"] = [self.freq32[layer].get(e, 0) for e in missing]
        x["_window"] = float(len(self.routed[layer]))
        if "G8_util" in G:
            for k in ("cpu_util", "gpu_util", "gpu_mem_util", "load1"):
                x[k] = self.sys[k]
        if "G9_memory" in G:
            x["gpu_free_gb"], x["host_avail_gb"] = self.sys["gpu_free_gb"], self.sys["host_avail_gb"]
            x["kv_pos"] = float(self.tok)
        if "G10_history" in G and hist:
            items = list(hist)
            for n in HIST_WINDOWS:
                w = items[-n:]
                for s in HIST_SIGNALS:
                    x[f"hist{n}_{s}"] = sum(r[s] for r in w) / len(w)
        return x


def build(cache) -> RuntimeQ:
    return RuntimeQ(cache)
