"""Offline expert-cache simulation for P3/P4/P5/P9 access replay and diagnostics.

The default configuration (captured state, native fraction, global timestamp LRU,
protected active experts) reproduces the live runtime decisions that
m_reduction_replay.CacheReplay verifies. Other settings are explicit study arms.
"""

import math
from collections import Counter, deque

import numpy as np

from experiments.split_policy.policy_q import round_balance

BIG = np.iinfo(np.int64).max


def floor_q(q, m, floor):
    # Corrected arms use F(q, m) = 0 if m == 0 else min(m, max(1, q)).
    return min(m, max(q, floor)) if m else 0


class QRule:
    """Admission count and fetch order. Kinds: native, fixed, full, reuse."""

    def __init__(self, kind, fraction=None, rho=None, per_expert=True, top_k=None, experts=None,
                 t_free=32.0, prior_n=8, window=32, kappa=1.0):
        self.kind, self.fraction, self.rho, self.per_expert = kind, fraction, rho, per_expert
        self.top_k, self.experts = top_k, experts
        self.t_free, self.prior_n, self.window, self.kappa = t_free, prior_n, window, kappa
        if kind == "reuse" and not rho or kind in ("native", "fixed") and fraction is None:
            raise ValueError("Incomplete q rule: " + kind)

    def reset(self, layers):
        self.windows = [deque(maxlen=self.window) for _ in range(layers)]
        self.counts = [Counter() for _ in range(layers)]

    def decide(self, sim, layer, ids, missing):
        m = len(missing)
        if self.kind == "full":
            return missing, m
        if self.kind in ("native", "fixed"):
            return missing, round_balance(self.fraction, m)
        window, count = self.windows[layer], self.counts[layer]
        prior = self.top_k / self.experts
        if self.per_expert:
            p = {e: (count[e] + self.prior_n * prior) / (len(window) + self.prior_n) for e in missing}
        else:
            p = {e: prior for e in missing}
        recency = sim.recency[layer]
        missing = sorted(missing, key=lambda e: (-p[e], -recency[e], e))
        free, age = sim.residence(layer, ids)
        q = 0
        while q < m:
            delta = max((q + 1) / self.rho, m - q - 1) - max(q / self.rho, m - q)
            residence = self.t_free if free > q else age
            if p[missing[q]] * residence < self.kappa * delta:
                break
            q += 1
        if len(window) == window.maxlen:
            count.subtract(window[0])
        window.append(ids)
        count.update(ids)
        return missing, q


class CacheSim:
    """Global slot cache with LRU, per-layer quota, SLRU or pin-protected replacement."""

    def __init__(self, state, layers, experts, rule, floor, replacement="lru", quotas=None,
                 slru_fraction=None, pins=None):
        self.layers, self.experts = layers, experts
        self.slots = np.asarray(state["slot_for_id"], dtype=np.int64).copy()
        self.owners = np.asarray(state["id_of_slot"], dtype=np.int64).copy()
        self.usage = np.asarray(state["usage"], dtype=np.int64).copy()
        self.recency = np.asarray(state["expert_recency"], dtype=np.int64).copy()
        self.step = int(state["step"])
        self.capacity = len(self.owners)
        self.order = np.arange(self.capacity, dtype=np.int64)
        owned = self.owners[self.owners >= 0]
        self.count = np.bincount(owned // experts, minlength=layers).astype(np.int64)
        self.owner_layer = np.where(self.owners >= 0, self.owners // experts, -1)
        self.rule, self.floor, self.replacement = rule, int(floor), replacement
        self.quotas = None if quotas is None else np.asarray(quotas, dtype=np.int64)
        self.pinned = np.zeros(layers * experts, dtype=bool)
        if pins:
            for layer, expert in pins:
                self.pinned[layer * experts + expert] = True
        if replacement == "quota":
            if self.quotas is None or int(self.quotas.sum()) != self.capacity or (self.quotas < 0).any():
                raise ValueError("Quota vector must be nonnegative and sum to the slot count")
            if any(self.layer_count(l) > self.quotas[l] for l in range(layers)):
                raise ValueError("Initial state violates quotas; build a feasible seed first")
        self.protected_cap = int(math.floor(slru_fraction * self.capacity)) if replacement == "slru" else 0
        self.segment = np.zeros(self.capacity, dtype=np.int64)
        self.segment_time = self.usage.copy()
        rule.reset(layers)

    def layer_count(self, layer):
        return int(self.count[layer])

    def keys(self, layer, protected):
        # Composite victim order: smaller key is evicted first; BIG marks illegal slots.
        if self.replacement == "slru":
            key = self.segment * (BIG // 4) + self.segment_time * self.capacity + self.order
        else:
            key = self.usage * self.capacity + self.order
        key = key.copy()
        if protected:
            key[list(protected)] = BIG
        owned = self.owners >= 0
        if self.pinned.any():
            key[owned & self.pinned[np.where(owned, self.owners, 0)]] = BIG
        if self.replacement == "quota":
            free = ~owned
            mine = self.owner_layer == layer
            if self.layer_count(layer) < self.quotas[layer]:
                key[~free] = BIG
            else:
                key[~mine] = BIG
        return key

    def residence(self, layer, ids):
        # Free capacity and legal LRU victim age, converted to layer-token visits.
        active = {layer * self.experts + e for e in ids}
        illegal = {s for s in range(self.capacity) if self.owners[s] in active}
        if self.replacement == "quota":
            free = max(0, int(self.quotas[layer]) - self.layer_count(layer))
        else:
            free = int((self.owners < 0).sum())
        key = self.keys(layer, illegal)
        victim = int(np.argmin(key))
        if key[victim] == BIG:
            return free, 0.0
        return free, max(0.0, (self.step - self.usage[victim]) / self.layers)

    def event(self, layer, ids):
        ids = list(dict.fromkeys(ids))
        before = self.slots[layer, ids].tolist()
        missing = [e for e in ids if self.slots[layer, e] < 0]
        missing.sort(key=lambda e: (-self.recency[layer, e], e))
        order, q = self.rule.decide(self, layer, ids, missing)
        m = len(missing)
        q = floor_q(q, m, self.floor)
        self.step += 1
        hits = [s for s in before if s >= 0]
        for slot in hits:
            self.usage[slot] = self.step
            if self.replacement == "slru":
                self.segment_time[slot] = self.step
                if self.segment[slot] == 0:
                    self.promote(slot)
        protected = set(hits)
        admitted, evicted = [], []
        for expert in order[:q]:
            key = self.keys(layer, protected)
            victim = int(np.argmin(key))
            if key[victim] == BIG:
                break
            previous = int(self.owners[victim])
            if previous >= 0:
                self.slots.reshape(-1)[previous] = -1
                self.count[previous // self.experts] -= 1
            self.owners[victim] = layer * self.experts + expert
            self.owner_layer[victim] = layer
            self.count[layer] += 1
            self.slots[layer, expert] = victim
            self.usage[victim] = self.step
            self.segment[victim], self.segment_time[victim] = 0, self.step
            protected.add(victim)
            admitted.append(expert)
            evicted.append(previous)
        self.recency[layer, ids] = self.step
        return {"m": m, "q": len(admitted), "admitted": admitted, "evicted": evicted,
                "missing": missing, "cpu": m - len(admitted)}

    def promote(self, slot):
        self.segment[slot] = 1
        while int(self.segment.sum()) > self.protected_cap:
            members = np.flatnonzero(self.segment == 1)
            demoted = members[np.argmin(self.segment_time[members] * self.capacity + members)]
            self.segment[demoted] = 0
            self.segment_time[demoted] = self.step

    def state(self):
        return {"slot_for_id": self.slots.tolist(), "id_of_slot": self.owners.tolist(),
                "usage": self.usage.tolist(), "expert_recency": self.recency.tolist(), "step": self.step}


def empty_state(layers, experts, capacity):
    return {"slot_for_id": [[-1] * experts for _ in range(layers)], "id_of_slot": [-1] * capacity,
            "usage": [0] * capacity, "expert_recency": [[0] * experts for _ in range(layers)], "step": 0}


def residents(state, experts):
    owners, usage = state["id_of_slot"], state["usage"]
    rows = [(usage[s], s, o) for s, o in enumerate(owners) if o >= 0]
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [(o // experts, o % experts, u) for u, _, o in rows]


def project_state(state, layers, experts, capacity, quotas=None):
    """Keep the most recent residents within capacity (and per-layer quotas), preserving recency."""
    out = empty_state(layers, experts, capacity)
    out["expert_recency"] = [list(row) for row in state["expert_recency"]]
    out["step"] = int(state["step"])
    taken, per_layer, dropped = 0, Counter(), []
    for layer, expert, used in residents(state, experts):
        if taken == capacity or quotas is not None and per_layer[layer] >= quotas[layer]:
            dropped.append((layer, expert))
            continue
        out["slot_for_id"][layer][expert] = taken
        out["id_of_slot"][taken] = layer * experts + expert
        out["usage"][taken] = used
        per_layer[layer] += 1
        taken += 1
    return out, dropped


def initial_state(trace, geometry, capacity, start, quotas=None):
    if start == "cold":
        return empty_state(geometry.layers, geometry.experts, capacity)
    if start != "native":
        raise ValueError("Unknown start state: " + start)
    if capacity == len(trace.initial["id_of_slot"]) and quotas is None:
        return trace.initial
    return project_state(trace.initial, geometry.layers, geometry.experts, capacity, quotas)[0]


def run(trace, geometry, rule, floor, capacity, start="native", replacement="lru", quotas=None,
        slru_fraction=None, pins=None, keep_events=False, seed_quotas=None):
    """Replay one request's decode accesses; returns per-event m/q arrays and counters.

    seed_quotas projects the initial state per layer (P5 common feasible seed) independently
    of the replacement quotas; quota replacement defaults to seeding with its own quotas.
    """
    seed = seed_quotas if seed_quotas is not None else (quotas if replacement == "quota" else None)
    state = initial_state(trace, geometry, capacity, start, seed)
    sim = CacheSim(state, geometry.layers, geometry.experts, rule, floor, replacement, quotas,
                   slru_fraction, pins)
    m = np.zeros(len(trace.ids), dtype=np.int32)
    q = np.zeros(len(trace.ids), dtype=np.int32)
    evictions = 0
    events = [] if keep_events else None
    for i, (layer, ids) in enumerate(zip(trace.layer, trace.ids)):
        out = sim.event(int(layer), ids)
        m[i], q[i] = out["m"], out["q"]
        evictions += sum(1 for e in out["evicted"] if e >= 0)
        if keep_events:
            events.append(out)
    return {"m": m, "q": q, "evictions": evictions, "events": events, "final": sim.state()}


def verify_native(trace, geometry):
    """Exact native fidelity check against the live m/q stream of one capture."""
    rule = QRule("native", fraction=float(trace.initial["fetch_fraction"]))
    out = run(trace, geometry, rule, floor=1, capacity=geometry.cache_slots)
    return bool(np.array_equal(out["m"], trace.live_m) and np.array_equal(out["q"], trace.live_q))


def belady(trace, geometry, capacity, start="native", quotas=None):
    """Fully admitting MIN with active-expert protection; per-event miss counts and miss sets."""
    L, E = geometry.layers, geometry.experts
    state = initial_state(trace, geometry, capacity, start, quotas)
    flat = [[layer * E + e for e in ids] for layer, ids in zip(trace.layer.tolist(), trace.ids)]
    upcoming = {}
    nexts = [None] * len(flat)
    for i in range(len(flat) - 1, -1, -1):
        nexts[i] = {f: upcoming.get(f, math.inf) for f in flat[i]}
        for f in flat[i]:
            upcoming[f] = i
    future = {f: math.inf for f in range(L * E)}
    future.update(upcoming)
    resident = {o for o in state["id_of_slot"] if o >= 0}
    count = Counter(o // E for o in resident)
    misses = np.zeros(len(flat), dtype=np.int32)
    miss_sets = []
    for i, ids in enumerate(flat):
        layer = ids[0] // E
        active = set(ids)
        missing = [f for f in ids if f not in resident]
        for f in ids:
            future[f] = nexts[i][f]
        for f in missing:
            if quotas is not None:
                full = count[layer] >= quotas[layer]
                pool = [r for r in resident if r // E == layer and r not in active] if full else None
            else:
                full = len(resident) >= capacity
                pool = [r for r in resident if r not in active] if full else None
            if full:
                if not pool:
                    continue
                victim = max(pool, key=lambda r: (future[r], -r))
                resident.discard(victim)
                count[victim // E] -= 1
            resident.add(f)
            count[layer] += 1
        misses[i] = len(missing)
        miss_sets.append({f % E for f in missing})
    return misses, miss_sets


def stack_ranks(trace, geometry, start="native"):
    """Per-layer fully admitting LRU stack ranks; ties within an event evict lower expert IDs first.

    A decode access of expert e hits at partitioned quota c iff rank < c. Rank -1 marks a
    first touch absent from the initial state (a miss at every quota).
    """
    E = geometry.experts
    key = np.full((geometry.layers, E), -1, dtype=np.int64)
    if start == "native":
        for layer, expert, used in residents(trace.initial, E):
            key[layer, expert] = int(trace.initial["expert_recency"][layer][expert]) * (E + 1) + expert + 1
    clock = int(key.max() // (E + 1)) + 1 if (key >= 0).any() else 1
    ranks = []
    for layer, ids in zip(trace.layer.tolist(), trace.ids):
        row = key[layer]
        ranks.append([int((row > row[e]).sum()) if row[e] >= 0 else -1 for e in ids])
        for e in ids:
            row[e] = clock * (E + 1) + e + 1
        clock += 1
    return ranks


def layer_curves(trace, geometry, quotas_range, start="native"):
    """Fully admitting partitioned-LRU miss counts per layer for every quota in range."""
    ranks = stack_ranks(trace, geometry, start)
    lo, hi = quotas_range
    curves = np.zeros((geometry.layers, hi - lo + 1), dtype=np.int64)
    capacities = np.arange(lo, hi + 1)
    for layer, row in zip(trace.layer.tolist(), ranks):
        for rank in row:
            curves[layer] += (rank < 0) | (rank >= capacities)
    return curves


def global_reuse_distance(trace, geometry):
    """Distinct-expert reuse distance in the global decode stream (None for first touches)."""
    E = geometry.experts
    last = np.full(geometry.layers * E, -1, dtype=np.int64)
    out = []
    for i, (layer, ids) in enumerate(zip(trace.layer.tolist(), trace.ids)):
        flat = [layer * E + e for e in ids]
        out.append([int((last > last[f]).sum()) if last[f] >= 0 else None for f in flat])
        last[flat] = i
    return out


def uniform_quotas(layers, capacity):
    base, extra = divmod(capacity, layers)
    return [base + (1 if layer < extra else 0) for layer in range(layers)]


def capacity_regimes(geometry):
    total = geometry.layers * geometry.experts
    return {"high_miss_10": math.ceil(0.10 * total), "primary": geometry.cache_slots,
            "low_miss_40": math.ceil(0.40 * total)}
