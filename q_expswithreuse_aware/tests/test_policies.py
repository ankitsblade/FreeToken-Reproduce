"""CPU-only checks of the policy implementations used by the study (no GPU, no server).

The engine references are FreeToken's own CPU mirror of the hybrid kernel
(freetoken.moe.offload_kernels._ensure_experts_hybrid_cpu, bit-identical to the GPU path per the
upstream kernel tests). The rule references below are written from docs/POLICY_SPEC.md, not
copied from the hook, so a mismatch means the hook and the specification disagree.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "q_expswithreuse_aware" / "policy"))
import qexp_hook  # noqa: E402
from reuse_aware_v2_hook import LEAN_COLUMNS, BranchFit, FractionMin1, ReuseAwareV2, ReuseVariant  # noqa: E402
from runtime_q_hook import Fraction, ReuseAware, round_balance  # noqa: E402

from freetoken.moe import offload_kernels  # noqa: E402

# (top_k, experts, B_P, B_H) as in the established plans
MODELS = {"gpt_oss": (4, 32, 6.7, 31.76), "qwen": (8, 256, 6.71, 31.38), "gemma": (8, 128, 6.71, 36.83)}
PRIOR_N = 8.0
T_FREE = 32.0


def lean_x(rng: random.Random, k: int, E: int, m: int) -> dict:
    n = float(rng.randint(0, 32))
    f32 = [rng.randint(0, int(n)) for _ in range(m)]
    prior = k / E
    return {
        "m": float(m), "n_hits": float(k - m), "_window": n, "_miss_f32": f32,
        "_p_miss": [(f + prior * PRIOR_N) / (n + PRIOR_N) for f in f32],
        "free_slots": float(rng.choice([0, 0, 0, 1, 2, 3, 5, 40])),
        "lru_age_tok": rng.choice([0.0, rng.uniform(0.05, 80.0)]),
        "_victim_p": [rng.random() * 0.4 for _ in range(m)], "_victims": [],
        "_alpha": rng.uniform(0.3, 3.5),
        "_cost": (rng.random(), rng.uniform(0.3, 3.0), rng.random() * 0.2, rng.uniform(0.05, 1.0)),
        "_vhit": rng.uniform(0.05, 1.0),
    }


# ---------------------------------------------------------------- specification references
def ref_p(x: dict, k: int, E: int) -> list[float]:
    prior = k / E
    return sorted(((f + prior * PRIOR_N) / (x["_window"] + PRIOR_N) for f in x["_miss_f32"]), reverse=True)


def ref_T(x: dict, j: int) -> float:
    return T_FREE if x["free_slots"] > j else x["lru_age_tok"]


def ref_delta(m: int, j: int, rho: float) -> float:
    return max((j + 1) / rho, m - j - 1) - max(j / rho, m - j)


def ref_C(cost: tuple, m: int, j: int) -> float:
    g0, tp, c0, tc = cost
    return max(g0 + j * tp, (c0 + (m - j) * tc) if m > j else 0.0)


def ref_policy(m: int, x: dict, k: int, E: int, rho: float, v1: bool, v2: bool, v3: bool) -> int:
    """POLICY_SPEC section 4: fetch the j-th candidate while value_j >= marginal cost_j."""
    p = ref_p(x, k, E)
    q = 0
    while q < m:
        r = p[q] - (x["_victim_p"][q] if v3 else 0.0)
        value = (x["_alpha"] if v2 else 1.0) * r * ref_T(x, q) * (x["_vhit"] if v1 else 1.0)
        cost = ref_C(x["_cost"], m, q + 1) - ref_C(x["_cost"], m, q) if v1 else ref_delta(m, q, rho)
        if value < cost:
            break
        q += 1
    return q


def flags(v1: bool, v2: bool, v3: bool) -> dict:
    s = {"kind": "reuse_aware_v"}
    if v1:
        s["cost"] = "runtime"
    if v2:
        s["reuse"] = "calibrated"
    if v3:
        s["evict"] = True
    return s


# ---------------------------------------------------------------- Policy B equivalence
@pytest.mark.parametrize("model", MODELS)
def test_all_variants_off_is_original_reuse_aware(model):
    k, E, bp, bh = MODELS[model]
    rng = random.Random(1)
    b = ReuseAware({"kind": "reuse_aware", "kappa": 1.0}, bp / bh, k, E)
    off = ReuseVariant({"kind": "reuse_aware_v"}, bp / bh, k, E)
    explicit_off = ReuseVariant({"kind": "reuse_aware_v", "cost": "paper", "reuse": "window", "evict": False},
                                bp / bh, k, E)
    n_fetching = 0
    for _ in range(20000):
        m = rng.randint(0, k)
        x = lean_x(rng, k, E, m)
        qb = b.decide(m, x, 0)
        assert off.decide(m, x, 0) == qb
        assert explicit_off.decide(m, x, 0) == qb
        assert ref_policy(m, x, k, E, bp / bh, False, False, False) == qb
        n_fetching += qb > 0
    assert n_fetching > 1000  # the comparison exercises fetching decisions, not only q = 0


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("v", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
def test_variants_match_specification(model, v):
    k, E, bp, bh = MODELS[model]
    rng = random.Random(hash((model, v)) & 0xFFFF)
    pol = ReuseVariant(flags(*map(bool, v)), bp / bh, k, E)
    for _ in range(5000):
        m = rng.randint(0, k)
        x = lean_x(rng, k, E, m)
        assert pol.decide(m, x, 0) == ref_policy(m, x, k, E, bp / bh, *map(bool, v))


@pytest.mark.parametrize("model", MODELS)
def test_v1_with_bandwidth_costs_reduces_to_b(model):
    """Dimension check: V1 with zero intercepts, t_p = S/B_P, t_c = S/B_H and v_hit = one CPU expert
    (S/B_H) is B's rule scaled by S/B_H on both sides. Only exact ties may differ (float rounding)."""
    k, E, bp, bh = MODELS[model]
    rng = random.Random(3)
    b = ReuseAware({"kind": "reuse_aware", "kappa": 1.0}, bp / bh, k, E)
    v1 = ReuseVariant({"kind": "reuse_aware_v", "cost": "runtime"}, bp / bh, k, E)
    s = 1.7  # ms per CPU expert, arbitrary
    differ = 0
    for _ in range(20000):
        m = rng.randint(0, k)
        x = lean_x(rng, k, E, m)
        x["_cost"], x["_vhit"] = (0.0, s * bh / bp, 0.0, s), s
        qb, qv = b.decide(m, x, 0), v1.decide(m, x, 0)
        if qb != qv:
            j = min(qb, qv)
            gap = ref_p(x, k, E)[j] * ref_T(x, j) - ref_delta(m, j, bp / bh)
            assert abs(gap) < 1e-9
            differ += 1
    assert differ < 20


def test_rule_boundaries():
    k, E, bp, bh = MODELS["qwen"]
    b = ReuseAware({"kind": "reuse_aware", "kappa": 1.0}, bp / bh, k, E)
    x = lean_x(random.Random(0), k, E, 0)
    assert b.decide(0, x, 0) == 0  # zero misses
    x = {**lean_x(random.Random(0), k, E, k), "free_slots": 0.0, "lru_age_tok": 0.0}
    # full cache with a victim used this token: T = 0, value 0, so B fetches only free fetches
    q = b.decide(k, x, 0)
    assert all(ref_delta(k, j, bp / bh) <= 0 for j in range(q))
    assert q == sum(1 for j in range(k) if ref_delta(k, j, bp / bh) <= 0)


# ---------------------------------------------------------------- q* against the engine
def make_cache(L: int, E: int, cap: int) -> object:
    class C:
        pass
    c = C()
    c.num_experts, c.cache_size, c.num_layers = E, cap, L
    c.slot_for_id = torch.full((L, E), -1, dtype=torch.int32)
    c.id_of_slot = torch.full((cap,), -1, dtype=torch.int32)
    c.usage = torch.zeros(cap, dtype=torch.int64)
    c.step = torch.zeros(1, dtype=torch.int64)
    c.active_mask = torch.zeros(E, dtype=torch.int32)
    c.evict_slots = torch.zeros(E, dtype=torch.int64)
    c.src_indices = torch.zeros(E, dtype=torch.int64)
    c.num_indices = torch.zeros(1, dtype=torch.int64)
    c.num_missing_full = torch.zeros(1, dtype=torch.int64)
    c.expert_recency = torch.full((L, E), -1, dtype=torch.int64)
    return c


@pytest.mark.parametrize("model", MODELS)
def test_qstar_rounding_matches_engine(model):
    k, E, bp, bh = MODELS[model]
    for frac in (bp / bh, 6.71 / 32.9, 0.1541, 0.5, 0.9):
        q16 = min(1 << 16, max(0, round(frac * (1 << 16))))
        for m in range(0, k + 1):
            c = make_cache(1, E, 4 * E)
            ids = torch.arange(m, dtype=torch.int64)
            offload_kernels._ensure_experts_hybrid_cpu(c, 0, ids.clone(), E, q16)
            assert int(c.num_indices.item()) == round_balance(frac, m) == Fraction(frac).decide(m, {}, 0)
            assert FractionMin1(frac).decide(m, {}, 0) == (max(1, round_balance(frac, m)) if m else 0)


def test_qstar_paper_fraction_choice_is_inert_on_qwen():
    """AUDIT 1.3: overlap pair (6.71/31.38) and standalone pair (6.71/32.9) give the same integer q."""
    for m in range(0, 9):
        assert round_balance(6.71 / 31.38, m) == round_balance(6.71 / 32.9, m)


# ---------------------------------------------------------------- V3 victim prediction
def bare_hook(cache, L: int, E: int, k: int) -> qexp_hook.QExpHook:
    h = object.__new__(qexp_hook.QExpHook)
    h.cache, h.cap, h.L, h.E, h.top_k, h.prior = cache, cache.cache_size, L, E, k, k / E
    h.routed = [deque(maxlen=32) for _ in range(L)]
    h.freq32 = [Counter() for _ in range(L)]
    h.hits_done = h.pred_done = 0.0
    h.fit_g, h.fit_c = BranchFit(1.0), BranchFit(0.2)
    h.cost, h.v_hit = (0.0, 1.0, 0.0, 0.2), 0.2
    return h


@pytest.mark.parametrize("seed", range(40))
def test_victim_prediction_matches_engine(seed):
    rng = random.Random(seed)
    L, E, k = 4, 16, 4
    cap = rng.choice([2 * E, 3 * E])
    c = make_cache(L, E, cap)
    # random history through the engine's own kernel so usage stamps and free slots are realistic
    for _ in range(rng.randint(0, 60)):
        layer = rng.randrange(L)
        ids = torch.tensor(rng.sample(range(E), k), dtype=torch.int64)
        offload_kernels._ensure_experts_hybrid_cpu(c, layer, ids, rng.randint(0, k), 0)
    layer = rng.randrange(L)
    routed = rng.sample(range(E), k)
    slots = c.slot_for_id[layer].tolist()
    missing = [e for e in routed if slots[e] < 0]
    h = bare_hook(c, L, E, k)
    x = h._lean_features(layer, routed, missing, slots)
    before = c.id_of_slot.tolist()
    offload_kernels._ensure_experts_hybrid_cpu(c, layer, torch.tensor(routed, dtype=torch.int64), len(missing), 0)
    n = int(c.num_indices.item())
    assert n == len(missing)
    engine_victims = [before[int(s)] for s in c.evict_slots[:n].tolist()]
    assert x["_victims"][:n] == engine_victims


def test_frozen_victim_prediction_breaks_ties_arbitrarily():
    """Documents audit finding F7: the frozen ReuseAwareV2 prediction differs from the engine
    when the k-th smallest usage stamp is tied; the corrected one never does (test above)."""
    rng = random.Random(1)
    L, E, k = 4, 16, 4
    cap = rng.choice([2 * E, 3 * E])
    c = make_cache(L, E, cap)
    for _ in range(rng.randint(0, 60)):
        layer = rng.randrange(L)
        ids = torch.tensor(rng.sample(range(E), k), dtype=torch.int64)
        offload_kernels._ensure_experts_hybrid_cpu(c, layer, ids, rng.randint(0, k), 0)
    layer = rng.randrange(L)
    routed = rng.sample(range(E), k)
    slots = c.slot_for_id[layer].tolist()
    missing = [e for e in routed if slots[e] < 0]
    h = bare_hook(c, L, E, k)
    frozen = ReuseAwareV2._lean_features(h, layer, routed, missing, slots)["_victims"]
    fixed = h._lean_features(layer, routed, missing, slots)
    before = c.id_of_slot.tolist()
    offload_kernels._ensure_experts_hybrid_cpu(c, layer, torch.tensor(routed, dtype=torch.int64), len(missing), 0)
    engine = [before[int(s)] for s in c.evict_slots[:int(c.num_indices.item())].tolist()]
    assert fixed["_victims"] == engine != frozen
    assert fixed["_victim_fix"] == sum(a != b for a, b in zip(engine, frozen)) > 0


def test_engine_victim_slots_order():
    u = np.array([5, 3, 3, 9, 3, 1, 7], dtype=np.int64)
    for k in range(1, 8):
        want = sorted(range(len(u)), key=lambda s: (u[s], s))[:k]
        assert qexp_hook.engine_victim_slots(u, k).tolist() == want


# ---------------------------------------------------------------- V1 fit and value of a hit
def test_branch_fit_sufficiency_and_least_squares():
    f = BranchFit(0.5)
    for i in range(7):
        f.add(float(i % 3), 1.0 + 2.0 * (i % 3))
    assert f.line() == (0.0, 0.5)  # 7 samples: prior
    f.add(2.0, 5.0)
    a, b = f.line()
    assert math.isclose(b, 2.0) and math.isclose(a, 1.0)
    g = BranchFit(0.5)
    for _ in range(20):
        g.add(1.0, 3.0)
    assert g.line() == (0.0, 0.5)  # one distinct x: prior
    h = BranchFit(0.5)
    for i in range(20):
        h.add(float(i % 4), 5.0 - i % 4)
    assert h.line() == (0.0, 0.5)  # negative slope: prior
    rng = np.random.default_rng(0)
    xs = rng.integers(0, 6, 200).astype(float)
    ys = 0.3 + 0.7 * xs + rng.normal(0, 0.05, 200)
    r = BranchFit(9.0)
    for xv, yv in zip(xs, ys):
        r.add(xv, yv)
    b_np, a_np = np.polyfit(xs, ys, 1)
    a, b = r.line()
    assert math.isclose(a, a_np, rel_tol=1e-6) and math.isclose(b, b_np, rel_tol=1e-6)


def test_v_hit_is_expected_cpu_branch_saving():
    h = object.__new__(ReuseAwareV2)
    h.fit_g, h.fit_c = BranchFit(1.8), BranchFit(0.4)
    for i in range(10):
        h.fit_g.add(float(i % 3), 0.3 + 1.8 * (i % 3))
        h.fit_c.add(float(1 + i % 4), 0.05 + 0.4 * (1 + i % 4))
    h.m_count = [0, 5, 7, 2, 1]
    h._refit()
    g0, tp, c0, tc = h.cost

    def c(n):
        return max(g0, c0 + n * tc) if n > 0 else max(g0, 0.0)
    want = sum(nm * (c(m) - c(m - 1)) for m, nm in enumerate(h.m_count) if m) / sum(h.m_count[1:])
    assert math.isclose(h.v_hit, want)
    assert h.v_hit <= tc + max(0.0, c0 - g0) + 1e-12  # at most one CPU expert (plus the c0 step at m = 1)


# ---------------------------------------------------------------- added counters
def test_remiss_counter_and_columns():
    assert QX_IS_SUPERSET()
    L, E, k = 2, 8, 4
    h = bare_hook(make_cache(L, E, 2 * E), L, E, k)
    h.calib_n, h.evicted_tok, h.qx, h.tok = 0, {}, {}, 10
    x = {"_fit_state": (3, 9, 1, 0), "_cost": (0.1, 1.5, 0.0, 0.4), "_victim_fix": 0}
    h.cur = {"missing": [1, 2], "x": x, "q": 1}
    h._after_layer(0, 2, 1, 1, 2, 1.0, 0.2, 0.4, [1 * E + 5, 0 * E + 3], [4])
    assert h.qx["n_remiss"] == 0 and h.calib_n == 1
    assert math.isclose(h.qx["c_pred_ms"], max(0.1 + 1.5, 0.0 + 1 * 0.4))
    h.tok = 14
    h.cur = {"missing": [3, 6], "x": x, "q": 0}
    h._after_layer(0, 2, 0, 2, 2, 0.0, 0.2, 0.8, [], [])
    assert h.qx["n_remiss"] == 1 and h.qx["remiss_tok"] == "4"
    assert h._extra_row(2, x, 0) is h.qx


def QX_IS_SUPERSET() -> bool:
    cols = qexp_hook.QX_COLUMNS
    return cols[:len(LEAN_COLUMNS)] == LEAN_COLUMNS and len(set(cols)) == len(cols)
