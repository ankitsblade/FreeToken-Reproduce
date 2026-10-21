"""Policy-equivalence and engine-replica tests for saketh_exp_2.

CPU tests: B + no V1/V2/V3 equals the original ReuseAware decision; each switch is inert when
its input is neutral; FractionMin1; fit_ok mirrors BranchFit's fallback. GPU tests (skipped
without CUDA) run the real hybrid ensure kernel on synthetic cache states and compare its fetch
count and evicted slots with the host-side replicas the hook uses.

Run: source scripts/env.sh && experiment_python -m pytest saketh_exp_2/tests -q
"""

from __future__ import annotations

import os
import random
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "policy"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments", "runtime_q", "instrumentation"))

from reuse_aware_v2_hook import BranchFit, FractionMin1, ReuseVariant  # noqa: E402
from runtime_q_hook import ReuseAware, round_balance  # noqa: E402
from sx2_hook import Sx2Hook, engine_victims, fit_ok  # noqa: E402

CASES = 20000
GEOMETRIES = [(4, 32), (8, 256), (8, 128)]  # gpt-oss, Qwen, Gemma (top_k, experts)


def random_x(rng: random.Random, top_k: int, m: int) -> dict:
    n = rng.randint(0, 32)
    return {
        "_window": float(n),
        "_miss_f32": [rng.randint(0, n) for _ in range(m)],
        "free_slots": float(rng.choice([0, 0, 0, rng.randint(0, 2 * top_k)])),
        "lru_age_tok": rng.choice([0.0, rng.uniform(0, 4), rng.uniform(0, 200)]),
        "_alpha": rng.uniform(0.2, 3.0),
        "_victim_p": [rng.uniform(0, 1) for _ in range(m)],
        "_cost": (rng.uniform(0, 1), rng.uniform(0.1, 3), rng.uniform(0, 1), rng.uniform(0.1, 1)),
        "_vhit": rng.uniform(0.05, 1.0),
        "cpu_ms_per_exp_ewma": 0.0, "pcie_ms_per_exp_ewma": 0.0,
    }


def test_variant_all_off_equals_original_reuse_aware():
    rng = random.Random(0)
    mismatches = 0
    for i in range(CASES):
        top_k, E = GEOMETRIES[i % 3]
        rho = rng.choice([6.71 / 31.38, 6.70 / 31.76, 6.71 / 36.83, rng.uniform(0.05, 1.5)])
        m = rng.randint(0, top_k)
        x = random_x(rng, top_k, m)
        b = ReuseAware({"kappa": 1.0}, rho, top_k, E).decide(m, x, 0)
        v = ReuseVariant({}, rho, top_k, E).decide(m, x, 0)
        mismatches += b != v
    assert mismatches == 0


@pytest.mark.parametrize("switch,neutral", [
    ({"reuse": "calibrated"}, {"_alpha": 1.0}),
    ({"evict": True}, "victims_zero"),
])
def test_switch_is_inert_on_neutral_input(switch, neutral):
    rng = random.Random(1)
    for i in range(CASES // 4):
        top_k, E = GEOMETRIES[i % 3]
        rho = rng.uniform(0.05, 1.5)
        m = rng.randint(0, top_k)
        x = random_x(rng, top_k, m)
        if neutral == "victims_zero":
            x["_victim_p"] = [0.0] * m
        else:
            x.update(neutral)
        assert ReuseVariant(switch, rho, top_k, E).decide(m, x, 0) == \
            ReuseAware({}, rho, top_k, E).decide(m, x, 0)


def test_fraction_min1_is_paper_minimum_fill():
    for f in (0.1728, 0.1896, 0.2040, 0.5, 1.0):
        pol = FractionMin1(f)
        assert pol.decide(0, {}, 0) == 0
        for m in range(1, 9):
            assert pol.decide(m, {}, 0) == max(1, round_balance(f, m))


def test_fit_ok_mirrors_branchfit_fallback():
    rng = random.Random(2)
    for _ in range(2000):
        fit = BranchFit(slope_prior=1.88)
        for _ in range(rng.randint(0, 20)):
            x = float(rng.choice([0, 1, 2, 3, 4] if rng.random() < 0.8 else [2]))
            fit.add(x, rng.uniform(-1, 1) + x * rng.uniform(-1, 3))
        assert fit_ok(fit) == (fit.line() != fit.prior)


def test_hook_builds_the_original_classes():
    h = object.__new__(Sx2Hook)
    h.bp, h.bh, h.E, h.top_k = 6.71, 31.38, 256, 8
    h.executor = types.SimpleNamespace(top_k=8)
    assert type(h._build({"kind": "reuse_aware", "kappa": 1.0}, 0)) is ReuseAware
    assert type(h._build({"kind": "reuse_aware_v"}, 0)) is ReuseVariant
    assert type(h._build({"kind": "fraction_min1", "fraction": 0.2}, 0)) is FractionMin1


def test_engine_victims_order_and_ties():
    usage = np.array([5, 0, 3, 0, 3, 9, 3], dtype=np.int64)
    assert engine_victims(usage, [], 4).tolist() == [1, 3, 2, 4]
    assert engine_victims(usage, [1, 2], 3).tolist() == [3, 4, 6]
    assert engine_victims(usage, [], 10).tolist() == [1, 3, 2, 4, 6, 0, 5]


# ----------------------------------------------------------------------------- GPU
torch = pytest.importorskip("torch")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def fake_cache(L: int, E: int, cap: int, rng: np.random.Generator, dev: str):
    """A synthetic cache state with heavy usage ties (several fills share one step)."""
    ids = np.full(cap, -1, dtype=np.int32)
    occ = rng.integers(cap // 2, cap + 1)
    gids = rng.choice(L * E, size=occ, replace=False)
    pos = rng.choice(cap, size=occ, replace=False)
    ids[pos] = gids
    usage = np.zeros(cap, dtype=np.int64)
    usage[pos] = rng.integers(1, 6, size=occ)  # few distinct stamps -> many ties
    slot_for_id = np.full((L, E), -1, dtype=np.int32)
    for s, g in zip(pos, gids):
        slot_for_id[g // E, g % E] = s
    plan = max(E, cap)
    t = lambda a, dt: torch.tensor(a, dtype=dt, device=dev)  # noqa: E731
    return types.SimpleNamespace(
        num_experts=E, cache_size=cap, slot_for_id=t(slot_for_id, torch.int32),
        id_of_slot=t(ids, torch.int32), usage=t(usage, torch.int64),
        step=torch.tensor(int(usage.max()) + 1, dtype=torch.int64, device=dev),
        active_mask=torch.zeros(E, dtype=torch.int32, device=dev),
        evict_slots=torch.full((plan,), -1, dtype=torch.int32, device=dev),
        src_indices=torch.full((plan,), -1, dtype=torch.int32, device=dev),
        num_indices=torch.zeros(1, dtype=torch.int64, device=dev),
        num_missing_full=torch.zeros(1, dtype=torch.int64, device=dev),
        expert_recency=t(rng.integers(-1, 50, size=(L, E)), torch.int64),
    )


@cuda
@pytest.mark.parametrize("L,E,top_k,cap", [(24, 32, 4, 77), (40, 256, 8, 512), (30, 128, 8, 384)])
def test_kernel_matches_host_replicas(L, E, top_k, cap):
    from freetoken.moe.offload_kernels import _ensure_experts_hybrid_gpu

    rng = np.random.default_rng(L * 1000 + cap)
    dev = "cuda"
    checks = victim_checks = 0
    for trial in range(300):
        c = fake_cache(L, E, cap, rng, dev)
        layer = int(rng.integers(0, L))
        experts = rng.choice(E, size=top_k, replace=False)
        ids = torch.tensor(experts, dtype=torch.int32, device=dev)
        slots_row = c.slot_for_id[layer].cpu().numpy()
        uniq = experts.tolist()
        m = sum(1 for e in uniq if slots_row[e] < 0)
        usage_before = c.usage.cpu().numpy().copy()
        ids_before = c.id_of_slot.cpu().numpy().copy()
        own = [int(slots_row[e]) for e in uniq if slots_row[e] >= 0]
        if trial % 2:
            f = float(rng.choice([0.1541, 0.1728, 0.1778, 0.1896, 0.2040, 0.2110]))
            frac_q16 = min(1 << 16, max(0, round(f * (1 << 16))))
            _ensure_experts_hybrid_gpu(c, layer, ids, cap, frac_q16)
            q_expected = round_balance(f, m)
        else:
            q_expected = int(rng.integers(0, m + 1))
            _ensure_experts_hybrid_gpu(c, layer, ids, q_expected, 0)
        torch.cuda.synchronize()
        q = int(c.num_indices.item())
        assert q == q_expected, (f"q {q} != {q_expected}", m)
        checks += 1
        changed = np.flatnonzero(c.id_of_slot.cpu().numpy() != ids_before)
        pred = engine_victims(usage_before, own, q)
        assert sorted(pred.tolist()) == sorted(changed.tolist()) or q == 0
        if q:
            victim_checks += 1
            assert c.evict_slots[:q].cpu().tolist() == pred.tolist()
    assert checks == 300 and victim_checks > 50
