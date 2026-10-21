"""Causal Reuse-Aware admission for the new eager m-reduction runtime.

Based on Saketh's ReuseAware rule in runtime_q/instrumentation/runtime_q_hook.py.
Unlike the historical logger, this hook selects the fetched expert identities too.
"""

import json
import os
from collections import Counter, deque


class ReusePolicy:
    def __init__(self, cache, control):
        self.control = control
        self.experts = cache.num_experts
        self.k = int(control["top_k"])
        self.prior = self.k / self.experts
        self.rho = float(control["rho"])
        if self.rho <= 0:
            raise ValueError("Reuse-Aware requires positive calibrated rho")
        self.reset(cache)

    def reset(self, cache):
        self.windows = [deque(maxlen=32) for _ in range(cache.num_layers)]
        self.counts = [Counter() for _ in range(cache.num_layers)]
        cache.m_reduction_priority = None
        cache.m_reduction_decision = None

    def decide(self, cache, layer, ids):
        import torch

        active = list(dict.fromkeys(ids.reshape(-1).tolist()))
        slots = cache.slot_for_id[layer].tolist()
        missing = [expert for expert in active if slots[expert] < 0]
        m = len(missing)
        owners = cache.id_of_slot.tolist()
        usage = cache.usage.tolist()
        recency = cache.expert_recency[layer].tolist()
        active_ids = {layer * self.experts + expert for expert in active}
        legal = [slot for slot, owner in enumerate(owners) if owner not in active_ids]
        step = int(cache.step.item())
        free = sum(owner < 0 for owner in owners)
        oldest = min(legal, key=lambda slot: (usage[slot], slot)) if legal else None
        age = max(0.0, (step - usage[oldest]) / cache.num_layers) if oldest is not None else 0.0
        window = self.windows[layer]
        count = self.counts[layer]
        if self.control.get("per_expert", True):
            probabilities = {expert: (count[expert] + 8 * self.prior) / (len(window) + 8)
                             for expert in missing}
        else:
            probabilities = {expert: self.prior for expert in missing}
        missing.sort(key=lambda expert: (-probabilities[expert], -recency[expert], expert))
        q = 0
        while q < m:
            delta = (max((q + 1) / self.rho, m - q - 1)
                     - max(q / self.rho, m - q))
            residence = 32.0 if free > q else age
            if probabilities[missing[q]] * residence < delta:
                break
            q += 1
        floor = int(self.control["fill_floor"])
        chosen_q = min(m, max(q, floor)) if m else 0
        if cache.m_reduction_priority is None:
            cache.m_reduction_priority = torch.zeros(self.experts, dtype=torch.int64,
                                                     device=cache.device)
        ranks = [0] * self.experts
        for rank, expert in enumerate(missing):
            ranks[expert] = self.experts - rank
        cache.m_reduction_priority.copy_(torch.tensor(ranks, dtype=torch.int64,
                                                     device=cache.device))
        cache.m_reduction_decision = {"raw_q": q, "q": chosen_q,
                                     "candidate_ids": missing, "rho": self.rho,
                                     "window": len(window), "free_slots": free,
                                     "victim_age_token_visits": age}
        if len(window) == window.maxlen:
            count.subtract(window[0])
        window.append(active)
        count.update(active)
        return chosen_q, 0.0


class FractionPolicy:
    """P4 paper/fixed arms: the native rounding rule with a declared fraction, native fetch order."""

    def __init__(self, control):
        self.fraction = float(control["fraction"])
        if not 0 < self.fraction <= 1:
            raise ValueError("Fixed admission fraction must be in (0, 1]")

    def reset(self, cache):
        cache.m_reduction_priority = None
        cache.m_reduction_decision = None

    def decide(self, cache, layer, ids):
        return -1, self.fraction


def before_hybrid(cache, layer, ids, max_fetch, fraction):
    path = os.getenv("FREETOKEN_M_REDUCTION_POLICY")
    if not path:
        return max_fetch, fraction
    if not hasattr(cache, "m_reduction_policy"):
        with open(path) as stream:
            control = json.load(stream)
        if control["kind"] == "reuse_aware":
            cache.m_reduction_policy = ReusePolicy(cache, control)
        elif control["kind"] == "fraction":
            cache.m_reduction_policy = FractionPolicy(control)
        else:
            raise ValueError("Unsupported m-reduction admission policy")
    return cache.m_reduction_policy.decide(cache, layer, ids)


def reset_policy(cache):
    policy = getattr(cache, "m_reduction_policy", None)
    if policy is not None:
        policy.reset(cache)
