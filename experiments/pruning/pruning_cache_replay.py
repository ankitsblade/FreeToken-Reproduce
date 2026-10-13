"""Global cache replay mirroring native hybrid recency admission and eviction."""
from __future__ import annotations

import heapq
import math

import numpy as np


class NativeHybridReplay:
    def __init__(self, layers, experts, slots, fraction, initial=None):
        if slots < 1 or not 0 <= fraction <= 1:
            raise ValueError("invalid cache capacity or fetch fraction")
        self.layers, self.experts, self.slots = layers, experts, slots
        self.frac_q16 = min(65536, max(0, round(fraction * 65536)))
        if initial is None:
            self.slot_for_id = np.full((layers, experts), -1, dtype=np.int64)
            self.id_of_slot = np.full(slots, -1, dtype=np.int64)
            self.usage = np.zeros(slots, dtype=np.int64)
            self.recency = np.full((layers, experts), -1, dtype=np.int64)
            self.step = 0
        else:
            self.slot_for_id = np.asarray(initial["slot_for_id"], dtype=np.int64).copy()
            self.id_of_slot = np.asarray(initial["id_of_slot"], dtype=np.int64).copy()
            self.usage = np.asarray(initial["usage"], dtype=np.int64).copy()
            self.recency = np.asarray(initial["expert_recency"], dtype=np.int64).copy()
            self.step = int(initial["step"])
            if self.slot_for_id.shape != (layers, experts) or len(self.id_of_slot) != slots:
                raise ValueError("initial cache geometry mismatch")
        self.heap = [(int(age), slot) for slot, age in enumerate(self.usage)]
        heapq.heapify(self.heap)

    def event(self, layer, ids):
        active = list(dict.fromkeys(map(int, ids)))
        if any(e < 0 or e >= self.experts for e in active):
            raise ValueError("original expert ID outside cache geometry")
        slots_before = [int(self.slot_for_id[layer, expert]) for expert in active]
        self.step += 1
        missing = []
        for expert, slot in zip(active, slots_before):
            if slot >= 0:
                self.usage[slot] = self.step
                heapq.heappush(self.heap, (self.step, slot))
            else:
                missing.append(expert)
        missing.sort(key=lambda e: (-self.recency[layer, e], e))
        m = len(missing)
        low = m * self.frac_q16 >> 16
        cost = lambda q: max(q * (65536 - self.frac_q16), (m-q) * self.frac_q16)
        q = min(m, low if cost(low) <= cost(low+1) else low+1)
        victims, old_ids, admitted = [], [], missing[:q]
        for expert in admitted:
            while True:
                age, slot = heapq.heappop(self.heap)
                if int(self.usage[slot]) == age:
                    break
            old_id = int(self.id_of_slot[slot])
            if old_id >= 0:
                self.slot_for_id.reshape(-1)[old_id] = -1
            self.id_of_slot[slot] = layer*self.experts+expert
            self.slot_for_id[layer, expert] = slot
            self.usage[slot] = self.step
            heapq.heappush(self.heap, (self.step, slot))
            victims.append(slot)
            old_ids.append(old_id)
        for expert in active:
            self.recency[layer, expert] = self.step
        if len(self.heap) > 4*self.slots:
            self.heap = [(int(age), slot) for slot, age in enumerate(self.usage)]
            heapq.heapify(self.heap)
        return {"m": m, "q": q, "slots_before": slots_before, "admitted_ids": admitted,
                "evict_slots": victims, "evicted_flat_ids": old_ids, "step": self.step,
                "cpu_executions": m-q, "active_count": len(active), "hits": len(active)-m}


def project_state(initial, mask, slots):
    layers, experts = mask.shape
    projected = {"slot_for_id": np.full((layers, experts), -1, dtype=np.int64),
                 "id_of_slot": np.full(slots, -1, dtype=np.int64), "usage": np.zeros(slots, dtype=np.int64),
                 "expert_recency": np.where(mask, np.asarray(initial["expert_recency"]), -1),
                 "step": initial["step"]}
    survivors = [s for s, identity in enumerate(initial["id_of_slot"])
                 if identity >= 0 and mask.reshape(-1)[identity]]
    if slots == len(initial["id_of_slot"]):
        for slot in survivors:
            identity = initial["id_of_slot"][slot]
            projected["slot_for_id"].reshape(-1)[identity] = slot
            projected["id_of_slot"][slot] = identity
            projected["usage"][slot] = initial["usage"][slot]
    else:
        survivors.sort(key=lambda s: (-initial["usage"][s], s))
        for slot, old_slot in enumerate(survivors[:slots]):
            identity = initial["id_of_slot"][old_slot]
            projected["slot_for_id"].reshape(-1)[identity] = slot
            projected["id_of_slot"][slot] = identity
            projected["usage"][slot] = initial["usage"][old_slot]
    return projected


def replay_stream(ids, E, slots, expert_bytes, fraction, mask=None, initial=None):
    cache = NativeHybridReplay(ids.shape[1], E, slots, fraction, initial)
    rows = []
    occupancy = int((cache.id_of_slot >= 0).sum())
    occupancy_sum, occupancy_peak = 0, occupancy
    for token in ids:
        for layer, selected in enumerate(token):
            active = [e for e in selected if mask is None or mask[layer, e]]
            event = cache.event(layer, active)
            rows.append(event)
            occupancy += sum(identity < 0 for identity in event["evicted_flat_ids"])
            occupancy_sum += occupancy
            occupancy_peak = max(occupancy_peak, occupancy)
    if not len(ids):
        return None
    misses = sum(r["m"] for r in rows)
    fills = sum(r["q"] for r in rows)
    accesses = sum(r["active_count"] for r in rows)
    return {"mean_m": float(np.mean([r["m"] for r in rows])),
            "p95_m": float(np.quantile([r["m"] for r in rows], .95, method="higher")),
            "misses_per_token": misses/len(ids), "miss_rate_retained_demand": misses/accesses if accesses else None,
            "miss_rate_original_demand": misses/ids.size, "miss_demand_bytes_per_token": misses*expert_bytes/len(ids),
            "h2d_fill_bytes_per_token": fills*expert_bytes/len(ids), "fills_per_token": fills/len(ids),
            "cpu_executions_per_token": (misses-fills)/len(ids),
            "evictions_per_token": sum(sum(e >= 0 for e in r["evicted_flat_ids"]) for r in rows)/len(ids),
            "retained_active_k": accesses/len(rows), "original_accesses": int(ids.size),
            "retained_accesses": accesses, "removed_accesses": int(ids.size)-accesses,
            "mean_occupancy": occupancy_sum/len(rows), "peak_occupancy": occupancy_peak,
            "fetch_fraction": float(fraction), "fetch_fraction_q16": cache.frac_q16,
            "cache_slots": slots, "cache_bytes": slots*expert_bytes,
            "final_occupancy": int((cache.id_of_slot >= 0).sum())}


def validate_native(directory, model):
    import json
    from experiments.pruning.collect_moe_routes import json_lines

    initial = json.loads((directory / "decode_initial_cache.json").read_text())
    cache = NativeHybridReplay(model["layers"], model["experts"], initial["cache_slots"],
                               initial["fetch_fraction"], initial)
    events = 0
    for row in json_lines(directory / "cache.jsonl.gz"):
        expected = cache.event(row["layer_id"], row["raw_ids"])
        for field in ("m", "q", "slots_before", "admitted_ids", "evict_slots", "evicted_flat_ids", "step", "cpu_executions"):
            if expected[field] != row[field]:
                return {"status": "mismatch", "event": events, "field": field,
                        "expected": expected[field], "observed": row[field]}
        events += 1
    return {"status": "matched", "events": events, "fetch_fraction": initial["fetch_fraction"]}
