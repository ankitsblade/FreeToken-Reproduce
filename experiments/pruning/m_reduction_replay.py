"""Qualify new Stage 0 traces against protected-active timestamp LRU replay."""

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json
from experiments.split_policy.policy_q import round_balance


class CacheReplay:
    def __init__(self, initial, control):
        self.slots = np.asarray(initial["slot_for_id"], dtype=np.int64).copy()
        self.owners = np.asarray(initial["id_of_slot"], dtype=np.int64).copy()
        self.usage = np.asarray(initial["usage"], dtype=np.int64).copy()
        self.recency = np.asarray(initial["expert_recency"], dtype=np.int64).copy()
        self.step = int(initial["step"])
        self.layers, self.experts = self.slots.shape
        self.control = control
        self.fraction = float(initial["fetch_fraction"])
        self.windows = [deque(maxlen=32) for _ in range(self.layers)]
        self.counts = [Counter() for _ in range(self.layers)]

    def event(self, layer, ids):
        ids = list(dict.fromkeys(ids))
        before = self.slots[layer, ids].tolist()
        missing = [expert for expert in ids if self.slots[layer, expert] < 0]
        missing.sort(key=lambda expert: (-self.recency[layer, expert], expert))
        m = len(missing)
        active = {layer * self.experts + expert for expert in ids}
        if self.control["kind"] == "reuse_aware":
            window, count = self.windows[layer], self.counts[layer]
            prior = self.control["top_k"] / self.experts
            p = {expert: (count[expert] + 8 * prior) / (len(window) + 8) for expert in missing}
            if not self.control.get("per_expert", True):
                p = {expert: prior for expert in missing}
            missing.sort(key=lambda expert: (-p[expert], -self.recency[layer, expert], expert))
            legal = np.flatnonzero(~np.isin(self.owners, list(active)))
            victim = min(legal, key=lambda slot: (self.usage[slot], slot)) if len(legal) else None
            age = max(0.0, (self.step - self.usage[victim]) / self.layers) if victim is not None else 0.0
            free = int((self.owners < 0).sum())
            rho = self.control["rho"]
            q = 0
            while q < m:
                delta = max((q+1)/rho, m-q-1) - max(q/rho, m-q)
                if p[missing[q]] * (32.0 if free > q else age) < delta:
                    break
                q += 1
            if len(window) == 32:
                count.subtract(window[0])
            window.append(ids)
            count.update(ids)
        else:
            q = round_balance(self.fraction, m)
        q = min(m, max(q, self.control["fill_floor"])) if m else 0
        self.step += 1
        for slot in before:
            if slot >= 0:
                self.usage[slot] = self.step
        protected = {slot for slot in before if slot >= 0}
        victims, evicted = [], []
        for expert in missing[:q]:
            legal = [slot for slot in range(len(self.owners)) if slot not in protected]
            victim = min(legal, key=lambda slot: (self.usage[slot], slot))
            previous = int(self.owners[victim])
            if previous >= 0:
                self.slots.reshape(-1)[previous] = -1
            self.owners[victim] = layer * self.experts + expert
            self.slots[layer, expert] = victim
            self.usage[victim] = self.step
            protected.add(victim)
            victims.append(victim)
            evicted.append(previous)
        self.recency[layer, ids] = self.step
        return {"m": m, "q": q, "slots_before": before, "admitted_ids": missing[:q],
                "evict_slots": victims, "evicted_flat_ids": evicted, "step": self.step,
                "cpu_executions": m-q, "active_recency_after": self.recency[layer, ids].tolist(),
                "active_slots_after": self.slots[layer, ids].tolist()}


def qualify(directory, control):
    initial_path = directory / "decode_initial_cache.json"
    if not initial_path.exists():
        return {"status": "no_decode", "events": 0}
    replay = CacheReplay(json.loads(initial_path.read_text()), control)
    events = 0
    for row in json_lines(directory / "cache.jsonl.partial"):
        expected = replay.event(row["layer_id"], row["raw_ids"])
        for field, value in expected.items():
            if row.get(field) != value:
                raise ValueError(f"Replay mismatch event {events}, field {field}: expected {value}, observed {row.get(field)}")
        events += 1
    final = json.loads((directory / "decode_final_cache.json").read_text())
    for field, expected in {"slot_for_id": replay.slots.tolist(), "id_of_slot": replay.owners.tolist(),
                            "usage": replay.usage.tolist(), "expert_recency": replay.recency.tolist(),
                            "step": replay.step}.items():
        if final[field] != expected:
            raise ValueError("Replay final-state mismatch: " + field)
    return {"status": "matched", "events": events}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.capture, json.loads(args.control.read_text()))
    atomic_json(args.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
