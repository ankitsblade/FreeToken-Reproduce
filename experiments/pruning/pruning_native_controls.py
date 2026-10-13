"""Qualify static-input rerouting and cold cache replay against FreeToken GPU kernels."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.pruning_analysis import load_model
from experiments.pruning.pruning_cache_replay import NativeHybridReplay


def native_cache(model, slots):
    import torch

    def full(shape, value, dtype):
        return torch.full(shape, value, dtype=dtype, device="cuda")

    L, E = model["layers"], model["experts"]
    return SimpleNamespace(num_experts=E, cache_size=slots,
                           slot_for_id=full((L, E), -1, torch.int32),
                           id_of_slot=full((slots,), -1, torch.int32),
                           usage=full((slots,), 0, torch.int64), step=full((), 0, torch.int64),
                           active_mask=full((E,), 0, torch.int32),
                           expert_recency=full((L, E), -1, torch.int64),
                           num_missing_full=full((), 0, torch.int64), num_indices=full((), 0, torch.int64),
                           evict_slots=full((E,), -1, torch.int32), src_indices=full((E,), -1, torch.int32))


def cold_control(ids, model, slots, fraction, mask=None):
    import torch
    from freetoken.moe.offload_kernels import ensure_experts_hybrid

    native = native_cache(model, slots)
    simulation = NativeHybridReplay(model["layers"], model["experts"], slots, fraction)
    events = 0
    for token in ids[:64]:
        for layer, selected in enumerate(token):
            active = [int(e) for e in selected if mask is None or mask[layer, e]]
            expected = simulation.event(layer, active)
            previous = native.id_of_slot.clone()
            before = native.slot_for_id[layer, active].cpu().tolist()
            ensure_experts_hybrid(native, layer, torch.tensor([active], dtype=torch.int32, device="cuda"),
                                  model["experts"], fraction)
            q = int(native.num_indices.item())
            observed = {"m": int(native.num_missing_full.item()), "q": q,
                        "slots_before": before, "admitted_ids": native.src_indices[:q].cpu().tolist(),
                        "evict_slots": native.evict_slots[:q].cpu().tolist(),
                        "evicted_flat_ids": previous[native.evict_slots[:q].long()].cpu().tolist(),
                        "step": int(native.step.item())}
            for field, actual in observed.items():
                if expected[field] != actual:
                    return {"status": "mismatch", "event": events, "field": field,
                            "expected": expected[field], "observed": actual}
            events += 1
    for name, expected in [("slot_for_id", simulation.slot_for_id), ("id_of_slot", simulation.id_of_slot),
                           ("usage", simulation.usage), ("expert_recency", simulation.recency)]:
        if not np.array_equal(expected, getattr(native, name).cpu().numpy()):
            return {"status": "mismatch", "event": events, "field": name}
    return {"status": "matched", "events": events, "slots": slots, "fetch_fraction": fraction}


def run(args):
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise ValueError("native controls require an idle GPU")
    import torch
    from freetoken.kernel import gpt_oss_fused_routing
    from freetoken.kernel.triton.moe_router import fused_topk_softmax

    study = Path(args.study).resolve()
    output = study / "native_controls75"
    output.mkdir(exist_ok=False)
    models = json.loads((study / "protocol/models.json").read_text())
    splits = json.loads((study / "protocol/splits.json").read_text())
    repeat_ids = splits["repeat_request_ids"]
    results = []
    for key, model in models.items():
        if not model["required"]:
            continue
        captures = load_model(study, key, model)
        router = ((lambda scores: gpt_oss_fused_routing(scores.contiguous(), model["top_k"]))
                  if key == "gpt_oss_mxfp4" else
                  (lambda scores: fused_topk_softmax(scores, model["top_k"], True)))
        for rid in repeat_ids:
            capture = captures[rid]
            scores_path = capture["directory"] / "scores.jsonl.gz"
            rows = list(json_lines(scores_path))
            if not rows:
                raise ValueError(f"no complete scores: {key}/{rid}")
            fold = splits["group_to_fold"][capture["manifest"]["request"]["group_id"]]
            ranking = study / "retention75" / f"{key}_decode_fold{fold}_mixed_ranking.npz"
            mask = np.load(ranking)["mask"]
            if int(mask.sum(axis=1).min()) < model["top_k"]:
                raise ValueError("mask cannot supply native top-k")
            rerouted = {}
            maximum_error = 0.
            for phase, data in capture["phases"].items():
                logits = np.full((len(data["positions"]), model["layers"], model["experts"]), np.nan, dtype=np.float32)
                lookup = {position: index for index, position in enumerate(data["positions"])}
                seen = set()
                for row in rows:
                    if row["phase"] != phase:
                        continue
                    index, layer = lookup[row["input_position"]], row["layer_id"]
                    if (index, layer) in seen or row["semantics"] != "native_router_logits":
                        raise ValueError("duplicate scores or unsupported router semantics")
                    seen.add((index, layer))
                    logits[index, layer] = row["scores"]
                if not np.isfinite(logits).all():
                    raise ValueError("scores are incomplete or nonfinite")
                new_ids = np.empty_like(data["ids"])
                new_weights = np.empty_like(data["weights"])
                for layer in range(model["layers"]):
                    raw = torch.from_numpy(logits[:, layer].copy()).cuda()
                    scores = raw.to(torch.bfloat16)
                    if not torch.equal(scores.float(), raw):
                        raise ValueError("BF16 restoration changes captured logits; qualify original dtype first")
                    weights, ids = router(scores)
                    ids, weights = ids.cpu().numpy(), weights.cpu().numpy()
                    error = float(np.max(np.abs(weights-data["weights"][:, layer])))
                    maximum_error = max(maximum_error, error)
                    if not np.array_equal(ids, data["ids"][:, layer]) or error > 2e-6:
                        raise ValueError(f"100% native routing identity failed: {key}/{rid}/{phase}/{layer}")
                    scores[:, ~torch.as_tensor(mask[layer], device="cuda")] = -float("inf")
                    weights, ids = router(scores)
                    new_ids[:, layer] = ids.cpu().numpy()
                    new_weights[:, layer] = weights.cpu().numpy()
                    if not mask[layer, new_ids[:, layer]].all():
                        raise ValueError("masked native router selected a removed expert")
                rerouted[phase + "_ids"] = new_ids
                rerouted[phase + "_weights"] = new_weights
            target = output / key / (rid + ".npz")
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **rerouted)
            result = {"model": key, "request_id": rid, "status": "matched100_and_rerouted75",
                      "score_sha256": sha256(scores_path), "mask_sha256": sha256(ranking),
                      "rerouted_sha256": sha256(target), "maximum_identity_weight_error": maximum_error,
                      "dtype": "BF16 restored losslessly and validated against native recorded IDs/weights",
                      "scope": "native masked-router selection at recorded hidden states; not masked inference"}
            results.append(result)
            atomic_json(output / "routing_identity.json", results)
            print(f"qualified rerouting {key}/{rid}", flush=True)
        representative = captures[repeat_ids[0]]
        initial = json.loads((representative["directory"] / "decode_initial_cache.json").read_text())
        ids = representative["phases"]["decode"]["ids"]
        fold = splits["group_to_fold"][representative["manifest"]["request"]["group_id"]]
        mask = np.load(study / "retention75" / f"{key}_decode_fold{fold}_mixed_ranking.npz")["mask"]
        reroutes = np.load(output / key / (repeat_ids[0] + ".npz"))["decode_ids"]
        cases = []
        relative = math.ceil(model["cache_slots"] * math.ceil(.75*model["experts"]) / model["experts"])
        for slots in sorted({model["cache_slots"], relative}):
            for policy, fraction in [("hybrid", initial["fetch_fraction"]), ("unconditional_fill", 1.)]:
                for mode, stream, filtering in [("U", ids, None), ("F", ids, mask), ("R", reroutes, None)]:
                    case = {"model": key, "mode": mode, "policy": policy,
                            **cold_control(stream, model, slots, fraction, filtering)}
                    cases.append(case)
                    atomic_json(output / f"{key}_cold_cache.json", cases)
                    if case["status"] != "matched":
                        raise ValueError("cold native cache control diverged; inspect first mismatch")
        print(f"qualified cold cache controls {key}: {len(cases)} cases", flush=True)
    atomic_json(output / "completion.json", {"status": "complete", "routing_requests": len(results),
                                             "retention": .75, "source_sha256": sha256(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    run(parser.parse_args())
