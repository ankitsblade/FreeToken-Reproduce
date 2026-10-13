"""Validate fresh Gemma full scores and complete native static-input R75 replay."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.pruning_analysis import interval, write_csv
from experiments.pruning.pruning_cache_replay import project_state, replay_stream, validate_native
from experiments.pruning.pruning_native_controls import cold_control
from freetoken.kernel.triton.gemma4_fused import gemma4_fused_routing


def dense_capture(directory, model):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["checksums"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"capture changed: {directory / name}")
    phases = {}
    rows = list(json_lines(directory / "routes.jsonl.gz"))
    for phase in ("prefill", "decode"):
        selected = [row for row in rows if row["phase"] == phase]
        positions = sorted({row["input_position"] for row in selected})
        lookup = {position: i for i, position in enumerate(positions)}
        shape = (len(positions), model["layers"], model["top_k"])
        ids, weights = np.full(shape, -1, dtype=np.int32), np.full(shape, np.nan)
        seen = set()
        for row in selected:
            index, layer = lookup[row["input_position"]], row["layer_id"]
            if (index, layer) in seen:
                raise ValueError("duplicate routing row")
            seen.add((index, layer))
            ids[index, layer], weights[index, layer] = row["expert_ids"], row["effective_weights"]
        if (ids < 0).any() or not np.isfinite(weights).all():
            raise ValueError("incomplete native routing")
        phases[phase] = dict(ids=ids, weights=weights, positions=positions)
    return manifest, phases


def run(args):
    study, parent = Path(args.study).resolve(), Path(args.parent).resolve()
    model = json.loads((study / "protocol/models.json").read_text())["gemma_nvfp4"]
    splits = json.loads((study / "protocol/splits.json").read_text())
    batch = study / "routes/gemma_nvfp4" / args.batch_id
    completion = json.loads((batch / "completion.json").read_text())
    if completion["resource_status"] != "passed" or completion["captures"] != 18:
        raise ValueError("fresh full-score repeat batch must qualify all 18 captures")
    output = study / "native_controls75"
    output.mkdir(exist_ok=False)
    identity, comparisons, replay_rows, cold = [], [], [], []
    for rid in splits["repeat_request_ids"]:
        paths = sorted(path.parent for path in batch.glob("*/manifest.json")
                       if json.loads(path.read_text())["request"]["request_id"] == rid)
        if len(paths) != 2:
            raise ValueError("each preselected request needs exactly two full-score repeats")
        directory = paths[0]
        manifest, phases = dense_capture(directory, model)
        original_paths = [path.parent for path in (parent / "routes/gemma_nvfp4").glob("*/*/manifest.json")
                          if json.loads(path.read_text()).get("stage") == "primary"
                          and json.loads(path.read_text())["request"]["request_id"] == rid]
        if len(original_paths) != 1:
            raise ValueError("need unique immutable primary reference")
        original_manifest, original = dense_capture(original_paths[0], model)
        other_manifest, other = dense_capture(paths[1], model)
        for label, reference_manifest, reference in (("primary", original_manifest, original),
                                                      ("fresh_repeat", other_manifest, other)):
            if manifest["validation"]["generated_token_ids"] != reference_manifest["validation"]["generated_token_ids"]:
                raise ValueError(f"fresh observer changes generated token history: {rid}/{label}")
            for phase, data in phases.items():
                if data["positions"] != reference[phase]["positions"] or not np.array_equal(data["ids"], reference[phase]["ids"]):
                    raise ValueError(f"fresh observer changes original routes: {rid}/{phase}/{label}")
                error = float(np.max(np.abs(data["weights"] - reference[phase]["weights"])))
                if error > 2e-6:
                    raise ValueError("fresh observer weights exceed the qualified numerical tolerance")
                comparisons.append(dict(request_id=rid, reference=label, phase=phase,
                                        generated_tokens_equal=True, expert_ids_equal=True, maximum_weight_error=error))
        fold = splits["group_to_fold"][manifest["request"]["group_id"]]
        ranking = parent / "retention75" / f"gemma_nvfp4_decode_fold{fold}_mixed_ranking.npz"
        mask = np.load(ranking, allow_pickle=False)["mask"]
        if mask.shape != (model["layers"], model["experts"]) or not (mask.sum(axis=1) == 96).all():
            raise ValueError("require the frozen 75% fold-specific mixed mask")
        metadata = json.loads((directory / "router_metadata.json").read_text())
        score_rows = list(json_lines(directory / "scores.jsonl.gz"))
        rerouted, max_error = {}, 0.
        for phase, data in phases.items():
            logits = np.full((len(data["positions"]), model["layers"], model["experts"]), np.nan, dtype=np.float32)
            lookup, seen = {p: i for i, p in enumerate(data["positions"])}, set()
            for row in score_rows:
                if row["phase"] != phase:
                    continue
                index, layer = lookup[row["input_position"]], row["layer_id"]
                if (index, layer) in seen or row["semantics"] != "native_router_logits":
                    raise ValueError("duplicate or non-native scores")
                seen.add((index, layer))
                logits[index, layer] = row["scores"]
            if not np.isfinite(logits).all():
                raise ValueError("incomplete/nonfinite full router scores")
            new_ids, new_weights = np.empty_like(data["ids"]), np.empty_like(data["weights"])
            for layer in range(model["layers"]):
                meta = metadata[str(layer)]
                raw = torch.from_numpy(logits[:, layer].copy()).cuda()
                dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[meta["logit_dtype"]]
                scores = raw.to(dtype)
                if not torch.equal(scores.float(), raw):
                    raise ValueError("score dtype restoration loses information")
                scale_raw = torch.tensor(meta["per_expert_scale"], device="cuda", dtype=torch.float32)
                scale_dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32}[meta["scale_dtype"]]
                scales = scale_raw.to(scale_dtype)
                if not torch.equal(scales.float(), scale_raw) or not torch.isfinite(scales).all():
                    raise ValueError("expert-scale dtype restoration loses information")
                weights, ids = gemma4_fused_routing(scores, scales, model["top_k"])
                error = float(np.max(np.abs(weights.cpu().numpy() - data["weights"][:, layer])))
                max_error = max(max_error, error)
                if not np.array_equal(ids.cpu().numpy(), data["ids"][:, layer]) or error > 2e-6:
                    raise ValueError(f"Gemma native 100% routing identity failed: {rid}/{phase}/{layer}")
                masked = scores.masked_fill(~torch.from_numpy(mask[layer].copy()).cuda(), -float("inf"))
                weights, ids = gemma4_fused_routing(masked, scales, model["top_k"])
                new_ids[:, layer], new_weights[:, layer] = ids.cpu().numpy(), weights.cpu().numpy()
                if not mask[layer, new_ids[:, layer]].all() or not np.isfinite(new_weights[:, layer]).all():
                    raise ValueError("native masked selection or weights invalid")
            rerouted[phase + "_ids"], rerouted[phase + "_weights"] = new_ids, new_weights
        target = output / "gemma_nvfp4" / (rid + ".npz")
        target.parent.mkdir(exist_ok=True)
        np.savez_compressed(target, **rerouted)
        identity.append(dict(model="gemma_nvfp4", request_id=rid, status="matched100_and_rerouted75",
                             score_sha256=sha256(directory / "scores.jsonl.gz"),
                             router_metadata_sha256=sha256(directory / "router_metadata.json"),
                             mask_sha256=sha256(ranking), rerouted_sha256=sha256(target),
                             maximum_identity_weight_error=max_error, source_capture=str(directory),
                             scope="native Gemma masked router at recorded original hidden states"))
        atomic_json(output / "routing_identity.json", identity)
        native = validate_native(directory, model)
        if native["status"] != "matched":
            raise ValueError("fresh native cache trace does not match replay")
        initial = json.loads((directory / "decode_initial_cache.json").read_text())
        ids = phases["decode"]["ids"]
        relative = math.ceil(model["cache_slots"] * .75)
        for slots in (model["cache_slots"], relative):
            for policy, fraction in (("calibrated_hybrid", initial["fetch_fraction"]), ("unconditional_fill", 1.)):
                for state in ("cold", "post_prefill"):
                    base = replay_stream(ids, model["experts"], model["cache_slots"], model["expert_bytes"], fraction,
                                         initial=None if state == "cold" else initial)
                    projected = None if state == "cold" else project_state(initial, mask, slots)
                    result = replay_stream(rerouted["decode_ids"], model["experts"], slots, model["expert_bytes"], fraction,
                                           initial=projected)
                    replay_rows.append(dict(request_id=rid, workload=manifest["request"]["workload"], state=state,
                                            policy=policy, slots=slots, baseline_mean_m=base["mean_m"],
                                            delta_mean_m=result["mean_m"]-base["mean_m"], **result))
                if rid == splits["repeat_request_ids"][0]:
                    for mode, stream, filtering in (("U", ids, None), ("F", ids, mask), ("R", rerouted["decode_ids"], None)):
                        case = dict(mode=mode, policy=policy, **cold_control(stream, model, slots, fraction, filtering))
                        cold.append(case)
                        if case["status"] != "matched":
                            raise ValueError("native Gemma cold cache control mismatch")
        print(f"Gemma native R75 qualified: {rid}; identity error={max_error}", flush=True)
    summaries = []
    for state in ("cold", "post_prefill"):
        for policy in ("calibrated_hybrid", "unconditional_fill"):
            for slots in (model["cache_slots"], math.ceil(model["cache_slots"]*.75)):
                selected = [r for r in replay_rows if (r["state"], r["policy"], r["slots"]) == (state, policy, slots)]
                delta = np.mean([r["delta_mean_m"] for r in selected])
                baseline = np.mean([r["baseline_mean_m"] for r in selected])
                ci = interval([r["delta_mean_m"] for r in selected])
                summaries.append(dict(state=state, policy=policy, slots=slots, requests=len(selected),
                                      baseline_mean_m=float(baseline), delta_mean_m=float(delta),
                                      delta_ci_low=ci[0], delta_ci_high=ci[1], relative_m_reduction=float(-delta/baseline)))
    write_csv(output / "fresh_capture_comparisons.csv", comparisons)
    write_csv(output / "R75_requests.csv", replay_rows)
    write_csv(output / "R75_summary.csv", summaries)
    atomic_json(output / "cold_cache.json", cold)
    atomic_json(output / "completion.json", dict(status="complete", routing_requests=len(identity), cold_cases=len(cold),
                R="qualified native Gemma full-score static-input rerouting on preselected nine requests",
                original_closeout="preserved; this is a supplement", source_sha256=sha256(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--batch-id", required=True)
    run(parser.parse_args())
