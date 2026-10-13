"""Compare qualified repeated captures without counting them as independent tasks."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.pruning_analysis import access_counts, jaccard, load_model, retained_mask, write_csv


def read_repeat(directory, model):
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["checksums"].items():
        if sha256(directory / name) != expected:
            raise ValueError("repeat capture checksum changed")
    phases = {}
    records = list(json_lines(directory / "routes.jsonl.gz"))
    for phase in ("prefill", "decode"):
        rows = [row for row in records if row["phase"] == phase]
        positions = sorted({row["input_position"] for row in rows})
        indices = {position: index for index, position in enumerate(positions)}
        ids = np.full((len(positions), model["layers"], model["top_k"]), -1, dtype=np.int32)
        weights = np.full(ids.shape, np.nan, dtype=np.float64)
        for row in rows:
            index = indices[row["input_position"]]
            ids[index, row["layer_id"]] = row["expert_ids"]
            weights[index, row["layer_id"]] = row["effective_weights"]
        if not len(positions) or (ids < 0).any() or not np.isfinite(weights).all():
            raise ValueError("incomplete repeat route data")
        phases[phase] = {"ids": ids, "weights": weights, "positions": positions,
                         "counts": access_counts(ids, model["experts"])}
    return {"manifest": manifest, "phases": phases, "directory": directory}


def run(args):
    primary_study, repeat_study = Path(args.primary_study).resolve(), Path(args.repeat_study).resolve()
    model = json.loads((primary_study / "protocol/models.json").read_text())[args.model]
    repeat_model = json.loads((repeat_study / "protocol/models.json").read_text())[args.model]
    if model != repeat_model:
        raise ValueError("repeat model geometry/checkpoint differs from primary")
    primary_protocol = json.loads((primary_study / "protocol/protocol.json").read_text())
    repeat_protocol = json.loads((repeat_study / "protocol/protocol.json").read_text())
    if primary_protocol != repeat_protocol:
        raise ValueError("repeat generation protocol differs from primary")
    for name in ("dataset.jsonl", "splits.json"):
        if sha256(primary_study / "protocol" / name) != sha256(repeat_study / "protocol" / name):
            raise ValueError("repeat task identities/splits differ from primary")
    primary_inputs = json.loads((primary_study / "protocol/rendered_inputs.json").read_text())[args.model]
    repeat_inputs = json.loads((repeat_study / "protocol/rendered_inputs.json").read_text())[args.model]
    if primary_inputs != repeat_inputs:
        raise ValueError("repeat rendered prompt tokens differ from primary")
    primary = load_model(primary_study, args.model, model)
    repeats = repeat_study / "routes" / args.model / args.batch_id
    completion = json.loads((repeats / "completion.json").read_text())
    if completion["resource_status"] != "passed" or completion["captures"] != 18:
        raise ValueError("repeat batch is not resource-qualified")
    wanted = json.loads((primary_study / "protocol/splits.json").read_text())["repeat_request_ids"]
    captures = {rid: [primary[rid]] for rid in wanted}
    for path in sorted(repeats.glob("*/manifest.json")):
        capture = read_repeat(path.parent, model)
        rid = capture["manifest"]["request"]["request_id"]
        if rid not in captures or capture["manifest"]["stage"] != "repeat":
            raise ValueError("unexpected repeat identity/stage")
        captures[rid].append(capture)
    if any(len(group) != 3 for group in captures.values()):
        raise ValueError("need one primary plus two qualified repeats per preselected request")
    repeat_config = json.loads((repeats / "configuration.json").read_text())
    for rid, group in captures.items():
        primary_config = json.loads((group[0]["directory"].parent / "configuration.json").read_text())
        if primary_config["inference_hashes"] != repeat_config["inference_hashes"]:
            raise ValueError("repeat engine implementation differs from primary")
    rows = []
    for rid, group in captures.items():
        for left, right in itertools.combinations(range(3), 2):
            a, b = group[left], group[right]
            av, bv = a["manifest"]["validation"], b["manifest"]["validation"]
            outputs_a, outputs_b = av["generated_token_ids"], bv["generated_token_ids"]
            shared = 0
            for x, y in zip(outputs_a, outputs_b):
                if x != y:
                    break
                shared += 1
            for phase in ("prefill", "decode"):
                ad, bd = a["phases"][phase], b["phases"][phase]
                n = min(len(ad["ids"]), len(bd["ids"]))
                if phase == "decode":
                    n = min(n, shared)
                sets_equal, weight_error = [], []
                for index in range(n):
                    if ad["positions"][index] != bd["positions"][index]:
                        raise ValueError("shared-prefix position mismatch")
                    for layer in range(model["layers"]):
                        ao, bo = np.argsort(ad["ids"][index, layer]), np.argsort(bd["ids"][index, layer])
                        equal = np.array_equal(ad["ids"][index, layer, ao], bd["ids"][index, layer, bo])
                        sets_equal.append(equal)
                        if equal:
                            weight_error.append(float(np.max(np.abs(ad["weights"][index, layer, ao]-bd["weights"][index, layer, bo]))))
                ap = ad["counts"] / ad["counts"].sum(axis=1, keepdims=True)
                bp = bd["counts"] / bd["counts"].sum(axis=1, keepdims=True)
                rows.append({"model": args.model, "request_id": rid,
                             "workload": a["manifest"]["request"]["workload"], "left_capture": left,
                             "right_capture": right, "phase": phase, "identical_generated_tokens": outputs_a == outputs_b,
                             "generated_tokens_left": len(outputs_a), "generated_tokens_right": len(outputs_b),
                             "common_generated_prefix": shared, "shared_input_tokens_compared": n,
                             "native_expert_set_agreement_on_shared_prefix": float(np.mean(sets_equal)) if sets_equal else None,
                             "maximum_weight_error_where_ids_equal": max(weight_error) if weight_error else None,
                             "full_phase_frequency_L1_mean": float(np.abs(ap-bp).sum(axis=1).mean()),
                             "full_phase_oracle_mask75_jaccard_mean": float(jaccard(retained_mask(ad["counts"], .75), retained_mask(bd["counts"], .75)).mean())})
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "pairs.csv", rows)
    atomic_json(output / "completion.json", {"status": "complete", "model": args.model,
                "independent_requests": 9, "extra_captures": 18, "pairs": 27,
                "interpretation": "repeated greedy full-request measurement sensitivity; shared-prefix route comparisons hold token histories fixed; whole-phase frequencies may include divergent outputs",
                "primary_study": str(primary_study), "repeat_study": str(repeat_study), "repeat_batch": args.batch_id,
                "same_runtime_arm": primary_study == repeat_study,
                "source_sha256": sha256(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-study", required=True)
    parser.add_argument("--repeat-study", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
