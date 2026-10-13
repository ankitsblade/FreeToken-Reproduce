"""Paired U/F/R cache demand analysis on held-out FreeToken routing streams."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.pruning_analysis import WORKLOADS, interval, load_model, write_csv
from experiments.pruning.pruning_cache_replay import project_state, replay_stream, validate_native


def run(args):
    study = Path(args.study).resolve()
    models = json.loads((study / "protocol/models.json").read_text())
    protocol = json.loads((study / "protocol/protocol.json").read_text())
    splits = json.loads((study / "protocol/splits.json").read_text())
    analysis = study / "retention75"
    if not (analysis / "analysis_complete.json").exists():
        raise ValueError("held-out 75% masks must be frozen before replay")
    captures_by_model = {key: load_model(study, key, model) for key, model in models.items() if model["required"]}
    output = study / "replay75"
    output.mkdir(exist_ok=False)
    native_directory = study / "native_controls75"
    cold_controls_available = (native_directory / "completion.json").exists()
    rerouting_available = cold_controls_available and (native_directory / "routing_identity.json").exists()
    identity_records = {}
    if rerouting_available:
        identity_records = {(row["model"], row["request_id"]): row
                            for row in json.loads((native_directory / "routing_identity.json").read_text())}
    requests, validation, identities = [], [], []
    for key, captures in captures_by_model.items():
        model = models[key]
        E, S = model["experts"], model["cache_slots"]
        relative = math.ceil(S * math.ceil(.75*E) / E)
        for rid, capture in captures.items():
            native = validate_native(capture["directory"], model)
            validation.append({"model": key, "request_id": rid, **native})
            if native["status"] != "matched":
                atomic_json(output / "native_validation.json", validation)
                raise ValueError("observed-state native replay mismatch")
            data = capture["phases"]["decode"]
            initial = json.loads((capture["directory"] / "decode_initial_cache.json").read_text())
            fold = splits["group_to_fold"][capture["manifest"]["request"]["group_id"]]
            workload = capture["manifest"]["request"]["workload"]
            baselines = {}
            for state in ("cold", "post_prefill"):
                for policy, fraction in [("calibrated_hybrid", initial["fetch_fraction"]), ("unconditional_fill", 1.)]:
                    state_value = None if state == "cold" else initial
                    baseline = replay_stream(data["ids"], E, S, model["expert_bytes"], fraction, initial=state_value)
                    all_mask = np.ones((model["layers"], E), dtype=bool)
                    projected100 = None if state == "cold" else project_state(initial, all_mask, S)
                    filtered100 = replay_stream(data["ids"], E, S, model["expert_bytes"], fraction,
                                                mask=all_mask, initial=projected100)
                    if filtered100 != baseline:
                        raise ValueError("100% F identity is not the baseline")
                    baselines[state, policy] = baseline
                    identities.append({"model": key, "request_id": rid, "state": state, "policy": policy,
                                       "U_equals_F100": True, "R100_native_identity": (key, rid) in identity_records})
                    requests.append({"model": key, "request_id": rid, "workload": workload, "fold": fold,
                                     "pool": "baseline", "mode": "U", "state": state, "policy": policy,
                                     "capacity": "baseline_S0", "scope": "all90", "delta_mean_m": 0.,
                                     "delta_fill_bytes_per_token": 0., **baseline})
            for pool in (*WORKLOADS, "mixed"):
                ranking_path = analysis / f"{key}_decode_fold{fold}_{pool}_ranking.npz"
                mask = np.load(ranking_path)["mask"]
                modes = [("F", data["ids"], mask, "all90")]
                if pool == "mixed" and (key, rid) in identity_records:
                    record = identity_records[key, rid]
                    rerouted_path = native_directory / key / (rid + ".npz")
                    if sha256(rerouted_path) != record["rerouted_sha256"] or sha256(ranking_path) != record["mask_sha256"]:
                        raise ValueError("rerouting or fitted mask changed after native identity qualification")
                    modes.append(("R", np.load(rerouted_path)["decode_ids"], None, "preselected9"))
                for mode, stream, filtering, scope in modes:
                    for capacity, slots in [("fixed_bytes", S), ("relative_fraction", relative)]:
                        for state in ("cold", "post_prefill"):
                            for policy, fraction in [("calibrated_hybrid", initial["fetch_fraction"]), ("unconditional_fill", 1.)]:
                                projected = None if state == "cold" else project_state(initial, mask, slots)
                                result = replay_stream(stream, E, slots, model["expert_bytes"], fraction,
                                                       mask=filtering, initial=projected)
                                baseline = baselines[state, policy]
                                requests.append({"model": key, "request_id": rid, "workload": workload, "fold": fold,
                                                 "pool": pool, "mode": mode, "state": state, "policy": policy,
                                                 "capacity": capacity, "scope": scope,
                                                 "delta_mean_m": result["mean_m"]-baseline["mean_m"],
                                                 "delta_fill_bytes_per_token": result["h2d_fill_bytes_per_token"]-baseline["h2d_fill_bytes_per_token"],
                                                 **result})
            print(f"replayed {key}/{rid}", flush=True)
        atomic_json(output / "native_validation.json", validation)
    write_csv(output / "request.csv", requests)
    atomic_json(output / "identity100.json", identities)
    grouped = {}
    keys = ("model", "pool", "mode", "workload", "state", "policy", "capacity", "scope")
    for row in requests:
        if row["mode"] != "U":
            grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, rows in grouped.items():
        if len({r["request_id"] for r in rows}) != len(rows):
            raise ValueError("duplicated independent request in cache summary")
        ci = interval([r["delta_mean_m"] for r in rows], repeats=protocol["bootstrap_resamples"])
        mean_m = float(np.mean([r["mean_m"] for r in rows]))
        delta = float(np.mean([r["delta_mean_m"] for r in rows]))
        baseline_mean = mean_m-delta
        summaries.append({**dict(zip(keys, key)), "requests": len(rows), "mean_m": mean_m,
                          "baseline_mean_m": baseline_mean, "paired_delta_mean_m": delta,
                          "paired_delta_ci_low": ci[0], "paired_delta_ci_high": ci[1],
                          "relative_m_reduction": -delta/baseline_mean if baseline_mean else None,
                          "mean_active_k": float(np.mean([r["retained_active_k"] for r in rows])),
                          "mean_fill_bytes_per_token": float(np.mean([r["h2d_fill_bytes_per_token"] for r in rows])),
                          "mean_cpu_executions_per_token": float(np.mean([r["cpu_executions_per_token"] for r in rows]))})
    write_csv(output / "summary.csv", summaries)
    atomic_json(output / "completion.json", {"status": "complete", "retention": .75,
                "request_rows": len(requests), "summary_cells": len(summaries),
                "observed_native_validation": "all accepted primary requests matched",
                "cold_policy_validation": "native_controls75" if cold_controls_available else "unavailable in this study stratum",
                "R": "preselected9 mixed decode masks only" if rerouting_available else "unavailable: no qualified full-score native rerouting",
                "capacity_reference": "paired U always uses original S0; relative retained capacity ceil(S0*K/E)",
                "state_projection": "fixed capacity preserves original slots; reduced capacity keeps most-recent resident retained IDs with old-slot tie break",
                "interpretation": "conditional simulated cache demand; no latency or actual masked-model claim",
                "hybrid_profile_scope": "observed native automatic policy; Gemma reuses the profile's NVFP4 fraction measured on Qwen geometry, not an independently measured Gemma optimum",
                "source_sha256": sha256(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    run(parser.parse_args())
