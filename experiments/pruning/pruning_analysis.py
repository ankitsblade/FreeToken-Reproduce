"""Deterministic held-out analysis of native packed route records."""
from __future__ import annotations

import csv
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np

from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.collect_moe_routes import json_lines

WORKLOADS = ("math", "code", "incident")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def retained_mask(counts, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("retention must be in (0,1]")
    counts = np.asarray(counts)
    if counts.ndim != 2 or np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("rankings require finite nonnegative per-layer counts")
    E = counts.shape[1]
    K = math.ceil(E * fraction)
    mask = np.zeros(counts.shape, dtype=bool)
    for layer, frequencies in enumerate(counts):
        order = np.lexsort((np.arange(E), -frequencies))
        mask[layer, order[:K]] = True
    return mask


def access_counts(ids, E):
    result = np.zeros((ids.shape[1], E), dtype=np.int64)
    for layer in range(ids.shape[1]):
        result[layer] = np.bincount(ids[:, layer].reshape(-1), minlength=E)
    return result


def metrics(ids, weights, mask):
    if len(ids) == 0:
        return None
    selected = mask[np.arange(ids.shape[1])[None, :, None], ids]
    token_c = selected.mean(axis=(1, 2))
    token_mass = (weights * selected).sum(axis=(1, 2)) / weights.sum(axis=(1, 2))
    mass_numerator = (weights * selected).sum(axis=(1, 2))
    mass_denominator = weights.sum(axis=(1, 2))
    layer_c = selected.mean(axis=(0, 2))
    layer_mass = (weights * selected).sum(axis=(0, 2)) / weights.sum(axis=(0, 2))
    denominator = float(weights.sum())
    low = token_c < .95
    longest = current = 0
    for value in low:
        current = current + 1 if value else 0
        longest = max(current, longest)
    windows = {}
    mass_windows = {}
    for window in (32, 64):
        if len(token_c) >= window:
            windows[window] = np.convolve(token_c, np.ones(window) / window, mode="valid")
            mass_windows[window] = (np.convolve(mass_numerator, np.ones(window), mode="valid") /
                                    np.convolve(mass_denominator, np.ones(window), mode="valid"))
        else:
            windows[window] = np.array([])
            mass_windows[window] = np.array([])
    return {"access_coverage": float(selected.mean()), "gate_mass": float((weights * selected).sum() / denominator),
            "accesses": int(ids.size), "retained_accesses": int(selected.sum()), "tokens": len(ids),
            "retained_mass": float((weights * selected).sum()), "total_mass": denominator,
            "token_median": float(np.median(token_c)), "token_p5": float(np.quantile(token_c, .05)),
            "token_p95": float(np.quantile(token_c, .95)), "token_min": float(token_c.min()),
            "worst_layer": float(layer_c.min()), "longest_below95": longest,
            "rolling32_min": float(windows[32].min()) if len(windows[32]) else None,
            "rolling64_min": float(windows[64].min()) if len(windows[64]) else None,
            "layer_coverage": layer_c, "layer_mass": layer_mass, "token_coverage": token_c,
            "token_mass": token_mass, "windows": windows, "mass_windows": mass_windows}


def interval(values, seed=20261006, repeats=10000):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("interval requires finite independent request observations")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def load_model(study, key, model):
    captures = {}
    for path in sorted((study / "routes" / key).rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("stage") != "primary":
            continue
        completion_path = path.parent.parent / "completion.json"
        if not completion_path.exists() or json.loads(completion_path.read_text()).get("resource_status") != "passed":
            raise ValueError(f"primary batch has not passed resource qualification: {path.parent.parent}")
        rid = manifest["request"]["request_id"]
        if rid in captures:
            raise ValueError(f"duplicate primary capture {key}/{rid}")
        for name, checksum in manifest["checksums"].items():
            if sha256(path.parent / name) != checksum:
                raise ValueError(f"capture checksum changed: {path.parent/name}")
        phases = {}
        for row in json_lines(path.parent / "routes.jsonl.gz"):
            phase = row["phase"]
            phases.setdefault(phase, []).append(row)
        dense = {}
        for phase, rows in phases.items():
            positions = sorted({r["input_position"] for r in rows})
            lookup = {p: i for i, p in enumerate(positions)}
            ids = np.full((len(positions), model["layers"], model["top_k"]), -1, dtype=np.int32)
            weights = np.full(ids.shape, np.nan, dtype=np.float64)
            for row in rows:
                ids[lookup[row["input_position"]], row["layer_id"]] = row["expert_ids"]
                weights[lookup[row["input_position"]], row["layer_id"]] = row["effective_weights"]
            if (ids < 0).any() or not np.isfinite(weights).all():
                raise ValueError("capture has incomplete phase/layer data")
            dense[phase] = {"ids": ids, "weights": weights, "positions": positions,
                            "counts": access_counts(ids, model["experts"])}
        captures[rid] = {"manifest": manifest, "directory": path.parent, "phases": dense}
    expected = {r["request_id"] for r in json_lines(study / "protocol/dataset.jsonl") if not r["pilot_flag"]}
    if set(captures) != expected:
        raise ValueError(f"{key}: need all {len(expected)} primary captures, have {len(captures)}")
    return captures


def bootstrap_summary(rows, repeats):
    out = []
    keys = ("model", "ranking_phase", "evaluation_phase", "calibration_pool", "evaluation_workload", "retention")
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    for key, group in groups.items():
        if len({r["request_id"] for r in group}) != len(group):
            raise ValueError("a request appears twice in an out-of-fold summary")
        values = np.array([r["access_coverage"] for r in group])
        weights = np.array([r["gate_mass"] for r in group])
        ci = interval(values, repeats=repeats)
        mass_ci = interval(weights, repeats=repeats)
        row = dict(zip(keys, key))
        row.update({"requests": len(group), "mean": float(values.mean()), "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)), "min": float(values.min()), "p5": float(np.quantile(values, .05)),
                    "p95": float(np.quantile(values, .95)), "ci_low": ci[0], "ci_high": ci[1],
                    "micro_coverage": sum(r["retained_accesses"] for r in group) / sum(r["accesses"] for r in group),
                    "gate_mass_mean": float(weights.mean()), "gate_mass_ci_low": mass_ci[0], "gate_mass_ci_high": mass_ci[1],
                    "worst_layer_mean": float(np.mean([r["worst_layer"] for r in group]))})
        out.append(row)
    return out


def jaccard(a, b):
    return np.logical_and(a, b).sum(axis=1) / np.logical_or(a, b).sum(axis=1)


def analyze(args):
    study = Path(args.study).resolve()
    fraction = args.retention
    if fraction != .75:
        closeout = study / "closeout75/artifact_manifest.json"
        if not closeout.exists() or not json.loads(closeout.read_text()).get("primary_closed"):
            raise ValueError("lower retention is blocked until 75% closeout")
        if fraction == .5 and not (study / "ablation60/completion.json").exists():
            raise ValueError("50% is blocked until 60% completion")
        if fraction not in (.6, .5):
            raise ValueError("unscheduled retention")
    output = study / ({.75: "retention75", .6: "ablation60", .5: "ablation50"}[fraction])
    protocol = json.loads((study / "protocol/protocol.json").read_text())
    models = json.loads((study / "protocol/models.json").read_text())
    all_captures = {key: load_model(study, key, model) for key, model in models.items() if model["required"]}
    output.mkdir(exist_ok=False)
    splits = json.loads((study / "protocol/splits.json").read_text())
    assignment = splits["group_to_fold"]
    requests, layers, windows, oracle_rows, stability, rank_rows = [], [], [], [], [], []
    sensitivity = []
    masks_record = {}
    for key, model in models.items():
        if not model["required"]:
            continue
        captures = all_captures[key]
        random_rng = np.random.default_rng(protocol["seed"])
        K = math.ceil(fraction * model["experts"])
        chance = []
        for _ in range(protocol["random_masks"]):
            left = np.zeros((model["layers"], model["experts"]), dtype=bool)
            right = np.zeros_like(left)
            for layer in range(model["layers"]):
                left[layer, random_rng.choice(model["experts"], K, replace=False)] = True
                right[layer, random_rng.choice(model["experts"], K, replace=False)] = True
            chance.append(jaccard(left, right))
        chance_mean = np.mean(chance, axis=0)
        fitted = {}
        for phase in ("prefill", "decode"):
            phase_ids = [rid for rid, cap in captures.items() if phase in cap["phases"]]
            for rid in phase_ids:
                data = captures[rid]["phases"][phase]
                oracle = retained_mask(data["counts"], fraction)
                m = metrics(data["ids"], data["weights"], oracle)
                identity = metrics(data["ids"], data["weights"], np.ones_like(oracle))
                if identity["access_coverage"] != 1 or abs(identity["gate_mass"] - 1) > 1e-12:
                    raise ValueError("100% identity failed")
                random_masks = []
                for _ in range(protocol["random_masks"]):
                    random_mask = np.zeros_like(oracle)
                    for l in range(model["layers"]):
                        random_mask[l, random_rng.choice(model["experts"], K, replace=False)] = True
                    random_masks.append(float((data["counts"] * random_mask).sum() / data["counts"].sum()))
                oracle_rows.append({"model": key, "phase": phase, "request_id": rid,
                                    "workload": captures[rid]["manifest"]["request"]["workload"],
                                    "retention": fraction, "actual_retention": K/model["experts"],
                                    "oracle_coverage": m["access_coverage"], "oracle_gate_mass": m["gate_mass"],
                                    "random_mean": float(np.mean(random_masks)), "random_expected": K/model["experts"],
                                    "control100_coverage": 1., "control100_gate_mass": 1.})
            for fold in range(5):
                training = [rid for rid in phase_ids if assignment[captures[rid]["manifest"]["request"]["group_id"]] != fold]
                heldout = [rid for rid in phase_ids if rid not in training]
                for pool in (*WORKLOADS, "mixed"):
                    train = [rid for rid in training if pool == "mixed" or captures[rid]["manifest"]["request"]["workload"] == pool]
                    counts = sum((captures[rid]["phases"][phase]["counts"] for rid in train),
                                 np.zeros((model["layers"], model["experts"]), dtype=np.int64))
                    if not train or (counts.sum(axis=1) == 0).any():
                        raise ValueError("empty calibration phase/pool")
                    mask = retained_mask(counts, fraction)
                    if fraction != .75:
                        frozen = np.load(study / "retention75" / f"{key}_{phase}_fold{fold}_{pool}_ranking.npz")
                        if not np.array_equal(frozen["counts"], counts) or not np.array_equal(frozen["calibration_ids"], train):
                            raise ValueError("calibration differs from frozen 75% ranking")
                        if np.any(mask & ~frozen["mask"]):
                            raise ValueError("lower-retention masks are not nested in the frozen 75% mask")
                    fitted[phase, fold, pool] = mask
                    masks_record[key, phase, fold, pool] = mask
                    np.savez_compressed(output / f"{key}_{phase}_fold{fold}_{pool}_ranking.npz",
                                        counts=counts, mask=mask, calibration_ids=np.array(train))
                    for layer, frequency in enumerate(counts):
                        order = np.lexsort((np.arange(model["experts"]), -frequency))
                        cutoff = frequency[order[K-1]]
                        for rank, expert in enumerate(order, 1):
                            rank_rows.append({"model": key, "phase": phase, "fold": fold, "pool": pool,
                                              "layer": layer, "expert": int(expert), "count": int(frequency[expert]),
                                              "rank": rank, "retained": bool(mask[layer, expert]),
                                              "cutoff_ties": int((frequency == cutoff).sum()), "unseen": int(frequency[expert]) == 0})
                    for eval_phase in ([phase, "prefill"] if phase == "decode" and pool == "mixed" else [phase]):
                        for rid in heldout:
                            if eval_phase not in captures[rid]["phases"]:
                                continue
                            data = captures[rid]["phases"][eval_phase]
                            result = metrics(data["ids"], data["weights"], mask)
                            base = {"model": key, "ranking_phase": phase, "evaluation_phase": eval_phase,
                                    "fold": fold, "calibration_pool": pool,
                                    "evaluation_workload": captures[rid]["manifest"]["request"]["workload"],
                                    "request_id": rid, "retention": fraction, "actual_retention": K/model["experts"]}
                            scalars = {k: v for k, v in result.items() if k not in
                                       ("layer_coverage", "layer_mass", "token_coverage", "token_mass", "windows", "mass_windows")}
                            requests.append({**base, **scalars})
                            for layer, value in enumerate(result["layer_coverage"]):
                                layers.append({**base, "layer": layer, "access_coverage": float(value),
                                               "gate_mass": float(result["layer_mass"][layer])})
                            if pool == "mixed":
                                cumulative = np.cumsum(result["token_coverage"]) / np.arange(1, len(data["ids"])+1)
                                for index, value in enumerate(result["token_coverage"]):
                                    windows.append({**base, "phase_token_index": index, "input_position": data["positions"][index],
                                                    "token_coverage": float(value), "token_gate_mass": float(result["token_mass"][index]),
                                                    "cumulative_coverage": float(cumulative[index]),
                                                    "rolling32": float(result["windows"][32][index-31]) if index >= 31 else None,
                                                    "rolling64": float(result["windows"][64][index-63]) if index >= 63 else None,
                                                    "rolling32_gate_mass": float(result["mass_windows"][32][index-31]) if index >= 31 else None,
                                                    "rolling64_gate_mass": float(result["mass_windows"][64][index-63]) if index >= 63 else None})
                    normalized = {rid: captures[rid]["phases"][phase]["counts"] /
                                  captures[rid]["phases"][phase]["counts"].sum(axis=1, keepdims=True) for rid in train}
                    for method in ("equal_request", "equal_workload"):
                        if method == "equal_request":
                            frequencies = sum(normalized.values()) / len(train)
                        else:
                            domain_means = []
                            for workload in WORKLOADS:
                                domain = [normalized[rid] for rid in train
                                          if captures[rid]["manifest"]["request"]["workload"] == workload]
                                if domain:
                                    domain_means.append(sum(domain) / len(domain))
                            frequencies = sum(domain_means) / len(domain_means)
                        alternative = retained_mask(frequencies, fraction)
                        for rid in heldout:
                            data = captures[rid]["phases"][phase]
                            result = metrics(data["ids"], data["weights"], alternative)
                            sensitivity.append({"model": key, "phase": phase, "fold": fold, "pool": pool,
                                                "method": method, "request_id": rid, "retention": fraction,
                                                "workload": captures[rid]["manifest"]["request"]["workload"],
                                                "access_coverage": result["access_coverage"], "gate_mass": result["gate_mass"],
                                                "jaccard_to_primary": float(jaccard(mask, alternative).mean())})
                    for repeat in range(10):
                        rng = np.random.default_rng(protocol["seed"] + 100*fold + repeat)
                        halves = [[], []]
                        for workload in WORKLOADS:
                            domain = sorted(rid for rid in train if captures[rid]["manifest"]["request"]["workload"] == workload)
                            rng.shuffle(domain)
                            halves[0].extend(domain[:len(domain)//2]); halves[1].extend(domain[len(domain)//2:])
                        masks = []
                        for half in halves:
                            frequencies = sum((captures[rid]["phases"][phase]["counts"] for rid in half), np.zeros_like(counts))
                            masks.append(retained_mask(frequencies, fraction))
                        for layer, value in enumerate(jaccard(*masks)):
                            stability.append({"model": key, "phase": phase, "kind": "disjoint_halves", "pool": pool,
                                              "left_fold": fold, "right_fold": fold, "repeat": repeat, "layer": layer,
                                              "jaccard": float(value), "random_reference": float(chance_mean[layer]),
                                              "excess_jaccard": float(value-chance_mean[layer]), "shared_requests": 0})
        for phase in ("prefill", "decode"):
            for fold in range(5):
                for left_pool, right_pool in itertools.combinations(WORKLOADS, 2):
                    for layer, value in enumerate(jaccard(fitted[phase, fold, left_pool], fitted[phase, fold, right_pool])):
                        stability.append({"model": key, "phase": phase, "kind": "cross_workload", "pool": f"{left_pool}_vs_{right_pool}",
                                          "left_fold": fold, "right_fold": fold, "repeat": 0, "layer": layer,
                                          "jaccard": float(value), "random_reference": float(chance_mean[layer]),
                                          "excess_jaccard": float(value-chance_mean[layer]), "shared_requests": 0})
            for pool in (*WORKLOADS, "mixed"):
                for left, right in itertools.combinations(range(5), 2):
                    for layer, value in enumerate(jaccard(fitted[phase,left,pool], fitted[phase,right,pool])):
                        shared = sum(assignment[c["manifest"]["request"]["group_id"]] not in (left, right) and
                                     (pool == "mixed" or c["manifest"]["request"]["workload"] == pool) and phase in c["phases"]
                                     for c in captures.values())
                        stability.append({"model": key, "phase": phase, "kind": "fold_overlap", "pool": pool,
                                          "left_fold": left, "right_fold": right, "repeat": 0, "layer": layer,
                                          "jaccard": float(value), "random_reference": float(chance_mean[layer]),
                                          "excess_jaccard": float(value-chance_mean[layer]), "shared_requests": shared})
        print(f"analyzed {key}: {len(captures)} complete primary requests", flush=True)
    write_csv(output / "request.csv", requests)
    write_csv(output / "layer.csv", layers)
    write_csv(output / "windows.csv", windows)
    write_csv(output / "oracle.csv", oracle_rows)
    write_csv(output / "stability.csv", stability)
    write_csv(output / "rankings.csv", rank_rows)
    write_csv(output / "ranking_sensitivity.csv", sensitivity)
    summaries = bootstrap_summary(requests, protocol["bootstrap_resamples"])
    write_csv(output / "summary.csv", summaries)
    atomic_json(output / "analysis_complete.json", {"retention": fraction, "status": "complete",
                                                  "request_rows": len(requests), "summary_cells": len(summaries)})
