"""Estimate held-out routed access coverage under hypothetical expert retention."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def measured_rows(trace: Path, benchmark: Path | None, row_index: int) -> list[dict]:
    with trace.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if benchmark:
        records = [json.loads(line) for line in benchmark.read_text().splitlines()]
        record = records[row_index]
        rows = rows[record["trace_start_row"]:record["trace_end_row"]]
        if {row["run_id"] for row in rows} != {record["trace_run_id"]}:
            raise ValueError("benchmark range does not match trace run")
    if not rows or len({row["run_id"] for row in rows}) != 1:
        raise ValueError("select exactly one nonempty trace run")
    return rows


def split_tokens(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    tokens = sorted({int(row["token_index"]) for row in rows})
    if len(tokens) < 2:
        raise ValueError("need at least two traced tokens for temporal holdout")
    first = set(tokens[:len(tokens) // 2])
    return ([row for row in rows if int(row["token_index"]) in first],
            [row for row in rows if int(row["token_index"]) not in first])


def access_counts(rows: list[dict], scope: str, layer: int | None = None) -> Counter:
    counts = Counter()
    for row in rows:
        row_layer = int(row["layer_id"])
        if layer is not None and row_layer != layer:
            continue
        for raw in row["routed_experts"].split(";"):
            expert = int(raw)
            key = (row_layer, expert) if scope == "global" else expert
            counts[key] += 1
    return counts


def evaluate_coverage(train: Counter, test: Counter, universe: list,
                      fraction: float) -> dict:
    kept = math.ceil(len(universe) * fraction)
    selected = set(sorted(universe, key=lambda key: (-train[key], key))[:kept])
    train_total = sum(train.values())
    test_total = sum(test.values())
    return {
        "retained_experts": kept,
        "train_access_coverage": (sum(train[key] for key in selected) / train_total
                                  if train_total else None),
        "test_access_coverage": (sum(test[key] for key in selected) / test_total
                                 if test_total else None),
        "test_removed_access_fraction": (sum(test[key] for key in test if key not in selected)
                                         / test_total if test_total else None),
        "test_accesses": test_total,
    }


def simulate(train_rows: list[dict], test_rows: list[dict], train_name: str,
             test_name: str) -> list[dict]:
    E = int(train_rows[0]["E"])
    layers = sorted({int(row["layer_id"]) for row in train_rows})
    if {int(row["E"]) for row in test_rows} != {E} or (
            {int(row["layer_id"]) for row in test_rows} != set(layers)):
        raise ValueError("train and test traces have different expert geometry")
    expert_bytes = int(train_rows[0]["expert_size_bytes"])
    test_tokens = sorted({int(row["token_index"]) for row in test_rows})
    windows = {token: min(3, 4 * i // len(test_tokens))
               for i, token in enumerate(test_tokens)}
    out = []
    for scope, layer in [("global", None), *(("layer", x) for x in layers)]:
        universe = ([(x, expert) for x in layers for expert in range(E)]
                    if layer is None else list(range(E)))
        train = access_counts(train_rows, scope, layer)
        for window in ("all", 0, 1, 2, 3):
            selected_rows = (test_rows if window == "all" else
                             [row for row in test_rows
                              if windows[int(row["token_index"])] == window])
            test = access_counts(selected_rows, scope, layer)
            for fraction in FRACTIONS:
                coverage = evaluate_coverage(train, test, universe, fraction)
                out.append({
                    "model_id": train_rows[0]["model_id"],
                    "train_workload": train_name, "test_workload": test_name,
                    "train_run_id": train_rows[0]["run_id"],
                    "test_run_id": test_rows[0]["run_id"],
                    "scope": scope, "layer_id": layer if layer is not None else "all",
                    "test_window": window,
                    "retained_fraction": fraction,
                    "expert_universe": len(universe),
                    "retained_bytes": coverage["retained_experts"] * expert_bytes,
                    **coverage,
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trace", type=Path, required=True)
    parser.add_argument("--train-benchmark", type=Path)
    parser.add_argument("--train-row", type=int, default=0)
    parser.add_argument("--test-trace", type=Path)
    parser.add_argument("--test-benchmark", type=Path)
    parser.add_argument("--test-row", type=int, default=0)
    parser.add_argument("--train-name", default="train")
    parser.add_argument("--test-name", default="test")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    train = measured_rows(args.train_trace, args.train_benchmark, args.train_row)
    if args.test_trace:
        test = measured_rows(args.test_trace, args.test_benchmark, args.test_row)
    else:
        train, test = split_tokens(train)
    rows = simulate(train, test, args.train_name, args.test_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
