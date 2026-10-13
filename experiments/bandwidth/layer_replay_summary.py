"""Combine controlled real-layer q replay logs without mixing captured states."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from freetoken.moe.layer_replay import predicted_q


METRICS = ("layer_wall_ms", "pcie_device_ms", "gpu_gemm_device_ms")
OPTIONAL_METRICS = ("cpu_task_ms", "cpu_effective_gbps")


def summarize(paths: list[Path]) -> list[dict]:
    out = []
    seen_paths = set()
    for path in paths:
        with path.open(newline="") as f:
            raw = list(csv.DictReader(f))
        config = json.loads(path.with_name(path.stem + "_config.json").read_text())
        layer = int(config["layer_id"])
        m = int(config["m"])
        repeats = int(config["repeats"])
        if path in seen_paths or not raw:
            raise ValueError("each captured state needs one nonempty replay file")
        seen_paths.add(path)
        groups = {q: [row for row in raw if int(row["q"]) == q]
                  for q in range(m + 1)}
        if len(raw) != (m + 1) * repeats or any(
                {int(row["repeat"]) for row in group} != set(range(repeats))
                or any(int(row["actual_m"]) != m or int(row["actual_q"]) != q
                       for row in group)
                for q, group in groups.items()):
            raise ValueError(f"incomplete or inconsistent q replay: {path}")
        medians = {q: statistics.median(float(row["layer_wall_ms"]) for row in group)
                   for q, group in groups.items()}
        oracle = min(medians, key=medians.get)
        predicted = predicted_q(m, float(config["fraction"]))
        for q, group in groups.items():
            result = {
                "source": str(path), "layer_id": layer, "m": m, "q": q,
                "match_index": config.get("match_index", ""),
                "trace_step": config.get("trace_step", ""),
                "cpu_timing": bool(config.get("cpu_timing", False)),
                "n": repeats, "q_predicted": predicted, "q_oracle": oracle,
                "q_error": predicted - oracle,
                "latency_penalty_predicted": medians[predicted] / medians[oracle] - 1,
                "expert_bytes": int(config["expert_bytes"]),
                "cache_slots": int(config["cache_slots"]),
                "max_abs_output_diff_vs_q0": max(
                    float(row["max_abs_output_diff_vs_q0"]) for row in group),
                "scope": "production_layer_fixed_input_and_cache",
            }
            for metric in METRICS + OPTIONAL_METRICS:
                if metric in OPTIONAL_METRICS and not any(row.get(metric) for row in group):
                    for suffix in ("median", "mean", "stdev", "min", "max"):
                        result[metric + "_" + suffix] = ""
                    continue
                values = [float(row[metric]) for row in group]
                result[metric + "_median"] = statistics.median(values)
                result[metric + "_mean"] = statistics.mean(values)
                result[metric + "_stdev"] = statistics.stdev(values)
                result[metric + "_min"] = min(values)
                result[metric + "_max"] = max(values)
            out.append(result)
    return sorted(out, key=lambda row: (row["layer_id"], row["source"], row["q"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    rows = summarize(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
