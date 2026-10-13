"""Align sampled GPU/CPU state with traced q-sweep runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


METRICS = {
    "utilization.gpu": "gpu_utilization_pct",
    "utilization.memory": "gpu_memory_utilization_pct",
    "power.draw": "gpu_power_w",
    "clocks.current.sm": "gpu_sm_mhz",
    "temperature.gpu": "gpu_temperature_c",
    "cpu_utilization_pct": "host_cpu_utilization_pct",
    "cpu_mhz_mean": "host_cpu_mhz_mean",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    benchmark = [json.loads(line) for line in args.benchmark.read_text().splitlines()]
    trace = read_csv(args.trace)
    monitor = []
    for row in read_csv(args.monitor):
        stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").astimezone()
        monitor.append((stamp, row))
    out = []
    for index, record in enumerate(benchmark):
        selected = trace[record["trace_start_row"]:record["trace_end_row"]]
        if not selected or {row["run_id"] for row in selected} != {record["trace_run_id"]}:
            raise ValueError("benchmark range does not match trace run")
        start = datetime.fromisoformat(selected[0]["timestamp"])
        end = datetime.fromisoformat(selected[-1]["timestamp"])
        samples = [row for stamp, row in monitor if start <= stamp <= end]
        result = {"run_index": index, "run_id": record["trace_run_id"],
                  "model": record["model"], "q": record["hybrid_fetch"],
                  "sample_count": len(samples), "scope": "measured_decode_trace"}
        for source, label in METRICS.items():
            values = [float(row[source]) for row in samples if row[source]]
            result[label + "_median"] = statistics.median(values) if values else ""
            result[label + "_min"] = min(values) if values else ""
            result[label + "_max"] = max(values) if values else ""
        out.append(result)
    write_csv(args.out, out)
    grouped = defaultdict(list)
    for row in out:
        if row["sample_count"]:
            grouped[row["q"]].append(row)
    summary = []
    for q, group in sorted(grouped.items()):
        result = {"model": group[0]["model"], "q": q, "runs_with_samples": len(group),
                  "monitor_samples": sum(row["sample_count"] for row in group),
                  "scope": "measured_decode_trace"}
        for label in METRICS.values():
            values = [row[label + "_median"] for row in group
                      if row[label + "_median"] != ""]
            result[label + "_median"] = statistics.median(values) if values else ""
        summary.append(result)
    if summary:
        write_csv(args.out.with_name(args.out.stem + "_by_q.csv"), summary)
    print(args.out)


if __name__ == "__main__":
    main()
