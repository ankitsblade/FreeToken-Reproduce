"""Summarize distinct real-layer states without pooling their q timings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from experiments.bandwidth.layer_replay_summary import summarize as summarize_q


def paired_interval(deltas: list[float], *, draws: int = 4000,
                    seed: int = 42) -> tuple[float, float, float]:
    rng = random.Random(seed)
    samples = sorted(statistics.median(rng.choices(deltas, k=len(deltas)))
                     for _ in range(draws))
    return statistics.median(deltas), samples[int(draws * .025)], samples[int(draws * .975)]


def label(path: Path) -> tuple[str, str]:
    name = path.stem
    model = "qwen" if name.startswith("qwen") else "gpt_oss" if name.startswith("gpt_oss") else "unknown"
    workload = next((kind for kind in ("code", "incident", "math") if kind in name),
                    "unspecified")
    return model, workload


def summarize(paths: list[Path], *, draws: int = 4000) -> list[dict]:
    out = []
    for path in paths:
        q_rows = summarize_q([path])
        config = json.loads(path.with_name(path.stem + "_config.json").read_text())
        with path.open(newline="") as f:
            raw = list(csv.DictReader(f))
        n = int(config["repeats"])
        m = int(config["m"])
        times = {(int(row["repeat"]), int(row["q"])): float(row["layer_wall_ms"])
                 for row in raw}
        medians = {int(row["q"]): float(row["layer_wall_ms_median"]) for row in q_rows}
        predicted = int(q_rows[0]["q_predicted"])
        oracle = min(medians, key=medians.get)
        nearest = min((q for q in range(m + 1) if q != predicted), key=medians.get)
        deltas = [times[i, predicted] - times[i, nearest] for i in range(n)]
        paired, low, high = paired_interval(deltas, draws=draws)
        model, workload = label(path)
        out.append({
            "source": str(path), "model": model, "workload": workload,
            "run_label": path.parent.name, "cpu_timing": bool(config.get("cpu_timing", False)),
            "routed_experts": ";".join(map(str, config.get("routed_experts", []))),
            "layer_id": config["layer_id"], "trace_step": config.get("trace_step", ""),
            "match_index": config.get("match_index", ""), "m": m, "n_per_q": n,
            "q_predicted": predicted, "q_oracle": oracle,
            "median_pred_ms": medians[predicted], "median_oracle_ms": medians[oracle],
            "penalty_pct": 100 * (medians[predicted] / medians[oracle] - 1),
            "nearest_other_q": nearest,
            "paired_pred_minus_other_median_ms": paired,
            "paired_ci95_low_ms": low, "paired_ci95_high_ms": high,
            "max_abs_output_diff_vs_q0": max(
                float(row["max_abs_output_diff_vs_q0"]) for row in raw),
            "scope": "single_captured_state_exploratory_paired_bootstrap",
        })
    return sorted(out, key=lambda row: (row["model"], row["m"], row["workload"], row["source"]))


def by_m(states: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in states:
        groups[(row["model"], row["m"], row.get("cpu_timing", False))].append(row)
    out = []
    for (model, m, instrumented), rows in sorted(groups.items()):
        penalties = [row["penalty_pct"] for row in rows]
        out.append({
            "model": model, "m": m, "cpu_timing": instrumented, "states": len(rows),
            "states_where_predicted_is_oracle": sum(
                row["q_predicted"] == row["q_oracle"] for row in rows),
            "median_penalty_pct": statistics.median(penalties),
            "max_penalty_pct": max(penalties),
            "scope": "captured_states_only_not_traffic_weighted",
        })
    return out


def by_case(states: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in states:
        groups[(Path(row["source"]).stem, row["cpu_timing"])].append(row)
    out = []
    for (case, instrumented), group in sorted(groups.items()):
        selections = {(row["layer_id"], row["m"], row["match_index"],
                       row["trace_step"], row["routed_experts"]) for row in group}
        out.append({
            "case": case, "cpu_timing": instrumented, "runs": len(group),
            "distinct_run_labels": len({row["run_label"] for row in group}),
            "matching_selection_metadata": len(selections) == 1,
            "q_oracles": ";".join(map(str, sorted({row["q_oracle"] for row in group}))),
            "runs_predicted_is_oracle": sum(row["q_predicted"] == row["q_oracle"]
                                            for row in group),
            "runs_paired_ci_favors_alternative": sum(row["paired_ci95_low_ms"] > 0
                                                     for row in group),
            "min_penalty_pct": min(row["penalty_pct"] for row in group),
            "max_penalty_pct": max(row["penalty_pct"] for row in group),
            "scope": "independent_runs_not_pooled_trials_or_cross_run_identical_snapshots",
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw", type=Path, nargs="+")
    source.add_argument("--case-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    if any(args.out.with_name(args.out.stem + suffix).exists()
           for suffix in (".csv", "_by_m.csv", "_by_case.csv")):
        parser.error("output already exists")
    if args.draws < 100:
        parser.error("--draws must be >=100")
    paths = (args.raw if args.raw else [
        config.with_name(config.name.removesuffix("_config.json") + ".csv")
        for config in sorted(args.case_dir.glob("*_config.json"))
    ])
    if not paths:
        parser.error("no replay configurations found")
    states = summarize(paths, draws=args.draws)
    write_csv(args.out, states)
    write_csv(args.out.with_name(args.out.stem + "_by_m.csv"), by_m(states))
    write_csv(args.out.with_name(args.out.stem + "_by_case.csv"), by_case(states))
    print(args.out)


if __name__ == "__main__":
    main()
