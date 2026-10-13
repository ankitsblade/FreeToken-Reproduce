"""Validate and consolidate study artifacts without pooling captured states."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.bandwidth.layer_replay_summary import summarize as summarize_q
from experiments.bandwidth.representative_state_summary import by_case, by_m, summarize, write_csv


REQUIRED_FIGURES = (
    "baseline_bandwidth_capacity", "baseline_decode_tps", "baseline_prefill_overlap_ttft",
    "baseline_cache_miss_vs_capacity", "transfer_bandwidth_vs_bytes",
    "transfer_latency_vs_bytes", "transfer_bandwidth_vs_batch", "traffic_misses_per_token",
    "traffic_bytes_per_token", "traffic_m_distribution", "traffic_fill_fraction_vs_m",
    "traffic_hit_rate_by_layer", "geometry_miss_vs_E", "geometry_miss_vs_k",
    "geometry_bytes_vs_k", "geometry_miss_vs_cache_fraction", "geometry_hit_vs_k_over_E",
    "q_layer_latency_vs_q", "q_predicted_vs_oracle", "q_penalty_vs_expert_size",
    "q_penalty_vs_m",
)


def audit_figures(root: Path) -> dict:
    missing = [name for name in REQUIRED_FIGURES
               if not (root / (name + ".png")).is_file()
               or (root / (name + ".png")).stat().st_size == 0]
    return {"required": len(REQUIRED_FIGURES), "missing": missing,
            "all_present": not missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--raw", type=Path, nargs="*", default=[])
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("use a fresh output directory")
    paths = list(args.raw)
    for folder in args.case_dir:
        configs = sorted(folder.glob("*_config.json"))
        if not configs:
            parser.error(f"no completed replay states in {folder}")
        paths.extend(config.with_name(config.name.removesuffix("_config.json") + ".csv")
                     for config in configs)
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("duplicate raw replay paths")
    q_rows = summarize_q(paths)
    states = summarize(paths)
    args.out_dir.mkdir(parents=True)
    write_csv(args.out_dir / "layer_summary.csv", q_rows)
    write_csv(args.out_dir / "state_summary.csv", states)
    write_csv(args.out_dir / "state_summary_by_m.csv", by_m(states))
    write_csv(args.out_dir / "state_summary_by_case.csv", by_case(states))
    archives = []
    unavailable_logs = []
    for index, path in enumerate(paths):
        benchmark = path.with_name(path.stem + "_benchmark.jsonl")
        if not benchmark.is_file():
            unavailable_logs.append(str(benchmark))
            continue
        for row_index, line in enumerate(benchmark.read_text().splitlines()):
            record = json.loads(line)
            source = Path(record["server_log"])
            if not source.is_file():
                unavailable_logs.append(str(source))
                continue
            target = args.out_dir / "server_logs" / f"{index:02d}_{path.stem}_{row_index}.log"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
            archives.append({"raw": str(path), "original": str(source), "archive": str(target)})
    audit = {
        "raw_sources": [str(path) for path in paths], "captured_states": len(states),
        "measured_trials": sum(int(row["n"]) for row in q_rows),
        "validated": "complete q cells, repeat ids, realized miss and fill counts",
        "figures": audit_figures(args.figures), "archived_logs": archives,
        "unavailable_logs": unavailable_logs,
        "scope": "finite captured states; not exhaustive workloads, quality, or hardware counters",
    }
    (args.out_dir / "artifact_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("captured_states", "measured_trials", "figures")}))


if __name__ == "__main__":
    main()
