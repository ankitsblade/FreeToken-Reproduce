"""Generate study figures from available measured and simulated CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finish(plt, out: Path, name: str) -> None:
    plt.tight_layout()
    plt.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close()


def line(plt, out: Path, name: str, series: dict[str, list[tuple[float, float]]],
         xlabel: str, ylabel: str, *, log_x: bool = False,
         zero_y: bool = False, legend_outside: bool = False) -> None:
    if not series:
        return
    plt.figure(figsize=(7, 4))
    for label, points in series.items():
        grouped = defaultdict(list)
        for x, y in points:
            grouped[x].append(y)
        xs = sorted(grouped)
        plt.plot(xs, [statistics.median(grouped[x]) for x in xs], marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_x:
        plt.xscale("log")
    if zero_y:
        ymax = max(y for points in series.values() for _, y in points)
        plt.ylim(0, ymax * 1.1 if ymax else 1)
    if legend_outside:
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    else:
        plt.legend()
    finish(plt, out, name)


def bars(plt, out: Path, name: str, labels: list[str], values: list[float], ylabel: str) -> None:
    if not labels:
        return
    plt.figure(figsize=(max(7, len(labels) * .8), 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    finish(plt, out, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--trace", action="append", help="raw decode trace CSV (repeat for models)")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    root = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    calibration = rows(root / "baseline/calibration_repeat5.csv")
    if not calibration:
        calibration = rows(root / "baseline/calibration.csv")
    if calibration:
        grouped = defaultdict(list)
        for row in calibration:
            for metric in ("B_P_GBps", "B_H_GBps"):
                if row[metric]:
                    grouped[f"{row['model']} {metric}"].append(float(row[metric]))
        bars(plt, out, "baseline_bandwidth_capacity", list(grouped),
             [statistics.median(x) for x in grouped.values()], "GB/s")

    candidates = [row for path in sorted((root / "baseline").glob("*.jsonl"))
                  for row in jsonl_rows(path)
                  if row.get("status", "ok") == "ok" and "decode_tok_s" in row]
    longest = {}
    for row in candidates:
        longest[row["model"]] = max(longest.get(row["model"], 0), row["decode_steps"])
    baseline = [row for row in candidates if row["decode_steps"] == longest[row["model"]]]
    if baseline:
        grouped = defaultdict(list)
        for row in baseline:
            key = (f"{Path(row['model']).name} {row['backend']} "
                   f"cache={row['cache_rate_requested']} overlap={row['prefill_overlap']}")
            grouped[key].append(float(row["decode_tok_s"]))
        bars(plt, out, "baseline_decode_tps", list(grouped),
             [statistics.median(x) for x in grouped.values()], "decode tokens/s")

    prefill_paths = sorted((root / "baseline").glob("*prefill*summary.csv"))
    if any("decode8" in path.name for path in prefill_paths):
        prefill_paths = [path for path in prefill_paths if "decode8" in path.name]
    prefill = [row for path in prefill_paths for row in rows(path)]
    if prefill:
        series = defaultdict(list)
        for row in prefill:
            label = (f"{Path(row['model']).name} cache={row['cache_rate']} "
                     f"mem={row.get('memory_ratio', '0.9')} "
                     f"overlap={row['prefill_overlap']}")
            series[label].append((float(row["actual_prompt_tokens_median"]),
                                  float(row["ttft_ms_median"])))
        line(plt, out, "baseline_prefill_overlap_ttft", series,
             "prompt tokens", "TTFT (ms)", legend_outside=True)

    transfers = rows(root / "transfer_microbench/banked/contiguous_vs_banked.csv")
    if not transfers:
        transfers = rows(root / "transfer_microbench/raw.csv")
    if transfers:
        by_bytes = defaultdict(list)
        by_latency = defaultdict(list)
        by_batch = defaultdict(list)
        paired_latency = defaultdict(list)
        for row in transfers:
            if not row["effective_GBps"]:
                continue
            model = "generic buffers" if row["model"].startswith("generic-") else row["model"]
            label = f"{model} {row['method']}" if model != "generic buffers" else model
            size = float(row["batch_bytes"]) / 2**20
            by_bytes[label].append((size, float(row["effective_GBps"])))
            by_latency[label].append((size, float(row["transfer_ms"])))
            if model != "generic buffers":
                by_batch[label].append((float(row["batch_experts"]), float(row["effective_GBps"])))
                paired_latency[(model, int(row["batch_experts"]), row["method"])].append(
                    float(row["transfer_ms"]))
        line(plt, out, "transfer_bandwidth_vs_bytes", by_bytes,
             "batch size (MiB)", "effective H2D GB/s", log_x=True, zero_y=True)
        line(plt, out, "transfer_latency_vs_bytes", by_latency,
             "batch size (MiB)", "transfer ms", log_x=True, zero_y=True)
        line(plt, out, "transfer_bandwidth_vs_batch", by_batch,
             "experts per batch", "effective H2D GB/s", zero_y=True)
        bank_penalty = defaultdict(list)
        for model, batch, method in paired_latency:
            if method != "banked_pinned_H2D":
                continue
            flat = paired_latency.get((model, batch, "contiguous_pinned_H2D"))
            if flat:
                banked = statistics.median(paired_latency[(model, batch, method)])
                bank_penalty[model].append((batch, 100 * (banked / statistics.median(flat) - 1)))
        line(plt, out, "transfer_banked_latency_penalty", bank_penalty,
             "experts per batch", "six-bank copy latency penalty (%)")

    traffic_paths = sorted((root / "traffic_trace").glob("*summary.csv"))
    traffic_paths = [path for path in traffic_paths
                     if not (path.name.endswith("cache20_summary.csv") and
                             path.with_name(path.stem.removesuffix("_summary") +
                                            "_greedy_summary.csv").exists())]
    traffic = [row for path in traffic_paths for row in rows(path)]
    if traffic:
        labels = [f"{Path(r['model_id']).name} cache={int(r['cache_slots']) / (int(r['moe_layers']) * int(r['experts_per_layer'])):.2f}"
                  for r in traffic]
        bars(plt, out, "traffic_misses_per_token", labels,
             [float(r["cache_misses_per_token"]) for r in traffic], "unique misses/token")
        bars(plt, out, "traffic_bytes_per_token", labels,
             [float(r["bytes_h2d_per_token"]) for r in traffic], "H2D bytes/token")
        bars(plt, out, "traffic_nominal_cpu_expert_bytes_per_token", labels,
             [float(r["nominal_cpu_expert_bytes_per_token"]) for r in traffic],
             "CPU-executed expert weight bytes/token (nominal)")
        capacity = defaultdict(list)
        for row in traffic:
            total = int(row["moe_layers"]) * int(row["experts_per_layer"])
            if total:
                capacity[Path(row["model_id"]).name].append(
                    (int(row["cache_slots"]) / total, float(row["miss_rate"]))
                )
        line(plt, out, "baseline_cache_miss_vs_capacity", capacity,
             "expert cache fraction", "decode miss fraction", zero_y=True)

    trace_paths = ([Path(path) for path in args.trace] if args.trace else
                   sorted((root / "traffic_trace").glob("*hybrid_cache20_greedy.csv")))
    if not trace_paths:
        trace_paths = sorted((root / "traffic_trace").glob("*hybrid_cache20.csv"))
    trace = [row for path in trace_paths for row in rows(path)]
    if trace:
        by_m = defaultdict(list)
        hits = defaultdict(list)
        m_values = []
        for row in trace:
            m = int(row["m"])
            m_values.append(m)
            if m:
                by_m[row["model_id"]].append((m, int(row["gpu_fill_count"]) / m))
            unique = int(row["unique_routed_count"])
            if unique:
                hits[row["model_id"]].append(
                    (int(row["layer_id"]), int(row["hit_count"]) / unique)
                )
        plt.figure(figsize=(7, 4))
        plt.hist(m_values, bins=range(max(m_values) + 2), align="left")
        plt.xlabel("unique missing experts per layer (m)")
        plt.ylabel("layer calls")
        finish(plt, out, "traffic_m_distribution")
        line(plt, out, "traffic_fill_fraction_vs_m", by_m, "m", "GPU fills / misses")
        line(plt, out, "traffic_hit_rate_by_layer", hits, "MoE layer", "cache hit rate")

    popularity = [row for path in sorted((root / "traffic_trace").glob("*popularity.csv"))
                  for row in rows(path) if row["scope"] == "global"]
    if popularity:
        coverage = defaultdict(list)
        for row in popularity:
            coverage[Path(row["model_id"]).name].append(
                (float(row["retained_fraction"]), float(row["access_coverage"]))
            )
        line(plt, out, "expert_popularity_coverage", coverage,
             "fraction of experts retained", "observed routed access coverage", zero_y=True)

    replay = rows(root / "geometry_replay/summary.csv")
    if replay:
        specs = [
            ("geometry_miss_vs_E", "E", "miss_rate", lambda r: int(r["k"]) == 4 and float(r["relative_cache_capacity"]) == .2,
             "experts/layer", "miss fraction"),
            ("geometry_miss_vs_k", "k", "miss_rate", lambda r: int(r["E"]) == 128 and float(r["relative_cache_capacity"]) == .2,
             "top-k", "miss fraction"),
            ("geometry_bytes_vs_k", "k", "bytes_h2d_per_token", lambda r: int(r["E"]) == 128 and float(r["relative_cache_capacity"]) == .2,
             "top-k", "H2D bytes/token"),
            ("geometry_miss_vs_cache_fraction", "relative_cache_capacity", "miss_rate",
             lambda r: int(r["E"]) == 128 and int(r["k"]) == 4,
             "cache fraction of expert pool", "miss fraction"),
            ("geometry_hit_vs_k_over_E", "k_over_E", "hit_rate",
             lambda r: float(r["relative_cache_capacity"]) == .2,
             "k/E", "LRU hit fraction"),
        ]
        for name, xkey, ykey, select, xlabel, ylabel in specs:
            series = defaultdict(list)
            for row in replay:
                if not select(row):
                    continue
                x = int(row["k"]) / int(row["E"]) if xkey == "k_over_E" else float(row[xkey])
                y = 1 - float(row["miss_rate"]) if ykey == "hit_rate" else float(row[ykey])
                series[row["profile"]].append((x, y))
            line(plt, out, name, series, xlabel, ylabel)
    measured_replay = [row for path in sorted((root / "geometry_replay").glob("*trace_replay.csv"))
                       for row in rows(path)]
    if measured_replay:
        for name, xkey, ykey, xlabel, ylabel in (
            ("trace_replay_miss_vs_E", "E", "miss_rate", "experts/layer", "miss fraction"),
            ("trace_replay_h2d_vs_k", "k", "bytes_h2d_per_token", "top-k", "H2D bytes/token"),
            ("trace_replay_miss_vs_cache_bytes", "cache_bytes", "miss_rate", "cache bytes", "miss fraction"),
        ):
            series = defaultdict(list)
            for row in measured_replay:
                if row.get("capacity_mode", "relative") != "relative":
                    continue
                native_e = 256 if row["profile"].startswith("qwen") else 32
                if name == "trace_replay_h2d_vs_k" and (int(row["E"]) != native_e or float(row["relative_cache_capacity"]) != .2):
                    continue
                if name == "trace_replay_miss_vs_E" and (int(row["k"]) != 4 or float(row["relative_cache_capacity"]) != .2):
                    continue
                if name == "trace_replay_miss_vs_cache_bytes" and (int(row["k"]) != 4 or int(row["E"]) != native_e):
                    continue
                series[row["profile"]].append((float(row[xkey]), float(row[ykey])))
            line(plt, out, name, series, xlabel, ylabel)
        fixed = defaultdict(list)
        for row in measured_replay:
            if row.get("capacity_mode") == "raw_slots" and int(row["k"]) == 4:
                fixed[f"{row['profile']} slots={row['cache_slots']}"].append(
                    (int(row["E"]), float(row["miss_rate"])))
        line(plt, out, "trace_replay_miss_vs_E_fixed_slots", fixed,
             "experts/layer", "miss fraction")

    counterfactual = [row for path in sorted((root / "geometry_replay/counterfactual").glob("*.csv"))
                      for row in rows(path)]
    if counterfactual:
        for mode, name in (("relative", "counterfactual_miss_vs_E_relative"),
                           ("raw_slots", "counterfactual_miss_vs_E_fixed_slots")):
            series = defaultdict(list)
            for row in counterfactual:
                if row["capacity_mode"] != mode or int(row["k"]) != 4:
                    continue
                if mode == "relative" and float(row["relative_cache_capacity"]) != .2:
                    continue
                label = row["profile"] if mode == "relative" else f"{row['profile']} slots={row['cache_slots']}"
                series[label].append((int(row["E"]), float(row["miss_rate"])))
            line(plt, out, name, series, "experts/layer", "miss fraction")

    q = [row for path in sorted((root / "qstar_oracle").glob("*summary.csv"))
         for row in rows(path) if "median_ms_per_token" in row]
    if q:
        series = defaultdict(list)
        for row in q:
            series[row["model"]].append((float(row["q_cap"]), float(row["median_ms_per_token"])))
        line(plt, out, "q_cap_decode_latency", series, "forced q cap (-1 = auto)",
             "median decode ms/token")

    oracle_dir = root / "qstar_oracle"
    matched_qwen = oracle_dir / "qwen_matched_layer_summary.csv"
    targeted_qwen = oracle_dir / "qwen_m4_matched_layer_summary.csv"
    layer_q = rows(oracle_dir / "gpt_oss_layer_summary.csv")
    original_qwen = rows(oracle_dir / "qwen_layer_summary.csv")
    chosen_qwen = {}
    for source in (original_qwen, rows(matched_qwen), rows(targeted_qwen)):
        for row in source:
            if row["complete_q_sweep"] == "True":
                chosen_qwen[(row["m"], row["q"])] = row
    layer_q += list(chosen_qwen.values())
    if layer_q:
        latency = defaultdict(list)
        predictions = defaultdict(list)
        penalties_by_m = defaultdict(list)
        penalties_by_size = defaultdict(list)
        penalties_by_batch = defaultdict(list)
        expert_sizes = {Path(row["model_id"]).name: int(row["expert_bytes"])
                        for row in traffic}
        for row in layer_q:
            model = Path(row["model"]).name
            m = int(row["m"])
            latency[f"{model} m={m}"].append((float(row["q"]),
                                               float(row["median_layer_ms"])))
            if (row["complete_q_sweep"] == "True" and
                    int(row.get("min_samples_per_q", 0)) >= 30 and
                    int(row["q"]) == int(row["q_oracle"])):
                predictions[f"{model} predicted"].append((m, float(row["q_predicted"])))
                predictions[f"{model} oracle"].append((m, float(row["q_oracle"])))
                penalty = 100 * float(row["latency_penalty_predicted"])
                penalties_by_m[model].append((m, penalty))
                if model in expert_sizes:
                    penalties_by_size[model].append((expert_sizes[model] / 2**20, penalty))
                    penalties_by_batch[model].append((m * expert_sizes[model] / 2**20, penalty))
        line(plt, out, "q_layer_latency_vs_q", latency, "actual GPU fills q", "layer latency (ms)")
        line(plt, out, "q_predicted_vs_oracle", predictions, "misses per layer m", "q")
        line(plt, out, "q_penalty_vs_m", penalties_by_m, "misses per layer m", "latency penalty (%)")
        line(plt, out, "q_penalty_vs_expert_size", penalties_by_size,
             "expert size (MiB)", "latency penalty (%)")
        line(plt, out, "q_penalty_vs_batch_bytes", penalties_by_batch,
             "missing expert bytes/layer (MiB)", "latency penalty (%)")

    branch_oracles = []
    for name in ("qwen", "gpt_oss"):
        data = rows(root / "qstar_oracle" / f"{name}_branch_oracle_raw_summary.csv")
        branch_oracles.extend(data)
        if data:
            wall = defaultdict(list)
            for row in data:
                wall[f"m={row['m']}"].append((int(row["q"]),
                                             float(row["overlap_wall_ms_median"])))
            line(plt, out, f"{name}_branch_oracle_wall_vs_q", wall,
                 "GPU fills q", "CPU + PCIe branch overlap wall time (ms)")
            m4 = [row for row in data if int(row["m"]) == 4]
            if m4:
                branches = defaultdict(list)
                for row in m4:
                    for key, label in (("cpu_branch_ms_median", "CPU MoE"),
                                       ("pcie_device_ms_median", "PCIe gather"),
                                       ("overlap_wall_ms_median", "overlap wall")):
                        branches[label].append((int(row["q"]), float(row[key])))
                line(plt, out, f"{name}_branch_oracle_m4_branches", branches,
                     "GPU fills q", "branch time (ms)")
    if branch_oracles:
        predictions = defaultdict(list)
        for row in branch_oracles:
            if int(row["q"]) == int(row["q_oracle"]):
                name = row["model"]
                predictions[f"{name} predicted"].append((int(row["m"]),
                                                        int(row["q_predicted"])))
                predictions[f"{name} branch oracle"].append((int(row["m"]),
                                                            int(row["q_oracle"])))
        line(plt, out, "q_branch_predicted_vs_oracle", predictions,
             "fixed misses per layer m", "q")

    controlled = rows(oracle_dir / "qwen_m4_controlled_layer_summary.csv")
    if controlled:
        wall = defaultdict(list)
        stages = defaultdict(list)
        for row in controlled:
            layer = row["layer_id"]
            q_value = int(row["q"])
            wall[f"layer {layer}"].append((q_value, float(row["layer_wall_ms_median"])))
            for key, label in (("pcie_device_ms_median", "PCIe gather"),
                               ("gpu_gemm_device_ms_median", "GPU GEMM")):
                stages[f"layer {layer} {label}"].append((q_value, float(row[key])))
        line(plt, out, "qwen_m4_controlled_layer_wall_vs_q", wall,
             "GPU fills q", "same-state full MoE layer wall time (ms)")
        line(plt, out, "qwen_m4_controlled_device_stages_vs_q", stages,
             "GPU fills q", "device stage time (ms)", legend_outside=True)

    representative = rows(oracle_dir / "representative_states/state_summary.csv")
    if representative:
        plt.figure(figsize=(7, 4))
        markers = {"code": "o", "incident": "s", "math": "^"}
        for model in sorted({row["model"] for row in representative}):
            for workload in sorted({row["workload"] for row in representative
                                    if row["model"] == model}):
                selected = [row for row in representative
                            if row["model"] == model and row["workload"] == workload]
                positions = defaultdict(int)
                counts = {m: sum(int(row["m"]) == m for row in selected)
                          for m in {int(row["m"]) for row in selected}}
                xs = []
                for row in selected:
                    m = int(row["m"])
                    model_shift = -.09 if model == "qwen" else .09
                    workload_shift = {"code": -.025, "incident": .025}.get(workload, 0)
                    xs.append(m + model_shift + workload_shift +
                              (positions[m] - (counts[m] - 1) / 2) * .035)
                    positions[m] += 1
                plt.scatter(xs,
                            [float(row["penalty_pct"]) for row in selected],
                            marker=markers.get(workload, "x"), label=f"{model} {workload}")
        plt.axhline(0, color="gray", linewidth=.8)
        plt.xlabel("missing experts per layer m")
        plt.ylabel("predicted-q penalty in captured state (%)")
        plt.legend()
        finish(plt, out, "q_representative_state_penalty_vs_m")

    final_layers = rows(root / "final_study/layer_summary.csv")
    for m in sorted({int(row["m"]) for row in final_layers}):
        selected = [row for row in final_layers if int(row["m"]) == m
                    and row["cpu_timing"] == "False"]
        if m == 5:
            selected = [row for row in selected
                        if "qwen_independent_run2_extended" in row["source"]]
        series = defaultdict(list)
        for row in selected:
            source = Path(row["source"])
            series[f"{source.parent.name}/{source.stem}"].append(
                (int(row["q"]), float(row["layer_wall_ms_median"])))
        line(plt, out, f"q_full_layer_m{m}_wall_vs_q", series,
             "GPU fills q", "same-state full-layer wall time (ms)", legend_outside=True)

    replicated = rows(root / "final_study/state_summary.csv")
    selected = [row for row in replicated if int(row["m"]) == 5
                and row["cpu_timing"] == "False"]
    if selected:
        display = {"representative_states": "original", "qwen_m5_independent_run1": "repeat 1",
                   "qwen_independent_run2_extended": "repeat 2"}
        order = list(display)
        labels = sorted({row["run_label"] for row in selected},
                        key=lambda label: order.index(label) if label in order else len(order))
        series = defaultdict(list)
        for row in selected:
            series[Path(row["source"]).stem].append(
                (labels.index(row["run_label"]), float(row["paired_pred_minus_other_median_ms"])))
        plt.figure(figsize=(8, 4))
        for case, points in sorted(series.items()):
            points = sorted(points)
            plt.plot([x for x, _ in points], [y for _, y in points], marker="o",
                     label=case.removeprefix("qwen_").removesuffix("_m5"))
        plt.axhline(0, color="gray", linewidth=.8)
        plt.xticks(range(len(labels)), [display.get(label, label) for label in labels])
        plt.ylabel("paired q* minus alternative (ms)")
        plt.title("Qwen m=5: independent captures")
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        finish(plt, out, "q_m5_independent_run_deltas")

    profiled = rows(oracle_dir / "qwen_m4_torch_profile_summary.csv")
    if profiled:
        activity = {"CUDA kernel active time": [
            (int(row["q"]), float(row["cuda_kernel_active_pct_median"]))
            for row in profiled]}
        line(plt, out, "qwen_m4_profiled_cuda_activity_vs_q", activity,
             "GPU fills q", "CUDA kernel active fraction of profiled layer (%)")

    nsys_profile = rows(oracle_dir / "nsys2025_qwen_m5_stages.csv")
    if nsys_profile:
        stages = defaultdict(list)
        for row in nsys_profile:
            for key, label in (("nvfp4_expert_kernel_active_ms_median", "expert GEMM kernels"),
                               ("gather_kernel_active_ms_median", "gather kernel")):
                stages[label].append((int(row["q"]), float(row[key])))
        line(plt, out, "qwen_m5_nsight_kernel_stages", stages,
             "GPU fills q", "profiled CUDA kernel active time (ms)")

    monitor = rows(oracle_dir / "qwen_m4_monitor_summary_by_q.csv")
    if monitor:
        utilization = defaultdict(list)
        for row in monitor:
            for key, label in (("gpu_utilization_pct_median", "GPU"),
                               ("host_cpu_utilization_pct_median", "whole-host CPU")):
                if row[key]:
                    utilization[label].append((int(row["q"]), float(row[key])))
        line(plt, out, "qwen_q_sweep_system_utilization", utilization,
             "forced q cap", "utilization (%)")

    pruning = [row for path in sorted((root / "pruning_optional").glob("*_temporal.csv"))
               for row in rows(path) if row["scope"] == "global" and row["test_window"] == "all"]
    if pruning:
        coverage = defaultdict(list)
        for row in pruning:
            model = Path(row["model_id"]).name
            fraction = float(row["retained_fraction"])
            coverage[f"{model} train"].append((fraction,
                                               float(row["train_access_coverage"])))
            coverage[f"{model} held-out"].append((fraction,
                                                  float(row["test_access_coverage"])))
        line(plt, out, "pruning_temporal_holdout_coverage", coverage,
             "retained expert fraction", "routed access coverage")

    transfer = [row for path in sorted((root / "pruning_optional").glob("*_math_to_*.csv"))
                for row in rows(path) if row["scope"] == "global" and
                row["test_window"] == "all"]
    if transfer:
        coverage = defaultdict(list)
        for row in transfer:
            label = f"{Path(row['model_id']).name} {row['test_workload']}"
            coverage[label].append((float(row["retained_fraction"]),
                                    float(row["test_access_coverage"])))
        line(plt, out, "pruning_cross_workload_coverage", coverage,
             "retained expert fraction", "held-out routed access coverage")
    print(out)


if __name__ == "__main__":
    main()
