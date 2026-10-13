"""Aggregate the q* policy sweep into summary CSVs.

Reads the per-cell perf JSONs (throughput/latency), trace CSVs (per-layer routing and
traffic over the *measured* decode window only), and system-monitor CSVs (CPU/GPU util
aligned to that window). Writes machine-readable summaries under results/summary/.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
OUT = study_dir("split_policy") / "results"
SUM = OUT / "summary"

POLICY_ORDER = ["A_adaptive", "B_gpu", "C_cpu", "D_fixed25", "E_fixed50", "F_fixed75"]
WORKLOAD_ORDER = ["low_miss", "med_miss", "high_miss"]
POLICY_LABEL = {
    "A_adaptive": "q* adaptive", "B_gpu": "q=m (GPU)", "C_cpu": "q=0 (CPU)",
    "D_fixed25": "q=25%", "E_fixed50": "q=50%", "F_fixed75": "q=75%",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def stdev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def measured_trace(cell_json: Path, trace_csv: Path) -> list[dict]:
    """Trace rows for the measured decode window only (bench records the row range)."""
    rows = read_jsonl(cell_json)
    trace = read_csv(trace_csv)
    if not rows or not trace:
        return []
    rec = rows[-1]
    start, end = rec.get("trace_start_row"), rec.get("trace_end_row")
    if start is None or end is None:
        return trace
    sel = trace[int(start):int(end)]
    run = rec.get("trace_run_id")
    return [r for r in sel if r.get("run_id") == run] if run else sel


def util_for_window(trace_rows: list[dict], monitor_csv: Path) -> dict:
    mon = read_csv(monitor_csv)
    if not trace_rows or not mon:
        return {}
    start = datetime.fromisoformat(trace_rows[0]["timestamp"])
    end = datetime.fromisoformat(trace_rows[-1]["timestamp"])
    picked = []
    for row in mon:
        try:
            stamp = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").astimezone()
        except (KeyError, ValueError):
            continue
        if start <= stamp <= end:
            picked.append(row)
    out = {"util_samples": len(picked)}
    keys = {"utilization.gpu": "gpu_util_pct", "utilization.memory": "gpu_mem_util_pct",
            "power.draw": "gpu_power_w", "cpu_utilization_pct": "cpu_util_pct",
            "cpu_mhz_mean": "cpu_mhz"}
    for src, label in keys.items():
        vals = [float(r[src]) for r in picked if r.get(src) not in (None, "")]
        out[label + "_median"] = statistics.median(vals) if vals else ""
    return out


def perf_summary() -> list[dict]:
    rows = []
    for wl in WORKLOAD_ORDER:
        for pol in POLICY_ORDER:
            recs = read_jsonl(OUT / "e2e" / "perf" / wl / f"{pol}.json")
            if not recs:
                continue
            def col(k):
                return [r.get(k) for r in recs]
            rows.append({
                "workload": wl, "policy": pol, "label": POLICY_LABEL[pol],
                "n": len(recs),
                "cache_rate": recs[0].get("cache_rate_requested"),
                "decode_tok_s_mean": mean(col("decode_tok_s")),
                "decode_tok_s_std": stdev(col("decode_tok_s")),
                "ms_per_token_mean": mean(col("ms_per_token")),
                "event_ms_p50_mean": mean(col("event_ms_p50")),
                "event_ms_p95_mean": mean(col("event_ms_p95")),
                "event_ms_p99_mean": mean(col("event_ms_p99")),
                "ttft_ms_mean": mean(col("ttft_ms")),
                "vram_gib": mean(col("vram_gib")),
                "output_sha1": recs[-1].get("output_sha1"),
            })
    return rows


def traffic_summary() -> tuple[list[dict], list[dict]]:
    """Per-cell routing/traffic over the measured decode window, plus per-m selected q."""
    cells, qbym = [], []
    for wl in WORKLOAD_ORDER:
        for pol in POLICY_ORDER:
            cj = OUT / "e2e" / "trace" / wl / f"{pol}.json"
            tc = OUT / "e2e" / "trace" / wl / f"{pol}.trace.csv"
            tr = measured_trace(cj, tc)
            if not tr:
                continue
            expert_bytes = int(tr[0]["expert_size_bytes"])
            n_layers = len({r["layer_id"] for r in tr})
            steps = len(tr) / n_layers if n_layers else 0
            m = [int(r["m"]) for r in tr]
            q = [int(r["gpu_fill_count"]) for r in tr]
            cpu_exec = [int(r["cpu_execute_count"]) for r in tr]
            h2d = [int(r["bytes_h2d"]) for r in tr]
            hit = [int(r["hit_count"]) for r in tr]
            uniq = [int(r["unique_routed_count"]) for r in tr]
            occ = [int(r["cache_occupancy"]) for r in tr]
            per_tok = len(tr) / steps if steps else 0  # layers per token
            util = util_for_window(tr, cj.with_suffix(".monitor.csv"))
            cells.append({
                "workload": wl, "policy": pol, "label": POLICY_LABEL[pol],
                "layer_calls": len(tr), "moe_layers": n_layers, "decode_steps": round(steps),
                "cache_slots": int(tr[0]["cache_slots"]),
                "expert_bytes": expert_bytes,
                "mean_m": mean(m), "mean_q": mean(q), "mean_cpu_exec": mean(cpu_exec),
                "mean_hit": mean(hit), "mean_unique": mean(uniq),
                "miss_rate": (sum(m) / sum(uniq)) if sum(uniq) else 0.0,
                "fetch_rate_q_over_m": (sum(q) / sum(m)) if sum(m) else 0.0,
                "bytes_h2d_per_token": mean(h2d) * per_tok,
                "cpu_bytes_per_token": mean([c * expert_bytes for c in cpu_exec]) * per_tok,
                "mean_cache_occupancy": mean(occ),
                **util,
            })
            # selected q as a function of m (correctness + policy-shape plots)
            bym = {}
            for mi, qi in zip(m, q):
                bym.setdefault(mi, []).append(qi)
            for mi in sorted(bym):
                qbym.append({"workload": wl, "policy": pol, "label": POLICY_LABEL[pol],
                             "m": mi, "n": len(bym[mi]), "mean_q": mean(bym[mi]),
                             "count_frac": len(bym[mi]) / len(m)})
    return cells, qbym


def correctness_report(cells: list[dict]) -> list[dict]:
    """Cross-policy check: same warm cache + greedy => identical routing (m) per workload.

    Compares the measured-window per-(m) routing counts across policies; q differs by
    design but the miss set m and its distribution should match if semantics are preserved.
    """
    rows = []
    for wl in WORKLOAD_ORDER:
        cj = {p: OUT / "e2e" / "trace" / wl / f"{p}.json" for p in POLICY_ORDER}
        tc = {p: OUT / "e2e" / "trace" / wl / f"{p}.trace.csv" for p in POLICY_ORDER}
        dists = {}
        shas = {}
        for p in POLICY_ORDER:
            tr = measured_trace(cj[p], tc[p])
            if not tr:
                continue
            from collections import Counter
            dists[p] = Counter(int(r["m"]) for r in tr)
            recs = read_jsonl(OUT / "e2e" / "perf" / wl / f"{p}.json")
            shas[p] = recs[-1].get("output_sha1") if recs else None
        if not dists:
            continue
        ref = dists.get("C_cpu") or next(iter(dists.values()))
        for p, d in dists.items():
            rows.append({
                "workload": wl, "policy": p, "label": POLICY_LABEL[p],
                "total_m": sum(k * v for k, v in d.items()),
                "m_dist_matches_cpu_baseline": dict(sorted(d.items())) == dict(sorted(ref.items())),
                "perf_output_sha1": shas.get(p),
                "sha1_matches_cpu_baseline": shas.get(p) == shas.get("C_cpu"),
            })
    return rows


def main() -> None:
    SUM.mkdir(parents=True, exist_ok=True)
    perf = perf_summary()
    write_csv(SUM / "summary_perf.csv", perf)
    cells, qbym = traffic_summary()
    write_csv(SUM / "summary_traffic.csv", cells)
    write_csv(SUM / "summary_q_by_m.csv", qbym)
    write_csv(SUM / "correctness.csv", correctness_report(cells))
    print(f"wrote {len(perf)} perf rows, {len(cells)} traffic rows, {len(qbym)} q-by-m rows")
    for r in perf:
        print(f"  {r['workload']:9s} {r['policy']:11s} "
              f"{r['decode_tok_s_mean']:.2f}+-{r['decode_tok_s_std']:.2f} tok/s")


if __name__ == "__main__":
    main()
