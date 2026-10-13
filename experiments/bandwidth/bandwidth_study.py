"""Reproducible Qwen/GPT-OSS bandwidth experiments. Run --help for subcommands."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from experiments.paths import python_command


PROFILES = {
    "qwen3.6-35b-nvfp4": {"layers": 40, "experts": 256, "top_k": 8,
                            "hidden": 2048, "inter": 512, "format": "nvfp4"},
    "gpt-oss-20b-mxfp4": {"layers": 24, "experts": 32, "top_k": 4,
                            "hidden": 2880, "inter": 2880, "format": "mxfp4_triton"},
}


def write_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def pcie_link(raw: str | None) -> dict:
    if not raw or "GPU Link Info" not in raw:
        return {}
    section = raw.split("GPU Link Info", 1)[1].split("Bridge Chip", 1)[0]
    generation = re.search(r"PCIe Generation\s+Max\s*:\s*(\d+)\s+Current\s*:\s*(\d+)", section)
    width = re.search(r"Link Width\s+Max\s*:\s*(\d+)x\s+Current\s*:\s*(\d+)x", section)
    if not generation or not width:
        return {}
    max_gen, current_gen = map(int, generation.groups())
    max_width, current_width = map(int, width.groups())
    gb_per_lane = {1: .25, 2: .5, 3: .985, 4: 1.969, 5: 3.938, 6: 7.877}
    return {
        "max_generation": max_gen, "idle_current_generation": current_gen,
        "device_max_width": max_width, "current_width": current_width,
        "approx_unidirectional_GBps_idle": gb_per_lane.get(current_gen, 0) * current_width,
        "approx_unidirectional_GBps_max_gen_current_width":
            gb_per_lane.get(max_gen, 0) * current_width,
    }


def system(args: argparse.Namespace) -> None:
    gpu = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit,pci.bus_id", "--format=csv,noheader")
    pci = command("nvidia-smi", "-q")
    cpu = command("lscpu", "-J")
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(), "os": platform.platform(),
        "python": platform.python_version(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "gpu_query": gpu, "pcie_raw": pci,
        "pcie_link": pcie_link(pci),
        "cpu_lscpu": json.loads(cpu) if cpu else None,
        "cpu_frequency_governor": governor.read_text().strip() if governor.exists() else None,
        "host_ram_meminfo": Path("/proc/meminfo").read_text().splitlines()[:5],
        "host_ram_type": "unavailable without privileged firmware query",
        "cuda_nvcc": command("nvcc", "--version"),
        "cxx": command("c++", "--version"),
        "kernel": platform.release(),
        "model_storage": args.model_storage,
        "checkpoint_revisions": args.checkpoint_revision,
        "configuration": {key: value for key, value in vars(args).items() if key != "func"},
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")
    print(target)


def calibration(args: argparse.Namespace) -> None:
    from freetoken.moe.benchbw import run_benchbw

    rows = []
    for i in range(args.repeat):
        folder = Path(args.out).with_suffix("")
        folder.mkdir(parents=True, exist_ok=True)
        results = {}
        for model, spec in (("qwen3.6-moe", "nvfp4"), ("gpt-oss-20b", "mxfp4_triton")):
            raw = folder / f"run-{i:02d}-{model}.json"
            results[model] = run_benchbw(out_path=str(raw), models=(model,),
                                         dtypes=(), formats=(spec,), device_index=args.gpu)
        profile = dict(results["qwen3.6-moe"])
        profile.pop("out_path", None)
        profile["workloads"] = {
            model: result["workloads"][model] for model, result in results.items()
        }
        combined = folder / f"profile-run-{i:02d}.json"
        combined.write_text(json.dumps(profile, indent=2) + "\n")
        for model, spec in (("qwen3.6-moe", "nvfp4"), ("gpt-oss-20b", "mxfp4_triton")):
            kernel = results[model]["workloads"][model]["kernels"][spec]
            rows.append({
                "run": i, "model": model, "format": spec,
                "B_P_GBps": kernel["pcie_gather_gbs"],
                "B_H_GBps": kernel["cpu_moe_gbs"],
                "B_P_overlap_GBps": kernel["pcie_gather_overlap_gbs"],
                "B_H_overlap_GBps": kernel["cpu_moe_overlap_gbs"],
                "recommended_backend": kernel["recommended"],
                "expert_bytes": kernel["expert_bytes"], "raw_profile": str(combined),
            })
    write_rows(args.out, rows)
    summaries = []
    for model in ("qwen3.6-moe", "gpt-oss-20b"):
        group = [r for r in rows if r["model"] == model]
        for key in ("B_P_GBps", "B_H_GBps", "B_P_overlap_GBps", "B_H_overlap_GBps"):
            values = [float(r[key]) for r in group if r[key] is not None]
            if values:
                summaries.append({"model": model, "capacity_metric": key,
                                  "mean": statistics.mean(values),
                                  "median": statistics.median(values),
                                  "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                                  "min": min(values), "max": max(values), "n": len(values)})
    summary_path = str(Path(args.out).with_name(Path(args.out).stem + "_summary.csv"))
    write_rows(summary_path, summaries)
    print(args.out)


def bank_sizes(name: str) -> list[int]:
    p = PROFILES[name]
    h, i = p["hidden"], p["inter"]
    if p["format"] == "nvfp4":
        return [2 * i * (h // 2), 2 * i * (h // 16), 2 * i * 2,
                h * (i // 2), h * (i // 16), h * 2]
    return [2 * i * (h // 2), 2 * i * (h // 32), 2 * i * 2,
            h * (i // 2), h * (i // 32), h * 2]


def profile_bytes(name: str) -> tuple[int, int]:
    sizes = bank_sizes(name)
    return len(sizes), sum(sizes)


def transfer(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for transfer measurements")
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    rows = []
    sizes = [(name, bank_sizes(name)) for name in PROFILES]
    sizes += [(f"generic-{mib}MiB", [mib << 20]) for mib in args.generic_mib]
    for name, bank_bytes in sizes:
        banks, expert_bytes = len(bank_bytes), sum(bank_bytes)
        for batch in args.batch:
            total = expert_bytes * batch
            if total > args.max_mib << 20:
                continue
            methods = ("contiguous_pinned_H2D", "banked_pinned_H2D") if banks > 1 else ("contiguous_pinned_H2D",)
            for method in methods:
                sizes_for_method = [total] if method.startswith("contiguous") else [n * batch for n in bank_bytes]
                host = [torch.empty(n, dtype=torch.uint8, pin_memory=True) for n in sizes_for_method]
                dev = [torch.empty(n, dtype=torch.uint8, device=device) for n in sizes_for_method]
                for source in host:
                    source.fill_(1)
                for _ in range(args.warmup):
                    for destination, source in zip(dev, host):
                        destination.copy_(source, non_blocking=True)
                torch.cuda.synchronize(device)
                for repeat in range(args.repeat):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for destination, source in zip(dev, host):
                        destination.copy_(source, non_blocking=True)
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                    rows.append({"model": name, "bank_count": banks,
                                 "expert_bytes": expert_bytes, "batch_experts": batch,
                                 "batch_bytes": total, "repeat": repeat,
                                 "transfer_ms": ms,
                                 "effective_GBps": total / (ms * 1e6) if ms else "",
                                 "method": method})
                del host, dev
    write_rows(args.out, rows)
    transfer_summary(
        argparse.Namespace(input=args.out,
                           out=str(Path(args.out).with_name("summary.csv")))
    )
    print(args.out)


def transfer_summary(args: argparse.Namespace) -> None:
    grouped = defaultdict(list)
    for row in read_rows(args.input):
        key = (row["model"], row["bank_count"], row["expert_bytes"],
               row["batch_experts"], row["batch_bytes"], row["method"])
        grouped[key].append(row)
    summary = []
    for (model, bank_count, expert_bytes, batch, batch_bytes, method), group in grouped.items():
        record = {"model": model, "bank_count": bank_count,
                  "expert_bytes": expert_bytes, "batch_experts": batch,
                  "batch_bytes": batch_bytes, "method": method, "n": len(group)}
        for key in ("transfer_ms", "effective_GBps"):
            values = [float(row[key]) for row in group if row[key] != ""]
            if values:
                record[f"{key}_mean"] = statistics.mean(values)
                record[f"{key}_median"] = statistics.median(values)
                record[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
                record[f"{key}_min"] = min(values)
                record[f"{key}_max"] = max(values)
        summary.append(record)
    write_rows(args.out, summary)
    print(args.out)


def link_probe(args: argparse.Namespace) -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for a loaded PCIe link probe")
    device = torch.device("cuda", args.gpu)
    torch.cuda.set_device(device)
    nbytes = args.mib << 20
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
    host.fill_(1)
    stop = threading.Event()
    copies = [0]

    def copy_loop() -> None:
        torch.cuda.set_device(device)
        while not stop.is_set():
            dst.copy_(host, non_blocking=True)
            torch.cuda.synchronize(device)
            copies[0] += 1

    worker = threading.Thread(target=copy_loop)
    worker.start()
    samples = []
    try:
        for _ in range(args.samples):
            raw = command("nvidia-smi", "-q")
            samples.append({"timestamp": datetime.now(timezone.utc).isoformat(),
                            "link": pcie_link(raw)})
            time.sleep(.2)
    finally:
        stop.set()
        worker.join()
    result = {"gpu": args.gpu, "buffer_bytes": nbytes, "copies": copies[0],
              "samples": samples}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


def replay_one(routing: list[list[list[int]]], experts: int, top_k: int,
               slots: int, expert_bytes: int, q_fraction: float) -> dict:
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()
    hits = misses = fills = evaluations = unique_total = 0
    miss_counts: list[int] = []
    for token in routing:
        for layer, selected in enumerate(token):
            active = list(dict.fromkeys(selected[:top_k]))
            protected = {(layer, expert % experts) for expert in active}
            evaluations += top_k
            unique_total += len(active)
            absent = []
            for expert in active:
                key = (layer, expert % experts)
                if key in cache:
                    hits += 1
                    cache.move_to_end(key)
                elif key not in absent:
                    absent.append(key)
            m = len(absent)
            misses += m
            miss_counts.append(m)
            frac = round(q_fraction * (1 << 16))
            lo = (m * frac) >> 16
            cost = lambda q: max(q * ((1 << 16) - frac), (m - q) * frac)
            q = min(m, lo if cost(lo) <= cost(lo + 1) else lo + 1)
            for key in absent[:q]:
                if len(cache) == slots:
                    victim = next((old for old in cache if old not in protected), None)
                    if victim is None:
                        raise ValueError("cache slots must hold all active experts")
                    del cache[victim]
                cache[key] = None
            fills += q
    return {"E": experts, "k": top_k, "cache_slots": slots,
            "cache_bytes": slots * expert_bytes,
            "relative_cache_capacity": slots / (len(routing[0]) * experts),
            "expert_bytes": expert_bytes, "tokens": len(routing),
            "expert_evaluations_per_token": evaluations / len(routing),
            "unique_per_token": unique_total / len(routing),
            "hits_per_token": hits / len(routing), "misses_per_token": misses / len(routing),
            "miss_rate": misses / unique_total if unique_total else 0,
            "mean_m": statistics.mean(miss_counts),
            "median_m": statistics.median(miss_counts),
            "p95_m": sorted(miss_counts)[math.ceil(.95 * len(miss_counts)) - 1],
            "gpu_fills_per_token": fills / len(routing),
            "cpu_executions_per_token": (misses - fills) / len(routing),
            "bytes_h2d_per_token": fills * expert_bytes / len(routing)}


def replay(args: argparse.Namespace) -> None:
    if not 0 <= args.q_fraction <= 1:
        raise SystemExit("--q-fraction must be in [0, 1]")
    if min(args.experts) < max(args.k):
        raise SystemExit("every --experts value must be >= every --k value")
    if not all(0 < x <= 1 for x in args.cache_fraction):
        raise SystemExit("--cache-fraction values must be in (0, 1]")
    if any(x < max(args.k) for x in args.cache_slots):
        raise SystemExit("--cache-slots must hold at least the maximum top-k")
    if args.tokens < 1:
        raise SystemExit("--tokens must be positive")
    rng = random.Random(args.seed)
    rows = []
    trace_routing = None
    if args.trace:
        trace_rows = read_rows(args.trace)
        if args.benchmark:
            selected = json.loads(Path(args.benchmark).read_text().splitlines()[args.benchmark_row])
            trace_rows = trace_rows[selected["trace_start_row"]:selected["trace_end_row"]]
        if args.run_id:
            trace_rows = [r for r in trace_rows if r["run_id"] == args.run_id]
        if not trace_rows:
            raise SystemExit("no routing rows selected")
        if len({r["run_id"] for r in trace_rows}) != 1:
            raise SystemExit("select one --run-id or one --benchmark row")
        by_token = defaultdict(dict)
        for row in trace_rows:
            by_token[int(row["token_index"])][int(row["layer_id"])] = [
                int(x) for x in row["routed_experts"].split(";")
            ]
        trace_routing = [list(dict(sorted(by_token[t].items())).values()) for t in sorted(by_token)]
        layer_counts = {len(token) for token in trace_routing}
        if len(layer_counts) != 1:
            raise SystemExit("routing trace has incomplete tokens")
    profiles = {args.profile: PROFILES[args.profile]} if args.trace else PROFILES
    for name, p in profiles.items():
        if trace_routing and len(trace_routing[0]) != p["layers"]:
            raise SystemExit(f"trace has {len(trace_routing[0])} layers; {name} expects {p['layers']}")
        _, eb = profile_bytes(name)
        max_k = max(args.k)
        for experts in args.experts:
            # A fixed routing stream is reused across every k and capacity cell.
            routing = trace_routing or [
                [rng.sample(range(experts), max_k) for _ in range(p["layers"])]
                for _ in range(args.tokens)
            ]
            for k in args.k:
                if k > experts or k > min(len(ids) for token in routing for ids in token):
                    continue
                capacities = [("relative", max(1, round(p["layers"] * experts * fraction)))
                              for fraction in args.cache_fraction]
                capacities += [("raw_slots", slots) for slots in args.cache_slots]
                for capacity_mode, slots in capacities:
                    result = replay_one(routing, experts, k, slots, eb, args.q_fraction)
                    rows.append({"profile": name, "seed": args.seed if not args.trace else "",
                                 "routing": "measured_trace_modulo_E" if args.trace else "uniform_without_replacement",
                                 "q_fraction": args.q_fraction,
                                 "capacity_mode": capacity_mode, **result})
    write_rows(args.out, rows)
    print(args.out)


def trace_summary(args: argparse.Namespace) -> None:
    rows = read_rows(args.trace)
    selected = None
    if args.benchmark:
        selected = json.loads(Path(args.benchmark).read_text().splitlines()[args.benchmark_row])
        rows = rows[selected["trace_start_row"]:selected["trace_end_row"]]
    if not rows:
        raise SystemExit("no trace rows in selected range")
    by_model = defaultdict(list)
    for row in rows:
        by_model[(row["run_id"], row["model_id"])].append(row)
    out = []
    for (run, model), group in by_model.items():
        layers = {int(r["layer_id"]) for r in group}
        tokens = len(group) / len(layers)
        counts = {layer: sum(int(r["layer_id"]) == layer for r in group) for layer in layers}
        if not tokens.is_integer() or len(set(counts.values())) != 1:
            raise SystemExit(f"incomplete token trace for {run}")
        total = lambda key: sum(int(r[key]) for r in group)
        unique, misses, fills = total("unique_routed_count"), total("miss_count"), total("gpu_fill_count")
        m = sorted(int(r["m"]) for r in group)
        out.append({"run_id": run, "model_id": model, "tokens": int(tokens),
                    "moe_layers": len(layers), "expert_evaluations_per_token":
                    sum(len(r["routed_experts"].split(";")) for r in group) / tokens,
                    "experts_per_layer": int(group[0]["E"]),
                    "top_k": int(group[0]["k"]),
                    "cache_slots": int(group[0]["cache_slots"]),
                    "cache_bytes": int(group[0]["cache_bytes"]),
                    "expert_bytes": int(group[0]["expert_size_bytes"]),
                    "unique_per_token": unique / tokens,
                    "cache_hits_per_token": total("hit_count") / tokens,
                    "cache_misses_per_token": misses / tokens,
                    "mean_m": statistics.mean(m), "median_m": statistics.median(m),
                    "p95_m": m[math.ceil(.95 * len(m)) - 1],
                    "gpu_fills_per_token": fills / tokens,
                    "cpu_executions_per_token": (misses - fills) / tokens,
                    "nominal_cpu_expert_bytes_per_token": (
                        (misses - fills) * int(group[0]["expert_size_bytes"]) / tokens
                    ),
                    "bytes_h2d_per_token": total("bytes_h2d") / tokens,
                    "miss_rate": misses / unique if unique else 0,
                    "gpu_fill_fraction": fills / misses if misses else 0,
                    "cpu_execution_fraction": (misses - fills) / misses if misses else 0,
                    "traced_decode_tok_s": selected.get("decode_tok_s", "") if selected else "",
                    "effective_h2d_GBps_trace_decode": (
                        total("bytes_h2d") / tokens * selected["decode_tok_s"] / 1e9
                        if selected and "decode_tok_s" in selected else ""
                    )})
    write_rows(args.out, out)
    print(args.out)


def popularity(args: argparse.Namespace) -> None:
    rows = read_rows(args.trace)
    if args.benchmark:
        selected = json.loads(Path(args.benchmark).read_text().splitlines()[args.benchmark_row])
        rows = rows[selected["trace_start_row"]:selected["trace_end_row"]]
    if not rows:
        raise SystemExit("no trace rows in selected range")
    if len({row["run_id"] for row in rows}) != 1:
        raise SystemExit("select one measured run with --benchmark")
    first_token = min(int(row["token_index"]) for row in rows)
    last_token = max(int(row["token_index"]) for row in rows)
    token_span = last_token - first_token + 1
    layers = {int(row["layer_id"]) for row in rows}
    experts = int(rows[0]["E"])
    counts = defaultdict(lambda: defaultdict(int))
    for row in rows:
        layer = int(row["layer_id"])
        window = min(3, 4 * (int(row["token_index"]) - first_token) // token_span)
        for expert in (int(value) for value in row["routed_experts"].split(";")):
            counts[("global", "all")][(layer, expert)] += 1
            counts[("layer", layer)][expert] += 1
            counts[("window", window)][(layer, expert)] += 1
    out = []
    for (scope, scope_id), frequency in sorted(counts.items()):
        universe = experts if scope == "layer" else experts * len(layers)
        accesses = sum(frequency.values())
        ranked = sorted(frequency.values(), reverse=True)
        for fraction in (.1, .25, .5, .75, 1.0):
            kept = min(universe, max(1, math.ceil(universe * fraction)))
            covered = sum(ranked[:kept]) / accesses
            out.append({"run_id": rows[0]["run_id"], "model_id": rows[0]["model_id"],
                        "scope": scope, "scope_id": scope_id,
                        "retained_fraction": fraction, "retained_experts": kept,
                        "expert_universe": universe, "routed_accesses": accesses,
                        "access_coverage": covered,
                        "hypothetical_removed_access_fraction": 1 - covered})
    write_rows(args.out, out)
    print(args.out)


def baseline(args: argparse.Namespace) -> None:
    bench = Path(__file__).resolve().parents[2] / "experiments/baselines" / "bench_decode_moe.py"
    target = Path(args.out)
    if target.exists():
        raise SystemExit(f"{target} already exists; choose a new --out for an isolated baseline")
    target.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.profile:
        env["FREETOKEN_BENCHBW_PATH"] = str(Path(args.profile).resolve())
    for cache_rate in args.cache_rate:
        for overlap in (True, False):
            for backend in ("auto", "offload", "hybrid"):
                for repeat in range(args.repeat):
                    cmd = [*python_command(), str(bench), "--model", args.model,
                           "--backend", backend, "--cache-rate", str(cache_rate),
                           "--decode", str(args.decode), "--json", str(target)]
                    if not overlap:
                        cmd.append("--no-overlap")
                    if args.aime:
                        cmd += ["--aime", args.aime]
                    if args.gpu:
                        cmd += ["--gpu", args.gpu]
                    if args.greedy:
                        cmd.append("--greedy")
                    if args.no_graph:
                        cmd.append("--no-graph")
                    if args.mem_ratio != 0.9:
                        cmd += ["--mem-ratio", str(args.mem_ratio)]
                    if args.server_timeout != 120:
                        cmd += ["--server-timeout", str(args.server_timeout)]
                    print(f"cache={cache_rate} overlap={overlap} backend={backend} repeat={repeat}", flush=True)
                    result = subprocess.run(cmd, check=False, env=env)
                    if result.returncode:
                        # Keep the matrix auditable when one cache/backend cell cannot
                        # reserve the minimum KV pool on a small GPU.
                        failure = {
                            "model": args.model,
                            "backend": backend,
                            "hybrid_fetch": -1,
                            "cache_slots_requested": 0,
                            "cache_rate_requested": cache_rate,
                            "prefill_overlap": overlap,
                            "status": "error",
                            "error": f"bench_decode_moe exited with status {result.returncode}",
                        }
                        with target.open("a") as f:
                            f.write(json.dumps(failure) + "\n")
                        print(f"[baseline] recorded failed cell: {failure['error']}", flush=True)
    print(target)


def baseline_summary(args: argparse.Namespace) -> None:
    grouped = defaultdict(list)
    failures = []
    for line in Path(args.input).read_text().splitlines():
        row = json.loads(line)
        if row.get("status", "ok") != "ok":
            failures.append(row)
            continue
        key = (row["model"], row["backend"], row["cache_rate_requested"],
               row["prefill_overlap"])
        grouped[key].append(row)
    summary = []
    metrics = ("decode_tok_s", "ms_per_token", "ttft_ms", "event_ms_p50",
               "event_ms_p95", "event_ms_p99")
    for (model, backend, cache_rate, overlap), group in grouped.items():
        row = {"model": model, "backend": backend, "cache_rate": cache_rate,
               "prefill_overlap": overlap, "n": len(group), "status": "ok", "error": ""}
        hashes = {item["output_sha1"] for item in group if item.get("output_sha1")}
        row["distinct_output_hashes"] = len(hashes)
        row["output_sha1"] = next(iter(hashes)) if len(hashes) == 1 else ""
        for metric in metrics:
            values = [float(item[metric]) for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        summary.append(row)
    for failure in failures:
        row = {"model": failure["model"], "backend": failure["backend"],
               "cache_rate": failure["cache_rate_requested"],
               "prefill_overlap": failure["prefill_overlap"], "n": 0,
               "status": "error", "error": failure.get("error", "unknown error"),
               "distinct_output_hashes": "", "output_sha1": ""}
        for metric in metrics:
            for suffix in ("mean", "median", "stdev", "min", "max"):
                row[f"{metric}_{suffix}"] = ""
        summary.append(row)
    write_rows(args.out, summary)
    print(f"{args.out} ({len(grouped)} successful cells, {len(failures)} failed cells)")


def prefill_summary(args: argparse.Namespace) -> None:
    grouped = defaultdict(list)
    for line in Path(args.input).read_text().splitlines():
        row = json.loads(line)
        key = (row["model"], row["backend"], row["cache_rate_requested"],
               row.get("memory_ratio_requested", 0.9),
               row["prefill_overlap"], row["target_prompt_tokens"])
        grouped[key].append(row)
    out = []
    for (model, backend, cache_rate, mem_ratio, overlap, target), group in sorted(grouped.items()):
        times = [float(row["ttft_ms"]) for row in group]
        lengths = [int(row["prompt_tokens"]) for row in group]
        out.append({"model": model, "backend": backend, "cache_rate": cache_rate,
                    "memory_ratio": mem_ratio,
                    "prefill_overlap": overlap, "target_prompt_tokens": target,
                    "actual_prompt_tokens_median": statistics.median(lengths),
                    "ttft_ms_mean": statistics.mean(times),
                    "ttft_ms_median": statistics.median(times),
                    "ttft_ms_stdev": statistics.stdev(times) if len(times) > 1 else 0,
                    "ttft_ms_min": min(times), "ttft_ms_max": max(times),
                    "n": len(times), "incremental_effective_prefill_tps": ""})
    by_setting = defaultdict(list)
    for row in out:
        by_setting[(row["model"], row["backend"], row["cache_rate"], row["memory_ratio"],
                    row["prefill_overlap"])].append(row)
    for group in by_setting.values():
        group.sort(key=lambda row: row["actual_prompt_tokens_median"])
        for previous, current in zip(group, group[1:]):
            delta_ms = current["ttft_ms_median"] - previous["ttft_ms_median"]
            if delta_ms > 0:
                current["incremental_effective_prefill_tps"] = (
                    1000 * (current["actual_prompt_tokens_median"]
                            - previous["actual_prompt_tokens_median"]) / delta_ms
                )
    write_rows(args.out, out)
    print(args.out)


def q_sweep(args: argparse.Namespace) -> None:
    if args.max_q < 1 or args.repeat < 1:
        raise SystemExit("--max-q and --repeat must be positive")
    if args.cache and args.cache_rate is not None:
        raise SystemExit("select only one of --cache and --cache-rate")
    if args.layer_timing and not args.trace_csv:
        raise SystemExit("--layer-timing requires --trace-csv")
    if args.warmup_q is not None and (args.warmup_q < 0 or not args.trace_csv or not args.forced_only):
        raise SystemExit("--warmup-q requires a nonnegative q, --trace-csv, and --forced-only")
    bench = Path(__file__).resolve().parents[2] / "experiments/baselines" / "bench_decode_moe.py"
    target = Path(args.out)
    if target.exists():
        raise SystemExit(f"{target} already exists; choose a new --out for an isolated sweep")
    target.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.profile:
        env["FREETOKEN_BENCHBW_PATH"] = str(Path(args.profile).resolve())
    if args.layer_timing:
        env["FREETOKEN_MOE_LAYER_TIMING"] = "1"
    values = list(range(args.max_q + 1)) if args.forced_only else [-1, *range(args.max_q + 1)]
    for repeat in range(args.repeat):
        for q in values if repeat % 2 == 0 else list(reversed(values)):
            cmd = [*python_command(), str(bench), "--model", args.model,
                   "--backend", "hybrid", "--hybrid-fetch", str(q),
                   "--decode", str(args.decode), "--json", str(target)]
            if args.trace_csv:
                cmd += ["--no-graph", "--trace-csv", args.trace_csv]
            if args.warmup_q is not None:
                cmd += ["--warmup-hybrid-fetch", str(args.warmup_q)]
            if args.aime:
                cmd += ["--aime", args.aime]
            if args.cache:
                cmd += ["--cache", str(args.cache)]
            if args.cache_rate is not None:
                cmd += ["--cache-rate", str(args.cache_rate)]
            if args.gpu:
                cmd += ["--gpu", args.gpu]
            if args.greedy:
                cmd.append("--greedy")
            print(f"q={q} repeat={repeat}: {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, check=True, env=env)
    print(target)


def q_summary(args: argparse.Namespace) -> None:
    by_q: dict[int, list[float]] = defaultdict(list)
    for line in Path(args.input).read_text().splitlines():
        row = json.loads(line)
        if row["backend"] == "hybrid" and row["model"] == args.model:
            by_q[int(row["hybrid_fetch"])].append(float(row["ms_per_token"]))
    if -1 not in by_q or len(by_q) < 3:
        raise SystemExit("need auto (-1) and at least two forced q values for this model")
    medians = {q: statistics.median(samples) for q, samples in by_q.items()}
    forced = {q: latency for q, latency in medians.items() if q >= 0}
    best_q = min(forced, key=forced.get)
    best = forced[best_q]
    rows = [{"model": args.model, "q_cap": q, "repetitions": len(by_q[q]),
             "median_ms_per_token": latency, "best_forced_q_cap": best_q,
             "penalty_vs_best_forced": latency / best - 1,
             "scope": "whole_decode_cap_sweep"}
            for q, latency in sorted(medians.items())]
    write_rows(args.out, rows)
    print(args.out)


def predicted_q(m: int, fraction: float) -> int:
    frac = min(1 << 16, max(0, round(fraction * (1 << 16))))
    lo = (m * frac) >> 16
    cost = lambda q: max(q * ((1 << 16) - frac), (m - q) * frac)
    return min(m, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def q_layer_summary(args: argparse.Namespace) -> None:
    if not 0 <= args.fraction <= 1:
        raise SystemExit("--fraction must be in [0, 1]")
    if getattr(args, "match_initial_layers", False) and not args.first_token_only:
        raise SystemExit("--match-initial-layers requires --first-token-only")
    traces = read_rows(args.trace)
    benchmarks = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines()]
    by_m_q = defaultdict(list)
    auto_by_m = defaultdict(list)
    output_hashes = {row["output_sha1"] for row in benchmarks if row.get("output_sha1")}
    routing_patterns = set()
    run_rows = []
    layer_signatures = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for benchmark in benchmarks:
        selected = traces[benchmark["trace_start_row"]:benchmark["trace_end_row"]]
        if not selected or {row["run_id"] for row in selected} != {benchmark["trace_run_id"]}:
            raise SystemExit("trace row range does not match benchmark run id")
        if getattr(args, "first_token_only", False):
            first = {layer: min(int(row["token_index"]) for row in selected
                                if row["layer_id"] == layer)
                     for layer in {row["layer_id"] for row in selected}}
            selected = [row for row in selected
                        if int(row["token_index"]) == first[row["layer_id"]]]
            routing_patterns.add(tuple((row["layer_id"], row["routed_experts"], row["m"])
                                       for row in sorted(selected, key=lambda item: int(item["layer_id"]))))
            for row in selected:
                layer = int(row["layer_id"])
                signature = (row["routed_experts"], row["m"])
                layer_signatures[layer][signature][benchmark["hybrid_fetch"]] += 1
        run_rows.append((benchmark, selected))
    matched_layers = {}
    for layer, signatures in layer_signatures.items():
        for signature, counts in signatures.items():
            m = int(signature[1])
            if not set(range(m + 1)).issubset(counts):
                continue
            key = (layer, m)
            candidate = (sum(counts[q] for q in range(m + 1)), signature)
            if key not in matched_layers or candidate > matched_layers[key]:
                matched_layers[key] = candidate
    for benchmark, selected in run_rows:
        for row in selected:
            if (getattr(args, "match_initial_layers", False) and
                    matched_layers.get((int(row["layer_id"]), int(row["m"])), (0, None))[1] !=
                    (row["routed_experts"], row["m"])):
                continue
            if not row["layer_time_ms"]:
                raise SystemExit("layer timing is missing; rerun q-sweep with --layer-timing")
            m = int(row["m"])
            if not m:
                continue
            if benchmark["hybrid_fetch"] > m:
                continue
            elapsed = float(row["layer_time_ms"])
            if benchmark["hybrid_fetch"] < 0:
                auto_by_m[m].append(elapsed)
            else:
                by_m_q[(m, int(row["gpu_fill_count"]))].append(elapsed)
    out = []
    for m in sorted({m for m, _ in by_m_q}):
        samples = {q: by_m_q[(m, q)] for q in range(m + 1) if (m, q) in by_m_q}
        complete = len(samples) == m + 1
        min_samples = min(map(len, samples.values())) if complete else 0
        medians = {q: statistics.median(values) for q, values in samples.items()}
        oracle = min(medians, key=medians.get) if complete else None
        predicted = predicted_q(m, args.fraction)
        for q, values in samples.items():
            out.append({
                "model": args.model, "m": m, "q": q, "n": len(values),
                "median_layer_ms": medians[q],
                "mean_layer_ms": statistics.mean(values),
                "stdev_layer_ms": statistics.stdev(values) if len(values) > 1 else 0,
                "complete_q_sweep": complete, "q_predicted": predicted,
                "min_samples_per_q": min_samples,
                "q_oracle": oracle if complete else "",
                "q_error": predicted - oracle if complete else "",
                "latency_penalty_predicted": (
                    medians[predicted] / medians[oracle] - 1
                    if complete and predicted in medians else ""
                ),
                "auto_n": len(auto_by_m[m]),
                "distinct_output_hashes": len(output_hashes),
                "distinct_initial_routing_and_m_patterns": len(routing_patterns),
                "matched_initial_layers": sum(key[1] == m for key in matched_layers),
                "initial_layers_total": len(layer_signatures),
                "auto_median_layer_ms": (
                    statistics.median(auto_by_m[m]) if auto_by_m[m] else ""
                ),
                "scope": ("matched_warmup_stable_layers_first_token"
                          if getattr(args, "match_initial_layers", False)
                          else "warmup_first_token" if getattr(args, "first_token_only", False)
                          else "eager_traced_layer_actual_m"),
            })
    write_rows(args.out, out)
    print(args.out)


def comparison(args: argparse.Namespace) -> None:
    qwen = read_rows(args.qwen_trace)[0]
    gpt = read_rows(args.gpt_trace)[0]
    calibration = read_rows(args.calibration)

    def capacity(model: str, metric: str) -> str:
        return next(row["median"] for row in calibration
                    if row["model"] == model and row["capacity_metric"] == metric)

    def throughput(path: str) -> str:
        matches = [row for row in read_rows(path)
                   if row["backend"] == "hybrid" and float(row["cache_rate"]) == .2
                   and row["prefill_overlap"] == "True" and row["status"] == "ok"]
        if len(matches) != 1:
            raise SystemExit(f"need one hybrid 20% overlap-on baseline in {path}")
        return matches[0]["decode_tok_s_median"]

    fields = [
        ("MoE layers", "moe_layers"),
        ("Experts per layer", "experts_per_layer"),
        ("Top-k", "top_k"),
        ("Expert bytes", "expert_bytes"),
        ("Expert evaluations per token", "expert_evaluations_per_token"),
        ("Unique experts per token", "unique_per_token"),
        ("Cache misses per token", "cache_misses_per_token"),
        ("Mean misses per layer", "mean_m"),
        ("Cache miss rate", "miss_rate"),
        ("GPU fills per token", "gpu_fills_per_token"),
        ("CPU executions per token", "cpu_executions_per_token"),
        ("Nominal CPU expert bytes per token", "nominal_cpu_expert_bytes_per_token"),
        ("H2D bytes per token", "bytes_h2d_per_token"),
    ]
    rows = [{"metric": label, "qwen": qwen[key], "gpt_oss": gpt[key]}
            for label, key in fields]
    rows += [
        {"metric": "B_P GB/s", "qwen": capacity("qwen3.6-moe", "B_P_GBps"),
         "gpt_oss": capacity("gpt-oss-20b", "B_P_GBps")},
        {"metric": "B_H GB/s", "qwen": capacity("qwen3.6-moe", "B_H_GBps"),
         "gpt_oss": capacity("gpt-oss-20b", "B_H_GBps")},
        {"metric": "Hybrid decode tokens/s median", "qwen": throughput(args.qwen_baseline),
         "gpt_oss": throughput(args.gpt_baseline)},
    ]
    write_rows(args.out, rows)
    print(args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    p = subs.add_parser("system")
    p.add_argument("--out", default="results/baseline/system_info.json")
    p.add_argument("--model-storage", default="unknown")
    p.add_argument("--checkpoint-revision", action="append", default=[])
    p.set_defaults(func=system)
    p = subs.add_parser("calibration")
    p.add_argument("--out", default="results/baseline/calibration.csv")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--repeat", type=int, default=5)
    p.set_defaults(func=calibration)
    p = subs.add_parser("transfer")
    p.add_argument("--out", default="results/transfer_microbench/banked/contiguous_vs_banked.csv")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--generic-mib", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--max-mib", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.set_defaults(func=transfer)
    p = subs.add_parser("transfer-summary")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/transfer_microbench/summary.csv")
    p.set_defaults(func=transfer_summary)
    p = subs.add_parser("link-probe")
    p.add_argument("--out", default="results/baseline/link_state.json")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--mib", type=int, default=64)
    p.add_argument("--samples", type=int, default=10)
    p.set_defaults(func=link_probe)
    p = subs.add_parser("replay")
    p.add_argument("--out", default="results/geometry_replay/summary.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--experts", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--cache-fraction", type=float, nargs="+", default=[.1, .2, .3, .4, .5])
    p.add_argument("--cache-slots", type=int, nargs="+", default=[])
    p.add_argument("--q-fraction", type=float, default=.4)
    p.add_argument("--trace", help="decode trace CSV to replay observed routing")
    p.add_argument("--run-id", help="select one run in the trace")
    p.add_argument("--benchmark", help="bench_decode_moe.py JSONL; select measured request")
    p.add_argument("--benchmark-row", type=int, default=0)
    p.add_argument("--profile", choices=tuple(PROFILES), default="qwen3.6-35b-nvfp4")
    p.set_defaults(func=replay)
    p = subs.add_parser("trace-summary")
    p.add_argument("--trace", required=True)
    p.add_argument("--benchmark", help="JSONL output of bench_decode_moe.py")
    p.add_argument("--benchmark-row", type=int, default=0)
    p.add_argument("--out", default="results/traffic_trace/summary.csv")
    p.set_defaults(func=trace_summary)
    p = subs.add_parser("popularity")
    p.add_argument("--trace", required=True)
    p.add_argument("--benchmark", help="JSONL output selecting the measured request")
    p.add_argument("--benchmark-row", type=int, default=0)
    p.add_argument("--out", default="results/traffic_trace/popularity.csv")
    p.set_defaults(func=popularity)
    p = subs.add_parser("q-sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--aime", help="local AIME JSONL; otherwise the benchmark downloads it")
    p.add_argument("--out", default="results/qstar_oracle/decode.jsonl")
    p.add_argument("--max-q", type=int, required=True)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--decode", type=int, default=256)
    p.add_argument("--cache", type=int, default=0)
    p.add_argument("--cache-rate", type=float)
    p.add_argument("--gpu")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--trace-csv", help="append eager per-layer traffic for each q run")
    p.add_argument("--warmup-q", type=int,
                   help="fill the same cache during warmup, then switch to each forced q")
    p.add_argument("--forced-only", action="store_true", help="omit the auto q=-1 run")
    p.add_argument("--layer-timing", action="store_true",
                   help="synchronize each traced hybrid layer and record its latency")
    p.add_argument("--profile", help="benchbw JSON profile for auto backend / q")
    p.set_defaults(func=q_sweep)
    p = subs.add_parser("baseline")
    p.add_argument("--model", required=True)
    p.add_argument("--aime")
    p.add_argument("--out", default="results/baseline/decode.jsonl")
    p.add_argument("--cache-rate", type=float, nargs="+", default=[.2, .4])
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--decode", type=int, default=256)
    p.add_argument("--gpu")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--mem-ratio", type=float, default=0.9)
    p.add_argument("--server-timeout", type=float, default=120)
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--profile", help="benchbw JSON profile for auto backend / q")
    p.set_defaults(func=baseline)
    p = subs.add_parser("baseline-summary")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/baseline/decode_summary.csv")
    p.set_defaults(func=baseline_summary)
    p = subs.add_parser("prefill-summary")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="results/baseline/prefill_summary.csv")
    p.set_defaults(func=prefill_summary)
    p = subs.add_parser("q-summary")
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="results/qstar_oracle/summary.csv")
    p.set_defaults(func=q_summary)
    p = subs.add_parser("q-layer-summary")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--trace", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--fraction", type=float, required=True)
    p.add_argument("--first-token-only", action="store_true",
                   help="compare only the first measured token after a matched warmup")
    p.add_argument("--match-initial-layers", action="store_true",
                   help="keep only first-token layers with identical routing and miss count in every run")
    p.add_argument("--out", default="results/qstar_oracle/layer_summary.csv")
    p.set_defaults(func=q_layer_summary)
    p = subs.add_parser("comparison")
    p.add_argument("--qwen-trace", required=True)
    p.add_argument("--gpt-trace", required=True)
    p.add_argument("--qwen-baseline", required=True)
    p.add_argument("--gpt-baseline", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--out", default="results/baseline/model_comparison.csv")
    p.set_defaults(func=comparison)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
