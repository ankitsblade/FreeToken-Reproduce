"""Orchestrate the q* CPU/GPU-split policy comparison on gpt-oss-20b.

Everything is held constant across policies (model, quantization, cache size, KV cache,
prompt, decode length, seed, backend=hybrid); only the q-selection policy changes. Each
cell spawns a fresh `ft serve` (via bench_decode_moe.py) so every run starts from a cold
expert cache warmed by one standardized warmup generate -- no policy sees another's warm
state. Resumable: a cell whose output JSON already has the requested repeats is skipped.

Policies (all backend=hybrid; q = misses fetched to GPU, m-q computed on CPU):
  A_adaptive  FreeToken q*      --hybrid-fetch -1  (calibrated pcie/cpu fraction)
  B_gpu       q = m             --hybrid-fetch -1  FORCE_FETCH_FRACTION=1.0
  C_cpu       q = 0             --hybrid-fetch 0
  D_fixed25   q = round(.25 m)  --hybrid-fetch -1  FORCE_FETCH_FRACTION=0.25
  E_fixed50   q = round(.50 m)  --hybrid-fetch -1  FORCE_FETCH_FRACTION=0.50
  F_fixed75   q = round(.75 m)  --hybrid-fetch -1  FORCE_FETCH_FRACTION=0.75

The fixed-fraction policies reuse FreeToken's native hybrid_fetch_fraction path (same
branch-balancing round the adaptive q* uses), so they differ from the adaptive policy in
exactly one value: the fraction. Rounding is round-to-branch-balance, which differs from a
strict floor by at most one expert; this is the fair comparison since q* uses that round.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from experiments.paths import python_command

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
BENCH = ROOT / "experiments/baselines" / "bench_decode_moe.py"
MONITOR = ROOT / "experiments/common/monitor_system.py"
OUT = study_dir("split_policy") / "results"
LOGS = study_dir("split_policy") / "logs"

# (hybrid_fetch flag, forced fetch fraction or None)
POLICIES = {
    "A_adaptive": (-1, None),
    "B_gpu": (-1, 1.0),
    "C_cpu": (0, None),
    "D_fixed25": (-1, 0.25),
    "E_fixed50": (-1, 0.50),
    "F_fixed75": (-1, 0.75),
}

# workload label -> cache slots as a fraction of L*E (the miss-pressure knob). Smaller
# cache -> more cache misses m per layer. Everything else is identical across workloads.
# high_miss uses the smallest cache the overlap double-buffer allows (cache_size >= 2*E =
# 64 slots; 0.10*L*E = 76). Smaller rates (e.g. 0.08 -> 61) trip that engine assertion.
WORKLOADS = {"low_miss": 0.40, "med_miss": 0.20, "high_miss": 0.10}


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def run_bench(model: str, policy: str, cache_rate: float, decode: int, out_json: Path,
              trace_csv: Path | None, seed_greedy: bool, log: Path,
              force_fraction: float | None, hybrid_fetch: int, gpu: str) -> None:
    env = os.environ.copy()
    if force_fraction is not None:
        env["FREETOKEN_MOE_FORCE_FETCH_FRACTION"] = str(force_fraction)
    else:
        env.pop("FREETOKEN_MOE_FORCE_FETCH_FRACTION", None)
    cmd = [
        *python_command(), str(BENCH), "--model", model, "--backend", "hybrid",
        "--hybrid-fetch", str(hybrid_fetch), "--decode", str(decode),
        "--cache-rate", str(cache_rate), "--gpu", gpu,
        "--aime", str(ROOT / "experiments" / "prompts" / "routing_workloads.jsonl"),
        "--problem", "0", "--json", str(out_json),
    ]
    if seed_greedy:
        cmd.append("--greedy")
    if trace_csv is not None:
        cmd += ["--no-graph", "--trace-csv", str(trace_csv)]
    monitor_csv = out_json.with_suffix(".monitor.csv")
    with log.open("a") as lf:
        lf.write(f"\n\n==== {time.strftime('%H:%M:%S')} {policy} rate={cache_rate} "
                 f"decode={decode} trace={trace_csv is not None} force={force_fraction} ====\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        mon = None
        if trace_csv is not None and not monitor_csv.exists():
            mon = subprocess.Popen(
                [*python_command(), str(MONITOR), "--watch-pid", str(proc.pid),
                 "--interval", "0.5", "--out", str(monitor_csv)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rc = proc.wait()
        if mon is not None:
            mon.wait()
        if rc != 0:
            raise SystemExit(f"bench failed for {policy} (rate {cache_rate}); see {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL"), required=False)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--perf-decode", type=int, default=128)
    ap.add_argument("--perf-repeats", type=int, default=3)
    ap.add_argument("--trace-decode", type=int, default=64)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--greedy", action="store_true", default=True,
                    help="greedy decode so policies share a token trajectory (default on)")
    ap.add_argument("--sampled", dest="greedy", action="store_false")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model or set $MODEL (source scripts/env.sh)")

    LOGS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model, "gpu": args.gpu, "greedy": args.greedy,
        "perf_decode": args.perf_decode, "perf_repeats": args.perf_repeats,
        "trace_decode": args.trace_decode,
        "workloads": {w: WORKLOADS[w] for w in args.workloads},
        "policies": {p: {"hybrid_fetch": POLICIES[p][0], "force_fraction": POLICIES[p][1]}
                     for p in args.policies},
        "backend": "hybrid", "problem": 0,
        "prompt_source": "experiments/prompts/routing_workloads.jsonl",
    }
    (OUT / "manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total = len(args.workloads) * len(args.policies)
    done = 0
    for wl in args.workloads:
        rate = WORKLOADS[wl]
        for pol in args.policies:
            done += 1
            hybrid_fetch, force = POLICIES[pol]
            tag = f"{wl}/{pol}"
            print(f"[{done}/{total}] {tag} (cache_rate={rate})", flush=True)
            log = LOGS / f"{wl}_{pol}.log"

            # Perf pass: CUDA graph on, no trace -> clean tok/s + latency percentiles.
            perf_json = OUT / "e2e" / "perf" / wl / f"{pol}.json"
            perf_json.parent.mkdir(parents=True, exist_ok=True)
            have = count_rows(perf_json)
            for r in range(have, args.perf_repeats):
                print(f"    perf repeat {r + 1}/{args.perf_repeats}", flush=True)
                run_bench(args.model, pol, rate, args.perf_decode, perf_json, None,
                          args.greedy, log, force, hybrid_fetch, args.gpu)

            # Trace pass: eager + per-layer traffic CSV + system monitor.
            trace_json = OUT / "e2e" / "trace" / wl / f"{pol}.json"
            trace_csv = OUT / "e2e" / "trace" / wl / f"{pol}.trace.csv"
            trace_json.parent.mkdir(parents=True, exist_ok=True)
            if count_rows(trace_json) == 0:
                print("    trace run", flush=True)
                run_bench(args.model, pol, rate, args.trace_decode, trace_json, trace_csv,
                          args.greedy, log, force, hybrid_fetch, args.gpu)
    print("ALL CELLS DONE", flush=True)


if __name__ == "__main__":
    main()
