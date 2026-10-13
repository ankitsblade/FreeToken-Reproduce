"""Sweep a fixed miss count across CPU MoE and PCIe gather branches."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import threading
import time
from pathlib import Path

import torch

from freetoken.moe.benchbw import (
    WORKLOADS,
    _build_cpu_moe_executor,
    _build_gather_rig,
    _cpu_moe_bank_sources,
    _expert_bytes,
)
from experiments.bandwidth.bandwidth_study import predicted_q


FORMATS = {"qwen3.6-moe": "nvfp4", "gpt-oss-20b": "mxfp4_triton"}


def measure_pair(cache, executor, task, io, ids: list[int], q: int,
                 start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> dict:
    m = len(ids)
    routed = ids[q:] + [-1] * (io["ids"].numel() - (m - q))
    io["ids"].copy_(torch.tensor(routed, dtype=torch.int32).view_as(io["ids"]))
    cache.src_indices[:q].copy_(torch.tensor(ids[:q], dtype=torch.int32, device=cache.device))
    cache.num_indices.fill_(q)
    torch.cuda.synchronize(cache.device)

    barrier = threading.Barrier(2)
    cpu = {}

    def run_cpu() -> None:
        barrier.wait()
        cpu["start"] = time.perf_counter()
        try:
            executor._ext.run_task(task)
        except BaseException as exc:
            cpu["error"] = exc
        cpu["end"] = time.perf_counter()

    worker = threading.Thread(target=run_cpu)
    worker.start()
    barrier.wait()
    pcie_start = time.perf_counter()
    start_event.record()
    cache.copy_missing()
    end_event.record()
    end_event.synchronize()
    pcie_end = time.perf_counter()
    worker.join()
    if "error" in cpu:
        raise cpu["error"]
    return {
        "cpu_branch_ms": 1000 * (cpu["end"] - cpu["start"]),
        "pcie_branch_ms": 1000 * (pcie_end - pcie_start),
        "pcie_device_ms": start_event.elapsed_time(end_event),
        "overlap_wall_ms": 1000 * (max(cpu["end"], pcie_end) -
                                   min(cpu["start"], pcie_start)),
    }


def summarize(raw: list[dict], fraction: float) -> list[dict]:
    grouped = {}
    for row in raw:
        grouped.setdefault((row["m"], row["q"]), []).append(row)
    out = []
    for m in sorted({row["m"] for row in raw}):
        medians = {q: statistics.median(x["overlap_wall_ms"] for x in grouped[(m, q)])
                   for q in range(m + 1)}
        oracle = min(medians, key=medians.get)
        predicted = predicted_q(m, fraction)
        for q in range(m + 1):
            group = grouped[(m, q)]
            item = {"model": group[0]["model"], "m": m, "q": q,
                    "n": len(group), "q_predicted": predicted, "q_oracle": oracle,
                    "q_error": predicted - oracle,
                    "latency_penalty_predicted": medians[predicted] / medians[oracle] - 1,
                    "expert_bytes": group[0]["expert_bytes"],
                    "scope": "synthetic_production_branches_no_gpu_gemm"}
            for key in ("cpu_branch_ms", "pcie_branch_ms", "pcie_device_ms",
                        "overlap_wall_ms", "cpu_effective_gbps", "pcie_effective_gbps"):
                values = [row[key] for row in group if row[key] is not None]
                item[f"{key}_median"] = statistics.median(values) if values else ""
                item[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0
            out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=FORMATS, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--misses", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wl, fmt = WORKLOADS[args.model], FORMATS[args.model]
    if not 0 <= args.fraction <= 1 or args.repeats < 2 or args.warmup < 0:
        parser.error("fraction must be in [0, 1], repeats >= 2, warmup >= 0")
    if not args.misses or min(args.misses) < 1 or max(args.misses) > wl.top_k:
        parser.error(f"miss counts must be in 1..{wl.top_k}")
    if args.out.exists():
        parser.error(f"{args.out} already exists")

    device = torch.device("cuda")
    expert_bytes = _expert_bytes(fmt, wl.hidden, wl.inter)
    cache, _ = _build_gather_rig(fmt, wl, device)
    E = cache.num_experts
    banks = _cpu_moe_bank_sources(fmt, wl.hidden, wl.inter, E)
    executor = _build_cpu_moe_executor(fmt, wl, banks, 0, E)
    io = executor._io_for(1)
    io["x"].fill_(0.01)
    io["w"].fill_(1 / wl.top_k)
    task = executor._task_for(0, 1)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    rng = random.Random(args.seed)
    raw = []
    for m in sorted(set(args.misses)):
        for repeat in range(-args.warmup, args.repeats):
            ids = [(repeat * m + i) % E for i in range(m)]
            caps = list(range(m + 1))
            rng.shuffle(caps)
            for q in caps:
                timing = measure_pair(cache, executor, task, io, ids, q,
                                      start_event, end_event)
                if repeat < 0:
                    continue
                cpu_bytes = (m - q) * expert_bytes
                pcie_bytes = q * expert_bytes
                raw.append({"model": args.model, "format": fmt, "seed": args.seed,
                            "repeat": repeat, "m": m, "q": q,
                            "expert_bytes": expert_bytes, "cpu_bytes": cpu_bytes,
                            "pcie_bytes": pcie_bytes, "cpu_isa": executor.isa,
                            **timing,
                            "cpu_effective_gbps": (cpu_bytes / (timing["cpu_branch_ms"] * 1e6)
                                                   if cpu_bytes else None),
                            "pcie_effective_gbps": (pcie_bytes / (timing["pcie_device_ms"] * 1e6)
                                                    if pcie_bytes else None)})
        print(f"{args.model} m={m}: {args.repeats} repeats/cell", flush=True)
    write_csv(args.out, raw)
    write_csv(args.out.with_name(args.out.stem + "_summary.csv"),
              summarize(raw, args.fraction))
    args.out.with_name(args.out.stem + "_config.json").write_text(json.dumps({
        "model": args.model, "format": fmt, "fraction": args.fraction,
        "misses": sorted(set(args.misses)), "repeats": args.repeats,
        "warmup": args.warmup, "seed": args.seed, "expert_bytes": expert_bytes,
        "synthetic_experts": E, "cpu_isa": executor.isa,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "caveat": "Production CPU MoE and gather branches only; routing, GPU GEMM, and cache admission excluded",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
