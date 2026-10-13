"""Estimate CUDA activity within profiled layer replay windows."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from contextlib import closing
from pathlib import Path


def covered_us(events: list[dict], start: float, end: float) -> float:
    intervals = sorted((max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
                       for event in events if event["ts"] < end and
                       event["ts"] + event["dur"] > start)
    total = 0.0
    right = start
    for left, stop in intervals:
        if stop > right:
            total += stop - max(left, right)
            right = stop
    return total


def summarize(trace: Path, config: Path) -> list[dict]:
    events = json.loads(trace.read_text())["traceEvents"]
    settings = json.loads(config.read_text())
    return summarize_events(events, settings)


def summarize_events(events: list[dict], settings: dict) -> list[dict]:
    repeats = int(settings["repeats"])
    captures = defaultdict(list)
    for event in events:
        if event.get("cat") == "user_annotation" and event.get("name", "").startswith(
                "moe_replay_q") and event.get("ph") == "X":
            q = int(event["name"].removeprefix("moe_replay_q"))
            captures[q].append(event)
    kernels = [event for event in events if event.get("cat") == "kernel" and
               event.get("ph") == "X"]
    h2d = [event for event in events if event.get("cat") == "gpu_memcpy" and
           "HtoD" in event.get("name", "") and event.get("ph") == "X"]
    roles = {"nvfp4_expert_kernel": [event for event in kernels
                                    if "_decode_nvfp4_marlin_kernel" in event.get("name", "")],
             "gather_kernel": [event for event in kernels
                               if "fast_index_copy_multi" in event.get("name", "")]}
    if set(captures) != set(range(int(settings["m"]) + 1)):
        raise ValueError("profile is missing one or more q windows")
    out = []
    for q, windows in sorted(captures.items()):
        if len(windows) < repeats:
            raise ValueError(f"q={q} has too few replay windows")
        measured = sorted(windows, key=lambda event: event["ts"])[-repeats:]
        durations = []
        kernel_pct = []
        h2d_pct = []
        role_times = defaultdict(list)
        for event in measured:
            start = event["ts"]
            end = start + event["dur"]
            durations.append(event["dur"] / 1000)
            kernel_pct.append(100 * covered_us(kernels, start, end) / event["dur"])
            h2d_pct.append(100 * covered_us(h2d, start, end) / event["dur"])
            for role, selected in roles.items():
                if selected:
                    role_times[role].append(covered_us(selected, start, end) / 1000)
        out.append({
            "layer_id": settings["layer_id"], "m": settings["m"], "q": q,
            "n": repeats, "profiled_wall_ms_median": statistics.median(durations),
            "cuda_kernel_active_pct_median": statistics.median(kernel_pct),
            "h2d_copy_active_pct_median": statistics.median(h2d_pct),
            "scope": "profiled_replay_cuda_timeline_not_sm_occupancy",
        })
        for role in roles:
            out[-1][role + "_active_ms_median"] = (
                statistics.median(role_times[role]) if role_times[role] else "")
    return out


def summarize_nsys(trace: Path, config: Path) -> list[dict]:
    settings = json.loads(config.read_text())
    pattern = re.compile(r"moe_replay_layer_(\d+)_m(\d+)_q(\d+)$")
    events = []
    pids = set()
    with closing(sqlite3.connect(trace.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        windows = db.execute(
            "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) "
            "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id "
            "WHERE n.end IS NOT NULL ORDER BY n.start")
        for start, end, tid, name in windows:
            match = pattern.fullmatch(name or "")
            if not match or tuple(map(int, match.groups()[:2])) != (
                    int(settings["layer_id"]), int(settings["m"])):
                continue
            pids.add(tid >> 24)
            events.append({"cat": "user_annotation", "ph": "X",
                           "name": "moe_replay_q" + match[3],
                           "ts": start / 1000, "dur": (end - start) / 1000})
        expected = 1 + (int(settings["warmup"]) + int(settings["repeats"])) * (
            int(settings["m"]) + 1)
        if len(pids) != 1 or len(events) != expected:
            raise ValueError("Nsight trace must contain one complete process-local replay sweep")
        pid = next(iter(pids))
        for start, end, name in db.execute(
                "SELECT k.start,k.end,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id WHERE (k.globalPid >> 24)=?", (pid,)):
            events.append({"cat": "kernel", "ph": "X", "name": name,
                           "ts": start / 1000, "dur": (end - start) / 1000})
        for start, end in db.execute(
                "SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE (globalPid >> 24)=? AND copyKind=1", (pid,)):
            events.append({"cat": "gpu_memcpy", "ph": "X", "name": "HtoD",
                           "ts": start / 1000, "dur": (end - start) / 1000})
    rows = summarize_events(events, settings)
    for row in rows:
        row["profiler"] = "nsight_systems_sqlite"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument("--sqlite", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} already exists")
    rows = (summarize_nsys(args.sqlite, args.config) if args.sqlite
            else summarize(args.trace, args.config))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.out)


if __name__ == "__main__":
    main()
