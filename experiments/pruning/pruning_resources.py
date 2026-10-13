"""Linux resource observations and prospective inference qualification."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def read_resources(root_pid):
    entries = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = {}
            for line in (directory / "status").read_text().splitlines():
                name, value = line.split(":", 1)
                if name in ("PPid", "VmRSS", "VmSwap"):
                    fields[name] = int(value.strip().split()[0])
            entries[int(directory.name)] = (fields["PPid"], fields.get("VmRSS", 0) * 1024,
                                            fields.get("VmSwap", 0) * 1024)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    children = {root_pid}
    while True:
        expanded = children | {pid for pid, (parent, _, _) in entries.items() if parent in children}
        if expanded == children:
            break
        children = expanded
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()}
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    pressure = {}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        name, *fields = line.split()
        pressure[name] = {key: float(value) for key, value in (field.split("=") for field in fields)}
    return {"monotonic_seconds": time.monotonic(),
            "rss_by_pid": {str(pid): entries[pid][1] for pid in children if pid in entries},
            "swap_by_pid": {str(pid): entries[pid][2] for pid in children if pid in entries},
            "collector_rss_bytes": entries.get(os.getpid(), (0, 0, 0))[1],
            "host_available_bytes": memory["MemAvailable"],
            "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
            "swap_in_bytes": int(vm["pswpin"]) * os.sysconf("SC_PAGE_SIZE"),
            "swap_out_bytes": int(vm["pswpout"]) * os.sysconf("SC_PAGE_SIZE"),
            "oom_kill_count": int(vm["oom_kill"]), "memory_pressure": pressure}


def monitor_resources(process, stop, target, phase=None):
    with target.open("x") as stream:
        while not stop.is_set():
            try:
                label = phase["name"] if phase is not None else "unspecified"
                row = {**read_resources(process.pid), "phase": label}
            except Exception as exc:
                row = {"error": repr(exc), "monotonic_seconds": time.monotonic()}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if "error" in row:
                return
            stop.wait(1)


def quiet_interval(previous, current, policy):
    return (current["host_available_bytes"] >= policy["minimum_host_available_bytes"]
            and current["swap_in_bytes"] - previous["swap_in_bytes"] <= policy["settle_swap_bytes_per_sample"]
            and current["swap_out_bytes"] - previous["swap_out_bytes"] <= policy["settle_swap_bytes_per_sample"]
            and current["memory_pressure"]["full"]["avg10"] <= policy["maximum_full_pressure_percent"])


def wait_for_quiet(process, policy):
    start = time.monotonic()
    previous = read_resources(process.pid)
    quiet_since = start
    while time.monotonic() - start < policy["settle_timeout_seconds"]:
        if process.poll() is not None:
            raise RuntimeError("server exited while waiting for memory to settle")
        time.sleep(1)
        current = read_resources(process.pid)
        if not quiet_interval(previous, current, policy):
            quiet_since = current["monotonic_seconds"]
        if current["monotonic_seconds"] - quiet_since >= policy["settle_quiet_seconds"]:
            return current
        previous = current
    raise RuntimeError("host memory did not settle within the bounded startup wait")


def summarize_resources(samples, boundaries, policy):
    reasons = []
    if not samples or any("error" in sample for sample in samples):
        return {"status": "failed", "reasons": ["incomplete resource observations"]}
    begin, end = boundaries["inference_begin"], boundaries["inference_end"]
    inference = [begin, *(r for r in samples if begin["monotonic_seconds"] < r["monotonic_seconds"] < end["monotonic_seconds"]), end]
    duration = end["monotonic_seconds"] - begin["monotonic_seconds"]
    if duration <= 0 or len(inference) < policy["minimum_inference_samples"]:
        reasons.append("insufficient inference resource observations")
    minimum = min(r["host_available_bytes"] for r in [*samples, *boundaries.values()])
    swap_in = end["swap_in_bytes"] - begin["swap_in_bytes"]
    swap_out = end["swap_out_bytes"] - begin["swap_out_bytes"]
    process_swap_begin = sum(begin["swap_by_pid"].values())
    process_swap_peak = max(sum(r["swap_by_pid"].values()) for r in inference)
    full_stall = end["memory_pressure"]["full"]["total"] - begin["memory_pressure"]["full"]["total"]
    full_percent = full_stall / (duration * 10000) if duration > 0 else None
    rolling = []
    left = 0
    for right, row in enumerate(inference):
        while left < right and row["monotonic_seconds"] - inference[left]["monotonic_seconds"] > policy["pressure_window_seconds"] + 1.5:
            left += 1
        seconds = row["monotonic_seconds"] - inference[left]["monotonic_seconds"]
        if seconds >= policy["pressure_window_seconds"]:
            delta = row["memory_pressure"]["full"]["total"] - inference[left]["memory_pressure"]["full"]["total"]
            rolling.append(delta / (seconds * 10000))
    if minimum < policy["minimum_host_available_bytes"]:
        reasons.append("startup or inference host-memory headroom below minimum")
    if swap_in > policy["maximum_inference_swap_bytes"] or swap_out > policy["maximum_inference_swap_bytes"]:
        reasons.append("system swapping during inference exceeds limit")
    if policy.get("maximum_server_swap_bytes") is not None and process_swap_peak > policy["maximum_server_swap_bytes"]:
        reasons.append("server process swap exceeds absolute limit")
    if process_swap_peak - process_swap_begin > policy["maximum_server_swap_growth_bytes"]:
        reasons.append("server process swap grows during inference")
    if (full_percent is not None and full_percent > policy["maximum_full_pressure_percent"]) or (
            rolling and max(rolling) > policy["maximum_full_pressure_percent"]):
        reasons.append("sustained full memory pressure during inference")
    if end["oom_kill_count"] != boundaries["startup_begin"]["oom_kill_count"]:
        reasons.append("system OOM kill occurred during the batch")
    if any(r["oom_kill_count"] != boundaries["startup_begin"]["oom_kill_count"] for r in samples):
        reasons.append("system OOM kill observed")
    return {"status": "passed" if not reasons else "failed", "reasons": reasons,
            "host_available_bytes_min": minimum, "inference_seconds": duration,
            "inference_samples": len(inference), "inference_swap_in_bytes": swap_in,
            "inference_swap_out_bytes": swap_out, "server_swap_bytes_peak": process_swap_peak,
            "server_swap_growth_bytes": max(0, process_swap_peak - process_swap_begin),
            "inference_full_pressure_percent": full_percent,
            "rolling_full_pressure_percent_max": max(rolling) if rolling else None,
            "startup_swap_out_bytes": begin["swap_out_bytes"] - boundaries["startup_begin"]["swap_out_bytes"],
            "startup_swap_in_bytes": begin["swap_in_bytes"] - boundaries["startup_begin"]["swap_in_bytes"],
            "scope": "system counters plus server-subtree observations; no bank-level swapping attribution"}
