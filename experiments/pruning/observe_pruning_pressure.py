"""Record process fault activity without changing the inference worker or gates."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def observation():
    processes = {}
    errors = 0
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text()
            fields = stat[stat.rfind(")") + 2:].split()
            values = {}
            for line in (directory / "status").read_text().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            processes[directory.name] = dict(
                name=values["Name"], ppid=int(fields[1]), starttime=int(fields[19]),
                minor_faults=int(fields[7]), major_faults=int(fields[9]),
                rss_bytes=int(values.get("VmRSS", "0 kB").split()[0]) * 1024,
                swap_bytes=int(values.get("VmSwap", "0 kB").split()[0]) * 1024,
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, ValueError, KeyError):
            errors += 1
    vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    return dict(
        monotonic_seconds=time.monotonic(), processes=processes,
        swap_in_bytes=int(vm["pswpin"]) * os.sysconf("SC_PAGE_SIZE"),
        swap_out_bytes=int(vm["pswpout"]) * os.sysconf("SC_PAGE_SIZE"),
        host_available_bytes=memory["MemAvailable"],
        swap_used_bytes=memory["SwapTotal"] - memory["SwapFree"],
        inaccessible_processes=errors,
    )


def run(args):
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=False)
    allowed = os.sched_getaffinity(0)
    outside_workers = {cpu for cpu in allowed if cpu >= 5}
    if outside_workers:
        os.sched_setaffinity(0, outside_workers)
    (target / "protocol.json").write_text(json.dumps(dict(
        sampling_seconds=5, affinity=sorted(os.sched_getaffinity(0)),
        capture_progress=args.capture_progress,
        interpretation="Major faults include file-backed faults; VmSwap excludes shmem. Correlation does not identify swap IO exactly.",
        scope="Process metadata only; no process memory contents; original gate unchanged.",
    ), indent=2) + "\n")
    with (target / "process_resources.jsonl").open("x") as stream:
        while True:
            state = json.loads(Path(args.capture_progress).read_text())
            if state["status"] != "running":
                (target / "completion.json").write_text(json.dumps(dict(
                    status="observer_stopped", capture_status=state["status"],
                ), indent=2) + "\n")
                return
            active = [job for job in state["jobs"] if job.get("status") == "running"]
            row = observation()
            row["active_batch"] = active[-1]["name"] if active else None
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            time.sleep(5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-progress", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
