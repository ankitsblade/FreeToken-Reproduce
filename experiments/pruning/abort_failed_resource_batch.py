"""Stop owned actual captures once recorded swap counters prove rejection."""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from experiments.pruning.moe_pruning import atomic_json, sha256


def run(args):
    progress_path = Path(args.capture_progress).resolve()
    stopped = set()
    while True:
        progress = json.loads(progress_path.read_text())
        if progress["status"] != "running":
            return
        active = [job for job in progress["jobs"] if job.get("status") == "running" and "command" in job]
        for job in active:
            command = job["command"]
            if "experiments.pruning.run_masked_inference75" not in command or job["name"] in stopped:
                continue
            study = Path(command[command.index("--study") + 1])
            model = command[command.index("--model") + 1]
            batch = command[command.index("--batch-id") + 1]
            target = study / "runs" / model / batch
            path = target / "resources.jsonl"
            if not path.exists():
                continue
            raw = path.read_text()
            lines = raw.splitlines() if raw.endswith("\n") else raw.splitlines()[:-1]
            rows = [json.loads(line) for line in lines]
            inference = [row for row in rows if row.get("phase") == "inference"]
            if len(inference) < 2:
                continue
            policy = json.loads((study / "protocol" / (model + "_resources.json")).read_text())
            first, last = inference[0], inference[-1]
            swap_in = last["swap_in_bytes"] - first["swap_in_bytes"]
            swap_out = last["swap_out_bytes"] - first["swap_out_bytes"]
            if max(swap_in, swap_out) <= policy["maximum_inference_swap_bytes"]:
                continue
            clients = []
            for directory in Path("/proc").iterdir():
                if not directory.name.isdigit():
                    continue
                try:
                    stat = (directory / "stat").read_text()
                    fields = stat[stat.rfind(")") + 2:].split()
                    argv = (directory / "cmdline").read_bytes().decode().split("\0")
                    if int(fields[1]) == job["pid"] and "experiments.pruning.run_masked_inference75" in argv:
                        clients.append(int(directory.name))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
            if len(clients) != 1:
                raise ValueError("cannot uniquely identify this controller's owned capture client")
            rejection = dict(status="failed", reasons=["system swapping during inference exceeds limit"],
                             observation="partial fail-fast rejection; first recorded inference sample gives a conservative lower bound",
                             swap_in_lower_bound_bytes=swap_in, swap_out_lower_bound_bytes=swap_out,
                             maximum_inference_swap_bytes=policy["maximum_inference_swap_bytes"],
                             first_recorded_inference_sample=first, last_recorded_inference_sample=last,
                             numerical_data_accepted=False, source_sha256=sha256(Path(__file__)))
            atomic_json(target / "resource_abort.json", rejection)
            atomic_json(target / "resource_summary.json", rejection)
            os.kill(clients[0], signal.SIGINT)
            stopped.add(job["name"])
            print(f"resource gate exceeded; interrupted owned capture {batch}, pid {clients[0]}", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-progress", required=True)
    run(parser.parse_args())
