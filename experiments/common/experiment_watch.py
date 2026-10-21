"""Keep a background progress record and restore the tmux display if it dies."""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT
from experiments.common.progress import display, failure_detail, read_json, resolve_root
from experiments.pruning.moe_pruning import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--monitor-dir", type=Path, required=True)
    parser.add_argument("--session", default="m_reduction_v3")
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.interval < 2:
        parser.error("interval must be at least two seconds")
    directory = args.monitor_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    last_tmux_error = None
    ticks = 0
    while True:
        ticks += 1
        now = datetime.now(timezone.utc).isoformat()
        root = resolve_root(registry=args.registry)
        state = read_json(root / "progress.json")
        error = failure_detail(root)
        snapshot = display(root, crop=False)
        temporary = directory / "latest.txt.tmp"
        temporary.write_text(snapshot + "\n")
        temporary.replace(directory / "latest.txt")
        pane = subprocess.run(["tmux", "display-message", "-p", "-t", args.session + ":progress",
                               "#{pane_dead}"], text=True, capture_output=True)
        healthy = pane.returncode == 0 and pane.stdout.strip() == "0"
        tmux_heartbeat = read_json(directory / "tmux_heartbeat.json")
        if healthy and tmux_heartbeat.get("utc"):
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(tmux_heartbeat["utc"])).total_seconds()
            healthy = age <= max(30, 4 * args.interval)
        if not healthy:
            cmd = "cd " + shlex.quote(str(ROOT)) + " && source scripts/env.sh && uv run --no-sync python -m experiments.common.progress"
            cmd += " --registry " + shlex.quote(str(args.registry.resolve())) + " --watch --interval " + str(args.interval)
            cmd += " --heartbeat " + shlex.quote(str(directory / "tmux_heartbeat.json"))
            shell = "bash -c " + shlex.quote(cmd)
            session = subprocess.run(["tmux", "has-session", "-t", args.session], capture_output=True)
            if session.returncode:
                restore = ["tmux", "new-session", "-d", "-s", args.session, "-n", "progress", shell]
            elif pane.returncode:
                restore = ["tmux", "new-window", "-d", "-t", args.session, "-n", "progress", shell]
            else:
                restore = ["tmux", "respawn-pane", "-k", "-t", args.session + ":progress", shell]
            result = subprocess.run(restore, text=True, capture_output=True)
            last_tmux_error = result.stderr.strip() if result.returncode else None
            if not result.returncode:
                print(now + " restored tmux progress display", flush=True)
        heartbeat = {"utc": now, "pid": os.getpid(), "ticks": ticks, "status": "monitoring",
                     "study": str(root), "experiment_status": state.get("status"),
                     "experiment_stage": state.get("stage"), "last_rejected_attempt": error,
                     "tmux_healthy": healthy, "tmux_restore_error": last_tmux_error}
        atomic_json(directory / "heartbeat.json", heartbeat)
        key = (str(root), state.get("status"), state.get("stage"), error, healthy)
        if key != previous:
            with (directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(heartbeat) + "\n")
            print(now + " " + str(state.get("status")) + ": " + str(state.get("stage")), flush=True)
            if error:
                print("Last rejected attempt: " + error, flush=True)
            if state.get("status") == "failed":
                atomic_json(directory / "pending_error.json", {**heartbeat, "active": True,
                            "action": "Inspect retained failure; bounded retries never relax acceptance gates."})
            else:
                pending = read_json(directory / "pending_error.json")
                if pending and pending.get("active", True):
                    atomic_json(directory / "pending_error.json", {**pending, "active": False,
                                "resumed_utc": now, "current_status": state.get("status")})
            previous = key
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
