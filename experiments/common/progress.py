"""Show recorded experiment progress in a terminal or tmux window."""

import argparse
import json
import re
import shutil
import time
import os
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, lines=8):
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            return stream.read().decode(errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def failure_detail(root):
    candidates = []
    for path in root.glob("captures/*/*/*/*/resource_summary.json"):
        row = read_json(path)
        if row.get("status") == "failed":
            candidates.append((path.stat().st_mtime, "; ".join(row.get("reasons", []))))
    for path in [*root.glob("controllers/*/*failure.json"), *root.glob("captures/*/*/*/*/**/failure.json")]:
        error = read_json(path).get("error", "")
        if error and not error.startswith("CalledProcessError"):
            candidates.append((path.stat().st_mtime, error))
    for path in root.glob("runs/*/*/run.json"):
        if read_json(path).get("status") in {"failed", "preflight_failed"}:
            errors = [line for line in tail(path.parent / "logs/console.log", 30)
                      if re.match(r"^(ValueError|RuntimeError|FileNotFoundError|OSError|AssertionError):", line)
                      or ": error:" in line]
            if errors:
                candidates.append((path.stat().st_mtime, errors[-1]))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def resolve_root(root=None, registry=None):
    return Path(read_json(registry)["active_study"]).resolve() if registry else root.resolve()


def display(root, crop=True):
    state = read_json(root / "progress.json")
    output = [f"Experiment: {state.get('experiment', root.name)}",
              f"Status: {state.get('status', 'waiting for controller')}",
              f"Stage: {state.get('stage', 'not started')}",
              f"Updated UTC: {state.get('updated_utc', '-')}",
              f"Monitor UTC: {datetime.now(timezone.utc).isoformat()}",
              f"Artifacts: {root}", ""]
    error = failure_detail(root)
    if error:
        output.insert(5, "Last rejected attempt: " + error)
    runs = sorted(root.glob("runs/*/*/run.json"))
    active_runs = [path for path in runs if read_json(path).get("status") == "running"]
    complete_runs = [path for path in runs if path not in active_runs]
    for path in active_runs + complete_runs[-2:]:
        row = read_json(path)
        baseline = read_json(path.parent / "baseline.json")
        results = sum(1 for result in path.parent.glob("repeat_*.jsonl")
                      if result.stat().st_size)
        detail = (f" [{results}/{baseline['repeats']} repetitions recorded]"
                  if baseline.get("repeats") else "")
        label = baseline.get("profile", row.get("module", path.parent.name))
        output.append(f"{label:24} "
                      f"{row.get('status', 'unknown'):18}{detail}")
    if state.get("message") and shutil.get_terminal_size(fallback=(120, 24)).lines >= 24:
        output.extend(["", str(state["message"])])
    gpu_modules = {"baseline", "m_reduction_collect", "m_reduction_nvfp4_collect", "m_reduction_calibrate", "m_reduction_numerics"}
    active = next((p for p in reversed(active_runs)
                   if read_json(p).get("module") in gpu_modules),
                  active_runs[-1] if active_runs else None)
    if active:
        logs = list((active.parent / "logs").glob("*.log"))
        if read_json(active).get("module") == "m_reduction_numerics":
            logs += list(root.glob("diagnostics/*/server.log"))
        if read_json(active).get("module") in {"m_reduction_collect", "m_reduction_nvfp4_collect"}:
            logs += list(root.glob("captures/*/*/*/*/server.log"))
            if logs:
                latest_capture = max(logs, key=lambda path: path.stat().st_mtime)
                if latest_capture.name == "server.log":
                    completed = read_json(latest_capture.parent / "completed.json")
                    config = read_json(latest_capture.parent / "configuration.json")
                    count = len(completed) if isinstance(completed, list) else 0
                    output.insert(5, f"Captures qualified: {count}/{len(config.get('request_ids', []))}")
        if logs:
            latest = max(logs, key=lambda p: p.stat().st_mtime)
            output.extend(["", f"Latest log: {latest.name}", *tail(latest)])
    output.extend(["", "Controller log:", *tail(root / "controller.log")])
    width, height = shutil.get_terminal_size(fallback=(120, 24))
    clean = [re.sub(r"\x1b\[[0-9;]*m", "", line).replace("\r", " ")
             for line in output]
    return "\n".join(line[:width - 1] for line in clean[:height - 1]) if crop else "\n".join(clean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path)
    source.add_argument("--registry", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--heartbeat", type=Path)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be at least one second")
    try:
        while True:
            if args.watch:
                print("\033[2J\033[H", end="")
            root = resolve_root(args.root, args.registry)
            print(display(root), flush=True)
            if args.heartbeat:
                from experiments.pruning.moe_pruning import atomic_json
                atomic_json(args.heartbeat, {"pid": os.getpid(), "study": str(root),
                            "utc": datetime.now(timezone.utc).isoformat(), "status": "monitoring"})
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
