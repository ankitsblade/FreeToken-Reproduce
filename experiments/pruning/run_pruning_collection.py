"""Run the qualified primary and repeat captures sequentially, then analyze 75%."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pruning.moe_pruning import ROOT, atomic_json


def uv_command(module, *arguments):
    return ["uv", "run", "--no-sync", "--project", str(ROOT), "python", "-m", module, *arguments]


def run(args):
    study = Path(args.study).resolve()
    output = study / "orchestration" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    progress = {"status": "running", "primary_closed": False, "jobs": [],
                "deferred_60_50": "blocked_until_75_closeout"}

    def save():
        progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "progress.json", progress)

    def wait_process(pid, expected, process=None, batch=None):
        deadline = time.monotonic() + 7200
        while True:
            if process is None and not Path(f"/proc/{pid}").exists():
                break
            if process is not None and process.poll() is not None:
                break
            if process is None:
                try:
                    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                except FileNotFoundError:
                    break
                tokens = [token.decode() for token in command if token]
                if not all(token in tokens for token in expected):
                    raise ValueError("the existing collector PID no longer identifies the expected job")
            if time.monotonic() > deadline:
                raise TimeoutError(f"job still running after two hours: PID {pid}; process preserved")
            if batch is not None and (batch / "completed.json").exists():
                progress["jobs"][-1]["captures_written"] = len(json.loads((batch / "completed.json").read_text()))
            save()
            time.sleep(30)
        if process is not None and process.returncode != 0:
            progress["jobs"][-1].update(status="failed", returncode=process.returncode)
            raise RuntimeError(f"job failed with exit code {process.returncode}")

    def launch(command, name, batch=None):
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            progress["jobs"].append({"name": name, "command": command, "pid": process.pid, "status": "running"})
            save()
            wait_process(process.pid, [], process, batch)
            progress["jobs"][-1]["status"] = "complete"
            save()

    with (study / "orchestration/collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            jobs = [("gpt_oss_mxfp4", "primary", "gpt_primary_v5", 90),
                    ("qwen_nvfp4", "primary", "qwen_primary_v5", 90),
                    ("gpt_oss_mxfp4", "repeat", "gpt_repeat_v5", 18),
                    ("qwen_nvfp4", "repeat", args.qwen_repeat_batch_id, 18)]
            for index, (model, stage, batch_id, count) in enumerate(jobs):
                batch = study / "routes" / model / batch_id
                command = uv_command("experiments.pruning.collect_moe_routes", "--study", str(study),
                                     "--model", model, "--stage", stage, "--batch-id", batch_id)
                if index == 0 and args.existing_collector_pid:
                    progress["jobs"].append({"name": batch_id, "pid": args.existing_collector_pid,
                                             "status": "running", "command": command, "attached": True})
                    save()
                    wait_process(args.existing_collector_pid,
                                 ["experiments.pruning.collect_moe_routes", model, stage, batch_id], batch=batch)
                elif not batch.exists():
                    launch(command, batch_id, batch)
                else:
                    progress["jobs"].append({"name": batch_id, "status": "checking_existing_batch"})
                completion = json.loads((batch / "completion.json").read_text())
                if completion.get("resource_status") != "passed" or completion.get("captures") != count:
                    raise ValueError(f"{batch_id}: required capture count/resource gate failed: {completion}")
                progress["jobs"][-1].update(status="complete", captures=count)
                save()
            for model, stage, _, _ in jobs:
                launch(uv_command("experiments.pruning.moe_pruning", "validate-replay", "--study", str(study),
                                  "--model", model, "--stage", stage), f"replay_{model}_{stage}")
            analysis = study / "retention75"
            if not analysis.exists():
                launch(uv_command("experiments.pruning.moe_pruning", "analyze", "--study", str(study),
                                  "--retention", "0.75"), "analysis75")
            if not (analysis / "analysis_complete.json").exists():
                raise ValueError("75% analysis is incomplete; preserve artifacts and investigate")
            progress["status"] = "collection_native_replay_and_coverage_analysis_complete"
            progress["next_action"] = "finish counterfactual replay, repeatability, family dispositions and 75% closeout"
            save()
        except BaseException as exc:
            progress.update(status="stopped_on_failure", error=repr(exc))
            save()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--existing-collector-pid", type=int)
    parser.add_argument("--qwen-repeat-batch-id", default="qwen_repeat_v5")
    run(parser.parse_args())
