"""Complete the Gemma-first 75% collection and offline work in serial stages."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pruning.moe_pruning import ROOT, atomic_json


def run(args):
    study = Path(args.study).resolve()
    output = study / "orchestration" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    progress = {"status": "running", "primary_closed": False, "kimi": "excluded_by_user", "jobs": []}

    def save():
        progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "progress.json", progress)

    def launch(module, arguments, name):
        command = ["uv", "run", "--no-sync", "--project", str(ROOT), "python", "-m", module, *arguments]
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            progress["jobs"].append({"name": name, "status": "running", "command": command, "pid": process.pid})
            save()
            while process.poll() is None:
                time.sleep(30)
                save()
            if process.returncode:
                progress["jobs"][-1].update(status="failed", returncode=process.returncode)
                raise RuntimeError(f"{name} failed; inspect its log")
            progress["jobs"][-1]["status"] = "complete"
            save()

    def accepted_batch(target, model, stage, batch, count):
        path = target / ("pilots" if stage == "pilot" else "routes") / model / batch / "completion.json"
        value = json.loads(path.read_text())
        if value.get("resource_status") != "passed" or value.get("captures") != count:
            raise ValueError(f"{batch}: incomplete or rejected resource/capture gate")

    def collect(target, module, model, stage, batch, count):
        completion = target / ("pilots" if stage == "pilot" else "routes") / model / batch / "completion.json"
        if args.reuse_completed and completion.exists():
            accepted_batch(target, model, stage, batch, count)
            progress["jobs"].append({"name": batch, "status": "complete", "reused": True})
            save()
            return
        launch(module, ["--study", str(target), "--model", model, "--stage", stage,
                        "--batch-id", batch], batch)
        accepted_batch(target, model, stage, batch, count)

    with (study / "orchestration/collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            gemma = study / "additional/gemma_v1"
            qwen = study / "additional/qwen_allocator_v6"
            progress["jobs"].append({"name": "gemma_full_pilot_v1", "status": "waiting_existing_collector"})
            save()
            pid = args.gemma_collector_pid
            while Path(f"/proc/{pid}").exists():
                command = Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"experiments.pruning.collect_gemma_routes" not in command:
                    raise ValueError("Gemma collector PID no longer matches the original job")
                time.sleep(30)
                save()
            accepted_batch(gemma, "gemma_nvfp4", "pilot", "gemma_full_pilot_v1", 15)
            progress["jobs"][-1]["status"] = "complete"
            save()
            for target, module, model, stage, batch, count in [
                (qwen, "experiments.pruning.collect_moe_routes", "qwen_nvfp4", "pilot", "qwen_allocator_full_pilot_v6", 15),
                (gemma, "experiments.pruning.collect_gemma_routes", "gemma_nvfp4", "primary", args.gemma_primary_batch, 90),
                (gemma, "experiments.pruning.collect_gemma_routes", "gemma_nvfp4", "repeat", "gemma_repeat_v1", 18),
                (qwen, "experiments.pruning.collect_moe_routes", "qwen_nvfp4", "repeat", args.qwen_repeat_batch, 18)]:
                collect(target, module, model, stage, batch, count)
            for model in ("qwen_nvfp4", "gpt_oss_mxfp4"):
                launch("experiments.pruning.moe_pruning", ["validate-replay", "--study", str(study), "--model", model,
                                                  "--stage", "primary"], f"native_{model}_primary")
            for target, model in [(study, "gpt_oss_mxfp4"), (qwen, "qwen_nvfp4")]:
                launch("experiments.pruning.moe_pruning", ["validate-replay", "--study", str(target), "--model", model,
                                                  "--stage", "repeat"], f"native_{model}_repeat")
            launch("experiments.pruning.moe_pruning", ["analyze", "--study", str(gemma), "--retention", ".75"], "gemma_analysis75")
            launch("experiments.pruning.pruning_gemma_controls", ["--study", str(gemma)], "gemma_native_controls")
            for target, label in [(study, "core"), (gemma, "gemma")]:
                launch("experiments.pruning.pruning_replay_analysis", ["--study", str(target)], f"{label}_replay75")
            for primary, repeat, model, batch in [(study, study, "gpt_oss_mxfp4", "gpt_repeat_v5"),
                                                 (study, qwen, "qwen_nvfp4", args.qwen_repeat_batch),
                                                 (gemma, gemma, "gemma_nvfp4", "gemma_repeat_v1")]:
                launch("experiments.pruning.pruning_repeatability", ["--primary-study", str(primary),
                       "--repeat-study", str(repeat), "--model", model, "--batch-id", batch,
                       "--output", str(study / "repeatability75" / model)], f"repeatability_{model}")
            launch("experiments.pruning.pruning_report75", ["--study", str(study), "--gemma-study", str(gemma),
                                                    "--qwen-repeat-study", str(qwen)], "report75")
            progress["status"] = "75_closed"
            progress["primary_closed"] = True
            save()
        except BaseException as exc:
            progress.update(status="stopped_on_failure", error=repr(exc))
            save()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gemma-collector-pid", type=int, required=True)
    parser.add_argument("--qwen-repeat-batch", default="qwen_allocator_repeat_v6")
    parser.add_argument("--gemma-primary-batch", default="gemma_primary_v1")
    parser.add_argument("--reuse-completed", action="store_true")
    run(parser.parse_args())
