"""Bounded retries of whole experiment batches after operational failures."""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, engine_sources
from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.pruning_resources import read_resources


def event(root, kind, **details):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **details}
    with (root / "recovery_events.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(kind + ": " + json.dumps(details), flush=True)


def wait_quiet(root, stage, state, timeout=600, quiet_seconds=30):
    policy = json.loads((ROOT / "configs/pruning/resource_policy.json").read_text())
    deadline = time.monotonic() + timeout
    anchor = None
    next_report = 0
    while time.monotonic() < deadline:
        row = read_resources(-1)
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                          "--format=csv,noheader"], text=True).strip()
        good = (not active and row["host_available_bytes"] >= policy["minimum_host_available_bytes"]
                and row["memory_pressure"]["full"]["avg10"] <= policy["maximum_full_pressure_percent"])
        if not good:
            anchor = None
        elif anchor is None:
            anchor = row
        elif (row["swap_in_bytes"] - anchor["swap_in_bytes"] > 1048576
              or row["swap_out_bytes"] - anchor["swap_out_bytes"] > 1048576):
            anchor = row
        elif row["monotonic_seconds"] - anchor["monotonic_seconds"] >= quiet_seconds:
            event(root, "quiet_interval_passed", seconds=quiet_seconds,
                  available_bytes=row["host_available_bytes"])
            return
        if time.monotonic() >= next_report:
            seconds = row["monotonic_seconds"] - anchor["monotonic_seconds"] if anchor else 0
            state(root, "waiting_for_quiet", stage,
                  f"Waiting for {quiet_seconds}s quiet host/GPU interval ({seconds:.0f}s so far); resource limits unchanged.")
            next_report = time.monotonic() + 15
        time.sleep(5)
    raise RuntimeError("Host/GPU did not stay quiet within the 600-second retry wait")


def latest_batch(root, model, arm, count):
    parent = root / "captures" / model / arm / "pilot"
    candidates = []
    for path in parent.glob("*"):
        control = path / "configuration.json"
        if control.is_file() and len(json.loads(control.read_text())["request_ids"]) == count:
            candidates.append(path)
    return max(candidates, key=lambda p: p.name) if candidates else None


def completed_batch(root, model, arm, count, plan, profile):
    path = latest_batch(root, model, arm, count)
    if path is None or not (path / "completion.json").is_file():
        return None
    completion = json.loads((path / "completion.json").read_text())
    config = json.loads((path / "configuration.json").read_text())
    if (completion["status"] != "complete" or completion["captures"] != count
            or completion["resources"]["status"] != "passed"):
        return None
    actual = {name: sha256(source) for name, source in engine_sources().items()}
    if config["inference_hashes"] != actual:
        raise ValueError("Cannot resume a batch with changed numerical runtime: " + str(path))
    dependencies = config["dependencies"]
    if dependencies["plan"] != sha256(plan) or dependencies["calibration"] != sha256(profile):
        raise ValueError("Cannot resume a batch with changed plan/calibration: " + str(path))
    collector = config.get("collector_module", "m_reduction_collect")
    if collector not in {"m_reduction_collect", "m_reduction_nvfp4_collect"}:
        raise ValueError("Unknown collector dependency: " + collector)
    sources = {"collector_source": "experiments/pruning/" + collector + ".py",
               "policy_source": "experiments/pruning/m_reduction_policy.py",
               "replay_source": "experiments/pruning/m_reduction_replay.py",
               "resource_source": "experiments/pruning/pruning_resources.py",
               "standard_validator": "experiments/pruning/collect_moe_routes.py",
               "gemma_validator": "experiments/pruning/collect_gemma_routes.py",
               "shared_helpers": "experiments/pruning/moe_pruning.py",
               "decode_helpers": "experiments/baselines/bench_decode_moe.py"}
    if any(dependencies[name] != sha256(ROOT / source) for name, source in sources.items()):
        raise ValueError("Cannot resume a batch with changed collection/qualification dependencies")
    if dependencies["resource_control"] != sha256(ROOT / json.loads(plan.read_text())["resource_policy"]):
        raise ValueError("Cannot resume a batch with changed resource gates")
    if dependencies["rendered_inputs"] != sha256(root / "protocol/rendered_inputs.json"):
        raise ValueError("Cannot resume a batch with changed rendered inputs")
    manifests = list(path.glob("*/manifest.json"))
    if len(manifests) != count:
        raise ValueError("Completed batch has incomplete request manifests")
    return path


def collect_with_retries(root, model, arm, count, plan, profile, launch, state, stage,
                         collector_module="m_reduction_collect"):
    retryable = {"system swapping during inference exceeds limit",
                 "startup or inference host-memory headroom below minimum",
                 "sustained full memory pressure during inference"}
    for attempt in range(1, 4):
        if attempt > 1:
            wait_quiet(root, stage, state)
        state(root, "running", stage, f"Whole-batch attempt {attempt}/3; failed measurements remain excluded.")
        try:
            arguments = ["--study", root, "--plan", plan, "--model", model,
                         "--arm", arm, "--stage", "pilot", "--calibration", profile]
            if count != 15:
                arguments += ["--limit", count]
            launch(root, collector_module, *arguments, no_swap=True)
            return
        except subprocess.CalledProcessError:
            path = latest_batch(root, model, arm, count)
            summary = path / "resource_summary.json" if path else None
            row = json.loads(summary.read_text()) if summary and summary.is_file() else {}
            reasons = row.get("reasons", [])
            event(root, "batch_failed", arm=arm, attempt=attempt, batch=str(path), reasons=reasons)
            if not reasons or not set(reasons).issubset(retryable) or attempt == 3:
                raise
            event(root, "whole_batch_retry_scheduled", arm=arm, next_attempt=attempt + 1)
