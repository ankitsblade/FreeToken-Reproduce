"""Collect corrected-floor primary and repeat captures for one pilot-qualified model."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, engine_sources
from experiments.pruning.m_reduction_guard import event, wait_quiet
from experiments.pruning.m_reduction_stage0 import launch, state
from experiments.pruning.moe_pruning import atomic_json, sha256


COUNTS = {"primary": 90, "repeat": 18}
RETRYABLE = {"system swapping during inference exceeds limit",
             "startup or inference host-memory headroom below minimum",
             "sustained full memory pressure during inference"}


def batches(root, model, subset):
    return sorted((root / "captures" / model / "native_floor1" / subset).glob("*"), key=lambda p: p.name)


def accepted(root, model, subset, reference):
    for path in reversed(batches(root, model, subset)):
        completion = path / "completion.json"
        config = path / "configuration.json"
        if not completion.is_file() or not config.is_file():
            continue
        done, control = json.loads(completion.read_text()), json.loads(config.read_text())
        if (done.get("status") == "complete" and done.get("captures") == COUNTS[subset]
                and done.get("resources", {}).get("status") == "passed"
                and control["inference_hashes"] == reference["inference_hashes"]
                and control["dependencies"] == reference["dependencies"]):
            return path
    return None


def retryable(path):
    if path is None:
        return False
    failure = path / "failure.json"
    # A startup settle timeout is a whole-machine resource rejection before any measurement.
    if failure.is_file() and "did not settle" in json.loads(failure.read_text()).get("error", ""):
        return True
    summary = path / "resource_summary.json"
    reasons = json.loads(summary.read_text()).get("reasons", []) if summary.is_file() else []
    return bool(reasons) and set(reasons).issubset(RETRYABLE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collector", choices=("m_reduction_collect", "m_reduction_nvfp4_collect"), required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--deferral", type=Path, required=True)
    parser.add_argument("--model-study", action="append", default=[])
    args = parser.parse_args()
    root = args.study.resolve()
    os.environ["EXPERIMENT_RUNS_ROOT"] = str(root / "runs")
    reference = json.loads((root / "qualification" / args.model / "native_floor1.json").read_text())
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    if reference.get("status") != "qualified" or reference["inference_hashes"] != hashes:
        raise ValueError("Runtime does not match this model's qualified native_floor1 pilot")
    attempt = root / "primary_controllers" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt.mkdir(parents=True)
    atomic_json(attempt / "control.json", {"model": args.model, "plan": str(args.plan), "plan_sha256": sha256(args.plan),
                "collector": args.collector, "calibration_sha256": sha256(args.calibration),
                "deferral_sha256": sha256(args.deferral), "inference_hashes": hashes,
                "source_sha256": sha256(Path(__file__)), "runtime": os.environ["FREETOKEN_SOURCE_DIR"]})
    stage = args.model + ": primary transition gate"
    try:
        bindings = [item for entry in args.model_study for item in ("--model-study", entry)]
        launch(root, "m_reduction_plan_gate", "--study", root, *bindings, "--deferral", args.deferral,
               "--require-ready", "primary", "--require-model", args.model, "--out", attempt / "primary_transition.json")
        for subset in COUNTS:
            stage = f"{args.model}: corrected-floor {subset} captures"
            if accepted(root, args.model, subset, reference):
                event(root, "completed_capture_set_reused", model=args.model, subset=subset)
                continue
            for number in range(1, 4):
                if number > 1:
                    wait_quiet(root, stage, state)
                state(root, "running", stage, f"Whole-batch attempt {number}/3 of {COUNTS[subset]} captures; full scores and exact replay.")
                before = set(batches(root, args.model, subset))
                try:
                    launch(root, args.collector, "--study", root, "--plan", args.plan, "--model", args.model,
                           "--arm", "native_floor1", "--stage", subset, "--calibration", args.calibration, no_swap=True)
                except subprocess.CalledProcessError:
                    created = [path for path in batches(root, args.model, subset) if path not in before]
                    batch = created[-1] if created else None
                    event(root, "capture_set_failed", model=args.model, subset=subset, attempt=number, batch=str(batch))
                    if number == 3 or not retryable(batch):
                        raise
                    continue
                if not accepted(root, args.model, subset, reference):
                    raise RuntimeError(f"{args.model} {subset} collector returned without an accepted batch")
                event(root, "capture_set_completed", model=args.model, subset=subset, captures=COUNTS[subset])
                break
        stage = args.model + ": post-collection plan audit"
        launch(root, "m_reduction_plan_gate", "--study", root, *bindings, "--deferral", args.deferral,
               "--out", attempt / "post_collection_audit.json")
        state(root, "awaiting_p2_support", args.model + ": primary and repeat captures complete",
              "Deferred models and full P2 support still gate later research stages.")
    except BaseException as exc:
        atomic_json(attempt / "failure.json", {"stage": stage, "error": repr(exc)})
        state(root, "failed", stage, repr(exc) + "; attempt retained; dependent stages stopped.")
        raise


if __name__ == "__main__":
    main()
