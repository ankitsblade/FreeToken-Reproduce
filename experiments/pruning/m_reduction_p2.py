"""Qualify P2 support for one model on the isolated teacher-forcing runtime.

Stages, each a whole batch with at most three resource-retried attempts:
1. Baseline teacher forcing of all 90 accepted primary histories (logits, MoE inputs,
   event-by-event identity with the source captures, greedy targets).
2. Zero-change native control on the nine repeat request IDs (KL 0, agreement 1).
3. Expert pre-gate norms and fold-specific probes from the baseline MoE inputs.
The model's p2_support.json is written only when all three complete.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, engine_sources
from experiments.pruning.m_reduction_guard import event, wait_quiet
from experiments.pruning.m_reduction_primary import retryable
from experiments.pruning.m_reduction_stage0 import launch, state
from experiments.pruning.moe_pruning import atomic_json, sha256

FEATURES = ["sequential_teacher_forcing", "full_vocabulary_distributions", "expert_norms_and_probes"]
EXPECTED = {"all": 90, "repeat_ids": 9}


def batches(root, model, mode, subset):
    return sorted((root / "captures" / model / ("teacher_" + mode) / subset).glob("*"), key=lambda p: p.name)


def accepted(root, model, mode, subset, hashes):
    for path in reversed(batches(root, model, mode, subset)):
        completion, config = path / "completion.json", path / "configuration.json"
        if not completion.is_file() or not config.is_file():
            continue
        done, control = json.loads(completion.read_text()), json.loads(config.read_text())
        if (done.get("status") == "complete" and done.get("captures") == EXPECTED[subset] and not done.get("limit")
                and done.get("resources", {}).get("status") == "passed" and control["inference_hashes"] == hashes):
            return path
    return None


def teacher(root, args, mode, subset, hashes, reference=None):
    stage = f"{args.model}: teacher forcing {mode}/{subset}"
    found = accepted(root, args.model, mode, subset, hashes)
    if found:
        event(root, "teacher_batch_reused", model=args.model, mode=mode, subset=subset, batch=str(found))
        return found
    extra = ["--reference-batch", reference] if reference else []
    for number in range(1, 4):
        if number > 1:
            wait_quiet(root, stage, state)
        state(root, "running", stage, f"Whole-batch attempt {number}/3 of {EXPECTED[subset]} histories.")
        before = set(batches(root, args.model, mode, subset))
        try:
            launch(root, "m_reduction_teacher", "--study", root, "--source-study", args.source_study, "--model", args.model,
                   "--calibration", args.calibration, "--mode", mode, "--subset", subset, *extra, no_swap=True)
        except subprocess.CalledProcessError:
            created = [p for p in batches(root, args.model, mode, subset) if p not in before]
            batch = created[-1] if created else None
            event(root, "teacher_batch_failed", model=args.model, mode=mode, subset=subset, attempt=number, batch=str(batch))
            if number == 3 or not retryable(batch):
                raise
            continue
        found = accepted(root, args.model, mode, subset, hashes)
        if not found:
            raise RuntimeError(stage + " returned without an accepted batch")
        event(root, "teacher_batch_completed", model=args.model, mode=mode, subset=subset, batch=str(found))
        return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="P2 support root")
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True, help="m_reduction_experts --validate report")
    args = parser.parse_args()
    root, source = args.study.resolve(), args.source_study.resolve()
    os.environ["EXPERIMENT_RUNS_ROOT"] = str(root / "runs")
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    reference = json.loads((source / "qualification" / args.model / "native_floor1.json").read_text())
    if reference.get("status") != "qualified":
        raise ValueError(args.model + ": no qualified native_floor1 pilot in the source study")
    validation = json.loads(args.validation.read_text())[args.model]
    if validation["max_relative_l2_cpu"] > 0.01 or validation["min_cosine_cpu"] < 0.9999:
        raise ValueError(args.model + ": reference expert function does not match native expert outputs")
    attempt = root / "controllers" / f"{args.model}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    attempt.mkdir(parents=True)
    atomic_json(attempt / "control.json", {"model": args.model, "source_study": str(source),
                "calibration_sha256": sha256(args.calibration), "validation_sha256": sha256(args.validation),
                "inference_hashes": hashes, "source_sha256": sha256(Path(__file__)),
                "runtime": os.environ["FREETOKEN_SOURCE_DIR"]})
    stage = args.model + ": P2 support"
    try:
        baseline = teacher(root, args, "baseline", "all", hashes)
        control = teacher(root, args, "native_control", "repeat_ids", hashes, reference=baseline)
        stage = args.model + ": expert norms and probes"
        outputs = root / "expert_outputs" / args.model / baseline.name
        if not (outputs / "completion.json").is_file():
            state(root, "running", stage, "Reference experts over 90 baseline histories; GPU only, no server.")
            launch(root, "m_reduction_experts", "--teacher-batch", baseline, "--validation", args.validation,
                   "--out", outputs, no_swap=True)
        support = {"status": "qualified", "qualified_features": FEATURES, "model": args.model,
                   "inference_hashes": hashes, "reproduces_inference_hashes": reference["inference_hashes"],
                   "runtime": os.environ["FREETOKEN_SOURCE_DIR"],
                   "runtime_patch_sha256": sha256(ROOT / "runtime/patches/m-reduction-p2-teacher.patch"),
                   "evidence": {"baseline_teacher_batch": str(baseline),
                                "baseline_completion_sha256": sha256(baseline / "completion.json"),
                                "native_control_batch": str(control),
                                "native_control_completion_sha256": sha256(control / "completion.json"),
                                "expert_outputs": str(outputs),
                                "expert_outputs_completion_sha256": sha256(outputs / "completion.json"),
                                "expert_validation": str(args.validation.resolve()),
                                "expert_validation_sha256": sha256(args.validation)},
                   "scope": ("Baseline teacher forcing reproduced every accepted primary capture event by event with "
                             "greedy targets; native control scored KL 0 against saved full-vocabulary logits. "
                             "Candidate live policies still need their own runtime qualification."),
                   "qualified_utc": datetime.now(timezone.utc).isoformat()}
        target = source / "qualification" / args.model / "p2_support.json"
        if target.exists():
            raise FileExistsError(str(target) + " already exists; P2 support records are immutable")
        atomic_json(target, support)
        atomic_json(attempt / "p2_support.json", support)
        state(root, "p2_qualified", args.model + ": P2 support qualified", str(target))
    except BaseException as exc:
        atomic_json(attempt / "failure.json", {"stage": stage, "error": repr(exc)})
        state(root, "failed", stage, repr(exc) + "; attempt retained; dependent stages stopped.")
        raise


if __name__ == "__main__":
    main()
