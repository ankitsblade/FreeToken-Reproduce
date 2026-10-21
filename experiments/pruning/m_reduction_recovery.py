"""Requalify a numerical correction with fresh profiles and retained failures."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
import uuid
from pathlib import Path

from experiments.paths import ROOT
from experiments.pruning.m_reduction_stage0 import launch, state
from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.m_reduction_guard import completed_batch, collect_with_retries, event, wait_quiet


def identities(root, model, arms, expected, stage_name, plan=None, profile=None):
    hashes = {}
    for arm in arms:
        batch = completed_batch(root, model, arm, expected, plan, profile)
        if batch is None:
            raise RuntimeError("Missing recovery arm " + arm)
        manifests = list(batch.glob("*/manifest.json"))
        hashes[arm] = {json.loads(p.read_text())["request"]["request_id"]:
                       json.loads(p.read_text())["validation"]["generated_token_sha256"] for p in manifests}
    if any(len(rows) != expected for rows in hashes.values()):
        raise RuntimeError("Recovery request membership is incomplete")
    reference = hashes["native_floor0"]
    differences = {arm: sorted(key for key in reference if rows.get(key) != reference[key])
                   for arm, rows in hashes.items()}
    report = {"status": "mismatch" if any(differences.values()) else "matched",
              "requests": expected, "token_hashes": hashes, "different_requests": differences}
    target = root / "qualification" / model / (stage_name + "_cross_arm_output.json")
    if target.exists():
        if json.loads(target.read_text()) != report:
            raise ValueError("Existing frozen output comparison differs")
    else:
        atomic_json(target, report)
    if report["status"] != "matched":
        raise RuntimeError("Corrected runtime still changes generated tokens: " + json.dumps(differences))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/m_reduction/stage0_mxfp4_alignment.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.study.resolve()
    os.environ["EXPERIMENT_RUNS_ROOT"] = str(root / "runs")
    plan = args.plan.resolve()
    protocol = json.loads(plan.read_text())
    model = "gpt_oss_mxfp4"
    dot64 = protocol.get("numeric_arm") == "mxfp4_alignment_dot64_v2"
    profile = root / "calibration/gpt_oss_mxfp4_aligned.json"
    stage = "Corrected GPT-OSS branch calibration"
    control = {"plan": str(plan), "plan_sha256": sha256(plan),
                "source_sha256": sha256(Path(__file__)), "runtime": os.environ["FREETOKEN_SOURCE_DIR"],
                "original_failure_preserved": True, "resume": args.resume}
    original = root / "recovery_controller.json"
    if args.resume:
        prior = json.loads(original.read_text())
        if prior["plan_sha256"] != control["plan_sha256"] or prior["runtime"] != control["runtime"]:
            raise ValueError("Resume requires the same numerical runtime and plan")
    elif original.exists():
        raise FileExistsError("Use --resume for an existing study")
    attempt = root / "controllers" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    atomic_json(attempt / "control.json", control)
    shutil.copy2(Path(__file__), attempt / "controller_source.py")
    shutil.copy2(ROOT / "experiments/pruning/m_reduction_guard.py", attempt / "guard_source.py")
    if not args.resume:
        atomic_json(original, control)
        shutil.copy2(Path(__file__), root / "recovery_controller_source.py")
    try:
        if args.resume:
            calibration = json.loads(profile.with_suffix(".control.json").read_text())
            if calibration["model"] != model or not calibration["mxfp4_alignment"] or calibration["mxfp4_dot64"] != dot64:
                raise ValueError("Resume calibration precision/model differs")
            event(root, "resume_started", controller=str(attempt), calibration_sha256=sha256(profile))
            wait_quiet(root, "Recovery resume: quiet host interval", state)
        else:
            state(root, "running", stage, "Fresh corrected CPU and PCIe profile; earlier bandwidth is diagnostic only.")
            launch(root, "m_reduction_calibrate", "--model", model, "--align-mxfp4",
                   *(["--dot64"] if dot64 else []), "--out", profile, no_swap=True)
        for arm in protocol["arms"]:
            stage = "Corrected GPT-OSS generation screen: " + arm
            if args.resume and completed_batch(root, model, arm, 2, plan, profile):
                event(root, "completed_screen_reused", arm=arm)
                continue
            state(root, "running", stage, "Two fixed pilot requests, observation pairs, replay and resource gates; not full qualification.")
            collect_with_retries(root, model, arm, 2, plan, profile, launch, state, stage)
        identities(root, model, protocol["arms"], 2, "screen", plan, profile)
        stage = "Corrected GPT-OSS fresh baseline"
        state(root, "running", stage, "Three fresh-process baseline repetitions on the corrected numerical arm.")
        import subprocess
        env = os.environ.copy()
        env["FREETOKEN_M_REDUCTION_MXFP4_ALIGN"] = "1"
        if dot64:
            env["FREETOKEN_M_REDUCTION_MXFP4_DOT64"] = "1"
        subprocess.run(["bash", str(ROOT / "scripts/baseline.sh"), "--profile", model,
                        "--model", str(ROOT / protocol["models"][model]["path"]), "--gpu", "0", "--repeats", "3",
                        "--calibration", str(profile)],
                       cwd=ROOT, env=env, check=True)
        for arm in protocol["arms"]:
            stage = "Corrected GPT-OSS full pilot: " + arm
            if args.resume and completed_batch(root, model, arm, 15, plan, profile):
                event(root, "completed_pilot_reused", arm=arm)
                continue
            state(root, "running", stage, "All 15 pilot observation pairs; full output identity gate remains required.")
            collect_with_retries(root, model, arm, 15, plan, profile, launch, state, stage)
        identities(root, model, protocol["arms"], 15, "pilot", plan, profile)
        launch(root, "m_reduction_plan_gate", "--study", root, "--out", attempt / "plan_readiness.json")
        state(root, "awaiting_model_qualification", "Corrected GPT-OSS pilot complete",
              "Output identity restored on 15 pilots; qualify Qwen/Gemma before primary collection.")
    except BaseException as exc:
        failure = {"stage": stage, "error": repr(exc)}
        atomic_json(attempt / "failure.json", failure)
        if not (root / "recovery_failure.json").exists():
            atomic_json(root / "recovery_failure.json", failure)
        state(root, "failed", stage, str(exc))
        raise


if __name__ == "__main__":
    main()
