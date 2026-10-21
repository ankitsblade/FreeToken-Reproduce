"""Qualify both repaired NVFP4 models before three-model primary collection."""

import argparse
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, engine_sources
from experiments.pruning.m_reduction_collect import freeze
from experiments.pruning.m_reduction_guard import completed_batch, collect_with_retries, event, wait_quiet
from experiments.pruning.m_reduction_plan_gate import coverage
from experiments.pruning.m_reduction_recovery import identities
from experiments.pruning.m_reduction_stage0 import launch, state
from experiments.pruning.moe_pruning import atomic_json, sha256


MODELS = ("qwen_nvfp4", "gemma_nvfp4")
COLLECTOR = "m_reduction_nvfp4_collect"


def diagnostic_gate(directory, hashes):
    records = {}
    for model in MODELS:
        probe = directory / "diagnostics" / (model + "-precision")
        config = json.loads((probe / "configuration.json").read_text())
        report = json.loads((probe / "comparison.json").read_text())
        checks = [report["all_routes"], *report["per_route"],
                  report["cpu_group_vs_rounded_routes"], report["gpu_group_vs_rounded_routes"]]
        if (not config.get("nvfp4_alignment") or config["runtime_hashes"] != hashes
                or config["layer_id"] != 0 or config["maximum_completion_tokens"] != 2
                or config["request"]["request_id"] != "math_000" or len(report["per_route"]) != 8
                or not all(row["equal"] and row["different_elements"] == 0 for row in checks)):
            raise ValueError(model + ": missing same-runtime precision diagnostic prerequisite")
        records[model] = {"directory": str(probe), "configuration_sha256": sha256(probe / "configuration.json"),
                          "comparison_sha256": sha256(probe / "comparison.json"),
                          "fixed_state_sha256": sha256(probe / "fixed_state.pt"),
                          "scope": "One first-decode hidden state; full pilot identity still required"}
    return records


def baseline_complete(root, model, profile):
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    for path in sorted(root.glob("runs/baselines/*/run.json"), reverse=True):
        run = json.loads(path.read_text())
        metadata = path.parent / "baseline.json"
        if run.get("status") != "completed" or not metadata.is_file():
            continue
        record = json.loads(metadata.read_text())
        repeats = list(path.parent.glob("repeat_*.jsonl"))
        if (record.get("profile") == model and record.get("status") == "completed"
                and record.get("repeats") == 3 and len(repeats) == 3
                and all(row.stat().st_size for row in repeats)
                and record.get("calibration_sha256") == sha256(profile)
                and record.get("numeric_controls", {}).get("nvfp4_alignment")):
            # The run recorder stores source hashes; keep an explicit baseline binding too.
            binding = root / "baseline_dependencies" / (path.parent.name + ".json")
            if binding.is_file() and json.loads(binding.read_text())["inference_hashes"] == hashes:
                return path.parent
    return None


def fresh_baseline(root, model, profile, protocol):
    env = os.environ.copy()
    for key in ("FREETOKEN_M_REDUCTION_MXFP4_ALIGN", "FREETOKEN_M_REDUCTION_MXFP4_DOT64",
                "FREETOKEN_CPU_MOE_NO_VNNI"):
        env.pop(key, None)
    env["FREETOKEN_M_REDUCTION_NVFP4_ALIGN"] = "1"
    before = set(root.glob("runs/baselines/*/run.json"))
    subprocess.run(["bash", str(ROOT / "scripts/baseline.sh"), "--profile", model,
                    "--model", str(ROOT / protocol["models"][model]["path"]), "--gpu", "0",
                    "--repeats", "3", "--calibration", str(profile)], cwd=ROOT, env=env, check=True)
    created = set(root.glob("runs/baselines/*/run.json")) - before
    if len(created) != 1:
        raise ValueError("Expected one recorded three-repetition baseline command")
    path = created.pop()
    atomic_json(root / "baseline_dependencies" / (path.parent.name + ".json"), {
        "run": str(path.parent), "model": model, "calibration_sha256": sha256(profile),
        "inference_hashes": {name: sha256(source) for name, source in engine_sources().items()},
        "baseline_source_sha256": sha256(ROOT / "experiments/baselines/baseline.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/m_reduction/stage0_nvfp4_precision.json")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--gpt-study", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.study.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["EXPERIMENT_RUNS_ROOT"] = str(root / "runs")
    plan = args.plan.resolve()
    protocol = json.loads(plan.read_text())
    policy_path = ROOT / "configs/m_reduction/coverage_policy.json"
    policy = json.loads(policy_path.read_text())
    if protocol.get("numeric_arm") != "nvfp4_precision_v1" or set(protocol["models"]) != set(policy["required_models"]):
        raise ValueError("NVFP4 repair plan must retain all three required models")
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    diagnostics = diagnostic_gate(args.diagnostics.resolve(), hashes)
    gpt = args.gpt_study.resolve()
    if not coverage(gpt, "gpt_oss_mxfp4", policy)["pilot_ready"]:
        raise ValueError("Qualified GPT-OSS pilot dependency is incomplete")
    control = {"plan": str(plan), "plan_sha256": sha256(plan), "runtime": os.environ["FREETOKEN_SOURCE_DIR"],
               "inference_hashes": hashes, "models": list(MODELS), "diagnostics": diagnostics,
               "gpt_qualified_study": str(gpt), "collector_module": COLLECTOR,
               "source_sha256": sha256(Path(__file__)), "coverage_policy_sha256": sha256(policy_path)}
    prior = root / "recovery_controller.json"
    if args.resume:
        if json.loads(prior.read_text()) != control:
            raise ValueError("Resume requires identical runtime, plan, controller and diagnostic dependencies")
    elif prior.exists():
        raise FileExistsError("Use --resume for an existing correction study")
    else:
        atomic_json(prior, control)
    attempt = root / "controllers" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    attempt.mkdir(parents=True)
    atomic_json(attempt / "control.json", {**control, "resume": args.resume})
    for module in (Path(__file__), ROOT / "experiments/pruning/m_reduction_guard.py",
                   ROOT / "experiments/pruning/m_reduction_calibrate.py", ROOT / ("experiments/pruning/" + COLLECTOR + ".py")):
        shutil.copy2(module, attempt / module.name)
    stage = "NVFP4 precision prerequisites"
    failures = {}
    try:
        launch(root, COLLECTOR, "--study", root, "--plan", plan, "--prepare-only")
        freeze(root / "protocol/resource_policy.json", json.loads((ROOT / protocol["resource_policy"]).read_text()))
        launch(root, "m_reduction_plan_gate", "--study", root,
               "--model-study", "gpt_oss_mxfp4=" + str(gpt), "--out", attempt / "preflight_plan_audit.json")
        audit = root / "checkpoint_audit.json"
        if not audit.exists():
            stage = "NVFP4 correction: three-model checkpoint audit"
            state(root, "running", stage, "Verify pinned shards and metadata before model qualification.")
            launch(root, "m_reduction_audit", "--study", root)
        elif json.loads(audit.read_text()).get("status") != "verified":
            raise ValueError("Existing checkpoint audit did not finish; preserve it and use a fresh study")
        for model in MODELS:
            try:
                profile = root / "calibration" / (model + "_precision.json")
                stage = model + ": fresh NVFP4 precision calibration"
                if profile.is_file():
                    calibration = json.loads(profile.with_suffix(".control.json").read_text())
                    if not args.resume or calibration.get("model") != model or not calibration.get("nvfp4_alignment"):
                        raise ValueError("Existing calibration must match an explicit same-arm resume")
                else:
                    state(root, "running", stage, "Measure actual corrected CPU/PCIe branches; borrowed diagnostic profiles are excluded.")
                    launch(root, "m_reduction_calibrate", "--model", model, "--align-nvfp4", "--out", profile, no_swap=True)
                for arm in protocol["arms"]:
                    stage = model + ": precision generation screen, " + arm
                    if args.resume and completed_batch(root, model, arm, 2, plan, profile):
                        event(root, "completed_screen_reused", model=model, arm=arm)
                        continue
                    collect_with_retries(root, model, arm, 2, plan, profile, launch, state, stage, collector_module=COLLECTOR)
                identities(root, model, protocol["arms"], 2, "screen", plan, profile)
                stage = model + ": three fresh precision baselines"
                if not (args.resume and baseline_complete(root, model, profile)):
                    wait_quiet(root, stage, state)
                    state(root, "running", stage, "Three fresh servers with this arm's actual bandwidth profile.")
                    fresh_baseline(root, model, profile, protocol)
                for arm in protocol["arms"]:
                    stage = model + ": full precision pilot, " + arm
                    if args.resume and completed_batch(root, model, arm, 15, plan, profile):
                        event(root, "completed_pilot_reused", model=model, arm=arm)
                        continue
                    collect_with_retries(root, model, arm, 15, plan, profile, launch, state, stage, collector_module=COLLECTOR)
                identities(root, model, protocol["arms"], 15, "pilot", plan, profile)
                event(root, "model_pilots_qualified", model=model, pairs=60)
            except Exception as exc:
                failures[model] = {"stage": stage, "error": repr(exc)}
                atomic_json(attempt / (model + "_failure.json"), failures[model])
                state(root, "failed", stage, str(exc) + "; retained attempt; dependent model work stopped.")
                event(root, "model_qualification_failed", model=model, stage=stage, error=repr(exc))
                # The other format-family model remains an independent qualification unit.
        stage = "NVFP4 precision: updated three-model qualification audit"
        launch(root, "m_reduction_plan_gate", "--study", root,
               "--model-study", "gpt_oss_mxfp4=" + str(gpt), "--out", attempt / "final_plan_audit.json")
        if failures:
            raise RuntimeError("NVFP4 qualification unfinished: " + json.dumps(failures))
        atomic_json(root / "recovery_completion.json", {"status": "nvfp4_pilots_complete",
                    "models": list(MODELS), "gpt_qualified_study": str(gpt), "primary_collected": False,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "limitations": "Primary/repeat collection and full P2 support remain required."})
        state(root, "awaiting_primary_and_p2", "All three models' corrected pilots qualified",
              "Require a fresh three-model primary transition audit before primary/repeat captures; full P2 support gates later stages.")
    except BaseException as exc:
        atomic_json(attempt / "failure.json", {"stage": stage, "error": repr(exc), "model_failures": failures})
        state(root, "failed", stage, str(exc) + "; original evidence retained.")
        raise


if __name__ == "__main__":
    main()
