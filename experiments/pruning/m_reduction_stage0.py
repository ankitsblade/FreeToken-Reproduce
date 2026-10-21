"""Run the serial P1/P2 admission/trace prerequisite with recorded child commands."""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT
from experiments.pruning.moe_pruning import atomic_json, sha256


def state(root, status, stage, message):
    atomic_json(root / "progress.json", {"experiment": "m_reduction_v3", "status": status,
                "stage": stage, "message": message, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(stage + ": " + message, flush=True)


def baseline_gate(root):
    while True:
        progress = json.loads((root / "progress.json").read_text())
        if progress["status"] == "failed":
            raise RuntimeError("Baseline queue failed; dependent collection stopped")
        if progress["status"] == "awaiting_qualification":
            break
        time.sleep(5)
    records = {}
    rejected = []
    for path in sorted(root.glob("runs/baselines/*/run.json")):
        run = json.loads(path.read_text())
        if run["status"] != "completed":
            rejected.append(str(path.parent))
            continue
        profile = json.loads((path.parent / "baseline.json").read_text())
        if profile["status"] != "completed":
            raise RuntimeError("Baseline has an incomplete run")
        rows = list(path.parent.glob("repeat_*.jsonl"))
        if len(rows) != 3 or any(not row.stat().st_size for row in rows):
            raise RuntimeError("Baseline repetition records are incomplete")
        records[profile["profile"]] = {"profile": profile["profile"], "run": str(path.parent), "repetitions": len(rows)}
    if set(records) != {"gpt_oss_mxfp4", "qwen_nvfp4", "gemma_nvfp4"}:
        raise RuntimeError("Three-model baseline prerequisite is incomplete")
    atomic_json(root / "baseline_command_completion.json", {"status": "commands_complete",
                "records": list(records.values()), "rejected_runs": rejected,
                "interpretation": "Command completion, not pressure qualification or P1 gate"})


def launch(root, module, *arguments, no_swap=False):
    command = ["bash", str(ROOT / "scripts/run.sh"), "pruning", module, *map(str, arguments)]
    if no_swap:
        command = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemorySwapMax=0", *command]
    print("Launching: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--wait-baselines", action="store_true")
    parser.add_argument("--pilots-only", action="store_true")
    args = parser.parse_args()
    root = args.study.resolve()
    os.environ["EXPERIMENT_RUNS_ROOT"] = str(root / "runs")
    stage = "Stage 0: baseline prerequisite"
    try:
        if args.wait_baselines:
            baseline_gate(root)
        plan_path = ROOT / "configs/m_reduction/stage0.json"
        plan = json.loads(plan_path.read_text())
        coverage_policy = json.loads((ROOT / "configs/m_reduction/coverage_policy.json").read_text())
        if set(plan["models"]) != set(coverage_policy["required_models"]) or any(not row["required"] for row in plan["models"].values()):
            raise ValueError("The updated plan requires all three MoE models")
        atomic_json(root / "stage0_controller.json", {"status": "started", "plan": str(plan_path),
                    "plan_sha256": sha256(plan_path), "runtime_source": os.environ.get("FREETOKEN_SOURCE_DIR"),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "pilots_only": args.pilots_only})
        stage = "Stage 0: full checkpoint identity audit"
        state(root, "running", stage, "Hashing all pinned shards and verifying tensor indices; no inference active.")
        launch(root, "m_reduction_audit", "--study", root)
        stage = "Stage 0: freeze dataset and tokenization"
        state(root, "running", stage, "Freeze the 15 pilots, 90 primary IDs and existing five-fold manifest.")
        launch(root, "m_reduction_collect", "--study", root, "--prepare-only")
        for model in plan["models"]:
            profile = root / "calibration" / (model + ".json")
            stage = f"Stage 0: {model} fresh branch calibration"
            state(root, "running", stage, "Measure standalone/concurrent native CPU and PCIe branches with five workers.")
            launch(root, "m_reduction_calibrate", "--model", model, "--out", profile, no_swap=True)
        for model in plan["models"]:
            for arm in plan["arms"]:
                stage = f"P1/P2: {model}, {arm}, 15 pilot pairs"
                state(root, "running", stage, "Observation-off/on pairs; full scores, exact replay and unchanged resource gates.")
                launch(root, "m_reduction_collect", "--study", root, "--model", model,
                       "--arm", arm, "--stage", "pilot", "--calibration", root / "calibration" / (model + ".json"),
                       no_swap=True)
            identities = {}
            for arm in plan["arms"]:
                manifests = list((root / "captures" / model / arm / "pilot").glob("*/*/manifest.json"))
                identities[arm] = {json.loads(path.read_text())["request"]["request_id"]:
                                   json.loads(path.read_text())["validation"]["generated_token_sha256"]
                                   for path in manifests}
            if any(len(rows) != 15 for rows in identities.values()):
                raise RuntimeError("Pilot arm task membership is incomplete")
            baseline = identities["native_floor0"]
            if any(rows != baseline for rows in identities.values()):
                atomic_json(root / "qualification" / model / "cross_arm_output.json",
                            {"status": "mismatch", "token_hashes": identities})
                raise RuntimeError(model + ": q policy changed generated token identities")
            atomic_json(root / "qualification" / model / "cross_arm_output.json",
                        {"status": "matched", "requests": 15, "token_hashes": identities})
        if not args.pilots_only:
            stage = "Stage 0: three-model pilot coverage gate"
            launch(root, "m_reduction_plan_gate", "--study", root,
                   "--out", root / "plan_compliance/primary_transition.json", "--require-ready", "primary")
            for model in plan["models"]:
                for subset in ("primary", "repeat"):
                    stage = f"P1/P2: {model} corrected-floor {subset} captures"
                    state(root, "running", stage, "Every capture records all native scores and requires event/final-state replay equality.")
                    launch(root, "m_reduction_collect", "--study", root, "--model", model,
                           "--arm", "native_floor1", "--stage", subset,
                           "--calibration", root / "calibration" / (model + ".json"), no_swap=True)
        atomic_json(root / "stage0_completion.json", {"status": "admission_and_trace_units_complete",
                    "finished_utc": datetime.now(timezone.utc).isoformat(), "pilots_only": args.pilots_only,
                    "limitations": "P2 sequential teacher forcing, norms/probes and P3-P9 remain to implement/qualify."})
        launch(root, "m_reduction_plan_gate", "--study", root, "--out", root / "plan_compliance/stage0_readiness.json")
        state(root, "awaiting_p2_support", "Stage 0 admission/trace units complete",
              "P1/P2 remain unfinished until all three models and full P2 support qualify; later stages require the plan gate.")
    except BaseException as exc:
        state(root, "failed", stage, repr(exc) + "; preserved attempt; dependent stages stopped.")
        atomic_json(root / "stage0_failure.json", {"stage": stage, "error": repr(exc),
                    "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
