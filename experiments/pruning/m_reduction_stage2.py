"""Serial Stage 2 GPU queue for the m-reduction study (plan sections 8-10).

Order, per model GPT-OSS -> Gemma -> Qwen within each unit, one GPU job at a time:
1. Live exact-arm confirmation (m_reduction_live --controller): fidelity, five paired
   timing repetitions, analysis.
2. P7 candidate teacher forcing on all 90 D0 histories, arm 1 and the R75 reference,
   each request under its own outer-fold mask, KL against the P2 baseline logits.
3. D1 quality: four shards (tasks: baseline + arm 1; sequences: KL for arm 1 and R75).
4. D2 AIME: baseline + arm 1 free-running, arm 1 KL on the baseline trajectory.
Every unit is skipped when an accepted result exists (resume), and a resource-rejected
attempt is retried at most three times after a quiet host interval.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT
from experiments.pruning.m_reduction_guard import wait_quiet
from experiments.pruning.moe_pruning import atomic_json

STUDIES = {"gpt_oss_mxfp4": "results/m_reduction/recovery_dot64_20261013T131424Z",
           "gemma_nvfp4": "results/m_reduction/recovery_nvfp4_precision_20261013T161122Z",
           "qwen_nvfp4": "results/m_reduction/recovery_nvfp4_precision_20261013T161122Z"}
CALIBRATION = {"gpt_oss_mxfp4": "calibration/gpt_oss_mxfp4_aligned.json",
               "gemma_nvfp4": "calibration/gemma_nvfp4_precision.json",
               "qwen_nvfp4": "calibration/qwen_nvfp4_precision.json"}
P2 = ROOT / "results/m_reduction/p2_support_20261014T123345Z"
OFFLINE = ROOT / "results/m_reduction/offline_stage1_20261014T200452Z"
RETRYABLE = ("system swapping during inference exceeds limit", "startup or inference host-memory headroom below minimum",
             "sustained full memory pressure during inference", "did not settle")


def state(root, status, stage, detail):
    atomic_json(root / "stage2_state.json", {"utc": datetime.now(timezone.utc).isoformat(), "status": status,
                                             "stage": stage, "detail": detail})
    print(f"[{status}] {stage}: {detail}", flush=True)


def run(root, module, *arguments):
    command = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemorySwapMax=0", "bash",
               str(ROOT / "scripts/run.sh"), "pruning", module, *map(str, arguments)]
    print("Launching: " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def newest(directory):
    runs = sorted(p for p in directory.glob("*") if p.is_dir()) if directory.is_dir() else []
    return runs[-1] if runs else None


def complete(batch):
    if batch is None or not (batch / "completion.json").is_file():
        return False
    done = json.loads((batch / "completion.json").read_text())
    return done.get("status") == "complete" and not done.get("limit")


def retryable(batch):
    text = ""
    for name in ("failure.json", "resource_summary.json", "completion.json"):
        if batch is not None and (batch / name).is_file():
            text += (batch / name).read_text()
    return any(reason in text for reason in RETRYABLE)


def unit(root, stage, directory, module, *arguments):
    if complete(newest(directory)):
        state(root, "skipped", stage, "accepted result exists: " + str(newest(directory)))
        return
    for attempt in range(1, 4):
        if attempt > 1:
            wait_quiet(root, stage, state)
        state(root, "running", stage, f"attempt {attempt}/3")
        code = run(root, module, *arguments)
        batch = newest(directory)
        if code == 0 and complete(batch):
            state(root, "completed", stage, str(batch))
            return
        if not retryable(batch):
            raise SystemExit(f"{stage}: failed (code {code}); see {batch}")
    raise SystemExit(f"{stage}: three resource-rejected attempts")


def p2_batch(model):
    support = json.loads((ROOT / STUDIES[model] / "qualification" / model / "p2_support.json").read_text())
    return support["evidence"]["baseline_teacher_batch"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gpt_oss_mxfp4", "gemma_nvfp4", "qwen_nvfp4"])
    parser.add_argument("--units", nargs="+", default=["live", "candidate", "d1", "d2"])
    args = parser.parse_args()
    root = args.study.resolve()
    for name in args.units:
        for model in args.models:
            source = ROOT / STUDIES[model]
            calibration = source / CALIBRATION[model]
            common = ["--study", root, "--source-study", source, "--model", model]
            if name == "live":
                stage = f"{model}: live exact arms"
                state(root, "running", stage, "fidelity, five paired timing repetitions and analysis")
                code = run(root, "m_reduction_live", "--controller", *common, "--calibration", calibration,
                           "--offline", OFFLINE)
                if code:
                    raise SystemExit(stage + f": controller failed (code {code})")
                state(root, "completed", stage, "see analysis/" + model)
            elif name == "candidate":
                for arm in ("arm1", "r75"):
                    unit(root, f"{model}: P7 {arm} candidate teacher forcing",
                         root / "captures" / model / f"teacher_candidate_{arm}" / "all", "m_reduction_teacher",
                         *common, "--calibration", calibration, "--mode", "candidate", "--candidate-arm", arm,
                         "--artefacts", OFFLINE / model / "p7" / f"{model}_artefacts.json",
                         "--reference-batch", p2_batch(model), "--subset", "all")
            elif name == "arm2":
                # Separate live2 runtime (merge/alias hooks): FREETOKEN_SOURCE_DIR must point at it.
                out = root / "arm2" / model
                checkpoint = ROOT / json.loads((source / "protocol/plan.json").read_text())["models"][model]["path"]
                if not (out / "identity_check.json").is_file():
                    out.mkdir(parents=True, exist_ok=True)
                    if run(root, "m_reduction_merge", "--identity-check", "--model", model, "--checkpoint", checkpoint,
                           "--layers", 0, "--out", out / "identity_check.json"):
                        raise SystemExit(model + ": requantization identity check failed")
                controls = out / "controls"
                if not controls.is_dir():
                    if run(root, "m_reduction_merge", "--build", "--model", model, "--source-study", source,
                           "--artefacts", OFFLINE / model / "p7" / f"{model}_artefacts.json", "--out", controls):
                        raise SystemExit(model + ": merge control build failed")
                reference = p2_batch(model)
                unit(root, f"{model}: requantization-only control teacher forcing",
                     root / "captures" / model / "teacher_candidate_requant" / "all", "m_reduction_teacher",
                     *common, "--calibration", calibration, "--mode", "candidate", "--candidate-arm", "requant",
                     "--merge-control", controls / "requant_only.json", "--reference-batch", reference, "--subset", "all")
                for fold in range(5):
                    unit(root, f"{model}: P7 arm 2 fold {fold} teacher forcing",
                         root / "captures" / model / f"teacher_candidate_arm2_fold{fold}" / "all", "m_reduction_teacher",
                         *common, "--calibration", calibration, "--mode", "candidate", "--candidate-arm", "arm2",
                         "--merge-control", controls / f"arm2_fold{fold}.json", "--fold", fold,
                         "--reference-batch", reference, "--subset", "all")
            elif name == "p7":
                if run(root, "m_reduction_p7_live", "--study", root, "--model", model):
                    raise SystemExit(model + ": P7 analysis failed")
            elif name in ("d1", "d2"):
                shards = 4 if name == "d1" else 1
                kl = "arm1,r75" if name == "d1" else "arm1"
                for shard in range(shards):
                    unit(root, f"{model}: {name.upper()} shard {shard}/{shards}",
                         root / "quality" / model / name / f"shard{shard:02d}of{shards:02d}", "m_reduction_quality",
                         *common, "--calibration", calibration, "--deploy", root / "p7_deploy" / f"{model}.json",
                         "--dataset", name, "--shard", shard, "--shards", shards, "--kl-arms", kl)
                code = run(root, "m_reduction_quality", "--analyze", *common, "--dataset", name)
                if code:
                    raise SystemExit(f"{model}: {name} analysis failed")
    state(root, "finished", "Stage 2 queue", "all requested units completed")


if __name__ == "__main__":
    main()
