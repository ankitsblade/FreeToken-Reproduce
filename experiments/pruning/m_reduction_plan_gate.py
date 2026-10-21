"""Audit current plan controls and refuse progression with missing model evidence."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT
from experiments.pruning.moe_pruning import atomic_json, sha256


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def flag(command, name):
    return command[command.index(name) + 1] if name in command else None


def coverage(study, model, policy):
    canonical = read(ROOT / "configs/m_reduction/stage0.json")
    requests = [json.loads(line) for line in (ROOT / canonical["dataset"]).read_text().splitlines()]
    pilot_ids = {row["request_id"] for row in requests if row["pilot_flag"]}
    primary_ids = {row["request_id"] for row in requests if not row["pilot_flag"]}
    repeat_ids = set(read(ROOT / canonical["splits"])["repeat_request_ids"])
    rows = {}
    qualifications = {}
    for arm in policy["required_pilot_arms"]:
        record = read(study / "qualification" / model / (arm + ".json"))
        qualifications[arm] = record
        batch = Path(record["batch"]) if record.get("batch") else None
        completion = read(batch / "completion.json") if batch else {}
        members = {read(path).get("request", {}).get("request_id") for path in batch.glob("*/manifest.json")} if batch else set()
        rows[arm] = (record.get("status") == "qualified" and record.get("captures") == policy["pilot_requests_per_arm"]
                     and completion.get("status") == "complete" and completion.get("resources", {}).get("status") == "passed"
                     and members == pilot_ids)
    identity = read(study / "qualification" / model / "pilot_cross_arm_output.json")
    if not identity:
        identity = read(study / "qualification" / model / "cross_arm_output.json")
    pilot_ready = (all(rows.values()) and identity.get("status") == "matched"
                   and identity.get("requests") == policy["pilot_requests_per_arm"]
                   and set(identity.get("token_hashes", {})) == set(policy["required_pilot_arms"]))
    if pilot_ready:
        reference = identity["token_hashes"]["native_floor0"]
        pilot_ready = (set(reference) == pilot_ids and all(tokens == reference for tokens in identity["token_hashes"].values())
                       and all(q["inference_hashes"] == qualifications["native_floor0"]["inference_hashes"] for q in qualifications.values()))
    counts = {"primary": 0, "repeat": 0}
    for subset in counts:
        members = set()
        for batch in (study / "captures" / model / "native_floor1" / subset).glob("*"):
            completion = read(batch / "completion.json")
            config = read(batch / "configuration.json")
            reference = qualifications["native_floor1"]
            if (not pilot_ready or completion.get("status") != "complete" or completion.get("resources", {}).get("status") != "passed"
                    or config.get("inference_hashes") != reference.get("inference_hashes")
                    or config.get("dependencies") != reference.get("dependencies")):
                continue
            for path in batch.glob("*/manifest.json"):
                manifest = read(path)
                validation = manifest.get("validation", {})
                if validation.get("status") != "valid" or validation.get("replay", {}).get("status") != "matched":
                    continue
                request_id = manifest["request"]["request_id"]
                repetition = path.parent.name.rsplit("_r", 1)[-1]
                if (subset == "primary" and request_id in primary_ids and repetition == "0"
                        or subset == "repeat" and request_id in repeat_ids and repetition in {"0", "1"}):
                    members.add((request_id, repetition))
        counts[subset] = len(members)
    support = read(study / "qualification" / model / "p2_support.json")
    p2_ready = (support.get("status") == "qualified"
                and set(policy["p2_support_required"]).issubset(support.get("qualified_features", []))
                and bool(support.get("evidence"))
                # P2 support runs on its own runtime; it must reproduce captures made by the qualified one.
                and support.get("reproduces_inference_hashes") == qualifications["native_floor1"].get("inference_hashes")
                and bool(support.get("inference_hashes")))
    return {"study": str(study), "pilot_arms_qualified": rows, "cross_arm_identity": identity.get("status", "pending"),
            "pilot_ready": pilot_ready, "resource_passed_captures": counts, "p2_support_qualified": p2_ready,
            "stage0_ready": pilot_ready and counts["primary"] >= policy["primary_requests_per_model"]
            and counts["repeat"] >= policy["repeat_captures_per_model"] and p2_ready}


def deferral_record(path, policy):
    if path is None:
        return None
    record = read(path)
    deferred = set(record.get("deferred_models", []))
    if (record.get("authorized_by") != "user" or record.get("scope") != "collection_order_only"
            or not record.get("deferred_model_remains_required") or not record.get("next_stage_requires_all_models")
            or not deferred or not deferred < set(policy["required_models"])):
        raise SystemExit("Deferral must be a user-authorized collection-order record for a strict model subset")
    return {"path": str(path), "sha256": sha256(path), "deferred_models": sorted(deferred)}


def audit(study, policy_path, model_studies, deferral=None):
    policy = read(policy_path)
    plan = read(study / "protocol/plan.json")
    expected = read(ROOT / "configs/m_reduction/stage0.json")
    mismatches = []
    if set(plan.get("models", {})) != set(policy["required_models"]):
        mismatches.append("Controlled plan must contain all three required models")
    for model in policy["required_models"]:
        if plan.get("models", {}).get(model) != expected["models"][model]:
            mismatches.append(model + ": model pins/geometry/budget differ from declared Stage 0 controls")
    for key in ("dataset", "splits", "dataset_sha256", "splits_sha256", "maximum_prompt_tokens",
                "maximum_completion_tokens", "warmup_completion_tokens", "cpu_threads", "sampling",
                "chat_template_kwargs", "arms", "resource_policy"):
        if plan.get(key) != expected[key]:
            mismatches.append("Controlled parameter differs: " + key)
    for file, key in (("dataset.jsonl", "dataset_sha256"), ("splits.json", "splits_sha256")):
        path = study / "protocol" / file
        if not path.is_file() or sha256(path) != expected[key]:
            mismatches.append("Frozen dataset/split differs: " + file)
    if read(study / "protocol/resource_policy.json") != read(ROOT / expected["resource_policy"]):
        mismatches.append("Frozen resource gates differ")
    configs = sorted(study.glob("captures/*/*/*/*/configuration.json"), key=lambda p: p.stat().st_mtime)
    active = None
    if configs:
        path = configs[-1]
        config = read(path)
        model = path.parents[3].name
        command = config["command"]
        required = {"--gpu": "0", "--moe-strategy": "hybrid", "--max-running-requests": "1",
                    "--cuda-graph-max-bs": "0", "--moe-hybrid-max-fetch": "-1", "--cache-type": "naive",
                    "--expert-load": "serial", "--moe-cpu-threads": "5", "--max-prefill-length": "1024"}
        required.update({"--moe-cache-size": str(expected["models"][model]["cache_slots"]),
                         "--max-seq-len-override": str(expected["models"][model]["max_seq_len"]),
                         "--memory-ratio": str(expected["models"][model]["memory_ratio"])})
        for name, value in required.items():
            if flag(command, name) != value:
                mismatches.append("Active serving parameter differs: " + name)
        if config["model"] != expected["models"][model]:
            mismatches.append("Active batch model controls differ")
        if model != "gpt_oss_mxfp4" and "--text-model-only" not in command:
            mismatches.append("NVFP4 batch must be text-only")
        active = {"model": model, "arm": config["arm"], "stage": config["stage"], "configuration": str(path),
                  "configuration_sha256": sha256(path), "numeric_arm": plan.get("numeric_arm", "native")}
    matrix = {model: coverage(model_studies.get(model, study), model, policy) for model in policy["required_models"]}
    all_pilots = all(row["pilot_ready"] for row in matrix.values())
    all_stage0 = all(row["stage0_ready"] for row in matrix.values())
    # A deferral only reorders primary collection; completion and later stages still need every model.
    deferred = set(deferral["deferred_models"]) if deferral else set()
    scoped = [model for model in policy["required_models"] if model not in deferred]
    primary_models = scoped if all(matrix[model]["pilot_ready"] for model in scoped) else []
    if all_pilots:
        primary_models = list(policy["required_models"])
    return {"utc": datetime.now(timezone.utc).isoformat(), "canonical_plan": str(ROOT / policy["source_plan"]),
            "canonical_plan_sha256": sha256(ROOT / policy["source_plan"]),
            "coverage_policy_sha256": sha256(policy_path), "auditor_source_sha256": sha256(Path(__file__)),
            "controls_status": "consistent" if not mismatches else "mismatch", "mismatches": mismatches,
            "active_batch": active, "required_experiments": policy["required_experiments"],
            "model_coverage": matrix, "all_three_models_pilot_qualified": all_pilots,
            "all_three_models_stage0_complete": all_stage0,
            "deferral": deferral, "primary_collection_models": [] if mismatches else primary_models,
            "primary_collection_allowed": not mismatches and bool(primary_models),
            "next_research_stage_allowed": not mismatches and all_stage0,
            "rule": "A model-specific repair does not complete P1/P2. Missing model evidence blocks advancement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/m_reduction/coverage_policy.json")
    parser.add_argument("--model-study", action="append", default=[], help="Explicit MODEL_ID=STUDY binding")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-ready", choices=("primary", "next-stage"))
    parser.add_argument("--deferral", type=Path, help="User-authorized collection-order deferral record")
    parser.add_argument("--require-model", help="With --require-ready primary, require this model in the allowed scope")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Plan audit must use a fresh output file")
    bindings = {}
    allowed = set(read(args.policy)["required_models"])
    for entry in args.model_study:
        model, path = entry.split("=", 1)
        if model not in allowed or model in bindings:
            parser.error("Unknown or repeated model binding")
        bindings[model] = Path(path).resolve()
    deferral = deferral_record(args.deferral.resolve() if args.deferral else None, read(args.policy))
    report = audit(args.study.resolve(), args.policy.resolve(), bindings, deferral)
    atomic_json(args.out.resolve(), report)
    print(json.dumps({key: report[key] for key in ("controls_status", "active_batch",
                     "all_three_models_pilot_qualified", "all_three_models_stage0_complete",
                     "primary_collection_models", "primary_collection_allowed",
                     "next_research_stage_allowed")}, indent=2), flush=True)
    if report["mismatches"]:
        raise SystemExit("Plan compliance mismatch: " + "; ".join(report["mismatches"]))
    if args.require_ready and not report["primary_collection_allowed" if args.require_ready == "primary" else "next_research_stage_allowed"]:
        raise SystemExit("Required three-model evidence is incomplete; transition refused. See " + str(args.out))
    if args.require_ready == "primary" and args.require_model and args.require_model not in report["primary_collection_models"]:
        raise SystemExit(args.require_model + " is not in the allowed primary collection scope. See " + str(args.out))


if __name__ == "__main__":
    main()
