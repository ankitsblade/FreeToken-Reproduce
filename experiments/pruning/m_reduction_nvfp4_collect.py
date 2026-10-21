"""NVFP4 precision collector; preserve the already qualified GPT collector identity.

This declares a separate collector dependency while retaining the P1/P2 contracts.
"""

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, engine_sources, python_command, runtime_pythonpath
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready
from experiments.pruning.collect_moe_routes import json_lines, render, request
from experiments.pruning.collect_moe_routes import validate_capture as validate_standard
from experiments.pruning.collect_gemma_routes import validate_capture as validate_gemma
from experiments.pruning.moe_pruning import atomic_json, sha256, snapshot
from experiments.pruning.m_reduction_replay import qualify
from experiments.pruning.pruning_resources import read_resources, monitor_resources, wait_for_quiet, summarize_resources


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Frozen control changed: " + str(path))
    else:
        atomic_json(path, value)


def prepare(study, plan):
    from transformers import AutoTokenizer

    rows = list(json_lines(ROOT / plan["dataset"]))
    if sha256(ROOT / plan["dataset"]) != plan["dataset_sha256"]:
        raise ValueError("Dataset differs from planned identity")
    if sha256(ROOT / plan["splits"]) != plan["splits_sha256"]:
        raise ValueError("Splits differ from planned identity")
    freeze(study / "protocol/plan.json", plan)
    for name, source in (("dataset.jsonl", plan["dataset"]), ("splits.json", plan["splits"])):
        target = study / "protocol" / name
        if target.exists():
            if sha256(target) != sha256(ROOT / source):
                raise ValueError("Frozen dataset control changed")
        else:
            shutil.copy2(ROOT / source, target)
    rendered = {}
    for key, model in plan["models"].items():
        local = ROOT / model["path"]
        revision = (local / ".cache/huggingface/download/config.json.metadata").read_text().splitlines()[0]
        if revision != model["revision"]:
            raise ValueError(key + ": checkpoint config revision differs")
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        rendered[key] = {row["request_id"]: render(tokenizer, row["prompt_text"], plan["chat_template_kwargs"])
                         for row in rows}
        if any(len(ids) > plan["maximum_prompt_tokens"] or
               len(ids) + plan["maximum_completion_tokens"] > model["max_seq_len"]
               for ids in rendered[key].values()):
            raise ValueError(key + ": rendered request exceeds fixed context budget")
    freeze(study / "protocol/rendered_inputs.json", rendered)
    print("Prepared all three model tokenizations and frozen dataset controls", flush=True)


def validate_scores(directory, model):
    routes = {(row["phase"], row["input_position"], row["layer_id"])
              for row in json_lines(directory / "routes.jsonl.partial")}
    observed = set()
    for row in json_lines(directory / "scores.jsonl.partial"):
        key = (row["phase"], row["input_position"], row["layer_id"])
        if key in observed or len(row["scores"]) != model["experts"]:
            raise ValueError("Duplicate/wrong geometry router scores")
        if row["semantics"] != "native_router_logits" or not row.get("dtype"):
            raise ValueError("Router score semantics/dtype missing")
        if not all(math.isfinite(value) for value in row["scores"]):
            raise ValueError("Nonfinite router scores")
        observed.add(key)
    if observed != routes:
        raise ValueError("Full native router scores are incomplete")
    layout = json.loads((directory / "cache_layout.json").read_text())
    expected = model["cache_slots"] * model["expert_bytes"]
    if layout["cache_slots"] != model["cache_slots"] or layout["expert_cache_bytes"] != expected:
        raise ValueError(f"Cache-byte geometry differs: {layout['expert_cache_bytes']} != {expected}")
    return {"full_score_events": len(observed), "expert_cache_bytes": expected}


def collect(args, plan):
    study = args.study.resolve()
    subprocess.run([*python_command(), "-m", "experiments.pruning.m_reduction_collect",
                    "--study", str(study), "--plan", str(args.plan.resolve()), "--prepare-only"],
                   cwd=ROOT, check=True)
    model = plan["models"][args.model]
    rows = list(json_lines(study / "protocol/dataset.jsonl"))
    rendered = json.loads((study / "protocol/rendered_inputs.json").read_text())[args.model]
    splits = json.loads((study / "protocol/splits.json").read_text())
    selected = [row for row in rows if row["pilot_flag"] == (args.stage == "pilot")]
    if args.stage == "repeat":
        selected = [row for row in rows if row["request_id"] in splits["repeat_request_ids"]]
    if args.limit:
        selected = selected[:args.limit]
    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    batch = study / "captures" / args.model / args.arm / args.stage / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    control = {**plan["arms"][args.arm], "top_k": model["top_k"]}
    if not args.calibration.is_file():
        raise FileNotFoundError("Fresh calibration profile is required")
    profile = json.loads(args.calibration.read_text())
    # Gemma's frozen plan entry omits "format"; its pinned checkpoint is NVFP4 and the
    # profile lookup below fails unless the fresh calibration measured that kernel.
    if model.get("format", "nvfp4") != "nvfp4":
        raise ValueError(args.model + ": NVFP4 collector requires an NVFP4 checkpoint")
    fmt = "nvfp4"
    entry = next(wl["kernels"][fmt] for wl in profile["workloads"].values() if fmt in wl["kernels"])
    if not entry.get("pcie_gather_overlap_gbs") or not entry.get("cpu_moe_overlap_gbs"):
        raise ValueError("Concurrent bandwidth calibration is incomplete")
    control["rho"] = entry["pcie_gather_overlap_gbs"] / entry["cpu_moe_overlap_gbs"]
    atomic_json(batch / "policy.json", control)
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    dependencies = {"plan": sha256(args.plan), "calibration": sha256(args.calibration),
                    "policy_source": sha256(ROOT / "experiments/pruning/m_reduction_policy.py"),
                    "collector_source": sha256(Path(__file__)),
                    "replay_source": sha256(ROOT / "experiments/pruning/m_reduction_replay.py"),
                    **{name: sha256(ROOT / path) for name, path in {
                        "resource_source": "experiments/pruning/pruning_resources.py",
                        "standard_validator": "experiments/pruning/collect_moe_routes.py",
                        "gemma_validator": "experiments/pruning/collect_gemma_routes.py",
                        "shared_helpers": "experiments/pruning/moe_pruning.py",
                        "decode_helpers": "experiments/baselines/bench_decode_moe.py",
                        "resource_control": plan["resource_policy"]}.items()},
                    "rendered_inputs": sha256(study / "protocol/rendered_inputs.json"),
                    "policy_control": sha256(batch / "policy.json")}
    qualification = study / "qualification" / args.model / (args.arm + ".json")
    if args.stage != "pilot":
        prior = json.loads(qualification.read_text())
        if prior["status"] != "qualified" or prior["inference_hashes"] != hashes or prior["dependencies"] != dependencies:
            raise ValueError("No matching complete pilot qualification")
    resource = json.loads((ROOT / plan["resource_policy"]).read_text())
    freeze(study / "protocol/resource_policy.json", resource)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FREETOKEN_MOE_", "FREETOKEN_HYBRID_", "FREETOKEN_M_REDUCTION_"))}
    arm = batch / "arm.json"
    env.update({"PYTHONPATH": runtime_pythonpath(), "FREETOKEN_BENCHBW_PATH": str(args.calibration.resolve()),
                "FREETOKEN_MOE_ROUTING_ARM": str(arm), "FREETOKEN_M_REDUCTION_FILL_FLOOR": str(control["fill_floor"]),
                "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                **resource["allocator_environment"]})
    if plan.get("numeric_arm") != "nvfp4_precision_v1" or args.model not in {"qwen_nvfp4", "gemma_nvfp4"}:
        raise ValueError("This collector requires the declared NVFP4 precision arm and Qwen/Gemma")
    env["FREETOKEN_M_REDUCTION_NVFP4_ALIGN"] = "1"
    if control["kind"] == "reuse_aware":
        env["FREETOKEN_M_REDUCTION_POLICY"] = str(batch / "policy.json")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPU is occupied: {active}")
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    command = [*python_command(), "-m", "freetoken.cli", "serve", "--model", str(ROOT / model["path"]),
               "--gpu", "0", "--host", "127.0.0.1", "--port", str(port), "--moe-strategy", "hybrid",
               "--max-running-requests", "1", "--max-seq-len-override", str(model["max_seq_len"]),
               "--max-prefill-length", "1024", "--memory-ratio", str(model["memory_ratio"]),
               "--cuda-graph-max-bs", "0", "--moe-hybrid-max-fetch", "-1",
               "--moe-cache-size", str(model["cache_slots"]), "--cache-type", "naive",
               "--expert-load", "serial", "--moe-cpu-threads", str(plan["cpu_threads"])]
    if args.model != "gpt_oss_mxfp4":
        command.append("--text-model-only")
    atomic_json(batch / "configuration.json", {"command": command, "arm": args.arm, "stage": args.stage,
                "model": model, "policy": control, "dependencies": dependencies, "inference_hashes": hashes,
                "collector_module": "m_reduction_nvfp4_collect",
                "source_snapshot": snapshot(batch / "source_snapshot"), "request_ids": [row["request_id"] for row in selected],
                "interpretation": "instrumented P1/P2 qualification, not serving performance"})
    stop = threading.Event()
    phase = {"name": "startup"}
    boundaries = {"startup_begin": read_resources(-1)}
    complete = []
    with (batch / "server.log").open("wb") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor = threading.Thread(target=monitor_resources, args=(proc, stop, batch / "resources.jsonl", phase))
        monitor.start()
        try:
            wait_ready(origin, proc, str(batch / "server.log"), 1800)
            model_id = get_json(origin + "/v1/models")["data"][0]["id"]
            atomic_json(batch / "startup_stats.json", get_json(origin + "/v1/stats"))
            warm = next(row for row in rows if row["request_id"] == "math_000")
            phase["name"] = "warmup"
            atomic_json(arm, {"armed": True, "capture_id": "warmup", "request_id": warm["request_id"],
                             "directory": str(batch / "warmup"), "prompt_tokens": len(rendered[warm["request_id"]]),
                             "observe": False, "require_pinned_banks": True})
            body = {"model": model_id, "messages": [{"role": "user", "content": warm["prompt_text"]}],
                    "max_tokens": plan["warmup_completion_tokens"], "stream": False,
                    **plan["sampling"], "chat_template_kwargs": plan["chat_template_kwargs"]}
            atomic_json(batch / "warmup_response.json", request(origin, body))
            atomic_json(arm, {"armed": False})
            phase["name"] = "settling"
            boundaries["ready"] = read_resources(proc.pid)
            boundaries["inference_begin"] = wait_for_quiet(proc, resource)
            phase["name"] = "inference"
            repetitions = 2 if args.stage == "repeat" else 1
            for repetition in range(repetitions):
                for row in selected:
                    directory = batch / f"{row['request_id']}_r{repetition}"
                    directory.mkdir()
                    metadata = {"armed": True, "capture_id": directory.name, "request_id": row["request_id"],
                                "directory": str(directory), "prompt_tokens": len(rendered[row["request_id"]]),
                                "require_pinned_banks": True, "full_scores": True}
                    body = {"model": model_id, "messages": [{"role": "user", "content": row["prompt_text"]}],
                            "max_tokens": plan["maximum_completion_tokens"], "stream": False,
                            **plan["sampling"], "chat_template_kwargs": plan["chat_template_kwargs"]}
                    try:
                        if args.stage == "pilot":
                            atomic_json(arm, {**metadata, "capture_id": directory.name + "_off",
                                             "directory": str(directory / "control"), "observe": False})
                            reference = request(origin, body)
                            atomic_json(directory / "control_response.json", reference)
                        atomic_json(arm, metadata)
                        response = request(origin, body)
                        atomic_json(directory / "response.json", response)
                        validator = validate_gemma if args.model == "gemma_nvfp4" else validate_standard
                        result = validator(directory, model, rendered[row["request_id"]], response)
                        if args.stage == "pilot" and (reference["choices"] != response["choices"] or reference["usage"] != response["usage"]):
                            raise ValueError("Observation changed response choices or token usage")
                        result.update(validate_scores(directory, model))
                        result["replay"] = qualify(directory, control)
                        checksums = {}
                        for name in ("routes", "batches", "cache", "scores"):
                            original = directory / (name + ".jsonl.partial")
                            target = directory / (name + ".jsonl.gz")
                            with original.open("rb") as source, gzip.open(target, "wb") as out:
                                shutil.copyfileobj(source, out)
                            checksums[target.name] = sha256(target)
                        for path in directory.glob("*.json"):
                            checksums[path.name] = sha256(path)
                        atomic_json(directory / "manifest.json", {"request": row, "model": args.model, "arm": args.arm,
                                    "stage": args.stage, "validation": result, "checksums": checksums})
                        complete.append(directory.name)
                        atomic_json(batch / "completed.json", complete)
                        print(f"Qualified capture {args.model}/{args.arm}/{directory.name}: "
                              f"{len(complete)}/{len(selected)*repetitions}, replay={result['replay']['status']}", flush=True)
                    except BaseException as exc:
                        atomic_json(directory / "failure.json", {"error": repr(exc)})
                        raise
                    finally:
                        atomic_json(arm, {"armed": False})
            boundaries["inference_end"] = read_resources(proc.pid)
            atomic_json(batch / "resource_boundaries.json", boundaries)
        except BaseException as exc:
            atomic_json(batch / "failure.json", {"error": repr(exc), "phase": phase["name"], "completed": complete,
                        "boundaries": boundaries})
            raise
        finally:
            phase["name"] = "shutdown"
            stop_server(proc)
            stop.set()
            monitor.join()
    summary = summarize_resources(list(json_lines(batch / "resources.jsonl")), boundaries, resource)
    atomic_json(batch / "resource_summary.json", summary)
    if summary["status"] != "passed":
        raise ValueError("Resource gate failed: " + "; ".join(summary["reasons"]))
    atomic_json(batch / "completion.json", {"status": "complete", "captures": len(complete), "resources": summary})
    if args.stage == "pilot" and len(complete) == 15:
        atomic_json(qualification, {"status": "qualified", "captures": 15, "batch": str(batch),
                    "inference_hashes": hashes, "dependencies": dependencies, "resources": summary})
    print(f"Completed {args.model}/{args.arm}/{args.stage}, {len(complete)} qualified captures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/m_reduction/stage0_nvfp4_precision.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", choices=("gpt_oss_mxfp4", "qwen_nvfp4", "gemma_nvfp4"))
    parser.add_argument("--arm", choices=("native_floor0", "native_floor1", "reuse_freq_floor0", "reuse_freq_floor1"))
    parser.add_argument("--stage", choices=("pilot", "primary", "repeat"), default="pilot")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--batch-id")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.prepare_only:
        prepare(args.study.resolve(), plan)
        return
    if not args.model or not args.arm or not args.calibration:
        parser.error("--model, --arm and --calibration are required for collection")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    collect(args, plan)


if __name__ == "__main__":
    main()
