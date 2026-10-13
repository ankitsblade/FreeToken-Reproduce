"""Run sequential natural-EOS captures with native request/phase routing metadata."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath, engine_sources
sys.path.insert(0, str(ROOT / "experiments/baselines"))
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready
from experiments.pruning.moe_pruning import atomic_json, sha256, snapshot
from experiments.pruning.pruning_resources import monitor_resources, read_resources, summarize_resources, wait_for_quiet


def uv_python(*args):
    return ["uv", "run", "--no-sync", "--project", str(ROOT), "python", *args]


def json_lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def validate_capture(directory: Path, model: dict, prompt_ids: list, response: dict) -> dict:
    batches = list(json_lines(directory / "batches.jsonl.partial"))
    if not batches:
        raise ValueError("no captured native batches")
    observed_inputs = {}
    phase_positions = {p: set() for p in ("prefill", "decode")}
    generated = []
    for batch in batches:
        if len(batch["sampled_token_ids"]) != 1:
            raise ValueError("capture is not single-request sampling")
        generated.extend(batch["sampled_token_ids"])
        for position, token in zip(batch["input_positions"], batch["input_token_ids"]):
            if position in observed_inputs:
                raise ValueError("repeated native forward position")
            observed_inputs[position] = token
            phase_positions[batch["phase"]].add(position)
    if phase_positions["prefill"] != set(range(len(prompt_ids))):
        raise ValueError("incomplete prefill or reused prefix")
    expected_input = prompt_ids + generated[:-1]
    if sorted(observed_inputs) != list(range(len(expected_input))):
        raise ValueError("forwarded positions do not match causal input/output lifecycle")
    if [observed_inputs[p] for p in sorted(observed_inputs)] != expected_input:
        raise ValueError("observed input IDs disagree with prompt/generated IDs")
    usage = response["usage"]
    if usage["prompt_tokens"] != len(prompt_ids) or usage["completion_tokens"] != len(generated):
        raise ValueError("API usage disagrees with native engine token accounting")
    seen = set()
    phase_counts = {"prefill": 0, "decode": 0}
    import math
    for row in json_lines(directory / "routes.jsonl.partial"):
        key = (row["phase"], row["input_position"], row["layer_id"])
        if key in seen:
            raise ValueError("duplicate routed token/layer")
        seen.add(key)
        ids, weights = row["expert_ids"], row["effective_weights"]
        if row["E"] != model["experts"] or row["k"] != model["top_k"]:
            raise ValueError("wrong routing geometry")
        if len(ids) != model["top_k"] or len(set(ids)) != len(ids):
            raise ValueError("wrong or duplicate top-k expert IDs")
        if any(not 0 <= expert < model["experts"] for expert in ids):
            raise ValueError("expert IDs were remapped or out of range")
        if len(weights) != len(ids) or any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("invalid native weights")
        if abs(sum(weights) - 1) > 2e-5:
            raise ValueError("qualified Qwen/GPT selected weights do not sum to one")
        if row["input_token_id"] != observed_inputs[row["input_position"]] or row["is_shared_expert"]:
            raise ValueError("route token mapping/shared branch contamination")
        phase_counts[row["phase"]] += len(ids)
    expected = {(phase, position, layer) for phase, positions in phase_positions.items()
                for position in positions for layer in range(model["layers"])}
    if seen != expected:
        raise ValueError("missing/extra routed positions or layers")
    cache_rows = list(json_lines(directory / "cache.jsonl.partial"))
    if len(cache_rows) != len(phase_positions["decode"]) * model["layers"]:
        raise ValueError("missing native decode cache events")
    for row in cache_rows:
        if row["m"] != len(row["missing_ids"]) or row["q"] != len(row["admitted_ids"]):
            raise ValueError("cache event accounting mismatch")
        if not 0 <= row["q"] <= row["m"] or row["cpu_executions"] != row["m"] - row["q"]:
            raise ValueError("hybrid compute split accounting mismatch")
        if row["cache_slots"] != model["cache_slots"]:
            raise ValueError("actual cache slots differ from locked configuration")
    return {"status": "valid", "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated),
            "decode_forwards": len(phase_positions["decode"]), "phase_access_counts": phase_counts,
            "generated_token_ids": generated,
            "generated_token_sha256": hashlib.sha256(json.dumps(generated).encode()).hexdigest()}


def request(origin, body):
    req = urllib.request.Request(origin + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def render(tokenizer, prompt, kwargs):
    result = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                          tokenize=True, add_generation_prompt=True, **kwargs)
    return list(result["input_ids"] if hasattr(result, "keys") else result)


def prepare(study):
    from transformers import AutoTokenizer
    protocol = json.loads((study / "protocol/protocol.json").read_text())
    rows = list(json_lines(ROOT / "experiments/prompts/pruning_requests.jsonl"))
    model_configs = json.loads((study / "protocol/models.json").read_text())
    rendered = {}
    for key, model in model_configs.items():
        if not model["required"]:
            continue
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
        rendered[key] = {row["request_id"]: render(tokenizer, row["prompt_text"], protocol["chat_template_kwargs"])
                         for row in rows}
        if max(map(len, rendered[key].values())) > protocol["maximum_prompt_tokens"]:
            raise ValueError(f"{key}: primary prompt exceeds the preregistered cap")
    for name, source in [("dataset.jsonl", "experiments/prompts/pruning_requests.jsonl"),
                         ("splits.json", "experiments/prompts/pruning_splits.json")]:
        target = study / "protocol" / name
        if target.exists():
            if sha256(target) != sha256(ROOT / source):
                raise ValueError("dataset/splits changed after freezing")
        else:
            shutil.copy2(ROOT / source, target)
    target = study / "protocol/rendered_inputs.json"
    if target.exists() and json.loads(target.read_text()) != rendered:
        raise ValueError("rendered input tokens changed")
    if not target.exists():
        atomic_json(target, rendered)
    return rows, protocol, model_configs, rendered


def collect(args):
    study = Path(args.study).resolve()
    subprocess.run(uv_python("-m", "experiments.pruning.collect_moe_routes", "--study", str(study), "--prepare-only"),
                   cwd=ROOT, check=True)
    rows = list(json_lines(study / "protocol/dataset.jsonl"))
    protocol = json.loads((study / "protocol/protocol.json").read_text())
    models = json.loads((study / "protocol/models.json").read_text())
    rendered = json.loads((study / "protocol/rendered_inputs.json").read_text())
    resource_path = study / "protocol/resource_policy.json"
    active_policy = study / "protocol/active_resource_policy.json"
    if active_policy.exists():
        name = json.loads(active_policy.read_text())["filename"]
        if Path(name).name != name:
            raise ValueError("active resource policy must name a file in the protocol directory")
        resource_path = study / "protocol" / name
    if not resource_path.exists():
        raise ValueError("freeze the resource-policy amendment before starting a new pilot")
    resource_policy = json.loads(resource_path.read_text())
    if resource_policy["version"] not in (2, 3, 4, 5):
        raise ValueError("unsupported resource policy version")
    model = models[args.model]
    if not model["required"]:
        raise ValueError("additional-family admission is not yet qualified")
    inference_hashes = {name: sha256(path) for name, path in engine_sources().items()}
    collection_hashes = {path: sha256(ROOT / path) for path in
                         ("experiments/pruning/collect_moe_routes.py", "experiments/pruning/pruning_resources.py", "experiments/pruning/moe_pruning.py")}
    control_hashes = {"resource_policy": sha256(resource_path),
                      "calibration": sha256(ROOT / protocol["calibration_profile"]),
                      **{name: sha256(study / "protocol" / name) for name in
                         ("protocol.json", "models.json", "dataset.jsonl", "splits.json", "rendered_inputs.json")}}
    qualification = study / "pilots" / args.model / "qualification.json"
    memory_ratio = args.memory_ratio or protocol["memory_ratio"]
    if args.stage != "pilot":
        qualified = json.loads(qualification.read_text())
        if (qualified.get("status") != "qualified" or qualified["inference_hashes"] != inference_hashes
                or qualified["captures"] != 15):
            raise ValueError("the inference implementation has not passed the complete pilot gate")
        if qualified.get("collection_hashes") != collection_hashes or qualified.get("control_hashes") != control_hashes:
            raise ValueError("collector or experiment controls differ from the qualified pilot")
        if args.memory_ratio and args.memory_ratio != qualified["memory_ratio"]:
            raise ValueError("memory allocation differs from qualified pilot")
        memory_ratio = qualified["memory_ratio"]
    selected = [r for r in rows if r["pilot_flag"] == (args.stage == "pilot")]
    if args.stage == "repeat":
        repeat_ids = json.loads((study / "protocol/splits.json").read_text())["repeat_request_ids"]
        selected = [r for r in rows if r["request_id"] in repeat_ids]
    if args.limit:
        selected = selected[:args.limit]
    batch_root = study / ("pilots" if args.stage == "pilot" else "routes") / args.model / args.batch_id
    batch_root.mkdir(parents=True, exist_ok=False)
    snap = snapshot(batch_root / "source_snapshot")
    arm = batch_root / "arm.json"
    profile = ROOT / protocol["calibration_profile"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FREETOKEN_MOE_", "FREETOKEN_HYBRID_"))}
    cuda = subprocess.check_output(uv_python("-c",
                                   'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")'), text=True).strip()
    env.update({"CUDA_HOME": cuda, "CC": "/usr/bin/gcc", "PATH": cuda + "/bin:" + env["PATH"],
                "LD_LIBRARY_PATH": cuda + "/lib:" + env.get("LD_LIBRARY_PATH", ""),
                "PYTHONPATH": runtime_pythonpath(),
                "FREETOKEN_BENCHBW_PATH": str(profile), "FREETOKEN_MOE_ROUTING_ARM": str(arm),
                "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1"})
    env.update(resource_policy.get("allocator_environment", {}))
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPU already has compute processes: {active}")
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    cmd = uv_python("-m", "freetoken.cli", "serve", "--model", str(ROOT / model["path"]),
           "--host", "127.0.0.1", "--port", str(port), "--moe-strategy", "hybrid",
           "--max-running-requests", "1", "--max-seq-len-override", "8448",
           "--memory-ratio", str(memory_ratio),
           "--cuda-graph-max-bs", "0", "--moe-hybrid-max-fetch", "-1",
           "--moe-cache-size", str(model["cache_slots"]), "--cache-type", "naive",
           "--expert-load", resource_policy["expert_load"])
    text_model_only = args.model in resource_policy.get("text_model_only_models", [])
    if text_model_only:
        cmd.append("--text-model-only")
    atomic_json(batch_root / "configuration.json", {"command": cmd, "protocol": protocol, "source": snap,
                "calibration_sha256": sha256(profile), "dataset_sha256": sha256(study / "protocol/dataset.jsonl"),
                "stage": args.stage, "inference_hashes": inference_hashes,
                "collection_hashes": collection_hashes, "control_hashes": control_hashes,
                "resource_policy": resource_policy, "tokenizer_lifetime": "worker_exits_before_server_start",
                "text_model_only": text_model_only,
                "scheduler_overlap": False, "expert_cache_initialization": "reset_before_each_request",
                "route_storage": protocol["route_storage"], "memory_ratio": memory_ratio})
    completed = []
    boundaries = {}
    phase = {"name": "startup"}
    with (batch_root / "server.log").open("wb") as log:
        boundaries["startup_begin"] = read_resources(-1)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor_stop = threading.Event()
        monitor = threading.Thread(target=monitor_resources,
                                   args=(proc, monitor_stop, batch_root / "resources.jsonl", phase))
        monitor.start()
        try:
            wait_ready(origin, proc, str(batch_root / "server.log"), 1800)
            model_id = get_json(origin + "/v1/models")["data"][0]["id"]
            atomic_json(batch_root / "startup_stats.json", get_json(origin + "/v1/stats"))
            if resource_policy.get("warmup_completion_tokens"):
                phase["name"] = "warmup"
                warmup_row = next(row for row in rows if row["pilot_flag"])
                atomic_json(arm, {"armed": True, "capture_id": args.batch_id + "_startup_warmup",
                            "request_id": warmup_row["request_id"], "directory": str(batch_root / "startup_warmup"),
                            "prompt_tokens": len(rendered[args.model][warmup_row["request_id"]]),
                            "observe": False, "require_pinned_banks": resource_policy["require_pinned_banks"]})
                warmup_body = {"model": model_id, "messages": [{"role": "user", "content": warmup_row["prompt_text"]}],
                               "max_tokens": resource_policy["warmup_completion_tokens"], "stream": False,
                               **protocol["sampling"], "chat_template_kwargs": protocol["chat_template_kwargs"]}
                try:
                    atomic_json(batch_root / "warmup_response.json", request(origin, warmup_body))
                finally:
                    atomic_json(arm, {"armed": False})
                boundaries["warmup_end"] = read_resources(proc.pid)
            phase["name"] = "settling"
            boundaries["ready"] = read_resources(proc.pid)
            boundaries["inference_begin"] = wait_for_quiet(proc, resource_policy)
            phase["name"] = "inference"
            atomic_json(batch_root / "resource_boundaries.json", boundaries)
            startup_samples = list(json_lines(batch_root / "resources.jsonl"))
            if not startup_samples or any("error" in sample for sample in startup_samples):
                raise ValueError("startup resource observations are incomplete")
            if min(sample["host_available_bytes"] for sample in startup_samples) < resource_policy["minimum_host_available_bytes"]:
                raise ValueError("startup host-memory headroom is below the qualification minimum")
            repeats = 2 if args.stage == "repeat" else 1
            for repeat in range(repeats):
                for row in selected:
                    capture_id = f"{args.batch_id}_{row['request_id']}_r{repeat}"
                    directory = batch_root / capture_id
                    directory.mkdir()
                    body = {"model": model_id, "messages": [{"role": "user", "content": row["prompt_text"]}],
                            "max_tokens": protocol["maximum_completion_tokens"], "stream": False,
                            **protocol["sampling"], "chat_template_kwargs": protocol["chat_template_kwargs"]}
                    metadata = {"armed": True, "capture_id": capture_id, "request_id": row["request_id"],
                                "directory": str(directory), "prompt_tokens": len(rendered[args.model][row["request_id"]]),
                                "require_pinned_banks": resource_policy["require_pinned_banks"],
                                "full_scores": args.stage == "repeat" or row["request_id"] in
                                    json.loads((study / "protocol/splits.json").read_text())["repeat_request_ids"]}
                    try:
                        if args.stage == "pilot":
                            control_dir = directory / "control"
                            atomic_json(arm, {**metadata, "capture_id": capture_id + "_control",
                                             "directory": str(control_dir), "observe": False})
                            control = request(origin, body)
                            atomic_json(directory / "control_response.json", control)
                        atomic_json(arm, metadata)
                        started = time.monotonic()
                        response = request(origin, body)
                        elapsed = time.monotonic() - started
                        atomic_json(directory / "response.json", response)
                        validation = validate_capture(directory, model, rendered[args.model][row["request_id"]], response)
                        if args.stage == "pilot":
                            if control["choices"] != response["choices"] or control["usage"] != response["usage"]:
                                raise ValueError("observation-off/on output or usage changed")
                            validation["observer_control"] = "identical_response_choices_and_usage"
                        checksums = {}
                        for kind in ("routes", "batches", "cache", "scores"):
                            source = directory / (kind + ".jsonl.partial")
                            target = directory / (kind + ".jsonl.gz")
                            with source.open("rb") as original, gzip.open(target, "wb") as destination:
                                shutil.copyfileobj(original, destination)
                            checksums[target.name] = sha256(target)
                        for name in ("expert_bank_residency.json", "decode_initial_cache.json"):
                            if (directory / name).exists():
                                checksums[name] = sha256(directory / name)
                        atomic_json(directory / "manifest.json", {"model_key": args.model, "request": row, "stage": args.stage,
                                    "capture_id": capture_id, "validation": validation, "checksums": checksums,
                                    "source": snap, "elapsed_observed_seconds": elapsed,
                                    "timing_interpretation": "instrumented_not_serving_performance",
                                    "finish_reason": response["choices"][0]["finish_reason"]})
                        completed.append(capture_id)
                        atomic_json(batch_root / "completed.json", completed)
                        print(f"accepted {args.model}/{capture_id}: prompt={validation['prompt_tokens']} "
                              f"generated={validation['generated_tokens']}", flush=True)
                    except Exception as exc:
                        atomic_json(directory / "failure.json", {"error": repr(exc), "request": row, "capture_id": capture_id})
                        raise
                    finally:
                        atomic_json(arm, {"armed": False})
            boundaries["inference_end"] = read_resources(proc.pid)
            atomic_json(batch_root / "resource_boundaries.json", boundaries)
        except BaseException as exc:
            atomic_json(batch_root / "batch_failure.json", {"error": repr(exc), "completed": completed,
                        "phase": phase["name"], "command": cmd, "resource_boundaries": boundaries})
            raise
        finally:
            phase["name"] = "shutdown"
            stop_server(proc)
            monitor_stop.set()
            monitor.join()
    resources = list(json_lines(batch_root / "resources.jsonl"))
    resource_summary = summarize_resources(resources, boundaries, resource_policy)
    atomic_json(batch_root / "resource_summary.json", resource_summary)
    atomic_json(batch_root / "completion.json", {"status": "captures_complete", "captures": len(completed),
                "resource_status": resource_summary["status"]})
    if resource_summary["status"] != "passed":
        atomic_json(batch_root / "qualification_failure.json", {"reason": "resource qualification failed",
                    "resource_summary": resource_summary, "command": cmd})
        raise ValueError("resource qualification failed: " + "; ".join(resource_summary["reasons"]))
    if args.stage == "pilot" and len(completed) == 15:
        atomic_json(qualification, {"status": "qualified", "captures": 15,
                                   "inference_hashes": inference_hashes, "batch_id": args.batch_id,
                                   "collection_hashes": collection_hashes, "control_hashes": control_hashes,
                                   "memory_ratio": memory_ratio, "resource_summary": resource_summary})
        print(f"qualified {args.model}: all 15 pilot controls and resource checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--stage", choices=("pilot", "primary", "repeat"))
    parser.add_argument("--batch-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--memory-ratio", type=float)
    args = parser.parse_args()
    if args.prepare_only:
        prepare(Path(args.study).resolve())
        return
    if not args.model or not args.stage or not args.batch_id:
        parser.error("--model, --stage and --batch-id are required for collection")
    collect(args)


if __name__ == "__main__":
    main()
