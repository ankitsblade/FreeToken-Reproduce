"""P2 sequential teacher forcing on accepted D0 histories (plan sections 3.2, 6 and 10.1).

Each accepted native_floor1 primary capture supplies its frozen rendered prompt IDs and
its generated continuation. The prompt is prefilled from the frozen token IDs (no chat
template is rendered, so GPT-OSS's embedded date cannot change it); the first target is
scored from the final prefill position and every later target through one sequential
decode step that feeds the forced token. Two modes exist:

- baseline: native policy; saves full-vocabulary logits in their native dtype and every
  decode routed MoE input row. It must reproduce the source capture event by event and
  the sampler's own greedy choice must equal every forced target.
- native_control: native policy again, scoring each step against the baseline logits
  online. Zero-change control: KL exactly 0, top-1 agreement 1, identical logits.

- candidate: a P7 decode mask (arm 1 miss-targeted or the R75 frequency reference) from the
  request's own outer-fold artefacts, on the live runtime's decode-only mask hook. Each
  step is scored online against the baseline logits (KL, top-1 agreement, NLL); prefill
  must stay native and no masked expert may be selected in decode.
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import threading
import urllib.request
import uuid
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from experiments.paths import ROOT, engine_sources, python_command, runtime_pythonpath
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready
from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json, sha256, snapshot
from experiments.pruning.m_reduction_traces import accepted_batch
from experiments.pruning.pruning_resources import monitor_resources, read_resources, summarize_resources, wait_for_quiet

STREAMS = ("routes", "batches", "cache", "scores", "teacher")
# Engine-assigned identifiers and the new diagnostic field differ by construction.
IGNORED = {"capture_id", "engine_uid", "sampler_token_ids"}
NUMERIC_ENV = {"mxfp4_alignment_dot64_v2": {"FREETOKEN_M_REDUCTION_MXFP4_ALIGN": "1",
                                            "FREETOKEN_M_REDUCTION_MXFP4_DOT64": "1"},
               "nvfp4_precision_v1": {"FREETOKEN_M_REDUCTION_NVFP4_ALIGN": "1"}}


def complete(origin, body):
    req = urllib.request.Request(origin + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as response:
        return json.load(response)


def history(directory, prompt_ids):
    """Frozen prompt check plus the generated continuation in emission order."""
    rows = list(json_lines(directory / "batches.jsonl.gz"))
    prefill = [row for row in rows if row["phase"] == "prefill"]
    seen = [token for row in prefill for token in row["input_token_ids"]]
    if seen != prompt_ids:
        raise ValueError(directory.name + ": captured prefill differs from the frozen rendering")
    forced = [prefill[-1]["sampled_token_ids"][0]] + [row["sampled_token_ids"][0] for row in rows if row["phase"] == "decode"]
    decode_inputs = [row["input_token_ids"][0] for row in rows if row["phase"] == "decode"]
    if decode_inputs != forced[:-1]:
        raise ValueError(directory.name + ": captured decode inputs are not the generated history")
    return forced


def stripped(row):
    return None if row is None else {key: value for key, value in row.items() if key not in IGNORED}


def compare(new, source):
    """Event-by-event identity against the accepted source capture; streams rows to bound host memory."""
    result = {}
    for name in ("routes", "batches", "cache", "scores"):
        rows = source_rows = mismatches = 0
        first = None
        pairs = zip_longest(json_lines(new / (name + ".jsonl.partial")), json_lines(source / (name + ".jsonl.gz")))
        for index, (a, b) in enumerate(pairs):
            rows += a is not None
            source_rows += b is not None
            if stripped(a) != stripped(b):
                mismatches += 1
                first = index if first is None else first
        result[name] = {"rows": rows, "source_rows": source_rows, "mismatches": mismatches, "first_mismatch": first}
    for name in ("decode_initial_cache.json", "decode_final_cache.json"):
        result[name] = json.loads((new / name).read_text()) == json.loads((source / name).read_text())
    return result


def binary_rows(directory, kind):
    header = json.loads((directory / (kind + "_header.json")).read_text())
    size = (directory / (kind + ".bin")).stat().st_size
    width = header["width"] * header["itemsize"]
    if size % width:
        raise ValueError(f"{directory.name}: {kind}.bin is not whole rows")
    return size // width, header


def validate(directory, source, forced, mode):
    teacher = list(json_lines(directory / "teacher.jsonl.partial"))
    if [row["step"] for row in teacher] != list(range(len(forced))) or [row["target"] for row in teacher] != forced:
        raise ValueError(directory.name + ": teacher-forced steps or targets are incomplete")
    identity = compare(directory, source)
    if any(isinstance(v, dict) and v["mismatches"] for v in identity.values()) or not all(
            v for v in identity.values() if isinstance(v, bool)):
        raise ValueError(directory.name + ": teacher forcing differs from the accepted capture: " + json.dumps(identity))
    result = {"targets": len(forced), "identity": identity,
              "sampler_equals_target": sum(row["sampler_token"] == row["target"] for row in teacher),
              "argmax_equals_target": sum(row["model_argmax"] == row["target"] for row in teacher),
              "phases": {phase: sum(row["phase"] == phase for row in teacher) for phase in ("prefill", "decode")},
              "logits_dtype": teacher[0]["logits_dtype"], "vocab": teacher[0]["vocab"]}
    if result["sampler_equals_target"] != len(forced) or result["argmax_equals_target"] != len(forced):
        raise ValueError(directory.name + ": greedy baseline choice differs from a forced target")
    if mode == "baseline":
        rows, header = binary_rows(directory, "logits")
        inputs, input_header = binary_rows(directory, "moe_inputs")
        if rows != len(forced) or inputs != identity["cache"]["rows"]:
            raise ValueError(directory.name + ": saved logits or MoE input rows are incomplete")
        result.update({"logits_rows": rows, "logits_header": header, "moe_input_rows": inputs,
                       "moe_input_header": input_header})
    else:
        bad = [row["step"] for row in teacher if row["kl_baseline_candidate"] != 0.0 or not row["top1_agree"]
               or not row["reference_identical"]]
        if bad:
            raise ValueError(directory.name + f": zero-change control differs at steps {bad[:10]}")
        result["zero_change_control"] = "passed"
    return result


def validate_candidate(directory, source, forced, removed, native_prefill=True):
    teacher = list(json_lines(directory / "teacher.jsonl.partial"))
    if [row["step"] for row in teacher] != list(range(len(forced))) or [row["target"] for row in teacher] != forced:
        raise ValueError(directory.name + ": teacher-forced steps or targets are incomplete")
    source_prefill = [stripped(row) for row in json_lines(source / "routes.jsonl.gz") if row["phase"] == "prefill"]
    prefill, selected_removed, decode_rows = [], 0, 0
    for row in json_lines(directory / "routes.jsonl.partial"):
        if row["phase"] == "prefill":
            prefill.append(stripped(row))
        else:
            decode_rows += 1
            if removed is not None:
                selected_removed += len(set(row["expert_ids"]) & set(removed[str(row["layer_id"])]))
    if native_prefill and prefill != source_prefill:
        raise ValueError(directory.name + ": candidate prefill routing differs from the native capture")
    if selected_removed:
        raise ValueError(directory.name + f": {selected_removed} decode selections use masked experts")
    cache = [row for row in json_lines(directory / "cache.jsonl.partial") if row["phase"] == "decode"]
    source_cache = [row for row in json_lines(source / "cache.jsonl.gz") if row["phase"] == "decode"]
    kl = [row["kl_baseline_candidate"] for row in teacher]
    return {"targets": len(forced), "decode_route_rows": decode_rows, "prefill_identical": prefill == source_prefill,
            "masked_selections": 0, "mean_kl": sum(kl) / len(kl),
            "top1_agreement": sum(row["top1_agree"] for row in teacher) / len(teacher),
            "mean_nll": sum(row["nll"] for row in teacher) / len(teacher),
            "mean_baseline_nll": sum(row["baseline_nll"] for row in teacher) / len(teacher),
            "decode_events": len(cache), "sum_m": sum(row["m"] for row in cache),
            "sum_q": sum(row["q"] for row in cache), "baseline_sum_m": sum(row["m"] for row in source_cache),
            "baseline_decode_events": len(source_cache)}


def release(directory):
    """Flush and drop this capture's page cache; ~150 MB/history of new files otherwise
    grows the cache until reclaim swaps out other processes (attempt 1, 14 October)."""
    for path in directory.iterdir():
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def seal(directory):
    checksums = {}
    for name in STREAMS:
        original, target = directory / (name + ".jsonl.partial"), directory / (name + ".jsonl.gz")
        with original.open("rb") as source, gzip.open(target, "wb") as out:
            shutil.copyfileobj(source, out)
        with gzip.open(target, "rb") as check:
            if check.read() != original.read_bytes():
                raise ValueError(directory.name + ": compressed stream differs")
        original.unlink()
        checksums[target.name] = sha256(target)
    for path in sorted(directory.glob("*.bin")) + sorted(directory.glob("*.json")):
        checksums[path.name] = sha256(path)
    return checksums


def collect(args):
    study, source_study = args.study.resolve(), args.source_study.resolve()
    if "teacher_step" not in (Path(os.environ["FREETOKEN_SOURCE_DIR"]) / "python/freetoken/moe/routing_trace.py").read_text():
        raise ValueError("FREETOKEN_SOURCE_DIR is not a P2 teacher-forcing runtime")
    source_batch = accepted_batch(source_study, args.model, "primary")
    source_config = json.loads((source_batch / "configuration.json").read_text())
    plan = json.loads((source_study / "protocol/plan.json").read_text())
    rendered = json.loads((source_study / "protocol/rendered_inputs.json").read_text())[args.model]
    model = source_config["model"]
    if sha256(args.calibration) != source_config["dependencies"]["calibration"]:
        raise ValueError("Calibration differs from the source capture's qualified calibration")
    captures = sorted(p for p in source_batch.iterdir() if (p / "manifest.json").is_file())
    if len(captures) != 90:
        raise ValueError("Accepted primary batch does not hold 90 captures")
    if args.subset == "repeat_ids":
        repeat = set(json.loads((source_study / "protocol/splits.json").read_text())["repeat_request_ids"])
        captures = [p for p in captures if json.loads((p / "manifest.json").read_text())["request"]["request_id"] in repeat]
        if len(captures) != len(repeat):
            raise ValueError("Repeat request IDs are missing from the primary batch")
    if args.limit:
        captures = captures[:args.limit]
    work = []
    for capture in captures:
        row = json.loads((capture / "manifest.json").read_text())["request"]
        work.append((capture, row, rendered[row["request_id"]], history(capture, rendered[row["request_id"]])))
    reference = None
    if args.mode in ("native_control", "candidate"):
        reference = args.reference_batch.resolve()
        if json.loads((reference / "completion.json").read_text()).get("status") != "complete":
            raise ValueError("Reference baseline teacher-forcing batch is not complete")
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    mode_dir = "teacher_" + args.mode + ("_" + args.candidate_arm if args.mode == "candidate" else "")
    if args.fold is not None:
        mode_dir += f"_fold{args.fold}"
    batch = study / "captures" / args.model / mode_dir / args.subset / batch_id
    batch.mkdir(parents=True)
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    masks = {}
    if args.mode == "candidate" and args.candidate_arm in ("arm2", "requant"):
        # Arm 2 / requantization-only control: load-time merged weights plus router aliasing
        # (both phases); no decode mask, so prefill is not native.
        if "alias_ids" not in (Path(os.environ["FREETOKEN_SOURCE_DIR"]) / "python/freetoken/moe/routing_trace.py").read_text():
            raise ValueError("Arm 2 teacher forcing needs the live2 runtime's merge/alias hooks")
        merge = json.loads(args.merge_control.read_text())
        if merge["model"] != args.model or (args.candidate_arm == "arm2") != (not merge.get("requant_all")):
            raise ValueError("merge control does not match the model/arm")
        if args.candidate_arm == "arm2" and merge["fold"] != args.fold:
            raise ValueError("arm 2 merge control is fitted for a different outer fold")
        atomic_json(batch / "merge_control.json", merge)
        group_fold = json.loads((source_study / "protocol/splits.json").read_text())["group_to_fold"]
        work = [(*item, int(group_fold[item[1]["group_id"]])) for item in work]
        if args.fold is not None:
            work = [item for item in work if item[4] == args.fold]
    elif args.mode == "candidate":
        if "mask_decode_logits" not in (Path(os.environ["FREETOKEN_SOURCE_DIR"]) / "python/freetoken/moe/routing_trace.py").read_text():
            raise ValueError("Candidate teacher forcing needs the live runtime's decode mask hook")
        artefacts = json.loads(args.artefacts.read_text())
        group_fold = json.loads((source_study / "protocol/splits.json").read_text())["group_to_fold"]
        (batch / "masks").mkdir()
        for fold, entry in artefacts.items():
            removed = entry[args.candidate_arm + "_removed"]
            atomic_json(batch / "masks" / f"fold{fold}.json", {"arm": args.candidate_arm, "fold": int(fold),
                        "layers": removed, "semantics": "decode-only logits -inf before native top-k"})
            masks[int(fold)] = removed
        work = [(*item, int(group_fold[item[1]["group_id"]])) for item in work]
    else:
        work = [(*item, None) for item in work]
    resource = json.loads((ROOT / plan["resource_policy"]).read_text())
    dependencies = {"source_batch": str(source_batch), "source_configuration": sha256(source_batch / "configuration.json"),
                    "calibration": sha256(args.calibration), "collector_source": sha256(Path(__file__)),
                    "runtime_patch": sha256(ROOT / "runtime/patches/m-reduction-p2-teacher.patch"),
                    "rendered_inputs": sha256(source_study / "protocol/rendered_inputs.json"),
                    "resource_control": sha256(ROOT / plan["resource_policy"]),
                    "reference_batch": str(reference) if reference else None,
                    "artefacts": sha256(args.artefacts) if args.artefacts else None,
                    "candidate_arm": args.candidate_arm,
                    "merge_control": sha256(args.merge_control) if args.merge_control else None,
                    "merge_source": sha256(ROOT / "experiments/pruning/m_reduction_merge.py") if args.merge_control else None,
                    "fold": args.fold}
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FREETOKEN_MOE_", "FREETOKEN_HYBRID_", "FREETOKEN_M_REDUCTION_"))}
    arm = batch / "arm.json"
    env.update({"PYTHONPATH": runtime_pythonpath(), "FREETOKEN_BENCHBW_PATH": str(args.calibration.resolve()),
                "FREETOKEN_MOE_ROUTING_ARM": str(arm),
                "FREETOKEN_M_REDUCTION_FILL_FLOOR": str(source_config["policy"]["fill_floor"]),
                "FREETOKEN_M_REDUCTION_TOKEN_PROMPTS": "1", "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **resource["allocator_environment"],
                **NUMERIC_ENV[plan["numeric_arm"]]})
    if (batch / "merge_control.json").exists():
        env["FREETOKEN_M_REDUCTION_MERGE"] = str(batch / "merge_control.json")
    if source_config["policy"]["kind"] != "native":
        raise ValueError("Teacher-forcing baselines replay the native policy only")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is occupied")
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    command = list(source_config["command"])
    command[command.index("--port") + 1] = str(port)
    atomic_json(batch / "configuration.json", {
        "command": command, "mode": args.mode, "model": model, "model_key": args.model,
        "numeric_arm": plan["numeric_arm"], "policy": source_config["policy"], "dependencies": dependencies,
        "inference_hashes": hashes, "source_snapshot": snapshot(batch / "source_snapshot"),
        "environment": {k: v for k, v in env.items() if k.startswith(("FREETOKEN_", "OMP_", "MKL_", "MALLOC_"))},
        "subset": args.subset, "request_ids": [row["request_id"] for _, row, _, _, _ in work],
        "interpretation": ("P7 candidate sequential teacher forcing against baseline logits" if args.mode == "candidate"
                           else "P2 sequential teacher forcing qualification, not serving performance")})
    stop = threading.Event()
    phase = {"name": "startup"}
    boundaries = {"startup_begin": read_resources(-1)}
    done = []
    with (batch / "server.log").open("wb") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor = threading.Thread(target=monitor_resources, args=(proc, stop, batch / "resources.jsonl", phase))
        monitor.start()
        try:
            wait_ready(origin, proc, str(batch / "server.log"), 1800)
            model_id = get_json(origin + "/v1/models")["data"][0]["id"]
            atomic_json(batch / "startup_stats.json", get_json(origin + "/v1/stats"))
            phase["name"] = "warmup"
            warm = rendered["math_000"]
            atomic_json(arm, {"armed": True, "capture_id": "warmup", "request_id": "math_000",
                             "directory": str(batch / "warmup"), "prompt_tokens": len(warm),
                             "observe": False, "require_pinned_banks": True})
            atomic_json(batch / "warmup_response.json", complete(origin, {
                "model": model_id, "prompt": warm, "max_tokens": plan["warmup_completion_tokens"],
                **plan["sampling"]}))
            atomic_json(arm, {"armed": False})
            phase["name"] = "settling"
            boundaries["ready"] = read_resources(proc.pid)
            boundaries["inference_begin"] = wait_for_quiet(proc, resource)
            phase["name"] = "inference"
            for source, row, prompt, forced, fold in work:
                directory = batch / source.name
                directory.mkdir()
                metadata = {"armed": True, "capture_id": directory.name, "request_id": row["request_id"],
                            "directory": str(directory), "prompt_tokens": len(prompt), "require_pinned_banks": True,
                            "full_scores": args.mode != "candidate", "prompt_token_ids": prompt, "forced_tokens": forced}
                if args.mode == "candidate" and masks:
                    metadata["decode_mask"] = str(batch / "masks" / f"fold{fold}.json")
                if args.mode == "baseline":
                    metadata.update({"save_logits": True, "save_moe_inputs": True})
                else:
                    metadata["kl_reference"] = {"path": str(reference / source.name / "logits.bin"),
                                                "header": str(reference / source.name / "logits_header.json")}
                try:
                    atomic_json(arm, metadata)
                    # ignore_eos only disables the stop check; every forced target is scored.
                    response = complete(origin, {"model": model_id, "prompt": prompt, "max_tokens": len(forced),
                                                 **plan["sampling"], "ignore_eos": True})
                    atomic_json(arm, {"armed": False})
                    atomic_json(directory / "response.json", response)
                    if response["usage"]["completion_tokens"] != len(forced):
                        raise ValueError(directory.name + ": completion length differs from the forced history")
                    result = (validate_candidate(directory, source, forced, masks.get(fold), native_prefill=bool(masks))
                              if args.mode == "candidate"
                              else validate(directory, source, forced, args.mode))
                    if fold is not None:
                        result["outer_fold"] = fold
                    checksums = seal(directory)
                    atomic_json(directory / "manifest.json", {"request": row, "model": args.model, "mode": args.mode,
                                "source_capture": str(source), "validation": result, "checksums": checksums})
                    release(directory)
                    done.append(directory.name)
                    atomic_json(batch / "completed.json", done)
                    print(f"Teacher-forced {args.model}/{args.mode}/{directory.name}: {len(done)}/{len(work)}", flush=True)
                except BaseException as exc:
                    atomic_json(directory / "failure.json", {"error": repr(exc)})
                    raise
                finally:
                    atomic_json(arm, {"armed": False})
            boundaries["inference_end"] = read_resources(proc.pid)
            atomic_json(batch / "resource_boundaries.json", boundaries)
        except BaseException as exc:
            atomic_json(batch / "failure.json", {"error": repr(exc), "phase": phase["name"], "completed": done,
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
    atomic_json(batch / "completion.json", {"status": "complete", "captures": len(done), "resources": summary,
                                            "limit": args.limit})
    print(f"Completed teacher forcing {args.model}/{args.mode}: {len(done)} captures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True, help="New P2 study root for teacher-forced captures")
    parser.add_argument("--source-study", type=Path, required=True, help="Study holding the accepted primary captures")
    parser.add_argument("--model", choices=("gpt_oss_mxfp4", "qwen_nvfp4", "gemma_nvfp4"), required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "native_control", "candidate"), default="baseline")
    parser.add_argument("--reference-batch", type=Path, help="Complete baseline batch for native_control/candidate KL")
    parser.add_argument("--artefacts", type=Path, help="P7 per-fold artefacts JSON (candidate mode)")
    parser.add_argument("--candidate-arm", choices=("arm1", "r75", "arm2", "requant"),
                        help="P7 decode mask family, arm 2 merge, or the requantization-only control (candidate mode)")
    parser.add_argument("--merge-control", type=Path, help="m_reduction_merge control (arm2/requant)")
    parser.add_argument("--fold", type=int, help="Evaluate only this outer fold's held-out histories")
    parser.add_argument("--subset", choices=("all", "repeat_ids"), default="all",
                        help="all 90 primary histories, or the nine repeat request IDs")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.mode in ("native_control", "candidate") and not args.reference_batch:
        parser.error(args.mode + " needs --reference-batch")
    if args.mode == "candidate" and args.candidate_arm in ("arm2", "requant"):
        if not args.merge_control or (args.candidate_arm == "arm2" and args.fold is None):
            parser.error("arm2 needs --merge-control and --fold; requant needs --merge-control")
    elif args.mode == "candidate" and not (args.artefacts and args.candidate_arm):
        parser.error("candidate needs --artefacts and --candidate-arm")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    collect(args)


if __name__ == "__main__":
    main()
