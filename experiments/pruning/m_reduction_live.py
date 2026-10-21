"""Live confirmation of exact cache arms (P4 q rules, P5 quotas, P9 SLRU), plan sections 4.3, 8 and 10.2.

Every arm runs on the isolated live runtime (quota/SLRU kernel modes, fixed-fraction and
Reuse-Aware admission hooks) with the accepted native_floor1 server command, frozen
rendered token-ID prompts, greedy natural-EOS generation and the plan's warmup/reset
lifecycle. Three entry points:

- ``--arm-run``: one fresh server for one arm over the nine repeat request IDs.
  ``--kind fidelity`` observes routes/cache events and requires every live decode event
  (m, q, admitted IDs) to equal the offline simulator under the arm's rule, and the
  generated tokens to equal the accepted primary capture. ``--kind timing`` is lean:
  observation off, only sampled token IDs logged for output identity, streamed timing.
- ``--controller``: fidelity for every arm, then five paired timing repetitions with arm
  order shuffled by seed 20261013 + repetition, each arm on a fresh server. A
  resource-rejected arm run rejects its whole repetition, which is repeated in fresh
  directories (at most three attempts).
- ``--analyze``: paired geometric latency ratios versus overlap_floor1 with 10,000
  request-cluster bootstrap resamples (seed 20261014) and the G2 exact-component decision
  combining the offline m evidence, output identity and the <=1.02 timing bound.
"""

import argparse
import http.client
import json
import math
import os
import random
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.paths import ROOT, engine_sources, runtime_pythonpath
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready
from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json, sha256, snapshot
from experiments.pruning.m_reduction_teacher import NUMERIC_ENV, history, release
from experiments.pruning.m_reduction_traces import accepted_batch
from experiments.pruning.pruning_resources import monitor_resources, read_resources, summarize_resources, wait_for_quiet


MODELS = ("gpt_oss_mxfp4", "qwen_nvfp4", "gemma_nvfp4")
BASELINE = "overlap_floor1"
RETRYABLE = {"system swapping during inference exceeds limit",
             "startup or inference host-memory headroom below minimum",
             "sustained full memory pressure during inference"}


def now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_arms(model, offline, calibration):
    """Arm table from the Stage 1 offline evidence; every arm keeps floor 1 and C fixed."""
    p4 = json.loads((offline / model / "p4/summary.json").read_text())[model]
    p5 = json.loads((offline / model / "p5/summary.json").read_text())[model]
    p9 = json.loads((offline / model / "p9/summary.json").read_text())[model]
    fmt = "mxfp4_triton" if model == "gpt_oss_mxfp4" else "nvfp4"
    profile = json.loads(Path(calibration).read_text())
    entry = next(w["kernels"][fmt] for w in profile["workloads"].values() if fmt in w["kernels"])
    rho = entry["pcie_gather_overlap_gbs"] / entry["cpu_moe_overlap_gbs"]
    if abs(rho - p4["derived"]["rho"]) > 1e-12:
        raise ValueError(model + ": calibration rho differs from the P4 offline evidence")
    reuse = {"kind": "reuse_aware", "fill_floor": 1, "rho": rho}
    arms = {
        BASELINE: {"policy": None, "replacement": None, "sim": {"rule": "native"}},
        "paper_floor1": {"policy": {"kind": "fraction", "fraction": p4["derived"]["paper_fraction"]},
                         "replacement": None, "sim": {"rule": "fixed", "fraction": p4["derived"]["paper_fraction"]}},
        "fixed25_floor1": {"policy": {"kind": "fraction", "fraction": 0.25}, "replacement": None,
                           "sim": {"rule": "fixed", "fraction": 0.25}},
        "reuse_uniform_floor1": {"policy": {**reuse, "per_expert": False}, "replacement": None,
                                 "sim": {"rule": "reuse", "per_expert": False, "rho": rho}},
        "reuse_freq_floor1": {"policy": {**reuse, "per_expert": True}, "replacement": None,
                              "sim": {"rule": "reuse", "per_expert": True, "rho": rho}},
        "quota_uniform": {"policy": None, "replacement": {"kind": "quota", "quotas": p5["uniform_quotas"]},
                          "sim": {"rule": "native", "replacement": "quota", "quotas": p5["uniform_quotas"]}},
        "quota_utility": {"policy": None, "replacement": {"kind": "quota", "quotas": p5["deployment_quotas"]},
                          "sim": {"rule": "native", "replacement": "quota", "quotas": p5["deployment_quotas"]}},
    }
    if p5["q_arm"] != BASELINE:
        raise ValueError(model + ": P5 quotas were fitted under a different q rule")
    # P9: only offline G2 survivors (decision defer_timing) need live timing.
    for name, row in p9["comparisons"].items():
        if row["g2_exact"]["decision"] == "defer_timing":
            if not name.startswith("slru_"):
                raise ValueError("Live pin arms are not implemented: " + name)
            fraction = float(name.split("_")[1])
            arms[name] = {"policy": None, "replacement": {"kind": "slru", "fraction": fraction},
                          "sim": {"rule": "native", "replacement": "slru", "slru_fraction": fraction}}
    return arms


def build_closeout_arms(model, offline, calibration, p1_seed):
    """Plan section 10.3 closeout block (own study, fresh paired repetitions): native floor 0
    (original rounding control), corrected native floor 1 and each model's best exact
    component that kept G2 live -- GPT-OSS quota_utility -- or, for Qwen, the P1 whole-prompt
    seeding arm (the only seeding cell meeting the offline 5% m criterion)."""
    stage2 = build_arms(model, offline, calibration)
    arms = {BASELINE: stage2[BASELINE],
            "overlap_floor0": {"policy": None, "replacement": None, "fill_floor": 0,
                               "sim": {"rule": "native", "fill_floor": 0}}}
    if model == "gpt_oss_mxfp4":
        arms["quota_utility"] = stage2["quota_utility"]
    if model == "qwen_nvfp4":
        p1 = json.loads((p1_seed / model / "summary.json").read_text())[model]["arms"]["seed_whole"]
        if not (p1["vs_native_mean_m"]["reduction"] >= 0.05 and p1["vs_native_mean_m"]["reduction_ci95"][0] > 0):
            raise ValueError(model + ": P1 seeding no longer meets the offline m criterion")
        arms["seed_whole"] = {"policy": None, "replacement": None, "seed": {"window": None},
                              "sim": {"rule": "native"}}
    return arms


def sim_events(cache_rows, initial, geometry, spec, fetch_fraction):
    from experiments.pruning.m_reduction_sim import CacheSim, QRule

    if spec["rule"] == "native":
        rule = QRule("native", fraction=fetch_fraction)
    elif spec["rule"] == "fixed":
        rule = QRule("fixed", fraction=spec["fraction"])
    else:
        rule = QRule("reuse", rho=spec["rho"], per_expert=spec["per_expert"], top_k=geometry["top_k"],
                     experts=geometry["experts"])
    sim = CacheSim(initial, geometry["layers"], geometry["experts"], rule, spec.get("fill_floor", 1),
                   spec.get("replacement", "lru"),
                   spec.get("quotas"), spec.get("slru_fraction"))
    mismatches, first = 0, None
    for index, row in enumerate(cache_rows):
        out = sim.event(row["layer_id"], list(dict.fromkeys(row["raw_ids"])))
        live = (row["m"], row["q"], row["admitted_ids"])
        if live != (out["m"], out["q"], out["admitted"]):
            mismatches += 1
            first = first if first is not None else {"event": index, "live": live,
                                                     "sim": (out["m"], out["q"], out["admitted"])}
    return {"events": len(cache_rows), "mismatches": mismatches, "first_mismatch": first}


def stream(origin, body):
    """Streamed completion with client-side arrival times (seconds since send)."""
    host, port = origin.split("//")[1].split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=3600)
    start = time.perf_counter()
    connection.request("POST", "/v1/completions", body=json.dumps({**body, "stream": True}),
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"completion failed: {response.status} {response.read()[:500]!r}")
    arrivals, text, finish = [], [], None
    for raw in response:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if "error" in chunk:
            raise RuntimeError("streamed error: " + json.dumps(chunk["error"]))
        choice = chunk["choices"][0]
        if choice.get("text"):
            arrivals.append(time.perf_counter() - start)
            text.append(choice["text"])
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    end = time.perf_counter() - start
    connection.close()
    return {"latency_seconds": end, "first_text_seconds": arrivals[0] if arrivals else None,
            "text_chunk_arrivals": arrivals, "text": "".join(text), "finish_reason": finish}


def generated_tokens(directory, kind):
    if kind == "timing":
        rows = [line.split() for line in (directory / "sampled_tokens.txt").read_text().splitlines()]
        prefill = [int(t) for phase, t in rows if phase == "prefill"]
        return prefill[-1:] + [int(t) for phase, t in rows if phase == "decode"]
    rows = list(json_lines(directory / "batches.jsonl.partial"))
    prefill = [row for row in rows if row["phase"] == "prefill"]
    return [prefill[-1]["sampled_token_ids"][0]] + [row["sampled_token_ids"][0] for row in rows if row["phase"] == "decode"]


def arm_run(args):
    study, source_study = args.study.resolve(), args.source_study.resolve()
    control = json.loads((study / "controls" / args.model / "arms.json").read_text())
    spec = control["arms"][args.arm]
    source_batch = Path(control["source_batch"])
    source_config = json.loads((source_batch / "configuration.json").read_text())
    plan = json.loads((source_study / "protocol/plan.json").read_text())
    rendered = json.loads((source_study / "protocol/rendered_inputs.json").read_text())[args.model]
    repeat = json.loads((source_study / "protocol/splits.json").read_text())["repeat_request_ids"]
    sources = {json.loads((p / "manifest.json").read_text())["request"]["request_id"]: p
               for p in source_batch.iterdir() if (p / "manifest.json").is_file()}
    cohort = [(rid, rendered[rid], sources[rid], history(sources[rid], rendered[rid])) for rid in repeat]
    if args.limit:
        cohort = cohort[:args.limit]
    batch = study / "runs_live" / args.model / args.kind / args.label / (now() + "_" + args.arm + "_" + uuid.uuid4().hex[:6])
    batch.mkdir(parents=True)
    resource = json.loads((ROOT / plan["resource_policy"]).read_text())
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FREETOKEN_MOE_", "FREETOKEN_HYBRID_", "FREETOKEN_M_REDUCTION_"))}
    arm_file = batch / "arm.json"
    env.update({"PYTHONPATH": runtime_pythonpath(), "FREETOKEN_BENCHBW_PATH": control["calibration"],
                "FREETOKEN_MOE_ROUTING_ARM": str(arm_file),
                "FREETOKEN_M_REDUCTION_FILL_FLOOR": str(spec.get("fill_floor", 1)),
                "FREETOKEN_M_REDUCTION_TOKEN_PROMPTS": "1", "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **resource["allocator_environment"],
                **NUMERIC_ENV[plan["numeric_arm"]]})
    if spec["policy"]:
        atomic_json(batch / "policy.json", {**spec["policy"], "top_k": source_config["model"]["top_k"]})
        env["FREETOKEN_M_REDUCTION_POLICY"] = str(batch / "policy.json")
    if spec["replacement"]:
        atomic_json(batch / "replacement.json", {**spec["replacement"], "top_k": source_config["model"]["top_k"]})
        env["FREETOKEN_M_REDUCTION_REPLACEMENT"] = str(batch / "replacement.json")
    if spec.get("seed"):
        atomic_json(batch / "seed.json", {**spec["seed"], "log": str(batch / "seed_log.jsonl")})
        env["FREETOKEN_M_REDUCTION_SEED"] = str(batch / "seed.json")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is occupied")
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    command = list(source_config["command"])
    command[command.index("--port") + 1] = str(port)
    hashes = {name: sha256(path) for name, path in engine_sources().items()}
    atomic_json(batch / "configuration.json", {
        "command": command, "kind": args.kind, "arm": args.arm, "arm_spec": spec, "label": args.label,
        "model_key": args.model, "model": source_config["model"], "numeric_arm": plan["numeric_arm"],
        "inference_hashes": hashes, "controls_sha256": sha256(study / "controls" / args.model / "arms.json"),
        "controller_source_sha256": sha256(Path(__file__)),
        "policy_source_sha256": sha256(ROOT / "experiments/pruning/m_reduction_policy.py"),
        "runtime_patch_sha256": sha256(ROOT / "runtime/patches/m-reduction-live.patch"),
        "source_snapshot": snapshot(batch / "source_snapshot"),
        "environment": {k: v for k, v in env.items() if k.startswith(("FREETOKEN_", "OMP_", "MKL_", "MALLOC_"))},
        "request_ids": [rid for rid, *_ in cohort],
        "interpretation": ("event-level fidelity of a live exact cache arm" if args.kind == "fidelity" else
                           "lean paired timing of a live exact cache arm; observation off")})
    stop = threading.Event()
    phase = {"name": "startup"}
    boundaries = {"startup_begin": read_resources(-1)}
    results = []
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
            atomic_json(arm_file, {"armed": True, "capture_id": "warmup", "request_id": "math_000",
                                   "directory": str(batch / "warmup"), "prompt_tokens": len(warm),
                                   "observe": False, "require_pinned_banks": True})
            stream(origin, {"model": model_id, "prompt": warm, "max_tokens": plan["warmup_completion_tokens"],
                            **plan["sampling"]})
            atomic_json(arm_file, {"armed": False})
            phase["name"] = "settling"
            boundaries["ready"] = read_resources(proc.pid)
            boundaries["inference_begin"] = wait_for_quiet(proc, resource)
            phase["name"] = "inference"
            for rid, prompt, source, expected in cohort:
                directory = batch / rid
                metadata = {"armed": True, "capture_id": f"{batch.name}_{rid}", "request_id": rid,
                            "directory": str(directory), "prompt_tokens": len(prompt), "require_pinned_banks": True,
                            "prompt_token_ids": prompt}
                metadata.update({"observe": True, "full_scores": False} if args.kind == "fidelity" else
                                {"observe": False, "record_tokens": True})
                atomic_json(arm_file, metadata)
                try:
                    timing = stream(origin, {"model": model_id, "prompt": prompt,
                                             "max_tokens": plan["maximum_completion_tokens"], **plan["sampling"]})
                finally:
                    atomic_json(arm_file, {"armed": False})
                tokens = generated_tokens(directory, args.kind)
                row = {"request_id": rid, "tokens": len(tokens), "output_identical": tokens == expected,
                       "first_difference": next((i for i, (a, b) in enumerate(zip(tokens, expected)) if a != b),
                                                None if len(tokens) == len(expected) else min(len(tokens), len(expected))),
                       **{k: v for k, v in timing.items() if k not in ("text", "text_chunk_arrivals")}}
                gaps = np.diff(timing["text_chunk_arrivals"])
                row["decode_ms_per_token"] = (1000 * (timing["latency_seconds"] - timing["first_text_seconds"])
                                              / max(1, len(tokens) - 1)) if timing["first_text_seconds"] else None
                row["chunk_interarrival_p95_ms"] = float(np.quantile(gaps, 0.95, method="higher") * 1000) if len(gaps) else None
                atomic_json(directory / "timing.json", {**timing, "tokens": tokens})
                if args.kind == "fidelity":
                    cache_rows = [r for r in json_lines(directory / "cache.jsonl.partial") if r["phase"] == "decode"]
                    initial = json.loads((directory / "decode_initial_cache.json").read_text())
                    geometry = {k: source_config["model"][k] for k in ("layers", "experts", "top_k")}
                    row["replay"] = sim_events(cache_rows, initial, geometry, spec["sim"], float(initial["fetch_fraction"]))
                    row["sum_m"] = sum(r["m"] for r in cache_rows)
                    row["sum_q"] = sum(r["q"] for r in cache_rows)
                    release(directory)
                results.append(row)
                atomic_json(batch / "requests.json", results)
                print(f"{args.model}/{args.kind}/{args.arm} {rid}: tokens={len(tokens)} identical={row['output_identical']}"
                      + (f" replay_mismatches={row['replay']['mismatches']}" if args.kind == "fidelity" else
                         f" latency={timing['latency_seconds']:.2f}s"), flush=True)
            boundaries["inference_end"] = read_resources(proc.pid)
            atomic_json(batch / "resource_boundaries.json", boundaries)
        except BaseException as exc:
            atomic_json(batch / "failure.json", {"error": repr(exc), "phase": phase["name"], "completed": len(results),
                                                 "boundaries": boundaries})
            raise
        finally:
            phase["name"] = "shutdown"
            stop_server(proc)
            stop.set()
            monitor.join()
    summary = summarize_resources(list(json_lines(batch / "resources.jsonl")), boundaries, resource)
    atomic_json(batch / "resource_summary.json", summary)
    failures = []
    if summary["status"] != "passed":
        failures.append("resource gate: " + "; ".join(summary["reasons"]))
    if args.kind == "fidelity":
        failures += [f"{r['request_id']}: replay mismatch" for r in results if r["replay"]["mismatches"]]
    status = "complete" if not failures else "rejected"
    atomic_json(batch / "completion.json", {"status": status, "failures": failures, "requests": len(results),
                                            "outputs_identical": sum(r["output_identical"] for r in results),
                                            "resources": summary, "limit": args.limit})
    print(json.dumps({"batch": str(batch), "status": status, "failures": failures}), flush=True)
    if failures:
        raise SystemExit(2 if summary["status"] != "passed" else 3)


def launch_arm(study, args, kind, arm, label):
    command = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemorySwapMax=0", "bash",
               str(ROOT / "scripts/run.sh"), "pruning", "m_reduction_live", "--arm-run", "--study", str(study),
               "--source-study", str(args.source_study.resolve()), "--model", args.model, "--arm", arm,
               "--kind", kind, "--label", label]
    print("Launching: " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def latest(study, model, kind, label, arm):
    runs = sorted((study / "runs_live" / model / kind / label).glob(f"*_{arm}_*"))
    return runs[-1] if runs else None


def resource_retryable(batch):
    if batch is None:
        return False
    failure = batch / "failure.json"
    if failure.is_file() and "did not settle" in json.loads(failure.read_text()).get("error", ""):
        return True
    summary = batch / "resource_summary.json"
    reasons = json.loads(summary.read_text()).get("reasons", []) if summary.is_file() else []
    return bool(reasons) and set(reasons).issubset(RETRYABLE)


def log(study, kind, **details):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "kind": kind, **details}
    with (study / "controller_events.jsonl").open("a") as stream_:
        stream_.write(json.dumps(row) + "\n")
    print(kind + ": " + json.dumps(details), flush=True)


def accepted_run(study, model, kind, label, arm):
    batch = latest(study, model, kind, label, arm)
    if batch and (batch / "completion.json").is_file():
        done = json.loads((batch / "completion.json").read_text())
        if done["status"] == "complete" and not done.get("limit"):
            return batch
    return None


def freeze_control(args):
    study = args.study.resolve()
    control_path = study / "controls" / args.model / "arms.json"
    source_batch = accepted_batch(args.source_study.resolve(), args.model, "primary")
    calibration = json.loads((source_batch / "configuration.json").read_text())["dependencies"]["calibration"]
    if sha256(args.calibration) != calibration:
        raise ValueError("Calibration differs from the accepted primary capture's")
    control = {"model": args.model, "source_batch": str(source_batch), "calibration": str(args.calibration.resolve()),
               "offline_stage1": str(args.offline.resolve()),
               "arms": (build_arms(args.model, args.offline.resolve(), args.calibration) if args.arm_set == "stage2" else
                        build_closeout_arms(args.model, args.offline.resolve(), args.calibration, args.p1_seed.resolve())),
               "repetitions": args.repetitions, "order_seed_base": 20261013,
               "timing_unit": ("each arm runs on a fresh server per repetition and serves the nine repeat IDs "
                               "in their declared order; arm order is shuffled within each repetition")}
    if args.arm_set == "closeout":
        control.update({"arm_set": "closeout", "p1_seed": str(args.p1_seed.resolve())})
    control_path.parent.mkdir(parents=True, exist_ok=True)
    if control_path.exists():
        if json.loads(control_path.read_text()) != control:
            raise ValueError("Frozen live arm control changed: " + str(control_path))
    else:
        atomic_json(control_path, control)
    return control


def controller(args):
    from experiments.pruning.m_reduction_guard import wait_quiet

    study = args.study.resolve()
    control = freeze_control(args)
    state = lambda *a, **k: None  # noqa: E731
    arms = list(control["arms"])
    for arm in arms:
        if accepted_run(study, args.model, "fidelity", "fidelity", arm):
            continue
        for attempt in range(1, 4):
            if attempt > 1:
                wait_quiet(study, "fidelity retry", state)
            code = launch_arm(study, args, "fidelity", arm, "fidelity")
            batch = latest(study, args.model, "fidelity", "fidelity", arm)
            log(study, "fidelity_run", model=args.model, arm=arm, attempt=attempt, code=code, batch=str(batch))
            if code == 0:
                break
            if not resource_retryable(batch) or attempt == 3:
                raise SystemExit(f"{args.model}/{arm}: fidelity failed; see {batch}")
    if args.fidelity_only:
        return
    for repetition in range(1, args.repetitions + 1):
        order = arms[:]
        random.Random(20261013 + repetition).shuffle(order)
        # Resume: a repetition is done once any attempt has every arm accepted.
        if any(all(accepted_run(study, args.model, "timing", f"rep{repetition}_attempt{a}", arm) for arm in order)
               for a in range(1, 4)):
            continue
        for attempt in range(1, 4):
            label = f"rep{repetition}_attempt{attempt}"
            done = [arm for arm in order if accepted_run(study, args.model, "timing", label, arm)]
            if len(done) == len(order):
                break
            if attempt > 1 or done:
                wait_quiet(study, "timing retry", state)
            atomic_json(study / "runs_live" / args.model / "timing" / label / "order.json",
                        {"repetition": repetition, "attempt": attempt, "seed": 20261013 + repetition, "order": order})
            failed = None
            for arm in order:
                if accepted_run(study, args.model, "timing", label, arm):
                    continue
                code = launch_arm(study, args, "timing", arm, label)
                batch = latest(study, args.model, "timing", label, arm)
                log(study, "timing_run", model=args.model, arm=arm, label=label, code=code, batch=str(batch))
                if code != 0:
                    failed = batch
                    break
            if failed is None:
                break
            if not resource_retryable(failed) or attempt == 3:
                raise SystemExit(f"{args.model} repetition {repetition}: timing failed; see {failed}")
            log(study, "repetition_rejected", model=args.model, repetition=repetition, attempt=attempt)
    analyze(args)


def accepted_repetitions(study, model, arms, repetitions):
    """Latest attempt per repetition in which every arm completed."""
    blocks = {}
    for repetition in range(1, repetitions + 1):
        for attempt in (3, 2, 1):
            label = f"rep{repetition}_attempt{attempt}"
            runs = {arm: accepted_run(study, model, "timing", label, arm) for arm in arms}
            if all(runs.values()):
                blocks[repetition] = runs
                break
    return blocks


def analyze(args):

    study = args.study.resolve()
    control = json.loads((study / "controls" / args.model / "arms.json").read_text())
    arms = list(control["arms"])
    blocks = accepted_repetitions(study, args.model, arms, control["repetitions"])
    fidelity = {arm: accepted_run(study, args.model, "fidelity", "fidelity", arm) for arm in arms}
    data = {}
    for repetition, runs in blocks.items():
        for arm, batch in runs.items():
            for row in json.loads((batch / "requests.json").read_text()):
                data.setdefault(arm, {})[(row["request_id"], repetition)] = row
    keys = sorted(data.get(BASELINE, {}))
    requests = sorted({rid for rid, _ in keys})
    offline = json.loads((Path(control["offline_stage1"]) / args.model / "p4/summary.json").read_text())[args.model]
    p5 = json.loads((Path(control["offline_stage1"]) / args.model / "p5/summary.json").read_text())[args.model]
    p9 = json.loads((Path(control["offline_stage1"]) / args.model / "p9/summary.json").read_text())[args.model]
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, len(requests), size=(10000, len(requests))) if requests else None
    summary = {"model": args.model, "repetitions_accepted": sorted(blocks), "requests": requests, "arms": {}}
    for arm in arms:
        rows = data.get(arm, {})
        result = {"fidelity_batch": str(fidelity[arm]) if fidelity[arm] else None,
                  "outputs_identical": all(r["output_identical"] for r in rows.values()) if rows else None,
                  "runs": len(rows)}
        if fidelity[arm]:
            fid = json.loads((fidelity[arm] / "requests.json").read_text())
            result["fidelity"] = {"replay_mismatches": sum(r["replay"]["mismatches"] for r in fid),
                                  "events": sum(r["replay"]["events"] for r in fid),
                                  "outputs_identical": all(r["output_identical"] for r in fid),
                                  "live_sum_m": sum(r["sum_m"] for r in fid), "live_sum_q": sum(r["sum_q"] for r in fid)}
        for metric in ("latency_seconds", "decode_ms_per_token", "first_text_seconds"):
            if arm == BASELINE or not rows or not keys:
                continue
            per_request = np.array([[math.log(rows[(rid, rep)][metric] / data[BASELINE][(rid, rep)][metric])
                                     for rep in sorted({r for q, r in keys if q == rid})] for rid in requests])
            means = per_request.mean(axis=1)
            boots = means[draws].mean(axis=1)
            result[metric + "_ratio"] = {"geometric_mean": float(math.exp(means.mean())),
                                         "ci95": [float(math.exp(np.quantile(boots, 0.025))),
                                                  float(math.exp(np.quantile(boots, 0.975)))],
                                         "per_request": {rid: float(math.exp(m)) for rid, m in zip(requests, means)},
                                         "raw_log_ratios": per_request.tolist()}
        for metric in ("latency_seconds", "decode_ms_per_token", "first_text_seconds"):
            if rows:
                result[metric + "_mean"] = float(np.mean([r[metric] for r in rows.values()]))
        summary["arms"][arm] = result
    # G2 exact components: offline m criteria (primary capacity, native start), unchanged
    # outputs, and the live timing bound on end-to-end latency (identical generated work).
    primary = offline["cells"][next(c for c in offline["cells"] if c.startswith("primary/"))]["arms"]
    offline_rows = {**{arm: primary[arm].get("vs_baseline_mean_m") for arm in primary},
                    "quota_uniform": p5["comparisons"]["uniform"]["vs_global_seeded_mean_m"],
                    "quota_utility": p5["comparisons"]["utility"]["vs_global_seeded_mean_m"],
                    **{name: row["vs_lru_mean_m"] for name, row in p9["comparisons"].items()}}
    p1 = (json.loads((Path(control["p1_seed"]) / args.model / "summary.json").read_text())[args.model]["arms"]
          if control.get("p1_seed") else {})
    if "seed_whole" in p1:
        offline_rows["seed_whole"] = p1["seed_whole"]["vs_native_mean_m"]
    offline_p95 = {**{arm: primary[arm].get("vs_baseline_p95_m") for arm in primary},
                   "quota_uniform": p5["comparisons"]["uniform"]["vs_global_seeded_p95_m"],
                   "quota_utility": p5["comparisons"]["utility"]["vs_global_seeded_p95_m"],
                   **{name: row["vs_lru_p95_m"] for name, row in p9["comparisons"].items()}}
    if "seed_whole" in p1:
        offline_p95["seed_whole"] = p1["seed_whole"]["vs_native_p95_m"]
    for arm, result in summary["arms"].items():
        if arm in (BASELINE, "overlap_floor0"):
            continue
        mean, p95 = offline_rows.get(arm), offline_p95.get(arm)
        m_ok = bool(mean and (mean["reduction"] >= 0.05 and mean["reduction_ci95"][0] > 0
                              or p95 and p95["reduction"] >= 0.10 and p95["reduction_ci95"][0] > 0))
        folds = bool(mean and sum(v > 0 for v in mean["fold_reductions"].values()) >= 4)
        domains = bool(mean and not any(d["ci95"][1] < 0 for d in mean["domains"].values()))
        timing = result.get("latency_seconds_ratio")
        timing_ok = None if not timing else timing["ci95"][1] <= 1.02
        outputs = result["outputs_identical"]
        decision = ("keep" if m_ok and folds and domains and outputs and timing_ok else
                    "reject" if (not m_ok or not folds or not domains or outputs is False or timing_ok is False) else
                    "incomplete")
        result["g2_exact"] = {"offline_mean_m": mean and {k: mean[k] for k in ("reduction", "reduction_ci95")},
                              "m_criterion": m_ok, "four_of_five_folds": folds,
                              "no_confidently_negative_domain": domains, "outputs_unchanged": outputs,
                              "timing_upper_le_1_02": timing_ok, "decision": decision}
    out = study / "analysis" / args.model
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / f"live_summary_{control.get('arm_set', 'stage2') + '_' if control.get('arm_set') else ''}{now()}.json", summary)
    print(json.dumps({arm: {k: v for k, v in r.items() if k in ("g2_exact", "latency_seconds_ratio", "fidelity")
                            and v is not None} for arm, r in summary["arms"].items()}, indent=1, default=str)[:6000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--arm-run", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--prepare", action="store_true", help="Freeze the arm control only")
    parser.add_argument("--arm")
    parser.add_argument("--kind", choices=("fidelity", "timing"))
    parser.add_argument("--label")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--offline", type=Path, help="Stage 1 offline evidence root")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--fidelity-only", action="store_true")
    parser.add_argument("--arm-set", choices=("stage2", "closeout"), default="stage2",
                        help="closeout: floor 0/1 and each model's best exact component, in its own study")
    parser.add_argument("--p1-seed", type=Path, help="P1 seeding offline evidence root (closeout)")
    args = parser.parse_args()
    if args.arm_run:
        arm_run(args)
    elif args.controller or args.prepare:
        if not args.calibration or not args.offline:
            parser.error("--controller/--prepare need --calibration and --offline")
        controller(args) if args.controller else freeze_control(args)
    elif args.analyze:
        analyze(args)
    else:
        parser.error("choose --arm-run, --controller or --analyze")


if __name__ == "__main__":
    main()
