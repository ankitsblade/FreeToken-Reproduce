"""D1/D2 quality for P7 decode-exclusion candidates on the live runtime (plan sections 3.2, 3.3 and 10.1).

One fresh native server per shard runs every arm paired inside it, each request with its
own reset and, for candidates, the frozen D0-fitted deployment decode mask:

- D1 tasks (300, frozen token IDs from the actual-masking v6 protocol): greedy natural-EOS
  generation capped at 256 tokens for the baseline and each task arm; scored afterwards
  by the frozen structured/code scorers.
- D1 reference sequences (100): a 64-token prefix is prefilled, the first of 256 targets is
  scored from the last prefix position and the rest through sequential teacher-forced
  decode. The baseline pass saves its full-vocabulary logits transiently; each KL arm is
  then scored online against them, and the transient logits are deleted after the last
  arm (their per-step NLL/argmax rows remain).
- D2 (AIME 2025-I, 15 problems, pinned revision): baseline and task-arm free-running
  generation, plus KL arms teacher-forced on the baseline trajectory.

``--analyze`` scores tasks and applies the section 10.1 gates with family/cluster bootstrap.
"""

import argparse
import json
import math
import os
import re
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.paths import ROOT, engine_sources, runtime_pythonpath
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready
from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import atomic_json, sha256, snapshot
from experiments.pruning.m_reduction_teacher import NUMERIC_ENV, complete, release, seal
from experiments.pruning.m_reduction_traces import accepted_batch
from experiments.pruning.pruning_resources import monitor_resources, read_resources, summarize_resources, wait_for_quiet

D1_STUDY = ROOT / "results/moe_pruning/study_20261006_v1/additional/actual_masking75_v6"
D1_HASHES = {"tasks.jsonl": "5b3d5aac670b14daabfc37d27a1f4f1182d80035989cf15001fcfdfffd69242e",
             "sequences.jsonl": "008900e85cc43b0bdcb418f5a3363f0bb84c87357ea1d8b84c3fa93ee24f85fd"}
D2_SOURCE = {"repo_id": "opencompass/AIME2025", "revision": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
             "filename": "aime2025-I.jsonl"}
D2_SUFFIX = "Return your final answer as \\boxed{n}, where n is an integer from 0 to 999."
D2_BUDGET = {"gpt_oss_mxfp4": (512, 1536), "qwen_nvfp4": (512, 1536), "gemma_nvfp4": (512, 768)}
D2_TEMPLATE = {"gpt_oss_mxfp4": {"reasoning_effort": "medium"}, "qwen_nvfp4": {"enable_thinking": True},
               "gemma_nvfp4": {}}
END_MARKERS = ("<|return|>", "<|end|>", "<|call|>", "<|im_end|>", "<end_of_turn>", "<eos>", "<|endoftext|>")


def now():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def d1_inputs(model):
    corpus = D1_STUDY / "quality_corpus"
    for name, expected in D1_HASHES.items():
        if sha256(corpus / name) != expected:
            raise ValueError("D1 corpus changed: " + name)
    rendered = json.loads((D1_STUDY / "protocol/rendered.json").read_text())[model]
    tasks = list(json_lines(corpus / "tasks.jsonl"))
    return ([{**task, "input_ids": rendered["tasks"][task["request_id"]]} for task in tasks],
            rendered["sequences"], rendered["chat_template_kwargs"])


def prepare_d2(study, model, checkpoint):
    """Acquire the pinned AIME file once, then freeze each model's rendered input IDs."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    protocol = study / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    source = protocol / "aime2025-I.jsonl"
    if not source.exists():
        path = hf_hub_download(repo_type="dataset", **D2_SOURCE)
        source.write_bytes(Path(path).read_bytes())
        atomic_json(protocol / "d2_acquisition.json", {**D2_SOURCE, "sha256": sha256(source),
                    "acquired_utc": datetime.now(timezone.utc).isoformat(),
                    "licence": "publisher lists MIT; card retained by Hugging Face revision pin"})
    target = protocol / f"d2_rendered_{model}.json"
    if target.exists():
        return json.loads(target.read_text())
    rows = list(json_lines(source))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    budget, generated = D2_BUDGET[model]
    problems, excluded = [], []
    for index, row in enumerate(rows, start=1):
        question = row.get("question") or row.get("problem")
        prompt = f"{question}\n\n{D2_SUFFIX}"
        result = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                               add_generation_prompt=True, **D2_TEMPLATE[model])
        ids = list(result["input_ids"] if hasattr(result, "keys") else result)
        entry = {"problem_id": f"aime2025_I_{index:02d}", "answer": str(row["answer"]).strip(), "input_ids": ids,
                 "prompt": prompt, "max_tokens": generated}
        (problems if len(ids) <= budget else excluded).append(entry)
    frozen = {"model": model, "template_kwargs": D2_TEMPLATE[model], "input_budget": budget,
              "generated_tokens": generated, "source_sha256": sha256(source), "problems": problems,
              "excluded_over_budget": [{"problem_id": e["problem_id"], "tokens": len(e["input_ids"])} for e in excluded],
              "tokenizer_config_sha256": sha256(Path(checkpoint) / "tokenizer_config.json"),
              "rendered_utc": datetime.now(timezone.utc).isoformat()}
    if len(rows) != 15:
        raise ValueError("AIME 2025-I must hold 15 problems")
    atomic_json(target, frozen)
    return frozen


def content(model, text):
    """Visible answer text: GPT-OSS final channel; others with end markers removed."""
    if model == "gpt_oss_mxfp4":
        marker = "<|channel|>final<|message|>"
        text = text.rsplit(marker, 1)[1] if marker in text else ""
    for end in END_MARKERS:
        text = text.split(end, 1)[0]
    return text


def last_boxed(text):
    values = re.findall(r"\\boxed\{\s*([0-9]{1,3})\s*\}", text)
    return int(values[-1]) if values else None


class Session:
    """One fresh native live-runtime server with resource monitoring and a routing arm file."""

    def __init__(self, batch, model, source_study, calibration):
        self.batch = batch
        source_batch = accepted_batch(source_study, model, "primary")
        self.config = json.loads((source_batch / "configuration.json").read_text())
        if sha256(calibration) != self.config["dependencies"]["calibration"]:
            raise ValueError("Calibration differs from the accepted primary capture's")
        self.plan = json.loads((source_study / "protocol/plan.json").read_text())
        self.resource = json.loads((ROOT / self.plan["resource_policy"]).read_text())
        self.arm_file = batch / "arm.json"
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("FREETOKEN_MOE_", "FREETOKEN_HYBRID_", "FREETOKEN_M_REDUCTION_"))}
        self.env.update({"PYTHONPATH": runtime_pythonpath(), "FREETOKEN_BENCHBW_PATH": str(Path(calibration).resolve()),
                         "FREETOKEN_MOE_ROUTING_ARM": str(self.arm_file), "FREETOKEN_M_REDUCTION_FILL_FLOOR": "1",
                         "FREETOKEN_M_REDUCTION_TOKEN_PROMPTS": "1", "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1",
                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **self.resource["allocator_environment"],
                         **NUMERIC_ENV[self.plan["numeric_arm"]]})
        self.port = free_port()
        self.origin = f"http://127.0.0.1:{self.port}"
        self.command = list(self.config["command"])
        self.command[self.command.index("--port") + 1] = str(self.port)
        self.phase = {"name": "startup"}
        self.boundaries = {"startup_begin": read_resources(-1)}

    def __enter__(self):
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("GPU is occupied")
        self.log = (self.batch / "server.log").open("wb")
        self.proc = subprocess.Popen(self.command, cwd=ROOT, env=self.env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.stop = threading.Event()
        self.monitor = threading.Thread(target=monitor_resources,
                                        args=(self.proc, self.stop, self.batch / "resources.jsonl", self.phase))
        self.monitor.start()
        wait_ready(self.origin, self.proc, str(self.batch / "server.log"), 1800)
        self.model_id = get_json(self.origin + "/v1/models")["data"][0]["id"]
        return self

    def warm(self, prompt):
        self.phase["name"] = "warmup"
        self.arm({"armed": True, "capture_id": "warmup", "request_id": "warmup", "observe": False,
                  "directory": str(self.batch / "warmup"), "prompt_tokens": len(prompt), "require_pinned_banks": True})
        complete(self.origin, {"model": self.model_id, "prompt": prompt, "max_tokens": 64, **self.plan["sampling"]})
        self.arm({"armed": False})
        self.phase["name"] = "settling"
        self.boundaries["ready"] = read_resources(self.proc.pid)
        self.boundaries["inference_begin"] = wait_for_quiet(self.proc, self.resource)
        self.phase["name"] = "inference"

    def arm(self, metadata):
        atomic_json(self.arm_file, metadata)

    def request(self, metadata, body):
        self.arm(metadata)
        try:
            return complete(self.origin, {"model": self.model_id, **self.plan["sampling"], **body})
        finally:
            self.arm({"armed": False})

    def __exit__(self, kind, value, traceback):
        if kind is None:
            self.boundaries["inference_end"] = read_resources(self.proc.pid)
            atomic_json(self.batch / "resource_boundaries.json", self.boundaries)
        self.phase["name"] = "shutdown"
        stop_server(self.proc)
        self.stop.set()
        self.monitor.join()
        self.log.close()
        if kind is None:
            self.summary = summarize_resources(list(json_lines(self.batch / "resources.jsonl")), self.boundaries,
                                               self.resource)
            atomic_json(self.batch / "resource_summary.json", self.summary)
        return False


def masks_for(batch, deploy, arms):
    paths = {}
    (batch / "masks").mkdir(exist_ok=True)
    for arm in arms:
        if arm == "baseline":
            continue
        path = batch / "masks" / (arm + ".json")
        atomic_json(path, {"arm": arm, "layers": deploy[arm + "_removed"], "deploy_sha256": deploy["sha256"],
                           "semantics": "decode-only logits -inf before native top-k; fitted on all 90 D0 captures"})
        paths[arm] = str(path)
    return paths


def generate(session, directory, request_id, prompt, max_tokens, mask):
    directory.mkdir(parents=True)
    metadata = {"armed": True, "capture_id": directory.parent.name + "_" + directory.name, "request_id": request_id,
                "directory": str(directory), "prompt_tokens": len(prompt), "require_pinned_banks": True,
                "observe": False, "record_tokens": True}
    if mask:
        metadata["decode_mask"] = mask
    response = session.request(metadata, {"prompt": prompt, "max_tokens": max_tokens})
    rows = [line.split() for line in (directory / "sampled_tokens.txt").read_text().splitlines()]
    tokens = [int(t) for phase, t in rows if phase == "prefill"][-1:] + [int(t) for phase, t in rows if phase == "decode"]
    atomic_json(directory / "response.json", {**response, "generated_token_ids": tokens})
    return response


def teacher(session, directory, request_id, prompt, forced, mask=None, reference=None, save_logits=False):
    directory.mkdir(parents=True)
    metadata = {"armed": True, "capture_id": directory.parent.name + "_" + directory.name, "request_id": request_id,
                "directory": str(directory), "prompt_tokens": len(prompt), "require_pinned_banks": True,
                "full_scores": False, "prompt_token_ids": prompt, "forced_tokens": forced}
    if mask:
        metadata["decode_mask"] = mask
    if reference:
        metadata["kl_reference"] = {"path": str(reference / "logits.bin"), "header": str(reference / "logits_header.json")}
    if save_logits:
        metadata["save_logits"] = True
    response = session.request(metadata, {"prompt": prompt, "max_tokens": len(forced), "ignore_eos": True})
    rows = list(json_lines(directory / "teacher.jsonl.partial"))
    if [row["step"] for row in rows] != list(range(len(forced))) or [row["target"] for row in rows] != forced:
        raise ValueError(directory.name + ": teacher-forced steps are incomplete")
    cache = [row for row in json_lines(directory / "cache.jsonl.partial") if row["phase"] == "decode"]
    result = {"targets": len(forced), "nll": [row["nll"] for row in rows],
              "argmax": [row["model_argmax"] for row in rows], "decode_events": len(cache),
              "sum_m": sum(row["m"] for row in cache), "sum_q": sum(row["q"] for row in cache),
              "completion_tokens": response["usage"]["completion_tokens"]}
    if reference:
        result.update({"kl": [row["kl_baseline_candidate"] for row in rows],
                       "top1_agree": [row["top1_agree"] for row in rows]})
    return result


def run_shard(args):
    study = args.study.resolve()
    deploy = json.loads(args.deploy.read_text())
    deploy["sha256"] = sha256(args.deploy)
    if deploy["model"] != args.model:
        raise ValueError("deployment artefact model differs")
    task_arms, kl_arms = args.task_arms.split(","), args.kl_arms.split(",")
    if task_arms[0] != "baseline":
        raise ValueError("task arms must start with the baseline")
    if args.dataset == "d1":
        tasks, sequences, kwargs = d1_inputs(args.model)
        tasks = tasks[args.shard::args.shards]
        sequences = sequences[args.shard::args.shards]
        if args.limit:
            tasks, sequences = tasks[:args.limit], sequences[:args.limit]
    else:
        frozen = json.loads((study / "protocol" / f"d2_rendered_{args.model}.json").read_text())
        problems = frozen["problems"][args.shard::args.shards][:args.limit or None]
    batch = study / "quality" / args.model / args.dataset / f"shard{args.shard:02d}of{args.shards:02d}" / (now() + "_" + uuid.uuid4().hex[:6])
    batch.mkdir(parents=True)
    masks = masks_for(batch, deploy, sorted(set(task_arms) | set(kl_arms)))
    atomic_json(batch / "configuration.json", {
        "dataset": args.dataset, "model": args.model, "shard": args.shard, "shards": args.shards,
        "task_arms": task_arms, "kl_arms": kl_arms, "deploy": str(args.deploy.resolve()), "deploy_sha256": deploy["sha256"],
        "inference_hashes": {name: sha256(path) for name, path in engine_sources().items()},
        "worker_source_sha256": sha256(Path(__file__)),
        "runtime_patch_sha256": sha256(ROOT / "runtime/patches/m-reduction-live.patch"),
        "source_snapshot": snapshot(batch / "source_snapshot"), "limit": args.limit,
        "transient_baseline_logits": "deleted after the last KL arm of each item; per-step rows retained"})
    done = []
    with Session(batch, args.model, args.source_study.resolve(), args.calibration) as session:
        warm = json.loads((args.source_study / "protocol/rendered_inputs.json").read_text())[args.model]["math_000"]
        session.warm(warm)
        try:
            if args.dataset == "d1":
                for task in tasks:
                    for arm in task_arms:
                        response = generate(session, batch / "tasks" / task["request_id"] / arm, task["request_id"],
                                            task["input_ids"], 256, masks.get(arm))
                    done.append(task["request_id"])
                    atomic_json(batch / "completed.json", done)
                    print(f"D1 task {args.model} shard {args.shard}: {len(done)}/{len(tasks) + len(sequences)}", flush=True)
                items = [(s["sequence_id"], s["input_ids"][:s["prefix_tokens"]], s["target_ids"]) for s in sequences]
            else:
                items = []
                for problem in problems:
                    for arm in task_arms:
                        response = generate(session, batch / "problems" / problem["problem_id"] / arm,
                                            problem["problem_id"], problem["input_ids"], problem["max_tokens"], masks.get(arm))
                        if arm == "baseline":
                            trajectory = json.loads((batch / "problems" / problem["problem_id"] / arm /
                                                     "response.json").read_text())["generated_token_ids"]
                    items.append((problem["problem_id"], problem["input_ids"], trajectory))
                    done.append(problem["problem_id"])
                    atomic_json(batch / "completed.json", done)
                    print(f"D2 problem {args.model} shard {args.shard}: {len(done)}/{len(problems)}", flush=True)
            for item_id, prompt, forced in items:
                root = batch / "teacher" / item_id
                results = {"baseline": teacher(session, root / "baseline", item_id, prompt, forced, save_logits=True)}
                for arm in kl_arms:
                    results[arm] = teacher(session, root / arm, item_id, prompt, forced, mask=masks[arm],
                                           reference=root / "baseline")
                logits = root / "baseline" / "logits.bin"
                results["baseline"]["transient_logits"] = {"sha256": sha256(logits), "bytes": logits.stat().st_size}
                logits.unlink()
                for name in ["baseline", *kl_arms]:
                    seal(root / name)
                    release(root / name)
                atomic_json(root / "result.json", results)
                done.append(item_id)
                atomic_json(batch / "completed.json", done)
                print(f"{args.dataset} teacher {args.model} shard {args.shard}: {item_id} "
                      + " ".join(f"{arm} kl={np.mean(results[arm]['kl']):.4f}" for arm in kl_arms), flush=True)
        except BaseException as exc:
            atomic_json(batch / "failure.json", {"error": repr(exc), "completed": done})
            raise
    status = "complete" if session.summary["status"] == "passed" else "rejected"
    atomic_json(batch / "completion.json", {"status": status, "items": len(done), "resources": session.summary,
                                            "limit": args.limit})
    print(json.dumps({"batch": str(batch), "status": status, "reasons": session.summary["reasons"]}), flush=True)
    if status != "complete":
        raise SystemExit(2)


def score_task(model, task, response):
    text = content(model, response["choices"][0]["text"])
    if task["workload"] != "code":
        from experiments.pruning.pruning_quality_score import score_structured
        return score_structured(task, text)
    payload = json.dumps({"text": text, "reference": task["reference"]})
    try:
        proc = subprocess.run(["uv", "run", "--no-sync", "--project", str(ROOT), "python", "-m",
                               "experiments.pruning.pruning_quality_score", "--code-worker"],
                              input=payload, text=True, capture_output=True, timeout=8, cwd=ROOT)
        if proc.returncode:
            return {"score": 0.0, "status": "restricted_worker_failure", "error": proc.stderr[-1000:], "unsafe_choice": False}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"score": 0.0, "status": "restricted_worker_timeout", "unsafe_choice": False}


def cluster_bootstrap(values, groups, seed=20261014):
    """Mean over items with 10,000 cluster resamples; returns point and 95% interval."""
    keys = sorted(set(groups))
    totals = np.array([sum(v for v, g in zip(values, groups) if g == k) for k in keys])
    counts = np.array([groups.count(k) for k in keys])
    draws = np.random.default_rng(seed).integers(0, len(keys), size=(10000, len(keys)))
    samples = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"mean": float(totals.sum() / counts.sum()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "clusters": len(keys), "items": int(counts.sum())}


def accepted_shards(study, model, dataset):
    found = {}
    for shard_dir in sorted((study / "quality" / model / dataset).glob("shard*")):
        for batch in sorted(shard_dir.iterdir(), reverse=True):
            done = batch / "completion.json"
            if done.is_file() and json.loads(done.read_text())["status"] == "complete" and not json.loads(done.read_text()).get("limit"):
                found[shard_dir.name] = batch
                break
    return found


def analyze(args):
    study = args.study.resolve()
    out = {"model": args.model, "dataset": args.dataset, "utc": now()}
    shards = accepted_shards(study, args.model, args.dataset)
    out["shards"] = {k: str(v) for k, v in shards.items()}
    teacher_rows = {}
    for batch in shards.values():
        for result in (batch / "teacher").glob("*/result.json"):
            teacher_rows[result.parent.name] = json.loads(result.read_text())
    if args.dataset == "d1":
        tasks, sequences, _ = d1_inputs(args.model)
        groups = {s["sequence_id"]: s["group_id"] for s in sequences}
        scored = {}
        for batch in shards.values():
            for task in tasks:
                for arm_dir in sorted((batch / "tasks" / task["request_id"]).glob("*")) if (batch / "tasks" / task["request_id"]).is_dir() else []:
                    response = json.loads((arm_dir / "response.json").read_text())
                    scored.setdefault(arm_dir.name, {})[task["request_id"]] = {
                        **score_task(args.model, task, response), "finish_reason": response["choices"][0].get("finish_reason"),
                        "completion_tokens": response["usage"]["completion_tokens"]}
        out["task_counts"] = {arm: len(rows) for arm, rows in scored.items()}
        out["tasks"] = {}
        base = scored.get("baseline", {})
        for arm, rows in scored.items():
            arm_out = {}
            for workload in ("math", "code", "incident"):
                ids = [t["request_id"] for t in tasks if t["workload"] == workload and t["request_id"] in rows and t["request_id"] in base]
                fam = [next(t["group_id"] for t in tasks if t["request_id"] == i) for i in ids]
                entry = {"score": cluster_bootstrap([rows[i]["score"] for i in ids], fam) if ids else None,
                         "truncations": sum(rows[i]["finish_reason"] == "length" for i in ids),
                         "invalid": sum(rows[i]["status"] != "scored" for i in ids),
                         "unsafe_choices": sum(bool(rows[i].get("unsafe_choice")) for i in ids)}
                if arm != "baseline" and ids:
                    loss = cluster_bootstrap([base[i]["score"] - rows[i]["score"] for i in ids], fam)
                    entry.update({"baseline_minus_candidate": loss, "loss_gate_upper_le_0_02": loss["ci95"][1] <= 0.02,
                                  "correct_to_wrong": sum(base[i]["score"] == 1 and rows[i]["score"] < 1 for i in ids),
                                  "wrong_to_correct": sum(base[i]["score"] < 1 and rows[i]["score"] == 1 for i in ids),
                                  "new_unsafe": sum(bool(rows[i].get("unsafe_choice")) and not base[i].get("unsafe_choice") for i in ids),
                                  "baseline_competent": (cluster_bootstrap([base[i]["score"] for i in ids], fam)["mean"] >= 0.2)})
                arm_out[workload] = entry
            out["tasks"][arm] = arm_out
        atomic_json(study / "analysis" / args.model / f"d1_task_scores_{now()}.json", scored)
    else:
        frozen = json.loads((study / "protocol" / f"d2_rendered_{args.model}.json").read_text())
        answers = {p["problem_id"]: int(p["answer"]) for p in frozen["problems"]}
        rows = {}
        for batch in shards.values():
            for arm_dir in (batch / "problems").glob("*/*"):
                response = json.loads((arm_dir / "response.json").read_text())
                value = last_boxed(content(args.model, response["choices"][0]["text"]))
                rows.setdefault(arm_dir.name, {})[arm_dir.parent.name] = {
                    "answer": value, "correct": value == answers[arm_dir.parent.name],
                    "finish_reason": response["choices"][0].get("finish_reason"),
                    "completion_tokens": response["usage"]["completion_tokens"]}
        out["problems"] = rows
        out["eligible"] = len(frozen["problems"])
        out["excluded_over_budget"] = frozen["excluded_over_budget"]
        base = rows.get("baseline", {})
        out["summary"] = {arm: {"correct": sum(r["correct"] for r in v.values()), "n": len(v),
                                "truncated": sum(r["finish_reason"] == "length" for r in v.values()),
                                "baseline_correct_to_wrong": sum(base.get(p, {}).get("correct") and not r["correct"] for p, r in v.items()),
                                "wrong_to_correct": sum((not base.get(p, {}).get("correct")) and r["correct"] for p, r in v.items())}
                          for arm, v in rows.items()}
    # Teacher-forced divergence and likelihood.
    if teacher_rows:
        if args.dataset == "d1":
            group_of = groups
        else:
            group_of = {item: item for item in teacher_rows}
        items = sorted(teacher_rows)
        out["teacher"] = {}
        arms = [a for a in next(iter(teacher_rows.values())) if a != "baseline"]
        base_nll = [float(np.mean(teacher_rows[i]["baseline"]["nll"])) for i in items]
        for arm in arms:
            kl = [float(np.mean(teacher_rows[i][arm]["kl"])) for i in items]
            agree = [float(np.mean(teacher_rows[i][arm]["top1_agree"])) for i in items]
            nll = [float(np.mean(teacher_rows[i][arm]["nll"])) for i in items]
            g = [group_of[i] for i in items]
            kl_b, agree_b = cluster_bootstrap(kl, g), cluster_bootstrap(agree, g)
            nll_b, base_b = cluster_bootstrap(nll, g), cluster_bootstrap(base_nll, g)
            # Perplexity ratio exp(mean NLL difference), cluster bootstrap on paired differences.
            diff = cluster_bootstrap([a - b for a, b in zip(nll, base_nll)], g)
            m = sum(teacher_rows[i][arm]["sum_m"] for i in items) / sum(teacher_rows[i][arm]["decode_events"] for i in items)
            m0 = sum(teacher_rows[i]["baseline"]["sum_m"] for i in items) / sum(teacher_rows[i]["baseline"]["decode_events"] for i in items)
            out["teacher"][arm] = {"kl": kl_b, "kl_gate_upper_le_0_02": kl_b["ci95"][1] <= 0.02,
                                   "top1_agreement": agree_b, "top1_gate_lower_ge_0_98": agree_b["ci95"][0] >= 0.98,
                                   "nll": nll_b, "baseline_nll": base_b,
                                   "perplexity_ratio": {"point": math.exp(diff["mean"]),
                                                        "ci95": [math.exp(v) for v in diff["ci95"]]},
                                   "perplexity_gate_upper_le_1_02": math.exp(diff["ci95"][1]) <= 1.02,
                                   "teacher_mean_m": m, "baseline_teacher_mean_m": m0,
                                   "teacher_m_reduction": 1 - m / m0 if m0 else None}
    target = study / "analysis" / args.model / f"{args.dataset}_quality_{now()}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, out)
    print(json.dumps({k: v for k, v in out.items() if k in ("tasks", "teacher", "summary")}, indent=1, default=str)[:8000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=("gpt_oss_mxfp4", "qwen_nvfp4", "gemma_nvfp4"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--deploy", type=Path)
    parser.add_argument("--dataset", choices=("d1", "d2"), default="d1")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--task-arms", default="baseline,arm1")
    parser.add_argument("--kl-arms", default="arm1,r75")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-d2", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.prepare_d2:
        source = json.loads((args.source_study / "protocol/plan.json").read_text())["models"][args.model]
        frozen = prepare_d2(args.study.resolve(), args.model, ROOT / source["path"])
        print(json.dumps({"eligible": len(frozen["problems"]), "excluded": frozen["excluded_over_budget"]}))
    elif args.analyze:
        analyze(args)
    else:
        if not args.calibration or not args.deploy:
            parser.error("shard runs need --calibration and --deploy")
        run_shard(args)


if __name__ == "__main__":
    main()
