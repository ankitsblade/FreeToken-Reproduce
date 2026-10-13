"""Qualify and execute paired actual FreeToken inference with 25% routed experts excluded."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path
from experiments.paths import ENGINE_SOURCE, runtime_pythonpath, engine_sources

import numpy as np

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import ROOT, atomic_json, sha256, snapshot
from experiments.pruning.pruning_analysis import load_model, retained_mask
from experiments.pruning.pruning_resources import monitor_resources, read_resources, summarize_resources, wait_for_quiet

sys.path.insert(0, str(ROOT / "experiments/baselines"))
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready

SOURCE_FILES = ("experiments/pruning/run_masked_inference75.py", "experiments/pruning/masked_inference_probe.py",
                "experiments/pruning/native_pruning_probe.py", "experiments/pruning/pruning_resources.py",
                "experiments/pruning/pruning_quality_score.py", "experiments/pruning/build_pruning_quality.py")
MODEL_ORDER = ("gemma_nvfp4", "gpt_oss_mxfp4", "qwen_nvfp4")
POOLS = ("mixed", "math", "code", "incident")
QUALITY_SHARDS = 12


def uv(*args):
    return ["uv", "run", "--no-sync", "--project", str(ROOT), "python", *args]


def hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def inference_hashes():
    return {name: sha256(path) for name, path in engine_sources().items()}


def request(origin, endpoint, body):
    req = urllib.request.Request(origin + endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"FreeToken HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def prepare(study, parent):
    from transformers import AutoTokenizer

    protocol_dir = study / "protocol"
    protocol_dir.mkdir(exist_ok=False)
    gemma = parent / "additional/gemma_v1"
    models = json.loads((parent / "protocol/models.json").read_text())
    models["gemma_nvfp4"] = json.loads((gemma / "protocol/models.json").read_text())["gemma_nvfp4"]
    models = {key: models[key] for key in MODEL_ORDER}
    atomic_json(protocol_dir / "models.json", models)
    corpus = study / "quality_corpus"
    manifest = json.loads((corpus / "manifest.json").read_text())
    for name, expected in manifest["hashes"].items():
        if sha256(corpus / name) != expected:
            raise ValueError("quality corpus changed after its freeze")
    original = list(json_lines(parent / "protocol/dataset.jsonl"))
    tasks = list(json_lines(corpus / "tasks.jsonl"))
    sequence_rows = list(json_lines(corpus / "sequences.jsonl"))
    if len(tasks) != 300 or len(sequence_rows) != 100:
        raise ValueError("quality partition requires all 300 tasks and 100 reference sequences")
    if len({row["request_id"] for row in tasks}) != 300 or len({row["sequence_id"] for row in sequence_rows}) != 100:
        raise ValueError("quality request IDs must be unique before partitioning")
    shards = {}
    for index in range(QUALITY_SHARDS):
        task_ordinals = list(range(index, len(tasks), QUALITY_SHARDS))
        sequence_ordinals = list(range(index, len(sequence_rows), QUALITY_SHARDS))
        shards[str(index)] = dict(shard_index=index, task_ordinals=task_ordinals, sequence_ordinals=sequence_ordinals,
                                  task_ids=[tasks[i]["request_id"] for i in task_ordinals],
                                  sequence_ids=[sequence_rows[i]["sequence_id"] for i in sequence_ordinals],
                                  captures=3*len(task_ordinals)+6*len(sequence_ordinals))
    atomic_json(protocol_dir / "quality_shards.json", dict(count=QUALITY_SHARDS, shards=shards,
                assignment="frozen original ordinal modulo shard count; paired arms remain in the same fresh process",
                resource_scope="unchanged limits per fresh inference batch; aggregate all shards without a whole-model 128 MiB swap cap"))
    if set(row["prompt_sha256"] for row in tasks) & set(row["prompt_sha256"] for row in original):
        raise ValueError("quality prompt overlaps routing-calibration prompts")
    pilots = [dict(request_id=r["request_id"], workload=r["workload"], prompt=r["prompt_text"]) for r in original if r["pilot_flag"]]
    atomic_json(protocol_dir / "pilots.json", pilots)
    mask_records, rendered = {}, {}
    for key, model in models.items():
        reference = gemma if key == "gemma_nvfp4" else parent
        captures = load_model(reference, key, model)
        mask_records[key] = {}
        target = study / "masks" / key
        target.mkdir(parents=True)
        for pool in POOLS:
            counts = np.zeros((model["layers"], model["experts"]), dtype=np.int64)
            used = []
            for rid, capture in captures.items():
                if pool == "mixed" or capture["manifest"]["request"]["workload"] == pool:
                    counts += capture["phases"]["decode"]["counts"]
                    used.append(rid)
            mask = retained_mask(counts, .75)
            path = target / (pool + "75.npz")
            np.savez_compressed(path, mask=mask, counts=counts)
            mask_records[key][pool] = dict(path=str(path), sha256=sha256(path), retained_per_layer=int(mask.sum(axis=1)[0]),
                                          training_requests=used, ranking_phase="decode", used_for_both_phases=True)
        np.savez_compressed(target / "all100.npz", mask=np.ones((model["layers"],model["experts"]),dtype=bool))
        del captures
        gc.collect()
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / model["path"]), local_files_only=True)
        kwargs = {"reasoning_effort": "low"} if key == "gpt_oss_mxfp4" else {"enable_thinking": False}
        inputs = {}
        for row in [*pilots, *tasks]:
            result = tokenizer.apply_chat_template([{"role":"user","content":row["prompt"]}], tokenize=True,
                                                  add_generation_prompt=True, **kwargs)
            ids = list(result["input_ids"] if hasattr(result,"keys") else result)
            if len(ids)>1024:
                raise ValueError("quality prompt exceeds supported cap")
            inputs[row["request_id"]] = ids
        sequences = []
        for row in json_lines(corpus / "sequences.jsonl"):
            raw_ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if len(raw_ids)<320:
                raise ValueError("reference sequence cannot supply 64-prefix and 256 scored tokens")
            for end in range(310, 330):
                prompt_text=tokenizer.decode(raw_ids[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                ids=tokenizer.encode(prompt_text, add_special_tokens=True)
                if len(ids)==320:
                    break
            else:
                raise ValueError("cannot freeze a supported text prompt with exactly 320 native input IDs")
            sequences.append(dict(sequence_id=row["sequence_id"], group_id=row["group_id"], input_ids=ids,
                                  prompt_text=prompt_text, prefix_tokens=64, target_ids=ids[64:320],
                                  tokenization="supported raw text completion; native add_special_tokens=True; exact worker IDs checked"))
        rendered[key] = dict(chat_template_kwargs=kwargs, tasks=inputs, sequences=sequences)
        source_policy = (gemma / "protocol/resource_policy.json" if key == "gemma_nvfp4" else
                         parent / ("additional/qwen_allocator_v6/protocol/resource_policy.json" if key == "qwen_nvfp4" else "protocol/resource_policy_v5.json"))
        policy = json.loads(source_policy.read_text())
        policy["allocator_environment"].update(MALLOC_MMAP_THRESHOLD_="131072", MALLOC_TRIM_THRESHOLD_="131072")
        atomic_json(protocol_dir / (key + "_resources.json"), policy)
        del tokenizer
    atomic_json(protocol_dir / "masks.json", mask_records)
    atomic_json(protocol_dir / "rendered.json", rendered)
    atomic_json(protocol_dir / "amendment.json", dict(status="frozen_before_actual_masked_outputs", retention=.75,
                user_authorization="Okay fix that and after this can we also run the actual inference with 25% pruned to see if it holds",
                scope="exploratory actual M75 despite failed original offline activation gates",
                parent_closeout_sha256=sha256(parent / "closeout75/artifact_manifest.json"),
                original_closeout_preserved=True, kimi="excluded", deferred60_50="not activated",
                arms=["baseline", "all100_control", "mixed75", "same_workload75"],
                nll_arms=["baseline", "all100_control", "mixed75", "math75", "code75", "incident75"],
                task_counts_per_model={"baseline":300,"mixed75":300,"same_workload75":300},
                conditional_nll_tokens_per_arm=25600, task_loss_margin=.02, perplexity_relative_increase_margin=.02,
                numerical_control_maximum_mean_nll_difference=1e-5,
                implementation="native logits eligibility with fixed original k and expert IDs; all shared branches preserved",
                allocation="original weights, cache slots and KV budgets; no weight-compaction or RAM-saving claim",
                generation="greedy natural EOS capped at 256; thinking off for Qwen/Gemma, low effort for GPT-OSS; distinct from routing-capture arm",
                startup_diagnostic_warmup="one fixed 320-token sequence under baseline, all100 and all four masks before settling/resource measurement",
                diagnostic_memory="lossless baseline logits/layer outputs cached on GPU; reusable pinned artifact buffers; original expert and KV budgets unchanged",
                artifact_io="Linux O_DIRECT with aligned pinned buffers and 4096-byte NPY v2 headers; no page-cache fallback",
                quality_execution="12 serial fresh-process shards; every task and reference sequence assigned exactly once; all paired arms in one shard",
                resource_amendment="128 MiB swap limit remains per fresh batch; total model collection may exceed it across accepted shards; no cumulative whole-model swap guarantee",
                performance="separate untraced fresh-process stage only after synthetic quality margins are met",
                corpus_manifest_sha256=sha256(corpus / "manifest.json"), source_hashes=hashes(),
                all100_mask_hashes={key:sha256(study / "masks" / key / "all100.npz") for key in models},
                inference_hashes=inference_hashes(),
                protocol_hashes={path.name:sha256(path) for path in protocol_dir.iterdir() if path.is_file()},
                checkpoint_metadata_hashes={key:{name:sha256(ROOT / model["path"] / name) for name in
                    ("config.json", "model.safetensors.index.json", "tokenizer_config.json")} for key,model in models.items()}))
    snapshot(study / "source_snapshot")


def assert_frozen(study):
    frozen = json.loads((study / "protocol/amendment.json").read_text())
    if frozen["source_hashes"] != hashes():
        raise ValueError("actual-inference implementation changed after protocol freeze")
    if frozen["inference_hashes"] != inference_hashes():
        raise ValueError("FreeToken implementation changed after actual-inference freeze")
    corpus = study / "quality_corpus"
    if sha256(corpus / "manifest.json") != frozen["corpus_manifest_sha256"]:
        raise ValueError("quality corpus manifest changed after freeze")
    for name, expected in json.loads((corpus / "manifest.json").read_text())["hashes"].items():
        if sha256(corpus / name) != expected:
            raise ValueError("quality corpus changed after freeze")
    for name, expected in frozen["protocol_hashes"].items():
        if sha256(study / "protocol" / name) != expected:
            raise ValueError("actual-inference controls changed after freeze")
    models=json.loads((study / "protocol/models.json").read_text())
    for key,records in frozen["checkpoint_metadata_hashes"].items():
        for name,expected in records.items():
            if sha256(ROOT / models[key]["path"] / name)!=expected:
                raise ValueError("actual-inference checkpoint metadata changed")
    masks = json.loads((study / "protocol/masks.json").read_text())
    for key, expected in frozen["all100_mask_hashes"].items():
        if sha256(study / "masks" / key / "all100.npz") != expected:
            raise ValueError("all100 control mask changed")
    for model in masks.values():
        for record in model.values():
            if sha256(Path(record["path"])) != record["sha256"]:
                raise ValueError("deployment mask changed")


def accepted_quality_batches(study, key):
    plan = json.loads((study / "protocol/quality_shards.json").read_text())
    batches, task_batches, sequence_batches, seen = [], {}, {}, set()
    for path in sorted((study / "runs" / key).glob("*/completion.json")):
        completion = json.loads(path.read_text())
        if completion.get("stage") != "quality" or completion.get("resource_status") != "passed":
            continue
        index = completion["shard_index"]
        if index in seen or str(index) not in plan["shards"]:
            raise ValueError("accepted quality shard is duplicated or outside the frozen plan")
        spec = plan["shards"][str(index)]
        if completion["status"] != "complete" or completion["source_hashes"] != hashes() or completion["captures"] != spec["captures"]:
            raise ValueError("accepted quality shard has invalid status, source hashes or count")
        if completion["task_ids"] != spec["task_ids"] or completion["sequence_ids"] != spec["sequence_ids"]:
            raise ValueError("accepted quality shard membership differs from the frozen plan")
        seen.add(index)
        batches.append(path.parent)
        for values, mapping in ((spec["task_ids"], task_batches), (spec["sequence_ids"], sequence_batches)):
            for value in values:
                if value in mapping:
                    raise ValueError("quality request appears in more than one accepted shard")
                mapping[value] = path.parent
    if seen != set(range(plan["count"])) or len(task_batches) != 300 or len(sequence_batches) != 100:
        raise ValueError("all frozen quality shards must be accepted before scoring or closeout")
    return batches, task_batches, sequence_batches


def body(model_id, row, rendered, max_tokens=256):
    return {"model":model_id,"messages":[{"role":"user","content":row["prompt"]}], "stream":False,
            "max_tokens":max_tokens,"temperature":0.,"top_p":1.,"top_k":-1,"ignore_eos":False,
            "chat_template_kwargs":rendered["chat_template_kwargs"]}


def run_batch(study, parent, key, stage, batch_id, shard_index=None):
    assert_frozen(study)
    spec = None
    if stage == "quality":
        plan = json.loads((study / "protocol/quality_shards.json").read_text())
        if shard_index is None or str(shard_index) not in plan["shards"]:
            raise ValueError("quality requires a shard index from the frozen partition")
        spec = plan["shards"][str(shard_index)]
    elif shard_index is not None:
        raise ValueError("pilot does not accept a quality shard index")
    model = json.loads((study / "protocol/models.json").read_text())[key]
    rendered = json.loads((study / "protocol/rendered.json").read_text())[key]
    masks = json.loads((study / "protocol/masks.json").read_text())[key]
    policy = json.loads((study / "protocol" / (key + "_resources.json")).read_text())
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
        raise RuntimeError("actual inference requires an idle GPU")
    target = study / "runs" / key / batch_id
    target.mkdir(parents=True, exist_ok=False)
    control = target / "control.json"
    all_mask = study / "masks" / key / "all100.npz"
    profile = ROOT / "results/baseline/calibration_repeat5/profile-run-04.json"
    env = {k:v for k,v in os.environ.items() if not k.startswith(("FREETOKEN_MOE_","FREETOKEN_HYBRID_","FREETOKEN_PRUNING_"))}
    cuda = subprocess.check_output(uv("-c",'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13")'),text=True).strip()
    env.update(CUDA_HOME=cuda,CC="/usr/bin/gcc",PATH=cuda+"/bin:"+env["PATH"],
               LD_LIBRARY_PATH=cuda+"/lib:"+env.get("LD_LIBRARY_PATH",""),PYTHONPATH=runtime_pythonpath(),
               FREETOKEN_BENCHBW_PATH=str(profile),FREETOKEN_DISABLE_OVERLAP_SCHEDULING="1",
               FREETOKEN_PRUNING_CONTROL=str(control),FREETOKEN_PRUNING_MASK=str(all_mask))
    env.update(policy["allocator_environment"])
    port=free_port()
    origin=f"http://127.0.0.1:{port}"
    cmd=uv("-m","experiments.pruning.masked_inference_probe","serve","--model",str(ROOT / model["path"]),"--host","127.0.0.1","--port",str(port),
           "--moe-strategy","hybrid","--max-running-requests","1","--max-seq-len-override","1280" if key=="gemma_nvfp4" else "2048",
           "--max-prefill-length","1024","--memory-ratio","0.8" if key=="gemma_nvfp4" else "0.9","--cuda-graph-max-bs","0",
           "--moe-hybrid-max-fetch","-1","--moe-cache-size",str(model["cache_slots"]),"--cache-type","naive","--expert-load","serial")
    if key != "gpt_oss_mxfp4": cmd.append("--text-model-only")
    atomic_json(target / "configuration.json",dict(command=cmd,model=model,stage=stage,policy=policy,source_hashes=hashes(),
                shard_index=shard_index,shard_specification=spec,
                dataset_sha256=sha256(study / "quality_corpus/tasks.jsonl"),calibration_sha256=sha256(profile)))
    completed, boundaries, phase = [], {}, {"name":"startup"}
    progress_path = target / "progress.json"

    def generate(row, arm, mask, pilot=False):
        directory = target / row["request_id"] / arm
        directory.mkdir(parents=True, exist_ok=False)
        metadata=dict(armed=True,mask=mask,directory=str(directory),audit_routes=pilot,require_pinned_banks=True)
        atomic_json(control,metadata)
        started=time.monotonic()
        response=request(origin,"/v1/chat/completions",body(model_id,row,rendered))
        elapsed=time.monotonic()-started
        if response["usage"]["prompt_tokens"] != len(rendered["tasks"][row["request_id"]]):
            raise ValueError("actual quality input token count differs from frozen rendering")
        atomic_json(directory / "response.json",response)
        atomic_json(directory / "measurement.json",dict(arm=arm,request_id=row["request_id"],workload=row["workload"],
                    mask=mask,elapsed_seconds=elapsed,finish_reason=response["choices"][0]["finish_reason"],
                    interpretation="qualification or exploratory quality; not a performance benchmark",
                    response_sha256=sha256(directory / "response.json")))
        if pilot:
            audit=json.loads((directory / "native_route_audit.json").read_text())
            expected_positions=response["usage"]["prompt_tokens"]+response["usage"]["completion_tokens"]-1
            if set(audit)!={str(i) for i in range(model["layers"])} or any(value!=expected_positions for value in audit.values()):
                raise ValueError("actual native route audit misses a token/layer")
        completed.append(str(directory.relative_to(target)))
        atomic_json(progress_path,dict(status="running",completed=completed))
        return response, directory

    def teacher(sequence, arm, mask):
        directory=target / sequence["sequence_id"] / arm
        directory.mkdir(parents=True,exist_ok=False)
        atomic_json(control,dict(armed=True,mask=mask,directory=str(directory),audit_routes=False,teacher_forcing=True,require_pinned_banks=True,
                    arm=arm,baseline_directory=str(target / sequence["sequence_id"] / "baseline_data"),
                    prefix_tokens=sequence["prefix_tokens"],target_ids=sequence["target_ids"],input_ids=sequence["input_ids"],total_input_tokens=len(sequence["input_ids"])))
        response=request(origin,"/v1/completions",dict(model=model_id,prompt=sequence["prompt_text"],max_tokens=1,temperature=0.,ignore_eos=False,stream=False))
        atomic_json(directory / "response.json",response)
        result=json.loads((directory / "teacher_forcing.json").read_text())
        if response["usage"]["prompt_tokens"]!=320 or result["scored_tokens"]!=256 or result["target_ids"]!=sequence["target_ids"]:
            raise ValueError("same-input teacher-forcing accounting differs")
        if len(result["layer_metrics"])!=model["layers"]:
            raise ValueError("same-input layer diagnostics incomplete")
        completed.append(str(directory.relative_to(target)))
        atomic_json(progress_path,dict(status="running",completed=completed))
        return result

    with (target / "server.log").open("wb") as log:
        boundaries["startup_begin"]=read_resources(-1)
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        stop=threading.Event()
        monitor=threading.Thread(target=monitor_resources,args=(proc,stop,target / "resources.jsonl",phase))
        monitor.start()
        try:
            wait_ready(origin,proc,str(target / "server.log"),1800)
            model_id=get_json(origin+"/v1/models")["data"][0]["id"]
            atomic_json(target / "startup_stats.json",get_json(origin+"/v1/stats"))
            pilots=json.loads((study / "protocol/pilots.json").read_text())
            phase["name"]="warmup"
            atomic_json(control,dict(armed=True,mask=None,directory=str(target / "warmup"),audit_routes=False))
            atomic_json(target / "warmup_response.json",request(origin,"/v1/chat/completions",body(model_id,pilots[0],rendered,64)))
            warm_sequence={**rendered["sequences"][0],"sequence_id":"startup_teacher_000"}
            teacher(warm_sequence,"baseline",None)
            teacher(warm_sequence,"all100_control",str(all_mask))
            for pool in POOLS:
                teacher(warm_sequence,pool+"75",masks[pool]["path"])
            atomic_json(target / "startup_diagnostic_warmup.json",dict(completed=completed.copy(),
                        interpretation="unmeasured JIT and diagnostic-buffer warmup; excluded from accepted pilot/quality counts"))
            completed.clear()
            atomic_json(control,{"armed":False})
            phase["name"]="settling"
            boundaries["inference_begin"]=wait_for_quiet(proc,policy)
            phase["name"]="inference"
            if stage=="pilot":
                identities=[]
                for row in pilots:
                    baseline, baseline_dir=generate(row,"baseline",None,True)
                    control_response, control_dir=generate(row,"all100_control",str(all_mask),True)
                    if baseline["choices"]!=control_response["choices"] or baseline["usage"]!=control_response["usage"]:
                        raise ValueError("100% actual native mask changes generated output")
                    if json.loads((baseline_dir / "sampled_token_ids.json").read_text())!=json.loads((control_dir / "sampled_token_ids.json").read_text()):
                        raise ValueError("100% actual native mask changes sampled IDs")
                    generate(row,"mixed75",masks["mixed"]["path"],True)
                    generate(row,"same_workload75",masks[row["workload"]]["path"],True)
                    identities.append(dict(request_id=row["request_id"],status="identical_sampled_ids_choices_usage100_and_valid_native75_routes"))
                    atomic_json(target / "identity_controls.json",identities)
                    print(f"actual M75 pilot {key}/{row['request_id']} passed",flush=True)
                for sequence in rendered["sequences"][:2]:
                    baseline=teacher(sequence,"baseline",None)
                    control_result=teacher(sequence,"all100_control",str(all_mask))
                    if abs(baseline["mean_nll"]-control_result["mean_nll"])>1e-5 or control_result["mean_kl"]>1e-7:
                        raise ValueError("100% same-input logit/NLL identity control failed")
                    teacher(sequence,"mixed75",masks["mixed"]["path"])
            else:
                qualification=json.loads((study / "qualification" / (key+".json")).read_text())
                if qualification["source_hashes"]!=hashes() or qualification["status"]!="qualified":
                    raise ValueError("actual masking implementation is not qualified")
                rows=list(json_lines(study / "quality_corpus/tasks.jsonl"))
                for local_index,index in enumerate(spec["task_ordinals"]):
                    row=rows[index]
                    if row["request_id"] != spec["task_ids"][local_index]:
                        raise ValueError("quality task partition changed")
                    treatments=[("baseline",None),("mixed75",masks["mixed"]["path"]),("same_workload75",masks[row["workload"]]["path"])]
                    if index%2: treatments.reverse()
                    for arm,mask in treatments: generate(row,arm,mask)
                    print(f"actual quality {key}/shard{shard_index:02d}: {local_index+1}/{len(spec['task_ids'])} tasks",flush=True)
                for local_index,index in enumerate(spec["sequence_ordinals"]):
                    sequence=rendered["sequences"][index]
                    if sequence["sequence_id"] != spec["sequence_ids"][local_index]:
                        raise ValueError("quality sequence partition changed")
                    baseline=teacher(sequence,"baseline",None)
                    control_result=teacher(sequence,"all100_control",str(all_mask))
                    if abs(baseline["mean_nll"]-control_result["mean_nll"])>1e-5 or control_result["mean_kl"]>1e-7:
                        raise ValueError("100% same-input numerical control changed during quality study")
                    arms=[(pool+"75",masks[pool]["path"]) for pool in POOLS]
                    if index%2: arms.reverse()
                    for arm,mask in arms: teacher(sequence,arm,mask)
                    print(f"actual NLL {key}/shard{shard_index:02d}: {local_index+1}/{len(spec['sequence_ids'])} sequences",flush=True)
            boundaries["inference_end"]=read_resources(proc.pid)
            atomic_json(target / "resource_boundaries.json",boundaries)
        except BaseException as exc:
            atomic_json(target / "failure.json",dict(error=repr(exc),completed=completed,phase=phase["name"]))
            raise
        finally:
            phase["name"]="shutdown"
            stop_server(proc)
            stop.set()
            monitor.join()
    summary=summarize_resources(list(json_lines(target / "resources.jsonl")),boundaries,policy)
    expected_captures=66 if stage=="pilot" else spec["captures"]
    if len(completed)!=expected_captures:
        raise ValueError("complete actual batch has an unexpected capture count")
    atomic_json(target / "resource_summary.json",summary)
    atomic_json(target / "completion.json",dict(status="complete" if summary["status"]=="passed" else "rejected_resources",
                resource_status=summary["status"],captures=len(completed),stage=stage,source_hashes=hashes(),
                shard_index=shard_index,task_ids=spec["task_ids"] if spec else [],sequence_ids=spec["sequence_ids"] if spec else []))
    if summary["status"]!="passed":
        raise ValueError("actual inference resource qualification failed: "+"; ".join(summary["reasons"]))
    if stage=="pilot":
        atomic_json(study / "qualification" / (key+".json"),dict(status="qualified",source_hashes=hashes(),batch_id=batch_id,
                    identity_pairs=15,actual75_route_controls=30,teacher_forcing_sequences=2,resource_summary=summary))
    print(f"actual inference {key}/{batch_id} complete: {len(completed)} captures",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--parent",required=True)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--model",choices=MODEL_ORDER)
    parser.add_argument("--stage",choices=("pilot","quality"))
    parser.add_argument("--batch-id")
    parser.add_argument("--shard-index",type=int)
    args=parser.parse_args()
    study,parent=Path(args.study).resolve(),Path(args.parent).resolve()
    if args.prepare_only:
        prepare(study,parent)
    elif args.model and args.stage and args.batch_id:
        run_batch(study,parent,args.model,args.stage,args.batch_id,args.shard_index)
    else:
        parser.error("model, stage and batch ID are required")


if __name__=="__main__":
    main()
