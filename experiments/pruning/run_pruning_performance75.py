"""Fresh-process untraced serving diagnostics for masks passing the synthetic quality screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from experiments.paths import runtime_pythonpath

import numpy as np

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import ROOT, atomic_json, sha256
from experiments.pruning.pruning_analysis import write_csv
from experiments.pruning.pruning_resources import monitor_resources, read_resources, summarize_resources, wait_for_quiet
from experiments.pruning.run_masked_inference75 import assert_frozen, uv

sys.path.insert(0,str(ROOT / "experiments/baselines"))
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready


def stream(origin,model_id,row,kwargs,max_tokens=256):
    body=dict(model=model_id,messages=[{"role":"user","content":row["prompt"]}],max_tokens=max_tokens,
              temperature=0.,top_p=1.,top_k=-1,ignore_eos=False,chat_template_kwargs=kwargs,stream=True,
              stream_options={"include_usage":True})
    req=urllib.request.Request(origin+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    start=time.monotonic()
    first_delta=first_visible=last_delta=None
    text,reasoning,events=[],[],[]
    usage=finish=None
    with urllib.request.urlopen(req,timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:"): continue
            data=line[5:].strip()
            if data==b"[DONE]": break
            event=json.loads(data)
            now=time.monotonic()
            events.append(dict(elapsed_seconds=now-start,data=event))
            if event.get("usage"): usage=event["usage"]
            for choice in event.get("choices",[]):
                delta=choice.get("delta",{})
                content=delta.get("content") or ""
                thought=delta.get("reasoning_content") or delta.get("reasoning") or ""
                if content or thought:
                    first_delta=now if first_delta is None else first_delta
                    last_delta=now
                if content:
                    first_visible=now if first_visible is None else first_visible
                    text.append(content)
                if thought: reasoning.append(thought)
                if choice.get("finish_reason"): finish=choice["finish_reason"]
    elapsed=time.monotonic()-start
    if not usage or not finish or first_delta is None:
        raise ValueError("serving stream lacks final usage, finish reason, or output delta")
    result=dict(request_id=row["request_id"],workload=row["workload"],usage=usage,finish_reason=finish,
                first_raw_api_delta_seconds=first_delta-start,first_visible_text_seconds=first_visible-start if first_visible else None,
                first_raw_model_token_seconds=None,end_to_end_seconds=elapsed,
                end_to_end_tokens_per_second=usage["completion_tokens"]/elapsed,
                api_decode_proxy_tokens_per_second=(usage["completion_tokens"]-1)/(last_delta-first_delta) if last_delta>first_delta else None,
                content="".join(text),reasoning_content="".join(reasoning),events=events)
    result["output_sha256"]=hashlib.sha256(json.dumps([result["content"],result["reasoning_content"]],ensure_ascii=False).encode()).hexdigest()
    return result


def monitor_gpu(stop,target,phase):
    with target.open("x") as stream:
        while not stop.is_set():
            try:
                value=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
                values=[int(line.strip())*(1<<20) for line in value.splitlines() if line.strip()]
                if len(values)!=1: raise ValueError("serving diagnostic expects this single-GPU device")
                row=dict(monotonic_seconds=time.monotonic(),phase=phase["name"],device_vram_bytes=values[0])
            except Exception as exc:
                row=dict(monotonic_seconds=time.monotonic(),phase=phase["name"],error=repr(exc))
            stream.write(json.dumps(row)+"\n"); stream.flush()
            stop.wait(1)


def batch(study,key,pool,rep,rows,kwargs,output):
    assert_frozen(study)
    frozen=json.loads((output / (key+"_protocol.json")).read_text())
    if any(sha256(ROOT / name)!=expected for name,expected in frozen["source_hashes"].items()):
        raise ValueError("untraced serving implementation changed after freeze")
    model=json.loads((study / "protocol/models.json").read_text())[key]
    policy=json.loads((study / "protocol" / (key+"_resources.json")).read_text())
    masks=json.loads((study / "protocol/masks.json").read_text())[key]
    target=output / key / f"rep{rep}_{pool}"
    target.mkdir(parents=True,exist_ok=False)
    mask=None if pool=="baseline" else masks[pool]["path"]
    env={k:v for k,v in os.environ.items() if not k.startswith(("FREETOKEN_MOE_","FREETOKEN_HYBRID_","FREETOKEN_PRUNING_"))}
    cuda=subprocess.check_output(uv("-c",'import sysconfig;print(sysconfig.get_paths()["purelib"]+"/nvidia/cu13")'),text=True).strip()
    env.update(CUDA_HOME=cuda,CC="/usr/bin/gcc",PATH=cuda+"/bin:"+env["PATH"],
               LD_LIBRARY_PATH=cuda+"/lib:"+env.get("LD_LIBRARY_PATH",""),PYTHONPATH=runtime_pythonpath(),
               FREETOKEN_BENCHBW_PATH=str(ROOT / "results/baseline/calibration_repeat5/profile-run-04.json"),FREETOKEN_DISABLE_OVERLAP_SCHEDULING="1")
    if mask: env["FREETOKEN_PRUNING_MASK"]=mask
    env.update(policy["allocator_environment"])
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
        raise ValueError("serving diagnostics require an idle GPU")
    port=free_port(); origin=f"http://127.0.0.1:{port}"
    cmd=uv("-m","experiments.pruning.performance_pruning_probe","serve","--model",str(ROOT / model["path"]),"--host","127.0.0.1","--port",str(port),
           "--moe-strategy","hybrid","--max-running-requests","1","--max-seq-len-override","1280" if key=="gemma_nvfp4" else "2048",
           "--max-prefill-length","1024","--memory-ratio","0.8" if key=="gemma_nvfp4" else "0.9","--cuda-graph-max-bs","0",
           "--moe-hybrid-max-fetch","-1","--moe-cache-size",str(model["cache_slots"]),"--cache-type","naive","--expert-load","serial")
    if key!="gpt_oss_mxfp4": cmd.append("--text-model-only")
    atomic_json(target / "configuration.json",dict(command=cmd,model=key,pool=pool,rep=rep,mask=mask,
                mask_sha256=sha256(Path(mask)) if mask else None,untraced=True,graphs=False,expert_cache="cold per request",policy=policy))
    phase={"name":"startup"}; boundaries={}; measurements=[]
    with (target / "server.log").open("wb") as log:
        boundaries["startup_begin"]=read_resources(-1)
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        stop=threading.Event()
        monitor=threading.Thread(target=monitor_resources,args=(proc,stop,target / "resources.jsonl",phase))
        gpu_monitor=threading.Thread(target=monitor_gpu,args=(stop,target / "gpu_memory.jsonl",phase))
        monitor.start()
        gpu_monitor.start()
        try:
            wait_ready(origin,proc,str(target / "server.log"),1800)
            model_id=get_json(origin+"/v1/models")["data"][0]["id"]
            atomic_json(target / "startup_stats.json",get_json(origin+"/v1/stats"))
            phase["name"]="warmup"
            for i,row in enumerate(rows): atomic_json(target / f"warmup_{i}.json",stream(origin,model_id,row,kwargs,64))
            phase["name"]="settling"
            boundaries["inference_begin"]=wait_for_quiet(proc,policy)
            phase["name"]="inference"
            for round_index in range(2):
                for row in rows:
                    result=stream(origin,model_id,row,kwargs)
                    atomic_json(target / f"round{round_index}_{row['request_id']}.json",result)
                    measurements.append(dict(model=key,pool=pool,rep=rep,round=round_index,request_id=row["request_id"],workload=row["workload"],
                                             first_raw_api_delta_seconds=result["first_raw_api_delta_seconds"],
                                             first_visible_text_seconds=result["first_visible_text_seconds"],end_to_end_seconds=result["end_to_end_seconds"],
                                             end_to_end_tokens_per_second=result["end_to_end_tokens_per_second"],api_decode_proxy_tokens_per_second=result["api_decode_proxy_tokens_per_second"],
                                             prompt_tokens=result["usage"]["prompt_tokens"],completion_tokens=result["usage"]["completion_tokens"],
                                             finish_reason=result["finish_reason"],output_sha256=result["output_sha256"]))
            boundaries["inference_end"]=read_resources(proc.pid)
            atomic_json(target / "resource_boundaries.json",boundaries)
            atomic_json(target / "final_stats.json",get_json(origin+"/v1/stats"))
        except BaseException as exc:
            atomic_json(target / "failure.json",dict(error=repr(exc),phase=phase["name"]))
            raise
        finally:
            phase["name"]="shutdown"; stop_server(proc); stop.set(); monitor.join(); gpu_monitor.join()
    summary=summarize_resources(list(json_lines(target / "resources.jsonl")),boundaries,policy)
    atomic_json(target / "resource_summary.json",summary)
    gpu_rows=list(json_lines(target / "gpu_memory.jsonl"))
    memory_rows=list(json_lines(target / "resources.jsonl"))
    if any("error" in row for row in gpu_rows): raise ValueError("device VRAM observations incomplete")
    atomic_json(target / "sampled_memory_peaks.json",dict(
        device_vram_peak_inference_bytes=max(row["device_vram_bytes"] for row in gpu_rows if row["phase"]=="inference"),
        server_subtree_rss_peak_inference_bytes=max(sum(row["rss_by_pid"].values()) for row in memory_rows if row.get("phase")=="inference"),
        sampling_interval_seconds=1,scope="sampled peaks; device VRAM includes display and allocator reservation; no subsecond peak guarantee"))
    atomic_json(target / "completion.json",dict(status="complete" if summary["status"]=="passed" else "rejected_resources",resource_status=summary["status"],measurements=measurements))
    if summary["status"]!="passed": raise ValueError("untraced serving resource gate failed")
    return measurements


def paired_summary(measurements,pools):
    summaries=[]
    metrics=("first_raw_api_delta_seconds","first_visible_text_seconds","end_to_end_seconds","end_to_end_tokens_per_second","api_decode_proxy_tokens_per_second")
    baseline={(rep,rid):[row for row in measurements if row["pool"]=="baseline" and row["rep"]==rep and row["request_id"]==rid]
              for rep in range(5) for rid in {row["request_id"] for row in measurements}}
    for pool in pools:
        for metric in metrics:
            ratios=[]; pairs=0
            for rep in range(5):
                original,masked=[],[]
                for rid in {row["request_id"] for row in measurements}:
                    rows=[row for row in measurements if row["pool"]==pool and row["rep"]==rep and row["request_id"]==rid
                          and (pool=="mixed" or row["workload"]==pool)]
                    if not rows: continue
                    base=baseline[rep,rid]
                    current=[r[metric] for r in rows if r[metric] is not None]
                    reference=[r[metric] for r in base if r[metric] is not None]
                    if len(current)!=2 or len(reference)!=2: continue
                    original.append(float(np.mean(reference))); masked.append(float(np.mean(current))); pairs+=1
                if original and all(value>0 for value in [*original,*masked]):
                    ratios.append(float(np.mean(masked)/np.mean(original)))
            if len(ratios)==5:
                logs=np.log(ratios)
                selected=np.random.default_rng(20261006).integers(0,5,size=(10000,5))
                lo,hi=np.quantile(np.exp(logs[selected].mean(axis=1)),[.025,.975])
                summaries.append(dict(pool=pool,metric=metric,fresh_process_pairs=5,paired_request_cells=pairs,
                                      geometric_masked_over_baseline=float(np.exp(logs.mean())),ratio_ci_low=float(lo),ratio_ci_high=float(hi),
                                      interpretation="lower is faster for latency; higher is faster for throughput; generated token counts can differ"))
            else:
                summaries.append(dict(pool=pool,metric=metric,fresh_process_pairs=len(ratios),paired_request_cells=pairs,
                                      geometric_masked_over_baseline=None,ratio_ci_low=None,ratio_ci_high=None,
                                      interpretation="unavailable: incomplete paired API endpoint, including single-delta continuations"))
    return summaries


def run(args):
    study=Path(args.study).resolve(); key=args.model
    decisions=json.loads((study / "analysis" / (key+"_decisions.json")).read_text())
    pools=[pool for pool,decision in decisions.items() if decision["eligible_for_matched_serving_diagnostic"]]
    output=study / "performance75"; output.mkdir(exist_ok=True)
    if not pools:
        atomic_json(output / (key+"_decision.json"),dict(status="skipped_by_quality_screen",decisions=decisions))
        return
    tasks=list(json_lines(study / "quality_corpus/tasks.jsonl"))
    ids=[f"quality_{workload}_{family:02d}_00" for workload in ("math","code","incident") for family in (0,4,8)]
    lookup={row["request_id"]:row for row in tasks}; rows=[lookup[rid] for rid in ids]
    kwargs=json.loads((study / "protocol/rendered.json").read_text())[key]["chat_template_kwargs"]
    atomic_json(output / (key+"_protocol.json"),dict(status="frozen_before_serving_results",pools=pools,repetitions=5,request_ids=ids,
                rounds_per_process=2,bootstrap_unit="fresh process repetition, paired by repetition/request",
                baseline_order="first on even repetition, last on odd repetition; rotate masked order",
                endpoint_scope="raw API delta and visible text; first raw model-token latency unavailable without instrumentation",
                throughput_scope="end-to-end throughput; API decode proxy includes un-emitted special tokens in usage",
                workload_scope="mixed all three; task-specific timing claims restricted to its named workload; other task timings are transfer diagnostics",
                source_hashes={name:sha256(ROOT / name) for name in ("experiments/pruning/run_pruning_performance75.py","experiments/pruning/performance_pruning_probe.py","experiments/pruning/native_pruning_probe.py")}))
    measurements=[]
    for rep in range(5):
        offset=rep%len(pools); order=pools[offset:]+pools[:offset]
        order=(["baseline"]+order) if rep%2==0 else (order+["baseline"])
        for pool in order:
            measurements.extend(batch(study,key,pool,rep,rows,kwargs,output))
            write_csv(output / (key+"_requests.csv"),measurements)
            print(f"untraced serving {key}/rep{rep}/{pool} complete",flush=True)
    write_csv(output / (key+"_paired_summary.csv"),paired_summary(measurements,pools))
    atomic_json(output / (key+"_decision.json"),dict(status="untraced_serving_diagnostic_complete",fresh_repetitions=5,pools=pools,
                requests_per_process=18,source_sha256=sha256(Path(__file__)),interpretation="synthetic quality-qualified serving diagnostic; no general quality or engine-kernel speedup claim"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--model",required=True,choices=("gemma_nvfp4","gpt_oss_mxfp4","qwen_nvfp4"))
    run(parser.parse_args())
