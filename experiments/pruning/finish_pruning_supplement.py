"""Serial execution of the Gemma R75 repair and qualified actual M75 supplement."""
from __future__ import annotations

import argparse
import json
import subprocess
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pruning.moe_pruning import ROOT, atomic_json


def run(args):
    parent = Path(args.parent).resolve()
    rerouting = parent / "additional/gemma_rerouting_v2"
    actual = Path(args.actual_study).resolve() if args.actual_study else parent / "additional/actual_masking75_v1"
    log_dir = parent / "orchestration" / args.run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    progress = dict(status="running", jobs=[], actual_quality_complete=False, deferred60_50="not_started")

    def save():
        progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(log_dir / "progress.json", progress)

    def launch(module, name, arguments, expected):
        if expected.exists():
            completion=json.loads(expected.read_text())
            if completion.get("status") in ("complete","qualified","captures_complete") and completion.get("resource_status","passed")=="passed":
                progress["jobs"].append(dict(name=name,status="accepted_existing",completion=str(expected)))
                save()
                return
            raise ValueError(f"existing output cannot be accepted: {expected}")
        command=["uv","run","--no-sync","--project",str(ROOT),"python","-m",module,*map(str,arguments)]
        with (log_dir / (name+".log")).open("w") as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            job=dict(name=name,status="running",pid=proc.pid,command=command)
            progress["jobs"].append(job)
            save()
            while proc.poll() is None:
                save()
                time.sleep(10)
            job["returncode"]=proc.returncode
            if proc.returncode:
                job["status"]="failed"
                save()
                raise RuntimeError(f"{name} failed; preserve log and failed batch")
        job["status"]="complete"
        if not expected.exists():
            raise ValueError(f"{name} did not write its expected completion artifact")
        save()

    save()
    try:
        qualification=rerouting / "pilots/gemma_nvfp4/qualification.json"
        pilot=rerouting / "pilots/gemma_nvfp4" / args.gemma_pilot_batch
        progress["jobs"].append(dict(name=args.gemma_pilot_batch,status="waiting_for_running_pilot"))
        for _ in range(120):
            if qualification.exists() and json.loads(qualification.read_text()).get("batch_id")==args.gemma_pilot_batch:
                break
            for name in ("batch_failure.json","qualification_failure.json"):
                if (pilot / name).exists():
                    raise RuntimeError((pilot / name).read_text())
            save()
            time.sleep(10)
        else:
            raise TimeoutError("fresh Gemma full-score pilot did not qualify within 20 minutes")
        progress["jobs"][-1]["status"]="qualified"
        save()
        repeat="gemma_full_scores_repeat_v2"
        launch("experiments.pruning.collect_gemma_rerouting",repeat,
               ["--study",rerouting,"--model","gemma_nvfp4","--stage","repeat","--batch-id",repeat],
               rerouting / "routes/gemma_nvfp4" / repeat / "completion.json")
        launch("experiments.pruning.pruning_gemma_rerouting","gemma_native_R75",
               ["--study",rerouting,"--parent",parent / "additional/gemma_v1","--batch-id",repeat],
               rerouting / "native_controls75/completion.json")
        progress["gemma_R75"]="complete"
        save()
        shard_plan=json.loads((actual / "protocol/quality_shards.json").read_text())
        for key in args.models:
            cells=[("pilot",None), *(("quality",i) for i in range(shard_plan["count"]))]
            for stage,shard_index in cells:
                for attempt in range(args.maximum_resource_attempts):
                    name=key+"_actual_"+stage+"_v1"+(f"_shard{shard_index:02d}" if shard_index is not None else "")+(f"_retry{attempt}" if attempt else "")
                    target=actual / "runs" / key / name
                    if stage=="quality":
                        model=json.loads((actual / "protocol/models.json").read_text())[key]
                        config=json.loads((ROOT / model["path"] / "config.json").read_text())
                        config=config.get("text_config",config)
                        sequences=len(shard_plan["shards"][str(shard_index)]["sequence_ids"])+1
                        raw_bytes=sequences*(256*4*(config["vocab_size"]+model["layers"]*config["hidden_size"])+(model["layers"]+1)*4096)
                        if shutil.disk_usage(actual).free < raw_bytes+(20<<30):
                            raise RuntimeError("insufficient disk headroom for lossless actual quality artifacts; preserve existing data")
                    try:
                        resource_path=target / "resource_summary.json"
                        if target.exists() and resource_path.exists() and json.loads(resource_path.read_text())["status"]=="failed":
                            raise RuntimeError("preserved rejected resource attempt")
                        arguments=["--study",actual,"--parent",parent,"--model",key,"--stage",stage,"--batch-id",name]
                        if shard_index is not None: arguments.extend(["--shard-index",shard_index])
                        launch("experiments.pruning.run_masked_inference75",name,
                               arguments,
                               target / "completion.json")
                        break
                    except RuntimeError:
                        resource_path=target / "resource_summary.json"
                        failure_path=target / "failure.json"
                        if not resource_path.exists() and failure_path.exists() and not (target / "completion.json").exists():
                            failure=json.loads(failure_path.read_text())
                            if (failure.get("phase")=="settling" and failure.get("completed")==[]
                                    and failure.get("error")=="RuntimeError('host memory did not settle within the bounded startup wait')"):
                                atomic_json(resource_path,dict(status="failed",
                                    reasons=["host memory did not settle within the bounded startup wait"],
                                    observation="startup resource rejection before measured inference",
                                    phase="settling",completed_captures=0,numerical_data_accepted=False))
                        if not resource_path.exists() or json.loads(resource_path.read_text())["status"]!="failed" or attempt==args.maximum_resource_attempts-1:
                            raise
                        progress.setdefault("resource_retries",[]).append(dict(model=key,stage=stage,
                            rejected_batch=name,reasons=json.loads(resource_path.read_text())["reasons"],
                            action="fresh identical process and full batch; no rejected outputs accepted"))
                        save()
        progress.update(status="actual_quality_collection_complete",actual_quality_complete=True,
                        next_action="score paired actual outputs, decide quality margins and qualified performance, export supplemental reports")
        save()
    except BaseException as exc:
        progress.update(status="stopped_on_failure",error=repr(exc))
        save()
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",required=True)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--gemma-pilot-batch",required=True)
    parser.add_argument("--actual-study")
    parser.add_argument("--models",nargs="+",choices=("gemma_nvfp4","gpt_oss_mxfp4","qwen_nvfp4"),default=["gemma_nvfp4","gpt_oss_mxfp4","qwen_nvfp4"])
    parser.add_argument("--maximum-resource-attempts",type=int,choices=range(1,9),default=3)
    run(parser.parse_args())
