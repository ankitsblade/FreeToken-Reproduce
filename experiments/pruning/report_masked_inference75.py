"""Score paired actual M75 outputs and report the frozen synthetic quality screen."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from experiments.pruning.collect_moe_routes import json_lines
from experiments.pruning.moe_pruning import ROOT, atomic_json, sha256
from experiments.pruning.pruning_analysis import write_csv
from experiments.pruning.pruning_quality_score import score_structured
from experiments.pruning.run_masked_inference75 import accepted_quality_batches, assert_frozen


def cluster_interval(values, groups, transform=lambda x:x):
    unique=sorted(set(groups))
    totals=np.array([sum(value for value,group in zip(values,groups) if group==key) for key in unique])
    counts=np.array([groups.count(key) for key in unique])
    rng=np.random.default_rng(20261006)
    selected=rng.integers(0,len(unique),size=(10000,len(unique)))
    samples=transform(totals[selected].sum(axis=1)/counts[selected].sum(axis=1))
    return np.quantile(samples,[.025,.975]).tolist()


def score(task,response):
    text=response["choices"][0]["message"].get("content") or ""
    if task["workload"]!="code":
        return score_structured(task,text)
    payload=json.dumps(dict(text=text,reference=task["reference"]))
    try:
        proc=subprocess.run(["uv","run","--no-sync","--project",str(ROOT),"python","-m","experiments.pruning.pruning_quality_score","--code-worker"],
                            input=payload,text=True,capture_output=True,timeout=8,cwd=ROOT)
        if proc.returncode:
            return dict(score=0.,status="restricted_worker_failure",error=proc.stderr[-1000:],unsafe_choice=False)
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return dict(score=0.,status="restricted_worker_timeout",unsafe_choice=False)


def mean_defined(values):
    defined=[value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def run(args):
    study=Path(args.study).resolve()
    assert_frozen(study)
    keys=(args.model,) if args.model else ("gemma_nvfp4","gpt_oss_mxfp4","qwen_nvfp4")
    output=study / "analysis"
    output.mkdir(exist_ok=True)
    tasks=list(json_lines(study / "quality_corpus/tasks.jsonl"))
    manifest=json.loads((study / "quality_corpus/manifest.json").read_text())
    for name,checksum in manifest["hashes"].items():
        if sha256(study / "quality_corpus" / name)!=checksum:
            raise ValueError("frozen quality corpus changed")
    sequences=list(json_lines(study / "quality_corpus/sequences.jsonl"))
    all_decisions={}
    for key in keys:
        batches,task_batches,sequence_batches=accepted_quality_batches(study,key)
        rows=[]
        for task in tasks:
            batch=task_batches[task["request_id"]]
            for arm in ("baseline","mixed75","same_workload75"):
                directory=batch / task["request_id"] / arm
                response=json.loads((directory / "response.json").read_text())
                score_path=directory / "score.json"
                if score_path.exists():
                    scored=json.loads(score_path.read_text())
                    if scored["scorer_sha256"]!=sha256(ROOT / "experiments/pruning/pruning_quality_score.py"):
                        raise ValueError("cached score uses a different scorer")
                else:
                    scored=score(task,response)
                    scored.update(scorer_sha256=sha256(ROOT / "experiments/pruning/pruning_quality_score.py"),response_sha256=sha256(directory / "response.json"))
                    atomic_json(score_path,scored)
                if scored["response_sha256"]!=sha256(directory / "response.json"):
                    raise ValueError("generated response changed after scoring")
                rows.append(dict(model=key,request_id=task["request_id"],group_id=task["group_id"],workload=task["workload"],
                                 arm=arm,score=scored["score"],status=scored["status"],unsafe_choice=scored["unsafe_choice"],
                                 prompt_tokens=response["usage"]["prompt_tokens"],generated_tokens=response["usage"]["completion_tokens"],
                                 finish_reason=response["choices"][0]["finish_reason"],final_content_present=bool(response["choices"][0]["message"].get("content")),
                                 response_sha256=scored["response_sha256"]))
        write_csv(output / (key+"_task_scores.csv"),rows)
        summaries=[]
        for workload in ("math","code","incident"):
            original={row["request_id"]:row for row in rows if row["workload"]==workload and row["arm"]=="baseline"}
            for arm in ("mixed75","same_workload75"):
                masked=[row for row in rows if row["workload"]==workload and row["arm"]==arm]
                if len(original)!=100 or len(masked)!=100:
                    raise ValueError("actual task scores are incomplete")
                losses=[original[row["request_id"]]["score"]-row["score"] for row in masked]
                groups=[row["group_id"] for row in masked]
                ci=cluster_interval(losses,groups)
                status="within_synthetic_margin" if ci[1]<=.02 else ("quality_drop" if ci[0]>.02 else "inconclusive_quality")
                summaries.append(dict(model=key,workload=workload,arm=arm,tasks=100,independent_template_clusters=len(set(groups)),
                                      baseline_score=float(np.mean([row["score"] for row in original.values()])),
                                      masked_score=float(np.mean([row["score"] for row in masked])),paired_loss=float(np.mean(losses)),
                                      loss_ci_low=ci[0],loss_ci_high=ci[1],margin=.02,status=status,
                                      baseline_unsafe_choices=sum(row["unsafe_choice"] for row in original.values()),
                                      masked_unsafe_choices=sum(row["unsafe_choice"] for row in masked),
                                      baseline_truncated=sum(row["finish_reason"]=="length" for row in original.values()),
                                      masked_truncated=sum(row["finish_reason"]=="length" for row in masked),
                                      baseline_fully_correct=sum(abs(row["score"]-1.)<1e-12 for row in original.values()),
                                      fully_correct_lost=sum(abs(original[row["request_id"]]["score"]-1.)<1e-12 and row["score"]<1.-1e-12 for row in masked),
                                      fully_correct_gained=sum(original[row["request_id"]]["score"]<1.-1e-12 and abs(row["score"]-1.)<1e-12 for row in masked),
                                      both_zero_score=sum(original[row["request_id"]]["score"]==0. and row["score"]==0. for row in masked)))
        write_csv(output / (key+"_task_summary.csv"),summaries)
        nll_rows=[]
        for sequence in sequences:
            batch=sequence_batches[sequence["sequence_id"]]
            reference=json.loads((batch / sequence["sequence_id"] / "baseline/teacher_forcing.json").read_text())
            for arm in ("all100_control","mixed75","math75","code75","incident75"):
                result=json.loads((batch / sequence["sequence_id"] / arm / "teacher_forcing.json").read_text())
                if result["scored_tokens"]!=256 or reference["target_ids"]!=result["target_ids"]:
                    raise ValueError("NLL pairing uses different or incomplete target IDs")
                delta=result["mean_nll"]-reference["mean_nll"]
                nll_rows.append(dict(model=key,sequence_id=sequence["sequence_id"],group_id=sequence["group_id"],arm=arm,scored_tokens=256,
                                     baseline_nll=reference["mean_nll"],masked_nll=result["mean_nll"],delta_nll=delta,
                                     relative_perplexity_increase=math.expm1(delta),mean_kl=result["mean_kl"],top1_agreement=result["top1_agreement"],
                                     top10_overlap=result["top10_overlap"],mean_layer_relative_l2=mean_defined([r["relative_l2"] for r in result["layer_metrics"].values()]),
                                     mean_layer_cosine=mean_defined([r["cosine"] for r in result["layer_metrics"].values()]),
                                     undefined_zero_norm_layers=sum(r["relative_l2"] is None or r["cosine"] is None for r in result["layer_metrics"].values())))
        write_csv(output / (key+"_nll_sequences.csv"),nll_rows)
        nll_summaries=[]
        for arm in ("all100_control","mixed75","math75","code75","incident75"):
            selected=[row for row in nll_rows if row["arm"]==arm]
            if len(selected)!=100:
                raise ValueError("NLL requires all 100 sequences")
            ci=cluster_interval([r["delta_nll"] for r in selected],[r["group_id"] for r in selected],np.expm1)
            baseline=float(np.mean([r["baseline_nll"] for r in selected]))
            masked=float(np.mean([r["masked_nll"] for r in selected]))
            status="within_synthetic_margin" if ci[1]<=.02 else ("quality_drop" if ci[0]>.02 else "inconclusive_quality")
            nll_summaries.append(dict(model=key,arm=arm,sequences=100,scored_tokens=25600,baseline_mean_nll=baseline,masked_mean_nll=masked,
                                     baseline_perplexity=math.exp(baseline),masked_perplexity=math.exp(masked),
                                     relative_perplexity_increase=math.expm1(masked-baseline),increase_ci_low=ci[0],increase_ci_high=ci[1],
                                     mean_kl=float(np.mean([r["mean_kl"] for r in selected])),top1_agreement=float(np.mean([r["top1_agreement"] for r in selected])),
                                     top10_overlap=float(np.mean([r["top10_overlap"] for r in selected])),status=status))
        write_csv(output / (key+"_nll_summary.csv"),nll_summaries)
        decisions={}
        for pool in ("mixed","math","code","incident"):
            scores=[r for r in summaries if r["arm"]==("mixed75" if pool=="mixed" else "same_workload75") and (pool=="mixed" or r["workload"]==pool)]
            nll=next(r for r in nll_summaries if r["arm"]==pool+"75")
            statuses=[r["status"] for r in scores]+[nll["status"]]
            unsafe_increase=any(r["masked_unsafe_choices"]>r["baseline_unsafe_choices"] for r in scores)
            qualified=all(status=="within_synthetic_margin" for status in statuses) and not unsafe_increase
            decisions[pool]=dict(status="synthetic_quality_screen_passed" if qualified else ("quality_drop" if "quality_drop" in statuses else "inconclusive_quality"),
                                 eligible_for_matched_serving_diagnostic=qualified,task_workloads=[r["workload"] for r in scores],
                                 unsafe_choice_count_increased=unsafe_increase,
                                 nll_scope="all 100 synthetic reference sequences, including transfer outside the mask's named workload",
                                 general_quality_preservation="not established by this synthetic corpus")
        atomic_json(output / (key+"_decisions.json"),decisions)
        all_decisions[key]=decisions
        print(f"scored actual M75 {key}: {json.dumps(decisions)}",flush=True)
    atomic_json(output / "current_completion.json",dict(status="scored_requested_models",models=list(keys),decisions=all_decisions,
                source_sha256=sha256(Path(__file__)),uncertainty="paired cluster bootstrap with 10,000 replicates, seed 20261006",
                collection="complete union of frozen resource-qualified fresh-process shards; no rejected outputs joined",
                scope="synthetic quality only; original failed activation gates remain unchanged; no serving speedup from collection timings"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--model",choices=("gemma_nvfp4","gpt_oss_mxfp4","qwen_nvfp4"))
    run(parser.parse_args())
