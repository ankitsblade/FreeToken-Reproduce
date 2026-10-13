"""Export a checked actual M75 closeout without modifying the original offline closeout."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from experiments.pruning.moe_pruning import ROOT, atomic_json, sha256, snapshot
from experiments.pruning.run_masked_inference75 import assert_frozen, accepted_quality_batches
from experiments.pruning.finish_actual_inference75 import validate_reference_layout


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def figures(output,summaries,nll):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target=output / "figures"; target.mkdir()
    plt.rcParams.update({"font.size":9,"svg.hashsalt":"actual-pruning75-20261006","axes.spines.top":False,"axes.spines.right":False})
    colors={"mixed75":"#3366aa","same_workload75":"#bb6b22"}
    models=("gemma_nvfp4","gpt_oss_mxfp4","qwen_nvfp4")
    names={"gemma_nvfp4":"Gemma","gpt_oss_mxfp4":"GPT-OSS","qwen_nvfp4":"Qwen"}
    fig,axes=plt.subplots(1,3,figsize=(10,3.6),sharey=True)
    for ax,key in zip(axes,models):
        for arm,offset in (("mixed75",-.16),("same_workload75",.16)):
            rows=[next(r for r in summaries if r["model"]==key and r["workload"]==workload and r["arm"]==arm) for workload in ("math","code","incident")]
            center=np.array([float(r["paired_loss"])*100 for r in rows])
            lo=np.array([float(r["loss_ci_low"])*100 for r in rows]); hi=np.array([float(r["loss_ci_high"])*100 for r in rows])
            ax.errorbar(np.arange(3)+offset,center,yerr=np.maximum(0,np.vstack((center-lo,hi-center))),fmt="o",capsize=3,color=colors[arm],label=arm)
        ax.axhline(2,color="#aa3333",linestyle="--",linewidth=1,label="2 pp margin")
        ax.axhline(0,color="#777777",linewidth=.7)
        ax.set_xticks(range(3),["Math","Code","Incident"]); ax.set_title(names[key])
    axes[0].set_ylabel("Baseline - masked score (percentage points)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Actual 75% retention: paired synthetic task losses (95% template-cluster intervals)")
    fig.tight_layout()
    for ext in ("png","pdf","svg"): fig.savefig(target / ("task_losses."+ext),dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.6))
    for ax,key in zip(axes,models):
        rows=[next(r for r in nll if r["model"]==key and r["arm"]==pool+"75") for pool in ("mixed","math","code","incident")]
        center=np.array([float(r["relative_perplexity_increase"])*100 for r in rows])
        lo=np.array([float(r["increase_ci_low"])*100 for r in rows]); hi=np.array([float(r["increase_ci_high"])*100 for r in rows])
        ax.errorbar(range(4),center,yerr=np.maximum(0,np.vstack((center-lo,hi-center))),fmt="o",capsize=3,color="#3366aa")
        ax.axhline(2,color="#aa3333",linestyle="--",linewidth=1)
        ax.axhline(0,color="#777777",linewidth=.7)
        ax.set_xticks(range(4),["Mixed","Math","Code","Incident"],rotation=20); ax.set_title(names[key])
        ax.set_ylabel("Relative perplexity increase (%)")
    fig.suptitle("Fixed-input native prefill NLL: 25,600 continuation tokens per mask/model")
    fig.tight_layout()
    for ext in ("png","pdf","svg"): fig.savefig(target / ("perplexity_increases."+ext),dpi=200)
    plt.close(fig)
    atomic_json(target / "configuration.json",dict(matplotlib=matplotlib.__version__,scope="synthetic corpus; paired cluster bootstrap; no architecture-causal inference"))


def run(args):
    parent,study=Path(args.parent).resolve(),Path(args.study).resolve()
    assert_frozen(study)
    if subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip():
        raise ValueError("closeout requires no active GPU experiment")
    validate_reference_layout(study)
    original=parent / "closeout75/artifact_manifest.json"
    original_hash=sha256(original)
    if original_hash!="967c23a59581d36eedc64cd2f1f7eb82f053e41190dd66d92ad640b82b6a1aba":
        raise ValueError("original offline closeout manifest changed")
    original_records=json.loads(original.read_text())["artifacts"]
    for index,(name,record) in enumerate(original_records.items()):
        path=parent / name
        if path.stat().st_size!=record["bytes"] or sha256(path)!=record["sha256"]:
            raise ValueError(f"original frozen artifact changed: {name}")
        if index%2000==0: print(f"verified original offline artifacts {index}/{len(original_records)}",flush=True)
    rerouting=parent / "additional/gemma_rerouting_v2"
    native=json.loads((rerouting / "native_controls75/completion.json").read_text())
    if native["status"]!="complete" or native["routing_requests"]!=9 or native["cold_cases"]!=12:
        raise ValueError("Gemma native R75 supplement incomplete")
    output=study / "closeout75_actual"; output.mkdir(exist_ok=False)
    summaries,nll,resources,decisions,performance=[],[],[],{},{}
    models=json.loads((study / "protocol/models.json").read_text())
    for key in models:
        qualified=json.loads((study / "qualification" / (key+".json")).read_text())
        if qualified["status"]!="qualified" or qualified["identity_pairs"]!=15:
            raise ValueError("actual pilot gate incomplete")
        pilots=[path.parent for path in (study / "runs" / key).glob("*/completion.json") if
                json.loads(path.read_text()).get("stage")=="pilot" and json.loads(path.read_text()).get("resource_status")=="passed"]
        if len(pilots)!=1 or json.loads((pilots[0]/"completion.json").read_text())["captures"]!=66:
            raise ValueError(f"need one complete accepted pilot: {key}")
        quality_batches,_,_=accepted_quality_batches(study,key)
        if sum(json.loads((batch/"completion.json").read_text())["captures"] for batch in quality_batches)!=1500:
            raise ValueError("aggregate actual quality capture count differs")
        for batch in [*pilots,*quality_batches]:
            completion=json.loads((batch / "completion.json").read_text())
            resource=json.loads((batch / "resource_summary.json").read_text())
            if resource["status"]!="passed": raise ValueError("actual resource gate failed")
            resources.append(dict(model=key,stage=completion["stage"],shard_index=completion["shard_index"],batch_id=batch.name,**resource))
        task_rows=csv_rows(study / "analysis" / (key+"_task_scores.csv"))
        if len(task_rows)!=900 or len({(r["request_id"],r["arm"]) for r in task_rows})!=900:
            raise ValueError("actual task score rows are incomplete/duplicated")
        summaries.extend(csv_rows(study / "analysis" / (key+"_task_summary.csv")))
        nll.extend(csv_rows(study / "analysis" / (key+"_nll_summary.csv")))
        decisions[key]=json.loads((study / "analysis" / (key+"_decisions.json")).read_text())
        performance[key]=json.loads((study / "performance75" / (key+"_decision.json")).read_text())
        if performance[key]["status"] not in ("skipped_by_quality_screen","untraced_serving_diagnostic_complete"):
            raise ValueError("serving decision incomplete")
        if any(d["eligible_for_matched_serving_diagnostic"] for d in decisions[key].values()) and performance[key]["status"]!="untraced_serving_diagnostic_complete":
            raise ValueError("a quality-qualified serving arm still requires a disposition")
    atomic_json(output / "resource_summary.json",resources)
    atomic_json(output / "decisions.json",decisions)
    figures(output,summaries,nll)
    lines=["# Actual 75% expert retention: inference supplement", "",
           "Gemma native rerouting is repaired and validated. Actual 25% expert exclusion was evaluated in FreeToken on all three qualified checkpoints.","",
           "The original offline closeout and its failed activation-screen decisions remain unchanged. This new stage was explicitly authorized by the user.","",
           "## Completed scope","",
           "- 15 actual baseline/all100 identity pairs and 30 native 75% route audits per model.",
           "- 300 fresh synthetic tasks per model under baseline, mixed75, and same-workload75: 2,700 task generations total.",
           "- 100 fixed text sequences per model under baseline, all100, and four 75% masks: 1,800 native prefill likelihood evaluations, 25,600 continuation tokens per mask/model.",
           "- Exact supported raw-text tokenization is checked inside the worker. Shared branches and native k remain active, and original expert IDs are retained.",
           "- Original weight allocations and cache budgets are unchanged. This establishes no checkpoint compaction or RAM saving.","",
           "Quality collection uses twelve frozen serial fresh-process shards per model. Each task and text sequence is assigned once; paired configurations remain together. Each batch passes the unchanged resource gates, including a 128 MiB system swap limit. Total system swap across the full collection may exceed 128 MiB; this is not a whole-model cumulative swap guarantee.","",
           "## Task quality","","Scores are on [0,1]. Loss is baseline minus masked, with paired 95% template-cluster bootstrap intervals.","",
           "| Model | Workload | Mask | Baseline | Masked | Loss, pp [95% interval] | Decision |",
           "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for row in summaries:
        lines.append(f"| {row['model']} | {row['workload']} | {row['arm']} | {float(row['baseline_score']):.3f} | {float(row['masked_score']):.3f} | {100*float(row['paired_loss']):.2f} [{100*float(row['loss_ci_low']):.2f}, {100*float(row['loss_ci_high']):.2f}] | {row['status']} |")
    lines.extend(["", "The paired CSVs also report baseline fully correct tasks, fully correct answers lost/gained, both-zero scores, and truncations. A near-zero paired loss when both configurations fail does not demonstrate useful task ability; interpret the baseline score alongside every margin decision."])
    lines.extend(["","![Paired task losses](figures/task_losses.png)","","## Same-input likelihood","",
                  "Perplexity comparisons are paired within each checkpoint/tokenizer. These raw perplexities cannot rank different model tokenizers.","",
                  "| Model | Mask | Relative perplexity increase, % [95% interval] | KL(base || masked) | Top-1 agreement | Decision |",
                  "| --- | --- | --- | ---: | ---: | --- |"])
    for row in nll:
        lines.append(f"| {row['model']} | {row['arm']} | {100*float(row['relative_perplexity_increase']):.2f} [{100*float(row['increase_ci_low']):.2f}, {100*float(row['increase_ci_high']):.2f}] | {float(row['mean_kl']):.5f} | {float(row['top1_agreement']):.3f} | {row['status']} |")
    lines.extend(["","![Fixed-input likelihood changes](figures/perplexity_increases.png)","","## Interpretation and limitations","",
                  "The predeclared margins are 2 percentage points task-score loss and 2% relative perplexity increase, evaluated using upper 95% paired cluster-bootstrap bounds. Wide intervals are inconclusive. Unsafe-choice increases block serving qualification.","",
                  "This is a synthetic elementary-task and authored-text study with shared templates. Results describe this corpus and do not establish broad real-world quality preservation. Generated final answers, natural EOS/truncation, code failures, incident failures, and all100 numerical controls are retained in the score artifacts.","",
                  "Code scores use frozen finite functional cases in a restricted local worker; passing those cases is not proof of general correctness. Incident evidence scores use a frozen keyword rubric that can miss valid paraphrases. The restricted code worker is not a general security sandbox. Each domain and the reference text have ten template clusters, so the 100 variants do not represent 100 independent problem families.","",
                  "This supplement therefore does not satisfy the original protocol's requirement for 100 independent tasks per workload for a comprehensive quality-preservation claim. Its serving eligibility is an exploratory synthetic-corpus screen only.","",
                  "Likelihood uses native prefill MoE on fixed causal inputs and identical vocabulary-head chunks. Autoregressive task runs propagate changed hidden states and generated histories. Quality collection timings and cache simulations are not serving speedups.","",
                  "## Serving dispositions",""])
    for key,value in performance.items(): lines.append(f"- {key}: `{value['status']}`.")
    lines.extend(["","Untraced serving, where qualified, uses five fresh-process repetitions and nine preselected tasks, two rounds/process. Raw API-delta latency and first visible text latency are distinct; first raw model-token latency is unavailable without instrumentation. Device VRAM and subtree RSS peaks are sampled at one second. Task-specific claims are restricted to their named workload. See `performance75` for raw streams, counts, hashes, resource evidence, and paired process-level comparisons.","",
                  "## Gemma rerouting repair","",
                  "Nine requests have exact native 100% identities and original/fresh-repeat agreement; all 12 native cache controls matched. At 768 slots, R75 calibrated hybrid miss demand falls 0.46%, below the original 5% screen. At 576 slots it increases 9.51%. Observed pre-decode caches are empty, so cold and projected post-prefill conditions coincide.","",
                  "## Provenance","",
                  f"Original offline manifest SHA256: `{original_hash}`. All {len(original_records):,} original artifact hashes were reverified before supplemental closeout.","",
                  "All runtime source/control hashes, checkpoint revisions/metadata, deployment frequency counts/masks, input IDs, corpus references, resource logs, rejected attempts, and native controls are preserved. Kimi is excluded; 60% and 50% are not started.",""])
    (output / "report.md").write_text("\n".join(lines))
    snapshot(output / "source_snapshot")
    plan_name = "docs/experiments/pruning/protocol.md"
    shutil.copy2(ROOT / plan_name, output / "source_snapshot" / plan_name)
    snapshot_manifest = output / "source_snapshot/files.json"
    snapshot_records = json.loads(snapshot_manifest.read_text())
    snapshot_records[plan_name] = sha256(ROOT / plan_name)
    atomic_json(snapshot_manifest, snapshot_records)
    atomic_json(output / "completion.json",dict(status="actual75_supplement_complete",models=list(models),task_generations=2700,
                same_input_evaluations=1800,original_closeout_preserved=True,original_manifest_sha256=original_hash,
                gemma_R75="complete",general_quality_preservation="not established",deferred60_50="not_started"))
    records={}
    prior_arms = sorted(path for path in (parent / "additional").glob("actual_masking75_v*") if path.is_dir() and path != study)
    for root in (study,rerouting,*prior_arms):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path==output / "artifact_manifest.json" or "__pycache__" in path.parts or path.name.endswith("launcher.log"):
                continue
            records[str(path.relative_to(parent))]=dict(sha256=sha256(path),bytes=path.stat().st_size)
            if len(records)%1000==0: print(f"hashed supplement artifacts {len(records)}",flush=True)
    atomic_json(output / "artifact_manifest.json",dict(status="closed",scope="Gemma R75 repair, actual M75 inference supplement, and preserved prior actual pilot arms",artifacts=records,
                preserved_prior_actual_arms=[str(path.relative_to(parent)) for path in prior_arms],
                original_manifest_sha256=original_hash,models=list(models),quality_scope="synthetic",source_sha256=sha256(Path(__file__))))
    print(f"actual supplement closed: {output / 'report.md'}; manifest SHA256 {sha256(output / 'artifact_manifest.json')}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--parent",required=True)
    run(parser.parse_args())
