"""Export paired statistics, scientific figures, masking decisions and immutable 75% closeout."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from experiments.pruning.moe_pruning import atomic_json, sha256, snapshot
from experiments.pruning.pruning_analysis import WORKLOADS, interval, write_csv


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def require_complete(path):
    value = json.loads(path.read_text())
    if value.get("status") != "complete":
        raise ValueError(f"required analysis incomplete: {path}")
    return value


def figures(output, rows_by_model):
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "moe_pruning_75_20261006"
    import matplotlib.pyplot as plt

    directory = output / "figures"
    directory.mkdir()
    metadata = {"scope": "held-out request means; CV folds are not independent replicates",
                "formats": ["svg", "pdf", "png"], "dpi": 180,
                "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
                "source_sha256": sha256(Path(__file__)), "svg_hashsalt": matplotlib.rcParams["svg.hashsalt"],
                "source_tables": {}}

    def save(fig, name):
        fig.tight_layout()
        for extension in metadata["formats"]:
            fig.savefig(directory / f"{name}.{extension}", dpi=metadata["dpi"])
        plt.close(fig)

    for model, tables in rows_by_model.items():
        source, summaries, layers, windows, stability = tables
        for filename in ("summary.csv", "layer.csv", "windows.csv", "stability.csv"):
            metadata["source_tables"][str(source / filename)] = sha256(source / filename)
        for phase in ("prefill", "decode"):
            matrix = np.empty((4, 3))
            for i, pool in enumerate((*WORKLOADS, "mixed")):
                for j, workload in enumerate(WORKLOADS):
                    cell = next(r for r in summaries if r["calibration_pool"] == pool and
                                r["ranking_phase"] == phase and r["evaluation_phase"] == phase and
                                r["evaluation_workload"] == workload)
                    matrix[i, j] = float(cell["mean"])
            fig, ax = plt.subplots(figsize=(6.5, 4))
            im = ax.imshow(matrix, vmin=.5, vmax=1, cmap="viridis")
            ax.set_xticks(range(3), WORKLOADS)
            ax.set_yticks(range(4), (*WORKLOADS, "mixed"))
            ax.set_xlabel("Held-out workload (request counts in summary.csv)")
            ax.set_ylabel("Calibration workload")
            ax.set_title(f"{model}: 75% {phase} access coverage")
            for i in range(4):
                for j in range(3):
                    ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center", color="white" if matrix[i,j] < .8 else "black")
            fig.colorbar(im, ax=ax, label="Macro request coverage")
            save(fig, f"{model}_{phase}_transfer")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, phase in zip(axes, ("prefill", "decode")):
            selected = [r for r in layers if r["ranking_phase"] == "decode" and r["evaluation_phase"] == phase and r["calibration_pool"] == "mixed"]
            group = {}
            for row in selected:
                group.setdefault(int(row["layer"]), []).append(float(row["access_coverage"]))
            indices = sorted(group)
            counts = {len(group[i]) for i in indices}
            label = f"Mean across {next(iter(counts))} requests" if len(counts) == 1 else "Mean; counts vary by layer"
            ax.plot(indices, [np.mean(group[i]) for i in indices], label=label)
            ax.plot(indices, [np.quantile(group[i], .05) for i in indices], label="Request p5")
            ax.set(title=f"{phase}, same mixed decode mask", xlabel="Original layer ID", ylabel="Access coverage", ylim=(0, 1.02))
            ax.legend(fontsize=8)
        save(fig, f"{model}_layer")
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        position_groups = {}
        for row in windows:
            if row["ranking_phase"] == "decode" and row["evaluation_phase"] == "decode" and row["calibration_pool"] == "mixed":
                position_groups.setdefault(int(row["phase_token_index"]), []).append(row)
        positions = sorted(position_groups)
        for field, label in [("token_coverage", "Per-token"), ("cumulative_coverage", "Cumulative"), ("rolling64", "Full 64-token window")]:
            x, y = [], []
            for index in positions:
                values = [float(r[field]) for r in position_groups[index] if r[field]]
                if values:
                    x.append(index); y.append(np.mean(values))
            axes[0].plot(x, y, label=label)
        axes[0].set(ylabel="Macro request access coverage", ylim=(0, 1.02), title=f"{model}: mixed decode masks, observed decode positions")
        axes[0].legend()
        axes[1].plot(positions, [len(position_groups[i]) for i in positions])
        axes[1].set(xlabel="Forwarded decode-token index", ylabel="Observed requests")
        save(fig, f"{model}_positions")
        selected = [r for r in stability if r["kind"] == "disjoint_halves" and r["phase"] == "decode" and r["pool"] == "mixed"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist([float(r["jaccard"]) for r in selected], bins=25, alpha=.7, label="Disjoint calibration halves")
        ax.axvline(np.mean([float(r["random_reference"]) for r in selected]), color="black", linestyle="--", label="Size-matched random reference")
        ax.set(xlabel="Per-layer retained-ID Jaccard", ylabel="Dependent subset/layer comparisons", title=f"{model}: 75% mixed decode stability")
        ax.legend(fontsize=8)
        save(fig, f"{model}_stability")
    atomic_json(directory / "plot_configuration.json", metadata)


def run(args):
    study, gemma, qwen_repeat = map(lambda p: Path(p).resolve(), [args.study, args.gemma_study, args.qwen_repeat_study])
    model_sources = {"qwen_nvfp4": study, "gpt_oss_mxfp4": study, "gemma_nvfp4": gemma}
    for key, source in model_sources.items():
        require_complete(source / "retention75/analysis_complete.json")
        require_complete(source / "replay75/completion.json")
        require_complete(source / "native_controls75/completion.json")
        require_complete(study / "repeatability75" / key / "completion.json")
        qualification = json.loads((source / "pilots" / key / "qualification.json").read_text())
        if qualification["status"] != "qualified" or qualification["captures"] != 15:
            raise ValueError("model pilot not qualified")
    if json.loads((qwen_repeat / "pilots/qwen_nvfp4/qualification.json").read_text())["status"] != "qualified":
        raise ValueError("replacement Qwen allocation arm is not qualified")
    output = study / "closeout75"
    output.mkdir(exist_ok=False)
    snapshot(output / "source_snapshot")
    paired, gates, report_rows, table_inputs = [], [], [], {}
    resource_rows, model_locks = [], []
    request_tables = {}
    seed_protocol = json.loads((study / "protocol/protocol.json").read_text())
    for key, source in model_sources.items():
        model = json.loads((source / "protocol/models.json").read_text())[key]
        model_locks.append({"model": key, "checkpoint": model["repository"], "revision": model["revision"],
                            "layers": model["layers"], "experts_per_layer": model["experts"],
                            "native_top_k": model["top_k"], "retained75_per_layer": math.ceil(.75*model["experts"]),
                            "expert_bytes": model["expert_bytes"], "cache_slots": model["cache_slots"],
                            "cache_bytes": model["expert_bytes"]*model["cache_slots"]})
        qualified = json.loads((source / "pilots" / key / "qualification.json").read_text())
        repeat = json.loads((study / "repeatability75" / key / "completion.json").read_text())
        primary_batches = [path for path in (source / "routes" / key).iterdir()
                           if path.is_dir() and (path / "completion.json").exists() and
                           json.loads((path / "completion.json").read_text()).get("captures") == 90 and
                           json.loads((path / "completion.json").read_text()).get("resource_status") == "passed"]
        if len(primary_batches) != 1:
            raise ValueError(f"{key}: primary resource provenance is ambiguous")
        batch_sources = [("pilot", source / "pilots" / key / qualified["batch_id"]),
                         ("primary", primary_batches[0]),
                         ("repeat", Path(repeat["repeat_study"]) / "routes" / key / repeat["repeat_batch"])]
        if Path(repeat["repeat_study"]) != source:
            arm = Path(repeat["repeat_study"])
            arm_qualification = json.loads((arm / "pilots" / key / "qualification.json").read_text())
            batch_sources.append(("repeat_arm_pilot", arm / "pilots" / key / arm_qualification["batch_id"]))
        for stage, batch in batch_sources:
            completion = json.loads((batch / "completion.json").read_text())
            resource_rows.append({"model": key, "stage": stage, "batch": str(batch.relative_to(study)),
                                  "captures": completion["captures"], **json.loads((batch / "resource_summary.json").read_text())})
        analysis = source / "retention75"
        def model_rows(filename):
            return [row for row in read_csv(analysis / filename) if row["model"] == key]

        summaries, requests = model_rows("summary.csv"), model_rows("request.csv")
        layers, windows, stability = (model_rows(name) for name in ("layer.csv", "windows.csv", "stability.csv"))
        oracle = model_rows("oracle.csv")
        if len({row["request_id"] for row in requests}) != 90:
            raise ValueError(f"{key}: report needs all 90 independent primary requests")
        request_tables[key] = requests
        table_inputs[key] = (analysis, summaries, layers, windows, stability)
        for row in summaries:
            if row["ranking_phase"] == row["evaluation_phase"] == "decode" and row["calibration_pool"] == row["evaluation_workload"]:
                reference = [r for r in oracle if r["phase"] == "decode" and r["workload"] == row["evaluation_workload"]]
                report_rows.append({"model": key, "workload": row["evaluation_workload"],
                                   "requests": int(row["requests"]), "heldout_mean": float(row["mean"]), "ci_low": float(row["ci_low"]),
                                   "ci_high": float(row["ci_high"]), "micro": float(row["micro_coverage"]),
                                   "gate_mass": float(row["gate_mass_mean"]),
                                   "oracle_mean": float(np.mean([float(r["oracle_coverage"]) for r in reference])),
                                   "random_reference_mean": float(np.mean([float(r["random_mean"]) for r in reference]))})
        coverage = [r for r in summaries if r["ranking_phase"] == "decode" and r["calibration_pool"] == "mixed"]
        failed = []
        for row in coverage:
            context = row["evaluation_phase"] + "/" + row["evaluation_workload"]
            if float(row["ci_low"]) < .98: failed.append(context + ": access CI lower < .98")
            if float(row["gate_mass_ci_low"]) < .99: failed.append(context + ": effective gate mass CI lower < .99")
            if float(row["p5"]) < .95: failed.append(context + ": request p5 < .95")
        for phase in ("prefill", "decode"):
            group = {}
            for row in layers:
                if row["ranking_phase"] == "decode" and row["evaluation_phase"] == phase and row["calibration_pool"] == "mixed":
                    group.setdefault((row["evaluation_workload"], row["layer"]), []).append(float(row["access_coverage"]))
            if min(map(np.mean, group.values())) < .95: failed.append(phase + ": a workload/layer mean < .95")
            candidates = [float(r["rolling64"]) for r in windows if r["ranking_phase"] == "decode" and r["evaluation_phase"] == phase and r["calibration_pool"] == "mixed" and r["rolling64"]]
            if candidates and min(candidates) < .90: failed.append(phase + ": a full 64-token window < .90")
        subset = [r for r in stability if r["kind"] == "disjoint_halves" and r["phase"] == "decode" and r["pool"] == "mixed"]
        excess = float(np.mean([float(r["excess_jaccard"]) for r in subset]))
        if excess < .10: failed.append("mean disjoint Jaccard excess < .10")
        replay = [row for row in read_csv(source / "replay75/summary.csv") if row["model"] == key]
        systems = [r for r in replay if r["mode"] == "R" and r["capacity"] == "fixed_bytes" and r["policy"] == "calibrated_hybrid" and r["state"] == "cold"]
        if len(systems) != 3 or any(not r["relative_m_reduction"] or float(r["relative_m_reduction"]) < .05 or float(r["paired_delta_ci_high"]) >= 0 for r in systems):
            failed.append("native R fixed-byte hybrid benefit screen not satisfied in every workload")
        if not failed:
            raise ValueError(f"{key} clears activation screen; independent quality work is required before closeout")
        gates.append({"model": key, "status": "skipped_by_gate", "failed_checks": failed,
                      "disjoint_jaccard_excess": excess, "quality": "not run; actual masking not activated"})
    for left, right in itertools.combinations(model_sources, 2):
        keys = ("ranking_phase", "evaluation_phase", "calibration_pool", "evaluation_workload")
        left_rows = {(tuple(row[k] for k in keys), row["request_id"]): row for row in request_tables[left]}
        right_rows = {(tuple(row[k] for k in keys), row["request_id"]): row for row in request_tables[right]}
        if len(left_rows) != len(request_tables[left]) or len(right_rows) != len(request_tables[right]):
            raise ValueError("duplicate independent request in a model comparison cell")
        for group in sorted({item[0] for item in left_rows}):
            ids = sorted(rid for g, rid in left_rows if g == group and (g, rid) in right_rows)
            differences = [float(left_rows[group, rid]["access_coverage"])-float(right_rows[group, rid]["access_coverage"]) for rid in ids]
            mass = [float(left_rows[group, rid]["gate_mass"])-float(right_rows[group, rid]["gate_mass"]) for rid in ids]
            ci = interval(differences, repeats=seed_protocol["bootstrap_resamples"])
            mass_ci = interval(mass, repeats=seed_protocol["bootstrap_resamples"])
            paired.append({"model_left": left, "model_right": right, **dict(zip(keys, group)),
                           "matched_requests": len(ids), "available_left": sum(g == group for g, _ in left_rows),
                           "available_right": sum(g == group for g, _ in right_rows),
                           "paired_coverage_delta": float(np.mean(differences)), "ci_low": ci[0], "ci_high": ci[1],
                           "paired_mass_delta": float(np.mean(mass)), "mass_ci_low": mass_ci[0], "mass_ci_high": mass_ci[1]})
    if len(report_rows) != 9 or len({(row["model"], row["workload"]) for row in report_rows}) != 9:
        raise ValueError("primary summary requires one cell per checkpoint and workload")
    write_csv(output / "primary_summary.csv", report_rows)
    write_csv(output / "checkpoint_lock.csv", model_locks)
    write_csv(output / "qualified_resources.csv", resource_rows)
    write_csv(output / "paired_models.csv", paired)
    atomic_json(output / "masking_decisions.json", gates)
    failed_batches = {}
    for filename in ("batch_failure.json", "qualification_failure.json", "resource_rejection.json"):
        for path in sorted(study.rglob(filename)):
            relative = str(path.parent.relative_to(study))
            failed_batches.setdefault(relative, {})[filename] = json.loads(path.read_text())
    atomic_json(output / "failed_batch_inventory.json", failed_batches)
    figures(output, table_inputs)
    ledger = []
    for model in model_sources:
        for experiment in ["primary90", "repeat18", "coverage75", "transfer", "sequence_layer", "stability", "observed_cache_validation", "cold_cache_validation", "filtered_cache", "paired_model_comparison", "repeatability"]:
            ledger.append({"model": model, "experiment": experiment, "status": "complete", "reason": "", "evidence": str(model_sources[model].relative_to(study)) or "."})
        ledger.append({"model": model, "experiment": "rerouted_cache", "status": "complete" if model != "gemma_nvfp4" else "blocked",
                       "reason": "mixed decode masks on preselected9 static inputs" if model != "gemma_nvfp4" else "full Gemma router scores/scaling not captured; F completed with dropping-work limits", "evidence": "replay75/completion.json"})
        ledger.append({"model": model, "experiment": "actual_mask_quality_performance", "status": "skipped_by_gate", "reason": "prospective activation checks failed", "evidence": "closeout75/masking_decisions.json"})
    ledger.append({"model": "kimi_k25", "experiment": "current_study", "status": "excluded_by_user", "reason": "user instructed ignore Kimi for now", "evidence": "protocol/scope_amendment_gemma_first.json"})
    write_csv(output / "ledger.csv", ledger)
    lines = ["# 75% expert retention study", "", "## Scope", "",
             "Three checkpoint configurations were measured through FreeToken on the RTX 4070: Qwen NVFP4, GPT-OSS MXFP4, and Gemma NVFP4. Kimi is excluded at the user's request.", "",
             "There are 90 distinct primary requests per checkpoint (30 math, 30 code, 30 incident), five global grouped folds, and 18 extra captures of nine preselected requests per checkpoint. Repeats are measurement sensitivity, not independent tasks.", "",
             "## Primary results", "", "Same-workload, decode-derived held-out macro access coverage. Intervals resample independent requests, conditional on fitted masks; overlapping training folds are not independent datasets.", "",
             "| Model | Workload | Held-out coverage (95% CI) | Oracle coverage | Effective gate mass |", "|---|---|---|---|---|"]
    for row in report_rows:
        lines.append(f"| {row['model']} | {row['workload']} | {row['heldout_mean']:.4f} ({row['ci_low']:.4f}, {row['ci_high']:.4f}) | {row['oracle_mean']:.4f} | {row['gate_mass']:.4f} |")
    lines += ["", "## Masking decision", "", "No checkpoint passed all prospective static-mask activation checks. Actual masked-model inference, task quality grading and serving-performance qualification were skipped by gate. The activation table lists each failed or unavailable check.", "",
              "## Cache interpretation", "", "U preserves original accesses. F removes accesses to excluded experts and reports variable active k, removed work, miss demand and fill traffic separately. R uses native masked routers at recorded hidden states on nine preselected Qwen/GPT requests; it does not propagate changed hidden states. Gemma R is unavailable because full scores and router scaling were not captured. Both fixed-byte and relative-fraction controls, calibrated hybrid and unconditional-fill diagnostics, and cold/post-prefill states are retained.", "",
              "Cache transitions matched native FreeToken GPU kernels in controlled cold replays, and warm baseline replay matched accepted primary events. Simulated demand is not measured latency or a serving speedup. Unconditional-fill results cannot be relabeled as automatic-hybrid results.", "",
              "## Qualification and limitations", "", "Qwen/GPT primary data retain their original qualified v5 configurations. Qwen repeat data use the separately qualified allocator v6 arm; repeatability tables disclose that sensitivity. Gemma uses text-only mode, memory ratio .8, 768 slots, a 1280-token sequence limit and 1024-token prefill limit. Exact commands, source/control hashes and checkpoint provenance accompany each batch.", "",
              "Gemma effective selected weights include per-expert rescaling and are not probabilities. Its reported mass is the retained fraction of observed effective weight. Its native automatic hybrid policy reuses the frozen profile's NVFP4 fraction measured on Qwen geometry; this does not qualify a Gemma-specific optimal fetch split. Shared branches are excluded from expert rankings and remain active. Prompt tokenizations and generated sequences differ across checkpoints; paired prompt comparisons do not establish matched hidden states or causal architecture effects.", "",
              "The locally authored routing corpus has no reference-answer rubric. Quality is unmeasured. Short sequences contribute only complete rolling windows; figure sample counts expose missing positions. Confidence intervals describe request variation conditional on fitted rankings; calibration-size sensitivity uses disjoint halves, with chance/tie controls in the source tables.", "",
              "Failed Gemma startup/admission attempts, its resource-rejected partial primary batch, and rejected Qwen repeat batches are preserved and excluded from accepted counts. The failed-batch inventory records their dispositions. No tests are claimed by this report; its evidence consists of actual qualification captures, route integrity checks and native hardware controls.", "",
              "## Artifacts", "", "Primary summary, exact checkpoint/cache locks, per-batch qualification resources (including startup swapping and absolute server swap stock), paired checkpoint comparisons, activation checks, ledger, source snapshot and SVG/PDF/PNG figures are in this directory. Per-request/layer/window/transfer/stability/ranking-sensitivity tables remain under each study's retention75 directory; replay and repeatability have separate tables and completion records. The artifact manifest freezes their hashes.", "",
              "60% and 50% have not been run as part of this 75% closeout. They require separate sequential ablation stages using the same frozen rankings and requests."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    protocol_ledger = json.loads((study / "protocol/ledger.json").read_text())
    for row in protocol_ledger:
        if row["model_key"] in model_sources:
            if row["phase"] in (3, 4, 5, 7):
                row.update(status="complete", reason="75% closeout completed; endpoint-specific omissions remain explicit in its ledger",
                           evidence="closeout75/ledger.csv")
            elif row["phase"] == 6:
                row.update(status="skipped_by_gate", reason="prospective actual-mask activation checks failed",
                           evidence="closeout75/masking_decisions.json")
    atomic_json(study / "protocol/ledger.json", protocol_ledger)
    final_state = {"phase": 7, "status": "75_closed", "primary_closed": True,
                "primary_accepted_requests": 270, "accepted_extra_repeat_captures": 54, "models": list(model_sources),
                "kimi": "excluded_by_user", "missing_endpoints": ["Gemma static-input R", "actual masked quality/performance skipped by gate"],
                "deferred_60_50": "not executed; 75% closeout prerequisite fulfilled"}
    records = {}
    for path in sorted(study.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(study)
        if relative.parts[0] == "orchestration" or relative.as_posix() in ("closeout75/artifact_manifest.json", "protocol/state.json", "protocol/analysis_progress.json") or path.name.endswith("launcher.log"):
            continue
        records[str(relative)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    atomic_json(output / "artifact_manifest.json", {"primary_closed": True, "retention": .75, "scope": "three measured checkpoint configurations; Kimi excluded by user; Gemma R missing; actual masking skipped by gate",
                "accepted_primary_requests": 270, "accepted_extra_repeat_captures": 54, "models": list(model_sources),
                "excluded_mutable_artifacts": ["orchestration/**", "*launcher.log", "protocol/state.json", "protocol/analysis_progress.json"], "artifacts": records})
    final_state["artifact_manifest_sha256"] = sha256(output / "artifact_manifest.json")
    atomic_json(study / "protocol/state.json", final_state)
    atomic_json(study / "protocol/analysis_progress.json", {"primary_closed": True,
                "coverage75": "complete for Qwen, GPT-OSS and Gemma; 270 accepted primary requests",
                "repeatability75": "complete; 54 qualified extra captures of 27 preselected model/request pairs",
                "cache75": "native warm/cold validation and F controls complete; Qwen/GPT R on preselected9 only; Gemma R unavailable",
                "kimi": "excluded_by_user", "report": "closeout75/report.md", "deferred_60_50": "not run"})
    print("75% study closed: 270 primary requests and 54 qualified extra repeat captures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--gemma-study", required=True)
    parser.add_argument("--qwen-repeat-study", required=True)
    run(parser.parse_args())
