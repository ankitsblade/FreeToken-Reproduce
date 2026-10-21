"""Shared control, gate and output handling for offline m-reduction stage runners."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT
from experiments.pruning.m_reduction_plan_gate import audit
from experiments.pruning.m_reduction_traces import load_model, parse_bindings, read
from experiments.pruning.moe_pruning import atomic_json, sha256

SOURCES = ("m_reduction_offline", "m_reduction_traces", "m_reduction_sim", "m_reduction_stats")


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model-study", action="append", required=True, help="MODEL_ID=STUDY binding for each model")
    p.add_argument("--models", nargs="+", help="Subset of bound models to analyze")
    p.add_argument("--out", type=Path, help="Fresh output directory (default: run outputs)")
    p.add_argument("--pre-gate-diagnostic", action="store_true",
                   help="Run before the next-stage gate passes; outputs are labelled diagnostic, not stage evidence")
    return p


def start(args, stage, module_file, parameters):
    bindings = parse_bindings(args.model_study)
    models = args.models or list(bindings)
    out = (args.out or Path(os.environ.get("EXPERIMENT_OUTPUT_ROOT", ROOT / "results/m_reduction/offline")) / stage).resolve()
    out.mkdir(parents=True, exist_ok=False)
    first = next(iter(bindings.values()))
    gate = audit(first, ROOT / "configs/m_reduction/coverage_policy.json", bindings)
    allowed = gate["next_research_stage_allowed"]
    if not allowed and not args.pre_gate_diagnostic:
        raise SystemExit(f"{stage}: next-stage gate is closed (all three models' Stage 0 and P2 support). "
                         "Use --pre-gate-diagnostic for a labelled diagnostic run.")
    status = "stage_evidence" if allowed else "pre_gate_diagnostic"
    control = {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(), "status": status,
               "models": models, "bindings": {k: str(v) for k, v in bindings.items()}, "parameters": parameters,
               "gate": {k: gate[k] for k in ("all_three_models_pilot_qualified", "all_three_models_stage0_complete",
                                             "next_research_stage_allowed", "canonical_plan_sha256")},
               "dataset_sha256": sha256(ROOT / "experiments/prompts/pruning_requests.jsonl"),
               "splits_sha256": sha256(ROOT / "experiments/prompts/pruning_splits.json"),
               "sources": {name: sha256(ROOT / f"experiments/pruning/{name}.py")
                           for name in (*SOURCES, Path(module_file).stem)},
               "interpretation": ("Offline replay/simulation on accepted native_floor1 captures; m is missing demand, "
                                  "not PCIe traffic. Fixed-logit screens cannot qualify quality.")}
    atomic_json(out / "control.json", control)
    return out, bindings, models, status


def load(bindings, model, subset="primary"):
    geometry, batch, traces = load_model(bindings[model], model, subset)
    return geometry, traces, {"batch": str(batch), "subset": subset, "captures": len(traces)}


def write_lines(path, rows):
    with Path(path).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def finish(out, summary, status):
    summary["status"] = status
    atomic_json(out / "summary.json", summary)
    print(json.dumps({"out": str(out), "status": status}, indent=2))
