"""Qualify Gemma cache transitions using its observed routes and native GPU cache kernel."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from experiments.pruning.moe_pruning import atomic_json, sha256, validate_replay
from experiments.pruning.pruning_analysis import load_model
from experiments.pruning.pruning_native_controls import cold_control


def run(args):
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise ValueError("native Gemma controls require an idle GPU")
    study = Path(args.study).resolve()
    key = "gemma_nvfp4"
    model = json.loads((study / "protocol/models.json").read_text())[key]
    for stage in ("pilot", "primary", "repeat"):
        validate_replay(argparse.Namespace(study=str(study), model=key, stage=stage))
    captures = load_model(study, key, model)
    splits = json.loads((study / "protocol/splits.json").read_text())
    rid = splits["repeat_request_ids"][0]
    capture = captures[rid]
    fold = splits["group_to_fold"][capture["manifest"]["request"]["group_id"]]
    mask = np.load(study / "retention75" / f"{key}_decode_fold{fold}_mixed_ranking.npz")["mask"]
    initial = json.loads((capture["directory"] / "decode_initial_cache.json").read_text())
    ids = capture["phases"]["decode"]["ids"]
    output = study / "native_controls75"
    output.mkdir(exist_ok=False)
    rows = []
    relative = math.ceil(model["cache_slots"] * math.ceil(.75*model["experts"]) / model["experts"])
    for slots in (model["cache_slots"], relative):
        for policy, fraction in [("calibrated_hybrid", initial["fetch_fraction"]), ("unconditional_fill", 1.)]:
            for mode, filtering in [("U", None), ("F", mask)]:
                row = {"model": key, "mode": mode, "policy": policy,
                       **cold_control(ids, model, slots, fraction, filtering)}
                rows.append(row)
                atomic_json(output / "cold_cache.json", rows)
                if row["status"] != "matched":
                    raise ValueError("native cold Gemma cache replay diverged")
    atomic_json(output / "completion.json", {"status": "complete", "cold_cases": len(rows),
                "R": "unavailable: full Gemma router logits/per-expert rescaling not captured",
                "source_sha256": sha256(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    run(parser.parse_args())
