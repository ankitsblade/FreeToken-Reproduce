"""P7 deployment masks fitted on all 90 D0 primary captures (plan sections 3.1 and 9.3).

Outer-fold artefacts from m_reduction_p7 serve D0 held-out evaluation. Secondary sets
(D1-D3) use one frozen deployment artefact per model, fitted on D0 only with the same
rules: arm 1 miss-targeted saliency removal and the R75 calibration-frequency reference.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.m_reduction_p4 import rules
from experiments.pruning.m_reduction_p7 import REMOVE, frequency_removal, saliency_removal, training_counts
from experiments.pruning.m_reduction_traces import calibration, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-study", type=Path, required=True)
    parser.add_argument("--expert-outputs", type=Path, required=True, help="m_reduction_experts output for the model")
    parser.add_argument("--q-arm", default="overlap_floor1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(str(args.out))
    if json.loads((args.expert_outputs / "completion.json").read_text()).get("status") != "complete":
        raise ValueError("expert outputs are incomplete")
    geometry, batch, traces = load_model(args.source_study, args.model)
    table, _ = rules(geometry, calibration(args.source_study, args.model))
    counts = training_counts(traces, geometry, table, args.q_arm)
    artefact = {"model": args.model, "fit": "all 90 D0 primary captures", "q_arm": args.q_arm,
                "remove_per_layer": REMOVE[args.model], "source_batch": str(batch),
                "expert_outputs": str(args.expert_outputs.resolve()),
                "expert_outputs_completion_sha256": sha256(args.expert_outputs / "completion.json"),
                "source_sha256": sha256(Path(__file__)),
                "p7_source_sha256": sha256(Path(__file__).with_name("m_reduction_p7.py")),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "arm1_removed": {str(l): v for l, v in saliency_removal(traces, geometry, counts, args.expert_outputs).items()},
                "r75_removed": {str(l): v for l, v in frequency_removal(counts, geometry).items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, artefact)
    print(json.dumps({"out": str(args.out), "layers": len(artefact["arm1_removed"])}))


if __name__ == "__main__":
    main()
