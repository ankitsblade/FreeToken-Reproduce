"""Load accepted m-reduction captures as decode access streams for offline analysis.

Only whole batches that completed, passed resource gates and match the model's
qualified native_floor1 pilot dependencies are read; rejected attempts are ignored.
"""

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from experiments.paths import ROOT

COUNTS = {"primary": 90, "repeat": 18}


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def parse_bindings(entries):
    bindings = {}
    for entry in entries:
        model, path = entry.split("=", 1)
        if model in bindings:
            raise ValueError("Repeated model binding: " + model)
        bindings[model] = Path(path).resolve()
    return bindings


def accepted_batch(study, model, subset):
    reference = read(study / "qualification" / model / "native_floor1.json")
    if reference.get("status") != "qualified":
        raise ValueError(model + ": native_floor1 pilot is not qualified in " + str(study))
    for path in sorted((study / "captures" / model / "native_floor1" / subset).glob("*"), reverse=True):
        completion, config = path / "completion.json", path / "configuration.json"
        if not completion.is_file() or not config.is_file():
            continue
        done, control = read(completion), read(config)
        if (done.get("status") == "complete" and done.get("captures") == COUNTS[subset]
                and done.get("resources", {}).get("status") == "passed"
                and control["inference_hashes"] == reference["inference_hashes"]
                and control["dependencies"] == reference["dependencies"]):
            return path
    raise FileNotFoundError(f"{model}: no accepted {subset} batch in {study}")


@dataclass
class Geometry:
    model: str
    layers: int
    experts: int
    top_k: int
    cache_slots: int
    expert_bytes: int
    fetch_fraction: float = 0.0
    rho: float = 0.0


@dataclass
class Trace:
    capture_id: str
    request_id: str
    workload: str
    fold: int
    repetition: int
    prompt_tokens: int
    directory: Path
    initial: dict
    layer: np.ndarray
    position: np.ndarray
    ids: list
    live_m: np.ndarray
    live_q: np.ndarray
    _scores: np.ndarray = field(default=None, repr=False)
    _routes: dict = field(default=None, repr=False)

    @property
    def token(self):
        return self.position - self.prompt_tokens

    def scores(self, experts):
        # Full native router logits aligned with the decode cache events.
        if self._scores is None:
            index = {(int(p), int(l)): i for i, (p, l) in enumerate(zip(self.position, self.layer))}
            values = np.full((len(self.layer), experts), np.nan, dtype=np.float32)
            for row in lines(self.directory / "scores.jsonl.gz"):
                if row["phase"] == "decode":
                    slot = index.get((row["input_position"], row["layer_id"]))
                    if slot is not None:
                        values[slot] = row["scores"]
            if np.isnan(values).any():
                raise ValueError(self.capture_id + ": decode events lack full router scores")
            self._scores = values
        return self._scores

    def router_scales(self):
        """Per-layer per-expert output scales (Gemma); empty when the router has none."""
        path = self.directory / "router_metadata.json"
        if not path.is_file():
            return {}
        return {int(layer): row["per_expert_scale"] for layer, row in read(path).items() if "per_expert_scale" in row}

    def routes(self):
        # Native selected IDs and effective weights for each decode event.
        if self._routes is None:
            self._routes = {(row["input_position"], row["layer_id"]): (row["expert_ids"], row["effective_weights"])
                            for row in lines(self.directory / "routes.jsonl.gz")
                            if row["phase"] == "decode" and not row.get("is_shared_expert")}
        return self._routes


def geometry(study, model):
    plan = read(study / "protocol/plan.json")["models"][model]
    profile = None
    policy = sorted((study / "captures" / model / "native_floor1" / "pilot").glob("*/policy.json"))
    if policy:
        profile = read(policy[-1])
    return Geometry(model, plan["layers"], plan["experts"], plan["top_k"], plan["cache_slots"],
                    plan["expert_bytes"], rho=(profile or {}).get("rho", 0.0))


def calibration(study, model):
    """The bandwidth profile whose hash the qualified native_floor1 pilot recorded."""
    import hashlib

    wanted = read(study / "qualification" / model / "native_floor1.json")["dependencies"]["calibration"]
    for path in sorted((study / "calibration").glob("*.json")):
        if not path.name.endswith(".control.json") and hashlib.sha256(path.read_bytes()).hexdigest() == wanted:
            profile = read(path)
            fmt = "mxfp4_triton" if model == "gpt_oss_mxfp4" else "nvfp4"
            entry = next(w["kernels"][fmt] for w in profile["workloads"].values() if fmt in w["kernels"])
            return {"path": str(path), "sha256": wanted, **{k: entry[k] for k in
                    ("cpu_moe_gbs", "pcie_gather_gbs", "cpu_moe_overlap_gbs", "pcie_gather_overlap_gbs")}}
    raise FileNotFoundError(model + ": calibration profile bound by the qualified pilot is missing")


def load_trace(directory, request, fold):
    manifest = read(directory / "manifest.json")
    if manifest["validation"]["status"] != "valid" or manifest["validation"]["replay"]["status"] != "matched":
        raise ValueError(str(directory) + ": capture did not pass validation and replay")
    initial = read(directory / "decode_initial_cache.json")
    layer, position, ids, live_m, live_q = [], [], [], [], []
    for row in lines(directory / "cache.jsonl.gz"):
        if row["phase"] != "decode":
            continue
        layer.append(row["layer_id"])
        position.append(row["input_position"])
        ids.append(tuple(dict.fromkeys(row["raw_ids"])))
        live_m.append(row["m"])
        live_q.append(row["q"])
    capture_id = directory.name
    return Trace(capture_id, request["request_id"], request["workload"], fold,
                 int(capture_id.rsplit("_r", 1)[-1]), manifest["validation"]["prompt_tokens"], directory, initial,
                 np.asarray(layer, dtype=np.int32), np.asarray(position, dtype=np.int64), ids,
                 np.asarray(live_m, dtype=np.int32), np.asarray(live_q, dtype=np.int32))


def load_model(study, model, subset="primary"):
    study = Path(study).resolve()
    rows = {row["request_id"]: row for row in lines(study / "protocol/dataset.jsonl")}
    folds = read(study / "protocol/splits.json")["group_to_fold"]
    batch = accepted_batch(study, model, subset)
    traces = []
    for directory in sorted(p for p in batch.iterdir() if (p / "manifest.json").is_file()):
        request = rows[read(directory / "manifest.json")["request"]["request_id"]]
        traces.append(load_trace(directory, request, int(folds.get(request["group_id"], -1))))
    if len(traces) != COUNTS[subset]:
        raise ValueError(f"{model}: expected {COUNTS[subset]} {subset} captures, found {len(traces)}")
    if subset == "primary":
        workloads = {}
        for trace in traces:
            workloads.setdefault(trace.workload, set()).add(trace.fold)
        if any(trace.fold < 0 for trace in traces) or any(len(f) != 5 for f in workloads.values()):
            raise ValueError(model + ": primary captures do not cover the frozen five-fold manifest")
    return geometry(study, model), batch, traces


def splits():
    return read(ROOT / "experiments/prompts/pruning_splits.json")
