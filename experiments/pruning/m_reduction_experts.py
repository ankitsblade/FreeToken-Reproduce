"""P2 expert pre-gate output norms and P7 common-input probes (plan sections 6 and 9.3).

Inputs are the decode routed MoE input rows saved by baseline teacher forcing. Each
layer's experts are dequantized from the pinned checkpoint and evaluated in FP32 on the
GPU: a reference expert function (weight-only dequantization, no activation
quantization), checked against the native CPU expert outputs saved by the numerical
same-state diagnostics before use. Outputs:

- norms/CAPTURE.npy: (decode events, k) pre-gate output L2 norms, route order as in
  the capture's routes stream; P7 arm 1 combines them with native effective weights.
- probes.npz: per outer fold, pool and layer, the (E, hidden) mean pre-gate output on
  128 routed-input rows from outer-training decode (43/43/42 math/code/incident for the
  mixed pool, or 128 from one domain), uniformly spaced chronologically per domain.
- probe_ids.json: the exact probe rows (capture, decode event index).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.paths import ROOT
from experiments.pruning.moe_pruning import atomic_json, sha256
from experiments.pruning.m_reduction_traces import lines, read

FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
POOLS = {"mixed": {"math": 43, "code": 43, "incident": 42}, "math": {"math": 128},
         "code": {"code": 128}, "incident": {"incident": 128}}
PREFIX = {"gpt_oss_mxfp4": "model.layers.{}.mlp.experts.", "qwen_nvfp4": "model.language_model.layers.{}.mlp.experts.",
          "gemma_nvfp4": "model.language_model.layers.{}.experts."}


class Checkpoint:
    def __init__(self, path):
        from safetensors import safe_open

        self.path = Path(path)
        self.index = read(self.path / "model.safetensors.index.json")["weight_map"]
        self.open = lambda name: safe_open(self.path / name, framework="pt", device="cpu")
        self.handles = {}

    def get(self, name):
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = self.open(shard)
        return self.handles[shard].get_tensor(name)


def unpack_fp4(packed, device):
    packed = packed.to(device)
    values = torch.stack([FP4.to(device)[(packed & 0x0F).long()], FP4.to(device)[(packed >> 4).long()]], dim=-1)
    return values.flatten(-2)


def mxfp4(blocks, scales, device):
    """GPT-OSS MXFP4: (..., groups, 16 bytes) blocks with E8M0 group exponents -> (..., in) FP32."""
    values = unpack_fp4(blocks, device)  # (..., groups, 32)
    exponent = scales.to(device).to(torch.int32) - 127
    return torch.ldexp(values, exponent.unsqueeze(-1).to(torch.float32)).flatten(-2)


def nvfp4(weight, scale, global_scale, device):
    """ModelOpt NVFP4: packed E2M1 pairs, FP8 E4M3 scale per 16 inputs, FP32 global scale."""
    values = unpack_fp4(weight, device)
    block = scale.to(device).to(torch.float32).repeat_interleave(16, dim=-1)
    return values * block * global_scale.to(device).to(torch.float32)


class Experts:
    """One layer's experts as FP32 GPU matrices; __call__(e, x) is the pre-gate output."""

    def __init__(self, model, checkpoint, layer, experts, device="cuda"):
        self.model, self.device = model, device
        prefix = PREFIX[model].format(layer)
        if model == "gpt_oss_mxfp4":
            self.gate_up = mxfp4(checkpoint.get(prefix + "gate_up_proj_blocks"),
                                 checkpoint.get(prefix + "gate_up_proj_scales"), device)
            self.gate_up_bias = checkpoint.get(prefix + "gate_up_proj_bias").to(device, torch.float32)
            self.down = mxfp4(checkpoint.get(prefix + "down_proj_blocks"), checkpoint.get(prefix + "down_proj_scales"), device)
            self.down_bias = checkpoint.get(prefix + "down_proj_bias").to(device, torch.float32)
            return

        def load(e, name):
            base = f"{prefix}{e}.{name}."
            return nvfp4(checkpoint.get(base + "weight"), checkpoint.get(base + "weight_scale"),
                         checkpoint.get(base + "weight_scale_2"), device)
        self.gate = torch.stack([load(e, "gate_proj") for e in range(experts)])
        self.up = torch.stack([load(e, "up_proj") for e in range(experts)])
        self.down = torch.stack([load(e, "down_proj") for e in range(experts)])

    def __call__(self, e, x):
        x = x.to(self.device, torch.float32)
        if self.model == "gpt_oss_mxfp4":
            gate_up = x @ self.gate_up[e].T + self.gate_up_bias[e]
            gate, up = gate_up[..., ::2].clamp(max=7.0), gate_up[..., 1::2].clamp(-7.0, 7.0)
            hidden = (up + 1) * (gate * torch.sigmoid(gate * 1.702))
            return hidden @ self.down[e].T + self.down_bias[e]
        gate, up = x @ self.gate[e].T, x @ self.up[e].T
        act = torch.nn.functional.silu(gate) if self.model == "qwen_nvfp4" else \
            torch.nn.functional.gelu(gate, approximate="tanh")
        return (act * up) @ self.down[e].T


def compare(reference, native):
    reference, native = reference.double().cpu().flatten(), native.double().cpu().flatten()
    delta = reference - native
    return {"relative_l2": float(delta.norm() / native.norm()), "max_abs": float(delta.abs().max()),
            "cosine": float(reference @ native / (reference.norm() * native.norm())),
            "norm_ratio": float(reference.norm() / native.norm())}


def validate(models, fixtures, out):
    """Reference expert outputs versus native per-route CPU/GPU outputs at one fixed state."""
    plan = read(ROOT / "configs/m_reduction/stage0.json")["models"]
    report = {}
    for model in models:
        path = Path(fixtures[model])
        state = torch.load(path / "fixed_state.pt", weights_only=False)
        layer = read(path / "comparison.json")["layer_id"]
        experts = Experts(model, Checkpoint(ROOT / plan[model]["path"]), layer, plan[model]["experts"])
        x, ids, weights = state["hidden_state"], state["ids"][0].tolist(), state["weights"][0]
        routes = []
        for route, expert in enumerate(ids):
            reference = float(weights[route]) * experts(expert, x)[0]
            routes.append({"expert": expert, "weight": float(weights[route]),
                           "versus_cpu": compare(reference, state["outputs"][f"cpu_route_{route}"][0]),
                           "versus_gpu": compare(reference, state["outputs"][f"gpu_route_{route}"][0])})
        report[model] = {"fixture": str(path), "fixture_sha256": sha256(path / "fixed_state.pt"), "layer": layer,
                         "routes": routes, "max_relative_l2_cpu": max(r["versus_cpu"]["relative_l2"] for r in routes),
                         "min_cosine_cpu": min(r["versus_cpu"]["cosine"] for r in routes)}
        del experts
        torch.cuda.empty_cache()
    atomic_json(out, report)
    print(json.dumps({m: {k: v for k, v in r.items() if k != "routes"} for m, r in report.items()}, indent=1))


def capture_rows(directory):
    header = read(directory / "moe_inputs_header.json")
    if header["dtype"] != "torch.bfloat16":
        raise ValueError("Saved MoE inputs are not BF16")
    rows = np.memmap(directory / "moe_inputs.bin", dtype=np.int16, mode="r").reshape(-1, header["width"])
    routes = [row for row in lines(directory / "routes.jsonl.gz") if row["phase"] == "decode"]
    if len(routes) != rows.shape[0]:
        raise ValueError(directory.name + ": MoE input rows and decode routes disagree")
    return rows, np.asarray([r["layer_id"] for r in routes]), np.asarray([r["expert_ids"] for r in routes]), \
        np.asarray([r["input_position"] for r in routes])


def as_tensor(rows):
    return torch.from_numpy(np.ascontiguousarray(rows)).view(torch.bfloat16)


def select_probes(meta, order, folds, workloads, layer, fold, pool):
    """Uniformly spaced chronological rows per domain from outer-training captures only."""
    chosen = []
    for domain, count in POOLS[pool].items():
        candidates = [(name, int(i)) for name in order if folds[name] != fold and workloads[name] == domain
                      for i in np.flatnonzero(meta[name][1] == layer)]
        if len(candidates) < count:
            raise ValueError(f"Probe pool {pool}/{domain} has only {len(candidates)} rows")
        picks = np.unique(np.linspace(0, len(candidates) - 1, count).round().astype(int))
        chosen += [candidates[i] for i in picks]
    return chosen


def collect(args):
    batch = args.teacher_batch.resolve()
    config = read(batch / "configuration.json")
    model = config["model_key"]
    if read(batch / "completion.json").get("status") != "complete" or config["mode"] != "baseline":
        raise ValueError("Expert outputs need a complete baseline teacher-forcing batch")
    validation = read(args.validation)
    if model not in validation:
        raise ValueError(model + ": reference expert function was not validated")
    geometry = config["model"]
    source = Path(config["dependencies"]["source_batch"]).parents[4]
    dataset = [row for row in lines(source / "protocol/dataset.jsonl")]
    order_ids = [row["request_id"] for row in dataset]
    group_fold = read(source / "protocol/splits.json")["group_to_fold"]
    by_id = {row["request_id"]: row for row in dataset}
    captures = sorted((p for p in batch.iterdir() if (p / "manifest.json").is_file()),
                      key=lambda p: order_ids.index(read(p / "manifest.json")["request"]["request_id"]))
    names = [p.name for p in captures]
    request = {p.name: by_id[read(p / "manifest.json")["request"]["request_id"]] for p in captures}
    folds = {name: int(group_fold[request[name]["group_id"]]) for name in names}
    workloads = {name: request[name]["workload"] for name in names}
    meta = {p.name: capture_rows(p) for p in captures}
    out = args.out.resolve()
    (out / "norms").mkdir(parents=True, exist_ok=False)
    atomic_json(out / "control.json", {"stage": "p2_expert_outputs", "model": model, "teacher_batch": str(batch),
                "teacher_completion_sha256": sha256(batch / "completion.json"),
                "validation": str(args.validation), "validation_sha256": sha256(args.validation),
                "source_sha256": sha256(Path(__file__)), "pools": POOLS, "probe_rows_per_pool": 128,
                "reference": "weight-only dequantized checkpoint experts, FP32 GPU, no activation quantization"})
    checkpoint = Checkpoint(ROOT / geometry["path"])
    norms = {name: np.full(meta[name][2].shape, np.nan, dtype=np.float32) for name in names}
    probes, probe_ids = {}, {}
    torch.backends.cuda.matmul.allow_tf32 = False
    for layer in range(geometry["layers"]):
        experts = Experts(model, checkpoint, layer, geometry["experts"])
        index = [(name, np.flatnonzero(meta[name][1] == layer)) for name in names]
        x = torch.cat([as_tensor(meta[name][0][rows]) for name, rows in index]).to("cuda")
        ids = np.concatenate([meta[name][2][rows] for name, rows in index])
        owner = np.concatenate([[k] * len(rows) for k, (_, rows) in enumerate(index)])
        local = np.concatenate([rows for _, rows in index])
        for e in range(geometry["experts"]):
            row, slot = np.nonzero(ids == e)
            if len(row):
                values = experts(e, x[torch.from_numpy(row).to("cuda")]).norm(dim=-1).cpu().numpy()
                for value, r, s in zip(values, row, slot):
                    norms[names[owner[r]]][local[r], s] = value
        sets = {(fold, pool): select_probes(meta, names, folds, workloads, layer, fold, pool)
                for fold in sorted(set(folds.values())) for pool in POOLS}
        union = sorted({item for chosen in sets.values() for item in chosen})
        position = {item: i for i, item in enumerate(union)}
        inputs = torch.cat([as_tensor(meta[name][0][[i]]) for name, i in union]).to("cuda")
        members = {key: torch.tensor([position[item] for item in chosen], device="cuda") for key, chosen in sets.items()}
        means = {key: np.zeros((geometry["experts"], inputs.shape[1]), dtype=np.float32) for key in sets}
        for e in range(geometry["experts"]):
            output = experts(e, inputs)
            for key, rows in members.items():
                means[key][e] = output[rows].mean(dim=0).cpu().numpy()
        for (fold, pool), value in means.items():
            probes[f"{model}_fold{fold}_{pool}_{layer}"] = value
            probe_ids[f"fold{fold}_{pool}_{layer}"] = [[name, int(meta[name][3][i]), i] for name, i in sets[(fold, pool)]]
        del experts, x, inputs
        torch.cuda.empty_cache()
        print(f"{model}: layer {layer + 1}/{geometry['layers']} norms and {len(sets)} probe sets", flush=True)
    for name, value in norms.items():
        if np.isnan(value).any():
            raise ValueError(name + ": some routed expert norms were not computed")
        np.save(out / "norms" / (name + ".npy"), value)
    np.savez(out / "probes.npz", **probes)
    atomic_json(out / "probe_ids.json", {"columns": ["capture", "input_position", "decode_event_index"], "sets": probe_ids})
    atomic_json(out / "completion.json", {"status": "complete", "captures": len(names), "layers": geometry["layers"],
                "probe_sets": len(probes), "probes_sha256": sha256(out / "probes.npz")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="Check the reference experts against native fixtures")
    parser.add_argument("--fixture", action="append", default=[], help="MODEL=DIR with fixed_state.pt and comparison.json")
    parser.add_argument("--teacher-batch", type=Path)
    parser.add_argument("--validation", type=Path, help="Validation report for collection, or output path with --validate")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.validate:
        fixtures = dict(entry.split("=", 1) for entry in args.fixture)
        validate(sorted(fixtures), fixtures, args.validation)
        return
    if not args.teacher_batch or not args.validation or not args.out:
        parser.error("--teacher-batch, --validation and --out are required")
    collect(args)


if __name__ == "__main__":
    main()
