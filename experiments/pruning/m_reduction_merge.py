"""P7 arm 2 expert merging and requantization at load time (plan section 9.3, steps 4-5).

The runtime streams each MoE layer's experts as checkpoint-form pieces before packing
them into host banks. With FREETOKEN_M_REDUCTION_MERGE pointing at a merge control, the
pieces of a layer are buffered until complete; every multi-member cluster's matrices are
dequantized, averaged with normalized calibration usage weights (equal weights for an
all-zero-usage cluster), rounded to BF16, requantized to the checkpoint's own format and
written into the cluster representative's rows. Members other than the representative
keep their original rows but are never routed to: the router maps every logical ID to its
physical representative after native top-k and effective-weight computation.

Formats (identical code paths for the identity requantization-only control):
- MXFP4 (GPT-OSS): E2M1 codes, one E8M0 exponent per 32 inputs, OCP MX shared exponent
  floor(log2(amax)) - 2, round-to-nearest-even onto the E2M1 grid with saturation. Biases
  are averaged in BF16.
- NVFP4 (Qwen/Gemma, ModelOpt): E2M1 codes, FP8 E4M3 scale per 16 inputs and a per-tensor
  global scale amax / (6 x 448), held as fp16 as the banks store it.

``--build`` writes one control per outer fold (alias from the P7 artefacts, usage from that
fold's training decode accesses) plus the requantization-only control; ``--identity-check``
reports how many checkpoint bytes requantize(dequantize(x)) reproduces.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

from experiments.pruning.m_reduction_experts import Checkpoint, mxfp4, nvfp4, unpack_fp4

GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
MIDPOINTS = (GRID[1:] + GRID[:-1]) / 2
TIE_UP = torch.tensor([0.75, 1.75, 3.5])  # midpoints whose upper neighbour has the even code


def e2m1_codes(x):
    """Round to nearest E2M1 magnitude (ties to the even code), saturating; 4-bit sign|magnitude.

    Bucketize against the midpoints (no per-grid broadcast, so whole layers fit in memory):
    a value on a midpoint stays in the lower bucket unless the upper code is the even one.
    """
    magnitude = x.abs().clamp(max=6.0)
    midpoints = MIDPOINTS.to(x.device)
    code = torch.bucketize(magnitude, midpoints, right=False)
    code = code + torch.isin(magnitude, TIE_UP.to(x.device)).to(code.dtype)
    sign = ((x < 0) & (code > 0)).to(torch.uint8) << 3
    return code.to(torch.uint8) | sign


def pack_fp4(codes):
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def quant_mxfp4(weights):
    """(..., K) FP32 -> ((..., K/32, 16) uint8 blocks, (..., K/32) uint8 E8M0 scales)."""
    groups = weights.reshape(*weights.shape[:-1], -1, 32)
    amax = groups.abs().amax(dim=-1)
    exponent = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp(min=1e-38))) - 2, torch.full_like(amax, -127))
    exponent = exponent.clamp(-127, 127)
    scaled = groups / torch.exp2(exponent).unsqueeze(-1)
    blocks = pack_fp4(e2m1_codes(scaled))
    return blocks, (exponent + 127).to(torch.uint8)


def quant_nvfp4(weights):
    """(N, K) FP32 -> (packed (N, K/2) uint8, FP8 E4M3 (N, K/16) scales, fp16 global)."""
    amax = weights.abs().max()
    global_scale = (amax / (6.0 * 448.0)).clamp(min=2.0 ** -24).to(torch.float16).float()  # fp16 subnormals allowed
    blocks = weights.reshape(weights.shape[0], -1, 16)
    block_amax = blocks.abs().amax(dim=-1)
    scale = (block_amax / 6.0 / global_scale).clamp(max=448.0).to(torch.float8_e4m3fn)
    effective = scale.float() * global_scale
    scaled = torch.where(effective.unsqueeze(-1) > 0, blocks / effective.clamp(min=1e-30).unsqueeze(-1),
                         torch.zeros_like(blocks))
    packed = pack_fp4(e2m1_codes(scaled.reshape(weights.shape)))
    return packed, scale, global_scale.to(torch.float16)


def layer_merge_mxfp4(piece, clusters, weights, device):
    """Whole-layer GPT-OSS piece (E, N, K/32, 16) etc.; returns a new piece."""
    out = {role: tensor.clone() for role, tensor in piece.items()}
    for representative, members in clusters:
        w = weights[members]
        for name in ("gate_up", "down"):
            dense = mxfp4(piece[name][members], piece[name + "_scale"][members], device)  # (m, N, K)
            merged = (dense * w.view(-1, 1, 1).to(device)).sum(0).to(torch.bfloat16).float()
            blocks, scales = quant_mxfp4(merged)
            out[name][representative] = blocks.cpu()
            out[name + "_scale"][representative] = scales.cpu()
            bias = piece[name + "_bias"][members].float()
            out[name + "_bias"][representative] = (bias * w.view(-1, 1)).sum(0).to(torch.bfloat16)
    return out


def layer_merge_nvfp4(piece, clusters, weights, device):
    """Whole-layer NVFP4 piece: per projection (E, N, K/2) codes, (E, N, K/16) FP8, (E, 1) fp16 global."""
    out = {role: tensor.clone() for role, tensor in piece.items()}
    for representative, members in clusters:
        w = weights[members]
        for name in ("gate", "up", "down"):
            dense = torch.stack([nvfp4(piece[name][e], piece[name + "_scale"][e], piece[name + "_global"][e].float(), device)
                                 for e in members])
            merged = (dense * w.view(-1, 1, 1).to(device)).sum(0).to(torch.bfloat16).float()
            packed, scale, global_scale = quant_nvfp4(merged)
            out[name][representative] = packed.cpu()
            out[name + "_scale"][representative] = scale.cpu()
            out[name + "_global"][representative] = global_scale.reshape(out[name + "_global"][representative].shape).cpu()
    return out


def clusters_of(alias, requant_all, experts):
    groups = {}
    for logical in range(experts):
        groups.setdefault(alias.get(logical, logical), []).append(logical)
    return [(rep, sorted(members)) for rep, members in sorted(groups.items()) if requant_all or len(members) > 1]


def merge_layer(layer, piece, control, device):
    experts = control["experts"]
    alias = {int(e): int(t) for e, t in control["alias"].get(str(layer), {}).items()}
    usage = torch.tensor(control["usage"][str(layer)], dtype=torch.float64)
    clusters = clusters_of(alias, control.get("requant_all", False), experts)
    weights = torch.zeros(experts, dtype=torch.float32)
    for rep, members in clusters:
        total = usage[members].sum()
        weights[members] = (usage[members] / total if total > 0 else torch.full((len(members),), 1.0 / len(members))).float()
    with torch.no_grad():
        if control["format"] == "mxfp4":
            return layer_merge_mxfp4(piece, clusters, weights, device)
        return layer_merge_nvfp4(piece, clusters, weights, device)


def wrap_pieces(pieces, experts):
    """Runtime hook: buffer each layer's pieces, merge when complete, yield one whole-layer piece."""
    control = json.loads(Path(os.environ["FREETOKEN_M_REDUCTION_MERGE"]).read_text())
    if control["experts"] != experts:
        raise ValueError("merge control geometry differs from the checkpoint")
    device = "cpu"  # the server has already sized its GPU pools; merge on the host
    pending = {}
    merged_layers = 0
    for layer, e0, e1, piece in pieces:
        rows = pending.setdefault(layer, {})
        rows[e0] = (e1, {role: tensor.clone() for role, tensor in piece.items()})
        if sum(end - start for start, (end, _) in rows.items()) < experts:
            continue
        del pending[layer]
        ordered = [rows[start][1] for start in sorted(rows)]
        whole = {role: torch.cat([part[role] for part in ordered]) for role in ordered[0]}
        yield layer, 0, experts, merge_layer(layer, whole, control, device)
        merged_layers += 1
    if pending:
        raise ValueError(f"merge buffer holds incomplete layers {sorted(pending)}")
    print(f"m-reduction merge: {merged_layers} layers merged ({control['label']})", flush=True)


def identity_check(model, checkpoint, layers, experts):
    """Fraction of checkpoint bytes reproduced by requantize(dequantize(x)) for sampled layers."""
    from experiments.pruning.m_reduction_experts import PREFIX

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {}
    for layer in layers:
        prefix = PREFIX[model].format(layer)
        if model == "gpt_oss_mxfp4":
            for name in ("gate_up_proj", "down_proj"):
                all_blocks, all_scales = checkpoint.get(prefix + name + "_blocks"), checkpoint.get(prefix + name + "_scales")
                stats = Counter()
                for e in range(experts):  # one expert at a time bounds memory
                    blocks, scales = all_blocks[e], all_scales[e]
                    dense = mxfp4(blocks, scales, device)
                    new_blocks, new_scales = quant_mxfp4(dense)
                    redense = mxfp4(new_blocks, new_scales, device)
                    stats["bytes"] += blocks.numel()
                    stats["blocks_identical"] += int((new_blocks.cpu() == blocks).sum())
                    stats["scales"] += scales.numel()
                    stats["scales_identical"] += int((new_scales.cpu() == scales).sum())
                    stats["values"] += dense.numel()
                    stats["values_identical"] += int((redense == dense).sum())
                    stats["err2"] += float((redense - dense).pow(2).sum())
                    stats["ref2"] += float(dense.pow(2).sum())
                report[f"{layer}.{name}"] = {"blocks_identical": stats["blocks_identical"] / stats["bytes"],
                                             "scales_identical": stats["scales_identical"] / stats["scales"],
                                             "values_identical": stats["values_identical"] / stats["values"],
                                             "relative_l2": (stats["err2"] / stats["ref2"]) ** 0.5}
        else:
            stats = Counter()
            for e in range(experts):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    base = f"{prefix}{e}.{name}."
                    w, s, g = (checkpoint.get(base + "weight"), checkpoint.get(base + "weight_scale"),
                               checkpoint.get(base + "weight_scale_2"))
                    dense = nvfp4(w, s, g.float().to(torch.float16).float(), device)
                    packed, scale, global_scale = quant_nvfp4(dense)
                    redense = nvfp4(packed, scale, global_scale.float(), device)
                    stats["values"] += dense.numel()
                    stats["values_identical"] += int((redense == dense).sum())
                    stats["err2"] += float((redense - dense).pow(2).sum())
                    stats["ref2"] += float(dense.pow(2).sum())
            report[str(layer)] = {"values_identical": stats["values_identical"] / stats["values"],
                                  "relative_l2": (stats["err2"] / stats["ref2"]) ** 0.5}
    return report


def build(args):
    from experiments.pruning.m_reduction_p4 import rules
    from experiments.pruning.m_reduction_p7 import training_counts
    from experiments.pruning.m_reduction_traces import calibration, load_model

    geometry, batch, traces = load_model(args.source_study, args.model)
    table, _ = rules(geometry, calibration(args.source_study, args.model))
    artefacts = json.loads(args.artefacts.read_text())
    fmt = "mxfp4" if args.model == "gpt_oss_mxfp4" else "nvfp4"
    args.out.mkdir(parents=True, exist_ok=False)
    for fold, entry in artefacts.items():
        train = [t for t in traces if t.fold != int(fold)]
        access, _ = training_counts(train, geometry, table, "overlap_floor1")
        usage = {str(l): [access[(l, e)] for e in range(geometry.experts)] for l in range(geometry.layers)}
        control = {"label": f"arm2_fold{fold}", "model": args.model, "format": fmt, "experts": geometry.experts,
                   "fold": int(fold), "alias": entry["arm2_alias"], "usage": usage, "requant_all": False,
                   "usage_source": "outer-training decode accesses under overlap_floor1",
                   "artefacts_sha256": __import__("hashlib").sha256(args.artefacts.read_bytes()).hexdigest()}
        (args.out / f"arm2_fold{fold}.json").write_text(json.dumps(control) + "\n")
    identity = {"label": "requant_only_all100", "model": args.model, "format": fmt, "experts": geometry.experts,
                "alias": {}, "usage": {str(l): [0] * geometry.experts for l in range(geometry.layers)},
                "requant_all": True}
    (args.out / "requant_only.json").write_text(json.dumps(identity) + "\n")
    print(json.dumps({"out": str(args.out), "folds": sorted(artefacts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-study", type=Path)
    parser.add_argument("--artefacts", type=Path, help="P7 per-fold artefacts JSON")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--identity-check", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    if args.build:
        build(args)
    elif args.identity_check:
        experts = {"gpt_oss_mxfp4": 32, "qwen_nvfp4": 256, "gemma_nvfp4": 128}[args.model]
        report = identity_check(args.model, Checkpoint(args.checkpoint), args.layers, experts)
        if args.out:
            args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
