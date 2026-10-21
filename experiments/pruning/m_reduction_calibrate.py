"""Record fresh native bandwidth profiles for m-reduction admission policies."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from experiments.paths import ROOT
from experiments.pruning.moe_pruning import atomic_json


GEOMETRIES = {
    "gpt_oss_mxfp4": ("gpt-oss-20b", "mxfp4_triton"),
    "qwen_nvfp4": ("qwen3.6-moe", "nvfp4"),
    "gemma_nvfp4": ("gemma4-26b", "nvfp4"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=GEOMETRIES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--align-mxfp4", action="store_true")
    parser.add_argument("--dot64", action="store_true")
    parser.add_argument("--align-nvfp4", action="store_true")
    args = parser.parse_args()
    if args.align_nvfp4 and args.model == "gpt_oss_mxfp4":
        parser.error("NVFP4 alignment is specific to Qwen/Gemma")
    if args.align_nvfp4:
        os.environ["FREETOKEN_M_REDUCTION_NVFP4_ALIGN"] = "1"
    else:
        os.environ.pop("FREETOKEN_M_REDUCTION_NVFP4_ALIGN", None)
    os.environ.pop("FREETOKEN_CPU_MOE_NO_VNNI", None)
    if args.dot64 and not args.align_mxfp4:
        parser.error("FP64 projections require MXFP4 alignment")
    if args.dot64:
        os.environ["FREETOKEN_M_REDUCTION_MXFP4_DOT64"] = "1"
    else:
        os.environ.pop("FREETOKEN_M_REDUCTION_MXFP4_DOT64", None)
    if args.align_mxfp4:
        if args.model != "gpt_oss_mxfp4":
            parser.error("MXFP4 alignment is specific to GPT-OSS")
        os.environ["FREETOKEN_M_REDUCTION_MXFP4_ALIGN"] = "1"
    else:
        os.environ.pop("FREETOKEN_M_REDUCTION_MXFP4_ALIGN", None)
    if args.out.exists():
        raise FileExistsError("Calibration output already exists")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                      "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"GPU is occupied: {active}")
    from freetoken.moe.benchbw import WORKLOADS, main as bench

    name, fmt = GEOMETRIES[args.model]
    plan = json.loads((ROOT / "configs/m_reduction/stage0.json").read_text())
    checkpoint = ROOT / plan["models"][args.model]["path"]
    config = json.loads((checkpoint / "config.json").read_text())
    text = config.get("text_config", config)
    geometry = (text["hidden_size"], text.get("moe_intermediate_size") or text["intermediate_size"],
                plan["models"][args.model]["experts"], plan["models"][args.model]["top_k"])
    wl = WORKLOADS[name]
    if geometry != (wl.hidden, wl.inter, wl.experts, wl.top_k):
        raise ValueError("Native calibration preset does not match checkpoint geometry")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["--model", name, "--formats", fmt, "--gpu", "0", "--cpu-threads", "5",
            "--cpu-iters", "8", "--pcie-mib", "256", "--pcie-iters", "30",
            "--kernel-cpu-iters", "64", "--kernel-pcie-iters", "20", "--out", str(args.out)]
    atomic_json(args.out.with_suffix(".control.json"), {"model": args.model,
                "mxfp4_alignment": args.align_mxfp4,
                "mxfp4_dot64": args.dot64,
                "nvfp4_alignment": args.align_nvfp4,
                "native_bench_arguments": argv, "geometry": geometry,
                "interpretation": "synthetic native kernel logical-byte bandwidth; not a DRAM hardware counter"})
    if bench(argv):
        raise RuntimeError("Native bandwidth calibration failed")
    profile = json.loads(args.out.read_text())
    entry = profile["workloads"][name]["kernels"][fmt]
    fields = ("cpu_moe_gbs", "pcie_gather_gbs", "cpu_moe_overlap_gbs", "pcie_gather_overlap_gbs")
    if any(not isinstance(entry.get(field), (float, int)) or entry[field] <= 0 for field in fields):
        raise ValueError("Standalone/concurrent branch measurements are incomplete")
    print(f"Calibrated {args.model}: " + json.dumps({field: entry[field] for field in fields}), flush=True)


if __name__ == "__main__":
    main()
