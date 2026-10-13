"""Run distinct real-layer q sweeps for common observed miss counts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from experiments.paths import python_command


from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
WORKLOADS = ROOT / "experiments/prompts/routing_workloads.jsonl"
CASES = {
    "qwen_code_m1": ("qwen3.6-35b-nvfp4", 0, 1, 16, 2),
    "qwen_code_m2": ("qwen3.6-35b-nvfp4", 0, 2, 35, 4),
    "qwen_incident_m2": ("qwen3.6-35b-nvfp4", 1, 2, 17, 4),
    "qwen_code_m3": ("qwen3.6-35b-nvfp4", 0, 3, 8, 4),
    "qwen_incident_m3": ("qwen3.6-35b-nvfp4", 1, 3, 8, 8),
    "qwen_code_m4": ("qwen3.6-35b-nvfp4", 0, 4, 2, 4),
    "qwen_incident_m4": ("qwen3.6-35b-nvfp4", 1, 4, 12, 8),
    "qwen_code_m5": ("qwen3.6-35b-nvfp4", 0, 5, 1, 4),
    "qwen_incident_m5": ("qwen3.6-35b-nvfp4", 1, 5, 5, 8),
    "qwen_code_m5_late": ("qwen3.6-35b-nvfp4", 0, 5, 1, 8),
    "qwen_incident_m5_late": ("qwen3.6-35b-nvfp4", 1, 5, 5, 12),
    "qwen_incident_m6": ("qwen3.6-35b-nvfp4", 1, 6, 11, 4),
    "qwen_incident_m7": ("qwen3.6-35b-nvfp4", 1, 7, 2, 4),
    "qwen_incident_m8": ("qwen3.6-35b-nvfp4", 1, 8, 36, 1),
    "gpt_oss_code_m4_layer0": ("gpt-oss-20b", 0, 4, 0, 4),
    "gpt_oss_code_m4_layer12": ("gpt-oss-20b", 0, 4, 12, 16),
    "gpt_oss_incident_m4": ("gpt-oss-20b", 1, 4, 20, 32),
}


def check_gpu_idle() -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(f"GPU compute process is already running: {result.stdout.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, action="append",
                        help="repeat to select cases; default is all")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results/qstar_oracle/representative_states")
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmup < 0:
        parser.error("--repeats must be >=2 and --warmup must be >=0")
    if not WORKLOADS.is_file():
        parser.error(f"missing local workloads: {WORKLOADS}")
    cases = args.case or list(CASES)
    for name in cases:
        model, problem, misses, layer, skip = CASES[name]
        prefix = args.out_dir / name
        raw = prefix.with_suffix(".csv")
        trace = prefix.with_name(prefix.name + "_trace.csv")
        benchmark = prefix.with_name(prefix.name + "_benchmark.jsonl")
        if any(path.exists() for path in (raw, trace, benchmark)):
            raise FileExistsError(f"output already exists for {name}; choose a fresh --out-dir")
        check_gpu_idle()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *python_command(), str(ROOT / "experiments/baselines/bench_decode_moe.py"),
            "--model", str(ROOT / "results/checkpoints" / model),
            "--backend", "hybrid", "--cache-rate", "0.2",
            "--decode", "64", "--no-graph", "--greedy",
            "--aime", str(WORKLOADS), "--problem", str(problem),
            "--layer-replay-out", str(raw),
            "--layer-replay-m", str(misses),
            "--layer-replay-layer", str(layer),
            "--layer-replay-skip-matches", str(skip),
            "--layer-replay-repeats", str(args.repeats),
            "--layer-replay-warmup", str(args.warmup),
            "--trace-csv", str(trace), "--json", str(benchmark),
        ]
        print(f"[representative] {name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
