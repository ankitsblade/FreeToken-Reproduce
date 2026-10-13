"""Experiment 1: prefill with MoE expert-prefetch overlap ON vs OFF.

Same model / host / env / workload / cache config as saketh_exps; the only variable is
--moe-prefill-overlap (ON) vs --disable-moe-prefill-overlap (OFF). For each setting we
spawn one fresh server (naive KV cache, so no prefix reuse), warm it, then sweep prompt
lengths with unique tags (fresh, uncached prefills) and time the first token. A system
monitor samples GPU/CPU utilization for that setting's server only, and every request
carries wall-clock stamps so utilization can be attributed to a prompt length.

TTFT here is essentially the prefill latency (first visible token after prefill). The
overlap benefit at length L is TTFT_off(L) - TTFT_on(L): the expert-transfer time hidden
behind prefill compute. How that benefit scales with L is the pattern of interest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from experiments.paths import python_command

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
sys.path.insert(0, str(ROOT / "experiments/baselines"))
from experiments.baselines.bench_decode_moe import (  # noqa: E402
    free_port, get_json, load_problem, parse_args as decode_args, pump_output,
    resolve_sampling, serve_cmd, stop_server, stream_generate, wait_ready,
)
from bench_prefill_bandwidth import make_prompt, token_count  # noqa: E402

OUT = study_dir("prefill_overlap") / "results"
LOGS = study_dir("prefill_overlap") / "logs"
MONITOR = ROOT / "experiments/common/monitor_system.py"


def run_setting(model: str, overlap: bool, lengths: list[int], repeats: int,
                cache_rate: float, gpu: str, results_path: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=True)
    passage, _ = load_problem(str(ROOT / "experiments" / "prompts" / "routing_workloads.jsonl"), 0)
    sampling, _ = resolve_sampling(model, False)
    settings = decode_args([
        "--model", model, "--backend", "hybrid", "--cache-rate", str(cache_rate),
        "--mem-ratio", "0.9", "--decode", "8", "--no-graph", "--gpu", gpu,
        *([] if overlap else ["--no-overlap"]),
    ])
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    cmd = serve_cmd(settings, "hybrid", port) + ["--cache-type", "naive"]
    tag = "on" if overlap else "off"
    log_path = LOGS / f"serve_overlap_{tag}.log"
    monitor_csv = OUT / f"monitor_overlap_{tag}.csv"
    monitor_csv.unlink(missing_ok=True)
    with open(log_path, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ.copy())
        pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
        pump.start()
        mon = None
        try:
            wait_ready(origin, proc, str(log_path), settings.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            mon = subprocess.Popen(
                [*python_command(), str(MONITOR), "--watch-pid", str(proc.pid),
                 "--interval", "0.25", "--out", str(monitor_csv)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # warm the expert cache to steady state
            stream_generate(origin, model_id, make_prompt(tokenizer, 256, "warm", passage),
                            sampling, settings)
            for wanted in lengths:
                for r in range(repeats):
                    t = f"prefill {uuid.uuid4().hex} {wanted} {r}"
                    prompt = make_prompt(tokenizer, wanted, t, passage)
                    wall_start = time.time()
                    res = stream_generate(origin, model_id, prompt, sampling, settings)
                    wall_first = wall_start + (res["stamps"][0] - res["t0"])
                    stats = get_json(f"{origin}/v1/stats")
                    row = {
                        "overlap": overlap, "target_tokens": wanted,
                        "prompt_tokens": res["usage"]["prompt_tokens"],
                        "repeat": r, "ttft_ms": (res["stamps"][0] - res["t0"]) * 1000,
                        "wall_start": wall_start, "wall_first_token": wall_first,
                        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
                    }
                    with results_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"  overlap={overlap} L={wanted} r={r} "
                          f"prompt={row['prompt_tokens']} ttft={row['ttft_ms']:.1f}ms", flush=True)
        finally:
            if mon is not None:
                mon.terminate()
                mon.wait()
            stop_server(proc)
            pump.join(timeout=10)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--cache-rate", type=float, default=0.2)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 7900])
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", default=str(OUT / "prefill_ttft.jsonl"))
    args = ap.parse_args()
    if not args.model:
        ap.error("set $MODEL (source scripts/env.sh) or pass --model")
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    results = Path(args.out)
    results.unlink(missing_ok=True)
    manifest = {"model": args.model, "cache_rate": args.cache_rate, "backend": "hybrid",
                "kv_cache": "naive", "lengths": args.lengths, "repeats": args.repeats,
                "variable": "moe_prefill_overlap", "workload": "routing_workloads.jsonl#0"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for overlap in (True, False):
        print(f"=== prefill overlap {'ON' if overlap else 'OFF'} ===", flush=True)
        run_setting(args.model, overlap, args.lengths, args.repeats, args.cache_rate,
                    args.gpu, results)
    print("PREFILL SWEEP DONE", flush=True)


if __name__ == "__main__":
    main()
