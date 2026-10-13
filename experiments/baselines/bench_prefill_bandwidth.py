"""Measure first-token latency versus prompt length with fresh, uncached requests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from experiments.baselines.bench_decode_moe import (
    free_port, get_json, load_problem, parse_args as decode_args, pump_output,
    resolve_sampling, serve_cmd, stop_server, stream_generate, wait_ready,
)


def token_count(tokenizer, prompt: str) -> int:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=True,
    )
    return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def make_prompt(tokenizer, target: int, tag: str, passage: str) -> str:
    unit = passage + "\n"
    low, high = 0, 1
    while token_count(tokenizer, f"[{tag}]\n" + unit * high) < target:
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if token_count(tokenizer, f"[{tag}]\n" + unit * mid) < target:
            low = mid
        else:
            high = mid
    choices = [f"[{tag}]\n" + unit * count for count in (low, high)]
    return min(choices, key=lambda prompt: abs(token_count(tokenizer, prompt) - target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="hybrid", choices=("auto", "offload", "hybrid"))
    parser.add_argument("--cache-rate", type=float, default=0.2)
    parser.add_argument("--mem-ratio", type=float, default=0.9)
    parser.add_argument("--target-tokens", type=int, nargs="+", default=[1024, 2048, 4096, 7900])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--decode", type=int, default=8,
                        help="output token limit; TTFT uses the first visible generated text")
    parser.add_argument("--aime")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile")
    args = parser.parse_args()
    if (min(args.target_tokens) < 1 or args.repeat < 1 or args.decode < 1 or
            not 0 < args.mem_ratio <= 1):
        parser.error("prompt lengths, repeat count, and decode must be positive; "
                     "mem-ratio must be in (0, 1]")
    target = Path(args.out)
    if target.exists():
        parser.error(f"{target} already exists")
    target.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    passage, _ = load_problem(args.aime, 0)
    sampling, _ = resolve_sampling(args.model, False)
    env = os.environ.copy()
    if args.profile:
        env["FREETOKEN_BENCHBW_PATH"] = str(Path(args.profile).resolve())
    for overlap in (True, False):
        settings = decode_args([
            "--model", args.model, "--backend", args.backend,
            "--cache-rate", str(args.cache_rate), "--mem-ratio", str(args.mem_ratio),
            "--decode", str(args.decode), "--no-graph",
            *([] if overlap else ["--no-overlap"]),
        ])
        port = free_port()
        origin = f"http://127.0.0.1:{port}"
        cmd = serve_cmd(settings, args.backend, port)
        cmd += ["--cache-type", "naive"]
        fd, log_path = tempfile.mkstemp(prefix="bench-prefill-", suffix=".log", dir=os.environ.get("EXPERIMENT_LOG_DIR"))
        with os.fdopen(fd, "wb") as log_f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=env)
            pump = threading.Thread(target=pump_output, args=(proc.stdout, log_f), daemon=True)
            pump.start()
            try:
                wait_ready(origin, proc, log_path, settings.server_timeout)
                model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
                warm = make_prompt(tokenizer, 128, "prefill warmup", passage)
                stream_generate(origin, model_id, warm, sampling, settings)
                for wanted in args.target_tokens:
                    for repeat in range(args.repeat):
                        tag = f"prefill {uuid.uuid4().hex} {wanted} {repeat}"
                        prompt = make_prompt(tokenizer, wanted, tag, passage)
                        result = stream_generate(origin, model_id, prompt, sampling, settings)
                        if not result["stamps"]:
                            raise RuntimeError(f"no first token for target {wanted}")
                        row = {
                            "model": args.model, "backend": args.backend,
                            "cache_rate_requested": args.cache_rate,
                            "memory_ratio_requested": args.mem_ratio,
                            "prefill_overlap": overlap,
                            "kv_cache_type": "naive",
                            "target_prompt_tokens": wanted,
                            "estimated_prompt_tokens": token_count(tokenizer, prompt),
                            "prompt_tokens": result["usage"]["prompt_tokens"],
                            "completion_tokens": result["usage"]["completion_tokens"],
                            "decode_limit": args.decode,
                            "ttft_ms": (result["stamps"][0] - result["t0"]) * 1000,
                            "repeat": repeat, "request_tag": tag,
                            "server_log": log_path,
                        }
                        with target.open("a") as output:
                            output.write(json.dumps(row) + "\n")
                        print(f"overlap={overlap} target={wanted} repeat={repeat} "
                              f"actual={row['prompt_tokens']} ttft_ms={row['ttft_ms']:.1f}",
                              flush=True)
            finally:
                stop_server(proc)
                pump.join(timeout=10)
    print(target)


if __name__ == "__main__":
    main()
