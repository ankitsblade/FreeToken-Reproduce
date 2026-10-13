"""Run runtime-q episodes against real `ft serve` processes.

One server per (cache_rate) group; inside it each episode = standardized warmup generate +
measured generate, both from a freshly reset expert cache (the hook resets on a new episode
id). Repeats are the outer loop so every policy is measured once per round (paired design).
Resumable: an episode whose key is already in the output JSONL is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from experiments.paths import python_command

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath
RQ = study_dir("runtime_q")
sys.path.insert(0, str(ROOT / "experiments/baselines"))
from experiments.baselines.bench_decode_moe import (  # noqa: E402
    free_port, get_json, load_problem, parse_args as decode_args, resolve_sampling,
    serve_cmd, stop_server, stream_generate, wait_ready,
)

HOOK = ROOT / "experiments/runtime_q/instrumentation/runtime_q_hook.py"
PROMPTS = ROOT / "experiments" / "prompts" / "routing_workloads.jsonl"
WORKLOADS = {"low_miss": 0.40, "med_miss": 0.20, "high_miss": 0.10}


def write_control(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def resolve_policy(spec: dict) -> dict:
    resolved = dict(spec)
    if "path" in resolved:
        path = Path(resolved["path"]).expanduser()
        if not path.is_absolute():
            path = Path(os.environ.get("RUNTIME_Q_MODEL_ROOT", RQ)) / path
        if not path.is_file():
            raise FileNotFoundError(f"Missing trained policy: {path}")
        resolved["path"] = str(path.resolve())
    if "base" in resolved:
        resolved["base"] = resolve_policy(resolved["base"])
    return resolved


def key_of(row: dict) -> tuple:
    return (row["workload"], row["cache_rate"], row["problem"], row["policy"], row["seed"], row["rep"])


def drain(src, log_f) -> None:
    for chunk in iter(lambda: src.read1(65536), b""):
        log_f.write(chunk)


def run_group(model: str, group: dict, out: Path, decode: int, gpu: str, log_dir: Path) -> None:
    rate, problem, workload = group["cache_rate"], group.get("problem", 0), group["workload"]
    done = set()
    if out.exists():
        done = {key_of(json.loads(l)) for l in out.read_text().splitlines() if l.strip()}
    todo = []
    reps = group.get("repeats", 1)
    for rep in range(reps):
        # rotate the order each round so no policy always sits in the same position
        shift = (rep * len(group["episodes"])) // max(1, reps)
        for ep in group["episodes"][shift:] + group["episodes"][:shift]:
            seed = ep.get("seed", 0) + rep
            k = (workload, rate, problem, ep["name"], seed, rep)
            if k not in done:
                todo.append((rep, seed, ep))
    if not todo:
        print(f"[skip] {workload} problem={problem}: all {len(done)} episodes done", flush=True)
        return
    args = decode_args(["--model", model, "--backend", "hybrid", "--hybrid-fetch", "-1",
                        "--decode", str(decode), "--cache-rate", str(rate), "--no-graph",
                        "--greedy", "--gpu", gpu])
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    cmd = serve_cmd(args, "hybrid", port) + ["--cache-type", "naive"]
    control = log_dir / f"control_{uuid.uuid4().hex[:8]}.json"
    env = os.environ.copy()
    env["FREETOKEN_MOE_RUNTIME_Q_MODULE"] = str(HOOK)
    env["FREETOKEN_MOE_RUNTIME_Q_CONTROL"] = str(control)
    text, _ = load_problem(str(PROMPTS), problem)
    sampling, _ = resolve_sampling(model, True)
    server_id = uuid.uuid4().hex[:8]
    slog = log_dir / f"serve_{workload}_p{problem}_{server_id}.log"
    # optional controlled CPU contention: N busy-loop processes for the whole group
    hogs = [subprocess.Popen([*python_command(), "-c", "while True: pass"])
            for _ in range(group.get("cpu_hogs", 0))]
    with open(slog, "wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        threading.Thread(target=drain, args=(proc.stdout, log_f), daemon=True).start()
        try:
            wait_ready(origin, proc, str(slog), 900)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            # burn-in: the first episodes after a server start run measurably slower on the
            # host side (load/paging transient); run and discard them.
            burn = {"name": "_burnin", "spec": {"kind": "fraction", "fraction": 0.25}}
            todo = [(-1, 0, burn)] * group.get("burnin", 3) + todo
            for order, (rep, seed, ep) in enumerate(todo):
                episode = uuid.uuid4().hex[:12]
                base = {"episode": episode, "policy": resolve_policy(ep["spec"]), "policy_name": ep["name"],
                        "workload": workload, "cache_rate": rate, "problem": problem,
                        "seed": seed, "groups": ep.get("groups", "full"),
                        "log": ep.get("log") and rep == 0 and str(RQ / ep["log"]) or None}
                write_control(control, {**base, "phase": "warmup"})
                stream_generate(origin, model_id, "[FreeToken benchmark warmup]\n" + text,
                                sampling, args)
                write_control(control, {**base, "phase": "measure"})
                r = stream_generate(origin, model_id, "[FreeToken benchmark measurement]\n" + text,
                                    sampling, args)
                stamps = r["stamps"]
                steps = r["usage"]["completion_tokens"] - 1
                span = stamps[-1] - stamps[0]
                gaps = sorted((b - a) * 1e3 for a, b in zip(stamps, stamps[1:]))
                row = {
                    "workload": workload, "cache_rate": rate, "problem": problem,
                    "policy": ep["name"], "seed": seed, "rep": rep, "episode": episode,
                    "server": server_id, "order": order, "decode_steps": steps,
                    "decode_tok_s": steps / span, "ms_per_token": span / steps * 1e3,
                    "p50_ms": gaps[len(gaps) // 2],
                    "p95_ms": gaps[min(len(gaps) - 1, int(len(gaps) * .95))],
                    "p99_ms": gaps[min(len(gaps) - 1, int(len(gaps) * .99))],
                    "ttft_ms": (stamps[0] - r["t0"]) * 1e3,
                    "output_sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12],
                    "groups": ep.get("groups", "full"), "spec": ep["spec"],
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                if rep < 0:
                    continue
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[{workload} p{problem} rep{rep}] {ep['name']:28s} "
                      f"{row['decode_tok_s']:6.2f} tok/s  {row['ms_per_token']:6.2f} ms/tok  "
                      f"sha {row['output_sha1']}", flush=True)
        finally:
            stop_server(proc)
            for h in hogs:
                h.kill()
            control.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True, help="JSON: {groups:[{workload,cache_rate,problem,repeats,episodes:[{name,spec,seed,log,groups}]}]}")
    ap.add_argument("--out")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default=os.environ.get("MODEL"))
    a = ap.parse_args()
    if not a.model:
        ap.error("pass --model or set MODEL")
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out) if a.out else RQ / "results" / (Path(a.plan).stem + "_episodes.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = RQ / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for group in plan["groups"]:
        run_group(a.model, group, out, a.decode, a.gpu, log_dir)
    print("PLAN DONE", flush=True)


if __name__ == "__main__":
    main()
