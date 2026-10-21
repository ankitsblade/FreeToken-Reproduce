"""Write the frozen prompt file and the run plans of the q* / Reuse-Aware / Runtime-Variable study.

Every setting below is copied from a cited source (docs/AUDIT.md, provenance table); nothing is
tuned here. Run once; the outputs are versioned next to this file and must not be edited by hand.

  prompts   AIME-25 problems 0-2 from the official decode benchmark's dataset (upstream
            benchmarks/bench_decode_moe.py: math-ai/aime25 test.jsonl, problem 0 by default);
            paper workload W1 is AIME (section 5.1).
  model     Qwen3.6-35B-A3B NVFP4 (paper section 5.1 model; NVFP4 is the paper's 8 GB-laptop
            build), pinned revision of configs/pruning/models.json.
  cache     --moe-cache-auto (paper section 3.3; upstream default when no size flag is given). The
            driver always passes --cache-rate, and the runtime maps rate 0 to auto
            (server/args.py: offload backends with no cache flag default to --moe-cache-auto).
  decode    256 tokens, the upstream benchmark default; one warm-up generate per episode.
  serving   the established qwen_nvfp4 profile: memory ratio 0.9, 2048-token limit, text only,
            1024-token prefill cap, 5 CPU workers, serial expert load, naive KV, greedy, eager.
  policies  A paper q* (min one fill), B original Reuse-Aware, the seven V1/V2/V3 combinations;
            A0 (no minimum) and E (built-in q*) as labelled diagnostics; unhooked A0 and E on
            problem 0 to measure the harness overhead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AIME = Path.home() / (".cache/huggingface/hub/datasets--math-ai--aime25/snapshots/"
                      "563bb8404243c5f09de6ec262f2db674fe5bce9b/test.jsonl")
AIME_SHA256 = "b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331"
PROMPTS = HERE / "prompts_aime25.jsonl"
HOOK = "q_expswithreuse_aware/policy/qexp_hook.py"
PROBLEMS = (0, 1, 2)
# Established Qwen environment: configs/runtime_q/reuse_aware_models/qwen.json (benchbw profile of
# that study; B_P / B_H overlap pair 6.71 / 31.38). Paper q* = B_P / B_H gives the same integer q for
# every m <= 8 with the standalone pair 6.71 / 32.9 (docs/AUDIT.md section 1.3).
ESTABLISHED = ROOT / "configs/runtime_q/reuse_aware_models/qwen.json"
QSTAR_FRACTION = 0.21383

V = {"V1": {"cost": "runtime"}, "V2": {"reuse": "calibrated"}, "V3": {"evict": True}}
C_ARMS = {
    "C_V1": V["V1"], "C_V2": V["V2"], "C_V3": V["V3"],
    "C_V1V2": {**V["V1"], **V["V2"]}, "C_V1V3": {**V["V1"], **V["V3"]},
    "C_V2V3": {**V["V2"], **V["V3"]}, "C_V1V2V3": {**V["V1"], **V["V2"], **V["V3"]},
}


def write_prompts() -> str:
    raw = AIME.read_bytes()
    if hashlib.sha256(raw).hexdigest() != AIME_SHA256:
        raise SystemExit(f"{AIME} does not match the pinned sha256")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    out = []
    for i in PROBLEMS:
        r = rows[i]
        out.append(json.dumps({"problem": r["problem"], "answer": str(r["answer"]), "boxed_instruction": True,
                               "task": f"aime25_{i:02d}",
                               "source": f"math-ai/aime25@563bb8404243c5f09de6ec262f2db674fe5bce9b:test.jsonl#{i}"}))
    PROMPTS.write_text("\n".join(out) + "\n")
    return hashlib.sha256(PROMPTS.read_bytes()).hexdigest()


def arm(name: str, spec: dict, log: str) -> dict:
    return {"name": name, "spec": spec, "groups": ["BASE"], "control": {"features": "lean"},
            "log": log, "log_rep": 3}


def arms(log: str) -> list[dict]:
    out = [arm("A_qstar_paper_min1", {"kind": "fraction_min1", "fraction": QSTAR_FRACTION}, log),
           arm("B_reuse_aware", {"kind": "reuse_aware", "kappa": 1.0}, log)]
    out += [arm(n, {"kind": "reuse_aware_v", **flags}, log) for n, flags in C_ARMS.items()]
    out += [arm("A0_qstar_paper", {"kind": "fraction", "fraction": QSTAR_FRACTION}, log),
            arm("E_qstar_engine", {"kind": "native"}, log)]
    return out


def plan(problems, repeats: int, burnin: int, episodes: list[dict], nohook: bool) -> dict:
    est = json.loads(ESTABLISHED.read_text())["groups"][0]
    env = dict(est["env"])
    base = {"workload": "auto", "cache_rate": 0, "prompts": str(PROMPTS.relative_to(ROOT)),
            "serve_args": est["serve_args"], "save_text": True}
    groups = []
    for prob in problems:
        groups.append({**base, "problem": prob, "repeats": repeats, "burnin": burnin, "hook": HOOK,
                       "env": env, "episodes": episodes})
        if nohook and prob == 0:
            plain = {k: v for k, v in env.items() if not k.startswith("FREETOKEN_MOE_RUNTIME_Q_")}
            groups.append({**base, "problem": prob, "repeats": 3, "burnin": 2, "hook": "none",
                           "reset": "rebuild", "env": {**plain, "FREETOKEN_MOE_FORCE_FETCH_FRACTION": QSTAR_FRACTION},
                           "episodes": [{"name": "A0_qstar_paper_nohook", "spec": {"kind": "native"}}]})
            groups.append({**base, "problem": prob, "repeats": 3, "burnin": 2, "hook": "none",
                           "reset": "rebuild", "env": plain,
                           "episodes": [{"name": "E_qstar_engine_nohook", "spec": {"kind": "native"}}]})
    return {"model": "qwen", "note": __doc__.strip(), "groups": groups}


def main() -> None:
    sha = write_prompts()
    plans = {
        "smoke_qwen_auto": plan((0,), 1, 1,
                                [dict(a, log_rep=0) for a in arms("results/smoke_lean_qwen.csv")
                                 if a["name"] in ("A_qstar_paper_min1", "B_reuse_aware", "C_V1V2V3")], False),
        "ablation_qwen_auto": plan(PROBLEMS, 4, 3, arms("results/ablation_lean_qwen.csv"), True),
    }
    (HERE / "plans").mkdir(exist_ok=True)
    for name, p in plans.items():
        path = HERE / "plans" / f"{name}.json"
        path.write_text(json.dumps(p, indent=1) + "\n")
        n = sum(g["repeats"] * len(g["episodes"]) + g["burnin"] for g in p["groups"])
        print(f"{path.relative_to(ROOT)}: {len(p['groups'])} servers, {n} episodes incl. burn-in, "
              f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"{PROMPTS.relative_to(ROOT)} sha256 {sha}")


if __name__ == "__main__":
    main()
