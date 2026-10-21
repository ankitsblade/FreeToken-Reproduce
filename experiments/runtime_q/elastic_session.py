"""Experiment 2, condition 2: serving sessions under KV pressure with runtime cache reallocation.

A session is an ordered list of requests on one server: short requests (the reuse-aware task
prompts, `decode` tokens) and long-context requests whose prompt + output need more KV tokens
than the large expert-cache geometry leaves. One runtime-q hook episode spans a session (the
expert cache warms across requests, as in real serving); every session starts cold (a rebuild
to its start geometry empties the slot cache).

Allocation modes (expert slots <-> KV pages through POST /v1/cache/rebuild, idle-only):
  fixed_large    big expert cache all session; a long request that does not fit is submitted
                 anyway and recorded as failed (runtime: "prompt is too long")
  fixed_fit      the largest expert cache that admits the session's longest request, all session
  elastic_naive  start big; before a long request that does not fit, shrink to fit; after it,
                 grow straight back
  elastic_rv     start big; shrink to fit as above; grow back only when the measured benefit
                 (decode ms/token at big vs small x expected short tokens until the next long
                 request) exceeds the measured cost (grow + re-shrink rebuild time + post-grow
                 refill penalty). Before a grow-back has been observed in the session it grows
                 back once (exploration) to measure that cost. All inputs are past observations.
This is an orchestrator outside the runtime that calls the runtime's explicit rebuild API between
requests; the runtime itself never resizes automatically.

Geometries: the startup --moe-cache-rate geometry is first rebuilt with 2% fewer KV pages so it
passes the runtime's rebuild fit-check (stricter than startup sizing on some models); every other
geometry keeps the same total bytes (moe * S_e + pages * S_p), so all targets fit by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from experiments.baselines.bench_decode_moe import load_problem
from experiments.paths import ROOT, study_dir
from experiments.runtime_q.driver import cache_geometry, gpu_snapshot, post_json, write_control
from experiments.runtime_q.elastic_validate import Server

PROMPTS = ROOT / "experiments/prompts/reuse_aware_tasks.jsonl"
PAPER = ROOT / "docs/references/2608.16157v1.pdf"
LONG_INSTRUCTION = ("Read the following excerpt of a systems research paper carefully. Then list the "
                    "three most important technical mechanisms it describes, one sentence each.\n\n")
MARGIN_TOKENS = 256


# --------------------------------------------------------------------------- prompts
def long_prompt(model_dir: str, tokens: int, cache: dict) -> tuple[str, int]:
    """Deterministic long-context prompt: the paper's text (repeated if needed) cut to `tokens`
    tokens of the model's own tokenizer, after a fixed instruction."""
    key = (model_dir, tokens)
    if key in cache:
        return cache[key]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    text = subprocess.run(["pdftotext", str(PAPER), "-"], capture_output=True, text=True, check=True).stdout
    ids: list[int] = []
    while len(ids) < tokens:
        ids += tok.encode(text, add_special_tokens=False).ids
    body = tok.decode(ids[:tokens])
    cache[key] = (LONG_INSTRUCTION + body, tokens)
    return cache[key]


# --------------------------------------------------------------------------- geometry
class Geometry:
    def __init__(self, g: dict) -> None:
        self.pe = g["unit_bytes"]["moe_per_expert"]
        self.pp = g["unit_bytes"]["kv_per_token"] * g["page_size"]
        self.page = g["page_size"]
        self.E = g["num_experts"]
        self.total = None
        self.big_moe = self.big_pages = None

    def set_big(self, moe: int, pages: int) -> None:
        self.big_moe, self.big_pages = moe, pages
        self.total = moe * self.pe + pages * self.pp

    def pages_for(self, moe: int) -> int:
        return int((self.total - moe * self.pe) // self.pp)

    def moe_fitting(self, kv_tokens: int) -> int:
        need_pages = math.ceil(kv_tokens / self.page)
        if need_pages <= self.big_pages:
            return self.big_moe
        moe = int((self.total - need_pages * self.pp) // self.pe)
        return moe


def resize(srv: Server, moe: int, pages: int) -> list[dict]:
    """Freeing side first (the engine resizes experts before KV; see Phase A test A3a)."""
    g = cache_geometry(srv.origin)
    if g["moe_cache_size"] == moe and g["num_pages"] == pages:
        return [srv.rebuild({"moe_cache_size": moe})]  # same size: cold reset only
    if moe > g["moe_cache_size"] and pages < g["num_pages"]:
        return [srv.rebuild({"num_pages": pages}), srv.rebuild({"moe_cache_size": moe})]
    return [srv.rebuild({"moe_cache_size": moe, "num_pages": pages})]


def ok(steps: list[dict]) -> bool:
    return all(s["result"].get("status") == "ok" for s in steps)


# --------------------------------------------------------------------------- requests
def generate(srv: Server, text: str, decode: int) -> dict:
    """Streamed request; HTTP / runtime errors are returned, not raised."""
    from experiments.runtime_q.elastic_validate import resolve_sampling
    import urllib.error
    import urllib.request
    sampling, _ = resolve_sampling(srv.args.model, True)
    body = {"model": srv.model_id, "messages": [{"role": "user", "content": text}], "max_tokens": decode,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}, **sampling}
    req = urllib.request.Request(f"{srv.origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, pieces, usage, err = [], [], None, None
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("error"):
                    err = chunk["error"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices", []):
                    d = ch.get("delta") or {}
                    t = d.get("reasoning_content") or d.get("content")
                    if t:
                        stamps.append(time.perf_counter())
                        pieces.append(t)
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    t1 = time.perf_counter()
    text_out = "".join(pieces)
    steps = (usage or {}).get("completion_tokens", 0) - 1
    served = err is None and usage is not None and steps >= decode - 1
    return {"served": served, "error": err, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "steps": steps, "ttft_ms": (stamps[0] - t0) * 1e3 if stamps else None,
            "e2e_ms": (t1 - t0) * 1e3,
            "ms_per_token": (stamps[-1] - stamps[0]) / max(1, len(stamps) - 1) * 1e3 if len(stamps) > 1 else None,
            "sha1": hashlib.sha1(text_out.encode()).hexdigest()[:12]}


class RVAllocator:
    """Grow-back decision of elastic_rv from this session's past measurements only."""

    def __init__(self) -> None:
        self.d = {}            # moe -> EWMA decode ms/token of short requests at that size
        self.run_tokens = []   # short tokens observed between consecutive long requests
        self.cur_run = 0
        self.rebuild_ms = []   # observed grow / shrink rebuild wall times
        self.refill_ms = []    # observed post-grow penalty on the first short request
        self.explored = False

    def observe_short(self, moe: int, r: dict) -> None:
        if r["served"] and r["ms_per_token"]:
            self.d[moe] = r["ms_per_token"] if moe not in self.d else 0.7 * self.d[moe] + 0.3 * r["ms_per_token"]
            self.cur_run += r["steps"] + 1

    def observe_long(self) -> None:
        self.run_tokens.append(self.cur_run)
        self.cur_run = 0

    def decide_grow(self, big: int, small: int) -> tuple[bool, dict]:
        info = {"d_big": self.d.get(big), "d_small": self.d.get(small),
                "runs": list(self.run_tokens), "rebuild_ms": list(self.rebuild_ms),
                "refill_ms": list(self.refill_ms)}
        if not self.explored:
            info["reason"] = "explore: no grow-back measured yet in this session"
            return True, info
        if info["d_small"] is None or info["d_big"] is None:
            info["reason"] = "no short request measured at the small size yet: stay"
            return False, info
        exp_tokens = sum(self.run_tokens) / len(self.run_tokens) if self.run_tokens else 0.0
        benefit = (info["d_small"] - info["d_big"]) * exp_tokens
        cost = 2 * (sum(self.rebuild_ms) / len(self.rebuild_ms) if self.rebuild_ms else 0.0) + \
            (sum(self.refill_ms) / len(self.refill_ms) if self.refill_ms else 0.0)
        info.update(expected_short_tokens=exp_tokens, benefit_ms=benefit, cost_ms=cost)
        grow = benefit > 1.1 * cost
        info["reason"] = "benefit > 1.1 x cost" if grow else "benefit <= 1.1 x cost: stay small"
        return grow, info


def run_session(a, srv: Server, geo: Geometry, cfg: dict, plan: dict, rep: int, out: Path,
                prompts: dict, tokcache: dict) -> dict:
    alloc = cfg["alloc"]
    longs = [r["tokens"] + r["decode"] + MARGIN_TOKENS for r in plan["requests"] if r["kind"] == "long"]
    fit_moe = geo.moe_fitting(max(longs)) if longs else geo.big_moe
    start_moe = fit_moe if alloc == "fixed_fit" else geo.big_moe
    t_s = time.perf_counter()
    steps = resize(srv, start_moe, geo.pages_for(start_moe))
    if not ok(steps):
        raise RuntimeError(f"session start resize failed: {steps}")
    episode = uuid.uuid4().hex[:12]
    logged = plan.get("log_rep") is not None and rep == plan["log_rep"]
    base = {"episode": episode, "policy": cfg["policy"], "policy_name": cfg["name"],
            "workload": f"session_{alloc}", "cache_rate": plan["rate"], "problem": -1, "seed": rep,
            "groups": ["BASE"], "features": "lean",
            "log": str(out / f"decisions_{plan['model']}.csv") if logged else None}
    rv = RVAllocator()
    events, rows = [{"t": 0.0, "event": "start", "steps": steps, "moe": start_moe}], []
    session_t0 = time.perf_counter()
    for i, rq in enumerate(plan["requests"]):
        g = cache_geometry(srv.origin)
        if rq["kind"] == "short":
            text, _ = load_problem(str(PROMPTS), rq["problem"])
            text = "[FreeToken elastic session]\n" + text
            need = None
        else:
            text, _ = long_prompt(a.model, rq["tokens"], tokcache)
            need = rq["tokens"] + rq["decode"] + MARGIN_TOKENS
        pre, post = [], []
        if need and alloc.startswith("elastic") and need > g["num_pages"] * g["page_size"]:
            moe = geo.moe_fitting(need)
            pre = resize(srv, moe, geo.pages_for(moe))
            rv.rebuild_ms += [s["wall_ms"] for s in pre]
            events.append({"t": time.perf_counter() - session_t0, "event": "shrink", "before": g["moe_cache_size"],
                           "after": moe, "steps": pre, "trigger": f"long request needs {need} KV tokens"})
        g_run = cache_geometry(srv.origin)
        write_control(srv.control, {**base, "phase": f"r{i:02d}_{rq['kind']}"})
        r = generate(srv, text, rq["decode"])
        r.update(i=i, kind=rq["kind"], need_kv_tokens=need, moe=g_run["moe_cache_size"],
                 kv_pages=g_run["num_pages"], pre_rebuild_ms=sum(s["wall_ms"] for s in pre),
                 t_end=time.perf_counter() - session_t0, gpu=gpu_snapshot("0"))
        if rq["kind"] == "short":
            if rows and rows[-1].get("after_grow") and r["served"] and r["ms_per_token"]:
                ref = rv.d.get(g_run["moe_cache_size"])
                if ref:
                    rv.refill_ms.append(max(0.0, (r["ms_per_token"] - ref) * (r["steps"])))
            rv.observe_short(g_run["moe_cache_size"], r)
        else:
            rv.observe_long()
        # grow-back decision after the request, while idle
        if alloc.startswith("elastic") and g_run["moe_cache_size"] < geo.big_moe:
            if alloc == "elastic_naive":
                grow, info = True, {"reason": "naive: grow back as soon as the request is done"}
            else:
                grow, info = rv.decide_grow(geo.big_moe, g_run["moe_cache_size"])
            if grow:
                post = resize(srv, geo.big_moe, geo.big_pages)
                rv.rebuild_ms += [s["wall_ms"] for s in post]
                rv.explored = True
                r["after_grow_next"] = True
            events.append({"t": time.perf_counter() - session_t0, "event": "grow" if grow else "stay",
                           "small": g_run["moe_cache_size"], "big": geo.big_moe, "steps": post,
                           "decision": info})
        r["post_rebuild_ms"] = sum(s["wall_ms"] for s in post)
        r["after_grow"] = bool(post) and ok(post)
        rows.append(r)
    makespan = (time.perf_counter() - session_t0) * 1e3
    rebuild_ms = sum(x["pre_rebuild_ms"] + x["post_rebuild_ms"] for x in rows)
    rec = {"model": plan["model"], "config": cfg["name"], "policy": cfg["policy"], "alloc": alloc,
           "rep": rep, "episode": episode, "logged": logged, "big_moe": geo.big_moe, "fit_moe": fit_moe,
           "big_pages": geo.big_pages, "start_moe": start_moe,
           "makespan_ms": makespan, "rebuild_ms": rebuild_ms,
           "served": sum(x["served"] for x in rows), "failed": sum(not x["served"] for x in rows),
           "n_requests": len(rows), "rebuilds": sum(len(e.get("steps", [])) for e in events[1:]),
           "requests": rows, "events": events, "setup_ms": (session_t0 - t_s) * 1e3,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--probe", default="", help="comma list of long-prompt token counts: measure prefill only")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    out = study_dir("runtime_q") / "sessions" / Path(a.plan).stem
    out.mkdir(parents=True, exist_ok=True)
    ns = SimpleNamespace(model=a.model, decode=plan.get("decode", 128), gpu=a.gpu, serve_args=plan["serve_args"],
                         graph=False, rate=plan["rate"], hook=plan.get("hook", "none"),
                         env=[f"{k}={v}" for k, v in plan["env"].items()])
    tag = ("probe_" if a.probe else "server_") + uuid.uuid4().hex[:6]
    srv = Server(ns, out, tag=tag)
    load_s = srv.start()
    tokcache: dict = {}
    try:
        g = cache_geometry(srv.origin)
        geo = Geometry(g)
        steps = srv.rebuild({"num_pages": int(g["num_pages"] * 0.98)})
        g = cache_geometry(srv.origin)
        geo.set_big(g["moe_cache_size"], g["num_pages"])
        with (out / "servers.jsonl").open("a") as f:
            f.write(json.dumps({"tag": tag, "load_s": load_s, "startup": steps, "geometry": g,
                                "big": [geo.big_moe, geo.big_pages], "cmd": srv.cmd, "gpu": gpu_snapshot(a.gpu),
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        if a.probe:
            for n in [int(x) for x in a.probe.split(",")]:
                need = n + 64 + MARGIN_TOKENS
                moe = geo.moe_fitting(need)
                st = resize(srv, moe, geo.pages_for(moe))
                text, _ = long_prompt(a.model, n, tokcache)
                r = generate(srv, text, 16)
                rec = {"tokens": n, "moe": moe, "pages": geo.pages_for(moe), "resize": st, **r}
                with (out / "probe.jsonl").open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[probe] {n} tok -> moe {moe} ({moe / (geo.big_moe or 1):.2f} of big) "
                      f"ttft {(r['ttft_ms'] or float('nan')) / 1e3:.1f}s served={r['served']} err={r['error']}",
                      flush=True)
            return
        done = set()
        res_path = out / "sessions.jsonl"
        if res_path.exists():
            done = {(json.loads(l)["config"], json.loads(l)["rep"]) for l in res_path.read_text().splitlines() if l}
        cfgs = plan["configs"]
        for rep in range(plan["reps"]):
            order = cfgs[:]
            random.Random(1000 + rep).shuffle(order)
            for cfg in order:
                if (cfg["name"], rep) in done:
                    continue
                rec = run_session(a, srv, geo, cfg, plan, rep, out, {}, tokcache)
                with res_path.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                short = [x for x in rec["requests"] if x["kind"] == "short" and x["served"]]
                print(f"[{plan['model']} rep{rep}] {cfg['name']:28s} makespan {rec['makespan_ms'] / 1e3:7.1f}s "
                      f"served {rec['served']}/{rec['n_requests']} rebuild {rec['rebuild_ms'] / 1e3:5.2f}s "
                      f"short {sum(x['ms_per_token'] for x in short) / max(1, len(short)):6.2f} ms/tok", flush=True)
        print("SESSIONS DONE", flush=True)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
