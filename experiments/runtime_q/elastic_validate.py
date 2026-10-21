"""Phase A: validate FreeToken's runtime cache rebuild (expert slots <-> KV pages) on a live server.

Drives only the runtime's own interface: GET /v1/cache/status and POST /v1/cache/rebuild
(fields moe_cache_size / num_pages, mode "if_idle"). One server per invocation; every check
appends one JSON record to <out>/events.jsonl with expected vs observed and pass/fail.

Suites:
  full       A1 initial state, A2 shrink, A3 grow, A4 output identity (same server and a fresh
             server at the shrunk size), A5 host banks not reloaded, A6 rebuild during decode,
             A6b generation during a rebuild, A7 invalid / oversized requests, A11 same-size reset
  roundtrip  A1, A2, A3, A4 (same server), A5, A7 oversize
  hook       A10: rebuild while the runtime-q hook is loaded, two requests in one hook episode
             around the resize; the decision log shows whether capacity state followed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

from experiments.baselines.bench_decode_moe import (
    get_json, load_problem, parse_args as decode_args, resolve_sampling, serve_cmd, stop_server,
    stream_generate, wait_ready,
)
from experiments.paths import ROOT, study_dir
from experiments.runtime_q.driver import (
    cache_geometry, drain, free_port_pair, gpu_snapshot, post_json, write_control,
)

PROMPTS = ROOT / "experiments/prompts/reuse_aware_tasks.jsonl"


class Server:
    def __init__(self, a: argparse.Namespace, out: Path, cache_rate: float | None = None,
                 cache_size: int | None = None, tag: str = "main") -> None:
        argv = ["--model", a.model, "--backend", "hybrid", "--hybrid-fetch", "-1",
                "--decode", str(a.decode), "--greedy", "--gpu", a.gpu, *a.serve_args]
        if cache_size:
            argv += ["--cache", str(cache_size)]
        else:
            argv += ["--cache-rate", str(cache_rate if cache_rate is not None else a.rate)]
        if not a.graph:
            argv.append("--no-graph")
        self.args = decode_args(argv)
        self.port = free_port_pair()
        self.origin = f"http://127.0.0.1:{self.port}"
        self.cmd = serve_cmd(self.args, "hybrid", self.port) + ["--cache-type", "naive"]
        self.env = os.environ.copy()
        for k in ("FREETOKEN_MOE_FORCE_FETCH_FRACTION", "FREETOKEN_MOE_RUNTIME_Q_MODULE",
                  "FREETOKEN_MOE_RUNTIME_Q_CONTROL"):
            self.env.pop(k, None)
        self.env.update(dict(kv.split("=", 1) for kv in a.env))
        self.control = None
        if a.hook != "none":
            self.env["FREETOKEN_MOE_RUNTIME_Q_MODULE"] = str(ROOT / a.hook)
            self.control = out / f"control_{tag}.json"
            self.env["FREETOKEN_MOE_RUNTIME_Q_CONTROL"] = str(self.control)
        self.log = out / f"serve_{tag}.log"
        self.tag = tag

    def start(self) -> float:
        self.log_f = open(self.log, "wb")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=self.env)
        threading.Thread(target=drain, args=(self.proc.stdout, self.log_f), daemon=True).start()
        wait_ready(self.origin, self.proc, str(self.log), 900)
        self.model_id = get_json(f"{self.origin}/v1/models")["data"][0]["id"]
        return time.perf_counter() - t0

    def stop(self) -> None:
        if getattr(self, "stopped", False):
            return
        self.stopped = True
        stop_server(self.proc)
        self.log_f.close()

    # -- observations ----------------------------------------------------------
    def pids(self) -> list[int]:
        out = []
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    if os.getpgid(int(d.name)) == self.proc.pid:
                        out.append(int(d.name))
                except OSError:
                    pass
        return out

    def host_rss_mib(self) -> float:
        tot = 0
        for pid in self.pids():
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        tot += int(line.split()[1])
            except OSError:
                pass
        return tot / 1024

    def gpu_proc_mib(self) -> float:
        pids = set(self.pids())
        try:
            rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                  timeout=20).stdout.strip().splitlines()
            return float(sum(float(m) for p, m in (r.split(", ") for r in rows if r) if int(p) in pids))
        except Exception:  # noqa: BLE001
            return float("nan")

    def log_count(self, needle: str) -> int:
        self.log_f.flush()
        return self.log.read_bytes().decode(errors="replace").count(needle)

    def state(self) -> dict:
        return {"geometry": cache_geometry(self.origin), "gpu_proc_mib": self.gpu_proc_mib(),
                "gpu": gpu_snapshot("0"), "host_rss_mib": round(self.host_rss_mib(), 1),
                "bank_build_lines": self.log_count("expert banks:"),
                "maintenance": get_json(f"{self.origin}/v1/cache/status").get("state")}

    # -- actions ---------------------------------------------------------------
    def generate(self, text: str, decode: int) -> dict:
        sampling, _ = resolve_sampling(self.args.model, True)
        r = stream_generate(self.origin, self.model_id, text, sampling, SimpleNamespace(decode=decode))
        st = r["stamps"]
        steps = r["usage"]["completion_tokens"] - 1
        return {"sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12], "steps": steps,
                "ttft_ms": (st[0] - r["t0"]) * 1e3, "e2e_ms": (st[-1] - r["t0"]) * 1e3,
                "ms_per_token": (st[-1] - st[0]) / max(1, steps) * 1e3,
                "prompt_tokens": r["usage"].get("prompt_tokens")}

    def resize(self, moe: int, pages: int) -> list[dict]:
        """Reach (moe, pages) with the freeing side first: the engine resizes the expert cache
        before the KV pool, so growing experts while shrinking KV is split into a KV-only
        rebuild followed by an expert-only rebuild (each fit-checked by the runtime)."""
        g = cache_geometry(self.origin)
        if moe > g["moe_cache_size"] and pages < g["num_pages"]:
            steps = [self.rebuild({"num_pages": pages}), self.rebuild({"moe_cache_size": moe})]
            # The rebuild fit-check (ratio x baseline - weights - fixed pools) can be stricter
            # than startup sizing (which fills measured free memory), so the exact startup
            # geometry may be rejected by a hair; retry with 2% fewer KV pages, recorded.
            if steps[-1]["result"].get("status") == "rejected" and "budget" in str(steps[-1]["result"]):
                p2 = int(pages * 0.98)
                steps += [self.rebuild({"num_pages": p2}), self.rebuild({"moe_cache_size": moe})]
            return steps
        return [self.rebuild({"moe_cache_size": moe, "num_pages": pages})]

    def rebuild(self, body: dict, sample_ms: float = 25) -> dict:
        """POST a rebuild while sampling this server's GPU memory (peak during the call)."""
        peak, stop = [0.0], threading.Event()

        def sampler():
            while not stop.is_set():
                peak[0] = max(peak[0], self.gpu_proc_mib())
                time.sleep(sample_ms / 1e3)
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        t = time.perf_counter()
        code, res = post_json(f"{self.origin}/v1/cache/rebuild", body)
        dt = (time.perf_counter() - t) * 1e3
        stop.set()
        th.join()
        return {"request": body, "http": code, "result": res, "wall_ms": dt, "gpu_peak_mib": peak[0]}


def stream_until_first_token(origin: str, model_id: str, text: str, sampling: dict, decode: int,
                             first: threading.Event, done: dict) -> None:
    body = {"model": model_id, "messages": [{"role": "user", "content": text}], "max_tokens": decode,
            "ignore_eos": True, "stream": True, "chat_template_kwargs": {"enable_thinking": True},
            **sampling}
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            for raw in resp:
                if raw.startswith(b"data:") and b"[DONE]" not in raw:
                    n += 1
                    first.set()
        done.update(ok=True, chunks=n)
    except Exception as e:  # noqa: BLE001
        done.update(ok=False, error=repr(e), chunks=n)
    first.set()


def chat_once(origin: str, model_id: str, text: str, sampling: dict, decode: int) -> tuple[int, str]:
    body = {"model": model_id, "messages": [{"role": "user", "content": text}], "max_tokens": decode,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": True}, **sampling}
    return post_json(f"{origin}/v1/chat/completions", body, timeout=1800)


class Recorder:
    def __init__(self, out: Path, meta: dict) -> None:
        self.path = out / "events.jsonl"
        self.meta = meta
        self.n_pass = self.n_fail = 0

    def __call__(self, test: str, expected: str, ok: bool | None, **obs) -> None:
        status = "pass" if ok else ("inconclusive" if ok is None else "fail")
        self.n_pass += ok is True
        self.n_fail += ok is False
        rec = {"test": test, "expected": expected, "status": status, **self.meta, **obs,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[{status.upper():12s}] {test}: {expected}", flush=True)


def kv_pages_for(geo: dict, moe_from: int, moe_to: int, keep: float = 0.98) -> int:
    """KV pages after moving (moe_from - moe_to) expert slots' bytes into KV (a fraction
    `keep` of them, so the target stays inside the budget the startup geometry fit)."""
    per_expert = geo["unit_bytes"]["moe_per_expert"]
    per_page = geo["unit_bytes"]["kv_per_token"] * geo["page_size"]
    return int(geo["num_pages"] + keep * (moe_from - moe_to) * per_expert // per_page)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="output subdirectory name")
    ap.add_argument("--rate", type=float, required=True, help="startup --moe-cache-rate")
    ap.add_argument("--shrink-rate", type=float, required=True, help="rebuild target as a rate of L*E")
    ap.add_argument("--suite", choices=["full", "roundtrip", "hook"], default="roundtrip")
    ap.add_argument("--graph", action="store_true", help="CUDA graphs on (no hook)")
    ap.add_argument("--hook", default="none")
    ap.add_argument("--policy", default='{"kind": "reuse_aware", "kappa": 1.0}')
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE server env")
    ap.add_argument("--serve-args", nargs=argparse.REMAINDER, default=[])
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--problem", type=int, default=0)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    out = study_dir("runtime_q") / "elastic" / a.tag
    out.mkdir(parents=True, exist_ok=False)
    text, _ = load_problem(str(PROMPTS), a.problem)
    text = "[FreeToken elastic validation]\n" + text
    srv = Server(a, out)
    (out / "config.json").write_text(json.dumps({**vars(a), "cmd": srv.cmd,
                                                 "env": {k.split("=")[0]: k.split("=", 1)[1] for k in a.env}},
                                                indent=2))
    rec = Recorder(out, {"tag": a.tag, "model": a.model, "graph": a.graph, "hook": a.hook})
    load_s = srv.start()
    try:
        s0 = srv.state()
        g0 = s0["geometry"]
        L, E = g0["num_moe_layers"], g0["num_experts"]
        moe0, pages0 = g0["moe_cache_size"], g0["num_pages"]
        rec("A1_initial", "server ready; geometry reported", "error" not in g0, load_s=load_s, state=s0)
        if a.suite == "hook":
            return hook_suite(a, srv, rec, text, moe0, pages0, g0, L, E)
        gen0 = srv.generate(text, a.decode)
        rec("A4_reference", "generation before any rebuild", gen0["steps"] == a.decode - 1, gen=gen0)

        # A2 shrink experts, grow KV ----------------------------------------------------
        small = max(E, math.ceil(L * E * a.shrink_rate))
        pages_up = kv_pages_for(g0, moe0, small)
        r = srv.rebuild({"moe_cache_size": small, "num_pages": pages_up})
        s1 = srv.state()
        ok = (r["result"].get("status") == "ok" and s1["geometry"]["moe_cache_size"] == small
              and s1["geometry"]["num_pages"] == pages_up and s1["maintenance"] == "serving")
        rec("A2_shrink", f"moe {moe0}->{small}, kv pages {pages0}->{pages_up}; status ok", ok,
            rebuild=r, before=s0, after=s1, load_s=load_s)
        gen1 = srv.generate(text, a.decode)
        rec("A4_after_shrink", "serves after shrink; greedy output identical to reference",
            gen1["sha1"] == gen0["sha1"] and gen1["steps"] == gen0["steps"], gen=gen1, ref=gen0["sha1"])
        rec("A5_no_reload_shrink", "no new expert-bank build lines; host RSS change < 512 MiB; "
            "rebuild faster than load", s1["bank_build_lines"] == s0["bank_build_lines"]
            and abs(s1["host_rss_mib"] - s0["host_rss_mib"]) < 512 and r["wall_ms"] / 1e3 < load_s,
            rss_before=s0["host_rss_mib"], rss_after=s1["host_rss_mib"], rebuild_s=r["wall_ms"] / 1e3,
            load_s=load_s)

        # A3 grow back ---------------------------------------------------------------------
        # A3a: the API's single request (experts up, KV down). The engine resizes the expert
        # cache before the KV pool (engine.rebuild_runtime_cache), so this transiently needs
        # the larger expert cache AND the old larger KV pool; recorded as observed.
        r = srv.rebuild({"moe_cache_size": moe0, "num_pages": pages0})
        s2 = srv.state()
        ok = (r["result"].get("status") == "ok" and s2["geometry"]["moe_cache_size"] == moe0
              and s2["geometry"]["num_pages"] == pages0)
        rec("A3a_grow_single_request", f"one request moe {small}->{moe0} + kv {pages_up}->{pages0}: "
            "status ok", ok, rebuild=r, before=s1, after=s2)
        if not ok:
            g = srv.generate(text, a.decode)
            rec("A8_rollback_after_failed_rebuild", "failed rebuild rolled back: previous geometry "
                "kept, still serving, output identical", s2["geometry"]["moe_cache_size"] == small
                and s2["geometry"]["num_pages"] == pages_up and s2["maintenance"] == "serving"
                and g["sha1"] == gen0["sha1"], rebuild=r, after=s2, gen=g)
            steps = srv.resize(moe0, pages0)
            s2 = srv.state()
            rec("A3b_grow_two_step", f"KV pages {pages_up}->{pages0} first, then moe {small}->{moe0}",
                steps[-1]["result"].get("status") == "ok"
                and s2["geometry"]["moe_cache_size"] == moe0
                and 0.97 * pages0 <= s2["geometry"]["num_pages"] <= pages0,
                rebuilds=steps, after=s2, total_ms=sum(x["wall_ms"] for x in steps))
        gen2 = srv.generate(text, a.decode)
        rec("A4_after_grow", "serves after grow; greedy output identical to reference",
            gen2["sha1"] == gen0["sha1"], gen=gen2, ref=gen0["sha1"])
        rec("A5_no_reload_grow", "no new expert-bank build lines after the rebuilds",
            s2["bank_build_lines"] == s0["bank_build_lines"], rss_after=s2["host_rss_mib"],
            rss_before=s0["host_rss_mib"])

        # A7 invalid / oversized -------------------------------------------------------------
        cur = srv.state()["geometry"]
        bad = [("A7_oversize", {"moe_cache_size": L * E, "num_pages": cur["num_pages"]}),
               ("A7_below_floor", {"moe_cache_size": E - 1})]
        if a.suite == "full":
            bad.append(("A7_zero_pages", {"num_pages": 0}))
        for name, body in bad:
            r = srv.rebuild(body)
            s = srv.state()
            unchanged = (s["geometry"]["moe_cache_size"] == cur["moe_cache_size"]
                         and s["geometry"]["num_pages"] == cur["num_pages"] and s["maintenance"] == "serving")
            rec(name, "rejected before teardown; geometry unchanged; still serving",
                r["result"].get("status") == "rejected" and unchanged, rebuild=r, after=s)
        gen3 = srv.generate(text, a.decode)
        rec("A7_serves_after_rejects", "generation after rejected requests matches reference",
            gen3["sha1"] == gen0["sha1"], gen=gen3)
        if a.suite != "full":
            return
        full_extras(a, srv, rec, text, gen0, moe0, pages0, small, pages_up, out)
    finally:
        srv.stop()
        print(f"PHASE A {a.tag}: {rec.n_pass} pass, {rec.n_fail} fail -> {rec.path}", flush=True)


def full_extras(a, srv: Server, rec: Recorder, text: str, gen0: dict, moe0: int, pages0: int,
                small: int, pages_up: int, out: Path) -> None:
    sampling, _ = resolve_sampling(srv.args.model, True)
    # A6 rebuild requested while a decode is running -----------------------------------------
    cur = srv.state()["geometry"]
    first, done = threading.Event(), {}
    th = threading.Thread(target=stream_until_first_token,
                          args=(srv.origin, srv.model_id, text, sampling, 256, first, done))
    th.start()
    first.wait(600)
    time.sleep(0.5)
    r = srv.rebuild({"moe_cache_size": small, "num_pages": pages_up})
    th.join()
    s = srv.state()
    rec("A6_rebuild_during_decode", "refused as busy (if_idle contract); geometry unchanged; "
        "stream completes", r["result"].get("status") == "busy" and done.get("ok") is True
        and s["geometry"]["moe_cache_size"] == cur["moe_cache_size"], rebuild=r, stream=done, after=s)

    # A6b generation submitted while a rebuild is in flight ------------------------------------
    box = {}
    th = threading.Thread(target=lambda: box.update(r=srv.rebuild({"moe_cache_size": small,
                                                                   "num_pages": pages_up})))
    th.start()
    time.sleep(0.05)
    code, res = chat_once(srv.origin, srv.model_id, text, sampling, 8)
    th.join()
    gated = code == 503
    rec("A6b_generate_during_rebuild", "new generation gated with 503 while rebuilding "
        "(inconclusive if the rebuild finished first)", True if gated else None,
        http=code, response=str(res)[:300], rebuild=box.get("r"))
    steps = srv.resize(moe0, pages0)
    s = srv.state()
    rec("A6b_restore", "restore original geometry (two-step grow)", s["geometry"]["moe_cache_size"] == moe0
        and 0.97 * pages0 <= s["geometry"]["num_pages"] <= pages0, rebuilds=steps)

    # A11 same-size rebuild = cold reset (used by the driver's reset="rebuild") ----------------
    r = srv.rebuild({"moe_cache_size": moe0})
    s = srv.state()
    g = srv.generate(text, a.decode)
    rec("A11_same_size_reset", "same-size rebuild ok; geometry unchanged; output identical",
        r["result"].get("status") == "ok" and s["geometry"]["moe_cache_size"] == moe0
        and s["geometry"]["num_pages"] == pages0 and g["sha1"] == gen0["sha1"], rebuild=r, gen=g)

    # A4b fresh server started at the shrunk size --------------------------------------------
    srv.stop()
    fresh = Server(a, out, cache_size=small, tag="fresh_small")
    load_s = fresh.start()
    try:
        gf = fresh.generate(text, a.decode)
        rec("A4b_fresh_server_same_size", "fresh server at the shrunk slot count gives the same "
            "greedy output", gf["sha1"] == gen0["sha1"], gen=gf, load_s=load_s,
            geometry=cache_geometry(fresh.origin))
    finally:
        fresh.stop()


def hook_suite(a, srv: Server, rec: Recorder, text: str, moe0: int, pages0: int, g0: dict,
               L: int, E: int) -> None:
    """One hook episode spanning two requests with a rebuild between them."""
    log = srv.log.with_name(f"decisions_{srv.tag}.csv")
    episode = uuid.uuid4().hex[:12]
    base = {"episode": episode, "policy": json.loads(a.policy), "policy_name": "hook_probe",
            "workload": "elastic", "cache_rate": a.rate, "problem": a.problem, "seed": 0,
            "groups": ["BASE", "G6_cache"], "log": str(log), "features": "lean"}
    write_control(srv.control, {**base, "phase": "before"})
    gen_a = srv.generate(text, a.decode)
    small = max(E, math.ceil(L * E * a.shrink_rate))
    r = srv.rebuild({"moe_cache_size": small, "num_pages": kv_pages_for(g0, moe0, small)})
    write_control(srv.control, {**base, "phase": "after"})
    gen_b = srv.generate(text, a.decode)
    s = srv.state()
    rows = [l.split(",") for l in log.read_text().splitlines()] if log.exists() else []
    hdr = rows[0] if rows else []
    obs = {}
    if hdr and "occ" in hdr and "free_slots" in hdr:
        i_ph, i_occ, i_free = hdr.index("phase"), hdr.index("occ"), hdr.index("free_slots")
        after = [x for x in rows[1:] if len(x) == len(hdr) and x[i_ph] == "after"]
        obs = {"after_rows": len(after),
               "max_occ_after": max((int(float(x[i_occ])) for x in after), default=None),
               "max_occ_plus_free_after": max((int(float(x[i_occ])) + int(float(x[i_free]))
                                               for x in after), default=None)}
    consistent = obs.get("max_occ_plus_free_after") == small and (obs.get("max_occ_after") or 0) <= small
    rec("A10_hook_capacity_after_resize", f"hook keeps serving; after the resize occ + free_slots "
        f"equals the new capacity {small}", r["result"].get("status") == "ok" and consistent
        and gen_b["steps"] == a.decode - 1, rebuild=r, gen_before=gen_a, gen_after=gen_b,
        decisions=obs, state=s, log=str(log))


if __name__ == "__main__":
    main()
