"""Launch one recorded run through uv, with a fresh output directory."""
import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from experiments.common.environment import collect
from experiments.paths import ROOT, python_command

STUDIES = {"baselines", "bandwidth", "prefill_overlap", "routing_locality", "split_policy", "runtime_q", "pruning", "common"}
ALIASES = {"prefill_overlap": {"run": "run_prefill", "analyze": "analyze_prefill"},
           "routing_locality": {"analyze": "lru_locality", "plot": "plots_locality"},
           "split_policy": {"run": "run_experiment", "plot": "plots"},
           "runtime_q": {"run": "driver", "train": "train_all", "analyze": "analyze_eval", "plot": "figures"}}
GPU_COMMANDS = {"baseline", "bench_decode_moe", "bench_prefill_bandwidth", "bench_load_weight_generic", "bench_offload_cache_copy",
                "run_prefill", "run_experiment", "driver", "collect_moe_routes", "collect_gemma_routes", "collect_gemma_rerouting",
                "run_masked_inference75", "run_pruning_performance75", "representative_layer_sweep", "branch_oracle", "bandwidth_study",
                "m_reduction_collect", "m_reduction_nvfp4_collect", "m_reduction_calibrate", "m_reduction_numerics",
                "elastic_validate", "elastic_session"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    if sys.argv[1:] == ["--list"]:
        print(json.dumps({study: {"modules": sorted(p.stem for p in (ROOT / "experiments" / study).glob("*.py") if p.stem != "__init__"),
                                 "aliases": ALIASES.get(study, {})} for study in sorted(STUDIES)}, indent=2))
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", choices=sorted(STUDIES))
    p.add_argument("command", help="Python module stem or run/analyze/plot alias")
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    name = ALIASES.get(a.study, {}).get(a.command, a.command)
    if not name.isidentifier() or not (ROOT / "experiments" / a.study / (name + ".py")).is_file():
        p.error(f"Unknown command {a.study}/{name}")
    args = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
    command = [*python_command(), "-m", f"experiments.{a.study}.{name}", *args]
    if "--help" in args or "-h" in args:
        source = ROOT / "experiments" / a.study / (name + ".py")
        if "ArgumentParser(" not in source.read_text():
            print(f"{a.study}/{name}: this analysis module takes no CLI arguments.\n"
                  "Use EXPERIMENT_OUTPUT_ROOT to select a working copy of its study data.")
            return
        raise SystemExit(subprocess.call(command, cwd=ROOT))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    base = Path(os.environ.get("EXPERIMENT_RUNS_ROOT", ROOT / "results/runs")).expanduser().resolve()
    run = base / a.study / run_id
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.setdefault("EXPERIMENT_OUTPUT_ROOT", str(run / "outputs"))
    env["EXPERIMENT_RUN_DIR"] = str(run)
    env["EXPERIMENT_LOG_DIR"] = str(run / "logs")
    env["PYTHONUNBUFFERED"] = "1"
    (run / "logs").mkdir()
    record = {"run_id": run_id, "study": a.study, "module": name, "started_utc": utc(),
              "status": "preflight", "command": command, "output_root": env["EXPERIMENT_OUTPUT_ROOT"]}
    def save():
        (run / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    save()
    print(f"Run directory: {run}", flush=True)
    try:
        metadata = collect(name in GPU_COMMANDS and "--dry-run" not in args)
        (run / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
        record["status"] = "running"
        save()
        with (run / "logs/console.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, start_new_session=True)
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                rc = proc.wait()
            except KeyboardInterrupt:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = 130
        record.update(status=("dry_run" if "--dry-run" in args else "completed") if rc == 0 else "failed", returncode=rc)
    except Exception as exc:
        record.update(status="preflight_failed", error=str(exc), returncode=1)
        print(str(exc), flush=True)
    finally:
        record["finished_utc"] = utc()
        save()
    print(f"{record['status']}: {run}", flush=True)
    raise SystemExit(record["returncode"])


if __name__ == "__main__":
    main()
