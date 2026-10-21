"""Record unpruned, eager serving baselines with fixed cache geometry."""
import argparse
import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

from experiments.paths import ROOT, python_command
from experiments.pruning.pruning_resources import read_resources, monitor_resources


def main():
    profiles = json.loads((ROOT / "configs/baselines.json").read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=profiles, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode", type=int, default=128)
    p.add_argument("--problem", type=int, default=0)
    p.add_argument("--cache-slots", type=int)
    p.add_argument("--memory-ratio", type=float)
    p.add_argument("--cpu-threads", type=int, default=5)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--allow-unpinned", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.repeats < 1 or a.decode < 2:
        p.error("repeats must be positive and decode must be at least two")
    model = a.model.expanduser().resolve()
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    profile = dict(profiles[a.profile])
    pinned = json.loads((ROOT / "configs/pruning/models.json").read_text())[a.profile]
    metadata = model / ".cache/huggingface/download/config.json.metadata"
    revision = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    if revision != pinned["revision"] and not a.allow_unpinned:
        p.error("Checkpoint revision is unknown or differs from the pin; use --allow-unpinned for a separately identified arm")
    if a.cpu_threads < 1:
        p.error("cpu-threads must be positive")
    profile["cpu_threads"] = a.cpu_threads
    actual = (text["num_hidden_layers"], text.get("num_experts", text.get("num_local_experts")),
              text.get("num_experts_per_tok", text.get("top_k_experts")))
    expected = tuple(profile[k] for k in ("layers", "experts", "top_k"))
    if actual != expected:
        p.error(f"Checkpoint geometry {actual} differs from {expected}")
    if a.cache_slots is not None:
        if a.cache_slots < 2 * profile["experts"]:
            p.error("cache-slots must hold at least two layers' expert universes")
        profile["cache_slots"] = a.cache_slots
    if a.memory_ratio is not None:
        if not 0 < a.memory_ratio < 1:
            p.error("memory-ratio must be between zero and one")
        profile["memory_ratio"] = a.memory_ratio
    command = [*python_command(), "-m", "experiments.baselines.bench_decode_moe", "--model", str(model),
               "--gpu", a.gpu, "--backend", "hybrid", "--decode", str(a.decode), "--problem", str(a.problem),
               "--cache", str(profile["cache_slots"]), "--mem-ratio", str(profile["memory_ratio"]),
               "--max-seq-len", str(profile["max_seq_len"]), "--max-prefill-length", "1024",
               "--cache-type", "naive", "--no-graph", "--greedy", "--serial-load", "--cpu-threads", str(a.cpu_threads),
               "--aime", str(ROOT / "experiments/prompts/routing_workloads.jsonl")]
    if a.profile != "gpt_oss_mxfp4":
        command.append("--text-model-only")
    if a.dry_run:
        print(json.dumps({"commands": [command] * a.repeats, "profile": profile}, indent=2))
        return
    run = Path(os.environ["EXPERIMENT_RUN_DIR"])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FREETOKEN_MOE_", "FREETOKEN_PRUNING_", "FREETOKEN_HYBRID_"))}
    env.pop("FREETOKEN_BENCHBW_PATH", None)
    numeric = {"mxfp4_alignment": env.get("FREETOKEN_M_REDUCTION_MXFP4_ALIGN") == "1",
               "mxfp4_dot64": env.get("FREETOKEN_M_REDUCTION_MXFP4_DOT64") == "1",
               "nvfp4_alignment": env.get("FREETOKEN_M_REDUCTION_NVFP4_ALIGN") == "1"}
    if numeric["nvfp4_alignment"] and a.profile == "gpt_oss_mxfp4":
        p.error("NVFP4 alignment cannot be used for GPT-OSS")
    calibration = a.calibration
    if calibration is None and any(numeric.values()):
        # Bind already running recovery controllers to their recorded fresh profile.
        runs_root = Path(env.get("EXPERIMENT_RUNS_ROOT", ""))
        recovery = runs_root.parent / "recovery_controller.json"
        if recovery.is_file():
            control = json.loads(recovery.read_text())
            if control["runtime"] != env.get("FREETOKEN_SOURCE_DIR"):
                p.error("Recovery baseline runtime differs from its controller")
            candidates = []
            for path in (runs_root.parent / "calibration").glob("*.control.json"):
                row = json.loads(path.read_text())
                if row.get("model") == a.profile and all(row.get(k, False) == v for k, v in numeric.items()):
                    candidates.append(path.with_name(path.name.removesuffix(".control.json") + ".json"))
            if len(candidates) == 1:
                calibration = candidates[0]
        if calibration is None:
            p.error("A numerical correction baseline requires --calibration from that arm")
    if calibration is not None:
        calibration = calibration.expanduser().resolve()
        if not calibration.is_file():
            p.error("Calibration profile is missing")
        if any(numeric.values()):
            controls = calibration.with_suffix(".control.json")
            if not controls.is_file():
                p.error("A numerical correction requires its calibration control file")
            controls = json.loads(controls.read_text())
            if controls.get("model") != a.profile or any(controls.get(k, False) != v for k, v in numeric.items()):
                p.error("Calibration model/precision differs from the baseline arm")
        env["FREETOKEN_BENCHBW_PATH"] = str(calibration)
    env["FREETOKEN_DISABLE_OVERLAP_SCHEDULING"] = "1"
    env["MALLOC_ARENA_MAX"] = "2"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    record = {"profile": a.profile, "settings": profile, "model": str(model), "config": config,
              "numeric_controls": numeric,
              "calibration": str(calibration) if calibration else None,
              "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest() if calibration else None,
              "expected_checkpoint": pinned, "observed_config_revision": revision,
              "revision_verified": revision == pinned["revision"],
              "config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
              "prompt_sha256": hashlib.sha256((ROOT / "experiments/prompts/routing_workloads.jsonl").read_bytes()).hexdigest(),
              "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), ROOT / "experiments/baselines/bench_decode_moe.py")},
              "repeats": a.repeats, "commands": [], "status": "running",
              "interpretation": "New-device unpruned baseline; no numerical equivalence to sealed 75% serving arms asserted"}
    metadata_files = ("generation_config.json", "tokenizer_config.json", "model.safetensors.index.json")
    record["metadata_sha256"] = {name: hashlib.sha256((model / name).read_bytes()).hexdigest() for name in metadata_files if (model / name).is_file()}
    for rep in range(a.repeats):
        active = subprocess.check_output(["nvidia-smi", "-i", a.gpu, "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if active:
            raise RuntimeError(f"GPU has compute processes: {active}")
        resources = read_resources(-1)
        if resources["host_available_bytes"] < 4 * (1 << 30):
            raise RuntimeError("Baseline requires at least 4 GiB available host RAM before loading")
        cmd = [*command, "--json", str(run / f"repeat_{rep:02d}.jsonl")]
        record["commands"].append(cmd)
        (run / "baseline.json").write_text(json.dumps(record, indent=2) + "\n")
        proc = subprocess.Popen(cmd, env=env, cwd=ROOT)
        stop = threading.Event()
        monitor = threading.Thread(target=monitor_resources, args=(proc, stop, run / f"resources_{rep:02d}.jsonl"))
        monitor.start()
        try:
            rc = proc.wait()
        finally:
            stop.set()
            monitor.join()
        if rc:
            raise subprocess.CalledProcessError(rc, cmd)
    record["status"] = "completed"
    (run / "baseline.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
