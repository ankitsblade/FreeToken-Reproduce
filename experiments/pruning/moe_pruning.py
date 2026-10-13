"""Audit and analyze complete-request, phase-aware MoE routing experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, study_dir, runtime_pythonpath, engine_sources
MODELS = ROOT / "configs/pruning/models.json"
PROTOCOL = ROOT / "configs/pruning/protocol.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def command(*args) -> dict:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {"command": list(args), "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def snapshot(target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    records = {}
    sources = engine_sources()
    for directory, suffixes in [("experiments", (".py", ".jsonl")), ("configs", (".json",)),
                               ("docs", (".md",)), ("scripts", (".sh",)), ("runtime", (".json", ".txt", ".patch"))]:
        for source in (ROOT / directory).rglob("*"):
            if source.is_file() and source.suffix in suffixes:
                sources[str(source.relative_to(ROOT))] = source
    for name in ("pyproject.toml", "uv.lock", "CONTRIBUTING.md", "AGENTS.md"):
        sources[name] = ROOT / name
    for name in ("pyproject.toml", "setup.py"):
        sources["runtime_source/" + name] = ENGINE_SOURCE / name
    for relative, source in sorted(sources.items()):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records[str(relative)] = sha256(source)
    atomic_json(target / "files.json", records)
    return {"manifest_sha256": sha256(target / "files.json"), "files": len(records)}


def audit(args) -> None:
    study = Path(args.study).resolve()
    study.mkdir(parents=True, exist_ok=False)
    atomic_json(study / "protocol/models.json", json.loads(MODELS.read_text()))
    atomic_json(study / "protocol/protocol.json", json.loads(PROTOCOL.read_text()))
    atomic_json(study / "protocol/resource_policy.json", json.loads((ROOT / "configs/pruning/resource_policy.json").read_text()))
    probes = {
        "gpu": command("nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version", "--format=csv"),
        "compute": command("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"),
        "cpu": command("lscpu", "--json"), "memory": command("free", "-b"),
        "disk": command("df", "-B1", "."), "head": command("git", "rev-parse", "HEAD"),
        "status": command("git", "status", "--short"),
        "versions": command("uv", "run", "--no-sync", "python", "-c",
                            "import torch,sys; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"),
    }
    atomic_json(study / "environment/system.json", probes)
    (study / "environment/tracked.diff").write_text(command("git", "diff", "--binary")["stdout"])
    atomic_json(study / "environment/source_snapshot.json", snapshot(study / "environment/source_snapshot"))
    models = json.loads(MODELS.read_text())
    reports = {}
    from safetensors import safe_open
    for key, model in models.items():
        if not model["required"]:
            reports[key] = {"status": model.get("status", "unqualified_resources"),
                            "reason": "Additional family requires metadata/resource qualification before weight loading"}
            continue
        local = ROOT / model["path"]
        config = json.loads((local / "config.json").read_text())
        text = config.get("text_config", config)
        actual = [text["num_hidden_layers"], text.get("num_experts", text.get("num_local_experts")), text["num_experts_per_tok"]]
        expected = [model["layers"], model["experts"], model["top_k"]]
        if actual != expected:
            raise ValueError(f"{key}: checkpoint geometry {actual} != {expected}")
        index = json.loads((local / "model.safetensors.index.json").read_text())
        records = []
        wanted = set(index["weight_map"])
        found = set()
        for name in sorted(set(index["weight_map"].values())):
            path = local / name
            metadata = (local / ".cache/huggingface/download" / (name + ".metadata")).read_text().splitlines()
            if metadata[0] != model["revision"]:
                raise ValueError(f"{key}/{name}: revision mismatch")
            digest = sha256(path)
            if len(metadata[1]) != 64 or digest != metadata[1]:
                raise ValueError(f"{key}/{name}: shard hash mismatch")
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                shard_keys = set(shard.keys())
            expected_keys = {k for k, value in index["weight_map"].items() if value == name}
            if shard_keys != expected_keys:
                raise ValueError(f"{key}/{name}: shard/index keys disagree")
            found.update(shard_keys)
            records.append({"file": name, "bytes": path.stat().st_size, "sha256": digest,
                            "revision": metadata[0], "tensor_keys": len(shard_keys)})
            print(f"validated {key}/{name}", flush=True)
        if found != wanted:
            raise ValueError(f"{key}: incomplete checkpoint index")
        small = {}
        for path in sorted(local.iterdir()):
            if path.is_file() and (path.suffix in (".json", ".jinja") or path.name.startswith("tokenizer")):
                meta = local / ".cache/huggingface/download" / (path.name + ".metadata")
                provenance = meta.read_text().splitlines()[0] if meta.exists() else None
                if provenance != model["revision"]:
                    raise ValueError(f"{key}/{path.name}: missing or wrong revision metadata")
                small[path.name] = {"sha256": sha256(path), "revision": provenance}
        reports[key] = {"status": "checkpoint_validated", "geometry": actual,
                        "shards": records, "configuration": small, "config": config}
        atomic_json(study / "environment/checkpoints.json", reports)
    atomic_json(study / "protocol/state.json", {"phase": 0, "status": "audit_complete",
                "created_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_status": reports})
    print(study)


def validate_replay(args):
    from experiments.pruning.pruning_cache_replay import validate_native

    study = Path(args.study).resolve()
    models = json.loads((study / "protocol/models.json").read_text())
    base = study / ("pilots" if args.stage == "pilot" else "routes") / args.model
    results = []
    for path in sorted(base.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest.get("stage") != args.stage:
            continue
        for name, checksum in manifest["checksums"].items():
            if sha256(path.parent / name) != checksum:
                raise ValueError(f"capture checksum changed: {path.parent/name}")
        if not (path.parent / "decode_initial_cache.json").exists():
            results.append({"capture_id": manifest["capture_id"], "status": "no_decode", "events": 0})
            continue
        results.append({"capture_id": manifest["capture_id"], **validate_native(path.parent, models[args.model])})
    if not results:
        raise ValueError("no accepted captures for replay validation")
    report = {"model": args.model, "stage": args.stage, "captures": len(results),
              "status": "matched" if all(r["status"] in ("matched", "no_decode") for r in results) else "mismatch",
              "interpretation": "native observed-state fidelity; cold-start counterfactual not yet qualified",
              "results": results}
    target = study / "replay_validation" / f"{args.model}_{args.stage}.json"
    atomic_json(target, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    if report["status"] != "matched":
        raise ValueError("native cache replay mismatch; inspect the report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--study", required=True)
    p.set_defaults(func=audit)
    p = sub.add_parser("analyze")
    p.add_argument("--study", required=True)
    p.add_argument("--retention", type=float, default=.75)
    from experiments.pruning.pruning_analysis import analyze

    p.set_defaults(func=analyze)
    p = sub.add_parser("validate-replay")
    p.add_argument("--study", required=True)
    p.add_argument("--model", choices=("qwen_nvfp4", "gpt_oss_mxfp4"), required=True)
    p.add_argument("--stage", choices=("pilot", "primary", "repeat"), required=True)
    p.set_defaults(func=validate_replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
