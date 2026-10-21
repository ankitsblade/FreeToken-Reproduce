"""Freeze and verify all three pinned checkpoints for the fresh m study."""

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from experiments.paths import ROOT
from experiments.pruning.moe_pruning import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((ROOT / "configs/m_reduction/stage0.json").read_text())
    target = args.study / "checkpoint_audit.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite checkpoint audit")
    audit = {"status": "running", "models": {}}
    atomic_json(target, audit)
    for key, model in plan["models"].items():
        root = ROOT / model["path"]
        config = json.loads((root / "config.json").read_text())
        text = config.get("text_config", config)
        geometry = [text["num_hidden_layers"], text.get("num_experts", text.get("num_local_experts")),
                    text.get("num_experts_per_tok", text.get("top_k_experts"))]
        if geometry != [model["layers"], model["experts"], model["top_k"]]:
            raise ValueError(key + ": geometry differs")
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        shards = []
        for name in sorted(set(index.values())):
            metadata = (root / ".cache/huggingface/download" / (name + ".metadata")).read_text().splitlines()
            path = root / name
            if metadata[0] != model["revision"] or sha256(path) != metadata[1]:
                raise ValueError(key + ": weight revision/hash differs: " + name)
            with safe_open(str(path), framework="np") as file:
                keys = set(file.keys())
            if keys != {key for key, shard in index.items() if shard == name}:
                raise ValueError("Shard/index tensor mismatch: " + name)
            shards.append({"file": name, "bytes": path.stat().st_size, "sha256": metadata[1],
                           "revision": metadata[0], "tensor_count": len(keys)})
            print(f"Verified {key}/{name}", flush=True)
        small = {}
        for path in sorted(root.iterdir()):
            if path.is_file() and (path.suffix in (".json", ".jinja") or path.name.startswith("tokenizer")):
                metadata = root / ".cache/huggingface/download" / (path.name + ".metadata")
                if not metadata.exists() or metadata.read_text().splitlines()[0] != model["revision"]:
                    raise ValueError("Missing/pin-mismatched metadata: " + str(path))
                small[path.name] = sha256(path)
        audit["models"][key] = {"status": "verified", "repository": model["repository"],
                                "revision": model["revision"], "geometry": geometry,
                                "shards": shards, "metadata_hashes": small}
        atomic_json(target, audit)
    audit["status"] = "verified"
    atomic_json(target, audit)
    print("All three checkpoint weight/metadata audits passed", flush=True)


if __name__ == "__main__":
    main()
