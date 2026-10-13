"""Finish scoring and closeout after serial actual M75 capture collection."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.pruning.moe_pruning import ROOT, atomic_json


def validate_reference_layout(study):
    import numpy as np
    from experiments.pruning.run_masked_inference75 import accepted_quality_batches

    records = []
    models = json.loads((study / "protocol/models.json").read_text())
    for key, model in models.items():
        config = json.loads((ROOT / model["path"] / "config.json").read_text())
        config = config.get("text_config", config)
        expected = {f"layer_{i:02d}.npy": (256, config["hidden_size"]) for i in range(model["layers"])}
        expected["logits.npy"] = (256, config["vocab_size"])
        quality_batches,_,_=accepted_quality_batches(study,key)
        pilots = [path.parent for path in (study / "runs" / key).glob("*/completion.json")
                  if json.loads(path.read_text()).get("resource_status") == "passed" and json.loads(path.read_text()).get("stage") == "pilot"]
        if len(pilots)!=1:
            raise ValueError("reference validation requires one accepted pilot")
        accepted = [*pilots,*quality_batches]
        for batch in accepted:
            completion = json.loads((batch / "completion.json").read_text())
            stage = completion["stage"]
            references = list(batch.glob("*/baseline_data"))
            if len(references) != (3 if stage == "pilot" else len(completion["sequence_ids"])+1):
                raise ValueError("lossless reference count differs from measured and warmup sequences")
            for directory in references:
                if {path.name for path in directory.iterdir()} != set(expected):
                    raise ValueError("reference arrays are missing or unexpected")
                for name, dimensions in expected.items():
                    path = directory / name
                    with path.open("rb") as stream:
                        if np.lib.format.read_magic(stream) != (2, 0):
                            raise ValueError("direct reference does not have an NPY v2 header")
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                        offset = stream.tell()
                    if shape != dimensions or fortran or dtype != np.dtype("<f4") or offset != 4096:
                        raise ValueError(f"invalid lossless reference layout: {path}")
                    if path.stat().st_size != offset + int(np.prod(shape)) * 4:
                        raise ValueError(f"incomplete lossless reference: {path}")
                    records.append(dict(path=str(path.relative_to(study)), shape=shape, dtype=str(dtype), data_offset=offset))
    atomic_json(study / "analysis/reference_array_validation.json", dict(status="passed", arrays=len(records), records=records))


def run(args):
    parent = Path(args.parent).resolve()
    study = Path(args.study).resolve()
    captures = Path(args.capture_progress).resolve()
    target = parent / "orchestration" / args.run_id
    target.mkdir(parents=True, exist_ok=False)
    progress = dict(status="waiting_for_accepted_captures", jobs=[], study=str(study))

    def save():
        progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(target / "progress.json", progress)

    def launch(name, module, arguments):
        command = ["uv", "run", "--no-sync", "--project", str(ROOT), "python", "-m", module, *map(str, arguments)]
        with (target / (name + ".log")).open("w") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            job = dict(name=name, status="running", pid=proc.pid, command=command)
            progress["jobs"].append(job)
            progress["status"] = name
            save()
            while proc.poll() is None:
                time.sleep(10)
                save()
            job.update(returncode=proc.returncode, status="complete" if proc.returncode == 0 else "failed")
            save()
            if proc.returncode:
                raise RuntimeError(f"{name} failed; see {target / (name + '.log')}")

    save()
    try:
        while True:
            state = json.loads(captures.read_text())
            if state["status"] == "stopped_on_failure":
                raise RuntimeError("capture controller stopped: " + state.get("error", "unknown error"))
            if state["status"] == "actual_quality_collection_complete" and state["actual_quality_complete"]:
                break
            time.sleep(10)
        launch("score_actual75", "experiments.pruning.report_masked_inference75", ["--study", study])
        for model in ("gemma_nvfp4", "gpt_oss_mxfp4", "qwen_nvfp4"):
            launch(model + "_serving_disposition", args.serving_module, ["--study", study, "--model", model])
        progress["status"] = "validating_lossless_reference_layout"
        save()
        validate_reference_layout(study)
        launch("export_actual75", "experiments.pruning.export_pruning_supplement", ["--study", study, "--parent", parent])
        manifest = study / "closeout75_actual/artifact_manifest.json"
        records = json.loads(manifest.read_text())["artifacts"]
        from experiments.pruning.moe_pruning import sha256

        progress["status"] = "verifying_supplement_manifest"
        save()
        for index, (name, expected) in enumerate(records.items()):
            path = parent / name
            if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
                raise ValueError(f"supplement artifact changed after export: {name}")
            if index % 1000 == 0:
                print(f"verified supplement artifacts {index}/{len(records)}", flush=True)
                save()
        progress.update(status="complete", report=str(study / "closeout75_actual/report.md"),
                        manifest_sha256=sha256(manifest), verified_artifacts=len(records), deferred60_50="not_started")
        save()
    except BaseException as exc:
        progress.update(status="stopped_on_failure", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--capture-progress", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--serving-module", choices=("experiments.pruning.run_pruning_performance75", "experiments.pruning.resume_pruning_performance75"), default="experiments.pruning.run_pruning_performance75")
    run(parser.parse_args())
