"""Resume frozen serving measurements, repeating incomplete paired repetitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.pruning import run_pruning_performance75 as frozen
from experiments.pruning.moe_pruning import ROOT, atomic_json, sha256
from experiments.pruning.run_masked_inference75 import assert_frozen


class ResourceRejection(RuntimeError):
    pass


def accepted(study, key, pool, rep, ids):
    target = study / "performance75" / key / f"rep{rep}_{pool}"
    completion = target / "completion.json"
    resources = target / "resource_summary.json"
    if not completion.exists() or not resources.exists():
        return None
    result = json.loads(completion.read_text())
    summary = json.loads(resources.read_text())
    if result.get("status") != "complete" or result.get("resource_status") != "passed" or summary.get("status") != "passed":
        return None
    configuration = json.loads((target / "configuration.json").read_text())
    policy = json.loads((study / "protocol" / (key + "_resources.json")).read_text())
    masks = json.loads((study / "protocol/masks.json").read_text())[key]
    expected_mask = None if pool == "baseline" else masks[pool]["sha256"]
    if (configuration.get("model") != key or configuration.get("pool") != pool
            or configuration.get("rep") != rep or configuration.get("policy") != policy
            or configuration.get("mask_sha256") != expected_mask
            or configuration.get("untraced") is not True or configuration.get("graphs") is not False
            or configuration.get("expert_cache") != "cold per request"):
        raise ValueError("saved serving configuration differs from the frozen arm")
    measurements = result["measurements"]
    cells = [(row["round"], row["request_id"]) for row in measurements]
    if (len(measurements) != 18 or set(cells) != {(i, rid) for i in range(2) for rid in ids}
            or any(row["model"] != key or row["pool"] != pool or row["rep"] != rep for row in measurements)):
        raise ValueError("saved serving request membership differs from the frozen arm")
    for row in measurements:
        raw = json.loads((target / f"round{row['round']}_{row['request_id']}.json").read_text())
        if (raw["output_sha256"] != row["output_sha256"]
                or raw["usage"]["completion_tokens"] != row["completion_tokens"]
                or raw["usage"]["prompt_tokens"] != row["prompt_tokens"]):
            raise ValueError("saved serving measurement differs from its raw response")
    return measurements


def prepare(study, key, pools, ids, attempt, recovery_id):
    for rep in range(5):
        targets = [study / "performance75" / key / f"rep{rep}_{pool}" for pool in ["baseline", *pools]]
        if not any(path.exists() for path in targets):
            continue
        if all(accepted(study, key, pool, rep, ids) is not None for pool in ["baseline", *pools]):
            continue
        archive = study / "diagnostics/serving_resource_recovery" / key / recovery_id / f"rep{rep}_attempt{attempt}"
        archive.mkdir(parents=True, exist_ok=False)
        moved = []
        for path in targets:
            if path.exists():
                moved.append(str(path.relative_to(study)))
                path.rename(archive / path.name)
        atomic_json(archive / "disposition.json", dict(
            status="excluded_incomplete_paired_repetition", original_paths=moved,
            reason="repeat the full repetition in the original treatment order after resource rejection",
            measurements_accepted=False,
        ))


def run(args):
    study, key = Path(args.study).resolve(), args.model
    assert_frozen(study)
    decisions = json.loads((study / "analysis" / (key + "_decisions.json")).read_text())
    pools = [name for name, decision in decisions.items() if decision["eligible_for_matched_serving_diagnostic"]]
    if not pools:
        return frozen.run(args)
    ids = [f"quality_{workload}_{family:02d}_00" for workload in ("math", "code", "incident") for family in (0, 4, 8)]
    protocol_path = study / "performance75" / (key + "_protocol.json")
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        if protocol["pools"] != pools or protocol["request_ids"] != ids or protocol["repetitions"] != 5 or protocol["rounds_per_process"] != 2:
            raise ValueError("serving resume differs from the frozen partition")
        if any(sha256(ROOT / name) != expected for name, expected in protocol["source_hashes"].items()):
            raise ValueError("frozen serving implementation changed")
    original_batch = frozen.batch
    recovery_id = datetime.now(timezone.utc).strftime("recovery_%Y%m%dT%H%M%S%fZ")
    cgroup = Path("/proc/self/cgroup").read_text().strip()
    cgroup_path = Path("/sys/fs/cgroup") / cgroup.split("::", 1)[1].lstrip("/")
    swap_limit = cgroup_path / "memory.swap.max"
    environment = dict(cgroup=cgroup, memory_swap_max=swap_limit.read_text().strip() if swap_limit.exists() else "unavailable",
                       recovery_id=recovery_id, scope="external process environment; frozen serving workers unchanged")
    atomic_json(study / "execution_amendments" / (key + "_" + recovery_id + "_environment.json"), environment)

    def resume_batch(study, key, pool, rep, rows, kwargs, output):
        target = output / key / f"rep{rep}_{pool}"
        if target.exists():
            measurements = accepted(study, key, pool, rep, ids)
            if measurements is None:
                raise ValueError("incomplete serving repetition was not archived")
            print(f"reuse accepted serving {key}/rep{rep}/{pool}", flush=True)
            return measurements
        try:
            measurements = original_batch(study, key, pool, rep, rows, kwargs, output)
            atomic_json(target / "execution_environment.json", environment)
            return measurements
        except (ValueError, RuntimeError) as exc:
            if target.exists():
                atomic_json(target / "execution_environment.json", environment)
            summary = target / "resource_summary.json"
            failure = target / "failure.json"
            rejection = summary.exists() and json.loads(summary.read_text()).get("status") == "failed"
            if not rejection and failure.exists():
                detail = json.loads(failure.read_text())
                rejection = (detail.get("phase") == "settling"
                             and detail.get("error") == "RuntimeError('host memory did not settle within the bounded startup wait')")
            if rejection:
                raise ResourceRejection(f"resource rejection in {key}/rep{rep}/{pool}") from exc
            raise

    frozen.batch = resume_batch
    try:
        for attempt in range(3):
            prepare(study, key, pools, ids, attempt, recovery_id)
            try:
                frozen.run(args)
                return
            except ResourceRejection:
                if attempt == 2:
                    raise
                print("resource rejection; preserve and repeat the incomplete paired repetition", flush=True)
    finally:
        frozen.batch = original_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--model", required=True, choices=("gemma_nvfp4", "gpt_oss_mxfp4", "qwen_nvfp4"))
    run(parser.parse_args())
