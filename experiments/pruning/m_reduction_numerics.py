"""Record same-hidden-state CPU/GPU expert comparisons; no timing claims."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from experiments.paths import ROOT, engine_sources, python_command, runtime_pythonpath
from experiments.baselines.bench_decode_moe import free_port, get_json, stop_server, wait_ready
from experiments.pruning.collect_moe_routes import json_lines, render, request
from experiments.pruning.moe_pruning import atomic_json, sha256


def compare(a, b):
    delta = a.double() - b.double()
    return {"equal": bool(a.equal(b)), "different_elements": int((a != b).sum()),
            "elements": a.numel(), "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt())}


def probe(layer, cache, x, weights, ids):
    """Called only by the separately recorded diagnostic runtime hook."""
    import torch
    from freetoken.moe.layer_replay import CacheSnapshot, SlotBackup

    target = os.environ.get("FREETOKEN_M_REDUCTION_NUMERICS_OUT")
    if not target or layer.layer_id != int(os.environ.get("FREETOKEN_M_REDUCTION_NUMERICS_LAYER", "0")) or getattr(cache, "_numerics_done", False):
        return
    cache._numerics_done = True
    directory = Path(target)
    snapshot = CacheSnapshot(cache)
    backup = SlotBackup()
    old = cache.hybrid_max_fetch, cache.hybrid_fetch_fraction, cache.decode_trace
    outputs = {}
    try:
        torch.cuda.synchronize()
        cache.decode_trace = None
        cache.hybrid_max_fetch = ids.numel()
        cache.hybrid_fetch_fraction = 0.0
        slots = ids.clone()
        cache.ensure_experts_hybrid(layer.layer_id, slots)
        backup.capture(cache)
        cache.copy_missing()
        if bool((slots < 0).any()):
            raise RuntimeError("Diagnostic could not place every routed expert on GPU")
        def gpu(w):
            return layer._expert_gemm(cache, x, w, slots, views=cache.bank_views(),
                                      n=None, alphas=cache.alphas_for_slots(layer.layer_id),
                                      is_prefill=False)
        outputs["cpu_all"] = cache.cpu_executor.decode(layer.layer_id, x, weights, ids).clone()
        outputs["gpu_all"] = gpu(weights).clone()
        for route in range(ids.shape[1]):
            w = torch.zeros_like(weights)
            w[:, route] = weights[:, route]
            cpu_ids = torch.full_like(ids, -1)
            cpu_ids[:, route] = ids[:, route]
            outputs[f"cpu_route_{route}"] = cache.cpu_executor.decode(layer.layer_id, x, w, cpu_ids).clone()
            outputs[f"gpu_route_{route}"] = gpu(w).clone()
        torch.cuda.synchronize()
        report = {"scope": "first_decode_identical_hidden_state", "layer_id": layer.layer_id,
                  "model": os.environ["FREETOKEN_M_REDUCTION_NUMERICS_MODEL"],
                  "expert_ids": ids.tolist(), "router_weights": weights.tolist(),
                  "all_routes": compare(outputs["cpu_all"], outputs["gpu_all"]),
                  "per_route": [compare(outputs[f"cpu_route_{i}"], outputs[f"gpu_route_{i}"])
                                for i in range(ids.shape[1])]}
        for branch in ("cpu", "gpu"):
            accumulation_dtype = torch.float64 if outputs[branch + "_all"].dtype == torch.float64 else torch.float32
            parts = torch.stack([outputs[f"{branch}_route_{i}"].to(accumulation_dtype) for i in range(ids.shape[1])])
            report[branch + "_group_vs_rounded_routes"] = compare(outputs[branch + "_all"],
                                                                  parts.sum(0).to(outputs[branch + "_all"].dtype))
        saved = {"hidden_state": x.cpu(), "weights": weights.cpu(), "ids": ids.cpu(),
                 "outputs": {key: value.cpu() for key, value in outputs.items()},
                 "bank_schema": list(cache.bank_schema),
                 "banks": {name: source[layer.layer_id].index_select(0, ids.flatten().cpu().long())
                           for name, (source, _) in zip(cache.bank_schema, cache.banks)}}
        torch.save(saved, directory / "fixed_state.pt")
        atomic_json(directory / "comparison.json", report)
        print("Same-state expert comparison recorded: " + str(directory), flush=True)
    finally:
        backup.restore(cache)
        snapshot.restore(cache)
        cache.hybrid_max_fetch, cache.hybrid_fetch_fraction, cache.decode_trace = old
        torch.cuda.synchronize()
        if not snapshot.matches(cache):
            raise RuntimeError("Numerical diagnostic did not restore cache bookkeeping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=("gpt_oss_mxfp4", "qwen_nvfp4", "gemma_nvfp4"), required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--align-mxfp4", action="store_true")
    parser.add_argument("--dot64", action="store_true")
    parser.add_argument("--align-nvfp4", action="store_true")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--cpu-no-vnni", action="store_true")
    args = parser.parse_args()
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT / "configs/m_reduction/stage0.json"
    plan = json.loads(plan_path.read_text())
    model = plan["models"][args.model]
    if not 0 <= args.layer < model["layers"]:
        parser.error("Requested layer is outside the model")
    if args.dot64 and not args.align_mxfp4:
        parser.error("FP64 projections require MXFP4 alignment")
    if args.cpu_no_vnni and args.model == "gpt_oss_mxfp4":
        parser.error("This activation-quantization diagnostic is specific to NVFP4")
    if args.align_nvfp4 and args.model == "gpt_oss_mxfp4":
        parser.error("NVFP4 alignment is specific to Qwen/Gemma")
    if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
        raise RuntimeError("GPU is occupied")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("FREETOKEN_MOE_", "FREETOKEN_HYBRID_", "FREETOKEN_M_REDUCTION_"))}
    arm = directory / "arm.json"
    env.update({"PYTHONPATH": runtime_pythonpath(), "FREETOKEN_BENCHBW_PATH": str(args.calibration.resolve()),
                "FREETOKEN_MOE_ROUTING_ARM": str(arm), "FREETOKEN_M_REDUCTION_FILL_FLOOR": "0",
                "FREETOKEN_M_REDUCTION_NUMERICS_OUT": str(directory),
                "FREETOKEN_M_REDUCTION_NUMERICS_MODEL": args.model,
                "FREETOKEN_M_REDUCTION_NUMERICS_LAYER": str(args.layer),
                "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    if args.align_mxfp4:
        if args.model != "gpt_oss_mxfp4":
            parser.error("MXFP4 alignment is specific to GPT-OSS")
        env["FREETOKEN_M_REDUCTION_MXFP4_ALIGN"] = "1"
    if args.dot64:
        env["FREETOKEN_M_REDUCTION_MXFP4_DOT64"] = "1"
    if args.align_nvfp4:
        env["FREETOKEN_M_REDUCTION_NVFP4_ALIGN"] = "1"
    if args.cpu_no_vnni:
        env["FREETOKEN_CPU_MOE_NO_VNNI"] = "1"
    else:
        env.pop("FREETOKEN_CPU_MOE_NO_VNNI", None)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    command = [*python_command(), "-m", "freetoken.cli", "serve", "--model", str(ROOT / model["path"]),
               "--gpu", "0", "--host", "127.0.0.1", "--port", str(port), "--moe-strategy", "hybrid",
               "--max-running-requests", "1", "--max-seq-len-override", str(model["max_seq_len"]),
               "--max-prefill-length", "1024", "--memory-ratio", str(model["memory_ratio"]),
               "--cuda-graph-max-bs", "0", "--moe-hybrid-max-fetch", "-1",
               "--moe-cache-size", str(model["cache_slots"]), "--cache-type", "naive",
               "--expert-load", "serial", "--moe-cpu-threads", str(plan["cpu_threads"])]
    if args.model != "gpt_oss_mxfp4":
        command.append("--text-model-only")
    row = next(row for row in json_lines(ROOT / plan["dataset"]) if row["request_id"] == "math_000")
    # Resolve tokenizer state in this controller before starting the weight loader.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / model["path"], local_files_only=True)
    prompt_ids = render(tokenizer, row["prompt_text"], plan["chat_template_kwargs"])
    del tokenizer
    atomic_json(directory / "configuration.json", {"command": command, "model": model,
                "request": row, "maximum_completion_tokens": 2, "sampling": plan["sampling"],
                "mxfp4_alignment": args.align_mxfp4,
                "nvfp4_alignment": args.align_nvfp4,
                "mxfp4_dot64": args.dot64, "layer_id": args.layer,
                "cpu_vnni_disabled": args.cpu_no_vnni,
                "runtime_hashes": {name: sha256(path) for name, path in engine_sources().items()},
                "diagnostic_source_sha256": sha256(Path(__file__)),
                "calibration": str(args.calibration.resolve()), "calibration_sha256": sha256(args.calibration),
                "interpretation": "Correctness diagnostic only; calibration reused without performance claims"})
    shutil.copy2(Path(__file__), directory / "diagnostic_source.py")
    with (directory / "server.log").open("wb") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(origin, proc, str(directory / "server.log"), 1800)
            model_id = get_json(origin + "/v1/models")["data"][0]["id"]
            atomic_json(arm, {"armed": True, "observe": False, "capture_id": "numerics",
                             "request_id": row["request_id"], "prompt_tokens": len(prompt_ids),
                             "directory": str(directory / "request"),
                             "require_pinned_banks": True})
            body = {"model": model_id, "messages": [{"role": "user", "content": row["prompt_text"]}],
                    "max_tokens": 2, "stream": False, **plan["sampling"],
                    "chat_template_kwargs": plan["chat_template_kwargs"]}
            atomic_json(directory / "response.json", request(origin, body))
            if not (directory / "comparison.json").is_file():
                raise RuntimeError("Same-state diagnostic hook did not run")
            print((directory / "comparison.json").read_text(), flush=True)
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
