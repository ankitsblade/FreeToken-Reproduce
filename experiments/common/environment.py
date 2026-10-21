"""Record machine and dependency identity without loading model weights."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE, engine_sources, python_command


def probe(*command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=30, cwd=ROOT)
        return {"command": list(command), "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "error": str(exc)}


def recorded_environment_value(key, value):
    # The engine's name contains TOKEN; inspect the setting name after its prefix.
    setting = key.removeprefix("FREETOKEN_").upper()
    return "<redacted>" if any(word in setting for word in
                               ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "AUTH")) else value


def collect(runtime=False):
    record = {"utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
              "machine": platform.machine(), "python": sys.version, "repo": str(ROOT),
              "engine_source": str(ENGINE_SOURCE), "environment": {k: recorded_environment_value(k, v) for k, v in os.environ.items()
                  if k.startswith(("FREETOKEN_", "OMP_", "MKL_", "CUDA_", "EXPERIMENT_"))},
              "gpu": probe("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free,driver_version", "--format=csv"),
              "compute_processes": probe("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"),
              "cpu": probe("lscpu", "--json"), "memory": probe("free", "-b"),
              "nvcc": probe("nvcc", "--version"), "git_head": probe("git", "rev-parse", "HEAD"),
              "git_status": probe("git", "status", "--short"), "packages": {}}
    record["package_inventory"] = probe("uv", "pip", "freeze", "--python", sys.executable)
    for name in ("numpy", "torch", "transformers", "freetoken", "flashlib", "triton", "sglang-kernel", "pandas"):
        try:
            record["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            record["packages"][name] = None
    if runtime:
        # A separate process releases its CUDA context before the idle-GPU gate.
        code = ("import json,torch,freetoken; from pathlib import Path; "
                "from freetoken.kernel import _cpu_moe; "
                "print(json.dumps({'import_path':str(Path(freetoken.__file__).resolve()),"
                "'cpu_extension':str(Path(_cpu_moe.__file__).resolve()),"
                "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
                "'devices':[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i)} "
                "for i in range(torch.cuda.device_count())]}))")
        runtime_probe = probe(*python_command(), "-c", code)
        if runtime_probe.get("returncode") != 0:
            raise RuntimeError(f"Runtime import failed: {runtime_probe}")
        record["runtime"] = json.loads(runtime_probe["stdout"])
        record["engine_hashes"] = {}
        for name, p in engine_sources().items():
            record["engine_hashes"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        if not record["runtime"]["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; inference cannot start")
        if not Path(record["runtime"]["import_path"]).is_relative_to(ENGINE_SOURCE):
            raise RuntimeError("FreeToken imported from an unexpected source directory")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--require-runtime", action="store_true")
    a = p.parse_args()
    record = collect(a.require_runtime)
    text = json.dumps(record, indent=2) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        print(f"Environment recorded: {a.out}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
