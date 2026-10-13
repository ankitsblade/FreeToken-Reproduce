"""Restore the pinned dependency outside the experiment source tree."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from experiments.paths import ROOT, ENGINE_SOURCE


def restore():
    record = json.loads((ROOT / "runtime/source.json").read_text())
    patch = ROOT / record["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != record["patch_sha256"]:
        raise ValueError("Runtime patch changed; update and qualify its source identity first")
    if ENGINE_SOURCE.exists():
        raise FileExistsError(f"Refusing to overwrite {ENGINE_SOURCE}")
    ENGINE_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ENGINE_SOURCE.parent) as tmp:
        tmp = Path(tmp)
        archive = tmp / "source.tar"
        subprocess.run(["git", "archive", "--format=tar", "-o", str(archive), record["base_commit"]], cwd=ROOT, check=True)
        staging = tmp / "source"
        staging.mkdir()
        with tarfile.open(archive) as tf:
            tf.extractall(staging, filter="data")
        subprocess.run(["git", "init", "-q", str(staging)], check=True)
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=staging, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=staging, check=True)
        shutil.rmtree(staging / ".git")
        shutil.move(staging, ENGINE_SOURCE)
    print(f"Restored pinned source at {ENGINE_SOURCE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["restore"])
    parser.parse_args()
    restore()
