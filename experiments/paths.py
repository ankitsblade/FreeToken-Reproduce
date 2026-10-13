"""Portable locations shared by collectors and offline analyses."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SOURCE = Path(os.environ.get("FREETOKEN_SOURCE_DIR", ROOT / ".runtime/freetoken")).expanduser().resolve()


def study_dir(name: str) -> Path:
    base = Path(os.environ.get("EXPERIMENT_OUTPUT_ROOT", ROOT / "results/historical")).expanduser().resolve()
    return base / name


def runtime_pythonpath() -> str:
    return os.pathsep.join([str(ENGINE_SOURCE / "python"), str(ROOT), os.environ.get("PYTHONPATH", "")])


def python_command() -> list[str]:
    return ["uv", "run", "--no-sync", "--project", str(ROOT), "python"]


def engine_sources() -> dict[str, Path]:
    base = ENGINE_SOURCE / "python/freetoken"
    if not base.is_dir():
        raise FileNotFoundError("FreeToken source is missing; run bash scripts/setup.sh")
    suffixes = {".py", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".so"}
    return {"runtime_source/" + str(p.relative_to(ENGINE_SOURCE)): p
            for p in sorted(base.rglob("*")) if p.is_file() and p.suffix in suffixes}
