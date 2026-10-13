"""Log the existing FreeToken startup memory calculation without changing its result."""
from __future__ import annotations

import runpy

from freetoken.engine import engine

_startup_budget = engine._startup_kv_budget


def logged_budget(memory_ratio, initial_free, current_free):
    import torch

    available = _startup_budget(memory_ratio, initial_free, current_free)
    print(f"GEMMA_MEMORY_BUDGET ratio={memory_ratio} initial_free={initial_free} "
          f"current_free={current_free} consumed={initial_free-current_free} kv_available={available} "
          f"torch_allocated={torch.cuda.memory_allocated()} torch_reserved={torch.cuda.memory_reserved()}",
          flush=True)
    return available


engine._startup_kv_budget = logged_budget


if __name__ == "__main__":
    runpy.run_module("freetoken.cli", run_name="__main__")
