"""Untraced native pruning entry point with a cold expert cache for each request."""
from __future__ import annotations

import runpy

from experiments.pruning import native_pruning_probe as native
from freetoken.engine.engine import Engine
from freetoken.server import launch

_engine_forward = Engine.forward_batch
_uid = None


def forward_batch(self, batch, args):
    global _uid
    if len(batch.reqs) == 1 and batch.reqs[0].uid >= 0 and batch.reqs[0].uid != _uid:
        _uid = batch.reqs[0].uid
        cache = self.moe_offload_cache
        if cache is None or cache.decode_target != "hybrid" or any(label != "pinned" for label in cache.layer_residency):
            raise ValueError("serving diagnostic requires pinned hybrid expert banks")
        cache.reset()
    return _engine_forward(self, batch, args)


def run_scheduler(args, ack_queue):
    native.install()
    Engine.forward_batch = forward_batch
    print("PERFORMANCE_PRUNING_PROBE installed; routing/logit observation disabled", flush=True)
    native._run_scheduler(args, ack_queue)


if __name__ == "__main__":
    from experiments.pruning.performance_pruning_probe import run_scheduler as worker

    launch._run_scheduler = worker
    runpy.run_module("freetoken.cli", run_name="__main__")
