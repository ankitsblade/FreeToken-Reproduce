"""Isolated FreeToken entry point for native Gemma scores and opt-in router masks."""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import torch

from experiments.pruning.moe_pruning import atomic_json
from freetoken.models.gemma4.moe import Gemma4MLP, Gemma4Router
from freetoken.server import launch

_mlp_init = Gemma4MLP.__init__
_router_forward = Gemma4Router.forward
_metadata_capture = None
_metadata_layers = {}
_mask_path = os.getenv("FREETOKEN_PRUNING_MASK")
_mask = None
_run_scheduler = launch._run_scheduler


def eligibility(layer_id, logits, top_k):
    global _mask
    if not _mask_path:
        return logits
    if _mask is None:
        import numpy as np

        values = np.load(_mask_path, allow_pickle=False)["mask"]
        if values.dtype != np.bool_ or values.ndim != 2 or (values.sum(axis=1) < top_k).any():
            raise ValueError("invalid expert eligibility mask")
        _mask = torch.from_numpy(values.copy()).to(logits.device)
    if layer_id is None or not 0 <= layer_id < len(_mask) or _mask.shape[1] != logits.shape[1]:
        raise ValueError("mask geometry does not match native router")
    return logits.masked_fill(~_mask[layer_id], -float("inf"))


def gemma_init(self, config, layer_id, *, prefix=""):
    _mlp_init(self, config, layer_id, prefix=prefix)
    self.router.pruning_layer_id = layer_id


def gemma_forward(self, x):
    global _metadata_capture, _metadata_layers
    if not os.getenv("FREETOKEN_MOE_ROUTING_ARM") and not _mask_path:
        return _router_forward(self, x)
    h = self.norm.forward(x)
    h = h * self.scale * self._scalar_root
    logits = self.proj.forward(h)
    layer_id = self.pruning_layer_id
    if os.getenv("FREETOKEN_MOE_ROUTING_ARM"):
        from freetoken.moe.routing_trace import observer

        trace = observer()
        if trace.batch is not None and trace.metadata.get("full_scores"):
            trace.observe_scores(layer_id, logits)
            if _metadata_capture != trace.capture_id:
                _metadata_capture, _metadata_layers = trace.capture_id, {}
            if str(layer_id) not in _metadata_layers:
                _metadata_layers[str(layer_id)] = {
                    "logit_dtype": str(logits.dtype), "scale_dtype": str(self.per_expert_scale.dtype),
                    "per_expert_scale": self.per_expert_scale.float().cpu().tolist(),
                    "semantics": "native selected-top-k softmax times learned per-expert scale",
                }
                atomic_json(Path(trace.metadata["directory"]) / "router_metadata.json", _metadata_layers)
    logits = eligibility(layer_id, logits, self._top_k)
    return self.gemma4_fused_routing(logits, self.per_expert_scale, self._top_k)


def install():
    Gemma4MLP.__init__ = gemma_init
    Gemma4Router.forward = gemma_forward
    if _mask_path:
        from freetoken.layers.moe import OffloadMoELayer
        from freetoken.models.gpt_oss.moe import GptOssOffloadMoELayer

        generic_forward = OffloadMoELayer.forward
        gpt_forward = GptOssOffloadMoELayer.forward

        def generic(self, hidden_states, router_logits=None):
            return generic_forward(self, hidden_states, eligibility(self.layer_id, router_logits, self.top_k))

        def gpt(self, hidden_states, router_logits=None):
            return gpt_forward(self, hidden_states, eligibility(self.layer_id, router_logits, self.top_k))

        OffloadMoELayer.forward = generic
        GptOssOffloadMoELayer.forward = gpt


def run_scheduler(args, ack_queue):
    install()
    from experiments.pruning import gemma_memory_probe

    print("NATIVE_PRUNING_PROBE installed in scheduler worker", flush=True)
    _run_scheduler(args, ack_queue)


if __name__ == "__main__":
    from experiments.pruning.native_pruning_probe import run_scheduler as worker

    launch._run_scheduler = worker
    runpy.run_module("freetoken.cli", run_name="__main__")
