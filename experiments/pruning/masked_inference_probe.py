"""Native router masking and same-input quality measurements in the FreeToken worker."""
from __future__ import annotations

import json
import os
import runpy
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from experiments.pruning import native_pruning_probe as native
from experiments.pruning.moe_pruning import atomic_json
from freetoken.core import get_global_ctx
from freetoken.engine.engine import Engine
from freetoken.engine.sample import Sampler
from freetoken.layers.embedding import ParallelLMHead
from freetoken.layers.moe import OffloadMoELayer
from freetoken.models.gemma4.model import Gemma4DecoderLayer
from freetoken.models.gpt_oss.model import GptOssDecoderLayer
from freetoken.models.qwen3_5_moe.model import Qwen3_5DecoderLayer
from freetoken.server import launch

_control = Path(os.environ["FREETOKEN_PRUNING_CONTROL"])
_active = None
_uid = None
_audit = {}
_layer_metrics = {}
_cap = None
_sampled_tokens = []
_baseline_directory = None
_baseline_layers = {}
_baseline_logits = None
_cpu_layer_buffer = None
_cpu_logit_buffer = None
_cpu_header_buffer = None
_engine_forward = Engine.forward_batch
_head_forward = ParallelLMHead.forward
_prefill_routed = OffloadMoELayer._prefill_routed
_decode_routed = OffloadMoELayer._decode_routed
_sample = Sampler.sample


def engine_forward(self, batch, args):
    global _active, _uid, _audit, _layer_metrics, _cap, _sampled_tokens
    global _baseline_directory, _baseline_layers, _baseline_logits
    if len(batch.reqs) == 1 and batch.reqs[0].uid >= 0 and _control.exists():
        uid = batch.reqs[0].uid
        if uid != _uid:
            metadata = json.loads(_control.read_text())
            _active = metadata if metadata.get("armed") else None
            _uid, _audit, _layer_metrics, _sampled_tokens = uid, {}, {}, []
            native._mask_path = _active.get("mask") if _active else None
            native._mask = None
            _cap = getattr(self.model, "_final_logit_softcapping", None)
            requested_baseline = _active.get("baseline_directory") if _active else None
            if requested_baseline != _baseline_directory:
                _baseline_directory, _baseline_layers, _baseline_logits = requested_baseline, {}, None
            if self.moe_offload_cache is not None:
                if _active and _active.get("require_pinned_banks"):
                    cache = self.moe_offload_cache
                    if cache.decode_target != "hybrid" or len(cache.layer_residency) != cache.num_layers or any(label != "pinned" for label in cache.layer_residency):
                        raise ValueError("actual inference requires qualified pinned hybrid expert banks")
                self.moe_offload_cache.reset()
    if _active and _active.get("teacher_forcing") and batch.is_prefill:
        if batch.input_ids.cpu().tolist()!=_active["input_ids"] or batch.positions.cpu().tolist()!=list(range(_active["total_input_tokens"])):
            raise ValueError("native teacher-forcing inputs differ from frozen supported text tokenization")
    result = _engine_forward(self, batch, args)
    if _active and _active.get("audit_routes"):
        atomic_json(Path(_active["directory"]) / "native_route_audit.json", _audit)
        atomic_json(Path(_active["directory"]) / "sampled_token_ids.json", _sampled_tokens)
    return result


def audit_routes(self, topk_weights, topk_ids):
    if _active and _active.get("audit_routes"):
        if topk_ids.shape[1] != self.top_k or not torch.isfinite(topk_weights).all():
            raise ValueError("masked native route has wrong k or nonfinite weights")
        if native._mask is not None and not native._mask[self.layer_id, topk_ids.long()].all():
            raise ValueError("actual inference selected an excluded expert")
        key = str(self.layer_id)
        _audit[key] = _audit.get(key, 0) + len(topk_ids)


def prefill_routed(self, hidden_states, topk_weights, topk_ids):
    audit_routes(self, topk_weights, topk_ids)
    return _prefill_routed(self, hidden_states, topk_weights, topk_ids)


def decode_routed(self, hidden_states, topk_weights, topk_ids):
    audit_routes(self, topk_weights, topk_ids)
    return _decode_routed(self, hidden_states, topk_weights, topk_ids)


def sample(self, logits, args):
    result = _sample(self, logits, args)
    if _active and _active.get("audit_routes"):
        _sampled_tokens.extend(result.cpu().tolist())
    return result


def selected_hidden(x):
    start = _active["prefix_tokens"] - 1
    count = len(_active["target_ids"])
    if len(x) != _active["total_input_tokens"]:
        raise ValueError("teacher-forced diagnostics require one complete prefill chunk")
    return x[start:start+count].float()


def aligned_pinned(shape, dtype=torch.float32):
    element_size = torch.empty((), dtype=dtype).element_size()
    count = int(np.prod(shape))
    storage = torch.empty(count + 4096 // element_size, dtype=dtype, device="cpu", pin_memory=True)
    offset = ((-storage.data_ptr()) % 4096) // element_size
    return storage[offset:offset + count].view(shape)


def write_aligned(fd, tensor):
    if tensor.data_ptr() % 4096 or tensor.numel() * tensor.element_size() % 4096 or not tensor.is_contiguous():
        raise ValueError("direct artifact writes require aligned contiguous buffers")
    data = memoryview(tensor.numpy()).cast("B")
    while data:
        written = os.write(fd, data)
        if written <= 0 or written % 4096:
            raise ValueError("incomplete aligned artifact write")
        data = data[written:]


def open_direct_array(path, shape):
    global _cpu_header_buffer
    header = repr(dict(descr="<f4", fortran_order=False, shape=tuple(shape))).encode("latin1")
    header_size = 4096 - 12
    if len(header) >= header_size:
        raise ValueError("array header exceeds direct-write alignment")
    data = np.lib.format.magic(2, 0) + struct.pack("<I", header_size) + header + b" " * (header_size - len(header) - 1) + b"\n"
    if _cpu_header_buffer is None:
        _cpu_header_buffer = aligned_pinned((4096,), torch.uint8)
    np.copyto(_cpu_header_buffer.numpy(), np.frombuffer(data, dtype=np.uint8))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o600)
    try:
        write_aligned(fd, _cpu_header_buffer)
    except BaseException:
        os.close(fd)
        raise
    return fd


def layer_wrapper(original):
    def forward(self, *args, **kwargs):
        global _cpu_layer_buffer
        result = original(self, *args, **kwargs)
        if _active and _active.get("teacher_forcing") and get_global_ctx().batch.is_prefill:
            x = (result[0].float() + result[1].float()) if isinstance(result, tuple) else result.float()
            current = selected_hidden(x)
            layer_id = self._layer_id
            baseline_dir = Path(_active["baseline_directory"])
            baseline_dir.mkdir(parents=True, exist_ok=True)
            source = baseline_dir / f"layer_{layer_id:02d}.npy"
            if _active["arm"] == "baseline":
                _baseline_layers[layer_id] = current.clone()
                if _cpu_layer_buffer is None or _cpu_layer_buffer.shape != current.shape:
                    _cpu_layer_buffer = aligned_pinned(current.shape)
                _cpu_layer_buffer.copy_(current)
                fd = open_direct_array(source, current.shape)
                try:
                    write_aligned(fd, _cpu_layer_buffer)
                finally:
                    os.close(fd)
                _layer_metrics[str(layer_id)] = {"relative_l2": 0., "cosine": 1., "zero_reference_norm": False}
            else:
                reference = _baseline_layers[layer_id]
                norm = torch.linalg.vector_norm(reference)
                difference = torch.linalg.vector_norm(current-reference)
                current_norm = torch.linalg.vector_norm(current)
                reference_zero, current_zero = float(norm) == 0., float(current_norm) == 0.
                relative = float(difference/norm) if not reference_zero else (0. if float(difference) == 0. else None)
                cosine = float(F.cosine_similarity(current.flatten(), reference.flatten(), dim=0)) if not (reference_zero or current_zero) else (1. if reference_zero and current_zero else None)
                _layer_metrics[str(layer_id)] = {"relative_l2": relative, "cosine": cosine, "zero_reference_norm": reference_zero}
        return result
    return forward


def head_forward(self, x):
    global _baseline_logits, _cpu_logit_buffer
    if not (_active and _active.get("teacher_forcing") and get_global_ctx().batch.is_prefill):
        return _head_forward(self, x)
    if self.tp_size != 1:
        raise ValueError("same-input logit diagnostics require the qualified TP=1 arm")
    hidden = selected_hidden(x).to(x.dtype)
    count = len(hidden)
    path = Path(_active["baseline_directory"]) / "logits.npy"
    reference = None
    if _active["arm"] == "baseline":
        _baseline_logits = torch.empty((count, self.num_embeddings), dtype=torch.float32, device=x.device)
        reference = open_direct_array(path, (count, self.num_embeddings))
        if _cpu_logit_buffer is None:
            _cpu_logit_buffer = aligned_pinned((16, self.num_embeddings))
    elif _baseline_logits is None:
        raise ValueError("same-input diagnostics require the matching GPU-resident baseline")
    targets = torch.tensor(_active["target_ids"], device=x.device, dtype=torch.long)
    nll, kl, agreement, overlap = [], [], [], []
    for start in range(0, count, 16):
        current = hidden[start:start+16].contiguous()
        logits = (F.linear(current, self.tied_embedding.weight, self.bias) if self.tied_embedding is not None
                  else self.quant_method.apply(self, current))
        if _cap is not None:
            logits = torch.tanh(logits / _cap) * _cap
        logits = logits.float()
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite same-input vocabulary logits")
        logp = F.log_softmax(logits, dim=-1)
        stop = start + len(logits)
        nll.extend((-logp.gather(1, targets[start:stop,None]).flatten()).cpu().tolist())
        if _active["arm"] == "baseline":
            _baseline_logits[start:stop].copy_(logits)
            _cpu_logit_buffer[:len(logits)].copy_(logits)
            write_aligned(reference, _cpu_logit_buffer[:len(logits)])
            kl.extend([0.]*len(logits))
            agreement.extend([1.]*len(logits))
            overlap.extend([1.]*len(logits))
        else:
            baseline = _baseline_logits[start:stop]
            logq = F.log_softmax(baseline, dim=-1)
            kl.extend((logq.exp()*(logq-logp)).sum(dim=-1).cpu().tolist())
            agreement.extend((baseline.argmax(dim=-1) == logits.argmax(dim=-1)).float().cpu().tolist())
            top_current = logits.topk(10, dim=-1).indices
            top_base = baseline.topk(10, dim=-1).indices
            overlap.extend((top_current[:,:,None] == top_base[:,None,:]).any(dim=-1).float().mean(dim=-1).cpu().tolist())
    if _active["arm"] == "baseline":
        os.close(reference)
    del reference
    atomic_json(Path(_active["directory"]) / "teacher_forcing.json", dict(
        arm=_active["arm"], scored_tokens=count, target_ids=_active["target_ids"], nll=nll,
        mean_nll=float(np.mean(nll)), kl_baseline_to_masked=kl, mean_kl=float(np.mean(kl)),
        top1_agreement=float(np.mean(agreement)), top10_overlap=float(np.mean(overlap)),
        layer_metrics=_layer_metrics, logit_softcap=_cap,
        semantics="same fixed causal input IDs; native prefill MoE; vocabulary head evaluated in identical 16-row chunks",
        reference_storage="lossless FP32 GPU baseline cache; baseline files written with reusable pinned buffers",
        artifact_io="Linux O_DIRECT, aligned pinned buffers, standard NPY v2 headers padded to 4096 bytes",
        timing_interpretation="instrumented quality diagnostic, not serving throughput"))
    return _head_forward(self, x)


def run_scheduler(args, ack_queue):
    native.install()
    Engine.forward_batch = engine_forward
    OffloadMoELayer._prefill_routed = prefill_routed
    OffloadMoELayer._decode_routed = decode_routed
    Sampler.sample = sample
    ParallelLMHead.forward = head_forward
    for cls in (Gemma4DecoderLayer, GptOssDecoderLayer, Qwen3_5DecoderLayer):
        cls.forward = layer_wrapper(cls.forward)
    print("MASKED_INFERENCE_PROBE installed in scheduler worker", flush=True)
    native._run_scheduler(args, ack_queue)


if __name__ == "__main__":
    from experiments.pruning.masked_inference_probe import run_scheduler as worker

    launch._run_scheduler = worker
    runpy.run_module("freetoken.cli", run_name="__main__")
