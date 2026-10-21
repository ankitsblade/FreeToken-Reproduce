"""Decode routing transforms for fixed-logit screens (P6 resident bias, P7 masking/aliasing).

Selection may change; gate weights always come from the original unbiased logits on the
selected set. S = softmax over all E is used only as a comparison score.
"""

import numpy as np

PROTECT_J = {"gpt_oss_mxfp4": 1, "qwen_nvfp4": 2, "gemma_nvfp4": 2}


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def native_order(z):
    # Descending logits; equal logits keep lower expert ID first (stable).
    return np.argsort(-z, kind="stable")


def native_select(z, k):
    return native_order(z)[:k].tolist()


def native_weights(z, selected, scale=None):
    """Softmax over the selected original logits; Gemma then multiplies its per-expert scales."""
    values = softmax(np.asarray([z[e] for e in selected], dtype=np.float64))
    return values if scale is None else values * np.asarray([scale[e] for e in selected], dtype=np.float64)


def check_native(trace, geometry, limit=None):
    """Zero-change control: captured decode selections/weights versus logit reconstruction."""
    scores, routes, scales = trace.scores(geometry.experts), trace.routes(), trace.router_scales()
    set_mismatch, max_weight_error, checked = 0, 0.0, 0
    for i, (layer, position) in enumerate(zip(trace.layer.tolist(), trace.position.tolist())):
        if limit is not None and checked >= limit:
            break
        ids, weights = routes[(position, layer)]
        z = scores[i].astype(np.float64)
        selected = native_select(z, geometry.top_k)
        if set(selected) != set(ids):
            set_mismatch += 1
        mine = dict(zip(ids, native_weights(z, ids, scales.get(layer))))
        max_weight_error = max(max_weight_error, max(abs(mine[e] - w) for e, w in zip(ids, weights)))
        checked += 1
    return {"events": checked, "selection_set_mismatches": set_mismatch, "max_effective_weight_abs_error": max_weight_error}


class RangeTracker:
    """Causal per-layer mean of the original-logit range, seeded from calibration decode events."""

    def __init__(self, sums, counts):
        self.sums, self.counts = np.asarray(sums, dtype=np.float64).copy(), np.asarray(counts, dtype=np.int64).copy()

    @classmethod
    def calibrate(cls, traces, geometry):
        sums, counts = np.zeros(geometry.layers), np.zeros(geometry.layers, dtype=np.int64)
        for trace in traces:
            scores = trace.scores(geometry.experts)
            np.add.at(sums, trace.layer, scores.max(axis=1) - scores.min(axis=1))
            np.add.at(counts, trace.layer, 1)
        return cls(sums, counts)

    def mean(self, layer):
        return self.sums[layer] / self.counts[layer] if self.counts[layer] else 0.0

    def update(self, layer, z):
        self.sums[layer] += float(z.max() - z.min())
        self.counts[layer] += 1


def family_a(z, k, j, resident, lam, mean_range):
    """Cache-Prior-inspired bias; hard-include original top-J, fill k-J from biased scores."""
    order = native_order(z)
    top = order[:j].tolist()
    bonus = lam * mean_range * (resident.astype(np.float64))
    bonus[top] = lam * mean_range
    biased = z + bonus
    rest = [e for e in native_order(biased).tolist() if e not in top]
    return top + rest[:k - j]


def family_b(z, k, j, resident, alpha, start=None):
    """Low-score substitution of missing unprotected selections by resident candidates."""
    s = softmax(z)
    order = native_order(z)
    s_next = s[order[k]]
    protected = set(order[:j].tolist())
    selected = list(start) if start is not None else order[:k].tolist()
    chosen = set(selected)
    candidates = [c for c in order.tolist() if resident[c] and c not in chosen and s[c] >= (1 - alpha) * s_next]
    weakest = sorted([e for e in selected if not resident[e] and e not in protected and s[e] < (1 + alpha) * s_next],
                     key=lambda e: (s[e], -int(np.flatnonzero(order == e)[0])))
    swaps = 0
    for e in weakest:
        if not candidates:
            break
        c = candidates.pop(0)
        selected[selected.index(e)] = c
        swaps += 1
    return selected, swaps


def mass_retained(z, original, selected):
    s = softmax(z)
    base = s[list(original)].sum()
    return float(s[[e for e in selected if e in set(original)]].sum() / base) if base > 0 else 1.0


def masked_select(z, k, removed):
    """P7 decode exclusion: removed experts get minus infinity before native top-k."""
    masked = z.copy()
    masked[list(removed)] = -np.inf
    return native_select(masked, k)


def physical(selected, alias):
    """P7 aliasing: map logical IDs to physical cluster experts; co-occurring aliases evaluate once."""
    return list(dict.fromkeys(alias[e] for e in selected))
