"""Shared policy -> q(m) mapping, mirroring FreeToken's kernel rounding exactly.

The hybrid kernel turns a fetch fraction f into an integer q per step by picking whichever
of floor(f*m) / floor(f*m)+1 makes the PCIe fetch and CPU overflow branches finish closest
together (offload_kernels.py). We reproduce that here so the analysis labels each policy's
operating point with the same q the runtime actually used.
"""

from __future__ import annotations

Q16 = 1 << 16

POLICY_FRACTION = {
    "D_fixed25": 0.25, "E_fixed50": 0.50, "F_fixed75": 0.75,
}


def round_balance(fraction: float, m: int) -> int:
    if m <= 0:
        return 0
    frac_q16 = min(Q16, max(0, round(fraction * Q16)))
    lo = (m * frac_q16) >> 16
    def cost(f):
        return max(f * (Q16 - frac_q16), (m - f) * frac_q16)
    q = lo if cost(lo) <= cost(lo + 1) else lo + 1
    return max(0, min(m, q))


def policy_q(policy: str, m: int, adaptive_fraction: float) -> int:
    if policy == "C_cpu":
        return 0
    if policy == "B_gpu":
        return m
    if policy == "A_adaptive":
        return round_balance(adaptive_fraction, m)
    return round_balance(POLICY_FRACTION[policy], m)
