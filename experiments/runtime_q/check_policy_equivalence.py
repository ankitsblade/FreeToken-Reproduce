"""CPU-only consistency check of the policy implementations (no GPU, no server).

1. reuse_aware_rv with every feature group off must decide exactly like the original ReuseAware
   (Policy B) on randomized lean features.
2. fraction_min1 must equal FreeToken's balance rounding of fraction * m, raised to 1 when m > 0.
Writes a JSON summary next to --out (or prints it).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "instrumentation"))
from reuse_aware_rv_hook import ReuseRV  # noqa: E402
from reuse_aware_v2_hook import FractionMin1  # noqa: E402
from runtime_q_hook import ReuseAware, round_balance  # noqa: E402

MODELS = {"gpt_oss": (4, 32, 6.7, 31.76, 13236480), "qwen": (8, 256, 6.71, 31.38, 1775616),
          "gemma": (8, 128, 6.71, 36.83, 3353856)}


def features(rng: random.Random, k: int, E: int, bp: float, bh: float, S: int) -> tuple[int, dict]:
    m = rng.randint(0, k)
    n = float(rng.randint(0, 32))
    tx, tc = S / (bp * 1e6), S / (bh * 1e6)
    x = {"n_hits": float(k - m), "_window": n, "_miss_f32": [rng.randint(0, int(n)) for _ in range(m)],
         "free_slots": float(rng.choice([0, 0, 1, 2, 5, 40])), "lru_age_tok": rng.uniform(0.5, 60),
         "_victim_p": [rng.random() * 0.3 for _ in range(m)], "_victims": [],
         "_rv_fit": {"tx": rng.uniform(0.5, 3), "a": rng.random(), "b": rng.random() * 0.1,
                     "c0": rng.random() * 0.1, "tc": rng.uniform(0.1, 1)},
         "_rv_prior": {"tx": tx, "tc": tc}, "_mcount": [rng.randint(0, 50) for _ in range(k + 1)],
         "_rv_ins_rate": rng.uniform(0, 10), "_rv_res_age": rng.uniform(1, 100)}
    prior = k / E
    x["_p_miss"] = [(f + prior * 8) / (n + 8) for f in x["_miss_f32"]]
    x["_p_hist"] = [rng.random() for _ in range(m)]
    x["_pv_hist"] = [rng.random() * 0.3 for _ in range(m)]
    return m, x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()
    rng = random.Random(0)
    summary = {}
    off = {"exec": False, "xfer": False, "cache": False, "hist": False}
    for name, (k, E, bp, bh, S) in MODELS.items():
        orig = ReuseAware({"kind": "reuse_aware", "kappa": 1.0}, bp / bh, k, E)
        rv = ReuseRV(off, k, E)
        full = ReuseRV({}, k, E)
        mism, differs = 0, 0
        for _ in range(a.n):
            m, x = features(rng, k, E, bp, bh, S)
            qo, qr = orig.decide(m, x, 0), rv.decide(m, x, 0)
            mism += qo != qr
            differs += full.decide(m, x, 0) != qo
        f = bp / bh
        mn = [(m, FractionMin1(f).decide(m, {}, 0), round_balance(f, m)) for m in range(k + 1)]
        ok_min1 = all(q == (max(1, r) if m else 0) for m, q, r in mn)
        summary[name] = {"cases": a.n, "rv_all_off_vs_original_mismatches": mism,
                         "rv_full_differs_from_original": differs, "fraction": round(f, 6),
                         "min1_table_m_q_minq_vs_engine_q": mn, "min1_ok": ok_min1}
    text = json.dumps(summary, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    assert all(v["rv_all_off_vs_original_mismatches"] == 0 and v["min1_ok"] for v in summary.values())


if __name__ == "__main__":
    main()
