# Experiment 1 — Prefill MoE-prefetch overlap ON vs OFF (gpt-oss-20b, RTX 4070)

**Attribution:** Kushal; imported results reviewed against saved artifacts, not rerun during cleanup. Runner code is in `experiments/prefill_overlap/`; original data and figures are in `results/historical/prefill_overlap/`.

**Measured versus inferred:** the 32 recorded requests establish ON/OFF TTFT and sampled power/utilization. Transfer/compute decomposition is inferred from those pairs; the imported `.qdstrm` files were not a parsed kernel-timeline attribution. The original report's CPU/GPU utilization and power observations are not hardware memory-bandwidth or SM-occupancy counters.

Objective report. Same model/host/env/workload/cache config as `saketh_exps`; the only
variable is `moe_prefill_overlap` (ON = default, OFF = `--disable-moe-prefill-overlap`).

## Setup

gpt-oss-20b MXFP4, hybrid backend, cache-rate 0.20, **naive KV cache** (no prefix reuse),
`--no-graph`, greedy first token. Fresh uncached prefills (unique tag per request). Prompt
lengths 1024/2048/4096/7900 tokens, 4 repeats each, one fresh server per overlap setting,
warmed once. System monitor samples that server at 4 Hz. Host: RTX 4070, **PCIe gen4 x4**
(measured), B_P (PCIe expert gather) 6.7 GB/s, B_H (CPU MoE) 38 GB/s.

## Results (TTFT = prefill first-token latency; mean of 4, std < 5 ms)

| prompt tokens | TTFT ON (ms) | TTFT OFF (ms) | benefit (ms) | benefit % |
|--:|--:|--:|--:|--:|
| 1041 | 1710.2 | 1880.1 | 169.9 | 9.0 |
| 2032 | 1720.7 | 2029.8 | 309.1 | 15.2 |
| 4095 | 1762.2 | 2393.2 | 631.0 | 26.4 |
| 7899 | 1830.1 | 3176.8 | 1346.7 | 42.4 |

Utilization (median over the setting's prefills):

| | GPU util | GPU power median / **peak** | CPU util | PCIe |
|---|--:|--:|--:|--:|
| overlap ON | 100% | 91 W / **186 W** | 10% | gen4 x4 |
| overlap OFF | 100% | 89 W / **128 W** | 10% | gen4 x4 |

## Patterns observed (beyond "TTFT improves")

1. **TTFT with overlap ON is nearly flat across an 8x prompt-length increase** (1710 ->
   1830 ms), while OFF rises steeply (1880 -> 3177 ms). The overlap does not just shave a
   fixed cost; it makes prefill latency almost insensitive to prompt length in this range.

2. **The benefit grows with prompt length** (170 -> 1347 ms; 9% -> 42%). From the ON/OFF
   pair, `benefit(L) = OFF - ON = min(compute(L), transfer)` -- the overlap hides the
   smaller of prefill compute and expert transfer. Here ON stays flat while OFF rises, so
   `compute(L) < transfer` at every tested length: **prefill on this host is
   expert-transfer-bound**, and the ON floor (~1710 ms) approximates overhead + transfer.
   The overlap hides essentially all of the prefill compute (170 ms at 1K up to 1347 ms at
   8K) behind the transfer. This is why the benefit keeps growing: longer prompts have more
   compute to tuck under the (roughly constant) transfer.

3. **GPU peak power is markedly higher with overlap ON (186 W vs 128 W)** at equal 100%
   utilization. Running the H2D copy engine and the compute kernels concurrently draws more
   instantaneous power than serializing them -- a direct, if indirect, signature that the
   two are genuinely overlapping rather than time-slicing.

4. **Where overlap "fails to help": very short prompts.** At 1K tokens the benefit is only
   9% (170 ms) because there is little compute to hide the transfer behind -- the transfer
   is largely exposed. Overlap efficiency (fraction of the hidden branch actually hidden)
   is a function of `compute(L)/transfer`; it is low when compute is small and approaches
   full hiding only once compute grows toward the transfer time.

5. **CPU stays ~10% in both cases**: prefill is GPU-compute and PCIe-transfer bound, not
   host-bound; the overlap is a GPU-side copy-vs-compute concurrency, not a CPU offload.

## Exposed / stall interpretation

- OFF exposed time (relative to the ON floor, overhead+transfer): 1041->170, 2032->319,
  4095->683, 7899->1467 ms. This is the prefill compute that OFF pays serially and ON hides.
- ON exposed time: the small rise above the floor (0 -> ~120 ms) is the compute that begins
  to exceed the transfer at the longest prompt -- i.e. the point where prefill starts to
  transition from transfer-bound toward compute-bound. Extrapolating, overlap's benefit
  would saturate once `compute(L) > transfer` (beyond ~8K tokens on this host).

## Profiler note (nsys)

Nsight Systems captures of one ON and one OFF prefill window are saved
(`results/historical/prefill_overlap/results/nsys_overlap_on.qdstrm`, `nsys_overlap_off.qdstrm`, `nsys_on2.qdstrm`). This host
has the target-side collector only (no host importer), so `.qdstrm` cannot be converted to
`.nsys-rep`/stats here; the files open in Nsight Systems UI on a machine with the full
install. The transfer-vs-compute decomposition above is therefore derived from the ON/OFF
TTFT pair and corroborated by the power measurement, not read off a kernel timeline.

## Conclusion (objective)

On this host, MoE prefill-prefetch overlap turns a transfer-bound prefill from
`transfer + compute` into `max(transfer, compute)`. Because expert transfer dominates
prefill compute for prompts up to ~8K tokens (PCIe gen4 x4, 6.7 GB/s), the overlap keeps
TTFT nearly constant with prompt length and its benefit grows with the prompt (9% at 1K to
42% at 8K). The overlap helps least at short prompts (little compute to hide the transfer
behind) and would saturate once compute exceeds transfer. These are observations for this
one machine/model/workload and are not claimed as novel.

## Artifacts

`results/historical/prefill_overlap/results/prefill_ttft.jsonl` (raw), `results/historical/prefill_overlap/results/summary_prefill.csv`, `results/historical/prefill_overlap/results/summary_util.json`,
`results/historical/prefill_overlap/results/monitor_overlap_{on,off}.csv`, `results/historical/prefill_overlap/results/nsys_overlap_*.qdstrm`,
`results/historical/prefill_overlap/figures/prefill1_ttft_vs_length.png`, `prefill2_benefit.png`, `prefill3_gpu_power.png`.
Reproduction commands are below.


## Reproduction and artifact map

Execute commands from the repository root unless a `cd` is shown. Use `uv`; give reruns fresh output locations. Recorded results below are historical evidence.

Compares FreeToken prefill with `moe_prefill_overlap` ON vs OFF, holding model, host, env,
workload, and cache config fixed at the `saketh_exps` configuration (gpt-oss-20b MXFP4,
hybrid, cache-rate 0.20, naive KV cache). Only the overlap flag changes.

## Reproduce
```bash
source scripts/env.sh                     # CUDA 13 environment; export MODEL separately

# TTFT + utilization sweep (fresh server per setting, monitor per setting)
uv run --no-sync python experiments/prefill_overlap/run_prefill.py --gpu 0 --cache-rate 0.2 \
  --lengths 1024 2048 4096 7900 --repeats 4

# Analysis + plots
uv run --no-sync python experiments/prefill_overlap/analyze_prefill.py

# Optional nsys timeline captures (saved as .qdstrm; open in Nsight UI on a full install)
#   see run_prefill.log / the nsys commands used in the session
```

## Outputs
- `results/historical/prefill_overlap/results/prefill_ttft.jsonl` raw per-request TTFT (overlap, length, repeat, wall stamps)
- `results/historical/prefill_overlap/results/monitor_overlap_{on,off}.csv` GPU/CPU util, power, PCIe during each setting
- `results/historical/prefill_overlap/results/summary_prefill.csv`, `results/historical/prefill_overlap/results/summary_util.json`
- `results/historical/prefill_overlap/results/nsys_overlap_{on,off}.qdstrm` Nsight captures (target-only host: open elsewhere)
- `results/historical/prefill_overlap/figures/prefill1_ttft_vs_length.png`, `prefill2_benefit.png`, `prefill3_gpu_power.png`
- `report.md` interpretation

[Master experiment log](../../EXPERIMENT_LOG.md) · [Documentation index](../../README.md).
