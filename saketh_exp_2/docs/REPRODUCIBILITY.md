# Reproducibility notes

## Identity

- **Runtime:** `.runtime/freetoken` = commit `4d0f84983700eba7dfb046fb4df3cbbcc488ecc2` (its `python/` and `benchmarks/` are identical to upstream FlashML-org `cab110e`) + `runtime/patches/experiment-hooks.patch` (sha256 `5db728e38ce6faec…`). Verified by patch-apply and recursive diff.
- **Repository HEAD at launch:** recorded in `results/logs/<stamp>/queue.log`. Input file hashes are in `input_sha256.txt`.
- **Hardware:** RTX 4070 12 GB (PCIe 4.0 x16, driver 595.84, CUDA 13.2), Ryzen 5 8500G (6 cores / 12 threads), 30 GiB RAM.
- **Checkpoints** (local HF metadata matches the pins in `configs/pruning/models.json`):
  - `nvidia/Qwen3.6-35B-A3B-NVFP4@1355db6a052410cfd62085d94b58866fd0f2c3c5` (paper model)
  - `openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee` (non-paper)
  - `nvidia/Gemma-4-26B-A4B-NVFP4@a19cfe00be84568a6867111c9a68c9c44fdcffe6` (non-paper)
- **Bandwidth profile:** `results/reuse_aware_models/study_20261020T071921Z/calibration/benchbw.json` (2026-10-20, same host), the established calibration of the policy study.

## Commands

```bash
.venv/bin/python saketh_exp_2/config/make_plans.py
bash -c 'source scripts/env.sh && experiment_python -m pytest saketh_exp_2/tests -q -p no:cacheprovider'
setsid bash saketh_exp_2/scripts/launch.sh < /dev/null > /dev/null 2>&1 &
```

Each step the launcher runs:
- **Hooked and unhooked-eager arms:** `EXPERIMENT_OUTPUT_ROOT=<raw> systemd-run --user --scope -p MemorySwapMax=0 bash scripts/run.sh runtime_q run --plan saketh_exp_2/config/plans/<smoke|main>_<model>.json --model <ckpt> --gpu 0 --decode <32|256>`
- **References:** `scripts/run.sh baselines bench_decode_moe --backend hybrid --hybrid-fetch -1 --problem p --decode 256 --greedy --aime saketh_exp_2/config/prompts_aime25.jsonl --cache-type naive [--no-graph] <serve flags>`, with the same env. Each is a fresh server.
- **Analysis:** `experiment_python saketh_exp_2/analysis/analyze.py --root <raw> --out <aggregated>`

## Deviations from the paper and the official benchmark (all recorded, held constant across arms)

| Setting | Here | Paper / official | Reason |
|---|---|---|---|
| Decode mode | eager, host hook with per-layer syncs | CUDA graph, in-kernel q | Host policies need the routing on the host; the cost is measured with unhooked eager and graph references |
| Precision | Qwen NVFP4 | BF16 (NVFP4 on the 8 GB laptop) | BF16 pool does not fit 30 GiB RAM |
| Models | + gpt-oss, Gemma | Qwen3.6, DSV4-Flash, GLM-5.2 | User decision; labelled non-paper |
| Problems | AIME #0–2 | official bench: #0 | Not relying on one prompt |
| Sampling | greedy | checkpoint sampling (greedy is an upstream option) | Paired determinism |
| Warm-up | cold reset → warm generate → measure, 3 burn-in episodes per server | fresh server + 1 identical warm request | Many paired arms per server (established driver) |
| Warm-up / measure prompts | different leading tags | identical | Avoids full prefix reuse (local benchmark change) |
| KV cache | naive | engine default | Established harness |
| CPU threads | 5 | paper 6 | Established profile on this 6-core host |
| Memory ratio / max seq | 0.9/2048 (Gemma 0.8/1280) | 0.9 / 8192 + D | Established per-model limits |
| Scheduler overlap | disabled (`FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1`) | enabled | Established harness env, also used for the references |
| TZ | Etc/UTC | local | GPT-OSS chat template embeds the date |
| Repetitions | 5 timed + 1 logged round, rotated order | not specified | User decision |
