# Qwen vs GPT-OSS bandwidth study reproduction

The study follows the archived original bandwidth plan and the FreeToken paper.
Its supported finite set is complete; subsequent work is in `docs/PLAN.md`. Run all model comparisons on one machine and one FreeToken
commit. Keep each result directory for one machine. `results/` is ignored by Git.

Final phase dispositions and limits are recorded in `docs/experiments/bandwidth/report.md`.
Commands below are reproduction recipes, not a request to relaunch or overwrite completed runs.
Run them from the repository root, with GPU ownership checked first.

## Setup and system record

The following installation recipe describes a fresh/restored environment. This workspace
already has the recorded environment and checkpoints; reading or analysing the results
does not require recreating them.

```bash
source scripts/env.sh
uv venv
source .venv/bin/activate
uv pip install --index-url https://pypi.org/simple 'torch==2.11.0'
uv pip install --index-url https://pypi.org/simple \
  'nvidia-cuda-nvcc==13.0.88' 'nvidia-nvvm==13.0.88' 'nvidia-cuda-crt==13.0.88'
uv pip install ninja
export CUDA_HOME="$(uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')"
test -e "$CUDA_HOME/lib/libcudart.so" || ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CC=/usr/bin/gcc
uv pip install -e ".[accel]"
uv run --no-sync python experiments/bandwidth/bandwidth_study.py system \
  --model-storage local-nvme \
  --checkpoint-revision nvidia/Qwen3.6-35B-A3B-NVFP4:1355db6a052410cfd62085d94b58866fd0f2c3c5 \
  --checkpoint-revision openai/gpt-oss-20b:6cee5e81ee83917806bbde320786a8fb61efebee
```

On the measured RTX 4070 host, the system `nvcc` remains CUDA 12.0; the
workspace-local compiler is CUDA 13.0.88. Pin `nvidia-nvvm` and
`nvidia-cuda-crt` as shown: an unconstrained install otherwise selects newer
13.4 components. The PyTorch CUDA 13 runtime wheel supplies only the versioned
`libcudart.so.13`; the local development symlink ensures FreeToken's `-lcudart`
link does not silently select the system CUDA 12 runtime. The CUDA 13 smoke
kernel ran on this GPU and the C++ extensions linked against `libcudart.so.13`.
`CC` also pins the host compiler for Triton's CUDA JIT lookup. These shell
exports must be set again for a new shell.

The pinned checkpoint downloads for this machine are:

```bash
source scripts/env.sh
HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_UPDATE_CHECK=1 \
  .venv/bin/hf download nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --revision 1355db6a052410cfd62085d94b58866fd0f2c3c5 \
  --local-dir results/checkpoints/qwen3.6-35b-nvfp4 --max-workers 3

HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_UPDATE_CHECK=1 \
  .venv/bin/hf download openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --include 'model-*.safetensors' --include '*.json' \
  --include 'tokenizer*' --include 'chat_template.jinja' \
  --local-dir results/checkpoints/gpt-oss-20b --max-workers 3
```

The GPT-OSS patterns exclude unrelated repository variants. The pinned Qwen
download is about 23.5 GB; the selected GPT-OSS files total about 13.8 GB.
The direct non-Xet path above completed both downloads on this host. The earlier
`HF_XET_HIGH_PERFORMANCE=1` retry stalled, so keep `HF_HUB_DISABLE_XET=1` for
this connection. Reuse the same local directories and pinned revisions when
re-running the command.

The system JSON includes the raw `nvidia-smi -q` output. PCIe generation can drop
while idle, so verify negotiated generation and width again during a transfer run.
Record the exact checkpoint revisions and storage type used for model benchmarks.

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py link-probe
```

## Phase 0: calibration and end-to-end baseline

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py calibration \
  --repeat 5 --out results/baseline/calibration_repeat5.csv
```

This writes per-run FreeToken calibration profiles plus
`results/baseline/calibration_repeat5.csv` and
`calibration_repeat5_summary.csv`. The combined profile for the last repetition is
`results/baseline/calibration_repeat5/profile-run-04.json`. Check that both CPU and PCIe
measurements succeeded before using it. The profile is synthetic but uses the
Qwen NVFP4 and GPT-OSS MXFP4 bank geometries and production kernels. This
precision change from the historical Qwen BF16 setup must be reported with
every comparison.

For each checkpoint, run a separate baseline output file. Start with the
one-repeat 64-token smoke matrix on a 12 GB GPU:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py baseline \
  --model /path/to/qwen-checkpoint --decode 64 --repeat 1 \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/baseline/qwen_decode.jsonl

uv run --no-sync python experiments/bandwidth/bandwidth_study.py baseline \
  --model /path/to/gpt-oss-checkpoint --decode 64 --repeat 1 \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/baseline/gpt_oss_decode.jsonl
```

Each cell starts a fresh server. The driver runs auto, offload, and hybrid at two
relative cache capacities, with prefill overlap on and off. Warmup and measured
requests use different leading tags to avoid a full prompt prefix hit. The JSONL
contains decode tokens/s, per-event latency quantiles, TTFT, requested cache rate,
backend, and overlap setting. A cell that cannot reserve the minimum KV pool is
recorded as `status=error` and the matrix continues. TTFT is end-to-end; it is
not a separate prefill throughput measurement. GPT-OSS can still reuse a shared
chat-template prefix (68 tokens were cached in a diagnostic run); use the naive
KV-cache prefill benchmark below for prompt-length comparisons. For a valid
baseline, also record exact cache occupancy
and miss rate using the trace run below, and check stable measurements before
making any causal claim.
The default decode request uses checkpoint-recommended sampling. The summary
records distinct output hashes per cell; use `--greedy` for an additional
token-sequence-controlled comparison when backend outputs can agree.

On this RTX 4070, Qwen NVFP4 could not start with a 40% expert cache because
the KV pool had no memory left. For the repeated comparison, run both models at
the feasible 20% cache setting:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py baseline \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --cache-rate 0.2 \
  --decode 256 --repeat 5 \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/baseline/qwen_cache20_decode256_repeat5.jsonl
```

Use a different output path for GPT-OSS, then summarize each JSONL with
`experiments/bandwidth/bandwidth_study.py baseline-summary`.

To measure the prompt length effect separately, run:

```bash
source scripts/env.sh
uv run --no-sync python experiments/baselines/bench_prefill_bandwidth.py \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --cache-rate 0.2 --repeat 3 \
  --target-tokens 1024 2048 4096 6000 --decode 8 \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/baseline/qwen_prefill_1k_6k_repeat3_decode8.jsonl
uv run --no-sync python experiments/bandwidth/bandwidth_study.py prefill-summary \
  --input results/baseline/qwen_prefill_1k_6k_repeat3_decode8.jsonl \
  --out results/baseline/qwen_prefill_1k_6k_repeat3_decode8_summary.csv
```

Repeat with the GPT-OSS checkpoint and distinct output paths. This benchmark
uses a naive KV cache and unique request tags to avoid prefix reuse. It records
actual prompt token counts and first-token latency for approximately 1K, 2K,
4K, and 6K Qwen prompts. A 7.9K Qwen request at 20% expert cache failed
with CUDA OOM at the default 90% GPU memory target
(`results/baseline/qwen_prefill_7900_failure.json`); 6K was validated with
overlap both on and off. Reducing only the expert cache to 10% or 5% did not
fix the 7.9K OOM because the automatic KV allocation used the freed VRAM;
`qwen_prefill_7900_cache10_failure.json` and
`qwen_prefill_7900_cache05_failure.json` preserve those attempts.
At the same 20% expert cache, the 80% memory target completed a separate
three-repeat 1K/2K/4K/7.9K Qwen sweep:

```bash
source scripts/env.sh
uv run --no-sync python experiments/baselines/bench_prefill_bandwidth.py \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --backend hybrid \
  --cache-rate 0.2 --mem-ratio 0.8 --target-tokens 1024 2048 4096 7900 \
  --repeat 3 --decode 8 \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/baseline/qwen_prefill_1k_8k_repeat3_mem80_decode8.jsonl
uv run --no-sync python experiments/bandwidth/bandwidth_study.py prefill-summary \
  --input results/baseline/qwen_prefill_1k_8k_repeat3_mem80_decode8.jsonl \
  --out results/baseline/qwen_prefill_1k_8k_repeat3_mem80_decode8_summary.csv
```

Keep the 80%- and 90%-memory-target results separate. Eight output tokens are allowed so
GPT-OSS can emit visible text after its initial special tokens; TTFT is the
first visible text delta, not the first raw token. Incremental tokens/s in
the summary comes from the change in median first-token latency between
adjacent lengths, so it still includes first-token overhead.

## Phase 1: same-machine transfer sizes

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py transfer --repeat 10 \
  --out results/transfer_microbench/banked/contiguous_vs_banked.csv
```

The CSV and adjacent `summary.csv` contain pinned H2D latency and effective
GB/s for one contiguous buffer and six separate buffers sized like the model's
quantized expert banks. This isolates copy granularity and batch size. Each bank
is contiguous and uses `torch.copy_`; `ft bench bw` measures the production
indexed multi-bank gather separately. Neither copy proxy is production expert
bandwidth. The earlier contiguous-only run remains in
`results/transfer_microbench/{raw,summary}.csv`.

## Phase 2: decode traffic

Trace one backend and one model per run. The trace requires eager decode and
synchronizes each layer, so its throughput is diagnostic rather than a baseline.

```bash
source scripts/env.sh
uv run --no-sync python experiments/baselines/bench_decode_moe.py \
  --model /path/to/qwen-checkpoint --backend hybrid --no-graph \
  --trace-csv results/traffic_trace/qwen_raw.csv \
  --json results/traffic_trace/qwen_benchmark.jsonl

uv run --no-sync python experiments/bandwidth/bandwidth_study.py trace-summary \
  --trace results/traffic_trace/qwen_raw.csv \
  --benchmark results/traffic_trace/qwen_benchmark.jsonl \
  --out results/traffic_trace/qwen_summary.csv
```

Repeat for GPT-OSS. The benchmark row records `trace_start_row` and
`trace_end_row`, which exclude the warmup request. The raw trace logs routed ids,
unique hits/misses, actual GPU fills, CPU-executed misses, bank bytes per complete
expert, cache slots/bytes/occupancy, and the runtime's rounded `q` decision.
`request_id` and branch/layer timing fields are blank because the cache does not
receive a request id and this diagnostic trace does not time concurrent branches.
For offload, every miss is filled; CPU-only layers do not enter this trace path.
For capacity curves, hold the generated sequence fixed with `--greedy --decode 64`
and `--cache-rate`. The measured Qwen rates were 0.1, 0.2, and 0.25;
GPT-OSS rates were 0.1, 0.2, and 0.4. Use distinct `--trace-csv`, `--json`,
and summary paths per rate. For example:

```bash
source scripts/env.sh
uv run --no-sync python experiments/baselines/bench_decode_moe.py \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --backend hybrid \
  --cache-rate 0.1 --no-graph --greedy --decode 64 \
  --trace-csv results/traffic_trace/qwen_hybrid_cache10.csv \
  --json results/traffic_trace/qwen_hybrid_cache10.jsonl
uv run --no-sync python experiments/bandwidth/bandwidth_study.py trace-summary \
  --trace results/traffic_trace/qwen_hybrid_cache10.csv \
  --benchmark results/traffic_trace/qwen_hybrid_cache10.jsonl \
  --out results/traffic_trace/qwen_hybrid_cache10_summary.csv
```

The earlier sampled 20% traces feed the core comparison table; the plotted
capacity curves use the matched greedy traces.

## Phase 3: controlled routing replay

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py replay --seed 42
```

This generates seeded uniform routing and sweeps E, k, relative cache capacity,
and the current Q16 rounded fetch fraction. It is a simulator, not a model run.
For a measured trace:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py replay \
  --trace results/traffic_trace/qwen_raw.csv \
  --benchmark results/traffic_trace/qwen_benchmark.jsonl \
  --profile qwen3.6-35b-nvfp4 --k 1 2 4 8 \
  --cache-slots 256 512 \
  --out results/geometry_replay/qwen_trace.csv
```

Trace replay folds ids modulo E to vary the expert universe. This is a
controlled approximation; collisions and changed routing probabilities mean it
does not represent a retrained model. Report the seed and trace provenance.
For GPT-OSS use `--profile gpt-oss-20b-mxfp4 --experts 32 --k 1 2 4`;
the observed trace cannot supply routing identities for more than 32 experts.
The replay CSV includes both slot count and byte capacity. `--cache-slots`
adds fixed-absolute-capacity cells alongside the default relative-capacity
cells, marked by `capacity_mode`.
At the calibrated ~0.16 fetch fraction, k=4 often rounds to zero fills, so
capacity cannot affect those replay cells. Repeat the command with
`--q-fraction 0.4 --out results/geometry_replay/counterfactual/qwen_fraction40.csv`
to study a labeled fill-producing policy counterfactual; do not present it as
the measured runtime policy.

For optional popularity coverage on a single measured request:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py popularity \
  --trace results/traffic_trace/qwen_hybrid_cache20.csv \
  --benchmark results/traffic_trace/qwen_hybrid_cache20.jsonl \
  --out results/traffic_trace/qwen_popularity.csv
```

This counts routed accesses; it does not measure accuracy after pruning.

The held-out retention calculation uses the first half of a measured request
to rank experts and scores the second half separately. It reports retained
25%, 50%, 75%, and 100% for the whole model, each layer, and four test-time
windows:

```bash
source scripts/env.sh
uv run --no-sync python experiments/pruning/pruning_replay.py \
  --train-trace results/traffic_trace/qwen_hybrid_cache20_greedy.csv \
  --train-benchmark results/traffic_trace/qwen_hybrid_cache20_greedy.jsonl \
  --train-name aime25_problem0_first_half \
  --test-name aime25_problem0_second_half \
  --out results/pruning_optional/qwen_temporal.csv
```

Use the corresponding GPT-OSS trace and benchmark for its temporal result.
To test transfer to a different prompt, pass `--test-trace` and
`--test-benchmark` for that measured request. This is an access-coverage
counterfactual; no model weights are removed and no quality result is implied.
The optional non-math prompts are in `experiments/prompts/routing_workloads.jsonl`.
For example, record the coding prompt with the same greedy settings:

```bash
source scripts/env.sh
FREETOKEN_BENCHBW_PATH="$PWD/results/baseline/calibration_repeat5/profile-run-04.json" \
uv run --no-sync python experiments/baselines/bench_decode_moe.py \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --backend hybrid \
  --cache-rate 0.2 --decode 64 --no-graph --greedy \
  --aime experiments/prompts/routing_workloads.jsonl --problem 0 \
  --trace-csv results/pruning_optional/qwen_code_trace.csv \
  --json results/pruning_optional/qwen_code_benchmark.jsonl

uv run --no-sync python experiments/pruning/pruning_replay.py \
  --train-trace results/traffic_trace/qwen_hybrid_cache20_greedy.csv \
  --train-benchmark results/traffic_trace/qwen_hybrid_cache20_greedy.jsonl \
  --test-trace results/pruning_optional/qwen_code_trace.csv \
  --test-benchmark results/pruning_optional/qwen_code_benchmark.jsonl \
  --train-name math --test-name code \
  --out results/pruning_optional/qwen_math_to_code.csv
```

Use the GPT-OSS checkpoint and corresponding GPT-OSS training trace for the
same cross-workload comparison. The local workload JSONL sets
`boxed_instruction=false`, avoiding a math-answer instruction on the coding
and incident prompts. One request per workload is only an exploratory sample.

## Phase 4: forced-q end-to-end sweep

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-sweep \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --max-q 8 \
  --repeat 3 --decode 64 --cache-rate 0.2 --greedy \
  --trace-csv results/qstar_oracle/qwen_timed_trace.csv --layer-timing \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/qstar_oracle/qwen_timed.jsonl

uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-summary \
  --input results/qstar_oracle/qwen_timed.jsonl \
  --model results/checkpoints/qwen3.6-35b-nvfp4 \
  --out results/qstar_oracle/qwen_summary.csv
```

Use `--max-q 4` for GPT-OSS. `q=-1` uses the calibrated runtime policy; forced
values are caps, so a layer with fewer than `q` misses fills only its misses.
The summary finds the best cap for whole-request decode latency. It is not a
per-`m` layer-latency oracle; that still requires timed branch/layer traces under
identical routing and cache state. Do not change the runtime policy based on
this cap sweep alone.

For GPT-OSS use the same sweep command with its checkpoint, `--max-q 4`,
and `gpt_oss_timed` output paths. Then summarize actual miss and fill counts:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-layer-summary \
  --benchmark results/qstar_oracle/qwen_timed.jsonl \
  --trace results/qstar_oracle/qwen_timed_trace.csv \
  --model results/checkpoints/qwen3.6-35b-nvfp4 \
  --fraction 0.16741516966067865 \
  --out results/qstar_oracle/qwen_layer_summary.csv
```

The timing trace synchronizes after every MoE layer and slows decode. The
summary groups by actual `m` and GPU fills, marks an oracle only when all
`q=0..m` values were observed, and compares medians with the runtime's Q16
rounded prediction. `min_samples_per_q` exposes thinly sampled cells; the
oracle-error figures require at least 30 observations at every q.
`distinct_output_hashes` reports whether generated text
was identical across runs. Cache states may differ among forced-q runs, so
verify hashes and routing distributions before interpreting small penalties.
In particular, a q=0 run may leave the expert cache empty after warmup, so
it cannot supply q=0 samples at lower miss counts that only occur in
fill-producing runs. Such m cells remain `complete_q_sweep=False`; a matched
cache-state experiment would be needed for their oracle.

For Qwen's lower miss-count cells, warm each server with q=8, switch to the
forced measured q immediately before the measured request, and compare only
the first measured decode token:

```bash
source scripts/env.sh
CC=/usr/bin/gcc uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-sweep \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --max-q 8 \
  --forced-only --warmup-q 8 --repeat 3 --decode 64 --cache-rate 0.2 --greedy \
  --trace-csv results/qstar_oracle/qwen_matched_timed_trace.csv --layer-timing \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/qstar_oracle/qwen_matched_timed.jsonl

uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-layer-summary \
  --benchmark results/qstar_oracle/qwen_matched_timed.jsonl \
  --trace results/qstar_oracle/qwen_matched_timed_trace.csv \
  --model results/checkpoints/qwen3.6-35b-nvfp4 \
  --fraction 0.16741516966067865 --first-token-only --match-initial-layers \
  --out results/qstar_oracle/qwen_matched_layer_summary.csv
```

The q switch is active only with decode tracing enabled. The warmup follows
the same prompt and cap in every run, but identical generated text does not
guarantee identical first-token routing or cache misses. The summary selects
per-layer `(routing, m)` signatures seen under every forced q; its sample
counts, `matched_initial_layers`, and raw pattern count expose how much of the
experiment remains comparable. Caps above a layer's actual `m` are excluded
from that cell so they cannot over-weight the `q=m` endpoint. This is still an
observational match, not a bitwise cache snapshot or an untraced throughput
measurement.

The targeted Qwen m<=4 follow-up uses eight repeats at q=0..4, with the same
q=8 warmup, synchronized layer trace, and greedy request. It can be summarized
using the same `q-layer-summary` invocation with its own paths:

```bash
source scripts/env.sh
CC=/usr/bin/gcc uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-sweep \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --max-q 4 --forced-only \
  --warmup-q 8 --repeat 8 --decode 64 --cache-rate 0.2 --greedy \
  --trace-csv results/qstar_oracle/qwen_m4_matched_timed_trace.csv --layer-timing \
  --profile results/baseline/calibration_repeat5/profile-run-04.json \
  --out results/qstar_oracle/qwen_m4_matched_timed.jsonl

uv run --no-sync python experiments/bandwidth/bandwidth_study.py q-layer-summary \
  --benchmark results/qstar_oracle/qwen_m4_matched_timed.jsonl \
  --trace results/qstar_oracle/qwen_m4_matched_timed_trace.csv \
  --model results/checkpoints/qwen3.6-35b-nvfp4 \
  --fraction 0.16741516966067865 --first-token-only --match-initial-layers \
  --out results/qstar_oracle/qwen_m4_matched_layer_summary.csv
```

To collect coarse utilization alongside that sweep, start
`experiments/common/monitor_system.py --watch-pid PID --interval 1` in another
terminal after the q-sweep starts, then align its CSV with the measured decode
windows using `experiments/common/monitor_summary.py --benchmark ... --trace ...
--monitor ... --out ...`. These are whole-host CPU and device-wide GPU samples,
not per-layer branch utilization; the monitor may miss the earliest runs. The
recorded sweep was not perfectly isolated from local diagnostics, so use the
branch-only sweep and repeat any small real-model latency difference in an
otherwise idle environment before drawing a policy conclusion.

For an additional controlled branch sweep, use the production pinned-bank
gather and CPU MoE kernels with fixed `m` and every `q=0..m`:

```bash
source scripts/env.sh
CC=/usr/bin/gcc uv run --no-sync python experiments/bandwidth/branch_oracle.py \
  --model qwen3.6-moe --fraction 0.16741516966067865 \
  --misses 1 2 3 4 5 6 7 8 --repeats 40 --warmup 3 \
  --out results/qstar_oracle/qwen_branch_oracle_raw.csv

CC=/usr/bin/gcc uv run --no-sync python experiments/bandwidth/branch_oracle.py \
  --model gpt-oss-20b --fraction 0.16349902534113062 \
  --misses 1 2 3 4 --repeats 40 --warmup 3 \
  --out results/qstar_oracle/gpt_oss_branch_oracle_raw.csv
```

Each command also writes a `_summary.csv` and `_config.json` next to the raw
CSV. Set the CUDA 13 environment exports from Setup first. The branch sweep
holds its miss count and synthetic expert layout fixed while timing CPU work,
GPU gather, and their overlap. It excludes router changes, cache admission,
and GPU expert GEMM, so its minimum is a branch oracle rather than a measured
end-to-end layer oracle.

To resolve Qwen's unmatched real-model m=4 cell, the opt-in layer replay freezes
one actual decode activation, routing decision, and pre-layer cache state.
It preflights each q to save the bank rows that will be overwritten, restores
both those rows and all cache bookkeeping between trials, then invokes the
production hybrid MoE layer. Run three distinct captured layers sequentially:

```bash
source scripts/env.sh
for layer in 0 28 39; do
  FREETOKEN_BENCHBW_PATH="$PWD/results/baseline/calibration_repeat5/profile-run-04.json" \
  uv run --no-sync python experiments/baselines/bench_decode_moe.py \
    --model results/checkpoints/qwen3.6-35b-nvfp4 --backend hybrid \
    --cache-rate 0.2 --decode 64 --no-graph --greedy \
    --trace-csv "results/qstar_oracle/qwen_m4_layer${layer}_repeat40_trace.csv" \
    --layer-replay-out "results/qstar_oracle/qwen_m4_layer${layer}_repeat40.csv" \
    --layer-replay-m 4 --layer-replay-layer "$layer" \
    --layer-replay-repeats 40 --layer-replay-warmup 3 \
    --json "results/qstar_oracle/qwen_m4_layer${layer}_repeat40_benchmark.jsonl"
done

uv run --no-sync python experiments/bandwidth/layer_replay_summary.py \
  --raw results/qstar_oracle/qwen_m4_layer0_repeat40.csv \
        results/qstar_oracle/qwen_m4_layer28_repeat40.csv \
        results/qstar_oracle/qwen_m4_layer39_repeat40.csv \
  --out results/qstar_oracle/qwen_m4_controlled_layer_summary.csv
```

Use the CUDA 13 environment exports from Setup, an idle GPU, and fresh output
paths. The replay starts only after the measured request is armed; if the
selected layer never has m=4, the benchmark fails rather than inventing a
sample. The raw CSV checks realized m and q, and records full-layer wall,
PCIe-device, GPU-GEMM-device, and output-difference measurements. Do not use
its request-level tokens/s as ordinary serving throughput: replay pauses one
request while it runs many diagnostic layer calls.

For a separate, short CUDA timeline, add `--layer-replay-profile-out
results/qstar_oracle/qwen_m4_torch_profile_chrome.json` to a three-repeat
replay command with a distinct raw output path, then run:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/profile_replay_summary.py \
  --trace results/qstar_oracle/qwen_m4_torch_profile_chrome.json \
  --config results/qstar_oracle/qwen_m4_torch_profile_replay_config.json \
  --out results/qstar_oracle/qwen_m4_torch_profile_summary.csv
```

The profile summary reports the fraction of each profiled layer window covered
by CUDA kernels, not SM occupancy. Profiling changes latency, so keep it
separate from the 40-repeat oracle. The earlier Nsight Systems could not decode
the current CUDA driver trace, and CPU hardware sampling is blocked by
`perf_event_paranoid=4`; actual host-DRAM bandwidth is not available here.
The final Nsight Systems 2025.3.2 retry succeeded; see Profiling closeout below.

## Representative-state q follow-up

For native CPU branch attribution, rebuild the C++ extensions with the CUDA 13
Build the external dependency with `bash scripts/setup.sh`. Then
set `FREETOKEN_MOE_LAYER_REPLAY_CPU_TIMING=1` when running selected replay cases
with a fresh output directory. The raw CSV adds `cpu_task_ms` and
`cpu_effective_gbps`. This opt-in clock spans native submit preparation,
worker dispatch, and compute until the last worker finishes. It excludes
activation D2H, coordinator waiting, and output H2D. Effective GB/s uses
CPU-assigned misses times complete expert bytes, not hardware DRAM counters.
An all-GPU split still measures CPU task dispatch overhead and reports zero
effective expert GB/s. Keep these instrumented trials separate from existing
oracle measurements. Without the flag, no native task clock is sampled.

```bash
source scripts/env.sh
FREETOKEN_MOE_LAYER_REPLAY_CPU_TIMING=1 \
FREETOKEN_BENCHBW_PATH="$PWD/results/baseline/calibration_repeat5/profile-run-04.json" \
uv run --no-sync python experiments/bandwidth/representative_layer_sweep.py \
  --case qwen_code_m5 --case qwen_code_m5_late \
  --case qwen_incident_m5 --case qwen_incident_m5_late \
  --out-dir results/qstar_oracle/qwen_m5_cpu_timing
```

This command completed on 2026-09-30 after GPU device access was restored.
Do not rerun into the same directory. A fifth math state used the AIME command
below with `FREETOKEN_MOE_LAYER_REPLAY_CPU_TIMING=1` and all output paths changed
to `results/qstar_oracle/qwen_m5_cpu_timing/`. The math case kept layer 0,
skip-matches 4, 40 repeats, and 3 warmup rounds.

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/representative_state_summary.py \
  --case-dir results/qstar_oracle/qwen_m5_cpu_timing \
  --out results/qstar_oracle/qwen_m5_cpu_timing/state_summary.csv

uv run --no-sync python experiments/bandwidth/layer_replay_summary.py \
  --raw results/qstar_oracle/qwen_m5_cpu_timing/qwen_code_m5.csv \
        results/qstar_oracle/qwen_m5_cpu_timing/qwen_code_m5_late.csv \
        results/qstar_oracle/qwen_m5_cpu_timing/qwen_incident_m5.csv \
        results/qstar_oracle/qwen_m5_cpu_timing/qwen_incident_m5_late.csv \
        results/qstar_oracle/qwen_m5_cpu_timing/qwen_math_m5.csv \
  --out results/qstar_oracle/qwen_m5_cpu_timing/branch_summary.csv
```

The summaries already exist; these commands document their provenance. The
branch summary includes optional CPU metrics while leaving them blank for
older, uninstrumented raw files. GPU access was verified with `nvidia-smi`
and PyTorch CUDA availability. The earlier missing-device sandbox issue is
resolved; no driver reinstall was required.

The follow-up samples Qwen m=3/4/5 and GPT-OSS m=4 on the coding and incident
prompts in `experiments/prompts/routing_workloads.jsonl`. Each case uses a fresh
server, the same 20% expert-cache setting, greedy decode, and 40 q trials per
captured state. It skips a specified number of matching measured layer calls
to include later-token states. Run only when the GPU is otherwise idle:

```bash
source scripts/env.sh
FREETOKEN_BENCHBW_PATH="$PWD/results/baseline/calibration_repeat5/profile-run-04.json" \
uv run --no-sync python experiments/bandwidth/representative_layer_sweep.py

uv run --no-sync python experiments/bandwidth/representative_state_summary.py \
  --case-dir results/qstar_oracle/representative_states \
  --out results/qstar_oracle/representative_states/state_summary.csv
```

Use the CUDA 13 environment exports from Setup. The runner refuses to overwrite
any case output and checks for an existing GPU compute process before each
server launch. Use repeated `--case NAME` options to run selected cases, or a
fresh `--out-dir` to repeat the sweep. The state summary keeps each snapshot
separate and adds exploratory paired-bootstrap intervals for the predicted q
versus its nearest alternative. These intervals describe timing variability
within captured states, not uncertainty across all model workloads. Request
tokens/s in these traced replay runs is diagnostic, not an ordinary serving
baseline.

The separate math-prompt m=5 cross-check used the cached AIME-25 `test.jsonl`
snapshot at revision `563bb8404243c5f09de6ec262f2db674fe5bce9b`:

```bash
source scripts/env.sh
FREETOKEN_BENCHBW_PATH="$PWD/results/baseline/calibration_repeat5/profile-run-04.json" \
uv run --no-sync python experiments/baselines/bench_decode_moe.py \
  --model results/checkpoints/qwen3.6-35b-nvfp4 --backend hybrid \
  --cache-rate 0.2 --decode 64 --no-graph --greedy \
  --aime "${FREETOKEN_AIME25_JSONL:?set the pinned AIME-25 JSONL path}" \
  --problem 0 --layer-replay-out results/qstar_oracle/representative_states/qwen_math_m5.csv \
  --layer-replay-m 5 --layer-replay-layer 0 --layer-replay-skip-matches 4 \
  --layer-replay-repeats 40 --layer-replay-warmup 3 \
  --trace-csv results/qstar_oracle/representative_states/qwen_math_m5_trace.csv \
  --json results/qstar_oracle/representative_states/qwen_math_m5_benchmark.jsonl
```

The absolute AIME path is specific to this host; use the same pinned dataset
revision at another local path when reproducing elsewhere. Run the summary
after this cross-check so it includes the math state.

## Figures

Install matplotlib if needed (`uv pip install matplotlib`), then run:

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/plot_bandwidth_study.py \
  --trace results/traffic_trace/qwen_raw.csv
```

The plotting script emits figures only when the corresponding CSV/JSONL exists.
It reads successful baseline JSONL records under `results/baseline/` and selects
the longest decode records per model for the throughput figure.
Per-`m` q-oracle plots use synchronized per-layer latency and only include
complete q sweeps with at least 30 observations per q. Such timings are
diagnostic and must not be compared directly with ordinary serving throughput.

Never compare traced throughput with untraced throughput. Report mean, median,
standard deviation, min, max, sample count, the exact commands, and the raw data.

## Final independent replication and artifact consolidation

Use the Setup CUDA exports, then disable native CPU timing and torch profiling
for independent replay runs. CUDA event intervals and synchronized replay wall
times remain diagnostic; these are not untraced serving requests.

```bash
source scripts/env.sh
unset FREETOKEN_MOE_LAYER_REPLAY_CPU_TIMING FREETOKEN_MOE_LAYER_REPLAY_PROFILE_OUT
export FREETOKEN_BENCHBW_PATH="$PWD/results/baseline/calibration_repeat5/profile-run-04.json"
export PYTHONPATH=python:.

uv run --no-sync python experiments/bandwidth/representative_layer_sweep.py \
  --case qwen_code_m5 --case qwen_code_m5_late \
  --case qwen_incident_m5 --case qwen_incident_m5_late \
  --out-dir results/qstar_oracle/qwen_m5_independent_run1

uv run --no-sync python experiments/bandwidth/representative_layer_sweep.py \
  --case qwen_code_m5 --case qwen_code_m5_late \
  --case qwen_incident_m5 --case qwen_incident_m5_late \
  --case qwen_code_m1 --case qwen_code_m2 --case qwen_incident_m2 \
  --case qwen_incident_m6 --case qwen_incident_m7 --case qwen_incident_m8 \
  --out-dir results/qstar_oracle/qwen_independent_run2_extended
```

Each independent directory also receives `qwen_math_m5` using the exact math
command above with its output directory substituted and CPU timing unset.
The runner uses 40 measured repeats per q, 3 warmup rounds, seed 42, cache
rate 0.2, eager hybrid decode, and greedy sampling. The added miss-count cases
are fixed in `experiments/bandwidth/representative_layer_sweep.py`; they were selected
from measured coding/incident traces, not from warmup rows.

```bash
source scripts/env.sh
uv run --no-sync python experiments/bandwidth/finalize_study.py \
  --case-dir results/qstar_oracle/representative_states \
             results/qstar_oracle/qwen_m5_cpu_timing \
             results/qstar_oracle/qwen_m5_independent_run1 \
             results/qstar_oracle/qwen_independent_run2_extended \
  --raw results/qstar_oracle/qwen_m4_layer0_repeat40.csv \
        results/qstar_oracle/qwen_m4_layer28_repeat40.csv \
        results/qstar_oracle/qwen_m4_layer39_repeat40.csv \
  --out-dir results/final_study

uv run --no-sync python experiments/bandwidth/plot_bandwidth_study.py \
  --results results --out results/figures_final
```

The finalizer refuses existing output directories, validates every q cell and
realized miss/fill count, keeps each capture separate, and archives available
server logs. By-case and by-m summaries separate native CPU timing from untimed
CPU instrumentation. Matching selection metadata across runs does not assert
identical hidden states or complete cache snapshots across processes. Paired
intervals describe within-capture variability, not model-wide uncertainty.
The artifact audit checks presence/nonempty files for the 21 required plots;
it does not certify their interpretation or hardware-counter measurements.

## Profiling closeout with Nsight Systems 2025.3.2

The CUDA 13 smoke succeeded with `nsys profile --trace=cuda --sample=none
--cpuctxsw=none --output=results/qstar_oracle/nsys2025_cuda13_smoke
results/cuda13_smoke`. An NVTX-range-triggered run launched through `nsys ...
uv run ...` did not activate capture and generated no report. The successful
retry moved `uv` outside the profiler and used full capture:

```bash
source scripts/env.sh
FREETOKEN_MOE_LAYER_REPLAY_CPU_TIMING=1 uv run --no-sync nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none --wait=all \
  --output=results/qstar_oracle/nsys2025_qwen_m5_full \
  uv run --no-sync python experiments/bandwidth/representative_layer_sweep.py \
  --case qwen_code_m5 --repeats 10 --warmup 1 \
  --out-dir results/qstar_oracle/nsys2025_full_capture

nsys export --type sqlite \
  --output results/qstar_oracle/nsys2025_qwen_m5_full.sqlite \
  results/qstar_oracle/nsys2025_qwen_m5_full.nsys-rep

uv run --no-sync python experiments/bandwidth/profile_replay_summary.py \
  --sqlite results/qstar_oracle/nsys2025_qwen_m5_full.sqlite \
  --config results/qstar_oracle/nsys2025_full_capture/qwen_code_m5_config.json \
  --out results/qstar_oracle/nsys2025_qwen_m5_stages.csv

uv run --no-sync python experiments/bandwidth/layer_replay_summary.py \
  --raw results/qstar_oracle/nsys2025_full_capture/qwen_code_m5.csv \
  --out results/qstar_oracle/nsys2025_full_capture/branch_summary.csv
```

These files exist; use fresh names for a repeat. The successful trace contains
67 trial windows: one q=0 reference, six warmup windows, and sixty measured
windows. The parser validates that count and a single process, then selects
the last ten windows per q and filters kernel/copy events to that process.
It unions overlapping active intervals. Role-specific metrics recognize the
NVFP4 expert and multi-bank gather kernel names; absent roles are blank.
The gather reads pinned host memory within a CUDA kernel, so HtoD memcpy
activity is not an expert-traffic counter. CPU sampling remains disabled;
timeline-active percentages must not be called SM occupancy.


## Preserved workspace and handoff facts

The earlier handoff recorded HEAD `4d0f84983700eba7dfb046fb4df3cbbcc488ecc2`, a dirty/uncommitted worktree and complete finite closeout. Existing user engine hooks and imported studies must be preserved. Checkpoints are under `results/checkpoints/`; the runtime calibration is `results/baseline/calibration_repeat5/profile-run-04.json`. These are workspace-specific, Git-ignored artifacts.

GPU access and Nsight Systems 2025.3.2 were restored during the study. Earlier absent-device/trace-import failures were historical and must not be shown as current blockers. A range-triggered profiler attempt produced no report; its trials remain separate from the successful full-capture data.

The native CPU-attribution batch contained five m=5 states and 1,200 trials, separate from untimed measurements. Its historical MoE suite result was 150 passed, five skipped and six deselected; final closeout later recorded 153 passed, five skipped and six deselected. These are distinct validation points, not new runs during cleanup.

[Report](report.md) · [Single experiment log](../../EXPERIMENT_LOG.md) · [Remaining plan](../../PLAN.md).
