# Reproducing the experiments

## 1. Scope and runtime identity

The target is **Linux x86_64 across NVIDIA GPUs**. Python is pinned to the 3.13 series.
All Python commands use `uv`. The runtime is the repository's original engine commit
plus the recorded experiment hook patch; see [source identity](../runtime/source.json),
[patch](../runtime/patches/experiment-hooks.patch), and
[observed runtime packages](../runtime/observed_environment.txt).

The root project installs analysis dependencies. It does not build a FreeToken app.
Full setup installs and, where necessary, compiles the external FreeToken dependency
under `.runtime/freetoken`. The frozen v6 pruning runtime has a different numerical
source identity and remains separate. Do not claim the current runtime is that frozen arm.

## 2. Set up another NVIDIA machine

Clone with Git history; the recorded base commit must be present. Install `uv`, an
NVIDIA driver compatible with the pinned CUDA 13 runtime, a C/C++ compiler, and system
development headers. The pinned package list supplies the CUDA 13 compiler wheels.
Have sufficient host RAM, VRAM and disk for the selected checkpoint's full weights.

```bash
bash scripts/setup.sh --analysis-only
bash scripts/doctor.sh --out results/machine_before_runtime.json
bash scripts/setup.sh
bash scripts/doctor.sh --require-runtime --out results/machine_after_runtime.json
source scripts/env.sh
```

Full setup restores the original engine from the base Git commit, checks and applies
the versioned hook patch, installs the observed dependencies with `uv pip`, and installs
that external source with `uv pip`. It uses the official PyTorch CUDA 13 index for the
observed `torch==2.11.0+cu130` build. The environment is kept in `.venv`; `--inexact`
preserves the inference packages while syncing the analysis lock.

To use an existing source dependency:

```bash
export FREETOKEN_SOURCE_DIR=/path/to/FreeToken
bash scripts/setup.sh --source "$FREETOKEN_SOURCE_DIR"
```

An arbitrary external checkout is a new source arm. Its tracing/replay hooks and
checkpoint backend must be qualified before a study relies on them. Setup does not
silently replace it with the preserved patched runtime.

Different NVIDIA GPUs can need different kernels. A successful CUDA diagnostic is
not qualification of every checkpoint backend. The old machine's compiled `.so`,
kernel caches and virtual environments must not be copied as portable dependencies;
recreate and compile the runtime on the destination. Missing NVFP4/MXFP4 kernels,
insufficient VRAM/RAM, or loader incompatibility are failures to record. Changing
quantization, cache size, driver or execution backend creates a separate arm.

## 3. Checkpoints and baseline profiles

Use [the model pins](../configs/pruning/models.json). Kimi remains excluded. Local
checkpoint paths are supplied by the caller; there are no hard-coded home directories.

| Profile | Repository | Revision | Expert cache | Sequence cap | VRAM ratio |
|---|---|---|---:|---:|---:|
| `gpt_oss_mxfp4` | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | 154 | 2048 | 0.9 |
| `qwen_nvfp4` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | `1355db6a052410cfd62085d94b58866fd0f2c3c5` | 2048 | 2048 | 0.9 |
| `gemma_nvfp4` | `nvidia/Gemma-4-26B-A4B-NVFP4` | `a19cfe00be84568a6867111c9a68c9c44fdcffe6` | 768 | 1280 | 0.8 |

Download or copy the pinned local Hugging Face checkpoint directory. Preserve its
`.cache/huggingface/download/*.metadata` files when copying. Authenticate separately
if the checkpoint requires access; credentials do not belong in logs or Git.

```bash
source scripts/env.sh
export MODEL=/path/to/checkpoints/gpt-oss-20b
uv run --no-sync hf download openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee --local-dir "$MODEL"
bash scripts/baseline.sh --profile gpt_oss_mxfp4 --model "$MODEL" --gpu 0 --dry-run
bash scripts/baseline.sh --profile gpt_oss_mxfp4 --model "$MODEL" --gpu 0 --repeats 3
```

The new baseline measures **unpruned** hybrid serving, with a fresh server per repeat,
one warmup request, greedy decode, 128 decode tokens, the fixed local routing workload,
eager execution, naive KV cache, a 1024-token prefill cap, serial expert loading and
five CPU expert workers. Non-GPT profiles are text-only. It checks model geometry and
the config file's recorded revision. Metadata hashes are recorded; this baseline does
not hash all weight shards or validate their download metadata as the pruning audit does.
FTW imports from the old studies remain historical artifacts; these baseline profiles
expect a Hugging Face checkpoint directory with `config.json`.

`--cache-slots`, `--memory-ratio`, `--cpu-threads` and `--allow-unpinned` are explicit
ways to define another arm. Report their use. Baseline timing is not numerical
equivalence to the sealed pruning supplement: its request, warmup and cache reset
contracts differ. For a policy comparison, hold the baseline contract constant.

By default the baseline is launched in a user systemd scope with `MemorySwapMax=0`.
If user scopes are unavailable, the command fails with an actionable error. Set
`EXPERIMENT_NO_SWAP=0` only for an explicitly recorded uncapped arm. This is a process
scope; it does not change the machine's global swap configuration. The original
pruning acceptance gates still apply to pruning experiments. A completed new baseline
is not automatically a pressure-qualified pruning performance arm.

## 4. Run a study through Bash

```bash
bash scripts/run.sh --list
bash scripts/run.sh prefill_overlap run --model "$MODEL" --gpu 0 --help
bash scripts/run.sh routing_locality analyze --help
bash scripts/run.sh split_policy run --model "$MODEL" --gpu 0 --help
bash scripts/run.sh runtime_q run --plan configs/runtime_q/pilot_plan.json --model "$MODEL" --help
bash scripts/run.sh bandwidth bandwidth_study --help
bash scripts/run.sh pruning moe_pruning --help
```

The syntax is `bash scripts/run.sh STUDY MODULE [arguments]`. Module means a Python
filename stem in that study. `run`, `analyze`, `plot` and `train` aliases are available
where listed by the corresponding package. Use module `--help` before collection.
Help commands do not create experiment records.

| Study package | Original contributor | Main entry points |
|---|---|---|
| `baselines` | Shared serving and microbenchmarks | `baseline`, `bench_decode_moe`, `bench_prefill_bandwidth` |
| `bandwidth` | Ankit | `bandwidth_study`, `branch_oracle`, `representative_layer_sweep`, `finalize_study` |
| `pruning` | Ankit | collectors, `moe_pruning`, masked inference and serving controllers |
| `prefill_overlap` | Kushal | `run_prefill`, `analyze_prefill` |
| `routing_locality` | Yeshwanth | `lru_locality`, `plots_locality` |
| `split_policy` | Saketh | `run_experiment`, `analyze`, `branch_analyze`, `plots`, `report` |
| `runtime_q` | Saketh | `driver`, `build_dataset`, `train_all`, analyses and plots |
| `common` | Shared | system monitors, environment capture and run recorder |

Each recorded invocation creates `results/runs/STUDY/UTC_UUID/` containing:

- `run.json`: exact command, start/end time, status, return code and output location.
- `environment.json`: CPU/RAM, GPU/driver, selected environment values, package versions,
  Git state, and engine source/binary hashes for inference entry points.
- `logs/console.log` and retained server logs for the shared decode helper.
- `outputs/STUDY/`: study data when the runner uses the shared study path.

Explicit `--out` or `--study` arguments remain authoritative; use a fresh location.
Controllers with frozen source checks must be used on newly qualified arms or through
the original frozen runtime. Failed runs stay on disk; they are not accepted measurements.
The runner does not blindly retry failures or lower quality/resource thresholds.

For a multi-step study, create one new output root and export it before each recorded
command. Reusing the same root is an explicit resume of that study, not a new independent
replicate. Existing runtime-q plan model paths are relative to the study directory;
`RUNTIME_Q_MODEL_ROOT` can identify a previously trained artifact root. Record that
dependency and keep its model files unchanged. Full runtime-q training also uses the
external runtime's Torch package, so analysis-only setup is insufficient for the MLP stage.

## 5. Historical evidence and reproduction copies

Original local bandwidth/pruning outputs remain at their original `results/` paths.
The three imported studies are consolidated under:

```text
results/historical/prefill_overlap/{results,logs,figures}
results/historical/routing_locality/{results,figures}
results/historical/split_policy/{results,logs,figures}
results/historical/runtime_q/{data,models,results,logs,figures,reports}
```

The code reads those historical study directories by default when called directly.
The Bash run recorder assigns a fresh output root. To regenerate figures or summaries
from old data, copy the relevant study into a new working output root before running
analysis. Do not point mutating commands at `results/historical` or sealed pruning outputs.
Generated tables and feature reports go beside the new run's data; maintained reports
in `docs/` are edited deliberately after review.

For cross-machine transfer, copy selected checkpoints, raw input data, frozen runtimes
and backups separately: these large artifacts are ignored by Git. Preserve hidden
checkpoint metadata. Frozen runtime symlinks to `.venv`, `results`, and the archived
engine fixtures must be repaired to the destination's locations without editing its
numerical Python/C++ source.

```bash
bash scripts/frozen.sh .pruning-runtimes/v6_resume_20261007 experiments.run_masked_inference75 --help
```

The launcher changes to the frozen runtime and uses only its engine/collector import
paths while sharing the uv-managed environment. This avoids importing the new packages
into a sealed study. Transferred frozen runtimes still require their original compatible
dependencies and platform-specific binaries to be rebuilt and identified on the new host.

## 6. Blocker handling

| Blocker | Action | Measurement handling |
|---|---|---|
| Missing `uv` or source base commit | Install uv; fetch the recorded commit or provide an explicit source checkout | No run |
| CUDA/compiler/CPU extension failure | Fix driver/toolchain and rebuild the external runtime with setup | Preserve setup error; no accepted inference |
| GPU already occupied | Wait for the selected GPU to be free | No baseline request sent |
| Missing trained runtime-q model | Train or explicitly reference its artifact root | Do not substitute another policy |
| RAM/swap pressure or OOM | Stop the failed unit, free resources, retain logs, retry the whole paired unit | Do not combine rejected partial timings |
| Changed cache, precision, source or checkpoint | Declare and qualify a separate arm | Keep old and new timings separate |
| Frozen source identity mismatch | Use original frozen launcher or qualify a new study | Do not amend sealed manifests |
| GPU lacks the checkpoint's kernel path | Qualify the required backend/dependencies or choose an explicitly new supported configuration | Do not claim equivalent quantization/results |

## 7. Backups and audit

`archives/layout_20261013/` holds verified ZIPs for the engine/build tree, original local
experiment sources and each contributor folder, plus the pre-cleanup tracked diff.
The backup manifest stores each archived file's size/hash and each ZIP hash. Original
import ZIPs are retained under `archives/imports/`. Reconstructible Python caches and
the teammate virtual environment were excluded from ZIPs; that environment is preserved
locally at `.runtime/environments/runtime-q-analysis` and is not portable.

The [layout manifest](_archive/layout_manifest_20261013.json) maps old source/data paths
to their new locations. Backup ZIPs, environments, runtime checkouts and raw results
are ignored. The observed package list is provenance; analysis constraints preserve the previously installed shared dependency versions. Baseline resource JSONL records span startup and serving, and are diagnostic evidence rather than the original inference-window acceptance check. the analysis `uv.lock` is tracked
for repeatable environment resolution. Keep the sole maintained history in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), and future research in [PLAN.md](PLAN.md).
