# Pruning resource diagnosis and recovery

Final disposition: quality and eligible serving diagnostics completed on 8 October. This page combines diagnosis and resolved recovery, with historical snapshots explicitly labelled. The chronological record is [the single experiment log](../../EXPERIMENT_LOG.md).

## What is established

Qwen is being rejected by a whole-machine resource screen. This is not a model quality failure. The most recent completed attempt reached 112/123 captures before the global swap-in counter exceeded 128 MiB. Its resource trace records 141.66 MiB swap-in, zero swap-out, approximately 134.56 MiB private server swap at both inference boundaries, a minimum 4.74 GiB available RAM, and full-pressure avg10 no higher than 0.03 percent. Its inference interval lasted approximately 421 seconds.

The original attempt for batch 7 had a different failure: 131.54 MiB swap-out and approximately 75.87 MiB server private-swap growth. That is actual eviction pressure. Later attempts mostly failed because already-swapped pages were read back during inference. A 10-second quiet interval is insufficient to guarantee that periodic background activity stays quiet for a whole eight-minute batch.

| Attempt for batch 7 | Swap in MiB | Swap out MiB | Server swap growth MiB | Min available GiB | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| initial | 72.94 | 131.54 | 75.86 | 4.57 | failed |
| retry2 | 128.73 | 0.00 | 0.00 | 4.61 | failed |
| retry3 | 134.94 | 0.00 | 0.00 | 4.56 | failed |
| retry4 | 129.50 | 0.65 | 0.00 | 4.39 | failed |
| retry5 | 141.66 | 0.00 | 0.00 | 4.74 | failed |


The unlisted retry1 failed before measured inference because startup memory did not settle within 90 seconds.

## Why retrying the same launch did not reliably help

- The gate measures global `/proc/vmstat` counters, so background activity can reject a Qwen batch even when its own private swap is stable.
- Startup of the server subtree (roughly 21.6 GiB summed RSS; shared pages may be counted more than once) competes with the desktop and other processes for a 30.39 GiB host. CUDA/CPU expert banks remain allocated when 25% of expert IDs are masked, so this experiment does not reduce bank allocation.
- Before mitigation, Firefox's process tree held approximately 1,604 MiB proportional resident memory and 603 MiB proportional swap. Other swap holders include the desktop and OpenClaw; these services are preserved.
- A fresh idle 30-second observation recorded 0.109 MiB system swap-in and 16 major faults in the active Codex process with 92 KiB private swap reduction. This confirms background swap activity exists; it does not prove which process generated all historical inference-period IO.
- Private `VmSwap` is not a complete swap-IO trace and excludes shared-memory swap. Major faults also include file-backed faults. Exact historical attribution is unavailable because previous captures did not record whole-machine process fault snapshots. These limits are described in the [Linux proc documentation](https://docs.kernel.org/filesystems/proc.html).
- Linux swap readahead can fetch additional pages. The current `page-cluster` value is 3. Its behavior is documented in the [kernel VM controls](https://www.kernel.org/doc/html/latest/admin-guide/sysctl/vm.html#page-cluster). No sysctl has been changed.

## Mitigation applied

The user confirmed Firefox was unused. Its root process was closed with SIGTERM and exited. Immediately afterward, available RAM increased by 1.19 GiB and system swap stock fell by 613.7 MiB. This is a concrete environment change before another attempt; remaining applications and services were preserved.

The frozen FreeToken v6 runtime, corpus, masks, engine hashes and resource controls still match. Batch 7 retries with the same full 123 evaluations and original treatment order. Its six previous attempts remain rejected. A recorded amendment permits two additional attempts after this environmental mitigation, with an eight-attempt total cap. Numerical criteria and the 128 MiB swap limit are unchanged.

An external five-second process observer records PID/start-time, parent PID, fault counters, RSS and private swap alongside the unchanged original gate. It does not read process memory or change inference. It runs outside expert worker cores and stops when collection ends, before serving diagnostics. It supports attribution if the same problem recurs.

## Recovery decision and fallbacks

1. A batch is accepted only after all captures and the existing full resource checks pass. A low intermediate counter is not a pass.
2. If background process activity correlates with swap recovery, close only user-confirmed unused applications. Active desktop, network, editor and Codex services are preserved.
3. If startup itself evicts the serving processes, diagnose peak startup RSS, shared-memory residency and private swap before considering an additional startup warmup. Such a warmup requires a separate recorded execution amendment and fresh qualification; never modify frozen inference code in place.
4. If meaningful pressure persists after the two added attempts, retain the blocker and data. A smaller declared batch partition or alternative host-residency arm requires a new frozen study/qualification and explicit accounting amendment; do not merge partial rejected outputs into accepted v6 shards.
5. Swapoff/swapon or swappiness changes are not required for the current recovery. Privileged system controls are unavailable without an interactive sudo password. No password was requested and no privileged change was made.

## Final recovery result

The post-Firefox quality recovery completed. Qwen batch 7 passed on retry 6, and the remaining five batches passed their first subsequent attempts. All three checkpoints now have twelve accepted quality batches. The six earlier rejected batch-7 attempts remain excluded and preserved.

Later serving measurements exposed similar resource failures on GPT-OSS. A transient user-owned systemd scope with `MemorySwapMax=0` prevented the serving subtree from adding swap; global VM settings remained unchanged. Whole paired repetitions were retried after global swap or memory-pressure rejections. GPT-OSS finished 15 accepted serving arms and Qwen ten. The final 75% supplement finished export and verification of 44,155 artifacts at 2026-10-08T07:15:37 UTC (12:45:37 IST).

Firefox closure was a useful capacity recovery, but process-specific swap protection was also needed for the serving stage. A zero server-swap counter does not guarantee the whole-machine pressure and swap gates will pass. All original numerical criteria remained in effect. See `docs/experiments/pruning/report.md` for accepted results and environment-comparability limits.

## Historical post-mitigation attribution snapshot

During the quality recovery, the process observer provided direct evidence beyond the earlier inference: across 206.41 seconds, system swap-in was 73.684 MiB. Most observed major-fault activity and private swap-stock recovery came from active Codex and networking services. Top observations:

| Process | Major fault delta | Private swap decrease MiB | In serving subtree |
| --- | ---: | ---: | --- |
| codex (PID 802801) | 3975 | 20.32 | False |
| warp-svc (PID 1478) | 1200 | 22.99 | False |
| python3 (PID 1007353) | 1038 | 4.10 | True |
| tailscaled (PID 2173) | 1011 | 6.86 | False |
| codex (PID 797875) | 938 | 7.82 | False |
| cloudflared (PID 3001) | 688 | 4.75 | False |

These applications are required services or active sessions and have been preserved. This supports background swap recovery as a contributor in the new run; it does not reconstruct the exact source of every old counter increment. Firefox closure increases capacity, but does not by itself guarantee an eight-minute whole-machine quiet interval. At that observation snapshot, the 128 MiB gate still applied and full-batch acceptance was pending. The final accepted outcome is recorded above.
