"""Write the saketh_exp_2 runtime-q plans (decision policy is the only variable).

Fixed for every arm of a model (provenance in docs/AUDIT.md section 4 and docs/REPRODUCIBILITY.md):
  workload   AIME-25 (math-ai/aime25 @563bb84) problems 0-2, the official bench_decode_moe
             prompt; 256 greedy decode tokens (--decode on the driver command line)
  cache      --moe-cache-auto (cache_rate 0 -> auto in server/args.py), as in the paper and
             the official benchmark; the resolved slot count is recorded per server
  serve      the established per-model flags of the policy study / baseline profiles
  env        the established calibration profile and harness env; TZ pinned to UTC
             because the GPT-OSS chat template embeds the local date
  timing     one warm-up + one measured generate per episode from a cold expert cache,
             3 burn-in episodes per server, 5 timed rounds + 1 logged round, order rotated

Arms (all on one hook, so harness cost is identical):
  A   paper q*: Eq. 4 with Table-1-style standalone bandwidths, at least one fill
  A0  the same without the minimum fill (diagnostic)
  E   built-in q* (engine kernel and its own profile fraction)
  B   original Reuse-Aware
  C_* B + V1/V2/V3, every subset
plus an unhooked E server per problem (cache reset by same-size rebuild) for harness cost.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "plans"
HOOK = "saketh_exp_2/policy/sx2_hook.py"
PROMPTS = "saketh_exp_2/config/prompts_aime25.jsonl"
PROFILE = ROOT / "results/reuse_aware_models/study_20261020T071921Z/calibration/benchbw.json"
PROBLEMS = (0, 1, 2)
TIMED, LOGGED_REP = 5, 5
BURNIN = 3

# Established serve flags (configs/policy_study/stage1_<model>.json, configs/baselines.json).
SERVE = {
    "gpt_oss": ["--mem-ratio", "0.9", "--max-seq-len", "2048", "--max-prefill-length", "1024",
                "--cpu-threads", "5", "--serial-load"],
    "qwen": ["--mem-ratio", "0.9", "--max-seq-len", "2048", "--text-model-only",
             "--max-prefill-length", "1024", "--cpu-threads", "5", "--serial-load"],
    "gemma": ["--mem-ratio", "0.8", "--max-seq-len", "1280", "--text-model-only",
              "--max-prefill-length", "1024", "--cpu-threads", "5", "--serial-load"],
}


def bandwidths(model: str) -> dict:
    """Paper Eq. 4 uses standalone B_P/B_H on the model's shapes (Table 1). B keeps its
    original rho convention, B_P / B_H with the overlapped CPU bandwidth (runtime_q defaults
    6.71/35.67 and the policy study). Gemma has no Gemma-shape NVFP4 entry in the profile, so
    its values come from the generic NVFP4 dtype entry (unresolved; see AUDIT.md)."""
    prof = json.loads(PROFILE.read_text())
    if model == "gpt_oss":
        k = prof["workloads"]["gpt-oss-20b"]["kernels"]["mxfp4_triton"]
    elif model == "qwen":
        k = prof["workloads"]["qwen3.6-moe"]["kernels"]["nvfp4"]
    else:
        k = prof["dtype_kernels"]["nvfp4"]
    return {"bp": k["pcie_gather_gbs"], "bh": k["cpu_moe_gbs"],
            "bp_ov": k["pcie_gather_overlap_gbs"], "bh_ov": k["cpu_moe_overlap_gbs"]}


def env(model: str) -> dict:
    bw = bandwidths(model)
    return {"FREETOKEN_BENCHBW_PATH": str(PROFILE), "FREETOKEN_MOE_RUNTIME_Q_BP": bw["bp_ov"],
            "FREETOKEN_MOE_RUNTIME_Q_BH": bw["bh_ov"], "FREETOKEN_DISABLE_OVERLAP_SCHEDULING": 1,
            "MALLOC_ARENA_MAX": 2, "OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "TZ": "Etc/UTC"}


def arms(model: str, log: str, log_rep: int) -> list:
    bw = bandwidths(model)
    f_paper = round(bw["bp"] / bw["bh"], 6)
    V = {"V1": {"cost": "runtime"}, "V2": {"reuse": "calibrated"}, "V3": {"evict": True}}
    specs = {
        "A_qstar_paper": {"kind": "fraction_min1", "fraction": f_paper},
        "A0_qstar_paper_nomin": {"kind": "fraction", "fraction": f_paper},
        "E_qstar_engine": {"kind": "native"},
        "B_reuse_aware": {"kind": "reuse_aware", "kappa": 1.0},
    }
    for combo in ("V1", "V2", "V3", "V1V2", "V1V3", "V2V3", "V1V2V3"):
        spec = {"kind": "reuse_aware_v"}
        for v in ("V1", "V2", "V3"):
            if v in combo:
                spec.update(V[v])
        specs[f"C_{combo}"] = spec
    return [{"name": n, "spec": s, "groups": ["BASE"], "control": {"features": "lean"},
             "log": log, "log_rep": log_rep} for n, s in specs.items()]


def hooked_group(model: str, problem: int, log: str, repeats: int, log_rep: int, burnin: int,
                 names=None) -> dict:
    eps = arms(model, log, log_rep)
    if names:
        eps = [e for e in eps if e["name"] in names]
    return {"workload": "auto", "cache_rate": 0, "problem": problem, "prompts": PROMPTS,
            "serve_args": SERVE[model], "save_text": True, "repeats": repeats, "burnin": burnin,
            "hook": HOOK, "env": env(model), "episodes": eps}


def nohook_group(model: str, problem: int) -> dict:
    return {"workload": "auto", "cache_rate": 0, "problem": problem, "prompts": PROMPTS,
            "serve_args": SERVE[model], "save_text": True, "repeats": TIMED, "burnin": 2,
            "hook": "none", "reset": "rebuild", "env": env(model),
            "episodes": [{"name": "E_qstar_engine_nohook", "spec": {"kind": "native"}}]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for model in SERVE:
        smoke = {"groups": [hooked_group(model, 0, f"results/smoke_lean_{model}.csv", 1, 0, 1,
                                         {"A_qstar_paper", "E_qstar_engine", "B_reuse_aware", "C_V1V2V3"})]}
        main_plan = {"groups": [hooked_group(model, p, f"results/main_lean_{model}.csv", TIMED + 1,
                                             LOGGED_REP, BURNIN) for p in PROBLEMS]
                     + [nohook_group(model, p) for p in PROBLEMS]}
        for name, plan in (("smoke", smoke), ("main", main_plan)):
            path = OUT / f"{name}_{model}.json"
            path.write_text(json.dumps(plan, indent=1) + "\n")
            print(path.relative_to(ROOT), f"paper f={arms(model, '', 0)[0]['spec']['fraction']}")


if __name__ == "__main__":
    main()
