# Runtime-aware q: result tables

Generated from the preserved runtime-q CSVs. Numerical cells are unchanged. `Decide us` includes feature/logging overhead; the separate 0.9 microsecond lean-rule result is a different measurement. See [report.md](report.md) for interpretation and [reproduction.md](reproduction.md) for commands.

All numbers: gpt-oss-20b MXFP4, RTX 4070, hybrid backend, eager decode through the runtime-q hook, greedy, 128 measured tokens, same prompt. ms/token is client-side.

## 1. Final comparison (3 rotated repeats; ± = 95% CI)

| Policy | low_miss ms/tok | med_miss ms/tok | high_miss ms/tok | Avg ms/tok | Avg tok/s | Geo speedup vs q* | Worst speedup | E2E regret ms | Hit rate | CPU MB/tok | PCIe MB/tok | Decide us | 1-step regret ms/tok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Reuse-aware rule (kappa=1) | 31.6 ± 2.6 | 45.0 ± 2.8 | 55.9 ± 10.0 | 44.2 | 24.0 | 1.519 | 1.16 | 1.6 | 0.55 | 533 | 39 | 205 | 2.35 |
| Reuse-aware, uniform reuse prior | 31.8 ± 1.6 | 47.2 ± 16.9 | 54.4 ± 5.0 | 44.5 | 23.8 | 1.505 | 1.20 | 2.0 | 0.52 | 586 | 22 | 240 | 1.30 |
| Reuse-aware rule (kappa=4) | 37.3 ± 4.8 | 42.8 ± 3.0 | 53.2 ± 3.0 | 44.4 | 23.0 | 1.487 | 1.22 | 1.9 | 0.47 | 658 | 10 | 217 | 0.46 |
| Ridge (H=8) | 33.4 ± 3.3 | 46.7 ± 3.6 | 58.2 ± 8.4 | 46.1 | 22.9 | 1.453 | 1.12 | 3.6 | 0.54 | 546 | 40 | 273 | 2.39 |
| m-table (q=1 if m>=3) | 32.6 ± 2.6 | 47.0 ± 2.4 | 64.9 ± 7.9 | 48.2 | 22.5 | 1.410 | 1.00 | 5.7 | 0.52 | 514 | 93 | 206 | 5.21 |
| q=25% | 42.8 ± 3.4 | 45.8 ± 3.6 | 54.4 ± 4.0 | 47.7 | 21.2 | 1.377 | 1.19 | 5.2 | 0.41 | 723 | 26 | 216 | 0.92 |
| LinUCB (H=16) | 34.3 ± 3.1 | 55.7 ± 2.5 | 59.3 ± 4.0 | 49.8 | 21.3 | 1.349 | 1.10 | 7.3 | 0.48 | 620 | 36 | 269 | 2.22 |
| Linear (H=16) | 37.9 ± 3.7 | 51.2 ± 4.4 | 59.7 ± 3.9 | 49.6 | 20.9 | 1.340 | 1.09 | 7.1 | 0.54 | 490 | 93 | 266 | 6.75 |
| Thompson sampling (H=16) | 37.4 ± 10.6 | 56.7 ± 11.8 | 61.9 ± 17.1 | 52.0 | 20.3 | 1.286 | 1.05 | 9.5 | 0.42 | 720 | 23 | 302 | 2.50 |
| q=50% | 34.9 ± 2.8 | 62.6 ± 2.7 | 85.5 ± 3.0 | 61.0 | 18.8 | 1.143 | 0.76 | 18.5 | 0.53 | 374 | 225 | 194 | 19.54 |
| Reuse-aware, no cache state | 32.9 ± 2.7 | 75.9 ± 2.6 | 107.4 ± 2.5 | 72.1 | 17.6 | 1.013 | 0.61 | 29.6 | 0.54 | 286 | 298 | 189 | 31.23 |
| q* (FreeToken) | 65.0 ± 3.4 | 66.0 ± 4.3 | 65.0 ± 4.2 | 65.3 | 15.3 | 1.000 | 1.00 | 22.8 | 0.00 | 1271 | 0 | 200 | 0.00 |
| q=0 | 65.3 ± 2.8 | 66.1 ± 3.9 | 64.7 ± 3.5 | 65.4 | 15.3 | 0.999 | 1.00 | 22.8 | 0.00 | 1271 | 0 | 200 | 0.00 |
| Tree (H=16) | 68.6 ± 4.8 | 69.7 ± 3.7 | 68.4 ± 3.1 | 68.9 | 14.5 | 0.948 | 0.95 | 26.4 | 0.00 | 1271 | 0 | 328 | 0.00 |
| LinUCB myopic (H=0) | 68.6 ± 3.2 | 70.2 ± 4.1 | 68.6 ± 3.4 | 69.1 | 14.5 | 0.945 | 0.94 | 26.6 | 0.00 | 1271 | 0 | 272 | 0.00 |
| ANN-GELU 2x64 (H=16) | 69.0 ± 3.4 | 70.5 ± 4.0 | 69.0 ± 3.6 | 69.5 | 14.4 | 0.940 | 0.94 | 27.0 | 0.00 | 1271 | 0 | 346 | 0.00 |
| GBM (H=16) | 66.6 ± 2.8 | 88.8 ± 48.7 | 88.6 ± 3.0 | 81.3 | 12.6 | 0.810 | 0.73 | 38.8 | 0.47 | 639 | 32 | 1545 | 1.52 |
| q=75% | 60.5 ± 2.7 | 102.2 ± 2.8 | 133.0 ± 2.4 | 98.6 | 11.3 | 0.697 | 0.49 | 56.0 | 0.50 | 124 | 506 | 176 | 58.70 |
| q=m | 65.0 ± 2.5 | 125.4 ± 2.5 | 210.9 ± 13.1 | 133.8 | 9.4 | 0.545 | 0.31 | 91.3 | 0.41 | 0 | 752 | 192 | 91.66 |

### Paired speedup vs q* (same round), 95% CI, paired t-test p

| workload | policy | speedup_vs_qstar | speedup_ci95 | paired_p |
|---|---|---|---|---|
| high_miss | f25 | 1.195 | ±0.009 | 0.0000 |
| high_miss | linucb_h16 | 1.096 | ±0.004 | 0.0001 |
| high_miss | ra_k1 | 1.167 | ±0.226 | 0.0779 |
| high_miss | ridge_h8 | 1.119 | ±0.178 | 0.0964 |
| high_miss | t0011 | 1.003 | ±0.147 | 0.9779 |
| low_miss | f25 | 1.519 | ±0.041 | 0.0000 |
| low_miss | linucb_h16 | 1.895 | ±0.105 | 0.0002 |
| low_miss | ra_k1 | 2.057 | ±0.059 | 0.0000 |
| low_miss | ridge_h8 | 1.945 | ±0.085 | 0.0000 |
| low_miss | t0011 | 1.994 | ±0.050 | 0.0000 |
| med_miss | f25 | 1.440 | ±0.020 | 0.0001 |
| med_miss | linucb_h16 | 1.184 | ±0.120 | 0.0198 |
| med_miss | ra_k1 | 1.467 | ±0.089 | 0.0016 |
| med_miss | ridge_h8 | 1.413 | ±0.024 | 0.0001 |
| med_miss | t0011 | 1.404 | ±0.018 | 0.0005 |

## Ablation anchors (same servers as the ablation runs)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms |
|---|---|---|---|---|---|---|
| f25 | 42.7 | 46.8 | 54.1 | 47.9 | 1.372 | 4.1 |
| qstar | 65.6 | 65.1 | 65.3 | 65.3 | 1.000 | 21.5 |
| ra_k1 | 32.5 | 44.7 | 54.9 | 44.0 | 1.517 | 0.3 |

## 2. Incremental feature-group addition (ridge, H=8)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms | n_features | mae | rmse | infer_us |
|---|---|---|---|---|---|---|---|---|---|---|
| ridge_00_base | 67.6 | 67.3 | 68.3 | 67.7 | 0.965 | 23.9 | 1 | 4.77 | 6.63 | 11 |
| ridge_01_plus_Routing | 67.8 | 67.5 | 67.9 | 67.7 | 0.965 | 23.9 | 3 | 4.74 | 6.60 | 12 |
| ridge_02_plus_CPU | 67.4 | 67.2 | 69.8 | 68.2 | 0.959 | 24.4 | 6 | 4.44 | 6.30 | 12 |
| ridge_03_plus_GPU | 52.7 | 54.9 | 60.8 | 56.1 | 1.167 | 12.3 | 8 | 4.35 | 6.16 | 12 |
| ridge_04_plus_Transfer | 62.4 | 63.0 | 64.7 | 63.3 | 1.031 | 19.6 | 10 | 4.07 | 5.74 | 12 |
| ridge_05_plus_Overlap | 48.7 | 53.0 | 59.8 | 53.9 | 1.218 | 10.1 | 15 | 4.06 | 5.72 | 12 |
| ridge_06_plus_Cache | 37.2 | 47.4 | 56.9 | 47.2 | 1.406 | 3.4 | 25 | 2.88 | 4.00 | 12 |
| ridge_07_plus_Reuse | 34.4 | 46.6 | 56.6 | 45.9 | 1.454 | 2.1 | 35 | 2.86 | 3.96 | 12 |
| ridge_08_plus_Utilization | 38.9 | 46.6 | 57.2 | 47.6 | 1.391 | 3.8 | 39 | 2.84 | 3.92 | 12 |
| ridge_09_plus_Memory | 42.1 | 59.3 | 86.2 | 62.5 | 1.090 | 18.7 | 41 | 2.87 | 3.94 | 12 |
| ridge_10_plus_History | 43.9 | 62.8 | 125.9 | 77.5 | 0.930 | 33.7 | 63 | 2.75 | 3.75 | 13 |

## 3. Leave-one-group-out (ridge, H=8)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms | n_features | mae | rmse | infer_us |
|---|---|---|---|---|---|---|---|---|---|---|
| ridge_minus_CPU | 47.5 | 70.6 | 163.8 | 94.0 | 0.797 | 50.2 | 60 | 2.75 | 3.75 | 13 |
| ridge_minus_Cache | 65.6 | 100.1 | 209.9 | 125.2 | 0.587 | 81.4 | 53 | 2.96 | 4.07 | 13 |
| ridge_minus_GPU | 41.7 | 60.8 | 137.4 | 80.0 | 0.928 | 36.2 | 61 | 2.75 | 3.75 | 13 |
| ridge_minus_History | 40.6 | 61.2 | 87.2 | 63.0 | 1.088 | 19.2 | 41 | 2.87 | 3.94 | 12 |
| ridge_minus_Memory | 33.5 | 45.9 | 56.1 | 45.2 | 1.479 | 1.4 | 61 | 2.72 | 3.73 | 13 |
| ridge_minus_Overlap | 49.4 | 62.7 | 150.9 | 87.7 | 0.842 | 43.9 | 58 | 2.75 | 3.75 | 13 |
| ridge_minus_Reuse | 53.9 | 68.8 | 139.3 | 87.3 | 0.814 | 43.5 | 53 | 2.76 | 3.76 | 13 |
| ridge_minus_Routing | 57.2 | 76.4 | 171.8 | 101.8 | 0.719 | 58.0 | 61 | 2.77 | 3.77 | 13 |
| ridge_minus_Transfer | 42.5 | 61.4 | 140.0 | 81.3 | 0.913 | 37.5 | 61 | 2.75 | 3.75 | 13 |
| ridge_minus_Utilization | 47.1 | 64.2 | 114.7 | 75.3 | 0.930 | 31.5 | 59 | 2.76 | 3.77 | 13 |

## 4. Individual signal ablation (ridge, H=8)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms | n_features | mae | rmse | infer_us |
|---|---|---|---|---|---|---|---|---|---|---|
| ridge_minus_B_H_recent | 41.0 | 46.1 | 75.1 | 54.1 | 1.252 | 10.3 | 62 | 2.75 | 3.74 | 13 |
| ridge_minus_T_CPU | 45.6 | 45.9 | 87.0 | 59.5 | 1.152 | 15.7 | 55 | 2.75 | 3.74 | 13 |
| ridge_minus_T_GPU | 40.2 | 50.4 | 74.6 | 55.1 | 1.227 | 11.3 | 58 | 2.76 | 3.76 | 13 |
| ridge_minus_T_overlap | 44.8 | 48.6 | 78.4 | 57.3 | 1.178 | 13.5 | 58 | 2.75 | 3.75 | 13 |
| ridge_minus_T_transfer | 52.8 | 55.4 | 80.3 | 62.8 | 1.059 | 19.1 | 61 | 2.75 | 3.75 | 13 |
| ridge_minus_cache | 65.3 | 47.9 | 218.4 | 110.6 | 0.742 | 66.8 | 53 | 2.96 | 4.07 | 13 |
| ridge_minus_history | 42.5 | 47.6 | 72.5 | 54.2 | 1.239 | 10.4 | 41 | 2.87 | 3.94 | 12 |
| ridge_minus_m | 54.5 | 45.7 | 76.7 | 58.9 | 1.135 | 15.2 | 62 | 2.77 | 3.77 | 13 |
| ridge_minus_memory | 34.6 | 45.9 | 56.3 | 45.6 | 1.462 | 1.8 | 61 | 2.72 | 3.73 | 13 |
| ridge_minus_reuse | 56.6 | 48.1 | 84.4 | 63.0 | 1.067 | 19.2 | 53 | 2.76 | 3.76 | 13 |
| ridge_minus_utilization | 48.2 | 45.7 | 82.9 | 58.9 | 1.152 | 15.1 | 59 | 2.76 | 3.77 | 13 |

## 5. Cache-information ladder (ridge, H=8)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms | n_features | mae | rmse | infer_us |
|---|---|---|---|---|---|---|---|---|---|---|
| ridge_C1_none | 69.5 | 67.1 | 70.1 | 68.9 | 0.948 | 25.1 | 3 | 4.74 | 6.60 | 12 |
| ridge_C2_occupancy | 67.8 | 67.4 | 69.0 | 68.1 | 0.960 | 24.3 | 6 | 4.00 | 5.78 | 12 |
| ridge_C3_occ_hitmiss | 68.2 | 67.9 | 68.5 | 68.2 | 0.958 | 24.4 | 7 | 3.93 | 5.68 | 12 |
| ridge_C4_occ_hitmiss_reuse | 69.1 | 67.4 | 68.3 | 68.3 | 0.957 | 24.5 | 17 | 3.90 | 5.63 | 12 |
| ridge_C5_full_cache | 33.9 | 47.1 | 56.6 | 45.9 | 1.455 | 2.1 | 23 | 2.86 | 3.96 | 12 |

## 6. History-window ablation (ridge, H=8)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms | n_features | mae | rmse | infer_us |
|---|---|---|---|---|---|---|---|---|---|---|
| ridge_hist00_current | 45.0 | 48.5 | 86.4 | 60.0 | 1.139 | 16.2 | 29 | 2.89 | 3.95 | 12 |
| ridge_hist01 | 46.8 | 47.8 | 83.9 | 59.5 | 1.141 | 15.7 | 31 | 2.89 | 3.95 | 12 |
| ridge_hist04 | 57.9 | 46.4 | 91.1 | 65.1 | 1.045 | 21.3 | 37 | 2.83 | 3.85 | 12 |
| ridge_hist08 | 58.7 | 46.2 | 92.9 | 66.0 | 1.034 | 22.2 | 41 | 2.80 | 3.81 | 13 |
| ridge_hist16 | 58.5 | 46.6 | 86.4 | 63.9 | 1.058 | 20.1 | 46 | 2.77 | 3.76 | 13 |
| ridge_hist32 | 65.2 | 46.1 | 85.3 | 65.6 | 1.028 | 21.8 | 51 | 2.76 | 3.75 | 13 |

## 7. ANN architecture ablation (GELU MLP, H=16)

| config | low_miss | med_miss | high_miss | avg_ms | geo_speedup_vs_qstar | e2e_regret_ms | n_features | mae | rmse | infer_us | n_params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_1x128 | 64.2 | 69.6 | 70.3 | 68.0 | 0.961 | 24.3 | 63 | 3.91 | 5.38 | 39 | 8577 |
| mlp_1x16 | 68.6 | 72.2 | 69.4 | 70.1 | 0.933 | 26.3 | 63 | 4.02 | 5.59 | 31 | 1073 |
| mlp_1x32 | 68.4 | 69.1 | 69.2 | 68.9 | 0.948 | 25.1 | 63 | 3.96 | 5.46 | 24 | 2145 |
| mlp_1x64 | 68.8 | 68.7 | 69.7 | 69.1 | 0.946 | 25.3 | 63 | 3.99 | 5.50 | 30 | 4289 |
| mlp_2x128 | 71.4 | 70.9 | 72.9 | 71.8 | 0.911 | 28.0 | 63 | 3.99 | 5.57 | 65 | 25089 |
| mlp_2x16 | 37.8 | 74.2 | 64.9 | 59.0 | 1.153 | 15.2 | 63 | 3.98 | 5.53 | 31 | 1345 |
| mlp_2x32 | 69.3 | 69.8 | 70.0 | 69.7 | 0.938 | 25.9 | 63 | 3.93 | 5.46 | 35 | 3201 |
| mlp_2x64 | 69.7 | 76.0 | 70.3 | 72.0 | 0.908 | 28.2 | 63 | 4.07 | 5.72 | 44 | 8449 |
| mlp_3x128 | 66.7 | 75.1 | 113.8 | 85.2 | 0.788 | 41.4 | 63 | 4.05 | 5.65 | 87 | 41601 |
| mlp_3x16 | 69.3 | 75.5 | 69.9 | 71.5 | 0.914 | 27.8 | 63 | 4.06 | 5.71 | 37 | 1617 |
| mlp_3x32 | 69.8 | 72.3 | 70.2 | 70.8 | 0.923 | 27.0 | 63 | 4.05 | 5.71 | 44 | 4257 |
| mlp_3x64 | 46.9 | 74.3 | 71.1 | 64.1 | 1.040 | 20.3 | 63 | 3.99 | 5.58 | 57 | 12609 |

### Removed-group table (vs Full ridge)

| Removed Group | Performance Degradation (ms/tok) | Degradation % | Regret Increase (ms/tok) | Offline MAE increase |
|---|---|---|---|---|
| CPU | +16.5 | +21.3% | +16.5 | +0.005 |
| Cache | +47.7 | +61.5% | +47.7 | +0.216 |
| GPU | +2.4 | +3.2% | +2.4 | +0.002 |
| History | -14.5 | -18.7% | -14.5 | +0.124 |
| Memory | -32.4 | -41.7% | -32.4 | -0.028 |
| Overlap | +10.1 | +13.1% | +10.1 | +0.001 |
| Reuse | +9.8 | +12.7% | +9.8 | +0.010 |
| Routing | +24.3 | +31.3% | +24.3 | +0.017 |
| Transfer | +3.8 | +4.9% | +3.8 | +0.000 |
| Utilization | -2.2 | -2.8% | -2.2 | +0.011 |

## 8. Generalization: unseen prompt (p1_*) and unseen cache sizes (rate*)

| policy | p1_high ms/tok | p1_low ms/tok | p1_med ms/tok | rate015 ms/tok | rate030 ms/tok | rate060 ms/tok | geo speedup | worst speedup |
|---|---|---|---|---|---|---|---|---|
| ra_k1 | 50.2 | 32.0 | 41.9 | 47.8 | 36.1 | 26.1 | 1.703 | 1.28 |
| ra_k1_online | 52.1 | 34.0 | 43.3 | 48.9 | 37.2 | 26.2 | 1.652 | 1.24 |
| t0011 | 62.7 | 31.7 | 44.2 | 54.6 | 36.9 | 31.6 | 1.539 | 1.03 |
| f25 | 51.9 | 42.2 | 46.4 | 48.2 | 42.1 | 42.2 | 1.429 | 1.24 |
| f50 | 90.8 | 31.9 | 58.6 | 75.5 | 44.6 | 26.5 | 1.303 | 0.71 |
| ridge_h8 | 82.5 | 47.6 | 62.6 | 72.9 | 57.6 | 34.4 | 1.130 | 0.78 |
| linear_h16 | 86.2 | 51.4 | 70.1 | 76.4 | 62.8 | 34.5 | 1.063 | 0.75 |
| qstar | 64.5 | 64.6 | 65.8 | 63.7 | 64.3 | 66.0 | 1.000 | 1.00 |

## 9. Decision overhead: full feature logging vs lean

| policy | high_miss ms/tok | low_miss ms/tok | med_miss ms/tok | geo speedup | worst speedup |
|---|---|---|---|---|---|
| ra_k1_uniformp_cacheonly | 50.3 | 30.5 | 41.4 | 1.616 | 1.27 |
| ra_k1_lean | 51.2 | 30.3 | 42.7 | 1.594 | 1.25 |
| ra_k1_full | 52.8 | 31.8 | 43.8 | 1.540 | 1.21 |
| f25_lean | 51.1 | 37.8 | 40.7 | 1.505 | 1.25 |
| qstar_lean | 60.0 | 60.9 | 61.1 | 1.064 | 1.05 |
| qstar_full | 63.8 | 65.3 | 64.5 | 1.000 | 1.00 |

## 10. Runtime adaptation: 4 CPU busy-loops during decode

| policy | low_miss_cpuhog4 ms/tok | med_miss_cpuhog4 ms/tok | geo speedup | worst speedup |
|---|---|---|---|---|
| ra_k1_online | 35.2 | 50.0 | 1.679 | 1.40 |
| ra_k1 | 38.3 | 49.0 | 1.624 | 1.43 |
| f25 | 47.6 | 51.0 | 1.428 | 1.37 |
| qstar | 71.0 | 69.9 | 1.000 | 1.00 |
| ridge_h8 | 64.7 | 92.3 | 0.911 | 0.76 |

## 11. Longer decode (384 tokens)

| policy | med_miss_decode384 ms/tok | geo speedup | worst speedup |
|---|---|---|---|
| ra_k1_online | 43.1 | 1.528 | 1.53 |
| ra_k1 | 44.4 | 1.485 | 1.49 |
| f25 | 45.0 | 1.464 | 1.46 |
| ridge_h8 | 54.6 | 1.206 | 1.21 |
| qstar | 65.9 | 1.000 | 1.00 |
