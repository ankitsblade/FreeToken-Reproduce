# q* policy comparison tables

Source: Saketh's saved summary CSVs in `results/historical/split_policy/results/summary/`. Generated result table, preserved without changing numerical cells. Read graph/eager context in [report.md](report.md).


## Workload: low_miss


### Performance and traffic

| Policy | avg q | miss m | CPU MB/tok | H2D MB/tok | tok/s | ms/tok | p95 ms | GPU util% | CPU util% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q* adaptive | 0.00 | 4.00 | 1270.7 | 0.0 | 24.32 | 41.11 | 43.23 | 21 | 41 |
| q=m (GPU) | 1.06 | 1.06 | 0.0 | 336.4 | 19.17 | 52.16 | 113.37 | 88 | 18 |
| q=0 (CPU) | 0.00 | 4.00 | 1270.7 | 0.0 | 24.27 | 41.20 | 43.86 | 21 | 41 |
| q=25% | 0.05 | 2.28 | 708.9 | 15.8 | 40.25 | 24.84 | 29.41 | 33 | 35 |
| q=50% | 0.20 | 0.87 | 214.7 | 62.8 | 50.54 | 19.79 | 32.51 | 60 | 24 |
| q=75% | 0.90 | 1.00 | 31.3 | 286.2 | 21.01 | 47.59 | 86.37 | 84 | 19 |

### Branch balance (realized per-token traffic / measured bandwidths)

| Policy | avg q | miss rate | T_CPU ms | T_GPU ms | max(T_CPU,T_GPU) ms | imbalance |
|---|--:|--:|--:|--:|--:|--:|
| q* adaptive | 0.00 | 1.000 | 35.62 | 0.00 | 35.62 | 1.000 |
| q=m (GPU) | 1.06 | 0.265 | 0.00 | 50.13 | 50.13 | 1.000 |
| q=0 (CPU) | 0.00 | 1.000 | 35.62 | 0.00 | 35.62 | 1.000 |
| q=25% | 0.05 | 0.570 | 19.87 | 2.35 | 19.87 | 0.882 |
| q=50% | 0.20 | 0.218 | 6.02 | 9.36 | 9.36 | 0.357 |
| q=75% | 0.90 | 0.250 | 0.88 | 42.65 | 42.65 | 0.979 |

## Workload: med_miss


### Performance and traffic

| Policy | avg q | miss m | CPU MB/tok | H2D MB/tok | tok/s | ms/tok | p95 ms | GPU util% | CPU util% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q* adaptive | 0.00 | 4.00 | 1270.7 | 0.0 | 24.06 | 41.58 | 44.40 | 21 | 41 |
| q=m (GPU) | 2.30 | 2.30 | 0.0 | 730.3 | 8.67 | 115.37 | 172.96 | 92 | 18 |
| q=0 (CPU) | 0.00 | 4.00 | 1270.7 | 0.0 | 24.18 | 41.35 | 44.04 | 20 | 41 |
| q=25% | 0.06 | 2.37 | 734.7 | 18.3 | 37.10 | 26.96 | 32.04 | 34 | 34 |
| q=50% | 0.82 | 2.12 | 413.7 | 259.9 | 21.34 | 46.86 | 72.12 | 74 | 24 |
| q=75% | 1.75 | 2.15 | 126.3 | 556.8 | 11.07 | 90.31 | 129.16 | 90 | 19 |

### Branch balance (realized per-token traffic / measured bandwidths)

| Policy | avg q | miss rate | T_CPU ms | T_GPU ms | max(T_CPU,T_GPU) ms | imbalance |
|---|--:|--:|--:|--:|--:|--:|
| q* adaptive | 0.00 | 1.000 | 35.62 | 0.00 | 35.62 | 1.000 |
| q=m (GPU) | 2.30 | 0.575 | 0.00 | 108.84 | 108.84 | 1.000 |
| q=0 (CPU) | 0.00 | 1.000 | 35.62 | 0.00 | 35.62 | 1.000 |
| q=25% | 0.06 | 0.593 | 20.60 | 2.72 | 20.60 | 0.868 |
| q=50% | 0.82 | 0.530 | 11.60 | 38.73 | 38.73 | 0.701 |
| q=75% | 1.75 | 0.538 | 3.54 | 82.98 | 82.98 | 0.957 |

## Workload: high_miss


### Performance and traffic

| Policy | avg q | miss m | CPU MB/tok | H2D MB/tok | tok/s | ms/tok | p95 ms | GPU util% | CPU util% |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| q* adaptive | 0.00 | 4.00 | 1270.7 | 0.0 | 24.17 | 41.37 | 43.92 | 21 | 41 |
| q=m (GPU) | 4.00 | 4.00 | 0.0 | 1270.7 | 4.96 | 201.55 | 207.76 | 95 | 18 |
| q=0 (CPU) | 0.00 | 4.00 | 1270.7 | 0.0 | 24.28 | 41.18 | 43.59 | 21 | 41 |
| q=25% | 0.23 | 2.93 | 858.1 | 73.5 | 29.37 | 34.05 | 43.45 | 44 | 34 |
| q=50% | 1.27 | 2.96 | 537.9 | 403.4 | 14.54 | 68.78 | 89.88 | 79 | 24 |
| q=75% | 2.32 | 3.11 | 249.2 | 737.5 | 8.29 | 120.66 | 136.79 | 90 | 20 |

### Branch balance (realized per-token traffic / measured bandwidths)

| Policy | avg q | miss rate | T_CPU ms | T_GPU ms | max(T_CPU,T_GPU) ms | imbalance |
|---|--:|--:|--:|--:|--:|--:|
| q* adaptive | 0.00 | 1.000 | 35.62 | 0.00 | 35.62 | 1.000 |
| q=m (GPU) | 4.00 | 1.000 | 0.00 | 189.37 | 189.37 | 1.000 |
| q=0 (CPU) | 0.00 | 1.000 | 35.62 | 0.00 | 35.62 | 1.000 |
| q=25% | 0.23 | 0.733 | 24.06 | 10.96 | 24.06 | 0.544 |
| q=50% | 1.27 | 0.741 | 15.08 | 60.12 | 60.12 | 0.749 |
| q=75% | 2.32 | 0.776 | 6.99 | 109.91 | 109.91 | 0.936 |
