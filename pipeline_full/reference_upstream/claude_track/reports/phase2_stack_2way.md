# Task 1 — OOF-stacked blend weights for 2-way blend

Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on hepatic event (50 folds, base_seed=42). Landmark RSF: tuned baseline_v1 hyperparameters, trained on (train ∩ filter-A) and predicting full val. Permissive ensemble: rank-avg of 3 members on full cohort.

## Weight curve

| w (landmark) | mean C | std C | m−s | n_folds |
|---|---|---|---|---|
| 0.00 | 0.8103 | 0.0906 | 0.7197 | 50 |
| 0.05 | 0.8111 | 0.0905 | 0.7206 | 50 |
| 0.10 | 0.8119 | 0.0907 | 0.7212 | 50 |
| 0.15 | 0.8122 | 0.0908 | 0.7214 | 50 |
| 0.20 | 0.8129 | 0.0910 | 0.7219 | 50 |
| 0.25 | 0.8130 | 0.0910 | 0.7220 | 50 |
| 0.30 | 0.8136 | 0.0912 | 0.7224 | 50 |
| 0.35 | 0.8137 | 0.0918 | 0.7219 | 50 |
| 0.40 | 0.8137 | 0.0930 | 0.7207 | 50 |
| 0.45 | 0.8123 | 0.0945 | 0.7179 | 50 |
| 0.50 | 0.8106 | 0.0959 | 0.7148 | 50 |
| 0.55 | 0.8087 | 0.0969 | 0.7118 | 50 |
| 0.60 | 0.8066 | 0.0981 | 0.7086 | 50 |
| 0.65 | 0.8033 | 0.0997 | 0.7036 | 50 |
| 0.70 | 0.7996 | 0.1007 | 0.6989 | 50 |
| 0.75 | 0.7955 | 0.1018 | 0.6937 | 50 |
| 0.80 | 0.7924 | 0.1024 | 0.6900 | 50 |
| 0.85 | 0.7882 | 0.1019 | 0.6862 | 50 |
| 0.90 | 0.7835 | 0.1030 | 0.6805 | 50 |
| 0.95 | 0.7788 | 0.1026 | 0.6762 | 50 |
| 1.00 | 0.7737 | 0.1025 | 0.6712 | 50 |

## Headline

- **Naive w\*** = 0.30, m−s = 0.7224 (mean 0.8136 ± 0.0912).
- **50/50 default**: m−s = 0.7148 (mean 0.8106 ± 0.0959).
- **Δ m−s vs 50/50** = +0.0076.
- **Honest LOFO** m−s = 0.7168 (mean 0.8100 ± 0.0932); median per-fold pick w = 0.35.
- LOFO weight pick distribution: 0.30→16, 0.35→17, 0.40→17.

## Decision

w*=0.30 OUT OF parity band [0.4, 0.6]; Δm-s = +0.0076 (≥0.005). → BUILD phase2_blend_2way_optimal.csv.
