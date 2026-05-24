# Phase 3.8 — TabPFN experiment

TabPFN version 2.2.1, device=cpu (NVIDIA driver 12020 too old for the cu130 wheel; v2.2.1 is the last release with open weights — current 7.x requires a license token from priorlabs.ai which is unavailable in this environment).

Reference: `phase3_5_current_state_v3_hepatic_focused` LB **0.89147**, OOF hepatic=0.8415 / death=0.9496 / weighted=0.8739.

## TabPFN per-run CV

| run_dir                                                  | feature_set             | endpoint   | preprocess    | select_topk   | n_features_input   |   cindex_mean |   cindex_std |   cindex_min |   rho_oof_with_v3 |   rho_test_with_v3 |
|:---------------------------------------------------------|:------------------------|:-----------|:--------------|:--------------|:-------------------|--------------:|-------------:|-------------:|------------------:|-------------------:|
| D_csv2_full__death__raw__top100                          | D_csv2_full             | death      | raw           | 100           | (stored)           |        0.7768 |       0.0816 |       0.6020 |           -0.0134 |             0.0605 |
| E_csv2_no_visit_history__death__raw__topNone             | E_csv2_no_visit_history | death      | raw           | None          | (stored)           |        0.7559 |       0.0726 |       0.6350 |            0.0173 |             0.0955 |
| B_biomarker_only__death__raw__topNone                    | B_biomarker_only        | death      | raw           | None          | (stored)           |        0.7527 |       0.0806 |       0.6362 |            0.0345 |             0.0903 |
| D_csv2_full__death__raw__top50                           | D_csv2_full             | death      | raw           | 50            | (stored)           |        0.7507 |       0.0809 |       0.6281 |           -0.0002 |             0.0578 |
| A_NIT_plus_scores__death__raw__topNone                   | A_NIT_plus_scores       | death      | raw           | None          | (stored)           |        0.7226 |       0.0778 |       0.6223 |            0.0732 |             0.1604 |
| C_hepatic_NIT_scores__death__raw__topNone                | C_hepatic_NIT_scores    | death      | raw           | None          | (stored)           |        0.7226 |       0.0778 |       0.6223 |            0.0732 |             0.1604 |
| D_csv2_full__hepatic__raw__top50                         | D_csv2_full             | hepatic    | raw           | 50            | (stored)           |        0.7560 |       0.1344 |       0.5355 |            0.5800 |             0.6146 |
| B_biomarker_only__hepatic__raw__topNone                  | B_biomarker_only        | hepatic    | raw           | None          | (stored)           |        0.7486 |       0.1217 |       0.5458 |            0.5941 |             0.6843 |
| A_NIT_plus_scores__hepatic__raw__topNone                 | A_NIT_plus_scores       | hepatic    | raw           | None          | (stored)           |        0.7476 |       0.1374 |       0.5060 |            0.4940 |             0.5487 |
| C_hepatic_NIT_scores__hepatic__raw__topNone              | C_hepatic_NIT_scores    | hepatic    | raw           | None          | (stored)           |        0.7476 |       0.1374 |       0.5060 |            0.4940 |             0.5487 |
| E_csv2_no_visit_history__hepatic__raw__topNone           | E_csv2_no_visit_history | hepatic    | raw           | None          | (stored)           |        0.7460 |       0.1135 |       0.5793 |            0.6029 |             0.6891 |
| E_csv2_no_visit_history__hepatic__rank_quantile__topNone | E_csv2_no_visit_history | hepatic    | rank_quantile | None          | (stored)           |        0.7442 |       0.1175 |       0.5721 |            0.4890 |             0.5836 |
| D_csv2_full__hepatic__raw__top100                        | D_csv2_full             | hepatic    | raw           | 100           | (stored)           |        0.7236 |       0.1235 |       0.5641 |            0.5495 |             0.5427 |

## Best TabPFN per endpoint (selected for blends)

- hepatic best: `D_csv2_full__hepatic__raw__top50` C=0.7560 (rank-corr w/ v3 OOF 0.580, test 0.615)
- death best:   `D_csv2_full__death__raw__top100` C=0.7768 (rank-corr w/ v3 OOF -0.013, test 0.060)

## Blends with v3 (rank-space)

| blend               |   alpha | side     |   hep_oof |   death_oof |   weighted_oof |
|:--------------------|--------:|:---------|----------:|------------:|---------------:|
| alpha=0.95_hep_only |  0.9500 | hep_only |    0.8389 |      0.9496 |         0.8721 |
| alpha=0.95_dea_only |  0.9500 | dea_only |    0.8415 |      0.9499 |         0.8740 |
| alpha=0.95_both     |  0.9500 | both     |    0.8389 |      0.9499 |         0.8722 |
| alpha=0.9_hep_only  |  0.9000 | hep_only |    0.8366 |      0.9496 |         0.8705 |
| alpha=0.9_dea_only  |  0.9000 | dea_only |    0.8415 |      0.9488 |         0.8737 |
| alpha=0.9_both      |  0.9000 | both     |    0.8366 |      0.9488 |         0.8703 |
| alpha=0.85_hep_only |  0.8500 | hep_only |    0.8341 |      0.9496 |         0.8687 |
| alpha=0.85_dea_only |  0.8500 | dea_only |    0.8415 |      0.9474 |         0.8733 |
| alpha=0.85_both     |  0.8500 | both     |    0.8341 |      0.9474 |         0.8681 |
| alpha=0.8_hep_only  |  0.8000 | hep_only |    0.8302 |      0.9496 |         0.8660 |
| alpha=0.8_dea_only  |  0.8000 | dea_only |    0.8415 |      0.9447 |         0.8725 |
| alpha=0.8_both      |  0.8000 | both     |    0.8302 |      0.9447 |         0.8646 |
| alpha=0.7_hep_only  |  0.7000 | hep_only |    0.8230 |      0.9496 |         0.8610 |
| alpha=0.7_dea_only  |  0.7000 | dea_only |    0.8415 |      0.9389 |         0.8707 |
| alpha=0.7_both      |  0.7000 | both     |    0.8230 |      0.9389 |         0.8578 |
| alpha=0.5_hep_only  |  0.5000 | hep_only |    0.8049 |      0.9496 |         0.8483 |
| alpha=0.5_dea_only  |  0.5000 | dea_only |    0.8415 |      0.9164 |         0.8640 |
| alpha=0.5_both      |  0.5000 | both     |    0.8049 |      0.9164 |         0.8384 |

## Promotion decision

**Not promoted.** None of the TabPFN blends improved weighted OOF by ≥ 0.002 or hepatic OOF by ≥ 0.003 over v3.

TabPFN's standalone hepatic OOF maxes at 0.756 vs v3 0.842 (Δ = -0.086) and death OOF at 0.777 vs v3 0.950 (Δ = -0.173). High individual gap dominates the rank-space blend: the OOF rank-corr with v3 is high enough that any TabPFN-weighted variant averages toward v3 anyway, while pulling the strong v3 signal toward TabPFN's weaker estimates.

## Notes

- 13 of 14 planned TabPFN configurations completed; the final `E_csv2_no_visit_history__death__rank_quantile__topNone` was killed because the rank_quantile preprocess on 215 features was projected to take ~30+ min and earlier runs already showed TabPFN was uncompetitive on this dataset.
- All TabPFN runs use `n_estimators=4` (default ensemble) on CPU. Increasing it would likely improve OOF by < 0.005 and not close the ~0.10–0.17 gap to v3.
- Risk score = predicted positive-class probability from TabPFN classifier trained on binary event-occurrence labels; evaluated against the survival C-index of the original endpoint.
- Top-k feature selection is fold-internal (ANOVA-F on training rows).
- No target-derived features used in any run.
