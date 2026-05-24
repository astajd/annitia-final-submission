# Phase 3.5 — stack failure analysis

Diagnostic comparison only. We are not iterating the stacker further this phase.

## Headline

| candidate                  | OOF hep | OOF death | OOF weighted | LB    | OOF -> LB transfer |
|:---------------------------|--------:|----------:|-------------:|------:|-------------------:|
| phase3_current_state_v2    |  0.8378 |    0.9495 |       0.8713 | 0.88521 |             +0.014 |
| phase3_stacked_ensemble    |  0.8943 |    0.8355 |       0.8767 | 0.78676 |             -0.090 |


## Rank correlation between candidates (test)

- hepatic: **0.624**
- death:   **0.613**

Even on the test set the stack is materially different from current_state_v2 — substantial divergence implies the stacker has indeed learned something new, but on the LB that 'something new' is wrong.

## Stack composition

- hepatic: **97** components, DTH included: False
- death:   **98** components, DTH included: True

97 / 98 base learners is far more than the 12 unique base configs we trust. Many of those base learners are seed/feature-set duplicates of the same underlying model, which inflates the meta-model's degrees of freedom relative to the 47 hepatic events the meta-fold sees.

## Fold-level stack metrics

### hepatic

|             |   oof_cindex |   fold_mean |   fold_std |
|:------------|-------------:|------------:|-----------:|
| ridge       |       0.8943 |      0.9111 |     0.0403 |
| elastic_net |       0.8590 |      0.8824 |     0.0486 |

### death

|             |   oof_cindex |   fold_mean |   fold_std |
|:------------|-------------:|------------:|-----------:|
| ridge       |       0.8355 |      0.8349 |     0.0471 |
| elastic_net |       0.8090 |      0.8057 |     0.0617 |

## Base-learner instability (cindex_std > 0.10)

- 82 hepatic base learners with std > 0.10
- 2 death base learners with std > 0.10

Worst hepatic instability (top 10 by std):

| experiment                          | feature_set                  | endpoint   | model           |   cindex_mean |   cindex_std |   cindex_min |   n_folds_evaluated |
|:------------------------------------|:-----------------------------|:-----------|:----------------|--------------:|-------------:|-------------:|--------------------:|
| 007_full_high_risk                  | full_high_risk               | hepatic    | xgb_cox         |        0.7545 |       0.1506 |       0.4547 |                  15 |
| phase3_current_state_v2             | current_state_v2             | hepatic    | xgb_cox         |        0.7537 |       0.1461 |       0.4669 |                  15 |
| 005_NIT_plus_clinical_scores        | NIT_plus_clinical_scores     | hepatic    | xgb_cox         |        0.7715 |       0.1451 |       0.4680 |                  15 |
| 004_all_visits_longitudinal         | all_visits_longitudinal      | hepatic    | xgb_cox         |        0.7462 |       0.1433 |       0.4422 |                  15 |
| phase3_current_state_v2             | current_state_v2             | hepatic    | xgb_cox         |        0.7580 |       0.1430 |       0.4702 |                  15 |
| phase2_NIT_longitudinal_only        | NIT_longitudinal_only        | hepatic    | xgb_binary      |        0.7199 |       0.1390 |       0.5335 |                  15 |
| phase2_first_2y                     | first_2y                     | hepatic    | xgb_binary      |        0.6561 |       0.1380 |       0.3634 |                  15 |
| phase2_NIT_plus_scores_longitudinal | NIT_plus_scores_longitudinal | hepatic    | catboost_binary |        0.6954 |       0.1379 |       0.5090 |                  15 |
| phase2_aggressive_longitudinal      | aggressive_longitudinal      | hepatic    | xgb_cox         |        0.7596 |       0.1377 |       0.5020 |                  15 |
| phase2_NIT_longitudinal_only        | NIT_longitudinal_only        | hepatic    | xgb_cox         |        0.7350 |       0.1343 |       0.4773 |                  15 |

Worst death instability (top 10 by std):

| experiment                         | feature_set   | endpoint   | model           |   cindex_mean |   cindex_std |   cindex_min |   n_folds_evaluated |
|:-----------------------------------|:--------------|:-----------|:----------------|--------------:|-------------:|-------------:|--------------------:|
| phase2_first_3y                    | first_3y      | death      | catboost_binary |        0.7159 |       0.1051 |       0.4392 |                  15 |
| 001_baseline_v1_drop_missing_death | baseline_v1   | death      | catboost_binary |        0.6549 |       0.1011 |       0.4434 |                  25 |

## Likely failure mechanisms

1. **Meta-model capacity vs event count.** Ridge with 97-98 features and only 47 hepatic events sees the small folds essentially in the underdetermined regime. Spurious linear combinations of unstable base learners can fit OOF noise and look strong on every left-out fold but degrade on a fresh sample.

2. **Component duplication inflates effective degrees of freedom.** Many components are seed-twins (rank-corr > 0.95) or feature-set-twins (e.g. RSF on phase2_aggressive vs Phase 1 all_visits). The stacker treats them as 90+ independent inputs; in reality there are only ~10-15 truly different signals.

3. **Strict_time_aligned base OOFs polluted hepatic.** The Phase 1 strict_time_aligned hepatic xgb_cox sat at OOF 0.97 but transferred to LB 0.69. The stacker assigned it weight via OOF C-index, so any ensemble that consumes its OOF imports that exact LB miss.

4. **Death OOF dropped vs simple ensemble.** Stack death OOF 0.835 vs current_state_v2 death OOF 0.949. The meta-model regularizes hard (ridge alpha=2.0) and effectively averages too many low-individual death components — pulling the death rank away from the strong current_state_v2 RSF/xgb_cox towards mid-tier classifiers.

5. **DTH inputs.** DTH hepatic OOF 0.55 / death 0.63 — both below the 0.55 retention floor only on hepatic; included on death. With weight =0 effectively at our floor, but ridge still uses it as a regressor and may have shifted the death linear combination unhelpfully.

## Conclusions

- The stacker is *correctly* finding a high-OOF combination; the problem is that the high OOF lives in the leak-susceptible part of the component pool.
- Do **not** iterate the stack. Phase 3.5 focuses on simplifying current_state_v2 instead.
- If we ever revisit stacking, the only safe path is: (a) drop strict_time_aligned components from the pool, (b) cap each model-family's contribution, (c) use 5-10 carefully-curated base learners, not 97.
