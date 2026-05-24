# Phase 3 submission candidates

All Phase 3 candidates produced by `python -m src.build_phase3_submissions`. 
OOF figures use the same hepatic-stratified repeated K-fold as Phase 1/2. 
Rank correlations are versus the current public best (`phase2_aggressive_longitudinal`, LB 0.86813).

| label                       | promoted   | leakage_tag                                                                   |   hep_cindex |   death_cindex |   weighted_score_cv |   rank_corr_pub_hep |   rank_corr_pub_death | submission_csv                                                                           |
|:----------------------------|:-----------|:------------------------------------------------------------------------------|-------------:|---------------:|--------------------:|--------------------:|----------------------:|:-----------------------------------------------------------------------------------------|
| phase3_current_state_v2     | True       | moderate-high (uses current age / last observed age and missingness, by spec) |       0.8378 |         0.9495 |              0.8713 |              0.8495 |                0.9552 | gpt/submissions/20260427_1220_phase3_current_state_v2.csv |
| phase3_stacked_ensemble     | True       | mixed — inherits from base components; meta-model is rank-only.               |       0.8943 |         0.8355 |              0.8767 |              0.6893 |                0.5804 | gpt/submissions/20260427_1220_phase3_stacked_ensemble.csv |
| phase3_discrete_time_hazard | False      | nan                                                                           |       0.5459 |         0.6303 |              0.5712 |            nan      |              nan      | nan                                                                                      |

## Notes

- Submit the candidate with the best **defensible** OOF score whose rank correlation against the public best is below ~0.95 (otherwise it provides no new information on the LB).

- The stacked candidate is the most informative diversity probe; the current-state-v2 candidate is the cleanest single-feature-set probe.

- DTH is only promoted to a submission if its weighted CV >= 0.80; below that we keep its OOFs in the stacker but do not submit alone.
