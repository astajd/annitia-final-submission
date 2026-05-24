# Phase 2 submission candidates

All four candidates produced by `python -m src.build_phase2_submissions`. 
OOF figures are *cross-validated* on the patient-level repeated stratified split shared across the contest. 
These are local upper-bounds; the leaderboard delta indicates the leakage premium.

| label                          | leakage_tag                              | hepatic_method   |   hepatic_oof | death_method   |   death_oof |   weighted_score_cv | submission                                                                                      |
|:-------------------------------|:-----------------------------------------|:-----------------|--------------:|:---------------|------------:|--------------------:|:------------------------------------------------------------------------------------------------|
| phase2_robust_longitudinal     | low/moderate                             | greedy           |        0.8398 | greedy         |      0.9120 |              0.8614 | gpt/submissions/20260426_1831_phase2_robust_longitudinal.csv     |
| phase2_aggressive_longitudinal | moderate/high                            | greedy           |        0.8518 | greedy         |      0.9523 |              0.8819 | gpt/submissions/20260426_1831_phase2_aggressive_longitudinal.csv |
| phase2_clean_clinical_NIT      | low/moderate                             | greedy           |        0.8341 | greedy         |      0.9066 |              0.8559 | gpt/submissions/20260426_1832_phase2_clean_clinical_NIT.csv      |
| phase2_best_oof_ensemble       | mixed (includes high-leakage components) | seed_bagged      |        0.9738 | seed_bagged    |      0.9527 |              0.9675 | gpt/submissions/20260426_1832_phase2_best_oof_ensemble.csv       |

## Notes

- `phase2_robust_longitudinal` is the **first leaderboard probe** of the Phase 2 cycle. It avoids the explicit follow-up proxies that we now know inflate CV.

- `phase2_clean_clinical_NIT` is the **qualitative submission**: every component is interpretable to a hepatologist (NIT trajectories, FIB-4/APRI, demographics).

- `phase2_aggressive_longitudinal` is the **leakage probe** — submit only after the robust candidate so we can compare LB deltas.

- `phase2_best_oof_ensemble` is exploratory: it is the highest local CV but mixes leaky Phase 1 models. Do not submit unless investigating the local-vs-LB gap further.
