# Forum intel

Captured: 2026-04-27. Source: organizer post on the Trustii.io / IHU ICAN
challenge forum, 2026-04-14.

These quotes change the strategic calculus on permissive longitudinal features
that include post-event observations. Phase 1's leakage audit found that
trajectory features over a per-row outcome cutoff are leaky (`strict_time_aligned`
— still excluded as Tier 5). What the organizer is *blessing* here is the
honest-but-permissive case: features that summarize all-visit history with no
per-row outcome dependence. That is `longitudinal_summary` and
`longitudinal_plus_meta`, currently Tier 4 in our framework.

## Quote 1 — permissive longitudinal features explicitly allowed

> "All accumulated visit history is naturally available... longitudinal
> features entirely valid. Designing features that disentangle this thing is
> precisely the kind of smart approach that will be rewarded in subjective
> scoring."

## Quote 2 — test set has post-event-equivalent observations

> "You should handle this scenario in your modeling, and during the notebook
> review, we will consider the best approach in subjective criteria scoring."

(Annotation: confirms test patients have visits that, in the train-set
analogue, would be post-event for affected patients. The competition is
explicitly asking for methodology that handles this — and rewarding the
"smart approach" in qualitative review.)

## Implications

1. **Permissive longitudinal features (Tier 4) are organizer-blessed for the
   quantitative track.** Phase 2c expands the Optuna candidate menu with
   `longitudinal_summary` and `longitudinal_plus_meta` configurations.
2. **The "smart approach" framing rewards methodology, not just C-index.**
   Phase 2d builds landmark features (Tier 2: clean longitudinal at a fixed
   reference time) — explicitly handling the post-event visit issue without
   per-row outcome dependence.
3. **Two-track submission strategy.** Phase 2e produces two ship candidates:
   `phase2_honest_ensemble.csv` (Tier 1 + Tier 2 only — defensible in
   notebook review) and `phase2_permissive_ensemble.csv` (Tier 4 — exploits
   the organizer's permission).
4. **Strict-leakage probe (`phase2_strict_leaky_probe.csv` LB 0.692) result
   stands.** Tier 5 features still don't transfer; that test was a separate
   question from "are Tier 4 features valid" and the organizer's quotes do
   not change its interpretation.

## Pending

User indicated they will paste the verbatim forum text if needed. The block
quotes above are reproduced from the user's message; the surrounding context
(thread title, exact UTC timestamp, post URL) is not yet captured here.
