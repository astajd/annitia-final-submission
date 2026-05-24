# References for the explainability section

Each entry lists: title, authors, year, journal/source, DOI or stable URL,
the **specific claim** it supports, and what range of liver-stiffness
thresholds the citation directly supports. The package's clinical
language is matched to these citations: any sentence that goes beyond
what a citation directly supports has been softened.

## A. Baveno VII (portal hypertension thresholds)

**1. de Franchis R, Bosch J, Garcia-Tsao G, Reiberger T, Ripoll C, et al.**
"Baveno VII — Renewing consensus in portal hypertension." *Journal of
Hepatology* 2022; 76(4): 959–974.
DOI: https://doi.org/10.1016/j.jhep.2021.12.022

- **Claim supported (directly):** the Baveno VII "rule of five" for VCTE
  liver stiffness measurement (LSM) — 10, 15, 20, 25 kPa — to denote
  progressively higher relative risks of decompensation and liver-related
  death independently of etiology of chronic liver disease; rule-out CSPH
  via LSM ≤15 kPa plus platelets ≥150 × 10⁹/L; rule-in CSPH via LSM
  ≥25 kPa; LSM ≥20 kPa as a trigger for variceal screening.
- **What this citation DOES NOT support:** that 12 kPa is the Baveno VII
  CSPH rule-in threshold. Baveno VII's CSPH rule-in threshold is **≥25 kPa**,
  not 12 kPa. Any claim of "12 kPa = Baveno VII CSPH rule-in" is not
  supported by this reference and is **not** used in this package or in
  the finalized docs.

**Corrigendum**: PubMed PMID 35431106
(https://pubmed.ncbi.nlm.nih.gov/35431106/).

## B. MASLD / NAFLD liver-stiffness thresholds

**2. Rinella ME, Neuschwander-Tetri BA, Siddiqui MS, Abdelmalek MF,
Caldwell S, Barb D, Kleiner DE, Loomba R.**
"AASLD Practice Guidance on the clinical assessment and management of
nonalcoholic fatty liver disease." *Hepatology* 2023; 77(5): 1797–1835.
DOI: https://doi.org/10.1097/HEP.0000000000000323

- **Claim supported (directly):** A three-tier VCTE risk
  stratification for NAFLD/MASLD: LSM <8 kPa rules out advanced fibrosis;
  LSM 8–12 kPa is the indeterminate / fibrotic-NASH zone; **LSM >12 kPa is
  associated with a high likelihood of advanced fibrosis** and is the
  high-risk category recommended for additional risk stratification.
- **What this citation supports for our package:** the 12 kPa value used
  as the LS gate is a clinically interpretable MASLD / NAFLD
  **advanced-fibrosis / high-risk** liver-stiffness threshold under the
  AASLD 2023 guidance. The package's wording is matched to this support:
  "a clinically interpretable MASLD/NIT high-risk or advanced-fibrosis
  threshold."
- **What this citation DOES NOT support:** any portal-hypertension
  rule-in interpretation of the 12 kPa value, or any decompensation /
  death-rate causal claim from this threshold alone.

**3. European Association for the Study of the Liver (EASL),
European Association for the Study of Diabetes (EASD), European
Association for the Study of Obesity (EASO).**
"EASL–EASD–EASO Clinical Practice Guidelines on the management of
metabolic dysfunction-associated steatotic liver disease (MASLD)."
*Journal of Hepatology* 2024; 81(3): 492–542.
DOI: https://doi.org/10.1016/j.jhep.2024.04.031

Executive summary also in *Diabetologia* 2024; 67(11): 2375–2392.
DOI: https://doi.org/10.1007/s00125-024-06196-3
(PMC: PMC11519095, PMC11519244 for the publisher correction).

- **Claim supported (directly):** for adults with MASLD, LSM by VCTE
  **≤15 kPa plus platelets ≥150 × 10⁹/L** may be used to rule out CSPH;
  LSM **≥20 kPa** is a criterion for recommending upper-GI endoscopy
  screening for varices; LSM ≥8 kPa is referenced as a significant-fibrosis
  cutpoint.
- **What this citation supports for our package:** an independent
  contemporary European consensus that places 12 kPa within the broader
  risk-stratification range used for MASLD NITs (between the
  significant-fibrosis cutpoint at 8 kPa and the CSPH-rule-out cutpoint
  at 15 kPa).
- **What this citation DOES NOT support:** 12 kPa as a guideline-named
  rule-in threshold for advanced fibrosis or CSPH. The 15 kPa value is
  for CSPH rule-OUT (combined with platelets), not rule-IN.

## C. Rank-based / concordance-index interpretation

**4. Harrell FE Jr, Lee KL, Mark DB.**
"Multivariable prognostic models: issues in developing models, evaluating
assumptions and adequacy, and measuring and reducing errors." *Statistics
in Medicine* 1996; 15(4): 361–387.
DOI: https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4
PubMed: PMID 8668867.

- **Claim supported (directly):** the concordance index (C-index) for
  censored survival data is rank-based — only the ordering of predicted
  risks matters, not their scale or calibration.
- **What this citation supports for our package:** the explicit decision
  to emit rank-only predictions (`scipy.stats.rankdata`) and to optimize
  rank-aggregated blends (mean-of-ranks across CV folds; rank-weighted
  blends in the final death and hepatic recipes). Calibration is not
  required for the challenge metric.

## D. Citation-claim discipline applied here

The clinical-wording rule from the brief is:

> The claim must shrink to match the citation.

Applying this rule across the package:

| package claim | matched citation | wording used |
|---|---|---|
| 12 kPa is a clinically interpretable MASLD/NIT high-risk or advanced-fibrosis threshold | Rinella et al. 2023 (AASLD) | "a clinically interpretable MASLD/NIT high-risk or advanced-fibrosis threshold" |
| 12 kPa is the Baveno VII CSPH rule-in threshold | NONE — Baveno VII's CSPH rule-in is ≥25 kPa | **not used; explicitly disclaimed** in METHODOLOGY.md §29-31, LEAKAGE_AUDIT.md §13, LIMITATIONS.md §15 |
| Risk strata for cACLD/CSPH under Baveno VII | de Franchis et al. 2022 (Baveno VII) | rule-of-five (10/15/20/25 kPa); CSPH rule-out ≤15 kPa + plt ≥150; CSPH rule-in ≥25 kPa |
| 12 kPa sits within the broader MASLD NIT risk-stratification range | EASL/EASD/EASO 2024 + Rinella 2023 | "a threshold within the range used for MASLD/NIT risk stratification" (safe fallback wording if reviewer prefers an EASL-only citation) |
| Rank-only outputs are appropriate for the C-index objective | Harrell et al. 1996 | "rank-based C-index objective; submissions emitted as `scipy.stats.rankdata`" |

If a reviewer concludes that the AASLD 2023 citation should not be
read as a clinical "threshold" recommendation (it is a risk-stratification
band, not a binary decision rule), the AASLD-anchored wording above can
be replaced by the EASL-only safer wording with no change to the
underlying claim.
