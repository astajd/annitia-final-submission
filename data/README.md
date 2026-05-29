# data/

**The official challenge data are included in this private review repository,
under `data/raw/`, following organizer clarification.** The ANNITIA /
Trustii.io / IHU ICAN dataset is licensed by the data provider. If this
repository is later made public, redistribution of the dataset should be
reconfirmed with the data provider.

## Included files

The following official challenge files are present in `data/raw/` relative to
this repo root.

| filename | expected rows incl. header | purpose |
|---|---:|---|
| `train.csv` | 1254 | training labels/features for full from-raw retraining (Path A) and upstream reference |
| `test.csv` | 424 | test features for full from-raw retraining (Path A) and upstream reference |
| `dictionary.csv` | — | feature definitions / documentation |
| `hello_world_submission.csv` | — | sample submission schema |

These files are required for Path A (full from-raw retraining,
`retrain_all_from_raw.sh`), for the upstream reference code, and for the
historical multi-candidate producer. The fast cached-output verification path
(Path B, `build_slot1_only.py`) does not require raw challenge data.

## Schema

- `train.csv`: 1253 patient rows. ID column `patient_id_anon`. Static
  features, longitudinal `<var>_v<i>` columns for visits 1–22, and the two
  outcomes (`evenements_hepatiques_majeurs` + age column, `death` + age column).

- `test.csv`: 423 patient rows. ID columns `trustii_id` and
  `patient_id_anon`. Same visit / static feature schema; no outcome columns.

- `hello_world_submission.csv`: sample submission with the required output
  schema `trustii_id, risk_hepatic_event, risk_death`.

## Where the loader looks

`pipeline_full/runnable/lib/claude_src/config.py` resolves the raw-data
location in this order:

1. `$ANNITIA_DATA_ROOT` (absolute path to `data/raw`, or to a directory
   containing `data/raw`).
2. `<repo_root>/data/raw/` (the default; this is where to drop the files
   when running from the assembled repo).

## Verification without raw data

The byte-frozen submission file in `../frozen/slot1_prediction.csv`
and its `SHA256SUMS` / `verify.sh` are independent of raw data and can be
verified without it. The fast cached-output verification script (Path B),
`../pipeline_full/runnable/build_slot1_only.py`, also runs without raw data.

The raw data in `data/raw/` are needed for Path A (full from-raw retraining,
`../pipeline_full/runnable/retrain_all_from_raw.sh`), for upstream-reference
inspection, and for the historical multi-candidate producer
(`build_final_3.py`).
