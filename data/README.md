# data/

**Raw challenge data are NOT redistributed with this repo.** The ANNITIA /
Trustii.io / IHU ICAN dataset is licensed by the data provider; reviewers
must obtain it from the official source.

## Required files

Place the official challenge files into `data/raw/` relative to this repo
root.

| filename | expected rows incl. header | purpose |
|---|---:|---|
| `train.csv` | 1254 | optional upstream-reference inspection / historical scripts |
| `test.csv` | 424 | optional upstream-reference inspection / historical scripts |
| `dictionary.csv` | — | feature definitions / documentation |
| `hello_world_submission.csv` | — | sample submission schema |

Only `train.csv` and `test.csv` are needed for optional upstream-reference
inspection and for the historical multi-candidate producer. The authoritative
slot1 verification path does not require raw challenge data.

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

## What if a reviewer cannot obtain the raw data?

The byte-frozen submission file in `../frozen/slot1_prediction.csv`
and its `SHA256SUMS` / `verify.sh` are independent of raw data and can be
verified without it. The authoritative slot1 verification script,
`../pipeline_full/runnable/build_slot1_only.py`, also runs without raw data.

Raw data are needed only for optional upstream-reference inspection and for
the historical multi-candidate producer (`build_final_3.py`), not for the
authoritative slot1 verification path.
