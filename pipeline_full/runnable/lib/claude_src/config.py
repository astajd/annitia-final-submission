"""Project-wide constants. Single source of truth for column names, paths, defaults.

Adapted for the assembled annitia-final-submission repo. Raw data location is
resolved in priority order:

  1. ANNITIA_DATA_ROOT environment variable (absolute path to data/raw, or to
     the directory containing data/raw).
  2. <repo_root>/data/raw, where <repo_root> is two levels above this file's
     pipeline_full/runnable/lib/claude_src/ location, i.e. the assembled-repo
     root.

Raw challenge data are expected under data/raw/ for full retraining; 
the fast slot1 verification path does not require raw data.
"""
import os
from pathlib import Path

# This file lives at:
#   <repo_root>/pipeline_full/runnable/lib/claude_src/config.py
# so the assembled-repo root is parents[4].
ROOT = Path(__file__).resolve().parents[4]


def _resolve_data_raw() -> Path:
    env = os.environ.get("ANNITIA_DATA_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if (p / "train.csv").exists():
            return p
        if (p / "raw" / "train.csv").exists():
            return p / "raw"
        if (p / "data" / "raw" / "train.csv").exists():
            return p / "data" / "raw"
        return p
    return ROOT / "data" / "raw"


DATA_RAW = _resolve_data_raw()
DATA_PROCESSED = ROOT / "data" / "processed"
EXPERIMENTS = ROOT / "experiments"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"

TRAIN_CSV = DATA_RAW / "train.csv"
TEST_CSV = DATA_RAW / "test.csv"

ID_COLS_TRAIN = ["patient_id_anon"]
ID_COLS_TEST = ["trustii_id", "patient_id_anon"]

TARGET_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

STATIC_FEATURES = [
    "gender", "T2DM", "Hypertension", "Dyslipidaemia",
    "bariatric_surgery", "bariatric_surgery_age",
]

LONGITUDINAL_VARS = [
    "Age", "BMI",
    "alt", "ast", "ggt", "bilirubin", "plt",
    "gluc_fast", "triglyc", "chol",
    "fibs_stiffness_med_BM_1",
    "fibrotest_BM_2",
    "aixp_aix_result_BM_3",
]
NIT_VARS = ["fibs_stiffness_med_BM_1", "fibrotest_BM_2", "aixp_aix_result_BM_3"]
MAX_VISITS = 22

CV_N_SPLITS = 5
CV_N_REPEATS = 10
CV_RANDOM_STATE = 42

HEPATIC_WEIGHT = 0.7
DEATH_WEIGHT = 0.3
