"""Project paths and global config constants.

All paths are computed from the location of this file so the project can be
moved without breaking. No hardcoded user-specific paths.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_CONFIGS = EXPERIMENTS_DIR / "configs"
EXPERIMENT_OUTPUTS = EXPERIMENTS_DIR / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Raw filenames as delivered by Trustii.
TRAIN_CSV = DATA_RAW / "DB-1773398340961.csv"
TEST_CSV = DATA_RAW / "DB-1773398340961-test.csv"
DICT_CSV = DATA_RAW / "Dictionary-1775552221412.csv"
SAMPLE_SUBMISSION_CSV = DATA_RAW / "hello_world_submission-1773575379610.csv"

# Endpoint columns in the train CSV.
HEPATIC_EVENT_COL = "evenements_hepatiques_majeurs"
HEPATIC_EVENT_AGE_COL = "evenements_hepatiques_age_occur"
DEATH_EVENT_COL = "death"
DEATH_EVENT_AGE_COL = "death_age_occur"

PATIENT_ID_COL = "patient_id_anon"
TRUSTII_ID_COL = "trustii_id"

# Submission column names.
SUB_HEPATIC_COL = "risk_hepatic_event"
SUB_DEATH_COL = "risk_death"

# Score weighting from the competition rules.
WEIGHT_HEPATIC = 0.7
WEIGHT_DEATH = 0.3

# Default repeated-CV configuration.
N_SPLITS = 5
N_REPEATS = 10
RANDOM_SEED = 20260426

# Small positive epsilon used when survival times must be strictly positive.
TIME_EPSILON = 1e-3

# Maximum visit number observed in the data (Age_v1 .. Age_v22).
MAX_VISITS = 22


def ensure_dirs() -> None:
    """Create the persistent directories if they don't exist yet."""
    for p in (
        DATA_RAW,
        DATA_PROCESSED,
        EXPERIMENT_CONFIGS,
        EXPERIMENT_OUTPUTS,
        REPORTS_DIR,
        SUBMISSIONS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)
