"""Project-wide constants. Single source of truth for column names, paths, defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
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
    "fibs_stiffness_med_BM_1",   # FibroScan
    "fibrotest_BM_2",            # FibroTest
    "aixp_aix_result_BM_3",      # Aixplorer
]
NIT_VARS = ["fibs_stiffness_med_BM_1", "fibrotest_BM_2", "aixp_aix_result_BM_3"]
MAX_VISITS = 22

# CV protocol — pre-registered
CV_N_SPLITS = 5
CV_N_REPEATS = 10
CV_RANDOM_STATE = 42

HEPATIC_WEIGHT = 0.7
DEATH_WEIGHT = 0.3
