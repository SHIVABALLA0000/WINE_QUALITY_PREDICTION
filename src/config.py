import os

# -----------------------------
# Reproducibility & performance
# -----------------------------
RANDOM_STATE = 42
N_JOBS = -1
DEBUG_MODE = False


# -----------------------------
# Nested Cross-Validation
# -----------------------------
OUTER_CV_SPLITS = 4
INNER_CV_SPLITS = 3
CV_SHUFFLE = True


# -----------------------------
# Hyperparameter Tuning Budgets
N_TRIALS_RF = 5
N_TRIALS_XGB = 5
N_TRIALS_ET = 5



# -----------------------------
# Artifacts
# -----------------------------
BASE_PATH = os.path.join(".", "wine_artifacts")
os.makedirs(BASE_PATH, exist_ok=True)

ARTIFACT_MODEL   = os.path.join(BASE_PATH, "wine_quality_model.joblib")
ARTIFACT_ENCODER = os.path.join(BASE_PATH, "label_encoder.joblib")
ARTIFACT_CARD    = os.path.join(BASE_PATH, "model_card.json")
ARTIFACT_REPORT  = os.path.join(BASE_PATH, "nestedcv_report.json")

