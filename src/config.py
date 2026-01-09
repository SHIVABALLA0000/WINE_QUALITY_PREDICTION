import os

RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 4
N_TRIALS_INNER = 5

BASE_PATH = "./wine_artifacts"
os.makedirs(BASE_PATH, exist_ok=True)

ARTIFACT_MODEL   = os.path.join(BASE_PATH, "wine_quality_model.joblib")
ARTIFACT_ENCODER = os.path.join(BASE_PATH, "label_encoder.joblib")
ARTIFACT_CARD    = os.path.join(BASE_PATH, "model_card.json")
ARTIFACT_REPORT  = os.path.join(BASE_PATH, "nestedcv_report.json")
