import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# interpretability/shap_analysis.py

import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
ARTIFACT_PATH = Path("wine_artifacts/wine_quality_model.joblib")
OUTPUT_DIR = Path("wine_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# Load trained artifacts
# -----------------------------
bundle = joblib.load(ARTIFACT_PATH)

preprocessor = bundle["preprocessor"]
stack_model = bundle["model"]

# -----------------------------
# Load DEV data only
# -----------------------------
# IMPORTANT: never use test data here
from src.data_utils import load_data

X, y, _ = load_data()

X_sample = X.sample(1000, random_state=42)
X_sample_p = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

np.save(OUTPUT_DIR / "feature_names.npy", feature_names)

# -----------------------------
# Extract base learners
# -----------------------------
rf = stack_model.named_estimators_["rf"]
xgb_model = stack_model.named_estimators_["xgb"]
et = stack_model.named_estimators_["et"]

# -----------------------------
# SHAP — Random Forest
# -----------------------------
explainer_rf = shap.TreeExplainer(rf)
shap_rf = explainer_rf.shap_values(X_sample_p)
np.save(OUTPUT_DIR / "shap_rf.npy", shap_rf)

# -----------------------------
# SHAP — XGBoost
# -----------------------------
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_xgb = explainer_xgb.shap_values(X_sample_p)
np.save(OUTPUT_DIR / "shap_xgb.npy", shap_xgb)

# -----------------------------
# SHAP — Extra Trees
# -----------------------------
explainer_et = shap.TreeExplainer(et)
shap_et = explainer_et.shap_values(X_sample_p)
np.save(OUTPUT_DIR / "shap_et.npy", shap_et)

# -----------------------------
# Meta-learner coefficients
# -----------------------------
meta = stack_model.final_estimator_

coef_df = pd.DataFrame(
    meta.coef_,
    columns=[f"base_prob_{i}" for i in range(meta.coef_.shape[1])]
)

coef_df.to_csv(OUTPUT_DIR / "meta_learner_coefficients.csv", index=False)

print("✅ SHAP artifacts generated successfully")
print("Saved:")
print("- shap_rf.npy")
print("- shap_xgb.npy")
print("- shap_et.npy")
print("- feature_names.npy")
print("- meta_learner_coefficients.csv")
