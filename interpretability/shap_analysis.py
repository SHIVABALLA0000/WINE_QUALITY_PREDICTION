import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path

# -----------------------------
# Load artifacts
# -----------------------------
ARTIFACT_PATH = Path("../wine_artifacts/wine_quality_model.joblib")


bundle = joblib.load(ARTIFACT_PATH)
preprocessor = bundle["preprocessor"]
stack_model = bundle["model"]

# -----------------------------
# Load some DEV data
# (use DEV, never test)
# -----------------------------
from src.data_utils import load_data
X, y, _ = load_data()

X_sample = X.sample(1000, random_state=42)
X_sample_p = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

# -----------------------------
# Extract base models
# -----------------------------
rf = stack_model.named_estimators_["rf"]
xgb_model = stack_model.named_estimators_["xgb"]
et = stack_model.named_estimators_["et"]

# -----------------------------
# SHAP — Random Forest
# -----------------------------
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_sample_p)

shap.summary_plot(
    shap_values_rf,
    X_sample_p,
    feature_names=feature_names,
    show=False
)

# -----------------------------
# SHAP — XGBoost
# -----------------------------
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_sample_p)

shap.summary_plot(
    shap_values_xgb,
    X_sample_p,
    feature_names=feature_names,
    show=False
)

# -----------------------------
# SHAP — Extra Trees
# -----------------------------
explainer_et = shap.TreeExplainer(et)
shap_values_et = explainer_et.shap_values(X_sample_p)

shap.summary_plot(
    shap_values_et,
    X_sample_p,
    feature_names=feature_names,
    show=False
)

# -----------------------------
# Meta-learner interpretation
# -----------------------------
meta = stack_model.final_estimator_

coef_df = pd.DataFrame(
    meta.coef_,
    columns=[f"base_prob_{i}" for i in range(meta.coef_.shape[1])]
)

print("\nMeta-learner coefficients:")
print(coef_df.head())
