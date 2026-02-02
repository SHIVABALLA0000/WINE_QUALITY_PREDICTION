# streamlit_app/pages/3_SHAP_Explainability.py

import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(layout="wide")

st.title("🔍 SHAP Explainability (Feature Attribution)")
st.markdown("""
This page explains **why the model makes predictions**, using
**SHAP (SHapley Additive exPlanations)**.

SHAP values are:
- Computed **offline**
- Loaded here for **transparent inspection**
- Used for **debugging, reporting, and trust**
""")

# -----------------------------
# Load SHAP artifacts
# -----------------------------
ARTIFACT_DIR = Path("wine_artifacts")

try:
    feature_names = np.load(ARTIFACT_DIR / "feature_names.npy", allow_pickle=True)
    shap_rf = np.load(ARTIFACT_DIR / "shap_rf.npy", allow_pickle=True)
    shap_xgb = np.load(ARTIFACT_DIR / "shap_xgb.npy", allow_pickle=True)
    shap_et = np.load(ARTIFACT_DIR / "shap_et.npy", allow_pickle=True)
except FileNotFoundError:
    st.error("SHAP artifacts not found. Run interpretability/shap_analysis.py first.")
    st.stop()

# -----------------------------
# Model selector
# -----------------------------
st.sidebar.header("SHAP Settings")

model_choice = st.sidebar.selectbox(
    "Select base model",
    ["Random Forest", "XGBoost", "Extra Trees"]
)

if model_choice == "Random Forest":
    shap_values = shap_rf
elif model_choice == "XGBoost":
    shap_values = shap_xgb
else:
    shap_values = shap_et

# -----------------------------
# Multi-class handling (IMPORTANT)
# -----------------------------
num_classes = len(shap_values)

class_idx = st.sidebar.selectbox(
    "Select wine quality class to explain",
    list(range(num_classes)),
    help="SHAP values are computed per class for multi-class models"
)

st.subheader(f"Global Feature Importance — {model_choice} (Class {class_idx})")

# -----------------------------
# SHAP Summary Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

shap.summary_plot(
    shap_values[class_idx],
    feature_names=feature_names,
    show=False
)

st.pyplot(fig)

# -----------------------------
# Explanation text
# -----------------------------
st.markdown("""
### 🧠 How to read this plot
- **Top features** → strongest influence on predictions
- **Red** → high feature value
- **Blue** → low feature value
- **Wider spread** → higher impact

This plot explains **why the model tends to predict this quality level**.
""")

# -----------------------------
# Meta-learner explanation
# -----------------------------
st.subheader("🧩 Meta-Learner (Stacking) Insight")

st.markdown("""
The final prediction is produced by a **logistic regression meta-learner**.

It learns:
- how much to trust each base model
- how to combine class probabilities
- how to resolve disagreement between models

This improves **calibration and reliability**, not just accuracy.
""")
