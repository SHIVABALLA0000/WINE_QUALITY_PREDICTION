import sys
from pathlib import Path

# -------------------------------------------------
# Make src importable (CRITICAL FIX)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

st.set_page_config(layout="wide")

st.title("📈 Partial Dependence Plots (Sanity Check)")
st.markdown("""
**PDPs show how predictions change as one feature varies**, averaged over the data.

✔ Validates domain intuition  
✔ Detects non-sensical behavior  
✔ Comes before SHAP
""")

# -------------------------------------------------
# Load model
# -------------------------------------------------
ARTIFACT_PATH = PROJECT_ROOT / "wine_artifacts" / "wine_quality_model.joblib"
bundle = joblib.load(ARTIFACT_PATH)

model = bundle["model"]
preprocessor = bundle["preprocessor"]

# -------------------------------------------------
# Load DEV data ONLY
# -------------------------------------------------
from src.data_utils import load_data

X, y, _ = load_data()

# Convert y → pandas Series (safe indexing)
y = pd.Series(y, index=X.index)

X_sample = X.sample(1500, random_state=42)
y_sample = y.loc[X_sample.index]

X_sample_p = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()

# -------------------------------------------------
# Select target class (multi-class FIX)
# -------------------------------------------------
target_class = st.selectbox(
    "Select wine quality class",
    sorted(y_sample.unique()),
    index=0
)

# -------------------------------------------------
# Feature selection
# -------------------------------------------------
feature = st.selectbox(
    "Select feature to inspect",
    sorted(set(f.split("__")[-1] for f in feature_names))
)

feature_idx = next(
    i for i, f in enumerate(feature_names)
    if f.endswith(feature)
)

# -------------------------------------------------
# PDP Plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

PartialDependenceDisplay.from_estimator(
    model,
    X_sample_p,
    features=[feature_idx],
    target=target_class,          # REQUIRED for multi-class
    feature_names=feature_names,
    ax=ax
)

st.pyplot(fig)

st.markdown("""
### ✅ What you want to see
- Smooth, monotonic trends
- Alcohol ↑ → quality ↑
- Density ↑ → quality ↓

❌ Sharp oscillations → unreliable model behavior
""")
