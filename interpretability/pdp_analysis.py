import sys
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import pandas as pd

# -------------------------------------------------
# Make src importable (FIX for pickle + imports)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
ARTIFACT_PATH = PROJECT_ROOT / "wine_artifacts" / "wine_quality_model.joblib"
bundle = joblib.load(ARTIFACT_PATH)

preprocessor = bundle["preprocessor"]
model = bundle["model"]

# -------------------------------------------------
# Load DEV data ONLY
# -------------------------------------------------
from src.data_utils import load_data

X, y, _ = load_data()

# Convert y → pandas Series for safe indexing
y = pd.Series(y, index=X.index)

X_dev = X.sample(1500, random_state=42)
y_dev = y.loc[X_dev.index]

X_dev_p = preprocessor.transform(X_dev)
feature_names = preprocessor.get_feature_names_out()

# -------------------------------------------------
# Choose dominant class (sanity check target)
# -------------------------------------------------
target_class = np.bincount(y_dev).argmax()
print(f"Using target class: {target_class}")

# -------------------------------------------------
# Domain-driven features to inspect
# -------------------------------------------------
features_to_plot = [
    "alcohol",
    "density",
    "sulphates",
    "volatile acidity",
]

# Map feature names → indices
feature_indices = [
    i for i, f in enumerate(feature_names)
    if f.split("__")[-1] in features_to_plot
]

# -------------------------------------------------
# PDP Plot
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

PartialDependenceDisplay.from_estimator(
    model,
    X_dev_p,
    features=feature_indices,
    target=target_class,
    feature_names=feature_names,
    ax=ax
)

plt.suptitle(
    f"PDP Sanity Check (Target Class = {target_class})",
    fontsize=14
)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "wine_artifacts" / "pdp_sanity.png")
plt.show()

print("✅ PDP sanity plot saved to wine_artifacts/pdp_sanity.png")
