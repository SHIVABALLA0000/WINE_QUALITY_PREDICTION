import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# -----------------------------
# Load trained model
# -----------------------------
ARTIFACT_PATH = Path("wine_artifacts/wine_quality_model.joblib")
bundle = joblib.load(ARTIFACT_PATH)

preprocessor = bundle["preprocessor"]
model = bundle["model"]

# -----------------------------
# Load DEV data only
# -----------------------------
from src.data_utils import load_data
X, y, _ = load_data()
y = pd.Series(y, index=X.index)
X_dev = X.sample(2000, random_state=42)
y_dev = y.loc[X_dev.index]
X_dev_p = preprocessor.transform(X_dev)
feature_names = preprocessor.get_feature_names_out()

# -----------------------------
# Permutation Importance
# -----------------------------
result = permutation_importance(
    model,
    X_dev_p,
    y_dev,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values("importance_mean", ascending=False)

# -----------------------------
# Save results
# -----------------------------
OUT_PATH = Path("wine_artifacts/permutation_importance.csv")
perm_df.to_csv(OUT_PATH, index=False)

print("✅ Permutation importance saved:", OUT_PATH)
print(perm_df.head(10))
