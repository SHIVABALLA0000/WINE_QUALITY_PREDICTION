# run_train.py
import json
import joblib
import numpy as np
import platform

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.data_utils import load_data
from src.train import train_with_nested_cv
from src.preprocess import build_preprocessor
from src.model import build_stacking_model
from src.metrics import f1_macro
from src.config import *


def main():
    X, y, le = load_data()

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    outer_scores, best_params = train_with_nested_cv(X_dev, y_dev)

    num_cols = X_dev.select_dtypes(include=np.number).columns.tolist()
    pre = build_preprocessor(num_cols)

    X_dev_p = pre.fit_transform(X_dev)
    X_test_p = pre.transform(X_test)

    classes = np.unique(y_dev)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_dev
    )
    class_weight_dict = dict(zip(classes, class_weights))

    final_model = build_stacking_model(
        rf_params=best_params["rf"],
        xgb_params=best_params["xgb"],
        et_params=best_params["et"],
        n_classes=len(classes),
        class_weight_dict=class_weight_dict
    )

    final_model.fit(X_dev_p, y_dev)

    test_preds = final_model.predict(X_test_p)
    test_f1 = f1_macro(y_test, test_preds)

    print("\nFINAL TEST F1-macro:", round(test_f1, 4))

    joblib.dump({"preprocessor": pre, "model": final_model}, ARTIFACT_MODEL)
    joblib.dump(le, ARTIFACT_ENCODER)

    with open(ARTIFACT_CARD, "w") as f:
        json.dump({
            "model": "Stacked Ensemble (RF + XGB + ET)",
            "metric": "F1-macro",
            "final_test_f1": float(test_f1),
            "python": platform.python_version()
        }, f, indent=2)

    print("Artifacts saved in:", BASE_PATH)


if __name__ == "__main__":
    main()
