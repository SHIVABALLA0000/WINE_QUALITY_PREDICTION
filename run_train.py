import json
import joblib
import numpy as np
import platform

from src.data_utils import load_and_prepare_data
from src.train import train_with_nested_cv
from src.preprocess import build_preprocessor
from src.model import build_model
from src.metrics import f1_macro
from src.config import *

def main():
    X_train, X_test, y_train, y_test, le = load_and_prepare_data()

    outer_scores, best_params_list = train_with_nested_cv(X_train, y_train)

    final_params = max(
        best_params_list,
        key=lambda p: best_params_list.count(p)
    )

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    final_pre = build_preprocessor(num_cols)
    X_train_p = final_pre.fit_transform(X_train)
    X_test_p = final_pre.transform(X_test)

    classes = np.unique(y_train)
    class_weight_dict = dict(
        zip(
            classes,
            np.bincount(y_train).sum() / (len(classes) * np.bincount(y_train))
        )
    )

    final_model = build_model(final_params, len(classes), class_weight_dict)
    final_model.fit(X_train_p, y_train)

    test_preds = final_model.predict(X_test_p)
    test_f1 = f1_macro(y_test, test_preds)

    print("\nFINAL TEST F1-macro:", round(test_f1, 4))

    joblib.dump(
        {"preprocessor": final_pre, "model": final_model},
        ARTIFACT_MODEL
    )
    joblib.dump(le, ARTIFACT_ENCODER)

    with open(ARTIFACT_REPORT, "w") as f:
        json.dump({
            "outer_scores": outer_scores,
            "mean_outer_f1": float(np.mean(outer_scores)),
            "std_outer_f1": float(np.std(outer_scores)),
            "best_params_per_fold": best_params_list
        }, f, indent=2)

    with open(ARTIFACT_CARD, "w") as f:
        json.dump({
            "model_name": "Wine Quality – Ordinal Classification",
            "task": "Multi-class classification",
            "metric": "F1-macro",
            "final_test_f1_macro": float(test_f1),
            "environment": {
                "python": platform.python_version()
            }
        }, f, indent=2)

    print("Artifacts saved in:", BASE_PATH)

if __name__ == "__main__":
    main()
