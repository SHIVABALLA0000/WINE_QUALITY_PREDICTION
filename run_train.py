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
from src.model import build_stacking_model, build_xgb
from src.metrics import f1_macro
from src.config import *

from src.stat_eval import (
    compare_models_cv,
    bootstrap_ci,
    evaluate_calibration,
    save_statistical_report
)


def main():
    X, y, le = load_data()

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Nested CV
    outer_scores_stack, outer_scores_xgb, best_params = train_with_nested_cv(X_dev, y_dev)

    stat_cv_results = compare_models_cv(
        outer_scores_stack,
        outer_scores_xgb
    )

    # Preprocessing full dev set
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

    # Final stacking model
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

    # Baseline XGB
    baseline_xgb = build_xgb(best_params["xgb"], len(classes))
    baseline_xgb.fit(X_dev_p, y_dev)
    xgb_test_f1 = f1_macro(y_test, baseline_xgb.predict(X_test_p))

    # Statistical evaluation
    bootstrap_results = bootstrap_ci(final_model, X_test_p, y_test)
    calibration_results = evaluate_calibration(final_model, X_test_p, y_test)

    statistical_report = {
        "cv_comparison": stat_cv_results,
        "bootstrap_test_ci": bootstrap_results,
        "calibration": calibration_results,
        "stack_test_f1": float(test_f1),
        "xgb_test_f1": float(xgb_test_f1)
    }

    save_statistical_report(statistical_report)

    print("\nFINAL TEST F1-macro (Stacking):", round(test_f1, 4))
    print("FINAL TEST F1-macro (XGB):", round(xgb_test_f1, 4))

    # Save artifacts
    joblib.dump({"preprocessor": pre, "model": final_model}, ARTIFACT_MODEL)
    joblib.dump(le, ARTIFACT_ENCODER)

    with open(ARTIFACT_CARD, "w") as f:
        json.dump({
            "model": "Stacked Ensemble (RF + XGB + ET)",
            "metric": "F1-macro",
            "final_test_f1": float(test_f1),
            "xgb_test_f1": float(xgb_test_f1),
            "python": platform.python_version()
        }, f, indent=2)

    print("Artifacts + Statistical report saved.")


if __name__ == "__main__":
    main()