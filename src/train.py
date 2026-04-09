# train.py
import json
import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import *
from .preprocess import build_preprocessor
from .metrics import f1_macro
from .tuning import (
    suggest_rf_params,
    suggest_xgb_params,
    suggest_et_params
)

from .model import (
    build_rf,
    build_xgb,
    build_et,
    build_stacking_model
)




# src/train.py

import numpy as np
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .model import build_rf, build_xgb, build_et, build_stacking_model
from .preprocess import build_preprocessor
from .metrics import f1_macro
from .config import *


def train_with_nested_cv(X_dev, y_dev):
    """
    Nested CV on DEV set only.
    Base learners tuned independently.
    Stacking + XGB evaluated in outer CV.
    """

    num_cols = X_dev.select_dtypes(include=np.number).columns.tolist()
    classes = np.unique(y_dev)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_dev
    )
    class_weight_dict = dict(zip(classes, class_weights))

    inner_cv = StratifiedKFold(
        n_splits=INNER_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    outer_scores_stack = []
    outer_scores_xgb = []

    best_rf_params = None
    best_xgb_params = None
    best_et_params = None

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_dev, y_dev), 1):

        X_tr, X_va = X_dev.iloc[tr_idx], X_dev.iloc[va_idx]
        y_tr, y_va = y_dev[tr_idx], y_dev[va_idx]

        pre = build_preprocessor(num_cols)
        X_tr_p = pre.fit_transform(X_tr)
        X_va_p = pre.transform(X_va)

        # ========================
        # RF TUNING
        # ========================
        def rf_objective(trial):
            params = suggest_rf_params(trial)
            scores = []

            for i_tr, i_va in inner_cv.split(X_tr_p, y_tr):
                model = build_rf(params, class_weight_dict)
                model.fit(X_tr_p[i_tr], y_tr[i_tr])
                preds = model.predict(X_tr_p[i_va])
                scores.append(f1_macro(y_tr[i_va], preds))

            return np.mean(scores)

        rf_study = optuna.create_study(direction="maximize")
        rf_study.optimize(rf_objective, n_trials=N_TRIALS_RF)
        best_rf_params = rf_study.best_params

        # ========================
        # XGB TUNING
        # ========================
        def xgb_objective(trial):
            params = suggest_xgb_params(trial)
            scores = []

            for i_tr, i_va in inner_cv.split(X_tr_p, y_tr):
                model = build_xgb(params, len(classes))
                model.fit(X_tr_p[i_tr], y_tr[i_tr])
                preds = model.predict(X_tr_p[i_va])
                scores.append(f1_macro(y_tr[i_va], preds))

            return np.mean(scores)

        xgb_study = optuna.create_study(direction="maximize")
        xgb_study.optimize(xgb_objective, n_trials=N_TRIALS_XGB)
        best_xgb_params = xgb_study.best_params

        # ========================
        # ET TUNING
        # ========================
        def et_objective(trial):
            params = suggest_et_params(trial)
            scores = []

            for i_tr, i_va in inner_cv.split(X_tr_p, y_tr):
                model = build_et(params)
                model.fit(X_tr_p[i_tr], y_tr[i_tr])
                preds = model.predict(X_tr_p[i_va])
                scores.append(f1_macro(y_tr[i_va], preds))

            return np.mean(scores)

        et_study = optuna.create_study(direction="maximize")
        et_study.optimize(et_objective, n_trials=N_TRIALS_RF)
        best_et_params = et_study.best_params

        # ========================
        # OUTER — STACKING
        # ========================
        stack = build_stacking_model(
            rf_params=best_rf_params,
            xgb_params=best_xgb_params,
            et_params=best_et_params,
            n_classes=len(classes),
            class_weight_dict=class_weight_dict
        )

        stack.fit(X_tr_p, y_tr)
        stack_preds = stack.predict(X_va_p)
        stack_score = f1_macro(y_va, stack_preds)
        outer_scores_stack.append(stack_score)

        # ========================
        # OUTER — XGB BASELINE
        # ========================
        xgb_model = build_xgb(best_xgb_params, len(classes))
        xgb_model.fit(X_tr_p, y_tr)
        xgb_preds = xgb_model.predict(X_va_p)
        xgb_score = f1_macro(y_va, xgb_preds)
        outer_scores_xgb.append(xgb_score)

        print(f"[Outer {fold}/{OUTER_CV_SPLITS}] Stack F1 = {stack_score:.4f}")
        print(f"[Outer {fold}/{OUTER_CV_SPLITS}] XGB   F1 = {xgb_score:.4f}")

    return outer_scores_stack, outer_scores_xgb, {
        "rf": best_rf_params,
        "xgb": best_xgb_params,
        "et": best_et_params
    }