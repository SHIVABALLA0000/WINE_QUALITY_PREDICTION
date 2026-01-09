import json
import joblib
import numpy as np
import optuna
import platform

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from .config import *
from .preprocess import build_preprocessor
from .model import build_model
from .metrics import f1_macro
from .tuning import suggest_params

def train_with_nested_cv(X_train, y_train):
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    outer_scores = []
    outer_best_params = []

    for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train[va_idx]

        def inner_objective(trial):
            params = suggest_params(trial)
            scores = []

            for itr, iva in inner_cv.split(X_tr, y_tr):
                pre = build_preprocessor(num_cols)
                X_t = pre.fit_transform(X_tr.iloc[itr])
                X_v = pre.transform(X_tr.iloc[iva])

                model = build_model(params, len(classes), class_weight_dict)
                model.fit(X_t, y_tr[itr])

                preds = model.predict(X_v)
                scores.append(f1_macro(y_tr[iva], preds))

            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(inner_objective, n_trials=N_TRIALS_INNER)

        best_params = study.best_params
        outer_best_params.append(best_params)

        pre = build_preprocessor(num_cols)
        X_tr_p = pre.fit_transform(X_tr)
        X_va_p = pre.transform(X_va)

        model = build_model(best_params, len(classes), class_weight_dict)
        model.fit(X_tr_p, y_tr)

        preds = model.predict(X_va_p)
        score = f1_macro(y_va, preds)

        outer_scores.append(score)
        print(f"[Outer {fold}/{OUTER_SPLITS}] F1-macro = {score:.4f}")

    return outer_scores, outer_best_params
