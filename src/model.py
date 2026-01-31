# model.py
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from .config import RANDOM_STATE


# -------------------------
# Base model builders
# -------------------------
def build_rf(params, class_weight_dict):
    return RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def build_xgb(params, n_classes):
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        **params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )


def build_et(params):
    return ExtraTreesClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# -------------------------
# STACKING MODEL
# -------------------------
def build_stacking_model(
    rf_params,
    xgb_params,
    et_params,
    n_classes,
    class_weight_dict
):
    """
    Reliability-first stacking ensemble (Option-A)
    """

    rf = build_rf(rf_params, class_weight_dict)
    xgb_model = build_xgb(xgb_params, n_classes)
    et = build_et(et_params)

    meta_learner = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight_dict
    )

    stack = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("xgb", xgb_model),
            ("et", et)
        ],
        final_estimator=meta_learner,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1
    )

    return stack
