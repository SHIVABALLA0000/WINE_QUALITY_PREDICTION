import xgboost as xgb
import lightgbm as lgb
from .config import RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def build_model(params, n_classes, class_weight_dict):
    if params["model_type"] == "XGB":
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample"],
            reg_lambda=params["reg_lambda"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist"
        )
    elif params["model_type"] == "LGBM":
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=n_classes,
            n_estimators=params["n_estimators"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample"],
            class_weight=class_weight_dict,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif params["model_type"] == "RF":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    elif params["model_type"] == "ET":
        return ExtraTreesClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
