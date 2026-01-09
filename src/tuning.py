def suggest_params(trial):
    
    model_type = trial.suggest_categorical(
        "model_type",
        ["XGB", "LGBM", "RF", "ET"]
    )

    # Shared parameter
    params = {
        "model_type": model_type,
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
    }

    # -------------------------
    # XGBoost
    # -------------------------
    if model_type == "XGB":
        params.update({
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample": trial.suggest_float("colsample", 0.7, 1.0),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-3, 10, log=True
            ),
        })

    # -------------------------
    # LightGBM
    # -------------------------
    elif model_type == "LGBM":
        params.update({
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.1, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample": trial.suggest_float("colsample", 0.7, 1.0),
        })

    # -------------------------
    # Random Forest
    # -------------------------
    elif model_type == "RF":
        params.update({
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "bootstrap": True,  # RF-specific
        })

    # -------------------------
    # Extra Trees
    # -------------------------
    elif model_type == "ET":
        params.update({
            "max_depth": trial.suggest_int("max_depth", 6, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2"]
            ),
            "bootstrap": False,  # ET-specific
        })

    return params
