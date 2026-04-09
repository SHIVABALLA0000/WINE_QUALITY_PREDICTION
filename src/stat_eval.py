# src/stat_eval.py

import numpy as np
import json
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.utils import resample


def compare_models_cv(scores_a, scores_b):
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    diff = scores_a - scores_b

    t_stat, p_t = ttest_rel(scores_a, scores_b)

    try:
        w_stat, p_w = wilcoxon(scores_a, scores_b)
    except:
        w_stat, p_w = None, None

    cohen_d = (
        np.mean(diff) / np.std(diff, ddof=1)
        if np.std(diff, ddof=1) != 0 else 0.0
    )

    return {
        "mean_model_a": float(np.mean(scores_a)),
        "mean_model_b": float(np.mean(scores_b)),
        "mean_difference": float(np.mean(diff)),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(p_t),
        "wilcoxon_stat": float(w_stat) if w_stat else None,
        "wilcoxon_pvalue": float(p_w) if p_w else None,
        "cohen_d": float(cohen_d)
    }


def bootstrap_ci(model, X_test, y_test, n_bootstrap=1000):
    boot_scores = []

    for _ in range(n_bootstrap):
        X_res, y_res = resample(X_test, y_test)
        preds = model.predict(X_res)
        boot_scores.append(f1_score(y_res, preds, average="macro"))

    return {
        "bootstrap_mean": float(np.mean(boot_scores)),
        "bootstrap_lower_95": float(np.percentile(boot_scores, 2.5)),
        "bootstrap_upper_95": float(np.percentile(boot_scores, 97.5))
    }


def evaluate_calibration(model, X_test, y_test):
    probs = model.predict_proba(X_test)

    brier = np.mean([
        brier_score_loss((y_test == i).astype(int), probs[:, i])
        for i in range(probs.shape[1])
    ])

    return {"brier_score": float(brier)}


def save_statistical_report(report_dict, path="wine_artifacts/statistical_report.json"):
    Path("wine_artifacts").mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=4)