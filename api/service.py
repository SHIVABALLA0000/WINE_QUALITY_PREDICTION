import pandas as pd
import joblib

COLUMN_RENAME_MAP = {
    "fixed_acidity": "fixed acidity",
    "volatile_acidity": "volatile acidity",
    "citric_acid": "citric acid",
    "residual_sugar": "residual sugar",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide": "total sulfur dioxide"
}

bundle = joblib.load("wine_artifacts/wine_quality_model.joblib")
_model = bundle["model"]
_preprocessor = bundle["preprocessor"]

CONFIDENCE_THRESHOLD = 0.70

def predict_wine_quality(data):
    X = pd.DataFrame([data.dict()])
    X = X.rename(columns=COLUMN_RENAME_MAP)

    X_processed = _preprocessor.transform(X)
    probs = _model.predict_proba(X_processed)[0]

    pred_class = int(probs.argmax())
    confidence = float(probs.max())

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "status": "low_confidence",
            "predicted_quality": pred_class,
            "confidence": confidence,
            "class_probabilities": {
                str(i): float(p) for i, p in enumerate(probs)
            },
            "reason": "Prediction is advisory only due to low confidence"
        }

    return {
        "status": "accepted",
        "predicted_quality": pred_class,
        "confidence": confidence,
        "class_probabilities": {
            str(i): float(p) for i, p in enumerate(probs)
        },
        "reason": None
    }
