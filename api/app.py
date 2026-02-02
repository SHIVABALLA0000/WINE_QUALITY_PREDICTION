from fastapi import FastAPI, Security
from api.auth import verify_api_key
from api.schemas import WineInput, PredictionResponse
from api.service import predict_wine_quality

app = FastAPI(title="Wine Quality Prediction API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: WineInput, api_key: str = Security(verify_api_key)):
    return predict_wine_quality(data)