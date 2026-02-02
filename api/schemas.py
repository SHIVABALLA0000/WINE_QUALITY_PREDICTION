from pydantic import BaseModel
from typing import Dict,Optional

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    wine_type: str  # "red" or "white"



class PredictionResponse(BaseModel):
    status: str
    predicted_quality: Optional[int] = None
    confidence: Optional[float] = None
    class_probabilities: Dict[str, float]
    reason: Optional[str] = None
