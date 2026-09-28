from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="Texto do laudo")


class PredictResponse(BaseModel):
    condition_label: int
    condition: str
    urgency: str
    confidence: float
    probabilities: dict[str, float]
    model_backend: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backend: str
