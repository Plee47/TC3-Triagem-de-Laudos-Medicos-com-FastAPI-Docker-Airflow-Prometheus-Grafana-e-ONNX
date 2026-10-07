from pydantic import BaseModel, Field, field_validator

from medical_triage.models.predictor import tokenize


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000, description="Texto do laudo")

    @field_validator("text")
    @classmethod
    def text_has_tokens(cls, value: str) -> str:
        # espacos, emoji, texto nao latino ou so pontuacao chegam vazios ao modelo,
        # que devolveria so o prior (classe 5, urgencia normal)
        if not tokenize(value):
            raise ValueError("texto sem nenhuma palavra legivel (o modelo le texto em ingles)")
        return value


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
