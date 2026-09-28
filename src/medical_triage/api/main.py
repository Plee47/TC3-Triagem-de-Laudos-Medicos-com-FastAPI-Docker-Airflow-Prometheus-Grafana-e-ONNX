"""API de inferencia: recebe o texto do laudo e devolve condicao + urgencia."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from medical_triage import __version__
from medical_triage.api.schemas import HealthResponse, PredictRequest, PredictResponse
from medical_triage.config import get_settings
from medical_triage.models.predictor import load_predictor
from medical_triage.triage import triage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        app.state.predictor = load_predictor(settings.model_backend, settings.model_dir)
        logger.info("Modelo carregado (backend=%s)", settings.model_backend)
    except (FileNotFoundError, OSError) as exc:
        app.state.predictor = None
        logger.warning("Modelo indisponivel: %s", exc)
    yield


app = FastAPI(title="Triagem de Laudos Medicos", version=__version__, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_backend=get_settings().model_backend,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado")

    start = time.perf_counter()
    proba = predictor.predict_proba([payload.text])[0]
    inference_ms = (time.perf_counter() - start) * 1000

    best = int(proba.argmax())
    result = triage(int(predictor.classes[best]))
    probabilities = {
        triage(int(label)).condition: round(float(p), 4)
        for label, p in zip(predictor.classes, proba, strict=True)
    }
    return PredictResponse(
        condition_label=result.label,
        condition=result.condition,
        urgency=result.urgency,
        confidence=round(float(proba[best]), 4),
        probabilities=probabilities,
        model_backend=predictor.backend,
        inference_ms=round(inference_ms, 3),
    )
