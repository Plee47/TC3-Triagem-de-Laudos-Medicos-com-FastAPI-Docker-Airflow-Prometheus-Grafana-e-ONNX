"""API de inferencia: recebe o texto do laudo e devolve condicao + urgencia."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from medical_triage import __version__
from medical_triage.api.metrics import (
    PrometheusMiddleware,
    metrics_response,
    record_prediction,
    set_model_state,
)
from medical_triage.api.schemas import HealthResponse, PredictRequest, PredictResponse
from medical_triage.config import get_settings
from medical_triage.models.predictor import load_predictor
from medical_triage.models.registry import read_metrics
from medical_triage.triage import triage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # uvicorn so configura os proprios loggers; sem isso os INFO do pacote nao aparecem
    # (nivel que o logging nao conhece, como o "trace" do uvicorn, vira INFO)
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    try:
        app.state.predictor = load_predictor(settings.model_backend, settings.model_dir)
        set_model_state(settings.model_backend, read_metrics(settings.model_dir))
        logger.info("Modelo carregado (backend=%s)", settings.model_backend)
    except (FileNotFoundError, OSError) as exc:
        app.state.predictor = None
        set_model_state(None, None)
        logger.warning("Modelo indisponivel: %s", exc)
    yield


app = FastAPI(title="Triagem de Laudos Medicos", version=__version__, lifespan=lifespan)
app.add_middleware(PrometheusMiddleware)


@app.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()


@app.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    predictor = request.app.state.predictor
    # sem modelo a API nao serve /predict: 503 derruba o HEALTHCHECK do container
    if predictor is None:
        response.status_code = 503
    return HealthResponse(
        status="ok" if predictor is not None else "indisponivel",
        model_loaded=predictor is not None,
        model_backend=get_settings().model_backend,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado")
    if not predictor.has_known_terms(payload.text):
        raise HTTPException(status_code=422, detail="Nenhum termo conhecido pelo modelo")

    start = time.perf_counter()
    proba = predictor.predict_proba([payload.text])[0]
    inference_s = time.perf_counter() - start

    best = int(proba.argmax())
    result = triage(int(predictor.classes[best]))
    confidence = float(proba[best])
    record_prediction(predictor.backend, inference_s, result.urgency, result.condition, confidence)
    probabilities = {
        triage(int(label)).condition: round(float(p), 4)
        for label, p in zip(predictor.classes, proba, strict=True)
    }
    return PredictResponse(
        condition_label=result.label,
        condition=result.condition,
        urgency=result.urgency,
        confidence=round(confidence, 4),
        probabilities=probabilities,
        model_backend=predictor.backend,
        inference_ms=round(inference_s * 1000, 3),
    )
