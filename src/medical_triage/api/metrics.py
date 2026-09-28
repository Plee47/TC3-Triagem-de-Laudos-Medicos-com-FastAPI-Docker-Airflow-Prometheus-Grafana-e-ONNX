"""Metricas Prometheus da API de triagem.

Rotas sao rotuladas pelo template (ex.: /predict), nunca pela URL crua, para
manter a cardinalidade baixa. /metrics nao e contabilizado.
"""

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match

# Buckets em segundos: a inferencia fica na casa de 1-30 ms.
LATENCY_BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)

HTTP_REQUESTS = Counter(
    "http_requests_total", "Requisicoes HTTP recebidas", ["method", "route", "status"]
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Tempo de resposta HTTP ponta a ponta",
    ["method", "route"],
    buckets=LATENCY_BUCKETS,
)
HTTP_IN_PROGRESS = Gauge("http_requests_in_progress", "Requisicoes HTTP em andamento")

INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Tempo de inferencia do modelo (sem overhead HTTP)",
    ["backend"],
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS = Counter(
    "triage_predictions_total", "Predicoes por urgencia e condicao", ["urgency", "condition"]
)
CONFIDENCE = Histogram(
    "triage_prediction_confidence",
    "Probabilidade da classe prevista",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_INFO = Gauge(
    "triage_model_info",
    "Modelo carregado (valor 1) com metadados nos rotulos",
    ["backend", "trained_at", "test_macro_f1"],
)
MODEL_LOADED = Gauge("triage_model_loaded", "1 se o modelo esta carregado")

_UNMATCHED = "unmatched"
_EXCLUDED = {"/metrics"}


def route_template(request: Request) -> str:
    for route in request.app.router.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route.path
    return _UNMATCHED


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        route = route_template(request)
        if route in _EXCLUDED:
            return await call_next(request)

        method = request.method
        status = "500"
        HTTP_IN_PROGRESS.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            HTTP_LATENCY.labels(method, route).observe(time.perf_counter() - start)
            HTTP_REQUESTS.labels(method, route, status).inc()
            HTTP_IN_PROGRESS.dec()


def record_prediction(
    backend: str, inference_s: float, urgency: str, condition: str, confidence: float
) -> None:
    INFERENCE_LATENCY.labels(backend).observe(inference_s)
    PREDICTIONS.labels(urgency, condition).inc()
    CONFIDENCE.observe(confidence)


def set_model_state(backend: str | None, metrics: dict | None) -> None:
    MODEL_INFO.clear()
    if backend is None:
        MODEL_LOADED.set(0)
        return
    MODEL_LOADED.set(1)
    test = (metrics or {}).get("test", {})
    MODEL_INFO.labels(
        backend=backend,
        trained_at=(metrics or {}).get("trained_at", "unknown"),
        test_macro_f1=f"{test.get('macro_f1', 0):.3f}",
    ).set(1)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
