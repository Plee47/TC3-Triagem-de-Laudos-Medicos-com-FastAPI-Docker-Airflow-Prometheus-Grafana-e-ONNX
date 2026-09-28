import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from medical_triage.api.main import app


@pytest.fixture
def client(model_dir):
    with TestClient(app) as c:
        yield c


def _value(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_metrics_endpoint_exposes_prometheus_format(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "http_requests_total" in resp.text


def test_predict_updates_request_and_model_metrics(client):
    req = {"method": "POST", "route": "/predict", "status": "200"}
    before = _value("http_requests_total", req)
    urgent_before = _value(
        "triage_predictions_total",
        {"urgency": "urgente", "condition": "cardiovascular_diseases"},
    )
    inf_before = _value("model_inference_duration_seconds_count", {"backend": "sklearn"})

    client.post("/predict", json={"text": "cardiac heart artery myocardial"})

    assert _value("http_requests_total", req) == before + 1
    assert (
        _value(
            "triage_predictions_total",
            {"urgency": "urgente", "condition": "cardiovascular_diseases"},
        )
        == urgent_before + 1
    )
    assert (
        _value("model_inference_duration_seconds_count", {"backend": "sklearn"}) == inf_before + 1
    )
    assert (
        _value("http_request_duration_seconds_count", {"method": "POST", "route": "/predict"}) >= 1
    )


def test_client_errors_are_counted_with_status(client):
    labels = {"method": "POST", "route": "/predict", "status": "422"}
    before = _value("http_requests_total", labels)
    client.post("/predict", json={"text": ""})
    assert _value("http_requests_total", labels) == before + 1


def test_unknown_routes_share_one_label(client):
    labels = {"method": "GET", "route": "unmatched", "status": "404"}
    before = _value("http_requests_total", labels)
    client.get("/nao-existe/123")
    client.get("/outra/rota")
    assert _value("http_requests_total", labels) == before + 2


def test_metrics_scrape_is_not_counted(client):
    client.get("/metrics")
    assert (
        _value("http_requests_total", {"method": "GET", "route": "/metrics", "status": "200"}) == 0
    )


def test_model_loaded_gauge(client):
    assert _value("triage_model_loaded") == 1
