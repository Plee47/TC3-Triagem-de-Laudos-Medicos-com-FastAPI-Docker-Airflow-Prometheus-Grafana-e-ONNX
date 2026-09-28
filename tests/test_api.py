import pytest
from fastapi.testclient import TestClient

from medical_triage.api.main import app
from medical_triage.config import get_settings


@pytest.fixture
def client(model_dir):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "model_backend": "sklearn"}


def test_predict_returns_urgency(client):
    resp = client.post("/predict", json={"text": "brain seizure epilepsy cerebral stroke"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["condition"] == "nervous_system_diseases"
    assert body["urgency"] == "urgente"
    assert len(body["probabilities"]) == 5
    assert body["inference_ms"] >= 0


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": 123}])
def test_predict_validates_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_without_model_is_503(client_without_model):
    assert client_without_model.get("/health").json()["model_loaded"] is False
    assert client_without_model.post("/predict", json={"text": "x"}).status_code == 503
