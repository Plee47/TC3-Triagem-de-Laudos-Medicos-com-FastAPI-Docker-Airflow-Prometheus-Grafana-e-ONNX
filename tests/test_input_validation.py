"""Texto que o modelo nao consegue ler vira 422, nunca triagem "normal"."""

import pytest
from fastapi.testclient import TestClient

from medical_triage.api.main import app
from medical_triage.config import get_settings
from medical_triage.models.predictor import OnnxPredictor, SklearnPredictor

README_TEXT = "Acute myocardial infarction in patients with coronary artery disease..."


@pytest.fixture(params=["sklearn", "onnx"])
def client(request, onnx_model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", request.param)
    get_settings.cache_clear()
    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("text", ["   ", "\n\t", "🙂🙂🙂", "患者急性心肌梗死", "!!!", "x"])
def test_unreadable_text_is_422(client, text):
    assert client.post("/predict", json={"text": text}).status_code == 422


def test_text_without_known_terms_is_422(client):
    resp = client.post("/predict", json={"text": "qqqqzz wwwwxx"})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Nenhum termo conhecido pelo modelo"


def test_readme_example_is_200(client):
    resp = client.post("/predict", json={"text": README_TEXT})
    assert resp.status_code == 200
    assert resp.json()["urgency"] == "urgente"


def test_known_terms_match_on_both_backends(onnx_model_dir):
    skl = SklearnPredictor(onnx_model_dir / "model.joblib")
    onx = OnnxPredictor(onnx_model_dir / "model.onnx")
    texts = [README_TEXT, "qqqqzz wwwwxx", "Tumor carcinoma: metástase", "!!!", "lorem ipsum"]
    assert [skl.has_known_terms(t) for t in texts] == [onx.has_known_terms(t) for t in texts]
    assert [skl.has_known_terms(t) for t in texts] == [True, False, True, False, False]
