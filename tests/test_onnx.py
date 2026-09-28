import numpy as np
import pytest
from fastapi.testclient import TestClient

from medical_triage.config import get_settings
from medical_triage.data.ingest import TEXT_COL
from medical_triage.models.export_onnx import check_parity, convert, export
from medical_triage.models.predictor import OnnxPredictor, SklearnPredictor, load_predictor
from medical_triage.models.train import train


@pytest.fixture
def onnx_model_dir(model_dir, tiny_df):
    export(model_dir, tiny_df[TEXT_COL].tolist())
    return model_dir


def test_export_has_exact_parity(trained_pipeline, tiny_df):
    texts = tiny_df[TEXT_COL].tolist() + ["a b c", "Heart  and a  β-blocker in 2 cases"]
    parity = check_parity(trained_pipeline, convert(trained_pipeline), texts)
    assert parity["label_agreement"] == 1.0
    assert parity["max_abs_proba_diff"] < 1e-4


def test_export_refuses_sublinear_tf(tiny_df, test_params):
    params = {**test_params, "tfidf": {**test_params["tfidf"], "sublinear_tf": True}}
    with pytest.raises(ValueError, match="sublinear_tf"):
        convert(train(tiny_df, params))


def test_onnx_predictor_matches_sklearn(onnx_model_dir):
    skl = SklearnPredictor(onnx_model_dir / "model.joblib")
    onx = OnnxPredictor(onnx_model_dir / "model.onnx")
    texts = ["brain seizure epilepsy", "liver hepatic bowel", "Tumor carcinoma — metástase"]
    np.testing.assert_array_equal(onx.classes, skl.classes)
    np.testing.assert_allclose(onx.predict_proba(texts), skl.predict_proba(texts), atol=1e-4)


def test_load_predictor_onnx_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_predictor("onnx", tmp_path)


def test_api_serves_onnx_backend(onnx_model_dir, monkeypatch):
    from medical_triage.api.main import app

    monkeypatch.setenv("MODEL_BACKEND", "onnx")
    get_settings.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").json()["model_backend"] == "onnx"
        body = client.post("/predict", json={"text": "cardiac heart artery myocardial"}).json()
    get_settings.cache_clear()
    assert body["model_backend"] == "onnx"
    assert body["urgency"] == "urgente"
