import joblib
import pandas as pd
import pytest

from medical_triage.config import get_settings, load_params
from medical_triage.data.ingest import LABEL_COL, TEXT_COL
from medical_triage.models.export_onnx import export
from medical_triage.models.train import train

# Vocabulario pequeno e bem separado por classe: suficiente para um modelo
# de teste aprender sem depender do dataset real.
_CLASS_TERMS = {
    1: "tumor carcinoma malignant neoplasm chemotherapy metastasis",
    2: "liver gastric bowel colon hepatic pancreatitis",
    3: "brain seizure neuron epilepsy stroke cerebral",
    4: "cardiac heart artery myocardial hypertension coronary",
    5: "patients syndrome infection inflammation general condition",
}


@pytest.fixture(scope="session")
def tiny_df() -> pd.DataFrame:
    rows = []
    for label, terms in _CLASS_TERMS.items():
        words = terms.split()
        for i in range(30):
            text = " ".join(words[i % len(words) :] + words[: i % len(words)]) + f" case {i}"
            rows.append({LABEL_COL: label, TEXT_COL: text})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def test_params() -> dict:
    params = load_params()
    params["tfidf"] = {**params["tfidf"], "min_df": 1, "max_features": 1000}
    return params


@pytest.fixture(scope="session")
def trained_pipeline(tiny_df, test_params):
    return train(tiny_df, test_params)


@pytest.fixture
def model_dir(tmp_path, trained_pipeline, monkeypatch):
    joblib.dump(trained_pipeline, tmp_path / "model.joblib")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    # so ha model.joblib aqui; fixa o backend para um .env local (MODEL_BACKEND=onnx) nao vazar
    monkeypatch.setenv("MODEL_BACKEND", "sklearn")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def onnx_model_dir(model_dir, tiny_df):
    export(model_dir, tiny_df[TEXT_COL].tolist())
    return model_dir
