import json

from medical_triage.models.predictor import SklearnPredictor
from medical_triage.models.train import evaluate


def test_pipeline_learns_tiny_dataset(trained_pipeline, tiny_df):
    metrics = evaluate(trained_pipeline, tiny_df)
    assert metrics["macro_f1"] > 0.9
    assert 0.0 <= metrics["urgente_recall"] <= 1.0
    assert set(metrics) >= {"accuracy", "urgency_macro_f1", "low_confidence_rate", "per_class"}


def test_low_confidence_rate_counts_unreadable_texts(trained_pipeline):
    import pandas as pd

    from medical_triage.data.ingest import LABEL_COL, TEXT_COL

    # sem nenhum termo do vocabulario o modelo cai no prior (~0,2 por classe)
    df = pd.DataFrame({LABEL_COL: [1, 4], TEXT_COL: ["zzz qqq", "lorem ipsum"]})
    assert evaluate(trained_pipeline, df)["low_confidence_rate"] == 1.0


def test_sklearn_predictor_returns_probabilities(model_dir):
    predictor = SklearnPredictor(model_dir / "model.joblib")
    proba = predictor.predict_proba(["myocardial infarction   heart artery"])
    assert proba.shape == (1, 5)
    assert abs(proba.sum() - 1.0) < 1e-6
    assert int(predictor.classes[proba.argmax()]) == 4


def test_run_writes_artifacts_to_output_dir(tmp_path, tiny_df, test_params, monkeypatch):
    from medical_triage.config import get_settings
    from medical_triage.models import train as train_mod

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    for name in ("train", "val", "test"):
        tiny_df.to_csv(processed / f"{name}.csv", index=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(train_mod, "load_params", lambda: test_params)
    get_settings.cache_clear()

    out = tmp_path / "candidate"
    out.mkdir()
    (out / "model.onnx").write_bytes(b"modelo antigo")
    metrics = train_mod.run(output_dir=out)
    get_settings.cache_clear()

    assert (out / "model.joblib").exists()
    assert (out / "metrics.json").exists()
    assert not (out / "model.onnx").exists()  # ONNX antigo nao sobrevive ao retreino
    assert metrics["test"]["macro_f1"] > 0.9
    saved = json.loads((out / "metrics.json").read_text())
    assert {"low_confidence_rate", "macro_f1"} <= set(saved["val"]) & set(saved["test"])


def test_convertible_vocabulary_drops_orphan_ngrams():
    from medical_triage.models.train import convertible_vocabulary

    vocab = {"heart": 0, "failure": 1, "heart failure": 2, "zoster ophthalmicus": 3, "zoster": 4}
    assert convertible_vocabulary(vocab) == ["failure", "heart", "heart failure", "zoster"]
