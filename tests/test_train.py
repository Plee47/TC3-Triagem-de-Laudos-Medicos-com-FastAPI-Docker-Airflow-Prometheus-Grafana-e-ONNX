from medical_triage.models.predictor import SklearnPredictor
from medical_triage.models.train import evaluate


def test_pipeline_learns_tiny_dataset(trained_pipeline, tiny_df):
    metrics = evaluate(trained_pipeline, tiny_df)
    assert metrics["macro_f1"] > 0.9
    assert 0.0 <= metrics["urgente_recall"] <= 1.0
    assert set(metrics) >= {"accuracy", "urgency_macro_f1", "per_class"}


def test_sklearn_predictor_returns_probabilities(model_dir):
    predictor = SklearnPredictor(model_dir / "model.joblib")
    proba = predictor.predict_proba(["myocardial infarction   heart artery"])
    assert proba.shape == (1, 5)
    assert abs(proba.sum() - 1.0) < 1e-6
    assert int(predictor.classes[proba.argmax()]) == 4
