"""Abstracao de inferencia: mesma interface para sklearn e ONNX Runtime."""

from pathlib import Path
from typing import Protocol

import joblib
import numpy as np

from medical_triage.data.preprocess import clean_text


class Predictor(Protocol):
    backend: str
    classes: np.ndarray

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...


class SklearnPredictor:
    backend = "sklearn"

    def __init__(self, model_path: Path) -> None:
        self._pipeline = joblib.load(model_path)
        self.classes = self._pipeline.classes_

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipeline.predict_proba([clean_text(t) for t in texts])


def load_predictor(backend: str, model_dir: Path) -> Predictor:
    if backend == "sklearn":
        return SklearnPredictor(model_dir / "model.joblib")
    raise ValueError(f"Backend nao suportado: {backend}")
