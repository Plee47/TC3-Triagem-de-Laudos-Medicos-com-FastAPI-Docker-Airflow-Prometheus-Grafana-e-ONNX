"""Abstracao de inferencia: mesma interface para sklearn e ONNX Runtime."""

import json
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import onnxruntime as ort

from medical_triage.config import ONNX_MODEL_FILE, SKLEARN_MODEL_FILE
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


class OnnxPredictor:
    backend = "onnx"

    def __init__(self, model_path: Path, intra_op_threads: int = 1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 1 thread por sessao: requisicoes sao de 1 laudo, e o paralelismo vem
        # dos workers/replicas; mais threads so disputam CPU no container.
        opts.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        meta = self._session.get_modelmeta().custom_metadata_map
        self.classes = np.array(json.loads(meta["classes"]))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        batch = np.array([clean_text(t) for t in texts], dtype=object).reshape(-1, 1)
        _, proba = self._session.run(None, {self._input: batch})
        return proba


def load_predictor(backend: str, model_dir: Path) -> Predictor:
    if backend == "sklearn":
        return SklearnPredictor(model_dir / SKLEARN_MODEL_FILE)
    if backend == "onnx":
        return OnnxPredictor(model_dir / ONNX_MODEL_FILE)
    raise ValueError(f"Backend nao suportado: {backend}")
