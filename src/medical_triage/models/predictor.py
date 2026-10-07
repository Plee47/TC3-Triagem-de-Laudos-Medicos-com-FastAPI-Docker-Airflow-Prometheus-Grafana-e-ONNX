"""Abstracao de inferencia: mesma interface para sklearn e ONNX Runtime."""

import json
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer

from medical_triage.config import ONNX_MODEL_FILE, SKLEARN_MODEL_FILE
from medical_triage.data.preprocess import clean_text

# train.py nao muda a tokenizacao do TfidfVectorizer: o analyzer padrao (lowercase,
# palavras de 2+ caracteres) devolve os mesmos unigramas que o modelo enxerga.
_analyzer = TfidfVectorizer().build_analyzer()


def tokenize(text: str) -> list[str]:
    """Unigramas que o TF-IDF extrai do texto, depois do clean_text."""
    return _analyzer(clean_text(text))


class Predictor(Protocol):
    backend: str
    classes: np.ndarray

    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    def has_known_terms(self, text: str) -> bool: ...


class SklearnPredictor:
    backend = "sklearn"

    def __init__(self, model_path: Path) -> None:
        self._pipeline = joblib.load(model_path)
        self.classes = self._pipeline.classes_
        # todo bigrama do vocabulario tem as duas palavras nele (convertible_vocabulary),
        # entao checar unigramas equivale a checar se o vetor TF-IDF tem algum termo
        self._vocabulary = self._pipeline.named_steps["tfidf"].vocabulary_

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipeline.predict_proba([clean_text(t) for t in texts])

    def has_known_terms(self, text: str) -> bool:
        return any(t in self._vocabulary for t in tokenize(text))


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
        self._vocabulary = _onnx_unigrams(model_path)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        batch = np.array([clean_text(t) for t in texts], dtype=object).reshape(-1, 1)
        _, proba = self._session.run(None, {self._input: batch})
        return proba

    def has_known_terms(self, text: str) -> bool:
        return any(t in self._vocabulary for t in tokenize(text))


def _onnx_unigrams(model_path: Path) -> frozenset[str]:
    """Unigramas do vocabulario, lidos do TfIdfVectorizer do grafo ONNX.

    pool_strings lista os unigramas e depois as palavras dos bigramas;
    ngram_counts[1] marca onde os bigramas comecam.
    """
    graph = onnx.load(str(model_path)).graph
    node = next(n for n in graph.node if n.op_type == "TfIdfVectorizer")
    attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    pool, counts = attrs["pool_strings"], attrs["ngram_counts"]
    end = counts[1] if len(counts) > 1 else len(pool)
    return frozenset(s.decode() for s in pool[:end])


def load_predictor(backend: str, model_dir: Path) -> Predictor:
    if backend == "sklearn":
        return SklearnPredictor(model_dir / SKLEARN_MODEL_FILE)
    if backend == "onnx":
        return OnnxPredictor(model_dir / ONNX_MODEL_FILE)
    raise ValueError(f"Backend nao suportado: {backend}")
