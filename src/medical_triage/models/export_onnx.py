"""Exporta o pipeline sklearn (TF-IDF + LogReg) para ONNX e valida a paridade.

Dois detalhes do conversor skl2onnx precisam de cuidado para o ONNX reproduzir
o sklearn exatamente:
- tokenizer: por padrao o conversor usa [a-zA-Z0-9_]+, que mantem tokens de
  1 caractere e muda os bigramas; usamos a regex de 2+ caracteres, equivalente
  ao token_pattern do sklearn para texto ASCII (clean_text garante ASCII);
- sublinear_tf: o conversor calcula log(1 + tf) em vez de 1 + log(tf), entao
  modelos com sublinear_tf=True sao recusados;
- locale: o StringNormalizer (lowercase) usa en_US.UTF-8 por padrao, ausente em
  imagens slim do Linux (a sessao nem inicializa); com texto ASCII o locale "C"
  da o mesmo resultado sem depender de pacote de idioma.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from medical_triage.config import (
    METRICS_FILE,
    ONNX_MODEL_FILE,
    SKLEARN_MODEL_FILE,
    get_settings,
)
from medical_triage.data.ingest import TEXT_COL
from medical_triage.data.preprocess import clean_text

logger = logging.getLogger(__name__)

TOKEN_REGEX = r"[a-zA-Z0-9_][a-zA-Z0-9_]+"
LOCALE = "C"
INPUT_NAME = "text"
TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}
PROBA_TOLERANCE = 1e-4  # float32 no ONNX vs float64 no sklearn


def convert(pipeline: Pipeline) -> bytes:
    if pipeline.named_steps["tfidf"].sublinear_tf:
        raise ValueError(
            "sublinear_tf=True nao e suportado: o skl2onnx calcula log(1+tf) e o "
            "sklearn 1+log(tf). Treine com tfidf.sublinear_tf=false."
        )
    onx = to_onnx(
        pipeline,
        initial_types=[(INPUT_NAME, StringTensorType([None, 1]))],
        options={
            LogisticRegression: {"zipmap": False},
            TfidfVectorizer: {"tokenexp": TOKEN_REGEX, "locale": LOCALE},
        },
        target_opset=TARGET_OPSET,
    )
    meta = onx.metadata_props.add()
    meta.key, meta.value = "classes", json.dumps([int(c) for c in pipeline.classes_])
    return onx.SerializeToString()


def check_parity(pipeline: Pipeline, onnx_bytes: bytes, texts: list[str]) -> dict[str, float]:
    """Compara predicoes e probabilidades do sklearn e do ONNX nos mesmos textos."""
    cleaned = [clean_text(t) for t in texts]
    sess = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    _, proba_onnx = sess.run(None, {INPUT_NAME: np.array(cleaned, dtype=object).reshape(-1, 1)})
    proba_skl = pipeline.predict_proba(cleaned)
    # Empate entre classes (ex.: texto sem nenhum token conhecido -> todas 0,2)
    # faz o argmax depender de ruido de float; esses casos nao contam no acordo.
    top2 = np.sort(proba_skl, axis=1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) > PROBA_TOLERANCE
    agree = proba_onnx.argmax(1)[decisive] == proba_skl.argmax(1)[decisive]
    return {
        "n": len(texts),
        "ties_excluded": int((~decisive).sum()),
        "label_agreement": float(agree.mean()) if agree.size else 1.0,
        "max_abs_proba_diff": float(np.abs(proba_onnx - proba_skl).max()),
    }


def export(model_dir: Path, parity_texts: list[str]) -> dict[str, Any]:
    """Gera model.onnx ao lado do model.joblib; falha se a paridade nao for exata."""
    pipeline = joblib.load(model_dir / SKLEARN_MODEL_FILE)
    onnx_bytes = convert(pipeline)
    parity = check_parity(pipeline, onnx_bytes, parity_texts)
    if parity["label_agreement"] < 1.0 or parity["max_abs_proba_diff"] > PROBA_TOLERANCE:
        raise ValueError(f"ONNX diverge do sklearn: {parity}")

    (model_dir / ONNX_MODEL_FILE).write_bytes(onnx_bytes)
    metrics_path = model_dir / METRICS_FILE
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["onnx_parity"] = parity
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("ONNX exportado (%.1f MB), paridade %s", len(onnx_bytes) / 1e6, parity)
    return parity


def run(model_dir: Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    texts = pd.read_csv(settings.processed_dir / "test.csv")[TEXT_COL].tolist()
    return export(model_dir or settings.model_dir, texts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
