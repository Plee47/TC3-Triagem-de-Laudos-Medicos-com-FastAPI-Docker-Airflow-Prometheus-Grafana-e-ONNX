"""Treino do classificador TF-IDF + Regressao Logistica e avaliacao."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.pipeline import Pipeline

from medical_triage.config import METRICS_FILE, SKLEARN_MODEL_FILE, get_settings, load_params
from medical_triage.data.ingest import LABEL_COL, TEXT_COL
from medical_triage.triage import to_urgency

logger = logging.getLogger(__name__)


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    tfidf = params["tfidf"]
    model = params["model"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, tfidf["ngram_max"]),
                    max_features=tfidf["max_features"],
                    min_df=tfidf["min_df"],
                    sublinear_tf=tfidf["sublinear_tf"],
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=model["C"],
                    max_iter=model["max_iter"],
                    class_weight="balanced",
                    random_state=params["seed"],
                ),
            ),
        ]
    )


def train(df: pd.DataFrame, params: dict[str, Any]) -> Pipeline:
    pipeline = build_pipeline(params)
    pipeline.fit(df[TEXT_COL], df[LABEL_COL])
    return pipeline


def evaluate(pipeline: Pipeline, df: pd.DataFrame) -> dict[str, Any]:
    """Metricas na granularidade da doenca (5 classes) e da urgencia (3 niveis)."""
    y_true = df[LABEL_COL].to_numpy()
    y_pred = pipeline.predict(df[TEXT_COL])
    urg_true, urg_pred = to_urgency(y_true), to_urgency(y_pred)
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "urgency_accuracy": float(accuracy_score(urg_true, urg_pred)),
        "urgency_macro_f1": float(f1_score(urg_true, urg_pred, average="macro")),
        "urgente_recall": float(
            recall_score(urg_true, urg_pred, labels=["urgente"], average="macro", zero_division=0)
        ),
        "per_class": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def run(output_dir: Path | None = None) -> dict[str, Any]:
    """Treina e grava modelo + metricas em output_dir (padrao: MODEL_DIR)."""
    settings = get_settings()
    output_dir = output_dir or settings.model_dir
    params = load_params()
    train_df = pd.read_csv(settings.processed_dir / "train.csv")
    val_df = pd.read_csv(settings.processed_dir / "val.csv")
    test_df = pd.read_csv(settings.processed_dir / "test.csv")

    pipeline = train(train_df, params)
    metrics = {
        "trained_at": datetime.now(UTC).isoformat(),
        "params": {k: params[k] for k in ("tfidf", "model", "seed")},
        "val": evaluate(pipeline, val_df),
        "test": evaluate(pipeline, test_df),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / SKLEARN_MODEL_FILE)
    (output_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(
        "Modelo salvo em %s | test macro_f1=%.3f urgente_recall=%.3f",
        output_dir,
        metrics["test"]["macro_f1"],
        metrics["test"]["urgente_recall"],
    )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
