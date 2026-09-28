"""Compara a latencia de inferencia sklearn vs ONNX Runtime (in-process).

Mede so o predict_proba (inclui clean_text), sem HTTP, com laudos reais do
conjunto de teste. A comparacao ponta a ponta via API fica com load_test.py.

Uso: python scripts/benchmark_latency.py [-n 2000] [--out reports/benchmark_inprocess.json]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from medical_triage.config import get_settings
from medical_triage.models.predictor import OnnxPredictor, SklearnPredictor

ROOT = Path(__file__).resolve().parents[1]


def time_single(predict, texts: list[str], warmup: int) -> list[float]:
    for t in texts[:warmup]:
        predict([t])
    out = []
    for t in texts:
        start = time.perf_counter()
        predict([t])
        out.append((time.perf_counter() - start) * 1000)
    return out


def time_batch(predict, texts: list[str], batch: int, rounds: int = 5) -> float:
    """Melhor throughput (laudos/s) em `rounds` passagens por lotes."""
    best = 0.0
    for _ in range(rounds):
        start = time.perf_counter()
        for i in range(0, len(texts), batch):
            predict(texts[i : i + batch])
        best = max(best, len(texts) / (time.perf_counter() - start))
    return best


def summarize(ms: list[float]) -> dict[str, float]:
    q = np.percentile(ms, [50, 95, 99])
    return {
        "mean_ms": round(statistics.mean(ms), 3),
        "p50_ms": round(float(q[0]), 3),
        "p95_ms": round(float(q[1]), 3),
        "p99_ms": round(float(q[2]), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "benchmark_inprocess.json")
    args = parser.parse_args()

    settings = get_settings()
    base = pd.read_csv(settings.processed_dir / "test.csv")["medical_abstract"].tolist()
    texts = [base[i % len(base)] for i in range(args.n)]

    backends = {
        "sklearn": SklearnPredictor(settings.sklearn_model_path),
        "onnx": OnnxPredictor(settings.onnx_model_path),
    }
    results = {}
    for name, predictor in backends.items():
        single = time_single(predictor.predict_proba, texts, args.warmup)
        results[name] = {
            **summarize(single),
            f"batch{args.batch}_docs_per_s": round(
                time_batch(predictor.predict_proba, texts, args.batch), 1
            ),
        }

    skl, onx = results["sklearn"], results["onnx"]
    results["speedup"] = {
        k: round(skl[k] / onx[k], 2) for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms")
    }
    results["speedup"]["batch_throughput"] = round(
        onx[f"batch{args.batch}_docs_per_s"] / skl[f"batch{args.batch}_docs_per_s"], 2
    )
    results["model_size_mb"] = {
        "sklearn": round(settings.sklearn_model_path.stat().st_size / 1e6, 2),
        "onnx": round(settings.onnx_model_path.stat().st_size / 1e6, 2),
    }
    results["config"] = {"n": args.n, "warmup": args.warmup, "batch": args.batch}

    print(json.dumps(results, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
