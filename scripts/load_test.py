"""Gera carga na API e mede latencia ponta a ponta (baseline / Grafana).

Uso: python scripts/load_test.py --url http://localhost:8080 -n 500 -c 4
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_samples(n: int) -> list[str]:
    path = ROOT / "data" / "raw" / "medical_tc_test.csv"
    texts = pd.read_csv(path)["medical_abstract"].tolist()
    return [texts[i % len(texts)] for i in range(n)]


def post(url: str, text: str) -> tuple[float, int]:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        status = 0
    return (time.perf_counter() - start) * 1000, status


def percentile(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[int(q) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("-n", type=int, default=500, help="numero de requisicoes")
    parser.add_argument("-c", type=int, default=1, help="concorrencia")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out", type=Path, help="salva o resumo em JSON")
    args = parser.parse_args()

    for text in load_samples(args.warmup):
        _, status = post(args.url, text)
        if status != 200:
            raise SystemExit(f"API nao respondeu 200 em {args.url}/predict (status={status})")

    samples = load_samples(args.n)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.c) as pool:
        results = list(pool.map(lambda t: post(args.url, t), samples))
    elapsed = time.perf_counter() - start

    latencies = [ms for ms, status in results if status == 200]
    errors = len(results) - len(latencies)
    summary = {
        "requests": args.n,
        "concurrency": args.c,
        "errors": errors,
        "throughput_rps": round(args.n / elapsed, 1),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "p99_ms": round(percentile(latencies, 99), 2),
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
