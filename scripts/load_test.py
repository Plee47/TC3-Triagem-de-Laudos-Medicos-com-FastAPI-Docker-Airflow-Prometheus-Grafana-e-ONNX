"""Gera carga na API e mede latencia ponta a ponta (baseline / Grafana).

Uso: python scripts/load_test.py --url http://localhost:8080 -n 500 -c 4
     python scripts/load_test.py -n 3000 -c 2 --rate 10 --invalid-ratio 0.05   # trafego p/ Grafana

--invalid-ratio envia essa fracao de requisicoes com texto vazio (422 esperado),
para exercitar os paineis de erro. Elas ficam fora das estatisticas de latencia.
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
    parser.add_argument("--rate", type=float, help="limita a taxa total (req/s)")
    parser.add_argument("--invalid-ratio", type=float, default=0.0)
    parser.add_argument("--out", type=Path, help="salva o resumo em JSON")
    args = parser.parse_args()

    for text in load_samples(args.warmup):
        _, status = post(args.url, text)
        if status != 200:
            raise SystemExit(f"API nao respondeu 200 em {args.url}/predict (status={status})")

    samples = load_samples(args.n)
    invalid_every = round(1 / args.invalid_ratio) if args.invalid_ratio > 0 else 0
    if invalid_every:
        samples = ["" if i % invalid_every == 0 else t for i, t in enumerate(samples)]
    pause = args.c / args.rate if args.rate else 0.0

    def send(text: str) -> tuple[float, int, bool]:
        if pause:
            time.sleep(pause)
        ms, status = post(args.url, text)
        return ms, status, text == ""

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.c) as pool:
        results = list(pool.map(send, samples))
    elapsed = time.perf_counter() - start

    valid = [(ms, status) for ms, status, invalid in results if not invalid]
    latencies = [ms for ms, status in valid if status == 200]
    summary = {
        "requests": args.n,
        "concurrency": args.c,
        "intentional_invalid": len(results) - len(valid),
        "errors": len(valid) - len(latencies),
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
