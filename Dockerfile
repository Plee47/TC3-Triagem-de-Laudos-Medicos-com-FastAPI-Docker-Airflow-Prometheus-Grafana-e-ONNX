FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir "poetry==2.4.1" "poetry-plugin-export==1.9.0"
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --only main --without-hashes -o requirements.txt


FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models \
    MODEL_BACKEND=sklearn

COPY --from=builder /build/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY params.yaml ./
# Artefatos gerados por `make train` / DAG do Airflow antes do build.
COPY models/ ./models/

RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "medical_triage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
