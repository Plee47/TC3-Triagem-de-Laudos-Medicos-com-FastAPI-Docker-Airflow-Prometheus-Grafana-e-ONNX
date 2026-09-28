# TC3 — Triagem de Laudos Médicos com FastAPI, Docker, Airflow, Prometheus/Grafana e ONNX

Pipeline de MLOps para triagem automática de laudos médicos: um classificador de texto leve (TF-IDF + Regressão Logística) define a condição do laudo e a traduz em urgência (**normal / atenção / urgente**). O modelo é servido por uma API FastAPI em Docker, com CI/CD no GitHub Actions, retreino orquestrado pelo Airflow, observabilidade com Prometheus + Grafana e otimização de latência via ONNX Runtime.

> Status: **Etapa 1 concluída** (dados, modelo base, API em Docker e baseline de latência). Etapas 2–4 em andamento.

## Sumário
- [Dataset e premissas](#dataset-e-premissas)
- [Decisão arquitetural (nuvem)](#decisão-arquitetural-nuvem)
- [Modelo e resultados](#modelo-e-resultados)
- [Latência — baseline](#latência--baseline)
- [Como executar](#como-executar)
- [API](#api)
- [Estrutura do repositório](#estrutura-do-repositório)

## Dataset e premissas

**Medical Abstracts TC Corpus** ([Kaggle](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus) · [repositório dos autores](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)): 14.438 abstracts médicos em inglês, 11.550 de treino e 2.888 de teste, em 5 classes. A ingestão (`medical_triage.data.ingest`) baixa os CSVs do repositório dos autores (mesmo conteúdo do Kaggle, sem exigir credencial) e valida o formato.

**Da condição para a urgência.** O corpus não tem rótulo de urgência. O modelo aprende as 5 condições (sinal real do dataset) e a urgência é derivada por uma tabela versionada em [`configs/urgency_map.yaml`](configs/urgency_map.yaml):

| Rótulo | Condição | Urgência |
|---|---|---|
| 3 | Nervous system diseases | **urgente** |
| 4 | Cardiovascular diseases | **urgente** |
| 1 | Neoplasms | atenção |
| 2 | Digestive system diseases | normal |
| 5 | General pathological conditions | normal |

> ⚠️ Esse mapeamento é uma **premissa de negócio do projeto**, não um critério clínico validado. Em produção ele seria definido com a equipe médica; por estar em configuração, pode mudar sem retreinar o modelo.

**Qualidade dos dados — o que encontramos e como tratamos:**

| Achado | Impacto | Tratamento |
|---|---|---|
| 1.956 abstracts do treino aparecem repetidos com **rótulos diferentes** (textos multi-condição) | rótulo ambíguo; deduplicar sem critério escolhe um rótulo arbitrário | cada texto fica com a condição **mais urgente** (conservador para triagem); empate → menor rótulo, que prefere uma condição específica à "general" |
| 988 abstracts do teste oficial **também estão no treino** | vazamento: a métrica de teste fica distorcida | esses textos são removidos do teste |

Depois do tratamento: treino 8.028 · validação 1.417 (split estratificado) · teste 1.782.

## Decisão arquitetural (nuvem)

**Cenário:** triagem de laudos na chegada ao hospital. O valor está em priorizar um caso urgente *no momento* em que o laudo é emitido, então a inferência é **real-time (online)**, e não batch. O batch continua útil para *retreinar* o modelo e reprocessar o histórico, e esse papel fica com o Airflow.

**Proposta na AWS:**

```
Sistema do hospital (HIS/RIS) ──HTTPS──▶ API Gateway / ALB ──▶ ECS Fargate (container FastAPI + ONNX Runtime)
                                                                   │   ▲
                                                 métricas /metrics │   │ carrega model.onnx
                                                                   ▼   │
                                         Amazon Managed Prometheus ─▶ Amazon Managed Grafana
                                                                       │
          MWAA (Airflow): ingestão → pré-processamento → treino → avaliação → export ONNX ──▶ S3 (registro de modelos)
                                                                                              │
                                              GitHub Actions: lint → test → build → push ECR ─┘
```

| Decisão | Escolha | Por quê |
|---|---|---|
| Padrão de inferência | Real-time, síncrono | Triagem exige resposta imediata; o modelo responde em milissegundos |
| Computação | **ECS Fargate** (container) | O mesmo Dockerfile do projeto roda sem mudança; autoescala por CPU e requisições; sem servidores para gerenciar. Lambda foi descartado por causa do cold start ao carregar o modelo; SageMaker Endpoint funciona, mas custa mais e acrescenta pouco para um modelo linear leve |
| Artefatos | S3 (modelos versionados) + ECR (imagens) | Separa o ciclo do modelo do ciclo do código |
| Retreino | MWAA (Airflow gerenciado) | É a mesma DAG usada localmente |
| Observabilidade | Managed Prometheus + Managed Grafana | São as mesmas métricas e dashboards da stack local |
| Segurança / LGPD | VPC privada, TLS, sem persistir o texto do laudo, logs sem PHI | Laudo é dado sensível de saúde |

**Equivalentes em outras nuvens:** GCP Cloud Run + Cloud Composer + Managed Prometheus; Azure Container Apps + Airflow no AKS/Data Factory + Azure Monitor.

## Modelo e resultados

`TfidfVectorizer` (unigramas + bigramas, 50k termos, `sublinear_tf`) → `LogisticRegression` (`class_weight="balanced"`). Os hiperparâmetros estão em [`params.yaml`](params.yaml). O treino leva cerca de 15 s em CPU.

| Conjunto | Acurácia | Macro-F1 (5 classes) | Macro-F1 urgência (3 níveis) | Recall de "urgente" |
|---|---|---|---|---|
| Validação | 0,755 | 0,750 | 0,808 | 0,827 |
| Teste (sem vazamento) | 0,782 | 0,777 | 0,822 | **0,862** |

A classe mais difícil é *general pathological conditions* (F1 de 0,66): ela é genérica por definição e se sobrepõe às demais. As métricas completas, por classe, ficam em `models/metrics.json`, gerado no treino.

## Latência — baseline

API em container Docker local, backend **sklearn**, 1 worker uvicorn, amostras reais do conjunto de teste (`scripts/load_test.py`, com 20 requisições de aquecimento):

| Cenário | Throughput | p50 | p95 | p99 | Erros |
|---|---|---|---|---|---|
| 500 req, concorrência 1 | 68 req/s | 11,2 ms | 30,4 ms | 34,0 ms | 0 |
| 1000 req, concorrência 4 | 110 req/s | 32,0 ms | 65,6 ms | 102,7 ms | 0 |

Os resultados brutos estão em [`reports/`](reports/). A comparação com o modelo otimizado (ONNX) entra na Etapa 4.

## Como executar

Pré-requisitos: Python 3.11+, [Poetry](https://python-poetry.org/) 2.x e Docker.

```bash
poetry install

# pipeline de dados + treino (gera models/model.joblib e models/metrics.json)
poetry run python -m medical_triage.data.ingest
poetry run python -m medical_triage.data.preprocess
poetry run python -m medical_triage.models.train

# testes e lint
poetry run pytest
poetry run ruff check . && poetry run ruff format --check .

# API local
poetry run uvicorn medical_triage.api.main:app --reload --port 8080

# API em Docker (o modelo precisa estar treinado antes do build)
docker build -t medical-triage .
docker run -d --name triage-api -p 8080:8000 medical-triage

# baseline de latência
poetry run python scripts/load_test.py --url http://localhost:8080 -n 500 -c 1
```

> A porta 8080 do host é usada porque a 8000 costuma estar ocupada por outras ferramentas locais. Dentro do container, a API escuta na porta 8000.

## API

`POST /predict`

```bash
curl -X POST localhost:8080/predict -H "Content-Type: application/json" \
  -d '{"text": "Acute myocardial infarction in patients with coronary artery disease..."}'
```

```json
{
  "condition_label": 4,
  "condition": "cardiovascular_diseases",
  "urgency": "urgente",
  "confidence": 0.9903,
  "probabilities": {"neoplasms": 0.0011, "digestive_system_diseases": 0.0009,
                    "nervous_system_diseases": 0.0039, "cardiovascular_diseases": 0.9903,
                    "general_pathological_conditions": 0.0038},
  "model_backend": "sklearn",
  "inference_ms": 10.6
}
```

`GET /health` informa se a API está no ar, se o modelo foi carregado e qual backend está em uso. A documentação interativa fica em `/docs`.

## Estrutura do repositório

```
src/medical_triage/
  config.py            settings (env) e params.yaml
  triage.py            condição → urgência
  data/ingest.py       download + validação do corpus
  data/preprocess.py   limpeza, resolução de rótulos, split, remoção de vazamento
  models/train.py      pipeline TF-IDF + LogReg, avaliação, persistência
  models/predictor.py  interface de inferência (sklearn; ONNX na Etapa 4)
  api/                 FastAPI (schemas + endpoints)
configs/urgency_map.yaml
params.yaml
scripts/load_test.py   carga e medição de latência
tests/                 pytest (dados, triagem, treino, API)
reports/               resultados de latência
```
