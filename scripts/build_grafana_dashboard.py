"""Gera monitoring/grafana/dashboards/medical-triage.json.

Uso: python scripts/build_grafana_dashboard.py

Cores: urgencia usa a paleta de status (normal=good, atencao=warning,
urgente=critical) sempre com rotulo; backends usam slots categoricos fixos
(sklearn=1, onnx=2); percentis usam uma rampa sequencial de um so matiz.
"""

import json
from pathlib import Path

OUT = (
    Path(__file__).resolve().parents[1]
    / "monitoring"
    / "grafana"
    / "dashboards"
    / "medical-triage.json"
)
DS = {"type": "prometheus", "uid": "prometheus"}
# Paleta (tema escuro do Grafana): status reservado para estado, categorico fixo por entidade
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
SKLEARN, ONNX = "#3987e5", "#d95926"
# rampa sequencial: no fundo escuro, mais claro = mais destaque (p99 e o mais critico)
P50, P95, P99 = "#1c5cab", "#3987e5", "#9ec5f4"
ACCENT = "#3987e5"
PREDICT = 'route="/predict"'

panels = []


def next_id():
    return len(panels) + 1


def target(expr, legend="", ref="A", instant=False):
    t = {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}
    if instant:
        t.update(instant=True, range=False)
    return t


def color_override(name, color):
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}],
    }


def add(kind, title, grid, targets, desc="", unit=None, overrides=(), options=None, defaults=None):
    fd = {"unit": unit} if unit else {}
    fd.update(defaults or {})
    panels.append(
        {
            "id": next_id(),
            "type": kind,
            "title": title,
            "description": desc,
            "datasource": DS,
            "gridPos": dict(zip("xywh", grid, strict=True)),
            "targets": targets,
            "fieldConfig": {"defaults": fd, "overrides": list(overrides)},
            "options": options or {},
        }
    )


def row(title, y):
    panels.append(
        {
            "id": next_id(),
            "type": "row",
            "title": title,
            "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
            "panels": [],
        }
    )


def stat(title, grid, expr, unit, desc, color=ACCENT, thresholds=None, decimals=None, legend=""):
    steps = thresholds or [{"color": color, "value": None}]
    defaults = {"color": {"mode": "thresholds"}, "thresholds": {"mode": "absolute", "steps": steps}}
    if decimals is not None:
        defaults["decimals"] = decimals
    add(
        "stat",
        title,
        grid,
        [target(expr, legend, instant=True)],
        desc,
        unit,
        defaults=defaults,
        options={
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "none",
            "textMode": "auto",
            "justifyMode": "center",
        },
    )


def timeseries(title, grid, targets, unit, desc, overrides=()):
    custom = {
        "lineWidth": 2,
        "fillOpacity": 0,
        "showPoints": "never",
        "spanNulls": 60000,  # nao liga pontos atraves de >60 s sem dados
        "axisSoftMin": 0,
    }
    add(
        "timeseries",
        title,
        grid,
        targets,
        desc,
        unit,
        overrides,
        defaults={"custom": custom, "color": {"mode": "palette-classic"}},
        options={
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    )


URGENCY_COLORS = [
    color_override("urgente", CRITICAL),
    color_override("atencao", WARNING),
    color_override("normal", GOOD),
]

# ---- Visao geral ----------------------------------------------------------
row("Visão geral", 0)
stat(
    "Total de requisições",
    (0, 1, 4, 4),
    "sum(increase(http_requests_total[$__range]))",
    "none",
    "Requisições HTTP no intervalo selecionado (exclui /metrics).",
    decimals=0,
)
stat(
    "Requisições/s",
    (4, 1, 4, 4),
    "sum(rate(http_requests_total[1m]))",
    "reqps",
    "Throughput atual (janela de 1 min).",
    decimals=1,
)
stat(
    "Taxa de erro 5xx",
    (8, 1, 4, 4),
    '100 * (sum(rate(http_requests_total{status=~"5.."}[5m])) or vector(0)) '
    "/ sum(rate(http_requests_total[5m]))",
    "percent",
    "Erros de servidor sobre o total (5 min). Alerta acima de 5%.",
    thresholds=[
        {"color": GOOD, "value": None},
        {"color": WARNING, "value": 1},
        {"color": CRITICAL, "value": 5},
    ],
    decimals=2,
)
stat(
    "Latência p95 /predict",
    (12, 1, 4, 4),
    "1000 * histogram_quantile(0.95, sum by (le) "
    "(rate(http_request_duration_seconds_bucket{" + PREDICT + "}[5m])))",
    "ms",
    "Percentil 95 do tempo de resposta HTTP do /predict (5 min). Alerta acima de 200 ms.",
    thresholds=[
        {"color": GOOD, "value": None},
        {"color": WARNING, "value": 100},
        {"color": CRITICAL, "value": 200},
    ],
    decimals=1,
)
stat(
    "Modelo em produção",
    (16, 1, 4, 4),
    "max by (backend, test_macro_f1) (triage_model_info)",
    "none",
    "Backend de inferência e macro-F1 de teste do modelo carregado.",
    legend="{{backend}} · F1 {{test_macro_f1}}",
)
panels[-1]["options"]["textMode"] = "name"
stat(
    "Em andamento",
    (20, 1, 4, 4),
    "sum(http_requests_in_progress)",
    "short",
    "Requisições sendo processadas agora.",
    decimals=0,
)

# ---- Trafego e latencia ---------------------------------------------------
row("Tráfego e latência", 5)
timeseries(
    "Requisições/s por classe de status",
    (0, 6, 12, 8),
    [
        target(
            "sum by (code) (label_replace(rate(http_requests_total[1m]), "
            '"code", "${1}xx", "status", "(.)..")) ',
            "{{code}}",
        )
    ],
    "reqps",
    "Throughput por classe de status HTTP.",
    [color_override("2xx", GOOD), color_override("4xx", SERIOUS), color_override("5xx", CRITICAL)],
)
timeseries(
    "Latência HTTP /predict (p50 · p95 · p99)",
    (12, 6, 12, 8),
    [
        target(
            "1000 * histogram_quantile(" + str(q) + ", sum by (le) "
            "(rate(http_request_duration_seconds_bucket{" + PREDICT + "}[1m])))",
            name,
            ref,
        )
        for q, name, ref in ((0.5, "p50", "A"), (0.95, "p95", "B"), (0.99, "p99", "C"))
    ],
    "ms",
    "Tempo de resposta ponta a ponta medido pelo middleware.",
    [color_override("p50", P50), color_override("p95", P95), color_override("p99", P99)],
)
timeseries(
    "Taxa de erro (%)",
    (0, 14, 12, 8),
    [
        target(
            '100 * (sum(rate(http_requests_total{status=~"' + c + '.."}[1m])) or vector(0)) '
            "/ sum(rate(http_requests_total[1m]))",
            c + "xx",
            ref,
        )
        for c, ref in (("4", "A"), ("5", "B"))
    ],
    "percent",
    "4xx = entrada inválida do cliente (ex.: texto vazio); 5xx = falha do serviço.",
    [color_override("4xx", SERIOUS), color_override("5xx", CRITICAL)],
)
timeseries(
    "Latência de inferência do modelo p95 por backend",
    (12, 14, 12, 8),
    [
        target(
            "1000 * histogram_quantile(0.95, sum by (le, backend) "
            "(rate(model_inference_duration_seconds_bucket[1m])))",
            "{{backend}}",
        )
    ],
    "ms",
    "Só o predict_proba, sem overhead HTTP: compara sklearn e ONNX Runtime.",
    [color_override("sklearn", SKLEARN), color_override("onnx", ONNX)],
)

# ---- Modelo ---------------------------------------------------------------
row("Predições do modelo", 22)
add(
    "bargauge",
    "Predições por urgência (intervalo selecionado)",
    (0, 23, 8, 8),
    [
        target(
            "sum by (urgency) (increase(triage_predictions_total[$__range]))",
            "{{urgency}}",
            instant=True,
        )
    ],
    "Quantos laudos o modelo classificou em cada nível de urgência.",
    "short",
    URGENCY_COLORS,
    options={
        "orientation": "horizontal",
        "displayMode": "basic",
        "showUnfilled": True,
        "valueMode": "text",
        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
    },
    defaults={"decimals": 0, "min": 0, "color": {"mode": "fixed", "fixedColor": ACCENT}},
)
timeseries(
    "Predições/min por urgência",
    (8, 23, 8, 8),
    [target("60 * sum by (urgency) (rate(triage_predictions_total[1m]))", "{{urgency}}")],
    "short",
    "Ritmo de classificação por nível de urgência.",
    URGENCY_COLORS,
)
timeseries(
    "Predições com baixa confiança (< 0,5)",
    (16, 23, 8, 8),
    [
        target(
            '100 * sum(rate(triage_prediction_confidence_bucket{le="0.5"}[5m])) '
            "/ sum(rate(triage_prediction_confidence_count[5m]))",
            "% baixa confiança",
        )
    ],
    "percent",
    "Fração de predições em que a classe vencedora tem probabilidade < 0,5. Baseline "
    "medido nos conjuntos de validação/teste: ~50% (linha tracejada). Subida sustentada "
    "sugere drift nos laudos; alerta acima de 65% por 15 min.",
    [color_override("% baixa confiança", ACCENT)],
)
panels[-1]["fieldConfig"]["defaults"].update(
    min=0,
    max=100,
    thresholds={
        "mode": "absolute",
        "steps": [{"color": "transparent", "value": None}, {"color": WARNING, "value": 65}],
    },
)
panels[-1]["fieldConfig"]["defaults"]["custom"]["thresholdsStyle"] = {"mode": "dashed"}

dashboard = {
    "uid": "medical-triage-api",
    "title": "Triagem de Laudos — API de Inferência",
    "tags": ["medical-triage", "fastapi", "mlops"],
    "timezone": "browser",
    "editable": False,
    "refresh": "5s",
    "time": {"from": "now-15m", "to": "now"},
    "schemaVersion": 41,
    "graphTooltip": 1,
    "panels": panels,
    "templating": {"list": []},
    "annotations": {"list": []},
}
OUT.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len([p for p in panels if p['type'] != 'row'])} paineis -> {OUT}")
