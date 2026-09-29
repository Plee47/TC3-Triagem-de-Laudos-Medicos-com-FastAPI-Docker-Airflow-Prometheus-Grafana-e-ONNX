"""DAG de retreino do classificador de triagem.

ingest -> preprocess -> train_candidate -> export_onnx -> quality_gate -> promote

A logica fica em medical_triage.* (testavel sem Airflow); a DAG so orquestra.
O candidato e treinado em models/candidate e so substitui o modelo em
producao se passar no quality gate (params.yaml > quality_gate). Reprovado,
o gate faz short-circuit e a promocao fica como "skipped". O export_onnx falha
a DAG se o ONNX divergir do sklearn: nada e promovido sem paridade.
"""

from datetime import datetime, timedelta

from airflow.sdk import dag, task

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="medical_triage_retrain",
    description="Retreino do classificador de laudos com quality gate",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    params={"download": False},
    tags=["medical-triage", "training"],
)
def medical_triage_retrain():
    @task
    def ingest(params=None) -> dict[str, int]:
        from medical_triage.data import ingest as ingest_mod

        # por padrao le os CSVs versionados; download=True rebaixa da fonte
        train, test = ingest_mod.run(download=bool(params and params.get("download")))
        return {"train_rows": len(train), "test_rows": len(test)}

    @task
    def preprocess(_ingested: dict[str, int]) -> dict[str, int]:
        from medical_triage.data import preprocess as preprocess_mod

        return preprocess_mod.run()

    @task
    def train_candidate(_sizes: dict[str, int]) -> dict[str, float]:
        from medical_triage.config import get_settings
        from medical_triage.models import train as train_mod

        metrics = train_mod.run(output_dir=get_settings().candidate_dir)
        return {k: v for k, v in metrics["test"].items() if isinstance(v, float)}

    @task
    def export_onnx(candidate_summary: dict[str, float]) -> dict[str, float]:
        from medical_triage.config import get_settings
        from medical_triage.models import export_onnx as export_mod

        parity = export_mod.run(model_dir=get_settings().candidate_dir)
        return {**candidate_summary, "onnx_max_abs_proba_diff": parity["max_abs_proba_diff"]}

    @task.short_circuit
    def quality_gate(candidate_summary: dict[str, float]) -> bool:
        import logging

        from medical_triage.config import get_settings, load_params
        from medical_triage.models.registry import quality_gate as gate
        from medical_triage.models.registry import read_metrics

        settings = get_settings()
        failures = gate(
            read_metrics(settings.candidate_dir),
            read_metrics(settings.model_dir),
            load_params()["quality_gate"],
        )
        log = logging.getLogger(__name__)
        if failures:
            log.warning("Candidato reprovado: %s", "; ".join(failures))
            return False
        log.info("Candidato aprovado: %s", candidate_summary)
        return True

    @task
    def promote() -> list[str]:
        from medical_triage.config import get_settings
        from medical_triage.models.registry import promote as promote_mod

        settings = get_settings()
        return [p.name for p in promote_mod(settings.candidate_dir, settings.model_dir)]

    summary = export_onnx(train_candidate(preprocess(ingest())))
    quality_gate(summary) >> promote()


medical_triage_retrain()
