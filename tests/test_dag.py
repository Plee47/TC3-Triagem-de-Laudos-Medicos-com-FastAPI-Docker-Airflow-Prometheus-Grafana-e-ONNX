"""Valida a DAG com o Airflow real (roda no job `dag-validate` do CI)."""

from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.models import DagBag  # noqa: E402

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder=str(DAGS_DIR))


def test_dag_imports_without_errors(dagbag):
    assert dagbag.import_errors == {}


def test_dag_structure(dagbag):
    dag = dagbag.get_dag("medical_triage_retrain")
    assert dag is not None
    order = ["ingest", "preprocess", "train_candidate", "export_onnx", "quality_gate", "promote"]
    assert set(dag.task_ids) == set(order)
    for upstream, downstream in zip(order, order[1:], strict=False):
        assert downstream in dag.get_task(upstream).downstream_task_ids
    assert dag.catchup is False
    # nasce despausada: o `dags trigger` do README nao fica parado em queued
    assert dag.is_paused_upon_creation is False
