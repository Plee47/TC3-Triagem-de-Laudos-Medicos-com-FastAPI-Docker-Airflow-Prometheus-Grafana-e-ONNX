import json
import shutil
from pathlib import Path

import pytest

from medical_triage.models.registry import promote, quality_gate, read_metrics

GATE = {"min_macro_f1": 0.70, "min_urgente_recall": 0.75, "max_regression": 0.02}


def _metrics(f1: float, recall: float = 0.9) -> dict:
    # teste ruim de proposito: o gate decide so na validacao
    return {
        "val": {"macro_f1": f1, "urgente_recall": recall},
        "test": {"macro_f1": 0.0, "urgente_recall": 0.0},
    }


def test_gate_approves_good_first_model():
    assert quality_gate(_metrics(0.78), None, GATE) == []


@pytest.mark.parametrize(
    "candidate, current, reason",
    [
        (_metrics(0.60), None, "macro_f1"),
        (_metrics(0.78, recall=0.50), None, "urgente_recall"),
        (_metrics(0.74), _metrics(0.78), "macro_f1 0.740 regrediu"),
        # recall de urgente acima do piso absoluto, mas caiu em relacao a producao
        (_metrics(0.78, recall=0.80), _metrics(0.78, recall=0.86), "urgente_recall 0.800 regrediu"),
    ],
)
def test_gate_rejects(candidate, current, reason):
    failures = quality_gate(candidate, current, GATE)
    assert any(reason in f for f in failures)


def test_gate_tolerates_small_regression():
    assert quality_gate(_metrics(0.77, recall=0.89), _metrics(0.78), GATE) == []


def test_promote_replaces_and_archives(tmp_path):
    prod, cand = tmp_path / "prod", tmp_path / "cand"
    prod.mkdir()
    cand.mkdir()
    (prod / "metrics.json").write_text(json.dumps({"v": 1}))
    (cand / "metrics.json").write_text(json.dumps({"v": 2}))

    promote(cand, prod)

    assert read_metrics(prod) == {"v": 2}
    archived = list((prod / "archive").glob("*/metrics.json"))
    assert len(archived) == 1 and json.loads(archived[0].read_text()) == {"v": 1}


def test_promote_failure_keeps_old_metrics_and_full_archive(tmp_path, monkeypatch):
    from medical_triage.models import registry

    prod, cand = tmp_path / "prod", tmp_path / "cand"
    prod.mkdir()
    cand.mkdir()
    names = ["metrics.json", "model.joblib", "model.onnx"]
    for name in names:
        (prod / name).write_text(json.dumps({"v": 1}))
        (cand / name).write_text(json.dumps({"v": 2}))

    real_copy = shutil.copy2

    def copy_failing_on_onnx(src, dst):
        if Path(src) == cand / "model.onnx":
            raise OSError("disco cheio")
        return real_copy(src, dst)

    # falha no meio da promocao: model.joblib ja foi trocado, model.onnx nao
    monkeypatch.setattr(registry.shutil, "copy2", copy_failing_on_onnx)
    with pytest.raises(OSError):
        promote(cand, prod)

    # metrics.json e trocado por ultimo, entao ainda descreve o modelo antigo
    assert read_metrics(prod) == {"v": 1}
    (archive,) = (prod / "archive").iterdir()
    assert {p.name: json.loads(p.read_text()) for p in archive.iterdir()} == {
        name: {"v": 1} for name in names
    }


def test_promote_empty_candidate_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote(tmp_path, tmp_path / "prod")


def test_read_metrics_missing_returns_none(tmp_path):
    assert read_metrics(tmp_path) is None


def test_prune_archive_keeps_most_recent(tmp_path):
    from medical_triage.models.registry import prune_archive

    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        (tmp_path / stamp).mkdir()
    removed = prune_archive(tmp_path, keep=2)
    assert [p.name for p in removed] == ["20260101T000000Z"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["20260102T000000Z", "20260103T000000Z"]
