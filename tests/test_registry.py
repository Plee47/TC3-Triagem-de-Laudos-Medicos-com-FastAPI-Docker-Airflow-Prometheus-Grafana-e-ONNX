import json

import pytest

from medical_triage.models.registry import promote, quality_gate, read_metrics

GATE = {"min_macro_f1": 0.70, "min_urgente_recall": 0.75, "max_regression": 0.02}


def _metrics(f1: float, recall: float = 0.9) -> dict:
    return {"test": {"macro_f1": f1, "urgente_recall": recall}}


def test_gate_approves_good_first_model():
    assert quality_gate(_metrics(0.78), None, GATE) == []


@pytest.mark.parametrize(
    "candidate, current, reason",
    [
        (_metrics(0.60), None, "macro_f1"),
        (_metrics(0.78, recall=0.50), None, "urgente_recall"),
        (_metrics(0.74), _metrics(0.78), "regrediu"),
    ],
)
def test_gate_rejects(candidate, current, reason):
    failures = quality_gate(candidate, current, GATE)
    assert any(reason in f for f in failures)


def test_gate_tolerates_small_regression():
    assert quality_gate(_metrics(0.77), _metrics(0.78), GATE) == []


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


def test_promote_empty_candidate_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote(tmp_path, tmp_path / "prod")


def test_read_metrics_missing_returns_none(tmp_path):
    assert read_metrics(tmp_path) is None
