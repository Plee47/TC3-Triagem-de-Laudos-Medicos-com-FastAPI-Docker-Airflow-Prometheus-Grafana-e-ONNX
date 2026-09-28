import pytest

from medical_triage.triage import URGENCY_LEVELS, load_urgency_map, to_urgency, triage


def test_map_covers_all_five_labels():
    assert set(load_urgency_map()) == {1, 2, 3, 4, 5}


def test_all_urgencies_are_valid_and_used():
    assert {t.urgency for t in load_urgency_map().values()} == set(URGENCY_LEVELS)


def test_cardiovascular_is_urgent():
    assert triage(4).urgency == "urgente"
    assert triage(4).condition == "cardiovascular_diseases"


def test_to_urgency_vectorized():
    assert to_urgency([1, 2, 3]) == ["atencao", "normal", "urgente"]


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        triage(9)
