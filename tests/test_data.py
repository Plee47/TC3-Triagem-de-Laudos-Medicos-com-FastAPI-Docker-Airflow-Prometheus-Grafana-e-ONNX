import pandas as pd
import pytest

from medical_triage.data.ingest import LABEL_COL, TEXT_COL, validate_dataset
from medical_triage.data.preprocess import (
    clean_frame,
    clean_text,
    drop_seen,
    resolve_multilabel,
    split_train_val,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("  heart\n\tfailure   case ") == "heart failure case"


def test_clean_frame_drops_duplicates_and_empty():
    df = pd.DataFrame({LABEL_COL: [1, 1, 2], TEXT_COL: ["a  b", "a b", "   "]})
    assert clean_frame(df)[TEXT_COL].tolist() == ["a b"]


def test_multilabel_keeps_most_urgent():
    # 5 (normal), 1 (atencao), 4 (urgente) para o mesmo texto -> fica 4
    df = pd.DataFrame({LABEL_COL: [5, 1, 4, 2], TEXT_COL: ["x", "x", "x", "y"]})
    out = resolve_multilabel(df).set_index(TEXT_COL)[LABEL_COL]
    assert out.to_dict() == {"x": 4, "y": 2}


def test_multilabel_tie_prefers_specific_condition():
    # 2 e 5 sao "normal"; 3 e 4 sao "urgente" -> menor rotulo
    df = pd.DataFrame({LABEL_COL: [5, 2, 4, 3], TEXT_COL: ["a", "a", "b", "b"]})
    out = resolve_multilabel(df).set_index(TEXT_COL)[LABEL_COL]
    assert out.to_dict() == {"a": 2, "b": 3}


def test_drop_seen_removes_overlap():
    seen = pd.DataFrame({LABEL_COL: [1], TEXT_COL: ["a"]})
    df = pd.DataFrame({LABEL_COL: [1, 2], TEXT_COL: ["a", "b"]})
    assert drop_seen(df, seen)[TEXT_COL].tolist() == ["b"]


def test_split_is_stratified(tiny_df):
    train, val = split_train_val(tiny_df, val_size=0.2, seed=42)
    assert len(train) + len(val) == len(tiny_df)
    assert set(val[LABEL_COL]) == {1, 2, 3, 4, 5}


def test_validate_accepts_good_data(tiny_df):
    validate_dataset(tiny_df, {1, 2, 3, 4, 5}, min_rows=10)


@pytest.mark.parametrize(
    "df, msg",
    [
        (pd.DataFrame({"x": [1]}), "Colunas ausentes"),
        (pd.DataFrame({LABEL_COL: [1], TEXT_COL: ["a"]}), "minimo"),
        (pd.DataFrame({LABEL_COL: [7] * 20, TEXT_COL: ["a"] * 20}), "inesperados"),
        (pd.DataFrame({LABEL_COL: [1] * 20, TEXT_COL: [" "] * 20}), "vazios"),
    ],
)
def test_validate_rejects_bad_data(df, msg):
    with pytest.raises(ValueError, match=msg):
        validate_dataset(df, {1, 2, 3, 4, 5}, min_rows=10)


def test_download_is_atomic_and_skips_existing(tmp_path, monkeypatch):
    import io

    from medical_triage.data import ingest

    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b"condition_label,medical_abstract\n1,x\n")

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    (tmp_path / "old.csv").write_text("ja existe")

    ingest.download_dataset(tmp_path, "http://x", ["new.csv", "old.csv"])

    assert (tmp_path / "new.csv").read_text().startswith("condition_label")
    assert (tmp_path / "old.csv").read_text() == "ja existe"
    assert calls == [("http://x/new.csv", ingest.DOWNLOAD_TIMEOUT_S)]
    assert not list(tmp_path.glob("*.part"))


def test_interrupted_download_leaves_no_target(tmp_path, monkeypatch):
    from medical_triage.data import ingest

    class Broken:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *_):
            raise ConnectionError("caiu")

    monkeypatch.setattr(ingest.urllib.request, "urlopen", lambda url, timeout: Broken())
    with pytest.raises(ConnectionError):
        ingest.download_dataset(tmp_path, "http://x", ["train.csv"])
    assert not (tmp_path / "train.csv").exists()
