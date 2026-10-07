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


def test_download_is_atomic(tmp_path, monkeypatch):
    import io

    from medical_triage.data import ingest

    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b"condition_label,medical_abstract\n1,x\n")

    monkeypatch.setattr(ingest.urllib.request, "urlopen", fake_urlopen)
    (tmp_path / "old.csv").write_text("versao antiga")

    ingest.download_dataset(tmp_path, "http://x", ["new.csv", "old.csv"])

    assert (tmp_path / "new.csv").read_text().startswith("condition_label")
    assert (tmp_path / "old.csv").read_text().startswith("condition_label")
    assert calls == [
        ("http://x/new.csv", ingest.DOWNLOAD_TIMEOUT_S),
        ("http://x/old.csv", ingest.DOWNLOAD_TIMEOUT_S),
    ]
    assert not list(tmp_path.glob("*.part"))


def test_run_uses_local_files_without_network(tmp_path, monkeypatch):
    from medical_triage.config import get_settings
    from medical_triage.data import ingest

    def no_network(*_args, **_kwargs):
        raise AssertionError("a ingestao padrao nao deve acessar a rede")

    monkeypatch.setattr(ingest.urllib.request, "urlopen", no_network)
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [f"{1 + i % 5},abstract number {i}" for i in range(ingest.MIN_ROWS)]
    for name in ("medical_tc_train.csv", "medical_tc_test.csv"):
        (raw / name).write_text("condition_label,medical_abstract\n" + "\n".join(rows))
    labels = "\n".join(f"{i},c{i}" for i in range(1, 6))
    (raw / "medical_tc_labels.csv").write_text("condition_label,condition_name\n" + labels)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        train, test = ingest.run()
    finally:
        get_settings.cache_clear()
    assert len(train) == len(test) == ingest.MIN_ROWS


def test_preprocess_run_splits_are_disjoint_and_leak_free(tmp_path, monkeypatch):
    from medical_triage.config import get_settings
    from medical_triage.data import preprocess

    raw = tmp_path / "raw"
    raw.mkdir()
    words = {1: "tumor", 2: "liver", 3: "brain", 4: "cardiac", 5: "syndrome"}
    base = [(label, f"{w} abstract {i}") for label, w in words.items() for i in range(10)]
    # mesmo texto com rotulos 1 e 4 (difere so no espaco) e um texto que fica vazio
    train_rows = base + [(1, "shared  abstract"), (4, "shared abstract"), (2, "心肌梗死")]
    test_rows = [(label, f"{w} test {i}") for label, w in words.items() for i in range(2)]
    # o teste repete todo o treino com espacos a mais: depois da limpeza e vazamento,
    # e tem que sair mesmo o que cair na validacao
    test_rows += [(label, f"  {text.replace(' ', '   ')} ") for label, text in base]
    for name, rows in {
        "medical_tc_train.csv": train_rows,
        "medical_tc_test.csv": test_rows,
    }.items():
        pd.DataFrame(rows, columns=[LABEL_COL, TEXT_COL]).to_csv(raw / name, index=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        counts = preprocess.run()
    finally:
        get_settings.cache_clear()

    splits = {n: pd.read_csv(tmp_path / "processed" / f"{n}.csv") for n in ("train", "val", "test")}
    texts = {n: set(df[TEXT_COL]) for n, df in splits.items()}
    assert not texts["train"] & texts["val"]
    assert not texts["train"] & texts["test"]
    assert not texts["val"] & texts["test"]
    full_train = pd.concat([splits["train"], splits["val"]])
    assert full_train.loc[full_train[TEXT_COL] == "shared abstract", LABEL_COL].tolist() == [4]
    assert "cardiac abstract 3" not in texts["test"]
    assert counts["test"] == len(splits["test"]) == 10
    assert counts["train_multilabel_texts"] == counts["train_duplicate_rows_dropped"] == 1
    assert counts["train_empty_dropped"] == 1
    assert counts["test_overlap_dropped"] == len(base)
    assert all(type(v) is int for v in counts.values())  # vai para o XCom da DAG


def test_missing_local_files_explain_how_to_restore(tmp_path):
    from medical_triage.data.ingest import check_local_files

    with pytest.raises(FileNotFoundError, match="--download"):
        check_local_files(tmp_path, ["medical_tc_train.csv"])


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
