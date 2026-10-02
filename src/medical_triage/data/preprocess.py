"""Limpeza leve do texto, resolucao de rotulos duplicados e split.

A normalizacao e minima de proposito: TfidfVectorizer ja faz lowercase e
tokenizacao, e qualquer transformacao aqui precisa ser repetida na API.

O corpus repete o mesmo abstract com rotulos diferentes quando ele trata de
mais de uma condicao (~17% do treino). Para triagem, cada texto fica com a
condicao mais urgente; empate de urgencia fica com o menor rotulo, o que
prefere uma condicao especifica a "general pathological conditions" (5).
O conjunto de teste oficial tambem compartilha abstracts com o treino; esses
textos sao removidos do teste para a avaliacao nao ser inflada.
"""

import logging
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from medical_triage.config import get_settings, load_params
from medical_triage.data.ingest import LABEL_COL, TEXT_COL, load_raw
from medical_triage.triage import URGENCY_LEVELS, to_urgency

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    r"""Colapsa espacos e translitera para ASCII.

    ASCII garante que o tokenizer do ONNX ([a-zA-Z0-9_]) e o do sklearn (\w, que
    aceita Unicode) vejam os mesmos tokens em laudos com acentos ou letras gregas.
    """
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return _WHITESPACE.sub(" ", ascii_text).strip()


def resolve_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    """Mantem uma linha por texto, com o rotulo mais urgente."""
    ranked = df.assign(_rank=[URGENCY_LEVELS.index(u) for u in to_urgency(df[LABEL_COL])])
    ranked = ranked.sort_values(["_rank", LABEL_COL], ascending=[False, True], kind="stable")
    return ranked.drop_duplicates(subset=TEXT_COL).drop(columns="_rank")


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[LABEL_COL, TEXT_COL]].copy()
    out[TEXT_COL] = out[TEXT_COL].map(clean_text)
    out = resolve_multilabel(out[out[TEXT_COL] != ""])
    return out.sort_index().reset_index(drop=True)


def cleaning_counts(df: pd.DataFrame) -> dict[str, int]:
    """Quanto o clean_frame descarta: vazios apos a limpeza e textos repetidos."""
    texts = df[TEXT_COL].map(clean_text)
    kept = df.assign(**{TEXT_COL: texts})[texts != ""]
    labels_per_text = kept.groupby(TEXT_COL)[LABEL_COL].nunique()
    return {
        "empty_dropped": int((texts == "").sum()),
        "multilabel_texts": int((labels_per_text > 1).sum()),
        "duplicate_rows_dropped": int(len(kept) - len(labels_per_text)),
    }


def drop_seen(df: pd.DataFrame, seen: pd.DataFrame) -> pd.DataFrame:
    """Remove de df os textos que ja aparecem em seen (evita vazamento)."""
    return df[~df[TEXT_COL].isin(set(seen[TEXT_COL]))].reset_index(drop=True)


def split_train_val(
    df: pd.DataFrame, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, val = train_test_split(df, test_size=val_size, stratify=df[LABEL_COL], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def run() -> dict[str, int]:
    settings = get_settings()
    params = load_params()
    train_raw, test_raw = load_raw(settings.raw_dir)
    full_train = clean_frame(train_raw)
    train, val = split_train_val(full_train, params["data"]["validation_size"], params["seed"])
    test_clean = clean_frame(test_raw)
    test = drop_seen(test_clean, full_train)

    # contagens da tabela de qualidade dos dados do README; vao tambem para o XCom
    counts = {}
    for name, raw in {"train": train_raw, "test": test_raw}.items():
        split_counts = cleaning_counts(raw)
        logger.info(
            "%s: %d textos com mais de um rotulo consolidados (%d linhas removidas), "
            "%d vazios apos a limpeza descartados",
            name,
            split_counts["multilabel_texts"],
            split_counts["duplicate_rows_dropped"],
            split_counts["empty_dropped"],
        )
        counts.update({f"{name}_{k}": v for k, v in split_counts.items()})
    counts["test_overlap_dropped"] = len(test_clean) - len(test)
    logger.info(
        "test: %d textos removidos por sobreposicao com treino+val",
        counts["test_overlap_dropped"],
    )

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, frame in {"train": train, "val": val, "test": test}.items():
        frame.to_csv(settings.processed_dir / f"{name}.csv", index=False)
        sizes[name] = len(frame)
    logger.info("Preprocessamento ok: %s", sizes)
    return {**sizes, **counts}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
