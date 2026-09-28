"""Ingestao do Medical Abstracts TC Corpus.

Baixa os CSVs publicos do repositorio dos autores (mesmo conteudo do Kaggle).
Se os arquivos ja estiverem em data/raw (ex.: zip do Kaggle extraido a mao),
o download e pulado.
"""

import logging
import urllib.request
from pathlib import Path

import pandas as pd

from medical_triage.config import get_settings, load_params

logger = logging.getLogger(__name__)

TEXT_COL = "medical_abstract"
LABEL_COL = "condition_label"
MIN_ROWS = 2000


def download_dataset(
    raw_dir: Path, base_url: str, files: list[str], force: bool = False
) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in files:
        target = raw_dir / name
        if target.exists() and not force:
            logger.info("Arquivo ja existe, pulando download: %s", target)
        else:
            url = f"{base_url}/{name}"
            logger.info("Baixando %s", url)
            urllib.request.urlretrieve(url, target)
        paths.append(target)
    return paths


def validate_dataset(df: pd.DataFrame, valid_labels: set[int], min_rows: int = MIN_ROWS) -> None:
    """Falha cedo se o CSV nao tiver o formato esperado."""
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    if len(df) < min_rows:
        raise ValueError(f"Dataset com {len(df)} linhas; minimo {min_rows}")
    unknown = set(df[LABEL_COL].unique()) - valid_labels
    if unknown:
        raise ValueError(f"Rotulos inesperados: {unknown}")
    if df[TEXT_COL].isna().any() or (df[TEXT_COL].str.strip() == "").any():
        raise ValueError("Existem textos vazios no dataset")


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(raw_dir / "medical_tc_train.csv")
    test = pd.read_csv(raw_dir / "medical_tc_test.csv")
    return train, test


def run(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    cfg = load_params()["data"]
    download_dataset(settings.raw_dir, cfg["base_url"], cfg["files"], force=force)
    train, test = load_raw(settings.raw_dir)
    labels = set(pd.read_csv(settings.raw_dir / "medical_tc_labels.csv")[LABEL_COL])
    validate_dataset(train, labels)
    validate_dataset(test, labels, min_rows=1)
    logger.info("Ingestao ok: treino=%d teste=%d", len(train), len(test))
    return train, test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
