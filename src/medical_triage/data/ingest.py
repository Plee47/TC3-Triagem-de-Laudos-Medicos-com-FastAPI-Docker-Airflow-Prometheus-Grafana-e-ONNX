"""Ingestao do Medical Abstracts TC Corpus.

Os CSVs ficam versionados em data/raw (licenca CC BY-SA 3.0, ver
data/raw/README.md), entao a ingestao le os arquivos locais e nao depende de
rede: treino, CI e DAG rodam sempre sobre os mesmos dados.

Para atualizar os arquivos a partir do repositorio dos autores (mesmo conteudo
do Kaggle): python -m medical_triage.data.ingest --download
"""

import argparse
import logging
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from medical_triage.config import get_settings, load_params

logger = logging.getLogger(__name__)

TEXT_COL = "medical_abstract"
LABEL_COL = "condition_label"
MIN_ROWS = 2000
DOWNLOAD_TIMEOUT_S = 60


def download_dataset(raw_dir: Path, base_url: str, files: list[str]) -> list[Path]:
    """Baixa (sobrescrevendo) os CSVs do repositorio dos autores. So sob demanda."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in files:
        target = raw_dir / name
        url = f"{base_url}/{name}"
        logger.info("Baixando %s", url)
        # baixa para .part e renomeia: um download interrompido nunca substitui
        # nem deixa pela metade o arquivo versionado
        partial = target.with_suffix(target.suffix + ".part")
        with (
            urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp,
            open(partial, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        partial.replace(target)
        paths.append(target)
    return paths


def check_local_files(raw_dir: Path, files: list[str]) -> None:
    missing = [name for name in files if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos do dataset ausentes em {raw_dir}: {missing}. Eles sao versionados "
            "no repositorio; restaure com `git checkout -- data/raw` ou baixe com "
            "`python -m medical_triage.data.ingest --download`."
        )


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


def run(download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    cfg = load_params()["data"]
    if download:
        download_dataset(settings.raw_dir, cfg["base_url"], cfg["files"])
    check_local_files(settings.raw_dir, cfg["files"])
    train, test = load_raw(settings.raw_dir)
    labels = set(pd.read_csv(settings.raw_dir / "medical_tc_labels.csv")[LABEL_COL])
    validate_dataset(train, labels)
    validate_dataset(test, labels, min_rows=1)
    logger.info("Ingestao ok: treino=%d teste=%d", len(train), len(test))
    return train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida (e opcionalmente baixa) o dataset")
    parser.add_argument(
        "--download",
        action="store_true",
        help="rebaixa os CSVs do repositorio dos autores antes de validar",
    )
    logging.basicConfig(level=logging.INFO)
    run(download=parser.parse_args().download)
