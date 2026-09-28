"""Quality gate e promocao do modelo candidato para producao.

Registro deliberadamente simples (diretorios locais): em nuvem, os mesmos
passos viram prefixos versionados no S3.
"""

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medical_triage.config import METRICS_FILE

logger = logging.getLogger(__name__)

ARCHIVE_KEEP = 5


def read_metrics(model_dir: Path) -> dict[str, Any] | None:
    path = model_dir / METRICS_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def quality_gate(
    candidate: dict[str, Any], current: dict[str, Any] | None, gate: dict[str, float]
) -> list[str]:
    """Retorna os motivos de reprovacao (lista vazia = aprovado)."""
    test = candidate["test"]
    failures = []
    if test["macro_f1"] < gate["min_macro_f1"]:
        failures.append(f"macro_f1 {test['macro_f1']:.3f} < {gate['min_macro_f1']}")
    if test["urgente_recall"] < gate["min_urgente_recall"]:
        failures.append(
            f"urgente_recall {test['urgente_recall']:.3f} < {gate['min_urgente_recall']}"
        )
    if current is not None:
        floor = current["test"]["macro_f1"] - gate["max_regression"]
        if test["macro_f1"] < floor:
            failures.append(f"macro_f1 {test['macro_f1']:.3f} regrediu abaixo de {floor:.3f}")
    return failures


def prune_archive(archive_root: Path, keep: int = ARCHIVE_KEEP) -> list[Path]:
    """Mantem so as `keep` versoes arquivadas mais recentes (nomes sao timestamps)."""
    if not archive_root.exists():
        return []
    versions = sorted(p for p in archive_root.iterdir() if p.is_dir())
    removed = versions[:-keep] if keep else versions
    for path in removed:
        shutil.rmtree(path)
    return removed


def promote(candidate_dir: Path, prod_dir: Path) -> list[Path]:
    """Copia os artefatos do candidato para producao, arquivando os anteriores."""
    artifacts = [p for p in candidate_dir.iterdir() if p.is_file()]
    if not artifacts:
        raise FileNotFoundError(f"Nenhum artefato em {candidate_dir}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = prod_dir / "archive" / stamp
    promoted = []
    for src in artifacts:
        dst = prod_dir / src.name
        if dst.exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, archive / src.name)
        # copia para temporario e troca: a API nunca le um arquivo pela metade
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        promoted.append(dst)
    prune_archive(prod_dir / "archive")
    logger.info("Promovidos %s para %s", [p.name for p in promoted], prod_dir)
    return promoted
