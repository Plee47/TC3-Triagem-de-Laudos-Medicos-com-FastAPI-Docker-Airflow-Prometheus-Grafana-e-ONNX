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

from medical_triage.config import METRICS_FILE, ONNX_MODEL_FILE, SKLEARN_MODEL_FILE

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
    """Retorna os motivos de reprovacao (lista vazia = aprovado).

    Decide na validacao: o teste fica so como estimativa final, sem influir na promocao.
    """
    val = candidate["val"]
    failures = []
    if val["macro_f1"] < gate["min_macro_f1"]:
        failures.append(f"macro_f1 {val['macro_f1']:.3f} < {gate['min_macro_f1']}")
    if val["urgente_recall"] < gate["min_urgente_recall"]:
        failures.append(
            f"urgente_recall {val['urgente_recall']:.3f} < {gate['min_urgente_recall']}"
        )
    if current is not None:
        for metric in ("macro_f1", "urgente_recall"):
            floor = current["val"][metric] - gate["max_regression"]
            if val[metric] < floor:
                failures.append(f"{metric} {val[metric]:.3f} regrediu abaixo de {floor:.3f}")
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
    """Copia os artefatos do candidato para producao, arquivando os anteriores.

    Arquiva o conjunto atual inteiro antes de trocar qualquer arquivo. Cada
    arquivo e trocado de forma atomica e o metrics.json vai por ultimo: se a
    promocao parar no meio, ele ainda descreve o modelo antigo.
    """
    # metrics.json por ultimo: e o marcador de que a promocao terminou
    artifacts = sorted(
        (p for p in candidate_dir.iterdir() if p.is_file()),
        key=lambda p: (p.name == METRICS_FILE, p.name),
    )
    if not artifacts:
        raise FileNotFoundError(f"Nenhum artefato em {candidate_dir}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # snapshot do conjunto de producao inteiro, mesmo o que o candidato nao traz
    names = {SKLEARN_MODEL_FILE, ONNX_MODEL_FILE, METRICS_FILE} | {p.name for p in artifacts}
    current = sorted(prod_dir / n for n in names if (prod_dir / n).exists())
    if current:
        archive = prod_dir / "archive" / stamp
        archive.mkdir(parents=True, exist_ok=True)
        for path in current:
            shutil.copy2(path, archive / path.name)
    promoted = []
    for src in artifacts:
        dst = prod_dir / src.name
        # copia para temporario e troca: a API nunca le um arquivo pela metade
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        promoted.append(dst)
    prune_archive(prod_dir / "archive")
    logger.info("Promovidos %s para %s", [p.name for p in promoted], prod_dir)
    return promoted
