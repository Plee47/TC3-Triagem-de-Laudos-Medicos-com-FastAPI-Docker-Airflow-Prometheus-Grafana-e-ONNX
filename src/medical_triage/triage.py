"""Traducao da classe do corpus (1-5) para a urgencia de triagem."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from medical_triage.config import get_settings

URGENCY_LEVELS = ("normal", "atencao", "urgente")


@dataclass(frozen=True)
class TriageLabel:
    label: int
    condition: str
    urgency: str


@lru_cache
def load_urgency_map(path: Path | None = None) -> dict[int, TriageLabel]:
    path = path or get_settings().configs_dir / "urgency_map.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["labels"]
    mapping = {
        int(label): TriageLabel(int(label), entry["condition"], entry["urgency"])
        for label, entry in raw.items()
    }
    invalid = {t.urgency for t in mapping.values()} - set(URGENCY_LEVELS)
    if invalid:
        raise ValueError(f"Urgencias invalidas em {path}: {invalid}")
    return mapping


def triage(label: int) -> TriageLabel:
    """Retorna condicao e urgencia para um rotulo previsto."""
    try:
        return load_urgency_map()[int(label)]
    except KeyError as exc:
        raise ValueError(f"Rotulo desconhecido: {label}") from exc


def to_urgency(labels) -> list[str]:
    """Versao vetorizada de triage() para avaliacao."""
    mapping = load_urgency_map()
    return [mapping[int(label)].urgency for label in labels]
