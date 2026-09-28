"""Configuracao central: caminhos do projeto, parametros e settings de runtime."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SKLEARN_MODEL_FILE = "model.joblib"
ONNX_MODEL_FILE = "model.onnx"
METRICS_FILE = "metrics.json"


class Settings(BaseSettings):
    """Settings lidas de variaveis de ambiente / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    model_backend: Literal["sklearn", "onnx"] = "sklearn"
    model_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    configs_dir: Path = PROJECT_ROOT / "configs"
    params_path: Path = PROJECT_ROOT / "params.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def sklearn_model_path(self) -> Path:
        return self.model_dir / SKLEARN_MODEL_FILE

    @property
    def onnx_model_path(self) -> Path:
        return self.model_dir / ONNX_MODEL_FILE

    @property
    def candidate_dir(self) -> Path:
        """Onde o retreino grava o modelo antes do quality gate."""
        return self.model_dir / "candidate"

    @property
    def metrics_path(self) -> Path:
        return self.model_dir / METRICS_FILE


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_params(path: Path | None = None) -> dict[str, Any]:
    """Carrega params.yaml (hiperparametros e config de dados)."""
    path = path or get_settings().params_path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
