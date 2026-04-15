from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float = 0.2
    num_ctx: int = 2048
    num_predict: int = 120


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_experiment_config() -> dict[str, Any]:
    return load_json(CONFIG_DIR / "experiment_config.json")


def load_model_configs() -> list[ModelConfig]:
    raw = load_json(CONFIG_DIR / "model_config.json")
    return [ModelConfig(**item) for item in raw["models"]]


def load_field_weights() -> dict[str, float]:
    return load_json(CONFIG_DIR / "field_weights.json")