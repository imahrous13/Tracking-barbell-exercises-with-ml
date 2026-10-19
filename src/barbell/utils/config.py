"""Configuration loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["project_root"]).resolve()