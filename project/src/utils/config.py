from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config into a dictionary."""
    resolved_path = resolve_path(path)
    with resolved_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {resolved_path} must contain a dictionary at top level")
    return payload
