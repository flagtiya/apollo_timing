"""Convert timing results into portable JSON values."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def to_json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return to_json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        number = value.item()
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, tuple):
        return [to_json_ready(item) for item in value]
    if isinstance(value, list):
        return [to_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    return value


def lattice_log_mass_increment_from_json(value: Any) -> float:
    """Restore the +inf diagnostic used before two lattice budgets are compared."""
    return np.inf if value is None else float(value)


__all__ = ["lattice_log_mass_increment_from_json", "to_json_ready"]
