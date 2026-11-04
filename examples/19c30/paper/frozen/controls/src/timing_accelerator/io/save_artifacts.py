from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from timing_accelerator.models.contracts import to_plain_data


def ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(to_plain_data(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
