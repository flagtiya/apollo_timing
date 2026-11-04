from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from apollo_timing.models.contracts import to_plain_data


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def atomic_write_json(path: Path, payload: Any) -> Path:
    text = json.dumps(to_plain_data(payload), indent=2, allow_nan=False) + "\n"
    return atomic_write_text(path, text)
