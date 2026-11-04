"""Hash and atomically write timing artifacts with explicit JSON formatting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from apollo_timing.models.serialization import to_json_ready


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(payload: Any) -> str:
    encoded = json.dumps(
        to_json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(
    path: Path,
    payload: Any,
    *,
    sort_keys: bool = False,
    trailing_newline: bool = False,
    normalize: bool = True,
) -> Path:
    """Replace one artifact only after its payload has serialized successfully.

    Formatting options preserve the existing byte format. Callers supplying
    plain JSON values can disable normalization to reject unsupported types
    and nonfinite values.
    """
    text = json.dumps(
        to_json_ready(payload) if normalize else payload,
        indent=2,
        sort_keys=sort_keys,
        allow_nan=False,
    )
    if trailing_newline:
        text += "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return destination


__all__ = ["json_digest", "sha256_file", "write_json_atomic"]
