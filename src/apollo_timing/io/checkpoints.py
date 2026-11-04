"""Validate solver checkpoint versions and run provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar


CheckpointItem = TypeVar("CheckpointItem")


def require_schema(
    payload: Mapping[str, Any],
    *,
    expected_schema_version: int,
    label: str,
) -> None:
    if payload.get("schema_version") != expected_schema_version:
        raise RuntimeError(
            f"{label} uses certification schema {payload.get('schema_version')!r}; "
            f"APOLLO requires schema {expected_schema_version}. Start a fresh output "
            "directory instead of resuming this result."
        )


def load_checkpoint_items(
    path: Path,
    *,
    expected_schema_version: int,
    run_fingerprint: str,
    item_name: str,
    loader: Callable[[Any], CheckpointItem],
) -> tuple[CheckpointItem, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_schema(
        payload,
        expected_schema_version=expected_schema_version,
        label=f"Checkpoint {path}",
    )
    if payload.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"Checkpoint provenance mismatch: {path}")
    raw_items = payload.get(item_name)
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Checkpoint has no valid {item_name}: {path}")
    return tuple(loader(item) for item in raw_items)


__all__ = ["load_checkpoint_items", "require_schema"]
