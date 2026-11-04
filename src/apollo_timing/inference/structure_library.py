from __future__ import annotations

from pathlib import Path

import yaml

from apollo_timing.models.contracts import StructureModelConfig


def load_structure_library(
    path: Path,
    allowed_fit_params: set[str] | None = None,
    forbidden_fit_params: set[str] | None = None,
) -> list[StructureModelConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    structures: list[StructureModelConfig] = []
    seen_ids: set[str] = set()
    forbidden = {name.upper() for name in (forbidden_fit_params or set())}

    for item in payload.get("structures", []):
        structure_id = str(item["structure_id"])
        if structure_id in seen_ids:
            raise ValueError(f"Duplicate structure_id '{structure_id}' in '{path}'.")
        seen_ids.add(structure_id)

        requested_active = [str(name).upper() for name in item.get("active_fit_params", [])]
        requested_core = [str(name).upper() for name in item.get("core_params", [])]
        forbidden_requested = sorted(set(requested_active + requested_core) & forbidden)
        if forbidden_requested:
            raise ValueError(
                f"Structure '{structure_id}' requests forbidden fit params: {forbidden_requested}. "
                "Remove these parameters from structure definitions."
            )
        ignored_active: list[str] = []
        ignored_core: list[str] = []

        if allowed_fit_params is None:
            active_fit_params = requested_active
            core_params = requested_core
        else:
            active_fit_params = [name for name in requested_active if name in allowed_fit_params]
            ignored_active = [name for name in requested_active if name not in allowed_fit_params]
            core_params = [name for name in requested_core if name in allowed_fit_params]
            ignored_core = [name for name in requested_core if name not in allowed_fit_params]

        structures.append(
            StructureModelConfig(
                structure_id=structure_id,
                binary_model=(
                    str(item["binary_model"]).upper()
                    if item.get("binary_model") is not None
                    else None
                ),
                active_fit_params=active_fit_params,
                core_params=core_params,
                ignored_fit_params=ignored_active or None,
                ignored_core_params=ignored_core or None,
            )
        )

    return structures
