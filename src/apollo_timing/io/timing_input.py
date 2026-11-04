"""Inspect system declarations without requiring a complete PINT model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from apollo_timing.io.par_compatibility import _active_parameter_line


@lru_cache(maxsize=1)
def _orbital_parameter_names() -> frozenset[str]:
    from pint.models.model_builder import AllComponents

    return frozenset(
        str(name).upper()
        for component_name, component in AllComponents().components.items()
        if component_name.startswith("Binary")
        for name in component.params
    ) - {"BINARY"}


@dataclass(frozen=True)
class TimingSystemDeclaration:
    binary_model: str | None
    orphan_orbit_parameters: tuple[str, ...]


def inspect_timing_system(path: Path) -> TimingSystemDeclaration:
    active = [
        item
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if (item := _active_parameter_line(line)) is not None
    ]
    if any(key.startswith("BINARY") and key != "BINARY" for key, _ in active):
        raise ValueError("Unsupported system declaration; expected a single BINARY model.")
    entries = [parts for key, parts in active if key == "BINARY"]
    if len(entries) > 1 or any(len(parts) < 2 for parts in entries):
        raise ValueError("Expected at most one nonempty BINARY declaration.")
    binary_model = entries[0][1] if entries else None
    orphan = (
        tuple(sorted({name for name, _ in active} & _orbital_parameter_names()))
        if binary_model is None
        else ()
    )
    return TimingSystemDeclaration(binary_model, orphan)
