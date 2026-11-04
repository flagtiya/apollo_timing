from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import tempfile
from typing import Iterator, TextIO


@dataclass(frozen=True)
class ParNormalization:
    """One deterministic compatibility edit applied before PINT parsing."""

    source_parameter: str
    target_parameter: str
    source_value: float
    target_value: float
    message: str


@dataclass(frozen=True)
class NormalizedParText:
    text: str
    normalizations: tuple[ParNormalization, ...]


def _active_parameter_line(raw_line: str) -> tuple[str, list[str]] | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.split()
    if not parts or parts[0].upper() == "C":
        return None
    return parts[0].upper(), parts


def _frequency_tokens_from_period(parts: list[str]) -> list[str]:
    try:
        period = float(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError("P0 must contain a numeric period in seconds.") from exc
    if period <= 0.0:
        raise ValueError("P0 must be positive before it can be converted to F0.")

    converted = ["F0", f"{1.0 / period:.17g}"]
    if len(parts) >= 3:
        converted.append(parts[2])
    if len(parts) >= 4:
        try:
            period_error = float(parts[3])
        except ValueError as exc:
            raise ValueError("P0 uncertainty must be numeric.") from exc
        converted.append(f"{abs(period_error) / (period * period):.17g}")
    if len(parts) > 4:
        converted.extend(parts[4:])
    return converted


def normalize_par_text(text: str) -> NormalizedParText:
    """Return PINT-compatible par text without changing the source file.

    TEMPO-style ``P0`` is converted to PINT's canonical ``F0``. If both are
    present they must describe the same spin frequency; the canonical F0 line
    is retained and the redundant P0 line is removed.
    """

    lines = str(text).splitlines()
    active = [item for line in lines if (item := _active_parameter_line(line))]
    p0_entries = [(index, parts) for index, (name, parts) in enumerate(active) if name == "P0"]
    f0_entries = [(index, parts) for index, (name, parts) in enumerate(active) if name == "F0"]
    if len(p0_entries) > 1:
        raise ValueError("A par file may contain at most one active P0 line.")
    if len(f0_entries) > 1:
        raise ValueError("A par file may contain at most one active F0 line.")
    if not p0_entries:
        suffix = "\n" if str(text).endswith("\n") else ""
        return NormalizedParText("\n".join(lines) + suffix, tuple())

    p0_parts = p0_entries[0][1]
    converted = _frequency_tokens_from_period(p0_parts)
    period = float(p0_parts[1])
    frequency = float(converted[1])
    if f0_entries:
        try:
            declared_frequency = float(f0_entries[0][1][1])
        except (IndexError, ValueError) as exc:
            raise ValueError("F0 must contain a numeric spin frequency.") from exc
        scale = max(abs(frequency), abs(declared_frequency), 1.0)
        if abs(frequency - declared_frequency) > 1.0e-12 * scale:
            raise ValueError(
                "P0 and F0 are inconsistent: "
                f"1/P0={frequency:.17g} Hz but F0={declared_frequency:.17g} Hz."
            )

    output: list[str] = []
    inserted = False
    for raw_line in lines:
        parsed = _active_parameter_line(raw_line)
        if parsed is None or parsed[0] != "P0":
            output.append(raw_line)
            continue
        if not f0_entries:
            leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            output.append(leading + " ".join(converted))
            inserted = True

    action = "converted P0 to F0" if inserted else "removed redundant P0"
    normalization = ParNormalization(
        source_parameter="P0",
        target_parameter="F0",
        source_value=period,
        target_value=frequency,
        message=f"{action}: {period:.17g} s -> {frequency:.17g} Hz",
    )
    suffix = "\n" if str(text).endswith("\n") else ""
    return NormalizedParText("\n".join(output) + suffix, (normalization,))


def normalized_par_stream(par_path: str | Path) -> tuple[TextIO, tuple[ParNormalization, ...]]:
    path = Path(par_path)
    normalized = normalize_par_text(path.read_text(encoding="utf-8"))
    stream = StringIO(normalized.text)
    stream.name = str(path)
    return stream, normalized.normalizations


@contextmanager
def pint_compatible_par_path(par_path: str | Path) -> Iterator[tuple[Path, tuple[ParNormalization, ...]]]:
    """Yield a temporary PINT-compatible path while preserving the input."""

    source = Path(par_path).resolve()
    normalized = normalize_par_text(source.read_text(encoding="utf-8"))
    if not normalized.normalizations:
        yield source, tuple()
        return
    with tempfile.TemporaryDirectory(prefix="apollo-par-") as directory:
        target = Path(directory) / source.name
        target.write_text(normalized.text, encoding="utf-8")
        yield target, normalized.normalizations


__all__ = [
    "NormalizedParText",
    "ParNormalization",
    "normalize_par_text",
    "normalized_par_stream",
    "pint_compatible_par_path",
]
