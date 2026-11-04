from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import io
from pathlib import Path
from typing import Any, Sequence

from astropy import units as u
import numpy as np
from pint.observatory import Observatory
from pint.pulsar_mjd import time_to_longdouble
from pint.toa import TOA, format_toa_line, read_toa_file

from timing_accelerator.io.artifacts import sha256_file as _sha256, write_json_atomic

_TIM_COMMANDS = {
    "EFAC",
    "EMAX",
    "EMIN",
    "END",
    "EQUAD",
    "FMAX",
    "FMIN",
    "FORMAT",
    "INCLUDE",
    "INFO",
    "JUMP",
    "MODE",
    "NOSKIP",
    "PHA1",
    "PHA2",
    "PHASE",
    "SKIP",
    "TIME",
}


@dataclass(frozen=True)
class ParsedTimSource:
    path: Path
    sha256: str
    input_format: str
    toas: tuple[TOA, ...]


@dataclass(frozen=True)
class TimingUpdateResult:
    merged_tim: Path
    normalized_baseline_tim: Path
    audit_json: Path
    baseline_tim: Path
    update_tim_paths: tuple[Path, ...]
    baseline_toa_count: int
    update_toa_count: int
    duplicate_toa_count: int
    added_toa_count: int
    final_toa_count: int
    start_mjd: float
    end_mjd: float


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    write_json_atomic(
        path,
        payload,
        sort_keys=True,
        trailing_newline=True,
        normalize=False,
    )


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def _looks_like_mjd(value: str) -> bool:
    parsed = _decimal(value)
    return parsed is not None and Decimal("10000") <= parsed <= Decimal("100000")


def _looks_like_frequency(value: str) -> bool:
    parsed = _decimal(value)
    return parsed is not None and Decimal("0") < parsed < Decimal("10000000")


def _is_zero(value: str) -> bool:
    parsed = _decimal(value)
    return parsed is not None and parsed == 0


def _looks_like_nonnegative_integer(value: str) -> bool:
    parsed = _decimal(value)
    return (
        parsed is not None
        and parsed >= 0
        and parsed == parsed.to_integral_value()
    )


def _legacy_record_to_tempo2(fields: list[str], *, path: Path, line_number: int) -> str:
    if (
        len(fields) >= 8
        and _looks_like_nonnegative_integer(fields[1])
        and _is_zero(fields[2])
        and _looks_like_frequency(fields[3])
        and _looks_like_mjd(fields[4])
        and _is_zero(fields[5])
    ):
        normalized = [fields[0], fields[3], fields[4], fields[6], fields[7], *fields[8:]]
    elif (
        len(fields) >= 7
        and _is_zero(fields[1])
        and _looks_like_frequency(fields[2])
        and _looks_like_mjd(fields[3])
        and _is_zero(fields[4])
    ):
        normalized = [fields[0], fields[2], fields[3], fields[5], fields[6], *fields[7:]]
    elif (
        len(fields) >= 6
        and _looks_like_frequency(fields[1])
        and _looks_like_mjd(fields[2])
        and _is_zero(fields[3])
    ):
        normalized = [fields[0], fields[1], fields[2], fields[4], fields[5], *fields[6:]]
    elif (
        len(fields) >= 5
        and _looks_like_frequency(fields[1])
        and _looks_like_mjd(fields[2])
    ):
        normalized = fields
    else:
        raise ValueError(
            f"Unsupported legacy TIM record at {path}:{line_number}: "
            f"{' '.join(fields)}"
        )
    return " ".join(normalized)


def _normalize_tim_text(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    has_tempo2_format = any(
        line.strip().upper().startswith("FORMAT 1")
        for line in lines
        if line.strip() and not line.lstrip().startswith(("#", "C "))
    )
    if has_tempo2_format:
        return "\n".join(line.lstrip() for line in lines) + "\n", "tempo2_format_1"

    normalized = ["FORMAT 1"]
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(("#", "C ")):
            normalized.append(raw_line)
            continue
        fields = stripped.split()
        if fields[0].upper() in _TIM_COMMANDS:
            if fields[0].upper() == "FORMAT":
                continue
            normalized.append(stripped)
            continue
        normalized.append(
            _legacy_record_to_tempo2(fields, path=path, line_number=line_number)
        )
    return "\n".join(normalized) + "\n", "legacy_whitespace_normalized"


def normalize_tempo2_tim_file(*, input_path: Path, output_path: Path) -> Path:
    """Write one canonical Tempo2 TIM without changing active TOA content."""

    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    normalized, _input_format = _normalize_tim_text(source)
    destination = Path(output_path).resolve()
    _atomic_write_text(destination, normalized)
    return destination


def _parse_tim_source(path: Path) -> ParsedTimSource:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    normalized, input_format = _normalize_tim_text(resolved)
    try:
        toas, _commands = read_toa_file(
            io.StringIO(normalized),
            process_includes=True,
            dir=resolved.parent,
        )
    except Exception as exc:
        raise ValueError(f"Failed to parse timing update input '{resolved}': {exc}") from exc
    if not toas:
        raise ValueError(f"Timing update input contains no active TOAs: {resolved}")
    return ParsedTimSource(
        path=resolved,
        sha256=_sha256(resolved),
        input_format=input_format,
        toas=tuple(toas),
    )


def _toa_name(toa: TOA) -> str:
    return str(toa.flags.get("name", "unknown"))


def _toa_mjd_long(toa: TOA) -> np.longdouble:
    return np.longdouble(time_to_longdouble(toa.mjd))


def _toa_identity(toa: TOA) -> tuple[str, str, str, str]:
    mjd = np.format_float_positional(_toa_mjd_long(toa), precision=12, unique=False)
    frequency = f"{float(toa.freq.to_value(u.MHz)):.6f}"
    observatory = Observatory.get(toa.obs).name.lower()
    return (_toa_name(toa), mjd, frequency, observatory)


def _toa_scientific_payload(toa: TOA) -> tuple[str, tuple[tuple[str, str], ...]]:
    error = f"{float(toa.error.to_value(u.us)):.9f}"
    flags = tuple(
        sorted(
            (str(key), str(value))
            for key, value in toa.flags.items()
            if key not in {"format", "name", "tim_jump"}
        )
    )
    return error, flags


def _format_tempo2_toa(toa: TOA) -> str:
    flags = {
        str(key): str(value)
        for key, value in toa.flags.items()
        if key not in {"format", "name", "tim_jump"}
    }
    return format_toa_line(
        toa.mjd,
        toa.error,
        toa.freq,
        Observatory.get(toa.obs),
        name=_toa_name(toa),
        flags=flags,
        format="Tempo2",
    ).rstrip()


def _source_payload(source: ParsedTimSource) -> dict[str, object]:
    mjds = [_toa_mjd_long(toa) for toa in source.toas]
    return {
        "path": str(source.path),
        "sha256": source.sha256,
        "input_format": source.input_format,
        "toa_count": len(source.toas),
        "start_mjd": float(min(mjds)),
        "end_mjd": float(max(mjds)),
    }


def merge_timing_updates(
    *,
    baseline_tim_path: Path,
    update_tim_paths: Sequence[Path],
    output_dir: Path,
) -> TimingUpdateResult:
    """Normalize, deduplicate, and audit an existing TIM plus new TOA files."""

    if not update_tim_paths:
        raise ValueError("At least one update TIM file is required.")
    output = Path(output_dir).resolve()
    baseline = _parse_tim_source(Path(baseline_tim_path))
    updates = tuple(_parse_tim_source(Path(path)) for path in update_tim_paths)

    retained: dict[tuple[str, str, str, str], tuple[TOA, Path]] = {}
    baseline_internal_duplicates = 0
    for toa in baseline.toas:
        identity = _toa_identity(toa)
        if identity in retained:
            previous, previous_path = retained[identity]
            if _toa_scientific_payload(previous) != _toa_scientific_payload(toa):
                raise ValueError(
                    f"Conflicting duplicate TOA '{identity[0]}' in {previous_path} "
                    f"and {baseline.path}."
                )
            baseline_internal_duplicates += 1
            continue
        retained[identity] = (toa, baseline.path)

    baseline_toas = sorted(
        (item[0] for item in retained.values()),
        key=lambda toa: (_toa_mjd_long(toa), _toa_name(toa)),
    )
    normalized_baseline_tim = output / "input.baseline.tim"
    baseline_lines = [
        "FORMAT 1",
        "C Generated by APOLLO audited timing update",
        *(_format_tempo2_toa(toa) for toa in baseline_toas),
    ]
    _atomic_write_text(
        normalized_baseline_tim,
        "\n".join(baseline_lines) + "\n",
    )

    duplicate_count = baseline_internal_duplicates
    added_count = 0
    update_audits: list[dict[str, object]] = []
    for source in updates:
        source_duplicates = 0
        source_added = 0
        for toa in source.toas:
            identity = _toa_identity(toa)
            if identity in retained:
                previous, previous_path = retained[identity]
                if _toa_scientific_payload(previous) != _toa_scientific_payload(toa):
                    raise ValueError(
                        f"Conflicting duplicate TOA '{identity[0]}' in {previous_path} "
                        f"and {source.path}."
                    )
                source_duplicates += 1
                duplicate_count += 1
                continue
            retained[identity] = (toa, source.path)
            source_added += 1
            added_count += 1
        source_payload = _source_payload(source)
        source_payload.update(
            {
                "duplicate_toa_count": source_duplicates,
                "added_toa_count": source_added,
            }
        )
        update_audits.append(source_payload)

    merged_toas = sorted(
        (item[0] for item in retained.values()),
        key=lambda toa: (_toa_mjd_long(toa), _toa_name(toa)),
    )
    merged_tim = output / "input.updated.tim"
    merged_lines = [
        "FORMAT 1",
        "C Generated by APOLLO audited timing update",
        *(_format_tempo2_toa(toa) for toa in merged_toas),
    ]
    _atomic_write_text(merged_tim, "\n".join(merged_lines) + "\n")

    mjds = [_toa_mjd_long(toa) for toa in merged_toas]
    audit_json = output / "timing_update.json"
    audit = {
        "schema_version": 1,
        "operation": "audited_timing_update",
        "baseline": {
            **_source_payload(baseline),
            "internal_duplicate_toa_count": baseline_internal_duplicates,
        },
        "updates": update_audits,
        "input_toa_count": len(baseline.toas) + sum(len(item.toas) for item in updates),
        "update_toa_count": sum(len(item.toas) for item in updates),
        "duplicate_toa_count": duplicate_count,
        "added_toa_count": added_count,
        "final_toa_count": len(merged_toas),
        "start_mjd": float(min(mjds)),
        "end_mjd": float(max(mjds)),
        "merged_tim": str(merged_tim),
        "merged_tim_sha256": _sha256(merged_tim),
        "normalized_baseline_tim": str(normalized_baseline_tim),
        "normalized_baseline_tim_sha256": _sha256(normalized_baseline_tim),
        "duplicate_identity": "name + MJD(1e-12 d) + frequency(1e-6 MHz) + observatory",
        "conflict_policy": "reject duplicate identities with different uncertainty or flags",
    }
    _atomic_write_json(audit_json, audit)

    return TimingUpdateResult(
        merged_tim=merged_tim,
        normalized_baseline_tim=normalized_baseline_tim,
        audit_json=audit_json,
        baseline_tim=baseline.path,
        update_tim_paths=tuple(item.path for item in updates),
        baseline_toa_count=len(baseline.toas),
        update_toa_count=sum(len(item.toas) for item in updates),
        duplicate_toa_count=duplicate_count,
        added_toa_count=added_count,
        final_toa_count=len(merged_toas),
        start_mjd=float(min(mjds)),
        end_mjd=float(max(mjds)),
    )


__all__ = [
    "TimingUpdateResult",
    "merge_timing_updates",
    "normalize_tempo2_tim_file",
]
