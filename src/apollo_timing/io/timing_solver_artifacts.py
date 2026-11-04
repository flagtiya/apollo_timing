from __future__ import annotations

from apollo_timing.backend.pint_certification import (
    attach_session_pulse_numbers as _attach_session_pulse_numbers,
    candidate_certification_rank as _candidate_certification_rank,
    disk_reload_metrics as _disk_reload_metrics,
)
from apollo_timing.io.solution_artifacts import (
    copy_candidate_to_solution as _copy_candidate_to_solution,
    write_solution_tim as _write_solution_tim,
)
from apollo_timing.io.artifacts import (
    sha256_file as _sha256, json_digest as _json_digest, write_json_atomic,
)
from pathlib import Path
from collections import Counter
from typing import Any, Mapping, Sequence


def resource_identity(path: Path | None) -> Any:
    """Content identity for a file or a local clock-correction directory."""
    if path is None:
        return None
    root = Path(path).resolve()
    if root.is_file():
        return {"path": str(root), "sha256": _sha256(root)}
    if root.is_dir():
        return {"path": str(root), "files": {str(p.relative_to(root)): _sha256(p)
                for p in sorted(root.rglob("*")) if p.is_file()}}
    raise FileNotFoundError(root)


def _format_annealing_failures(
    summaries: Sequence[Mapping[str, Any]],
) -> str:
    details: list[str] = []
    for summary in summaries:
        if int(summary.get("output_mode_count", 0)) > 0:
            continue
        diagnostics = summary.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        reason = str(diagnostics.get("failure_reason") or "unknown annealing failure")
        stage_index = diagnostics.get("failure_stage_index")
        stages = diagnostics.get("stages")
        stage: Mapping[str, Any] | None = None
        if isinstance(stage_index, int) and isinstance(stages, Sequence):
            if 0 <= stage_index < len(stages) and isinstance(
                stages[stage_index], Mapping
            ):
                stage = stages[stage_index]
        temperature = (
            f"tau={float(stage['phase_sigma_cycles']):g}"
            if stage is not None and "phase_sigma_cycles" in stage
            else "tau=unknown"
        )
        exception_text = ""
        if stage is not None:
            failures = stage.get("failure_reasons")
            if isinstance(failures, Sequence) and failures:
                labels = [
                    f"{item[0]} ({item[1]}x)"
                    for item in failures[:2]
                    if isinstance(item, Sequence) and len(item) == 2
                ]
                if labels:
                    exception_text = "; exceptions=" + ", ".join(labels)
        details.append(
            f"{summary.get('mode_id', 'unknown mode')}: {temperature}, "
            f"{reason}{exception_text}"
        )
        if len(details) >= 3:
            break
    return " | ".join(details) or "No finite annealed mode was returned."


def _format_certification_failures(records: Sequence[Mapping[str, Any]]) -> str:
    """Show common candidate errors; full diagnostics remain in the portfolio."""
    reasons = Counter(
        " ".join(str(record.get("error") or "unknown certification failure").split())
        for record in records
        if "chi2" not in record
    )
    details = [
        f"{count}x {reason[:500]}{'...' if len(reason) > 500 else ''}"
        for reason, count in reasons.most_common(3)
    ]
    if len(reasons) > 3:
        details.append(f"{len(reasons) - 3} additional error types in the portfolio")
    return " | ".join(details) or "No candidate error details were returned."


def _atomic_json(path, payload):
    write_json_atomic(path, payload)


__all__ = [
    "resource_identity", "_atomic_json", "_format_annealing_failures", "_sha256",
    "_format_certification_failures",
    "_json_digest", "_attach_session_pulse_numbers", "_candidate_certification_rank",
    "_copy_candidate_to_solution", "_disk_reload_metrics", "_write_solution_tim",
]
