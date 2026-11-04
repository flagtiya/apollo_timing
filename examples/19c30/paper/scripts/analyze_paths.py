#!/usr/bin/env python3
"""Count J1912 solver paths after inference without changing their labels.

Counts describe the recorded candidate portfolio, never global uniqueness.
Reference pulse numbers enter only the final comparison, not path grouping.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import json
import math
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]

def portable_json(value):
    if isinstance(value, dict):
        return {k.replace("${BUNDLE}", str(BUNDLE)): portable_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_json(v) for v in value]
    if isinstance(value, str):
        return value.replace("${BUNDLE}", str(BUNDLE))
    return value

from typing import Any


MJD_QUANTUM = Decimal("0.000000000001")
DEFAULT_OUTPUT = Path.cwd() / "paper-analysis"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return portable_json(json.loads(path.read_text(encoding="utf-8")))


def _tim_path(record: dict[str, Any], portfolio: Path) -> Path | None:
    value = record.get("artifacts", {}).get("tim")
    if not value:
        return None
    path = Path(value)
    return (path if path.is_absolute() else portfolio.parent / path).resolve()


def _pulse_numbers(path: Path) -> dict[tuple[Any, ...], Decimal]:
    """Map full TOA identities to integer PNs, retaining decimal MJD precision.

    The 1e-12-day quantum tolerates PINT text serialization (0.0864 us).
    Name, frequency, error, observatory and duplicate occurrence all matter.
    """
    result: dict[tuple[Any, ...], Decimal] = {}
    occurrences: dict[tuple[Any, ...], int] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields or fields[0] in {"C", "FORMAT", "MODE"} or fields[0].startswith("#"):
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid TOA row at {path}:{line_number}")
        try:
            frequency, mjd, error = map(Decimal, fields[1:4])
            if not all(value.is_finite() for value in (frequency, mjd, error)):
                raise ValueError("Nonfinite TOA identity")
            pn = Decimal(fields[fields.index("-pn") + 1])
            if not pn.is_finite() or pn != pn.to_integral_value():
                raise ValueError("Pulse number must be a finite integer")
        except (InvalidOperation, ValueError, IndexError) as exc:
            raise ValueError(f"Invalid TOA or PN at {path}:{line_number}: {exc}") from exc
        identity = (
            fields[0], frequency, mjd.quantize(MJD_QUANTUM, rounding=ROUND_HALF_EVEN),
            error, fields[4].lower(),
        )
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        result[(*identity, occurrence)] = pn
    if not result:
        raise ValueError(f"No pulse-number TOAs in {path}")
    return result


def _path_signature(pulses: dict[tuple[Any, ...], Decimal]) -> tuple[Any, ...]:
    ordered = sorted(pulses)
    anchor = pulses[ordered[0]]
    return tuple((identity, pulses[identity] - anchor) for identity in ordered)


def _compare_reference(
    pulses: dict[tuple[Any, ...], Decimal], reference: dict[tuple[Any, ...], Decimal]
) -> dict[str, Any]:
    common = pulses.keys() & reference.keys()
    complete = len(common) == len(pulses)
    offsets = {pulses[key] - reference[key] for key in common}
    agrees = complete and len(offsets) == 1
    return {
        "available": True, "candidate_toas": len(pulses), "reference_toas": len(reference),
        "matched_toas": len(common), "candidate_coverage_complete": complete,
        "agrees_modulo_global_integer": agrees,
        "global_integer_offset": str(next(iter(offsets))) if agrees else None,
        "reason": "agrees" if agrees else ("missing_candidate_toas" if not complete else "distinct_path"),
    }


def _candidate_diagnostics(record: dict[str, Any], portfolio: Path, remember: Any) -> dict[str, Any]:
    audit_path = portfolio.parent / "audit.json"
    result: dict[str, Any] = {}
    if audit_path.is_file():
        remember(audit_path)
        audit = _read_json(audit_path)
        selected = audit.get("search", {}).get("selected_conditional_modes", [])
        result["conditional_modes"] = [
            {key: mode.get(key) for key in (
                "cell_id", "strict_score", "log_likelihood", "posterior_weight",
                "lattice_mass_converged", "lattice_log_mass_increment", "lattice_visited_nodes",
            )} for mode in selected
        ]
        result["solver_config"] = audit.get("solver_config")
        result["scientific_contract"] = audit.get("scientific_contract")
    checkpoints = portfolio.parent / "checkpoints" / "annealing"
    matches = []
    for checkpoint in sorted(checkpoints.glob("*.json")):
        remember(checkpoint)
        payload = _read_json(checkpoint)
        for mode in payload.get("modes", []):
            if mode.get("integer_offsets") == record.get("integer_offsets"):
                score = record.get("annealed_total_chi2", record.get("chi2"))
                distance = abs(float(mode.get("total_chi2", math.inf)) - float(score)) if score is not None else math.inf
                matches.append((distance, checkpoint, payload, mode))
    if matches:
        _, checkpoint, payload, mode = min(matches, key=lambda item: item[0])
        trace = mode.get("trace", [])
        result["annealing"] = {
            "checkpoint": str(checkpoint), "completed_schedule": payload.get("diagnostics", {}).get("completed_schedule"),
            "failure_reason": payload.get("diagnostics", {}).get("failure_reason"),
            "mode_converged": mode.get("converged"), "total_chi2": mode.get("total_chi2"),
            "shape_chi2": mode.get("shape_chi2"), "phase_chi2": mode.get("phase_chi2"),
            "final_trace": trace[-1] if trace else None,
            "interpretation": "Trace lattice log_likelihood is not path-level model evidence.",
        }
    return result


def analyze_solver_output(output: Path, reference_tim: Path | None = None) -> dict[str, Any]:
    """Read every nested portfolio and count fits and paths after inference."""
    output = Path(output).resolve()
    artifacts: dict[str, str] = {}

    def remember(path: Path) -> None:
        artifacts[str(path.resolve())] = _sha256(path)

    # Prefer the deepest portfolio for diagnostics when promoted root records
    # refer to the same child candidate TIM. Every source portfolio is retained.
    portfolios = sorted(output.rglob("candidate_portfolio.json"), key=lambda p: (-len(p.parts), str(p)))
    deduplicated: dict[str, dict[str, Any]] = {}
    for portfolio in portfolios:
        remember(portfolio)
        records = _read_json(portfolio)
        if not isinstance(records, list):
            raise ValueError(f"Expected candidate list in {portfolio}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Invalid candidate record in {portfolio}")
            tim = _tim_path(record, portfolio)
            # Exact copied failure records have no TIM, so deduplicate by their
            # complete payload; include fingerprint when one was serialized.
            key = str(tim) if tim else "missing_tim:" + hashlib.sha256(
                json.dumps(record, sort_keys=True, allow_nan=False).encode("utf-8")
            ).hexdigest()
            if key in deduplicated:
                existing = deduplicated[key]
                existing["source_portfolios"].append(str(portfolio))
                if existing["record"] != record:
                    existing["conflicting_copies"] = True
                continue
            deduplicated[key] = {
                "record": record, "portfolio": portfolio, "tim": tim,
                "source_portfolios": [str(portfolio)], "conflicting_copies": False,
            }

    reference = None
    reference_error = None
    if reference_tim is not None:
        reference_tim = Path(reference_tim).resolve()
        try:
            remember(reference_tim)
            reference = _pulse_numbers(reference_tim)
        except (OSError, ValueError) as exc:
            reference_error = str(exc)

    groups: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {"accepted": {}, "rejected": {}}
    candidates = []
    numerical_completed = 0
    for entry in deduplicated.values():
        record, portfolio, tim = entry["record"], entry["portfolio"], entry["tim"]
        chi2 = record.get("chi2")
        completed = isinstance(chi2, (int, float)) and not isinstance(chi2, bool) and math.isfinite(chi2)
        numerical_completed += int(completed)
        accepted = record.get("accepted") is True
        category = "accepted" if accepted else "rejected"
        candidate = {
            key: record.get(key) for key in (
                "candidate_id", "run_fingerprint", "source_cell_id", "destination_cell_id", "chi2",
                "reduced_chi2", "weighted_rms_us", "disk_reload_chi2", "disk_reload_chi2_delta",
                "support_valid", "support_boundary_hit", "active_bounds", "target_reached",
                "integer_offsets", "annealed_total_chi2", "error", "reason",
            )
        }
        candidate.update({
            "record_index": len(candidates), "accepted": accepted, "numerical_fit_completed": completed,
            "tim": str(tim) if tim else None, "source_portfolios": entry["source_portfolios"],
            "conflicting_copies": entry["conflicting_copies"], "pulse_path_error": None,
            "path_group": None, "reference_comparison": {
                "available": False,
                "reason": (reference_error or "reference_not_supplied") if accepted else "rejected_candidate_not_compared",
            },
            "diagnostics": _candidate_diagnostics(record, portfolio, remember),
        })
        try:
            if tim is None:
                raise ValueError("No candidate TIM artifact recorded")
            remember(tim)
            pulses = _pulse_numbers(tim)
            signature = _path_signature(pulses)
            group = groups[category].setdefault(signature, {
                "path_group": f"{category}_{len(groups[category]):04d}",
                "toa_count": len(pulses), "candidate_indices": [], "tim_artifacts": [],
            })
            group["candidate_indices"].append(candidate["record_index"])
            group["tim_artifacts"].append(str(tim))
            candidate["path_group"] = group["path_group"]
            if accepted and reference is not None:
                candidate["reference_comparison"] = _compare_reference(pulses, reference)
        except (OSError, ValueError) as exc:
            candidate["pulse_path_error"] = str(exc)
        par_value = record.get("artifacts", {}).get("par")
        if par_value:
            par = Path(par_value)
            par = par if par.is_absolute() else portfolio.parent / par
            if par.is_file():
                remember(par)
        candidates.append(candidate)

    for signature, group in groups["rejected"].items():
        counterpart = groups["accepted"].get(signature)
        group["matches_accepted_path_group"] = counterpart["path_group"] if counterpart else None
    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    return {
        "schema_version": 1, "output": str(output),
        "interpretation": "Post-inference counts in the recorded finite portfolio; no global uniqueness or evidence weights.",
        "counts": {
            "portfolio_files": len(portfolios), "raw_candidate_records": sum(len(_read_json(p)) for p in portfolios),
            "attempted_numerical_fits": len(candidates), "completed_numerical_fits": numerical_completed,
            "failed_numerical_fits": len(candidates) - numerical_completed,
            "accepted_candidates": len(accepted_candidates), "rejected_candidates": len(candidates) - len(accepted_candidates),
            "distinct_accepted_paths": len(groups["accepted"]), "distinct_rejected_paths": len(groups["rejected"]),
            "rejected_paths_distinct_from_accepted": sum(signature not in groups["accepted"] for signature in groups["rejected"]),
            "unavailable_accepted_paths": sum(c["path_group"] is None for c in accepted_candidates),
            "unavailable_rejected_paths": sum(c["path_group"] is None for c in candidates if not c["accepted"]),
            "accepted_reference_agreements": sum(c["reference_comparison"].get("agrees_modulo_global_integer") is True for c in accepted_candidates),
            "conflicting_candidate_copies": sum(c["conflicting_copies"] for c in candidates),
        },
        "candidates": candidates, "accepted_path_groups": list(groups["accepted"].values()),
        "rejected_path_groups": list(groups["rejected"].values()),
        "provenance": {
            "analyzer_sha256": _sha256(Path(__file__)),
            "toa_identity": "name, Decimal frequency, Decimal MJD rounded to 1e-12 day, Decimal error, observatory, duplicate occurrence",
            "path_equivalence": "Identical complete TOA coverage and one integer global PN offset",
            "reference_tim": str(reference_tim) if reference_tim is not None else None,
            "reference_error": reference_error, "artifact_sha256": dict(sorted(artifacts.items())),
        },
    }


def audit_archived_prefixes(oldroot: Path, outputdir: Path) -> dict[str, Any]:
    """Count archived 12--23 prefixes in a separate output directory."""
    oldroot, outputdir = Path(oldroot).resolve(), Path(outputdir).resolve()
    if outputdir == oldroot or oldroot in outputdir.parents:
        raise ValueError("Reviewer outputs must be outside the archived result tree")
    reference = oldroot / "reference" / "apollo" / "solution.tim"
    cases = []
    for sessions in range(12, 24):
        run = oldroot / "runs" / "prefix" / f"n{sessions:02d}"
        if sessions == 23:
            run = oldroot / "runs" / "full" / "n23"
        analysis = analyze_solver_output(run, reference)
        analysis["session_count"] = sessions
        record = run / "benchmark_record.json"
        if record.is_file():
            payload = _read_json(record)
            analysis["archived_benchmark"] = {
                key: payload.get(key) for key in (
                    "outcome", "outcome_reason", "wall_seconds", "user_cpu_seconds",
                    "system_cpu_seconds", "max_rss_kib", "return_code", "timeout", "host", "command",
                )
            }
            analysis["archived_benchmark"]["rss_interpretation"] = "Historical RUSAGE_CHILDREN cumulative peak; not isolated per-run RSS."
            analysis["provenance"]["artifact_sha256"][str(record)] = _sha256(record)
        cases.append(analysis)
    report = {"schema_version": 1, "archived_root": str(oldroot), "cases": cases}
    outputdir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = outputdir / "archived_prefixes_audit.json", outputdir / "archived_prefixes_audit.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["session_count", "output", *cases[0]["counts"].keys()]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            writer.writerow({"session_count": case["session_count"], "output": case["output"], **case["counts"]})
    manifest = {
        "schema_version": 1, "analyzer_sha256": _sha256(Path(__file__)),
        "outputs": {path.name: _sha256(path) for path in (json_path, csv_path)},
    }
    (outputdir / "archived_prefixes_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-root", type=Path)
    parser.add_argument("--solver-output", type=Path)
    parser.add_argument("--reference-tim", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.archived_root is not None:
        report = audit_archived_prefixes(args.archived_root, args.output_dir)
        print(json.dumps({"cases": len(report["cases"]), "output_dir": str(args.output_dir)}))
    elif args.solver_output is not None:
        print(json.dumps(analyze_solver_output(args.solver_output, args.reference_tim), indent=2, sort_keys=True, allow_nan=False))
    else:
        parser.error("Supply --archived-root or --solver-output")


if __name__ == "__main__":
    main()
