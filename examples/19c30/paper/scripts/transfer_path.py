#!/usr/bin/env python3
"""Transfer the recorded N=16 alternative path to N=15 for a fixed-path check.

This is a diagnostic with a transferred fitted model and integer path. It is
neither a blind search nor a candidate count in the primary validation matrix.
No reference-23 solution enters this calculation.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
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

import shutil
import signal
import time


FIT_NAMES = ("RAJ", "DECJ", "F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_toa_rows(path: Path) -> list[tuple[tuple[object, ...], list[str], str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] in {"C", "FORMAT", "MODE"} or fields[0].startswith("#"):
            continue
        key = (
            fields[0], Decimal(fields[1]), Decimal(fields[2]).quantize(Decimal("1e-12")),
            Decimal(fields[3]), fields[4].lower(),
        )
        rows.append((key, fields, line))
    if len({key for key, _, _ in rows}) != len(rows):
        raise ValueError("Duplicate TOA identity would make path transfer ambiguous")
    return rows


def support_rows(audit: dict, values: dict) -> list[dict]:
    return [
        {"parameter": name, "value": values[name], "lower": lower, "upper": upper,
         "inside": lower <= values[name] <= upper,
         "at_boundary": values[name] == lower or values[name] == upper}
        for name, lower, upper in zip(
            audit["scientific_contract"]["fit_parameters"],
            audit["physical_support_lower"], audit["physical_support_upper"], strict=True,
        )
    ]


def run(oldroot: Path, output: Path, snapshot: Path, ephemeris: Path, timeout: int) -> dict:
    started = time.monotonic()
    oldroot, output, snapshot = oldroot.resolve(), output.resolve(), snapshot.resolve()
    if output == oldroot or oldroot in output.parents:
        raise ValueError("Diagnostic outputs must be outside the archived result tree")
    output.mkdir(parents=True, exist_ok=True)
    prefix_root = oldroot / "runs" / "prefix"
    original_tim = oldroot / "inputs" / "prefixes" / "sessions_15_chronological_prefix.tim"
    run15 = prefix_root / "n15"
    run16 = prefix_root / "n16"
    audit_path = run15 / "audit.json"
    portfolio_path = run16 / "candidate_portfolio.json"
    audit = portable_json(json.loads(audit_path.read_text()))
    records = portable_json(json.loads(portfolio_path.read_text()))
    alternate = next(record for record in records if record["candidate_id"] == "candidate_0001")
    if alternate.get("accepted") is not True:
        raise ValueError("The archived N=16 alternate must already be accepted")
    source_par, source_tim = Path(alternate["artifacts"]["par"]), Path(alternate["artifacts"]["tim"])
    initial15_par = Path(audit["inputs"]["par"])
    primary15_tim = run15 / "solution.tim"
    protected = [original_tim, audit_path, portfolio_path, source_par, source_tim, initial15_par, primary15_tim]
    before = {str(path): sha256(path) for path in protected}
    report = {
        "schema_version": 1, "scope": "transferred_N16_alternate_fixed_path_on_original_N15_prefix",
        "blind_rerun": False, "primary_matrix_evidence": False, "reference_23_used": False,
        "manual_integer_changes": False, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "diagnostic_script_sha256": sha256(Path(__file__)),
        "runtime_environment": {name: os.environ.get(name) for name in (
            "PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TIMING_ACCELERATOR_EPHEM_PATH",
        )},
        "source_candidate_id": alternate["candidate_id"], "source_N16_chi2": alternate["chi2"],
        "fit_parameters": list(FIT_NAMES), "thresholds": {"strict_reduced_chi2": 20.0, "target_reduced_chi2": 10.0, "disk_reload_chi2_tolerance": 0.01},
        "initial_support": support_rows(audit, alternate["fitted_parameters"]),
        "original_sha256_before": before, "status": "preflight", "timeout_seconds": timeout,
    }

    def save() -> None:
        report["wall_seconds"] = time.monotonic() - started
        report["original_sha256_after"] = {str(path): sha256(path) for path in protected}
        report["original_hashes_unchanged"] = before == report["original_sha256_after"]
        report["output_sha256"] = {
            path.name: sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name not in {"report.json", "stdout.log", "stderr.log"}
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

    save()
    if not all(row["inside"] for row in report["initial_support"]):
        report["status"] = "skipped_initial_model_outside_N15_support"
        save()
        return report
    source_rows = {key: fields for key, fields, _ in read_toa_rows(source_tim)}
    prefix_rows = read_toa_rows(original_tim)
    lines, inherited_pns = ["FORMAT 1"], []
    for key, fields, line in prefix_rows:
        match = source_rows[key]
        pn_text = match[match.index("-pn") + 1]
        pn = Decimal(pn_text)
        if pn != pn.to_integral_value():
            raise ValueError("Source pulse numbers must be integers")
        if "-pn" in fields:
            raise ValueError("Original prefix TIM unexpectedly has pulse numbers")
        inherited_pns.append(int(pn))
        lines.append(line + " -pn " + pn_text)
    if len(prefix_rows) != 64:
        raise ValueError("This diagnostic requires the complete 64-TOA N=15 input")
    transferred_tim = output / "input.original_N15_with_N16_alternate_pn.tim"
    transferred_tim.write_text("\n".join(lines) + "\n")
    transferred_par = output / "input_from_N16_alternate.par"
    shutil.copyfile(source_par, transferred_par)
    report["toa_count"] = len(prefix_rows)
    report["transfer"] = {"original_TOA_rows_preserved": True, "all_inherited_PNs_preserved": True, "tim": str(transferred_tim), "par": str(transferred_par)}
    save()

    def alarm_handler(_number: int, _frame: object) -> None:
        raise TimeoutError(f"Diagnostic exceeded {timeout} seconds")

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(max(timeout - int(time.monotonic() - started), 1))
    try:
        import numpy as np
        from astropy import units as u
        import pint.logging
        from pint.fitter import WLSFitter
        from pint.residuals import Residuals
        from timing_accelerator.backend import pint_backend
        from timing_accelerator.backend.pint_certification import disk_reload_metrics
        from timing_accelerator.inference.bayesian_binary_lattice import PintPhaseEvaluator
        from timing_accelerator.inference.pint_solution_artifacts import write_pint_model_parfile_roundtrip_safe

        pint.logging.setup(level="WARNING")
        loaded_source = Path(pint_backend.__file__).resolve()
        if snapshot not in loaded_source.parents:
            raise RuntimeError("Diagnostic must import the frozen source snapshot")
        report["snapshot_backend"] = {"path": str(loaded_source), "sha256": sha256(loaded_source)}
        report["ephemeris"] = {"path": str(ephemeris), "sha256": sha256(ephemeris)}
        evaluator = PintPhaseEvaluator.from_files(par_path=initial15_par, tim_path=original_tim)
        base = np.asarray(evaluator.base_pulse_numbers, dtype=np.longdouble)
        inherited = np.asarray(inherited_pns, dtype=np.longdouble)
        local_rows = []
        for index, session in enumerate(evaluator.sessions):
            indices = np.asarray(session.toa_indices, dtype=int)
            delta = inherited[indices] - base[indices]
            local_rows.append({
                "session_index": index, "toa_indices": indices.tolist(),
                "PN_minus_N15_base": [int(value) for value in delta],
                "local_structure_eligible_at_radius_zero": bool(np.all(delta == delta[0])),
            })
        report["local_structure"] = {
            "basis": "Actual N15 discovered PAR and public PintPhaseEvaluator local base pulse numbers",
            "local_integer_radius": audit["solver_config"]["cell_search"]["local_integer_radius"],
            "all_sessions_eligible_at_radius_zero": all(row["local_structure_eligible_at_radius_zero"] for row in local_rows),
            "sessions": local_rows,
        }
        primary_rows = {key: fields for key, fields, _ in read_toa_rows(primary15_tim)}
        primary_deltas = [
            inherited_pns[index] - int(Decimal(primary_rows[key][primary_rows[key].index("-pn") + 1]))
            for index, (key, _, _) in enumerate(prefix_rows)
        ]
        report["comparison_with_archived_N15_primary"] = {
            "distinct_modulo_global_integer": len(set(primary_deltas)) > 1,
            "PN_differences_in_original_TOA_order": primary_deltas,
        }
        model = pint_backend._load_pint_model(transferred_par)
        for name in model.params:
            if hasattr(model[name], "frozen"):
                model[name].frozen = name not in FIT_NAMES
        if set(model.free_params) != set(FIT_NAMES):
            raise ValueError("Exactly the nine archived fit flags must remain free")
        toas = pint_backend._load_pint_toas(transferred_tim, seed_model=model)
        pinned_before = np.asarray(toas.table["pulse_number"], dtype=np.longdouble).copy()
        if not np.array_equal(pinned_before, inherited):
            raise ValueError("PINT failed to preserve transferred PN order or values")

        def stats(current: object) -> dict:
            residuals = Residuals(toas, current, subtract_mean=True, use_abs_phase=True, track_mode="use_pulse_numbers")
            return {"chi2": float(residuals.chi2), "reduced_chi2": float(residuals.reduced_chi2), "weighted_rms_us": float(residuals.rms_weighted().to_value(u.us))}

        report["prefit"] = stats(model)
        fitter = WLSFitter(toas, model, track_mode="use_pulse_numbers")
        fitter.fit_toas(maxiter=10)
        fitted = fitter.model
        report["postfit"] = stats(fitted)
        report["postfit_parameters"] = {name: float(fitted[name].value) for name in FIT_NAMES}
        report["postfit_support"] = support_rows(audit, report["postfit_parameters"])
        report["pinned_PNs_unchanged_after_fit"] = bool(np.array_equal(pinned_before, np.asarray(toas.table["pulse_number"], dtype=np.longdouble)))
        solution_par = output / "fixed_path_refit.par"
        write_pint_model_parfile_roundtrip_safe(model=fitted, path=solution_par)
        solution_tim = output / "fixed_path_refit.tim"
        shutil.copyfile(transferred_tim, solution_tim)
        disk_chi2, disk_reduced, disk_wrms = disk_reload_metrics(solution_par, solution_tim)
        delta = abs(disk_chi2 - report["postfit"]["chi2"])
        report["disk_reload"] = {"chi2": disk_chi2, "reduced_chi2": disk_reduced, "weighted_rms_us": disk_wrms, "chi2_delta": delta}
        support_ok = all(row["inside"] for row in report["postfit_support"])
        report["diagnostic_strict_quality_gate_passed"] = bool(support_ok and disk_reduced <= 20.0 and delta <= 0.01 and report["pinned_PNs_unchanged_after_fit"])
        report["diagnostic_target_quality_gate_passed"] = bool(support_ok and disk_reduced <= 10.0 and delta <= 0.01 and report["pinned_PNs_unchanged_after_fit"])
        report["status"] = "completed_transferred_fixed_path_diagnostic"
    except Exception as exc:
        report["status"] = "timeout" if isinstance(exc, TimeoutError) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        signal.alarm(0)
        save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ephemeris", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    report = run(args.old_root, args.output, args.snapshot, args.ephemeris, args.timeout)
    print(json.dumps({key: report.get(key) for key in (
        "status", "toa_count", "prefit", "postfit", "disk_reload",
        "diagnostic_strict_quality_gate_passed", "original_hashes_unchanged", "error",
    )}, indent=2))


if __name__ == "__main__":
    main()
