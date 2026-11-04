"""Bounded system hypotheses and common reporting for automatic timing."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from apollo_timing import __version__
from apollo_timing.backend.pint_backend import TimingResourceError, _load_pint_model, _load_pint_toas
from apollo_timing.inference.bayesian_coarse_binary_bootstrap import (
    _isolated_model_text,
)
from apollo_timing.inference.binary_parameter_preparation import (
    prepare_discovered_binary_seed,
)
from apollo_timing.inference.toa_white_noise_certification import (
    evaluate_fixed_path_normalized_likelihood,
)
from apollo_timing.io.timing_input import inspect_timing_system
from apollo_timing.io.timing_solver_artifacts import (
    _atomic_json,
    _json_digest,
    _sha256,
    resource_identity,
)


def _toa_identity(toas: Any) -> list[tuple[Any, ...]]:
    return sorted(
        (str(obs), mjd.scale, float(freq), float(error), float(mjd.jd1), float(mjd.jd2))
        for obs, mjd, freq, error in zip(
            toas.table["obs"],
            toas.table["mjd"],
            toas.get_freqs().value,
            toas.get_errors().value,
        )
    )


def _same_complete_toas(candidate: Any, reference: Any) -> bool:
    left, right = _toa_identity(candidate), _toa_identity(reference)
    if len(left) != len(right):
        return False
    for a, b in zip(left, right, strict=True):
        # Compare two-part times by their difference, not rounded bins. A
        # <1e-14-day serialization shift can straddle any bin boundary.
        if (a[:4] != b[:4] or not all(math.isfinite(v) for v in (*a[2:], *b[2:]))
                or abs((a[4] - b[4]) + (a[5] - b[5])) > 1e-14):
            return False
    return True


def score_timing_candidate(
    result: Any, *, reference_tim: Path, clock_override_path: Path | None
) -> dict[str, Any]:
    """Compare only full-data, connected candidates with covariance normalization."""
    model = _load_pint_model(result.solution_par)
    candidate_toas = _load_pint_toas(
        result.solution_tim, seed_model=model, clock_override_path=clock_override_path
    )
    reference = _load_pint_toas(
        reference_tim, seed_model=model, clock_override_path=clock_override_path
    )
    if not _same_complete_toas(candidate_toas, reference):
        raise ValueError(
            "System hypotheses must use the same complete set of TOAs and reported errors."
        )
    if "pulse_number" not in candidate_toas.table.colnames or not np.all(
        np.isfinite(candidate_toas.table["pulse_number"])
    ):
        raise ValueError("System comparison requires a certified pulse-number path.")
    metrics = evaluate_fixed_path_normalized_likelihood(
        model=model, toas=candidate_toas
    )
    if not math.isfinite(metrics.bic):
        raise ValueError(
            "Non-finite normalized likelihood cannot rank a system hypothesis."
        )
    return metrics.to_payload()


def solve_automatic_timing(
    *,
    par: Path,
    tim: Path,
    output: Path,
    binary_config: Any,
    isolated_config: Any,
    blind_config: Any,
    clock: Path | None,
    resume: bool,
    source_type: str,
    minimum_delta_bic: float = 10.0,
    baseline_par: Path | None = None,
    baseline_tim: Path | None = None,
) -> Any:
    from apollo_timing.pipeline import apollo_timing as entry
    from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
        SOLVER_SCHEMA_VERSION,
        _resume_completed_result,
    )

    if not math.isfinite(minimum_delta_bic) or minimum_delta_bic <= 0:
        raise ValueError("System-selection BIC margin must be positive and finite.")
    output.mkdir(parents=True, exist_ok=True)
    declaration = inspect_timing_system(par)
    declared = declaration.binary_model
    fingerprint = _json_digest(
        dict(
            schema=SOLVER_SCHEMA_VERSION,
            version=__version__,
            par=_sha256(par),
            tim=_sha256(tim),
            source_type=source_type,
            binary=binary_config,
            isolated=isolated_config,
            discovery=blind_config,
            clock=resource_identity(clock),
            minimum_delta_bic=minimum_delta_bic,
            baseline_par=_sha256(baseline_par) if baseline_par else None,
            baseline_tim=_sha256(baseline_tim) if baseline_tim else None,
        )
    )
    if resume and (output / "certification.json").is_file():
        previous = json.loads(
            (output / "certification.json").read_text(encoding="utf-8")
        )
        if previous.get("run_fingerprint") != fingerprint:
            raise RuntimeError("Automatic timing resume inputs or policy changed.")
        completed = _resume_completed_result(output=output, run_fingerprint=fingerprint)
        if completed is not None:
            return completed
    attempts: list[dict[str, Any]] = []
    results: list[tuple[dict[str, Any], Any]] = []
    diagnostics: list[tuple[dict[str, Any], Any]] = []

    def record_result(label: str, result: Any) -> None:
        record = dict(
            hypothesis_id=label,
            search_output=str(result.output_dir),
            output=str(result.output_dir),
            accepted=bool(result.accepted),
            model_name=result.model_name,
            model_family=result.model_family,
        )
        child_certification = json.loads(
            (Path(result.output_dir) / "certification.json").read_text(encoding="utf-8")
        )
        record["run_outcome"] = child_certification.get("run_outcome", {})
        if np.isfinite(result.chi2):
            diagnostics.append((record, result))
        if result.accepted:
            try:
                record.update(
                    score_timing_candidate(
                        result, reference_tim=tim, clock_override_path=clock
                    )
                )
                results.append((record, result))
            except TimingResourceError:
                raise
            except (ArithmeticError, RuntimeError, ValueError, OSError) as error:
                record.update(
                    error=f"{type(error).__name__}: {error}", comparison_eligible=False
                )
        attempts.append(record)

    compare_isolated = source_type == "auto" and declared is None
    if compare_isolated:
        try:
            isolated = entry.solve_isolated_pulsar(
                par_path=par,
                tim_path=tim,
                output_dir=output / "system_candidates" / "isolated",
                config=isolated_config,
                clock_override_path=clock,
                resume=resume,
                baseline_par_path=baseline_par,
                baseline_tim_path=baseline_tim,
            )
            record_result("isolated", isolated)
        except TimingResourceError:
            raise
        except (
            ArithmeticError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            attempts.append(
                dict(
                    hypothesis_id="isolated",
                    accepted=False,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    discovery_source = par
    if declared:
        discovery_source = output / "discovery.input.par"
        discovery_source.write_text(_isolated_model_text(par), encoding="utf-8")
    discovery = None
    discovery_error = None
    try:
        discovery = entry.discover_blind_binary_orbits(
            par_path=discovery_source,
            tim_path=tim,
            output_dir=output / "blind_orbit_discovery",
            config=blind_config,
            clock_override_path=clock,
            resume=resume,
        )
    except TimingResourceError:
        raise
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        discovery_error = f"{type(error).__name__}: {error}"
    if discovery is not None:
        for mode in discovery.modes:
            try:
                seed = prepare_discovered_binary_seed(
                    source_par=par,
                    proposal_par=mode.par_path,
                    output_path=output / "prepared_modes" / f"{mode.mode_id}.par",
                    parameter_values=getattr(mode, "parameter_values", None),
                )
                candidate = entry.solve_bayesian_binary_lattice(
                    par_path=seed,
                    tim_path=tim,
                    output_dir=output / "system_candidates" / mode.mode_id,
                    config=binary_config,
                    clock_override_path=clock,
                    resume=resume,
                    baseline_par_path=baseline_par,
                    baseline_tim_path=baseline_tim,
                )
                record_result(mode.mode_id, candidate)
            except TimingResourceError:
                raise
            except (
                ArithmeticError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                attempts.append(
                    dict(
                        hypothesis_id=mode.mode_id,
                        accepted=False,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
    incomplete = bool(
        discovery_error
        or any(
            "error" in attempt
            or attempt.get("run_outcome", {}).get("execution") == "incomplete"
            for attempt in attempts
        )
    )
    summary: dict[str, Any] = dict(
        schema_version=SOLVER_SCHEMA_VERSION,
        architecture="apollo_bounded_system_hypotheses",
        run_fingerprint=fingerprint,
        input_model=declared or "UNDECLARED",
        source_type=source_type,
        period_strata_days=blind_config.period_strata_days,
        discovery_error=discovery_error,
        discovery_audit=str(discovery.audit_path) if discovery is not None else None,
        criterion="full_data_normalized_likelihood_BIC_not_model_posterior_probability",
        minimum_delta_bic=minimum_delta_bic,
        candidates=attempts,
        execution="incomplete" if incomplete else "completed",
        uniqueness_claimed=False,
    )
    if not results and not diagnostics:
        summary.update(
            selected_family="unresolved",
            scientific_status="no_certified_system_hypothesis",
        )
        _atomic_json(output / "model_selection.json", summary)
        _atomic_json(output / "audit.json", summary)
        raise RuntimeError(
            "Automatic/explicit binary search did not produce a comparable certified candidate. "
            f"See {output / 'model_selection.json'}; no isolated fallback was invented."
        )
    ranked = (
        sorted(results, key=lambda item: (item[0]["bic"], item[0]["hypothesis_id"]))
        if results
        else sorted(
            diagnostics, key=lambda item: (item[1].chi2, item[0]["hypothesis_id"])
        )
    )
    preferred, result = ranked[0]
    # Family comparisons are binary-vs-isolated, not ELL1-vs-DD representation votes.
    other_family = next(
        (
            item
            for item in ranked
            if (item[0]["model_family"] == "isolated")
            != (preferred["model_family"] == "isolated")
        ),
        None,
    )
    delta = (
        other_family[0]["bic"] - preferred["bic"]
        if results and other_family is not None
        else None
    )
    ambiguous = delta is not None and delta < minimum_delta_bic
    summary.update(
        selected_family=(
            ("isolated" if preferred["model_family"] == "isolated" else "binary")
            if results
            else "unresolved"
        ),
        preferred_hypothesis_id=preferred["hypothesis_id"],
        delta_bic_to_other_family=delta,
        system_status=(
            ("ambiguous" if ambiguous else "preferred_within_searched_scope")
            if results
            else "diagnostic_only"
        ),
        preferred_candidate_role=(
            "certified" if results else "uncertified_lowest_chi2_not_system_evidence"
        ),
        accepted_hypotheses=len(results),
        selected_candidate_output=str(result.output_dir),
    )
    child = Path(result.output_dir)
    certification = json.loads(
        (child / "certification.json").read_text(encoding="utf-8")
    )
    artifacts = {}
    for key, name in (
        ("par", "solution.par"),
        ("tim", "solution.tim"),
        ("residuals_csv", "residuals.csv"),
        ("residuals_png", "residuals.png"),
    ):
        shutil.copy2(
            getattr(
                result, {"par": "solution_par", "tim": "solution_tim"}.get(key, key)
            ),
            output / name,
        )
        artifacts[key] = str(output / name)
    child_outcome = certification.get("run_outcome", {})
    outcome = dict(
        execution=(
            "incomplete"
            if incomplete or child_outcome.get("execution") == "incomplete"
            else "completed"
        ),
        certification="accepted" if results else "rejected",
        ambiguity=(
            "multiple"
            if len(ranked) > 1 or child_outcome.get("ambiguity") == "multiple"
            else "not_assessed"
        ),
        uniqueness_claimed=False,
        best_fit_role="preferred_certified_candidate" if results else "diagnostic_only",
    )
    certification.update(
        architecture=summary["architecture"],
        run_fingerprint=fingerprint,
        model_identity=dict(
            input=declared or "UNDECLARED",
            search=certification.get("model_identity", {}).get(
                "search", result.model_name
            ),
            output=result.model_name,
        ),
        automatic_system_selection=summary,
        run_outcome=outcome,
        artifacts=artifacts,
    )
    if not results:
        certification.update(
            accepted=False,
            target_reached=False,
            solution_claim={
                "status": "diagnostic_only_no_comparable_certified_hypothesis"
            },
        )
    _atomic_json(output / "certification.json", certification)
    _atomic_json(output / "model_selection.json", summary)
    _atomic_json(output / "candidate_portfolio.json", attempts)
    _atomic_json(
        output / "audit.json",
        {
            **summary,
            "status": outcome["execution"],
            "run_outcome": outcome,
            "model_identity": certification["model_identity"],
            "parameter_policy": isolated_config.parameter_policy,
            "inputs": dict(
                par=str(par),
                tim=str(tim),
                par_sha256=_sha256(par),
                tim_sha256=_sha256(tim),
            ),
        },
    )
    files = [
        p
        for p in output.iterdir()
        if p.is_file() and p.name != "artifact_manifest.json"
    ]
    # References to child portfolios remain live dependencies of the top-level result.
    files += [p for p in (output / "system_candidates").rglob("*") if p.is_file()]
    files += [p for p in (output / "prepared_modes").rglob("*") if p.is_file()]
    files += [p for p in (output / "blind_orbit_discovery").rglob("*") if p.is_file()]
    _atomic_json(
        output / "artifact_manifest.json",
        dict(
            schema_version=SOLVER_SCHEMA_VERSION,
            run_fingerprint=fingerprint,
            artifacts={
                str(p.relative_to(output)): dict(path=str(p), sha256=_sha256(p))
                for p in files
            },
        ),
    )
    result = replace(
        result,
        output_dir=output,
        solution_par=output / "solution.par",
        solution_tim=output / "solution.tim",
        residuals_csv=output / "residuals.csv",
        residuals_png=output / "residuals.png",
    )
    if not results:
        result = replace(
            result,
            accepted=False,
            target_reached=False,
            scientific_status="diagnostic_only_no_comparable_certified_hypothesis",
        )
        if hasattr(result, "pint_certified"):
            result = replace(result, pint_certified=False)
    if len(ranked) > 1 or not results:
        from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
            _finalize_observation_plan,
        )

        result = _finalize_observation_plan(
            result=result, config=binary_config, clock_override_path=clock
        )
    return result
