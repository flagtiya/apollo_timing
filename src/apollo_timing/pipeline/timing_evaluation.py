"""Exact fixed-model evaluation and bounded integer-only connection search."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
from pint.residuals import Residuals

from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
from apollo_timing.inference.bayesian_binary_lattice import (
    PintPhaseEvaluator,
    detect_phase_model_name,
)
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    _initialize_session_local_pulse_numbers,
    infer_contiguous_source_sessions,
)
from apollo_timing.inference.correlated_integer_lattice import (
    CorrelatedIntegerLatticeConfig,
    solve_correlated_integer_lattice,
)
from apollo_timing.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.inference.bayesian_binary_model_selection import (
    select_binary_model_on_fixed_path,
)
from apollo_timing.io.timing_solver_artifacts import (
    _atomic_json,
    _attach_session_pulse_numbers,
    _disk_reload_metrics,
    _json_digest,
    _sha256,
    _write_solution_tim,
    resource_identity,
)
from apollo_timing.inference.timing_parameter_exploration import (
    explore_parameters_on_fixed_path,
    validate_parameter_bounds,
)
from apollo_timing.reports.residual_plot import write_residual_plot
from apollo_timing.inference.timing_constraints import (
    bind_sky_constraint, bind_parameter_bounds, parameter_bounds_report,
)


def evaluate_timing_model(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: Any,
    clock_override_path: Path | None = None,
    resume: bool = False,
    search_integer_paths: bool = False,
) -> Any:
    """Evaluate or infer integer paths, with only explicitly authorized expansion.

    With missing pulse numbers, pure evaluation is explicitly unconnected.
    Integer-only search uses the full correlated session-intercept likelihood,
    not independent rounding of successive gaps.
    """
    from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
        SOLVER_SCHEMA_VERSION,
        _result_from_certification,
        _resume_completed_result,
        _summarize_outcome,
    )

    par, tim, output = (
        Path(par_path).resolve(),
        Path(tim_path).resolve(),
        Path(output_dir).resolve(),
    )
    config = bind_sky_constraint(config, par)
    model = _load_pint_model(par)
    config = bind_parameter_bounds(config, model)
    optional_absent = tuple(name.upper() for name in config.parameter_exploration.parameters
                            if name.upper() not in model.params or model[name.upper()].value is None)
    fingerprint = _json_digest(
        dict(
            schema=SOLVER_SCHEMA_VERSION,
            task="fixed_search" if search_integer_paths else "evaluate",
            par=_sha256(par),
            tim=_sha256(tim),
            config=config,
            clock=resource_identity(clock_override_path),
        )
    )
    if resume and (output / "certification.json").exists():
        previous = json.loads(
            (output / "certification.json").read_text(encoding="utf-8")
        )
        if previous.get("run_fingerprint") != fingerprint:
            raise RuntimeError(
                "Fixed-model evaluation/search resume inputs or policy changed."
            )
        completed = _resume_completed_result(output=output, run_fingerprint=fingerprint)
        if completed is not None:
            return completed
    output.mkdir(parents=True, exist_ok=True)
    sky_constraint = config.sky_position_constraint
    if sky_constraint is not None and not sky_constraint.report(model)["within_bounds"]:
        raise ValueError("Evaluated model is outside the declared sky region.")
    validate_parameter_bounds(model, config.parameter_exploration)
    declared_free = tuple(model.free_params)
    if search_integer_paths and declared_free:
        raise ValueError(
            "Integer-only search requires all physical parameters to be fixed."
        )
    if (
        search_integer_paths
        and detect_phase_model_name(model) == "ISOLATED"
        and config.isolated_update_model_selection.enabled
    ):
        raise ValueError(
            "The legacy isolated portfolio needs --parameter-policy auto-basic or input fit flags; use --explore-parameter for a fixed-model null."
        )
    toas = _load_pint_toas(
        tim, seed_model=model, clock_override_path=clock_override_path
    )
    supplied = "pulse_number" in toas.table.colnames and np.all(
        np.isfinite(
            np.asarray(
                np.ma.filled(toas.table["pulse_number"], np.nan), dtype=np.longdouble
            )
        )
    )
    for name in declared_free:
        model[name].frozen = True
    for name, lower, upper in config.parameter_exploration.bounds:
        if (
            name in model.params
            and model[name].value is not None
            and not lower <= float(model[name].value) <= upper
        ):
            raise ValueError(f"Fixed {name} conflicts with the supplied bounds.")
    paths: list[Any] = [None]
    connected = bool(supplied)
    lattice_report = None
    evaluator = None
    if search_integer_paths and not supplied:
        sessions = infer_contiguous_source_sessions(toas)
        base = _initialize_session_local_pulse_numbers(toas, model, sessions)
        evaluator = PintPhaseEvaluator(
            model=model,
            toas=toas,
            parameter_names=(),
            sessions=sessions,
            base_pulse_numbers=base,
        )
        profile = evaluator.evaluate_fixed_local_component(())
        intercepts = np.asarray(profile.session_intercepts)
        variances = np.asarray(profile.session_intercept_variances)
        if len(sessions) > 1:
            lattice = solve_correlated_integer_lattice(
                centers=intercepts[1:] - intercepts[0],
                design_matrix=np.ones((len(sessions) - 1, 1)),
                state_mean=[0.0],
                state_covariance=[[variances[0]]],
                noise_variances=variances[1:],
                config=CorrelatedIntegerLatticeConfig(
                    candidate_count=config.maximum_modes_to_certify
                ),
            )
            paths = [candidate.integer_vector for candidate in lattice.candidates]
            lattice_report = dict(
                tail_truncated=lattice.tail_truncated,
                beam_truncated=lattice.beam_truncated,
                top_k_complete=lattice.top_k_complete,
            )
        else:
            paths = [()]
        connected = True
    elif not supplied:
        toas.compute_pulse_numbers(model)
    if not paths:
        raise RuntimeError("Fixed-model integer search retained no paths.")
    records = []
    for index, path in enumerate(paths):
        directory = output / "candidates" / f"fixed_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        candidate_toas = (
            deepcopy(toas)
            if path is None
            else _attach_session_pulse_numbers(
                toas=toas,
                base_pulse_numbers=evaluator.base_pulse_numbers,
                sessions=evaluator.sessions,
                relative_integer_offsets=path,
            )
        )
        candidate_model = deepcopy(model)
        expansion = {"status": "disabled"}
        comparison = {"status": "disabled"}
        covariance = None
        if (
            search_integer_paths
            and config.physical_model_selection.enabled
            and detect_phase_model_name(model) != "ISOLATED"
        ):
            selection = select_binary_model_on_fixed_path(
                baseline_fitter=SimpleNamespace(model=candidate_model),
                certified_toas=candidate_toas,
                baseline_fit_parameters=(),
                observation_span_days=float(np.ptp(toas.get_mjds().value)),
                fit_iterations=config.fit_iterations,
                config=config.physical_model_selection,
                **({"sky_constraint": sky_constraint} if sky_constraint else {}),
                **({"parameter_bounds": config.parameter_exploration.bounds, "optional_absent": optional_absent}
                   if config.parameter_exploration.bounds else {}),
            )
            candidate_model = selection.fitter.model
            comparison = dict(
                status="evaluated",
                selected_candidate_id=selection.selected_candidate_id,
                portfolio=selection.portfolio,
                hypothesis_weights=selection.hypothesis_weights,
            )
        if search_integer_paths and config.parameter_exploration.parameters:
            candidate_model, residuals, expansion, covariance = (
                explore_parameters_on_fixed_path(
                    model=candidate_model,
                    toas=candidate_toas,
                    config=config.parameter_exploration,
                    input_bounds={},
                    **({"sky_constraint": sky_constraint} if sky_constraint else {}),
                )
            )
        else:
            residuals = Residuals(
                candidate_toas,
                candidate_model,
                subtract_mean=True,
                use_abs_phase=True,
                track_mode="use_pulse_numbers",
            )
        outpar = write_pint_model_parfile_roundtrip_safe(
            model=candidate_model, path=directory / "solution.par"
        )
        outtim = _write_solution_tim(
            toas=candidate_toas,
            path=directory / "solution.tim",
            comment="APOLLO fixed-model candidate; read certification.json for path scope",
        )
        csv, png = write_residual_plot(
            outpar,
            outtim,
            directory,
            clock_override_path=clock_override_path,
            artifact_stem="solution",
        )
        disk, disk_reduced, disk_wrms = _disk_reload_metrics(
            outpar, outtim, clock_override_path=clock_override_path
        )
        delta = abs(disk - float(residuals.chi2))
        sky_report = sky_constraint.report(_load_pint_model(outpar)) if sky_constraint else None
        bounds_report = parameter_bounds_report(_load_pint_model(outpar), config.parameter_exploration.bounds,
                                               optional_absent=optional_absent)
        accepted = (
            connected
            and bounds_report["within_bounds"]
            and (sky_report is None or sky_report["within_bounds"])
            and np.isfinite(disk_reduced)
            and disk_reduced < config.strict_reduced_chi2_threshold
            and delta <= config.disk_reload_chi2_tolerance
        )
        records.append(
            dict(
                candidate_id=f"fixed_{index:03d}",
                sky_position_constraint=sky_report,
                parameter_bounds=bounds_report,
                model_name=detect_phase_model_name(candidate_model),
                accepted=bool(accepted),
                target_reached=bool(
                    accepted and disk_reduced < config.target_reduced_chi2_threshold
                ),
                chi2=disk,
                disk_reload_chi2=disk,
                reduced_chi2=disk_reduced,
                weighted_rms_us=disk_wrms,
                disk_reload_chi2_delta=delta,
                fit_parameters=tuple(candidate_model.free_params),
                parameter_exploration=expansion,
                covariance_parameter_names=tuple(candidate_model.free_params),
                binary_model_selection=comparison,
                parameter_covariance=covariance,
                pulse_path_signature=(
                    tuple(
                        int(x)
                        for x in (
                            np.asarray(
                                candidate_toas.table["pulse_number"],
                                dtype=np.longdouble,
                            )
                            - np.longdouble(candidate_toas.table["pulse_number"][0])
                        )
                    )
                    if connected
                    else None
                ),
                artifacts=dict(
                    par=str(outpar),
                    tim=str(outtim),
                    residuals_csv=str(csv),
                    residuals_png=str(png),
                ),
            )
        )
    best = min(records, key=lambda item: (not item["accepted"], item["chi2"]))
    import shutil

    artifacts = {}
    for key, name in (
        ("par", "solution.par"),
        ("tim", "solution.tim"),
        ("residuals_csv", "residuals.csv"),
        ("residuals_png", "residuals.png"),
    ):
        shutil.copy2(best["artifacts"][key], output / name)
        artifacts[key] = str(output / name)
    name = best["model_name"]
    outcome = _summarize_outcome(
        records,
        incomplete=bool(
            lattice_report
            and (lattice_report["tail_truncated"] or lattice_report["beam_truncated"])
        )
        or any(
            item.get("status") in {"failed", "constraint_rejected"}
            for record in records
            for item in record["binary_model_selection"].get("portfolio", ())
        ),
    )
    certification = {
        **best,
        "clock_coverage_warnings": getattr(toas, "apollo_clock_coverage_warnings", []),
        "schema_version": SOLVER_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "architecture": "apollo_fixed_model_timing",
        "engine": (
            "fixed_model_integer_lattice"
            if search_integer_paths
            else "fixed_model_evaluation"
        ),
        "model_family": (
            "isolated"
            if name == "ISOLATED"
            else "binary"
        ),
        "model_identity": dict(input=name, search=name, output=name),
        "task": "solve" if search_integer_paths else "evaluate",
        "continuous_parameters_fitted": bool(best["fit_parameters"]),
        "declared_free_parameters": declared_free,
        "connected": connected,
        "supplied_pulse_numbers": bool(supplied),
        "certification_scope": (
            "fixed_model_session_path_portfolio"
            if search_integer_paths
            else (
                "supplied_path_evaluation"
                if supplied
                else "unconnected_nearest_pulse_diagnostic"
            )
        ),
        "solution_claim": dict(
            status=(
                "qualified_retained_solution"
                if search_integer_paths and best["accepted"]
                else "evaluated_not_refitted"
            )
        ),
        "run_outcome": outcome,
        "lattice": lattice_report,
        "local_path_scope": "model_initialized_local_pulse_numbers",
        "portfolio_counts": dict(
            searched_cells=1 if search_integer_paths else 0,
            annealed_modes=0,
            certified_modes=sum(r["accepted"] for r in records),
        ),
        "artifacts": artifacts,
    }
    _atomic_json(output / "certification.json", certification)
    _atomic_json(output / "candidate_portfolio.json", records)
    _atomic_json(
        output / "audit.json",
        {
            **certification,
            "parameter_contract": dict(
                policy=(
                    "explicit_expansion_only"
                    if search_integer_paths
                    else "no_continuous_fitting"
                ),
                source_par_sha256=_sha256(par),
            ),
        },
    )
    if not search_integer_paths:
        _atomic_json(output / "evaluation.json", certification)
    files = [
        p
        for p in output.rglob("*")
        if p.is_file() and p.name != "artifact_manifest.json"
    ]
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
    return _result_from_certification(output, certification, resumed=False)
