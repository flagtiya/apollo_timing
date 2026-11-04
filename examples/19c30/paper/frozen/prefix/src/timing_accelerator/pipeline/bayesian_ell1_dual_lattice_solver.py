from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.residuals import Residuals

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
    _update_fit_metadata,
)
from timing_accelerator.inference.active_timing_contracts import to_json_ready
from timing_accelerator.inference.bayesian_ell1_dual_lattice import (
    ELL1AnnealedMode,
    ELL1AnnealingConfig,
    ELL1AnnealingStep,
    ELL1CellSearchConfig,
    ELL1ConditionalMode,
    ELL1_FIT_PARAMETERS,
    anneal_ell1_conditional_mode_diagnosed,
    build_ell1_dual_lattice_search_context,
    search_ell1_orbital_cycle_cell,
)
from timing_accelerator.inference.pint_phase_basis import (
    PROHIBITED_AUTOMATIC_PARAMETERS,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    extract_fitter_parameter_covariance,
    fit_pint_model_with_physical_bounds,
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.reports.residual_plot import write_residual_plot


SOLVER_ARCHITECTURE = "apollo_bayesian_ell1_dual_lattice_annealed_solver"
SOLVER_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BayesianELL1DualLatticeSolverConfig:
    maximum_cell_width: float = 0.25
    support_width_scale: float = 1.0
    minimum_f1_half_width: float = 1.0e-10
    minimum_eccentricity_half_width: float = 5.0e-3
    astrometric_bound_arcsec: float = 300.0
    cell_search: ELL1CellSearchConfig = field(default_factory=ELL1CellSearchConfig)
    annealing: ELL1AnnealingConfig = field(default_factory=ELL1AnnealingConfig)
    maximum_modes_to_anneal: int = 8
    conditional_score_window: float = 2_000.0
    maximum_modes_to_certify: int = 8
    annealed_chi2_window: float = 2_000.0
    fit_iterations: int = 10
    strict_reduced_chi2_threshold: float = 20.0
    target_reduced_chi2_threshold: float = 10.0
    disk_reload_chi2_tolerance: float = 1.0e-2

    def __post_init__(self) -> None:
        positive_floats = (
            self.maximum_cell_width,
            self.support_width_scale,
            self.minimum_f1_half_width,
            self.minimum_eccentricity_half_width,
            self.astrometric_bound_arcsec,
            self.conditional_score_window,
            self.annealed_chi2_window,
            self.strict_reduced_chi2_threshold,
            self.target_reduced_chi2_threshold,
            self.disk_reload_chi2_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("ELL1 solver scales and thresholds must be positive.")
        positive_integers = (
            self.cell_search.orbital_samples_per_cell,
            self.cell_search.phase_samples,
            self.cell_search.modes_per_cell,
            self.cell_search.fast_shape_maximum_evaluations,
            self.maximum_modes_to_anneal,
            self.maximum_modes_to_certify,
            self.fit_iterations,
        )
        if not all(int(value) > 0 for value in positive_integers):
            raise ValueError("ELL1 solver budgets must be positive integers.")
        if self.target_reduced_chi2_threshold > self.strict_reduced_chi2_threshold:
            raise ValueError("The target reduced chi-square cannot exceed acceptance.")


@dataclass(frozen=True)
class BayesianELL1DualLatticeSolverResult:
    output_dir: Path
    solution_par: Path
    solution_tim: Path
    residuals_csv: Path
    residuals_png: Path
    fit_parameters: tuple[str, ...]
    chi2: float
    disk_reload_chi2: float
    reduced_chi2: float
    weighted_rms_us: float
    accepted: bool
    target_reached: bool
    searched_cells: int
    annealed_modes: int
    certified_modes: int
    resumed: bool = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(payload: Any) -> str:
    encoded = json.dumps(
        to_json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_json_ready(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _require_current_schema(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("schema_version") != SOLVER_SCHEMA_VERSION:
        raise RuntimeError(
            f"{label} uses certification schema {payload.get('schema_version')!r}; "
            f"APOLLO requires schema {SOLVER_SCHEMA_VERSION}. Start a fresh output "
            "directory instead of resuming this result."
        )


def _candidate_certification_rank(record: Mapping[str, Any]) -> tuple[bool, bool, float]:
    """Rank accepted, then in-support, then lower-chi-square certifications."""

    chi2 = float(record.get("chi2", np.inf))
    if not np.isfinite(chi2):
        chi2 = np.inf
    return (
        not bool(record.get("accepted", False)),
        not bool(record.get("support_valid", False)),
        chi2,
    )


def _conditional_mode_from_payload(payload: Mapping[str, Any]) -> ELL1ConditionalMode:
    return ELL1ConditionalMode(
        cell_id=str(payload["cell_id"]),
        grid_index=tuple(int(value) for value in payload["grid_index"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        local_integer_offsets=tuple(
            tuple(int(value) for value in component)
            for component in payload["local_integer_offsets"]
        ),
        shape_chi2=float(payload["shape_chi2"]),
        strict_score=float(payload["strict_score"]),
        log_likelihood=float(payload["log_likelihood"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        posterior_weight=float(payload["posterior_weight"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=float(payload["lattice_log_mass_increment"]),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        posterior_in_support=bool(payload["posterior_in_support"]),
        source_cell_id=str(payload.get("source_cell_id", payload["cell_id"])),
        posterior_transport_fraction=float(
            payload.get("posterior_transport_fraction", 1.0)
        ),
    )


def _annealing_step_from_payload(payload: Mapping[str, Any]) -> ELL1AnnealingStep:
    return ELL1AnnealingStep(
        phase_sigma_cycles=float(payload["phase_sigma_cycles"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        lattice_strict_score=float(payload["lattice_strict_score"]),
        lattice_log_likelihood=float(payload["lattice_log_likelihood"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=float(payload["lattice_log_mass_increment"]),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        function_evaluations=int(payload["function_evaluations"]),
        converged=bool(payload["converged"]),
    )


def _annealed_mode_from_payload(payload: Mapping[str, Any]) -> ELL1AnnealedMode:
    return ELL1AnnealedMode(
        cell_id=str(payload["cell_id"]),
        source_grid_index=tuple(
            int(value) for value in payload["source_grid_index"]
        ),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        local_integer_offsets=tuple(
            tuple(int(value) for value in component)
            for component in payload["local_integer_offsets"]
        ),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        converged=bool(payload["converged"]),
        trace=tuple(
            _annealing_step_from_payload(step) for step in payload["trace"]
        ),
        source_cell_id=str(payload.get("source_cell_id", payload["cell_id"])),
    )


def _load_checkpoint_items(
    path: Path,
    *,
    run_fingerprint: str,
    item_name: str,
    loader: Any,
) -> tuple[Any, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_current_schema(payload, label=f"Checkpoint {path}")
    if payload.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError(f"Checkpoint provenance mismatch: {path}")
    raw_items = payload.get(item_name)
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Checkpoint has no valid {item_name}: {path}")
    return tuple(loader(item) for item in raw_items)


def _write_solution_tim(*, toas: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    toas.write_TOA_file(
        str(temporary),
        format="tempo2",
        include_pn=True,
        include_info=True,
        comment="APOLLO PINT-certified integer pulse path",
    )
    temporary.replace(path)
    return path


def _attach_session_pulse_numbers(
    *,
    toas: object,
    base_pulse_numbers: Sequence[float],
    sessions: Sequence[object],
    relative_integer_offsets: Sequence[int],
) -> object:
    session_offsets = np.concatenate(
        (np.zeros(1, dtype=int), np.asarray(relative_integer_offsets, dtype=int))
    )
    if len(session_offsets) != len(sessions):
        raise ValueError("The integer path must contain one offset per session gap.")
    pulse_numbers = np.asarray(base_pulse_numbers, dtype=float).copy()
    for session, offset in zip(sessions, session_offsets, strict=True):
        pulse_numbers[np.asarray(session.toa_indices, dtype=int)] += int(offset)
    certified_toas = deepcopy(toas)
    certified_toas.table["pulse_number"] = pulse_numbers
    certified_toas.table["pulse_number"].unit = u.dimensionless_unscaled
    return certified_toas


def _disk_reload_metrics(
    par_path: Path,
    tim_path: Path,
    *,
    clock_override_path: Path | None,
) -> tuple[float, float, float]:
    model = _load_pint_model(par_path)
    toas = _load_pint_toas(
        tim_path,
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    residuals = Residuals(
        toas,
        model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    return (
        float(residuals.chi2),
        float(residuals.reduced_chi2),
        float(residuals.rms_weighted().to_value(u.us)),
    )


def _result_from_certification(
    output: Path,
    certification: Mapping[str, Any],
    *,
    resumed: bool,
) -> BayesianELL1DualLatticeSolverResult:
    artifacts = certification["artifacts"]
    counts = certification["portfolio_counts"]
    return BayesianELL1DualLatticeSolverResult(
        output_dir=output,
        solution_par=Path(artifacts["par"]),
        solution_tim=Path(artifacts["tim"]),
        residuals_csv=Path(artifacts["residuals_csv"]),
        residuals_png=Path(artifacts["residuals_png"]),
        fit_parameters=tuple(certification["fit_parameters"]),
        chi2=float(certification["chi2"]),
        disk_reload_chi2=float(certification["disk_reload_chi2"]),
        reduced_chi2=float(certification["reduced_chi2"]),
        weighted_rms_us=float(certification["weighted_rms_us"]),
        accepted=bool(certification["accepted"]),
        target_reached=bool(certification["target_reached"]),
        searched_cells=int(counts["searched_cells"]),
        annealed_modes=int(counts["annealed_modes"]),
        certified_modes=int(counts["certified_modes"]),
        resumed=resumed,
    )


def _resume_completed_result(
    *,
    output: Path,
    run_fingerprint: str,
) -> BayesianELL1DualLatticeSolverResult | None:
    certification_path = output / "certification.json"
    manifest_path = output / "artifact_manifest.json"
    if not certification_path.is_file() or not manifest_path.is_file():
        return None
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    _require_current_schema(certification, label="Completed ELL1 certification")
    if certification.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("Completed ELL1 result does not match resume inputs.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_current_schema(manifest, label="Completed ELL1 artifact manifest")
    for name, record in manifest.get("artifacts", {}).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
            raise RuntimeError(f"Resume artifact failed SHA-256: {name}")
    return _result_from_certification(output, certification, resumed=True)


def _select_conditional_modes(
    modes: Sequence[ELL1ConditionalMode],
    config: BayesianELL1DualLatticeSolverConfig,
) -> tuple[ELL1ConditionalMode, ...]:
    ranked = sorted(
        (mode for mode in modes if mode.posterior_in_support),
        key=lambda mode: (mode.strict_score, -mode.log_likelihood, mode.cell_id),
    )
    if not ranked:
        return tuple()
    ceiling = ranked[0].strict_score + config.conditional_score_window
    selected: list[ELL1ConditionalMode] = []
    seen: set[tuple[str, tuple[int, ...], tuple[float, ...]]] = set()
    for mode in ranked:
        if mode.strict_score > ceiling:
            break
        key = (
            mode.cell_id,
            mode.integer_offsets,
            tuple(np.round(mode.normalized_values, decimals=8)),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(mode)
        if len(selected) >= config.maximum_modes_to_anneal:
            break
    return tuple(selected)


def _select_annealed_modes(
    modes: Sequence[ELL1AnnealedMode],
    config: BayesianELL1DualLatticeSolverConfig,
) -> tuple[ELL1AnnealedMode, ...]:
    ranked = sorted(
        (mode for mode in modes if np.isfinite(mode.total_chi2)),
        key=lambda mode: (mode.total_chi2, mode.cell_id, mode.integer_offsets),
    )
    if not ranked:
        return tuple()
    ceiling = ranked[0].total_chi2 + config.annealed_chi2_window
    selected: list[ELL1AnnealedMode] = []
    seen: set[tuple[int, ...]] = set()
    for mode in ranked:
        if mode.total_chi2 > ceiling or mode.integer_offsets in seen:
            continue
        seen.add(mode.integer_offsets)
        selected.append(mode)
        if len(selected) >= config.maximum_modes_to_certify:
            break
    return tuple(selected)


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
            if 0 <= stage_index < len(stages) and isinstance(stages[stage_index], Mapping):
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


def _certify_mode(
    *,
    context: Any,
    seed_par_path: Path,
    mode: ELL1AnnealedMode,
    candidate_id: str,
    candidate_dir: Path,
    config: BayesianELL1DualLatticeSolverConfig,
    clock_override_path: Path | None,
    run_fingerprint: str,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    fit_parameters = tuple(context.evaluator.parameter_names)

    def initialize_mode_model() -> Any:
        initialized = _load_pint_model(seed_par_path)
        for name, value in zip(
            fit_parameters,
            mode.parameter_values,
            strict=True,
        ):
            _set_numeric_model_parameter_value(initialized, name, float(value))
        for name in initialized.params:
            parameter = initialized[name]
            if hasattr(parameter, "frozen"):
                parameter.frozen = name not in fit_parameters
        if (
            set(initialized.free_params) != set(ELL1_FIT_PARAMETERS)
            or len(initialized.free_params) != 9
        ):
            raise RuntimeError(
                "PINT certification violated the nine-parameter ELL1 contract."
            )
        return initialized

    certified_toas = _attach_session_pulse_numbers(
        toas=context.evaluator.toas,
        base_pulse_numbers=context.evaluator.base_pulse_numbers,
        sessions=context.evaluator.sessions,
        relative_integer_offsets=mode.integer_offsets,
    )
    optimizer_mode = "pint_wls"
    optimizer_fallback_reason: str | None = None
    optimizer_function_evaluations: int | None = None
    optimizer_termination_status: int | None = None
    optimizer_termination_message: str | None = None
    optimizer_optimality: float | None = None
    active_bounds: tuple[str, ...] = tuple()
    model = initialize_mode_model()
    try:
        fitter = WLSFitter(certified_toas, model, track_mode="use_pulse_numbers")
        fitter.fit_toas(maxiter=config.fit_iterations)
        fitted_model = fitter.model
        residuals = Residuals(
            certified_toas,
            fitted_model,
            subtract_mean=True,
            use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        parameter_values = np.asarray(
            [float(fitted_model[name].value) for name in fit_parameters],
            dtype=float,
        )
        support_valid = context.support.contains_physical(parameter_values)
        if not support_valid:
            optimizer_fallback_reason = "unconstrained WLS left Bayesian support"
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        fitter = None
        fitted_model = None
        residuals = None
        support_valid = False
        optimizer_fallback_reason = f"{type(error).__name__}: {error}"

    if not support_valid:
        bounded = fit_pint_model_with_physical_bounds(
            model=initialize_mode_model(),
            toas=certified_toas,
            parameter_names=fit_parameters,
            lower_bounds=context.support.physical_lower,
            upper_bounds=context.support.physical_upper,
            initial_values=mode.parameter_values,
            maximum_evaluations=max(100, 50 * int(config.fit_iterations)),
        )
        fitter = None
        fitted_model = bounded.model
        residuals = bounded.residuals
        parameter_covariance = bounded.covariance
        optimizer_mode = "bounded_pint_least_squares"
        optimizer_function_evaluations = bounded.function_evaluations
        optimizer_termination_status = bounded.termination_status
        optimizer_termination_message = bounded.termination_message
        optimizer_optimality = bounded.optimality
        active_bounds = bounded.active_bounds
        parameter_values = np.asarray(
            [float(fitted_model[name].value) for name in fit_parameters],
            dtype=float,
        )
        support_valid = context.support.contains_physical(parameter_values)
    else:
        parameter_covariance = extract_fitter_parameter_covariance(
            fitter,
            fit_parameters,
        )

    chi2 = float(residuals.chi2)
    reduced_chi2 = float(residuals.reduced_chi2)
    weighted_rms_us = float(residuals.rms_weighted().to_value(u.us))
    _update_fit_metadata(fitted_model, certified_toas, residuals, chi2)
    par_path = write_pint_model_parfile_roundtrip_safe(
        model=fitted_model,
        path=candidate_dir / "solution.par",
    )
    tim_path = _write_solution_tim(
        toas=certified_toas,
        path=candidate_dir / "solution.tim",
    )
    residuals_csv, residuals_png = write_residual_plot(
        par_path,
        tim_path,
        candidate_dir,
        clock_override_path=clock_override_path,
        artifact_stem="solution",
    )
    disk_chi2, disk_reduced_chi2, disk_wrms_us = _disk_reload_metrics(
        par_path,
        tim_path,
        clock_override_path=clock_override_path,
    )
    reload_delta = abs(disk_chi2 - chi2)
    accepted = bool(
        support_valid
        and reduced_chi2 < config.strict_reduced_chi2_threshold
        and disk_reduced_chi2 < config.strict_reduced_chi2_threshold
        and reload_delta <= config.disk_reload_chi2_tolerance
    )
    target_reached = bool(
        support_valid
        and reduced_chi2 < config.target_reduced_chi2_threshold
        and disk_reduced_chi2 < config.target_reduced_chi2_threshold
        and reload_delta <= config.disk_reload_chi2_tolerance
    )
    record = {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "candidate_id": candidate_id,
        "source_cell_id": mode.source_cell_id or mode.cell_id,
        "destination_cell_id": mode.cell_id,
        "source_grid_index": mode.source_grid_index,
        "annealed_total_chi2": mode.total_chi2,
        "annealed_shape_chi2": mode.shape_chi2,
        "annealed_phase_chi2": mode.phase_chi2,
        "integer_offsets": mode.integer_offsets,
        "fit_parameters": fit_parameters,
        "prohibited_automatic_parameters": PROHIBITED_AUTOMATIC_PARAMETERS,
        "parameter_expansion": "disabled",
        "optimizer_mode": optimizer_mode,
        "optimizer_fallback_reason": optimizer_fallback_reason,
        "optimizer_function_evaluations": optimizer_function_evaluations,
        "optimizer_termination_status": optimizer_termination_status,
        "optimizer_termination_message": optimizer_termination_message,
        "optimizer_optimality": optimizer_optimality,
        "support_valid": support_valid,
        "active_bounds": active_bounds,
        "support_boundary_hit": bool(active_bounds),
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "weighted_rms_us": weighted_rms_us,
        "disk_reload_chi2": disk_chi2,
        "disk_reload_reduced_chi2": disk_reduced_chi2,
        "disk_reload_weighted_rms_us": disk_wrms_us,
        "disk_reload_chi2_delta": reload_delta,
        "accepted": accepted,
        "target_reached": target_reached,
        "fitted_parameters": {
            name: float(fitted_model[name].value) for name in fit_parameters
        },
        "covariance_parameter_names": fit_parameters,
        "parameter_covariance": parameter_covariance,
        "artifacts": {
            "par": str(par_path),
            "tim": str(tim_path),
            "residuals_csv": str(residuals_csv),
            "residuals_png": str(residuals_png),
        },
    }
    _atomic_json(candidate_dir / "certification.json", record)
    return record


def _copy_candidate_to_solution(
    *,
    candidate: Mapping[str, Any],
    output: Path,
    incumbent_index: int,
) -> None:
    artifacts = candidate["artifacts"]
    destinations = {
        "par": output / "solution.par",
        "tim": output / "solution.tim",
        "residuals_csv": output / "residuals.csv",
        "residuals_png": output / "residuals.png",
    }
    for name, destination in destinations.items():
        shutil.copyfile(Path(artifacts[name]), destination)
    stem = f"{incumbent_index:03d}_chi2_{float(candidate['chi2']):.9f}"
    incumbent = output / "incumbents"
    incumbent.mkdir(parents=True, exist_ok=True)
    for name, suffix in (
        ("par", ".par"),
        ("tim", ".tim"),
        ("residuals_csv", "_residuals.csv"),
        ("residuals_png", "_residuals.png"),
    ):
        shutil.copyfile(Path(artifacts[name]), incumbent / f"{stem}{suffix}")


def solve_bayesian_ell1_dual_lattice(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: BayesianELL1DualLatticeSolverConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> BayesianELL1DualLatticeSolverResult:
    """Search, anneal, and PINT-certify the complete ELL1 cell portfolio."""

    active = config or BayesianELL1DualLatticeSolverConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    clock_override = (
        Path(clock_override_path).resolve()
        if clock_override_path is not None
        else None
    )
    for path in (par, tim):
        if not path.is_file():
            raise FileNotFoundError(path)
    if clock_override is not None and not clock_override.is_file():
        raise FileNotFoundError(clock_override)
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "par_sha256": _sha256(par),
        "tim_sha256": _sha256(tim),
        "clock_override_sha256": (
            _sha256(clock_override) if clock_override is not None else None
        ),
    }
    run_fingerprint = _json_digest(
        {
            "solver_schema_version": SOLVER_SCHEMA_VERSION,
            "inputs": input_hashes,
            "solver_config": active,
        }
    )
    audit_path = output / "audit.json"
    if resume and audit_path.is_file():
        prior_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        _require_current_schema(prior_audit, label="ELL1 resume audit")
        if prior_audit.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("Resume par, tim, clock, or solver config does not match.")
        completed = _resume_completed_result(
            output=output,
            run_fingerprint=run_fingerprint,
        )
        if completed is not None:
            return completed

    audit: dict[str, Any] = {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "architecture": SOLVER_ARCHITECTURE,
        "run_fingerprint": run_fingerprint,
        "inputs": {
            "par": str(par),
            "tim": str(tim),
            "clock_override": str(clock_override) if clock_override else None,
            **input_hashes,
        },
        "solver_config": active,
        "scientific_contract": {
            "fit_parameters": ELL1_FIT_PARAMETERS,
            "parameter_expansion": "disabled",
            "prohibited_automatic_parameters": PROHIBITED_AUTOMATIC_PARAMETERS,
            "source_class_prior": "forbidden",
            "historical_solution_seed": "forbidden",
            "manual_integer_wrap": "forbidden",
            "orbital_cycle_cell_policy": "all cells must be searched",
            "cell_shape_pruning": "forbidden",
            "global_optimum_claim": (
                "Finite all-cell and posterior-beam search; no exhaustive-global claim."
            ),
        },
    }
    _atomic_json(audit_path, audit)

    context = build_ell1_dual_lattice_search_context(
        par_path=par,
        tim_path=tim,
        maximum_cell_width=active.maximum_cell_width,
        support_width_scale=active.support_width_scale,
        minimum_f1_half_width=active.minimum_f1_half_width,
        minimum_eccentricity_half_width=active.minimum_eccentricity_half_width,
        astrometric_bound_arcsec=active.astrometric_bound_arcsec,
        clock_override_path=clock_override,
    )
    if (
        len(context.evaluator.parameter_names) != len(ELL1_FIT_PARAMETERS)
        or set(context.evaluator.parameter_names) != set(ELL1_FIT_PARAMETERS)
    ):
        raise RuntimeError("Input fit flags changed while building ELL1 context.")
    audit.update(
        {
            "toa_count": len(context.evaluator.toas.table),
            "session_count": len(context.evaluator.sessions),
            "observation_span_days": context.coordinates.observation_span_days,
            "cell_count": len(context.cells),
            "physical_support_lower": context.support.physical_lower,
            "physical_support_upper": context.support.physical_upper,
        }
    )
    _atomic_json(audit_path, audit)

    cell_dir = output / "checkpoints" / "cells"
    all_conditional: list[ELL1ConditionalMode] = []
    cell_summaries: list[dict[str, Any]] = []
    for cell in context.cells:
        checkpoint = cell_dir / f"{cell.cell_id}.json"
        if resume and checkpoint.is_file():
            modes = _load_checkpoint_items(
                checkpoint,
                run_fingerprint=run_fingerprint,
                item_name="modes",
                loader=_conditional_mode_from_payload,
            )
            resumed_cell = True
        else:
            modes = search_ell1_orbital_cycle_cell(
                context=context,
                cell=cell,
                config=active.cell_search,
            )
            _atomic_json(
                checkpoint,
                {
                    "schema_version": SOLVER_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "cell_id": cell.cell_id,
                    "cell_lower": cell.lower,
                    "cell_upper": cell.upper,
                    "modes": modes,
                },
            )
            resumed_cell = False
        all_conditional.extend(modes)
        cell_summaries.append(
            {
                "cell_id": cell.cell_id,
                "mode_count": len(modes),
                "resumed": resumed_cell,
                "best_strict_score": min(
                    (mode.strict_score for mode in modes), default=None
                ),
                "lattice_mass_converged": all(
                    mode.lattice_mass_converged for mode in modes
                ),
            }
        )

    selected_conditional = _select_conditional_modes(all_conditional, active)
    audit["search"] = {
        "cells": cell_summaries,
        "selected_conditional_modes": selected_conditional,
        "annealing": [],
        "selected_annealed_modes": [],
        "candidate_certifications": [],
    }
    if not selected_conditional:
        audit["status"] = "failed"
        audit["failure"] = {
            "stage": "conditional_selection",
            "conditional_mode_count": len(all_conditional),
            "in_support_mode_count": sum(
                mode.posterior_in_support for mode in all_conditional
            ),
        }
        _atomic_json(audit_path, audit)
        raise RuntimeError(
            "No in-support ELL1 posterior mode survived all-cell search. "
            f"Diagnostics: {audit_path}"
        )
    _atomic_json(audit_path, audit)
    anneal_dir = output / "checkpoints" / "annealing"
    all_annealed: list[ELL1AnnealedMode] = []
    anneal_summaries: list[dict[str, Any]] = []
    for rank, mode in enumerate(selected_conditional):
        mode_id = (
            f"{rank:04d}_{mode.cell_id}_"
            f"{mode.grid_index[0]}_{mode.grid_index[1]}"
        )
        checkpoint = anneal_dir / f"{mode_id}.json"
        if resume and checkpoint.is_file():
            annealed = _load_checkpoint_items(
                checkpoint,
                run_fingerprint=run_fingerprint,
                item_name="modes",
                loader=_annealed_mode_from_payload,
            )
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            diagnostics_payload = checkpoint_payload.get(
                "diagnostics",
                {
                    "completed_schedule": bool(annealed),
                    "failure_reason": "checkpoint predates structured diagnostics",
                    "stages": [],
                },
            )
            resumed_mode = True
        else:
            annealing_result = anneal_ell1_conditional_mode_diagnosed(
                context=context,
                mode=mode,
                config=active.annealing,
            )
            annealed = annealing_result.modes
            diagnostics_payload = to_json_ready(annealing_result)
            diagnostics_payload.pop("modes", None)
            _atomic_json(
                checkpoint,
                {
                    "schema_version": SOLVER_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "mode_id": mode_id,
                    "source_mode": mode,
                    "modes": annealed,
                    "diagnostics": diagnostics_payload,
                },
            )
            resumed_mode = False
        all_annealed.extend(annealed)
        anneal_summaries.append(
            {
                "mode_id": mode_id,
                "source_strict_score": mode.strict_score,
                "output_mode_count": len(annealed),
                "best_total_chi2": min(
                    (item.total_chi2 for item in annealed), default=None
                ),
                "diagnostics": diagnostics_payload,
                "resumed": resumed_mode,
            }
        )

    selected_annealed = _select_annealed_modes(all_annealed, active)
    audit["search"]["annealing"] = anneal_summaries
    audit["search"]["selected_annealed_modes"] = selected_annealed
    if not selected_annealed:
        audit["status"] = "failed"
        audit["failure"] = {
            "stage": "annealing",
            "selected_conditional_mode_count": len(selected_conditional),
            "annealed_mode_count": len(all_annealed),
            "finite_annealed_mode_count": sum(
                np.isfinite(mode.total_chi2) for mode in all_annealed
            ),
        }
        _atomic_json(audit_path, audit)
        failure_summary = _format_annealing_failures(anneal_summaries)
        raise RuntimeError(
            "ELL1 posterior annealing produced no certifiable mode. "
            f"{failure_summary} Diagnostics: {audit_path}"
        )
    candidate_records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    incumbent_count = 0
    for rank, mode in enumerate(selected_annealed):
        candidate_id = f"candidate_{rank:04d}"
        candidate_dir = output / "candidates" / candidate_id
        candidate_checkpoint = candidate_dir / "certification.json"
        if resume and candidate_checkpoint.is_file():
            record = json.loads(candidate_checkpoint.read_text(encoding="utf-8"))
            _require_current_schema(
                record,
                label=f"ELL1 candidate certification {candidate_id}",
            )
            if record.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError(
                    f"Candidate certification provenance mismatch: {candidate_id}"
                )
        else:
            try:
                record = _certify_mode(
                    context=context,
                    seed_par_path=par,
                    mode=mode,
                    candidate_id=candidate_id,
                    candidate_dir=candidate_dir,
                    config=active,
                    clock_override_path=clock_override,
                    run_fingerprint=run_fingerprint,
                )
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                record = {
                    "schema_version": SOLVER_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "candidate_id": candidate_id,
                    "source_cell_id": mode.source_cell_id or mode.cell_id,
                    "destination_cell_id": mode.cell_id,
                    "annealed_total_chi2": mode.total_chi2,
                    "error": f"{type(error).__name__}: {error}",
                    "accepted": False,
                    "target_reached": False,
                }
                _atomic_json(candidate_checkpoint, record)
        candidate_records.append(record)
        if "chi2" not in record:
            continue
        record_rank = _candidate_certification_rank(record)
        best_rank = (
            _candidate_certification_rank(best_record)
            if best_record is not None
            else None
        )
        if best_rank is None or record_rank < best_rank:
            best_record = record
            _copy_candidate_to_solution(
                candidate=record,
                output=output,
                incumbent_index=incumbent_count,
            )
            incumbent_count += 1
    if best_record is None:
        raise RuntimeError("Every annealed ELL1 mode failed PINT certification.")

    best_artifacts = {
        "par": str(output / "solution.par"),
        "tim": str(output / "solution.tim"),
        "residuals_csv": str(output / "residuals.csv"),
        "residuals_png": str(output / "residuals.png"),
    }
    final_certification = {
        **{key: value for key, value in best_record.items() if key != "artifacts"},
        "architecture": SOLVER_ARCHITECTURE,
        "portfolio_counts": {
            "searched_cells": len(context.cells),
            "conditional_modes": len(all_conditional),
            "selected_conditional_modes": len(selected_conditional),
            "annealed_modes": len(all_annealed),
            "selected_annealed_modes": len(selected_annealed),
            "certified_modes": sum("chi2" in record for record in candidate_records),
        },
        "thresholds": {
            "strict_reduced_chi2": active.strict_reduced_chi2_threshold,
            "target_reduced_chi2": active.target_reduced_chi2_threshold,
            "disk_reload_chi2_tolerance": active.disk_reload_chi2_tolerance,
        },
        "artifacts": best_artifacts,
    }
    _atomic_json(output / "certification.json", final_certification)
    _atomic_json(output / "candidate_portfolio.json", candidate_records)
    audit["status"] = "completed"
    audit["search"]["candidate_certifications"] = candidate_records
    _atomic_json(audit_path, audit)
    manifest_paths = (
        output / "solution.par",
        output / "solution.tim",
        output / "residuals.csv",
        output / "residuals.png",
        output / "audit.json",
        output / "certification.json",
        output / "candidate_portfolio.json",
    )
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": SOLVER_ARCHITECTURE,
            "run_fingerprint": run_fingerprint,
            "artifacts": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in manifest_paths
            },
        },
    )
    return _result_from_certification(output, final_certification, resumed=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run APOLLO all-cell Bayesian ELL1 dual-lattice search, posterior "
            "annealing, and PINT certification."
        )
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orbital-samples-per-cell", type=int, default=2)
    parser.add_argument("--phase-samples", type=int, default=12)
    parser.add_argument("--modes-per-cell", type=int, default=4)
    parser.add_argument("--chart-evaluations", type=int, default=60)
    parser.add_argument("--maximum-modes-to-anneal", type=int, default=8)
    parser.add_argument("--maximum-modes-to-certify", type=int, default=8)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--strict-reduced-chi2", type=float, default=20.0)
    parser.add_argument("--target-reduced-chi2", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BayesianELL1DualLatticeSolverConfig(
        cell_search=ELL1CellSearchConfig(
            orbital_samples_per_cell=args.orbital_samples_per_cell,
            phase_samples=args.phase_samples,
            modes_per_cell=args.modes_per_cell,
            fast_shape_maximum_evaluations=args.chart_evaluations,
        ),
        maximum_modes_to_anneal=args.maximum_modes_to_anneal,
        maximum_modes_to_certify=args.maximum_modes_to_certify,
        fit_iterations=args.fit_iterations,
        strict_reduced_chi2_threshold=args.strict_reduced_chi2,
        target_reduced_chi2_threshold=args.target_reduced_chi2,
    )
    result = solve_bayesian_ell1_dual_lattice(
        par_path=args.par,
        tim_path=args.tim,
        output_dir=args.output,
        config=config,
        clock_override_path=args.clock_override,
        resume=args.resume,
    )
    print(json.dumps(to_json_ready(result), indent=2, allow_nan=False))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BayesianELL1DualLatticeSolverConfig",
    "BayesianELL1DualLatticeSolverResult",
    "build_arg_parser",
    "main",
    "solve_bayesian_ell1_dual_lattice",
]
