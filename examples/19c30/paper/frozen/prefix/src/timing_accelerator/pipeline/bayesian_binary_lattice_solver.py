from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.residuals import Residuals

from timing_accelerator import __version__
from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
    _update_fit_metadata,
)
from timing_accelerator.inference.active_timing_contracts import to_json_ready
from timing_accelerator.inference.bayesian_binary_lattice import (
    BinaryAnnealedMode,
    BinaryAnnealingConfig,
    BinaryAnnealingStep,
    BinaryCellSearchConfig,
    BinaryConditionalMode,
    PintBinaryLatticeProjector,
    anneal_phase_lattice_mode,
    build_phase_lattice_search_context,
    detect_phase_model_name,
    search_phase_lattice_cell,
)
from timing_accelerator.inference.bayesian_binary_model_adapter import (
    BINARY_MODEL_CAPABILITIES,
    canonical_binary_model_name,
    detect_binary_model_name,
)
from timing_accelerator.inference.bayesian_binary_model_selection import (
    BinaryModelSelectionConfig,
    PhysicalModelSelectionConfig,
    select_binary_model_on_fixed_path,
)
from timing_accelerator.inference.bayesian_partial_ell1_bootstrap import (
    PartialELL1BootstrapConfig,
    bootstrap_partial_ell1,
    is_incomplete_ell1_par,
)
from timing_accelerator.inference.bayesian_coarse_binary_bootstrap import (
    CoarseBinaryBootstrapConfig,
    bootstrap_coarse_binary,
    inspect_coarse_binary_par,
    is_incomplete_binary_par,
)
from timing_accelerator.inference.bayesian_observation_planner import (
    PintObservationPlannerConfig,
    plan_apollo_observations,
)
from timing_accelerator.inference.observation_utility import (
    BayesianObservationDesignConfig,
)
from timing_accelerator.inference.bayesian_ell1_dual_lattice import (
    ELL1AnnealingConfig,
    ELL1CellSearchConfig,
    ELL1_FIT_PARAMETERS,
)
from timing_accelerator.inference.pint_phase_basis import (
    PROHIBITED_AUTOMATIC_PARAMETERS,
)
from timing_accelerator.inference.bayesian_timing_update import (
    TimingUpdatePriorConfig,
    condition_timing_update_context,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    extract_fitter_parameter_covariance,
    fit_pint_model_with_physical_bounds,
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.pipeline.bayesian_ell1_dual_lattice_solver import (
    BayesianELL1DualLatticeSolverConfig,
    _attach_session_pulse_numbers,
    _atomic_json,
    _candidate_certification_rank,
    _copy_candidate_to_solution,
    _disk_reload_metrics,
    _json_digest,
    _sha256,
    _write_solution_tim,
    solve_bayesian_ell1_dual_lattice,
)
from timing_accelerator.reports.residual_plot import write_residual_plot


SOLVER_ARCHITECTURE = "apollo_binary_phase_orbit_lattice_solver"
PARTIAL_ELL1_SOLVER_ARCHITECTURE = (
    "apollo_partial_ell1_bootstrap_plus_bayesian_lattice_solver"
)
COARSE_BINARY_SOLVER_ARCHITECTURE = (
    "apollo_canonical_keplerian_bootstrap_plus_bayesian_lattice_solver"
)
SOLVER_SCHEMA_VERSION = 5


def _par_declares_binary_model(path: Path) -> bool:
    declared = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "C ")):
            continue
        fields = line.split()
        if len(fields) >= 2 and fields[0].upper() == "BINARY2":
            raise ValueError("APOLLO binary search does not support BINARY2 models.")
        if len(fields) >= 2 and fields[0].upper() == "BINARY":
            name = canonical_binary_model_name(fields[1])
            if name not in BINARY_MODEL_CAPABILITIES:
                raise ValueError(f"APOLLO binary search does not support BINARY {name}.")
            declared = True
    return declared


def _require_current_schema(payload: Mapping[str, Any], *, label: str) -> None:
    if payload.get("schema_version") != SOLVER_SCHEMA_VERSION:
        raise RuntimeError(
            f"{label} uses certification schema {payload.get('schema_version')!r}; "
            f"APOLLO requires schema {SOLVER_SCHEMA_VERSION}. Start a fresh output "
            "directory instead of resuming this result."
        )


def _solver_result_rank(result: Any) -> tuple[bool, float]:
    chi2 = float(result.chi2)
    return (not bool(result.accepted), chi2 if np.isfinite(chi2) else np.inf)






def _reconcile_model_selection_portfolio(
    portfolio: Sequence[Mapping[str, Any]],
    *,
    proposed_candidate_id: str,
    selected_candidate_id: str,
    proposed_support_valid: bool,
) -> tuple[dict[str, Any], ...]:
    """Separate the model selector's proposal from the certified selection."""

    reconciled: list[dict[str, Any]] = []
    for source in portfolio:
        record = dict(source)
        candidate_id = str(record.get("candidate_id", ""))
        proposed = candidate_id == proposed_candidate_id
        selected = candidate_id == selected_candidate_id
        record["proposed_selected"] = proposed
        record["selected"] = selected
        if proposed and not proposed_support_valid:
            record["search_support_valid"] = False
            record["selection_outcome"] = "rejected_outside_search_support"
            record["selection_rejection_reason"] = (
                "The fitted shared parameters left the Bayesian search support."
            )
        elif selected:
            record["search_support_valid"] = True
            record["selection_outcome"] = "selected"
        else:
            record["search_support_valid"] = None
            record["selection_outcome"] = "not_selected"
        reconciled.append(record)
    return tuple(reconciled)


@dataclass(frozen=True)
class BayesianBinaryLatticeSolverConfig:
    maximum_cell_width: float = 0.25
    support_width_scale: float = 1.0
    minimum_eccentricity_half_width: float = 5.0e-3
    astrometric_bound_arcsec: float = 300.0
    cell_search: BinaryCellSearchConfig = field(default_factory=BinaryCellSearchConfig)
    annealing: BinaryAnnealingConfig = field(default_factory=BinaryAnnealingConfig)
    maximum_modes_to_anneal: int = 8
    conditional_score_window: float = 2_000.0
    maximum_modes_to_certify: int = 8
    annealed_chi2_window: float = 2_000.0
    fit_iterations: int = 10
    strict_reduced_chi2_threshold: float = 20.0
    target_reduced_chi2_threshold: float = 10.0
    disk_reload_chi2_tolerance: float = 1.0e-2
    use_mature_ell1_engine: bool = True
    plan_observations_on_failure: bool = True
    observation_planner: PintObservationPlannerConfig = field(
        default_factory=PintObservationPlannerConfig
    )
    partial_ell1_bootstrap: PartialELL1BootstrapConfig = field(
        default_factory=PartialELL1BootstrapConfig
    )
    coarse_binary_bootstrap: CoarseBinaryBootstrapConfig = field(
        default_factory=CoarseBinaryBootstrapConfig
    )
    physical_model_selection: BinaryModelSelectionConfig = field(
        default_factory=BinaryModelSelectionConfig
    )
    timing_update_prior: TimingUpdatePriorConfig = field(
        default_factory=TimingUpdatePriorConfig
    )

    def __post_init__(self) -> None:
        positive_floats = (
            self.maximum_cell_width,
            self.support_width_scale,
            self.minimum_eccentricity_half_width,
            self.astrometric_bound_arcsec,
            self.conditional_score_window,
            self.annealed_chi2_window,
            self.strict_reduced_chi2_threshold,
            self.target_reduced_chi2_threshold,
            self.disk_reload_chi2_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("Binary solver scales and thresholds must be positive.")
        positive_integers = (
            self.cell_search.orbital_samples_per_cell,
            self.cell_search.phase_samples,
            self.cell_search.modes_per_cell,
            self.cell_search.chart_maximum_evaluations,
            self.maximum_modes_to_anneal,
            self.maximum_modes_to_certify,
            self.fit_iterations,
        )
        if not all(int(value) > 0 for value in positive_integers):
            raise ValueError("Binary solver budgets must be positive integers.")
        if self.target_reduced_chi2_threshold > self.strict_reduced_chi2_threshold:
            raise ValueError("The target reduced chi-square cannot exceed acceptance.")


@dataclass(frozen=True)
class BayesianBinaryLatticeSolverResult:
    model_name: str
    model_family: str
    engine: str
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
    observation_plan_dir: Path | None = None
    observation_plan_status: str = "not_evaluated"
    scientific_status: str = "not_evaluated"


def _conditional_mode_from_payload(payload: Mapping[str, Any]) -> BinaryConditionalMode:
    return BinaryConditionalMode(
        model_name=str(payload["model_name"]),
        cell_id=str(payload["cell_id"]),
        grid_index=tuple(int(value) for value in payload["grid_index"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        local_integer_offsets=tuple(
            tuple(int(value) for value in component)
            for component in payload["local_integer_offsets"]
        ),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        strict_score=float(payload["strict_score"]),
        log_likelihood=float(payload["log_likelihood"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        posterior_weight=float(payload["posterior_weight"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=float(payload["lattice_log_mass_increment"]),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        posterior_in_support=bool(payload["posterior_in_support"]),
        prior_component_id=str(payload.get("prior_component_id", "local_broad")),
        prior_log_weight=float(payload.get("prior_log_weight", 0.0)),
    )


def _annealing_step_from_payload(payload: Mapping[str, Any]) -> BinaryAnnealingStep:
    return BinaryAnnealingStep(
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


def _annealed_mode_from_payload(payload: Mapping[str, Any]) -> BinaryAnnealedMode:
    return BinaryAnnealedMode(
        model_name=str(payload["model_name"]),
        cell_id=str(payload["cell_id"]),
        source_grid_index=tuple(int(value) for value in payload["source_grid_index"]),
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
        trace=tuple(_annealing_step_from_payload(step) for step in payload["trace"]),
        prior_component_id=str(payload.get("prior_component_id", "local_broad")),
        prior_log_weight=float(payload.get("prior_log_weight", 0.0)),
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


def _select_conditional_modes(
    modes: Sequence[BinaryConditionalMode],
    config: BayesianBinaryLatticeSolverConfig,
) -> tuple[BinaryConditionalMode, ...]:
    ranked = sorted(
        (
            mode
            for mode in modes
            if mode.posterior_in_support
            and np.isfinite(mode.log_likelihood)
            and np.isfinite(mode.total_chi2)
        ),
        key=lambda mode: (-mode.log_likelihood, mode.total_chi2, mode.cell_id),
    )
    if not ranked:
        return tuple()
    log_likelihood_floor = (
        ranked[0].log_likelihood - config.conditional_score_window
    )
    selected: list[BinaryConditionalMode] = []
    seen_components: set[str] = set()
    for mode in ranked:
        if mode.prior_component_id not in seen_components:
            selected.append(mode)
            seen_components.add(mode.prior_component_id)
            if len(selected) >= config.maximum_modes_to_anneal:
                return tuple(selected)
    selected_ids = {id(mode) for mode in selected}
    for mode in ranked:
        if mode.log_likelihood < log_likelihood_floor:
            continue
        if id(mode) not in selected_ids:
            selected.append(mode)
        if len(selected) >= config.maximum_modes_to_anneal:
            break
    return tuple(selected)


def _select_annealed_modes(
    modes: Sequence[BinaryAnnealedMode],
    config: BayesianBinaryLatticeSolverConfig,
) -> tuple[BinaryAnnealedMode, ...]:
    ranked = sorted(
        (mode for mode in modes if np.isfinite(mode.total_chi2)),
        key=lambda mode: (mode.total_chi2, mode.cell_id, mode.integer_offsets),
    )
    if not ranked:
        return tuple()
    ceiling = ranked[0].total_chi2 + config.annealed_chi2_window
    selected: list[BinaryAnnealedMode] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    seen_components: set[str] = set()
    for mode in ranked:
        if mode.prior_component_id in seen_components:
            continue
        selected.append(mode)
        seen_components.add(mode.prior_component_id)
        seen.add((mode.prior_component_id, mode.integer_offsets))
        if len(selected) >= config.maximum_modes_to_certify:
            return tuple(selected)
    for mode in ranked:
        key = (mode.prior_component_id, mode.integer_offsets)
        if mode.total_chi2 > ceiling or key in seen:
            continue
        seen.add(key)
        selected.append(mode)
        if len(selected) >= config.maximum_modes_to_certify:
            break
    return tuple(selected)


def _result_from_certification(
    output: Path,
    certification: Mapping[str, Any],
    *,
    resumed: bool,
) -> BayesianBinaryLatticeSolverResult:
    artifacts = certification["artifacts"]
    counts = certification["portfolio_counts"]
    return BayesianBinaryLatticeSolverResult(
        model_name=str(certification["model_name"]),
        model_family=str(certification["model_family"]),
        engine=str(certification["engine"]),
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
        scientific_status=str(
            certification.get("solution_claim", {}).get(
                "status", "not_evaluated"
            )
        ),
    )


def _resume_completed_result(
    *,
    output: Path,
    run_fingerprint: str,
) -> BayesianBinaryLatticeSolverResult | None:
    certification_path = output / "certification.json"
    manifest_path = output / "artifact_manifest.json"
    if not certification_path.is_file() or not manifest_path.is_file():
        return None
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    _require_current_schema(certification, label="Completed binary certification")
    if certification.get("run_fingerprint") != run_fingerprint:
        raise RuntimeError("Completed binary result does not match resume inputs.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_current_schema(manifest, label="Completed binary artifact manifest")
    for name, record in manifest.get("artifacts", {}).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
            raise RuntimeError(f"Resume artifact failed SHA-256: {name}")
    return _result_from_certification(output, certification, resumed=True)


def _resume_completed_partial_ell1_result(
    *,
    par: Path,
    tim: Path,
    output: Path,
    config: BayesianBinaryLatticeSolverConfig,
    clock_override: Path | None,
) -> BayesianBinaryLatticeSolverResult | None:
    audit_path = output / "audit.json"
    certification_path = output / "certification.json"
    manifest_path = output / "artifact_manifest.json"
    if not all(path.is_file() for path in (audit_path, certification_path, manifest_path)):
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("architecture") != PARTIAL_ELL1_SOLVER_ARCHITECTURE:
        return None
    _require_current_schema(audit, label="Completed partial ELL1 audit")
    inputs = audit.get("inputs", {})
    if inputs.get("par_sha256") != _sha256(par) or inputs.get("tim_sha256") != _sha256(tim):
        raise RuntimeError("Completed partial ELL1 result does not match resume inputs.")
    expected_clock_hash = _sha256(clock_override) if clock_override is not None else None
    if inputs.get("clock_override_sha256") not in (None, expected_clock_hash):
        raise RuntimeError("Completed partial ELL1 result uses a different clock override.")
    bootstrap = audit.get("bootstrap", {})
    recommended_scale = float(bootstrap.get("recommended_support_width_scale", 1.0))
    expected_config = replace(
        config,
        support_width_scale=min(config.support_width_scale, recommended_scale),
    )
    if to_json_ready(expected_config) != audit.get("effective_solver_config"):
        raise RuntimeError("Completed partial ELL1 result uses a different solver config.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_current_schema(manifest, label="Completed partial ELL1 artifact manifest")
    for name, record in manifest.get("artifacts", {}).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
            raise RuntimeError(f"Resume artifact failed SHA-256: {name}")
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    _require_current_schema(
        certification,
        label="Completed partial ELL1 certification",
    )
    counts = certification["portfolio_counts"]
    return BayesianBinaryLatticeSolverResult(
        model_name="ELL1",
        model_family="ell1",
        engine="partial_ell1_bootstrap+mature_ell1_phase_adapted",
        output_dir=output,
        solution_par=output / "solution.par",
        solution_tim=output / "solution.tim",
        residuals_csv=output / "residuals.csv",
        residuals_png=output / "residuals.png",
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
        resumed=True,
    )


def _validate_annealed_mode_contract(
    *,
    context: Any,
    mode: BinaryAnnealedMode,
) -> Any:
    """Replay an APOLLO candidate before PINT is allowed to refine it."""

    exact = PintBinaryLatticeProjector(context).evaluate_fixed_path(
        mode.normalized_values,
        local_integer_offsets=mode.local_integer_offsets,
        integer_offsets=mode.integer_offsets,
    )
    mismatches: list[str] = []
    if not np.allclose(
        exact.parameter_values,
        mode.parameter_values,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        maximum_delta = float(
            np.max(np.abs(exact.parameter_values - mode.parameter_values))
        )
        mismatches.append(f"parameter_max_abs_delta={maximum_delta:.17g}")
    for name, recorded, replayed in (
        ("shape_chi2", mode.shape_chi2, exact.shape_chi2),
        ("phase_chi2", mode.phase_chi2, exact.phase_chi2),
        ("total_chi2", mode.total_chi2, exact.total_chi2),
    ):
        if not np.isclose(
            float(recorded),
            float(replayed),
            rtol=1.0e-12,
            atol=1.0e-6,
        ):
            mismatches.append(
                f"{name}:recorded={float(recorded):.17g},"
                f"replayed={float(replayed):.17g},"
                f"delta={float(replayed - recorded):.17g}"
            )
    if mismatches:
        raise RuntimeError(
            "APOLLO candidate state is internally inconsistent before PINT "
            "certification; continuous parameters, local pulse component, "
            "integer path, and exact chi-square must describe one state. "
            + "; ".join(mismatches)
        )
    return exact


def _certify_mode(
    *,
    context: Any,
    seed_par_path: Path,
    mode: BinaryAnnealedMode,
    candidate_id: str,
    candidate_dir: Path,
    config: BayesianBinaryLatticeSolverConfig,
    clock_override_path: Path | None,
    run_fingerprint: str,
    timing_update_active: bool = False,
) -> dict[str, Any]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    exact_candidate = _validate_annealed_mode_contract(context=context, mode=mode)
    input_fit_parameters = tuple(context.evaluator.parameter_names)

    def initialize_mode_model() -> Any:
        initialized = _load_pint_model(seed_par_path)
        for name, value in zip(
            input_fit_parameters,
            mode.parameter_values,
            strict=True,
        ):
            _set_numeric_model_parameter_value(initialized, name, float(value))
        for name in initialized.params:
            parameter = initialized[name]
            if hasattr(parameter, "frozen"):
                parameter.frozen = name not in input_fit_parameters
        if (
            len(initialized.free_params) != len(input_fit_parameters)
            or set(initialized.free_params) != set(input_fit_parameters)
        ):
            raise RuntimeError("PINT certification changed the input fit-flag contract.")
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
        baseline_fitter = WLSFitter(
            certified_toas,
            model,
            track_mode="use_pulse_numbers",
        )
        baseline_fitter.fit_toas(maxiter=config.fit_iterations)
        baseline_model = baseline_fitter.model
        baseline_residuals = Residuals(
            certified_toas,
            baseline_model,
            subtract_mean=True,
            use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        baseline_values = np.asarray(
            [float(baseline_model[name].value) for name in input_fit_parameters],
            dtype=float,
        )
        support_valid = context.support.contains_physical(baseline_values)
        if not support_valid:
            optimizer_fallback_reason = "unconstrained WLS left Bayesian support"
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        baseline_fitter = None
        baseline_model = None
        baseline_residuals = None
        baseline_values = np.asarray(mode.parameter_values, dtype=float)
        support_valid = False
        optimizer_fallback_reason = f"{type(error).__name__}: {error}"

    if not support_valid:
        bounded = fit_pint_model_with_physical_bounds(
            model=initialize_mode_model(),
            toas=certified_toas,
            parameter_names=input_fit_parameters,
            lower_bounds=context.support.physical_lower,
            upper_bounds=context.support.physical_upper,
            initial_values=mode.parameter_values,
            maximum_evaluations=max(100, 50 * int(config.fit_iterations)),
        )
        fitted_model = bounded.model
        residuals = bounded.residuals
        fit_parameters = input_fit_parameters
        parameter_covariance = bounded.covariance
        optimizer_mode = "bounded_pint_least_squares"
        optimizer_function_evaluations = bounded.function_evaluations
        optimizer_termination_status = bounded.termination_status
        optimizer_termination_message = bounded.termination_message
        optimizer_optimality = bounded.optimality
        active_bounds = bounded.active_bounds
        fitted_values = np.asarray(
            [float(fitted_model[name].value) for name in input_fit_parameters],
            dtype=float,
        )
        support_valid = context.support.contains_physical(fitted_values)
        selected_model_name = detect_binary_model_name(fitted_model)
        selected_model_family = BINARY_MODEL_CAPABILITIES[selected_model_name].family
        comparison_architecture = "apollo_fixed_path_binary_model_portfolio"
        comparison_status = "skipped_after_bounded_support_recovery"
        model_comparison = {
            "architecture": comparison_architecture,
            "status": comparison_status,
            "selected": False,
            "selected_candidate_id": "baseline",
            "selected_model_name": selected_model_name,
            "selected_model_family": selected_model_family,
            "proposed_candidate_id": None,
            "proposed_model_name": None,
            "proposed_model_family": None,
            "proposed_support_valid": None,
            "hypothesis_weights": {"keplerian_geometry": 1.0},
            "proposed_hypothesis_weights": {"keplerian_geometry": 1.0},
            "portfolio": (),
        }
    else:
        model_selection = select_binary_model_on_fixed_path(
            baseline_fitter=baseline_fitter,
            certified_toas=certified_toas,
            baseline_fit_parameters=input_fit_parameters,
            observation_span_days=float(context.adapter.observation_span_days),
            fit_iterations=config.fit_iterations,
            config=config.physical_model_selection,
        )
        proposed_model = model_selection.fitter.model
        proposed_values = baseline_values.copy()
        for index, name in enumerate(input_fit_parameters):
            if name in proposed_model.params and proposed_model[name].value is not None:
                proposed_values[index] = float(proposed_model[name].value)
        proposed_support_valid = context.support.contains_physical(proposed_values)
        proposed_candidate_id = model_selection.selected_candidate_id
        proposed_model_name = model_selection.selected_model_name
        proposed_model_family = model_selection.selected_model_family
        if proposed_support_valid:
            fitted_model = proposed_model
            residuals = model_selection.residuals
            fit_parameters = model_selection.fit_parameters
            selected_model_name = model_selection.selected_model_name
            selected_model_family = model_selection.selected_model_family
            selected_candidate_id = model_selection.selected_candidate_id
            selected_extension = model_selection.selected
            parameter_covariance = extract_fitter_parameter_covariance(
                model_selection.fitter,
                fit_parameters,
            )
        else:
            fitted_model = baseline_model
            residuals = baseline_residuals
            fit_parameters = input_fit_parameters
            selected_model_name = detect_binary_model_name(fitted_model)
            selected_model_family = BINARY_MODEL_CAPABILITIES[selected_model_name].family
            selected_candidate_id = "baseline"
            selected_extension = False
            parameter_covariance = extract_fitter_parameter_covariance(
                baseline_fitter,
                fit_parameters,
            )
        support_valid = True
        reconciled_portfolio = _reconcile_model_selection_portfolio(
            model_selection.portfolio,
            proposed_candidate_id=proposed_candidate_id,
            selected_candidate_id=selected_candidate_id,
            proposed_support_valid=proposed_support_valid,
        )
        model_comparison = {
            "architecture": "apollo_fixed_path_binary_model_portfolio",
            "status": (
                "evaluated"
                if proposed_support_valid
                else "selected_hypothesis_rejected_outside_search_support"
            ),
            "selected": selected_extension,
            "selected_candidate_id": selected_candidate_id,
            "selected_model_name": selected_model_name,
            "selected_model_family": selected_model_family,
            "proposed_candidate_id": proposed_candidate_id,
            "proposed_model_name": proposed_model_name,
            "proposed_model_family": proposed_model_family,
            "proposed_support_valid": proposed_support_valid,
            "hypothesis_weights": model_selection.hypothesis_weights,
            "proposed_hypothesis_weights": model_selection.hypothesis_weights,
            "portfolio": reconciled_portfolio,
        }

    chi2 = float(residuals.chi2)
    reduced_chi2 = float(residuals.reduced_chi2)
    weighted_rms_us = float(residuals.rms_weighted().to_value(u.us))
    model_selection_complete = bool(
        model_comparison.get("model_selection_complete", True)
    )
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
        and model_selection_complete
        and reduced_chi2 < config.strict_reduced_chi2_threshold
        and disk_reduced_chi2 < config.strict_reduced_chi2_threshold
        and reload_delta <= config.disk_reload_chi2_tolerance
    )
    target_reached = bool(
        support_valid
        and model_selection_complete
        and reduced_chi2 < config.target_reduced_chi2_threshold
        and disk_reduced_chi2 < config.target_reduced_chi2_threshold
        and reload_delta <= config.disk_reload_chi2_tolerance
    )
    record = {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "candidate_id": candidate_id,
        "model_name": selected_model_name,
        "model_family": selected_model_family,
        "source_cell_id": mode.cell_id,
        "source_grid_index": mode.source_grid_index,
        "prior_component_id": mode.prior_component_id,
        "prior_log_weight": mode.prior_log_weight,
        "annealed_total_chi2": mode.total_chi2,
        "annealed_shape_chi2": mode.shape_chi2,
        "annealed_phase_chi2": mode.phase_chi2,
        "candidate_contract_replay": {
            "shape_chi2": exact_candidate.shape_chi2,
            "phase_chi2": exact_candidate.phase_chi2,
            "total_chi2": exact_candidate.total_chi2,
            "consistent": True,
        },
        "integer_offsets": mode.integer_offsets,
        "fit_parameters": fit_parameters,
        "coordinate_names": context.adapter.coordinate_names,
        "prohibited_automatic_parameters": PROHIBITED_AUTOMATIC_PARAMETERS,
        "parameter_expansion": (
            "evidence_selected_high_order_extension"
            if any(
                name.startswith("F")
                and name[1:].isdigit()
                and int(name[1:]) >= 2
                for name in fit_parameters
            )
            or any(name.startswith("GL") for name in fit_parameters)
            else "disabled"
        ),
        "binary_model_selection": model_comparison,
        "basic_portfolio_complete": model_comparison.get(
            "basic_portfolio_complete", True
        ),
        "high_order_portfolio_complete": model_comparison.get(
            "high_order_portfolio_complete", True
        ),
        "model_selection_complete": model_selection_complete,
        "high_order_portfolio_status": model_comparison.get(
            "high_order_portfolio_status", "not_applicable"
        ),
        "physical_model_comparison": model_comparison,
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
        "reported_error_metric_scope": (
            "fixed_pulse_number_pint_reported_toa_errors"
        ),
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




def _convert_ell1_result(result: Any) -> BayesianBinaryLatticeSolverResult:
    return BayesianBinaryLatticeSolverResult(
        model_name="ELL1",
        model_family="ell1",
        engine="mature_ell1_phase_adapted",
        output_dir=result.output_dir,
        solution_par=result.solution_par,
        solution_tim=result.solution_tim,
        residuals_csv=result.residuals_csv,
        residuals_png=result.residuals_png,
        fit_parameters=result.fit_parameters,
        chi2=result.chi2,
        disk_reload_chi2=result.disk_reload_chi2,
        reduced_chi2=result.reduced_chi2,
        weighted_rms_us=result.weighted_rms_us,
        accepted=result.accepted,
        target_reached=result.target_reached,
        searched_cells=result.searched_cells,
        annealed_modes=result.annealed_modes,
        certified_modes=result.certified_modes,
        resumed=result.resumed,
    )


def _ell1_config(config: BayesianBinaryLatticeSolverConfig) -> BayesianELL1DualLatticeSolverConfig:
    return BayesianELL1DualLatticeSolverConfig(
        maximum_cell_width=config.maximum_cell_width,
        support_width_scale=config.support_width_scale,
        minimum_eccentricity_half_width=config.minimum_eccentricity_half_width,
        astrometric_bound_arcsec=config.astrometric_bound_arcsec,
        cell_search=ELL1CellSearchConfig(
            orbital_samples_per_cell=config.cell_search.orbital_samples_per_cell,
            phase_samples=config.cell_search.phase_samples,
            modes_per_cell=config.cell_search.modes_per_cell,
            fast_shape_maximum_evaluations=config.cell_search.chart_maximum_evaluations,
            finite_difference_step=config.cell_search.finite_difference_step,
            fast_prior_sigma=config.cell_search.coordinate_prior_sigma,
            local_integer_radius=config.cell_search.local_integer_radius,
            lattice=config.cell_search.lattice,
        ),
        annealing=ELL1AnnealingConfig(
            phase_sigma_schedule=config.annealing.phase_sigma_schedule,
            coordinate_prior_sigma=config.annealing.coordinate_prior_sigma,
            paths_per_state=config.annealing.paths_per_state,
            beam_size=config.annealing.beam_size,
            finite_difference_step=config.annealing.finite_difference_step,
            maximum_closure_evaluations=config.annealing.maximum_closure_evaluations,
            lattice=config.annealing.lattice,
        ),
        maximum_modes_to_anneal=config.maximum_modes_to_anneal,
        conditional_score_window=config.conditional_score_window,
        maximum_modes_to_certify=config.maximum_modes_to_certify,
        annealed_chi2_window=config.annealed_chi2_window,
        fit_iterations=config.fit_iterations,
        strict_reduced_chi2_threshold=config.strict_reduced_chi2_threshold,
        target_reduced_chi2_threshold=config.target_reduced_chi2_threshold,
        disk_reload_chi2_tolerance=config.disk_reload_chi2_tolerance,
    )


def _register_observation_plan_artifacts(
    *,
    output: Path,
    paths: Sequence[Path],
) -> None:
    manifest_path = Path(output) / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": SOLVER_ARCHITECTURE,
            "artifacts": {},
        }
    artifacts = dict(manifest.get("artifacts") or {})
    for path in paths:
        artifact = Path(path)
        if artifact.is_file():
            artifacts[f"observation_plan/{artifact.name}"] = {
                "path": str(artifact),
                "sha256": _sha256(artifact),
            }
    manifest["artifacts"] = artifacts
    _atomic_json(manifest_path, manifest)


def _finalize_observation_plan(
    *,
    result: BayesianBinaryLatticeSolverResult,
    config: BayesianBinaryLatticeSolverConfig,
    clock_override_path: Path | None,
) -> BayesianBinaryLatticeSolverResult:
    if result.accepted:
        return replace(
            result,
            observation_plan_status="not_required_qualified_solution",
        )
    if not config.plan_observations_on_failure:
        return replace(result, observation_plan_status="disabled")
    try:
        plan = plan_apollo_observations(
            run_output_dir=result.output_dir,
            output_dir=Path(result.output_dir) / "observation_plan",
            config=config.observation_planner,
            clock_override_path=clock_override_path,
        )
    except (ArithmeticError, OSError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        diagnostic_path = Path(result.output_dir) / "observation_plan_error.json"
        _atomic_json(
            diagnostic_path,
            {
                "schema_version": SOLVER_SCHEMA_VERSION,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "scientific_effect": (
                    "The timing-search products remain valid, but APOLLO could not construct "
                    "a posterior-predictive observing recommendation from this portfolio."
                ),
            },
        )
        _register_observation_plan_artifacts(
            output=result.output_dir,
            paths=(diagnostic_path,),
        )
        return replace(
            result,
            observation_plan_status=f"failed: {type(error).__name__}: {error}",
        )
    _register_observation_plan_artifacts(
        output=result.output_dir,
        paths=tuple(plan.artifacts.values()),
    )
    return replace(
        result,
        observation_plan_dir=plan.output_dir,
        observation_plan_status=f"generated:{plan.diagnosis.status}",
    )


def _solve_partial_ell1_portfolio(
    *,
    par: Path,
    tim: Path,
    output: Path,
    config: BayesianBinaryLatticeSolverConfig,
    clock_override: Path | None,
    resume: bool,
) -> BayesianBinaryLatticeSolverResult:
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        completed = _resume_completed_partial_ell1_result(
            par=par,
            tim=tim,
            output=output,
            config=config,
            clock_override=clock_override,
        )
        if completed is not None:
            return completed
    bootstrap = bootstrap_partial_ell1(
        par_path=par,
        tim_path=tim,
        output_dir=output / "partial_ell1_bootstrap",
        config=config.partial_ell1_bootstrap,
        clock_override_path=clock_override,
        resume=resume,
    )
    child_config = replace(
        config,
        support_width_scale=min(
            config.support_width_scale,
            bootstrap.recommended_support_width_scale,
        ),
        plan_observations_on_failure=False,
    )
    child_results: list[tuple[object, BayesianBinaryLatticeSolverResult]] = []
    failures: list[dict[str, str]] = []
    for mode in bootstrap.modes:
        child_output = output / "bootstrap_mode_solutions" / mode.mode_id
        try:
            result = solve_bayesian_binary_lattice(
                par_path=mode.par_path,
                tim_path=bootstrap.normalized_tim_path,
                output_dir=child_output,
                config=child_config,
                clock_override_path=clock_override,
                resume=resume,
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "mode_id": mode.mode_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        child_results.append((mode, result))
    if not child_results:
        failure_audit = {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": PARTIAL_ELL1_SOLVER_ARCHITECTURE,
            "status": "failed",
            "failure_stage": "complete_mode_portfolio",
            "inputs": {
                "par": str(par),
                "tim": str(tim),
                "par_sha256": _sha256(par),
                "tim_sha256": _sha256(tim),
                "clock_override_sha256": (
                    _sha256(clock_override) if clock_override is not None else None
                ),
            },
            "bootstrap": bootstrap,
            "effective_solver_config": child_config,
            "failed_modes": failures,
        }
        _atomic_json(output / "audit.json", failure_audit)
        failure_summary = " | ".join(
            f"{failure['mode_id']}: {failure['error']}" for failure in failures[:3]
        )
        raise RuntimeError(
            "Every partial ELL1 bootstrap mode failed the complete APOLLO search. "
            f"{failure_summary} Diagnostics: {output / 'audit.json'}"
        )
    mode, best = min(child_results, key=lambda item: _solver_result_rank(item[1]))

    artifact_sources = {
        "solution.par": best.solution_par,
        "solution.tim": best.solution_tim,
        "residuals.csv": best.residuals_csv,
        "residuals.png": best.residuals_png,
    }
    for name, source in artifact_sources.items():
        shutil.copyfile(Path(source), output / name)

    child_certification_path = Path(best.output_dir) / "certification.json"
    child_certification = json.loads(child_certification_path.read_text(encoding="utf-8"))
    artifacts = {
        "par": str(output / "solution.par"),
        "tim": str(output / "solution.tim"),
        "residuals_csv": str(output / "residuals.csv"),
        "residuals_png": str(output / "residuals.png"),
    }
    certification = {
        **{key: value for key, value in child_certification.items() if key != "artifacts"},
        "architecture": PARTIAL_ELL1_SOLVER_ARCHITECTURE,
        "engine": "partial_ell1_bootstrap+mature_ell1_phase_adapted",
        "partial_ell1_bootstrap": {
            "audit": str(bootstrap.audit_path),
            "normalized_tim": str(bootstrap.normalized_tim_path),
            "selected_mode_id": mode.mode_id,
            "selected_mode_par": str(mode.par_path),
            "informative_session_id": bootstrap.informative_session_id,
            "informative_session_toas": bootstrap.informative_session_toas,
            "mode_count": len(bootstrap.modes),
            "recommended_support_width_scale": (
                bootstrap.recommended_support_width_scale
            ),
            "effective_support_width_scale": child_config.support_width_scale,
            "failed_modes": failures,
        },
        "artifacts": artifacts,
    }
    _atomic_json(output / "certification.json", certification)
    _atomic_json(
        output / "candidate_portfolio.json",
        [
            {
                "mode_id": candidate_mode.mode_id,
                "seed_par": str(candidate_mode.par_path),
                "informative_session_chi2": candidate_mode.informative_session_chi2,
                "all_session_shape_chi2": candidate_mode.all_session_shape_chi2,
                "search_output": str(candidate_result.output_dir),
                "chi2": candidate_result.chi2,
                "reduced_chi2": candidate_result.reduced_chi2,
                "accepted": candidate_result.accepted,
                "target_reached": candidate_result.target_reached,
            }
            for candidate_mode, candidate_result in child_results
        ],
    )
    _atomic_json(
        output / "audit.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": PARTIAL_ELL1_SOLVER_ARCHITECTURE,
            "inputs": {
                "par": str(par),
                "tim": str(tim),
                "par_sha256": _sha256(par),
                "tim_sha256": _sha256(tim),
                "clock_override_sha256": (
                    _sha256(clock_override) if clock_override is not None else None
                ),
            },
            "bootstrap": bootstrap,
            "effective_solver_config": child_config,
            "selected_mode_id": mode.mode_id,
            "failed_modes": failures,
        },
    )
    manifest_paths = tuple(output / name for name in artifact_sources)
    manifest_paths += (
        output / "audit.json",
        output / "certification.json",
        output / "candidate_portfolio.json",
    )
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": PARTIAL_ELL1_SOLVER_ARCHITECTURE,
            "artifacts": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in manifest_paths
            },
        },
    )
    return BayesianBinaryLatticeSolverResult(
        model_name="ELL1",
        model_family="ell1",
        engine="partial_ell1_bootstrap+mature_ell1_phase_adapted",
        output_dir=output,
        solution_par=output / "solution.par",
        solution_tim=output / "solution.tim",
        residuals_csv=output / "residuals.csv",
        residuals_png=output / "residuals.png",
        fit_parameters=best.fit_parameters,
        chi2=best.chi2,
        disk_reload_chi2=best.disk_reload_chi2,
        reduced_chi2=best.reduced_chi2,
        weighted_rms_us=best.weighted_rms_us,
        accepted=best.accepted,
        target_reached=best.target_reached,
        searched_cells=best.searched_cells,
        annealed_modes=best.annealed_modes,
        certified_modes=best.certified_modes,
        resumed=best.resumed,
    )


def _resume_completed_coarse_binary_result(
    *,
    par: Path,
    tim: Path,
    output: Path,
    config: BayesianBinaryLatticeSolverConfig,
    clock_override: Path | None,
) -> BayesianBinaryLatticeSolverResult | None:
    audit_path = output / "audit.json"
    certification_path = output / "certification.json"
    manifest_path = output / "artifact_manifest.json"
    if not all(path.is_file() for path in (audit_path, certification_path, manifest_path)):
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("architecture") != COARSE_BINARY_SOLVER_ARCHITECTURE:
        return None
    _require_current_schema(audit, label="Completed coarse binary audit")
    inputs = audit.get("inputs", {})
    if inputs.get("par_sha256") != _sha256(par) or inputs.get("tim_sha256") != _sha256(tim):
        raise RuntimeError("Completed coarse binary result does not match resume inputs.")
    expected_clock_hash = _sha256(clock_override) if clock_override is not None else None
    if inputs.get("clock_override_sha256") not in (None, expected_clock_hash):
        raise RuntimeError("Completed coarse binary result uses a different clock override.")
    bootstrap = audit.get("bootstrap", {})
    recommended_scale = float(bootstrap.get("recommended_support_width_scale", 1.0))
    expected_config = replace(
        config,
        support_width_scale=min(config.support_width_scale, recommended_scale),
        plan_observations_on_failure=False,
    )
    if to_json_ready(expected_config) != audit.get("effective_solver_config"):
        raise RuntimeError("Completed coarse binary result uses a different solver config.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_current_schema(manifest, label="Completed coarse binary artifact manifest")
    for name, record in manifest.get("artifacts", {}).items():
        path = Path(str(record.get("path", "")))
        if not path.is_file() or _sha256(path) != str(record.get("sha256", "")):
            raise RuntimeError(f"Resume artifact failed SHA-256: {name}")
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    _require_current_schema(
        certification,
        label="Completed coarse binary certification",
    )
    counts = certification["portfolio_counts"]
    return BayesianBinaryLatticeSolverResult(
        model_name=str(certification["model_name"]),
        model_family=str(certification["model_family"]),
        engine=str(certification["engine"]),
        output_dir=output,
        solution_par=output / "solution.par",
        solution_tim=output / "solution.tim",
        residuals_csv=output / "residuals.csv",
        residuals_png=output / "residuals.png",
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
        resumed=True,
    )


def _solve_coarse_binary_portfolio(
    *,
    par: Path,
    tim: Path,
    output: Path,
    config: BayesianBinaryLatticeSolverConfig,
    clock_override: Path | None,
    resume: bool,
) -> BayesianBinaryLatticeSolverResult:
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        completed = _resume_completed_coarse_binary_result(
            par=par,
            tim=tim,
            output=output,
            config=config,
            clock_override=clock_override,
        )
        if completed is not None:
            return completed
    bootstrap = bootstrap_coarse_binary(
        par_path=par,
        tim_path=tim,
        output_dir=output / "coarse_binary_bootstrap",
        config=config.coarse_binary_bootstrap,
        clock_override_path=clock_override,
        resume=resume,
    )
    child_config = replace(
        config,
        support_width_scale=min(
            config.support_width_scale,
            bootstrap.recommended_support_width_scale,
        ),
        plan_observations_on_failure=False,
    )
    child_results: list[tuple[object, BayesianBinaryLatticeSolverResult]] = []
    failures: list[dict[str, str]] = []
    for mode in bootstrap.modes:
        child_output = output / "bootstrap_mode_solutions" / mode.mode_id
        try:
            result = solve_bayesian_binary_lattice(
                par_path=mode.par_path,
                tim_path=bootstrap.normalized_tim_path,
                output_dir=child_output,
                config=child_config,
                clock_override_path=clock_override,
                resume=resume,
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            failures.append(
                {
                    "mode_id": mode.mode_id,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        child_results.append((mode, result))
    bootstrap_payload = bootstrap.to_dict()
    if not child_results:
        failure_audit = {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": COARSE_BINARY_SOLVER_ARCHITECTURE,
            "status": "failed",
            "failure_stage": "complete_mode_portfolio",
            "inputs": {
                "par": str(par),
                "tim": str(tim),
                "par_sha256": _sha256(par),
                "tim_sha256": _sha256(tim),
                "clock_override_sha256": (
                    _sha256(clock_override) if clock_override is not None else None
                ),
            },
            "bootstrap": bootstrap_payload,
            "effective_solver_config": to_json_ready(child_config),
            "failed_modes": failures,
        }
        _atomic_json(output / "audit.json", failure_audit)
        failure_summary = " | ".join(
            f"{failure['mode_id']}: {failure['error']}" for failure in failures[:3]
        )
        raise RuntimeError(
            f"Every coarse {bootstrap.model_name} bootstrap mode failed the complete "
            f"APOLLO search. {failure_summary} Diagnostics: {output / 'audit.json'}"
        )
    mode, best = min(child_results, key=lambda item: _solver_result_rank(item[1]))
    artifact_sources = {
        "solution.par": best.solution_par,
        "solution.tim": best.solution_tim,
        "residuals.csv": best.residuals_csv,
        "residuals.png": best.residuals_png,
    }
    for name, source in artifact_sources.items():
        shutil.copyfile(Path(source), output / name)

    child_certification_path = Path(best.output_dir) / "certification.json"
    child_certification = json.loads(child_certification_path.read_text(encoding="utf-8"))
    engine = f"canonical_keplerian_bootstrap+{best.engine}"
    artifacts = {
        "par": str(output / "solution.par"),
        "tim": str(output / "solution.tim"),
        "residuals_csv": str(output / "residuals.csv"),
        "residuals_png": str(output / "residuals.png"),
    }
    certification = {
        **{key: value for key, value in child_certification.items() if key != "artifacts"},
        "architecture": COARSE_BINARY_SOLVER_ARCHITECTURE,
        "model_name": bootstrap.model_name,
        "model_family": bootstrap.model_family,
        "engine": engine,
        "coarse_binary_bootstrap": {
            "audit": str(bootstrap.audit_path),
            "normalized_tim": str(bootstrap.normalized_tim_path),
            "selected_mode_id": mode.mode_id,
            "selected_mode_par": str(mode.par_path),
            "informative_session_id": bootstrap.informative_session_id,
            "informative_session_toas": bootstrap.informative_session_toas,
            "mode_count": len(bootstrap.modes),
            "recommended_support_width_scale": (
                bootstrap.recommended_support_width_scale
            ),
            "effective_support_width_scale": child_config.support_width_scale,
            "failed_modes": failures,
        },
        "artifacts": artifacts,
    }
    _atomic_json(output / "certification.json", certification)
    _atomic_json(
        output / "candidate_portfolio.json",
        [
            {
                "mode_id": candidate_mode.mode_id,
                "seed_par": str(candidate_mode.par_path),
                "informative_session_chi2": candidate_mode.informative_session_chi2,
                "all_session_shape_chi2": candidate_mode.all_session_shape_chi2,
                "search_output": str(candidate_result.output_dir),
                "chi2": candidate_result.chi2,
                "reduced_chi2": candidate_result.reduced_chi2,
                "accepted": candidate_result.accepted,
                "target_reached": candidate_result.target_reached,
            }
            for candidate_mode, candidate_result in child_results
        ],
    )
    _atomic_json(
        output / "audit.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": COARSE_BINARY_SOLVER_ARCHITECTURE,
            "inputs": {
                "par": str(par),
                "tim": str(tim),
                "par_sha256": _sha256(par),
                "tim_sha256": _sha256(tim),
                "clock_override_sha256": (
                    _sha256(clock_override) if clock_override is not None else None
                ),
            },
            "bootstrap": bootstrap_payload,
            "effective_solver_config": to_json_ready(child_config),
            "selected_mode_id": mode.mode_id,
            "failed_modes": failures,
        },
    )
    manifest_paths = tuple(output / name for name in artifact_sources)
    manifest_paths += (
        output / "audit.json",
        output / "certification.json",
        output / "candidate_portfolio.json",
    )
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": COARSE_BINARY_SOLVER_ARCHITECTURE,
            "artifacts": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in manifest_paths
            },
        },
    )
    return BayesianBinaryLatticeSolverResult(
        model_name=bootstrap.model_name,
        model_family=bootstrap.model_family,
        engine=engine,
        output_dir=output,
        solution_par=output / "solution.par",
        solution_tim=output / "solution.tim",
        residuals_csv=output / "residuals.csv",
        residuals_png=output / "residuals.png",
        fit_parameters=best.fit_parameters,
        chi2=best.chi2,
        disk_reload_chi2=best.disk_reload_chi2,
        reduced_chi2=best.reduced_chi2,
        weighted_rms_us=best.weighted_rms_us,
        accepted=best.accepted,
        target_reached=best.target_reached,
        searched_cells=best.searched_cells,
        annealed_modes=best.annealed_modes,
        certified_modes=best.certified_modes,
        resumed=best.resumed,
    )


def _solve_phase_lattice_model(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: BayesianBinaryLatticeSolverConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
    baseline_par_path: Path | None = None,
    baseline_tim_path: Path | None = None,
    solver_architecture: str = SOLVER_ARCHITECTURE,
) -> BayesianBinaryLatticeSolverResult:
    """Run the private phase-lattice numerical kernel for one prepared model."""

    active = config or BayesianBinaryLatticeSolverConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    clock_override = (
        Path(clock_override_path).resolve() if clock_override_path is not None else None
    )
    baseline_par = (
        Path(baseline_par_path).resolve() if baseline_par_path is not None else None
    )
    baseline_tim = (
        Path(baseline_tim_path).resolve() if baseline_tim_path is not None else None
    )
    if (baseline_par is None) != (baseline_tim is None):
        raise ValueError(
            "Timing-update conditioning requires both baseline par and baseline TIM."
        )
    for path in (par, tim):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (baseline_par, baseline_tim):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if clock_override is not None and not clock_override.is_file():
        raise FileNotFoundError(clock_override)

    if not _par_declares_binary_model(par):
        raise ValueError("The prepared binary kernel requires a par file with BINARY.")

    if is_incomplete_ell1_par(par) and inspect_coarse_binary_par(par).model_name == "ELL1":
        result = _solve_partial_ell1_portfolio(
            par=par,
            tim=tim,
            output=output,
            config=active,
            clock_override=clock_override,
            resume=resume,
        )
        return _finalize_observation_plan(
            result=result,
            config=active,
            clock_override_path=clock_override,
        )
    if is_incomplete_binary_par(par):
        result = _solve_coarse_binary_portfolio(
            par=par,
            tim=tim,
            output=output,
            config=active,
            clock_override=clock_override,
            resume=resume,
        )
        return _finalize_observation_plan(
            result=result,
            config=active,
            clock_override_path=clock_override,
        )

    seed_model = _load_pint_model(par)
    model_name = detect_phase_model_name(seed_model)
    input_fit_parameters = tuple(str(name).upper() for name in seed_model.free_params)
    if (
        active.use_mature_ell1_engine
        and baseline_tim is None
        and model_name == "ELL1"
        and len(input_fit_parameters) == len(ELL1_FIT_PARAMETERS)
        and set(input_fit_parameters) == set(ELL1_FIT_PARAMETERS)
    ):
        result = solve_bayesian_ell1_dual_lattice(
            par_path=par,
            tim_path=tim,
            output_dir=output,
            config=_ell1_config(active),
            clock_override_path=clock_override,
            resume=resume,
        )
        return _finalize_observation_plan(
            result=_convert_ell1_result(result),
            config=active,
            clock_override_path=clock_override,
        )

    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "par_sha256": _sha256(par),
        "tim_sha256": _sha256(tim),
        "clock_override_sha256": _sha256(clock_override) if clock_override else None,
        "baseline_par_sha256": _sha256(baseline_par) if baseline_par else None,
        "baseline_tim_sha256": _sha256(baseline_tim) if baseline_tim else None,
    }
    run_fingerprint = _json_digest(
        {
            "solver_schema_version": SOLVER_SCHEMA_VERSION,
            "inputs": input_hashes,
            "solver_config": active,
            "solver_architecture": solver_architecture,
            "model_name": model_name,
            "fit_parameters": input_fit_parameters,
        }
    )
    audit_path = output / "audit.json"
    if resume and audit_path.is_file():
        prior = json.loads(audit_path.read_text(encoding="utf-8"))
        _require_current_schema(prior, label="Binary resume audit")
        if prior.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("Resume inputs or solver config do not match.")
        completed = _resume_completed_result(output=output, run_fingerprint=run_fingerprint)
        if completed is not None:
            return _finalize_observation_plan(
                result=completed,
                config=active,
                clock_override_path=clock_override,
            )

    capability: Any = BINARY_MODEL_CAPABILITIES[model_name]
    audit: dict[str, Any] = {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "architecture": solver_architecture,
        "run_fingerprint": run_fingerprint,
        "inputs": {
            "par": str(par),
            "tim": str(tim),
            "clock_override": str(clock_override) if clock_override else None,
            "baseline_par": str(baseline_par) if baseline_par else None,
            "baseline_tim": str(baseline_tim) if baseline_tim else None,
            **input_hashes,
        },
        "model_capability": capability,
        "solver_config": active,
        "scientific_contract": {
            "fit_parameters": input_fit_parameters,
            "parameter_expansion": "disabled",
            "prohibited_automatic_parameters": PROHIBITED_AUTOMATIC_PARAMETERS,
            "source_class_prior": "forbidden",
            "historical_solution_seed": "forbidden",
            "manual_integer_wrap": "forbidden",
            "ordinary_wls_search_fallback": "forbidden",
            "phase_lattice_cell_policy": "all orbital-cycle cells must be searched",
            "final_certifier": "PINT WLS with fixed pulse numbers",
            "global_optimum_claim": (
                "Finite all-cell and posterior-beam search; no exhaustive-global claim."
            ),
        },
    }
    _atomic_json(audit_path, audit)

    context = build_phase_lattice_search_context(
        par_path=par,
        tim_path=tim,
        maximum_cell_width=active.maximum_cell_width,
        support_width_scale=active.support_width_scale,
        astrometric_bound_arcsec=active.astrometric_bound_arcsec,
        clock_override_path=clock_override,
    )
    update_conditioning = None
    if baseline_par is not None and baseline_tim is not None:
        context, update_conditioning = condition_timing_update_context(
            context=context,
            baseline_par_path=baseline_par,
            baseline_tim_path=baseline_tim,
            clock_override_path=clock_override,
            config=active.timing_update_prior,
        )
    if tuple(context.evaluator.parameter_names) != input_fit_parameters:
        raise RuntimeError("Input fit flags changed while building binary context.")
    audit.update(
        {
            "toa_count": len(context.evaluator.toas.table),
            "session_count": len(context.evaluator.sessions),
            "observation_span_days": context.adapter.observation_span_days,
            "cell_count": len(context.cells),
            "coordinate_names": context.adapter.coordinate_names,
            "physical_support_lower": context.support.physical_lower,
            "physical_support_upper": context.support.physical_upper,
            "search_support_lower": context.support.search_lower,
            "search_support_upper": context.support.search_upper,
            "timing_update_conditioning": update_conditioning,
        }
    )
    _atomic_json(audit_path, audit)


    cell_dir = output / "checkpoints" / "cells"
    all_conditional: list[BinaryConditionalMode] = []
    cell_summaries: list[dict[str, Any]] = []
    for cell in (context.cells):
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
            modes = search_phase_lattice_cell(
                context=context,
                cell=cell,
                config=active.cell_search,
            )
            _atomic_json(
                checkpoint,
                {
                    "schema_version": SOLVER_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "model_name": model_name,
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
                "best_strict_score": min((mode.strict_score for mode in modes), default=None),
                "lattice_mass_converged": all(mode.lattice_mass_converged for mode in modes),
            }
        )

    selected_conditional = _select_conditional_modes(all_conditional, active)
    if not selected_conditional:
        raise RuntimeError("No in-support posterior mode survived phase-lattice search.")
    anneal_dir = output / "checkpoints" / "annealing"
    all_annealed: list[BinaryAnnealedMode] = []
    anneal_summaries: list[dict[str, Any]] = []
    for rank, mode in enumerate(selected_conditional):
        mode_id = f"{rank:04d}_{mode.cell_id}_{mode.grid_index[0]}_{mode.grid_index[1]}"
        checkpoint = anneal_dir / f"{mode_id}.json"
        if resume and checkpoint.is_file():
            annealed = _load_checkpoint_items(
                checkpoint,
                run_fingerprint=run_fingerprint,
                item_name="modes",
                loader=_annealed_mode_from_payload,
            )
            resumed_mode = True
        else:
            annealed = anneal_phase_lattice_mode(
                context=context,
                mode=mode,
                config=active.annealing,
            )
            _atomic_json(
                checkpoint,
                {
                    "schema_version": SOLVER_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "mode_id": mode_id,
                    "source_mode": mode,
                    "modes": annealed,
                },
            )
            resumed_mode = False
        all_annealed.extend(annealed)
        anneal_summaries.append(
            {
                "mode_id": mode_id,
                "source_strict_score": mode.strict_score,
                "output_mode_count": len(annealed),
                "best_total_chi2": min((item.total_chi2 for item in annealed), default=None),
                "resumed": resumed_mode,
            }
        )




    selected_annealed = _select_annealed_modes(all_annealed, active)
    if not selected_annealed:
        raise RuntimeError("Posterior annealing produced no certifiable timing mode.")
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
                label=f"Binary candidate certification {candidate_id}",
            )
            if record.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError(f"Candidate provenance mismatch: {candidate_id}")
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
                    timing_update_active=update_conditioning is not None,
                )
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                record = {
                    "schema_version": SOLVER_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "candidate_id": candidate_id,
                    "model_name": model_name,
                    "source_cell_id": mode.cell_id,
                    "prior_component_id": mode.prior_component_id,
                    "prior_log_weight": mode.prior_log_weight,
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
        raise RuntimeError("Every annealed timing mode failed PINT certification.")


    artifacts = {
        "par": str(output / "solution.par"),
        "tim": str(output / "solution.tim"),
        "residuals_csv": str(output / "residuals.csv"),
        "residuals_png": str(output / "residuals.png"),
    }
    solution_status = (
        "qualified_retained_solution"
        if bool(best_record["accepted"])
        else "insufficient_evidence_for_unique_solution"
    )
    solution_interpretation = (
        "The retained solution passed fixed-pulse-number PINT certification; "
        "global uniqueness outside the finite searched portfolio is not claimed."
        if bool(best_record["accepted"])
        else "No retained phase path passed fixed-pulse-number PINT certification "
        "after the required model portfolio. The current data do not support a "
        "unique timing solution; unmodeled physics or finite-search limitations are "
        "not excluded."
    )
    final_certification = {
        **{key: value for key, value in best_record.items() if key != "artifacts"},
        "architecture": solver_architecture,
        "engine": "generic_phase_adapted_timing_model",
        "model_name": best_record.get("model_name", model_name),
        "model_family": best_record.get("model_family", context.adapter.family),
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
        "certification_scope": best_record.get(
            "certification_scope",
            "fixed_pulse_number_pint_reported_toa_errors",
        ),
        "solution_claim": {
            "status": solution_status,
            "pint_certified": bool(best_record["accepted"]),
            "uniqueness_certified": False,
            "uniqueness_scope": None,
            "interpretation": solution_interpretation,
        },
        "artifacts": artifacts,
    }
    _atomic_json(output / "certification.json", final_certification)
    _atomic_json(output / "candidate_portfolio.json", candidate_records)
    audit["search"] = {
        "cells": cell_summaries,
        "selected_conditional_modes": selected_conditional,
        "annealing": anneal_summaries,
        "selected_annealed_modes": selected_annealed,
        "candidate_certifications": candidate_records,
    }
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
    manifest_artifacts = {
        path.name: {"path": str(path), "sha256": _sha256(path)}
        for path in manifest_paths
    }
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": solver_architecture,
            "run_fingerprint": run_fingerprint,
            "artifacts": manifest_artifacts,
        },
    )
    return _finalize_observation_plan(
        result=_result_from_certification(output, final_certification, resumed=False),
        config=active,
        clock_override_path=clock_override,
    )


def solve_bayesian_binary_lattice(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: BayesianBinaryLatticeSolverConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
    baseline_par_path: Path | None = None,
    baseline_tim_path: Path | None = None,
) -> BayesianBinaryLatticeSolverResult:
    """Run the binary phase-orbit posterior and integer-path workflow."""

    par = Path(par_path)
    if not par.is_file():
        raise FileNotFoundError(par)
    if not _par_declares_binary_model(par):
        raise ValueError(
            "The binary APOLLO entry point requires a par file with BINARY. "
            "Use blind orbit discovery to prepare a binary seed first."
        )
    return _solve_phase_lattice_model(
        par_path=par,
        tim_path=tim_path,
        output_dir=output_dir,
        config=config,
        clock_override_path=clock_override_path,
        resume=resume,
        baseline_par_path=baseline_par_path,
        baseline_tim_path=baseline_tim_path,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run APOLLO binary phase-orbit Bayesian lattice search for a par "
            "file that declares BINARY."
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
    parser.add_argument(
        "--no-binary-model-selection",
        action="store_true",
        help="Keep the seed BINARY model after phase-path certification.",
    )
    parser.add_argument("--model-selection-min-delta-bic", type=float, default=10.0)
    parser.add_argument(
        "--model-selection-trigger-reduced-chi2",
        type=float,
        default=1.5,
    )
    parser.add_argument("--model-selection-multistarts", type=int, default=8)
    parser.add_argument(
        "--no-observation-plan",
        action="store_true",
        help="Do not rank future observing windows when certification fails.",
    )
    parser.add_argument("--observation-horizon-days", type=float, default=180.0)
    parser.add_argument("--observation-duration-min", type=float, default=20.0)
    parser.add_argument("--observation-toas-per-session", type=int, default=8)
    parser.add_argument("--observation-top", type=int, default=20)
    parser.add_argument("--observation-posterior-samples", type=int, default=256)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BayesianBinaryLatticeSolverConfig(
        cell_search=BinaryCellSearchConfig(
            orbital_samples_per_cell=args.orbital_samples_per_cell,
            phase_samples=args.phase_samples,
            modes_per_cell=args.modes_per_cell,
            chart_maximum_evaluations=args.chart_evaluations,
        ),
        maximum_modes_to_anneal=args.maximum_modes_to_anneal,
        maximum_modes_to_certify=args.maximum_modes_to_certify,
        fit_iterations=args.fit_iterations,
        strict_reduced_chi2_threshold=args.strict_reduced_chi2,
        target_reduced_chi2_threshold=args.target_reduced_chi2,
        physical_model_selection=BinaryModelSelectionConfig(
            enabled=not args.no_binary_model_selection,
            minimum_delta_bic=args.model_selection_min_delta_bic,
            extension_trigger_reduced_chi2=(
                args.model_selection_trigger_reduced_chi2
            ),
            maximum_multistarts_per_hypothesis=args.model_selection_multistarts,
        ),
        plan_observations_on_failure=not args.no_observation_plan,
        observation_planner=PintObservationPlannerConfig(
            horizon_days=args.observation_horizon_days,
            duration_min=args.observation_duration_min,
            toas_per_session=args.observation_toas_per_session,
            maximum_recommendations=args.observation_top,
            design=BayesianObservationDesignConfig(
                posterior_samples_per_mode=args.observation_posterior_samples
            ),
        ),
    )
    result = solve_bayesian_binary_lattice(
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
    "BayesianBinaryLatticeSolverConfig",
    "BayesianBinaryLatticeSolverResult",
    "BinaryModelSelectionConfig",
    "PhysicalModelSelectionConfig",
    "build_arg_parser",
    "main",
    "solve_bayesian_binary_lattice",
]
