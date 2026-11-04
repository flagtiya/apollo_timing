from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import shutil
from typing import Any

from timing_accelerator.io.artifacts import sha256_file as _sha256, write_json_atomic
from timing_accelerator.backend.pint_backend import _load_pint_model
from timing_accelerator.inference.bayesian_binary_lattice import (
    BinaryAnnealingConfig,
    BinaryCellSearchConfig,
    IsolatedColdStartConfig,
    PintPhaseEvaluator,
)
from timing_accelerator.inference.bayesian_isolated_session_update import (
    ISOLATED_SESSION_UPDATE_ARCHITECTURE,
    IsolatedSessionUpdateConfig,
    IsolatedSessionMarginalizationDecision,
    decide_isolated_session_marginalization,
    solve_isolated_session_update,
    write_isolated_marginalized_tim,
    write_isolated_session_update_artifacts,
)
from timing_accelerator.inference.bayesian_isolated_update_model_selection import (
    IsolatedUpdateModelSelectionConfig,
)
from timing_accelerator.inference.bayesian_observation_planner import (
    PintObservationPlannerConfig,
)
from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
)
from timing_accelerator.inference.bayesian_timing_update import (
    TimingUpdatePriorConfig,
)
from timing_accelerator.pipeline.bayesian_binary_lattice_solver import (
    BayesianBinaryLatticeSolverConfig,
    BayesianBinaryLatticeSolverResult,
    _solve_phase_lattice_model,
)


SOLVER_ARCHITECTURE = "apollo_isolated_spin_sky_phase_lattice_solver"
AUTOMATIC_ISOLATED_FIT_PARAMETERS = (
    "F0",
    "F1",
    "RAJ",
    "DECJ",
    "ELONG",
    "ELAT",
)


@dataclass(frozen=True)
class IsolatedCellSearchConfig:
    """Single-star posterior search controls with no orbital-cell parameter."""

    phase_samples: int = 12
    modes_per_cell: int = 32
    paths_per_component: int = 8
    chart_maximum_evaluations: int = 60
    finite_difference_step: float = 2.0e-5
    coordinate_prior_sigma: float = 0.75
    local_integer_radius: int = 0
    lattice: SpinIntegerMarginalConfig = field(
        default_factory=lambda: SpinIntegerMarginalConfig(
            initial_lattice_candidates=32,
            maximum_lattice_candidates=128,
            lattice_log_mass_tolerance=1.0e-5,
            maximum_lattice_nodes=500_000,
            added_session_phase_sigma_cycles=0.05,
        )
    )

    def __post_init__(self) -> None:
        positive_integers = (
            self.phase_samples,
            self.modes_per_cell,
            self.paths_per_component,
            self.chart_maximum_evaluations,
        )
        if any(int(value) <= 0 for value in positive_integers):
            raise ValueError("Isolated cell-search budgets must be positive integers.")
        positive_scales = (
            self.finite_difference_step,
            self.coordinate_prior_sigma,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_scales):
            raise ValueError("Isolated cell-search scales must be positive and finite.")
        if int(self.local_integer_radius) < 0:
            raise ValueError("The isolated local integer radius cannot be negative.")

    def to_core_config(self) -> BinaryCellSearchConfig:
        return BinaryCellSearchConfig(
            orbital_samples_per_cell=1,
            phase_samples=self.phase_samples,
            modes_per_cell=self.modes_per_cell,
            paths_per_component=self.paths_per_component,
            chart_maximum_evaluations=self.chart_maximum_evaluations,
            finite_difference_step=self.finite_difference_step,
            coordinate_prior_sigma=self.coordinate_prior_sigma,
            local_integer_radius=self.local_integer_radius,
            lattice=self.lattice,
        )


@dataclass(frozen=True)
class IsolatedAnnealingConfig:
    """Single-star phase-noise transport controls."""

    phase_sigma_schedule: tuple[float, ...] = (
        0.10,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.001,
        0.0005,
        0.0002,
        0.0001,
        0.0,
    )
    coordinate_prior_sigma: float = 0.20
    paths_per_state: int = 3
    integer_path_transport: bool = True
    beam_size: int = 4
    finite_difference_step: float = 2.0e-5
    maximum_closure_evaluations: int = 300
    lattice: SpinIntegerMarginalConfig = field(
        default_factory=lambda: SpinIntegerMarginalConfig(
            initial_lattice_candidates=32,
            maximum_lattice_candidates=128,
            lattice_log_mass_tolerance=1.0e-6,
            maximum_lattice_nodes=1_000_000,
        )
    )

    def __post_init__(self) -> None:
        schedule = tuple(float(value) for value in self.phase_sigma_schedule)
        if not schedule or any(not math.isfinite(value) or value < 0.0 for value in schedule):
            raise ValueError("The isolated phase-noise schedule must be nonnegative.")
        if any(right > left for left, right in zip(schedule, schedule[1:])):
            raise ValueError("The isolated phase-noise schedule must be nonincreasing.")
        if schedule[-1] != 0.0:
            raise ValueError("The isolated phase-noise schedule must end at zero.")
        positive = (self.coordinate_prior_sigma, self.finite_difference_step)
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Isolated annealing scales must be positive and finite.")
        integers = (
            self.paths_per_state,
            self.beam_size,
            self.maximum_closure_evaluations,
        )
        if any(int(value) <= 0 for value in integers):
            raise ValueError("Isolated annealing budgets must be positive integers.")

    def to_core_config(self) -> BinaryAnnealingConfig:
        return BinaryAnnealingConfig(
            phase_sigma_schedule=self.phase_sigma_schedule,
            coordinate_prior_sigma=self.coordinate_prior_sigma,
            paths_per_state=self.paths_per_state,
            integer_path_transport=self.integer_path_transport,
            integer_path_proposal="lattice",
            beam_size=self.beam_size,
            finite_difference_step=self.finite_difference_step,
            maximum_closure_evaluations=self.maximum_closure_evaluations,
            lattice=self.lattice,
        )


@dataclass(frozen=True)
class IsolatedPulsarSolverConfig:
    """Configuration owned by the isolated spin-sky inference workflow."""

    maximum_cell_width: float = 0.25
    support_width_scale: float = 1.0
    astrometric_bound_arcsec: float = 300.0
    cell_search: IsolatedCellSearchConfig = field(
        default_factory=IsolatedCellSearchConfig
    )
    annealing: IsolatedAnnealingConfig = field(
        default_factory=IsolatedAnnealingConfig
    )
    maximum_modes_to_anneal: int = 8
    conditional_score_window: float = 2_000.0
    maximum_modes_to_certify: int = 8
    annealed_chi2_window: float = 2_000.0
    fit_iterations: int = 10
    strict_reduced_chi2_threshold: float = 20.0
    target_reduced_chi2_threshold: float = 10.0
    disk_reload_chi2_tolerance: float = 1.0e-2
    plan_observations_on_failure: bool = True
    observation_planner: PintObservationPlannerConfig = field(
        default_factory=PintObservationPlannerConfig
    )
    timing_update_prior: TimingUpdatePriorConfig = field(
        default_factory=TimingUpdatePriorConfig
    )
    isolated_update_model_selection: IsolatedUpdateModelSelectionConfig = field(
        default_factory=IsolatedUpdateModelSelectionConfig
    )
    isolated_cold_start: IsolatedColdStartConfig = field(
        default_factory=IsolatedColdStartConfig
    )
    session_update: IsolatedSessionUpdateConfig = field(
        default_factory=IsolatedSessionUpdateConfig
    )

    def __post_init__(self) -> None:
        positive = (
            self.maximum_cell_width,
            self.support_width_scale,
            self.astrometric_bound_arcsec,
            self.conditional_score_window,
            self.annealed_chi2_window,
            self.strict_reduced_chi2_threshold,
            self.target_reduced_chi2_threshold,
            self.disk_reload_chi2_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Isolated solver scales and thresholds must be positive.")
        budgets = (
            self.maximum_modes_to_anneal,
            self.maximum_modes_to_certify,
            self.fit_iterations,
        )
        if any(int(value) <= 0 for value in budgets):
            raise ValueError("Isolated solver budgets must be positive integers.")
        if self.target_reduced_chi2_threshold > self.strict_reduced_chi2_threshold:
            raise ValueError("The target reduced chi-square cannot exceed acceptance.")

    def to_core_config(self) -> BayesianBinaryLatticeSolverConfig:
        """Translate to the private numerical kernel without binary policy knobs."""

        return BayesianBinaryLatticeSolverConfig(
            maximum_cell_width=self.maximum_cell_width,
            support_width_scale=self.support_width_scale,
            astrometric_bound_arcsec=self.astrometric_bound_arcsec,
            cell_search=self.cell_search.to_core_config(),
            annealing=self.annealing.to_core_config(),
            maximum_modes_to_anneal=self.maximum_modes_to_anneal,
            conditional_score_window=self.conditional_score_window,
            maximum_modes_to_certify=self.maximum_modes_to_certify,
            annealed_chi2_window=self.annealed_chi2_window,
            fit_iterations=self.fit_iterations,
            strict_reduced_chi2_threshold=self.strict_reduced_chi2_threshold,
            target_reduced_chi2_threshold=self.target_reduced_chi2_threshold,
            disk_reload_chi2_tolerance=self.disk_reload_chi2_tolerance,
            use_mature_ell1_engine=False,
            plan_observations_on_failure=self.plan_observations_on_failure,
            observation_planner=self.observation_planner,
            timing_update_prior=self.timing_update_prior,
            isolated_update_model_selection=self.isolated_update_model_selection,
            isolated_cold_start=self.isolated_cold_start,
        )


@dataclass(frozen=True)
class IsolatedPulsarSolverResult:
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
    certification_scope: str = "fixed_pulse_number_pint"
    pint_certified: bool = True
    scientific_status: str = "pint_certification"

    @property
    def searched_states(self) -> int:
        """Compatibility alias for releases that mislabeled cells as states."""

        return self.searched_cells


def _atomic_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload, sort_keys=True)


def _write_automatic_isolated_fit_par(source: Path, target: Path) -> Path:
    """Create the maximal basic model used by isolated phase-path search.

    Unsupported fixed spin derivatives must not inject phase curvature into a
    new-source search. They are reset here and may only return through the
    fixed-path isolated model-comparison stage.
    """

    lines = Path(source).read_text(encoding="utf-8").splitlines()
    parameter_names = {
        fields[0].upper()
        for raw_line in lines
        if len(fields := raw_line.strip().split()) >= 2
        and not raw_line.lstrip().startswith(("#", "C "))
    }
    if "F0" not in parameter_names:
        raise ValueError("Automatic isolated fitting requires F0 in the input par.")
    if {"RAJ", "DECJ"}.issubset(parameter_names):
        astrometry = ("RAJ", "DECJ")
    elif {"ELONG", "ELAT"}.issubset(parameter_names):
        astrometry = ("ELONG", "ELAT")
    else:
        raise ValueError(
            "Automatic isolated fitting requires RAJ/DECJ or ELONG/ELAT in the input par."
        )
    selected = {"F0", "F1", *astrometry}

    prepared: list[str] = []
    inserted_f1 = "F1" in parameter_names
    for raw_line in lines:
        fields = raw_line.strip().split()
        name = fields[0].upper() if len(fields) >= 2 else ""
        if name in selected:
            if len(fields) >= 3 and fields[2] in {"0", "1"}:
                fields[2] = "1"
            else:
                fields.insert(2, "1")
            prepared.append(" ".join(fields))
        elif name.startswith("F") and name[1:].isdigit() and int(name[1:]) >= 2:
            fields[1] = "0"
            if len(fields) >= 3 and fields[2] in {"0", "1"}:
                fields[2] = "0"
            prepared.append(" ".join(fields))
        else:
            prepared.append(raw_line)
        if name == "F0" and not inserted_f1:
            prepared.append("F1 0 1")
            inserted_f1 = True

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(prepared) + "\n", encoding="utf-8")
    return target


def _convert_result(
    result: BayesianBinaryLatticeSolverResult,
) -> IsolatedPulsarSolverResult:
    return IsolatedPulsarSolverResult(
        model_name=result.model_name,
        model_family=result.model_family,
        engine=result.engine,
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
        observation_plan_dir=result.observation_plan_dir,
        observation_plan_status=result.observation_plan_status,
        certification_scope="fixed_pulse_number_pint",
        pint_certified=bool(result.accepted),
        scientific_status=(
            getattr(
                result,
                "scientific_status",
                "qualified_retained_solution"
                if result.accepted
                else "insufficient_evidence_for_unique_solution",
            )
        ),
    )


def _record_isolated_input_policy(
    *,
    output: Path,
    source_par: Path,
    effective_par: Path,
    fit_parameter_source: str,
) -> None:
    audit_path = output / "audit.json"
    audit_updated = False
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        architecture = str(audit.get("architecture") or SOLVER_ARCHITECTURE)
        audit["architecture"] = architecture
        audit["isolated_input_policy"] = {
            "source_par": str(source_par),
            "source_par_sha256": _sha256(source_par),
            "effective_par": str(effective_par),
            "effective_par_sha256": _sha256(effective_par),
            "fit_parameter_source": fit_parameter_source,
            "automatic_parameter_allowlist": AUTOMATIC_ISOLATED_FIT_PARAMETERS,
            "cold_start_basic_parameter_policy": (
                "Search every cold-start phase path with F0, F1, and one complete "
                "astrometric pair; select their smallest supported subset by BIC "
                "after phase closure."
            ),
            "cold_start_spin_derivative_policy": (
                "Fixed F2 and higher values are reset to zero; evidence-based "
                "fixed-path model comparison may release them after phase closure."
            ),
            "search_architecture": (
                "full-sequence session posterior + relative integer path + sparse "
                "whole-session contamination marginalization"
                if architecture == ISOLATED_SESSION_UPDATE_ARCHITECTURE
                else "shared continuous posterior + integer phase lattice + posterior "
                "portfolio + phase-noise annealing + fixed-pulse-number PINT certification"
            ),
        }
        _atomic_json(audit_path, audit)
        audit_updated = True

    certification_path = output / "certification.json"
    certification_updated = False
    if certification_path.is_file():
        certification = json.loads(certification_path.read_text(encoding="utf-8"))
        certification.setdefault("architecture", SOLVER_ARCHITECTURE)
        _atomic_json(certification_path, certification)
        certification_updated = True

    manifest_path = output / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = dict(manifest.get("artifacts") or {})
        if effective_par != source_par:
            artifacts[effective_par.name] = {
                "path": str(effective_par),
                "sha256": _sha256(effective_par),
            }
        if audit_updated and "audit.json" in artifacts:
            artifacts["audit.json"] = {
                "path": str(audit_path),
                "sha256": _sha256(audit_path),
            }
        if certification_updated and "certification.json" in artifacts:
            artifacts["certification.json"] = {
                "path": str(certification_path),
                "sha256": _sha256(certification_path),
            }
        manifest.setdefault("architecture", SOLVER_ARCHITECTURE)
        manifest["artifacts"] = artifacts
        _atomic_json(manifest_path, manifest)


def _record_session_marginalization(
    *,
    output: Path,
    decision: IsolatedSessionMarginalizationDecision,
    full_data_result: Any,
    refit_result: Any | None,
    marginalized_tim: Path | None,
    detection_directory: Path | None,
) -> Path:
    """Record before/after evidence without rewriting the source TIM."""

    output = Path(output)
    summary_path = output / "session_marginalization.json"
    before = full_data_result.best_mode
    after = None if refit_result is None else refit_result.best_mode
    payload = {
        "schema_version": 1,
        "architecture": ISOLATED_SESSION_UPDATE_ARCHITECTURE,
        "decision": {
            "session_indices": decision.session_indices,
            "toa_indices": decision.toa_indices,
            "session_epochs_mjd": decision.session_epochs_mjd,
            "session_probabilities": decision.session_probabilities,
            "standardized_residuals": decision.standardized_residuals,
            "subset_probability": decision.subset_probability,
            "eligible_for_refit": decision.eligible_for_refit,
            "reason": decision.reason,
        },
        "full_data": {
            "status": full_data_result.status,
            "unique_solution": full_data_result.unique_solution,
            "chi2": before.inlier_session_chi2,
            "reduced_chi2": before.reduced_inlier_session_chi2,
            "weighted_rms_us": before.inlier_weighted_rms_us,
            "artifact_directory": (
                None if detection_directory is None else str(detection_directory)
            ),
        },
        "marginalized_refit": (
            None
            if refit_result is None
            else {
                "status": refit_result.status,
                "unique_solution": refit_result.unique_solution,
                "chi2": after.inlier_session_chi2,
                "reduced_chi2": after.reduced_inlier_session_chi2,
                "weighted_rms_us": after.inlier_weighted_rms_us,
                "tim_file": str(marginalized_tim),
            }
        ),
        "scientific_interpretation": (
            "evidence_qualified_refit_accepted"
            if refit_result is not None and refit_result.unique_solution
            else "candidate_outliers_refit_but_timing_model_remains_rejected"
            if refit_result is not None
            else "session_marginalization_not_evidence_qualified"
        ),
        "source_tim_was_modified": False,
    }
    _atomic_json(summary_path, payload)

    for metadata_name in ("audit.json", "certification.json"):
        metadata_path = output / metadata_name
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["session_marginalization_summary"] = str(summary_path)
        metadata["source_tim_was_modified"] = False
        _atomic_json(metadata_path, metadata)

    manifest_path = output / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = dict(manifest.get("artifacts") or {})
        tracked = [summary_path, output / "audit.json", output / "certification.json"]
        if marginalized_tim is not None:
            tracked.append(Path(marginalized_tim))
        if detection_directory is not None:
            tracked.extend(
                Path(detection_directory) / name
                for name in (
                    "isolated_session_update.json",
                    "residuals.png",
                    "artifact_manifest.json",
                )
            )
        for path in tracked:
            if path.is_file():
                key = str(path.relative_to(output)).replace("\\", "/")
                artifacts[key] = {"path": str(path), "sha256": _sha256(path)}
        manifest["artifacts"] = artifacts
        _atomic_json(manifest_path, manifest)
    return summary_path


def solve_isolated_pulsar(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: IsolatedPulsarSolverConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
    baseline_par_path: Path | None = None,
    baseline_tim_path: Path | None = None,
) -> IsolatedPulsarSolverResult:
    """Run the isolated spin-sky posterior and integer-path workflow."""

    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    for path in (par, tim):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_model = _load_pint_model(par)
    if "BINARY" in source_model.params and source_model["BINARY"].value is not None:
        raise ValueError("A binary model must be sent to the binary APOLLO entry point.")

    output.mkdir(parents=True, exist_ok=True)
    timing_update_requested = (
        baseline_par_path is not None or baseline_tim_path is not None
    )
    if (baseline_par_path is None) != (baseline_tim_path is None):
        raise ValueError(
            "An isolated timing update requires both baseline par and baseline TIM."
        )
    effective_par = par
    fit_parameter_source = "input_fit_flags"
    required_update_parameters = {"RAJ", "DECJ", "F0", "F1"}
    input_free_parameters = {
        str(name).upper() for name in source_model.free_params
    }
    if not timing_update_requested or not required_update_parameters.issubset(
        input_free_parameters
    ):
        effective_par = _write_automatic_isolated_fit_par(
            par,
            output / "input.auto_fit.par",
        )
        fit_parameter_source = (
            "automatic_basic_isolated_update_model"
            if timing_update_requested
            else "automatic_maximal_basic_cold_start_model"
        )
        if baseline_par_path is not None:
            # Timing-update conditioning must see the same free-parameter model
            # as the shared solver, even when the source par has no fit flags.
            shutil.copy2(effective_par, baseline_par_path)

    active = config or IsolatedPulsarSolverConfig()
    if timing_update_requested:
        evaluator = PintPhaseEvaluator.from_files(
            par_path=effective_par,
            tim_path=tim,
            clock_override_path=clock_override_path,
        )
        full_data_result = solve_isolated_session_update(
            evaluator=evaluator,
            config=active.session_update,
        )
        decision = decide_isolated_session_marginalization(
            evaluator=evaluator,
            result=full_data_result,
            config=active.session_update,
        )
        session_result = full_data_result
        fit_evaluator = evaluator
        fit_tim = tim
        detection_directory = None
        marginalized_tim = None
        if decision.eligible_for_refit:
            detection_directory = output / "full_data_outlier_detection"
            write_isolated_session_update_artifacts(
                evaluator=evaluator,
                result=full_data_result,
                output_directory=detection_directory,
                tim_path=tim,
            )
            marginalized_tim = write_isolated_marginalized_tim(
                evaluator=evaluator,
                decision=decision,
                path=output / "marginalized.tim",
            )
            fit_tim = marginalized_tim
            fit_evaluator = PintPhaseEvaluator.from_files(
                par_path=effective_par,
                tim_path=fit_tim,
                clock_override_path=clock_override_path,
            )
            refit_config = replace(
                active.session_update,
                maximum_contaminated_sessions=0,
                enable_automatic_session_marginalization=False,
            )
            session_result = solve_isolated_session_update(
                evaluator=fit_evaluator,
                config=refit_config,
            )
        artifacts = write_isolated_session_update_artifacts(
            evaluator=fit_evaluator,
            result=session_result,
            output_directory=output,
            tim_path=fit_tim,
            marginalization_decision=(
                decision if decision.eligible_for_refit else None
            ),
            source_tim_path=tim,
        )
        _record_isolated_input_policy(
            output=output,
            source_par=par,
            effective_par=effective_par,
            fit_parameter_source=fit_parameter_source,
        )
        _record_session_marginalization(
            output=output,
            decision=decision,
            full_data_result=full_data_result,
            refit_result=(session_result if decision.eligible_for_refit else None),
            marginalized_tim=marginalized_tim,
            detection_directory=detection_directory,
        )
        best = session_result.best_mode
        return IsolatedPulsarSolverResult(
            model_name="ISOLATED",
            model_family="isolated",
            engine=ISOLATED_SESSION_UPDATE_ARCHITECTURE,
            output_dir=output,
            solution_par=artifacts.preferred_par_path,
            solution_tim=artifacts.solution_tim_path,
            residuals_csv=artifacts.residuals_csv_path,
            residuals_png=artifacts.residuals_png_path,
            fit_parameters=best.native_parameter_names,
            chi2=best.inlier_session_chi2,
            disk_reload_chi2=best.inlier_session_chi2,
            reduced_chi2=best.reduced_inlier_session_chi2,
            weighted_rms_us=best.inlier_weighted_rms_us,
            accepted=session_result.unique_solution,
            target_reached=session_result.unique_solution,
            searched_cells=1,
            annealed_modes=len(session_result.modes),
            certified_modes=1 if session_result.unique_solution else 0,
            resumed=False,
            observation_plan_dir=None,
            observation_plan_status=(
                "not_required_unique_session_update"
                if session_result.unique_solution
                else "not_evaluated"
            ),
            certification_scope="session_posterior_update_candidate",
            pint_certified=False,
            scientific_status=(
                session_result.status
                if session_result.unique_solution
                else "model_selection_incomplete"
            ),
        )
    shared_result = _solve_phase_lattice_model(
        par_path=effective_par,
        tim_path=tim,
        output_dir=output,
        config=active.to_core_config(),
        clock_override_path=clock_override_path,
        resume=resume,
        baseline_par_path=baseline_par_path,
        baseline_tim_path=baseline_tim_path,
        solver_architecture=SOLVER_ARCHITECTURE,
    )
    if shared_result.model_name != "ISOLATED":
        raise RuntimeError("The shared phase-lattice solver changed the model family.")
    _record_isolated_input_policy(
        output=output,
        source_par=par,
        effective_par=effective_par,
        fit_parameter_source=fit_parameter_source,
    )
    return _convert_result(shared_result)


__all__ = [
    "AUTOMATIC_ISOLATED_FIT_PARAMETERS",
    "IsolatedAnnealingConfig",
    "IsolatedCellSearchConfig",
    "IsolatedPulsarSolverConfig",
    "IsolatedPulsarSolverResult",
    "IsolatedSessionUpdateConfig",
    "_write_automatic_isolated_fit_par",
    "solve_isolated_pulsar",
]
