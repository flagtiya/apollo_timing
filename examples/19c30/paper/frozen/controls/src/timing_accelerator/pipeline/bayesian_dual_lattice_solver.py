from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.residuals import Residuals

from timing_accelerator.io.artifacts import sha256_file as _sha256, write_json_atomic
from timing_accelerator.backend.pint_certification import (
    attach_session_pulse_numbers as _attach_session_pulse_numbers,
    disk_reload_metrics as _disk_reload_metrics,
)
from timing_accelerator.io.solution_artifacts import write_solution_tim
from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _update_fit_metadata,
)
from timing_accelerator.models.serialization import to_json_ready
from timing_accelerator.inference.bayesian_dual_lattice_conditional import (
    PintEquivalentDDPhaseProjector,
)
from timing_accelerator.inference.bayesian_dual_lattice_enumerator import (
    DUAL_LATTICE_FIT_PARAMETERS,
    build_dual_lattice_search_context,
    normalized_bounds_for_cell,
    profile_fixed_local_pulse_component,
)
from timing_accelerator.inference.bayesian_phase_basis_release import (
    ExtendedIntegerClosureConfig,
    certify_phase_basis_adequacy,
    refine_extended_integer_path,
    score_phase_basis_portfolio,
    select_phase_basis_release,
)
from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
)
from timing_accelerator.inference.pint_phase_basis import (
    PROHIBITED_AUTOMATIC_PARAMETERS,
    PintPhaseBasisConfig,
    apply_phase_basis_coordinates,
    build_pint_phase_basis_portfolio,
    pint_absolute_phase_values,
    set_dd_base_parameters,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.reports.residual_plot import write_residual_plot


SOLVER_ARCHITECTURE = "apollo_bayesian_dual_lattice_phase_basis_solver"
SOLVER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConditionalModeCheckpoint:
    """Validated local posterior chart consumed by the release solver."""

    cell_id: str
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float


@dataclass(frozen=True)
class BayesianDualLatticeSolverConfig:
    maximum_cell_width: float = 0.25
    support_width_scale: float = 1.0
    minimum_f1_half_width: float = 1.0e-10
    finite_difference_step: float = 2.0e-5
    initial_lattice_candidates: int = 64
    maximum_lattice_candidates: int = 512
    lattice_log_mass_tolerance: float = 1.0e-6
    maximum_lattice_nodes: int = 5_000_000
    maximum_closure_evaluations: int = 600
    fit_iterations: int = 10
    strict_chi2_threshold: float = 200.0
    target_chi2_threshold: float = 100.0
    minimum_log_bayes_factor: float = 5.0
    minimum_strict_improvement: float = 100.0
    disk_reload_chi2_tolerance: float = 1.0e-2
    phase_basis: PintPhaseBasisConfig = field(default_factory=PintPhaseBasisConfig)

    def __post_init__(self) -> None:
        positive_floats = (
            self.maximum_cell_width,
            self.support_width_scale,
            self.minimum_f1_half_width,
            self.finite_difference_step,
            self.lattice_log_mass_tolerance,
            self.strict_chi2_threshold,
            self.target_chi2_threshold,
            self.minimum_log_bayes_factor,
            self.minimum_strict_improvement,
            self.disk_reload_chi2_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("Solver scales, tolerances, and thresholds must be positive.")
        positive_integers = (
            self.initial_lattice_candidates,
            self.maximum_lattice_candidates,
            self.maximum_lattice_nodes,
            self.maximum_closure_evaluations,
            self.fit_iterations,
        )
        if not all(int(value) > 0 for value in positive_integers):
            raise ValueError("Solver budgets must be positive integers.")
        if self.initial_lattice_candidates > self.maximum_lattice_candidates:
            raise ValueError("Initial lattice candidates cannot exceed the maximum.")
        if self.target_chi2_threshold > self.strict_chi2_threshold:
            raise ValueError("The target chi-square cannot exceed the acceptance threshold.")


@dataclass(frozen=True)
class BayesianDualLatticeSolverResult:
    output_dir: Path
    solution_par: Path
    solution_tim: Path
    residuals_csv: Path
    residuals_png: Path
    selected_phase_bases: tuple[str, ...]
    fit_parameters: tuple[str, ...]
    chi2: float
    disk_reload_chi2: float
    reduced_chi2: float
    weighted_rms_us: float
    accepted: bool
    target_reached: bool
    resumed: bool = False


def _atomic_json(path: Path, payload: Any) -> None:
    write_json_atomic(path, payload)


def load_conditional_mode_checkpoint(path: Path) -> ConditionalModeCheckpoint:
    """Load and validate the mode schema without assuming benchmark identity."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("best_mode"), Mapping
    ):
        raise ValueError("The mode checkpoint must contain a best_mode object.")
    mode = payload["best_mode"]
    normalized = np.asarray(mode.get("normalized_values"), dtype=float)
    physical = np.asarray(mode.get("parameter_values"), dtype=float)
    if normalized.shape != (8,) or not np.all(np.isfinite(normalized)):
        raise ValueError("The mode checkpoint needs eight finite normalized values.")
    if physical.shape != (8,) or not np.all(np.isfinite(physical)):
        raise ValueError("The mode checkpoint needs eight finite DD values.")
    raw_components = mode.get("local_integer_offsets")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("The mode checkpoint needs local integer components.")
    local_components: list[tuple[int, ...]] = []
    for component in raw_components:
        if not isinstance(component, list) or not component:
            raise ValueError("Every local integer component must be nonempty.")
        local_components.append(tuple(int(value) for value in component))
    cell_id = str(mode.get("cell_id", "")).strip()
    if not cell_id:
        raise ValueError("The mode checkpoint needs an orbital-cycle cell id.")
    shape_chi2 = float(mode.get("shape_chi2"))
    if not np.isfinite(shape_chi2) or shape_chi2 < 0.0:
        raise ValueError("The checkpoint shape chi-square must be finite and nonnegative.")
    return ConditionalModeCheckpoint(
        cell_id=cell_id,
        normalized_values=normalized,
        parameter_values=physical,
        local_integer_offsets=tuple(local_components),
        shape_chi2=shape_chi2,
    )


def _write_solution_tim(*, toas: object, path: Path) -> Path:
    return write_solution_tim(
        toas=toas, path=path,
        comment="APOLLO Bayesian dual-lattice certified integer path",
    )


def _copy_incumbent(output: Path, chi2: float) -> None:
    stem = f"chi2_{chi2:.9f}"
    incumbent = output / "incumbents"
    incumbent.mkdir(parents=True, exist_ok=True)
    for source, suffix in (
        (output / "solution.par", ".par"),
        (output / "solution.tim", ".tim"),
        (output / "residuals.csv", "_residuals.csv"),
        (output / "residuals.png", "_residuals.png"),
    ):
        shutil.copyfile(source, incumbent / f"{stem}{suffix}")


def _result_from_certification(
    output: Path,
    certification: Mapping[str, Any],
    *,
    resumed: bool,
) -> BayesianDualLatticeSolverResult:
    artifacts = certification["artifacts"]
    return BayesianDualLatticeSolverResult(
        output_dir=output,
        solution_par=Path(artifacts["par"]),
        solution_tim=Path(artifacts["tim"]),
        residuals_csv=Path(artifacts["residuals_csv"]),
        residuals_png=Path(artifacts["residuals_png"]),
        selected_phase_bases=tuple(certification["selected_phase_bases"]),
        fit_parameters=tuple(certification["fit_parameters"]),
        chi2=float(certification["chi2"]),
        disk_reload_chi2=float(certification["disk_reload_chi2"]),
        reduced_chi2=float(certification["reduced_chi2"]),
        weighted_rms_us=float(certification["weighted_rms_us"]),
        accepted=bool(certification["accepted"]),
        target_reached=bool(certification["target_reached"]),
        resumed=resumed,
    )


def _resume_accepted_result(
    *,
    output: Path,
    input_hashes: Mapping[str, str | None],
    config: BayesianDualLatticeSolverConfig,
) -> BayesianDualLatticeSolverResult | None:
    audit_path = output / "audit.json"
    certification_path = output / "certification.json"
    if not audit_path.is_file() or not certification_path.is_file():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    audited_hashes = {
        name: audit.get("inputs", {}).get(name) for name in input_hashes
    }
    if dict(input_hashes) != audited_hashes:
        raise RuntimeError("Resume inputs do not match the solver audit.")
    if audit.get("solver_config") != to_json_ready(config):
        raise RuntimeError("Resume configuration does not match the solver audit.")
    if bool(certification.get("accepted")):
        manifest_path = output / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Accepted resume output has no artifact manifest.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise RuntimeError("Accepted resume output has an invalid artifact manifest.")
        for name, record in artifacts.items():
            if not isinstance(record, Mapping):
                raise RuntimeError(f"Artifact manifest record is invalid: {name}")
            artifact_path = Path(str(record.get("path", "")))
            expected_hash = str(record.get("sha256", ""))
            if not artifact_path.is_file() or _sha256(artifact_path) != expected_hash:
                raise RuntimeError(f"Accepted resume artifact failed SHA-256: {name}")
        return _result_from_certification(output, certification, resumed=True)
    return None


def solve_bayesian_dual_lattice(
    *,
    par_path: Path,
    tim_path: Path,
    mode_path: Path,
    output_dir: Path,
    config: BayesianDualLatticeSolverConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> BayesianDualLatticeSolverResult:
    """Run model release, exact path closure, and bound PINT certification."""

    active = config or BayesianDualLatticeSolverConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    mode_file = Path(mode_path).resolve()
    clock_override = (
        Path(clock_override_path).resolve()
        if clock_override_path is not None
        else None
    )
    output = Path(output_dir).resolve()
    for label, path in (("par", par), ("tim", tim), ("mode", mode_file)):
        if not path.is_file():
            raise FileNotFoundError(f"APOLLO {label} input does not exist: {path}")
    if clock_override is not None and not clock_override.is_file():
        raise FileNotFoundError(
            f"APOLLO clock override does not exist: {clock_override}"
        )
    if output.exists() and not resume:
        raise FileExistsError(f"Output exists; pass resume=True: {output}")
    output.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "par_sha256": _sha256(par),
        "tim_sha256": _sha256(tim),
        "conditional_mode_sha256": _sha256(mode_file),
        "clock_override_sha256": (
            _sha256(clock_override) if clock_override is not None else None
        ),
    }
    if resume:
        accepted = _resume_accepted_result(
            output=output,
            input_hashes=input_hashes,
            config=active,
        )
        if accepted is not None:
            return accepted

    checkpoint = load_conditional_mode_checkpoint(mode_file)
    context = build_dual_lattice_search_context(
        par_path=par,
        tim_path=tim,
        maximum_cell_width=active.maximum_cell_width,
        support_width_scale=active.support_width_scale,
        minimum_f1_half_width=active.minimum_f1_half_width,
        clock_override_path=clock_override,
    )
    if len(checkpoint.local_integer_offsets) != len(context.evaluator.sessions):
        raise ValueError("Checkpoint local components do not match source sessions.")
    for component, session in zip(
        checkpoint.local_integer_offsets,
        context.evaluator.sessions,
        strict=True,
    ):
        if len(component) != len(session.toa_indices):
            raise ValueError("A checkpoint local component does not match its session.")
    cell_by_id = {cell.cell_id: cell for cell in context.cells}
    if checkpoint.cell_id not in cell_by_id:
        raise ValueError(
            "Checkpoint cell is outside the search tiling; check support and cell width."
        )

    projector = PintEquivalentDDPhaseProjector(
        evaluator=context.evaluator,
        coordinates=context.coordinates,
        support=context.support,
    )
    linearization = projector.linearize_full(
        checkpoint.normalized_values,
        local_integer_offsets=checkpoint.local_integer_offsets,
        finite_difference_step=active.finite_difference_step,
    )
    portfolio = build_pint_phase_basis_portfolio(
        projector=projector,
        parameter_values=checkpoint.parameter_values,
        config=active.phase_basis,
    )
    lattice_config = SpinIntegerMarginalConfig(
        initial_lattice_candidates=active.initial_lattice_candidates,
        maximum_lattice_candidates=active.maximum_lattice_candidates,
        lattice_log_mass_tolerance=active.lattice_log_mass_tolerance,
        maximum_lattice_nodes=active.maximum_lattice_nodes,
    )
    model_scores = score_phase_basis_portfolio(
        linearization,
        available_bases=portfolio.bases,
        candidate_models=portfolio.candidate_models,
        base_prior_covariance=np.eye(8),
        lattice_config=lattice_config,
    )
    decision = select_phase_basis_release(
        model_scores,
        target_strict_score=active.strict_chi2_threshold,
        minimum_log_bayes_factor=active.minimum_log_bayes_factor,
        minimum_strict_improvement=active.minimum_strict_improvement,
    )
    adequacy = certify_phase_basis_adequacy(
        model_scores,
        decision,
        strict_threshold=active.strict_chi2_threshold,
        minimum_log_bayes_factor=active.minimum_log_bayes_factor,
    )
    audit = {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "architecture": SOLVER_ARCHITECTURE,
        "scientific_contract": {
            "base_fit_parameters": DUAL_LATTICE_FIT_PARAMETERS,
            "candidate_physical_basis_blocks": portfolio.candidate_models,
            "prohibited_automatic_parameters": PROHIBITED_AUTOMATIC_PARAMETERS,
            "source_class_prior_policy": "forbidden",
            "external_historical_seed_policy": "forbidden",
            "manual_wrap_policy": "forbidden",
            "conditional_mode_role": "posterior_chart_not_certified_solution",
            "model_release_policy": "integer_marginal_evidence_minimum_sufficient_block",
            "evidence_claim_boundary": (
                "Bayes factors are conditional on the recorded normalized Gaussian priors "
                "and audited integer-lattice truncation."
            ),
            "scientific_chi2_threshold": active.strict_chi2_threshold,
            "target_chi2": active.target_chi2_threshold,
        },
        "inputs": {
            "par": str(par),
            "tim": str(tim),
            "conditional_mode": str(mode_file),
            "clock_override": (
                str(clock_override) if clock_override is not None else None
            ),
            **input_hashes,
            "checkpoint_embedded_source_hashes": False,
        },
        "solver_config": active,
        "observation_span_days": context.coordinates.observation_span_days,
        "reference_mjd": context.coordinates.reference_mjd,
        "session_count": len(context.evaluator.sessions),
        "toa_count": len(context.evaluator.toas.table),
        "cell_count": len(context.cells),
        "source_shape_chi2": checkpoint.shape_chi2,
        "phase_basis_coordinate_scales": {
            name: portfolio.bases[name].coordinate_scale for name in portfolio.bases
        },
        "phase_basis_coordinate_units": {
            name: portfolio.bases[name].coordinate_unit for name in portfolio.bases
        },
        "phase_basis_normalized_bounds": portfolio.normalized_bounds,
        "continuous_prior": {
            "base": "N(0, I_8) in the normalized local dual-lattice chart",
            "phase_bases": "N(0, 1) independently in each recorded normalized basis",
        },
        "integer_lattice": {
            "initial_candidates": active.initial_lattice_candidates,
            "maximum_candidates": active.maximum_lattice_candidates,
            "maximum_nodes": active.maximum_lattice_nodes,
            "log_mass_tolerance": active.lattice_log_mass_tolerance,
        },
        "model_scores": model_scores,
        "release_decision": decision,
        "model_adequacy": adequacy,
    }
    _atomic_json(output / "audit.json", audit)
    _atomic_json(output / "model_adequacy.json", adequacy)

    selected_names = tuple(decision.selected_basis_names)
    posterior = np.asarray(decision.selected_score.posterior_mean, dtype=float)
    start = np.concatenate(
        (checkpoint.normalized_values + posterior[:8], posterior[8:])
    )
    lower8, upper8 = normalized_bounds_for_cell(
        context.support,
        cell_by_id[checkpoint.cell_id],
    )
    selected_limits = np.asarray(
        [portfolio.normalized_bounds[name] for name in selected_names],
        dtype=float,
    )
    lower = np.concatenate((lower8, -selected_limits))
    upper = np.concatenate((upper8, selected_limits))
    model = context.evaluator.model
    toas = context.evaluator.toas

    def evaluate_profile(state: np.ndarray):
        slow, fast = context.support.denormalize(np.asarray(state[:8], dtype=float))
        physical = context.coordinates.combine_physical(slow, fast)
        set_dd_base_parameters(model, physical)
        apply_phase_basis_coordinates(
            model,
            basis_names=selected_names,
            normalized_values=np.asarray(state[8:], dtype=float),
            reference_values=portfolio.reference_values,
            coordinate_steps=portfolio.coordinate_steps,
        )
        raw = np.asarray(
            pint_absolute_phase_values(model, toas)
            - context.evaluator.base_pulse_numbers,
            dtype=float,
        )
        return profile_fixed_local_pulse_component(
            raw,
            projector.phase_sigmas,
            projector.sessions,
            checkpoint.local_integer_offsets,
        )

    closure = refine_extended_integer_path(
        evaluate_profile=evaluate_profile,
        initial_state=start,
        lower_bounds=lower,
        upper_bounds=upper,
        integer_offsets=decision.selected_score.integer_offsets,
        config=ExtendedIntegerClosureConfig(
            maximum_function_evaluations=active.maximum_closure_evaluations
        ),
    )
    _atomic_json(output / "closure.json", closure)

    final_state = np.asarray(closure.state, dtype=float)
    slow, fast = context.support.denormalize(final_state[:8])
    final_base = context.coordinates.combine_physical(slow, fast)
    set_dd_base_parameters(model, final_base)
    apply_phase_basis_coordinates(
        model,
        basis_names=selected_names,
        normalized_values=np.asarray(final_state[8:], dtype=float),
        reference_values=portfolio.reference_values,
        coordinate_steps=portfolio.coordinate_steps,
    )
    fit_parameters = (*DUAL_LATTICE_FIT_PARAMETERS, *selected_names)
    for name in model.params:
        parameter = model[name]
        if hasattr(parameter, "frozen"):
            parameter.frozen = name not in fit_parameters

    certified_toas = _attach_session_pulse_numbers(
        toas=toas,
        base_pulse_numbers=context.evaluator.base_pulse_numbers,
        sessions=context.evaluator.sessions,
        relative_integer_offsets=closure.integer_offsets,
    )
    fitter = WLSFitter(certified_toas, model, track_mode="use_pulse_numbers")
    fitter.fit_toas(maxiter=active.fit_iterations)
    fitted_model = fitter.model
    residuals = Residuals(
        certified_toas,
        fitted_model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    chi2 = float(residuals.chi2)
    reduced_chi2 = float(residuals.reduced_chi2)
    weighted_rms_us = float(residuals.rms_weighted().to_value(u.us))
    _update_fit_metadata(fitted_model, certified_toas, residuals, chi2)
    solution_par = write_pint_model_parfile_roundtrip_safe(
        model=fitted_model,
        path=output / "solution.par",
    )
    solution_tim = _write_solution_tim(
        toas=certified_toas,
        path=output / "solution.tim",
    )
    solution_csv, solution_png = write_residual_plot(
        solution_par,
        solution_tim,
        output,
        clock_override_path=clock_override,
        artifact_stem="solution",
    )
    residuals_csv = output / "residuals.csv"
    residuals_png = output / "residuals.png"
    shutil.copyfile(solution_csv, residuals_csv)
    shutil.copyfile(solution_png, residuals_png)
    disk_chi2, disk_reduced_chi2, disk_weighted_rms_us = _disk_reload_metrics(
        solution_par,
        solution_tim,
        clock_override_path=clock_override,
    )
    reload_delta = abs(disk_chi2 - chi2)
    accepted = bool(
        chi2 < active.strict_chi2_threshold
        and disk_chi2 < active.strict_chi2_threshold
        and reload_delta <= active.disk_reload_chi2_tolerance
    )
    target_reached = bool(
        chi2 < active.target_chi2_threshold
        and disk_chi2 < active.target_chi2_threshold
    )
    fitted_parameters = {
        name: float(fitted_model[name].value) for name in fit_parameters
    }
    certification = {
        "schema_version": SOLVER_SCHEMA_VERSION,
        "backend": "PINT",
        "track_mode": "use_pulse_numbers",
        "use_absolute_phase": True,
        "toa_count": len(certified_toas.table),
        "session_count": len(context.evaluator.sessions),
        "fit_parameters": fit_parameters,
        "selected_phase_bases": selected_names,
        "strict_base_model_rejected": adequacy.strict_base_model_rejected,
        "model_expansion_required": bool(selected_names),
        "model_adequacy_interpretation": adequacy.interpretation,
        "prohibited_automatic_parameters": PROHIBITED_AUTOMATIC_PARAMETERS,
        "unselected_candidate_parameters": tuple(
            name
            for name in ("F2", "RAJ", "DECJ")
            if name not in selected_names
        ),
        "integer_offsets": closure.integer_offsets,
        "fitted_parameters": fitted_parameters,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "weighted_rms_us": weighted_rms_us,
        "disk_reload_chi2": disk_chi2,
        "disk_reload_reduced_chi2": disk_reduced_chi2,
        "disk_reload_weighted_rms_us": disk_weighted_rms_us,
        "disk_reload_chi2_delta": reload_delta,
        "disk_reload_chi2_tolerance": active.disk_reload_chi2_tolerance,
        "scientific_threshold": active.strict_chi2_threshold,
        "target_threshold": active.target_chi2_threshold,
        "accepted": accepted,
        "target_reached": target_reached,
        "artifacts": {
            "par": str(solution_par),
            "tim": str(solution_tim),
            "residuals_csv": str(residuals_csv),
            "residuals_png": str(residuals_png),
        },
    }
    _atomic_json(output / "certification.json", certification)
    _atomic_json(
        output / "model_adequacy.json",
        {
            "posterior_certificate": adequacy,
            "exact_pint_certification": certification,
            "claim_boundary": (
                "Acceptance applies to the selected minimum-sufficient model; rejection of "
                "the base model is not rejection of the APOLLO solver."
            ),
        },
    )
    manifest_paths = (
        solution_par,
        solution_tim,
        residuals_csv,
        residuals_png,
        output / "audit.json",
        output / "closure.json",
        output / "certification.json",
        output / "model_adequacy.json",
    )
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": SOLVER_SCHEMA_VERSION,
            "architecture": SOLVER_ARCHITECTURE,
            "artifacts": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in manifest_paths
            },
        },
    )
    if not accepted:
        raise RuntimeError(
            "PINT certification failed: "
            f"chi2={chi2}, disk_chi2={disk_chi2}, reload_delta={reload_delta}"
        )
    _copy_incumbent(output, chi2)
    return _result_from_certification(output, certification, resumed=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run APOLLO Bayesian dual-lattice phase-basis release and PINT certification."
        )
    )
    parser.add_argument("--par", type=Path, required=True)
    parser.add_argument("--tim", type=Path, required=True)
    parser.add_argument("--mode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument("--maximum-cell-width", type=float, default=0.25)
    parser.add_argument("--support-width-scale", type=float, default=1.0)
    parser.add_argument("--minimum-f1-half-width", type=float, default=1.0e-10)
    parser.add_argument("--finite-difference-step", type=float, default=2.0e-5)
    parser.add_argument("--maximum-lattice-nodes", type=int, default=5_000_000)
    parser.add_argument("--maximum-closure-evaluations", type=int, default=600)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--f2-bound-hz-s2", type=float, default=5.0e-19)
    parser.add_argument("--astrometric-bound-arcsec", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BayesianDualLatticeSolverConfig(
        maximum_cell_width=args.maximum_cell_width,
        support_width_scale=args.support_width_scale,
        minimum_f1_half_width=args.minimum_f1_half_width,
        finite_difference_step=args.finite_difference_step,
        maximum_lattice_nodes=args.maximum_lattice_nodes,
        maximum_closure_evaluations=args.maximum_closure_evaluations,
        fit_iterations=args.fit_iterations,
        phase_basis=PintPhaseBasisConfig(
            f2_bound_hz_s2=args.f2_bound_hz_s2,
            astrometric_bound_arcsec=args.astrometric_bound_arcsec,
        ),
    )
    result = solve_bayesian_dual_lattice(
        par_path=args.par,
        tim_path=args.tim,
        mode_path=args.mode,
        output_dir=args.output,
        config=config,
        clock_override_path=args.clock_override,
        resume=args.resume,
    )
    status = "PINT-CERTIFIED-RESUME" if result.resumed else "PINT-CERTIFIED"
    print(
        status,
        "CHI2",
        result.chi2,
        "DISK_CHI2",
        result.disk_reload_chi2,
        "CHI2R",
        result.reduced_chi2,
        "WRMS_US",
        result.weighted_rms_us,
        "BASES",
        ",".join(result.selected_phase_bases) or "BASE",
        "OUTPUT",
        result.output_dir,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BayesianDualLatticeSolverConfig",
    "BayesianDualLatticeSolverResult",
    "ConditionalModeCheckpoint",
    "build_arg_parser",
    "load_conditional_mode_checkpoint",
    "main",
    "solve_bayesian_dual_lattice",
]
