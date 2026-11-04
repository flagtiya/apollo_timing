from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from apollo_timing.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
)


@dataclass(frozen=True)
class ProjectedPhaseBasis:
    """One normalized physical phase basis after session projection."""

    name: str
    coordinate_scale: float
    coordinate_unit: str
    shape_jacobian: np.ndarray
    relative_session_jacobian: np.ndarray


@dataclass(frozen=True)
class PhaseBasisModelScore:
    """Collapsed continuous/integer posterior score for one basis model."""

    basis_names: tuple[str, ...]
    log_likelihood: float
    log_bayes_factor: float
    shape_quadratic: float
    integer_quadratic: float
    strict_score: float
    posterior_mean: np.ndarray
    integer_offsets: tuple[int, ...]
    retained_lattice_candidates: int
    lattice_log_mass_increment: float
    lattice_mass_converged: bool
    lattice_visited_nodes: int


@dataclass(frozen=True)
class PhaseBasisReleaseDecision:
    selected_basis_names: tuple[str, ...]
    selected_score: PhaseBasisModelScore
    base_score: PhaseBasisModelScore
    reason: str


@dataclass(frozen=True)
class PhaseBasisAdequacyCertificate:
    """Model-level interpretation of a collapsed phase-basis comparison."""

    base_strict_score: float
    selected_basis_names: tuple[str, ...]
    selected_strict_score: float
    selected_log_bayes_factor: float
    strict_threshold: float
    minimum_log_bayes_factor: float
    strict_base_model_rejected: bool
    expanded_model_supported: bool
    expanded_model_reaches_threshold: bool
    smallest_supported_block_selected: bool
    interpretation: str


@dataclass(frozen=True)
class ExtendedIntegerClosureConfig:
    maximum_function_evaluations: int = 600
    function_tolerance: float = 1.0e-12
    parameter_tolerance: float = 1.0e-12
    gradient_tolerance: float = 1.0e-11


@dataclass(frozen=True)
class ExtendedIntegerClosureResult:
    state: np.ndarray
    integer_offsets: tuple[int, ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    function_evaluations: int
    converged: bool
    status: int
    message: str


def _session_indices(session: object) -> np.ndarray:
    values = getattr(session, "toa_indices", session)
    indices = np.asarray(values, dtype=int)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Every phase-basis session must contain TOA indices.")
    return indices


def project_raw_phase_basis(
    raw_phase_jacobian: Sequence[float],
    phase_sigmas: Sequence[float],
    sessions: Sequence[object],
    *,
    name: str,
    coordinate_scale: float,
    coordinate_unit: str,
) -> ProjectedPhaseBasis:
    """Project a raw phase derivative into shape and session-anchor blocks.

    Use the APOLLO likelihood's weighted intercept projection so a proposed
    physical parameter can be compared within the existing collapsed
    continuous/integer posterior before selecting an integer path.
    """

    raw = np.asarray(raw_phase_jacobian, dtype=float)
    sigmas = np.asarray(phase_sigmas, dtype=float)
    session_tuple = tuple(sessions)
    if raw.ndim != 1 or sigmas.shape != raw.shape or raw.size == 0:
        raise ValueError("Raw phase derivatives and uncertainties must align.")
    if not np.all(np.isfinite(raw)) or not np.all(
        np.isfinite(sigmas) & (sigmas > 0.0)
    ):
        raise ValueError("Phase-basis inputs must be finite with positive errors.")
    scale = float(coordinate_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("A phase-basis coordinate scale must be positive.")
    if not session_tuple:
        raise ValueError("A phase basis requires at least one session.")

    projected = np.empty_like(raw)
    anchors: list[float] = []
    covered: list[int] = []
    for session in session_tuple:
        indices = _session_indices(session)
        covered.extend(int(value) for value in indices)
        weights = 1.0 / np.square(sigmas[indices])
        anchor = float(np.sum(weights * raw[indices]) / np.sum(weights))
        projected[indices] = (raw[indices] - anchor) / sigmas[indices]
        anchors.append(anchor)
    if tuple(covered) != tuple(range(raw.size)):
        raise ValueError("Phase-basis sessions must cover every TOA exactly once.")
    anchor_array = np.asarray(anchors, dtype=float)
    relative = anchor_array[1:] - anchor_array[0]
    return ProjectedPhaseBasis(
        name=str(name).upper(),
        coordinate_scale=scale,
        coordinate_unit=str(coordinate_unit),
        shape_jacobian=projected,
        relative_session_jacobian=relative,
    )


def augment_linearization_with_phase_bases(
    linearization: SpinOrbitLinearization,
    bases: Sequence[ProjectedPhaseBasis],
) -> SpinOrbitLinearization:
    """Append physical phase bases to one APOLLO linearized posterior."""

    basis_tuple = tuple(bases)
    if not basis_tuple:
        return linearization
    shape = np.asarray(linearization.shape_jacobian, dtype=float)
    relative = np.asarray(linearization.relative_session_jacobian, dtype=float)
    if len({basis.name for basis in basis_tuple}) != len(basis_tuple):
        raise ValueError("Phase-basis names must be unique within one model.")
    for basis in basis_tuple:
        if basis.shape_jacobian.shape != (shape.shape[0],):
            raise ValueError(f"Shape basis {basis.name} does not align with the TOAs.")
        if basis.relative_session_jacobian.shape != (relative.shape[0],):
            raise ValueError(
                f"Anchor basis {basis.name} does not align with the sessions."
            )
    return replace(
        linearization,
        spin_reference=np.concatenate(
            (
                np.asarray(linearization.spin_reference, dtype=float),
                np.zeros(len(basis_tuple), dtype=float),
            )
        ),
        shape_jacobian=np.column_stack(
            (shape, *(basis.shape_jacobian for basis in basis_tuple))
        ),
        relative_session_jacobian=np.column_stack(
            (
                relative,
                *(basis.relative_session_jacobian for basis in basis_tuple),
            )
        ),
    )


def score_phase_basis_portfolio(
    linearization: SpinOrbitLinearization,
    *,
    available_bases: Mapping[str, ProjectedPhaseBasis],
    candidate_models: Sequence[Sequence[str]],
    base_prior_covariance: Sequence[Sequence[float]],
    basis_prior_variance: Mapping[str, float] | None = None,
    lattice_config: SpinIntegerMarginalConfig | None = None,
) -> tuple[PhaseBasisModelScore, ...]:
    """Compare physical basis models after marginalizing every integer path."""

    normalized_bases = {str(name).upper(): basis for name, basis in available_bases.items()}
    base_covariance = np.asarray(base_prior_covariance, dtype=float)
    base_dimension = int(np.asarray(linearization.shape_jacobian).shape[1])
    if base_covariance.shape != (base_dimension, base_dimension):
        raise ValueError("Base prior covariance must match the linearization dimension.")
    prior_variance = {
        str(name).upper(): float(value)
        for name, value in (basis_prior_variance or {}).items()
    }
    models = [tuple()]
    models.extend(
        tuple(str(name).upper() for name in candidate) for candidate in candidate_models
    )
    unique_models: list[tuple[str, ...]] = []
    for model in models:
        if model not in unique_models:
            unique_models.append(model)

    pending: list[tuple[tuple[str, ...], object]] = []
    active_config = lattice_config or SpinIntegerMarginalConfig()
    for names in unique_models:
        missing = [name for name in names if name not in normalized_bases]
        if missing:
            raise ValueError("Unknown phase bases: " + ", ".join(missing))
        if len(set(names)) != len(names):
            raise ValueError("A candidate model cannot repeat a phase basis.")
        augmented = augment_linearization_with_phase_bases(
            linearization,
            [normalized_bases[name] for name in names],
        )
        covariance = np.zeros(
            (base_dimension + len(names), base_dimension + len(names)),
            dtype=float,
        )
        covariance[:base_dimension, :base_dimension] = base_covariance
        for index, name in enumerate(names, start=base_dimension):
            variance = prior_variance.get(name, 1.0)
            if not np.isfinite(variance) or variance <= 0.0:
                raise ValueError(f"Prior variance for {name} must be positive.")
            covariance[index, index] = variance
        marginal = marginalize_spin_and_integer_paths(
            augmented,
            spin_prior_mean=np.zeros(base_dimension + len(names), dtype=float),
            spin_prior_covariance=covariance,
            config=active_config,
        )
        if not marginal.path_modes:
            raise RuntimeError("A phase-basis model returned no integer path.")
        pending.append((names, marginal))

    base_log_likelihood = float(pending[0][1].log_likelihood)
    scores: list[PhaseBasisModelScore] = []
    for names, marginal in pending:
        path = marginal.path_modes[0]
        shape_quadratic = float(marginal.shape_quadratic)
        integer_quadratic = float(path.squared_mahalanobis)
        scores.append(
            PhaseBasisModelScore(
                basis_names=names,
                log_likelihood=float(marginal.log_likelihood),
                log_bayes_factor=float(marginal.log_likelihood - base_log_likelihood),
                shape_quadratic=shape_quadratic,
                integer_quadratic=integer_quadratic,
                strict_score=shape_quadratic + integer_quadratic,
                posterior_mean=np.asarray(path.spin_posterior_mean, dtype=float),
                integer_offsets=tuple(int(value) for value in path.integer_offsets),
                retained_lattice_candidates=int(
                    marginal.retained_lattice_candidates
                ),
                lattice_log_mass_increment=float(
                    marginal.lattice_log_mass_increment
                ),
                lattice_mass_converged=bool(marginal.lattice_mass_converged),
                lattice_visited_nodes=int(marginal.lattice_visited_nodes),
            )
        )
    return tuple(scores)


def select_phase_basis_release(
    scores: Sequence[PhaseBasisModelScore],
    *,
    target_strict_score: float = 200.0,
    minimum_log_bayes_factor: float = 5.0,
    minimum_strict_improvement: float = 100.0,
) -> PhaseBasisReleaseDecision:
    """Release the smallest basis block with decisive posterior support."""

    score_tuple = tuple(scores)
    base = next((score for score in score_tuple if not score.basis_names), None)
    if base is None:
        raise ValueError("A phase-basis portfolio must contain the base model.")
    eligible = [
        score
        for score in score_tuple
        if score.basis_names
        and score.log_bayes_factor >= float(minimum_log_bayes_factor)
        and base.strict_score - score.strict_score
        >= float(minimum_strict_improvement)
    ]
    sufficient = [
        score for score in eligible if score.strict_score <= float(target_strict_score)
    ]
    if sufficient:
        selected = min(
            sufficient,
            key=lambda score: (
                len(score.basis_names),
                score.strict_score,
                -score.log_likelihood,
                score.basis_names,
            ),
        )
        reason = "smallest_decisive_basis_reaching_target"
    elif eligible:
        selected = max(
            eligible,
            key=lambda score: (
                score.log_likelihood,
                -score.strict_score,
                -len(score.basis_names),
            ),
        )
        reason = "best_decisive_basis_without_target_closure"
    else:
        selected = base
        reason = "no_decisive_missing_physical_basis"
    return PhaseBasisReleaseDecision(
        selected_basis_names=selected.basis_names,
        selected_score=selected,
        base_score=base,
        reason=reason,
    )


def certify_phase_basis_adequacy(
    scores: Sequence[PhaseBasisModelScore],
    decision: PhaseBasisReleaseDecision,
    *,
    strict_threshold: float = 200.0,
    minimum_log_bayes_factor: float = 5.0,
) -> PhaseBasisAdequacyCertificate:
    """Separate failure of the base model from success of an expanded model.

    This certificate prevents an accepted expanded fit from being reported as
    evidence that the original fit-parameter contract was sufficient.
    """

    score_tuple = tuple(scores)
    base = next((score for score in score_tuple if not score.basis_names), None)
    if base is None:
        raise ValueError("A model-adequacy certificate requires a base score.")
    selected = next(
        (
            score
            for score in score_tuple
            if score.basis_names == decision.selected_basis_names
        ),
        None,
    )
    if selected is None or selected != decision.selected_score:
        raise ValueError("The release decision does not match the scored portfolio.")

    threshold = float(strict_threshold)
    minimum_bf = float(minimum_log_bayes_factor)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("The strict adequacy threshold must be positive.")
    if not np.isfinite(minimum_bf):
        raise ValueError("The minimum log Bayes factor must be finite.")

    supported = bool(
        selected.basis_names
        and selected.log_bayes_factor >= minimum_bf
        and selected.strict_score < base.strict_score
    )
    reaches = bool(supported and selected.strict_score <= threshold)
    supported_reaching = [
        score
        for score in score_tuple
        if score.basis_names
        and score.log_bayes_factor >= minimum_bf
        and score.strict_score <= threshold
        and score.strict_score < base.strict_score
    ]
    smallest = bool(
        supported_reaching
        and len(selected.basis_names)
        == min(len(score.basis_names) for score in supported_reaching)
    )
    base_rejected = bool(base.strict_score > threshold)
    if base_rejected and reaches:
        interpretation = "base_rejected_minimal_physical_expansion_supported"
    elif base_rejected:
        interpretation = "base_rejected_no_tested_expansion_reaches_threshold"
    elif selected.basis_names and supported:
        interpretation = "base_adequate_expansion_supported_but_not_required"
    else:
        interpretation = "base_model_adequate"
    return PhaseBasisAdequacyCertificate(
        base_strict_score=float(base.strict_score),
        selected_basis_names=selected.basis_names,
        selected_strict_score=float(selected.strict_score),
        selected_log_bayes_factor=float(selected.log_bayes_factor),
        strict_threshold=threshold,
        minimum_log_bayes_factor=minimum_bf,
        strict_base_model_rejected=base_rejected,
        expanded_model_supported=supported,
        expanded_model_reaches_threshold=reaches,
        smallest_supported_block_selected=smallest,
        interpretation=interpretation,
    )


def refine_extended_integer_path(
    *,
    evaluate_profile: Callable[[np.ndarray], object],
    initial_state: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    integer_offsets: Sequence[int],
    config: ExtendedIntegerClosureConfig | None = None,
) -> ExtendedIntegerClosureResult:
    """Jointly refine continuous coordinates for one marginalized path mode."""

    active = config or ExtendedIntegerClosureConfig()
    start = np.asarray(initial_state, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    path = np.asarray(integer_offsets, dtype=float)
    if (
        start.ndim != 1
        or lower.shape != start.shape
        or upper.shape != start.shape
        or np.any(lower >= upper)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
    ):
        raise ValueError("Extended closure needs finite aligned state bounds.")
    state = np.clip(start, lower, upper)
    reference_profile = evaluate_profile(state)
    shape_size = np.asarray(
        reference_profile.normalized_local_residuals,
        dtype=float,
    ).size
    session_count = np.asarray(reference_profile.session_intercepts).size
    if path.shape != (session_count - 1,):
        raise ValueError("The integer path must contain one offset per non-anchor session.")
    failure = np.full(shape_size + path.size, 1.0e15, dtype=float)

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            profile = evaluate_profile(values)
            shape = np.asarray(profile.normalized_local_residuals, dtype=float)
            anchors = np.asarray(profile.session_intercepts, dtype=float)
            variances = np.asarray(profile.session_intercept_variances, dtype=float)
            covariance = np.diag(variances[1:]) + float(variances[0])
            phase = np.linalg.solve(
                np.linalg.cholesky(covariance),
                anchors[1:] - anchors[0] - path,
            )
            result = np.concatenate((shape, phase))
            if result.shape == failure.shape and np.all(np.isfinite(result)):
                return result
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        return failure

    fit = least_squares(
        residual,
        state,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=max(int(active.maximum_function_evaluations), 1),
        ftol=max(float(active.function_tolerance), np.finfo(float).eps),
        xtol=max(float(active.parameter_tolerance), np.finfo(float).eps),
        gtol=max(float(active.gradient_tolerance), np.finfo(float).eps),
    )
    final_state = np.asarray(fit.x, dtype=float)
    profile = evaluate_profile(final_state)
    shape = np.asarray(profile.normalized_local_residuals, dtype=float)
    anchors = np.asarray(profile.session_intercepts, dtype=float)
    variances = np.asarray(profile.session_intercept_variances, dtype=float)
    covariance = np.diag(variances[1:]) + float(variances[0])
    phase = np.linalg.solve(
        np.linalg.cholesky(covariance),
        anchors[1:] - anchors[0] - path,
    )
    shape_chi2 = float(shape @ shape)
    phase_chi2 = float(phase @ phase)
    return ExtendedIntegerClosureResult(
        state=final_state,
        integer_offsets=tuple(int(value) for value in path),
        shape_chi2=shape_chi2,
        phase_chi2=phase_chi2,
        total_chi2=shape_chi2 + phase_chi2,
        function_evaluations=int(fit.nfev),
        converged=bool(fit.success and np.isfinite(shape_chi2 + phase_chi2)),
        status=int(fit.status),
        message=str(fit.message),
    )


__all__ = [
    "ExtendedIntegerClosureConfig",
    "ExtendedIntegerClosureResult",
    "PhaseBasisModelScore",
    "PhaseBasisAdequacyCertificate",
    "PhaseBasisReleaseDecision",
    "ProjectedPhaseBasis",
    "augment_linearization_with_phase_bases",
    "certify_phase_basis_adequacy",
    "project_raw_phase_basis",
    "refine_extended_integer_path",
    "score_phase_basis_portfolio",
    "select_phase_basis_release",
]
