from __future__ import annotations

from apollo_timing.inference._shared_helpers import (
    _normalized_values,
    _stable_inverse,
)

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from apollo_timing.inference.information_ordered_binary_search import (
    _orbital_alias_normalized_bounds,
    estimate_orbital_alias_count,
)
from apollo_timing.inference.integer_least_squares import (
    IntegerLeastSquaresResult,
    solve_integer_least_squares,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
    integer_session_offsets_to_gap_wraps,
)


@dataclass(frozen=True)
class GlobalLatticeTrackerConfig:
    """Numerical controls for the hard-source Bayesian lattice tracker."""

    finite_difference_step: float = 2.0e-3
    prior_precision: float = 1.0
    invariant_max_nfev: int = 120
    invariant_mode_count: int = 16
    invariant_mode_separation: float = 0.08
    lattice_candidates_per_mode: int = 8
    maximum_lattice_nodes: int = 2_000_000
    reacquisition_kappas: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)
    reacquisition_proposal_fraction: float = 0.4
    maximum_stage_proposals: int = 96
    maximum_stage_modes: int = 24
    maximum_modes_per_path: int = 2
    maximum_modes_per_root: int = 6
    mode_separation: float = 0.04
    fixed_path_max_nfev: int = 120
    phase_tempering_ladder: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
    initial_phase_information_ratio: float = 1.0
    maximum_phase_temperature_ratio: float = 10.0
    maximum_adaptive_phase_steps: int = 12
    sessions_per_stage: int = 1
    session_entropy_penalty: float = 1.0
    transport_retry_multiplier: float = 4.0
    transport_retry_modes: int = 4
    transport_objective_jump: float = 25.0
    transport_objective_ratio: float = 3.0
    constrain_invariant_orbital_alias: bool = True
    orbital_alias_oversampling: float = 1.0
    maximum_orbital_alias_count: int = 256


@dataclass(frozen=True)
class InvariantModeSeed:
    parameter_values: np.ndarray
    source_id: str


@dataclass(frozen=True)
class InvariantMode:
    mode_id: str
    source_id: str
    parameter_values: np.ndarray
    normalized_values: np.ndarray
    covariance: np.ndarray
    shape_chi2: float
    prior_chi2: float
    objective_chi2: float
    log_mass: float
    posterior_mass: float
    nfev: int
    success: bool


@dataclass(frozen=True)
class LatticeProposal:
    root_mode_id: str
    parent_mode_id: str | None
    source: str
    active_sessions: tuple[int, ...]
    integer_offsets: tuple[int, ...]
    initial_normalized_values: np.ndarray
    squared_mahalanobis: float
    log_predictive_density: float
    kappa: float
    initial_parameter_values: np.ndarray | None = None


@dataclass(frozen=True)
class GlobalLatticeMode:
    mode_id: str
    root_mode_id: str
    parent_mode_id: str | None
    proposal_source: str
    anchor_session: int
    active_sessions: tuple[int, ...]
    integer_offsets: tuple[int, ...]
    parameter_values: np.ndarray
    normalized_values: np.ndarray
    covariance: np.ndarray
    shape_chi2: float
    phase_chi2: float
    prior_chi2: float
    objective_chi2: float
    log_mass: float
    posterior_mass: float
    squared_mahalanobis: float
    nfev: int
    success: bool

    def chronological_integer_offsets(self, session_count: int) -> tuple[int, ...]:
        if len(self.active_sessions) != session_count - 1:
            raise ValueError("Chronological offsets require all sessions to be active.")
        integers = np.zeros(session_count, dtype=np.int64)
        for session, value in zip(
            self.active_sessions,
            self.integer_offsets,
            strict=True,
        ):
            integers[session] = int(value)
        integers -= integers[0]
        return tuple(int(value) for value in integers[1:])

    def chronological_wraps(self, session_count: int) -> tuple[int, ...]:
        return integer_session_offsets_to_gap_wraps(
            self.chronological_integer_offsets(session_count)
        )


@dataclass(frozen=True)
class LatticeStageSummary:
    stage_index: int
    active_sessions: tuple[int, ...]
    proposed_count: int
    refitted_count: int
    selected_count: int
    unique_proposed_paths: int
    unique_selected_paths: int
    selected_root_count: int
    transport_retry_count: int
    transport_retry_improved: bool
    minimum_lattice_distance: float
    best_shape_chi2: float
    best_phase_chi2: float
    best_objective_chi2: float
    failure_class: str | None


@dataclass(frozen=True)
class GlobalLatticeTrackerResult:
    anchor_session: int
    session_order: tuple[int, ...]
    invariant_modes: tuple[InvariantMode, ...]
    final_modes: tuple[GlobalLatticeMode, ...]
    stage_history: tuple[LatticeStageSummary, ...]
    stage_portfolios: tuple[tuple[GlobalLatticeMode, ...], ...]
    reached_all_sessions: bool
    failure_class: str | None


@dataclass(frozen=True)
class ProfileLinearization:
    normalized_values: np.ndarray
    profile: SessionShapeProfile
    shape_jacobian: np.ndarray
    offset_jacobian: np.ndarray


def _normalized_bounds(coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("The coordinate system must expose matching normalized bounds.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
        raise ValueError("Normalized search bounds must be finite and ordered.")
    return lower, upper


def _physical_values(coordinates: Any, normalized_values: Sequence[float]) -> np.ndarray:
    search_values = coordinates.denormalize(np.asarray(normalized_values, dtype=float))
    return np.asarray(coordinates.to_physical(search_values), dtype=float)






def _stable_covariance(matrix: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors * clipped) @ eigenvectors.T


def _laplace_log_mass(objective_chi2: float, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(_stable_covariance(covariance))
    if sign <= 0 or not np.isfinite(logdet):
        return -math.inf
    dimension = covariance.shape[0]
    return float(
        -0.5 * float(objective_chi2)
        + 0.5 * float(logdet)
        + 0.5 * dimension * math.log(2.0 * math.pi)
    )


def _normalize_mode_masses(modes: Sequence[Any]) -> tuple[Any, ...]:
    if not modes:
        return ()
    log_masses = np.asarray([float(mode.log_mass) for mode in modes], dtype=float)
    finite = np.isfinite(log_masses)
    if not np.any(finite):
        masses = np.full(len(modes), 1.0 / len(modes), dtype=float)
    else:
        log_norm = float(logsumexp(log_masses[finite]))
        masses = np.zeros(len(modes), dtype=float)
        masses[finite] = np.exp(log_masses[finite] - log_norm)
    return tuple(
        replace(mode, posterior_mass=float(mass))
        for mode, mass in zip(modes, masses, strict=True)
    )


def adaptive_phase_tempering_ladder(
    *,
    declared_ladder: Sequence[float],
    shape_chi2: float,
    phase_chi2: float,
    information_dimension: int,
    initial_information_ratio: float,
    maximum_temperature_ratio: float,
    maximum_adaptive_steps: int,
) -> tuple[float, ...]:
    """Start phase annealing where phase and non-phase information are comparable."""

    declared = tuple(
        sorted(
            {
                min(max(float(value), 1.0e-12), 1.0)
                for value in declared_ladder
                if np.isfinite(float(value)) and float(value) > 0.0
            }
            | {1.0}
        )
    )
    information_target = max(
        float(shape_chi2),
        float(information_dimension),
        1.0,
    ) * max(float(initial_information_ratio), 1.0e-12)
    initial_temperature = min(
        declared[0],
        information_target / max(float(phase_chi2), information_target),
    )
    maximum_ratio = max(float(maximum_temperature_ratio), 1.0)
    adaptive: list[float] = []
    temperature = max(initial_temperature, 1.0e-12)
    for _ in range(max(int(maximum_adaptive_steps), 1)):
        if temperature >= declared[0] * (1.0 - 1.0e-12):
            break
        adaptive.append(temperature)
        temperature = min(declared[0], temperature * maximum_ratio)
    return tuple(sorted({*adaptive, *declared}))


def _profile_shape_residuals(profile: SessionShapeProfile) -> np.ndarray:
    residuals = np.asarray(profile.normalized_residuals, dtype=float)
    if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
        raise ValueError("The invariant profile residuals must be finite and one-dimensional.")
    return residuals


def _relative_phase_terms(
    profile: SessionShapeProfile,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    sessions = np.asarray(active_sessions, dtype=int)
    if offsets.ndim != 1 or variances.shape != offsets.shape:
        raise ValueError("Session offsets and variances must have matching shapes.")
    if not 0 <= int(anchor_session) < offsets.size:
        raise ValueError("The anchor session is out of range.")
    if sessions.size == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    if np.any((sessions < 0) | (sessions >= offsets.size)) or int(anchor_session) in sessions:
        raise ValueError("Active phase sessions must be valid and exclude the anchor.")
    relative = offsets[sessions] - offsets[int(anchor_session)]
    covariance = np.diag(variances[sessions])
    covariance += float(variances[int(anchor_session)])
    return relative, _stable_covariance(covariance)


def fixed_path_residual_vector(
    profile: SessionShapeProfile,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    integer_offsets: Sequence[int],
    normalized_values: Sequence[float] | None = None,
    prior_precision: float = 0.0,
) -> np.ndarray:
    """Return the exact decomposed residual for one complete active path."""

    sessions = tuple(int(value) for value in active_sessions)
    integers = np.asarray(integer_offsets, dtype=float)
    if integers.shape != (len(sessions),):
        raise ValueError("Integer offsets must match all active phase sessions.")
    shape = _profile_shape_residuals(profile)
    relative, covariance = _relative_phase_terms(
        profile,
        anchor_session=anchor_session,
        active_sessions=sessions,
    )
    if relative.size:
        phase = np.linalg.solve(np.linalg.cholesky(covariance), relative - integers)
    else:
        phase = np.zeros(0, dtype=float)
    result = [shape, phase]
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        if normalized_values is None:
            raise ValueError("Normalized values are required when the prior is active.")
        result.append(np.sqrt(weight) * np.asarray(normalized_values, dtype=float))
    vector = np.concatenate(result)
    if not np.all(np.isfinite(vector)):
        raise ValueError("The fixed-path residual contains non-finite values.")
    return vector


def exact_fixed_path_chi2(
    profile: SessionShapeProfile,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    integer_offsets: Sequence[int],
) -> tuple[float, float, float]:
    shape = _profile_shape_residuals(profile)
    relative, covariance = _relative_phase_terms(
        profile,
        anchor_session=anchor_session,
        active_sessions=active_sessions,
    )
    integers = np.asarray(integer_offsets, dtype=float)
    if integers.shape != relative.shape:
        raise ValueError("Integer offsets must match all active phase sessions.")
    delta = relative - integers
    phase_chi2 = float(delta @ np.linalg.solve(covariance, delta)) if delta.size else 0.0
    shape_chi2 = float(shape @ shape)
    return shape_chi2, phase_chi2, shape_chi2 + phase_chi2


def finite_difference_profile(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    step: float,
) -> ProfileLinearization:
    center = np.asarray(normalized_values, dtype=float)
    lower, upper = _normalized_bounds(coordinates)
    if center.shape != lower.shape:
        raise ValueError("The normalized state does not match the coordinate dimension.")
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    shape = _profile_shape_residuals(profile)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    shape_jacobian = np.empty((shape.size, center.size), dtype=float)
    offset_jacobian = np.empty((offsets.size, center.size), dtype=float)
    displacement = max(float(step), 1.0e-7)
    for index in range(center.size):
        plus_values = center.copy()
        minus_values = center.copy()
        plus_values[index] = min(center[index] + displacement, upper[index])
        minus_values[index] = max(center[index] - displacement, lower[index])
        denominator = float(plus_values[index] - minus_values[index])
        if denominator <= 0.0:
            raise ValueError("A finite-difference coordinate is fixed at its bounds.")
        plus = evaluator.evaluate_values(_physical_values(coordinates, plus_values))
        minus = evaluator.evaluate_values(_physical_values(coordinates, minus_values))
        plus_shape = _profile_shape_residuals(plus)
        minus_shape = _profile_shape_residuals(minus)
        plus_offsets = np.asarray(plus.session_offsets, dtype=float)
        minus_offsets = np.asarray(minus.session_offsets, dtype=float)
        if plus_shape.shape != shape.shape or minus_shape.shape != shape.shape:
            raise ValueError("Finite-difference profiles changed shape dimension.")
        if plus_offsets.shape != offsets.shape or minus_offsets.shape != offsets.shape:
            raise ValueError("Finite-difference profiles changed session dimension.")
        shape_jacobian[:, index] = (plus_shape - minus_shape) / denominator
        offset_jacobian[:, index] = (plus_offsets - minus_offsets) / denominator
    return ProfileLinearization(
        normalized_values=center.copy(),
        profile=profile,
        shape_jacobian=shape_jacobian,
        offset_jacobian=offset_jacobian,
    )


def refine_invariant_mode_portfolio(
    *,
    evaluator: Any,
    coordinates: Any,
    seeds: Sequence[InvariantModeSeed],
    config: GlobalLatticeTrackerConfig | None = None,
) -> tuple[InvariantMode, ...]:
    """Refit Stage-A seeds against the complete integer-invariant target."""

    active = config or GlobalLatticeTrackerConfig()
    lower, upper = _normalized_bounds(coordinates)
    prior_precision = max(float(active.prior_precision), 0.0)
    alias_count = 0
    try:
        reference_profile = evaluator.evaluate_values(
            _physical_values(coordinates, np.zeros_like(lower))
        )
        epochs = np.asarray(reference_profile.session_epochs, dtype=float)
        epochs = epochs[np.isfinite(epochs)]
        if epochs.size >= 2:
            pb_index = coordinates.parameter_names.index("PB")
            alias_count = estimate_orbital_alias_count(
                observation_span_days=float(np.ptp(epochs)),
                pb_lower_days=float(coordinates.physical_lower[pb_index]),
                pb_upper_days=float(coordinates.physical_upper[pb_index]),
                oversampling=max(float(active.orbital_alias_oversampling), 1.0),
                maximum_count=max(int(active.maximum_orbital_alias_count), 1),
            )
    except Exception:
        alias_count = 0
    candidates: list[InvariantMode] = []
    for seed_index, seed in enumerate(seeds):
        try:
            local_lower = lower.copy()
            local_upper = upper.copy()
            if bool(active.constrain_invariant_orbital_alias) and alias_count > 0:
                pb_index = coordinates.parameter_names.index("PB")
                frequency_lower = 1.0 / float(coordinates.physical_upper[pb_index])
                frequency_upper = 1.0 / float(coordinates.physical_lower[pb_index])
                seed_frequency = 1.0 / float(seed.parameter_values[pb_index])
                if alias_count == 1 or frequency_upper <= frequency_lower:
                    alias_index = 0
                else:
                    alias_index = int(
                        np.rint(
                            (seed_frequency - frequency_lower)
                            / (frequency_upper - frequency_lower)
                            * (alias_count - 1)
                        )
                    )
                pb_lower, pb_upper = _orbital_alias_normalized_bounds(
                    coordinates,
                    alias_index=alias_index,
                    alias_count=alias_count,
                )
                local_lower[pb_index] = max(local_lower[pb_index], pb_lower)
                local_upper[pb_index] = min(local_upper[pb_index], pb_upper)
            start = np.clip(
                _normalized_values(coordinates, seed.parameter_values),
                local_lower,
                local_upper,
            )
            start_profile = evaluator.evaluate_values(_physical_values(coordinates, start))
            failure = np.full(
                _profile_shape_residuals(start_profile).size
                + (start.size if prior_precision > 0.0 else 0),
                1.0e12,
                dtype=float,
            )

            def residual_vector(values: np.ndarray) -> np.ndarray:
                try:
                    profile = evaluator.evaluate_values(_physical_values(coordinates, values))
                    parts = [_profile_shape_residuals(profile)]
                    if prior_precision > 0.0:
                        parts.append(np.sqrt(prior_precision) * values)
                    result = np.concatenate(parts)
                    return result if result.shape == failure.shape else failure
                except Exception:
                    return failure

            optimization = least_squares(
                residual_vector,
                start,
                bounds=(local_lower, local_upper),
                method="trf",
                jac="2-point",
                diff_step=max(float(active.finite_difference_step), 1.0e-7),
                max_nfev=max(int(active.invariant_max_nfev), 1),
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            normalized = np.asarray(optimization.x, dtype=float)
            profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
            shape = _profile_shape_residuals(profile)
            shape_chi2 = float(shape @ shape)
            prior_chi2 = float(prior_precision * (normalized @ normalized))
            objective = shape_chi2 + prior_chi2
            hessian = np.asarray(optimization.jac, dtype=float).T @ np.asarray(
                optimization.jac,
                dtype=float,
            )
            covariance = _stable_inverse(hessian)
            candidates.append(
                InvariantMode(
                    mode_id=f"inv_{seed_index:04d}",
                    source_id=str(seed.source_id),
                    parameter_values=_physical_values(coordinates, normalized),
                    normalized_values=normalized,
                    covariance=covariance,
                    shape_chi2=shape_chi2,
                    prior_chi2=prior_chi2,
                    objective_chi2=objective,
                    log_mass=_laplace_log_mass(objective, covariance),
                    posterior_mass=0.0,
                    nfev=int(optimization.nfev),
                    success=bool(optimization.success),
                )
            )
        except Exception:
            continue
    selected: list[InvariantMode] = []
    separation = max(float(active.invariant_mode_separation), 0.0)
    for candidate in sorted(candidates, key=lambda item: (item.objective_chi2, -item.log_mass)):
        if any(
            np.linalg.norm(candidate.normalized_values - existing.normalized_values)
            < separation
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max(int(active.invariant_mode_count), 1):
            break
    if not selected:
        raise RuntimeError("Stage-A invariant refinement retained no valid mode.")
    return _normalize_mode_masses(selected)


def _wrapped_mixture_variance(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> float:
    angles = 2.0 * math.pi * np.asarray(means, dtype=float)
    resultant = np.sum(weights * np.exp(1j * angles))
    center = math.atan2(float(resultant.imag), float(resultant.real)) / (2.0 * math.pi)
    deltas = (np.asarray(means, dtype=float) - center + 0.5) % 1.0 - 0.5
    return float(np.sum(weights * (np.asarray(variances, dtype=float) + deltas * deltas)))


def choose_anchor_and_session_order(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    finite_difference_step: float,
) -> tuple[int, tuple[int, ...], dict[str, Any]]:
    if not invariant_modes:
        raise ValueError("At least one invariant mode is required.")
    linearizations = [
        finite_difference_profile(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=mode.normalized_values,
            step=finite_difference_step,
        )
        for mode in invariant_modes
    ]
    session_count = int(linearizations[0].profile.session_count)
    if any(item.profile.session_count != session_count for item in linearizations):
        raise ValueError("Invariant modes disagree on the source-session count.")
    weights = np.asarray([mode.posterior_mass for mode in invariant_modes], dtype=float)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(invariant_modes), dtype=float)
    weights /= np.sum(weights)
    predictive: dict[tuple[int, int], float] = {}
    anchor_scores: list[float] = []
    for anchor in range(session_count):
        session_variances = []
        for session in range(session_count):
            if session == anchor:
                continue
            means = []
            variances = []
            for mode, linearization in zip(invariant_modes, linearizations, strict=True):
                profile = linearization.profile
                row = (
                    linearization.offset_jacobian[session]
                    - linearization.offset_jacobian[anchor]
                )
                noise = float(
                    profile.session_offset_variances[session]
                    + profile.session_offset_variances[anchor]
                )
                means.append(
                    float(profile.session_offsets[session] - profile.session_offsets[anchor])
                )
                variances.append(noise + float(row @ mode.covariance @ row))
            value = max(
                _wrapped_mixture_variance(
                    np.asarray(means),
                    np.asarray(variances),
                    weights,
                ),
                np.finfo(float).tiny,
            )
            predictive[(anchor, session)] = value
            session_variances.append(value)
        logs = np.log(np.asarray(session_variances, dtype=float))
        anchor_scores.append(float(np.median(logs) + 0.25 * np.max(logs)))
    anchor = int(np.argmin(anchor_scores))
    epochs = np.asarray(linearizations[0].profile.session_epochs, dtype=float)
    remaining = set(range(session_count)) - {anchor}
    order: list[int] = []
    left_count = 0
    right_count = 0
    while remaining:
        left = sorted(
            (session for session in remaining if epochs[session] < epochs[anchor]),
            key=lambda session: (predictive[(anchor, session)], session),
        )
        right = sorted(
            (session for session in remaining if epochs[session] >= epochs[anchor]),
            key=lambda session: (predictive[(anchor, session)], session),
        )
        if left and right and left_count > right_count:
            selected = right[0]
        elif left and right and right_count > left_count:
            selected = left[0]
        else:
            selected = min(
                remaining,
                key=lambda session: (predictive[(anchor, session)], session),
            )
        order.append(int(selected))
        remaining.remove(selected)
        if epochs[selected] < epochs[anchor]:
            left_count += 1
        else:
            right_count += 1
    return anchor, tuple(order), {
        "anchor_scores": tuple(float(value) for value in anchor_scores),
        "predictive_variances": {
            f"{left}:{right}": float(value)
            for (left, right), value in predictive.items()
        },
    }


def propose_complete_integer_paths(
    center: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    candidate_count: int,
    maximum_lattice_nodes: int = 2_000_000,
) -> IntegerLeastSquaresResult:
    """Enumerate complete correlated integer vectors; no prefix is fixed."""

    return solve_integer_least_squares(
        center,
        _stable_covariance(np.asarray(covariance, dtype=float)),
        candidate_count=max(int(candidate_count), 1),
        max_visited_nodes=max(int(maximum_lattice_nodes), 1),
    )


def rank_predictive_sessions(
    *,
    evaluator: Any,
    coordinates: Any,
    parent_modes: Sequence[GlobalLatticeMode],
    invariant_modes: Sequence[InvariantMode],
    anchor_session: int,
    remaining_sessions: Sequence[int],
    finite_difference_step: float,
    session_entropy_penalty: float = 1.0,
) -> tuple[int, ...]:
    """Rank unconnected sessions from the current mixture posterior.

    Recompute the ranking after each bridge stage. Low posterior entropy in a
    session's phase prediction, modulo the integer lattice, indicates an easier
    connection. This information-order policy imposes no chronological prior.
    """

    if not remaining_sessions:
        return ()
    states = tuple(parent_modes) if parent_modes else tuple(
        _initial_parent_modes(invariant_modes, anchor_session=anchor_session)
    )
    if not states:
        return tuple(int(value) for value in remaining_sessions)
    weights = np.asarray([mode.posterior_mass for mode in states], dtype=float)
    if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
        weights = np.ones(len(states), dtype=float)
    weights /= np.sum(weights)
    records: list[tuple[GlobalLatticeMode, float, ProfileLinearization]] = []
    for mode, weight in zip(states, weights, strict=True):
        try:
            records.append(
                (
                    mode,
                    float(weight),
                    finite_difference_profile(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        normalized_values=mode.normalized_values,
                        step=finite_difference_step,
                    ),
                )
            )
        except Exception:
            continue
    if not records:
        return tuple(int(value) for value in remaining_sessions)
    active_weights = np.asarray([record[1] for record in records], dtype=float)
    active_weights /= np.sum(active_weights)
    values: dict[int, float] = {}
    for session in remaining_sessions:
        means = []
        variances = []
        for mode, _, linearization in records:
            profile = linearization.profile
            row = (
                linearization.offset_jacobian[int(session)]
                - linearization.offset_jacobian[int(anchor_session)]
            )
            mean = float(
                profile.session_offsets[int(session)]
                - profile.session_offsets[int(anchor_session)]
            )
            noise = float(
                profile.session_offset_variances[int(session)]
                + profile.session_offset_variances[int(anchor_session)]
            )
            variance = noise + float(row @ mode.covariance @ row)
            means.append(mean)
            variances.append(variance)
        wrapped_variance = _wrapped_mixture_variance(
            np.asarray(means, dtype=float),
            np.asarray(variances, dtype=float),
            active_weights,
        )
        information_gains = []
        for mode, _, linearization in records:
            row = (
                linearization.offset_jacobian[int(session)]
                - linearization.offset_jacobian[int(anchor_session)]
            )
            profile = linearization.profile
            noise = float(
                profile.session_offset_variances[int(session)]
                + profile.session_offset_variances[int(anchor_session)]
            )
            precision_update = np.outer(row, row) / max(noise, 1.0e-18)
            sign, logdet = np.linalg.slogdet(
                np.eye(mode.covariance.shape[0])
                + mode.covariance @ precision_update
            )
            information_gains.append(
                0.5 * float(logdet) if sign > 0 and np.isfinite(logdet) else 0.0
            )
        expected_information_gain = float(
            np.sum(active_weights * np.asarray(information_gains, dtype=float))
        )
        values[int(session)] = expected_information_gain - 0.5 * max(
            float(session_entropy_penalty), 0.0
        ) * math.log1p(max(wrapped_variance, 0.0))
    return tuple(
        sorted(
            (int(value) for value in remaining_sessions),
            key=lambda session: (-values[session], session),
        )
    )


def _phase_geometry(
    linearization: ProfileLinearization,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sessions = np.asarray(active_sessions, dtype=int)
    relative, noise = _relative_phase_terms(
        linearization.profile,
        anchor_session=anchor_session,
        active_sessions=sessions,
    )
    jacobian = (
        linearization.offset_jacobian[sessions]
        - linearization.offset_jacobian[int(anchor_session)]
    )
    return relative, jacobian, noise


def _conditional_gaussian(
    linearization: ProfileLinearization,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    integer_offsets: Sequence[int] | None,
    prior_precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(linearization.normalized_values, dtype=float)
    shape = _profile_shape_residuals(linearization.profile)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    hessian = shape_jacobian.T @ shape_jacobian
    gradient = shape_jacobian.T @ shape
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        hessian += weight * np.eye(z.size)
        gradient += weight * z
    if active_sessions:
        if integer_offsets is None:
            raise ValueError("An integer path is required for active phase sessions.")
        relative, phase_jacobian, noise = _phase_geometry(
            linearization,
            anchor_session=anchor_session,
            active_sessions=active_sessions,
        )
        integers = np.asarray(integer_offsets, dtype=float)
        if integers.shape != relative.shape:
            raise ValueError("The integer path does not match the active sessions.")
        noise_precision = _stable_inverse(noise)
        hessian += phase_jacobian.T @ noise_precision @ phase_jacobian
        gradient += phase_jacobian.T @ noise_precision @ (relative - integers)
    covariance = _stable_inverse(hessian)
    mean = z - covariance @ gradient
    return mean, covariance


def _predictive_lattice(
    linearization: ProfileLinearization,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    conditional_mean: np.ndarray,
    conditional_covariance: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, np.ndarray]:
    relative, jacobian, noise = _phase_geometry(
        linearization,
        anchor_session=anchor_session,
        active_sessions=active_sessions,
    )
    delta = np.asarray(conditional_mean, dtype=float) - linearization.normalized_values
    center = relative + jacobian @ delta
    covariance = noise + (float(kappa) ** 2) * (
        jacobian @ conditional_covariance @ jacobian.T
    )
    return center, _stable_covariance(covariance)


def _proposal_initial_state(
    linearization: ProfileLinearization,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    integer_offsets: Sequence[int],
    prior_precision: float,
    conditional_mean: np.ndarray,
    conditional_covariance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    relative, phase_jacobian, noise = _phase_geometry(
        linearization,
        anchor_session=anchor_session,
        active_sessions=active_sessions,
    )
    if not active_sessions:
        return np.clip(np.asarray(conditional_mean, dtype=float), lower, upper)
    integers = np.asarray(integer_offsets, dtype=float)
    delta_from_reference = (
        np.asarray(conditional_mean, dtype=float)
        - np.asarray(linearization.normalized_values, dtype=float)
    )
    predictive_mean = relative + phase_jacobian @ delta_from_reference
    predictive_covariance = noise + phase_jacobian @ np.asarray(
        conditional_covariance,
        dtype=float,
    ) @ phase_jacobian.T
    innovation = integers - predictive_mean
    conditioned = np.asarray(conditional_mean, dtype=float) + np.asarray(
        conditional_covariance,
        dtype=float,
    ) @ phase_jacobian.T @ _stable_inverse(predictive_covariance) @ innovation
    return np.clip(conditioned, lower, upper)


def generate_stage_lattice_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    anchor_session: int,
    active_sessions: Sequence[int],
    parent_modes: Sequence[GlobalLatticeMode],
    invariant_modes: Sequence[InvariantMode],
    config: GlobalLatticeTrackerConfig | None = None,
) -> tuple[LatticeProposal, ...]:
    """Generate full-path local and reacquisition proposals for one stage."""

    active = config or GlobalLatticeTrackerConfig()
    sessions = tuple(int(value) for value in active_sessions)
    lower, upper = _normalized_bounds(coordinates)
    raw: list[LatticeProposal] = []

    def add_lattice(
        *,
        root_mode_id: str,
        parent_mode_id: str | None,
        source: str,
        linearization: ProfileLinearization,
        mean: np.ndarray,
        covariance: np.ndarray,
        kappa: float,
    ) -> None:
        center, integer_covariance = _predictive_lattice(
            linearization,
            anchor_session=anchor_session,
            active_sessions=sessions,
            conditional_mean=mean,
            conditional_covariance=covariance,
            kappa=kappa,
        )
        result = propose_complete_integer_paths(
            center,
            integer_covariance,
            candidate_count=active.lattice_candidates_per_mode,
            maximum_lattice_nodes=active.maximum_lattice_nodes,
        )
        sign, logdet = np.linalg.slogdet(integer_covariance)
        log_normalization = (
            0.5 * len(center) * math.log(2.0 * math.pi) + 0.5 * logdet
            if sign > 0 and np.isfinite(logdet)
            else math.inf
        )
        for item in result.candidates:
            path = tuple(int(value) for value in item.integer_vector)
            effective_covariance = (float(kappa) ** 2) * covariance
            initial = _proposal_initial_state(
                linearization,
                anchor_session=anchor_session,
                active_sessions=sessions,
                integer_offsets=path,
                prior_precision=active.prior_precision,
                conditional_mean=mean,
                conditional_covariance=effective_covariance,
                lower=lower,
                upper=upper,
            )
            raw.append(
                LatticeProposal(
                    root_mode_id=root_mode_id,
                    parent_mode_id=parent_mode_id,
                    source=source,
                    active_sessions=sessions,
                    integer_offsets=path,
                    initial_normalized_values=initial,
                    squared_mahalanobis=float(item.squared_mahalanobis),
                    log_predictive_density=float(
                        -0.5 * float(item.squared_mahalanobis) - log_normalization
                    ),
                    kappa=float(kappa),
                )
            )

    for parent in parent_modes:
        try:
            linearization = finite_difference_profile(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=parent.normalized_values,
                step=active.finite_difference_step,
            )
            mean, covariance = _conditional_gaussian(
                linearization,
                anchor_session=anchor_session,
                active_sessions=parent.active_sessions,
                integer_offsets=parent.integer_offsets,
                prior_precision=active.prior_precision,
            )
            add_lattice(
                root_mode_id=parent.root_mode_id,
                parent_mode_id=parent.mode_id,
                source="predictive",
                linearization=linearization,
                mean=mean,
                covariance=covariance,
                kappa=1.0,
            )
            inherited = {session: value for session, value in zip(
                parent.active_sessions,
                parent.integer_offsets,
                strict=True,
            )}
            center, _ = _predictive_lattice(
                linearization,
                anchor_session=anchor_session,
                active_sessions=sessions,
                conditional_mean=mean,
                conditional_covariance=covariance,
                kappa=1.0,
            )
            control_path = tuple(
                int(inherited.get(session, int(np.rint(center[index]))))
                for index, session in enumerate(sessions)
            )
            raw.append(
                LatticeProposal(
                    root_mode_id=parent.root_mode_id,
                    parent_mode_id=parent.mode_id,
                    source="inherited-control",
                    active_sessions=sessions,
                    integer_offsets=control_path,
                    initial_normalized_values=_proposal_initial_state(
                        linearization,
                        anchor_session=anchor_session,
                        active_sessions=sessions,
                        integer_offsets=control_path,
                        prior_precision=active.prior_precision,
                        conditional_mean=mean,
                        conditional_covariance=covariance,
                        lower=lower,
                        upper=upper,
                    ),
                    squared_mahalanobis=math.inf,
                    log_predictive_density=-math.inf,
                    kappa=1.0,
                )
            )
        except Exception:
            continue

    kappas = tuple(
        sorted(
            {
                max(float(value), 1.0)
                for value in active.reacquisition_kappas
                if np.isfinite(float(value)) and float(value) > 0.0
            }
        )
    )
    for root in invariant_modes:
        try:
            linearization = finite_difference_profile(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=root.normalized_values,
                step=active.finite_difference_step,
            )
            mean, covariance = _conditional_gaussian(
                linearization,
                anchor_session=anchor_session,
                active_sessions=(),
                integer_offsets=(),
                prior_precision=active.prior_precision,
            )
            for kappa in kappas:
                add_lattice(
                    root_mode_id=root.mode_id,
                    parent_mode_id=None,
                    source="invariant-reacquisition",
                    linearization=linearization,
                    mean=mean,
                    covariance=covariance,
                    kappa=kappa,
                )
        except Exception:
            continue

    target = max(int(active.maximum_stage_proposals), 1)
    selected: list[LatticeProposal] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def add_unique(proposal: LatticeProposal) -> bool:
        rounded_state = tuple(
            int(value)
            for value in np.rint(proposal.initial_normalized_values * 1000.0).astype(int)
        )
        key = proposal.integer_offsets, rounded_state
        if key in seen:
            return False
        seen.add(key)
        selected.append(proposal)
        return True

    predictive = sorted(
        (item for item in raw if item.source == "predictive"),
        key=lambda item: (-item.log_predictive_density, item.root_mode_id),
    )
    reacquisition = sorted(
        (item for item in raw if item.source == "invariant-reacquisition"),
        key=lambda item: (-item.log_predictive_density, item.root_mode_id),
    )
    controls = sorted(
        (item for item in raw if item.source == "inherited-control"),
        key=lambda item: (item.root_mode_id, item.integer_offsets),
    )
    reacquisition_target = min(
        len(reacquisition),
        max(1, int(round(target * np.clip(
            float(active.reacquisition_proposal_fraction), 0.0, 1.0
        )))),
    )
    predictive_target = min(len(predictive), target - reacquisition_target)
    for proposal in predictive[:predictive_target]:
        add_unique(proposal)
    # Keep at least one proposal from each kappa family before filling by
    # predictive density. This portfolio quota imposes no source-specific prior.
    for kappa in sorted({item.kappa for item in reacquisition}):
        if len(selected) >= target or len(selected) >= predictive_target + reacquisition_target:
            break
        family = next((item for item in reacquisition if item.kappa == kappa), None)
        if family is not None:
            add_unique(family)
    for proposal in reacquisition:
        if len(selected) >= target or len(
            [item for item in selected if item.source == "invariant-reacquisition"]
        ) >= reacquisition_target:
            break
        add_unique(proposal)
    if controls and len(selected) < target:
        add_unique(controls[0])
    ranked = sorted(
        raw,
        key=lambda item: (
            item.source == "inherited-control",
            -item.log_predictive_density,
            item.root_mode_id,
            item.integer_offsets,
        ),
    )
    for proposal in ranked:
        if len(selected) >= target:
            break
        add_unique(proposal)
    return tuple(selected)


def refit_fixed_path_proposal(
    *,
    evaluator: Any,
    coordinates: Any,
    anchor_session: int,
    proposal: LatticeProposal,
    config: GlobalLatticeTrackerConfig | None = None,
    mode_id: str = "mode",
) -> GlobalLatticeMode:
    active = config or GlobalLatticeTrackerConfig()
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(np.asarray(proposal.initial_normalized_values, dtype=float), lower, upper)
    start_profile = evaluator.evaluate_values(_physical_values(coordinates, start))
    failure_size = _profile_shape_residuals(start_profile).size + len(
        proposal.active_sessions
    )
    if active.prior_precision > 0.0:
        failure_size += start.size
    failure = np.full(failure_size, 1.0e12, dtype=float)

    shape_size = _profile_shape_residuals(start_profile).size
    phase_size = len(proposal.active_sessions)

    def residual_vector(values: np.ndarray, phase_temperature: float) -> np.ndarray:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            result = fixed_path_residual_vector(
                profile,
                anchor_session=anchor_session,
                active_sessions=proposal.active_sessions,
                integer_offsets=proposal.integer_offsets,
                normalized_values=values,
                prior_precision=active.prior_precision,
            )
            result = result.copy()
            result[shape_size : shape_size + phase_size] *= math.sqrt(
                max(float(phase_temperature), 0.0)
            )
            return result if result.shape == failure.shape else failure
        except Exception:
            return failure

    start_shape_chi2, start_phase_chi2, _ = exact_fixed_path_chi2(
        start_profile,
        anchor_session=anchor_session,
        active_sessions=proposal.active_sessions,
        integer_offsets=proposal.integer_offsets,
    )
    ladder = adaptive_phase_tempering_ladder(
        declared_ladder=active.phase_tempering_ladder,
        shape_chi2=start_shape_chi2,
        phase_chi2=start_phase_chi2,
        information_dimension=shape_size + phase_size,
        initial_information_ratio=active.initial_phase_information_ratio,
        maximum_temperature_ratio=active.maximum_phase_temperature_ratio,
        maximum_adaptive_steps=active.maximum_adaptive_phase_steps,
    )
    per_bridge_nfev = max(
        8,
        int(math.ceil(max(int(active.fixed_path_max_nfev), 1) / len(ladder))),
    )
    normalized = start.copy()
    total_nfev = 0
    successful = True
    optimization = None
    for phase_temperature in ladder:
        optimization = least_squares(
            lambda values, temperature=phase_temperature: residual_vector(
                values,
                temperature,
            ),
            normalized,
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            diff_step=max(float(active.finite_difference_step), 1.0e-7),
            max_nfev=per_bridge_nfev,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        normalized = np.asarray(optimization.x, dtype=float)
        total_nfev += int(optimization.nfev)
        successful = successful and bool(optimization.success)
    if optimization is None:
        raise RuntimeError("The fixed-path phase bridge has no temperature step.")
    profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
    shape_chi2, phase_chi2, direct_chi2 = exact_fixed_path_chi2(
        profile,
        anchor_session=anchor_session,
        active_sessions=proposal.active_sessions,
        integer_offsets=proposal.integer_offsets,
    )
    prior_chi2 = float(
        max(float(active.prior_precision), 0.0) * (normalized @ normalized)
    )
    objective = direct_chi2 + prior_chi2
    jacobian = np.asarray(optimization.jac, dtype=float)
    covariance = _stable_inverse(jacobian.T @ jacobian)
    return GlobalLatticeMode(
        mode_id=mode_id,
        root_mode_id=proposal.root_mode_id,
        parent_mode_id=proposal.parent_mode_id,
        proposal_source=proposal.source,
        anchor_session=int(anchor_session),
        active_sessions=proposal.active_sessions,
        integer_offsets=proposal.integer_offsets,
        parameter_values=_physical_values(coordinates, normalized),
        normalized_values=normalized,
        covariance=covariance,
        shape_chi2=shape_chi2,
        phase_chi2=phase_chi2,
        prior_chi2=prior_chi2,
        objective_chi2=objective,
        log_mass=_laplace_log_mass(objective, covariance),
        posterior_mass=0.0,
        squared_mahalanobis=float(proposal.squared_mahalanobis),
        nfev=int(total_nfev),
        success=bool(successful),
    )


def select_diverse_lattice_modes(
    modes: Sequence[GlobalLatticeMode],
    *,
    maximum_modes: int,
    maximum_modes_per_path: int,
    maximum_modes_per_root: int,
    minimum_separation: float,
) -> tuple[GlobalLatticeMode, ...]:
    """Keep posterior modes without letting one path or root fill the beam."""

    target = max(int(maximum_modes), 1)
    path_cap = max(int(maximum_modes_per_path), 1)
    root_cap = max(int(maximum_modes_per_root), 1)
    separation = max(float(minimum_separation), 0.0)
    ranked = sorted(modes, key=lambda item: (-item.log_mass, item.objective_chi2))
    selected: list[GlobalLatticeMode] = []
    path_counts: dict[tuple[int, ...], int] = {}
    root_counts: dict[str, int] = {}

    def add(candidate: GlobalLatticeMode) -> bool:
        path = candidate.integer_offsets
        if path_counts.get(path, 0) >= path_cap:
            return False
        if root_counts.get(candidate.root_mode_id, 0) >= root_cap:
            return False
        if any(
            candidate.integer_offsets == existing.integer_offsets
            and np.linalg.norm(candidate.normalized_values - existing.normalized_values)
            < separation
            for existing in selected
        ):
            return False
        selected.append(candidate)
        path_counts[path] = path_counts.get(path, 0) + 1
        root_counts[candidate.root_mode_id] = root_counts.get(candidate.root_mode_id, 0) + 1
        return True

    best_by_root: dict[str, GlobalLatticeMode] = {}
    best_by_path: dict[tuple[int, ...], GlobalLatticeMode] = {}
    for candidate in ranked:
        best_by_root.setdefault(candidate.root_mode_id, candidate)
        best_by_path.setdefault(candidate.integer_offsets, candidate)
    for candidate in sorted(best_by_root.values(), key=lambda item: -item.log_mass):
        add(candidate)
        if len(selected) >= target:
            return _normalize_mode_masses(selected)
    for candidate in sorted(best_by_path.values(), key=lambda item: -item.log_mass):
        add(candidate)
        if len(selected) >= target:
            return _normalize_mode_masses(selected)
    for candidate in ranked:
        add(candidate)
        if len(selected) >= target:
            break
    return _normalize_mode_masses(selected)


def _initial_parent_modes(
    invariant_modes: Sequence[InvariantMode],
    *,
    anchor_session: int,
) -> tuple[GlobalLatticeMode, ...]:
    return tuple(
        GlobalLatticeMode(
            mode_id=f"stage_000_{mode.mode_id}",
            root_mode_id=mode.mode_id,
            parent_mode_id=None,
            proposal_source="invariant",
            anchor_session=anchor_session,
            active_sessions=(),
            integer_offsets=(),
            parameter_values=mode.parameter_values.copy(),
            normalized_values=mode.normalized_values.copy(),
            covariance=mode.covariance.copy(),
            shape_chi2=mode.shape_chi2,
            phase_chi2=0.0,
            prior_chi2=mode.prior_chi2,
            objective_chi2=mode.objective_chi2,
            log_mass=mode.log_mass,
            posterior_mass=mode.posterior_mass,
            squared_mahalanobis=0.0,
            nfev=mode.nfev,
            success=mode.success,
        )
        for mode in invariant_modes
    )


def classify_tracker_failure(
    *,
    invariant_modes: Sequence[InvariantMode],
    stage_history: Sequence[LatticeStageSummary],
    final_modes: Sequence[GlobalLatticeMode],
    session_count: int,
) -> str | None:
    if not invariant_modes:
        return "continuous-mode-discovery"
    if not stage_history or not final_modes:
        return "lattice-coverage"
    final_stage = stage_history[-1]
    if final_stage.minimum_lattice_distance > 1.0e4:
        return "lattice-coverage"
    if final_stage.best_shape_chi2 > min(mode.shape_chi2 for mode in invariant_modes) + 100.0:
        return "nonlinear-transport"
    if final_stage.unique_selected_paths <= 1 and session_count > 2:
        return "path-diversity"
    if len(final_modes[0].active_sessions) != session_count - 1:
        return "incomplete-session-transport"
    return None


def run_global_lattice_mode_tracker(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: GlobalLatticeTrackerConfig | None = None,
) -> GlobalLatticeTrackerResult:
    active = config or GlobalLatticeTrackerConfig()
    modes = tuple(invariant_modes)
    if not modes:
        raise ValueError("The global lattice tracker requires Stage-A invariant modes.")
    anchor, _, _ = choose_anchor_and_session_order(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=modes,
        finite_difference_step=active.finite_difference_step,
    )
    session_count = int(evaluator.evaluate_values(modes[0].parameter_values).session_count)
    parents = _initial_parent_modes(modes, anchor_session=anchor)
    history: list[LatticeStageSummary] = []
    portfolios: list[tuple[GlobalLatticeMode, ...]] = []
    remaining = set(range(session_count)) - {anchor}
    session_order: list[int] = []
    batch = max(int(active.sessions_per_stage), 1)
    stage_index = 0
    while remaining:
        stage_index += 1
        ranked_sessions = rank_predictive_sessions(
            evaluator=evaluator,
            coordinates=coordinates,
            parent_modes=parents,
            invariant_modes=modes,
            anchor_session=anchor,
            remaining_sessions=tuple(sorted(remaining)),
            finite_difference_step=active.finite_difference_step,
            session_entropy_penalty=active.session_entropy_penalty,
        )
        selected_sessions = ranked_sessions[: min(batch, len(ranked_sessions))]
        session_order.extend(int(value) for value in selected_sessions)
        remaining.difference_update(selected_sessions)
        active_sessions = tuple(session_order)
        proposals = generate_stage_lattice_proposals(
            evaluator=evaluator,
            coordinates=coordinates,
            anchor_session=anchor,
            active_sessions=active_sessions,
            parent_modes=parents,
            invariant_modes=modes,
            config=active,
        )
        refitted: list[GlobalLatticeMode] = []
        for proposal_index, proposal in enumerate(proposals):
            try:
                refitted.append(
                    refit_fixed_path_proposal(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        anchor_session=anchor,
                        proposal=proposal,
                        config=active,
                        mode_id=f"stage_{stage_index:03d}_mode_{proposal_index:04d}",
                    )
                )
            except Exception:
                continue
        selected = select_diverse_lattice_modes(
            refitted,
            maximum_modes=active.maximum_stage_modes,
            maximum_modes_per_path=active.maximum_modes_per_path,
            maximum_modes_per_root=active.maximum_modes_per_root,
            minimum_separation=active.mode_separation,
        ) if refitted else ()
        retry_count = 0
        retry_improved = False
        current_best = (
            min((item.objective_chi2 for item in selected), default=math.inf)
        )
        previous_best = history[-1].best_objective_chi2 if history else math.inf
        objective_jump = bool(
            np.isfinite(previous_best)
            and current_best
            > max(
                previous_best + max(float(active.transport_objective_jump), 0.0),
                previous_best * max(float(active.transport_objective_ratio), 1.0),
            )
        )
        if objective_jump and selected:
            retry_config = replace(
                active,
                fixed_path_max_nfev=max(
                    int(math.ceil(
                        active.fixed_path_max_nfev
                        * max(float(active.transport_retry_multiplier), 1.0)
                    )),
                    active.fixed_path_max_nfev + 1,
                ),
            )
            proposal_by_index = {
                index: proposal for index, proposal in enumerate(proposals)
            }
            retry_candidates = sorted(
                selected,
                key=lambda item: (item.objective_chi2, -item.log_mass),
            )[: max(int(active.transport_retry_modes), 1)]
            retried: list[GlobalLatticeMode] = []
            for mode in retry_candidates:
                try:
                    proposal_index = int(mode.mode_id.rsplit("_", 1)[-1])
                    proposal = proposal_by_index[proposal_index]
                    retried.append(
                        refit_fixed_path_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            anchor_session=anchor,
                            proposal=proposal,
                            config=retry_config,
                            mode_id=f"{mode.mode_id}_retry",
                        )
                    )
                except (KeyError, ValueError, RuntimeError):
                    continue
            retry_count = len(retried)
            if retried:
                before_retry = min(item.objective_chi2 for item in selected)
                after_retry = min(item.objective_chi2 for item in retried)
                retry_improved = bool(after_retry < before_retry)
                selected = select_diverse_lattice_modes(
                    [*refitted, *retried],
                    maximum_modes=active.maximum_stage_modes,
                    maximum_modes_per_path=active.maximum_modes_per_path,
                    maximum_modes_per_root=active.maximum_modes_per_root,
                    minimum_separation=active.mode_separation,
                )
        portfolios.append(tuple(selected))
        finite_distances = [
            item.squared_mahalanobis
            for item in proposals
            if np.isfinite(item.squared_mahalanobis)
        ]
        failure = None
        if not proposals:
            failure = "lattice-coverage"
        elif not refitted:
            failure = "nonlinear-transport"
        elif not selected:
            failure = "path-diversity"
        history.append(
            LatticeStageSummary(
                stage_index=stage_index,
                active_sessions=tuple(active_sessions),
                proposed_count=len(proposals),
                refitted_count=len(refitted),
                selected_count=len(selected),
                unique_proposed_paths=len({item.integer_offsets for item in proposals}),
                unique_selected_paths=len({item.integer_offsets for item in selected}),
                selected_root_count=len({item.root_mode_id for item in selected}),
                transport_retry_count=retry_count,
                transport_retry_improved=retry_improved,
                minimum_lattice_distance=(
                    min(finite_distances) if finite_distances else math.inf
                ),
                best_shape_chi2=(
                    min(item.shape_chi2 for item in selected) if selected else math.inf
                ),
                best_phase_chi2=(
                    min(item.phase_chi2 for item in selected) if selected else math.inf
                ),
                best_objective_chi2=(
                    min(item.objective_chi2 for item in selected) if selected else math.inf
                ),
                failure_class=failure,
            )
        )
        if not selected:
            parents = ()
            break
        parents = selected
    failure_class = classify_tracker_failure(
        invariant_modes=modes,
        stage_history=history,
        final_modes=parents,
        session_count=session_count,
    )
    reached_all = bool(
        parents and len(parents[0].active_sessions) == session_count - 1
    )
    return GlobalLatticeTrackerResult(
        anchor_session=anchor,
        session_order=tuple(session_order),
        invariant_modes=modes,
        final_modes=tuple(parents),
        stage_history=tuple(history),
        stage_portfolios=tuple(portfolios),
        reached_all_sessions=reached_all,
        failure_class=failure_class,
    )


__all__ = [
    "adaptive_phase_tempering_ladder",
    "GlobalLatticeMode",
    "GlobalLatticeTrackerConfig",
    "GlobalLatticeTrackerResult",
    "InvariantMode",
    "InvariantModeSeed",
    "LatticeProposal",
    "LatticeStageSummary",
    "choose_anchor_and_session_order",
    "exact_fixed_path_chi2",
    "finite_difference_profile",
    "fixed_path_residual_vector",
    "generate_stage_lattice_proposals",
    "propose_complete_integer_paths",
    "rank_predictive_sessions",
    "refine_invariant_mode_portfolio",
    "refit_fixed_path_proposal",
    "run_global_lattice_mode_tracker",
    "select_diverse_lattice_modes",
]
