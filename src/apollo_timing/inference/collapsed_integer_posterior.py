from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit, logsumexp
from scipy.optimize import least_squares, minimize

from apollo_timing.inference.information_ordered_binary_search import (
    DDSearchCoordinateSystem,
)
from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
    integer_session_offsets_to_gap_wraps,
)


@dataclass(frozen=True)
class CollapsedIntegerConfig:
    """Scientific controls for the collapsed integer posterior.

    Defaults prioritize posterior coverage over runtime to establish
    recoverability before optimizing the budget.
    """

    particle_count: int = 128
    invariant_ess_target_fraction: float = 0.65
    phase_ess_target_fraction: float = 0.65
    joint_ess_target_fraction: float = 0.65
    bridge_ess_target_fraction: float = 0.65
    tempering_order: str = "phase_noise_bridge"
    phase_noise_start_cycles: float = 1.0
    phase_noise_schedule_power: float = 1.0
    phase_noise_parameterization: str = "sigma"
    bridge_ess_scan_points: int = 16
    bridge_rejuvenation_information_fraction: float = 0.995
    maximum_tempering_steps: int = 96
    minimum_beta_step: float = 1.0e-7
    maximum_beta_step: float = 0.25
    move_steps_per_tempering_step: int = 3
    proposal_scale: float = 0.04
    differential_move_probability: float = 0.70
    differential_scale: float = 0.0
    differential_jitter: float = 1.0e-4
    phase_metric_move_probability: float = 0.55
    phase_metric_chart_count: int = 8
    phase_metric_finite_difference_step: float = 1.0e-5
    phase_metric_scale: float = 1.0
    phase_metric_minimum_std: float = 1.0e-8
    phase_metric_maximum_std: float = 0.08
    global_jump_probability: float = 0.15
    mode_discovery_start_count: int = 0
    mode_discovery_spin_strata: int = 9
    mode_discovery_max_nfev: int = 64
    mode_seed_fraction: float = 0.25
    mode_covariance_minimum_std: float = 1.0e-8
    mode_covariance_maximum_std: float = 0.05
    mode_covariance_scale: float = 1.0
    bridge_anchor_count: int = 8
    bridge_anchor_schedule_steps: int = 20
    bridge_anchor_children_per_parent: int = 2
    bridge_anchor_portfolio_count: int = 16
    bridge_anchor_branch_start_sigma_cycles: float = 0.35
    bridge_anchor_branch_stride: int = 2
    bridge_anchor_child_log_mass_floor: float = -8.0
    bridge_anchor_mode_separation: float = 1.0e-5
    bridge_anchor_maxiter: int = 12
    bridge_anchor_finite_difference_step: float = 1.0e-6
    bridge_anchor_lattice_finite_difference_step: float = 1.0e-4
    bridge_anchor_lattice_maximum_nodes: int = 1_000_000
    bridge_anchor_lattice_covariance_scales: tuple[float, ...] = (1.0,)
    conditional_clock_lattice: bool = False
    conditional_clock_parameter_names: tuple[str, ...] = ("F0", "F1")
    conditional_clock_parent_count: int = 4
    conditional_clock_candidate_count: int = 16
    conditional_clock_maxiter: int = 24
    mode_jump_probability: float = 0.30
    hermite_order: int = 24
    integer_tail_tolerance: float = 1.0e-12
    maximum_wrap_terms: int = 9
    integer_candidate_count: int = 32
    joint_optimization_path_count: int = 32
    joint_optimization_maxiter: int = 24
    joint_optimization_finite_difference_step: float = 1.0e-6
    joint_move_steps: int = 16
    joint_proposal_scale: float = 0.012
    random_seed: int = 0
    maximum_paths: int = 64


@dataclass(frozen=True)
class SessionLikelihoodFactors:
    invariant_log_likelihood: float
    wrapped_log_likelihood: float
    relative_offsets: np.ndarray
    relative_variance: np.ndarray
    wrapped_tail_bound: float
    direct_shape_chi2: float


@dataclass(frozen=True)
class IntegerPathCandidate:
    integer_offsets: tuple[int, ...]
    wraps: tuple[int, ...]
    log_abs_likelihood: float
    log_conditional_mass: float
    proposal_source: str = "posterior"


@dataclass
class _Particle:
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: SessionShapeProfile
    factors: SessionLikelihoodFactors
    log_weight: float
    log_prior: float = 0.0
    accepted_moves: int = 0
    cached_phase_variance: float | None = None
    cached_wrapped_log_likelihood: float | None = None


@dataclass(frozen=True)
class _CorrelatedIntegerPathProposal:
    candidate: IntegerPathCandidate
    normalized_start: np.ndarray
    predictive_chi2: float
    covariance_scale: float = 1.0


@dataclass(frozen=True)
class CorrelatedCompletePathProposal:
    """A complete integer path coupled to its conditional continuous start."""

    integer_offsets: tuple[int, ...]
    wraps: tuple[int, ...]
    normalized_start: np.ndarray
    predictive_chi2: float
    log_conditional_mass: float
    covariance_scale: float
    proposal_source: str


@dataclass(frozen=True)
class CorrelatedCompletePathProposalResult:
    proposals: tuple[CorrelatedCompletePathProposal, ...]
    profile_evaluations: int
    lattice_visited_nodes: int


@dataclass(frozen=True)
class _AcquiredAnchorPath:
    """A complete lattice path paired with its nonlinear continuous chart."""

    candidate: IntegerPathCandidate
    particle: _Particle
    bridge_log_target: float
    exact_log_target: float
    phase_noise_sigma_cycles: float
    acquisition_family: str


@dataclass(frozen=True)
class _IndependentModeProposal:
    center: np.ndarray
    covariance: np.ndarray
    log_mixture_weight: float


@dataclass(frozen=True)
class _BoundedModeProposal:
    transformed_center: np.ndarray
    transformed_covariance: np.ndarray
    log_mixture_weight: float


@dataclass(frozen=True)
class CollapsedParticleSummary:
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    invariant_log_likelihood: float
    wrapped_log_likelihood: float
    log_weight: float
    log_prior: float = 0.0


@dataclass(frozen=True)
class InvariantModeSummary:
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    normalized_covariance: np.ndarray
    invariant_log_likelihood: float
    direct_shape_chi2: float
    normalized_boundary_margin: float


@dataclass(frozen=True)
class JointPathSummary:
    integer_offsets: tuple[int, ...]
    wraps: tuple[int, ...]
    source_particle_index: int
    proposal_source: str
    posterior_mass: float | None
    best_log_joint: float
    parameter_values: np.ndarray
    normalized_values: np.ndarray
    continuous_optimization_evaluations: int
    continuous_optimization_improved: bool
    accepted_continuous_moves: int
    accepted_integer_moves: int


@dataclass(frozen=True)
class PersistentPathAcquisitionSummary:
    integer_offsets: tuple[int, ...]
    wraps: tuple[int, ...]
    proposal_source: str
    acquisition_family: str
    phase_noise_sigma_cycles: float
    bridge_log_target: float
    exact_log_target: float
    parameter_values: np.ndarray
    normalized_values: np.ndarray


@dataclass(frozen=True)
class CollapsedIntegerInferenceResult:
    particles: tuple[CollapsedParticleSummary, ...]
    bridge_anchors: tuple[CollapsedParticleSummary, ...]
    path_posterior: tuple[IntegerPathCandidate, ...]
    search_path_portfolio: tuple[IntegerPathCandidate, ...]
    persistent_path_acquisitions: tuple[
        PersistentPathAcquisitionSummary,
        ...,
    ]
    joint_paths: tuple[JointPathSummary, ...]
    history: tuple[dict[str, Any], ...]
    reached_beta_one: bool
    log_evidence: float
    convergence: dict[str, Any]


def _finite_logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.any(np.isfinite(array)):
        return float("-inf")
    return float(logsumexp(array))


def _integer_terms(
    value: float,
    variance: float,
    *,
    tail_tolerance: float,
    maximum_terms: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    variance = max(float(variance), np.finfo(float).tiny)
    standard_deviation = math.sqrt(variance)
    tail_width = math.sqrt(max(2.0 * math.log(1.0 / max(tail_tolerance, 1.0e-300)), 1.0))
    radius = max(1, int(math.ceil(tail_width * standard_deviation)) + 1)
    radius = min(radius, max(int(maximum_terms), 1))
    center = int(np.rint(float(value)))
    integers = center + np.arange(-radius, radius + 1, dtype=np.int64)
    log_terms = -0.5 * (
        np.square(float(value) - integers.astype(float)) / variance
        + math.log(2.0 * math.pi * variance)
    )
    # This is a conservative diagnostic. It is zero when the requested tail
    # radius is represented; otherwise the caller must treat support as open.
    tail_bound = 0.0 if radius < max(int(maximum_terms), 1) else tail_tolerance
    return integers, log_terms, tail_bound


def fixed_integer_log_likelihood(
    profile: SessionShapeProfile,
    integer_offsets: Sequence[int],
    *,
    added_session_phase_variance: float = 0.0,
) -> float:
    """Return the exact session-mean likelihood after integrating one phase offset.

    Integrate the common continuous phase offset analytically. The anchor
    integer is zero; `integer_offsets` contains the other relative session
    integers. `SessionLikelihoodFactors` returns the orthogonal local likelihood
    separately, so it is excluded here.
    """

    offsets = np.asarray(profile.session_offsets, dtype=float)
    added_variance = float(added_session_phase_variance)
    if not np.isfinite(added_variance) or added_variance < 0.0:
        raise ValueError("Added session phase variance must be finite and non-negative.")
    variances = (
        np.asarray(profile.session_offset_variances, dtype=float)
        + added_variance
    )
    integers = np.concatenate(
        (np.zeros(1, dtype=float), np.asarray(integer_offsets, dtype=float))
    )
    if offsets.ndim != 1 or variances.shape != offsets.shape:
        raise ValueError("Session offsets and variances must be one-dimensional and aligned.")
    if integers.shape != offsets.shape:
        raise ValueError("Integer offsets must match the relative session dimension.")
    if not np.all(np.isfinite(offsets) & np.isfinite(variances) & (variances > 0.0)):
        raise ValueError("Session offsets and variances must be finite and positive.")

    precision = 1.0 / variances
    centered = offsets - integers
    phase_offset = float(np.sum(precision * centered) / np.sum(precision))
    minimum_chi2 = float(np.sum(precision * np.square(centered - phase_offset)))
    log_normalization = -0.5 * float(
        np.sum(np.log(2.0 * math.pi * variances))
    )
    log_integral = 0.5 * math.log(2.0 * math.pi / float(np.sum(precision)))
    return float(-0.5 * minimum_chi2 + log_normalization + log_integral)


def _nearest_integer_partition_paths(offsets: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Enumerate every nearest-integer path over one common-phase period."""

    if offsets.size <= 1:
        return ((),)
    lower = float(offsets[0]) - 0.5
    upper = lower + 1.0
    boundaries = [lower, upper]
    for value in offsets[1:]:
        boundary = lower + ((float(value) - 0.5 - lower) % 1.0)
        if lower + 1.0e-12 < boundary < upper - 1.0e-12:
            boundaries.append(boundary)
    ordered = np.unique(np.asarray(boundaries, dtype=float))
    paths: set[tuple[int, ...]] = set()
    for left, right in zip(ordered[:-1], ordered[1:], strict=True):
        delta = 0.5 * (float(left) + float(right))
        path = tuple(int(value) for value in np.rint(offsets[1:] - delta))
        paths.add(path)
    return tuple(sorted(paths))


def _partition_path_log_sum(
    profile: SessionShapeProfile,
    *,
    added_session_phase_variance: float,
    tail_tolerance: float,
) -> tuple[float, float]:
    """Sum narrow wrapped-normal components using analytic phase integrals.

    A complete relative-integer path is constant between adjacent half-cycle
    boundaries.  Enumerating those partitions avoids sampling the common-phase
    peak, whose width shrinks as the number of sessions grows.
    """

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = (
        np.asarray(profile.session_offset_variances, dtype=float)
        + float(added_session_phase_variance)
    )
    paths = _nearest_integer_partition_paths(offsets)
    log_masses = [
        fixed_integer_log_likelihood(
            profile,
            path,
            added_session_phase_variance=added_session_phase_variance,
        )
        for path in paths
    ]
    maximum_variance = float(np.max(variances))
    omitted_bound = min(
        1.0,
        float(offsets.size)
        * math.exp(-0.125 / max(maximum_variance, np.finfo(float).tiny)),
    )
    if omitted_bound < tail_tolerance:
        omitted_bound = 0.0
    return _finite_logsumexp(log_masses), float(omitted_bound)


def _periodic_wrapped_log_likelihood(
    offsets: np.ndarray,
    variances: np.ndarray,
    *,
    grid_count: int,
    tail_tolerance: float,
    maximum_wrap_terms: int,
) -> tuple[float, float]:
    """Integrate the wrapped product on an evenly sampled phase circle."""

    deltas = float(offsets[0]) + np.arange(grid_count, dtype=float) / grid_count
    log_density = np.zeros(grid_count, dtype=float)
    tail_bound = 0.0
    for value, variance in zip(offsets, variances, strict=True):
        standard_deviation = math.sqrt(float(variance))
        tail_width = math.sqrt(
            max(2.0 * math.log(1.0 / max(tail_tolerance, 1.0e-300)), 1.0)
        )
        radius = max(1, int(math.ceil(tail_width * standard_deviation)) + 1)
        radius = min(radius, max(int(maximum_wrap_terms), 1))
        difference = float(value) - deltas
        centers = np.rint(difference).astype(np.int64)
        shifts = np.arange(-radius, radius + 1, dtype=np.int64)
        integers = centers[:, None] + shifts[None, :]
        terms = -0.5 * (
            np.square(difference[:, None] - integers.astype(float)) / variance
            + math.log(2.0 * math.pi * float(variance))
        )
        log_density += logsumexp(terms, axis=1)
        if radius >= max(int(maximum_wrap_terms), 1):
            tail_bound = max(tail_bound, tail_tolerance)
    return float(logsumexp(log_density) - math.log(grid_count)), float(tail_bound)


def collapsed_wrapped_log_likelihood(
    profile: SessionShapeProfile,
    *,
    hermite_order: int = 24,
    integer_tail_tolerance: float = 1.0e-12,
    maximum_wrap_terms: int = 9,
    added_session_phase_variance: float = 0.0,
) -> tuple[float, float]:
    """Marginalize all relative session integers over one phase period.

    Use periodic quadrature for smooth posteriors. When many precise sessions
    make the common-phase peak narrower than a practical grid cell, integrate
    each nearest-integer partition analytically. This avoids losing the dominant
    path with fixed-order quadrature.
    """

    offsets = np.asarray(profile.session_offsets, dtype=float)
    base_variances = np.asarray(profile.session_offset_variances, dtype=float)
    added_variance = float(added_session_phase_variance)
    if not np.isfinite(added_variance) or added_variance < 0.0:
        raise ValueError("Added session phase variance must be finite and non-negative.")
    variances = base_variances + added_variance
    if offsets.ndim != 1 or variances.shape != offsets.shape:
        raise ValueError("Session offsets and variances must be one-dimensional and aligned.")
    if offsets.size <= 1:
        return 0.0, 0.0
    if not np.all(np.isfinite(offsets) & np.isfinite(variances) & (variances > 0.0)):
        raise ValueError("Session offsets and variances must be finite and positive.")

    posterior_sigma = 1.0 / math.sqrt(float(np.sum(1.0 / variances)))
    requested_points = max(
        64,
        2 * max(int(hermite_order), 4),
        int(math.ceil(8.0 / max(posterior_sigma, np.finfo(float).eps))),
    )
    grid_count = 1 << int(math.ceil(math.log2(requested_points)))
    maximum_grid_count = max(4096, 16 * max(int(hermite_order), 4))
    if grid_count <= maximum_grid_count:
        numerical, tail_bound = _periodic_wrapped_log_likelihood(
            offsets,
            variances,
            grid_count=grid_count,
            tail_tolerance=integer_tail_tolerance,
            maximum_wrap_terms=maximum_wrap_terms,
        )
        path_lower_bound, path_tail = _partition_path_log_sum(
            profile,
            added_session_phase_variance=added_variance,
            tail_tolerance=integer_tail_tolerance,
        )
        return float(max(numerical, path_lower_bound)), float(
            max(tail_bound, path_tail if numerical < path_lower_bound else 0.0)
        )
    return _partition_path_log_sum(
        profile,
        added_session_phase_variance=added_variance,
        tail_tolerance=integer_tail_tolerance,
    )


def evaluate_session_likelihood_factors(
    profile: SessionShapeProfile,
    *,
    hermite_order: int = 24,
    integer_tail_tolerance: float = 1.0e-12,
    maximum_wrap_terms: int = 9,
    added_session_phase_variance: float = 0.0,
) -> SessionLikelihoodFactors:
    wrapped, tail_bound = collapsed_wrapped_log_likelihood(
        profile,
        hermite_order=hermite_order,
        integer_tail_tolerance=integer_tail_tolerance,
        maximum_wrap_terms=maximum_wrap_terms,
        added_session_phase_variance=added_session_phase_variance,
    )
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    return SessionLikelihoodFactors(
        invariant_log_likelihood=float(-0.5 * profile.chi2),
        wrapped_log_likelihood=float(wrapped),
        relative_offsets=offsets[1:] - offsets[0],
        relative_variance=np.diag(variances[1:]) + variances[0],
        wrapped_tail_bound=float(tail_bound),
        direct_shape_chi2=float(profile.chi2),
    )


def exact_relative_chi2(
    profile: SessionShapeProfile,
    integer_offsets: Sequence[int],
) -> float:
    """Return the integer-dependent ambiguity chi-square in relative coordinates."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    relative = offsets[1:] - offsets[0]
    integers = np.asarray(integer_offsets, dtype=float)
    if integers.shape != relative.shape:
        raise ValueError("Integer offsets must match the relative session dimension.")
    covariance = np.diag(variances[1:]) + variances[0]
    delta = relative - integers
    solved = np.linalg.solve(covariance, delta)
    return float(delta @ solved)


def exact_decomposed_chi2(
    profile: SessionShapeProfile,
    integer_offsets: Sequence[int],
) -> float:
    return float(profile.chi2 + exact_relative_chi2(profile, integer_offsets))


def _top_integer_options(
    value: float,
    variance: float,
    *,
    tail_tolerance: float,
    maximum_terms: int,
    count: int,
) -> list[tuple[int, float]]:
    integers, log_terms, _ = _integer_terms(
        value,
        variance,
        tail_tolerance=tail_tolerance,
        maximum_terms=maximum_terms,
    )
    order = np.argsort(log_terms)[::-1][: max(int(count), 1)]
    return [
        (int(integers[index]), float(log_terms[index]))
        for index in order.tolist()
    ]


def _beam_integer_paths(
    options_by_session: Sequence[Sequence[tuple[int, float]]],
    count: int,
) -> list[tuple[tuple[int, ...], float]]:
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for options in options_by_session:
        expanded = [
            (path + (integer,), score + log_score)
            for path, score in beam
            for integer, log_score in options
        ]
        expanded.sort(key=lambda item: item[1], reverse=True)
        beam = expanded[: max(int(count), 1)]
    return beam


def conditional_integer_paths(
    profile: SessionShapeProfile,
    *,
    candidate_count: int = 32,
    hermite_order: int = 24,
    integer_tail_tolerance: float = 1.0e-12,
    maximum_wrap_terms: int = 9,
    added_session_phase_variance: float = 0.0,
) -> tuple[IntegerPathCandidate, ...]:
    """Recover high-mass relative integer paths after continuous collapse."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    added_variance = float(added_session_phase_variance)
    if not np.isfinite(added_variance) or added_variance < 0.0:
        raise ValueError("Added session phase variance must be finite and non-negative.")
    variances = (
        np.asarray(profile.session_offset_variances, dtype=float)
        + added_variance
    )
    if offsets.size <= 1:
        return (IntegerPathCandidate((), (), 0.0, 0.0),)
    nodes, _ = hermgauss(max(int(hermite_order), 4))
    anchor_variance = max(float(variances[0]), np.finfo(float).tiny)
    anchor = float(offsets[0])
    paths: set[tuple[int, ...]] = set()
    for node in nodes:
        delta = anchor + math.sqrt(2.0 * anchor_variance) * float(node)
        options = [
            _top_integer_options(
                float(value - delta),
                float(variance),
                tail_tolerance=integer_tail_tolerance,
                maximum_terms=maximum_wrap_terms,
                count=3,
            )
            for value, variance in zip(offsets[1:], variances[1:], strict=True)
        ]
        paths.update(
            path
            for path, _ in _beam_integer_paths(options, max(int(candidate_count), 1))
        )

    collapsed_log, _ = collapsed_wrapped_log_likelihood(
        profile,
        hermite_order=hermite_order,
        integer_tail_tolerance=integer_tail_tolerance,
        maximum_wrap_terms=maximum_wrap_terms,
        added_session_phase_variance=added_variance,
    )
    ranked: list[IntegerPathCandidate] = []
    for path in paths:
        log_abs = fixed_integer_log_likelihood(
            profile,
            path,
            added_session_phase_variance=added_variance,
        )
        ranked.append(
            IntegerPathCandidate(
                integer_offsets=tuple(int(value) for value in path),
                wraps=integer_session_offsets_to_gap_wraps(path),
                log_abs_likelihood=float(log_abs),
                log_conditional_mass=float(log_abs - collapsed_log),
            )
        )
    ranked.sort(key=lambda item: item.log_conditional_mass, reverse=True)
    return tuple(ranked[: max(int(candidate_count), 1)])


def _continuous_marginalized_integer_paths(
    particle: _Particle,
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    candidate_count: int,
    added_session_phase_variance: float,
    config: CollapsedIntegerConfig,
    coordinate_indices: Sequence[int] | None = None,
) -> tuple[tuple[_CorrelatedIntegerPathProposal, ...], int, int]:
    """Propose complete paths after marginalizing one local shape chart.

    The pointwise conditional path posterior is too narrow when a coherent
    integer path requires a coupled move in the binary parameters. This chart
    propagates the local all-session shape covariance into the complete
    relative-session phase vector before performing integer least squares.
    Every proposal is still corrected by the nonlinear fixed-path target.
    """

    center = np.asarray(particle.normalized_values, dtype=float)
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    dimension = center.size
    active_indices = tuple(
        range(dimension)
        if coordinate_indices is None
        else (int(value) for value in coordinate_indices)
    )
    if not active_indices or len(set(active_indices)) != len(active_indices):
        raise ValueError("Correlated lattice coordinates must be unique and non-empty.")
    if min(active_indices) < 0 or max(active_indices) >= dimension:
        raise ValueError("Correlated lattice coordinate index is out of bounds.")
    active_dimension = len(active_indices)
    profile = particle.profile
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    relative = offsets[1:] - offsets[0]
    if relative.size == 0:
        candidate = IntegerPathCandidate((), (), 0.0, 0.0, "anchor-correlated-lattice")
        return (
            (_CorrelatedIntegerPathProposal(candidate, center.copy(), 0.0),),
            0,
            0,
        )

    shape_jacobian = np.empty((shape.size, active_dimension), dtype=float)
    phase_jacobian = np.empty((relative.size, active_dimension), dtype=float)
    evaluations = 0
    requested_step = max(
        float(config.bridge_anchor_lattice_finite_difference_step),
        1.0e-10,
    )

    def evaluate(normalized_values: np.ndarray) -> SessionShapeProfile:
        nonlocal evaluations
        physical = coordinates.to_physical(
            coordinates.denormalize(normalized_values)
        )
        evaluations += 1
        return evaluator.evaluate_values(physical)

    for active_index, parameter_index in enumerate(active_indices):
        left_step = min(
            requested_step,
            0.45 * float(center[parameter_index] - lower[parameter_index]),
        )
        right_step = min(
            requested_step,
            0.45 * float(upper[parameter_index] - center[parameter_index]),
        )
        left_profile = None
        right_profile = None
        if left_step > 0.0:
            left = center.copy()
            left[parameter_index] -= left_step
            try:
                left_profile = evaluate(left)
            except (
                ArithmeticError,
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                left_profile = None
        if right_step > 0.0:
            right = center.copy()
            right[parameter_index] += right_step
            try:
                right_profile = evaluate(right)
            except (
                ArithmeticError,
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                right_profile = None

        if left_profile is not None and right_profile is not None:
            denominator = left_step + right_step
            shape_jacobian[:, active_index] = (
                np.asarray(right_profile.normalized_residuals, dtype=float)
                - np.asarray(left_profile.normalized_residuals, dtype=float)
            ) / denominator
            phase_jacobian[:, active_index] = (
                np.asarray(right_profile.session_offsets[1:], dtype=float)
                - float(right_profile.session_offsets[0])
                - np.asarray(left_profile.session_offsets[1:], dtype=float)
                + float(left_profile.session_offsets[0])
            ) / denominator
        elif right_profile is not None:
            shape_jacobian[:, active_index] = (
                np.asarray(right_profile.normalized_residuals, dtype=float) - shape
            ) / right_step
            phase_jacobian[:, active_index] = (
                np.asarray(right_profile.session_offsets[1:], dtype=float)
                - float(right_profile.session_offsets[0])
                - relative
            ) / right_step
        elif left_profile is not None:
            shape_jacobian[:, active_index] = (
                shape - np.asarray(left_profile.normalized_residuals, dtype=float)
            ) / left_step
            phase_jacobian[:, active_index] = (
                relative
                - np.asarray(left_profile.session_offsets[1:], dtype=float)
                + float(left_profile.session_offsets[0])
            ) / left_step
        else:
            raise ValueError(
                "Correlated lattice could not linearize every search parameter."
            )

    if not np.all(np.isfinite(shape_jacobian)) or not np.all(
        np.isfinite(phase_jacobian)
    ):
        raise ValueError("Correlated lattice Jacobians must be finite.")

    shape_information = shape_jacobian.T @ shape_jacobian
    shape_information = 0.5 * (shape_information + shape_information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(shape_information)
    maximum_information = max(float(np.max(eigenvalues)), 0.0)
    information_floor = max(
        maximum_information * 1.0e-12,
        np.finfo(float).tiny,
    )
    active_lower = lower[list(active_indices)]
    active_upper = upper[list(active_indices)]
    prior_variances = np.maximum(
        np.square(active_upper - active_lower) / 12.0,
        1.0e-16,
    )
    projected_prior_variances = (
        np.square(eigenvectors).T @ prior_variances
    )
    covariance_eigenvalues = np.where(
        eigenvalues > information_floor,
        np.minimum(
            1.0 / np.maximum(eigenvalues, information_floor),
            projected_prior_variances,
        ),
        projected_prior_variances,
    )
    shape_covariance = (
        eigenvectors * covariance_eigenvalues
    ) @ eigenvectors.T
    shape_covariance = 0.5 * (shape_covariance + shape_covariance.T)
    shape_gradient = shape_jacobian.T @ shape
    shape_mode_delta = -shape_covariance @ shape_gradient
    predictive_center = relative + phase_jacobian @ shape_mode_delta

    added_variance = max(float(added_session_phase_variance), 0.0)
    session_variances = (
        np.asarray(profile.session_offset_variances, dtype=float)
        + added_variance
    )
    measurement_covariance = (
        np.diag(session_variances[1:]) + session_variances[0]
    )
    measurement_precision_phase_jacobian = np.linalg.solve(
        measurement_covariance,
        phase_jacobian,
    )
    interior = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
    scales: list[float] = []
    for value in config.bridge_anchor_lattice_covariance_scales:
        scale = float(value)
        if not np.isfinite(scale) or scale <= 0.0:
            continue
        if not any(
            math.isclose(scale, previous, rel_tol=1.0e-12)
            for previous in scales
        ):
            scales.append(scale)
    if not scales:
        scales = [1.0]

    proposals: list[_CorrelatedIntegerPathProposal] = []
    visited_nodes = 0
    for covariance_scale in scales:
        scale_squared = covariance_scale * covariance_scale
        predictive_covariance = (
            measurement_covariance
            + scale_squared
            * phase_jacobian
            @ shape_covariance
            @ phase_jacobian.T
        )
        predictive_covariance = 0.5 * (
            predictive_covariance + predictive_covariance.T
        )
        predictive_eigenvalues, predictive_eigenvectors = np.linalg.eigh(
            predictive_covariance
        )
        predictive_floor = max(
            float(np.max(predictive_eigenvalues)) * 1.0e-13,
            np.finfo(float).tiny,
        )
        predictive_covariance = (
            predictive_eigenvectors
            * np.maximum(predictive_eigenvalues, predictive_floor)
        ) @ predictive_eigenvectors.T
        predictive_covariance = 0.5 * (
            predictive_covariance + predictive_covariance.T
        )
        maximum_nodes = max(
            int(config.bridge_anchor_lattice_maximum_nodes),
            1,
        )
        try:
            integer_result = solve_integer_least_squares(
                predictive_center,
                predictive_covariance,
                candidate_count=max(int(candidate_count), 1),
                max_visited_nodes=maximum_nodes,
                return_partial_on_budget=True,
            )
        except RuntimeError:
            continue
        visited_nodes += int(integer_result.visited_nodes)
        if not integer_result.candidates:
            continue

        conditional_information = (
            shape_information / scale_squared
            + phase_jacobian.T @ measurement_precision_phase_jacobian
        )
        conditional_information = 0.5 * (
            conditional_information + conditional_information.T
        )
        best_chi2 = float(integer_result.candidates[0].squared_mahalanobis)
        for integer_candidate in integer_result.candidates:
            integers = np.asarray(integer_candidate.integer_vector, dtype=float)
            conditional_gradient = (
                shape_gradient / scale_squared
                + phase_jacobian.T
                @ np.linalg.solve(measurement_covariance, relative - integers)
            )
            try:
                conditional_delta = np.linalg.solve(
                    conditional_information,
                    -conditional_gradient,
                )
            except np.linalg.LinAlgError:
                conditional_delta = np.linalg.lstsq(
                    conditional_information,
                    -conditional_gradient,
                    rcond=1.0e-12,
                )[0]
            normalized_start = center.copy()
            normalized_start[list(active_indices)] = np.clip(
                center[list(active_indices)] + conditional_delta,
                active_lower + interior[list(active_indices)],
                active_upper - interior[list(active_indices)],
            )
            normalized_start = _project_normalized_segment_to_support(
                center,
                normalized_start,
                coordinates=coordinates,
            )
            predictive_chi2 = float(integer_candidate.squared_mahalanobis)
            path = tuple(int(value) for value in integer_candidate.integer_vector)
            candidate = IntegerPathCandidate(
                integer_offsets=path,
                wraps=integer_session_offsets_to_gap_wraps(path),
                log_abs_likelihood=float(-0.5 * predictive_chi2),
                log_conditional_mass=float(-0.5 * (predictive_chi2 - best_chi2)),
                proposal_source=(
                    "anchor-correlated-lattice-scale-"
                    f"{covariance_scale:g}"
                ),
            )
            proposals.append(
                _CorrelatedIntegerPathProposal(
                    candidate=candidate,
                    normalized_start=np.asarray(normalized_start, dtype=float),
                    predictive_chi2=predictive_chi2,
                    covariance_scale=covariance_scale,
                )
            )
    return (
        tuple(proposals),
        evaluations,
        visited_nodes,
    )


def _normalized_state_is_in_support(
    values: np.ndarray,
    *,
    coordinates: DDSearchCoordinateSystem,
) -> bool:
    """Return whether a normalized chart state represents prior support."""

    normalized = np.asarray(values, dtype=float)
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if (
        normalized.shape != lower.shape
        or not np.all(np.isfinite(normalized))
        or np.any(normalized <= lower)
        or np.any(normalized >= upper)
    ):
        return False
    try:
        coordinates.to_physical(coordinates.denormalize(normalized))
        prior_density = getattr(
            coordinates,
            "log_normalized_prior_density",
            None,
        )
        return not callable(prior_density) or np.isfinite(
            float(prior_density(normalized))
        )
    except (
        ArithmeticError,
        FloatingPointError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return False


def _project_normalized_segment_to_support(
    origin: Sequence[float],
    proposed: Sequence[float],
    *,
    coordinates: DDSearchCoordinateSystem,
    scan_steps: int = 32,
    bisection_steps: int = 48,
) -> np.ndarray:
    """Keep a coupled proposal in the origin's connected support interval.

    A box in a nonlinear search chart can contain points that map outside the
    declared physical prior. Preserve the proposal direction and find the
    first physical-support boundary along the segment from a valid mode.
    """

    start = np.asarray(origin, dtype=float)
    target = np.asarray(proposed, dtype=float)
    if not _normalized_state_is_in_support(start, coordinates=coordinates):
        raise ValueError("Support projection requires a valid origin state.")
    if _normalized_state_is_in_support(target, coordinates=coordinates):
        return target.copy()

    direction = target - start
    last_valid = 0.0
    first_invalid = 1.0
    for fraction in np.linspace(
        0.0,
        1.0,
        num=max(int(scan_steps), 2) + 1,
    )[1:]:
        candidate = start + float(fraction) * direction
        if _normalized_state_is_in_support(candidate, coordinates=coordinates):
            last_valid = float(fraction)
            continue
        first_invalid = float(fraction)
        break

    for _ in range(max(int(bisection_steps), 1)):
        midpoint = 0.5 * (last_valid + first_invalid)
        candidate = start + midpoint * direction
        if _normalized_state_is_in_support(candidate, coordinates=coordinates):
            last_valid = midpoint
        else:
            first_invalid = midpoint
    retained_fraction = last_valid * (1.0 - 1.0e-10)
    return start + retained_fraction * direction


class _NormalizedCoordinateFacade:
    """Minimal chart used by the public model-independent proposal API."""

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.normalized_lower = lower.copy()
        self.normalized_upper = upper.copy()

    @staticmethod
    def denormalize(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)

    @staticmethod
    def to_physical(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=float)


class _NormalizedProfileEvaluator:
    def __init__(
        self,
        evaluate_profile: Callable[[np.ndarray], SessionShapeProfile],
    ) -> None:
        self._evaluate_profile = evaluate_profile

    def evaluate_values(self, values: Sequence[float]) -> SessionShapeProfile:
        return self._evaluate_profile(np.asarray(values, dtype=float))


def propose_correlated_complete_integer_paths(
    *,
    normalized_values: Sequence[float],
    profile: SessionShapeProfile,
    evaluate_profile: Callable[[np.ndarray], SessionShapeProfile],
    normalized_lower: Sequence[float],
    normalized_upper: Sequence[float],
    candidate_count: int = 8,
    added_session_phase_variance: float = 0.0,
    config: CollapsedIntegerConfig | None = None,
) -> CorrelatedCompletePathProposalResult:
    """Generate complete path and continuous-state proposals from one chart.

    The local continuous covariance is inferred from the all-session shape
    likelihood and propagated into the complete relative-session phase
    lattice. This entry point operates in normalized coordinates, so it can
    be shared by every APOLLO timing-model adapter.
    """

    active = config or CollapsedIntegerConfig()
    center = np.asarray(normalized_values, dtype=float)
    lower = np.asarray(normalized_lower, dtype=float)
    upper = np.asarray(normalized_upper, dtype=float)
    if center.ndim != 1 or lower.shape != center.shape or upper.shape != center.shape:
        raise ValueError("Normalized center and bounds must be aligned vectors.")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(lower)) or not np.all(
        np.isfinite(upper)
    ):
        raise ValueError("Normalized center and bounds must be finite.")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Normalized bounds must contain the proposal center.")

    evaluator = _NormalizedProfileEvaluator(evaluate_profile)
    coordinates = _NormalizedCoordinateFacade(lower, upper)
    particle = _Particle(
        normalized_values=center.copy(),
        parameter_values=center.copy(),
        profile=profile,
        factors=evaluate_session_likelihood_factors(
            profile,
            hermite_order=active.hermite_order,
            integer_tail_tolerance=active.integer_tail_tolerance,
            maximum_wrap_terms=active.maximum_wrap_terms,
        ),
        log_weight=0.0,
    )
    internal, evaluations, visited_nodes = _continuous_marginalized_integer_paths(
        particle,
        evaluator=evaluator,
        coordinates=coordinates,
        candidate_count=max(int(candidate_count), 1),
        added_session_phase_variance=float(added_session_phase_variance),
        config=active,
    )
    proposals = tuple(
        CorrelatedCompletePathProposal(
            integer_offsets=item.candidate.integer_offsets,
            wraps=item.candidate.wraps,
            normalized_start=np.asarray(item.normalized_start, dtype=float).copy(),
            predictive_chi2=float(item.predictive_chi2),
            log_conditional_mass=float(item.candidate.log_conditional_mass),
            covariance_scale=float(item.covariance_scale),
            proposal_source=str(item.candidate.proposal_source),
        )
        for item in internal
    )
    return CorrelatedCompletePathProposalResult(
        proposals=proposals,
        profile_evaluations=int(evaluations),
        lattice_visited_nodes=int(visited_nodes),
    )


def _normalised_ess(log_weights: np.ndarray) -> float:
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return 0.0
    normalized = np.full(log_weights.shape, -np.inf, dtype=float)
    normalized[finite] = log_weights[finite] - logsumexp(log_weights[finite])
    weights = np.exp(normalized[finite])
    return float(1.0 / np.sum(np.square(weights)))


def _conditional_ess(
    log_weights: np.ndarray,
    log_increment: np.ndarray,
) -> float:
    normalized = np.asarray(log_weights, dtype=float) - logsumexp(log_weights)
    increment = np.asarray(log_increment, dtype=float)
    log_first_moment = float(logsumexp(normalized + increment))
    log_second_moment = float(logsumexp(normalized + 2.0 * increment))
    return float(
        len(normalized)
        * math.exp(2.0 * log_first_moment - log_second_moment)
    )


def _choose_next_beta(
    current_beta: float,
    log_factor: np.ndarray,
    log_weights: np.ndarray,
    *,
    target_ess: float,
    minimum_step: float,
    maximum_step: float,
) -> float:
    current_beta = float(current_beta)
    upper = min(1.0, current_beta + max(float(maximum_step), float(minimum_step)))
    remaining = upper - current_beta
    if remaining <= 0.0:
        return 1.0

    def ess_at(beta: float) -> float:
        return _normalised_ess(log_weights + (beta - current_beta) * log_factor)

    if ess_at(upper) >= target_ess:
        return upper
    low = 0.0
    high = remaining
    for _ in range(48):
        middle = 0.5 * (low + high)
        if ess_at(current_beta + middle) >= target_ess:
            low = middle
        else:
            high = middle
    return min(1.0, current_beta + max(low, float(minimum_step)))


def _systematic_resample(
    log_weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.exp(log_weights - logsumexp(log_weights))
    count = len(weights)
    positions = (rng.random() + np.arange(count, dtype=float)) / count
    cumulative = np.cumsum(weights)
    return np.searchsorted(cumulative, positions, side="right").astype(int)


def _clone_particle(particle: _Particle) -> _Particle:
    return _Particle(
        normalized_values=particle.normalized_values.copy(),
        parameter_values=particle.parameter_values.copy(),
        profile=particle.profile,
        factors=particle.factors,
        log_weight=float(particle.log_weight),
        log_prior=float(particle.log_prior),
        accepted_moves=int(particle.accepted_moves),
        cached_phase_variance=particle.cached_phase_variance,
        cached_wrapped_log_likelihood=particle.cached_wrapped_log_likelihood,
    )


def _mode_count(particles: Sequence[_Particle], separation: float = 0.08) -> int:
    centers: list[np.ndarray] = []
    for particle in sorted(particles, key=lambda item: item.log_weight, reverse=True):
        if all(
            float(np.linalg.norm(particle.normalized_values - center)) >= separation
            for center in centers
        ):
            centers.append(particle.normalized_values.copy())
    return len(centers)


def _log_gaussian_density(value: np.ndarray, center: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(value, dtype=float) - np.asarray(center, dtype=float)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        return float("-inf")
    solved = np.linalg.solve(covariance, delta)
    dimension = delta.size
    return float(
        -0.5
        * (
            dimension * math.log(2.0 * math.pi)
            + logdet
            + float(delta @ solved)
        )
    )


def _discover_invariant_mode_proposals(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: CollapsedIntegerConfig,
    additional_normalized_starts: Sequence[Sequence[float]] = (),
) -> tuple[_IndependentModeProposal, ...]:
    """Build exact-MH independent proposals from generic invariant modes.

    Optimization constructs only a proposal density. Acceptance or rejection
    uses the tempered exact target, preserving the posterior regardless of the
    optimizer score.
    """

    start_count = max(int(config.mode_discovery_start_count), 0)
    spin_strata = max(int(config.mode_discovery_spin_strata), 0)
    if (
        start_count == 0
        and spin_strata == 0
        and len(additional_normalized_starts) == 0
    ):
        return ()
    starts = coordinates.sample_prior(start_count, seed=int(config.random_seed) + 101)
    normalized_starts = [coordinates.normalize(row) for row in starts]
    center = coordinates.normalize(coordinates.search_center)
    if start_count > 0 or spin_strata > 0 or len(additional_normalized_starts) == 0:
        normalized_starts.append(center)
    for raw_start in additional_normalized_starts:
        start = np.asarray(raw_start, dtype=float)
        if start.shape == center.shape and np.all(np.isfinite(start)):
            normalized_starts.append(start.copy())
    lower = coordinates.normalized_lower
    upper = coordinates.normalized_upper
    names = tuple(
        str(name).upper()
        for name in getattr(coordinates, "parameter_names", ())
    )
    if spin_strata > 0 and "F0" in names and "F1" in names:
        f0_index = names.index("F0")
        f1_index = names.index("F1")
        fractions = (np.arange(spin_strata, dtype=float) + 0.5) / spin_strata
        for fraction in fractions:
            stratified = center.copy()
            # DD search coordinates define F0 at the reference epoch. Holding
            # it fixed while stratifying F1 follows the spin-trend degeneracy
            # without changing the represented reference frequency.
            stratified[f0_index] = center[f0_index]
            stratified[f1_index] = (
                lower[f1_index]
                + float(fraction) * (upper[f1_index] - lower[f1_index])
            )
            normalized_starts.append(stratified)
    interior = np.minimum(np.maximum((upper - lower) * 1.0e-9, 1.0e-9), 1.0e-5)
    records: list[tuple[float, np.ndarray, np.ndarray]] = []

    def residual_vector(values: np.ndarray) -> np.ndarray:
        if np.any(values <= lower) or np.any(values >= upper):
            return np.full(1, 1.0e12, dtype=float)
        try:
            physical = coordinates.to_physical(coordinates.denormalize(values))
            profile = evaluator.evaluate_values(physical)
            slopes = np.asarray(profile.standardized_session_slopes, dtype=float)
            orthogonal = np.asarray(
                profile.normalized_orthogonal_residuals,
                dtype=float,
            )
            shape = (
                orthogonal
                if orthogonal.size > 0
                else np.asarray(profile.normalized_residuals, dtype=float)
            )
            residual = np.concatenate((slopes, shape))
            if residual.size == 0 or not np.all(np.isfinite(residual)):
                return np.full(max(residual.size, 1), 1.0e12, dtype=float)
            return residual
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.full(1, 1.0e12, dtype=float)

    raw_proposal_std = max(
        float(config.mode_covariance_maximum_std),
        float(config.mode_covariance_minimum_std),
        1.0e-12,
    )
    raw_covariance = np.eye(center.size, dtype=float) * raw_proposal_std**2
    for start in normalized_starts:
        candidate = np.asarray(start, dtype=float)
        residual = residual_vector(candidate)
        objective = float(np.dot(residual, residual))
        if np.isfinite(objective):
            records.append((objective, candidate.copy(), raw_covariance.copy()))

    optimization_budget = max(int(config.mode_discovery_max_nfev), 0)
    for start in normalized_starts:
        if optimization_budget == 0:
            continue
        clipped = np.clip(np.asarray(start, dtype=float), lower + interior, upper - interior)
        try:
            optimized = least_squares(
                residual_vector,
                clipped,
                bounds=(lower + interior, upper - interior),
                method="trf",
                jac="2-point",
                diff_step=1.0e-4,
                max_nfev=optimization_budget,
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
            )
            values = np.asarray(optimized.x, dtype=float)
            objective = float(np.dot(residual_vector(values), residual_vector(values)))
            if not np.isfinite(objective):
                continue
            jacobian = np.asarray(optimized.jac, dtype=float)
            if jacobian.ndim != 2 or jacobian.shape[1] != values.size:
                covariance = np.eye(values.size, dtype=float) * 0.05**2
            else:
                information = jacobian.T @ jacobian
                information = 0.5 * (information + information.T)
                eigenvalues, eigenvectors = np.linalg.eigh(information)
                maximum_information = max(float(np.max(eigenvalues)), 0.0)
                information_floor = max(
                    maximum_information * 1.0e-12,
                    np.finfo(float).tiny,
                )
                minimum_variance = max(
                    float(config.mode_covariance_minimum_std),
                    1.0e-12,
                ) ** 2
                maximum_variance = max(
                    float(config.mode_covariance_maximum_std),
                    math.sqrt(minimum_variance),
                ) ** 2
                covariance_eigenvalues = np.where(
                    eigenvalues > information_floor,
                    1.0 / np.maximum(eigenvalues, information_floor),
                    maximum_variance,
                )
                covariance_eigenvalues = np.clip(
                    covariance_eigenvalues
                    * max(float(config.mode_covariance_scale), 1.0e-8) ** 2,
                    minimum_variance,
                    maximum_variance,
                )
                covariance = (
                    eigenvectors * covariance_eigenvalues
                ) @ eigenvectors.T
            records.append((objective, values, covariance))
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue

    records.sort(key=lambda item: item[0])
    selected: list[tuple[np.ndarray, np.ndarray]] = []
    for _, values, covariance in records:
        if any(float(np.linalg.norm(values - previous)) < 0.04 for previous, _ in selected):
            continue
        selected.append((values, covariance))
    selected = selected[: max(start_count + spin_strata, 1)]
    if not selected:
        return ()
    mixture_weight = (1.0 - np.clip(float(config.global_jump_probability), 0.0, 0.9)) / len(selected)
    return tuple(
        _IndependentModeProposal(
            center=values.copy(),
            covariance=covariance.copy(),
            log_mixture_weight=float(math.log(max(mixture_weight, 1.0e-300))),
        )
        for values, covariance in selected
    )


def discover_invariant_mode_charts(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: CollapsedIntegerConfig | None = None,
    additional_normalized_starts: Sequence[Sequence[float]] = (),
) -> tuple[InvariantModeSummary, ...]:
    """Return all-session phase+slope charts before integer conditioning."""

    active = config or CollapsedIntegerConfig()
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    summaries: list[InvariantModeSummary] = []
    for proposal in _discover_invariant_mode_proposals(
        evaluator=evaluator,
        coordinates=coordinates,
        config=active,
        additional_normalized_starts=additional_normalized_starts,
    ):
        particle = _evaluate_particle(
            evaluator,
            coordinates,
            np.asarray(proposal.center, dtype=float),
            active,
        )
        if particle is None:
            continue
        boundary_margin = float(
            np.min(
                np.minimum(
                    particle.normalized_values - lower,
                    upper - particle.normalized_values,
                )
            )
        )
        summaries.append(
            InvariantModeSummary(
                normalized_values=particle.normalized_values.copy(),
                parameter_values=particle.parameter_values.copy(),
                normalized_covariance=np.asarray(
                    proposal.covariance,
                    dtype=float,
                ).copy(),
                invariant_log_likelihood=float(
                    particle.factors.invariant_log_likelihood
                ),
                direct_shape_chi2=float(particle.factors.direct_shape_chi2),
                normalized_boundary_margin=boundary_margin,
            )
        )
    summaries.sort(key=lambda item: item.direct_shape_chi2)
    return tuple(summaries)


def _sample_mode_proposal(
    proposals: Sequence[_IndependentModeProposal],
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    weights = np.asarray([math.exp(item.log_mixture_weight) for item in proposals], dtype=float)
    weights /= np.sum(weights)
    index = int(rng.choice(len(proposals), p=weights))
    selected = proposals[index]
    sample = rng.multivariate_normal(selected.center, selected.covariance)
    return np.asarray(sample, dtype=float), _mode_proposal_log_density(sample, proposals)


def _mode_proposal_log_density(
    value: np.ndarray,
    proposals: Sequence[_IndependentModeProposal],
) -> float:
    return _finite_logsumexp(
        [
            proposal.log_mixture_weight
            + _log_gaussian_density(value, proposal.center, proposal.covariance)
            for proposal in proposals
        ]
    )


def _bounded_mode_proposals(
    proposals: Sequence[_IndependentModeProposal],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[_BoundedModeProposal, ...]:
    if not proposals:
        return ()
    widths = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    raw_log_weights = np.asarray(
        [proposal.log_mixture_weight for proposal in proposals],
        dtype=float,
    )
    normalized_log_weights = raw_log_weights - logsumexp(raw_log_weights)
    output: list[_BoundedModeProposal] = []
    for proposal, log_weight in zip(
        proposals,
        normalized_log_weights,
        strict=True,
    ):
        fraction = np.clip(
            (np.asarray(proposal.center, dtype=float) - lower) / widths,
            1.0e-8,
            1.0 - 1.0e-8,
        )
        transformed_center = np.log(fraction) - np.log1p(-fraction)
        derivative = 1.0 / (widths * fraction * (1.0 - fraction))
        transformed_covariance = (
            derivative[:, None]
            * np.asarray(proposal.covariance, dtype=float)
            * derivative[None, :]
        )
        transformed_covariance = 0.5 * (
            transformed_covariance + transformed_covariance.T
        )
        eigenvalues, eigenvectors = np.linalg.eigh(transformed_covariance)
        eigenvalues = np.clip(eigenvalues, 1.0e-16, 25.0)
        transformed_covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        output.append(
            _BoundedModeProposal(
                transformed_center=transformed_center,
                transformed_covariance=transformed_covariance,
                log_mixture_weight=float(log_weight),
            )
        )
    return tuple(output)


def _sample_bounded_mode_proposal(
    proposals: Sequence[_BoundedModeProposal],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = np.exp(
        np.asarray([proposal.log_mixture_weight for proposal in proposals], dtype=float)
    )
    index = int(rng.choice(len(proposals), p=probabilities))
    selected = proposals[index]
    transformed = rng.multivariate_normal(
        selected.transformed_center,
        selected.transformed_covariance,
    )
    return np.asarray(lower, dtype=float) + (
        np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    ) * expit(transformed)


def _bounded_mode_log_density(
    value: np.ndarray,
    proposals: Sequence[_BoundedModeProposal],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    widths = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    fraction = (np.asarray(value, dtype=float) - lower) / widths
    if np.any(fraction <= 0.0) or np.any(fraction >= 1.0):
        return float("-inf")
    transformed = np.log(fraction) - np.log1p(-fraction)
    log_jacobian = -float(
        np.sum(np.log(widths) + np.log(fraction) + np.log1p(-fraction))
    )
    return float(
        log_jacobian
        + _finite_logsumexp(
            [
                proposal.log_mixture_weight
                + _log_gaussian_density(
                    transformed,
                    proposal.transformed_center,
                    proposal.transformed_covariance,
                )
                for proposal in proposals
            ]
        )
    )


def _evaluate_particle(
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    normalized_values: np.ndarray,
    config: CollapsedIntegerConfig,
) -> _Particle | None:
    lower = coordinates.normalized_lower
    upper = coordinates.normalized_upper
    if np.any(normalized_values <= lower) or np.any(normalized_values >= upper):
        return None
    try:
        parameter_values = coordinates.to_physical(
            coordinates.denormalize(normalized_values)
        )
        prior_density = getattr(
            coordinates,
            "log_normalized_prior_density",
            None,
        )
        log_prior = (
            float(prior_density(normalized_values))
            if callable(prior_density)
            else 0.0
        )
        if not np.isfinite(log_prior):
            return None
        profile = evaluator.evaluate_values(parameter_values)
        factors = evaluate_session_likelihood_factors(
            profile,
            hermite_order=config.hermite_order,
            integer_tail_tolerance=config.integer_tail_tolerance,
            maximum_wrap_terms=config.maximum_wrap_terms,
        )
    except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    if not np.isfinite(factors.invariant_log_likelihood + factors.wrapped_log_likelihood):
        return None
    return _Particle(
        normalized_values=np.asarray(normalized_values, dtype=float).copy(),
        parameter_values=np.asarray(parameter_values, dtype=float).copy(),
        profile=profile,
        factors=factors,
        log_weight=0.0,
        log_prior=log_prior,
    )


def _phase_noise_bridge_variance(
    progress: float,
    config: CollapsedIntegerConfig,
    *,
    reference_measurement_variance: float | None = None,
) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    if progress >= 1.0:
        return 0.0
    start_variance = max(float(config.phase_noise_start_cycles), 0.0) ** 2
    parameterization = str(config.phase_noise_parameterization).strip().lower()
    if parameterization == "sigma":
        sigma = math.sqrt(start_variance) * (
            (1.0 - progress)
            ** max(float(config.phase_noise_schedule_power), 1.0e-6)
        )
        return float(sigma * sigma)
    if parameterization != "measurement_precision":
        raise ValueError(
            "Collapsed phase_noise_parameterization must be 'sigma' or "
            "'measurement_precision'."
        )
    if reference_measurement_variance is None:
        raise ValueError(
            "Measurement-precision phase annealing requires a positive "
            "reference measurement variance."
        )
    measurement_variance = max(
        float(reference_measurement_variance),
        np.finfo(float).eps,
    )
    start_precision = 1.0 / (measurement_variance + start_variance)
    exact_precision = 1.0 / measurement_variance
    precision_progress = progress ** max(
        float(config.phase_noise_schedule_power),
        1.0e-6,
    )
    effective_precision = (
        (1.0 - precision_progress) * start_precision
        + precision_progress * exact_precision
    )
    return float(max(1.0 / effective_precision - measurement_variance, 0.0))


def _particle_wrapped_log_likelihood(
    particle: _Particle,
    added_phase_variance: float,
    config: CollapsedIntegerConfig,
) -> float:
    variance = max(float(added_phase_variance), 0.0)
    if variance <= np.finfo(float).eps:
        return float(particle.factors.wrapped_log_likelihood)
    if (
        particle.cached_phase_variance == variance
        and particle.cached_wrapped_log_likelihood is not None
    ):
        return float(particle.cached_wrapped_log_likelihood)
    wrapped, _ = collapsed_wrapped_log_likelihood(
        particle.profile,
        hermite_order=config.hermite_order,
        integer_tail_tolerance=config.integer_tail_tolerance,
        maximum_wrap_terms=config.maximum_wrap_terms,
        added_session_phase_variance=variance,
    )
    particle.cached_phase_variance = variance
    particle.cached_wrapped_log_likelihood = float(wrapped)
    return float(wrapped)


def _target_log_value(
    particle: _Particle,
    invariant_beta: float,
    phase_beta: float,
    *,
    config: CollapsedIntegerConfig,
    added_phase_variance: float = 0.0,
) -> float:
    return float(
        particle.log_prior
        + invariant_beta * particle.factors.invariant_log_likelihood
        + phase_beta
        * _particle_wrapped_log_likelihood(
            particle,
            added_phase_variance,
            config,
        )
    )


def _phase_metric_covariances(
    particles: Sequence[_Particle],
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    added_phase_variance: float,
    config: CollapsedIntegerConfig,
) -> tuple[np.ndarray, ...]:
    chart_count = max(int(config.phase_metric_chart_count), 0)
    if chart_count == 0 or not particles:
        return ()
    ranked = sorted(particles, key=lambda item: item.log_weight, reverse=True)
    representatives: list[_Particle] = []
    for particle in ranked:
        if all(
            float(
                np.linalg.norm(
                    particle.normalized_values - selected.normalized_values
                )
            )
            >= 0.03
            for selected in representatives
        ):
            representatives.append(particle)
        if len(representatives) >= chart_count:
            break
    if len(representatives) < chart_count:
        for particle in ranked:
            if any(particle is selected for selected in representatives):
                continue
            representatives.append(particle)
            if len(representatives) >= chart_count:
                break

    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    requested_step = max(
        float(config.phase_metric_finite_difference_step),
        1.0e-10,
    )
    minimum_variance = max(float(config.phase_metric_minimum_std), 1.0e-12) ** 2
    maximum_variance = max(
        float(config.phase_metric_maximum_std),
        math.sqrt(minimum_variance),
    ) ** 2
    scale_squared = max(float(config.phase_metric_scale), 1.0e-8) ** 2
    output: list[np.ndarray] = []

    def relative_offsets(normalized_values: np.ndarray) -> np.ndarray:
        parameter_values = coordinates.to_physical(
            coordinates.denormalize(normalized_values)
        )
        profile = evaluator.evaluate_values(parameter_values)
        offsets = np.asarray(profile.session_offsets, dtype=float)
        if offsets.ndim != 1 or offsets.size <= 1 or not np.all(np.isfinite(offsets)):
            raise ValueError("Phase metric requires finite session offsets.")
        return offsets[1:] - offsets[0]

    for representative in representatives:
        center = np.asarray(representative.normalized_values, dtype=float)
        dimension = center.size
        jacobian = np.empty((representative.profile.session_count - 1, dimension))
        valid = True
        for parameter_index in range(dimension):
            left_step = min(requested_step, 0.45 * (center[parameter_index] - lower[parameter_index]))
            right_step = min(requested_step, 0.45 * (upper[parameter_index] - center[parameter_index]))
            if left_step <= 0.0 or right_step <= 0.0:
                valid = False
                break
            left = center.copy()
            right = center.copy()
            left[parameter_index] -= left_step
            right[parameter_index] += right_step
            try:
                jacobian[:, parameter_index] = (
                    relative_offsets(right) - relative_offsets(left)
                ) / (left_step + right_step)
            except (
                ArithmeticError,
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                valid = False
                break
        if not valid or not np.all(np.isfinite(jacobian)):
            continue
        variances = (
            np.asarray(representative.profile.session_offset_variances, dtype=float)
            + max(float(added_phase_variance), 0.0)
        )
        relative_covariance = np.diag(variances[1:]) + variances[0]
        try:
            information = jacobian.T @ np.linalg.solve(relative_covariance, jacobian)
            information = 0.5 * (information + information.T)
            eigenvalues, eigenvectors = np.linalg.eigh(information)
        except (ValueError, np.linalg.LinAlgError):
            continue
        information_floor = max(
            float(np.max(eigenvalues)) * 1.0e-12,
            np.finfo(float).tiny,
        )
        covariance_eigenvalues = np.where(
            eigenvalues > information_floor,
            1.0 / np.maximum(eigenvalues, information_floor),
            maximum_variance,
        )
        covariance_eigenvalues = np.clip(
            covariance_eigenvalues * scale_squared,
            minimum_variance,
            maximum_variance,
        )
        covariance = (eigenvectors * covariance_eigenvalues) @ eigenvectors.T
        output.append(0.5 * (covariance + covariance.T))
    return tuple(output)


def _transport_bridge_anchors(
    anchors: Sequence[_Particle],
    origins: Sequence[_Particle],
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    added_phase_variance: float,
    config: CollapsedIntegerConfig,
) -> tuple[list[_Particle], int]:
    """Track full-shape posterior modes while phase noise is annealed."""

    if not anchors:
        return [], 0
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    interior = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
    bounds = list(zip(lower + interior, upper - interior, strict=True))
    evaluations = 0
    transported: list[_Particle] = []

    for index, anchor in enumerate(anchors):
        starts = [anchor]
        if index < len(origins) and float(
            np.linalg.norm(
                origins[index].normalized_values - anchor.normalized_values
            )
        ) > 1.0e-10:
            starts.append(origins[index])
        best_particle = anchor
        best_target = _target_log_value(
            anchor,
            1.0,
            1.0,
            config=config,
            added_phase_variance=added_phase_variance,
        )
        for start in starts:
            cache: dict[bytes, tuple[float, _Particle | None]] = {}

            def objective(values: np.ndarray) -> float:
                nonlocal evaluations
                array = np.asarray(values, dtype=float)
                key = array.tobytes()
                if key not in cache:
                    particle = _evaluate_particle(
                        evaluator,
                        coordinates,
                        array,
                        config,
                    )
                    evaluations += 1
                    if particle is None:
                        cache[key] = (1.0e300, None)
                    else:
                        cache[key] = (
                            -_target_log_value(
                                particle,
                                1.0,
                                1.0,
                                config=config,
                                added_phase_variance=added_phase_variance,
                            ),
                            particle,
                        )
                return float(cache[key][0])

            try:
                optimized = minimize(
                    objective,
                    np.clip(start.normalized_values, lower + interior, upper - interior),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": max(int(config.bridge_anchor_maxiter), 1),
                        "maxfun": max(
                            int(config.bridge_anchor_maxiter)
                            * (start.normalized_values.size + 1),
                            16,
                        ),
                        "ftol": 1.0e-10,
                        "gtol": 1.0e-6,
                        "eps": max(
                            float(config.bridge_anchor_finite_difference_step),
                            1.0e-10,
                        ),
                    },
                )
                candidate = _evaluate_particle(
                    evaluator,
                    coordinates,
                    np.asarray(optimized.x, dtype=float),
                    config,
                )
                evaluations += 1
            except (
                ArithmeticError,
                FloatingPointError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                candidate = None
            if candidate is None:
                continue
            candidate_target = _target_log_value(
                candidate,
                1.0,
                1.0,
                config=config,
                added_phase_variance=added_phase_variance,
            )
            if candidate_target > best_target:
                best_particle = candidate
                best_target = candidate_target
        transported.append(best_particle)
    return transported, evaluations


def _anchor_phase_noise_schedule(
    anchors: Sequence[_Particle],
    config: CollapsedIntegerConfig,
) -> tuple[float, ...]:
    if not anchors:
        return ()
    start = max(float(config.phase_noise_start_cycles), 0.0)
    if start <= np.finfo(float).eps:
        return (0.0,)
    measurement_sigmas = np.concatenate(
        [
            np.sqrt(
                np.maximum(
                    np.asarray(anchor.profile.session_offset_variances, dtype=float),
                    0.0,
                )
            )
            for anchor in anchors
        ]
    )
    finite_positive = measurement_sigmas[
        np.isfinite(measurement_sigmas) & (measurement_sigmas > 0.0)
    ]
    measurement_floor = (
        0.5 * float(np.median(finite_positive))
        if finite_positive.size
        else start * 1.0e-6
    )
    floor = max(
        min(measurement_floor, 0.5 * start),
        start * 1.0e-8,
        np.finfo(float).eps,
    )
    step_count = max(int(config.bridge_anchor_schedule_steps), 2)
    positive = np.geomspace(start, floor, num=step_count)
    return tuple(float(value) for value in positive[1:]) + (0.0,)


def _branch_bridge_anchor_modes(
    parents: Sequence[_Particle],
    transported: Sequence[_Particle],
    *,
    invariant_origins: Sequence[_Particle] = (),
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    added_phase_variance: float,
    config: CollapsedIntegerConfig,
) -> tuple[
    list[_Particle],
    list[_Particle],
    list[_AcquiredAnchorPath],
    int,
    int,
    dict[str, int],
]:
    child_limit = max(int(config.bridge_anchor_children_per_parent), 0)
    portfolio_limit = max(int(config.bridge_anchor_portfolio_count), 1)
    if len(parents) != len(transported):
        raise ValueError("Anchor parents and transported modes must remain aligned.")
    candidates = [_clone_particle(particle) for particle in transported]
    generated: list[_Particle] = []
    acquired_paths: list[_AcquiredAnchorPath] = []
    evaluations = 0
    path_proposals = 0
    lattice_evaluations = 0
    lattice_visited_nodes = 0
    lattice_failed_charts = 0

    acquisition_sources: list[tuple[str, _Particle]] = []

    def add_source(family: str, particle: _Particle) -> None:
        if any(
            float(
                np.linalg.norm(
                    particle.normalized_values - previous.normalized_values
                )
            )
            < 1.0e-9
            for _, previous in acquisition_sources
        ):
            return
        acquisition_sources.append((family, particle))

    for source_index, source in enumerate(transported):
        add_source(f"continuation-{source_index}", source)
    for source_index, source in enumerate(invariant_origins):
        add_source(f"invariant-reacquisition-{source_index}", source)

    for acquisition_family, source in acquisition_sources:
        try:
            proposals, local_lattice_evaluations, local_visited_nodes = (
                _continuous_marginalized_integer_paths(
                    source,
                    evaluator=evaluator,
                    coordinates=coordinates,
                    candidate_count=max(child_limit, 1),
                    added_session_phase_variance=added_phase_variance,
                    config=config,
                )
            )
        except (
            ArithmeticError,
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            lattice_failed_charts += 1
            continue
        lattice_evaluations += local_lattice_evaluations
        lattice_visited_nodes += local_visited_nodes
        evaluations += local_lattice_evaluations
        if not proposals:
            continue
        grouped_proposals: dict[float, list[_CorrelatedIntegerPathProposal]] = {}
        for proposal in proposals:
            grouped_proposals.setdefault(proposal.covariance_scale, []).append(
                proposal
            )
        retained_proposals: list[_CorrelatedIntegerPathProposal] = []
        for scale_proposals in grouped_proposals.values():
            ranked_scale = sorted(
                scale_proposals,
                key=lambda item: item.candidate.log_conditional_mass,
                reverse=True,
            )
            maximum_log_mass = float(
                ranked_scale[0].candidate.log_conditional_mass
            )
            retained_proposals.extend(
                proposal
                for proposal in ranked_scale[:child_limit]
                if float(
                    proposal.candidate.log_conditional_mass - maximum_log_mass
                )
                >= float(config.bridge_anchor_child_log_mass_floor)
            )
        for proposal in retained_proposals:
            path_proposals += 1
            values, _, particle, local_evaluations = _optimize_fixed_integer_path(
                evaluator=evaluator,
                coordinates=coordinates,
                start=proposal.normalized_start,
                integer_offsets=proposal.candidate.integer_offsets,
                config=config,
                added_phase_variance=added_phase_variance,
                maximum_iterations=config.bridge_anchor_maxiter,
                finite_difference_step=config.bridge_anchor_finite_difference_step,
            )
            evaluations += local_evaluations
            if particle is None or not np.all(np.isfinite(values)):
                continue
            generated.append(particle)
            candidates.append(particle)
            bridge_log_target = float(
                particle.factors.invariant_log_likelihood
                + fixed_integer_log_likelihood(
                    particle.profile,
                    proposal.candidate.integer_offsets,
                    added_session_phase_variance=added_phase_variance,
                )
            )
            exact_log_target = float(
                particle.factors.invariant_log_likelihood
                + fixed_integer_log_likelihood(
                    particle.profile,
                    proposal.candidate.integer_offsets,
                )
            )
            acquired_paths.append(
                _AcquiredAnchorPath(
                    candidate=proposal.candidate,
                    particle=_clone_particle(particle),
                    bridge_log_target=bridge_log_target,
                    exact_log_target=exact_log_target,
                    phase_noise_sigma_cycles=math.sqrt(
                        max(float(added_phase_variance), 0.0)
                    ),
                    acquisition_family=acquisition_family,
                )
            )

    def bridge_score(particle: _Particle) -> float:
        return _target_log_value(
            particle,
            1.0,
            1.0,
            config=config,
            added_phase_variance=added_phase_variance,
        )

    ranked = sorted(candidates, key=bridge_score, reverse=True)
    selected: list[_Particle] = []
    separation = max(float(config.bridge_anchor_mode_separation), 0.0)

    def append_if_distinct(particle: _Particle) -> bool:
        if any(
            float(
                np.linalg.norm(
                    particle.normalized_values - previous.normalized_values
                )
            )
            < separation
            for previous in selected
        ):
            return False
        selected.append(_clone_particle(particle))
        return True

    # Invariant modes are a persistent proposal bank, not posterior samples.
    # Reserving them prevents an earlier noisy bridge from irreversibly
    # deleting the continuous orbit chart needed by a later complete path.
    for particle in invariant_origins:
        append_if_distinct(particle)
        if len(selected) >= portfolio_limit:
            return (
                selected,
                generated,
                acquired_paths,
                evaluations,
                path_proposals,
                {
                    "linearization_evaluations": lattice_evaluations,
                    "visited_nodes": lattice_visited_nodes,
                    "failed_charts": lattice_failed_charts,
                    "acquisition_sources": len(acquisition_sources),
                    "invariant_reacquisition_sources": len(invariant_origins),
                },
            )
    for particle in ranked:
        append_if_distinct(particle)
        if len(selected) >= portfolio_limit:
            break
    return (
        selected,
        generated,
        acquired_paths,
        evaluations,
        path_proposals,
        {
            "linearization_evaluations": lattice_evaluations,
            "visited_nodes": lattice_visited_nodes,
            "failed_charts": lattice_failed_charts,
            "acquisition_sources": len(acquisition_sources),
            "invariant_reacquisition_sources": len(invariant_origins),
        },
    )


def _run_full_shape_anchor_continuation(
    origins: Sequence[_Particle],
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: CollapsedIntegerConfig,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[
    list[_Particle],
    list[_Particle],
    list[_AcquiredAnchorPath],
    tuple[dict[str, Any], ...],
]:
    current = [_clone_particle(anchor) for anchor in origins]
    archive = [_clone_particle(anchor) for anchor in origins]
    acquired_archive: list[_AcquiredAnchorPath] = []
    history: list[dict[str, Any]] = []
    schedule = _anchor_phase_noise_schedule(origins, config)
    last_branch_step: int | None = None
    for step_index, sigma in enumerate(schedule, start=1):
        added_variance = float(sigma * sigma)
        parents = current
        transported, transport_evaluations = _transport_bridge_anchors(
            current,
            origins,
            evaluator=evaluator,
            coordinates=coordinates,
            added_phase_variance=added_variance,
            config=config,
        )
        branch_evaluations = 0
        path_proposals = 0
        generated: list[_Particle] = []
        acquired_paths: list[_AcquiredAnchorPath] = []
        lattice_diagnostics = {
            "linearization_evaluations": 0,
            "visited_nodes": 0,
            "failed_charts": 0,
            "acquisition_sources": 0,
            "invariant_reacquisition_sources": 0,
        }
        should_branch = bool(
            int(config.bridge_anchor_children_per_parent) > 0
            and sigma <= max(
                float(config.bridge_anchor_branch_start_sigma_cycles),
                0.0,
            )
            and (
                last_branch_step is None
                or step_index - last_branch_step
                >= max(int(config.bridge_anchor_branch_stride), 1)
                or sigma <= np.finfo(float).eps
            )
        )
        if should_branch:
            (
                current,
                generated,
                acquired_paths,
                branch_evaluations,
                path_proposals,
                lattice_diagnostics,
            ) = (
                _branch_bridge_anchor_modes(
                    parents,
                    transported,
                    invariant_origins=origins,
                    evaluator=evaluator,
                    coordinates=coordinates,
                    added_phase_variance=added_variance,
                    config=config,
                )
            )
            last_branch_step = step_index
        else:
            portfolio_limit = max(int(config.bridge_anchor_portfolio_count), 1)
            current = []

            def append_current(particle: _Particle) -> None:
                separation = max(float(config.bridge_anchor_mode_separation), 0.0)
                if any(
                    float(
                        np.linalg.norm(
                            particle.normalized_values - previous.normalized_values
                        )
                    )
                    < separation
                    for previous in current
                ):
                    return
                current.append(_clone_particle(particle))

            for origin in origins:
                append_current(origin)
            for particle in sorted(
                transported,
                key=lambda item: _target_log_value(
                    item,
                    1.0,
                    1.0,
                    config=config,
                    added_phase_variance=added_variance,
                ),
                reverse=True,
            ):
                append_current(particle)
                if len(current) >= portfolio_limit:
                    break
        acquired_archive.extend(acquired_paths)
        for anchor in [*current, *generated]:
            if any(
                float(
                    np.linalg.norm(
                        anchor.normalized_values - previous.normalized_values
                    )
                )
                < 1.0e-9
                for previous in archive
            ):
                continue
            archive.append(_clone_particle(anchor))
        summary = {
            "factor": "anchor_phase_noise_bridge",
            "anchor_step": int(step_index),
            "anchor_step_count": len(schedule),
            "phase_noise_sigma_cycles": float(sigma),
            "phase_noise_variance": added_variance,
            "anchor_evaluations": int(
                transport_evaluations + branch_evaluations
            ),
            "anchor_transport_evaluations": int(transport_evaluations),
            "anchor_branch_evaluations": int(branch_evaluations),
            "anchor_branch_active": should_branch,
            "anchor_branch_strategy": (
                "continuous_marginalized_correlated_lattice"
                if should_branch
                else "none"
            ),
            "anchor_path_proposals": int(path_proposals),
            "anchor_lattice_linearization_evaluations": int(
                lattice_diagnostics["linearization_evaluations"]
            ),
            "anchor_lattice_visited_nodes": int(
                lattice_diagnostics["visited_nodes"]
            ),
            "anchor_lattice_failed_charts": int(
                lattice_diagnostics["failed_charts"]
            ),
            "anchor_acquisition_source_count": int(
                lattice_diagnostics["acquisition_sources"]
            ),
            "anchor_invariant_reacquisition_source_count": int(
                lattice_diagnostics["invariant_reacquisition_sources"]
            ),
            "anchor_acquired_complete_path_count": len(acquired_paths),
            "anchor_persistent_complete_path_count": len(acquired_archive),
            "active_anchor_count": len(current),
            "anchor_archive_count": len(archive),
            "best_anchor_invariant_chi2": float(min(
                (-2.0 * anchor.factors.invariant_log_likelihood for anchor in current),
                default=float("inf"),
            )),
            "best_anchor_bridge_wrapped_log_likelihood": float(max(
                (
                    _particle_wrapped_log_likelihood(
                        anchor,
                        added_variance,
                        config,
                    )
                    for anchor in current
                ),
                default=float("-inf"),
            )),
            "best_anchor_wrapped_log_likelihood": float(max(
                (anchor.factors.wrapped_log_likelihood for anchor in current),
                default=float("-inf"),
            )),
            "best_anchor_bridge_joint_log_target": float(max(
                (
                    anchor.factors.invariant_log_likelihood
                    + _particle_wrapped_log_likelihood(
                        anchor,
                        added_variance,
                        config,
                    )
                    for anchor in current
                ),
                default=float("-inf"),
            )),
            "best_anchor_exact_joint_log_target": float(max(
                [
                    *(
                        anchor.factors.invariant_log_likelihood
                        + anchor.factors.wrapped_log_likelihood
                        for anchor in archive
                    ),
                    *(path.exact_log_target for path in acquired_archive),
                ],
                default=float("-inf"),
            )),
        }
        history.append(summary)
        if progress_callback is not None:
            progress_callback(summary)
    return current, archive, acquired_archive, tuple(history)


def _acquire_conditional_clock_paths(
    parents: Sequence[_Particle],
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: CollapsedIntegerConfig,
) -> tuple[list[_AcquiredAnchorPath], dict[str, Any]]:
    """Enumerate shared-clock/path modes inside each physical shape island."""

    parameter_names = tuple(
        str(value).upper()
        for value in getattr(
            coordinates,
            "parameter_names",
            getattr(evaluator, "parameter_names", ()),
        )
    )
    requested_names = tuple(
        str(value).upper() for value in config.conditional_clock_parameter_names
    )
    coordinate_indices = tuple(
        parameter_names.index(name)
        for name in requested_names
        if name in parameter_names
    )
    if not config.conditional_clock_lattice or not coordinate_indices:
        return [], {
            "factor": "conditional_clock_lattice",
            "enabled": False,
            "parameter_names": tuple(),
            "parent_count": 0,
            "proposal_count": 0,
            "acquired_path_count": 0,
            "profile_evaluations": 0,
            "lattice_visited_nodes": 0,
        }

    ranked_parents = sorted(
        parents,
        key=lambda item: item.factors.invariant_log_likelihood,
        reverse=True,
    )
    selected_parents: list[_Particle] = []
    for parent in ranked_parents:
        if any(
            float(np.linalg.norm(parent.normalized_values - previous.normalized_values))
            < 1.0e-9
            for previous in selected_parents
        ):
            continue
        selected_parents.append(parent)
        if len(selected_parents) >= max(
            int(config.conditional_clock_parent_count),
            1,
        ):
            break

    acquisitions: list[_AcquiredAnchorPath] = []
    evaluations = 0
    visited_nodes = 0
    proposal_count = 0
    failed_parents = 0
    for parent_index, parent in enumerate(selected_parents):
        try:
            proposals, local_evaluations, local_nodes = (
                _continuous_marginalized_integer_paths(
                    parent,
                    evaluator=evaluator,
                    coordinates=coordinates,
                    candidate_count=max(
                        int(config.conditional_clock_candidate_count),
                        1,
                    ),
                    added_session_phase_variance=0.0,
                    config=config,
                    coordinate_indices=coordinate_indices,
                )
            )
        except (
            ArithmeticError,
            FloatingPointError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            failed_parents += 1
            continue
        evaluations += local_evaluations
        visited_nodes += local_nodes
        for proposal in proposals:
            proposal_count += 1
            normalized, log_target, particle, local_evaluations = (
                _optimize_fixed_integer_path(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    start=proposal.normalized_start,
                    integer_offsets=proposal.candidate.integer_offsets,
                    config=config,
                    maximum_iterations=config.conditional_clock_maxiter,
                    coordinate_indices=coordinate_indices,
                )
            )
            evaluations += local_evaluations
            if particle is None or not np.isfinite(log_target):
                continue
            candidate = IntegerPathCandidate(
                integer_offsets=proposal.candidate.integer_offsets,
                wraps=proposal.candidate.wraps,
                log_abs_likelihood=proposal.candidate.log_abs_likelihood,
                log_conditional_mass=proposal.candidate.log_conditional_mass,
                proposal_source=(
                    "conditional-clock:"
                    + str(proposal.candidate.proposal_source)
                ),
            )
            acquisitions.append(
                _AcquiredAnchorPath(
                    candidate=candidate,
                    particle=_clone_particle(particle),
                    bridge_log_target=float(log_target),
                    exact_log_target=float(log_target),
                    phase_noise_sigma_cycles=0.0,
                    acquisition_family=f"conditional-clock-{parent_index:04d}",
                )
            )

    acquisitions.sort(key=lambda item: item.exact_log_target, reverse=True)
    retained: list[_AcquiredAnchorPath] = []

    def retain(acquisition: _AcquiredAnchorPath) -> None:
        if any(
            acquisition.candidate.integer_offsets
            == previous.candidate.integer_offsets
            and float(
                np.linalg.norm(
                    acquisition.particle.normalized_values
                    - previous.particle.normalized_values
                )
            )
            < 1.0e-8
            for previous in retained
        ):
            return
        retained.append(acquisition)

    represented_families: set[str] = set()
    for acquisition in acquisitions:
        if acquisition.acquisition_family in represented_families:
            continue
        retain(acquisition)
        represented_families.add(acquisition.acquisition_family)
        if len(retained) >= max(int(config.maximum_paths), 1):
            break
    for acquisition in acquisitions:
        if len(retained) >= max(int(config.maximum_paths), 1):
            break
        retain(acquisition)
    best_exact = max(
        (item.exact_log_target for item in retained),
        default=float("-inf"),
    )
    return retained, {
        "factor": "conditional_clock_lattice",
        "enabled": True,
        "information_order": str(config.tempering_order).strip().lower(),
        "parameter_names": tuple(
            parameter_names[index] for index in coordinate_indices
        ),
        "parent_count": len(selected_parents),
        "failed_parent_count": int(failed_parents),
        "proposal_count": int(proposal_count),
        "acquired_path_count": len(retained),
        "profile_evaluations": int(evaluations),
        "lattice_visited_nodes": int(visited_nodes),
        "best_exact_log_target": float(best_exact),
    }


def _rejuvenate(
    particles: list[_Particle],
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    invariant_beta: float,
    phase_beta: float,
    added_phase_variance: float = 0.0,
    config: CollapsedIntegerConfig,
    rng: np.random.Generator,
    independent_proposals: Sequence[_IndependentModeProposal] = (),
    phase_metric_covariances: Sequence[np.ndarray] = (),
) -> int:
    accepted = 0
    lower = coordinates.normalized_lower
    upper = coordinates.normalized_upper
    ensemble = np.vstack(
        [np.asarray(particle.normalized_values, dtype=float) for particle in particles]
    )
    dimension = int(ensemble.shape[1])
    differential_scale = float(config.differential_scale)
    if not np.isfinite(differential_scale) or differential_scale <= 0.0:
        differential_scale = 2.38 / math.sqrt(max(2 * dimension, 1))
    for particle in particles:
        for _ in range(max(int(config.move_steps_per_tempering_step), 0)):
            if independent_proposals and rng.random() < np.clip(
                float(config.mode_jump_probability), 0.0, 1.0
            ):
                proposal_values, proposal_log_q = _sample_mode_proposal(
                    independent_proposals,
                    rng,
                )
                if np.any(proposal_values <= lower) or np.any(proposal_values >= upper):
                    continue
                proposal = _evaluate_particle(
                    evaluator,
                    coordinates,
                    proposal_values,
                    config,
                )
                if proposal is None:
                    continue
                current_log_q = _mode_proposal_log_density(
                    particle.normalized_values,
                    independent_proposals,
                )
                log_alpha = (
                    _target_log_value(
                        proposal,
                        invariant_beta,
                        phase_beta,
                        config=config,
                        added_phase_variance=added_phase_variance,
                    )
                    - _target_log_value(
                        particle,
                        invariant_beta,
                        phase_beta,
                        config=config,
                        added_phase_variance=added_phase_variance,
                    )
                    + current_log_q
                    - proposal_log_q
                )
                if math.log(rng.random()) < min(0.0, log_alpha):
                    particle.normalized_values = proposal.normalized_values
                    particle.parameter_values = proposal.parameter_values
                    particle.profile = proposal.profile
                    particle.factors = proposal.factors
                    particle.log_prior = proposal.log_prior
                    particle.cached_phase_variance = proposal.cached_phase_variance
                    particle.cached_wrapped_log_likelihood = (
                        proposal.cached_wrapped_log_likelihood
                    )
                    particle.accepted_moves += 1
                    accepted += 1
                continue
            if phase_metric_covariances and rng.random() < np.clip(
                float(config.phase_metric_move_probability),
                0.0,
                1.0,
            ):
                metric_covariance = phase_metric_covariances[
                    int(rng.integers(len(phase_metric_covariances)))
                ]
                proposal_values = particle.normalized_values + rng.multivariate_normal(
                    np.zeros(dimension, dtype=float),
                    metric_covariance,
                )
            elif ensemble.shape[0] >= 3 and rng.random() < np.clip(
                float(config.differential_move_probability), 0.0, 1.0
            ):
                left, right = rng.choice(ensemble.shape[0], size=2, replace=False)
                proposal_values = (
                    particle.normalized_values
                    + differential_scale * (ensemble[int(left)] - ensemble[int(right)])
                    + rng.normal(
                        0.0,
                        max(float(config.differential_jitter), 0.0),
                        size=dimension,
                    )
                )
            elif rng.random() < np.clip(
                float(config.global_jump_probability), 0.0, 1.0
            ):
                proposal_values = rng.uniform(lower, upper)
            else:
                proposal_values = particle.normalized_values + rng.normal(
                    0.0,
                    max(float(config.proposal_scale), 1.0e-8),
                    size=particle.normalized_values.size,
                )
            proposal = _evaluate_particle(
                evaluator,
                coordinates,
                proposal_values,
                config,
            )
            if proposal is None:
                continue
            log_alpha = _target_log_value(
                proposal,
                invariant_beta,
                phase_beta,
                config=config,
                added_phase_variance=added_phase_variance,
            ) - _target_log_value(
                particle,
                invariant_beta,
                phase_beta,
                config=config,
                added_phase_variance=added_phase_variance,
            )
            if math.log(rng.random()) < min(0.0, log_alpha):
                particle.normalized_values = proposal.normalized_values
                particle.parameter_values = proposal.parameter_values
                particle.profile = proposal.profile
                particle.factors = proposal.factors
                particle.log_prior = proposal.log_prior
                particle.cached_phase_variance = proposal.cached_phase_variance
                particle.cached_wrapped_log_likelihood = (
                    proposal.cached_wrapped_log_likelihood
                )
                particle.accepted_moves += 1
                accepted += 1
    return accepted


def run_collapsed_integer_smc(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: CollapsedIntegerConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mode_discovery_starts: Sequence[Sequence[float]] = (),
) -> tuple[
    list[_Particle],
    list[_Particle],
    list[_AcquiredAnchorPath],
    tuple[dict[str, Any], ...],
    float,
    bool,
]:
    """Run adaptive SMC with all session integers analytically marginalized."""

    active = config or CollapsedIntegerConfig()
    rng = np.random.default_rng(int(active.random_seed))
    requested = max(int(active.particle_count), 8)
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    independent_proposals = _discover_invariant_mode_proposals(
        evaluator=evaluator,
        coordinates=coordinates,
        config=active,
        additional_normalized_starts=mode_discovery_starts,
    )
    prior_center_particle = _evaluate_particle(
        evaluator,
        coordinates,
        np.asarray(
            coordinates.normalize(coordinates.search_center),
            dtype=float,
        ),
        active,
    )
    anchor_origins = [
        particle
        for proposal in independent_proposals[: max(int(active.bridge_anchor_count), 0)]
        if (
            particle := _evaluate_particle(
                evaluator,
                coordinates,
                np.asarray(proposal.center, dtype=float),
                active,
            )
        )
        is not None
    ]
    bridge_anchors = [_clone_particle(particle) for particle in anchor_origins]
    bounded_modes = _bounded_mode_proposals(
        independent_proposals,
        lower=lower,
        upper=upper,
    )
    mode_fraction = (
        float(np.clip(active.mode_seed_fraction, 0.0, 0.9))
        if bounded_modes
        else 0.0
    )
    log_uniform_density = -float(np.sum(np.log(upper - lower)))
    particles: list[_Particle] = []
    raw_log_weights: list[float] = []
    mode_draw_count = 0
    exact_mode_seed_count = 0
    maximum_attempts = max(20 * requested, requested)
    normalized_prior_sampler = getattr(
        coordinates,
        "sample_normalized_prior",
        None,
    )
    normalized_prior_density = getattr(
        coordinates,
        "log_normalized_prior_density",
        None,
    )
    prior_draws = (
        np.asarray(
            normalized_prior_sampler(
                maximum_attempts,
                seed=int(active.random_seed) + 17,
            ),
            dtype=float,
        )
        if callable(normalized_prior_sampler)
        else None
    )
    prior_draw_index = 0

    def append_initial_particle(normalized: np.ndarray) -> bool:
        particle = _evaluate_particle(
            evaluator,
            coordinates,
            normalized,
            active,
        )
        if particle is None:
            return False
        log_prior_density = (
            float(normalized_prior_density(normalized))
            if callable(normalized_prior_density)
            else log_uniform_density
        )
        if not np.isfinite(log_prior_density):
            return False
        if mode_fraction > 0.0:
            log_mode_density = _bounded_mode_log_density(
                normalized,
                bounded_modes,
                lower=lower,
                upper=upper,
            )
            log_proposal_density = float(
                np.logaddexp(
                    math.log1p(-mode_fraction) + log_prior_density,
                    math.log(mode_fraction) + log_mode_density,
                )
            )
        else:
            log_proposal_density = log_prior_density
        particles.append(particle)
        raw_log_weights.append(log_prior_density - log_proposal_density)
        return True

    for proposal in independent_proposals:
        if len(particles) >= requested:
            break
        center = np.asarray(proposal.center, dtype=float)
        if any(
            np.array_equal(center, previous.normalized_values)
            for previous in particles
        ):
            continue
        if append_initial_particle(center):
            exact_mode_seed_count += 1

    for _ in range(maximum_attempts):
        if len(particles) >= requested:
            break
        if bounded_modes and rng.random() < mode_fraction:
            normalized = _sample_bounded_mode_proposal(
                bounded_modes,
                lower=lower,
                upper=upper,
                rng=rng,
            )
            mode_draw_count += 1
        else:
            if prior_draws is None:
                normalized = rng.uniform(lower, upper)
            else:
                normalized = prior_draws[prior_draw_index]
                prior_draw_index += 1
        append_initial_particle(np.asarray(normalized, dtype=float))
    if len(particles) < max(4, requested // 8):
        raise RuntimeError("Collapsed SMC could not evaluate enough prior particles.")
    count = len(particles)
    raw_weights = np.asarray(raw_log_weights, dtype=float)
    initial_log_normalizer = float(logsumexp(raw_weights) - math.log(count))
    normalized_initial_weights = raw_weights - logsumexp(raw_weights)
    for particle, log_weight in zip(
        particles,
        normalized_initial_weights,
        strict=True,
    ):
        particle.log_weight = float(log_weight)

    history: list[dict[str, Any]] = []
    tempering_order = str(active.tempering_order).strip().lower()
    if tempering_order == "phase_noise_bridge":
        bridge_anchors, anchor_archive, acquired_anchor_paths, anchor_history = (
            _run_full_shape_anchor_continuation(
                anchor_origins,
                evaluator=evaluator,
                coordinates=coordinates,
                config=active,
                progress_callback=progress_callback,
            )
        )
        history.extend(anchor_history)
    else:
        anchor_archive = [_clone_particle(anchor) for anchor in anchor_origins]
        acquired_anchor_paths = []
    log_evidence = initial_log_normalizer
    invariant_beta = 0.0
    phase_beta = 0.0
    session_phase_variances = np.concatenate(
        [
            np.asarray(
                particle.profile.session_offset_variances,
                dtype=float,
            )
            for particle in particles
        ]
    )
    finite_session_phase_variances = session_phase_variances[
        np.isfinite(session_phase_variances) & (session_phase_variances > 0.0)
    ]
    bridge_reference_measurement_variance = float(
        np.median(finite_session_phase_variances)
    ) if finite_session_phase_variances.size else np.finfo(float).eps

    def temper_factor(
        factor_name: str,
        beta: float,
        other_beta: float,
        target_fraction: float,
    ) -> float:
        nonlocal log_evidence
        if factor_name == "invariant":
            values = np.asarray(
                [particle.factors.invariant_log_likelihood for particle in particles],
                dtype=float,
            )
        elif factor_name == "phase":
            values = np.asarray(
                [particle.factors.wrapped_log_likelihood for particle in particles],
                dtype=float,
            )
        elif factor_name == "joint":
            values = np.asarray(
                [
                    particle.factors.invariant_log_likelihood
                    + particle.factors.wrapped_log_likelihood
                    for particle in particles
                ],
                dtype=float,
            )
        else:
            raise ValueError(f"Unknown collapsed tempering factor: {factor_name}")
        target_ess = max(float(target_fraction), 0.1) * len(particles)
        next_beta = _choose_next_beta(
            beta,
            values,
            np.asarray([particle.log_weight for particle in particles], dtype=float),
            target_ess=target_ess,
            minimum_step=max(float(active.minimum_beta_step), 1.0e-12),
            maximum_step=max(float(active.maximum_beta_step), float(active.minimum_beta_step)),
        )
        delta = next_beta - beta
        old_log_weights = np.asarray([particle.log_weight for particle in particles], dtype=float)
        incremented = old_log_weights + delta * values
        increment_log_evidence = float(logsumexp(incremented))
        log_evidence += increment_log_evidence
        normalized = incremented - logsumexp(incremented)
        ess = _normalised_ess(incremented)
        force_resample = (
            str(active.tempering_order).strip().lower() == "phase_noise_bridge"
        )
        resampled = bool(
            force_resample or ess <= target_ess * (1.0 + 1.0e-6)
        )
        if resampled:
            indices = _systematic_resample(normalized, rng)
            particles[:] = [_clone_particle(particles[index]) for index in indices]
            for particle in particles:
                particle.log_weight = -math.log(len(particles))
            resampled = True
        else:
            for particle, log_weight in zip(particles, normalized, strict=True):
                particle.log_weight = float(log_weight)
        if factor_name == "invariant":
            active_invariant_beta = next_beta
            active_phase_beta = other_beta
        elif factor_name == "phase":
            active_invariant_beta = other_beta
            active_phase_beta = next_beta
        else:
            active_invariant_beta = next_beta
            active_phase_beta = next_beta
        phase_metrics = ()
        if (
            str(active.tempering_order).strip().lower() == "phase_noise_bridge"
            and active_phase_beta > 0.0
            and int(active.move_steps_per_tempering_step) > 0
        ):
            phase_metrics = _phase_metric_covariances(
                particles,
                evaluator=evaluator,
                coordinates=coordinates,
                added_phase_variance=0.0,
                config=active,
            )
        accepted = _rejuvenate(
            particles,
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_beta=active_invariant_beta,
            phase_beta=active_phase_beta,
            config=active,
            rng=rng,
            independent_proposals=independent_proposals,
            phase_metric_covariances=phase_metrics,
        )
        summary = {
            "factor": factor_name,
            "information_order": tempering_order,
            "beta": float(next_beta),
            "invariant_beta": float(active_invariant_beta),
            "phase_beta": float(active_phase_beta),
            "delta_beta": float(delta),
            "ess": float(ess),
            "target_ess": float(target_ess),
            "resampled": bool(resampled),
            "accepted_moves": int(accepted),
            "mode_count": int(_mode_count(particles)),
            "best_invariant_chi2": float(
                min(-2.0 * particle.factors.invariant_log_likelihood for particle in particles)
            ),
            "best_wrapped_log_likelihood": float(
                max(particle.factors.wrapped_log_likelihood for particle in particles)
            ),
        }
        history.append(summary)
        if progress_callback is not None:
            progress_callback(summary)
        return next_beta

    def bridge_phase_values(progress: float) -> tuple[np.ndarray, float]:
        added_variance = _phase_noise_bridge_variance(
            progress,
            active,
            reference_measurement_variance=(
                bridge_reference_measurement_variance
                if str(active.phase_noise_parameterization).strip().lower()
                == "measurement_precision"
                else None
            ),
        )
        values = np.asarray(
            [
                _particle_wrapped_log_likelihood(
                    particle,
                    added_variance,
                    active,
                )
                for particle in particles
            ],
            dtype=float,
        )
        return values, added_variance

    def bridge_target_values(
        progress: float,
        shape_beta: float,
    ) -> tuple[np.ndarray, float]:
        phase_values, added_variance = bridge_phase_values(progress)
        shape_values = np.asarray(
            [particle.factors.invariant_log_likelihood for particle in particles],
            dtype=float,
        )
        return phase_values + float(shape_beta) * shape_values, added_variance

    def choose_next_bridge_progress(
        current_progress: float,
        current_phase_values: np.ndarray,
        target_ess: float,
    ) -> float:
        minimum_step = max(float(active.minimum_beta_step), 1.0e-12)
        maximum_step = max(float(active.maximum_beta_step), minimum_step)
        upper = min(1.0, current_progress + maximum_step)
        log_weights = np.asarray(
            [particle.log_weight for particle in particles],
            dtype=float,
        )

        def ess_at(progress: float) -> float:
            candidate_values, _ = bridge_phase_values(progress)
            return _conditional_ess(
                log_weights,
                candidate_values - current_phase_values,
            )

        scan_points = max(int(active.bridge_ess_scan_points), 2)
        low = float(current_progress)
        high: float | None = None
        for candidate in np.linspace(
            current_progress,
            upper,
            scan_points + 1,
            dtype=float,
        )[1:]:
            if ess_at(float(candidate)) < target_ess:
                high = float(candidate)
                break
            low = float(candidate)
        if high is None:
            return float(upper)
        for _ in range(24):
            middle = 0.5 * (low + high)
            if ess_at(middle) >= target_ess:
                low = middle
            else:
                high = middle
        return float(min(1.0, max(low, current_progress + minimum_step)))

    def choose_next_bridge_shape_beta(
        *,
        current_beta: float,
        phase_increment: np.ndarray,
        phase_increment_ess: float,
        joint_target_ess: float,
    ) -> tuple[float, float, float]:
        minimum_step = max(float(active.minimum_beta_step), 1.0e-12)
        maximum_step = max(float(active.maximum_beta_step), minimum_step)
        upper = min(1.0, current_beta + maximum_step)
        log_weights = np.asarray(
            [particle.log_weight for particle in particles],
            dtype=float,
        )
        shape_values = np.asarray(
            [particle.factors.invariant_log_likelihood for particle in particles],
            dtype=float,
        )

        def diagnostics(beta: float) -> tuple[float, float]:
            shape_increment = (float(beta) - current_beta) * shape_values
            shape_ess = _conditional_ess(log_weights, shape_increment)
            joint_ess = _conditional_ess(
                log_weights,
                phase_increment + shape_increment,
            )
            return shape_ess, joint_ess

        def admissible(beta: float) -> bool:
            shape_ess, joint_ess = diagnostics(beta)
            return bool(
                shape_ess >= phase_increment_ess
                and joint_ess >= joint_target_ess
            )

        scan_points = max(int(active.bridge_ess_scan_points), 2)
        low = float(current_beta)
        high: float | None = None
        for candidate in np.linspace(
            current_beta,
            upper,
            scan_points + 1,
            dtype=float,
        )[1:]:
            if not admissible(float(candidate)):
                high = float(candidate)
                break
            low = float(candidate)
        if high is not None:
            for _ in range(24):
                middle = 0.5 * (low + high)
                if admissible(middle):
                    low = middle
                else:
                    high = middle
        next_beta = float(low)
        shape_ess, joint_ess = diagnostics(next_beta)
        return next_beta, float(shape_ess), float(joint_ess)

    def apply_bridge_transition(
        *,
        current_progress: float | None,
        next_progress: float,
        current_shape_beta: float,
        next_shape_beta: float,
        current_values: np.ndarray,
        current_phase_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal log_evidence
        next_values, added_variance = bridge_target_values(
            next_progress,
            next_shape_beta,
        )
        next_phase_values, _ = bridge_phase_values(next_progress)
        old_log_weights = np.asarray(
            [particle.log_weight for particle in particles],
            dtype=float,
        )
        phase_increment = next_phase_values - current_phase_values
        shape_values = np.asarray(
            [particle.factors.invariant_log_likelihood for particle in particles],
            dtype=float,
        )
        shape_increment = (next_shape_beta - current_shape_beta) * shape_values
        phase_increment_ess = _conditional_ess(old_log_weights, phase_increment)
        shape_increment_ess = _conditional_ess(old_log_weights, shape_increment)
        incremented = old_log_weights + next_values - current_values
        increment_log_evidence = float(logsumexp(incremented))
        log_evidence += increment_log_evidence
        normalized = incremented - logsumexp(incremented)
        ess = _normalised_ess(incremented)
        conditional_ess = _conditional_ess(
            old_log_weights,
            next_values - current_values,
        )
        target_ess = max(float(active.bridge_ess_target_fraction), 0.1) * len(particles)
        resampled = bool(ess <= target_ess * (1.0 + 1.0e-6))
        if resampled:
            indices = _systematic_resample(normalized, rng)
            particles[:] = [_clone_particle(particles[index]) for index in indices]
            for particle in particles:
                particle.log_weight = -math.log(len(particles))
        else:
            for particle, log_weight in zip(particles, normalized, strict=True):
                particle.log_weight = float(log_weight)
        information_threshold = float(np.clip(
            active.bridge_rejuvenation_information_fraction,
            0.0,
            1.0,
        )) * len(particles)
        rejuvenation_skipped = bool(
            conditional_ess >= information_threshold
            and next_shape_beta <= np.finfo(float).eps
        )
        if rejuvenation_skipped:
            accepted = 0
        else:
            phase_metrics = _phase_metric_covariances(
                particles,
                evaluator=evaluator,
                coordinates=coordinates,
                added_phase_variance=added_variance,
                config=active,
            ) if int(active.move_steps_per_tempering_step) > 0 else ()
            accepted = _rejuvenate(
                particles,
                evaluator=evaluator,
                coordinates=coordinates,
                invariant_beta=next_shape_beta,
                phase_beta=1.0,
                added_phase_variance=added_variance,
                config=active,
                rng=rng,
                independent_proposals=independent_proposals,
                phase_metric_covariances=phase_metrics,
            )
        refreshed_values, _ = bridge_target_values(
            next_progress,
            next_shape_beta,
        )
        refreshed_phase_values, _ = bridge_phase_values(next_progress)
        summary = {
            "factor": "phase_noise_bridge",
            "information_order": tempering_order,
            "bridge_progress": float(next_progress),
            "delta_progress": float(
                next_progress if current_progress is None else next_progress - current_progress
            ),
            "beta": float(next_progress),
            "invariant_beta": float(next_shape_beta),
            "delta_invariant_beta": float(next_shape_beta - current_shape_beta),
            "phase_beta": 1.0,
            "phase_noise_sigma_cycles": float(math.sqrt(added_variance)),
            "phase_noise_variance": float(added_variance),
            "phase_noise_parameterization": str(
                active.phase_noise_parameterization
            ).strip().lower(),
            "reference_measurement_sigma_cycles": float(
                math.sqrt(bridge_reference_measurement_variance)
            ),
            "ess": float(ess),
            "conditional_ess": float(conditional_ess),
            "phase_increment_ess": float(phase_increment_ess),
            "shape_increment_ess": float(shape_increment_ess),
            "target_ess": float(target_ess),
            "initial_mode_draw_count": int(mode_draw_count)
            if current_progress is None
            else None,
            "initial_exact_mode_seed_count": int(exact_mode_seed_count)
            if current_progress is None
            else None,
            "resampled": resampled,
            "accepted_moves": int(accepted),
            "rejuvenation_skipped": rejuvenation_skipped,
            "mode_count": int(_mode_count(particles)),
            "best_invariant_chi2": float(
                min(-2.0 * particle.factors.invariant_log_likelihood for particle in particles)
            ),
            "best_bridge_wrapped_log_likelihood": float(
                max(
                    _particle_wrapped_log_likelihood(
                        particle,
                        added_variance,
                        active,
                    )
                    for particle in particles
                )
            ),
            "best_wrapped_log_likelihood": float(
                max(particle.factors.wrapped_log_likelihood for particle in particles)
            ),
        }
        history.append(summary)
        if progress_callback is not None:
            progress_callback(summary)
        return refreshed_values, refreshed_phase_values

    if tempering_order == "phase_noise_bridge":
        bridge_progress = 0.0
        bridge_shape_beta = 0.0
        current_target_values, current_phase_values = apply_bridge_transition(
            current_progress=None,
            next_progress=bridge_progress,
            current_shape_beta=bridge_shape_beta,
            next_shape_beta=bridge_shape_beta,
            current_values=np.zeros(len(particles), dtype=float),
            current_phase_values=np.zeros(len(particles), dtype=float),
        )
        joint_target_ess = (
            max(float(active.bridge_ess_target_fraction), 0.1) * len(particles)
        )
        phase_target_ess = math.sqrt(
            max(float(active.bridge_ess_target_fraction), 0.1)
        ) * len(particles)
        for _ in range(max(int(active.maximum_tempering_steps), 1)):
            if bridge_progress >= 1.0 - 1.0e-12:
                break
            next_progress = choose_next_bridge_progress(
                bridge_progress,
                current_phase_values,
                phase_target_ess,
            )
            next_phase_values, _ = bridge_phase_values(next_progress)
            log_weights = np.asarray(
                [particle.log_weight for particle in particles],
                dtype=float,
            )
            phase_increment = next_phase_values - current_phase_values
            phase_increment_ess = _conditional_ess(
                log_weights,
                phase_increment,
            )
            next_shape_beta, _, _ = choose_next_bridge_shape_beta(
                current_beta=bridge_shape_beta,
                phase_increment=phase_increment,
                phase_increment_ess=phase_increment_ess,
                joint_target_ess=joint_target_ess,
            )
            current_target_values, current_phase_values = apply_bridge_transition(
                current_progress=bridge_progress,
                next_progress=next_progress,
                current_shape_beta=bridge_shape_beta,
                next_shape_beta=next_shape_beta,
                current_values=current_target_values,
                current_phase_values=current_phase_values,
            )
            bridge_progress = next_progress
            bridge_shape_beta = next_shape_beta
        added_variance = _phase_noise_bridge_variance(
            bridge_progress,
            active,
            reference_measurement_variance=(
                bridge_reference_measurement_variance
                if str(active.phase_noise_parameterization).strip().lower()
                == "measurement_precision"
                else None
            ),
        )
        phase_beta = 1.0 if added_variance <= np.finfo(float).eps else 0.0
        invariant_beta = bridge_shape_beta
        if phase_beta >= 1.0 - 1.0e-12:
            for _ in range(max(int(active.maximum_tempering_steps), 1)):
                if invariant_beta >= 1.0 - 1.0e-12:
                    break
                invariant_beta = temper_factor(
                    "invariant",
                    invariant_beta,
                    phase_beta,
                    active.invariant_ess_target_fraction,
                )
    elif tempering_order == "shape_then_phase_noise":
        # Establish the phase-disconnected physical posterior before phase
        # coherence can remove modes. This order is needed
        # when a poor initial ephemeris places the correct phase path far from
        # the highest-prior local alias.
        for _ in range(max(int(active.maximum_tempering_steps), 1)):
            if invariant_beta >= 1.0 - 1.0e-12:
                break
            invariant_beta = temper_factor(
                "invariant",
                invariant_beta,
                phase_beta,
                active.invariant_ess_target_fraction,
            )

        clock_acquisitions, clock_history = _acquire_conditional_clock_paths(
            [
                *(
                    (prior_center_particle,)
                    if prior_center_particle is not None
                    else ()
                ),
                *particles,
                *anchor_archive,
            ],
            evaluator=evaluator,
            coordinates=coordinates,
            config=active,
        )
        if active.conditional_clock_lattice:
            acquired_anchor_paths.extend(clock_acquisitions)
            history.append(clock_history)
            if progress_callback is not None:
                progress_callback(clock_history)

        bridge_progress = 0.0
        bridge_shape_beta = invariant_beta
        shape_values = np.asarray(
            [particle.factors.invariant_log_likelihood for particle in particles],
            dtype=float,
        )
        current_target_values, current_phase_values = apply_bridge_transition(
            current_progress=None,
            next_progress=bridge_progress,
            current_shape_beta=bridge_shape_beta,
            next_shape_beta=bridge_shape_beta,
            current_values=shape_values,
            current_phase_values=np.zeros(len(particles), dtype=float),
        )
        phase_target_ess = math.sqrt(
            max(float(active.bridge_ess_target_fraction), 0.1)
        ) * len(particles)
        for _ in range(max(int(active.maximum_tempering_steps), 1)):
            if bridge_progress >= 1.0 - 1.0e-12:
                break
            next_progress = choose_next_bridge_progress(
                bridge_progress,
                current_phase_values,
                phase_target_ess,
            )
            current_target_values, current_phase_values = apply_bridge_transition(
                current_progress=bridge_progress,
                next_progress=next_progress,
                current_shape_beta=bridge_shape_beta,
                next_shape_beta=bridge_shape_beta,
                current_values=current_target_values,
                current_phase_values=current_phase_values,
            )
            bridge_progress = next_progress

        added_variance = _phase_noise_bridge_variance(
            bridge_progress,
            active,
            reference_measurement_variance=bridge_reference_measurement_variance,
        )
        phase_beta = 1.0 if added_variance <= np.finfo(float).eps else 0.0
        invariant_beta = bridge_shape_beta
    elif tempering_order == "joint":
        joint_beta = 0.0
        for _ in range(max(int(active.maximum_tempering_steps), 1)):
            if joint_beta >= 1.0 - 1.0e-12:
                break
            joint_beta = temper_factor(
                "joint",
                joint_beta,
                joint_beta,
                active.joint_ess_target_fraction,
            )
        invariant_beta = joint_beta
        phase_beta = joint_beta
    elif tempering_order == "sequential":
        for _ in range(max(int(active.maximum_tempering_steps), 1)):
            if invariant_beta >= 1.0 - 1.0e-12:
                break
            invariant_beta = temper_factor(
                "invariant",
                invariant_beta,
                phase_beta,
                active.invariant_ess_target_fraction,
            )
        for _ in range(max(int(active.maximum_tempering_steps), 1)):
            if phase_beta >= 1.0 - 1.0e-12:
                break
            phase_beta = temper_factor(
                "phase",
                phase_beta,
                invariant_beta,
                active.phase_ess_target_fraction,
            )
    else:
        raise ValueError(
            "Collapsed tempering_order must be 'shape_then_phase_noise', "
            "'phase_noise_bridge', 'joint', or 'sequential'."
        )
    reached = bool(invariant_beta >= 1.0 - 1.0e-10 and phase_beta >= 1.0 - 1.0e-10)
    final_anchors: list[_Particle] = []
    for anchor in anchor_archive:
        if any(
            float(np.linalg.norm(anchor.normalized_values - previous.normalized_values))
            < 1.0e-9
            for previous in final_anchors
        ):
            continue
        final_anchors.append(anchor)
    return (
        particles,
        final_anchors,
        acquired_anchor_paths,
        tuple(history),
        float(log_evidence),
        reached,
    )


def _aggregate_path_posterior(
    particles: Sequence[_Particle],
    *,
    config: CollapsedIntegerConfig,
) -> tuple[tuple[IntegerPathCandidate, ...], dict[tuple[int, ...], tuple[int, float]]]:
    path_log_masses: dict[tuple[int, ...], float] = {}
    source_by_path: dict[tuple[int, ...], tuple[int, float]] = {}
    for index, particle in enumerate(particles):
        candidates = conditional_integer_paths(
            particle.profile,
            candidate_count=config.integer_candidate_count,
            hermite_order=config.hermite_order,
            integer_tail_tolerance=config.integer_tail_tolerance,
            maximum_wrap_terms=config.maximum_wrap_terms,
        )
        for candidate in candidates:
            contribution = particle.log_weight + candidate.log_conditional_mass
            previous = path_log_masses.get(candidate.integer_offsets, float("-inf"))
            path_log_masses[candidate.integer_offsets] = float(
                np.logaddexp(previous, contribution)
            )
            if contribution > source_by_path.get(candidate.integer_offsets, (-1, float("-inf")))[1]:
                source_by_path[candidate.integer_offsets] = (index, contribution)

    if not path_log_masses:
        return (), source_by_path
    normalizer = logsumexp(np.asarray(list(path_log_masses.values()), dtype=float))
    ranked = sorted(path_log_masses.items(), key=lambda item: item[1], reverse=True)
    output: list[IntegerPathCandidate] = []
    for path, log_mass in ranked[: max(int(config.maximum_paths), 1)]:
        output.append(
            IntegerPathCandidate(
                integer_offsets=path,
                wraps=integer_session_offsets_to_gap_wraps(path),
                log_abs_likelihood=float(log_mass),
                log_conditional_mass=float(log_mass - normalizer),
            )
        )
    return tuple(output), source_by_path


def _anchor_path_portfolio(
    anchors: Sequence[_Particle],
    *,
    source_index_offset: int,
    config: CollapsedIntegerConfig,
) -> tuple[
    tuple[IntegerPathCandidate, ...],
    dict[tuple[int, ...], tuple[int, float]],
]:
    records: dict[
        tuple[int, ...],
        tuple[float, int, IntegerPathCandidate],
    ] = {}
    per_anchor_count = min(max(int(config.integer_candidate_count), 1), 4)
    for anchor_index, anchor in enumerate(anchors):
        candidates = conditional_integer_paths(
            anchor.profile,
            candidate_count=per_anchor_count,
            hermite_order=config.hermite_order,
            integer_tail_tolerance=config.integer_tail_tolerance,
            maximum_wrap_terms=config.maximum_wrap_terms,
        )
        for candidate in candidates:
            exact_score = float(
                anchor.factors.invariant_log_likelihood
                + fixed_integer_log_likelihood(
                    anchor.profile,
                    candidate.integer_offsets,
                )
            )
            previous = records.get(candidate.integer_offsets)
            if previous is None or exact_score > previous[0]:
                records[candidate.integer_offsets] = (
                    exact_score,
                    source_index_offset + anchor_index,
                    candidate,
                )
    ranked = sorted(records.values(), key=lambda item: item[0], reverse=True)
    output = tuple(
        IntegerPathCandidate(
            integer_offsets=candidate.integer_offsets,
            wraps=candidate.wraps,
            log_abs_likelihood=float(score),
            log_conditional_mass=float(candidate.log_conditional_mass),
            proposal_source="anchor",
        )
        for score, _, candidate in ranked[: max(int(config.maximum_paths), 1)]
    )
    source_by_path = {
        candidate.integer_offsets: (source_index, score)
        for score, source_index, candidate in ranked
    }
    return output, source_by_path


def _acquired_anchor_path_portfolio(
    acquisitions: Sequence[_AcquiredAnchorPath],
    *,
    source_index_offset: int,
    config: CollapsedIntegerConfig,
) -> tuple[
    tuple[IntegerPathCandidate, ...],
    dict[tuple[int, ...], tuple[int, float]],
]:
    """Return complete paths exactly as acquired by the correlated lattice.

    Recomputing a pointwise conditional top-K list can discard the coherent path
    that generated a nonlinear child. The persistent portfolio keeps each path
    paired with its optimized continuous source.
    """

    records: dict[
        tuple[int, ...],
        tuple[float, int, _AcquiredAnchorPath],
    ] = {}
    for acquisition_index, acquisition in enumerate(acquisitions):
        path = tuple(acquisition.candidate.integer_offsets)
        source_index = source_index_offset + acquisition_index
        previous = records.get(path)
        if previous is None or acquisition.exact_log_target > previous[0]:
            records[path] = (
                float(acquisition.exact_log_target),
                source_index,
                acquisition,
            )

    ranked = sorted(records.values(), key=lambda item: item[0], reverse=True)
    selected: list[tuple[float, int, _AcquiredAnchorPath]] = []
    selected_acquisition_ids: set[int] = set()
    represented_families: set[str] = set()
    for record in ranked:
        family = record[2].acquisition_family
        if family in represented_families:
            continue
        selected.append(record)
        selected_acquisition_ids.add(id(record[2]))
        represented_families.add(family)
        if len(selected) >= max(int(config.maximum_paths), 1):
            break
    for record in ranked:
        if len(selected) >= max(int(config.maximum_paths), 1):
            break
        if id(record[2]) in selected_acquisition_ids:
            continue
        selected.append(record)
        selected_acquisition_ids.add(id(record[2]))
    output = tuple(
        IntegerPathCandidate(
            integer_offsets=acquisition.candidate.integer_offsets,
            wraps=acquisition.candidate.wraps,
            log_abs_likelihood=float(score),
            log_conditional_mass=float(
                acquisition.candidate.log_conditional_mass
            ),
            proposal_source=(
                "persistent-complete-path:"
                + str(acquisition.candidate.proposal_source)
            ),
        )
        for score, _, acquisition in selected
    )
    source_by_path = {
        acquisition.candidate.integer_offsets: (source_index, score)
        for score, source_index, acquisition in selected
    }
    return output, source_by_path


def _joint_log_target(
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    normalized_values: np.ndarray,
    integer_offsets: Sequence[int],
    config: CollapsedIntegerConfig,
    *,
    added_phase_variance: float = 0.0,
) -> tuple[float, _Particle | None]:
    particle = _evaluate_particle(evaluator, coordinates, normalized_values, config)
    if particle is None:
        return float("-inf"), None
    fixed_log = fixed_integer_log_likelihood(
        particle.profile,
        integer_offsets,
        added_session_phase_variance=added_phase_variance,
    )
    return float(
        particle.log_prior
        + particle.factors.invariant_log_likelihood
        + fixed_log
    ), particle


def _optimize_fixed_integer_path(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    start: np.ndarray,
    integer_offsets: Sequence[int],
    config: CollapsedIntegerConfig,
    added_phase_variance: float = 0.0,
    maximum_iterations: int | None = None,
    finite_difference_step: float | None = None,
    coordinate_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, float, _Particle | None, int]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    interior = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
    initial = np.clip(np.asarray(start, dtype=float), lower + interior, upper - interior)
    active_indices = tuple(
        range(initial.size)
        if coordinate_indices is None
        else (int(value) for value in coordinate_indices)
    )
    if not active_indices or len(set(active_indices)) != len(active_indices):
        raise ValueError("Fixed-path optimization coordinates must be unique and non-empty.")
    if min(active_indices) < 0 or max(active_indices) >= initial.size:
        raise ValueError("Fixed-path optimization coordinate index is out of bounds.")
    active_lower = lower[list(active_indices)] + interior[list(active_indices)]
    active_upper = upper[list(active_indices)] - interior[list(active_indices)]
    bounds = list(zip(active_lower, active_upper, strict=True))
    evaluations = 0
    cache: dict[bytes, tuple[float, _Particle | None]] = {}

    def objective(active_values: np.ndarray) -> float:
        nonlocal evaluations
        array = initial.copy()
        array[list(active_indices)] = np.asarray(active_values, dtype=float)
        key = array.tobytes()
        if key not in cache:
            log_target, particle = _joint_log_target(
                evaluator,
                coordinates,
                array,
                integer_offsets,
                config,
                added_phase_variance=added_phase_variance,
            )
            evaluations += 1
            cache[key] = (
                -log_target if particle is not None else 1.0e300,
                particle,
            )
        return float(cache[key][0])

    try:
        iteration_limit = (
            max(int(maximum_iterations), 1)
            if maximum_iterations is not None
            else max(int(config.joint_optimization_maxiter), 1)
        )
        difference_step = (
            max(float(finite_difference_step), 1.0e-10)
            if finite_difference_step is not None
            else max(
                float(config.joint_optimization_finite_difference_step),
                1.0e-10,
            )
        )
        optimized = minimize(
            objective,
            initial[list(active_indices)],
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": iteration_limit,
                "maxfun": max(
                    iteration_limit * (len(active_indices) + 1),
                    16,
                ),
                "ftol": 1.0e-11,
                "gtol": 1.0e-7,
                "eps": difference_step,
            },
        )
        normalized = initial.copy()
        normalized[list(active_indices)] = np.asarray(optimized.x, dtype=float)
        log_target, particle = _joint_log_target(
            evaluator,
            coordinates,
            normalized,
            integer_offsets,
            config,
            added_phase_variance=added_phase_variance,
        )
        evaluations += 1
    except (
        ArithmeticError,
        FloatingPointError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return initial, float("-inf"), None, evaluations
    return normalized, float(log_target), particle, evaluations


def refine_joint_integer_paths(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    particles: Sequence[_Particle],
    path_posterior: Sequence[IntegerPathCandidate],
    source_by_path: dict[tuple[int, ...], tuple[int, float]],
    config: CollapsedIntegerConfig,
) -> tuple[JointPathSummary, ...]:
    """Rejuvenate continuous values and integer blocks against the exact target."""

    rng = np.random.default_rng(int(config.random_seed) + 7919)
    results: list[JointPathSummary] = []
    for candidate_index, candidate in enumerate(path_posterior):
        source_index = source_by_path.get(candidate.integer_offsets, (0, 0.0))[0]
        source = particles[source_index]
        normalized = source.normalized_values.copy()
        integer_offsets = tuple(candidate.integer_offsets)
        current_log, current_particle = _joint_log_target(
            evaluator,
            coordinates,
            normalized,
            integer_offsets,
            config,
        )
        if current_particle is None:
            continue
        optimization_evaluations = 0
        optimization_improved = False
        if candidate_index < max(int(config.joint_optimization_path_count), 0):
            (
                optimized_values,
                optimized_log,
                optimized_particle,
                optimization_evaluations,
            ) = _optimize_fixed_integer_path(
                evaluator=evaluator,
                coordinates=coordinates,
                start=normalized,
                integer_offsets=integer_offsets,
                config=config,
            )
            if optimized_particle is not None and optimized_log > current_log:
                normalized = optimized_values
                current_log = optimized_log
                current_particle = optimized_particle
                optimization_improved = True
        accepted_continuous = 0
        accepted_integer = 0
        for _ in range(max(int(config.joint_move_steps), 0)):
            proposal_values = normalized + rng.normal(
                0.0,
                max(float(config.joint_proposal_scale), 1.0e-8),
                size=normalized.size,
            )
            proposal_log, proposal_particle = _joint_log_target(
                evaluator,
                coordinates,
                proposal_values,
                integer_offsets,
                config,
            )
            if proposal_particle is not None and math.log(rng.random()) < min(
                0.0,
                proposal_log - current_log,
            ):
                normalized = proposal_values
                current_log = proposal_log
                current_particle = proposal_particle
                accepted_continuous += 1

            conditional = conditional_integer_paths(
                current_particle.profile,
                candidate_count=config.integer_candidate_count,
                hermite_order=config.hermite_order,
                integer_tail_tolerance=config.integer_tail_tolerance,
                maximum_wrap_terms=config.maximum_wrap_terms,
            )
            if conditional:
                probabilities = np.asarray(
                    [item.log_conditional_mass for item in conditional],
                    dtype=float,
                )
                probabilities -= logsumexp(probabilities)
                selected = conditional[int(rng.choice(len(conditional), p=np.exp(probabilities)))]
                proposal_path = selected.integer_offsets
                proposal_log, proposal_particle = _joint_log_target(
                    evaluator,
                    coordinates,
                    normalized,
                    proposal_path,
                    config,
                )
                if proposal_particle is not None and math.log(rng.random()) < min(
                    0.0,
                    proposal_log - current_log,
                ):
                    integer_offsets = proposal_path
                    current_log = proposal_log
                    current_particle = proposal_particle
                    accepted_integer += 1
        results.append(
            JointPathSummary(
                integer_offsets=integer_offsets,
                wraps=integer_session_offsets_to_gap_wraps(integer_offsets),
                source_particle_index=int(source_index),
                proposal_source=str(candidate.proposal_source),
                posterior_mass=(
                    float(math.exp(candidate.log_conditional_mass))
                    if candidate.proposal_source == "posterior"
                    else None
                ),
                best_log_joint=float(current_log),
                parameter_values=current_particle.parameter_values.copy(),
                normalized_values=normalized.copy(),
                continuous_optimization_evaluations=int(optimization_evaluations),
                continuous_optimization_improved=bool(optimization_improved),
                accepted_continuous_moves=accepted_continuous,
                accepted_integer_moves=accepted_integer,
            )
        )
    return tuple(sorted(results, key=lambda item: item.best_log_joint, reverse=True))


def infer_collapsed_integer_paths(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: CollapsedIntegerConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    mode_discovery_starts: Sequence[Sequence[float]] = (),
) -> CollapsedIntegerInferenceResult:
    active = config or CollapsedIntegerConfig()
    (
        particles,
        bridge_anchors,
        acquired_anchor_paths,
        history,
        log_evidence,
        reached,
    ) = run_collapsed_integer_smc(
        evaluator=evaluator,
        coordinates=coordinates,
        config=active,
        progress_callback=progress_callback,
        mode_discovery_starts=mode_discovery_starts,
    )
    posterior_paths, source_by_path = _aggregate_path_posterior(
        particles,
        config=active,
    )
    combined_particles = [
        *particles,
        *bridge_anchors,
        *(acquisition.particle for acquisition in acquired_anchor_paths),
    ]
    anchor_paths, anchor_sources = _anchor_path_portfolio(
        bridge_anchors,
        source_index_offset=len(particles),
        config=active,
    )
    acquired_paths, acquired_sources = _acquired_anchor_path_portfolio(
        acquired_anchor_paths,
        source_index_offset=len(particles) + len(bridge_anchors),
        config=active,
    )
    merged_paths: dict[tuple[int, ...], IntegerPathCandidate] = {
        candidate.integer_offsets: candidate for candidate in posterior_paths
    }
    ordered_path_keys = [candidate.integer_offsets for candidate in posterior_paths]
    for proposal_paths, proposal_sources in (
        (anchor_paths, anchor_sources),
        (acquired_paths, acquired_sources),
    ):
        for candidate in proposal_paths:
            path = candidate.integer_offsets
            proposal_source = proposal_sources[path]
            previous_source = source_by_path.get(path)
            if previous_source is None:
                merged_paths[path] = candidate
                source_by_path[path] = proposal_source
                ordered_path_keys.append(path)
                continue
            previous_particle = combined_particles[previous_source[0]]
            previous_score = float(
                previous_particle.factors.invariant_log_likelihood
                + fixed_integer_log_likelihood(previous_particle.profile, path)
            )
            if proposal_source[1] > previous_score:
                merged_paths[path] = candidate
                source_by_path[path] = proposal_source
    search_path_portfolio = tuple(
        merged_paths[path]
        for path in sorted(
            ordered_path_keys,
            key=lambda candidate_path: (
                combined_particles[source_by_path[candidate_path][0]].factors.invariant_log_likelihood
                + fixed_integer_log_likelihood(
                    combined_particles[source_by_path[candidate_path][0]].profile,
                    candidate_path,
                )
            ),
            reverse=True,
        )[: max(int(active.maximum_paths), 1)]
    )
    joint_paths = refine_joint_integer_paths(
        evaluator=evaluator,
        coordinates=coordinates,
        particles=combined_particles,
        path_posterior=search_path_portfolio,
        source_by_path=source_by_path,
        config=active,
    )
    particle_summaries = tuple(
        CollapsedParticleSummary(
            normalized_values=particle.normalized_values.copy(),
            parameter_values=particle.parameter_values.copy(),
            invariant_log_likelihood=float(particle.factors.invariant_log_likelihood),
            wrapped_log_likelihood=float(particle.factors.wrapped_log_likelihood),
            log_weight=float(particle.log_weight),
            log_prior=float(particle.log_prior),
        )
        for particle in particles
    )
    anchor_summaries = tuple(
        CollapsedParticleSummary(
            normalized_values=anchor.normalized_values.copy(),
            parameter_values=anchor.parameter_values.copy(),
            invariant_log_likelihood=float(anchor.factors.invariant_log_likelihood),
            wrapped_log_likelihood=float(anchor.factors.wrapped_log_likelihood),
            log_weight=float("-inf"),
            log_prior=float(anchor.log_prior),
        )
        for anchor in bridge_anchors
    )
    acquisition_summaries = tuple(
        PersistentPathAcquisitionSummary(
            integer_offsets=acquisition.candidate.integer_offsets,
            wraps=acquisition.candidate.wraps,
            proposal_source=acquisition.candidate.proposal_source,
            acquisition_family=acquisition.acquisition_family,
            phase_noise_sigma_cycles=float(
                acquisition.phase_noise_sigma_cycles
            ),
            bridge_log_target=float(acquisition.bridge_log_target),
            exact_log_target=float(acquisition.exact_log_target),
            parameter_values=acquisition.particle.parameter_values.copy(),
            normalized_values=acquisition.particle.normalized_values.copy(),
        )
        for acquisition in acquired_anchor_paths
    )
    convergence = {
        "particle_count": len(particles),
        "bridge_anchor_origin_budget": max(int(active.bridge_anchor_count), 0),
        "bridge_anchor_archive_count": len(bridge_anchors),
        "persistent_complete_path_acquisition_count": len(
            acquired_anchor_paths
        ),
        "persistent_complete_path_unique_count": len(acquired_paths),
        "history_length": len(history),
        "tempering_order": str(active.tempering_order).strip().lower(),
        "reached_invariant_beta_one": bool(any(
            float(item.get("invariant_beta", 0.0)) >= 1.0 - 1.0e-10
            for item in history
        )),
        "reached_phase_beta_one": bool(any(
            float(item.get("phase_beta", 0.0)) >= 1.0 - 1.0e-10
            for item in history
        )),
        "reached_zero_phase_noise": bool(any(
            float(item.get("phase_noise_variance", float("inf")))
            <= np.finfo(float).eps
            for item in history
        )),
        "mode_count_final": int(_mode_count(particles)),
        "posterior_path_count": len(posterior_paths),
        "search_path_count": len(search_path_portfolio),
        "wrapped_tail_bound_max": float(max(
            (particle.factors.wrapped_tail_bound for particle in particles),
            default=0.0,
        )),
    }
    return CollapsedIntegerInferenceResult(
        particles=particle_summaries,
        bridge_anchors=anchor_summaries,
        path_posterior=posterior_paths,
        search_path_portfolio=search_path_portfolio,
        persistent_path_acquisitions=acquisition_summaries,
        joint_paths=joint_paths,
        history=history,
        reached_beta_one=bool(reached),
        log_evidence=float(log_evidence),
        convergence=convergence,
    )
