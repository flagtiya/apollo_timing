from __future__ import annotations

from apollo_timing.inference._shared_helpers import (
    _normalized_values,
)

from apollo_timing.inference._shared_helpers import _physical_values

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import i0e, logsumexp

from apollo_timing.inference.bayesian_global_lattice_mode_tracker import (
    GlobalLatticeMode,
    GlobalLatticeTrackerConfig,
    InvariantMode,
    LatticeProposal,
    refit_fixed_path_proposal,
    select_diverse_lattice_modes,
)
from apollo_timing.inference.collapsed_integer_posterior import (
    conditional_integer_paths,
)


@dataclass(frozen=True)
class FullSequenceCircularConfig:
    """Controls for all-session circular posterior transport."""

    phase_sigma_ladder: tuple[float, ...] = (
        0.50,
        0.35,
        0.25,
        0.18,
        0.12,
        0.08,
        0.05,
        0.03,
        0.02,
        0.012,
        0.0075,
        0.004,
    )
    initial_mode_count: int = 16
    parents_per_temperature: int = 16
    sigma_points_per_parent: int = 5
    sigma_point_radius: float = 1.5
    sigma_point_radii: tuple[float, ...] = ()
    local_trust_radius: float = 6.0
    max_nfev_per_start: int = 36
    absolute_jacobian_step: float = 0.02
    phase_likelihood: str = "wrapped-gaussian"
    wrapped_gaussian_log_tolerance: float = -27.631021115928547
    wrapped_gaussian_max_grid_points: int = 16_384
    prior_precision: float = 1.0
    maximum_modes: int = 32
    minimum_normalized_separation: float = 0.02
    minimum_phase_separation: float = 0.01
    path_parent_modes: int = 24
    integer_candidates_per_mode: int = 8
    maximum_path_proposals: int = 96
    maximum_exact_modes: int = 24
    maximum_exact_modes_per_path: int = 2
    maximum_exact_modes_per_root: int = 8
    exact_mode_separation: float = 0.01
    exact_refit_max_nfev: int = 160
    exact_phase_tempering_ladder: tuple[float, ...] = (0.03, 0.10, 0.30, 1.0)
    maximum_lattice_nodes: int = 2_000_000


@dataclass(frozen=True)
class FullSequenceCircularMode:
    mode_id: str
    root_mode_id: str
    parent_mode_id: str | None
    proposal_source: str
    parameter_values: np.ndarray
    normalized_values: np.ndarray
    covariance: np.ndarray
    phase_sigma: float
    shape_chi2: float
    circular_deviance: float
    prior_chi2: float
    objective_chi2: float
    log_mass: float
    posterior_mass: float
    phase_coherence: float
    phase_signature: tuple[float, ...]
    nfev: int
    success: bool


@dataclass(frozen=True)
class CircularTemperatureSummary:
    temperature_index: int
    phase_sigma: float
    parent_count: int
    start_count: int
    optimized_count: int
    selected_count: int
    root_count: int
    best_shape_chi2: float
    best_circular_deviance: float
    best_objective_chi2: float
    maximum_phase_coherence: float


@dataclass(frozen=True)
class FullSequenceCircularResult:
    circular_modes: tuple[FullSequenceCircularMode, ...]
    exact_modes: tuple[GlobalLatticeMode, ...]
    temperature_history: tuple[CircularTemperatureSummary, ...]
    path_proposal_count: int
    source_session_count: int
    reached_final_temperature: bool


def _log_i0(value: float) -> float:
    scalar = abs(float(value))
    return float(math.log(float(i0e(scalar))) + scalar)


def circular_phase_deviance(
    session_offsets: Sequence[float],
    session_variances: Sequence[float],
    *,
    phase_sigma: float,
) -> tuple[float, float, tuple[float, ...]]:
    """Marginalize the common phase gauge under a von Mises noise model.

    The returned deviance is the log-likelihood ratio to perfect circular
    alignment at the same declared noise scale. No integer path or phase tree
    is selected in this calculation.
    """

    offsets = np.asarray(session_offsets, dtype=float)
    variances = np.asarray(session_variances, dtype=float)
    if offsets.ndim != 1 or variances.shape != offsets.shape or offsets.size == 0:
        raise ValueError("Session phase arrays must be non-empty and aligned.")
    if not np.all(np.isfinite(offsets)) or not np.all(
        np.isfinite(variances) & (variances > 0.0)
    ):
        raise ValueError("Session phases and variances must be finite and positive.")
    sigma = float(phase_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("The circular phase sigma must be finite and positive.")

    broadened = variances + sigma * sigma
    concentration = 1.0 / (4.0 * math.pi * math.pi * broadened)
    phase_vectors = np.exp(2.0j * math.pi * offsets)
    resultant_vector = np.sum(concentration * phase_vectors)
    total_concentration = float(np.sum(concentration))
    resultant = float(abs(resultant_vector))
    deviance = max(
        2.0 * (_log_i0(total_concentration) - _log_i0(resultant)),
        0.0,
    )
    coherence = float(np.clip(resultant / total_concentration, 0.0, 1.0))
    gauge = float(np.angle(resultant_vector) / (2.0 * math.pi))
    signature = (offsets - gauge + 0.5) % 1.0 - 0.5
    return deviance, coherence, tuple(float(value) for value in signature)


def circular_phase_residuals(
    session_offsets: Sequence[float],
    session_variances: Sequence[float],
    *,
    phase_sigma: float,
) -> np.ndarray:
    """Factor the marginalized circular deviance across all sessions.

    The vector has squared norm exactly equal to `circular_phase_deviance`.
    This factorization exposes the full gauge-free session-phase Jacobian to
    local posterior whitening. A single square-root deviance residual does not.
    """

    offsets = np.asarray(session_offsets, dtype=float)
    variances = np.asarray(session_variances, dtype=float)
    deviance, _, _ = circular_phase_deviance(
        offsets,
        variances,
        phase_sigma=phase_sigma,
    )
    sigma = float(phase_sigma)
    concentration = 1.0 / (
        4.0 * math.pi * math.pi * (variances + sigma * sigma)
    )
    resultant_vector = np.sum(
        concentration * np.exp(2.0j * math.pi * offsets)
    )
    gauge = float(np.angle(resultant_vector) / (2.0 * math.pi))
    raw = np.sqrt(2.0 * concentration) * np.sin(
        math.pi * (offsets - gauge)
    )
    raw_norm2 = float(raw @ raw)
    if deviance <= np.finfo(float).eps or raw_norm2 <= np.finfo(float).tiny:
        return np.zeros_like(offsets)
    return math.sqrt(deviance / raw_norm2) * raw


def _wrapped_phase(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return (array + 0.5) % 1.0 - 0.5


def _wrapped_normal_log_density(
    residuals: Sequence[float] | np.ndarray,
    variance: float,
    *,
    log_tolerance: float,
) -> np.ndarray:
    """Evaluate a unit-period wrapped normal without selecting an integer."""

    residual = _wrapped_phase(residuals)
    active_variance = float(variance)
    if not np.isfinite(active_variance) or active_variance <= 0.0:
        raise ValueError("Wrapped-normal variance must be finite and positive.")
    tolerance = min(float(log_tolerance), -1.0)
    tail_radius = math.sqrt(max(-2.0 * active_variance * tolerance, 0.0))
    image_count = max(int(math.ceil(0.5 + tail_radius)), 1)
    images = np.arange(-image_count, image_count + 1, dtype=float)
    exponent = -0.5 * np.square(residual[..., None] + images) / active_variance
    return (
        logsumexp(exponent, axis=-1)
        - 0.5 * math.log(2.0 * math.pi * active_variance)
    )


def _wrapped_gaussian_grid_size(
    variances: np.ndarray,
    *,
    phase_sigma: float,
    log_tolerance: float,
    maximum_points: int,
) -> int:
    broadened = np.asarray(variances, dtype=float) + float(phase_sigma) ** 2
    tolerance = min(float(log_tolerance), -1.0)
    harmonic_counts = np.ceil(
        np.sqrt(-tolerance / (2.0 * math.pi * math.pi * broadened))
    )
    required = max(2 * int(np.sum(harmonic_counts)) + 1, 256)
    points = 1 << int(math.ceil(math.log2(required)))
    return min(points, max(int(maximum_points), 256))


def _wrapped_gaussian_log_evidence(
    session_offsets: np.ndarray,
    broadened_variances: np.ndarray,
    *,
    grid_size: int,
    log_tolerance: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    # Subtracting the first offset makes common-gauge invariance exact even
    # though the remaining one-dimensional integral uses a finite grid.
    relative = _wrapped_phase(session_offsets - float(session_offsets[0]))
    gauge_grid = np.arange(int(grid_size), dtype=float) / float(grid_size)
    log_integrand = np.zeros(gauge_grid.size, dtype=float)
    for offset, variance in zip(relative, broadened_variances, strict=True):
        log_integrand += _wrapped_normal_log_density(
            offset - gauge_grid,
            float(variance),
            log_tolerance=log_tolerance,
        )
    log_evidence = float(logsumexp(log_integrand) - math.log(gauge_grid.size))
    return log_evidence, relative, gauge_grid, log_integrand


@lru_cache(maxsize=128)
def _wrapped_gaussian_reference_log_evidence(
    variances: tuple[float, ...],
    phase_sigma: float,
    log_tolerance: float,
    maximum_points: int,
) -> tuple[float, int]:
    variance_array = np.asarray(variances, dtype=float)
    grid_size = _wrapped_gaussian_grid_size(
        variance_array,
        phase_sigma=phase_sigma,
        log_tolerance=log_tolerance,
        maximum_points=maximum_points,
    )
    broadened = variance_array + float(phase_sigma) ** 2
    log_evidence, _, _, _ = _wrapped_gaussian_log_evidence(
        np.zeros(variance_array.size, dtype=float),
        broadened,
        grid_size=grid_size,
        log_tolerance=log_tolerance,
    )
    return log_evidence, grid_size


def wrapped_gaussian_phase_deviance(
    session_offsets: Sequence[float],
    session_variances: Sequence[float],
    *,
    phase_sigma: float,
    log_tolerance: float = -27.631021115928547,
    maximum_grid_points: int = 16_384,
) -> tuple[float, float, tuple[float, ...]]:
    """Marginalize every integer phase and the common phase gauge exactly.

    The likelihood integrates the product of per-session wrapped Gaussians over
    one common phase gauge. It preserves the circular harmonic structure of
    Gaussian TOA errors without using a von Mises approximation.
    """

    offsets = np.asarray(session_offsets, dtype=float)
    variances = np.asarray(session_variances, dtype=float)
    if offsets.ndim != 1 or variances.shape != offsets.shape or offsets.size == 0:
        raise ValueError("Session phase arrays must be non-empty and aligned.")
    if not np.all(np.isfinite(offsets)) or not np.all(
        np.isfinite(variances) & (variances > 0.0)
    ):
        raise ValueError("Session phases and variances must be finite and positive.")
    sigma = float(phase_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("The wrapped-Gaussian phase sigma must be positive.")

    variance_key = tuple(float(value) for value in variances)
    reference_log_evidence, grid_size = _wrapped_gaussian_reference_log_evidence(
        variance_key,
        sigma,
        float(log_tolerance),
        max(int(maximum_grid_points), 256),
    )
    broadened = variances + sigma * sigma
    log_evidence, relative, gauge_grid, log_integrand = (
        _wrapped_gaussian_log_evidence(
            offsets,
            broadened,
            grid_size=grid_size,
            log_tolerance=log_tolerance,
        )
    )
    deviance = max(2.0 * (reference_log_evidence - log_evidence), 0.0)

    maximum_index = int(np.argmax(log_integrand))
    previous_value = float(log_integrand[(maximum_index - 1) % grid_size])
    center_value = float(log_integrand[maximum_index])
    next_value = float(log_integrand[(maximum_index + 1) % grid_size])
    curvature = previous_value - 2.0 * center_value + next_value
    subgrid = 0.0
    if np.isfinite(curvature) and curvature < 0.0:
        subgrid = float(
            np.clip(
                0.5 * (previous_value - next_value) / curvature,
                -0.5,
                0.5,
            )
        )
    gauge = (float(gauge_grid[maximum_index]) + subgrid / grid_size) % 1.0
    signature = _wrapped_phase(relative - gauge)

    inverse_variance = 1.0 / broadened
    resultant = np.sum(inverse_variance * np.exp(2.0j * math.pi * relative))
    coherence = float(
        np.clip(abs(resultant) / float(np.sum(inverse_variance)), 0.0, 1.0)
    )
    return deviance, coherence, tuple(float(value) for value in signature)


def wrapped_gaussian_phase_residuals(
    session_offsets: Sequence[float],
    session_variances: Sequence[float],
    *,
    phase_sigma: float,
    log_tolerance: float = -27.631021115928547,
    maximum_grid_points: int = 16_384,
) -> np.ndarray:
    """Factor exact wrapped-Gaussian evidence into session-resolved residuals."""

    offsets = np.asarray(session_offsets, dtype=float)
    variances = np.asarray(session_variances, dtype=float)
    deviance, _, signature_values = wrapped_gaussian_phase_deviance(
        offsets,
        variances,
        phase_sigma=phase_sigma,
        log_tolerance=log_tolerance,
        maximum_grid_points=maximum_grid_points,
    )
    signature = np.asarray(signature_values, dtype=float)
    broadened = variances + float(phase_sigma) ** 2
    log_peak = np.asarray(
        [
            float(
                _wrapped_normal_log_density(
                    np.asarray(0.0),
                    float(variance),
                    log_tolerance=log_tolerance,
                )
            )
            for variance in broadened
        ],
        dtype=float,
    )
    log_session = np.asarray(
        [
            float(
                _wrapped_normal_log_density(
                    np.asarray(residual),
                    float(variance),
                    log_tolerance=log_tolerance,
                )
            )
            for residual, variance in zip(signature, broadened, strict=True)
        ],
        dtype=float,
    )
    conditional_deviance = np.maximum(2.0 * (log_peak - log_session), 0.0)
    raw = np.sign(signature) * np.sqrt(conditional_deviance)
    raw_norm2 = float(raw @ raw)
    if deviance <= np.finfo(float).eps or raw_norm2 <= np.finfo(float).tiny:
        return np.zeros_like(offsets)
    return math.sqrt(deviance / raw_norm2) * raw


def _phase_deviance(
    session_offsets: Sequence[float],
    session_variances: Sequence[float],
    *,
    phase_sigma: float,
    config: FullSequenceCircularConfig,
) -> tuple[float, float, tuple[float, ...]]:
    model = str(config.phase_likelihood).strip().lower()
    if model == "wrapped-gaussian":
        return wrapped_gaussian_phase_deviance(
            session_offsets,
            session_variances,
            phase_sigma=phase_sigma,
            log_tolerance=config.wrapped_gaussian_log_tolerance,
            maximum_grid_points=config.wrapped_gaussian_max_grid_points,
        )
    if model == "von-mises":
        return circular_phase_deviance(
            session_offsets,
            session_variances,
            phase_sigma=phase_sigma,
        )
    raise ValueError(f"Unsupported full-sequence phase likelihood: {model!r}.")


def _phase_residuals(
    session_offsets: Sequence[float],
    session_variances: Sequence[float],
    *,
    phase_sigma: float,
    config: FullSequenceCircularConfig,
) -> np.ndarray:
    model = str(config.phase_likelihood).strip().lower()
    if model == "wrapped-gaussian":
        return wrapped_gaussian_phase_residuals(
            session_offsets,
            session_variances,
            phase_sigma=phase_sigma,
            log_tolerance=config.wrapped_gaussian_log_tolerance,
            maximum_grid_points=config.wrapped_gaussian_max_grid_points,
        )
    if model == "von-mises":
        return circular_phase_residuals(
            session_offsets,
            session_variances,
            phase_sigma=phase_sigma,
        )
    raise ValueError(f"Unsupported full-sequence phase likelihood: {model!r}.")


def _normalized_bounds(coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("Normalized coordinate bounds must be aligned.")
    return lower, upper







def _covariance_from_jacobian(
    jacobian: np.ndarray,
    *,
    support_precision: float = 3.0,
) -> np.ndarray:
    """Return local covariance without forming ill-conditioned normal equations."""

    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or not np.all(np.isfinite(jac)):
        raise ValueError("A circular-transport Jacobian must be finite and two-dimensional.")
    dimension = int(jac.shape[1])
    _left, singular_values, right = np.linalg.svd(jac, full_matrices=True)
    precision_floor = max(float(support_precision), np.finfo(float).eps)
    inverse = np.full(dimension, 1.0 / precision_floor, dtype=float)
    represented = min(int(singular_values.size), dimension)
    with np.errstate(over="ignore"):
        precision = np.square(singular_values[:represented]) + precision_floor
    inverse[:represented] = 1.0 / precision
    covariance = (right.T * inverse[np.newaxis, :]) @ right
    return 0.5 * (covariance + covariance.T)


def _covariance_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (
        np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, scale * 1.0e-12)
    order = np.argsort(clipped)[::-1]
    return eigenvectors[:, order] @ np.diag(np.sqrt(clipped[order]))


def _laplace_log_mass(objective_chi2: float, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        return -math.inf
    dimension = covariance.shape[0]
    return float(
        -0.5 * float(objective_chi2)
        + 0.5 * float(logdet)
        + 0.5 * dimension * math.log(2.0 * math.pi)
    )


def _absolute_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    step: float,
) -> np.ndarray:
    center = np.asarray(values, dtype=float)
    baseline = np.asarray(function(center), dtype=float)
    jacobian = np.empty((baseline.size, center.size), dtype=float)
    displacement = max(float(step), 1.0e-6)
    for index in range(center.size):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(float(plus[index] + displacement), float(upper[index]))
        minus[index] = max(float(minus[index] - displacement), float(lower[index]))
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            jacobian[:, index] = 0.0
            continue
        jacobian[:, index] = (
            np.asarray(function(plus), dtype=float)
            - np.asarray(function(minus), dtype=float)
        ) / denominator
    return jacobian


def _phase_signature_distance(
    left: FullSequenceCircularMode,
    right: FullSequenceCircularMode,
) -> float:
    left_signature = np.asarray(left.phase_signature, dtype=float)
    right_signature = np.asarray(right.phase_signature, dtype=float)
    difference = (left_signature - right_signature + 0.5) % 1.0 - 0.5
    return float(np.sqrt(np.mean(np.square(difference))))


def _normalize_mode_masses(
    modes: Sequence[FullSequenceCircularMode],
) -> tuple[FullSequenceCircularMode, ...]:
    if not modes:
        return ()
    log_masses = np.asarray([mode.log_mass for mode in modes], dtype=float)
    finite = np.isfinite(log_masses)
    masses = np.zeros(len(modes), dtype=float)
    if np.any(finite):
        masses[finite] = np.exp(log_masses[finite] - logsumexp(log_masses[finite]))
    else:
        masses.fill(1.0 / len(modes))
    return tuple(
        replace(mode, posterior_mass=float(mass))
        for mode, mass in zip(modes, masses, strict=True)
    )


def _select_circular_modes(
    modes: Sequence[FullSequenceCircularMode],
    *,
    maximum_modes: int,
    minimum_normalized_separation: float,
    minimum_phase_separation: float,
) -> tuple[FullSequenceCircularMode, ...]:
    ranked = sorted(modes, key=lambda mode: (-mode.log_mass, mode.objective_chi2))
    target = max(int(maximum_modes), 1)
    selected: list[FullSequenceCircularMode] = []

    def add(candidate: FullSequenceCircularMode) -> bool:
        for existing in selected:
            normalized_distance = float(
                np.linalg.norm(candidate.normalized_values - existing.normalized_values)
            )
            if normalized_distance < 1.0e-8:
                return False
            if (
                normalized_distance < max(float(minimum_normalized_separation), 0.0)
                and _phase_signature_distance(candidate, existing)
                < max(float(minimum_phase_separation), 0.0)
            ):
                return False
        selected.append(candidate)
        return True

    # These are proposal-coverage quotas. Final scientific ranking still uses
    # the common exact fixed-path target after integer decoding.
    add(min(ranked, key=lambda mode: (mode.shape_chi2, mode.objective_chi2)))
    add(max(ranked, key=lambda mode: (mode.phase_coherence, -mode.objective_chi2)))
    add(min(ranked, key=lambda mode: mode.objective_chi2))
    if len(selected) >= target:
        return _normalize_mode_masses(selected)

    best_by_root: dict[str, FullSequenceCircularMode] = {}
    for mode in ranked:
        best_by_root.setdefault(mode.root_mode_id, mode)
    for mode in sorted(best_by_root.values(), key=lambda item: (-item.log_mass, item.objective_chi2)):
        add(mode)
        if len(selected) >= target:
            return _normalize_mode_masses(selected)
    for mode in ranked:
        add(mode)
        if len(selected) >= target:
            break
    return _normalize_mode_masses(selected)


def _initial_circular_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceCircularConfig,
    maximum_modes: int,
) -> tuple[FullSequenceCircularMode, ...]:
    output: list[FullSequenceCircularMode] = []
    for index, invariant in enumerate(
        sorted(invariant_modes, key=lambda mode: (mode.objective_chi2, -mode.log_mass))[
            : max(int(maximum_modes), 1)
        ]
    ):
        normalized = _normalized_values(coordinates, invariant.parameter_values)
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        deviance, coherence, signature = _phase_deviance(
            profile.session_offsets,
            reference_variances,
            phase_sigma=phase_sigma,
            config=config,
        )
        shape = float(profile.chi2)
        prior = max(float(config.prior_precision), 0.0) * float(
            normalized @ normalized
        )
        objective = shape + deviance + prior
        output.append(
            FullSequenceCircularMode(
                mode_id=f"circular_seed_{index:04d}",
                root_mode_id=str(invariant.mode_id),
                parent_mode_id=None,
                proposal_source="invariant-root",
                parameter_values=_physical_values(coordinates, normalized),
                normalized_values=normalized,
                covariance=np.asarray(invariant.covariance, dtype=float).copy(),
                phase_sigma=float(phase_sigma),
                shape_chi2=shape,
                circular_deviance=deviance,
                prior_chi2=prior,
                objective_chi2=objective,
                log_mass=_laplace_log_mass(objective, invariant.covariance),
                posterior_mass=0.0,
                phase_coherence=coherence,
                phase_signature=signature,
                nfev=0,
                success=bool(invariant.success),
            )
        )
    return _normalize_mode_masses(output)


def _phase_informed_sigma_point_starts(
    mode: FullSequenceCircularMode,
    *,
    evaluator: Any,
    coordinates: Any,
    reference_variances: np.ndarray,
    phase_sigma: float,
    count: int,
    radius: float,
    radii: Sequence[float],
    probe_step: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[tuple[np.ndarray, str], ...]:
    target = max(int(count), 1)
    root = _covariance_root(mode.covariance)
    starts: list[tuple[np.ndarray, str]] = [
        (np.asarray(mode.normalized_values, dtype=float).copy(), "continuation")
    ]
    pair_count = min((target - 1) // 2, root.shape[1])
    directions = np.eye(root.shape[1], dtype=float)
    singular_values = np.zeros(root.shape[1], dtype=float)
    try:
        center = np.asarray(mode.normalized_values, dtype=float)
        displacement = max(float(probe_step), 1.0e-4)
        offset_jacobian = np.empty(
            (len(mode.phase_signature), center.size), dtype=float
        )
        for axis in range(center.size):
            plus = np.clip(center + displacement * root[:, axis], lower, upper)
            minus = np.clip(center - displacement * root[:, axis], lower, upper)
            denominator = 2.0 * displacement
            plus_offsets = np.asarray(
                evaluator.evaluate_values(
                    _physical_values(coordinates, plus)
                ).session_offsets,
                dtype=float,
            )
            minus_offsets = np.asarray(
                evaluator.evaluate_values(
                    _physical_values(coordinates, minus)
                ).session_offsets,
                dtype=float,
            )
            offset_jacobian[:, axis] = (plus_offsets - minus_offsets) / denominator
        concentration = 1.0 / (
            4.0
            * math.pi
            * math.pi
            * (reference_variances + float(phase_sigma) ** 2)
        )
        weights = concentration / float(np.sum(concentration))
        gauge_derivative = weights @ offset_jacobian
        gauge_free = offset_jacobian - gauge_derivative[None, :]
        weighted = np.sqrt(concentration)[:, None] * gauge_free
        _, singular_values, right_vectors = np.linalg.svd(
            weighted,
            full_matrices=False,
        )
        directions = right_vectors
    except Exception:
        pass
    declared_radii = tuple(
        float(value)
        for value in radii
        if np.isfinite(float(value)) and float(value) > 0.0
    ) or (max(float(radius), 0.0),)
    for shell_radius in declared_radii:
        for axis in range(pair_count):
            local_direction = directions[axis]
            direction = shell_radius * (root @ local_direction)
            starts.append(
                (
                    np.clip(mode.normalized_values + direction, lower, upper),
                    (
                        f"phase-svd-plus-{axis}-r{shell_radius:.6g}-"
                        f"{singular_values[axis]:.6g}"
                    ),
                )
            )
            starts.append(
                (
                    np.clip(mode.normalized_values - direction, lower, upper),
                    (
                        f"phase-svd-minus-{axis}-r{shell_radius:.6g}-"
                        f"{singular_values[axis]:.6g}"
                    ),
                )
            )
    return tuple(starts)


def _rescore_circular_mode(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: FullSequenceCircularMode,
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceCircularConfig,
    mode_id: str,
) -> FullSequenceCircularMode:
    normalized = np.asarray(mode.normalized_values, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
    deviance, coherence, signature = _phase_deviance(
        profile.session_offsets,
        reference_variances,
        phase_sigma=phase_sigma,
        config=config,
    )
    shape = float(profile.chi2)
    prior = max(float(config.prior_precision), 0.0) * float(
        normalized @ normalized
    )
    objective = shape + deviance + prior
    return replace(
        mode,
        mode_id=mode_id,
        parent_mode_id=mode.mode_id,
        proposal_source="temperature-control",
        parameter_values=_physical_values(coordinates, normalized),
        phase_sigma=float(phase_sigma),
        shape_chi2=shape,
        circular_deviance=deviance,
        prior_chi2=prior,
        objective_chi2=objective,
        log_mass=_laplace_log_mass(objective, mode.covariance),
        posterior_mass=0.0,
        phase_coherence=coherence,
        phase_signature=signature,
        nfev=1,
    )


def _transport_start(
    *,
    evaluator: Any,
    coordinates: Any,
    parent: FullSequenceCircularMode,
    normalized_start: np.ndarray,
    proposal_source: str,
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceCircularConfig,
    mode_id: str,
) -> FullSequenceCircularMode:
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(np.asarray(normalized_start, dtype=float), lower, upper)
    root = _covariance_root(parent.covariance)
    trust = max(float(config.local_trust_radius), 1.0e-4)
    local_lower = np.full(start.size, -trust, dtype=float)
    local_upper = np.full(start.size, trust, dtype=float)
    profile_cache: dict[tuple[float, ...], Any] = {}
    evaluation_count = 0

    def profile_at(local_values: np.ndarray) -> tuple[np.ndarray, Any]:
        nonlocal evaluation_count
        normalized = start + root @ np.asarray(local_values, dtype=float)
        if np.any(normalized < lower) or np.any(normalized > upper):
            raise ValueError("The circular transport left the declared prior bounds.")
        key = tuple(float(value) for value in np.round(normalized, 14))
        if key not in profile_cache:
            profile_cache[key] = evaluator.evaluate_values(
                _physical_values(coordinates, normalized)
            )
            evaluation_count += 1
        return normalized, profile_cache[key]

    _, start_profile = profile_at(np.zeros(start.size, dtype=float))
    shape_size = int(np.asarray(start_profile.normalized_residuals, dtype=float).size)
    prior_active = max(float(config.prior_precision), 0.0) > 0.0
    phase_size = int(reference_variances.size)
    failure = np.full(
        shape_size + phase_size + (start.size if prior_active else 0),
        1.0e10,
    )

    def residual_vector(local_values: np.ndarray) -> np.ndarray:
        try:
            normalized, profile = profile_at(local_values)
            phase_residuals = _phase_residuals(
                profile.session_offsets,
                reference_variances,
                phase_sigma=phase_sigma,
                config=config,
            )
            parts = [
                np.asarray(profile.normalized_residuals, dtype=float),
                phase_residuals,
            ]
            if prior_active:
                parts.append(math.sqrt(float(config.prior_precision)) * normalized)
            result = np.concatenate(parts)
            return result if result.shape == failure.shape and np.all(np.isfinite(result)) else failure
        except Exception:
            return failure

    optimization = least_squares(
        residual_vector,
        np.zeros(start.size, dtype=float),
        bounds=(local_lower, local_upper),
        method="trf",
        jac=lambda values: _absolute_difference_jacobian(
            residual_vector,
            np.asarray(values, dtype=float),
            lower=local_lower,
            upper=local_upper,
            step=max(float(config.absolute_jacobian_step), 1.0e-6),
        ),
        max_nfev=max(int(config.max_nfev_per_start), 1),
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    final_local = np.asarray(optimization.x, dtype=float)
    final_normalized, final_profile = profile_at(final_local)
    deviance, coherence, signature = _phase_deviance(
        final_profile.session_offsets,
        reference_variances,
        phase_sigma=phase_sigma,
        config=config,
    )
    shape = float(final_profile.chi2)
    prior = max(float(config.prior_precision), 0.0) * float(
        final_normalized @ final_normalized
    )
    objective = shape + deviance + prior
    local_covariance = _covariance_from_jacobian(
        np.asarray(optimization.jac, dtype=float),
    )
    covariance = root @ local_covariance @ root.T
    covariance = 0.5 * (covariance + covariance.T)
    return FullSequenceCircularMode(
        mode_id=mode_id,
        root_mode_id=parent.root_mode_id,
        parent_mode_id=parent.mode_id,
        proposal_source=proposal_source,
        parameter_values=_physical_values(coordinates, final_normalized),
        normalized_values=final_normalized,
        covariance=covariance,
        phase_sigma=float(phase_sigma),
        shape_chi2=shape,
        circular_deviance=deviance,
        prior_chi2=prior,
        objective_chi2=objective,
        log_mass=_laplace_log_mass(objective, covariance),
        posterior_mass=0.0,
        phase_coherence=coherence,
        phase_signature=signature,
        nfev=int(evaluation_count),
        success=bool(optimization.success),
    )


def transport_full_sequence_circular_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: FullSequenceCircularConfig | None = None,
    resume_modes: Sequence[FullSequenceCircularMode] | None = None,
    progress_callback: Callable[[CircularTemperatureSummary], None] | None = None,
) -> tuple[
    tuple[FullSequenceCircularMode, ...],
    tuple[CircularTemperatureSummary, ...],
    np.ndarray,
]:
    """Transport a multimode posterior with every session active at every step."""

    active = config or FullSequenceCircularConfig()
    ladder = tuple(
        float(value)
        for value in active.phase_sigma_ladder
        if np.isfinite(float(value)) and float(value) > 0.0
    )
    if not ladder:
        raise ValueError("At least one positive circular phase sigma is required.")
    if any(right >= left for left, right in zip(ladder, ladder[1:], strict=False)):
        raise ValueError("The circular phase sigma ladder must be strictly decreasing.")
    if not invariant_modes:
        raise ValueError("Full-sequence transport requires invariant posterior modes.")
    if str(active.phase_likelihood).strip().lower() not in {
        "wrapped-gaussian",
        "von-mises",
    }:
        raise ValueError(
            "The full-sequence phase likelihood must be 'wrapped-gaussian' "
            "or 'von-mises'."
        )

    reference_profile = evaluator.evaluate_values(
        np.asarray(invariant_modes[0].parameter_values, dtype=float)
    )
    reference_variances = np.asarray(
        reference_profile.session_offset_variances, dtype=float
    )
    lower, upper = _normalized_bounds(coordinates)
    if resume_modes:
        previous_sigma = min(float(mode.phase_sigma) for mode in resume_modes)
        if ladder[0] >= previous_sigma:
            raise ValueError(
                "A resumed circular ladder must start below the checkpoint sigma."
            )
        current = _normalize_mode_masses(tuple(resume_modes))
    else:
        current = _initial_circular_modes(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            reference_variances=reference_variances,
            phase_sigma=ladder[0],
            config=active,
            maximum_modes=active.initial_mode_count,
        )
    history: list[CircularTemperatureSummary] = []
    mode_serial = 0
    for temperature_index, phase_sigma in enumerate(ladder):
        parents = tuple(
            sorted(current, key=lambda mode: (-mode.log_mass, mode.objective_chi2))[
                : max(int(active.parents_per_temperature), 1)
            ]
        )
        starts: list[tuple[FullSequenceCircularMode, np.ndarray, str]] = []
        for parent in parents:
            for start, source in _phase_informed_sigma_point_starts(
                parent,
                evaluator=evaluator,
                coordinates=coordinates,
                reference_variances=reference_variances,
                phase_sigma=phase_sigma,
                count=active.sigma_points_per_parent,
                radius=active.sigma_point_radius,
                radii=active.sigma_point_radii,
                probe_step=active.absolute_jacobian_step,
                lower=lower,
                upper=upper,
            ):
                starts.append((parent, start, source))

        optimized: list[FullSequenceCircularMode] = []
        for parent, start, source in starts:
            try:
                optimized.append(
                    _transport_start(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        parent=parent,
                        normalized_start=start,
                        proposal_source=source,
                        reference_variances=reference_variances,
                        phase_sigma=phase_sigma,
                        config=active,
                        mode_id=f"circular_t{temperature_index:02d}_{mode_serial:05d}",
                    )
                )
                mode_serial += 1
            except Exception:
                continue
        controls: list[FullSequenceCircularMode] = []
        for parent in parents:
            try:
                controls.append(
                    _rescore_circular_mode(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        mode=parent,
                        reference_variances=reference_variances,
                        phase_sigma=phase_sigma,
                        config=active,
                        mode_id=f"circular_t{temperature_index:02d}_{mode_serial:05d}",
                    )
                )
                mode_serial += 1
            except Exception:
                continue
        if not optimized and not controls:
            raise RuntimeError(
                f"Full-sequence circular transport failed at sigma={phase_sigma}."
            )
        current = _select_circular_modes(
            tuple(optimized) + tuple(controls),
            maximum_modes=active.maximum_modes,
            minimum_normalized_separation=active.minimum_normalized_separation,
            minimum_phase_separation=active.minimum_phase_separation,
        )
        summary = CircularTemperatureSummary(
            temperature_index=temperature_index,
            phase_sigma=phase_sigma,
            parent_count=len(parents),
            start_count=len(starts),
            optimized_count=len(optimized),
            selected_count=len(current),
            root_count=len({mode.root_mode_id for mode in current}),
            best_shape_chi2=min(mode.shape_chi2 for mode in current),
            best_circular_deviance=min(mode.circular_deviance for mode in current),
            best_objective_chi2=min(mode.objective_chi2 for mode in current),
            maximum_phase_coherence=max(mode.phase_coherence for mode in current),
        )
        history.append(summary)
        if progress_callback is not None:
            progress_callback(summary)
    return current, tuple(history), reference_variances


def recover_full_sequence_integer_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    circular_modes: Sequence[FullSequenceCircularMode],
    config: FullSequenceCircularConfig | None = None,
) -> tuple[tuple[GlobalLatticeMode, ...], int]:
    """Conditionally decode complete paths only after circular transport."""

    active = config or FullSequenceCircularConfig()
    if not circular_modes:
        return (), 0
    proposals_by_key: dict[tuple[int, ...], LatticeProposal] = {}
    ranked_parents = sorted(
        circular_modes, key=lambda mode: (-mode.log_mass, mode.objective_chi2)
    )[: max(int(active.path_parent_modes), 1)]
    session_count = 0
    for mode in ranked_parents:
        profile = evaluator.evaluate_values(mode.parameter_values)
        session_count = int(profile.session_count)
        for candidate in conditional_integer_paths(
            profile,
            candidate_count=max(int(active.integer_candidates_per_mode), 1),
        ):
            proposal = LatticeProposal(
                root_mode_id=mode.root_mode_id,
                parent_mode_id=mode.mode_id,
                source="full-sequence-circular-conditional",
                active_sessions=tuple(range(1, session_count)),
                integer_offsets=tuple(candidate.integer_offsets),
                initial_normalized_values=mode.normalized_values.copy(),
                squared_mahalanobis=max(-2.0 * float(candidate.log_conditional_mass), 0.0),
                log_predictive_density=float(mode.log_mass + candidate.log_conditional_mass),
                kappa=1.0,
            )
            previous = proposals_by_key.get(proposal.integer_offsets)
            if previous is None or proposal.log_predictive_density > previous.log_predictive_density:
                proposals_by_key[proposal.integer_offsets] = proposal
    proposals = sorted(
        proposals_by_key.values(),
        key=lambda proposal: -proposal.log_predictive_density,
    )[: max(int(active.maximum_path_proposals), 1)]
    exact_config = GlobalLatticeTrackerConfig(
        prior_precision=max(float(active.prior_precision), 0.0),
        fixed_path_max_nfev=max(int(active.exact_refit_max_nfev), 1),
        phase_tempering_ladder=tuple(active.exact_phase_tempering_ladder),
        maximum_lattice_nodes=max(int(active.maximum_lattice_nodes), 1),
    )
    exact_modes: list[GlobalLatticeMode] = []
    for index, proposal in enumerate(proposals):
        try:
            exact_modes.append(
                refit_fixed_path_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    anchor_session=0,
                    proposal=proposal,
                    config=exact_config,
                    mode_id=f"full_sequence_exact_{index:04d}",
                )
            )
        except Exception:
            continue
    selected = select_diverse_lattice_modes(
        exact_modes,
        maximum_modes=max(int(active.maximum_exact_modes), 1),
        maximum_modes_per_path=max(int(active.maximum_exact_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_exact_modes_per_root), 1),
        minimum_separation=max(float(active.exact_mode_separation), 0.0),
    )
    return selected, len(proposals)


def run_full_sequence_circular_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: FullSequenceCircularConfig | None = None,
    resume_modes: Sequence[FullSequenceCircularMode] | None = None,
    progress_callback: Callable[[CircularTemperatureSummary], None] | None = None,
) -> FullSequenceCircularResult:
    active = config or FullSequenceCircularConfig()
    circular_modes, history, _ = transport_full_sequence_circular_modes(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=invariant_modes,
        config=active,
        resume_modes=resume_modes,
        progress_callback=progress_callback,
    )
    exact_modes, proposal_count = recover_full_sequence_integer_modes(
        evaluator=evaluator,
        coordinates=coordinates,
        circular_modes=circular_modes,
        config=active,
    )
    source_session_count = 0
    if circular_modes:
        source_session_count = int(
            evaluator.evaluate_values(circular_modes[0].parameter_values).session_count
        )
    return FullSequenceCircularResult(
        circular_modes=circular_modes,
        exact_modes=exact_modes,
        temperature_history=history,
        path_proposal_count=proposal_count,
        source_session_count=source_session_count,
        reached_final_temperature=bool(
            history
            and math.isclose(
                history[-1].phase_sigma,
                active.phase_sigma_ladder[-1],
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ),
    )


__all__ = [
    "CircularTemperatureSummary",
    "FullSequenceCircularConfig",
    "FullSequenceCircularMode",
    "FullSequenceCircularResult",
    "circular_phase_deviance",
    "circular_phase_residuals",
    "recover_full_sequence_integer_modes",
    "run_full_sequence_circular_transport",
    "transport_full_sequence_circular_modes",
    "wrapped_gaussian_phase_deviance",
    "wrapped_gaussian_phase_residuals",
]
