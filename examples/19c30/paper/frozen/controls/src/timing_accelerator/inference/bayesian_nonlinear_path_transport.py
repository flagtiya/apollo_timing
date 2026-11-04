from __future__ import annotations

"""Tempered Bayesian transport of complete pulsar phase paths.

The phase forest is useful for discovering locally compatible islands, but a
partition can become an irreversible commitment when the DD phase trajectory
is strongly nonlinear.  This module instead keeps complete integer paths as
the discrete posterior state.  At every temperature it reopens the whole
path, refits the exact PINT likelihood, and transports a diverse mode bank to
the next, narrower wrapped-phase posterior.
"""

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from timing_accelerator.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from timing_accelerator.inference.collapsed_integer_posterior import (
    collapsed_wrapped_log_likelihood,
)


@dataclass(frozen=True)
class NonlinearPathTransportConfig:
    """Controls for complete-path posterior transport."""

    prior_precision: float = 1.0
    phase_noise_ladder: tuple[float, ...] = (
        0.50,
        0.35,
        0.25,
        0.18,
        0.12,
        0.08,
        0.05,
        0.03,
        0.02,
        0.01,
        0.005,
        0.0,
    )
    # The production hard-source route tempers the exact phase likelihood.
    # ``phase_noise_ladder`` remains available only for controlled ablations.
    adaptive_phase_tempering: bool = True
    phase_beta_initial: float = 1.0e-8
    phase_beta_ess_target_fraction: float = 0.70
    phase_beta_minimum_step: float = 1.0e-8
    phase_beta_maximum_step: float = 0.20
    phase_beta_maximum_growth_factor: float = 4.0
    phase_beta_maximum_steps: int = 64
    phase_beta_refinement_steps: int = 2
    protected_ancestor_modes: int = 1
    phase_beta_overlap_samples: int = 8
    initial_samples_per_mode: int = 96
    refresh_samples_per_mode: int = 24
    initial_scales: tuple[float, ...] = (
        0.04,
        0.08,
        0.16,
        0.32,
        0.64,
        1.0,
        4.0,
        16.0,
        64.0,
    )
    initial_path_families: int = 32
    refresh_path_families: int = 16
    prefit_path_families: int = 48
    path_beam: int = 24
    maximum_modes_per_path: int = 2
    maximum_starts_per_path: int = 3
    proposal_mode_separation: float = 0.005
    point_path_candidates: int = 2
    collapsed_starts_per_mode: int = 4
    collapsed_refresh_starts_per_mode: int = 2
    collapsed_maxiter: int = 40
    collapsed_hermite_order: int = 16
    collapsed_maximum_wrap_terms: int = 7
    joint_refinement_rounds: int = 4
    joint_path_candidates: int = 4
    island_bridge_counts: tuple[int, ...] = (2, 3)
    island_parent_modes: int = 2
    island_phase_noise: float = 0.03
    island_path_candidates: int = 2
    island_joint_refinement_rounds: int = 3
    island_mode_beam: int = 8
    island_lattice_candidates_per_scale: int = 8
    maximum_island_lattice_proposals: int = 64
    island_collapsed_noise_ladder: tuple[float, ...] = (0.30, 0.10, 0.03, 0.01, 0.0)
    island_collapsed_starts_per_mode: int = 4
    island_collapsed_maxiter: int = 48
    bridge_gauge_candidates_per_scale: int = 8
    maximum_island_bridge_proposals: int = 32
    bridge_block_counts: tuple[int, ...] = (2, 3, 4, 6, 8, 12)
    bridge_block_candidates_per_scale: int = 2
    maximum_block_proposals_per_mode: int = 24
    lattice_kappas: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    lattice_candidates_per_scale: int = 4
    maximum_lattice_nodes: int = 2_000_000
    jitter_samples_per_mode: int = 4
    jitter_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    fit_max_nfev: int = 48
    finite_difference_step: float = 1.0e-6
    phase_step_cycles: float = 0.03
    mode_separation: float = 0.01
    random_seed: int = 20260724


@dataclass(frozen=True)
class CompletePathProposal:
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    source: str
    predictive_score: float
    parent_mode_id: str | None = None


@dataclass(frozen=True)
class NonlinearPathMode:
    mode_id: str
    parent_mode_id: str | None
    proposal_source: str
    temperature_index: int
    phase_noise: float
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
    nfev: int
    path_update_count: int
    success: bool
    phase_beta: float = 1.0

    def chronological_integer_offsets(self, session_count: int) -> tuple[int, ...]:
        if len(self.integer_offsets) != int(session_count) - 1:
            raise ValueError("The complete path does not match the session count.")
        return self.integer_offsets


@dataclass(frozen=True)
class NonlinearPathStageSummary:
    temperature_index: int
    phase_noise: float
    input_mode_count: int
    proposal_count: int
    unique_path_count: int
    inherited_path_count: int
    reopened_path_count: int
    collapsed_mode_count: int
    joint_path_update_count: int
    block_path_count: int
    island_bridge_path_count: int
    refitted_count: int
    retained_count: int
    best_shape_chi2: float
    best_phase_chi2: float
    best_objective_chi2: float
    phase_beta: float = 1.0
    effective_sample_size: float = 0.0
    ancestor_overlap: float = 0.0


@dataclass(frozen=True)
class NonlinearPathTransportResult:
    session_count: int
    final_modes: tuple[NonlinearPathMode, ...]
    stage_history: tuple[NonlinearPathStageSummary, ...]
    level_portfolios: tuple[tuple[NonlinearPathMode, ...], ...]
    reached_all_sessions: bool
    failure_class: str | None


def _normalized_bounds(coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("Normalized coordinate bounds must have matching dimensions.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
        raise ValueError("Normalized coordinate bounds must be finite and ordered.")
    return lower, upper


def _physical_values(coordinates: Any, normalized_values: Sequence[float]) -> np.ndarray:
    search = coordinates.denormalize(np.asarray(normalized_values, dtype=float))
    return np.asarray(coordinates.to_physical(search), dtype=float)


def _stable_covariance(matrix: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    if symmetric.size == 0:
        return symmetric.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors * clipped) @ eigenvectors.T


def _stable_inverse(matrix: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    if symmetric.size == 0:
        return symmetric.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors / clipped) @ eigenvectors.T


def relative_phase_geometry(profile: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return all session phases relative to session zero and their covariance."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    if offsets.ndim != 1 or variances.shape != offsets.shape or offsets.size < 2:
        raise ValueError("At least two aligned session offsets are required.")
    relative = offsets[1:] - offsets[0]
    covariance = np.diag(variances[1:]) + float(variances[0])
    return relative, _stable_covariance(covariance)


def nearest_complete_path(profile: Any) -> tuple[int, ...]:
    """Return a cheap complete path used only for proposal ownership."""

    relative, _ = relative_phase_geometry(profile)
    return tuple(int(value) for value in np.rint(relative))


def complete_path_candidates(
    profile: Any,
    *,
    candidate_count: int = 2,
    maximum_lattice_nodes: int = 2_000_000,
    phase_noise: float = 0.0,
    phase_beta: float = 1.0,
) -> tuple[tuple[int, ...], ...]:
    """Return high-probability complete paths from the correlated phase lattice.

    Relative session offsets share the anchor uncertainty, so independently
    rounding every offset is not the Bayesian lattice rule.  The integer
    least-squares enumeration is still only a proposal mechanism; every path
    is subsequently refit against the nonlinear evaluator.
    """

    relative, covariance = relative_phase_geometry(profile)
    noise = max(float(phase_noise), 0.0)
    if noise > 0.0:
        covariance = covariance + noise * noise * np.eye(relative.size, dtype=float)
    # The proposal geometry is posterior-predictive, not the tempered target.
    # Scaling this covariance by 1 / beta would make the integer lattice
    # explode while phase information is intentionally absent. The target
    # correction is applied by ``tempered_complete_path_residual`` instead.
    result = solve_integer_least_squares(
        relative,
        covariance,
        candidate_count=max(int(candidate_count), 1),
        max_visited_nodes=max(int(maximum_lattice_nodes), 1),
    )
    paths = [
        tuple(int(value) for value in candidate.integer_vector)
        for candidate in result.candidates
    ]
    if not paths:
        return (nearest_complete_path(profile),)
    return tuple(paths)


def temporal_bridge_block_basis(
    profile: Any,
    *,
    block_count: int,
) -> np.ndarray:
    """Build non-anchor integer gauge directions from the largest time gaps."""

    session_count = int(profile.session_count)
    epochs = np.asarray(profile.session_epochs, dtype=float)
    if epochs.shape != (session_count,) or session_count < 2:
        raise ValueError("Session epochs must cover at least two sessions.")
    requested = min(max(int(block_count), 1), session_count)
    order = np.argsort(epochs, kind="stable")
    gaps = np.diff(epochs[order])
    cut_after = set(
        int(index)
        for index in np.argsort(gaps, kind="stable")[-(requested - 1) :]
    ) if requested > 1 else set()
    components: list[list[int]] = [[]]
    for position, session in enumerate(order.tolist()):
        components[-1].append(int(session))
        if position in cut_after and position < session_count - 1:
            components.append([])
    movable = [component for component in components if 0 not in component]
    basis = np.zeros((session_count - 1, len(movable)), dtype=float)
    for column, component in enumerate(movable):
        for session in component:
            if session > 0:
                basis[session - 1, column] = 1.0
    return basis


def temporal_phase_islands(
    profile: Any,
    *,
    island_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition sessions at the largest observing gaps.

    The partition is a temporary inference coordinate system.  Each island has
    its own integer phase gauge until the low-dimensional bridge posterior is
    recovered; no session is discarded or permanently merged.
    """

    session_count = int(profile.session_count)
    epochs = np.asarray(profile.session_epochs, dtype=float)
    if epochs.shape != (session_count,) or session_count < 2:
        raise ValueError("Session epochs must cover at least two sessions.")
    requested = min(max(int(island_count), 1), session_count)
    order = np.argsort(epochs, kind="stable")
    gaps = np.diff(epochs[order])
    cut_after = (
        {
            int(index)
            for index in np.argsort(gaps, kind="stable")[-(requested - 1) :]
        }
        if requested > 1
        else set()
    )
    islands: list[list[int]] = [[]]
    for position, session in enumerate(order.tolist()):
        islands[-1].append(int(session))
        if position in cut_after and position < session_count - 1:
            islands.append([])
    result = tuple(tuple(island) for island in islands if island)
    if not result or result[0][0] != 0:
        raise ValueError("The chronological phase-island anchor must be session zero.")
    return result


def island_relative_phase_geometry(
    profile: Any,
    islands: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Return phase geometry after marginalizing one integer gauge per island."""

    session_count = int(profile.session_count)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    normalized = tuple(tuple(int(session) for session in island) for island in islands)
    covered = [session for island in normalized for session in island]
    if sorted(covered) != list(range(session_count)) or any(not island for island in normalized):
        raise ValueError("Phase islands must cover every session exactly once.")

    relative_parts: list[np.ndarray] = []
    covariance_parts: list[np.ndarray] = []
    nonanchors: list[int] = []
    anchors: list[int] = []
    for island in normalized:
        anchor = int(island[0])
        members = np.asarray(island[1:], dtype=int)
        anchors.append(anchor)
        if members.size == 0:
            continue
        relative_parts.append(offsets[members] - offsets[anchor])
        covariance_parts.append(
            np.diag(variances[members]) + float(variances[anchor])
        )
        nonanchors.extend(int(session) for session in members)

    if not relative_parts:
        return (
            np.zeros(0, dtype=float),
            np.zeros((0, 0), dtype=float),
            tuple(nonanchors),
            tuple(anchors),
        )
    relative = np.concatenate(relative_parts)
    covariance = np.zeros((relative.size, relative.size), dtype=float)
    cursor = 0
    for part in covariance_parts:
        width = int(part.shape[0])
        covariance[cursor : cursor + width, cursor : cursor + width] = part
        cursor += width
    return relative, _stable_covariance(covariance), tuple(nonanchors), tuple(anchors)


def canonical_island_path_candidates(
    profile: Any,
    islands: Sequence[Sequence[int]],
    *,
    candidate_count: int,
    maximum_combinations: int,
    maximum_lattice_nodes: int = 2_000_000,
    phase_noise: float = 0.0,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate internal integer paths while fixing every island anchor to zero."""

    session_count = int(profile.session_count)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    combinations: list[tuple[np.ndarray, float]] = [
        (np.zeros(session_count, dtype=np.int64), 0.0)
    ]
    keep = max(int(maximum_combinations), 1)
    noise = max(float(phase_noise), 0.0)
    for island in islands:
        sessions = tuple(int(session) for session in island)
        if len(sessions) <= 1:
            continue
        anchor = sessions[0]
        members = np.asarray(sessions[1:], dtype=int)
        relative = offsets[members] - offsets[anchor]
        covariance = np.diag(variances[members]) + float(variances[anchor])
        if noise > 0.0:
            covariance = covariance + noise * noise * np.eye(members.size, dtype=float)
        try:
            lattice = solve_integer_least_squares(
                relative,
                _stable_covariance(covariance),
                candidate_count=max(int(candidate_count), 1),
                max_visited_nodes=max(int(maximum_lattice_nodes), 1),
            )
            local = [
                (
                    np.asarray(candidate.integer_vector, dtype=np.int64),
                    float(candidate.squared_mahalanobis),
                )
                for candidate in lattice.candidates
            ]
        except Exception:
            local = []
        if not local:
            local = [(np.rint(relative).astype(np.int64), math.inf)]
        expanded: list[tuple[np.ndarray, float]] = []
        for base, base_score in combinations:
            for local_path, local_score in local:
                path = base.copy()
                path[members] = local_path
                expanded.append((path, float(base_score + local_score)))
        combinations = sorted(expanded, key=lambda item: item[1])[:keep]
    return tuple(
        tuple(int(value) for value in path[1:])
        for path, _ in combinations
    )


def tempered_island_path_residual(
    profile: Any,
    *,
    islands: Sequence[Sequence[int]],
    integer_offsets: Sequence[int],
    phase_noise: float,
    normalized_values: Sequence[float] | None = None,
    prior_precision: float = 0.0,
) -> np.ndarray:
    """Evaluate a path after analytically removing one phase gauge per island."""

    shape = np.asarray(profile.normalized_residuals, dtype=float)
    relative, covariance, nonanchors, _ = island_relative_phase_geometry(
        profile,
        islands,
    )
    path = np.concatenate(
        (np.zeros(1, dtype=float), np.asarray(integer_offsets, dtype=float))
    )
    if path.shape != (int(profile.session_count),):
        raise ValueError("The island path must cover every non-anchor session.")
    integer_relative: list[float] = []
    for island in islands:
        anchor = int(island[0])
        integer_relative.extend(
            float(path[int(session)] - path[anchor]) for session in island[1:]
        )
    integers = np.asarray(integer_relative, dtype=float)
    if integers.shape != relative.shape or len(nonanchors) != relative.size:
        raise ValueError("The island integer path does not match its phase geometry.")
    noise = max(float(phase_noise), 0.0)
    if noise > 0.0 and covariance.size:
        covariance = covariance + noise * noise * np.eye(relative.size, dtype=float)
    parts = [shape]
    if relative.size:
        parts.append(
            np.linalg.solve(np.linalg.cholesky(covariance), relative - integers)
        )
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        if normalized_values is None:
            raise ValueError("Normalized values are required when the prior is active.")
        parts.append(math.sqrt(weight) * np.asarray(normalized_values, dtype=float))
    result = np.concatenate(parts)
    if not np.all(np.isfinite(result)):
        raise ValueError("The island-path residual contains non-finite values.")
    return result


def tempered_complete_path_residual(
    profile: Any,
    *,
    integer_offsets: Sequence[int],
    phase_noise: float,
    phase_beta: float = 1.0,
    normalized_values: Sequence[float] | None = None,
    prior_precision: float = 0.0,
) -> np.ndarray:
    """Evaluate one complete path under an annealed exact Gaussian target."""

    shape = np.asarray(profile.normalized_residuals, dtype=float)
    relative, covariance = relative_phase_geometry(profile)
    integers = np.asarray(integer_offsets, dtype=float)
    if integers.shape != relative.shape:
        raise ValueError("The integer path must cover every non-anchor session.")
    noise = max(float(phase_noise), 0.0)
    if noise > 0.0:
        covariance = covariance + noise * noise * np.eye(relative.size, dtype=float)
    beta = min(max(float(phase_beta), 0.0), 1.0)
    phase = math.sqrt(beta) * np.linalg.solve(
        np.linalg.cholesky(covariance),
        relative - integers,
    )
    parts = [shape, phase]
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        if normalized_values is None:
            raise ValueError("Normalized values are required when the prior is active.")
        parts.append(math.sqrt(weight) * np.asarray(normalized_values, dtype=float))
    result = np.concatenate(parts)
    if not np.all(np.isfinite(result)):
        raise ValueError("The complete-path residual contains non-finite values.")
    return result


def collapsed_continuous_objective(
    profile: Any,
    *,
    phase_noise: float,
    normalized_values: Sequence[float] | None = None,
    prior_precision: float = 0.0,
    hermite_order: int = 16,
    maximum_wrap_terms: int = 7,
) -> float:
    """Evaluate the continuous posterior after summing all integer paths.

    The local projected residuals contain the signed session slope and the
    remaining within-session shape exactly once.  Session intercept integers
    are marginalized by the wrapped Gaussian; no integer path is selected at
    this stage.
    """

    noise = max(float(phase_noise), 0.0)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    tempered_profile = replace(
        profile,
        session_offset_variances=variances + noise * noise,
    )
    wrapped_log_likelihood, _ = collapsed_wrapped_log_likelihood(
        tempered_profile,
        hermite_order=max(int(hermite_order), 4),
        maximum_wrap_terms=max(int(maximum_wrap_terms), 1),
    )
    local = np.asarray(profile.normalized_residuals, dtype=float)
    objective = float(local @ local - 2.0 * wrapped_log_likelihood)
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        if normalized_values is None:
            raise ValueError("Normalized values are required when the prior is active.")
        normalized = np.asarray(normalized_values, dtype=float)
        objective += float(weight * (normalized @ normalized))
    return objective


def collapsed_island_continuous_objective(
    profile: Any,
    *,
    islands: Sequence[Sequence[int]],
    phase_noise: float = 0.0,
    normalized_values: Sequence[float] | None = None,
    prior_precision: float = 0.0,
    hermite_order: int = 16,
    maximum_wrap_terms: int = 7,
) -> float:
    """Marginalize one wrapped integer lattice independently in every island."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    if offsets.shape != variances.shape:
        raise ValueError("Session offsets and variances must have matching shapes.")
    log_likelihood = 0.0
    noise = max(float(phase_noise), 0.0)
    for island in islands:
        indices = np.asarray(tuple(int(session) for session in island), dtype=int)
        if indices.size == 0:
            raise ValueError("Phase islands must be non-empty.")
        island_profile = replace(
            profile,
            session_offsets=offsets[indices],
            session_offset_variances=variances[indices] + noise * noise,
            session_count=int(indices.size),
            session_epochs=np.asarray(profile.session_epochs, dtype=float)[indices],
        )
        wrapped, _ = collapsed_wrapped_log_likelihood(
            island_profile,
            hermite_order=max(int(hermite_order), 4),
            maximum_wrap_terms=max(int(maximum_wrap_terms), 1),
        )
        log_likelihood += float(wrapped)
    objective = float(np.asarray(profile.normalized_residuals) @ np.asarray(profile.normalized_residuals))
    objective -= 2.0 * log_likelihood
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        if normalized_values is None:
            raise ValueError("Normalized values are required when the prior is active.")
        normalized = np.asarray(normalized_values, dtype=float)
        objective += float(weight * (normalized @ normalized))
    return objective


def _collapsed_mode_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[Any],
    phase_noise: float,
    temperature_index: int,
    starts_per_mode: int,
    path_family_limit: int,
    config: NonlinearPathTransportConfig,
    phase_beta: float = 1.0,
) -> tuple[CompletePathProposal, ...]:
    """Discover continuous modes before conditioning on an integer path."""

    lower, upper = _normalized_bounds(coordinates)
    requested = max(int(starts_per_mode), 0)
    if requested == 0:
        return ()
    proposals: list[CompletePathProposal] = []
    for mode_index, mode in enumerate(modes):
        center = np.clip(np.asarray(mode.normalized_values, dtype=float), lower, upper)
        covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
        starts = [center]
        if requested > 1:
            exponent = int(math.ceil(math.log2(requested - 1)))
            unit = qmc.Sobol(
                d=center.size,
                scramble=True,
                seed=int(config.random_seed)
                + 130363 * (temperature_index + 1)
                + 7919 * mode_index,
            ).random_base2(exponent)[: requested - 1]
            normal = ndtri(np.clip(unit, 1.0e-9, 1.0 - 1.0e-9))
            chol = np.linalg.cholesky(covariance)
            scales = tuple(float(value) for value in config.initial_scales if value > 0.0)
            if not scales:
                scales = (1.0,)
            starts.extend(
                np.clip(
                    center + scales[index % len(scales)] * (chol @ row),
                    lower,
                    upper,
                )
                for index, row in enumerate(normal)
            )

        for start_index, start in enumerate(starts):
            try:
                steps = _phase_resolving_steps(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    normalized_values=np.asarray(start, dtype=float),
                    config=config,
                )
            except Exception:
                continue

            def objective(values: np.ndarray) -> float:
                try:
                    profile = evaluator.evaluate_values(
                        _physical_values(coordinates, values)
                    )
                    value = collapsed_continuous_objective(
                        profile,
                        phase_noise=phase_noise,
                        normalized_values=values,
                        prior_precision=config.prior_precision,
                        hermite_order=config.collapsed_hermite_order,
                        maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
                    )
                    return value if np.isfinite(value) else 1.0e30
                except Exception:
                    return 1.0e30

            def gradient(values: np.ndarray) -> np.ndarray:
                center_values = np.asarray(values, dtype=float)
                result = np.zeros_like(center_values)
                for parameter_index, step in enumerate(steps):
                    plus = center_values.copy()
                    minus = center_values.copy()
                    plus[parameter_index] = min(
                        plus[parameter_index] + step,
                        upper[parameter_index],
                    )
                    minus[parameter_index] = max(
                        minus[parameter_index] - step,
                        lower[parameter_index],
                    )
                    denominator = float(
                        plus[parameter_index] - minus[parameter_index]
                    )
                    if denominator > 0.0:
                        result[parameter_index] = (
                            objective(plus) - objective(minus)
                        ) / denominator
                return result

            optimization = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                jac=gradient,
                bounds=tuple(zip(lower, upper, strict=True)),
                options={
                    "maxiter": max(int(config.collapsed_maxiter), 1),
                    "ftol": 1.0e-10,
                    "gtol": 1.0e-7,
                    "maxls": 24,
                },
            )
            normalized = np.clip(np.asarray(optimization.x, dtype=float), lower, upper)
            try:
                profile = evaluator.evaluate_values(
                    _physical_values(coordinates, normalized)
                )
                paths = complete_path_candidates(
                    profile,
                    candidate_count=config.point_path_candidates,
                    maximum_lattice_nodes=config.maximum_lattice_nodes,
                    phase_noise=phase_noise,
                    phase_beta=phase_beta,
                )
            except Exception:
                continue
            for path_index, path in enumerate(paths):
                proposal = CompletePathProposal(
                    path,
                    normalized,
                    (
                        f"collapsed:t{temperature_index:02d}:m{mode_index}:"
                        f"s{start_index}:path{path_index}"
                    ),
                    math.inf,
                )
                proposals.append(
                    replace(
                        proposal,
                        predictive_score=_score_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            proposal=proposal,
                            phase_noise=phase_noise,
                            prior_precision=config.prior_precision,
                            phase_beta=phase_beta,
                        ),
                    )
                )
    return _deduplicate_proposals(
        proposals,
        limit=max(int(path_family_limit), 1),
        maximum_starts_per_path=config.maximum_starts_per_path,
        mode_separation=config.proposal_mode_separation,
    )


def _phase_resolving_steps(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    config: NonlinearPathTransportConfig,
) -> np.ndarray:
    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(normalized_values, dtype=float)
    base_step = max(float(config.finite_difference_step), 1.0e-9)
    requested = max(float(config.phase_step_cycles), 1.0e-5)
    center_profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    center_relative, _ = relative_phase_geometry(center_profile)
    rates = np.zeros(center.size, dtype=float)
    for index in range(center.size):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + base_step, upper[index])
        minus[index] = max(minus[index] - base_step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            continue
        plus_profile = evaluator.evaluate_values(_physical_values(coordinates, plus))
        minus_profile = evaluator.evaluate_values(_physical_values(coordinates, minus))
        plus_relative, _ = relative_phase_geometry(plus_profile)
        minus_relative, _ = relative_phase_geometry(minus_profile)
        if plus_relative.shape != center_relative.shape or minus_relative.shape != center_relative.shape:
            raise ValueError("Finite differences changed the session dimension.")
        rates[index] = float(np.max(np.abs((plus_relative - minus_relative) / denominator)))
    steps = np.full(center.size, base_step, dtype=float)
    observed = rates > 0.0
    steps[observed] = np.minimum(steps[observed], requested / rates[observed])
    return np.maximum(steps, 1.0e-10)


def _relative_phase_jacobian(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    steps: np.ndarray,
) -> tuple[Any, np.ndarray]:
    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(normalized_values, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    relative, _ = relative_phase_geometry(profile)
    jacobian = np.empty((relative.size, center.size), dtype=float)
    for index, step in enumerate(np.asarray(steps, dtype=float)):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + step, upper[index])
        minus[index] = max(minus[index] - step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            jacobian[:, index] = 0.0
            continue
        plus_profile = evaluator.evaluate_values(_physical_values(coordinates, plus))
        minus_profile = evaluator.evaluate_values(_physical_values(coordinates, minus))
        plus_relative, _ = relative_phase_geometry(plus_profile)
        minus_relative, _ = relative_phase_geometry(minus_profile)
        jacobian[:, index] = (plus_relative - minus_relative) / denominator
    return profile, jacobian


def _fit_complete_path_once(
    *,
    evaluator: Any,
    coordinates: Any,
    proposal: CompletePathProposal,
    phase_noise: float,
    phase_beta: float,
    temperature_index: int,
    mode_index: int,
    config: NonlinearPathTransportConfig,
) -> NonlinearPathMode | None:
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(np.asarray(proposal.normalized_values, dtype=float), lower, upper)
    try:
        steps = _phase_resolving_steps(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=start,
            config=config,
        )
        start_profile = evaluator.evaluate_values(_physical_values(coordinates, start))
        failure = np.full(
            np.asarray(start_profile.normalized_residuals).size
            + len(proposal.integer_offsets)
            + (start.size if config.prior_precision > 0.0 else 0),
            1.0e12,
            dtype=float,
        )
    except Exception:
        return None

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            result = tempered_complete_path_residual(
                profile,
                integer_offsets=proposal.integer_offsets,
                phase_noise=phase_noise,
                phase_beta=phase_beta,
                normalized_values=values,
                prior_precision=config.prior_precision,
            )
            return result if result.shape == failure.shape else failure
        except Exception:
            return failure

    def jacobian(values: np.ndarray) -> np.ndarray:
        center = np.asarray(values, dtype=float)
        base = residual(center)
        result = np.empty((base.size, center.size), dtype=float)
        for index, step in enumerate(steps):
            plus = center.copy()
            minus = center.copy()
            plus[index] = min(plus[index] + step, upper[index])
            minus[index] = max(minus[index] - step, lower[index])
            denominator = float(plus[index] - minus[index])
            if denominator <= 0.0:
                result[:, index] = 0.0
            else:
                result[:, index] = (residual(plus) - residual(minus)) / denominator
        return result

    try:
        optimization = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max(int(config.fit_max_nfev), 1),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        normalized = np.asarray(optimization.x, dtype=float)
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        relative, covariance = relative_phase_geometry(profile)
        noise = max(float(phase_noise), 0.0)
        if noise > 0.0:
            covariance = covariance + noise * noise * np.eye(relative.size, dtype=float)
        delta = relative - np.asarray(proposal.integer_offsets, dtype=float)
        shape_chi2 = float(shape @ shape)
        phase_chi2 = float(delta @ np.linalg.solve(covariance, delta))
        prior_chi2 = float(max(config.prior_precision, 0.0) * (normalized @ normalized))
        objective = shape_chi2 + float(min(max(phase_beta, 0.0), 1.0)) * phase_chi2 + prior_chi2
        fit_jacobian = np.asarray(optimization.jac, dtype=float)
        covariance_fit = _stable_inverse(fit_jacobian.T @ fit_jacobian)
        sign, logdet = np.linalg.slogdet(covariance_fit)
        log_mass = float(-0.5 * objective + (0.5 * logdet if sign > 0.0 else -math.inf))
        return NonlinearPathMode(
            mode_id=f"transport_t{temperature_index:02d}_{mode_index:04d}",
            parent_mode_id=proposal.parent_mode_id,
            proposal_source=proposal.source,
            temperature_index=temperature_index,
            phase_noise=float(phase_noise),
            integer_offsets=proposal.integer_offsets,
            parameter_values=_physical_values(coordinates, normalized),
            normalized_values=normalized,
            covariance=covariance_fit,
            shape_chi2=shape_chi2,
            phase_chi2=phase_chi2,
            prior_chi2=prior_chi2,
            objective_chi2=objective,
            log_mass=log_mass,
            posterior_mass=0.0,
            nfev=int(optimization.nfev),
            path_update_count=0,
            success=bool(optimization.success),
            phase_beta=float(min(max(phase_beta, 0.0), 1.0)),
        )
    except Exception:
        return None


def _fit_complete_path(
    *,
    evaluator: Any,
    coordinates: Any,
    proposal: CompletePathProposal,
    phase_noise: float,
    phase_beta: float,
    temperature_index: int,
    mode_index: int,
    config: NonlinearPathTransportConfig,
) -> NonlinearPathMode | None:
    """Alternate exact continuous fits with complete integer-path reopening."""

    current = _fit_complete_path_once(
        evaluator=evaluator,
        coordinates=coordinates,
        proposal=proposal,
        phase_noise=phase_noise,
        phase_beta=phase_beta,
        temperature_index=temperature_index,
        mode_index=mode_index,
        config=config,
    )
    if current is None:
        return None
    total_nfev = int(current.nfev)
    update_count = 0
    visited = {current.integer_offsets}
    for round_index in range(max(int(config.joint_refinement_rounds), 0)):
        try:
            profile = evaluator.evaluate_values(
                np.asarray(current.parameter_values, dtype=float)
            )
            paths = complete_path_candidates(
                profile,
                candidate_count=max(int(config.joint_path_candidates), 1),
                maximum_lattice_nodes=config.maximum_lattice_nodes,
                phase_noise=phase_noise,
                phase_beta=phase_beta,
            )
        except Exception:
            break
        alternatives: list[NonlinearPathMode] = []
        for path_index, path in enumerate(paths):
            if path == current.integer_offsets or path in visited:
                continue
            candidate = CompletePathProposal(
                integer_offsets=path,
                normalized_values=np.asarray(current.normalized_values, dtype=float),
                source=(
                    f"joint-reopen:{current.mode_id}:r{round_index}:"
                    f"path{path_index}"
                ),
                predictive_score=math.inf,
                parent_mode_id=current.mode_id,
            )
            refitted = _fit_complete_path_once(
                evaluator=evaluator,
                coordinates=coordinates,
                proposal=candidate,
                phase_noise=phase_noise,
                phase_beta=phase_beta,
                temperature_index=temperature_index,
                mode_index=(mode_index * 100 + 10 * (round_index + 1) + path_index),
                config=config,
            )
            if refitted is not None:
                total_nfev += int(refitted.nfev)
                alternatives.append(refitted)
        if not alternatives:
            break
        best = min(alternatives, key=lambda mode: mode.objective_chi2)
        if best.objective_chi2 >= current.objective_chi2 - 1.0e-9:
            break
        current = best
        visited.add(current.integer_offsets)
        update_count += 1
    return replace(
        current,
        parent_mode_id=proposal.parent_mode_id,
        nfev=total_nfev,
        path_update_count=update_count,
    )


def _collapsed_island_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    islands: Sequence[Sequence[int]],
    invariant_modes: Sequence[Any],
    config: NonlinearPathTransportConfig,
) -> tuple[NonlinearPathMode, ...]:
    """Find continuous island modes before selecting any internal integer path."""

    lower, upper = _normalized_bounds(coordinates)
    ranked_parents = sorted(
        invariant_modes,
        key=lambda mode: float(getattr(mode, "objective_chi2", math.inf)),
    )[: max(int(config.island_parent_modes), 1)]
    ladder = tuple(
        sorted(
            {
                float(value)
                for value in config.island_collapsed_noise_ladder
                if np.isfinite(float(value)) and float(value) >= 0.0
            },
            reverse=True,
        )
    ) or (0.0,)
    requested = max(int(config.island_collapsed_starts_per_mode), 1)
    modes: list[NonlinearPathMode] = []
    for parent_index, parent in enumerate(ranked_parents):
        center = np.clip(np.asarray(parent.normalized_values, dtype=float), lower, upper)
        parent_covariance = _stable_covariance(np.asarray(parent.covariance, dtype=float))
        starts = [center]
        if requested > 1:
            exponent = int(math.ceil(math.log2(requested - 1)))
            unit = qmc.Sobol(
                d=center.size,
                scramble=True,
                seed=int(config.random_seed) + 32452843 * (parent_index + 1),
            ).random_base2(exponent)[: requested - 1]
            normal = ndtri(np.clip(unit, 1.0e-9, 1.0 - 1.0e-9))
            chol = np.linalg.cholesky(parent_covariance)
            scales = tuple(float(value) for value in config.initial_scales if value > 0.0) or (1.0,)
            starts.extend(
                np.clip(
                    center + scales[index % len(scales)] * (chol @ row),
                    lower,
                    upper,
                )
                for index, row in enumerate(normal)
            )

        for start_index, start in enumerate(starts):
            state = np.asarray(start, dtype=float)
            total_nfev = 0
            final_optimization: Any | None = None
            for phase_noise in ladder:
                try:
                    steps = _phase_resolving_steps(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        normalized_values=state,
                        config=config,
                    )
                except Exception:
                    final_optimization = None
                    break

                def objective(values: np.ndarray) -> float:
                    try:
                        profile = evaluator.evaluate_values(
                            _physical_values(coordinates, values)
                        )
                        value = collapsed_island_continuous_objective(
                            profile,
                            islands=islands,
                            phase_noise=phase_noise,
                            normalized_values=values,
                            prior_precision=config.prior_precision,
                            hermite_order=config.collapsed_hermite_order,
                            maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
                        )
                        return value if np.isfinite(value) else 1.0e30
                    except Exception:
                        return 1.0e30

                def gradient(values: np.ndarray) -> np.ndarray:
                    values = np.asarray(values, dtype=float)
                    result = np.zeros_like(values)
                    for parameter_index, step in enumerate(steps):
                        plus = values.copy()
                        minus = values.copy()
                        plus[parameter_index] = min(
                            plus[parameter_index] + step,
                            upper[parameter_index],
                        )
                        minus[parameter_index] = max(
                            minus[parameter_index] - step,
                            lower[parameter_index],
                        )
                        denominator = float(
                            plus[parameter_index] - minus[parameter_index]
                        )
                        if denominator > 0.0:
                            result[parameter_index] = (
                                objective(plus) - objective(minus)
                            ) / denominator
                    return result

                final_optimization = minimize(
                    objective,
                    state,
                    method="L-BFGS-B",
                    jac=gradient,
                    bounds=tuple(zip(lower, upper, strict=True)),
                    options={
                        "maxiter": max(int(config.island_collapsed_maxiter), 1),
                        "ftol": 1.0e-10,
                        "gtol": 1.0e-7,
                        "maxls": 24,
                    },
                )
                total_nfev += int(final_optimization.nfev)
                state = np.clip(
                    np.asarray(final_optimization.x, dtype=float),
                    lower,
                    upper,
                )
            if final_optimization is None:
                continue
            try:
                profile = evaluator.evaluate_values(_physical_values(coordinates, state))
                paths = canonical_island_path_candidates(
                    profile,
                    islands,
                    candidate_count=max(int(config.island_path_candidates), 1),
                    maximum_combinations=max(int(config.island_mode_beam), 1),
                    maximum_lattice_nodes=config.maximum_lattice_nodes,
                    phase_noise=config.island_phase_noise,
                )
                inverse_hessian = final_optimization.hess_inv
                covariance = _stable_covariance(
                    np.asarray(
                        inverse_hessian.todense()
                        if hasattr(inverse_hessian, "todense")
                        else inverse_hessian,
                        dtype=float,
                    )
                )
            except Exception:
                continue
            for path_index, path in enumerate(paths):
                try:
                    residual = tempered_island_path_residual(
                        profile,
                        islands=islands,
                        integer_offsets=path,
                        phase_noise=config.island_phase_noise,
                        normalized_values=state,
                        prior_precision=config.prior_precision,
                    )
                except Exception:
                    continue
                shape = np.asarray(profile.normalized_residuals, dtype=float)
                shape_chi2 = float(shape @ shape)
                prior_chi2 = float(
                    max(config.prior_precision, 0.0) * (state @ state)
                )
                objective_chi2 = float(residual @ residual)
                phase_chi2 = max(objective_chi2 - shape_chi2 - prior_chi2, 0.0)
                sign, logdet = np.linalg.slogdet(covariance)
                log_mass = float(
                    -0.5 * objective_chi2
                    + (0.5 * logdet if sign > 0.0 else -math.inf)
                )
                modes.append(
                    NonlinearPathMode(
                        mode_id=(
                            f"collapsed_island_{len(islands):02d}_"
                            f"{parent_index:02d}_{start_index:02d}_{path_index:02d}"
                        ),
                        parent_mode_id=getattr(parent, "mode_id", None),
                        proposal_source=(
                            f"collapsed-island:{len(islands)}:m{parent_index}:"
                            f"s{start_index}:p{path_index}"
                        ),
                        temperature_index=-1,
                        phase_noise=float(config.island_phase_noise),
                        integer_offsets=path,
                        parameter_values=_physical_values(coordinates, state),
                        normalized_values=state,
                        covariance=covariance,
                        shape_chi2=shape_chi2,
                        phase_chi2=phase_chi2,
                        prior_chi2=prior_chi2,
                        objective_chi2=objective_chi2,
                        log_mass=log_mass,
                        posterior_mass=0.0,
                        nfev=total_nfev,
                        path_update_count=0,
                        success=bool(final_optimization.success),
                    )
                )
    collapsed_config = replace(
        config,
        path_beam=max(int(config.island_mode_beam), 1),
        maximum_modes_per_path=1,
    )
    return _select_modes(modes, config=collapsed_config)


def _island_relative_phase_jacobian(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    islands: Sequence[Sequence[int]],
    steps: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(normalized_values, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    relative, covariance, nonanchors, _ = island_relative_phase_geometry(profile, islands)
    jacobian = np.empty((relative.size, center.size), dtype=float)
    for index, step in enumerate(np.asarray(steps, dtype=float)):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + step, upper[index])
        minus[index] = max(minus[index] - step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            jacobian[:, index] = 0.0
            continue
        plus_profile = evaluator.evaluate_values(_physical_values(coordinates, plus))
        minus_profile = evaluator.evaluate_values(_physical_values(coordinates, minus))
        plus_relative, _, _, _ = island_relative_phase_geometry(plus_profile, islands)
        minus_relative, _, _, _ = island_relative_phase_geometry(minus_profile, islands)
        jacobian[:, index] = (plus_relative - minus_relative) / denominator
    return profile, relative, covariance, jacobian, nonanchors


def _predictive_island_path_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    islands: Sequence[Sequence[int]],
    invariant_modes: Sequence[Any],
    phase_noise: float,
    config: NonlinearPathTransportConfig,
) -> tuple[CompletePathProposal, ...]:
    """Generate island paths from the correlated continuous posterior, not rounding."""

    lower, upper = _normalized_bounds(coordinates)
    session_count = sum(len(island) for island in islands)
    proposals: list[CompletePathProposal] = []
    ranked = sorted(
        invariant_modes,
        key=lambda mode: float(getattr(mode, "objective_chi2", math.inf)),
    )[: max(int(config.island_parent_modes), 1)]
    for parent_index, parent in enumerate(ranked):
        center = np.asarray(parent.normalized_values, dtype=float)
        continuous_covariance_base = _stable_covariance(
            np.asarray(parent.covariance, dtype=float)
        )
        try:
            steps = _phase_resolving_steps(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=center,
                config=config,
            )
            profile, phase_center, phase_covariance, phase_jacobian, nonanchors = (
                _island_relative_phase_jacobian(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    normalized_values=center,
                    islands=islands,
                    steps=steps,
                )
            )
        except Exception:
            continue
        if phase_noise > 0.0 and phase_covariance.size:
            phase_covariance = phase_covariance + phase_noise * phase_noise * np.eye(
                phase_covariance.shape[0],
                dtype=float,
            )
        for raw_kappa in config.lattice_kappas:
            kappa = max(float(raw_kappa), 1.0e-6)
            continuous_covariance = (kappa * kappa) * continuous_covariance_base
            predictive_covariance = _stable_covariance(
                phase_covariance
                + phase_jacobian @ continuous_covariance @ phase_jacobian.T
            )
            try:
                lattice = solve_integer_least_squares(
                    phase_center,
                    predictive_covariance,
                    candidate_count=max(
                        int(config.island_lattice_candidates_per_scale),
                        1,
                    ),
                    max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
                )
                gain = (
                    continuous_covariance
                    @ phase_jacobian.T
                    @ _stable_inverse(predictive_covariance)
                )
            except Exception:
                continue
            for candidate_index, candidate in enumerate(lattice.candidates):
                internal = np.asarray(candidate.integer_vector, dtype=np.int64)
                path_array = np.zeros(session_count, dtype=np.int64)
                path_array[np.asarray(nonanchors, dtype=int)] = internal
                displacement = internal.astype(float) - phase_center
                state = np.clip(center + gain @ displacement, lower, upper)
                proposal = CompletePathProposal(
                    integer_offsets=tuple(int(value) for value in path_array[1:]),
                    normalized_values=state,
                    source=(
                        f"island-predictive:{len(islands)}:m{parent_index}:"
                        f"k{kappa:g}:c{candidate_index}"
                    ),
                    predictive_score=math.inf,
                    parent_mode_id=getattr(parent, "mode_id", None),
                )
                try:
                    state_profile = evaluator.evaluate_values(
                        _physical_values(coordinates, state)
                    )
                    residual = tempered_island_path_residual(
                        state_profile,
                        islands=islands,
                        integer_offsets=proposal.integer_offsets,
                        phase_noise=phase_noise,
                        normalized_values=state,
                        prior_precision=config.prior_precision,
                    )
                    score = float(residual @ residual)
                except Exception:
                    score = math.inf
                proposals.append(replace(proposal, predictive_score=score))
    return _deduplicate_proposals(
        proposals,
        limit=max(int(config.maximum_island_lattice_proposals), 1),
        maximum_starts_per_path=config.maximum_starts_per_path,
        mode_separation=config.proposal_mode_separation,
    )


def _fit_island_path_once(
    *,
    evaluator: Any,
    coordinates: Any,
    islands: Sequence[Sequence[int]],
    proposal: CompletePathProposal,
    phase_noise: float,
    mode_index: int,
    config: NonlinearPathTransportConfig,
) -> NonlinearPathMode | None:
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(np.asarray(proposal.normalized_values, dtype=float), lower, upper)
    try:
        steps = _phase_resolving_steps(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=start,
            config=config,
        )
        start_profile = evaluator.evaluate_values(_physical_values(coordinates, start))
        relative, _, _, _ = island_relative_phase_geometry(start_profile, islands)
        failure = np.full(
            np.asarray(start_profile.normalized_residuals).size
            + relative.size
            + (start.size if config.prior_precision > 0.0 else 0),
            1.0e12,
            dtype=float,
        )
    except Exception:
        return None

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            result = tempered_island_path_residual(
                profile,
                islands=islands,
                integer_offsets=proposal.integer_offsets,
                phase_noise=phase_noise,
                normalized_values=values,
                prior_precision=config.prior_precision,
            )
            return result if result.shape == failure.shape else failure
        except Exception:
            return failure

    def jacobian(values: np.ndarray) -> np.ndarray:
        center = np.asarray(values, dtype=float)
        base = residual(center)
        result = np.empty((base.size, center.size), dtype=float)
        for index, step in enumerate(steps):
            plus = center.copy()
            minus = center.copy()
            plus[index] = min(plus[index] + step, upper[index])
            minus[index] = max(minus[index] - step, lower[index])
            denominator = float(plus[index] - minus[index])
            result[:, index] = (
                0.0
                if denominator <= 0.0
                else (residual(plus) - residual(minus)) / denominator
            )
        return result

    try:
        optimization = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max(int(config.fit_max_nfev), 1),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        normalized = np.asarray(optimization.x, dtype=float)
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        relative, covariance, _, _ = island_relative_phase_geometry(profile, islands)
        path = np.concatenate(
            (np.zeros(1, dtype=float), np.asarray(proposal.integer_offsets, dtype=float))
        )
        integer_relative = np.asarray(
            [
                path[int(session)] - path[int(island[0])]
                for island in islands
                for session in island[1:]
            ],
            dtype=float,
        )
        noise = max(float(phase_noise), 0.0)
        if noise > 0.0 and covariance.size:
            covariance = covariance + noise * noise * np.eye(relative.size, dtype=float)
        delta = relative - integer_relative
        shape_chi2 = float(shape @ shape)
        phase_chi2 = (
            float(delta @ np.linalg.solve(covariance, delta))
            if delta.size
            else 0.0
        )
        prior_chi2 = float(max(config.prior_precision, 0.0) * (normalized @ normalized))
        objective = shape_chi2 + phase_chi2 + prior_chi2
        fit_jacobian = np.asarray(optimization.jac, dtype=float)
        covariance_fit = _stable_inverse(fit_jacobian.T @ fit_jacobian)
        sign, logdet = np.linalg.slogdet(covariance_fit)
        log_mass = float(-0.5 * objective + (0.5 * logdet if sign > 0.0 else -math.inf))
        return NonlinearPathMode(
            mode_id=f"island_{len(islands):02d}_{mode_index:04d}",
            parent_mode_id=proposal.parent_mode_id,
            proposal_source=proposal.source,
            temperature_index=-1,
            phase_noise=float(phase_noise),
            integer_offsets=proposal.integer_offsets,
            parameter_values=_physical_values(coordinates, normalized),
            normalized_values=normalized,
            covariance=covariance_fit,
            shape_chi2=shape_chi2,
            phase_chi2=phase_chi2,
            prior_chi2=prior_chi2,
            objective_chi2=objective,
            log_mass=log_mass,
            posterior_mass=0.0,
            nfev=int(optimization.nfev),
            path_update_count=0,
            success=bool(optimization.success),
        )
    except Exception:
        return None


def _fit_island_path(
    *,
    evaluator: Any,
    coordinates: Any,
    islands: Sequence[Sequence[int]],
    proposal: CompletePathProposal,
    mode_index: int,
    config: NonlinearPathTransportConfig,
) -> NonlinearPathMode | None:
    """Jointly refine continuous parameters and all within-island integers."""

    phase_noise = max(float(config.island_phase_noise), 0.0)
    current = _fit_island_path_once(
        evaluator=evaluator,
        coordinates=coordinates,
        islands=islands,
        proposal=proposal,
        phase_noise=phase_noise,
        mode_index=mode_index,
        config=config,
    )
    if current is None:
        return None
    total_nfev = int(current.nfev)
    update_count = 0
    visited = {current.integer_offsets}
    for round_index in range(max(int(config.island_joint_refinement_rounds), 0)):
        try:
            profile = evaluator.evaluate_values(np.asarray(current.parameter_values, dtype=float))
            paths = canonical_island_path_candidates(
                profile,
                islands,
                candidate_count=max(int(config.island_path_candidates), 1),
                maximum_combinations=max(int(config.island_path_candidates), 1),
                maximum_lattice_nodes=config.maximum_lattice_nodes,
                phase_noise=phase_noise,
            )
        except Exception:
            break
        alternatives: list[NonlinearPathMode] = []
        for path_index, path in enumerate(paths):
            if path == current.integer_offsets or path in visited:
                continue
            candidate = CompletePathProposal(
                integer_offsets=path,
                normalized_values=np.asarray(current.normalized_values, dtype=float),
                source=f"island-reopen:{current.mode_id}:r{round_index}:p{path_index}",
                predictive_score=math.inf,
                parent_mode_id=current.mode_id,
            )
            refitted = _fit_island_path_once(
                evaluator=evaluator,
                coordinates=coordinates,
                islands=islands,
                proposal=candidate,
                phase_noise=phase_noise,
                mode_index=mode_index * 100 + 10 * (round_index + 1) + path_index,
                config=config,
            )
            if refitted is not None:
                total_nfev += int(refitted.nfev)
                alternatives.append(refitted)
        if not alternatives:
            break
        best = min(alternatives, key=lambda mode: mode.objective_chi2)
        if best.objective_chi2 >= current.objective_chi2 - 1.0e-9:
            break
        current = best
        visited.add(current.integer_offsets)
        update_count += 1
    return replace(current, nfev=total_nfev, path_update_count=update_count)


def _initial_island_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    islands: Sequence[Sequence[int]],
    invariant_modes: Sequence[Any],
    config: NonlinearPathTransportConfig,
) -> tuple[NonlinearPathMode, ...]:
    predictive = _predictive_island_path_proposals(
        evaluator=evaluator,
        coordinates=coordinates,
        islands=islands,
        invariant_modes=invariant_modes,
        phase_noise=config.island_phase_noise,
        config=config,
    )
    fitted: list[NonlinearPathMode] = []
    for proposal_index, proposal in enumerate(
        predictive[: max(int(config.island_mode_beam), 1)]
    ):
        mode = _fit_island_path(
            evaluator=evaluator,
            coordinates=coordinates,
            islands=islands,
            proposal=proposal,
            mode_index=proposal_index,
            config=config,
        )
        if mode is not None:
            fitted.append(mode)
    if not fitted:
        collapsed_modes = _collapsed_island_modes(
            evaluator=evaluator,
            coordinates=coordinates,
            islands=islands,
            invariant_modes=invariant_modes,
            config=config,
        )
        if collapsed_modes:
            return collapsed_modes
    island_config = replace(
        config,
        path_beam=max(int(config.island_mode_beam), 1),
        maximum_modes_per_path=1,
    )
    return _select_modes(fitted, config=island_config)


def _recover_island_bridge_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    islands: Sequence[Sequence[int]],
    modes: Sequence[NonlinearPathMode],
    phase_noise: float,
    config: NonlinearPathTransportConfig,
) -> tuple[CompletePathProposal, ...]:
    """Recover the low-dimensional integer gauges connecting fitted islands."""

    lower, upper = _normalized_bounds(coordinates)
    session_count = sum(len(island) for island in islands)
    basis = np.zeros((session_count - 1, max(len(islands) - 1, 0)), dtype=float)
    for column, island in enumerate(islands[1:]):
        for session in island:
            if int(session) > 0:
                basis[int(session) - 1, column] = 1.0
    proposals: list[CompletePathProposal] = []
    for mode in modes:
        center = np.asarray(mode.normalized_values, dtype=float)
        canonical = np.asarray(mode.integer_offsets, dtype=float)
        try:
            steps = _phase_resolving_steps(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=center,
                config=config,
            )
            profile, phase_jacobian = _relative_phase_jacobian(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=center,
                steps=steps,
            )
            phase_center, phase_covariance = relative_phase_geometry(profile)
        except Exception:
            continue
        noise = max(float(phase_noise), 0.0)
        if noise > 0.0:
            phase_covariance = phase_covariance + noise * noise * np.eye(
                phase_center.size,
                dtype=float,
            )
        if basis.shape[1] == 0:
            gauge_candidates = [(np.zeros(0, dtype=np.int64), 0.0, 1.0)]
        else:
            gauge_candidates: list[tuple[np.ndarray, float, float]] = []
            for raw_kappa in config.lattice_kappas:
                kappa = max(float(raw_kappa), 1.0e-6)
                continuous_covariance = (
                    kappa * kappa * _stable_covariance(np.asarray(mode.covariance, dtype=float))
                )
                predictive_covariance = _stable_covariance(
                    phase_covariance
                    + phase_jacobian @ continuous_covariance @ phase_jacobian.T
                )
                predictive_precision = _stable_inverse(predictive_covariance)
                gauge_precision = _stable_covariance(
                    basis.T @ predictive_precision @ basis
                )
                gauge_covariance = _stable_inverse(gauge_precision)
                gauge_center = (
                    gauge_covariance
                    @ basis.T
                    @ predictive_precision
                    @ (phase_center - canonical)
                )
                try:
                    lattice = solve_integer_least_squares(
                        gauge_center,
                        gauge_covariance,
                        candidate_count=max(
                            int(config.bridge_gauge_candidates_per_scale),
                            1,
                        ),
                        max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
                    )
                except Exception:
                    continue
                gauge_candidates.extend(
                    (
                        np.asarray(candidate.integer_vector, dtype=np.int64),
                        float(candidate.squared_mahalanobis),
                        kappa,
                    )
                    for candidate in lattice.candidates
                )
        for candidate_index, (gauge, _, kappa) in enumerate(gauge_candidates):
            continuous_covariance = (
                kappa * kappa * _stable_covariance(np.asarray(mode.covariance, dtype=float))
            )
            predictive_covariance = _stable_covariance(
                phase_covariance
                + phase_jacobian @ continuous_covariance @ phase_jacobian.T
            )
            gain = (
                continuous_covariance
                @ phase_jacobian.T
                @ _stable_inverse(predictive_covariance)
            )
            path_array = np.rint(canonical + basis @ gauge).astype(np.int64)
            displacement = path_array.astype(float) - phase_center
            state = np.clip(center + gain @ displacement, lower, upper)
            proposal = CompletePathProposal(
                integer_offsets=tuple(int(value) for value in path_array),
                normalized_values=state,
                source=(
                    f"island-bridge:{len(islands)}:{mode.mode_id}:"
                    f"k{kappa:g}:g{candidate_index}"
                ),
                predictive_score=math.inf,
                parent_mode_id=mode.mode_id,
            )
            proposals.append(
                replace(
                    proposal,
                    predictive_score=_score_proposal(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        proposal=proposal,
                        phase_noise=phase_noise,
                        prior_precision=config.prior_precision,
                    ),
                )
            )
    return _deduplicate_proposals(
        proposals,
        limit=max(int(config.maximum_island_bridge_proposals), 1),
        maximum_starts_per_path=config.maximum_starts_per_path,
        mode_separation=config.proposal_mode_separation,
    )


def _discover_island_bridge_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    phase_noise: float,
    config: NonlinearPathTransportConfig,
) -> tuple[tuple[CompletePathProposal, ...], int]:
    profile = evaluator.evaluate_values(np.asarray(invariant_modes[0].parameter_values, dtype=float))
    proposals: list[CompletePathProposal] = []
    mode_count = 0
    for requested in config.island_bridge_counts:
        try:
            islands = temporal_phase_islands(profile, island_count=int(requested))
        except Exception:
            continue
        if len(islands) <= 1:
            continue
        modes = _initial_island_modes(
            evaluator=evaluator,
            coordinates=coordinates,
            islands=islands,
            invariant_modes=invariant_modes,
            config=config,
        )
        mode_count += len(modes)
        proposals.extend(
            _recover_island_bridge_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                islands=islands,
                modes=modes,
                phase_noise=phase_noise,
                config=config,
            )
        )
    return (
        _deduplicate_proposals(
            proposals,
            limit=max(int(config.maximum_island_bridge_proposals), 1),
            maximum_starts_per_path=config.maximum_starts_per_path,
            mode_separation=config.proposal_mode_separation,
        ),
        mode_count,
    )


def _score_proposal(
    *,
    evaluator: Any,
    coordinates: Any,
    proposal: CompletePathProposal,
    phase_noise: float,
    prior_precision: float,
    phase_beta: float = 1.0,
) -> float:
    try:
        profile = evaluator.evaluate_values(
            _physical_values(coordinates, proposal.normalized_values)
        )
        residual = tempered_complete_path_residual(
            profile,
            integer_offsets=proposal.integer_offsets,
            phase_noise=phase_noise,
            phase_beta=phase_beta,
            normalized_values=proposal.normalized_values,
            prior_precision=prior_precision,
        )
        return float(residual @ residual)
    except Exception:
        return math.inf


def _deduplicate_proposals(
    proposals: Sequence[CompletePathProposal],
    *,
    limit: int,
    maximum_starts_per_path: int,
    mode_separation: float,
) -> tuple[CompletePathProposal, ...]:
    grouped: dict[tuple[int, ...], list[CompletePathProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.integer_offsets, []).append(proposal)

    selected: list[CompletePathProposal] = []
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: min(item.predictive_score for item in group),
    )
    for group in ordered_groups[: max(int(limit), 1)]:
        group_selected: list[CompletePathProposal] = []
        for proposal in sorted(group, key=lambda item: item.predictive_score):
            if any(
                np.linalg.norm(
                    np.asarray(proposal.normalized_values, dtype=float)
                    - np.asarray(other.normalized_values, dtype=float)
                )
                < max(float(mode_separation), 0.0)
                for other in group_selected
            ):
                continue
            group_selected.append(proposal)
            if len(group_selected) >= max(int(maximum_starts_per_path), 1):
                break
        selected.extend(group_selected)
    return tuple(selected)


def _select_modes(
    modes: Sequence[NonlinearPathMode],
    *,
    config: NonlinearPathTransportConfig,
) -> tuple[NonlinearPathMode, ...]:
    grouped: dict[tuple[int, ...], list[NonlinearPathMode]] = {}
    for mode in modes:
        grouped.setdefault(mode.integer_offsets, []).append(mode)
    if not grouped:
        return ()

    path_log_masses = {
        path: float(logsumexp([mode.log_mass for mode in group]))
        for path, group in grouped.items()
    }
    evidence_paths = sorted(
        grouped,
        key=lambda path: (
            -path_log_masses[path],
            min(mode.objective_chi2 for mode in grouped[path]),
        ),
    )
    map_path = min(
        grouped,
        key=lambda path: min(mode.objective_chi2 for mode in grouped[path]),
    )
    selected_paths = evidence_paths[: max(int(config.path_beam), 1)]
    if map_path not in selected_paths:
        selected_paths[-1] = map_path

    selected: list[NonlinearPathMode] = []
    for path in selected_paths:
        path_modes: list[NonlinearPathMode] = []
        evidence_ranked = sorted(
            grouped[path],
            key=lambda item: (-item.log_mass, item.objective_chi2),
        )
        map_mode = min(grouped[path], key=lambda item: item.objective_chi2)
        candidates = [map_mode, *evidence_ranked]
        for mode in candidates:
            if any(
                np.linalg.norm(other.normalized_values - mode.normalized_values)
                < max(float(config.mode_separation), 0.0)
                for other in path_modes
            ):
                continue
            path_modes.append(mode)
            if len(path_modes) >= max(int(config.maximum_modes_per_path), 1):
                break
        selected.extend(path_modes)
    if not selected:
        return ()
    return _with_posterior_masses(selected)


def _with_posterior_masses(
    modes: Sequence[NonlinearPathMode],
) -> tuple[NonlinearPathMode, ...]:
    selected = list(modes)
    if not selected:
        return ()
    finite = np.asarray([mode.log_mass for mode in selected], dtype=float)
    if not np.any(np.isfinite(finite)):
        weights = np.full(len(selected), 1.0 / len(selected), dtype=float)
    else:
        maximum = float(np.max(finite[np.isfinite(finite)]))
        weights = np.exp(np.clip(finite - maximum, -745.0, 0.0))
        weights[~np.isfinite(weights)] = 0.0
        total = float(np.sum(weights))
        weights = (
            weights / total
            if total > 0.0
            else np.full(len(selected), 1.0 / len(selected), dtype=float)
        )
    return tuple(
        replace(mode, posterior_mass=float(weight))
        for mode, weight in zip(selected, weights, strict=True)
    )


def effective_sample_size(log_weights: Sequence[float]) -> float:
    """Return ESS for unnormalized log weights."""

    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if values.ndim != 1 or values.size == 0 or not np.any(finite):
        return 0.0
    shifted = np.where(finite, values - np.max(values[finite]), -math.inf)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    squared = float(weights @ weights)
    return total * total / squared if total > 0.0 and squared > 0.0 else 0.0


def select_adaptive_phase_beta(
    *,
    current_beta: float,
    exact_phase_chi2: Sequence[float],
    ess_target_fraction: float,
    minimum_step: float,
    maximum_step: float,
    maximum_growth_factor: float,
) -> float:
    """Choose the next exact-phase temperature from conditional ESS.

    The mode bank is treated as a deterministic mixture, so the conditional
    ESS is computed from equal-weight ancestors. Posterior mass is used for
    ranking after rejuvenation, not to erase a mode before overlap is tested.
    """

    beta = min(max(float(current_beta), 0.0), 1.0)
    if beta >= 1.0:
        return 1.0
    chi2 = np.asarray(exact_phase_chi2, dtype=float)
    if chi2.ndim != 1 or chi2.size == 0 or not np.all(np.isfinite(chi2)):
        raise ValueError("Exact phase chi-square values must be finite and non-empty.")
    floor = min(max(float(minimum_step), np.finfo(float).eps), 1.0 - beta)
    upper = min(beta + max(float(maximum_step), floor), 1.0)
    if beta > 0.0:
        upper = min(
            upper,
            max(beta + floor, beta * max(float(maximum_growth_factor), 1.0)),
        )
    target = min(max(float(ess_target_fraction), 0.0), 1.0) * chi2.size

    def conditional_ess(candidate: float) -> float:
        return effective_sample_size(-0.5 * (float(candidate) - beta) * chi2)

    if conditional_ess(upper) >= target:
        return float(upper)
    lower = beta + floor
    if conditional_ess(lower) <= target:
        return float(lower)
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if conditional_ess(midpoint) < target:
            upper = midpoint
        else:
            lower = midpoint
    return float(lower)


def _exact_phase_chi2_for_state(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    integer_offsets: Sequence[int],
) -> float:
    profile = evaluator.evaluate_values(_physical_values(coordinates, normalized_values))
    relative, covariance = relative_phase_geometry(profile)
    delta = relative - np.asarray(integer_offsets, dtype=float)
    return float(delta @ np.linalg.solve(covariance, delta))


def _mode_phase_overlap_chi2(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: NonlinearPathMode,
    config: NonlinearPathTransportConfig,
    seed: int,
) -> np.ndarray:
    """Draw local continuous states for conditional phase-overlap ESS."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.clip(np.asarray(mode.normalized_values, dtype=float), lower, upper)
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    requested = max(int(config.phase_beta_overlap_samples), 1)
    if requested == 1:
        states = [center]
    else:
        exponent = int(math.ceil(math.log2(requested - 1)))
        unit = qmc.Sobol(
            d=center.size,
            scramble=True,
            seed=int(seed),
        ).random_base2(exponent)[: requested - 1]
        normal = ndtri(np.clip(unit, 1.0e-9, 1.0 - 1.0e-9))
        try:
            chol = np.linalg.cholesky(covariance)
            states = [
                np.clip(center + chol @ row, lower, upper) for row in normal
            ]
        except np.linalg.LinAlgError:
            states = [center]
    values: list[float] = []
    for state in states:
        try:
            values.append(
                _exact_phase_chi2_for_state(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    normalized_values=np.asarray(state, dtype=float),
                    integer_offsets=mode.integer_offsets,
                )
            )
        except Exception:
            continue
    if not values:
        return np.asarray([float(mode.phase_chi2)], dtype=float)
    return np.asarray(values, dtype=float)


def _select_mode_aware_phase_beta(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[NonlinearPathMode],
    current_beta: float,
    config: NonlinearPathTransportConfig,
    temperature_index: int,
) -> tuple[float, float]:
    """Use within-mode overlap ESS while allowing between-mode evidence change."""

    candidates: list[float] = []
    conditional_ess: list[float] = []
    for mode_index, mode in enumerate(modes):
        samples = _mode_phase_overlap_chi2(
            evaluator=evaluator,
            coordinates=coordinates,
            mode=mode,
            config=config,
            seed=(int(config.random_seed) + 15485863 * (temperature_index + 1) + mode_index),
        )
        candidate = select_adaptive_phase_beta(
            current_beta=current_beta,
            exact_phase_chi2=samples,
            ess_target_fraction=config.phase_beta_ess_target_fraction,
            minimum_step=config.phase_beta_minimum_step,
            maximum_step=config.phase_beta_maximum_step,
            maximum_growth_factor=config.phase_beta_maximum_growth_factor,
        )
        candidates.append(candidate)
        conditional_ess.append(
            effective_sample_size(-0.5 * (candidate - current_beta) * samples)
        )
    if not candidates:
        return 1.0, 0.0
    return float(min(candidates)), float(min(conditional_ess, default=0.0))


def _select_ancestry_proposals(
    proposals: Sequence[CompletePathProposal],
    *,
    parent_modes: Sequence[NonlinearPathMode],
    config: NonlinearPathTransportConfig,
) -> tuple[CompletePathProposal, ...]:
    """Retain transport support per posterior component before global ranking."""

    target = max(
        int(config.prefit_path_families),
        len(parent_modes) * max(int(config.protected_ancestor_modes), 1),
    )
    selected: list[CompletePathProposal] = []
    selected_ids: set[int] = set()
    for parent in parent_modes:
        children = [item for item in proposals if item.parent_mode_id == parent.mode_id]
        children.sort(
            key=lambda item: (
                not item.source.startswith("inherit:"),
                item.predictive_score,
            )
        )
        for child in children[: max(int(config.protected_ancestor_modes), 1)]:
            selected.append(child)
            selected_ids.add(id(child))
    for proposal in sorted(proposals, key=lambda item: item.predictive_score):
        if id(proposal) in selected_ids:
            continue
        if any(
            proposal.integer_offsets == other.integer_offsets
            and np.linalg.norm(
                np.asarray(proposal.normalized_values, dtype=float)
                - np.asarray(other.normalized_values, dtype=float)
            ) < max(float(config.proposal_mode_separation), 0.0)
            for other in selected
        ):
            continue
        selected.append(proposal)
        if len(selected) >= target:
            break
    return tuple(selected[:target])


def _select_modes_with_ancestry(
    modes: Sequence[NonlinearPathMode],
    *,
    parent_modes: Sequence[NonlinearPathMode],
    config: NonlinearPathTransportConfig,
) -> tuple[NonlinearPathMode, ...]:
    """Select a Laplace mixture without silently deleting an ancestor family."""

    target = max(
        int(config.path_beam) * max(int(config.maximum_modes_per_path), 1),
        len(parent_modes) * max(int(config.protected_ancestor_modes), 1),
    )
    selected: list[NonlinearPathMode] = []
    for parent in parent_modes:
        children = [mode for mode in modes if mode.parent_mode_id == parent.mode_id]
        children.sort(key=lambda mode: (-mode.log_mass, mode.objective_chi2))
        selected.extend(children[: max(int(config.protected_ancestor_modes), 1)])
    normal = _select_modes(modes, config=config)
    seen = {
        (
            mode.integer_offsets,
            tuple(np.rint(np.asarray(mode.normalized_values) * 1.0e9).astype(np.int64)),
        )
        for mode in selected
    }
    for mode in normal:
        if len(selected) >= target:
            break
        key = (
            mode.integer_offsets,
            tuple(np.rint(np.asarray(mode.normalized_values) * 1.0e9).astype(np.int64)),
        )
        if key in seen:
            continue
        selected.append(mode)
        seen.add(key)
    selected.sort(key=lambda mode: (-mode.log_mass, mode.objective_chi2))
    # The protected ancestor quota is part of the transport invariant. It is
    # applied before global evidence ordering and must not be truncated again.
    return _with_posterior_masses(selected)


def _initial_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    phase_noise: float,
    config: NonlinearPathTransportConfig,
    phase_beta: float = 1.0,
    sample_count: int | None = None,
    path_family_limit: int | None = None,
    seed_offset: int = 0,
    source_prefix: str = "initial",
) -> tuple[CompletePathProposal, ...]:
    lower, upper = _normalized_bounds(coordinates)
    requested = max(
        int(config.initial_samples_per_mode if sample_count is None else sample_count),
        0,
    )
    proposals: list[CompletePathProposal] = []
    for mode_index, mode in enumerate(invariant_modes):
        center = np.asarray(mode.normalized_values, dtype=float)
        covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
        candidates: list[tuple[np.ndarray, str]] = [(center, f"invariant:{mode_index}")]
        if requested > 0:
            exponent = int(math.ceil(math.log2(requested)))
            unit = qmc.Sobol(
                d=center.size,
                scramble=True,
                seed=int(config.random_seed)
                + int(seed_offset)
                + 7919 * mode_index,
            ).random_base2(exponent)[:requested]
            normal = ndtri(np.clip(unit, 1.0e-9, 1.0 - 1.0e-9))
            chol = np.linalg.cholesky(covariance)
            scales = tuple(float(value) for value in config.initial_scales if value > 0.0)
            if not scales:
                scales = (1.0,)
            for sample_index, row in enumerate(normal):
                scale = scales[sample_index % len(scales)]
                state = np.clip(center + scale * (chol @ row), lower, upper)
                candidates.append((state, f"initial-sobol:{mode_index}:k{scale:g}"))
        for state, source in candidates:
            try:
                profile = evaluator.evaluate_values(_physical_values(coordinates, state))
                paths = complete_path_candidates(
                    profile,
                    candidate_count=config.point_path_candidates,
                    maximum_lattice_nodes=config.maximum_lattice_nodes,
                    phase_noise=phase_noise,
                    phase_beta=phase_beta,
                )
                for path_index, path in enumerate(paths):
                    proposal = CompletePathProposal(
                        path,
                        state,
                        f"{source_prefix}:{source}:path{path_index}",
                        math.inf,
                        getattr(mode, "mode_id", None),
                    )
                    score = _score_proposal(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        proposal=proposal,
                        phase_noise=phase_noise,
                        prior_precision=config.prior_precision,
                        phase_beta=phase_beta,
                    )
                    proposals.append(replace(proposal, predictive_score=score))
            except Exception:
                continue
    return _deduplicate_proposals(
        proposals,
        limit=max(
            int(config.initial_path_families if path_family_limit is None else path_family_limit),
            1,
        ),
        maximum_starts_per_path=config.maximum_starts_per_path,
        mode_separation=config.proposal_mode_separation,
    )


def _bridge_block_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: NonlinearPathMode,
    profile: Any,
    phase_center: np.ndarray,
    phase_covariance: np.ndarray,
    phase_jacobian: np.ndarray,
    phase_noise: float,
    phase_beta: float,
    config: NonlinearPathTransportConfig,
) -> tuple[CompletePathProposal, ...]:
    """Propose complete paths through joint multi-block integer gauges."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(mode.normalized_values, dtype=float)
    current_path = np.asarray(mode.integer_offsets, dtype=np.int64)
    if current_path.shape != phase_center.shape:
        return ()
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    proposals: list[CompletePathProposal] = []
    for raw_kappa in config.lattice_kappas:
        kappa = max(float(raw_kappa), 1.0e-6)
        continuous_covariance = (kappa * kappa) * covariance
        predictive_covariance = _stable_covariance(
            phase_covariance
            + phase_jacobian @ continuous_covariance @ phase_jacobian.T
        )
        predictive_precision = _stable_inverse(predictive_covariance)
        gain = (
            continuous_covariance
            @ phase_jacobian.T
            @ predictive_precision
        )
        for block_count in config.bridge_block_counts:
            try:
                basis = temporal_bridge_block_basis(
                    profile,
                    block_count=int(block_count),
                )
            except Exception:
                continue
            if basis.shape[1] == 0:
                continue
            block_precision = _stable_covariance(
                basis.T @ predictive_precision @ basis
            )
            block_covariance = _stable_inverse(block_precision)
            block_center = (
                block_covariance
                @ basis.T
                @ predictive_precision
                @ (phase_center - current_path.astype(float))
            )
            try:
                lattice = solve_integer_least_squares(
                    block_center,
                    block_covariance,
                    candidate_count=max(
                        int(config.bridge_block_candidates_per_scale),
                        1,
                    ),
                    max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
                )
            except Exception:
                continue
            for candidate_index, candidate in enumerate(lattice.candidates):
                gauge = np.asarray(candidate.integer_vector, dtype=np.int64)
                path_array = current_path + np.asarray(
                    basis @ gauge,
                    dtype=np.int64,
                )
                path = tuple(int(value) for value in path_array)
                if path == mode.integer_offsets:
                    continue
                displacement = path_array.astype(float) - phase_center
                state = np.clip(center + gain @ displacement, lower, upper)
                proposal = CompletePathProposal(
                    path,
                    state,
                    (
                        f"block-lattice:{mode.mode_id}:blocks{basis.shape[1] + 1}:"
                        f"k{kappa:g}:c{candidate_index}"
                    ),
                    math.inf,
                    mode.mode_id,
                )
                proposals.append(
                    replace(
                        proposal,
                        predictive_score=_score_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            proposal=proposal,
                            phase_noise=phase_noise,
                            prior_precision=config.prior_precision,
                            phase_beta=phase_beta,
                        ),
                    )
                )
    return _deduplicate_proposals(
        proposals,
        limit=max(int(config.maximum_block_proposals_per_mode), 1),
        maximum_starts_per_path=1,
        mode_separation=config.proposal_mode_separation,
    )


def _transport_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[NonlinearPathMode],
    phase_noise: float,
    phase_beta: float,
    config: NonlinearPathTransportConfig,
    rng: np.random.Generator,
) -> tuple[CompletePathProposal, ...]:
    lower, upper = _normalized_bounds(coordinates)
    proposals: list[CompletePathProposal] = []
    for mode_index, mode in enumerate(modes):
        inherited = CompletePathProposal(
            mode.integer_offsets,
            np.asarray(mode.normalized_values, dtype=float),
            f"inherit:{mode.mode_id}",
            math.inf,
            mode.mode_id,
        )
        inherited_score = _score_proposal(
            evaluator=evaluator,
            coordinates=coordinates,
            proposal=inherited,
            phase_noise=phase_noise,
            prior_precision=config.prior_precision,
            phase_beta=phase_beta,
        )
        proposals.append(replace(inherited, predictive_score=inherited_score))

        center = np.asarray(mode.normalized_values, dtype=float)
        covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
        chol = np.linalg.cholesky(covariance)
        scales = tuple(float(value) for value in config.jitter_scales if value > 0.0)
        if not scales:
            scales = (1.0,)
        for sample_index in range(max(int(config.jitter_samples_per_mode), 0)):
            scale = scales[sample_index % len(scales)]
            state = np.clip(
                center + scale * (chol @ rng.normal(size=center.size)),
                lower,
                upper,
            )
            try:
                profile = evaluator.evaluate_values(_physical_values(coordinates, state))
                paths = complete_path_candidates(
                    profile,
                    candidate_count=config.point_path_candidates,
                    maximum_lattice_nodes=config.maximum_lattice_nodes,
                    phase_noise=phase_noise,
                    phase_beta=phase_beta,
                )
                for path_index, path in enumerate(paths):
                    proposal = CompletePathProposal(
                        path,
                        state,
                        f"jitter:{mode.mode_id}:k{scale:g}:path{path_index}",
                        math.inf,
                        mode.mode_id,
                    )
                    score = _score_proposal(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        proposal=proposal,
                        phase_noise=phase_noise,
                        prior_precision=config.prior_precision,
                        phase_beta=phase_beta,
                    )
                    proposals.append(replace(proposal, predictive_score=score))
            except Exception:
                continue

        try:
            steps = _phase_resolving_steps(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=center,
                config=config,
            )
            profile, phase_jacobian = _relative_phase_jacobian(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=center,
                steps=steps,
            )
            phase_center, phase_covariance = relative_phase_geometry(profile)
        except Exception:
            continue
        if phase_noise > 0.0:
            phase_covariance = phase_covariance + phase_noise * phase_noise * np.eye(
                phase_center.size, dtype=float
            )
        proposals.extend(
            _bridge_block_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                mode=mode,
                profile=profile,
                phase_center=phase_center,
                phase_covariance=phase_covariance,
                phase_jacobian=phase_jacobian,
                phase_noise=phase_noise,
                phase_beta=phase_beta,
                config=config,
            )
        )
        for raw_kappa in config.lattice_kappas:
            kappa = max(float(raw_kappa), 1.0e-6)
            continuous_covariance = (kappa * kappa) * covariance
            predictive_covariance = _stable_covariance(
                phase_covariance
                + phase_jacobian @ continuous_covariance @ phase_jacobian.T
            )
            try:
                lattice = solve_integer_least_squares(
                    phase_center,
                    predictive_covariance,
                    candidate_count=max(int(config.lattice_candidates_per_scale), 1),
                    max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
                )
                gain = (
                    continuous_covariance
                    @ phase_jacobian.T
                    @ _stable_inverse(predictive_covariance)
                )
                sign, logdet = np.linalg.slogdet(predictive_covariance)
                volume = float(logdet) if sign > 0.0 else math.inf
                for candidate in lattice.candidates:
                    path = tuple(int(value) for value in candidate.integer_vector)
                    displacement = np.asarray(path, dtype=float) - phase_center
                    state = np.clip(center + gain @ displacement, lower, upper)
                    proposal = CompletePathProposal(
                        path,
                        state,
                        f"lattice:{mode.mode_id}:k{kappa:g}",
                        float(candidate.squared_mahalanobis + volume),
                        mode.mode_id,
                    )
                    exact_score = _score_proposal(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        proposal=proposal,
                        phase_noise=phase_noise,
                        prior_precision=config.prior_precision,
                        phase_beta=phase_beta,
                    )
                    proposals.append(replace(proposal, predictive_score=exact_score))
            except Exception:
                continue
    return _select_ancestry_proposals(
        proposals,
        parent_modes=modes,
        config=config,
    )


def _initialize_adaptive_beta_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    island_bridge_seeds: Sequence[CompletePathProposal],
    config: NonlinearPathTransportConfig,
) -> tuple[float, tuple[NonlinearPathMode, ...], NonlinearPathStageSummary]:
    beta = min(max(float(config.phase_beta_initial), 0.0), 1.0)
    if beta <= 0.0:
        beta = min(max(float(config.phase_beta_minimum_step), 1.0e-12), 1.0)
    initial = _initial_proposals(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=invariant_modes,
        phase_noise=0.0,
        phase_beta=beta,
        config=config,
        source_prefix="root",
    )
    collapsed = _collapsed_mode_proposals(
        evaluator=evaluator,
        coordinates=coordinates,
        modes=invariant_modes,
        phase_noise=max(float(config.phase_noise_ladder[0]), 0.0)
        if config.phase_noise_ladder
        else 0.0,
        phase_beta=beta,
        temperature_index=0,
        starts_per_mode=config.collapsed_starts_per_mode,
        path_family_limit=config.initial_path_families,
        config=config,
    )
    island = tuple(
        replace(
            proposal,
            predictive_score=_score_proposal(
                evaluator=evaluator,
                coordinates=coordinates,
                proposal=proposal,
                phase_noise=0.0,
                phase_beta=beta,
                prior_precision=config.prior_precision,
            ),
        )
        for proposal in island_bridge_seeds
    )
    proposals = _deduplicate_proposals(
        (*initial, *collapsed, *island),
        limit=max(int(config.prefit_path_families), 1),
        maximum_starts_per_path=config.maximum_starts_per_path,
        mode_separation=config.proposal_mode_separation,
    )
    fitted: list[NonlinearPathMode] = []
    for proposal_index, proposal in enumerate(proposals):
        mode = _fit_complete_path(
            evaluator=evaluator,
            coordinates=coordinates,
            proposal=proposal,
            phase_noise=0.0,
            phase_beta=beta,
            temperature_index=0,
            mode_index=proposal_index,
            config=config,
        )
        if mode is not None:
            fitted.append(mode)
    current = _select_modes(fitted, config=config)
    stage = NonlinearPathStageSummary(
        temperature_index=0,
        phase_noise=0.0,
        input_mode_count=len(invariant_modes),
        proposal_count=len(proposals),
        unique_path_count=len({item.integer_offsets for item in proposals}),
        inherited_path_count=0,
        reopened_path_count=len(
            {item.integer_offsets for item in proposals if item.source.startswith("root:")}
        ),
        collapsed_mode_count=len(
            {
                item.integer_offsets
                for item in proposals
                if item.source.startswith("collapsed:")
            }
        ),
        joint_path_update_count=sum(mode.path_update_count for mode in fitted),
        block_path_count=0,
        island_bridge_path_count=len(
            {
                item.integer_offsets
                for item in proposals
                if item.source.startswith("island-bridge:")
            }
        ),
        refitted_count=len(fitted),
        retained_count=len(current),
        best_shape_chi2=min((mode.shape_chi2 for mode in current), default=math.inf),
        best_phase_chi2=min((mode.phase_chi2 for mode in current), default=math.inf),
        best_objective_chi2=min(
            (mode.objective_chi2 for mode in current), default=math.inf
        ),
        phase_beta=beta,
        effective_sample_size=effective_sample_size([mode.log_mass for mode in current]),
        ancestor_overlap=1.0 if current else 0.0,
    )
    return beta, current, stage


def _run_adaptive_beta_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    config: NonlinearPathTransportConfig,
    stage_callback: Any | None = None,
    resume_modes: Sequence[NonlinearPathMode] = (),
    resume_history: Sequence[NonlinearPathStageSummary] = (),
) -> NonlinearPathTransportResult:
    """Transport a Laplace mode mixture through the exact phase likelihood."""

    if not invariant_modes:
        raise ValueError("At least one invariant DD mode is required.")
    first_profile = evaluator.evaluate_values(
        np.asarray(invariant_modes[0].parameter_values, dtype=float)
    )
    session_count = int(first_profile.session_count)
    rng = np.random.default_rng(int(config.random_seed))
    history: list[NonlinearPathStageSummary] = []
    portfolios: list[tuple[NonlinearPathMode, ...]] = []
    try:
        island_bridge_seeds, _ = _discover_island_bridge_proposals(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            phase_noise=0.0,
            config=config,
        )
    except Exception:
        island_bridge_seeds = ()

    if resume_modes:
        current = tuple(resume_modes)
        history.extend(resume_history)
        portfolios.append(current)
        beta = float(current[0].phase_beta)
        start_temperature_index = (
            max((stage.temperature_index for stage in history), default=0) + 1
        )
    else:
        beta, current, initial_stage = _initialize_adaptive_beta_transport(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            island_bridge_seeds=island_bridge_seeds,
            config=config,
        )
        portfolios.append(current)
        history.append(initial_stage)
        start_temperature_index = 1
        if stage_callback is not None:
            stage_callback(history[-1], current)

    beta_one_refinements = 0
    for temperature_index in range(
        start_temperature_index,
        start_temperature_index + max(int(config.phase_beta_maximum_steps), 1),
    ):
        if not current:
            break
        if beta >= 1.0:
            beta_one_refinements += 1
            if beta_one_refinements >= max(int(config.phase_beta_refinement_steps), 1):
                break
            next_beta = 1.0
            overlap_ess = float(len(current))
        else:
            next_beta, overlap_ess = _select_mode_aware_phase_beta(
                evaluator=evaluator,
                coordinates=coordinates,
                modes=current,
                current_beta=beta,
                config=config,
                temperature_index=temperature_index,
            )
        parents = current
        transported = _transport_proposals(
            evaluator=evaluator,
            coordinates=coordinates,
            modes=parents,
            phase_noise=0.0,
            phase_beta=next_beta,
            config=config,
            rng=rng,
        )
        reopened = _initial_proposals(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            phase_noise=0.0,
            phase_beta=next_beta,
            config=config,
            sample_count=config.refresh_samples_per_mode,
            path_family_limit=config.refresh_path_families,
            seed_offset=104729 * temperature_index,
            source_prefix=f"reopen_t{temperature_index:02d}",
        )
        island = tuple(
            replace(
                proposal,
                predictive_score=_score_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    proposal=proposal,
                    phase_noise=0.0,
                    phase_beta=next_beta,
                    prior_precision=config.prior_precision,
                ),
            )
            for proposal in island_bridge_seeds
        )
        proposals = _select_ancestry_proposals(
            (*transported, *reopened, *island),
            parent_modes=parents,
            config=config,
        )
        fitted = []
        for proposal_index, proposal in enumerate(proposals):
            mode = _fit_complete_path(
                evaluator=evaluator,
                coordinates=coordinates,
                proposal=proposal,
                phase_noise=0.0,
                phase_beta=next_beta,
                temperature_index=temperature_index,
                mode_index=proposal_index,
                config=config,
            )
            if mode is not None:
                fitted.append(mode)
        current = _select_modes_with_ancestry(
            fitted,
            parent_modes=parents,
            config=config,
        )
        represented = {
            mode.parent_mode_id for mode in current if mode.parent_mode_id is not None
        }
        parent_ids = {mode.mode_id for mode in parents}
        overlap = (
            len(represented & parent_ids) / len(parent_ids) if parent_ids else 1.0
        )
        portfolios.append(current)
        history.append(
            NonlinearPathStageSummary(
                temperature_index=temperature_index,
                phase_noise=0.0,
                input_mode_count=len(parents),
                proposal_count=len(proposals),
                unique_path_count=len({item.integer_offsets for item in proposals}),
                inherited_path_count=len(
                    {
                        item.integer_offsets
                        for item in proposals
                        if item.source.startswith("inherit:")
                    }
                ),
                reopened_path_count=len(
                    {
                        item.integer_offsets
                        for item in proposals
                        if item.source.startswith("reopen_")
                    }
                ),
                collapsed_mode_count=0,
                joint_path_update_count=sum(mode.path_update_count for mode in fitted),
                block_path_count=len(
                    {
                        item.integer_offsets
                        for item in proposals
                        if item.source.startswith("block-lattice:")
                    }
                ),
                island_bridge_path_count=len(
                    {
                        item.integer_offsets
                        for item in proposals
                        if item.source.startswith("island-bridge:")
                    }
                ),
                refitted_count=len(fitted),
                retained_count=len(current),
                best_shape_chi2=min(
                    (mode.shape_chi2 for mode in current), default=math.inf
                ),
                best_phase_chi2=min(
                    (mode.phase_chi2 for mode in current), default=math.inf
                ),
                best_objective_chi2=min(
                    (mode.objective_chi2 for mode in current), default=math.inf
                ),
                phase_beta=next_beta,
                effective_sample_size=overlap_ess,
                ancestor_overlap=float(overlap),
            )
        )
        if stage_callback is not None:
            stage_callback(history[-1], current)
        beta = next_beta

    reached = bool(current) and beta >= 1.0
    return NonlinearPathTransportResult(
        session_count=session_count,
        final_modes=current,
        stage_history=tuple(history),
        level_portfolios=tuple(portfolios),
        reached_all_sessions=reached,
        failure_class=None if reached else "adaptive-phase-tempering-incomplete",
    )


def _run_noise_ladder_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    config: NonlinearPathTransportConfig | None = None,
    stage_callback: Any | None = None,
) -> NonlinearPathTransportResult:
    """Transport a complete-path posterior through a wrapped-phase ladder."""

    active = config or NonlinearPathTransportConfig()
    ladder = tuple(
        sorted(
            {
                float(value)
                for value in active.phase_noise_ladder
                if np.isfinite(float(value)) and float(value) >= 0.0
            },
            reverse=True,
        )
    )
    if not ladder:
        raise ValueError("At least one finite non-negative phase noise is required.")
    if not invariant_modes:
        raise ValueError("At least one invariant DD mode is required.")
    rng = np.random.default_rng(int(active.random_seed))
    first_profile = evaluator.evaluate_values(
        np.asarray(invariant_modes[0].parameter_values, dtype=float)
    )
    session_count = int(first_profile.session_count)
    history: list[NonlinearPathStageSummary] = []
    portfolios: list[tuple[NonlinearPathMode, ...]] = []
    current: tuple[NonlinearPathMode, ...] = ()
    try:
        island_bridge_seeds, _ = _discover_island_bridge_proposals(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            phase_noise=0.0,
            config=active,
        )
    except Exception:
        island_bridge_seeds = ()

    for temperature_index, phase_noise in enumerate(ladder):
        island_bridge_proposals = tuple(
            replace(
                proposal,
                predictive_score=_score_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    proposal=proposal,
                    phase_noise=phase_noise,
                    prior_precision=active.prior_precision,
                ),
            )
            for proposal in island_bridge_seeds
        )
        if temperature_index == 0:
            initial = _initial_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                invariant_modes=invariant_modes,
                phase_noise=phase_noise,
                config=active,
                source_prefix="root",
            )
            collapsed = _collapsed_mode_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                modes=invariant_modes,
                phase_noise=phase_noise,
                temperature_index=temperature_index,
                starts_per_mode=active.collapsed_starts_per_mode,
                path_family_limit=active.initial_path_families,
                config=active,
            )
            proposals = _deduplicate_proposals(
                (*initial, *collapsed, *island_bridge_proposals),
                limit=max(int(active.prefit_path_families), 1),
                maximum_starts_per_path=active.maximum_starts_per_path,
                mode_separation=active.proposal_mode_separation,
            )
        else:
            transported = _transport_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                modes=current,
                phase_noise=phase_noise,
                phase_beta=1.0,
                config=active,
                rng=rng,
            )
            collapsed = _collapsed_mode_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                modes=(*invariant_modes, *current),
                phase_noise=phase_noise,
                temperature_index=temperature_index,
                starts_per_mode=active.collapsed_refresh_starts_per_mode,
                path_family_limit=active.refresh_path_families,
                config=active,
            )
            reopened = _initial_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                invariant_modes=invariant_modes,
                phase_noise=phase_noise,
                config=active,
                sample_count=active.refresh_samples_per_mode,
                path_family_limit=active.refresh_path_families,
                seed_offset=104729 * temperature_index,
                source_prefix=f"reopen_t{temperature_index:02d}",
            )
            proposals = _deduplicate_proposals(
                (
                    *transported,
                    *collapsed,
                    *reopened,
                    *island_bridge_proposals,
                ),
                limit=max(int(active.prefit_path_families), 1),
                maximum_starts_per_path=active.maximum_starts_per_path,
                mode_separation=active.proposal_mode_separation,
            )
        fitted: list[NonlinearPathMode] = []
        for proposal_index, proposal in enumerate(proposals):
            mode = _fit_complete_path(
                evaluator=evaluator,
                coordinates=coordinates,
                proposal=proposal,
                phase_noise=phase_noise,
                phase_beta=1.0,
                temperature_index=temperature_index,
                mode_index=proposal_index,
                config=active,
            )
            if mode is not None:
                fitted.append(mode)
        current = _select_modes(fitted, config=active)
        portfolios.append(current)
        history.append(
            NonlinearPathStageSummary(
                temperature_index=temperature_index,
                phase_noise=phase_noise,
                input_mode_count=(len(invariant_modes) if temperature_index == 0 else len(portfolios[-2])),
                proposal_count=len(proposals),
                unique_path_count=len({proposal.integer_offsets for proposal in proposals}),
                inherited_path_count=len(
                    {
                        proposal.integer_offsets
                        for proposal in proposals
                        if proposal.source.startswith("inherit:")
                    }
                ),
                reopened_path_count=(
                    len(
                        {
                            proposal.integer_offsets
                            for proposal in proposals
                            if proposal.source.startswith("reopen_")
                        }
                    )
                    if temperature_index > 0
                    else len(
                        {
                            proposal.integer_offsets
                            for proposal in proposals
                            if proposal.source.startswith("root:")
                        }
                    )
                ),
                collapsed_mode_count=len(
                    {
                        proposal.integer_offsets
                        for proposal in proposals
                        if proposal.source.startswith("collapsed:")
                    }
                ),
                joint_path_update_count=sum(
                    mode.path_update_count for mode in fitted
                ),
                block_path_count=len(
                    {
                        proposal.integer_offsets
                        for proposal in proposals
                        if proposal.source.startswith("block-lattice:")
                    }
                ),
                island_bridge_path_count=len(
                    {
                        proposal.integer_offsets
                        for proposal in proposals
                        if proposal.source.startswith("island-bridge:")
                    }
                ),
                refitted_count=len(fitted),
                retained_count=len(current),
                best_shape_chi2=min((mode.shape_chi2 for mode in current), default=math.inf),
                best_phase_chi2=min((mode.phase_chi2 for mode in current), default=math.inf),
                best_objective_chi2=min(
                    (mode.objective_chi2 for mode in current), default=math.inf
                ),
            )
        )
        if stage_callback is not None:
            stage_callback(history[-1], current)
        if not current:
            break

    reached = bool(current) and math.isclose(
        float(current[0].phase_noise),
        float(ladder[-1]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    failure = None if reached else "nonlinear-path-transport-collapse"
    return NonlinearPathTransportResult(
        session_count=session_count,
        final_modes=current,
        stage_history=tuple(history),
        level_portfolios=tuple(portfolios),
        reached_all_sessions=reached,
        failure_class=failure,
    )


def run_bayesian_nonlinear_path_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    config: NonlinearPathTransportConfig | None = None,
    stage_callback: Any | None = None,
    resume_modes: Sequence[NonlinearPathMode] = (),
    resume_history: Sequence[NonlinearPathStageSummary] = (),
) -> NonlinearPathTransportResult:
    """Run APOLLO's hard-source Bayesian path transport.

    Adaptive exact-phase tempering is the production route. The former
    additive-noise ladder remains available only when explicitly disabled for
    ablation and regression comparison.
    """

    active = config or NonlinearPathTransportConfig()
    if active.adaptive_phase_tempering:
        return _run_adaptive_beta_transport(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            config=active,
            stage_callback=stage_callback,
            resume_modes=resume_modes,
            resume_history=resume_history,
        )
    return _run_noise_ladder_transport(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=invariant_modes,
        config=active,
        stage_callback=stage_callback,
    )


__all__ = [
    "CompletePathProposal",
    "NonlinearPathMode",
    "NonlinearPathStageSummary",
    "NonlinearPathTransportConfig",
    "NonlinearPathTransportResult",
    "collapsed_continuous_objective",
    "canonical_island_path_candidates",
    "complete_path_candidates",
    "collapsed_island_continuous_objective",
    "island_relative_phase_geometry",
    "nearest_complete_path",
    "relative_phase_geometry",
    "effective_sample_size",
    "select_adaptive_phase_beta",
    "run_bayesian_nonlinear_path_transport",
    "tempered_complete_path_residual",
    "tempered_island_path_residual",
    "temporal_phase_islands",
]
