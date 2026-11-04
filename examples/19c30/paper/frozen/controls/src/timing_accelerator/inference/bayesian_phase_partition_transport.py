from __future__ import annotations

"""Bayesian transport on a nested phase-partition posterior.

This is the hard-source route for long-gap binary timing.  A mode contains
only continuous DD parameters.  Integer session connections are marginalized
inside each phase block and are used only to construct predictive proposals
when blocks are merged.  The integer path is recovered after the final
single-block posterior has been formed.

The distinction is important: a wrong integer path cannot become an ancestor
that the continuous posterior is forced to follow.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from timing_accelerator.inference.collapsed_integer_posterior import (
    conditional_integer_paths,
    collapsed_wrapped_log_likelihood,
)
from timing_accelerator.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from timing_accelerator.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
    integer_session_offsets_to_gap_wraps,
)


@dataclass(frozen=True)
class PhasePartitionTransportConfig:
    """Controls for the partition-collapsed Bayesian transport."""

    prior_precision: float = 1.0
    adaptive_partition_merges: bool = True
    maximum_partition_merges: int = 0
    partition_counts: tuple[int, ...] = (18, 14, 10, 8, 6, 4, 3, 2, 1)
    # Empty means adaptive conditional-ESS continuation. Explicit values are
    # retained for deterministic tests and controlled ablations.
    bridge_betas: tuple[float, ...] = ()
    bridge_ess_target_fraction: float = 0.55
    bridge_minimum_beta_step: float = 1.0e-5
    bridge_maximum_beta_step: float = 0.25
    bridge_maximum_steps: int = 48
    bridge_overlap_samples: int = 8
    direct_bridge_first: bool = True
    direct_bridge_shape_ratio_limit: float = 3.0
    direct_bridge_shape_additive_limit: float = 25.0
    direct_bridge_phase_chi2_additive_limit: float = 25.0
    direct_bridge_phase_chi2_per_dimension_limit: float = 9.0
    laplace_covariance_scale: float = 1.0
    mode_beam: int = 32
    protected_modes_per_parent: int = 2
    protected_parent_mass_floor: float = 1.0e-6
    protected_parent_beam_fraction: float = 0.5
    maximum_proposal_parents: int = 4
    exploration_proposal_parents: int = 1
    maximum_optimized_proposals: int = 8
    protected_optimized_proposals_per_parent: int = 1
    tempered_maximum_optimized_proposals: int = 4
    tempered_optimizer_maxiter: int = 12
    conditional_alias_refinement: bool = True
    bridge_homotopy_betas: tuple[float, ...] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        0.05,
        0.2,
        0.5,
        1.0,
    )
    proposal_aware_merge_acquisition: bool = True
    merge_acquisition_workers: int = 1
    maximum_merge_acquisition_homotopy_paths: int = 1
    merge_acquisition_homotopy_max_nfev: int = 24
    maximum_bridge_homotopy_paths: int = 4
    bridge_homotopy_max_nfev: int = 48
    proposal_candidates_per_scale: int = 8
    maximum_proposals_per_parent: int = 32
    maximum_lattice_nodes: int = 2_000_000
    predictive_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    bridge_gauge_tail_probability: float = 1.0e-3
    optimizer_maxiter: int = 48
    finite_difference_step: float = 1.0e-6
    phase_step_cycles: float = 0.05
    mode_separation: float = 0.025
    collapsed_hermite_order: int = 24
    collapsed_maximum_wrap_terms: int = 12
    path_candidates: int = 64
    path_fit_max_nfev: int = 96
    random_seed: int = 20260724


@dataclass(frozen=True)
class PhasePartitionMode:
    mode_id: str
    parent_mode_id: str | None
    proposal_source: str
    level_index: int
    partition: tuple[tuple[int, ...], ...]
    parameter_values: np.ndarray
    normalized_values: np.ndarray
    covariance: np.ndarray
    shape_chi2: float
    collapsed_phase_deviance: float
    prior_chi2: float
    objective_chi2: float
    log_mass: float
    posterior_mass: float
    nfev: int
    phase_beta: float
    success: bool


@dataclass(frozen=True)
class PhasePartitionPathMode:
    mode_id: str
    parent_mode_id: str | None
    proposal_source: str
    partition: tuple[tuple[int, ...], ...]
    integer_offsets: tuple[int, ...]
    wraps: tuple[int, ...]
    parameter_values: np.ndarray
    normalized_values: np.ndarray
    covariance: np.ndarray
    shape_chi2: float
    phase_chi2: float
    prior_chi2: float
    objective_chi2: float
    nfev: int
    success: bool

    def chronological_integer_offsets(self, session_count: int) -> tuple[int, ...]:
        if len(self.integer_offsets) != int(session_count) - 1:
            raise ValueError("The complete path does not match the session count.")
        return self.integer_offsets


@dataclass(frozen=True)
class PhasePartitionStageSummary:
    level_index: int
    beta: float
    old_partition: tuple[tuple[int, ...], ...]
    new_partition: tuple[tuple[int, ...], ...]
    input_mode_count: int
    proposal_count: int
    retained_count: int
    unique_mode_count: int
    best_shape_chi2: float
    best_collapsed_phase_deviance: float
    best_objective_chi2: float
    ancestor_coverage: float
    bridge_effective_sample_size: float = 0.0
    bridge_relative_effective_sample_size: float = 0.0
    candidate_merge_count: int = 1
    selected_merge_index: int = -1
    selected_bridge_full_ess_fraction: float = 0.0
    selected_bridge_log_evidence_increment: float = 0.0
    selected_bridge_initial_adaptive_beta: float = 0.0
    selected_bridge_log_factor_standard_deviation: float = math.inf
    selected_merge_homotopy_transportable: bool = False
    selected_merge_homotopy_shape_chi2: float = math.inf
    selected_merge_homotopy_objective_chi2: float = math.inf
    direct_bridge_attempted: bool = False
    direct_bridge_accepted: bool = False
    direct_bridge_best_shape_chi2: float = math.inf
    direct_bridge_best_phase_chi2: float = math.inf
    direct_bridge_phase_chi2_limit: float = math.inf
    direct_bridge_best_objective_chi2: float = math.inf
    direct_bridge_proposal_count: int = 0
    direct_bridge_optimized_proposal_count: int = 0
    optimized_proposal_count: int = 0
    merge_scores: tuple[PhasePartitionMergeScore, ...] = ()


@dataclass(frozen=True)
class PhasePartitionMergeScore:
    """Posterior-overlap diagnostic for one candidate island merge."""

    merge_index: int
    partition: tuple[tuple[int, ...], ...]
    effective_sample_size: float
    relative_effective_sample_size: float
    log_evidence_increment: float
    initial_adaptive_beta: float = 0.0
    log_factor_standard_deviation: float = math.inf
    homotopy_attempted: bool = False
    homotopy_transportable: bool = False
    homotopy_shape_chi2: float = math.inf
    homotopy_phase_chi2: float = math.inf
    homotopy_phase_chi2_limit: float = math.inf
    homotopy_bridge_integer: int | None = None
    homotopy_objective_chi2: float = math.inf


@dataclass(frozen=True)
class PhasePartitionBridgeGaugePosterior:
    """One-dimensional posterior introduced by one adjacent island merge."""

    merge_index: int
    phase_center: float
    measurement_variance: float
    phase_jacobian: np.ndarray
    old_integer_vector: np.ndarray


@dataclass(frozen=True)
class PhasePartitionBridgeGaugeCandidate:
    """A discrete bridge gauge and its conditional continuous proposal."""

    bridge_integer: int
    predictive_scale: float
    log_weight: float
    normalized_values: np.ndarray
    new_integer_vector: np.ndarray


@dataclass(frozen=True)
class PhaseIslandGaugePosterior:
    """Joint continuous posterior of all island gauges relative to island zero."""

    phase_center: np.ndarray
    measurement_covariance: np.ndarray
    phase_jacobian: np.ndarray
    old_integer_vector: np.ndarray


@dataclass(frozen=True)
class PhaseIslandGaugeCandidate:
    """One joint integer island-gauge mode and its conditional DD proposal."""

    gauge_integers: tuple[int, ...]
    predictive_scale: float
    log_weight: float
    normalized_values: np.ndarray
    new_integer_vector: np.ndarray


@dataclass(frozen=True)
class PhasePartitionTransportResult:
    session_count: int
    final_modes: tuple[PhasePartitionMode, ...]
    path_modes: tuple[PhasePartitionPathMode, ...]
    stage_history: tuple[PhasePartitionStageSummary, ...]
    partition_history: tuple[tuple[tuple[int, ...], ...], ...]
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


def _stable_covariance(matrix: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    if symmetric.size == 0:
        return symmetric.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors * clipped) @ eigenvectors.T


def _stable_inverse(matrix: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (values + values.T)
    if symmetric.size == 0:
        return symmetric.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors / clipped) @ eigenvectors.T


def temporal_phase_partition(
    profile: SessionShapeProfile,
    *,
    block_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Return a nested temporal partition cut at the largest observation gaps."""

    count = int(profile.session_count)
    if count < 1:
        raise ValueError("A phase partition requires at least one session.")
    epochs = np.asarray(profile.session_epochs, dtype=float)
    if epochs.shape != (count,) or not np.all(np.isfinite(epochs)):
        raise ValueError("Session epochs must match the session count.")
    requested = min(max(int(block_count), 1), count)
    order = np.argsort(epochs, kind="stable")
    gaps = np.diff(epochs[order])
    cut_after = (
        set(int(index) for index in np.argsort(gaps, kind="stable")[-(requested - 1) :])
        if requested > 1
        else set()
    )
    blocks: list[list[int]] = [[]]
    for position, session in enumerate(order.tolist()):
        blocks[-1].append(int(session))
        if position in cut_after and position < count - 1:
            blocks.append([])
    result = tuple(tuple(block) for block in blocks if block)
    if sorted(session for block in result for session in block) != list(range(count)):
        raise ValueError("The phase partition must cover every session exactly once.")
    return result


def adjacent_phase_partition_merges(
    partition: Sequence[Sequence[int]],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return every partition formed by joining one adjacent phase island."""

    blocks = tuple(tuple(int(session) for session in block) for block in partition)
    if any(not block for block in blocks):
        raise ValueError("Phase-island blocks must be non-empty.")
    if len(blocks) <= 1:
        return ()
    candidates: list[tuple[tuple[int, ...], ...]] = []
    for merge_index in range(len(blocks) - 1):
        merged = blocks[merge_index] + blocks[merge_index + 1]
        candidates.append(
            blocks[:merge_index] + (merged,) + blocks[merge_index + 2 :]
        )
    return tuple(candidates)


def adjacent_phase_partition_merge_index(
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
) -> int | None:
    """Return the joined block index when exactly one adjacent pair was merged."""

    old_blocks = tuple(tuple(int(session) for session in block) for block in old_partition)
    new_blocks = tuple(tuple(int(session) for session in block) for block in new_partition)
    if len(old_blocks) != len(new_blocks) + 1:
        return None
    for merge_index, candidate in enumerate(adjacent_phase_partition_merges(old_blocks)):
        if candidate == new_blocks:
            return int(merge_index)
    return None


def _partition_integer_segments(
    partition: Sequence[Sequence[int]],
    integer_vector: Sequence[int],
) -> tuple[np.ndarray, ...]:
    blocks = tuple(tuple(int(session) for session in block) for block in partition)
    integers = np.asarray(integer_vector, dtype=float)
    expected = sum(max(len(block) - 1, 0) for block in blocks)
    if integers.shape != (expected,):
        raise ValueError("The integer vector must match the partition dimension.")
    segments: list[np.ndarray] = []
    cursor = 0
    for block in blocks:
        width = max(len(block) - 1, 0)
        segments.append(integers[cursor : cursor + width].copy())
        cursor += width
    return tuple(segments)


def merged_partition_integer_vector(
    *,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    old_integer_vector: Sequence[int],
    bridge_integer: int,
) -> np.ndarray:
    """Join two fixed internal paths with the single new island-gauge integer."""

    merge_index = adjacent_phase_partition_merge_index(old_partition, new_partition)
    if merge_index is None:
        raise ValueError("The new partition must merge exactly one adjacent block pair.")
    segments = _partition_integer_segments(old_partition, old_integer_vector)
    result: list[float] = []
    block_index = 0
    while block_index < len(segments):
        if block_index != merge_index:
            result.extend(float(value) for value in segments[block_index])
            block_index += 1
            continue
        gauge = float(int(bridge_integer))
        result.extend(float(value) for value in segments[block_index])
        result.append(gauge)
        result.extend(gauge + float(value) for value in segments[block_index + 1])
        block_index += 2
    expected = sum(max(len(tuple(block)) - 1, 0) for block in new_partition)
    if len(result) != expected:
        raise ValueError("The merged integer vector has the wrong dimension.")
    return np.asarray(result, dtype=float)


def partition_bridge_gauge_geometry(
    *,
    profile: SessionShapeProfile,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    old_integer_vector: Sequence[int],
) -> tuple[float, float, int]:
    """Condition two island phase gauges on their fixed internal integer paths."""

    merge_index = adjacent_phase_partition_merge_index(old_partition, new_partition)
    if merge_index is None:
        raise ValueError("Bridge-gauge geometry requires one adjacent block merge.")
    blocks = tuple(tuple(int(session) for session in block) for block in old_partition)
    segments = _partition_integer_segments(blocks, old_integer_vector)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)

    def block_gauge(block_index: int) -> tuple[float, float]:
        sessions = np.asarray(blocks[block_index], dtype=int)
        internal = np.concatenate((np.zeros(1, dtype=float), segments[block_index]))
        local_variances = variances[sessions]
        if (
            sessions.size != internal.size
            or not np.all(np.isfinite(local_variances))
            or np.any(local_variances <= 0.0)
        ):
            raise ValueError("Session phase variances must be finite and positive.")
        precision = 1.0 / local_variances
        variance = 1.0 / float(np.sum(precision))
        center = variance * float(precision @ (offsets[sessions] - internal))
        return center, variance

    left_center, left_variance = block_gauge(merge_index)
    right_center, right_variance = block_gauge(merge_index + 1)
    return (
        float(right_center - left_center),
        float(left_variance + right_variance),
        int(merge_index),
    )


def partition_island_gauge_geometry(
    *,
    profile: SessionShapeProfile,
    partition: Sequence[Sequence[int]],
    old_integer_vector: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Condition every island gauge and express it relative to island zero."""

    blocks = tuple(tuple(int(session) for session in block) for block in partition)
    if not blocks or any(not block for block in blocks):
        raise ValueError("A phase-island partition must contain non-empty blocks.")
    segments = _partition_integer_segments(blocks, old_integer_vector)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    centers: list[float] = []
    gauge_variances: list[float] = []
    for block, segment in zip(blocks, segments, strict=True):
        sessions = np.asarray(block, dtype=int)
        internal = np.concatenate((np.zeros(1, dtype=float), segment))
        local_variances = variances[sessions]
        if (
            sessions.size != internal.size
            or not np.all(np.isfinite(local_variances))
            or np.any(local_variances <= 0.0)
        ):
            raise ValueError("Session phase variances must be finite and positive.")
        precision = 1.0 / local_variances
        gauge_variance = 1.0 / float(np.sum(precision))
        centers.append(
            gauge_variance
            * float(precision @ (offsets[sessions] - internal))
        )
        gauge_variances.append(gauge_variance)
    if len(blocks) == 1:
        return np.zeros(0, dtype=float), np.zeros((0, 0), dtype=float)
    relative = np.asarray(centers[1:], dtype=float) - float(centers[0])
    covariance = np.diag(np.asarray(gauge_variances[1:], dtype=float))
    covariance += float(gauge_variances[0])
    return relative, _stable_covariance(covariance)


def single_block_integer_vector_from_island_gauges(
    *,
    partition: Sequence[Sequence[int]],
    old_integer_vector: Sequence[int],
    gauge_integers: Sequence[int],
) -> np.ndarray:
    """Reconstruct one complete path from fixed internals and joint gauges."""

    blocks = tuple(tuple(int(session) for session in block) for block in partition)
    segments = _partition_integer_segments(blocks, old_integer_vector)
    gauges = np.asarray(gauge_integers, dtype=float)
    if gauges.shape != (max(len(blocks) - 1, 0),):
        raise ValueError("The gauge vector must contain one integer per non-reference island.")
    result: list[float] = [float(value) for value in segments[0]]
    for block_index in range(1, len(blocks)):
        gauge = float(gauges[block_index - 1])
        result.append(gauge)
        result.extend(gauge + float(value) for value in segments[block_index])
    expected = sum(len(block) for block in blocks) - 1
    if len(result) != expected:
        raise ValueError("The complete island path has the wrong dimension.")
    return np.asarray(result, dtype=float)


def partition_integer_vector_from_complete_path(
    *,
    partition: Sequence[Sequence[int]],
    integer_offsets: Sequence[int],
) -> np.ndarray:
    """Project one complete session path into within-island coordinates."""

    path = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.asarray(integer_offsets, dtype=np.int64))
    )
    blocks = tuple(tuple(int(session) for session in block) for block in partition)
    covered = [session for block in blocks for session in block]
    if any(not block for block in blocks) or sorted(covered) != list(range(path.size)):
        raise ValueError("The partition and complete integer path must cover every session.")
    return np.asarray(
        [
            int(path[session] - path[block[0]])
            for block in blocks
            for session in block[1:]
        ],
        dtype=float,
    )


def partition_relative_phase_geometry(
    profile: SessionShapeProfile,
    partition: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    """Return within-block phase coordinates and their block-diagonal covariance."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    count = int(profile.session_count)
    blocks = tuple(tuple(int(session) for session in block) for block in partition)
    covered = [session for block in blocks for session in block]
    if sorted(covered) != list(range(count)) or any(not block for block in blocks):
        raise ValueError("The phase partition must cover every session exactly once.")

    relative_parts: list[np.ndarray] = []
    covariance_parts: list[np.ndarray] = []
    nonanchors: list[int] = []
    anchors: list[int] = []
    for block in blocks:
        anchor = int(block[0])
        anchors.append(anchor)
        members = np.asarray(block[1:], dtype=int)
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


def partition_wrapped_log_likelihood(
    profile: SessionShapeProfile,
    partition: Sequence[Sequence[int]],
    *,
    hermite_order: int = 24,
    maximum_wrap_terms: int = 12,
) -> float:
    """Sum exact wrapped-normal log likelihoods over phase blocks."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    epochs = np.asarray(profile.session_epochs, dtype=float)
    total = 0.0
    for block in partition:
        indices = np.asarray(tuple(int(session) for session in block), dtype=int)
        if indices.size <= 1:
            continue
        subprofile = replace(
            profile,
            session_offsets=offsets[indices],
            session_offset_variances=variances[indices],
            session_count=int(indices.size),
            session_epochs=epochs[indices],
        )
        value, _ = collapsed_wrapped_log_likelihood(
            subprofile,
            hermite_order=max(int(hermite_order), 4),
            maximum_wrap_terms=max(int(maximum_wrap_terms), 1),
        )
        total += float(value)
    return float(total)


def partition_target_objective(
    profile: SessionShapeProfile,
    *,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    beta: float,
    normalized_values: Sequence[float],
    prior_precision: float,
    hermite_order: int,
    maximum_wrap_terms: int,
) -> float:
    """Evaluate the exact geometric bridge between two collapsed posteriors."""

    shape = np.asarray(profile.normalized_residuals, dtype=float)
    old_log = partition_wrapped_log_likelihood(
        profile,
        old_partition,
        hermite_order=hermite_order,
        maximum_wrap_terms=maximum_wrap_terms,
    )
    new_log = partition_wrapped_log_likelihood(
        profile,
        new_partition,
        hermite_order=hermite_order,
        maximum_wrap_terms=maximum_wrap_terms,
    )
    weight = max(float(prior_precision), 0.0)
    normalized = np.asarray(normalized_values, dtype=float)
    fraction = min(max(float(beta), 0.0), 1.0)
    return float(
        shape @ shape
        - 2.0 * ((1.0 - fraction) * old_log + fraction * new_log)
        + weight * (normalized @ normalized)
    )


def effective_sample_size(log_weights: Sequence[float]) -> float:
    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if values.ndim != 1 or values.size == 0 or not np.any(finite):
        return 0.0
    shifted = np.where(finite, values - np.max(values[finite]), -math.inf)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    squared = float(weights @ weights)
    return total * total / squared if total > 0.0 and squared > 0.0 else 0.0


def select_adaptive_bridge_beta(
    *,
    current_beta: float,
    bridge_log_factors: Sequence[float],
    log_base_weights: Sequence[float] | None = None,
    ess_target_fraction: float,
    minimum_step: float,
    maximum_step: float,
) -> tuple[float, float]:
    """Choose a bridge step by conditional ESS under a posterior mixture."""

    beta = min(max(float(current_beta), 0.0), 1.0)
    if beta >= 1.0:
        factors = np.asarray(bridge_log_factors, dtype=float)
        base = (
            np.zeros(factors.size, dtype=float)
            if log_base_weights is None
            else np.asarray(log_base_weights, dtype=float)
        )
        return 1.0, effective_sample_size(base)
    factors = np.asarray(bridge_log_factors, dtype=float)
    if factors.ndim != 1 or factors.size == 0 or not np.all(np.isfinite(factors)):
        raise ValueError("Bridge overlap factors must be finite and non-empty.")
    base = (
        np.zeros(factors.size, dtype=float)
        if log_base_weights is None
        else np.asarray(log_base_weights, dtype=float)
    )
    if base.shape != factors.shape or not np.any(np.isfinite(base)):
        raise ValueError("Bridge base weights must match the finite overlap factors.")
    floor = min(max(float(minimum_step), np.finfo(float).eps), 1.0 - beta)
    upper = min(beta + max(float(maximum_step), floor), 1.0)
    base_ess = effective_sample_size(base)
    target = min(max(float(ess_target_fraction), 0.0), 1.0) * base_ess

    def overlap(candidate: float) -> float:
        return effective_sample_size(base + (float(candidate) - beta) * factors)

    if overlap(upper) >= target:
        return float(upper), float(overlap(upper))
    lower = beta + floor
    if overlap(lower) <= target:
        return float(lower), float(overlap(lower))
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if overlap(midpoint) < target:
            upper = midpoint
        else:
            lower = midpoint
    return float(lower), float(overlap(lower))


def _phase_resolving_steps(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    partition: Sequence[Sequence[int]],
    config: PhasePartitionTransportConfig,
) -> np.ndarray:
    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(normalized_values, dtype=float)
    base_step = max(float(config.finite_difference_step), 1.0e-9)
    requested = max(float(config.phase_step_cycles), 1.0e-5)
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    center_phase, _, _, _ = partition_relative_phase_geometry(profile, partition)
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
        plus_phase, _, _, _ = partition_relative_phase_geometry(plus_profile, partition)
        minus_phase, _, _, _ = partition_relative_phase_geometry(minus_profile, partition)
        if plus_phase.shape != center_phase.shape or minus_phase.shape != center_phase.shape:
            raise ValueError("Finite-difference profiles changed partition dimension.")
        if center_phase.size:
            rates[index] = float(np.max(np.abs((plus_phase - minus_phase) / denominator)))
    steps = np.full(center.size, base_step, dtype=float)
    observed = rates > 0.0
    steps[observed] = np.minimum(steps[observed], requested / rates[observed])
    return np.maximum(steps, 1.0e-10)


def _absolute_step_residual_jacobian(
    residual: Callable[[np.ndarray], np.ndarray],
    values: Sequence[float],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    steps: Sequence[float],
) -> np.ndarray:
    """Differentiate a residual vector with absolute coordinate-wise steps."""

    center = np.asarray(values, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    step_values = np.asarray(steps, dtype=float)
    center_residual = np.asarray(residual(center), dtype=float)
    jacobian = np.zeros((center_residual.size, center.size), dtype=float)
    for index, step in enumerate(step_values):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + step, upper_values[index])
        minus[index] = max(minus[index] - step, lower_values[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            continue
        plus_residual = np.asarray(residual(plus), dtype=float)
        minus_residual = np.asarray(residual(minus), dtype=float)
        if plus_residual.shape != center_residual.shape or minus_residual.shape != center_residual.shape:
            raise ValueError("Finite-difference residual dimension changed.")
        jacobian[:, index] = (plus_residual - minus_residual) / denominator
    return jacobian


def _whitened_partition_phase_residual(
    profile: SessionShapeProfile,
    partition: Sequence[Sequence[int]],
    integer_vector: Sequence[int],
) -> np.ndarray:
    phase, covariance, _, _ = partition_relative_phase_geometry(profile, partition)
    integers = np.asarray(integer_vector, dtype=float)
    if integers.shape != phase.shape:
        raise ValueError("The integer vector must match the partition phase dimension.")
    if phase.size == 0:
        return np.zeros(0, dtype=float)
    return np.linalg.solve(np.linalg.cholesky(covariance), phase - integers)


def _map_partition_integer_vector(
    profile: SessionShapeProfile,
    partition: Sequence[Sequence[int]],
    *,
    maximum_lattice_nodes: int,
) -> np.ndarray:
    phase, covariance, _, _ = partition_relative_phase_geometry(profile, partition)
    if phase.size == 0:
        return np.zeros(0, dtype=float)
    result = solve_integer_least_squares(
        phase,
        covariance,
        candidate_count=1,
        max_visited_nodes=max(int(maximum_lattice_nodes), 1),
    )
    if not result.candidates:
        raise ValueError("No integer mode was available for the phase partition.")
    return np.asarray(result.candidates[0].integer_vector, dtype=float)


def _map_partition_phase_chi2(
    profile: SessionShapeProfile,
    partition: Sequence[Sequence[int]],
    *,
    maximum_lattice_nodes: int,
) -> float:
    integers = _map_partition_integer_vector(
        profile,
        partition,
        maximum_lattice_nodes=maximum_lattice_nodes,
    )
    residual = _whitened_partition_phase_residual(profile, partition, integers)
    return float(residual @ residual)


def _phase_bridge_acceptance_limit(
    *,
    previous_phase_chi2: float,
    new_phase_dimension: int,
    config: PhasePartitionTransportConfig,
) -> float:
    incremental = max(float(config.direct_bridge_phase_chi2_additive_limit), 0.0)
    per_dimension = max(
        float(config.direct_bridge_phase_chi2_per_dimension_limit),
        0.0,
    )
    return float(
        min(
            max(float(previous_phase_chi2), 0.0) + incremental,
            per_dimension * max(int(new_phase_dimension), 1),
        )
    )


def _bridge_gauge_linearization(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    config: PhasePartitionTransportConfig,
) -> PhasePartitionBridgeGaugePosterior:
    """Linearize only the new scalar phase gauge created by an island merge."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.clip(np.asarray(normalized_values, dtype=float), lower, upper)
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    old_integers = _map_partition_integer_vector(
        profile,
        old_partition,
        maximum_lattice_nodes=config.maximum_lattice_nodes,
    )
    phase_center, measurement_variance, merge_index = partition_bridge_gauge_geometry(
        profile=profile,
        old_partition=old_partition,
        new_partition=new_partition,
        old_integer_vector=old_integers,
    )
    steps = _phase_resolving_steps(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        partition=new_partition,
        config=config,
    )
    jacobian = np.zeros(center.size, dtype=float)
    for index, step in enumerate(steps):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + step, upper[index])
        minus[index] = max(minus[index] - step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            continue
        plus_profile = evaluator.evaluate_values(_physical_values(coordinates, plus))
        minus_profile = evaluator.evaluate_values(_physical_values(coordinates, minus))
        plus_center, _, _ = partition_bridge_gauge_geometry(
            profile=plus_profile,
            old_partition=old_partition,
            new_partition=new_partition,
            old_integer_vector=old_integers,
        )
        minus_center, _, _ = partition_bridge_gauge_geometry(
            profile=minus_profile,
            old_partition=old_partition,
            new_partition=new_partition,
            old_integer_vector=old_integers,
        )
        jacobian[index] = (plus_center - minus_center) / denominator
    return PhasePartitionBridgeGaugePosterior(
        merge_index=merge_index,
        phase_center=float(phase_center),
        measurement_variance=max(float(measurement_variance), np.finfo(float).eps),
        phase_jacobian=jacobian,
        old_integer_vector=old_integers,
    )


def _bridge_gauge_predictive_candidates(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhasePartitionMode,
    new_partition: Sequence[Sequence[int]],
    config: PhasePartitionTransportConfig,
) -> tuple[PhasePartitionBridgeGaugePosterior, tuple[PhasePartitionBridgeGaugeCandidate, ...]]:
    """Build a stratified discrete posterior for the one new bridge integer."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.clip(np.asarray(mode.normalized_values, dtype=float), lower, upper)
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    posterior = _bridge_gauge_linearization(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        old_partition=mode.partition,
        new_partition=new_partition,
        config=config,
    )
    scales = tuple(
        sorted(
            {
                max(float(scale), 1.0e-8)
                for scale in config.predictive_scales
                if np.isfinite(float(scale)) and float(scale) > 0.0
            }
        )
    ) or (1.0,)
    count = max(int(config.proposal_candidates_per_scale), 1)
    tail = min(
        max(float(config.bridge_gauge_tail_probability), 1.0e-9),
        0.499999,
    )
    probabilities = (
        np.asarray([0.5], dtype=float)
        if count == 1
        else np.linspace(tail, 1.0 - tail, count)
    )
    records: list[PhasePartitionBridgeGaugeCandidate] = []
    seen: set[tuple[float, int]] = set()
    jacobian = np.asarray(posterior.phase_jacobian, dtype=float)
    for scale in scales:
        continuous_covariance = float(scale) ** 2 * covariance
        predictive_variance = max(
            float(
                posterior.measurement_variance
                + jacobian @ continuous_covariance @ jacobian
            ),
            np.finfo(float).eps,
        )
        sigma = math.sqrt(predictive_variance)
        sampled = np.rint(
            posterior.phase_center
            + sigma * ndtri(np.clip(probabilities, tail, 1.0 - tail))
        ).astype(np.int64)
        nearby = np.asarray(
            [math.floor(posterior.phase_center), math.ceil(posterior.phase_center)],
            dtype=np.int64,
        )
        gain = continuous_covariance @ jacobian / predictive_variance
        for bridge_integer in np.unique(np.concatenate((sampled, nearby))):
            key = (float(scale), int(bridge_integer))
            if key in seen:
                continue
            seen.add(key)
            displacement = float(bridge_integer) - posterior.phase_center
            state = np.clip(center + gain * displacement, lower, upper)
            log_weight = -0.5 * (
                displacement * displacement / predictive_variance
                + math.log(predictive_variance)
            )
            records.append(
                PhasePartitionBridgeGaugeCandidate(
                    bridge_integer=int(bridge_integer),
                    predictive_scale=float(scale),
                    log_weight=float(log_weight),
                    normalized_values=state,
                    new_integer_vector=merged_partition_integer_vector(
                        old_partition=mode.partition,
                        new_partition=new_partition,
                        old_integer_vector=posterior.old_integer_vector,
                        bridge_integer=int(bridge_integer),
                    ),
                )
            )
    records.sort(
        key=lambda candidate: (
            -candidate.log_weight,
            abs(float(candidate.bridge_integer) - posterior.phase_center),
            candidate.predictive_scale,
            candidate.bridge_integer,
        )
    )
    return posterior, tuple(records)


def _select_bridge_homotopy_candidates(
    candidates: Sequence[PhasePartitionBridgeGaugeCandidate],
    *,
    limit: int,
) -> tuple[PhasePartitionBridgeGaugeCandidate, ...]:
    """Allocate homotopy paths across distinct bridge integers before repeats."""

    requested = max(int(limit), 0)
    if requested == 0:
        return ()
    ordered = tuple(candidates)
    selected: list[PhasePartitionBridgeGaugeCandidate] = []
    represented: set[int] = set()
    for candidate in ordered:
        if candidate.bridge_integer in represented:
            continue
        selected.append(candidate)
        represented.add(candidate.bridge_integer)
        if len(selected) >= requested:
            return tuple(selected)
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in ordered:
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        if len(selected) >= requested:
            break
    return tuple(selected)


def _island_gauge_linearization(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    partition: Sequence[Sequence[int]],
    config: PhasePartitionTransportConfig,
) -> PhaseIslandGaugePosterior:
    """Linearize all relative island gauges without choosing a merge order."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.clip(np.asarray(normalized_values, dtype=float), lower, upper)
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    old_integers = _map_partition_integer_vector(
        profile,
        partition,
        maximum_lattice_nodes=config.maximum_lattice_nodes,
    )
    phase_center, measurement_covariance = partition_island_gauge_geometry(
        profile=profile,
        partition=partition,
        old_integer_vector=old_integers,
    )
    single_partition = (
        tuple(int(session) for block in partition for session in block),
    )
    steps = _phase_resolving_steps(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        partition=single_partition,
        config=config,
    )
    jacobian = np.zeros((phase_center.size, center.size), dtype=float)
    for index, step in enumerate(steps):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + step, upper[index])
        minus[index] = max(minus[index] - step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            continue
        plus_profile = evaluator.evaluate_values(_physical_values(coordinates, plus))
        minus_profile = evaluator.evaluate_values(_physical_values(coordinates, minus))
        plus_center, _ = partition_island_gauge_geometry(
            profile=plus_profile,
            partition=partition,
            old_integer_vector=old_integers,
        )
        minus_center, _ = partition_island_gauge_geometry(
            profile=minus_profile,
            partition=partition,
            old_integer_vector=old_integers,
        )
        jacobian[:, index] = (plus_center - minus_center) / denominator
    return PhaseIslandGaugePosterior(
        phase_center=phase_center,
        measurement_covariance=measurement_covariance,
        phase_jacobian=jacobian,
        old_integer_vector=old_integers,
    )


def _island_gauge_predictive_candidates(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhasePartitionMode,
    config: PhasePartitionTransportConfig,
) -> tuple[PhaseIslandGaugePosterior, tuple[PhaseIslandGaugeCandidate, ...]]:
    """Enumerate joint integer modes of all island gauges under DD uncertainty."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.clip(np.asarray(mode.normalized_values, dtype=float), lower, upper)
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    posterior = _island_gauge_linearization(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        partition=mode.partition,
        config=config,
    )
    if posterior.phase_center.size == 0:
        return posterior, ()
    scales = tuple(
        sorted(
            {
                max(float(scale), 1.0e-8)
                for scale in config.predictive_scales
                if np.isfinite(float(scale)) and float(scale) > 0.0
            }
        )
    ) or (1.0,)
    records: list[PhaseIslandGaugeCandidate] = []
    seen: set[tuple[float, tuple[int, ...]]] = set()
    for scale in scales:
        continuous_covariance = float(scale) ** 2 * covariance
        predictive_covariance = _stable_covariance(
            posterior.measurement_covariance
            + posterior.phase_jacobian
            @ continuous_covariance
            @ posterior.phase_jacobian.T
        )
        try:
            lattice = solve_integer_least_squares(
                posterior.phase_center,
                predictive_covariance,
                candidate_count=max(int(config.proposal_candidates_per_scale), 1),
                max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
            )
            gain = (
                continuous_covariance
                @ posterior.phase_jacobian.T
                @ _stable_inverse(predictive_covariance)
            )
            sign, logdet = np.linalg.slogdet(predictive_covariance)
            if sign <= 0.0:
                logdet = math.inf
        except Exception:
            continue
        for candidate in lattice.candidates:
            gauges = tuple(int(value) for value in candidate.integer_vector)
            key = (float(scale), gauges)
            if key in seen:
                continue
            seen.add(key)
            displacement = np.asarray(gauges, dtype=float) - posterior.phase_center
            state = np.clip(center + gain @ displacement, lower, upper)
            records.append(
                PhaseIslandGaugeCandidate(
                    gauge_integers=gauges,
                    predictive_scale=float(scale),
                    log_weight=float(
                        -0.5 * (float(candidate.squared_mahalanobis) + logdet)
                    ),
                    normalized_values=state,
                    new_integer_vector=single_block_integer_vector_from_island_gauges(
                        partition=mode.partition,
                        old_integer_vector=posterior.old_integer_vector,
                        gauge_integers=gauges,
                    ),
                )
            )
    records.sort(
        key=lambda candidate: (
            -candidate.log_weight,
            candidate.predictive_scale,
            candidate.gauge_integers,
        )
    )
    return posterior, tuple(records)


def _select_island_homotopy_candidates(
    candidates: Sequence[PhaseIslandGaugeCandidate],
    *,
    limit: int,
) -> tuple[PhaseIslandGaugeCandidate, ...]:
    requested = max(int(limit), 0)
    selected: list[PhaseIslandGaugeCandidate] = []
    represented: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if candidate.gauge_integers in represented:
            continue
        selected.append(candidate)
        represented.add(candidate.gauge_integers)
        if len(selected) >= requested:
            break
    return tuple(selected)


def _conditional_partition_homotopy(
    *,
    evaluator: Any,
    coordinates: Any,
    start: Sequence[float],
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    new_integer_vector: Sequence[int],
    old_integer_vector: Sequence[int] | None = None,
    enforce_phase_acceptance: bool = True,
    config: PhasePartitionTransportConfig,
) -> np.ndarray | None:
    """Transport one transient integer proposal along the exact DD manifold."""

    lower, upper = _normalized_bounds(coordinates)
    state = np.clip(np.asarray(start, dtype=float), lower, upper)
    try:
        start_profile = evaluator.evaluate_values(_physical_values(coordinates, state))
        old_integers = (
            _map_partition_integer_vector(
                start_profile,
                old_partition,
                maximum_lattice_nodes=config.maximum_lattice_nodes,
            )
            if old_integer_vector is None
            else np.asarray(old_integer_vector, dtype=float)
        )
        old_phase, _, _, _ = partition_relative_phase_geometry(
            start_profile,
            old_partition,
        )
        if old_integers.shape != old_phase.shape:
            return None
        old_phase_residual = _whitened_partition_phase_residual(
            start_profile,
            old_partition,
            old_integers,
        )
        old_phase_chi2 = float(old_phase_residual @ old_phase_residual)
        new_integers = np.asarray(new_integer_vector, dtype=float)
        new_phase, _, _, _ = partition_relative_phase_geometry(
            start_profile,
            new_partition,
        )
        if new_integers.shape != new_phase.shape:
            return None
        shape_size = np.asarray(start_profile.normalized_residuals, dtype=float).size
    except Exception:
        return None

    betas = sorted(
        {
            min(max(float(beta), 0.0), 1.0)
            for beta in config.bridge_homotopy_betas
            if float(beta) > 0.0
        }
    )
    if not betas or betas[-1] < 1.0:
        betas.append(1.0)
    prior_size = state.size if float(config.prior_precision) > 0.0 else 0
    failure_size = shape_size + old_integers.size + new_integers.size + prior_size

    for beta in betas:
        old_weight = math.sqrt(max(1.0 - float(beta), 0.0))
        new_weight = math.sqrt(min(max(float(beta), 0.0), 1.0))

        def residual(values: np.ndarray) -> np.ndarray:
            try:
                profile = evaluator.evaluate_values(_physical_values(coordinates, values))
                parts = [
                    np.asarray(profile.normalized_residuals, dtype=float),
                    old_weight
                    * _whitened_partition_phase_residual(
                        profile,
                        old_partition,
                        old_integers,
                    ),
                    new_weight
                    * _whitened_partition_phase_residual(
                        profile,
                        new_partition,
                        new_integers,
                    ),
                ]
                if float(config.prior_precision) > 0.0:
                    parts.append(
                        math.sqrt(float(config.prior_precision))
                        * np.asarray(values, dtype=float)
                    )
                result = np.concatenate(parts)
                if result.size == failure_size and np.all(np.isfinite(result)):
                    return result
            except Exception:
                pass
            return np.full(failure_size, 1.0e15, dtype=float)

        try:
            steps = _phase_resolving_steps(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=state,
                partition=new_partition,
                config=config,
            )

            def jacobian(values: np.ndarray) -> np.ndarray:
                return _absolute_step_residual_jacobian(
                    residual,
                    values,
                    lower=lower,
                    upper=upper,
                    steps=steps,
                )

            optimization = least_squares(
                residual,
                state,
                bounds=(lower, upper),
                method="trf",
                jac=jacobian,
                x_scale="jac",
                max_nfev=max(int(config.bridge_homotopy_max_nfev), 1),
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-10,
            )
            candidate = np.clip(np.asarray(optimization.x, dtype=float), lower, upper)
            if not np.all(np.isfinite(candidate)):
                return None
            state = candidate
        except Exception:
            return None
    try:
        endpoint_profile = evaluator.evaluate_values(
            _physical_values(coordinates, state)
        )
        endpoint_residual = _whitened_partition_phase_residual(
            endpoint_profile,
            new_partition,
            new_integers,
        )
        endpoint_phase_chi2 = float(endpoint_residual @ endpoint_residual)
        phase_limit = _phase_bridge_acceptance_limit(
            previous_phase_chi2=old_phase_chi2,
            new_phase_dimension=new_integers.size,
            config=config,
        )
        if not np.isfinite(endpoint_phase_chi2) or (
            enforce_phase_acceptance and endpoint_phase_chi2 > phase_limit
        ):
            return None
    except Exception:
        return None
    return state


def _partition_jacobian(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    partition: Sequence[Sequence[int]],
    steps: np.ndarray,
) -> tuple[SessionShapeProfile, np.ndarray, np.ndarray]:
    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(normalized_values, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    phase, _, _, _ = partition_relative_phase_geometry(profile, partition)
    shape_jacobian = np.empty((shape.size, center.size), dtype=float)
    phase_jacobian = np.empty((phase.size, center.size), dtype=float)
    for index, step in enumerate(np.asarray(steps, dtype=float)):
        plus = center.copy()
        minus = center.copy()
        plus[index] = min(plus[index] + step, upper[index])
        minus[index] = max(minus[index] - step, lower[index])
        denominator = float(plus[index] - minus[index])
        if denominator <= 0.0:
            shape_jacobian[:, index] = 0.0
            phase_jacobian[:, index] = 0.0
            continue
        plus_profile = evaluator.evaluate_values(_physical_values(coordinates, plus))
        minus_profile = evaluator.evaluate_values(_physical_values(coordinates, minus))
        plus_shape = np.asarray(plus_profile.normalized_residuals, dtype=float)
        minus_shape = np.asarray(minus_profile.normalized_residuals, dtype=float)
        plus_phase, _, _, _ = partition_relative_phase_geometry(plus_profile, partition)
        minus_phase, _, _, _ = partition_relative_phase_geometry(minus_profile, partition)
        shape_jacobian[:, index] = (plus_shape - minus_shape) / denominator
        if phase.size:
            phase_jacobian[:, index] = (plus_phase - minus_phase) / denominator
    return profile, shape_jacobian, phase_jacobian


def _partition_laplace_covariance(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: np.ndarray,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    beta: float,
    steps: np.ndarray,
    config: PhasePartitionTransportConfig,
) -> np.ndarray:
    """Build the local posterior covariance from timing-model Jacobians."""

    profile, shape_jacobian, new_phase_jacobian = _partition_jacobian(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=normalized_values,
        partition=new_partition,
        steps=steps,
    )
    dimension = int(np.asarray(normalized_values).size)
    precision = shape_jacobian.T @ shape_jacobian
    precision += max(float(config.prior_precision), 0.0) * np.eye(dimension)

    new_weight = min(max(float(beta), 0.0), 1.0)
    if new_phase_jacobian.size and new_weight > 0.0:
        _, new_covariance, _, _ = partition_relative_phase_geometry(
            profile,
            new_partition,
        )
        precision += new_weight * (
            new_phase_jacobian.T
            @ _stable_inverse(new_covariance)
            @ new_phase_jacobian
        )

    old_dimension = sum(max(len(tuple(block)) - 1, 0) for block in old_partition)
    old_weight = 1.0 - new_weight
    if old_dimension > 0 and old_weight > 0.0:
        old_profile, _, old_phase_jacobian = _partition_jacobian(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=normalized_values,
            partition=old_partition,
            steps=steps,
        )
        _, old_covariance, _, _ = partition_relative_phase_geometry(
            old_profile,
            old_partition,
        )
        precision += old_weight * (
            old_phase_jacobian.T
            @ _stable_inverse(old_covariance)
            @ old_phase_jacobian
        )

    scale = max(float(config.laplace_covariance_scale), 1.0e-8)
    return _stable_covariance(scale * _stable_inverse(precision), floor=1.0e-12)


def _with_masses(modes: Sequence[PhasePartitionMode]) -> tuple[PhasePartitionMode, ...]:
    if not modes:
        return ()
    logs = np.asarray([mode.log_mass for mode in modes], dtype=float)
    finite = np.isfinite(logs)
    weights = np.zeros(len(modes), dtype=float)
    if np.any(finite):
        weights[finite] = np.exp(logs[finite] - logsumexp(logs[finite]))
    else:
        weights.fill(1.0 / len(modes))
    return tuple(replace(mode, posterior_mass=float(weight)) for mode, weight in zip(modes, weights, strict=True))


def _select_modes(
    modes: Sequence[PhasePartitionMode],
    *,
    parents: Sequence[PhasePartitionMode],
    config: PhasePartitionTransportConfig,
) -> tuple[PhasePartitionMode, ...]:
    if not modes:
        return ()
    selected: list[PhasePartitionMode] = []
    target = max(int(config.mode_beam), 1)
    ordered_parents = sorted(
        parents,
        key=lambda mode: (-float(mode.posterior_mass), mode.objective_chi2),
    )
    maximum_protected_parents = max(
        1,
        int(math.floor(target * min(max(float(config.protected_parent_beam_fraction), 0.0), 1.0))),
    )
    maximum_mass = max(
        (float(parent.posterior_mass) for parent in ordered_parents),
        default=0.0,
    )
    protected_parents = [
        parent
        for index, parent in enumerate(ordered_parents)
        if index == 0
        or float(parent.posterior_mass)
        >= maximum_mass * max(float(config.protected_parent_mass_floor), 0.0)
    ][:maximum_protected_parents]
    for parent in protected_parents:
        children = [mode for mode in modes if mode.parent_mode_id == parent.mode_id]
        children.sort(key=lambda mode: (mode.objective_chi2, -mode.log_mass))
        selected.extend(children[: max(int(config.protected_modes_per_parent), 1)])
    ordered = sorted(modes, key=lambda mode: (mode.objective_chi2, -mode.log_mass))
    seen: list[np.ndarray] = [mode.normalized_values for mode in selected]
    for mode in ordered:
        if len(selected) >= target:
            break
        if any(
            float(np.linalg.norm(np.asarray(mode.normalized_values) - value))
            < max(float(config.mode_separation), 0.0)
            for value in seen
        ):
            continue
        selected.append(mode)
        seen.append(np.asarray(mode.normalized_values))
    # A parent quota is a coverage constraint, but it must not exceed the beam.
    selected = sorted(selected, key=lambda mode: (mode.objective_chi2, -mode.log_mass))[:target]
    return _with_masses(selected)


def _select_proposal_parents(
    parents: Sequence[PhasePartitionMode],
    *,
    config: PhasePartitionTransportConfig,
) -> tuple[PhasePartitionMode, ...]:
    if not parents:
        return ()
    ordered = sorted(
        parents,
        key=lambda mode: (-float(mode.posterior_mass), mode.objective_chi2),
    )
    maximum_mass = max(float(mode.posterior_mass) for mode in ordered)
    floor = maximum_mass * max(float(config.protected_parent_mass_floor), 0.0)
    selected = [
        mode
        for index, mode in enumerate(ordered)
        if index == 0 or float(mode.posterior_mass) >= floor
    ]
    selected_ids = {mode.mode_id for mode in selected}
    exploration = sorted(
        (mode for mode in parents if mode.mode_id not in selected_ids),
        key=lambda mode: mode.objective_chi2,
    )[: max(int(config.exploration_proposal_parents), 0)]
    selected.extend(exploration)
    return tuple(selected[: max(int(config.maximum_proposal_parents), 1)])


def _select_scored_proposals(
    proposals: Sequence[tuple[np.ndarray, str, str | None]],
    objectives: Sequence[float],
    *,
    config: PhasePartitionTransportConfig,
) -> tuple[tuple[np.ndarray, str, str | None], ...]:
    if len(proposals) != len(objectives):
        raise ValueError("Proposal starts and exact objective scores must align.")
    ranked = sorted(
        zip(proposals, objectives, strict=True),
        key=lambda item: (float(item[1]), item[0][1]),
    )
    limit = max(int(config.maximum_optimized_proposals), 1)
    protected_count = max(int(config.protected_optimized_proposals_per_parent), 0)
    selected: list[tuple[np.ndarray, str, str | None]] = []
    parent_ids = tuple(dict.fromkeys(proposal[2] for proposal in proposals))
    for parent_id in parent_ids:
        family = [item for item in ranked if item[0][2] == parent_id]
        selected.extend(proposal for proposal, _ in family[:protected_count])
    selected = selected[:limit]
    selected_keys = {
        (proposal[1], proposal[2], np.ascontiguousarray(proposal[0]).tobytes())
        for proposal in selected
    }
    for proposal, _ in ranked:
        key = (proposal[1], proposal[2], np.ascontiguousarray(proposal[0]).tobytes())
        if key in selected_keys:
            continue
        selected.append(proposal)
        selected_keys.add(key)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _deduplicate_proposals(
    proposals: Sequence[tuple[np.ndarray, str, str | None]],
    *,
    limit: int,
    mode_separation: float,
) -> tuple[tuple[np.ndarray, str, str | None], ...]:
    selected: list[tuple[np.ndarray, str, str | None]] = []
    ordered = sorted(
        enumerate(proposals),
        key=lambda indexed: (
            (
                0
                if str(indexed[1][1]).startswith("inherit:")
                else 1
                if str(indexed[1][1]).startswith("bridge-homotopy:")
                else 2
            ),
            indexed[0],
        ),
    )
    for _, (state, source, parent_id) in ordered:
        is_homotopy = str(source).startswith("bridge-homotopy:")
        separation = 1.0e-10 if is_homotopy else max(float(mode_separation), 0.0)
        if any(
            float(np.linalg.norm(state - other[0])) < separation
            and parent_id == other[2]
            and (
                not is_homotopy
                or str(other[1]).startswith("bridge-homotopy:")
            )
            for other in selected
        ):
            continue
        selected.append((state, source, parent_id))
        if len(selected) >= max(int(limit), 1):
            break
    return tuple(selected)


def _predictive_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhasePartitionMode,
    partition: Sequence[Sequence[int]],
    config: PhasePartitionTransportConfig,
    nonlinear_bridge: bool = False,
) -> tuple[tuple[np.ndarray, str, str | None], ...]:
    lower, upper = _normalized_bounds(coordinates)
    center = np.asarray(mode.normalized_values, dtype=float)
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    proposals: list[tuple[np.ndarray, str, str | None]] = [
        (center.copy(), f"inherit:{mode.mode_id}", mode.mode_id)
    ]
    if adjacent_phase_partition_merge_index(mode.partition, partition) is not None:
        try:
            posterior, bridge_candidates = _bridge_gauge_predictive_candidates(
                evaluator=evaluator,
                coordinates=coordinates,
                mode=mode,
                new_partition=partition,
                config=config,
            )
            for candidate_index, candidate in enumerate(bridge_candidates):
                proposals.append(
                    (
                        candidate.normalized_values,
                        (
                            f"bridge-gauge:{mode.mode_id}:"
                            f"scale{candidate.predictive_scale:g}:"
                            f"k{candidate.bridge_integer}:candidate{candidate_index}"
                        ),
                        mode.mode_id,
                    )
                )
            if nonlinear_bridge:
                homotopy_candidates = _select_bridge_homotopy_candidates(
                    bridge_candidates,
                    limit=config.maximum_bridge_homotopy_paths,
                )
                for target_index, candidate in enumerate(homotopy_candidates):
                    endpoint = _conditional_partition_homotopy(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        start=candidate.normalized_values,
                        old_partition=mode.partition,
                        new_partition=partition,
                        old_integer_vector=posterior.old_integer_vector,
                        new_integer_vector=candidate.new_integer_vector,
                        config=config,
                    )
                    if endpoint is not None:
                        proposals.append(
                            (
                                endpoint,
                                (
                                    f"bridge-homotopy:{mode.mode_id}:"
                                    f"scale{candidate.predictive_scale:g}:"
                                    f"k{candidate.bridge_integer}:candidate{target_index}"
                                ),
                                mode.mode_id,
                            )
                        )
            return _deduplicate_proposals(
                proposals,
                limit=max(int(config.maximum_proposals_per_parent), 1),
                mode_separation=config.mode_separation,
            )
        except Exception:
            return tuple(proposals)
    try:
        steps = _phase_resolving_steps(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=center,
            partition=partition,
            config=config,
        )
        profile, _, phase_jacobian = _partition_jacobian(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=center,
            partition=partition,
            steps=steps,
        )
        phase_center, phase_covariance, _, _ = partition_relative_phase_geometry(
            profile,
            partition,
        )
    except Exception:
        return tuple(proposals)
    if phase_center.size == 0:
        return tuple(proposals)

    integer_targets: dict[tuple[int, ...], tuple[int, int, float, str]] = {}

    def retain_integer_targets(lattice: Any, source: str, source_rank: int) -> None:
        for candidate_rank, candidate in enumerate(lattice.candidates):
            vector = tuple(int(value) for value in candidate.integer_vector)
            score = float(candidate.squared_mahalanobis)
            ranking = (int(source_rank), int(candidate_rank), score, source)
            previous = integer_targets.get(vector)
            if previous is None or ranking[:3] < previous[:3]:
                integer_targets[vector] = ranking

    try:
        local_lattice = solve_integer_least_squares(
            phase_center,
            phase_covariance,
            candidate_count=max(int(config.proposal_candidates_per_scale), 1),
            max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
        )
        retain_integer_targets(local_lattice, "local", 0)
    except Exception:
        pass

    for scale_rank, raw_scale in enumerate(config.predictive_scales, start=1):
        scale = max(float(raw_scale), 1.0e-6)
        continuous_covariance = scale * scale * covariance
        predictive_covariance = _stable_covariance(
            phase_covariance + phase_jacobian @ continuous_covariance @ phase_jacobian.T
        )
        try:
            lattice = solve_integer_least_squares(
                phase_center,
                predictive_covariance,
                candidate_count=max(int(config.proposal_candidates_per_scale), 1),
                max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
            )
            gain = (
                continuous_covariance
                @ phase_jacobian.T
                @ _stable_inverse(predictive_covariance)
            )
        except Exception:
            continue
        retain_integer_targets(lattice, f"scale{scale:g}", scale_rank)
        for candidate_index, candidate in enumerate(lattice.candidates):
            displacement = np.asarray(candidate.integer_vector, dtype=float) - phase_center
            state = np.clip(center + gain @ displacement, lower, upper)
            proposals.append(
                (
                    state,
                    f"predictive:{mode.mode_id}:scale{scale:g}:candidate{candidate_index}",
                    mode.mode_id,
                )
            )
    if nonlinear_bridge:
        ranked_targets = sorted(
            integer_targets.items(),
            key=lambda item: (*item[1][:3], item[0]),
        )[: max(int(config.maximum_bridge_homotopy_paths), 0)]
        for target_index, (integer_vector, (_, _, _, target_source)) in enumerate(
            ranked_targets
        ):
            endpoint = _conditional_partition_homotopy(
                evaluator=evaluator,
                coordinates=coordinates,
                start=center,
                old_partition=mode.partition,
                new_partition=partition,
                new_integer_vector=integer_vector,
                config=config,
            )
            if endpoint is not None:
                proposals.append(
                    (
                        endpoint,
                        (
                            f"bridge-homotopy:{mode.mode_id}:{target_source}:"
                            f"candidate{target_index}"
                        ),
                        mode.mode_id,
                    )
                )
    return _deduplicate_proposals(
        proposals,
        limit=max(int(config.maximum_proposals_per_parent), 1),
        mode_separation=config.mode_separation,
    )


def _mode_bridge_log_factors(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhasePartitionMode,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    config: PhasePartitionTransportConfig,
    seed: int,
) -> np.ndarray:
    """Sample the new/old collapsed likelihood ratio inside one mode."""

    lower, upper = _normalized_bounds(coordinates)
    center = np.clip(np.asarray(mode.normalized_values, dtype=float), lower, upper)
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    requested = max(int(config.bridge_overlap_samples), 1)
    states = [center]
    if requested > 1:
        exponent = int(math.ceil(math.log2(requested - 1)))
        unit = qmc.Sobol(
            d=center.size,
            scramble=True,
            seed=int(seed),
        ).random_base2(exponent)[: requested - 1]
        normal = ndtri(np.clip(unit, 1.0e-9, 1.0 - 1.0e-9))
        try:
            chol = np.linalg.cholesky(covariance)
            states.extend(
                np.clip(center + chol @ sample, lower, upper)
                for sample in normal
            )
        except np.linalg.LinAlgError:
            pass
    factors: list[float] = []
    for state in states:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, state))
            old_log = partition_wrapped_log_likelihood(
                profile,
                old_partition,
                hermite_order=config.collapsed_hermite_order,
                maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
            )
            new_log = partition_wrapped_log_likelihood(
                profile,
                new_partition,
                hermite_order=config.collapsed_hermite_order,
                maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
            )
            difference = float(new_log - old_log)
            if np.isfinite(difference):
                factors.append(difference)
        except Exception:
            continue
    if not factors:
        raise ValueError("No finite bridge overlap samples were available.")
    return np.asarray(factors, dtype=float)


def _bridge_mixture_samples(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[PhasePartitionMode],
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    level_index: int,
    step_index: int,
    config: PhasePartitionTransportConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return likelihood-ratio samples and their current mixture weights."""

    masses = np.asarray([max(float(mode.posterior_mass), 0.0) for mode in modes])
    if not np.any(masses > 0.0):
        masses = np.ones(len(modes), dtype=float)
    masses /= float(np.sum(masses))
    factors_by_mode: list[np.ndarray] = []
    base_weights: list[np.ndarray] = []
    for mode_index, mode in enumerate(modes):
        if masses[mode_index] <= 0.0:
            continue
        factors = _mode_bridge_log_factors(
            evaluator=evaluator,
            coordinates=coordinates,
            mode=mode,
            old_partition=old_partition,
            new_partition=new_partition,
            config=config,
            seed=(
                int(config.random_seed)
                + 15485863 * (level_index + 1)
                + 32452843 * (step_index + 1)
                + mode_index
            ),
        )
        factors_by_mode.append(factors)
        base_weights.append(
            np.full(
                factors.size,
                math.log(float(masses[mode_index])) - math.log(float(factors.size)),
                dtype=float,
            )
        )
    if not factors_by_mode:
        raise ValueError("No posterior mode supplied finite bridge overlap samples.")
    return np.concatenate(factors_by_mode), np.concatenate(base_weights)


def score_adjacent_phase_partition_merges(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[PhasePartitionMode],
    partition: Sequence[Sequence[int]],
    level_index: int,
    config: PhasePartitionTransportConfig,
) -> tuple[PhasePartitionMergeScore, ...]:
    """Score every adjacent island merge under the current posterior mixture."""

    old_partition = tuple(tuple(int(item) for item in block) for block in partition)
    scores: list[PhasePartitionMergeScore] = []
    for merge_index, candidate in enumerate(adjacent_phase_partition_merges(old_partition)):
        try:
            factors, base = _bridge_mixture_samples(
                evaluator=evaluator,
                coordinates=coordinates,
                modes=modes,
                old_partition=old_partition,
                new_partition=candidate,
                level_index=level_index,
                step_index=merge_index,
                config=config,
            )
        except Exception:
            continue
        base_ess = effective_sample_size(base)
        full_ess = effective_sample_size(base + factors)
        relative = min(full_ess / base_ess, 1.0) if base_ess > 0.0 else 0.0
        evidence = float(logsumexp(base + factors) - logsumexp(base))
        initial_beta, _ = select_adaptive_bridge_beta(
            current_beta=0.0,
            bridge_log_factors=factors,
            log_base_weights=base,
            ess_target_fraction=config.bridge_ess_target_fraction,
            minimum_step=config.bridge_minimum_beta_step,
            maximum_step=1.0,
        )
        normalized_base = np.exp(base - logsumexp(base))
        factor_mean = float(normalized_base @ factors)
        factor_standard_deviation = math.sqrt(
            max(
                float(normalized_base @ np.square(factors - factor_mean)),
                0.0,
            )
        )
        scores.append(
            PhasePartitionMergeScore(
                merge_index=merge_index,
                partition=candidate,
                effective_sample_size=full_ess,
                relative_effective_sample_size=relative,
                log_evidence_increment=evidence,
                initial_adaptive_beta=float(initial_beta),
                log_factor_standard_deviation=float(factor_standard_deviation),
            )
        )
    return tuple(scores)


def score_merge_homotopy_endpoints(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[PhasePartitionMode],
    old_partition: Sequence[Sequence[int]],
    scores: Sequence[PhasePartitionMergeScore],
    config: PhasePartitionTransportConfig,
) -> tuple[PhasePartitionMergeScore, ...]:
    """Test each merge against a nonlinear endpoint from the leading mode."""

    if not modes:
        return tuple(scores)
    parent = min(
        modes,
        key=lambda mode: (
            mode.objective_chi2,
            -float(mode.posterior_mass),
        ),
    )
    previous_shape = float(parent.shape_chi2)
    shape_limit = max(
        previous_shape * max(float(config.direct_bridge_shape_ratio_limit), 1.0),
        previous_shape + max(float(config.direct_bridge_shape_additive_limit), 0.0),
    )
    def evaluate_score(score: PhasePartitionMergeScore) -> PhasePartitionMergeScore:
        try:
            acquisition_config = replace(
                config,
                bridge_homotopy_max_nfev=max(
                    int(config.merge_acquisition_homotopy_max_nfev),
                    1,
                ),
            )
            posterior, bridge_candidates = _bridge_gauge_predictive_candidates(
                evaluator=evaluator,
                coordinates=coordinates,
                mode=parent,
                new_partition=score.partition,
                config=acquisition_config,
            )
            acquisition_candidates = _select_bridge_homotopy_candidates(
                bridge_candidates,
                limit=config.maximum_merge_acquisition_homotopy_paths,
            )
            start_profile = evaluator.evaluate_values(parent.parameter_values)
            old_residual = _whitened_partition_phase_residual(
                start_profile,
                old_partition,
                posterior.old_integer_vector,
            )
            old_phase_chi2 = float(old_residual @ old_residual)
            new_phase_dimension = sum(
                max(len(tuple(block)) - 1, 0)
                for block in score.partition
            )
            phase_limit = _phase_bridge_acceptance_limit(
                previous_phase_chi2=old_phase_chi2,
                new_phase_dimension=new_phase_dimension,
                config=config,
            )
            endpoints: list[tuple[float, float, float, int]] = []
            for candidate in acquisition_candidates:
                endpoint = _conditional_partition_homotopy(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    start=candidate.normalized_values,
                    old_partition=old_partition,
                    new_partition=score.partition,
                    old_integer_vector=posterior.old_integer_vector,
                    new_integer_vector=candidate.new_integer_vector,
                    enforce_phase_acceptance=False,
                    config=acquisition_config,
                )
                if endpoint is None:
                    continue
                profile = evaluator.evaluate_values(
                    _physical_values(coordinates, endpoint)
                )
                shape = np.asarray(profile.normalized_residuals, dtype=float)
                shape_chi2 = float(shape @ shape)
                phase_residual = _whitened_partition_phase_residual(
                    profile,
                    score.partition,
                    candidate.new_integer_vector,
                )
                phase_chi2 = float(phase_residual @ phase_residual)
                objective_chi2 = partition_target_objective(
                    profile,
                    old_partition=old_partition,
                    new_partition=score.partition,
                    beta=1.0,
                    normalized_values=endpoint,
                    prior_precision=config.prior_precision,
                    hermite_order=config.collapsed_hermite_order,
                    maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
                )
                endpoints.append(
                    (
                        float(objective_chi2),
                        shape_chi2,
                        phase_chi2,
                        int(candidate.bridge_integer),
                    )
                )
            if not endpoints:
                raise ValueError("The conditional bridge endpoint was unavailable.")
            objective_chi2, shape_chi2, phase_chi2, bridge_integer = min(
                endpoints,
                key=lambda endpoint: (
                    endpoint[2] / max(phase_limit, np.finfo(float).eps),
                    endpoint[0],
                    endpoint[1],
                ),
            )
            return replace(
                score,
                homotopy_attempted=True,
                homotopy_transportable=bool(
                    np.isfinite(objective_chi2)
                    and shape_chi2 <= shape_limit
                    and phase_chi2 <= phase_limit
                ),
                homotopy_shape_chi2=shape_chi2,
                homotopy_phase_chi2=phase_chi2,
                homotopy_phase_chi2_limit=phase_limit,
                homotopy_bridge_integer=bridge_integer,
                homotopy_objective_chi2=float(objective_chi2),
            )
        except Exception:
            return replace(score, homotopy_attempted=True)

    workers = max(int(config.merge_acquisition_workers), 1)
    if workers == 1 or len(scores) <= 1:
        return tuple(evaluate_score(score) for score in scores)
    with ThreadPoolExecutor(max_workers=min(workers, len(scores))) as executor:
        return tuple(executor.map(evaluate_score, scores))


def select_next_phase_partition_merge(
    scores: Sequence[PhasePartitionMergeScore],
    *,
    ess_target_fraction: float,
) -> PhasePartitionMergeScore:
    """Choose the most informative merge that remains directly transportable."""

    candidates = tuple(scores)
    if not candidates:
        raise ValueError("At least one finite phase-island merge score is required.")
    transported = [score for score in candidates if score.homotopy_transportable]
    if transported:
        return min(
            transported,
            key=lambda score: (
                score.homotopy_objective_chi2,
                score.homotopy_shape_chi2,
                -score.relative_effective_sample_size,
                score.merge_index,
            ),
        )
    diagnosed = [
        score
        for score in candidates
        if score.homotopy_attempted
        and np.isfinite(score.homotopy_phase_chi2)
        and np.isfinite(score.homotopy_phase_chi2_limit)
        and score.homotopy_phase_chi2_limit > 0.0
    ]
    if diagnosed:
        return min(
            diagnosed,
            key=lambda score: (
                score.homotopy_phase_chi2 / score.homotopy_phase_chi2_limit,
                score.homotopy_objective_chi2,
                score.homotopy_shape_chi2,
                score.merge_index,
            ),
        )
    target = min(max(float(ess_target_fraction), 0.0), 1.0)
    feasible = [
        score
        for score in candidates
        if score.relative_effective_sample_size >= target
    ]
    if feasible:
        return min(
            feasible,
            key=lambda score: (
                score.relative_effective_sample_size,
                -score.log_evidence_increment,
                score.merge_index,
            ),
        )
    return max(
        candidates,
        key=lambda score: (
            score.initial_adaptive_beta,
            score.relative_effective_sample_size,
            -score.log_factor_standard_deviation,
            -score.merge_index,
        ),
    )


def _select_mixture_bridge_beta(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[PhasePartitionMode],
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    current_beta: float,
    level_index: int,
    step_index: int,
    config: PhasePartitionTransportConfig,
) -> tuple[float, float, float]:
    factors, base = _bridge_mixture_samples(
        evaluator=evaluator,
        coordinates=coordinates,
        modes=modes,
        old_partition=old_partition,
        new_partition=new_partition,
        level_index=level_index,
        step_index=step_index,
        config=config,
    )
    candidate, overlap = select_adaptive_bridge_beta(
        current_beta=current_beta,
        bridge_log_factors=factors,
        log_base_weights=base,
        ess_target_fraction=config.bridge_ess_target_fraction,
        minimum_step=config.bridge_minimum_beta_step,
        maximum_step=config.bridge_maximum_beta_step,
    )
    base_ess = effective_sample_size(base)
    relative = min(overlap / base_ess, 1.0) if base_ess > 0.0 else 0.0
    return candidate, overlap, relative


def _evaluate_partition_control_mode(
    *,
    evaluator: Any,
    coordinates: Any,
    parent: PhasePartitionMode,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    beta: float,
    level_index: int,
    control_index: int,
    config: PhasePartitionTransportConfig,
) -> PhasePartitionMode | None:
    """Score an unchanged parent so a bridge step cannot erase its basin."""

    try:
        normalized = np.asarray(parent.normalized_values, dtype=float).copy()
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        new_log = partition_wrapped_log_likelihood(
            profile,
            new_partition,
            hermite_order=config.collapsed_hermite_order,
            maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
        )
        objective = partition_target_objective(
            profile,
            old_partition=old_partition,
            new_partition=new_partition,
            beta=beta,
            normalized_values=normalized,
            prior_precision=config.prior_precision,
            hermite_order=config.collapsed_hermite_order,
            maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
        )
        covariance = _stable_covariance(np.asarray(parent.covariance, dtype=float))
        sign, logdet = np.linalg.slogdet(covariance)
        return PhasePartitionMode(
            mode_id=(
                f"partition_l{level_index:02d}_b{beta:.6g}_"
                f"control{control_index:04d}"
            ),
            parent_mode_id=parent.mode_id,
            proposal_source=f"control:{parent.mode_id}",
            level_index=level_index,
            partition=tuple(tuple(int(item) for item in block) for block in new_partition),
            parameter_values=np.asarray(parent.parameter_values, dtype=float).copy(),
            normalized_values=normalized,
            covariance=covariance,
            shape_chi2=float(shape @ shape),
            collapsed_phase_deviance=float(-2.0 * new_log),
            prior_chi2=max(float(config.prior_precision), 0.0)
            * float(normalized @ normalized),
            objective_chi2=float(objective),
            log_mass=float(
                -0.5 * objective + (0.5 * logdet if sign > 0.0 else -math.inf)
            ),
            posterior_mass=0.0,
            nfev=0,
            phase_beta=float(beta),
            success=True,
        )
    except Exception:
        return None


def _optimize_mode(
    *,
    evaluator: Any,
    coordinates: Any,
    start: np.ndarray,
    parent_mode_id: str | None,
    source: str,
    old_partition: Sequence[Sequence[int]],
    new_partition: Sequence[Sequence[int]],
    beta: float,
    level_index: int,
    mode_index: int,
    config: PhasePartitionTransportConfig,
) -> PhasePartitionMode | None:
    lower, upper = _normalized_bounds(coordinates)
    state = np.clip(np.asarray(start, dtype=float), lower, upper)
    try:
        steps = _phase_resolving_steps(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=state,
            partition=new_partition,
            config=config,
        )
    except Exception:
        return None
    cache: dict[bytes, float] = {}

    def objective(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        key = np.ascontiguousarray(values, dtype=np.float64).tobytes()
        if key in cache:
            return cache[key]
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            value = partition_target_objective(
                profile,
                old_partition=old_partition,
                new_partition=new_partition,
                beta=beta,
                normalized_values=values,
                prior_precision=config.prior_precision,
                hermite_order=config.collapsed_hermite_order,
                maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
            )
            result = float(value) if np.isfinite(value) else 1.0e30
        except Exception:
            result = 1.0e30
        cache[key] = result
        return result

    def gradient(values: np.ndarray) -> np.ndarray:
        center = np.asarray(values, dtype=float)
        result = np.zeros_like(center)
        for index, step in enumerate(steps):
            plus = center.copy()
            minus = center.copy()
            plus[index] = min(plus[index] + step, upper[index])
            minus[index] = max(minus[index] - step, lower[index])
            denominator = float(plus[index] - minus[index])
            if denominator > 0.0:
                result[index] = (objective(plus) - objective(minus)) / denominator
        return result

    try:
        use_conditional_alias = bool(
            config.conditional_alias_refinement and float(beta) >= 1.0 - 1.0e-12
        )
        if use_conditional_alias:
            start_profile = evaluator.evaluate_values(
                _physical_values(coordinates, state)
            )
            start_phase, _, _, _ = partition_relative_phase_geometry(
                start_profile,
                new_partition,
            )
            local_integers = np.rint(start_phase)
            shape_size = np.asarray(
                start_profile.normalized_residuals,
                dtype=float,
            ).size
            prior_size = state.size if float(config.prior_precision) > 0.0 else 0
            failure_size = shape_size + local_integers.size + prior_size

            def conditional_residual(values: np.ndarray) -> np.ndarray:
                try:
                    local_profile = evaluator.evaluate_values(
                        _physical_values(coordinates, values)
                    )
                    local_shape = np.asarray(
                        local_profile.normalized_residuals,
                        dtype=float,
                    )
                    local_phase, local_covariance, _, _ = (
                        partition_relative_phase_geometry(
                            local_profile,
                            new_partition,
                        )
                    )
                    phase_residual = np.linalg.solve(
                        np.linalg.cholesky(local_covariance),
                        local_phase - local_integers,
                    )
                    parts = [local_shape, phase_residual]
                    if float(config.prior_precision) > 0.0:
                        parts.append(
                            math.sqrt(float(config.prior_precision))
                            * np.asarray(values, dtype=float)
                        )
                    result = np.concatenate(parts)
                    if result.size == failure_size and np.all(np.isfinite(result)):
                        return result
                except Exception:
                    pass
                return np.full(failure_size, 1.0e15, dtype=float)

            def conditional_jacobian(values: np.ndarray) -> np.ndarray:
                return _absolute_step_residual_jacobian(
                    conditional_residual,
                    values,
                    lower=lower,
                    upper=upper,
                    steps=steps,
                )

            optimization = least_squares(
                conditional_residual,
                state,
                bounds=(lower, upper),
                method="trf",
                jac=conditional_jacobian,
                x_scale="jac",
                max_nfev=max(int(config.optimizer_maxiter), 1),
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-10,
            )
        else:
            optimization = minimize(
                objective,
                state,
                method="L-BFGS-B",
                jac=gradient,
                bounds=tuple(zip(lower, upper, strict=True)),
                options={
                    "maxiter": max(int(config.optimizer_maxiter), 1),
                    "ftol": 1.0e-11,
                    "gtol": 1.0e-8,
                    "maxls": 32,
                },
            )
        normalized = np.clip(np.asarray(optimization.x, dtype=float), lower, upper)
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        new_log = partition_wrapped_log_likelihood(
            profile,
            new_partition,
            hermite_order=config.collapsed_hermite_order,
            maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
        )
        objective_value = objective(normalized)
        prior_chi2 = max(float(config.prior_precision), 0.0) * float(normalized @ normalized)
        try:
            covariance = _partition_laplace_covariance(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=normalized,
                old_partition=old_partition,
                new_partition=new_partition,
                beta=beta,
                steps=steps,
                config=config,
            )
        except Exception:
            if hasattr(optimization, "hess_inv"):
                covariance = _stable_covariance(
                    np.asarray(
                        optimization.hess_inv.todense()
                        if hasattr(optimization.hess_inv, "todense")
                        else optimization.hess_inv,
                        dtype=float,
                    ),
                    floor=1.0e-8,
                )
            elif hasattr(optimization, "jac"):
                jacobian = np.asarray(optimization.jac, dtype=float)
                covariance = _stable_covariance(
                    _stable_inverse(jacobian.T @ jacobian),
                    floor=1.0e-8,
                )
            else:
                covariance = np.eye(state.size, dtype=float)
        sign, logdet = np.linalg.slogdet(covariance)
        log_mass = float(
            -0.5 * objective_value + (0.5 * logdet if sign > 0.0 else -math.inf)
        )
        return PhasePartitionMode(
            mode_id=f"partition_l{level_index:02d}_b{beta:.6g}_{mode_index:04d}",
            parent_mode_id=parent_mode_id,
            proposal_source=source,
            level_index=level_index,
            partition=tuple(tuple(int(item) for item in block) for block in new_partition),
            parameter_values=_physical_values(coordinates, normalized),
            normalized_values=normalized,
            covariance=covariance,
            shape_chi2=float(shape @ shape),
            collapsed_phase_deviance=float(-2.0 * new_log),
            prior_chi2=float(prior_chi2),
            objective_chi2=float(objective_value),
            log_mass=log_mass,
            posterior_mass=0.0,
            nfev=int(getattr(optimization, "nfev", 0)),
            phase_beta=float(beta),
            success=bool(optimization.success),
        )
    except Exception:
        return None


def _fixed_path_fit(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhasePartitionMode,
    integer_offsets: Sequence[int],
    config: PhasePartitionTransportConfig,
    path_index: int,
) -> PhasePartitionPathMode | None:
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(np.asarray(mode.normalized_values, dtype=float), lower, upper)
    try:
        start_profile = evaluator.evaluate_values(_physical_values(coordinates, start))
        relative = np.asarray(start_profile.session_offsets[1:] - start_profile.session_offsets[0], dtype=float)
        covariance = np.diag(np.asarray(start_profile.session_offset_variances[1:], dtype=float))
        covariance += float(start_profile.session_offset_variances[0])
        chol = np.linalg.cholesky(_stable_covariance(covariance))
        path = np.asarray(integer_offsets, dtype=float)
        if path.shape != relative.shape:
            return None
        base_step = max(float(config.finite_difference_step), 1.0e-9)
        failure_size = np.asarray(start_profile.normalized_residuals).size + path.size
    except Exception:
        return None

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            rel = np.asarray(profile.session_offsets[1:] - profile.session_offsets[0], dtype=float)
            cov = np.diag(np.asarray(profile.session_offset_variances[1:], dtype=float))
            cov += float(profile.session_offset_variances[0])
            phase = np.linalg.solve(np.linalg.cholesky(_stable_covariance(cov)), rel - path)
            result = np.concatenate((np.asarray(profile.normalized_residuals, dtype=float), phase))
            return result if result.size == failure_size else np.full(failure_size, 1.0e30)
        except Exception:
            return np.full(failure_size, 1.0e30)

    def jacobian(values: np.ndarray) -> np.ndarray:
        center = np.asarray(values, dtype=float)
        base = residual(center)
        result = np.empty((base.size, center.size), dtype=float)
        for index in range(center.size):
            plus = center.copy()
            minus = center.copy()
            plus[index] = min(plus[index] + base_step, upper[index])
            minus[index] = max(minus[index] - base_step, lower[index])
            denominator = float(plus[index] - minus[index])
            result[:, index] = (
                (residual(plus) - residual(minus)) / denominator
                if denominator > 0.0
                else 0.0
            )
        return result

    try:
        optimization = least_squares(
            residual,
            start,
            jac=jacobian,
            bounds=(lower, upper),
            max_nfev=max(int(config.path_fit_max_nfev), 1),
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-10,
        )
        normalized = np.asarray(optimization.x, dtype=float)
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        rel = np.asarray(profile.session_offsets[1:] - profile.session_offsets[0], dtype=float)
        cov = np.diag(np.asarray(profile.session_offset_variances[1:], dtype=float))
        cov += float(profile.session_offset_variances[0])
        delta = rel - path
        phase_chi2 = float(delta @ np.linalg.solve(_stable_covariance(cov), delta))
        covariance_fit = _stable_covariance(
            _stable_inverse(np.asarray(optimization.jac).T @ np.asarray(optimization.jac)),
            floor=1.0e-8,
        )
        objective = float(shape @ shape) + phase_chi2
        return PhasePartitionPathMode(
            mode_id=f"path_{mode.mode_id}_{path_index:04d}",
            parent_mode_id=mode.mode_id,
            proposal_source="final-conditional-path",
            partition=mode.partition,
            integer_offsets=tuple(int(value) for value in path.tolist()),
            wraps=integer_session_offsets_to_gap_wraps(path),
            parameter_values=_physical_values(coordinates, normalized),
            normalized_values=normalized,
            covariance=covariance_fit,
            shape_chi2=float(shape @ shape),
            phase_chi2=phase_chi2,
            prior_chi2=0.0,
            objective_chi2=objective,
            nfev=int(getattr(optimization, "nfev", 0)),
            success=bool(optimization.success),
        )
    except Exception:
        return None


def recover_final_integer_paths(
    *,
    evaluator: Any,
    coordinates: Any,
    modes: Sequence[PhasePartitionMode],
    config: PhasePartitionTransportConfig,
) -> tuple[PhasePartitionPathMode, ...]:
    """Recover and refit complete paths only after the collapsed search ends."""

    if int(config.path_candidates) <= 0:
        return ()
    candidates: list[PhasePartitionPathMode] = []
    seen: set[tuple[int, ...]] = set()
    ranked_modes = sorted(modes, key=lambda mode: (mode.objective_chi2, -mode.log_mass))
    for mode in ranked_modes:
        profile = evaluator.evaluate_values(np.asarray(mode.parameter_values, dtype=float))
        paths = conditional_integer_paths(
            profile,
            candidate_count=max(int(config.path_candidates), 1),
            hermite_order=config.collapsed_hermite_order,
            maximum_wrap_terms=config.collapsed_maximum_wrap_terms,
        )
        for candidate in paths:
            if candidate.integer_offsets in seen:
                continue
            seen.add(candidate.integer_offsets)
            fitted = _fixed_path_fit(
                evaluator=evaluator,
                coordinates=coordinates,
                mode=mode,
                integer_offsets=candidate.integer_offsets,
                config=config,
                path_index=len(candidates),
            )
            if fitted is not None:
                candidates.append(fitted)
    return tuple(sorted(candidates, key=lambda mode: mode.objective_chi2))


def run_bayesian_phase_island_global_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    initial_modes: Sequence[PhasePartitionMode],
    initial_level_index: int = 0,
    config: PhasePartitionTransportConfig | None = None,
    stage_callback: Callable[
        [PhasePartitionStageSummary, Sequence[PhasePartitionMode]], None
    ]
    | None = None,
) -> PhasePartitionTransportResult:
    """Close all trusted phase islands jointly, without choosing a merge order."""

    active = config or PhasePartitionTransportConfig()
    if not initial_modes:
        raise ValueError("Global island transport requires checkpoint modes.")
    current = _with_masses(
        tuple(
            replace(
                mode,
                parameter_values=np.asarray(mode.parameter_values, dtype=float).copy(),
                normalized_values=np.asarray(mode.normalized_values, dtype=float).copy(),
                covariance=_stable_covariance(np.asarray(mode.covariance, dtype=float)),
            )
            for mode in initial_modes
        )
    )
    old_partition = current[0].partition
    if any(mode.partition != old_partition for mode in current):
        raise ValueError("Global island modes must share one checkpoint partition.")
    sessions = tuple(int(session) for block in old_partition for session in block)
    if sorted(sessions) != list(range(len(sessions))):
        raise ValueError("The checkpoint partition must cover every session exactly once.")
    new_partition = (sessions,)
    if len(old_partition) == 1:
        paths = recover_final_integer_paths(
            evaluator=evaluator,
            coordinates=coordinates,
            modes=current,
            config=active,
        )
        return PhasePartitionTransportResult(
            session_count=len(sessions),
            final_modes=current,
            path_modes=paths,
            stage_history=(),
            partition_history=(old_partition,),
            reached_all_sessions=True,
            failure_class=None,
        )

    level_index = max(int(initial_level_index), 0) + 1
    proposal_parents = _select_proposal_parents(current, config=active)
    fitted: list[PhasePartitionMode] = []
    proposal_count = 0
    for parent in proposal_parents:
        try:
            posterior, candidates = _island_gauge_predictive_candidates(
                evaluator=evaluator,
                coordinates=coordinates,
                mode=parent,
                config=active,
            )
        except Exception:
            continue
        selected_candidates = _select_island_homotopy_candidates(
            candidates,
            limit=active.maximum_bridge_homotopy_paths,
        )
        proposal_count += len(selected_candidates)
        for candidate_index, candidate in enumerate(selected_candidates):
            endpoint = _conditional_partition_homotopy(
                evaluator=evaluator,
                coordinates=coordinates,
                # beta=0 is the trusted island posterior. The predictive
                # update proposes only the joint integer mode; starting the
                # exact bridge from that potentially remote linear update
                # breaks the homotopy before it begins.
                start=parent.normalized_values,
                old_partition=old_partition,
                new_partition=new_partition,
                old_integer_vector=posterior.old_integer_vector,
                new_integer_vector=candidate.new_integer_vector,
                enforce_phase_acceptance=False,
                config=active,
            )
            if endpoint is None:
                continue
            optimized = _optimize_mode(
                evaluator=evaluator,
                coordinates=coordinates,
                start=endpoint,
                parent_mode_id=parent.mode_id,
                source=(
                    f"global-island-homotopy:{parent.mode_id}:"
                    f"scale{candidate.predictive_scale:g}:candidate{candidate_index}"
                ),
                old_partition=old_partition,
                new_partition=new_partition,
                beta=1.0,
                level_index=level_index,
                mode_index=len(fitted),
                config=active,
            )
            if optimized is not None:
                fitted.append(optimized)

    selected = _select_modes(fitted, parents=current, config=active)
    previous_shape = min((mode.shape_chi2 for mode in current), default=math.inf)
    shape_limit = max(
        previous_shape * max(float(active.direct_bridge_shape_ratio_limit), 1.0),
        previous_shape + max(float(active.direct_bridge_shape_additive_limit), 0.0),
    )
    previous_phase_values: list[float] = []
    for mode in current:
        try:
            previous_phase_values.append(
                _map_partition_phase_chi2(
                    evaluator.evaluate_values(mode.parameter_values),
                    old_partition,
                    maximum_lattice_nodes=active.maximum_lattice_nodes,
                )
            )
        except Exception:
            continue
    previous_phase = min(previous_phase_values, default=math.inf)
    phase_limit = _phase_bridge_acceptance_limit(
        previous_phase_chi2=previous_phase,
        new_phase_dimension=len(sessions) - 1,
        config=active,
    )
    phase_by_mode: list[tuple[PhasePartitionMode, float]] = []
    for mode in selected:
        try:
            phase_chi2 = _map_partition_phase_chi2(
                evaluator.evaluate_values(mode.parameter_values),
                new_partition,
                maximum_lattice_nodes=active.maximum_lattice_nodes,
            )
        except Exception:
            phase_chi2 = math.inf
        phase_by_mode.append((mode, float(phase_chi2)))
    eligible = tuple(
        mode
        for mode, phase_chi2 in phase_by_mode
        if mode.shape_chi2 <= shape_limit and phase_chi2 <= phase_limit
    )
    final_modes = _with_masses(
        sorted(eligible, key=lambda mode: (mode.objective_chi2, -mode.log_mass))
    ) if eligible else selected
    summary = PhasePartitionStageSummary(
        level_index=level_index,
        beta=1.0,
        old_partition=old_partition,
        new_partition=new_partition,
        input_mode_count=len(current),
        proposal_count=proposal_count,
        retained_count=len(final_modes),
        unique_mode_count=len(fitted),
        best_shape_chi2=min((mode.shape_chi2 for mode in final_modes), default=math.inf),
        best_collapsed_phase_deviance=min(
            (mode.collapsed_phase_deviance for mode in final_modes),
            default=math.inf,
        ),
        best_objective_chi2=min(
            (mode.objective_chi2 for mode in final_modes),
            default=math.inf,
        ),
        ancestor_coverage=float(
            len({mode.parent_mode_id for mode in final_modes}) / max(len(current), 1)
        ),
        candidate_merge_count=0,
        selected_merge_index=-1,
        direct_bridge_attempted=True,
        direct_bridge_accepted=bool(eligible),
        direct_bridge_best_shape_chi2=min(
            (mode.shape_chi2 for mode in selected),
            default=math.inf,
        ),
        direct_bridge_best_phase_chi2=min(
            (phase for _, phase in phase_by_mode),
            default=math.inf,
        ),
        direct_bridge_phase_chi2_limit=phase_limit,
        direct_bridge_best_objective_chi2=min(
            (mode.objective_chi2 for mode in selected),
            default=math.inf,
        ),
        direct_bridge_proposal_count=proposal_count,
        direct_bridge_optimized_proposal_count=len(fitted),
        optimized_proposal_count=len(fitted),
    )
    if stage_callback is not None:
        stage_callback(summary, final_modes)
    reached = bool(eligible)
    paths = (
        recover_final_integer_paths(
            evaluator=evaluator,
            coordinates=coordinates,
            modes=final_modes,
            config=active,
        )
        if reached
        else ()
    )
    return PhasePartitionTransportResult(
        session_count=len(sessions),
        final_modes=final_modes,
        path_modes=paths,
        stage_history=(summary,),
        partition_history=(old_partition, new_partition),
        reached_all_sessions=reached,
        failure_class=None if reached else "global-island-closure-phase-inconsistent",
    )


def run_bayesian_phase_partition_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    initial_modes: Sequence[PhasePartitionMode] | None = None,
    initial_partition: Sequence[Sequence[int]] | None = None,
    initial_level_index: int = 0,
    config: PhasePartitionTransportConfig | None = None,
    stage_callback: Callable[[PhasePartitionStageSummary, Sequence[PhasePartitionMode]], None] | None = None,
) -> PhasePartitionTransportResult:
    """Run nested partition bridges and recover the final integer portfolio."""

    active = config or PhasePartitionTransportConfig()
    resumed = tuple(initial_modes or ())
    if not invariant_modes and not resumed:
        raise ValueError("At least one invariant or resumed partition mode is required.")
    first_state = resumed[0] if resumed else invariant_modes[0]
    first_profile = evaluator.evaluate_values(np.asarray(first_state.parameter_values, dtype=float))
    session_count = int(first_profile.session_count)
    singleton = tuple((index,) for index in range(session_count))
    adaptive_merges = bool(active.adaptive_partition_merges)
    fixed_partitions: list[tuple[tuple[int, ...], ...]] = []
    if resumed and not adaptive_merges:
        raise ValueError("Checkpoint continuation requires adaptive partition merges.")
    if not adaptive_merges:
        requested_counts = [session_count]
        requested_counts.extend(int(value) for value in active.partition_counts)
        requested_counts.append(1)
        counts: list[int] = []
        for value in requested_counts:
            clipped = min(max(int(value), 1), session_count)
            if clipped not in counts:
                counts.append(clipped)
        counts = sorted(counts, reverse=True)
        if counts[0] != session_count:
            counts.insert(0, session_count)
        fixed_partitions = [
            temporal_phase_partition(first_profile, block_count=value)
            for value in counts
        ]
    if resumed:
        current = tuple(
            replace(
                mode,
                parameter_values=np.asarray(mode.parameter_values, dtype=float).copy(),
                normalized_values=np.asarray(mode.normalized_values, dtype=float).copy(),
                covariance=_stable_covariance(np.asarray(mode.covariance, dtype=float)),
            )
            for mode in resumed
        )
        old_partition = tuple(
            tuple(int(item) for item in block)
            for block in (initial_partition or current[0].partition)
        )
        if any(mode.partition != old_partition for mode in current):
            raise ValueError("All resumed modes must share the checkpoint partition.")
        if sorted(item for block in old_partition for item in block) != list(range(session_count)):
            raise ValueError("The resumed partition must cover every session exactly once.")
    else:
        current = tuple(
            PhasePartitionMode(
                mode_id=f"invariant_{index:04d}",
                parent_mode_id=getattr(mode, "mode_id", None),
                proposal_source="integer-invariant",
                level_index=0,
                partition=singleton,
                parameter_values=np.asarray(mode.parameter_values, dtype=float).copy(),
                normalized_values=np.asarray(mode.normalized_values, dtype=float).copy(),
                covariance=_stable_covariance(np.asarray(mode.covariance, dtype=float)),
                shape_chi2=float(mode.shape_chi2),
                collapsed_phase_deviance=0.0,
                prior_chi2=float(getattr(mode, "prior_chi2", 0.0)),
                objective_chi2=float(getattr(mode, "objective_chi2", mode.shape_chi2)),
                log_mass=float(getattr(mode, "log_mass", -0.5 * mode.shape_chi2)),
                posterior_mass=0.0,
                nfev=int(getattr(mode, "nfev", 0)),
                phase_beta=0.0,
                success=bool(getattr(mode, "success", True)),
            )
            for index, mode in enumerate(invariant_modes)
        )
        old_partition = singleton
    current = _with_masses(current)
    history: list[PhasePartitionStageSummary] = []
    portfolios: list[tuple[tuple[int, ...], ...]] = [old_partition]

    transport_complete = True
    level_index = max(int(initial_level_index), 0) if resumed else 0
    completed_merges = 0
    while len(old_partition) > 1:
        if 0 < int(active.maximum_partition_merges) <= completed_merges:
            transport_complete = False
            break
        level_index += 1
        merge_scores: tuple[PhasePartitionMergeScore, ...] = ()
        selected_merge: PhasePartitionMergeScore | None = None
        if adaptive_merges:
            merge_scores = score_adjacent_phase_partition_merges(
                evaluator=evaluator,
                coordinates=coordinates,
                modes=current,
                partition=old_partition,
                level_index=level_index,
                config=active,
            )
            if active.proposal_aware_merge_acquisition and len(merge_scores) > 1:
                merge_scores = score_merge_homotopy_endpoints(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    modes=current,
                    old_partition=old_partition,
                    scores=merge_scores,
                    config=active,
                )
            try:
                selected_merge = select_next_phase_partition_merge(
                    merge_scores,
                    ess_target_fraction=active.bridge_ess_target_fraction,
                )
            except ValueError:
                transport_complete = False
                break
            new_partition = selected_merge.partition
        else:
            if level_index >= len(fixed_partitions):
                transport_complete = False
                break
            new_partition = fixed_partitions[level_index]
        fixed_betas = tuple(
            sorted(
                {
                    min(max(float(value), 0.0), 1.0)
                    for value in active.bridge_betas
                    if float(value) > 0.0
                }
            )
        )
        if fixed_betas and fixed_betas[-1] < 1.0:
            fixed_betas = (*fixed_betas, 1.0)
        fraction = 0.0
        step_index = 0
        tempered_config = replace(
            active,
            maximum_optimized_proposals=max(
                int(active.tempered_maximum_optimized_proposals),
                1,
            ),
            optimizer_maxiter=max(int(active.tempered_optimizer_maxiter), 1),
        )

        def fit_bridge_modes(
            parents: Sequence[PhasePartitionMode],
            beta: float,
            fit_config: PhasePartitionTransportConfig,
        ) -> tuple[
            tuple[PhasePartitionMode, ...],
            list[PhasePartitionMode],
            int,
            float,
            int,
        ]:
            proposals: list[tuple[np.ndarray, str, str | None]] = []
            proposal_parents = _select_proposal_parents(parents, config=fit_config)
            for parent in proposal_parents:
                proposals.extend(
                    _predictive_proposals(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        mode=parent,
                        partition=new_partition,
                        config=fit_config,
                        nonlinear_bridge=float(beta) >= 1.0 - 1.0e-12,
                    )
                )
            fitted: list[PhasePartitionMode] = []
            for control_index, parent in enumerate(proposal_parents):
                control = _evaluate_partition_control_mode(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    parent=parent,
                    old_partition=old_partition,
                    new_partition=new_partition,
                    beta=beta,
                    level_index=level_index,
                    control_index=control_index,
                    config=fit_config,
                )
                if control is not None:
                    fitted.append(control)
            proposal_objectives: list[float] = []
            for state, _, _ in proposals:
                try:
                    profile = evaluator.evaluate_values(
                        _physical_values(coordinates, state)
                    )
                    proposal_objectives.append(
                        partition_target_objective(
                            profile,
                            old_partition=old_partition,
                            new_partition=new_partition,
                            beta=beta,
                            normalized_values=state,
                            prior_precision=fit_config.prior_precision,
                            hermite_order=fit_config.collapsed_hermite_order,
                            maximum_wrap_terms=fit_config.collapsed_maximum_wrap_terms,
                        )
                    )
                except Exception:
                    proposal_objectives.append(math.inf)
            optimized_proposals = _select_scored_proposals(
                proposals,
                proposal_objectives,
                config=fit_config,
            )
            for proposal_index, (state, source, parent_id) in enumerate(
                optimized_proposals
            ):
                mode = _optimize_mode(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    start=state,
                    parent_mode_id=parent_id,
                    source=source,
                    old_partition=old_partition,
                    new_partition=new_partition,
                    beta=beta,
                    level_index=level_index,
                    mode_index=len(proposal_parents) + proposal_index,
                    config=fit_config,
                )
                if mode is not None:
                    fitted.append(mode)
            selected = _select_modes(fitted, parents=parents, config=fit_config)
            parent_ids = {mode.mode_id for mode in parents}
            represented = {
                mode.parent_mode_id
                for mode in selected
                if mode.parent_mode_id is not None
            }
            total_mass = float(sum(max(mode.posterior_mass, 0.0) for mode in parents))
            coverage = (
                float(
                    sum(
                        max(mode.posterior_mass, 0.0)
                        for mode in parents
                        if mode.mode_id in represented
                    )
                    / total_mass
                )
                if total_mass > 0.0
                else (
                    float(len(parent_ids & represented) / len(parent_ids))
                    if parent_ids
                    else 1.0
                )
            )
            return (
                selected,
                fitted,
                len(proposals) + len(proposal_parents),
                coverage,
                len(optimized_proposals),
            )

        while fraction < 1.0 and step_index < max(int(active.bridge_maximum_steps), 1):
            direct_attempted = False
            direct_accepted = False
            direct_best_shape = math.inf
            direct_best_phase = math.inf
            direct_phase_limit = math.inf
            direct_best_objective = math.inf
            direct_proposal_count = 0
            direct_optimized_proposal_count = 0
            if fixed_betas:
                if step_index >= len(fixed_betas):
                    break
                next_fraction = fixed_betas[step_index]
                overlap_ess = float(len(current))
                relative_overlap_ess = 1.0
                current_fit_config = active
            elif step_index == 0 and fraction == 0.0 and active.direct_bridge_first:
                direct_attempted = True
                next_fraction = 1.0
                overlap_ess = (
                    float(selected_merge.effective_sample_size)
                    if selected_merge is not None
                    else float(len(current))
                )
                relative_overlap_ess = (
                    float(selected_merge.relative_effective_sample_size)
                    if selected_merge is not None
                    else 1.0
                )
                current_fit_config = active
            else:
                next_fraction, overlap_ess, relative_overlap_ess = _select_mixture_bridge_beta(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    modes=current,
                    old_partition=old_partition,
                    new_partition=new_partition,
                    current_beta=fraction,
                    level_index=level_index,
                    step_index=step_index,
                    config=active,
                )
                current_fit_config = tempered_config
            selected, fitted, proposal_count, coverage, optimized_proposal_count = fit_bridge_modes(
                current,
                next_fraction,
                current_fit_config,
            )
            if direct_attempted:
                direct_best_shape = min(
                    (mode.shape_chi2 for mode in selected),
                    default=math.inf,
                )
                direct_best_objective = min(
                    (mode.objective_chi2 for mode in selected),
                    default=math.inf,
                )
                direct_proposal_count = proposal_count
                direct_optimized_proposal_count = optimized_proposal_count
                previous_shape = min(
                    (mode.shape_chi2 for mode in current),
                    default=math.inf,
                )
                shape_limit = max(
                    previous_shape
                    * max(float(active.direct_bridge_shape_ratio_limit), 1.0),
                    previous_shape
                    + max(float(active.direct_bridge_shape_additive_limit), 0.0),
                )
                previous_phase_values: list[float] = []
                for mode in current:
                    try:
                        previous_phase_values.append(
                            _map_partition_phase_chi2(
                                evaluator.evaluate_values(mode.parameter_values),
                                old_partition,
                                maximum_lattice_nodes=active.maximum_lattice_nodes,
                            )
                        )
                    except Exception:
                        continue
                previous_phase = min(previous_phase_values, default=math.inf)
                new_phase_dimension = sum(
                    max(len(tuple(block)) - 1, 0)
                    for block in new_partition
                )
                direct_phase_limit = _phase_bridge_acceptance_limit(
                    previous_phase_chi2=previous_phase,
                    new_phase_dimension=new_phase_dimension,
                    config=active,
                )
                phase_by_mode: list[tuple[PhasePartitionMode, float]] = []
                for mode in selected:
                    try:
                        phase_chi2 = _map_partition_phase_chi2(
                            evaluator.evaluate_values(mode.parameter_values),
                            new_partition,
                            maximum_lattice_nodes=active.maximum_lattice_nodes,
                        )
                    except Exception:
                        phase_chi2 = math.inf
                    phase_by_mode.append((mode, float(phase_chi2)))
                direct_best_phase = min(
                    (phase_chi2 for _, phase_chi2 in phase_by_mode),
                    default=math.inf,
                )
                eligible = tuple(
                    mode
                    for mode, phase_chi2 in phase_by_mode
                    if mode.shape_chi2 <= shape_limit
                    and phase_chi2 <= direct_phase_limit
                )
                direct_accepted = bool(eligible)
                if direct_accepted:
                    selected = _with_masses(
                        sorted(
                            eligible,
                            key=lambda mode: (mode.objective_chi2, -mode.log_mass),
                        )[: max(int(active.mode_beam), 1)]
                    )
                if not direct_accepted:
                    next_fraction, overlap_ess, relative_overlap_ess = _select_mixture_bridge_beta(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        modes=current,
                        old_partition=old_partition,
                        new_partition=new_partition,
                        current_beta=fraction,
                        level_index=level_index,
                        step_index=step_index,
                        config=active,
                    )
                    selected, fitted, proposal_count, coverage, optimized_proposal_count = fit_bridge_modes(
                        current,
                        next_fraction,
                        tempered_config,
                    )
            summary = PhasePartitionStageSummary(
                level_index=level_index,
                beta=next_fraction,
                old_partition=tuple(tuple(block) for block in old_partition),
                new_partition=tuple(tuple(block) for block in new_partition),
                input_mode_count=len(current),
                proposal_count=proposal_count,
                retained_count=len(selected),
                unique_mode_count=len(
                    {
                        tuple(
                            np.rint(np.asarray(mode.normalized_values) * 1.0e8).astype(np.int64)
                        )
                        for mode in fitted
                    }
                ),
                best_shape_chi2=min((mode.shape_chi2 for mode in selected), default=math.inf),
                best_collapsed_phase_deviance=min((mode.collapsed_phase_deviance for mode in selected), default=math.inf),
                best_objective_chi2=min((mode.objective_chi2 for mode in selected), default=math.inf),
                ancestor_coverage=coverage,
                bridge_effective_sample_size=float(overlap_ess),
                bridge_relative_effective_sample_size=float(relative_overlap_ess),
                candidate_merge_count=max(len(merge_scores), 1),
                selected_merge_index=(
                    int(selected_merge.merge_index)
                    if selected_merge is not None
                    else -1
                ),
                selected_bridge_full_ess_fraction=(
                    float(selected_merge.relative_effective_sample_size)
                    if selected_merge is not None
                    else 0.0
                ),
                selected_bridge_log_evidence_increment=(
                    float(selected_merge.log_evidence_increment)
                    if selected_merge is not None
                    else 0.0
                ),
                selected_bridge_initial_adaptive_beta=(
                    float(selected_merge.initial_adaptive_beta)
                    if selected_merge is not None
                    else 0.0
                ),
                selected_bridge_log_factor_standard_deviation=(
                    float(selected_merge.log_factor_standard_deviation)
                    if selected_merge is not None
                    else math.inf
                ),
                selected_merge_homotopy_transportable=(
                    bool(selected_merge.homotopy_transportable)
                    if selected_merge is not None
                    else False
                ),
                selected_merge_homotopy_shape_chi2=(
                    float(selected_merge.homotopy_shape_chi2)
                    if selected_merge is not None
                    else math.inf
                ),
                selected_merge_homotopy_objective_chi2=(
                    float(selected_merge.homotopy_objective_chi2)
                    if selected_merge is not None
                    else math.inf
                ),
                direct_bridge_attempted=direct_attempted,
                direct_bridge_accepted=direct_accepted,
                direct_bridge_best_shape_chi2=direct_best_shape,
                direct_bridge_best_phase_chi2=direct_best_phase,
                direct_bridge_phase_chi2_limit=direct_phase_limit,
                direct_bridge_best_objective_chi2=direct_best_objective,
                direct_bridge_proposal_count=direct_proposal_count,
                direct_bridge_optimized_proposal_count=direct_optimized_proposal_count,
                optimized_proposal_count=optimized_proposal_count,
                merge_scores=merge_scores,
            )
            history.append(summary)
            current = selected
            fraction = float(next_fraction)
            step_index += 1
            if stage_callback is not None:
                stage_callback(summary, current)
            if not current:
                break
        if fraction < 1.0:
            transport_complete = False
        portfolios.append(new_partition)
        if not current or not transport_complete:
            break
        old_partition = new_partition
        completed_merges += 1

    reached = (
        bool(current)
        and transport_complete
        and len(old_partition) == 1
        and current[0].partition == old_partition
    )
    path_modes = recover_final_integer_paths(
        evaluator=evaluator,
        coordinates=coordinates,
        modes=current,
        config=active,
    ) if reached else ()
    return PhasePartitionTransportResult(
        session_count=session_count,
        final_modes=current,
        path_modes=path_modes,
        stage_history=tuple(history),
        partition_history=tuple(
            tuple(tuple(block) for block in partition)
            for partition in portfolios
        ),
        reached_all_sessions=reached,
        failure_class=None if reached else "partition-collapsed-transport-incomplete",
    )
