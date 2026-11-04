"""Bayesian phase-island and block-gauge inference.

This search is independent of the chronological lattice tracker. Its state is
a partition of source sessions into phase components. Each component has one
unconstrained phase gauge, and only internal phase differences enter its
likelihood. All components share the DD parameters, which are refit jointly.

The forest represents intermediate bridges toward the one-component partition,
the final scientific target in which every session has a relative integer
phase.
"""

from __future__ import annotations

from apollo_timing.inference._shared_helpers import (
    _normalized_bounds,
    _physical_values,
)

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
)


@dataclass(frozen=True)
class PhaseForestConfig:
    """Controls for a generic phase-island Bayesian search."""

    finite_difference_step: float = 2.0e-3
    prior_precision: float = 1.0
    phase_tempering_ladder: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0)
    lattice_kappas: tuple[float, ...] = (1.0, 4.0, 16.0, 64.0)
    lattice_candidates_per_proposal: int = 6
    maximum_lattice_nodes: int = 2_000_000
    initial_island_merges: int = 0
    initial_island_max_size: int = 3
    initial_island_partitions: int = 12
    initial_island_edge_prefixes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)
    merge_candidates_per_parent: int = 12
    maximum_target_partitions_per_level: int = 24
    parent_modes_per_level: int = 12
    root_modes_per_level: int = 4
    maximum_refit_proposals_per_target: int = 12
    forest_beam: int = 16
    maximum_modes_per_topology: int = 3
    maximum_modes_per_root: int = 8
    mode_separation: float = 0.04
    fixed_path_max_nfev: int = 120
    initial_refit_max_nfev: int = 120
    global_closure_rank_margin: int = 0
    global_closure_parent_modes: int = 6
    collapsed_island_prior_samples: int = 0
    collapsed_island_samples_per_scale: int = 0
    collapsed_island_scales: tuple[float, ...] = (4.0, 16.0, 64.0, 256.0)
    collapsed_island_retained_starts: int = 16
    collapsed_island_integer_candidates: int = 2
    collapsed_island_minimum_separation: float = 0.08
    collapsed_island_phase_sigmas: tuple[float, ...] = (
        0.50,
        0.25,
        0.12,
        0.06,
        0.03,
        0.015,
        0.0075,
    )
    collapsed_island_anneal_max_nfev: int = 48
    collapsed_island_phase_step_cycles: float = 0.03
    collapsed_global_partition: bool = True
    random_seed: int = 20260724


@dataclass(frozen=True)
class PhaseComponent:
    """A connected phase island with a local integer gauge."""

    component_id: str
    session_indices: tuple[int, ...]
    anchor_session: int
    integer_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        sessions = tuple(sorted(int(value) for value in self.session_indices))
        if not sessions:
            raise ValueError("A phase component must contain at least one session.")
        if int(self.anchor_session) not in sessions:
            raise ValueError("The component anchor must belong to the component.")
        expected = len(sessions) - 1
        if len(self.integer_offsets) != expected:
            raise ValueError("Component integer offsets must exclude its anchor.")
        if sessions != tuple(self.session_indices):
            raise ValueError("Component session indices must be sorted.")

    @property
    def non_anchor_sessions(self) -> tuple[int, ...]:
        return tuple(
            session for session in self.session_indices if session != self.anchor_session
        )

    @property
    def size(self) -> int:
        return len(self.session_indices)


@dataclass(frozen=True)
class PhaseForestMode:
    mode_id: str
    root_mode_id: str
    parent_mode_id: str | None
    proposal_source: str
    components: tuple[PhaseComponent, ...]
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
    merge_score: float
    nfev: int
    success: bool

    @property
    def component_count(self) -> int:
        return len(self.components)

    @property
    def partition_key(self) -> tuple[tuple[int, ...], ...]:
        return tuple(component.session_indices for component in self.components)

    @property
    def integer_vector(self) -> tuple[int, ...]:
        values: list[int] = []
        for component in self.components:
            values.extend(int(value) for value in component.integer_offsets)
        return tuple(values)

    def chronological_integer_offsets(self, session_count: int) -> tuple[int, ...]:
        if self.component_count != 1 or self.components[0].size != session_count:
            raise ValueError("Chronological offsets require one component containing all sessions.")
        component = self.components[0]
        values = np.zeros(session_count, dtype=np.int64)
        for session, value in zip(component.non_anchor_sessions, component.integer_offsets, strict=True):
            values[session] = int(value)
        values -= values[component.anchor_session]
        return tuple(int(value) for value in values[1:])


@dataclass(frozen=True)
class PhaseMergeProposal:
    parent_mode_id: str
    target_partition: tuple[tuple[int, ...], ...]
    merged_component_ids: tuple[str, str]
    merge_score: float


@dataclass(frozen=True)
class PhaseForestStageSummary:
    stage_index: int
    component_count: int
    parent_count: int
    target_count: int
    proposed_count: int
    refitted_count: int
    selected_count: int
    unique_topologies: int
    closure_refitted_count: int
    minimum_lattice_distance: float
    best_shape_chi2: float
    best_phase_chi2: float
    best_objective_chi2: float
    best_closure_objective_chi2: float
    failure_class: str | None


@dataclass(frozen=True)
class PhaseForestResult:
    session_count: int
    initial_modes: tuple[PhaseForestMode, ...]
    closure_modes: tuple[PhaseForestMode, ...]
    final_modes: tuple[PhaseForestMode, ...]
    stage_history: tuple[PhaseForestStageSummary, ...]
    level_portfolios: tuple[tuple[PhaseForestMode, ...], ...]
    reached_all_sessions: bool
    failure_class: str | None


@dataclass(frozen=True)
class ForestProfileLinearization:
    normalized_values: np.ndarray
    profile: SessionShapeProfile
    shape_jacobian: np.ndarray
    session_offset_jacobian: np.ndarray
    phase_center: np.ndarray
    phase_jacobian: np.ndarray
    phase_covariance: np.ndarray


@dataclass(frozen=True)
class ForestLatticeProposal:
    target_partition: tuple[tuple[int, ...], ...]
    integer_vector: tuple[int, ...]
    initial_normalized_values: np.ndarray
    squared_mahalanobis: float
    lattice_score: float
    kappa: float
    source: str


@dataclass(frozen=True)
class CollapsedIslandStart:
    normalized_values: np.ndarray
    integer_vector: tuple[int, ...]
    shape_chi2: float
    lattice_chi2: float
    collapsed_objective_chi2: float
    source: str


@dataclass(frozen=True)
class CollapsedProfilePoint:
    """One expensive evaluator result shared by all island topologies."""

    normalized_values: np.ndarray
    source: str
    profile: SessionShapeProfile
    shape_chi2: float
    prior_chi2: float






def _stable_inverse(matrix: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if symmetric.size == 0:
        return symmetric.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors / clipped) @ eigenvectors.T


def _stable_covariance(matrix: np.ndarray, *, floor: float = 1.0e-12) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if symmetric.size == 0:
        return symmetric.copy()
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors * clipped) @ eigenvectors.T


def _laplace_log_mass(objective: float, covariance: np.ndarray) -> float:
    covariance = _stable_covariance(covariance)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        return -math.inf
    dimension = covariance.shape[0]
    return float(-0.5 * float(objective) + 0.5 * float(logdet) + 0.5 * dimension * math.log(2.0 * math.pi))


def _normalize_masses(modes: Sequence[PhaseForestMode]) -> tuple[PhaseForestMode, ...]:
    if not modes:
        return ()
    logs = np.asarray([float(mode.log_mass) for mode in modes], dtype=float)
    finite = np.isfinite(logs)
    masses = np.zeros(len(modes), dtype=float)
    if np.any(finite):
        masses[finite] = np.exp(logs[finite] - logsumexp(logs[finite]))
    else:
        masses.fill(1.0 / len(modes))
    return tuple(replace(mode, posterior_mass=float(mass)) for mode, mass in zip(modes, masses, strict=True))


def singleton_partition(session_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple((index,) for index in range(int(session_count)))


def _canonical_partition(partition: Iterable[Iterable[int]]) -> tuple[tuple[int, ...], ...]:
    components = [tuple(sorted(int(value) for value in sessions)) for sessions in partition]
    if any(not component for component in components):
        raise ValueError("A phase partition cannot contain an empty component.")
    flattened = [session for component in components for session in component]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A phase partition cannot repeat a session.")
    return tuple(sorted(components, key=lambda component: (component[0], len(component), component)))


def _partition_components(
    partition: Sequence[Sequence[int]],
    integer_vector: Sequence[int],
    *,
    component_prefix: str = "C",
) -> tuple[PhaseComponent, ...]:
    canonical = _canonical_partition(partition)
    values = tuple(int(value) for value in integer_vector)
    expected = sum(len(component) - 1 for component in canonical)
    if len(values) != expected:
        raise ValueError("The integer vector does not match the phase partition.")
    components: list[PhaseComponent] = []
    cursor = 0
    for index, sessions in enumerate(canonical):
        anchor = sessions[0]
        dimension = len(sessions) - 1
        components.append(
            PhaseComponent(
                component_id=f"{component_prefix}{index:03d}",
                session_indices=sessions,
                anchor_session=anchor,
                integer_offsets=values[cursor : cursor + dimension],
            )
        )
        cursor += dimension
    return tuple(components)


def component_contrast_matrix(
    components: Sequence[PhaseComponent],
    session_count: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for component in components:
        for session in component.non_anchor_sessions:
            row = np.zeros(int(session_count), dtype=float)
            row[session] = 1.0
            row[component.anchor_session] = -1.0
            rows.append(row)
    return np.asarray(rows, dtype=float).reshape((-1, int(session_count)))


def forest_phase_geometry(
    profile: SessionShapeProfile,
    components: Sequence[PhaseComponent],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    if offsets.ndim != 1 or variances.shape != offsets.shape:
        raise ValueError("Session offset arrays must have matching one-dimensional shapes.")
    contrast = component_contrast_matrix(components, offsets.size)
    phase_center = contrast @ offsets
    phase_covariance = contrast @ np.diag(variances) @ contrast.T
    if phase_center.size == 0:
        return phase_center, np.zeros((0, 0), dtype=float), contrast
    return phase_center, _stable_covariance(phase_covariance), contrast


def forest_objective(
    profile: SessionShapeProfile,
    components: Sequence[PhaseComponent],
    *,
    integer_vector: Sequence[int] | None = None,
) -> tuple[float, float, float]:
    phase_center, phase_covariance, _ = forest_phase_geometry(profile, components)
    integers = np.asarray(
        tuple(int(value) for value in integer_vector)
        if integer_vector is not None
        else tuple(value for component in components for value in component.integer_offsets),
        dtype=float,
    )
    if integers.shape != phase_center.shape:
        raise ValueError("The integer vector does not match the component phase dimension.")
    delta = phase_center - integers
    phase_chi2 = float(delta @ np.linalg.solve(phase_covariance, delta)) if delta.size else 0.0
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    shape_chi2 = float(shape @ shape)
    return shape_chi2, phase_chi2, shape_chi2 + phase_chi2


def forest_residual_vector(
    profile: SessionShapeProfile,
    components: Sequence[PhaseComponent],
    *,
    integer_vector: Sequence[int],
    normalized_values: Sequence[float] | None = None,
    prior_precision: float = 0.0,
    phase_temperature: float = 1.0,
) -> np.ndarray:
    phase_center, phase_covariance, _ = forest_phase_geometry(profile, components)
    integers = np.asarray(integer_vector, dtype=float)
    if integers.shape != phase_center.shape:
        raise ValueError("The integer vector does not match the component phase dimension.")
    parts = [np.asarray(profile.normalized_residuals, dtype=float)]
    if integers.size:
        chol = np.linalg.cholesky(phase_covariance)
        phase = np.linalg.solve(chol, phase_center - integers)
        parts.append(math.sqrt(max(float(phase_temperature), 0.0)) * phase)
    if prior_precision > 0.0:
        if normalized_values is None:
            raise ValueError("Normalized values are required when the prior is active.")
        parts.append(math.sqrt(float(prior_precision)) * np.asarray(normalized_values, dtype=float))
    result = np.concatenate(parts)
    if not np.all(np.isfinite(result)):
        raise ValueError("The phase forest residual contains non-finite values.")
    return result


def linearize_forest(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    normalized_values: Sequence[float],
    step: float,
) -> ForestProfileLinearization:
    center = np.asarray(normalized_values, dtype=float)
    lower, upper = _normalized_bounds(coordinates)
    if center.shape != lower.shape:
        raise ValueError("The normalized state does not match the coordinate dimension.")
    profile = evaluator.evaluate_values(_physical_values(coordinates, center))
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    phase_center, phase_covariance, contrast = forest_phase_geometry(profile, components)
    shape_jacobian = np.empty((shape.size, center.size), dtype=float)
    session_offset_jacobian = np.empty((profile.session_count, center.size), dtype=float)
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
        plus_shape = np.asarray(plus.normalized_residuals, dtype=float)
        minus_shape = np.asarray(minus.normalized_residuals, dtype=float)
        shape_jacobian[:, index] = (plus_shape - minus_shape) / denominator
        session_offset_jacobian[:, index] = (
            np.asarray(plus.session_offsets, dtype=float)
            - np.asarray(minus.session_offsets, dtype=float)
        ) / denominator
    phase_jacobian = contrast @ session_offset_jacobian
    return ForestProfileLinearization(
        normalized_values=center.copy(),
        profile=profile,
        shape_jacobian=shape_jacobian,
        session_offset_jacobian=session_offset_jacobian,
        phase_center=phase_center,
        phase_jacobian=phase_jacobian,
        phase_covariance=phase_covariance,
    )




def propose_forest_lattice(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    normalized_values: Sequence[float],
    covariance: np.ndarray,
    config: PhaseForestConfig,
    source: str,
) -> tuple[ForestLatticeProposal, ...]:
    """Decode complete block-gauge integer vectors from one joint Gaussian."""

    lower, upper = _normalized_bounds(coordinates)
    linearization = linearize_forest(
        evaluator=evaluator,
        coordinates=coordinates,
        components=components,
        normalized_values=normalized_values,
        step=config.finite_difference_step,
    )
    dimension = linearization.phase_center.size
    if dimension == 0:
        return (
            ForestLatticeProposal(
                target_partition=tuple(component.session_indices for component in components),
                integer_vector=(),
                initial_normalized_values=np.clip(np.asarray(normalized_values, dtype=float), lower, upper),
                squared_mahalanobis=0.0,
                lattice_score=0.0,
                kappa=1.0,
                source=source,
            ),
        )

    target_partition = tuple(component.session_indices for component in components)
    unique: dict[tuple[float, tuple[int, ...]], ForestLatticeProposal] = {}
    scale_order: list[float] = []
    base_covariance = _stable_covariance(np.asarray(covariance, dtype=float))
    for kappa in config.lattice_kappas:
        scale = max(float(kappa), 1.0e-6)
        scale_order.append(scale)
        continuous_covariance = _stable_covariance(scale * scale * base_covariance)
        predictive_covariance = linearization.phase_covariance + (
            linearization.phase_jacobian
            @ continuous_covariance
            @ linearization.phase_jacobian.T
        )
        predictive_covariance = _stable_covariance(predictive_covariance)
        sign, logdet = np.linalg.slogdet(predictive_covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            continue
        try:
            integer_result = solve_integer_least_squares(
                linearization.phase_center,
                predictive_covariance,
                candidate_count=max(int(config.lattice_candidates_per_proposal), 1),
                max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        gain = continuous_covariance @ linearization.phase_jacobian.T @ _stable_inverse(predictive_covariance)
        for candidate in integer_result.candidates:
            displacement = np.asarray(candidate.integer_vector, dtype=float) - linearization.phase_center
            start = linearization.normalized_values + gain @ displacement
            start = np.clip(start, lower, upper)
            proposal = ForestLatticeProposal(
                target_partition=target_partition,
                integer_vector=tuple(int(value) for value in candidate.integer_vector),
                initial_normalized_values=start,
                squared_mahalanobis=float(candidate.squared_mahalanobis),
                lattice_score=float(candidate.squared_mahalanobis + logdet),
                kappa=scale,
                source=source,
            )
            proposal_key = (scale, proposal.integer_vector)
            previous = unique.get(proposal_key)
            if previous is None or proposal.lattice_score < previous.lattice_score:
                unique[proposal_key] = proposal
    # Each kappa is an acquisition scale, not a model parameter.  Preserve a
    # round-robin quota across scales so a broad proposal family cannot
    # monopolize the refit budget merely because its Mahalanobis distances are
    # numerically smaller.
    by_scale: dict[float, list[ForestLatticeProposal]] = {
        scale: [] for scale in scale_order
    }
    for proposal in unique.values():
        by_scale.setdefault(proposal.kappa, []).append(proposal)
    for proposals in by_scale.values():
        proposals.sort(
            key=lambda item: (item.lattice_score, item.squared_mahalanobis)
        )
    ordered: list[ForestLatticeProposal] = []
    for rank in range(max((len(items) for items in by_scale.values()), default=0)):
        for scale in scale_order:
            items = by_scale.get(scale, [])
            if rank < len(items):
                ordered.append(items[rank])
    return tuple(ordered)


def _normalized_prior_samples(
    *,
    coordinates: Any,
    count: int,
    seed: int,
) -> np.ndarray:
    requested = max(int(count), 0)
    lower, upper = _normalized_bounds(coordinates)
    if requested == 0:
        return np.zeros((0, lower.size), dtype=float)
    if hasattr(coordinates, "sample_prior"):
        try:
            search_values = np.asarray(
                coordinates.sample_prior(requested, seed=int(seed)),
                dtype=float,
            )
            normalized = np.vstack(
                [coordinates.normalize(row) for row in search_values]
            )
            if normalized.shape == (requested, lower.size) and np.all(
                np.isfinite(normalized)
            ):
                return np.clip(normalized, lower, upper)
        except Exception:
            pass
    exponent = int(math.ceil(math.log2(max(requested, 1))))
    unit = qmc.Sobol(
        d=lower.size,
        scramble=True,
        seed=int(seed),
    ).random_base2(exponent)[:requested]
    return lower + unit * (upper - lower)


def _inflated_mode_samples(
    *,
    invariant_modes: Sequence[Any],
    coordinates: Any,
    samples_per_scale: int,
    scales: Sequence[float],
    seed: int,
) -> tuple[tuple[np.ndarray, str], ...]:
    requested = max(int(samples_per_scale), 0)
    if requested == 0:
        return ()
    lower, upper = _normalized_bounds(coordinates)
    samples: list[tuple[np.ndarray, str]] = []
    for mode_index, mode in enumerate(invariant_modes):
        center = np.asarray(mode.normalized_values, dtype=float)
        covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
        chol = np.linalg.cholesky(covariance)
        for scale_index, raw_scale in enumerate(scales):
            scale = max(float(raw_scale), 1.0e-6)
            exponent = int(math.ceil(math.log2(max(requested, 1))))
            unit = qmc.Sobol(
                d=center.size,
                scramble=True,
                seed=int(seed) + 1009 * mode_index + 37 * scale_index,
            ).random_base2(exponent)[:requested]
            normal = ndtri(np.clip(unit, 1.0e-9, 1.0 - 1.0e-9))
            states = center + scale * (normal @ chol.T)
            for state in states:
                samples.append(
                    (
                        np.clip(np.asarray(state, dtype=float), lower, upper),
                        f"inflated_mode_{mode_index:03d}_k{scale:g}",
                    )
                )
    return tuple(samples)


def build_collapsed_profile_bank(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    config: PhaseForestConfig,
) -> tuple[CollapsedProfilePoint, ...]:
    """Evaluate the global continuous proposal bank exactly once."""

    prior_points = _normalized_prior_samples(
        coordinates=coordinates,
        count=config.collapsed_island_prior_samples,
        seed=config.random_seed,
    )
    candidates: list[tuple[np.ndarray, str]] = [
        (np.asarray(mode.normalized_values, dtype=float), f"invariant_{index:03d}")
        for index, mode in enumerate(invariant_modes)
    ]
    candidates.extend(
        (row, "prior_sobol") for row in np.asarray(prior_points, dtype=float)
    )
    candidates.extend(
        _inflated_mode_samples(
            invariant_modes=invariant_modes,
            coordinates=coordinates,
            samples_per_scale=config.collapsed_island_samples_per_scale,
            scales=config.collapsed_island_scales,
            seed=config.random_seed,
        )
    )
    bank: list[CollapsedProfilePoint] = []
    seen_states: set[tuple[float, ...]] = set()
    for normalized_values, source in candidates:
        state = np.asarray(normalized_values, dtype=float)
        key = tuple(float(value) for value in np.round(state, 12))
        if key in seen_states:
            continue
        seen_states.add(key)
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, state))
            shape = np.asarray(profile.normalized_residuals, dtype=float)
            bank.append(
                CollapsedProfilePoint(
                    normalized_values=state.copy(),
                    source=source,
                    profile=profile,
                    shape_chi2=float(shape @ shape),
                    prior_chi2=float(
                        max(float(config.prior_precision), 0.0) * (state @ state)
                    ),
                )
            )
        except Exception:
            continue
    return tuple(bank)


def _circular_phase_residual_vector(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    normalized_values: np.ndarray,
    phase_sigma: float,
    prior_precision: float,
) -> np.ndarray:
    """Evaluate a smooth circular likelihood with integer phase marginalized."""

    profile = evaluator.evaluate_values(_physical_values(coordinates, normalized_values))
    parts = [np.asarray(profile.normalized_residuals, dtype=float)]
    phase_center, phase_covariance, _ = forest_phase_geometry(profile, components)
    if phase_center.size:
        covariance = phase_covariance + max(float(phase_sigma), 1.0e-8) ** 2 * np.eye(
            phase_center.size,
            dtype=float,
        )
        circular = np.sin(math.pi * phase_center) / math.pi
        parts.append(np.linalg.solve(np.linalg.cholesky(covariance), circular))
    if prior_precision > 0.0:
        parts.append(math.sqrt(float(prior_precision)) * np.asarray(normalized_values, dtype=float))
    result = np.concatenate(parts)
    if not np.all(np.isfinite(result)):
        raise ValueError("The circular phase posterior contains non-finite values.")
    return result


def _phase_resolving_steps(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    normalized_values: np.ndarray,
    config: PhaseForestConfig,
) -> np.ndarray:
    """Choose absolute parameter steps that resolve one fraction of a cycle."""

    linearization = linearize_forest(
        evaluator=evaluator,
        coordinates=coordinates,
        components=components,
        normalized_values=normalized_values,
        step=config.finite_difference_step,
    )
    phase_jacobian = np.asarray(linearization.phase_jacobian, dtype=float)
    maximum_rate = (
        np.max(np.abs(phase_jacobian), axis=0)
        if phase_jacobian.size
        else np.zeros(normalized_values.size, dtype=float)
    )
    requested = max(float(config.collapsed_island_phase_step_cycles), 1.0e-5)
    steps = np.full(
        normalized_values.size,
        max(float(config.finite_difference_step), 1.0e-7),
        dtype=float,
    )
    observed = maximum_rate > 0.0
    steps[observed] = np.minimum(
        steps[observed],
        requested / maximum_rate[observed],
    )
    return np.maximum(steps, 1.0e-9)


def _anneal_circular_mode(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    start: np.ndarray,
    config: PhaseForestConfig,
) -> np.ndarray | None:
    """Transport one mode from broad circular phase noise to narrow noise."""

    lower, upper = _normalized_bounds(coordinates)
    current = np.clip(np.asarray(start, dtype=float), lower, upper)
    try:
        absolute_steps = _phase_resolving_steps(
            evaluator=evaluator,
            coordinates=coordinates,
            components=components,
            normalized_values=current,
            config=config,
        )
    except Exception:
        return None

    for phase_sigma in tuple(
        float(value)
        for value in config.collapsed_island_phase_sigmas
        if np.isfinite(float(value)) and float(value) > 0.0
    ):
        failure: np.ndarray | None = None

        def residual(values: np.ndarray) -> np.ndarray:
            nonlocal failure
            try:
                result = _circular_phase_residual_vector(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    components=components,
                    normalized_values=np.asarray(values, dtype=float),
                    phase_sigma=phase_sigma,
                    prior_precision=config.prior_precision,
                )
                if failure is None:
                    failure = np.full(result.shape, 1.0e12, dtype=float)
                return result if result.shape == failure.shape else failure
            except Exception:
                if failure is None:
                    failure = np.full(1, 1.0e12, dtype=float)
                return failure

        def jacobian(values: np.ndarray) -> np.ndarray:
            center = np.asarray(values, dtype=float)
            center_residual = residual(center)
            jac = np.empty((center_residual.size, center.size), dtype=float)
            for index, requested_step in enumerate(absolute_steps):
                plus = center.copy()
                minus = center.copy()
                plus[index] = min(plus[index] + requested_step, upper[index])
                minus[index] = max(minus[index] - requested_step, lower[index])
                denominator = float(plus[index] - minus[index])
                if denominator <= 0.0:
                    jac[:, index] = 0.0
                    continue
                jac[:, index] = (residual(plus) - residual(minus)) / denominator
            return jac

        try:
            optimization = least_squares(
                residual,
                current,
                jac=jacobian,
                bounds=(lower, upper),
                method="trf",
                max_nfev=max(int(config.collapsed_island_anneal_max_nfev), 1),
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            if not np.all(np.isfinite(optimization.x)):
                return None
            current = np.asarray(optimization.x, dtype=float)
        except Exception:
            return None
    return current


def _circular_mode_score(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    normalized_values: np.ndarray,
    phase_sigma: float,
    config: PhaseForestConfig,
) -> float:
    residual = _circular_phase_residual_vector(
        evaluator=evaluator,
        coordinates=coordinates,
        components=components,
        normalized_values=normalized_values,
        phase_sigma=phase_sigma,
        prior_precision=config.prior_precision,
    )
    return float(residual @ residual)


def discover_collapsed_island_starts(
    *,
    evaluator: Any,
    coordinates: Any,
    partition: Sequence[Sequence[int]],
    invariant_modes: Sequence[Any],
    config: PhaseForestConfig,
    profile_bank: Sequence[CollapsedProfilePoint] | None = None,
) -> tuple[CollapsedIslandStart, ...]:
    """Discover continuous modes with integer phase marginalized throughout."""

    canonical = _canonical_partition(partition)
    zero_integer = tuple(
        0 for component in canonical for _ in range(len(component) - 1)
    )
    components = _partition_components(
        canonical,
        zero_integer,
        component_prefix="collapsed_",
    )
    bank = tuple(profile_bank) if profile_bank is not None else build_collapsed_profile_bank(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=invariant_modes,
        config=config,
    )
    if not bank:
        return ()
    broad_sigma = max(
        tuple(
            float(value)
            for value in config.collapsed_island_phase_sigmas
            if np.isfinite(float(value)) and float(value) > 0.0
        ),
        default=0.5,
    )
    broad_scored: list[tuple[float, CollapsedProfilePoint]] = []
    for point in bank:
        try:
            broad_scored.append(
                (
                    _circular_mode_score(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        components=components,
                        normalized_values=np.asarray(point.normalized_values, dtype=float),
                        phase_sigma=broad_sigma,
                        config=config,
                    ),
                    point,
                )
            )
        except Exception:
            continue
    broad_scored.sort(key=lambda item: (item[0], item[1].shape_chi2))
    retained = max(int(config.collapsed_island_retained_starts), 1)
    separation = max(float(config.collapsed_island_minimum_separation), 0.0)
    active: list[tuple[np.ndarray, str]] = []
    for _, point in broad_scored:
        state = np.asarray(point.normalized_values, dtype=float)
        if any(np.linalg.norm(state - existing) < separation for existing, _ in active):
            continue
        active.append((state.copy(), point.source))
        if len(active) >= retained:
            break
    if not active:
        return ()

    transported: list[tuple[float, np.ndarray, str]] = []
    for state, source in active:
        result = _anneal_circular_mode(
            evaluator=evaluator,
            coordinates=coordinates,
            components=components,
            start=state,
            config=config,
        )
        if result is None:
            continue
        try:
            transported.append(
                (
                    _circular_mode_score(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        components=components,
                        normalized_values=result,
                        phase_sigma=min(
                            tuple(
                                float(value)
                                for value in config.collapsed_island_phase_sigmas
                                if np.isfinite(float(value)) and float(value) > 0.0
                            ),
                            default=0.01,
                        ),
                        config=config,
                    ),
                    result,
                    source,
                )
            )
        except Exception:
            continue

    scored: list[CollapsedIslandStart] = []
    for _, state, source in sorted(transported, key=lambda item: item[0]):
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, state))
            shape = np.asarray(profile.normalized_residuals, dtype=float)
            phase_center, phase_covariance, _ = forest_phase_geometry(profile, components)
            if phase_center.size:
                integer_result = solve_integer_least_squares(
                    phase_center,
                    phase_covariance,
                    candidate_count=max(int(config.collapsed_island_integer_candidates), 1),
                    max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
                )
                candidates = integer_result.candidates
            else:
                candidates = ()
            if not candidates:
                candidates = (None,)
            log_terms = np.asarray(
                [-0.5 * float(item.squared_mahalanobis) for item in candidates if item is not None],
                dtype=float,
            )
            collapsed_objective = float(shape @ shape)
            if log_terms.size:
                collapsed_objective -= 2.0 * float(logsumexp(log_terms))
            for integer_candidate in candidates:
                scored.append(
                    CollapsedIslandStart(
                        normalized_values=np.asarray(state, dtype=float).copy(),
                        integer_vector=(
                            tuple(int(value) for value in integer_candidate.integer_vector)
                            if integer_candidate is not None
                            else ()
                        ),
                        shape_chi2=float(shape @ shape),
                        lattice_chi2=(
                            float(integer_candidate.squared_mahalanobis)
                            if integer_candidate is not None
                            else 0.0
                        ),
                        collapsed_objective_chi2=collapsed_objective,
                        source=source,
                    )
                )
        except Exception:
            continue
    ranked = sorted(
        scored,
        key=lambda item: (item.collapsed_objective_chi2, item.shape_chi2, item.lattice_chi2),
    )
    selected: list[CollapsedIslandStart] = []
    for candidate in ranked:
        if any(
            candidate.integer_vector == existing.integer_vector
            and np.linalg.norm(candidate.normalized_values - existing.normalized_values) < separation
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= retained:
            break
    return tuple(selected)


def _refit_forest_proposal(
    *,
    evaluator: Any,
    coordinates: Any,
    components: Sequence[PhaseComponent],
    proposal: ForestLatticeProposal,
    root_mode_id: str,
    parent_mode_id: str | None,
    config: PhaseForestConfig,
    mode_id: str,
    merge_score: float,
) -> PhaseForestMode | None:
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(np.asarray(proposal.initial_normalized_values, dtype=float), lower, upper)
    current = start
    total_nfev = 0
    success = True
    ladder = tuple(sorted({float(value) for value in config.phase_tempering_ladder if 0.0 < float(value) <= 1.0}))
    if not ladder or ladder[-1] < 1.0:
        ladder = ladder + (1.0,)
    for temperature in ladder:
        try:
            initial_profile = evaluator.evaluate_values(_physical_values(coordinates, current))
            failure = np.full(
                initial_profile.normalized_residuals.size
                + len(proposal.integer_vector)
                + (current.size if config.prior_precision > 0.0 else 0),
                1.0e12,
                dtype=float,
            )

            def residual(values: np.ndarray) -> np.ndarray:
                try:
                    profile = evaluator.evaluate_values(_physical_values(coordinates, values))
                    result = forest_residual_vector(
                        profile,
                        components,
                        integer_vector=proposal.integer_vector,
                        normalized_values=values,
                        prior_precision=config.prior_precision,
                        phase_temperature=temperature,
                    )
                    return result if result.shape == failure.shape else failure
                except Exception:
                    return failure

            optimization = least_squares(
                residual,
                current,
                bounds=(lower, upper),
                method="trf",
                jac="2-point",
                diff_step=max(float(config.finite_difference_step), 1.0e-7),
                max_nfev=max(int(config.fixed_path_max_nfev), 1),
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            current = np.asarray(optimization.x, dtype=float)
            total_nfev += int(optimization.nfev)
            success = success and bool(optimization.success)
        except Exception:
            return None
    try:
        profile = evaluator.evaluate_values(_physical_values(coordinates, current))
        shape_chi2, phase_chi2, direct = forest_objective(
            profile,
            components,
            integer_vector=proposal.integer_vector,
        )
        prior_chi2 = float(max(float(config.prior_precision), 0.0) * (current @ current))
        objective = direct + prior_chi2
        jacobian = np.asarray(optimization.jac, dtype=float)
        covariance = _stable_inverse(jacobian.T @ jacobian)
        return PhaseForestMode(
            mode_id=mode_id,
            root_mode_id=root_mode_id,
            parent_mode_id=parent_mode_id,
            proposal_source=proposal.source,
            components=_partition_components(
                proposal.target_partition,
                proposal.integer_vector,
                component_prefix=f"{mode_id}_C",
            ),
            parameter_values=_physical_values(coordinates, current),
            normalized_values=current,
            covariance=covariance,
            shape_chi2=shape_chi2,
            phase_chi2=phase_chi2,
            prior_chi2=prior_chi2,
            objective_chi2=objective,
            log_mass=_laplace_log_mass(objective, covariance),
            posterior_mass=0.0,
            squared_mahalanobis=proposal.squared_mahalanobis,
            merge_score=float(merge_score),
            nfev=total_nfev,
            success=success,
        )
    except Exception:
        return None


def _pair_edge_scores(
    *,
    profile: SessionShapeProfile,
    phase_jacobian: np.ndarray,
    covariance: np.ndarray,
) -> tuple[tuple[float, int, int], ...]:
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    edges: list[tuple[float, int, int]] = []
    for left in range(offsets.size):
        for right in range(left + 1, offsets.size):
            row = phase_jacobian[right] - phase_jacobian[left]
            variance = float(variances[left] + variances[right] + row @ covariance @ row)
            wrapped = float((offsets[right] - offsets[left] + 0.5) % 1.0 - 0.5)
            score = -0.5 * (wrapped * wrapped / max(variance, 1.0e-15) + math.log(max(variance, 1.0e-15)))
            edges.append((score, left, right))
    return tuple(sorted(edges, reverse=True))


def propose_initial_island_partitions(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_mode: Any,
    config: PhaseForestConfig,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Extract candidate islands from invariant phase compatibility only."""

    linearization = linearize_forest(
        evaluator=evaluator,
        coordinates=coordinates,
        components=(),
        normalized_values=invariant_mode.normalized_values,
        step=config.finite_difference_step,
    )
    profile = linearization.profile
    session_count = profile.session_count
    edges = _pair_edge_scores(
        profile=profile,
        phase_jacobian=linearization.session_offset_jacobian,
        covariance=np.asarray(invariant_mode.covariance, dtype=float),
    )
    target_merges = int(config.initial_island_merges)
    if target_merges <= 0:
        target_merges = max(1, min(session_count // 4, session_count - 1))
    target_merges = min(target_merges, session_count - 1)

    partitions: list[tuple[tuple[int, ...], ...]] = [singleton_partition(session_count)]
    edge_orders: list[tuple[tuple[float, int, int], ...]] = [edges]
    for shift in range(min(4, len(edges))):
        edge_orders.append(edges[shift:] + edges[:shift])
    for edge_order in edge_orders:
        current: list[set[int]] = [{index} for index in range(session_count)]
        selected = 0
        for _, left, right in edge_order:
            left_group = next(group for group in current if left in group)
            right_group = next(group for group in current if right in group)
            if left_group is right_group:
                continue
            if len(left_group) + len(right_group) > max(int(config.initial_island_max_size), 2):
                continue
            left_group.update(right_group)
            current.remove(right_group)
            selected += 1
            if selected >= target_merges:
                break
        if selected == target_merges:
            partitions.append(_canonical_partition(current))
    # Prefixes provide topological alternatives with different island sizes.
    for prefix in config.initial_island_edge_prefixes:
        current: list[set[int]] = [{index} for index in range(session_count)]
        for _, left, right in edges[: max(int(prefix), 1)]:
            left_group = next(group for group in current if left in group)
            right_group = next(group for group in current if right in group)
            if left_group is right_group or len(left_group) + len(right_group) > max(int(config.initial_island_max_size), 2):
                continue
            left_group.update(right_group)
            current.remove(right_group)
        partitions.append(_canonical_partition(current))
    unique: list[tuple[tuple[int, ...], ...]] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for partition in partitions:
        if partition in seen:
            continue
        seen.add(partition)
        unique.append(partition)
        if len(unique) >= max(int(config.initial_island_partitions), 1):
            break
    return tuple(unique)


def _merge_partition(
    components: Sequence[PhaseComponent],
    left_index: int,
    right_index: int,
) -> tuple[tuple[int, ...], ...]:
    if left_index == right_index:
        raise ValueError("A phase component cannot be merged with itself.")
    merged = tuple(sorted(set(components[left_index].session_indices) | set(components[right_index].session_indices)))
    result = [component.session_indices for index, component in enumerate(components) if index not in {left_index, right_index}]
    result.append(merged)
    return _canonical_partition(result)


def propose_component_merges(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhaseForestMode,
    config: PhaseForestConfig,
) -> tuple[PhaseMergeProposal, ...]:
    """Rank block merges by the posterior of their new gauge constraint."""

    if mode.component_count <= 1:
        return ()
    linearization = linearize_forest(
        evaluator=evaluator,
        coordinates=coordinates,
        components=mode.components,
        normalized_values=mode.normalized_values,
        step=config.finite_difference_step,
    )
    profile = linearization.profile
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    candidates: list[PhaseMergeProposal] = []
    for left_index in range(mode.component_count):
        left = mode.components[left_index]
        for right_index in range(left_index + 1, mode.component_count):
            right = mode.components[right_index]
            left_anchor = left.anchor_session
            right_anchor = right.anchor_session
            anchor_row = (
                linearization.session_offset_jacobian[right_anchor]
                - linearization.session_offset_jacobian[left_anchor]
            )
            variance = float(variances[left_anchor] + variances[right_anchor] + anchor_row @ mode.covariance @ anchor_row)
            wrapped = float((offsets[right_anchor] - offsets[left_anchor] + 0.5) % 1.0 - 0.5)
            score = -0.5 * (wrapped * wrapped / max(variance, 1.0e-15) + math.log(max(variance, 1.0e-15)))
            candidates.append(
                PhaseMergeProposal(
                    parent_mode_id=mode.mode_id,
                    target_partition=_merge_partition(mode.components, left_index, right_index),
                    merged_component_ids=(left.component_id, right.component_id),
                    merge_score=score,
                )
            )
    candidates.sort(key=lambda item: item.merge_score, reverse=True)
    unique: list[PhaseMergeProposal] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for candidate in candidates:
        if candidate.target_partition in seen:
            continue
        seen.add(candidate.target_partition)
        unique.append(candidate)
        if len(unique) >= max(int(config.merge_candidates_per_parent), 1):
            break
    return tuple(unique)


def _select_modes(
    modes: Sequence[PhaseForestMode],
    *,
    maximum_modes: int,
    maximum_modes_per_topology: int,
    maximum_modes_per_root: int,
    minimum_separation: float,
) -> tuple[PhaseForestMode, ...]:
    ranked = sorted(modes, key=lambda item: (-item.log_mass, item.objective_chi2, item.squared_mahalanobis))
    selected: list[PhaseForestMode] = []
    topology_counts: dict[tuple[tuple[int, ...], ...], int] = {}
    root_counts: dict[str, int] = {}
    def add(mode: PhaseForestMode) -> bool:
        if topology_counts.get(mode.partition_key, 0) >= max(int(maximum_modes_per_topology), 1):
            return False
        if root_counts.get(mode.root_mode_id, 0) >= max(int(maximum_modes_per_root), 1):
            return False
        if any(
            mode.partition_key == other.partition_key
            and np.linalg.norm(mode.normalized_values - other.normalized_values) < max(float(minimum_separation), 0.0)
            and mode.integer_vector == other.integer_vector
            for other in selected
        ):
            return False
        selected.append(mode)
        topology_counts[mode.partition_key] = topology_counts.get(mode.partition_key, 0) + 1
        root_counts[mode.root_mode_id] = root_counts.get(mode.root_mode_id, 0) + 1
        return True

    best_by_topology: dict[tuple[tuple[int, ...], ...], PhaseForestMode] = {}
    best_by_root: dict[str, PhaseForestMode] = {}
    for mode in ranked:
        best_by_topology.setdefault(mode.partition_key, mode)
        best_by_root.setdefault(mode.root_mode_id, mode)
    for mode in sorted(best_by_topology.values(), key=lambda item: (-item.log_mass, item.objective_chi2)):
        add(mode)
        if len(selected) >= max(int(maximum_modes), 1):
            return _normalize_masses(selected)
    for mode in sorted(best_by_root.values(), key=lambda item: (-item.log_mass, item.objective_chi2)):
        add(mode)
        if len(selected) >= max(int(maximum_modes), 1):
            return _normalize_masses(selected)
    for mode in ranked:
        add(mode)
        if len(selected) >= max(int(maximum_modes), 1):
            break
    return _normalize_masses(selected)


def _refit_target_partition(
    *,
    evaluator: Any,
    coordinates: Any,
    target_partition: tuple[tuple[int, ...], ...],
    parent: PhaseForestMode,
    root_modes: Sequence[Any],
    config: PhaseForestConfig,
    merge_score: float,
    source_prefix: str,
) -> list[PhaseForestMode]:
    zero_integer = tuple(0 for component in target_partition for _ in range(len(component) - 1))
    target_components = _partition_components(target_partition, zero_integer, component_prefix="target_")
    starts: list[tuple[Any, np.ndarray, np.ndarray, str, str | None]] = [
        (parent, parent.normalized_values, parent.covariance, "parent", parent.mode_id)
    ]
    for index, root in enumerate(root_modes[: max(int(config.root_modes_per_level), 1)]):
        starts.append((root, root.normalized_values, root.covariance, f"root_{index}", None))
    modes: list[PhaseForestMode] = []
    for seed, normalized_values, covariance, source, parent_id in starts:
        proposals = propose_forest_lattice(
            evaluator=evaluator,
            coordinates=coordinates,
            components=target_components,
            normalized_values=normalized_values,
            covariance=covariance,
            config=config,
            source=f"{source_prefix}:{source}",
        )
        for proposal_index, proposal in enumerate(proposals[: max(int(config.maximum_refit_proposals_per_target), 1)]):
            mode = _refit_forest_proposal(
                evaluator=evaluator,
                coordinates=coordinates,
                components=target_components,
                proposal=proposal,
                root_mode_id=str(seed.root_mode_id if isinstance(seed, PhaseForestMode) else seed.mode_id),
                parent_mode_id=parent_id,
                config=config,
                mode_id=f"{source_prefix}_{proposal_index:03d}_{len(modes):04d}",
                merge_score=merge_score,
            )
            if mode is not None:
                modes.append(mode)
    return modes


def _collapsed_start_proposal(
    *,
    partition: tuple[tuple[int, ...], ...],
    start: CollapsedIslandStart,
    source_prefix: str,
) -> ForestLatticeProposal:
    """Turn a marginalized-island mode into a conditional lattice proposal.

    The collapsed search commits to no integer path. This conversion first
    conditions on one posterior integer mode, then evaluates it with the exact
    joint objective.
    """

    return ForestLatticeProposal(
        target_partition=partition,
        integer_vector=tuple(int(value) for value in start.integer_vector),
        initial_normalized_values=np.asarray(start.normalized_values, dtype=float).copy(),
        squared_mahalanobis=float(start.lattice_chi2),
        lattice_score=float(start.collapsed_objective_chi2),
        kappa=1.0,
        source=f"{source_prefix}:{start.source}",
    )


def build_initial_phase_forest(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    config: PhaseForestConfig | None = None,
) -> tuple[PhaseForestMode, ...]:
    active = config or PhaseForestConfig()
    if not invariant_modes:
        raise ValueError("At least one invariant mode is required.")
    session_count = int(evaluator.evaluate_values(invariant_modes[0].parameter_values).session_count)
    all_modes: list[PhaseForestMode] = []
    for root_index, root in enumerate(invariant_modes):
        partitions = propose_initial_island_partitions(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_mode=root,
            config=active,
        )
        for partition_index, partition in enumerate(partitions):
            if partition == singleton_partition(session_count):
                components = _partition_components(partition, (), component_prefix=f"root{root_index}_")
                all_modes.append(
                    PhaseForestMode(
                        mode_id=f"initial_{root_index:03d}_{partition_index:03d}",
                        root_mode_id=str(root.mode_id),
                        parent_mode_id=None,
                        proposal_source="invariant-singleton",
                        components=components,
                        parameter_values=np.asarray(root.parameter_values, dtype=float).copy(),
                        normalized_values=np.asarray(root.normalized_values, dtype=float).copy(),
                        covariance=np.asarray(root.covariance, dtype=float).copy(),
                        shape_chi2=float(root.shape_chi2),
                        phase_chi2=0.0,
                        prior_chi2=float(root.prior_chi2),
                        objective_chi2=float(root.objective_chi2),
                        log_mass=float(root.log_mass),
                        posterior_mass=0.0,
                        squared_mahalanobis=0.0,
                        merge_score=math.inf,
                        nfev=int(root.nfev),
                        success=bool(root.success),
                    )
                )
                continue
            zero_integer = tuple(0 for component in partition for _ in range(len(component) - 1))
            target_components = _partition_components(partition, zero_integer, component_prefix="initial_")
            proposals = propose_forest_lattice(
                evaluator=evaluator,
                coordinates=coordinates,
                components=target_components,
                normalized_values=root.normalized_values,
                covariance=root.covariance,
                config=active,
                source="invariant-island",
            )
            for proposal_index, proposal in enumerate(proposals[: max(int(active.maximum_refit_proposals_per_target), 1)]):
                mode = _refit_forest_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    components=target_components,
                    proposal=proposal,
                    root_mode_id=str(root.mode_id),
                    parent_mode_id=None,
                    config=replace(active, fixed_path_max_nfev=active.initial_refit_max_nfev),
                    mode_id=f"initial_{root_index:03d}_{partition_index:03d}_{proposal_index:03d}",
                    merge_score=math.inf,
                )
                if mode is not None:
                    all_modes.append(mode)

    # The complete partition defines the collapsed posterior target. Acquire it
    # directly before a chronological or pairwise bridge can
    # fix an incorrect integer prefix.
    if active.collapsed_global_partition:
        full_partition = (tuple(range(session_count)),)
        collapsed_profile_bank = build_collapsed_profile_bank(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            config=active,
        )
        collapsed_starts = discover_collapsed_island_starts(
            evaluator=evaluator,
            coordinates=coordinates,
            partition=full_partition,
            invariant_modes=invariant_modes,
            config=active,
            profile_bank=collapsed_profile_bank,
        )
        target_components = _partition_components(
            full_partition,
            tuple(0 for _ in range(session_count - 1)),
            component_prefix="initial_collapsed_",
        )
        for root_index, root in enumerate(invariant_modes):
            root_proposals = propose_forest_lattice(
                evaluator=evaluator,
                coordinates=coordinates,
                components=target_components,
                normalized_values=root.normalized_values,
                covariance=root.covariance,
                config=active,
                source="initial-global-lattice-root",
            )
            for proposal_index, proposal in enumerate(
                root_proposals[: max(int(active.maximum_refit_proposals_per_target), 1)]
            ):
                mode = _refit_forest_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    components=target_components,
                    proposal=proposal,
                    root_mode_id=str(root.mode_id),
                    parent_mode_id=None,
                    config=replace(active, fixed_path_max_nfev=active.initial_refit_max_nfev),
                    mode_id=(
                        f"initial_global_lattice_{root_index:03d}_"
                        f"{proposal_index:03d}"
                    ),
                    merge_score=math.inf,
                )
                if mode is not None:
                    all_modes.append(mode)
        for collapsed_index, collapsed_start in enumerate(collapsed_starts):
            proposal = _collapsed_start_proposal(
                partition=full_partition,
                start=collapsed_start,
                source_prefix="initial-collapsed-global",
            )
            mode = _refit_forest_proposal(
                evaluator=evaluator,
                coordinates=coordinates,
                components=target_components,
                proposal=proposal,
                root_mode_id=f"collapsed:{collapsed_start.source}",
                parent_mode_id=None,
                config=replace(active, fixed_path_max_nfev=active.initial_refit_max_nfev),
                mode_id=f"initial_collapsed_global_{collapsed_index:03d}",
                merge_score=math.inf,
            )
            if mode is not None:
                all_modes.append(mode)
    # Partial bridge targets have different dimensions and must never compete
    # in one Laplace-mass ranking.  In particular, the singleton invariant
    # target has fewer constraints and would otherwise fill the beam before
    # any phase island can enter the forest.
    selected: list[PhaseForestMode] = []
    for component_count in sorted({mode.component_count for mode in all_modes}, reverse=True):
        level_modes = [mode for mode in all_modes if mode.component_count == component_count]
        selected.extend(
            _select_modes(
                level_modes,
                maximum_modes=max(int(active.forest_beam), 1),
                maximum_modes_per_topology=max(int(active.maximum_modes_per_topology), 1),
                maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
                minimum_separation=active.mode_separation,
            )
        )
    return tuple(selected)


def classify_phase_forest_failure(
    *,
    initial_modes: Sequence[PhaseForestMode],
    final_modes: Sequence[PhaseForestMode],
    stage_history: Sequence[PhaseForestStageSummary],
    session_count: int,
) -> str | None:
    if not initial_modes:
        return "phase-island-discovery"
    if not stage_history or not final_modes:
        return "block-merge-coverage"
    if final_modes[0].component_count == 1 and final_modes[0].components[0].size == session_count:
        return None
    if stage_history[-1].refitted_count == 0:
        return "nonlinear-block-transport"
    if stage_history[-1].unique_topologies <= 1 and session_count > 2:
        return "forest-topology-collapse"
    if any(stage.failure_class == "nonlinear-block-transport" for stage in stage_history):
        return "nonlinear-block-transport"
    return None


def run_bayesian_phase_forest(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    config: PhaseForestConfig | None = None,
) -> PhaseForestResult:
    """Run the partition forest from phase islands to one global component."""

    active = config or PhaseForestConfig()
    if not invariant_modes:
        raise ValueError("At least one invariant mode is required.")
    session_count = int(evaluator.evaluate_values(invariant_modes[0].parameter_values).session_count)
    initial_modes = build_initial_phase_forest(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=invariant_modes,
        config=active,
    )
    levels: dict[int, tuple[PhaseForestMode, ...]] = {}
    for mode in initial_modes:
        levels.setdefault(mode.component_count, tuple())
        levels[mode.component_count] = _select_modes(
            levels[mode.component_count] + (mode,),
            maximum_modes=active.forest_beam,
            maximum_modes_per_topology=active.maximum_modes_per_topology,
            maximum_modes_per_root=active.maximum_modes_per_root,
            minimum_separation=active.mode_separation,
        )
    stage_history: list[PhaseForestStageSummary] = []
    level_portfolios: list[tuple[PhaseForestMode, ...]] = [initial_modes]
    closure_portfolio: tuple[PhaseForestMode, ...] = ()
    maximum_level = max(levels, default=session_count)
    for component_count in range(maximum_level - 1, 0, -1):
        parents = levels.get(component_count + 1, ())
        if not parents:
            continue
        parent_pool = sorted(parents, key=lambda item: (-item.log_mass, item.objective_chi2))[: max(int(active.parent_modes_per_level), 1)]
        merge_proposals: list[PhaseMergeProposal] = []
        for parent in parent_pool:
            merge_proposals.extend(
                propose_component_merges(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    mode=parent,
                    config=active,
                )
            )
        unique_merges: list[PhaseMergeProposal] = []
        seen_merges: set[tuple[tuple[int, ...], ...]] = set()
        for proposal in sorted(merge_proposals, key=lambda item: item.merge_score, reverse=True):
            if proposal.target_partition in seen_merges:
                continue
            seen_merges.add(proposal.target_partition)
            unique_merges.append(proposal)
            if len(unique_merges) >= max(int(active.maximum_target_partitions_per_level), 1):
                break
        candidate_modes: list[PhaseForestMode] = []
        for merge_index, merge in enumerate(unique_merges):
            candidate_parents = [parent for parent in parent_pool if parent.mode_id == merge.parent_mode_id]
            if not candidate_parents:
                candidate_parents = parent_pool[:1]
            for parent in candidate_parents:
                candidate_modes.extend(
                    _refit_target_partition(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        target_partition=merge.target_partition,
                        parent=parent,
                        root_modes=invariant_modes,
                        config=active,
                        merge_score=merge.merge_score,
                        source_prefix=f"stage_{component_count:03d}_merge_{merge_index:03d}",
                    )
                )
        selected = _select_modes(
            candidate_modes,
            maximum_modes=active.forest_beam,
            maximum_modes_per_topology=active.maximum_modes_per_topology,
            maximum_modes_per_root=active.maximum_modes_per_root,
            minimum_separation=active.mode_separation,
        )
        if selected:
            levels[component_count] = _select_modes(
                levels.get(component_count, ()) + selected,
                maximum_modes=active.forest_beam,
                maximum_modes_per_topology=active.maximum_modes_per_topology,
                maximum_modes_per_root=active.maximum_modes_per_root,
                minimum_separation=active.mode_separation,
            )
            selected = levels[component_count]
            level_portfolios.append(selected)
        closure_candidates: list[PhaseForestMode] = []
        parameter_dimension = int(invariant_modes[0].normalized_values.size)
        unresolved_gauges = max(int(component_count) - 1, 0)
        closure_rank = parameter_dimension + max(int(active.global_closure_rank_margin), 0)
        if selected and unresolved_gauges <= closure_rank:
            full_partition = (tuple(range(session_count)),)
            for closure_index, parent in enumerate(
                sorted(selected, key=lambda item: (-item.log_mass, item.objective_chi2))[
                    : max(int(active.global_closure_parent_modes), 1)
                ]
            ):
                closure_candidates.extend(
                    _refit_target_partition(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        target_partition=full_partition,
                        parent=parent,
                        root_modes=invariant_modes if closure_index == 0 else (),
                        config=active,
                        merge_score=0.0,
                        source_prefix=(
                            f"closure_from_{component_count:03d}_{closure_index:03d}"
                        ),
                    )
                )
            closure_portfolio = _select_modes(
                closure_portfolio + tuple(closure_candidates),
                maximum_modes=active.forest_beam,
                maximum_modes_per_topology=active.forest_beam,
                maximum_modes_per_root=active.maximum_modes_per_root,
                minimum_separation=active.mode_separation,
            )
        finite_distances = [mode.squared_mahalanobis for mode in candidate_modes if np.isfinite(mode.squared_mahalanobis)]
        failure = None
        if not unique_merges:
            failure = "block-merge-coverage"
        elif not candidate_modes:
            failure = "nonlinear-block-transport"
        elif not selected:
            failure = "forest-topology-collapse"
        stage_history.append(
            PhaseForestStageSummary(
                stage_index=len(stage_history) + 1,
                component_count=component_count,
                parent_count=len(parents),
                target_count=len(unique_merges),
                proposed_count=len(candidate_modes),
                refitted_count=len(candidate_modes),
                selected_count=len(selected),
                unique_topologies=len({mode.partition_key for mode in candidate_modes}),
                closure_refitted_count=len(closure_candidates),
                minimum_lattice_distance=min(finite_distances) if finite_distances else math.inf,
                best_shape_chi2=min((mode.shape_chi2 for mode in selected), default=math.inf),
                best_phase_chi2=min((mode.phase_chi2 for mode in selected), default=math.inf),
                best_objective_chi2=min((mode.objective_chi2 for mode in selected), default=math.inf),
                best_closure_objective_chi2=min(
                    (mode.objective_chi2 for mode in closure_candidates),
                    default=math.inf,
                ),
                failure_class=failure,
            )
        )
    final_modes = _select_modes(
        levels.get(1, ()) + closure_portfolio,
        maximum_modes=active.forest_beam,
        maximum_modes_per_topology=active.forest_beam,
        maximum_modes_per_root=active.maximum_modes_per_root,
        minimum_separation=active.mode_separation,
    )
    reached = bool(final_modes and final_modes[0].component_count == 1 and final_modes[0].components[0].size == session_count)
    failure = classify_phase_forest_failure(
        initial_modes=initial_modes,
        final_modes=final_modes,
        stage_history=stage_history,
        session_count=session_count,
    )
    return PhaseForestResult(
        session_count=session_count,
        initial_modes=tuple(initial_modes),
        closure_modes=tuple(closure_portfolio),
        final_modes=tuple(final_modes),
        stage_history=tuple(stage_history),
        level_portfolios=tuple(level_portfolios),
        reached_all_sessions=reached,
        failure_class=failure,
    )


__all__ = [
    "CollapsedIslandStart",
    "CollapsedProfilePoint",
    "ForestLatticeProposal",
    "ForestProfileLinearization",
    "PhaseComponent",
    "PhaseForestConfig",
    "PhaseForestMode",
    "PhaseForestResult",
    "PhaseForestStageSummary",
    "PhaseMergeProposal",
    "build_initial_phase_forest",
    "build_collapsed_profile_bank",
    "classify_phase_forest_failure",
    "component_contrast_matrix",
    "discover_collapsed_island_starts",
    "forest_objective",
    "forest_phase_geometry",
    "forest_residual_vector",
    "linearize_forest",
    "propose_component_merges",
    "propose_initial_island_partitions",
    "propose_forest_lattice",
    "run_bayesian_phase_forest",
    "singleton_partition",
]
