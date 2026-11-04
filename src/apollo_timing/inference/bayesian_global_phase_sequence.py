"""Globally scored Bayesian search over phase-island partitions.

Every partial partition is scored through complete integer paths over the same
session set.  A partition therefore controls which integer relations are
already fixed, but it never removes the unresolved sessions from the scoring
likelihood.
"""

from __future__ import annotations

import heapq
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from apollo_timing.inference.bayesian_phase_forest import (
    ForestLatticeProposal,
    PhaseComponent,
    PhaseForestConfig,
    PhaseForestMode,
    _canonical_partition,
    _merge_partition,
    _normalized_bounds,
    _partition_components,
    _physical_values,
    _refit_forest_proposal,
    _stable_covariance,
    _stable_inverse,
    forest_objective,
    linearize_forest,
    propose_forest_lattice,
    propose_initial_island_partitions,
    singleton_partition,
)
from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)


@dataclass(frozen=True)
class GlobalPhaseSequenceConfig:
    """Budgets for the non-greedy global partition frontier."""

    completion_candidates_per_scale: int = 4
    exact_completions_per_node: int = 2
    branch_completions_per_node: int = 3
    maximum_child_partitions: int = 0
    maximum_frontier_expansions: int = 64
    maximum_generated_nodes: int = 512
    maximum_retained_goals: int = 32
    state_separation_decimals: int = 7
    initial_proposals_per_partition: int = 8
    initial_predictive_candidates_per_partition: int = 4
    initial_nonlinear_refits: int = 16
    initial_nonlinear_refits_per_partition: int = 2
    joint_coarsening_fractions: tuple[float, ...] = (0.75, 0.5)
    joint_partitions_per_fraction: int = 2
    joint_rejuvenation_proposals_per_partition: int = 2
    inherited_completions_per_partition: int = 1
    joint_invariant_reacquisition_modes: int = 1


@dataclass(frozen=True)
class GlobalPhaseCompletion:
    """One complete path compatible with a partial forest state."""

    integer_offsets: tuple[int, ...]
    component_gauges: tuple[int, ...]
    normalized_start: np.ndarray
    predictive_chi2: float
    irreducible_chi2: float
    gauge_chi2: float
    log_predictive_density: float
    kappa: float
    source: str

    @property
    def predictive_deviance(self) -> float:
        return float(-2.0 * self.log_predictive_density)


@dataclass(frozen=True)
class GlobalPhaseSequenceNode:
    """A partial partition accompanied by full-data completion evidence."""

    node_id: str
    parent_node_id: str | None
    depth: int
    mode: PhaseForestMode
    completions: tuple[GlobalPhaseCompletion, ...]
    closure_modes: tuple[PhaseForestMode, ...]
    predictive_objective_chi2: float
    best_closure_objective_chi2: float
    priority_chi2: float

    @property
    def partition_key(self) -> tuple[tuple[int, ...], ...]:
        return self.mode.partition_key

    @property
    def component_count(self) -> int:
        return self.mode.component_count


@dataclass(frozen=True)
class GlobalPhaseSequenceResult:
    session_count: int
    initial_nodes: tuple[GlobalPhaseSequenceNode, ...]
    explored_nodes: tuple[GlobalPhaseSequenceNode, ...]
    frontier_nodes: tuple[GlobalPhaseSequenceNode, ...]
    goal_modes: tuple[PhaseForestMode, ...]
    expansion_count: int
    generated_node_count: int
    exhausted_frontier: bool
    failure_class: str | None


@dataclass(frozen=True)
class _PredictiveInitialCandidate:
    root_mode_id: str
    partition: tuple[tuple[int, ...], ...]
    components: tuple[PhaseComponent, ...]
    proposal: ForestLatticeProposal
    provisional_mode: PhaseForestMode
    predictive_objective_chi2: float


def full_phase_contrast_matrix(session_count: int) -> np.ndarray:
    """Contrast every session against session zero."""

    count = int(session_count)
    if count < 1:
        raise ValueError("At least one session is required.")
    contrast = np.zeros((max(count - 1, 0), count), dtype=float)
    for session in range(1, count):
        contrast[session - 1, session] = 1.0
        contrast[session - 1, 0] = -1.0
    return contrast


def component_gauge_basis(
    components: Sequence[PhaseComponent],
    session_count: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return ``N = base + G g`` for complete session pulse offsets.

    ``base`` and the rows of ``G`` exclude reference session zero.  The
    integer relations inside every component are fixed; one integer gauge for
    every non-reference component remains unresolved.
    """

    count = int(session_count)
    if count < 1:
        raise ValueError("At least one session is required.")
    owner = np.full(count, -1, dtype=int)
    internal = np.zeros(count, dtype=np.int64)
    for component_index, component in enumerate(components):
        offsets = {int(component.anchor_session): 0}
        offsets.update(
            {
                int(session): int(value)
                for session, value in zip(
                    component.non_anchor_sessions,
                    component.integer_offsets,
                    strict=True,
                )
            }
        )
        for session in component.session_indices:
            session_index = int(session)
            if not 0 <= session_index < count or owner[session_index] >= 0:
                raise ValueError("Components must partition every session exactly once.")
            owner[session_index] = component_index
            internal[session_index] = offsets[session_index]
    if np.any(owner < 0):
        raise ValueError("Components must cover every session.")

    reference_component = int(owner[0])
    reference_internal = int(internal[0])
    base = internal.copy()
    reference_mask = owner == reference_component
    base[reference_mask] -= reference_internal

    free_components = tuple(
        index for index in range(len(components)) if index != reference_component
    )
    gauge_basis = np.zeros((count, len(free_components)), dtype=np.int64)
    for gauge_index, component_index in enumerate(free_components):
        gauge_basis[owner == component_index, gauge_index] = 1
    gauge_ids = tuple(components[index].component_id for index in free_components)
    return base[1:].astype(float), gauge_basis[1:].astype(float), gauge_ids


def partition_integer_vector_from_global_path(
    partition: Sequence[Sequence[int]],
    integer_offsets: Sequence[int],
) -> tuple[int, ...]:
    """Project a chronological complete path into one forest partition."""

    canonical = _canonical_partition(partition)
    session_count = sum(len(component) for component in canonical)
    values = np.zeros(session_count, dtype=np.int64)
    supplied = tuple(int(value) for value in integer_offsets)
    if len(supplied) != max(session_count - 1, 0):
        raise ValueError("The complete path must exclude only session zero.")
    if supplied:
        values[1:] = supplied
    projected: list[int] = []
    for component in canonical:
        anchor = int(component[0])
        for session in component[1:]:
            projected.append(int(values[int(session)] - values[anchor]))
    return tuple(projected)


def enumerate_partition_children(
    components: Sequence[PhaseComponent],
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate every one-merge child without filtering by local score."""

    children = {
        _merge_partition(components, left, right)
        for left in range(len(components))
        for right in range(left + 1, len(components))
    }
    return tuple(sorted(children))


def select_budgeted_partition_children(
    children: Sequence[tuple[tuple[int, ...], ...]],
    *,
    maximum_count: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Draw a reproducible uniform topology portfolio under an explicit cap."""

    values = tuple(children)
    requested = int(maximum_count)
    if requested <= 0 or requested >= len(values):
        return values
    rng = np.random.default_rng(max(int(seed), 0))
    indices = np.sort(rng.choice(len(values), size=requested, replace=False))
    return tuple(values[int(index)] for index in indices)


def sample_joint_coarsened_partitions(
    components: Sequence[PhaseComponent],
    *,
    fractions: Sequence[float],
    samples_per_fraction: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Sample balanced multi-merge partitions over the complete component set."""

    component_sessions = tuple(component.session_indices for component in components)
    component_count = len(component_sessions)
    requested = max(int(samples_per_fraction), 0)
    if component_count <= 2 or requested == 0:
        return ()
    rng = np.random.default_rng(max(int(seed), 0))
    sampled: set[tuple[tuple[int, ...], ...]] = set()
    for fraction in fractions:
        value = float(fraction)
        if not np.isfinite(value) or not 0.0 < value < 1.0:
            continue
        target_count = max(1, min(component_count - 1, int(round(value * component_count))))
        for _ in range(requested):
            order = rng.permutation(component_count)
            groups: list[list[int]] = [[] for _ in range(target_count)]
            for rank, component_index in enumerate(order):
                groups[rank % target_count].extend(
                    int(session) for session in component_sessions[int(component_index)]
                )
            sampled.add(_canonical_partition(groups))
    return tuple(sorted(sampled))


def propose_global_phase_completions(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhaseForestMode,
    forest_config: PhaseForestConfig,
    sequence_config: GlobalPhaseSequenceConfig | None = None,
    source: str = "global-completion",
) -> tuple[GlobalPhaseCompletion, ...]:
    """Complete unresolved component gauges using the full session sequence.

    The predictive residual always has ``S-1`` dimensions, including after two
    components merge. The old partial Laplace score changed dimension at each
    merge.
    """

    active = sequence_config or GlobalPhaseSequenceConfig()
    lower, upper = _normalized_bounds(coordinates)
    linearization = linearize_forest(
        evaluator=evaluator,
        coordinates=coordinates,
        components=mode.components,
        normalized_values=mode.normalized_values,
        step=forest_config.finite_difference_step,
    )
    profile = linearization.profile
    session_count = int(profile.session_count)
    contrast = full_phase_contrast_matrix(session_count)
    phase_center = contrast @ np.asarray(profile.session_offsets, dtype=float)
    measurement_covariance = contrast @ np.diag(
        np.asarray(profile.session_offset_variances, dtype=float)
    ) @ contrast.T
    phase_jacobian = contrast @ linearization.session_offset_jacobian
    base, gauge_basis, _ = component_gauge_basis(mode.components, session_count)

    unique: dict[tuple[int, ...], GlobalPhaseCompletion] = {}
    for raw_kappa in forest_config.lattice_kappas:
        kappa = max(float(raw_kappa), 1.0e-8)
        continuous_covariance = _stable_covariance(
            kappa * kappa * np.asarray(mode.covariance, dtype=float)
        )
        predictive_covariance = _stable_covariance(
            measurement_covariance
            + phase_jacobian @ continuous_covariance @ phase_jacobian.T
        )
        precision = _stable_inverse(predictive_covariance)
        sign, logdet = np.linalg.slogdet(predictive_covariance)
        if sign <= 0.0 or not np.isfinite(logdet):
            continue
        residual_center = phase_center - base
        candidates: list[tuple[tuple[int, ...], float, float]] = []
        irreducible = 0.0
        if gauge_basis.shape[1]:
            gauge_precision = _stable_covariance(
                gauge_basis.T @ precision @ gauge_basis
            )
            gauge_covariance = _stable_inverse(gauge_precision)
            gauge_center = gauge_covariance @ (
                gauge_basis.T @ precision @ residual_center
            )
            projected_residual = residual_center - gauge_basis @ gauge_center
            irreducible = max(
                float(projected_residual @ precision @ projected_residual),
                0.0,
            )
            try:
                integer_result = solve_integer_least_squares(
                    gauge_center,
                    gauge_covariance,
                    candidate_count=max(
                        int(active.completion_candidates_per_scale), 1
                    ),
                    max_visited_nodes=max(int(forest_config.maximum_lattice_nodes), 1),
                )
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            candidates = [
                (
                    tuple(int(value) for value in candidate.integer_vector),
                    float(candidate.squared_mahalanobis),
                    irreducible + float(candidate.squared_mahalanobis),
                )
                for candidate in integer_result.candidates
            ]
        else:
            fixed_chi2 = float(residual_center @ precision @ residual_center)
            irreducible = fixed_chi2
            candidates = [((), 0.0, fixed_chi2)]

        gain = continuous_covariance @ phase_jacobian.T @ precision
        normalizer = float(
            logdet + phase_center.size * math.log(2.0 * math.pi)
        )
        for gauges, gauge_chi2, decomposed_chi2 in candidates:
            gauge_values = np.asarray(gauges, dtype=float)
            complete = base + gauge_basis @ gauge_values
            integer_offsets = tuple(int(round(value)) for value in complete)
            delta = phase_center - np.asarray(integer_offsets, dtype=float)
            predictive_chi2 = float(delta @ precision @ delta)
            if not np.isclose(
                predictive_chi2,
                decomposed_chi2,
                rtol=1.0e-7,
                atol=1.0e-7,
            ):
                gauge_chi2 = max(predictive_chi2 - irreducible, 0.0)
            start = np.clip(
                np.asarray(mode.normalized_values, dtype=float) - gain @ delta,
                lower,
                upper,
            )
            completion = GlobalPhaseCompletion(
                integer_offsets=integer_offsets,
                component_gauges=gauges,
                normalized_start=start,
                predictive_chi2=predictive_chi2,
                irreducible_chi2=irreducible,
                gauge_chi2=gauge_chi2,
                log_predictive_density=-0.5 * (predictive_chi2 + normalizer),
                kappa=kappa,
                source=f"{source}:kappa={kappa:g}",
            )
            previous = unique.get(integer_offsets)
            if (
                previous is None
                or completion.log_predictive_density
                > previous.log_predictive_density
            ):
                unique[integer_offsets] = completion
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                -item.log_predictive_density,
                item.predictive_chi2,
                item.integer_offsets,
            ),
        )
    )


def _completion_proposal(
    *,
    partition: tuple[tuple[int, ...], ...],
    completion: GlobalPhaseCompletion,
    source: str,
) -> ForestLatticeProposal:
    integer_vector = partition_integer_vector_from_global_path(
        partition,
        completion.integer_offsets,
    )
    return ForestLatticeProposal(
        target_partition=partition,
        integer_vector=integer_vector,
        initial_normalized_values=np.asarray(
            completion.normalized_start, dtype=float
        ).copy(),
        squared_mahalanobis=float(completion.predictive_chi2),
        lattice_score=float(-2.0 * completion.log_predictive_density),
        kappa=float(completion.kappa),
        source=source,
    )


def refit_global_phase_completion(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhaseForestMode,
    completion: GlobalPhaseCompletion,
    forest_config: PhaseForestConfig,
    mode_id: str,
) -> PhaseForestMode | None:
    """Condition and exactly refit one complete path over all sessions."""

    session_count = len(completion.integer_offsets) + 1
    partition = (tuple(range(session_count)),)
    components = _partition_components(
        partition,
        completion.integer_offsets,
        component_prefix=f"{mode_id}_C",
    )
    proposal = _completion_proposal(
        partition=partition,
        completion=completion,
        source=f"global-closure:{completion.source}",
    )
    return _refit_forest_proposal(
        evaluator=evaluator,
        coordinates=coordinates,
        components=components,
        proposal=proposal,
        root_mode_id=mode.root_mode_id,
        parent_mode_id=mode.mode_id,
        config=forest_config,
        mode_id=mode_id,
        merge_score=0.0,
    )


def _build_node(
    *,
    evaluator: Any,
    coordinates: Any,
    mode: PhaseForestMode,
    forest_config: PhaseForestConfig,
    sequence_config: GlobalPhaseSequenceConfig,
    node_id: str,
    parent_node_id: str | None,
    depth: int,
) -> GlobalPhaseSequenceNode | None:
    completions = propose_global_phase_completions(
        evaluator=evaluator,
        coordinates=coordinates,
        mode=mode,
        forest_config=forest_config,
        sequence_config=sequence_config,
        source=f"node:{node_id}",
    )
    if not completions:
        return None
    closure_modes: list[PhaseForestMode] = []
    for completion_index, completion in enumerate(
        completions[: max(int(sequence_config.exact_completions_per_node), 0)]
    ):
        closure = refit_global_phase_completion(
            evaluator=evaluator,
            coordinates=coordinates,
            mode=mode,
            completion=completion,
            forest_config=forest_config,
            mode_id=f"{node_id}_closure_{completion_index:03d}",
        )
        if closure is not None:
            closure_modes.append(closure)
    predictive_objective = float(
        mode.shape_chi2
        + mode.prior_chi2
        + completions[0].predictive_deviance
    )
    best_closure_objective = min(
        (closure.objective_chi2 for closure in closure_modes),
        default=math.inf,
    )
    priority = (
        float(best_closure_objective)
        if np.isfinite(best_closure_objective)
        else predictive_objective
    )
    return GlobalPhaseSequenceNode(
        node_id=node_id,
        parent_node_id=parent_node_id,
        depth=int(depth),
        mode=mode,
        completions=completions,
        closure_modes=tuple(closure_modes),
        predictive_objective_chi2=predictive_objective,
        best_closure_objective_chi2=float(best_closure_objective),
        priority_chi2=priority,
    )


def _child_mode_from_completion(
    *,
    evaluator: Any,
    coordinates: Any,
    parent: GlobalPhaseSequenceNode,
    target_partition: tuple[tuple[int, ...], ...],
    completion: GlobalPhaseCompletion,
    forest_config: PhaseForestConfig,
    mode_id: str,
) -> PhaseForestMode | None:
    integer_vector = partition_integer_vector_from_global_path(
        target_partition,
        completion.integer_offsets,
    )
    components = _partition_components(
        target_partition,
        integer_vector,
        component_prefix=f"{mode_id}_C",
    )
    proposal = _completion_proposal(
        partition=target_partition,
        completion=completion,
        source=f"global-branch:{completion.source}",
    )
    return _refit_forest_proposal(
        evaluator=evaluator,
        coordinates=coordinates,
        components=components,
        proposal=proposal,
        root_mode_id=parent.mode.root_mode_id,
        parent_mode_id=parent.mode.mode_id,
        config=forest_config,
        mode_id=mode_id,
        merge_score=0.0,
    )


def _rejuvenated_child_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    parent: GlobalPhaseSequenceNode,
    invariant_modes: Sequence[Any],
    target_partition: tuple[tuple[int, ...], ...],
    forest_config: PhaseForestConfig,
    maximum_proposals: int,
    mode_prefix: str,
) -> tuple[PhaseForestMode, ...]:
    """Redraw all target-partition integers instead of inheriting a path prefix."""

    zero_integer = tuple(
        0 for component in target_partition for _ in range(len(component) - 1)
    )
    components = _partition_components(
        target_partition,
        zero_integer,
        component_prefix=f"{mode_prefix}_C",
    )
    modes: list[PhaseForestMode] = []
    seeds: list[tuple[str, np.ndarray, np.ndarray, str]] = [
        (
            parent.mode.root_mode_id,
            np.asarray(parent.mode.normalized_values, dtype=float),
            np.asarray(parent.mode.covariance, dtype=float),
            "parent",
        )
    ]
    seeds.extend(
        (
            str(mode.mode_id),
            np.asarray(mode.normalized_values, dtype=float),
            np.asarray(mode.covariance, dtype=float),
            f"invariant-{index:03d}",
        )
        for index, mode in enumerate(invariant_modes)
    )
    for seed_index, (
        root_mode_id,
        normalized_values,
        covariance,
        source,
    ) in enumerate(seeds):
        proposals = propose_forest_lattice(
            evaluator=evaluator,
            coordinates=coordinates,
            components=components,
            normalized_values=normalized_values,
            covariance=covariance,
            config=forest_config,
            source=f"global-rejuvenation:{source}:{parent.node_id}",
        )
        for proposal_index, proposal in enumerate(
            proposals[: max(int(maximum_proposals), 0)]
        ):
            mode = _refit_forest_proposal(
                evaluator=evaluator,
                coordinates=coordinates,
                components=components,
                proposal=proposal,
                root_mode_id=root_mode_id,
                parent_mode_id=parent.mode.mode_id,
                config=forest_config,
                mode_id=(
                    f"{mode_prefix}_rejuvenated_s{seed_index:02d}_"
                    f"{proposal_index:03d}"
                ),
                merge_score=0.0,
            )
            if mode is not None:
                modes.append(mode)
    return tuple(modes)


def _mode_state_key(
    mode: PhaseForestMode,
    *,
    decimals: int,
) -> tuple[object, ...]:
    return (
        mode.partition_key,
        mode.integer_vector,
        mode.root_mode_id,
        tuple(float(value) for value in np.round(mode.normalized_values, decimals)),
    )


def _deduplicate_goal_modes(
    modes: Sequence[PhaseForestMode],
    maximum_count: int,
) -> tuple[PhaseForestMode, ...]:
    ranked = sorted(modes, key=lambda item: (item.objective_chi2, -item.log_mass))
    selected: list[PhaseForestMode] = []
    seen: set[tuple[int, ...]] = set()
    for mode in ranked:
        path = mode.integer_vector
        if path in seen:
            continue
        seen.add(path)
        selected.append(mode)
        if len(selected) >= max(int(maximum_count), 1):
            break
    return tuple(selected)


def _provisional_initial_mode(
    *,
    evaluator: Any,
    coordinates: Any,
    root: Any,
    components: tuple[PhaseComponent, ...],
    proposal: ForestLatticeProposal,
    forest_config: PhaseForestConfig,
    mode_id: str,
) -> PhaseForestMode | None:
    try:
        normalized = np.asarray(proposal.initial_normalized_values, dtype=float)
        profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        shape_chi2, phase_chi2, direct = forest_objective(
            profile,
            components,
            integer_vector=proposal.integer_vector,
        )
        prior_chi2 = float(
            max(float(forest_config.prior_precision), 0.0)
            * (normalized @ normalized)
        )
        covariance = _stable_covariance(
            float(proposal.kappa) ** 2 * np.asarray(root.covariance, dtype=float)
        )
        return PhaseForestMode(
            mode_id=mode_id,
            root_mode_id=str(root.mode_id),
            parent_mode_id=None,
            proposal_source=f"predictive-initial:{proposal.source}",
            components=_partition_components(
                proposal.target_partition,
                proposal.integer_vector,
                component_prefix=f"{mode_id}_C",
            ),
            parameter_values=_physical_values(coordinates, normalized),
            normalized_values=normalized.copy(),
            covariance=covariance,
            shape_chi2=float(shape_chi2),
            phase_chi2=float(phase_chi2),
            prior_chi2=prior_chi2,
            objective_chi2=float(direct + prior_chi2),
            log_mass=float(-0.5 * (direct + prior_chi2)),
            posterior_mass=0.0,
            squared_mahalanobis=float(proposal.squared_mahalanobis),
            merge_score=math.inf,
            nfev=0,
            success=True,
        )
    except Exception:
        return None


def build_predictive_initial_sequence_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    forest_config: PhaseForestConfig,
    sequence_config: GlobalPhaseSequenceConfig,
) -> tuple[PhaseForestMode, ...]:
    """Filter a broad island bank before running nonlinear refits."""

    if not invariant_modes:
        return ()
    session_count = int(
        evaluator.evaluate_values(invariant_modes[0].parameter_values).session_count
    )
    singleton_modes: list[PhaseForestMode] = []
    predictive_candidates: list[_PredictiveInitialCandidate] = []
    for root_index, root in enumerate(invariant_modes):
        singleton_components = _partition_components(
            singleton_partition(session_count),
            (),
            component_prefix=f"predictive_root_{root_index:03d}_",
        )
        singleton_modes.append(
            PhaseForestMode(
                mode_id=f"predictive_root_{root_index:03d}",
                root_mode_id=str(root.mode_id),
                parent_mode_id=None,
                proposal_source="predictive-invariant-singleton",
                components=singleton_components,
                parameter_values=np.asarray(root.parameter_values, dtype=float).copy(),
                normalized_values=np.asarray(root.normalized_values, dtype=float).copy(),
                covariance=np.asarray(root.covariance, dtype=float).copy(),
                shape_chi2=float(root.shape_chi2),
                phase_chi2=0.0,
                prior_chi2=float(root.prior_chi2),
                objective_chi2=float(root.objective_chi2),
                log_mass=float(root.log_mass),
                posterior_mass=float(root.posterior_mass),
                squared_mahalanobis=0.0,
                merge_score=math.inf,
                nfev=int(root.nfev),
                success=bool(root.success),
            )
        )
        partitions = propose_initial_island_partitions(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_mode=root,
            config=forest_config,
        )
        for partition_index, partition in enumerate(partitions):
            if partition == singleton_partition(session_count):
                continue
            zero_integer = tuple(
                0 for component in partition for _ in range(len(component) - 1)
            )
            components = _partition_components(
                partition,
                zero_integer,
                component_prefix=f"predictive_{root_index:03d}_{partition_index:03d}_",
            )
            proposals = propose_forest_lattice(
                evaluator=evaluator,
                coordinates=coordinates,
                components=components,
                normalized_values=root.normalized_values,
                covariance=root.covariance,
                config=forest_config,
                source=f"predictive-root-{root_index:03d}-partition-{partition_index:03d}",
            )
            provisional: list[tuple[float, PhaseForestMode, ForestLatticeProposal]] = []
            for proposal_index, proposal in enumerate(
                proposals[
                    : max(int(sequence_config.initial_proposals_per_partition), 1)
                ]
            ):
                mode = _provisional_initial_mode(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    root=root,
                    components=components,
                    proposal=proposal,
                    forest_config=forest_config,
                    mode_id=(
                        f"predictive_r{root_index:03d}_p{partition_index:03d}_"
                        f"c{proposal_index:03d}"
                    ),
                )
                if mode is None:
                    continue
                provisional.append(
                    (
                        float(mode.objective_chi2 + proposal.lattice_score),
                        mode,
                        proposal,
                    )
                )
            provisional.sort(key=lambda item: item[0])
            for _, mode, proposal in provisional[
                : max(
                    int(
                        sequence_config.initial_predictive_candidates_per_partition
                    ),
                    1,
                )
            ]:
                completions = propose_global_phase_completions(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    mode=mode,
                    forest_config=forest_config,
                    sequence_config=sequence_config,
                    source=f"predictive-initial:{mode.mode_id}",
                )
                if not completions:
                    continue
                predictive_candidates.append(
                    _PredictiveInitialCandidate(
                        root_mode_id=str(root.mode_id),
                        partition=tuple(component.session_indices for component in components),
                        components=components,
                        proposal=proposal,
                        provisional_mode=mode,
                        predictive_objective_chi2=float(
                            mode.shape_chi2
                            + mode.prior_chi2
                            + completions[0].predictive_deviance
                        ),
                    )
                )

    ranked = sorted(
        predictive_candidates,
        key=lambda item: (
            item.predictive_objective_chi2,
            item.provisional_mode.objective_chi2,
            item.proposal.lattice_score,
        ),
    )
    selected: list[_PredictiveInitialCandidate] = []
    selected_mode_ids: set[str] = set()
    per_partition: dict[tuple[str, tuple[tuple[int, ...], ...]], int] = {}
    maximum_refits = max(int(sequence_config.initial_nonlinear_refits), 0)
    per_partition_limit = max(
        int(sequence_config.initial_nonlinear_refits_per_partition), 1
    )
    # First protect the best candidate from each root/partition family.
    best_by_family: dict[
        tuple[str, tuple[tuple[int, ...], ...]], _PredictiveInitialCandidate
    ] = {}
    for candidate in ranked:
        best_by_family.setdefault(
            (candidate.root_mode_id, candidate.partition), candidate
        )
    for candidate in sorted(
        best_by_family.values(),
        key=lambda item: item.predictive_objective_chi2,
    ):
        if len(selected) >= maximum_refits:
            break
        selected.append(candidate)
        selected_mode_ids.add(candidate.provisional_mode.mode_id)
        family = (candidate.root_mode_id, candidate.partition)
        per_partition[family] = 1
    for candidate in ranked:
        if len(selected) >= maximum_refits:
            break
        if candidate.provisional_mode.mode_id in selected_mode_ids:
            continue
        family = (candidate.root_mode_id, candidate.partition)
        if per_partition.get(family, 0) >= per_partition_limit:
            continue
        selected.append(candidate)
        selected_mode_ids.add(candidate.provisional_mode.mode_id)
        per_partition[family] = per_partition.get(family, 0) + 1

    refitted: list[PhaseForestMode] = []
    for candidate_index, candidate in enumerate(selected):
        mode = _refit_forest_proposal(
            evaluator=evaluator,
            coordinates=coordinates,
            components=candidate.components,
            proposal=candidate.proposal,
            root_mode_id=candidate.root_mode_id,
            parent_mode_id=None,
            config=forest_config,
            mode_id=f"predictive_initial_refit_{candidate_index:04d}",
            merge_score=math.inf,
        )
        if mode is not None:
            refitted.append(mode)
    return tuple(singleton_modes + refitted)


def run_global_phase_sequence_search(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[Any],
    forest_config: PhaseForestConfig | None = None,
    sequence_config: GlobalPhaseSequenceConfig | None = None,
    initial_modes: Sequence[PhaseForestMode] | None = None,
    resume_result: GlobalPhaseSequenceResult | None = None,
) -> GlobalPhaseSequenceResult:
    """Explore multiple partitions in one globally scored priority frontier."""

    active_forest = forest_config or PhaseForestConfig()
    active_sequence = sequence_config or GlobalPhaseSequenceConfig()
    if not invariant_modes:
        raise ValueError("At least one invariant mode is required.")
    session_count = int(
        evaluator.evaluate_values(invariant_modes[0].parameter_values).session_count
    )
    frontier: list[tuple[float, int, GlobalPhaseSequenceNode]] = []
    initial_nodes: list[GlobalPhaseSequenceNode] = []
    all_nodes: list[GlobalPhaseSequenceNode] = []
    goals: list[PhaseForestMode] = []
    seen_states: set[tuple[object, ...]] = set()
    serial = 0
    explored: list[GlobalPhaseSequenceNode] = []
    expansion_count = 0
    if resume_result is not None:
        if int(resume_result.session_count) != session_count:
            raise ValueError("The global sequence checkpoint has a different session count.")
        initial_nodes.extend(resume_result.initial_nodes)
        explored.extend(resume_result.explored_nodes)
        all_nodes.extend(resume_result.explored_nodes)
        all_nodes.extend(resume_result.frontier_nodes)
        goals.extend(resume_result.goal_modes)
        expansion_count = int(resume_result.expansion_count)
        for node in all_nodes:
            seen_states.add(
                _mode_state_key(
                    node.mode,
                    decimals=active_sequence.state_separation_decimals,
                )
            )
        for node in resume_result.frontier_nodes:
            heapq.heappush(frontier, (node.priority_chi2, serial, node))
            serial += 1
    else:
        seeds = (
            tuple(initial_modes)
            if initial_modes is not None
            else build_predictive_initial_sequence_modes(
                evaluator=evaluator,
                coordinates=coordinates,
                invariant_modes=invariant_modes,
                forest_config=active_forest,
                sequence_config=active_sequence,
            )
        )
        for seed_index, mode in enumerate(seeds):
            state_key = _mode_state_key(
                mode,
                decimals=active_sequence.state_separation_decimals,
            )
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            node = _build_node(
                evaluator=evaluator,
                coordinates=coordinates,
                mode=mode,
                forest_config=active_forest,
                sequence_config=active_sequence,
                node_id=f"global_root_{seed_index:04d}",
                parent_node_id=None,
                depth=0,
            )
            if node is None:
                continue
            initial_nodes.append(node)
            all_nodes.append(node)
            goals.extend(node.closure_modes)
            heapq.heappush(frontier, (node.priority_chi2, serial, node))
            serial += 1
    while (
        frontier
        and expansion_count < max(int(active_sequence.maximum_frontier_expansions), 0)
        and len(all_nodes) < max(int(active_sequence.maximum_generated_nodes), 1)
    ):
        _, _, parent = heapq.heappop(frontier)
        explored.append(parent)
        if parent.component_count <= 1:
            continue
        expansion_count += 1
        node_seed = int.from_bytes(
            hashlib.sha256(parent.node_id.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        )
        child_partitions = enumerate_partition_children(parent.mode.components)
        if int(active_sequence.maximum_child_partitions) > 0:
            child_partitions = select_budgeted_partition_children(
                child_partitions,
                maximum_count=active_sequence.maximum_child_partitions,
                seed=int(active_forest.random_seed) ^ node_seed,
            )
        joint_partitions = sample_joint_coarsened_partitions(
            parent.mode.components,
            fractions=active_sequence.joint_coarsening_fractions,
            samples_per_fraction=active_sequence.joint_partitions_per_fraction,
            seed=int(active_forest.random_seed) ^ node_seed ^ 0x9E3779B97F4A7C15,
        )
        child_partitions = tuple(
            sorted(set(child_partitions) | set(joint_partitions))
        )
        inherited_count = min(
            max(int(active_sequence.branch_completions_per_node), 1),
            max(int(active_sequence.inherited_completions_per_partition), 0),
        )
        branch_completions = parent.completions[
            : inherited_count
        ]
        stop_generation = False
        for partition_index, target_partition in enumerate(child_partitions):
            child_modes: list[PhaseForestMode] = []
            for completion_index, completion in enumerate(branch_completions):
                mode_id = (
                    f"global_d{parent.depth + 1:03d}_n{len(all_nodes):05d}_"
                    f"p{partition_index:03d}_inherited_{completion_index:03d}"
                )
                child_mode = _child_mode_from_completion(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    parent=parent,
                    target_partition=target_partition,
                    completion=completion,
                    forest_config=active_forest,
                    mode_id=mode_id,
                )
                if child_mode is not None:
                    child_modes.append(child_mode)
            child_modes.extend(
                _rejuvenated_child_modes(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    parent=parent,
                    invariant_modes=invariant_modes[
                        : max(
                            int(
                                active_sequence.joint_invariant_reacquisition_modes
                            ),
                            0,
                        )
                    ],
                    target_partition=target_partition,
                    forest_config=active_forest,
                    maximum_proposals=(
                        active_sequence.joint_rejuvenation_proposals_per_partition
                    ),
                    mode_prefix=(
                        f"global_d{parent.depth + 1:03d}_n{len(all_nodes):05d}_"
                        f"p{partition_index:03d}"
                    ),
                )
            )
            for child_mode_index, child_mode in enumerate(child_modes):
                state_key = _mode_state_key(
                    child_mode,
                    decimals=active_sequence.state_separation_decimals,
                )
                if state_key in seen_states:
                    continue
                seen_states.add(state_key)
                child = _build_node(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    mode=child_mode,
                    forest_config=active_forest,
                    sequence_config=active_sequence,
                    node_id=(
                        f"global_d{parent.depth + 1:03d}_n{len(all_nodes):05d}_"
                        f"p{partition_index:03d}_m{child_mode_index:03d}_node"
                    ),
                    parent_node_id=parent.node_id,
                    depth=parent.depth + 1,
                )
                if child is None:
                    continue
                all_nodes.append(child)
                goals.extend(child.closure_modes)
                heapq.heappush(frontier, (child.priority_chi2, serial, child))
                serial += 1
                if len(all_nodes) >= max(
                    int(active_sequence.maximum_generated_nodes), 1
                ):
                    stop_generation = True
                    break
            if stop_generation:
                break

    retained_goals = _deduplicate_goal_modes(
        goals,
        active_sequence.maximum_retained_goals,
    )
    remaining = tuple(item[2] for item in sorted(frontier))
    failure = None
    if not initial_nodes:
        failure = "global-sequence-initialization"
    elif not retained_goals:
        failure = "global-completion-coverage"
    return GlobalPhaseSequenceResult(
        session_count=session_count,
        initial_nodes=tuple(initial_nodes),
        explored_nodes=tuple(explored),
        frontier_nodes=remaining,
        goal_modes=retained_goals,
        expansion_count=expansion_count,
        generated_node_count=len(all_nodes),
        exhausted_frontier=not frontier,
        failure_class=failure,
    )
