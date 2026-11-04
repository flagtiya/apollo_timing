"""Full-sequence Bayesian transport between continuous and integer modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from apollo_timing.inference.bayesian_circular_island_closure import (
    CircularIslandClosureMode,
    _exact_path_refinement,
)
from apollo_timing.inference.bayesian_phase_partition_transport import (
    _normalized_bounds,
    _physical_values,
    _whitened_partition_phase_residual,
)
from apollo_timing.inference.information_ordered_binary_search import (
    _linearize_profile_in_search_coordinates,
    _relative_phase_covariance,
    posterior_predictive_integer_path_proposals,
)
from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)


@dataclass(frozen=True)
class GlobalPredictiveLatticeConfig:
    rounds: int = 3
    parents_per_round: int = 12
    predictive_betas: tuple[float, ...] = (1.0, 0.1, 0.01)
    integer_candidates_per_parent: int = 64
    refinements_per_parent: int = 8
    maximum_refinements_per_round: int = 64
    maximum_modes: int = 32
    mode_separation: float = 0.02
    prior_precision: float = 1.0
    finite_difference_fraction: float = 1.0e-5
    finite_difference_step: float = 1.0e-6
    exact_max_nfev: int = 256
    phase_step_cycles: float = 0.03
    maximum_lattice_nodes: int = 2_000_000


@dataclass(frozen=True)
class GlobalPredictiveProposal:
    parent_index: int
    beta: float
    normalized_start: np.ndarray
    integer_offsets: tuple[int, ...]
    predictive_chi2: float
    prefit_shape_chi2: float
    prefit_phase_chi2: float
    prefit_objective_chi2: float


@dataclass(frozen=True)
class GlobalPredictiveRoundSummary:
    round_index: int
    parent_count: int
    proposal_count: int
    refinement_count: int
    retained_mode_count: int
    unique_path_count: int
    best_shape_chi2: float
    best_phase_chi2: float
    best_objective_chi2: float


@dataclass(frozen=True)
class GlobalPredictiveLatticeResult:
    modes: tuple[CircularIslandClosureMode, ...]
    round_history: tuple[GlobalPredictiveRoundSummary, ...]
    total_profile_evaluations: int
    total_refinements: int


def _state_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(
        np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))
    )


def _select_seed_states(
    *,
    evaluator: Any,
    coordinates: Any,
    seed_states: Sequence[Sequence[float]],
    count: int,
    separation: float,
) -> tuple[np.ndarray, ...]:
    lower, upper = _normalized_bounds(coordinates)
    records: list[tuple[float, np.ndarray]] = []
    for state in seed_states:
        normalized = np.asarray(state, dtype=float)
        if normalized.shape != lower.shape or not np.all(np.isfinite(normalized)):
            continue
        normalized = np.clip(normalized, lower, upper)
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
        except Exception:
            continue
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        records.append((float(shape @ shape + normalized @ normalized), normalized))
    selected: list[np.ndarray] = []
    for _, state in sorted(records, key=lambda item: item[0]):
        if any(_state_distance(state, previous) < separation for previous in selected):
            continue
        selected.append(state)
        if len(selected) >= max(int(count), 1):
            break
    return tuple(selected)


def _predictive_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    parent_state: np.ndarray,
    parent_index: int,
    config: GlobalPredictiveLatticeConfig,
) -> tuple[GlobalPredictiveProposal, ...]:
    profile, shape_jacobian, offset_jacobian = (
        _linearize_profile_in_search_coordinates(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_search_values=parent_state,
            finite_difference_fraction=config.finite_difference_fraction,
        )
    )
    phase_noise = _relative_phase_covariance(profile)
    phase_cholesky = np.linalg.cholesky(phase_noise)
    relative = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0], dtype=float
    )
    relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
    phase_jacobian_white = np.linalg.solve(phase_cholesky, relative_jacobian)
    nearest = solve_integer_least_squares(
        relative,
        phase_noise,
        candidate_count=1,
        max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
    ).candidates[0].integer_vector
    lower, upper = _normalized_bounds(coordinates)
    full_partition = (tuple(range(int(profile.session_count))),)
    records: list[GlobalPredictiveProposal] = []
    for beta in config.predictive_betas:
        active_beta = min(max(float(beta), 1.0e-12), 1.0)
        prior_jacobian = math.sqrt(max(float(config.prior_precision), 0.0)) * np.eye(
            parent_state.size
        )
        residual_jacobian = np.concatenate(
            (
                shape_jacobian,
                math.sqrt(active_beta) * phase_jacobian_white,
                prior_jacobian,
            ),
            axis=0,
        )
        residual_at_center = np.concatenate(
            (
                np.asarray(profile.normalized_residuals, dtype=float),
                np.zeros(relative.size, dtype=float),
                math.sqrt(max(float(config.prior_precision), 0.0)) * parent_state,
            )
        )
        proposals = posterior_predictive_integer_path_proposals(
            profile=profile,
            current_path=nearest,
            normalized_center=parent_state,
            residual_jacobian=residual_jacobian,
            residual_at_center=residual_at_center,
            slope_residual_count=shape_jacobian.shape[0],
            orthogonal_residual_count=0,
            phase_residual_count=relative.size,
            beta=active_beta,
            lower_bounds=lower,
            upper_bounds=upper,
            candidate_count=max(int(config.integer_candidates_per_parent), 1),
        )
        for proposal in proposals:
            candidate_profile = evaluator.evaluate_values(
                _physical_values(coordinates, proposal.normalized_start)
            )
            shape = np.asarray(candidate_profile.normalized_residuals, dtype=float)
            phase = _whitened_partition_phase_residual(
                candidate_profile,
                full_partition,
                proposal.integer_offsets,
            )
            shape_chi2 = float(shape @ shape)
            phase_chi2 = float(phase @ phase)
            prior_chi2 = max(float(config.prior_precision), 0.0) * float(
                proposal.normalized_start @ proposal.normalized_start
            )
            records.append(
                GlobalPredictiveProposal(
                    parent_index=parent_index,
                    beta=active_beta,
                    normalized_start=np.asarray(proposal.normalized_start, dtype=float),
                    integer_offsets=tuple(int(value) for value in proposal.integer_offsets),
                    predictive_chi2=float(proposal.squared_mahalanobis),
                    prefit_shape_chi2=shape_chi2,
                    prefit_phase_chi2=phase_chi2,
                    prefit_objective_chi2=shape_chi2 + phase_chi2 + prior_chi2,
                )
            )
    return tuple(records)


def select_predictive_proposals(
    proposals: Sequence[GlobalPredictiveProposal],
    *,
    per_parent: int,
    maximum_count: int,
) -> tuple[GlobalPredictiveProposal, ...]:
    """Protect shape, predictive mass, and exact prefit quality per parent."""

    grouped: dict[int, list[GlobalPredictiveProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.parent_index, []).append(proposal)
    protected: list[GlobalPredictiveProposal] = []
    for records in grouped.values():
        candidates = [
            min(records, key=lambda item: item.prefit_objective_chi2),
            min(records, key=lambda item: item.prefit_shape_chi2),
            min(records, key=lambda item: item.predictive_chi2),
        ]
        candidates.extend(sorted(records, key=lambda item: item.prefit_objective_chi2))
        selected_paths: set[tuple[int, ...]] = set()
        for candidate in candidates:
            if candidate.integer_offsets in selected_paths:
                continue
            protected.append(candidate)
            selected_paths.add(candidate.integer_offsets)
            if len(selected_paths) >= max(int(per_parent), 1):
                break
    ordered = sorted(protected, key=lambda item: item.prefit_objective_chi2)
    return tuple(ordered[: max(int(maximum_count), 1)])


def _select_modes(
    modes: Sequence[CircularIslandClosureMode],
    *,
    count: int,
    separation: float,
) -> tuple[CircularIslandClosureMode, ...]:
    selected: list[CircularIslandClosureMode] = []
    for mode in sorted(modes, key=lambda item: item.objective_chi2):
        if any(
            mode.integer_offsets == previous.integer_offsets
            and _state_distance(mode.normalized_values, previous.normalized_values)
            < max(float(separation), 0.0)
            for previous in selected
        ):
            continue
        selected.append(mode)
        if len(selected) >= max(int(count), 1):
            break
    return tuple(selected)


def run_bayesian_global_predictive_lattice(
    *,
    evaluator: Any,
    coordinates: Any,
    seed_states: Sequence[Sequence[float]],
    config: GlobalPredictiveLatticeConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> GlobalPredictiveLatticeResult:
    """Alternate full-data predictive lattice transport and exact refinement."""

    active = config or GlobalPredictiveLatticeConfig()
    parents = _select_seed_states(
        evaluator=evaluator,
        coordinates=coordinates,
        seed_states=seed_states,
        count=active.parents_per_round,
        separation=active.mode_separation,
    )
    if not parents:
        raise ValueError("Global predictive lattice requires a valid seed state.")
    survivors: tuple[CircularIslandClosureMode, ...] = ()
    history: list[GlobalPredictiveRoundSummary] = []
    evaluations = 0
    refinements = 0
    for round_index in range(max(int(active.rounds), 1)):
        proposals: list[GlobalPredictiveProposal] = []
        for parent_index, parent in enumerate(parents):
            generated = _predictive_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                parent_state=np.asarray(parent, dtype=float),
                parent_index=parent_index,
                config=active,
            )
            proposals.extend(generated)
            evaluations += len(generated)
        selected = select_predictive_proposals(
            proposals,
            per_parent=active.refinements_per_parent,
            maximum_count=active.maximum_refinements_per_round,
        )
        fitted: list[CircularIslandClosureMode] = []
        for proposal_index, proposal in enumerate(selected):
            mode = _exact_path_refinement(
                evaluator=evaluator,
                coordinates=coordinates,
                start=proposal.normalized_start,
                integer_offsets=proposal.integer_offsets,
                acquisition_chi2=proposal.prefit_objective_chi2,
                internal_phase_chi2=math.nan,
                source=(
                    f"global_predictive_r{round_index:02d}:"
                    f"p{proposal.parent_index:03d}:b{proposal.beta:g}"
                ),
                mode_index=proposal_index,
                config=active,
            )
            refinements += 1
            if mode is not None:
                fitted.append(mode)
        survivors = _select_modes(
            (*survivors, *fitted),
            count=active.maximum_modes,
            separation=active.mode_separation,
        )
        history.append(
            GlobalPredictiveRoundSummary(
                round_index=round_index,
                parent_count=len(parents),
                proposal_count=len(proposals),
                refinement_count=len(selected),
                retained_mode_count=len(survivors),
                unique_path_count=len({mode.integer_offsets for mode in survivors}),
                best_shape_chi2=min(
                    (mode.shape_chi2 for mode in survivors), default=math.inf
                ),
                best_phase_chi2=min(
                    (mode.phase_chi2 for mode in survivors), default=math.inf
                ),
                best_objective_chi2=min(
                    (mode.objective_chi2 for mode in survivors), default=math.inf
                ),
            )
        )
        if progress_callback is not None:
            progress_callback({"stage": "global-predictive-round", **history[-1].__dict__})
        if not survivors:
            break
        parents = tuple(
            np.asarray(mode.normalized_values, dtype=float)
            for mode in survivors[: max(int(active.parents_per_round), 1)]
        )
    return GlobalPredictiveLatticeResult(
        modes=survivors,
        round_history=tuple(history),
        total_profile_evaluations=evaluations,
        total_refinements=refinements,
    )
