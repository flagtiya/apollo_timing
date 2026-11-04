from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
)
from timing_accelerator.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from timing_accelerator.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
)


@dataclass(frozen=True)
class NonlinearMixedIntegerConfig:
    """Controls for trust-region Bayesian continuous/integer reassignment."""

    iteration_count: int = 8
    population_size: int = 8
    parents_per_iteration: int = 4
    trust_region_radii: tuple[float, ...] = (
        0.002,
        0.005,
        0.01,
        0.02,
        0.05,
        0.10,
    )
    step_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    lattice_candidates_per_region: int = 8
    exact_integer_candidates: int = 4
    maximum_lattice_nodes: int = 2_000_000
    minimum_state_separation: float = 1.0e-7
    minimum_improvement: float = 1.0e-6
    stalled_iteration_limit: int = 3


@dataclass(frozen=True)
class ProfiledIntegerPath:
    integer_offsets: tuple[int, ...]
    phase_chi2: float


@dataclass(frozen=True)
class NonlinearMixedIntegerState:
    state_id: str
    root_id: str
    parent_id: str | None
    normalized_values: np.ndarray
    integer_offsets: tuple[int, ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    proposal_path: tuple[int, ...] | None
    proposal_trust_radius: float | None
    proposal_step_fraction: float | None
    predicted_total_chi2: float | None
    path_reassigned: bool


@dataclass(frozen=True)
class NonlinearMixedIntegerIteration:
    iteration_index: int
    parent_count: int
    proposal_count: int
    selected_count: int
    selected_path_count: int
    reassigned_proposal_count: int
    best_total_chi2: float
    best_shape_chi2: float
    best_phase_chi2: float
    improved: bool


@dataclass(frozen=True)
class NonlinearMixedIntegerResult:
    final_states: tuple[NonlinearMixedIntegerState, ...]
    iteration_history: tuple[NonlinearMixedIntegerIteration, ...]
    completed_iterations: int


def _relative_phase_terms(
    profile: SessionShapeProfile,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    if offsets.ndim != 1 or variances.shape != offsets.shape or offsets.size < 2:
        raise ValueError("A complete integer profile requires at least two sessions.")
    if not np.all(np.isfinite(offsets)) or not np.all(
        np.isfinite(variances) & (variances > 0.0)
    ):
        raise ValueError("Session offsets and variances must be finite and valid.")
    relative = offsets[1:] - offsets[0]
    covariance = np.diag(variances[1:])
    covariance += float(variances[0])
    return relative, 0.5 * (covariance + covariance.T)


def profile_complete_integer_paths(
    profile: SessionShapeProfile,
    *,
    candidate_count: int = 4,
    maximum_lattice_nodes: int = 2_000_000,
) -> tuple[ProfiledIntegerPath, ...]:
    """Solve the complete exact integer lattice at one nonlinear point."""

    relative, covariance = _relative_phase_terms(profile)
    lattice = solve_integer_least_squares(
        relative,
        covariance,
        candidate_count=max(int(candidate_count), 1),
        max_visited_nodes=max(int(maximum_lattice_nodes), 1),
    )
    return tuple(
        ProfiledIntegerPath(
            integer_offsets=tuple(
                int(value) for value in candidate.integer_vector
            ),
            phase_chi2=float(candidate.squared_mahalanobis),
        )
        for candidate in lattice.candidates
    )


def _profile_shape_chi2(profile: SessionShapeProfile) -> float:
    residuals = np.asarray(profile.normalized_residuals, dtype=float)
    if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
        raise ValueError("Shape residuals must be a finite vector.")
    return float(residuals @ residuals)


def _exact_states_at_point(
    *,
    state_prefix: str,
    root_id: str,
    parent_id: str | None,
    normalized_values: np.ndarray,
    profile: SessionShapeProfile,
    candidate_count: int,
    maximum_lattice_nodes: int,
    proposal_path: tuple[int, ...] | None = None,
    proposal_trust_radius: float | None = None,
    proposal_step_fraction: float | None = None,
    predicted_total_chi2: float | None = None,
) -> tuple[NonlinearMixedIntegerState, ...]:
    shape_chi2 = _profile_shape_chi2(profile)
    paths = profile_complete_integer_paths(
        profile,
        candidate_count=candidate_count,
        maximum_lattice_nodes=maximum_lattice_nodes,
    )
    return tuple(
        NonlinearMixedIntegerState(
            state_id=f"{state_prefix}_p{path_index:03d}",
            root_id=str(root_id),
            parent_id=parent_id,
            normalized_values=np.asarray(normalized_values, dtype=float).copy(),
            integer_offsets=path.integer_offsets,
            shape_chi2=shape_chi2,
            phase_chi2=path.phase_chi2,
            total_chi2=shape_chi2 + path.phase_chi2,
            proposal_path=proposal_path,
            proposal_trust_radius=proposal_trust_radius,
            proposal_step_fraction=proposal_step_fraction,
            predicted_total_chi2=predicted_total_chi2,
            path_reassigned=bool(
                proposal_path is not None
                and path.integer_offsets != proposal_path
            ),
        )
        for path_index, path in enumerate(paths)
    )


def _linear_predicted_chi2(
    linearization: SpinOrbitLinearization,
    displacement: np.ndarray,
    integer_offsets: Sequence[int],
) -> float:
    shape = np.asarray(linearization.profile.normalized_residuals, dtype=float)
    shape = shape + np.asarray(linearization.shape_jacobian, dtype=float) @ displacement
    phase = np.asarray(linearization.relative_session_offsets, dtype=float)
    phase = phase + (
        np.asarray(linearization.relative_session_jacobian, dtype=float)
        @ displacement
    )
    phase -= np.asarray(integer_offsets, dtype=float)
    covariance = np.asarray(linearization.relative_offset_covariance, dtype=float)
    return float(shape @ shape + phase @ np.linalg.solve(covariance, phase))


def _select_states(
    states: Sequence[NonlinearMixedIntegerState],
    *,
    maximum_count: int,
    minimum_separation: float,
) -> tuple[NonlinearMixedIntegerState, ...]:
    ranked = sorted(
        states,
        key=lambda state: (
            state.total_chi2,
            state.shape_chi2,
            state.integer_offsets,
            state.state_id,
        ),
    )
    selected: list[NonlinearMixedIntegerState] = []
    separation = max(float(minimum_separation), 0.0)
    for state in ranked:
        if any(
            state.integer_offsets == retained.integer_offsets
            and np.linalg.norm(
                state.normalized_values - retained.normalized_values
            ) < separation
            for retained in selected
        ):
            continue
        selected.append(state)
        if len(selected) >= max(int(maximum_count), 1):
            break
    return tuple(selected)


def run_nonlinear_mixed_integer_population(
    *,
    evaluate_profile: Callable[[np.ndarray], SessionShapeProfile],
    linearize: Callable[[np.ndarray], SpinOrbitLinearization],
    initial_values: Sequence[Sequence[float]],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    config: NonlinearMixedIntegerConfig | None = None,
    checkpoint_callback: Callable[
        [NonlinearMixedIntegerIteration, tuple[NonlinearMixedIntegerState, ...]],
        None,
    ]
    | None = None,
) -> NonlinearMixedIntegerResult:
    """Alternate Bayesian integer marginalization and exact nonlinear profiling.

    A path is used only to define a local conditional continuous direction. At
    every exact trial point the complete integer lattice is solved again before
    the state can enter the population. This prevents a linearized path from
    remaining frozen across a nonlinear parameter update.
    """

    active = config or NonlinearMixedIntegerConfig()
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape or np.any(lower >= upper):
        raise ValueError("Mixed-integer bounds must be finite and ordered.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("Mixed-integer bounds must be finite and ordered.")

    initial_states: list[NonlinearMixedIntegerState] = []
    for root_index, values in enumerate(initial_values):
        point = np.asarray(values, dtype=float)
        if point.shape != lower.shape:
            raise ValueError("Every mixed-integer initial point must match the bounds.")
        point = np.clip(point, lower, upper)
        profile = evaluate_profile(point)
        initial_states.extend(
            _exact_states_at_point(
                state_prefix=f"initial_{root_index:04d}",
                root_id=f"root_{root_index:04d}",
                parent_id=None,
                normalized_values=point,
                profile=profile,
                candidate_count=active.exact_integer_candidates,
                maximum_lattice_nodes=active.maximum_lattice_nodes,
            )
        )
    current = _select_states(
        initial_states,
        maximum_count=active.population_size,
        minimum_separation=active.minimum_state_separation,
    )
    if not current:
        raise RuntimeError("No valid initial mixed-integer state was generated.")

    radii = tuple(
        sorted(
            {
                float(value)
                for value in active.trust_region_radii
                if np.isfinite(float(value)) and float(value) > 0.0
            }
        )
    )
    fractions = tuple(
        sorted(
            {
                float(value)
                for value in active.step_fractions
                if np.isfinite(float(value)) and 0.0 < float(value) <= 1.0
            }
        )
    )
    if not radii or not fractions:
        raise ValueError("At least one trust radius and step fraction are required.")

    history: list[NonlinearMixedIntegerIteration] = []
    stalled = 0
    for iteration_index in range(max(int(active.iteration_count), 0)):
        previous_best = min(state.total_chi2 for state in current)
        parents = tuple(
            sorted(current, key=lambda state: state.total_chi2)[
                : max(int(active.parents_per_iteration), 1)
            ]
        )
        proposals: list[NonlinearMixedIntegerState] = []
        seen_points: set[tuple[int, ...]] = set()
        for parent_index, parent in enumerate(parents):
            try:
                local = linearize(parent.normalized_values)
            except Exception:
                continue
            for radius_index, radius in enumerate(radii):
                prior_sigma = np.minimum(
                    np.full(lower.size, radius, dtype=float),
                    0.5 * (upper - lower),
                )
                try:
                    marginal = marginalize_spin_and_integer_paths(
                        local,
                        spin_prior_mean=np.zeros(lower.size, dtype=float),
                        spin_prior_covariance=np.diag(np.square(prior_sigma)),
                        config=SpinIntegerMarginalConfig(
                            initial_lattice_candidates=max(
                                int(active.lattice_candidates_per_region), 1
                            ),
                            maximum_lattice_candidates=max(
                                int(active.lattice_candidates_per_region), 1
                            ),
                            lattice_log_mass_tolerance=0.0,
                            maximum_lattice_nodes=max(
                                int(active.maximum_lattice_nodes), 1
                            ),
                        ),
                    )
                except Exception:
                    continue
                path_modes = marginal.path_modes[
                    : max(int(active.lattice_candidates_per_region), 1)
                ]
                for path_index, path_mode in enumerate(path_modes):
                    full_step = np.asarray(
                        path_mode.spin_posterior_mean,
                        dtype=float,
                    )
                    for fraction_index, fraction in enumerate(fractions):
                        displacement = fraction * full_step
                        trial = np.clip(
                            parent.normalized_values + displacement,
                            lower,
                            upper,
                        )
                        key = tuple(
                            int(value)
                            for value in np.rint(trial * 1.0e10).astype(np.int64)
                        )
                        if key in seen_points:
                            continue
                        seen_points.add(key)
                        actual_displacement = trial - parent.normalized_values
                        proposal_path = tuple(
                            int(value) for value in path_mode.integer_offsets
                        )
                        predicted = _linear_predicted_chi2(
                            local,
                            actual_displacement,
                            proposal_path,
                        )
                        try:
                            profile = evaluate_profile(trial)
                            proposals.extend(
                                _exact_states_at_point(
                                    state_prefix=(
                                        f"iter_{iteration_index:03d}_"
                                        f"parent_{parent_index:03d}_"
                                        f"radius_{radius_index:03d}_"
                                        f"path_{path_index:03d}_"
                                        f"fraction_{fraction_index:03d}"
                                    ),
                                    root_id=parent.root_id,
                                    parent_id=parent.state_id,
                                    normalized_values=trial,
                                    profile=profile,
                                    candidate_count=active.exact_integer_candidates,
                                    maximum_lattice_nodes=active.maximum_lattice_nodes,
                                    proposal_path=proposal_path,
                                    proposal_trust_radius=radius,
                                    proposal_step_fraction=fraction,
                                    predicted_total_chi2=predicted,
                                )
                            )
                        except Exception:
                            continue
        current = _select_states(
            tuple(current) + tuple(proposals),
            maximum_count=active.population_size,
            minimum_separation=active.minimum_state_separation,
        )
        best = min(current, key=lambda state: state.total_chi2)
        improved = bool(
            best.total_chi2
            < previous_best - max(float(active.minimum_improvement), 0.0)
        )
        stalled = 0 if improved else stalled + 1
        summary = NonlinearMixedIntegerIteration(
            iteration_index=iteration_index,
            parent_count=len(parents),
            proposal_count=len(proposals),
            selected_count=len(current),
            selected_path_count=len(
                {state.integer_offsets for state in current}
            ),
            reassigned_proposal_count=sum(
                state.path_reassigned for state in proposals
            ),
            best_total_chi2=best.total_chi2,
            best_shape_chi2=best.shape_chi2,
            best_phase_chi2=best.phase_chi2,
            improved=improved,
        )
        history.append(summary)
        if checkpoint_callback is not None:
            checkpoint_callback(summary, current)
        if stalled >= max(int(active.stalled_iteration_limit), 1):
            break

    return NonlinearMixedIntegerResult(
        final_states=current,
        iteration_history=tuple(history),
        completed_iterations=len(history),
    )


__all__ = [
    "NonlinearMixedIntegerConfig",
    "NonlinearMixedIntegerIteration",
    "NonlinearMixedIntegerResult",
    "NonlinearMixedIntegerState",
    "ProfiledIntegerPath",
    "profile_complete_integer_paths",
    "run_nonlinear_mixed_integer_population",
]
