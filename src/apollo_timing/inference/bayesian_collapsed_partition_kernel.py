"""Globally propose timing modes with all partition integers marginalized."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from apollo_timing.inference.bayesian_circular_island_closure import (
    CircularIslandClosureMode,
    CircularIslandClosureResult,
    _deduplicate_starts,
    _exact_path_refinement,
    _initial_latent_population,
)
from apollo_timing.inference.bayesian_phase_partition_transport import (
    PhasePartitionMode,
    _normalized_bounds,
    _physical_values,
    _stable_covariance,
    partition_wrapped_log_likelihood,
)
from apollo_timing.inference.collapsed_integer_posterior import (
    conditional_integer_paths,
)


@dataclass(frozen=True)
class CollapsedPartitionKernelConfig:
    prior_precision: float = 1.0
    search_sigma: float = 128.0
    global_restarts: int = 2
    population_multiplier: int = 8
    global_generations: int = 24
    local_starts_per_restart: int = 4
    local_maxiter: int = 24
    hermite_order: int = 32
    maximum_wrap_terms: int = 9
    conditional_paths_per_mode: int = 4
    exact_max_nfev: int = 128
    finite_difference_step: float = 1.0e-6
    phase_step_cycles: float = 0.03
    maximum_lattice_nodes: int = 2_000_000
    maximum_modes: int = 32
    mode_separation: float = 2.0
    use_full_sequence_likelihood: bool = True
    random_seed: int = 20260724


def collapsed_partition_objective(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    partition: Sequence[Sequence[int]],
    prior_precision: float,
    hermite_order: int,
    maximum_wrap_terms: int,
) -> float:
    """Return the exact partition-collapsed negative twice log posterior."""

    normalized = np.asarray(normalized_values, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    wrapped = partition_wrapped_log_likelihood(
        profile,
        partition,
        hermite_order=max(int(hermite_order), 4),
        maximum_wrap_terms=max(int(maximum_wrap_terms), 1),
    )
    return float(
        shape @ shape
        - 2.0 * wrapped
        + max(float(prior_precision), 0.0) * (normalized @ normalized)
    )


def collapsed_likelihood_partition(
    profile: Any,
    proposal_partition: Sequence[Sequence[int]],
    *,
    use_full_sequence_likelihood: bool,
) -> tuple[tuple[int, ...], ...]:
    """Keep the proposal partition out of the production full-data likelihood."""

    if use_full_sequence_likelihood:
        return (tuple(range(int(profile.session_count))),)
    return tuple(
        tuple(int(session) for session in block) for block in proposal_partition
    )


def portfolio_initial_population(
    *,
    parent_state: Sequence[float],
    proposal_states: Sequence[Sequence[float]],
    lower: Sequence[float],
    upper: Sequence[float],
    population_multiplier: int,
    seed: int,
) -> np.ndarray:
    """Build a global DE population from the complete mode portfolio."""

    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    parent = np.asarray(parent_state, dtype=float)
    if parent.shape != lower_values.shape or upper_values.shape != lower_values.shape:
        raise ValueError("Portfolio states and normalized bounds must align.")
    if np.any(lower_values >= upper_values):
        raise ValueError("Normalized bounds must have positive width.")

    states: list[np.ndarray] = []
    for candidate in (parent, *proposal_states):
        values = np.asarray(candidate, dtype=float)
        if values.shape != parent.shape or not np.all(np.isfinite(values)):
            continue
        clipped = np.clip(values, lower_values, upper_values)
        if any(np.allclose(clipped, previous, rtol=0.0, atol=1.0e-12) for previous in states):
            continue
        states.append(clipped)

    dimension = int(parent.size)
    prior_count = max(dimension, int(population_multiplier) * dimension // 2)
    target_count = max(
        len(states) + prior_count,
        int(population_multiplier) * dimension,
        5,
    )
    rng = np.random.default_rng(int(seed))
    while len(states) < target_count:
        states.append(rng.uniform(lower_values, upper_values))
    return np.vstack(states)


def run_bayesian_collapsed_partition_kernel(
    *,
    evaluator: Any,
    coordinates: Any,
    trusted_parent: PhasePartitionMode,
    proposal_states: Sequence[Sequence[float]] = (),
    config: CollapsedPartitionKernelConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CircularIslandClosureResult:
    """Discover continuous modes before conditionally recovering full paths."""

    active = config or CollapsedPartitionKernelConfig()
    partition = tuple(
        tuple(int(session) for session in block) for block in trusted_parent.partition
    )
    if len(partition) <= 1:
        raise ValueError("Collapsed partition discovery requires at least two islands.")
    lower, upper = _normalized_bounds(coordinates)
    parent_state = np.clip(
        np.asarray(trusted_parent.normalized_values, dtype=float), lower, upper
    )
    parent_covariance = _stable_covariance(
        np.asarray(trusted_parent.covariance, dtype=float)
    )
    parent_cholesky = np.linalg.cholesky(parent_covariance)
    parent_profile = evaluator.evaluate_values(
        _physical_values(coordinates, parent_state)
    )
    likelihood_partition = collapsed_likelihood_partition(
        parent_profile,
        partition,
        use_full_sequence_likelihood=active.use_full_sequence_likelihood,
    )
    evaluations = 0

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            result = collapsed_partition_objective(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=values,
                partition=likelihood_partition,
                prior_precision=active.prior_precision,
                hermite_order=active.hermite_order,
                maximum_wrap_terms=active.maximum_wrap_terms,
            )
            if np.isfinite(result):
                return float(result)
        except Exception:
            pass
        return 1.0e30

    sigma_limit = max(float(active.search_sigma), 1.0e-6)
    latent_bounds = tuple((-sigma_limit, sigma_limit) for _ in parent_state)

    def latent_to_state(latent: Sequence[float]) -> np.ndarray:
        return parent_state + parent_cholesky @ np.asarray(latent, dtype=float)

    def latent_objective(latent: np.ndarray) -> float:
        state = latent_to_state(latent)
        if np.any(state < lower) or np.any(state > upper):
            distance = np.maximum(lower - state, 0.0) + np.maximum(state - upper, 0.0)
            return float(1.0e20 + 1.0e20 * (distance @ distance))
        return objective(state)

    records: list[tuple[float, np.ndarray, str]] = [
        (objective(parent_state), parent_state.copy(), "trusted-parent")
    ]
    use_portfolio_population = bool(proposal_states)
    for restart in range(max(int(active.global_restarts), 1)):
        if use_portfolio_population:
            population = portfolio_initial_population(
                parent_state=parent_state,
                proposal_states=proposal_states,
                lower=lower,
                upper=upper,
                population_multiplier=active.population_multiplier,
                seed=int(active.random_seed) + restart,
            )
            optimization_objective = objective
            optimization_bounds = tuple(
                (float(lo), float(hi))
                for lo, hi in zip(lower, upper, strict=True)
            )
            def state_from_population(values: np.ndarray) -> np.ndarray:
                return np.asarray(values, dtype=float)
        else:
            population = _initial_latent_population(
                dimension=parent_state.size,
                population_multiplier=active.population_multiplier,
                sigma_limit=sigma_limit,
                seed=int(active.random_seed) + restart,
            )
            optimization_objective = latent_objective
            optimization_bounds = latent_bounds
            state_from_population = latent_to_state
        optimization = differential_evolution(
            optimization_objective,
            bounds=optimization_bounds,
            strategy="best1bin",
            maxiter=max(int(active.global_generations), 1),
            popsize=max(int(active.population_multiplier), 2),
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=int(active.random_seed) + restart,
            init=population,
            polish=False,
            updating="immediate",
            workers=1,
            tol=0.0,
            atol=0.0,
        )
        final_population = np.asarray(optimization.population, dtype=float)
        energies = np.asarray(optimization.population_energies, dtype=float)
        order = np.argsort(energies)[: max(int(active.local_starts_per_restart), 1)]
        for rank, index in enumerate(order.tolist()):
            records.append(
                (
                    float(energies[index]),
                    state_from_population(final_population[index]),
                    f"collapsed_de_restart{restart}:rank{rank}",
                )
            )
        if progress_callback is not None:
            best = int(np.argmin(energies))
            progress_callback(
                {
                    "stage": "global",
                    "restart": restart,
                    "best_collapsed_objective": float(energies[best]),
                    "best_max_abs_sigma": float(
                        np.max(
                            np.abs(
                                state_from_population(final_population[best])
                                - parent_state
                            )
                        )
                    ),
                    "evaluations": evaluations,
                    "full_sequence_likelihood": active.use_full_sequence_likelihood,
                    "proposal_geometry": (
                        "portfolio-global"
                        if use_portfolio_population
                        else "parent-local"
                    ),
                    "population_size": int(final_population.shape[0]),
                }
            )

    starts = _deduplicate_starts(
        records,
        limit=max(int(active.maximum_modes), 1),
        separation=active.mode_separation,
        metric_cholesky=parent_cholesky,
    )
    local_records: list[tuple[float, np.ndarray, str]] = []
    bounds = tuple((float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True))
    for start_index, (start_value, start, source) in enumerate(starts):
        state = np.clip(np.asarray(start, dtype=float), lower, upper)
        value = float(start_value)
        if int(active.local_maxiter) > 0:
            try:
                optimized = minimize(
                    objective,
                    state,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": max(int(active.local_maxiter), 1),
                        "ftol": 1.0e-10,
                        "maxls": 12,
                    },
                )
                proposed = np.clip(np.asarray(optimized.x, dtype=float), lower, upper)
                proposed_value = objective(proposed)
                if proposed_value <= value:
                    state = proposed
                    value = proposed_value
            except Exception:
                pass
        local_records.append((value, state, source))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "local",
                    "start_index": start_index,
                    "best_collapsed_objective": min(item[0] for item in local_records),
                }
            )

    local_modes = _deduplicate_starts(
        local_records,
        limit=max(int(active.maximum_modes), 1),
        separation=active.mode_separation,
        metric_cholesky=parent_cholesky,
    )
    exact_modes: list[CircularIslandClosureMode] = []
    seen_paths: set[tuple[int, ...]] = set()
    for acquisition, state, source in local_modes:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, state))
            paths = conditional_integer_paths(
                profile,
                candidate_count=max(int(active.conditional_paths_per_mode), 1),
                hermite_order=max(int(active.hermite_order), 4),
                maximum_wrap_terms=max(int(active.maximum_wrap_terms), 1),
            )
        except Exception:
            continue
        for conditional in paths:
            path = tuple(int(value) for value in conditional.integer_offsets)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            fitted = _exact_path_refinement(
                evaluator=evaluator,
                coordinates=coordinates,
                start=state,
                integer_offsets=path,
                acquisition_chi2=acquisition,
                internal_phase_chi2=math.nan,
                source=f"{source}:conditional",
                mode_index=len(exact_modes),
                config=active,
            )
            if fitted is not None:
                exact_modes.append(fitted)
    exact_modes.sort(key=lambda mode: mode.objective_chi2)
    return CircularIslandClosureResult(
        partition=partition,
        trusted_parent_id=trusted_parent.mode_id,
        trusted_internal_integers=(),
        global_evaluations=evaluations,
        local_attempts=len(local_records),
        modes=tuple(exact_modes[: max(int(active.maximum_modes), 1)]),
    )
