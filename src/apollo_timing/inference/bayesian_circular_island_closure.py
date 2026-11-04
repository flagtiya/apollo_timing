"""Global nonlinear closure of trusted phase islands on a phase torus."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from apollo_timing.inference.bayesian_phase_partition_transport import (
    PhasePartitionMode,
    _absolute_step_residual_jacobian,
    _map_partition_integer_vector,
    _normalized_bounds,
    _phase_resolving_steps,
    _physical_values,
    _stable_covariance,
    _stable_inverse,
    _whitened_partition_phase_residual,
    partition_island_gauge_geometry,
    partition_integer_vector_from_complete_path,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    integer_session_offsets_to_gap_wraps,
)


@dataclass(frozen=True)
class CircularIslandClosureConfig:
    prior_precision: float = 1.0
    search_sigma: float = 128.0
    global_restarts: int = 4
    population_multiplier: int = 12
    global_generations: int = 80
    local_starts_per_restart: int = 8
    local_max_nfev: int = 128
    circular_betas: tuple[float, ...] = (
        1.0e-6,
        1.0e-5,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        0.05,
        0.2,
        0.5,
        1.0,
    )
    exact_max_nfev: int = 192
    finite_difference_step: float = 1.0e-6
    phase_step_cycles: float = 0.03
    maximum_lattice_nodes: int = 2_000_000
    maximum_modes: int = 64
    mode_separation: float = 2.0
    random_seed: int = 20260724


@dataclass(frozen=True)
class CircularIslandClosureMode:
    mode_id: str
    source: str
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    covariance: np.ndarray
    integer_offsets: tuple[int, ...]
    wraps: tuple[int, ...]
    acquisition_chi2: float
    shape_chi2: float
    internal_phase_chi2: float
    phase_chi2: float
    objective_chi2: float
    nfev: int
    success: bool


@dataclass(frozen=True)
class CircularIslandClosureResult:
    partition: tuple[tuple[int, ...], ...]
    trusted_parent_id: str
    trusted_internal_integers: tuple[int, ...]
    global_evaluations: int
    local_attempts: int
    modes: tuple[CircularIslandClosureMode, ...]


def circular_phase_residual(
    phase_center: Sequence[float],
    phase_covariance: np.ndarray,
) -> np.ndarray:
    """Return an integer-shift-invariant torus residual for mode discovery.

    Acquisition uses diagonal covariance because correlated sine residuals are
    not invariant to independent integer shifts. Refinement restores the exact
    covariance for every discovered mode.
    """

    center = np.asarray(phase_center, dtype=float)
    covariance = np.asarray(phase_covariance, dtype=float)
    if covariance.shape != (center.size, center.size):
        raise ValueError("Phase covariance must match the island-gauge dimension.")
    variances = np.diag(covariance)
    if not np.all(np.isfinite(center)) or not np.all(
        np.isfinite(variances) & (variances > 0.0)
    ):
        raise ValueError("Island-gauge phases and variances must be finite.")
    return np.sin(math.pi * center) / (math.pi * np.sqrt(variances))


def _acquisition_residual(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    partition: Sequence[Sequence[int]],
    internal_integers: Sequence[int],
    prior_precision: float,
    gauge_beta: float = 1.0,
) -> np.ndarray:
    profile = evaluator.evaluate_values(
        _physical_values(coordinates, normalized_values)
    )
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    internal = _whitened_partition_phase_residual(
        profile,
        partition,
        internal_integers,
    )
    gauge_center, gauge_covariance = partition_island_gauge_geometry(
        profile=profile,
        partition=partition,
        old_integer_vector=internal_integers,
    )
    parts = [
        shape,
        internal,
        math.sqrt(min(max(float(gauge_beta), 0.0), 1.0))
        * circular_phase_residual(gauge_center, gauge_covariance),
    ]
    if float(prior_precision) > 0.0:
        parts.append(
            math.sqrt(float(prior_precision))
            * np.asarray(normalized_values, dtype=float)
        )
    return np.concatenate(parts)


def _deduplicate_starts(
    records: Sequence[tuple[float, np.ndarray, str]],
    *,
    limit: int,
    separation: float,
    metric_cholesky: np.ndarray | None = None,
) -> tuple[tuple[float, np.ndarray, str], ...]:
    selected: list[tuple[float, np.ndarray, str]] = []

    def distance(left: np.ndarray, right: np.ndarray) -> float:
        delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
        if metric_cholesky is not None:
            delta = np.linalg.solve(metric_cholesky, delta)
        return float(np.linalg.norm(delta))

    for record in sorted(records, key=lambda item: item[0]):
        if any(
            distance(record[1], previous[1]) < max(separation, 0.0)
            for previous in selected
        ):
            continue
        selected.append(record)
        if len(selected) >= max(int(limit), 1):
            break
    return tuple(selected)


def _initial_latent_population(
    *,
    dimension: int,
    population_multiplier: int,
    sigma_limit: float,
    seed: int,
) -> np.ndarray:
    """Cover posterior shells while retaining the trusted parent at z=0."""

    count = max(int(population_multiplier) * int(dimension), 5)
    rng = np.random.default_rng(int(seed))
    population = rng.uniform(-sigma_limit, sigma_limit, size=(count, dimension))
    shell_count = max(count // 2, 1)
    shell_scales = np.geomspace(
        min(1.0, sigma_limit),
        sigma_limit,
        num=shell_count,
    )
    shell_values = rng.normal(size=(shell_count, dimension)) * shell_scales[:, None]
    population[:shell_count] = np.clip(shell_values, -sigma_limit, sigma_limit)
    population[0] = 0.0
    return population


def _exact_path_refinement(
    *,
    evaluator: Any,
    coordinates: Any,
    start: np.ndarray,
    integer_offsets: Sequence[int],
    acquisition_chi2: float,
    internal_phase_chi2: float,
    source: str,
    mode_index: int,
    config: CircularIslandClosureConfig,
) -> CircularIslandClosureMode | None:
    lower, upper = _normalized_bounds(coordinates)
    state = np.clip(np.asarray(start, dtype=float), lower, upper)
    path = np.asarray(integer_offsets, dtype=float)
    full_partition = (tuple(range(path.size + 1)),)
    try:
        profile = evaluator.evaluate_values(_physical_values(coordinates, state))
        shape_size = np.asarray(profile.normalized_residuals, dtype=float).size
        if path.shape != (int(profile.session_count) - 1,):
            return None
        steps = _phase_resolving_steps(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=state,
            partition=full_partition,
            config=config,
        )
    except Exception:
        return None
    failure_size = shape_size + path.size

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            local_profile = evaluator.evaluate_values(
                _physical_values(coordinates, values)
            )
            result = np.concatenate(
                (
                    np.asarray(local_profile.normalized_residuals, dtype=float),
                    _whitened_partition_phase_residual(
                        local_profile,
                        full_partition,
                        path,
                    ),
                )
            )
            if result.size == failure_size and np.all(np.isfinite(result)):
                return result
        except Exception:
            pass
        return np.full(failure_size, 1.0e15, dtype=float)

    def jacobian(values: np.ndarray) -> np.ndarray:
        return _absolute_step_residual_jacobian(
            residual,
            values,
            lower=lower,
            upper=upper,
            steps=steps,
        )

    try:
        optimization = least_squares(
            residual,
            state,
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            max_nfev=max(int(config.exact_max_nfev), 1),
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-10,
        )
        normalized = np.clip(np.asarray(optimization.x, dtype=float), lower, upper)
        profile = evaluator.evaluate_values(
            _physical_values(coordinates, normalized)
        )
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        phase = _whitened_partition_phase_residual(
            profile,
            full_partition,
            path,
        )
        shape_chi2 = float(shape @ shape)
        phase_chi2 = float(phase @ phase)
        jac = np.asarray(optimization.jac, dtype=float)
        covariance = _stable_covariance(_stable_inverse(jac.T @ jac), floor=1.0e-8)
        return CircularIslandClosureMode(
            mode_id=f"circular_island_{mode_index:04d}",
            source=source,
            normalized_values=normalized,
            parameter_values=_physical_values(coordinates, normalized),
            covariance=covariance,
            integer_offsets=tuple(int(value) for value in path.tolist()),
            wraps=integer_session_offsets_to_gap_wraps(path),
            acquisition_chi2=float(acquisition_chi2),
            shape_chi2=shape_chi2,
            internal_phase_chi2=float(internal_phase_chi2),
            phase_chi2=phase_chi2,
            objective_chi2=shape_chi2 + phase_chi2,
            nfev=int(getattr(optimization, "nfev", 0)),
            success=bool(optimization.success),
        )
    except Exception:
        return None


def run_bayesian_circular_island_closure(
    *,
    evaluator: Any,
    coordinates: Any,
    trusted_parent: PhasePartitionMode,
    trusted_complete_integer_offsets: Sequence[int] | None = None,
    config: CircularIslandClosureConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CircularIslandClosureResult:
    """Discover all island-gauge aliases jointly on the nonlinear phase torus."""

    active = config or CircularIslandClosureConfig()
    partition = tuple(tuple(int(session) for session in block) for block in trusted_parent.partition)
    if len(partition) <= 1:
        raise ValueError("Circular closure requires at least two phase islands.")
    lower, upper = _normalized_bounds(coordinates)
    parent_state = np.clip(
        np.asarray(trusted_parent.normalized_values, dtype=float),
        lower,
        upper,
    )
    parent_profile = evaluator.evaluate_values(
        _physical_values(coordinates, parent_state)
    )
    parent_covariance = _stable_covariance(
        np.asarray(trusted_parent.covariance, dtype=float)
    )
    parent_cholesky = np.linalg.cholesky(parent_covariance)
    internal_integers = (
        _map_partition_integer_vector(
            parent_profile,
            partition,
            maximum_lattice_nodes=active.maximum_lattice_nodes,
        )
        if trusted_complete_integer_offsets is None
        else partition_integer_vector_from_complete_path(
            partition=partition,
            integer_offsets=trusted_complete_integer_offsets,
        )
    )
    failure_size = (
        np.asarray(parent_profile.normalized_residuals, dtype=float).size
        + internal_integers.size
        + len(partition)
        - 1
        + (parent_state.size if float(active.prior_precision) > 0.0 else 0)
    )
    evaluations = 0

    def residual(values: np.ndarray) -> np.ndarray:
        try:
            result = _acquisition_residual(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=values,
                partition=partition,
                internal_integers=internal_integers,
                prior_precision=active.prior_precision,
                gauge_beta=1.0,
            )
            if result.size == failure_size and np.all(np.isfinite(result)):
                return result
        except Exception:
            pass
        return np.full(failure_size, 1.0e15, dtype=float)

    def objective(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        value = residual(np.asarray(values, dtype=float))
        return float(value @ value)

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

    global_records: list[tuple[float, np.ndarray, str]] = []
    global_records.append((objective(parent_state), parent_state.copy(), "trusted-parent"))
    for restart in range(max(int(active.global_restarts), 1)):
        initial_population = _initial_latent_population(
            dimension=parent_state.size,
            population_multiplier=active.population_multiplier,
            sigma_limit=sigma_limit,
            seed=int(active.random_seed) + restart,
        )
        optimization = differential_evolution(
            latent_objective,
            bounds=latent_bounds,
            strategy="best1bin",
            maxiter=max(int(active.global_generations), 1),
            popsize=max(int(active.population_multiplier), 2),
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=int(active.random_seed) + restart,
            init=initial_population,
            polish=False,
            updating="immediate",
            workers=1,
            tol=0.0,
            atol=0.0,
        )
        population = np.asarray(optimization.population, dtype=float)
        energies = np.asarray(optimization.population_energies, dtype=float)
        order = np.argsort(energies)[: max(int(active.local_starts_per_restart), 1)]
        for rank, index in enumerate(order.tolist()):
            global_records.append(
                (
                    float(energies[index]),
                    latent_to_state(population[index]),
                    f"de_restart{restart}:rank{rank}",
                )
            )
        if progress_callback is not None:
            best_index = int(np.argmin(energies))
            progress_callback(
                {
                    "stage": "global",
                    "restart": restart,
                    "best_acquisition_chi2": float(np.min(energies)),
                    "best_max_abs_sigma": float(
                        np.max(np.abs(population[best_index]))
                    ),
                    "evaluations": evaluations,
                }
            )

    starts = _deduplicate_starts(
        global_records,
        limit=max(int(active.maximum_modes), 1),
        separation=active.mode_separation,
        metric_cholesky=parent_cholesky,
    )
    local_records: list[tuple[float, np.ndarray, str, int]] = []
    circular_betas = tuple(
        sorted(
            {
                min(max(float(beta), 0.0), 1.0)
                for beta in active.circular_betas
                if float(beta) > 0.0
            }
        )
    )
    if not circular_betas or circular_betas[-1] < 1.0:
        circular_betas = (*circular_betas, 1.0)
    for start_index, (_, start, source) in enumerate(starts):
        try:
            state = np.asarray(start, dtype=float).copy()
            total_nfev = 0
            for beta in circular_betas:
                local_steps = _phase_resolving_steps(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    normalized_values=state,
                    partition=(tuple(range(int(parent_profile.session_count))),),
                    config=active,
                )

                def tempered_residual(
                    values: np.ndarray,
                    active_beta: float = beta,
                ) -> np.ndarray:
                    try:
                        result = _acquisition_residual(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            normalized_values=values,
                            partition=partition,
                            internal_integers=internal_integers,
                            prior_precision=active.prior_precision,
                            gauge_beta=active_beta,
                        )
                        if result.size == failure_size and np.all(np.isfinite(result)):
                            return result
                    except Exception:
                        pass
                    return np.full(failure_size, 1.0e15, dtype=float)

                def local_jacobian(values: np.ndarray) -> np.ndarray:
                    return _absolute_step_residual_jacobian(
                        tempered_residual,
                        values,
                        lower=lower,
                        upper=upper,
                        steps=local_steps,
                    )

                optimization = least_squares(
                    tempered_residual,
                    state,
                    bounds=(lower, upper),
                    method="trf",
                    jac=local_jacobian,
                    x_scale="jac",
                    max_nfev=max(int(active.local_max_nfev), 1),
                    ftol=1.0e-10,
                    xtol=1.0e-10,
                    gtol=1.0e-9,
                )
                state = np.clip(
                    np.asarray(optimization.x, dtype=float),
                    lower,
                    upper,
                )
                total_nfev += int(getattr(optimization, "nfev", 0))
            value = residual(state)
            local_records.append(
                (float(value @ value), state, source, total_nfev)
            )
        except Exception:
            continue
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "local",
                    "start_index": start_index,
                    "best_acquisition_chi2": min(
                        record[0] for record in local_records
                    ),
                }
            )

    local_modes = _deduplicate_starts(
        [(value, state, source) for value, state, source, _ in local_records],
        limit=max(int(active.maximum_modes), 1),
        separation=active.mode_separation,
        metric_cholesky=parent_cholesky,
    )
    exact_modes: list[CircularIslandClosureMode] = []
    seen_paths: set[tuple[int, ...]] = set()
    full_partition = (tuple(range(int(parent_profile.session_count))),)
    for acquisition, state, source in local_modes:
        profile = evaluator.evaluate_values(_physical_values(coordinates, state))
        path = _map_partition_integer_vector(
            profile,
            full_partition,
            maximum_lattice_nodes=active.maximum_lattice_nodes,
        )
        path_key = tuple(int(value) for value in path.tolist())
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        internal = _whitened_partition_phase_residual(
            profile,
            partition,
            internal_integers,
        )
        fitted = _exact_path_refinement(
            evaluator=evaluator,
            coordinates=coordinates,
            start=state,
            integer_offsets=path,
            acquisition_chi2=acquisition,
            internal_phase_chi2=float(internal @ internal),
            source=source,
            mode_index=len(exact_modes),
            config=active,
        )
        if fitted is not None:
            exact_modes.append(fitted)
    exact_modes.sort(key=lambda mode: mode.objective_chi2)
    return CircularIslandClosureResult(
        partition=partition,
        trusted_parent_id=trusted_parent.mode_id,
        trusted_internal_integers=tuple(
            int(value) for value in internal_integers.tolist()
        ),
        global_evaluations=evaluations,
        local_attempts=len(local_records),
        modes=tuple(exact_modes[: max(int(active.maximum_modes), 1)]),
    )
