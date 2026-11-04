from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import ndtri
from scipy.stats import qmc

from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)


@dataclass(frozen=True)
class ConditionedPathDirectionConfig:
    """Proposal controls for joint continuous/integer path acquisition."""

    finite_difference_step: float = 2.0e-4
    covariance_scale_factors: tuple[float, ...] = (1.0, 10.0, 100.0)
    phase_precision_factors: tuple[float, ...] = (1.0,)
    include_support_covariance: bool = True
    support_sigma_fraction: float = 0.35
    maximum_directions: int = 6
    line_samples_per_direction: int = 12
    direction_sigma_samples: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
    combination_draws: int = 24
    maximum_candidates: int = 48
    maximum_states_per_path: int = 2
    lattice_candidates_per_channel: int = 128
    maximum_lattice_nodes: int = 2_000_000
    minimum_phase_span_cycles: float = 0.5
    covariance_eigenvalue_floor: float = 1.0e-12
    random_seed: int = 20260725


@dataclass(frozen=True)
class JointContinuousIntegerLinearization:
    """One all-session linear chart over every allowed continuous parameter."""

    parameter_names: tuple[str, ...]
    base_normalized_values: np.ndarray
    shape_residuals: np.ndarray
    shape_jacobian: np.ndarray
    relative_session_offsets: np.ndarray
    relative_session_jacobian: np.ndarray
    relative_offset_covariance: np.ndarray
    prior_mean_displacement: np.ndarray
    local_prior_covariance: np.ndarray
    lower_displacement: np.ndarray
    upper_displacement: np.ndarray


@dataclass(frozen=True)
class ConditionedPathProposal:
    integer_offsets: tuple[int, ...]
    source_groups: tuple[str, ...]
    conditioning_channel: str
    conditioned_displacement: np.ndarray
    conditioned_normalized_values: np.ndarray
    conditioned_covariance: np.ndarray
    phase_precision_factor: float
    channel_objective_chi2: np.ndarray
    channel_closure_chi2: np.ndarray
    channel_exact_closure_chi2: np.ndarray
    channel_shape_chi2: np.ndarray
    channel_phase_chi2: np.ndarray
    channel_prior_chi2: np.ndarray
    selected_channel_index: int


@dataclass(frozen=True)
class ConditionedPathAcquisitionResult:
    channel_names: tuple[str, ...]
    channel_lattice_candidate_counts: tuple[int, ...]
    channel_lattice_visited_nodes: tuple[int, ...]
    generated_unique_path_count: int
    proposals: tuple[ConditionedPathProposal, ...]


@dataclass(frozen=True)
class _ConditionedState:
    displacement: np.ndarray
    covariance: np.ndarray
    objective_chi2: float
    closure_chi2: float
    exact_closure_chi2: float
    shape_chi2: float
    phase_chi2: float
    prior_chi2: float


def _positive_definite(
    values: Sequence[Sequence[float]],
    *,
    dimension: int,
    floor: float,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("A conditioned-path covariance has invalid geometry.")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum = max(float(floor) * scale, np.finfo(float).eps)
    return (eigenvectors * np.maximum(eigenvalues, minimum)) @ eigenvectors.T


def _validate_linearization(
    linearization: JointContinuousIntegerLinearization,
    *,
    floor: float,
) -> tuple[np.ndarray, ...]:
    base = np.asarray(linearization.base_normalized_values, dtype=float)
    shape = np.asarray(linearization.shape_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    relative = np.asarray(linearization.relative_session_offsets, dtype=float)
    relative_jacobian = np.asarray(
        linearization.relative_session_jacobian,
        dtype=float,
    )
    dimension = int(base.size)
    if dimension == 0 or len(linearization.parameter_names) != dimension:
        raise ValueError("A joint chart requires named continuous coordinates.")
    if shape.ndim != 1 or shape_jacobian.shape != (shape.size, dimension):
        raise ValueError("Shape data do not match the joint continuous chart.")
    if relative.ndim != 1 or relative_jacobian.shape != (
        relative.size,
        dimension,
    ):
        raise ValueError("Session phase data do not match the joint chart.")
    arrays = (base, shape, shape_jacobian, relative, relative_jacobian)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("A joint continuous/integer chart must be finite.")
    phase_covariance = _positive_definite(
        linearization.relative_offset_covariance,
        dimension=relative.size,
        floor=floor,
    )
    prior_mean = np.asarray(linearization.prior_mean_displacement, dtype=float)
    lower = np.asarray(linearization.lower_displacement, dtype=float)
    upper = np.asarray(linearization.upper_displacement, dtype=float)
    if (
        prior_mean.shape != (dimension,)
        or lower.shape != (dimension,)
        or upper.shape != (dimension,)
        or not np.all(np.isfinite(prior_mean) & np.isfinite(lower) & np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Joint prior means and support bounds have invalid geometry.")
    prior_mean = np.clip(prior_mean, lower, upper)
    local_covariance = _positive_definite(
        linearization.local_prior_covariance,
        dimension=dimension,
        floor=floor,
    )
    return (
        base,
        shape,
        shape_jacobian,
        relative,
        relative_jacobian,
        phase_covariance,
        prior_mean,
        local_covariance,
        lower,
        upper,
    )


def _prior_channels(
    local_covariance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    config: ConditionedPathDirectionConfig,
) -> tuple[tuple[str, np.ndarray], ...]:
    channels = []
    for scale in config.covariance_scale_factors:
        value = float(scale)
        if np.isfinite(value) and value > 0.0:
            channels.append((f"local-x{value:g}", local_covariance * value * value))
    if config.include_support_covariance:
        fraction = float(config.support_sigma_fraction)
        if not np.isfinite(fraction) or fraction <= 0.0:
            raise ValueError("Support covariance fraction must be positive.")
        support_sigma = np.maximum((upper - lower) * fraction, 1.0e-12)
        channels.append(("bounded-support", np.diag(np.square(support_sigma))))
    if not channels:
        raise ValueError("At least one conditioned-path covariance channel is required.")
    return tuple(channels)


def _shape_chart(
    shape: np.ndarray,
    shape_jacobian: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior_precision = np.linalg.inv(prior_covariance)
    precision = prior_precision + shape_jacobian.T @ shape_jacobian
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    mean = covariance @ (
        prior_precision @ prior_mean - shape_jacobian.T @ shape
    )
    return mean, covariance, precision


def _ray_interval(
    origin: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    minimum = -math.inf
    maximum = math.inf
    for center, vector, low, high in zip(
        origin,
        direction,
        lower,
        upper,
        strict=True,
    ):
        if abs(float(vector)) <= 1.0e-15:
            if center < low or center > high:
                return math.inf, -math.inf
            continue
        left = (low - center) / vector
        right = (high - center) / vector
        minimum = max(minimum, min(left, right))
        maximum = min(maximum, max(left, right))
    return float(minimum), float(maximum)


def _path(values: np.ndarray) -> tuple[int, ...]:
    rounded = np.rint(values)
    if not np.all(np.isfinite(rounded)):
        raise ValueError("A generated complete integer path is not finite.")
    return tuple(int(value) for value in rounded)


def _support_bin(value: float, minimum: float, maximum: float) -> int:
    if maximum <= minimum:
        return 0
    fraction = min(max((value - minimum) / (maximum - minimum), 0.0), 1.0)
    return min(int(4.0 * fraction), 3)


def _condition_path(
    *,
    path: tuple[int, ...],
    shape: np.ndarray,
    shape_jacobian: np.ndarray,
    relative: np.ndarray,
    relative_jacobian: np.ndarray,
    phase_covariance: np.ndarray,
    phase_precision_factor: float,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> _ConditionedState:
    exact_phase_precision = np.linalg.inv(phase_covariance)
    beta = float(phase_precision_factor)
    if not np.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError("A phase precision factor must lie in (0, 1].")
    phase_precision = beta * exact_phase_precision
    prior_precision = np.linalg.inv(prior_covariance)
    prior_root = np.linalg.cholesky(prior_precision).T
    phase_root = np.linalg.cholesky(phase_precision).T
    integers = np.asarray(path, dtype=float)
    design = np.vstack((prior_root, shape_jacobian, phase_root @ relative_jacobian))
    target = np.concatenate(
        (
            prior_root @ prior_mean,
            -shape,
            phase_root @ (integers - relative),
        )
    )
    solution = lsq_linear(
        design,
        target,
        bounds=(lower, upper),
        method="trf",
        tol=1.0e-11,
        lsmr_tol=1.0e-11,
        max_iter=200,
    )
    displacement = np.asarray(solution.x, dtype=float)
    shape_residual = shape + shape_jacobian @ displacement
    phase_residual = relative + relative_jacobian @ displacement - integers
    prior_residual = displacement - prior_mean
    shape_chi2 = float(shape_residual @ shape_residual)
    phase_chi2 = float(phase_residual @ phase_precision @ phase_residual)
    exact_phase_chi2 = float(
        phase_residual @ exact_phase_precision @ phase_residual
    )
    prior_chi2 = float(prior_residual @ prior_precision @ prior_residual)
    precision = (
        prior_precision
        + shape_jacobian.T @ shape_jacobian
        + relative_jacobian.T @ phase_precision @ relative_jacobian
    )
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    closure = shape_chi2 + phase_chi2
    return _ConditionedState(
        displacement=displacement,
        covariance=covariance,
        objective_chi2=closure + prior_chi2,
        closure_chi2=closure,
        exact_closure_chi2=shape_chi2 + exact_phase_chi2,
        shape_chi2=shape_chi2,
        phase_chi2=phase_chi2,
        prior_chi2=prior_chi2,
    )


def acquire_conditioned_path_directions(
    linearization: JointContinuousIntegerLinearization,
    *,
    seed_paths: Sequence[Sequence[int]] = (),
    excluded_paths: Sequence[Sequence[int]] = (),
    config: ConditionedPathDirectionConfig | None = None,
) -> ConditionedPathAcquisitionResult:
    """Acquire complete paths from all-session continuous/integer geometry.

    The returned portfolio is a proposal distribution. Its covariance channels
    and support strata must not be interpreted as calibrated posterior mass.
    """

    active = config or ConditionedPathDirectionConfig()
    floor = max(float(active.covariance_eigenvalue_floor), np.finfo(float).eps)
    (
        base,
        shape,
        shape_jacobian,
        relative,
        relative_jacobian,
        phase_covariance,
        prior_mean,
        local_covariance,
        lower,
        upper,
    ) = _validate_linearization(linearization, floor=floor)
    prior_channels = _prior_channels(
        local_covariance,
        lower,
        upper,
        config=active,
    )
    phase_factors = tuple(
        sorted(
            {
                float(value)
                for value in active.phase_precision_factors
                if np.isfinite(float(value))
                and 0.0 < float(value) <= 1.0
            }
            | {1.0}
        )
    )
    channels = tuple(
        (
            f"{name}:phase-b{beta:g}",
            covariance,
            beta,
        )
        for name, covariance in prior_channels
        for beta in phase_factors
    )
    generated: dict[tuple[int, ...], set[str]] = {}

    def add_path(path: tuple[int, ...], source_group: str) -> None:
        if len(path) != relative.size:
            raise ValueError("Every acquired integer path must cover all sessions.")
        generated.setdefault(path, set()).add(source_group)

    for path_values in seed_paths:
        add_path(tuple(int(value) for value in path_values), "existing-posterior")
    excluded = {
        tuple(int(value) for value in path_values)
        for path_values in excluded_paths
    }
    if any(len(path) != relative.size for path in excluded):
        raise ValueError("Every excluded integer path must cover all sessions.")

    maximum_directions = max(int(active.maximum_directions), 1)
    line_count = max(int(active.line_samples_per_direction), 0)
    minimum_span = max(float(active.minimum_phase_span_cycles), 0.0)
    lattice_candidate_counts = []
    lattice_visited_nodes = []
    for channel_index, (channel_name, covariance, phase_factor) in enumerate(
        channels
    ):
        covariance = _positive_definite(
            covariance,
            dimension=base.size,
            floor=floor,
        )
        mean, shape_covariance, shape_precision = _shape_chart(
            shape,
            shape_jacobian,
            prior_mean,
            covariance,
        )
        mean = np.clip(mean, lower, upper)
        add_path(
            _path(relative + relative_jacobian @ mean),
            f"c{channel_index:02d}:center",
        )
        phase_covariance_from_parameters = (
            relative_jacobian @ shape_covariance @ relative_jacobian.T
        )
        phase_covariance_from_parameters = 0.5 * (
            phase_covariance_from_parameters
            + phase_covariance_from_parameters.T
        )
        integer_center = relative + relative_jacobian @ mean
        integer_covariance = (
            phase_covariance / phase_factor
            + phase_covariance_from_parameters
        )
        integer_covariance = 0.5 * (
            integer_covariance + integer_covariance.T
        )
        try:
            lattice = solve_integer_least_squares(
                integer_center,
                integer_covariance,
                candidate_count=max(
                    int(active.lattice_candidates_per_channel), 1
                ),
                max_visited_nodes=max(int(active.maximum_lattice_nodes), 1),
            )
        except RuntimeError:
            lattice = None
        if lattice is None:
            lattice_candidate_counts.append(0)
            lattice_visited_nodes.append(0)
        else:
            lattice_candidate_counts.append(len(lattice.candidates))
            lattice_visited_nodes.append(int(lattice.visited_nodes))
            count = max(len(lattice.candidates), 1)
            for rank, candidate in enumerate(lattice.candidates):
                rank_bin = min(4 * rank // count, 3)
                add_path(
                    tuple(int(value) for value in candidate.integer_vector),
                    f"c{channel_index:02d}:phase-b{phase_factor:g}:"
                    f"schur-lattice:b{rank_bin:02d}",
                )
        eigenvalues, phase_eigenvectors = np.linalg.eigh(
            phase_covariance_from_parameters
        )
        order = np.argsort(eigenvalues)[::-1]
        directions = []
        for direction_index in order[:maximum_directions]:
            eigenvalue = float(eigenvalues[direction_index])
            if eigenvalue <= floor:
                continue
            phase_vector = phase_eigenvectors[:, direction_index]
            direction = (
                shape_covariance @ relative_jacobian.T @ phase_vector
            ) / math.sqrt(eigenvalue)
            norm = math.sqrt(max(float(direction @ shape_precision @ direction), 0.0))
            if norm <= 0.0:
                continue
            direction /= norm
            phase_direction = relative_jacobian @ direction
            minimum, maximum = _ray_interval(mean, direction, lower, upper)
            if minimum > maximum:
                continue
            phase_span = (maximum - minimum) * float(np.linalg.norm(phase_direction))
            if phase_span < minimum_span:
                continue
            directions.append(direction)
            samples = [0.0]
            samples.extend(
                value
                for sigma in active.direction_sigma_samples
                for value in (-float(sigma), float(sigma))
                if np.isfinite(float(sigma)) and float(sigma) > 0.0
            )
            if line_count:
                samples.extend(
                    np.linspace(minimum, maximum, line_count + 2, dtype=float)[1:-1]
                )
            samples.extend((minimum, maximum))
            for value in samples:
                coefficient = min(max(float(value), minimum), maximum)
                displacement = np.clip(
                    mean + coefficient * direction,
                    lower,
                    upper,
                )
                source = (
                    f"c{channel_index:02d}:d{len(directions) - 1:02d}:"
                    f"b{_support_bin(coefficient, minimum, maximum):02d}"
                )
                add_path(
                    _path(relative + relative_jacobian @ displacement),
                    source,
                )

        combination_count = max(int(active.combination_draws), 0)
        if directions and combination_count:
            basis = np.column_stack(directions)
            exponent = int(math.ceil(math.log2(combination_count)))
            unit = qmc.Sobol(
                d=basis.shape[1] + 1,
                scramble=True,
                seed=max(int(active.random_seed), 0) + 1009 * channel_index,
            ).random_base2(exponent)[:combination_count]
            normals = ndtri(np.clip(unit[:, :-1], 1.0e-9, 1.0 - 1.0e-9))
            for draw_index, (normal, radial_unit) in enumerate(
                zip(normals, unit[:, -1], strict=True)
            ):
                direction = basis @ normal
                norm = math.sqrt(
                    max(float(direction @ shape_precision @ direction), 0.0)
                )
                if norm <= 0.0:
                    continue
                direction /= norm
                minimum, maximum = _ray_interval(mean, direction, lower, upper)
                if minimum > maximum:
                    continue
                coefficient = minimum + float(radial_unit) * (maximum - minimum)
                displacement = np.clip(
                    mean + coefficient * direction,
                    lower,
                    upper,
                )
                source = (
                    f"c{channel_index:02d}:combination:"
                    f"b{_support_bin(coefficient, minimum, maximum):02d}"
                )
                add_path(
                    _path(relative + relative_jacobian @ displacement),
                    source,
                )

    states_by_path = {}
    candidates = []
    for path, source_groups in generated.items():
        states = tuple(
            _condition_path(
                path=path,
                shape=shape,
                shape_jacobian=shape_jacobian,
                relative=relative,
                relative_jacobian=relative_jacobian,
                phase_covariance=phase_covariance,
                phase_precision_factor=phase_factor,
                prior_mean=prior_mean,
                prior_covariance=_positive_definite(
                    covariance,
                    dimension=base.size,
                    floor=floor,
                ),
                lower=lower,
                upper=upper,
            )
            for _, covariance, phase_factor in channels
        )
        states_by_path[path] = states
        objective = np.asarray([state.objective_chi2 for state in states])
        closure = np.asarray([state.closure_chi2 for state in states])
        exact_closure = np.asarray(
            [state.exact_closure_chi2 for state in states]
        )
        shape_scores = np.asarray([state.shape_chi2 for state in states])
        phase_scores = np.asarray([state.phase_chi2 for state in states])
        prior_scores = np.asarray([state.prior_chi2 for state in states])
        for channel_index, state in enumerate(states):
            candidates.append(
                ConditionedPathProposal(
                    integer_offsets=path,
                    source_groups=tuple(sorted(source_groups)),
                    conditioning_channel=channels[channel_index][0],
                    conditioned_displacement=state.displacement,
                    conditioned_normalized_values=base + state.displacement,
                    conditioned_covariance=state.covariance,
                    phase_precision_factor=channels[channel_index][2],
                    channel_objective_chi2=objective,
                    channel_closure_chi2=closure,
                    channel_exact_closure_chi2=exact_closure,
                    channel_shape_chi2=shape_scores,
                    channel_phase_chi2=phase_scores,
                    channel_prior_chi2=prior_scores,
                    selected_channel_index=channel_index,
                )
            )

    target = max(int(active.maximum_candidates), 1)
    path_cap = max(int(active.maximum_states_per_path), 1)
    selected: list[ConditionedPathProposal] = []
    selected_keys = set()
    path_counts: dict[tuple[int, ...], int] = {}

    def add(proposal: ConditionedPathProposal) -> bool:
        key = (proposal.integer_offsets, proposal.selected_channel_index)
        if (
            proposal.integer_offsets in excluded
            or key in selected_keys
            or path_counts.get(proposal.integer_offsets, 0) >= path_cap
        ):
            return False
        selected.append(proposal)
        selected_keys.add(key)
        path_counts[proposal.integer_offsets] = (
            path_counts.get(proposal.integer_offsets, 0) + 1
        )
        return True

    rankings = []
    for channel_index in range(len(channels)):
        channel_candidates = [
            item
            for item in candidates
            if item.selected_channel_index == channel_index
        ]
        rankings.append(
            sorted(
                channel_candidates,
                key=lambda item, index=channel_index: float(
                    item.channel_objective_chi2[index]
                ),
            )
        )
        rankings.append(
            sorted(
                channel_candidates,
                key=lambda item, index=channel_index: float(
                    item.channel_exact_closure_chi2[index]
                ),
            )
        )
        rankings.append(
            sorted(
                channel_candidates,
                key=lambda item, index=channel_index: float(
                    item.channel_closure_chi2[index]
                ),
            )
        )
    for ranking in rankings:
        if ranking:
            add(ranking[0])

    group_best = {}
    for proposal in candidates:
        channel_index = proposal.selected_channel_index
        score = float(proposal.channel_exact_closure_chi2[channel_index])
        for group in proposal.source_groups:
            key = (group, channel_index)
            current = group_best.get(key)
            if current is None or score < current[0]:
                group_best[key] = (score, proposal)
    for _, proposal in sorted(group_best.values(), key=lambda item: item[0]):
        add(proposal)
        if len(selected) >= target:
            break

    rank_length = max((len(ranking) for ranking in rankings), default=0)
    for rank in range(rank_length):
        for ranking in rankings:
            if rank < len(ranking):
                add(ranking[rank])
                if len(selected) >= target:
                    break
        if len(selected) >= target:
            break

    return ConditionedPathAcquisitionResult(
        channel_names=tuple(name for name, _, _ in channels),
        channel_lattice_candidate_counts=tuple(lattice_candidate_counts),
        channel_lattice_visited_nodes=tuple(lattice_visited_nodes),
        generated_unique_path_count=len(generated),
        proposals=tuple(selected[:target]),
    )


__all__ = [
    "ConditionedPathAcquisitionResult",
    "ConditionedPathDirectionConfig",
    "ConditionedPathProposal",
    "JointContinuousIntegerLinearization",
    "acquire_conditioned_path_directions",
]
