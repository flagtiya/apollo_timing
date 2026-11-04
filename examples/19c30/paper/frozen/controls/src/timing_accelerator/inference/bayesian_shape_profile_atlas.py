from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import qmc

from timing_accelerator.inference.bayesian_conditioned_path_acquisition import (
    ConditionedPathDirectionConfig,
)
from timing_accelerator.inference.bayesian_global_lattice_mode_tracker import (
    LatticeProposal,
    _physical_values,
    exact_fixed_path_chi2,
    finite_difference_profile,
)


@dataclass(frozen=True)
class ShapeProfileAtlasConfig:
    """Controls for direct exploration of a wrap-invariant shape manifold."""

    finite_difference_step: float = 2.0e-4
    maximum_weak_dimensions: int = 3
    weak_singular_value_ratio: float = 30.0
    direction_grid_points: int = 17
    combination_draws: int = 64
    support_inset_fraction: float = 1.0e-7
    profile_max_iterations: int = 120
    profile_constraint_tolerance: float = 2.0e-6
    maximum_shape_delta_chi2: float = 100.0
    minimum_point_separation: float = 2.0e-3
    prior_precision: float = 0.0
    random_seed: int = 20260811


@dataclass(frozen=True)
class ShapeProfileGeometry:
    center_normalized_values: np.ndarray
    shape_chi2: float
    shape_jacobian: np.ndarray
    singular_values: np.ndarray
    weak_basis: np.ndarray
    strong_basis: np.ndarray
    weak_support_lower: np.ndarray
    weak_support_upper: np.ndarray


@dataclass(frozen=True)
class ShapeProfileAtlasPoint:
    point_id: str
    root_id: str
    source: str
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    weak_coordinates: np.ndarray
    shape_chi2: float
    prior_chi2: float
    objective_chi2: float
    profile_success: bool
    profile_iterations: int
    profile_constraint_error: float


@dataclass(frozen=True)
class ShapeProfilePathCandidate:
    candidate_id: str
    atlas_point_id: str
    proposal: LatticeProposal
    exact_start_shape_chi2: float
    exact_start_phase_chi2: float
    exact_start_closure_chi2: float
    acquisition_linear_chi2: float


def _normalized_bounds(coordinates: Any, inset_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if (
        lower.ndim != 1
        or upper.shape != lower.shape
        or not np.all(np.isfinite(lower) & np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Shape-profile atlas requires finite normalized bounds.")
    fraction = max(float(inset_fraction), 0.0)
    inset = np.maximum((upper - lower) * fraction, 1.0e-12)
    return lower + inset, upper - inset


def _ray_interval(
    center: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, float]:
    minimum = -math.inf
    maximum = math.inf
    for value, component, low, high in zip(
        center, direction, lower, upper, strict=True
    ):
        if abs(float(component)) <= 1.0e-15:
            if value < low or value > high:
                return math.inf, -math.inf
            continue
        left = (low - value) / component
        right = (high - value) / component
        minimum = max(minimum, min(left, right))
        maximum = min(maximum, max(left, right))
    return float(minimum), float(maximum)


def shape_profile_geometry(
    *,
    evaluator: Any,
    coordinates: Any,
    center_normalized_values: Sequence[float],
    config: ShapeProfileAtlasConfig | None = None,
) -> ShapeProfileGeometry:
    """Identify weak tangent coordinates from the exact shape Jacobian."""

    active = config or ShapeProfileAtlasConfig()
    center = np.asarray(center_normalized_values, dtype=float)
    lower, upper = _normalized_bounds(
        coordinates,
        active.support_inset_fraction,
    )
    if center.shape != lower.shape or np.any(center < lower) or np.any(center > upper):
        raise ValueError("A shape-profile root lies outside normalized support.")
    linearization = finite_difference_profile(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        step=max(float(active.finite_difference_step), 1.0e-7),
    )
    jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    _, singular_values, right = np.linalg.svd(jacobian, full_matrices=True)
    dimension = center.size
    padded = np.zeros(dimension, dtype=float)
    padded[: singular_values.size] = singular_values
    positive = padded[padded > np.finfo(float).tiny]
    weakest = float(np.min(positive)) if positive.size else 0.0
    threshold = weakest * max(float(active.weak_singular_value_ratio), 1.0)
    weak_count = int(np.count_nonzero(padded <= threshold))
    weak_count = min(
        max(weak_count, 1),
        max(int(active.maximum_weak_dimensions), 1),
        dimension,
    )
    weak_basis = np.asarray(right[-weak_count:].T, dtype=float)
    strong_basis = np.asarray(right[:-weak_count].T, dtype=float)
    support_lower = np.empty(weak_count, dtype=float)
    support_upper = np.empty(weak_count, dtype=float)
    for index in range(weak_count):
        support_lower[index], support_upper[index] = _ray_interval(
            center,
            weak_basis[:, index],
            lower,
            upper,
        )
    if np.any(support_lower >= support_upper):
        raise RuntimeError("Shape weak directions have no interior support.")
    return ShapeProfileGeometry(
        center_normalized_values=center.copy(),
        shape_chi2=float(linearization.profile.chi2),
        shape_jacobian=jacobian,
        singular_values=padded,
        weak_basis=weak_basis,
        strong_basis=strong_basis,
        weak_support_lower=support_lower,
        weak_support_upper=support_upper,
    )


def shape_profile_targets(
    geometry: ShapeProfileGeometry,
    *,
    coordinates: Any,
    config: ShapeProfileAtlasConfig | None = None,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    """Cover the declared weak-coordinate support without phase-based pruning."""

    active = config or ShapeProfileAtlasConfig()
    lower, upper = _normalized_bounds(
        coordinates,
        active.support_inset_fraction,
    )
    sampling_lower, sampling_upper = shape_profile_sampling_bounds(
        geometry,
        config=active,
    )
    weak_count = geometry.weak_basis.shape[1]
    records: list[tuple[str, np.ndarray, np.ndarray]] = []

    def add(source: str, weak: np.ndarray) -> None:
        target = (
            geometry.center_normalized_values
            + geometry.weak_basis @ np.asarray(weak, dtype=float)
        )
        tolerance = 32.0 * np.finfo(float).eps
        if np.any(target < lower - tolerance) or np.any(target > upper + tolerance):
            return
        records.append((source, np.asarray(weak, dtype=float).copy(), target))

    add("root", np.zeros(weak_count, dtype=float))
    grid_count = max(int(active.direction_grid_points), 2)
    for direction in range(weak_count):
        for grid_index, value in enumerate(
            np.linspace(
                sampling_lower[direction],
                sampling_upper[direction],
                grid_count,
                dtype=float,
            )
        ):
            weak = np.zeros(weak_count, dtype=float)
            weak[direction] = float(value)
            add(f"axis-{direction:02d}-{grid_index:03d}", weak)

    draws = max(int(active.combination_draws), 0)
    if draws > 0 and weak_count > 1:
        exponent = int(math.ceil(math.log2(draws)))
        unit = qmc.Sobol(
            d=weak_count,
            scramble=True,
            seed=max(int(active.random_seed), 0),
        ).random_base2(exponent)[:draws]
        weak_values = sampling_lower + unit * (
            sampling_upper - sampling_lower
        )
        for draw_index, weak in enumerate(weak_values):
            add(f"combination-{draw_index:04d}", weak)

    selected: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for record in records:
        key = np.round(record[2], 12).astype(np.float64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        selected.append(record)
    return tuple(selected)


def shape_profile_sampling_bounds(
    geometry: ShapeProfileGeometry,
    *,
    config: ShapeProfileAtlasConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect physical ray support with the declared shape-Fisher support."""

    active = config or ShapeProfileAtlasConfig()
    weak_count = geometry.weak_basis.shape[1]
    weak_singular_values = np.asarray(
        geometry.singular_values[-weak_count:],
        dtype=float,
    )
    shape_radius = math.sqrt(
        max(float(active.maximum_shape_delta_chi2), np.finfo(float).eps)
    )
    fisher_half_width = np.asarray(
        [
            (
                shape_radius / singular_value
                if singular_value > np.finfo(float).tiny
                else math.inf
            )
            for singular_value in weak_singular_values
        ],
        dtype=float,
    )
    sampling_lower = np.maximum(
        np.asarray(geometry.weak_support_lower, dtype=float),
        -fisher_half_width,
    )
    sampling_upper = np.minimum(
        np.asarray(geometry.weak_support_upper, dtype=float),
        fisher_half_width,
    )
    if np.any(sampling_lower >= sampling_upper):
        raise RuntimeError("Shape-profile Fisher support has no interior volume.")
    return sampling_lower, sampling_upper


def profile_shape_target(
    *,
    evaluator: Any,
    coordinates: Any,
    geometry: ShapeProfileGeometry,
    target_normalized_values: Sequence[float],
    weak_coordinates: Sequence[float],
    root_id: str,
    point_id: str,
    source: str,
    initial_normalized_values: Sequence[float] | None = None,
    config: ShapeProfileAtlasConfig | None = None,
) -> ShapeProfileAtlasPoint:
    """Optimize strong coordinates at one fixed weak-manifold location."""

    active = config or ShapeProfileAtlasConfig()
    target = np.asarray(target_normalized_values, dtype=float)
    weak = np.asarray(weak_coordinates, dtype=float)
    lower, upper = _normalized_bounds(
        coordinates,
        active.support_inset_fraction,
    )
    weak_projection = geometry.weak_basis.T
    fixed_value = weak_projection @ target
    constraint = LinearConstraint(weak_projection, fixed_value, fixed_value)
    prior_precision = max(float(active.prior_precision), 0.0)

    def objective(values: np.ndarray) -> float:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            result = float(profile.chi2) + prior_precision * float(values @ values)
            return result if np.isfinite(result) else 1.0e30
        except Exception:
            return 1.0e30

    initial = np.asarray(
        target if initial_normalized_values is None else initial_normalized_values,
        dtype=float,
    )
    if initial.shape != target.shape or not np.all(np.isfinite(initial)):
        raise ValueError("A shape-profile initial state must match the target.")
    initial += geometry.weak_basis @ (
        weak_projection @ target - weak_projection @ initial
    )
    optimization = minimize(
        objective,
        np.clip(initial, lower, upper),
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=(constraint,),
        options={
            "maxiter": max(int(active.profile_max_iterations), 1),
            "ftol": 1.0e-10,
            "disp": False,
        },
    )
    values = np.asarray(optimization.x, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, values))
    shape_chi2 = float(profile.chi2)
    prior_chi2 = prior_precision * float(values @ values)
    constraint_error = float(
        np.max(np.abs(weak_projection @ values - fixed_value))
    )
    success = bool(
        optimization.success
        and constraint_error <= max(float(active.profile_constraint_tolerance), 0.0)
        and np.isfinite(shape_chi2)
    )
    return ShapeProfileAtlasPoint(
        point_id=str(point_id),
        root_id=str(root_id),
        source=str(source),
        normalized_values=values,
        parameter_values=_physical_values(coordinates, values),
        weak_coordinates=weak.copy(),
        shape_chi2=shape_chi2,
        prior_chi2=prior_chi2,
        objective_chi2=shape_chi2 + prior_chi2,
        profile_success=success,
        profile_iterations=int(getattr(optimization, "nit", 0)),
        profile_constraint_error=constraint_error,
    )


def build_shape_profile_atlas(
    *,
    evaluator: Any,
    coordinates: Any,
    roots: Sequence[tuple[str, Sequence[float]]],
    config: ShapeProfileAtlasConfig | None = None,
) -> tuple[ShapeProfileGeometry, tuple[ShapeProfileAtlasPoint, ...]]:
    """Build a direct all-support atlas around wrap-invariant shape roots."""

    active = config or ShapeProfileAtlasConfig()
    if not roots:
        raise ValueError("Shape-profile atlas requires at least one root.")
    geometries = []
    raw_points = []
    for root_index, (root_id, root_values) in enumerate(roots):
        geometry = shape_profile_geometry(
            evaluator=evaluator,
            coordinates=coordinates,
            center_normalized_values=root_values,
            config=active,
        )
        geometries.append(geometry)
        for target_index, (source, weak, target) in enumerate(
            shape_profile_targets(geometry, coordinates=coordinates, config=active)
        ):
            try:
                raw_points.append(
                    profile_shape_target(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        geometry=geometry,
                        target_normalized_values=target,
                        weak_coordinates=weak,
                        root_id=str(root_id),
                        point_id=f"atlas-r{root_index:03d}-p{target_index:05d}",
                        source=source,
                        config=active,
                    )
                )
            except Exception:
                continue

    if not raw_points:
        return geometries[0], ()
    best_shape = min(point.shape_chi2 for point in raw_points)
    supported = [
        point
        for point in raw_points
        if point.profile_success
        and point.shape_chi2
        <= best_shape + max(float(active.maximum_shape_delta_chi2), 0.0)
    ]
    selected: list[ShapeProfileAtlasPoint] = []
    separation = max(float(active.minimum_point_separation), 0.0)
    for point in sorted(
        supported,
        key=lambda item: (item.objective_chi2, item.point_id),
    ):
        if any(
            np.linalg.norm(point.normalized_values - existing.normalized_values)
            < separation
            for existing in selected
        ):
            continue
        selected.append(point)
    return geometries[0], tuple(selected)


def shape_profile_covariance(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    finite_difference_step: float,
    prior_precision: float = 0.0,
) -> np.ndarray:
    """Return a support-capped local Gaussian chart for path acquisition."""

    linearization = finite_difference_profile(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=normalized_values,
        step=max(float(finite_difference_step), 1.0e-7),
    )
    jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    information = jacobian.T @ jacobian
    information += max(float(prior_precision), 0.0) * np.eye(information.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (information + information.T))
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    support_variances = np.square(upper - lower) / 12.0
    projected_support = np.square(eigenvectors).T @ support_variances
    scale = max(float(np.max(eigenvalues)), 1.0)
    floor = max(scale * 1.0e-14, np.finfo(float).tiny)
    variances = np.where(
        eigenvalues > floor,
        np.minimum(1.0 / np.maximum(eigenvalues, floor), projected_support),
        projected_support,
    )
    covariance = (eigenvectors * variances) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def acquire_shape_profile_paths(
    *,
    evaluator: Any,
    coordinates: Any,
    likelihood: Any,
    atlas_points: Sequence[ShapeProfileAtlasPoint],
    finite_difference_step: float = 2.0e-4,
    acquisition_config: ConditionedPathDirectionConfig | None = None,
    maximum_paths_per_point: int = 4,
    maximum_candidates: int = 256,
) -> tuple[ShapeProfilePathCandidate, ...]:
    """Acquire and rank complete paths using only full-sequence evidence."""

    active = acquisition_config or ConditionedPathDirectionConfig()
    point_buckets: list[list[ShapeProfilePathCandidate]] = []
    for point in atlas_points:
        try:
            covariance = shape_profile_covariance(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=point.normalized_values,
                finite_difference_step=finite_difference_step,
            )
            acquisition = likelihood.acquire_conditioned_paths_from_full_chart(
                point.normalized_values,
                covariance,
                config=active,
            )
        except Exception:
            point_buckets.append([])
            continue
        point_candidates = []
        for proposal_index, item in enumerate(acquisition.proposals):
            normalized = np.asarray(item.conditioned_normalized_values, dtype=float)
            path = tuple(int(value) for value in item.integer_offsets)
            try:
                profile = evaluator.evaluate_values(
                    _physical_values(coordinates, normalized)
                )
                shape_chi2, phase_chi2, closure = exact_fixed_path_chi2(
                    profile,
                    anchor_session=0,
                    active_sessions=tuple(range(1, int(profile.session_count))),
                    integer_offsets=path,
                )
            except Exception:
                continue
            selected_channel = int(item.selected_channel_index)
            lattice_proposal = LatticeProposal(
                root_mode_id=point.root_id,
                parent_mode_id=point.point_id,
                source=f"shape-profile-atlas:{item.conditioning_channel}",
                active_sessions=tuple(range(1, int(profile.session_count))),
                integer_offsets=path,
                initial_normalized_values=normalized,
                squared_mahalanobis=float(
                    item.channel_exact_closure_chi2[selected_channel]
                ),
                log_predictive_density=-0.5 * float(
                    item.channel_objective_chi2[selected_channel]
                ),
                kappa=1.0,
            )
            point_candidates.append(
                ShapeProfilePathCandidate(
                    candidate_id=(
                        f"{point.point_id}-path-{proposal_index:04d}"
                    ),
                    atlas_point_id=point.point_id,
                    proposal=lattice_proposal,
                    exact_start_shape_chi2=float(shape_chi2),
                    exact_start_phase_chi2=float(phase_chi2),
                    exact_start_closure_chi2=float(closure),
                    acquisition_linear_chi2=float(
                        item.channel_exact_closure_chi2[selected_channel]
                    ),
                )
            )
        target_per_point = max(int(maximum_paths_per_point), 1)
        rankings = (
            sorted(
                point_candidates,
                key=lambda item: (
                    item.exact_start_closure_chi2,
                    item.acquisition_linear_chi2,
                    item.candidate_id,
                ),
            ),
            sorted(
                point_candidates,
                key=lambda item: (
                    item.acquisition_linear_chi2,
                    item.exact_start_closure_chi2,
                    item.candidate_id,
                ),
            ),
        )
        portfolio: list[ShapeProfilePathCandidate] = []
        seen_point_paths: set[tuple[int, ...]] = set()
        maximum_rank = max((len(ranking) for ranking in rankings), default=0)
        for rank in range(maximum_rank):
            for ranking in rankings:
                if rank >= len(ranking):
                    continue
                candidate = ranking[rank]
                path = candidate.proposal.integer_offsets
                if path in seen_point_paths:
                    continue
                seen_point_paths.add(path)
                portfolio.append(candidate)
                if len(portfolio) >= target_per_point:
                    break
            if len(portfolio) >= target_per_point:
                break
        point_buckets.append(portfolio)

    selected: list[ShapeProfilePathCandidate] = []
    seen_paths: set[tuple[int, ...]] = set()
    target = max(int(maximum_candidates), 1)
    maximum_rank = max((len(bucket) for bucket in point_buckets), default=0)
    for rank in range(maximum_rank):
        for bucket in point_buckets:
            if rank >= len(bucket):
                continue
            candidate = bucket[rank]
            path = candidate.proposal.integer_offsets
            if path in seen_paths:
                continue
            seen_paths.add(path)
            selected.append(candidate)
            if len(selected) >= target:
                return tuple(selected)
    return tuple(selected)


__all__ = [
    "ShapeProfileAtlasConfig",
    "ShapeProfileAtlasPoint",
    "ShapeProfileGeometry",
    "ShapeProfilePathCandidate",
    "acquire_shape_profile_paths",
    "build_shape_profile_atlas",
    "profile_shape_target",
    "shape_profile_covariance",
    "shape_profile_geometry",
    "shape_profile_sampling_bounds",
    "shape_profile_targets",
]
