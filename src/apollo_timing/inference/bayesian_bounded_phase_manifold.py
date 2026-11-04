from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, lsq_linear, minimize
from scipy.spatial import Delaunay, HalfspaceIntersection

from apollo_timing.inference.bayesian_global_lattice_mode_tracker import (
    LatticeProposal,
    _physical_values,
    exact_fixed_path_chi2,
    finite_difference_profile,
)
from apollo_timing.inference.bayesian_shape_profile_atlas import (
    ShapeProfileAtlasConfig,
    ShapeProfileGeometry,
    profile_shape_target,
    shape_profile_sampling_bounds,
)
from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)


@dataclass(frozen=True)
class BoundedPhaseManifoldConfig:
    """Controls for a nonlinear weak-manifold/integer-lattice atlas."""

    maximum_phase_prediction_error_cycles: float = 0.15
    maximum_shape_prediction_error: float = 2.0
    maximum_depth: int = 8
    maximum_cells: int = 256
    minimum_scaled_edge_length: float = 1.0e-4
    lattice_covariance_scales: tuple[float, ...] = (1.0, 3.0)
    lattice_candidates_per_scale: int = 32
    maximum_lattice_nodes: int = 500_000
    maximum_paths_per_cell: int = 4
    maximum_candidates: int = 256
    global_lattice_covariance_scales: tuple[float, ...] = (1.0,)
    global_lattice_candidates_per_scale: int = 256
    maximum_global_prefits: int = 256
    maximum_global_candidates: int = 128
    maximum_local_chart_cells: int = 16
    local_chart_lattice_candidates: int = 64
    maximum_local_chart_prefits: int = 32
    maximum_paths_per_local_chart: int = 6
    joint_incumbent_chi2: float = math.inf
    maximum_strong_phase_directions: int = 5
    strong_direction_sigma_samples: tuple[float, ...] = (
        1.0,
        3.0,
        10.0,
        30.0,
    )
    require_resolved_cells: bool = True
    random_seed: int = 20260812


@dataclass(frozen=True)
class PhaseManifoldSample:
    sample_id: str
    weak_coordinates: np.ndarray
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    shape_residuals: np.ndarray
    relative_session_offsets: np.ndarray
    relative_offset_covariance: np.ndarray
    shape_chi2: float
    profile_success: bool
    profile_iterations: int


@dataclass(frozen=True)
class BoundedPhaseManifoldCell:
    cell_id: str
    parent_cell_id: str | None
    depth: int
    vertices: np.ndarray
    centroid: np.ndarray
    sample_ids: tuple[str, ...]
    phase_intercept: np.ndarray
    phase_jacobian: np.ndarray
    phase_quadratic: np.ndarray
    shape_intercept: np.ndarray
    shape_jacobian: np.ndarray
    shape_quadratic: np.ndarray
    normalized_intercept: np.ndarray
    normalized_jacobian: np.ndarray
    normalized_quadratic: np.ndarray
    relative_offset_covariance: np.ndarray
    maximum_phase_prediction_error_cycles: float
    maximum_shape_prediction_error: float
    maximum_phase_span_cycles: float
    shape_surrogate_lower_chi2: float
    minimum_sample_joint_chi2: float
    linearized_joint_chi2: float
    acquisition_score: float
    incumbent_competitive: bool
    split_edge: tuple[int, int]
    surrogate_order: int
    resolved: bool
    resolution_limited: bool


@dataclass(frozen=True)
class BoundedPhaseManifoldAtlas:
    root_id: str
    weak_lower: np.ndarray
    weak_upper: np.ndarray
    polytope_vertices: np.ndarray
    samples: tuple[PhaseManifoldSample, ...]
    cells: tuple[BoundedPhaseManifoldCell, ...]
    prediction_tolerance_cycles: float
    unresolved_cell_count: int


@dataclass(frozen=True)
class BoundedPhaseManifoldPathCandidate:
    candidate_id: str
    cell_id: str
    source: str
    proposal: LatticeProposal
    weak_coordinates: np.ndarray
    predicted_shape_chi2: float
    predicted_phase_chi2: float
    exact_start_shape_chi2: float
    exact_start_phase_chi2: float
    exact_start_closure_chi2: float


def _positive_definite_covariance(covariance: np.ndarray) -> np.ndarray:
    result = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    eigenvalues = np.linalg.eigvalsh(result)
    floor = max(float(eigenvalues[-1]) * 1.0e-12, 1.0e-12)
    if float(eigenvalues[0]) < floor:
        result = result + np.eye(result.shape[0]) * (
            floor - float(eigenvalues[0])
        )
    return result


def _sample_nearest_path_chi2(sample: PhaseManifoldSample) -> float:
    phase = np.asarray(sample.relative_session_offsets, dtype=float)
    residual = phase - np.rint(phase)
    covariance = _positive_definite_covariance(
        sample.relative_offset_covariance
    )
    phase_chi2 = float(residual @ np.linalg.solve(covariance, residual))
    return float(sample.shape_chi2) + phase_chi2


def _shape_surrogate_lower_chi2(
    *,
    vertices: np.ndarray,
    centroid: np.ndarray,
    intercept: np.ndarray,
    jacobian: np.ndarray,
    quadratic: np.ndarray,
    error_radius: float,
) -> float:
    vertex_zero = vertices[0]
    edge = (vertices[1:] - vertex_zero).T
    dimension = edge.shape[1]

    def objective(tangent: np.ndarray) -> float:
        weak = vertex_zero + edge @ tangent
        residual = _quadratic_value(
            intercept,
            jacobian,
            quadratic,
            weak - centroid,
        )
        return float(residual @ residual)

    start = np.full(dimension, 1.0 / (dimension + 1), dtype=float)
    optimum = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=Bounds(np.zeros(dimension), np.ones(dimension)),
        constraints=(
            LinearConstraint(np.ones((1, dimension)), -math.inf, 1.0),
        ),
        options={"maxiter": 120, "ftol": 1.0e-10, "disp": False},
    )
    predicted_norm = math.sqrt(max(float(optimum.fun), 0.0))
    lower_norm = max(predicted_norm - max(float(error_radius), 0.0), 0.0)
    return lower_norm * lower_norm


def _linearized_joint_lattice_chi2(
    *,
    linearization: Any,
    maximum_lattice_nodes: int,
) -> float:
    profile = linearization.profile
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    relative_phase = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0],
        dtype=float,
    )
    relative_phase_jacobian = (
        np.asarray(linearization.offset_jacobian[1:], dtype=float)
        - np.asarray(linearization.offset_jacobian[0], dtype=float)
    )
    information = shape_jacobian.T @ shape_jacobian
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (information + information.T)
    )
    floor = max(float(np.max(eigenvalues)) * 1.0e-12, 1.0e-12)
    inverse_information = (
        eigenvectors * (1.0 / np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    shape_delta = -inverse_information @ shape_jacobian.T @ shape
    shape_residual = shape + shape_jacobian @ shape_delta
    phase_mode = relative_phase + relative_phase_jacobian @ shape_delta
    covariance = _positive_definite_covariance(
        _relative_covariance(profile)
        + relative_phase_jacobian
        @ inverse_information
        @ relative_phase_jacobian.T
    )
    result = solve_integer_least_squares(
        phase_mode,
        covariance,
        candidate_count=1,
        max_visited_nodes=max(int(maximum_lattice_nodes), 1),
        return_partial_on_budget=True,
    )
    if not result.candidates:
        return math.inf
    return float(shape_residual @ shape_residual) + float(
        result.candidates[0].squared_mahalanobis
    )


def _normalized_bounds(coordinates: Any, inset_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    inset = np.maximum(
        (upper - lower) * max(float(inset_fraction), 0.0),
        1.0e-12,
    )
    return lower + inset, upper - inset


def _weak_polytope(
    *,
    geometry: ShapeProfileGeometry,
    coordinates: Any,
    atlas_config: ShapeProfileAtlasConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact weak support polytope and its bounding Fisher box."""

    weak_lower, weak_upper = shape_profile_sampling_bounds(
        geometry,
        config=atlas_config,
    )
    normalized_lower, normalized_upper = _normalized_bounds(
        coordinates,
        atlas_config.support_inset_fraction,
    )
    center = np.asarray(geometry.center_normalized_values, dtype=float)
    basis = np.asarray(geometry.weak_basis, dtype=float)
    dimension = basis.shape[1]
    rows = []
    limits = []
    for row, low, high, value in zip(
        basis,
        normalized_lower,
        normalized_upper,
        center,
        strict=True,
    ):
        rows.extend((row, -row))
        limits.extend((high - value, value - low))
    for index in range(dimension):
        unit = np.zeros(dimension, dtype=float)
        unit[index] = 1.0
        rows.extend((unit, -unit))
        limits.extend((weak_upper[index], -weak_lower[index]))
    matrix = np.asarray(rows, dtype=float)
    limit = np.asarray(limits, dtype=float)
    if dimension == 1:
        return (
            weak_lower,
            weak_upper,
            np.asarray([[weak_lower[0]], [weak_upper[0]]], dtype=float),
        )
    norms = np.linalg.norm(matrix, axis=1)
    objective = np.zeros(dimension + 1, dtype=float)
    objective[-1] = -1.0
    constraints = np.column_stack((matrix, norms))
    optimum = linprog(
        objective,
        A_ub=constraints,
        b_ub=limit,
        bounds=[(None, None)] * dimension + [(0.0, None)],
        method="highs",
    )
    if not optimum.success or float(optimum.x[-1]) <= 1.0e-10:
        raise RuntimeError("The weak shape support has no full-dimensional interior.")
    interior = np.asarray(optimum.x[:-1], dtype=float)
    halfspaces = np.column_stack((matrix, -limit))
    intersection = HalfspaceIntersection(halfspaces, interior)
    vertices = np.asarray(intersection.intersections, dtype=float)
    vertices = np.unique(np.round(vertices, decimals=13), axis=0)
    if vertices.shape[0] < dimension + 1:
        raise RuntimeError("The weak shape support cannot form a simplex atlas.")
    return weak_lower, weak_upper, vertices


def _initial_simplices(vertices: np.ndarray) -> tuple[np.ndarray, ...]:
    dimension = int(vertices.shape[1])
    if dimension == 1:
        order = np.argsort(vertices[:, 0])
        return tuple(vertices[[order[index], order[index + 1]]] for index in range(len(order) - 1))
    triangulation = Delaunay(vertices, qhull_options="QJ")
    simplices = []
    seen = set()
    for indices in triangulation.simplices:
        simplex = np.asarray(vertices[np.asarray(indices, dtype=int)], dtype=float)
        edge = simplex[1:] - simplex[0]
        if abs(float(np.linalg.det(edge))) <= 1.0e-14:
            continue
        key = tuple(sorted(tuple(np.round(row, 12)) for row in simplex))
        if key in seen:
            continue
        seen.add(key)
        simplices.append(simplex)
    if not simplices:
        raise RuntimeError("The weak support triangulation contains no valid cells.")
    return tuple(simplices)


def _relative_covariance(profile: Any) -> np.ndarray:
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    covariance = np.diag(variances[1:]) + float(variances[0])
    return 0.5 * (covariance + covariance.T)


def _affine_map(
    vertices: np.ndarray,
    values: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        (
            np.ones(vertices.shape[0], dtype=float),
            vertices - centroid,
        )
    )
    coefficients = np.linalg.solve(design, values)
    return np.asarray(coefficients[0], dtype=float), np.asarray(coefficients[1:].T, dtype=float)


def _quadratic_design(values: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    delta = np.atleast_2d(np.asarray(values, dtype=float)) - centroid
    columns = [np.ones(delta.shape[0], dtype=float)]
    columns.extend(delta[:, index] for index in range(delta.shape[1]))
    columns.extend(
        delta[:, left] * delta[:, right]
        for left in range(delta.shape[1])
        for right in range(left, delta.shape[1])
    )
    return np.column_stack(columns)


def _quadratic_map(
    points: np.ndarray,
    values: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dimension = int(points.shape[1])
    coefficients = np.linalg.lstsq(
        _quadratic_design(points, centroid),
        values,
        rcond=None,
    )[0]
    intercept = np.asarray(coefficients[0], dtype=float)
    jacobian = np.asarray(coefficients[1 : dimension + 1].T, dtype=float)
    quadratic = np.zeros((values.shape[1], dimension, dimension), dtype=float)
    index = dimension + 1
    for left in range(dimension):
        for right in range(left, dimension):
            value = np.asarray(coefficients[index], dtype=float)
            if left == right:
                quadratic[:, left, right] = value
            else:
                quadratic[:, left, right] = 0.5 * value
                quadratic[:, right, left] = 0.5 * value
            index += 1
    return intercept, jacobian, quadratic


def _quadratic_value(
    intercept: np.ndarray,
    jacobian: np.ndarray,
    quadratic: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    return (
        intercept
        + jacobian @ delta
        + np.einsum("i,oij,j->o", delta, quadratic, delta)
    )


def _quadratic_jacobian(
    jacobian: np.ndarray,
    quadratic: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    return jacobian + 2.0 * np.einsum("oij,j->oi", quadratic, delta)


def _simplex_uniform_covariance(vertices: np.ndarray) -> np.ndarray:
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    dimension = int(vertices.shape[1])
    covariance = centered.T @ centered / ((dimension + 1) * (dimension + 2))
    return 0.5 * (covariance + covariance.T)


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
            continue
        left = (low - center) / vector
        right = (high - center) / vector
        minimum = max(minimum, min(left, right))
        maximum = min(maximum, max(left, right))
    return float(minimum), float(maximum)


def build_bounded_phase_manifold_atlas(
    *,
    evaluator: Any,
    coordinates: Any,
    geometry: ShapeProfileGeometry,
    root_id: str,
    atlas_config: ShapeProfileAtlasConfig | None = None,
    config: BoundedPhaseManifoldConfig | None = None,
    initial_atlas: BoundedPhaseManifoldAtlas | None = None,
) -> BoundedPhaseManifoldAtlas:
    """Tessellate a profiled weak manifold until phase maps are locally valid."""

    shape_config = atlas_config or ShapeProfileAtlasConfig()
    active = config or BoundedPhaseManifoldConfig()
    weak_lower, weak_upper, polytope_vertices = _weak_polytope(
        geometry=geometry,
        coordinates=coordinates,
        atlas_config=shape_config,
    )
    domain_scale = np.maximum(weak_upper - weak_lower, 1.0e-12)
    sample_cache: dict[bytes, PhaseManifoldSample] = {}
    if initial_atlas is not None:
        if initial_atlas.root_id != str(root_id):
            raise ValueError("A resumed manifold atlas must use the same shape root.")
        for item in initial_atlas.samples:
            key = np.round(item.weak_coordinates, 13).astype(np.float64).tobytes()
            sample_cache[key] = item

    def sample(values: Sequence[float]) -> PhaseManifoldSample:
        weak = np.asarray(values, dtype=float)
        key = np.round(weak, 13).astype(np.float64).tobytes()
        cached = sample_cache.get(key)
        if cached is not None:
            return cached
        target = geometry.center_normalized_values + geometry.weak_basis @ weak
        initial = target
        if sample_cache:
            nearest = min(
                sample_cache.values(),
                key=lambda item: float(
                    np.linalg.norm((item.weak_coordinates - weak) / domain_scale)
                ),
            )
            initial = (
                nearest.normalized_values
                + geometry.weak_basis @ (weak - nearest.weak_coordinates)
            )
        point = profile_shape_target(
            evaluator=evaluator,
            coordinates=coordinates,
            geometry=geometry,
            target_normalized_values=target,
            weak_coordinates=weak,
            root_id=str(root_id),
            point_id=f"manifold-sample-{len(sample_cache):06d}",
            source="adaptive-simplex",
            initial_normalized_values=initial,
            config=shape_config,
        )
        profile = evaluator.evaluate_values(point.parameter_values)
        relative = np.asarray(
            profile.session_offsets[1:] - profile.session_offsets[0],
            dtype=float,
        )
        result = PhaseManifoldSample(
            sample_id=point.point_id,
            weak_coordinates=weak.copy(),
            normalized_values=np.asarray(point.normalized_values, dtype=float),
            parameter_values=np.asarray(point.parameter_values, dtype=float),
            shape_residuals=np.asarray(profile.normalized_residuals, dtype=float),
            relative_session_offsets=relative,
            relative_offset_covariance=_relative_covariance(profile),
            shape_chi2=float(profile.chi2),
            profile_success=bool(point.profile_success),
            profile_iterations=int(point.profile_iterations),
        )
        sample_cache[key] = result
        return result

    serial = 0
    if initial_atlas is not None and initial_atlas.cells:
        serial = 1 + max(
            int(item.cell_id.rsplit("-", 1)[-1])
            for item in initial_atlas.cells
        )

    def analyze(
        vertices: np.ndarray,
        *,
        depth: int,
        parent: str | None,
    ) -> BoundedPhaseManifoldCell:
        nonlocal serial
        cell_id = f"phase-cell-{serial:06d}"
        serial += 1
        centroid = np.mean(vertices, axis=0)
        vertex_samples = tuple(sample(row) for row in vertices)
        center_sample = sample(centroid)
        affine_phase_values = np.vstack(
            [item.relative_session_offsets for item in vertex_samples]
        )
        affine_phase_intercept, affine_phase_jacobian = _affine_map(
            vertices,
            affine_phase_values,
            centroid,
        )
        training = [*vertex_samples, center_sample]
        edge_records = []
        for left, right in itertools.combinations(range(vertices.shape[0]), 2):
            midpoint = 0.5 * (vertices[left] + vertices[right])
            item = sample(midpoint)
            training.append(item)
            predicted = (
                affine_phase_intercept
                + affine_phase_jacobian @ (midpoint - centroid)
            )
            edge_records.append(
                (
                    float(np.max(np.abs(item.relative_session_offsets - predicted))),
                    left,
                    right,
                )
            )
        unique_training = tuple(
            {item.sample_id: item for item in training}.values()
        )
        training_points = np.vstack(
            [item.weak_coordinates for item in unique_training]
        )
        phase_values = np.vstack(
            [item.relative_session_offsets for item in unique_training]
        )
        shape_values = np.vstack(
            [item.shape_residuals for item in unique_training]
        )
        normalized_values = np.vstack(
            [item.normalized_values for item in unique_training]
        )
        phase_intercept, phase_jacobian, phase_quadratic = _quadratic_map(
            training_points,
            phase_values,
            centroid,
        )
        shape_intercept, shape_jacobian, shape_quadratic = _quadratic_map(
            training_points,
            shape_values,
            centroid,
        )
        (
            normalized_intercept,
            normalized_jacobian,
            normalized_quadratic,
        ) = _quadratic_map(
            training_points,
            normalized_values,
            centroid,
        )
        validation = [
            sample(0.5 * (centroid + vertex))
            for vertex in vertices
        ]
        phase_errors = []
        shape_errors = []
        for item in validation:
            delta = item.weak_coordinates - centroid
            phase_errors.append(
                float(
                    np.max(
                        np.abs(
                            item.relative_session_offsets
                            - _quadratic_value(
                                phase_intercept,
                                phase_jacobian,
                                phase_quadratic,
                                delta,
                            )
                        )
                    )
                )
            )
            shape_errors.append(
                float(
                    np.linalg.norm(
                        item.shape_residuals
                        - _quadratic_value(
                            shape_intercept,
                            shape_jacobian,
                            shape_quadratic,
                            delta,
                        )
                    )
                )
            )
        maximum_phase_error = max(phase_errors, default=0.0)
        maximum_shape_error = max(shape_errors, default=0.0)
        shape_lower_chi2 = _shape_surrogate_lower_chi2(
            vertices=vertices,
            centroid=centroid,
            intercept=shape_intercept,
            jacobian=shape_jacobian,
            quadratic=shape_quadratic,
            error_radius=maximum_shape_error,
        )
        minimum_sample_joint = min(
            _sample_nearest_path_chi2(item)
            for item in (*unique_training, *validation)
        )
        try:
            center_linearization = finite_difference_profile(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=center_sample.normalized_values,
                step=max(float(shape_config.finite_difference_step), 1.0e-7),
            )
            linearized_joint = _linearized_joint_lattice_chi2(
                linearization=center_linearization,
                maximum_lattice_nodes=active.maximum_lattice_nodes,
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            linearized_joint = math.inf
        acquisition_score = min(minimum_sample_joint, linearized_joint)
        incumbent_competitive = shape_lower_chi2 <= float(
            active.joint_incumbent_chi2
        )
        phase_resolved = maximum_phase_error <= max(
            float(active.maximum_phase_prediction_error_cycles),
            0.0,
        )
        shape_resolved = maximum_shape_error <= max(
            float(active.maximum_shape_prediction_error),
            0.0,
        )
        longest = max(
            itertools.combinations(range(vertices.shape[0]), 2),
            key=lambda pair: float(
                np.linalg.norm((vertices[pair[0]] - vertices[pair[1]]) / domain_scale)
            ),
        )
        if edge_records:
            curved = max(edge_records, key=lambda item: (item[0], item[1], item[2]))
            split_edge = (int(curved[1]), int(curved[2]))
            if curved[0] <= 0.25 * maximum_phase_error:
                split_edge = (int(longest[0]), int(longest[1]))
        else:
            split_edge = (int(longest[0]), int(longest[1]))
        scaled_edge = float(
            np.linalg.norm(
                (vertices[split_edge[0]] - vertices[split_edge[1]]) / domain_scale
            )
        )
        resolution_limited = bool(
            depth >= max(int(active.maximum_depth), 0)
            or scaled_edge <= max(float(active.minimum_scaled_edge_length), 0.0)
        )
        ids = tuple(
            dict.fromkeys(
                item.sample_id for item in (*unique_training, *validation)
            )
        )
        return BoundedPhaseManifoldCell(
            cell_id=cell_id,
            parent_cell_id=parent,
            depth=int(depth),
            vertices=np.asarray(vertices, dtype=float),
            centroid=np.asarray(centroid, dtype=float),
            sample_ids=ids,
            phase_intercept=phase_intercept,
            phase_jacobian=phase_jacobian,
            phase_quadratic=phase_quadratic,
            shape_intercept=shape_intercept,
            shape_jacobian=shape_jacobian,
            shape_quadratic=shape_quadratic,
            normalized_intercept=normalized_intercept,
            normalized_jacobian=normalized_jacobian,
            normalized_quadratic=normalized_quadratic,
            relative_offset_covariance=center_sample.relative_offset_covariance,
            maximum_phase_prediction_error_cycles=maximum_phase_error,
            maximum_shape_prediction_error=maximum_shape_error,
            maximum_phase_span_cycles=float(np.max(np.ptp(phase_values, axis=0))),
            shape_surrogate_lower_chi2=float(shape_lower_chi2),
            minimum_sample_joint_chi2=float(minimum_sample_joint),
            linearized_joint_chi2=float(linearized_joint),
            acquisition_score=float(acquisition_score),
            incumbent_competitive=bool(incumbent_competitive),
            split_edge=split_edge,
            surrogate_order=2,
            resolved=bool(phase_resolved and shape_resolved),
            resolution_limited=resolution_limited,
        )

    if initial_atlas is None:
        leaves = [
            analyze(simplex, depth=0, parent=None)
            for simplex in _initial_simplices(polytope_vertices)
        ]
    else:
        leaves = [
            analyze(
                np.asarray(cell.vertices, dtype=float),
                depth=int(cell.depth),
                parent=cell.parent_cell_id,
            )
            for cell in initial_atlas.cells
        ]
    maximum_cells = max(int(active.maximum_cells), len(leaves))
    while len(leaves) < maximum_cells:
        eligible = [
            cell
            for cell in leaves
            if not cell.resolved and not cell.resolution_limited
        ]
        if not eligible:
            break
        parent = min(
            eligible,
            key=lambda item: (
                not item.incumbent_competitive,
                item.acquisition_score,
                -max(
                    item.maximum_phase_prediction_error_cycles
                    / max(
                        float(active.maximum_phase_prediction_error_cycles),
                        1.0e-12,
                    ),
                    item.maximum_shape_prediction_error
                    / max(float(active.maximum_shape_prediction_error), 1.0e-12),
                ),
                item.depth,
                item.cell_id,
            ),
        )
        left, right = parent.split_edge
        midpoint = 0.5 * (parent.vertices[left] + parent.vertices[right])
        first = parent.vertices.copy()
        second = parent.vertices.copy()
        first[left] = midpoint
        second[right] = midpoint
        leaves.remove(parent)
        leaves.extend(
            (
                analyze(first, depth=parent.depth + 1, parent=parent.cell_id),
                analyze(second, depth=parent.depth + 1, parent=parent.cell_id),
            )
        )
    leaves = sorted(leaves, key=lambda item: item.cell_id)
    unresolved = sum(not cell.resolved for cell in leaves)
    return BoundedPhaseManifoldAtlas(
        root_id=str(root_id),
        weak_lower=weak_lower,
        weak_upper=weak_upper,
        polytope_vertices=polytope_vertices,
        samples=tuple(sample_cache.values()),
        cells=tuple(leaves),
        prediction_tolerance_cycles=max(
            float(active.maximum_phase_prediction_error_cycles),
            0.0,
        ),
        unresolved_cell_count=int(unresolved),
    )


def _condition_path_in_simplex(
    *,
    cell: BoundedPhaseManifoldCell,
    integer_offsets: Sequence[int],
    strong_basis: np.ndarray,
    strong_shape_jacobian: np.ndarray,
    strong_phase_jacobian: np.ndarray,
    normalized_lower: np.ndarray,
    normalized_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    integers = np.asarray(integer_offsets, dtype=float)
    precision = np.linalg.inv(cell.relative_offset_covariance)
    vertex_zero = cell.vertices[0]
    edge = (cell.vertices[1:] - vertex_zero).T
    dimension = edge.shape[1]
    strong_dimension = int(strong_basis.shape[1])

    def state(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tangent = values[:dimension]
        strong = values[dimension:]
        weak = vertex_zero + edge @ tangent
        delta = weak - cell.centroid
        shape_center = _quadratic_value(
            cell.shape_intercept,
            cell.shape_jacobian,
            cell.shape_quadratic,
            delta,
        )
        phase_center = _quadratic_value(
            cell.phase_intercept,
            cell.phase_jacobian,
            cell.phase_quadratic,
            delta,
        )
        normalized_center = _quadratic_value(
            cell.normalized_intercept,
            cell.normalized_jacobian,
            cell.normalized_quadratic,
            delta,
        )
        normalized = normalized_center + strong_basis @ strong
        shape = shape_center + strong_shape_jacobian @ strong
        phase = (
            phase_center
            + strong_phase_jacobian @ strong
            - integers
        )
        return weak, normalized, shape, phase

    def terms(values: np.ndarray) -> tuple[float, np.ndarray, float, float]:
        weak, _, shape, phase = state(values)
        delta = weak - cell.centroid
        shape_chi2 = float(shape @ shape)
        phase_chi2 = float(phase @ precision @ phase)
        shape_q_derivative = _quadratic_jacobian(
            cell.shape_jacobian,
            cell.shape_quadratic,
            delta,
        )
        phase_q_derivative = _quadratic_jacobian(
            cell.phase_jacobian,
            cell.phase_quadratic,
            delta,
        )
        gradient_q = 2.0 * (
            shape_q_derivative.T @ shape
            + phase_q_derivative.T @ precision @ phase
        )
        gradient_strong = 2.0 * (
            strong_shape_jacobian.T @ shape
            + strong_phase_jacobian.T @ precision @ phase
        )
        gradient = np.concatenate((edge.T @ gradient_q, gradient_strong))
        return shape_chi2 + phase_chi2, gradient, shape_chi2, phase_chi2

    def physical_support(values: np.ndarray) -> np.ndarray:
        return state(values)[1]

    start = np.concatenate(
        (
            np.full(dimension, 1.0 / (dimension + 1), dtype=float),
            np.zeros(strong_dimension, dtype=float),
        )
    )
    parameter_bounds = Bounds(
        np.concatenate((np.zeros(dimension), np.full(strong_dimension, -2.0))),
        np.concatenate((np.ones(dimension), np.full(strong_dimension, 2.0))),
    )
    constraints = (
        LinearConstraint(
            np.concatenate(
                (np.ones(dimension, dtype=float), np.zeros(strong_dimension))
            )[None, :],
            -math.inf,
            1.0,
        ),
        {
            "type": "ineq",
            "fun": lambda value: physical_support(value) - normalized_lower,
        },
        {
            "type": "ineq",
            "fun": lambda value: normalized_upper - physical_support(value),
        },
    )
    optimization = minimize(
        lambda value: terms(value)[0],
        start,
        jac=lambda value: terms(value)[1],
        method="SLSQP",
        bounds=parameter_bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1.0e-10, "disp": False},
    )
    values = np.asarray(optimization.x, dtype=float)
    weak, normalized, _, _ = state(values)
    _, _, shape_chi2, phase_chi2 = terms(values)
    return weak, normalized, shape_chi2, phase_chi2


def _condition_global_linear_path(
    *,
    normalized_values: np.ndarray,
    shape_residuals: np.ndarray,
    shape_jacobian: np.ndarray,
    relative_session_offsets: np.ndarray,
    relative_phase_jacobian: np.ndarray,
    relative_offset_covariance: np.ndarray,
    integer_offsets: Sequence[int],
    normalized_lower: np.ndarray,
    normalized_upper: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Condition the complete continuous linear model on one integer path."""

    center = np.asarray(normalized_values, dtype=float)
    shape = np.asarray(shape_residuals, dtype=float)
    shape_jacobian = np.asarray(shape_jacobian, dtype=float)
    phase = np.asarray(relative_session_offsets, dtype=float)
    phase_jacobian = np.asarray(relative_phase_jacobian, dtype=float)
    integers = np.asarray(integer_offsets, dtype=float)
    covariance = _positive_definite_covariance(relative_offset_covariance)
    cholesky = np.linalg.cholesky(covariance)
    whitened_phase_jacobian = np.linalg.solve(cholesky, phase_jacobian)
    whitened_phase_target = np.linalg.solve(cholesky, integers - phase)
    design = np.vstack((shape_jacobian, whitened_phase_jacobian))
    target = np.concatenate((-shape, whitened_phase_target))
    conditioned = lsq_linear(
        design,
        target,
        bounds=(normalized_lower - center, normalized_upper - center),
        method="trf",
        tol=1.0e-12,
        lsmr_tol=1.0e-12,
        max_iter=300,
    )
    delta = np.asarray(conditioned.x, dtype=float)
    predicted_shape = shape + shape_jacobian @ delta
    predicted_phase = phase + phase_jacobian @ delta - integers
    phase_whitened = np.linalg.solve(cholesky, predicted_phase)
    return (
        np.clip(center + delta, normalized_lower, normalized_upper),
        float(predicted_shape @ predicted_shape),
        float(phase_whitened @ phase_whitened),
    )


def acquire_global_marginal_lattice_paths(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    root_id: str,
    atlas_config: ShapeProfileAtlasConfig | None = None,
    config: BoundedPhaseManifoldConfig | None = None,
) -> tuple[BoundedPhaseManifoldPathCandidate, ...]:
    """Marginalize the complete continuous posterior before ranking paths.

    This first inference stage treats weak and strong coordinates jointly. The
    nonlinear manifold later corrects the global Gaussian posterior; no
    shape-only filter precedes it.
    """

    shape_config = atlas_config or ShapeProfileAtlasConfig()
    active = config or BoundedPhaseManifoldConfig()
    center = np.asarray(normalized_values, dtype=float)
    normalized_lower, normalized_upper = _normalized_bounds(
        coordinates,
        shape_config.support_inset_fraction,
    )
    linearization = finite_difference_profile(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        step=max(float(shape_config.finite_difference_step), 1.0e-7),
    )
    profile = linearization.profile
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    relative_phase = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0],
        dtype=float,
    )
    relative_phase_jacobian = (
        np.asarray(linearization.offset_jacobian[1:], dtype=float)
        - np.asarray(linearization.offset_jacobian[0], dtype=float)
    )
    relative_covariance = _relative_covariance(profile)

    information = shape_jacobian.T @ shape_jacobian
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (information + information.T)
    )
    floor = max(float(np.max(eigenvalues)) * 1.0e-12, 1.0e-12)
    inverse_information = (
        eigenvectors * (1.0 / np.maximum(eigenvalues, floor))
    ) @ eigenvectors.T
    shape_mode_delta = -inverse_information @ shape_jacobian.T @ shape
    shape_mode_residuals = shape + shape_jacobian @ shape_mode_delta
    shape_floor_chi2 = float(shape_mode_residuals @ shape_mode_residuals)
    phase_mode = relative_phase + relative_phase_jacobian @ shape_mode_delta
    phase_parameter_covariance = (
        relative_phase_jacobian
        @ inverse_information
        @ relative_phase_jacobian.T
    )

    generated: dict[tuple[int, ...], tuple[float, str]] = {}
    for scale in active.global_lattice_covariance_scales:
        factor = float(scale)
        if not np.isfinite(factor) or factor <= 0.0:
            continue
        covariance = _positive_definite_covariance(
            relative_covariance
            + factor * factor * phase_parameter_covariance
        )
        try:
            result = solve_integer_least_squares(
                phase_mode,
                covariance,
                candidate_count=max(
                    int(active.global_lattice_candidates_per_scale),
                    1,
                ),
                max_visited_nodes=max(int(active.maximum_lattice_nodes), 1),
                return_partial_on_budget=True,
            )
        except (RuntimeError, ValueError):
            continue
        for candidate in result.candidates:
            path = tuple(int(value) for value in candidate.integer_vector)
            score = shape_floor_chi2 + float(candidate.squared_mahalanobis)
            current = generated.get(path)
            if current is None or score < current[0]:
                generated[path] = (score, f"global-marginal-x{factor:g}")

    ranked_paths = sorted(
        generated.items(),
        key=lambda item: (item[1][0], item[0]),
    )[: max(int(active.maximum_global_prefits), 1)]
    candidates = []
    active_sessions = tuple(range(1, int(profile.session_count)))
    for serial, (path, (_, source)) in enumerate(ranked_paths):
        try:
            conditioned, predicted_shape, predicted_phase = (
                _condition_global_linear_path(
                    normalized_values=center,
                    shape_residuals=shape,
                    shape_jacobian=shape_jacobian,
                    relative_session_offsets=relative_phase,
                    relative_phase_jacobian=relative_phase_jacobian,
                    relative_offset_covariance=relative_covariance,
                    integer_offsets=path,
                    normalized_lower=normalized_lower,
                    normalized_upper=normalized_upper,
                )
            )
            parameter_values = _physical_values(coordinates, conditioned)
            exact_profile = evaluator.evaluate_values(parameter_values)
            exact_shape, exact_phase, exact_closure = exact_fixed_path_chi2(
                exact_profile,
                anchor_session=0,
                active_sessions=active_sessions,
                integer_offsets=path,
            )
        except Exception:
            continue
        proposal = LatticeProposal(
            root_mode_id=str(root_id),
            parent_mode_id="global-continuous-posterior",
            source=f"bounded-phase-manifold:{source}",
            active_sessions=active_sessions,
            integer_offsets=path,
            initial_normalized_values=np.asarray(conditioned, dtype=float),
            squared_mahalanobis=float(predicted_phase),
            log_predictive_density=-0.5 * float(predicted_shape + predicted_phase),
            kappa=1.0,
        )
        candidates.append(
            BoundedPhaseManifoldPathCandidate(
                candidate_id=f"global-path-{serial:07d}",
                cell_id="global-continuous-posterior",
                source=source,
                proposal=proposal,
                weak_coordinates=np.empty(0, dtype=float),
                predicted_shape_chi2=float(predicted_shape),
                predicted_phase_chi2=float(predicted_phase),
                exact_start_shape_chi2=float(exact_shape),
                exact_start_phase_chi2=float(exact_phase),
                exact_start_closure_chi2=float(exact_closure),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.exact_start_closure_chi2,
                item.predicted_shape_chi2 + item.predicted_phase_chi2,
                item.candidate_id,
            ),
        )[: max(int(active.maximum_global_candidates), 1)]
    )


def acquire_bounded_phase_manifold_paths(
    *,
    evaluator: Any,
    coordinates: Any,
    geometry: ShapeProfileGeometry,
    atlas: BoundedPhaseManifoldAtlas,
    atlas_config: ShapeProfileAtlasConfig | None = None,
    config: BoundedPhaseManifoldConfig | None = None,
) -> tuple[BoundedPhaseManifoldPathCandidate, ...]:
    """Solve complete integer intersections independently inside every cell."""

    shape_config = atlas_config or ShapeProfileAtlasConfig()
    active = config or BoundedPhaseManifoldConfig()
    samples = {item.sample_id: item for item in atlas.samples}
    normalized_lower, normalized_upper = _normalized_bounds(
        coordinates,
        shape_config.support_inset_fraction,
    )
    point_buckets: list[list[BoundedPhaseManifoldPathCandidate]] = []
    serial = 0
    ranked_local_cells = sorted(
        atlas.cells,
        key=lambda item: (
            not item.incumbent_competitive,
            item.acquisition_score,
            item.cell_id,
        ),
    )[: max(int(active.maximum_local_chart_cells), 0)]
    local_config = replace(
        active,
        global_lattice_covariance_scales=(1.0,),
        global_lattice_candidates_per_scale=max(
            int(active.local_chart_lattice_candidates),
            1,
        ),
        maximum_global_prefits=max(int(active.maximum_local_chart_prefits), 1),
        maximum_global_candidates=max(
            int(active.maximum_paths_per_local_chart),
            1,
        ),
    )
    for chart_rank, cell in enumerate(ranked_local_cells):
        center_sample = min(
            (samples[sample_id] for sample_id in cell.sample_ids),
            key=lambda item: float(
                np.linalg.norm(item.weak_coordinates - cell.centroid)
            ),
        )
        local_candidates = acquire_global_marginal_lattice_paths(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=center_sample.normalized_values,
            root_id=atlas.root_id,
            atlas_config=shape_config,
            config=local_config,
        )
        bucket = []
        for local_rank, candidate in enumerate(local_candidates):
            source = f"cell-marginal-r{chart_rank:03d}:{candidate.source}"
            proposal = replace(
                candidate.proposal,
                parent_mode_id=cell.cell_id,
                source=f"bounded-phase-manifold:{source}",
            )
            bucket.append(
                replace(
                    candidate,
                    candidate_id=(
                        f"cell-marginal-{chart_rank:03d}-{local_rank:04d}"
                    ),
                    cell_id=cell.cell_id,
                    source=source,
                    proposal=proposal,
                    weak_coordinates=center_sample.weak_coordinates.copy(),
                )
            )
        point_buckets.append(bucket)

    for cell in atlas.cells:
        if active.require_resolved_cells and not cell.resolved:
            point_buckets.append([])
            continue
        generated: dict[tuple[int, ...], str] = {}
        center_sample = min(
            (samples[sample_id] for sample_id in cell.sample_ids),
            key=lambda item: float(
                np.linalg.norm(item.weak_coordinates - cell.centroid)
            ),
        )
        linearization = finite_difference_profile(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=center_sample.normalized_values,
            step=max(float(shape_config.finite_difference_step), 1.0e-7),
        )
        strong_basis = np.asarray(geometry.strong_basis, dtype=float)
        strong_shape_jacobian = (
            np.asarray(linearization.shape_jacobian, dtype=float) @ strong_basis
        )
        relative_jacobian = (
            np.asarray(linearization.offset_jacobian[1:], dtype=float)
            - np.asarray(linearization.offset_jacobian[0], dtype=float)
        )
        strong_phase_jacobian = relative_jacobian @ strong_basis
        strong_information = strong_shape_jacobian.T @ strong_shape_jacobian
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (strong_information + strong_information.T)
        )
        floor = max(float(np.max(eigenvalues)) * 1.0e-12, 1.0e-12)
        strong_covariance = (
            eigenvectors * (1.0 / np.maximum(eigenvalues, floor))
        ) @ eigenvectors.T
        for sample_id in cell.sample_ids:
            item = samples[sample_id]
            path = tuple(int(value) for value in np.rint(item.relative_session_offsets))
            generated.setdefault(path, "cell-sample")
        phase_samples = np.vstack(
            [samples[sample_id].relative_session_offsets for sample_id in cell.sample_ids]
        )
        phase_center = np.mean(phase_samples, axis=0)
        nonlinear_phase_covariance = np.cov(phase_samples, rowvar=False, ddof=0)
        nonlinear_phase_covariance = 0.5 * (
            nonlinear_phase_covariance + nonlinear_phase_covariance.T
        )
        simplex_covariance = _simplex_uniform_covariance(cell.vertices)
        strong_phase_covariance = (
            strong_phase_jacobian
            @ strong_covariance
            @ strong_phase_jacobian.T
        )
        phase_eigenvalues, phase_eigenvectors = np.linalg.eigh(
            0.5 * (strong_phase_covariance + strong_phase_covariance.T)
        )
        direction_order = np.argsort(phase_eigenvalues)[::-1]
        strong_directions = []
        for direction_index in direction_order[
            : max(int(active.maximum_strong_phase_directions), 1)
        ]:
            phase_variance = float(phase_eigenvalues[direction_index])
            if phase_variance <= floor:
                continue
            phase_vector = phase_eigenvectors[:, direction_index]
            strong_direction = (
                strong_covariance @ strong_phase_jacobian.T @ phase_vector
            ) / math.sqrt(phase_variance)
            shape_norm = math.sqrt(
                max(
                    float(
                        strong_direction
                        @ strong_information
                        @ strong_direction
                    ),
                    0.0,
                )
            )
            if shape_norm <= 0.0:
                continue
            strong_direction /= shape_norm
            strong_directions.append(
                (
                    np.asarray(strong_direction, dtype=float),
                    strong_phase_jacobian @ strong_direction,
                )
            )
        sigma_samples = tuple(
            float(value)
            for value in active.strong_direction_sigma_samples
            if np.isfinite(float(value)) and float(value) > 0.0
        )
        for sample_id in cell.sample_ids:
            item = samples[sample_id]
            for direction_index, (
                strong_direction,
                phase_direction,
            ) in enumerate(strong_directions):
                normalized_direction = strong_basis @ strong_direction
                minimum, maximum = _ray_interval(
                    item.normalized_values,
                    normalized_direction,
                    normalized_lower,
                    normalized_upper,
                )
                coefficients = [minimum, maximum]
                coefficients.extend(
                    value
                    for sigma in sigma_samples
                    for value in (-sigma, sigma)
                    if minimum <= value <= maximum
                )
                for coefficient in coefficients:
                    if not np.isfinite(coefficient):
                        continue
                    path = tuple(
                        int(value)
                        for value in np.rint(
                            item.relative_session_offsets
                            + coefficient * phase_direction
                        )
                    )
                    generated.setdefault(
                        path,
                        f"cell-strong-d{direction_index:02d}",
                    )
        for sample_id in cell.sample_ids:
            item = samples[sample_id]
            delta = item.weak_coordinates - cell.centroid
            weak_phase_jacobian = _quadratic_jacobian(
                cell.phase_jacobian,
                cell.phase_quadratic,
                delta,
            )
            for scale in active.lattice_covariance_scales:
                factor = float(scale)
                if not np.isfinite(factor) or factor <= 0.0:
                    continue
                chart_covariance = (
                    item.relative_offset_covariance
                    + factor
                    * factor
                    * (
                        weak_phase_jacobian
                        @ simplex_covariance
                        @ weak_phase_jacobian.T
                        + strong_phase_covariance
                    )
                )
                chart_covariance = 0.5 * (
                    chart_covariance + chart_covariance.T
                )
                try:
                    chart_lattice = solve_integer_least_squares(
                        item.relative_session_offsets,
                        chart_covariance,
                        candidate_count=1,
                        max_visited_nodes=max(
                            int(active.maximum_lattice_nodes),
                            1,
                        ),
                        return_partial_on_budget=True,
                    )
                except (RuntimeError, ValueError):
                    continue
                for candidate in chart_lattice.candidates:
                    path = tuple(
                        int(value) for value in candidate.integer_vector
                    )
                    generated.setdefault(
                        path,
                        f"cell-chart-{sample_id}-x{factor:g}",
                    )
        for scale in active.lattice_covariance_scales:
            factor = float(scale)
            if not np.isfinite(factor) or factor <= 0.0:
                continue
            integer_covariance = (
                cell.relative_offset_covariance
                + factor * factor
                * (
                    nonlinear_phase_covariance
                    + strong_phase_covariance
                )
            )
            integer_covariance = 0.5 * (
                integer_covariance + integer_covariance.T
            )
            try:
                lattice = solve_integer_least_squares(
                    phase_center,
                    integer_covariance,
                    candidate_count=max(int(active.lattice_candidates_per_scale), 1),
                    max_visited_nodes=max(int(active.maximum_lattice_nodes), 1),
                    return_partial_on_budget=True,
                )
            except (RuntimeError, ValueError):
                continue
            for item in lattice.candidates:
                path = tuple(int(value) for value in item.integer_vector)
                generated.setdefault(path, f"cell-schur-x{factor:g}")

        candidates = []
        conditioned_paths = []
        for path, source in generated.items():
            try:
                (
                    weak,
                    normalized,
                    predicted_shape,
                    predicted_phase,
                ) = _condition_path_in_simplex(
                    cell=cell,
                    integer_offsets=path,
                    strong_basis=strong_basis,
                    strong_shape_jacobian=strong_shape_jacobian,
                    strong_phase_jacobian=strong_phase_jacobian,
                    normalized_lower=normalized_lower,
                    normalized_upper=normalized_upper,
                )
                conditioned_paths.append(
                    (
                        float(predicted_shape + predicted_phase),
                        path,
                        source,
                        weak,
                        normalized,
                        float(predicted_shape),
                        float(predicted_phase),
                    )
                )
            except Exception:
                continue
        prefit_target = max(int(active.maximum_paths_per_cell), 1) * 4
        prefit = []
        seen_prefit: set[tuple[int, ...]] = set()
        source_best = {}
        for item in conditioned_paths:
            current = source_best.get(item[2])
            if current is None or item[0] < current[0]:
                source_best[item[2]] = item
        for item in sorted(source_best.values(), key=lambda value: value[0]):
            if item[1] not in seen_prefit:
                prefit.append(item)
                seen_prefit.add(item[1])
        for item in sorted(conditioned_paths, key=lambda value: value[0]):
            if item[1] in seen_prefit:
                continue
            prefit.append(item)
            seen_prefit.add(item[1])
            if len(prefit) >= prefit_target:
                break

        for (
            _,
            path,
            source,
            weak,
            normalized,
            predicted_shape,
            predicted_phase,
        ) in prefit[:prefit_target]:
            try:
                normalized = np.clip(normalized, normalized_lower, normalized_upper)
                parameter_values = _physical_values(coordinates, normalized)
                profile = evaluator.evaluate_values(parameter_values)
                shape_chi2, phase_chi2, closure = exact_fixed_path_chi2(
                    profile,
                    anchor_session=0,
                    active_sessions=tuple(range(1, int(profile.session_count))),
                    integer_offsets=path,
                )
            except Exception:
                serial += 1
                continue
            proposal = LatticeProposal(
                root_mode_id=atlas.root_id,
                parent_mode_id=cell.cell_id,
                source=f"bounded-phase-manifold:{source}",
                active_sessions=tuple(range(1, int(profile.session_count))),
                integer_offsets=path,
                initial_normalized_values=np.asarray(normalized, dtype=float),
                squared_mahalanobis=float(phase_chi2),
                log_predictive_density=-0.5 * float(closure),
                kappa=1.0,
            )
            candidates.append(
                BoundedPhaseManifoldPathCandidate(
                    candidate_id=f"bounded-path-{serial:07d}",
                    cell_id=cell.cell_id,
                    source=source,
                    proposal=proposal,
                    weak_coordinates=weak,
                    predicted_shape_chi2=predicted_shape,
                    predicted_phase_chi2=predicted_phase,
                    exact_start_shape_chi2=float(shape_chi2),
                    exact_start_phase_chi2=float(phase_chi2),
                    exact_start_closure_chi2=float(closure),
                )
            )
            serial += 1
        point_buckets.append(
            sorted(
                candidates,
                key=lambda item: (
                    item.exact_start_closure_chi2,
                    item.predicted_shape_chi2 + item.predicted_phase_chi2,
                    item.candidate_id,
                ),
            )[: max(int(active.maximum_paths_per_cell), 1)]
        )

    selected = []
    seen_paths: set[tuple[int, ...]] = set()
    target = max(int(active.maximum_candidates), 1)
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
    "BoundedPhaseManifoldAtlas",
    "BoundedPhaseManifoldCell",
    "BoundedPhaseManifoldConfig",
    "BoundedPhaseManifoldPathCandidate",
    "PhaseManifoldSample",
    "acquire_bounded_phase_manifold_paths",
    "acquire_global_marginal_lattice_paths",
    "build_bounded_phase_manifold_atlas",
]
