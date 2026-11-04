from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinIntegerMarginalEvaluation,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
)


@dataclass(frozen=True)
class ShapeTruncatedManifoldConfig:
    """Controls for local charts of an exact nonlinear shape posterior."""

    shape_chi2_threshold: float = 200.0
    maximum_weak_dimensions: int = 3
    weak_singular_value_ratio: float = 30.0
    chart_shape_delta_chi2: float = 25.0
    maximum_chart_step: float = 0.10
    minimum_chart_step: float = 5.0e-4
    support_inset_fraction: float = 1.0e-8
    profile_max_iterations: int = 160
    profile_constraint_tolerance: float = 2.0e-6
    integer_candidates: int = 128
    maximum_lattice_nodes: int = 2_000_000


@dataclass(frozen=True)
class ShapeTruncatedChartGeometry:
    center_normalized_values: np.ndarray
    shape_chi2: float
    shape_jacobian: np.ndarray
    singular_values: np.ndarray
    right_basis: np.ndarray
    weak_basis: np.ndarray
    strong_basis: np.ndarray
    chart_half_widths: np.ndarray
    normalized_lower: np.ndarray
    normalized_upper: np.ndarray


@dataclass(frozen=True)
class ShapeTruncatedNeighbor:
    normalized_values: np.ndarray
    shape_chi2: float
    profile_success: bool
    profile_iterations: int
    constraint_error: float


@dataclass(frozen=True)
class ShapeTruncatedPathProposal:
    integer_offsets: tuple[int, ...]
    squared_mahalanobis: float
    posterior_weight: float
    displacement: np.ndarray
    proposed_normalized_values: np.ndarray
    predicted_shape_chi2: float
    chart_radius_squared: float
    inside_bounds: bool
    inside_linear_shape_support: bool
    inside_chart_trust_region: bool


def _finite_bounds(
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    *,
    inset_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower_values, dtype=float)
    upper = np.asarray(upper_values, dtype=float)
    if (
        lower.ndim != 1
        or upper.shape != lower.shape
        or not np.all(np.isfinite(lower) & np.isfinite(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Shape-truncated charts require finite ordered bounds.")
    fraction = max(float(inset_fraction), 0.0)
    inset = np.maximum(fraction * (upper - lower), 1.0e-12)
    return lower + inset, upper - inset


def _ray_half_width(
    center: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    positive = math.inf
    negative = math.inf
    for value, component, low, high in zip(
        center,
        direction,
        lower,
        upper,
        strict=True,
    ):
        coefficient = float(component)
        if coefficient > 0.0:
            positive = min(positive, (high - value) / coefficient)
            negative = min(negative, (value - low) / coefficient)
        elif coefficient < 0.0:
            positive = min(positive, (low - value) / coefficient)
            negative = min(negative, (value - high) / coefficient)
    return max(min(float(positive), float(negative)), 0.0)


def build_shape_truncated_chart_geometry(
    linearization: SpinOrbitLinearization,
    *,
    normalized_lower: Sequence[float],
    normalized_upper: Sequence[float],
    config: ShapeTruncatedManifoldConfig | None = None,
) -> ShapeTruncatedChartGeometry:
    """Build a local chart whose scale is set by exact shape information."""

    active = config or ShapeTruncatedManifoldConfig()
    center = np.asarray(linearization.spin_reference, dtype=float)
    residuals = np.asarray(linearization.profile.normalized_residuals, dtype=float)
    jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    if residuals.ndim != 1 or jacobian.shape[0] != residuals.size:
        raise ValueError("Shape residuals and Jacobian must be aligned.")
    if jacobian.shape[1] != center.size:
        raise ValueError("Shape Jacobian and chart center dimensions differ.")
    lower, upper = _finite_bounds(
        normalized_lower,
        normalized_upper,
        inset_fraction=active.support_inset_fraction,
    )
    if center.shape != lower.shape or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Shape chart center lies outside normalized support.")

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
    right_basis = np.asarray(right.T, dtype=float)
    weak_basis = right_basis[:, -weak_count:]
    strong_basis = right_basis[:, :-weak_count]

    target_delta = max(float(active.chart_shape_delta_chi2), 1.0e-12)
    maximum_step = max(float(active.maximum_chart_step), 1.0e-12)
    minimum_step = max(float(active.minimum_chart_step), 1.0e-12)
    half_widths = np.empty(dimension, dtype=float)
    for index, direction in enumerate(right_basis.T):
        singular_value = float(padded[index])
        fisher_width = (
            math.sqrt(target_delta) / singular_value
            if singular_value > np.finfo(float).tiny
            else maximum_step
        )
        support_width = _ray_half_width(center, direction, lower, upper)
        half_widths[index] = min(
            max(fisher_width, minimum_step),
            maximum_step,
            support_width,
        )
    if np.any(half_widths <= 0.0):
        raise ValueError("Shape chart has no interior trust region.")
    return ShapeTruncatedChartGeometry(
        center_normalized_values=center.copy(),
        shape_chi2=float(residuals @ residuals),
        shape_jacobian=jacobian.copy(),
        singular_values=padded,
        right_basis=right_basis,
        weak_basis=weak_basis,
        strong_basis=strong_basis,
        chart_half_widths=half_widths,
        normalized_lower=lower,
        normalized_upper=upper,
    )


def profile_shape_truncated_neighbor(
    *,
    evaluate_shape_residuals: Callable[[np.ndarray], np.ndarray],
    geometry: ShapeTruncatedChartGeometry,
    weak_direction_index: int,
    direction_sign: int,
    step_fraction: float = 1.0,
    config: ShapeTruncatedManifoldConfig | None = None,
) -> ShapeTruncatedNeighbor:
    """Move one local tangent step and re-profile every strong coordinate."""

    active = config or ShapeTruncatedManifoldConfig()
    direction_index = int(weak_direction_index)
    if not 0 <= direction_index < geometry.weak_basis.shape[1]:
        raise IndexError("Weak direction index is outside the chart basis.")
    sign = 1.0 if int(direction_sign) >= 0 else -1.0
    basis_index = geometry.right_basis.shape[1] - geometry.weak_basis.shape[1]
    basis_index += direction_index
    step = (
        sign
        * max(float(step_fraction), 0.0)
        * float(geometry.chart_half_widths[basis_index])
    )
    target = (
        geometry.center_normalized_values
        + step * geometry.weak_basis[:, direction_index]
    )
    target = np.clip(
        target,
        geometry.normalized_lower,
        geometry.normalized_upper,
    )
    projection = geometry.weak_basis.T
    fixed = projection @ target
    constraint = LinearConstraint(projection, fixed, fixed)

    def objective(values: np.ndarray) -> float:
        try:
            residuals = np.asarray(evaluate_shape_residuals(values), dtype=float)
        except Exception:
            return 1.0e30
        if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
            return 1.0e30
        return float(residuals @ residuals)

    target_shape_chi2 = objective(target)
    target_constraint_error = float(np.max(np.abs(projection @ target - fixed)))
    if (
        np.isfinite(target_shape_chi2)
        and target_shape_chi2 <= float(active.shape_chi2_threshold)
        and target_constraint_error
        <= max(float(active.profile_constraint_tolerance), 0.0)
    ):
        return ShapeTruncatedNeighbor(
            normalized_values=target,
            shape_chi2=target_shape_chi2,
            profile_success=True,
            profile_iterations=0,
            constraint_error=target_constraint_error,
        )

    optimization = minimize(
        objective,
        target,
        method="SLSQP",
        bounds=Bounds(geometry.normalized_lower, geometry.normalized_upper),
        constraints=(constraint,),
        options={
            "maxiter": max(int(active.profile_max_iterations), 1),
            "ftol": 1.0e-10,
            "disp": False,
        },
    )
    values = np.asarray(optimization.x, dtype=float)
    shape_chi2 = objective(values)
    constraint_error = float(np.max(np.abs(projection @ values - fixed)))
    success = bool(
        optimization.success
        and np.isfinite(shape_chi2)
        and shape_chi2 <= float(active.shape_chi2_threshold)
        and constraint_error
        <= max(float(active.profile_constraint_tolerance), 0.0)
    )
    return ShapeTruncatedNeighbor(
        normalized_values=values,
        shape_chi2=shape_chi2,
        profile_success=success,
        profile_iterations=int(getattr(optimization, "nit", 0)),
        constraint_error=constraint_error,
    )


def chart_prior_covariance(
    geometry: ShapeTruncatedChartGeometry,
) -> np.ndarray:
    """Return a Gaussian chart measure aligned with its local shape axes."""

    variances = np.square(np.asarray(geometry.chart_half_widths, dtype=float))
    covariance = geometry.right_basis @ np.diag(variances) @ geometry.right_basis.T
    return 0.5 * (covariance + covariance.T)


def marginalize_shape_truncated_integer_paths(
    linearization: SpinOrbitLinearization,
    geometry: ShapeTruncatedChartGeometry,
    *,
    config: ShapeTruncatedManifoldConfig | None = None,
) -> tuple[SpinIntegerMarginalEvaluation, tuple[ShapeTruncatedPathProposal, ...]]:
    """Marginalize complete paths, retaining explicit chart-support diagnostics."""

    active = config or ShapeTruncatedManifoldConfig()
    dimension = geometry.center_normalized_values.size
    marginal = marginalize_spin_and_integer_paths(
        linearization,
        spin_prior_mean=np.zeros(dimension, dtype=float),
        spin_prior_covariance=chart_prior_covariance(geometry),
        config=SpinIntegerMarginalConfig(
            initial_lattice_candidates=max(int(active.integer_candidates), 1),
            maximum_lattice_candidates=max(int(active.integer_candidates), 1),
            maximum_lattice_nodes=max(int(active.maximum_lattice_nodes), 1),
        ),
    )
    proposals = []
    inverse_width = 1.0 / np.asarray(geometry.chart_half_widths, dtype=float)
    for path_mode in marginal.path_modes:
        displacement = np.asarray(path_mode.spin_posterior_mean, dtype=float)
        proposed = geometry.center_normalized_values + displacement
        linear_shape = (
            np.asarray(linearization.profile.normalized_residuals, dtype=float)
            + geometry.shape_jacobian @ displacement
        )
        coordinates = geometry.right_basis.T @ displacement
        radius_squared = float(np.sum(np.square(coordinates * inverse_width)))
        inside_bounds = bool(
            np.all(proposed >= geometry.normalized_lower)
            and np.all(proposed <= geometry.normalized_upper)
        )
        predicted_shape = float(linear_shape @ linear_shape)
        proposals.append(
            ShapeTruncatedPathProposal(
                integer_offsets=tuple(path_mode.integer_offsets),
                squared_mahalanobis=float(path_mode.squared_mahalanobis),
                posterior_weight=float(path_mode.posterior_weight),
                displacement=displacement.copy(),
                proposed_normalized_values=proposed,
                predicted_shape_chi2=predicted_shape,
                chart_radius_squared=radius_squared,
                inside_bounds=inside_bounds,
                inside_linear_shape_support=bool(
                    predicted_shape <= float(active.shape_chi2_threshold)
                ),
                inside_chart_trust_region=bool(radius_squared <= 1.0),
            )
        )
    return marginal, tuple(proposals)


__all__ = [
    "ShapeTruncatedChartGeometry",
    "ShapeTruncatedManifoldConfig",
    "ShapeTruncatedNeighbor",
    "ShapeTruncatedPathProposal",
    "build_shape_truncated_chart_geometry",
    "chart_prior_covariance",
    "marginalize_shape_truncated_integer_paths",
    "profile_shape_truncated_neighbor",
]
