from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinOrbitLinearization,
)


@dataclass(frozen=True)
class ProfiledSlowShapeGeometry:
    center: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    shape_chi2: float


def profiled_slow_shape_geometry(
    linearization: SpinOrbitLinearization,
    *,
    slow_dimension: int,
) -> ProfiledSlowShapeGeometry:
    """Profile fast linear coordinates from the local shape Fisher matrix."""

    jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    center = np.asarray(linearization.spin_reference, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] != center.size:
        raise ValueError("Shape Jacobian and reference coordinates must align.")
    split = int(slow_dimension)
    if not 0 < split < center.size:
        raise ValueError("Slow and fast shape dimensions must both be non-empty.")
    hessian = jacobian.T @ jacobian
    slow_slow = hessian[:split, :split]
    slow_fast = hessian[:split, split:]
    fast_fast = hessian[split:, split:]
    fast_eigenvalues, fast_eigenvectors = np.linalg.eigh(
        0.5 * (fast_fast + fast_fast.T)
    )
    fast_floor = max(
        float(np.max(np.abs(fast_eigenvalues))) * 1.0e-12,
        np.finfo(float).eps,
    )
    fast_inverse = (
        fast_eigenvectors
        / np.maximum(fast_eigenvalues, fast_floor)
    ) @ fast_eigenvectors.T
    precision = slow_slow - slow_fast @ fast_inverse @ slow_fast.T
    precision = 0.5 * (precision + precision.T)
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    floor = max(
        float(np.max(np.abs(eigenvalues))) * 1.0e-10,
        np.finfo(float).eps,
    )
    precision = (
        eigenvectors * np.maximum(eigenvalues, floor)
    ) @ eigenvectors.T
    covariance = (
        eigenvectors / np.maximum(eigenvalues, floor)
    ) @ eigenvectors.T
    residuals = np.asarray(
        linearization.profile.normalized_residuals,
        dtype=float,
    )
    return ProfiledSlowShapeGeometry(
        center=center[:split].copy(),
        covariance=0.5 * (covariance + covariance.T),
        precision=precision,
        shape_chi2=float(residuals @ residuals),
    )


__all__ = ["ProfiledSlowShapeGeometry", "profiled_slow_shape_geometry"]
