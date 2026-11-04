from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np


@dataclass(frozen=True)
class OrthogonalSessionBlock:
    label: Hashable
    toa_indices: tuple[int, ...]
    reference_time: float
    polynomial_rank: int
    clock_residual: float
    clock_variance: float
    clock_design: np.ndarray
    local_residuals: np.ndarray
    local_design: np.ndarray
    shape_residuals: np.ndarray
    shape_design: np.ndarray
    orthogonal_basis: np.ndarray


@dataclass(frozen=True)
class OrthogonalSessionObservables:
    blocks: tuple[OrthogonalSessionBlock, ...]
    clock_residuals: np.ndarray
    clock_covariance: np.ndarray
    clock_design: np.ndarray
    local_residuals: np.ndarray
    local_design: np.ndarray
    shape_residuals: np.ndarray
    shape_design: np.ndarray
    raw_chi2: float
    transformed_chi2: float
    gaussian_log_normalization: float
    observation_count: int
    parameter_count: int

    @property
    def session_count(self) -> int:
        return len(self.blocks)

    @property
    def transformed_dimension(self) -> int:
        return int(
            self.clock_residuals.size
            + self.local_residuals.size
            + self.shape_residuals.size
        )

    @property
    def integer_invariant_residuals(self) -> np.ndarray:
        return np.concatenate((self.local_residuals, self.shape_residuals))

    @property
    def integer_invariant_design(self) -> np.ndarray:
        return np.vstack((self.local_design, self.shape_design))


def _validated_inputs(
    residuals: Sequence[float],
    sigmas: Sequence[float],
    times: Sequence[float],
    design_matrix: Sequence[Sequence[float]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(residuals, dtype=float)
    sigma = np.asarray(sigmas, dtype=float)
    t = np.asarray(times, dtype=float)
    if y.ndim != 1 or sigma.shape != y.shape or t.shape != y.shape:
        raise ValueError("Residuals, sigmas, and times must be matching vectors.")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(t)):
        raise ValueError("Residuals and times must be finite.")
    if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
        raise ValueError("All residual sigmas must be finite and positive.")
    if design_matrix is None:
        design = np.zeros((y.size, 0), dtype=float)
    else:
        design = np.asarray(design_matrix, dtype=float)
    if design.ndim != 2 or design.shape[0] != y.size:
        raise ValueError("The design matrix must have one row per residual.")
    if not np.all(np.isfinite(design)):
        raise ValueError("The design matrix must be finite.")
    return y, sigma, t, design


def _session_indices(labels: Sequence[Hashable]) -> tuple[tuple[Hashable, np.ndarray], ...]:
    order: list[Hashable] = []
    grouped: dict[Hashable, list[int]] = {}
    for index, label in enumerate(labels):
        try:
            is_new = label not in grouped
        except TypeError as exc:
            raise ValueError("Session labels must be hashable.") from exc
        if is_new:
            order.append(label)
            grouped[label] = []
        grouped[label].append(index)
    if not order:
        raise ValueError("At least one session is required.")
    return tuple((label, np.asarray(grouped[label], dtype=int)) for label in order)


def _orthogonal_session_basis(
    times: np.ndarray,
    inverse_sigmas: np.ndarray,
    *,
    maximum_polynomial_order: int,
    rank_tolerance: float,
) -> tuple[np.ndarray, int, float]:
    weight_sum = float(inverse_sigmas @ inverse_sigmas)
    q_clock = inverse_sigmas / np.sqrt(weight_sum)
    reference = float(
        np.sum(times * inverse_sigmas * inverse_sigmas) / weight_sum
    )
    columns = [q_clock]
    maximum = min(max(int(maximum_polynomial_order), 0), times.size - 1)
    if maximum > 0:
        centered = times - reference
        scale = float(np.max(np.abs(centered)))
        if scale > 0.0:
            polynomial = np.column_stack(
                [inverse_sigmas * (centered / scale) ** order for order in range(1, maximum + 1)]
            )
            polynomial = polynomial - np.outer(q_clock, q_clock @ polynomial)
            u, singular_values, _ = np.linalg.svd(polynomial, full_matrices=False)
            if singular_values.size:
                threshold = max(float(rank_tolerance), 0.0) * singular_values[0]
                rank = int(np.count_nonzero(singular_values > threshold))
                columns.extend(u[:, column] for column in range(rank))
    local = np.column_stack(columns)
    complete, _ = np.linalg.qr(local, mode="complete")
    for column in range(local.shape[1]):
        if float(complete[:, column] @ local[:, column]) < 0.0:
            complete[:, column] *= -1.0
    return complete, int(local.shape[1]), reference


def build_orthogonal_session_observables(
    residuals: Sequence[float],
    sigmas: Sequence[float],
    times: Sequence[float],
    session_labels: Sequence[Hashable],
    *,
    design_matrix: Sequence[Sequence[float]] | None = None,
    maximum_polynomial_order: int = 2,
    rank_tolerance: float = 1.0e-10,
) -> OrthogonalSessionObservables:
    """Factor one Gaussian TOA likelihood into disjoint session coordinates."""

    y, sigma, t, design = _validated_inputs(residuals, sigmas, times, design_matrix)
    labels = tuple(session_labels)
    if len(labels) != y.size:
        raise ValueError("Session labels must match the residual vector.")
    blocks: list[OrthogonalSessionBlock] = []
    clock_residuals: list[float] = []
    clock_variances: list[float] = []
    clock_design: list[np.ndarray] = []
    local_residuals: list[np.ndarray] = []
    local_design: list[np.ndarray] = []
    shape_residuals: list[np.ndarray] = []
    shape_design: list[np.ndarray] = []

    for label, indices in _session_indices(labels):
        inverse = 1.0 / sigma[indices]
        whitened_y = y[indices] * inverse
        whitened_design = design[indices, :] * inverse[:, None]
        basis, local_rank, reference = _orthogonal_session_basis(
            t[indices],
            inverse,
            maximum_polynomial_order=maximum_polynomial_order,
            rank_tolerance=rank_tolerance,
        )
        coordinates = basis.T @ whitened_y
        coordinate_design = basis.T @ whitened_design
        clock_scale = float(np.linalg.norm(inverse))
        clock_mean = float(coordinates[0] / clock_scale)
        clock_row = coordinate_design[0, :] / clock_scale
        local_values = coordinates[1:local_rank]
        local_rows = coordinate_design[1:local_rank, :]
        shape_values = coordinates[local_rank:]
        shape_rows = coordinate_design[local_rank:, :]
        blocks.append(
            OrthogonalSessionBlock(
                label=label,
                toa_indices=tuple(int(value) for value in indices),
                reference_time=reference,
                polynomial_rank=local_rank,
                clock_residual=clock_mean,
                clock_variance=1.0 / (clock_scale * clock_scale),
                clock_design=clock_row.copy(),
                local_residuals=local_values.copy(),
                local_design=local_rows.copy(),
                shape_residuals=shape_values.copy(),
                shape_design=shape_rows.copy(),
                orthogonal_basis=basis.copy(),
            )
        )
        clock_residuals.append(clock_mean)
        clock_variances.append(1.0 / (clock_scale * clock_scale))
        clock_design.append(clock_row)
        local_residuals.append(local_values)
        local_design.append(local_rows)
        shape_residuals.append(shape_values)
        shape_design.append(shape_rows)

    parameter_count = int(design.shape[1])
    local_vector = np.concatenate(local_residuals) if local_residuals else np.zeros(0)
    shape_vector = np.concatenate(shape_residuals) if shape_residuals else np.zeros(0)
    local_matrix = np.vstack(local_design) if any(row.size for row in local_design) else np.zeros((0, parameter_count))
    shape_matrix = np.vstack(shape_design) if any(row.size for row in shape_design) else np.zeros((0, parameter_count))
    clock_vector = np.asarray(clock_residuals, dtype=float)
    clock_matrix = np.vstack(clock_design)
    clock_covariance = np.diag(np.asarray(clock_variances, dtype=float))
    raw_chi2 = float(np.sum((y / sigma) ** 2))
    transformed_chi2 = float(
        np.sum(clock_vector * clock_vector / np.diag(clock_covariance))
        + local_vector @ local_vector
        + shape_vector @ shape_vector
    )
    return OrthogonalSessionObservables(
        blocks=tuple(blocks),
        clock_residuals=clock_vector,
        clock_covariance=clock_covariance,
        clock_design=clock_matrix,
        local_residuals=local_vector,
        local_design=local_matrix,
        shape_residuals=shape_vector,
        shape_design=shape_matrix,
        raw_chi2=raw_chi2,
        transformed_chi2=transformed_chi2,
        gaussian_log_normalization=float(
            -0.5 * np.sum(np.log(2.0 * np.pi * sigma * sigma))
        ),
        observation_count=int(y.size),
        parameter_count=parameter_count,
    )


__all__ = [
    "OrthogonalSessionBlock",
    "OrthogonalSessionObservables",
    "build_orthogonal_session_observables",
]
