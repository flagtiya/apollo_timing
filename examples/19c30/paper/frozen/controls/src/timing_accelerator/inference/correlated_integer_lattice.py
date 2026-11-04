from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class CorrelatedIntegerLatticeConfig:
    """Controls for a static-state Gaussian integer lattice search."""

    candidate_count: int = 64
    beam_width: int = 4096
    tail_sigma: float = 6.0
    maximum_integer_options_per_branch: int = 4096
    covariance_eigenvalue_floor: float = 1.0e-14

    def __post_init__(self) -> None:
        if (
            int(self.candidate_count) <= 0
            or int(self.beam_width) <= 0
            or int(self.maximum_integer_options_per_branch) <= 0
        ):
            raise ValueError("Correlated-lattice budgets must be positive.")
        if not np.isfinite(self.tail_sigma) or float(self.tail_sigma) <= 0.0:
            raise ValueError("The correlated-lattice tail radius must be positive.")
        if (
            not np.isfinite(self.covariance_eigenvalue_floor)
            or float(self.covariance_eigenvalue_floor) <= 0.0
        ):
            raise ValueError("The covariance eigenvalue floor must be positive.")


@dataclass(frozen=True)
class CorrelatedIntegerLatticeCandidate:
    integer_vector: tuple[int, ...]
    state_mean: np.ndarray
    state_covariance: np.ndarray
    squared_mahalanobis: float


@dataclass(frozen=True)
class CorrelatedIntegerLatticeResult:
    candidates: tuple[CorrelatedIntegerLatticeCandidate, ...]
    information_order: tuple[int, ...]
    predictive_standard_deviations: tuple[float, ...]
    retained_branch_counts: tuple[int, ...]
    generated_branch_count: int
    minimum_pruned_lower_bound: float
    tail_truncated: bool
    beam_truncated: bool
    top_k_complete: bool


@dataclass(frozen=True)
class _Branch:
    assignments: tuple[int, ...]
    whitened_mean: np.ndarray
    score: float


def _covariance_geometry(
    covariance: Sequence[Sequence[float]],
    *,
    dimension: int,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return covariance, square root, and inverse square root stably."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("A correlated-lattice covariance has invalid geometry.")
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(symmetric)
    if np.any(diagonal <= 0.0):
        raise ValueError("A correlated-lattice covariance has nonpositive variance.")

    # Scale to a correlation matrix before regularization. The phase intercept
    # and normalized physical coordinates can differ by many orders of
    # magnitude, so one eigenvalue floor in native units changes the posterior.
    scales = np.sqrt(diagonal)
    correlation = symmetric / np.outer(scales, scales)
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum = max(
        float(floor) * eigenvalue_scale,
        np.finfo(float).eps * eigenvalue_scale,
    )
    eigenvalues = np.maximum(eigenvalues, minimum)
    factor = scales[:, np.newaxis] * (
        eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]
    )
    whitener = (
        eigenvectors.T / scales[np.newaxis, :]
    ) / np.sqrt(eigenvalues)[:, np.newaxis]
    positive = factor @ factor.T
    positive = 0.5 * (positive + positive.T)
    return positive, factor, whitener


def _information_order(
    whitened_design: np.ndarray,
    noise_variances: np.ndarray,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    """Choose an order and factor every prefix in square-root information form."""

    dimension = int(whitened_design.shape[1])
    remaining = set(range(whitened_design.shape[0]))
    order: list[int] = []
    standard_deviations: list[float] = []
    gains: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    weighted_rows: list[np.ndarray] = []
    current = np.eye(dimension, dtype=float)
    while remaining:
        predictive = {
            index: float(
                whitened_design[index] @ current @ whitened_design[index]
                + noise_variances[index]
            )
            for index in remaining
        }
        selected = min(remaining, key=lambda index: (predictive[index], index))
        variance = max(predictive[selected], np.finfo(float).eps)
        if not np.isfinite(variance):
            raise RuntimeError(
                "Correlated-lattice predictive variance became non-finite."
            )
        row = whitened_design[selected]
        gain = current @ row / variance
        weighted_rows.append(row / math.sqrt(float(noise_variances[selected])))
        augmented = np.vstack(
            (np.eye(dimension, dtype=float), np.asarray(weighted_rows, dtype=float))
        )
        _left, singular_values, right = np.linalg.svd(
            augmented,
            full_matrices=False,
        )
        current = (
            right.T * (1.0 / np.square(singular_values))[np.newaxis, :]
        ) @ right
        current = 0.5 * (current + current.T)
        order.append(int(selected))
        standard_deviations.append(math.sqrt(variance))
        gains.append(gain)
        covariances.append(current.copy())
        remaining.remove(selected)
    return (
        tuple(order),
        tuple(standard_deviations),
        tuple(gains),
        tuple(covariances),
    )


def _bounded_state_map(
    *,
    integer_vector: np.ndarray,
    centers: np.ndarray,
    design: np.ndarray,
    prior_mean: np.ndarray,
    prior_whitener: np.ndarray,
    noise_variances: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the complete fixed-path continuous MAP inside physical support."""

    inverse_noise_sigma = 1.0 / np.sqrt(noise_variances)
    augmented_design = np.vstack(
        (
            prior_whitener,
            design * inverse_noise_sigma[:, np.newaxis],
        )
    )
    augmented_target = np.concatenate(
        (
            prior_whitener @ prior_mean,
            (integer_vector - centers) * inverse_noise_sigma,
        )
    )
    reference = np.clip(prior_mean, lower, upper)
    column_norms = np.linalg.norm(augmented_design, axis=0)
    variable_scale = 1.0 / np.maximum(column_norms, np.finfo(float).tiny)
    scaled_design = augmented_design * variable_scale[np.newaxis, :]
    scaled_target = augmented_target - augmented_design @ reference
    scaled_lower = (lower - reference) / variable_scale
    scaled_upper = (upper - reference) / variable_scale
    solution = lsq_linear(
        scaled_design,
        scaled_target,
        bounds=(scaled_lower, scaled_upper),
        method="bvls",
        lsq_solver="exact",
        tol=1.0e-12,
        max_iter=2000,
    )
    bounded_state = reference + variable_scale * np.asarray(solution.x, dtype=float)
    if not solution.success or not np.all(np.isfinite(bounded_state)):
        raise RuntimeError(
            "The complete correlated integer path has no stable bounded MAP: "
            f"status={solution.status}, message={solution.message}, "
            f"iterations={solution.nit}, optimality={solution.optimality:.6g}, "
            f"cost={solution.cost:.6g}."
        )
    residual = augmented_design @ bounded_state - augmented_target
    return bounded_state, float(residual @ residual)


def solve_correlated_integer_lattice(
    *,
    centers: Sequence[float],
    design_matrix: Sequence[Sequence[float]],
    state_mean: Sequence[float],
    state_covariance: Sequence[Sequence[float]],
    noise_variances: Sequence[float],
    state_lower: Sequence[float] | None = None,
    state_upper: Sequence[float] | None = None,
    config: CorrelatedIntegerLatticeConfig | None = None,
) -> CorrelatedIntegerLatticeResult:
    """Return complete integer paths under one global linear-Gaussian posterior.

    The unknown continuous state is analytically conditioned for every integer
    branch. Measurements are processed in an information order only to factor
    the same all-row Mahalanobis objective; no session is scored greedily in a
    different target distribution.
    """

    active = config or CorrelatedIntegerLatticeConfig()
    center = np.asarray(centers, dtype=float)
    design = np.asarray(design_matrix, dtype=float)
    mean = np.asarray(state_mean, dtype=float)
    noise = np.asarray(noise_variances, dtype=float)
    if center.ndim != 1 or center.size == 0:
        raise ValueError("At least one integer ambiguity is required.")
    if design.ndim != 2 or design.shape[0] != center.size:
        raise ValueError("The integer design matrix must align with the centers.")
    if mean.shape != (design.shape[1],):
        raise ValueError("The continuous-state mean has the wrong dimension.")
    if noise.shape != center.shape or not np.all(np.isfinite(noise) & (noise > 0.0)):
        raise ValueError("Integer measurement variances must be finite and positive.")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(design)):
        raise ValueError("Correlated-lattice inputs must be finite.")
    covariance, covariance_factor, covariance_whitener = _covariance_geometry(
        state_covariance,
        dimension=mean.size,
        floor=float(active.covariance_eigenvalue_floor),
    )
    if state_lower is None:
        lower = np.full(mean.size, -math.inf, dtype=float)
    else:
        lower = np.asarray(state_lower, dtype=float)
    if state_upper is None:
        upper = np.full(mean.size, math.inf, dtype=float)
    else:
        upper = np.asarray(state_upper, dtype=float)
    if (
        lower.shape != mean.shape
        or upper.shape != mean.shape
        or np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower >= upper)
    ):
        raise ValueError("Continuous-state bounds have invalid geometry.")

    whitened_design = design @ covariance_factor
    order, predictive_sigmas, gains, whitened_posterior_covariances = (
        _information_order(
            whitened_design,
            noise,
        )
    )
    branches = (
        _Branch(
            assignments=tuple(),
            whitened_mean=np.zeros(mean.size, dtype=float),
            score=0.0,
        ),
    )
    generated_count = 0
    minimum_pruned = math.inf
    tail_truncated = False
    beam_truncated = False
    retained_counts: list[int] = []
    assignment_indices: tuple[int, ...] = tuple()
    predictive_center = center + design @ mean

    for stage, row_index in enumerate(order):
        row = whitened_design[row_index]
        variance = float(predictive_sigmas[stage] ** 2)
        gain = gains[stage]
        radius = float(active.tail_sigma) * math.sqrt(variance)
        generated: list[_Branch] = []
        next_assignment_indices = assignment_indices + (row_index,)
        option_cap = max(int(active.maximum_integer_options_per_branch), 1)
        for branch in branches:
            prediction = float(
                predictive_center[row_index] + row @ branch.whitened_mean
            )
            if not np.isfinite(prediction):
                raise RuntimeError(
                    "Correlated-lattice branch prediction became non-finite."
                )
            integer_minimum = int(math.ceil(prediction - radius))
            integer_maximum = int(math.floor(prediction + radius))
            if integer_minimum > integer_maximum:
                nearest = int(np.rint(prediction))
                integer_minimum = nearest
                integer_maximum = nearest
            option_count = integer_maximum - integer_minimum + 1
            if option_count > option_cap:
                tail_truncated = True
                half = option_cap // 2
                nearest = int(np.rint(prediction))
                integer_minimum = nearest - half
                integer_maximum = integer_minimum + option_cap - 1
            for integer in range(integer_minimum, integer_maximum + 1):
                innovation = float(integer - prediction)
                child_mean = branch.whitened_mean + gain * innovation
                assignments = branch.assignments + (int(integer),)
                score = float(branch.score + innovation * innovation / variance)
                generated.append(
                    _Branch(
                        assignments=assignments,
                        whitened_mean=child_mean,
                        score=score,
                    )
                )
        generated_count += len(generated)
        if not generated:
            return CorrelatedIntegerLatticeResult(
                candidates=tuple(),
                information_order=order,
                predictive_standard_deviations=predictive_sigmas,
                retained_branch_counts=tuple(retained_counts),
                generated_branch_count=generated_count,
                minimum_pruned_lower_bound=minimum_pruned,
                tail_truncated=tail_truncated,
                beam_truncated=beam_truncated,
                top_k_complete=False,
            )
        generated.sort(key=lambda branch: (branch.score, branch.assignments))
        width = max(int(active.beam_width), int(active.candidate_count))
        if len(generated) > width:
            beam_truncated = True
            minimum_pruned = min(minimum_pruned, generated[width].score)
            generated = generated[:width]
        branches = tuple(generated)
        retained_counts.append(len(branches))
        assignment_indices = next_assignment_indices

    inverse_order = np.argsort(np.asarray(order, dtype=int))
    final_whitened_covariance = whitened_posterior_covariances[-1]
    final_covariance = (
        covariance_factor @ final_whitened_covariance @ covariance_factor.T
    )
    final_covariance = 0.5 * (final_covariance + final_covariance.T)
    candidate_limit = max(int(active.candidate_count), 1)
    bounded_candidates: list[CorrelatedIntegerLatticeCandidate] = []
    worst_bounded_score = math.inf
    for branch in branches:
        if (
            len(bounded_candidates) >= candidate_limit
            and branch.score >= worst_bounded_score
        ):
            break
        ordered_integers = np.asarray(branch.assignments, dtype=np.int64)[
            inverse_order
        ]
        unbounded_mean = mean + covariance_factor @ branch.whitened_mean
        if np.all(unbounded_mean >= lower) and np.all(unbounded_mean <= upper):
            bounded_mean = unbounded_mean
            bounded_score = float(branch.score)
        else:
            bounded_mean, bounded_score = _bounded_state_map(
                integer_vector=np.asarray(ordered_integers, dtype=float),
                centers=center,
                design=design,
                prior_mean=mean,
                prior_whitener=covariance_whitener,
                noise_variances=noise,
                lower=lower,
                upper=upper,
            )
        bounded_candidates.append(
            CorrelatedIntegerLatticeCandidate(
                integer_vector=tuple(int(value) for value in ordered_integers),
                state_mean=np.asarray(bounded_mean, dtype=float).copy(),
                state_covariance=final_covariance.copy(),
                squared_mahalanobis=float(bounded_score),
            )
        )
        bounded_candidates.sort(
            key=lambda candidate: (
                candidate.squared_mahalanobis,
                candidate.integer_vector,
            )
        )
        del bounded_candidates[candidate_limit:]
        if len(bounded_candidates) >= candidate_limit:
            worst_bounded_score = bounded_candidates[-1].squared_mahalanobis
    candidates = tuple(bounded_candidates)
    worst_retained = max(
        (candidate.squared_mahalanobis for candidate in candidates),
        default=math.inf,
    )
    top_k_complete = bool(
        not tail_truncated
        and (
            not beam_truncated
            or minimum_pruned >= worst_retained
        )
    )
    return CorrelatedIntegerLatticeResult(
        candidates=candidates,
        information_order=order,
        predictive_standard_deviations=predictive_sigmas,
        retained_branch_counts=tuple(retained_counts),
        generated_branch_count=generated_count,
        minimum_pruned_lower_bound=float(minimum_pruned),
        tail_truncated=tail_truncated,
        beam_truncated=beam_truncated,
        top_k_complete=top_k_complete,
    )


__all__ = [
    "CorrelatedIntegerLatticeCandidate",
    "CorrelatedIntegerLatticeConfig",
    "CorrelatedIntegerLatticeResult",
    "solve_correlated_integer_lattice",
]
