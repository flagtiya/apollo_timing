from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


def _finite_vector(values: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector of length {size}.")
    return result


@dataclass(frozen=True)
class FixedPulseTimingEvaluation:
    parameter_values: np.ndarray
    raw_phase_residuals: np.ndarray
    centered_phase_residuals: np.ndarray
    normalized_residuals: np.ndarray
    common_phase_cycles: float
    chi2: float
    efac_mle: float
    peak_to_peak_cycles: float
    maximum_absolute_cycles: float


@dataclass(frozen=True)
class IdentifiableDirectionProposal:
    normalized_step: np.ndarray
    identifiable_rank: int
    singular_values: np.ndarray
    information_ratios: np.ndarray
    residual_scores_sigma: np.ndarray
    predicted_chi2: float
    predicted_delta_chi2: float
    direction_loadings: tuple[dict[str, float], ...]


def fixed_pulse_timing_evaluation(
    *,
    parameter_values: Sequence[float],
    raw_phase_residuals: Sequence[float],
    phase_sigmas: Sequence[float],
) -> FixedPulseTimingEvaluation:
    raw = np.asarray(raw_phase_residuals, dtype=float)
    sigma = np.asarray(phase_sigmas, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("Fixed-pulse phase residuals must be a nonempty finite vector.")
    if sigma.shape != raw.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Fixed-pulse phase sigmas must be finite and positive.")
    values = np.asarray(parameter_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Fixed-pulse parameter values must be a finite vector.")
    weights = 1.0 / np.square(sigma)
    common = float(np.sum(weights * raw) / np.sum(weights))
    centered = raw - common
    normalized = centered / sigma
    chi2 = float(normalized @ normalized)
    return FixedPulseTimingEvaluation(
        parameter_values=values.copy(),
        raw_phase_residuals=raw.copy(),
        centered_phase_residuals=centered,
        normalized_residuals=normalized,
        common_phase_cycles=common,
        chi2=chi2,
        efac_mle=float(math.sqrt(chi2 / raw.size)),
        peak_to_peak_cycles=float(np.ptp(centered)),
        maximum_absolute_cycles=float(np.max(np.abs(centered))),
    )


def evaluate_fixed_pulse_numbers(
    evaluator: object,
    parameter_values: Sequence[float],
) -> FixedPulseTimingEvaluation:
    values = _finite_vector(
        parameter_values,
        len(evaluator.parameter_names),
        "PINT fixed-pulse parameters",
    )
    return fixed_pulse_timing_evaluation(
        parameter_values=values,
        raw_phase_residuals=evaluator.raw_phase_residuals(values),
        phase_sigmas=evaluator.reported_phase_sigmas(),
    )


def scaled_central_jacobian(
    *,
    plus_residuals: Sequence[Sequence[float]],
    minus_residuals: Sequence[Sequence[float]],
    normalized_steps: Sequence[float],
) -> np.ndarray:
    plus = np.asarray(plus_residuals, dtype=float)
    minus = np.asarray(minus_residuals, dtype=float)
    steps = np.asarray(normalized_steps, dtype=float)
    if plus.ndim != 2 or minus.shape != plus.shape or plus.shape[0] != steps.size:
        raise ValueError("Central-difference residuals and steps are not aligned.")
    if (
        not np.all(np.isfinite(plus))
        or not np.all(np.isfinite(minus))
        or not np.all(np.isfinite(steps))
        or np.any(steps <= 0.0)
    ):
        raise ValueError("Central-difference inputs must be finite with positive steps.")
    return ((plus - minus) / (2.0 * steps[:, np.newaxis])).T


def _project_columns(
    residual: np.ndarray,
    jacobian: np.ndarray,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    selected = tuple(int(index) for index in indices)
    if not selected:
        return residual.copy(), jacobian.copy()
    basis, _ = np.linalg.qr(jacobian[:, selected], mode="reduced")
    return (
        residual - basis @ (basis.T @ residual),
        jacobian - basis @ (basis.T @ jacobian),
    )


def analyze_parameter_blocks(
    *,
    residuals: Sequence[float],
    jacobian: Sequence[Sequence[float]],
    parameter_names: Sequence[str],
    blocks: Mapping[str, Sequence[str]],
    noise_scale: float,
    conditional_on: Sequence[str] = (),
    minimum_information_ratio: float = 1.0,
    relative_singular_tolerance: float = 1.0e-9,
) -> dict[str, dict[str, object]]:
    residual = np.asarray(residuals, dtype=float)
    matrix = np.asarray(jacobian, dtype=float)
    names = tuple(str(name) for name in parameter_names)
    if matrix.shape != (residual.size, len(names)):
        raise ValueError("Block Jacobian does not match residuals and parameters.")
    scale = float(noise_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Block-analysis noise scale must be positive.")
    index = {name: position for position, name in enumerate(names)}
    conditional_indices = [index[name] for name in conditional_on if name in index]
    projected_residual, projected_matrix = _project_columns(
        residual,
        matrix,
        conditional_indices,
    )
    diagnostics: dict[str, dict[str, object]] = {}
    for block_name, members in blocks.items():
        active_names = tuple(name for name in members if name in index)
        active_indices = [index[name] for name in active_names]
        if not active_indices:
            continue
        block_matrix = projected_matrix[:, active_indices]
        left, singular, _ = np.linalg.svd(block_matrix, full_matrices=False)
        threshold = max(
            float(singular[0]) * float(relative_singular_tolerance)
            if singular.size
            else math.inf,
            scale * float(minimum_information_ratio),
        )
        rank = int(np.count_nonzero(singular >= threshold))
        score = left.T @ projected_residual
        gain = float(np.sum(np.square(score[:rank]))) if rank else 0.0
        diagnostics[str(block_name)] = {
            "parameters": active_names,
            "identifiable_rank": rank,
            "singular_values": singular.copy(),
            "information_ratios": singular / scale,
            "conditional_delta_chi2": gain,
            "conditional_delta_chi2_efac_scaled": gain / (scale * scale),
            "conditional_delta_bic_approx": (
                gain / (scale * scale) - rank * math.log(max(residual.size, 2))
            ),
        }
    return diagnostics


def propose_identifiable_direction(
    *,
    residuals: Sequence[float],
    jacobian: Sequence[Sequence[float]],
    parameter_names: Sequence[str],
    active_parameters: Sequence[str],
    noise_scale: float,
    trust_radius: float = 0.25,
    minimum_information_ratio: float = 1.0,
    relative_singular_tolerance: float = 1.0e-9,
) -> IdentifiableDirectionProposal:
    residual = np.asarray(residuals, dtype=float)
    matrix = np.asarray(jacobian, dtype=float)
    names = tuple(str(name) for name in parameter_names)
    if matrix.shape != (residual.size, len(names)):
        raise ValueError("Direction Jacobian does not match residuals and parameters.")
    scale = float(noise_scale)
    radius = float(trust_radius)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Direction noise scale and trust radius must be positive.")
    index = {name: position for position, name in enumerate(names)}
    active_names = tuple(name for name in active_parameters if name in index)
    if not active_names:
        raise ValueError("Direction proposal has no active parameters.")
    active_indices = [index[name] for name in active_names]
    active_matrix = matrix[:, active_indices]
    left, singular, right = np.linalg.svd(active_matrix, full_matrices=False)
    threshold = max(
        float(singular[0]) * float(relative_singular_tolerance),
        scale * float(minimum_information_ratio),
    )
    rank = int(np.count_nonzero(singular >= threshold))
    scores = left.T @ residual
    coefficients = np.zeros(right.shape[0], dtype=float)
    if rank:
        coefficients[:rank] = (
            -singular[:rank]
            * scores[:rank]
            / (np.square(singular[:rank]) + scale * scale)
        )
    active_step = right.T @ coefficients
    norm = float(np.max(np.abs(active_step))) if active_step.size else 0.0
    if norm > radius:
        active_step *= radius / norm
    full_step = np.zeros(len(names), dtype=float)
    full_step[active_indices] = active_step
    predicted = residual + matrix @ full_step
    order_count = min(rank, 12)
    loadings = []
    for direction in range(order_count):
        order = np.argsort(np.abs(right[direction]))[::-1][: min(6, len(active_names))]
        loadings.append(
            {active_names[position]: float(right[direction, position]) for position in order}
        )
    start_chi2 = float(residual @ residual)
    predicted_chi2 = float(predicted @ predicted)
    return IdentifiableDirectionProposal(
        normalized_step=full_step,
        identifiable_rank=rank,
        singular_values=singular.copy(),
        information_ratios=singular / scale,
        residual_scores_sigma=np.abs(scores) / scale,
        predicted_chi2=predicted_chi2,
        predicted_delta_chi2=predicted_chi2 - start_chi2,
        direction_loadings=tuple(loadings),
    )


__all__ = [
    "FixedPulseTimingEvaluation",
    "IdentifiableDirectionProposal",
    "analyze_parameter_blocks",
    "evaluate_fixed_pulse_numbers",
    "fixed_pulse_timing_evaluation",
    "propose_identifiable_direction",
    "scaled_central_jacobian",
]
