from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Mapping, Sequence

import numpy as np

from timing_accelerator.inference.active_timing_contracts import ActiveTimingBranch


@dataclass(frozen=True)
class BayesianAliasLatticeConfig:
    beam_size: int = 5000
    phase_sigma: float = 0.08
    prior_scale: float = 1.0
    radius_sigma: float = 5.0
    min_radius: int = 4
    max_enum_per_gap: int = 101
    wrap_prior_scale: float = 200.0
    top_k: int = 16
    gap_order: tuple[int, ...] | None = None
    sphere_max_nodes: int = 1_000_000


def _branch_id(index: int) -> str:
    return f"branch_{index:04d}"


def _ordered_gap_indices(scaled_gap_jacobian: np.ndarray) -> list[int]:
    row_ranges = np.sum(np.abs(scaled_gap_jacobian), axis=1)
    return [int(index) for index in np.argsort(row_ranges)]


def _stable_covariance(matrix: np.ndarray) -> np.ndarray:
    symmetrized = 0.5 * (matrix + matrix.T)
    return np.asarray(symmetrized, dtype=float)


def _condition_on_local_phase_slopes(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    parameter_scales: np.ndarray,
    local_phase_slope_jacobian: np.ndarray | None,
    reference_local_phase_slopes: np.ndarray | None,
    local_phase_slope_sigmas: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    supplied = (
        local_phase_slope_jacobian is not None,
        reference_local_phase_slopes is not None,
        local_phase_slope_sigmas is not None,
    )
    if not any(supplied):
        return (
            np.asarray(mean, dtype=float),
            np.asarray(covariance, dtype=float),
            0.0,
            np.zeros(0, dtype=float),
        )
    if not all(supplied):
        raise ValueError("Local phase-slope Jacobian, residuals, and sigmas must be supplied together.")

    active_mean = np.asarray(mean, dtype=float)
    active_covariance = _stable_covariance(np.asarray(covariance, dtype=float))
    scales = np.asarray(parameter_scales, dtype=float)
    jacobian = np.asarray(local_phase_slope_jacobian, dtype=float)
    reference = np.asarray(reference_local_phase_slopes, dtype=float)
    sigmas = np.asarray(local_phase_slope_sigmas, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[1] != active_mean.size:
        raise ValueError("local_phase_slope_jacobian must match the parameter dimension.")
    if reference.shape != (jacobian.shape[0],) or sigmas.shape != reference.shape:
        raise ValueError("Local phase-slope residuals and sigmas must match the Jacobian rows.")
    if scales.shape != active_mean.shape:
        raise ValueError("parameter_scales must match the local phase-slope Jacobian columns.")
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise ValueError("Local phase-slope sigmas must be finite and positive.")
    if jacobian.shape[0] == 0:
        return active_mean, active_covariance, 0.0, np.zeros(0, dtype=float)

    scaled_jacobian = jacobian * scales[None, :]
    identity = np.eye(active_mean.size, dtype=float)
    innovations: list[float] = []
    log_likelihood = 0.0
    for row, observed_reference, sigma in zip(
        scaled_jacobian,
        reference,
        sigmas,
        strict=True,
    ):
        innovation = float(-observed_reference - row @ active_mean)
        measurement_variance = float(sigma * sigma)
        row_covariance = active_covariance @ row
        predicted_variance = float(measurement_variance + row @ row_covariance)
        predicted_variance = max(predicted_variance, measurement_variance, 1.0e-30)
        gain = row_covariance / predicted_variance
        active_mean = active_mean + gain * innovation
        residual_transform = identity - np.outer(gain, row)
        active_covariance = _stable_covariance(
            residual_transform @ active_covariance @ residual_transform.T
            + measurement_variance * np.outer(gain, gain)
        )
        innovations.append(innovation)
        log_likelihood += -0.5 * (
            math.log(2.0 * math.pi * predicted_variance)
            + innovation * innovation / predicted_variance
        )
    return (
        active_mean,
        active_covariance,
        float(log_likelihood),
        np.asarray(innovations, dtype=float),
    )


def _candidate_integer_range(
    predicted_mean: float,
    predicted_sigma: float,
    config: BayesianAliasLatticeConfig,
) -> range:
    radius = max(
        int(config.min_radius),
        int(math.ceil(float(config.radius_sigma) * max(float(predicted_sigma), 1.0e-12))),
    )
    center = int(round(float(predicted_mean)))
    low = center - radius
    high = center + radius
    max_count = max(int(config.max_enum_per_gap), 1)
    if high - low + 1 > max_count:
        half = max_count // 2
        low = center - half
        high = center + (max_count - half - 1)
    return range(low, high + 1)


def _final_wrap_tuple(partial_wraps: dict[int, int], gap_count: int) -> tuple[int, ...]:
    return tuple(int(partial_wraps[index]) for index in range(gap_count))


def _marginal_integer_cost(
    wraps: np.ndarray,
    reference: np.ndarray,
    precision: np.ndarray,
    wrap_prior_scale: float,
) -> float:
    delta = np.asarray(wraps, dtype=float) - np.asarray(reference, dtype=float)
    quadratic_cost = float(delta @ np.asarray(precision, dtype=float) @ delta)
    wrap_cost = 2.0 * float(np.sum(np.abs(wraps))) / max(float(wrap_prior_scale), 1.0e-12)
    return float(quadratic_cost + wrap_cost)


def _babai_integer_vector(reference: np.ndarray, upper_precision_root: np.ndarray) -> np.ndarray:
    wraps = np.zeros(reference.shape[0], dtype=int)
    for index in range(reference.shape[0] - 1, -1, -1):
        trailing = float(
            upper_precision_root[index, index + 1 :]
            @ (wraps[index + 1 :] - reference[index + 1 :])
        )
        center = float(reference[index] - trailing / upper_precision_root[index, index])
        wraps[index] = int(round(center))
    return wraps


def _initial_integer_candidates(
    reference: np.ndarray,
    upper_precision_root: np.ndarray,
    top_k: int,
) -> list[np.ndarray]:
    base = _babai_integer_vector(reference, upper_precision_root)
    candidates = [base.copy()]
    dimension = int(reference.shape[0])
    if top_k <= 1 or dimension <= 0:
        return candidates
    radius = int(math.ceil((max(int(top_k), 1) - 1) / (2.0 * dimension)))
    for step in range(1, radius + 1):
        for index in range(dimension):
            for direction in (-1, 1):
                candidate = base.copy()
                candidate[index] += direction * step
                candidates.append(candidate)
    return candidates


def _centered_integer_window(
    low: int,
    high: int,
    center: float,
    limit: int,
) -> list[int]:
    count = int(high - low + 1)
    if count <= max(int(limit), 0):
        return list(range(low, high + 1))
    active_limit = max(int(limit), 0)
    if active_limit <= 0:
        return []
    window_low = max(low, int(round(center)) - active_limit // 2)
    window_high = min(high, window_low + active_limit - 1)
    window_low = max(low, window_high - active_limit + 1)
    return list(range(window_low, window_high + 1))


def _enumerate_marginal_integer_vectors(
    *,
    reference: np.ndarray,
    precision: np.ndarray,
    wrap_prior_scale: float,
    top_k: int,
    max_nodes: int,
    initial_wraps: Sequence[Sequence[int]] = (),
) -> tuple[list[tuple[float, tuple[int, ...]]], dict[str, object]]:
    dimension = int(reference.shape[0])
    requested = max(int(top_k), 0)
    if requested <= 0:
        return [], {
            "complete": True,
            "visited_node_count": 0,
            "range_truncated": False,
        }

    upper_root = np.linalg.cholesky(_stable_covariance(precision)).T
    external_seeds = [
        np.asarray(wraps, dtype=int)
        for wraps in initial_wraps
    ]
    if any(seed.shape != (dimension,) for seed in external_seeds):
        raise ValueError("Each initial wrap seed must match the gap count.")
    seeded = _initial_integer_candidates(reference, upper_root, requested)
    seeded.extend(external_seeds)
    unique_seed_costs = {
        tuple(int(item) for item in candidate.tolist()): _marginal_integer_cost(
            candidate,
            reference,
            precision,
            wrap_prior_scale,
        )
        for candidate in seeded
    }
    ranked_seeds = sorted(
        ((cost, wraps) for wraps, cost in unique_seed_costs.items()),
        key=lambda item: (item[0], item[1]),
    )[:requested]
    result_heap: list[tuple[float, tuple[int, ...]]] = [
        (-float(cost), wraps) for cost, wraps in ranked_seeds
    ]
    heapq.heapify(result_heap)
    result_wraps = {wraps for _, wraps in ranked_seeds}
    work = np.zeros(dimension, dtype=int)
    visited_nodes = 0
    node_budget_exhausted = False
    range_truncated = False
    active_max_nodes = max(int(max_nodes), 1)
    prior_denominator = max(float(wrap_prior_scale), 1.0e-12)

    def worst_cost() -> float:
        if len(result_heap) < requested:
            return float("inf")
        return float(-result_heap[0][0])

    def record(cost: float, wraps: tuple[int, ...]) -> None:
        if wraps in result_wraps:
            return
        if len(result_heap) < requested:
            heapq.heappush(result_heap, (-float(cost), wraps))
            result_wraps.add(wraps)
            return
        current_worst = worst_cost()
        if float(cost) >= current_worst:
            return
        _, removed_wraps = heapq.heapreplace(result_heap, (-float(cost), wraps))
        result_wraps.remove(removed_wraps)
        result_wraps.add(wraps)

    def visit(index: int, partial_cost: float) -> None:
        nonlocal visited_nodes, node_budget_exhausted, range_truncated
        if node_budget_exhausted:
            return
        if visited_nodes >= active_max_nodes:
            node_budget_exhausted = True
            return
        visited_nodes += 1
        threshold = worst_cost()
        if partial_cost > threshold:
            return
        if index < 0:
            record(
                partial_cost,
                tuple(int(item) for item in work.tolist()),
            )
            return

        trailing = float(
            upper_root[index, index + 1 :]
            @ (work[index + 1 :] - reference[index + 1 :])
        )
        diagonal = float(upper_root[index, index])
        center = float(reference[index] - trailing / diagonal)
        remaining = max(float(threshold - partial_cost), 0.0)
        radius = math.sqrt(remaining) / abs(diagonal)
        low = int(math.ceil(center - radius - 1.0e-12))
        high = int(math.floor(center + radius + 1.0e-12))
        remaining_nodes = max(active_max_nodes - visited_nodes, 0)
        if high - low + 1 > remaining_nodes:
            range_truncated = True
            shrinkage = 1.0 / (prior_denominator * diagonal * diagonal)
            if center > shrinkage:
                local_center = center - shrinkage
            elif center < -shrinkage:
                local_center = center + shrinkage
            else:
                local_center = 0.0
            values = _centered_integer_window(
                low,
                high,
                local_center,
                remaining_nodes,
            )
        else:
            values = list(range(low, high + 1))
        values.sort(
            key=lambda value: (
                (diagonal * (float(value) - reference[index]) + trailing) ** 2
                + 2.0 * abs(int(value)) / prior_denominator,
                abs(float(value) - center),
                int(value),
            )
        )
        for value in values:
            local_cost = float(
                (diagonal * (float(value) - reference[index]) + trailing) ** 2
                + 2.0 * abs(int(value)) / prior_denominator
            )
            if partial_cost + local_cost > worst_cost():
                continue
            work[index] = int(value)
            visit(index - 1, partial_cost + local_cost)
            if node_budget_exhausted:
                return

    visit(dimension - 1, 0.0)
    ranked = sorted(
        [(-negative_cost, wraps) for negative_cost, wraps in result_heap],
        key=lambda item: (item[0], item[1]),
    )
    return ranked, {
        "complete": not node_budget_exhausted and not range_truncated,
        "visited_node_count": int(visited_nodes),
        "node_budget_exhausted": bool(node_budget_exhausted),
        "range_truncated": bool(range_truncated),
        "initial_candidate_count": len(unique_seed_costs),
        "external_seed_count": len(
            {tuple(int(item) for item in seed.tolist()) for seed in external_seeds}
        ),
        "initial_radius_cost": (
            float(ranked_seeds[-1][0]) if ranked_seeds else None
        ),
    }


def generate_marginalized_integer_lattice_branches(
    structure_id: str,
    binary_model: str | None,
    theta_names: list[str],
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: BayesianAliasLatticeConfig | None = None,
    initial_wraps: Sequence[Sequence[int]] | None = None,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
    fixed_wraps: Mapping[int, int] | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, object]]:
    """Generate phase or phase+slope branches with optional prefix conditioning.

    ``fixed_wraps`` represents already certified integer links.  Those links
    are conditioned into the continuous Gaussian posterior before the
    remaining integers are enumerated.  They are therefore not merely
    post-filtered candidates and cannot be rewritten by the continuation
    search.
    """
    active_config = config or BayesianAliasLatticeConfig()
    jacobian = np.asarray(gap_jacobian, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("Marginal integer lattice requires a two-dimensional gap_jacobian.")
    if reference.shape != (jacobian.shape[0],):
        raise ValueError("reference_phase_residuals must match the gap_jacobian row count.")
    if scales.shape != (jacobian.shape[1],):
        raise ValueError("parameter_scales must match the gap_jacobian column count.")
    if jacobian.shape[0] == 0:
        return [], {
            "proposal_source": "bayesian-marginal-integer-lattice",
            "candidate_count": 0,
            "kept_count": 0,
            "complete": True,
            "visited_node_count": 0,
        }

    fixed_by_index = {
        int(index): int(value)
        for index, value in (fixed_wraps or {}).items()
    }
    if any(index < 0 or index >= jacobian.shape[0] for index in fixed_by_index):
        raise ValueError("fixed_wraps indices must refer to existing gaps.")
    if any(
        not isinstance(value, (int, np.integer))
        for value in fixed_by_index.values()
    ):
        raise ValueError("fixed_wraps values must be integers.")

    safe_scales = np.maximum(np.abs(scales), 1.0e-30)
    scaled_jacobian = jacobian * safe_scales
    gap_count = int(jacobian.shape[0])
    dimension = int(jacobian.shape[1])
    gap_order = (
        _ordered_gap_indices(scaled_jacobian)
        if active_config.gap_order is None
        else [int(index) for index in active_config.gap_order]
    )
    if sorted(gap_order) != list(range(gap_count)):
        raise ValueError("gap_order must be a permutation of all gap indices.")
    phase_variance = max(float(active_config.phase_sigma), 1.0e-12) ** 2
    prior_variance = max(float(active_config.prior_scale), 1.0e-12) ** 2
    prior_covariance = prior_variance * np.eye(dimension, dtype=float)
    conditioned_mean, conditioned_covariance, slope_log_likelihood, slope_innovation = (
        _condition_on_local_phase_slopes(
            mean=np.zeros(dimension, dtype=float),
            covariance=prior_covariance,
            parameter_scales=safe_scales,
            local_phase_slope_jacobian=local_phase_slope_jacobian,
            reference_local_phase_slopes=reference_local_phase_slopes,
            local_phase_slope_sigmas=local_phase_slope_sigmas,
        )
    )
    fixed_indices = np.asarray(sorted(fixed_by_index), dtype=int)
    free_indices = np.asarray(
        [index for index in range(gap_count) if index not in fixed_by_index],
        dtype=int,
    )
    fixed_mean = np.asarray(conditioned_mean, dtype=float)
    fixed_covariance = np.asarray(conditioned_covariance, dtype=float)
    fixed_log_evidence = float(slope_log_likelihood)
    fixed_log_normalizer = 0.0
    if fixed_indices.size:
        fixed_jacobian = scaled_jacobian[fixed_indices]
        fixed_reference = reference[fixed_indices] + fixed_jacobian @ fixed_mean
        fixed_covariance_in_observation = (
            phase_variance * np.eye(fixed_indices.size, dtype=float)
            + fixed_jacobian @ fixed_covariance @ fixed_jacobian.T
        )
        fixed_precision = _stable_covariance(
            np.linalg.solve(
                fixed_covariance_in_observation,
                np.eye(fixed_indices.size, dtype=float),
            )
        )
        fixed_delta = np.asarray(
            [fixed_by_index[int(index)] for index in fixed_indices],
            dtype=float,
        ) - fixed_reference
        fixed_mean = fixed_mean + fixed_covariance @ fixed_jacobian.T @ fixed_precision @ fixed_delta
        fixed_covariance = _stable_covariance(
            fixed_covariance
            - fixed_covariance
            @ fixed_jacobian.T
            @ fixed_precision
            @ fixed_jacobian
            @ fixed_covariance
        )
        sign, log_determinant = np.linalg.slogdet(fixed_covariance_in_observation)
        if sign <= 0:
            raise ValueError("Fixed-wrap phase covariance must be positive definite.")
        fixed_log_normalizer = -0.5 * (
            fixed_indices.size * math.log(2.0 * math.pi) + float(log_determinant)
        )
        fixed_log_evidence += (
            fixed_log_normalizer
            - 0.5 * float(fixed_delta @ fixed_precision @ fixed_delta)
            - sum(abs(fixed_by_index[int(index)]) for index in fixed_indices)
            / max(float(active_config.wrap_prior_scale), 1.0e-12)
        )

    if free_indices.size:
        free_jacobian = scaled_jacobian[free_indices]
        free_reference = reference[free_indices] + free_jacobian @ fixed_mean
        marginal_covariance = (
            phase_variance * np.eye(free_indices.size, dtype=float)
            + free_jacobian @ fixed_covariance @ free_jacobian.T
        )
        precision = _stable_covariance(
            np.linalg.solve(
                marginal_covariance,
                np.eye(free_indices.size, dtype=float),
            )
        )
        # The upper-triangular sphere recursion visits coordinates from last to first.
        factor_order = list(reversed([int(index) for index in gap_order if index in set(free_indices.tolist())]))
        ordered_indices = np.asarray(
            [int(np.where(free_indices == index)[0][0]) for index in factor_order],
            dtype=int,
        )
        free_initial_wraps = []
        for wraps in (initial_wraps or ()):
            seed = np.asarray(wraps, dtype=int)
            if seed.shape == (gap_count,):
                seed = seed[free_indices]
            if seed.shape != (free_indices.size,):
                raise ValueError("Each initial wrap seed must match the free gap count or full gap count.")
            free_initial_wraps.append(seed[ordered_indices])
        ordered_ranked, enumeration_diagnostics = _enumerate_marginal_integer_vectors(
            reference=free_reference[ordered_indices],
            precision=precision[np.ix_(ordered_indices, ordered_indices)],
            wrap_prior_scale=float(active_config.wrap_prior_scale),
            top_k=max(int(active_config.top_k), 0),
            max_nodes=max(int(active_config.sphere_max_nodes), 1),
            initial_wraps=free_initial_wraps,
        )
        ranked: list[tuple[float, tuple[int, ...]]] = []
        free_ordered_to_local = np.asarray(ordered_indices, dtype=int)
        for cost, ordered_wraps in ordered_ranked:
            local_wraps = np.zeros(free_indices.size, dtype=int)
            local_wraps[free_ordered_to_local] = np.asarray(ordered_wraps, dtype=int)
            ranked.append((float(cost), tuple(int(item) for item in local_wraps.tolist())))
        ranked.sort(key=lambda item: (item[0], item[1]))
        sign, log_determinant = np.linalg.slogdet(marginal_covariance)
        if sign <= 0:
            raise ValueError("Marginal phase covariance must be positive definite.")
        log_normalizer = -0.5 * (
            free_indices.size * math.log(2.0 * math.pi) + float(log_determinant)
        )
        posterior_covariance = _stable_covariance(
            fixed_covariance
            - fixed_covariance
            @ free_jacobian.T
            @ precision
            @ free_jacobian
            @ fixed_covariance
        )
    else:
        ranked = [(0.0, tuple())] if int(active_config.top_k) > 0 else []
        enumeration_diagnostics = {
            "complete": True,
            "visited_node_count": 0,
            "range_truncated": False,
            "initial_candidate_count": 0,
            "external_seed_count": 0,
            "initial_radius_cost": 0.0,
        }
        factor_order = []
        log_normalizer = 0.0
        posterior_covariance = fixed_covariance

    branches: list[ActiveTimingBranch] = []
    for rank, (cost, free_wraps) in enumerate(ranked):
        wraps_array = np.zeros(gap_count, dtype=int)
        for index, value in fixed_by_index.items():
            wraps_array[index] = value
        if free_indices.size:
            wraps_array[free_indices] = np.asarray(free_wraps, dtype=int)
            free_delta = np.asarray(free_wraps, dtype=float) - free_reference
            posterior_mean = (
                fixed_mean
                + fixed_covariance
                @ free_jacobian.T
                @ precision
                @ free_delta
            )
        else:
            posterior_mean = fixed_mean
        wraps = tuple(int(item) for item in wraps_array.tolist())
        theta_offset = posterior_mean * safe_scales
        predicted_wraps = reference + jacobian @ theta_offset
        residual = predicted_wraps - wraps_array
        log_score = float(fixed_log_evidence + log_normalizer - 0.5 * cost)
        evidence_mode = "phase+slope" if slope_innovation.size else "phase-only"
        branches.append(
            ActiveTimingBranch(
                branch_id=_branch_id(rank),
                structure_id=str(structure_id),
                binary_model=binary_model.upper() if binary_model is not None else None,
                wraps=tuple(int(item) for item in wraps),
                theta_names=tuple(str(name).upper() for name in theta_names),
                theta_offset=np.asarray(theta_offset, dtype=float),
                log_prior=log_score,
                proposal_source="bayesian-marginal-integer-lattice",
                diagnostics={
                    "certification_theta_offset": (
                        np.asarray(theta_offset, dtype=float)
                        if slope_innovation.size
                        else np.zeros(dimension, dtype=float)
                    ),
                    "proposal_score": log_score,
                    "ordered_rank": int(rank),
                    "search_method": "marginalized-integer-sphere",
                    "gap_order": list(gap_order),
                    "factor_order": list(factor_order),
                    "fixed_gap_indices": [int(index) for index in fixed_indices.tolist()],
                    "free_gap_indices": [int(index) for index in free_indices.tolist()],
                    "predicted_wraps": np.asarray(predicted_wraps, dtype=float),
                    "wrap_residual_norm": float(np.linalg.norm(residual)),
                    "posterior_mean_unit_theta": np.asarray(posterior_mean, dtype=float),
                    "posterior_covariance_unit_theta": np.asarray(
                        posterior_covariance,
                        dtype=float,
                    ),
                    "enumeration_complete": bool(enumeration_diagnostics["complete"]),
                    "evidence_mode": evidence_mode,
                    "local_slope_log_likelihood": float(slope_log_likelihood),
                    "local_slope_innovation": np.asarray(slope_innovation, dtype=float),
                },
            )
        )

    return branches, {
        "proposal_source": "bayesian-marginal-integer-lattice",
        "search_method": "marginalized-integer-sphere",
        "candidate_count": len(ranked),
        "hypothesis_count": len(ranked),
        "kept_count": len(branches),
        "gap_order": list(gap_order),
        "factor_order": list(factor_order),
        "fixed_gap_indices": [int(index) for index in fixed_indices.tolist()],
        "free_gap_indices": [int(index) for index in free_indices.tolist()],
        "fixed_wraps": {str(index): int(value) for index, value in fixed_by_index.items()},
        "phase_sigma": float(active_config.phase_sigma),
        "prior_scale": float(active_config.prior_scale),
        "wrap_prior_scale": float(active_config.wrap_prior_scale),
        "sphere_max_nodes": int(active_config.sphere_max_nodes),
        "evidence_mode": "phase+slope" if slope_innovation.size else "phase-only",
        "local_slope_evidence_count": int(slope_innovation.size),
        "local_slope_log_likelihood": float(slope_log_likelihood),
        "fixed_wrap_log_evidence": float(fixed_log_evidence),
        **enumeration_diagnostics,
    }


def generate_bayesian_alias_lattice_branches(
    structure_id: str,
    binary_model: str | None,
    theta_names: list[str],
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: BayesianAliasLatticeConfig | None = None,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, object]]:
    active_config = config or BayesianAliasLatticeConfig()
    jacobian = np.asarray(gap_jacobian, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("Bayesian alias lattice requires a two-dimensional gap_jacobian.")
    if reference.shape != (jacobian.shape[0],):
        raise ValueError("reference_phase_residuals must match the gap_jacobian row count.")
    if scales.shape != (jacobian.shape[1],):
        raise ValueError("parameter_scales must match the gap_jacobian column count.")
    if jacobian.shape[0] == 0:
        return [], {
            "proposal_source": "bayesian-alias-lattice",
            "candidate_count": 0,
            "kept_count": 0,
            "gap_order": [],
        }

    safe_scales = np.maximum(np.abs(scales), 1.0e-30)
    scaled_jacobian = jacobian * safe_scales
    dimension = int(jacobian.shape[1])
    gap_count = int(jacobian.shape[0])
    ordered_gaps = (
        _ordered_gap_indices(scaled_jacobian)
        if active_config.gap_order is None
        else [int(index) for index in active_config.gap_order]
    )
    if sorted(ordered_gaps) != list(range(gap_count)):
        raise ValueError("gap_order must be a permutation of all gap indices.")
    prior_covariance = np.diag([float(active_config.prior_scale) ** 2] * dimension)
    conditioned_mean, conditioned_covariance, slope_log_likelihood, slope_innovation = (
        _condition_on_local_phase_slopes(
            mean=np.zeros(dimension, dtype=float),
            covariance=prior_covariance,
            parameter_scales=safe_scales,
            local_phase_slope_jacobian=local_phase_slope_jacobian,
            reference_local_phase_slopes=reference_local_phase_slopes,
            local_phase_slope_sigmas=local_phase_slope_sigmas,
        )
    )
    beam: list[tuple[float, dict[int, int], np.ndarray, np.ndarray]] = [
        (slope_log_likelihood, {}, conditioned_mean, conditioned_covariance)
    ]
    largest_beam = 1

    for gap_index in ordered_gaps:
        row = scaled_jacobian[gap_index]
        next_beam: list[tuple[float, dict[int, int], np.ndarray, np.ndarray]] = []
        for log_score, partial_wraps, mean, covariance in beam:
            predicted_mean = float(reference[gap_index] + row @ mean)
            row_covariance = covariance @ row
            predicted_variance = float(
                active_config.phase_sigma * active_config.phase_sigma
                + row @ row_covariance
            )
            predicted_variance = max(predicted_variance, 1.0e-18)
            inverse_variance = 1.0 / predicted_variance
            log_normalizer = -0.5 * math.log(2.0 * math.pi * predicted_variance)
            kalman_gain = row_covariance * inverse_variance
            for wrap in _candidate_integer_range(
                predicted_mean=predicted_mean,
                predicted_sigma=math.sqrt(predicted_variance),
                config=active_config,
            ):
                innovation = float(wrap - reference[gap_index] - row @ mean)
                log_predictive = (
                    log_normalizer
                    - 0.5 * innovation * innovation * inverse_variance
                    - abs(int(wrap)) / max(float(active_config.wrap_prior_scale), 1.0e-12)
                )
                updated_wraps = dict(partial_wraps)
                updated_wraps[int(gap_index)] = int(wrap)
                updated_mean = mean + kalman_gain * innovation
                residual_transform = np.eye(dimension, dtype=float) - np.outer(
                    kalman_gain,
                    row,
                )
                updated_covariance = _stable_covariance(
                    residual_transform @ covariance @ residual_transform.T
                    + active_config.phase_sigma
                    * active_config.phase_sigma
                    * np.outer(kalman_gain, kalman_gain)
                )
                next_beam.append(
                    (
                        float(log_score + log_predictive),
                        updated_wraps,
                        updated_mean,
                        updated_covariance,
                    )
                )
        next_beam.sort(key=lambda item: item[0], reverse=True)
        beam = next_beam[: max(int(active_config.beam_size), 1)]
        largest_beam = max(largest_beam, len(beam))

    branches: list[ActiveTimingBranch] = []
    for rank, (log_score, partial_wraps, mean, covariance) in enumerate(
        beam[: max(int(active_config.top_k), 0)]
    ):
        wraps = _final_wrap_tuple(partial_wraps, gap_count)
        theta_offset = mean * safe_scales
        predicted_wraps = reference + jacobian @ theta_offset
        residual = predicted_wraps - np.rint(predicted_wraps)
        branches.append(
            ActiveTimingBranch(
                branch_id=_branch_id(rank),
                structure_id=str(structure_id),
                binary_model=binary_model.upper() if binary_model is not None else None,
                wraps=wraps,
                theta_names=tuple(str(name).upper() for name in theta_names),
                theta_offset=np.asarray(theta_offset, dtype=float),
                log_prior=float(log_score),
                proposal_source="bayesian-alias-lattice",
                diagnostics={
                    "certification_theta_offset": (
                        np.asarray(theta_offset, dtype=float)
                        if slope_innovation.size
                        else np.zeros(dimension, dtype=float)
                    ),
                    "proposal_score": float(log_score),
                    "ordered_rank": int(rank),
                    "gap_order": list(ordered_gaps),
                    "predicted_wraps": np.asarray(predicted_wraps, dtype=float),
                    "wrap_residual_norm": float(np.linalg.norm(residual)),
                    "posterior_mean_unit_theta": np.asarray(mean, dtype=float),
                    "posterior_covariance_unit_theta": np.asarray(covariance, dtype=float),
                    "evidence_mode": "phase+slope" if slope_innovation.size else "phase-only",
                    "local_slope_log_likelihood": float(slope_log_likelihood),
                    "local_slope_innovation": np.asarray(slope_innovation, dtype=float),
                },
            )
        )

    return branches, {
        "proposal_source": "bayesian-alias-lattice",
        "candidate_count": int(largest_beam),
        "hypothesis_count": int(largest_beam),
        "kept_count": len(branches),
        "gap_order": list(ordered_gaps),
        "beam_size": int(active_config.beam_size),
        "phase_sigma": float(active_config.phase_sigma),
        "prior_scale": float(active_config.prior_scale),
        "radius_sigma": float(active_config.radius_sigma),
        "min_radius": int(active_config.min_radius),
        "max_enum_per_gap": int(active_config.max_enum_per_gap),
        "wrap_prior_scale": float(active_config.wrap_prior_scale),
        "evidence_mode": "phase+slope" if slope_innovation.size else "phase-only",
        "local_slope_evidence_count": int(slope_innovation.size),
        "local_slope_log_likelihood": float(slope_log_likelihood),
    }
