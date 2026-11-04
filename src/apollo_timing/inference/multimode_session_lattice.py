from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm, qmc

from apollo_timing.inference.active_timing_contracts import (
    ActiveTimingBranch,
    BranchCertification,
)
from apollo_timing.inference.integer_least_squares import (
    build_marginalized_integer_covariance,
    solve_integer_least_squares,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    SessionShapeLinearization,
    integer_session_offsets_to_gap_wraps,
)


@dataclass(frozen=True)
class MultimodeSessionLatticeConfig:
    max_modes: int = 8
    candidates_per_mode: int = 16
    max_candidates: int = 64
    max_visited_nodes: int = 1_000_000
    decimal_precision: int = 80
    finite_difference_fraction: float = 1.0e-4
    global_phase_prior_scale: float = 1.0
    global_phase_relinearization_rounds: int = 0
    global_phase_relinearization_candidates_per_mode: int = 2
    shape_optimization_enabled: bool = False
    shape_optimization_max_modes: int = 4
    shape_optimization_trust_radius: float = 64.0
    shape_optimization_max_nfev: int = 100
    posterior_particle_count: int = 512
    posterior_particle_anchor_fraction: float = 0.25
    posterior_particle_anchor_scale: float = 0.2
    posterior_particle_ess_target: float = 0.7
    posterior_particle_resample_threshold: float = 0.6
    posterior_particle_mutation_scale: float = 0.08
    posterior_particle_mutation_steps: int = 2
    posterior_particle_max_tempering_steps: int = 24
    posterior_particle_evidence_mode: str = "phase+slope"
    posterior_particle_seed: int = 0
    posterior_particle_max_paths: int = 256


@dataclass(frozen=True)
class JointSessionIntegerMode:
    """A nonlinear posterior mode with an explicit integer session path."""

    integer_offsets: tuple[int, ...]
    initial_integer_offsets: tuple[int, ...]
    parameter_values: np.ndarray
    shape_chi2: float
    ambiguity_chi2: float
    objective_chi2: float
    prior_chi2: float
    posterior_chi2: float
    source_mode_index: int
    integer_updates: int
    success: bool
    nfev: int


def select_diverse_certification_branches(
    *,
    branches: Sequence[ActiveTimingBranch],
    exact_scores: Mapping[str, float],
    max_candidates: int,
    exact_fraction: float = 0.5,
    max_per_wrap_path: int = 2,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Select a path-diverse union of lattice and exact-score candidates."""
    pool = list(branches)
    target = min(max(int(max_candidates), 1), len(pool))
    if target >= len(pool):
        return pool, {
            "enabled": False,
            "reason": "pool-within-budget",
            "pool_count": len(pool),
            "selected_count": len(pool),
            "unique_path_count": len({branch.wraps for branch in pool}),
        }

    fraction = min(max(float(exact_fraction), 0.0), 1.0)
    path_cap = max(int(max_per_wrap_path), 1)
    exact_ranked = sorted(
        pool,
        key=lambda branch: (
            float(exact_scores.get(branch.branch_id, math.inf)),
            -float(branch.log_prior),
            branch.branch_id,
        ),
    )
    bayesian_ranked = sorted(
        pool,
        key=lambda branch: (
            -float(branch.log_prior),
            float(exact_scores.get(branch.branch_id, math.inf)),
            branch.branch_id,
        ),
    )
    exact_ranks = {
        branch.branch_id: rank for rank, branch in enumerate(exact_ranked)
    }
    bayesian_ranks = {
        branch.branch_id: rank for rank, branch in enumerate(bayesian_ranked)
    }
    consensus_ranked = sorted(
        pool,
        key=lambda branch: (
            exact_ranks[branch.branch_id] + bayesian_ranks[branch.branch_id],
            max(exact_ranks[branch.branch_id], bayesian_ranks[branch.branch_id]),
            branch.branch_id,
        ),
    )

    selected: list[ActiveTimingBranch] = []
    selected_ids: set[str] = set()
    path_counts: dict[tuple[int, ...], int] = {}

    def take(
        ranked: Sequence[ActiveTimingBranch],
        count: int,
        *,
        enforce_cap: bool,
    ) -> None:
        for branch in ranked:
            if len(selected) >= count:
                break
            if branch.branch_id in selected_ids:
                continue
            current_count = path_counts.get(branch.wraps, 0)
            if enforce_cap and current_count >= path_cap:
                continue
            selected.append(branch)
            selected_ids.add(branch.branch_id)
            path_counts[branch.wraps] = current_count + 1

    exact_target = int(round(target * fraction))
    take(exact_ranked, exact_target, enforce_cap=True)
    take(bayesian_ranked, target, enforce_cap=True)
    take(consensus_ranked, target, enforce_cap=True)
    if len(selected) < target:
        take(consensus_ranked, target, enforce_cap=False)

    return selected, {
        "enabled": True,
        "pool_count": len(pool),
        "selected_count": len(selected),
        "unique_pool_path_count": len({branch.wraps for branch in pool}),
        "unique_selected_path_count": len({branch.wraps for branch in selected}),
        "exact_fraction": fraction,
        "exact_target": exact_target,
        "max_per_wrap_path": path_cap,
    }


def select_refinement_portfolio_branches(
    *,
    branches: Sequence[ActiveTimingBranch],
    exact_scores: Mapping[str, float],
    max_candidates: int,
    exact_fraction: float = 0.5,
    mode_fraction: float = 0.25,
    max_per_wrap_path: int = 1,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Select phase paths from exact, Bayesian, and nonlinear-island views."""
    pool = list(branches)
    target = min(max(int(max_candidates), 1), len(pool))
    if target >= len(pool):
        return pool, {
            "enabled": False,
            "reason": "pool-within-budget",
            "pool_count": len(pool),
            "selected_count": len(pool),
        }

    exact_share = min(max(float(exact_fraction), 0.0), 1.0)
    mode_share = min(max(float(mode_fraction), 0.0), 1.0 - exact_share)
    path_cap = max(int(max_per_wrap_path), 1)
    exact_ranked = sorted(
        pool,
        key=lambda branch: (
            float(exact_scores.get(branch.branch_id, math.inf)),
            -float(branch.log_prior),
            branch.branch_id,
        ),
    )
    bayesian_ranked = sorted(
        pool,
        key=lambda branch: (
            -float(branch.log_prior),
            float(exact_scores.get(branch.branch_id, math.inf)),
            branch.branch_id,
        ),
    )
    exact_ranks = {
        branch.branch_id: rank for rank, branch in enumerate(exact_ranked)
    }
    bayesian_ranks = {
        branch.branch_id: rank for rank, branch in enumerate(bayesian_ranked)
    }
    consensus_ranked = sorted(
        pool,
        key=lambda branch: (
            exact_ranks[branch.branch_id] + bayesian_ranks[branch.branch_id],
            max(exact_ranks[branch.branch_id], bayesian_ranks[branch.branch_id]),
            branch.branch_id,
        ),
    )

    selected: list[ActiveTimingBranch] = []
    selected_ids: set[str] = set()
    path_counts: dict[tuple[int, ...], int] = {}
    source_counts: dict[str, int] = {
        "exact": 0,
        "bayesian": 0,
        "mode": 0,
        "consensus": 0,
    }

    def add(branch: ActiveTimingBranch, source: str, *, enforce_cap: bool = True) -> bool:
        if branch.branch_id in selected_ids:
            return False
        path_count = path_counts.get(branch.wraps, 0)
        if enforce_cap and path_count >= path_cap:
            return False
        selected.append(branch)
        selected_ids.add(branch.branch_id)
        path_counts[branch.wraps] = path_count + 1
        source_counts[source] += 1
        return True

    def take(ranked: Sequence[ActiveTimingBranch], count: int, source: str) -> None:
        for branch in ranked:
            if len(selected) >= count:
                break
            add(branch, source)

    exact_target = int(round(target * exact_share))
    mode_target = int(round(target * mode_share))
    bayesian_target = max(target - mode_target, exact_target)
    take(exact_ranked, exact_target, "exact")
    take(bayesian_ranked, bayesian_target, "bayesian")

    grouped: dict[int, list[ActiveTimingBranch]] = {}
    for branch in consensus_ranked:
        diagnostics = branch.diagnostics or {}
        mode_index = int(diagnostics.get("mode_index", -1))
        grouped.setdefault(mode_index, []).append(branch)
    represented_modes = {
        int((branch.diagnostics or {}).get("mode_index", -1))
        for branch in selected
    }
    group_order = sorted(
        grouped,
        key=lambda mode_index: (
            mode_index in represented_modes,
            min(
                exact_ranks[branch.branch_id] + bayesian_ranks[branch.branch_id]
                for branch in grouped[mode_index]
            ),
            mode_index,
        ),
    )
    group_positions = {mode_index: 0 for mode_index in group_order}
    while len(selected) < target and group_order:
        progressed = False
        for mode_index in group_order:
            group = grouped[mode_index]
            position = group_positions[mode_index]
            while position < len(group):
                branch = group[position]
                position += 1
                group_positions[mode_index] = position
                if add(branch, "mode"):
                    progressed = True
                    break
            if len(selected) >= target:
                break
        if not progressed:
            break

    take(consensus_ranked, target, "consensus")
    if len(selected) < target:
        for branch in consensus_ranked:
            if len(selected) >= target:
                break
            add(branch, "consensus", enforce_cap=False)

    return selected, {
        "enabled": True,
        "pool_count": len(pool),
        "selected_count": len(selected),
        "unique_pool_path_count": len({branch.wraps for branch in pool}),
        "unique_selected_path_count": len({branch.wraps for branch in selected}),
        "mode_count": len(grouped),
        "selected_mode_count": len(
            {
                int((branch.diagnostics or {}).get("mode_index", -1))
                for branch in selected
            }
        ),
        "exact_fraction": exact_share,
        "mode_fraction": mode_share,
        "max_per_wrap_path": path_cap,
        "source_counts": source_counts,
    }


def _relative_session_offset_covariance(profile: Any) -> np.ndarray:
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    if variances.ndim != 1 or variances.size != int(profile.session_count):
        raise ValueError("Session offset variances must match the session count.")
    if variances.size <= 1:
        return np.zeros((0, 0), dtype=float)
    if not np.all(np.isfinite(variances) & (variances > 0.0)):
        raise ValueError("Session offset variances must be finite and positive.")
    covariance = np.diag(variances[1:]) + float(variances[0])
    jitter = max(float(np.trace(covariance)) * 1.0e-14, 1.0e-24)
    return covariance + np.eye(covariance.shape[0], dtype=float) * jitter


def score_session_integer_path(
    profile: Any,
    integer_offsets: Sequence[int],
) -> tuple[float, float, float]:
    """Score one session-integer path at the TOA uncertainty scale."""
    relative = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0],
        dtype=float,
    )
    integers = np.asarray(integer_offsets, dtype=float)
    if integers.shape != relative.shape:
        raise ValueError("Integer offsets must match the relative session dimension.")
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    shape_chi2 = float(shape @ shape)
    covariance = _relative_session_offset_covariance(profile)
    ambiguity = relative - integers
    ambiguity_chi2 = (
        float(ambiguity @ np.linalg.solve(covariance, ambiguity))
        if covariance.size
        else 0.0
    )
    return shape_chi2 + ambiguity_chi2, shape_chi2, ambiguity_chi2


def optimize_wrapped_session_posterior_modes(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    phase_temperatures: Sequence[float],
    max_modes: int,
    trust_radius: float,
    max_nfev: int,
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    prior_weight: float = 1.0,
    minimum_separation: float = 0.25,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Anneal a smooth wrapped likelihood before fixing integer paths.

    Squared ``sin(pi * phase)`` has minima only at integer phase offsets and is
    smooth across alias boundaries. Whiten the periodic term with the correlated
    session-offset covariance. Decreasing noise inflation factors allow basin
    relocation before reaching the TOA uncertainty scale at temperature one.
    """
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    center = (
        np.asarray(prior_center, dtype=float)
        if prior_center is not None
        else np.asarray(evaluator.base_parameter_values, dtype=float)
    )
    prior_sigma = (
        np.asarray(prior_scales, dtype=float)
        if prior_scales is not None
        else scales.copy()
    )
    if center.shape != scales.shape or prior_sigma.shape != scales.shape:
        raise ValueError("Prior and parameter scales must match the parameter dimension.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(prior_sigma) & (prior_sigma > 0.0)):
        raise ValueError("Prior scales must be finite and positive.")
    temperatures = tuple(float(value) for value in phase_temperatures)
    if not temperatures or not all(np.isfinite(value) and value > 0.0 for value in temperatures):
        raise ValueError("At least one finite positive phase temperature is required.")

    target = max(int(max_modes), 1)
    radius = max(float(trust_radius), 1.0e-12)
    weight = max(float(prior_weight), 0.0)
    separation = max(float(minimum_separation), 0.0)
    active_modes: list[np.ndarray] = []
    for mode in initial_modes:
        values = np.asarray(mode, dtype=float)
        if values.shape != scales.shape or not np.all(np.isfinite(values)):
            continue
        if any(
            np.linalg.norm((values - existing) / scales) < separation
            for existing in active_modes
        ):
            continue
        active_modes.append(values.copy())
    if not active_modes:
        active_modes.append(center.copy())

    stage_records: list[dict[str, Any]] = []
    for temperature in temperatures:
        ranked: list[tuple[float, float, float, float, int, np.ndarray, int, bool]] = []
        failed_count = 0
        for source_mode_index, start in enumerate(active_modes):
            try:
                initial_profile = evaluator.evaluate_values(start)
                shape_size = int(initial_profile.normalized_residuals.size)
                ambiguity_size = max(int(initial_profile.session_count) - 1, 0)
                prior_size = scales.size if weight > 0.0 else 0
                failure = np.full(
                    shape_size + ambiguity_size + prior_size,
                    1.0e12,
                    dtype=float,
                )

                def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
                    values = start + scales * np.asarray(scaled_offset, dtype=float)
                    try:
                        profile = evaluator.evaluate_values(values)
                        shape = np.asarray(profile.normalized_residuals, dtype=float)
                        relative = np.asarray(
                            profile.session_offsets[1:] - profile.session_offsets[0],
                            dtype=float,
                        )
                        if shape.shape != (shape_size,) or relative.shape != (ambiguity_size,):
                            return failure
                        covariance = _relative_session_offset_covariance(profile)
                        periodic = (
                            np.linalg.solve(
                                np.linalg.cholesky(covariance),
                                np.sin(np.pi * relative) / np.pi,
                            )
                            / float(temperature)
                            if covariance.size
                            else np.zeros(0, dtype=float)
                        )
                        result = [shape, periodic]
                        if weight > 0.0:
                            result.append(
                                np.sqrt(weight)
                                * (values - center)
                                / prior_sigma
                            )
                        vector = np.concatenate(result)
                        return vector if np.all(np.isfinite(vector)) else failure
                    except Exception:
                        return failure

                optimization = least_squares(
                    residual_vector,
                    np.zeros(scales.size, dtype=float),
                    method="trf",
                    jac="2-point",
                    bounds=(-radius, radius),
                    max_nfev=max(int(max_nfev), 1),
                    ftol=1.0e-9,
                    xtol=1.0e-9,
                    gtol=1.0e-9,
                )
                values = start + scales * np.asarray(optimization.x, dtype=float)
                profile = evaluator.evaluate_values(values)
                shape = np.asarray(profile.normalized_residuals, dtype=float)
                relative = np.asarray(
                    profile.session_offsets[1:] - profile.session_offsets[0],
                    dtype=float,
                )
                shape_chi2 = float(shape @ shape)
                covariance = _relative_session_offset_covariance(profile)
                periodic_residual = (
                    np.linalg.solve(
                        np.linalg.cholesky(covariance),
                        np.sin(np.pi * relative) / np.pi,
                    )
                    / float(temperature)
                    if covariance.size
                    else np.zeros(0, dtype=float)
                )
                periodic_chi2 = float(periodic_residual @ periodic_residual)
                prior_chi2 = float(
                    weight * np.sum(np.square((values - center) / prior_sigma))
                )
                ranked.append(
                    (
                        shape_chi2 + periodic_chi2 + prior_chi2,
                        shape_chi2,
                        periodic_chi2,
                        prior_chi2,
                        int(source_mode_index),
                        values.copy(),
                        int(optimization.nfev),
                        bool(optimization.success),
                    )
                )
            except Exception:
                failed_count += 1

        selected: list[tuple[float, float, float, float, int, np.ndarray, int, bool]] = []
        for candidate in sorted(ranked, key=lambda item: (item[0], item[1], item[4])):
            if any(
                np.linalg.norm((candidate[5] - existing[5]) / scales) < separation
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= target:
                break
        if not selected:
            raise RuntimeError("Wrapped session posterior annealing produced no valid mode.")
        active_modes = [candidate[5] for candidate in selected]
        stage_records.append(
            {
                "phase_temperature": temperature,
                "input_mode_count": len(ranked) + failed_count,
                "evaluated_mode_count": len(ranked),
                "failed_count": failed_count,
                "selected_mode_count": len(selected),
                "modes": [
                    {
                        "posterior_chi2": candidate[0],
                        "shape_chi2": candidate[1],
                        "periodic_chi2": candidate[2],
                        "prior_chi2": candidate[3],
                        "source_mode_index": candidate[4],
                        "parameter_values": candidate[5],
                        "nfev": candidate[6],
                        "success": candidate[7],
                    }
                    for candidate in selected
                ],
            }
        )

    return tuple(active_modes), {
        "proposal_source": "apollo-wrapped-session-posterior-annealing",
        "phase_temperatures": temperatures,
        "max_modes": target,
        "trust_radius": radius,
        "max_nfev": max(int(max_nfev), 1),
        "prior_weight": weight,
        "minimum_separation": separation,
        "stages": stage_records,
    }


def generate_wrapped_session_mode_branches(
    *,
    evaluator: Any,
    modes: Sequence[Sequence[float]],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    max_candidates: int | None = None,
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    prior_weight: float = 0.0,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Convert nonlinear wrapped-session modes directly into phase branches."""
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if base.shape != (len(names),):
        raise ValueError("Base parameter values must match parameter names.")
    center = base if prior_center is None else np.asarray(prior_center, dtype=float)
    scales = (
        np.ones_like(base)
        if prior_scales is None
        else np.asarray(prior_scales, dtype=float)
    )
    if center.shape != base.shape or scales.shape != base.shape:
        raise ValueError("Prior values must match the parameter dimension.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Prior scales must be finite and positive.")
    weight = max(float(prior_weight), 0.0)

    scored: list[tuple[float, int, ActiveTimingBranch]] = []
    failed_count = 0
    for mode_index, mode in enumerate(modes):
        values = np.asarray(mode, dtype=float)
        if values.shape != base.shape or not np.all(np.isfinite(values)):
            failed_count += 1
            continue
        try:
            profile = evaluator.evaluate_values(values)
            relative = np.asarray(
                profile.session_offsets[1:] - profile.session_offsets[0],
                dtype=float,
            )
            integer_offsets = tuple(
                int(value) for value in np.rint(relative).astype(np.int64)
            )
            objective_chi2, shape_chi2, ambiguity_chi2 = score_session_integer_path(
                profile,
                integer_offsets,
            )
            prior_chi2 = float(
                weight * np.sum(np.square((values - center) / scales))
            )
            posterior_chi2 = objective_chi2 + prior_chi2
            theta_offset = values - base
            branch = ActiveTimingBranch(
                branch_id=(
                    f"stage_{int(stage_index):02d}_wrapped_session_mode_"
                    f"{mode_index:04d}"
                ),
                structure_id=str(structure_id),
                binary_model=(
                    str(binary_model).upper()
                    if binary_model is not None
                    else None
                ),
                wraps=integer_session_offsets_to_gap_wraps(integer_offsets),
                theta_names=names,
                theta_offset=theta_offset,
                log_prior=-0.5 * posterior_chi2,
                proposal_source="apollo-covariance-wrapped-session-mode",
                diagnostics={
                    "proposal_family": "covariance-wrapped-session-mode",
                    "proposal_score": -0.5 * posterior_chi2,
                    "mode_index": mode_index,
                    "integer_session_offsets": integer_offsets,
                    "shape_chi2": shape_chi2,
                    "ambiguity_chi2": ambiguity_chi2,
                    "objective_chi2": objective_chi2,
                    "prior_chi2": prior_chi2,
                    "posterior_chi2": posterior_chi2,
                    "certification_theta_offset": theta_offset,
                    "wrap_residual_norm": math.sqrt(max(ambiguity_chi2, 0.0)),
                },
            )
            scored.append((posterior_chi2, mode_index, branch))
        except Exception:
            failed_count += 1

    limit = (
        len(scored)
        if max_candidates is None
        else max(int(max_candidates), 1)
    )
    selected: list[ActiveTimingBranch] = []
    seen_paths: set[tuple[int, ...]] = set()
    for _, _, branch in sorted(scored, key=lambda item: (item[0], item[1])):
        if branch.wraps in seen_paths:
            continue
        seen_paths.add(branch.wraps)
        selected.append(branch)
        if len(selected) >= limit:
            break

    return selected, {
        "proposal_source": "apollo-covariance-wrapped-session-mode",
        "input_mode_count": len(modes),
        "evaluated_mode_count": len(scored),
        "failed_count": failed_count,
        "unique_path_count": len(seen_paths),
        "kept_count": len(selected),
        "prior_weight": weight,
    }


def refine_session_integer_path_branches(
    *,
    evaluator: Any,
    branches: Sequence[ActiveTimingBranch],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    active_parameter_names: Sequence[str],
    parameter_scales: Sequence[float],
    trust_radius: float,
    max_nfev: int,
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    prior_weight: float = 0.0,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Refine proposed integer paths against the exact session likelihood."""
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    active_names = tuple(str(name).upper() for name in active_parameter_names)
    scales = np.asarray(parameter_scales, dtype=float)
    if base.shape != (len(names),) or scales.shape != base.shape:
        raise ValueError("Base values and scales must match parameter names.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    if not active_names or len(set(active_names)) != len(active_names):
        raise ValueError("Active parameter names must be non-empty and unique.")
    missing = [name for name in active_names if name not in names]
    if missing:
        raise ValueError(f"Unknown active parameters: {', '.join(missing)}")
    active_indices = np.asarray([names.index(name) for name in active_names], dtype=int)
    center = base if prior_center is None else np.asarray(prior_center, dtype=float)
    prior_sigma = scales if prior_scales is None else np.asarray(prior_scales, dtype=float)
    if center.shape != base.shape or prior_sigma.shape != base.shape:
        raise ValueError("Prior values must match the parameter dimension.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(prior_sigma) & (prior_sigma > 0.0)):
        raise ValueError("Prior scales must be finite and positive.")
    radius = max(float(trust_radius), 1.0e-12)
    weight = max(float(prior_weight), 0.0)

    refined: list[ActiveTimingBranch] = []
    records: list[dict[str, Any]] = []
    failed_count = 0
    for branch_index, branch in enumerate(branches):
        start = base + np.asarray(branch.theta_offset, dtype=float)
        if start.shape != base.shape or not np.all(np.isfinite(start)):
            failed_count += 1
            continue
        integer_offsets = tuple(
            int(value)
            for value in np.cumsum(np.asarray(branch.wraps, dtype=np.int64))
        )
        try:
            initial_profile = evaluator.evaluate_values(start)
            initial_objective, initial_shape, initial_ambiguity = (
                score_session_integer_path(initial_profile, integer_offsets)
            )
            initial_prior = float(
                weight * np.sum(np.square((start - center) / prior_sigma))
            )
            shape_size = int(initial_profile.normalized_residuals.size)
            ambiguity_size = len(integer_offsets)
            prior_size = active_indices.size if weight > 0.0 else 0
            failure = np.full(
                shape_size + ambiguity_size + prior_size,
                1.0e12,
                dtype=float,
            )

            def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
                values = start.copy()
                values[active_indices] += (
                    scales[active_indices] * np.asarray(scaled_offset, dtype=float)
                )
                try:
                    profile = evaluator.evaluate_values(values)
                    shape = np.asarray(profile.normalized_residuals, dtype=float)
                    relative = np.asarray(
                        profile.session_offsets[1:] - profile.session_offsets[0],
                        dtype=float,
                    )
                    if shape.shape != (shape_size,) or relative.shape != (ambiguity_size,):
                        return failure
                    covariance = _relative_session_offset_covariance(profile)
                    ambiguity = (
                        np.linalg.solve(
                            np.linalg.cholesky(covariance),
                            relative - np.asarray(integer_offsets, dtype=float),
                        )
                        if covariance.size
                        else np.zeros(0, dtype=float)
                    )
                    result = [shape, ambiguity]
                    if weight > 0.0:
                        result.append(
                            np.sqrt(weight)
                            * (values[active_indices] - center[active_indices])
                            / prior_sigma[active_indices]
                        )
                    vector = np.concatenate(result)
                    return vector if np.all(np.isfinite(vector)) else failure
                except Exception:
                    return failure

            optimization = least_squares(
                residual_vector,
                np.zeros(active_indices.size, dtype=float),
                method="trf",
                jac="2-point",
                bounds=(-radius, radius),
                max_nfev=max(int(max_nfev), 1),
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            candidate = start.copy()
            candidate[active_indices] += (
                scales[active_indices] * np.asarray(optimization.x, dtype=float)
            )
            final_profile = evaluator.evaluate_values(candidate)
            final_objective, final_shape, final_ambiguity = score_session_integer_path(
                final_profile,
                integer_offsets,
            )
            final_prior = float(
                weight * np.sum(np.square((candidate - center) / prior_sigma))
            )
            accepted = (
                np.isfinite(final_objective + final_prior)
                and final_objective + final_prior <= initial_objective + initial_prior
            )
            values = candidate if accepted else start
            objective = final_objective if accepted else initial_objective
            shape_chi2 = final_shape if accepted else initial_shape
            ambiguity_chi2 = final_ambiguity if accepted else initial_ambiguity
            prior_chi2 = final_prior if accepted else initial_prior
            posterior_chi2 = objective + prior_chi2
            theta_offset = values - base
            refined.append(
                ActiveTimingBranch(
                    branch_id=f"exact_path_refined_{branch_index:04d}",
                    structure_id=branch.structure_id,
                    binary_model=branch.binary_model,
                    wraps=branch.wraps,
                    theta_names=names,
                    theta_offset=theta_offset,
                    log_prior=-0.5 * posterior_chi2,
                    proposal_source="apollo-exact-session-path-refinement",
                    diagnostics={
                        **(branch.diagnostics or {}),
                        "proposal_family": "exact-session-path-refinement",
                        "source_branch_id": branch.branch_id,
                        "active_parameter_names": active_names,
                        "integer_session_offsets": integer_offsets,
                        "initial_objective_chi2": initial_objective,
                        "initial_shape_chi2": initial_shape,
                        "initial_ambiguity_chi2": initial_ambiguity,
                        "shape_chi2": shape_chi2,
                        "ambiguity_chi2": ambiguity_chi2,
                        "objective_chi2": objective,
                        "prior_chi2": prior_chi2,
                        "posterior_chi2": posterior_chi2,
                        "accepted": bool(accepted),
                        "success": bool(optimization.success),
                        "nfev": int(optimization.nfev),
                        "certification_theta_offset": theta_offset,
                        "wrap_residual_norm": math.sqrt(max(ambiguity_chi2, 0.0)),
                        "proposal_score": -0.5 * posterior_chi2,
                    },
                )
            )
            records.append(
                {
                    "source_branch_id": branch.branch_id,
                    "initial_objective_chi2": initial_objective,
                    "final_objective_chi2": objective,
                    "accepted": bool(accepted),
                    "success": bool(optimization.success),
                    "nfev": int(optimization.nfev),
                }
            )
        except Exception:
            failed_count += 1

    return refined, {
        "proposal_source": "apollo-exact-session-path-refinement",
        "input_branch_count": len(branches),
        "refined_branch_count": len(refined),
        "failed_count": failed_count,
        "active_parameter_names": active_names,
        "trust_radius": radius,
        "max_nfev": max(int(max_nfev), 1),
        "prior_weight": weight,
        "branches": records,
    }


def explore_joint_session_integer_modes(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    max_modes: int,
    trust_radius: float,
    max_nfev: int,
    max_integer_updates: int = 3,
    integer_candidates_per_start: int = 8,
    finite_difference_fraction: float = 1.0e-4,
    integer_max_visited_nodes: int = 1_000_000,
    integer_decimal_precision: int = 80,
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    prior_weight: float = 1.0,
    minimum_separation: float = 0.25,
) -> tuple[tuple[JointSessionIntegerMode, ...], dict[str, Any]]:
    """Explore nonlinear modes while retaining explicit integer session offsets.

    The shape-only optimizer profiles out every session mean. Here, each start
    fixes the nearest integer session path and optimizes a joint residual:
    intra-session shape plus the correlated distance to that path. Alternating
    integer updates with the continuous solve keeps distinct paths in separate
    posterior modes, preventing their merger into one shape-only basin.
    """
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    center = (
        np.asarray(prior_center, dtype=float)
        if prior_center is not None
        else np.asarray(evaluator.base_parameter_values, dtype=float)
    )
    prior_sigma = (
        np.asarray(prior_scales, dtype=float)
        if prior_scales is not None
        else scales.copy()
    )
    if center.shape != scales.shape or prior_sigma.shape != scales.shape:
        raise ValueError("Prior and parameter scales must match the parameter dimension.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(prior_sigma) & (prior_sigma > 0.0)):
        raise ValueError("Prior scales must be finite and positive.")

    radius = max(float(trust_radius), 1.0e-12)
    target = max(int(max_modes), 1)
    integer_update_limit = max(int(max_integer_updates), 1)
    weight = max(float(prior_weight), 0.0)
    separation = max(float(minimum_separation), 0.0)
    candidates: list[JointSessionIntegerMode] = []
    failed_count = 0

    seed_jobs: list[tuple[int, np.ndarray, tuple[int, ...]]] = []
    linearized_start_count = 0
    direct_start_count = 0
    for source_mode_index, initial_mode in enumerate(initial_modes):
        current = np.asarray(initial_mode, dtype=float)
        if current.shape != scales.shape or not np.all(np.isfinite(current)):
            failed_count += 1
            continue
        try:
            direct_profile = evaluator.evaluate_values(current)
            direct_relative = np.asarray(
                direct_profile.session_offsets[1:]
                - direct_profile.session_offsets[0],
                dtype=float,
            )
            seed_jobs.append(
                (
                    int(source_mode_index),
                    current.copy(),
                    tuple(
                        int(value)
                        for value in np.rint(direct_relative).astype(np.int64)
                    ),
                )
            )
            direct_start_count += 1
        except Exception:
            failed_count += 1
            continue
        try:
            linearization = evaluator.linearize_values(
                current,
                parameter_steps=np.maximum(
                    scales * max(float(finite_difference_fraction), 1.0e-12),
                    np.finfo(float).tiny,
                ),
            )
            integer_result = solve_integer_least_squares(
                linearization.marginalized_integer_center,
                linearization.marginalized_integer_covariance,
                candidate_count=max(int(integer_candidates_per_start), 1),
                max_visited_nodes=max(int(integer_max_visited_nodes), 1),
                decimal_precision=max(int(integer_decimal_precision), 32),
            )
            for integer_candidate in integer_result.candidates:
                conditional = linearization.condition_on_integer_offsets(
                    integer_candidate.integer_vector
                )
                seed_jobs.append(
                    (
                        int(source_mode_index),
                        np.asarray(conditional.parameter_values, dtype=float),
                        tuple(int(value) for value in integer_candidate.integer_vector),
                    )
                )
            linearized_start_count += len(integer_result.candidates)
        except Exception:
            pass

    for source_mode_index, initial_mode, initial_integer_offsets in seed_jobs:
        current = np.asarray(initial_mode, dtype=float)
        seen_integer_paths: set[tuple[int, ...]] = set()
        total_nfev = 0
        successful = False
        updates_done = 0
        try:
            evaluator.evaluate_values(current)
        except Exception:
            failed_count += 1
            continue

        for update_index in range(integer_update_limit):
            try:
                profile = evaluator.evaluate_values(current)
                relative = np.asarray(
                    profile.session_offsets[1:] - profile.session_offsets[0],
                    dtype=float,
                )
                if update_index == 0:
                    integer_offsets = tuple(initial_integer_offsets)
                else:
                    integer_offsets = tuple(
                        int(value) for value in np.rint(relative).astype(np.int64)
                    )
                if integer_offsets in seen_integer_paths:
                    break
                seen_integer_paths.add(integer_offsets)
                covariance = _relative_session_offset_covariance(profile)
                cholesky = (
                    np.linalg.cholesky(covariance)
                    if covariance.size
                    else np.zeros((0, 0), dtype=float)
                )
                shape_size = int(profile.normalized_residuals.size)
                ambiguity_size = len(integer_offsets)
                prior_size = scales.size if weight > 0.0 else 0
                failure = np.full(
                    shape_size + ambiguity_size + prior_size,
                    1.0e12,
                    dtype=float,
                )

                def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
                    values = current + scales * np.asarray(scaled_offset, dtype=float)
                    try:
                        candidate_profile = evaluator.evaluate_values(values)
                        shape = np.asarray(
                            candidate_profile.normalized_residuals,
                            dtype=float,
                        )
                        candidate_relative = np.asarray(
                            candidate_profile.session_offsets[1:]
                            - candidate_profile.session_offsets[0],
                            dtype=float,
                        )
                        if shape.shape != (shape_size,):
                            return failure
                        if ambiguity_size:
                            ambiguity = np.linalg.solve(
                                cholesky,
                                candidate_relative
                                - np.asarray(integer_offsets, dtype=float),
                            )
                        else:
                            ambiguity = np.zeros(0, dtype=float)
                        result = [shape, ambiguity]
                        if weight > 0.0:
                            result.append(
                                np.sqrt(weight)
                                * (values - center)
                                / prior_sigma
                            )
                        vector = np.concatenate(result)
                        return vector if np.all(np.isfinite(vector)) else failure
                    except Exception:
                        return failure

                optimization = least_squares(
                    residual_vector,
                    np.zeros(scales.size, dtype=float),
                    method="trf",
                    jac="2-point",
                    bounds=(-radius, radius),
                    max_nfev=max(int(max_nfev), 1),
                    ftol=1.0e-9,
                    xtol=1.0e-9,
                    gtol=1.0e-9,
                )
                total_nfev += int(optimization.nfev)
                successful = successful or bool(optimization.success)
                next_values = current + scales * np.asarray(
                    optimization.x,
                    dtype=float,
                )
                if not np.all(np.isfinite(next_values)):
                    break
                current = next_values
                updates_done = update_index + 1
                next_profile = evaluator.evaluate_values(current)
                next_relative = np.asarray(
                    next_profile.session_offsets[1:]
                    - next_profile.session_offsets[0],
                    dtype=float,
                )
                next_integer_offsets = tuple(
                    int(value) for value in np.rint(next_relative).astype(np.int64)
                )
                if next_integer_offsets == integer_offsets:
                    break
            except Exception:
                break

        try:
            final_profile = evaluator.evaluate_values(current)
            final_relative = np.asarray(
                final_profile.session_offsets[1:] - final_profile.session_offsets[0],
                dtype=float,
            )
            final_integer_offsets = tuple(
                int(value) for value in np.rint(final_relative).astype(np.int64)
            )
            final_covariance = _relative_session_offset_covariance(final_profile)
            if final_covariance.size:
                final_ambiguity = final_relative - np.asarray(
                    final_integer_offsets,
                    dtype=float,
                )
                final_ambiguity_chi2 = float(
                    final_ambiguity
                    @ np.linalg.solve(final_covariance, final_ambiguity)
                )
            else:
                final_ambiguity_chi2 = 0.0
            final_shape_chi2 = float(
                np.asarray(final_profile.normalized_residuals, dtype=float)
                @ np.asarray(final_profile.normalized_residuals, dtype=float)
            )
            final_objective = final_shape_chi2 + final_ambiguity_chi2
            final_prior_chi2 = float(
                weight
                * np.sum(np.square((current - center) / prior_sigma))
            )
            candidates.append(
                JointSessionIntegerMode(
                    integer_offsets=final_integer_offsets,
                    initial_integer_offsets=tuple(initial_integer_offsets),
                    parameter_values=current.copy(),
                    shape_chi2=final_shape_chi2,
                    ambiguity_chi2=final_ambiguity_chi2,
                    objective_chi2=final_objective,
                    prior_chi2=final_prior_chi2,
                    posterior_chi2=final_objective + final_prior_chi2,
                    source_mode_index=int(source_mode_index),
                    integer_updates=int(updates_done),
                    success=bool(successful),
                    nfev=int(total_nfev),
                )
            )
        except Exception:
            failed_count += 1

    selected: list[JointSessionIntegerMode] = []
    for candidate in sorted(
        candidates,
        key=lambda mode: (mode.posterior_chi2, mode.objective_chi2, mode.source_mode_index),
    ):
        if any(candidate.integer_offsets == existing.integer_offsets for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= target:
            break

    return tuple(selected), {
        "proposal_source": "apollo-joint-session-integer-mode",
        "initial_mode_count": len(initial_modes),
        "direct_start_count": direct_start_count,
        "linearized_start_count": linearized_start_count,
        "integer_candidates_per_start": max(int(integer_candidates_per_start), 1),
        "evaluated_mode_count": len(candidates),
        "failed_count": failed_count,
        "selected_mode_count": len(selected),
        "trust_radius": radius,
        "max_nfev": max(int(max_nfev), 1),
        "max_integer_updates": integer_update_limit,
        "prior_weight": weight,
        "minimum_separation": separation,
        "modes": [
            {
                "integer_offsets": mode.integer_offsets,
                "initial_integer_offsets": mode.initial_integer_offsets,
                "parameter_values": mode.parameter_values,
                "shape_chi2": mode.shape_chi2,
                "ambiguity_chi2": mode.ambiguity_chi2,
                "objective_chi2": mode.objective_chi2,
                "prior_chi2": mode.prior_chi2,
                "posterior_chi2": mode.posterior_chi2,
                "source_mode_index": mode.source_mode_index,
                "integer_updates": mode.integer_updates,
                "success": mode.success,
                "nfev": mode.nfev,
            }
            for mode in selected
        ],
    }


def extend_parent_session_integer_modes(
    *,
    evaluator: Any,
    parent_modes: Sequence[Sequence[float]],
    parent_integer_offsets: Sequence[Sequence[int]],
    parameter_scales: Sequence[float],
    max_modes: int,
    trust_radius: float,
    max_nfev: int,
    new_integer_candidates_per_session: int = 5,
    new_integer_radius: int = 2,
    max_path_candidates: int = 128,
    preoptimization_candidate_limit: int | None = None,
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    parameter_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    prior_weight: float = 1.0,
    minimum_separation: float = 0.25,
) -> tuple[tuple[JointSessionIntegerMode, ...], dict[str, Any]]:
    """Extend certified integer paths by Bayesian sequential prediction.

    ``parent_integer_offsets`` contains cumulative session integers from the
    previous prefix posterior. Keep these fixed and enumerate only newly
    observed session offsets near the continuous prediction. Optimize each
    complete candidate path against the exact current session-shape likelihood
    and its correlated integer-offset term.

    This path-first update evaluates
    ``p(path_new, theta | data_new, parent_path)`` without re-solving historical
    integers when a session is appended.
    """
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    center = (
        np.asarray(prior_center, dtype=float)
        if prior_center is not None
        else np.asarray(evaluator.base_parameter_values, dtype=float)
    )
    prior_sigma = (
        np.asarray(prior_scales, dtype=float)
        if prior_scales is not None
        else scales.copy()
    )
    if center.shape != scales.shape or prior_sigma.shape != scales.shape:
        raise ValueError("Prior and parameter scales must match the parameter dimension.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(prior_sigma) & (prior_sigma > 0.0)):
        raise ValueError("Prior scales must be finite and positive.")
    if parameter_bounds is None:
        bounds = tuple((None, None) for _ in range(scales.size))
    else:
        if len(parameter_bounds) != scales.size:
            raise ValueError("Parameter bounds must match the parameter dimension.")
        bounds = tuple(parameter_bounds)
        for lower, upper in bounds:
            if lower is not None and not np.isfinite(float(lower)):
                raise ValueError("Lower parameter bounds must be finite or None.")
            if upper is not None and not np.isfinite(float(upper)):
                raise ValueError("Upper parameter bounds must be finite or None.")
            if lower is not None and upper is not None and float(lower) >= float(upper):
                raise ValueError("Each lower parameter bound must be below its upper bound.")

    radius = max(float(trust_radius), 1.0e-12)
    target = max(int(max_modes), 1)
    per_session = max(int(new_integer_candidates_per_session), 1)
    integer_radius = max(int(new_integer_radius), 0)
    path_limit = max(int(max_path_candidates), 1)
    optimization_limit = (
        None
        if preoptimization_candidate_limit is None
        or int(preoptimization_candidate_limit) <= 0
        else max(int(preoptimization_candidate_limit), 1)
    )
    weight = max(float(prior_weight), 0.0)
    separation = max(float(minimum_separation), 0.0)
    if len(parent_modes) != len(parent_integer_offsets):
        raise ValueError("Parent modes and parent integer paths must be aligned.")
    if not parent_modes:
        raise ValueError("At least one parent mode is required for path extension.")

    candidates: list[JointSessionIntegerMode] = []
    seed_jobs: list[tuple[float, int, np.ndarray, tuple[int, ...]]] = []
    failed_count = 0
    path_candidate_count = 0
    extension_sizes: set[int] = set()

    def _candidate_deltas() -> tuple[int, ...]:
        values = tuple(range(-integer_radius, integer_radius + 1))
        return tuple(sorted(values, key=lambda value: (abs(value), value))[:per_session])

    for source_mode_index, (initial_mode, parent_path_values) in enumerate(
        zip(parent_modes, parent_integer_offsets, strict=True)
    ):
        initial = np.asarray(initial_mode, dtype=float)
        parent_path = tuple(int(value) for value in parent_path_values)
        if initial.shape != scales.shape or not np.all(np.isfinite(initial)):
            failed_count += 1
            continue
        if any(
            (lower is not None and initial[index] < float(lower))
            or (upper is not None and initial[index] >= float(upper))
            for index, (lower, upper) in enumerate(bounds)
        ):
            failed_count += 1
            continue
        try:
            initial_profile = evaluator.evaluate_values(initial)
            total_integer_count = max(int(initial_profile.session_count) - 1, 0)
            extension_count = total_integer_count - len(parent_path)
            if extension_count <= 0:
                raise ValueError(
                    "Parent path must be shorter than the current session path."
                )
            extension_sizes.add(extension_count)
            relative = np.asarray(
                initial_profile.session_offsets[1:]
                - initial_profile.session_offsets[0],
                dtype=float,
            )
            centers = np.rint(relative[len(parent_path) :]).astype(np.int64)
            delta_values = _candidate_deltas()
            extension_paths = [
                tuple(int(center_value + delta) for center_value, delta in zip(centers, deltas, strict=True))
                for deltas in product(delta_values, repeat=extension_count)
            ]
            extension_paths.sort(
                key=lambda values: sum(
                    (int(value) - int(center_value)) ** 2
                    for value, center_value in zip(values, centers, strict=True)
                )
            )
            extension_paths = extension_paths[:path_limit]
            path_candidate_count += len(extension_paths)
        except Exception:
            failed_count += 1
            continue

        covariance = _relative_session_offset_covariance(initial_profile)
        try:
            covariance_inverse = np.linalg.inv(covariance) if covariance.size else np.zeros((0, 0))
            initial_shape = np.asarray(initial_profile.normalized_residuals, dtype=float)
            for extension_path in extension_paths:
                integer_path = parent_path + extension_path
                ambiguity = relative - np.asarray(integer_path, dtype=float)
                ambiguity_chi2 = (
                    float(ambiguity @ covariance_inverse @ ambiguity)
                    if covariance.size
                    else 0.0
                )
                shape_chi2 = float(initial_shape @ initial_shape)
                prior_chi2 = float(
                    weight * np.sum(np.square((initial - center) / prior_sigma))
                )
                seed_jobs.append(
                    (
                        shape_chi2 + ambiguity_chi2 + prior_chi2,
                        int(source_mode_index),
                        initial.copy(),
                        integer_path,
                    )
                )
        except Exception:
            failed_count += len(extension_paths)

    seed_jobs.sort(key=lambda item: (item[0], item[1], item[3]))
    if optimization_limit is not None:
        seed_jobs = seed_jobs[:optimization_limit]

    for _, source_mode_index, initial, integer_path in seed_jobs:
        try:
            profile = evaluator.evaluate_values(initial)
            covariance = _relative_session_offset_covariance(profile)
            cholesky = (
                np.linalg.cholesky(covariance)
                if covariance.size
                else np.zeros((0, 0), dtype=float)
            )
            shape_size = int(profile.normalized_residuals.size)
            ambiguity_size = len(integer_path)
            prior_size = scales.size if weight > 0.0 else 0
            failure = np.full(
                shape_size + ambiguity_size + prior_size,
                1.0e12,
                dtype=float,
            )

            def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
                values = initial + scales * np.asarray(scaled_offset, dtype=float)
                try:
                    candidate_profile = evaluator.evaluate_values(values)
                    shape = np.asarray(candidate_profile.normalized_residuals, dtype=float)
                    candidate_relative = np.asarray(
                        candidate_profile.session_offsets[1:]
                        - candidate_profile.session_offsets[0],
                        dtype=float,
                    )
                    if shape.shape != (shape_size,) or candidate_relative.shape != (ambiguity_size,):
                        return failure
                    ambiguity = (
                        np.linalg.solve(
                            cholesky,
                            candidate_relative - np.asarray(integer_path, dtype=float),
                        )
                        if ambiguity_size
                        else np.zeros(0, dtype=float)
                    )
                    result = [shape, ambiguity]
                    if weight > 0.0:
                        result.append(np.sqrt(weight) * (values - center) / prior_sigma)
                    vector = np.concatenate(result)
                    return vector if np.all(np.isfinite(vector)) else failure
                except Exception:
                    return failure

            scaled_lower = np.full(scales.size, -radius, dtype=float)
            scaled_upper = np.full(scales.size, radius, dtype=float)
            for index, (lower, upper) in enumerate(bounds):
                if lower is not None:
                    scaled_lower[index] = max(
                        scaled_lower[index],
                        (float(lower) - initial[index]) / scales[index],
                    )
                if upper is not None:
                    scaled_upper[index] = min(
                        scaled_upper[index],
                        (float(upper) - initial[index]) / scales[index],
                    )
            if np.any(scaled_lower > scaled_upper):
                failed_count += 1
                continue

            optimization = least_squares(
                residual_vector,
                np.zeros(scales.size, dtype=float),
                method="trf",
                jac="2-point",
                bounds=(scaled_lower, scaled_upper),
                max_nfev=max(int(max_nfev), 1),
                ftol=1.0e-9,
                xtol=1.0e-9,
                gtol=1.0e-9,
            )
            values = initial + scales * np.asarray(optimization.x, dtype=float)
            if any(
                (lower is not None and values[index] < float(lower))
                or (upper is not None and values[index] >= float(upper))
                for index, (lower, upper) in enumerate(bounds)
            ):
                failed_count += 1
                continue
            final_profile = evaluator.evaluate_values(values)
            final_relative = np.asarray(
                final_profile.session_offsets[1:] - final_profile.session_offsets[0],
                dtype=float,
            )
            final_covariance = _relative_session_offset_covariance(final_profile)
            final_ambiguity = final_relative - np.asarray(integer_path, dtype=float)
            ambiguity_chi2 = (
                float(final_ambiguity @ np.linalg.solve(final_covariance, final_ambiguity))
                if final_covariance.size
                else 0.0
            )
            shape = np.asarray(final_profile.normalized_residuals, dtype=float)
            shape_chi2 = float(shape @ shape)
            prior_chi2 = float(weight * np.sum(np.square((values - center) / prior_sigma)))
            candidates.append(
                JointSessionIntegerMode(
                    integer_offsets=integer_path,
                    initial_integer_offsets=integer_path,
                    parameter_values=values.copy(),
                    shape_chi2=shape_chi2,
                    ambiguity_chi2=ambiguity_chi2,
                    objective_chi2=shape_chi2 + ambiguity_chi2,
                    prior_chi2=prior_chi2,
                    posterior_chi2=shape_chi2 + ambiguity_chi2 + prior_chi2,
                    source_mode_index=int(source_mode_index),
                    integer_updates=0,
                    success=bool(optimization.success),
                    nfev=int(optimization.nfev),
                )
            )
        except Exception:
            failed_count += 1

    selected: list[JointSessionIntegerMode] = []
    for candidate in sorted(
        candidates,
        key=lambda mode: (mode.posterior_chi2, mode.objective_chi2, mode.source_mode_index),
    ):
        if any(candidate.integer_offsets == existing.integer_offsets for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= target:
            break

    return tuple(selected), {
        "proposal_source": "apollo-parent-path-sequential-extension",
        "parent_mode_count": len(parent_modes),
        "path_candidate_count": path_candidate_count,
        "extension_sizes": tuple(sorted(extension_sizes)),
        "new_integer_candidates_per_session": per_session,
        "new_integer_radius": integer_radius,
        "max_path_candidates": path_limit,
        "preoptimization_candidate_limit": optimization_limit,
        "pre_scored_candidate_count": path_candidate_count,
        "optimization_candidate_count": len(seed_jobs),
        "evaluated_mode_count": len(candidates),
        "failed_count": failed_count,
        "selected_mode_count": len(selected),
        "trust_radius": radius,
        "max_nfev": max(int(max_nfev), 1),
        "prior_weight": weight,
        "minimum_separation": separation,
        "parameter_bounds": bounds,
        "modes": [
            {
                "integer_offsets": mode.integer_offsets,
                "parameter_values": mode.parameter_values,
                "shape_chi2": mode.shape_chi2,
                "ambiguity_chi2": mode.ambiguity_chi2,
                "objective_chi2": mode.objective_chi2,
                "prior_chi2": mode.prior_chi2,
                "posterior_chi2": mode.posterior_chi2,
                "source_mode_index": mode.source_mode_index,
                "nfev": mode.nfev,
                "success": mode.success,
            }
            for mode in selected
        ],
    }


def select_prior_sampled_session_shape_starts(
    *,
    evaluator: Any,
    prior_center: Sequence[float],
    parameter_scales: Sequence[float],
    anchor_modes: Sequence[Sequence[float]] = (),
    sample_count: int,
    max_starts: int,
    prior_temperature: float = 1.0,
    minimum_separation: float = 0.5,
    seed: int = 0,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    center = np.asarray(prior_center, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if center.ndim != 1 or scales.shape != center.shape:
        raise ValueError("Prior center and parameter scales must be one-dimensional and aligned.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")

    temperature = max(float(prior_temperature), 1.0e-12)
    target = max(int(max_starts), 1)
    requested_samples = max(int(sample_count), 0)
    separation = max(float(minimum_separation), 0.0)
    normalized_scales = scales * temperature

    candidates: list[tuple[str, int, np.ndarray]] = [("prior-center", 0, center.copy())]
    for anchor_index, anchor in enumerate(anchor_modes):
        values = np.asarray(anchor, dtype=float)
        if values.shape == center.shape and np.all(np.isfinite(values)):
            candidates.append(("anchor", anchor_index, values.copy()))

    if requested_samples:
        exponent = int(math.ceil(math.log2(requested_samples)))
        unit_samples = qmc.Sobol(
            d=center.size,
            scramble=True,
            seed=int(seed),
        ).random_base2(exponent)[:requested_samples]
        epsilon = np.finfo(float).eps
        normal_samples = norm.ppf(np.clip(unit_samples, epsilon, 1.0 - epsilon))
        for sample_index, normalized_offset in enumerate(normal_samples):
            candidates.append(
                (
                    "sobol-prior",
                    sample_index,
                    center + normalized_scales * normalized_offset,
                )
            )

    ranked: list[tuple[float, float, str, int, np.ndarray]] = []
    failed_count = 0
    for source, source_index, values in candidates:
        try:
            profile = evaluator.evaluate_values(values)
            residuals = np.asarray(profile.normalized_residuals, dtype=float)
        except Exception:
            failed_count += 1
            continue
        if residuals.ndim != 1 or not np.all(np.isfinite(residuals)):
            failed_count += 1
            continue
        shape_chi2 = float(residuals @ residuals)
        normalized_offset = (values - center) / normalized_scales
        prior_chi2 = float(normalized_offset @ normalized_offset)
        ranked.append(
            (
                shape_chi2 + prior_chi2,
                shape_chi2,
                source,
                source_index,
                values.copy(),
            )
        )

    selected: list[tuple[float, float, str, int, np.ndarray]] = []
    for candidate in sorted(ranked, key=lambda item: item[:4]):
        values = candidate[4]
        if any(
            np.linalg.norm((values - existing[4]) / normalized_scales) < separation
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= target:
            break
    if not selected:
        raise RuntimeError("Prior-sampled session-shape exploration produced no valid start.")

    return tuple(item[4] for item in selected), {
        "sample_count": requested_samples,
        "anchor_mode_count": len(anchor_modes),
        "evaluated_count": len(ranked),
        "failed_count": failed_count,
        "selected_count": len(selected),
        "prior_temperature": temperature,
        "minimum_separation": separation,
        "seed": int(seed),
        "selected": [
            {
                "posterior_deviance": item[0],
                "shape_chi2": item[1],
                "source": item[2],
                "source_index": item[3],
                "values": item[4],
            }
            for item in selected
        ],
    }


def select_prior_predictive_circular_phase_starts(
    *,
    evaluator: Any,
    prior_center: Sequence[float],
    parameter_scales: Sequence[float],
    anchor_modes: Sequence[Sequence[float]] = (),
    sample_count: int,
    max_starts: int,
    phase_sigma: float = 0.2,
    prior_temperature: float = 1.0,
    minimum_separation: float = 0.25,
    seed: int = 0,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Select diverse local modes from a prior-predictive circular phase scan.

    This produces proposals without assigning final evidence. Session offsets
    are scored modulo one cycle, leaving integer paths unfixed during the QMC
    scan.
    """

    center = np.asarray(prior_center, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if center.ndim != 1 or scales.shape != center.shape:
        raise ValueError("Prior center and parameter scales must align.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    sigma = max(float(phase_sigma), 1.0e-12)
    temperature = max(float(prior_temperature), 1.0e-12)
    normalized_scales = scales * temperature
    target = max(int(max_starts), 1)
    requested_samples = max(int(sample_count), 0)
    separation = max(float(minimum_separation), 0.0)

    candidates: list[tuple[str, int, np.ndarray]] = [
        ("prior-center", 0, center.copy())
    ]
    for anchor_index, anchor in enumerate(anchor_modes):
        values = np.asarray(anchor, dtype=float)
        if values.shape == center.shape and np.all(np.isfinite(values)):
            candidates.append(("anchor", anchor_index, values.copy()))
    if requested_samples:
        exponent = int(math.ceil(math.log2(requested_samples)))
        unit_samples = qmc.Sobol(
            d=center.size,
            scramble=True,
            seed=int(seed),
        ).random_base2(exponent)[:requested_samples]
        epsilon = np.finfo(float).eps
        normal_samples = norm.ppf(
            np.clip(unit_samples, epsilon, 1.0 - epsilon)
        )
        for sample_index, normalized_offset in enumerate(normal_samples):
            candidates.append(
                (
                    "sobol-prior",
                    sample_index,
                    center + normalized_scales * normalized_offset,
                )
            )

    ranked: list[
        tuple[
            float,
            float,
            float,
            float,
            tuple[int, ...],
            str,
            int,
            np.ndarray,
        ]
    ] = []
    failed_count = 0
    for source, source_index, values in candidates:
        try:
            profile = evaluator.evaluate_values(values)
            shape = np.asarray(profile.normalized_residuals, dtype=float)
            relative = np.asarray(
                profile.session_offsets[1:] - profile.session_offsets[0],
                dtype=float,
            )
        except Exception:
            failed_count += 1
            continue
        if not np.all(np.isfinite(shape)) or not np.all(np.isfinite(relative)):
            failed_count += 1
            continue
        integer_path = tuple(int(value) for value in np.rint(relative))
        fractional = relative - np.asarray(integer_path, dtype=float)
        shape_chi2 = float(shape @ shape)
        circular_chi2 = float(np.sum(np.square(fractional / sigma)))
        normalized_offset = (values - center) / normalized_scales
        prior_chi2 = float(normalized_offset @ normalized_offset)
        ranked.append(
            (
                shape_chi2 + circular_chi2 + prior_chi2,
                shape_chi2,
                circular_chi2,
                prior_chi2,
                integer_path,
                source,
                source_index,
                values.copy(),
            )
        )

    selected: list[
        tuple[
            float,
            float,
            float,
            float,
            tuple[int, ...],
            str,
            int,
            np.ndarray,
        ]
    ] = []
    represented_paths: set[tuple[int, ...]] = set()
    for candidate in sorted(ranked, key=lambda item: item[:7]):
        values = candidate[7]
        if candidate[4] in represented_paths:
            continue
        if any(
            np.linalg.norm((values - existing[7]) / normalized_scales)
            < separation
            for existing in selected
        ):
            continue
        selected.append(candidate)
        represented_paths.add(candidate[4])
        if len(selected) >= target:
            break
    if not selected:
        raise RuntimeError("Circular prior-predictive scan produced no valid start.")

    return tuple(item[7] for item in selected), {
        "proposal_source": "apollo-prior-predictive-circular-phase",
        "sample_count": requested_samples,
        "anchor_mode_count": len(anchor_modes),
        "evaluated_count": len(ranked),
        "failed_count": failed_count,
        "selected_count": len(selected),
        "unique_evaluated_path_count": len({item[4] for item in ranked}),
        "phase_sigma": sigma,
        "prior_temperature": temperature,
        "minimum_separation": separation,
        "seed": int(seed),
        "selected": [
            {
                "proposal_deviance": item[0],
                "shape_chi2": item[1],
                "circular_phase_chi2": item[2],
                "prior_chi2": item[3],
                "integer_path": item[4],
                "source": item[5],
                "source_index": item[6],
                "values": item[7],
            }
            for item in selected
        ],
    }


def refine_piecewise_circular_phase_modes(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    phase_sigmas: Sequence[float],
    max_modes: int,
    trust_radius: float,
    max_nfev: int,
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    prior_weight: float = 1.0,
    minimum_separation: float = 0.25,
    exploration_mode_indices: Sequence[int] = (),
    exploration_fraction: float = 0.0,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    """Refine circular proposals without smoothing across phase islands.

    Each start defines its nearest integer session path. Hold that path fixed
    while optimizing continuous timing parameters to keep distinct phase islands
    from collapsing into one shape-only mode. Decreasing phase sigmas continue
    broad circular proposals toward narrow phase-connected local modes. The
    downstream branch linker and PINT certification evaluate exact TOA-scale
    evidence.
    """

    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    center = (
        np.asarray(prior_center, dtype=float)
        if prior_center is not None
        else np.asarray(evaluator.base_parameter_values, dtype=float)
    )
    prior_sigma = (
        np.asarray(prior_scales, dtype=float)
        if prior_scales is not None
        else scales.copy()
    )
    if center.shape != scales.shape or prior_sigma.shape != scales.shape:
        raise ValueError("Prior and parameter scales must match the parameter dimension.")
    if not np.all(np.isfinite(center)):
        raise ValueError("Prior center must be finite.")
    if not np.all(np.isfinite(prior_sigma) & (prior_sigma > 0.0)):
        raise ValueError("Prior scales must be finite and positive.")
    sigmas = tuple(float(value) for value in phase_sigmas)
    if not sigmas or not all(np.isfinite(value) and value > 0.0 for value in sigmas):
        raise ValueError("At least one finite positive phase sigma is required.")
    if any(right >= left for left, right in zip(sigmas, sigmas[1:])):
        raise ValueError("Phase sigmas must be strictly decreasing.")

    target = max(int(max_modes), 1)
    radius = max(float(trust_radius), 1.0e-12)
    weight = max(float(prior_weight), 0.0)
    separation = max(float(minimum_separation), 0.0)
    exploration_sources = {int(value) for value in exploration_mode_indices}
    exploration_share = min(max(float(exploration_fraction), 0.0), 1.0)
    active: list[tuple[int, tuple[int, ...], np.ndarray]] = []
    failed_count = 0
    represented_paths: set[tuple[int, ...]] = set()
    for source_mode_index, mode in enumerate(initial_modes):
        values = np.asarray(mode, dtype=float)
        if values.shape != scales.shape or not np.all(np.isfinite(values)):
            failed_count += 1
            continue
        try:
            profile = evaluator.evaluate_values(values)
            relative = np.asarray(
                profile.session_offsets[1:] - profile.session_offsets[0],
                dtype=float,
            )
        except Exception:
            failed_count += 1
            continue
        if not np.all(np.isfinite(relative)):
            failed_count += 1
            continue
        integer_path = tuple(int(value) for value in np.rint(relative))
        if integer_path in represented_paths:
            continue
        represented_paths.add(integer_path)
        active.append((int(source_mode_index), integer_path, values.copy()))

    if not active:
        raise RuntimeError("Piecewise circular refinement received no valid phase mode.")

    stage_records: list[dict[str, Any]] = []
    for phase_sigma in sigmas:
        ranked: list[
            tuple[
                float,
                float,
                float,
                float,
                int,
                tuple[int, ...],
                np.ndarray,
                int,
                bool,
            ]
        ] = []
        stage_failed_count = 0
        for source_mode_index, integer_path, start in active:
            try:
                initial_profile = evaluator.evaluate_values(start)
                shape_size = int(initial_profile.normalized_residuals.size)
                ambiguity_size = len(integer_path)
                prior_size = scales.size if weight > 0.0 else 0
                failure = np.full(
                    shape_size + ambiguity_size + prior_size,
                    1.0e12,
                    dtype=float,
                )

                def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
                    values = start + scales * np.asarray(scaled_offset, dtype=float)
                    try:
                        profile = evaluator.evaluate_values(values)
                        shape = np.asarray(profile.normalized_residuals, dtype=float)
                        relative = np.asarray(
                            profile.session_offsets[1:] - profile.session_offsets[0],
                            dtype=float,
                        )
                        if shape.shape != (shape_size,) or relative.shape != (
                            ambiguity_size,
                        ):
                            return failure
                        phase = (
                            relative - np.asarray(integer_path, dtype=float)
                        ) / float(phase_sigma)
                        result = [shape, phase]
                        if weight > 0.0:
                            result.append(
                                np.sqrt(weight) * (values - center) / prior_sigma
                            )
                        vector = np.concatenate(result)
                        return vector if np.all(np.isfinite(vector)) else failure
                    except Exception:
                        return failure

                optimization = least_squares(
                    residual_vector,
                    np.zeros(scales.size, dtype=float),
                    method="trf",
                    jac="2-point",
                    bounds=(-radius, radius),
                    max_nfev=max(int(max_nfev), 1),
                    ftol=1.0e-9,
                    xtol=1.0e-9,
                    gtol=1.0e-9,
                )
                values = start + scales * np.asarray(optimization.x, dtype=float)
                final_profile = evaluator.evaluate_values(values)
                shape = np.asarray(final_profile.normalized_residuals, dtype=float)
                relative = np.asarray(
                    final_profile.session_offsets[1:]
                    - final_profile.session_offsets[0],
                    dtype=float,
                )
                phase = (
                    relative - np.asarray(integer_path, dtype=float)
                ) / float(phase_sigma)
                shape_chi2 = float(shape @ shape)
                phase_chi2 = float(phase @ phase)
                prior_chi2 = float(
                    weight * np.sum(np.square((values - center) / prior_sigma))
                )
                ranked.append(
                    (
                        shape_chi2 + phase_chi2 + prior_chi2,
                        shape_chi2,
                        phase_chi2,
                        prior_chi2,
                        source_mode_index,
                        integer_path,
                        values.copy(),
                        int(optimization.nfev),
                        bool(optimization.success),
                    )
                )
            except Exception:
                stage_failed_count += 1

        selected: list[
            tuple[
                float,
                float,
                float,
                float,
                int,
                tuple[int, ...],
                np.ndarray,
                int,
                bool,
            ]
        ] = []
        selected_paths: set[tuple[int, ...]] = set()
        sorted_candidates = sorted(
            ranked,
            key=lambda item: (item[0], item[1], item[4]),
        )

        def add_candidate(candidate: tuple[Any, ...]) -> bool:
            if candidate[5] in selected_paths:
                return False
            if any(
                np.linalg.norm((candidate[6] - existing[6]) / scales) < separation
                for existing in selected
            ):
                return False
            selected.append(candidate)
            selected_paths.add(candidate[5])
            return True

        exploration_target = min(
            int(round(target * exploration_share)),
            sum(candidate[4] in exploration_sources for candidate in sorted_candidates),
        )
        if exploration_target:
            for candidate in sorted_candidates:
                if candidate[4] not in exploration_sources:
                    continue
                add_candidate(candidate)
                if len(selected) >= exploration_target:
                    break
        for candidate in sorted_candidates:
            add_candidate(candidate)
            if len(selected) >= target:
                break
        if not selected:
            raise RuntimeError("Piecewise circular refinement produced no valid mode.")
        active = [
            (candidate[4], candidate[5], candidate[6].copy())
            for candidate in selected
        ]
        stage_records.append(
            {
                "phase_sigma": phase_sigma,
                "input_mode_count": len(ranked) + stage_failed_count,
                "evaluated_mode_count": len(ranked),
                "failed_count": stage_failed_count,
                "selected_mode_count": len(selected),
                "exploration_target": exploration_target,
                "selected_exploration_mode_count": sum(
                    candidate[4] in exploration_sources for candidate in selected
                ),
                "modes": [
                    {
                        "posterior_deviance": candidate[0],
                        "shape_chi2": candidate[1],
                        "phase_chi2": candidate[2],
                        "prior_chi2": candidate[3],
                        "source_mode_index": candidate[4],
                        "integer_path": candidate[5],
                        "parameter_values": candidate[6],
                        "nfev": candidate[7],
                        "success": candidate[8],
                    }
                    for candidate in selected
                ],
            }
        )

    return tuple(item[2] for item in active), {
        "proposal_source": "apollo-piecewise-circular-phase-refinement",
        "initial_mode_count": len(initial_modes),
        "valid_initial_mode_count": len(represented_paths),
        "failed_count": failed_count,
        "phase_sigmas": sigmas,
        "max_modes": target,
        "trust_radius": radius,
        "max_nfev": max(int(max_nfev), 1),
        "prior_weight": weight,
        "minimum_separation": separation,
        "exploration_mode_count": len(exploration_sources),
        "exploration_fraction": exploration_share,
        "stages": stage_records,
    }


def optimize_session_shape_modes(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    max_modes: int,
    trust_radius: float,
    max_nfev: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    radius = max(float(trust_radius), 1.0e-12)
    target = max(int(max_modes), 1)

    ranked_modes: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for mode_index, mode in enumerate(initial_modes):
        values = np.asarray(mode, dtype=float)
        if values.shape != scales.shape or not np.all(np.isfinite(values)):
            continue
        profile = evaluator.evaluate_values(values)
        residuals = np.asarray(profile.normalized_residuals, dtype=float)
        ranked_modes.append(
            (float(residuals @ residuals), mode_index, values.copy(), residuals)
        )

    selected_starts: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for candidate in sorted(ranked_modes, key=lambda item: (item[0], item[1])):
        values = candidate[2]
        if any(
            np.linalg.norm((values - selected[2]) / scales) <= 1.0e-6
            for selected in selected_starts
        ):
            continue
        selected_starts.append(candidate)
        if len(selected_starts) >= target:
            break

    optimized_modes: list[np.ndarray] = []
    mode_records: list[dict[str, Any]] = []
    for initial_chi2, mode_index, start, initial_residuals in selected_starts:
        def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
            try:
                profile = evaluator.evaluate_values(start + scales * scaled_offset)
                residuals = np.asarray(profile.normalized_residuals, dtype=float)
            except Exception:
                return np.full_like(initial_residuals, 1.0e12)
            if residuals.shape != initial_residuals.shape or not np.all(np.isfinite(residuals)):
                return np.full_like(initial_residuals, 1.0e12)
            return residuals

        optimization = least_squares(
            residual_vector,
            np.zeros(scales.size, dtype=float),
            method="trf",
            jac="3-point",
            bounds=(-radius, radius),
            max_nfev=max(int(max_nfev), 1),
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        candidate_values = start + scales * np.asarray(optimization.x, dtype=float)
        candidate_residuals = residual_vector(np.asarray(optimization.x, dtype=float))
        candidate_chi2 = float(candidate_residuals @ candidate_residuals)
        accepted = np.isfinite(candidate_chi2) and candidate_chi2 <= initial_chi2
        final_values = candidate_values if accepted else start
        final_chi2 = candidate_chi2 if accepted else initial_chi2
        optimized_modes.append(np.asarray(final_values, dtype=float))
        mode_records.append(
            {
                "source_mode_index": mode_index,
                "initial_shape_chi2": initial_chi2,
                "final_shape_chi2": final_chi2,
                "accepted": bool(accepted),
                "success": bool(optimization.success),
                "status": int(optimization.status),
                "message": str(optimization.message),
                "nfev": int(optimization.nfev),
                "scaled_offset": np.asarray(optimization.x, dtype=float),
                "trust_boundary_reached": bool(
                    np.any(np.abs(np.asarray(optimization.x, dtype=float)) >= radius * (1.0 - 1.0e-8))
                ),
            }
        )

    return tuple(optimized_modes), {
        "requested_mode_count": target,
        "input_mode_count": len(initial_modes),
        "optimized_mode_count": len(optimized_modes),
        "trust_radius": radius,
        "max_nfev": max(int(max_nfev), 1),
        "modes": mode_records,
    }


def _certification_quality_key(
    certification: BranchCertification,
) -> tuple[float, float, str]:
    chi2 = (
        float(certification.reduced_chi2)
        if certification.reduced_chi2 is not None
        else math.inf
    )
    wrms = (
        float(certification.weighted_rms_us)
        if certification.weighted_rms_us is not None
        else math.inf
    )
    return chi2, wrms, certification.branch_id


def certified_parameter_modes(
    *,
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    certifications: Sequence[BranchCertification],
    max_modes: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if base.shape != (len(names),):
        raise ValueError("Base parameter values must match parameter names.")
    target = max(int(max_modes), 1)
    certified_limit = max(target - 1, 0)
    modes: list[np.ndarray] = []
    source_branch_ids: list[str] = []
    for certification in sorted(certifications, key=_certification_quality_key):
        if len(modes) >= certified_limit:
            break
        if any(name not in certification.fitted_theta for name in names):
            continue
        values = np.asarray(
            [certification.fitted_theta[name] for name in names],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            continue
        if any(np.allclose(values, mode, rtol=1.0e-12, atol=0.0) for mode in modes):
            continue
        modes.append(values)
        source_branch_ids.append(certification.branch_id)

    seed_included = not any(
        np.allclose(base, mode, rtol=1.0e-12, atol=0.0)
        for mode in modes
    )
    if seed_included and len(modes) < target:
        modes.append(base.copy())
        source_branch_ids.append("seed")
    if not modes:
        modes.append(base.copy())
        source_branch_ids.append("seed")
        seed_included = True

    return tuple(modes[:target]), {
        "mode_count": min(len(modes), target),
        "certified_mode_count": sum(
            branch_id != "seed" for branch_id in source_branch_ids[:target]
        ),
        "seed_mode_included": "seed" in source_branch_ids[:target],
        "source_branch_ids": source_branch_ids[:target],
    }


def generate_session_lattice_branches_from_linearizations(
    *,
    linearizations: Sequence[SessionShapeLinearization],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    active_config = config or MultimodeSessionLatticeConfig()
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if base.shape != (len(names),):
        raise ValueError("Base parameter values must match parameter names.")

    scored: list[tuple[float, float, int, int, ActiveTimingBranch]] = []
    mode_records: list[dict[str, Any]] = []
    for mode_index, linearization in enumerate(linearizations):
        integer_result = solve_integer_least_squares(
            linearization.marginalized_integer_center,
            linearization.marginalized_integer_covariance,
            candidate_count=max(int(active_config.candidates_per_mode), 1),
            max_visited_nodes=max(int(active_config.max_visited_nodes), 1),
            decimal_precision=max(int(active_config.decimal_precision), 32),
        )
        mode_records.append(
            {
                "mode_index": mode_index,
                "candidate_count": len(integer_result.candidates),
                "decorrelation_iterations": integer_result.decorrelation_iterations,
                "visited_nodes": integer_result.visited_nodes,
                "shape_chi2": linearization.profile.chi2,
            }
        )
        for candidate_index, integer_candidate in enumerate(integer_result.candidates):
            conditional = linearization.condition_on_integer_offsets(
                integer_candidate.integer_vector
            )
            theta_offset = np.asarray(
                conditional.parameter_values - base,
                dtype=float,
            )
            wraps = integer_session_offsets_to_gap_wraps(
                integer_candidate.integer_vector
            )
            proposal_cost = float(
                conditional.linearized_chi2
                + integer_candidate.squared_mahalanobis
            )
            branch = ActiveTimingBranch(
                branch_id=(
                    f"stage_{int(stage_index):02d}_session_lattice_"
                    f"m{mode_index:03d}_i{candidate_index:04d}"
                ),
                structure_id=str(structure_id),
                binary_model=(
                    str(binary_model).upper()
                    if binary_model is not None
                    else None
                ),
                wraps=wraps,
                theta_names=names,
                theta_offset=theta_offset,
                log_prior=-0.5 * proposal_cost,
                proposal_source="apollo-marginalized-session-lattice",
                diagnostics={
                    "proposal_family": "marginalized-session-lattice",
                    "proposal_score": -0.5 * proposal_cost,
                    "mode_index": mode_index,
                    "integer_candidate_index": candidate_index,
                    "integer_session_offsets": integer_candidate.integer_vector,
                    "integer_squared_mahalanobis": (
                        integer_candidate.squared_mahalanobis
                    ),
                    "linearized_chi2": conditional.linearized_chi2,
                    "certification_theta_offset": theta_offset,
                    "wrap_residual_norm": math.sqrt(
                        max(integer_candidate.squared_mahalanobis, 0.0)
                    ),
                },
            )
            scored.append(
                (
                    proposal_cost,
                    float(integer_candidate.squared_mahalanobis),
                    mode_index,
                    candidate_index,
                    branch,
                )
            )

    ranked = sorted(scored, key=lambda item: item[:4])
    selected: list[ActiveTimingBranch] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    seen_paths: set[tuple[int, ...]] = set()
    candidate_limit = max(int(active_config.max_candidates), 1)

    # Cover distinct integer paths before spending budget on alternate
    # continuous starts for paths already represented by another shape mode.
    for _, _, _, _, branch in ranked:
        if branch.wraps in seen_paths:
            continue
        key = (
            branch.wraps,
            tuple(float(value) for value in np.round(branch.theta_offset, 15)),
        )
        if key in seen:
            continue
        seen.add(key)
        seen_paths.add(branch.wraps)
        selected.append(branch)
        if len(selected) >= candidate_limit:
            break

    if len(selected) < candidate_limit:
        for _, _, _, _, branch in ranked:
            key = (
                branch.wraps,
                tuple(float(value) for value in np.round(branch.theta_offset, 15)),
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(branch)
            if len(selected) >= candidate_limit:
                break

    return selected, {
        "proposal_source": "apollo-marginalized-session-lattice",
        "mode_count": len(linearizations),
        "candidate_count": len(scored),
        "unique_candidate_path_count": len(
            {branch.wraps for _, _, _, _, branch in scored}
        ),
        "kept_count": len(selected),
        "unique_kept_path_count": len({branch.wraps for branch in selected}),
        "modes": mode_records,
    }


def generate_wrap_invariant_global_phase_posterior_branches(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    mode_count: int = 8,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Infer every session integer from a bounded shape posterior.

    Session-internal residual shape is invariant to integer phase wraps and
    contains slope and curvature evidence. First condition the continuous timing
    parameters on that shape under an explicit broad-scale Gaussian prior. Then
    jointly predict all relative session phases and solve their integer
    least-squares posterior. Discard all integer edges and component topology
    from the active linker.
    """

    scales = np.asarray(parameter_scales, dtype=float)
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if scales.ndim != 1 or base.shape != scales.shape or len(names) != scales.size:
        raise ValueError("Global phase posterior parameter dimensions must align.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Global phase posterior scales must be finite and positive.")
    if not np.all(np.isfinite(base)):
        raise ValueError("Global phase posterior base values must be finite.")

    active_config = config or MultimodeSessionLatticeConfig()
    prior_scale = float(active_config.global_phase_prior_scale)
    if not np.isfinite(prior_scale) or prior_scale <= 0.0:
        raise ValueError("Global phase prior scale must be finite and positive.")
    target_modes = max(int(mode_count), 1)
    step_fraction = max(float(active_config.finite_difference_fraction), 1.0e-12)
    parameter_steps = np.maximum(scales * step_fraction, np.finfo(float).tiny)

    selected_records: list[tuple[int, np.ndarray, float]] = []
    for source_mode_index, values_like in enumerate(initial_modes):
        values = np.asarray(
            getattr(values_like, "parameter_values", values_like),
            dtype=float,
        )
        if values.shape != base.shape or not np.all(np.isfinite(values)):
            continue
        if any(
            np.linalg.norm((values - existing[1]) / scales) <= 1.0e-3
            for existing in selected_records
        ):
            continue
        log_evidence = float(getattr(values_like, "laplace_log_evidence", 0.0))
        if not np.isfinite(log_evidence):
            log_evidence = 0.0
        selected_records.append((source_mode_index, values.copy(), log_evidence))
        if len(selected_records) >= target_modes:
            break

    linearizations: list[
        tuple[int, np.ndarray, float, SessionShapeLinearization]
    ] = []
    failed_modes: list[dict[str, Any]] = []
    for source_mode_index, values, log_evidence in selected_records:
        try:
            linearization = evaluator.linearize_values(
                values,
                parameter_steps=parameter_steps,
            )
        except Exception as error:
            failed_modes.append(
                {
                    "source_mode_index": source_mode_index,
                    "stage": "linearization",
                    "error": str(error),
                }
            )
            continue
        linearizations.append(
            (source_mode_index, values, log_evidence, linearization)
        )
    if not linearizations:
        raise RuntimeError(
            "Wrap-invariant global phase posterior produced no valid linearization."
        )

    prior_precision = np.eye(scales.size, dtype=float) / (prior_scale * prior_scale)
    scored: list[tuple[float, float, int, int, ActiveTimingBranch]] = []
    mode_diagnostics: list[dict[str, Any]] = []
    for mode_index, (
        source_mode_index,
        values,
        log_evidence,
        linearization,
    ) in enumerate(linearizations):
        residuals = np.asarray(
            linearization.profile.normalized_residuals,
            dtype=float,
        )
        shape_jacobian = (
            np.asarray(linearization.shape_jacobian, dtype=float) / step_fraction
        )
        relative_offsets = np.asarray(
            linearization.relative_session_offsets,
            dtype=float,
        )
        relative_jacobian = (
            np.asarray(linearization.relative_session_jacobian, dtype=float)
            / step_fraction
        )
        relative_noise = np.asarray(
            linearization.relative_offset_covariance,
            dtype=float,
        )
        expected_shape = (residuals.size, scales.size)
        expected_relative = (relative_offsets.size, scales.size)
        if (
            shape_jacobian.shape != expected_shape
            or relative_jacobian.shape != expected_relative
            or relative_noise.shape
            != (relative_offsets.size, relative_offsets.size)
            or relative_offsets.size < 1
            or not np.all(np.isfinite(residuals))
            or not np.all(np.isfinite(shape_jacobian))
            or not np.all(np.isfinite(relative_offsets))
            or not np.all(np.isfinite(relative_jacobian))
            or not np.all(np.isfinite(relative_noise))
        ):
            failed_modes.append(
                {
                    "source_mode_index": source_mode_index,
                    "stage": "posterior-geometry",
                    "error": "invalid linearized geometry",
                }
            )
            continue

        posterior_precision = (
            shape_jacobian.T @ shape_jacobian + prior_precision
        )
        posterior_precision = 0.5 * (
            posterior_precision + posterior_precision.T
        )
        precision_jitter = max(
            float(np.trace(posterior_precision)) * 1.0e-14,
            1.0e-14,
        )
        posterior_precision += np.eye(scales.size) * precision_jitter
        try:
            posterior_covariance = np.linalg.solve(
                posterior_precision,
                np.eye(scales.size, dtype=float),
            )
            shape_posterior_mean = np.linalg.solve(
                posterior_precision,
                -(shape_jacobian.T @ residuals),
            )
        except np.linalg.LinAlgError as error:
            failed_modes.append(
                {
                    "source_mode_index": source_mode_index,
                    "stage": "shape-posterior",
                    "error": str(error),
                }
            )
            continue

        # Active-linker Laplace evidence includes a mode's current partial
        # integer topology, while exploratory shape starts have no such
        # quantity.  Re-score both with the same wrap-invariant Gaussian
        # marginal likelihood before ranking their integer paths.
        shape_gradient = shape_jacobian.T @ residuals
        shape_chi2_after_linear_fit = max(
            float(residuals @ residuals)
            - float(shape_gradient @ posterior_covariance @ shape_gradient),
            0.0,
        )
        mode_scaled_offset = (values - base) / scales
        mode_prior_chi2 = float(
            mode_scaled_offset @ prior_precision @ mode_scaled_offset
        )
        prior_sign, prior_logdet = np.linalg.slogdet(prior_precision)
        posterior_sign, posterior_logdet = np.linalg.slogdet(
            posterior_precision
        )
        if prior_sign <= 0 or posterior_sign <= 0:
            failed_modes.append(
                {
                    "source_mode_index": source_mode_index,
                    "stage": "shape-evidence",
                    "error": "non-positive posterior determinant",
                }
            )
            continue
        log_evidence = float(
            -0.5 * shape_chi2_after_linear_fit
            + 0.5 * (prior_logdet - posterior_logdet)
            - 0.5 * mode_prior_chi2
        )

        predictive_center = (
            relative_offsets + relative_jacobian @ shape_posterior_mean
        )
        predictive_covariance = (
            relative_noise
            + relative_jacobian
            @ posterior_covariance
            @ relative_jacobian.T
        )
        predictive_covariance = 0.5 * (
            predictive_covariance + predictive_covariance.T
        )
        predictive_jitter = max(
            float(np.trace(predictive_covariance)) * 1.0e-12,
            1.0e-12,
        )
        predictive_covariance += (
            np.eye(relative_offsets.size, dtype=float) * predictive_jitter
        )
        try:
            integer_result = solve_integer_least_squares(
                predictive_center,
                predictive_covariance,
                candidate_count=max(int(active_config.candidates_per_mode), 1),
                max_visited_nodes=max(int(active_config.max_visited_nodes), 1),
                decimal_precision=max(int(active_config.decimal_precision), 32),
            )
        except Exception as error:
            failed_modes.append(
                {
                    "source_mode_index": source_mode_index,
                    "stage": "integer-least-squares",
                    "error": str(error),
                }
            )
            continue

        mode_candidate_count = 0
        for candidate_index, integer_candidate in enumerate(
            integer_result.candidates
        ):
            integers = np.asarray(integer_candidate.integer_vector, dtype=float)
            innovation = integers - predictive_center
            conditional_scaled_offset = (
                shape_posterior_mean
                + posterior_covariance
                @ relative_jacobian.T
                @ np.linalg.solve(predictive_covariance, innovation)
            )
            candidate_values = values + scales * conditional_scaled_offset
            shape_residuals = residuals + shape_jacobian @ conditional_scaled_offset
            phase_residuals = (
                relative_offsets
                + relative_jacobian @ conditional_scaled_offset
                - integers
            )
            conditional_shape_chi2 = float(shape_residuals @ shape_residuals)
            conditional_prior_chi2 = float(
                conditional_scaled_offset
                @ prior_precision
                @ conditional_scaled_offset
            )
            conditional_phase_chi2 = float(
                phase_residuals
                @ np.linalg.solve(relative_noise, phase_residuals)
            )
            proposal_cost = float(
                integer_candidate.squared_mahalanobis - 2.0 * log_evidence
            )
            full_integer_path = (0,) + tuple(
                int(value) for value in integer_candidate.integer_vector
            )
            branch = ActiveTimingBranch(
                branch_id=(
                    f"wrap_invariant_global_phase_m{mode_index:03d}_"
                    f"i{candidate_index:04d}"
                ),
                structure_id=str(structure_id),
                binary_model=(
                    str(binary_model).upper() if binary_model is not None else None
                ),
                wraps=integer_session_offsets_to_gap_wraps(
                    integer_candidate.integer_vector
                ),
                theta_names=names,
                theta_offset=np.asarray(candidate_values - base, dtype=float),
                log_prior=float(
                    log_evidence
                    - 0.5 * integer_candidate.squared_mahalanobis
                ),
                proposal_source=(
                    "apollo-bayesian-wrap-invariant-global-phase-posterior"
                ),
                diagnostics={
                    "proposal_family": (
                        "wrap-invariant-shape-conditioned-joint-phase-posterior"
                    ),
                    "mode_index": mode_index,
                    "source_mode_index": source_mode_index,
                    "integer_candidate_index": candidate_index,
                    "integer_session_offsets": full_integer_path,
                    "integer_squared_mahalanobis": (
                        integer_candidate.squared_mahalanobis
                    ),
                    "posterior_mode_log_evidence": log_evidence,
                    "shape_marginal_log_evidence": log_evidence,
                    "shape_chi2_after_linear_fit": shape_chi2_after_linear_fit,
                    "mode_prior_chi2": mode_prior_chi2,
                    "global_phase_prior_scale": prior_scale,
                    "shape_posterior_scaled_mean": shape_posterior_mean,
                    "shape_posterior_scaled_covariance": posterior_covariance,
                    "posterior_predictive_center": predictive_center,
                    "posterior_predictive_covariance": predictive_covariance,
                    "conditional_scaled_parameter_offset": (
                        conditional_scaled_offset
                    ),
                    "conditional_shape_chi2": conditional_shape_chi2,
                    "conditional_phase_chi2": conditional_phase_chi2,
                    "conditional_prior_chi2": conditional_prior_chi2,
                    "certification_theta_offset": np.asarray(
                        candidate_values - base,
                        dtype=float,
                    ),
                },
            )
            scored.append(
                (
                    proposal_cost,
                    float(integer_candidate.squared_mahalanobis),
                    mode_index,
                    candidate_index,
                    branch,
                )
            )
            mode_candidate_count += 1

        mode_diagnostics.append(
            {
                "mode_index": mode_index,
                "source_mode_index": source_mode_index,
                "candidate_count": mode_candidate_count,
                "visited_nodes": integer_result.visited_nodes,
                "posterior_mode_log_evidence": log_evidence,
                "shape_posterior_scaled_norm": float(
                    np.linalg.norm(shape_posterior_mean)
                ),
                "predictive_center_min": float(np.min(predictive_center)),
                "predictive_center_max": float(np.max(predictive_center)),
            }
        )

    relinearization_records: list[dict[str, Any]] = []
    relinearization_rounds = max(
        int(active_config.global_phase_relinearization_rounds),
        0,
    )
    relinearization_candidates = max(
        int(active_config.global_phase_relinearization_candidates_per_mode),
        1,
    )
    for round_index in range(relinearization_rounds):
        seeds: list[tuple[int, float, ActiveTimingBranch]] = []
        for mode_index in range(len(linearizations)):
            mode_scored = sorted(
                (
                    item
                    for item in scored
                    if int(item[2]) == mode_index
                ),
                key=lambda item: item[:4],
            )
            seeds.extend(
                (mode_index, float(item[0]), item[4])
                for item in mode_scored[:relinearization_candidates]
            )
        seed_keys: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
        relinearized_scored: list[
            tuple[float, float, int, int, ActiveTimingBranch]
        ] = []
        round_failures = 0
        for seed_index, (mode_index, _, seed_branch) in enumerate(seeds):
            seed_key = (
                seed_branch.wraps,
                tuple(
                    float(value)
                    for value in np.round(seed_branch.theta_offset, 14)
                ),
            )
            if seed_key in seed_keys:
                continue
            seed_keys.add(seed_key)
            seed_values = base + np.asarray(seed_branch.theta_offset, dtype=float)
            try:
                generated, _ = (
                    generate_wrap_invariant_global_phase_posterior_branches(
                        evaluator=evaluator,
                        initial_modes=(seed_values,),
                        parameter_scales=scales,
                        base_parameter_values=base,
                        parameter_names=names,
                        structure_id=structure_id,
                        binary_model=binary_model,
                        stage_index=stage_index,
                        mode_count=1,
                        config=replace(
                            active_config,
                            global_phase_relinearization_rounds=0,
                            max_candidates=max(
                                int(active_config.candidates_per_mode),
                                1,
                            ),
                        ),
                    )
                )
            except Exception:
                round_failures += 1
                continue
            for candidate_index, branch in enumerate(generated):
                relinearized_branch = replace(
                    branch,
                    branch_id=(
                        f"wrap_invariant_relinearized_r{round_index + 1:02d}_"
                        f"s{seed_index:03d}_i{candidate_index:03d}"
                    ),
                    diagnostics={
                        **(branch.diagnostics or {}),
                        "relinearization_round": round_index + 1,
                        "relinearization_mode_index": mode_index,
                        "relinearization_seed_branch_id": seed_branch.branch_id,
                    },
                )
                relinearized_scored.append(
                    (
                        float(-2.0 * relinearized_branch.log_prior),
                        float(
                            relinearized_branch.diagnostics.get(
                                "integer_squared_mahalanobis",
                                math.inf,
                            )
                        ),
                        mode_index,
                        candidate_index,
                        relinearized_branch,
                    )
                )
        if not relinearized_scored:
            relinearization_records.append(
                {
                    "round": round_index + 1,
                    "seed_count": len(seeds),
                    "generated_count": 0,
                    "failed_count": round_failures,
                }
            )
            break
        scored.extend(relinearized_scored)
        relinearization_records.append(
            {
                "round": round_index + 1,
                "seed_count": len(seeds),
                "generated_count": len(relinearized_scored),
                "failed_count": round_failures,
                "unique_generated_path_count": len(
                    {item[4].wraps for item in relinearized_scored}
                ),
            }
        )

    if not scored:
        raise RuntimeError(
            "Wrap-invariant global phase posterior produced no integer candidates."
        )
    ranked = sorted(scored, key=lambda item: item[:4])
    selected: list[ActiveTimingBranch] = []
    seen_paths: set[tuple[int, ...]] = set()
    seen_branches: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    candidate_limit = max(int(active_config.max_candidates), 1)
    for _, _, _, _, branch in ranked:
        if branch.wraps in seen_paths:
            continue
        key = (
            branch.wraps,
            tuple(float(value) for value in np.round(branch.theta_offset, 15)),
        )
        if key in seen_branches:
            continue
        seen_branches.add(key)
        seen_paths.add(branch.wraps)
        selected.append(branch)
        if len(selected) >= candidate_limit:
            break
    if len(selected) < candidate_limit:
        for _, _, _, _, branch in ranked:
            key = (
                branch.wraps,
                tuple(float(value) for value in np.round(branch.theta_offset, 15)),
            )
            if key in seen_branches:
                continue
            seen_branches.add(key)
            selected.append(branch)
            if len(selected) >= candidate_limit:
                break

    return selected, {
        "proposal_source": (
            "apollo-bayesian-wrap-invariant-global-phase-posterior"
        ),
        "proposal_family": (
            "wrap-invariant-shape-conditioned-joint-phase-posterior"
        ),
        "mode_count_requested": target_modes,
        "mode_count_selected": len(selected_records),
        "mode_count_linearized": len(linearizations),
        "failed_modes": failed_modes,
        "parameter_steps": parameter_steps,
        "global_phase_prior_scale": prior_scale,
        "global_phase_relinearization_rounds": relinearization_rounds,
        "global_phase_relinearization_candidates_per_mode": (
            relinearization_candidates
        ),
        "relinearization": relinearization_records,
        "candidate_count": len(scored),
        "unique_candidate_path_count": len({item[4].wraps for item in scored}),
        "kept_count": len(selected),
        "unique_kept_path_count": len({branch.wraps for branch in selected}),
        "ignored_active_component_topology": True,
        "modes": mode_diagnostics,
    }


def generate_posterior_particle_global_phase_branches(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Build global phase paths with an annealed posterior particle system.

    The preceding Gaussian generator uses a local Laplace approximation. This
    generator samples a wrapped phase posterior in continuous DD coordinates,
    recording the integer path induced by each particle. Its target combines the
    nearest wrapped phase likelihood with an optional within-session slope or
    full shape likelihood. Annealing and mutation retain separated posterior
    modes long enough to aggregate path mass.
    """
    active_config = config or MultimodeSessionLatticeConfig()
    scales = np.asarray(parameter_scales, dtype=float)
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if scales.ndim != 1 or base.shape != scales.shape or len(names) != scales.size:
        raise ValueError("Posterior particle parameter dimensions must align.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Posterior particle parameter scales must be positive.")
    if not np.all(np.isfinite(base)):
        raise ValueError("Posterior particle base values must be finite.")

    prior_scale = float(active_config.global_phase_prior_scale)
    if not np.isfinite(prior_scale) or prior_scale <= 0.0:
        raise ValueError("Global phase prior scale must be finite and positive.")
    particle_count = max(int(active_config.posterior_particle_count), 32)
    anchor_fraction = min(
        max(float(active_config.posterior_particle_anchor_fraction), 0.0),
        0.9,
    )
    anchor_scale = max(float(active_config.posterior_particle_anchor_scale), 1.0e-4)
    evidence_mode = str(active_config.posterior_particle_evidence_mode).lower()
    allowed_modes = {"phase-only", "phase+slope", "shape", "phase+shape"}
    if evidence_mode not in allowed_modes:
        raise ValueError(
            "Posterior particle evidence mode must be phase-only, phase+slope, "
            "shape, or phase+shape."
        )

    rng = np.random.default_rng(int(active_config.posterior_particle_seed))
    selected_anchors: list[np.ndarray] = []
    for values_like in initial_modes:
        values = np.asarray(
            getattr(values_like, "parameter_values", values_like),
            dtype=float,
        )
        if values.shape != base.shape or not np.all(np.isfinite(values)):
            continue
        scaled = (values - base) / scales
        if any(np.linalg.norm(scaled - existing) <= 1.0e-3 for existing in selected_anchors):
            continue
        selected_anchors.append(scaled.copy())
        if len(selected_anchors) >= max(int(active_config.max_modes), 1):
            break

    def _logsumexp(values: np.ndarray) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return -math.inf
        maximum = float(np.max(finite))
        return maximum + math.log(float(np.sum(np.exp(finite - maximum))))

    def _log_normal(values: np.ndarray, center: np.ndarray, sigma: float) -> float:
        centered = (values - center) / sigma
        return float(
            -0.5 * np.dot(centered, centered)
            - values.size * math.log(sigma)
            - 0.5 * values.size * math.log(2.0 * math.pi)
        )

    def _profile_terms(profile: Any) -> tuple[float, float, np.ndarray, np.ndarray]:
        relative = np.asarray(
            profile.session_offsets[1:] - profile.session_offsets[0],
            dtype=float,
        )
        integers = np.rint(relative).astype(np.int64)
        covariance = _relative_session_offset_covariance(profile)
        ambiguity = relative - integers.astype(float)
        phase_chi2 = (
            float(ambiguity @ np.linalg.solve(covariance, ambiguity))
            if covariance.size
            else 0.0
        )
        slope_chi2 = float(getattr(profile, "slope_chi2", profile.chi2))
        shape_chi2 = float(profile.chi2)
        if not np.isfinite(phase_chi2) or not np.isfinite(slope_chi2) or not np.isfinite(shape_chi2):
            raise ValueError("Non-finite wrapped phase or shape evidence.")
        if evidence_mode == "phase-only":
            evidence_chi2 = phase_chi2
        elif evidence_mode == "phase+slope":
            evidence_chi2 = phase_chi2 + max(slope_chi2, 0.0)
        elif evidence_mode == "shape":
            evidence_chi2 = shape_chi2
        else:
            evidence_chi2 = phase_chi2 + max(shape_chi2, 0.0)
        return (
            float(evidence_chi2),
            float(phase_chi2),
            integers,
            np.asarray(profile.normalized_residuals, dtype=float),
        )

    def _evaluate(scaled_values: np.ndarray):
        values = base + scales * np.asarray(scaled_values, dtype=float)
        try:
            profile = evaluator.evaluate_values(values)
            evidence_chi2, phase_chi2, integers, _ = _profile_terms(profile)
        except (ArithmeticError, FloatingPointError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return -math.inf, -math.inf, None, None, None
        prior_chi2 = float(np.dot(scaled_values, scaled_values) / (prior_scale * prior_scale))
        log_prior = -0.5 * prior_chi2
        log_likelihood = -0.5 * evidence_chi2
        return log_prior + log_likelihood, log_likelihood, profile, integers, values

    def _normalize_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
        normalizer = _logsumexp(log_weights)
        if not np.isfinite(normalizer):
            raise RuntimeError("Posterior particle weights became non-finite.")
        return log_weights - normalizer, normalizer

    def _effective_sample_size(log_weights: np.ndarray) -> float:
        weights = np.exp(log_weights)
        return float(1.0 / np.sum(np.square(weights)))

    def _systematic_indices(weights: np.ndarray) -> np.ndarray:
        count = weights.size
        positions = (rng.random() + np.arange(count, dtype=float)) / count
        cumulative = np.cumsum(weights)
        return np.searchsorted(cumulative, positions, side="right").clip(0, count - 1)

    # Use a proper mixture proposal so deterministic shape modes improve
    # coverage without changing the target prior through untracked injection.
    anchor_count = len(selected_anchors)
    anchor_mix = anchor_fraction if anchor_count else 0.0
    scaled_particles = np.empty((particle_count, scales.size), dtype=float)
    for index in range(particle_count):
        if anchor_count and index < int(round(particle_count * anchor_mix)):
            anchor = selected_anchors[index % anchor_count]
            scaled_particles[index] = anchor + rng.normal(0.0, anchor_scale, scales.size)
        else:
            scaled_particles[index] = rng.normal(0.0, prior_scale, scales.size)

    log_prior = np.empty(particle_count, dtype=float)
    log_likelihood = np.empty(particle_count, dtype=float)
    profiles: list[Any | None] = [None] * particle_count
    integers: list[np.ndarray | None] = [None] * particle_count
    for index, particle in enumerate(scaled_particles):
        total, likelihood, profile, integer_offsets, _ = _evaluate(particle)
        log_prior[index] = total - likelihood if np.isfinite(total) else -math.inf
        log_likelihood[index] = likelihood
        profiles[index] = profile
        integers[index] = integer_offsets

    log_proposal = np.empty(particle_count, dtype=float)
    for index, particle in enumerate(scaled_particles):
        components = [
            math.log(max(1.0 - anchor_mix, 1.0e-300))
            + _log_normal(particle, np.zeros(scales.size), prior_scale)
        ]
        if anchor_count:
            component_weight = anchor_mix / anchor_count
            components.extend(
                math.log(max(component_weight, 1.0e-300))
                + _log_normal(particle, anchor, anchor_scale)
                for anchor in selected_anchors
            )
        log_proposal[index] = _logsumexp(np.asarray(components, dtype=float))
    log_weights, _ = _normalize_log_weights(log_prior - log_proposal)

    beta = 0.0
    annealing_records: list[dict[str, Any]] = []
    target_ess = min(max(float(active_config.posterior_particle_ess_target), 0.2), 0.95)
    resample_threshold = min(
        max(float(active_config.posterior_particle_resample_threshold), 0.0),
        1.0,
    )
    max_tempering_steps = max(int(active_config.posterior_particle_max_tempering_steps), 1)
    mutation_scale = max(float(active_config.posterior_particle_mutation_scale), 1.0e-5)
    mutation_steps = max(int(active_config.posterior_particle_mutation_steps), 0)
    for _ in range(max_tempering_steps):
        if beta >= 1.0 - 1.0e-12:
            break
        remaining = 1.0 - beta
        trial_log_weights = log_weights + remaining * log_likelihood
        if _effective_sample_size(trial_log_weights - _logsumexp(trial_log_weights)) >= target_ess * particle_count:
            next_beta = 1.0
        else:
            low = 0.0
            high = remaining
            for _ in range(36):
                middle = 0.5 * (low + high)
                candidate_log_weights = log_weights + middle * log_likelihood
                candidate_log_weights -= _logsumexp(candidate_log_weights)
                if _effective_sample_size(candidate_log_weights) >= target_ess * particle_count:
                    low = middle
                else:
                    high = middle
            next_beta = beta + low
            if next_beta <= beta + 1.0e-10:
                next_beta = min(beta + max(remaining / 8.0, 1.0e-3), 1.0)
        delta = next_beta - beta
        log_weights, log_increment = _normalize_log_weights(log_weights + delta * log_likelihood)
        beta = next_beta
        ess = _effective_sample_size(log_weights)
        resampled = False
        if ess <= resample_threshold * particle_count or beta >= 1.0 - 1.0e-12:
            weights = np.exp(log_weights)
            indices = _systematic_indices(weights)
            scaled_particles = scaled_particles[indices].copy()
            log_prior = log_prior[indices].copy()
            log_likelihood = log_likelihood[indices].copy()
            profiles = [profiles[int(index)] for index in indices]
            integers = [integers[int(index)] for index in indices]
            log_weights = np.full(particle_count, -math.log(particle_count), dtype=float)
            resampled = True
            accepted = 0
            attempted = 0
            for _ in range(mutation_steps):
                for index in range(particle_count):
                    proposal = scaled_particles[index] + rng.normal(
                        0.0, mutation_scale, scales.size
                    )
                    proposed_total, proposed_likelihood, proposed_profile, proposed_integers, _ = _evaluate(proposal)
                    if not np.isfinite(proposed_total) or not np.isfinite(
                        proposed_likelihood
                    ):
                        attempted += 1
                        continue
                    proposed_prior = proposed_total - proposed_likelihood
                    attempted += 1
                    if (
                        math.log(rng.random())
                        < (proposed_prior + beta * proposed_likelihood)
                        - (log_prior[index] + beta * log_likelihood[index])
                    ):
                        scaled_particles[index] = proposal
                        log_prior[index] = proposed_prior
                        log_likelihood[index] = proposed_likelihood
                        profiles[index] = proposed_profile
                        integers[index] = proposed_integers
                        accepted += 1
            acceptance = float(accepted / attempted) if attempted else 0.0
            if acceptance < 0.15:
                mutation_scale *= 0.7
            elif acceptance > 0.5:
                mutation_scale *= 1.3
        annealing_records.append(
            {
                "beta": float(beta),
                "delta_beta": float(delta),
                "effective_sample_size": float(ess),
                "log_evidence_increment": float(log_increment),
                "resampled": resampled,
                "mutation_scale": float(mutation_scale),
                "mutation_acceptance": float(acceptance) if resampled else None,
            }
        )
    if beta < 1.0 - 1.0e-8:
        raise RuntimeError("Posterior particle annealing did not reach beta=1.")

    final_weights = np.exp(log_weights - _logsumexp(log_weights))
    path_records: dict[tuple[int, ...], dict[str, Any]] = {}
    for index, (weight, profile, integer_offsets) in enumerate(
        zip(final_weights, profiles, integers, strict=True)
    ):
        if profile is None or integer_offsets is None or not np.isfinite(weight):
            continue
        path = integer_session_offsets_to_gap_wraps(integer_offsets)
        target_log = float(log_prior[index] + log_likelihood[index])
        record = path_records.setdefault(
            path,
            {
                "mass": 0.0,
                "best_log_posterior": -math.inf,
                "best_index": index,
                "integer_offsets": np.asarray(integer_offsets, dtype=np.int64).copy(),
            },
        )
        record["mass"] += float(weight)
        if target_log > float(record["best_log_posterior"]):
            record["best_log_posterior"] = target_log
            record["best_index"] = index
            record["integer_offsets"] = np.asarray(integer_offsets, dtype=np.int64).copy()

    if not path_records:
        raise RuntimeError("Posterior particle search produced no finite phase paths.")
    ranked_paths = sorted(
        path_records.items(),
        key=lambda item: (
            -float(item[1]["mass"]),
            -float(item[1]["best_log_posterior"]),
            item[0],
        ),
    )
    path_limit = max(int(active_config.posterior_particle_max_paths), 1)
    selected = []
    path_diagnostics = []
    for rank, (path, record) in enumerate(ranked_paths[:path_limit]):
        index = int(record["best_index"])
        values = base + scales * scaled_particles[index]
        selected.append(
            ActiveTimingBranch(
                branch_id=f"posterior_particle_phase_path_{rank:04d}",
                structure_id=str(structure_id),
                binary_model=(str(binary_model).upper() if binary_model is not None else None),
                wraps=path,
                theta_names=names,
                theta_offset=np.asarray(values - base, dtype=float),
                log_prior=float(math.log(max(float(record["mass"]), 1.0e-300))),
                proposal_source="apollo-bayesian-posterior-particle-phase-path",
                diagnostics={
                    "proposal_family": "annealed-wrap-invariant-posterior-particle-path",
                    "posterior_particle_index": index,
                    "posterior_path_mass": float(record["mass"]),
                    "posterior_path_rank": rank,
                    "posterior_particle_log_posterior": float(record["best_log_posterior"]),
                    "posterior_particle_evidence_mode": evidence_mode,
                    "integer_session_offsets": (0,) + tuple(
                        int(value) for value in record["integer_offsets"]
                    ),
                    "certification_theta_offset": np.asarray(values - base, dtype=float),
                },
            )
        )
        path_diagnostics.append(
            {
                "rank": rank,
                "wraps": path,
                "mass": float(record["mass"]),
                "best_log_posterior": float(record["best_log_posterior"]),
            }
        )
    return selected, {
        "proposal_source": "apollo-bayesian-posterior-particle-phase-path",
        "proposal_family": "annealed-wrap-invariant-posterior-particle-path",
        "evidence_mode": evidence_mode,
        "particle_count": particle_count,
        "anchor_count": anchor_count,
        "anchor_fraction": anchor_mix,
        "unique_particle_path_count": len(path_records),
        "selected_path_count": len(selected),
        "effective_sample_size": float(_effective_sample_size(log_weights)),
        "annealing": annealing_records,
        "paths": path_diagnostics,
    }


def generate_global_phase_path_refresh_branches(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    mode_count: int = 8,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Refresh all session integers from a partial Bayesian posterior forest.

    The active edge linker grows a phase forest one bridge at a time. A large
    posterior jump at a bridge makes that local factorization unreliable. This
    refresh relinearizes a diverse portfolio of posterior modes, marginalizes
    the continuous timing parameters, and enumerates the joint session-integer
    vector. It proposes global paths without source-specific wrap overrides.
    """

    scales = np.asarray(parameter_scales, dtype=float)
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if scales.ndim != 1 or base.shape != scales.shape or len(names) != scales.size:
        raise ValueError("Global phase refresh parameter dimensions must align.")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Global phase refresh scales must be finite and positive.")
    if not np.all(np.isfinite(base)):
        raise ValueError("Global phase refresh base values must be finite.")

    active_config = config or MultimodeSessionLatticeConfig()
    target_modes = max(int(mode_count), 1)
    step_fraction = max(float(active_config.finite_difference_fraction), 1.0e-12)
    parameter_steps = np.maximum(scales * step_fraction, np.finfo(float).tiny)
    selected_records: list[tuple[np.ndarray, np.ndarray, float, Any]] = []
    for values_like in initial_modes:
        values = np.asarray(
            getattr(values_like, "parameter_values", values_like),
            dtype=float,
        )
        if values.shape != base.shape or not np.all(np.isfinite(values)):
            continue
        covariance_like = getattr(values_like, "scaled_covariance", None)
        if covariance_like is None:
            covariance = np.eye(scales.size, dtype=float)
        else:
            covariance = np.asarray(covariance_like, dtype=float)
            if covariance.shape != (scales.size, scales.size) or not np.all(
                np.isfinite(covariance)
            ):
                covariance = np.eye(scales.size, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors
            @ np.diag(np.clip(eigenvalues, 1.0e-12, 1.0e12))
            @ eigenvectors.T
        )
        if any(
            np.linalg.norm((values - existing[0]) / scales) <= 1.0e-3
            and np.allclose(covariance, existing[1], rtol=1.0e-3, atol=1.0e-12)
            for existing in selected_records
        ):
            continue
        log_evidence = float(getattr(values_like, "laplace_log_evidence", 0.0))
        if not np.isfinite(log_evidence):
            log_evidence = 0.0
        selected_records.append((values.copy(), covariance, log_evidence, values_like))
        if len(selected_records) >= target_modes:
            break

    linearizations: list[
        tuple[np.ndarray, np.ndarray, float, Any, SessionShapeLinearization]
    ] = []
    failed_mode_indices: list[int] = []
    for mode_index, (values, covariance, log_evidence, values_like) in enumerate(
        selected_records
    ):
        try:
            linearizations.append(
                (
                    values,
                    covariance,
                    log_evidence,
                    values_like,
                    evaluator.linearize_values(
                        values,
                        parameter_steps=parameter_steps,
                    ),
                )
            )
        except Exception:
            failed_mode_indices.append(mode_index)
    if not linearizations:
        raise RuntimeError("Global phase refresh produced no valid linearization.")

    scored: list[tuple[float, float, int, int, ActiveTimingBranch]] = []
    mode_diagnostics: list[dict[str, Any]] = []
    for mode_index, (
        values,
        covariance,
        log_evidence,
        values_like,
        linearization,
    ) in enumerate(linearizations):
        session_count = int(linearization.profile.session_count)
        labels = getattr(values_like, "component_labels", None)
        integers = getattr(values_like, "session_integers", None)
        groups_by_key: dict[tuple[str, int], list[int]] = {}
        if labels is not None and len(labels) == session_count:
            for session_index, label in enumerate(labels):
                key = (
                    ("component", int(label))
                    if label is not None
                    else ("singleton", int(session_index))
                )
                groups_by_key.setdefault(key, []).append(session_index)
        else:
            for session_index in range(session_count):
                groups_by_key[("singleton", session_index)] = [session_index]
        groups = sorted(
            (tuple(sorted(members)) for members in groups_by_key.values()),
            key=lambda members: members[0],
        )
        root_group_index = next(
            index for index, members in enumerate(groups) if 0 in members
        )
        component_anchors = [
            members[0]
            for index, members in enumerate(groups)
            if index != root_group_index
        ]
        relative = np.asarray(
            [
                linearization.profile.session_offsets[anchor]
                - linearization.profile.session_offsets[0]
                for anchor in component_anchors
            ],
            dtype=float,
        )
        scaled_relative_jacobian = np.asarray(
            [
                linearization.session_offset_jacobian[anchor]
                - linearization.session_offset_jacobian[0]
                for anchor in component_anchors
            ],
            dtype=float,
        ) / step_fraction
        root_variance = float(
            linearization.profile.session_offset_variances[0]
        )
        noise = np.diag(
            np.asarray(
                [
                    root_variance
                    + float(
                        linearization.profile.session_offset_variances[anchor]
                    )
                    for anchor in component_anchors
                ],
                dtype=float,
            )
        )
        if not component_anchors:
            continue
        predictive_covariance = noise + scaled_relative_jacobian @ covariance @ scaled_relative_jacobian.T
        predictive_covariance = 0.5 * (
            predictive_covariance + predictive_covariance.T
        )
        predictive_covariance += np.eye(predictive_covariance.shape[0]) * max(
            float(np.trace(predictive_covariance)) * 1.0e-12,
            1.0e-12,
        )
        try:
            integer_result = solve_integer_least_squares(
                relative,
                predictive_covariance,
                candidate_count=max(int(active_config.candidates_per_mode), 1),
                max_visited_nodes=max(int(active_config.max_visited_nodes), 1),
                decimal_precision=max(int(active_config.decimal_precision), 32),
            )
        except Exception as error:
            failed_mode_indices.append(mode_index)
            mode_diagnostics.append(
                {
                    "mode_index": mode_index,
                    "candidate_count": 0,
                    "error": str(error),
                }
            )
            continue

        precision = np.linalg.pinv(predictive_covariance)
        mode_records = 0
        calibration_scale = getattr(values_like, "calibration_scale", None)
        posterior_mass = getattr(values_like, "posterior_mass", None)
        orbit_slope_chi2 = getattr(values_like, "slope_chi2", None)
        orbit_source_start_index = getattr(values_like, "source_start_index", None)
        for candidate_index, integer_candidate in enumerate(integer_result.candidates):
            innovation = np.asarray(integer_candidate.integer_vector, dtype=float) - relative
            conditional_scaled_offset = (
                covariance
                @ scaled_relative_jacobian.T
                @ precision
                @ innovation
            )
            candidate_values = values + scales * conditional_scaled_offset
            component_integers = tuple(
                int(value) for value in integer_candidate.integer_vector
            )
            translations = {root_group_index: 0}
            for translation_index, group_index in enumerate(
                index
                for index in range(len(groups))
                if index != root_group_index
            ):
                translations[group_index] = component_integers[translation_index]
            full_integer_path = [0 for _ in range(session_count)]
            for group_index, members in enumerate(groups):
                translation = int(translations[group_index])
                anchor = members[0]
                for session_index in members:
                    internal_offset = 0
                    if (
                        integers is not None
                        and len(integers) == session_count
                        and integers[session_index] is not None
                        and integers[anchor] is not None
                    ):
                        internal_offset = int(integers[session_index]) - int(
                            integers[anchor]
                        )
                    full_integer_path[session_index] = (
                        translation + internal_offset
                    )
            full_integer_path[0] = 0
            full_integer_path = tuple(int(value) for value in full_integer_path)
            proposal_cost = float(
                integer_candidate.squared_mahalanobis - 2.0 * log_evidence
            )
            branch = ActiveTimingBranch(
                branch_id=(
                    f"global_phase_refresh_m{mode_index:03d}_"
                    f"i{candidate_index:04d}"
                ),
                structure_id=str(structure_id),
                binary_model=(
                    str(binary_model).upper() if binary_model is not None else None
                ),
                wraps=integer_session_offsets_to_gap_wraps(full_integer_path[1:]),
                theta_names=names,
                theta_offset=np.asarray(candidate_values - base, dtype=float),
                log_prior=float(log_evidence - 0.5 * integer_candidate.squared_mahalanobis),
                proposal_source="apollo-bayesian-global-phase-path-refresh",
                diagnostics={
                    "proposal_family": "posterior-predictive-joint-phase-path",
                    "mode_index": mode_index,
                    "integer_candidate_index": candidate_index,
                    "integer_session_offsets": full_integer_path,
                    "component_translation_integers": component_integers,
                    "component_anchors": component_anchors,
                    "integer_squared_mahalanobis": integer_candidate.squared_mahalanobis,
                    "posterior_mode_log_evidence": log_evidence,
                    "posterior_mode_mass": posterior_mass,
                    "slope_calibration_scale": calibration_scale,
                    "orbit_slope_chi2": orbit_slope_chi2,
                    "orbit_source_start_index": orbit_source_start_index,
                    "posterior_mode_parameter_values": values.copy(),
                    "posterior_predictive_center": relative,
                    "posterior_predictive_covariance": predictive_covariance,
                    "conditional_scaled_parameter_offset": conditional_scaled_offset,
                    "certification_theta_offset": np.asarray(
                        candidate_values - base,
                        dtype=float,
                    ),
                },
            )
            scored.append(
                (
                    proposal_cost,
                    float(integer_candidate.squared_mahalanobis),
                    mode_index,
                    candidate_index,
                    branch,
                )
            )
            mode_records += 1
        mode_diagnostics.append(
            {
                "mode_index": mode_index,
                "candidate_count": mode_records,
                "visited_nodes": integer_result.visited_nodes,
                "posterior_mode_log_evidence": log_evidence,
                "posterior_mode_mass": posterior_mass,
                "slope_calibration_scale": calibration_scale,
                "orbit_slope_chi2": orbit_slope_chi2,
                "orbit_source_start_index": orbit_source_start_index,
                "predictive_center_min": float(np.min(relative)),
                "predictive_center_max": float(np.max(relative)),
            }
        )

    ranked = sorted(scored, key=lambda item: item[:4])
    selected: list[ActiveTimingBranch] = []
    seen_paths: set[tuple[int, ...]] = set()
    seen_branches: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    candidate_limit = max(int(active_config.max_candidates), 1)
    for _, _, _, _, branch in ranked:
        if branch.wraps in seen_paths:
            continue
        key = (
            branch.wraps,
            tuple(float(value) for value in np.round(branch.theta_offset, 15)),
        )
        if key in seen_branches:
            continue
        seen_branches.add(key)
        seen_paths.add(branch.wraps)
        selected.append(branch)
        if len(selected) >= candidate_limit:
            break
    if len(selected) < candidate_limit:
        for _, _, _, _, branch in ranked:
            key = (
                branch.wraps,
                tuple(float(value) for value in np.round(branch.theta_offset, 15)),
            )
            if key in seen_branches:
                continue
            seen_branches.add(key)
            selected.append(branch)
            if len(selected) >= candidate_limit:
                break

    refreshed = [
        ActiveTimingBranch(
            branch_id=branch.branch_id,
            structure_id=branch.structure_id,
            binary_model=branch.binary_model,
            wraps=branch.wraps,
            theta_names=branch.theta_names,
            theta_offset=np.asarray(branch.theta_offset, dtype=float),
            log_prior=branch.log_prior,
            proposal_source="apollo-bayesian-global-phase-path-refresh",
            diagnostics={
                **(branch.diagnostics or {}),
                "proposal_family": "posterior-predictive-joint-phase-path",
                "refresh_mode_count": len(linearizations),
                "refresh_failed_mode_indices": failed_mode_indices,
                "certification_theta_offset": np.asarray(
                    branch.theta_offset,
                    dtype=float,
                ),
            },
        )
        for branch in selected
    ]
    return refreshed, {
        "proposal_source": "apollo-bayesian-global-phase-path-refresh",
        "mode_count_requested": target_modes,
        "mode_count_selected": len(selected_records),
        "mode_count_linearized": len(linearizations),
        "failed_mode_indices": failed_mode_indices,
        "parameter_steps": parameter_steps,
        "candidate_count": len(scored),
        "unique_candidate_path_count": len({item[4].wraps for item in scored}),
        "kept_count": len(refreshed),
        "unique_kept_path_count": len({branch.wraps for branch in refreshed}),
        "modes": mode_diagnostics,
    }


def generate_shape_conditioned_link_lattice_branches(
    *,
    linearizations: Sequence[SessionShapeLinearization],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    link_parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    """Enumerate phase links on a low-shape posterior parameter block.

    Parameters outside ``link_parameter_names`` remain at each nonlinear
    session-shape mode.  This prevents an integer-lattice linearization from
    moving well-constrained binary-shape parameters while spin-like parameters
    trace the long-baseline phase aliases.
    """
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    link_names = tuple(str(name).upper() for name in link_parameter_names)
    if base.shape != (len(names),):
        raise ValueError("Base parameter values must match parameter names.")
    if not link_names or len(set(link_names)) != len(link_names):
        raise ValueError("Link parameter names must be non-empty and unique.")
    missing = [name for name in link_names if name not in names]
    if missing:
        raise ValueError(f"Unknown link parameters: {', '.join(missing)}")
    link_indices = np.asarray([names.index(name) for name in link_names], dtype=int)

    projected: list[SessionShapeLinearization] = []
    for linearization in linearizations:
        if linearization.parameter_values.shape != base.shape:
            raise ValueError("Every linearization must match the full parameter set.")
        shape_jacobian = linearization.shape_jacobian[:, link_indices]
        relative_jacobian = linearization.relative_session_jacobian[:, link_indices]
        projected.append(
            SessionShapeLinearization(
                parameter_values=linearization.parameter_values[link_indices].copy(),
                parameter_steps=linearization.parameter_steps[link_indices].copy(),
                profile=linearization.profile,
                shape_jacobian=shape_jacobian,
                session_offset_jacobian=(
                    linearization.session_offset_jacobian[:, link_indices]
                ),
                relative_session_offsets=linearization.relative_session_offsets.copy(),
                relative_session_jacobian=relative_jacobian,
                relative_offset_covariance=(
                    linearization.relative_offset_covariance.copy()
                ),
                marginalized_integer_covariance=build_marginalized_integer_covariance(
                    shape_jacobian,
                    relative_jacobian,
                    linearization.relative_offset_covariance,
                ),
            )
        )

    link_branches, diagnostics = generate_session_lattice_branches_from_linearizations(
        linearizations=projected,
        base_parameter_values=base[link_indices],
        parameter_names=link_names,
        structure_id=structure_id,
        binary_model=binary_model,
        stage_index=stage_index,
        config=config,
    )
    expanded: list[ActiveTimingBranch] = []
    conditioned_names = tuple(name for name in names if name not in set(link_names))
    for branch in link_branches:
        mode_index = int((branch.diagnostics or {})["mode_index"])
        full_offset = np.asarray(
            linearizations[mode_index].parameter_values - base,
            dtype=float,
        )
        full_offset[link_indices] = np.asarray(branch.theta_offset, dtype=float)
        expanded.append(
            ActiveTimingBranch(
                branch_id=branch.branch_id.replace(
                    "session_lattice",
                    "shape_link_lattice",
                ),
                structure_id=branch.structure_id,
                binary_model=branch.binary_model,
                wraps=branch.wraps,
                theta_names=names,
                theta_offset=full_offset,
                log_prior=branch.log_prior,
                proposal_source="apollo-shape-conditioned-link-lattice",
                diagnostics={
                    **(branch.diagnostics or {}),
                    "proposal_family": "shape-conditioned-link-lattice",
                    "link_parameter_names": link_names,
                    "shape_conditioned_parameter_names": conditioned_names,
                    "shape_mode_values": linearizations[mode_index].parameter_values,
                    "certification_theta_offset": full_offset,
                },
            )
        )

    return expanded, {
        **diagnostics,
        "proposal_source": "apollo-shape-conditioned-link-lattice",
        "link_parameter_names": link_names,
        "shape_conditioned_parameter_names": conditioned_names,
    }


def generate_multimode_session_lattice_branches(
    *,
    evaluator: Any,
    parameter_names: Sequence[str],
    parameter_scales: Sequence[float],
    certifications: Sequence[BranchCertification],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    active_config = config or MultimodeSessionLatticeConfig()
    base = np.asarray(evaluator.base_parameter_values, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.shape != base.shape or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite, positive, and dimensionally matched.")
    step_fraction = max(float(active_config.finite_difference_fraction), 1.0e-12)
    steps = np.maximum(scales * step_fraction, np.finfo(float).tiny)
    modes, mode_selection = certified_parameter_modes(
        base_parameter_values=base,
        parameter_names=parameter_names,
        certifications=certifications,
        max_modes=active_config.max_modes,
    )
    shape_optimization: dict[str, Any] = {
        "enabled": False,
        "reason": "disabled",
    }
    if bool(active_config.shape_optimization_enabled):
        optimized_modes, optimization_diagnostics = optimize_session_shape_modes(
            evaluator=evaluator,
            initial_modes=modes,
            parameter_scales=scales,
            max_modes=min(
                max(int(active_config.shape_optimization_max_modes), 1),
                max(int(active_config.max_modes), 1),
            ),
            trust_radius=max(
                float(active_config.shape_optimization_trust_radius),
                1.0e-12,
            ),
            max_nfev=max(int(active_config.shape_optimization_max_nfev), 1),
        )
        refined_modes = list(optimized_modes)
        for original_mode in modes:
            if any(
                np.linalg.norm((original_mode - refined_mode) / scales) <= 1.0e-6
                for refined_mode in refined_modes
            ):
                continue
            refined_modes.append(np.asarray(original_mode, dtype=float))
            if len(refined_modes) >= max(int(active_config.max_modes), 1):
                break
        modes = tuple(refined_modes[: max(int(active_config.max_modes), 1)])
        shape_optimization = {
            "enabled": True,
            "reason": "optimized",
            **optimization_diagnostics,
            "lattice_mode_count": len(modes),
        }
    linearizations = tuple(
        evaluator.linearize_values(mode, parameter_steps=steps)
        for mode in modes
    )
    branches, diagnostics = generate_session_lattice_branches_from_linearizations(
        linearizations=linearizations,
        base_parameter_values=base,
        parameter_names=parameter_names,
        structure_id=structure_id,
        binary_model=binary_model,
        stage_index=stage_index,
        config=active_config,
    )
    return branches, {
        **diagnostics,
        "mode_selection": mode_selection,
        "parameter_steps": steps,
        "shape_optimization": shape_optimization,
    }
