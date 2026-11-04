from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from timing_accelerator.inference.active_timing_contracts import ActiveTimingBranch
from timing_accelerator.inference.wrap_invariant_binary_likelihood import (
    integer_session_offsets_to_gap_wraps,
)


@dataclass(frozen=True)
class ActiveSessionLinkConfig:
    """Budget and posterior controls for Bayesian active session linking."""

    beam_size: int = 24
    max_beam_size: int = 96
    initial_integer_radius: int = 2
    max_integer_radius: int = 12
    candidate_sigma: float = 4.0
    max_jobs_per_stage: int = 96
    max_jobs_per_stage_limit: int = 384
    integer_candidates_per_edge: int = 1
    trust_radius: float = 8.0
    max_nfev: int = 60
    finite_difference_fraction: float = 1.0e-4
    prior_weight: float = 1.0
    max_modes_per_path: int = 2
    objective_fraction: float = 0.25
    island_fraction: float = 0.25
    predictive_fraction: float = 0.5
    predictive_candidate_limit: int = 0
    auto_expand_rounds: int = 0
    boundary_fraction_trigger: float = 0.05
    tail_mass_trigger: float = 0.05
    bridge_log_evidence_gate: float = -10.0
    bridge_standardized_innovation_gate: float = 5.0
    relocation_edge_count: int = 8
    relocation_trust_radius: float = 16.0
    relocation_max_nfev: int = 80
    continuous_mode_merge_radius: float = 0.05
    topology_particles: bool = False
    topology_edges_per_mode: int = 3
    topology_bottleneck_edges_per_mode: int = 1
    topology_fraction: float = 0.0
    predictive_relinearization_count: int = 0
    bridge_outlier_probability: float = 0.0
    bridge_outlier_scale: float = 0.25
    backward_rejuvenation: bool = False
    backward_edge_count: int = 6
    backward_integer_radius: int = 2
    backward_bridge_integer_radius: int = 0
    backward_max_jobs: int = 192
    backward_restart_count: int = 8
    backward_min_cut_size: int = 1
    backward_max_cut_size: int = 1
    backward_cutset_pool_size: int = 4
    backward_reconnection_beam_size: int = 4
    stop_on_bridge_veto: bool = False
    global_path_refresh_on_bridge_jump: bool = False


@dataclass(frozen=True)
class ActiveSessionPathMode:
    """One continuous posterior mode tied to a partial session-integer tree."""

    session_integers: tuple[int | None, ...]
    component_labels: tuple[int | None, ...]
    linked_sessions: tuple[int, ...]
    root_session: int
    parameter_values: np.ndarray
    shape_chi2: float
    ambiguity_chi2: float
    prior_chi2: float
    posterior_chi2: float
    laplace_log_evidence: float
    future_bridge_log_evidence: float
    predictive_deviance: float
    predictive_anchor_session: int | None
    predictive_added_session: int | None
    scaled_covariance: np.ndarray
    source_mode_index: int
    parent_mode_index: int
    anchor_session: int | None
    added_session: int | None
    edge_integer: int | None
    predicted_edge_phase: float | None
    predicted_edge_sigma: float | None
    candidate_radius: int
    success: bool
    nfev: int


def active_session_path_mode_to_payload(
    mode: ActiveSessionPathMode,
) -> dict[str, Any]:
    return {
        **mode.__dict__,
        "parameter_values": np.asarray(mode.parameter_values, dtype=float).tolist(),
        "scaled_covariance": np.asarray(mode.scaled_covariance, dtype=float).tolist(),
    }


def active_session_path_mode_from_payload(
    payload: Mapping[str, Any],
) -> ActiveSessionPathMode:
    return ActiveSessionPathMode(
        session_integers=tuple(
            None if value is None else int(value)
            for value in payload["session_integers"]
        ),
        component_labels=tuple(
            None if value is None else int(value)
            for value in payload["component_labels"]
        ),
        linked_sessions=tuple(int(value) for value in payload["linked_sessions"]),
        root_session=int(payload["root_session"]),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        shape_chi2=float(payload["shape_chi2"]),
        ambiguity_chi2=float(payload["ambiguity_chi2"]),
        prior_chi2=float(payload["prior_chi2"]),
        posterior_chi2=float(payload["posterior_chi2"]),
        laplace_log_evidence=float(payload["laplace_log_evidence"]),
        future_bridge_log_evidence=float(
            payload.get("future_bridge_log_evidence", 0.0)
        ),
        predictive_deviance=float(
            payload.get("predictive_deviance", payload["posterior_chi2"])
        ),
        predictive_anchor_session=(
            None
            if payload.get("predictive_anchor_session") is None
            else int(payload["predictive_anchor_session"])
        ),
        predictive_added_session=(
            None
            if payload.get("predictive_added_session") is None
            else int(payload["predictive_added_session"])
        ),
        scaled_covariance=np.asarray(payload["scaled_covariance"], dtype=float),
        source_mode_index=int(payload["source_mode_index"]),
        parent_mode_index=int(payload["parent_mode_index"]),
        anchor_session=(
            None
            if payload.get("anchor_session") is None
            else int(payload["anchor_session"])
        ),
        added_session=(
            None
            if payload.get("added_session") is None
            else int(payload["added_session"])
        ),
        edge_integer=(
            None if payload.get("edge_integer") is None else int(payload["edge_integer"])
        ),
        predicted_edge_phase=(
            None
            if payload.get("predicted_edge_phase") is None
            else float(payload["predicted_edge_phase"])
        ),
        predicted_edge_sigma=(
            None
            if payload.get("predicted_edge_sigma") is None
            else float(payload["predicted_edge_sigma"])
        ),
        candidate_radius=int(payload["candidate_radius"]),
        success=bool(payload["success"]),
        nfev=int(payload["nfev"]),
    )


def _profile_relative_covariance(
    profile: Any,
    *,
    root_session: int,
    other_sessions: Sequence[int],
) -> np.ndarray:
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    others = np.asarray(tuple(int(value) for value in other_sessions), dtype=int)
    if others.size == 0:
        return np.zeros((0, 0), dtype=float)
    covariance = np.diag(variances[others]) + float(variances[int(root_session)])
    jitter = max(float(np.trace(covariance)) * 1.0e-14, 1.0e-24)
    return covariance + np.eye(others.size, dtype=float) * jitter


def _component_groups(
    component_labels: Sequence[int | None],
) -> tuple[tuple[int, ...], ...]:
    grouped: dict[int, list[int]] = {}
    for session_index, label in enumerate(component_labels):
        if label is not None:
            grouped.setdefault(int(label), []).append(session_index)
    return tuple(
        tuple(indices)
        for _, indices in sorted(grouped.items())
        if len(indices) >= 2
    )


def _linked_profile_terms(
    profile: Any,
    session_integers: Sequence[int | None],
    *,
    component_labels: Sequence[int | None],
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    groups = _component_groups(component_labels)
    ambiguity_terms: list[np.ndarray] = []
    for group in groups:
        root_session = int(group[0])
        other_sessions = tuple(int(value) for value in group[1:])
        root_integer = int(session_integers[root_session])
        observed = np.asarray(
            [
                float(profile.session_offsets[index])
                - float(profile.session_offsets[root_session])
                for index in other_sessions
            ],
            dtype=float,
        )
        expected = np.asarray(
            [int(session_integers[index]) - root_integer for index in other_sessions],
            dtype=float,
        )
        covariance = _profile_relative_covariance(
            profile,
            root_session=root_session,
            other_sessions=other_sessions,
        )
        ambiguity_terms.append(
            np.linalg.solve(np.linalg.cholesky(covariance), observed - expected)
        )
    ambiguity = (
        np.concatenate(ambiguity_terms)
        if ambiguity_terms
        else np.zeros(0, dtype=float)
    )
    return shape, ambiguity, groups


def _parameter_domain_ok(
    values: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
) -> bool:
    return bool(
        np.all(np.isfinite(values))
        and all(
            (lower is None or float(values[index]) >= float(lower))
            and (upper is None or float(values[index]) < float(upper))
            for index, (lower, upper) in enumerate(bounds)
        )
    )


def _finite_difference_geometry(
    *,
    evaluator: Any,
    values: np.ndarray,
    parameter_scales: np.ndarray,
    fraction: float,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Return shape and session-offset Jacobians in prior-scale coordinates."""
    profile = evaluator.evaluate_values(values)
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    shape_jacobian = np.empty((shape.size, values.size), dtype=float)
    offset_jacobian = np.empty((offsets.size, values.size), dtype=float)
    step_fraction = max(float(fraction), 1.0e-12)

    for parameter_index, scale in enumerate(parameter_scales):
        displacement = np.zeros_like(values)
        displacement[parameter_index] = float(scale) * step_fraction
        plus = minus = None
        try:
            plus = evaluator.evaluate_values(values + displacement)
        except Exception:
            plus = None
        try:
            minus = evaluator.evaluate_values(values - displacement)
        except Exception:
            minus = None
        if plus is not None and minus is not None:
            denominator = 2.0 * step_fraction
            shape_jacobian[:, parameter_index] = (
                np.asarray(plus.normalized_residuals, dtype=float)
                - np.asarray(minus.normalized_residuals, dtype=float)
            ) / denominator
            offset_jacobian[:, parameter_index] = (
                np.asarray(plus.session_offsets, dtype=float)
                - np.asarray(minus.session_offsets, dtype=float)
            ) / denominator
        elif plus is not None:
            shape_jacobian[:, parameter_index] = (
                np.asarray(plus.normalized_residuals, dtype=float) - shape
            ) / step_fraction
            offset_jacobian[:, parameter_index] = (
                np.asarray(plus.session_offsets, dtype=float) - offsets
            ) / step_fraction
        elif minus is not None:
            shape_jacobian[:, parameter_index] = (
                shape - np.asarray(minus.normalized_residuals, dtype=float)
            ) / step_fraction
            offset_jacobian[:, parameter_index] = (
                offsets - np.asarray(minus.session_offsets, dtype=float)
            ) / step_fraction
        else:
            raise ValueError(
                f"No finite-difference probe was valid for parameter index "
                f"{parameter_index}."
            )
    return profile, shape_jacobian, offset_jacobian


def _posterior_covariance_from_geometry(
    *,
    profile: Any,
    shape_jacobian: np.ndarray,
    offset_jacobian: np.ndarray,
    mode: ActiveSessionPathMode,
    parameter_scales: np.ndarray,
    prior_scales: np.ndarray,
    prior_weight: float,
) -> np.ndarray:
    precision = shape_jacobian.T @ shape_jacobian
    _, _, groups = _linked_profile_terms(
        profile,
        mode.session_integers,
        component_labels=mode.component_labels,
    )
    for group in groups:
        root = int(group[0])
        other_sessions = tuple(int(value) for value in group[1:])
        relative_jacobian = (
            offset_jacobian[np.asarray(other_sessions)] - offset_jacobian[root]
        )
        covariance = _profile_relative_covariance(
            profile,
            root_session=root,
            other_sessions=other_sessions,
        )
        precision += relative_jacobian.T @ np.linalg.solve(
            covariance,
            relative_jacobian,
        )
    if prior_weight > 0.0:
        prior_diagonal = (
            math.sqrt(float(prior_weight)) * parameter_scales / prior_scales
        )
        precision += np.diag(np.square(prior_diagonal))
    jitter = max(float(np.trace(precision)) * 1.0e-12, 1.0e-12)
    return np.linalg.pinv(precision + np.eye(precision.shape[0]) * jitter)


def _discrete_gaussian(
    mean: float,
    sigma: float,
    *,
    radius: int,
) -> tuple[dict[int, float], float]:
    nearest = int(np.rint(float(mean)))
    integers = np.arange(nearest - int(radius), nearest + int(radius) + 1)
    variance = max(float(sigma) ** 2, 1.0e-18)
    log_likelihoods = (
        -0.5 * np.square(integers.astype(float) - float(mean)) / variance
        - 0.5 * math.log(2.0 * math.pi * variance)
    )
    maximum = float(np.max(log_likelihoods))
    weights = np.exp(log_likelihoods - maximum)
    log_evidence = maximum + math.log(float(np.sum(weights)))
    weights /= float(np.sum(weights))
    return (
        {
            int(value): float(weight)
            for value, weight in zip(integers, weights, strict=True)
        },
        float(log_evidence),
    )


def _discrete_robust_bridge(
    mean: float,
    sigma: float,
    *,
    radius: int,
    outlier_probability: float,
    outlier_scale: float,
) -> tuple[dict[int, float], float]:
    """Return a Gaussian-core/Cauchy-tail proposal over bridge integers.

    The tail is used only to propose and retain phase families. Candidate
    optimization and final certification continue to use the exact Gaussian
    TOA likelihood.
    """
    probability = min(max(float(outlier_probability), 0.0), 1.0 - 1.0e-12)
    if probability <= 0.0:
        return _discrete_gaussian(mean, sigma, radius=radius)

    nearest = int(np.rint(float(mean)))
    integers = np.arange(nearest - int(radius), nearest + int(radius) + 1)
    residuals = integers.astype(float) - float(mean)
    variance = max(float(sigma) ** 2, 1.0e-18)
    scale = max(float(outlier_scale), float(sigma), 1.0e-12)
    gaussian = (
        math.log1p(-probability)
        - 0.5 * np.square(residuals) / variance
        - 0.5 * math.log(2.0 * math.pi * variance)
    )
    cauchy = (
        math.log(probability)
        - math.log(math.pi * scale)
        - np.log1p(np.square(residuals / scale))
    )
    log_likelihoods = np.logaddexp(gaussian, cauchy)
    maximum = float(np.max(log_likelihoods))
    weights = np.exp(log_likelihoods - maximum)
    log_evidence = maximum + math.log(float(np.sum(weights)))
    weights /= float(np.sum(weights))
    return (
        {
            int(value): float(weight)
            for value, weight in zip(integers, weights, strict=True)
        },
        float(log_evidence),
    )


def _strict_future_bridge_diagnostic(
    *,
    profile: Any,
    mode: ActiveSessionPathMode,
    offset_jacobian: np.ndarray,
    config: ActiveSessionLinkConfig,
) -> dict[str, Any]:
    """Find the most plausible next component link under strict phase evidence."""

    def component_key(session_index: int) -> tuple[str, int]:
        label = mode.component_labels[session_index]
        return (
            ("component", int(label))
            if label is not None
            else ("singleton", int(session_index))
        )

    covariance = np.asarray(mode.scaled_covariance, dtype=float)
    ranked: list[
        tuple[
            float,
            float,
            int,
            int,
            float,
            float,
            int,
            tuple[str, int],
            tuple[str, int],
        ]
    ] = []
    for anchor in range(int(profile.session_count)):
        for added in range(anchor + 1, int(profile.session_count)):
            if component_key(anchor) == component_key(added):
                continue
            jacobian = offset_jacobian[added] - offset_jacobian[anchor]
            measurement_variance = float(
                profile.session_offset_variances[added]
                + profile.session_offset_variances[anchor]
            )
            predictive_variance = max(
                measurement_variance + float(jacobian @ covariance @ jacobian),
                1.0e-18,
            )
            sigma = math.sqrt(predictive_variance)
            radius = min(
                max(
                    int(config.initial_integer_radius),
                    int(math.ceil(float(config.candidate_sigma) * sigma)),
                ),
                max(int(config.max_integer_radius), int(config.initial_integer_radius)),
            )
            predicted = float(profile.session_offsets[added]) - float(
                profile.session_offsets[anchor]
            )
            _, log_evidence = _discrete_gaussian(
                predicted,
                sigma,
                radius=radius,
            )
            innovation = abs(predicted - float(np.rint(predicted))) / sigma
            ranked.append(
                (
                    -float(log_evidence),
                    float(predictive_variance),
                    int(anchor),
                    int(added),
                    float(predicted),
                    float(innovation),
                    int(radius),
                    component_key(anchor),
                    component_key(added),
                )
            )

    if not ranked:
        return {
            "future_bridge_log_evidence": 0.0,
            "predictive_deviance": float(mode.posterior_chi2),
            "predictive_anchor_session": None,
            "predictive_added_session": None,
            "predictive_phase": None,
            "predictive_sigma": None,
            "predictive_standardized_innovation": None,
            "predictive_radius": 0,
            "future_bridge_candidate_count": 0,
        }

    best = min(ranked)
    component_best: dict[
        tuple[str, int],
        tuple[
            float,
            float,
            int,
            int,
            float,
            float,
            int,
            tuple[str, int],
            tuple[str, int],
        ],
    ] = {}
    for edge in ranked:
        for key in (edge[7], edge[8]):
            if key not in component_best or edge < component_best[key]:
                component_best[key] = edge
    bottleneck = max(
        component_best.values(),
        key=lambda edge: (edge[0], edge[1], edge[2], edge[3]),
    )
    log_evidence = -float(bottleneck[0])
    return {
        "future_bridge_log_evidence": log_evidence,
        "predictive_deviance": float(mode.posterior_chi2) - 2.0 * log_evidence,
        "predictive_anchor_session": int(bottleneck[2]),
        "predictive_added_session": int(bottleneck[3]),
        "predictive_phase": float(bottleneck[4]),
        "predictive_sigma": math.sqrt(float(bottleneck[1])),
        "predictive_standardized_innovation": float(bottleneck[5]),
        "predictive_radius": int(bottleneck[6]),
        "predictive_bottleneck_component": bottleneck[7],
        "best_future_bridge_log_evidence": -float(best[0]),
        "best_future_anchor_session": int(best[2]),
        "best_future_added_session": int(best[3]),
        "future_bridge_candidate_count": len(ranked),
    }


def _score_future_bridges(
    *,
    evaluator: Any,
    candidates: Sequence[ActiveSessionPathMode],
    parameter_scales: np.ndarray,
    config: ActiveSessionLinkConfig,
    offset_jacobian: np.ndarray | None = None,
) -> tuple[list[ActiveSessionPathMode], dict[str, Any]]:
    """Attach one-step phase predictability without changing strict objectives."""

    scored: list[ActiveSessionPathMode] = []
    failures = 0
    diagnostics: list[dict[str, Any]] = []
    relinearized_indices: set[int] = set()
    if offset_jacobian is not None:
        relinearization_limit = max(
            min(int(config.predictive_relinearization_count), len(candidates)),
            0,
        )
        if relinearization_limit > 0:
            represented_paths: set[
                tuple[tuple[int | None, ...], tuple[int | None, ...]]
            ] = set()
            objective_ranked_indices = sorted(
                range(len(candidates)),
                key=lambda index: (
                    float(candidates[index].posterior_chi2),
                    -float(candidates[index].laplace_log_evidence),
                    index,
                ),
            )
            for candidate_index in objective_ranked_indices:
                path_key = (
                    candidates[candidate_index].session_integers,
                    candidates[candidate_index].component_labels,
                )
                if path_key in represented_paths:
                    continue
                represented_paths.add(path_key)
                relinearized_indices.add(candidate_index)
                if len(relinearized_indices) >= relinearization_limit:
                    break
            for candidate_index in objective_ranked_indices:
                if len(relinearized_indices) >= relinearization_limit:
                    break
                relinearized_indices.add(candidate_index)

    for candidate_index, mode in enumerate(candidates):
        try:
            component_count = len(
                {
                    (
                        ("component", int(label))
                        if label is not None
                        else ("singleton", session_index)
                    )
                    for session_index, label in enumerate(mode.component_labels)
                }
            )
            if component_count <= 1:
                predictive = {
                    "future_bridge_log_evidence": 0.0,
                    "predictive_deviance": float(mode.posterior_chi2),
                    "predictive_anchor_session": None,
                    "predictive_added_session": None,
                    "predictive_phase": None,
                    "predictive_sigma": None,
                    "predictive_standardized_innovation": None,
                    "predictive_radius": 0,
                    "future_bridge_candidate_count": 0,
                }
            else:
                if (
                    offset_jacobian is None
                    or candidate_index in relinearized_indices
                ):
                    profile, _, candidate_offset_jacobian = _finite_difference_geometry(
                        evaluator=evaluator,
                        values=np.asarray(mode.parameter_values, dtype=float),
                        parameter_scales=parameter_scales,
                        fraction=max(
                            float(config.finite_difference_fraction),
                            1.0e-12,
                        ),
                    )
                else:
                    profile = evaluator.evaluate_values(mode.parameter_values)
                    candidate_offset_jacobian = np.asarray(
                        offset_jacobian,
                        dtype=float,
                    )
                predictive = _strict_future_bridge_diagnostic(
                    profile=profile,
                    mode=mode,
                    offset_jacobian=candidate_offset_jacobian,
                    config=config,
                )
        except Exception:
            failures += 1
            predictive = {
                "future_bridge_log_evidence": -math.inf,
                "predictive_deviance": math.inf,
                "predictive_anchor_session": None,
                "predictive_added_session": None,
                "predictive_phase": None,
                "predictive_sigma": None,
                "predictive_standardized_innovation": None,
                "predictive_radius": 0,
                "future_bridge_candidate_count": 0,
            }
        scored.append(
            replace(
                mode,
                future_bridge_log_evidence=float(
                    predictive["future_bridge_log_evidence"]
                ),
                predictive_deviance=float(predictive["predictive_deviance"]),
                predictive_anchor_session=predictive["predictive_anchor_session"],
                predictive_added_session=predictive["predictive_added_session"],
            )
        )
        diagnostics.append({"candidate_index": candidate_index, **predictive})

    finite_scores = [
        float(mode.predictive_deviance)
        for mode in scored
        if np.isfinite(mode.predictive_deviance)
    ]
    return scored, {
        "predictive_scored_count": len(scored),
        "predictive_failure_count": failures,
        "predictive_relinearized_count": len(relinearized_indices),
        "predictive_linearization": (
            "candidate-specific"
            if offset_jacobian is None
            else "hybrid-candidate-specific"
            if relinearized_indices
            else "shared-stage"
        ),
        "best_predictive_deviance": min(finite_scores) if finite_scores else None,
        "candidate_predictive_scores": diagnostics,
    }


def _mixture_entropy(
    distributions: Sequence[dict[int, float]],
    weights: Sequence[float],
) -> tuple[float, int]:
    mixture: dict[int, float] = {}
    for distribution, mode_weight in zip(distributions, weights, strict=True):
        for integer, probability in distribution.items():
            mixture[integer] = mixture.get(integer, 0.0) + float(mode_weight) * probability
    total = sum(mixture.values())
    if total <= 0.0:
        return math.inf, 0
    probabilities = np.asarray([value / total for value in mixture.values()], dtype=float)
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-300))))
    return entropy, len(mixture)


def _normalized_mode_weights(modes: Sequence[ActiveSessionPathMode]) -> np.ndarray:
    log_weights = np.asarray([mode.laplace_log_evidence for mode in modes], dtype=float)
    finite = np.isfinite(log_weights)
    if not np.any(finite):
        return np.full(len(modes), 1.0 / max(len(modes), 1), dtype=float)
    floor = float(np.min(log_weights[finite])) - 100.0
    log_weights = np.where(finite, log_weights, floor)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(np.maximum(log_weights, -745.0))
    return weights / float(np.sum(weights))


def _select_active_edge(
    *,
    modes: Sequence[ActiveSessionPathMode],
    profiles: Sequence[Any],
    component_labels: Sequence[int | None],
    offset_jacobian: np.ndarray,
    scaled_covariance: np.ndarray,
    config: ActiveSessionLinkConfig,
) -> tuple[int, int, dict[str, Any]]:
    session_count = int(profiles[0].session_count)
    def component_key(session_index: int) -> tuple[str, int]:
        label = component_labels[session_index]
        return (
            ("component", int(label))
            if label is not None
            else ("singleton", int(session_index))
        )

    edges = (
        (left, right)
        for left in range(session_count)
        for right in range(left + 1, session_count)
        if component_key(left) != component_key(right)
    )

    mode_weights = _normalized_mode_weights(modes)
    ranked: list[
        tuple[float, float, float, int, int, int, float, float, float]
    ] = []
    for anchor, candidate in edges:
        jacobian = offset_jacobian[candidate] - offset_jacobian[anchor]
        measurement_variance = float(
            profiles[0].session_offset_variances[candidate]
            + profiles[0].session_offset_variances[anchor]
        )
        predictive_variance = max(
            measurement_variance + float(jacobian @ scaled_covariance @ jacobian),
            1.0e-18,
        )
        sigma = math.sqrt(predictive_variance)
        radius = min(
            max(
                int(config.initial_integer_radius),
                int(math.ceil(float(config.candidate_sigma) * sigma)),
            ),
            max(int(config.max_integer_radius), int(config.initial_integer_radius)),
        )
        distribution_evidence = [
            _discrete_robust_bridge(
                float(profile.session_offsets[candidate])
                - float(profile.session_offsets[anchor]),
                sigma,
                radius=radius,
                outlier_probability=float(config.bridge_outlier_probability),
                outlier_scale=float(config.bridge_outlier_scale),
            )
            for profile in profiles
        ]
        gaussian_evidence = [
            _discrete_gaussian(
                float(profile.session_offsets[candidate])
                - float(profile.session_offsets[anchor]),
                sigma,
                radius=radius,
            )[1]
            for profile in profiles
        ]
        distributions = [item[0] for item in distribution_evidence]
        mode_log_evidence = np.asarray(
            [item[1] for item in distribution_evidence],
            dtype=float,
        )
        mixture_log_terms = np.log(np.maximum(mode_weights, 1.0e-300)) + mode_log_evidence
        maximum_log_term = float(np.max(mixture_log_terms))
        bridge_log_evidence = maximum_log_term + math.log(
            float(np.sum(np.exp(mixture_log_terms - maximum_log_term)))
        )
        gaussian_log_terms = (
            np.log(np.maximum(mode_weights, 1.0e-300))
            + np.asarray(gaussian_evidence, dtype=float)
        )
        gaussian_maximum = float(np.max(gaussian_log_terms))
        gaussian_bridge_log_evidence = gaussian_maximum + math.log(
            float(np.sum(np.exp(gaussian_log_terms - gaussian_maximum)))
        )
        posterior_mode_weights = np.exp(mixture_log_terms - bridge_log_evidence)
        entropy, support = _mixture_entropy(
            distributions,
            posterior_mode_weights,
        )
        representative_mean = (
            float(profiles[0].session_offsets[candidate])
            - float(profiles[0].session_offsets[anchor])
        )
        standardized_innovation = abs(
            representative_mean - float(np.rint(representative_mean))
        ) / sigma
        ranked.append(
            (
                -bridge_log_evidence,
                entropy,
                predictive_variance,
                support,
                int(anchor),
                int(candidate),
                sigma,
                standardized_innovation,
                gaussian_bridge_log_evidence,
            )
        )
    if not ranked:
        raise RuntimeError("No active-linking edge is available.")
    ranked_edges: list[dict[str, Any]] = []
    for item in sorted(ranked, key=lambda value: value[:6]):
        anchor, candidate = int(item[4]), int(item[5])
        anchor_label = component_labels[anchor]
        candidate_label = component_labels[candidate]
        if anchor_label is None and candidate_label is not None:
            anchor, candidate = candidate, anchor
        elif anchor_label is None and candidate_label is None:
            left_variance = float(profiles[0].session_offset_variances[anchor])
            right_variance = float(profiles[0].session_offset_variances[candidate])
            if right_variance < left_variance:
                anchor, candidate = candidate, anchor
        ranked_edges.append(
            {
                "anchor_session": anchor,
                "added_session": candidate,
                "bridge_log_evidence": float(-item[0]),
                "gaussian_bridge_log_evidence": float(item[8]),
                "predictive_entropy": float(item[1]),
                "predictive_variance": float(item[2]),
                "predictive_sigma": float(item[6]),
                "standardized_integer_innovation": float(item[7]),
                "mixture_integer_support": int(item[3]),
            }
        )
    best_edge = ranked_edges[0]
    component_best_edges: dict[tuple[str, int], dict[str, Any]] = {}
    for edge in ranked_edges:
        for key in (
            component_key(int(edge["anchor_session"])),
            component_key(int(edge["added_session"])),
        ):
            current = component_best_edges.get(key)
            if current is None or float(edge["bridge_log_evidence"]) > float(
                current["bridge_log_evidence"]
            ):
                component_best_edges[key] = edge
    bottleneck_edges: list[dict[str, Any]] = []
    bottleneck_edge_keys: set[tuple[int, int]] = set()
    for edge in sorted(
        component_best_edges.values(),
        key=lambda value: (
            float(value["bridge_log_evidence"]),
            -float(value["predictive_variance"]),
            int(value["anchor_session"]),
            int(value["added_session"]),
        ),
    ):
        edge_key = (
            int(edge["anchor_session"]),
            int(edge["added_session"]),
        )
        if edge_key in bottleneck_edge_keys:
            continue
        bottleneck_edge_keys.add(edge_key)
        bottleneck_edges.append(edge)
    retained_ranked_edges = ranked_edges[: max(int(config.relocation_edge_count), 1)]
    return (
        int(best_edge["anchor_session"]),
        int(best_edge["added_session"]),
        {
            **best_edge,
            "edge_candidate_count": len(ranked),
            "ranked_edges": retained_ranked_edges,
            "bottleneck_edges": bottleneck_edges,
        },
    )


def _scaled_bounds(
    *,
    start: np.ndarray,
    scales: np.ndarray,
    bounds: Sequence[tuple[float | None, float | None]],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_values = np.full(start.size, -float(radius), dtype=float)
    upper_values = np.full(start.size, float(radius), dtype=float)
    for index, (lower, upper) in enumerate(bounds):
        if lower is not None:
            lower_values[index] = max(
                lower_values[index],
                (float(lower) - float(start[index])) / float(scales[index]),
            )
        if upper is not None:
            upper_values[index] = min(
                upper_values[index],
                (float(upper) - float(start[index])) / float(scales[index]),
            )
    return lower_values, upper_values


def _optimize_path_job(
    *,
    evaluator: Any,
    parent: ActiveSessionPathMode,
    parent_mode_index: int,
    session_integers: tuple[int | None, ...],
    component_labels: tuple[int | None, ...],
    linked_sessions: tuple[int, ...],
    anchor_session: int,
    added_session: int,
    edge_integer: int,
    predicted_edge_phase: float,
    predicted_edge_sigma: float,
    candidate_radius: int,
    parameter_scales: np.ndarray,
    prior_center: np.ndarray,
    prior_scales: np.ndarray,
    parameter_bounds: Sequence[tuple[float | None, float | None]],
    config: ActiveSessionLinkConfig,
) -> ActiveSessionPathMode | None:
    start = np.asarray(parent.parameter_values, dtype=float)
    weight = max(float(config.prior_weight), 0.0)
    initial_profile = evaluator.evaluate_values(start)
    shape_size = int(np.asarray(initial_profile.normalized_residuals).size)
    ambiguity_size = sum(
        len(group) - 1 for group in _component_groups(component_labels)
    )
    prior_size = start.size if weight > 0.0 else 0
    failure = np.full(shape_size + ambiguity_size + prior_size, 1.0e12, dtype=float)

    def residual_vector(scaled_offset: np.ndarray) -> np.ndarray:
        values = start + parameter_scales * np.asarray(scaled_offset, dtype=float)
        if not _parameter_domain_ok(values, parameter_bounds):
            return failure
        try:
            profile = evaluator.evaluate_values(values)
            shape, ambiguity, _ = _linked_profile_terms(
                profile,
                session_integers,
                component_labels=component_labels,
            )
            components = [shape, ambiguity]
            if weight > 0.0:
                components.append(
                    math.sqrt(weight) * (values - prior_center) / prior_scales
                )
            result = np.concatenate(components)
            return result if result.shape == failure.shape and np.all(np.isfinite(result)) else failure
        except Exception:
            return failure

    lower, upper = _scaled_bounds(
        start=start,
        scales=parameter_scales,
        bounds=parameter_bounds,
        radius=max(float(config.trust_radius), 1.0e-12),
    )
    if np.any(lower > upper):
        return None
    optimization = least_squares(
        residual_vector,
        np.zeros(start.size, dtype=float),
        method="trf",
        jac="2-point",
        bounds=(lower, upper),
        max_nfev=max(int(config.max_nfev), 1),
        ftol=1.0e-9,
        xtol=1.0e-9,
        gtol=1.0e-9,
    )
    values = start + parameter_scales * np.asarray(optimization.x, dtype=float)
    if not _parameter_domain_ok(values, parameter_bounds):
        return None
    profile = evaluator.evaluate_values(values)
    shape, ambiguity, _ = _linked_profile_terms(
        profile,
        session_integers,
        component_labels=component_labels,
    )
    shape_chi2 = float(shape @ shape)
    ambiguity_chi2 = float(ambiguity @ ambiguity)
    prior_chi2 = float(
        weight * np.sum(np.square((values - prior_center) / prior_scales))
    )
    posterior_chi2 = shape_chi2 + ambiguity_chi2 + prior_chi2
    jacobian = np.asarray(optimization.jac, dtype=float)
    precision = jacobian.T @ jacobian
    jitter = max(float(np.trace(precision)) * 1.0e-12, 1.0e-12)
    precision = precision + np.eye(precision.shape[0], dtype=float) * jitter
    sign, log_determinant = np.linalg.slogdet(precision)
    if sign <= 0 or not np.isfinite(log_determinant):
        log_determinant = math.inf
    covariance = np.linalg.pinv(precision)
    laplace_log_evidence = -0.5 * posterior_chi2 - 0.5 * float(log_determinant)
    return ActiveSessionPathMode(
        session_integers=session_integers,
        component_labels=component_labels,
        linked_sessions=linked_sessions,
        root_session=int(parent.root_session),
        parameter_values=values.copy(),
        shape_chi2=shape_chi2,
        ambiguity_chi2=ambiguity_chi2,
        prior_chi2=prior_chi2,
        posterior_chi2=posterior_chi2,
        laplace_log_evidence=laplace_log_evidence,
        future_bridge_log_evidence=0.0,
        predictive_deviance=posterior_chi2,
        predictive_anchor_session=None,
        predictive_added_session=None,
        scaled_covariance=covariance,
        source_mode_index=int(parent.source_mode_index),
        parent_mode_index=int(parent_mode_index),
        anchor_session=int(anchor_session),
        added_session=int(added_session),
        edge_integer=int(edge_integer),
        predicted_edge_phase=float(predicted_edge_phase),
        predicted_edge_sigma=float(predicted_edge_sigma),
        candidate_radius=int(candidate_radius),
        success=bool(optimization.success),
        nfev=int(optimization.nfev),
    )


def _select_mode_portfolio(
    candidates: Sequence[ActiveSessionPathMode],
    *,
    target: int,
    config: ActiveSessionLinkConfig,
    parameter_scales: np.ndarray | None = None,
) -> tuple[list[ActiveSessionPathMode], dict[str, Any]]:
    if not candidates:
        return [], {"candidate_count": 0, "selected_count": 0}

    scales = (
        np.ones_like(np.asarray(candidates[0].parameter_values, dtype=float))
        if parameter_scales is None
        else np.asarray(parameter_scales, dtype=float)
    )
    merge_radius = max(float(config.continuous_mode_merge_radius), 0.0)
    path_groups: dict[
        tuple[tuple[int | None, ...], tuple[int | None, ...]],
        list[ActiveSessionPathMode],
    ] = {}
    for mode in sorted(
        candidates,
        key=lambda item: (
            -float(item.laplace_log_evidence),
            float(item.posterior_chi2),
        ),
    ):
        path_key = (mode.session_integers, mode.component_labels)
        representatives = path_groups.setdefault(path_key, [])
        if any(
            float(
                np.linalg.norm(
                    (
                        np.asarray(mode.parameter_values, dtype=float)
                        - np.asarray(existing.parameter_values, dtype=float)
                    )
                    / scales
                )
            )
            <= merge_radius
            for existing in representatives
        ):
            continue
        representatives.append(mode)
    posterior_candidates = [
        mode for representatives in path_groups.values() for mode in representatives
    ]
    evidence_ranked = sorted(
        posterior_candidates,
        key=lambda mode: (
            -float(mode.laplace_log_evidence),
            float(mode.posterior_chi2),
            mode.session_integers,
            mode.source_mode_index,
        ),
    )
    objective_ranked = sorted(
        posterior_candidates,
        key=lambda mode: (
            float(mode.posterior_chi2),
            -float(mode.laplace_log_evidence),
            mode.session_integers,
            mode.source_mode_index,
        ),
    )
    predictive_ranked = sorted(
        posterior_candidates,
        key=lambda mode: (
            float(mode.predictive_deviance),
            float(mode.posterior_chi2),
            -float(mode.laplace_log_evidence),
            mode.session_integers,
            mode.source_mode_index,
        ),
    )
    path_counts: dict[tuple[int | None, ...], int] = {}
    selected: list[ActiveSessionPathMode] = []
    selected_ids: set[int] = set()

    def add(mode: ActiveSessionPathMode) -> bool:
        identity = id(mode)
        if identity in selected_ids:
            return False
        path_count = path_counts.get(mode.session_integers, 0)
        if path_count >= max(int(config.max_modes_per_path), 1):
            return False
        selected.append(mode)
        selected_ids.add(identity)
        path_counts[mode.session_integers] = path_count + 1
        return True

    objective_target = min(
        int(target),
        int(round(int(target) * min(max(float(config.objective_fraction), 0.0), 1.0))),
    )
    for mode in objective_ranked:
        if len(selected) >= objective_target:
            break
        add(mode)

    predictive_target = min(
        int(target),
        len(selected)
        + int(round(int(target) * min(max(float(config.predictive_fraction), 0.0), 1.0))),
    )
    for mode in predictive_ranked:
        if len(selected) >= predictive_target:
            break
        add(mode)

    topology_target = min(
        int(target),
        len(selected)
        + int(round(int(target) * min(max(float(config.topology_fraction), 0.0), 1.0))),
    )
    represented_topologies = {mode.component_labels for mode in selected}
    for mode in objective_ranked:
        if len(selected) >= topology_target:
            break
        if mode.component_labels in represented_topologies:
            continue
        if add(mode):
            represented_topologies.add(mode.component_labels)

    island_target = min(
        int(target),
        len(selected)
        + int(round(int(target) * min(max(float(config.island_fraction), 0.0), 1.0))),
    )
    represented = {mode.source_mode_index for mode in selected}
    for mode in evidence_ranked:
        if len(selected) >= island_target:
            break
        if mode.source_mode_index in represented:
            continue
        if add(mode):
            represented.add(mode.source_mode_index)

    for mode in evidence_ranked:
        if len(selected) >= int(target):
            break
        add(mode)

    log_weights = np.asarray(
        [float(mode.laplace_log_evidence) for mode in evidence_ranked],
        dtype=float,
    )
    log_weights -= float(np.max(log_weights))
    weights = np.exp(np.maximum(log_weights, -745.0))
    weights /= float(np.sum(weights))
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
    selected_identity = {id(mode) for mode in selected}
    selected_mass = float(
        sum(weight for weight, mode in zip(weights, evidence_ranked, strict=True) if id(mode) in selected_identity)
    )
    return selected, {
        "candidate_count": len(candidates),
        "posterior_candidate_count": len(posterior_candidates),
        "duplicate_candidate_count": len(candidates) - len(posterior_candidates),
        "selected_count": len(selected),
        "unique_candidate_path_count": len(
            {
                (mode.session_integers, mode.component_labels)
                for mode in posterior_candidates
            }
        ),
        "unique_selected_path_count": len(
            {(mode.session_integers, mode.component_labels) for mode in selected}
        ),
        "unique_candidate_topology_count": len(
            {mode.component_labels for mode in posterior_candidates}
        ),
        "unique_selected_topology_count": len(
            {mode.component_labels for mode in selected}
        ),
        "selected_source_mode_count": len({mode.source_mode_index for mode in selected}),
        "predictive_selected_count": sum(
            id(mode) in selected_identity
            for mode in predictive_ranked[:predictive_target]
        ),
        "best_candidate_predictive_deviance": (
            float(predictive_ranked[0].predictive_deviance)
            if predictive_ranked
            else None
        ),
        "best_selected_predictive_deviance": (
            float(min(mode.predictive_deviance for mode in selected))
            if selected
            else None
        ),
        "posterior_entropy": entropy,
        "effective_mode_count": float(math.exp(min(entropy, 700.0))),
        "selected_posterior_mass": selected_mass,
        "tail_mass": max(1.0 - selected_mass, 0.0),
    }


def select_active_session_resume_portfolio(
    *,
    modes: Sequence[ActiveSessionPathMode],
    max_modes: int,
    config: ActiveSessionLinkConfig,
    parameter_scales: Sequence[float],
) -> tuple[tuple[ActiveSessionPathMode, ...], dict[str, Any]]:
    """Merge checkpoint forests with the normal Bayesian path portfolio."""

    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    pool = list(modes)
    if not pool:
        raise ValueError("At least one checkpoint mode is required.")
    selected, diagnostics = _select_mode_portfolio(
        pool,
        target=min(max(int(max_modes), 1), len(pool)),
        config=config,
        parameter_scales=scales,
    )
    return tuple(selected), {
        "proposal_source": "apollo-checkpoint-posterior-forest-portfolio",
        **diagnostics,
    }


def _connect_session_components(
    session_integers: Sequence[int | None],
    component_labels: Sequence[int | None],
    *,
    anchor_session: int,
    added_session: int,
    edge_integer: int,
) -> tuple[tuple[int | None, ...], tuple[int | None, ...], tuple[int, ...]] | None:
    """Return a mode after imposing one integer edge between components."""

    integers = list(session_integers)
    labels = list(component_labels)
    anchor = int(anchor_session)
    added = int(added_session)
    anchor_label = labels[anchor]
    added_label = labels[added]
    if anchor_label is None and added_label is None:
        merged_label = min(anchor, added)
        labels[anchor] = merged_label
        labels[added] = merged_label
        integers[anchor] = 0
        integers[added] = int(edge_integer)
    elif anchor_label is not None and added_label is None:
        labels[added] = int(anchor_label)
        integers[added] = int(integers[anchor]) + int(edge_integer)
    elif anchor_label is None and added_label is not None:
        labels[anchor] = int(added_label)
        integers[anchor] = int(integers[added]) - int(edge_integer)
    elif int(anchor_label) != int(added_label):
        shift = (
            int(integers[anchor])
            + int(edge_integer)
            - int(integers[added])
        )
        for session_index, label in enumerate(labels):
            if label == added_label:
                labels[session_index] = int(anchor_label)
                integers[session_index] = int(integers[session_index]) + shift
    else:
        return None
    linked_after = tuple(
        index for index, label in enumerate(labels) if label is not None
    )
    return tuple(integers), tuple(labels), linked_after


def _split_phase_component(
    *,
    session_integers: Sequence[int | None],
    component_labels: Sequence[int | None],
    left_sessions: Sequence[int],
    right_sessions: Sequence[int],
) -> tuple[tuple[int | None, ...], tuple[int | None, ...], tuple[int, ...]] | None:
    """Split one connected phase component into two conditioned blocks."""

    left = tuple(sorted({int(value) for value in left_sessions}))
    right = tuple(sorted({int(value) for value in right_sessions}))
    if not left or not right or set(left) & set(right):
        return None
    label = component_labels[left[0]]
    if label is None or any(component_labels[index] != label for index in (*left, *right)):
        return None
    component = {
        index
        for index, candidate_label in enumerate(component_labels)
        if candidate_label == label
    }
    if set(left) | set(right) != component:
        return None
    if any(session_integers[index] is None for index in component):
        return None
    labels = list(component_labels)
    left_label = min(left)
    right_label = min(right)
    for index in left:
        labels[index] = left_label
    for index in right:
        labels[index] = right_label
    linked_sessions = tuple(
        index for index, candidate_label in enumerate(labels) if candidate_label is not None
    )
    return tuple(session_integers), tuple(labels), linked_sessions


def _ordered_phase_block_splits(
    mode: ActiveSessionPathMode,
    *,
    anchor_session: int,
    added_session: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate nonlocal phase-block moves across the posterior forest."""

    del anchor_session, added_session
    relevant_labels = {
        int(label) for label in mode.component_labels if label is not None
    }
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for label in sorted(relevant_labels):
        sessions = tuple(
            index
            for index, candidate_label in enumerate(mode.component_labels)
            if candidate_label == label
        )
        for split_index in range(1, len(sessions)):
            splits.append((sessions[:split_index], sessions[split_index:]))
    return splits


def _generate_phase_split_merge_candidates(
    *,
    evaluator: Any,
    modes: Sequence[ActiveSessionPathMode],
    baseline_candidates: Sequence[ActiveSessionPathMode],
    parameter_scales: np.ndarray,
    prior_center: np.ndarray,
    prior_scales: np.ndarray,
    parameter_bounds: Sequence[tuple[float | None, float | None]],
    config: ActiveSessionLinkConfig,
) -> tuple[list[ActiveSessionPathMode], dict[int, dict[str, Any]], dict[str, Any]]:
    """Rejuvenate a failed bridge by resampling an earlier phase block.

    A high bridge chi-square can be caused by an old integer decision rather
    than by the bridge itself.  This move temporarily splits one of the two
    conditioned components, fits the new bridge without that old relation,
    and then samples the removed block offset from the updated posterior.
    """

    ranked_baselines = sorted(
        (
            candidate
            for candidate in baseline_candidates
            if candidate.anchor_session is not None
            and candidate.added_session is not None
            and candidate.edge_integer is not None
            and 0 <= int(candidate.parent_mode_index) < len(modes)
        ),
        key=lambda candidate: (
            float(candidate.posterior_chi2),
            -float(candidate.laplace_log_evidence),
            int(candidate.parent_mode_index),
            int(candidate.anchor_session),
            int(candidate.added_session),
            int(candidate.edge_integer),
        ),
    )
    baseline_limit = max(int(config.backward_restart_count), 1)
    selected_baselines: list[ActiveSessionPathMode] = []
    represented_parents: set[int] = set()
    for candidate in ranked_baselines:
        parent_index = int(candidate.parent_mode_index)
        if parent_index in represented_parents:
            continue
        selected_baselines.append(candidate)
        represented_parents.add(parent_index)
        if len(selected_baselines) >= baseline_limit:
            break
    if len(selected_baselines) < baseline_limit:
        selected_ids = {id(candidate) for candidate in selected_baselines}
        for candidate in ranked_baselines:
            if id(candidate) in selected_ids:
                continue
            selected_baselines.append(candidate)
            selected_ids.add(id(candidate))
            if len(selected_baselines) >= baseline_limit:
                break
    if not selected_baselines:
        return [], {}, {"phase_split_partial_job_count": 0}

    optimization_config = replace(
        config,
        trust_radius=max(float(config.trust_radius), float(config.relocation_trust_radius)),
        max_nfev=max(int(config.max_nfev), int(config.relocation_max_nfev)),
    )
    partial_records: list[
        tuple[
            ActiveSessionPathMode,
            int,
            ActiveSessionPathMode,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []
    failed_count = 0
    for baseline_index, baseline in enumerate(selected_baselines):
        parent_index = int(baseline.parent_mode_index)
        parent = modes[parent_index]
        bridge_anchor = int(baseline.anchor_session)
        bridge_added = int(baseline.added_session)
        for left, right in _ordered_phase_block_splits(
            parent,
            anchor_session=bridge_anchor,
            added_session=bridge_added,
        ):
            cut = _split_phase_component(
                session_integers=parent.session_integers,
                component_labels=parent.component_labels,
                left_sessions=left,
                right_sessions=right,
            )
            if cut is None:
                failed_count += 1
                continue
            cut_integers, cut_labels, cut_linked = cut
            cut_parent = replace(
                parent,
                session_integers=cut_integers,
                component_labels=cut_labels,
                linked_sessions=cut_linked,
                root_session=min(cut_linked),
                parameter_values=np.asarray(baseline.parameter_values, dtype=float).copy(),
            )
            connected = _connect_session_components(
                cut_parent.session_integers,
                cut_parent.component_labels,
                anchor_session=bridge_anchor,
                added_session=bridge_added,
                edge_integer=int(baseline.edge_integer),
            )
            if connected is None:
                failed_count += 1
                continue
            integers, labels, linked_after = connected
            try:
                partial = _optimize_path_job(
                    evaluator=evaluator,
                    parent=replace(cut_parent, root_session=min(linked_after)),
                    parent_mode_index=parent_index,
                    session_integers=integers,
                    component_labels=labels,
                    linked_sessions=linked_after,
                    anchor_session=bridge_anchor,
                    added_session=bridge_added,
                    edge_integer=int(baseline.edge_integer),
                    predicted_edge_phase=float(baseline.predicted_edge_phase),
                    predicted_edge_sigma=max(float(baseline.predicted_edge_sigma), 1.0e-12),
                    candidate_radius=int(baseline.candidate_radius),
                    parameter_scales=parameter_scales,
                    prior_center=prior_center,
                    prior_scales=prior_scales,
                    parameter_bounds=parameter_bounds,
                    config=optimization_config,
                )
            except Exception:
                partial = None
            if partial is None:
                failed_count += 1
                continue
            partial_records.append((partial, baseline_index, baseline, left, right))

    ranked_partials = sorted(
        partial_records,
        key=lambda item: (
            float(item[0].posterior_chi2),
            -float(item[0].laplace_log_evidence),
            item[1],
            item[3],
        ),
    )
    partial_limit = min(
        len(ranked_partials),
        max(
            int(config.backward_edge_count),
            len(selected_baselines),
            1,
        ),
    )
    selected_partials: list[
        tuple[
            ActiveSessionPathMode,
            int,
            ActiveSessionPathMode,
            tuple[int, ...],
            tuple[int, ...],
        ]
    ] = []
    represented_baselines: set[int] = set()
    for record in ranked_partials:
        if record[1] in represented_baselines:
            continue
        selected_partials.append(record)
        represented_baselines.add(record[1])
        if len(selected_partials) >= partial_limit:
            break
    selected_partial_ids = {id(record[0]) for record in selected_partials}
    for record in ranked_partials:
        if len(selected_partials) >= partial_limit:
            break
        if id(record[0]) in selected_partial_ids:
            continue
        selected_partials.append(record)
        selected_partial_ids.add(id(record[0]))

    reconnection_jobs: list[tuple[float, int, int, float, float]] = []
    integer_radius = max(int(config.backward_integer_radius), 1)
    for partial_index, (partial, _, _, left, right) in enumerate(selected_partials):
        profile = evaluator.evaluate_values(partial.parameter_values)
        reconnect_anchor = int(left[-1])
        reconnect_added = int(right[0])
        predicted = float(
            profile.session_offsets[reconnect_added]
            - profile.session_offsets[reconnect_anchor]
        )
        variance = max(
            float(profile.session_offset_variances[reconnect_added])
            + float(profile.session_offset_variances[reconnect_anchor]),
            1.0e-18,
        )
        sigma = math.sqrt(variance)
        nearest = int(np.rint(predicted))
        for edge_integer in range(nearest - integer_radius, nearest + integer_radius + 1):
            log_weight = -0.5 * (float(edge_integer) - predicted) ** 2 / variance
            reconnection_jobs.append(
                (log_weight, partial_index, int(edge_integer), predicted, sigma)
            )
    job_limit = max(int(config.backward_max_jobs), 1)
    selected_jobs: list[tuple[float, int, int, float, float]] = []
    jobs_by_partial: dict[int, list[tuple[float, int, int, float, float]]] = {}
    for job in reconnection_jobs:
        jobs_by_partial.setdefault(job[1], []).append(job)
    for jobs in jobs_by_partial.values():
        jobs.sort(key=lambda item: (-item[0], abs(item[2]), item[2]))
    selected_job_keys: set[tuple[int, int]] = set()
    for integer_rank in range(2 * integer_radius + 1):
        for partial_index in sorted(jobs_by_partial):
            partial_jobs = jobs_by_partial[partial_index]
            if integer_rank >= len(partial_jobs):
                continue
            job = partial_jobs[integer_rank]
            selected_jobs.append(job)
            selected_job_keys.add((job[1], job[2]))
            if len(selected_jobs) >= job_limit:
                break
        if len(selected_jobs) >= job_limit:
            break

    candidates: list[ActiveSessionPathMode] = []
    provenance: dict[int, dict[str, Any]] = {}
    for _, partial_index, edge_integer, predicted, sigma in selected_jobs:
        partial, _, baseline, left, right = selected_partials[partial_index]
        reconnect_anchor = int(left[-1])
        reconnect_added = int(right[0])
        reconnected = _connect_session_components(
            partial.session_integers,
            partial.component_labels,
            anchor_session=reconnect_anchor,
            added_session=reconnect_added,
            edge_integer=edge_integer,
        )
        if reconnected is None:
            failed_count += 1
            continue
        integers, labels, linked_after = reconnected
        try:
            candidate = _optimize_path_job(
                evaluator=evaluator,
                parent=replace(partial, root_session=min(linked_after)),
                parent_mode_index=int(baseline.parent_mode_index),
                session_integers=integers,
                component_labels=labels,
                linked_sessions=linked_after,
                anchor_session=reconnect_anchor,
                added_session=reconnect_added,
                edge_integer=edge_integer,
                predicted_edge_phase=predicted,
                predicted_edge_sigma=sigma,
                candidate_radius=integer_radius,
                parameter_scales=parameter_scales,
                prior_center=prior_center,
                prior_scales=prior_scales,
                parameter_bounds=parameter_bounds,
                config=optimization_config,
            )
        except Exception:
            candidate = None
        if candidate is None:
            failed_count += 1
            continue
        candidate = replace(
            candidate,
            anchor_session=int(baseline.anchor_session),
            added_session=int(baseline.added_session),
            edge_integer=int(baseline.edge_integer),
            predicted_edge_phase=float(baseline.predicted_edge_phase),
            predicted_edge_sigma=float(baseline.predicted_edge_sigma),
            candidate_radius=int(baseline.candidate_radius),
        )
        candidates.append(candidate)
        original_integer = int(baseline.session_integers[reconnect_added]) - int(
            baseline.session_integers[reconnect_anchor]
        )
        provenance[id(candidate)] = {
            "split_left_sessions": left,
            "split_right_sessions": right,
            "reconnect_edge": (reconnect_anchor, reconnect_added),
            "original_reconnect_integer": original_integer,
            "resampled_reconnect_integer": int(edge_integer),
            "integer_delta": int(edge_integer) - original_integer,
            "bridge_edge": (
                int(baseline.anchor_session),
                int(baseline.added_session),
            ),
        }
    return candidates, provenance, {
        "phase_split_baseline_count": len(selected_baselines),
        "phase_split_partial_job_count": len(partial_records),
        "phase_split_selected_partial_count": len(selected_partials),
        "phase_split_reconnection_job_count": len(selected_jobs),
        "phase_split_candidate_count": len(candidates),
        "phase_split_failed_job_count": failed_count,
        "phase_split_best_partial_chi2": (
            float(ranked_partials[0][0].posterior_chi2)
            if ranked_partials
            else None
        ),
        "phase_split_selected_blocks": [
            {
                "left_sessions": record[3],
                "right_sessions": record[4],
                "partial_chi2": float(record[0].posterior_chi2),
                "bridge_edge": (
                    int(record[2].anchor_session),
                    int(record[2].added_session),
                ),
            }
            for record in selected_partials
        ],
    }


def _generate_phase_block_vector_candidates(
    *,
    evaluator: Any,
    modes: Sequence[ActiveSessionPathMode],
    baseline_candidates: Sequence[ActiveSessionPathMode],
    restart_values: Sequence[np.ndarray],
    parameter_scales: np.ndarray,
    prior_center: np.ndarray,
    prior_scales: np.ndarray,
    parameter_bounds: Sequence[tuple[float | None, float | None]],
    config: ActiveSessionLinkConfig,
) -> tuple[list[ActiveSessionPathMode], dict[int, dict[str, Any]], dict[str, Any]]:
    """Jointly resample a small vector of conditioned phase-block offsets."""

    ranked_baselines = sorted(
        (
            candidate
            for candidate in baseline_candidates
            if candidate.anchor_session is not None
            and candidate.added_session is not None
            and candidate.edge_integer is not None
            and 0 <= int(candidate.parent_mode_index) < len(modes)
        ),
        key=lambda candidate: (
            float(candidate.posterior_chi2),
            -float(candidate.laplace_log_evidence),
            int(candidate.parent_mode_index),
            int(candidate.anchor_session),
            int(candidate.added_session),
            int(candidate.edge_integer),
        ),
    )
    baseline_limit = max(int(config.backward_restart_count), 1)
    selected_baselines: list[ActiveSessionPathMode] = []
    represented_parents: set[int] = set()
    for candidate in ranked_baselines:
        parent_index = int(candidate.parent_mode_index)
        if parent_index in represented_parents:
            continue
        selected_baselines.append(candidate)
        represented_parents.add(parent_index)
        if len(selected_baselines) >= baseline_limit:
            break
    if not selected_baselines:
        return [], {}, {"phase_vector_screening_job_count": 0}

    optimization_config = replace(
        config,
        trust_radius=max(float(config.trust_radius), float(config.relocation_trust_radius)),
        max_nfev=max(int(config.max_nfev), int(config.relocation_max_nfev)),
    )
    failed_count = 0

    def tree_edges(parent: ActiveSessionPathMode) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        for group in _component_groups(parent.component_labels):
            result.extend(zip(group[:-1], group[1:], strict=True))
        return tuple(result)

    def optimize_cutset(
        baseline_index: int,
        cut_edges: tuple[tuple[int, int], ...],
        start_values: np.ndarray | None = None,
    ) -> ActiveSessionPathMode | None:
        nonlocal failed_count
        baseline = selected_baselines[baseline_index]
        parent_index = int(baseline.parent_mode_index)
        parent = modes[parent_index]
        cut = _cut_historical_edges(
            session_integers=parent.session_integers,
            component_labels=parent.component_labels,
            tree_edges=tree_edges(parent),
            edges=cut_edges,
        )
        if cut is None:
            failed_count += 1
            return None
        cut_integers, cut_labels, cut_linked = cut
        cut_parent = replace(
            parent,
            session_integers=cut_integers,
            component_labels=cut_labels,
            linked_sessions=cut_linked,
            root_session=min(cut_linked),
            parameter_values=np.asarray(
                baseline.parameter_values if start_values is None else start_values,
                dtype=float,
            ).copy(),
        )
        connected = _connect_session_components(
            cut_parent.session_integers,
            cut_parent.component_labels,
            anchor_session=int(baseline.anchor_session),
            added_session=int(baseline.added_session),
            edge_integer=int(baseline.edge_integer),
        )
        if connected is None:
            failed_count += 1
            return None
        integers, labels, linked_after = connected
        try:
            return _optimize_path_job(
                evaluator=evaluator,
                parent=replace(cut_parent, root_session=min(linked_after)),
                parent_mode_index=parent_index,
                session_integers=integers,
                component_labels=labels,
                linked_sessions=linked_after,
                anchor_session=int(baseline.anchor_session),
                added_session=int(baseline.added_session),
                edge_integer=int(baseline.edge_integer),
                predicted_edge_phase=float(baseline.predicted_edge_phase),
                predicted_edge_sigma=max(float(baseline.predicted_edge_sigma), 1.0e-12),
                candidate_radius=int(baseline.candidate_radius),
                parameter_scales=parameter_scales,
                prior_center=prior_center,
                prior_scales=prior_scales,
                parameter_bounds=parameter_bounds,
                config=optimization_config,
            )
        except Exception:
            failed_count += 1
            return None

    screening_records: list[
        tuple[ActiveSessionPathMode, int, tuple[int, int]]
    ] = []
    screening_by_baseline: dict[
        int, list[tuple[ActiveSessionPathMode, tuple[int, int]]]
    ] = {}
    for baseline_index, baseline in enumerate(selected_baselines):
        parent = modes[int(baseline.parent_mode_index)]
        for edge in tree_edges(parent):
            partial = optimize_cutset(baseline_index, (edge,))
            if partial is None:
                continue
            screening_records.append((partial, baseline_index, edge))
            screening_by_baseline.setdefault(baseline_index, []).append((partial, edge))

    min_cut_size = max(int(config.backward_min_cut_size), 1)
    max_cut_size = max(int(config.backward_max_cut_size), min_cut_size)
    pool_size = max(int(config.backward_cutset_pool_size), max_cut_size)
    partial_records: list[
        tuple[
            ActiveSessionPathMode,
            int,
            ActiveSessionPathMode,
            tuple[tuple[int, int], ...],
        ]
    ] = []
    cutset_job_count = 0
    for baseline_index, baseline in enumerate(selected_baselines):
        screened = sorted(
            screening_by_baseline.get(baseline_index, ()),
            key=lambda item: (
                float(item[0].posterior_chi2),
                -float(item[0].laplace_log_evidence),
                item[1],
            ),
        )
        pool = tuple(item[1] for item in screened[:pool_size])
        single_modes = {item[1]: item[0] for item in screened}
        restart_pool = [
            np.asarray(values, dtype=float)
            for values in restart_values
            if np.asarray(values, dtype=float).shape == parameter_scales.shape
            and _parameter_domain_ok(np.asarray(values, dtype=float), parameter_bounds)
        ]
        selected_restarts: list[np.ndarray] = []
        restart_target = max(int(config.backward_restart_count), 0)
        reference_values = [np.asarray(baseline.parameter_values, dtype=float)]
        while restart_pool and len(selected_restarts) < restart_target:
            restart_index = max(
                range(len(restart_pool)),
                key=lambda index: min(
                    float(
                        np.linalg.norm(
                            (restart_pool[index] - reference) / parameter_scales
                        )
                    )
                    for reference in (*reference_values, *selected_restarts)
                ),
            )
            restart = restart_pool.pop(restart_index)
            if any(
                float(np.linalg.norm((restart - existing) / parameter_scales))
                <= max(float(config.continuous_mode_merge_radius), 1.0e-6)
                for existing in (*reference_values, *selected_restarts)
            ):
                continue
            selected_restarts.append(restart)
        cutset_starts = (*reference_values, *selected_restarts)
        for cut_size in range(min_cut_size, min(max_cut_size, len(pool)) + 1):
            for cutset_like in combinations(pool, cut_size):
                cutset = tuple(sorted(cutset_like))
                for restart_index, start_values in enumerate(cutset_starts):
                    cutset_job_count += 1
                    partial = (
                        single_modes.get(cutset[0])
                        if cut_size == 1 and restart_index == 0
                        else optimize_cutset(
                            baseline_index,
                            cutset,
                            start_values=start_values,
                        )
                    )
                    if partial is not None:
                        partial_records.append(
                            (partial, baseline_index, baseline, cutset)
                        )

    ranked_partials = sorted(
        partial_records,
        key=lambda item: (
            float(item[0].posterior_chi2),
            -len(item[3]),
            -float(item[0].laplace_log_evidence),
            item[1],
            item[3],
        ),
    )
    partial_limit = min(
        len(ranked_partials),
        max(int(config.backward_edge_count), len(selected_baselines), 1),
    )
    selected_partials: list[
        tuple[
            ActiveSessionPathMode,
            int,
            ActiveSessionPathMode,
            tuple[tuple[int, int], ...],
        ]
    ] = []
    represented_baselines: set[int] = set()
    for record in ranked_partials:
        if record[1] in represented_baselines:
            continue
        selected_partials.append(record)
        represented_baselines.add(record[1])
        if len(selected_partials) >= partial_limit:
            break
    selected_ids = {id(record[0]) for record in selected_partials}
    for record in ranked_partials:
        if len(selected_partials) >= partial_limit:
            break
        if id(record[0]) in selected_ids:
            continue
        selected_partials.append(record)
        selected_ids.add(id(record[0]))

    integer_radius = max(int(config.backward_integer_radius), 1)
    vector_proposal_diagnostics: list[dict[str, Any]] = []
    vector_jobs: list[
        tuple[
            float,
            int,
            tuple[int, ...],
            tuple[float, ...],
            tuple[float, ...],
        ]
    ] = []
    for partial_index, (partial, _, baseline, cutset) in enumerate(selected_partials):
        profile, _, offset_jacobian = _finite_difference_geometry(
            evaluator=evaluator,
            values=np.asarray(partial.parameter_values, dtype=float),
            parameter_scales=parameter_scales,
            fraction=max(float(config.finite_difference_fraction), 1.0e-12),
        )
        proposal_axes: list[tuple[int, ...]] = []
        predictions: list[float] = []
        sigmas: list[float] = []
        variances: list[float] = []
        radii: list[int] = []
        for reconnect_anchor, reconnect_added in reversed(cutset):
            predicted = float(
                profile.session_offsets[reconnect_added]
                - profile.session_offsets[reconnect_anchor]
            )
            measurement_variance = max(
                float(profile.session_offset_variances[reconnect_added])
                + float(profile.session_offset_variances[reconnect_anchor]),
                1.0e-18,
            )
            jacobian = (
                np.asarray(offset_jacobian[reconnect_added], dtype=float)
                - np.asarray(offset_jacobian[reconnect_anchor], dtype=float)
            )
            variance = max(
                measurement_variance
                + float(jacobian @ partial.scaled_covariance @ jacobian),
                measurement_variance,
            )
            sigma = math.sqrt(variance)
            radius = min(
                max(
                    integer_radius,
                    int(math.ceil(float(config.candidate_sigma) * sigma)),
                ),
                max(int(config.max_integer_radius), integer_radius),
            )
            nearest = int(np.rint(predicted))
            original_integer = int(
                baseline.session_integers[reconnect_added]
            ) - int(baseline.session_integers[reconnect_anchor])
            proposal_axes.append(
                tuple(
                    sorted(
                        {
                            *range(
                                nearest - radius,
                                nearest + radius + 1,
                            ),
                            original_integer,
                        }
                    )
                )
            )
            predictions.append(predicted)
            variances.append(variance)
            sigmas.append(sigma)
            radii.append(radius)
        vector_proposal_diagnostics.append(
            {
                "partial_index": partial_index,
                "cutset": cutset,
                "predictive_sigmas": tuple(sigmas),
                "integer_radii": tuple(radii),
                "integer_support_size": int(
                    np.prod([len(axis) for axis in proposal_axes], dtype=np.int64)
                ),
            }
        )
        for integer_vector_like in product(*proposal_axes):
            integer_vector = tuple(int(value) for value in integer_vector_like)
            log_weight = -0.5 * sum(
                (float(value) - predicted) ** 2 / variance
                for value, predicted, variance in zip(
                    integer_vector,
                    predictions,
                    variances,
                    strict=True,
                )
            )
            vector_jobs.append(
                (
                    log_weight,
                    partial_index,
                    integer_vector,
                    tuple(predictions),
                    tuple(sigmas),
                )
            )

    ranked_vector_jobs = sorted(
        vector_jobs,
        key=lambda item: (-item[0], item[1], item[2]),
    )
    job_limit = max(int(config.backward_max_jobs), 1)
    selected_vector_jobs: list[
        tuple[
            float,
            int,
            tuple[int, ...],
            tuple[float, ...],
            tuple[float, ...],
        ]
    ] = []
    represented_partials: set[int] = set()
    for job in ranked_vector_jobs:
        if job[1] in represented_partials:
            continue
        selected_vector_jobs.append(job)
        represented_partials.add(job[1])
        if len(selected_vector_jobs) >= job_limit:
            break
    selected_job_keys = {(job[1], job[2]) for job in selected_vector_jobs}
    for job in ranked_vector_jobs:
        if len(selected_vector_jobs) >= job_limit:
            break
        key = (job[1], job[2])
        if key in selected_job_keys:
            continue
        selected_vector_jobs.append(job)
        selected_job_keys.add(key)

    candidates: list[ActiveSessionPathMode] = []
    provenance: dict[int, dict[str, Any]] = {}
    for _, partial_index, integer_vector, predictions, sigmas in selected_vector_jobs:
        partial, _, baseline, cutset = selected_partials[partial_index]
        integers = partial.session_integers
        labels = partial.component_labels
        linked_after = partial.linked_sessions
        edits: list[dict[str, Any]] = []
        valid = True
        last_anchor = last_added = 0
        for (
            (reconnect_anchor, reconnect_added),
            edge_integer,
        ) in zip(reversed(cutset), integer_vector, strict=True):
            connected = _connect_session_components(
                integers,
                labels,
                anchor_session=reconnect_anchor,
                added_session=reconnect_added,
                edge_integer=edge_integer,
            )
            if connected is None:
                valid = False
                failed_count += 1
                break
            integers, labels, linked_after = connected
            original_integer = int(
                baseline.session_integers[reconnect_added]
            ) - int(baseline.session_integers[reconnect_anchor])
            edits.append(
                {
                    "reconnect_edge": (reconnect_anchor, reconnect_added),
                    "original_integer": original_integer,
                    "resampled_integer": int(edge_integer),
                    "integer_delta": int(edge_integer) - original_integer,
                }
            )
            last_anchor, last_added = reconnect_anchor, reconnect_added
        if not valid:
            continue
        try:
            candidate = _optimize_path_job(
                evaluator=evaluator,
                parent=replace(partial, root_session=min(linked_after)),
                parent_mode_index=int(baseline.parent_mode_index),
                session_integers=integers,
                component_labels=labels,
                linked_sessions=linked_after,
                anchor_session=last_anchor,
                added_session=last_added,
                edge_integer=int(integer_vector[-1]),
                predicted_edge_phase=float(predictions[-1]),
                predicted_edge_sigma=float(sigmas[-1]),
                candidate_radius=integer_radius,
                parameter_scales=parameter_scales,
                prior_center=prior_center,
                prior_scales=prior_scales,
                parameter_bounds=parameter_bounds,
                config=optimization_config,
            )
        except Exception:
            candidate = None
        if candidate is None:
            failed_count += 1
            continue
        candidate = replace(
            candidate,
            anchor_session=int(baseline.anchor_session),
            added_session=int(baseline.added_session),
            edge_integer=int(baseline.edge_integer),
            predicted_edge_phase=float(baseline.predicted_edge_phase),
            predicted_edge_sigma=float(baseline.predicted_edge_sigma),
            candidate_radius=int(baseline.candidate_radius),
        )
        candidates.append(candidate)
        provenance[id(candidate)] = {
            "cutset": cutset,
            "historical_edits": tuple(edits),
            "bridge_edge": (
                int(baseline.anchor_session),
                int(baseline.added_session),
            ),
        }

    return candidates, provenance, {
        "phase_vector_baseline_count": len(selected_baselines),
        "phase_vector_restart_bank_count": len(restart_values),
        "phase_vector_screening_job_count": len(screening_records),
        "phase_vector_cutset_job_count": cutset_job_count,
        "phase_vector_partial_candidate_count": len(partial_records),
        "phase_vector_selected_partial_count": len(selected_partials),
        "phase_vector_reconnection_job_count": len(selected_vector_jobs),
        "phase_vector_reconnection_strategy": "joint-integer-grid",
        "phase_vector_integer_vector_count": len(vector_jobs),
        "phase_vector_proposals": vector_proposal_diagnostics,
        "phase_vector_candidate_count": len(candidates),
        "phase_vector_failed_job_count": failed_count,
        "phase_vector_best_partial_chi2": (
            float(ranked_partials[0][0].posterior_chi2)
            if ranked_partials
            else None
        ),
        "phase_vector_selected_cutsets": [
            {
                "cutset": record[3],
                "partial_chi2": float(record[0].posterior_chi2),
                "bridge_edge": (
                    int(record[2].anchor_session),
                    int(record[2].added_session),
                ),
            }
            for record in selected_partials
        ],
    }


def _cut_historical_edges(
    *,
    session_integers: Sequence[int | None],
    component_labels: Sequence[int | None],
    tree_edges: Sequence[tuple[int, int]],
    edges: Sequence[tuple[int, int]],
) -> tuple[tuple[int | None, ...], tuple[int | None, ...], tuple[int, ...]] | None:
    """Remove committed edges and reconstruct the resulting phase forest."""

    cut_edges = {
        frozenset((int(anchor), int(added))) for anchor, added in edges
    }
    if not cut_edges:
        return None
    adjacency: list[list[int]] = [list() for _ in session_integers]
    for left, right in tree_edges:
        if frozenset((int(left), int(right))) in cut_edges:
            continue
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    labels: list[int | None] = [None for _ in component_labels]
    unvisited = {
        index for index, label in enumerate(component_labels) if label is not None
    }
    while unvisited:
        start = min(unvisited)
        component: set[int] = set()
        pending = [start]
        while pending:
            session = pending.pop()
            if session in component:
                continue
            component.add(session)
            pending.extend(adjacency[session])
        component &= unvisited
        if not component:
            return None
        label = min(component)
        for session in component:
            if session_integers[session] is None:
                return None
            labels[session] = label
        unvisited -= component
    linked_sessions = tuple(
        index for index, label in enumerate(labels) if label is not None
    )
    return tuple(session_integers), tuple(labels), linked_sessions


def _cut_historical_edge(
    *,
    session_integers: Sequence[int | None],
    component_labels: Sequence[int | None],
    tree_edges: Sequence[tuple[int, int]],
    edge: tuple[int, int],
) -> tuple[tuple[int | None, ...], tuple[int | None, ...], tuple[int, ...]] | None:
    return _cut_historical_edges(
        session_integers=session_integers,
        component_labels=component_labels,
        tree_edges=tree_edges,
        edges=(edge,),
    )


def _historical_rejuvenation_edges(
    *,
    stage_records: Sequence[Mapping[str, Any]],
    component_labels: Sequence[int | None],
    anchor_session: int,
    added_session: int,
    edge_count: int,
) -> list[tuple[int, tuple[int, int], dict[str, Any]]]:
    """Rank committed edges that can explain a later bridge inconsistency."""

    relevant_labels = {
        component_labels[int(anchor_session)],
        component_labels[int(added_session)],
    }
    scored: list[tuple[float, float, int, tuple[int, int], dict[str, Any]]] = []
    for record in stage_records:
        if (
            record.get("anchor_session") is None
            or record.get("added_session") is None
        ):
            continue
        edge = (int(record["anchor_session"]), int(record["added_session"]))
        left, right = edge
        label = component_labels[left]
        if label is None or label != component_labels[right]:
            continue
        if label not in relevant_labels:
            continue
        scored.append(
            (
                float(record.get("bridge_log_evidence", 0.0)),
                -float(record.get("standardized_integer_innovation", 0.0)),
                int(record.get("stage_index", 0)),
                edge,
                dict(record),
            )
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [
        (stage_index, edge, record)
        for _, _, stage_index, edge, record in scored[: max(int(edge_count), 0)]
    ]


def _generate_backward_rejuvenation_candidates(
    *,
    evaluator: Any,
    modes: Sequence[ActiveSessionPathMode],
    restart_values: Sequence[np.ndarray],
    edge_options: Sequence[Mapping[str, Any]],
    stage_records: Sequence[Mapping[str, Any]],
    parameter_scales: np.ndarray,
    prior_center: np.ndarray,
    prior_scales: np.ndarray,
    parameter_bounds: Sequence[tuple[float | None, float | None]],
    config: ActiveSessionLinkConfig,
) -> tuple[list[ActiveSessionPathMode], dict[int, dict[str, Any]], dict[str, Any]]:
    """Re-sample a weak old edge after conditioning on a vetoed bridge.

    Each move cuts one historical edge, fits the resulting forest with the new
    bridge, and only then reconnects the cut edge from its updated posterior.
    This bounded leave-one-edge-out move lets a later contradiction revise an
    early greedy decision without changing the continuous model or its priors.
    """

    if not edge_options or not stage_records:
        return [], {}, {"backward_rejuvenation_job_count": 0}
    anchor = int(edge_options[0]["anchor_session"])
    added = int(edge_options[0]["added_session"])
    tree_edges = [
        (int(record["anchor_session"]), int(record["added_session"]))
        for record in stage_records
    ]
    historical = _historical_rejuvenation_edges(
        stage_records=stage_records,
        component_labels=modes[0].component_labels,
        anchor_session=anchor,
        added_session=added,
        edge_count=config.backward_edge_count,
    )
    if not historical:
        return [], {}, {"backward_rejuvenation_job_count": 0}
    max_cut_size = min(
        max(int(config.backward_max_cut_size), 1),
        len(historical),
    )
    min_cut_size = min(
        max(int(config.backward_min_cut_size), 1),
        max_cut_size,
    )
    historical_blocks = [
        tuple(historical[:cut_size])
        for cut_size in range(min_cut_size, max_cut_size + 1)
    ]

    bridge = edge_options[0]
    jobs: list[tuple[float, int, int, int, float, int, int]] = []
    bridge_radius = max(int(config.backward_bridge_integer_radius), 0)
    bridge_anchor = int(bridge["anchor_session"])
    bridge_added = int(bridge["added_session"])
    bridge_sigma = math.sqrt(max(float(bridge["predictive_variance"]), 1.0e-18))
    candidate_radius = min(
        max(
            int(config.initial_integer_radius),
            int(math.ceil(float(config.candidate_sigma) * bridge_sigma)),
        ),
        max(int(config.max_integer_radius), int(config.initial_integer_radius)),
    )
    starts: list[tuple[int, int, np.ndarray, Any]] = []
    restart_limit = max(int(config.backward_restart_count), 0)
    for parent_index, parent in enumerate(modes):
        parent_starts = [np.asarray(parent.parameter_values, dtype=float)]
        for values_like in restart_values[:restart_limit]:
            values = np.asarray(values_like, dtype=float)
            if any(
                float(np.linalg.norm((values - existing) / parameter_scales))
                <= max(float(config.continuous_mode_merge_radius), 1.0e-6)
                for existing in parent_starts
            ):
                continue
            parent_starts.append(values)
        for start_index, values in enumerate(parent_starts):
            try:
                profile = evaluator.evaluate_values(values)
            except Exception:
                continue
            starts.append((parent_index, start_index, values, profile))
    bridge_variance = max(bridge_sigma**2, 1.0e-18)
    for parent_index, start_index, _values, profile in starts:
        parent = modes[parent_index]
        predicted = float(profile.session_offsets[bridge_added]) - float(
            profile.session_offsets[bridge_anchor]
        )
        nearest = int(np.rint(predicted))
        for block_index, _ in enumerate(historical_blocks):
            for bridge_integer in range(
                nearest - bridge_radius,
                nearest + bridge_radius + 1,
            ):
                jobs.append(
                    (
                        float(parent.laplace_log_evidence)
                        - 0.5
                        * (float(bridge_integer) - predicted) ** 2
                        / bridge_variance,
                        parent_index,
                        start_index,
                        block_index,
                        predicted,
                        int(bridge_integer),
                        candidate_radius,
                    )
                )
    ranked_jobs = sorted(
        jobs,
        key=lambda item: (-item[0], item[1], item[2], item[3], item[5]),
    )
    job_limit = max(int(config.backward_max_jobs), 1)
    selected_jobs: list[tuple[float, int, int, int, float, int, int]] = []
    selected_ids: set[tuple[int, int, int, int]] = set()
    for block_index in range(len(historical_blocks)):
        job = next(
            (item for item in ranked_jobs if item[3] == block_index),
            None,
        )
        if job is None:
            continue
        selected_jobs.append(job)
        selected_ids.add((job[1], job[2], job[3], job[5]))
        if len(selected_jobs) >= job_limit:
            break
    for job in ranked_jobs:
        if len(selected_jobs) >= job_limit:
            break
        identity = (job[1], job[2], job[3], job[5])
        if identity in selected_ids:
            continue
        selected_jobs.append(job)
        selected_ids.add(identity)
    candidates: list[ActiveSessionPathMode] = []
    provenance: dict[int, dict[str, Any]] = {}
    failed_count = 0
    bridge_mode_count = 0
    reconnection_job_count = 0
    optimization_config = replace(
        config,
        trust_radius=max(float(config.trust_radius), float(config.relocation_trust_radius)),
        max_nfev=max(int(config.max_nfev), int(config.relocation_max_nfev)),
    )
    for (
        _, parent_index, start_index, block_index, predicted, bridge_integer, radius,
    ) in selected_jobs:
        parent = modes[parent_index]
        start_values = next(
            values
            for candidate_parent, candidate_start, values, _ in starts
            if candidate_parent == parent_index and candidate_start == start_index
        )
        historical_block = historical_blocks[block_index]
        cut = _cut_historical_edges(
            session_integers=parent.session_integers,
            component_labels=parent.component_labels,
            tree_edges=tree_edges,
            edges=tuple(item[1] for item in historical_block),
        )
        if cut is None:
            failed_count += 1
            continue
        cut_integers, cut_labels, cut_linked = cut
        cut_parent = replace(
            parent,
            session_integers=cut_integers,
            component_labels=cut_labels,
            linked_sessions=cut_linked,
            root_session=min(cut_linked),
            parameter_values=np.asarray(start_values, dtype=float).copy(),
        )
        connected = _connect_session_components(
            cut_parent.session_integers,
            cut_parent.component_labels,
            anchor_session=bridge_anchor,
            added_session=bridge_added,
            edge_integer=bridge_integer,
        )
        if connected is None:
            failed_count += 1
            continue
        integers, labels, linked_after = connected
        rooted_parent = replace(cut_parent, root_session=min(linked_after))
        try:
            bridge_mode = _optimize_path_job(
                evaluator=evaluator,
                parent=rooted_parent,
                parent_mode_index=parent_index,
                session_integers=integers,
                component_labels=labels,
                linked_sessions=linked_after,
                anchor_session=bridge_anchor,
                added_session=bridge_added,
                edge_integer=bridge_integer,
                predicted_edge_phase=predicted,
                predicted_edge_sigma=math.sqrt(
                    max(float(bridge["predictive_variance"]), 1.0e-18)
                ),
                candidate_radius=max(int(radius), int(config.initial_integer_radius)),
                parameter_scales=parameter_scales,
                prior_center=prior_center,
                prior_scales=prior_scales,
                parameter_bounds=parameter_bounds,
                config=optimization_config,
            )
        except Exception:
            bridge_mode = None
        if bridge_mode is None:
            failed_count += 1
            continue
        bridge_mode_count += 1
        historical_radius = max(int(config.backward_integer_radius), 1)
        reconnection_modes: list[
            tuple[ActiveSessionPathMode, tuple[dict[str, Any], ...]]
        ] = [(bridge_mode, ())]
        for historical_stage, historical_edge, _ in reversed(historical_block):
            historical_anchor, historical_added = historical_edge
            original_integer = int(parent.session_integers[historical_added]) - int(
                parent.session_integers[historical_anchor]
            )
            reconnected_candidates: list[
                tuple[ActiveSessionPathMode, tuple[dict[str, Any], ...]]
            ] = []
            for partial_mode, edits in reconnection_modes:
                partial_profile = evaluator.evaluate_values(
                    partial_mode.parameter_values
                )
                predicted_historical = float(
                    partial_profile.session_offsets[historical_added]
                    - partial_profile.session_offsets[historical_anchor]
                )
                historical_nearest = int(np.rint(predicted_historical))
                for historical_integer in range(
                    historical_nearest - historical_radius,
                    historical_nearest + historical_radius + 1,
                ):
                    reconnection_job_count += 1
                    reconnected = _connect_session_components(
                        partial_mode.session_integers,
                        partial_mode.component_labels,
                        anchor_session=historical_anchor,
                        added_session=historical_added,
                        edge_integer=historical_integer,
                    )
                    if reconnected is None:
                        failed_count += 1
                        continue
                    final_integers, final_labels, final_linked = reconnected
                    try:
                        candidate = _optimize_path_job(
                            evaluator=evaluator,
                            parent=replace(
                                partial_mode,
                                root_session=min(final_linked),
                            ),
                            parent_mode_index=parent_index,
                            session_integers=final_integers,
                            component_labels=final_labels,
                            linked_sessions=final_linked,
                            anchor_session=historical_anchor,
                            added_session=historical_added,
                            edge_integer=historical_integer,
                            predicted_edge_phase=predicted_historical,
                            predicted_edge_sigma=bridge_sigma,
                            candidate_radius=historical_radius,
                            parameter_scales=parameter_scales,
                            prior_center=prior_center,
                            prior_scales=prior_scales,
                            parameter_bounds=parameter_bounds,
                            config=optimization_config,
                        )
                    except Exception:
                        candidate = None
                    if candidate is None:
                        failed_count += 1
                        continue
                    reconnected_candidates.append(
                        (
                            candidate,
                            edits
                            + (
                                {
                                    "historical_stage": historical_stage,
                                    "historical_edge": historical_edge,
                                    "original_historical_integer": original_integer,
                                    "resampled_historical_integer": historical_integer,
                                    "integer_delta": historical_integer
                                    - original_integer,
                                },
                            ),
                        )
                    )
            if not reconnected_candidates:
                reconnection_modes = []
                break
            reconnection_candidates = [
                item[0] for item in reconnected_candidates
            ]
            retained, _ = _select_mode_portfolio(
                reconnection_candidates,
                target=max(int(config.backward_reconnection_beam_size), 1),
                config=config,
                parameter_scales=parameter_scales,
            )
            retained_ids = {id(mode) for mode in retained}
            reconnection_modes = [
                item
                for item in reconnected_candidates
                if id(item[0]) in retained_ids
            ]
        for candidate, edits in reconnection_modes:
            candidate = replace(
                candidate,
                anchor_session=bridge_anchor,
                added_session=bridge_added,
                edge_integer=bridge_integer,
                predicted_edge_phase=predicted,
                predicted_edge_sigma=bridge_sigma,
                candidate_radius=radius,
            )
            candidates.append(candidate)
            provenance[id(candidate)] = {
                "cut_size": len(historical_block),
                "restart_index": start_index,
                "historical_edits": edits,
                "bridge_edge": (bridge_anchor, bridge_added),
            }
    return candidates, provenance, {
        "backward_rejuvenation_job_count": len(selected_jobs),
        "backward_rejuvenation_bridge_mode_count": bridge_mode_count,
        "backward_rejuvenation_reconnection_job_count": reconnection_job_count,
        "backward_rejuvenation_optimized_job_count": len(candidates),
        "backward_rejuvenation_failed_job_count": failed_count,
        "backward_rejuvenation_edge_count": len(historical),
        "backward_rejuvenation_max_cut_size": max_cut_size,
        "backward_rejuvenation_min_cut_size": min_cut_size,
    }


def explore_bayesian_active_session_paths(
    *,
    evaluator: Any,
    initial_modes: Sequence[Sequence[float]],
    parameter_scales: Sequence[float],
    prior_center: Sequence[float] | None = None,
    prior_scales: Sequence[float] | None = None,
    parameter_bounds: Sequence[tuple[float | None, float | None]] | None = None,
    config: ActiveSessionLinkConfig = ActiveSessionLinkConfig(),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_callback: (
        Callable[[Sequence[dict[str, Any]], Sequence[ActiveSessionPathMode]], None]
        | None
    ) = None,
    resume_modes: Sequence[ActiveSessionPathMode] = (),
    resume_stage_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[tuple[ActiveSessionPathMode, ...], dict[str, Any]]:
    """Infer a phase-connection tree by linking the lowest-entropy edge first.

    All within-session residual shape is used from the start, while integer
    phase evidence is admitted only for sessions already linked into the tree.
    The next bridge is selected from the posterior predictive integer entropy.
    This postpones isolated long gaps until denser observations have narrowed
    the continuous timing posterior.
    """
    scales = np.asarray(parameter_scales, dtype=float)
    if scales.ndim != 1 or not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("Parameter scales must be finite and positive.")
    center = (
        np.asarray(prior_center, dtype=float)
        if prior_center is not None
        else np.asarray(evaluator.base_parameter_values, dtype=float)
    )
    prior_sigma = scales.copy() if prior_scales is None else np.asarray(prior_scales, dtype=float)
    if center.shape != scales.shape or prior_sigma.shape != scales.shape:
        raise ValueError("Prior center, scales, and parameter scales must align.")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(prior_sigma) & (prior_sigma > 0.0)):
        raise ValueError("Prior center and scales must be finite and valid.")
    bounds = (
        tuple((None, None) for _ in range(scales.size))
        if parameter_bounds is None
        else tuple(parameter_bounds)
    )
    if len(bounds) != scales.size:
        raise ValueError("Parameter bounds must match the parameter dimension.")

    seeds: list[tuple[int, np.ndarray, Any]] = []
    for source_mode_index, values_like in enumerate(initial_modes):
        values = np.asarray(values_like, dtype=float)
        if values.shape != scales.shape or not _parameter_domain_ok(values, bounds):
            continue
        try:
            profile = evaluator.evaluate_values(values)
        except Exception:
            continue
        if int(profile.session_count) < 2:
            raise ValueError("Active session linking requires at least two sessions.")
        seeds.append((source_mode_index, values.copy(), profile))
    if not seeds:
        raise RuntimeError("No valid active-linking initial mode is available.")

    session_count = int(seeds[0][2].session_count)
    unlinked_integers = tuple(None for _ in range(session_count))
    unlinked_components = tuple(None for _ in range(session_count))
    seed_modes: list[ActiveSessionPathMode] = []
    for source_mode_index, values, profile in seeds:
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        shape_chi2 = float(shape @ shape)
        prior_chi2 = float(
            max(float(config.prior_weight), 0.0)
            * np.sum(np.square((values - center) / prior_sigma))
        )
        seed_modes.append(
            ActiveSessionPathMode(
                session_integers=unlinked_integers,
                component_labels=unlinked_components,
                linked_sessions=(),
                root_session=-1,
                parameter_values=values,
                shape_chi2=shape_chi2,
                ambiguity_chi2=0.0,
                prior_chi2=prior_chi2,
                posterior_chi2=shape_chi2 + prior_chi2,
                laplace_log_evidence=-0.5 * (shape_chi2 + prior_chi2),
                future_bridge_log_evidence=0.0,
                predictive_deviance=shape_chi2 + prior_chi2,
                predictive_anchor_session=None,
                predictive_added_session=None,
                scaled_covariance=np.eye(scales.size, dtype=float),
                source_mode_index=int(source_mode_index),
                parent_mode_index=-1,
                anchor_session=None,
                added_session=None,
                edge_integer=None,
                predicted_edge_phase=None,
                predicted_edge_sigma=None,
                candidate_radius=0,
                success=True,
                nfev=0,
            )
        )

    initial_limit = max(int(config.max_beam_size), int(config.beam_size), 1)
    if resume_modes:
        modes = list(resume_modes)
        if any(
            len(mode.session_integers) != session_count
            or len(mode.component_labels) != session_count
            or np.asarray(mode.parameter_values).shape != scales.shape
            for mode in modes
        ):
            raise ValueError("Resume modes do not match the active-linking problem.")
        stage_records = [dict(record) for record in resume_stage_records]
    else:
        modes = sorted(seed_modes, key=lambda mode: mode.posterior_chi2)[:initial_limit]
        stage_records = []
    current_beam_size = max(
        min(
            max(int(config.beam_size), 1),
            max(int(config.max_beam_size), 1),
        ),
        len(modes) if resume_modes else 0,
    )

    start_stage = len(stage_records) + 1
    for stage_index in range(start_stage, session_count):
        representative = min(modes, key=lambda mode: mode.posterior_chi2)
        representative_profile, shape_jacobian, offset_jacobian = _finite_difference_geometry(
            evaluator=evaluator,
            values=np.asarray(representative.parameter_values, dtype=float),
            parameter_scales=scales,
            fraction=max(float(config.finite_difference_fraction), 1.0e-12),
        )
        scaled_covariance = _posterior_covariance_from_geometry(
            profile=representative_profile,
            shape_jacobian=shape_jacobian,
            offset_jacobian=offset_jacobian,
            mode=representative,
            parameter_scales=scales,
            prior_scales=prior_sigma,
            prior_weight=max(float(config.prior_weight), 0.0),
        )
        profiles = [evaluator.evaluate_values(mode.parameter_values) for mode in modes]
        anchor_session, added_session, edge_diagnostics = _select_active_edge(
            modes=modes,
            profiles=profiles,
            component_labels=representative.component_labels,
            offset_jacobian=offset_jacobian,
            scaled_covariance=scaled_covariance,
            config=config,
        )
        particle_topology_diagnostics: list[dict[str, Any]] = []
        if bool(config.topology_particles):
            edge_options = []
            edge_limit = max(int(config.topology_edges_per_mode), 1)
            bottleneck_edge_limit = max(
                int(config.topology_bottleneck_edges_per_mode),
                0,
            )
            for parent_index, (parent, profile) in enumerate(
                zip(modes, profiles, strict=True)
            ):
                _, _, particle_diagnostics = _select_active_edge(
                    modes=(parent,),
                    profiles=(profile,),
                    component_labels=parent.component_labels,
                    offset_jacobian=offset_jacobian,
                    scaled_covariance=scaled_covariance,
                    config=replace(config, relocation_edge_count=edge_limit),
                )
                particle_topology_diagnostics.append(particle_diagnostics)
                particle_edges = list(
                    particle_diagnostics["ranked_edges"][:edge_limit]
                )
                existing_edge_keys = {
                    (
                        int(edge["anchor_session"]),
                        int(edge["added_session"]),
                    )
                    for edge in particle_edges
                }
                for edge in particle_diagnostics.get("bottleneck_edges", ()):
                    if bottleneck_edge_limit <= 0:
                        break
                    edge_key = (
                        int(edge["anchor_session"]),
                        int(edge["added_session"]),
                    )
                    if edge_key in existing_edge_keys:
                        continue
                    particle_edges.append(edge)
                    existing_edge_keys.add(edge_key)
                    bottleneck_edge_limit -= 1
                bottleneck_edge_limit = max(
                    int(config.topology_bottleneck_edges_per_mode),
                    0,
                )
                for edge in particle_edges:
                    edge_options.append(
                        {
                            **edge,
                            "parent_mode_index": int(parent_index),
                        }
                    )
            ranked_edges = sorted(
                edge_options,
                key=lambda edge: (
                    -float(edge["bridge_log_evidence"]),
                    float(edge["predictive_entropy"]),
                    float(edge["predictive_variance"]),
                    int(edge["parent_mode_index"]),
                    int(edge["anchor_session"]),
                    int(edge["added_session"]),
                ),
            )
            best_particle_edge = ranked_edges[0]
            anchor_session = int(best_particle_edge["anchor_session"])
            added_session = int(best_particle_edge["added_session"])
            edge_diagnostics = {
                **best_particle_edge,
                "edge_candidate_count": int(
                    sum(
                        item["edge_candidate_count"]
                        for item in particle_topology_diagnostics
                    )
                ),
                "ranked_edges": ranked_edges,
                "topology_particle_count": len(modes),
                "topology_edge_option_count": len(edge_options),
            }
        relocation_triggered = bool(
            float(
                edge_diagnostics.get(
                    "gaussian_bridge_log_evidence",
                    edge_diagnostics["bridge_log_evidence"],
                )
            )
            < float(config.bridge_log_evidence_gate)
            or float(edge_diagnostics["standardized_integer_innovation"])
            > float(config.bridge_standardized_innovation_gate)
        )
        if (
            relocation_triggered
            and bool(config.stop_on_bridge_veto)
            and not bool(config.backward_rejuvenation)
        ):
            veto = {
                "stage_index": stage_index,
                "status": "bridge-veto",
                "component_count": (
                    len(_component_groups(representative.component_labels))
                    + sum(label is None for label in representative.component_labels)
                ),
                **{
                    key: value
                    for key, value in edge_diagnostics.items()
                    if key != "ranked_edges"
                },
                "ranked_edges": edge_diagnostics["ranked_edges"],
            }
            if checkpoint_callback is not None:
                checkpoint_callback(stage_records, modes)
            return tuple(modes), {
                "proposal_source": "apollo-bayesian-active-session-linker",
                "completed": False,
                "status": "bridge-veto",
                "initial_mode_count": len(seeds),
                "session_count": session_count,
                "final_mode_count": 0,
                "partial_mode_count": len(modes),
                "config": config.__dict__,
                "stages": stage_records,
                "veto": veto,
            }
        ranked_edges = list(edge_diagnostics["ranked_edges"])
        if not bool(config.topology_particles):
            edge_options = (
                ranked_edges[: max(int(config.relocation_edge_count), 1)]
                if relocation_triggered
                else ranked_edges[:1]
            )
        jobs: list[tuple[float, int, int, int, float, int]] = []
        for edge_index, edge in enumerate(edge_options):
            option_anchor = int(edge["anchor_session"])
            option_added = int(edge["added_session"])
            option_sigma = math.sqrt(
                max(float(edge["predictive_variance"]), 1.0e-18)
            )
            configured_parent = edge.get("parent_mode_index")
            parent_indices = (
                (int(configured_parent),)
                if configured_parent is not None
                else tuple(range(len(modes)))
            )
            for parent_index in parent_indices:
                parent = modes[parent_index]
                profile = profiles[parent_index]
                predicted = float(profile.session_offsets[option_added]) - float(
                    profile.session_offsets[option_anchor]
                )
                radius = min(
                    max(
                        int(config.initial_integer_radius),
                        int(math.ceil(float(config.candidate_sigma) * option_sigma)),
                    ),
                    max(int(config.max_integer_radius), int(config.initial_integer_radius)),
                )
                nearest = int(np.rint(predicted))
                proposal_distribution, _ = _discrete_robust_bridge(
                    predicted,
                    option_sigma,
                    radius=radius,
                    outlier_probability=float(config.bridge_outlier_probability),
                    outlier_scale=float(config.bridge_outlier_scale),
                )
                for edge_integer in range(nearest - radius, nearest + radius + 1):
                    predictive_log_weight = (
                        float(parent.laplace_log_evidence)
                        + math.log(
                            max(
                                float(proposal_distribution[edge_integer]),
                                1.0e-300,
                            )
                        )
                    )
                    jobs.append(
                        (
                            predictive_log_weight,
                            edge_index,
                            parent_index,
                            int(edge_integer),
                            predicted,
                            radius,
                        )
                    )
        job_limit = max(
            int(
                config.max_jobs_per_stage_limit
                if relocation_triggered
                else config.max_jobs_per_stage
            ),
            len(modes),
            len(edge_options),
            len({(job[1], job[2]) for job in jobs})
            * max(int(config.integer_candidates_per_edge), 1),
            1,
        )
        jobs_by_parent: dict[
            tuple[int, int],
            list[tuple[float, int, int, int, float, int]],
        ] = {}
        for job in jobs:
            jobs_by_parent.setdefault((job[1], job[2]), []).append(job)
        selected_jobs: list[tuple[float, int, int, int, float, int]] = []
        selected_job_keys: set[tuple[int, int, int]] = set()
        ranked_jobs_by_parent = {
            edge_parent: sorted(
                parent_jobs,
                key=lambda item: (-item[0], abs(item[3]), item[3]),
            )
            for edge_parent, parent_jobs in jobs_by_parent.items()
        }
        for integer_rank in range(max(int(config.integer_candidates_per_edge), 1)):
            for edge_parent in sorted(ranked_jobs_by_parent):
                parent_jobs = ranked_jobs_by_parent[edge_parent]
                if integer_rank >= len(parent_jobs):
                    continue
                job = parent_jobs[integer_rank]
                selected_jobs.append(job)
                selected_job_keys.add((job[1], job[2], job[3]))
                if len(selected_jobs) >= job_limit:
                    break
            if len(selected_jobs) >= job_limit:
                break
        for job in sorted(jobs, key=lambda item: (-item[0], item[1], item[2], item[3])):
            if len(selected_jobs) >= job_limit:
                break
            key = (job[1], job[2], job[3])
            if key in selected_job_keys:
                continue
            selected_jobs.append(job)
            selected_job_keys.add(key)

        candidates: list[ActiveSessionPathMode] = []
        rejuvenation_provenance: dict[int, dict[str, Any]] = {}
        rejuvenation_diagnostics: dict[str, Any] = {
            "backward_rejuvenation_triggered": False,
            "backward_rejuvenation_job_count": 0,
            "phase_split_rejuvenation_triggered": False,
        }
        failed_count = 0
        optimization_config = (
            replace(
                config,
                trust_radius=max(
                    float(config.trust_radius),
                    float(config.relocation_trust_radius),
                ),
                max_nfev=max(
                    int(config.max_nfev),
                    int(config.relocation_max_nfev),
                ),
            )
            if relocation_triggered
            else config
        )
        for (
            _,
            edge_index,
            parent_index,
            edge_integer,
            predicted,
            radius,
        ) in selected_jobs:
            edge = edge_options[edge_index]
            option_anchor = int(edge["anchor_session"])
            option_added = int(edge["added_session"])
            option_sigma = math.sqrt(
                max(float(edge["predictive_variance"]), 1.0e-18)
            )
            parent = modes[parent_index]
            connected = _connect_session_components(
                parent.session_integers,
                parent.component_labels,
                anchor_session=option_anchor,
                added_session=option_added,
                edge_integer=edge_integer,
            )
            if connected is None:
                continue
            integers, labels, linked_after = connected
            root_session = min(linked_after)
            rooted_parent = ActiveSessionPathMode(
                **{
                    **parent.__dict__,
                    "root_session": root_session,
                }
            )
            try:
                candidate = _optimize_path_job(
                    evaluator=evaluator,
                    parent=rooted_parent,
                    parent_mode_index=parent_index,
                    session_integers=integers,
                    component_labels=labels,
                    linked_sessions=linked_after,
                    anchor_session=option_anchor,
                    added_session=option_added,
                    edge_integer=edge_integer,
                    predicted_edge_phase=predicted,
                    predicted_edge_sigma=option_sigma,
                    candidate_radius=radius,
                    parameter_scales=scales,
                    prior_center=center,
                    prior_scales=prior_sigma,
                    parameter_bounds=bounds,
                    config=optimization_config,
                )
            except Exception:
                candidate = None
            if candidate is None:
                failed_count += 1
            else:
                candidates.append(candidate)
        baseline_candidate_count = len(candidates)
        baseline_best_chi2 = (
            float(min(mode.posterior_chi2 for mode in candidates))
            if candidates
            else math.inf
        )
        baseline_bridge_chi2_increment = float(
            baseline_best_chi2 - min(mode.posterior_chi2 for mode in modes)
        )
        rejuvenation_gate = max(
            float(config.bridge_standardized_innovation_gate) ** 2,
            1.0,
        )
        rejuvenation_triggered = bool(
            config.backward_rejuvenation
            and baseline_bridge_chi2_increment > rejuvenation_gate
        )
        if rejuvenation_triggered:
            split_candidates, rejuvenation_provenance, split_diagnostics = (
                _generate_phase_block_vector_candidates(
                    evaluator=evaluator,
                    modes=modes,
                    baseline_candidates=candidates,
                    restart_values=tuple(
                        np.asarray(mode.parameter_values, dtype=float)
                        for mode in seed_modes
                    ),
                    parameter_scales=scales,
                    prior_center=center,
                    prior_scales=prior_sigma,
                    parameter_bounds=bounds,
                    config=config,
                )
            )
            candidates.extend(split_candidates)
            rejuvenation_diagnostics = {
                "backward_rejuvenation_triggered": True,
                "phase_split_rejuvenation_triggered": True,
                **split_diagnostics,
                "backward_rejuvenation_candidate_count": len(split_candidates),
                "backward_rejuvenation_best_chi2": (
                    float(min(mode.posterior_chi2 for mode in split_candidates))
                    if split_candidates
                    else None
                ),
                "baseline_candidate_count": baseline_candidate_count,
                "baseline_best_chi2": (
                    baseline_best_chi2 if np.isfinite(baseline_best_chi2) else None
                ),
                "baseline_bridge_chi2_increment": baseline_bridge_chi2_increment,
                "phase_split_rejuvenation_gate": rejuvenation_gate,
            }
        if (
            relocation_triggered
            and candidates
            and not bool(config.topology_particles)
        ):
            grouped_candidates: dict[
                tuple[int, int],
                list[ActiveSessionPathMode],
            ] = {}
            for candidate in candidates:
                key = (int(candidate.anchor_session), int(candidate.added_session))
                grouped_candidates.setdefault(key, []).append(candidate)
            chosen_edge_key = min(
                grouped_candidates,
                key=lambda key: (
                    min(mode.posterior_chi2 for mode in grouped_candidates[key]),
                    -max(
                        mode.laplace_log_evidence
                        for mode in grouped_candidates[key]
                    ),
                    key,
                ),
            )
            candidates = grouped_candidates[chosen_edge_key]
            chosen_edge = next(
                edge
                for edge in edge_options
                if (
                    int(edge["anchor_session"]),
                    int(edge["added_session"]),
                )
                == chosen_edge_key
            )
            anchor_session, added_session = chosen_edge_key
            edge_diagnostics = {
                **chosen_edge,
                "edge_candidate_count": edge_diagnostics["edge_candidate_count"],
                "ranked_edges": ranked_edges,
                "relocation_triggered": True,
                "relocation_edge_count": len(edge_options),
                "linear_best_edge": (
                    int(ranked_edges[0]["anchor_session"]),
                    int(ranked_edges[0]["added_session"]),
                ),
            }
        else:
            edge_diagnostics = {
                **edge_diagnostics,
                "relocation_triggered": relocation_triggered,
                "relocation_edge_count": len(edge_options),
            }
        if not candidates:
            raise RuntimeError(
                f"Active session linking failed while adding session {added_session}."
            )
        unscored_candidates = candidates
        if float(config.predictive_fraction) > 0.0:
            predictive_limit = max(int(config.predictive_candidate_limit), 0)
            predictive_candidates = list(candidates)
            predictive_prefilter = {
                "enabled": False,
                "reason": "pool-within-budget",
                "input_candidate_count": len(candidates),
                "selected_candidate_count": len(candidates),
            }
            if predictive_limit and len(candidates) > predictive_limit:
                predictive_candidates, predictive_prefilter = (
                    _select_mode_portfolio(
                        candidates,
                        target=predictive_limit,
                        config=replace(config, predictive_fraction=0.0),
                        parameter_scales=scales,
                    )
                )
                predictive_prefilter = {
                    "enabled": True,
                    "input_candidate_count": len(candidates),
                    "selected_candidate_count": len(predictive_candidates),
                    **predictive_prefilter,
                }
            scored_candidates, predictive_diagnostics = _score_future_bridges(
                evaluator=evaluator,
                candidates=predictive_candidates,
                parameter_scales=scales,
                config=config,
                offset_jacobian=offset_jacobian,
            )
            scored_by_identity = {
                id(unscored): scored
                for unscored, scored in zip(
                    predictive_candidates,
                    scored_candidates,
                    strict=True,
                )
            }
            candidates = [
                scored_by_identity.get(
                    id(candidate),
                    replace(
                        candidate,
                        future_bridge_log_evidence=-math.inf,
                        predictive_deviance=math.inf,
                        predictive_anchor_session=None,
                        predictive_added_session=None,
                    ),
                )
                for candidate in candidates
            ]
            predictive_diagnostics = {
                **predictive_diagnostics,
                "candidate_limit": predictive_limit,
                "prefilter": predictive_prefilter,
            }
        else:
            candidates = [
                replace(
                    candidate,
                    future_bridge_log_evidence=0.0,
                    predictive_deviance=float(candidate.posterior_chi2),
                    predictive_anchor_session=None,
                    predictive_added_session=None,
                )
                for candidate in candidates
            ]
            predictive_diagnostics = {
                "enabled": False,
                "reason": "zero-predictive-portfolio-fraction",
                "predictive_scored_count": 0,
                "predictive_failure_count": 0,
                "predictive_linearization": "disabled",
                "candidate_count": len(candidates),
            }
        rejuvenation_provenance = {
            id(scored): rejuvenation_provenance[id(unscored)]
            for unscored, scored in zip(
                unscored_candidates,
                candidates,
                strict=True,
            )
            if id(unscored) in rejuvenation_provenance
        }
        beam_size_before = current_beam_size
        target = current_beam_size
        selected, selection_diagnostics = _select_mode_portfolio(
            candidates,
            target=target,
            config=config,
            parameter_scales=scales,
        )
        expansion_rounds = 0
        while (
            expansion_rounds < max(int(config.auto_expand_rounds), 0)
            and target < max(int(config.max_beam_size), 1)
            and float(selection_diagnostics.get("tail_mass", 0.0))
            >= float(config.tail_mass_trigger)
        ):
            target = min(
                max(target * 2, target + 1),
                max(int(config.max_beam_size), 1),
                len(candidates),
            )
            expanded, expanded_diagnostics = _select_mode_portfolio(
                candidates,
                target=target,
                config=config,
                parameter_scales=scales,
            )
            if len(expanded) <= len(selected):
                break
            selected = expanded
            selection_diagnostics = expanded_diagnostics
            expansion_rounds += 1
        current_beam_size = max(current_beam_size, len(selected))
        if not selected:
            raise RuntimeError(
                f"Active session linking retained no mode at stage {stage_index}."
            )
        selected_representative = min(
            selected,
            key=lambda mode: (mode.posterior_chi2, -mode.laplace_log_evidence),
        )
        anchor_session = int(selected_representative.anchor_session)
        added_session = int(selected_representative.added_session)
        if bool(config.topology_particles):
            selected_edge = next(
                (
                    edge
                    for edge in edge_options
                    if int(edge["anchor_session"]) == anchor_session
                    and int(edge["added_session"]) == added_session
                    and int(edge.get("parent_mode_index", -1))
                    == int(selected_representative.parent_mode_index)
                ),
                None,
            )
            edge_diagnostics = {
                **edge_diagnostics,
                **({} if selected_edge is None else selected_edge),
                "ranked_edges": ranked_edges,
                "relocation_triggered": relocation_triggered,
                "relocation_edge_count": len(edge_options),
                "topology_particle_count": len(modes),
                "topology_edge_option_count": len(edge_options),
            }
        boundary_count = sum(
            abs(int(mode.edge_integer) - int(np.rint(float(mode.predicted_edge_phase))))
            >= int(mode.candidate_radius)
            for mode in selected
        )
        boundary_fraction = float(boundary_count / len(selected))
        tail_mass = float(selection_diagnostics.get("tail_mass", 0.0))
        recommendation = (
            "expand-integer-radius"
            if boundary_fraction >= float(config.boundary_fraction_trigger)
            else "expand-beam"
            if tail_mass >= float(config.tail_mass_trigger)
            else "none"
        )
        selected_rejuvenation = [
            rejuvenation_provenance[id(mode)]
            for mode in selected
            if id(mode) in rejuvenation_provenance
        ]
        bridge_chi2_increment = float(
            min(mode.posterior_chi2 for mode in selected)
            - min(mode.posterior_chi2 for mode in modes)
        )
        bridge_chi2_increment_gate = max(
            float(config.bridge_standardized_innovation_gate) ** 2,
            1.0,
        )
        if (
            rejuvenation_triggered
            and bool(config.stop_on_bridge_veto)
            and bridge_chi2_increment > bridge_chi2_increment_gate
        ):
            veto = {
                "stage_index": stage_index,
                "status": "bridge-veto-after-rejuvenation",
                **{
                    key: value
                    for key, value in edge_diagnostics.items()
                    if key != "ranked_edges"
                },
                "ranked_edges": edge_diagnostics["ranked_edges"],
                **rejuvenation_diagnostics,
                "bridge_chi2_increment": bridge_chi2_increment,
                "bridge_chi2_increment_gate": bridge_chi2_increment_gate,
                "backward_rejuvenation_selected_count": len(
                    selected_rejuvenation
                ),
                "backward_rejuvenation_selected_edits": selected_rejuvenation,
                "selection": selection_diagnostics,
            }
            if checkpoint_callback is not None:
                checkpoint_callback(stage_records, modes)
            return tuple(modes), {
                "proposal_source": "apollo-bayesian-active-session-linker",
                "completed": False,
                "status": "bridge-veto-after-rejuvenation",
                "initial_mode_count": len(seeds),
                "session_count": session_count,
                "final_mode_count": 0,
                "partial_mode_count": len(modes),
                "config": config.__dict__,
                "stages": stage_records,
                "veto": veto,
            }
        if (
            bool(config.global_path_refresh_on_bridge_jump)
            and bridge_chi2_increment > bridge_chi2_increment_gate
        ):
            veto = {
                "stage_index": stage_index,
                "status": "global-phase-path-refresh",
                **{
                    key: value
                    for key, value in edge_diagnostics.items()
                    if key != "ranked_edges"
                },
                "ranked_edges": edge_diagnostics["ranked_edges"],
                "bridge_chi2_increment": bridge_chi2_increment,
                "bridge_chi2_increment_gate": bridge_chi2_increment_gate,
                "selection": selection_diagnostics,
                "parent_mode_count": len(modes),
            }
            if checkpoint_callback is not None:
                # Preserve the forest before the rejected bridge.  A global
                # path refresh must not inherit the newly selected bad edge.
                checkpoint_callback(stage_records, modes)
            return tuple(modes), {
                "proposal_source": "apollo-bayesian-active-session-linker",
                "completed": False,
                "status": "global-phase-path-refresh",
                "initial_mode_count": len(seeds),
                "session_count": session_count,
                "final_mode_count": 0,
                "partial_mode_count": len(modes),
                "config": config.__dict__,
                "stages": stage_records,
                "veto": veto,
            }
        selected_parent = modes[int(selected_representative.parent_mode_index)]
        stage_record = {
                "stage_index": stage_index,
                "component_count_before": (
                    len(_component_groups(selected_parent.component_labels))
                    + sum(label is None for label in selected_parent.component_labels)
                ),
                "component_count_after": (
                    len(_component_groups(selected_representative.component_labels))
                    + sum(
                        label is None
                        for label in selected_representative.component_labels
                    )
                ),
                "linked_before": selected_parent.linked_sessions,
                "linked_after": selected_representative.linked_sessions,
                **edge_diagnostics,
                "input_mode_count": len(modes),
                "generated_job_count": len(jobs),
                "optimized_job_count": len(selected_jobs),
                "integer_candidates_per_edge": max(
                    int(config.integer_candidates_per_edge),
                    1,
                ),
                "failed_job_count": failed_count,
                "beam_size_before": beam_size_before,
                "beam_size_after": len(selected),
                "beam_expansion_rounds": expansion_rounds,
                "integer_boundary_fraction": boundary_fraction,
                "budget_recommendation": recommendation,
                **rejuvenation_diagnostics,
                "bridge_chi2_increment": bridge_chi2_increment,
                "bridge_chi2_increment_gate": bridge_chi2_increment_gate,
                "backward_rejuvenation_selected_count": len(selected_rejuvenation),
                "backward_rejuvenation_selected_edits": selected_rejuvenation,
                "selection": selection_diagnostics,
                "predictive_scoring": predictive_diagnostics,
                "best_posterior_chi2": float(min(mode.posterior_chi2 for mode in selected)),
                "best_laplace_log_evidence": float(
                    max(mode.laplace_log_evidence for mode in selected)
                ),
            }
        stage_records.append(stage_record)
        if progress_callback is not None:
            progress_callback(stage_record)
        modes = selected
        if checkpoint_callback is not None:
            checkpoint_callback(stage_records, modes)

    final_modes = tuple(
        mode
        for mode in modes
        if all(value is not None for value in mode.session_integers)
        and len({int(label) for label in mode.component_labels}) == 1
    )
    return final_modes, {
        "proposal_source": "apollo-bayesian-active-session-linker",
        "completed": True,
        "initial_mode_count": len(seeds),
        "session_count": session_count,
        "final_mode_count": len(final_modes),
        "final_unique_path_count": len({mode.session_integers for mode in final_modes}),
        "config": config.__dict__,
        "stages": stage_records,
    }


def active_session_path_modes_to_branches(
    *,
    modes: Sequence[ActiveSessionPathMode],
    base_parameter_values: Sequence[float],
    parameter_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
) -> list[ActiveTimingBranch]:
    base = np.asarray(base_parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    branches: list[ActiveTimingBranch] = []
    for mode_index, mode in enumerate(modes):
        integers = tuple(int(value) for value in mode.session_integers)
        relative_to_first = tuple(value - integers[0] for value in integers[1:])
        theta_offset = np.asarray(mode.parameter_values, dtype=float) - base
        branches.append(
            ActiveTimingBranch(
                branch_id=f"active_session_path_{mode_index:04d}",
                structure_id=str(structure_id),
                binary_model=(str(binary_model).upper() if binary_model is not None else None),
                wraps=integer_session_offsets_to_gap_wraps(relative_to_first),
                theta_names=names,
                theta_offset=theta_offset,
                log_prior=float(mode.laplace_log_evidence),
                proposal_source="apollo-bayesian-active-session-linker",
                diagnostics={
                    "proposal_family": "bayesian-active-session-link-tree",
                    "session_integers": integers,
                    "component_labels": mode.component_labels,
                    "relative_session_integers": relative_to_first,
                    "linked_sessions": mode.linked_sessions,
                    "root_session": mode.root_session,
                    "shape_chi2": mode.shape_chi2,
                    "ambiguity_chi2": mode.ambiguity_chi2,
                    "prior_chi2": mode.prior_chi2,
                    "posterior_chi2": mode.posterior_chi2,
                    "laplace_log_evidence": mode.laplace_log_evidence,
                    "future_bridge_log_evidence": mode.future_bridge_log_evidence,
                    "predictive_deviance": mode.predictive_deviance,
                    "predictive_anchor_session": mode.predictive_anchor_session,
                    "predictive_added_session": mode.predictive_added_session,
                    "source_mode_index": mode.source_mode_index,
                    "certification_theta_offset": theta_offset,
                },
            )
        )
    return branches
