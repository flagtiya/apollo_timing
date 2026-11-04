from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AliasManifoldConfig:
    max_wrap_residual: float = 0.25
    certifiable_probability: float = 0.9
    certifiable_log_odds: float = 2.0


@dataclass(frozen=True)
class AliasManifoldCandidate:
    candidate_id: str
    parameter_offset: np.ndarray
    wraps: tuple[int, ...]
    wrap_residual_norm: float
    log_prior: float
    pulse_number_reference: str = "candidate"


@dataclass(frozen=True)
class LocalPhaseSlopeEvidence:
    cluster_ids: tuple[str, ...]
    cluster_mjd: np.ndarray
    session_span_seconds: np.ndarray
    reference_phase_change: np.ndarray
    phase_change_jacobian: np.ndarray
    phase_change_sigma: np.ndarray


@dataclass(frozen=True)
class AliasGeometryPayload:
    gap_ids: list[str]
    parameter_names: list[str]
    gap_jacobian: np.ndarray
    reference_phase_residuals: np.ndarray
    design_by_epoch: dict[float, np.ndarray]
    toa_error_phase: float
    local_phase_slope_evidence: LocalPhaseSlopeEvidence | None = None


@dataclass(frozen=True)
class BranchPosterior:
    branch_probabilities: dict[str, float]
    log_evidences: dict[str, float]
    entropy: float
    best_candidate_id: str | None
    log_odds_best_second: float | None
    certifiable: bool


@dataclass(frozen=True)
class FollowupRecommendation:
    epoch: float
    score: float
    expected_phase_separation: float
    phase_predictions: dict[str, float]


def _build_candidate(
    predicted_wraps: np.ndarray,
    offset: np.ndarray,
    candidate_index: int,
    max_wrap_residual: float,
    log_prior_scale: float,
) -> AliasManifoldCandidate | None:
    wraps = np.rint(predicted_wraps).astype(int)
    residual = predicted_wraps - wraps
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm > max_wrap_residual:
        return None
    log_prior = -0.5 * float(np.dot(offset, offset)) / max(float(log_prior_scale), 1.0e-12) ** 2
    return AliasManifoldCandidate(
        candidate_id=f"alias_{candidate_index:03d}",
        parameter_offset=offset,
        wraps=tuple(int(item) for item in wraps.tolist()),
        wrap_residual_norm=residual_norm,
        log_prior=log_prior,
        pulse_number_reference="seed" if wraps.size > 0 else "candidate",
    )


def build_alias_manifold_candidates(
    gap_jacobian: np.ndarray,
    parameter_offsets: list[np.ndarray],
    config: AliasManifoldConfig | None = None,
    log_prior_scale: float = 1.0,
) -> list[AliasManifoldCandidate]:
    active_config = config or AliasManifoldConfig()
    jacobian = np.asarray(gap_jacobian, dtype=float)
    candidates: list[AliasManifoldCandidate] = []
    for index, raw_offset in enumerate(parameter_offsets):
        offset = np.asarray(raw_offset, dtype=float)
        predicted_wraps = jacobian @ offset
        candidate = _build_candidate(
            predicted_wraps=predicted_wraps,
            offset=offset,
            candidate_index=len(candidates),
            max_wrap_residual=float(active_config.max_wrap_residual),
            log_prior_scale=log_prior_scale,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def build_alias_candidates_from_geometry(
    payload: AliasGeometryPayload,
    parameter_offsets: list[np.ndarray],
    config: AliasManifoldConfig | None = None,
    log_prior_scale: float = 1.0,
) -> list[AliasManifoldCandidate]:
    active_config = config or AliasManifoldConfig()
    jacobian = np.asarray(payload.gap_jacobian, dtype=float)
    reference_phase_residuals = np.asarray(payload.reference_phase_residuals, dtype=float)
    candidates: list[AliasManifoldCandidate] = []
    for raw_offset in parameter_offsets:
        offset = np.asarray(raw_offset, dtype=float)
        if jacobian.size == 0:
            predicted_wraps = np.zeros(0, dtype=float)
        else:
            predicted_wraps = reference_phase_residuals + jacobian @ offset
        candidate = _build_candidate(
            predicted_wraps=predicted_wraps,
            offset=offset,
            candidate_index=len(candidates),
            max_wrap_residual=float(active_config.max_wrap_residual),
            log_prior_scale=log_prior_scale,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _logdet_covariance(covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.asarray(covariance, dtype=float))
    if sign <= 0:
        return float("-inf")
    return float(logdet)


def compute_branch_posterior(
    candidates: list[AliasManifoldCandidate],
    log_likelihoods: dict[str, float],
    parameter_covariances: dict[str, np.ndarray],
    config: AliasManifoldConfig | None = None,
) -> BranchPosterior:
    active_config = config or AliasManifoldConfig()
    log_evidences: dict[str, float] = {}
    for candidate in candidates:
        log_likelihood = float(log_likelihoods[candidate.candidate_id])
        covariance = np.asarray(parameter_covariances[candidate.candidate_id], dtype=float)
        dimension = int(covariance.shape[0])
        log_occam_volume = 0.5 * dimension * math.log(2.0 * math.pi) + 0.5 * _logdet_covariance(covariance)
        log_evidences[candidate.candidate_id] = log_likelihood + candidate.log_prior + log_occam_volume

    if not log_evidences:
        return BranchPosterior({}, {}, 0.0, None, None, False)

    max_log_evidence = max(log_evidences.values())
    normalizer = sum(math.exp(value - max_log_evidence) for value in log_evidences.values())
    log_normalizer = max_log_evidence + math.log(normalizer)
    branch_probabilities = {
        candidate_id: math.exp(log_evidence - log_normalizer)
        for candidate_id, log_evidence in log_evidences.items()
    }
    ranked = sorted(branch_probabilities.items(), key=lambda item: item[1], reverse=True)
    best_candidate_id = ranked[0][0]
    entropy = -sum(probability * math.log(probability) for probability in branch_probabilities.values() if probability > 0.0)
    log_odds = None
    if len(ranked) > 1:
        if ranked[1][1] <= 0.0:
            log_odds = float("inf")
        elif ranked[0][1] <= 0.0:
            log_odds = float("-inf")
        else:
            log_odds = math.log(ranked[0][1]) - math.log(ranked[1][1])
    certifiable = ranked[0][1] >= float(active_config.certifiable_probability)
    if log_odds is not None:
        certifiable = certifiable and log_odds >= float(active_config.certifiable_log_odds)

    return BranchPosterior(
        branch_probabilities=branch_probabilities,
        log_evidences=log_evidences,
        entropy=float(entropy),
        best_candidate_id=best_candidate_id,
        log_odds_best_second=log_odds,
        certifiable=bool(certifiable),
    )


def recommend_followup_epoch(
    candidate_parameter_offsets: dict[str, np.ndarray],
    posterior_weights: dict[str, float],
    design_by_epoch: dict[float, np.ndarray],
    toa_error_phase: float,
) -> FollowupRecommendation:
    if not design_by_epoch:
        raise ValueError("design_by_epoch must contain at least one candidate epoch.")
    best: FollowupRecommendation | None = None
    candidate_ids = list(candidate_parameter_offsets)
    for epoch, raw_design in design_by_epoch.items():
        design = np.asarray(raw_design, dtype=float)
        phase_predictions = {
            candidate_id: float(design @ np.asarray(offset, dtype=float))
            for candidate_id, offset in candidate_parameter_offsets.items()
        }
        weighted_separation = 0.0
        for left_index, left_id in enumerate(candidate_ids):
            for right_id in candidate_ids[left_index + 1 :]:
                pair_weight = float(posterior_weights.get(left_id, 0.0)) * float(posterior_weights.get(right_id, 0.0))
                weighted_separation += pair_weight * abs(phase_predictions[left_id] - phase_predictions[right_id])
        expected_phase_separation = weighted_separation / max(float(toa_error_phase), 1.0e-12)
        recommendation = FollowupRecommendation(
            epoch=float(epoch),
            score=float(expected_phase_separation),
            expected_phase_separation=float(expected_phase_separation),
            phase_predictions=phase_predictions,
        )
        if best is None or recommendation.score > best.score:
            best = recommendation
    if best is None:
        raise ValueError("No follow-up recommendation could be computed.")
    return best
