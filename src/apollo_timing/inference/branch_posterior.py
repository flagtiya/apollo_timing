from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from apollo_timing.inference.active_timing_contracts import (
    ActiveTimingBranch,
    ActiveTimingPosterior,
    BranchCertification,
)


@dataclass(frozen=True)
class BranchPosteriorConfig:
    certifiable_probability: float = 0.9
    certifiable_log_odds: float = 2.0
    ambiguity_probability_floor: float = 0.01
    max_ambiguity_pairs: int = 5
    max_reduced_chi2: float | None = None
    max_weighted_rms_us: float | None = None


def _logdet_covariance(covariance: np.ndarray) -> float:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return float("-inf")
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(logdet):
        return float("-inf")
    return float(logdet)


def laplace_log_evidence(log_prior: float, certification: BranchCertification) -> float:
    if certification.log_evidence is not None:
        return float(certification.log_evidence)
    covariance = np.asarray(certification.covariance, dtype=float)
    dimension = int(covariance.shape[0]) if covariance.ndim == 2 else 0
    logdet = _logdet_covariance(covariance)
    if not np.isfinite(logdet):
        return float("-inf")
    log_occam_volume = 0.5 * dimension * math.log(2.0 * math.pi) + 0.5 * logdet
    return float(certification.log_likelihood) + float(log_prior) + float(log_occam_volume)


def _quality_gate(
    certification: BranchCertification,
    config: BranchPosteriorConfig,
) -> dict[str, bool | float | None]:
    reduced_ok = (
        True
        if config.max_reduced_chi2 is None
        else (
            certification.reduced_chi2 is not None
            and float(certification.reduced_chi2) <= float(config.max_reduced_chi2)
        )
    )
    rms_ok = (
        True
        if config.max_weighted_rms_us is None
        else (
            certification.weighted_rms_us is not None
            and float(certification.weighted_rms_us) <= float(config.max_weighted_rms_us)
        )
    )
    return {
        "passed": bool(reduced_ok and rms_ok),
        "reduced_chi2": certification.reduced_chi2,
        "weighted_rms_us": certification.weighted_rms_us,
        "max_reduced_chi2": config.max_reduced_chi2,
        "max_weighted_rms_us": config.max_weighted_rms_us,
    }


def _normalize_log_evidences(log_evidences: dict[str, float]) -> dict[str, float]:
    finite_items = {
        branch_id: float(logz)
        for branch_id, logz in log_evidences.items()
        if np.isfinite(float(logz))
    }
    if not finite_items:
        return {branch_id: 0.0 for branch_id in log_evidences}
    max_logz = max(finite_items.values())
    norm = sum(math.exp(logz - max_logz) for logz in finite_items.values())
    log_norm = max_logz + math.log(norm)
    return {
        branch_id: math.exp(logz - log_norm) if branch_id in finite_items else 0.0
        for branch_id, logz in log_evidences.items()
    }


def _ambiguity_pairs(
    probabilities: dict[str, float],
    config: BranchPosteriorConfig,
) -> list[dict[str, float | str]]:
    ranked = [
        (branch_id, probability)
        for branch_id, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        if probability >= float(config.ambiguity_probability_floor)
    ]
    pairs: list[dict[str, float | str]] = []
    for left_index, (left_id, left_probability) in enumerate(ranked):
        for right_id, right_probability in ranked[left_index + 1 :]:
            pairs.append(
                {
                    "left_branch_id": left_id,
                    "right_branch_id": right_id,
                    "joint_probability": float(left_probability * right_probability),
                    "left_probability": float(left_probability),
                    "right_probability": float(right_probability),
                }
            )
    return sorted(pairs, key=lambda item: float(item["joint_probability"]), reverse=True)[
        : max(int(config.max_ambiguity_pairs), 0)
    ]


def compute_active_branch_posterior(
    branches: list[ActiveTimingBranch],
    certifications: list[BranchCertification],
    config: BranchPosteriorConfig | None = None,
) -> ActiveTimingPosterior:
    active_config = config or BranchPosteriorConfig()
    branch_by_id = {branch.branch_id: branch for branch in branches}
    certifications_by_id = {item.branch_id: item for item in certifications}
    log_evidences: dict[str, float] = {}
    quality_gates: dict[str, dict[str, bool | float | None]] = {}

    for branch in branches:
        certification = certifications_by_id.get(branch.branch_id)
        if certification is None:
            log_evidences[branch.branch_id] = float("-inf")
            quality_gates[branch.branch_id] = {
                "passed": False,
                "reduced_chi2": None,
                "weighted_rms_us": None,
                "max_reduced_chi2": active_config.max_reduced_chi2,
                "max_weighted_rms_us": active_config.max_weighted_rms_us,
            }
            continue
        log_evidences[branch.branch_id] = laplace_log_evidence(branch.log_prior, certification)
        quality_gates[branch.branch_id] = _quality_gate(certification, active_config)

    probabilities = _normalize_log_evidences(log_evidences)
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    best_branch_id = ranked[0][0] if ranked else None
    entropy = -sum(probability * math.log(probability) for probability in probabilities.values() if probability > 0.0)
    log_odds = None
    if len(ranked) > 1:
        top = ranked[0][1]
        second = ranked[1][1]
        if second <= 0.0:
            log_odds = float("inf")
        elif top <= 0.0:
            log_odds = float("-inf")
        else:
            log_odds = math.log(top) - math.log(second)
    best_probability = ranked[0][1] if ranked else 0.0
    best_quality_passed = bool(quality_gates.get(best_branch_id or "", {}).get("passed", False))
    certifiable = bool(
        best_branch_id in branch_by_id
        and best_probability >= float(active_config.certifiable_probability)
        and (log_odds is None or log_odds >= float(active_config.certifiable_log_odds))
        and best_quality_passed
    )
    return ActiveTimingPosterior(
        branch_probabilities=probabilities,
        log_evidences=log_evidences,
        entropy=float(entropy),
        best_branch_id=best_branch_id,
        log_odds_best_second=log_odds,
        certifiable=certifiable,
        ambiguity_pairs=_ambiguity_pairs(probabilities, active_config),
        quality_gates=quality_gates,
    )
