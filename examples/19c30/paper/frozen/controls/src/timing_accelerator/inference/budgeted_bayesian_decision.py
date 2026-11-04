from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from timing_accelerator.inference.active_timing_contracts import (
    ActiveTimingBranch,
    BranchCertification,
)


@dataclass(frozen=True)
class BudgetedDecisionConfig:
    budget: int = 64
    quality_floor: int = 0
    family_floor: int = 1
    portfolio_floor: int = 0
    quality_weight: float = 1.0
    diagnostic_weight: float = 0.5
    diversity_weight: float = 0.35
    uncertainty_weight: float = 0.0
    prior_weight: float = 0.15
    lookahead_weight: float = 0.0
    prior_score_mode: str = "family-relative-proposal"
    max_family_fraction: float = 0.65
    basin_floor: int = 0
    max_basin_fraction: float = 1.0


def _branch_family(branch: ActiveTimingBranch) -> str:
    diagnostics = branch.diagnostics or {}
    return str(
        diagnostics.get("proposal_family")
        or diagnostics.get("continuous_proposal_family")
        or branch.proposal_source
        or "unknown"
    ).lower()


def _branch_basin(branch: ActiveTimingBranch) -> str:
    return ",".join(str(int(item)) for item in branch.wraps)


def _branch_portfolio(branch: ActiveTimingBranch) -> str | None:
    diagnostics = branch.diagnostics or {}
    if "portfolio_index" not in diagnostics:
        return None
    try:
        portfolio_index = int(diagnostics["portfolio_index"])
    except (TypeError, ValueError):
        return None
    return f"{_branch_family(branch)}:p{portfolio_index}"


def _safe_log_quality(certification: BranchCertification | None) -> float:
    if certification is None:
        return -1.0e12
    chi2r = certification.reduced_chi2
    wrms = certification.weighted_rms_us
    chi2r_value = float(chi2r) if chi2r is not None and np.isfinite(float(chi2r)) else 1.0e12
    wrms_value = float(wrms) if wrms is not None and np.isfinite(float(wrms)) else 1.0e12
    return -0.55 * math.log1p(max(chi2r_value, 0.0)) - 0.45 * math.log1p(max(wrms_value, 0.0))


def _diagnostic_score(branch: ActiveTimingBranch) -> float:
    diagnostics = branch.diagnostics or {}
    if "diagnostic_log_likelihood" in diagnostics:
        value = float(diagnostics["diagnostic_log_likelihood"])
        return value if np.isfinite(value) else 0.0
    residual_norm = float(diagnostics.get("wrap_residual_norm", 0.0))
    jump_sigma = float(diagnostics.get("jump_diagnostic_sigma", 1.0))
    if not np.isfinite(residual_norm):
        return 0.0
    scale = max(jump_sigma, 1.0e-12)
    return -0.5 * (residual_norm / scale) ** 2


def _lookahead_score(branch: ActiveTimingBranch) -> float:
    diagnostics = branch.diagnostics or {}
    value = float(diagnostics.get("lookahead_log_likelihood", 0.0))
    return value if np.isfinite(value) else -1.0e12


def _proposal_log_prior(branch: ActiveTimingBranch) -> float:
    diagnostics = branch.diagnostics or {}
    if "proposal_log_prior" in diagnostics:
        value = float(diagnostics["proposal_log_prior"])
    elif _branch_family(branch) in {"local-wrap", "exploration-wrap"} and "diagnostic_log_likelihood" in diagnostics:
        value = float(diagnostics["diagnostic_log_likelihood"])
    elif "proposal_score" in diagnostics:
        value = float(diagnostics["proposal_score"])
    else:
        value = float(branch.log_prior)
    return value if np.isfinite(value) else -1.0e12


def _prior_calibration_group(branch: ActiveTimingBranch) -> str:
    return _branch_portfolio(branch) or _branch_family(branch)


def _calibrated_prior_scores(
    branches: Sequence[ActiveTimingBranch],
    mode: str,
) -> dict[str, float]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "raw":
        return {
            branch.branch_id: float(branch.log_prior)
            if np.isfinite(float(branch.log_prior))
            else -1.0e12
            for branch in branches
        }
    if normalized_mode == "proposal":
        return {branch.branch_id: _proposal_log_prior(branch) for branch in branches}
    if normalized_mode != "family-relative-proposal":
        raise ValueError(f"Unsupported prior_score_mode: {mode}")

    family_max: dict[str, float] = {}
    for branch in branches:
        family = _prior_calibration_group(branch)
        value = _proposal_log_prior(branch)
        family_max[family] = max(family_max.get(family, float("-inf")), value)
    return {
        branch.branch_id: _proposal_log_prior(branch) - family_max[_prior_calibration_group(branch)]
        for branch in branches
    }


def _feature_vector(branch: ActiveTimingBranch) -> np.ndarray:
    theta = np.atleast_1d(np.asarray(branch.theta_offset, dtype=float))
    theta_scale = max(float(np.max(np.abs(theta))), 1.0)
    theta = theta / theta_scale
    wraps = np.atleast_1d(np.asarray(branch.wraps, dtype=float))
    if wraps.size:
        wraps = wraps / max(float(np.max(np.abs(wraps))), 1.0)
    family_value = float(abs(hash(_branch_family(branch))) % 997) / 997.0
    return np.concatenate([theta, 0.5 * wraps, np.asarray([family_value], dtype=float)])


def _distance_to_selected(branch: ActiveTimingBranch, selected: Sequence[ActiveTimingBranch]) -> float:
    if not selected:
        return 1.0
    vector = _feature_vector(branch)
    distances = []
    for selected_branch in selected:
        selected_vector = _feature_vector(selected_branch)
        size = max(vector.size, selected_vector.size)
        left = np.zeros(size, dtype=float)
        right = np.zeros(size, dtype=float)
        left[: vector.size] = vector
        right[: selected_vector.size] = selected_vector
        distances.append(float(np.linalg.norm(left - right)))
    return min(distances)


def _base_score(
    branch: ActiveTimingBranch,
    certification: BranchCertification | None,
    config: BudgetedDecisionConfig,
    prior_score: float,
) -> float:
    return (
        float(config.quality_weight) * _safe_log_quality(certification)
        + float(config.diagnostic_weight) * _diagnostic_score(branch)
        + float(config.prior_weight) * float(prior_score)
        + float(config.lookahead_weight) * _lookahead_score(branch)
    )


def _uncertainty_scores(scored: Sequence[tuple[float, ActiveTimingBranch]]) -> dict[str, float]:
    if not scored:
        return {}
    finite_scores = [float(score) for score, _ in scored if np.isfinite(float(score))]
    if not finite_scores:
        return {branch.branch_id: 0.0 for _, branch in scored}
    best_score = max(finite_scores)
    relative_weights = {
        branch.branch_id: math.exp(max(min(float(score) - best_score, 0.0), -60.0))
        if np.isfinite(float(score))
        else 0.0
        for score, branch in scored
    }
    total = sum(relative_weights.values())
    if total <= 0.0:
        return {branch.branch_id: 0.0 for _, branch in scored}
    probabilities = {
        branch_id: value / total
        for branch_id, value in relative_weights.items()
    }
    max_entropy = math.log(max(len(probabilities), 2))
    normalized_entropy = {
        branch_id: -probability * math.log(max(probability, 1.0e-300)) / max_entropy
        for branch_id, probability in probabilities.items()
    }
    return {
        branch.branch_id: float(normalized_entropy.get(branch.branch_id, 0.0))
        for _, branch in scored
    }


def _max_family_count(config: BudgetedDecisionConfig, budget: int) -> int:
    if budget <= 0:
        return 0
    fraction = min(max(float(config.max_family_fraction), 0.0), 1.0)
    return max(int(math.ceil(fraction * budget)), max(int(config.family_floor), 1))


def _max_basin_count(config: BudgetedDecisionConfig, budget: int) -> int:
    if budget <= 0:
        return 0
    fraction = min(max(float(config.max_basin_fraction), 0.0), 1.0)
    return max(int(math.ceil(fraction * budget)), max(int(config.basin_floor), 1))


def select_budgeted_decision_branches(
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    config: BudgetedDecisionConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, object]]:
    active_config = config or BudgetedDecisionConfig()
    budget = max(int(active_config.budget), 0)
    if budget == 0 or not branches:
        return [], {
            "budget": budget,
            "quality_floor": 0,
            "quality_selected_count": 0,
            "quality_selected_branch_ids": [],
            "family_candidate_count": {},
            "family_selected_count": {},
            "portfolio_candidate_count": {},
            "portfolio_selected_count": {},
            "basin_candidate_count": {},
            "basin_selected_count": {},
            "candidate_scores": {},
        }

    certification_by_id = {certification.branch_id: certification for certification in certifications}
    calibrated_prior_by_id = _calibrated_prior_scores(
        branches,
        active_config.prior_score_mode,
    )
    scored = [
        (
            _base_score(
                branch,
                certification_by_id.get(branch.branch_id),
                active_config,
                calibrated_prior_by_id[branch.branch_id],
            ),
            branch,
        )
        for branch in branches
    ]
    scored.sort(key=lambda item: (-item[0], item[1].branch_id))
    uncertainty_by_id = _uncertainty_scores(scored)
    target = min(budget, len(scored))
    family_candidate_count: dict[str, int] = {}
    portfolio_candidate_count: dict[str, int] = {}
    basin_candidate_count: dict[str, int] = {}
    for _, branch in scored:
        family = _branch_family(branch)
        family_candidate_count[family] = family_candidate_count.get(family, 0) + 1
        portfolio = _branch_portfolio(branch)
        if portfolio is not None:
            portfolio_candidate_count[portfolio] = portfolio_candidate_count.get(portfolio, 0) + 1
        basin = _branch_basin(branch)
        basin_candidate_count[basin] = basin_candidate_count.get(basin, 0) + 1

    selected: list[ActiveTimingBranch] = []
    selected_ids: set[str] = set()
    family_selected_count: dict[str, int] = {}
    portfolio_selected_count: dict[str, int] = {}
    basin_selected_count: dict[str, int] = {}
    max_family_count = _max_family_count(active_config, target)
    max_basin_count = _max_basin_count(active_config, target)
    quality_selected_ids: list[str] = []

    def can_select(
        branch: ActiveTimingBranch,
        *,
        enforce_family: bool = True,
        enforce_basin: bool = True,
    ) -> bool:
        if branch.branch_id in selected_ids:
            return False
        family = _branch_family(branch)
        basin = _branch_basin(branch)
        if enforce_family and family_selected_count.get(family, 0) >= max_family_count:
            return False
        if enforce_basin and basin_selected_count.get(basin, 0) >= max_basin_count:
            return False
        return True

    def add_selected(branch: ActiveTimingBranch) -> None:
        selected.append(branch)
        selected_ids.add(branch.branch_id)
        family = _branch_family(branch)
        portfolio = _branch_portfolio(branch)
        basin = _branch_basin(branch)
        family_selected_count[family] = family_selected_count.get(family, 0) + 1
        if portfolio is not None:
            portfolio_selected_count[portfolio] = portfolio_selected_count.get(portfolio, 0) + 1
        basin_selected_count[basin] = basin_selected_count.get(basin, 0) + 1

    quality_floor = min(max(int(active_config.quality_floor), 0), target)
    if quality_floor > 0:
        quality_ranked = sorted(
            (branch for _, branch in scored),
            key=lambda branch: (
                -_safe_log_quality(certification_by_id.get(branch.branch_id)),
                branch.branch_id,
            ),
        )
        quality_basins: set[str] = set()
        for branch in quality_ranked:
            basin = _branch_basin(branch)
            if basin in quality_basins:
                continue
            add_selected(branch)
            quality_selected_ids.append(branch.branch_id)
            quality_basins.add(basin)
            if len(quality_selected_ids) >= quality_floor:
                break
        if len(quality_selected_ids) < quality_floor:
            for branch in quality_ranked:
                if branch.branch_id in selected_ids:
                    continue
                add_selected(branch)
                quality_selected_ids.append(branch.branch_id)
                if len(quality_selected_ids) >= quality_floor:
                    break

    family_floor = max(int(active_config.family_floor), 0)
    if family_floor > 0:
        families_by_best_score: list[tuple[float, str]] = []
        for family in sorted(family_candidate_count):
            best_score = next(
                score
                for score, branch in scored
                if _branch_family(branch) == family
            )
            families_by_best_score.append((best_score, family))
        families_by_best_score.sort(key=lambda item: (-item[0], item[1]))
        ranked_families = [family for _, family in families_by_best_score]
        for _ in range(family_floor):
            for family in ranked_families:
                if len(selected) >= target:
                    break
                if family_selected_count.get(family, 0) >= max_family_count:
                    continue
                family_best = next(
                    (
                        branch
                        for _, branch in scored
                        if _branch_family(branch) == family and can_select(branch)
                    ),
                    None,
                )
                if family_best is None:
                    continue
                add_selected(family_best)
            if len(selected) >= target:
                break

    portfolio_floor = max(int(active_config.portfolio_floor), 0)
    if portfolio_floor > 0 and portfolio_candidate_count:
        portfolios_by_best_score: list[tuple[float, str]] = []
        for portfolio in sorted(portfolio_candidate_count):
            best_score = next(
                score
                for score, branch in scored
                if _branch_portfolio(branch) == portfolio
            )
            portfolios_by_best_score.append((best_score, portfolio))
        portfolios_by_best_score.sort(key=lambda item: (-item[0], item[1]))
        ranked_portfolios = [portfolio for _, portfolio in portfolios_by_best_score]
        for floor_rank in range(portfolio_floor):
            for portfolio in ranked_portfolios:
                if len(selected) >= target:
                    break
                if portfolio_selected_count.get(portfolio, 0) > floor_rank:
                    continue
                portfolio_best = next(
                    (
                        branch
                        for _, branch in scored
                        if _branch_portfolio(branch) == portfolio and can_select(branch)
                    ),
                    None,
                )
                if portfolio_best is None:
                    continue
                add_selected(portfolio_best)
            if len(selected) >= target:
                break

    basin_floor = max(int(active_config.basin_floor), 0)
    if basin_floor > 0:
        basins_by_best_score: list[tuple[float, str]] = []
        for basin in sorted(basin_candidate_count):
            best_score = next(
                score
                for score, branch in scored
                if _branch_basin(branch) == basin
            )
            basins_by_best_score.append((best_score, basin))
        basins_by_best_score.sort(key=lambda item: (-item[0], item[1]))
        ranked_basins = [basin for _, basin in basins_by_best_score]
        for _ in range(basin_floor):
            for basin in ranked_basins:
                if len(selected) >= target:
                    break
                if basin_selected_count.get(basin, 0) >= max_basin_count:
                    continue
                basin_best = next(
                    (
                        branch
                        for _, branch in scored
                        if _branch_basin(branch) == basin and can_select(branch)
                    ),
                    None,
                )
                if basin_best is None:
                    continue
                add_selected(basin_best)
            if len(selected) >= target:
                break

    if not selected and scored and target > 0:
        _, best_base_branch = scored[0]
        add_selected(best_base_branch)

    while len(selected) < target:
        best_branch: ActiveTimingBranch | None = None
        best_score = float("-inf")
        for base_score, branch in scored:
            if not can_select(branch):
                continue
            diversity = _distance_to_selected(branch, selected)
            uncertainty = float(uncertainty_by_id.get(branch.branch_id, 0.0))
            score = (
                base_score
                + float(active_config.diversity_weight) * math.log1p(max(diversity, 0.0))
                + float(active_config.uncertainty_weight) * uncertainty
            )
            if score > best_score or (
                score == best_score
                and best_branch is not None
                and branch.branch_id < best_branch.branch_id
            ):
                best_branch = branch
                best_score = score
        if best_branch is None:
            for _, branch in scored:
                if can_select(branch, enforce_family=True, enforce_basin=False):
                    best_branch = branch
                    break
        if best_branch is None:
            for _, branch in scored:
                if can_select(branch, enforce_family=False, enforce_basin=False):
                    best_branch = branch
                    break
        if best_branch is None:
            break
        add_selected(best_branch)

    candidate_scores = {
        branch.branch_id: {
            "score": float(score),
            "acquisition_score": float(
                score
                + float(active_config.uncertainty_weight) * float(uncertainty_by_id.get(branch.branch_id, 0.0))
            ),
            "family": _branch_family(branch),
            "portfolio": _branch_portfolio(branch),
            "basin": _branch_basin(branch),
            "quality_score": float(_safe_log_quality(certification_by_id.get(branch.branch_id))),
            "diagnostic_score": float(_diagnostic_score(branch)),
            "uncertainty_score": float(uncertainty_by_id.get(branch.branch_id, 0.0)),
            "lookahead_score": float(_lookahead_score(branch)),
            "raw_log_prior": float(branch.log_prior),
            "proposal_log_prior": float(_proposal_log_prior(branch)),
            "calibrated_prior_score": float(calibrated_prior_by_id[branch.branch_id]),
            "log_prior": float(branch.log_prior),
        }
        for score, branch in scored
    }
    return selected, {
        "budget": int(budget),
        "prior_score_mode": str(active_config.prior_score_mode),
        "kept_count": len(selected),
        "quality_floor": int(quality_floor),
        "quality_selected_count": len(quality_selected_ids),
        "quality_selected_branch_ids": quality_selected_ids,
        "family_candidate_count": family_candidate_count,
        "family_selected_count": {
            family: int(family_selected_count.get(family, 0))
            for family in family_candidate_count
        },
        "portfolio_candidate_count": portfolio_candidate_count,
        "portfolio_selected_count": {
            portfolio: int(portfolio_selected_count.get(portfolio, 0))
            for portfolio in portfolio_candidate_count
        },
        "basin_candidate_count": basin_candidate_count,
        "basin_selected_count": {
            basin: int(basin_selected_count.get(basin, 0))
            for basin in basin_candidate_count
        },
        "max_basin_count": int(max_basin_count),
        "candidate_scores": candidate_scores,
    }
