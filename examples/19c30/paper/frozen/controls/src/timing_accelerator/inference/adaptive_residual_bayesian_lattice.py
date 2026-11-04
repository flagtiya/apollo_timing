from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.active_timing_contracts import (
    ActiveTimingBranch,
    BranchCertification,
)
from timing_accelerator.inference.alias_branch_certifier import _build_wrapped_toas
from timing_accelerator.inference.budgeted_bayesian_decision import (
    BudgetedDecisionConfig,
    select_budgeted_decision_branches,
)
from timing_accelerator.inference.branch_proposal import SEED_PULSE_NUMBER_PROPOSAL_SOURCES
from timing_accelerator.inference.cluster_manifest import load_cluster_manifest, resolve_cluster_manifest
from timing_accelerator.inference.pint_alias_geometry import (
    _compute_gap_phase_residuals,
    _parameter_phase_derivative,
)


def _set_model_parameter_value(model, parameter_name: str, value: float) -> None:
    name = str(parameter_name).upper()
    if name not in model.params:
        return
    _set_numeric_model_parameter_value(model, name, float(value))


@dataclass(frozen=True)
class AdaptiveResidualConfig:
    top_k: int = 16
    parent_top_k: int = 8
    ridge: float = 1.0e-4
    step_scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    eigen_step_scales: tuple[float, ...] = (-1.0, 1.0)
    max_eigen_directions: int = 2
    residual_noise_floor: float = 1.0e-3
    wrap_prior_scale: float = 200.0
    block_coordinate_groups: tuple[tuple[str, ...], ...] = ()
    block_step_scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    enable_gap_wrap_resampling: bool = False
    max_gap_wrap_shift: int = 2
    gap_wrap_shift_scales: tuple[float, ...] = (1.0,)
    gap_wrap_top_k: int = 4
    gap_wrap_min_abs_residual: float = 0.05
    gap_wrap_joint_ridge: float | None = None
    gap_wrap_joint_scale: float = 0.25
    combine_gap_wrap_with_continuous: bool = True
    decision_quality_weight: float = 1.0
    decision_diagnostic_weight: float = 0.5
    decision_diversity_weight: float = 0.35
    decision_prior_weight: float = 0.15
    decision_family_floor: int = 1
    decision_max_family_fraction: float = 0.65


@dataclass(frozen=True)
class ClusterResidualFingerprint:
    branch_id: str
    cluster_ids: tuple[str, ...]
    cluster_mjd: np.ndarray
    median_phase_residual: np.ndarray
    mad_phase_residual: np.ndarray
    toa_counts: np.ndarray
    gap_phase_residual: np.ndarray


def _score_certification(certification: BranchCertification) -> tuple[float, float, str]:
    chi2 = float(certification.reduced_chi2) if certification.reduced_chi2 is not None else float("inf")
    wrms = float(certification.weighted_rms_us) if certification.weighted_rms_us is not None else float("inf")
    return (chi2, wrms, str(certification.branch_id))


def _weighted_residual_regression(
    cluster_design_matrix: np.ndarray,
    residual: np.ndarray,
    residual_scale: np.ndarray,
    parameter_scales: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(cluster_design_matrix, dtype=float)
    target = -np.asarray(residual, dtype=float)
    scale = np.maximum(np.asarray(residual_scale, dtype=float), 1.0e-12)
    parameter_scale = np.maximum(np.asarray(parameter_scales, dtype=float), 1.0e-30)
    if design.ndim != 2:
        raise ValueError("cluster_design_matrix must be two-dimensional.")
    if target.shape != (design.shape[0],):
        raise ValueError("median_phase_residual length must match cluster_design_matrix rows.")
    if scale.shape != target.shape:
        raise ValueError("residual_scale length must match median_phase_residual.")
    if parameter_scale.shape != (design.shape[1],):
        raise ValueError("parameter_scales length must match cluster_design_matrix columns.")
    scaled_design = design * parameter_scale[None, :]
    weighted_design = scaled_design / scale[:, None]
    weighted_target = target / scale
    precision = weighted_design.T @ weighted_design + np.eye(design.shape[1], dtype=float) * max(float(ridge), 0.0)
    rhs = weighted_design.T @ weighted_target
    scaled_covariance = np.linalg.pinv(precision)
    scaled_delta = scaled_covariance @ rhs
    delta = scaled_delta * parameter_scale
    covariance = (scaled_covariance * parameter_scale[:, None]) * parameter_scale[None, :]
    return delta, covariance


def _wraps_from_offset(
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    theta_offset: np.ndarray,
) -> tuple[int, ...]:
    predicted = np.asarray(reference_phase_residuals, dtype=float) + np.asarray(gap_jacobian, dtype=float) @ theta_offset
    return tuple(int(item) for item in np.rint(predicted).astype(int).tolist())


@dataclass(frozen=True)
class _OffsetProposal:
    offset: np.ndarray
    family: str
    block_group: tuple[str, ...] = ()


@dataclass(frozen=True)
class _WrapProposal:
    shift: tuple[int, ...]
    family: str
    source_gap_indices: tuple[int, ...] = ()


def _candidate_offsets(
    delta_theta: np.ndarray,
    covariance: np.ndarray,
    theta_names: Sequence[str],
    config: AdaptiveResidualConfig,
) -> list[_OffsetProposal]:
    offsets: list[_OffsetProposal] = []
    for scale in config.step_scales:
        offsets.append(
            _OffsetProposal(
                offset=np.asarray(delta_theta, dtype=float) * float(scale),
                family="full-residual",
            )
        )
    if covariance.size:
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
        order = list(np.argsort(eigenvalues)[::-1])[: max(int(config.max_eigen_directions), 0)]
        for index in order:
            width = float(np.sqrt(max(float(eigenvalues[index]), 0.0)))
            if width <= 0.0:
                continue
            direction = np.asarray(eigenvectors[:, index], dtype=float) * width
            for scale in config.eigen_step_scales:
                offsets.append(
                    _OffsetProposal(
                        offset=np.asarray(delta_theta, dtype=float) + float(scale) * direction,
                        family="eigen-residual",
                    )
                )
    name_to_index = {str(name).upper(): index for index, name in enumerate(theta_names)}
    for group in config.block_coordinate_groups:
        indices = [
            name_to_index[str(name).upper()]
            for name in group
            if str(name).upper() in name_to_index
        ]
        if not indices:
            continue
        block_delta = np.zeros_like(np.asarray(delta_theta, dtype=float))
        block_delta[np.asarray(indices, dtype=int)] = np.asarray(delta_theta, dtype=float)[
            np.asarray(indices, dtype=int)
        ]
        for scale in config.block_step_scales:
            offsets.append(
                _OffsetProposal(
                    offset=block_delta * float(scale),
                    family="block-coordinate",
                    block_group=tuple(str(name).upper() for name in group),
                )
            )
    return offsets


def _gap_wrap_shift_vector(
    gap_phase_residual: np.ndarray,
    scale: float,
    max_abs_shift: int,
) -> np.ndarray:
    residual = np.asarray(gap_phase_residual, dtype=float)
    if residual.ndim != 1:
        raise ValueError("gap_phase_residual must be one-dimensional.")
    shift = np.rint(residual * float(scale)).astype(int)
    return np.clip(shift, -max(int(max_abs_shift), 0), max(int(max_abs_shift), 0))


def _signed_gap_shift(
    residual_value: float,
    *,
    scale: float,
    max_abs_shift: int,
) -> int:
    if not np.isfinite(float(residual_value)) or float(residual_value) == 0.0:
        return 0
    magnitude = max(1, int(np.rint(abs(float(residual_value) * float(scale)))))
    magnitude = min(magnitude, max(int(max_abs_shift), 0))
    return int(np.sign(float(residual_value))) * magnitude


def _gap_wrap_proposals(
    fingerprint: ClusterResidualFingerprint,
    parent_wraps: tuple[int, ...],
    config: AdaptiveResidualConfig,
) -> list[_WrapProposal]:
    if not config.enable_gap_wrap_resampling:
        return [_WrapProposal(shift=tuple(0 for _ in parent_wraps), family="continuous")]
    continuous = _WrapProposal(shift=tuple(0 for _ in parent_wraps), family="continuous")
    proposals: list[_WrapProposal] = []
    if len(parent_wraps) == 0:
        return [continuous]
    residual = np.asarray(fingerprint.gap_phase_residual, dtype=float)
    if residual.shape != (len(parent_wraps),):
        return [continuous]

    max_shift = max(int(config.max_gap_wrap_shift), 0)
    if max_shift <= 0:
        return [continuous]
    ranked_gap_indices = list(np.argsort(np.abs(residual))[::-1])[: max(int(config.gap_wrap_top_k), 0)]
    for scale in config.gap_wrap_shift_scales:
        shift = _gap_wrap_shift_vector(residual, float(scale), max_shift)
        if np.any(shift != 0):
            proposals.append(
                _WrapProposal(
                    shift=tuple(int(item) for item in shift.tolist()),
                    family="gap-wrap-resample",
                    source_gap_indices=tuple(int(item) for item in np.flatnonzero(shift).tolist()),
                )
            )
        for gap_index in ranked_gap_indices:
            single = np.zeros_like(shift)
            single_shift = int(shift[gap_index])
            if single_shift == 0 and abs(float(residual[gap_index])) >= float(config.gap_wrap_min_abs_residual):
                single_shift = _signed_gap_shift(
                    float(residual[gap_index]),
                    scale=float(scale),
                    max_abs_shift=max_shift,
                )
            single[gap_index] = single_shift
            if single[gap_index] == 0:
                continue
            proposals.append(
                _WrapProposal(
                    shift=tuple(int(item) for item in single.tolist()),
                    family="gap-wrap-resample",
                    source_gap_indices=(int(gap_index),),
                )
            )
    deduped: list[_WrapProposal] = []
    seen: set[tuple[int, ...]] = set()
    for proposal in proposals:
        if proposal.shift in seen:
            continue
        seen.add(proposal.shift)
        deduped.append(proposal)
    return deduped + [continuous]


def _shifted_wraps(parent_wraps: tuple[int, ...], proposal: _WrapProposal) -> tuple[int, ...]:
    if len(parent_wraps) != len(proposal.shift):
        raise ValueError("Gap wrap shift dimension must match parent wraps.")
    return tuple(int(wrap) + int(shift) for wrap, shift in zip(parent_wraps, proposal.shift, strict=True))


def _diagnostic_residual_from_wraps(
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    theta_offset: np.ndarray,
    wraps: tuple[int, ...],
) -> np.ndarray:
    predicted = np.asarray(reference_phase_residuals, dtype=float) + np.asarray(gap_jacobian, dtype=float) @ theta_offset
    return predicted - np.asarray(wraps, dtype=float)


_FAMILY_PRIORITY = {
    "gap-wrap-resample": 1.45,
    "gap-wrap-joint": 1.35,
    "block-coordinate": 1.15,
    "full-residual": 1.0,
    "eigen-residual": 0.9,
    "wrap-only": 0.85,
}


def _branch_family(branch: ActiveTimingBranch) -> str:
    diagnostics = branch.diagnostics or {}
    family = str(diagnostics.get("proposal_family", "full-residual"))
    return family.lower()


def uses_seed_pulse_number_reference(branch: ActiveTimingBranch) -> bool:
    return str(branch.proposal_source).lower() in SEED_PULSE_NUMBER_PROPOSAL_SOURCES


def _candidate_sort_key(branch: ActiveTimingBranch) -> tuple[float, float, float, int, int, str]:
    diagnostics = branch.diagnostics or {}
    family = _branch_family(branch)
    source_count = len(diagnostics.get("gap_wrap_source_indices", []))
    shift_l1 = sum(abs(int(item)) for item in diagnostics.get("gap_wrap_shift", []))
    diagnostic_strength = float(source_count) + 0.01 * float(shift_l1)
    return (
        -float(_FAMILY_PRIORITY.get(family, 1.0)),
        -diagnostic_strength,
        -float(diagnostics.get("proposal_score", branch.log_prior)),
        int(diagnostics.get("parent_rank", 0)),
        int(diagnostics.get("candidate_index", 0)),
        str(branch.branch_id),
    )


def _allocate_family_budgets(family_sizes: dict[str, int], total_budget: int) -> dict[str, int]:
    families = [family for family, size in family_sizes.items() if int(size) > 0]
    budget = max(int(total_budget), 0)
    if budget == 0 or not families:
        return {}
    if budget >= sum(int(size) for size in family_sizes.values()):
        return {family: int(family_sizes[family]) for family in families}
    if budget < len(families):
        ranked = sorted(
            families,
            key=lambda family: (
                float(_FAMILY_PRIORITY.get(family, 1.0)),
                int(family_sizes[family]),
                family,
            ),
            reverse=True,
        )
        return {family: (1 if family in ranked[:budget] else 0) for family in families}

    budgets = {family: 1 for family in families}
    remaining = budget - len(families)
    weights = {
        family: float(_FAMILY_PRIORITY.get(family, 1.0)) * math.sqrt(float(family_sizes[family]))
        for family in families
    }
    while remaining > 0:
        expandable = [
            family
            for family in families
            if budgets[family] < int(family_sizes[family])
        ]
        if not expandable:
            break
        chosen = max(
            expandable,
            key=lambda family: (
                weights[family] / float(budgets[family] + 1),
                weights[family],
                family,
            ),
        )
        budgets[chosen] += 1
        remaining -= 1
    return budgets


def _pseudo_certification_for_proposal(branch: ActiveTimingBranch) -> BranchCertification:
    diagnostics = branch.diagnostics or {}
    residual_norm = float(diagnostics.get("wrap_residual_norm", 0.0))
    if not np.isfinite(residual_norm):
        residual_norm = 1.0e6
    diagnostic_log_likelihood = float(diagnostics.get("diagnostic_log_likelihood", -0.5 * residual_norm * residual_norm))
    return BranchCertification(
        branch_id=branch.branch_id,
        log_likelihood=diagnostic_log_likelihood,
        fitted_theta={
            name: float(value)
            for name, value in zip(branch.theta_names, branch.theta_offset, strict=True)
        },
        covariance=np.eye(len(branch.theta_names), dtype=float),
        reduced_chi2=max(residual_norm * residual_norm, 0.0),
        weighted_rms_us=max(residual_norm, 0.0),
        backend="proposal-diagnostic",
        log_evidence=diagnostic_log_likelihood,
    )


def _select_budgeted_family_branches(
    branches: list[ActiveTimingBranch],
    top_k: int,
    config: AdaptiveResidualConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, object]]:
    budget = max(int(top_k), 0)
    if budget == 0 or not branches:
        return [], {
            "family_budget": {},
            "family_candidate_count": {},
            "family_selected_count": {},
        }
    if config is not None:
        selected, diagnostics = select_budgeted_decision_branches(
            branches,
            [_pseudo_certification_for_proposal(branch) for branch in branches],
            config=BudgetedDecisionConfig(
                budget=budget,
                family_floor=int(config.decision_family_floor),
                quality_weight=float(config.decision_quality_weight),
                diagnostic_weight=float(config.decision_diagnostic_weight),
                diversity_weight=float(config.decision_diversity_weight),
                prior_weight=float(config.decision_prior_weight),
                max_family_fraction=float(config.decision_max_family_fraction),
            ),
        )
        return selected, {
            "family_budget": diagnostics.get("family_selected_count", {}),
            "family_candidate_count": diagnostics.get("family_candidate_count", {}),
            "family_selected_count": diagnostics.get("family_selected_count", {}),
            "decision": diagnostics,
        }
    by_family: dict[str, list[ActiveTimingBranch]] = {}
    for branch in branches:
        by_family.setdefault(_branch_family(branch), []).append(branch)
    for family_branches in by_family.values():
        family_branches.sort(key=_candidate_sort_key)
    family_sizes = {family: len(family_branches) for family, family_branches in by_family.items()}
    family_budget = _allocate_family_budgets(family_sizes, budget)

    selected: list[ActiveTimingBranch] = []
    selected_ids: set[str] = set()
    for family in sorted(
        by_family,
            key=lambda item: _candidate_sort_key(by_family[item][0]),
    ):
        family_selected: list[ActiveTimingBranch] = []
        used_parents: set[str] = set()
        for branch in by_family[family]:
            parent_id = str((branch.diagnostics or {}).get("parent_branch_id", branch.branch_id))
            if parent_id in used_parents:
                continue
            family_selected.append(branch)
            used_parents.add(parent_id)
            if len(family_selected) >= int(family_budget.get(family, 0)):
                break
        if len(family_selected) < int(family_budget.get(family, 0)):
            family_selected_ids = {branch.branch_id for branch in family_selected}
            for branch in by_family[family]:
                if branch.branch_id in family_selected_ids:
                    continue
                family_selected.append(branch)
                family_selected_ids.add(branch.branch_id)
                if len(family_selected) >= int(family_budget.get(family, 0)):
                    break
        for branch in family_selected:
            if branch.branch_id in selected_ids:
                continue
            selected.append(branch)
            selected_ids.add(branch.branch_id)

    if len(selected) < budget:
        for branch in sorted(branches, key=_candidate_sort_key):
            if branch.branch_id in selected_ids:
                continue
            selected.append(branch)
            selected_ids.add(branch.branch_id)
            if len(selected) >= budget:
                break

    selected = sorted(selected, key=_candidate_sort_key)[:budget]
    return selected, {
        "family_budget": {family: int(value) for family, value in family_budget.items()},
        "family_candidate_count": {family: int(value) for family, value in family_sizes.items()},
        "family_selected_count": {
            family: int(sum(1 for branch in selected if _branch_family(branch) == family))
            for family in by_family
        },
    }


def _joint_gap_wrap_offset(
    *,
    parent_theta_offset: np.ndarray,
    target_wraps: tuple[int, ...],
    cluster_design_matrix: np.ndarray,
    median_phase_residual: np.ndarray,
    residual_scale: np.ndarray,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: AdaptiveResidualConfig,
) -> np.ndarray:
    parent_offset = np.asarray(parent_theta_offset, dtype=float)
    jacobian = np.asarray(gap_jacobian, dtype=float)
    parameter_scale = np.maximum(np.asarray(parameter_scales, dtype=float), 1.0e-30)
    if parent_offset.shape != parameter_scale.shape:
        raise ValueError("parent_theta_offset length must match parameter_scales.")

    design_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []

    cluster_design = np.asarray(cluster_design_matrix, dtype=float)
    cluster_target = -np.asarray(median_phase_residual, dtype=float)
    cluster_scale = np.maximum(np.asarray(residual_scale, dtype=float), 1.0e-12)
    if cluster_design.size:
        if cluster_design.ndim != 2 or cluster_design.shape[1] != parent_offset.size:
            raise ValueError("cluster_design_matrix shape must match parameter dimension.")
        if cluster_design.shape[0] != cluster_target.size or cluster_scale.shape != cluster_target.shape:
            raise ValueError("Cluster residual arrays must match cluster_design_matrix rows.")
        design_blocks.append(cluster_design / cluster_scale[:, None])
        target_blocks.append(cluster_target / cluster_scale)

    target = np.asarray(target_wraps, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float)
    if jacobian.size:
        if jacobian.ndim != 2 or jacobian.shape[1] != parent_offset.size:
            raise ValueError("gap_jacobian shape must match parameter dimension.")
        if target.shape != (jacobian.shape[0],) or reference.shape != target.shape:
            raise ValueError("target_wraps and reference_phase_residuals must match gap_jacobian rows.")
        gap_scale = np.full(target.shape, max(float(config.gap_wrap_joint_scale), 1.0e-12), dtype=float)
        current_prediction = reference + jacobian @ parent_offset
        design_blocks.append(jacobian / gap_scale[:, None])
        target_blocks.append((target - current_prediction) / gap_scale)

    if not design_blocks:
        return np.zeros_like(parent_offset)

    design = np.vstack(design_blocks)
    target_vector = np.concatenate(target_blocks)
    scaled_design = design * parameter_scale[None, :]
    ridge = float(config.gap_wrap_joint_ridge if config.gap_wrap_joint_ridge is not None else config.ridge)
    precision = scaled_design.T @ scaled_design + np.eye(parent_offset.size, dtype=float) * max(ridge, 0.0)
    rhs = scaled_design.T @ target_vector
    scaled_delta = np.linalg.pinv(precision) @ rhs
    return scaled_delta * parameter_scale


def generate_residual_guided_branches(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    fingerprints: Sequence[ClusterResidualFingerprint],
    theta_names: Sequence[str],
    cluster_design_matrix: np.ndarray,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: AdaptiveResidualConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, object]]:
    active_config = config or AdaptiveResidualConfig()
    theta_names_tuple = tuple(str(name).upper() for name in theta_names)
    parent_by_id = {branch.branch_id: branch for branch in parent_branches}
    fingerprint_by_id = {fingerprint.branch_id: fingerprint for fingerprint in fingerprints}
    ranked_certifications = sorted(certifications, key=_score_certification)[: max(int(active_config.parent_top_k), 1)]
    branches: list[ActiveTimingBranch] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()

    for parent_rank, certification in enumerate(ranked_certifications):
        parent = parent_by_id.get(certification.branch_id)
        fingerprint = fingerprint_by_id.get(certification.branch_id)
        if parent is None or fingerprint is None:
            continue
        residual_scale = np.maximum(
            np.asarray(fingerprint.mad_phase_residual, dtype=float),
            float(active_config.residual_noise_floor),
        )
        delta_theta, covariance = _weighted_residual_regression(
            cluster_design_matrix=np.asarray(cluster_design_matrix, dtype=float),
            residual=np.asarray(fingerprint.median_phase_residual, dtype=float),
            residual_scale=residual_scale,
            parameter_scales=np.asarray(parameter_scales, dtype=float),
            ridge=float(active_config.ridge),
        )
        offset_proposals = _candidate_offsets(delta_theta, covariance, theta_names_tuple, active_config)
        wrap_proposals = _gap_wrap_proposals(fingerprint, tuple(int(item) for item in parent.wraps), active_config)
        if active_config.enable_gap_wrap_resampling and (
            not active_config.combine_gap_wrap_with_continuous
            or not offset_proposals
        ):
            offset_proposals = [
                _OffsetProposal(
                    offset=np.zeros_like(np.asarray(parent.theta_offset, dtype=float)),
                    family="wrap-only",
                )
            ]
        candidate_index = 0
        zero_offset = np.zeros_like(np.asarray(parent.theta_offset, dtype=float))
        for wrap_proposal in wrap_proposals:
            if wrap_proposal.family == "gap-wrap-resample":
                target_wraps = _shifted_wraps(tuple(int(item) for item in parent.wraps), wrap_proposal)
                joint_offset = _joint_gap_wrap_offset(
                    parent_theta_offset=np.asarray(parent.theta_offset, dtype=float),
                    target_wraps=target_wraps,
                    cluster_design_matrix=np.asarray(cluster_design_matrix, dtype=float),
                    median_phase_residual=np.asarray(fingerprint.median_phase_residual, dtype=float),
                    residual_scale=residual_scale,
                    gap_jacobian=np.asarray(gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(reference_phase_residuals, dtype=float),
                    parameter_scales=np.asarray(parameter_scales, dtype=float),
                    config=active_config,
                )
                active_offset_proposals = [
                    _OffsetProposal(
                        offset=joint_offset,
                        family="gap-wrap-joint",
                    )
                ]
            else:
                active_offset_proposals = offset_proposals or [
                    _OffsetProposal(
                        offset=zero_offset,
                        family="wrap-only",
                    )
                ]
            for proposal in active_offset_proposals:
                offset_delta = np.asarray(proposal.offset, dtype=float)
                theta_offset = np.asarray(parent.theta_offset, dtype=float) + offset_delta
                if wrap_proposal.family == "continuous":
                    wraps = _wraps_from_offset(gap_jacobian, reference_phase_residuals, theta_offset)
                else:
                    wraps = _shifted_wraps(tuple(int(item) for item in parent.wraps), wrap_proposal)
                dedupe_key = (
                    wraps,
                    tuple(float(f"{value:.18e}") for value in theta_offset.tolist()),
                )
                if dedupe_key in seen:
                    candidate_index += 1
                    continue
                seen.add(dedupe_key)
                log_prior = (
                    float(parent.log_prior)
                    - 0.5 * float(np.dot(offset_delta, offset_delta))
                    - sum(abs(item) for item in wraps) / max(float(active_config.wrap_prior_scale), 1.0e-12)
                    - sum(abs(item) for item in wrap_proposal.shift) / max(float(active_config.wrap_prior_scale), 1.0e-12)
                )
                branch_id = f"arbl_{len(branches):04d}"
                predicted_wraps = (
                    np.asarray(reference_phase_residuals, dtype=float)
                    + np.asarray(gap_jacobian, dtype=float) @ theta_offset
                )
                residual = _diagnostic_residual_from_wraps(
                    gap_jacobian=gap_jacobian,
                    reference_phase_residuals=reference_phase_residuals,
                    theta_offset=theta_offset,
                    wraps=wraps,
                )
                residual_norm = float(np.linalg.norm(residual))
                proposal_family = (
                    "gap-wrap-resample"
                    if wrap_proposal.family == "gap-wrap-resample"
                    else proposal.family
                )
                diagnostic_sigma = max(float(active_config.gap_wrap_joint_scale), 1.0e-12)
                if wrap_proposal.family == "gap-wrap-resample":
                    diagnostic_log_likelihood = len(wrap_proposal.source_gap_indices) + 0.01 * sum(
                        abs(int(item)) for item in wrap_proposal.shift
                    )
                else:
                    diagnostic_log_likelihood = -0.5 * (residual_norm / diagnostic_sigma) ** 2
                branches.append(
                    ActiveTimingBranch(
                        branch_id=branch_id,
                        structure_id=parent.structure_id,
                        binary_model=parent.binary_model,
                        wraps=wraps,
                        theta_names=theta_names_tuple,
                        theta_offset=np.asarray(theta_offset, dtype=float),
                        log_prior=float(log_prior),
                        proposal_source="adaptive-residual-bayesian-lattice",
                        diagnostics={
                            "proposal_score": float(log_prior),
                            "parent_branch_id": str(parent.branch_id),
                            "parent_rank": int(parent_rank),
                            "residual_guided_delta_theta": np.asarray(delta_theta, dtype=float),
                            "residual_guided_offset_delta": np.asarray(offset_delta, dtype=float),
                            "proposal_family": proposal_family,
                            "continuous_proposal_family": proposal.family,
                            "block_coordinate_group": list(proposal.block_group),
                            "gap_wrap_shift": list(wrap_proposal.shift),
                            "gap_wrap_source_indices": list(wrap_proposal.source_gap_indices),
                            "candidate_index": int(candidate_index),
                            "predicted_wraps": predicted_wraps,
                            "wrap_residual_norm": residual_norm,
                            "diagnostic_log_likelihood": diagnostic_log_likelihood,
                            "jump_diagnostic_sigma": diagnostic_sigma,
                        },
                    )
                )
                candidate_index += 1

    selected_branches, budget_diagnostics = _select_budgeted_family_branches(
        branches,
        max(int(active_config.top_k), 0),
        active_config,
    )

    return selected_branches, {
        "proposal_source": "adaptive-residual-bayesian-lattice",
        "candidate_count": len(branches),
        "kept_count": len(selected_branches),
        "parent_count": len(ranked_certifications),
        "theta_names": list(theta_names_tuple),
        "block_coordinate_candidate_count": sum(
            1
            for branch in selected_branches
            if branch.diagnostics
            and branch.diagnostics.get("proposal_family") == "block-coordinate"
        ),
        "gap_wrap_candidate_count": sum(
            1
            for branch in selected_branches
            if branch.diagnostics
            and branch.diagnostics.get("proposal_family") == "gap-wrap-resample"
        ),
        **budget_diagnostics,
    }


def _cluster_median_design_matrix(
    *,
    model,
    toas,
    manifest,
    parameter_names: Sequence[str],
) -> np.ndarray:
    columns = []
    for parameter_name in parameter_names:
        derivative = _parameter_phase_derivative(model, toas, str(parameter_name).upper())
        rows = []
        for cluster in manifest.clusters:
            if not cluster.toa_indices:
                raise ValueError(f"Cluster '{cluster.cluster_id}' has no resolved TOA indices.")
            rows.append(float(np.median(derivative[np.asarray(cluster.toa_indices, dtype=int)])))
        columns.append(rows)
    if not columns:
        return np.zeros((len(manifest.clusters), 0), dtype=float)
    return np.asarray(columns, dtype=float).T


def _cluster_fingerprint_from_phase_residuals(
    *,
    branch_id: str,
    phase_residuals: np.ndarray,
    mjds: np.ndarray,
    manifest,
) -> ClusterResidualFingerprint:
    cluster_ids: list[str] = []
    cluster_mjd: list[float] = []
    median_phase_residual: list[float] = []
    mad_phase_residual: list[float] = []
    toa_counts: list[int] = []
    residuals = np.asarray(phase_residuals, dtype=float)
    for cluster in manifest.clusters:
        if not cluster.toa_indices:
            raise ValueError(f"Cluster '{cluster.cluster_id}' has no resolved TOA indices.")
        indices = np.asarray(cluster.toa_indices, dtype=int)
        cluster_residuals = residuals[indices]
        median = float(np.median(cluster_residuals))
        cluster_ids.append(str(cluster.cluster_id))
        cluster_mjd.append(float(np.mean(np.asarray(mjds, dtype=float)[indices])))
        median_phase_residual.append(median)
        mad_phase_residual.append(float(np.median(np.abs(cluster_residuals - median))))
        toa_counts.append(int(len(indices)))
    return ClusterResidualFingerprint(
        branch_id=str(branch_id),
        cluster_ids=tuple(cluster_ids),
        cluster_mjd=np.asarray(cluster_mjd, dtype=float),
        median_phase_residual=np.asarray(median_phase_residual, dtype=float),
        mad_phase_residual=np.asarray(mad_phase_residual, dtype=float),
        toa_counts=np.asarray(toa_counts, dtype=int),
        gap_phase_residual=_compute_gap_phase_residuals(residuals, manifest),
    )


def build_cluster_residual_fingerprints(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    theta_names: Sequence[str],
    clock_override_path: Path | None = None,
) -> tuple[list[ClusterResidualFingerprint], np.ndarray, dict[str, object]]:
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    manifest = resolve_cluster_manifest(load_cluster_manifest(Path(cluster_manifest_path)), tim_path)
    seed_model = _load_pint_model(par_path)
    base_toas = _load_pint_toas(
        tim_path,
        seed_model=seed_model,
        clock_override_path=clock_override_path,
    )
    cluster_design_matrix = _cluster_median_design_matrix(
        model=seed_model,
        toas=base_toas,
        manifest=manifest,
        parameter_names=theta_names,
    )
    certification_by_id = {item.branch_id: item for item in certifications}
    fingerprints: list[ClusterResidualFingerprint] = []
    mjds = np.asarray(base_toas.get_mjds().to_value(u.day), dtype=float)
    from pint.residuals import Residuals

    for branch in branches:
        certification = certification_by_id.get(branch.branch_id)
        if certification is None:
            continue
        model = deepcopy(seed_model)
        for parameter_name, fitted_value in certification.fitted_theta.items():
            _set_model_parameter_value(model, parameter_name, float(fitted_value))
        pulse_number_model = seed_model if uses_seed_pulse_number_reference(branch) else model
        wrapped_toas = _build_wrapped_toas(base_toas, pulse_number_model, manifest, branch.wraps)
        residuals = Residuals(
            wrapped_toas,
            model,
            use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        phase_residuals = np.asarray(residuals.phase_resids.to_value(u.dimensionless_unscaled), dtype=float)
        fingerprints.append(
            _cluster_fingerprint_from_phase_residuals(
                branch_id=branch.branch_id,
                phase_residuals=phase_residuals,
                mjds=mjds,
                manifest=manifest,
            )
        )
    return fingerprints, cluster_design_matrix, {
        "fingerprint_count": len(fingerprints),
        "cluster_count": len(manifest.clusters),
        "theta_names": [str(name).upper() for name in theta_names],
    }
