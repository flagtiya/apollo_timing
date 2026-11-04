from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from timing_accelerator.inference.active_timing_contracts import ActiveTimingBranch, BranchCertification
from timing_accelerator.inference.alias_branch_certifier import AliasBranchCertificationResult
from timing_accelerator.inference.alias_manifold_solver import AliasManifoldCandidate


SEED_PULSE_NUMBER_PROPOSAL_SOURCES = {
    "geometry-grid",
    "pint-geometry-grid",
    "bayesian-alias-lattice",
    "bayesian-marginal-integer-lattice",
    "adaptive-residual-bayesian-lattice",
    "staged-budgeted-bayesian-search",
    "apollo-path-first",
    "autonomous-uncertain-gap-resampler",
    "autonomous-global-restart-lattice",
    "autonomous-stratified-far-restart-lattice",
    "bayesian-gap-wrap-repair",
    "apollo-marginalized-session-lattice",
    "apollo-covariance-wrapped-session-mode",
    "apollo-exact-session-path-refinement",
    "apollo-bayesian-global-phase-path-refresh",
    "apollo-bayesian-information-ordered-continuation",
    "apollo-bayesian-posterior-particle-phase-path",
    "apollo-bayesian-wrap-invariant-global-phase-posterior",
}


@dataclass(frozen=True)
class BranchProposalConfig:
    max_wrap_residual: float = 0.25
    offset_prior_scale: float = 1.0
    model_log_prior: float = 0.0
    wrap_prior_scale: float = 8.0
    wrap_hypothesis_dimensions: int = 0
    wrap_hypothesis_radius: int = 1
    wrap_hypothesis_max_variants: int = 27
    top_k: int | None = None
    fallback_min_keep: int = 2
    proposal_source: str = "geometry-grid"


def _branch_id(index: int) -> str:
    return f"branch_{index:04d}"


def _source_candidate_index(branch: ActiveTimingBranch) -> int:
    if branch.diagnostics and "source_candidate_index" in branch.diagnostics:
        return int(branch.diagnostics["source_candidate_index"])
    if branch.branch_id.startswith("branch_"):
        try:
            return int(branch.branch_id.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0
    return 0


def _wrap_variant_index(branch: ActiveTimingBranch) -> int:
    if branch.diagnostics and "wrap_variant_index" in branch.diagnostics:
        return int(branch.diagnostics["wrap_variant_index"])
    return 0


def _certification_theta_offset(branch: ActiveTimingBranch) -> np.ndarray:
    if branch.diagnostics and "certification_theta_offset" in branch.diagnostics:
        return np.asarray(branch.diagnostics["certification_theta_offset"], dtype=float)
    return np.asarray(branch.theta_offset, dtype=float)


def _proposal_score(branch: ActiveTimingBranch) -> float:
    if branch.diagnostics and "proposal_score" in branch.diagnostics:
        return float(branch.diagnostics["proposal_score"])
    return float(branch.log_prior)


def _selection_sort_key(branch: ActiveTimingBranch) -> tuple[float, int, int, str]:
    return (-_proposal_score(branch), _source_candidate_index(branch), _wrap_variant_index(branch), branch.branch_id)


def _stable_output_sort_key(branch: ActiveTimingBranch) -> tuple[int, int, float, str]:
    return (_source_candidate_index(branch), _wrap_variant_index(branch), -_proposal_score(branch), branch.branch_id)


def _relabel_selected_branches(branches: list[ActiveTimingBranch]) -> list[ActiveTimingBranch]:
    relabeled: list[ActiveTimingBranch] = []
    for new_index, branch in enumerate(sorted(branches, key=_stable_output_sort_key)):
        diagnostics = dict(branch.diagnostics or {})
        diagnostics["source_candidate_index"] = _source_candidate_index(branch)
        diagnostics["wrap_variant_index"] = _wrap_variant_index(branch)
        diagnostics["selected_rank"] = new_index
        relabeled.append(
            replace(
                branch,
                branch_id=_branch_id(new_index),
                diagnostics=diagnostics,
            )
        )
    return relabeled


def _offset_log_prior(offset: np.ndarray, scale: float) -> float:
    active_scale = max(float(scale), 1.0e-12)
    return -0.5 * float(np.dot(offset, offset)) / (active_scale * active_scale)


def _wrap_log_prior(wraps: np.ndarray, scale: float) -> float:
    active_scale = max(float(scale), 1.0e-12)
    return -float(np.sum(np.abs(wraps))) / active_scale


def _wrap_hypotheses(
    predicted_wraps: np.ndarray,
    config: BranchProposalConfig,
) -> list[tuple[np.ndarray, float, int]]:
    nearest_wraps = np.rint(predicted_wraps).astype(int)
    if nearest_wraps.size == 0:
        return [(nearest_wraps, 0.0, 0)]

    max_dimensions = min(int(config.wrap_hypothesis_dimensions), int(nearest_wraps.size))
    radius = max(int(config.wrap_hypothesis_radius), 0)
    if max_dimensions <= 0 or radius <= 0:
        residual = predicted_wraps - nearest_wraps
        return [(nearest_wraps, float(np.linalg.norm(residual)), 0)]

    ambiguity = np.abs(predicted_wraps - nearest_wraps)
    selected_dimensions = sorted(
        range(int(nearest_wraps.size)),
        key=lambda index: (-float(ambiguity[index]), index),
    )[:max_dimensions]

    hypotheses: list[tuple[np.ndarray, float, int]] = []
    max_variants = max(int(config.wrap_hypothesis_max_variants), 1)
    for variant_index, delta_combo in enumerate(product(range(-radius, radius + 1), repeat=len(selected_dimensions))):
        wraps = np.asarray(nearest_wraps, dtype=int).copy()
        for dimension_index, delta in zip(selected_dimensions, delta_combo, strict=True):
            wraps[dimension_index] += int(delta)
        residual = predicted_wraps - wraps
        hypotheses.append((wraps, float(np.linalg.norm(residual)), variant_index))
        if len(hypotheses) >= max_variants:
            break
    return hypotheses


def _build_candidate(
    structure_id: str,
    binary_model: str | None,
    theta_names: list[str],
    offset: np.ndarray,
    predicted_wraps: np.ndarray,
    wraps: np.ndarray,
    residual_norm: float,
    source_index: int,
    wrap_variant_index: int,
    config: BranchProposalConfig,
) -> ActiveTimingBranch:
    log_prior = (
        float(config.model_log_prior)
        + _offset_log_prior(offset, config.offset_prior_scale)
        + _wrap_log_prior(wraps, config.wrap_prior_scale)
    )
    proposal_score = float(log_prior) - 0.5 * (
        float(residual_norm) / max(float(config.max_wrap_residual), 1.0e-12)
    ) ** 2
    return ActiveTimingBranch(
        branch_id=f"{_branch_id(source_index)}_w{wrap_variant_index:03d}",
        structure_id=str(structure_id),
        binary_model=binary_model.upper() if binary_model is not None else None,
        wraps=tuple(int(item) for item in wraps.tolist()),
        theta_names=tuple(str(name).upper() for name in theta_names),
        theta_offset=np.asarray(offset, dtype=float),
        log_prior=float(log_prior),
        proposal_source=config.proposal_source,
        diagnostics={
            "source_candidate_index": int(source_index),
            "wrap_variant_index": int(wrap_variant_index),
            "wrap_residual_norm": float(residual_norm),
            "predicted_wraps": np.asarray(predicted_wraps, dtype=float),
            "proposal_score": float(proposal_score),
            "passed_residual_gate": float(residual_norm) <= float(config.max_wrap_residual),
        },
    )


def _select_candidates(
    scored_candidates: list[tuple[float, ActiveTimingBranch]],
    config: BranchProposalConfig,
) -> list[ActiveTimingBranch]:
    if not scored_candidates:
        return []

    if config.top_k is not None:
        return [
            candidate
            for _, candidate in sorted(scored_candidates, key=lambda item: _selection_sort_key(item[1]))[
                : max(int(config.top_k), 0)
            ]
        ]

    passing_candidates = [
        candidate
        for _, candidate in scored_candidates
        if candidate.diagnostics and candidate.diagnostics.get("passed_residual_gate")
    ]
    if passing_candidates:
        return sorted(passing_candidates, key=_stable_output_sort_key)

    keep_count = max(int(config.fallback_min_keep), 1)
    return [
        candidate
        for _, candidate in sorted(scored_candidates, key=lambda item: _selection_sort_key(item[1]))[
            : min(keep_count, len(scored_candidates))
        ]
    ]


def generate_active_timing_branches(
    structure_id: str,
    binary_model: str | None,
    theta_names: list[str],
    parameter_offset_grid: list[np.ndarray],
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray | None = None,
    config: BranchProposalConfig | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, float | int | bool | list[int | None]]]:
    active_config = config or BranchProposalConfig()
    jacobian = np.asarray(gap_jacobian, dtype=float)
    reference = (
        np.zeros(jacobian.shape[0], dtype=float)
        if reference_phase_residuals is None
        else np.asarray(reference_phase_residuals, dtype=float)
    )
    scored_candidates: list[tuple[float, ActiveTimingBranch]] = []
    residual_norms: list[float] = []
    rejected = 0

    for source_index, raw_offset in enumerate(parameter_offset_grid):
        offset = np.asarray(raw_offset, dtype=float)
        predicted_wraps = reference + (jacobian @ offset if jacobian.size else np.zeros(0, dtype=float))
        for wraps, residual_norm, variant_index in _wrap_hypotheses(predicted_wraps, active_config):
            residual_norms.append(float(residual_norm))
            if float(residual_norm) > float(active_config.max_wrap_residual):
                rejected += 1
            candidate = _build_candidate(
                structure_id=structure_id,
                binary_model=binary_model,
                theta_names=theta_names,
                offset=offset,
                predicted_wraps=predicted_wraps,
                wraps=wraps,
                residual_norm=float(residual_norm),
                source_index=source_index,
                wrap_variant_index=variant_index,
                config=active_config,
            )
            scored_candidates.append((_proposal_score(candidate), candidate))

    selected = _select_candidates(scored_candidates, active_config)
    branches = _relabel_selected_branches(selected)
    any_passing = any(
        candidate.diagnostics and candidate.diagnostics.get("passed_residual_gate")
        for _, candidate in scored_candidates
    )
    diagnostics = {
        "candidate_count": len(parameter_offset_grid),
        "hypothesis_count": len(scored_candidates),
        "kept_count": len(branches),
        "rejected_count": rejected,
        "max_wrap_residual": float(max(residual_norms)) if residual_norms else 0.0,
        "fallback_used": bool(not any_passing),
        "selected_source_indices": [
            int(branch.diagnostics["source_candidate_index"])
            if branch.diagnostics and "source_candidate_index" in branch.diagnostics
            else None
            for branch in branches
        ],
        "selected_wrap_variants": [
            int(branch.diagnostics["wrap_variant_index"])
            if branch.diagnostics and "wrap_variant_index" in branch.diagnostics
            else None
            for branch in branches
        ],
    }
    return branches, diagnostics


def branch_to_alias_candidate(branch: ActiveTimingBranch) -> AliasManifoldCandidate:
    residual = 0.0
    if branch.diagnostics and "wrap_residual_norm" in branch.diagnostics:
        residual = float(branch.diagnostics["wrap_residual_norm"])
    return AliasManifoldCandidate(
        candidate_id=branch.branch_id,
        parameter_offset=_certification_theta_offset(branch),
        wraps=tuple(int(item) for item in branch.wraps),
        wrap_residual_norm=residual,
        log_prior=float(branch.log_prior),
        pulse_number_reference=(
            "seed"
            if len(branch.wraps) > 0
            and str(branch.proposal_source).lower() in SEED_PULSE_NUMBER_PROPOSAL_SOURCES
            else "candidate"
        ),
    )


def alias_certification_to_branch_certification(
    result: AliasBranchCertificationResult,
    backend: str = "local-laplace",
) -> BranchCertification:
    return BranchCertification(
        branch_id=result.candidate_id,
        log_likelihood=float(result.log_likelihood),
        fitted_theta={str(key).upper(): float(value) for key, value in result.fitted_parameters.items()},
        covariance=np.asarray(result.covariance, dtype=float),
        reduced_chi2=float(result.reduced_chi2),
        weighted_rms_us=float(result.weighted_rms_us),
        backend=backend,
        chi2=float(result.chi2),
        notes=list(result.notes),
    )
