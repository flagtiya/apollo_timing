from __future__ import annotations

import math
import time
from dataclasses import replace

from timing_accelerator.backend.certify import certify_candidates
from timing_accelerator.backend.pint_backend import refine_candidate_with_pint
from timing_accelerator.models.contracts import (
    AutomatedFinalStructureScore,
    AutomaticFinalStructureConfig,
    BayesianReleaseScore,
    BootstrapRequest,
    Candidate,
    ExactSearchRoundSummary,
    FailureState,
    FinalBlockFollowupStep,
    FinalBlockModelScore,
    PosteriorSummary,
    PosteriorUnlockState,
    RecoveryStageConfig,
    RecoveryStageResult,
    SimulationConfig,
)
from timing_accelerator.search.candidate_ranker import rank_candidates
from timing_accelerator.search.hard_param_grid import generate_candidates


def _correlation_group_confidence(
    group: list[str],
    posterior_summary: PosteriorSummary,
) -> float:
    if not posterior_summary.parameter_correlations:
        return 0.0
    pairwise: list[float] = []
    for left_index, left_name in enumerate(group):
        for right_name in group[left_index + 1 :]:
            correlation = posterior_summary.parameter_correlations.get(f"{left_name}:{right_name}")
            if correlation is None:
                correlation = posterior_summary.parameter_correlations.get(f"{right_name}:{left_name}")
            if correlation is not None:
                pairwise.append(abs(correlation))
    if not pairwise:
        return 0.0
    return max(pairwise)


def _select_posterior_group(
    stage: RecoveryStageConfig,
    prior_posterior: PosteriorSummary,
) -> tuple[list[str] | None, float]:
    if not prior_posterior.recommended_joint_parameters:
        return None, 0.0

    current_names = set(stage.hard_params or stage.fit_params)
    ranked_groups: list[tuple[float, list[str]]] = []
    for group in prior_posterior.recommended_joint_parameters:
        if stage.max_posterior_group_size is not None and len(group) > stage.max_posterior_group_size:
            continue
        confidence = _correlation_group_confidence(group, prior_posterior)
        overlap = bool(current_names.intersection(group)) if current_names else True
        if overlap or not current_names:
            ranked_groups.append((confidence, group))

    if not ranked_groups:
        return None, 0.0

    ranked_groups.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    confidence, group = ranked_groups[0]
    return list(group), confidence


def plan_recovery_stage(
    stage: RecoveryStageConfig,
    prior_posterior: PosteriorSummary | None,
    posterior_unlock_state: PosteriorUnlockState,
) -> tuple[RecoveryStageConfig, PosteriorSummary | None, list[FailureState]]:
    planned_stage = replace(stage)
    failures: list[FailureState] = []
    planned_stage.planning_source = "manual-config"
    planned_stage.planning_confidence = 1.0

    if not stage.use_posterior_guidance:
        return planned_stage, None, failures

    if stage.require_trusted_posterior_unlock and posterior_unlock_state.status != "unlocked":
        planned_stage.planning_source = "deterministic-fallback"
        planned_stage.planning_confidence = 0.0
        failures.append(
            FailureState(
                stage=stage.stage_id,
                code="posterior-guidance-locked",
                message=(
                    "Posterior guidance was requested before a trusted local-fit posterior unlock "
                    f"was established. reason={posterior_unlock_state.reason}"
                ),
                suggestion="Keep deterministic grouping until a trusted local-fit stage unlocks posterior guidance.",
            )
        )
        return planned_stage, None, failures

    if (
        prior_posterior is None
        or prior_posterior.status != "usable"
        or not prior_posterior.parameter_sigmas
    ):
        planned_stage.planning_source = "deterministic-fallback"
        planned_stage.planning_confidence = 0.0
        failures.append(
            FailureState(
                stage=stage.stage_id,
                code="posterior-unavailable",
                message="Posterior guidance was requested but no usable posterior summary was available.",
                suggestion="Keep the deterministic grid or insert a local-fit stage before guided refinement.",
            )
        )
        return planned_stage, None, failures

    rewrite_applied = False
    confidence = 0.0
    if stage.rewrite_hard_params_from_posterior:
        selected_group, confidence = _select_posterior_group(stage, prior_posterior)
        if selected_group is not None and confidence >= stage.planning_confidence_threshold:
            planned_stage.hard_params = selected_group
            rewrite_applied = True
    if stage.rewrite_fit_params_from_posterior:
        selected_group, group_confidence = _select_posterior_group(stage, prior_posterior)
        confidence = max(confidence, group_confidence)
        if selected_group is not None and group_confidence >= stage.planning_confidence_threshold:
            planned_stage.fit_params = selected_group
            rewrite_applied = True

    if rewrite_applied:
        planned_stage.planning_source = "posterior-rewrite"
        planned_stage.planning_confidence = confidence
        return planned_stage, PosteriorSummary(
            status=prior_posterior.status,
            source_stage=prior_posterior.source_stage,
            guidance_strategy="posterior-rewrite",
            parameter_sigmas=prior_posterior.parameter_sigmas,
            parameter_correlations=prior_posterior.parameter_correlations,
            condition_number=prior_posterior.condition_number,
            adapted_grid_offsets=prior_posterior.adapted_grid_offsets,
            recommended_joint_parameters=prior_posterior.recommended_joint_parameters,
            message=prior_posterior.message,
        ), failures

    planned_stage.planning_source = "deterministic-fallback"
    planned_stage.planning_confidence = confidence
    failures.append(
        FailureState(
            stage=stage.stage_id,
            code="posterior-grouping-untrusted",
            message="Posterior guidance did not meet the confidence threshold for stage rewriting.",
            suggestion="Keep deterministic grouping or lower the grouping threshold after validating correlations.",
        )
    )
    return planned_stage, None, failures


def _apply_basin_shaping_center_offsets(
    center_parameters: dict[str, float],
    stage: RecoveryStageConfig,
) -> dict[str, float]:
    shaped_center = dict(center_parameters)
    for name, offset in (stage.basin_shaping_center_offsets or {}).items():
        if name in shaped_center:
            shaped_center[name] = shaped_center[name] + offset
    return shaped_center


def _build_posterior_unlock_state(
    previous_state: PosteriorUnlockState,
    stage: RecoveryStageConfig,
    posterior_summary: PosteriorSummary,
    failures: list[FailureState],
) -> tuple[PosteriorUnlockState, list[FailureState]]:
    blocking_failures = {failure.code for failure in failures if failure.code.startswith("local-fit-")}
    if blocking_failures:
        return previous_state, failures

    if posterior_summary.status != "usable":
        return (
            previous_state,
            failures
            + [
                FailureState(
                    stage=stage.stage_id,
                    code="posterior-untrusted-for-guidance",
                    message=(
                        "Local fit did not establish a trusted posterior unlock because the posterior "
                        f"summary was not usable. status={posterior_summary.status}"
                    ),
                    suggestion="Reduce the free-parameter set or improve basin entry before using posterior guidance.",
                )
            ],
        )

    condition_number = posterior_summary.condition_number
    max_condition_number = stage.posterior_unlock_max_condition_number
    if (
        max_condition_number is not None
        and (
            condition_number is None
            or not math.isfinite(condition_number)
            or condition_number > max_condition_number
        )
    ):
        return (
            previous_state,
            failures
            + [
                FailureState(
                    stage=stage.stage_id,
                    code="posterior-untrusted-for-guidance",
                    message=(
                        "Local fit produced a posterior summary, but it did not meet the trusted unlock "
                        "conditioning threshold: "
                        f"condition_number={condition_number}, threshold={max_condition_number}"
                    ),
                    suggestion="Tighten the basin or reduce fit dimensionality before enabling posterior-guided rewrites.",
                )
            ],
        )

    return (
        PosteriorUnlockState(
            status="unlocked",
            unlocked_by_stage=stage.stage_id,
            reason="Trusted local-fit posterior unlock established.",
        ),
        failures,
    )


def _current_released_fit_params(
    stage: RecoveryStageConfig,
    candidate: Candidate,
) -> list[str]:
    released = list(stage.fit_params)
    if candidate.fitted_parameters:
        for name in candidate.fitted_parameters:
            if name not in candidate.hard_parameters and name not in released:
                released.append(name)
    return released


def _coupling_strength_for_block(
    block: list[str],
    posterior_summary: PosteriorSummary,
) -> float:
    if not posterior_summary.parameter_correlations:
        return 0.0
    pairwise: list[float] = []
    for left_index, left_name in enumerate(block):
        for right_name in block[left_index + 1 :]:
            correlation = posterior_summary.parameter_correlations.get(f"{left_name}:{right_name}")
            if correlation is None:
                correlation = posterior_summary.parameter_correlations.get(f"{right_name}:{left_name}")
            if correlation is not None:
                pairwise.append(abs(correlation))
    return max(pairwise) if pairwise else 0.0


def _prior_width_for_parameter(request: BootstrapRequest, name: str) -> float:
    if name in request.simulation.grid_offsets:
        return max(abs(offset) for offset in request.simulation.grid_offsets[name]) or 0.01
    value = request.initial_parameters.get(name, 1.0)
    return max(abs(value) * 0.05, 0.01)


def _bayesian_release_candidate_scores(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
    prior_posterior: PosteriorSummary | None,
) -> tuple[list[BayesianReleaseScore], Candidate | None, list[str] | None, list[FailureState]]:
    if not stage.bayesian_release_blocks:
        return [], None, None, []

    base_fit_params = _current_released_fit_params(stage, current_candidate)
    base_chi2 = (
        current_candidate.fit_chi2
        if current_candidate.fit_chi2 is not None
        else (-current_candidate.exact_score if current_candidate.exact_score is not None else None)
    )
    if base_chi2 is None:
        return [], None, None, [
            FailureState(
                stage=stage.stage_id,
                code="bayesian-release-base-unavailable",
                message="Bayesian release policy requires a current candidate with fit or certification quality.",
                suggestion="Insert a local-fit or exact-certification stage before Bayesian release.",
            )
        ]

    heuristic_preferred_block = (
        list(stage.bayesian_release_fallback_order[0])
        if stage.bayesian_release_fallback_order
        else None
    )

    scored_candidates: list[tuple[BayesianReleaseScore, Candidate]] = []
    failures: list[FailureState] = []
    for block in stage.bayesian_release_blocks:
        if all(name in base_fit_params for name in block):
            continue
        trial_fit_params = list(dict.fromkeys(base_fit_params + block))
        trial_candidate, trial_posterior, trial_failures, _ = refine_candidate_with_pint(
            request,
            current_candidate,
            fit_params=trial_fit_params,
            stage_id=f"{stage.stage_id}-bayes-{'-'.join(name.lower() for name in block)}",
            quality_gate_max_reduced_chi2=stage.quality_gate_max_reduced_chi2,
            quality_gate_max_weighted_rms_us=stage.quality_gate_max_weighted_rms_us,
        )
        trial_fail_codes = {failure.code for failure in trial_failures if failure.code.startswith("local-fit-")}
        trial_chi2 = trial_candidate.fit_chi2 if trial_candidate.fit_chi2 is not None else base_chi2
        log_likelihood_gain = 0.5 * (base_chi2 - trial_chi2)
        log_prior_volume = 0.0
        log_posterior_volume = 0.0
        for name in block:
            prior_width = max(_prior_width_for_parameter(request, name), 1.0e-30)
            log_prior_volume += math.log(prior_width)
            sigma = None
            if trial_posterior.parameter_sigmas:
                sigma = trial_posterior.parameter_sigmas.get(name)
            sigma = max(abs(sigma), 1.0e-30) if sigma is not None else prior_width
            log_posterior_volume += math.log(sigma)
        laplace_constant_term = 0.5 * len(block) * math.log(2.0 * math.pi)
        approximate_log_bayes_factor = (
            log_likelihood_gain
            + laplace_constant_term
            + log_posterior_volume
            - log_prior_volume
        )
        coupling_strength = (
            _coupling_strength_for_block(block, prior_posterior)
            if prior_posterior is not None
            else 0.0
        )
        condition_penalty = 0.0
        if trial_posterior.condition_number is not None and trial_posterior.condition_number > 1.0e8:
            condition_penalty = stage.bayesian_release_complexity_penalty_scale * math.log10(
                trial_posterior.condition_number / 1.0e8 + 1.0
            )
        heuristic_prior_bonus = 0.0
        if heuristic_preferred_block is not None and block == heuristic_preferred_block:
            heuristic_prior_bonus = 0.25
        evidence_score = (
            approximate_log_bayes_factor
            + 0.25 * coupling_strength
            + heuristic_prior_bonus
            - condition_penalty
        )
        if trial_fail_codes:
            evidence_score -= 10.0
        score = BayesianReleaseScore(
            block=list(block),
            approximate_log_bayes_factor=approximate_log_bayes_factor,
            log_likelihood_gain=log_likelihood_gain,
            log_prior_volume=log_prior_volume,
            log_posterior_volume=log_posterior_volume,
            laplace_constant_term=laplace_constant_term,
            coupling_strength=coupling_strength,
            condition_penalty=condition_penalty,
            heuristic_prior_bonus=heuristic_prior_bonus,
            evidence_score=evidence_score,
        )
        scored_candidates.append((score, trial_candidate))

    if not scored_candidates:
        return [], None, None, failures

    scored_candidates.sort(key=lambda item: item[0].evidence_score, reverse=True)
    selected_score, selected_candidate = scored_candidates[0]
    selected_score.selected = True

    selected_block: list[str] | None = None
    if selected_score.evidence_score >= stage.bayesian_release_score_threshold:
        selected_block = selected_score.block
    elif stage.bayesian_release_fallback_order:
        selected_score.selected = False
        for block in stage.bayesian_release_fallback_order:
            if any(name not in base_fit_params for name in block):
                selected_block = list(block)
                break
        if selected_block is not None:
            for score, candidate in scored_candidates:
                if score.block == selected_block:
                    score.selected = True
                    score.fallback_used = True
                    selected_score = score
                    selected_candidate = candidate
                    break

    if selected_block is None:
        failures.append(
            FailureState(
                stage=stage.stage_id,
                code="bayesian-release-not-supported",
                message="No candidate release block met the Bayesian release threshold and no fallback order remained.",
                suggestion="Lower the threshold or improve the current posterior before releasing more parameters.",
            )
        )
        return [item[0] for item in scored_candidates], None, None, failures

    return [item[0] for item in scored_candidates], selected_candidate, selected_block, failures


def _final_block_marginalized_candidate_scores(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
) -> tuple[list[FinalBlockModelScore], Candidate | None, list[str] | None, list[FailureState]]:
    if not stage.final_block_core_blocks:
        return [], None, None, []

    base_chi2 = (
        current_candidate.fit_chi2
        if current_candidate.fit_chi2 is not None
        else (-current_candidate.exact_score if current_candidate.exact_score is not None else None)
    )
    if base_chi2 is None:
        return [], None, None, [
            FailureState(
                stage=stage.stage_id,
                code="final-block-base-unavailable",
                message="Final-block marginalized comparison requires a candidate with an established local-fit quality baseline.",
                suggestion="Insert a local-fit precondition stage before final-block model comparison.",
            )
        ]

    marginalized_params = list(stage.final_block_marginalized_params or [])
    scored_candidates: list[tuple[FinalBlockModelScore, Candidate]] = []
    failures: list[FailureState] = []
    for core_block in stage.final_block_core_blocks:
        trial_fit_params = list(dict.fromkeys(list(core_block) + marginalized_params))
        trial_candidate, trial_posterior, trial_failures, _ = refine_candidate_with_pint(
            request,
            current_candidate,
            fit_params=trial_fit_params,
            stage_id=f"{stage.stage_id}-final-block-{'-'.join(name.lower() for name in core_block)}",
            quality_gate_max_reduced_chi2=None,
            quality_gate_max_weighted_rms_us=None,
            local_solver_mode=stage.local_solver_mode,
            local_solver_max_iterations=stage.local_solver_max_iterations,
            local_solver_initial_damping=stage.local_solver_initial_damping,
            local_solver_damping_multiplier=stage.local_solver_damping_multiplier,
            local_solver_min_fractional_improvement=stage.local_solver_min_fractional_improvement,
        )
        trial_fail_codes = {failure.code for failure in trial_failures if failure.code.startswith("local-fit-")}
        trial_chi2 = trial_candidate.fit_chi2 if trial_candidate.fit_chi2 is not None else base_chi2
        log_likelihood_gain = 0.5 * (base_chi2 - trial_chi2)
        log_prior_volume = 0.0
        log_posterior_volume = 0.0
        for name in core_block:
            prior_width = max(_prior_width_for_parameter(request, name), 1.0e-30)
            log_prior_volume += math.log(prior_width)
            sigma = None
            if trial_posterior.parameter_sigmas:
                sigma = trial_posterior.parameter_sigmas.get(name)
            sigma = max(abs(sigma), 1.0e-30) if sigma is not None else prior_width
            log_posterior_volume += math.log(sigma)
        approximate_log_bayes_factor = log_likelihood_gain + log_posterior_volume - log_prior_volume
        condition_penalty = 0.0
        if trial_posterior.condition_number is not None and trial_posterior.condition_number > 1.0e12:
            condition_penalty = math.log10(trial_posterior.condition_number / 1.0e12 + 1.0)
        evidence_score = approximate_log_bayes_factor - condition_penalty
        if trial_fail_codes:
            evidence_score -= 10.0
        score = FinalBlockModelScore(
            core_block=list(core_block),
            marginalized_params=marginalized_params,
            approximate_log_bayes_factor=approximate_log_bayes_factor,
            log_likelihood_gain=log_likelihood_gain,
            log_prior_volume=log_prior_volume,
            log_posterior_volume=log_posterior_volume,
            condition_penalty=condition_penalty,
            evidence_score=evidence_score,
        )
        scored_candidates.append((score, trial_candidate))

    if not scored_candidates:
        return [], None, None, failures

    scored_candidates.sort(key=lambda item: item[0].evidence_score, reverse=True)
    selected_score, selected_candidate = scored_candidates[0]
    selected_score.selected = True
    selected_block: list[str] | None = None
    if selected_score.evidence_score >= stage.final_block_score_threshold:
        selected_block = list(selected_score.core_block)
    elif stage.final_block_fallback_block:
        selected_score.selected = False
        selected_block = list(stage.final_block_fallback_block)
        for score, candidate in scored_candidates:
            if score.core_block == selected_block:
                score.selected = True
                score.fallback_used = True
                selected_score = score
                selected_candidate = candidate
                break

    if selected_block is None:
        failures.append(
            FailureState(
                stage=stage.stage_id,
                code="final-block-not-supported",
                message="No final core block met the marginalized Bayesian threshold and no fallback block was configured.",
                suggestion="Lower the final-block threshold or add a fallback core block.",
            )
        )
        return [item[0] for item in scored_candidates], None, None, failures

    return [item[0] for item in scored_candidates], selected_candidate, selected_block, failures


def _run_final_block_followup_sequence(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
    current_fit_params: list[str],
) -> tuple[Candidate, list[str], list[FinalBlockFollowupStep], list[FailureState]]:
    accepted_candidate = current_candidate
    accepted_fit_params = list(current_fit_params)
    steps: list[FinalBlockFollowupStep] = []
    failures: list[FailureState] = []

    ordered_steps: list[tuple[str, list[str]]] = []
    for block in stage.final_block_nuisance_groups or []:
        ordered_steps.append(("nuisance", list(block)))
    for block in stage.final_block_expansion_blocks or []:
        ordered_steps.append(("expansion", list(block)))

    for step_kind, block in ordered_steps:
        trial_fit_params = list(dict.fromkeys(accepted_fit_params + block))
        trial_candidate, _, trial_failures, trial_metrics = refine_candidate_with_pint(
            request,
            accepted_candidate,
            fit_params=trial_fit_params,
            stage_id=f"{stage.stage_id}-{step_kind}-{'-'.join(name.lower() for name in block)}",
            quality_gate_max_reduced_chi2=None,
            quality_gate_max_weighted_rms_us=None,
            local_solver_mode=stage.local_solver_mode,
            local_solver_max_iterations=stage.local_solver_max_iterations,
            local_solver_initial_damping=stage.local_solver_initial_damping,
            local_solver_damping_multiplier=stage.local_solver_damping_multiplier,
            local_solver_min_fractional_improvement=stage.local_solver_min_fractional_improvement,
        )
        blocking = any(failure.code.startswith("local-fit-") for failure in trial_failures)
        step_result = FinalBlockFollowupStep(
            step_kind=step_kind,
            block=list(block),
            cumulative_fit_params=trial_fit_params,
            fit_reduced_chi2=trial_metrics.get("fit_reduced_chi2"),
            weighted_rms_us=trial_metrics.get("weighted_rms_us"),
            accepted=not blocking,
        )
        steps.append(step_result)
        if blocking:
            failures.extend(trial_failures)
            break
        accepted_candidate = trial_candidate
        accepted_fit_params = trial_fit_params

    return accepted_candidate, accepted_fit_params, steps, failures


def _evaluate_automatic_final_structure(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
    structure: AutomaticFinalStructureConfig,
) -> tuple[AutomatedFinalStructureScore, Candidate, list[FinalBlockFollowupStep], list[FailureState]]:
    base_chi2 = (
        current_candidate.fit_chi2
        if current_candidate.fit_chi2 is not None
        else (-current_candidate.exact_score if current_candidate.exact_score is not None else None)
    )
    if base_chi2 is None:
        raise ValueError("Automatic final-structure inference requires a candidate with fit quality.")

    core_fit_params = list(dict.fromkeys(list(structure.core_block) + list(structure.marginalized_params)))
    core_candidate, core_posterior, core_failures, _ = refine_candidate_with_pint(
        request,
        current_candidate,
        fit_params=core_fit_params,
        stage_id=f"{stage.stage_id}-auto-{structure.structure_id}",
        quality_gate_max_reduced_chi2=None,
        quality_gate_max_weighted_rms_us=None,
        local_solver_mode=stage.local_solver_mode,
        local_solver_max_iterations=stage.local_solver_max_iterations,
        local_solver_initial_damping=stage.local_solver_initial_damping,
        local_solver_damping_multiplier=stage.local_solver_damping_multiplier,
        local_solver_min_fractional_improvement=stage.local_solver_min_fractional_improvement,
    )

    working_stage = replace(
        stage,
        final_block_nuisance_groups=structure.nuisance_groups,
        final_block_expansion_blocks=structure.expansion_blocks,
    )
    sequence_candidate, _, followup_steps, followup_failures = _run_final_block_followup_sequence(
        request,
        working_stage,
        core_candidate,
        core_fit_params,
    )
    final_candidate = sequence_candidate
    final_chi2 = final_candidate.fit_chi2 if final_candidate.fit_chi2 is not None else (
        core_candidate.fit_chi2 if core_candidate.fit_chi2 is not None else base_chi2
    )
    log_likelihood_gain = 0.5 * (base_chi2 - final_chi2)
    log_prior_volume = 0.0
    log_posterior_volume = 0.0
    for name in structure.core_block:
        prior_width = max(_prior_width_for_parameter(request, name), 1.0e-30)
        log_prior_volume += math.log(prior_width)
        sigma = None
        if core_posterior.parameter_sigmas:
            sigma = core_posterior.parameter_sigmas.get(name)
        sigma = max(abs(sigma), 1.0e-30) if sigma is not None else prior_width
        log_posterior_volume += math.log(sigma)
    complexity_penalty = 0.1 * sum(len(block) for block in structure.nuisance_groups or [])
    complexity_penalty += 0.1 * sum(len(block) for block in structure.expansion_blocks or [])
    if core_posterior.condition_number is not None and core_posterior.condition_number > 1.0e12:
        complexity_penalty += math.log10(core_posterior.condition_number / 1.0e12 + 1.0)
    score = AutomatedFinalStructureScore(
        structure_id=structure.structure_id,
        core_block=list(structure.core_block),
        marginalized_params=list(structure.marginalized_params),
        nuisance_groups=structure.nuisance_groups,
        expansion_blocks=structure.expansion_blocks,
        approximate_log_bayes_factor=log_likelihood_gain + log_posterior_volume - log_prior_volume,
        log_likelihood_gain=log_likelihood_gain,
        log_prior_volume=log_prior_volume,
        log_posterior_volume=log_posterior_volume,
        complexity_penalty=complexity_penalty,
        evidence_score=log_likelihood_gain + log_posterior_volume - log_prior_volume - complexity_penalty,
    )
    combined_failures = core_failures + followup_failures
    if any(failure.code.startswith("local-fit-") for failure in combined_failures):
        score.evidence_score -= 10.0
    return score, final_candidate, followup_steps, combined_failures


def _automatic_final_structure_inference(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
) -> tuple[list[AutomatedFinalStructureScore], Candidate | None, str | None, list[str] | None, list[FinalBlockFollowupStep] | None, list[FailureState]]:
    if not stage.automatic_final_structures:
        return [], None, None, None, None, []

    scored_candidates: list[tuple[AutomatedFinalStructureScore, Candidate, list[FinalBlockFollowupStep], list[FailureState]]] = []
    failures: list[FailureState] = []
    for structure in stage.automatic_final_structures:
        score, candidate, followup_steps, structure_failures = _evaluate_automatic_final_structure(
            request=request,
            stage=stage,
            current_candidate=current_candidate,
            structure=structure,
        )
        scored_candidates.append((score, candidate, followup_steps, structure_failures))

    if not scored_candidates:
        return [], None, None, None, None, failures

    scored_candidates.sort(key=lambda item: item[0].evidence_score, reverse=True)
    selected_score, selected_candidate, selected_followup_steps, selected_failures = scored_candidates[0]
    selected_score.selected = True
    failures.extend(selected_failures)
    return (
        [item[0] for item in scored_candidates],
        selected_candidate,
        selected_score.structure_id,
        selected_score.core_block,
        selected_followup_steps or None,
        failures,
    )


def build_recovery_stages(request: BootstrapRequest) -> list[RecoveryStageConfig]:
    configured_stages = request.simulation.recovery_stages
    if configured_stages:
        return configured_stages

    stages = [
        RecoveryStageConfig(
            stage_id="default-coarse-search",
            kind="coarse-search",
            hard_params=list(request.simulation.hard_params),
            top_k=request.simulation.top_k,
        )
    ]
    if request.simulation.post_certification_fit_params:
        stages.append(
            RecoveryStageConfig(
                stage_id="local-fit",
                kind="local-fit",
                fit_params=list(request.simulation.post_certification_fit_params),
                full_solution_stage=True,
            )
        )
    return stages


def _stage_request(
    request: BootstrapRequest,
    simulation: SimulationConfig,
) -> BootstrapRequest:
    return BootstrapRequest(
        par_path=request.par_path,
        tim_path=request.tim_path,
        simulation_config_path=request.simulation_config_path,
        initial_parameters=request.initial_parameters,
        toa_count=request.toa_count,
        simulation=simulation,
    )


def _merge_candidate_parameters(
    base_candidate: Candidate,
    generated_candidates: list[Candidate],
) -> list[Candidate]:
    merged: list[Candidate] = []
    for candidate in generated_candidates:
        hard_parameters = dict(base_candidate.hard_parameters)
        hard_parameters.update(candidate.hard_parameters)
        merged.append(
            Candidate(
                candidate_id=candidate.candidate_id,
                hard_parameters=hard_parameters,
            )
        )
    return merged


def _build_stage_center(request: BootstrapRequest, current_candidate: Candidate) -> dict[str, float]:
    center = {
        name: value
        for name, value in request.initial_parameters.items()
        if name in request.simulation.hard_params or name in current_candidate.hard_parameters
    }
    center.update(current_candidate.hard_parameters)
    return center


def _build_adapted_grid_offsets(
    stage: RecoveryStageConfig,
    request: BootstrapRequest,
    prior_posterior: PosteriorSummary | None,
) -> tuple[dict[str, list[float]], str, PosteriorSummary | None, list[FailureState]]:
    base_offsets = dict(request.simulation.grid_offsets)
    if stage.grid_offsets:
        base_offsets.update(stage.grid_offsets)

    if not stage.use_posterior_guidance:
        return base_offsets, "deterministic-grid", None, []
    if stage.planning_source == "deterministic-fallback":
        return base_offsets, "deterministic-fallback", None, []

    if (
        prior_posterior is None
        or prior_posterior.status != "usable"
        or not prior_posterior.parameter_sigmas
    ):
        return base_offsets, "deterministic-fallback", None, [
            FailureState(
                stage=stage.stage_id,
                code="posterior-unavailable",
                message="Posterior guidance was requested but no usable posterior summary was available.",
                suggestion="Keep the deterministic grid or insert a local-fit stage before guided refinement.",
            )
        ]

    adapted_offsets: dict[str, list[float]] = {}
    for name in stage.hard_params:
        original_offsets = base_offsets.get(name, [0.0])
        sigma = prior_posterior.parameter_sigmas.get(name)
        if sigma is None:
            adapted_offsets[name] = list(original_offsets)
            continue
        max_abs_offset = max(abs(offset) for offset in original_offsets) or 1.0
        target_half_width = max(3.0 * abs(sigma), max_abs_offset * 0.05)
        scale = min(1.0, target_half_width / max_abs_offset)
        adapted_offsets[name] = [offset * scale for offset in original_offsets]

    return adapted_offsets, "posterior-guided", PosteriorSummary(
        status=prior_posterior.status,
        source_stage=prior_posterior.source_stage,
        guidance_strategy="posterior-guided",
        parameter_sigmas=prior_posterior.parameter_sigmas,
        parameter_correlations=prior_posterior.parameter_correlations,
        condition_number=prior_posterior.condition_number,
        adapted_grid_offsets=adapted_offsets,
        recommended_joint_parameters=prior_posterior.recommended_joint_parameters,
        message=prior_posterior.message,
    ), []


def _run_coarse_search_stage(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
    prior_posterior: PosteriorSummary | None,
    posterior_unlock_state: PosteriorUnlockState,
) -> tuple[RecoveryStageResult, Candidate, list[Candidate], list[Candidate], PosteriorSummary | None, dict[str, float]]:
    planned_stage, planning_posterior, planning_failures = plan_recovery_stage(
        stage,
        prior_posterior,
        posterior_unlock_state,
    )
    center_parameters = _build_stage_center(request, current_candidate)
    for name in planned_stage.hard_params:
        if name not in center_parameters and name in request.initial_parameters:
            center_parameters[name] = request.initial_parameters[name]
    shaped_center_parameters = _apply_basin_shaping_center_offsets(center_parameters, planned_stage)
    grid_offsets, guidance_strategy, stage_posterior, stage_failures = _build_adapted_grid_offsets(
        planned_stage,
        request,
        prior_posterior,
    )
    combined_failures = planning_failures + stage_failures
    stage_start = time.perf_counter()
    stage_top_k = planned_stage.top_k or request.simulation.top_k
    metrics: dict[str, float] = {}
    exact_search_rounds: list[ExactSearchRoundSummary] | None = None

    if request.simulation.regime == "full-fit" and request.simulation.certification_backend == "pint":
        hard_params = planned_stage.hard_params or request.simulation.hard_params
        base_grid_offsets = grid_offsets

        def _dedupe(candidate_items: list[Candidate]) -> list[Candidate]:
            deduped: dict[tuple[tuple[str, float], ...], Candidate] = {}
            for item in candidate_items:
                key = tuple(sorted((name, float(value)) for name, value in item.hard_parameters.items()))
                deduped[key] = item
            return list(deduped.values())

        def _candidate_hits_edge(candidate_item: Candidate, pool: list[Candidate]) -> bool:
            for name in hard_params:
                values = [cand.hard_parameters[name] for cand in pool if name in cand.hard_parameters]
                if not values:
                    continue
                value = candidate_item.hard_parameters.get(name)
                if value is None:
                    continue
                if value <= min(values) or value >= max(values):
                    return True
            return False

        def _scaled_offsets(scale_factor: float) -> dict[str, list[float]]:
            return {
                name: [offset * scale_factor for offset in offsets]
                for name, offsets in base_grid_offsets.items()
            }

        def _exact_rank(candidate_items: list[Candidate]) -> tuple[list[Candidate], dict[str, float]]:
            stage_simulation = replace(
                request.simulation,
                top_k=len(candidate_items),
                certification_fit_params=list(planned_stage.fit_params or request.simulation.certification_fit_params or []),
                post_certification_fit_params=None,
            )
            certification = certify_candidates(
                _stage_request(request, stage_simulation),
                candidate_items,
            )
            ranked_items = sorted(
                certification.candidates,
                key=lambda item: item.exact_score if item.exact_score is not None else float("-inf"),
                reverse=True,
            )
            for candidate_item in ranked_items:
                candidate_item.basin_score = (
                    candidate_item.exact_score
                    if candidate_item.exact_score is not None
                    else candidate_item.approximate_score
                )
            return ranked_items, (certification.metrics or {})

        round_ranked_candidates: list[Candidate] = []
        all_generated_candidates: list[Candidate] = []
        aggregated_ranked: dict[tuple[tuple[str, float], ...], Candidate] = {}
        exact_search_rounds = []

        initial_generated = generate_candidates(
            request,
            hard_params=hard_params,
            center_parameters=shaped_center_parameters,
            grid_offsets=base_grid_offsets,
            candidate_prefix=f"{planned_stage.stage_id}-r0",
        )
        round_candidates = _merge_candidate_parameters(current_candidate, initial_generated)
        round_candidates = _dedupe(round_candidates)
        round_ranked_candidates, round_metrics = _exact_rank(round_candidates)
        metrics.update(round_metrics)
        all_generated_candidates.extend(round_candidates)
        for item in round_ranked_candidates:
            key = tuple(sorted((name, float(value)) for name, value in item.hard_parameters.items()))
            aggregated_ranked[key] = item
        current_scale_factor = 1.0
        exact_search_rounds.append(
            ExactSearchRoundSummary(
                round_index=0,
                scale_factor=current_scale_factor,
                candidate_count=len(round_candidates),
                retained_candidate_count=min(stage_top_k, len(round_ranked_candidates)),
                best_candidate_id=round_ranked_candidates[0].candidate_id if round_ranked_candidates else None,
                best_exact_score=round_ranked_candidates[0].exact_score if round_ranked_candidates else None,
                edge_hit=_candidate_hits_edge(round_ranked_candidates[0], round_candidates) if round_ranked_candidates else False,
            )
        )

        for round_index in range(1, planned_stage.adaptive_exact_search_rounds + 1):
            seed_candidates = round_ranked_candidates[: max(1, planned_stage.adaptive_exact_search_branching)]
            next_round_candidates: list[Candidate] = []
            round_edge_hit = False
            next_scale_factor = current_scale_factor
            for seed in seed_candidates:
                seed_edge_hit = _candidate_hits_edge(seed, round_candidates)
                round_edge_hit = round_edge_hit or seed_edge_hit
                seed_scale_factor = current_scale_factor * (
                    planned_stage.adaptive_exact_search_expand_edge_factor
                    if seed_edge_hit
                    else planned_stage.adaptive_exact_search_shrink_factor
                )
                next_scale_factor = seed_scale_factor
                generated = generate_candidates(
                    request,
                    hard_params=hard_params,
                    center_parameters=_build_stage_center(request, seed),
                    grid_offsets=_scaled_offsets(seed_scale_factor),
                    candidate_prefix=f"{planned_stage.stage_id}-r{round_index}",
                )
                next_round_candidates.extend(_merge_candidate_parameters(seed, generated))
            round_candidates = _dedupe(next_round_candidates)
            round_ranked_candidates, round_metrics = _exact_rank(round_candidates)
            for key, value in round_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value
            all_generated_candidates.extend(round_candidates)
            for item in round_ranked_candidates:
                key = tuple(sorted((name, float(value)) for name, value in item.hard_parameters.items()))
                aggregated_ranked[key] = item
            current_scale_factor = next_scale_factor
            exact_search_rounds.append(
                ExactSearchRoundSummary(
                    round_index=round_index,
                    scale_factor=current_scale_factor,
                    candidate_count=len(round_candidates),
                    retained_candidate_count=min(stage_top_k, len(round_ranked_candidates)),
                    best_candidate_id=round_ranked_candidates[0].candidate_id if round_ranked_candidates else None,
                    best_exact_score=round_ranked_candidates[0].exact_score if round_ranked_candidates else None,
                    edge_hit=round_edge_hit,
                )
            )

        ranked_candidates = sorted(
            aggregated_ranked.values(),
            key=lambda item: item.exact_score if item.exact_score is not None else float("-inf"),
            reverse=True,
        )
        retained_candidates = ranked_candidates[:stage_top_k]
        selected_candidate = ranked_candidates[0] if ranked_candidates else current_candidate
        generated_candidates = all_generated_candidates
        guidance_strategy = f"{guidance_strategy}+adaptive-exact-pint-ranking"
    else:
        generated_stage_candidates = generate_candidates(
            request,
            hard_params=planned_stage.hard_params or request.simulation.hard_params,
            center_parameters=shaped_center_parameters,
            grid_offsets=grid_offsets,
            candidate_prefix=planned_stage.stage_id,
        )
        generated_candidates = _merge_candidate_parameters(current_candidate, generated_stage_candidates)
        ranked_candidates = rank_candidates(
            request,
            generated_candidates,
            center_parameters=shaped_center_parameters,
            posterior_summary=prior_posterior if guidance_strategy == "posterior-guided" else None,
            basin_shaping_rank_weights=planned_stage.basin_shaping_rank_weights,
        )
        retained_candidates = ranked_candidates[:stage_top_k]

        stage_simulation = replace(
            request.simulation,
            top_k=stage_top_k,
            certification_fit_params=list(request.simulation.certification_fit_params or []),
            post_certification_fit_params=None,
        )
        certification = certify_candidates(
            _stage_request(request, stage_simulation),
            retained_candidates,
        )
        for candidate in certification.candidates:
            candidate.basin_score = (
                candidate.exact_score
                if candidate.exact_score is not None
                else candidate.approximate_score
            )
        selected_candidate = certification.best_candidate
        if certification.metrics:
            metrics.update(certification.metrics)
    stage_wall_time = time.perf_counter() - stage_start
    metrics[f"stage_{stage.stage_id}_wall_time_seconds"] = stage_wall_time

    stage_result = RecoveryStageResult(
        stage_id=stage.stage_id,
        kind=stage.kind,
        planning_source=planned_stage.planning_source,
        planning_confidence=planned_stage.planning_confidence,
        original_hard_params=list(stage.hard_params),
        original_fit_params=list(stage.fit_params),
        planned_hard_params=list(planned_stage.hard_params),
        planned_fit_params=list(planned_stage.fit_params),
        input_candidate_id=current_candidate.candidate_id,
        input_hard_parameters=dict(current_candidate.hard_parameters),
        candidate_count=len(generated_candidates),
        retained_candidate_count=len(retained_candidates),
        exact_search_rounds=exact_search_rounds,
        selected_candidate=selected_candidate,
        wall_time_seconds=stage_wall_time,
        guidance_strategy=guidance_strategy,
        basin_shaping_center_offsets=planned_stage.basin_shaping_center_offsets,
        basin_shaping_rank_weights=planned_stage.basin_shaping_rank_weights,
        posterior_unlock_state=posterior_unlock_state.status,
        posterior_unlock_reason=posterior_unlock_state.reason,
        posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
        posterior_summary=planning_posterior or stage_posterior,
        failures=combined_failures or None,
    )
    return (
        stage_result,
        selected_candidate,
        generated_candidates,
        retained_candidates,
        prior_posterior,
        metrics,
    )


def _run_local_fit_stage(
    request: BootstrapRequest,
    stage: RecoveryStageConfig,
    current_candidate: Candidate,
    prior_posterior: PosteriorSummary | None,
    posterior_unlock_state: PosteriorUnlockState,
) -> tuple[RecoveryStageResult, Candidate, PosteriorSummary | None, PosteriorUnlockState, dict[str, float]]:
    planned_stage, planning_posterior, planning_failures = plan_recovery_stage(
        stage,
        prior_posterior,
        posterior_unlock_state,
    )
    basin_score_threshold = planned_stage.basin_score_threshold
    current_basin_score = (
        current_candidate.basin_score
        if current_candidate.basin_score is not None
        else current_candidate.exact_score
    )
    if (
        basin_score_threshold is not None
        and (current_basin_score is None or current_basin_score < basin_score_threshold)
    ):
        failure = FailureState(
            stage=stage.stage_id,
            code="basin-entry-not-established",
            message=(
                "Local timing fit was skipped because basin entry was not established: "
                f"score={current_basin_score}, threshold={basin_score_threshold}"
            ),
            suggestion="Broaden grouped coarse search or improve basin-entry scoring before unlocking local fit.",
        )
        current_candidate.basin_score = current_basin_score
        return (
            RecoveryStageResult(
                stage_id=stage.stage_id,
                kind=stage.kind,
                local_solver_mode=planned_stage.local_solver_mode,
                planning_source=planned_stage.planning_source,
                planning_confidence=planned_stage.planning_confidence,
                original_hard_params=list(stage.hard_params),
                original_fit_params=list(stage.fit_params),
                planned_hard_params=list(planned_stage.hard_params),
                planned_fit_params=list(planned_stage.fit_params),
                input_candidate_id=current_candidate.candidate_id,
                input_hard_parameters=dict(current_candidate.hard_parameters),
                selected_candidate=current_candidate,
                guidance_strategy="basin-entry-gated",
                basin_score_threshold=basin_score_threshold,
                basin_entry_established=False,
                full_solution_stage=planned_stage.full_solution_stage,
                full_solution_accepted=False if planned_stage.full_solution_stage else None,
                posterior_unlock_state=posterior_unlock_state.status,
                posterior_unlock_reason=posterior_unlock_state.reason,
                posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
                failures=(planning_failures + [failure]) or None,
                posterior_summary=planning_posterior,
            ),
            current_candidate,
            planning_posterior,
            posterior_unlock_state,
            {f"stage_{stage.stage_id}_wall_time_seconds": 0.0},
        )

    if request.simulation.certification_backend != "pint":
        failure = FailureState(
            stage=stage.stage_id,
            code="local-fit-backend-unsupported",
            message="Local low-level timing fit is only implemented for the PINT backend.",
            suggestion="Use the PINT backend or remove the local-fit stage for mock validation runs.",
        )
        return (
            RecoveryStageResult(
                stage_id=stage.stage_id,
                kind=stage.kind,
                local_solver_mode=planned_stage.local_solver_mode,
                planning_source=planned_stage.planning_source,
                planning_confidence=planned_stage.planning_confidence,
                original_hard_params=list(stage.hard_params),
                original_fit_params=list(stage.fit_params),
                planned_hard_params=list(planned_stage.hard_params),
                planned_fit_params=list(planned_stage.fit_params),
                input_candidate_id=current_candidate.candidate_id,
                input_hard_parameters=dict(current_candidate.hard_parameters),
                selected_candidate=current_candidate,
                guidance_strategy="deterministic-fallback",
                basin_score_threshold=basin_score_threshold,
                basin_entry_established=True if basin_score_threshold is not None else None,
                full_solution_stage=planned_stage.full_solution_stage,
                full_solution_accepted=False if planned_stage.full_solution_stage else None,
                posterior_unlock_state=posterior_unlock_state.status,
                posterior_unlock_reason=posterior_unlock_state.reason,
                posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
                failures=(planning_failures + [failure]) or None,
                posterior_summary=planning_posterior,
            ),
            current_candidate,
            None,
            posterior_unlock_state,
            {f"stage_{stage.stage_id}_wall_time_seconds": 0.0},
        )

    bayesian_release_scores: list[BayesianReleaseScore] | None = None
    bayesian_release_selected_block: list[str] | None = None
    bayesian_release_heuristic_block: list[str] | None = None
    final_block_scores: list[FinalBlockModelScore] | None = None
    final_block_selected_core_block: list[str] | None = None
    final_block_followup_steps: list[FinalBlockFollowupStep] | None = None
    automated_final_structure_scores: list[AutomatedFinalStructureScore] | None = None
    automated_final_structure_selected_id: str | None = None
    followup_failures: list[FailureState] = []
    effective_fit_params = planned_stage.fit_params or list(request.simulation.post_certification_fit_params or [])
    if planned_stage.bayesian_release_blocks:
        bayesian_release_heuristic_block = (
            list(planned_stage.bayesian_release_fallback_order[0])
            if planned_stage.bayesian_release_fallback_order
            else None
        )
        (
            bayesian_release_scores,
            selected_release_candidate,
            bayesian_release_selected_block,
            release_failures,
        ) = _bayesian_release_candidate_scores(
            request,
            planned_stage,
            current_candidate,
            prior_posterior,
        )
        if release_failures:
            return (
                RecoveryStageResult(
                stage_id=stage.stage_id,
                kind=stage.kind,
                local_solver_mode=planned_stage.local_solver_mode,
                planning_source="bayesian-release-policy",
                planning_confidence=None,
                original_hard_params=list(stage.hard_params),
                original_fit_params=list(stage.fit_params),
                planned_hard_params=list(planned_stage.hard_params),
                    planned_fit_params=list(effective_fit_params),
                    input_candidate_id=current_candidate.candidate_id,
                    input_hard_parameters=dict(current_candidate.hard_parameters),
                    selected_candidate=current_candidate,
                    wall_time_seconds=0.0,
                    weighted_rms_us=None,
                    basin_score_threshold=basin_score_threshold,
                    basin_entry_established=True if basin_score_threshold is not None else None,
                    full_solution_stage=planned_stage.full_solution_stage,
                    full_solution_accepted=False if planned_stage.full_solution_stage else None,
                    bayesian_release_selected_block=bayesian_release_selected_block,
                    bayesian_release_heuristic_block=bayesian_release_heuristic_block,
                    bayesian_release_scores=bayesian_release_scores,
                    guidance_strategy="bayesian-release-policy",
                    posterior_unlock_state=posterior_unlock_state.status,
                    posterior_unlock_reason=posterior_unlock_state.reason,
                    posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
                    posterior_summary=prior_posterior,
                    failures=(planning_failures + release_failures) or None,
                ),
                current_candidate,
                prior_posterior,
                posterior_unlock_state,
                {f"stage_{stage.stage_id}_wall_time_seconds": 0.0},
            )
        if selected_release_candidate is not None and bayesian_release_selected_block is not None:
            current_candidate = selected_release_candidate
            effective_fit_params = _current_released_fit_params(planned_stage, selected_release_candidate)
        planned_stage.planning_source = "bayesian-release-policy"

    if planned_stage.automatic_final_structures:
        (
            automated_final_structure_scores,
            selected_structure_candidate,
            automated_final_structure_selected_id,
            final_block_selected_core_block,
            final_block_followup_steps,
            structure_failures,
        ) = _automatic_final_structure_inference(
            request=request,
            stage=planned_stage,
            current_candidate=current_candidate,
        )
        if structure_failures:
            followup_failures.extend(structure_failures)
        if selected_structure_candidate is not None:
            current_candidate = selected_structure_candidate
        planned_stage.planning_source = "automated-final-structure-inference"

    if planned_stage.final_block_core_blocks:
        (
            final_block_scores,
            selected_final_block_candidate,
            final_block_selected_core_block,
            final_block_failures,
        ) = _final_block_marginalized_candidate_scores(
            request,
            planned_stage,
            current_candidate,
        )
        if final_block_failures:
            return (
                RecoveryStageResult(
                    stage_id=stage.stage_id,
                    kind=stage.kind,
                    local_solver_mode=planned_stage.local_solver_mode,
                    planning_source="final-block-marginalized-bayesian",
                    planning_confidence=None,
                    original_hard_params=list(stage.hard_params),
                    original_fit_params=list(stage.fit_params),
                    planned_hard_params=list(planned_stage.hard_params),
                    planned_fit_params=list(effective_fit_params),
                    input_candidate_id=current_candidate.candidate_id,
                    input_hard_parameters=dict(current_candidate.hard_parameters),
                    selected_candidate=current_candidate,
                    wall_time_seconds=0.0,
                    weighted_rms_us=None,
                    basin_score_threshold=basin_score_threshold,
                    basin_entry_established=True if basin_score_threshold is not None else None,
                    full_solution_stage=planned_stage.full_solution_stage,
                    full_solution_accepted=False if planned_stage.full_solution_stage else None,
                    bayesian_release_selected_block=bayesian_release_selected_block,
                    bayesian_release_heuristic_block=bayesian_release_heuristic_block,
                    bayesian_release_scores=bayesian_release_scores,
                    final_block_selected_core_block=final_block_selected_core_block,
                    final_block_scores=final_block_scores,
                    guidance_strategy="final-block-marginalized-bayesian",
                    posterior_unlock_state=posterior_unlock_state.status,
                    posterior_unlock_reason=posterior_unlock_state.reason,
                    posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
                    posterior_summary=prior_posterior,
                    failures=(planning_failures + final_block_failures) or None,
                ),
                current_candidate,
                prior_posterior,
                posterior_unlock_state,
                {f"stage_{stage.stage_id}_wall_time_seconds": 0.0},
            )
        if selected_final_block_candidate is not None and final_block_selected_core_block is not None:
            current_candidate = selected_final_block_candidate
        planned_stage.planning_source = "final-block-marginalized-bayesian"

    if (
        final_block_selected_core_block is not None
        and (planned_stage.final_block_nuisance_groups or planned_stage.final_block_expansion_blocks)
    ):
        sequence_fit_params = list(
            dict.fromkeys(final_block_selected_core_block + list(planned_stage.final_block_marginalized_params or []))
        )
        (
            current_candidate,
            _accepted_sequence_fit_params,
            final_block_followup_steps,
            followup_failures,
        ) = _run_final_block_followup_sequence(
            request,
            planned_stage,
            current_candidate,
            sequence_fit_params,
        )
        if followup_failures:
            planned_stage.planning_source = "final-block-staged-expansion"
        elif final_block_followup_steps:
            planned_stage.planning_source = "final-block-staged-expansion"

    stage_start = time.perf_counter()
    fitted_candidate, posterior_summary, failures, fit_metrics = refine_candidate_with_pint(
        request,
        current_candidate,
        fit_params=effective_fit_params,
        stage_id=stage.stage_id,
        quality_gate_max_reduced_chi2=stage.quality_gate_max_reduced_chi2,
        quality_gate_max_weighted_rms_us=stage.quality_gate_max_weighted_rms_us,
        local_solver_mode=planned_stage.local_solver_mode,
        local_solver_max_iterations=planned_stage.local_solver_max_iterations,
        local_solver_initial_damping=planned_stage.local_solver_initial_damping,
        local_solver_damping_multiplier=planned_stage.local_solver_damping_multiplier,
        local_solver_min_fractional_improvement=planned_stage.local_solver_min_fractional_improvement,
    )
    blocking_failures = {
        failure.code
        for failure in failures
        if failure.code.startswith("local-fit-")
    }
    def _candidate_quality(candidate: Candidate) -> float:
        if candidate.fit_chi2 is not None:
            return candidate.fit_chi2
        if candidate.exact_score is not None:
            return -candidate.exact_score
        return float("inf")
    updated_unlock_state, unlock_failures = _build_posterior_unlock_state(
        posterior_unlock_state,
        planned_stage,
        posterior_summary,
        failures,
    )
    failures = unlock_failures
    if blocking_failures:
        if request.simulation.regime == "full-fit" and _candidate_quality(fitted_candidate) < _candidate_quality(current_candidate):
            selected_candidate = fitted_candidate
        else:
            selected_candidate = current_candidate
    else:
        selected_candidate = fitted_candidate
    stage_wall_time = time.perf_counter() - stage_start
    metrics = {
        f"stage_{stage.stage_id}_wall_time_seconds": stage_wall_time,
        f"stage_{stage.stage_id}_fit_chi2": fit_metrics["fit_chi2"],
        f"stage_{stage.stage_id}_fit_reduced_chi2": fit_metrics["fit_reduced_chi2"],
        f"stage_{stage.stage_id}_weighted_rms_us": fit_metrics["weighted_rms_us"],
    }

    return (
        RecoveryStageResult(
            stage_id=stage.stage_id,
            kind=stage.kind,
            local_solver_mode=planned_stage.local_solver_mode,
            local_solver_diagnostics=fit_metrics.get("local_solver_diagnostics"),
            planning_source=planned_stage.planning_source,
            planning_confidence=planned_stage.planning_confidence,
            original_hard_params=list(stage.hard_params),
            original_fit_params=list(stage.fit_params),
            planned_hard_params=list(planned_stage.hard_params),
            planned_fit_params=list(effective_fit_params),
            input_candidate_id=current_candidate.candidate_id,
            input_hard_parameters=dict(current_candidate.hard_parameters),
            selected_candidate=selected_candidate,
            wall_time_seconds=stage_wall_time,
            weighted_rms_us=fit_metrics["weighted_rms_us"],
            basin_score_threshold=basin_score_threshold,
            basin_entry_established=True if basin_score_threshold is not None else None,
            full_solution_stage=planned_stage.full_solution_stage,
            full_solution_accepted=(not blocking_failures) if planned_stage.full_solution_stage else None,
            bayesian_release_selected_block=bayesian_release_selected_block,
            bayesian_release_heuristic_block=bayesian_release_heuristic_block,
            bayesian_release_scores=bayesian_release_scores,
            final_block_selected_core_block=final_block_selected_core_block,
            final_block_scores=final_block_scores,
            final_block_followup_steps=final_block_followup_steps,
            automated_final_structure_selected_id=automated_final_structure_selected_id,
            automated_final_structure_scores=automated_final_structure_scores,
            guidance_strategy=posterior_summary.guidance_strategy,
            posterior_unlock_state=updated_unlock_state.status,
            posterior_unlock_reason=updated_unlock_state.reason,
            posterior_unlocked_by_stage=updated_unlock_state.unlocked_by_stage,
            posterior_summary=posterior_summary,
            failures=(planning_failures + followup_failures + failures) or None,
        ),
        selected_candidate,
        posterior_summary,
        updated_unlock_state,
        metrics,
    )


def run_staged_recovery(
    request: BootstrapRequest,
) -> tuple[
    list[RecoveryStageResult],
    list[Candidate],
    list[Candidate],
    Candidate,
    list[Candidate],
    list[FailureState],
    dict[str, float],
    PosteriorUnlockState,
]:
    stages = build_recovery_stages(request)
    current_candidate = Candidate(
        candidate_id="seed",
        hard_parameters={
            name: value
            for name, value in request.initial_parameters.items()
            if name in request.simulation.hard_params
        },
    )
    stage_results: list[RecoveryStageResult] = []
    all_generated_candidates: list[Candidate] = []
    final_ranked_candidates: list[Candidate] = [current_candidate]
    successful_full_solution_candidates: list[Candidate] = []
    failures: list[FailureState] = []
    metrics: dict[str, float] = {}
    certification_wall_time_seconds = 0.0
    prior_posterior: PosteriorSummary | None = None
    posterior_unlock_state = PosteriorUnlockState(
        status="locked",
        reason="No trusted local-fit posterior has been established yet.",
    )

    for stage in stages:
        if stage.kind == "coarse-search":
            (
                stage_result,
                current_candidate,
                generated_candidates,
                retained_candidates,
                _,
                stage_metrics,
            ) = _run_coarse_search_stage(
                request,
                stage,
                current_candidate,
                prior_posterior,
                posterior_unlock_state,
            )
            all_generated_candidates.extend(generated_candidates)
            final_ranked_candidates = retained_candidates
            if stage_result.failures:
                failures.extend(stage_result.failures)
            certification_wall_time_seconds += sum(
                value
                for key, value in stage_metrics.items()
                if "certification" in key or "pint_" in key
            )
            metrics.update(stage_metrics)
        elif stage.kind == "local-fit":
            (
                stage_result,
                current_candidate,
                prior_posterior,
                posterior_unlock_state,
                stage_metrics,
            ) = _run_local_fit_stage(
                request,
                stage,
                current_candidate,
                prior_posterior,
                posterior_unlock_state,
            )
            if stage_result.failures:
                failures.extend(stage_result.failures)
            certification_wall_time_seconds += stage_result.wall_time_seconds
            metrics.update(stage_metrics)
            if (
                stage_result.full_solution_stage
                and stage_result.full_solution_accepted
                and stage_result.selected_candidate is not None
            ):
                successful_full_solution_candidates.append(stage_result.selected_candidate)
        else:
            failure = FailureState(
                stage=stage.stage_id,
                code="unknown-stage-kind",
                message=f"Unsupported recovery stage kind: {stage.kind}",
                suggestion="Use 'coarse-search' or 'local-fit'.",
            )
            stage_result = RecoveryStageResult(
                stage_id=stage.stage_id,
                kind=stage.kind,
                input_candidate_id=current_candidate.candidate_id,
                input_hard_parameters=dict(current_candidate.hard_parameters),
                selected_candidate=current_candidate,
                guidance_strategy="deterministic-fallback",
                full_solution_stage=False,
                full_solution_accepted=None,
                posterior_unlock_state=posterior_unlock_state.status,
                posterior_unlock_reason=posterior_unlock_state.reason,
                posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
                failures=[failure],
            )
            failures.append(failure)
            metrics[f"stage_{stage.stage_id}_wall_time_seconds"] = 0.0

        stage_results.append(stage_result)

    metrics["certification_wall_time_seconds"] = certification_wall_time_seconds
    return (
        stage_results,
        all_generated_candidates,
        final_ranked_candidates,
        current_candidate,
        successful_full_solution_candidates,
        failures,
        metrics,
        posterior_unlock_state,
    )
