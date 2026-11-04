from __future__ import annotations

from timing_accelerator.models.contracts import PipelineResult


def build_summary(result: PipelineResult) -> str:
    official_candidate = result.certified_candidate
    debug_candidate = result.debug_candidate
    lines = [
        "# Timing Accelerator Summary",
        "",
        "## Official Outcome",
        f"- status: {result.official_status}",
        f"- candidate_id: {official_candidate.candidate_id if official_candidate else 'none'}",
        f"- execution_layer: {result.execution_layer}",
        "",
        "## Preflight",
        f"- ok_for_local_accelerator: {result.preflight.ok_for_local_accelerator}",
        f"- phase_wrap_risk: {result.preflight.phase_wrap_risk}",
        "",
        "## Candidate Counts",
        f"- generated: {len(result.generated_candidates)}",
        f"- retained_for_certification: {len(result.ranked_candidates)}",
        "",
        "## Debug Candidate",
        f"- candidate_id: {debug_candidate.candidate_id if debug_candidate else 'none'}",
        f"- approximate_score: {debug_candidate.approximate_score if debug_candidate else None}",
        f"- exact_score: {debug_candidate.exact_score if debug_candidate else None}",
        f"- fit_chi2: {debug_candidate.fit_chi2 if debug_candidate else None}",
        f"- fit_reduced_chi2: {debug_candidate.fit_reduced_chi2 if debug_candidate else None}",
        "",
        "## Debug Hard Parameters",
    ]
    for name, value in sorted((debug_candidate.hard_parameters if debug_candidate else {}).items()):
        lines.append(f"- {name}: {value}")
    if debug_candidate and debug_candidate.fitted_parameters:
        lines.extend(
            [
                "",
                "## Debug Fitted Parameters",
            ]
        )
        for name, value in sorted(debug_candidate.fitted_parameters.items()):
            lines.append(f"- {name}: {value}")
    if result.metrics:
        lines.extend(
            [
                "",
                "## Timing Metrics",
            ]
        )
        for name, value in sorted(result.metrics.items()):
            lines.append(f"- {name}: {value}")
    if result.stages:
        lines.extend(["", "## Debug Recovery Stages"])
        for stage in result.stages:
            lines.append(
                f"- {stage.stage_id} ({stage.kind}): "
                f"candidate_count={stage.candidate_count}, "
                f"retained_candidate_count={stage.retained_candidate_count}, "
                f"wall_time_seconds={stage.wall_time_seconds}, "
                f"guidance={stage.guidance_strategy}"
            )
            lines.append(
                f"- {stage.stage_id} planning: "
                f"source={stage.planning_source}, "
                f"confidence={stage.planning_confidence}, "
                f"original_hard={stage.original_hard_params}, "
                f"planned_hard={stage.planned_hard_params}, "
                f"original_fit={stage.original_fit_params}, "
                f"planned_fit={stage.planned_fit_params}"
            )
            if stage.bayesian_release_selected_block or stage.bayesian_release_scores:
                lines.append(
                    f"- {stage.stage_id} bayesian_release: "
                    f"selected_block={stage.bayesian_release_selected_block}, "
                    f"heuristic_block={stage.bayesian_release_heuristic_block}"
                )
                for score in stage.bayesian_release_scores or []:
                    lines.append(
                        f"- {stage.stage_id} bayesian_score: block={score.block}, "
                        f"log_bf={score.approximate_log_bayes_factor}, "
                        f"log_like_gain={score.log_likelihood_gain}, "
                        f"log_prior_volume={score.log_prior_volume}, "
                        f"log_posterior_volume={score.log_posterior_volume}, "
                        f"laplace_constant={score.laplace_constant_term}, "
                        f"coupling={score.coupling_strength}, "
                        f"condition_penalty={score.condition_penalty}, "
                        f"heuristic_prior_bonus={score.heuristic_prior_bonus}, "
                        f"evidence_score={score.evidence_score}, "
                        f"selected={score.selected}, fallback={score.fallback_used}"
                    )
            if stage.basin_shaping_center_offsets or stage.basin_shaping_rank_weights:
                lines.append(
                    f"- {stage.stage_id} shaping: "
                    f"center_offsets={stage.basin_shaping_center_offsets}, "
                    f"rank_weights={stage.basin_shaping_rank_weights}"
                )
            if stage.basin_score_threshold is not None or stage.basin_entry_established is not None:
                lines.append(
                    f"- {stage.stage_id} basin: "
                    f"threshold={stage.basin_score_threshold}, "
                    f"established={stage.basin_entry_established}, "
                    f"selected_basin_score={stage.selected_candidate.basin_score if stage.selected_candidate else None}"
                )
            if stage.posterior_summary is not None:
                lines.append(
                    f"- {stage.stage_id} posterior: "
                    f"status={stage.posterior_summary.status}, "
                    f"condition_number={stage.posterior_summary.condition_number}"
                )
            if stage.posterior_unlock_state is not None:
                lines.append(
                    f"- {stage.stage_id} posterior_unlock: "
                    f"state={stage.posterior_unlock_state}, "
                    f"unlocked_by={stage.posterior_unlocked_by_stage}, "
                    f"reason={stage.posterior_unlock_reason}"
                )
    if result.posterior_unlock_state is not None:
        lines.extend(
            [
                "",
                "## Posterior Unlock",
                f"- state: {result.posterior_unlock_state}",
                f"- unlocked_by_stage: {result.posterior_unlocked_by_stage}",
                f"- reason: {result.posterior_unlock_reason}",
            ]
        )
    if result.calibration:
        calibration = result.calibration
        lines.extend(
            [
                "",
                "## Calibration",
                f"- status: {calibration.status}",
                f"- evaluation_mode: {calibration.evaluation_mode}",
                f"- reference_par_path: {calibration.reference_par_path}",
                f"- reference_fit_chi2: {calibration.reference_fit_chi2}",
                f"- reference_fit_reduced_chi2: {calibration.reference_fit_reduced_chi2}",
                f"- reference_weighted_rms_us: {calibration.reference_weighted_rms_us}",
                f"- recovered_fit_chi2: {calibration.recovered_fit_chi2}",
                f"- recovered_fit_reduced_chi2: {calibration.recovered_fit_reduced_chi2}",
                f"- recovered_weighted_rms_us: {calibration.recovered_weighted_rms_us}",
                f"- quality_ratio_to_reference: {calibration.quality_ratio_to_reference}",
                f"- reduced_chi2_gap_to_reference: {calibration.reduced_chi2_gap_to_reference}",
                f"- weighted_rms_gap_us_to_reference: {calibration.weighted_rms_gap_us_to_reference}",
            ]
        )
        if calibration.hard_parameter_deltas:
            lines.extend(["", "## Hard Parameter Deltas To Reference"])
            for name, value in sorted(calibration.hard_parameter_deltas.items()):
                lines.append(f"- {name}: {value}")
        if calibration.fitted_parameter_deltas:
            lines.extend(["", "## Fitted Parameter Deltas To Reference"])
            for name, value in sorted(calibration.fitted_parameter_deltas.items()):
                lines.append(f"- {name}: {value}")
        if calibration.likely_error_points:
            lines.extend(["", "## Likely Error Points"])
            for item in calibration.likely_error_points:
                detail = f"- {item.code}: {item.message}"
                if item.evidence:
                    detail += f" Evidence: {item.evidence}"
                lines.append(detail)
    if result.failures:
        lines.extend(["", "## Failure States"])
        for failure in result.failures:
            detail = f"- {failure.stage}/{failure.code}: {failure.message}"
            if failure.suggestion:
                detail += f" Next step: {failure.suggestion}"
            lines.append(detail)
    lines.append("")
    return "\n".join(lines)
