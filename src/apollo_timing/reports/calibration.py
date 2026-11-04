from __future__ import annotations

from apollo_timing.models.contracts import PipelineResult


def build_calibration_report(result: PipelineResult) -> str:
    calibration = result.calibration
    if calibration is None:
        return "# Calibration report\n\nNo reference calibration target was configured.\n"

    lines = [
        "# Calibration report",
        "",
        "## Outcome",
        f"- official_status: {result.official_status}",
        f"- status: {calibration.status}",
        f"- evaluation_mode: {calibration.evaluation_mode}",
        f"- reference_par_path: {calibration.reference_par_path}",
        f"- reference_fit_reduced_chi2: {calibration.reference_fit_reduced_chi2}",
        f"- reference_weighted_rms_us: {calibration.reference_weighted_rms_us}",
        f"- recovered_fit_reduced_chi2: {calibration.recovered_fit_reduced_chi2}",
        f"- recovered_weighted_rms_us: {calibration.recovered_weighted_rms_us}",
        f"- quality_ratio_to_reference: {calibration.quality_ratio_to_reference}",
        f"- reduced_chi2_gap_to_reference: {calibration.reduced_chi2_gap_to_reference}",
        f"- weighted_rms_gap_us_to_reference: {calibration.weighted_rms_gap_us_to_reference}",
        "",
        "## Posterior guidance status",
        f"- state: {result.posterior_unlock_state}",
        f"- unlocked_by_stage: {result.posterior_unlocked_by_stage}",
        f"- reason: {result.posterior_unlock_reason}",
        "",
        "## Debug stage summary",
    ]
    for stage in result.stages or []:
        lines.append(
            f"- {stage.stage_id} ({stage.kind}): wall_time_seconds={stage.wall_time_seconds}, "
            f"guidance={stage.guidance_strategy}, "
            f"planning_source={stage.planning_source}, "
            f"selected_candidate={stage.selected_candidate.candidate_id if stage.selected_candidate else 'n/a'}"
        )
        lines.append(
            f"- {stage.stage_id} plan: original_hard={stage.original_hard_params}, "
            f"planned_hard={stage.planned_hard_params}, "
            f"original_fit={stage.original_fit_params}, "
            f"planned_fit={stage.planned_fit_params}, "
            f"planning_confidence={stage.planning_confidence}"
        )
        if stage.bayesian_release_selected_block or stage.bayesian_release_scores:
            lines.append(
                f"- {stage.stage_id} bayesian_release: selected_block={stage.bayesian_release_selected_block}, "
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
                f"- {stage.stage_id} shaping: center_offsets={stage.basin_shaping_center_offsets}, "
                f"rank_weights={stage.basin_shaping_rank_weights}"
            )
        if stage.basin_score_threshold is not None or stage.basin_entry_established is not None:
            lines.append(
                f"- {stage.stage_id} basin: threshold={stage.basin_score_threshold}, "
                f"established={stage.basin_entry_established}, "
                f"selected_basin_score={stage.selected_candidate.basin_score if stage.selected_candidate else None}"
            )
        if stage.posterior_unlock_state is not None:
            lines.append(
                f"- {stage.stage_id} posterior_unlock: state={stage.posterior_unlock_state}, "
                f"unlocked_by={stage.posterior_unlocked_by_stage}, "
                f"reason={stage.posterior_unlock_reason}"
            )
    if calibration.hard_parameter_deltas:
        lines.extend(["", "## Hard-parameter differences"])
        for name, value in sorted(calibration.hard_parameter_deltas.items()):
            lines.append(f"- {name}: {value}")
    if calibration.fitted_parameter_deltas:
        lines.extend(["", "## Fitted-parameter differences"])
        for name, value in sorted(calibration.fitted_parameter_deltas.items()):
            lines.append(f"- {name}: {value}")
    if calibration.likely_error_points:
        lines.extend(["", "## Possible sources of error"])
        for item in calibration.likely_error_points:
            line = f"- {item.code}: {item.message}"
            if item.evidence:
                line += f" Evidence: {item.evidence}"
            lines.append(line)
    if result.failures:
        lines.extend(["", "## Failure categories"])
        for failure in result.failures:
            lines.append(
                f"- {failure.stage}/{failure.code}: {failure.message}"
                + (f" Next step: {failure.suggestion}" if failure.suggestion else "")
            )
    lines.append("")
    return "\n".join(lines)
