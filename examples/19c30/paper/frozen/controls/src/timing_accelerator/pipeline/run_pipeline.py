from __future__ import annotations

import argparse
import time
from pathlib import Path

from timing_accelerator.architecture import resolve_execution_layer
from timing_accelerator.backend.pint_backend import build_reference_calibration
from timing_accelerator.config import PipelinePaths
from timing_accelerator.diagnostics.preflight import run_preflight
from timing_accelerator.io.load_par_tim import build_bootstrap_request
from timing_accelerator.io.save_artifacts import ensure_output_dir, write_json, write_text
from timing_accelerator.models.contracts import FailureState, PipelineResult, to_plain_data
from timing_accelerator.models.parameter_partition import build_parameter_partition
from timing_accelerator.pipeline.staged_recovery import run_staged_recovery
from timing_accelerator.priors.from_initial_solution import build_prior
from timing_accelerator.reports.calibration import build_calibration_report
from timing_accelerator.reports.summary import build_summary


def _official_reduced_chi2_policy_limit(request) -> float:
    return 100.0 if request.simulation.especially_sparse_benchmark else 20.0


def run_pipeline(
    par_path: str | Path,
    tim_path: str | Path,
    simulation_config_path: str | Path,
    output_dir: str | Path,
) -> PipelineResult:
    pipeline_start = time.perf_counter()
    metrics: dict[str, float] = {}
    paths = PipelinePaths(
        par_path=Path(par_path),
        tim_path=Path(tim_path),
        simulation_config_path=Path(simulation_config_path),
        output_dir=Path(output_dir),
    )
    stage_start = time.perf_counter()
    request = build_bootstrap_request(
        par_path=paths.par_path,
        tim_path=paths.tim_path,
        simulation_config_path=paths.simulation_config_path,
    )
    execution_layer = resolve_execution_layer(request.simulation)
    metrics["request_build_wall_time_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    preflight = run_preflight(request)
    metrics["preflight_wall_time_seconds"] = time.perf_counter() - stage_start
    if not preflight.ok_for_local_accelerator:
        raise ValueError("Bootstrap pipeline requires at least one TOA.")

    stage_start = time.perf_counter()
    prior = build_prior(request)
    metrics["prior_wall_time_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    partition = build_parameter_partition(request)
    metrics["partition_wall_time_seconds"] = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    (
        stage_results,
        generated_candidates,
        retained_candidates,
        debug_candidate,
        successful_full_solution_candidates,
        failures,
        staged_metrics,
        posterior_unlock_state,
    ) = run_staged_recovery(request)
    metrics["staged_recovery_wall_time_seconds"] = time.perf_counter() - stage_start
    if staged_metrics:
        metrics.update(staged_metrics)
    official_candidate = None
    if len(successful_full_solution_candidates) > 1:
        failures.append(
            FailureState(
                stage="full-solution",
                code="ambiguous-full-solution",
                message=(
                    "More than one full-solution fit stage produced a passing candidate, so the run "
                    "does not yield a unique final solution."
                ),
                suggestion="Tighten the search plan so exactly one full-solution candidate survives.",
            )
        )
    elif len(successful_full_solution_candidates) == 1:
        official_candidate = successful_full_solution_candidates[0]
    else:
        failures.append(
            FailureState(
                stage="full-solution",
                code="full-solution-not-found",
                message="No passing full-solution fit was produced by the configured pipeline.",
                suggestion="Use staged search only to generate better initial conditions for the full-solution fit.",
            )
        )

    calibration = build_reference_calibration(
        request,
        official_candidate=official_candidate,
        debug_candidate=debug_candidate,
        stage_results=stage_results,
        failures=failures,
    )
    if official_candidate is not None and official_candidate.fit_reduced_chi2 is not None:
        policy_limit = _official_reduced_chi2_policy_limit(request)
        if official_candidate.fit_reduced_chi2 >= policy_limit:
            failures.append(
                FailureState(
                    stage="full-solution",
                    code="official-reduced-chi2-policy",
                    message=(
                        "The official full-solution candidate violated the reduced-chi2 policy: "
                        f"{official_candidate.fit_reduced_chi2} >= {policy_limit}"
                    ),
                    suggestion="Tighten the official convergence path until reduced chi-squared satisfies the benchmark policy.",
                )
            )
            official_candidate = None
    if official_candidate is not None and calibration is not None and calibration.status != "success":
        failures.append(
            FailureState(
                stage="calibration",
                code="full-solution-incorrect",
                message=(
                    "A passing full-solution fit was found, but it does not match the calibrated "
                    f"reference solution. status={calibration.status}"
                ),
                suggestion=(
                    "Treat this run as failure and use the debug artifacts to improve the initial "
                    "conditions or final full-solution fit design."
                ),
            )
        )
        official_candidate = None
    metrics["pipeline_wall_time_seconds"] = time.perf_counter() - pipeline_start
    official_status = "success" if official_candidate is not None else "failure"

    output_dir_path = ensure_output_dir(paths.output_dir)
    result = PipelineResult(
        preflight=preflight,
        prior=prior,
        partition=partition,
        generated_candidates=generated_candidates,
        ranked_candidates=retained_candidates,
        certified_candidate=official_candidate,
        debug_candidate=debug_candidate,
        official_status=official_status,
        output_dir=output_dir_path,
        execution_layer=execution_layer,
        metrics=metrics,
        stages=stage_results,
        posterior_unlock_state=posterior_unlock_state.status,
        posterior_unlock_reason=posterior_unlock_state.reason,
        posterior_unlocked_by_stage=posterior_unlock_state.unlocked_by_stage,
        calibration=calibration,
        failures=failures or None,
    )

    write_json(output_dir_path / "preflight.json", preflight)
    write_json(output_dir_path / "prior.json", prior)
    write_json(output_dir_path / "partition.json", partition)
    write_json(output_dir_path / "generated_candidates.json", generated_candidates)
    write_json(output_dir_path / "ranked_candidates.json", retained_candidates)
    write_json(output_dir_path / "stages.json", stage_results)
    for index, stage_result in enumerate(stage_results, start=1):
        write_json(
            output_dir_path / f"stage_{index:02d}_{stage_result.stage_id}.json",
            stage_result,
        )
    write_json(
        output_dir_path / "result.json",
        {
            "execution_layer": execution_layer,
            "official_status": official_status,
            "preflight": preflight,
            "prior": prior,
            "partition": partition,
            "debug_generated_candidate_count": len(generated_candidates),
            "debug_retained_candidate_count": len(retained_candidates),
            "certified_candidate": official_candidate,
            "debug_candidate": debug_candidate,
            "metrics": metrics,
            "debug_stages": stage_results,
            "posterior_unlock_state": posterior_unlock_state.status,
            "posterior_unlock_reason": posterior_unlock_state.reason,
            "posterior_unlocked_by_stage": posterior_unlock_state.unlocked_by_stage,
            "calibration": calibration,
            "failures": failures or None,
        },
    )
    if calibration is not None:
        write_json(output_dir_path / "calibration.json", calibration)
        write_text(output_dir_path / "calibration_report.md", build_calibration_report(result))
    write_text(output_dir_path / "summary.md", build_summary(result))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bootstrap timing accelerator pipeline.")
    parser.add_argument("--par", required=True, help="Path to a lightweight bootstrap par file.")
    parser.add_argument("--tim", required=True, help="Path to a lightweight bootstrap tim file.")
    parser.add_argument(
        "--simulation-config",
        required=True,
        help="Path to a simulation config JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where bootstrap artifacts will be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_pipeline(
        par_path=args.par,
        tim_path=args.tim,
        simulation_config_path=args.simulation_config,
        output_dir=args.output_dir,
    )
    if result.certified_candidate is not None:
        print(f"Official full-solution candidate: {result.certified_candidate.candidate_id}")
    else:
        print("Official full-solution candidate: none")
        print("Full-solution recovery status: failure")
    if result.execution_layer is not None:
        print(f"Execution layer: {result.execution_layer}")
    print(f"Artifacts written to: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
