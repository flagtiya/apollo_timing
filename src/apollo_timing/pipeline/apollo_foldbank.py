"""Orchestrate finite folding coverage using the library's scientific owners."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from apollo_timing import __version__
from apollo_timing.backend.presto_backend import (
    _prepfold_command,
    run_fold_tasks,
    write_fold_script,
)
from apollo_timing.inference.bayesian_partial_ell1_bootstrap import (
    _isolated_model_text,
    _par_values,
    is_incomplete_ell1_par,
)
from apollo_timing.inference.foldbank_bootstrap import (
    FoldbankBootstrapConfig,
    build_bootstrap_portfolio,
    prepare_foldbank_training,
)
from apollo_timing.inference.foldbank_candidates import (
    _enumerate_orbital_period_aliases,
    _evaluate_mode,
    _sample_continuous_foldbank_modes,
    _screen_relative_chi2_by_alias,
    _select_alias_stratified_coverage_bank,
)
from apollo_timing.io.foldbank_artifacts import (
    _clock_fingerprint,
    _prepare_run_output,
    bootstrap_products,
    foldbank_run_audit,
    implementation_fingerprint,
    load_bootstrap_portfolio,
    runtime_versions,
    store_models,
)
from apollo_timing.io.save_artifacts import (
    atomic_write_json as _atomic_json,
)
from apollo_timing.io.save_artifacts import (
    atomic_write_text,
)
from apollo_timing.io.save_artifacts import (
    sha256_file as _sha256,
)
from apollo_timing.io.search_psrfits import (
    _match_sessions_to_observations,
    _validate_observation_sets,
    _validate_two_session_contract,
    inspect_search_psrfits,
    load_search_observations,
)
from apollo_timing.models.foldbank import FoldbankConfig, FoldbankResult, FoldTask
from apollo_timing.reports.foldbank import (
    _scientific_validation,
    _write_scientific_comparisons,
)


def _input_pb_days(path: Path) -> float:
    values = _par_values(path)
    try:
        pb = float(values["PB"].replace("D", "E").replace("d", "e"))
    except (KeyError, ValueError) as error:
        raise ValueError("Foldbank input needs a positive PB in days.") from error
    if not 0 < pb < float("inf"):
        raise ValueError("Foldbank input needs a positive PB in days.")
    return pb


def build_foldbank(
    *,
    par_path: Path,
    tim_path: Path,
    search_paths: Sequence[Path],
    output_dir: Path,
    validation_search_paths: Sequence[Path] = (),
    config: FoldbankConfig | None = None,
    bootstrap_config: FoldbankBootstrapConfig | None = None,
    clock_override_path: Path | None = None,
    bootstrap_cache: Path | None = None,
    resume: bool = False,
    run_presto: bool = False,
    prepfold: str = "prepfold",
    pixi: Path | None = None,
    pixi_manifest: Path | None = None,
    toa_origin: str = "unknown",
) -> FoldbankResult:
    """Build a PRESTO fold bank from two or more training sessions and held-out data."""

    active = config or FoldbankConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    allowed_toa_origins = {
        "blind_local_search",
        "external_discovery",
        "truth_assisted",
        "unknown",
    }
    if toa_origin not in allowed_toa_origins:
        raise ValueError(
            "TOA origin must be blind_local_search, external_discovery, truth_assisted, or unknown."
        )
    if len(search_paths) < 2:
        raise ValueError("apollo-foldbank requires at least two --search PSRFITS files.")
    if not is_incomplete_ell1_par(par) or not any(
        line.split()[:2] == ["BINARY", "ELL1"]
        for line in par.read_text(encoding="utf-8").upper().splitlines()
    ):
        raise ValueError("apollo-foldbank currently requires an incomplete BINARY ELL1 par.")
    observations = load_search_observations(search_paths, prefix="epoch")
    source_names = {item.source for item in observations}
    if len(source_names) != 1:
        raise ValueError("The training observations do not name the same source.")
    validation_observations = load_search_observations(validation_search_paths, prefix="validation")
    if any(item.source not in source_names for item in validation_observations):
        raise ValueError("Validation observations must name the training source.")
    training_paths = {item.path for item in observations}
    if any(item.path in training_paths for item in validation_observations):
        raise ValueError("A validation observation cannot also be a training observation.")
    _validate_observation_sets(observations, validation_observations)

    observation_midpoints = [
        item.start_mjd + 0.5 * item.duration_seconds / 86400.0 for item in observations
    ]
    epoch_baseline_days = observation_midpoints[-1] - observation_midpoints[0]
    period_aliases = _enumerate_orbital_period_aliases(
        reference_pb_days=_input_pb_days(par),
        baseline_days=epoch_baseline_days,
        minimum_pb_days=active.orbital_period_min_days,
        maximum_pb_days=active.orbital_period_max_days,
        maximum_aliases=active.maximum_orbital_aliases,
    )
    period_range_requested = active.orbital_period_min_days is not None
    base_bootstrap_config = bootstrap_config or FoldbankBootstrapConfig()
    if not isinstance(base_bootstrap_config, FoldbankBootstrapConfig):
        raise TypeError("Foldbank needs FoldbankBootstrapConfig, not the timing bootstrap config.")
    implementation = implementation_fingerprint(
        (
            "inference/foldbank_bootstrap.py",
            "inference/session_shape_refinement.py",
            "inference/bayesian_partial_ell1_bootstrap.py",
            "inference/bayesian_dual_lattice_enumerator.py",
            "inference/foldbank_candidates.py",
            "inference/pint_solution_artifacts.py",
            "backend/pint_backend.py",
            "backend/presto_backend.py",
            "reports/foldbank.py",
            "models/foldbank.py",
            "io/save_artifacts.py",
            "io/foldbank_artifacts.py",
            "io/search_psrfits.py",
            "io/par_compatibility.py",
            "io/timing_update.py",
            "pipeline/apollo_foldbank.py",
        )
    )
    bootstrap_contract = json.loads(
        json.dumps(
            {
                "schema_version": 2,
                "package_version": __version__,
                "implementation": implementation,
                "runtime": runtime_versions(),
                "par_sha256": _sha256(par),
                "tim_sha256": _sha256(tim),
                "clock_files": _clock_fingerprint(clock_override_path),
                "config": asdict(base_bootstrap_config),
                "period_range_requested": period_range_requested,
                "aliases": [item.to_dict() for item in period_aliases],
            },
            allow_nan=False,
        )
    )
    run_contract = json.loads(
        json.dumps(
            {
                "bootstrap": bootstrap_contract,
                "config": asdict(active),
                "training": [
                    {**item.to_dict(), "sha256": _sha256(item.path)} for item in observations
                ],
                "validation": [
                    {**item.to_dict(), "sha256": _sha256(item.path)}
                    for item in validation_observations
                ],
                "toa_origin": toa_origin,
                "run_presto": run_presto,
                "command": list(
                    _prepfold_command(prepfold=prepfold, pixi=pixi, pixi_manifest=pixi_manifest)
                ),
                "bootstrap_cache": str(Path(bootstrap_cache).resolve())
                if bootstrap_cache
                else None,
            },
            allow_nan=False,
        )
    )
    _prepare_run_output(output, contract=run_contract, resume=resume)
    with foldbank_run_audit(output) as advance:
        training = prepare_foldbank_training(
            par_path=par,
            tim_path=tim,
            output_dir=output,
            clock_override_path=clock_override_path,
        )
        sessions, toas = training.sessions, training.toas
        _validate_two_session_contract(sessions, config=active)
        matches = _match_sessions_to_observations(sessions=sessions, observations=observations)
        advance("bootstrap")
        if bootstrap_cache is not None:
            portfolio = load_bootstrap_portfolio(
                cache_dir=bootstrap_cache,
                aliases=period_aliases,
                contract=bootstrap_contract,
            )
        else:
            portfolio = build_bootstrap_portfolio(
                aliases=period_aliases,
                reference_pb_days=_input_pb_days(par),
                period_range_requested=period_range_requested,
                par_path=par,
                tim_path=tim,
                output_dir=output,
                config=base_bootstrap_config,
                training=training,
                clock_override_path=clock_override_path,
                resume=resume,
                workers=active.bootstrap_workers,
            )
        _atomic_json(
            output / "bootstrap_portfolio.json",
            {
                "aliases": list(portfolio.alias_records),
                "entries": [entry.to_dict() for entry in portfolio.entries],
            },
        )
        all_modes = list(portfolio.modes)
        mode_alias_by_id = portfolio.mode_alias_by_id
        if not all_modes:
            raise RuntimeError("No orbital-period cell produced a usable folding mode.")
        advance("candidate_coverage")
        posterior_modes, posterior_audits = _sample_continuous_foldbank_modes(
            seed_modes=all_modes,
            mode_alias_by_id=mode_alias_by_id,
            toas=toas,
            sessions=sessions,
            output_dir=output,
            config=active,
            maximum_eccentricity=base_bootstrap_config.maximum_eccentricity,
            maximum_a1_seconds=base_bootstrap_config.projected_semimajor_axis_stages_seconds[-1],
        )
        all_modes.extend(posterior_modes)
        evaluated = tuple(
            _evaluate_mode(mode=mode, toas=toas, matches=matches, config=active)
            for mode in all_modes
        )
        evaluated = _screen_relative_chi2_by_alias(
            evaluated,
            mode_alias_by_id=mode_alias_by_id,
            maximum_delta_chi2=active.maximum_candidate_delta_chi2,
        )
        selected = _select_alias_stratified_coverage_bank(
            evaluated,
            mode_alias_by_id=mode_alias_by_id,
            config=active,
        )
        if not selected:
            raise RuntimeError(
                "No Apollo mode preserves coherent phase in all training observations."
            )

        eligible_solutions = sorted(
            (item for item in evaluated if item.eligible),
            key=lambda item: (
                item.mode.all_session_shape_chi2,
                item.mode.informative_session_chi2,
            ),
        )
        advance("model_storage")
        model_records, candidate_records = store_models(
            output_dir=output,
            eligible=eligible_solutions,
            selected=selected,
        )
        model_by_mode = {item.mode.mode_id: model_id for model_id, _, item in model_records}
        selected_by_mode = {
            item.mode.mode_id: candidate_id for candidate_id, _, item in candidate_records
        }
        baseline_dir = output / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / "discovery.par"
        atomic_write_text(baseline_path, _isolated_model_text(par))
        folded_bank_layer = "models" if validation_observations else "candidates"
        folded_bank_records = model_records if validation_observations else candidate_records
        records = [
            ("baseline_discovery", baseline_path),
            *((record_id, path) for record_id, path, _ in folded_bank_records),
        ]
        observation_parts = (
            [(item, active.validation_parts) for item in validation_observations]
            if validation_observations
            else [(observation, len(session.toa_indices)) for session, observation in matches]
        )
        tasks = tuple(
            FoldTask(record_id, path, observation, parts)
            for record_id, path in records
            for observation, parts in observation_parts
        )
        command = _prepfold_command(
            prepfold=prepfold,
            pixi=pixi,
            pixi_manifest=pixi_manifest,
        )
        advance("folding")
        script = write_fold_script(
            path=output / "fold_commands.sh",
            command=command,
            tasks=tasks,
            output_dir=output,
            config=active,
        )
        fold_results: tuple[dict[str, object], ...] = ()
        if run_presto:
            fold_results = run_fold_tasks(
                command=command,
                tasks=tasks,
                output_dir=output,
                config=active,
            )

        evaluated_payload = []
        for item in evaluated:
            candidate_id = selected_by_mode.get(item.mode.mode_id)
            evaluated_payload.append(
                {
                    "mode_id": item.mode.mode_id,
                    "orbital_period_alias_id": mode_alias_by_id[item.mode.mode_id].alias_id,
                    "source_par": str(item.mode.par_path),
                    "informative_session_chi2": item.mode.informative_session_chi2,
                    "all_session_shape_chi2": item.mode.all_session_shape_chi2,
                    "eligible": item.eligible,
                    "selected": candidate_id is not None,
                    "candidate_id": candidate_id,
                    "model_id": model_by_mode.get(item.mode.mode_id),
                    "rejection_reasons": list(item.rejection_reasons),
                    "sessions": [score.to_dict() for score in item.session_scores],
                }
            )
        advance("scientific_reports")
        validation = _scientific_validation(
            fold_results=fold_results,
            candidate_ids=[item[0] for item in folded_bank_records],
            observation_ids=[
                item.observation_id for item in (validation_observations or observations)
            ],
            detection_sigma_threshold=active.detection_sigma_threshold,
            toa_origin=toa_origin,
            evaluation_scope=(
                "heldout_validation_epochs" if validation_observations else "training_epochs"
            ),
        )
        comparison_figures = _write_scientific_comparisons(
            fold_results=fold_results,
            validation=validation,
            observations=(*observations, *validation_observations),
            output_dir=output,
        )
        advance("manifest")
        manifest = {
            "schema_version": 4,
            "bootstrap_contract": bootstrap_contract,
            "bootstrap_products": bootstrap_products(portfolio),
            "bootstrap_entries": [entry.to_dict() for entry in portfolio.entries],
            "architecture": "apollo_foldbank_multi_session_alias_stratified_coverage",
            "scientific_contract": {
                "observation_count": len(observations),
                "minimum_observation_count": 2,
                "toas_per_observation": [
                    active.minimum_toas_per_session,
                    active.maximum_toas_per_session,
                ],
                "session_constant_phase": "profiled_independently",
                "orbital_parameters": "shared_jointly",
                "orbital_period_aliases": "fully_tiled_over_supplied_pb_prior",
                "posterior_refinement": "performed_independently_per_solved_pb_alias",
                "candidate_retention": "alias_stratified_trajectory_coverage",
                "presto_search": "disabled",
                "success_requires_truth_independent_toas": True,
                "heldout_validation_excluded_from_inference": True,
                "phase_connected_timing_certification": False,
                "shape_score_scope": "reported_error_weighted_fixed_local_integer_session_profile",
                "reference_pb_policy": "additional_seed_in_existing_cell",
                "continuous_samples": "finite_local_sobol_coverage_not_normalized_posterior",
            },
            "inputs": {
                "par": {"path": str(par), "sha256": _sha256(par)},
                "tim": {
                    "path": str(tim),
                    "sha256": _sha256(tim),
                    "toa_origin": toa_origin,
                },
                "search_observations": [
                    {**item.to_dict(), "sha256": _sha256(item.path)} for item in observations
                ],
                "validation_search_observations": [
                    {**item.to_dict(), "sha256": _sha256(item.path)}
                    for item in validation_observations
                ],
            },
            "config": asdict(active),
            "bootstrap_cache": {
                "used": bootstrap_cache is not None,
                "path": str(Path(bootstrap_cache).resolve()) if bootstrap_cache else None,
            },
            "orbital_period_prior": {
                "reference_pb_days": _input_pb_days(par),
                "epoch_baseline_days": epoch_baseline_days,
                "alias_count": len(period_aliases),
                "completed_alias_count": sum(
                    record["mode_count"] > 0 for record in portfolio.alias_records
                ),
                "aliases": list(portfolio.alias_records),
            },
            "bootstrap_audits": [str(item.audit_path) for item in portfolio.results],
            "posterior_sampling": list(posterior_audits),
            "evaluated_modes": evaluated_payload,
            "models": [
                {
                    "model_id": model_id,
                    "source_mode_id": item.mode.mode_id,
                    "orbital_period_alias_id": mode_alias_by_id[item.mode.mode_id].alias_id,
                    "par_path": str(path),
                    "sha256": _sha256(path),
                }
                for model_id, path, item in model_records
            ],
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "source_mode_id": item.mode.mode_id,
                    "orbital_period_alias_id": mode_alias_by_id[item.mode.mode_id].alias_id,
                    "par_path": str(path),
                    "sha256": _sha256(path),
                }
                for candidate_id, path, item in candidate_records
            ],
            "baseline": {
                "model": "isolated discovery-stage par with the incomplete binary block removed",
                "par_path": str(baseline_path),
                "sha256": _sha256(baseline_path),
            },
            "presto": {
                "command_prefix": list(command),
                "fold_script": str(script),
                "folded_bank_layer": folded_bank_layer,
                "executed": run_presto,
                "results": list(fold_results),
            },
            "scientific_validation": validation,
            "comparison_figures": list(comparison_figures),
        }
        manifest_path = _atomic_json(output / "manifest.json", manifest)
        return FoldbankResult(
            output_dir=output,
            manifest_path=manifest_path,
            command_script_path=script,
            candidate_paths=tuple(path for _, path, _ in candidate_records),
            fold_results=fold_results,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = FoldbankConfig()
    bootstrap_defaults = FoldbankBootstrapConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Build a PRESTO parfile fold bank from two or more observations with 8-16 TOAs each."
        )
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument("--search", required=True, action="append", type=Path)
    parser.add_argument(
        "--validation-search",
        action="append",
        default=[],
        type=Path,
        help=(
            "Held-out search-mode PSRFITS observation used only after Apollo "
            "inference to fold every plausible solution."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument(
        "--bootstrap-cache",
        type=Path,
        help=(
            "Reuse completed discrete Apollo alias bootstraps from a prior fold-bank "
            "run; only posterior expansion and candidate retention are recomputed."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-presto", action="store_true")
    parser.add_argument("--prepfold", default="prepfold")
    parser.add_argument(
        "--presto-workers",
        type=int,
        default=defaults.presto_workers,
    )
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=defaults.bootstrap_workers,
        help="Independent orbital-alias bootstrap processes.",
    )
    parser.add_argument(
        "--posterior-workers",
        type=int,
        default=defaults.posterior_workers,
        help="Independent continuous-posterior seed processes.",
    )
    parser.add_argument("--pixi", type=Path)
    parser.add_argument("--pixi-manifest", type=Path)
    parser.add_argument(
        "--toa-origin",
        choices=(
            "blind_local_search",
            "external_discovery",
            "truth_assisted",
            "unknown",
        ),
        default="unknown",
        help="Audited origin of the two subintegration-TOA sessions.",
    )
    parser.add_argument(
        "--pb-min",
        type=float,
        help="Lower bound of the discovery-stage orbital-period prior in days.",
    )
    parser.add_argument(
        "--pb-max",
        type=float,
        help="Upper bound of the discovery-stage orbital-period prior in days.",
    )
    parser.add_argument(
        "--max-pb-aliases",
        type=int,
        default=defaults.maximum_orbital_aliases,
        help="Hard completeness budget for inter-epoch orbital-cycle cells.",
    )
    parser.add_argument("--a1-stage", dest="a1_stages", action="append", type=float)
    parser.add_argument(
        "--retained-modes",
        type=int,
        default=max(bootstrap_defaults.retained_exact_modes, 32),
    )
    parser.add_argument(
        "--max-eccentricity",
        type=float,
        default=bootstrap_defaults.maximum_eccentricity,
        help="Absolute EPS1/EPS2 bound used by every bootstrap refinement stage.",
    )
    parser.add_argument(
        "--max-f0-offset",
        type=float,
        default=bootstrap_defaults.maximum_spin_frequency_offset_hz,
        help=(
            "Maximum absolute F0 offset from the discovery par, in Hz, when "
            "--refine-spin-frequency is enabled."
        ),
    )
    parser.add_argument(
        "--refine-spin-frequency",
        action="store_true",
        help=(
            "Fit F0 inside --max-f0-offset during each orbital-alias bootstrap. "
            "Use this when the discovery F0 is an apparent frequency measured "
            "before the binary orbit is known."
        ),
    )
    parser.add_argument(
        "--max-phase-span",
        type=float,
        default=defaults.maximum_phase_span_cycles,
    )
    parser.add_argument(
        "--max-coherent-loss",
        type=float,
        default=defaults.maximum_coherent_loss,
    )
    parser.add_argument(
        "--max-delta-chi2",
        type=float,
        default=defaults.maximum_candidate_delta_chi2,
        help="Maximum joint-session chi-square above the best retained solution.",
    )
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=defaults.posterior_draws,
        help="Deterministic Sobol draws per selected continuous posterior mode.",
    )
    parser.add_argument(
        "--posterior-retained",
        type=int,
        default=defaults.posterior_retained_per_mode,
    )
    parser.add_argument(
        "--posterior-seed-modes",
        type=int,
        default=defaults.posterior_seed_modes,
        help="Number of high-probability posterior seed modes retained per PB alias.",
    )
    parser.add_argument(
        "--posterior-box-sigma",
        type=float,
        default=defaults.posterior_box_sigma,
    )
    parser.add_argument(
        "--trajectory-separation",
        type=float,
        default=defaults.minimum_trajectory_separation_cycles,
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=defaults.maximum_candidates,
    )
    parser.add_argument("--profile-bins", type=int, default=defaults.profile_bins)
    parser.add_argument("--subbands", type=int, default=defaults.subbands)
    parser.add_argument(
        "--validation-parts",
        type=int,
        default=defaults.validation_parts,
        help="Number of PRESTO time parts for each held-out observation.",
    )
    parser.add_argument(
        "--detection-sigma",
        type=float,
        default=defaults.detection_sigma_threshold,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    fold_config = FoldbankConfig(
        maximum_phase_span_cycles=args.max_phase_span,
        maximum_coherent_loss=args.max_coherent_loss,
        maximum_candidate_delta_chi2=args.max_delta_chi2,
        minimum_trajectory_separation_cycles=args.trajectory_separation,
        maximum_candidates=args.max_candidates,
        orbital_period_min_days=args.pb_min,
        orbital_period_max_days=args.pb_max,
        maximum_orbital_aliases=args.max_pb_aliases,
        profile_bins=args.profile_bins,
        subbands=args.subbands,
        validation_parts=args.validation_parts,
        detection_sigma_threshold=args.detection_sigma,
        posterior_draws=args.posterior_draws,
        posterior_retained_per_mode=args.posterior_retained,
        posterior_seed_modes=args.posterior_seed_modes,
        posterior_box_sigma=args.posterior_box_sigma,
        bootstrap_workers=args.bootstrap_workers,
        posterior_workers=args.posterior_workers,
        presto_workers=args.presto_workers,
    )
    defaults = FoldbankBootstrapConfig()
    bootstrap_config = replace(
        defaults,
        projected_semimajor_axis_stages_seconds=(
            tuple(args.a1_stages)
            if args.a1_stages is not None
            else defaults.projected_semimajor_axis_stages_seconds
        ),
        retained_exact_modes=args.retained_modes,
        maximum_eccentricity=args.max_eccentricity,
        maximum_spin_frequency_offset_hz=args.max_f0_offset,
        refine_spin_frequency=args.refine_spin_frequency,
    )
    result = build_foldbank(
        par_path=args.par,
        tim_path=args.tim,
        search_paths=args.search,
        validation_search_paths=args.validation_search,
        output_dir=args.output,
        config=fold_config,
        bootstrap_config=bootstrap_config,
        clock_override_path=args.clock_override,
        bootstrap_cache=args.bootstrap_cache,
        resume=args.resume,
        run_presto=args.run_presto,
        prepfold=args.prepfold,
        pixi=args.pixi,
        pixi_manifest=args.pixi_manifest,
        toa_origin=args.toa_origin,
    )
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "FoldbankBootstrapConfig",
    "FoldbankConfig",
    "FoldbankResult",
    "build_foldbank",
    "inspect_search_psrfits",
    "build_arg_parser",
    "main",
]
