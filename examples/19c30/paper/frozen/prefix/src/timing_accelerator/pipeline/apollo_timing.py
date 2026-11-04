from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Literal, Sequence, TypeAlias

from timing_accelerator import __version__
from timing_accelerator.inference.active_timing_contracts import to_json_ready
from timing_accelerator.inference.bayesian_binary_lattice import (
    BinaryAnnealingConfig,
    BinaryCellSearchConfig,
)
from timing_accelerator.inference.bayesian_blind_orbit_discovery import (
    BlindOrbitDiscoveryConfig,
    BlindOrbitDiscoveryResult,
    discover_blind_binary_orbits,
)
from timing_accelerator.inference.bayesian_binary_model_selection import (
    BinaryModelSelectionConfig,
)
from timing_accelerator.io.timing_update import (
    TimingUpdateResult,
    merge_timing_updates,
    normalize_tempo2_tim_file,
)
from timing_accelerator.inference.bayesian_observation_planner import (
    PintObservationPlannerConfig,
)
from timing_accelerator.inference.observation_utility import (
    BayesianObservationDesignConfig,
)
from timing_accelerator.pipeline.bayesian_binary_lattice_solver import (
    BayesianBinaryLatticeSolverConfig,
    BayesianBinaryLatticeSolverResult,
    solve_bayesian_binary_lattice,
)


ApolloTimingResult: TypeAlias = BayesianBinaryLatticeSolverResult
ApolloSourceType: TypeAlias = Literal["binary"]

BLIND_APOLLO_ARCHITECTURE = (
    "apollo_blind_orbit_discovery_plus_phase_lattice_solver"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_json_ready(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _write_model_selection(
    *,
    output: Path,
    selected_family: str,
    discovery: BlindOrbitDiscoveryResult | None,
    candidate_attempts: Sequence[dict[str, object]],
    discovery_error: str | None = None,
    binary_model_selection: dict[str, object] | None = None,
) -> Path:
    return _atomic_json(
        output / "model_selection.json",
        {
            "schema_version": 1,
            "architecture": BLIND_APOLLO_ARCHITECTURE,
            "selected_family": selected_family,
            "discovery_error": discovery_error,
            "blind_discovery": discovery,
            "binary_candidate_attempts": tuple(candidate_attempts),
            "binary_model_selection": binary_model_selection,
        },
    )


def _materialize_blind_binary_result(
    *,
    output: Path,
    result: BayesianBinaryLatticeSolverResult,
    discovery: BlindOrbitDiscoveryResult,
    candidate_attempts: Sequence[dict[str, object]],
) -> BayesianBinaryLatticeSolverResult:
    output.mkdir(parents=True, exist_ok=True)
    artifact_sources = {
        "solution.par": result.solution_par,
        "solution.tim": result.solution_tim,
        "residuals.csv": result.residuals_csv,
        "residuals.png": result.residuals_png,
    }
    for name, source in artifact_sources.items():
        shutil.copy2(Path(source), output / name)

    child_output = Path(result.output_dir)
    for name in ("candidate_portfolio.json",):
        source = child_output / name
        if source.is_file():
            shutil.copy2(source, output / name)

    child_certification = json.loads(
        (child_output / "certification.json").read_text(encoding="utf-8")
    )
    root_artifacts = {
        "par": str(output / "solution.par"),
        "tim": str(output / "solution.tim"),
        "residuals_csv": str(output / "residuals.csv"),
        "residuals_png": str(output / "residuals.png"),
    }
    certification = {
        **child_certification,
        "architecture": BLIND_APOLLO_ARCHITECTURE,
        "engine": "blind_keplerian_discovery_plus_phase_lattice",
        "blind_orbit_discovery": {
            "audit": str(discovery.audit_path),
            "normalized_tim": str(discovery.normalized_tim_path),
            "emitted_mode_count": len(discovery.modes),
            "selected_candidate_output": str(child_output),
        },
        "artifacts": root_artifacts,
    }
    _atomic_json(output / "certification.json", certification)

    child_audit_path = child_output / "audit.json"
    child_audit = (
        json.loads(child_audit_path.read_text(encoding="utf-8"))
        if child_audit_path.is_file()
        else {}
    )
    _atomic_json(
        output / "audit.json",
        {
            **child_audit,
            "architecture": BLIND_APOLLO_ARCHITECTURE,
            "blind_orbit_discovery": discovery,
            "binary_candidate_attempts": tuple(candidate_attempts),
            "selected_candidate_output": str(child_output),
        },
    )
    _write_model_selection(
        output=output,
        selected_family="binary",
        discovery=discovery,
        candidate_attempts=candidate_attempts,
        binary_model_selection=child_certification.get("binary_model_selection"),
    )
    manifest_paths = tuple(
        output / name
        for name in (
            "solution.par",
            "solution.tim",
            "residuals.csv",
            "residuals.png",
            "audit.json",
            "certification.json",
            "model_selection.json",
        )
    )
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "architecture": BLIND_APOLLO_ARCHITECTURE,
            "artifacts": {
                path.name: {"path": str(path), "sha256": _sha256(path)}
                for path in manifest_paths
            },
        },
    )
    return replace(
        result,
        output_dir=output,
        solution_par=output / "solution.par",
        solution_tim=output / "solution.tim",
        residuals_csv=output / "residuals.csv",
        residuals_png=output / "residuals.png",
    )


def par_declares_binary_model(par_path: Path) -> bool:
    for raw_line in Path(par_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "C ")):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].upper() == "BINARY":
            return bool(parts[1].strip())
    return False


def _record_timing_update_artifacts(
    *,
    output: Path,
    timing_update: TimingUpdateResult | None,
    baseline_par_path: Path | None = None,
) -> None:
    if timing_update is None:
        return
    update_payload = json.loads(timing_update.audit_json.read_text(encoding="utf-8"))
    if baseline_par_path is not None:
        update_payload["baseline_par"] = {
            "path": str(baseline_par_path),
            "sha256": _sha256(baseline_par_path),
        }
        _atomic_json(timing_update.audit_json, update_payload)

    audit_path = output / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        conditioning = audit.get("timing_update_conditioning")
        if conditioning is not None:
            update_payload["posterior_conditioning"] = conditioning
            _atomic_json(timing_update.audit_json, update_payload)
        audit["timing_update"] = update_payload
        _atomic_json(audit_path, audit)

    manifest_path = output / "artifact_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = dict(manifest.get("artifacts") or {})
        for path in (
            baseline_par_path,
            timing_update.normalized_baseline_tim,
            timing_update.merged_tim,
            timing_update.audit_json,
        ):
            if path is None:
                continue
            artifacts[path.name] = {
                "path": str(path),
                "sha256": _sha256(path),
            }
        if audit_path.is_file() and "audit.json" in artifacts:
            artifacts["audit.json"] = {
                "path": str(audit_path),
                "sha256": _sha256(audit_path),
            }
        manifest["timing_update"] = update_payload
        manifest["artifacts"] = artifacts
        _atomic_json(manifest_path, manifest)


def _write_input_normalization_audit(
    *,
    source_tim: Path,
    normalized_tim: Path,
    output: Path,
) -> Path:
    return _atomic_json(
        output / "input_normalization.json",
        {
            "schema_version": 1,
            "operation": "canonical_tempo2_normalization",
            "source_tim": {
                "path": str(source_tim),
                "sha256": _sha256(source_tim),
            },
            "normalized_tim": {
                "path": str(normalized_tim),
                "sha256": _sha256(normalized_tim),
            },
            "source_tim_was_modified": False,
        },
    )


def _record_input_normalization_artifacts(
    *,
    output: Path,
    normalized_tim: Path | None,
    normalization_audit: Path | None,
) -> None:
    if normalized_tim is None or normalization_audit is None:
        return
    normalization = json.loads(normalization_audit.read_text(encoding="utf-8"))
    audit_path = output / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["input_normalization"] = normalization
        _atomic_json(audit_path, audit)

    manifest_path = output / "artifact_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = dict(manifest.get("artifacts") or {})
    for path in (normalized_tim, normalization_audit):
        artifacts[path.name] = {
            "path": str(path),
            "sha256": _sha256(path),
        }
    if audit_path.is_file() and "audit.json" in artifacts:
        artifacts["audit.json"] = {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
        }
    manifest["input_normalization"] = normalization
    manifest["artifacts"] = artifacts
    _atomic_json(manifest_path, manifest)


def solve_apollo_timing(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    binary_config: BayesianBinaryLatticeSolverConfig | None = None,
    blind_orbit_config: BlindOrbitDiscoveryConfig | None = None,
    source_type: ApolloSourceType = "binary",
    discover_undeclared_binary: bool = True,
    blind_candidate_support_width_scale: float = 1.0,
    clock_override_path: Path | None = None,
    resume: bool = False,
    update_tim_paths: Sequence[Path] = (),
) -> ApolloTimingResult:
    """Solve a binary model, discovering its orbit when none is supplied."""

    par = Path(par_path)
    if not par.is_file():
        raise FileNotFoundError(par)
    if source_type != "binary":
        raise ValueError(
            "This distribution accepts source_type='binary' only."
        )
    output = Path(output_dir).resolve()
    source_tim = Path(tim_path).resolve()
    if not source_tim.is_file():
        raise FileNotFoundError(source_tim)
    timing_update = None
    persistent_baseline_par = None
    normalized_input_tim = None
    normalization_audit = None
    effective_tim = source_tim
    if update_tim_paths:
        timing_update = merge_timing_updates(
            baseline_tim_path=effective_tim,
            update_tim_paths=update_tim_paths,
            output_dir=output,
        )
        effective_tim = timing_update.merged_tim
        discover_undeclared_binary = False
        persistent_baseline_par = output / "input.baseline.par"
        shutil.copy2(par, persistent_baseline_par)
    else:
        normalized_input_tim = normalize_tempo2_tim_file(
            input_path=source_tim,
            output_path=output / "input.normalized.tim",
        )
        normalization_audit = _write_input_normalization_audit(
            source_tim=source_tim,
            normalized_tim=normalized_input_tim,
            output=output,
        )
        effective_tim = normalized_input_tim

    baseline_par = persistent_baseline_par
    baseline_tim = (
        timing_update.normalized_baseline_tim if timing_update is not None else None
    )

    def finish(result: ApolloTimingResult) -> ApolloTimingResult:
        model_selection_path = output / "model_selection.json"
        certification_path = output / "certification.json"
        if not model_selection_path.is_file() and certification_path.is_file():
            certification = json.loads(certification_path.read_text(encoding="utf-8"))
            _write_model_selection(
                output=output,
                selected_family=str(getattr(result, "model_family", "unknown")),
                discovery=None,
                candidate_attempts=(),
                binary_model_selection=certification.get("binary_model_selection"),
            )
        manifest_path = output / "artifact_manifest.json"
        if model_selection_path.is_file() and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = dict(manifest.get("artifacts") or {})
            artifacts[model_selection_path.name] = {
                "path": str(model_selection_path),
                "sha256": _sha256(model_selection_path),
            }
            manifest["artifacts"] = artifacts
            _atomic_json(manifest_path, manifest)
        _record_input_normalization_artifacts(
            output=output,
            normalized_tim=normalized_input_tim,
            normalization_audit=normalization_audit,
        )
        _record_timing_update_artifacts(
            output=output,
            timing_update=timing_update,
            baseline_par_path=persistent_baseline_par,
        )
        return result

    for raw_line in par.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.split()
        if not tokens or tokens[0].startswith("#") or tokens[0] == "C":
            continue
        if tokens[0].upper() == "BINARY2" or (
            len(tokens) > 1
            and tokens[0].upper() == "BINARY"
            and tokens[1].upper() == "M12BNBODY"
        ):
            raise ValueError("This distribution supports binary models only.")
    declares_binary = par_declares_binary_model(par)
    if declares_binary:
        return finish(
            solve_bayesian_binary_lattice(
                par_path=par,
                tim_path=effective_tim,
                output_dir=output,
                config=binary_config,
                clock_override_path=clock_override_path,
                resume=resume,
                baseline_par_path=baseline_par,
                baseline_tim_path=baseline_tim,
            )
        )
    if source_type == "binary" and not discover_undeclared_binary:
        raise ValueError(
            "Explicit binary mode requires blind orbit discovery when the par file "
            "does not declare BINARY."
        )
    if blind_candidate_support_width_scale <= 0.0:
        raise ValueError("Blind candidate support width scale must be positive.")
    downstream_binary_config = replace(
        binary_config or BayesianBinaryLatticeSolverConfig(),
        support_width_scale=float(blind_candidate_support_width_scale),
    )
    candidate_attempts: list[dict[str, object]] = []
    discovery: BlindOrbitDiscoveryResult | None = None
    discovery_error: str | None = None
    try:
        discovery = discover_blind_binary_orbits(
            par_path=par,
            tim_path=effective_tim,
            output_dir=output / "blind_orbit_discovery",
            config=blind_orbit_config,
            clock_override_path=clock_override_path,
            resume=resume,
        )
    except RuntimeError as error:
        discovery_error = f"{type(error).__name__}: {error}"

    accepted_results: list[BayesianBinaryLatticeSolverResult] = []
    if discovery is not None:
        for mode in discovery.modes:
            candidate_output = output / "blind_binary_candidates" / mode.mode_id
            try:
                candidate_result = solve_bayesian_binary_lattice(
                    par_path=mode.par_path,
                    tim_path=effective_tim,
                    output_dir=candidate_output,
                    config=downstream_binary_config,
                    clock_override_path=clock_override_path,
                    resume=resume,
                )
            except (ArithmeticError, RuntimeError, ValueError) as error:
                candidate_attempts.append(
                    {
                        "mode_id": mode.mode_id,
                        "par": str(mode.par_path),
                        "output": str(candidate_output),
                        "error": f"{type(error).__name__}: {error}",
                        "accepted": False,
                    }
                )
                continue
            candidate_attempts.append(
                {
                    "mode_id": mode.mode_id,
                    "par": str(mode.par_path),
                    "output": str(candidate_output),
                    "chi2": candidate_result.chi2,
                    "reduced_chi2": candidate_result.reduced_chi2,
                    "model_name": getattr(candidate_result, "model_name", None),
                    "model_family": getattr(candidate_result, "model_family", None),
                    "accepted": candidate_result.accepted,
                    "target_reached": candidate_result.target_reached,
                }
            )
            if candidate_result.accepted:
                accepted_results.append(candidate_result)
            if candidate_result.target_reached:
                break

    if accepted_results and discovery is not None:
        best = min(accepted_results, key=lambda item: item.chi2)
        return finish(
            _materialize_blind_binary_result(
                output=output,
                result=best,
                discovery=discovery,
                candidate_attempts=candidate_attempts,
            )
        )

    if source_type == "binary":
        _write_model_selection(
            output=output,
            selected_family="binary_unresolved",
            discovery=discovery,
            candidate_attempts=candidate_attempts,
            discovery_error=discovery_error,
        )
        attempted = len(candidate_attempts)
        detail = discovery_error or "no discovered binary candidate was certified"
        raise RuntimeError(
            "Explicit binary search did not produce a certifiable binary solution "
            f"after {attempted} candidate attempt(s): {detail}. The solver did not "
            "return an unresolved binary candidate."
        )



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run APOLLO binary timing, including blind orbital discovery "
            "when the input par file supplies no orbit."
        )
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument(
        "--update-tim",
        action="append",
        default=[],
        type=Path,
        help=(
            "Add TOAs from another TIM file to the baseline --tim input. May be "
            "repeated; duplicate TOAs are audited and removed."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-type",
        choices=("binary",),
        default="binary",
        help=(
            "Binary-only workflow; retained for reproducible command lines."
        ),
    )
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-blind-orbit-discovery",
        action="store_true",
        help="Require an input orbit; fail if the par file has no BINARY model.",
    )
    parser.add_argument("--blind-workers", type=int, default=1)
    parser.add_argument(
        "--blind-candidate-support-width-scale",
        type=float,
        default=1.0,
        help="Posterior support retained around each discovered binary mode.",
    )
    parser.add_argument("--orbital-samples-per-cell", type=int, default=2)
    parser.add_argument("--phase-samples", type=int, default=12)
    parser.add_argument(
        "--modes-per-cell",
        type=int,
        default=32,
        help="Retain a joint continuous/integer posterior portfolio per cell.",
    )
    parser.add_argument("--chart-evaluations", type=int, default=60)
    parser.add_argument(
        "--phase-sigma-schedule",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Override the binary phase-noise annealing schedule in cycles. "
            "The schedule must be nonincreasing and end at zero; use a "
            "single 0 for the no-annealing ablation."
        ),
    )
    parser.add_argument("--maximum-modes-to-anneal", type=int, default=24)
    parser.add_argument("--maximum-modes-to-certify", type=int, default=16)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--strict-reduced-chi2", type=float, default=20.0)
    parser.add_argument("--target-reduced-chi2", type=float, default=10.0)
    parser.add_argument(
        "--no-binary-model-selection",
        action="store_true",
        help="Keep the seed BINARY model after phase-path certification.",
    )
    parser.add_argument("--model-selection-min-delta-bic", type=float, default=10.0)
    parser.add_argument(
        "--model-selection-trigger-reduced-chi2",
        type=float,
        default=1.5,
    )
    parser.add_argument("--model-selection-multistarts", type=int, default=8)
    parser.add_argument(
        "--no-observation-plan",
        action="store_true",
        help="Do not rank future observing windows after failed certification.",
    )
    parser.add_argument("--observation-horizon-days", type=float, default=180.0)
    parser.add_argument("--observation-duration-min", type=float, default=20.0)
    parser.add_argument("--observation-toas-per-session", type=int, default=8)
    parser.add_argument("--observation-top", type=int, default=20)
    parser.add_argument("--observation-posterior-samples", type=int, default=256)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    annealing = BinaryAnnealingConfig(
        phase_sigma_schedule=(
            tuple(args.phase_sigma_schedule)
            if args.phase_sigma_schedule is not None
            else BinaryAnnealingConfig().phase_sigma_schedule
        )
    )
    binary_config = BayesianBinaryLatticeSolverConfig(
        cell_search=BinaryCellSearchConfig(
            orbital_samples_per_cell=args.orbital_samples_per_cell,
            phase_samples=args.phase_samples,
            modes_per_cell=args.modes_per_cell,
            chart_maximum_evaluations=args.chart_evaluations,
        ),
        annealing=annealing,
        maximum_modes_to_anneal=args.maximum_modes_to_anneal,
        maximum_modes_to_certify=args.maximum_modes_to_certify,
        fit_iterations=args.fit_iterations,
        strict_reduced_chi2_threshold=args.strict_reduced_chi2,
        target_reduced_chi2_threshold=args.target_reduced_chi2,
        physical_model_selection=BinaryModelSelectionConfig(
            enabled=not args.no_binary_model_selection,
            minimum_delta_bic=args.model_selection_min_delta_bic,
            extension_trigger_reduced_chi2=(
                args.model_selection_trigger_reduced_chi2
            ),
            maximum_multistarts_per_hypothesis=args.model_selection_multistarts,
        ),
        plan_observations_on_failure=not args.no_observation_plan,
        observation_planner=PintObservationPlannerConfig(
            horizon_days=args.observation_horizon_days,
            duration_min=args.observation_duration_min,
            toas_per_session=args.observation_toas_per_session,
            maximum_recommendations=args.observation_top,
            design=BayesianObservationDesignConfig(
                posterior_samples_per_mode=args.observation_posterior_samples
            ),
        ),
    )
    blind_orbit_config = BlindOrbitDiscoveryConfig(
        parallel_workers=args.blind_workers,
    )
    result = solve_apollo_timing(
        par_path=args.par,
        tim_path=args.tim,
        output_dir=args.output,
        binary_config=binary_config,
        blind_orbit_config=blind_orbit_config,
        source_type=args.source_type,
        discover_undeclared_binary=(
            not args.no_blind_orbit_discovery
            and not bool(args.update_tim)
        ),
        blind_candidate_support_width_scale=(
            args.blind_candidate_support_width_scale
        ),
        clock_override_path=args.clock_override,
        resume=args.resume,
        update_tim_paths=tuple(args.update_tim),
    )
    import json

    print(json.dumps(to_json_ready(result), indent=2, allow_nan=False))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ApolloTimingResult",
    "ApolloSourceType",
    "BLIND_APOLLO_ARCHITECTURE",
    "build_arg_parser",
    "main",
    "par_declares_binary_model",
    "solve_apollo_timing",
]
