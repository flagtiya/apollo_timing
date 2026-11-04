from __future__ import annotations


from apollo_timing.io.artifacts import sha256_file as _sha256, write_json_atomic


import argparse
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
from functools import wraps
import json
from pathlib import Path
import shlex
import shutil
import sys
from typing import Literal, Sequence, TypeAlias
from uuid import uuid4

from loguru import logger

from apollo_timing import __version__
from apollo_timing.backend.pint_backend import (
    TimingResourceError, _local_ephemeris_identity, _resolve_clock_directory,
    _temporary_ephemeris_override,
)
from apollo_timing.inference.timing_parameter_exploration import ParameterExplorationConfig
from apollo_timing.inference.timing_constraints import bind_sky_constraint
from apollo_timing.inference.active_timing_contracts import to_json_ready
from apollo_timing.io.timing_input import inspect_timing_system
from apollo_timing.io.timing_solver_artifacts import _json_digest, resource_identity
from apollo_timing.inference.bayesian_binary_lattice import (
    BinaryAnnealingConfig,
    BinaryCellSearchConfig,
)
from apollo_timing.inference.bayesian_blind_orbit_discovery import (
    BlindOrbitDiscoveryConfig,
    BlindOrbitDiscoveryResult,
    discover_blind_binary_orbits as discover_blind_binary_orbits,
)
from apollo_timing.inference.bayesian_binary_model_selection import (
    BinaryModelSelectionConfig,
)
from apollo_timing.inference.bayesian_isolated_update_model_selection import (
    IsolatedUpdateModelSelectionConfig,
)
from apollo_timing.inference.isolated_glitch_search import IsolatedGlitchSearchConfig
from apollo_timing.io.timing_update import (
    TimingUpdateResult,
    merge_timing_updates,
    normalize_tempo2_tim_file,
)
from apollo_timing.inference.bayesian_observation_planner import (
    PintObservationPlannerConfig,
)
from apollo_timing.inference.observation_utility import (
    BayesianObservationDesignConfig,
)
from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
    BayesianBinaryLatticeSolverConfig,
    BayesianBinaryLatticeSolverResult,
    solve_bayesian_binary_lattice,
)
from apollo_timing.pipeline.isolated_pulsar_solver import (
    IsolatedAnnealingConfig,
    IsolatedCellSearchConfig,
    IsolatedPulsarSolverConfig,
    IsolatedPulsarSolverResult,
    solve_isolated_pulsar,
)

def _atomic_json(path, payload):
    return write_json_atomic(path, payload, trailing_newline=True)


ApolloTimingResult: TypeAlias = (
    BayesianBinaryLatticeSolverResult | IsolatedPulsarSolverResult
)
ApolloSourceType: TypeAlias = Literal["auto", "single", "binary"]


class _TimingArgumentParser(argparse.ArgumentParser):
    def _parse_optional(self, arg_string: str):
        # argparse recognizes decimal negatives, but not scientific notation.
        if arg_string.startswith("-"):
            try:
                float(arg_string)
            except ValueError:
                pass
            else:
                return None
        return super()._parse_optional(arg_string)

BLIND_APOLLO_ARCHITECTURE = (
    "apollo_blind_orbit_discovery_plus_phase_lattice_solver"
)


def _write_model_selection(
    *,
    output: Path,
    selected_family: str,
    discovery: BlindOrbitDiscoveryResult | None,
    candidate_attempts: Sequence[dict[str, object]],
    discovery_error: str | None = None,
    binary_model_selection: dict[str, object] | None = None,
    isolated_update_model_selection: dict[str, object] | None = None,
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
            "isolated_update_model_selection": isolated_update_model_selection,
        },
    )


def par_declares_binary_model(par_path: Path) -> bool:
    return inspect_timing_system(par_path).binary_model is not None


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


def _tim_input_paths(paths: Sequence[Path]) -> set[Path]:
    pending = [Path(path).resolve() for path in paths]
    inputs = set()
    while pending:
        path = pending.pop()
        if path in inputs:
            continue
        inputs.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[0].upper() == "INCLUDE":
                fields = shlex.split(line, comments=True)
                if len(fields) != 2:
                    raise ValueError(f"Invalid TIM INCLUDE in '{path}': {line}")
                included = Path(fields[1]).expanduser()
                pending.append((path.parent / included).resolve())
    return inputs


def _output_lock(function):
    @wraps(function)
    def locked(*args, **kwargs):
        output = Path(kwargs["output_dir"]).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        # Keep the lock outside the directory that a restart may rename.
        with (output.parent / f".{output.name}.apollo.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another APOLLO run is using this output: {output}") from exc
            try:
                return function(*args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
    return locked


def _prepare_output_directory(output: Path, *, resume: bool, policy: str,
                              protected: Sequence[Path]) -> Path | None:
    if policy not in {"backup", "error"}:
        raise ValueError("output_policy must be 'backup' or 'error'.")
    if output.is_symlink():
        raise ValueError(f"Output must not be a symbolic link: {output}")
    output = output.resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    if resume or not output.exists() or not any(output.iterdir()):
        return None
    if policy == "error":
        raise FileExistsError(f"Output exists: {output}; use --resume or --output-policy backup.")
    # Moving an input or resource directory would invalidate both the new run
    # and its recorded provenance. Never restart in an ancestor of these paths.
    for path in (*protected, Path.home(), Path.cwd(), Path(__file__).resolve()):
        if Path(path).resolve().is_relative_to(output):
            raise ValueError(f"Output contains an input, resource or working directory: {output}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = output.with_name(f"{output.name}.backup-{stamp}-{uuid4().hex[:8]}")
    output.rename(backup)
    logger.warning("Existing output preserved at {}; starting a fresh run in {}", backup, output)
    return backup


@_output_lock
def solve_apollo_timing(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    binary_config: BayesianBinaryLatticeSolverConfig | None = None,
    isolated_config: IsolatedPulsarSolverConfig | None = None,
    blind_orbit_config: BlindOrbitDiscoveryConfig | None = None,
    source_type: ApolloSourceType = "auto",
    discover_undeclared_binary: bool | None = None,
    task: str = "solve",
    system_selection_min_delta_bic: float = 10.0,
    blind_candidate_support_width_scale: float = 1.0,
    clock_override_path: Path | None = None,
    resume: bool = False,
    output_policy: Literal["backup", "error"] = "backup",
    update_tim_paths: Sequence[Path] = (),
) -> ApolloTimingResult:
    """Dispatch to one scientific workflow, with optional automatic discovery."""

    par = Path(par_path)
    if not par.is_file():
        raise FileNotFoundError(par)
    clock_override_path = _resolve_clock_directory(clock_override_path)
    if source_type not in {"auto", "single", "binary"}:
        raise ValueError("source_type must be 'auto', 'single', or 'binary'.")
    if binary_config is not None:
        binary_config = bind_sky_constraint(binary_config, par)
    if isolated_config is not None:
        isolated_config = bind_sky_constraint(isolated_config, par)
    regions = [c.sky_position_constraint for c in (binary_config, isolated_config)
               if c is not None and c.sky_position_constraint is not None]
    if regions:
        if any(region != regions[0] for region in regions):
            raise ValueError("Automatic system hypotheses require the same sky constraint.")
        binary_config = replace(binary_config or BayesianBinaryLatticeSolverConfig(), sky_position_constraint=regions[0])
        isolated_config = replace(isolated_config or IsolatedPulsarSolverConfig(), sky_position_constraint=regions[0])
    if task not in {"solve", "evaluate", "update"}:
        raise ValueError("Timing task must be solve, evaluate or update.")
    if (isolated_config is not None
            and isolated_config.isolated_update_model_selection.glitch_search.enabled
            and source_type != "single"):
        raise ValueError("Explicit glitch search requires source_type='single'.")
    if task == "update" and not update_tim_paths:
        raise ValueError("The update task requires at least one --update-tim file.")
    if task == "evaluate" and update_tim_paths:
        raise ValueError("Evaluation cannot also request a timing update.")
    if task == "evaluate" and (
        (binary_config is not None and (binary_config.parameter_exploration.parameters or binary_config.physical_model_selection.enabled))
        or (isolated_config is not None and (isolated_config.parameter_exploration.parameters or isolated_config.isolated_update_model_selection.enabled))
    ):
        raise ValueError("Evaluation does not authorize parameter or model searches.")
    discovery_authorized = (source_type in {"auto", "binary"} and not update_tim_paths) if discover_undeclared_binary is None else discover_undeclared_binary
    declaration = inspect_timing_system(par)
    from apollo_timing.inference.bayesian_coarse_binary_bootstrap import is_incomplete_binary_par

    expands_structure = task != "evaluate" and source_type != "single" and (
        (declaration.binary_model is not None and is_incomplete_binary_par(par))
        or (declaration.binary_model is None and discovery_authorized)
    )
    if expands_structure and any(
        c is not None and (c.custom_bounds or c.parameter_exploration.bounds
                          or c.sky_position_constraint is not None)
        for c in (binary_config, isolated_config)
    ):
        raise ValueError("Hard parameter/sky constraints cannot accompany orbit initialization, discovery or automatic system comparison. Run discovery first, then constrain a complete declared model; use discovery controls for search domains.")
    if declaration.orphan_orbit_parameters:
        raise ValueError(
            "Orbital parameters without BINARY cannot be treated as isolated or "
            "silently replaced by discovery. Declare the model first: "
            + ", ".join(declaration.orphan_orbit_parameters)
        )
    output_argument = Path(output_dir).expanduser()
    output = output_argument.resolve()
    source_tim = Path(tim_path).resolve()
    if not source_tim.is_file():
        raise FileNotFoundError(source_tim)
    request = dict(schema=1, version=__version__, task="update" if update_tim_paths else task,
        source_type=source_type, discovery_authorized=discovery_authorized,
        par_sha256=_sha256(par), tim_sha256=_sha256(source_tim),
        updates=[dict(path=str(Path(p).resolve()), sha256=_sha256(Path(p))) for p in update_tim_paths],
        binary_config=binary_config, isolated_config=isolated_config,
        blind_config=blind_orbit_config, system_selection_min_delta_bic=system_selection_min_delta_bic,
        blind_candidate_support_width_scale=blind_candidate_support_width_scale,
        clock=resource_identity(clock_override_path),
        local_ephemeris=_local_ephemeris_identity(par))
    request_fingerprint = _json_digest(request)
    protected = [par, *_tim_input_paths((source_tim, *update_tim_paths))]
    if clock_override_path is not None:
        protected.append(clock_override_path)
        logger.info("Clock source: {} (local; no index or network required)", clock_override_path)
    if request["local_ephemeris"] is not None:
        protected.append(Path(request["local_ephemeris"]["path"]))
        logger.info("Ephemeris source: {}", request["local_ephemeris"]["path"])
    output_backup = _prepare_output_directory(
        output_argument, resume=resume, policy=output_policy, protected=protected,
    )
    request_path = output / "request.json"
    if request_path.is_file():
        previous_request = json.loads(request_path.read_text(encoding="utf-8"))
        if previous_request.get("request_fingerprint") != request_fingerprint:
            raise RuntimeError("Timing request inputs, task, authorization or configuration changed; use a fresh output directory.")
    _atomic_json(request_path, dict(request_fingerprint=request_fingerprint, **request))
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
        if regions:
            from apollo_timing.backend.pint_backend import _load_pint_model

            sky_report = regions[0].report(_load_pint_model(result.solution_par))
            if not sky_report["within_bounds"]:
                raise RuntimeError("Final saved solution left the original requested sky region.")
            if certification_path.is_file():
                certification = json.loads(certification_path.read_text(encoding="utf-8"))
                certification["sky_position_constraint"] = sky_report
                _atomic_json(certification_path, certification)
                manifest_path = output / "artifact_manifest.json"
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest.setdefault("artifacts", {})[certification_path.name] = dict(path=str(certification_path), sha256=_sha256(certification_path))
                    _atomic_json(manifest_path, manifest)
        if not model_selection_path.is_file() and certification_path.is_file():
            certification = json.loads(certification_path.read_text(encoding="utf-8"))
            _write_model_selection(
                output=output,
                selected_family=str(getattr(result, "model_family", "unknown")),
                discovery=None,
                candidate_attempts=(),
                binary_model_selection=certification.get("binary_model_selection"),
                isolated_update_model_selection=certification.get(
                    "isolated_update_model_selection"
                ),
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
        certification = json.loads(certification_path.read_text(encoding="utf-8")) if certification_path.is_file() else {}
        summary_path = output / "run_summary.json"
        _atomic_json(summary_path, dict(
            request_fingerprint=request_fingerprint, task=request["task"],
            system_structure=declaration, parameter_policy=(isolated_config or IsolatedPulsarSolverConfig()).parameter_policy,
            model_identity=certification.get("model_identity"), run_outcome=certification.get("run_outcome"),
            automatic_system_selection=certification.get("automatic_system_selection"),
            sky_position_constraint=certification.get("sky_position_constraint"),
            parameter_bounds=certification.get("parameter_bounds"),
            clock_coverage_warnings=certification.get("clock_coverage_warnings", []),
            local_ephemeris=request["local_ephemeris"],
            clock_source=str(clock_override_path) if clock_override_path else "PINT cache/network",
            previous_output_backup=str(output_backup) if output_backup else None,
            physical_interpretation=certification.get("physical_interpretation"),
            certification_scope=certification.get("certification_scope", "fixed_pulse_number_pint"),
            accepted=bool(getattr(result, "accepted", False)), target_reached=bool(getattr(result, "target_reached", False)),
            goal_met=(task == "evaluate" or bool(getattr(result, "accepted", False))), uniqueness_required=False,
            scientific_status=getattr(result, "scientific_status", "not_evaluated"),
            next_steps=[] if getattr(result, "accepted", False) or task == "evaluate" else [
                "Inspect coverage, residuals and candidate paths before attributing failure to insufficient data.",
                "Optional: authorize finite --explore-parameter/--explore-spin-order or --binary-model-selection; these are suggestions, not diagnoses."]
        ))
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for artifact in (request_path, summary_path):
                manifest.setdefault("artifacts", {})[artifact.name] = dict(path=str(artifact), sha256=_sha256(artifact))
            _atomic_json(manifest_path, manifest)
        return result

    declares_binary = declaration.binary_model is not None
    if source_type == "single" and declares_binary:
        raise ValueError(
            "Explicit single-star mode cannot accept a par file with BINARY."
        )
    if task == "evaluate":
        if source_type == "binary" and not declares_binary:
            raise ValueError("Binary evaluation requires a declared complete model.")
        from apollo_timing.pipeline.timing_evaluation import evaluate_timing_model
        return finish(evaluate_timing_model(
            par_path=par, tim_path=effective_tim, output_dir=output,
            config=(binary_config or BayesianBinaryLatticeSolverConfig()) if declares_binary else
                   (isolated_config.to_core_config() if isolated_config is not None else binary_config or BayesianBinaryLatticeSolverConfig()),
            clock_override_path=clock_override_path, resume=resume,
        ))
    if declares_binary and not _par_has_orbital_period(par):
        if not discovery_authorized:
            raise ValueError("The declared binary has no PB/FB0; enable bounded orbit discovery.")
        return finish(_automatic_timing(
            par=par, tim=effective_tim, output=output, source_type=source_type,
            binary_config=binary_config, isolated_config=isolated_config,
            blind_orbit_config=blind_orbit_config, clock=clock_override_path,
            resume=resume, minimum_delta_bic=system_selection_min_delta_bic,
            support_width_scale=blind_candidate_support_width_scale,
            baseline_par=baseline_par, baseline_tim=baseline_tim,
        ))
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
    if source_type == "single" or (
        source_type == "auto" and not discovery_authorized
    ):
        return finish(
            solve_isolated_pulsar(
                par_path=par,
                tim_path=effective_tim,
                output_dir=output,
                config=isolated_config,
                clock_override_path=clock_override_path,
                resume=resume,
                baseline_par_path=baseline_par,
                baseline_tim_path=baseline_tim,
            )
        )

    if not discovery_authorized:
        raise ValueError("An undeclared binary requires bounded orbit discovery.")
    return finish(_automatic_timing(
        par=par, tim=effective_tim, output=output, source_type=source_type,
        binary_config=binary_config, isolated_config=isolated_config,
        blind_orbit_config=blind_orbit_config, clock=clock_override_path,
        resume=resume, minimum_delta_bic=system_selection_min_delta_bic,
        support_width_scale=blind_candidate_support_width_scale,
        baseline_par=baseline_par, baseline_tim=baseline_tim,
    ))


def _par_has_orbital_period(path: Path) -> bool:
    from apollo_timing.io.par_compatibility import _active_parameter_line
    return any(item[0] in {"PB", "FB0"} for line in path.read_text(encoding="utf-8").splitlines()
               if (item := _active_parameter_line(line)) is not None)


def _automatic_timing(*, binary_config=None, isolated_config=None, blind_orbit_config=None,
                      support_width_scale=1.0, **kwargs):
    from apollo_timing.pipeline.automatic_timing import solve_automatic_timing
    if not 0 < support_width_scale < float("inf"):
        raise ValueError("Blind candidate support width scale must be positive and finite.")
    return solve_automatic_timing(
        binary_config=replace(binary_config or BayesianBinaryLatticeSolverConfig(),
                              support_width_scale=support_width_scale),
        isolated_config=isolated_config or IsolatedPulsarSolverConfig(),
        blind_config=blind_orbit_config or BlindOrbitDiscoveryConfig(), **kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = _TimingArgumentParser(
        prog="apollo-timing",
        description=(
            "Run APOLLO timing search with automatic isolated/binary model "
            "dispatch from the input par file."
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
    parser.add_argument("--output", required=True, type=Path,
                        help="Result directory. Existing contents are backed up before a fresh run.")
    parser.add_argument("--output-policy", choices=("backup", "error"), default="backup",
                        help="Existing output: backup and restart (default), or stop without changes. --resume never overwrites.")
    parser.add_argument(
        "--source-type",
        choices=("auto", "single", "binary"),
        default="auto",
        help=(
            "Select the single-star or binary workflow. "
            "The default auto preserves declarations; otherwise it compares bounded "
            "isolated/binary hypotheses. Use --no-blind-orbit-discovery to opt out."
        ),
    )
    parser.add_argument("--clock-dir", "--clock-override", dest="clock_override", type=Path,
                        help="Local clock directory; defaults to bundled clocks when available. No index.txt needed or remote fallback for missing local files.")
    ephemeris = parser.add_mutually_exclusive_group()
    ephemeris.add_argument("--ephem-file", type=Path,
                           help="Local BSP file for the PAR's EPHEM, without network fallback.")
    ephemeris.add_argument("--ephem-dir", type=Path,
                           help="Local directory containing EPHEM-named BSP files, e.g. de430.bsp.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the same run, strictly checking inputs, resources and configuration.")
    parser.add_argument("--custom-bounds", default=None, help="Legacy JSON physical bounds; equivalent to repeated --parameter-bound.")
    parser.add_argument("--task", choices=("solve", "evaluate", "update"), default="solve",
                        help="Evaluate without fitting, solve, or update with --update-tim.")
    parser.add_argument("--parameter-policy", choices=("input", "auto-basic"), default="input",
                        help="Preserve input fit flags, or explicitly release the isolated basic block.")
    parser.add_argument("--system-selection-min-delta-bic", type=float, default=10.0)
    parser.add_argument("--support-width-scale", type=float, default=1.0)
    parser.add_argument("--maximum-cell-width", type=float, default=0.25)
    parser.add_argument("--astrometric-bound-arcsec", type=float, default=300.0)
    parser.add_argument("--sky-radius-arcmin", type=float, default=None,
                        help="Joint spherical position limit about the input par position at POSEPOCH (PEPOCH fallback); not per-coordinate widths or fit authorization.")
    discovery = parser.add_mutually_exclusive_group()
    discovery.add_argument(
        "--discover-binary", action="store_true",
        help="Authorize bounded orbit discovery when BINARY is absent.",
    )
    discovery.add_argument(
        "--no-blind-orbit-discovery",
        action="store_true",
        help="Treat a par without BINARY as isolated without testing an orbit.",
    )
    parser.add_argument("--blind-workers", type=int, default=1)
    parser.add_argument("--discovery-period-min-days", type=float, default=0.2)
    parser.add_argument("--discovery-period-max-days", type=float, default=2000.0)
    parser.add_argument("--discovery-preset", choices=("balanced", "thorough"), default="thorough")
    parser.add_argument("--discovery-max-modes", type=int, default=4)
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
            "Override the single/binary phase-noise annealing schedule in cycles. "
            "The schedule must be nonincreasing and end at zero; use a "
            "single 0 for the no-annealing ablation."
        ),
    )
    parser.add_argument("--maximum-modes-to-anneal", type=int, default=24)
    parser.add_argument("--maximum-modes-to-certify", type=int, default=16)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--strict-reduced-chi2", type=float, default=20.0)
    parser.add_argument("--target-reduced-chi2", type=float, default=10.0)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--binary-model-selection", action="store_true",
        help="Authorize fixed-path binary physical-model comparison.",
    )
    comparison.add_argument(
        "--no-binary-model-selection",
        action="store_true",
        help="Keep the seed BINARY model after phase-path certification.",
    )
    parser.add_argument("--model-selection-min-delta-bic", type=float, default=10.0)
    parser.add_argument("--binary-models", nargs="+", metavar="MODEL",
                        help="Restrict the explicitly enabled binary model portfolio (e.g. ELL1 DD).")
    parser.add_argument(
        "--model-selection-trigger-reduced-chi2",
        type=float,
        default=1.5,
    )
    parser.add_argument("--model-selection-multistarts", type=int, default=8)
    isolated_selection = parser.add_mutually_exclusive_group()
    isolated_selection.add_argument("--isolated-model-selection", action="store_true",
        help="Authorize the legacy isolated basic/high-order fixed-path portfolio.")
    isolated_selection.add_argument(
        "--no-isolated-update-model-selection",
        action="store_true",
        help="Preserve input fit flags when updating an isolated timing model.",
    )
    parser.add_argument("--isolated-update-max-spin-order", type=int, default=5)
    parser.add_argument("--isolated-update-min-delta-bic", type=float, default=10.0)
    parser.add_argument("--isolated-update-min-sigma", type=float, default=5.0)
    parser.add_argument("--glitch-search", action="store_true",
                        help="Fit a new phase-continuous single-star glitch on retained paths; preserve input fit flags and compare smooth controls.")
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument("--glitch-epoch-range", nargs=2, type=float, metavar=("LOWER_MJD", "UPPER_MJD"))
    epoch_group.add_argument("--glitch-epoch-mjd", type=float,
                             help="Externally fixed event epoch; otherwise profile-search the declared range or observed span.")
    parser.add_argument("--glitch-epoch-samples", type=int, default=24)
    parser.add_argument("--glitch-frequency-order", type=int, choices=(1, 2), default=2,
                        help="1: GLF0; 2: compare GLF0 with GLF0+GLF1. GLPH stays fixed at zero.")
    parser.add_argument("--glitch-frequency-half-width", type=float, default=5e-7, metavar="HZ")
    parser.add_argument("--glitch-f1-half-width", type=float, default=5e-14, metavar="HZ_PER_S")
    parser.add_argument("--glitch-smooth-order", type=int, choices=range(2, 6), default=2)
    parser.add_argument("--explore-parameter", action="append", default=[], metavar="NAME",
                        help="Authorize one optional parameter; repeat for a finite hypothesis graph.")
    parser.add_argument("--explore-spin-order", type=int, choices=range(2, 13), metavar="2..12")
    parser.add_argument("--parameter-bound", nargs=3, action="append", default=[], metavar=("NAME", "LOWER", "UPPER"),
                        help="Hard interval for fixed-model timing in PINT value units; does not release a parameter or combine with exploration.")
    parser.add_argument("--exploration-bound", nargs=3, action="append", default=[], metavar=("NAME", "LOWER", "UPPER"),
                        help="Finite search domain for an explicitly authorized optional parameter; not a hard constraint on the input model.")
    parser.add_argument("--exploration-max-hypotheses", type=int, default=16)
    parser.add_argument("--exploration-max-evaluations", type=int, default=500)
    parser.add_argument("--exploration-min-delta-bic", type=float, default=10.0)
    parser.add_argument("--exploration-min-sigma", type=float, default=5.0)
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
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.custom_bounds is not None:
        try:
            legacy = json.loads(args.custom_bounds)
            if not isinstance(legacy, dict):
                raise ValueError("Custom bounds must be a JSON object")
            for name, interval in legacy.items():
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    raise ValueError("Each custom bound needs [lower, upper]")
                args.parameter_bound.append([name, str(interval[0]), str(interval[1])])
        except (ValueError, TypeError) as error:
            raise SystemExit(f"Invalid --custom-bounds: {error}") from error
    parameters = list(args.explore_parameter)
    if args.explore_spin_order is not None:
        parameters.extend(f"F{order}" for order in range(2, args.explore_spin_order + 1))
    try:
        if args.glitch_search and args.source_type != "single":
            raise ValueError("--glitch-search requires --source-type single.")
        if args.glitch_search and args.no_isolated_update_model_selection:
            raise ValueError("--glitch-search cannot disable isolated model selection.")
        glitch_controls = ("glitch_epoch_range", "glitch_epoch_mjd", "glitch_epoch_samples",
                           "glitch_frequency_order", "glitch_frequency_half_width",
                           "glitch_f1_half_width", "glitch_smooth_order")
        if not args.glitch_search and any(getattr(args, n) != parser.get_default(n) for n in glitch_controls):
            raise ValueError("Glitch search controls require --glitch-search.")
        glitch_config = IsolatedGlitchSearchConfig(
            enabled=args.glitch_search,
            epoch_bounds_mjd=tuple(args.glitch_epoch_range) if args.glitch_epoch_range else None,
            fixed_epoch_mjd=args.glitch_epoch_mjd, epoch_grid_size=args.glitch_epoch_samples,
            maximum_frequency_order=args.glitch_frequency_order,
            frequency_half_width_hz=args.glitch_frequency_half_width,
            frequency_derivative_half_width_hz_s=args.glitch_f1_half_width,
            smooth_spin_order=args.glitch_smooth_order,
        )
        if args.glitch_search and args.parameter_policy != "input":
            raise ValueError("Explicit glitch search preserves input fit flags; use --parameter-policy input.")
        exploration = ParameterExplorationConfig(parameters=tuple(parameters),
            bounds=tuple((name.upper(), float(lo), float(hi)) for name, lo, hi in args.parameter_bound),
            search_bounds=tuple((name.upper(), float(lo), float(hi)) for name, lo, hi in args.exploration_bound),
            maximum_hypotheses=args.exploration_max_hypotheses,
            maximum_evaluations=args.exploration_max_evaluations,
            minimum_delta_bic=args.exploration_min_delta_bic, minimum_sigma=args.exploration_min_sigma)
        if args.task == "evaluate" and (parameters or args.binary_model_selection or args.isolated_model_selection or args.glitch_search):
            raise ValueError("Evaluation does not authorize parameter or model searches.")
        if args.binary_models and not args.binary_model_selection:
            raise ValueError("--binary-models requires --binary-model-selection.")
        if (args.parameter_bound or args.sky_radius_arcmin is not None) and (
            parameters or args.binary_model_selection or args.isolated_model_selection or args.glitch_search
        ):
            raise ValueError("Hard parameter/sky constraints require fixed-model timing; run exploration separately with --exploration-bound.")
        lower, upper = args.discovery_period_min_days, args.discovery_period_max_days
        import math
        if args.sky_radius_arcmin is not None and not (math.isfinite(args.sky_radius_arcmin) and 0 < args.sky_radius_arcmin < 10800):
            raise ValueError("Sky radius must be between 0 and 10800 arcmin (exclusive).")
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError("Discovery periods require finite 0 < minimum < maximum.")
    except ValueError as error:
        parser.error(str(error))
    annealing = BinaryAnnealingConfig(
        phase_sigma_schedule=(
            tuple(args.phase_sigma_schedule)
            if args.phase_sigma_schedule is not None
            else BinaryAnnealingConfig().phase_sigma_schedule
        )
    )
    binary_config = BayesianBinaryLatticeSolverConfig(
        sky_radius_arcmin=args.sky_radius_arcmin,
        parameter_exploration=exploration,
        maximum_cell_width=args.maximum_cell_width,
        support_width_scale=args.support_width_scale,
        astrometric_bound_arcsec=args.astrometric_bound_arcsec,
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
            enabled=args.binary_model_selection,
            enabled_models=tuple(args.binary_models) if args.binary_models else BinaryModelSelectionConfig().enabled_models,
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
    isolated_config = IsolatedPulsarSolverConfig(
        sky_radius_arcmin=args.sky_radius_arcmin,
        annealing=IsolatedAnnealingConfig(phase_sigma_schedule=tuple(args.phase_sigma_schedule)) if args.phase_sigma_schedule is not None else IsolatedAnnealingConfig(),
        parameter_policy=args.parameter_policy,
        parameter_exploration=exploration,
        maximum_cell_width=args.maximum_cell_width,
        support_width_scale=args.support_width_scale,
        astrometric_bound_arcsec=args.astrometric_bound_arcsec,
        cell_search=IsolatedCellSearchConfig(
            phase_samples=args.phase_samples,
            modes_per_cell=args.modes_per_cell,
            chart_maximum_evaluations=args.chart_evaluations,
        ),
        maximum_modes_to_anneal=args.maximum_modes_to_anneal,
        maximum_modes_to_certify=args.maximum_modes_to_certify,
        fit_iterations=args.fit_iterations,
        strict_reduced_chi2_threshold=args.strict_reduced_chi2,
        target_reduced_chi2_threshold=args.target_reduced_chi2,
        isolated_update_model_selection=IsolatedUpdateModelSelectionConfig(
            enabled=args.isolated_model_selection or args.glitch_search,
            maximum_spin_frequency_order=args.isolated_update_max_spin_order,
            minimum_delta_bic=args.isolated_update_min_delta_bic,
            minimum_added_parameter_sigma=args.isolated_update_min_sigma,
            glitch_search=glitch_config,
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
    strata = sorted({lower, upper, *(edge for pair in BlindOrbitDiscoveryConfig().period_strata_days
                                     for edge in pair if lower < edge < upper)})
    blind_orbit_config = BlindOrbitDiscoveryConfig(parallel_workers=args.blind_workers,
        period_strata_days=tuple(zip(strata[:-1], strata[1:])), maximum_modes_to_emit=args.discovery_max_modes)
    if args.discovery_preset == "balanced":
        blind_orbit_config = replace(blind_orbit_config, random_seeds=(19,), sparse_random_seeds=(19, 73),
            sparse_period_strata_to_expand=1, differential_evolution_population=8, differential_evolution_iterations=100,
            conditional_period_samples=1024, conditional_period_anchor_modes=16,
            conditional_period_max_anchor_combinations=2048, exact_refinement_evaluations=2000)
    try:
        with _temporary_ephemeris_override(args.ephem_file, args.ephem_dir):
            result = solve_apollo_timing(
                par_path=args.par,
                tim_path=args.tim,
                output_dir=args.output,
                binary_config=binary_config,
                isolated_config=isolated_config,
                blind_orbit_config=blind_orbit_config,
                source_type=args.source_type,
                task=args.task,
                system_selection_min_delta_bic=args.system_selection_min_delta_bic,
                discover_undeclared_binary=(
                    False if args.no_blind_orbit_discovery else True if args.discover_binary else None
                ),
                blind_candidate_support_width_scale=(
                    args.blind_candidate_support_width_scale
                ),
                clock_override_path=args.clock_override,
                resume=args.resume,
                output_policy=args.output_policy,
                update_tim_paths=tuple(args.update_tim),
            )
    except TimingResourceError as error:
        print(f"APOLLO resource error (not an orbit rejection): {error}", file=sys.stderr)
        if (args.output / "request.json").is_file():
            _atomic_json(args.output / "resource_error.json", dict(
                category="timing_resource_unavailable", message=str(error),
                scientific_interpretation="not_an_orbit_rejection", exit_code=3,
            ))
        return 3
    print(json.dumps(to_json_ready(result), indent=2, allow_nan=False))
    return 0 if args.task == "evaluate" or result.accepted else 2


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

__all__ += ['discover_blind_binary_orbits']
