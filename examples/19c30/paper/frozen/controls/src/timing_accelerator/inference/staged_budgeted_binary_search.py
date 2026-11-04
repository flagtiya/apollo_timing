from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from itertools import combinations, product
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.active_timing_contracts import (
    ActiveTimingBranch,
    BranchCertification,
    to_json_ready,
)
from timing_accelerator.inference.adaptive_residual_bayesian_lattice import (
    AdaptiveResidualConfig,
    ClusterResidualFingerprint,
    build_cluster_residual_fingerprints,
    generate_residual_guided_branches,
)
from timing_accelerator.inference.alias_branch_certifier import build_alias_branch_certifier
from timing_accelerator.inference.alias_manifold_solver import (
    AliasManifoldCandidate,
    LocalPhaseSlopeEvidence,
)
from timing_accelerator.inference.bayesian_alias_lattice import (
    BayesianAliasLatticeConfig,
    _condition_on_local_phase_slopes,
    generate_bayesian_alias_lattice_branches,
)
from timing_accelerator.inference.branch_proposal import (
    branch_to_alias_candidate,
)
from timing_accelerator.inference.budgeted_bayesian_decision import (
    BudgetedDecisionConfig,
    select_budgeted_decision_branches,
)
from timing_accelerator.inference.cluster_manifest import (
    load_cluster_manifest,
    resolve_cluster_manifest,
    resolve_source_session_manifest,
    write_cluster_manifest,
)
from timing_accelerator.inference.certifier_pool import certify_branches
from timing_accelerator.inference.pint_alias_geometry import build_pint_alias_geometry
from timing_accelerator.inference.pint_solution_artifacts import (
    write_certified_solution_parfile,
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.inference.multimode_session_lattice import (
    MultimodeSessionLatticeConfig,
    generate_multimode_session_lattice_branches,
)
from timing_accelerator.inference.pulse_number_reference import (
    ensure_pulse_numbers,
    prepare_stable_pulse_number_reference,
)
from timing_accelerator.inference.wrap_invariant_binary_likelihood import (
    PintSessionShapeLikelihood,
    SessionShapeLinearization,
    SessionShapeProfile,
)
from timing_accelerator.models.contracts import ClusterManifest
from timing_accelerator.inference.prior_transform import (
    build_parameter_prior_specs,
    extract_fit_flagged_parameters,
)
from timing_accelerator.inference.wrap_state import apply_wraps_to_toas


@dataclass(frozen=True)
class PrefixTimingInputs:
    tim_path: Path
    cluster_manifest_path: Path
    cluster_count: int
    gap_count: int


@dataclass(frozen=True)
class StagedBudgetedSearchConfig:
    start_cluster_count: int = 4
    end_cluster_count: int | None = None
    beam_size: int = 24
    certify_budget: int = 96
    final_certify_budget: int | None = None
    initial_lattice_enabled: bool = True
    initial_lattice_top_k: int = 96
    initial_lattice_beam_size: int = 5000
    initial_lattice_phase_sigma: float | None = None
    initial_lattice_prior_scale: float = 1.0
    initial_lattice_prior_scales: tuple[float, ...] = ()
    initial_lattice_radius_sigma: float = 5.0
    initial_lattice_min_radius: int = 4
    initial_lattice_max_enum_per_gap: int = 101
    stage_lattice_refresh_enabled: bool = False
    stage_lattice_refresh_top_k: int = 64
    stage_lattice_refresh_beam_size: int = 5000
    stage_lattice_refresh_phase_sigma: float | None = None
    stage_lattice_refresh_prior_scale: float = 1.0
    stage_lattice_refresh_prior_scales: tuple[float, ...] = ()
    stage_lattice_refresh_radius_sigma: float = 5.0
    stage_lattice_refresh_min_radius: int = 4
    stage_lattice_refresh_max_enum_per_gap: int = 101
    adaptive_global_reacquisition_enabled: bool = True
    adaptive_global_reacquisition_chi2_ratio: float = 4.0
    adaptive_global_reacquisition_min_reduced_chi2: float = 5.0
    adaptive_global_reacquisition_top_k_multiplier: float = 2.0
    adaptive_global_reacquisition_certify_budget_multiplier: float = 2.0
    adaptive_prior_tempering_enabled: bool = True
    adaptive_prior_tempering_factor: float = 2.0
    adaptive_prior_tempering_steps: int = 2
    adaptive_slope_tempering_enabled: bool = True
    adaptive_slope_tempering_factor: float = 2.0
    adaptive_multi_anchor_reacquisition_enabled: bool = True
    adaptive_multi_anchor_parent_top_k: int = 8
    adaptive_multi_anchor_candidate_top_k: int = 96
    adaptive_multi_anchor_certify_budget_multiplier: float = 1.0
    adaptive_multi_anchor_max_rounds: int = 3
    adaptive_multi_anchor_min_fractional_improvement: float = 0.01
    conditioned_lattice_refresh_enabled: bool = False
    conditioned_lattice_parent_top_k: int = 8
    conditioned_lattice_top_k: int = 64
    conditioned_lattice_per_parent_top_k: int = 16
    conditioned_lattice_beam_size: int | None = None
    conditioned_lattice_window_gaps: int | None = None
    conditioned_lattice_prior_scale: float | None = None
    conditioned_lattice_prior_scales: tuple[float, ...] = ()
    posterior_lattice_refresh_enabled: bool = False
    posterior_lattice_parent_top_k: int = 8
    posterior_lattice_top_k: int = 96
    posterior_lattice_per_parent_top_k: int = 16
    posterior_lattice_beam_size: int | None = None
    posterior_lattice_phase_sigma: float | None = None
    posterior_lattice_prior_scale: float | None = None
    posterior_lattice_prior_scales: tuple[float, ...] = ()
    posterior_lattice_radius_sigma: float | None = None
    posterior_lattice_min_radius: int | None = None
    posterior_lattice_max_enum_per_gap: int | None = None
    posterior_lattice_stable_order_enabled: bool = False
    posterior_lattice_stable_order_top_k: int = 192
    posterior_lattice_stable_order_per_parent_top_k: int = 72
    formal_uncertainty_scale: float = 1000.0
    physical_prior_width_fraction: float = 0.25
    wrap_radius: int = 2
    exploration_wrap_radius: int = 8
    exploration_stride: int = 2
    phase_sigma: float = 0.20
    wrap_prior_scale: float = 2000.0
    offset_prior_scale: float = 1.0
    maxiter: int = 40
    decision_quality_floor: int = 1
    decision_quality_floor_fraction: float = 0.10
    decision_family_floor: int = 1
    decision_family_floor_fraction: float = 0.0
    decision_portfolio_floor: int = 0
    decision_quality_weight: float = 1.0
    decision_diagnostic_weight: float = 1.0
    decision_diversity_weight: float = 0.45
    decision_uncertainty_weight: float = 0.75
    decision_prior_weight: float = 0.10
    decision_lookahead_weight: float = 1.0
    parent_lookahead_phase_sigma: float | None = None
    parent_lookahead_cluster_horizon: int | None = 1
    parent_lookahead_end_cluster_count: int | None = None
    phase_slope_enabled: bool = True
    phase_slope_sigma_scale: float = 1.0
    phase_slope_sigma_floor: float = 0.0
    session_lattice_enabled: bool = False
    session_lattice_final_only: bool = True
    session_lattice_trigger_reduced_chi2: float = 5.0
    session_lattice_max_modes: int = 8
    session_lattice_candidates_per_mode: int = 16
    session_lattice_max_candidates: int = 64
    session_lattice_max_visited_nodes: int = 1_000_000
    session_lattice_decimal_precision: int = 80
    session_lattice_finite_difference_fraction: float = 1.0e-4
    session_lattice_shape_optimization_enabled: bool = False
    session_lattice_shape_optimization_max_modes: int = 4
    session_lattice_shape_optimization_trust_radius: float = 64.0
    session_lattice_shape_optimization_max_nfev: int = 100
    session_lattice_relinearization_rounds: int = 3
    session_lattice_certify_budget: int = 64
    session_lattice_min_fractional_improvement: float = 0.01
    information_rank_start_enabled: bool = False
    information_rank_rtol: float = 1.0e-4
    decision_prior_score_mode: str = "family-relative-proposal"
    decision_max_family_fraction: float = 0.55
    decision_basin_floor: int = 1
    decision_max_basin_fraction: float = 0.35
    certify_parallel_workers: int = 1
    certify_parallel_backend: str = "thread"
    residual_probe_offsets: tuple[float, ...] = (-1.0, 0.0, 1.0)
    residual_repair_enabled: bool = True
    residual_repair_rounds: int = 1
    residual_repair_top_k: int = 48
    residual_repair_parent_top_k: int = 8
    residual_repair_ridge: float = 1.0e-4
    residual_repair_step_scales: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)
    residual_repair_block_step_scales: tuple[float, ...] = (-1.0, 1.0)
    residual_repair_max_gap_wrap_shift: int = 3
    residual_repair_gap_wrap_shift_scales: tuple[float, ...] = (0.5, 1.0)
    residual_repair_gap_wrap_top_k: int = 8
    residual_repair_gap_wrap_joint_scale: float = 0.25
    session_reconnect_enabled: bool = True
    session_reconnect_top_k: int = 48
    session_reconnect_parent_top_k: int = 12
    session_reconnect_cluster_top_k: int = 8
    session_reconnect_max_wrap_shift: int = 8
    session_reconnect_residual_floor: float = 0.25
    session_reconnect_joint_scale: float = 0.30
    session_reconnect_prior_scale: float = 24.0
    uncertainty_resample_enabled: bool = True
    uncertainty_resample_top_k: int = 32
    uncertainty_resample_parent_top_k: int = 6
    uncertainty_resample_gap_top_k: int = 6
    uncertainty_resample_max_gap_shift: int = 3
    uncertainty_resample_combination_size: int = 2
    uncertainty_resample_residual_floor: float = 0.15
    uncertainty_resample_joint_scale: float = 0.20
    residual_noise_floor: float = 1.0e-3
    parameter_release_enabled: bool = False
    parameter_release_candidate_blocks: tuple[Any, ...] = (
        ("F2",),
    )
    parameter_release_probe_top_k: int = 8
    parameter_release_min_cluster_count: int = 0
    parameter_release_min_wrms_improvement_us: float = 5.0
    parameter_release_min_fractional_improvement: float = 0.05
    parameter_release_bic_penalty_weight: float = 1.0
    parameter_release_min_evidence_gain: float = 10.0
    parameter_release_max_block_size: int = 2
    stop_reduced_chi2: float | None = None
    stop_weighted_rms_us: float | None = None
    keep_stage_inputs: bool = True
    resume_from_stage_artifacts: bool = False
    refresh_parent_cache_enabled: bool = True


@dataclass(frozen=True)
class StagedSearchResult:
    branches: list[ActiveTimingBranch]
    certifications: list[BranchCertification]
    best_branch: ActiveTimingBranch | None
    best_certification: BranchCertification | None
    stage_records: list[dict[str, Any]]
    output_dir: Path
    active_par_path: Path | None = None
    fit_parameter_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StagedResumeState:
    current_branches: list[ActiveTimingBranch]
    certifications: list[BranchCertification]
    stage_records: list[dict[str, Any]]
    next_cluster_count: int
    resumed_stage_count: int


@dataclass(frozen=True)
class StagedResidualRepairPayload:
    cluster_ids: tuple[str, ...]
    cluster_mjd: np.ndarray
    median_phase_residual: np.ndarray
    mad_phase_residual: np.ndarray
    toa_counts: np.ndarray
    gap_phase_residual: np.ndarray
    cluster_design_matrix: np.ndarray
    gap_jacobian: np.ndarray
    reference_phase_residuals: np.ndarray


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(to_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _next_attempt_dir(parent: Path) -> tuple[Path, int]:
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    existing_numbers = []
    for path in parent.glob("attempt_*"):
        try:
            existing_numbers.append(int(path.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt_number = max(existing_numbers, default=0) + 1
    while True:
        attempt_dir = parent / f"attempt_{attempt_number:04d}"
        try:
            attempt_dir.mkdir(parents=False, exist_ok=False)
            return attempt_dir, attempt_number
        except FileExistsError:
            attempt_number += 1


def _artifact_score(metadata: dict[str, Any]) -> tuple[float, float, str]:
    reduced_chi2 = metadata.get("reduced_chi2")
    weighted_rms_us = metadata.get("weighted_rms_us")
    return (
        float(reduced_chi2)
        if reduced_chi2 is not None and np.isfinite(float(reduced_chi2))
        else float("inf"),
        float(weighted_rms_us)
        if weighted_rms_us is not None and np.isfinite(float(weighted_rms_us))
        else float("inf"),
        str(metadata.get("attempt_id", "")),
    )


def _candidate_improves_best(candidate: dict[str, Any], best_path: Path) -> bool:
    if not Path(best_path).exists():
        return True
    incumbent = _load_json(best_path)
    return _artifact_score(candidate) < _artifact_score(incumbent)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_bytes(source.read_bytes())
    temporary_path.replace(destination)


def _verify_artifact_pair(par_path: Path, metadata_path: Path) -> None:
    if not Path(par_path).read_bytes():
        raise OSError(f"Artifact par file is empty: {par_path}")
    _load_json(metadata_path)


def _metadata_par_path(output_dir: Path, metadata: dict[str, Any]) -> Path:
    raw_path = metadata.get("par_path", metadata.get("best_par_path"))
    if raw_path is None:
        raise ValueError("Artifact metadata does not contain a par path.")
    path = Path(raw_path)
    return path if path.is_absolute() else Path(output_dir) / path


def _write_pulse_number_reference_parfile(
    *,
    path: Path,
    seed_par_path: Path,
    branch: ActiveTimingBranch,
) -> tuple[Path, str]:
    candidate = branch_to_alias_candidate(branch)
    path = Path(path)
    if candidate.pulse_number_reference == "seed":
        _copy_file_atomic(seed_par_path, path)
        return path, "seed"

    model = _load_pint_model(seed_par_path)
    for name, offset in zip(branch.theta_names, candidate.parameter_offset, strict=True):
        if name not in model.params:
            raise ValueError(f"Pulse-number reference parameter '{name}' is missing from the timing model.")
        _set_numeric_model_parameter_value(
            model,
            name,
            float(model[name].value) + float(offset),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    write_pint_model_parfile_roundtrip_safe(model=model, path=temporary_path)
    temporary_path.replace(path)
    return path, "candidate"


def _write_anchor_model_parfile(
    *,
    path: Path,
    seed_par_path: Path,
    theta_names: Sequence[str],
    anchor_offset: np.ndarray,
) -> Path:
    names = tuple(str(name).upper() for name in theta_names)
    offset = np.asarray(anchor_offset, dtype=float)
    if offset.shape != (len(names),) or np.any(~np.isfinite(offset)):
        raise ValueError("anchor_offset must be finite and match theta_names.")
    model = _load_pint_model(Path(seed_par_path))
    for name, delta in zip(names, offset, strict=True):
        if name not in model.params:
            raise ValueError(f"Anchor parameter '{name}' is missing from the timing model.")
        _set_numeric_model_parameter_value(
            model,
            name,
            float(model[name].value) + float(delta),
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    write_pint_model_parfile_roundtrip_safe(model=model, path=temporary_path)
    temporary_path.replace(path)
    return path


def _write_anchor_pulse_number_reference(
    *,
    path: Path,
    anchor_par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parent_wraps: Sequence[int],
    clock_override_path: Path | None,
) -> tuple[Path, tuple[int, ...]]:
    model = _load_pint_model(Path(anchor_par_path))
    toas = _load_pint_toas(
        Path(tim_path),
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    ensure_pulse_numbers(toas, model)
    manifest = resolve_cluster_manifest(
        load_cluster_manifest(Path(cluster_manifest_path)),
        Path(tim_path),
    )
    target_gap_count = max(len(manifest.clusters) - 1, 0)
    inherited = tuple(int(item) for item in parent_wraps)
    if len(inherited) > target_gap_count:
        raise ValueError("Parent wrap dimension exceeds the anchor prefix gap count.")
    inherited = inherited + tuple(0 for _ in range(target_gap_count - len(inherited)))
    adjusted_toas = apply_wraps_to_toas(toas, manifest, list(inherited))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    adjusted_toas.write_TOA_file(
        str(temporary_path),
        format="tempo2",
        include_pn=True,
        include_info=True,
        comment="APOLLO nonlinear anchor pulse-number reference",
    )
    temporary_path.replace(path)
    return path, inherited


def _write_stage_best_artifacts(
    *,
    output_dir: Path,
    stage_index: int,
    active_par_path: Path,
    fit_parameter_names: Sequence[str] | None,
    branch: ActiveTimingBranch | None,
    certification: BranchCertification | None,
    tim_path: Path | None = None,
    clock_override_path: Path | None = None,
    stable_pulse_number_reference_path: Path | None = None,
) -> dict[str, Any] | None:
    if branch is None or certification is None or not certification.fitted_theta:
        return None
    output_dir = Path(output_dir)
    history_dir = output_dir / "best_par_history"
    artifact_stem = f"stage_{int(stage_index):02d}_best"
    attempt_dir, attempt_number = _next_attempt_dir(
        history_dir / "attempts" / f"stage_{int(stage_index):02d}"
    )
    if stable_pulse_number_reference_path is not None:
        pulse_reference_path = attempt_dir / "pulse_number_reference.par"
        _copy_file_atomic(stable_pulse_number_reference_path, pulse_reference_path)
        pulse_reference = "stable-full-tim"
    else:
        pulse_reference_path, pulse_reference = _write_pulse_number_reference_parfile(
            path=attempt_dir / "pulse_number_reference.par",
            seed_par_path=active_par_path,
            branch=branch,
        )
    par_path = write_certified_solution_parfile(
        path=attempt_dir / "best.par",
        seed_par_path=active_par_path,
        certification=certification,
        fit_parameter_names=fit_parameter_names,
    )
    if tim_path is not None:
        from timing_accelerator.reports.residual_plot import write_residual_plot

        try:
            write_residual_plot(
                par_path=par_path,
                tim_path=tim_path,
                output_dir=attempt_dir,
                clock_override_path=clock_override_path,
            )
        except Exception as exc:  # pragma: no cover - search results remain valid without a plot
            (attempt_dir / "best_residuals_error.txt").write_text(
                f"Residual plot generation failed: {exc}\n",
                encoding="utf-8",
            )
    metadata = {
        "attempt_id": f"stage_{int(stage_index):02d}_attempt_{attempt_number:04d}",
        "attempt_sequence": int(attempt_number),
        "stage": int(stage_index),
        "branch_id": branch.branch_id,
        "branch_source": branch.proposal_source,
        "wraps": [int(item) for item in branch.wraps],
        "weighted_rms_us": certification.weighted_rms_us,
        "reduced_chi2": certification.reduced_chi2,
        "chi2": certification.chi2,
        "log_likelihood": certification.log_likelihood,
        "backend": certification.backend,
        "active_par_path": str(Path(active_par_path)),
        "pulse_number_reference": pulse_reference,
        "pulse_number_reference_par_path": str(pulse_reference_path.relative_to(output_dir)),
        "fit_parameter_names": (
            [str(name).upper() for name in fit_parameter_names]
            if fit_parameter_names is not None
            else None
        ),
        "par_path": str(par_path.relative_to(output_dir)),
    }
    stage_best_json = history_dir / f"{artifact_stem}.json"
    global_best_json = history_dir / "global_best.json"
    stage_best_updated = _candidate_improves_best(metadata, stage_best_json)
    index_path = history_dir / "index.json"
    previous_index = _load_json(index_path) if index_path.exists() else {}
    previous_attempts = list(previous_index.get("attempts", []))

    if global_best_json.exists():
        global_best_updated = _candidate_improves_best(metadata, global_best_json)
        selected_global = metadata if global_best_updated else _load_json(global_best_json)
    else:
        global_candidates = previous_attempts + [metadata]
        selected_global = min(global_candidates, key=_artifact_score)
        global_best_updated = selected_global is metadata

    metadata.update(
        {
            "stage_best_updated": bool(stage_best_updated),
            "global_best_updated": bool(global_best_updated),
        }
    )
    _write_json_atomic(attempt_dir / "best.json", metadata)
    _verify_artifact_pair(par_path, attempt_dir / "best.json")

    if stage_best_updated:
        _copy_file_atomic(par_path, history_dir / f"{artifact_stem}.par")
        _write_json_atomic(stage_best_json, metadata)
        _verify_artifact_pair(history_dir / f"{artifact_stem}.par", stage_best_json)

    if not global_best_json.exists() or global_best_updated:
        selected_global_par = _metadata_par_path(output_dir, selected_global)
        _copy_file_atomic(selected_global_par, history_dir / "global_best.par")
        _write_json_atomic(global_best_json, selected_global)
        _verify_artifact_pair(history_dir / "global_best.par", global_best_json)

    attempts = previous_attempts + [metadata]
    attempts.sort(
        key=lambda item: (
            int(item.get("stage", -1)),
            int(item.get("attempt_sequence", 0)),
            str(item.get("attempt_id", "")),
        )
    )
    _write_json_atomic(
        index_path,
        {
            "attempts": attempts,
            "global_best": _load_json(global_best_json),
        },
    )
    return metadata


def _write_parameter_release_probe_artifacts(
    *,
    release_dir: Path,
    block: Sequence[str],
    seed_par_path: Path,
    fit_parameter_names: Sequence[str] | None,
    branch: ActiveTimingBranch,
    certification: BranchCertification,
    tim_path: Path | None = None,
    clock_override_path: Path | None = None,
) -> dict[str, Any]:
    normalized_block = tuple(str(name).upper() for name in block)
    attempt_name = "_".join(normalized_block).lower() or "none"
    attempt_dir = Path(release_dir) / "attempts" / attempt_name
    run_dir, attempt_number = _next_attempt_dir(attempt_dir / "runs")
    pulse_reference_path, pulse_reference = _write_pulse_number_reference_parfile(
        path=run_dir / "pulse_number_reference.par",
        seed_par_path=seed_par_path,
        branch=branch,
    )
    best_par = write_certified_solution_parfile(
        path=run_dir / "best.par",
        seed_par_path=seed_par_path,
        certification=certification,
        fit_parameter_names=fit_parameter_names,
    )
    if tim_path is not None:
        from timing_accelerator.reports.residual_plot import write_residual_plot

        try:
            write_residual_plot(
                par_path=best_par,
                tim_path=Path(tim_path),
                output_dir=run_dir,
                clock_override_path=clock_override_path,
            )
        except Exception as exc:  # pragma: no cover - model comparison remains valid without a plot
            (run_dir / "best_residuals_error.txt").write_text(
                f"Residual plot generation failed: {exc}\n",
                encoding="utf-8",
            )
    diagnostics = branch.diagnostics or {}
    metadata = {
        "attempt_id": f"parameter-release-{attempt_name}-attempt-{attempt_number:04d}",
        "attempt_sequence": int(attempt_number),
        "block": list(normalized_block),
        "branch_id": branch.branch_id,
        "branch_source": branch.proposal_source,
        "proposal_family": diagnostics.get("proposal_family"),
        "parent_branch_id": diagnostics.get("parameter_release_parent_branch_id"),
        "wraps": [int(item) for item in branch.wraps],
        "weighted_rms_us": certification.weighted_rms_us,
        "reduced_chi2": certification.reduced_chi2,
        "chi2": certification.chi2,
        "log_likelihood": certification.log_likelihood,
        "backend": certification.backend,
        "fit_parameter_names": (
            [str(name).upper() for name in fit_parameter_names]
            if fit_parameter_names is not None
            else None
        ),
        "seed_par_path": str(Path(seed_par_path)),
        "pulse_number_reference": pulse_reference,
        "pulse_number_reference_par_path": str(pulse_reference_path),
        "best_par_path": str(best_par),
    }
    block_best_path = attempt_dir / "best.json"
    global_best_path = Path(release_dir) / "global_best.json"
    block_best_updated = _candidate_improves_best(metadata, block_best_path)
    if global_best_path.exists():
        global_best_updated = _candidate_improves_best(metadata, global_best_path)
        selected_global = metadata if global_best_updated else _load_json(global_best_path)
    else:
        prior_block_bests = [
            _load_json(path)
            for path in sorted((Path(release_dir) / "attempts").glob("*/best.json"))
        ]
        selected_global = min(prior_block_bests + [metadata], key=_artifact_score)
        global_best_updated = selected_global is metadata
    metadata.update(
        {
            "block_best_updated": bool(block_best_updated),
            "global_best_updated": bool(global_best_updated),
        }
    )
    _write_json_atomic(run_dir / "best.json", metadata)
    _verify_artifact_pair(best_par, run_dir / "best.json")
    if block_best_updated:
        _copy_file_atomic(best_par, attempt_dir / "best.par")
        _write_json_atomic(block_best_path, metadata)
        _verify_artifact_pair(attempt_dir / "best.par", block_best_path)
    if not global_best_path.exists() or global_best_updated:
        selected_global_par = _metadata_par_path(Path(release_dir), selected_global)
        _copy_file_atomic(selected_global_par, Path(release_dir) / "global_best.par")
        _write_json_atomic(global_best_path, selected_global)
        _verify_artifact_pair(Path(release_dir) / "global_best.par", global_best_path)

    audit_path = attempt_dir / "audit.json"
    previous_audit = _load_json(audit_path) if audit_path.exists() else {}
    attempts = list(previous_audit.get("attempts", []))
    attempts.append(metadata)
    _write_json_atomic(
        audit_path,
        {
            **previous_audit,
            "attempt_id": f"parameter-release-{attempt_name}",
            "block": list(normalized_block),
            "status": "complete",
            "attempts": attempts,
            "best": _load_json(block_best_path),
        },
    )
    return metadata


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _branch_from_json(payload: dict[str, Any]) -> ActiveTimingBranch:
    return ActiveTimingBranch(
        branch_id=str(payload["branch_id"]),
        structure_id=str(payload["structure_id"]),
        binary_model=(
            str(payload["binary_model"]).upper()
            if payload.get("binary_model") is not None
            else None
        ),
        wraps=tuple(int(item) for item in payload.get("wraps", [])),
        theta_names=tuple(str(item).upper() for item in payload.get("theta_names", [])),
        theta_offset=np.asarray(payload.get("theta_offset", []), dtype=float),
        log_prior=float(payload.get("log_prior", 0.0)),
        proposal_source=str(payload.get("proposal_source", "staged-budgeted-bayesian-search")),
        diagnostics=payload.get("diagnostics"),
    )


def _certification_from_json(payload: dict[str, Any]) -> BranchCertification:
    return BranchCertification(
        branch_id=str(payload["branch_id"]),
        log_likelihood=float(payload.get("log_likelihood", float("-inf"))),
        fitted_theta={
            str(name).upper(): float(value)
            for name, value in (payload.get("fitted_theta") or {}).items()
        },
        covariance=np.asarray(payload.get("covariance", []), dtype=float),
        reduced_chi2=(
            float(payload["reduced_chi2"])
            if payload.get("reduced_chi2") is not None
            else None
        ),
        weighted_rms_us=(
            float(payload["weighted_rms_us"])
            if payload.get("weighted_rms_us") is not None
            else None
        ),
        backend=str(payload.get("backend", "artifact-resume")),
        log_evidence=(
            float(payload["log_evidence"])
            if payload.get("log_evidence") is not None
            else None
        ),
        chi2=(
            float(payload["chi2"])
            if payload.get("chi2") is not None
            else None
        ),
        notes=list(payload["notes"]) if payload.get("notes") is not None else None,
    )


def _load_resume_state_from_artifacts(
    *,
    output_dir: Path,
    start_cluster_count: int,
    end_cluster_count: int,
    stage_cluster_counts: Sequence[int] | None = None,
    enabled: bool,
) -> StagedResumeState | None:
    if not bool(enabled):
        return None
    output_dir = Path(output_dir)
    stage_records: list[dict[str, Any]] = []
    current_branches: list[ActiveTimingBranch] | None = None
    certifications: list[BranchCertification] | None = None
    next_cluster_count = int(start_cluster_count)
    schedule = tuple(
        range(int(start_cluster_count), int(end_cluster_count) + 1)
        if stage_cluster_counts is None
        else (
            int(item)
            for item in stage_cluster_counts
            if int(start_cluster_count) <= int(item) <= int(end_cluster_count)
        )
    )
    for schedule_index, cluster_count in enumerate(schedule):
        stage_path = output_dir / f"stage_{cluster_count:02d}.json"
        survivor_path = output_dir / f"stage_{cluster_count:02d}_survivors.json"
        certification_path = output_dir / f"stage_{cluster_count:02d}_certifications.json"
        if not (stage_path.exists() and survivor_path.exists() and certification_path.exists()):
            break
        record = _load_json(stage_path)
        survivors_payload = _load_json(survivor_path)
        certifications_payload = _load_json(certification_path)
        survivors = [_branch_from_json(item) for item in survivors_payload]
        loaded_certifications = [
            _certification_from_json(item)
            for item in certifications_payload
        ]
        if not survivors or not loaded_certifications:
            break
        certification_by_id = {
            certification.branch_id: certification
            for certification in loaded_certifications
        }
        survivors = [
            _with_certification_diagnostics(
                survivor,
                certification_by_id[survivor.branch_id],
            )
            if survivor.branch_id in certification_by_id
            else survivor
            for survivor in survivors
        ]
        stage_records.append(record)
        current_branches = survivors
        certifications = loaded_certifications
        next_cluster_count = (
            int(schedule[schedule_index + 1])
            if schedule_index + 1 < len(schedule)
            else int(end_cluster_count) + 1
        )
    if current_branches is None or certifications is None:
        return None
    return StagedResumeState(
        current_branches=current_branches,
        certifications=certifications,
        stage_records=stage_records,
        next_cluster_count=next_cluster_count,
        resumed_stage_count=len(stage_records),
    )


def _write_stage_progress(
    stage_dir: Path,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    record = {"event": str(event), **(payload or {})}
    with (stage_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_json_ready(record), sort_keys=True) + "\n")


def _tim_source_name(raw_line: str) -> str | None:
    line = raw_line.strip()
    if not line or line.startswith(("#", "C ")):
        return None
    upper = line.upper()
    if upper.startswith(("FORMAT", "MODE")):
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    return parts[0]


def write_prefix_timing_inputs(
    *,
    tim_path: Path,
    cluster_manifest_path: Path,
    prefix_cluster_count: int,
    output_dir: Path,
) -> PrefixTimingInputs:
    tim_path = Path(tim_path)
    cluster_manifest_path = Path(cluster_manifest_path)
    output_dir = Path(output_dir)
    manifest = load_cluster_manifest(cluster_manifest_path)
    cluster_count = min(max(int(prefix_cluster_count), 1), len(manifest.clusters))
    kept_clusters = list(manifest.clusters[:cluster_count])
    kept_cluster_ids = {cluster.cluster_id for cluster in kept_clusters}
    kept_sources = {
        source_name
        for cluster in kept_clusters
        for source_name in cluster.source_filenames
    }
    kept_gaps = [
        gap
        for gap in manifest.gaps
        if gap.left_cluster in kept_cluster_ids and gap.right_cluster in kept_cluster_ids
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix_tim_path = output_dir / "prefix.tim"
    prefix_manifest_path = output_dir / "prefix_clusters.yaml"
    kept_tim_lines: list[str] = []
    for raw_line in tim_path.read_text(encoding="utf-8").splitlines():
        source_name = _tim_source_name(raw_line)
        if source_name is None or source_name in kept_sources:
            kept_tim_lines.append(raw_line)
    prefix_tim_path.write_text("\n".join(kept_tim_lines) + "\n", encoding="utf-8")
    write_cluster_manifest(
        type(manifest)(clusters=kept_clusters, gaps=kept_gaps),
        prefix_manifest_path,
    )
    return PrefixTimingInputs(
        tim_path=prefix_tim_path,
        cluster_manifest_path=prefix_manifest_path,
        cluster_count=cluster_count,
        gap_count=max(cluster_count - 1, 0),
    )


def _atomic_parent_stage_cluster_counts(
    manifest: ClusterManifest,
    *,
    start_cluster_count: int,
    end_cluster_count: int,
) -> tuple[int, ...]:
    cluster_count = len(manifest.clusters)
    if cluster_count == 0:
        return ()

    boundaries = tuple(
        index + 1
        for index, cluster in enumerate(manifest.clusters)
        if index + 1 == cluster_count
        or (
            str(cluster.parent_cluster_id or cluster.cluster_id)
            != str(
                manifest.clusters[index + 1].parent_cluster_id
                or manifest.clusters[index + 1].cluster_id
            )
        )
    )
    requested_start = min(max(int(start_cluster_count), 1), cluster_count)
    requested_end = min(
        max(int(end_cluster_count), requested_start),
        cluster_count,
    )
    aligned_start = next(item for item in boundaries if item >= requested_start)
    aligned_end = next(item for item in boundaries if item >= requested_end)
    return tuple(item for item in boundaries if aligned_start <= item <= aligned_end)


def _parameter_value(model, name: str) -> float:
    return float(model[str(name).upper()].value)


def _base_parameter_values(par_path: Path, theta_names: Sequence[str]) -> dict[str, float]:
    model = _load_pint_model(par_path)
    return {str(name).upper(): _parameter_value(model, str(name).upper()) for name in theta_names}


def _physical_prior_widths(par_path: Path, theta_names: Sequence[str]) -> dict[str, float]:
    names = [str(name).upper() for name in theta_names]
    specs = build_parameter_prior_specs(par_path, allowed_params=set(names))
    width_by_name = {
        str(spec.name).upper(): max(
            abs(float(spec.upper) - float(spec.center)),
            abs(float(spec.center) - float(spec.lower)),
        )
        for spec in specs
    }
    return {name: float(width_by_name.get(name, 0.0)) for name in names}


def _fit_prior_parameter_scales(
    par_path: Path,
    theta_names: Sequence[str],
    *,
    formal_uncertainty_scale: float = 1000.0,
    physical_width_fraction: float = 0.25,
) -> np.ndarray:
    model = _load_pint_model(par_path)
    names = [str(name).upper() for name in theta_names]
    physical_widths = _physical_prior_widths(par_path, names)
    scales: list[float] = []
    for name in names:
        raw_uncertainty = model[name].uncertainty_value
        formal_uncertainty = abs(float(raw_uncertainty or 0.0))
        if not np.isfinite(formal_uncertainty):
            formal_uncertainty = 0.0
        formal_scale = formal_uncertainty * max(float(formal_uncertainty_scale), 0.0)
        physical_scale = float(physical_widths.get(name, 0.0)) * max(
            float(physical_width_fraction),
            0.0,
        )
        scales.append(max(formal_scale, physical_scale, 1.0e-30))
    return np.asarray(scales, dtype=float)


def _initial_branch(
    *,
    structure_id: str,
    binary_model: str | None,
    theta_names: Sequence[str],
) -> ActiveTimingBranch:
    return ActiveTimingBranch(
        branch_id="stage_0000",
        structure_id=str(structure_id),
        binary_model=binary_model.upper() if binary_model else None,
        wraps=(),
        theta_names=tuple(str(name).upper() for name in theta_names),
        theta_offset=np.zeros(len(theta_names), dtype=float),
        log_prior=0.0,
        proposal_source="staged-budgeted-bayesian-search",
        diagnostics={"proposal_family": "seed", "stage": 1},
    )


def _active_prior_scales(primary: float, extras: Sequence[float]) -> tuple[float, ...]:
    values: list[float] = [float(primary)]
    values.extend(float(item) for item in extras)
    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        if not np.isfinite(value) or value <= 0.0:
            continue
        key = float(f"{value:.12g}")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(value))
    return tuple(deduped or (1.0,))


def _phase_slope_inputs(
    evidence: LocalPhaseSlopeEvidence | None,
    config: StagedBudgetedSearchConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not bool(config.phase_slope_enabled) or evidence is None:
        return None, None, None
    sigma_scale = max(float(config.phase_slope_sigma_scale), 1.0e-12)
    sigma_floor = max(float(config.phase_slope_sigma_floor), 0.0)
    sigmas = np.maximum(
        np.asarray(evidence.phase_change_sigma, dtype=float) * sigma_scale,
        sigma_floor,
    )
    return (
        np.asarray(evidence.phase_change_jacobian, dtype=float),
        np.asarray(evidence.reference_phase_change, dtype=float),
        np.asarray(sigmas, dtype=float),
    )


def _incremental_phase_slope_inputs(
    *,
    evidence: LocalPhaseSlopeEvidence | None,
    config: StagedBudgetedSearchConfig,
    source_session_cluster_ids: Sequence[str],
    inherited_gap_count: int,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    jacobian, reference, sigmas = _phase_slope_inputs(evidence, config)
    if jacobian is None or reference is None or sigmas is None or evidence is None:
        return None, None, None

    first_new_session = min(
        max(int(inherited_gap_count) + 1, 0),
        len(source_session_cluster_ids),
    )
    new_session_ids = {
        str(cluster_id)
        for cluster_id in source_session_cluster_ids[first_new_session:]
    }
    evidence_ids = tuple(str(cluster_id) for cluster_id in evidence.cluster_ids)
    row_mask = np.asarray(
        [cluster_id in new_session_ids for cluster_id in evidence_ids],
        dtype=bool,
    )
    if not np.any(row_mask):
        return None, None, None
    return jacobian[row_mask], reference[row_mask], sigmas[row_mask]


def _information_rank_diagnostics(
    *,
    gap_jacobian: np.ndarray,
    local_phase_slope_jacobian: np.ndarray | None,
    local_phase_slope_sigmas: np.ndarray | None,
    parameter_scales: np.ndarray,
    phase_sigma: float,
    rank_rtol: float,
) -> dict[str, Any]:
    gaps = np.asarray(gap_jacobian, dtype=float)
    scales = np.asarray(parameter_scales, dtype=float).reshape(-1)
    if gaps.ndim != 2 or gaps.shape[1] != scales.size:
        raise ValueError("gap_jacobian must have one column per fitted parameter.")

    weighted_blocks = [gaps / max(abs(float(phase_sigma)), 1.0e-12)]
    slope_count = 0
    if local_phase_slope_jacobian is not None:
        slopes = np.asarray(local_phase_slope_jacobian, dtype=float)
        if slopes.ndim != 2 or slopes.shape[1] != scales.size:
            raise ValueError(
                "local_phase_slope_jacobian must have one column per fitted parameter."
            )
        if local_phase_slope_sigmas is None:
            raise ValueError("local phase-slope sigmas are required with slope geometry.")
        slope_sigmas = np.asarray(local_phase_slope_sigmas, dtype=float).reshape(-1)
        if slope_sigmas.size != slopes.shape[0]:
            raise ValueError("local phase-slope sigmas must match the slope rows.")
        valid_slopes = (
            np.all(np.isfinite(slopes), axis=1)
            & np.isfinite(slope_sigmas)
            & (slope_sigmas > 0.0)
        )
        slopes = slopes[valid_slopes]
        slope_sigmas = slope_sigmas[valid_slopes]
        slope_count = int(slopes.shape[0])
        if slope_count:
            weighted_blocks.append(slopes / slope_sigmas[:, None])

    weighted_design = np.vstack(weighted_blocks) * scales[None, :]
    valid_rows = np.all(np.isfinite(weighted_design), axis=1)
    weighted_design = weighted_design[valid_rows]
    column_norms = np.linalg.norm(weighted_design, axis=0)
    normalized_design = np.divide(
        weighted_design,
        column_norms[None, :],
        out=np.zeros_like(weighted_design),
        where=column_norms[None, :] > 0.0,
    )
    singular_values = np.linalg.svd(normalized_design, compute_uv=False)
    relative_tolerance = max(abs(float(rank_rtol)), np.finfo(float).eps)
    threshold = (
        float(singular_values[0]) * relative_tolerance
        if singular_values.size
        else 0.0
    )
    information_rank = int(np.count_nonzero(singular_values > threshold))
    target_rank = int(scales.size)
    return {
        "gap_count": int(gaps.shape[0]),
        "slope_evidence_count": slope_count,
        "parameter_count": target_rank,
        "information_rank": information_rank,
        "target_rank": target_rank,
        "full_rank": bool(information_rank >= target_rank),
        "rank_rtol": relative_tolerance,
        "singular_value_threshold": threshold,
        "singular_values": [float(value) for value in singular_values],
        "parameter_column_norms": [float(value) for value in column_norms],
    }


def _choose_information_rank_start_cluster_count(
    *,
    requested_start_cluster_count: int,
    end_cluster_count: int,
    parameter_scales: np.ndarray,
    phase_sigma: float,
    phase_slope_sigma_scale: float,
    phase_slope_sigma_floor: float,
    rank_rtol: float,
    geometry_for_cluster_count: Callable[[int], Any],
) -> tuple[int, dict[str, Any]]:
    requested = min(
        max(int(requested_start_cluster_count), 1),
        max(int(end_cluster_count), 1),
    )
    prefix_diagnostics: list[dict[str, Any]] = []
    sigma_scale = max(float(phase_slope_sigma_scale), 1.0e-12)
    sigma_floor = max(float(phase_slope_sigma_floor), 0.0)
    for cluster_count in range(requested, int(end_cluster_count) + 1):
        geometry = geometry_for_cluster_count(cluster_count)
        evidence = getattr(geometry, "local_phase_slope_evidence", None)
        slope_jacobian = None
        slope_sigmas = None
        if evidence is not None:
            slope_jacobian = np.asarray(evidence.phase_change_jacobian, dtype=float)
            slope_sigmas = np.maximum(
                np.asarray(evidence.phase_change_sigma, dtype=float) * sigma_scale,
                sigma_floor,
            )
        diagnostics = _information_rank_diagnostics(
            gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
            local_phase_slope_jacobian=slope_jacobian,
            local_phase_slope_sigmas=slope_sigmas,
            parameter_scales=np.asarray(parameter_scales, dtype=float),
            phase_sigma=phase_sigma,
            rank_rtol=rank_rtol,
        )
        prefix_diagnostics.append(
            {"cluster_count": int(cluster_count), **diagnostics}
        )
        if diagnostics["full_rank"]:
            return int(cluster_count), {
                "status": "full-rank-prefix-found",
                "requested_start_cluster_count": requested,
                "selected_start_cluster_count": int(cluster_count),
                "prefixes": prefix_diagnostics,
            }

    return requested, {
        "status": "no-full-rank-prefix",
        "requested_start_cluster_count": requested,
        "selected_start_cluster_count": requested,
        "prefixes": prefix_diagnostics,
    }


def _future_local_phase_slope_evidence(
    current: LocalPhaseSlopeEvidence | None,
    lookahead: LocalPhaseSlopeEvidence | None,
) -> LocalPhaseSlopeEvidence | None:
    if lookahead is None:
        return None
    current_cluster_ids = set(current.cluster_ids if current is not None else ())
    indices = np.asarray(
        [
            index
            for index, cluster_id in enumerate(lookahead.cluster_ids)
            if cluster_id not in current_cluster_ids
        ],
        dtype=int,
    )
    parameter_count = int(np.asarray(lookahead.phase_change_jacobian).shape[1])
    return LocalPhaseSlopeEvidence(
        cluster_ids=tuple(lookahead.cluster_ids[index] for index in indices),
        cluster_mjd=np.asarray(lookahead.cluster_mjd, dtype=float)[indices],
        session_span_seconds=np.asarray(lookahead.session_span_seconds, dtype=float)[indices],
        reference_phase_change=np.asarray(lookahead.reference_phase_change, dtype=float)[indices],
        phase_change_jacobian=(
            np.asarray(lookahead.phase_change_jacobian, dtype=float)[indices]
            if indices.size
            else np.zeros((0, parameter_count), dtype=float)
        ),
        phase_change_sigma=np.asarray(lookahead.phase_change_sigma, dtype=float)[indices],
    )


def _lattice_portfolio_candidates(
    *,
    seed_branch: ActiveTimingBranch,
    stage_index: int,
    family: str,
    branch_prefix: str,
    proposal_source: str,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    top_k: int,
    beam_size: int,
    phase_sigma: float,
    prior_scales: Sequence[float],
    radius_sigma: float,
    min_radius: int,
    max_enum_per_gap: int,
    wrap_prior_scale: float,
    gap_order: Sequence[int] | None = None,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
    slope_sigma_scale: float = 1.0,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    target = max(int(top_k), 0)
    scale_matrix = np.diag(np.asarray(parameter_scales, dtype=float))
    portfolio_candidates: list[list[ActiveTimingBranch]] = []
    portfolio_records: list[dict[str, Any]] = []
    per_temperature_top_k = max(target, 1)
    for portfolio_index, prior_scale in enumerate(prior_scales):
        lattice_branches, lattice_diagnostics = generate_bayesian_alias_lattice_branches(
            structure_id=seed_branch.structure_id,
            binary_model=seed_branch.binary_model,
            theta_names=list(seed_branch.theta_names),
            gap_jacobian=np.asarray(gap_jacobian, dtype=float),
            reference_phase_residuals=np.asarray(reference_phase_residuals, dtype=float),
            parameter_scales=np.asarray(parameter_scales, dtype=float),
            local_phase_slope_jacobian=local_phase_slope_jacobian,
            reference_local_phase_slopes=reference_local_phase_slopes,
            local_phase_slope_sigmas=local_phase_slope_sigmas,
            config=BayesianAliasLatticeConfig(
                beam_size=max(int(beam_size), 1),
                phase_sigma=float(phase_sigma),
                prior_scale=float(prior_scale),
                radius_sigma=float(radius_sigma),
                min_radius=int(min_radius),
                max_enum_per_gap=int(max_enum_per_gap),
                wrap_prior_scale=float(wrap_prior_scale),
                top_k=per_temperature_top_k,
                gap_order=(
                    None
                    if gap_order is None
                    else tuple(int(index) for index in gap_order)
                ),
            ),
        )
        relabelled_portfolio: list[ActiveTimingBranch] = []
        portfolio_records.append(
            {
                "portfolio_index": int(portfolio_index),
                "prior_scale": float(prior_scale),
                "candidate_count": lattice_diagnostics.get("candidate_count", len(lattice_branches)),
                "kept_count": len(lattice_branches),
                "gap_order": lattice_diagnostics.get("gap_order", []),
                "evidence_mode": lattice_diagnostics.get("evidence_mode", "phase-only"),
                "local_slope_evidence_count": int(
                    lattice_diagnostics.get("local_slope_evidence_count", 0)
                ),
                "local_slope_log_likelihood": float(
                    lattice_diagnostics.get("local_slope_log_likelihood", 0.0)
                ),
                "slope_sigma_scale": float(slope_sigma_scale),
            }
        )
        for branch in lattice_branches:
            diagnostics = dict(branch.diagnostics or {})
            unit_covariance = np.asarray(
                diagnostics["posterior_covariance_unit_theta"],
                dtype=float,
            )
            diagnostics["proposal_covariance"] = (
                scale_matrix @ unit_covariance @ scale_matrix
            )
            diagnostics["proposal_family"] = str(family)
            diagnostics["stage"] = int(stage_index)
            diagnostics["portfolio_index"] = int(portfolio_index)
            diagnostics["portfolio_prior_scale"] = float(prior_scale)
            diagnostics["portfolio_slope_sigma_scale"] = float(slope_sigma_scale)
            diagnostics["portfolio_source_branch_id"] = branch.branch_id
            relabelled_portfolio.append(
                replace(
                    branch,
                    branch_id=f"{branch_prefix}_p{portfolio_index:04d}_{len(relabelled_portfolio):04d}",
                    proposal_source="staged-budgeted-bayesian-search",
                    diagnostics=diagnostics,
                )
            )
        portfolio_candidates.append(relabelled_portfolio)

    candidates: list[ActiveTimingBranch] = []
    seen_wraps: set[tuple[int, ...]] = set()

    def add_candidate(branch: ActiveTimingBranch) -> bool:
        if len(candidates) >= target:
            return False
        wraps = tuple(int(item) for item in branch.wraps)
        if wraps in seen_wraps:
            return False
        seen_wraps.add(wraps)
        candidates.append(
            replace(
                branch,
                branch_id=f"{branch_prefix}_{len(candidates):04d}",
            )
        )
        return True

    if target > 0 and portfolio_candidates:
        active_portfolio_count = max(
            sum(1 for items in portfolio_candidates if items),
            1,
        )
        portfolio_floor = max(1, target // active_portfolio_count)
        for items in portfolio_candidates:
            selected_from_portfolio = 0
            for branch in items:
                if selected_from_portfolio >= portfolio_floor:
                    break
                if add_candidate(branch):
                    selected_from_portfolio += 1
        scored_candidates = sorted(
            [branch for items in portfolio_candidates for branch in items],
            key=lambda branch: (
                -float(branch.log_prior),
                str(branch.branch_id),
            ),
        )
        for branch in scored_candidates:
            if len(candidates) >= target:
                break
            add_candidate(branch)

    for record in portfolio_records:
        portfolio_index = int(record["portfolio_index"])
        record["selected_count"] = sum(
            1
            for branch in candidates
            if int((branch.diagnostics or {}).get("portfolio_index", -1)) == portfolio_index
        )
    return candidates, {
        "proposal_source": str(proposal_source),
        "portfolio_count": len(tuple(prior_scales)),
        "portfolios": portfolio_records,
        "candidate_count": int(sum(int(item.get("candidate_count", 0)) for item in portfolio_records)),
        "kept_count": len(candidates),
    }


def _initial_stage_candidates(
    *,
    seed_branch: ActiveTimingBranch,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    gap_count = int(np.asarray(gap_jacobian, dtype=float).shape[0])
    if gap_count <= 0:
        branch = replace(
            seed_branch,
            wraps=(),
            diagnostics={**(seed_branch.diagnostics or {}), "proposal_family": "zero-wrap"},
        )
        return [branch], {"proposal_source": "zero-wrap", "candidate_count": 1, "kept_count": 1}
    if not bool(config.initial_lattice_enabled):
        branch = replace(
            seed_branch,
            wraps=tuple(0 for _ in range(gap_count)),
            diagnostics={**(seed_branch.diagnostics or {}), "proposal_family": "zero-wrap"},
        )
        return [branch], {"proposal_source": "zero-wrap", "candidate_count": 1, "kept_count": 1}

    candidates, lattice_diagnostics = _lattice_portfolio_candidates(
        seed_branch=seed_branch,
        stage_index=gap_count + 1,
        family="initial-lattice",
        branch_prefix=f"stage_{gap_count + 1:02d}_init",
        proposal_source="initial-bayesian-alias-lattice",
        gap_jacobian=np.asarray(gap_jacobian, dtype=float),
        reference_phase_residuals=np.asarray(reference_phase_residuals, dtype=float),
        parameter_scales=np.asarray(parameter_scales, dtype=float),
        top_k=max(int(config.initial_lattice_top_k), 1),
        beam_size=max(int(config.initial_lattice_beam_size), 1),
        phase_sigma=(
            float(config.initial_lattice_phase_sigma)
            if config.initial_lattice_phase_sigma is not None
            else float(config.phase_sigma)
        ),
        prior_scales=_active_prior_scales(
            float(config.initial_lattice_prior_scale),
            config.initial_lattice_prior_scales,
        ),
        radius_sigma=float(config.initial_lattice_radius_sigma),
        min_radius=int(config.initial_lattice_min_radius),
        max_enum_per_gap=int(config.initial_lattice_max_enum_per_gap),
        wrap_prior_scale=float(config.wrap_prior_scale),
        local_phase_slope_jacobian=local_phase_slope_jacobian,
        reference_local_phase_slopes=reference_local_phase_slopes,
        local_phase_slope_sigmas=local_phase_slope_sigmas,
    )
    if not candidates:
        branch = replace(
            seed_branch,
            wraps=tuple(0 for _ in range(gap_count)),
            diagnostics={**(seed_branch.diagnostics or {}), "proposal_family": "zero-wrap"},
        )
        return [branch], {
            "proposal_source": "zero-wrap",
            "candidate_count": 1,
            "kept_count": 1,
            "lattice": lattice_diagnostics,
            "reason": "empty-lattice",
        }
    return candidates, {
        "proposal_source": "initial-bayesian-alias-lattice",
        "candidate_count": lattice_diagnostics.get("candidate_count", len(candidates)),
        "kept_count": len(candidates),
        "lattice": lattice_diagnostics,
    }


def _stage_lattice_refresh_candidates(
    *,
    seed_branch: ActiveTimingBranch,
    stage_index: int,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
    branch_suffix: str = "",
    slope_sigma_scale: float = 1.0,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    gap_count = int(np.asarray(gap_jacobian, dtype=float).shape[0])
    if gap_count <= 0 or not bool(config.stage_lattice_refresh_enabled):
        return [], {
            "proposal_source": "stage-bayesian-alias-lattice-refresh",
            "enabled": bool(config.stage_lattice_refresh_enabled),
            "candidate_count": 0,
            "kept_count": 0,
        }
    candidates, lattice_diagnostics = _lattice_portfolio_candidates(
        seed_branch=seed_branch,
        stage_index=stage_index,
        family="stage-lattice-refresh",
        branch_prefix=f"stage_{stage_index:02d}_refresh{branch_suffix}",
        proposal_source="stage-bayesian-alias-lattice-refresh",
        gap_jacobian=np.asarray(gap_jacobian, dtype=float),
        reference_phase_residuals=np.asarray(reference_phase_residuals, dtype=float),
        parameter_scales=np.asarray(parameter_scales, dtype=float),
        top_k=max(int(config.stage_lattice_refresh_top_k), 1),
        beam_size=max(int(config.stage_lattice_refresh_beam_size), 1),
        phase_sigma=(
            float(config.stage_lattice_refresh_phase_sigma)
            if config.stage_lattice_refresh_phase_sigma is not None
            else float(config.phase_sigma)
        ),
        prior_scales=_active_prior_scales(
            float(config.stage_lattice_refresh_prior_scale),
            config.stage_lattice_refresh_prior_scales,
        ),
        radius_sigma=float(config.stage_lattice_refresh_radius_sigma),
        min_radius=int(config.stage_lattice_refresh_min_radius),
        max_enum_per_gap=int(config.stage_lattice_refresh_max_enum_per_gap),
        wrap_prior_scale=float(config.wrap_prior_scale),
        local_phase_slope_jacobian=local_phase_slope_jacobian,
        reference_local_phase_slopes=reference_local_phase_slopes,
        local_phase_slope_sigmas=local_phase_slope_sigmas,
        slope_sigma_scale=float(slope_sigma_scale),
    )
    return candidates, {
        "proposal_source": "stage-bayesian-alias-lattice-refresh",
        "enabled": True,
        "candidate_count": lattice_diagnostics.get("candidate_count", len(candidates)),
        "kept_count": len(candidates),
        "lattice": lattice_diagnostics,
    }


def _conditioned_lattice_prior_scales(config: StagedBudgetedSearchConfig) -> tuple[float, ...]:
    primary = (
        float(config.stage_lattice_refresh_prior_scale)
        if config.conditioned_lattice_prior_scale is None
        else float(config.conditioned_lattice_prior_scale)
    )
    extras: Sequence[float]
    if config.conditioned_lattice_prior_scale is None and not config.conditioned_lattice_prior_scales:
        extras = config.stage_lattice_refresh_prior_scales
    else:
        extras = config.conditioned_lattice_prior_scales
    return _active_prior_scales(primary, extras)


def _posterior_lattice_prior_scales(config: StagedBudgetedSearchConfig) -> tuple[float, ...]:
    if config.posterior_lattice_prior_scale is None and not config.posterior_lattice_prior_scales:
        return _conditioned_lattice_prior_scales(config)
    primary = (
        float(config.stage_lattice_refresh_prior_scale)
        if config.posterior_lattice_prior_scale is None
        else float(config.posterior_lattice_prior_scale)
    )
    return _active_prior_scales(primary, config.posterior_lattice_prior_scales)


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    peak = max(float(value) for value in values)
    if not np.isfinite(peak):
        return peak
    return float(peak + math.log(sum(math.exp(float(value) - peak) for value in values)))


@dataclass(frozen=True)
class _IntegerBridgeConditioning:
    chosen_wrap: int
    predicted_wrap: float
    innovation: float
    predictive_variance: float
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    log_likelihood: float


def _predictive_covariance_from_diagnostics(
    diagnostics: dict[str, Any],
    parameter_count: int,
) -> tuple[np.ndarray | None, str | None]:
    expected_shape = (int(parameter_count), int(parameter_count))
    for key in (
        "bridge_conditioned_covariance",
        "proposal_covariance",
        "certified_covariance",
    ):
        covariance = np.asarray(diagnostics.get(key, []), dtype=float)
        if covariance.shape == expected_shape and np.all(np.isfinite(covariance)):
            return covariance, key
    return None, None


def _condition_integer_bridges(
    *,
    mean: np.ndarray,
    covariance: np.ndarray,
    jacobian_row: np.ndarray,
    reference_phase_residual: float,
    chosen_wraps: Sequence[int],
    phase_sigma: float,
) -> list[_IntegerBridgeConditioning]:
    active_mean = np.asarray(mean, dtype=float)
    row = np.asarray(jacobian_row, dtype=float)
    active_covariance = np.asarray(covariance, dtype=float)
    if active_mean.ndim != 1 or row.shape != active_mean.shape:
        raise ValueError("Bridge mean and Jacobian row must have matching one-dimensional shapes")
    if active_covariance.shape != (active_mean.size, active_mean.size):
        raise ValueError("Bridge covariance shape must match the continuous parameter dimension")

    active_covariance = 0.5 * (active_covariance + active_covariance.T)
    noise_variance = max(float(phase_sigma), 1.0e-12) ** 2
    row_covariance = active_covariance @ row
    predictive_variance = max(
        noise_variance + float(row @ row_covariance),
        1.0e-18,
    )
    predicted_wrap = float(reference_phase_residual) + float(row @ active_mean)
    kalman_gain = row_covariance / predictive_variance
    identity = np.eye(active_mean.size, dtype=float)
    update_operator = identity - np.outer(kalman_gain, row)
    posterior_covariance = (
        update_operator @ active_covariance @ update_operator.T
        + noise_variance * np.outer(kalman_gain, kalman_gain)
    )
    posterior_covariance = 0.5 * (posterior_covariance + posterior_covariance.T)
    log_normalizer = -0.5 * math.log(2.0 * math.pi * predictive_variance)

    conditioned: list[_IntegerBridgeConditioning] = []
    for chosen_wrap in chosen_wraps:
        innovation = float(chosen_wrap) - predicted_wrap
        conditioned.append(
            _IntegerBridgeConditioning(
                chosen_wrap=int(chosen_wrap),
                predicted_wrap=float(predicted_wrap),
                innovation=float(innovation),
                predictive_variance=float(predictive_variance),
                posterior_mean=active_mean + kalman_gain * innovation,
                posterior_covariance=posterior_covariance.copy(),
                log_likelihood=float(
                    log_normalizer
                    - 0.5 * innovation * innovation / predictive_variance
                ),
            )
        )
    return conditioned


def _sequential_bridge_lookahead(
    *,
    offset: np.ndarray,
    covariance: np.ndarray,
    future_jacobian: np.ndarray,
    future_reference: np.ndarray,
    phase_sigma: float,
    beam_size: int = 256,
    radius_sigma: float = 5.0,
    min_wrap_radius: int = 2,
    max_wrap_radius: int = 12,
) -> dict[str, Any]:
    active_beam: list[tuple[float, tuple[int, ...], np.ndarray, np.ndarray]] = [
        (
            0.0,
            (),
            np.asarray(offset, dtype=float).copy(),
            0.5 * (np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T),
        )
    ]
    largest_beam = 1
    for row, reference in zip(future_jacobian, future_reference, strict=True):
        expanded: list[tuple[float, tuple[int, ...], np.ndarray, np.ndarray]] = []
        for log_weight, wraps, mean, active_covariance in active_beam:
            predicted_wrap = float(reference) + float(row @ mean)
            predictive_variance = max(
                max(float(phase_sigma), 1.0e-12) ** 2
                + float(row @ active_covariance @ row),
                1.0e-18,
            )
            predicted_sigma = math.sqrt(float(predictive_variance))
            wrap_radius = max(
                int(min_wrap_radius),
                int(math.ceil(float(radius_sigma) * predicted_sigma)),
            )
            wrap_radius = min(wrap_radius, max(int(max_wrap_radius), int(min_wrap_radius)))
            nearest_wrap = int(np.rint(predicted_wrap))
            conditioned = _condition_integer_bridges(
                mean=mean,
                covariance=active_covariance,
                jacobian_row=row,
                reference_phase_residual=float(reference),
                chosen_wraps=range(
                    nearest_wrap - wrap_radius,
                    nearest_wrap + wrap_radius + 1,
                ),
                phase_sigma=phase_sigma,
            )
            for bridge in conditioned:
                expanded.append(
                    (
                        float(log_weight + bridge.log_likelihood),
                        wraps + (bridge.chosen_wrap,),
                        bridge.posterior_mean,
                        bridge.posterior_covariance,
                    )
                )
        expanded.sort(key=lambda item: (-item[0], item[1]))
        active_beam = expanded[: max(int(beam_size), 1)]
        largest_beam = max(largest_beam, len(active_beam))

    log_evidence = _logsumexp([item[0] for item in active_beam])
    map_log_weight, map_wraps, _, _ = active_beam[0]
    path_probabilities = [
        math.exp(max(float(item[0]) - log_evidence, -745.0))
        for item in active_beam
    ]
    path_entropy = -sum(
        probability * math.log(max(probability, 1.0e-300))
        for probability in path_probabilities
    )
    return {
        "log_evidence": float(log_evidence),
        "map_log_weight": float(map_log_weight),
        "map_wraps": list(map_wraps),
        "path_count": len(active_beam),
        "largest_beam": int(largest_beam),
        "path_entropy": float(path_entropy),
    }


def _with_parent_predictive_lookahead(
    parent_branches: Sequence[ActiveTimingBranch],
    *,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    phase_sigma: float,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
) -> list[ActiveTimingBranch]:
    jacobian = np.asarray(gap_jacobian, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float)
    noise_variance = max(float(phase_sigma), 1.0e-12) ** 2
    scored: list[ActiveTimingBranch] = []
    for parent in parent_branches:
        diagnostics = dict(parent.diagnostics or {})
        offset = np.asarray(parent.theta_offset, dtype=float)
        future_start = min(len(parent.wraps), jacobian.shape[0])
        if offset.shape != (jacobian.shape[1],):
            diagnostics.update(
                {
                    "lookahead_log_likelihood": -1.0e12,
                    "lookahead_future_gap_count": int(jacobian.shape[0] - future_start),
                    "lookahead_status": "invalid-parent-offset",
                }
            )
            scored.append(replace(parent, diagnostics=diagnostics))
            continue

        covariance, covariance_source = _predictive_covariance_from_diagnostics(
            diagnostics,
            jacobian.shape[1],
        )
        conditioning_covariance = (
            covariance
            if covariance is not None
            else np.zeros((offset.size, offset.size), dtype=float)
        )
        (
            conditioned_offset,
            conditioned_covariance,
            slope_log_likelihood,
            slope_innovation,
        ) = _condition_on_local_phase_slopes(
            mean=offset,
            covariance=conditioning_covariance,
            parameter_scales=np.ones(offset.size, dtype=float),
            local_phase_slope_jacobian=local_phase_slope_jacobian,
            reference_local_phase_slopes=reference_local_phase_slopes,
            local_phase_slope_sigmas=local_phase_slope_sigmas,
        )
        future_jacobian = jacobian[future_start:]
        future_reference = reference[future_start:]
        predicted = future_reference + future_jacobian @ conditioned_offset
        nearest_wraps = np.rint(predicted)
        residuals = predicted - nearest_wraps
        evidence_mode = "phase+slope" if slope_innovation.size else "phase-only"
        slope_diagnostics = {
            "lookahead_evidence_mode": evidence_mode,
            "lookahead_slope_evidence_count": int(slope_innovation.size),
            "lookahead_slope_log_likelihood": float(slope_log_likelihood),
            "lookahead_slope_innovation": np.asarray(slope_innovation, dtype=float),
            "lookahead_slope_conditioned_offset": np.asarray(
                conditioned_offset,
                dtype=float,
            ),
        }
        if covariance is not None:
            bridge = _sequential_bridge_lookahead(
                offset=conditioned_offset,
                covariance=conditioned_covariance,
                future_jacobian=future_jacobian,
                future_reference=future_reference,
                phase_sigma=phase_sigma,
            )
            diagnostics.update(
                {
                    "lookahead_log_likelihood": float(
                        slope_log_likelihood + float(bridge["log_evidence"])
                    ),
                    "lookahead_future_gap_count": int(future_jacobian.shape[0]),
                    "lookahead_future_gap_start": int(future_start),
                    "lookahead_predicted_wraps": predicted,
                    "lookahead_nearest_wraps": nearest_wraps.astype(int),
                    "lookahead_wrap_residuals": residuals,
                    "lookahead_score_mode": "sequential-bridge",
                    "lookahead_map_wraps": bridge["map_wraps"],
                    "lookahead_map_log_weight": bridge["map_log_weight"],
                    "lookahead_path_count": bridge["path_count"],
                    "lookahead_largest_beam": bridge["largest_beam"],
                    "lookahead_path_entropy": bridge["path_entropy"],
                    "lookahead_covariance_source": covariance_source,
                    "lookahead_status": "ok",
                    **slope_diagnostics,
                }
            )
            scored.append(replace(parent, diagnostics=diagnostics))
            continue

        variances: list[float] = []
        log_likelihood = 0.0
        for row, residual in zip(future_jacobian, residuals, strict=True):
            predictive_variance = noise_variance
            predictive_variance = max(float(predictive_variance), 1.0e-18)
            variances.append(predictive_variance)
            log_likelihood += (
                -0.5 * math.log(2.0 * math.pi * predictive_variance)
                -0.5 * float(residual) ** 2 / predictive_variance
            )
        diagnostics.update(
            {
                "lookahead_log_likelihood": float(slope_log_likelihood + log_likelihood),
                "lookahead_future_gap_count": int(future_jacobian.shape[0]),
                "lookahead_future_gap_start": int(future_start),
                "lookahead_predicted_wraps": predicted,
                "lookahead_nearest_wraps": nearest_wraps.astype(int),
                "lookahead_wrap_residuals": residuals,
                "lookahead_predictive_variances": np.asarray(variances, dtype=float),
                "lookahead_score_mode": "independent-nearest",
                "lookahead_map_wraps": nearest_wraps.astype(int),
                "lookahead_status": "ok",
                **slope_diagnostics,
            }
        )
        scored.append(replace(parent, diagnostics=diagnostics))
    return scored


def _lookahead_cluster_count(
    cluster_count: int,
    end_cluster_count: int,
    horizon: int | None,
) -> int:
    current = max(int(cluster_count), 1)
    end = max(int(end_cluster_count), current)
    if horizon is None:
        return end
    return min(current + max(int(horizon) - 1, 0), end)


_PARENT_LATTICE_CACHE_SCHEMA = 3


def _fingerprint_json(payload: Any) -> str:
    encoded = json.dumps(
        to_json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parent_lattice_cache_path(checkpoint_dir: Path, parent_branch_id: str) -> Path:
    branch_key = hashlib.sha256(str(parent_branch_id).encode("utf-8")).hexdigest()[:16]
    return Path(checkpoint_dir) / f"parent_{branch_key}.json"


def _load_parent_lattice_checkpoint(
    path: Path,
    *,
    fingerprint: str,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        payload = _load_json(path)
        if int(payload.get("schema", -1)) != _PARENT_LATTICE_CACHE_SCHEMA:
            return None
        if str(payload.get("fingerprint", "")) != str(fingerprint):
            return None
        candidates = [_branch_from_json(item) for item in payload.get("candidates", [])]
        diagnostics = dict(payload.get("diagnostics") or {})
    except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError):
        return None
    return candidates, diagnostics


def _write_parent_lattice_checkpoint(
    path: Path,
    *,
    fingerprint: str,
    parent_branch_id: str,
    candidates: Sequence[ActiveTimingBranch],
    diagnostics: dict[str, Any],
) -> None:
    _write_json_atomic(
        path,
        {
            "schema": _PARENT_LATTICE_CACHE_SCHEMA,
            "fingerprint": str(fingerprint),
            "parent_branch_id": str(parent_branch_id),
            "candidates": list(candidates),
            "diagnostics": diagnostics,
        },
    )


_WINDOW_GAPS_UNSET = object()


def _parent_prefix_stable_gap_order(
    parent: ActiveTimingBranch,
    *,
    active_jacobian: np.ndarray,
    parameter_scales: np.ndarray,
    resampled_gap_start: int,
) -> list[int]:
    gap_count = int(np.asarray(active_jacobian, dtype=float).shape[0])
    scaled_jacobian = np.asarray(active_jacobian, dtype=float) * np.asarray(
        parameter_scales,
        dtype=float,
    )
    information_order = [
        int(index)
        for index in np.argsort(np.sum(np.abs(scaled_jacobian), axis=1))
    ]
    inherited = (parent.diagnostics or {}).get("gap_order")
    if not isinstance(inherited, (list, tuple)):
        return information_order

    stable_order: list[int] = []
    seen: set[int] = set()
    for raw_index in inherited:
        active_index = int(raw_index) - int(resampled_gap_start)
        if active_index < 0 or active_index >= gap_count or active_index in seen:
            continue
        seen.add(active_index)
        stable_order.append(active_index)
    stable_order.extend(index for index in information_order if index not in seen)
    return stable_order


def _conditioned_lattice_refresh_candidates(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    stage_index: int,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
    proposal_source: str = "conditioned-bayesian-alias-lattice-refresh",
    proposal_family: str = "conditioned-lattice-refresh",
    branch_prefix: str | None = None,
    enabled: bool | None = None,
    parent_top_k: int | None = None,
    total_top_k: int | None = None,
    per_parent_top_k: int | None = None,
    beam_size: int | None = None,
    window_gaps: int | None | object = _WINDOW_GAPS_UNSET,
    prior_scales: Sequence[float] | None = None,
    phase_sigma: float | None = None,
    radius_sigma: float | None = None,
    min_radius: int | None = None,
    max_enum_per_gap: int | None = None,
    checkpoint_dir: Path | None = None,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
    lookahead_gap_jacobian: np.ndarray | None = None,
    lookahead_reference_phase_residuals: np.ndarray | None = None,
    lookahead_local_phase_slope_jacobian: np.ndarray | None = None,
    lookahead_reference_local_phase_slopes: np.ndarray | None = None,
    lookahead_local_phase_slope_sigmas: np.ndarray | None = None,
    gap_order_strategy: str = "information",
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    is_enabled = bool(config.conditioned_lattice_refresh_enabled) if enabled is None else bool(enabled)
    source = str(proposal_source)
    family = str(proposal_family)
    prefix = branch_prefix or f"stage_{stage_index:02d}_cond"
    if not bool(config.phase_slope_enabled):
        local_phase_slope_jacobian = None
        reference_local_phase_slopes = None
        local_phase_slope_sigmas = None
        lookahead_local_phase_slope_jacobian = None
        lookahead_reference_local_phase_slopes = None
        lookahead_local_phase_slope_sigmas = None
    local_slope_evidence_count = (
        0
        if local_phase_slope_jacobian is None
        else int(np.asarray(local_phase_slope_jacobian, dtype=float).shape[0])
    )
    gap_count = int(np.asarray(gap_jacobian, dtype=float).shape[0])
    if gap_count <= 0 or not is_enabled:
        return [], {
            "proposal_source": source,
            "enabled": is_enabled,
            "candidate_count": 0,
            "kept_count": 0,
        }

    active_parent_top_k = max(
        int(
            config.conditioned_lattice_parent_top_k
            if parent_top_k is None
            else parent_top_k
        ),
        0,
    )
    if config.parent_lookahead_phase_sigma is not None:
        lookahead_phase_sigma = float(config.parent_lookahead_phase_sigma)
    else:
        lookahead_phase_sigma = (
            float(config.stage_lattice_refresh_phase_sigma)
            if phase_sigma is None and config.stage_lattice_refresh_phase_sigma is not None
            else float(config.phase_sigma if phase_sigma is None else phase_sigma)
        )
    lookahead_parents = _with_parent_predictive_lookahead(
        parent_branches,
        gap_jacobian=np.asarray(
            gap_jacobian if lookahead_gap_jacobian is None else lookahead_gap_jacobian,
            dtype=float,
        ),
        reference_phase_residuals=np.asarray(
            reference_phase_residuals
            if lookahead_reference_phase_residuals is None
            else lookahead_reference_phase_residuals,
            dtype=float,
        ),
        phase_sigma=lookahead_phase_sigma,
        local_phase_slope_jacobian=lookahead_local_phase_slope_jacobian,
        reference_local_phase_slopes=lookahead_reference_local_phase_slopes,
        local_phase_slope_sigmas=lookahead_local_phase_slope_sigmas,
    )
    parent_certifications = [
        BranchCertification(
            branch_id=branch.branch_id,
            log_likelihood=-float((branch.diagnostics or {}).get("certified_reduced_chi2", math.inf)),
            fitted_theta={
                name: float(value)
                for name, value in zip(branch.theta_names, branch.theta_offset, strict=True)
            },
            covariance=np.eye(len(branch.theta_names), dtype=float),
            reduced_chi2=float((branch.diagnostics or {}).get("certified_reduced_chi2", math.inf)),
            weighted_rms_us=float((branch.diagnostics or {}).get("certified_weighted_rms_us", math.inf)),
            backend="conditioned-parent-diagnostic",
        )
        for branch in lookahead_parents
    ]
    ranked_parents, parent_selection_decision = select_budgeted_decision_branches(
        lookahead_parents,
        parent_certifications,
        config=BudgetedDecisionConfig(
            budget=active_parent_top_k,
            quality_floor=_decision_quality_floor(config, active_parent_top_k),
            family_floor=0,
            portfolio_floor=int(config.decision_portfolio_floor),
            quality_weight=float(config.decision_quality_weight),
            diagnostic_weight=0.0,
            diversity_weight=float(config.decision_diversity_weight),
            uncertainty_weight=float(config.decision_uncertainty_weight),
            prior_weight=float(config.decision_prior_weight),
            lookahead_weight=float(config.decision_lookahead_weight),
            prior_score_mode=str(config.decision_prior_score_mode),
            max_family_fraction=1.0,
            basin_floor=int(config.decision_basin_floor),
            max_basin_fraction=float(config.decision_max_basin_fraction),
        ),
    )
    if not ranked_parents:
        return [], {
            "proposal_source": source,
            "enabled": True,
            "candidate_count": 0,
            "kept_count": 0,
            "reason": "no-parents",
            "parent_selection_decision": parent_selection_decision,
        }

    jacobian = np.asarray(gap_jacobian, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float)
    raw_window_gaps = (
        config.conditioned_lattice_window_gaps
        if window_gaps is _WINDOW_GAPS_UNSET
        else window_gaps
    )
    window_gaps = (
        None
        if raw_window_gaps is None
        else max(int(raw_window_gaps), 0)
    )
    resampled_gap_start = 0
    if window_gaps is not None and 0 < window_gaps < gap_count:
        resampled_gap_start = gap_count - window_gaps
    active_jacobian = jacobian[resampled_gap_start:]
    active_reference = reference[resampled_gap_start:]
    candidates: list[ActiveTimingBranch] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    parent_records: list[dict[str, Any]] = []
    parent_cache_hit_count = 0
    parent_cache_miss_count = 0
    active_prior_scales = (
        _conditioned_lattice_prior_scales(config)
        if prior_scales is None
        else tuple(float(item) for item in prior_scales)
    )
    active_per_parent_top_k = max(
        int(config.conditioned_lattice_per_parent_top_k if per_parent_top_k is None else per_parent_top_k),
        1,
    )
    active_total_top_k = max(
        int(config.conditioned_lattice_top_k if total_top_k is None else total_top_k),
        0,
    )
    active_beam_size = (
        int(config.stage_lattice_refresh_beam_size)
        if (config.conditioned_lattice_beam_size if beam_size is None else beam_size) is None
        else int(config.conditioned_lattice_beam_size if beam_size is None else beam_size)
    )
    active_phase_sigma = (
        float(config.stage_lattice_refresh_phase_sigma)
        if phase_sigma is None and config.stage_lattice_refresh_phase_sigma is not None
        else float(config.phase_sigma if phase_sigma is None else phase_sigma)
    )
    active_radius_sigma = float(
        config.stage_lattice_refresh_radius_sigma
        if radius_sigma is None
        else radius_sigma
    )
    active_min_radius = int(
        config.stage_lattice_refresh_min_radius
        if min_radius is None
        else min_radius
    )
    active_max_enum_per_gap = int(
        config.stage_lattice_refresh_max_enum_per_gap
        if max_enum_per_gap is None
        else max_enum_per_gap
    )
    order_strategy = str(gap_order_strategy).strip().lower()
    if order_strategy not in {"information", "parent-prefix-then-append"}:
        raise ValueError(f"Unsupported gap_order_strategy: {gap_order_strategy!r}")

    for parent_rank, parent in enumerate(ranked_parents):
        parent_offset = np.asarray(parent.theta_offset, dtype=float)
        if parent_offset.shape != (jacobian.shape[1],):
            continue
        if resampled_gap_start > 0 and len(parent.wraps) < resampled_gap_start:
            continue
        conditioned_reference = active_reference + active_jacobian @ parent_offset
        conditioned_slope_reference = (
            None
            if reference_local_phase_slopes is None
            else np.asarray(reference_local_phase_slopes, dtype=float)
            + np.asarray(local_phase_slope_jacobian, dtype=float) @ parent_offset
        )
        seed_branch = replace(
            parent,
            wraps=(),
            theta_offset=np.zeros_like(parent_offset),
        )
        active_gap_order = (
            None
            if order_strategy == "information"
            else _parent_prefix_stable_gap_order(
                parent,
                active_jacobian=active_jacobian,
                parameter_scales=np.asarray(parameter_scales, dtype=float),
                resampled_gap_start=resampled_gap_start,
            )
        )
        cache_identity = {
            "schema": _PARENT_LATTICE_CACHE_SCHEMA,
            "stage": int(stage_index),
            "source": source,
            "family": family,
            "parent": {
                "branch_id": parent.branch_id,
                "structure_id": parent.structure_id,
                "binary_model": parent.binary_model,
                "wraps": parent.wraps,
                "theta_names": parent.theta_names,
                "theta_offset": parent_offset,
                "log_prior": float(parent.log_prior),
            },
            "active_jacobian": active_jacobian,
            "conditioned_reference": conditioned_reference,
            "local_phase_slope_jacobian": local_phase_slope_jacobian,
            "conditioned_reference_local_phase_slopes": conditioned_slope_reference,
            "local_phase_slope_sigmas": local_phase_slope_sigmas,
            "parameter_scales": np.asarray(parameter_scales, dtype=float),
            "settings": {
                "top_k": active_per_parent_top_k,
                "beam_size": max(int(active_beam_size), 1),
                "phase_sigma": active_phase_sigma,
                "prior_scales": active_prior_scales,
                "radius_sigma": active_radius_sigma,
                "min_radius": active_min_radius,
                "max_enum_per_gap": active_max_enum_per_gap,
                "wrap_prior_scale": float(config.wrap_prior_scale),
                "gap_order_strategy": order_strategy,
                "gap_order": active_gap_order,
            },
        }
        cache_fingerprint = _fingerprint_json(cache_identity)
        cache_path = (
            _parent_lattice_cache_path(Path(checkpoint_dir), parent.branch_id)
            if checkpoint_dir is not None
            else None
        )
        cached = (
            _load_parent_lattice_checkpoint(cache_path, fingerprint=cache_fingerprint)
            if cache_path is not None
            else None
        )
        if cached is not None:
            parent_candidates, parent_diagnostics = cached
            parent_cache_hit_count += 1
            cache_status = "hit"
        else:
            parent_candidates, parent_diagnostics = _lattice_portfolio_candidates(
                seed_branch=seed_branch,
                stage_index=stage_index,
                family=family,
                branch_prefix=f"{prefix}_p{parent_rank:04d}",
                proposal_source=source,
                gap_jacobian=active_jacobian,
                reference_phase_residuals=conditioned_reference,
                parameter_scales=np.asarray(parameter_scales, dtype=float),
                top_k=active_per_parent_top_k,
                beam_size=max(int(active_beam_size), 1),
                phase_sigma=active_phase_sigma,
                prior_scales=active_prior_scales,
                radius_sigma=active_radius_sigma,
                min_radius=active_min_radius,
                max_enum_per_gap=active_max_enum_per_gap,
                wrap_prior_scale=float(config.wrap_prior_scale),
                gap_order=active_gap_order,
                local_phase_slope_jacobian=local_phase_slope_jacobian,
                reference_local_phase_slopes=conditioned_slope_reference,
                local_phase_slope_sigmas=local_phase_slope_sigmas,
            )
            parent_cache_miss_count += 1
            cache_status = "miss"
            if cache_path is not None:
                _write_parent_lattice_checkpoint(
                    cache_path,
                    fingerprint=cache_fingerprint,
                    parent_branch_id=parent.branch_id,
                    candidates=parent_candidates,
                    diagnostics=parent_diagnostics,
                )
        parent_records.append(
            {
                "parent_branch_id": parent.branch_id,
                "parent_rank": int(parent_rank),
                "candidate_count": int(parent_diagnostics.get("candidate_count", len(parent_candidates))),
                "kept_count": len(parent_candidates),
                "resampled_gap_start": int(resampled_gap_start),
                "resampled_gap_count": int(active_jacobian.shape[0]),
                "cache_status": cache_status,
                "cache_fingerprint": cache_fingerprint,
                "cache_path": cache_path,
                "gap_order": active_gap_order,
                "local_slope_evidence_count": int(local_slope_evidence_count),
            }
        )
        for branch in parent_candidates:
            theta_offset = parent_offset + np.asarray(branch.theta_offset, dtype=float)
            suffix_wraps = tuple(int(item) for item in branch.wraps)
            combined_wraps = tuple(int(item) for item in parent.wraps[:resampled_gap_start]) + suffix_wraps
            key = (
                combined_wraps,
                tuple(float(f"{value:.18e}") for value in theta_offset.tolist()),
            )
            if key in seen:
                continue
            seen.add(key)
            predicted_wraps = reference + jacobian @ theta_offset
            wrap_residual = predicted_wraps - np.asarray(combined_wraps, dtype=float)
            diagnostics = dict(branch.diagnostics or {})
            if diagnostics.get("evidence_mode") == "phase+slope":
                diagnostics["certification_theta_offset"] = (
                    parent_offset
                    + np.asarray(
                        diagnostics.get("certification_theta_offset", branch.theta_offset),
                        dtype=float,
                    )
                )
            diagnostics.update(
                {
                    "proposal_family": family,
                    "stage": int(stage_index),
                    "parent_branch_id": parent.branch_id,
                    "parent_rank": int(parent_rank),
                    "conditioned_reference_phase_residuals": conditioned_reference,
                    "conditioned_reference_local_phase_slopes": conditioned_slope_reference,
                    "parent_theta_offset": parent_offset,
                    "delta_theta_offset": np.asarray(branch.theta_offset, dtype=float),
                    "resampled_gap_start": int(resampled_gap_start),
                    "resampled_gap_count": int(active_jacobian.shape[0]),
                    "suffix_wraps": suffix_wraps,
                    "predicted_wraps": predicted_wraps,
                    "wrap_residual_norm": float(np.linalg.norm(wrap_residual)),
                    "parent_path_log_prior": float(parent.log_prior),
                    "proposal_log_prior": float(branch.log_prior),
                    "gap_order_strategy": order_strategy,
                }
            )
            candidates.append(
                replace(
                    branch,
                    branch_id=f"{prefix}_{len(candidates):04d}",
                    wraps=combined_wraps,
                    theta_offset=np.asarray(theta_offset, dtype=float),
                    log_prior=float(parent.log_prior) + float(branch.log_prior),
                    proposal_source="staged-budgeted-bayesian-search",
                    diagnostics=diagnostics,
                )
            )

    selected, decision = select_budgeted_decision_branches(
        candidates,
        [
            BranchCertification(
                branch_id=branch.branch_id,
                log_likelihood=float((branch.diagnostics or {}).get("proposal_score", branch.log_prior)),
                fitted_theta={
                    name: float(value)
                    for name, value in zip(branch.theta_names, branch.theta_offset, strict=True)
                },
                covariance=np.eye(len(branch.theta_names), dtype=float),
                reduced_chi2=float((branch.diagnostics or {}).get("wrap_residual_norm", 0.0)) ** 2,
                weighted_rms_us=float((branch.diagnostics or {}).get("wrap_residual_norm", 0.0)),
                backend=f"{family}-diagnostic",
            )
            for branch in candidates
        ],
        config=BudgetedDecisionConfig(
            budget=active_total_top_k,
            family_floor=_decision_family_floor(config, active_total_top_k),
            portfolio_floor=int(config.decision_portfolio_floor),
            quality_weight=0.35,
            diagnostic_weight=max(float(config.decision_diagnostic_weight), 1.0),
            diversity_weight=max(float(config.decision_diversity_weight), 0.75),
            prior_weight=float(config.decision_prior_weight),
            prior_score_mode=str(config.decision_prior_score_mode),
            max_family_fraction=1.0,
            basin_floor=int(config.decision_basin_floor),
            max_basin_fraction=float(config.decision_max_basin_fraction),
        ),
    )
    return list(selected), {
        "proposal_source": source,
        "enabled": True,
        "candidate_count": len(candidates),
        "kept_count": len(selected),
        "parent_count": len(ranked_parents),
        "resampled_gap_start": int(resampled_gap_start),
        "resampled_gap_count": int(active_jacobian.shape[0]),
        "gap_order_strategy": order_strategy,
        "parents": parent_records,
        "parent_cache_hit_count": int(parent_cache_hit_count),
        "parent_cache_miss_count": int(parent_cache_miss_count),
        "local_slope_evidence_count": int(local_slope_evidence_count),
        "evidence_mode": "phase+slope" if local_slope_evidence_count else "phase-only",
        "parent_selection_decision": parent_selection_decision,
        "decision": decision,
    }


def _posterior_lattice_refresh_candidates(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    stage_index: int,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
    checkpoint_dir: Path | None = None,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
    lookahead_gap_jacobian: np.ndarray | None = None,
    lookahead_reference_phase_residuals: np.ndarray | None = None,
    lookahead_local_phase_slope_jacobian: np.ndarray | None = None,
    lookahead_reference_local_phase_slopes: np.ndarray | None = None,
    lookahead_local_phase_slope_sigmas: np.ndarray | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    beam_size = (
        config.posterior_lattice_beam_size
        if config.posterior_lattice_beam_size is not None
        else config.conditioned_lattice_beam_size
    )
    return _conditioned_lattice_refresh_candidates(
        parent_branches=parent_branches,
        stage_index=stage_index,
        gap_jacobian=gap_jacobian,
        reference_phase_residuals=reference_phase_residuals,
        parameter_scales=parameter_scales,
        config=config,
        proposal_source="posterior-conditioned-bayesian-alias-lattice-refresh",
        proposal_family="posterior-lattice-refresh",
        branch_prefix=f"stage_{stage_index:02d}_post",
        enabled=bool(config.posterior_lattice_refresh_enabled),
        parent_top_k=int(config.posterior_lattice_parent_top_k),
        total_top_k=int(config.posterior_lattice_top_k),
        per_parent_top_k=int(config.posterior_lattice_per_parent_top_k),
        beam_size=beam_size,
        window_gaps=None,
        prior_scales=_posterior_lattice_prior_scales(config),
        phase_sigma=config.posterior_lattice_phase_sigma,
        radius_sigma=config.posterior_lattice_radius_sigma,
        min_radius=config.posterior_lattice_min_radius,
        max_enum_per_gap=config.posterior_lattice_max_enum_per_gap,
        checkpoint_dir=checkpoint_dir,
        local_phase_slope_jacobian=local_phase_slope_jacobian,
        reference_local_phase_slopes=reference_local_phase_slopes,
        local_phase_slope_sigmas=local_phase_slope_sigmas,
        lookahead_gap_jacobian=lookahead_gap_jacobian,
        lookahead_reference_phase_residuals=lookahead_reference_phase_residuals,
        lookahead_local_phase_slope_jacobian=lookahead_local_phase_slope_jacobian,
        lookahead_reference_local_phase_slopes=lookahead_reference_local_phase_slopes,
        lookahead_local_phase_slope_sigmas=lookahead_local_phase_slope_sigmas,
    )


def _posterior_stable_order_lattice_refresh_candidates(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    stage_index: int,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
    checkpoint_dir: Path | None = None,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
    lookahead_gap_jacobian: np.ndarray | None = None,
    lookahead_reference_phase_residuals: np.ndarray | None = None,
    lookahead_local_phase_slope_jacobian: np.ndarray | None = None,
    lookahead_reference_local_phase_slopes: np.ndarray | None = None,
    lookahead_local_phase_slope_sigmas: np.ndarray | None = None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    beam_size = (
        config.posterior_lattice_beam_size
        if config.posterior_lattice_beam_size is not None
        else config.conditioned_lattice_beam_size
    )
    return _conditioned_lattice_refresh_candidates(
        parent_branches=parent_branches,
        stage_index=stage_index,
        gap_jacobian=gap_jacobian,
        reference_phase_residuals=reference_phase_residuals,
        parameter_scales=parameter_scales,
        config=config,
        proposal_source="posterior-prefix-stable-bayesian-alias-lattice-refresh",
        proposal_family="posterior-prefix-stable-refresh",
        branch_prefix=f"stage_{stage_index:02d}_stable",
        enabled=bool(config.posterior_lattice_stable_order_enabled),
        parent_top_k=int(config.posterior_lattice_parent_top_k),
        total_top_k=int(config.posterior_lattice_stable_order_top_k),
        per_parent_top_k=int(config.posterior_lattice_stable_order_per_parent_top_k),
        beam_size=beam_size,
        window_gaps=None,
        prior_scales=_posterior_lattice_prior_scales(config),
        phase_sigma=config.posterior_lattice_phase_sigma,
        radius_sigma=config.posterior_lattice_radius_sigma,
        min_radius=config.posterior_lattice_min_radius,
        max_enum_per_gap=config.posterior_lattice_max_enum_per_gap,
        checkpoint_dir=checkpoint_dir,
        local_phase_slope_jacobian=local_phase_slope_jacobian,
        reference_local_phase_slopes=reference_local_phase_slopes,
        local_phase_slope_sigmas=local_phase_slope_sigmas,
        lookahead_gap_jacobian=lookahead_gap_jacobian,
        lookahead_reference_phase_residuals=lookahead_reference_phase_residuals,
        lookahead_local_phase_slope_jacobian=lookahead_local_phase_slope_jacobian,
        lookahead_reference_local_phase_slopes=lookahead_reference_local_phase_slopes,
        lookahead_local_phase_slope_sigmas=lookahead_local_phase_slope_sigmas,
        gap_order_strategy="parent-prefix-then-append",
    )


def _branch_family(branch: ActiveTimingBranch) -> str:
    diagnostics = branch.diagnostics or {}
    return str(diagnostics.get("proposal_family") or branch.proposal_source or "unknown").lower()


def _decision_family_floor(config: StagedBudgetedSearchConfig, budget: int) -> int:
    absolute_floor = max(int(config.decision_family_floor), 0)
    fraction = max(float(config.decision_family_floor_fraction), 0.0)
    fractional_floor = int(math.ceil(fraction * max(int(budget), 0)))
    return max(absolute_floor, fractional_floor)


def _decision_quality_floor(config: StagedBudgetedSearchConfig, budget: int) -> int:
    absolute_floor = max(int(config.decision_quality_floor), 0)
    fraction = max(float(config.decision_quality_floor_fraction), 0.0)
    fractional_floor = int(math.ceil(fraction * max(int(budget), 0)))
    return max(absolute_floor, fractional_floor)


def _score_certification(certification: BranchCertification) -> tuple[float, float, str]:
    chi2 = certification.reduced_chi2
    wrms = certification.weighted_rms_us
    return (
        float(chi2) if chi2 is not None and np.isfinite(float(chi2)) else float("inf"),
        float(wrms) if wrms is not None and np.isfinite(float(wrms)) else float("inf"),
        str(certification.branch_id),
    )


def _is_certified_parent(branch: ActiveTimingBranch) -> bool:
    diagnostics = branch.diagnostics or {}
    chi2 = diagnostics.get("certified_reduced_chi2")
    wrms = diagnostics.get("certified_weighted_rms_us")
    return (
        chi2 is not None
        and wrms is not None
        and np.isfinite(float(chi2))
        and np.isfinite(float(wrms))
    )


def _adaptive_global_reacquisition_decision(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    stage_certifications: Sequence[BranchCertification],
    config: StagedBudgetedSearchConfig,
) -> dict[str, Any]:
    parent_values = [
        float((branch.diagnostics or {}).get("certified_reduced_chi2"))
        for branch in parent_branches
        if (branch.diagnostics or {}).get("certified_reduced_chi2") is not None
        and np.isfinite(float((branch.diagnostics or {}).get("certified_reduced_chi2")))
    ]
    stage_values = [
        float(certification.reduced_chi2)
        for certification in stage_certifications
        if certification.reduced_chi2 is not None
        and np.isfinite(float(certification.reduced_chi2))
    ]
    parent_best = min(parent_values) if parent_values else None
    stage_best = min(stage_values) if stage_values else None
    ratio = max(float(config.adaptive_global_reacquisition_chi2_ratio), 1.0)
    absolute_floor = max(
        float(config.adaptive_global_reacquisition_min_reduced_chi2),
        0.0,
    )
    relative_threshold = None if parent_best is None else float(parent_best * ratio)
    absolute_quality_enabled = absolute_floor > 0.0
    trigger_threshold = absolute_floor if absolute_quality_enabled else relative_threshold
    if trigger_threshold is not None and relative_threshold is not None:
        trigger_threshold = min(float(trigger_threshold), relative_threshold)

    if not bool(config.adaptive_global_reacquisition_enabled):
        triggered = False
        reason = "disabled"
    elif stage_best is None:
        triggered = True
        reason = "no-certified-stage-candidate"
    elif relative_threshold is not None and stage_best > relative_threshold:
        triggered = True
        reason = "certified-quality-collapse"
    elif absolute_quality_enabled and stage_best > absolute_floor:
        triggered = True
        reason = "absolute-quality-failure"
    elif parent_best is None:
        triggered = False
        reason = "no-certified-parent-quality-sufficient"
    else:
        triggered = False
        reason = "quality-stable"

    return {
        "enabled": bool(config.adaptive_global_reacquisition_enabled),
        "triggered": bool(triggered),
        "reason": reason,
        "parent_best_reduced_chi2": parent_best,
        "stage_best_reduced_chi2": stage_best,
        "chi2_ratio_threshold": ratio,
        "absolute_reduced_chi2_threshold": absolute_floor,
        "trigger_reduced_chi2_threshold": trigger_threshold,
    }


def _adaptive_global_reacquisition_config(
    *,
    config: StagedBudgetedSearchConfig,
    base_certify_budget: int,
) -> tuple[StagedBudgetedSearchConfig, int]:
    top_k_multiplier = max(
        float(config.adaptive_global_reacquisition_top_k_multiplier),
        1.0,
    )
    certify_multiplier = max(
        float(config.adaptive_global_reacquisition_certify_budget_multiplier),
        1.0,
    )
    top_k = int(
        math.ceil(
            max(
                int(config.initial_lattice_top_k),
                int(config.stage_lattice_refresh_top_k),
                1,
            )
            * top_k_multiplier
        )
    )
    certify_budget = int(math.ceil(max(int(base_certify_budget), 1) * certify_multiplier))
    reacquisition_config = replace(
        config,
        stage_lattice_refresh_enabled=True,
        stage_lattice_refresh_top_k=top_k,
        stage_lattice_refresh_beam_size=max(
            int(config.initial_lattice_beam_size),
            int(config.stage_lattice_refresh_beam_size),
            1,
        ),
        stage_lattice_refresh_phase_sigma=config.initial_lattice_phase_sigma,
        stage_lattice_refresh_prior_scale=float(config.initial_lattice_prior_scale),
        stage_lattice_refresh_prior_scales=tuple(config.initial_lattice_prior_scales),
        stage_lattice_refresh_radius_sigma=float(config.initial_lattice_radius_sigma),
        stage_lattice_refresh_min_radius=int(config.initial_lattice_min_radius),
        stage_lattice_refresh_max_enum_per_gap=int(config.initial_lattice_max_enum_per_gap),
    )
    return reacquisition_config, certify_budget


def _adaptive_prior_tempering_decision(
    *,
    best_branch: ActiveTimingBranch | None,
    best_certification: BranchCertification | None,
    searched_prior_scales: Sequence[float],
    config: StagedBudgetedSearchConfig,
) -> dict[str, Any]:
    active_scales = _active_prior_scales(1.0, searched_prior_scales)
    widest_scale = max(active_scales)
    reduced_chi2 = (
        float(best_certification.reduced_chi2)
        if best_certification is not None
        and best_certification.reduced_chi2 is not None
        and np.isfinite(float(best_certification.reduced_chi2))
        else None
    )
    raw_winning_scale = (
        (best_branch.diagnostics or {}).get("portfolio_prior_scale")
        if best_branch is not None
        else None
    )
    try:
        winning_scale = float(raw_winning_scale)
    except (TypeError, ValueError):
        winning_scale = None
    if winning_scale is not None and not np.isfinite(winning_scale):
        winning_scale = None
    quality_threshold = max(
        float(config.adaptive_global_reacquisition_min_reduced_chi2),
        0.0,
    )

    if not bool(config.adaptive_prior_tempering_enabled):
        triggered = False
        reason = "disabled"
    elif reduced_chi2 is None:
        triggered = False
        reason = "no-certified-reacquisition-candidate"
    elif quality_threshold > 0.0 and reduced_chi2 <= quality_threshold:
        triggered = False
        reason = "quality-sufficient"
    elif winning_scale is None:
        triggered = False
        reason = "missing-winning-prior-scale"
    elif not np.isclose(winning_scale, widest_scale, rtol=1.0e-9, atol=0.0):
        triggered = False
        reason = "interior-prior-temperature"
    else:
        triggered = True
        reason = "poor-quality-at-prior-boundary"

    return {
        "enabled": bool(config.adaptive_prior_tempering_enabled),
        "triggered": bool(triggered),
        "reason": reason,
        "best_reduced_chi2": reduced_chi2,
        "quality_threshold": quality_threshold,
        "winning_prior_scale": winning_scale,
        "widest_searched_prior_scale": widest_scale,
        "searched_prior_scales": list(active_scales),
    }


def _adaptive_prior_tempering_config(
    *,
    config: StagedBudgetedSearchConfig,
    certify_budget: int,
) -> tuple[StagedBudgetedSearchConfig, int]:
    searched_scales = _active_prior_scales(
        float(config.stage_lattice_refresh_prior_scale),
        config.stage_lattice_refresh_prior_scales,
    )
    factor = max(float(config.adaptive_prior_tempering_factor), 1.0)
    step_count = max(int(config.adaptive_prior_tempering_steps), 0)
    if not bool(config.adaptive_prior_tempering_enabled) or factor <= 1.0 or step_count == 0:
        return config, max(int(certify_budget), 1)

    widest_scale = max(searched_scales)
    new_scales = tuple(
        float(widest_scale * factor**step)
        for step in range(1, step_count + 1)
    )
    tempered_config = replace(
        config,
        stage_lattice_refresh_prior_scale=new_scales[0],
        stage_lattice_refresh_prior_scales=new_scales[1:],
    )
    return tempered_config, max(int(certify_budget), 1)


def _adaptive_slope_tempering_decision(
    *,
    best_certification: BranchCertification | None,
    evidence: LocalPhaseSlopeEvidence | None,
    config: StagedBudgetedSearchConfig,
) -> dict[str, Any]:
    reduced_chi2 = (
        float(best_certification.reduced_chi2)
        if best_certification is not None
        and best_certification.reduced_chi2 is not None
        and np.isfinite(float(best_certification.reduced_chi2))
        else None
    )
    quality_threshold = max(
        float(config.adaptive_global_reacquisition_min_reduced_chi2),
        0.0,
    )
    base_scale = max(float(config.phase_slope_sigma_scale), 1.0e-12)
    sigma_floor = max(float(config.phase_slope_sigma_floor), 0.0)
    factor = max(float(config.adaptive_slope_tempering_factor), 1.0)
    formal_sigmas = (
        np.zeros(0, dtype=float)
        if evidence is None
        else np.asarray(evidence.phase_change_sigma, dtype=float).reshape(-1)
    )
    valid_sigmas = formal_sigmas[
        np.isfinite(formal_sigmas) & (formal_sigmas > 0.0)
    ]
    floor_crossing_scale = (
        float(np.max(sigma_floor / valid_sigmas))
        if valid_sigmas.size and sigma_floor > 0.0
        else base_scale
    )
    target_scale = max(floor_crossing_scale, base_scale * factor)
    if factor > 1.0:
        geometric_steps = max(
            int(math.ceil(math.log(target_scale / base_scale) / math.log(factor))),
            1,
        )
        tempered_scale = float(base_scale * factor**geometric_steps)
    else:
        tempered_scale = base_scale

    if not bool(config.adaptive_slope_tempering_enabled):
        triggered = False
        reason = "disabled"
    elif not bool(config.phase_slope_enabled):
        triggered = False
        reason = "phase-slope-disabled"
    elif evidence is None or not valid_sigmas.size:
        triggered = False
        reason = "no-slope-evidence"
    elif reduced_chi2 is None:
        triggered = False
        reason = "no-certified-reacquisition-candidate"
    elif quality_threshold > 0.0 and reduced_chi2 <= quality_threshold:
        triggered = False
        reason = "quality-sufficient"
    elif factor <= 1.0 or np.isclose(tempered_scale, base_scale):
        triggered = False
        reason = "no-new-slope-temperature"
    else:
        triggered = True
        reason = "poor-quality-needs-slope-tempering"

    return {
        "enabled": bool(config.adaptive_slope_tempering_enabled),
        "triggered": bool(triggered),
        "reason": reason,
        "best_reduced_chi2": reduced_chi2,
        "quality_threshold": quality_threshold,
        "base_slope_sigma_scale": base_scale,
        "slope_sigma_floor": sigma_floor,
        "formal_sigma_min": float(np.min(valid_sigmas)) if valid_sigmas.size else None,
        "formal_sigma_max": float(np.max(valid_sigmas)) if valid_sigmas.size else None,
        "floor_crossing_scale": floor_crossing_scale,
        "tempered_slope_sigma_scale": tempered_scale,
        "tempering_factor": factor,
    }


_SPIN_DERIVATIVE_RE = re.compile(r"^F(\d+)$")


def _normalise_parameter_release_blocks(blocks: Sequence[Any]) -> tuple[tuple[str, ...], ...]:
    normalised: list[tuple[str, ...]] = []
    seen_blocks: set[tuple[str, ...]] = set()
    for raw_block in blocks:
        raw_items = [raw_block] if isinstance(raw_block, str) else list(raw_block)
        block: list[str] = []
        seen_names: set[str] = set()
        for item in raw_items:
            name = str(item).upper()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            block.append(name)
        block_tuple = tuple(block)
        if not block_tuple or block_tuple in seen_blocks:
            continue
        seen_blocks.add(block_tuple)
        normalised.append(block_tuple)
    return tuple(normalised)


def _par_parameter_names(par_path: Path) -> set[str]:
    names: set[str] = set()
    for raw_line in Path(par_path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(line.split()[0].upper())
    return names


def _missing_parameter_line(parameter_name: str) -> str:
    name = str(parameter_name).upper()
    match = _SPIN_DERIVATIVE_RE.match(name)
    if match is None or int(match.group(1)) < 2:
        raise ValueError(f"Cannot automatically add missing timing parameter '{name}'.")
    order = int(match.group(1))
    uncertainty_exponent = 11 + 8 * order
    return f"{name:<14} 0.0                       1  1e-{uncertainty_exponent}"


def _model_native_parameter_line(model, parameter_name: str) -> str:
    name = str(parameter_name).upper()
    if name not in model.params:
        raise ValueError(f"Cannot automatically add missing timing parameter '{name}'.")
    parameter = model[name]
    if parameter.value is None:
        parameter.value = 0.0
    if hasattr(parameter, "frozen"):
        parameter.frozen = False
    line = str(parameter.as_parfile_line()).strip()
    if not line:
        raise ValueError(f"PINT could not serialize missing timing parameter '{name}'.")
    return line


def _spin_insert_index(lines: Sequence[str]) -> int:
    insert_index = 0
    for index, raw_line in enumerate(lines):
        parts = raw_line.strip().split()
        if parts and _SPIN_DERIVATIVE_RE.match(parts[0].upper()):
            insert_index = index + 1
    return insert_index


def _write_parameter_release_par(
    *,
    par_path: Path,
    parameter_names: Sequence[str],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    par_path = Path(par_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = tuple(str(name).upper() for name in parameter_names)
    existing = _par_parameter_names(par_path)
    missing = [name for name in requested if name not in existing]
    if not missing:
        return par_path, {
            "source_par_path": par_path,
            "par_path": par_path,
            "added_parameters": [],
            "existing_parameters": list(requested),
        }

    lines = par_path.read_text(encoding="utf-8").splitlines()
    insert_index = _spin_insert_index(lines)
    model = None
    for name in missing:
        match = _SPIN_DERIVATIVE_RE.match(name)
        if match is not None and int(match.group(1)) >= 2:
            lines.insert(insert_index, _missing_parameter_line(name))
            insert_index += 1
            continue
        if model is None:
            model = _load_pint_model(par_path)
        lines.append(_model_native_parameter_line(model, name))
    release_name = "_".join(requested).lower() or "none"
    augmented_path = output_dir / f"augmented_{release_name}.par"
    augmented_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return augmented_path, {
        "source_par_path": par_path,
        "par_path": augmented_path,
        "added_parameters": missing,
        "existing_parameters": [name for name in requested if name in existing],
    }


def _augment_theta_names(
    theta_names: Sequence[str],
    parameter_names: Sequence[str],
) -> tuple[str, ...]:
    expanded = [str(name).upper() for name in theta_names]
    seen = set(expanded)
    for name in parameter_names:
        upper_name = str(name).upper()
        if upper_name not in seen:
            seen.add(upper_name)
            expanded.append(upper_name)
    return tuple(expanded)


def _augment_fit_parameter_names(
    fit_parameter_names: Sequence[str] | None,
    parameter_names: Sequence[str],
) -> tuple[str, ...] | None:
    if fit_parameter_names is None:
        return None
    return _augment_theta_names(fit_parameter_names, parameter_names)


def _augment_branch_parameters(
    branch: ActiveTimingBranch,
    *,
    theta_names: Sequence[str],
    branch_id: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> ActiveTimingBranch:
    target_theta_names = tuple(str(name).upper() for name in theta_names)
    offset_by_name = {
        str(name).upper(): float(value)
        for name, value in zip(branch.theta_names, np.asarray(branch.theta_offset, dtype=float), strict=True)
    }
    missing_existing = [name for name in branch.theta_names if str(name).upper() not in target_theta_names]
    if missing_existing:
        raise ValueError(
            "Expanded theta_names must include all existing branch parameters: "
            + ", ".join(str(name).upper() for name in missing_existing)
        )
    merged_diagnostics = dict(branch.diagnostics or {})
    if "certification_theta_offset" in merged_diagnostics:
        certification_offset_by_name = {
            str(name).upper(): float(value)
            for name, value in zip(
                branch.theta_names,
                np.asarray(merged_diagnostics["certification_theta_offset"], dtype=float),
                strict=True,
            )
        }
        merged_diagnostics["certification_theta_offset"] = [
            float(certification_offset_by_name.get(name, 0.0))
            for name in target_theta_names
        ]
    merged_diagnostics["parameter_release_source_branch_id"] = branch.branch_id
    if diagnostics:
        merged_diagnostics.update(diagnostics)
    return replace(
        branch,
        branch_id=branch_id or branch.branch_id,
        theta_names=target_theta_names,
        theta_offset=np.asarray(
            [float(offset_by_name.get(name, 0.0)) for name in target_theta_names],
            dtype=float,
        ),
        diagnostics=merged_diagnostics,
    )


def _resume_dynamic_parameter_state(
    *,
    par_path: Path,
    output_dir: Path,
    current_branches: list[ActiveTimingBranch],
    resolved_theta_names: Sequence[str],
    resolved_fit_parameter_names: Sequence[str] | None,
) -> tuple[Path, list[str], list[str] | None, dict[str, Any]]:
    active_theta_names = list(_augment_theta_names(resolved_theta_names, []))
    seen = set(active_theta_names)
    for branch in current_branches:
        for name in branch.theta_names:
            upper_name = str(name).upper()
            if upper_name in seen:
                continue
            seen.add(upper_name)
            active_theta_names.append(upper_name)

    active_fit_parameter_names = (
        None
        if resolved_fit_parameter_names is None
        else list(_augment_theta_names(resolved_fit_parameter_names, active_theta_names))
    )
    resume_dir = Path(output_dir) / "resume_parameter_release"
    active_par_path, par_diagnostics = _write_parameter_release_par(
        par_path=par_path,
        parameter_names=active_theta_names,
        output_dir=resume_dir,
    )

    for index, branch in enumerate(list(current_branches)):
        branch_theta_names = tuple(str(name).upper() for name in branch.theta_names)
        if branch_theta_names == tuple(active_theta_names):
            continue
        current_branches[index] = _augment_branch_parameters(
            branch,
            theta_names=active_theta_names,
            diagnostics={"resume_dynamic_parameter_state": True},
        )

    diagnostics = {
        **par_diagnostics,
        "source": "resume-dynamic-parameter-state",
        "branch_count": len(current_branches),
        "theta_names": active_theta_names,
        "fit_parameter_names": active_fit_parameter_names,
    }
    return active_par_path, active_theta_names, active_fit_parameter_names, diagnostics


def _count_tim_toa_lines(tim_path: Path) -> int:
    return sum(
        1
        for raw_line in Path(tim_path).read_text(encoding="utf-8").splitlines()
        if _tim_source_name(raw_line) is not None
    )


def _finite_float(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _select_parameter_release_probe(
    *,
    baseline_certification: BranchCertification,
    probes: Sequence[dict[str, Any]],
    toa_count: int,
    config: StagedBudgetedSearchConfig,
) -> dict[str, Any]:
    baseline_wrms = _finite_float(baseline_certification.weighted_rms_us)
    if baseline_wrms is None:
        return {"selected": False, "reason": "baseline-wrms-unavailable", "probes": []}
    baseline_chi2 = _finite_float(baseline_certification.chi2)
    if baseline_chi2 is None:
        baseline_chi2 = _finite_float(baseline_certification.reduced_chi2)
    log_n = math.log(max(int(toa_count), 2))
    records: list[dict[str, Any]] = []
    for probe in probes:
        best_certification = probe.get("best_certification")
        if not isinstance(best_certification, BranchCertification):
            records.append(
                {
                    "block": list(probe.get("block", ())),
                    "accepted": False,
                    "reason": "no-best-certification",
                }
            )
            continue
        probe_wrms = _finite_float(best_certification.weighted_rms_us)
        if probe_wrms is None:
            records.append(
                {
                    "block": list(probe.get("block", ())),
                    "accepted": False,
                    "reason": "probe-wrms-unavailable",
                }
            )
            continue
        improvement = float(baseline_wrms) - float(probe_wrms)
        fractional_improvement = improvement / max(abs(float(baseline_wrms)), 1.0e-12)
        probe_chi2 = _finite_float(best_certification.chi2)
        if probe_chi2 is None:
            probe_chi2 = _finite_float(best_certification.reduced_chi2)
        if baseline_chi2 is not None and probe_chi2 is not None:
            likelihood_gain = float(baseline_chi2) - float(probe_chi2)
        else:
            likelihood_gain = improvement
        added_parameter_count = int(probe.get("added_parameter_count", len(tuple(probe.get("block", ()))))) 
        penalty = (
            max(float(config.parameter_release_bic_penalty_weight), 0.0)
            * max(added_parameter_count, 0)
            * log_n
        )
        evidence_gain = float(likelihood_gain) - float(penalty)
        minimum_absolute_improvement = max(
            float(config.parameter_release_min_wrms_improvement_us),
            0.0,
        )
        minimum_fractional_improvement = max(
            float(config.parameter_release_min_fractional_improvement),
            0.0,
        )
        absolute_threshold_enabled = minimum_absolute_improvement > 0.0
        fractional_threshold_enabled = minimum_fractional_improvement > 0.0
        absolute_threshold_passed = (
            absolute_threshold_enabled
            and improvement >= minimum_absolute_improvement
        )
        fractional_threshold_passed = (
            fractional_threshold_enabled
            and fractional_improvement >= minimum_fractional_improvement
        )
        if absolute_threshold_enabled or fractional_threshold_enabled:
            wrms_threshold_passed = (
                improvement > 0.0
                and (absolute_threshold_passed or fractional_threshold_passed)
            )
        else:
            wrms_threshold_passed = improvement > 0.0
        evidence_threshold_passed = evidence_gain >= max(
            float(config.parameter_release_min_evidence_gain),
            0.0,
        )
        accepted = wrms_threshold_passed and evidence_threshold_passed
        records.append(
            {
                "block": [str(name).upper() for name in probe.get("block", ())],
                "best_branch_id": best_certification.branch_id,
                "best_weighted_rms_us": probe_wrms,
                "best_reduced_chi2": best_certification.reduced_chi2,
                "best_chi2": best_certification.chi2,
                "wrms_improvement_us": improvement,
                "fractional_improvement": fractional_improvement,
                "added_parameter_count": added_parameter_count,
                "bic_penalty": penalty,
                "likelihood_gain": likelihood_gain,
                "evidence_gain": evidence_gain,
                "absolute_wrms_threshold_passed": absolute_threshold_passed,
                "fractional_wrms_threshold_passed": fractional_threshold_passed,
                "wrms_threshold_passed": wrms_threshold_passed,
                "evidence_threshold_passed": evidence_threshold_passed,
                "accepted": accepted,
                "reason": "accepted" if accepted else "threshold-not-met",
            }
        )

    accepted_records = [record for record in records if bool(record.get("accepted"))]
    if not accepted_records:
        return {
            "selected": False,
            "reason": "no-probe-passed-threshold",
            "baseline_branch_id": baseline_certification.branch_id,
            "baseline_weighted_rms_us": baseline_wrms,
            "baseline_reduced_chi2": baseline_certification.reduced_chi2,
            "baseline_chi2": baseline_certification.chi2,
            "probes": records,
        }
    selected_record = sorted(
        accepted_records,
        key=lambda record: (
            -float(record.get("evidence_gain", float("-inf"))),
            -float(record.get("wrms_improvement_us", float("-inf"))),
            len(record.get("block", [])),
            str(record.get("block", [])),
        ),
    )[0]
    return {
        "selected": True,
        "reason": "selected",
        "baseline_branch_id": baseline_certification.branch_id,
        "baseline_weighted_rms_us": baseline_wrms,
        "baseline_reduced_chi2": baseline_certification.reduced_chi2,
        "baseline_chi2": baseline_certification.chi2,
        "selected_block": list(selected_record["block"]),
        "selected_branch_id": selected_record.get("best_branch_id"),
        "selected_weighted_rms_us": selected_record.get("best_weighted_rms_us"),
        "selected_reduced_chi2": selected_record.get("best_reduced_chi2"),
        "selected_evidence_gain": selected_record.get("evidence_gain"),
        "probes": records,
    }


def _with_certification_diagnostics(
    branch: ActiveTimingBranch,
    certification: BranchCertification,
) -> ActiveTimingBranch:
    diagnostics = dict(branch.diagnostics or {})
    diagnostics.pop("bridge_conditioned_covariance", None)
    diagnostics.pop("proposal_covariance", None)
    diagnostics["certified_offset_source"] = str(certification.backend)
    diagnostics["certified_reduced_chi2"] = certification.reduced_chi2
    diagnostics["certified_weighted_rms_us"] = certification.weighted_rms_us
    diagnostics["certified_covariance"] = np.asarray(certification.covariance, dtype=float)
    return replace(branch, diagnostics=diagnostics)


def _normalize_branch_to_certification(
    branch: ActiveTimingBranch,
    certification: BranchCertification,
    *,
    base_parameter_values: dict[str, float],
    parameter_scales: np.ndarray | None = None,
) -> ActiveTimingBranch:
    theta_names = tuple(str(name).upper() for name in branch.theta_names)
    proposal_offset = np.asarray(branch.theta_offset, dtype=float)
    offset = proposal_offset.copy()
    for index, name in enumerate(theta_names):
        if name in certification.fitted_theta and name in base_parameter_values:
            offset[index] = float(certification.fitted_theta[name]) - float(base_parameter_values[name])
    normalized = _with_certification_diagnostics(branch, certification)
    normalized_diagnostics = dict(normalized.diagnostics or {})
    normalized_diagnostics["certification_theta_offset"] = np.asarray(offset, dtype=float)
    normalized = replace(normalized, diagnostics=normalized_diagnostics)
    if parameter_scales is None:
        return replace(normalized, theta_offset=offset)

    scales = np.asarray(parameter_scales, dtype=float)
    if scales.shape != offset.shape or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite, positive, and match theta_offset.")
    proposal_parameter_log_prior = -0.5 * float(
        np.sum(np.square(proposal_offset / scales))
    )
    certified_parameter_log_prior = -0.5 * float(np.sum(np.square(offset / scales)))
    diagnostics = dict(normalized.diagnostics or {})
    diagnostics.update(
        {
            "proposal_parameter_log_prior": proposal_parameter_log_prior,
            "certified_parameter_log_prior": certified_parameter_log_prior,
            "certified_parameter_prior_delta": (
                certified_parameter_log_prior - proposal_parameter_log_prior
            ),
        }
    )
    return replace(
        normalized,
        theta_offset=offset,
        log_prior=(
            float(branch.log_prior)
            + certified_parameter_log_prior
            - proposal_parameter_log_prior
        ),
        diagnostics=diagnostics,
    )


def _rebase_anchor_branch(
    branch: ActiveTimingBranch,
    *,
    anchor_offset: np.ndarray,
    anchor_id: str,
    parent_branch_id: str,
    parent_wraps: Sequence[int] = (),
) -> ActiveTimingBranch:
    local_offset = np.asarray(branch.theta_offset, dtype=float)
    anchor = np.asarray(anchor_offset, dtype=float)
    if anchor.shape != local_offset.shape or np.any(~np.isfinite(anchor)):
        raise ValueError("anchor_offset must be finite and match branch.theta_offset.")
    diagnostics = dict(branch.diagnostics or {})
    local_certification_offset = np.asarray(
        diagnostics.get("certification_theta_offset", local_offset),
        dtype=float,
    )
    if local_certification_offset.shape != local_offset.shape:
        raise ValueError(
            "Anchor-local certification_theta_offset must match branch.theta_offset."
        )
    local_wraps = tuple(int(item) for item in branch.wraps)
    inherited_wraps = tuple(int(item) for item in parent_wraps)
    if inherited_wraps and len(inherited_wraps) != len(local_wraps):
        raise ValueError("parent_wraps must match the anchor-local wrap dimension.")
    global_wraps = tuple(
        local_wrap
        + (inherited_wraps[index] if inherited_wraps else 0)
        for index, local_wrap in enumerate(local_wraps)
    )
    diagnostics.update(
        {
            "proposal_family": "multi-anchor-reacquisition",
            "anchor_id": str(anchor_id),
            "anchor_parent_branch_id": str(parent_branch_id),
            "anchor_parent_wraps": inherited_wraps,
            "anchor_local_wraps": local_wraps,
            "anchor_global_wraps": global_wraps,
            "anchor_theta_offset": anchor,
            "anchor_local_theta_offset": local_offset,
            "anchor_local_certification_theta_offset": local_certification_offset,
            "certification_theta_offset": anchor + local_certification_offset,
        }
    )
    return replace(
        branch,
        wraps=global_wraps,
        theta_offset=anchor + local_offset,
        diagnostics=diagnostics,
    )


def _candidate_new_wraps(
    *,
    predicted_wrap: float,
    config: StagedBudgetedSearchConfig,
) -> list[int]:
    nearest = int(np.rint(float(predicted_wrap)))
    local_values = list(
        range(
            nearest - max(int(config.wrap_radius), 0),
            nearest + max(int(config.wrap_radius), 0) + 1,
        )
    )
    values: list[int] = [int(value) for value in local_values]
    exploration_radius = max(int(config.exploration_wrap_radius), 0)
    stride = max(int(config.exploration_stride), 1)
    for delta in range(-exploration_radius, exploration_radius + 1, stride):
        values.append(int(nearest + delta))
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _diagnostic_log_likelihood(
    *,
    predicted_wrap: float,
    chosen_wrap: int,
    config: StagedBudgetedSearchConfig,
) -> float:
    sigma = max(float(config.phase_sigma), 1.0e-12)
    residual = float(predicted_wrap) - float(chosen_wrap)
    return -0.5 * (residual / sigma) ** 2


def _expand_parent_branch(
    *,
    parent: ActiveTimingBranch,
    parent_rank: int,
    stage_index: int,
    gap_jacobian_row: np.ndarray,
    reference_phase_residual: float,
    config: StagedBudgetedSearchConfig,
) -> list[ActiveTimingBranch]:
    theta_offset = np.asarray(parent.theta_offset, dtype=float)
    jacobian_row = np.asarray(gap_jacobian_row, dtype=float)
    predicted_wrap = float(reference_phase_residual) + float(jacobian_row @ theta_offset)
    new_wraps = _candidate_new_wraps(predicted_wrap=predicted_wrap, config=config)
    parent_diagnostics = dict(parent.diagnostics or {})
    covariance, covariance_source = _predictive_covariance_from_diagnostics(
        parent_diagnostics,
        theta_offset.size,
    )
    conditioned_by_wrap: dict[int, _IntegerBridgeConditioning] = {}
    if covariance is not None:
        conditioned_by_wrap = {
            item.chosen_wrap: item
            for item in _condition_integer_bridges(
                mean=theta_offset,
                covariance=covariance,
                jacobian_row=jacobian_row,
                reference_phase_residual=reference_phase_residual,
                chosen_wraps=new_wraps,
                phase_sigma=config.phase_sigma,
            )
        }
    branches: list[ActiveTimingBranch] = []
    parent_family = _branch_family(parent)
    for candidate_rank, new_wrap in enumerate(new_wraps):
        wraps = tuple(list(parent.wraps) + [int(new_wrap)])
        bridge = conditioned_by_wrap.get(int(new_wrap))
        if bridge is None:
            child_theta_offset = theta_offset.copy()
            diagnostic_log_likelihood = _diagnostic_log_likelihood(
                predicted_wrap=predicted_wrap,
                chosen_wrap=int(new_wrap),
                config=config,
            )
        else:
            child_theta_offset = bridge.posterior_mean.copy()
            diagnostic_log_likelihood = bridge.log_likelihood
        wrap_penalty = sum(abs(item) for item in wraps) / max(float(config.wrap_prior_scale), 1.0e-12)
        offset_penalty = 0.5 * float(theta_offset @ theta_offset) / max(
            float(config.offset_prior_scale), 1.0e-30
        ) ** 2
        branch_id = f"stage_{stage_index:02d}_p{parent_rank:04d}_w{candidate_rank:04d}"
        diagnostics = dict(parent_diagnostics)
        diagnostics.update(
            {
                "stage": int(stage_index),
                "parent_branch_id": parent.branch_id,
                "parent_rank": int(parent_rank),
                "candidate_rank": int(candidate_rank),
                "proposal_family": (
                    "local-wrap"
                    if abs(int(new_wrap) - int(np.rint(predicted_wrap))) <= max(int(config.wrap_radius), 0)
                    else "exploration-wrap"
                ),
                "parent_family": parent_family,
                "predicted_new_wrap": float(predicted_wrap),
                "chosen_new_wrap": int(new_wrap),
                "new_wrap_residual": float(predicted_wrap) - float(new_wrap),
                "diagnostic_log_likelihood": float(diagnostic_log_likelihood),
                "parent_path_log_prior": float(parent.log_prior),
                "proposal_log_prior": (
                    float(diagnostic_log_likelihood)
                    - abs(int(new_wrap)) / max(float(config.wrap_prior_scale), 1.0e-12)
                ),
                "bridge_conditioning": (
                    "kalman" if bridge is not None else "legacy-no-covariance"
                ),
            }
        )
        if bridge is not None:
            diagnostics.update(
                {
                    "bridge_covariance_source": covariance_source,
                    "bridge_predictive_variance": bridge.predictive_variance,
                    "bridge_innovation": bridge.innovation,
                    "bridge_conditioned_covariance": bridge.posterior_covariance,
                }
            )
        branches.append(
            ActiveTimingBranch(
                branch_id=branch_id,
                structure_id=parent.structure_id,
                binary_model=parent.binary_model,
                wraps=wraps,
                theta_names=parent.theta_names,
                theta_offset=child_theta_offset,
                log_prior=float(parent.log_prior) + float(diagnostic_log_likelihood) - wrap_penalty - offset_penalty,
                proposal_source="staged-budgeted-bayesian-search",
                diagnostics=diagnostics,
            )
        )
    return branches


def _expand_parent_gap_block(
    *,
    parent: ActiveTimingBranch,
    parent_rank: int,
    stage_index: int,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    config: StagedBudgetedSearchConfig,
    local_phase_slope_jacobian: np.ndarray | None = None,
    reference_local_phase_slopes: np.ndarray | None = None,
    local_phase_slope_sigmas: np.ndarray | None = None,
) -> list[ActiveTimingBranch]:
    jacobian = np.asarray(gap_jacobian, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float).reshape(-1)
    if jacobian.ndim != 2 or reference.shape != (jacobian.shape[0],):
        raise ValueError("Gap geometry and reference residuals must have matching rows.")
    slope_inputs = (
        local_phase_slope_jacobian,
        reference_local_phase_slopes,
        local_phase_slope_sigmas,
    )
    if any(item is not None for item in slope_inputs) and not all(
        item is not None for item in slope_inputs
    ):
        raise ValueError(
            "Local phase-slope Jacobian, residuals, and sigmas must be supplied together."
        )
    first_new_gap = len(parent.wraps)
    if first_new_gap > jacobian.shape[0]:
        raise ValueError("Parent wrap dimension exceeds the target gap count.")
    if first_new_gap == jacobian.shape[0]:
        return [parent]

    frontier = [parent]
    beam_size = max(int(config.beam_size), 1)
    gap_indices = list(range(first_new_gap, jacobian.shape[0]))
    intermediate_beam_size = beam_size
    for block_gap_rank, gap_index in enumerate(gap_indices):
        expanded: list[ActiveTimingBranch] = []
        for frontier_rank, branch in enumerate(frontier):
            children = _expand_parent_branch(
                parent=branch,
                parent_rank=parent_rank * beam_size + frontier_rank,
                stage_index=stage_index,
                gap_jacobian_row=jacobian[gap_index],
                reference_phase_residual=float(reference[gap_index]),
                config=config,
            )
            for child_rank, child in enumerate(children):
                expanded.append(
                    replace(
                        child,
                        branch_id=(
                            f"stage_{stage_index:02d}_atomic_p{parent_rank:04d}_"
                            f"g{gap_index:04d}_{frontier_rank:04d}_{child_rank:04d}"
                        ),
                    )
                )
        branching_factor = max(
            int(math.ceil(len(expanded) / max(len(frontier), 1))),
            1,
        )
        intermediate_beam_size = max(
            intermediate_beam_size,
            beam_size * branching_factor,
        )
        is_final_gap = block_gap_rank == len(gap_indices) - 1
        active_beam_size = beam_size if is_final_gap else intermediate_beam_size
        if is_final_gap and len(gap_indices) > 1 and all(
            item is not None for item in slope_inputs
        ):
            slope_scored: list[ActiveTimingBranch] = []
            for branch in expanded:
                branch_diagnostics = dict(branch.diagnostics or {})
                covariance, covariance_source = _predictive_covariance_from_diagnostics(
                    branch_diagnostics,
                    len(branch.theta_offset),
                )
                conditioning_covariance = (
                    covariance
                    if covariance is not None
                    else np.zeros(
                        (len(branch.theta_offset), len(branch.theta_offset)),
                        dtype=float,
                    )
                )
                (
                    conditioned_offset,
                    conditioned_covariance,
                    slope_log_likelihood,
                    slope_innovation,
                ) = _condition_on_local_phase_slopes(
                    mean=np.asarray(branch.theta_offset, dtype=float),
                    covariance=conditioning_covariance,
                    parameter_scales=np.ones(len(branch.theta_offset), dtype=float),
                    local_phase_slope_jacobian=local_phase_slope_jacobian,
                    reference_local_phase_slopes=reference_local_phase_slopes,
                    local_phase_slope_sigmas=local_phase_slope_sigmas,
                )
                joint_score = float(branch.log_prior) + float(slope_log_likelihood)
                branch_diagnostics.update(
                    {
                        "atomic_slope_evidence_count": int(slope_innovation.size),
                        "atomic_slope_log_likelihood": float(slope_log_likelihood),
                        "atomic_slope_innovation": np.asarray(
                            slope_innovation,
                            dtype=float,
                        ),
                        "atomic_slope_conditioned_offset": np.asarray(
                            conditioned_offset,
                            dtype=float,
                        ),
                        "atomic_slope_covariance_source": covariance_source,
                        "atomic_joint_score": joint_score,
                    }
                )
                if covariance is not None:
                    branch_diagnostics["bridge_conditioned_covariance"] = (
                        conditioned_covariance
                    )
                slope_scored.append(
                    replace(
                        branch,
                        theta_offset=np.asarray(conditioned_offset, dtype=float),
                        log_prior=joint_score,
                        diagnostics=branch_diagnostics,
                    )
                )
            expanded = slope_scored
        frontier = sorted(
            expanded,
            key=lambda branch: (-float(branch.log_prior), branch.wraps, branch.branch_id),
        )[:active_beam_size]

    return [
        replace(
            branch,
            branch_id=f"stage_{stage_index:02d}_atomic_p{parent_rank:04d}_{rank:04d}",
            diagnostics={
                **(branch.diagnostics or {}),
                "parent_branch_id": parent.branch_id,
                "atomic_gap_indices": gap_indices,
                "atomic_gap_count": len(gap_indices),
            },
        )
        for rank, branch in enumerate(frontier)
    ]


def _select_certification_candidates(
    branches: list[ActiveTimingBranch],
    *,
    budget: int,
    config: StagedBudgetedSearchConfig,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    pseudo_certifications = [
        BranchCertification(
            branch_id=branch.branch_id,
            log_likelihood=float((branch.diagnostics or {}).get("diagnostic_log_likelihood", branch.log_prior)),
            fitted_theta={
                name: float(value)
                for name, value in zip(branch.theta_names, branch.theta_offset, strict=True)
            },
            covariance=np.eye(len(branch.theta_names), dtype=float),
            reduced_chi2=max(
                abs(float((branch.diagnostics or {}).get("new_wrap_residual", 0.0))),
                0.0,
            ),
            weighted_rms_us=max(
                abs(float((branch.diagnostics or {}).get("new_wrap_residual", 0.0))),
                0.0,
            ),
            backend="stage-diagnostic",
        )
        for branch in branches
    ]
    selected, diagnostics = select_budgeted_decision_branches(
        branches,
        pseudo_certifications,
        config=BudgetedDecisionConfig(
            budget=max(int(budget), 0),
            family_floor=_decision_family_floor(config, budget),
            portfolio_floor=int(config.decision_portfolio_floor),
            quality_weight=0.35,
            diagnostic_weight=float(config.decision_diagnostic_weight),
            diversity_weight=float(config.decision_diversity_weight),
            uncertainty_weight=float(config.decision_uncertainty_weight),
            prior_weight=float(config.decision_prior_weight),
            lookahead_weight=float(config.decision_lookahead_weight),
            prior_score_mode=str(config.decision_prior_score_mode),
            max_family_fraction=float(config.decision_max_family_fraction),
            basin_floor=int(config.decision_basin_floor),
            max_basin_fraction=float(config.decision_max_basin_fraction),
        ),
    )
    return selected, diagnostics


def _select_stage_survivors(
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    config: StagedBudgetedSearchConfig,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    selected, diagnostics = select_budgeted_decision_branches(
        list(branches),
        list(certifications),
        config=BudgetedDecisionConfig(
            budget=max(int(config.beam_size), 0),
            quality_floor=_decision_quality_floor(config, config.beam_size),
            family_floor=_decision_family_floor(config, config.beam_size),
            portfolio_floor=int(config.decision_portfolio_floor),
            quality_weight=float(config.decision_quality_weight),
            diagnostic_weight=float(config.decision_diagnostic_weight),
            diversity_weight=float(config.decision_diversity_weight),
            uncertainty_weight=float(config.decision_uncertainty_weight),
            prior_weight=float(config.decision_prior_weight),
            lookahead_weight=float(config.decision_lookahead_weight),
            prior_score_mode=str(config.decision_prior_score_mode),
            max_family_fraction=float(config.decision_max_family_fraction),
            basin_floor=int(config.decision_basin_floor),
            max_basin_fraction=float(config.decision_max_basin_fraction),
        ),
    )
    return selected, diagnostics


def _default_repair_block_groups(theta_names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    available = {str(name).upper() for name in theta_names}
    candidates = [
        ("F0", "F1"),
        ("PB", "TASC"),
        ("A1",),
        ("EPS1", "EPS2"),
        ("RAJ", "DECJ"),
        ("RAJ", "DECJ", "F0", "F1"),
        ("PB", "TASC", "A1", "EPS1", "EPS2"),
    ]
    return tuple(
        tuple(name for name in group if name in available)
        for group in candidates
        if any(name in available for name in group)
    )


def _repair_config(
    config: StagedBudgetedSearchConfig,
    theta_names: Sequence[str],
) -> AdaptiveResidualConfig:
    return AdaptiveResidualConfig(
        top_k=int(config.residual_repair_top_k),
        parent_top_k=int(config.residual_repair_parent_top_k),
        ridge=float(config.residual_repair_ridge),
        step_scales=tuple(float(item) for item in config.residual_repair_step_scales),
        eigen_step_scales=(-1.0, 1.0),
        max_eigen_directions=3,
        residual_noise_floor=float(config.residual_noise_floor),
        wrap_prior_scale=float(config.wrap_prior_scale),
        block_coordinate_groups=_default_repair_block_groups(theta_names),
        block_step_scales=tuple(float(item) for item in config.residual_repair_block_step_scales),
        enable_gap_wrap_resampling=True,
        max_gap_wrap_shift=int(config.residual_repair_max_gap_wrap_shift),
        gap_wrap_shift_scales=tuple(float(item) for item in config.residual_repair_gap_wrap_shift_scales),
        gap_wrap_top_k=int(config.residual_repair_gap_wrap_top_k),
        gap_wrap_min_abs_residual=0.05,
        gap_wrap_joint_ridge=float(config.residual_repair_ridge),
        gap_wrap_joint_scale=float(config.residual_repair_gap_wrap_joint_scale),
        combine_gap_wrap_with_continuous=True,
        decision_quality_weight=float(config.decision_quality_weight),
        decision_diagnostic_weight=float(config.decision_diagnostic_weight),
        decision_diversity_weight=float(config.decision_diversity_weight),
        decision_prior_weight=float(config.decision_prior_weight),
        decision_family_floor=_decision_family_floor(config, config.residual_repair_top_k),
        decision_max_family_fraction=float(config.decision_max_family_fraction),
    )


def _payload_to_fingerprint(
    branch_id: str,
    payload: StagedResidualRepairPayload,
) -> ClusterResidualFingerprint:
    return ClusterResidualFingerprint(
        branch_id=str(branch_id),
        cluster_ids=tuple(payload.cluster_ids),
        cluster_mjd=np.asarray(payload.cluster_mjd, dtype=float),
        median_phase_residual=np.asarray(payload.median_phase_residual, dtype=float),
        mad_phase_residual=np.asarray(payload.mad_phase_residual, dtype=float),
        toa_counts=np.asarray(payload.toa_counts, dtype=int),
        gap_phase_residual=np.asarray(payload.gap_phase_residual, dtype=float),
    )


def _generate_residual_repair_candidates(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    payloads: dict[str, StagedResidualRepairPayload],
    theta_names: Sequence[str],
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    if not bool(config.residual_repair_enabled):
        return [], {"enabled": False, "proposal_source": "adaptive-residual-bayesian-lattice"}
    fingerprints = [
        _payload_to_fingerprint(branch.branch_id, payloads[branch.branch_id])
        for branch in parent_branches
        if branch.branch_id in payloads
    ]
    if not fingerprints:
        return [], {
            "enabled": True,
            "proposal_source": "adaptive-residual-bayesian-lattice",
            "candidate_count": 0,
            "kept_count": 0,
            "reason": "no-fingerprints",
        }
    first_payload = payloads[fingerprints[0].branch_id]
    repaired, diagnostics = generate_residual_guided_branches(
        parent_branches=parent_branches,
        certifications=certifications,
        fingerprints=fingerprints,
        theta_names=theta_names,
        cluster_design_matrix=np.asarray(first_payload.cluster_design_matrix, dtype=float),
        gap_jacobian=np.asarray(first_payload.gap_jacobian, dtype=float),
        reference_phase_residuals=np.asarray(first_payload.reference_phase_residuals, dtype=float),
        parameter_scales=np.asarray(parameter_scales, dtype=float),
        config=_repair_config(config, theta_names),
    )
    relabelled: list[ActiveTimingBranch] = []
    for index, branch in enumerate(repaired):
        diagnostics_copy = dict(branch.diagnostics or {})
        diagnostics_copy["repair_source_branch_id"] = branch.branch_id
        relabelled.append(
            replace(
                branch,
                branch_id=f"repair_{index:04d}",
                proposal_source="staged-budgeted-bayesian-search",
                diagnostics=diagnostics_copy,
            )
        )
    return relabelled, diagnostics


def _gap_shift_values(
    residual_value: float,
    *,
    max_abs_shift: int,
    residual_floor: float,
) -> tuple[int, ...]:
    if not np.isfinite(float(residual_value)):
        return ()
    max_shift = max(int(max_abs_shift), 0)
    if max_shift <= 0:
        return ()
    residual = float(residual_value)
    magnitude = int(np.rint(abs(residual)))
    if abs(residual) >= float(residual_floor):
        magnitude = max(magnitude, 1)
    magnitude = min(max(magnitude, 0), max_shift)
    if magnitude == 0:
        return ()
    signed = int(np.sign(residual) or 1) * magnitude
    values = [signed]
    if magnitude > 1:
        values.append(int(np.sign(signed)) * (magnitude - 1))
    if magnitude < max_shift:
        values.append(int(np.sign(signed)) * (magnitude + 1))
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value == 0 or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped)


def _joint_offset_for_wraps(
    *,
    parent_offset: np.ndarray,
    target_wraps: tuple[int, ...],
    cluster_design_matrix: np.ndarray,
    median_phase_residual: np.ndarray,
    residual_scale: np.ndarray,
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    parameter_scales: np.ndarray,
    ridge: float,
    gap_scale: float,
) -> np.ndarray:
    parent = np.asarray(parent_offset, dtype=float)
    parameter_scale = np.maximum(np.asarray(parameter_scales, dtype=float), 1.0e-30)
    if parent.shape != parameter_scale.shape:
        raise ValueError("parent_offset length must match parameter_scales.")

    design_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []

    cluster_design = np.asarray(cluster_design_matrix, dtype=float)
    cluster_target = -np.asarray(median_phase_residual, dtype=float)
    cluster_scale = np.maximum(np.asarray(residual_scale, dtype=float), 1.0e-12)
    if cluster_design.size:
        if cluster_design.ndim != 2 or cluster_design.shape[1] != parent.size:
            raise ValueError("cluster_design_matrix shape must match parameter dimension.")
        if cluster_design.shape[0] != cluster_target.size or cluster_scale.shape != cluster_target.shape:
            raise ValueError("Cluster residual arrays must match cluster_design_matrix rows.")
        design_blocks.append(cluster_design / cluster_scale[:, None])
        target_blocks.append(cluster_target / cluster_scale)

    gap_design = np.asarray(gap_jacobian, dtype=float)
    target = np.asarray(target_wraps, dtype=float)
    reference = np.asarray(reference_phase_residuals, dtype=float)
    if gap_design.size:
        if gap_design.ndim != 2 or gap_design.shape[1] != parent.size:
            raise ValueError("gap_jacobian shape must match parameter dimension.")
        if target.shape != (gap_design.shape[0],) or reference.shape != target.shape:
            raise ValueError("target_wraps and reference_phase_residuals must match gap_jacobian rows.")
        current_prediction = reference + gap_design @ parent
        active_gap_scale = max(float(gap_scale), 1.0e-12)
        design_blocks.append(gap_design / active_gap_scale)
        target_blocks.append((target - current_prediction) / active_gap_scale)

    if not design_blocks:
        return np.zeros_like(parent)
    design = np.vstack(design_blocks)
    target_vector = np.concatenate(target_blocks)
    scaled_design = design * parameter_scale[None, :]
    precision = scaled_design.T @ scaled_design + np.eye(parent.size, dtype=float) * max(float(ridge), 0.0)
    rhs = scaled_design.T @ target_vector
    scaled_delta = np.linalg.pinv(precision) @ rhs
    return scaled_delta * parameter_scale


def _session_reconnect_shift_values(
    residual_value: float,
    config: StagedBudgetedSearchConfig,
) -> tuple[int, ...]:
    return _gap_shift_values(
        float(residual_value),
        max_abs_shift=int(config.session_reconnect_max_wrap_shift),
        residual_floor=float(config.session_reconnect_residual_floor),
    )


def _session_reconnect_noise(
    residual_scale: np.ndarray,
    indices: Sequence[int],
    config: StagedBudgetedSearchConfig,
) -> float:
    if not indices:
        return max(float(config.residual_noise_floor), 1.0e-12)
    selected = np.asarray(residual_scale, dtype=float)[np.asarray(indices, dtype=int)]
    finite = selected[np.isfinite(selected)]
    if finite.size == 0:
        return max(float(config.residual_noise_floor), 1.0e-12)
    return max(float(np.median(finite)), float(config.residual_noise_floor), 1.0e-12)


def _session_reconnect_evidence(
    *,
    residual_value: float,
    residual_scale: np.ndarray,
    target_indices: Sequence[int],
    shift: np.ndarray,
    config: StagedBudgetedSearchConfig,
) -> float:
    noise = _session_reconnect_noise(residual_scale, target_indices, config)
    strength = abs(float(residual_value)) / noise
    support = math.log1p(max(len(tuple(target_indices)), 1))
    shift_penalty = sum(abs(int(item)) for item in shift.tolist()) / max(
        float(config.session_reconnect_prior_scale),
        1.0e-12,
    )
    return float(strength + support - shift_penalty)


def _session_reconnect_suffix_moves(
    *,
    median_phase_residual: np.ndarray,
    residual_scale: np.ndarray,
    parent_wraps: tuple[int, ...],
    config: StagedBudgetedSearchConfig,
) -> list[dict[str, Any]]:
    cluster_count = int(median_phase_residual.size)
    gap_count = len(parent_wraps)
    moves: list[dict[str, Any]] = []
    if cluster_count != gap_count + 1 or gap_count <= 0:
        return moves
    for split_index in range(1, cluster_count):
        left = np.asarray(median_phase_residual[:split_index], dtype=float)
        right = np.asarray(median_phase_residual[split_index:], dtype=float)
        if left.size == 0 or right.size == 0:
            continue
        contrast = float(np.median(right) - np.median(left))
        if abs(contrast) < float(config.session_reconnect_residual_floor):
            continue
        target_indices = tuple(range(split_index, cluster_count))
        for shift_value in _session_reconnect_shift_values(contrast, config):
            shift = np.zeros(gap_count, dtype=int)
            shift[split_index - 1] = int(shift_value)
            moves.append(
                {
                    "mode": "suffix",
                    "shift": shift,
                    "target_indices": target_indices,
                    "residual_value": contrast,
                    "evidence": _session_reconnect_evidence(
                        residual_value=contrast,
                        residual_scale=residual_scale,
                        target_indices=target_indices,
                        shift=shift,
                        config=config,
                    ),
                }
            )
    return moves


def _session_reconnect_isolated_moves(
    *,
    median_phase_residual: np.ndarray,
    residual_scale: np.ndarray,
    parent_wraps: tuple[int, ...],
    config: StagedBudgetedSearchConfig,
) -> list[dict[str, Any]]:
    gap_count = len(parent_wraps)
    cluster_count = int(median_phase_residual.size)
    moves: list[dict[str, Any]] = []
    if cluster_count != gap_count + 1 or gap_count <= 0:
        return moves
    ranked_indices = sorted(
        range(cluster_count),
        key=lambda index: (-abs(float(median_phase_residual[index])), int(index)),
    )[: max(int(config.session_reconnect_cluster_top_k), 0)]
    for cluster_index in ranked_indices:
        residual = float(median_phase_residual[cluster_index])
        if abs(residual) < float(config.session_reconnect_residual_floor):
            continue
        for shift_value in _session_reconnect_shift_values(residual, config):
            shift = np.zeros(gap_count, dtype=int)
            if cluster_index == 0:
                shift[0] = -int(shift_value)
            elif cluster_index == gap_count:
                shift[cluster_index - 1] = int(shift_value)
            else:
                shift[cluster_index - 1] = int(shift_value)
                shift[cluster_index] = -int(shift_value)
            moves.append(
                {
                    "mode": "isolated-session",
                    "shift": shift,
                    "target_indices": (int(cluster_index),),
                    "residual_value": residual,
                    "evidence": _session_reconnect_evidence(
                        residual_value=residual,
                        residual_scale=residual_scale,
                        target_indices=(int(cluster_index),),
                        shift=shift,
                        config=config,
                    ),
                }
            )
    return moves


def _session_reconnect_moves(
    *,
    payload: StagedResidualRepairPayload,
    parent_wraps: tuple[int, ...],
    config: StagedBudgetedSearchConfig,
) -> list[dict[str, Any]]:
    median_phase_residual = np.asarray(payload.median_phase_residual, dtype=float)
    residual_scale = np.maximum(
        np.asarray(payload.mad_phase_residual, dtype=float),
        float(config.residual_noise_floor),
    )
    if median_phase_residual.shape != residual_scale.shape:
        return []
    moves = _session_reconnect_suffix_moves(
        median_phase_residual=median_phase_residual,
        residual_scale=residual_scale,
        parent_wraps=parent_wraps,
        config=config,
    )
    moves.extend(
        _session_reconnect_isolated_moves(
            median_phase_residual=median_phase_residual,
            residual_scale=residual_scale,
            parent_wraps=parent_wraps,
            config=config,
        )
    )
    moves.sort(
        key=lambda move: (
            -float(move["evidence"]),
            str(move["mode"]),
            tuple(int(item) for item in np.asarray(move["shift"], dtype=int).tolist()),
        )
    )
    return moves


def _generate_session_reconnect_candidates(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    payloads: dict[str, StagedResidualRepairPayload],
    theta_names: Sequence[str],
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    if not bool(config.session_reconnect_enabled):
        return [], {"enabled": False, "proposal_source": "bayesian-session-reconnect"}
    certification_by_id = {certification.branch_id: certification for certification in certifications}
    ranked_parents = [
        branch
        for branch in sorted(
            parent_branches,
            key=lambda branch: _score_certification(
                certification_by_id.get(
                    branch.branch_id,
                    BranchCertification(
                        branch_id=branch.branch_id,
                        log_likelihood=float("-inf"),
                        fitted_theta={},
                        covariance=np.eye(len(branch.theta_names), dtype=float),
                        reduced_chi2=None,
                        weighted_rms_us=None,
                        backend="missing",
                    ),
                )
            ),
        )
        if branch.branch_id in payloads
    ][: max(int(config.session_reconnect_parent_top_k), 0)]
    if not ranked_parents:
        return [], {
            "enabled": True,
            "proposal_source": "bayesian-session-reconnect",
            "candidate_count": 0,
            "kept_count": 0,
            "reason": "no-parents",
        }

    theta_names_tuple = tuple(str(name).upper() for name in theta_names)
    candidates: list[ActiveTimingBranch] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    move_count_by_mode: dict[str, int] = {}
    for parent_rank, parent in enumerate(ranked_parents):
        payload = payloads[parent.branch_id]
        parent_wraps = tuple(int(item) for item in parent.wraps)
        if not parent_wraps:
            continue
        residual_scale = np.maximum(
            np.asarray(payload.mad_phase_residual, dtype=float),
            float(config.residual_noise_floor),
        )
        moves = _session_reconnect_moves(
            payload=payload,
            parent_wraps=parent_wraps,
            config=config,
        )
        for move in moves:
            shift = np.asarray(move["shift"], dtype=int)
            if shift.shape != (len(parent_wraps),) or not np.any(shift != 0):
                continue
            target_wraps = tuple(
                int(wrap) + int(shift_value)
                for wrap, shift_value in zip(parent_wraps, shift.tolist(), strict=True)
            )
            offset_delta = _joint_offset_for_wraps(
                parent_offset=np.asarray(parent.theta_offset, dtype=float),
                target_wraps=target_wraps,
                cluster_design_matrix=np.asarray(payload.cluster_design_matrix, dtype=float),
                median_phase_residual=np.asarray(payload.median_phase_residual, dtype=float),
                residual_scale=residual_scale,
                gap_jacobian=np.asarray(payload.gap_jacobian, dtype=float),
                reference_phase_residuals=np.asarray(payload.reference_phase_residuals, dtype=float),
                parameter_scales=np.asarray(parameter_scales, dtype=float),
                ridge=float(config.residual_repair_ridge),
                gap_scale=float(config.session_reconnect_joint_scale),
            )
            theta_offset = np.asarray(parent.theta_offset, dtype=float) + offset_delta
            key = (
                target_wraps,
                tuple(float(f"{value:.18e}") for value in theta_offset.tolist()),
            )
            if key in seen:
                continue
            seen.add(key)
            predicted_wraps = (
                np.asarray(payload.reference_phase_residuals, dtype=float)
                + np.asarray(payload.gap_jacobian, dtype=float) @ theta_offset
            )
            wrap_residual = predicted_wraps - np.asarray(target_wraps, dtype=float)
            residual_norm = float(np.linalg.norm(wrap_residual))
            evidence = float(move["evidence"])
            diagnostic_log_likelihood = evidence - 0.5 * (
                residual_norm / max(float(config.session_reconnect_joint_scale), 1.0e-12)
            ) ** 2
            shift_l1 = sum(abs(int(item)) for item in shift.tolist())
            log_prior = (
                float(parent.log_prior)
                + evidence
                - shift_l1 / max(float(config.session_reconnect_prior_scale), 1.0e-12)
                - 0.5 * float(offset_delta @ offset_delta)
            )
            mode = str(move["mode"])
            move_count_by_mode[mode] = move_count_by_mode.get(mode, 0) + 1
            target_indices = tuple(int(index) for index in move["target_indices"])
            candidates.append(
                ActiveTimingBranch(
                    branch_id=f"sr_{len(candidates):04d}",
                    structure_id=parent.structure_id,
                    binary_model=parent.binary_model,
                    wraps=target_wraps,
                    theta_names=theta_names_tuple,
                    theta_offset=np.asarray(theta_offset, dtype=float),
                    log_prior=float(log_prior),
                    proposal_source="staged-budgeted-bayesian-search",
                    diagnostics={
                        "proposal_family": "session-reconnect",
                        "parent_branch_id": parent.branch_id,
                        "parent_rank": int(parent_rank),
                        "session_reconnect_mode": mode,
                        "session_reconnect_evidence": evidence,
                        "session_reconnect_residual_phase": float(move["residual_value"]),
                        "target_cluster_indices": list(target_indices),
                        "target_cluster_ids": [
                            str(payload.cluster_ids[index])
                            for index in target_indices
                            if 0 <= index < len(payload.cluster_ids)
                        ],
                        "gap_wrap_shift": [int(item) for item in shift.tolist()],
                        "gap_wrap_source_indices": [int(item) for item in np.flatnonzero(shift).tolist()],
                        "predicted_wraps": predicted_wraps,
                        "wrap_residual_norm": residual_norm,
                        "diagnostic_log_likelihood": float(diagnostic_log_likelihood),
                        "residual_guided_offset_delta": offset_delta,
                    },
                )
            )

    selected, diagnostics = select_budgeted_decision_branches(
        candidates,
        [
            BranchCertification(
                branch_id=branch.branch_id,
                log_likelihood=float((branch.diagnostics or {}).get("diagnostic_log_likelihood", branch.log_prior)),
                fitted_theta={
                    name: float(value)
                    for name, value in zip(branch.theta_names, branch.theta_offset, strict=True)
                },
                covariance=np.eye(len(branch.theta_names), dtype=float),
                reduced_chi2=float((branch.diagnostics or {}).get("wrap_residual_norm", 0.0)) ** 2,
                weighted_rms_us=float((branch.diagnostics or {}).get("wrap_residual_norm", 0.0)),
                backend="session-reconnect-diagnostic",
                log_evidence=float((branch.diagnostics or {}).get("diagnostic_log_likelihood", branch.log_prior)),
            )
            for branch in candidates
        ],
        config=BudgetedDecisionConfig(
            budget=max(int(config.session_reconnect_top_k), 0),
            family_floor=_decision_family_floor(config, config.session_reconnect_top_k),
            portfolio_floor=int(config.decision_portfolio_floor),
            quality_weight=0.20,
            diagnostic_weight=max(float(config.decision_diagnostic_weight), 1.0),
            diversity_weight=max(float(config.decision_diversity_weight), 0.75),
            prior_weight=float(config.decision_prior_weight),
            prior_score_mode=str(config.decision_prior_score_mode),
            max_family_fraction=1.0,
            basin_floor=int(config.decision_basin_floor),
            max_basin_fraction=float(config.decision_max_basin_fraction),
        ),
    )
    return list(selected), {
        "enabled": True,
        "proposal_source": "bayesian-session-reconnect",
        "candidate_count": len(candidates),
        "kept_count": len(selected),
        "parent_count": len(ranked_parents),
        "move_count_by_mode": move_count_by_mode,
        "decision": diagnostics,
    }


def _generate_uncertain_gap_resample_candidates(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    payloads: dict[str, StagedResidualRepairPayload],
    theta_names: Sequence[str],
    parameter_scales: np.ndarray,
    config: StagedBudgetedSearchConfig,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    if not bool(config.uncertainty_resample_enabled):
        return [], {"enabled": False, "proposal_source": "budgeted-uncertain-gap-resampler"}
    certification_by_id = {certification.branch_id: certification for certification in certifications}
    ranked_parents = [
        branch
        for branch in sorted(
            parent_branches,
            key=lambda branch: _score_certification(
                certification_by_id.get(
                    branch.branch_id,
                    BranchCertification(
                        branch_id=branch.branch_id,
                        log_likelihood=float("-inf"),
                        fitted_theta={},
                        covariance=np.eye(len(branch.theta_names), dtype=float),
                        reduced_chi2=None,
                        weighted_rms_us=None,
                        backend="missing",
                    ),
                )
            ),
        )
        if branch.branch_id in payloads
    ][: max(int(config.uncertainty_resample_parent_top_k), 0)]
    if not ranked_parents:
        return [], {
            "enabled": True,
            "proposal_source": "budgeted-uncertain-gap-resampler",
            "candidate_count": 0,
            "kept_count": 0,
            "reason": "no-parents",
        }

    theta_names_tuple = tuple(str(name).upper() for name in theta_names)
    candidates: list[ActiveTimingBranch] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    combination_candidate_count = 0
    single_candidate_count = 0
    for parent_rank, parent in enumerate(ranked_parents):
        payload = payloads[parent.branch_id]
        parent_wraps = tuple(int(item) for item in parent.wraps)
        if not parent_wraps:
            continue
        residual = np.asarray(payload.gap_phase_residual, dtype=float)
        if residual.shape != (len(parent_wraps),):
            continue
        ranked_gap_indices = [
            int(index)
            for index in sorted(
                range(len(residual)),
                key=lambda index: (-abs(float(residual[index])), int(index)),
            )[: max(int(config.uncertainty_resample_gap_top_k), 0)]
        ]
        gap_shift_options = {
            gap_index: _gap_shift_values(
                float(residual[gap_index]),
                max_abs_shift=int(config.uncertainty_resample_max_gap_shift),
                residual_floor=float(config.uncertainty_resample_residual_floor),
            )
            for gap_index in ranked_gap_indices
        }
        gap_shift_options = {
            gap_index: values
            for gap_index, values in gap_shift_options.items()
            if values
        }
        if not gap_shift_options:
            continue
        index_groups: list[tuple[int, ...]] = [(gap_index,) for gap_index in gap_shift_options]
        max_combination_size = min(
            max(int(config.uncertainty_resample_combination_size), 1),
            len(gap_shift_options),
        )
        for size in range(2, max_combination_size + 1):
            index_groups.extend(tuple(int(item) for item in group) for group in combinations(gap_shift_options, size))

        residual_scale = np.maximum(
            np.asarray(payload.mad_phase_residual, dtype=float),
            float(config.residual_noise_floor),
        )
        for group in index_groups:
            value_options = [gap_shift_options[gap_index] for gap_index in group]
            for values in product(*value_options):
                shift = np.zeros(len(parent_wraps), dtype=int)
                for gap_index, shift_value in zip(group, values, strict=True):
                    shift[int(gap_index)] = int(shift_value)
                target_wraps = tuple(
                    int(wrap) + int(shift_value)
                    for wrap, shift_value in zip(parent_wraps, shift.tolist(), strict=True)
                )
                offset_delta = _joint_offset_for_wraps(
                    parent_offset=np.asarray(parent.theta_offset, dtype=float),
                    target_wraps=target_wraps,
                    cluster_design_matrix=np.asarray(payload.cluster_design_matrix, dtype=float),
                    median_phase_residual=np.asarray(payload.median_phase_residual, dtype=float),
                    residual_scale=residual_scale,
                    gap_jacobian=np.asarray(payload.gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(payload.reference_phase_residuals, dtype=float),
                    parameter_scales=np.asarray(parameter_scales, dtype=float),
                    ridge=float(config.residual_repair_ridge),
                    gap_scale=float(config.uncertainty_resample_joint_scale),
                )
                theta_offset = np.asarray(parent.theta_offset, dtype=float) + offset_delta
                key = (
                    target_wraps,
                    tuple(float(f"{value:.18e}") for value in theta_offset.tolist()),
                )
                if key in seen:
                    continue
                seen.add(key)
                predicted_wraps = (
                    np.asarray(payload.reference_phase_residuals, dtype=float)
                    + np.asarray(payload.gap_jacobian, dtype=float) @ theta_offset
                )
                wrap_residual = predicted_wraps - np.asarray(target_wraps, dtype=float)
                residual_norm = float(np.linalg.norm(wrap_residual))
                shifted_abs_residual = float(np.sum(np.abs(residual[np.nonzero(shift)[0]])))
                combination_size = int(np.count_nonzero(shift))
                if combination_size <= 1:
                    single_candidate_count += 1
                else:
                    combination_candidate_count += 1
                log_prior = (
                    float(parent.log_prior)
                    + shifted_abs_residual
                    - 0.5 * (residual_norm / max(float(config.uncertainty_resample_joint_scale), 1.0e-12)) ** 2
                    - sum(abs(item) for item in target_wraps) / max(float(config.wrap_prior_scale), 1.0e-12)
                    - sum(abs(int(item)) for item in shift.tolist()) / max(float(config.wrap_prior_scale), 1.0e-12)
                )
                candidates.append(
                    ActiveTimingBranch(
                        branch_id=f"ugr_{len(candidates):04d}",
                        structure_id=parent.structure_id,
                        binary_model=parent.binary_model,
                        wraps=target_wraps,
                        theta_names=theta_names_tuple,
                        theta_offset=np.asarray(theta_offset, dtype=float),
                        log_prior=float(log_prior),
                        proposal_source="staged-budgeted-bayesian-search",
                        diagnostics={
                            "proposal_family": "uncertain-gap-resample",
                            "parent_branch_id": parent.branch_id,
                            "parent_rank": int(parent_rank),
                            "gap_wrap_shift": [int(item) for item in shift.tolist()],
                            "gap_wrap_source_indices": [int(item) for item in np.nonzero(shift)[0].tolist()],
                            "gap_residual_values": [float(residual[index]) for index in np.nonzero(shift)[0].tolist()],
                            "combination_size": combination_size,
                            "predicted_wraps": predicted_wraps,
                            "wrap_residual_norm": residual_norm,
                            "diagnostic_log_likelihood": float(log_prior),
                            "residual_guided_offset_delta": offset_delta,
                        },
                    )
                )

    selected, diagnostics = select_budgeted_decision_branches(
        candidates,
        [
            BranchCertification(
                branch_id=branch.branch_id,
                log_likelihood=float((branch.diagnostics or {}).get("diagnostic_log_likelihood", branch.log_prior)),
                fitted_theta={
                    name: float(value)
                    for name, value in zip(branch.theta_names, branch.theta_offset, strict=True)
                },
                covariance=np.eye(len(branch.theta_names), dtype=float),
                reduced_chi2=float((branch.diagnostics or {}).get("wrap_residual_norm", 0.0)) ** 2,
                weighted_rms_us=float((branch.diagnostics or {}).get("wrap_residual_norm", 0.0)),
                backend="uncertain-gap-diagnostic",
            )
            for branch in candidates
        ],
        config=BudgetedDecisionConfig(
            budget=max(int(config.uncertainty_resample_top_k), 0),
            family_floor=_decision_family_floor(config, config.uncertainty_resample_top_k),
            portfolio_floor=int(config.decision_portfolio_floor),
            quality_weight=0.35,
            diagnostic_weight=max(float(config.decision_diagnostic_weight), 1.0),
            diversity_weight=max(float(config.decision_diversity_weight), 0.75),
            prior_weight=float(config.decision_prior_weight),
            prior_score_mode=str(config.decision_prior_score_mode),
            max_family_fraction=1.0,
            basin_floor=int(config.decision_basin_floor),
            max_basin_fraction=float(config.decision_max_basin_fraction),
        ),
    )
    return list(selected), {
        "enabled": True,
        "proposal_source": "budgeted-uncertain-gap-resampler",
        "candidate_count": len(candidates),
        "kept_count": len(selected),
        "parent_count": len(ranked_parents),
        "single_candidate_count": int(single_candidate_count),
        "combination_candidate_count": int(combination_candidate_count),
        "decision": diagnostics,
    }


def _build_repair_payloads(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    theta_names: Sequence[str],
    gap_jacobian: np.ndarray,
    reference_phase_residuals: np.ndarray,
    clock_override_path: Path | None,
) -> dict[str, StagedResidualRepairPayload]:
    fingerprints, cluster_design_matrix, _ = build_cluster_residual_fingerprints(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        branches=branches,
        certifications=certifications,
        theta_names=theta_names,
        clock_override_path=clock_override_path,
    )
    return {
        fingerprint.branch_id: StagedResidualRepairPayload(
            cluster_ids=fingerprint.cluster_ids,
            cluster_mjd=np.asarray(fingerprint.cluster_mjd, dtype=float),
            median_phase_residual=np.asarray(fingerprint.median_phase_residual, dtype=float),
            mad_phase_residual=np.asarray(fingerprint.mad_phase_residual, dtype=float),
            toa_counts=np.asarray(fingerprint.toa_counts, dtype=int),
            gap_phase_residual=np.asarray(fingerprint.gap_phase_residual, dtype=float),
            cluster_design_matrix=np.asarray(cluster_design_matrix, dtype=float),
            gap_jacobian=np.asarray(gap_jacobian, dtype=float),
            reference_phase_residuals=np.asarray(reference_phase_residuals, dtype=float),
        )
        for fingerprint in fingerprints
    }


def _certify_branches(
    *,
    branches: Sequence[ActiveTimingBranch],
    certifier: Callable[[AliasManifoldCandidate], Any],
    max_workers: int = 1,
    parallel_backend: str = "thread",
) -> list[BranchCertification]:
    return certify_branches(
        branches=branches,
        certifier=certifier,
        max_workers=max_workers,
        parallel_backend=parallel_backend,
        capture_failures=True,
    )


def _session_lattice_reacquisition_candidates(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    theta_names: Sequence[str],
    parameter_scales: Sequence[float],
    certifications: Sequence[BranchCertification],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    is_final_stage: bool,
    config: StagedBudgetedSearchConfig,
    clock_override_path: Path | None,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    if not bool(config.session_lattice_enabled):
        return [], {"enabled": False, "reason": "disabled"}
    if bool(config.session_lattice_final_only) and not is_final_stage:
        return [], {"enabled": True, "reason": "waiting-for-full-data"}

    finite_quality = [
        float(certification.reduced_chi2)
        for certification in certifications
        if certification.reduced_chi2 is not None
        and np.isfinite(float(certification.reduced_chi2))
    ]
    best_quality = min(finite_quality, default=math.inf)
    quality_threshold = max(float(config.session_lattice_trigger_reduced_chi2), 0.0)
    if best_quality <= quality_threshold:
        return [], {
            "enabled": True,
            "reason": "quality-sufficient",
            "best_reduced_chi2": best_quality,
            "trigger_reduced_chi2": quality_threshold,
        }

    evaluator = PintSessionShapeLikelihood.from_files(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        parameter_names=theta_names,
        clock_override_path=clock_override_path,
    )
    branches, diagnostics = generate_multimode_session_lattice_branches(
        evaluator=evaluator,
        parameter_names=theta_names,
        parameter_scales=parameter_scales,
        certifications=certifications,
        structure_id=structure_id,
        binary_model=binary_model,
        stage_index=stage_index,
        config=MultimodeSessionLatticeConfig(
            max_modes=max(int(config.session_lattice_max_modes), 1),
            candidates_per_mode=max(
                int(config.session_lattice_candidates_per_mode),
                1,
            ),
            max_candidates=max(int(config.session_lattice_max_candidates), 1),
            max_visited_nodes=max(
                int(config.session_lattice_max_visited_nodes),
                1,
            ),
            decimal_precision=max(
                int(config.session_lattice_decimal_precision),
                32,
            ),
            finite_difference_fraction=max(
                float(config.session_lattice_finite_difference_fraction),
                1.0e-12,
            ),
            shape_optimization_enabled=bool(
                config.session_lattice_shape_optimization_enabled
            ),
            shape_optimization_max_modes=max(
                int(config.session_lattice_shape_optimization_max_modes),
                1,
            ),
            shape_optimization_trust_radius=max(
                float(config.session_lattice_shape_optimization_trust_radius),
                1.0e-12,
            ),
            shape_optimization_max_nfev=max(
                int(config.session_lattice_shape_optimization_max_nfev),
                1,
            ),
        ),
    )
    return branches, {
        "enabled": True,
        "reason": "quality-triggered",
        "best_reduced_chi2": (
            best_quality if np.isfinite(best_quality) else None
        ),
        "trigger_reduced_chi2": quality_threshold,
        **diagnostics,
    }


def _best_pair(
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
) -> tuple[ActiveTimingBranch | None, BranchCertification | None]:
    branch_by_id = {branch.branch_id: branch for branch in branches}
    ranked = sorted(
        [certification for certification in certifications if certification.branch_id in branch_by_id],
        key=_score_certification,
    )
    if not ranked:
        return None, None
    return branch_by_id[ranked[0].branch_id], ranked[0]


def _quality_passed(certification: BranchCertification | None, config: StagedBudgetedSearchConfig) -> bool:
    if certification is None:
        return False
    if config.stop_reduced_chi2 is not None:
        if certification.reduced_chi2 is None or float(certification.reduced_chi2) > float(config.stop_reduced_chi2):
            return False
    if config.stop_weighted_rms_us is not None:
        if certification.weighted_rms_us is None or float(certification.weighted_rms_us) > float(config.stop_weighted_rms_us):
            return False
    return config.stop_reduced_chi2 is not None or config.stop_weighted_rms_us is not None


def _select_diverse_certified_parents(
    *,
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    budget: int,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    target = max(int(budget), 0)
    certification_by_id = {
        certification.branch_id: certification
        for certification in certifications
    }
    ranked = [
        branch
        for branch in sorted(
            branches,
            key=lambda branch: _score_certification(
                certification_by_id.get(
                    branch.branch_id,
                    BranchCertification(
                        branch_id=branch.branch_id,
                        log_likelihood=float("-inf"),
                        fitted_theta={},
                        covariance=np.eye(len(branch.theta_names), dtype=float),
                        reduced_chi2=None,
                        weighted_rms_us=None,
                        backend="missing",
                    ),
                )
            ),
        )
        if branch.branch_id in certification_by_id
    ]
    candidate_basins = {tuple(int(item) for item in branch.wraps) for branch in ranked}
    if target <= 0:
        return [], {
            "budget": target,
            "candidate_count": len(ranked),
            "candidate_basin_count": len(candidate_basins),
            "selected_count": 0,
            "selected_basin_count": 0,
        }

    selected: list[ActiveTimingBranch] = []
    selected_ids: set[str] = set()
    selected_basins: set[tuple[int, ...]] = set()
    for branch in ranked:
        basin = tuple(int(item) for item in branch.wraps)
        if basin in selected_basins:
            continue
        selected.append(branch)
        selected_ids.add(branch.branch_id)
        selected_basins.add(basin)
        if len(selected) >= target:
            break
    if len(selected) < target:
        for branch in ranked:
            if branch.branch_id in selected_ids:
                continue
            selected.append(branch)
            selected_ids.add(branch.branch_id)
            selected_basins.add(tuple(int(item) for item in branch.wraps))
            if len(selected) >= target:
                break
    return selected, {
        "budget": target,
        "candidate_count": len(ranked),
        "candidate_basin_count": len(candidate_basins),
        "selected_count": len(selected),
        "selected_basin_count": len(selected_basins),
        "selected_branch_ids": [branch.branch_id for branch in selected],
        "selected_wraps": [list(branch.wraps) for branch in selected],
    }


def _select_anchor_parents(
    *,
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    budget: int,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    selected, diagnostics = _select_diverse_certified_parents(
        branches=branches,
        certifications=certifications,
        budget=budget,
    )
    return selected, {**diagnostics, "selection_strategy": "quality-then-distinct-wrap-basin"}


def _multi_anchor_round_continuation(
    *,
    round_index: int,
    max_rounds: int,
    quality_before: float | None,
    quality_after: float | None,
    quality_threshold: float,
    min_fractional_improvement: float,
) -> dict[str, Any]:
    before = (
        float(quality_before)
        if quality_before is not None and np.isfinite(float(quality_before))
        else None
    )
    after = (
        float(quality_after)
        if quality_after is not None and np.isfinite(float(quality_after))
        else None
    )
    improvement = before - after if before is not None and after is not None else None
    fractional_improvement = (
        improvement / max(abs(before), np.finfo(float).eps)
        if improvement is not None and before is not None
        else None
    )
    decision = {
        "continue": False,
        "reason": "unknown",
        "round_index": int(round_index),
        "max_rounds": max(int(max_rounds), 1),
        "quality_before": before,
        "quality_after": after,
        "quality_threshold": max(float(quality_threshold), 0.0),
        "improvement": improvement,
        "fractional_improvement": fractional_improvement,
        "min_fractional_improvement": max(float(min_fractional_improvement), 0.0),
    }
    if after is None:
        return {**decision, "reason": "no-finite-quality"}
    if after <= decision["quality_threshold"]:
        return {**decision, "reason": "quality-sufficient"}
    if int(round_index) >= decision["max_rounds"]:
        return {**decision, "reason": "max-rounds-reached"}
    if (
        fractional_improvement is None
        or fractional_improvement < decision["min_fractional_improvement"]
    ):
        return {**decision, "reason": "improvement-stalled"}
    return {**decision, "continue": True, "reason": "material-improvement"}


def _run_multi_anchor_round_controller(
    *,
    parent_branches: Sequence[ActiveTimingBranch],
    parent_certifications: Sequence[BranchCertification],
    initial_quality: float | None,
    quality_threshold: float,
    max_rounds: int,
    min_fractional_improvement: float,
    run_round: Callable[
        [int, Sequence[ActiveTimingBranch], Sequence[BranchCertification]],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    round_parents = list(parent_branches)
    round_parent_certifications = list(parent_certifications)
    best_quality = (
        float(initial_quality)
        if initial_quality is not None and np.isfinite(float(initial_quality))
        else None
    )
    all_candidates: list[ActiveTimingBranch] = []
    all_selected: list[ActiveTimingBranch] = []
    all_certifications: list[BranchCertification] = []
    all_normalized: list[ActiveTimingBranch] = []
    round_records: list[dict[str, Any]] = []

    for round_index in range(1, max(int(max_rounds), 1) + 1):
        result = run_round(
            round_index,
            round_parents,
            round_parent_certifications,
        )
        candidates = list(result["candidates"])
        selected = list(result["selected_branches"])
        certifications = list(result["certifications"])
        normalized = list(result["normalized_branches"])
        finite_round_quality = [
            float(certification.reduced_chi2)
            for certification in certifications
            if certification.reduced_chi2 is not None
            and np.isfinite(float(certification.reduced_chi2))
        ]
        quality_before = best_quality
        if finite_round_quality:
            round_quality = min(finite_round_quality)
            best_quality = (
                round_quality
                if best_quality is None
                else min(best_quality, round_quality)
            )
        continuation = _multi_anchor_round_continuation(
            round_index=round_index,
            max_rounds=max_rounds,
            quality_before=quality_before,
            quality_after=best_quality,
            quality_threshold=quality_threshold,
            min_fractional_improvement=min_fractional_improvement,
        )
        all_candidates.extend(candidates)
        all_selected.extend(selected)
        all_certifications.extend(certifications)
        all_normalized.extend(normalized)
        round_records.append(
            {
                **dict(result.get("diagnostics") or {}),
                "generated_candidate_count": len(candidates),
                "selected_candidate_count": len(selected),
                "certification_count": len(certifications),
                "continuation": continuation,
            }
        )
        if not continuation["continue"]:
            break
        if not normalized or len(normalized) != len(certifications):
            round_records[-1]["continuation"] = {
                **continuation,
                "continue": False,
                "reason": "no-round-posterior",
            }
            break
        round_parents = normalized
        round_parent_certifications = certifications

    return {
        "candidates": all_candidates,
        "selected_branches": all_selected,
        "certifications": all_certifications,
        "normalized_branches": all_normalized,
        "rounds": round_records,
        "completed_rounds": len(round_records),
        "best_quality": best_quality,
    }


def _select_multi_anchor_certification_candidates(
    candidates: Sequence[ActiveTimingBranch],
    *,
    budget: int,
    config: StagedBudgetedSearchConfig,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    target = max(int(budget), 0)
    candidate_list = list(candidates)
    groups: dict[str, list[ActiveTimingBranch]] = {}
    for branch in candidate_list:
        anchor_id = str((branch.diagnostics or {}).get("anchor_id", "unknown"))
        groups.setdefault(anchor_id, []).append(branch)
    selected: list[ActiveTimingBranch] = []
    selected_ids: set[str] = set()
    anchor_selected_count = {anchor_id: 0 for anchor_id in sorted(groups)}
    if target > 0:
        for anchor_id in sorted(groups):
            floor_candidates, _ = _select_certification_candidates(
                groups[anchor_id],
                budget=1,
                config=config,
            )
            if not floor_candidates:
                continue
            selected.append(floor_candidates[0])
            selected_ids.add(floor_candidates[0].branch_id)
            anchor_selected_count[anchor_id] = 1
            if len(selected) >= target:
                break
    remaining = [branch for branch in candidate_list if branch.branch_id not in selected_ids]
    if len(selected) < target and remaining:
        fill_candidates, fill_diagnostics = _select_certification_candidates(
            remaining,
            budget=target - len(selected),
            config=config,
        )
        for branch in fill_candidates:
            if branch.branch_id in selected_ids:
                continue
            selected.append(branch)
            selected_ids.add(branch.branch_id)
            anchor_id = str((branch.diagnostics or {}).get("anchor_id", "unknown"))
            anchor_selected_count[anchor_id] = anchor_selected_count.get(anchor_id, 0) + 1
            if len(selected) >= target:
                break
    return selected, {
        "budget": target,
        "candidate_count": len(candidate_list),
        "anchor_count": len(groups),
        "anchor_candidate_count": {
            anchor_id: len(items) for anchor_id, items in sorted(groups.items())
        },
        "anchor_selected_count": anchor_selected_count,
        "selected_count": len(selected),
        "selected_branch_ids": [branch.branch_id for branch in selected],
    }


def _generate_multi_anchor_reacquisition_candidates(
    *,
    search_par_path: Path,
    prefix: PrefixTimingInputs,
    stage_dir: Path,
    stage_index: int,
    parent_branches: Sequence[ActiveTimingBranch],
    parent_certifications: Sequence[BranchCertification],
    theta_names: Sequence[str],
    structure_id: str,
    binary_model: str | None,
    config: StagedBudgetedSearchConfig,
    clock_override_path: Path | None,
    slope_sigma_scale: float,
    round_index: int = 1,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    if not bool(config.adaptive_multi_anchor_reacquisition_enabled):
        return [], {"enabled": False, "reason": "disabled"}
    selected_parents, parent_selection = _select_anchor_parents(
        branches=parent_branches,
        certifications=parent_certifications,
        budget=int(config.adaptive_multi_anchor_parent_top_k),
    )
    if not selected_parents:
        return [], {
            "enabled": True,
            "reason": "no-certified-anchor-parents",
            "parent_selection": parent_selection,
        }

    target = max(int(config.adaptive_multi_anchor_candidate_top_k), 1)
    per_anchor_top_k = max(int(math.ceil(target / len(selected_parents))), 1)
    active_theta_names = tuple(str(name).upper() for name in theta_names)
    anchor_config = replace(
        config,
        stage_lattice_refresh_enabled=True,
        stage_lattice_refresh_top_k=per_anchor_top_k,
        phase_slope_sigma_scale=float(slope_sigma_scale),
    )
    certification_by_id = {
        certification.branch_id: certification
        for certification in parent_certifications
    }
    candidates: list[ActiveTimingBranch] = []
    anchor_records: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
    resolved_round_index = max(int(round_index), 1)
    root = Path(stage_dir) / "multi_anchor_reacquisition"
    if resolved_round_index > 1:
        root = root / f"round_{resolved_round_index:02d}"
    for anchor_index, parent in enumerate(selected_parents):
        anchor_id = f"anchor_{anchor_index:02d}"
        branch_anchor_id = (
            anchor_id
            if resolved_round_index == 1
            else f"anchor_r{resolved_round_index:02d}_{anchor_index:02d}"
        )
        anchor_dir = root / anchor_id
        anchor_offset = np.asarray(parent.theta_offset, dtype=float)
        anchor_par_path = _write_anchor_model_parfile(
            path=anchor_dir / "anchor.par",
            seed_par_path=search_par_path,
            theta_names=active_theta_names,
            anchor_offset=anchor_offset,
        )
        anchor_tim_path, inherited_wraps = _write_anchor_pulse_number_reference(
            path=anchor_dir / "anchor.tim",
            anchor_par_path=anchor_par_path,
            tim_path=prefix.tim_path,
            cluster_manifest_path=prefix.cluster_manifest_path,
            parent_wraps=parent.wraps,
            clock_override_path=clock_override_path,
        )
        anchor_geometry = build_pint_alias_geometry(
            par_path=anchor_par_path,
            tim_path=anchor_tim_path,
            cluster_manifest_path=prefix.cluster_manifest_path,
            parameter_names=active_theta_names,
            clock_override_path=clock_override_path,
        )
        (
            anchor_slope_jacobian,
            anchor_slope_reference,
            anchor_slope_sigmas,
        ) = _phase_slope_inputs(
            getattr(anchor_geometry, "local_phase_slope_evidence", None),
            anchor_config,
        )
        anchor_parameter_scales = _fit_prior_parameter_scales(
            anchor_par_path,
            active_theta_names,
            formal_uncertainty_scale=float(config.formal_uncertainty_scale),
            physical_width_fraction=float(config.physical_prior_width_fraction),
        )
        local_candidates, generation = _stage_lattice_refresh_candidates(
            seed_branch=_initial_branch(
                structure_id=structure_id,
                binary_model=binary_model,
                theta_names=active_theta_names,
            ),
            stage_index=stage_index,
            gap_jacobian=np.asarray(anchor_geometry.gap_jacobian, dtype=float),
            reference_phase_residuals=np.asarray(
                anchor_geometry.reference_phase_residuals,
                dtype=float,
            ),
            parameter_scales=anchor_parameter_scales,
            config=anchor_config,
            local_phase_slope_jacobian=anchor_slope_jacobian,
            reference_local_phase_slopes=anchor_slope_reference,
            local_phase_slope_sigmas=anchor_slope_sigmas,
            branch_suffix=f"_{anchor_id}",
            slope_sigma_scale=float(slope_sigma_scale),
        )
        kept_count = 0
        for local_rank, local_branch in enumerate(local_candidates):
            rebased = _rebase_anchor_branch(
                local_branch,
                anchor_offset=anchor_offset,
                anchor_id=anchor_id,
                parent_branch_id=parent.branch_id,
                parent_wraps=inherited_wraps,
            )
            diagnostics = dict(rebased.diagnostics or {})
            parent_certification = certification_by_id.get(parent.branch_id)
            diagnostics.update(
                {
                    "stage": int(stage_index),
                    "anchor_round": resolved_round_index,
                    "anchor_rank": int(anchor_index),
                    "anchor_local_rank": int(local_rank),
                    "anchor_parent_reduced_chi2": (
                        parent_certification.reduced_chi2
                        if parent_certification is not None
                        else None
                    ),
                    "anchor_parent_weighted_rms_us": (
                        parent_certification.weighted_rms_us
                        if parent_certification is not None
                        else None
                    ),
                    "anchor_par_path": anchor_par_path,
                    "anchor_tim_path": anchor_tim_path,
                }
            )
            rebased = replace(
                rebased,
                branch_id=f"stage_{stage_index:02d}_{branch_anchor_id}_{local_rank:04d}",
                proposal_source="staged-budgeted-bayesian-search",
                diagnostics=diagnostics,
            )
            certification_offset = np.asarray(
                diagnostics["certification_theta_offset"],
                dtype=float,
            )
            key = (
                rebased.wraps,
                tuple(float(f"{value:.18e}") for value in certification_offset),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(rebased)
            kept_count += 1
        anchor_records.append(
            {
                "anchor_id": anchor_id,
                "parent_branch_id": parent.branch_id,
                "parent_wraps": list(parent.wraps),
                "inherited_wraps": list(inherited_wraps),
                "anchor_theta_offset": anchor_offset,
                "anchor_par_path": anchor_par_path,
                "anchor_tim_path": anchor_tim_path,
                "generated_candidate_count": len(local_candidates),
                "kept_candidate_count": kept_count,
                "generation": generation,
            }
        )
    return candidates, {
        "enabled": True,
        "reason": "generated",
        "round_index": resolved_round_index,
        "anchor_count": len(selected_parents),
        "candidate_count": len(candidates),
        "candidate_target": target,
        "per_anchor_top_k": per_anchor_top_k,
        "slope_sigma_scale": float(slope_sigma_scale),
        "parent_selection": parent_selection,
        "anchors": anchor_records,
    }


def _select_parameter_release_parents(
    *,
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    budget: int,
) -> tuple[list[ActiveTimingBranch], dict[str, Any]]:
    return _select_diverse_certified_parents(
        branches=branches,
        certifications=certifications,
        budget=budget,
    )


def _run_parameter_release_probe(
    *,
    par_path: Path,
    prefix: PrefixTimingInputs,
    stage_dir: Path,
    stage_index: int,
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    theta_names: Sequence[str],
    fit_parameter_names: Sequence[str] | None,
    config: StagedBudgetedSearchConfig,
    clock_override_path: Path | None,
) -> dict[str, Any]:
    if not bool(config.parameter_release_enabled):
        return {"selected": False, "reason": "disabled"}
    if int(stage_index) < int(config.parameter_release_min_cluster_count):
        return {
            "selected": False,
            "reason": "before-min-cluster-count",
            "min_cluster_count": int(config.parameter_release_min_cluster_count),
        }
    baseline_branch, baseline_certification = _best_pair(branches, certifications)
    if baseline_branch is None or baseline_certification is None:
        return {"selected": False, "reason": "no-baseline"}
    if _quality_passed(baseline_certification, config):
        return {"selected": False, "reason": "quality-already-passed"}

    ranked_parents, parent_selection = _select_parameter_release_parents(
        branches=branches,
        certifications=certifications,
        budget=int(config.parameter_release_probe_top_k),
    )
    if not ranked_parents:
        return {"selected": False, "reason": "no-probe-parents"}

    release_dir = stage_dir / "parameter_release"
    blocks = _normalise_parameter_release_blocks(config.parameter_release_candidate_blocks)
    current_theta_names = tuple(str(name).upper() for name in theta_names)
    current_fit_names = (
        None
        if fit_parameter_names is None
        else tuple(str(name).upper() for name in fit_parameter_names)
    )
    probes_for_selection: list[dict[str, Any]] = []
    payload_by_block: dict[tuple[str, ...], dict[str, Any]] = {}
    max_block_size = max(int(config.parameter_release_max_block_size), 0)
    _write_stage_progress(
        stage_dir,
        "parameter-release-probe-start",
        {
            "stage": int(stage_index),
            "parent_count": len(ranked_parents),
            "parent_selection": parent_selection,
            "blocks": [list(block) for block in blocks],
            "theta_names": list(current_theta_names),
            "fit_parameter_names": list(current_fit_names) if current_fit_names is not None else None,
        },
    )

    for block in blocks:
        if max_block_size and len(block) > max_block_size:
            probes_for_selection.append(
                {
                    "block": block,
                    "added_parameter_count": len(block),
                    "status": "skipped",
                    "reason": "block-too-large",
                }
            )
            continue
        expanded_theta_names = _augment_theta_names(current_theta_names, block)
        expanded_fit_names = _augment_fit_parameter_names(current_fit_names, block)
        added_parameter_count = len(set(expanded_theta_names) - set(current_theta_names))
        fit_added_count = (
            0
            if current_fit_names is None or expanded_fit_names is None
            else len(set(expanded_fit_names) - set(current_fit_names))
        )
        if added_parameter_count <= 0 and fit_added_count <= 0:
            probes_for_selection.append(
                {
                    "block": block,
                    "added_parameter_count": 0,
                    "status": "skipped",
                    "reason": "already-active",
                }
            )
            continue
        attempt_name = "_".join(block).lower() or "none"
        attempt_dir = release_dir / "attempts" / attempt_name
        attempt_audit_path = attempt_dir / "audit.json"
        _write_json_atomic(
            attempt_audit_path,
            {
                "attempt_id": f"parameter-release-{attempt_name}",
                "block": list(block),
                "status": "started",
                "stage": int(stage_index),
                "parent_branch_ids": [branch.branch_id for branch in ranked_parents],
                "fit_parameter_names": (
                    list(expanded_fit_names)
                    if expanded_fit_names is not None
                    else None
                ),
            },
        )
        try:
            probe_par_path, par_diagnostics = _write_parameter_release_par(
                par_path=par_path,
                parameter_names=block,
                output_dir=release_dir,
            )
            probe_branches = [
                _augment_branch_parameters(
                    branch,
                    theta_names=expanded_theta_names,
                    branch_id=f"stage_{stage_index:02d}_param_{'_'.join(block).lower()}_{index:04d}",
                    diagnostics={
                        "proposal_family": "parameter-release-probe",
                        "parameter_release_block": list(block),
                        "parameter_release_parent_branch_id": branch.branch_id,
                        "parameter_release_parent_rank": int(index),
                    },
                )
                for index, branch in enumerate(ranked_parents)
            ]
            probe_certifier = build_alias_branch_certifier(
                par_path=probe_par_path,
                tim_path=prefix.tim_path,
                cluster_manifest_path=prefix.cluster_manifest_path,
                parameter_names=expanded_theta_names,
                fit_parameter_names=expanded_fit_names,
                maxiter=int(config.maxiter),
                clock_override_path=clock_override_path,
            )
            probe_certifications = _certify_branches(
                branches=probe_branches,
                certifier=probe_certifier,
                max_workers=int(config.certify_parallel_workers),
                parallel_backend=config.certify_parallel_backend,
            )
            probe_base_values = _base_parameter_values(probe_par_path, expanded_theta_names)
            normalized_probe_branches = [
                _normalize_branch_to_certification(
                    branch,
                    certification,
                    base_parameter_values=probe_base_values,
                )
                for branch, certification in zip(probe_branches, probe_certifications, strict=True)
            ]
            best_probe_branch, best_probe_certification = _best_pair(
                normalized_probe_branches,
                probe_certifications,
            )
            _write_json_atomic(attempt_dir / "candidates.json", normalized_probe_branches)
            _write_json_atomic(attempt_dir / "certifications.json", probe_certifications)
            best_probe_artifact = None
            if best_probe_branch is not None and best_probe_certification is not None:
                best_probe_artifact = _write_parameter_release_probe_artifacts(
                    release_dir=release_dir,
                    block=block,
                    seed_par_path=probe_par_path,
                    fit_parameter_names=expanded_fit_names,
                    branch=best_probe_branch,
                    certification=best_probe_certification,
                    tim_path=prefix.tim_path,
                    clock_override_path=clock_override_path,
                )
            else:
                _write_json_atomic(
                    attempt_audit_path,
                    {
                        **_load_json(attempt_audit_path),
                        "status": "no-best-certification",
                    },
                )
            probe_record = {
                "block": block,
                "added_parameter_count": max(added_parameter_count, fit_added_count),
                "status": "certified",
                "par_diagnostics": par_diagnostics,
                "best_certification": best_probe_certification,
                "best_branch_id": best_probe_branch.branch_id if best_probe_branch is not None else None,
                "best_artifact": best_probe_artifact,
            }
            probes_for_selection.append(probe_record)
            payload_by_block[block] = {
                "par_path": probe_par_path,
                "theta_names": expanded_theta_names,
                "fit_parameter_names": expanded_fit_names,
                "base_values": probe_base_values,
                "parameter_scales": _fit_prior_parameter_scales(
                    probe_par_path,
                    expanded_theta_names,
                    formal_uncertainty_scale=float(config.formal_uncertainty_scale),
                    physical_width_fraction=float(config.physical_prior_width_fraction),
                ),
                "branches": normalized_probe_branches,
                "certifications": probe_certifications,
                "par_diagnostics": par_diagnostics,
            }
            _write_stage_progress(
                stage_dir,
                "parameter-release-probe-block-ready",
                {
                    "stage": int(stage_index),
                    "block": list(block),
                    "candidate_count": len(probe_branches),
                    "best_branch_id": best_probe_branch.branch_id if best_probe_branch is not None else None,
                    "best_weighted_rms_us": (
                        best_probe_certification.weighted_rms_us
                        if best_probe_certification is not None
                        else None
                    ),
                    "best_reduced_chi2": (
                        best_probe_certification.reduced_chi2
                        if best_probe_certification is not None
                        else None
                    ),
                    "par_diagnostics": par_diagnostics,
                },
            )
        except Exception as exc:  # pragma: no cover - exercised by runtime PINT/model failures
            _write_json_atomic(
                attempt_audit_path,
                {
                    **(_load_json(attempt_audit_path) if attempt_audit_path.exists() else {}),
                    "attempt_id": f"parameter-release-{attempt_name}",
                    "block": list(block),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            probes_for_selection.append(
                {
                    "block": block,
                    "added_parameter_count": len(block),
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_stage_progress(
                stage_dir,
                "parameter-release-probe-block-failed",
                {
                    "stage": int(stage_index),
                    "block": list(block),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    decision = _select_parameter_release_probe(
        baseline_certification=baseline_certification,
        probes=probes_for_selection,
        toa_count=_count_tim_toa_lines(prefix.tim_path),
        config=config,
    )
    selected_block = tuple(str(name).upper() for name in decision.get("selected_block", []))
    selected_payload = payload_by_block.get(selected_block)
    if bool(decision.get("selected")) and selected_payload is not None:
        decision = {
            **decision,
            "parent_selection": parent_selection,
            "selected_par_path": selected_payload["par_path"],
            "selected_theta_names": list(selected_payload["theta_names"]),
            "selected_fit_parameter_names": (
                list(selected_payload["fit_parameter_names"])
                if selected_payload["fit_parameter_names"] is not None
                else None
            ),
            "selected_par_diagnostics": selected_payload["par_diagnostics"],
        }
        _write_stage_progress(
            stage_dir,
            "parameter-release-selected",
            {
                "stage": int(stage_index),
                "decision": decision,
            },
        )
        return {
            "selected": True,
            "decision": decision,
            **selected_payload,
        }
    _write_stage_progress(
        stage_dir,
        "parameter-release-not-selected",
        {
            "stage": int(stage_index),
            "decision": decision,
        },
    )
    return {
        "selected": False,
        "decision": {**decision, "parent_selection": parent_selection},
    }


def _stage_record(
    *,
    stage_index: int,
    cluster_count: int,
    candidates: Sequence[ActiveTimingBranch],
    certified: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
    survivors: Sequence[ActiveTimingBranch],
    certification_decision: dict[str, Any],
    survivor_decision: dict[str, Any],
    repair_decision: dict[str, Any] | None = None,
    parameter_release_decision: dict[str, Any] | None = None,
    active_par_path: Path | None = None,
    fit_parameter_names: Sequence[str] | None = None,
    best_par_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best_branch, best_certification = _best_pair(survivors, certifications)
    return {
        "stage": int(stage_index),
        "cluster_count": int(cluster_count),
        "gap_count": max(int(cluster_count) - 1, 0),
        "candidate_count": len(candidates),
        "certified_count": len(certified),
        "survivor_count": len(survivors),
        "best_branch_id": best_branch.branch_id if best_branch is not None else None,
        "best_reduced_chi2": best_certification.reduced_chi2 if best_certification is not None else None,
        "best_weighted_rms_us": best_certification.weighted_rms_us if best_certification is not None else None,
        "best_wraps": list(best_branch.wraps) if best_branch is not None else None,
        "certification_decision": certification_decision,
        "survivor_decision": survivor_decision,
        "repair_decision": repair_decision or {},
        "parameter_release_decision": parameter_release_decision or {},
        "active_par_path": active_par_path,
        "fit_parameter_names": (
            [str(name).upper() for name in fit_parameter_names]
            if fit_parameter_names is not None
            else None
        ),
        "best_par_artifact": best_par_artifact,
    }


def _resolve_initial_fit_flag_policy(
    *,
    par_path: Path,
    theta_names: Sequence[str] | str,
    fit_parameter_names: Sequence[str] | str | None,
    config: StagedBudgetedSearchConfig,
) -> tuple[list[str], list[str]]:
    if bool(config.parameter_release_enabled):
        release_blocks = _normalise_parameter_release_blocks(
            config.parameter_release_candidate_blocks
        )
        unsupported = sorted(
            {
                name
                for block in release_blocks
                for name in block
                if name != "F2"
            }
        )
        if unsupported:
            raise ValueError(
                "APOLLO automatic parameter release currently supports only F2; "
                "unsupported parameters: " + ", ".join(unsupported)
            )
    initial_fit_flags = [
        str(name).upper() for name in extract_fit_flagged_parameters(Path(par_path))
    ]

    def resolve_requested(
        value: Sequence[str] | str | None,
        *,
        label: str,
    ) -> list[str]:
        if isinstance(value, str):
            if value.lower() != "fit_flags":
                raise ValueError(f"Unsupported {label} shorthand: {value}")
            requested = list(initial_fit_flags)
        elif value is None:
            raise ValueError(
                f"APOLLO {label} must equal the input par fit flags; None is not allowed."
            )
        else:
            requested = [str(name).upper() for name in value]
        if requested != initial_fit_flags:
            raise ValueError(
                f"APOLLO {label} must exactly equal the input par fit flags: "
                + ", ".join(initial_fit_flags)
            )
        return requested

    return (
        resolve_requested(theta_names, label="theta_names"),
        resolve_requested(fit_parameter_names, label="fit_parameter_names"),
    )


def run_staged_budgeted_binary_search(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    output_dir: Path,
    theta_names: Sequence[str] | str = "fit_flags",
    fit_parameter_names: Sequence[str] | str | None = "fit_flags",
    structure_id: str = "staged_binary",
    binary_model: str | None = "ELL1",
    clock_override_path: Path | None = None,
    config: StagedBudgetedSearchConfig | None = None,
) -> StagedSearchResult:
    active_config = config or StagedBudgetedSearchConfig()
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    cluster_manifest_path = Path(cluster_manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_theta_names, resolved_fit_parameter_names = _resolve_initial_fit_flag_policy(
        par_path=par_path,
        theta_names=theta_names,
        fit_parameter_names=fit_parameter_names,
        config=active_config,
    )
    initial_fit_parameter_names = tuple(resolved_fit_parameter_names)

    manifest = resolve_source_session_manifest(
        load_cluster_manifest(cluster_manifest_path),
        tim_path,
    )
    cluster_manifest_path = write_cluster_manifest(
        manifest,
        output_dir / "source_session_clusters.yaml",
    )
    stable_pulse_number_reference = prepare_stable_pulse_number_reference(
        par_path=par_path,
        full_tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        output_path=output_dir / "pulse_number_reference.par",
        output_tim_path=output_dir / "pulse_number_reference.tim",
        clock_override_path=clock_override_path,
    )
    search_par_path = stable_pulse_number_reference.path
    search_tim_path = stable_pulse_number_reference.tim_path
    _write_json(
        output_dir / "pulse_number_reference.json",
        stable_pulse_number_reference.metadata,
    )
    end_cluster_count = min(
        int(active_config.end_cluster_count or len(manifest.clusters)),
        len(manifest.clusters),
    )
    start_cluster_count = min(max(int(active_config.start_cluster_count), 1), end_cluster_count)
    active_par_path = par_path
    base_values = _base_parameter_values(active_par_path, resolved_theta_names)
    parameter_scales = _fit_prior_parameter_scales(
        active_par_path,
        resolved_theta_names,
        formal_uncertainty_scale=float(active_config.formal_uncertainty_scale),
        physical_width_fraction=float(active_config.physical_prior_width_fraction),
    )

    information_rank_start_decision: dict[str, Any] = {
        "status": "disabled",
        "requested_start_cluster_count": int(start_cluster_count),
        "selected_start_cluster_count": int(start_cluster_count),
        "prefixes": [],
    }
    if bool(active_config.information_rank_start_enabled):
        rank_probe_dir = output_dir / "information_rank_start"

        def _rank_probe_geometry(cluster_count: int) -> Any:
            probe_dir = rank_probe_dir / f"stage_{int(cluster_count):02d}"
            probe = write_prefix_timing_inputs(
                tim_path=search_tim_path,
                cluster_manifest_path=cluster_manifest_path,
                prefix_cluster_count=int(cluster_count),
                output_dir=probe_dir,
            )
            return build_pint_alias_geometry(
                par_path=search_par_path,
                tim_path=probe.tim_path,
                cluster_manifest_path=probe.cluster_manifest_path,
                parameter_names=resolved_theta_names,
                clock_override_path=clock_override_path,
            )

        start_cluster_count, information_rank_start_decision = (
            _choose_information_rank_start_cluster_count(
                requested_start_cluster_count=start_cluster_count,
                end_cluster_count=end_cluster_count,
                parameter_scales=parameter_scales,
                phase_sigma=float(active_config.phase_sigma),
                phase_slope_sigma_scale=float(active_config.phase_slope_sigma_scale),
                phase_slope_sigma_floor=float(active_config.phase_slope_sigma_floor),
                rank_rtol=float(active_config.information_rank_rtol),
                geometry_for_cluster_count=_rank_probe_geometry,
            )
        )
    stage_cluster_counts = _atomic_parent_stage_cluster_counts(
        manifest,
        start_cluster_count=start_cluster_count,
        end_cluster_count=end_cluster_count,
    )
    if not stage_cluster_counts:
        raise ValueError("APOLLO staged search has no complete parent-local stage boundary.")
    start_cluster_count = int(stage_cluster_counts[0])
    end_cluster_count = int(stage_cluster_counts[-1])
    lookahead_end_cluster_count = max(
        end_cluster_count,
        min(
            int(active_config.parent_lookahead_end_cluster_count or end_cluster_count),
            len(manifest.clusters),
        ),
    )
    information_rank_start_decision = {
        **information_rank_start_decision,
        "atomic_parent_stage_counts": list(stage_cluster_counts),
        "atomic_aligned_start_cluster_count": start_cluster_count,
        "atomic_aligned_end_cluster_count": end_cluster_count,
    }
    _write_json(output_dir / "information_rank_start.json", information_rank_start_decision)

    current_branches = [
        _initial_branch(
            structure_id=structure_id,
            binary_model=binary_model,
            theta_names=resolved_theta_names,
        )
    ]
    all_certifications: list[BranchCertification] = []
    stage_records: list[dict[str, Any]] = []
    loop_start_cluster_count = start_cluster_count
    resume_state = _load_resume_state_from_artifacts(
        output_dir=output_dir,
        start_cluster_count=start_cluster_count,
        end_cluster_count=end_cluster_count,
        stage_cluster_counts=stage_cluster_counts,
        enabled=bool(active_config.resume_from_stage_artifacts),
    )
    if resume_state is not None:
        current_branches = resume_state.current_branches
        all_certifications = resume_state.certifications
        stage_records = resume_state.stage_records
        loop_start_cluster_count = min(
            max(int(resume_state.next_cluster_count), start_cluster_count),
            end_cluster_count + 1,
        )
        (
            active_par_path,
            resolved_theta_names,
            resolved_fit_parameter_names,
            resume_dynamic_diagnostics,
        ) = _resume_dynamic_parameter_state(
            par_path=par_path,
            output_dir=output_dir,
            current_branches=current_branches,
            resolved_theta_names=resolved_theta_names,
            resolved_fit_parameter_names=resolved_fit_parameter_names,
        )
        resumed_fit_flags = tuple(extract_fit_flagged_parameters(active_par_path))
        resumed_theta_names = {
            tuple(str(name).upper() for name in branch.theta_names)
            for branch in current_branches
        }
        if (
            tuple(resolved_theta_names) != initial_fit_parameter_names
            or tuple(resolved_fit_parameter_names or ()) != initial_fit_parameter_names
            or resumed_fit_flags != initial_fit_parameter_names
            or resumed_theta_names != {initial_fit_parameter_names}
        ):
            raise ValueError(
                "APOLLO resume artifacts changed the input par fit flags; start a clean "
                "fixed-model run instead of resuming a parameter-release run."
            )
        base_values = _base_parameter_values(active_par_path, resolved_theta_names)
        parameter_scales = _fit_prior_parameter_scales(
            active_par_path,
            resolved_theta_names,
            formal_uncertainty_scale=float(active_config.formal_uncertainty_scale),
            physical_width_fraction=float(active_config.physical_prior_width_fraction),
        )
        _write_json(output_dir / "resume_dynamic_parameter_state.json", resume_dynamic_diagnostics)

    for cluster_count in (
        item for item in stage_cluster_counts if item >= loop_start_cluster_count
    ):
        stage_index = cluster_count
        stage_dir = output_dir / f"stage_{stage_index:02d}"
        _write_stage_progress(
            stage_dir,
            "stage-start",
            {
                "stage": stage_index,
                "cluster_count": cluster_count,
                "parent_count": len(current_branches),
            },
        )
        prefix = write_prefix_timing_inputs(
            tim_path=search_tim_path,
            cluster_manifest_path=cluster_manifest_path,
            prefix_cluster_count=cluster_count,
            output_dir=stage_dir,
        )
        _write_stage_progress(
            stage_dir,
            "prefix-ready",
            {
                "stage": stage_index,
                "prefix_tim_path": prefix.tim_path,
                "prefix_cluster_manifest_path": prefix.cluster_manifest_path,
                "prefix_cluster_count": prefix.cluster_count,
                "prefix_gap_count": prefix.gap_count,
            },
        )
        geometry = build_pint_alias_geometry(
            par_path=search_par_path,
            tim_path=prefix.tim_path,
            cluster_manifest_path=prefix.cluster_manifest_path,
            parameter_names=resolved_theta_names,
            clock_override_path=clock_override_path,
        )
        _write_stage_progress(
            stage_dir,
            "geometry-ready",
            {
                "stage": stage_index,
                "gap_count": int(np.asarray(geometry.gap_jacobian).shape[0]),
                "parameter_count": len(resolved_theta_names),
            },
        )
        lookahead_cluster_count = _lookahead_cluster_count(
            cluster_count,
            lookahead_end_cluster_count,
            active_config.parent_lookahead_cluster_horizon,
        )
        lookahead_prefix = prefix
        lookahead_geometry = geometry
        if lookahead_cluster_count > cluster_count:
            lookahead_prefix = write_prefix_timing_inputs(
                tim_path=search_tim_path,
                cluster_manifest_path=cluster_manifest_path,
                prefix_cluster_count=lookahead_cluster_count,
                output_dir=stage_dir / "lookahead",
            )
            lookahead_geometry = build_pint_alias_geometry(
                par_path=search_par_path,
                tim_path=lookahead_prefix.tim_path,
                cluster_manifest_path=lookahead_prefix.cluster_manifest_path,
                parameter_names=resolved_theta_names,
                clock_override_path=clock_override_path,
            )
        (
            local_phase_slope_jacobian,
            reference_local_phase_slopes,
            local_phase_slope_sigmas,
        ) = _phase_slope_inputs(
            getattr(geometry, "local_phase_slope_evidence", None),
            active_config,
        )
        source_session_cluster_ids = tuple(
            str(cluster.cluster_id)
            for cluster in resolve_source_session_manifest(
                load_cluster_manifest(prefix.cluster_manifest_path),
                prefix.tim_path,
            ).clusters
        )
        future_local_phase_slope_evidence = _future_local_phase_slope_evidence(
            getattr(geometry, "local_phase_slope_evidence", None),
            getattr(lookahead_geometry, "local_phase_slope_evidence", None),
        )
        (
            lookahead_local_phase_slope_jacobian,
            lookahead_reference_local_phase_slopes,
            lookahead_local_phase_slope_sigmas,
        ) = _phase_slope_inputs(future_local_phase_slope_evidence, active_config)
        active_lookahead_phase_sigma = (
            float(active_config.parent_lookahead_phase_sigma)
            if active_config.parent_lookahead_phase_sigma is not None
            else float(active_config.phase_sigma)
        )

        def score_with_lookahead(
            branches: Sequence[ActiveTimingBranch],
            *,
            slope_sigma_scale: float | None = None,
        ) -> list[ActiveTimingBranch]:
            active_slope_jacobian = lookahead_local_phase_slope_jacobian
            active_slope_reference = lookahead_reference_local_phase_slopes
            active_slope_sigmas = lookahead_local_phase_slope_sigmas
            if slope_sigma_scale is not None:
                (
                    active_slope_jacobian,
                    active_slope_reference,
                    active_slope_sigmas,
                ) = _phase_slope_inputs(
                    future_local_phase_slope_evidence,
                    replace(
                        active_config,
                        phase_slope_sigma_scale=float(slope_sigma_scale),
                    ),
                )
            return _with_parent_predictive_lookahead(
                branches,
                gap_jacobian=np.asarray(lookahead_geometry.gap_jacobian, dtype=float),
                reference_phase_residuals=np.asarray(
                    lookahead_geometry.reference_phase_residuals,
                    dtype=float,
                ),
                phase_sigma=active_lookahead_phase_sigma,
                local_phase_slope_jacobian=active_slope_jacobian,
                reference_local_phase_slopes=active_slope_reference,
                local_phase_slope_sigmas=active_slope_sigmas,
            )

        _write_stage_progress(
            stage_dir,
            "lookahead-geometry-ready",
            {
                "stage": stage_index,
                "current_cluster_count": cluster_count,
                "lookahead_cluster_count": lookahead_cluster_count,
                "lookahead_gap_count": int(
                    np.asarray(lookahead_geometry.gap_jacobian, dtype=float).shape[0]
                ),
                "phase_sigma": active_lookahead_phase_sigma,
                "phase_slope_enabled": bool(active_config.phase_slope_enabled),
                "current_slope_evidence_count": (
                    0
                    if local_phase_slope_jacobian is None
                    else int(local_phase_slope_jacobian.shape[0])
                ),
                "lookahead_slope_evidence_count": (
                    0
                    if lookahead_local_phase_slope_jacobian is None
                    else int(lookahead_local_phase_slope_jacobian.shape[0])
                ),
            },
        )
        if cluster_count == start_cluster_count:
            seed_branch = replace(
                current_branches[0],
                diagnostics={**(current_branches[0].diagnostics or {}), "stage": stage_index},
            )
            candidates, initial_decision = _initial_stage_candidates(
                seed_branch=seed_branch,
                gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                parameter_scales=parameter_scales,
                config=active_config,
                local_phase_slope_jacobian=local_phase_slope_jacobian,
                reference_local_phase_slopes=reference_local_phase_slopes,
                local_phase_slope_sigmas=local_phase_slope_sigmas,
            )
            _write_stage_progress(
                stage_dir,
                "initial-stage-candidates-ready",
                {
                    "stage": stage_index,
                    "candidate_count": len(candidates),
                    "decision": initial_decision,
                },
            )
        else:
            initial_decision = {"proposal_source": "incremental-gap-expansion"}
            candidates = []
            ranked_parents = sorted(
                (
                    branch
                    for branch in score_with_lookahead(current_branches)
                    if _is_certified_parent(branch)
                ),
                key=lambda branch: (
                    float((branch.diagnostics or {}).get("certified_reduced_chi2", math.inf)),
                    float((branch.diagnostics or {}).get("certified_weighted_rms_us", math.inf)),
                    branch.branch_id,
                ),
            )
            for parent_rank, parent in enumerate(ranked_parents):
                (
                    incremental_slope_jacobian,
                    incremental_slope_reference,
                    incremental_slope_sigmas,
                ) = _incremental_phase_slope_inputs(
                    evidence=getattr(geometry, "local_phase_slope_evidence", None),
                    config=active_config,
                    source_session_cluster_ids=source_session_cluster_ids,
                    inherited_gap_count=len(parent.wraps),
                )
                candidates.extend(
                    _expand_parent_gap_block(
                        parent=parent,
                        parent_rank=parent_rank,
                        stage_index=stage_index,
                        gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                        reference_phase_residuals=np.asarray(
                            geometry.reference_phase_residuals,
                            dtype=float,
                        ),
                        config=active_config,
                        local_phase_slope_jacobian=incremental_slope_jacobian,
                        reference_local_phase_slopes=incremental_slope_reference,
                        local_phase_slope_sigmas=incremental_slope_sigmas,
                    )
                )
            inherited_gap_count = min(
                (len(parent.wraps) for parent in ranked_parents),
                default=prefix.gap_count,
            )
            new_gap_indices = list(range(inherited_gap_count, prefix.gap_count))
            _write_stage_progress(
                stage_dir,
                "incremental-candidates-ready",
                {
                    "stage": stage_index,
                    "parent_count": len(ranked_parents),
                    "candidate_count": len(candidates),
                    "new_gap_indices": new_gap_indices,
                    "atomic_gap_count": len(new_gap_indices),
                },
            )
            if active_config.stage_lattice_refresh_enabled:
                _write_stage_progress(
                    stage_dir,
                    "stage-lattice-refresh-start",
                    {"stage": stage_index},
                )
                refresh_candidates, refresh_decision = _stage_lattice_refresh_candidates(
                    seed_branch=_initial_branch(
                        structure_id=structure_id,
                        binary_model=binary_model,
                        theta_names=resolved_theta_names,
                    ),
                    stage_index=stage_index,
                    gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                    parameter_scales=parameter_scales,
                    config=active_config,
                    local_phase_slope_jacobian=local_phase_slope_jacobian,
                    reference_local_phase_slopes=reference_local_phase_slopes,
                    local_phase_slope_sigmas=local_phase_slope_sigmas,
                )
                candidates.extend(refresh_candidates)
                initial_decision = {
                    **initial_decision,
                    "refresh_decision": refresh_decision,
                }
                _write_stage_progress(
                    stage_dir,
                    "stage-lattice-refresh-ready",
                    {
                        "stage": stage_index,
                        "new_candidate_count": len(refresh_candidates),
                        "total_candidate_count": len(candidates),
                        "decision": refresh_decision,
                    },
                )
            if active_config.conditioned_lattice_refresh_enabled:
                _write_stage_progress(
                    stage_dir,
                    "conditioned-lattice-refresh-start",
                    {
                        "stage": stage_index,
                        "parent_count": len(ranked_parents),
                    },
                )
                conditioned_candidates, conditioned_decision = _conditioned_lattice_refresh_candidates(
                    parent_branches=ranked_parents,
                    stage_index=stage_index,
                    gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                    parameter_scales=parameter_scales,
                    config=active_config,
                    checkpoint_dir=(
                        stage_dir / "conditioned_lattice_parent_cache"
                        if active_config.refresh_parent_cache_enabled
                        else None
                    ),
                    local_phase_slope_jacobian=local_phase_slope_jacobian,
                    reference_local_phase_slopes=reference_local_phase_slopes,
                    local_phase_slope_sigmas=local_phase_slope_sigmas,
                    lookahead_gap_jacobian=np.asarray(
                        lookahead_geometry.gap_jacobian,
                        dtype=float,
                    ),
                    lookahead_reference_phase_residuals=np.asarray(
                        lookahead_geometry.reference_phase_residuals,
                        dtype=float,
                    ),
                    lookahead_local_phase_slope_jacobian=(
                        lookahead_local_phase_slope_jacobian
                    ),
                    lookahead_reference_local_phase_slopes=(
                        lookahead_reference_local_phase_slopes
                    ),
                    lookahead_local_phase_slope_sigmas=(
                        lookahead_local_phase_slope_sigmas
                    ),
                )
                candidates.extend(conditioned_candidates)
                initial_decision = {
                    **initial_decision,
                    "conditioned_refresh_decision": conditioned_decision,
                }
                _write_stage_progress(
                    stage_dir,
                    "conditioned-lattice-refresh-ready",
                    {
                        "stage": stage_index,
                        "new_candidate_count": len(conditioned_candidates),
                        "total_candidate_count": len(candidates),
                        "decision": conditioned_decision,
                    },
                )
            if active_config.posterior_lattice_refresh_enabled:
                _write_stage_progress(
                    stage_dir,
                    "posterior-lattice-refresh-start",
                    {
                        "stage": stage_index,
                        "parent_count": len(ranked_parents),
                    },
                )
                posterior_candidates, posterior_decision = _posterior_lattice_refresh_candidates(
                    parent_branches=ranked_parents,
                    stage_index=stage_index,
                    gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                    parameter_scales=parameter_scales,
                    config=active_config,
                    checkpoint_dir=(
                        stage_dir / "posterior_lattice_parent_cache"
                        if active_config.refresh_parent_cache_enabled
                        else None
                    ),
                    local_phase_slope_jacobian=local_phase_slope_jacobian,
                    reference_local_phase_slopes=reference_local_phase_slopes,
                    local_phase_slope_sigmas=local_phase_slope_sigmas,
                    lookahead_gap_jacobian=np.asarray(
                        lookahead_geometry.gap_jacobian,
                        dtype=float,
                    ),
                    lookahead_reference_phase_residuals=np.asarray(
                        lookahead_geometry.reference_phase_residuals,
                        dtype=float,
                    ),
                    lookahead_local_phase_slope_jacobian=(
                        lookahead_local_phase_slope_jacobian
                    ),
                    lookahead_reference_local_phase_slopes=(
                        lookahead_reference_local_phase_slopes
                    ),
                    lookahead_local_phase_slope_sigmas=(
                        lookahead_local_phase_slope_sigmas
                    ),
                )
                candidates.extend(posterior_candidates)
                initial_decision = {
                    **initial_decision,
                    "posterior_refresh_decision": posterior_decision,
                }
                _write_stage_progress(
                    stage_dir,
                    "posterior-lattice-refresh-ready",
                    {
                        "stage": stage_index,
                        "new_candidate_count": len(posterior_candidates),
                        "total_candidate_count": len(candidates),
                        "decision": posterior_decision,
                    },
                )
            if active_config.posterior_lattice_stable_order_enabled:
                _write_stage_progress(
                    stage_dir,
                    "posterior-stable-order-refresh-start",
                    {
                        "stage": stage_index,
                        "parent_count": len(ranked_parents),
                    },
                )
                stable_order_candidates, stable_order_decision = (
                    _posterior_stable_order_lattice_refresh_candidates(
                        parent_branches=ranked_parents,
                        stage_index=stage_index,
                        gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                        reference_phase_residuals=np.asarray(
                            geometry.reference_phase_residuals,
                            dtype=float,
                        ),
                        parameter_scales=parameter_scales,
                        config=active_config,
                        checkpoint_dir=(
                            stage_dir / "posterior_stable_order_parent_cache"
                            if active_config.refresh_parent_cache_enabled
                            else None
                        ),
                        local_phase_slope_jacobian=local_phase_slope_jacobian,
                        reference_local_phase_slopes=reference_local_phase_slopes,
                        local_phase_slope_sigmas=local_phase_slope_sigmas,
                        lookahead_gap_jacobian=np.asarray(
                            lookahead_geometry.gap_jacobian,
                            dtype=float,
                        ),
                        lookahead_reference_phase_residuals=np.asarray(
                            lookahead_geometry.reference_phase_residuals,
                            dtype=float,
                        ),
                        lookahead_local_phase_slope_jacobian=(
                            lookahead_local_phase_slope_jacobian
                        ),
                        lookahead_reference_local_phase_slopes=(
                            lookahead_reference_local_phase_slopes
                        ),
                        lookahead_local_phase_slope_sigmas=(
                            lookahead_local_phase_slope_sigmas
                        ),
                    )
                )
                candidates.extend(stable_order_candidates)
                initial_decision = {
                    **initial_decision,
                    "posterior_stable_order_refresh_decision": stable_order_decision,
                }
                _write_stage_progress(
                    stage_dir,
                    "posterior-stable-order-refresh-ready",
                    {
                        "stage": stage_index,
                        "new_candidate_count": len(stable_order_candidates),
                        "total_candidate_count": len(candidates),
                        "decision": stable_order_decision,
                    },
                )
        candidates = score_with_lookahead(candidates)
        _write_stage_progress(
            stage_dir,
            "candidate-lookahead-ready",
            {
                "stage": stage_index,
                "candidate_count": len(candidates),
                "lookahead_cluster_count": lookahead_cluster_count,
            },
        )
        certify_budget = (
            int(active_config.final_certify_budget)
            if active_config.final_certify_budget is not None and cluster_count == end_cluster_count
            else int(active_config.certify_budget)
        )
        selected_for_certification, certification_decision = _select_certification_candidates(
            list(candidates),
            budget=certify_budget,
            config=active_config,
        )
        certification_decision = {
            **certification_decision,
            "initial_decision": initial_decision,
        }
        _write_stage_progress(
            stage_dir,
            "certification-selection-ready",
            {
                "stage": stage_index,
                "candidate_count": len(candidates),
                "selected_count": len(selected_for_certification),
                "certify_budget": certify_budget,
                "decision": certification_decision,
            },
        )
        certifier = build_alias_branch_certifier(
            par_path=search_par_path,
            tim_path=prefix.tim_path,
            cluster_manifest_path=prefix.cluster_manifest_path,
            parameter_names=resolved_theta_names,
            fit_parameter_names=resolved_fit_parameter_names,
            maxiter=int(active_config.maxiter),
            clock_override_path=clock_override_path,
        )
        _write_stage_progress(
            stage_dir,
            "initial-certification-start",
            {
                "stage": stage_index,
                "selected_count": len(selected_for_certification),
                "parallel_workers": int(active_config.certify_parallel_workers),
                "parallel_backend": active_config.certify_parallel_backend,
            },
        )
        initial_certifications = _certify_branches(
            branches=selected_for_certification,
            certifier=certifier,
            max_workers=int(active_config.certify_parallel_workers),
            parallel_backend=active_config.certify_parallel_backend,
        )
        _write_stage_progress(
            stage_dir,
            "initial-certification-ready",
            {
                "stage": stage_index,
                "certification_count": len(initial_certifications),
            },
        )
        normalized_initial = score_with_lookahead([
            _normalize_branch_to_certification(
                branch,
                certification,
                base_parameter_values=base_values,
                parameter_scales=parameter_scales,
            )
            for branch, certification in zip(selected_for_certification, initial_certifications, strict=True)
        ])
        reacquisition_decision = _adaptive_global_reacquisition_decision(
            parent_branches=current_branches,
            stage_certifications=initial_certifications,
            config=active_config,
        )
        if reacquisition_decision["triggered"]:
            reacquisition_config, reacquisition_budget = (
                _adaptive_global_reacquisition_config(
                    config=active_config,
                    base_certify_budget=certify_budget,
                )
            )
            _write_stage_progress(
                stage_dir,
                "adaptive-global-reacquisition-start",
                {
                    "stage": stage_index,
                    "decision": reacquisition_decision,
                    "certify_budget": reacquisition_budget,
                },
            )
            reacquisition_candidates, reacquisition_generation = (
                _stage_lattice_refresh_candidates(
                    seed_branch=_initial_branch(
                        structure_id=structure_id,
                        binary_model=binary_model,
                        theta_names=resolved_theta_names,
                    ),
                    stage_index=stage_index,
                    gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(
                        geometry.reference_phase_residuals,
                        dtype=float,
                    ),
                    parameter_scales=parameter_scales,
                    config=reacquisition_config,
                    local_phase_slope_jacobian=local_phase_slope_jacobian,
                    reference_local_phase_slopes=reference_local_phase_slopes,
                    local_phase_slope_sigmas=local_phase_slope_sigmas,
                )
            )
            already_certified_keys = {
                (
                    branch.wraps,
                    tuple(
                        float(f"{value:.18e}")
                        for value in np.asarray(branch.theta_offset, dtype=float)
                    ),
                )
                for branch in selected_for_certification
            }
            novel_reacquisition_candidates: list[ActiveTimingBranch] = []
            for branch in reacquisition_candidates:
                key = (
                    branch.wraps,
                    tuple(
                        float(f"{value:.18e}")
                        for value in np.asarray(branch.theta_offset, dtype=float)
                    ),
                )
                if key in already_certified_keys:
                    continue
                already_certified_keys.add(key)
                novel_reacquisition_candidates.append(branch)
            novel_reacquisition_candidates = score_with_lookahead(
                novel_reacquisition_candidates
            )
            selected_reacquisition, reacquisition_selection = (
                _select_certification_candidates(
                    novel_reacquisition_candidates,
                    budget=reacquisition_budget,
                    config=reacquisition_config,
                )
            )
            reacquisition_certifications = _certify_branches(
                branches=selected_reacquisition,
                certifier=certifier,
                max_workers=int(active_config.certify_parallel_workers),
                parallel_backend=active_config.certify_parallel_backend,
            )
            normalized_reacquisition = score_with_lookahead(
                [
                    _normalize_branch_to_certification(
                        branch,
                        certification,
                        base_parameter_values=base_values,
                        parameter_scales=parameter_scales,
                    )
                    for branch, certification in zip(
                        selected_reacquisition,
                        reacquisition_certifications,
                        strict=True,
                    )
                ]
            )
            candidates.extend(reacquisition_candidates)
            selected_for_certification.extend(selected_reacquisition)
            initial_certifications.extend(reacquisition_certifications)
            normalized_initial.extend(normalized_reacquisition)
            reacquisition_best_branch, reacquisition_best = _best_pair(
                normalized_initial,
                initial_certifications,
            )
            searched_prior_scales = _active_prior_scales(
                float(reacquisition_config.stage_lattice_refresh_prior_scale),
                reacquisition_config.stage_lattice_refresh_prior_scales,
            )
            tempering_decision = _adaptive_prior_tempering_decision(
                best_branch=reacquisition_best_branch,
                best_certification=reacquisition_best,
                searched_prior_scales=searched_prior_scales,
                config=active_config,
            )
            tempered_candidates: list[ActiveTimingBranch] = []
            novel_tempered_candidates: list[ActiveTimingBranch] = []
            selected_tempered: list[ActiveTimingBranch] = []
            tempered_certifications: list[BranchCertification] = []
            if tempering_decision["triggered"]:
                tempered_config, tempered_budget = _adaptive_prior_tempering_config(
                    config=reacquisition_config,
                    certify_budget=reacquisition_budget,
                )
                _write_stage_progress(
                    stage_dir,
                    "adaptive-prior-tempering-start",
                    {
                        "stage": stage_index,
                        "decision": tempering_decision,
                        "certify_budget": tempered_budget,
                    },
                )
                tempered_candidates, tempered_generation = (
                    _stage_lattice_refresh_candidates(
                        seed_branch=_initial_branch(
                            structure_id=structure_id,
                            binary_model=binary_model,
                            theta_names=resolved_theta_names,
                        ),
                        stage_index=stage_index,
                        gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                        reference_phase_residuals=np.asarray(
                            geometry.reference_phase_residuals,
                            dtype=float,
                        ),
                        parameter_scales=parameter_scales,
                        config=tempered_config,
                        local_phase_slope_jacobian=local_phase_slope_jacobian,
                        reference_local_phase_slopes=reference_local_phase_slopes,
                        local_phase_slope_sigmas=local_phase_slope_sigmas,
                        branch_suffix="_temper",
                    )
                )
                for branch in tempered_candidates:
                    key = (
                        branch.wraps,
                        tuple(
                            float(f"{value:.18e}")
                            for value in np.asarray(branch.theta_offset, dtype=float)
                        ),
                    )
                    if key in already_certified_keys:
                        continue
                    already_certified_keys.add(key)
                    novel_tempered_candidates.append(branch)
                novel_tempered_candidates = score_with_lookahead(
                    novel_tempered_candidates
                )
                selected_tempered, tempered_selection = (
                    _select_certification_candidates(
                        novel_tempered_candidates,
                        budget=tempered_budget,
                        config=tempered_config,
                    )
                )
                tempered_certifications = _certify_branches(
                    branches=selected_tempered,
                    certifier=certifier,
                    max_workers=int(active_config.certify_parallel_workers),
                    parallel_backend=active_config.certify_parallel_backend,
                )
                normalized_tempered = score_with_lookahead(
                    [
                        _normalize_branch_to_certification(
                            branch,
                            certification,
                            base_parameter_values=base_values,
                            parameter_scales=parameter_scales,
                        )
                        for branch, certification in zip(
                            selected_tempered,
                            tempered_certifications,
                            strict=True,
                        )
                    ]
                )
                candidates.extend(tempered_candidates)
                selected_for_certification.extend(selected_tempered)
                initial_certifications.extend(tempered_certifications)
                normalized_initial.extend(normalized_tempered)
                reacquisition_best_branch, reacquisition_best = _best_pair(
                    normalized_initial,
                    initial_certifications,
                )
                tempering_decision = {
                    **tempering_decision,
                    "generation": tempered_generation,
                    "generated_candidate_count": len(tempered_candidates),
                    "novel_candidate_count": len(novel_tempered_candidates),
                    "selected_candidate_count": len(selected_tempered),
                    "certify_budget": tempered_budget,
                    "selection": tempered_selection,
                    "best_reduced_chi2_after_tempering": (
                        reacquisition_best.reduced_chi2
                        if reacquisition_best is not None
                        else None
                    ),
                    "best_weighted_rms_us_after_tempering": (
                        reacquisition_best.weighted_rms_us
                        if reacquisition_best is not None
                        else None
                    ),
                }
                _write_stage_progress(
                    stage_dir,
                    "adaptive-prior-tempering-ready",
                    {
                        "stage": stage_index,
                        "decision": tempering_decision,
                    },
                )
            slope_tempering_decision = _adaptive_slope_tempering_decision(
                best_certification=reacquisition_best,
                evidence=getattr(geometry, "local_phase_slope_evidence", None),
                config=active_config,
            )
            slope_tempered_candidates: list[ActiveTimingBranch] = []
            novel_slope_tempered_candidates: list[ActiveTimingBranch] = []
            selected_slope_tempered: list[ActiveTimingBranch] = []
            slope_tempered_certifications: list[BranchCertification] = []
            multi_anchor_candidates: list[ActiveTimingBranch] = []
            selected_multi_anchor: list[ActiveTimingBranch] = []
            multi_anchor_certifications: list[BranchCertification] = []
            multi_anchor_decision: dict[str, Any] = {
                "enabled": bool(active_config.adaptive_multi_anchor_reacquisition_enabled),
                "triggered": False,
                "reason": "not-run",
            }
            multi_anchor_slope_scale = float(active_config.phase_slope_sigma_scale)
            if slope_tempering_decision["triggered"]:
                slope_sigma_scale = float(
                    slope_tempering_decision["tempered_slope_sigma_scale"]
                )
                slope_tempered_config = replace(
                    active_config,
                    phase_slope_sigma_scale=slope_sigma_scale,
                )
                (
                    slope_tempered_jacobian,
                    slope_tempered_reference,
                    slope_tempered_sigmas,
                ) = _phase_slope_inputs(
                    getattr(geometry, "local_phase_slope_evidence", None),
                    slope_tempered_config,
                )
                _write_stage_progress(
                    stage_dir,
                    "adaptive-slope-tempering-start",
                    {
                        "stage": stage_index,
                        "decision": slope_tempering_decision,
                        "certify_budget": reacquisition_budget,
                    },
                )
                slope_tempered_candidates, slope_tempered_generation = (
                    _stage_lattice_refresh_candidates(
                        seed_branch=_initial_branch(
                            structure_id=structure_id,
                            binary_model=binary_model,
                            theta_names=resolved_theta_names,
                        ),
                        stage_index=stage_index,
                        gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                        reference_phase_residuals=np.asarray(
                            geometry.reference_phase_residuals,
                            dtype=float,
                        ),
                        parameter_scales=parameter_scales,
                        config=reacquisition_config,
                        local_phase_slope_jacobian=slope_tempered_jacobian,
                        reference_local_phase_slopes=slope_tempered_reference,
                        local_phase_slope_sigmas=slope_tempered_sigmas,
                        branch_suffix="_slope_temper",
                        slope_sigma_scale=slope_sigma_scale,
                    )
                )
                for branch in slope_tempered_candidates:
                    key = (
                        branch.wraps,
                        tuple(
                            float(f"{value:.18e}")
                            for value in np.asarray(branch.theta_offset, dtype=float)
                        ),
                    )
                    if key in already_certified_keys:
                        continue
                    already_certified_keys.add(key)
                    novel_slope_tempered_candidates.append(branch)
                novel_slope_tempered_candidates = score_with_lookahead(
                    novel_slope_tempered_candidates,
                    slope_sigma_scale=slope_sigma_scale,
                )
                selected_slope_tempered, slope_tempered_selection = (
                    _select_certification_candidates(
                        novel_slope_tempered_candidates,
                        budget=reacquisition_budget,
                        config=reacquisition_config,
                    )
                )
                slope_tempered_certifications = _certify_branches(
                    branches=selected_slope_tempered,
                    certifier=certifier,
                    max_workers=int(active_config.certify_parallel_workers),
                    parallel_backend=active_config.certify_parallel_backend,
                )
                normalized_slope_tempered = score_with_lookahead(
                    [
                        _normalize_branch_to_certification(
                            branch,
                            certification,
                            base_parameter_values=base_values,
                            parameter_scales=parameter_scales,
                        )
                        for branch, certification in zip(
                            selected_slope_tempered,
                            slope_tempered_certifications,
                            strict=True,
                        )
                    ],
                    slope_sigma_scale=slope_sigma_scale,
                )
                candidates.extend(slope_tempered_candidates)
                selected_for_certification.extend(selected_slope_tempered)
                initial_certifications.extend(slope_tempered_certifications)
                normalized_initial.extend(normalized_slope_tempered)
                reacquisition_best_branch, reacquisition_best = _best_pair(
                    normalized_initial,
                    initial_certifications,
                )
                slope_tempering_decision = {
                    **slope_tempering_decision,
                    "generation": slope_tempered_generation,
                    "generated_candidate_count": len(slope_tempered_candidates),
                    "novel_candidate_count": len(novel_slope_tempered_candidates),
                    "selected_candidate_count": len(selected_slope_tempered),
                    "certify_budget": reacquisition_budget,
                    "selection": slope_tempered_selection,
                    "best_reduced_chi2_after_tempering": (
                        reacquisition_best.reduced_chi2
                        if reacquisition_best is not None
                        else None
                    ),
                    "best_weighted_rms_us_after_tempering": (
                        reacquisition_best.weighted_rms_us
                        if reacquisition_best is not None
                        else None
                    ),
                }
                _write_stage_progress(
                    stage_dir,
                    "adaptive-slope-tempering-ready",
                    {
                        "stage": stage_index,
                        "decision": slope_tempering_decision,
                    },
                )
                multi_anchor_slope_scale = slope_sigma_scale
            if bool(active_config.adaptive_multi_anchor_reacquisition_enabled):
                multi_anchor_best = reacquisition_best
                multi_anchor_quality = (
                    float(multi_anchor_best.reduced_chi2)
                    if multi_anchor_best is not None
                    and multi_anchor_best.reduced_chi2 is not None
                    and np.isfinite(float(multi_anchor_best.reduced_chi2))
                    else None
                )
                multi_anchor_threshold = max(
                    float(active_config.adaptive_global_reacquisition_min_reduced_chi2),
                    0.0,
                )
                if multi_anchor_quality is None or multi_anchor_quality > multi_anchor_threshold:
                    parent_ids = {branch.branch_id for branch in current_branches}
                    anchor_parent_certifications = [
                        certification
                        for certification in all_certifications
                        if certification.branch_id in parent_ids
                    ]
                    multi_anchor_budget = int(
                        math.ceil(
                            max(int(reacquisition_budget), 1)
                            * max(
                                float(active_config.adaptive_multi_anchor_certify_budget_multiplier),
                                1.0,
                            )
                        )
                    )

                    def run_multi_anchor_round(
                        round_index: int,
                        round_parent_branches: Sequence[ActiveTimingBranch],
                        round_parent_certifications: Sequence[BranchCertification],
                    ) -> dict[str, Any]:
                        round_candidates, round_generation = (
                            _generate_multi_anchor_reacquisition_candidates(
                                search_par_path=search_par_path,
                                prefix=prefix,
                                stage_dir=stage_dir,
                                stage_index=stage_index,
                                parent_branches=round_parent_branches,
                                parent_certifications=round_parent_certifications,
                                theta_names=resolved_theta_names,
                                structure_id=structure_id,
                                binary_model=binary_model,
                                config=reacquisition_config,
                                clock_override_path=clock_override_path,
                                slope_sigma_scale=multi_anchor_slope_scale,
                                round_index=round_index,
                            )
                        )
                        round_candidates = score_with_lookahead(
                            round_candidates,
                            slope_sigma_scale=multi_anchor_slope_scale,
                        )
                        round_selected, round_selection = (
                            _select_multi_anchor_certification_candidates(
                                round_candidates,
                                budget=multi_anchor_budget,
                                config=reacquisition_config,
                            )
                        )
                        round_certifications = _certify_branches(
                            branches=round_selected,
                            certifier=certifier,
                            max_workers=int(active_config.certify_parallel_workers),
                            parallel_backend=active_config.certify_parallel_backend,
                        )
                        round_normalized = score_with_lookahead(
                            [
                                _normalize_branch_to_certification(
                                    branch,
                                    certification,
                                    base_parameter_values=base_values,
                                    parameter_scales=parameter_scales,
                                )
                                for branch, certification in zip(
                                    round_selected,
                                    round_certifications,
                                    strict=True,
                                )
                            ],
                            slope_sigma_scale=multi_anchor_slope_scale,
                        )
                        _, round_best = _best_pair(
                            round_normalized,
                            round_certifications,
                        )
                        round_diagnostics = {
                            **round_generation,
                            "certify_budget": multi_anchor_budget,
                            "selection": round_selection,
                            "best_reduced_chi2": (
                                round_best.reduced_chi2
                                if round_best is not None
                                else None
                            ),
                            "best_weighted_rms_us": (
                                round_best.weighted_rms_us
                                if round_best is not None
                                else None
                            ),
                        }
                        _write_stage_progress(
                            stage_dir,
                            "adaptive-multi-anchor-round-ready",
                            {
                                "stage": stage_index,
                                "round": round_index,
                                "decision": round_diagnostics,
                            },
                        )
                        return {
                            "candidates": round_candidates,
                            "selected_branches": round_selected,
                            "certifications": round_certifications,
                            "normalized_branches": round_normalized,
                            "diagnostics": round_diagnostics,
                        }

                    multi_anchor_result = _run_multi_anchor_round_controller(
                        parent_branches=current_branches,
                        parent_certifications=anchor_parent_certifications,
                        initial_quality=multi_anchor_quality,
                        quality_threshold=multi_anchor_threshold,
                        max_rounds=int(active_config.adaptive_multi_anchor_max_rounds),
                        min_fractional_improvement=float(
                            active_config.adaptive_multi_anchor_min_fractional_improvement
                        ),
                        run_round=run_multi_anchor_round,
                    )
                    multi_anchor_candidates = list(multi_anchor_result["candidates"])
                    selected_multi_anchor = list(
                        multi_anchor_result["selected_branches"]
                    )
                    multi_anchor_certifications = list(
                        multi_anchor_result["certifications"]
                    )
                    normalized_multi_anchor = list(
                        multi_anchor_result["normalized_branches"]
                    )
                    candidates.extend(multi_anchor_candidates)
                    selected_for_certification.extend(selected_multi_anchor)
                    initial_certifications.extend(multi_anchor_certifications)
                    normalized_initial.extend(normalized_multi_anchor)
                    reacquisition_best_branch, reacquisition_best = _best_pair(
                        normalized_initial,
                        initial_certifications,
                    )
                    multi_anchor_rounds = list(multi_anchor_result["rounds"])
                    multi_anchor_decision = {
                        "enabled": True,
                        "triggered": bool(multi_anchor_rounds),
                        "reason": "iterative-relinearization-complete",
                        "quality_before": multi_anchor_quality,
                        "quality_threshold": multi_anchor_threshold,
                        "max_rounds": max(
                            int(active_config.adaptive_multi_anchor_max_rounds),
                            1,
                        ),
                        "completed_rounds": int(
                            multi_anchor_result["completed_rounds"]
                        ),
                        "rounds": multi_anchor_rounds,
                        "stop_reason": (
                            multi_anchor_rounds[-1]["continuation"]["reason"]
                            if multi_anchor_rounds
                            else "no-rounds"
                        ),
                        "generated_candidate_count": len(multi_anchor_candidates),
                        "selected_candidate_count": len(selected_multi_anchor),
                        "certify_budget_per_round": multi_anchor_budget,
                        "best_reduced_chi2_after": (
                            reacquisition_best.reduced_chi2
                            if reacquisition_best is not None
                            else None
                        ),
                        "best_weighted_rms_us_after": (
                            reacquisition_best.weighted_rms_us
                            if reacquisition_best is not None
                            else None
                        ),
                    }
                else:
                    multi_anchor_decision = {
                        **multi_anchor_decision,
                        "reason": "quality-sufficient-after-reacquisition",
                        "quality": multi_anchor_quality,
                        "quality_threshold": multi_anchor_threshold,
                    }
                _write_stage_progress(
                    stage_dir,
                    "adaptive-multi-anchor-reacquisition-ready",
                    {
                        "stage": stage_index,
                        "decision": multi_anchor_decision,
                    },
                )
            reacquisition_decision = {
                **reacquisition_decision,
                "generation": reacquisition_generation,
                "generated_candidate_count": (
                    len(reacquisition_candidates)
                    + len(tempered_candidates)
                    + len(slope_tempered_candidates)
                    + len(multi_anchor_candidates)
                ),
                "novel_candidate_count": (
                    len(novel_reacquisition_candidates)
                    + len(novel_tempered_candidates)
                    + len(novel_slope_tempered_candidates)
                    + len(multi_anchor_candidates)
                ),
                "selected_candidate_count": (
                    len(selected_reacquisition)
                    + len(selected_tempered)
                    + len(selected_slope_tempered)
                    + len(selected_multi_anchor)
                ),
                "certify_budget": reacquisition_budget,
                "selection": reacquisition_selection,
                "prior_tempering": tempering_decision,
                "slope_tempering": slope_tempering_decision,
                "multi_anchor_reacquisition": multi_anchor_decision,
                "best_reduced_chi2_after_reacquisition": (
                    reacquisition_best.reduced_chi2
                    if reacquisition_best is not None
                    else None
                ),
                "best_weighted_rms_us_after_reacquisition": (
                    reacquisition_best.weighted_rms_us
                    if reacquisition_best is not None
                    else None
                ),
            }
            _write_stage_progress(
                stage_dir,
                "adaptive-global-reacquisition-ready",
                {
                    "stage": stage_index,
                    "decision": reacquisition_decision,
                },
            )
        certification_decision = {
            **certification_decision,
            "adaptive_global_reacquisition": reacquisition_decision,
        }
        repair_decision: dict[str, Any] = {
            "enabled": bool(active_config.residual_repair_enabled),
            "rounds": [],
        }
        repair_candidates: list[ActiveTimingBranch] = []
        session_candidates: list[ActiveTimingBranch] = []
        uncertainty_candidates: list[ActiveTimingBranch] = []
        parameter_release_candidates: list[ActiveTimingBranch] = []
        certifications = list(initial_certifications)
        normalized = list(normalized_initial)
        seen_keys = {
            (
                branch.wraps,
                tuple(float(f"{value:.18e}") for value in np.asarray(branch.theta_offset, dtype=float)),
            )
            for branch in normalized
        }
        session_lattice_decision: dict[str, Any] = {
            "enabled": bool(active_config.session_lattice_enabled),
            "rounds": [],
            "reason": "disabled",
        }
        if bool(active_config.session_lattice_enabled):
            max_session_rounds = max(
                int(active_config.session_lattice_relinearization_rounds),
                1,
            )
            session_quality = _best_pair(normalized, certifications)[1]
            quality_before = (
                float(session_quality.reduced_chi2)
                if session_quality is not None
                and session_quality.reduced_chi2 is not None
                and np.isfinite(float(session_quality.reduced_chi2))
                else None
            )
            session_lattice_decision = {
                "enabled": True,
                "rounds": [],
                "reason": "not-triggered",
                "quality_before": quality_before,
                "quality_threshold": max(
                    float(active_config.session_lattice_trigger_reduced_chi2),
                    0.0,
                ),
            }
            for session_round in range(1, max_session_rounds + 1):
                _write_stage_progress(
                    stage_dir,
                    "session-lattice-round-start",
                    {
                        "stage": stage_index,
                        "round": session_round,
                        "certification_count": len(certifications),
                        "quality_before": quality_before,
                    },
                )
                round_candidates, round_generation = (
                    _session_lattice_reacquisition_candidates(
                        par_path=search_par_path,
                        tim_path=prefix.tim_path,
                        cluster_manifest_path=prefix.cluster_manifest_path,
                        theta_names=resolved_theta_names,
                        parameter_scales=parameter_scales,
                        certifications=certifications,
                        structure_id=structure_id,
                        binary_model=binary_model,
                        stage_index=stage_index,
                        is_final_stage=(cluster_count == end_cluster_count),
                        config=active_config,
                        clock_override_path=clock_override_path,
                    )
                )
                novel_round_candidates: list[ActiveTimingBranch] = []
                for branch in round_candidates:
                    key = (
                        branch.wraps,
                        tuple(
                            float(f"{value:.18e}")
                            for value in np.asarray(branch.theta_offset, dtype=float)
                        ),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    novel_round_candidates.append(
                        replace(
                            branch,
                            branch_id=f"{branch.branch_id}_r{session_round:02d}",
                        )
                    )
                selected_session_round, round_selection = (
                    _select_certification_candidates(
                        novel_round_candidates,
                        budget=max(
                            int(active_config.session_lattice_certify_budget),
                            1,
                        ),
                        config=active_config,
                    )
                )
                round_certifications = _certify_branches(
                    branches=selected_session_round,
                    certifier=certifier,
                    max_workers=int(active_config.certify_parallel_workers),
                    parallel_backend=active_config.certify_parallel_backend,
                )
                normalized_round = score_with_lookahead(
                    [
                        _normalize_branch_to_certification(
                            branch,
                            certification,
                            base_parameter_values=base_values,
                            parameter_scales=parameter_scales,
                        )
                        for branch, certification in zip(
                            selected_session_round,
                            round_certifications,
                            strict=True,
                        )
                    ]
                )
                candidates.extend(round_candidates)
                selected_for_certification.extend(selected_session_round)
                certifications.extend(round_certifications)
                normalized.extend(normalized_round)
                _, session_best = _best_pair(normalized, certifications)
                quality_after = (
                    float(session_best.reduced_chi2)
                    if session_best is not None
                    and session_best.reduced_chi2 is not None
                    and np.isfinite(float(session_best.reduced_chi2))
                    else None
                )
                continuation = _multi_anchor_round_continuation(
                    round_index=session_round,
                    max_rounds=max_session_rounds,
                    quality_before=quality_before,
                    quality_after=quality_after,
                    quality_threshold=float(
                        active_config.session_lattice_trigger_reduced_chi2
                    ),
                    min_fractional_improvement=float(
                        active_config.session_lattice_min_fractional_improvement
                    ),
                )
                round_record = {
                    "round": session_round,
                    "generation": round_generation,
                    "generated_candidate_count": len(round_candidates),
                    "novel_candidate_count": len(novel_round_candidates),
                    "selected_candidate_count": len(selected_session_round),
                    "selection": round_selection,
                    "quality_before": quality_before,
                    "quality_after": quality_after,
                    "continuation": continuation,
                }
                session_lattice_decision["rounds"].append(round_record)
                _write_stage_progress(
                    stage_dir,
                    "session-lattice-round-ready",
                    {
                        "stage": stage_index,
                        "round": session_round,
                        "decision": round_record,
                    },
                )
                quality_before = quality_after
                if not continuation["continue"]:
                    session_lattice_decision["reason"] = continuation["reason"]
                    break
            certification_decision = {
                **certification_decision,
                "session_lattice": session_lattice_decision,
            }
        if active_config.residual_repair_enabled and normalized_initial:
            repair_rounds = max(int(active_config.residual_repair_rounds), 0)
            for repair_round in range(repair_rounds):
                normalized = score_with_lookahead(normalized)
                _write_stage_progress(
                    stage_dir,
                    "repair-round-start",
                    {
                        "stage": stage_index,
                        "round": int(repair_round + 1),
                        "normalized_count": len(normalized),
                        "certification_count": len(certifications),
                    },
                )
                preliminary_survivors, survivor_probe_decision = _select_stage_survivors(
                    normalized,
                    certifications,
                    replace(
                        active_config,
                        beam_size=max(int(active_config.residual_repair_parent_top_k), 1),
                    ),
                )
                parent_ids = {branch.branch_id for branch in preliminary_survivors}
                parent_certifications = [
                    certification
                    for certification in certifications
                    if certification.branch_id in parent_ids
                ]
                if not preliminary_survivors or not parent_certifications:
                    _write_stage_progress(
                        stage_dir,
                        "repair-round-no-parents",
                        {
                            "stage": stage_index,
                            "round": int(repair_round + 1),
                            "survivor_probe_decision": survivor_probe_decision,
                        },
                    )
                    repair_decision["rounds"].append(
                        {
                            "round": int(repair_round + 1),
                            "status": "no-parents",
                            "survivor_probe_decision": survivor_probe_decision,
                        }
                    )
                    break
                repair_payloads = _build_repair_payloads(
                    par_path=search_par_path,
                    tim_path=prefix.tim_path,
                    cluster_manifest_path=prefix.cluster_manifest_path,
                    branches=preliminary_survivors,
                    certifications=parent_certifications,
                    theta_names=resolved_theta_names,
                    gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                    reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                    clock_override_path=clock_override_path,
                )
                _write_stage_progress(
                    stage_dir,
                    "repair-payloads-ready",
                    {
                        "stage": stage_index,
                        "round": int(repair_round + 1),
                        "parent_count": len(preliminary_survivors),
                        "payload_count": len(repair_payloads),
                    },
                )
                round_candidates, round_decision = _generate_residual_repair_candidates(
                    parent_branches=preliminary_survivors,
                    certifications=parent_certifications,
                    payloads=repair_payloads,
                    theta_names=resolved_theta_names,
                    parameter_scales=parameter_scales,
                    config=active_config,
                )
                _write_stage_progress(
                    stage_dir,
                    "repair-candidates-ready",
                    {
                        "stage": stage_index,
                        "round": int(repair_round + 1),
                        "candidate_count": len(round_candidates),
                        "decision": round_decision,
                    },
                )
                deduped_round: list[ActiveTimingBranch] = []
                for repair_index, repair_branch in enumerate(round_candidates):
                    key = (
                        repair_branch.wraps,
                        tuple(float(f"{value:.18e}") for value in np.asarray(repair_branch.theta_offset, dtype=float)),
                    )
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped_round.append(
                        replace(
                            repair_branch,
                            branch_id=(
                                f"stage_{stage_index:02d}_repair"
                                f"_r{repair_round + 1:02d}_{len(repair_candidates) + len(deduped_round):04d}"
                            ),
                        )
                    )
                if not deduped_round:
                    _write_stage_progress(
                        stage_dir,
                        "repair-round-no-new-candidates",
                        {
                            "stage": stage_index,
                            "round": int(repair_round + 1),
                            "parent_branch_ids": sorted(parent_ids),
                        },
                    )
                    repair_decision["rounds"].append(
                        {
                            "round": int(repair_round + 1),
                            "status": "no-new-candidates",
                            "parent_branch_ids": sorted(parent_ids),
                            "survivor_probe_decision": survivor_probe_decision,
                            "proposal": round_decision,
                        }
                    )
                    break
                _write_stage_progress(
                    stage_dir,
                    "repair-certification-start",
                    {
                        "stage": stage_index,
                        "round": int(repair_round + 1),
                        "candidate_count": len(deduped_round),
                        "parallel_workers": int(active_config.certify_parallel_workers),
                        "parallel_backend": active_config.certify_parallel_backend,
                    },
                )
                round_certifications = _certify_branches(
                    branches=deduped_round,
                    certifier=certifier,
                    max_workers=int(active_config.certify_parallel_workers),
                    parallel_backend=active_config.certify_parallel_backend,
                )
                _write_stage_progress(
                    stage_dir,
                    "repair-certification-ready",
                    {
                        "stage": stage_index,
                        "round": int(repair_round + 1),
                        "certification_count": len(round_certifications),
                    },
                )
                normalized_round = [
                    _normalize_branch_to_certification(
                        branch,
                        certification,
                        base_parameter_values=base_values,
                        parameter_scales=parameter_scales,
                    )
                    for branch, certification in zip(deduped_round, round_certifications, strict=True)
                ]
                repair_candidates.extend(deduped_round)
                certifications.extend(round_certifications)
                normalized.extend(normalized_round)
                round_best_branch, round_best_certification = _best_pair(normalized, certifications)
                repair_decision["rounds"].append(
                    {
                        "round": int(repair_round + 1),
                        "status": "certified",
                        "parent_branch_ids": sorted(parent_ids),
                        "new_candidate_count": len(deduped_round),
                        "survivor_probe_decision": survivor_probe_decision,
                        "proposal": round_decision,
                        "best_branch_id": (
                            round_best_branch.branch_id
                            if round_best_branch is not None
                            else None
                        ),
                        "best_reduced_chi2": (
                            round_best_certification.reduced_chi2
                            if round_best_certification is not None
                            else None
                        ),
                        "best_weighted_rms_us": (
                            round_best_certification.weighted_rms_us
                            if round_best_certification is not None
                            else None
                        ),
                    }
                )
                if _quality_passed(round_best_certification, active_config):
                    break
            repair_decision["certified_repair_count"] = len(repair_candidates)
            if bool(active_config.session_reconnect_enabled):
                normalized = score_with_lookahead(normalized)
                _write_stage_progress(
                    stage_dir,
                    "session-reconnect-start",
                    {
                        "stage": stage_index,
                        "normalized_count": len(normalized),
                        "certification_count": len(certifications),
                    },
                )
                preliminary_survivors, session_parent_decision = _select_stage_survivors(
                    normalized,
                    certifications,
                    replace(
                        active_config,
                        beam_size=max(int(active_config.session_reconnect_parent_top_k), 1),
                    ),
                )
                parent_ids = {branch.branch_id for branch in preliminary_survivors}
                parent_certifications = [
                    certification
                    for certification in certifications
                    if certification.branch_id in parent_ids
                ]
                if preliminary_survivors and parent_certifications:
                    session_payloads = _build_repair_payloads(
                        par_path=search_par_path,
                        tim_path=prefix.tim_path,
                        cluster_manifest_path=prefix.cluster_manifest_path,
                        branches=preliminary_survivors,
                        certifications=parent_certifications,
                        theta_names=resolved_theta_names,
                        gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                        reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                        clock_override_path=clock_override_path,
                    )
                    _write_stage_progress(
                        stage_dir,
                        "session-reconnect-payloads-ready",
                        {
                            "stage": stage_index,
                            "parent_count": len(preliminary_survivors),
                            "payload_count": len(session_payloads),
                        },
                    )
                    round_session_candidates, session_decision = _generate_session_reconnect_candidates(
                        parent_branches=preliminary_survivors,
                        certifications=parent_certifications,
                        payloads=session_payloads,
                        theta_names=resolved_theta_names,
                        parameter_scales=parameter_scales,
                        config=active_config,
                    )
                    _write_stage_progress(
                        stage_dir,
                        "session-reconnect-candidates-ready",
                        {
                            "stage": stage_index,
                            "candidate_count": len(round_session_candidates),
                            "decision": session_decision,
                        },
                    )
                    deduped_session: list[ActiveTimingBranch] = []
                    for session_branch in round_session_candidates:
                        key = (
                            session_branch.wraps,
                            tuple(
                                float(f"{value:.18e}")
                                for value in np.asarray(session_branch.theta_offset, dtype=float)
                            ),
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        deduped_session.append(
                            replace(
                                session_branch,
                                branch_id=(
                                    f"stage_{stage_index:02d}_session"
                                    f"_{len(session_candidates) + len(deduped_session):04d}"
                                ),
                            )
                        )
                    if deduped_session:
                        _write_stage_progress(
                            stage_dir,
                            "session-reconnect-certification-start",
                            {
                                "stage": stage_index,
                                "candidate_count": len(deduped_session),
                                "parallel_workers": int(active_config.certify_parallel_workers),
                                "parallel_backend": active_config.certify_parallel_backend,
                            },
                        )
                        session_certifications = _certify_branches(
                            branches=deduped_session,
                            certifier=certifier,
                            max_workers=int(active_config.certify_parallel_workers),
                            parallel_backend=active_config.certify_parallel_backend,
                        )
                        _write_stage_progress(
                            stage_dir,
                            "session-reconnect-certification-ready",
                            {
                                "stage": stage_index,
                                "certification_count": len(session_certifications),
                            },
                        )
                        normalized_session = [
                            _normalize_branch_to_certification(
                                branch,
                                certification,
                                base_parameter_values=base_values,
                                parameter_scales=parameter_scales,
                            )
                            for branch, certification in zip(
                                deduped_session,
                                session_certifications,
                                strict=True,
                            )
                        ]
                        session_candidates.extend(deduped_session)
                        certifications.extend(session_certifications)
                        normalized.extend(normalized_session)
                    session_best_branch, session_best_certification = _best_pair(normalized, certifications)
                    repair_decision["session_reconnect"] = {
                        "enabled": True,
                        "parent_branch_ids": sorted(parent_ids),
                        "new_candidate_count": len(deduped_session),
                        "parent_decision": session_parent_decision,
                        "proposal": session_decision,
                        "best_branch_id": (
                            session_best_branch.branch_id
                            if session_best_branch is not None
                            else None
                        ),
                        "best_reduced_chi2": (
                            session_best_certification.reduced_chi2
                            if session_best_certification is not None
                            else None
                        ),
                        "best_weighted_rms_us": (
                            session_best_certification.weighted_rms_us
                            if session_best_certification is not None
                            else None
                        ),
                    }
                else:
                    _write_stage_progress(
                        stage_dir,
                        "session-reconnect-no-parents",
                        {
                            "stage": stage_index,
                            "parent_decision": session_parent_decision,
                        },
                    )
                    repair_decision["session_reconnect"] = {
                        "enabled": True,
                        "new_candidate_count": 0,
                        "reason": "no-parents",
                        "parent_decision": session_parent_decision,
                    }
            else:
                repair_decision["session_reconnect"] = {"enabled": False}
            if bool(active_config.uncertainty_resample_enabled):
                normalized = score_with_lookahead(normalized)
                _write_stage_progress(
                    stage_dir,
                    "uncertainty-resample-start",
                    {
                        "stage": stage_index,
                        "normalized_count": len(normalized),
                        "certification_count": len(certifications),
                    },
                )
                preliminary_survivors, uncertainty_parent_decision = _select_stage_survivors(
                    normalized,
                    certifications,
                    replace(
                        active_config,
                        beam_size=max(int(active_config.uncertainty_resample_parent_top_k), 1),
                    ),
                )
                parent_ids = {branch.branch_id for branch in preliminary_survivors}
                parent_certifications = [
                    certification
                    for certification in certifications
                    if certification.branch_id in parent_ids
                ]
                if preliminary_survivors and parent_certifications:
                    uncertainty_payloads = _build_repair_payloads(
                        par_path=search_par_path,
                        tim_path=prefix.tim_path,
                        cluster_manifest_path=prefix.cluster_manifest_path,
                        branches=preliminary_survivors,
                        certifications=parent_certifications,
                        theta_names=resolved_theta_names,
                        gap_jacobian=np.asarray(geometry.gap_jacobian, dtype=float),
                        reference_phase_residuals=np.asarray(geometry.reference_phase_residuals, dtype=float),
                        clock_override_path=clock_override_path,
                    )
                    _write_stage_progress(
                        stage_dir,
                        "uncertainty-payloads-ready",
                        {
                            "stage": stage_index,
                            "parent_count": len(preliminary_survivors),
                            "payload_count": len(uncertainty_payloads),
                        },
                    )
                    round_uncertainty_candidates, uncertainty_decision = _generate_uncertain_gap_resample_candidates(
                        parent_branches=preliminary_survivors,
                        certifications=parent_certifications,
                        payloads=uncertainty_payloads,
                        theta_names=resolved_theta_names,
                        parameter_scales=parameter_scales,
                        config=active_config,
                    )
                    _write_stage_progress(
                        stage_dir,
                        "uncertainty-candidates-ready",
                        {
                            "stage": stage_index,
                            "candidate_count": len(round_uncertainty_candidates),
                            "decision": uncertainty_decision,
                        },
                    )
                    deduped_uncertainty: list[ActiveTimingBranch] = []
                    for uncertainty_branch in round_uncertainty_candidates:
                        key = (
                            uncertainty_branch.wraps,
                            tuple(
                                float(f"{value:.18e}")
                                for value in np.asarray(uncertainty_branch.theta_offset, dtype=float)
                            ),
                        )
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        deduped_uncertainty.append(
                            replace(
                                uncertainty_branch,
                                branch_id=(
                                    f"stage_{stage_index:02d}_uncertain"
                                    f"_{len(uncertainty_candidates) + len(deduped_uncertainty):04d}"
                                ),
                            )
                        )
                    if deduped_uncertainty:
                        _write_stage_progress(
                            stage_dir,
                            "uncertainty-certification-start",
                            {
                                "stage": stage_index,
                                "candidate_count": len(deduped_uncertainty),
                                "parallel_workers": int(active_config.certify_parallel_workers),
                                "parallel_backend": active_config.certify_parallel_backend,
                            },
                        )
                        uncertainty_certifications = _certify_branches(
                            branches=deduped_uncertainty,
                            certifier=certifier,
                            max_workers=int(active_config.certify_parallel_workers),
                            parallel_backend=active_config.certify_parallel_backend,
                        )
                        _write_stage_progress(
                            stage_dir,
                            "uncertainty-certification-ready",
                            {
                                "stage": stage_index,
                                "certification_count": len(uncertainty_certifications),
                            },
                        )
                        normalized_uncertainty = [
                            _normalize_branch_to_certification(
                                branch,
                                certification,
                                base_parameter_values=base_values,
                                parameter_scales=parameter_scales,
                            )
                            for branch, certification in zip(
                                deduped_uncertainty,
                                uncertainty_certifications,
                                strict=True,
                            )
                        ]
                        uncertainty_candidates.extend(deduped_uncertainty)
                        certifications.extend(uncertainty_certifications)
                        normalized.extend(normalized_uncertainty)
                    uncertainty_best_branch, uncertainty_best_certification = _best_pair(normalized, certifications)
                    repair_decision["uncertainty_resample"] = {
                        "enabled": True,
                        "parent_branch_ids": sorted(parent_ids),
                        "new_candidate_count": len(deduped_uncertainty),
                        "parent_decision": uncertainty_parent_decision,
                        "proposal": uncertainty_decision,
                        "best_branch_id": (
                            uncertainty_best_branch.branch_id
                            if uncertainty_best_branch is not None
                            else None
                        ),
                        "best_reduced_chi2": (
                            uncertainty_best_certification.reduced_chi2
                            if uncertainty_best_certification is not None
                            else None
                        ),
                        "best_weighted_rms_us": (
                            uncertainty_best_certification.weighted_rms_us
                            if uncertainty_best_certification is not None
                            else None
                        ),
                    }
                else:
                    _write_stage_progress(
                        stage_dir,
                        "uncertainty-resample-no-parents",
                        {
                            "stage": stage_index,
                            "parent_decision": uncertainty_parent_decision,
                        },
                    )
                    repair_decision["uncertainty_resample"] = {
                        "enabled": True,
                        "new_candidate_count": 0,
                        "reason": "no-parents",
                        "parent_decision": uncertainty_parent_decision,
                    }
        parameter_release_decision: dict[str, Any] = {
            "enabled": False,
            "reason": "retired-from-apollo-main-path",
            "fit_parameter_names": list(initial_fit_parameter_names),
        }
        normalized = score_with_lookahead(normalized)
        survivors, survivor_decision = _select_stage_survivors(
            normalized,
            certifications,
            active_config,
        )
        _write_stage_progress(
            stage_dir,
            "survivors-ready",
            {
                "stage": stage_index,
                "survivor_count": len(survivors),
                "certification_count": len(certifications),
                "decision": survivor_decision,
            },
        )
        current_branches = survivors
        all_certifications = certifications
        best_branch, best_certification = _best_pair(survivors, certifications)
        best_par_artifact = _write_stage_best_artifacts(
            output_dir=output_dir,
            stage_index=stage_index,
            active_par_path=active_par_path,
            fit_parameter_names=resolved_fit_parameter_names,
            branch=best_branch,
            certification=best_certification,
            tim_path=tim_path,
            clock_override_path=clock_override_path,
            stable_pulse_number_reference_path=search_par_path,
        )
        record = _stage_record(
            stage_index=stage_index,
            cluster_count=cluster_count,
            candidates=(
                list(candidates)
                + repair_candidates
                + session_candidates
                + uncertainty_candidates
                + parameter_release_candidates
            ),
            certified=list(selected_for_certification)
            + repair_candidates
            + session_candidates
            + uncertainty_candidates
            + parameter_release_candidates,
            certifications=certifications,
            survivors=survivors,
            certification_decision=certification_decision,
            survivor_decision=survivor_decision,
            repair_decision=repair_decision,
            parameter_release_decision=parameter_release_decision,
            active_par_path=active_par_path,
            fit_parameter_names=resolved_fit_parameter_names,
            best_par_artifact=best_par_artifact,
        )
        stage_records.append(record)
        _write_json(output_dir / f"stage_{stage_index:02d}.json", record)
        _write_json(output_dir / f"stage_{stage_index:02d}_survivors.json", survivors)
        _write_json(output_dir / f"stage_{stage_index:02d}_certifications.json", certifications)
        if _quality_passed(best_certification, active_config):
            break

    final_parameter_release_decision: dict[str, Any] = {
        "enabled": bool(active_config.parameter_release_enabled),
        "selected": False,
        "reason": "disabled",
        "candidate_blocks": [
            list(block)
            for block in _normalise_parameter_release_blocks(
                active_config.parameter_release_candidate_blocks
            )
        ],
    }
    if bool(active_config.parameter_release_enabled):
        comparison_stage_index = (
            int(stage_records[-1]["stage"])
            if stage_records
            else int(end_cluster_count)
        )
        comparison_dir = output_dir / "final_model_comparison"
        comparison_prefix = write_prefix_timing_inputs(
            tim_path=search_tim_path,
            cluster_manifest_path=cluster_manifest_path,
            prefix_cluster_count=end_cluster_count,
            output_dir=comparison_dir / "input",
        )
        parameter_release_result = _run_parameter_release_probe(
            par_path=search_par_path,
            prefix=comparison_prefix,
            stage_dir=comparison_dir,
            stage_index=comparison_stage_index,
            branches=current_branches,
            certifications=all_certifications,
            theta_names=resolved_theta_names,
            fit_parameter_names=resolved_fit_parameter_names,
            config=active_config,
            clock_override_path=clock_override_path,
        )
        final_parameter_release_decision = {
            "enabled": True,
            "selected": bool(parameter_release_result.get("selected")),
            **dict(parameter_release_result.get("decision") or {}),
        }
        if bool(parameter_release_result.get("selected")):
            current_branches = list(parameter_release_result["branches"])
            all_certifications = list(parameter_release_result["certifications"])
            active_par_path = Path(parameter_release_result["par_path"])
            resolved_theta_names = list(parameter_release_result["theta_names"])
            resolved_fit_parameter_names = (
                list(parameter_release_result["fit_parameter_names"])
                if parameter_release_result["fit_parameter_names"] is not None
                else None
            )
        _write_json(
            comparison_dir / "model_comparison.json",
            final_parameter_release_decision,
        )
        if stage_records:
            stage_records[-1] = {
                **stage_records[-1],
                "parameter_release_decision": final_parameter_release_decision,
            }
            _write_json(
                output_dir / f"stage_{comparison_stage_index:02d}.json",
                stage_records[-1],
            )

    best_branch, best_certification = _best_pair(current_branches, all_certifications)
    result = StagedSearchResult(
        branches=current_branches,
        certifications=all_certifications,
        best_branch=best_branch,
        best_certification=best_certification,
        stage_records=stage_records,
        output_dir=output_dir,
        active_par_path=active_par_path,
        fit_parameter_names=(
            tuple(resolved_fit_parameter_names)
            if resolved_fit_parameter_names is not None
            else None
        ),
    )
    _write_json(output_dir / "staged_search_summary.json", {
        "input_par_path": par_path,
        "active_par_path": active_par_path,
        "pulse_number_reference": stable_pulse_number_reference.metadata,
        "theta_names": resolved_theta_names,
        "fit_parameter_names": resolved_fit_parameter_names,
        "information_rank_start": information_rank_start_decision,
        "final_parameter_release": final_parameter_release_decision,
        "stage_count": len(stage_records),
        "best_branch": best_branch,
        "best_certification": best_certification,
        "stages": stage_records,
    })
    _write_json(output_dir / "active_branches.json", current_branches)
    _write_json(output_dir / "branch_certifications.json", all_certifications)
    return result
