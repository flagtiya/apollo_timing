from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from astropy import units as u
from pint.residuals import Residuals
import yaml

from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
from apollo_timing.inference.adaptive_residual_bayesian_lattice import (
    _cluster_median_design_matrix,
    _set_model_parameter_value,
    uses_seed_pulse_number_reference,
)
from apollo_timing.inference.active_timing_contracts import (
    ActiveTimingBranch,
    BranchCertification,
    to_json_ready,
)
from apollo_timing.inference.alias_branch_certifier import (
    _build_wrapped_toas,
    build_alias_branch_certifier,
)
from apollo_timing.inference.certifier_pool import certify_branches
from apollo_timing.inference.cluster_manifest import (
    load_cluster_manifest,
    resolve_cluster_manifest,
)
from apollo_timing.inference.prior_transform import extract_fit_flagged_parameters
from apollo_timing.inference.staged_budgeted_binary_search import (
    write_prefix_timing_inputs,
)


@dataclass(frozen=True)
class TransitionResidualMetrics:
    global_weighted_rms_us: float
    history_weighted_rms_us: float
    new_cluster_weighted_rms_us: float
    new_cluster_median_phase: float
    cluster_metrics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TransitionAuditRecord:
    branch_id: str
    new_wrap: int
    global_weighted_rms_us: float | None
    history_weighted_rms_us: float | None
    new_cluster_weighted_rms_us: float | None
    history_degradation_ratio: float | None
    predictive_new_cluster_phase: float | None
    predictive_phase_sigma: float | None
    predictive_z: float | None
    parameter_mahalanobis2: float | None
    fitted_theta: dict[str, float]
    certification_failed: bool
    notes: tuple[str, ...] = ()
    prefit_global_weighted_rms_us: float | None = None
    prefit_history_weighted_rms_us: float | None = None
    prefit_new_cluster_weighted_rms_us: float | None = None
    reduced_chi2: float | None = None
    chi2: float | None = None
    postfit_new_cluster_phase: float | None = None
    prefit_cluster_metrics: tuple[dict[str, Any], ...] = ()
    postfit_cluster_metrics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TransitionAuditRunResult:
    output_dir: Path
    parent_branch: ActiveTimingBranch
    parent_certification: BranchCertification
    records: list[TransitionAuditRecord]
    summary: dict[str, Any]
    json_path: Path
    markdown_path: Path


def build_direct_extension_candidates(
    *,
    parent: ActiveTimingBranch,
    new_wrap_values: Iterable[int],
) -> list[ActiveTimingBranch]:
    candidates: list[ActiveTimingBranch] = []
    for index, new_wrap in enumerate(new_wrap_values):
        diagnostics = dict(parent.diagnostics or {})
        diagnostics.update(
            {
                "proposal_family": "transition-causal-audit",
                "audit_parent_branch_id": str(parent.branch_id),
                "audit_new_wrap": int(new_wrap),
                "audit_preserves_history_wraps": True,
            }
        )
        candidates.append(
            replace(
                parent,
                branch_id=f"transition_audit_{index:04d}_w{int(new_wrap):+d}",
                wraps=tuple(int(item) for item in parent.wraps) + (int(new_wrap),),
                theta_offset=np.asarray(parent.theta_offset, dtype=float).copy(),
                proposal_source=parent.proposal_source,
                diagnostics=diagnostics,
            )
        )
    return candidates


def _finite_or_inf(value: float | None) -> float:
    if value is None or not np.isfinite(float(value)):
        return float("inf")
    return float(value)


def _parameter_mahalanobis2(
    *,
    parameter_names: Sequence[str],
    parent_certification: BranchCertification,
    fitted_theta: dict[str, float],
) -> float | None:
    names = tuple(str(name).upper() for name in parameter_names)
    if not names or any(name not in parent_certification.fitted_theta for name in names):
        return None
    if any(name not in fitted_theta for name in names):
        return None
    covariance = np.asarray(parent_certification.covariance, dtype=float)
    if covariance.shape != (len(names), len(names)) or not np.all(np.isfinite(covariance)):
        return None
    delta = np.asarray(
        [
            float(fitted_theta[name]) - float(parent_certification.fitted_theta[name])
            for name in names
        ],
        dtype=float,
    )
    return float(delta @ np.linalg.pinv(covariance) @ delta)


def audit_direct_extensions(
    *,
    parent: ActiveTimingBranch,
    parent_certification: BranchCertification,
    new_wrap_values: Iterable[int],
    certifier: Callable[[Any], Any],
    residual_evaluator: Callable[
        [ActiveTimingBranch, dict[str, float]],
        TransitionResidualMetrics,
    ],
    parent_history_weighted_rms_us: float,
    predictive_phase_sigma: float | None,
    acceptance_wrms_us: float,
    max_history_degradation_ratio: float,
    max_workers: int = 1,
    parallel_backend: str = "thread",
) -> tuple[list[TransitionAuditRecord], dict[str, Any]]:
    candidates = build_direct_extension_candidates(
        parent=parent,
        new_wrap_values=new_wrap_values,
    )
    certifications = certify_branches(
        branches=candidates,
        certifier=certifier,
        max_workers=max_workers,
        parallel_backend=parallel_backend,
        capture_failures=True,
    )
    certification_by_id = {item.branch_id: item for item in certifications}
    parent_history_wrms = max(float(parent_history_weighted_rms_us), 1.0e-12)
    sigma = (
        max(float(predictive_phase_sigma), 1.0e-12)
        if predictive_phase_sigma is not None and np.isfinite(float(predictive_phase_sigma))
        else None
    )
    records: list[TransitionAuditRecord] = []
    for candidate in candidates:
        certification = certification_by_id[candidate.branch_id]
        failed = certification.backend == "certification-failed"
        if failed:
            records.append(
                TransitionAuditRecord(
                    branch_id=candidate.branch_id,
                    new_wrap=int(candidate.wraps[-1]),
                    global_weighted_rms_us=None,
                    history_weighted_rms_us=None,
                    new_cluster_weighted_rms_us=None,
                    history_degradation_ratio=None,
                    predictive_new_cluster_phase=None,
                    predictive_phase_sigma=sigma,
                    predictive_z=None,
                    parameter_mahalanobis2=None,
                    fitted_theta={},
                    certification_failed=True,
                    notes=tuple(certification.notes or ()),
                    prefit_global_weighted_rms_us=None,
                    prefit_history_weighted_rms_us=None,
                    prefit_new_cluster_weighted_rms_us=None,
                    reduced_chi2=certification.reduced_chi2,
                    chi2=certification.chi2,
                )
            )
            continue
        prefit = residual_evaluator(candidate, parent_certification.fitted_theta)
        postfit = residual_evaluator(candidate, certification.fitted_theta)
        predictive_phase = float(prefit.new_cluster_median_phase)
        records.append(
            TransitionAuditRecord(
                branch_id=candidate.branch_id,
                new_wrap=int(candidate.wraps[-1]),
                global_weighted_rms_us=float(postfit.global_weighted_rms_us),
                history_weighted_rms_us=float(postfit.history_weighted_rms_us),
                new_cluster_weighted_rms_us=float(postfit.new_cluster_weighted_rms_us),
                history_degradation_ratio=float(postfit.history_weighted_rms_us)
                / parent_history_wrms,
                predictive_new_cluster_phase=predictive_phase,
                predictive_phase_sigma=sigma,
                predictive_z=(predictive_phase / sigma if sigma is not None else None),
                parameter_mahalanobis2=_parameter_mahalanobis2(
                    parameter_names=parent.theta_names,
                    parent_certification=parent_certification,
                    fitted_theta=certification.fitted_theta,
                ),
                fitted_theta={
                    str(name).upper(): float(value)
                    for name, value in certification.fitted_theta.items()
                },
                certification_failed=False,
                notes=tuple(certification.notes or ()),
                prefit_global_weighted_rms_us=float(prefit.global_weighted_rms_us),
                prefit_history_weighted_rms_us=float(prefit.history_weighted_rms_us),
                prefit_new_cluster_weighted_rms_us=float(
                    prefit.new_cluster_weighted_rms_us
                ),
                reduced_chi2=certification.reduced_chi2,
                chi2=certification.chi2,
                postfit_new_cluster_phase=float(postfit.new_cluster_median_phase),
                prefit_cluster_metrics=tuple(prefit.cluster_metrics),
                postfit_cluster_metrics=tuple(postfit.cluster_metrics),
            )
        )
    return records, summarize_transition_audit(
        records,
        acceptance_wrms_us=acceptance_wrms_us,
        max_history_degradation_ratio=max_history_degradation_ratio,
    )


def summarize_transition_audit(
    records: Sequence[TransitionAuditRecord],
    *,
    acceptance_wrms_us: float,
    max_history_degradation_ratio: float,
) -> dict[str, Any]:
    successful = [record for record in records if not record.certification_failed]
    ranked = sorted(
        successful,
        key=lambda record: (
            _finite_or_inf(record.global_weighted_rms_us),
            _finite_or_inf(record.history_degradation_ratio),
            abs(int(record.new_wrap)),
            str(record.branch_id),
        ),
    )
    viable = [
        record
        for record in ranked
        if _finite_or_inf(record.global_weighted_rms_us) <= float(acceptance_wrms_us)
        and _finite_or_inf(record.history_degradation_ratio)
        <= float(max_history_degradation_ratio)
    ]
    history_breaking_local_fits = [
        record
        for record in successful
        if _finite_or_inf(record.new_cluster_weighted_rms_us)
        < _finite_or_inf(record.history_weighted_rms_us)
        and _finite_or_inf(record.history_degradation_ratio)
        > float(max_history_degradation_ratio)
    ]
    best_global = ranked[0] if ranked else None
    best_viable = viable[0] if viable else None
    predictive_ranked = sorted(
        successful,
        key=lambda record: (
            abs(float(record.predictive_z))
            if record.predictive_z is not None and np.isfinite(float(record.predictive_z))
            else float("inf"),
            abs(int(record.new_wrap)),
            str(record.branch_id),
        ),
    )
    best_predictive = predictive_ranked[0] if predictive_ranked else None
    predictive_global_agreement = bool(
        best_predictive is not None
        and best_global is not None
        and int(best_predictive.new_wrap) == int(best_global.new_wrap)
    )
    if viable:
        diagnosis = "viable-direct-extension-found"
    elif (
        best_global is not None
        and _finite_or_inf(best_global.history_degradation_ratio)
        > float(max_history_degradation_ratio)
    ):
        diagnosis = "parent-basin-not-extendable"
    else:
        diagnosis = "no-viable-direct-extension-in-enumerated-new-gap-range"
    return {
        "candidate_count": len(records),
        "successful_certification_count": len(successful),
        "failed_certification_count": len(records) - len(successful),
        "viable_direct_extension_found": bool(viable),
        "viable_direct_extension_count": len(viable),
        "best_global_branch_id": best_global.branch_id if best_global else None,
        "best_global_new_wrap": best_global.new_wrap if best_global else None,
        "best_global_weighted_rms_us": (
            best_global.global_weighted_rms_us if best_global else None
        ),
        "best_viable_branch_id": best_viable.branch_id if best_viable else None,
        "best_viable_new_wrap": best_viable.new_wrap if best_viable else None,
        "best_predictive_branch_id": (
            best_predictive.branch_id if best_predictive else None
        ),
        "best_predictive_new_wrap": (
            best_predictive.new_wrap if best_predictive else None
        ),
        "best_predictive_z": (
            best_predictive.predictive_z if best_predictive else None
        ),
        "predictive_global_agreement": predictive_global_agreement,
        "history_breaking_local_fit_count": len(history_breaking_local_fits),
        "acceptance_wrms_us": float(acceptance_wrms_us),
        "max_history_degradation_ratio": float(max_history_degradation_ratio),
        "diagnosis": diagnosis,
        "ranked_records": [asdict(record) for record in ranked],
    }


def _format_optional(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def write_transition_audit_artifacts(
    *,
    output_dir: Path,
    metadata: dict[str, Any],
    records: Sequence[TransitionAuditRecord],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "transition_causal_audit.json"
    markdown_path = output_dir / "transition_causal_audit.md"
    json_path.write_text(
        json.dumps(
            to_json_ready(
                {
                    "metadata": metadata,
                    "summary": summary,
                    "records": list(records),
                }
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ranked = sorted(
        records,
        key=lambda record: (
            _finite_or_inf(record.global_weighted_rms_us),
            _finite_or_inf(record.history_degradation_ratio),
            abs(int(record.new_wrap)),
        ),
    )
    lines = [
        "# APOLLO Transition Causal Audit",
        "",
        f"Parent branch: `{metadata.get('parent_branch_id', 'unknown')}`",
        "",
        f"Diagnosis: `{summary.get('diagnosis', 'unknown')}`",
        "",
        f"Viable direct extension: `{summary.get('viable_direct_extension_found', False)}`",
        "",
        "| Rank | Branch | New wrap | Prefit history WRMS (us) | Global WRMS (us) | History WRMS (us) | New cluster WRMS (us) | History ratio | Predictive z | Parameter M2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(ranked, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    record.branch_id,
                    str(record.new_wrap),
                    _format_optional(record.prefit_history_weighted_rms_us),
                    _format_optional(record.global_weighted_rms_us),
                    _format_optional(record.history_weighted_rms_us),
                    _format_optional(record.new_cluster_weighted_rms_us),
                    _format_optional(record.history_degradation_ratio),
                    _format_optional(record.predictive_z),
                    _format_optional(record.parameter_mahalanobis2),
                ]
            )
            + " |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _branch_from_payload(payload: dict[str, Any]) -> ActiveTimingBranch:
    return ActiveTimingBranch(
        branch_id=str(payload["branch_id"]),
        structure_id=str(payload["structure_id"]),
        binary_model=(
            str(payload["binary_model"]).upper()
            if payload.get("binary_model") is not None
            else None
        ),
        wraps=tuple(int(item) for item in payload.get("wraps", ())),
        theta_names=tuple(str(item).upper() for item in payload.get("theta_names", ())),
        theta_offset=np.asarray(payload.get("theta_offset", ()), dtype=float),
        log_prior=float(payload.get("log_prior", 0.0)),
        proposal_source=str(payload.get("proposal_source", "artifact")),
        diagnostics=payload.get("diagnostics"),
    )


def _certification_from_payload(payload: dict[str, Any]) -> BranchCertification:
    return BranchCertification(
        branch_id=str(payload["branch_id"]),
        log_likelihood=float(payload.get("log_likelihood", float("-inf"))),
        fitted_theta={
            str(name).upper(): float(value)
            for name, value in (payload.get("fitted_theta") or {}).items()
        },
        covariance=np.asarray(payload.get("covariance", ()), dtype=float),
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
        backend=str(payload.get("backend", "artifact")),
        log_evidence=(
            float(payload["log_evidence"])
            if payload.get("log_evidence") is not None
            else None
        ),
        chi2=float(payload["chi2"]) if payload.get("chi2") is not None else None,
        notes=tuple(payload.get("notes") or ()),
    )


def _select_parent_from_artifacts(
    *,
    survivors_path: Path,
    certifications_path: Path,
    parent_branch_id: str | None,
) -> tuple[ActiveTimingBranch, BranchCertification]:
    branches = [
        _branch_from_payload(item)
        for item in json.loads(Path(survivors_path).read_text(encoding="utf-8"))
    ]
    certifications = [
        _certification_from_payload(item)
        for item in json.loads(Path(certifications_path).read_text(encoding="utf-8"))
    ]
    certification_by_id = {item.branch_id: item for item in certifications}
    eligible = [branch for branch in branches if branch.branch_id in certification_by_id]
    if parent_branch_id is not None:
        eligible = [branch for branch in eligible if branch.branch_id == str(parent_branch_id)]
    if not eligible:
        raise ValueError("No parent branch has both survivor and certification artifacts.")
    parent = min(
        eligible,
        key=lambda branch: (
            _finite_or_inf(certification_by_id[branch.branch_id].weighted_rms_us),
            _finite_or_inf(certification_by_id[branch.branch_id].reduced_chi2),
            branch.branch_id,
        ),
    )
    return parent, certification_by_id[parent.branch_id]


def _weighted_rms_us(
    time_residual_seconds: np.ndarray,
    error_seconds: np.ndarray,
    indices: np.ndarray,
) -> float:
    residual = np.asarray(time_residual_seconds, dtype=float)[indices]
    errors = np.asarray(error_seconds, dtype=float)[indices]
    finite = np.isfinite(residual) & np.isfinite(errors) & (errors > 0.0)
    if not np.any(finite):
        return float("nan")
    residual = residual[finite]
    weights = 1.0 / np.square(errors[finite])
    return float(np.sqrt(np.sum(weights * np.square(residual)) / np.sum(weights)) * 1.0e6)


def build_transition_residual_evaluator(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    history_cluster_count: int,
    clock_override_path: Path | None = None,
) -> Callable[[ActiveTimingBranch, dict[str, float]], TransitionResidualMetrics]:
    manifest = resolve_cluster_manifest(
        load_cluster_manifest(Path(cluster_manifest_path)),
        Path(tim_path),
    )
    if int(history_cluster_count) <= 0 or int(history_cluster_count) >= len(manifest.clusters):
        raise ValueError("history_cluster_count must leave at least one target cluster.")
    seed_model = _load_pint_model(Path(par_path))
    base_toas = _load_pint_toas(
        Path(tim_path),
        seed_model=seed_model,
        clock_override_path=clock_override_path,
    )
    error_seconds = np.asarray(base_toas.get_errors().to_value(u.s), dtype=float)

    def evaluate(
        branch: ActiveTimingBranch,
        fitted_theta: dict[str, float],
    ) -> TransitionResidualMetrics:
        model = deepcopy(seed_model)
        for name, value in fitted_theta.items():
            _set_model_parameter_value(model, str(name).upper(), float(value))
        pulse_number_model = seed_model if uses_seed_pulse_number_reference(branch) else model
        wrapped_toas = _build_wrapped_toas(
            base_toas,
            pulse_number_model,
            manifest,
            branch.wraps,
        )
        residuals = Residuals(
            wrapped_toas,
            model,
            use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        time_residual_seconds = np.asarray(residuals.time_resids.to_value(u.s), dtype=float)
        phase_residuals = np.asarray(
            residuals.phase_resids.to_value(u.dimensionless_unscaled),
            dtype=float,
        )
        history_indices = np.asarray(
            [
                index
                for cluster in manifest.clusters[: int(history_cluster_count)]
                for index in cluster.toa_indices
            ],
            dtype=int,
        )
        target_indices = np.asarray(
            [
                index
                for cluster in manifest.clusters[int(history_cluster_count) :]
                for index in cluster.toa_indices
            ],
            dtype=int,
        )
        cluster_metrics: list[dict[str, Any]] = []
        for cluster in manifest.clusters:
            indices = np.asarray(cluster.toa_indices, dtype=int)
            cluster_phase = phase_residuals[indices]
            median_phase = float(np.median(cluster_phase))
            cluster_metrics.append(
                {
                    "cluster_id": str(cluster.cluster_id),
                    "toa_count": int(indices.size),
                    "weighted_rms_us": _weighted_rms_us(
                        time_residual_seconds,
                        error_seconds,
                        indices,
                    ),
                    "chi2": float(
                        np.sum(
                            np.square(
                                time_residual_seconds[indices] / error_seconds[indices]
                            )
                        )
                    ),
                    "median_phase_residual": median_phase,
                    "mad_phase_residual": float(
                        np.median(np.abs(cluster_phase - median_phase))
                    ),
                }
            )
        return TransitionResidualMetrics(
            global_weighted_rms_us=float(residuals.rms_weighted().to_value(u.us)),
            history_weighted_rms_us=_weighted_rms_us(
                time_residual_seconds,
                error_seconds,
                history_indices,
            ),
            new_cluster_weighted_rms_us=_weighted_rms_us(
                time_residual_seconds,
                error_seconds,
                target_indices,
            ),
            new_cluster_median_phase=float(np.median(phase_residuals[target_indices])),
            cluster_metrics=tuple(cluster_metrics),
        )

    return evaluate


def _predictive_phase_sigma(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parameter_names: Sequence[str],
    parent_certification: BranchCertification,
    history_cluster_count: int,
    clock_override_path: Path | None,
    floor: float,
) -> float:
    covariance = np.asarray(parent_certification.covariance, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if covariance.shape != (len(names), len(names)):
        return max(float(floor), 1.0e-12)
    manifest = resolve_cluster_manifest(
        load_cluster_manifest(Path(cluster_manifest_path)),
        Path(tim_path),
    )
    model = _load_pint_model(Path(par_path))
    for name, value in parent_certification.fitted_theta.items():
        _set_model_parameter_value(model, str(name).upper(), float(value))
    toas = _load_pint_toas(
        Path(tim_path),
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    design = _cluster_median_design_matrix(
        model=model,
        toas=toas,
        manifest=manifest,
        parameter_names=names,
    )
    target_row = np.asarray(design[int(history_cluster_count)], dtype=float)
    variance = float(target_row @ covariance @ target_row)
    if not np.isfinite(variance) or variance < 0.0:
        return max(float(floor), 1.0e-12)
    return float(np.sqrt(variance + max(float(floor), 1.0e-12) ** 2))


def run_transition_causal_audit(
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parent_survivors_path: Path,
    parent_certifications_path: Path,
    output_dir: Path,
    parent_cluster_count: int,
    target_cluster_count: int | None = None,
    parent_branch_id: str | None = None,
    new_wrap_min: int = -32,
    new_wrap_max: int = 32,
    fit_parameter_names: Sequence[str] | str | None = None,
    maxiter: int = 120,
    acceptance_wrms_us: float = 50.0,
    max_history_degradation_ratio: float = 1.5,
    predictive_phase_floor: float = 0.05,
    max_workers: int = 1,
    parallel_backend: str = "thread",
    clock_override_path: Path | None = None,
) -> TransitionAuditRunResult:
    parent_count = int(parent_cluster_count)
    target_count = int(target_cluster_count or (parent_count + 1))
    if target_count != parent_count + 1:
        raise ValueError("Direct transition audit currently requires exactly one new cluster.")
    if int(new_wrap_min) > int(new_wrap_max):
        raise ValueError("new_wrap_min must not exceed new_wrap_max.")
    parent, parent_certification = _select_parent_from_artifacts(
        survivors_path=Path(parent_survivors_path),
        certifications_path=Path(parent_certifications_path),
        parent_branch_id=parent_branch_id,
    )
    if len(parent.wraps) != parent_count - 1:
        raise ValueError(
            f"Parent branch has {len(parent.wraps)} wraps; expected {parent_count - 1}."
        )
    parameter_names = tuple(str(name).upper() for name in parent.theta_names)
    if isinstance(fit_parameter_names, str):
        if fit_parameter_names.lower() != "fit_flags":
            raise ValueError(f"Unsupported fit parameter shorthand: {fit_parameter_names}")
        resolved_fit_names = tuple(extract_fit_flagged_parameters(Path(par_path)))
    elif fit_parameter_names is None:
        resolved_fit_names = parameter_names
    else:
        resolved_fit_names = tuple(str(name).upper() for name in fit_parameter_names)

    output_dir = Path(output_dir)
    inputs = write_prefix_timing_inputs(
        tim_path=Path(tim_path),
        cluster_manifest_path=Path(cluster_manifest_path),
        prefix_cluster_count=target_count,
        output_dir=output_dir / "target_inputs",
    )
    certifier = build_alias_branch_certifier(
        par_path=Path(par_path),
        tim_path=inputs.tim_path,
        cluster_manifest_path=inputs.cluster_manifest_path,
        parameter_names=parameter_names,
        fit_parameter_names=resolved_fit_names,
        maxiter=max(int(maxiter), 1),
        clock_override_path=clock_override_path,
    )
    evaluator = build_transition_residual_evaluator(
        par_path=Path(par_path),
        tim_path=inputs.tim_path,
        cluster_manifest_path=inputs.cluster_manifest_path,
        history_cluster_count=parent_count,
        clock_override_path=clock_override_path,
    )
    predictive_sigma = _predictive_phase_sigma(
        par_path=Path(par_path),
        tim_path=inputs.tim_path,
        cluster_manifest_path=inputs.cluster_manifest_path,
        parameter_names=parameter_names,
        parent_certification=parent_certification,
        history_cluster_count=parent_count,
        clock_override_path=clock_override_path,
        floor=predictive_phase_floor,
    )
    parent_wrms = parent_certification.weighted_rms_us
    if parent_wrms is None or not np.isfinite(float(parent_wrms)):
        raise ValueError("Parent certification requires a finite weighted_rms_us.")
    records, summary = audit_direct_extensions(
        parent=parent,
        parent_certification=parent_certification,
        new_wrap_values=range(int(new_wrap_min), int(new_wrap_max) + 1),
        certifier=certifier,
        residual_evaluator=evaluator,
        parent_history_weighted_rms_us=float(parent_wrms),
        predictive_phase_sigma=predictive_sigma,
        acceptance_wrms_us=acceptance_wrms_us,
        max_history_degradation_ratio=max_history_degradation_ratio,
        max_workers=max_workers,
        parallel_backend=parallel_backend,
    )
    metadata = {
        "par_path": Path(par_path),
        "tim_path": Path(tim_path),
        "cluster_manifest_path": Path(cluster_manifest_path),
        "parent_survivors_path": Path(parent_survivors_path),
        "parent_certifications_path": Path(parent_certifications_path),
        "parent_branch_id": parent.branch_id,
        "parent_wraps": list(parent.wraps),
        "parent_weighted_rms_us": float(parent_wrms),
        "parent_cluster_count": parent_count,
        "target_cluster_count": target_count,
        "new_wrap_min": int(new_wrap_min),
        "new_wrap_max": int(new_wrap_max),
        "parameter_names": list(parameter_names),
        "fit_parameter_names": list(resolved_fit_names),
        "predictive_phase_sigma": predictive_sigma,
        "maxiter": int(maxiter),
    }
    json_path, markdown_path = write_transition_audit_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        records=records,
        summary=summary,
    )
    return TransitionAuditRunResult(
        output_dir=output_dir,
        parent_branch=parent,
        parent_certification=parent_certification,
        records=records,
        summary=summary,
        json_path=json_path,
        markdown_path=markdown_path,
    )


def _resolve_config_path(base_dir: Path, value: Any) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def run_transition_causal_audit_from_config(
    *,
    config_path: Path,
    output_dir: Path | None = None,
) -> TransitionAuditRunResult:
    config_path = Path(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_dir = config_path.parent
    path_names = (
        "par_path",
        "tim_path",
        "cluster_manifest_path",
        "parent_survivors_path",
        "parent_certifications_path",
    )
    missing = [name for name in path_names if not payload.get(name)]
    if missing:
        raise ValueError("Transition audit config requires: " + ", ".join(missing))
    resolved_output = (
        Path(output_dir)
        if output_dir is not None
        else _resolve_config_path(base_dir, payload.get("output_dir", "transition_causal_audit"))
    )
    clock_override = payload.get("clock_override_path")
    return run_transition_causal_audit(
        par_path=_resolve_config_path(base_dir, payload["par_path"]),
        tim_path=_resolve_config_path(base_dir, payload["tim_path"]),
        cluster_manifest_path=_resolve_config_path(base_dir, payload["cluster_manifest_path"]),
        parent_survivors_path=_resolve_config_path(base_dir, payload["parent_survivors_path"]),
        parent_certifications_path=_resolve_config_path(
            base_dir,
            payload["parent_certifications_path"],
        ),
        output_dir=resolved_output,
        parent_cluster_count=int(payload["parent_cluster_count"]),
        target_cluster_count=(
            int(payload["target_cluster_count"])
            if payload.get("target_cluster_count") is not None
            else None
        ),
        parent_branch_id=payload.get("parent_branch_id"),
        new_wrap_min=int(payload.get("new_wrap_min", -32)),
        new_wrap_max=int(payload.get("new_wrap_max", 32)),
        fit_parameter_names=payload.get("fit_parameter_names"),
        maxiter=int(payload.get("maxiter", 120)),
        acceptance_wrms_us=float(payload.get("acceptance_wrms_us", 50.0)),
        max_history_degradation_ratio=float(
            payload.get("max_history_degradation_ratio", 1.5)
        ),
        predictive_phase_floor=float(payload.get("predictive_phase_floor", 0.05)),
        max_workers=int(payload.get("max_workers", 1)),
        parallel_backend=str(payload.get("parallel_backend", "thread")),
        clock_override_path=(
            _resolve_config_path(base_dir, clock_override)
            if clock_override is not None
            else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit one APOLLO stage transition without heuristic branch pruning."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_transition_causal_audit_from_config(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(to_json_ready(result.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
