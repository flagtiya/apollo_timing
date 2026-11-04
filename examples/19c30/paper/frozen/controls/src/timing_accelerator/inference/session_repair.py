from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from timing_accelerator.inference.active_timing_contracts import (
    ActiveTimingBranch,
    BranchCertification,
    to_json_ready,
)
from timing_accelerator.inference.cluster_manifest import (
    load_cluster_manifest,
    resolve_cluster_manifest,
)
from timing_accelerator.inference.prior_transform import extract_fit_flagged_parameters
from timing_accelerator.models.contracts import ClusterManifest


@dataclass(frozen=True)
class ResidualSessionScore:
    cluster_index: int
    cluster_id: str
    source_filenames: list[str]
    toa_count: int
    mean_abs_residual_us: float
    max_abs_residual_us: float
    mjd_min: float
    mjd_max: float


@dataclass(frozen=True)
class SessionRepairConfig:
    max_sessions: int = 3
    wrap_radius: int = 2
    max_candidates: int = 64
    include_single_gap: bool = True
    include_session_shift: bool = True
    combine_sessions: bool = False
    wrap_delta_prior_scale: float = 8.0


@dataclass(frozen=True)
class SessionRepairRunResult:
    output_dir: Path
    source_run_dir: Path
    base_branch: ActiveTimingBranch
    base_certification: BranchCertification
    session_scores: list[ResidualSessionScore]
    repair_branches: list[ActiveTimingBranch]
    repair_certifications: list[BranchCertification]
    best_repair_certification: BranchCertification | None
    best_repair_par_path: Path | None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _resolve_relative_path(base_dir: Path, value: Any | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _required_path(base_dir: Path, payload: dict[str, Any], key: str) -> Path:
    path = _resolve_relative_path(base_dir, payload.get(key))
    if path is None:
        raise ValueError(f"Session repair requires '{key}' in the active timing config.")
    return path


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_active_branches(path: Path) -> list[ActiveTimingBranch]:
    branches: list[ActiveTimingBranch] = []
    for item in _load_json(path):
        branches.append(
            ActiveTimingBranch(
                branch_id=str(item["branch_id"]),
                structure_id=str(item["structure_id"]),
                binary_model=(
                    str(item["binary_model"])
                    if item.get("binary_model") is not None
                    else None
                ),
                wraps=tuple(int(value) for value in item.get("wraps", [])),
                theta_names=tuple(str(value).upper() for value in item.get("theta_names", [])),
                theta_offset=np.asarray(item.get("theta_offset", []), dtype=float),
                log_prior=float(item.get("log_prior", 0.0)),
                proposal_source=str(item.get("proposal_source", "unknown")),
                diagnostics=item.get("diagnostics"),
            )
        )
    return branches


def load_branch_certifications(path: Path) -> list[BranchCertification]:
    certifications: list[BranchCertification] = []
    for item in _load_json(path):
        certifications.append(
            BranchCertification(
                branch_id=str(item["branch_id"]),
                log_likelihood=float(item.get("log_likelihood", -1.0e300)),
                fitted_theta={
                    str(key).upper(): float(value)
                    for key, value in (item.get("fitted_theta") or {}).items()
                },
                covariance=np.asarray(item.get("covariance", []), dtype=float),
                reduced_chi2=_optional_float(item.get("reduced_chi2")),
                weighted_rms_us=_optional_float(item.get("weighted_rms_us")),
                backend=str(item.get("backend", "unknown")),
                log_evidence=_optional_float(item.get("log_evidence")),
                chi2=_optional_float(item.get("chi2")),
                notes=list(item.get("notes") or []),
            )
        )
    return certifications


def _finite_rms(certification: BranchCertification) -> float:
    value = certification.weighted_rms_us
    if value is None or not math.isfinite(float(value)):
        return float("inf")
    return float(value)


def best_branch_by_certified_rms(
    branches: Sequence[ActiveTimingBranch],
    certifications: Sequence[BranchCertification],
) -> tuple[ActiveTimingBranch, BranchCertification]:
    branch_by_id = {branch.branch_id: branch for branch in branches}
    pairs = [
        (branch_by_id[certification.branch_id], certification)
        for certification in certifications
        if certification.branch_id in branch_by_id
    ]
    if not pairs:
        raise ValueError("No branch has a matching certification.")
    return min(
        pairs,
        key=lambda item: (
            _finite_rms(item[1]),
            (
                float(item[1].reduced_chi2)
                if item[1].reduced_chi2 is not None
                else float("inf")
            ),
            -float(item[1].log_likelihood),
            item[0].branch_id,
        ),
    )


def _read_residual_rows(path: Path) -> list[dict[str, float]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float]] = []
        for row in reader:
            rows.append(
                {
                    "mjd": float(row["mjd"]),
                    "residual_us": float(row["residual_us"]),
                    "error_us": float(row["error_us"]),
                }
            )
    return rows


def rank_residual_sessions(
    residual_csv_path: Path,
    manifest: ClusterManifest,
    tim_path: Path,
) -> list[ResidualSessionScore]:
    resolved_manifest = resolve_cluster_manifest(manifest, Path(tim_path))
    residual_rows = _read_residual_rows(residual_csv_path)
    scores: list[ResidualSessionScore] = []
    for cluster_index, cluster in enumerate(resolved_manifest.clusters):
        if cluster.toa_indices is None:
            raise ValueError(f"Cluster '{cluster.cluster_id}' has no resolved TOA indices.")
        if any(index >= len(residual_rows) for index in cluster.toa_indices):
            raise ValueError(
                f"Residual CSV '{residual_csv_path}' has fewer rows than the cluster manifest expects."
            )
        residuals = np.asarray(
            [residual_rows[index]["residual_us"] for index in cluster.toa_indices],
            dtype=float,
        )
        mjds = np.asarray(
            [residual_rows[index]["mjd"] for index in cluster.toa_indices],
            dtype=float,
        )
        abs_residuals = np.abs(residuals)
        scores.append(
            ResidualSessionScore(
                cluster_index=int(cluster_index),
                cluster_id=str(cluster.cluster_id),
                source_filenames=list(cluster.source_filenames),
                toa_count=int(len(cluster.toa_indices)),
                mean_abs_residual_us=float(np.mean(abs_residuals)),
                max_abs_residual_us=float(np.max(abs_residuals)),
                mjd_min=float(np.min(mjds)),
                mjd_max=float(np.max(mjds)),
            )
        )
    return sorted(
        scores,
        key=lambda score: (
            -score.mean_abs_residual_us,
            -score.max_abs_residual_us,
            score.cluster_index,
        ),
    )


def adjacent_gap_indices_for_cluster(cluster_index: int, gap_count: int) -> list[int]:
    cluster_index = int(cluster_index)
    gap_count = int(gap_count)
    if cluster_index < 0:
        raise ValueError("cluster_index must be non-negative.")
    if gap_count < 0:
        raise ValueError("gap_count must be non-negative.")
    if cluster_index > gap_count:
        raise ValueError("cluster_index is outside the cluster_count implied by gap_count.")
    indices: list[int] = []
    if cluster_index > 0:
        indices.append(cluster_index - 1)
    if cluster_index < gap_count:
        indices.append(cluster_index)
    return indices


def _ordered_nonzero_deltas(radius: int) -> list[int]:
    active_radius = max(int(radius), 0)
    deltas: list[int] = []
    for magnitude in range(1, active_radius + 1):
        deltas.extend([magnitude, -magnitude])
    return deltas


def _repair_log_prior(
    base_log_prior: float,
    delta_by_gap: dict[int, int],
    prior_scale: float,
) -> float:
    scale = max(float(prior_scale), 1.0e-12)
    return float(base_log_prior) - sum(abs(int(delta)) for delta in delta_by_gap.values()) / scale


def _repair_diagnostics(
    base_branch: ActiveTimingBranch,
    score: ResidualSessionScore,
    session_rank: int,
    mode: str,
    delta_by_gap: dict[int, int],
    log_prior: float,
) -> dict[str, Any]:
    base_diagnostics = base_branch.diagnostics or {}
    diagnostics: dict[str, Any] = {}
    if "certification_theta_offset" in base_diagnostics:
        diagnostics["certification_theta_offset"] = base_diagnostics["certification_theta_offset"]
    diagnostics.update(
        {
            "proposal_score": float(log_prior),
            "source_branch_id": base_branch.branch_id,
            "source_proposal_source": base_branch.proposal_source,
            "target_session_rank": int(session_rank),
            "target_cluster_index": int(score.cluster_index),
            "target_cluster_id": score.cluster_id,
            "target_source_filenames": list(score.source_filenames),
            "target_mean_abs_residual_us": float(score.mean_abs_residual_us),
            "target_max_abs_residual_us": float(score.max_abs_residual_us),
            "repair_mode": mode,
            "changed_gap_indices": sorted(int(index) for index in delta_by_gap),
            "wrap_delta": {
                str(int(index)): int(delta)
                for index, delta in sorted(delta_by_gap.items())
            },
            "repair_abs_delta": int(sum(abs(int(delta)) for delta in delta_by_gap.values())),
        }
    )
    return diagnostics


def generate_session_repair_branches(
    base_branch: ActiveTimingBranch,
    session_scores: Sequence[ResidualSessionScore],
    config: SessionRepairConfig | None = None,
) -> list[ActiveTimingBranch]:
    active_config = config or SessionRepairConfig()
    base_wraps = tuple(int(value) for value in base_branch.wraps)
    gap_count = len(base_wraps)
    deltas = _ordered_nonzero_deltas(active_config.wrap_radius)
    seen_wraps: set[tuple[int, ...]] = {base_wraps}
    branches: list[ActiveTimingBranch] = []

    def append_candidate(
        score: ResidualSessionScore,
        session_rank: int,
        mode: str,
        delta_by_gap: dict[int, int],
        extra_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        if len(branches) >= max(int(active_config.max_candidates), 0):
            return
        wraps = list(base_wraps)
        for gap_index, delta in delta_by_gap.items():
            if gap_index < 0 or gap_index >= gap_count:
                raise ValueError(f"Gap index {gap_index} is outside the wrap vector.")
            wraps[gap_index] += int(delta)
        wrap_tuple = tuple(int(value) for value in wraps)
        if wrap_tuple in seen_wraps:
            return
        seen_wraps.add(wrap_tuple)
        log_prior = _repair_log_prior(
            base_branch.log_prior,
            delta_by_gap,
            active_config.wrap_delta_prior_scale,
        )
        diagnostics = _repair_diagnostics(
            base_branch=base_branch,
            score=score,
            session_rank=session_rank,
            mode=mode,
            delta_by_gap=delta_by_gap,
            log_prior=log_prior,
        )
        if extra_diagnostics:
            diagnostics.update(extra_diagnostics)
        branches.append(
            ActiveTimingBranch(
                branch_id=f"{base_branch.branch_id}_repair_{len(branches):04d}",
                structure_id=base_branch.structure_id,
                binary_model=base_branch.binary_model,
                wraps=wrap_tuple,
                theta_names=tuple(base_branch.theta_names),
                theta_offset=np.asarray(base_branch.theta_offset, dtype=float).copy(),
                log_prior=float(log_prior),
                proposal_source="session-repair",
                diagnostics=diagnostics,
            )
        )

    moves_by_session: list[tuple[int, ResidualSessionScore, list[dict[str, Any]]]] = []
    selected_scores = list(session_scores[: max(int(active_config.max_sessions), 0)])
    for session_rank, score in enumerate(selected_scores):
        adjacent_gaps = adjacent_gap_indices_for_cluster(score.cluster_index, gap_count)
        session_moves: list[dict[str, Any]] = []
        if active_config.include_single_gap:
            for gap_index in adjacent_gaps:
                for delta in deltas:
                    move = {
                        "mode": "single-gap",
                        "delta_by_gap": {int(gap_index): int(delta)},
                    }
                    session_moves.append(move)
                    append_candidate(
                        score=score,
                        session_rank=session_rank,
                        mode=move["mode"],
                        delta_by_gap=move["delta_by_gap"],
                    )
        if active_config.include_session_shift and 0 < score.cluster_index < gap_count:
            left_gap = int(score.cluster_index) - 1
            right_gap = int(score.cluster_index)
            for delta in deltas:
                move = {
                    "mode": "session-shift",
                    "delta_by_gap": {left_gap: int(delta), right_gap: -int(delta)},
                }
                session_moves.append(move)
                append_candidate(
                    score=score,
                    session_rank=session_rank,
                    mode=move["mode"],
                    delta_by_gap=move["delta_by_gap"],
                )
        if session_moves:
            moves_by_session.append((session_rank, score, session_moves))

    if active_config.combine_sessions and len(moves_by_session) >= 2:
        for combo_size in range(2, len(moves_by_session) + 1):
            for session_combo in combinations(moves_by_session, combo_size):
                move_lists = [item[2] for item in session_combo]
                for move_combo in product(*move_lists):
                    delta_by_gap: dict[int, int] = {}
                    coupled_repairs: list[dict[str, Any]] = []
                    for (session_rank, score, _), move in zip(session_combo, move_combo, strict=True):
                        move_delta = {
                            int(gap_index): int(delta)
                            for gap_index, delta in move["delta_by_gap"].items()
                        }
                        for gap_index, delta in move_delta.items():
                            delta_by_gap[gap_index] = delta_by_gap.get(gap_index, 0) + delta
                        coupled_repairs.append(
                            {
                                "target_session_rank": int(session_rank),
                                "target_cluster_index": int(score.cluster_index),
                                "target_cluster_id": score.cluster_id,
                                "target_source_filenames": list(score.source_filenames),
                                "repair_mode": str(move["mode"]),
                                "wrap_delta": {
                                    str(index): int(delta)
                                    for index, delta in sorted(move_delta.items())
                                },
                            }
                        )
                    delta_by_gap = {
                        gap_index: delta
                        for gap_index, delta in delta_by_gap.items()
                        if int(delta) != 0
                    }
                    if not delta_by_gap:
                        continue
                    first_session_rank, first_score, _ = session_combo[0]
                    append_candidate(
                        score=first_score,
                        session_rank=first_session_rank,
                        mode="coupled-session",
                        delta_by_gap=delta_by_gap,
                        extra_diagnostics={
                            "coupled_session_count": int(combo_size),
                            "coupled_repairs": coupled_repairs,
                        },
                    )
                    if len(branches) >= max(int(active_config.max_candidates), 0):
                        return branches
    return branches


def _resolve_backend_fit_parameter_names(
    backend_payload: dict[str, Any],
    base_dir: Path,
) -> list[str] | None:
    raw_names = backend_payload.get("fit_parameter_names")
    if raw_names is None:
        return None
    if isinstance(raw_names, str):
        if raw_names.lower() != "fit_flags":
            raise ValueError(f"Unsupported fit_parameter_names shorthand: {raw_names}")
        par_path = _resolve_relative_path(base_dir, backend_payload.get("par_path"))
        if par_path is None:
            raise ValueError("fit_parameter_names: fit_flags requires backend.par_path.")
        return [name.upper() for name in extract_fit_flagged_parameters(par_path)]
    return [str(name).upper() for name in raw_names]


def _failed_certification(
    branch: ActiveTimingBranch,
    parameter_count: int,
    exc: Exception,
) -> BranchCertification:
    dimension = max(int(parameter_count), 1)
    return BranchCertification(
        branch_id=branch.branch_id,
        log_likelihood=-1.0e300,
        fitted_theta={},
        covariance=np.eye(dimension, dtype=float),
        reduced_chi2=None,
        weighted_rms_us=None,
        backend="local-laplace",
        log_evidence=None,
        chi2=None,
        notes=[f"certification-error: {type(exc).__name__}: {exc}"],
    )


def certify_session_repair_branches(
    branches: Sequence[ActiveTimingBranch],
    *,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parameter_names: Sequence[str],
    fit_parameter_names: Sequence[str] | None = None,
    maxiter: int = 5,
    clock_override_path: Path | None = None,
) -> list[BranchCertification]:
    from timing_accelerator.inference.alias_branch_certifier import build_alias_branch_certifier
    from timing_accelerator.inference.branch_proposal import (
        alias_certification_to_branch_certification,
        branch_to_alias_candidate,
    )

    parameter_names = [str(name).upper() for name in parameter_names]
    certifier = build_alias_branch_certifier(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        parameter_names=parameter_names,
        fit_parameter_names=fit_parameter_names,
        maxiter=maxiter,
        clock_override_path=clock_override_path,
    )
    certifications: list[BranchCertification] = []
    for branch in branches:
        try:
            certifications.append(
                alias_certification_to_branch_certification(
                    certifier(branch_to_alias_candidate(branch))
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by real PINT edge cases
            certifications.append(_failed_certification(branch, len(parameter_names), exc))
    return certifications


def _best_certification_by_rms(
    certifications: Sequence[BranchCertification],
) -> BranchCertification | None:
    valid = [
        certification
        for certification in certifications
        if math.isfinite(_finite_rms(certification))
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda certification: (
            _finite_rms(certification),
            (
                float(certification.reduced_chi2)
                if certification.reduced_chi2 is not None
                else float("inf")
            ),
            -float(certification.log_likelihood),
            certification.branch_id,
        ),
    )


def _set_model_uncertainties_from_covariance(model, certification: BranchCertification) -> None:
    from astropy import units as u

    covariance = np.asarray(certification.covariance, dtype=float)
    if covariance.ndim != 2:
        return
    uncertainty_updates = {}
    for index, name in enumerate(certification.fitted_theta):
        if name not in model.params:
            continue
        if index >= covariance.shape[0] or index >= covariance.shape[1]:
            continue
        diagonal_entry = float(covariance[index, index])
        if not np.isfinite(diagonal_entry) or diagonal_entry <= 0.0:
            continue
        uncertainty_updates[name] = np.sqrt(diagonal_entry) * model[name].units
    if uncertainty_updates:
        model.set_param_uncertainties(uncertainty_updates)


def _write_solution_parfile(
    path: Path,
    seed_par_path: Path,
    certification: BranchCertification,
) -> Path:
    from timing_accelerator.backend.pint_backend import (
        _load_pint_model,
        _set_numeric_model_parameter_value,
    )

    model = _load_pint_model(seed_par_path)
    for name, value in certification.fitted_theta.items():
        if name not in model.params:
            continue
        _set_numeric_model_parameter_value(model, name, float(value))
    _set_model_uncertainties_from_covariance(model, certification)
    from timing_accelerator.inference.pint_solution_artifacts import (
        write_pint_model_parfile_roundtrip_safe,
    )

    return write_pint_model_parfile_roundtrip_safe(model=model, path=path)


def _write_best_repair_artifacts(
    output_dir: Path,
    best_certification: BranchCertification | None,
    par_path: Path,
    tim_path: Path,
    clock_override_path: Path | None,
) -> Path | None:
    if best_certification is None or not best_certification.fitted_theta:
        return None
    best_par_path = _write_solution_parfile(
        output_dir / "best_repair.par",
        par_path,
        best_certification,
    )
    try:
        from timing_accelerator.reports.residual_plot import write_residual_plot

        write_residual_plot(
            par_path=best_par_path,
            tim_path=tim_path,
            output_dir=output_dir,
            clock_override_path=clock_override_path,
            artifact_stem="best_repair",
        )
    except Exception as exc:  # pragma: no cover - plotting is a reporting convenience
        (output_dir / "best_repair_residuals_error.txt").write_text(
            f"Residual plot generation failed: {exc}\n",
            encoding="utf-8",
        )
    return best_par_path


def _summary_markdown(result: SessionRepairRunResult) -> str:
    base_rms = result.base_certification.weighted_rms_us
    best_rms = (
        result.best_repair_certification.weighted_rms_us
        if result.best_repair_certification is not None
        else None
    )
    improvement = (
        float(base_rms) - float(best_rms)
        if base_rms is not None and best_rms is not None
        else None
    )
    lines = [
        "# Session Repair Summary",
        "",
        f"- source_run: {result.source_run_dir}",
        f"- base_branch_id: {result.base_branch.branch_id}",
        f"- base_weighted_rms_us: {base_rms}",
        f"- generated_candidates: {len(result.repair_branches)}",
        f"- certified_candidates: {len(result.repair_certifications)}",
        f"- best_repair_branch_id: {result.best_repair_certification.branch_id if result.best_repair_certification else None}",
        f"- best_repair_weighted_rms_us: {best_rms}",
        f"- improvement_us: {improvement}",
    ]
    if result.best_repair_par_path is not None:
        lines.append(f"- best_repair_par: {result.best_repair_par_path}")
    lines.extend(["", "## Worst Residual Sessions", ""])
    lines.append("| rank | cluster | source | n_toa | mean_abs_us | max_abs_us | mjd_range |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | --- |")
    for rank, score in enumerate(result.session_scores[:10], start=1):
        lines.append(
            "| "
            f"{rank} | {score.cluster_id} | {', '.join(score.source_filenames)} | "
            f"{score.toa_count} | {score.mean_abs_residual_us:.6g} | "
            f"{score.max_abs_residual_us:.6g} | {score.mjd_min:.6f}-{score.mjd_max:.6f} |"
        )
    lines.extend(["", "## Top Repair Certifications", ""])
    lines.append("| rank | branch | weighted_rms_us | reduced_chi2 | notes |")
    lines.append("| ---: | --- | ---: | ---: | --- |")
    ranked_certs = sorted(
        result.repair_certifications,
        key=lambda item: (
            _finite_rms(item),
            (
                float(item.reduced_chi2)
                if item.reduced_chi2 is not None
                else float("inf")
            ),
            item.branch_id,
        ),
    )
    for rank, certification in enumerate(ranked_certs[:10], start=1):
        notes = "; ".join(certification.notes or [])
        lines.append(
            "| "
            f"{rank} | {certification.branch_id} | {certification.weighted_rms_us} | "
            f"{certification.reduced_chi2} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def _repair_diagnostics_payload(
    result: SessionRepairRunResult,
    config: SessionRepairConfig,
) -> dict[str, Any]:
    best_rms = (
        result.best_repair_certification.weighted_rms_us
        if result.best_repair_certification is not None
        else None
    )
    return {
        "source_run_dir": result.source_run_dir,
        "base_branch_id": result.base_branch.branch_id,
        "base_weighted_rms_us": result.base_certification.weighted_rms_us,
        "best_repair_branch_id": (
            result.best_repair_certification.branch_id
            if result.best_repair_certification is not None
            else None
        ),
        "best_repair_weighted_rms_us": best_rms,
        "improvement_us": (
            float(result.base_certification.weighted_rms_us) - float(best_rms)
            if result.base_certification.weighted_rms_us is not None and best_rms is not None
            else None
        ),
        "config": asdict(config),
        "session_scores": result.session_scores,
    }


def run_session_repair(
    source_run_dir: Path,
    output_dir: Path,
    *,
    config: SessionRepairConfig | None = None,
    base_branch_id: str | None = None,
) -> SessionRepairRunResult:
    active_config = config or SessionRepairConfig()
    source_run_dir = Path(source_run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = source_run_dir / "active_binary_timing.yaml"
    payload = _load_yaml(config_path)
    base_dir = config_path.parent
    backend_payload = payload.get("backend") or payload.get("geometry_from_pint") or {}
    par_path = _required_path(base_dir, backend_payload, "par_path")
    tim_path = _required_path(base_dir, backend_payload, "tim_path")
    cluster_manifest_path = _required_path(base_dir, backend_payload, "cluster_manifest_path")
    clock_override_path = _resolve_relative_path(base_dir, backend_payload.get("clock_override_path"))

    branches = load_active_branches(source_run_dir / "active_branches.json")
    certifications = load_branch_certifications(source_run_dir / "branch_certifications.json")
    if base_branch_id is None:
        base_branch, base_certification = best_branch_by_certified_rms(branches, certifications)
    else:
        branch_by_id = {branch.branch_id: branch for branch in branches}
        certification_by_id = {cert.branch_id: cert for cert in certifications}
        if base_branch_id not in branch_by_id or base_branch_id not in certification_by_id:
            raise ValueError(f"Unknown or uncertified base branch '{base_branch_id}'.")
        base_branch = branch_by_id[base_branch_id]
        base_certification = certification_by_id[base_branch_id]

    manifest = load_cluster_manifest(cluster_manifest_path)
    session_scores = rank_residual_sessions(
        source_run_dir / "best_residuals.csv",
        manifest,
        tim_path,
    )
    repair_branches = generate_session_repair_branches(
        base_branch,
        session_scores,
        active_config,
    )
    _write_json(output_dir / "session_repair_candidates.json", repair_branches)

    fit_parameter_names = _resolve_backend_fit_parameter_names(backend_payload, base_dir)
    repair_certifications = certify_session_repair_branches(
        repair_branches,
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        parameter_names=base_branch.theta_names,
        fit_parameter_names=fit_parameter_names,
        maxiter=int(backend_payload.get("maxiter", 5)),
        clock_override_path=clock_override_path,
    )
    _write_json(output_dir / "session_repair_certifications.json", repair_certifications)
    best_repair_certification = _best_certification_by_rms(repair_certifications)
    best_repair_par_path = _write_best_repair_artifacts(
        output_dir=output_dir,
        best_certification=best_repair_certification,
        par_path=par_path,
        tim_path=tim_path,
        clock_override_path=clock_override_path,
    )
    result = SessionRepairRunResult(
        output_dir=output_dir,
        source_run_dir=source_run_dir,
        base_branch=base_branch,
        base_certification=base_certification,
        session_scores=session_scores,
        repair_branches=repair_branches,
        repair_certifications=repair_certifications,
        best_repair_certification=best_repair_certification,
        best_repair_par_path=best_repair_par_path,
    )
    _write_json(output_dir / "session_repair_diagnostics.json", _repair_diagnostics_payload(result, active_config))
    (output_dir / "session_repair_summary.md").write_text(
        _summary_markdown(result),
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run session-driven local wrap repair for an APOLLO active timing run."
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-branch-id", default=None)
    parser.add_argument("--max-sessions", type=int, default=3)
    parser.add_argument("--wrap-radius", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--no-single-gap", action="store_true")
    parser.add_argument("--no-session-shift", action="store_true")
    parser.add_argument("--combine-sessions", action="store_true")
    args = parser.parse_args(argv)
    result = run_session_repair(
        source_run_dir=args.source_run,
        output_dir=args.output_dir,
        base_branch_id=args.base_branch_id,
        config=SessionRepairConfig(
            max_sessions=args.max_sessions,
            wrap_radius=args.wrap_radius,
            max_candidates=args.max_candidates,
            include_single_gap=not args.no_single_gap,
            include_session_shift=not args.no_session_shift,
            combine_sessions=bool(args.combine_sessions),
        ),
    )
    best_rms = (
        result.best_repair_certification.weighted_rms_us
        if result.best_repair_certification is not None
        else None
    )
    print(f"base_branch_id={result.base_branch.branch_id}")
    print(f"base_weighted_rms_us={result.base_certification.weighted_rms_us}")
    print(f"repair_candidates={len(result.repair_branches)}")
    print(
        "best_repair_branch_id="
        f"{result.best_repair_certification.branch_id if result.best_repair_certification else None}"
    )
    print(f"best_repair_weighted_rms_us={best_rms}")


if __name__ == "__main__":
    main()
