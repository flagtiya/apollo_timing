from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from timing_accelerator.inference.alias_branch_certifier import (
    AliasBranchCertificationResult,
    build_alias_branch_certifier,
)
from timing_accelerator.inference.alias_manifold_solver import (
    AliasGeometryPayload,
    AliasManifoldCandidate,
    AliasManifoldConfig,
    BranchPosterior,
    FollowupRecommendation,
    build_alias_candidates_from_geometry,
    compute_branch_posterior,
    recommend_followup_epoch,
)
from timing_accelerator.inference.pint_alias_geometry import build_pint_alias_geometry


@dataclass(frozen=True)
class AliasManifoldEngineResult:
    output_dir: Path
    geometry: AliasGeometryPayload
    candidates: list[AliasManifoldCandidate]
    branch_results: list[AliasBranchCertificationResult]
    posterior: BranchPosterior
    followup_recommendation: FollowupRecommendation | None


def _json_ready(value):
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_alias_manifold_engine(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parameter_names: Sequence[str],
    parameter_offset_grid: Sequence[np.ndarray],
    output_dir: Path,
    fit_parameter_names: Sequence[str] | None = None,
    candidate_followup_epochs: Sequence[float] | None = None,
    config: AliasManifoldConfig | None = None,
    certifier: Callable[[AliasManifoldCandidate], AliasBranchCertificationResult] | None = None,
    clock_override_path: Path | None = None,
) -> AliasManifoldEngineResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    active_config = config or AliasManifoldConfig()
    geometry = build_pint_alias_geometry(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        parameter_names=parameter_names,
        candidate_followup_epochs=candidate_followup_epochs,
        clock_override_path=clock_override_path,
    )
    candidates = build_alias_candidates_from_geometry(
        payload=geometry,
        parameter_offsets=[np.asarray(item, dtype=float) for item in parameter_offset_grid],
        config=active_config,
    )
    active_certifier = certifier or build_alias_branch_certifier(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        parameter_names=parameter_names,
        fit_parameter_names=fit_parameter_names,
        clock_override_path=clock_override_path,
    )
    branch_results = [active_certifier(candidate) for candidate in candidates]
    log_likelihoods = {
        result.candidate_id: float(result.log_likelihood)
        for result in branch_results
    }
    parameter_covariances = {
        result.candidate_id: np.asarray(result.covariance, dtype=float)
        for result in branch_results
    }
    posterior = compute_branch_posterior(
        candidates=candidates,
        log_likelihoods=log_likelihoods,
        parameter_covariances=parameter_covariances,
        config=active_config,
    )
    followup_recommendation = None
    if (
        not posterior.certifiable
        and geometry.design_by_epoch
        and len(candidates) >= 2
        and posterior.branch_probabilities
    ):
        followup_recommendation = recommend_followup_epoch(
            candidate_parameter_offsets={
                candidate.candidate_id: np.asarray(candidate.parameter_offset, dtype=float)
                for candidate in candidates
            },
            posterior_weights=posterior.branch_probabilities,
            design_by_epoch=geometry.design_by_epoch,
            toa_error_phase=geometry.toa_error_phase,
        )

    _write_json(output_dir / "alias_geometry.json", geometry)
    _write_json(output_dir / "alias_candidates.json", candidates)
    _write_json(output_dir / "alias_branch_fits.json", branch_results)
    _write_json(output_dir / "alias_branch_posterior.json", posterior)
    if followup_recommendation is not None:
        _write_json(output_dir / "alias_followup_recommendation.json", followup_recommendation)

    return AliasManifoldEngineResult(
        output_dir=output_dir,
        geometry=geometry,
        candidates=candidates,
        branch_results=branch_results,
        posterior=posterior,
        followup_recommendation=followup_recommendation,
    )
