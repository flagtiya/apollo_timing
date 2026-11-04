from __future__ import annotations

from apollo_timing.inference._shared_helpers import (
    _resolve_relative_path,
)

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
import yaml

from apollo_timing.inference.active_timing_contracts import (
    ActiveTimingBranch,
    ActiveTimingPosterior,
    ActiveTimingRunResult,
    BranchCertification,
    ObservationWindow,
    to_json_ready,
)
from apollo_timing.inference.alias_branch_certifier import build_alias_branch_certifier
from apollo_timing.inference.bayesian_alias_lattice import (
    BayesianAliasLatticeConfig,
    generate_bayesian_alias_lattice_branches,
)
from apollo_timing.inference.branch_posterior import (
    BranchPosteriorConfig,
    compute_active_branch_posterior,
)
from apollo_timing.inference.branch_proposal import (
    BranchProposalConfig,
    alias_certification_to_branch_certification,
    branch_to_alias_candidate,
    generate_active_timing_branches,
)
from apollo_timing.inference.fast_single_star_certifier import (
    FastSingleStarSearchConfig,
    run_fast_single_star_search,
)
from apollo_timing.inference.layered_binary_flow import (
    BinaryTimingCandidate,
    build_nested_engine_certifier,
)
from apollo_timing.inference.observation_utility import (
    ObservationUtilityConfig,
    rank_observation_windows,
)
from apollo_timing.inference.pint_alias_geometry import build_pint_alias_geometry
from apollo_timing.inference.pint_solution_artifacts import write_certified_solution_parfile
from apollo_timing.inference.prior_transform import (
    build_parameter_prior_specs,
    extract_fit_flagged_parameters,
)
from apollo_timing.inference.staged_budgeted_binary_search import (
    StagedBudgetedSearchConfig,
    StagedSearchResult,
    run_staged_budgeted_binary_search,
)
from apollo_timing.io.save_artifacts import ensure_output_dir, write_text


def _load_payload(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}




def _array_list(values: list[Any]) -> list[np.ndarray]:
    return [np.asarray(item, dtype=float) for item in values]


def _matrix(value: Any) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=float)


def _vector(value: Any, length: int = 0) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=float)
    return np.asarray(value, dtype=float)


def _float_keyed_matrix_map(payload: dict[str, Any] | None) -> dict[float, np.ndarray]:
    return {
        float(key): np.asarray(value, dtype=float)
        for key, value in (payload or {}).items()
    }


def _windows(payload: list[dict[str, Any]] | None) -> list[ObservationWindow]:
    return [
        ObservationWindow(
            epoch_mjd=float(item["epoch_mjd"]),
            duration_min=float(item.get("duration_min", 20.0)),
            toa_count=int(item.get("toa_count", 8)),
            obs=item.get("obs"),
            freq_mhz=(float(item["freq_mhz"]) if "freq_mhz" in item else None),
            toa_error_us=(float(item["toa_error_us"]) if "toa_error_us" in item else None),
            cost=float(item.get("cost", 0.0)),
        )
        for item in (payload or [])
    ]


def _write_json(path: Path, payload: Any) -> None:
    import json

    path.write_text(
        json.dumps(to_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


_APOLLO_PATH_FIRST_SOURCES = {
    "apollo-path-first",
    "staged-budgeted-bayesian-search",
}


def _staged_search_config_from_payload(payload: dict[str, Any]) -> StagedBudgetedSearchConfig:
    allowed_names = {field.name for field in fields(StagedBudgetedSearchConfig)}
    config_payload: dict[str, Any] = {}
    for name, value in payload.items():
        if name not in allowed_names:
            continue
        config_payload[name] = tuple(value) if isinstance(value, list) else value
    return StagedBudgetedSearchConfig(**config_payload)


def _resolve_staged_required_path(
    *,
    name: str,
    base_dir: Path,
    staged_payload: dict[str, Any],
    backend_payload: dict[str, Any],
    geometry_payload: dict[str, Any],
) -> Path:
    value = staged_payload.get(name) or backend_payload.get(name) or geometry_payload.get(name)
    path = _resolve_relative_path(base_dir, value)
    if path is None:
        raise ValueError(f"apollo-path-first requires staged_search.{name} or backend/geometry_from_pint.{name}.")
    return path


def _run_apollo_path_first_proposal(
    *,
    payload: dict[str, Any],
    base_dir: Path,
    output_dir: Path,
    theta_names: list[str],
) -> tuple[list[ActiveTimingBranch], list[BranchCertification], dict[str, Any]]:
    staged_payload = payload.get("staged_search") or payload.get("staged_budgeted_search") or {}
    backend_payload = payload.get("backend") or {}
    geometry_payload = payload.get("geometry_from_pint") or {}
    staged_config = _staged_search_config_from_payload(staged_payload)
    staged_result: StagedSearchResult = run_staged_budgeted_binary_search(
        par_path=_resolve_staged_required_path(
            name="par_path",
            base_dir=base_dir,
            staged_payload=staged_payload,
            backend_payload=backend_payload,
            geometry_payload=geometry_payload,
        ),
        tim_path=_resolve_staged_required_path(
            name="tim_path",
            base_dir=base_dir,
            staged_payload=staged_payload,
            backend_payload=backend_payload,
            geometry_payload=geometry_payload,
        ),
        cluster_manifest_path=_resolve_staged_required_path(
            name="cluster_manifest_path",
            base_dir=base_dir,
            staged_payload=staged_payload,
            backend_payload=backend_payload,
            geometry_payload=geometry_payload,
        ),
        output_dir=output_dir / "path_first_stages",
        theta_names=staged_payload.get("theta_names", theta_names),
        fit_parameter_names=staged_payload.get(
            "fit_parameter_names",
            backend_payload.get("fit_parameter_names", "fit_flags"),
        ),
        structure_id=str(payload.get("structure_id", "active_local")),
        binary_model=payload.get("binary_model"),
        clock_override_path=_resolve_relative_path(
            base_dir,
            staged_payload.get("clock_override_path")
            or backend_payload.get("clock_override_path")
            or geometry_payload.get("clock_override_path"),
        ),
        config=staged_config,
    )
    best = staged_result.best_certification
    diagnostics = {
        "proposal_source": "apollo-path-first",
        "engine": "staged-budgeted-bayesian-search",
        "staged_output_dir": staged_result.output_dir,
        "stage_count": len(staged_result.stage_records),
        "best_branch_id": staged_result.best_branch.branch_id if staged_result.best_branch else None,
        "best_reduced_chi2": best.reduced_chi2 if best else None,
        "best_weighted_rms_us": best.weighted_rms_us if best else None,
        "active_par_path": staged_result.active_par_path,
        "fit_parameter_names": staged_result.fit_parameter_names,
        "config": staged_config,
        "stages": staged_result.stage_records,
    }
    return list(staged_result.branches), list(staged_result.certifications), diagnostics


def _posterior_config_from_payload(payload: dict[str, Any]) -> BranchPosteriorConfig:
    return BranchPosteriorConfig(
        certifiable_probability=float(payload.get("certifiable_probability", 0.9)),
        certifiable_log_odds=float(payload.get("certifiable_log_odds", 2.0)),
        ambiguity_probability_floor=float(payload.get("ambiguity_probability_floor", 0.01)),
        max_ambiguity_pairs=int(payload.get("max_ambiguity_pairs", 5)),
        max_reduced_chi2=(
            float(payload["max_reduced_chi2"])
            if "max_reduced_chi2" in payload
            else None
        ),
        max_weighted_rms_us=(
            float(payload["max_weighted_rms_us"])
            if "max_weighted_rms_us" in payload
            else None
        ),
    )


def _mock_certifications(branches, backend_payload: dict[str, Any]) -> list[BranchCertification]:
    log_likelihoods = {
        str(key): float(value)
        for key, value in (backend_payload.get("log_likelihoods") or {}).items()
    }
    default_log_likelihood = float(backend_payload.get("default_log_likelihood", -1.0))
    reduced_chi2 = float(backend_payload.get("reduced_chi2", 1.0))
    weighted_rms_us = float(backend_payload.get("weighted_rms_us", 1.0))
    certifications: list[BranchCertification] = []
    for branch in branches:
        dimension = len(branch.theta_names)
        certification_theta_offset = _certification_theta_offset(branch)
        certifications.append(
            BranchCertification(
                branch_id=branch.branch_id,
                log_likelihood=float(log_likelihoods.get(branch.branch_id, default_log_likelihood)),
                fitted_theta={
                    name: float(value)
                    for name, value in zip(branch.theta_names, certification_theta_offset, strict=True)
                },
                covariance=np.eye(dimension, dtype=float),
                reduced_chi2=reduced_chi2,
                weighted_rms_us=weighted_rms_us,
                backend="mock",
                log_evidence=(
                    float(backend_payload["log_evidences"][branch.branch_id])
                    if "log_evidences" in backend_payload
                    and branch.branch_id in backend_payload["log_evidences"]
                    else None
                ),
            )
        )
    return certifications


def _certification_theta_offset(branch: ActiveTimingBranch) -> np.ndarray:
    diagnostics = branch.diagnostics or {}
    if "certification_theta_offset" in diagnostics:
        return np.asarray(diagnostics["certification_theta_offset"], dtype=float)
    return np.asarray(branch.theta_offset, dtype=float)


def _local_laplace_certifications(
    branches,
    payload: dict[str, Any],
    base_dir: Path,
    theta_names: list[str],
) -> list[BranchCertification]:
    required = ["par_path", "tim_path", "cluster_manifest_path"]
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(f"local-laplace backend requires: {', '.join(missing)}")
    certifier = build_alias_branch_certifier(
        par_path=_resolve_relative_path(base_dir, payload["par_path"]),
        tim_path=_resolve_relative_path(base_dir, payload["tim_path"]),
        cluster_manifest_path=_resolve_relative_path(base_dir, payload["cluster_manifest_path"]),
        parameter_names=theta_names,
        fit_parameter_names=_resolve_backend_fit_parameter_names(payload, base_dir),
        maxiter=int(payload.get("maxiter", 5)),
        clock_override_path=_resolve_relative_path(base_dir, payload.get("clock_override_path")),
    )
    return [
        alias_certification_to_branch_certification(certifier(branch_to_alias_candidate(branch)))
        for branch in branches
    ]


def _load_numeric_parameter(path: Path, name: str) -> float:
    target = name.upper()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) >= 2 and parts[0].upper() == target:
            if target == "RAJ":
                return float(Angle(parts[1], unit=u.hourangle).hour)
            if target == "DECJ":
                return float(Angle(parts[1], unit=u.deg).degree)
            return float(parts[1])
    raise KeyError(f"Parameter '{name}' not found in '{path}'.")


def _load_model_numeric_parameters(path: Path, names: list[str]) -> dict[str, float]:
    from apollo_timing.backend.pint_backend import _load_pint_model

    model = _load_pint_model(path)
    values: dict[str, float] = {}
    for name in names:
        if name not in model.params:
            raise KeyError(f"Parameter '{name}' not found in '{path}'.")
        values[name] = float(model[name].value)
    return values


def _fit_flagged_initial_theta(par_path: Path, names: list[str]) -> dict[str, float]:
    return {
        name: _load_numeric_parameter(par_path, name)
        for name in names
    }


def _resolve_theta_names(payload: dict[str, Any], base_dir: Path) -> list[str]:
    raw_theta_names = payload.get("theta_names")
    if raw_theta_names is None:
        raise ValueError("Active timing config requires theta_names or theta_names: fit_flags.")
    if isinstance(raw_theta_names, str):
        if raw_theta_names.lower() != "fit_flags":
            raise ValueError(f"Unsupported theta_names shorthand: {raw_theta_names}")
        par_path = None
        if payload.get("backend") and payload["backend"].get("par_path"):
            par_path = _resolve_relative_path(base_dir, payload["backend"]["par_path"])
        elif payload.get("geometry_from_pint") and payload["geometry_from_pint"].get("par_path"):
            par_path = _resolve_relative_path(base_dir, payload["geometry_from_pint"]["par_path"])
        if par_path is None:
            raise ValueError("theta_names: fit_flags requires backend.par_path or geometry_from_pint.par_path.")
        return [str(name).upper() for name in extract_fit_flagged_parameters(par_path)]
    return [str(name).upper() for name in raw_theta_names]


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
        return [str(name).upper() for name in extract_fit_flagged_parameters(par_path)]
    return [str(name).upper() for name in raw_names]


def _parameter_offset_grid_from_payload(
    payload: dict[str, Any],
    base_dir: Path,
    theta_names: list[str],
) -> list[np.ndarray]:
    grid = _array_list(payload.get("parameter_offset_grid") or [])
    derived_payload = payload.get("parameter_offset_grid_from_par") or []
    if isinstance(derived_payload, dict):
        derived_payload = [derived_payload]
    for item in derived_payload:
        if not isinstance(item, dict):
            raise ValueError("parameter_offset_grid_from_par entries must be mappings.")
        source_par_path = _resolve_relative_path(base_dir, item.get("source_par_path") or item.get("source"))
        target_par_path = _resolve_relative_path(base_dir, item.get("target_par_path") or item.get("target"))
        if source_par_path is None or target_par_path is None:
            raise ValueError(
                "parameter_offset_grid_from_par requires source_par_path and target_par_path."
            )
        source_values = _load_model_numeric_parameters(source_par_path, theta_names)
        target_values = _load_model_numeric_parameters(target_par_path, theta_names)
        grid.append(
            np.asarray(
                [target_values[name] - source_values[name] for name in theta_names],
                dtype=float,
            )
        )
    return grid


def _parameter_scales_from_initial_priors(
    payload: dict[str, Any],
    base_dir: Path,
    theta_names: list[str],
) -> np.ndarray:
    par_path = None
    if payload.get("backend") and payload["backend"].get("par_path"):
        par_path = _resolve_relative_path(base_dir, payload["backend"]["par_path"])
    elif payload.get("geometry_from_pint") and payload["geometry_from_pint"].get("par_path"):
        par_path = _resolve_relative_path(base_dir, payload["geometry_from_pint"]["par_path"])
    if par_path is None:
        raise ValueError("Bayesian alias lattice requires backend.par_path or geometry_from_pint.par_path.")
    prior_by_name = {
        prior.name: prior
        for prior in build_parameter_prior_specs(par_path, allowed_params=set(theta_names))
    }
    missing = [name for name in theta_names if name not in prior_by_name]
    if missing:
        raise ValueError(
            "Bayesian alias lattice requires fit-flag prior widths for all theta names; "
            f"missing {missing}."
        )
    return np.asarray(
        [
            max(abs(float(prior_by_name[name].upper) - float(prior_by_name[name].center)), 1.0e-30)
            for name in theta_names
        ],
        dtype=float,
    )


def _branch_penalty_scale(
    payload: dict[str, Any],
    result,
    name: str,
) -> float:
    specific_key = f"branch_penalty_{name.lower()}_scale"
    if specific_key in payload:
        return max(float(payload[specific_key]), 1.0e-30)
    scales = payload.get("branch_penalty_scales") or {}
    if name in scales:
        return max(float(scales[name]), 1.0e-30)
    if name.upper() in scales:
        return max(float(scales[name.upper()]), 1.0e-30)
    half_width = result.final_half_widths.get(name)
    if half_width is not None and float(half_width) > 0.0:
        return max(float(half_width), 1.0e-30)
    if name == "F0":
        return max(float(result.final_f0_half_width), 1.0e-24)
    if name == "F1":
        return max(float(result.final_f1_half_width), 1.0e-30)
    return 1.0


def _single_star_control_certifications(
    branches,
    payload: dict[str, Any],
    base_dir: Path,
    structure_id: str,
) -> list[BranchCertification]:
    required = ["par_path", "tim_path", "cluster_manifest_path", "structure_library_path"]
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(f"single-star-control backend requires: {', '.join(missing)}")
    par_path = _resolve_relative_path(base_dir, payload["par_path"])
    result = run_fast_single_star_search(
        par_path=par_path,
        tim_path=_resolve_relative_path(base_dir, payload["tim_path"]),
        cluster_manifest_path=_resolve_relative_path(base_dir, payload["cluster_manifest_path"]),
        structure_library_path=_resolve_relative_path(base_dir, payload["structure_library_path"]),
        config=FastSingleStarSearchConfig(
            f0_half_width=float(payload.get("f0_half_width", 1.0e-6)),
            f1_half_width=float(payload.get("f1_half_width", 1.0e-14)),
            coarse_grid_size=int(payload.get("coarse_grid_size", 7)),
            refinement_rounds=int(payload.get("refinement_rounds", 3)),
            shrink_factor=float(payload.get("shrink_factor", 0.25)),
            structure_id=str(payload.get("structure_id", structure_id)),
            linearized_fit_iterations=int(payload.get("linearized_fit_iterations", 5)),
            use_scipy_refinement=bool(payload.get("use_scipy_refinement", True)),
            fit_parameter_names=payload.get("fit_parameter_names"),
            grid_parameter_names=payload.get("grid_parameter_names", ("F0", "F1")),
        ),
    )
    fit_parameter_names = list(result.fit_parameter_names)
    initial_theta = _fit_flagged_initial_theta(par_path, fit_parameter_names)
    fitted_theta = {
        name: float(result.best_state.active_theta[name])
        for name in fit_parameter_names
        if name in (result.best_state.active_theta or {})
    }
    branch_theta_names = [str(name).upper() for name in (branches[0].theta_names if branches else [])]
    dimension = max(len(branch_theta_names), 1)
    covariance_diagonal = [
        _branch_penalty_scale(payload, result, name) ** 2
        for name in branch_theta_names
    ]
    covariance = np.diag(covariance_diagonal or [1.0])
    certifications: list[BranchCertification] = []
    for branch in branches:
        offset = np.asarray(branch.theta_offset, dtype=float)
        compared_names = [str(name).upper() for name in branch.theta_names]
        compared_indices = [
            index
            for index, name in enumerate(compared_names)
            if name in fitted_theta and name in initial_theta
        ]
        branch_recovered_offset = np.asarray(
            [
                fitted_theta[compared_names[index]] - initial_theta[compared_names[index]]
                for index in compared_indices
            ],
            dtype=float,
        )
        comparable_offset = offset[compared_indices] if compared_indices else np.asarray([], dtype=float)
        if comparable_offset.shape != branch_recovered_offset.shape:
            distance_penalty = 1.0e6
        elif branch_recovered_offset.size == 0:
            distance_penalty = 0.0
        else:
            scale = np.asarray(
                [
                    _branch_penalty_scale(payload, result, compared_names[index])
                    for index in compared_indices
                ],
                dtype=float,
            )
            normalized_delta = (comparable_offset - branch_recovered_offset) / scale
            distance_penalty = 0.5 * float(normalized_delta @ normalized_delta)
        log_likelihood = float(result.best_evaluation.log_likelihood) - distance_penalty
        certifications.append(
            BranchCertification(
                branch_id=branch.branch_id,
                log_likelihood=log_likelihood,
                fitted_theta=fitted_theta,
                covariance=covariance[:dimension, :dimension],
                reduced_chi2=float(result.best_evaluation.reduced_chi2),
                weighted_rms_us=float(result.best_evaluation.weighted_rms_us),
                backend="single-star-control",
                log_evidence=log_likelihood + float(branch.log_prior),
                chi2=float(result.best_evaluation.chi2),
                notes=[
                    f"evaluation_count={result.evaluation_count}",
                    f"fit_parameter_names={','.join(result.fit_parameter_names)}",
                    "single-star-control-validates-route-c-fixture",
                ],
            )
        )
    return certifications


def _build_geometry_from_config(
    payload: dict[str, Any],
    theta_names: list[str],
    base_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray], float]:
    if payload.get("geometry_from_pint"):
        geometry_config = payload["geometry_from_pint"]
        geometry = build_pint_alias_geometry(
            par_path=_resolve_relative_path(base_dir, geometry_config["par_path"]),
            tim_path=_resolve_relative_path(base_dir, geometry_config["tim_path"]),
            cluster_manifest_path=_resolve_relative_path(base_dir, geometry_config["cluster_manifest_path"]),
            parameter_names=theta_names,
            candidate_followup_epochs=[
                float(item["epoch_mjd"])
                for item in payload.get("observation_windows", [])
            ],
            clock_override_path=_resolve_relative_path(base_dir, geometry_config.get("clock_override_path")),
        )
        return (
            np.asarray(geometry.gap_jacobian, dtype=float),
            np.asarray(geometry.reference_phase_residuals, dtype=float),
            {
                float(epoch): np.asarray(row, dtype=float)
                for epoch, row in geometry.design_by_epoch.items()
            },
            float(geometry.toa_error_phase),
        )
    gap_jacobian = _matrix(payload.get("gap_jacobian"))
    if gap_jacobian.size == 0:
        gap_jacobian = np.zeros((0, len(theta_names)), dtype=float)
    reference = _vector(payload.get("reference_phase_residuals"), length=gap_jacobian.shape[0])
    design_by_epoch = _float_keyed_matrix_map(payload.get("design_by_epoch"))
    toa_error_phase = float(payload.get("toa_error_phase", 1.0))
    return gap_jacobian, reference, design_by_epoch, toa_error_phase


def _select_escalation_branches(
    branches: list[ActiveTimingBranch],
    posterior: ActiveTimingPosterior,
    top_k: int,
) -> list[ActiveTimingBranch]:
    branch_by_id = {branch.branch_id: branch for branch in branches}
    ranked = sorted(
        posterior.branch_probabilities.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [
        branch_by_id[branch_id]
        for branch_id, _ in ranked[: max(int(top_k), 0)]
        if branch_id in branch_by_id
    ]


def _should_run_nested_escalation(
    posterior: ActiveTimingPosterior,
    payload: dict[str, Any] | None,
) -> bool:
    if not payload or not bool(payload.get("enabled", False)):
        return False
    if posterior.certifiable and not bool(payload.get("force", False)):
        return False
    entropy_min = payload.get("posterior_entropy_min")
    if entropy_min is not None and float(posterior.entropy) < float(entropy_min):
        return False
    log_odds_max = payload.get("log_odds_max")
    if log_odds_max is not None:
        log_odds = posterior.log_odds_best_second
        if log_odds is not None and float(log_odds) > float(log_odds_max):
            return False
    return True


def _replace_certifications(
    certifications: list[BranchCertification],
    replacements: list[BranchCertification],
) -> list[BranchCertification]:
    replacement_by_id = {item.branch_id: item for item in replacements}
    replaced_ids: set[str] = set()
    merged: list[BranchCertification] = []
    for certification in certifications:
        replacement = replacement_by_id.get(certification.branch_id)
        if replacement is None:
            merged.append(certification)
            continue
        merged.append(replacement)
        replaced_ids.add(replacement.branch_id)
    for replacement in replacements:
        if replacement.branch_id not in replaced_ids:
            merged.append(replacement)
    return merged


def _mock_nested_certifications(
    branches: list[ActiveTimingBranch],
    existing_by_id: dict[str, BranchCertification],
    payload: dict[str, Any],
) -> tuple[list[BranchCertification], list[dict[str, Any]]]:
    log_evidences = {
        str(key): float(value)
        for key, value in (payload.get("log_evidences") or {}).items()
    }
    log_likelihoods = {
        str(key): float(value)
        for key, value in (payload.get("log_likelihoods") or {}).items()
    }
    default_log_evidence = payload.get("default_log_evidence")
    default_log_likelihood = payload.get("default_log_likelihood")
    replacements: list[BranchCertification] = []
    records: list[dict[str, Any]] = []
    for branch in branches:
        existing = existing_by_id.get(branch.branch_id)
        log_evidence = (
            log_evidences[branch.branch_id]
            if branch.branch_id in log_evidences
            else (
                float(default_log_evidence)
                if default_log_evidence is not None
                else (existing.log_evidence if existing is not None else None)
            )
        )
        log_likelihood = (
            log_likelihoods[branch.branch_id]
            if branch.branch_id in log_likelihoods
            else (
                float(default_log_likelihood)
                if default_log_likelihood is not None
                else (
                    existing.log_likelihood
                    if existing is not None
                    else (float(log_evidence) if log_evidence is not None else -1.0)
                )
            )
        )
        dimension = len(branch.theta_names)
        replacements.append(
            BranchCertification(
                branch_id=branch.branch_id,
                log_likelihood=float(log_likelihood),
        fitted_theta=(
            existing.fitted_theta
            if existing is not None
            else {
                name: float(value)
                for name, value in zip(branch.theta_names, _certification_theta_offset(branch), strict=True)
            }
        ),
                covariance=(
                    existing.covariance
                    if existing is not None
                    else np.eye(dimension, dtype=float)
                ),
                reduced_chi2=(
                    float(payload["reduced_chi2"])
                    if "reduced_chi2" in payload
                    else (existing.reduced_chi2 if existing is not None else 1.0)
                ),
                weighted_rms_us=(
                    float(payload["weighted_rms_us"])
                    if "weighted_rms_us" in payload
                    else (existing.weighted_rms_us if existing is not None else 1.0)
                ),
                backend="nested-mock",
                log_evidence=(float(log_evidence) if log_evidence is not None else None),
                chi2=existing.chi2 if existing is not None else None,
                notes=["nested-escalation-mock"],
            )
        )
        records.append(
            {
                "branch_id": branch.branch_id,
                "mode": "mock",
                "status": "replaced",
                "log_evidence": log_evidence,
            }
        )
    return replacements, records


def _branch_to_binary_candidate(branch: ActiveTimingBranch) -> BinaryTimingCandidate:
    if branch.binary_model is None:
        raise ValueError("Nested escalation requires a binary_model for each selected branch.")
    return BinaryTimingCandidate(
        candidate_id=branch.branch_id,
        binary_model=branch.binary_model,
        parameter_offsets={
            name: float(value)
            for name, value in zip(branch.theta_names, _certification_theta_offset(branch), strict=True)
        },
        wrap_penalty=float(sum(abs(item) for item in branch.wraps)),
        log_prior=float(branch.log_prior),
    )


def _extract_nested_quality(result) -> tuple[float | None, float | None]:
    report = result.uniqueness_gate_report or {}
    quality = report.get("quality_gate") if isinstance(report, dict) else None
    if not isinstance(quality, dict):
        return None, None
    return (
        (
            float(quality["best_state_reduced_chi2"])
            if quality.get("best_state_reduced_chi2") is not None
            else None
        ),
        (
            float(quality["best_state_weighted_rms_us"])
            if quality.get("best_state_weighted_rms_us") is not None
            else None
        ),
    )


def _nested_result_to_branch_certification(
    branch: ActiveTimingBranch,
    result,
) -> BranchCertification:
    state = result.refined_best_state or result.best_state
    fitted_theta: dict[str, float] = {}
    if state is not None:
        theta_payload = state.active_theta or state.theta or {}
        fitted_theta = {
            str(key).upper(): float(value)
            for key, value in theta_payload.items()
            if isinstance(value, (int, float))
        }
    if not fitted_theta:
        fitted_theta = {
            name: float(value)
            for name, value in zip(branch.theta_names, _certification_theta_offset(branch), strict=True)
        }
    reduced_chi2, weighted_rms_us = _extract_nested_quality(result)
    dimension = max(len(branch.theta_names), 1)
    return BranchCertification(
        branch_id=branch.branch_id,
        log_likelihood=(
            float(result.best_log_likelihood)
            if result.best_log_likelihood is not None
            else (float(result.log_evidence) if result.log_evidence is not None else float("-inf"))
        ),
        fitted_theta=fitted_theta,
        covariance=np.eye(dimension, dtype=float),
        reduced_chi2=reduced_chi2,
        weighted_rms_us=weighted_rms_us,
        backend="nested",
        log_evidence=(float(result.log_evidence) if result.log_evidence is not None else None),
        notes=[
            f"nested_output_dir={result.output_dir}",
            f"posterior_sample_count={result.posterior_sample_count}",
        ],
    )


def _real_nested_certifications(
    branches: list[ActiveTimingBranch],
    payload: dict[str, Any],
    base_dir: Path,
    output_dir: Path,
) -> tuple[list[BranchCertification], list[dict[str, Any]]]:
    required = ["par_path", "tim_path", "cluster_manifest_path", "engine_config_path"]
    missing = [name for name in required if not payload.get(name)]
    if missing:
        raise ValueError(f"nested escalation requires: {', '.join(missing)}")
    certifier = build_nested_engine_certifier(
        par_path=_resolve_relative_path(base_dir, payload["par_path"]),
        tim_path=_resolve_relative_path(base_dir, payload["tim_path"]),
        cluster_manifest_path=_resolve_relative_path(base_dir, payload["cluster_manifest_path"]),
        engine_config_path=_resolve_relative_path(base_dir, payload["engine_config_path"]),
        active_fit_params=payload.get("active_fit_params"),
    )
    replacements: list[BranchCertification] = []
    records: list[dict[str, Any]] = []
    nested_root = ensure_output_dir(output_dir / "nested_escalation")
    for rank, branch in enumerate(branches, start=1):
        branch_dir = nested_root / f"{rank:02d}_{branch.branch_id}"
        result = certifier(_branch_to_binary_candidate(branch), branch_dir)
        replacements.append(_nested_result_to_branch_certification(branch, result))
        records.append(
            {
                "branch_id": branch.branch_id,
                "mode": "nested",
                "status": "replaced",
                "output_dir": result.output_dir,
                "log_evidence": result.log_evidence,
                "best_log_likelihood": result.best_log_likelihood,
            }
        )
    return replacements, records


def _run_nested_escalation(
    branches: list[ActiveTimingBranch],
    certifications: list[BranchCertification],
    posterior: ActiveTimingPosterior,
    posterior_config: BranchPosteriorConfig,
    payload: dict[str, Any] | None,
    base_dir: Path,
    output_dir: Path,
) -> tuple[list[BranchCertification], ActiveTimingPosterior, dict[str, Any]]:
    record: dict[str, Any] = {
        "enabled": bool(payload.get("enabled", False)) if payload else False,
        "triggered": False,
        "reason": "disabled",
        "selected_branch_ids": [],
        "results": [],
    }
    if not _should_run_nested_escalation(posterior, payload):
        if payload and bool(payload.get("enabled", False)):
            record["reason"] = "trigger-conditions-not-met"
        return certifications, posterior, record

    assert payload is not None
    top_k = int(payload.get("top_k", 2))
    selected_branches = _select_escalation_branches(branches, posterior, top_k)
    record["triggered"] = bool(selected_branches)
    record["reason"] = "triggered" if selected_branches else "no-selected-branches"
    record["selected_branch_ids"] = [branch.branch_id for branch in selected_branches]
    existing_by_id = {item.branch_id: item for item in certifications}
    mode = str(payload.get("mode", "nested")).lower()
    if mode == "mock":
        replacements, result_records = _mock_nested_certifications(selected_branches, existing_by_id, payload)
    elif mode == "nested":
        replacements, result_records = _real_nested_certifications(selected_branches, payload, base_dir, output_dir)
    else:
        raise ValueError(f"Unsupported nested escalation mode: {mode}")
    record["mode"] = mode
    record["results"] = result_records
    merged = _replace_certifications(certifications, replacements)
    return (
        merged,
        compute_active_branch_posterior(branches, merged, posterior_config),
        record,
    )


def _summary_markdown(result: ActiveTimingRunResult) -> str:
    lines = [
        "# Active Binary Timing Summary",
        "",
        f"- branches: {len(result.branches)}",
        f"- certifications: {len(result.certifications)}",
        f"- best_branch_id: {result.posterior.best_branch_id}",
        f"- certifiable: {result.posterior.certifiable}",
        f"- posterior_entropy: {result.posterior.entropy:.6g}",
        f"- recommendations: {len(result.recommendations)}",
    ]
    if result.recommendations:
        top = result.recommendations[0]
        lines.extend(
            [
                "",
                "## Top Observation Recommendation",
                "",
                f"- epoch_mjd: {top.window.epoch_mjd}",
                f"- utility: {top.utility:.6g}",
                f"- phase_separation_sigma: {top.phase_separation_sigma:.6g}",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_best_branch_parfile(
    result: ActiveTimingRunResult,
    payload: dict[str, Any],
    base_dir: Path,
    staged_search_summary: dict[str, Any] | None = None,
) -> Path | None:
    staged_payload = staged_search_summary or {}
    best_branch_id = staged_payload.get("best_branch_id") or result.posterior.best_branch_id
    if best_branch_id is None:
        return None
    certification_by_id = {item.branch_id: item for item in result.certifications}
    certification = certification_by_id.get(best_branch_id)
    if certification is None:
        return None
    if certification.backend not in {"local-laplace", "single-star-control", "nested"}:
        return None
    backend_payload = payload.get("backend") or {}
    par_path_value = (
        staged_payload.get("active_par_path")
        or backend_payload.get("par_path")
        or payload.get("geometry_from_pint", {}).get("par_path")
    )
    if par_path_value is None:
        return None
    seed_par_path = _resolve_relative_path(base_dir, par_path_value)
    if seed_par_path is None:
        return None
    fit_parameter_names = staged_payload.get("fit_parameter_names")
    best_par_path = result.output_dir / "best.par"
    return write_certified_solution_parfile(
        path=best_par_path,
        seed_par_path=seed_par_path,
        certification=certification,
        fit_parameter_names=fit_parameter_names,
    )


def _write_best_residual_plot(
    result: ActiveTimingRunResult,
    payload: dict[str, Any],
    base_dir: Path,
    best_par_path: Path | None,
) -> None:
    if best_par_path is None or not bool(payload.get("plot_residuals", True)):
        return
    backend_payload = payload.get("backend") or {}
    tim_path_value = backend_payload.get("tim_path") or payload.get("geometry_from_pint", {}).get("tim_path")
    if tim_path_value is None:
        return
    tim_path = _resolve_relative_path(base_dir, tim_path_value)
    if tim_path is None:
        return
    clock_path_value = backend_payload.get("clock_override_path") or payload.get("geometry_from_pint", {}).get(
        "clock_override_path"
    )
    clock_override_path = _resolve_relative_path(base_dir, clock_path_value)
    try:
        from apollo_timing.reports.residual_plot import write_residual_plot

        write_residual_plot(
            par_path=best_par_path,
            tim_path=tim_path,
            output_dir=result.output_dir,
            clock_override_path=clock_override_path,
        )
    except Exception as exc:  # pragma: no cover - plotting is a reporting convenience
        (result.output_dir / "best_residuals_error.txt").write_text(
            f"Residual plot generation failed: {exc}\n",
            encoding="utf-8",
        )


def run_active_binary_timing(
    config_path: Path,
    output_dir: Path | None = None,
) -> ActiveTimingRunResult:
    config_path = Path(config_path)
    payload = _load_payload(config_path)
    base_dir = config_path.parent
    output_dir = ensure_output_dir(Path(output_dir or payload.get("output_dir") or config_path.with_suffix("")))
    theta_names = _resolve_theta_names(payload, base_dir)
    proposal_payload = payload.get("proposal") or {}
    proposal_source = str(proposal_payload.get("proposal_source", "geometry-grid")).lower()
    staged_search_summary: dict[str, Any] | None = None
    if proposal_source in _APOLLO_PATH_FIRST_SOURCES:
        branches, certifications, proposal_diagnostics = _run_apollo_path_first_proposal(
            payload=payload,
            base_dir=base_dir,
            output_dir=output_dir,
            theta_names=theta_names,
        )
        staged_search_summary = proposal_diagnostics
        design_by_epoch: dict[float, np.ndarray] = {}
        toa_error_phase = 1.0
    else:
        gap_jacobian, reference_phase_residuals, design_by_epoch, toa_error_phase = _build_geometry_from_config(
            payload,
            theta_names,
            base_dir,
        )
        if proposal_source == "bayesian-alias-lattice":
            branches, proposal_diagnostics = generate_bayesian_alias_lattice_branches(
                structure_id=str(payload.get("structure_id", "active_local")),
                binary_model=payload.get("binary_model"),
                theta_names=theta_names,
                gap_jacobian=gap_jacobian,
                reference_phase_residuals=reference_phase_residuals,
                parameter_scales=_parameter_scales_from_initial_priors(payload, base_dir, theta_names),
                config=BayesianAliasLatticeConfig(
                    beam_size=int(proposal_payload.get("beam_size", 5000)),
                    phase_sigma=float(proposal_payload.get("phase_sigma", 0.08)),
                    prior_scale=float(proposal_payload.get("prior_scale", 1.0)),
                    radius_sigma=float(proposal_payload.get("radius_sigma", 5.0)),
                    min_radius=int(proposal_payload.get("min_radius", 4)),
                    max_enum_per_gap=int(proposal_payload.get("max_enum_per_gap", 101)),
                    wrap_prior_scale=float(proposal_payload.get("wrap_prior_scale", 200.0)),
                    top_k=int(proposal_payload.get("top_k", 16)),
                ),
            )
        else:
            branches, proposal_diagnostics = generate_active_timing_branches(
                structure_id=str(payload.get("structure_id", "active_local")),
                binary_model=payload.get("binary_model"),
                theta_names=theta_names,
                parameter_offset_grid=_parameter_offset_grid_from_payload(payload, base_dir, theta_names),
                gap_jacobian=gap_jacobian,
                reference_phase_residuals=reference_phase_residuals,
                config=BranchProposalConfig(
                    max_wrap_residual=float(proposal_payload.get("max_wrap_residual", 0.25)),
                    offset_prior_scale=float(proposal_payload.get("offset_prior_scale", 1.0)),
                    model_log_prior=float(proposal_payload.get("model_log_prior", 0.0)),
                    wrap_prior_scale=float(proposal_payload.get("wrap_prior_scale", 8.0)),
                    wrap_hypothesis_dimensions=int(proposal_payload.get("wrap_hypothesis_dimensions", 0)),
                    wrap_hypothesis_radius=int(proposal_payload.get("wrap_hypothesis_radius", 1)),
                    wrap_hypothesis_max_variants=int(proposal_payload.get("wrap_hypothesis_max_variants", 27)),
                    top_k=(int(proposal_payload["top_k"]) if "top_k" in proposal_payload else None),
                    fallback_min_keep=int(proposal_payload.get("fallback_min_keep", 2)),
                    proposal_source=str(proposal_payload.get("proposal_source", "geometry-grid")),
                ),
            )
        backend_payload = payload.get("backend") or {"mode": "mock"}
        backend_mode = str(backend_payload.get("mode", "mock")).lower()
        if backend_mode == "mock":
            certifications = _mock_certifications(branches, backend_payload)
        elif backend_mode == "local-laplace":
            certifications = _local_laplace_certifications(branches, backend_payload, base_dir, theta_names)
        elif backend_mode == "single-star-control":
            certifications = _single_star_control_certifications(
                branches,
                backend_payload,
                base_dir,
                structure_id=str(payload.get("structure_id", "active_local")),
            )
        else:
            raise ValueError(f"Unsupported active timing backend mode: {backend_mode}")

    posterior_config = _posterior_config_from_payload(payload.get("posterior") or {})
    posterior = compute_active_branch_posterior(branches, certifications, posterior_config)
    certifications, posterior, nested_escalation_record = _run_nested_escalation(
        branches=branches,
        certifications=certifications,
        posterior=posterior,
        posterior_config=posterior_config,
        payload=payload.get("nested_escalation"),
        base_dir=base_dir,
        output_dir=output_dir,
    )
    utility_payload = payload.get("observation_utility") or {}
    recommendations = []
    if not posterior.certifiable and payload.get("observation_windows"):
        recommendations = rank_observation_windows(
            branches=branches,
            posterior=posterior,
            windows=_windows(payload.get("observation_windows")),
            design_by_epoch=design_by_epoch,
            toa_error_phase=toa_error_phase,
            config=ObservationUtilityConfig(
                pairwise_separation_cap=float(utility_payload.get("pairwise_separation_cap", 1.0e12)),
                max_ranked_recommendations=(
                    int(utility_payload["max_ranked_recommendations"])
                    if "max_ranked_recommendations" in utility_payload
                    else None
                ),
                max_reported_pairs=int(utility_payload.get("max_reported_pairs", 5)),
            ),
        )

    result = ActiveTimingRunResult(
        output_dir=output_dir,
        branches=branches,
        certifications=certifications,
        posterior=posterior,
        recommendations=recommendations,
    )
    _write_json(output_dir / "active_branches.json", branches)
    _write_json(output_dir / "branch_certifications.json", certifications)
    _write_json(output_dir / "active_posterior.json", posterior)
    _write_json(output_dir / "observation_recommendations.json", recommendations)
    _write_json(output_dir / "proposal_diagnostics.json", proposal_diagnostics)
    if staged_search_summary is not None:
        _write_json(output_dir / "staged_search_summary.json", staged_search_summary)
    _write_json(output_dir / "nested_escalation.json", nested_escalation_record)
    write_text(output_dir / "active_timing_summary.md", _summary_markdown(result))
    best_par_path = _write_best_branch_parfile(
        result,
        payload,
        base_dir,
        staged_search_summary=staged_search_summary,
    )
    _write_best_residual_plot(result, payload, base_dir, best_par_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bayesian active binary timing.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_active_binary_timing(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
