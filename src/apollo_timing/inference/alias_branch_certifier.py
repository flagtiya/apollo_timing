from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from astropy import units as u
from pint.residuals import Residuals

from apollo_timing.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_model,
    _load_pint_toas,
    _run_fit,
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.alias_manifold_solver import AliasManifoldCandidate
from apollo_timing.inference.cluster_manifest import load_cluster_manifest, resolve_cluster_manifest
from apollo_timing.inference.pulse_number_reference import (
    ensure_local_island_pulse_numbers,
    ensure_pulse_numbers,
)
from apollo_timing.inference.wrap_state import apply_wraps_to_toas


@dataclass(frozen=True)
class AliasBranchCertificationResult:
    candidate_id: str
    parameter_names: list[str]
    fitted_parameters: dict[str, float]
    covariance: np.ndarray
    log_likelihood: float
    chi2: float
    reduced_chi2: float
    weighted_rms_us: float
    notes: list[str]


def _build_wrapped_toas(
    base_toas,
    pulse_number_model,
    manifest,
    wraps: Sequence[int],
    *,
    local_island_reference: bool = False,
):
    toas = deepcopy(base_toas)
    if local_island_reference:
        ensure_local_island_pulse_numbers(toas, pulse_number_model, manifest)
    else:
        ensure_pulse_numbers(toas, pulse_number_model)
    return apply_wraps_to_toas(toas, manifest, list(wraps))


def _extract_parameter_covariance(
    full_covariance: np.ndarray | None,
    design_parameter_names: Sequence[str] | None,
    parameter_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    fallback = np.diag([1.0e-24] * len(parameter_names)).astype(float)
    if full_covariance is None or not design_parameter_names:
        return fallback, ["covariance-unavailable-fallback"]
    name_to_index = {
        name: index
        for index, name in enumerate(design_parameter_names)
        if name != "Offset"
    }
    if any(name not in name_to_index for name in parameter_names):
        return fallback, ["covariance-missing-parameter-fallback"]
    indices = [name_to_index[name] for name in parameter_names]
    extracted = np.asarray(full_covariance, dtype=float)[np.ix_(indices, indices)]
    if extracted.shape != (len(parameter_names), len(parameter_names)):
        return fallback, ["covariance-shape-fallback"]
    if not np.all(np.isfinite(extracted)):
        return fallback, ["covariance-nonfinite-fallback"]
    if np.linalg.slogdet(extracted)[0] <= 0:
        return fallback, ["covariance-nonpositive-fallback"]
    return extracted, []


def _evaluate_log_likelihood(wrapped_toas, model) -> tuple[float, float, float, float]:
    residuals = Residuals(
        wrapped_toas,
        model,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    chi2 = float(residuals.chi2)
    sigma_seconds = np.asarray(wrapped_toas.get_errors().to(u.s).value, dtype=float)
    log_sigma_term = float(sum(math.log(2.0 * math.pi * sigma * sigma) for sigma in sigma_seconds))
    return (
        -0.5 * (chi2 + log_sigma_term),
        chi2,
        float(residuals.reduced_chi2),
        float(residuals.rms_weighted().to(u.us).value),
    )


def build_alias_branch_certifier(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parameter_names: Sequence[str],
    fit_parameter_names: Sequence[str] | None = None,
    maxiter: int = 5,
    clock_override_path: Path | None = None,
) -> Callable[[AliasManifoldCandidate], AliasBranchCertificationResult]:
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    parameter_names = [str(name) for name in parameter_names]
    active_fit_parameter_names = [str(name) for name in (fit_parameter_names or parameter_names)]
    manifest = resolve_cluster_manifest(load_cluster_manifest(Path(cluster_manifest_path)), tim_path)
    seed_model = _load_pint_model(par_path)
    base_values = {name: float(seed_model[name].value) for name in parameter_names}
    for parameter_name in active_fit_parameter_names:
        if parameter_name not in seed_model.params:
            raise ValueError(f"Fit parameter '{parameter_name}' is missing from the timing model.")
    base_toas = _load_pint_toas(
        tim_path,
        seed_model=seed_model,
        clock_override_path=clock_override_path,
    )

    def certify(candidate: AliasManifoldCandidate) -> AliasBranchCertificationResult:
        if len(candidate.parameter_offset) != len(parameter_names):
            raise ValueError(
                f"Candidate parameter offset dimension {len(candidate.parameter_offset)} "
                f"does not match parameter_names dimension {len(parameter_names)}."
            )
        model = _load_pint_model(par_path)
        _freeze_all_parameters(model)
        for parameter_name, offset in zip(parameter_names, candidate.parameter_offset, strict=True):
            if parameter_name not in model.params:
                raise ValueError(f"Parameter '{parameter_name}' is missing from the timing model.")
            _set_numeric_model_parameter_value(
                model,
                parameter_name,
                base_values[parameter_name] + float(offset),
            )
        for parameter_name in active_fit_parameter_names:
            model[parameter_name].frozen = False

        pulse_number_reference = getattr(candidate, "pulse_number_reference", "candidate")
        if pulse_number_reference == "seed":
            pulse_number_model = seed_model
        elif pulse_number_reference == "candidate":
            pulse_number_model = model
        else:
            raise ValueError(
                "Candidate pulse_number_reference must be either 'seed' or 'candidate'."
            )
        wrapped_toas = _build_wrapped_toas(
            base_toas,
            pulse_number_model,
            manifest,
            candidate.wraps,
            local_island_reference=(pulse_number_reference == "seed"),
        )
        fitted_model, _, _, fit_diagnostics = _run_fit(
            wrapped_toas,
            model,
            list(active_fit_parameter_names),
            local_solver_max_iterations=max(int(maxiter), 1),
        )
        covariance, notes = _extract_parameter_covariance(
            fit_diagnostics.get("covariance_matrix"),
            fit_diagnostics.get("design_parameter_names"),
            active_fit_parameter_names,
        )
        log_likelihood, chi2, reduced_chi2, weighted_rms_us = _evaluate_log_likelihood(
            wrapped_toas,
            fitted_model,
        )
        return AliasBranchCertificationResult(
            candidate_id=candidate.candidate_id,
            parameter_names=list(active_fit_parameter_names),
            fitted_parameters={
                name: float(fitted_model[name].value)
                for name in active_fit_parameter_names
            },
            covariance=covariance,
            log_likelihood=float(log_likelihood),
            chi2=chi2,
            reduced_chi2=reduced_chi2,
            weighted_rms_us=weighted_rms_us,
            notes=notes,
        )

    return certify
