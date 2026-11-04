from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import math
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import DownhillWLSFitter
from pint.models.model_builder import ModelBuilder, convert_tcb_tdb
from pint.residuals import Residuals
from timing_accelerator.inference.bayesian_dual_lattice_enumerator import infer_contiguous_source_sessions


@dataclass(frozen=True)
class ToaWhiteNoiseCertificationConfig:
    """Post-closure additive TOA-noise evidence and fit controls."""

    enabled: bool = True
    minimum_noise_delta_bic: float = 10.0
    minimum_path_delta_bic: float = 10.0
    maximum_equad_cycles: float = 0.02
    timing_fit_iterations: int = 10
    noise_fit_iterations: int = 4

    def __post_init__(self) -> None:
        positive = (
            self.minimum_noise_delta_bic,
            self.minimum_path_delta_bic,
            self.maximum_equad_cycles,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("TOA white-noise evidence thresholds must be positive.")
        if int(self.timing_fit_iterations) <= 0 or int(self.noise_fit_iterations) <= 0:
            raise ValueError("TOA white-noise fit budgets must be positive.")


@dataclass(frozen=True)
class ToaWhiteNoiseFit:
    model: object
    fitter: object
    residuals: object
    reported_error_log_likelihood: float
    reported_error_bic: float
    noise_conditioned_log_likelihood: float
    noise_conditioned_bic: float
    delta_bic: float
    equad_us: float
    equad_uncertainty_us: float | None
    equad_cycles: float
    boundary_valid: bool
    chi2: float
    reduced_chi2: float
    weighted_rms_us: float
    timing_fit_parameters: tuple[str, ...]
    noise_parameter_count: int
    toa_count: int
    covariance_rank: int
    covariance_dimension: int
    covariance_full_rank: bool
    covariance_error: str | None

    @property
    def evidence_passed(self) -> bool:
        return bool(self.delta_bic > 0.0 and self.boundary_valid)

    def to_payload(self) -> dict[str, Any]:
        return {
            "noise_model": "global_equad_like_toa_generation_scatter",
            "reported_error_log_likelihood": self.reported_error_log_likelihood,
            "reported_error_bic": self.reported_error_bic,
            "noise_conditioned_log_likelihood": self.noise_conditioned_log_likelihood,
            "noise_conditioned_bic": self.noise_conditioned_bic,
            "delta_bic": self.delta_bic,
            "equad_us": self.equad_us,
            "equad_uncertainty_us": self.equad_uncertainty_us,
            "equad_cycles": self.equad_cycles,
            "boundary_valid": self.boundary_valid,
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "weighted_rms_us": self.weighted_rms_us,
            "timing_fit_parameters": self.timing_fit_parameters,
            "noise_parameter_count": self.noise_parameter_count,
            "toa_count": self.toa_count,
            "covariance_rank": self.covariance_rank,
            "covariance_dimension": self.covariance_dimension,
            "covariance_full_rank": self.covariance_full_rank,
            "covariance_error": self.covariance_error,
            "metric_scope": "fixed_pulse_number_pint_global_equad",
            "causal_interpretation": (
                "consistent_with_underestimated_template_matching_uncertainty_or_"
                "pulse_jitter_like_toa_scatter"
            ),
            "cause_identified": False,
        }


@dataclass(frozen=True)
class FixedPathLikelihoodMetrics:
    log_likelihood: float
    bic: float
    chi2: float
    reduced_chi2: float
    weighted_rms_us: float
    parameter_count: int
    toa_count: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "log_likelihood": self.log_likelihood,
            "bic": self.bic,
            "chi2": self.chi2,
            "reduced_chi2": self.reduced_chi2,
            "weighted_rms_us": self.weighted_rms_us,
            "parameter_count": self.parameter_count,
            "toa_count": self.toa_count,
        }


def normalized_likelihood_bic(
    *,
    log_likelihood: float,
    parameter_count: int,
    toa_count: int,
) -> float:
    """Return BIC from a likelihood that includes its covariance normalization."""

    if not np.isfinite(log_likelihood):
        raise ValueError("Normalized log likelihood must be finite.")
    if int(parameter_count) < 0 or int(toa_count) <= 0:
        raise ValueError("BIC counts require nonnegative parameters and positive TOAs.")
    return float(
        -2.0 * float(log_likelihood)
        + int(parameter_count) * math.log(int(toa_count))
    )


def _residuals(toas: object, model: object) -> Residuals:
    return Residuals(
        toas,
        model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )


def evaluate_fixed_path_normalized_likelihood(
    *,
    model: object,
    toas: object,
    parameter_count: int | None = None,
) -> FixedPathLikelihoodMetrics:
    """Evaluate one fixed pulse-number model with its covariance normalization."""

    residuals = _residuals(toas, model)
    count = (
        len(tuple(model.free_params))
        if parameter_count is None
        else int(parameter_count)
    )
    toa_count = int(len(toas.table))
    log_likelihood = float(residuals.lnlikelihood())
    return FixedPathLikelihoodMetrics(
        log_likelihood=log_likelihood,
        bic=normalized_likelihood_bic(
            log_likelihood=log_likelihood,
            parameter_count=count,
            toa_count=toa_count,
        ),
        chi2=float(residuals.chi2),
        reduced_chi2=float(residuals.reduced_chi2),
        weighted_rms_us=float(residuals.rms_weighted().to_value(u.us)),
        parameter_count=count,
        toa_count=toa_count,
    )


def _covariance_diagnostics(
    *,
    fitter: object,
    parameter_names: Sequence[str],
) -> tuple[int, int, bool, str | None]:
    names = tuple(str(name).upper() for name in parameter_names)
    dimension = len(names)
    try:
        labeled = fitter.parameter_covariance_matrix.get_label_matrix(list(names))
        matrix = np.asarray(labeled.matrix, dtype=float)
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise ValueError("PINT returned an invalid white-noise covariance matrix.")
        diagonal = np.diag(matrix)
        if np.any(diagonal <= 0.0):
            raise ValueError("White-noise covariance has a nonpositive diagonal.")
        scale = np.sqrt(diagonal)
        correlation = matrix / np.outer(scale, scale)
        correlation = 0.5 * (correlation + correlation.T)
        rank = int(np.linalg.matrix_rank(correlation, tol=1.0e-8))
        return rank, dimension, rank == dimension, None
    except (
        ArithmeticError,
        AttributeError,
        KeyError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return 0, dimension, False, f"{type(error).__name__}: {error}"


def _initial_equad_us(*, residuals: Residuals, toas: object) -> float:
    values = np.asarray(residuals.time_resids.to_value(u.us), dtype=float)
    errors = np.asarray(toas.get_errors().to_value(u.us), dtype=float)
    excess_variance = max(
        float(np.mean(np.square(values)) - np.mean(np.square(errors))),
        0.0,
    )
    return float(max(math.sqrt(excess_variance), np.median(errors), 1.0e-6))


def _build_global_equad_model(
    *,
    model: object,
    toas: object,
    initial_equad_us: float,
) -> object:
    epochs = np.asarray(toas.get_mjds().value, dtype=float)
    lower = float(np.min(epochs) - 1.0)
    upper = float(np.max(epochs) + 1.0)
    source = model.as_parfile(format="pint")
    source += (
        f"\nEQUAD mjd {lower:.12f} {upper:.12f} "
        f"{float(initial_equad_us):.17g} 1\n"
    )
    conditioned = ModelBuilder()(StringIO(source), allow_tcb=True)
    if "UNITS" in conditioned.params and conditioned["UNITS"].value == "TCB":
        convert_tcb_tdb(conditioned)

    scale_component = conditioned.components.get("ScaleToaError")
    if scale_component is None:
        raise RuntimeError("PINT did not construct the global EQUAD component.")
    for name in tuple(scale_component.params):
        parameter = conditioned[name]
        if name != "EQUAD1" and parameter.value is None:
            scale_component.remove_param(name)
    conditioned.setup()
    conditioned.validate()
    return conditioned


def fit_global_equad_on_fixed_path(
    *,
    model: object,
    toas: object,
    config: ToaWhiteNoiseCertificationConfig | None = None,
) -> ToaWhiteNoiseFit:
    """Jointly refit timing parameters and one additive white-noise scale."""

    active = config or ToaWhiteNoiseCertificationConfig()
    timing_fit_parameters = tuple(str(name).upper() for name in model.free_params)
    reported = evaluate_fixed_path_normalized_likelihood(
        model=model,
        toas=toas,
        parameter_count=len(timing_fit_parameters),
    )

    conditioned_model = _build_global_equad_model(
        model=model,
        toas=toas,
        initial_equad_us=_initial_equad_us(
            residuals=_residuals(toas, model),
            toas=toas,
        ),
    )
    fitter = DownhillWLSFitter(
        toas,
        conditioned_model,
        track_mode="use_pulse_numbers",
    )
    fitter.fit_toas(
        maxiter=int(active.timing_fit_iterations),
        noise_fit_niter=int(active.noise_fit_iterations),
    )
    equad_us = abs(float(fitter.model.EQUAD1.value))
    fitter.model.EQUAD1.value = equad_us
    uncertainty = getattr(fitter.model.EQUAD1, "uncertainty_value", None)
    equad_uncertainty_us = (
        abs(float(uncertainty))
        if uncertainty is not None and np.isfinite(float(uncertainty))
        else None
    )
    noise_parameter_count = sum(
        name.startswith(("EFAC", "EQUAD", "ECORR"))
        for name in fitter.model.free_params
    )
    conditioned = evaluate_fixed_path_normalized_likelihood(
        model=fitter.model,
        toas=toas,
        parameter_count=len(timing_fit_parameters) + noise_parameter_count,
    )
    residuals = _residuals(toas, fitter.model)
    # PINT estimates EQUAD in the outer noise loop; its timing covariance
    # matrix therefore contains only the timing parameters.
    covariance_names = timing_fit_parameters
    covariance_rank, covariance_dimension, covariance_full_rank, covariance_error = (
        _covariance_diagnostics(
            fitter=fitter,
            parameter_names=covariance_names,
        )
    )
    spin_frequency_hz = abs(float(fitter.model.F0.value))
    equad_cycles = float(equad_us * 1.0e-6 * spin_frequency_hz)
    boundary_valid = bool(
        np.isfinite(equad_cycles)
        and 0.0 < equad_cycles < float(active.maximum_equad_cycles)
    )
    return ToaWhiteNoiseFit(
        model=fitter.model,
        fitter=fitter,
        residuals=residuals,
        reported_error_log_likelihood=reported.log_likelihood,
        reported_error_bic=reported.bic,
        noise_conditioned_log_likelihood=conditioned.log_likelihood,
        noise_conditioned_bic=conditioned.bic,
        delta_bic=float(reported.bic - conditioned.bic),
        equad_us=equad_us,
        equad_uncertainty_us=equad_uncertainty_us,
        equad_cycles=equad_cycles,
        boundary_valid=boundary_valid,
        chi2=conditioned.chi2,
        reduced_chi2=conditioned.reduced_chi2,
        weighted_rms_us=conditioned.weighted_rms_us,
        timing_fit_parameters=timing_fit_parameters,
        noise_parameter_count=int(noise_parameter_count),
        toa_count=conditioned.toa_count,
        covariance_rank=covariance_rank,
        covariance_dimension=covariance_dimension,
        covariance_full_rank=covariance_full_rank,
        covariance_error=covariance_error,
    )


def timing_hypothesis_identity(model: object) -> dict[str, Any]:
    """Identify fitted terms and fixed model inputs, excluding noise and metadata."""
    noise_prefixes = ("EFAC", "EQUAD", "ECORR", "TN")
    metadata = {"PSR", "TRACK", "NTOA", "CHI2", "CHI2R", "TRES", "START", "FINISH", "TZRMJD", "TZRSITE", "TZRFRQ"}
    free = sorted(name for name in model.free_params if not name.startswith(noise_prefixes))
    fixed = {}
    for name in model.params:
        parameter = model[name]
        if name in free or name in metadata or name.startswith(noise_prefixes) or parameter.value is None:
            continue
        fixed[name] = {"value": str(parameter.value), "key": getattr(parameter, "key", None),
                       "key_value": [str(value) for value in (getattr(parameter, "key_value", None) or [])]}
    return {"fit_parameters": free, "fixed_parameters": fixed}


def certify_isolated_toa_white_noise_portfolio(
    *,
    candidate_records: Sequence[Mapping[str, Any]],
    reported_error_winner_candidate_id: str,
    config: ToaWhiteNoiseCertificationConfig | None = None,
    disk_reload_tolerance: float = 1.0e-2,
) -> dict[str, Any]:
    """Certify a conditional solution across retained isolated pulse paths."""

    active = config or ToaWhiteNoiseCertificationConfig()
    if not np.isfinite(disk_reload_tolerance) or disk_reload_tolerance <= 0.0:
        raise ValueError("Disk reload tolerance must be positive and finite.")
    retained = [
        record
        for record in candidate_records
        if record.get("model_family") == "isolated" and "chi2" in record
    ]
    evaluated = []
    for record in retained:
        conditional = record.get("toa_white_noise_certification")
        if not isinstance(conditional, Mapping):
            continue
        fit = conditional.get("fit")
        if not isinstance(fit, Mapping):
            continue
        bic = float(fit.get("noise_conditioned_bic", math.inf))
        if not np.isfinite(bic):
            continue
        evaluated.append((bic, str(record.get("candidate_id")), record, conditional))
    evaluated.sort(key=lambda item: (item[0], item[1]))

    portfolio_complete = bool(retained) and len(evaluated) == len(retained)
    distinct = []
    seen_paths = set()
    for item in evaluated:
        signature = item[2].get("pulse_path_signature")
        if signature is None:
            continue
        signature = tuple(int(value) for value in signature)
        if signature not in seen_paths:
            distinct.append(item)
            seen_paths.add(signature)
    winner = distinct[0] if distinct else None
    runner_up = distinct[1] if len(distinct) > 1 else None
    winner_id = None if winner is None else winner[1]
    runner_up_id = None if runner_up is None else runner_up[1]
    path_delta_bic = (
        None if winner is None or runner_up is None else float(runner_up[0] - winner[0])
    )
    winner_record = None if winner is None else winner[2]
    winner_conditional = None if winner is None else winner[3]
    winner_fit = (
        {}
        if winner_conditional is None
        else winner_conditional.get("fit", {})
    )
    reported_winner = next(
        (
            record
            for record in retained
            if str(record.get("candidate_id"))
            == str(reported_error_winner_candidate_id)
        ),
        None,
    )
    reported_error_structural_gates_passed = bool(
        reported_winner is not None
        and not bool(reported_winner.get("accepted", False))
        and bool(reported_winner.get("support_valid", False))
        and bool(reported_winner.get("model_selection_complete", False))
        and float(reported_winner.get("disk_reload_chi2_delta", math.inf))
        <= float(disk_reload_tolerance)
    )
    requirements = {
        "reported_error_solution_failed_only_quality_acceptance": (
            reported_error_structural_gates_passed
        ),
        "conditional_path_portfolio_complete": portfolio_complete,
        "physical_path_identity_complete": all(record.get("pulse_path_signature") is not None for record in retained),
        "conditional_runner_up_available": runner_up is not None,
        "reported_error_winner_unchanged": bool(
            winner_record is not None and reported_winner is not None
            and reported_winner.get("pulse_path_signature") is not None
            and tuple(winner_record["pulse_path_signature"]) == tuple(reported_winner["pulse_path_signature"])
        ),
        "reported_timing_hypothesis_unchanged": bool(
            reported_winner is not None and winner_conditional is not None
            and reported_winner.get("timing_hypothesis") is not None
            and reported_winner["timing_hypothesis"] == winner_conditional.get("timing_hypothesis")
        ),
        "observation_white_noise_adequacy_passed": bool(
            winner_conditional is not None
            and winner_conditional.get("observation_white_noise", {}).get("passed", False)
        ),
        "conditional_path_delta_bic_passed": bool(
            path_delta_bic is not None
            and path_delta_bic >= active.minimum_path_delta_bic
        ),
        "timing_model_selection_complete": bool(
            winner_record is not None
            and winner_record.get("model_selection_complete", False)
            and winner_conditional is not None
            and winner_conditional.get("model_selection_complete", False)
        ),
        "timing_model_selection_consistent": bool(
            winner_conditional is not None
            and winner_conditional.get("model_selection_consistent", False)
        ),
        "noise_model_delta_bic_passed": bool(
            float(winner_fit.get("delta_bic", -math.inf))
            >= active.minimum_noise_delta_bic
        ),
        "equad_boundary_valid": bool(winner_fit.get("boundary_valid", False)),
        "conditional_support_valid": bool(
            winner_conditional is not None
            and winner_conditional.get("support_valid", False)
        ),
        "conditional_disk_reload_consistent": bool(
            winner_conditional is not None
            and winner_conditional.get("disk_reload_consistent", False)
        ),
    }
    accepted = bool(requirements and all(requirements.values()))
    return {
        "status": (
            "noise_conditioned_unique_solution"
            if accepted
            else "suspected_toa_generation_noise_not_certified"
        ),
        "accepted": accepted,
        "applies_to": "isolated_single_star_only",
        "noise_model": "global_equad_like_toa_generation_scatter",
        "reported_error_winner_candidate_id": str(
            reported_error_winner_candidate_id
        ),
        "noise_conditioned_winner_candidate_id": winner_id,
        "noise_conditioned_runner_up_candidate_id": runner_up_id,
        "retained_candidate_count": len(retained),
        "evaluated_candidate_count": len(evaluated),
        "retained_path_count": len({tuple(r["pulse_path_signature"]) for r in retained if r.get("pulse_path_signature") is not None}),
        "evaluated_path_count": len(distinct),
        "noise_conditioned_path_delta_bic": path_delta_bic,
        "minimum_noise_delta_bic": active.minimum_noise_delta_bic,
        "minimum_path_delta_bic": active.minimum_path_delta_bic,
        "maximum_equad_cycles": active.maximum_equad_cycles,
        "requirements": requirements,
        "uniqueness_scope": "finite_retained_phase_path_portfolio",
        "reported_error_metric_scope": (
            "fixed_pulse_number_pint_reported_toa_errors"
        ),
        "noise_conditioned_metric_scope": (
            "fixed_pulse_number_pint_global_equad"
        ),
        "causal_interpretation": (
            "additive_white_noise_hypothesis_checked_conditionally_not_a_physical_cause_identification"
        ),
        "cause_identified": False,
    }


def observation_white_noise_diagnostic(
    *, residual_us: np.ndarray, sigma_us: np.ndarray, design: np.ndarray,
    groups: Sequence[Sequence[int]], simulations: int = 8192,
) -> dict[str, Any]:
    """Check session-common residual power conditional on the fitted design.

    Project simulated white errors through the timing fit before comparison.
    This is an adequacy veto, not a selection-calibrated discovery p-value.
    """
    r = np.asarray(residual_us, dtype=float)
    if simulations < 1:
        raise ValueError("White-noise diagnostic simulation count must be positive.")
    sigma = np.asarray(sigma_us, dtype=float)
    matrix = np.asarray(design, dtype=float)
    if (r.ndim != 1 or sigma.shape != r.shape or matrix.ndim != 2 or matrix.shape[0] != r.size
            or np.any(sigma <= 0) or not all(np.all(np.isfinite(x)) for x in (r, sigma, matrix))):
        raise ValueError("White-noise diagnostic requires finite aligned residuals, errors and design.")
    indices = [np.asarray(group, dtype=int) for group in groups]
    if not indices or any(group.size == 0 for group in indices) or sorted(np.concatenate(indices).tolist()) != list(range(r.size)):
        raise ValueError("Observation groups must partition the TOAs.")
    scope = "conditional_fixed_path_linearized_fit_not_selection_calibrated"
    if len(groups) < 3 or len(groups) == r.size:
        return {"passed": False, "status": "insufficient_group_replication", "scope": scope}
    white_design = matrix / sigma[:, None]
    scales = np.linalg.norm(white_design, axis=0)
    white_design = white_design[:, scales > 0] / scales[scales > 0]
    left, singular, _ = np.linalg.svd(white_design, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-12)) if singular.size else 0
    q = left[:, :rank]
    if r.size <= rank:
        return {"passed": False, "status": "no_residual_degrees_of_freedom", "scope": scope}
    def fraction(z):
        total = np.sum(z*z, axis=0)
        between = np.zeros_like(total)
        for ix in indices:
            direction = 1 / sigma[ix]
            direction /= np.linalg.norm(direction)
            between += np.square(direction @ z[ix])
        return between / np.maximum(total, np.finfo(float).tiny)
    observed = float(fraction(r / sigma))
    rng = np.random.default_rng(91202)
    exceed = 0
    for start in range(0, simulations, 256):
        z = rng.normal(size=(r.size, min(256, simulations - start)))
        z -= q @ (q.T @ z)
        exceed += int(np.sum(fraction(z) >= observed))
    tail = (1 + exceed) / (1 + simulations)
    return {"passed": tail >= .01, "status": "evaluated", "between_observation_fraction": observed,
            "conditional_upper_tail": tail, "minimum_upper_tail": .01, "simulations": simulations,
            "timing_design_rank": rank, "observation_count": len(groups), "scope": scope}


def evaluate_observation_white_noise(*, model: object, toas: object) -> dict[str, Any]:
    try:
        design, _names, _units = model.designmatrix(toas, incfrozen=False, incoffset=True)
        return observation_white_noise_diagnostic(
            residual_us=np.asarray(_residuals(toas, model).time_resids.to_value(u.us)),
            sigma_us=np.asarray(model.scaled_toa_uncertainty(toas).to_value(u.us)),
            design=design,
            groups=[session.toa_indices for session in infer_contiguous_source_sessions(toas)],
        )
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        return {"passed": False, "status": "evaluation_failed", "error": str(error)}


__all__ = [
    "FixedPathLikelihoodMetrics",
    "ToaWhiteNoiseCertificationConfig",
    "ToaWhiteNoiseFit",
    "certify_isolated_toa_white_noise_portfolio",
    "evaluate_fixed_path_normalized_likelihood",
    "fit_global_equad_on_fixed_path",
    "normalized_likelihood_bic",
]
