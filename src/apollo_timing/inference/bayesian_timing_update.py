from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.residuals import Residuals

from apollo_timing.inference.bayesian_binary_lattice import (
    BinaryLatticeSearchContext,
    ContinuousPriorComponent,
    PintPhaseEvaluator,
)
from apollo_timing.inference.pint_solution_artifacts import (
    extract_fitter_parameter_covariance,
)


@dataclass(frozen=True)
class TimingUpdatePriorConfig:
    """Posterior mixture used when extending an existing timing solution."""

    baseline_posterior_weight: float = 0.98
    broad_escape_weight: float = 0.02
    posterior_covariance_scale: float = 4.0
    minimum_normalized_sigma: float = 1.0e-7
    maximum_normalized_sigma: float = 5.0e-2
    broad_escape_normalized_sigma: float = 0.20
    baseline_fit_iterations: int = 10

    def __post_init__(self) -> None:
        weights = (
            float(self.baseline_posterior_weight),
            float(self.broad_escape_weight),
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("Timing-update prior weights must be positive and finite.")
        if not np.isclose(sum(weights), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("Timing-update prior weights must sum to one.")
        scales = (
            self.posterior_covariance_scale,
            self.minimum_normalized_sigma,
            self.maximum_normalized_sigma,
            self.broad_escape_normalized_sigma,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("Timing-update prior scales must be positive and finite.")
        if self.minimum_normalized_sigma >= self.maximum_normalized_sigma:
            raise ValueError("Timing-update covariance floors must be ordered.")
        if int(self.baseline_fit_iterations) <= 0:
            raise ValueError("Baseline timing-update fit iterations must be positive.")


@dataclass(frozen=True)
class TimingUpdateConditioning:
    baseline_par: Path
    baseline_tim: Path
    fit_parameters: tuple[str, ...]
    baseline_toa_count: int
    baseline_session_count: int
    baseline_chi2: float
    baseline_reduced_chi2: float
    baseline_weighted_rms_us: float
    physical_mean: np.ndarray
    physical_covariance: np.ndarray
    normalized_mean: np.ndarray
    normalized_covariance: np.ndarray
    baseline_covered_merged_sessions: tuple[int, ...]
    fixed_relative_integer_offsets: tuple[tuple[int, int], ...]
    components: tuple[ContinuousPriorComponent, ...]


def _certified_baseline_toas(evaluator: PintPhaseEvaluator) -> object:
    toas = deepcopy(evaluator.toas)
    pulse_numbers = np.asarray(evaluator.base_pulse_numbers, dtype=float).copy()
    toas.table["pulse_number"] = pulse_numbers
    toas.table["pulse_number"].unit = u.dimensionless_unscaled
    return toas


def _fallback_physical_covariance(
    evaluator: PintPhaseEvaluator,
    context: BinaryLatticeSearchContext,
) -> np.ndarray:
    variances = []
    for index, name in enumerate(evaluator.parameter_names):
        uncertainty = getattr(evaluator.model[name], "uncertainty_value", None)
        if uncertainty is not None and np.isfinite(float(uncertainty)) and float(
            uncertainty
        ) > 0.0:
            sigma = abs(float(uncertainty))
        else:
            sigma = float(
                context.support.physical_upper[index]
                - context.support.physical_lower[index]
            ) / 20.0
        variances.append(max(sigma * sigma, np.finfo(float).tiny))
    return np.diag(np.asarray(variances, dtype=float))


def _normalization_jacobian(
    context: BinaryLatticeSearchContext,
    physical_mean: np.ndarray,
    physical_covariance: np.ndarray,
) -> np.ndarray:
    dimension = physical_mean.size
    jacobian = np.empty((dimension, dimension), dtype=float)
    lower = np.asarray(context.support.physical_lower, dtype=float)
    upper = np.asarray(context.support.physical_upper, dtype=float)
    support_width = upper - lower
    for index in range(dimension):
        covariance_sigma = math.sqrt(max(float(physical_covariance[index, index]), 0.0))
        requested = max(
            covariance_sigma * 0.1,
            support_width[index] * 1.0e-7,
            abs(float(physical_mean[index])) * 1.0e-10,
            np.finfo(float).eps,
        )
        plus_step = min(requested, float(upper[index] - physical_mean[index]))
        minus_step = min(requested, float(physical_mean[index] - lower[index]))
        if plus_step > 0.0 and minus_step > 0.0:
            plus = physical_mean.copy()
            minus = physical_mean.copy()
            plus[index] += plus_step
            minus[index] -= minus_step
            jacobian[:, index] = (
                context.support.normalize_physical(plus)
                - context.support.normalize_physical(minus)
            ) / (plus_step + minus_step)
        elif plus_step > 0.0:
            plus = physical_mean.copy()
            plus[index] += plus_step
            jacobian[:, index] = (
                context.support.normalize_physical(plus)
                - context.support.normalize_physical(physical_mean)
            ) / plus_step
        elif minus_step > 0.0:
            minus = physical_mean.copy()
            minus[index] -= minus_step
            jacobian[:, index] = (
                context.support.normalize_physical(physical_mean)
                - context.support.normalize_physical(minus)
            ) / minus_step
        else:
            raise ValueError("The baseline posterior mean is pinned to an empty support.")
    return jacobian


def _regularize_normalized_covariance(
    covariance: np.ndarray,
    config: TimingUpdatePriorConfig,
) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(covariance, dtype=float) + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum = float(config.minimum_normalized_sigma) ** 2
    maximum = float(config.maximum_normalized_sigma) ** 2
    clipped = np.clip(eigenvalues, minimum, maximum)
    result = (eigenvectors * clipped) @ eigenvectors.T
    return 0.5 * (result + result.T)


def _covered_merged_sessions(
    context: BinaryLatticeSearchContext,
    baseline_epochs_mjd: Sequence[float],
) -> tuple[int, ...]:
    baseline = np.asarray(baseline_epochs_mjd, dtype=float)
    merged = np.asarray(context.evaluator.observation_epochs_mjd, dtype=float)
    covered = []
    for session_index, session in enumerate(context.evaluator.sessions):
        session_epochs = merged[np.asarray(session.toa_indices, dtype=int)]
        if any(
            np.any(np.isclose(epoch, baseline, rtol=0.0, atol=5.0e-10))
            for epoch in session_epochs
        ):
            covered.append(session_index)
    return tuple(covered)


def condition_timing_update_context(
    *,
    context: BinaryLatticeSearchContext,
    baseline_par_path: Path,
    baseline_tim_path: Path,
    clock_override_path: Path | None = None,
    config: TimingUpdatePriorConfig | None = None,
) -> tuple[BinaryLatticeSearchContext, TimingUpdateConditioning]:
    """Condition a merged-data phase lattice on the pre-update posterior."""

    active = config or TimingUpdatePriorConfig()
    baseline_par = Path(baseline_par_path).resolve()
    baseline_tim = Path(baseline_tim_path).resolve()
    baseline = PintPhaseEvaluator.from_files(
        par_path=baseline_par,
        tim_path=baseline_tim,
        clock_override_path=clock_override_path,
    )
    if baseline.model_name != context.evaluator.model_name:
        raise ValueError("Timing update changed the declared PINT model family.")
    if tuple(baseline.parameter_names) != tuple(context.evaluator.parameter_names):
        raise ValueError("Timing update changed the input fit-parameter contract.")

    certified_toas = _certified_baseline_toas(baseline)
    fitter = WLSFitter(
        certified_toas,
        baseline.model,
        track_mode="use_pulse_numbers",
    )
    fitter.fit_toas(maxiter=int(active.baseline_fit_iterations))
    residuals = Residuals(
        certified_toas,
        fitter.model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    physical_mean = np.asarray(
        [float(fitter.model[name].value) for name in baseline.parameter_names],
        dtype=float,
    )
    if not context.support.contains_physical(physical_mean):
        physical_mean = np.asarray(context.support.physical_center, dtype=float).copy()
    try:
        physical_covariance = extract_fitter_parameter_covariance(
            fitter,
            baseline.parameter_names,
        )
    except (AttributeError, RuntimeError, ValueError, np.linalg.LinAlgError):
        physical_covariance = _fallback_physical_covariance(baseline, context)
    physical_covariance = (
        float(active.posterior_covariance_scale) ** 2
        * np.asarray(physical_covariance, dtype=float)
    )
    normalized_mean = context.support.normalize_physical(physical_mean)
    jacobian = _normalization_jacobian(
        context,
        physical_mean,
        physical_covariance,
    )
    normalized_covariance = _regularize_normalized_covariance(
        jacobian @ physical_covariance @ jacobian.T,
        active,
    )

    covered = _covered_merged_sessions(
        context,
        baseline.observation_epochs_mjd,
    )
    fixed_offsets = (
        tuple((session_index - 1, 0) for session_index in covered if session_index > 0)
        if covered and covered[0] == 0
        else tuple()
    )
    broad_mean = context.support.normalize_search(context.support.search_center)
    dimension = context.support.dimension
    components = (
        ContinuousPriorComponent(
            component_id="baseline_posterior",
            normalized_mean=normalized_mean,
            normalized_covariance=normalized_covariance,
            log_weight=math.log(float(active.baseline_posterior_weight)),
            fixed_integer_offsets=fixed_offsets,
        ),
        ContinuousPriorComponent(
            component_id="broad_escape",
            normalized_mean=broad_mean,
            normalized_covariance=np.diag(
                np.full(
                    dimension,
                    float(active.broad_escape_normalized_sigma) ** 2,
                    dtype=float,
                )
            ),
            log_weight=math.log(float(active.broad_escape_weight)),
        ),
    )
    conditioned_context = replace(
        context,
        continuous_prior_components=components,
    )
    conditioning = TimingUpdateConditioning(
        baseline_par=baseline_par,
        baseline_tim=baseline_tim,
        fit_parameters=tuple(baseline.parameter_names),
        baseline_toa_count=len(baseline.toas.table),
        baseline_session_count=len(baseline.sessions),
        baseline_chi2=float(residuals.chi2),
        baseline_reduced_chi2=float(residuals.reduced_chi2),
        baseline_weighted_rms_us=float(residuals.rms_weighted().to_value(u.us)),
        physical_mean=physical_mean,
        physical_covariance=physical_covariance,
        normalized_mean=normalized_mean,
        normalized_covariance=normalized_covariance,
        baseline_covered_merged_sessions=covered,
        fixed_relative_integer_offsets=fixed_offsets,
        components=components,
    )
    return conditioned_context, conditioning


__all__ = [
    "TimingUpdateConditioning",
    "TimingUpdatePriorConfig",
    "condition_timing_update_context",
]
