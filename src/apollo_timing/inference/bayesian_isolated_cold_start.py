from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from apollo_timing.inference.bayesian_robust_phase_likelihood import (
    RobustPhaseNoiseConfig,
)


@dataclass(frozen=True)
class IsolatedColdStartConfig:
    """Budgets and physical support for isolated-pulsar cold starts."""

    enabled: bool = True
    astrometric_bound_arcsec: float = 7200.0
    proper_motion_half_width_mas_per_year: float = 500.0
    frequency_half_width_hz: float = 5.0e-4
    frequency_derivative_half_width_hz_s: float = 2.0e-11
    proposal_frequency_half_width_hz: float = 2.0e-5
    proposal_frequency_derivative_half_width_hz_s: float = 2.0e-13
    sobol_power: int = 4
    maximum_function_evaluations: int = 160
    retained_modes: int = 8
    minimum_normalized_mode_separation: float = 0.02
    component_prior_sigma: float = 0.15
    robust_loss_scale: float = 1.5
    random_seed: int = 2337
    phase_noise: RobustPhaseNoiseConfig = field(
        default_factory=RobustPhaseNoiseConfig
    )

    def __post_init__(self) -> None:
        positive = (
            self.astrometric_bound_arcsec,
            self.proper_motion_half_width_mas_per_year,
            self.frequency_half_width_hz,
            self.frequency_derivative_half_width_hz_s,
            self.proposal_frequency_half_width_hz,
            self.proposal_frequency_derivative_half_width_hz_s,
            self.minimum_normalized_mode_separation,
            self.component_prior_sigma,
            self.robust_loss_scale,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Isolated cold-start scales must be finite and positive.")
        if not 1 <= int(self.sobol_power) <= 12:
            raise ValueError("The isolated Sobol power must lie between 1 and 12.")
        if int(self.maximum_function_evaluations) < 8 or int(self.retained_modes) <= 0:
            raise ValueError("Isolated cold-start budgets must be positive.")
        if self.proposal_frequency_half_width_hz > self.frequency_half_width_hz:
            raise ValueError("The cold-start F0 proposal must lie inside final support.")
        if (
            self.proposal_frequency_derivative_half_width_hz_s
            > self.frequency_derivative_half_width_hz_s
        ):
            raise ValueError("The cold-start F1 proposal must lie inside final support.")


@dataclass(frozen=True)
class IsolatedColdStartMode:
    mode_id: str
    source: str
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    slope_deviance: float
    function_evaluations: int
    converged: bool


def _session_slope_residuals(
    evaluator: object,
    parameter_values: Sequence[float],
) -> np.ndarray:
    phase = np.asarray(
        evaluator.raw_phase_residuals(parameter_values),
        dtype=float,
    )
    epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
    sigmas = np.asarray(evaluator.phase_sigmas(), dtype=float)
    result: list[float] = []
    for session in evaluator.sessions:
        indices = np.asarray(session.toa_indices, dtype=int)
        if indices.size < 2:
            continue
        seconds = (epochs[indices] - float(np.mean(epochs[indices]))) * 86400.0
        relative = np.unwrap(
            phase[indices] - float(phase[indices[0]]),
            period=1.0,
        )
        design = np.column_stack((np.ones(indices.size), seconds))
        weights = 1.0 / np.square(sigmas[indices])
        normal = design.T @ (weights[:, None] * design)
        try:
            covariance = np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            continue
        coefficients = covariance @ (design.T @ (weights * relative))
        slope_sigma = math.sqrt(max(float(covariance[1, 1]), np.finfo(float).tiny))
        result.append(float(coefficients[1]) / slope_sigma)
    if len(result) < max(3, len(evaluator.parameter_names)):
        raise ValueError("Cold-start slope inference has too few informative sessions.")
    return np.asarray(result, dtype=float)


def _soft_l1_deviance(residuals: np.ndarray, scale: float) -> float:
    normalized = np.asarray(residuals, dtype=float) / float(scale)
    return float(
        np.sum(2.0 * scale * scale * (np.sqrt(1.0 + normalized * normalized) - 1.0))
    )


def _proposal_normalized_bounds(
    *,
    evaluator: object,
    support: object,
    config: IsolatedColdStartConfig,
) -> tuple[np.ndarray, np.ndarray]:
    physical_lower = np.asarray(support.physical_lower, dtype=float).copy()
    physical_upper = np.asarray(support.physical_upper, dtype=float).copy()
    physical_center = np.asarray(support.physical_center, dtype=float)
    by_name = {
        str(name).upper(): index
        for index, name in enumerate(evaluator.parameter_names)
    }
    if "F0" in by_name:
        index = by_name["F0"]
        width = float(config.proposal_frequency_half_width_hz)
        physical_lower[index] = max(physical_lower[index], physical_center[index] - width)
        physical_upper[index] = min(physical_upper[index], physical_center[index] + width)
    if "F1" in by_name:
        index = by_name["F1"]
        width = float(config.proposal_frequency_derivative_half_width_hz_s)
        physical_lower[index] = max(physical_lower[index], physical_center[index] - width)
        physical_upper[index] = min(physical_upper[index], physical_center[index] + width)
    corners = np.asarray(
        [
            support.normalize_physical(values)
            for values in product(
                *zip(physical_lower, physical_upper, strict=True)
            )
        ],
        dtype=float,
    )
    epsilon = 64.0 * np.finfo(float).eps
    lower = np.maximum(np.min(corners, axis=0), -1.0 + epsilon)
    upper = np.minimum(np.max(corners, axis=0), 1.0 - epsilon)
    if np.any(lower >= upper):
        raise ValueError("The isolated cold-start proposal has empty support.")
    return lower, upper


def search_isolated_cold_start_modes(
    *,
    evaluator: object,
    support: object,
    config: IsolatedColdStartConfig | None = None,
) -> tuple[IsolatedColdStartMode, ...]:
    """Generate a position-spin portfolio before using any session integers."""

    active = config or IsolatedColdStartConfig()
    if not active.enabled:
        return tuple()
    dimension = int(support.dimension)
    if dimension != len(evaluator.parameter_names):
        raise ValueError("The cold-start support must match every fitted parameter.")
    lower, upper = _proposal_normalized_bounds(
        evaluator=evaluator,
        support=support,
        config=active,
    )
    center = support.normalize_search(support.search_center)

    def residual(normalized: np.ndarray) -> np.ndarray:
        try:
            physical = support.denormalize_physical(normalized)
            if support.contains_physical(physical):
                return _session_slope_residuals(evaluator, physical)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        return np.full(len(evaluator.sessions), 1.0e8, dtype=float)

    starts = [np.asarray(center, dtype=float)]
    sobol = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=int(active.random_seed),
    ).random_base2(int(active.sobol_power))
    starts.extend(lower + sobol * (upper - lower))

    candidates: list[IsolatedColdStartMode] = []
    center_residual = residual(center)
    candidates.append(
        IsolatedColdStartMode(
            mode_id="cold_initial_center",
            source="input_center",
            normalized_values=np.asarray(center, dtype=float),
            parameter_values=support.denormalize_physical(center),
            slope_deviance=_soft_l1_deviance(
                center_residual,
                active.robust_loss_scale,
            ),
            function_evaluations=1,
            converged=True,
        )
    )
    for start_index, start in enumerate(starts):
        if not support.contains_physical(support.denormalize_physical(start)):
            continue
        fit = least_squares(
            residual,
            np.clip(start, lower, upper),
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            x_scale="jac",
            loss="soft_l1",
            f_scale=float(active.robust_loss_scale),
            max_nfev=max(int(active.maximum_function_evaluations), 8),
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        state = np.asarray(fit.x, dtype=float)
        values = support.denormalize_physical(state)
        if not support.contains_physical(values):
            continue
        current = residual(state)
        candidates.append(
            IsolatedColdStartMode(
                mode_id=f"cold_slope_{start_index:04d}",
                source="session_slope_sobol",
                normalized_values=state,
                parameter_values=np.asarray(values, dtype=float),
                slope_deviance=_soft_l1_deviance(
                    current,
                    active.robust_loss_scale,
                ),
                function_evaluations=int(fit.nfev),
                converged=bool(fit.success),
            )
        )

    candidates.sort(
        key=lambda item: (
            not item.converged,
            item.slope_deviance,
            item.mode_id,
        )
    )
    selected: list[IsolatedColdStartMode] = []
    for candidate in candidates:
        if any(
            np.linalg.norm(candidate.normalized_values - existing.normalized_values)
            < float(active.minimum_normalized_mode_separation)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= int(active.retained_modes):
            break
    center_mode = next(item for item in candidates if item.source == "input_center")
    if not any(item.source == "input_center" for item in selected):
        if len(selected) >= int(active.retained_modes):
            selected[-1] = center_mode
        else:
            selected.append(center_mode)
    selected.sort(key=lambda item: (item.slope_deviance, item.mode_id))
    return tuple(selected)


__all__ = [
    "IsolatedColdStartConfig",
    "IsolatedColdStartMode",
    "search_isolated_cold_start_modes",
]
