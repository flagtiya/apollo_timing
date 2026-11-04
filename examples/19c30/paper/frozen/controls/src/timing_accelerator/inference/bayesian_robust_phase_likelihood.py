from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class RobustPhaseNoiseConfig:
    """A small contamination model for phase-lattice proposal generation."""

    enabled: bool = True
    contamination_probability: float = 0.02
    contamination_sigma_cycles: float = 0.25
    minimum_jitter_cycles: float = 0.0
    jitter_mad_scale: float = 1.0
    maximum_jitter_cycles: float = 0.02
    maximum_em_iterations: int = 12

    def __post_init__(self) -> None:
        probability = float(self.contamination_probability)
        positive = (
            self.contamination_sigma_cycles,
            self.jitter_mad_scale,
            self.maximum_jitter_cycles,
        )
        if not 0.0 < probability < 0.5:
            raise ValueError("The phase-contamination probability must lie in (0, 0.5).")
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Robust phase-noise scales must be finite and positive.")
        if (
            not np.isfinite(self.minimum_jitter_cycles)
            or self.minimum_jitter_cycles < 0.0
            or self.minimum_jitter_cycles > self.maximum_jitter_cycles
        ):
            raise ValueError("The minimum phase jitter must lie inside its upper bound.")
        if int(self.maximum_em_iterations) <= 0:
            raise ValueError("The robust phase likelihood needs at least one EM iteration.")


@dataclass(frozen=True)
class RobustPhaseProfile:
    normalized_local_residuals: np.ndarray
    session_intercepts: np.ndarray
    session_intercept_variances: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    local_deviance: float
    local_log_likelihood: float
    contamination_probabilities: np.ndarray


def _reported_vector(values: Sequence[float], size: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result) & (result > 0.0)):
        raise ValueError(f"{label} must be a positive finite vector of length {size}.")
    return result


def _linear_session_residuals(
    phase_values: np.ndarray,
    epochs_mjd: np.ndarray,
    sessions: Sequence[object],
) -> np.ndarray:
    residuals: list[float] = []
    for session in sessions:
        indices = np.asarray(session.toa_indices, dtype=int)
        if indices.size < 3:
            continue
        seconds = (epochs_mjd[indices] - float(np.mean(epochs_mjd[indices]))) * 86400.0
        relative = np.unwrap(
            phase_values[indices] - float(phase_values[indices[0]]),
            period=1.0,
        )
        design = np.column_stack((np.ones(indices.size), seconds))
        coefficients = np.linalg.lstsq(design, relative, rcond=None)[0]
        residuals.extend((relative - design @ coefficients).tolist())
    return np.asarray(residuals, dtype=float)


@dataclass(frozen=True)
class RobustPhaseNoiseModel:
    reported_sigmas: np.ndarray
    core_sigmas: np.ndarray
    jitter_cycles: float
    config: RobustPhaseNoiseConfig

    @classmethod
    def infer(
        cls,
        *,
        phase_values: Sequence[float],
        reported_sigmas: Sequence[float],
        epochs_mjd: Sequence[float],
        sessions: Sequence[object],
        config: RobustPhaseNoiseConfig | None = None,
    ) -> "RobustPhaseNoiseModel":
        active = config or RobustPhaseNoiseConfig()
        phase = np.asarray(phase_values, dtype=float)
        epochs = np.asarray(epochs_mjd, dtype=float)
        if phase.ndim != 1 or epochs.shape != phase.shape or not np.all(
            np.isfinite(phase) & np.isfinite(epochs)
        ):
            raise ValueError("Phase-noise inference needs aligned finite phase epochs.")
        reported = _reported_vector(
            reported_sigmas,
            phase.size,
            "Reported phase uncertainties",
        )
        detrended = _linear_session_residuals(phase, epochs, sessions)
        if detrended.size:
            center = float(np.median(detrended))
            mad = 1.4826 * float(np.median(np.abs(detrended - center)))
        else:
            mad = 0.0
        jitter = max(
            float(active.minimum_jitter_cycles),
            float(active.jitter_mad_scale) * mad,
            float(np.median(reported)),
        )
        jitter = min(jitter, float(active.maximum_jitter_cycles))
        core = np.sqrt(np.square(reported) + jitter * jitter)
        return cls(
            reported_sigmas=reported.copy(),
            core_sigmas=core,
            jitter_cycles=float(jitter),
            config=active,
        )

    def _profile_intercept(
        self,
        shifted: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, float, float, float]:
        core = self.core_sigmas[indices]
        probability = float(self.config.contamination_probability)
        outlier_sigma = float(self.config.contamination_sigma_cycles)
        log_core_norm = (
            math.log1p(-probability)
            - np.log(core)
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_outlier_norm = (
            math.log(probability)
            - math.log(outlier_sigma)
            - 0.5 * math.log(2.0 * math.pi)
        )
        intercept = float(np.median(shifted))
        core_probability = np.ones(shifted.size, dtype=float)
        for _ in range(int(self.config.maximum_em_iterations)):
            delta = shifted - intercept
            log_core = log_core_norm - 0.5 * np.square(delta / core)
            log_outlier = log_outlier_norm - 0.5 * np.square(delta / outlier_sigma)
            normalization = logsumexp(
                np.vstack((log_core, log_outlier)),
                axis=0,
            )
            core_probability = np.exp(log_core - normalization)
            weights = (
                core_probability / np.square(core)
                + (1.0 - core_probability) / outlier_sigma**2
            )
            updated = float(np.sum(weights * shifted) / np.sum(weights))
            if abs(updated - intercept) <= 1.0e-12:
                intercept = updated
                break
            intercept = updated

        delta = shifted - intercept
        log_core = log_core_norm - 0.5 * np.square(delta / core)
        log_outlier = log_outlier_norm - 0.5 * np.square(delta / outlier_sigma)
        log_mixture = logsumexp(np.vstack((log_core, log_outlier)), axis=0)
        zero_log_mixture = logsumexp(
            np.vstack(
                (
                    log_core_norm,
                    np.full(shifted.size, log_outlier_norm, dtype=float),
                )
            ),
            axis=0,
        )
        deviance = np.maximum(2.0 * (zero_log_mixture - log_mixture), 0.0)
        outlier_probability = np.exp(log_outlier - log_mixture)
        information = np.sum(
            (1.0 - outlier_probability) / np.square(core)
            + outlier_probability / outlier_sigma**2
        )
        variance = float(1.0 / max(information, np.finfo(float).tiny))
        return (
            intercept,
            delta,
            deviance,
            float(np.sum(log_mixture)),
            variance,
            outlier_probability,
        )

    def profile(
        self,
        phase_residuals: Sequence[float],
        sessions: Sequence[object],
        *,
        fixed_local_integer_offsets: Sequence[Sequence[int]] | None = None,
    ) -> RobustPhaseProfile:
        residuals = np.asarray(phase_residuals, dtype=float)
        if residuals.shape != self.core_sigmas.shape or not np.all(np.isfinite(residuals)):
            raise ValueError("Robust phase residuals must align with the inferred noise model.")
        session_tuple = tuple(sessions)
        fixed = (
            None
            if fixed_local_integer_offsets is None
            else tuple(
                tuple(int(value) for value in offsets)
                for offsets in fixed_local_integer_offsets
            )
        )
        if fixed is not None and len(fixed) != len(session_tuple):
            raise ValueError("A fixed local integer component is required per session.")

        normalized = np.empty_like(residuals)
        contamination = np.empty_like(residuals)
        intercepts: list[float] = []
        variances: list[float] = []
        integer_modes: list[tuple[int, ...]] = []
        total_deviance = 0.0
        total_log_likelihood = 0.0
        for session_index, session in enumerate(session_tuple):
            indices = np.asarray(session.toa_indices, dtype=int)
            if fixed is None:
                local = residuals[indices]
                integers = np.concatenate(
                    (
                        np.zeros(1, dtype=int),
                        np.rint(local[1:] - local[0]).astype(int),
                    )
                )
            else:
                integers = np.asarray(fixed[session_index], dtype=int)
                if integers.shape != indices.shape or int(integers[0]) != 0:
                    raise ValueError(
                        "Each local integer component must align with its session anchor."
                    )
            shifted = residuals[indices] - integers
            (
                intercept,
                delta,
                deviance,
                log_likelihood,
                variance,
                outlier_probability,
            ) = self._profile_intercept(shifted, indices)
            normalized[indices] = np.sign(delta) * np.sqrt(deviance)
            contamination[indices] = outlier_probability
            intercepts.append(intercept)
            variances.append(variance)
            integer_modes.append(tuple(int(value) for value in integers))
            total_deviance += float(np.sum(deviance))
            total_log_likelihood += log_likelihood
        return RobustPhaseProfile(
            normalized_local_residuals=normalized,
            session_intercepts=np.asarray(intercepts, dtype=float),
            session_intercept_variances=np.asarray(variances, dtype=float),
            local_integer_offsets=tuple(integer_modes),
            local_deviance=float(total_deviance),
            local_log_likelihood=float(total_log_likelihood),
            contamination_probabilities=contamination,
        )


__all__ = [
    "RobustPhaseNoiseConfig",
    "RobustPhaseNoiseModel",
    "RobustPhaseProfile",
]
