from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Sequence

import dynesty
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, ndtr, ndtri


@dataclass(frozen=True)
class IsolatedTimingUpdateHypothesis:
    """One physical model in the isolated timing-update portfolio."""

    hypothesis_id: str
    profile_state_count: int = 1
    glitch_frequency_order: int = 0

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("An isolated update hypothesis needs an id.")
        if int(self.profile_state_count) not in {1, 2}:
            raise ValueError("The first isolated update backend supports one or two states.")
        if int(self.glitch_frequency_order) not in {0, 1, 2}:
            raise ValueError("Glitch frequency order must be zero, one, or two.")


@dataclass(frozen=True)
class IsolatedTimingUpdateConfig:
    """Declared support for a one-year isolated-pulsar timing update."""

    astrometric_half_width_arcsec: float = 600.0
    frequency_half_width_hz: float = 2.0e-7
    frequency_derivative_half_width_hz_s: float = 2.0e-14
    minimum_jitter_cycles: float = 1.0e-6
    maximum_jitter_cycles: float = 2.0e-2
    integer_radius: int = 2
    student_t_degrees_of_freedom: float = 4.0
    minimum_state_offset_cycles: float = 0.02
    maximum_state_offset_cycles: float = 0.5
    minimum_state_weight: float = 0.02
    maximum_state_weight: float = 0.98
    glitch_epoch_lower_mjd: float | None = None
    glitch_epoch_upper_mjd: float | None = None
    glitch_frequency_half_width_hz: float = 5.0e-7
    glitch_frequency_derivative_half_width_hz_s: float = 5.0e-14

    def __post_init__(self) -> None:
        positive = (
            self.astrometric_half_width_arcsec,
            self.frequency_half_width_hz,
            self.frequency_derivative_half_width_hz_s,
            self.minimum_jitter_cycles,
            self.maximum_jitter_cycles,
            self.student_t_degrees_of_freedom,
            self.minimum_state_offset_cycles,
            self.maximum_state_offset_cycles,
            self.minimum_state_weight,
            self.maximum_state_weight,
            self.glitch_frequency_half_width_hz,
            self.glitch_frequency_derivative_half_width_hz_s,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Isolated timing-update scales must be positive and finite.")
        if self.minimum_jitter_cycles >= self.maximum_jitter_cycles:
            raise ValueError("Timing-update jitter bounds must be ordered.")
        if not 0.0 < self.minimum_state_offset_cycles < self.maximum_state_offset_cycles <= 0.5:
            raise ValueError("Profile-state offsets must lie inside (0, 0.5].")
        if not 0.0 < self.minimum_state_weight < self.maximum_state_weight < 1.0:
            raise ValueError("Profile-state weight bounds must lie inside (0, 1).")
        if int(self.integer_radius) < 0:
            raise ValueError("The timing-update integer radius cannot be negative.")
        if (self.glitch_epoch_lower_mjd is None) != (
            self.glitch_epoch_upper_mjd is None
        ):
            raise ValueError("Both glitch epoch bounds must be supplied together.")
        if self.glitch_epoch_lower_mjd is not None and (
            not np.isfinite(self.glitch_epoch_lower_mjd)
            or not np.isfinite(self.glitch_epoch_upper_mjd)
            or self.glitch_epoch_lower_mjd >= self.glitch_epoch_upper_mjd
        ):
            raise ValueError("Glitch epoch bounds must be finite and ordered.")


@dataclass(frozen=True)
class IsolatedTimingUpdateNestedConfig:
    live_point_count: int = 192
    evidence_tolerance: float = 0.5
    maximum_calls: int | None = 30000
    bound: str = "multi"
    sample: str = "rslice"
    random_seed: int = 20260730
    print_progress: bool = False

    def __post_init__(self) -> None:
        if int(self.live_point_count) <= 0:
            raise ValueError("Nested live-point count must be positive.")
        if not np.isfinite(self.evidence_tolerance) or self.evidence_tolerance < 0.0:
            raise ValueError("Nested evidence tolerance must be finite and nonnegative.")
        if self.maximum_calls is not None and int(self.maximum_calls) <= 0:
            raise ValueError("Nested maximum calls must be positive when supplied.")


@dataclass(frozen=True)
class IsolatedTimingUpdatePriorTransform:
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("Timing-update prior bounds must be matching vectors.")
        if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
            raise ValueError("Timing-update prior bounds must be finite and ordered.")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    def __call__(self, unit_values: Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=float)
        if unit.shape != self.lower.shape or not np.all(
            np.isfinite(unit) & (unit >= 0.0) & (unit <= 1.0)
        ):
            raise ValueError("Timing-update unit coordinates must lie in [0, 1].")
        return self.lower + unit * (self.upper - self.lower)

    def log_density(self, values: Sequence[float]) -> float:
        sample = np.asarray(values, dtype=float)
        if sample.shape != self.lower.shape or not np.all(
            np.isfinite(sample) & (sample >= self.lower) & (sample <= self.upper)
        ):
            return -math.inf
        return float(-np.sum(np.log(self.upper - self.lower)))


@dataclass(frozen=True)
class IsolatedTimingUpdateNestedResult:
    hypothesis: IsolatedTimingUpdateHypothesis
    parameter_names: tuple[str, ...]
    samples: np.ndarray
    log_likelihood: np.ndarray
    posterior_weights: np.ndarray
    log_evidence: float
    log_evidence_error: float
    best_values: np.ndarray
    best_log_likelihood: float
    call_count: int
    effective_sample_size: float
    reached_call_limit: bool


@dataclass(frozen=True)
class IsolatedTimingUpdateConditionalMode:
    """Laplace mode of the linear timing block at one nonlinear sample."""

    parameter_names: tuple[str, ...]
    physical_values: np.ndarray
    normalized_values: np.ndarray
    log_likelihood: float
    log_marginal_likelihood: float
    hessian: np.ndarray

    def __post_init__(self) -> None:
        physical = np.asarray(self.physical_values, dtype=float)
        normalized = np.asarray(self.normalized_values, dtype=float)
        hessian = np.asarray(self.hessian, dtype=float)
        size = len(self.parameter_names)
        if physical.shape != (size,) or normalized.shape != (size,):
            raise ValueError("Conditional timing values must match their names.")
        if hessian.shape != (size, size):
            raise ValueError("Conditional timing Hessian has the wrong shape.")
        if not np.all(
            np.isfinite(physical) & np.isfinite(normalized)
        ) or not np.all(np.isfinite(hessian)):
            raise ValueError("Conditional timing mode must be finite.")
        object.__setattr__(self, "physical_values", physical.copy())
        object.__setattr__(self, "normalized_values", normalized.copy())
        object.__setattr__(self, "hessian", hessian.copy())


@dataclass(frozen=True)
class IsolatedTimingUpdateMode:
    """One retained nonlinear outer-posterior mode."""

    values: np.ndarray
    log_posterior: float
    log_mass: float
    hessian: np.ndarray
    converged: bool
    boundary_active: bool

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        hessian = np.asarray(self.hessian, dtype=float)
        if values.ndim != 1 or hessian.shape != (values.size, values.size):
            raise ValueError("Isolated timing mode dimensions do not match.")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(hessian)):
            raise ValueError("Isolated timing modes must be finite.")
        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "hessian", hessian.copy())


@dataclass(frozen=True)
class IsolatedTimingUpdateModePortfolio:
    """Evidence-preserving mixture of nonlinear timing-update modes."""

    modes: tuple[IsolatedTimingUpdateMode, ...]
    log_evidence: float
    mode_weights: np.ndarray

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("An isolated timing mode portfolio cannot be empty.")
        weights = np.asarray(self.mode_weights, dtype=float)
        if weights.shape != (len(self.modes),) or not np.all(
            np.isfinite(weights) & (weights >= 0.0)
        ):
            raise ValueError("Timing mode weights must be finite and nonnegative.")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("Timing mode weights need positive total mass.")
        object.__setattr__(self, "mode_weights", weights / total)


@dataclass(frozen=True)
class IsolatedTimingUpdateGaussianPrior:
    """A previous timing posterior summary used for a sequential update."""

    sky_east_mean_arcsec: float = 0.0
    sky_east_sigma_arcsec: float = 1.0
    sky_north_mean_arcsec: float = 0.0
    sky_north_sigma_arcsec: float = 1.0
    frequency_mean_hz: float = 0.0
    frequency_sigma_hz: float = 1.0e-9
    frequency_derivative_mean_hz_s: float = 0.0
    frequency_derivative_sigma_hz_s: float = 1.0e-16
    global_phase_mean_cycles: float = 0.0
    global_phase_sigma_cycles: float = 0.1

    def __post_init__(self) -> None:
        means = (
            self.sky_east_mean_arcsec,
            self.sky_north_mean_arcsec,
            self.frequency_mean_hz,
            self.frequency_derivative_mean_hz_s,
            self.global_phase_mean_cycles,
        )
        sigmas = (
            self.sky_east_sigma_arcsec,
            self.sky_north_sigma_arcsec,
            self.frequency_sigma_hz,
            self.frequency_derivative_sigma_hz_s,
            self.global_phase_sigma_cycles,
        )
        if any(not np.isfinite(value) for value in means) or any(
            not np.isfinite(value) or value <= 0.0 for value in sigmas
        ):
            raise ValueError("Sequential timing prior means and scales must be finite.")


@dataclass(frozen=True)
class IsolatedTimingUpdateGaussianPriorTransform:
    """Unit-cube transform for a truncated Gaussian/uniform prior mixture."""

    lower: np.ndarray
    upper: np.ndarray
    gaussian_indices: tuple[int, ...]
    gaussian_means: np.ndarray
    gaussian_sigmas: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        means = np.asarray(self.gaussian_means, dtype=float)
        sigmas = np.asarray(self.gaussian_sigmas, dtype=float)
        if lower.ndim != 1 or upper.shape != lower.shape:
            raise ValueError("Gaussian timing prior bounds must be matching vectors.")
        if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
            raise ValueError("Gaussian timing prior bounds must be finite and ordered.")
        indices = tuple(int(index) for index in self.gaussian_indices)
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= lower.size for index in indices
        ):
            raise ValueError("Gaussian timing prior indices must be unique and valid.")
        if means.shape != (len(indices),) or sigmas.shape != (len(indices),):
            raise ValueError("Gaussian timing prior moments must match their indices.")
        if not np.all(np.isfinite(means)) or not np.all(
            np.isfinite(sigmas) & (sigmas > 0.0)
        ):
            raise ValueError("Gaussian timing prior moments must be finite and positive.")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())
        object.__setattr__(self, "gaussian_indices", indices)
        object.__setattr__(self, "gaussian_means", means.copy())
        object.__setattr__(self, "gaussian_sigmas", sigmas.copy())

    def __call__(self, unit_values: Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=float)
        if unit.shape != self.lower.shape or not np.all(
            np.isfinite(unit) & (unit >= 0.0) & (unit <= 1.0)
        ):
            raise ValueError("Timing prior unit coordinates must lie in [0, 1].")
        result = self.lower + unit * (self.upper - self.lower)
        for index, mean, sigma in zip(
            self.gaussian_indices,
            self.gaussian_means,
            self.gaussian_sigmas,
            strict=True,
        ):
            lo = ndtr((self.lower[index] - mean) / sigma)
            hi = ndtr((self.upper[index] - mean) / sigma)
            probability = lo + unit[index] * (hi - lo)
            result[index] = mean + sigma * ndtri(probability)
        return result

    def log_density(self, values: Sequence[float]) -> float:
        sample = np.asarray(values, dtype=float)
        if sample.shape != self.lower.shape or not np.all(
            np.isfinite(sample) & (sample >= self.lower) & (sample <= self.upper)
        ):
            return -math.inf
        gaussian_set = set(self.gaussian_indices)
        density = 0.0
        for index, mean, sigma in zip(
            self.gaussian_indices,
            self.gaussian_means,
            self.gaussian_sigmas,
            strict=True,
        ):
            lower_probability = ndtr((self.lower[index] - mean) / sigma)
            upper_probability = ndtr((self.upper[index] - mean) / sigma)
            density += float(
                -0.5 * ((sample[index] - mean) / sigma) ** 2
                - math.log(sigma * math.sqrt(2.0 * math.pi))
                - math.log(upper_probability - lower_probability)
            )
        for index in range(sample.size):
            if index not in gaussian_set:
                density -= math.log(self.upper[index] - self.lower[index])
        return float(density)


def isolated_update_parameter_names(
    hypothesis: IsolatedTimingUpdateHypothesis,
) -> tuple[str, ...]:
    names = [
        "SKY_EAST_ARCSEC",
        "SKY_NORTH_ARCSEC",
        "DELTA_F0_HZ",
        "DELTA_F1_HZ_S",
        "GLOBAL_PHASE_CYCLES",
        "LOG_JITTER_CYCLES",
    ]
    if hypothesis.profile_state_count == 2:
        names.extend(("STATE_OFFSET_CYCLES", "STATE_WEIGHT"))
    if hypothesis.glitch_frequency_order >= 1:
        names.extend(("GLITCH_EPOCH_MJD", "GLF0_HZ"))
    if hypothesis.glitch_frequency_order >= 2:
        names.append("GLF1_HZ_S")
    return tuple(names)


def build_isolated_update_prior(
    *,
    hypothesis: IsolatedTimingUpdateHypothesis,
    config: IsolatedTimingUpdateConfig,
    observation_epochs_mjd: Sequence[float],
) -> IsolatedTimingUpdatePriorTransform:
    astrometry = float(config.astrometric_half_width_arcsec)
    lower = [
        -astrometry,
        -astrometry,
        -float(config.frequency_half_width_hz),
        -float(config.frequency_derivative_half_width_hz_s),
        -0.5,
        math.log(float(config.minimum_jitter_cycles)),
    ]
    upper = [
        astrometry,
        astrometry,
        float(config.frequency_half_width_hz),
        float(config.frequency_derivative_half_width_hz_s),
        0.5,
        math.log(float(config.maximum_jitter_cycles)),
    ]
    if hypothesis.profile_state_count == 2:
        lower.extend(
            (
                float(config.minimum_state_offset_cycles),
                float(config.minimum_state_weight),
            )
        )
        upper.extend(
            (
                float(config.maximum_state_offset_cycles),
                float(config.maximum_state_weight),
            )
        )
    if hypothesis.glitch_frequency_order >= 1:
        epochs = np.asarray(observation_epochs_mjd, dtype=float)
        if epochs.ndim != 1 or epochs.size < 2 or not np.all(np.isfinite(epochs)):
            raise ValueError("Glitch support needs at least two finite TOA epochs.")
        epoch_lower = (
            float(config.glitch_epoch_lower_mjd)
            if config.glitch_epoch_lower_mjd is not None
            else float(np.min(epochs))
        )
        epoch_upper = (
            float(config.glitch_epoch_upper_mjd)
            if config.glitch_epoch_upper_mjd is not None
            else float(np.max(epochs))
        )
        if epoch_lower >= epoch_upper:
            raise ValueError("The effective glitch epoch support is empty.")
        lower.extend((epoch_lower, -float(config.glitch_frequency_half_width_hz)))
        upper.extend((epoch_upper, float(config.glitch_frequency_half_width_hz)))
    if hypothesis.glitch_frequency_order >= 2:
        lower.append(-float(config.glitch_frequency_derivative_half_width_hz_s))
        upper.append(float(config.glitch_frequency_derivative_half_width_hz_s))
    return IsolatedTimingUpdatePriorTransform(
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )


class IsolatedTimingUpdateLogLikelihood:
    """Exact PINT phase likelihood with integers and profile states integrated."""

    def __init__(
        self,
        *,
        evaluator: Any,
        hypothesis: IsolatedTimingUpdateHypothesis,
        config: IsolatedTimingUpdateConfig,
    ) -> None:
        names = tuple(str(name).upper() for name in evaluator.parameter_names)
        required = {"RAJ", "DECJ", "F0", "F1"}
        if not required.issubset(names):
            raise ValueError(
                "Isolated timing update requires free RAJ, DECJ, F0, and F1."
            )
        epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
        sigmas = np.asarray(evaluator.reported_phase_sigmas(), dtype=float)
        if epochs.ndim != 1 or sigmas.shape != epochs.shape or not np.all(
            np.isfinite(epochs) & np.isfinite(sigmas) & (sigmas > 0.0)
        ):
            raise ValueError("Timing-update epochs and phase errors must align.")
        sessions = tuple(evaluator.sessions)
        covered = tuple(index for session in sessions for index in session.toa_indices)
        if covered != tuple(range(epochs.size)):
            raise ValueError("Timing-update sessions must cover every TOA exactly once.")
        self.evaluator = evaluator
        self.hypothesis = hypothesis
        self.config = config
        self.parameter_names = isolated_update_parameter_names(hypothesis)
        self.model_parameter_names = names
        self.model_parameter_index = {name: index for index, name in enumerate(names)}
        self.base_parameter_values = np.asarray(
            evaluator.base_parameter_values, dtype=float
        ).copy()
        self.epochs_mjd = epochs
        self.reported_sigmas = sigmas
        self.sessions = sessions
        self.base_ra_hours = float(
            self.base_parameter_values[self.model_parameter_index["RAJ"]]
        )
        self.base_dec_degrees = float(
            self.base_parameter_values[self.model_parameter_index["DECJ"]]
        )
        self.cos_dec = math.cos(math.radians(self.base_dec_degrees))
        if abs(self.cos_dec) <= 1.0e-8:
            raise ValueError("Equatorial tangent coordinates are singular at the pole.")

    def physical_parameter_values(self, values: Sequence[float]) -> np.ndarray:
        sample = np.asarray(values, dtype=float)
        if sample.shape != (len(self.parameter_names),) or not np.all(
            np.isfinite(sample)
        ):
            raise ValueError("Timing-update values do not match the hypothesis.")
        by_name = dict(zip(self.parameter_names, sample, strict=True))
        result = self.base_parameter_values.copy()
        result[self.model_parameter_index["RAJ"]] = self.base_ra_hours + float(
            by_name["SKY_EAST_ARCSEC"]
        ) / (15.0 * 3600.0 * self.cos_dec)
        result[self.model_parameter_index["DECJ"]] = self.base_dec_degrees + float(
            by_name["SKY_NORTH_ARCSEC"]
        ) / 3600.0
        result[self.model_parameter_index["F0"]] += float(by_name["DELTA_F0_HZ"])
        result[self.model_parameter_index["F1"]] += float(
            by_name["DELTA_F1_HZ_S"]
        )
        return result

    def phase_values(self, values: Sequence[float]) -> np.ndarray:
        sample = np.asarray(values, dtype=float)
        by_name = dict(zip(self.parameter_names, sample, strict=True))
        phase = np.asarray(
            self.evaluator.raw_phase_residuals(
                self.physical_parameter_values(sample)
            ),
            dtype=float,
        )
        if self.hypothesis.glitch_frequency_order >= 1:
            epoch = float(by_name["GLITCH_EPOCH_MJD"])
            dt_seconds = (self.epochs_mjd - epoch) * 86400.0
            after = dt_seconds > 0.0
            phase = phase + after * float(by_name["GLF0_HZ"]) * dt_seconds
            if self.hypothesis.glitch_frequency_order >= 2:
                phase = phase + after * 0.5 * float(
                    by_name["GLF1_HZ_S"]
                ) * dt_seconds**2
        return phase

    def _student_t_log_density(
        self,
        residuals: np.ndarray,
        scales: np.ndarray,
    ) -> np.ndarray:
        degrees = float(self.config.student_t_degrees_of_freedom)
        normalization = (
            gammaln(0.5 * (degrees + 1.0))
            - gammaln(0.5 * degrees)
            - 0.5 * math.log(degrees * math.pi)
            - np.log(scales)
        )
        return normalization - 0.5 * (degrees + 1.0) * np.log1p(
            np.square(residuals / scales) / degrees
        )

    def _session_component_log_likelihood(
        self,
        *,
        phase: np.ndarray,
        scales: np.ndarray,
        indices: np.ndarray,
        global_phase: float,
        state_offset: float,
    ) -> float:
        shifted = phase[indices] - global_phase - state_offset
        weights = 1.0 / np.square(scales[indices])
        integer_center = int(np.rint(np.sum(weights * shifted) / np.sum(weights)))
        radius = int(self.config.integer_radius)
        integer_values = range(integer_center - radius, integer_center + radius + 1)
        terms = [
            float(
                np.sum(
                    self._student_t_log_density(
                        shifted - integer,
                        scales[indices],
                    )
                )
            )
            for integer in integer_values
        ]
        return float(logsumexp(terms) - math.log(len(terms)))

    def __call__(self, values: Sequence[float]) -> float:
        sample = np.asarray(values, dtype=float)
        if sample.shape != (len(self.parameter_names),) or not np.all(
            np.isfinite(sample)
        ):
            return -math.inf
        by_name = dict(zip(self.parameter_names, sample, strict=True))
        jitter = math.exp(float(by_name["LOG_JITTER_CYCLES"]))
        if not (
            self.config.minimum_jitter_cycles
            <= jitter
            <= self.config.maximum_jitter_cycles
        ):
            return -math.inf
        try:
            phase = self.phase_values(sample)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return -math.inf
        if phase.shape != self.epochs_mjd.shape or not np.all(np.isfinite(phase)):
            return -math.inf
        scales = np.sqrt(np.square(self.reported_sigmas) + jitter * jitter)
        global_phase = float(by_name["GLOBAL_PHASE_CYCLES"])
        if self.hypothesis.profile_state_count == 1:
            state_offsets = (0.0,)
            state_log_weights = (0.0,)
        else:
            offset = float(by_name["STATE_OFFSET_CYCLES"])
            weight = float(by_name["STATE_WEIGHT"])
            if not (
                self.config.minimum_state_offset_cycles
                <= offset
                <= self.config.maximum_state_offset_cycles
                and self.config.minimum_state_weight
                <= weight
                <= self.config.maximum_state_weight
            ):
                return -math.inf
            state_offsets = (0.0, offset)
            state_log_weights = (math.log1p(-weight), math.log(weight))
        total = 0.0
        for session in self.sessions:
            indices = np.asarray(session.toa_indices, dtype=int)
            components = [
                state_log_weight
                + self._session_component_log_likelihood(
                    phase=phase,
                    scales=scales,
                    indices=indices,
                    global_phase=global_phase,
                    state_offset=state_offset,
                )
                for state_offset, state_log_weight in zip(
                    state_offsets, state_log_weights, strict=True
                )
            ]
            total += float(logsumexp(components))
        return float(total) if np.isfinite(total) else -math.inf


def collapsed_isolated_update_parameter_names(
    hypothesis: IsolatedTimingUpdateHypothesis,
) -> tuple[str, ...]:
    """Parameters left after integrating the semilinear spin block."""

    names = [
        "SKY_EAST_ARCSEC",
        "SKY_NORTH_ARCSEC",
        "LOG_JITTER_CYCLES",
    ]
    if hypothesis.profile_state_count == 2:
        names.extend(("STATE_OFFSET_CYCLES", "STATE_WEIGHT"))
    if hypothesis.glitch_frequency_order >= 1:
        names.append("GLITCH_EPOCH_MJD")
    return tuple(names)


def build_collapsed_isolated_update_prior(
    *,
    hypothesis: IsolatedTimingUpdateHypothesis,
    config: IsolatedTimingUpdateConfig,
    observation_epochs_mjd: Sequence[float],
) -> IsolatedTimingUpdatePriorTransform:
    """Build the prior for the nonlinear part of an anchored update."""

    astrometry = float(config.astrometric_half_width_arcsec)
    lower = [
        -astrometry,
        -astrometry,
        math.log(float(config.minimum_jitter_cycles)),
    ]
    upper = [
        astrometry,
        astrometry,
        math.log(float(config.maximum_jitter_cycles)),
    ]
    if hypothesis.profile_state_count == 2:
        lower.extend(
            (
                float(config.minimum_state_offset_cycles),
                float(config.minimum_state_weight),
            )
        )
        upper.extend(
            (
                float(config.maximum_state_offset_cycles),
                float(config.maximum_state_weight),
            )
        )
    if hypothesis.glitch_frequency_order >= 1:
        epochs = np.asarray(observation_epochs_mjd, dtype=float)
        if epochs.ndim != 1 or epochs.size < 2 or not np.all(np.isfinite(epochs)):
            raise ValueError("Glitch support needs at least two finite TOA epochs.")
        epoch_lower = (
            float(config.glitch_epoch_lower_mjd)
            if config.glitch_epoch_lower_mjd is not None
            else float(np.min(epochs))
        )
        epoch_upper = (
            float(config.glitch_epoch_upper_mjd)
            if config.glitch_epoch_upper_mjd is not None
            else float(np.max(epochs))
        )
        if epoch_lower >= epoch_upper:
            raise ValueError("The effective glitch epoch support is empty.")
        lower.append(epoch_lower)
        upper.append(epoch_upper)
    return IsolatedTimingUpdatePriorTransform(
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
    )


def build_gaussian_collapsed_isolated_update_prior(
    *,
    hypothesis: IsolatedTimingUpdateHypothesis,
    config: IsolatedTimingUpdateConfig,
    observation_epochs_mjd: Sequence[float],
    previous_posterior: IsolatedTimingUpdateGaussianPrior,
    truncation_sigma: float = 6.0,
) -> IsolatedTimingUpdateGaussianPriorTransform:
    """Build a sequential-update prior with a finite broad support."""

    if not np.isfinite(truncation_sigma) or truncation_sigma <= 0.0:
        raise ValueError("Sequential timing prior truncation must be positive.")
    lower = [
        previous_posterior.sky_east_mean_arcsec
        - truncation_sigma * previous_posterior.sky_east_sigma_arcsec,
        previous_posterior.sky_north_mean_arcsec
        - truncation_sigma * previous_posterior.sky_north_sigma_arcsec,
        math.log(float(config.minimum_jitter_cycles)),
    ]
    upper = [
        previous_posterior.sky_east_mean_arcsec
        + truncation_sigma * previous_posterior.sky_east_sigma_arcsec,
        previous_posterior.sky_north_mean_arcsec
        + truncation_sigma * previous_posterior.sky_north_sigma_arcsec,
        math.log(float(config.maximum_jitter_cycles)),
    ]
    gaussian_indices = [0, 1]
    gaussian_means = [
        previous_posterior.sky_east_mean_arcsec,
        previous_posterior.sky_north_mean_arcsec,
    ]
    gaussian_sigmas = [
        previous_posterior.sky_east_sigma_arcsec,
        previous_posterior.sky_north_sigma_arcsec,
    ]
    if hypothesis.profile_state_count == 2:
        lower.extend(
            (
                float(config.minimum_state_offset_cycles),
                float(config.minimum_state_weight),
            )
        )
        upper.extend(
            (
                float(config.maximum_state_offset_cycles),
                float(config.maximum_state_weight),
            )
        )
    if hypothesis.glitch_frequency_order >= 1:
        epochs = np.asarray(observation_epochs_mjd, dtype=float)
        if epochs.ndim != 1 or epochs.size < 2 or not np.all(np.isfinite(epochs)):
            raise ValueError("Glitch support needs at least two finite TOA epochs.")
        epoch_lower = (
            float(config.glitch_epoch_lower_mjd)
            if config.glitch_epoch_lower_mjd is not None
            else float(np.min(epochs))
        )
        epoch_upper = (
            float(config.glitch_epoch_upper_mjd)
            if config.glitch_epoch_upper_mjd is not None
            else float(np.max(epochs))
        )
        if epoch_lower >= epoch_upper:
            raise ValueError("The effective glitch epoch support is empty.")
        lower.append(epoch_lower)
        upper.append(epoch_upper)
    return IsolatedTimingUpdateGaussianPriorTransform(
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        gaussian_indices=tuple(gaussian_indices),
        gaussian_means=np.asarray(gaussian_means, dtype=float),
        gaussian_sigmas=np.asarray(gaussian_sigmas, dtype=float),
    )


class CollapsedAnchoredIsolatedTimingUpdateLogLikelihood:
    """Timing-update posterior with the certified baseline path preserved.

    The historical sessions retain their known integer pulse path. Integer
    offsets for extension sessions and shared profile-state labels are summed
    out. Global phase, F0, F1, and optional glitch amplitudes form a semilinear
    block that is integrated with a local Laplace approximation at every
    nonlinear sample.
    """

    _F0_SCALE_HZ = 1.0e-9
    _F1_SCALE_HZ_S = 1.0e-16

    def __init__(
        self,
        *,
        evaluator: Any,
        hypothesis: IsolatedTimingUpdateHypothesis,
        config: IsolatedTimingUpdateConfig,
        anchored_session_indices: Sequence[int],
        previous_posterior: IsolatedTimingUpdateGaussianPrior | None = None,
        likelihood_session_indices: Sequence[int] | None = None,
    ) -> None:
        names = tuple(str(name).upper() for name in evaluator.parameter_names)
        required = {"RAJ", "DECJ", "F0", "F1"}
        if not required.issubset(names):
            raise ValueError(
                "Anchored isolated timing update requires RAJ, DECJ, F0, and F1."
            )
        epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
        sigmas = np.asarray(evaluator.reported_phase_sigmas(), dtype=float)
        if epochs.ndim != 1 or sigmas.shape != epochs.shape or not np.all(
            np.isfinite(epochs) & np.isfinite(sigmas) & (sigmas > 0.0)
        ):
            raise ValueError("Timing-update epochs and phase errors must align.")
        sessions = tuple(evaluator.sessions)
        covered = tuple(index for session in sessions for index in session.toa_indices)
        if covered != tuple(range(epochs.size)):
            raise ValueError("Timing-update sessions must cover every TOA exactly once.")
        anchored = tuple(sorted({int(value) for value in anchored_session_indices}))
        if not anchored or anchored[0] < 0 or anchored[-1] >= len(sessions):
            raise ValueError("Anchored sessions must identify a nonempty valid subset.")
        extension = tuple(index for index in range(len(sessions)) if index not in anchored)
        if not extension:
            raise ValueError("An anchored timing update needs at least one new session.")
        likelihood_sessions = (
            tuple(range(len(sessions)))
            if likelihood_session_indices is None
            else tuple(sorted({int(value) for value in likelihood_session_indices}))
        )
        if not likelihood_sessions or likelihood_sessions[0] < 0 or likelihood_sessions[-1] >= len(sessions):
            raise ValueError("Likelihood sessions must identify a valid nonempty subset.")
        if previous_posterior is not None and any(
            index in anchored for index in likelihood_sessions
        ):
            raise ValueError(
                "A sequential posterior prior cannot reuse its baseline sessions."
            )
        anchored_toas = np.asarray(
            [index for session_index in anchored for index in sessions[session_index].toa_indices],
            dtype=int,
        )
        if hypothesis.glitch_frequency_order >= 1:
            if config.glitch_epoch_lower_mjd is None:
                raise ValueError("Anchored glitch searches need declared epoch support.")
            anchor_end = float(np.max(epochs[anchored_toas]))
            if float(config.glitch_epoch_lower_mjd) < anchor_end:
                raise ValueError(
                    "An update glitch cannot precede the end of the certified baseline."
                )

        self.evaluator = evaluator
        self.hypothesis = hypothesis
        self.config = config
        self.parameter_names = collapsed_isolated_update_parameter_names(hypothesis)
        self.model_parameter_names = names
        self.model_parameter_index = {name: index for index, name in enumerate(names)}
        self.base_parameter_values = np.asarray(
            evaluator.base_parameter_values, dtype=float
        ).copy()
        self.epochs_mjd = epochs
        self.reported_sigmas = sigmas
        self.sessions = sessions
        self.anchored_session_indices = anchored
        self.extension_session_indices = extension
        self.likelihood_session_indices = likelihood_sessions
        self.initialization_session_indices = (
            tuple(index for index in anchored if index in likelihood_sessions)
            or likelihood_sessions
        )
        self.previous_posterior = previous_posterior
        self.anchored_toa_indices = anchored_toas
        self.base_ra_hours = float(
            self.base_parameter_values[self.model_parameter_index["RAJ"]]
        )
        self.base_dec_degrees = float(
            self.base_parameter_values[self.model_parameter_index["DECJ"]]
        )
        self.cos_dec = math.cos(math.radians(self.base_dec_degrees))
        if abs(self.cos_dec) <= 1.0e-8:
            raise ValueError("Equatorial tangent coordinates are singular at the pole.")

        self.conditional_parameter_names = [
            "GLOBAL_PHASE_CYCLES",
            "DELTA_F0_HZ",
            "DELTA_F1_HZ_S",
        ]
        self.conditional_scales = [
            1.0,
            self._F0_SCALE_HZ,
            self._F1_SCALE_HZ_S,
        ]
        self.conditional_lower = [
            -0.5,
            -float(config.frequency_half_width_hz) / self._F0_SCALE_HZ,
            -float(config.frequency_derivative_half_width_hz_s)
            / self._F1_SCALE_HZ_S,
        ]
        self.conditional_upper = [
            0.5,
            float(config.frequency_half_width_hz) / self._F0_SCALE_HZ,
            float(config.frequency_derivative_half_width_hz_s)
            / self._F1_SCALE_HZ_S,
        ]
        if hypothesis.glitch_frequency_order >= 1:
            self.conditional_parameter_names.append("GLF0_HZ")
            self.conditional_scales.append(self._F0_SCALE_HZ)
            self.conditional_lower.append(
                -float(config.glitch_frequency_half_width_hz) / self._F0_SCALE_HZ
            )
            self.conditional_upper.append(
                float(config.glitch_frequency_half_width_hz) / self._F0_SCALE_HZ
            )
        if hypothesis.glitch_frequency_order >= 2:
            self.conditional_parameter_names.append("GLF1_HZ_S")
            self.conditional_scales.append(self._F1_SCALE_HZ_S)
            self.conditional_lower.append(
                -float(config.glitch_frequency_derivative_half_width_hz_s)
                / self._F1_SCALE_HZ_S
            )
            self.conditional_upper.append(
                float(config.glitch_frequency_derivative_half_width_hz_s)
                / self._F1_SCALE_HZ_S
            )
        self.conditional_parameter_names = tuple(self.conditional_parameter_names)
        self.conditional_scales = np.asarray(self.conditional_scales, dtype=float)
        self.conditional_lower = np.asarray(self.conditional_lower, dtype=float)
        self.conditional_upper = np.asarray(self.conditional_upper, dtype=float)
        if previous_posterior is None:
            self.conditional_gaussian_indices = tuple()
            self.conditional_gaussian_means = np.empty(0, dtype=float)
            self.conditional_gaussian_sigmas = np.empty(0, dtype=float)
        else:
            self.conditional_gaussian_indices = (0, 1, 2)
            self.conditional_gaussian_means = np.asarray(
                (
                    previous_posterior.global_phase_mean_cycles,
                    previous_posterior.frequency_mean_hz / self._F0_SCALE_HZ,
                    previous_posterior.frequency_derivative_mean_hz_s
                    / self._F1_SCALE_HZ_S,
                ),
                dtype=float,
            )
            self.conditional_gaussian_sigmas = np.asarray(
                (
                    previous_posterior.global_phase_sigma_cycles,
                    previous_posterior.frequency_sigma_hz / self._F0_SCALE_HZ,
                    previous_posterior.frequency_derivative_sigma_hz_s
                    / self._F1_SCALE_HZ_S,
                ),
                dtype=float,
            )
        self._spin_basis = self._build_spin_basis()

    def _build_spin_basis(self) -> np.ndarray:
        base = np.asarray(
            self.evaluator.raw_phase_residuals(self.base_parameter_values), dtype=float
        )
        columns = []
        for name, scale in (
            ("F0", self._F0_SCALE_HZ),
            ("F1", self._F1_SCALE_HZ_S),
        ):
            values = self.base_parameter_values.copy()
            values[self.model_parameter_index[name]] += scale
            shifted = np.asarray(
                self.evaluator.raw_phase_residuals(values), dtype=float
            )
            columns.append(shifted - base)
        self.evaluator.raw_phase_residuals(self.base_parameter_values)
        basis = np.column_stack(columns)
        if basis.shape != (self.epochs_mjd.size, 2) or not np.all(np.isfinite(basis)):
            raise ValueError("PINT did not produce a finite semilinear spin basis.")
        return basis

    def physical_parameter_values(self, values: Sequence[float]) -> np.ndarray:
        sample = np.asarray(values, dtype=float)
        if sample.shape != (len(self.parameter_names),) or not np.all(np.isfinite(sample)):
            raise ValueError("Collapsed timing-update values do not match the hypothesis.")
        by_name = dict(zip(self.parameter_names, sample, strict=True))
        result = self.base_parameter_values.copy()
        result[self.model_parameter_index["RAJ"]] = self.base_ra_hours + float(
            by_name["SKY_EAST_ARCSEC"]
        ) / (15.0 * 3600.0 * self.cos_dec)
        result[self.model_parameter_index["DECJ"]] = self.base_dec_degrees + float(
            by_name["SKY_NORTH_ARCSEC"]
        ) / 3600.0
        return result

    def _base_phase(self, values: np.ndarray) -> np.ndarray:
        phase = np.asarray(
            self.evaluator.raw_phase_residuals(self.physical_parameter_values(values)),
            dtype=float,
        )
        if phase.shape != self.epochs_mjd.shape or not np.all(np.isfinite(phase)):
            raise ValueError("PINT returned invalid update phases.")
        return phase

    def _conditional_design(self, values: np.ndarray) -> np.ndarray:
        columns = [
            -np.ones(self.epochs_mjd.size, dtype=float),
            self._spin_basis[:, 0],
            self._spin_basis[:, 1],
        ]
        if self.hypothesis.glitch_frequency_order >= 1:
            by_name = dict(zip(self.parameter_names, values, strict=True))
            epoch = float(by_name["GLITCH_EPOCH_MJD"])
            dt_seconds = np.maximum((self.epochs_mjd - epoch) * 86400.0, 0.0)
            columns.append(dt_seconds * self._F0_SCALE_HZ)
            if self.hypothesis.glitch_frequency_order >= 2:
                columns.append(0.5 * dt_seconds**2 * self._F1_SCALE_HZ_S)
        return np.column_stack(columns)

    def _student_t_log_density(
        self,
        residuals: np.ndarray,
        scales: np.ndarray,
    ) -> np.ndarray:
        degrees = float(self.config.student_t_degrees_of_freedom)
        normalization = (
            gammaln(0.5 * (degrees + 1.0))
            - gammaln(0.5 * degrees)
            - 0.5 * math.log(degrees * math.pi)
            - np.log(scales)
        )
        return normalization - 0.5 * (degrees + 1.0) * np.log1p(
            np.square(residuals / scales) / degrees
        )

    def _session_log_likelihood(
        self,
        *,
        residual_phase: np.ndarray,
        scales: np.ndarray,
        session_index: int,
        state_offset: float,
    ) -> float:
        indices = np.asarray(self.sessions[session_index].toa_indices, dtype=int)
        shifted = residual_phase[indices] - state_offset
        if session_index in self.anchored_session_indices:
            integer_values = (0,)
        else:
            weights = 1.0 / np.square(scales[indices])
            center = int(np.rint(np.sum(weights * shifted) / np.sum(weights)))
            radius = int(self.config.integer_radius)
            integer_values = range(center - radius, center + radius + 1)
        terms = [
            float(np.sum(self._student_t_log_density(shifted - integer, scales[indices])))
            for integer in integer_values
        ]
        return float(logsumexp(terms))

    def _conditional_log_likelihood(
        self,
        normalized_values: np.ndarray,
        *,
        phase: np.ndarray,
        design: np.ndarray,
        scales: np.ndarray,
        state_offset: float,
        state_weight: float,
    ) -> float:
        value, _ = self._conditional_log_likelihood_and_gradient(
            normalized_values,
            phase=phase,
            design=design,
            scales=scales,
            state_offset=state_offset,
            state_weight=state_weight,
        )
        return value

    def _conditional_log_likelihood_and_gradient(
        self,
        normalized_values: np.ndarray,
        *,
        phase: np.ndarray,
        design: np.ndarray,
        scales: np.ndarray,
        state_offset: float,
        state_weight: float,
    ) -> tuple[float, np.ndarray]:
        value, gradient, _ = self._conditional_log_likelihood_gradient_hessian(
            normalized_values,
            phase=phase,
            design=design,
            scales=scales,
            state_offset=state_offset,
            state_weight=state_weight,
        )
        return value, gradient

    def _conditional_log_likelihood_gradient_hessian(
        self,
        normalized_values: np.ndarray,
        *,
        phase: np.ndarray,
        design: np.ndarray,
        scales: np.ndarray,
        state_offset: float,
        state_weight: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        values = np.asarray(normalized_values, dtype=float)
        if values.shape != self.conditional_lower.shape or not np.all(
            np.isfinite(values)
            & (values >= self.conditional_lower)
            & (values <= self.conditional_upper)
        ):
            return -math.inf, np.zeros_like(values), np.zeros((values.size, values.size))
        residual_phase = phase + design @ values
        if self.hypothesis.profile_state_count == 1:
            offsets = (0.0,)
            log_weights = (0.0,)
        else:
            offsets = (0.0, state_offset)
            log_weights = (math.log1p(-state_weight), math.log(state_weight))
        total = 0.0
        total_gradient = np.zeros_like(values)
        total_hessian = np.zeros((values.size, values.size), dtype=float)
        degrees = float(self.config.student_t_degrees_of_freedom)
        for session_index in self.likelihood_session_indices:
            indices = np.asarray(self.sessions[session_index].toa_indices, dtype=int)
            state_terms = []
            state_gradients = []
            state_hessians = []
            for offset, log_weight in zip(offsets, log_weights, strict=True):
                shifted = residual_phase[indices] - offset
                if session_index in self.anchored_session_indices:
                    integer_values = (0,)
                else:
                    weights = 1.0 / np.square(scales[indices])
                    center = int(np.rint(np.sum(weights * shifted) / np.sum(weights)))
                    radius = int(self.config.integer_radius)
                    integer_values = range(center - radius, center + radius + 1)
                integer_terms = []
                integer_gradients = []
                integer_hessians = []
                for integer in integer_values:
                    residuals = shifted - integer
                    integer_terms.append(
                        float(np.sum(self._student_t_log_density(residuals, scales[indices])))
                    )
                    phase_derivative = -(
                        (degrees + 1.0)
                        * residuals
                        / (degrees * np.square(scales[indices]) + np.square(residuals))
                    )
                    component_gradient = design[indices].T @ phase_derivative
                    phase_curvature = -(
                        (degrees + 1.0)
                        * (
                            degrees * np.square(scales[indices])
                            - np.square(residuals)
                        )
                        / np.square(
                            degrees * np.square(scales[indices])
                            + np.square(residuals)
                        )
                    )
                    integer_gradients.append(component_gradient)
                    integer_hessians.append(
                        design[indices].T
                        @ (phase_curvature[:, None] * design[indices])
                    )
                integer_terms_array = np.asarray(integer_terms, dtype=float)
                integer_log_sum = float(logsumexp(integer_terms_array))
                integer_probabilities = np.exp(integer_terms_array - integer_log_sum)
                state_terms.append(float(log_weight) + integer_log_sum)
                integer_gradients_array = np.asarray(integer_gradients, dtype=float)
                integer_hessians_array = np.asarray(integer_hessians, dtype=float)
                mixed_gradient = np.sum(
                    integer_probabilities[:, None] * integer_gradients_array,
                    axis=0,
                )
                mixed_hessian = np.sum(
                    integer_probabilities[:, None, None]
                    * (
                        integer_hessians_array
                        + integer_gradients_array[:, :, None]
                        * integer_gradients_array[:, None, :]
                    ),
                    axis=0,
                ) - np.outer(mixed_gradient, mixed_gradient)
                state_gradients.append(mixed_gradient)
                state_hessians.append(mixed_hessian)
            state_terms_array = np.asarray(state_terms, dtype=float)
            session_log_sum = float(logsumexp(state_terms_array))
            state_probabilities = np.exp(state_terms_array - session_log_sum)
            state_gradients_array = np.asarray(state_gradients, dtype=float)
            state_hessians_array = np.asarray(state_hessians, dtype=float)
            session_gradient = np.sum(
                state_probabilities[:, None] * state_gradients_array,
                axis=0,
            )
            session_hessian = np.sum(
                state_probabilities[:, None, None]
                * (
                    state_hessians_array
                    + state_gradients_array[:, :, None]
                    * state_gradients_array[:, None, :]
                ),
                axis=0,
            ) - np.outer(session_gradient, session_gradient)
            total += session_log_sum
            total_gradient += session_gradient
            total_hessian += session_hessian
        if self.conditional_gaussian_indices:
            indices = np.asarray(self.conditional_gaussian_indices, dtype=int)
            standardized = (
                values[indices] - self.conditional_gaussian_means
            ) / self.conditional_gaussian_sigmas
            total += float(
                -0.5 * np.sum(np.square(standardized))
                - np.sum(
                    np.log(
                        self.conditional_gaussian_sigmas
                        * math.sqrt(2.0 * math.pi)
                    )
                )
            )
            total_gradient[indices] -= (
                values[indices] - self.conditional_gaussian_means
            ) / np.square(self.conditional_gaussian_sigmas)
            total_hessian[indices, indices] -= 1.0 / np.square(
                self.conditional_gaussian_sigmas
            )
        if (
            not np.isfinite(total)
            or not np.all(np.isfinite(total_gradient))
            or not np.all(np.isfinite(total_hessian))
        ):
            return -math.inf, np.zeros_like(values), np.zeros_like(total_hessian)
        return float(total), total_gradient, 0.5 * (total_hessian + total_hessian.T)

    def _conditional_starts(
        self,
        *,
        phase: np.ndarray,
        design: np.ndarray,
        scales: np.ndarray,
        state_offset: float,
        state_weight: float,
    ) -> tuple[np.ndarray, ...]:
        anchor_sessions = self.initialization_session_indices
        toa_indices = np.asarray(
            [
                index
                for session_index in anchor_sessions
                for index in self.sessions[session_index].toa_indices
            ],
            dtype=int,
        )
        phase_anchor = phase[toa_indices]
        design_anchor = design[toa_indices, :3]
        sqrt_weights = 1.0 / scales[toa_indices]
        weighted_design = design_anchor * sqrt_weights[:, None]
        projector = np.linalg.pinv(weighted_design)

        if self.hypothesis.profile_state_count == 1:
            assignments = (tuple(0 for _ in anchor_sessions),)
        elif len(anchor_sessions) <= 12:
            assignments = product((0, 1), repeat=len(anchor_sessions))
        else:
            assignments = [tuple(0 for _ in anchor_sessions)]
            assignments.extend(
                tuple(int(index == selected) for index in range(len(anchor_sessions)))
                for selected in range(len(anchor_sessions))
            )

        candidates = []
        for assignment in assignments:
            states = np.zeros(toa_indices.size, dtype=float)
            cursor = 0
            for label, session_index in zip(assignment, anchor_sessions, strict=True):
                count = len(self.sessions[session_index].toa_indices)
                states[cursor : cursor + count] = float(label) * state_offset
                cursor += count
            target = (states - phase_anchor) * sqrt_weights
            start = np.zeros(len(self.conditional_parameter_names), dtype=float)
            start[:3] = projector @ target
            start = np.clip(
                start,
                self.conditional_lower + 1.0e-10,
                self.conditional_upper - 1.0e-10,
            )
            anchored_residuals = (
                phase_anchor + design_anchor @ start[:3] - states
            ) / scales[toa_indices]
            score = -float(np.sum(np.square(anchored_residuals)))
            candidates.append((score, start))
        zero = np.zeros(len(self.conditional_parameter_names), dtype=float)
        zero[0] = float(np.clip(np.median(phase[toa_indices]), -0.49, 0.49))
        zero_residuals = (
            phase_anchor + design_anchor @ zero[:3]
        ) / scales[toa_indices]
        candidates.append((-float(np.sum(np.square(zero_residuals))), zero))
        if self.previous_posterior is not None and not any(
            index in self.anchored_session_indices
            for index in self.likelihood_session_indices
        ):
            prior_start = np.zeros(len(self.conditional_parameter_names), dtype=float)
            prior_start[:3] = self.conditional_gaussian_means
            candidates.append((float("inf"), prior_start))
        candidates.sort(key=lambda item: item[0], reverse=True)
        starts = []
        seen = set()
        for _, candidate in candidates:
            key = tuple(np.round(candidate, decimals=7))
            if key in seen:
                continue
            seen.add(key)
            starts.append(candidate)
            if len(starts) >= 1:
                break
        return tuple(starts)

    def _numerical_hessian(
        self,
        function: Any,
        values: np.ndarray,
    ) -> np.ndarray:
        center = np.asarray(values, dtype=float)
        size = center.size
        steps = np.full(size, 1.0e-3, dtype=float)
        steps[0] = 1.0e-5
        widths = self.conditional_upper - self.conditional_lower
        distances = np.minimum(
            center - self.conditional_lower,
            self.conditional_upper - center,
        )
        steps = np.minimum(steps, 0.25 * distances)
        steps = np.minimum(steps, 1.0e-3 * widths)
        if np.any(steps <= 1.0e-10):
            raise ValueError("Conditional mode lies on a prior boundary.")
        hessian = np.zeros((size, size), dtype=float)
        f0 = float(function(center))
        for row in range(size):
            delta = np.zeros(size, dtype=float)
            delta[row] = steps[row]
            hessian[row, row] = (
                float(function(center + delta))
                - 2.0 * f0
                + float(function(center - delta))
            ) / steps[row] ** 2
            for column in range(row):
                left = np.zeros(size, dtype=float)
                right = np.zeros(size, dtype=float)
                left[row] = steps[row]
                right[column] = steps[column]
                value = (
                    float(function(center + left + right))
                    - float(function(center + left - right))
                    - float(function(center - left + right))
                    + float(function(center - left - right))
                ) / (4.0 * steps[row] * steps[column])
                hessian[row, column] = value
                hessian[column, row] = value
        return 0.5 * (hessian + hessian.T)

    def conditional_mode(
        self,
        values: Sequence[float],
    ) -> IsolatedTimingUpdateConditionalMode:
        sample = np.asarray(values, dtype=float)
        if sample.shape != (len(self.parameter_names),) or not np.all(np.isfinite(sample)):
            raise ValueError("Collapsed timing-update values do not match the hypothesis.")
        by_name = dict(zip(self.parameter_names, sample, strict=True))
        jitter = math.exp(float(by_name["LOG_JITTER_CYCLES"]))
        if not self.config.minimum_jitter_cycles <= jitter <= self.config.maximum_jitter_cycles:
            raise ValueError("Collapsed timing-update jitter is outside its support.")
        if self.hypothesis.profile_state_count == 1:
            state_offset = 0.0
            state_weight = 0.0
        else:
            state_offset = float(by_name["STATE_OFFSET_CYCLES"])
            state_weight = float(by_name["STATE_WEIGHT"])
            if not (
                self.config.minimum_state_offset_cycles
                <= state_offset
                <= self.config.maximum_state_offset_cycles
                and self.config.minimum_state_weight
                <= state_weight
                <= self.config.maximum_state_weight
            ):
                raise ValueError("Profile-state values are outside their support.")
        phase = self._base_phase(sample)
        design = self._conditional_design(sample)
        scales = np.sqrt(np.square(self.reported_sigmas) + jitter * jitter)

        cached_values = None
        cached_result = None

        def objective_terms(
            candidate: np.ndarray,
        ) -> tuple[float, np.ndarray, np.ndarray]:
            nonlocal cached_values, cached_result
            current = np.asarray(candidate, dtype=float)
            if cached_values is None or not np.array_equal(current, cached_values):
                value, gradient, hessian = (
                    self._conditional_log_likelihood_gradient_hessian(
                        current,
                        phase=phase,
                        design=design,
                        scales=scales,
                        state_offset=state_offset,
                        state_weight=state_weight,
                    )
                )
                if not np.isfinite(value):
                    value = -1.0e300
                    gradient = np.zeros_like(current)
                    hessian = -np.eye(current.size, dtype=float)
                cached_values = current.copy()
                cached_result = (-value, -gradient, -hessian)
            return cached_result

        def objective(candidate: np.ndarray) -> float:
            return float(objective_terms(candidate)[0])

        def objective_gradient(candidate: np.ndarray) -> np.ndarray:
            return np.asarray(objective_terms(candidate)[1], dtype=float)

        def objective_hessian(candidate: np.ndarray) -> np.ndarray:
            return np.asarray(objective_terms(candidate)[2], dtype=float)

        best_result = None
        for start in self._conditional_starts(
            phase=phase,
            design=design,
            scales=scales,
            state_offset=state_offset,
            state_weight=state_weight,
        ):
            result = minimize(
                objective,
                start,
                jac=objective_gradient,
                hess=objective_hessian,
                method="trust-exact",
                options={
                    "maxiter": 80,
                    "gtol": 1.0e-7,
                    "initial_trust_radius": 0.25,
                    "max_trust_radius": 20.0,
                },
            )
            if np.isfinite(result.fun) and (
                best_result is None or float(result.fun) < float(best_result.fun)
            ):
                best_result = result
        if best_result is None:
            raise ValueError("Conditional timing block has no finite mode.")
        mode = np.asarray(best_result.x, dtype=float)
        log_likelihood, _, log_likelihood_hessian = (
            self._conditional_log_likelihood_gradient_hessian(
                mode,
                phase=phase,
                design=design,
                scales=scales,
                state_offset=state_offset,
                state_weight=state_weight,
            )
        )
        hessian = -log_likelihood_hessian
        sign, log_determinant = np.linalg.slogdet(hessian)
        if sign <= 0.0 or not np.isfinite(log_determinant):
            raise ValueError("Conditional timing mode is not locally identifiable.")
        dimension = mode.size
        flat_indices = tuple(
            index
            for index in range(dimension)
            if index not in self.conditional_gaussian_indices
        )
        log_prior_volume = float(
            np.sum(
                np.log(self.conditional_upper[list(flat_indices)] - self.conditional_lower[list(flat_indices)])
            )
            if flat_indices
            else 0.0
        )
        log_marginal = (
            log_likelihood
            + 0.5 * dimension * math.log(2.0 * math.pi)
            - 0.5 * float(log_determinant)
            - log_prior_volume
        )
        return IsolatedTimingUpdateConditionalMode(
            parameter_names=self.conditional_parameter_names,
            physical_values=mode * self.conditional_scales,
            normalized_values=mode,
            log_likelihood=log_likelihood,
            log_marginal_likelihood=float(log_marginal),
            hessian=hessian,
        )

    def __call__(self, values: Sequence[float]) -> float:
        try:
            return float(self.conditional_mode(values).log_marginal_likelihood)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return -math.inf


def _isolated_update_outer_hessian(
    objective: Any,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, bool, int]:
    """Finite-difference the outer posterior, including boundary directions."""

    center = np.asarray(values, dtype=float)
    size = center.size
    widths = upper - lower
    steps = np.maximum(1.0e-6, 1.0e-4 * widths)
    directions = np.ones(size, dtype=float)
    boundary_count = 0
    for index in range(size):
        if center[index] - lower[index] <= 2.0 * steps[index]:
            directions[index] = 1.0
            boundary_count += 1
        elif upper[index] - center[index] <= 2.0 * steps[index]:
            directions[index] = -1.0
            boundary_count += 1
        else:
            directions[index] = 0.0
        if directions[index] != 0.0:
            steps[index] = min(steps[index], 0.25 * widths[index])
    f0 = float(objective(center))
    hessian = np.zeros((size, size), dtype=float)
    for row in range(size):
        first = np.zeros(size, dtype=float)
        first[row] = steps[row]
        sign = directions[row]
        if sign == 0.0:
            hessian[row, row] = (
                float(objective(center + first))
                - 2.0 * f0
                + float(objective(center - first))
            ) / steps[row] ** 2
        else:
            first *= sign
            second = 2.0 * first
            hessian[row, row] = (
                float(objective(center + second))
                - 2.0 * float(objective(center + first))
                + f0
            ) / steps[row] ** 2
        for column in range(row):
            row_boundary = directions[row] != 0.0
            column_boundary = directions[column] != 0.0
            row_step = np.zeros(size, dtype=float)
            column_step = np.zeros(size, dtype=float)
            row_step[row] = steps[row] * (directions[row] if row_boundary else 1.0)
            column_step[column] = steps[column] * (
                directions[column] if column_boundary else 1.0
            )
            if row_boundary and column_boundary:
                value = (
                    float(objective(center + row_step + column_step))
                    - float(objective(center + row_step))
                    - float(objective(center + column_step))
                    + f0
                ) / (steps[row] * steps[column])
            elif row_boundary:
                value = (
                    float(objective(center + row_step + column_step))
                    - float(objective(center + row_step - column_step))
                    - float(objective(center + column_step))
                    + float(objective(center - column_step))
                ) / (2.0 * steps[row] * steps[column])
            elif column_boundary:
                value = (
                    float(objective(center + row_step + column_step))
                    - float(objective(center - row_step + column_step))
                    - float(objective(center + row_step))
                    + float(objective(center - row_step))
                ) / (2.0 * steps[row] * steps[column])
            else:
                value = (
                    float(objective(center + row_step + column_step))
                    - float(objective(center + row_step - column_step))
                    - float(objective(center - row_step + column_step))
                    + float(objective(center - row_step - column_step))
                ) / (4.0 * steps[row] * steps[column])
            hessian[row, column] = value
            hessian[column, row] = hessian[row, column]
    return 0.5 * (hessian + hessian.T), bool(boundary_count), boundary_count


def run_isolated_timing_update_mode_portfolio(
    *,
    likelihood: Any,
    prior: Any,
    starting_values: Sequence[Sequence[float]],
    maximum_modes: int = 16,
    duplicate_tolerance: float = 1.0e-3,
) -> IsolatedTimingUpdateModePortfolio:
    """Refine and combine a supplied set of outer posterior mode starts."""

    starts = np.asarray(starting_values, dtype=float)
    if starts.ndim != 2 or starts.shape[1] != len(likelihood.parameter_names):
        raise ValueError("Timing mode starts must align with outer parameters.")
    if starts.shape[0] == 0 or int(maximum_modes) <= 0:
        raise ValueError("Timing mode refinement needs at least one start.")
    lower = np.asarray(prior.lower, dtype=float)
    upper = np.asarray(prior.upper, dtype=float)
    if lower.shape != (starts.shape[1],) or upper.shape != lower.shape:
        raise ValueError("Timing mode starts and prior bounds differ.")

    def negative_log_posterior(values: np.ndarray) -> float:
        log_likelihood = float(likelihood(values))
        log_prior = float(prior.log_density(values))
        result = log_likelihood + log_prior
        return -result if np.isfinite(result) else 1.0e300

    candidates: list[IsolatedTimingUpdateMode] = []
    for start in starts[: max(int(maximum_modes) * 4, int(maximum_modes))]:
        clipped = np.clip(start, lower + 1.0e-10, upper - 1.0e-10)
        result = minimize(
            negative_log_posterior,
            clipped,
            method="L-BFGS-B",
            bounds=tuple(zip(lower, upper, strict=True)),
            options={"maxiter": 160, "ftol": 1.0e-10, "gtol": 1.0e-6},
        )
        values = np.asarray(result.x, dtype=float)
        log_posterior = -float(result.fun)
        if not np.isfinite(log_posterior):
            continue
        hessian, boundary_active, boundary_count = _isolated_update_outer_hessian(
            negative_log_posterior,
            values,
            lower,
            upper,
        )
        sign, log_determinant = np.linalg.slogdet(hessian)
        if sign > 0.0 and np.isfinite(log_determinant):
            log_mass = (
                log_posterior
                + 0.5 * values.size * math.log(2.0 * math.pi)
                - 0.5 * float(log_determinant)
                - boundary_count * math.log(2.0)
            )
        else:
            log_mass = -math.inf
        duplicate = False
        scale = np.maximum(upper - lower, 1.0e-12)
        for existing in candidates:
            if np.linalg.norm((values - existing.values) / scale) <= duplicate_tolerance:
                duplicate = True
                break
        if not duplicate:
            candidates.append(
                IsolatedTimingUpdateMode(
                    values=values,
                    log_posterior=log_posterior,
                    log_mass=float(log_mass),
                    hessian=hessian,
                    converged=bool(result.success),
                    boundary_active=boundary_active,
                )
            )
        if len(candidates) >= int(maximum_modes):
            break
    if not candidates:
        raise ValueError("No finite isolated timing posterior mode was found.")
    log_masses = np.asarray([mode.log_mass for mode in candidates], dtype=float)
    if not np.any(np.isfinite(log_masses)):
        raise ValueError("Isolated timing modes have no positive-definite mass.")
    log_evidence = float(logsumexp(log_masses))
    weights = np.exp(log_masses - log_evidence)
    weights[~np.isfinite(weights)] = 0.0
    return IsolatedTimingUpdateModePortfolio(
        modes=tuple(candidates),
        log_evidence=log_evidence,
        mode_weights=weights,
    )


def run_isolated_timing_update_nested(
    *,
    likelihood: IsolatedTimingUpdateLogLikelihood,
    prior: IsolatedTimingUpdatePriorTransform,
    config: IsolatedTimingUpdateNestedConfig | None = None,
) -> IsolatedTimingUpdateNestedResult:
    active = config or IsolatedTimingUpdateNestedConfig()
    dimension = len(likelihood.parameter_names)
    if prior.lower.shape != (dimension,):
        raise ValueError("Timing-update likelihood and prior dimensions differ.")
    sampler = dynesty.NestedSampler(
        likelihood,
        prior,
        dimension,
        nlive=max(int(active.live_point_count), 2 * dimension + 1),
        bound=str(active.bound),
        sample=str(active.sample),
        rstate=np.random.default_rng(max(int(active.random_seed), 0)),
    )
    maximum_calls = (
        max(int(active.maximum_calls), 1)
        if active.maximum_calls is not None
        else None
    )
    sampler.run_nested(
        maxcall=maximum_calls,
        dlogz=float(active.evidence_tolerance),
        add_live=True,
        print_progress=bool(active.print_progress),
    )
    nested = sampler.results
    samples = np.asarray(nested.samples, dtype=float)
    log_likelihood = np.asarray(nested.logl, dtype=float)
    log_evidence = float(np.asarray(nested.logz, dtype=float)[-1])
    log_weights = np.asarray(nested.logwt, dtype=float)
    weights = np.exp(log_weights - log_evidence)
    weights /= np.sum(weights)
    best_index = int(np.argmax(log_likelihood))
    call_count = int(sampler.ncall)
    return IsolatedTimingUpdateNestedResult(
        hypothesis=likelihood.hypothesis,
        parameter_names=likelihood.parameter_names,
        samples=samples,
        log_likelihood=log_likelihood,
        posterior_weights=weights,
        log_evidence=log_evidence,
        log_evidence_error=float(np.asarray(nested.logzerr, dtype=float)[-1]),
        best_values=samples[best_index].copy(),
        best_log_likelihood=float(log_likelihood[best_index]),
        call_count=call_count,
        effective_sample_size=float(1.0 / np.sum(np.square(weights))),
        reached_call_limit=bool(
            maximum_calls is not None and call_count >= maximum_calls
        ),
    )


def weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantiles: Sequence[float],
) -> np.ndarray:
    samples = np.asarray(values, dtype=float)
    probabilities = np.asarray(weights, dtype=float)
    requested = np.asarray(quantiles, dtype=float)
    if samples.ndim != 1 or probabilities.shape != samples.shape:
        raise ValueError("Weighted quantiles need aligned one-dimensional samples.")
    if not np.all(np.isfinite(samples)) or not np.all(
        np.isfinite(probabilities) & (probabilities >= 0.0)
    ):
        raise ValueError("Weighted quantile inputs must be finite and nonnegative.")
    if not np.all((requested >= 0.0) & (requested <= 1.0)):
        raise ValueError("Weighted quantiles must lie in [0, 1].")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("Weighted quantiles require positive total weight.")
    order = np.argsort(samples)
    sorted_values = samples[order]
    cumulative = np.cumsum(probabilities[order]) / total
    return np.interp(requested, cumulative, sorted_values)


def hypothesis_posterior_weights(
    results: Sequence[IsolatedTimingUpdateNestedResult],
) -> dict[str, float]:
    portfolio = tuple(results)
    if not portfolio:
        raise ValueError("Hypothesis weighting requires at least one result.")
    ids = tuple(item.hypothesis.hypothesis_id for item in portfolio)
    if len(set(ids)) != len(ids):
        raise ValueError("Timing-update hypothesis ids must be unique.")
    evidence = np.asarray([item.log_evidence for item in portfolio], dtype=float)
    weights = np.exp(evidence - logsumexp(evidence))
    return {name: float(weight) for name, weight in zip(ids, weights, strict=True)}


__all__ = [
    "CollapsedAnchoredIsolatedTimingUpdateLogLikelihood",
    "IsolatedTimingUpdateConfig",
    "IsolatedTimingUpdateConditionalMode",
    "IsolatedTimingUpdateGaussianPrior",
    "IsolatedTimingUpdateGaussianPriorTransform",
    "IsolatedTimingUpdateHypothesis",
    "IsolatedTimingUpdateMode",
    "IsolatedTimingUpdateModePortfolio",
    "IsolatedTimingUpdateLogLikelihood",
    "IsolatedTimingUpdateNestedConfig",
    "IsolatedTimingUpdateNestedResult",
    "IsolatedTimingUpdatePriorTransform",
    "build_collapsed_isolated_update_prior",
    "build_gaussian_collapsed_isolated_update_prior",
    "build_isolated_update_prior",
    "collapsed_isolated_update_parameter_names",
    "hypothesis_posterior_weights",
    "isolated_update_parameter_names",
    "run_isolated_timing_update_nested",
    "run_isolated_timing_update_mode_portfolio",
    "weighted_quantile",
]
