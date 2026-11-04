from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import logsumexp

from apollo_timing.io.artifacts import sha256_file as _sha256, write_json_atomic
from apollo_timing.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
)
from apollo_timing.models.serialization import to_json_ready
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    SourceSession,
    infer_contiguous_source_sessions,
    profile_two_level_pulse_lattice,
)
from apollo_timing.inference.bayesian_partial_ell1_bootstrap import (
    _active_par_lines,
    _phase_cycles,
    _phase_sigmas,
    normalize_tempo2_tim_file,
)
from apollo_timing.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.io.timing_solver_artifacts import _json_digest, resource_identity


BLIND_ORBIT_ARCHITECTURE = "apollo_blind_keplerian_session_mixture"
_NON_MODEL_LINES = {"MODE", "NITS"}
_REPLACED_PARAMETERS = {
    "BINARY",
    "PB",
    "FB0",
    "A1",
    "T0",
    "TASC",
    "ECC",
    "OM",
    "EPS1",
    "EPS2",
    "F0",
    "F1",
}


@dataclass(frozen=True)
class BlindOrbitDiscoveryConfig:
    """Budgets and physical support for an undeclared binary orbit."""

    period_strata_days: tuple[tuple[float, float], ...] = (
        (0.2, 2.0),
        (2.0, 10.0),
        (10.0, 30.0),
        (30.0, 80.0),
        (80.0, 200.0),
        (200.0, 600.0),
        (600.0, 2_000.0),
    )
    random_seeds: tuple[int, ...] = (19, 73, 181)
    sparse_session_threshold: int = 10
    sparse_period_strata_to_expand: int = 2
    sparse_random_seeds: tuple[int, ...] = (
        5,
        11,
        17,
        23,
        31,
        43,
        59,
        71,
        89,
        107,
        127,
        149,
        173,
        199,
        227,
        257,
        293,
        331,
        373,
        419,
        467,
        521,
        577,
        631,
    )
    parallel_workers: int = 1
    minimum_frequency_support_hz: float = 0.02
    fractional_frequency_support: float = 1.5e-3
    local_integer_chi2_window: float = 100.0
    maximum_session_modes: int = 512
    frequency_temperature_hz: float = 1.0e-4
    acceleration_temperature_hz_per_s: float = 1.0e-9
    maximum_eccentricity: float = 0.95
    maximum_projected_semimajor_axis_seconds: float = 5_000.0
    maximum_mass_function_solar: float = 100.0
    spin_frequency_offset_hz: float = 3.0e-6
    spin_frequency_derivative_hz_per_day: float = 1.0e-7
    differential_evolution_population: int = 14
    differential_evolution_iterations: int = 550
    conditional_period_strata: int = 2
    conditional_period_samples: int = 4096
    conditional_period_anchor_modes: int = 34
    conditional_period_max_anchor_combinations: int = 16_384
    conditional_period_em_iterations: int = 8
    conditional_period_modes_per_stratum: int = 8
    conditional_period_refinement_samples: int = 65
    conditional_period_temperature_factors: tuple[float, ...] = (
        10.0,
        5.0,
        2.0,
        1.0,
    )
    circular_period_scout_random_seeds: tuple[int, ...] = (19, 73)
    circular_period_scout_population: int = 6
    circular_period_scout_iterations: int = 50
    maximum_initial_wrapped_residual: float = 0.25
    maximum_modes_to_refine: int = 8
    maximum_modes_to_emit: int = 4
    exact_refinement_evaluations: int = 10_000
    maximum_pint_shape_reduced_chi2: float = 25.0
    ell1_truncation_fraction_of_toa_error: float = 0.25

    def __post_init__(self) -> None:
        if not self.period_strata_days:
            raise ValueError("Blind orbit discovery requires period strata.")
        previous_upper = 0.0
        for lower, upper in self.period_strata_days:
            if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower < upper):
                raise ValueError("Every blind-orbit period stratum must be positive.")
            if lower < previous_upper:
                raise ValueError("Blind-orbit period strata cannot overlap.")
            previous_upper = upper
        positive_integers = (
            self.parallel_workers,
            self.sparse_session_threshold,
            self.sparse_period_strata_to_expand,
            self.maximum_session_modes,
            self.differential_evolution_population,
            self.differential_evolution_iterations,
            self.conditional_period_strata,
            self.conditional_period_samples,
            self.conditional_period_anchor_modes,
            self.conditional_period_max_anchor_combinations,
            self.conditional_period_em_iterations,
            self.conditional_period_modes_per_stratum,
            self.conditional_period_refinement_samples,
            self.circular_period_scout_population,
            self.circular_period_scout_iterations,
            self.maximum_modes_to_refine,
            self.maximum_modes_to_emit,
            self.exact_refinement_evaluations,
        )
        if any(int(value) <= 0 for value in positive_integers):
            raise ValueError("Blind-orbit search budgets must be positive integers.")
        positive = (
            self.minimum_frequency_support_hz,
            self.fractional_frequency_support,
            self.local_integer_chi2_window,
            self.frequency_temperature_hz,
            self.acceleration_temperature_hz_per_s,
            self.maximum_projected_semimajor_axis_seconds,
            self.maximum_mass_function_solar,
            self.spin_frequency_offset_hz,
            self.spin_frequency_derivative_hz_per_day,
            self.maximum_initial_wrapped_residual,
            self.maximum_pint_shape_reduced_chi2,
            self.ell1_truncation_fraction_of_toa_error,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Blind-orbit physical supports must be positive.")
        temperature_factors = tuple(
            float(value) for value in self.conditional_period_temperature_factors
        )
        if (
            len(temperature_factors) < 2
            or any(not np.isfinite(value) or value <= 0.0 for value in temperature_factors)
            or any(
                left <= right
                for left, right in zip(temperature_factors, temperature_factors[1:])
            )
            or not np.isclose(temperature_factors[-1], 1.0)
        ):
            raise ValueError(
                "Conditional-period temperatures must decrease to the base temperature."
            )
        if not 0.0 < self.maximum_eccentricity < 1.0:
            raise ValueError("Blind-orbit eccentricity support must lie below one.")
        if (
            not self.random_seeds
            or not self.sparse_random_seeds
            or not self.circular_period_scout_random_seeds
        ):
            raise ValueError("Blind-orbit random-seed portfolios cannot be empty.")


@dataclass(frozen=True)
class SessionDerivativeMode:
    chi2: float
    frequency_hz: float
    acceleration_hz_per_s: float
    degrees_of_freedom: int
    relative_integer_offsets: tuple[int, ...]


@dataclass(frozen=True)
class BlindOrbitDiscoveryMode:
    mode_id: str
    par_path: Path
    period_stratum_days: tuple[float, float]
    random_seed: int
    surrogate_score: float
    initial_wrapped_rms_cycles: float
    initial_max_wrapped_residual_cycles: float
    refined_relative_chi2: float
    refined_relative_reduced_chi2: float
    pint_shape_chi2: float
    pint_shape_reduced_chi2: float
    parameter_values: dict[str, float]
    model_name: str = "DD"
    ell1_truncation_seconds: float = math.inf
    ell1_tolerance_seconds: float = 0.0
    path_acquisition_tolerance_seconds: float = 0.0
    representation_policy: str = "dd_only_path_accuracy"
    continuous_mode_index: int = 0


@dataclass(frozen=True)
class _BlindModeRepresentation:
    model: object
    model_name: str
    ell1_truncation_seconds: float
    ell1_tolerance_seconds: float
    path_acquisition_tolerance_seconds: float
    representation_policy: str


@dataclass(frozen=True)
class BlindOrbitDiscoveryResult:
    normalized_tim_path: Path
    audit_path: Path
    modes: tuple[BlindOrbitDiscoveryMode, ...]
    session_count: int
    toa_count: int
    observation_span_days: float

    def to_dict(self) -> dict[str, object]:
        return to_json_ready(self)


@dataclass(frozen=True)
class _DiscoveryData:
    f0_hz: float
    pepoch_mjd: float
    reference_mjd: float
    session_reference_mjd: float
    mjds: np.ndarray
    phases: np.ndarray
    phase_sigmas: np.ndarray
    anchor_indices: np.ndarray
    non_anchor_mask: np.ndarray
    session_epochs_mjd: np.ndarray
    session_toa_counts: np.ndarray
    local_frequencies_hz: np.ndarray
    local_accelerations_hz_per_s: np.ndarray
    local_score_offsets: np.ndarray
    local_mode_mask: np.ndarray
    acceleration_session_mask: np.ndarray


@dataclass(frozen=True)
class _SurrogateMode:
    period_stratum_days: tuple[float, float]
    random_seed: int
    score: float
    state: np.ndarray
    wrapped_rms_cycles: float
    maximum_wrapped_residual_cycles: float


def _solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    eccentricity = float(eccentricity)
    if not np.isfinite(eccentricity) or not 0.0 <= eccentricity < 1.0:
        raise ValueError("Elliptic Kepler eccentricity must lie in [0, 1).")
    mean = np.mod(np.asarray(mean_anomaly, dtype=float), 2.0 * math.pi)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Kepler mean anomalies must be finite.")
    anomaly = np.array(mean, dtype=float, copy=True)
    for _ in range(15):
        denominator = 1.0 - eccentricity * np.cos(anomaly)
        anomaly -= (
            anomaly - eccentricity * np.sin(anomaly) - mean
        ) / denominator
    # Newton's usual seed can diverge near periastron at high eccentricity.
    # Preserve converged values; the elliptic equation is monotone on [0, 2pi].
    residual = anomaly - eccentricity * np.sin(anomaly) - mean
    failed = (
        ~np.isfinite(anomaly)
        | (anomaly < 0.0)
        | (anomaly > 2.0 * math.pi)
        | (np.abs(residual) > 2.0e-14)
    )
    if np.any(failed):
        target = mean[failed]
        lower = np.zeros_like(target)
        upper = np.full_like(target, 2.0 * math.pi)
        for _ in range(60):
            midpoint = 0.5 * (lower + upper)
            value = midpoint - eccentricity * np.sin(midpoint) - target
            lower = np.where(value < 0.0, midpoint, lower)
            upper = np.where(value >= 0.0, midpoint, upper)
        anomaly[failed] = 0.5 * (lower + upper)
    return anomaly


def _delay_from_state(
    state: Sequence[float],
    mjds: np.ndarray,
    *,
    reference_mjd: float,
) -> np.ndarray:
    values = np.asarray(state, dtype=float)
    period_days = math.exp(float(values[0]))
    eccentricity = float(values[1])
    omega = float(values[2])
    mean = float(values[3]) + 2.0 * math.pi * (
        np.asarray(mjds, dtype=float) - float(reference_mjd)
    ) / period_days
    anomaly = _solve_kepler(mean, eccentricity)
    root = math.sqrt(max(1.0 - eccentricity**2, 1.0e-12))
    return math.exp(float(values[4])) * (
        math.sin(omega) * (np.cos(anomaly) - eccentricity)
        + root * math.cos(omega) * np.sin(anomaly)
    )


def _velocity_from_state(
    state: np.ndarray,
    mjds: np.ndarray,
    *,
    data: _DiscoveryData,
) -> np.ndarray:
    step_seconds = 10.0
    step_days = step_seconds / 86400.0
    right = _delay_from_state(
        state, np.asarray(mjds, dtype=float) + step_days,
        reference_mjd=data.session_reference_mjd,
    )
    left = _delay_from_state(
        state, np.asarray(mjds, dtype=float) - step_days,
        reference_mjd=data.session_reference_mjd,
    )
    return float(data.f0_hz) * (right - left) / (2.0 * step_seconds)


def _mass_function_solar(state: np.ndarray) -> float:
    period_seconds = math.exp(float(state[0])) * 86400.0
    a1_seconds = math.exp(float(state[4]))
    tsun_seconds = 4.925490947e-6
    return float(
        (4.0 * math.pi**2 / tsun_seconds) * a1_seconds**3 / period_seconds**2
    )


def _enumerate_session_derivative_modes(
    *,
    times_seconds: np.ndarray,
    wrapped_phases: np.ndarray,
    phase_sigmas: np.ndarray,
    frequency_bound_hz: float,
    config: BlindOrbitDiscoveryConfig,
) -> tuple[SessionDerivativeMode, ...]:
    times = np.asarray(times_seconds, dtype=float)
    wrapped = np.asarray(wrapped_phases, dtype=float)
    sigmas = np.asarray(phase_sigmas, dtype=float)
    if times.ndim != 1 or wrapped.shape != times.shape or sigmas.shape != times.shape:
        raise ValueError("Session derivative inputs must be aligned vectors.")
    if times.size < 2:
        return ()
    scale = max(float(np.ptp(times)) / 2.0, 1.0)
    normalized_times = (times - float(np.mean(times))) / scale
    degree = _local_derivative_degree(times.size)
    design = np.column_stack(
        [normalized_times**power for power in range(degree + 1)]
    )
    weighted_design = design / sigmas[:, None]
    weighted_pseudoinverse = np.linalg.pinv(weighted_design)
    count = times.size
    pairs = {(0, count - 1), (0, count // 2), (count // 2, count - 1)}
    pairs.update((index, index + 1) for index in range(count - 1))
    by_integer_path: dict[tuple[int, ...], SessionDerivativeMode] = {}
    for left, right in sorted(pairs):
        if right <= left or times[right] <= times[left]:
            continue
        delta_time = float(times[right] - times[left])
        delta_phase = float(wrapped[right] - wrapped[left])
        lower_integer = int(
            math.ceil(-float(frequency_bound_hz) * delta_time - delta_phase)
        )
        upper_integer = int(
            math.floor(float(frequency_bound_hz) * delta_time - delta_phase)
        )
        for integer_difference in range(lower_integer, upper_integer + 1):
            slope = (delta_phase + integer_difference) / delta_time
            predicted = wrapped[left] + slope * (times - times[left])
            integers = np.rint(predicted - wrapped)
            for _ in range(8):
                coefficients = weighted_pseudoinverse @ ((wrapped + integers) / sigmas)
                updated = np.rint(design @ coefficients - wrapped)
                if np.array_equal(updated, integers):
                    break
                integers = updated
            coefficients = weighted_pseudoinverse @ ((wrapped + integers) / sigmas)
            normalized = (wrapped + integers - design @ coefficients) / sigmas
            chi2 = float(normalized @ normalized)
            relative = tuple(int(value) for value in integers - integers[0])
            mode = SessionDerivativeMode(
                chi2=chi2,
                frequency_hz=float(coefficients[1] / scale),
                acceleration_hz_per_s=(
                    float(2.0 * coefficients[2] / scale**2) if degree >= 2 else 0.0
                ),
                degrees_of_freedom=max(int(count - degree - 1), 1),
                relative_integer_offsets=relative,
            )
            prior = by_integer_path.get(relative)
            if prior is None or mode.chi2 < prior.chi2:
                by_integer_path[relative] = mode
    ordered = sorted(
        by_integer_path.values(),
        key=lambda item: (item.chi2, abs(item.frequency_hz)),
    )
    if not ordered:
        return ()
    limit = ordered[0].chi2 + float(config.local_integer_chi2_window)
    return tuple(
        mode
        for mode in ordered
        if mode.chi2 <= limit
    )[: int(config.maximum_session_modes)]


def _local_derivative_degree(toa_count: int) -> int:
    """Use only locally identifiable clock derivatives.

    Four-TOA sessions are common in survey timing. A cubic fit has one
    coefficient per datum, giving every integer path a nearly zero residual.
    These saturated fits remove the local evidence needed to distinguish
    frequency aliases in the global orbit search.
    """

    count = int(toa_count)
    if count < 2:
        raise ValueError("A local derivative fit requires at least two TOAs.")
    return 2 if count >= 6 else 1


def _build_discovery_data(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
    config: BlindOrbitDiscoveryConfig,
) -> _DiscoveryData:
    session_tuple = tuple(sessions)
    mjds = np.asarray(toas.get_mjds().value, dtype=float)
    phases = np.asarray(_phase_cycles(model, toas), dtype=float)
    sigmas = np.asarray(_phase_sigmas(model, toas), dtype=float)
    f0_hz = abs(float(model["F0"].value))
    frequency_bound = max(
        float(config.minimum_frequency_support_hz),
        float(config.fractional_frequency_support) * f0_hz,
    )
    local_modes = []
    anchor_indices = np.empty(mjds.size, dtype=int)
    non_anchor = np.ones(mjds.size, dtype=bool)
    for session in session_tuple:
        indices = np.asarray(session.toa_indices, dtype=int)
        anchor_indices[indices] = int(indices[0])
        non_anchor[indices[0]] = False
        times = (mjds[indices] - float(session.epoch_mjd)) * 86400.0
        wrapped = np.asarray(phases[indices] - np.rint(phases[indices]), dtype=float)
        modes = _enumerate_session_derivative_modes(
            times_seconds=times,
            wrapped_phases=wrapped,
            phase_sigmas=sigmas[indices],
            frequency_bound_hz=frequency_bound,
            config=config,
        )
        if not modes:
            raise RuntimeError(
                f"Session '{session.session_id}' produced no local derivative mode."
            )
        local_modes.append(modes)
    maximum_modes = max(len(modes) for modes in local_modes)
    shape = (len(session_tuple), maximum_modes)
    frequencies = np.full(shape, np.nan, dtype=float)
    accelerations = np.full(shape, np.nan, dtype=float)
    score_offsets = np.full(shape, np.nan, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    acceleration_mask = np.zeros(len(session_tuple), dtype=bool)
    for row, (session, modes) in enumerate(zip(session_tuple, local_modes, strict=True)):
        minimum_chi2 = modes[0].chi2
        for column, mode in enumerate(modes):
            frequencies[row, column] = mode.frequency_hz
            accelerations[row, column] = mode.acceleration_hz_per_s
            score_offsets[row, column] = (
                max(mode.chi2 - minimum_chi2, 0.0) / 25.0
            )
            mask[row, column] = True
        acceleration_mask[row] = (
            len(session.toa_indices) >= 6 and modes[0].degrees_of_freedom >= 2
        )
    return _DiscoveryData(
        f0_hz=f0_hz,
        pepoch_mjd=float(model["PEPOCH"].value),
        reference_mjd=float(np.mean(mjds)),
        session_reference_mjd=float(
            np.mean([session.epoch_mjd for session in session_tuple])
        ),
        mjds=mjds,
        phases=phases,
        phase_sigmas=sigmas,
        anchor_indices=anchor_indices,
        non_anchor_mask=non_anchor,
        session_epochs_mjd=np.asarray(
            [session.epoch_mjd for session in session_tuple], dtype=float
        ),
        session_toa_counts=np.asarray(
            [len(session.toa_indices) for session in session_tuple], dtype=int
        ),
        local_frequencies_hz=frequencies,
        local_accelerations_hz_per_s=accelerations,
        local_score_offsets=score_offsets,
        local_mode_mask=mask,
        acceleration_session_mask=acceleration_mask,
    )


def _state_bounds(
    period_stratum_days: tuple[float, float],
    config: BlindOrbitDiscoveryConfig,
) -> tuple[np.ndarray, np.ndarray]:
    lower_period, upper_period = period_stratum_days
    lower = np.asarray(
        (
            math.log(lower_period),
            0.0,
            0.0,
            0.0,
            math.log(1.0e-4),
            -config.spin_frequency_offset_hz,
            -config.spin_frequency_derivative_hz_per_day,
        ),
        dtype=float,
    )
    upper = np.asarray(
        (
            math.log(upper_period),
            config.maximum_eccentricity,
            2.0 * math.pi,
            2.0 * math.pi,
            math.log(config.maximum_projected_semimajor_axis_seconds),
            config.spin_frequency_offset_hz,
            config.spin_frequency_derivative_hz_per_day,
        ),
        dtype=float,
    )
    return lower, upper


def _surrogate_score(
    state: np.ndarray,
    *,
    data: _DiscoveryData,
    config: BlindOrbitDiscoveryConfig,
) -> float:
    try:
        epochs = data.session_epochs_mjd
        velocity = _velocity_from_state(state, epochs, data=data)
        step_days = 30.0 / 86400.0
        acceleration = (
            _velocity_from_state(state, epochs + step_days, data=data)
            - _velocity_from_state(state, epochs - step_days, data=data)
        ) / 60.0
        delta_days = epochs - data.session_reference_mjd
        predicted_frequency = state[5] + state[6] * delta_days + velocity
        predicted_acceleration = state[6] / 86400.0 + acceleration
        component = 0.5 * np.square(
            (data.local_frequencies_hz - predicted_frequency[:, None])
            / float(config.frequency_temperature_hz)
        )
        component += 0.5 * data.local_score_offsets
        component += (
            0.5
            * np.square(
                (data.local_accelerations_hz_per_s - predicted_acceleration[:, None])
                / float(config.acceleration_temperature_hz_per_s)
            )
            * data.acceleration_session_mask[:, None]
        )
        component = np.where(data.local_mode_mask, component, np.inf)
        score = float(-np.sum(logsumexp(-component, axis=1)))
        mass_function = _mass_function_solar(state)
        if mass_function > config.maximum_mass_function_solar:
            score += (
                math.log(mass_function / config.maximum_mass_function_solar) / 0.3
            ) ** 2
        return score if np.isfinite(score) else math.inf
    except (ArithmeticError, FloatingPointError, ValueError):
        return math.inf


def _relative_wrapped_residuals(
    state: np.ndarray,
    *,
    data: _DiscoveryData,
) -> np.ndarray:
    delta_seconds = (data.mjds - data.session_reference_mjd) * 86400.0
    spin_phase = state[5] * delta_seconds
    spin_phase += 0.5 * (state[6] / 86400.0) * np.square(delta_seconds)
    model_phase = data.phases - spin_phase - data.f0_hz * _delay_from_state(
        state,
        data.mjds,
        reference_mjd=data.session_reference_mjd,
    )
    relative = np.asarray(
        model_phase - model_phase[data.anchor_indices], dtype=float
    )
    wrapped = relative - np.rint(relative)
    return wrapped[data.non_anchor_mask]


def _search_period_stratum(
    task: tuple[
        _DiscoveryData,
        BlindOrbitDiscoveryConfig,
        tuple[float, float],
        int,
    ],
) -> _SurrogateMode:
    data, config, period_stratum, random_seed = task
    lower, upper = _state_bounds(period_stratum, config)
    fit = differential_evolution(
        lambda state: _surrogate_score(state, data=data, config=config),
        tuple(zip(lower, upper, strict=True)),
        seed=int(random_seed),
        popsize=int(config.differential_evolution_population),
        maxiter=int(config.differential_evolution_iterations),
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    state = np.asarray(fit.x, dtype=float)
    wrapped = _relative_wrapped_residuals(state, data=data)
    return _SurrogateMode(
        period_stratum_days=period_stratum,
        random_seed=int(random_seed),
        score=float(fit.fun),
        state=state,
        wrapped_rms_cycles=float(np.sqrt(np.mean(np.square(wrapped)))),
        maximum_wrapped_residual_cycles=float(np.max(np.abs(wrapped))),
    )


def _run_surrogate_tasks(
    tasks: Sequence[
        tuple[
            _DiscoveryData,
            BlindOrbitDiscoveryConfig,
            tuple[float, float],
            int,
        ]
    ],
    *,
    parallel_workers: int,
) -> list[_SurrogateMode]:
    task_list = list(tasks)
    if int(parallel_workers) > 1 and len(task_list) > 1:
        with ProcessPoolExecutor(
            max_workers=min(int(parallel_workers), len(task_list))
        ) as executor:
            return list(executor.map(_search_period_stratum, task_list))
    return [_search_period_stratum(task) for task in task_list]


def _search_circular_period_cell(
    task: tuple[
        _DiscoveryData,
        BlindOrbitDiscoveryConfig,
        tuple[float, float],
        int,
    ],
) -> _SurrogateMode:
    """Search one narrow period cell in the non-degenerate circular chart."""

    data, config, period_cell, random_seed = task
    lower_period, upper_period = period_cell
    bounds = (
        (math.log(lower_period), math.log(upper_period)),
        (0.0, 2.0 * math.pi),
        (math.log(1.0e-4), math.log(config.maximum_projected_semimajor_axis_seconds)),
        (-config.spin_frequency_offset_hz, config.spin_frequency_offset_hz),
        (
            -config.spin_frequency_derivative_hz_per_day,
            config.spin_frequency_derivative_hz_per_day,
        ),
    )

    def expand(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                values[0],
                0.0,
                0.0,
                values[1],
                values[2],
                values[3],
                values[4],
            ),
            dtype=float,
        )

    fit = differential_evolution(
        lambda values: _surrogate_score(expand(values), data=data, config=config),
        bounds,
        seed=int(random_seed),
        popsize=int(config.circular_period_scout_population),
        maxiter=int(config.circular_period_scout_iterations),
        tol=1.0e-8,
        polish=True,
        workers=1,
        updating="immediate",
    )
    state = expand(np.asarray(fit.x, dtype=float))
    wrapped = _relative_wrapped_residuals(state, data=data)
    return _SurrogateMode(
        period_stratum_days=period_cell,
        random_seed=int(random_seed),
        score=float(fit.fun),
        state=state,
        wrapped_rms_cycles=float(np.sqrt(np.mean(np.square(wrapped)))),
        maximum_wrapped_residual_cycles=float(np.max(np.abs(wrapped))),
    )


def _conditional_period_cell(
    periods: np.ndarray,
    index: int,
    *,
    support: tuple[float, float],
) -> tuple[float, float]:
    lower = (
        float(support[0])
        if int(index) == 0
        else 0.5 * (float(periods[index - 1]) + float(periods[index]))
    )
    upper = (
        float(support[1])
        if int(index) == len(periods) - 1
        else 0.5 * (float(periods[index]) + float(periods[index + 1]))
    )
    return lower, upper


def _profile_conditional_period_grid(
    *,
    periods: np.ndarray,
    temperature_hz: float,
    data: _DiscoveryData,
    config: BlindOrbitDiscoveryConfig,
    anchors: Sequence[int],
    frequencies: Sequence[np.ndarray],
    offsets: Sequence[np.ndarray],
    observations: np.ndarray,
    prior_precision: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Profile the circular mixed-integer frequency model on one period grid."""

    period_values = np.asarray(periods, dtype=float)
    temperature = float(temperature_hz)
    best_scores = np.full(period_values.size, math.inf, dtype=float)
    best_coefficients = np.full((period_values.size, 4), np.nan, dtype=float)
    delta_days = data.session_epochs_mjd - data.session_reference_mjd
    for period_index, period_days in enumerate(period_values):
        angle = 2.0 * math.pi * delta_days / float(period_days)
        design = np.column_stack(
            (np.ones_like(delta_days), delta_days, np.cos(angle), np.sin(angle))
        )
        anchor_design = design[list(anchors)]
        normal = anchor_design.T @ anchor_design / temperature**2 + prior_precision
        try:
            coefficients = (
                observations @ anchor_design / temperature**2
            ) @ np.linalg.inv(normal)
        except np.linalg.LinAlgError:
            continue
        full_normal = design.T @ design / temperature**2 + prior_precision
        try:
            full_inverse = np.linalg.inv(full_normal)
        except np.linalg.LinAlgError:
            continue
        for _ in range(int(config.conditional_period_em_iterations)):
            predicted = coefficients @ design.T
            chosen = np.empty_like(predicted)
            for row, frequency_modes in enumerate(frequencies):
                insertion = np.searchsorted(frequency_modes, predicted[:, row])
                left = np.clip(insertion - 1, 0, len(frequency_modes) - 1)
                right = np.clip(insertion, 0, len(frequency_modes) - 1)
                left_cost = np.square(frequency_modes[left] - predicted[:, row])
                left_cost += temperature**2 * offsets[row][left]
                right_cost = np.square(frequency_modes[right] - predicted[:, row])
                right_cost += temperature**2 * offsets[row][right]
                selected = np.where(right_cost < left_cost, right, left)
                chosen[:, row] = frequency_modes[selected]
            updated = (chosen @ design / temperature**2) @ full_inverse
            if float(np.max(np.abs(updated - coefficients))) < 1.0e-13:
                coefficients = updated
                break
            coefficients = updated

        predicted = coefficients @ design.T
        score = np.zeros(coefficients.shape[0], dtype=float)
        for row, frequency_modes in enumerate(frequencies):
            component = 0.5 * np.square(
                (frequency_modes[None, :] - predicted[:, row, None]) / temperature
            )
            component += 0.5 * offsets[row][None, :]
            score -= logsumexp(-component, axis=1)
        score += 0.5 * np.square(
            coefficients[:, 0] / float(config.spin_frequency_offset_hz)
        )
        score += 0.5 * np.square(
            coefficients[:, 1]
            / float(config.spin_frequency_derivative_hz_per_day)
        )
        amplitude_hz = np.hypot(coefficients[:, 2], coefficients[:, 3])
        a1_seconds = (
            amplitude_hz
            * float(period_days)
            * 86400.0
            / (2.0 * math.pi * data.f0_hz)
        )
        period_seconds = float(period_days) * 86400.0
        mass_function = (
            (4.0 * math.pi**2 / 4.925490947e-6)
            * np.power(a1_seconds, 3)
            / period_seconds**2
        )
        physical = (
            (np.abs(coefficients[:, 0]) <= config.spin_frequency_offset_hz)
            & (
                np.abs(coefficients[:, 1])
                <= config.spin_frequency_derivative_hz_per_day
            )
            & (a1_seconds >= 1.0e-4)
            & (a1_seconds <= config.maximum_projected_semimajor_axis_seconds)
            & (mass_function <= config.maximum_mass_function_solar)
        )
        if np.any(physical):
            physical_indices = np.flatnonzero(physical)
            best = int(physical_indices[np.argmin(score[physical_indices])])
            best_scores[period_index] = float(score[best])
            best_coefficients[period_index] = coefficients[best]
    return best_scores, best_coefficients


def _circular_period_cell_tasks(
    data: _DiscoveryData,
    config: BlindOrbitDiscoveryConfig,
    scouts: Sequence[_SurrogateMode],
) -> tuple[
    tuple[
        tuple[
            _DiscoveryData,
            BlindOrbitDiscoveryConfig,
            tuple[float, float],
            int,
        ],
        ...,
    ],
    tuple[_SurrogateMode, ...],
    dict[str, object],
]:
    """Profile circular integer mixtures across complete selected strata."""

    selected_strata = (
        _sparse_expansion_strata(
            scouts,
            maximum_strata=int(config.conditional_period_strata),
        )
        if scouts
        else tuple(config.period_strata_days)
    )
    mode_counts = np.sum(data.local_mode_mask, axis=1).astype(int)
    first = sorted(range(len(mode_counts)), key=lambda index: (mode_counts[index], index))[:2]
    anchors = list(first)
    available = [index for index in range(len(mode_counts)) if index not in anchors]
    if available:
        final_anchor = max(available, key=lambda index: data.session_epochs_mjd[index])
        anchors.append(final_anchor)
    if len(anchors) < 3:
        return (), (), {
            "selected_period_strata_days": selected_strata,
            "anchor_session_indices": tuple(anchors),
            "reason": "fewer_than_three_anchor_sessions",
            "strata": (),
        }
    anchors = anchors[:3]

    frequencies = []
    offsets = []
    for row in range(len(mode_counts)):
        mask = data.local_mode_mask[row]
        order = np.argsort(data.local_frequencies_hz[row, mask])
        frequencies.append(data.local_frequencies_hz[row, mask][order])
        offsets.append(data.local_score_offsets[row, mask][order])

    limits = [
        min(len(frequencies[index]), int(config.conditional_period_anchor_modes))
        for index in anchors
    ]
    maximum_combinations = int(config.conditional_period_max_anchor_combinations)
    while math.prod(limits) > maximum_combinations:
        largest = max(range(len(limits)), key=lambda index: limits[index])
        if limits[largest] <= 1:
            break
        limits[largest] -= 1
    anchor_values = []
    for anchor, limit in zip(anchors, limits, strict=True):
        rank = np.argsort(offsets[anchor])[:limit]
        anchor_values.append(frequencies[anchor][rank])
    mesh = np.meshgrid(*anchor_values, indexing="ij")
    observations = np.column_stack([values.ravel() for values in mesh])

    prior_precision = np.diag(
        (
            1.0 / float(config.spin_frequency_offset_hz) ** 2,
            1.0 / float(config.spin_frequency_derivative_hz_per_day) ** 2,
            0.0,
            0.0,
        )
    )
    temperature_factors = tuple(config.conditional_period_temperature_factors)
    coarse_temperature = (
        float(config.frequency_temperature_hz) * float(temperature_factors[0])
    )
    proposal_modes = []
    stratum_audits = []
    for period_stratum in selected_strata:
        lower_period, upper_period = period_stratum
        orbital_frequency = np.linspace(
            1.0 / float(upper_period),
            1.0 / float(lower_period),
            int(config.conditional_period_samples),
        )
        periods = np.sort(1.0 / orbital_frequency)
        best_scores, best_coefficients = _profile_conditional_period_grid(
            periods=periods,
            temperature_hz=coarse_temperature,
            data=data,
            config=config,
            anchors=anchors,
            frequencies=frequencies,
            offsets=offsets,
            observations=observations,
            prior_precision=prior_precision,
        )

        local_minima = [
            index
            for index in range(1, len(periods) - 1)
            if np.isfinite(best_scores[index])
            and best_scores[index] <= best_scores[index - 1]
            and best_scores[index] <= best_scores[index + 1]
        ]
        local_minima.extend(
            index
            for index in (0, len(periods) - 1)
            if np.isfinite(best_scores[index])
        )
        selected = sorted(local_minima, key=lambda index: best_scores[index])[
            : int(config.conditional_period_modes_per_stratum)
        ]
        selected_payload = []
        for index in selected:
            coarse_cell = _conditional_period_cell(
                periods,
                index,
                support=period_stratum,
            )
            period_days = float(periods[index])
            coefficients = best_coefficients[index]
            regularized_score = float(best_scores[index])
            refined_cell = coarse_cell
            temperature_history = [
                {
                    "temperature_hz": coarse_temperature,
                    "period_days": period_days,
                    "period_cell_days": coarse_cell,
                    "regularized_mixture_score": regularized_score,
                }
            ]
            for factor in temperature_factors[1:]:
                refine_frequency = np.linspace(
                    1.0 / float(refined_cell[1]),
                    1.0 / float(refined_cell[0]),
                    int(config.conditional_period_refinement_samples),
                )
                refine_periods = np.sort(1.0 / refine_frequency)
                refine_temperature = (
                    float(config.frequency_temperature_hz) * float(factor)
                )
                refine_scores, refine_coefficients = (
                    _profile_conditional_period_grid(
                        periods=refine_periods,
                        temperature_hz=refine_temperature,
                        data=data,
                        config=config,
                        anchors=anchors,
                        frequencies=frequencies,
                        offsets=offsets,
                        observations=observations,
                        prior_precision=prior_precision,
                    )
                )
                finite = np.flatnonzero(np.isfinite(refine_scores))
                if finite.size == 0:
                    break
                refined_index = int(finite[np.argmin(refine_scores[finite])])
                period_days = float(refine_periods[refined_index])
                coefficients = refine_coefficients[refined_index]
                regularized_score = float(refine_scores[refined_index])
                refined_cell = _conditional_period_cell(
                    refine_periods,
                    refined_index,
                    support=refined_cell,
                )
                temperature_history.append(
                    {
                        "temperature_hz": refine_temperature,
                        "period_days": period_days,
                        "period_cell_days": refined_cell,
                        "regularized_mixture_score": regularized_score,
                    }
                )
            amplitude_hz = math.hypot(float(coefficients[2]), float(coefficients[3]))
            a1_seconds = (
                amplitude_hz
                * period_days
                * 86400.0
                / (2.0 * math.pi * data.f0_hz)
            )
            phase = math.atan2(-float(coefficients[3]), float(coefficients[2])) % (
                2.0 * math.pi
            )
            state = np.asarray(
                (
                    math.log(period_days),
                    0.0,
                    0.0,
                    phase,
                    math.log(a1_seconds),
                    float(coefficients[0]),
                    float(coefficients[1]),
                )
            )
            wrapped = _relative_wrapped_residuals(state, data=data)
            mode = _SurrogateMode(
                period_stratum_days=coarse_cell,
                random_seed=0,
                score=_surrogate_score(state, data=data, config=config),
                state=state,
                wrapped_rms_cycles=float(np.sqrt(np.mean(np.square(wrapped)))),
                maximum_wrapped_residual_cycles=float(np.max(np.abs(wrapped))),
            )
            proposal_modes.append(mode)
            selected_payload.append(
                {
                    "period_days": period_days,
                    "period_cell_days": coarse_cell,
                    "refined_period_cell_days": refined_cell,
                    "regularized_mixture_score": regularized_score,
                    "temperature_refinement": temperature_history,
                    "state": state,
                }
            )
        stratum_audits.append(
            {
                "period_stratum_days": period_stratum,
                "period_sample_count": len(periods),
                "coarse_temperature_hz": coarse_temperature,
                "refinement_sample_count": int(
                    config.conditional_period_refinement_samples
                ),
                "finite_period_count": int(np.sum(np.isfinite(best_scores))),
                "minimum_regularized_mixture_score": (
                    float(np.min(best_scores[np.isfinite(best_scores)]))
                    if np.any(np.isfinite(best_scores))
                    else math.inf
                ),
                "selected_modes": selected_payload,
            }
        )

    tasks = tuple(
        (data, config, mode.period_stratum_days, int(random_seed))
        for mode in proposal_modes
        for random_seed in config.circular_period_scout_random_seeds
    )
    audit = {
        "selected_period_strata_days": selected_strata,
        "anchor_session_indices": tuple(anchors),
        "anchor_mode_limits": tuple(limits),
        "anchor_combination_count": int(observations.shape[0]),
        "strata": tuple(stratum_audits),
    }
    return tasks, tuple(proposal_modes), audit


def _run_circular_period_cell_tasks(
    tasks: Sequence[
        tuple[
            _DiscoveryData,
            BlindOrbitDiscoveryConfig,
            tuple[float, float],
            int,
        ]
    ],
    *,
    parallel_workers: int,
) -> list[_SurrogateMode]:
    task_list = list(tasks)
    if int(parallel_workers) > 1 and len(task_list) > 1:
        with ProcessPoolExecutor(
            max_workers=min(int(parallel_workers), len(task_list))
        ) as executor:
            return list(executor.map(_search_circular_period_cell, task_list))
    return [_search_circular_period_cell(task) for task in task_list]


def _sparse_expansion_strata(
    surrogate: Sequence[_SurrogateMode],
    *,
    maximum_strata: int,
) -> tuple[tuple[float, float], ...]:
    best_by_stratum: dict[tuple[float, float], _SurrogateMode] = {}
    for item in surrogate:
        prior = best_by_stratum.get(item.period_stratum_days)
        item_key = (
            item.maximum_wrapped_residual_cycles,
            item.wrapped_rms_cycles,
            item.score,
        )
        if prior is None or item_key < (
            prior.maximum_wrapped_residual_cycles,
            prior.wrapped_rms_cycles,
            prior.score,
        ):
            best_by_stratum[item.period_stratum_days] = item
    ordered = sorted(
        best_by_stratum.values(),
        key=lambda item: (
            item.maximum_wrapped_residual_cycles,
            item.wrapped_rms_cycles,
            item.score,
        ),
    )
    return tuple(
        item.period_stratum_days
        for item in ordered[: max(int(maximum_strata), 1)]
    )


def _refine_fixed_integer_path(
    mode: _SurrogateMode,
    *,
    data: _DiscoveryData,
    config: BlindOrbitDiscoveryConfig,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    state = np.asarray(mode.state, dtype=float)
    delta_seconds = (data.mjds - data.session_reference_mjd) * 86400.0

    def model_phase(values: np.ndarray) -> np.ndarray:
        spin = values[5] * delta_seconds
        spin += 0.5 * (values[6] / 86400.0) * np.square(delta_seconds)
        return data.phases - spin - data.f0_hz * _delay_from_state(
            values,
            data.mjds,
            reference_mjd=data.session_reference_mjd,
        )

    initial = model_phase(state)
    initial_relative = np.asarray(
        initial - initial[data.anchor_indices], dtype=float
    )
    integer_path = np.rint(initial_relative)
    relative_sigmas = np.sqrt(
        np.square(data.phase_sigmas)
        + np.square(data.phase_sigmas[data.anchor_indices])
    )
    lower, upper = _state_bounds(mode.period_stratum_days, config)

    def residuals(values: np.ndarray) -> np.ndarray:
        phase = model_phase(values)
        relative = np.asarray(phase - phase[data.anchor_indices], dtype=float)
        return (
            (relative - integer_path) / relative_sigmas
        )[data.non_anchor_mask]

    fit = least_squares(
        residuals,
        state,
        bounds=(lower, upper),
        x_scale=np.asarray((0.01, 0.05, 0.2, 0.2, 0.2, 1.0e-6, 1.0e-8)),
        diff_step=1.0e-5,
        max_nfev=int(config.exact_refinement_evaluations),
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    final = residuals(np.asarray(fit.x, dtype=float))
    chi2 = float(final @ final)
    dof = max(int(final.size - fit.x.size), 1)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)
    covariance *= max(chi2 / dof, 1.0)
    return np.asarray(fit.x, dtype=float), chi2, chi2 / dof, covariance


def _state_parameter_values(
    state: np.ndarray,
    *,
    base_model: object,
    data: _DiscoveryData,
) -> dict[str, float]:
    period_days = math.exp(float(state[0]))
    a1_seconds = math.exp(float(state[4]))
    t0_mjd = data.session_reference_mjd - float(state[3]) * period_days / (
        2.0 * math.pi
    )
    t0_mjd += round((data.pepoch_mjd - t0_mjd) / period_days) * period_days
    base_f1 = (
        float(base_model["F1"].value) if "F1" in base_model.params else 0.0
    )
    spin_at_pepoch = float(state[5]) + float(state[6]) * (
        data.pepoch_mjd - data.session_reference_mjd
    )
    return {
        "F0": float(base_model["F0"].value) - spin_at_pepoch,
        "F1": base_f1 - float(state[6]) / 86400.0,
        "PB": period_days,
        "A1": a1_seconds,
        "ECC": float(state[1]),
        "OM": math.degrees(float(state[2])) % 360.0,
        "T0": float(t0_mjd),
        "MASS_FUNCTION": _mass_function_solar(state),
    }


def _complete_dd_text(
    par_path: Path,
    *,
    parameter_values: dict[str, float],
) -> str:
    retained = []
    for line in _active_par_lines(Path(par_path)):
        key = line.split()[0].upper()
        if key in _REPLACED_PARAMETERS or key in _NON_MODEL_LINES:
            continue
        retained.append(line)
    retained.extend(
        (
            f"F0 {parameter_values['F0']:.18g} 1",
            f"F1 {parameter_values['F1']:.18g} 1",
            "BINARY DD",
            f"PB {parameter_values['PB']:.18g} 1",
            f"A1 {parameter_values['A1']:.18g} 1",
            f"ECC {parameter_values['ECC']:.18g} 1",
            f"OM {parameter_values['OM']:.18g} 1",
            f"T0 {parameter_values['T0']:.18g} 1",
        )
    )
    return "\n".join(retained) + "\n"


def _complete_ell1_text(
    par_path: Path,
    *,
    parameter_values: dict[str, float],
) -> str:
    retained = []
    for line in _active_par_lines(Path(par_path)):
        key = line.split()[0].upper()
        if key in _REPLACED_PARAMETERS or key in _NON_MODEL_LINES:
            continue
        retained.append(line)
    eccentricity = float(parameter_values["ECC"])
    omega_radians = math.radians(float(parameter_values["OM"]))
    period_days = float(parameter_values["PB"])
    tasc_mjd = float(parameter_values["T0"]) - (
        omega_radians / (2.0 * math.pi)
    ) * period_days
    eps1 = eccentricity * math.sin(omega_radians)
    eps2 = eccentricity * math.cos(omega_radians)
    retained.extend(
        (
            f"F0 {parameter_values['F0']:.18g} 1",
            f"F1 {parameter_values['F1']:.18g} 1",
            "BINARY ELL1",
            f"PB {period_days:.18g} 1",
            f"A1 {parameter_values['A1']:.18g} 1",
            f"TASC {tasc_mjd:.18g} 1",
            f"EPS1 {eps1:.18g} 1",
            f"EPS2 {eps2:.18g} 1",
        )
    )
    return "\n".join(retained) + "\n"


def _blind_mode_models(
    par_path: Path,
    *,
    parameter_values: dict[str, float],
    toas: object,
    config: BlindOrbitDiscoveryConfig,
) -> tuple[_BlindModeRepresentation, ...]:
    errors_seconds = np.asarray(toas.get_errors().to_value("s"), dtype=float)
    finite_errors = errors_seconds[np.isfinite(errors_seconds) & (errors_seconds > 0.0)]
    median_error = float(np.median(finite_errors)) if finite_errors.size else 1.0e-6
    truncation = (
        abs(float(parameter_values["A1"]))
        * float(parameter_values["ECC"]) ** 2
    )
    tolerance = (
        float(config.ell1_truncation_fraction_of_toa_error) * median_error
    )
    spin_frequency = abs(float(parameter_values["F0"]))
    if not np.isfinite(spin_frequency) or spin_frequency <= 0.0:
        raise ValueError("Blind-orbit candidate F0 must be finite and nonzero.")
    acquisition_tolerance = (
        float(config.maximum_initial_wrapped_residual) / spin_frequency
    )
    if truncation <= tolerance:
        model_names = ("ELL1",)
        policy = "ell1_only_final_precision"
    elif truncation <= acquisition_tolerance:
        model_names = ("ELL1", "DD")
        policy = "ell1_then_dd_path_acquisition"
    else:
        model_names = ("DD",)
        policy = "dd_only_path_accuracy"

    representations = []
    for model_name in model_names:
        model_text = (
            _complete_ell1_text(par_path, parameter_values=parameter_values)
            if model_name == "ELL1"
            else _complete_dd_text(par_path, parameter_values=parameter_values)
        )
        representations.append(
            _BlindModeRepresentation(
                model=_pint_model_from_text(model_text),
                model_name=model_name,
                ell1_truncation_seconds=truncation,
                ell1_tolerance_seconds=tolerance,
                path_acquisition_tolerance_seconds=acquisition_tolerance,
                representation_policy=policy,
            )
        )
    return tuple(representations)




def _pint_model_from_text(text: str) -> object:
    from io import StringIO
    from pint.models import get_model

    return get_model(StringIO(text), allow_tcb=True)


def _atomic_json(path: Path, payload: object) -> Path:
    return write_json_atomic(path, payload, trailing_newline=True)


def _resume_discovery_result(
    *,
    output: Path,
    audit_path: Path,
    payload: dict[str, object],
) -> BlindOrbitDiscoveryResult | None:
    if payload.get("architecture") != BLIND_ORBIT_ARCHITECTURE:
        return None
    emitted = payload.get("emitted_modes")
    if not isinstance(emitted, list):
        return None
    modes = []
    for item in emitted:
        if not isinstance(item, dict):
            return None
        recorded_path = Path(str(item["par_path"]))
        local_path = output / "modes" / recorded_path.name
        par_path = local_path if local_path.is_file() else recorded_path
        if not par_path.is_file():
            return None
        modes.append(
            BlindOrbitDiscoveryMode(
                mode_id=str(item["mode_id"]),
                par_path=par_path,
                period_stratum_days=tuple(
                    float(value) for value in item["period_stratum_days"]
                ),
                random_seed=int(item["random_seed"]),
                surrogate_score=float(item["surrogate_score"]),
                initial_wrapped_rms_cycles=float(
                    item["initial_wrapped_rms_cycles"]
                ),
                initial_max_wrapped_residual_cycles=float(
                    item["initial_max_wrapped_residual_cycles"]
                ),
                refined_relative_chi2=float(item["refined_relative_chi2"]),
                refined_relative_reduced_chi2=float(
                    item["refined_relative_reduced_chi2"]
                ),
                pint_shape_chi2=float(item["pint_shape_chi2"]),
                pint_shape_reduced_chi2=float(
                    item["pint_shape_reduced_chi2"]
                ),
                parameter_values={
                    str(name): float(value)
                    for name, value in dict(item["parameter_values"]).items()
                },
                model_name=str(item.get("model_name", "DD")),
                ell1_truncation_seconds=float(
                    item.get("ell1_truncation_seconds", math.inf)
                ),
                ell1_tolerance_seconds=float(
                    item.get("ell1_tolerance_seconds", 0.0)
                ),
                path_acquisition_tolerance_seconds=float(
                    item.get("path_acquisition_tolerance_seconds", 0.0)
                ),
                representation_policy=str(
                    item.get("representation_policy", "dd_only_path_accuracy")
                ),
                continuous_mode_index=int(item.get("continuous_mode_index", 0)),
            )
        )
    normalized_tim = output / "input.normalized.tim"
    if not normalized_tim.is_file():
        return None
    return BlindOrbitDiscoveryResult(
        normalized_tim_path=normalized_tim,
        audit_path=audit_path,
        modes=tuple(modes),
        session_count=int(payload["session_count"]),
        toa_count=int(payload["toa_count"]),
        observation_span_days=float(payload["observation_span_days"]),
    )


def discover_blind_binary_orbits(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: BlindOrbitDiscoveryConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> BlindOrbitDiscoveryResult:
    """Infer a Keplerian posterior portfolio without BINARY or PB inputs."""

    active = config or BlindOrbitDiscoveryConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    if not par.is_file() or not tim.is_file():
        missing = par if not par.is_file() else tim
        raise FileNotFoundError(missing)
    if any(
        line.split()[0].upper() == "BINARY"
        for line in _active_par_lines(par)
    ):
        raise ValueError("Blind orbit discovery requires a par without BINARY.")
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "audit.json"
    run_fingerprint = _json_digest(dict(schema=2, par=_sha256(par), tim=_sha256(tim),
                                        config=active, clock=resource_identity(clock_override_path)))
    if resume and audit_path.is_file():
        prior = json.loads(audit_path.read_text(encoding="utf-8"))
        if prior.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("Blind discovery resume inputs, clock, configuration or schema changed.")
        for name, digest in prior.get("artifact_hashes", {}).items():
            path = output / name
            if not path.is_file() or _sha256(path) != digest:
                raise RuntimeError(f"Blind discovery cached artifact changed: {name}")
        if not prior.get("artifact_hashes"):
            raise RuntimeError("Blind discovery cache has no artifact integrity record.")
        if (
            prior.get("inputs", {}).get("par_sha256") != _sha256(par)
            or prior.get("inputs", {}).get("tim_sha256") != _sha256(tim)
        ):
            raise RuntimeError("Blind discovery resume inputs changed.")
        completed = _resume_discovery_result(
            output=output,
            audit_path=audit_path,
            payload=prior,
        )
        if completed is not None:
            return completed

    normalized_tim = normalize_tempo2_tim_file(
        input_path=tim,
        output_path=output / "input.normalized.tim",
    )
    base_model = _load_pint_model(par)
    toas = _load_pint_toas(
        normalized_tim,
        seed_model=base_model,
        clock_override_path=clock_override_path,
    )
    sessions = infer_contiguous_source_sessions(toas)
    if len(sessions) < 3:
        raise RuntimeError("Blind binary discovery requires at least three sessions.")
    data = _build_discovery_data(
        model=base_model,
        toas=toas,
        sessions=sessions,
        config=active,
    )
    (
        circular_tasks,
        circular_period_proposals,
        circular_period_audit,
    ) = _circular_period_cell_tasks(data, active, ())
    surrogate = list(circular_period_proposals)
    initial_gate = float(active.maximum_initial_wrapped_residual)
    period_scan_eligible = any(
        item.maximum_wrapped_residual_cycles <= initial_gate
        for item in circular_period_proposals
    )
    circular_modes = []
    circular_cell_search_executed = False
    if not period_scan_eligible:
        circular_cell_search_executed = True
        circular_modes = _run_circular_period_cell_tasks(
            circular_tasks,
            parallel_workers=active.parallel_workers,
        )
        surrogate.extend(circular_modes)

    broad_fallback_triggered = not any(
        item.maximum_wrapped_residual_cycles <= initial_gate for item in surrogate
    )
    keplerian_scout_modes = []
    expanded_strata: tuple[tuple[float, float], ...] = ()
    if broad_fallback_triggered:
        initial_tasks = [
            (data, active, tuple(period_stratum), int(seed))
            for period_stratum in active.period_strata_days
            for seed in active.random_seeds
        ]
        keplerian_scout_modes = _run_surrogate_tasks(
            initial_tasks,
            parallel_workers=active.parallel_workers,
        )
        if len(sessions) <= int(active.sparse_session_threshold):
            expanded_strata = _sparse_expansion_strata(
                keplerian_scout_modes,
                maximum_strata=active.sparse_period_strata_to_expand,
            )
            initial_seeds = {int(seed) for seed in active.random_seeds}
            expansion_tasks = [
                (data, active, tuple(period_stratum), int(seed))
                for period_stratum in expanded_strata
                for seed in active.sparse_random_seeds
                if int(seed) not in initial_seeds
            ]
            keplerian_scout_modes.extend(
                _run_surrogate_tasks(
                    expansion_tasks,
                    parallel_workers=active.parallel_workers,
                )
            )
        surrogate.extend(keplerian_scout_modes)
    surrogate.sort(
        key=lambda item: (
            item.maximum_wrapped_residual_cycles,
            item.wrapped_rms_cycles,
            item.score,
        )
    )
    eligible = [
        item
        for item in surrogate
        if item.maximum_wrapped_residual_cycles
        <= float(active.maximum_initial_wrapped_residual)
    ][: int(active.maximum_modes_to_refine)]
    refined = []
    for item in eligible:
        try:
            state, chi2, reduced_chi2, covariance = _refine_fixed_integer_path(
                item,
                data=data,
                config=active,
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        refined.append((reduced_chi2, chi2, item, state, covariance))
    refined.sort(key=lambda item: (item[0], item[1], item[2].score))

    mode_groups: list[list[BlindOrbitDiscoveryMode]] = []
    mode_dir = output / "modes"
    for index, (reduced_chi2, chi2, item, state, covariance) in enumerate(
        refined[: int(active.maximum_modes_to_emit)]
    ):
        parameters = _state_parameter_values(
            state,
            base_model=base_model,
            data=data,
        )
        representations = _blind_mode_models(
            par,
            parameter_values=parameters,
            toas=toas,
            config=active,
        )
        state_modes = []
        for representation in representations:
            model = representation.model
            model_name = representation.model_name
            profile = profile_two_level_pulse_lattice(
                np.asarray(_phase_cycles(model, toas), dtype=float),
                _phase_sigmas(model, toas),
                sessions,
                integer_radius=0,
            )
            pint_dof = max(len(toas.table) - len(sessions) - 7, 1)
            pint_reduced = float(profile.local_chi2 / pint_dof)
            mode_id = f"blind_{model_name.lower()}_{index:03d}"
            mode_path = mode_dir / f"{mode_id}.par"
            write_pint_model_parfile_roundtrip_safe(model=model, path=mode_path)
            parameter_payload = dict(parameters)
            parameter_payload["STATE_COVARIANCE_TRACE"] = float(
                np.trace(covariance)
            )
            state_modes.append(
                BlindOrbitDiscoveryMode(
                    mode_id=mode_id,
                    par_path=mode_path,
                    period_stratum_days=item.period_stratum_days,
                    random_seed=item.random_seed,
                    surrogate_score=item.score,
                    initial_wrapped_rms_cycles=item.wrapped_rms_cycles,
                    initial_max_wrapped_residual_cycles=(
                        item.maximum_wrapped_residual_cycles
                    ),
                    refined_relative_chi2=chi2,
                    refined_relative_reduced_chi2=reduced_chi2,
                    pint_shape_chi2=float(profile.local_chi2),
                    pint_shape_reduced_chi2=pint_reduced,
                    parameter_values=parameter_payload,
                    model_name=model_name,
                    ell1_truncation_seconds=(
                        representation.ell1_truncation_seconds
                    ),
                    ell1_tolerance_seconds=(
                        representation.ell1_tolerance_seconds
                    ),
                    path_acquisition_tolerance_seconds=(
                        representation.path_acquisition_tolerance_seconds
                    ),
                    representation_policy=representation.representation_policy,
                    continuous_mode_index=index,
                )
            )
        mode_groups.append(state_modes)
    mode_groups.sort(
        key=lambda group: min(
            (
                mode.pint_shape_reduced_chi2,
                mode.refined_relative_reduced_chi2,
                mode.surrogate_score,
            )
            for mode in group
        )
    )
    modes = [mode for group in mode_groups for mode in group]
    result = BlindOrbitDiscoveryResult(
        normalized_tim_path=normalized_tim,
        audit_path=audit_path,
        modes=tuple(modes),
        session_count=len(sessions),
        toa_count=len(toas.table),
        observation_span_days=float(np.ptp(data.mjds)),
    )
    _atomic_json(
        audit_path,
        {
            "schema_version": 2,
            "run_fingerprint": run_fingerprint,
            "artifact_hashes": {str(path.relative_to(output)): _sha256(path)
                for path in (normalized_tim, *(mode.par_path for mode in modes))},
            "architecture": BLIND_ORBIT_ARCHITECTURE,
            "inputs": {
                "par": str(par),
                "tim": str(tim),
                "par_sha256": _sha256(par),
                "tim_sha256": _sha256(tim),
            },
            "scientific_contract": {
                "historical_solution_seed": "forbidden",
                "source_class_prior": "forbidden",
                "manual_orbital_parameter_seed": "forbidden",
                "manual_integer_wrap": "forbidden",
                "local_integer_policy": "marginalized session derivative mixture",
                "global_model": "Kepler equation constrained DD coordinates",
                "candidate_ranker": "uncompressed TOA session-shape likelihood",
                "path_acquisition_representation": (
                    "ELL1 when its truncation is below the integer-path gate; "
                    "DD is retained when ELL1 is not final-precision accurate"
                ),
                "final_certifier": "downstream APOLLO lattice and PINT WLS",
            },
            "config": active,
            "toa_count": result.toa_count,
            "session_count": result.session_count,
            "observation_span_days": result.observation_span_days,
            "keplerian_scout_modes": keplerian_scout_modes,
            "broad_keplerian_fallback": {
                "triggered": broad_fallback_triggered,
                "trigger_reason": (
                    "deterministic_period_and_cell_searches_failed_initial_gate"
                    if broad_fallback_triggered
                    else "deterministic_period_scan_passed_initial_gate"
                ),
                "mode_count": len(keplerian_scout_modes),
            },
            "conditional_circular_search": {
                "triggered": True,
                "trigger_reason": "primary_deterministic_period_scan",
                "cell_search_executed": circular_cell_search_executed,
                "task_count": (
                    len(circular_tasks) if circular_cell_search_executed else 0
                ),
                "potential_task_count": len(circular_tasks),
                "period_proposal_count": len(circular_period_proposals),
                "eligible_period_proposal_count": sum(
                    item.maximum_wrapped_residual_cycles
                    <= float(active.maximum_initial_wrapped_residual)
                    for item in circular_period_proposals
                ),
                "period_scan": circular_period_audit,
                "mode_count": len(circular_modes),
                "eligible_mode_count": sum(
                    item.maximum_wrapped_residual_cycles
                    <= float(active.maximum_initial_wrapped_residual)
                    for item in circular_modes
                ),
                "modes": circular_modes,
            },
            "surrogate_modes": surrogate,
            "sparse_expansion": {
                "triggered": bool(expanded_strata),
                "session_threshold": active.sparse_session_threshold,
                "expanded_period_strata_days": expanded_strata,
                "additional_seed_count": (
                    len(active.sparse_random_seeds) if expanded_strata else 0
                ),
            },
            "emitted_modes": result.modes,
        },
    )
    return result


__all__ = [
    "BLIND_ORBIT_ARCHITECTURE",
    "BlindOrbitDiscoveryConfig",
    "BlindOrbitDiscoveryMode",
    "BlindOrbitDiscoveryResult",
    "SessionDerivativeMode",
    "discover_blind_binary_orbits",
]


def _blind_mode_model(
    par_path: Path,
    *,
    parameter_values: dict[str, float],
    toas: object,
    config: BlindOrbitDiscoveryConfig,
) -> tuple[object, str, float, float]:
    representation = _blind_mode_models(
        par_path,
        parameter_values=parameter_values,
        toas=toas,
        config=config,
    )[0]
    return (
        representation.model,
        representation.model_name,
        representation.ell1_truncation_seconds,
        representation.ell1_tolerance_seconds,
    )
