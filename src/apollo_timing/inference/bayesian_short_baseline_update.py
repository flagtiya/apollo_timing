from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import betaln, logsumexp

from apollo_timing.inference.bayesian_robust_phase_likelihood import (
    RobustPhaseNoiseConfig,
    RobustPhaseNoiseModel,
)
from apollo_timing.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from apollo_timing.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)


@dataclass(frozen=True)
class ShortBaselineUpdateConfig:
    """Controls a sequential isolated-pulsar update over a few new sessions."""

    maximum_baseline_state_modes: int = 128
    maximum_integer_candidates: int = 32
    maximum_output_modes: int = 128
    maximum_integer_nodes: int = 1_000_000
    enable_shared_profile_state: bool = False
    historical_posterior_probability: float = 0.98
    broad_reconstruction_probability: float = 0.02
    two_state_prior_probability: float = 0.5
    state_frequency_alpha: float = 1.0
    state_frequency_beta: float = 1.0
    state_offset_prior_mean_cycles: float = 0.45
    state_offset_prior_sigma_cycles: float = 0.12
    minimum_state_offset_cycles: float = 0.10
    maximum_state_offset_cycles: float = 0.60
    phase_connection_probability_threshold: float = 0.95
    maximum_historical_posterior_shift_sigma: float = 5.0
    previous_posterior_sigma_scale: float = 4.0
    coordinate_prior_sigmas: tuple[float, float, float, float] = (
        600.0,
        600.0,
        200.0,
        200.0,
    )
    global_phase_prior_sigma_cycles: float = 0.5
    phase_noise: RobustPhaseNoiseConfig = field(
        default_factory=RobustPhaseNoiseConfig
    )

    def __post_init__(self) -> None:
        positive_integers = (
            self.maximum_baseline_state_modes,
            self.maximum_integer_candidates,
            self.maximum_output_modes,
            self.maximum_integer_nodes,
        )
        if any(int(value) <= 0 for value in positive_integers):
            raise ValueError("Short-baseline update budgets must be positive.")
        probability = float(self.two_state_prior_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError("The two-state prior probability must lie in (0, 1).")
        update_probabilities = (
            float(self.historical_posterior_probability),
            float(self.broad_reconstruction_probability),
        )
        if any(value <= 0.0 or value >= 1.0 for value in update_probabilities):
            raise ValueError("Update-component probabilities must lie in (0, 1).")
        if not np.isclose(sum(update_probabilities), 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError("Update-component probabilities must sum to one.")
        if not 0.5 < float(self.phase_connection_probability_threshold) < 1.0:
            raise ValueError("The phase-connection threshold must lie in (0.5, 1).")
        positive = (
            self.state_frequency_alpha,
            self.state_frequency_beta,
            self.state_offset_prior_sigma_cycles,
            self.global_phase_prior_sigma_cycles,
            self.maximum_historical_posterior_shift_sigma,
            self.previous_posterior_sigma_scale,
            *self.coordinate_prior_sigmas,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Short-baseline update scales must be positive and finite.")
        if not (
            0.0
            < float(self.minimum_state_offset_cycles)
            < float(self.maximum_state_offset_cycles)
            < 1.0
        ):
            raise ValueError("Profile-state offset support must lie inside (0, 1).")


@dataclass(frozen=True)
class ShortBaselineUpdateMode:
    prior_component_id: str
    state_count: int
    state_labels: tuple[int, ...]
    extension_integer_offsets: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    coordinate_mean: np.ndarray
    coordinate_covariance: np.ndarray
    native_parameter_names: tuple[str, ...]
    native_parameter_values: np.ndarray
    native_parameter_covariance: np.ndarray
    profile_offset_cycles: float | None
    profile_offset_sigma_cycles: float | None
    historical_posterior_shift_sigma: float
    log_weight: float
    posterior_weight: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.coordinate_mean, dtype=float)
        covariance = np.asarray(self.coordinate_covariance, dtype=float)
        native = np.asarray(self.native_parameter_values, dtype=float)
        native_covariance = np.asarray(self.native_parameter_covariance, dtype=float)
        if mean.shape != (len(self.coordinate_names),):
            raise ValueError("Update coordinates do not match their names.")
        if covariance.shape != (mean.size, mean.size):
            raise ValueError("Update coordinate covariance has the wrong shape.")
        if native.shape != (len(self.native_parameter_names),):
            raise ValueError("Native timing values do not match their names.")
        if native_covariance.shape != (native.size, native.size):
            raise ValueError("Native timing covariance has the wrong shape.")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("Update coordinates and covariance must be finite.")
        if not np.all(np.isfinite(native)) or not np.all(
            np.isfinite(native_covariance)
        ):
            raise ValueError("Native timing values and covariance must be finite.")
        object.__setattr__(self, "coordinate_mean", mean.copy())
        object.__setattr__(self, "coordinate_covariance", covariance.copy())
        object.__setattr__(self, "native_parameter_values", native.copy())
        object.__setattr__(self, "native_parameter_covariance", native_covariance.copy())


@dataclass(frozen=True)
class ShortBaselineUpdateResult:
    modes: tuple[ShortBaselineUpdateMode, ...]
    status: str
    unique_phase_connection: bool
    unique_solution: bool
    historical_posterior_consistent: bool
    best_historical_posterior_shift_sigma: float
    dominant_integer_path_probability: float
    dominant_profile_model_probability: float
    dominant_prior_component_probability: float
    retained_probability: float
    log_evidence: float
    baseline_session_indices: tuple[int, ...]
    extension_session_indices: tuple[int, ...]
    session_epochs_mjd: np.ndarray
    session_intercepts_cycles: np.ndarray
    session_sigmas_cycles: np.ndarray
    prior_native_parameter_covariance: np.ndarray
    declared_historical_native_parameter_covariance: np.ndarray
    phase_path_probabilities: tuple[tuple[tuple[int, ...], float], ...]
    profile_model_probabilities: tuple[tuple[int, float], ...]
    prior_component_probabilities: tuple[tuple[str, float], ...]

    @property
    def best_mode(self) -> ShortBaselineUpdateMode:
        if not self.modes:
            raise ValueError("The short-baseline update retained no modes.")
        return self.modes[0]

    @property
    def marginal_variance_reduction(self) -> tuple[tuple[str, float], ...]:
        prior = np.diag(np.asarray(self.prior_native_parameter_covariance, dtype=float))
        posterior = np.diag(self.best_mode.native_parameter_covariance)
        ratios = np.divide(
            prior,
            posterior,
            out=np.full(prior.shape, math.inf, dtype=float),
            where=posterior > 0.0,
        )
        return tuple(
            (name, float(value))
            for name, value in zip(_NATIVE_PARAMETER_NAMES, ratios, strict=True)
        )


@dataclass(frozen=True)
class ShortBaselineUpdateArtifacts:
    output_directory: Path
    manifest_path: Path
    candidate_par_paths: tuple[Path, ...]
    preferred_par_path: Path | None


@dataclass(frozen=True)
class _BaselineStateMode:
    prior_component_id: str
    prior_log_weight: float
    state_count: int
    labels: tuple[int, ...]
    mean: np.ndarray
    covariance: np.ndarray
    log_evidence: float


@dataclass(frozen=True)
class _RawUpdateMode:
    prior_component_id: str
    state_count: int
    labels: tuple[int, ...]
    integer_offsets: tuple[int, ...]
    mean: np.ndarray
    covariance: np.ndarray
    log_weight: float


_COORDINATE_NAMES = (
    "SKY_EAST_ARCSEC",
    "SKY_NORTH_ARCSEC",
    "DELTA_F0_X1E9_HZ",
    "DELTA_F1_X1E16_HZ_S",
    "GLOBAL_PHASE_CYCLES",
)
_NATIVE_PARAMETER_NAMES = ("RAJ", "DECJ", "F0", "F1")


def _regularized_cholesky(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix).T)
    scale = max(float(np.max(np.diag(symmetric))), 1.0)
    for exponent in range(12):
        jitter = (10.0**exponent) * np.finfo(float).eps * scale
        try:
            return np.linalg.cholesky(symmetric + jitter * np.eye(symmetric.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise ValueError("The short-baseline predictive covariance is not positive definite.")


def _gaussian_log_density(residual: np.ndarray, covariance: np.ndarray) -> float:
    cholesky = _regularized_cholesky(covariance)
    solved = np.linalg.solve(cholesky, np.asarray(residual, dtype=float))
    log_determinant = 2.0 * float(np.sum(np.log(np.diag(cholesky))))
    return float(
        -0.5
        * (
            residual.size * math.log(2.0 * math.pi)
            + log_determinant
            + float(solved @ solved)
        )
    )


def _linear_gaussian_update(
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    noise_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    predictive_covariance = (
        noise_covariance + design @ prior_covariance @ design.T
    )
    innovation = target - design @ prior_mean
    cholesky = _regularized_cholesky(predictive_covariance)
    solved_design_covariance = np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, design @ prior_covariance),
    )
    gain = solved_design_covariance.T
    mean = prior_mean + gain @ innovation
    covariance = prior_covariance - gain @ design @ prior_covariance
    covariance = 0.5 * (covariance + covariance.T)
    return (
        mean,
        covariance,
        _gaussian_log_density(innovation, predictive_covariance),
    )


def _specific_state_log_prior(
    labels: Sequence[int],
    *,
    alpha: float,
    beta: float,
) -> float:
    values = tuple(int(value) for value in labels)
    count_one = sum(values)
    count_zero = len(values) - count_one
    return float(
        betaln(alpha + count_one, beta + count_zero) - betaln(alpha, beta)
    )


def _state_assignments(session_count: int) -> tuple[tuple[int, ...], ...]:
    if session_count <= 0:
        raise ValueError("A profile-state assignment needs at least one session.")
    assignments = []
    for suffix in product((0, 1), repeat=session_count - 1):
        # The first session fixes the label exchange symmetry.  The all-zero
        # assignment is retained because a profile state can first appear in
        # the extension data rather than in the historical baseline.
        assignments.append((0, *suffix))
    return tuple(assignments)


def _canonical_state_log_prior(
    labels: Sequence[int],
    *,
    alpha: float,
    beta: float,
) -> float:
    """Return a normalized prior for a nontrivial canonical state partition."""

    values = tuple(int(value) for value in labels)
    if not values or values[0] != 0 or not any(values):
        raise ValueError("A canonical two-state partition must be nontrivial.")
    normalization = _canonical_state_log_normalization(len(values), alpha, beta)
    return float(
        _specific_state_log_prior(values, alpha=alpha, beta=beta) - normalization
    )


@lru_cache(maxsize=32)
def _canonical_state_log_normalization(
    session_count: int,
    alpha: float,
    beta: float,
) -> float:
    return float(
        logsumexp(
            np.asarray(
                [
                    _specific_state_log_prior(candidate, alpha=alpha, beta=beta)
                    for candidate in _state_assignments(session_count)
                    if any(candidate)
                ],
                dtype=float,
            )
        )
    )


def _session_phase_basis(
    evaluator: Any,
    noise_model: RobustPhaseNoiseModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    names = tuple(str(name).upper() for name in evaluator.parameter_names)
    if not set(_NATIVE_PARAMETER_NAMES).issubset(names):
        raise ValueError(
            "Short-baseline isolated updates require free RAJ, DECJ, F0, and F1."
        )
    indices = {name: names.index(name) for name in _NATIVE_PARAMETER_NAMES}
    base_values = np.asarray(evaluator.base_parameter_values, dtype=float).copy()
    base_phase = np.asarray(evaluator.raw_phase_residuals(base_values), dtype=float)
    base_profile = noise_model.profile(base_phase, evaluator.sessions)
    local_offsets = base_profile.local_integer_offsets
    base_intercepts = np.asarray(base_profile.session_intercepts, dtype=float)

    dec_degrees = float(base_values[indices["DECJ"]])
    cos_dec = math.cos(math.radians(dec_degrees))
    if abs(cos_dec) <= 1.0e-8:
        raise ValueError("Equatorial tangent coordinates are singular at the pole.")
    native_steps = np.asarray(
        (
            1.0 / (15.0 * 3600.0 * cos_dec),
            1.0 / 3600.0,
            1.0e-9,
            1.0e-16,
        ),
        dtype=float,
    )
    parameter_indices = np.asarray(
        tuple(indices[name] for name in _NATIVE_PARAMETER_NAMES), dtype=int
    )
    columns = []
    try:
        for parameter_index, step in zip(parameter_indices, native_steps, strict=True):
            plus_values = base_values.copy()
            minus_values = base_values.copy()
            plus_values[parameter_index] += step
            minus_values[parameter_index] -= step
            plus = noise_model.profile(
                evaluator.raw_phase_residuals(plus_values),
                evaluator.sessions,
                fixed_local_integer_offsets=local_offsets,
            ).session_intercepts
            minus = noise_model.profile(
                evaluator.raw_phase_residuals(minus_values),
                evaluator.sessions,
                fixed_local_integer_offsets=local_offsets,
            ).session_intercepts
            columns.append(0.5 * (np.asarray(plus) - np.asarray(minus)))
    finally:
        evaluator.raw_phase_residuals(base_values)
    basis = np.column_stack(columns)
    return (
        base_intercepts,
        np.sqrt(np.asarray(base_profile.session_intercept_variances, dtype=float)),
        basis,
        native_steps,
        local_offsets,
    )


def _baseline_modes(
    *,
    intercepts: np.ndarray,
    variances: np.ndarray,
    basis: np.ndarray,
    baseline_indices: np.ndarray,
    parameter_prior_covariance: np.ndarray,
    prior_component_id: str,
    prior_log_weight: float,
    condition_timing_parameters: bool,
    config: ShortBaselineUpdateConfig,
) -> list[_BaselineStateMode]:
    target = -intercepts[baseline_indices]
    noise = np.diag(variances[baseline_indices])
    phase_design = -np.ones((baseline_indices.size, 1), dtype=float)
    if condition_timing_parameters:
        one_design = np.column_stack((basis[baseline_indices], phase_design))
        one_prior_mean = np.zeros(5, dtype=float)
        one_prior_covariance = np.zeros((5, 5), dtype=float)
        one_prior_covariance[:4, :4] = parameter_prior_covariance
        one_prior_covariance[4, 4] = (
            float(config.global_phase_prior_sigma_cycles) ** 2
        )
        one_mean, one_covariance, one_evidence = _linear_gaussian_update(
            prior_mean=one_prior_mean,
            prior_covariance=one_prior_covariance,
            design=one_design,
            target=target,
            noise_covariance=noise,
        )
    else:
        phase_prior_mean = np.zeros(1, dtype=float)
        phase_prior_covariance = np.asarray(
            ((float(config.global_phase_prior_sigma_cycles) ** 2,),), dtype=float
        )
        phase_mean, phase_covariance, one_evidence = _linear_gaussian_update(
            prior_mean=phase_prior_mean,
            prior_covariance=phase_prior_covariance,
            design=phase_design,
            target=target,
            noise_covariance=noise,
        )
        one_mean = np.concatenate((np.zeros(4, dtype=float), phase_mean))
        one_covariance = np.zeros((5, 5), dtype=float)
        one_covariance[:4, :4] = parameter_prior_covariance
        one_covariance[4:, 4:] = phase_covariance
    modes = [
        _BaselineStateMode(
            prior_component_id=prior_component_id,
            prior_log_weight=prior_log_weight,
            state_count=1,
            labels=tuple(0 for _ in baseline_indices),
            mean=one_mean,
            covariance=one_covariance,
            log_evidence=one_evidence,
        )
    ]
    if not config.enable_shared_profile_state:
        return modes

    candidates = []
    assignments = _state_assignments(baseline_indices.size)
    for labels in assignments:
        state_column = -np.asarray(labels, dtype=float)
        if condition_timing_parameters:
            design = np.column_stack(
                (basis[baseline_indices], phase_design, state_column)
            )
            prior_mean = np.asarray(
                (0.0, 0.0, 0.0, 0.0, 0.0, config.state_offset_prior_mean_cycles),
                dtype=float,
            )
            prior_covariance = np.zeros((6, 6), dtype=float)
            prior_covariance[:4, :4] = parameter_prior_covariance
            prior_covariance[4, 4] = (
                float(config.global_phase_prior_sigma_cycles) ** 2
            )
            prior_covariance[5, 5] = (
                float(config.state_offset_prior_sigma_cycles) ** 2
            )
            mean, covariance, evidence = _linear_gaussian_update(
                prior_mean=prior_mean,
                prior_covariance=prior_covariance,
                design=design,
                target=target,
                noise_covariance=noise,
            )
        else:
            design = np.column_stack((phase_design, state_column))
            state_prior_mean = np.asarray(
                (0.0, float(config.state_offset_prior_mean_cycles)), dtype=float
            )
            state_prior_covariance = np.diag(
                (
                    float(config.global_phase_prior_sigma_cycles) ** 2,
                    float(config.state_offset_prior_sigma_cycles) ** 2,
                )
            )
            nuisance_mean, nuisance_covariance, evidence = _linear_gaussian_update(
                prior_mean=state_prior_mean,
                prior_covariance=state_prior_covariance,
                design=design,
                target=target,
                noise_covariance=noise,
            )
            mean = np.concatenate((np.zeros(4, dtype=float), nuisance_mean))
            covariance = np.zeros((6, 6), dtype=float)
            covariance[:4, :4] = parameter_prior_covariance
            covariance[4:, 4:] = nuisance_covariance
        offset = float(mean[-1])
        if any(labels) and not (
            float(config.minimum_state_offset_cycles)
            <= offset
            <= float(config.maximum_state_offset_cycles)
        ):
            continue
        ranking = evidence + _specific_state_log_prior(
            labels,
            alpha=float(config.state_frequency_alpha),
            beta=float(config.state_frequency_beta),
        )
        candidates.append(
            (
                ranking,
                _BaselineStateMode(
                    prior_component_id=prior_component_id,
                    prior_log_weight=prior_log_weight,
                    state_count=2,
                    labels=labels,
                    mean=mean,
                    covariance=covariance,
                    log_evidence=evidence,
                ),
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[: int(config.maximum_baseline_state_modes)]
    if candidates and not any(not any(mode.labels) for _, mode in selected):
        zero_state = next((item for item in candidates if not any(item[1].labels)), None)
        if zero_state is not None:
            selected = [*selected[:-1], zero_state]
    modes.extend(mode for _, mode in selected)
    return modes


def _native_solution(
    *,
    evaluator: Any,
    coordinate_mean: np.ndarray,
    coordinate_covariance: np.ndarray,
    native_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(str(name).upper() for name in evaluator.parameter_names)
    indices = np.asarray(tuple(names.index(name) for name in _NATIVE_PARAMETER_NAMES))
    base = np.asarray(evaluator.base_parameter_values, dtype=float)[indices]
    values = base + coordinate_mean[:4] * native_steps
    jacobian = np.diag(native_steps)
    covariance = jacobian @ coordinate_covariance[:4, :4] @ jacobian.T
    return values, 0.5 * (covariance + covariance.T)


def _previous_parameter_covariance(
    *,
    evaluator: Any,
    native_steps: np.ndarray,
    config: ShortBaselineUpdateConfig,
    sigma_scale: float,
) -> np.ndarray:
    fallback = np.asarray(config.coordinate_prior_sigmas, dtype=float)
    sigmas = fallback.copy()
    model = getattr(evaluator, "model", None)
    if model is not None:
        for index, name in enumerate(_NATIVE_PARAMETER_NAMES):
            try:
                uncertainty = float(model[name].uncertainty_value)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            coordinate_sigma = abs(uncertainty / float(native_steps[index]))
            if np.isfinite(coordinate_sigma) and coordinate_sigma > 0.0:
                sigmas[index] = coordinate_sigma
    sigmas *= float(sigma_scale)
    return np.diag(np.square(sigmas))


def solve_short_baseline_isolated_update(
    *,
    evaluator: Any,
    baseline_session_indices: Sequence[int],
    config: ShortBaselineUpdateConfig | None = None,
) -> ShortBaselineUpdateResult:
    """Update a phase-connected isolated solution with a few unconnected sessions.

    Use historical sessions once to construct a Gaussian timing posterior, then
    enumerate extension-session integer offsets from its predictive covariance.
    A repeated profile state contributes a shared offset, with no time-local
    jump or glitch.
    """

    active = config or ShortBaselineUpdateConfig()
    session_count = len(evaluator.sessions)
    baseline = tuple(sorted({int(value) for value in baseline_session_indices}))
    if not baseline or baseline[0] < 0 or baseline[-1] >= session_count:
        raise ValueError("Baseline sessions must identify a valid nonempty subset.")
    extension = tuple(index for index in range(session_count) if index not in baseline)
    if not extension:
        raise ValueError("A short-baseline update needs at least one extension session.")
    if baseline != tuple(range(len(baseline))):
        raise ValueError("The certified baseline must be a contiguous session prefix.")

    base_values = np.asarray(evaluator.base_parameter_values, dtype=float)
    phase_values = np.asarray(evaluator.raw_phase_residuals(base_values), dtype=float)
    noise_model = RobustPhaseNoiseModel.infer(
        phase_values=phase_values,
        reported_sigmas=evaluator.reported_phase_sigmas(),
        epochs_mjd=evaluator.observation_epochs_mjd,
        sessions=evaluator.sessions,
        config=active.phase_noise,
    )
    intercepts, session_sigmas, basis, native_steps, _local_offsets = (
        _session_phase_basis(evaluator, noise_model)
    )
    declared_parameter_covariance = _previous_parameter_covariance(
        evaluator=evaluator,
        native_steps=native_steps,
        config=active,
        sigma_scale=1.0,
    )
    parameter_prior_covariance = _previous_parameter_covariance(
        evaluator=evaluator,
        native_steps=native_steps,
        config=active,
        sigma_scale=float(active.previous_posterior_sigma_scale),
    )
    variances = np.square(session_sigmas)
    native_prior_jacobian = np.diag(native_steps)
    native_prior_covariance = (
        native_prior_jacobian
        @ parameter_prior_covariance
        @ native_prior_jacobian.T
    )
    declared_native_prior_covariance = (
        native_prior_jacobian
        @ declared_parameter_covariance
        @ native_prior_jacobian.T
    )
    baseline_array = np.asarray(baseline, dtype=int)
    extension_array = np.asarray(extension, dtype=int)
    broad_prior_covariance = np.diag(
        np.square(np.asarray(active.coordinate_prior_sigmas, dtype=float))
    )
    baseline_modes = []
    prior_components = (
        (
            "historical_posterior",
            parameter_prior_covariance,
            math.log(float(active.historical_posterior_probability)),
            False,
        ),
        (
            "broad_reconstruction",
            broad_prior_covariance,
            math.log(float(active.broad_reconstruction_probability)),
            True,
        ),
    )
    for component_id, covariance, log_prior, condition_baseline in prior_components:
        baseline_modes.extend(
            _baseline_modes(
                intercepts=intercepts,
                variances=variances,
                basis=basis,
                baseline_indices=baseline_array,
                parameter_prior_covariance=covariance,
                prior_component_id=component_id,
                prior_log_weight=log_prior,
                condition_timing_parameters=condition_baseline,
                config=active,
            )
        )

    raw_modes: list[_RawUpdateMode] = []
    one_state_log_prior = math.log1p(-float(active.two_state_prior_probability))
    two_state_log_prior = math.log(float(active.two_state_prior_probability))
    for baseline_mode in baseline_modes:
        if baseline_mode.state_count == 1:
            extension_assignments = (tuple(0 for _ in extension),)
            model_log_prior = one_state_log_prior
        else:
            extension_assignments = tuple(product((0, 1), repeat=len(extension)))
            model_log_prior = two_state_log_prior
        for extension_labels in extension_assignments:
            if baseline_mode.state_count == 1:
                design = np.column_stack(
                    (
                        basis[extension_array],
                        -np.ones(extension_array.size, dtype=float),
                    )
                )
                state_log_prior = 0.0
            else:
                design = np.column_stack(
                    (
                        basis[extension_array],
                        -np.ones(extension_array.size, dtype=float),
                        -np.asarray(extension_labels, dtype=float),
                    )
                )
                full_labels = (*baseline_mode.labels, *extension_labels)
                if not any(full_labels):
                    continue
                state_log_prior = _canonical_state_log_prior(
                    full_labels,
                    alpha=float(active.state_frequency_alpha),
                    beta=float(active.state_frequency_beta),
                )
            predictive_mean = intercepts[extension_array] + design @ baseline_mode.mean
            predictive_covariance = (
                np.diag(variances[extension_array])
                + design @ baseline_mode.covariance @ design.T
            )
            integer_result = solve_integer_least_squares(
                predictive_mean,
                predictive_covariance,
                candidate_count=int(active.maximum_integer_candidates),
                max_visited_nodes=int(active.maximum_integer_nodes),
                return_partial_on_budget=True,
            )
            for integer_candidate in integer_result.candidates:
                integers = np.asarray(integer_candidate.integer_vector, dtype=float)
                target = integers - intercepts[extension_array]
                mean, covariance, update_log_evidence = _linear_gaussian_update(
                    prior_mean=baseline_mode.mean,
                    prior_covariance=baseline_mode.covariance,
                    design=design,
                    target=target,
                    noise_covariance=np.diag(variances[extension_array]),
                )
                if baseline_mode.state_count == 2 and not (
                    float(active.minimum_state_offset_cycles)
                    <= float(mean[-1])
                    <= float(active.maximum_state_offset_cycles)
                ):
                    continue
                raw_modes.append(
                    _RawUpdateMode(
                        prior_component_id=baseline_mode.prior_component_id,
                        state_count=baseline_mode.state_count,
                        labels=(*baseline_mode.labels, *extension_labels),
                        integer_offsets=tuple(int(value) for value in integers),
                        mean=mean,
                        covariance=covariance,
                        log_weight=float(
                            model_log_prior
                            + baseline_mode.prior_log_weight
                            + state_log_prior
                            + baseline_mode.log_evidence
                            + update_log_evidence
                        ),
                    )
                )
    if not raw_modes:
        raise RuntimeError("The short-baseline update produced no finite phase path.")

    all_log_weights = np.asarray([mode.log_weight for mode in raw_modes], dtype=float)
    log_evidence = float(logsumexp(all_log_weights))
    all_weights = np.exp(all_log_weights - log_evidence)
    order = np.argsort(all_weights)[::-1]

    path_weights: dict[tuple[int, ...], float] = {}
    model_weights: dict[int, float] = {}
    prior_component_weights: dict[str, float] = {}
    for mode, weight in zip(raw_modes, all_weights, strict=True):
        path_weights[mode.integer_offsets] = (
            path_weights.get(mode.integer_offsets, 0.0) + float(weight)
        )
        model_weights[mode.state_count] = (
            model_weights.get(mode.state_count, 0.0) + float(weight)
        )
        prior_component_weights[mode.prior_component_id] = (
            prior_component_weights.get(mode.prior_component_id, 0.0)
            + float(weight)
        )

    retained_limit = int(active.maximum_output_modes)
    retained_indices = [int(index) for index in order[:retained_limit]]
    for component_id in ("historical_posterior", "broad_reconstruction"):
        component_indices = [
            index
            for index, mode in enumerate(raw_modes)
            if mode.prior_component_id == component_id
        ]
        if not component_indices or any(
            raw_modes[index].prior_component_id == component_id
            for index in retained_indices
        ):
            continue
        best_component_index = max(
            component_indices,
            key=lambda index: all_log_weights[index],
        )
        if retained_indices:
            retained_indices[-1] = best_component_index
        else:
            retained_indices.append(best_component_index)
    retained_indices = sorted(
        set(retained_indices),
        key=lambda index: all_log_weights[index],
        reverse=True,
    )
    retained_probability = float(np.sum(all_weights[retained_indices]))
    modes = []
    for index in retained_indices:
        raw = raw_modes[index]
        native_values, native_covariance = _native_solution(
            evaluator=evaluator,
            coordinate_mean=raw.mean,
            coordinate_covariance=raw.covariance,
            native_steps=native_steps,
        )
        coordinate_names = (
            (*_COORDINATE_NAMES, "PROFILE_OFFSET_CYCLES")
            if raw.state_count == 2
            else _COORDINATE_NAMES
        )
        historical_shift = math.sqrt(
            max(
                float(
                    raw.mean[:4]
                    @ np.linalg.pinv(declared_parameter_covariance)
                    @ raw.mean[:4]
                ),
                0.0,
            )
        )
        modes.append(
            ShortBaselineUpdateMode(
                prior_component_id=raw.prior_component_id,
                state_count=raw.state_count,
                state_labels=raw.labels,
                extension_integer_offsets=raw.integer_offsets,
                coordinate_names=coordinate_names,
                coordinate_mean=raw.mean,
                coordinate_covariance=raw.covariance,
                native_parameter_names=_NATIVE_PARAMETER_NAMES,
                native_parameter_values=native_values,
                native_parameter_covariance=native_covariance,
                profile_offset_cycles=(
                    float(raw.mean[-1]) if raw.state_count == 2 else None
                ),
                profile_offset_sigma_cycles=(
                    math.sqrt(max(float(raw.covariance[-1, -1]), 0.0))
                    if raw.state_count == 2
                    else None
                ),
                historical_posterior_shift_sigma=historical_shift,
                log_weight=raw.log_weight,
                posterior_weight=float(all_weights[index]),
            )
        )

    ranked_paths = tuple(
        sorted(path_weights.items(), key=lambda item: (-item[1], item[0]))
    )
    ranked_models = tuple(
        sorted(model_weights.items(), key=lambda item: (-item[1], item[0]))
    )
    ranked_prior_components = tuple(
        sorted(prior_component_weights.items(), key=lambda item: (-item[1], item[0]))
    )
    dominant_path = float(ranked_paths[0][1])
    dominant_model = float(ranked_models[0][1])
    dominant_prior_component = float(ranked_prior_components[0][1])
    threshold = float(active.phase_connection_probability_threshold)
    unique_phase_connection = dominant_path >= threshold
    conditional_solution_identified = (
        unique_phase_connection and dominant_model >= threshold
    )
    best_historical_shift = float(modes[0].historical_posterior_shift_sigma)
    historical_posterior_consistent = best_historical_shift <= float(
        active.maximum_historical_posterior_shift_sigma
    )
    historical_update_identified = (
        ranked_prior_components[0][0] == "historical_posterior"
        and dominant_prior_component >= threshold
    )
    unique_solution = (
        conditional_solution_identified
        and historical_update_identified
        and historical_posterior_consistent
    )
    if unique_solution:
        status = "unique"
    elif conditional_solution_identified and (
        not historical_posterior_consistent
        or (
            ranked_prior_components[0][0] == "broad_reconstruction"
            and dominant_prior_component >= threshold
        )
    ):
        status = "model_revision_required"
    else:
        status = "data_insufficient"
    epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
    session_epochs = np.asarray(
        [
            float(np.mean(epochs[np.asarray(session.toa_indices, dtype=int)]))
            for session in evaluator.sessions
        ],
        dtype=float,
    )
    return ShortBaselineUpdateResult(
        modes=tuple(modes),
        status=status,
        unique_phase_connection=unique_phase_connection,
        unique_solution=unique_solution,
        historical_posterior_consistent=historical_posterior_consistent,
        best_historical_posterior_shift_sigma=best_historical_shift,
        dominant_integer_path_probability=dominant_path,
        dominant_profile_model_probability=dominant_model,
        dominant_prior_component_probability=dominant_prior_component,
        retained_probability=retained_probability,
        log_evidence=log_evidence,
        baseline_session_indices=baseline,
        extension_session_indices=extension,
        session_epochs_mjd=session_epochs,
        session_intercepts_cycles=intercepts,
        session_sigmas_cycles=session_sigmas,
        prior_native_parameter_covariance=native_prior_covariance,
        declared_historical_native_parameter_covariance=(
            declared_native_prior_covariance
        ),
        phase_path_probabilities=ranked_paths,
        profile_model_probabilities=ranked_models,
        prior_component_probabilities=ranked_prior_components,
    )


def write_short_baseline_update_artifacts(
    *,
    evaluator: Any,
    result: ShortBaselineUpdateResult,
    output_directory: Path,
    maximum_candidates: int = 16,
) -> ShortBaselineUpdateArtifacts:
    """Write a candidate-par portfolio for a short timing update.

    The shared profile-state offset stays marginalized as a nuisance parameter
    and is recorded in the manifest. It is never converted into arbitrary
    per-session JUMPs. Declare a preferred par only when the posterior meets the
    configured uniqueness criterion.
    """

    candidate_limit = int(maximum_candidates)
    if candidate_limit <= 0:
        raise ValueError("At least one short-baseline candidate must be written.")
    if not hasattr(evaluator, "model"):
        raise ValueError("Artifact output requires a PINT-backed evaluator.")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
    candidate_paths: list[Path] = []
    candidate_records: list[dict[str, object]] = []
    selected_modes = list(result.modes[:candidate_limit])
    for component_id, _probability in result.prior_component_probabilities:
        if any(mode.prior_component_id == component_id for mode in selected_modes):
            continue
        representative = next(
            (
                mode
                for mode in result.modes
                if mode.prior_component_id == component_id
            ),
            None,
        )
        if representative is None:
            continue
        if selected_modes:
            selected_modes[-1] = representative
        else:
            selected_modes.append(representative)

    for index, mode in enumerate(selected_modes):
        model = deepcopy(evaluator.model)
        uncertainty_updates = {}
        marginal_sigmas = np.sqrt(
            np.maximum(np.diag(mode.native_parameter_covariance), 0.0)
        )
        for name, value, sigma in zip(
            mode.native_parameter_names,
            mode.native_parameter_values,
            marginal_sigmas,
            strict=True,
        ):
            model[name].value = float(value)
            units = getattr(model[name], "units", None)
            if units is not None and np.isfinite(sigma) and sigma > 0.0:
                uncertainty_updates[name] = float(sigma) * units
        if uncertainty_updates:
            model.set_param_uncertainties(uncertainty_updates)
        metadata = {
            "START": float(np.min(epochs)),
            "FINISH": float(np.max(epochs)),
            "NTOA": int(len(evaluator.toas.table)),
        }
        for name, value in metadata.items():
            if name in model.params:
                model[name].value = value
        for stale_name in ("CHI2R", "TRES"):
            if stale_name in model.params:
                model.remove_param(stale_name)

        path = output / f"candidate_{index:03d}.par"
        write_pint_model_parfile_roundtrip_safe(model=model, path=path)
        candidate_paths.append(path)
        candidate_records.append(
            {
                "rank": index,
                "par_file": path.name,
                "posterior_probability": float(mode.posterior_weight),
                "log_posterior_relative_to_best": float(
                    mode.log_weight - result.best_mode.log_weight
                ),
                "prior_component_id": mode.prior_component_id,
                "extension_integer_offsets": list(mode.extension_integer_offsets),
                "profile_state_count": int(mode.state_count),
                "profile_state_labels": list(mode.state_labels),
                "profile_offset_cycles": mode.profile_offset_cycles,
                "profile_offset_sigma_cycles": mode.profile_offset_sigma_cycles,
                "historical_posterior_shift_sigma": (
                    mode.historical_posterior_shift_sigma
                ),
                "parameters": {
                    name: {
                        "value": float(value),
                        "sigma": float(sigma),
                    }
                    for name, value, sigma in zip(
                        mode.native_parameter_names,
                        mode.native_parameter_values,
                        marginal_sigmas,
                        strict=True,
                    )
                },
                "native_parameter_covariance": (
                    mode.native_parameter_covariance.tolist()
                ),
            }
        )

    preferred = candidate_paths[0] if result.unique_solution else None
    manifest = output / "short_baseline_update.json"
    payload = {
        "status": result.status,
        "unique_phase_connection": bool(result.unique_phase_connection),
        "unique_solution": bool(result.unique_solution),
        "historical_posterior_consistent": bool(
            result.historical_posterior_consistent
        ),
        "best_historical_posterior_shift_sigma": float(
            result.best_historical_posterior_shift_sigma
        ),
        "dominant_integer_path_probability": float(
            result.dominant_integer_path_probability
        ),
        "dominant_profile_model_probability": float(
            result.dominant_profile_model_probability
        ),
        "dominant_prior_component_probability": float(
            result.dominant_prior_component_probability
        ),
        "retained_probability": float(result.retained_probability),
        "preferred_par_file": preferred.name if preferred is not None else None,
        "highest_probability_candidate": (
            candidate_paths[0].name if candidate_paths else None
        ),
        "baseline_session_indices": list(result.baseline_session_indices),
        "extension_session_indices": list(result.extension_session_indices),
        "session_epochs_mjd": result.session_epochs_mjd.tolist(),
        "phase_path_probabilities": [
            {"integer_offsets": list(path), "probability": float(probability)}
            for path, probability in result.phase_path_probabilities
        ],
        "profile_model_probabilities": [
            {"state_count": int(count), "probability": float(probability)}
            for count, probability in result.profile_model_probabilities
        ],
        "prior_component_probabilities": [
            {"component_id": component_id, "probability": float(probability)}
            for component_id, probability in result.prior_component_probabilities
        ],
        "prior_native_parameter_covariance": (
            result.prior_native_parameter_covariance.tolist()
        ),
        "declared_historical_native_parameter_covariance": (
            result.declared_historical_native_parameter_covariance.tolist()
        ),
        "marginal_variance_reduction": dict(result.marginal_variance_reduction),
        "candidate_par_files_are_pint_certified": False,
        "shared_profile_offset_is_written_as_jump": False,
        "candidates": candidate_records,
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ShortBaselineUpdateArtifacts(
        output_directory=output,
        manifest_path=manifest,
        candidate_par_paths=tuple(candidate_paths),
        preferred_par_path=preferred,
    )


__all__ = [
    "ShortBaselineUpdateConfig",
    "ShortBaselineUpdateArtifacts",
    "ShortBaselineUpdateMode",
    "ShortBaselineUpdateResult",
    "solve_short_baseline_isolated_update",
    "write_short_baseline_update_artifacts",
]
