from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Sequence

import numpy as np
from scipy.linalg import helmert
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from timing_accelerator.inference.active_timing_contracts import (
    ActiveTimingBranch,
    ActiveTimingPosterior,
    ObservationRecommendation,
    ObservationWindow,
)


@dataclass(frozen=True)
class ObservationUtilityConfig:
    pairwise_separation_cap: float = 1.0e12
    max_ranked_recommendations: int | None = None
    max_reported_pairs: int = 5


@dataclass(frozen=True)
class BayesianObservationDesignConfig:
    """Numerical controls for posterior-predictive observation design."""

    posterior_samples_per_mode: int = 256
    branch_resolution_probability: float = 0.9
    branch_resolution_log_odds: float = 2.0
    covariance_floor: float = 1.0e-12
    random_seed: int = 1729
    max_reported_pairs: int = 5

    def __post_init__(self) -> None:
        samples = int(self.posterior_samples_per_mode)
        if samples <= 0 or samples & (samples - 1):
            raise ValueError("Posterior samples per mode must be a positive power of two.")
        probability = float(self.branch_resolution_probability)
        if not np.isfinite(probability) or not 0.5 < probability < 1.0:
            raise ValueError("Branch resolution probability must lie between 0.5 and 1.")
        if not np.isfinite(self.branch_resolution_log_odds) or self.branch_resolution_log_odds < 0.0:
            raise ValueError("Branch resolution log odds must be finite and nonnegative.")
        if not np.isfinite(self.covariance_floor) or self.covariance_floor <= 0.0:
            raise ValueError("The predictive covariance floor must be positive.")
        if int(self.max_reported_pairs) < 0:
            raise ValueError("The reported ambiguity-pair limit cannot be negative.")


@dataclass(frozen=True)
class PredictiveObservationMode:
    """One branch's Gaussian predictive density for a future session contrast."""

    mode_id: str
    probability: float
    mean: np.ndarray
    covariance: np.ndarray
    parameter_information_nats: float = 0.0


@dataclass(frozen=True)
class PredictiveObservationScore:
    """Expected information supplied by one proposed observing session."""

    total_information_nats: float
    branch_information_nats: float
    parameter_information_nats: float
    prior_branch_entropy_nats: float
    expected_branch_entropy_nats: float
    expected_branch_resolution_probability: float
    expected_max_branch_probability: float
    effective_branch_count_before: float
    effective_branch_count_after: float
    posterior_sample_count: int
    top_discriminated_pairs: tuple[dict[str, float | str], ...]


def session_contrast_matrix(toa_count: int) -> np.ndarray:
    """Return orthonormal contrasts that marginalize one session phase/JUMP."""

    count = int(toa_count)
    if count < 2:
        raise ValueError("Observation design requires at least two TOAs per session.")
    return np.asarray(helmert(count, full=False), dtype=float)


def session_wrapped_phase_matrix(toa_count: int) -> np.ndarray:
    """Return one common-phase row followed by orthonormal session contrasts.

    The first coordinate is the arithmetic mean phase in cycles and is periodic
    modulo one. The remaining coordinates describe the within-session shape and
    are ordinary linear coordinates.
    """

    count = int(toa_count)
    if count < 2:
        raise ValueError("Observation design requires at least two TOAs per session.")
    return np.vstack(
        (
            np.full((1, count), 1.0 / float(count), dtype=float),
            session_contrast_matrix(count),
        )
    )


def project_session_phase(phase_cycles: Sequence[float]) -> np.ndarray:
    """Project a phase sequence onto the session shape+slope subspace."""

    phase = np.asarray(phase_cycles, dtype=float)
    if phase.ndim != 1 or not np.all(np.isfinite(phase)):
        raise ValueError("Session phases must be one finite vector.")
    return session_contrast_matrix(len(phase)) @ phase


def project_wrapped_session_phase(phase_cycles: Sequence[float]) -> np.ndarray:
    """Project phases onto common wrapped phase plus linear shape coordinates."""

    phase = np.asarray(phase_cycles, dtype=float)
    if phase.ndim != 1 or not np.all(np.isfinite(phase)):
        raise ValueError("Session phases must be one finite vector.")
    projected = session_wrapped_phase_matrix(len(phase)) @ phase
    projected[0] = (projected[0] + 0.5) % 1.0 - 0.5
    return projected


def _regularized_covariance(covariance: np.ndarray, floor: float) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("Predictive covariance must be a nonempty square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Predictive covariance must be finite.")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, float(floor) * scale)
    return (eigenvectors * clipped) @ eigenvectors.T


def gaussian_parameter_information(
    parameter_covariance: np.ndarray,
    projected_phase_design: np.ndarray,
    measurement_covariance: np.ndarray,
    *,
    covariance_floor: float = 1.0e-12,
) -> float:
    """Return linear-Gaussian parameter mutual information in nats."""

    parameters = _regularized_covariance(parameter_covariance, covariance_floor)
    design = np.asarray(projected_phase_design, dtype=float)
    if design.ndim != 2 or design.shape[1] != parameters.shape[0]:
        raise ValueError("The projected design matrix does not align with the covariance.")
    if not np.all(np.isfinite(design)):
        raise ValueError("The projected design matrix must be finite.")
    noise = _regularized_covariance(measurement_covariance, covariance_floor)
    if noise.shape[0] != design.shape[0]:
        raise ValueError("Measurement covariance does not align with the design matrix.")
    predictive = _regularized_covariance(
        noise + design @ parameters @ design.T,
        covariance_floor,
    )
    noise_sign, noise_logdet = np.linalg.slogdet(noise)
    predictive_sign, predictive_logdet = np.linalg.slogdet(predictive)
    if noise_sign <= 0.0 or predictive_sign <= 0.0:
        raise ValueError("Observation-design covariance is not positive definite.")
    return max(float(0.5 * (predictive_logdet - noise_logdet)), 0.0)


def _normal_logpdf(
    samples: np.ndarray,
    mean: np.ndarray,
    cholesky: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(samples, dtype=float) - np.asarray(mean, dtype=float)
    whitened = np.linalg.solve(cholesky, centered.T).T
    dimension = centered.shape[1]
    normalization = dimension * math.log(2.0 * math.pi) + 2.0 * np.log(
        np.diag(cholesky)
    ).sum()
    return -0.5 * (np.einsum("ij,ij->i", whitened, whitened) + normalization)


def _wrapped_normal_logpdf(residual: np.ndarray, variance: float) -> np.ndarray:
    """Evaluate a unit-period wrapped normal density from phase residuals."""

    values = np.asarray(residual, dtype=float)
    active_variance = float(variance)
    if not np.isfinite(active_variance) or active_variance <= 0.0:
        raise ValueError("Wrapped-normal variance must be finite and positive.")
    centered = values - np.rint(values)
    if active_variance < 0.0225:
        sigma = math.sqrt(active_variance)
        radius = max(int(math.ceil(8.0 * sigma)) + 1, 2)
        offsets = np.arange(-radius, radius + 1, dtype=float)
        exponents = -0.5 * np.square(centered[:, None] + offsets[None, :]) / active_variance
        return logsumexp(exponents, axis=1) - 0.5 * math.log(
            2.0 * math.pi * active_variance
        )

    tolerance = 1.0e-14
    maximum_harmonic = max(
        1,
        int(
            math.ceil(
                math.sqrt(-math.log(tolerance) / (2.0 * math.pi**2 * active_variance))
            )
        ),
    )
    harmonics = np.arange(1, maximum_harmonic + 1, dtype=float)
    coefficients = np.exp(-2.0 * math.pi**2 * active_variance * harmonics**2)
    density = 1.0 + 2.0 * np.sum(
        coefficients[None, :]
        * np.cos(2.0 * math.pi * centered[:, None] * harmonics[None, :]),
        axis=1,
    )
    return np.log(np.maximum(density, np.finfo(float).tiny))


def _integer_grid(bounds: np.ndarray) -> np.ndarray:
    axes = [np.arange(-int(bound), int(bound) + 1) for bound in bounds]
    return np.asarray(list(product(*axes)), dtype=float)


def _multivariate_wrapped_normal_logpdf(
    residual: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate a multivariate unit-torus Gaussian by the cheaper dual sum."""

    values = np.asarray(residual, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if values.ndim != 2 or matrix.shape != (values.shape[1], values.shape[1]):
        raise ValueError("Wrapped residuals and covariance do not align.")
    if values.shape[1] == 1:
        return _wrapped_normal_logpdf(values[:, 0], float(matrix[0, 0]))

    centered = values - np.rint(values)
    precision = np.linalg.inv(matrix)
    log_tolerance = -math.log(1.0e-14)
    spatial_bounds = np.maximum(
        np.ceil(np.sqrt(2.0 * log_tolerance * np.diag(matrix))).astype(int) + 1,
        1,
    )
    fourier_bounds = np.maximum(
        np.ceil(
            np.sqrt(
                log_tolerance * np.diag(precision) / (2.0 * math.pi**2)
            )
        ).astype(int),
        1,
    )
    spatial_count = int(np.prod(2 * spatial_bounds + 1))
    fourier_count = int(np.prod(2 * fourier_bounds + 1))
    chunk_size = 4096

    if fourier_count <= spatial_count:
        harmonics = _integer_grid(fourier_bounds)
        quadratic = np.einsum("ij,jk,ik->i", harmonics, matrix, harmonics)
        coefficients = np.exp(-2.0 * math.pi**2 * quadratic)
        keep = coefficients >= 1.0e-14
        harmonics = harmonics[keep]
        coefficients = coefficients[keep]
        density = np.zeros(len(centered), dtype=float)
        for start in range(0, len(harmonics), chunk_size):
            stop = start + chunk_size
            angles = 2.0 * math.pi * centered @ harmonics[start:stop].T
            density += np.cos(angles) @ coefficients[start:stop]
        return np.log(np.maximum(density, np.finfo(float).tiny))

    shifts = _integer_grid(spatial_bounds)
    normalization = 0.5 * (
        values.shape[1] * math.log(2.0 * math.pi)
        + float(np.linalg.slogdet(matrix)[1])
    )
    accumulated = np.full(len(centered), -np.inf, dtype=float)
    for start in range(0, len(shifts), chunk_size):
        stop = start + chunk_size
        displaced = centered[:, None, :] + shifts[None, start:stop, :]
        exponent = -0.5 * np.einsum(
            "nki,ij,nkj->nk",
            displaced,
            precision,
            displaced,
        )
        accumulated = np.logaddexp(
            accumulated,
            logsumexp(exponent, axis=1),
        )
    return accumulated - normalization


def _wrapped_gaussian_logpdf(
    samples: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    covariance_floor: float,
    wrapped_indices: Sequence[int] = (0,),
) -> np.ndarray:
    """Evaluate a Gaussian with selected unit-period coordinates."""

    observations = np.asarray(samples, dtype=float)
    center = np.asarray(mean, dtype=float)
    matrix = _regularized_covariance(covariance, covariance_floor)
    if observations.ndim != 2 or observations.shape[1] != center.size:
        raise ValueError("Wrapped predictive samples do not align with the mean.")
    wrapped = tuple(int(index) for index in wrapped_indices)
    if not wrapped or len(set(wrapped)) != len(wrapped):
        raise ValueError("Wrapped coordinate indices must be unique and nonempty.")
    if min(wrapped) < 0 or max(wrapped) >= center.size:
        raise ValueError("A wrapped coordinate index is outside the predictive vector.")
    linear = tuple(index for index in range(center.size) if index not in wrapped)
    wrapped_mean = center[np.asarray(wrapped)]
    wrapped_covariance = matrix[np.ix_(wrapped, wrapped)]
    if not linear:
        return _multivariate_wrapped_normal_logpdf(
            observations[:, wrapped] - wrapped_mean,
            wrapped_covariance,
        )

    shape_mean = center[np.asarray(linear)]
    shape_covariance = matrix[np.ix_(linear, linear)]
    shape_factor = np.linalg.cholesky(shape_covariance)
    shape_values = observations[:, linear]
    shape_logpdf = _normal_logpdf(shape_values, shape_mean, shape_factor)
    wrapped_shape_covariance = matrix[np.ix_(wrapped, linear)]
    conditional_gain = np.linalg.solve(
        shape_covariance,
        wrapped_shape_covariance.T,
    )
    shape_offset = shape_values - shape_mean
    conditional_mean = wrapped_mean + shape_offset @ conditional_gain
    conditional_covariance = (
        wrapped_covariance - wrapped_shape_covariance @ conditional_gain
    )
    conditional_covariance = _regularized_covariance(
        conditional_covariance,
        covariance_floor,
    )
    return shape_logpdf + _multivariate_wrapped_normal_logpdf(
        observations[:, wrapped] - conditional_mean,
        conditional_covariance,
    )


def _quasi_normal_samples(
    *,
    dimension: int,
    count: int,
    seed: int,
) -> np.ndarray:
    exponent = int(math.log2(count))
    uniform = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)
    return ndtri(np.clip(uniform, np.finfo(float).eps, 1.0 - np.finfo(float).eps))


def _predictive_pair_diagnostics(
    modes: Sequence[PredictiveObservationMode],
    probabilities: np.ndarray,
    covariances: Sequence[np.ndarray],
    *,
    covariance_floor: float,
    maximum_pairs: int,
) -> tuple[dict[str, float | str], ...]:
    pairs: list[dict[str, float | str]] = []
    for left_index, left in enumerate(modes):
        for right_index in range(left_index + 1, len(modes)):
            right = modes[right_index]
            pooled = _regularized_covariance(
                0.5 * (covariances[left_index] + covariances[right_index]),
                covariance_floor,
            )
            difference = np.asarray(left.mean, dtype=float) - np.asarray(right.mean, dtype=float)
            squared_separation = float(difference @ np.linalg.solve(pooled, difference))
            pair_probability = float(probabilities[left_index] * probabilities[right_index])
            pairs.append(
                {
                    "left_mode_id": left.mode_id,
                    "right_mode_id": right.mode_id,
                    "pair_probability": pair_probability,
                    "predictive_separation_sigma": math.sqrt(max(squared_separation, 0.0)),
                    "weighted_separation": pair_probability * squared_separation,
                }
            )
    pairs.sort(key=lambda item: float(item["weighted_separation"]), reverse=True)
    return tuple(pairs[:maximum_pairs])


def _wrapped_predictive_pair_diagnostics(
    modes: Sequence[PredictiveObservationMode],
    probabilities: np.ndarray,
    covariances: Sequence[np.ndarray],
    *,
    covariance_floor: float,
    maximum_pairs: int,
    wrapped_indices: Sequence[int],
) -> tuple[dict[str, float | str], ...]:
    wrapped = tuple(int(index) for index in wrapped_indices)
    pairs: list[dict[str, float | str]] = []
    for left_index, left in enumerate(modes):
        for right_index in range(left_index + 1, len(modes)):
            right = modes[right_index]
            pooled = _regularized_covariance(
                0.5 * (covariances[left_index] + covariances[right_index]),
                covariance_floor,
            )
            precision = np.linalg.inv(pooled)
            difference = np.asarray(left.mean, dtype=float) - np.asarray(
                right.mean, dtype=float
            )
            linear = tuple(index for index in range(difference.size) if index not in wrapped)
            wrapped_precision = precision[np.ix_(wrapped, wrapped)]
            continuous_integer = difference[np.asarray(wrapped)].copy()
            if linear:
                continuous_integer += np.linalg.solve(
                    wrapped_precision,
                    precision[np.ix_(wrapped, linear)] @ difference[np.asarray(linear)],
                )
            nearest = np.rint(continuous_integer).astype(int)
            integer_candidates = product(
                *(range(value - 1, value + 2) for value in nearest)
            )
            squared_separation = min(
                float(displaced @ precision @ displaced)
                for candidate in integer_candidates
                for displaced in (
                    difference
                    - np.asarray(
                        [
                            float(candidate[wrapped.index(index)])
                            if index in wrapped
                            else 0.0
                            for index in range(difference.size)
                        ]
                    ),
                )
            )
            pair_probability = float(
                probabilities[left_index] * probabilities[right_index]
            )
            pairs.append(
                {
                    "left_mode_id": left.mode_id,
                    "right_mode_id": right.mode_id,
                    "pair_probability": pair_probability,
                    "predictive_separation_sigma": math.sqrt(
                        max(squared_separation, 0.0)
                    ),
                    "weighted_separation": pair_probability * squared_separation,
                }
            )
    pairs.sort(key=lambda item: float(item["weighted_separation"]), reverse=True)
    return tuple(pairs[:maximum_pairs])


def _score_predictive_observation(
    modes: Sequence[PredictiveObservationMode],
    config: BayesianObservationDesignConfig | None,
    *,
    wrapped_indices: Sequence[int],
) -> PredictiveObservationScore:
    active = config or BayesianObservationDesignConfig()
    portfolio = tuple(modes)
    if not portfolio:
        raise ValueError("Observation design requires at least one posterior mode.")
    dimension = np.asarray(portfolio[0].mean, dtype=float).size
    if dimension <= 0:
        raise ValueError("Predictive session coordinates cannot be empty.")
    raw_probabilities = np.asarray([mode.probability for mode in portfolio], dtype=float)
    if np.any(~np.isfinite(raw_probabilities)) or np.any(raw_probabilities < 0.0):
        raise ValueError("Posterior mode probabilities must be finite and nonnegative.")
    probability_sum = float(raw_probabilities.sum())
    if probability_sum <= 0.0:
        raise ValueError("At least one posterior mode must have positive probability.")
    probabilities = raw_probabilities / probability_sum
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    cholesky_factors: list[np.ndarray] = []
    for mode in portfolio:
        mean = np.asarray(mode.mean, dtype=float)
        if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("Every predictive mode mean must align and be finite.")
        covariance = _regularized_covariance(mode.covariance, active.covariance_floor)
        if covariance.shape != (dimension, dimension):
            raise ValueError("Every predictive mode covariance must align with its mean.")
        means.append(mean)
        covariances.append(covariance)
        cholesky_factors.append(np.linalg.cholesky(covariance))

    positive = probabilities > 0.0
    prior_entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    log_probabilities = np.full(len(portfolio), -np.inf, dtype=float)
    log_probabilities[positive] = np.log(probabilities[positive])
    expected_entropy = 0.0
    expected_max_probability = 0.0
    expected_resolution = 0.0
    total_samples = 0
    for source_index, source_probability in enumerate(probabilities):
        if source_probability <= 0.0:
            continue
        standard = _quasi_normal_samples(
            dimension=dimension,
            count=active.posterior_samples_per_mode,
            seed=int(active.random_seed) + source_index,
        )
        samples = means[source_index] + standard @ cholesky_factors[source_index].T
        if wrapped_indices:
            wrapped = np.asarray(tuple(wrapped_indices), dtype=int)
            samples[:, wrapped] = (samples[:, wrapped] + 0.5) % 1.0 - 0.5
            log_likelihoods = np.column_stack(
                [
                    _wrapped_gaussian_logpdf(
                        samples,
                        mean,
                        covariance,
                        covariance_floor=active.covariance_floor,
                        wrapped_indices=wrapped_indices,
                    )
                    for mean, covariance in zip(means, covariances, strict=True)
                ]
            )
        else:
            log_likelihoods = np.column_stack(
                [
                    _normal_logpdf(samples, mean, factor)
                    for mean, factor in zip(means, cholesky_factors, strict=True)
                ]
            )
        log_joint = log_likelihoods + log_probabilities[None, :]
        log_normalizer = logsumexp(log_joint, axis=1)
        posterior = np.exp(log_joint - log_normalizer[:, None])
        sample_entropy = -np.sum(
            np.where(posterior > 0.0, posterior * np.log(posterior), 0.0),
            axis=1,
        )
        maximum = np.max(posterior, axis=1)
        if len(portfolio) == 1:
            log_odds = np.full(len(samples), np.inf, dtype=float)
        else:
            ordered = np.sort(posterior, axis=1)
            log_odds = np.log(
                np.maximum(ordered[:, -1], np.finfo(float).tiny)
                / np.maximum(ordered[:, -2], np.finfo(float).tiny)
            )
        resolved = (maximum >= active.branch_resolution_probability) & (
            log_odds >= active.branch_resolution_log_odds
        )
        expected_entropy += float(source_probability * np.mean(sample_entropy))
        expected_max_probability += float(source_probability * np.mean(maximum))
        expected_resolution += float(source_probability * np.mean(resolved))
        total_samples += len(samples)

    branch_information = max(prior_entropy - expected_entropy, 0.0)
    parameter_information = float(
        sum(
            probability * max(float(mode.parameter_information_nats), 0.0)
            for probability, mode in zip(probabilities, portfolio, strict=True)
        )
    )
    pair_diagnostics = (
        _wrapped_predictive_pair_diagnostics
        if wrapped_indices
        else _predictive_pair_diagnostics
    )
    return PredictiveObservationScore(
        total_information_nats=float(branch_information + parameter_information),
        branch_information_nats=float(branch_information),
        parameter_information_nats=parameter_information,
        prior_branch_entropy_nats=prior_entropy,
        expected_branch_entropy_nats=float(expected_entropy),
        expected_branch_resolution_probability=float(expected_resolution),
        expected_max_branch_probability=float(expected_max_probability),
        effective_branch_count_before=float(math.exp(prior_entropy)),
        effective_branch_count_after=float(math.exp(expected_entropy)),
        posterior_sample_count=total_samples,
        top_discriminated_pairs=(
            pair_diagnostics(
                portfolio,
                probabilities,
                covariances,
                covariance_floor=active.covariance_floor,
                maximum_pairs=active.max_reported_pairs,
                wrapped_indices=wrapped_indices,
            )
            if wrapped_indices
            else pair_diagnostics(
                portfolio,
                probabilities,
                covariances,
                covariance_floor=active.covariance_floor,
                maximum_pairs=active.max_reported_pairs,
            )
        ),
    )


def score_predictive_observation(
    modes: Sequence[PredictiveObservationMode],
    config: BayesianObservationDesignConfig | None = None,
) -> PredictiveObservationScore:
    """Score a session from its branch-conditional posterior predictives.

    The input means and covariances must already live in the orthonormal
    session-contrast space, so an arbitrary integer phase or JUMP is absent.
    """

    return _score_predictive_observation(
        modes,
        config,
        wrapped_indices=(),
    )


def score_wrapped_predictive_observation(
    modes: Sequence[PredictiveObservationMode],
    config: BayesianObservationDesignConfig | None = None,
) -> PredictiveObservationScore:
    """Score common fractional phase and within-session shape jointly.

    The first predictive coordinate is interpreted modulo one cycle. Integer
    pulse offsets are therefore marginalized while fractional phase evidence is
    retained. Remaining coordinates are ordinary Gaussian session contrasts.
    """

    return _score_predictive_observation(
        modes,
        config,
        wrapped_indices=(0,),
    )


def score_multi_wrapped_predictive_observation(
    modes: Sequence[PredictiveObservationMode],
    wrapped_indices: Sequence[int],
    config: BayesianObservationDesignConfig | None = None,
) -> PredictiveObservationScore:
    """Score a joint predictive with one wrapped coordinate per session."""

    indices = tuple(int(index) for index in wrapped_indices)
    if not indices:
        raise ValueError("Joint wrapped observation design requires wrapped coordinates.")
    return _score_predictive_observation(
        modes,
        config,
        wrapped_indices=indices,
    )


def _branch_offset_map(branches: list[ActiveTimingBranch]) -> dict[str, np.ndarray]:
    return {
        branch.branch_id: np.asarray(branch.theta_offset, dtype=float)
        for branch in branches
    }


def _phase_predictions(
    offsets_by_id: dict[str, np.ndarray],
    design_row: np.ndarray,
) -> dict[str, float]:
    design = np.asarray(design_row, dtype=float)
    return {
        branch_id: float(design @ offset)
        for branch_id, offset in offsets_by_id.items()
    }


def _pair_scores(
    phase_predictions: dict[str, float],
    probabilities: dict[str, float],
    toa_error_phase: float,
    config: ObservationUtilityConfig,
) -> tuple[float, float, list[dict[str, float | str]]]:
    branch_ids = list(phase_predictions)
    sigma = max(float(toa_error_phase), 1.0e-12)
    utility = 0.0
    weighted_linear_separation = 0.0
    pairs: list[dict[str, float | str]] = []
    for left_index, left_id in enumerate(branch_ids):
        for right_id in branch_ids[left_index + 1 :]:
            pair_weight = float(probabilities.get(left_id, 0.0)) * float(probabilities.get(right_id, 0.0))
            if pair_weight <= 0.0:
                continue
            separation_sigma = abs(float(phase_predictions[left_id]) - float(phase_predictions[right_id])) / sigma
            capped_quadratic = min(separation_sigma * separation_sigma, float(config.pairwise_separation_cap))
            contribution = pair_weight * capped_quadratic
            utility += contribution
            weighted_linear_separation += pair_weight * separation_sigma
            pairs.append(
                {
                    "left_branch_id": left_id,
                    "right_branch_id": right_id,
                    "pair_probability": float(pair_weight),
                    "phase_separation_sigma": float(separation_sigma),
                    "utility_contribution": float(contribution),
                }
            )
    pairs = sorted(pairs, key=lambda item: float(item["utility_contribution"]), reverse=True)[
        : max(int(config.max_reported_pairs), 0)
    ]
    return float(utility), float(weighted_linear_separation), pairs


def _entropy_drop_proxy(utility: float) -> float:
    return float(1.0 - math.exp(-max(float(utility), 0.0)))


def rank_observation_windows(
    branches: list[ActiveTimingBranch],
    posterior: ActiveTimingPosterior,
    windows: list[ObservationWindow],
    design_by_epoch: dict[float, np.ndarray],
    toa_error_phase: float,
    config: ObservationUtilityConfig | None = None,
) -> list[ObservationRecommendation]:
    active_config = config or ObservationUtilityConfig()
    offsets_by_id = _branch_offset_map(branches)
    recommendations: list[ObservationRecommendation] = []
    for window in windows:
        epoch = float(window.epoch_mjd)
        if epoch not in design_by_epoch:
            continue
        phase_predictions = _phase_predictions(offsets_by_id, np.asarray(design_by_epoch[epoch], dtype=float))
        utility, weighted_linear_separation, pairs = _pair_scores(
            phase_predictions=phase_predictions,
            probabilities=posterior.branch_probabilities,
            toa_error_phase=toa_error_phase,
            config=active_config,
        )
        utility = max(float(utility) - float(window.cost), 0.0)
        recommendations.append(
            ObservationRecommendation(
                rank=0,
                window=window,
                utility=utility,
                expected_entropy_drop=_entropy_drop_proxy(utility),
                expected_log_bayes_gain=utility,
                phase_separation_sigma=weighted_linear_separation,
                top_discriminated_pairs=pairs,
                phase_predictions=phase_predictions,
            )
        )

    ranked = sorted(
        recommendations,
        key=lambda item: (-float(item.utility), float(item.window.epoch_mjd)),
    )
    if active_config.max_ranked_recommendations is not None:
        ranked = ranked[: max(int(active_config.max_ranked_recommendations), 0)]
    return [
        ObservationRecommendation(
            rank=index,
            window=item.window,
            utility=item.utility,
            expected_entropy_drop=item.expected_entropy_drop,
            expected_log_bayes_gain=item.expected_log_bayes_gain,
            phase_separation_sigma=item.phase_separation_sigma,
            top_discriminated_pairs=item.top_discriminated_pairs,
            phase_predictions=item.phase_predictions,
        )
        for index, item in enumerate(ranked, start=1)
    ]
