from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import expit, logit, logsumexp

from apollo_timing.inference.bayesian_noise_annealed_orbit_transport import (
    NoiseAnnealedOrbitSeed,
)


@dataclass(frozen=True)
class GlobalNoiseSMCConfig:
    """Controls for global continuous posterior transport before path recovery."""

    phase_noise_ladder: tuple[float, ...] = (
        0.1,
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0,
    )
    initial_phase_noise: float | None = None
    particle_count: int = 256
    broad_proposal_weight: float = 0.35
    chart_scale_factors: tuple[float, ...] = (1.0, 4.0, 16.0)
    effective_sample_size_target_fraction: float = 0.75
    resample_threshold_fraction: float = 0.85
    maximum_bridge_steps: int = 128
    mutation_steps: int = 2
    mutation_fraction: float = 1.0
    local_proposal_scale: float = 0.8
    local_proposal_scale_factors: tuple[float, ...] = (0.01, 0.1, 1.0)
    directional_slice_probability: float = 0.35
    directional_slice_axis_probability: float = 0.25
    maximum_slice_shrink_steps: int = 48
    minimum_slice_interval: float = 1.0e-14
    differential_evolution_probability: float = 0.25
    independence_proposal_probability: float = 0.15
    proposal_jitter: float = 1.0e-3
    covariance_eigenvalue_floor: float = 1.0e-10
    minimum_unique_particle_fraction: float = 0.5
    maximum_rejuvenation_rounds: int = 4
    portfolio_mode_count: int = 64
    minimum_mode_separation: float = 5.0e-3
    maximum_archive_size: int = 100_000
    retain_seed_centers_for_mode_discovery: bool = False
    random_seed: int = 20260725


@dataclass(frozen=True)
class GlobalNoiseSMCInformationTarget:
    """One normalized posterior target in an information path."""

    kind: str
    phase_noise: float | None = None
    partition_index: int | None = None


@dataclass(frozen=True)
class GlobalNoiseSMCInformationPath:
    """A complete, independently sampled factor ordering."""

    path_id: str
    targets: tuple[GlobalNoiseSMCInformationTarget, ...]


@dataclass(frozen=True)
class GlobalNoiseSMCPosteriorPopulation:
    """Weighted particles already distributed as one declared source target."""

    particles: np.ndarray
    log_weights: np.ndarray
    source_target: GlobalNoiseSMCInformationTarget


@dataclass(frozen=True)
class GlobalNoiseSMCArchivePoint:
    point_id: str
    source: str
    factor_index: int
    bridge_fraction: float
    orbit_normalized_values: np.ndarray
    curvature_log_likelihood: float
    slope_log_likelihood: float
    shape_log_likelihood: float
    partition_integer_log_likelihoods: np.ndarray
    island_integer_log_likelihoods: np.ndarray
    integer_log_likelihoods: np.ndarray

    @property
    def exact_log_likelihood(self) -> float:
        return float(
            self.shape_log_likelihood + self.integer_log_likelihoods[-1]
        )


@dataclass(frozen=True)
class GlobalNoiseSMCMode:
    mode_id: str
    source_point_id: str
    source: str
    factor_index: int
    bridge_fraction: float
    orbit_normalized_values: np.ndarray
    orbit_covariance: np.ndarray
    curvature_log_likelihood: float
    slope_log_likelihood: float
    shape_log_likelihood: float
    partition_integer_log_likelihoods: np.ndarray
    island_integer_log_likelihoods: np.ndarray
    integer_log_likelihoods: np.ndarray

    @property
    def exact_log_likelihood(self) -> float:
        return float(
            self.shape_log_likelihood + self.integer_log_likelihoods[-1]
        )


@dataclass(frozen=True)
class GlobalNoiseSMCStage:
    step_index: int
    factor_index: int
    factor: str
    source_phase_noise: float | None
    target_phase_noise: float | None
    bridge_fraction: float
    delta_bridge_fraction: float
    effective_sample_size: float
    mode_effective_sample_size: float
    unique_particle_count: int
    diversity_satisfied: bool
    resampled: bool
    rejuvenation_round_count: int
    mutation_attempt_count: int
    mutation_accept_count: int
    mutation_acceptance_fraction: float
    best_target_log_likelihood: float
    best_exact_log_likelihood: float
    finite_particle_count: int


@dataclass(frozen=True)
class GlobalNoiseSMCResult:
    phase_noise_ladder: tuple[float, ...]
    particles: np.ndarray
    log_weights: np.ndarray
    curvature_log_likelihoods: np.ndarray
    slope_log_likelihoods: np.ndarray
    shape_log_likelihoods: np.ndarray
    partition_integer_log_likelihoods: np.ndarray
    island_integer_log_likelihoods: np.ndarray
    integer_log_likelihoods: np.ndarray
    portfolio_modes: tuple[GlobalNoiseSMCMode, ...]
    archive: tuple[GlobalNoiseSMCArchivePoint, ...]
    stage_history: tuple[GlobalNoiseSMCStage, ...]
    likelihood_call_count: int
    reached_exact_endpoint: bool


@dataclass(frozen=True)
class GlobalNoiseSMCPortfolioStage:
    path_index: int
    path_id: str
    initial_phase_noise: float | None
    stage: GlobalNoiseSMCStage


@dataclass(frozen=True)
class GlobalNoiseSMCPathResult:
    path_index: int
    path_id: str
    initial_phase_noise: float | None
    information_path: GlobalNoiseSMCInformationPath
    result: GlobalNoiseSMCResult


@dataclass(frozen=True)
class GlobalNoiseSMCPortfolioResult:
    information_paths: tuple[float | None, ...]
    path_specs: tuple[GlobalNoiseSMCInformationPath, ...]
    path_results: tuple[GlobalNoiseSMCPathResult, ...]
    portfolio_modes: tuple[GlobalNoiseSMCMode, ...]
    archive: tuple[GlobalNoiseSMCArchivePoint, ...]
    likelihood_call_count: int
    reached_exact_endpoint: bool


def _validated_noise_ladder(values: Sequence[float]) -> tuple[float, ...]:
    ladder = tuple(float(value) for value in values)
    if not ladder or not all(
        np.isfinite(value) and value >= 0.0 for value in ladder
    ):
        raise ValueError("The global phase-noise ladder must be finite and non-negative.")
    if ladder[-1] != 0.0:
        raise ValueError("The global phase-noise ladder must end at zero.")
    if any(left <= right for left, right in zip(ladder, ladder[1:])):
        raise ValueError("The global phase-noise ladder must decrease toward zero.")
    return ladder


def _normalize_log_weights(values: Sequence[float]) -> np.ndarray:
    log_weights = np.asarray(values, dtype=float)
    normalizer = float(logsumexp(log_weights))
    if not np.isfinite(normalizer):
        raise RuntimeError("Global SMC particle weights lost all finite support.")
    return log_weights - normalizer


def _effective_sample_size(log_weights: Sequence[float]) -> float:
    normalized = _normalize_log_weights(log_weights)
    weights = np.exp(normalized)
    return float(1.0 / np.sum(np.square(weights)))


def _mode_effective_sample_size(
    log_weights: Sequence[float],
    particles: np.ndarray,
) -> float:
    """Compute ESS after exact duplicate particles are treated as one mode."""

    group_indices = np.empty(len(particles), dtype=int)
    grouped: dict[bytes, int] = {}
    for index, point in enumerate(particles):
        key = np.round(np.asarray(point, dtype=np.float64), 13).tobytes()
        if key not in grouped:
            grouped[key] = len(grouped)
        group_indices[index] = grouped[key]
    return _grouped_mode_effective_sample_size(
        log_weights,
        group_indices=group_indices,
        group_count=len(grouped),
    )


def _grouped_mode_effective_sample_size(
    log_weights: Sequence[float],
    *,
    group_indices: np.ndarray,
    group_count: int,
) -> float:
    weights = np.exp(_normalize_log_weights(log_weights))
    mode_weights = np.bincount(
        group_indices,
        weights=weights,
        minlength=int(group_count),
    )
    return float(1.0 / np.sum(np.square(mode_weights)))


def _unique_particle_count(particles: np.ndarray) -> int:
    return len(
        {
            np.round(np.asarray(point, dtype=np.float64), 13).tobytes()
            for point in particles
        }
    )


def _systematic_resample(
    log_weights: Sequence[float],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = np.exp(_normalize_log_weights(log_weights))
    positions = (rng.random() + np.arange(weights.size, dtype=float)) / weights.size
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _weighted_covariance(
    particles: np.ndarray,
    log_weights: Sequence[float],
) -> np.ndarray:
    values = np.asarray(particles, dtype=float)
    weights = np.exp(_normalize_log_weights(log_weights))
    mean = weights @ values
    centered = values - mean
    denominator = 1.0 - float(np.sum(np.square(weights)))
    if denominator <= np.finfo(float).eps:
        return np.cov(values, rowvar=False)
    return (centered * weights[:, None]).T @ centered / denominator


def _regularized_group_covariances(
    particles: np.ndarray,
    log_weights: Sequence[float],
    *,
    group_indices: np.ndarray,
    group_count: int,
) -> np.ndarray:
    """Estimate family-local geometry without trusting small rank-deficient groups."""

    values = np.asarray(particles, dtype=float)
    labels = np.asarray(group_indices, dtype=int)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("Global SMC proposal-family labels have the wrong shape.")
    dimension = values.shape[1]
    global_covariance = np.asarray(
        _weighted_covariance(values, log_weights),
        dtype=float,
    )
    if dimension == 1:
        global_covariance = np.asarray(
            [[float(np.ravel(global_covariance)[0])]],
            dtype=float,
        )
    global_covariance = 0.5 * (global_covariance + global_covariance.T)
    covariances = np.repeat(
        global_covariance[None, :, :],
        max(int(group_count), 1),
        axis=0,
    )
    for group_index in range(covariances.shape[0]):
        selected = np.flatnonzero(labels == group_index)
        selected_log_weights = np.asarray(log_weights, dtype=float)[selected]
        finite = np.isfinite(selected_log_weights)
        selected = selected[finite]
        selected_log_weights = selected_log_weights[finite]
        if selected.size < 2:
            continue
        group_log_weights = _normalize_log_weights(selected_log_weights)
        group_covariance = np.asarray(
            _weighted_covariance(values[selected], group_log_weights),
            dtype=float,
        )
        if dimension == 1:
            group_covariance = np.asarray(
                [[float(np.ravel(group_covariance)[0])]],
                dtype=float,
            )
        if not np.all(np.isfinite(group_covariance)):
            continue
        group_covariance = 0.5 * (group_covariance + group_covariance.T)
        group_ess = _effective_sample_size(group_log_weights)
        local_fraction = min(
            max((group_ess - 1.0) / (group_ess + dimension), 0.0),
            1.0,
        )
        covariances[group_index] = (
            local_fraction * group_covariance
            + (1.0 - local_fraction) * global_covariance
        )
    return covariances


def _reflect_unit(values: Sequence[float]) -> np.ndarray:
    point = np.mod(np.asarray(values, dtype=float), 2.0)
    return np.where(point <= 1.0, point, 2.0 - point)


class _BoundedOrbitProposal:
    def __init__(
        self,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        seeds: Sequence[NoiseAnnealedOrbitSeed],
        broad_weight: float,
        chart_scale_factors: Sequence[float],
        covariance_floor: float,
    ) -> None:
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.width = self.upper - self.lower
        self.dimension = int(self.lower.size)
        self.broad_weight = min(max(float(broad_weight), 1.0e-6), 1.0)
        scales = tuple(
            float(value)
            for value in chart_scale_factors
            if np.isfinite(float(value)) and float(value) > 0.0
        )
        centers = []
        roots = []
        inverse_covariances = []
        log_determinants = []
        floor = max(float(covariance_floor), np.finfo(float).eps)
        chart_family_indices = []
        for family_index, seed in enumerate(seeds):
            center = np.asarray(seed.orbit_normalized_values, dtype=float)
            covariance = np.asarray(seed.orbit_covariance, dtype=float)
            if center.shape != self.lower.shape or covariance.shape != (
                self.dimension,
                self.dimension,
            ):
                raise ValueError("A global orbit chart has incompatible geometry.")
            probability = np.clip(
                (center - self.lower) / self.width,
                1.0e-8,
                1.0 - 1.0e-8,
            )
            logit_center = logit(probability)
            jacobian = np.diag(1.0 / (self.width * probability * (1.0 - probability)))
            logit_covariance = jacobian @ covariance @ jacobian.T
            logit_covariance = 0.5 * (logit_covariance + logit_covariance.T)
            eigenvalues, eigenvectors = np.linalg.eigh(logit_covariance)
            logit_covariance = (
                eigenvectors
                @ np.diag(np.maximum(eigenvalues, floor))
                @ eigenvectors.T
            )
            for scale in scales:
                scaled_covariance = np.square(scale) * logit_covariance
                centers.append(logit_center.copy())
                roots.append(np.linalg.cholesky(scaled_covariance))
                inverse_covariances.append(np.linalg.inv(scaled_covariance))
                log_determinants.append(float(np.linalg.slogdet(scaled_covariance)[1]))
                chart_family_indices.append(family_index)
        self.logit_centers = np.asarray(centers, dtype=float)
        self.roots = np.asarray(roots, dtype=float)
        self.inverse_covariances = np.asarray(inverse_covariances, dtype=float)
        self.log_determinants = np.asarray(log_determinants, dtype=float)
        self.chart_family_indices = np.asarray(chart_family_indices, dtype=int)
        if self.logit_centers.size == 0:
            self.logit_centers = np.empty((0, self.dimension), dtype=float)
            self.roots = np.empty((0, self.dimension, self.dimension), dtype=float)
            self.inverse_covariances = np.empty_like(self.roots)
            self.log_determinants = np.empty(0, dtype=float)
            self.chart_family_indices = np.empty(0, dtype=int)
            self.broad_weight = 1.0
        self.family_count = len(seeds) if self.chart_count else 0
        self.family_component_indices = tuple(
            np.flatnonzero(self.chart_family_indices == family_index)
            for family_index in range(self.family_count)
        )
        if self.family_count:
            chart_family_weight = (1.0 - self.broad_weight) / self.family_count
            self.nominal_group_weights = np.asarray(
                [self.broad_weight]
                + [chart_family_weight] * self.family_count,
                dtype=float,
            )
        else:
            self.nominal_group_weights = np.asarray([1.0], dtype=float)
        self.group_weights = self.nominal_group_weights.copy()

    @property
    def chart_count(self) -> int:
        return int(self.logit_centers.shape[0])

    @property
    def group_count(self) -> int:
        return 1 + self.family_count

    def _set_group_weights(self, values: Sequence[float]) -> None:
        weights = np.asarray(values, dtype=float)
        if (
            weights.shape != (self.group_count,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not float(np.sum(weights)) > 0.0
        ):
            raise ValueError("Global proposal group weights are invalid.")
        self.group_weights = weights / np.sum(weights)

    def stratified_group_indices(
        self,
        count: int,
        *,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Allocate deterministic-mixture draws across broad and seed families."""

        size = int(count)
        if size < self.group_count:
            raise ValueError(
                "Stratified global initialization needs at least one particle "
                "per proposal family."
            )
        desired = size * self.nominal_group_weights
        quotas = np.floor(desired).astype(int)
        remainder = size - int(np.sum(quotas))
        order = np.argsort(-(desired - quotas), kind="stable")
        quotas[order[:remainder]] += 1

        for empty_group in np.flatnonzero(quotas == 0):
            donors = np.flatnonzero(quotas > 1)
            if donors.size == 0:
                raise RuntimeError("Cannot construct stratified proposal quotas.")
            donor = int(donors[np.argmax(quotas[donors] - desired[donors])])
            quotas[donor] -= 1
            quotas[empty_group] = 1

        self._set_group_weights(quotas / size)
        groups = np.repeat(np.arange(self.group_count, dtype=int), quotas)
        rng.shuffle(groups)
        return groups

    def draw_group_unit(
        self,
        group_index: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str]:
        group = int(group_index)
        if group == 0:
            return rng.random(self.dimension), "global-uniform"
        if group < 0 or group > self.family_count:
            raise ValueError("Global proposal group index is out of range.")
        family_index = group - 1
        components = self.family_component_indices[family_index]
        chart_index = int(components[int(rng.integers(components.size))])
        latent = self.logit_centers[chart_index] + self.roots[chart_index] @ rng.normal(
            size=self.dimension
        )
        return (
            expit(latent),
            f"chart-family-{family_index:04d}-component-{chart_index:04d}",
        )

    def draw_unit(
        self,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str]:
        group_index = int(rng.choice(self.group_count, p=self.group_weights))
        return self.draw_group_unit(group_index, rng=rng)

    def log_density_unit(self, unit_values: Sequence[float]) -> float:
        unit = np.clip(np.asarray(unit_values, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
        components = []
        broad_weight = float(self.group_weights[0])
        if broad_weight > 0.0:
            components.append(math.log(broad_weight))
        for family_index in range(self.family_count):
            family_weight = float(self.group_weights[family_index + 1])
            if family_weight <= 0.0:
                continue
            components.append(
                math.log(family_weight)
                + self.log_density_group_unit(unit, family_index + 1)
            )
        return float(logsumexp(np.asarray(components, dtype=float)))

    def log_density_group_unit(
        self,
        unit_values: Sequence[float],
        group_index: int,
    ) -> float:
        group = int(group_index)
        if group == 0:
            return 0.0
        if group < 0 or group > self.family_count:
            raise ValueError("Global proposal group index is out of range.")
        unit = np.clip(np.asarray(unit_values, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
        latent = logit(unit)
        log_jacobian = float(-np.sum(np.log(unit) + np.log1p(-unit)))
        normalizer = self.dimension * math.log(2.0 * math.pi)
        family_index = group - 1
        chart_scores = []
        for chart_index in self.family_component_indices[family_index]:
            displacement = latent - self.logit_centers[chart_index]
            quadratic = float(
                displacement
                @ self.inverse_covariances[chart_index]
                @ displacement
            )
            chart_scores.append(
                -0.5
                * (
                    normalizer
                    + self.log_determinants[chart_index]
                    + quadratic
                )
                + log_jacobian
            )
        return float(logsumexp(chart_scores) - math.log(len(chart_scores)))

    def to_orbit(self, unit_values: Sequence[float]) -> np.ndarray:
        return self.lower + np.asarray(unit_values, dtype=float) * self.width

    def to_unit(self, orbit_values: Sequence[float]) -> np.ndarray:
        return (np.asarray(orbit_values, dtype=float) - self.lower) / self.width

    def most_likely_group(self, unit_values: Sequence[float]) -> int:
        """Assign an existing particle to a proposal family for local moves."""

        scores = np.full(self.group_count, -math.inf, dtype=float)
        for group_index, weight in enumerate(self.group_weights):
            if float(weight) <= 0.0:
                continue
            scores[group_index] = math.log(float(weight)) + self.log_density_group_unit(
                unit_values,
                group_index,
            )
        return int(np.argmax(scores))


class _GlobalEvaluator:
    def __init__(self, likelihood: Any, ladder: tuple[float, ...]) -> None:
        self.likelihood = likelihood
        self.ladder = ladder
        hierarchy = getattr(likelihood, "session_partition_hierarchy", None)
        self.partition_count = max(len(hierarchy) if hierarchy is not None else 1, 1)
        self.cache: dict[
            bytes,
            tuple[float, float, np.ndarray, np.ndarray],
        ] = {}
        self.call_count = 0

    @staticmethod
    def _key(values: Sequence[float]) -> bytes:
        return np.round(np.asarray(values, dtype=np.float64), 13).tobytes()

    def evaluate(
        self,
        values: Sequence[float],
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        key = self._key(values)
        if key not in self.cache:
            try:
                partition_evaluator = getattr(
                    self.likelihood,
                    "evaluate_global_and_partition_noise_channels",
                    None,
                )
                joint_evaluator = getattr(
                    self.likelihood,
                    "evaluate_global_and_island_noise_channels",
                    None,
                )
                if partition_evaluator is not None:
                    evaluations, partition_evaluations = partition_evaluator(
                        values,
                        self.ladder,
                    )
                elif joint_evaluator is None:
                    evaluations = self.likelihood.evaluate_noise_channels(
                        values,
                        self.ladder,
                    )
                    partition_evaluations = (evaluations,)
                else:
                    evaluations, island_evaluations = joint_evaluator(
                        values,
                        self.ladder,
                    )
                    partition_evaluations = (island_evaluations,)
                if len(evaluations) != len(self.ladder) or not partition_evaluations:
                    raise ValueError("Noise-channel evaluation has the wrong size.")
                shape = float(evaluations[-1].shape_log_likelihood)
                curvature = float(
                    getattr(
                        evaluations[-1],
                        "curvature_log_likelihood",
                        shape,
                    )
                )
                slope = float(
                    getattr(
                        evaluations[-1],
                        "slope_log_likelihood",
                        shape - curvature,
                    )
                )
                integer = np.asarray(
                    [item.integer_log_likelihood for item in evaluations],
                    dtype=float,
                )
                partition_integer = np.asarray(
                    [
                        [
                            item.integer_log_likelihood
                            for item in partition
                        ]
                        for partition in partition_evaluations
                    ],
                    dtype=float,
                )
                if integer.shape != (len(self.ladder),) or (
                    partition_integer.ndim != 2
                    or partition_integer.shape[1] != len(self.ladder)
                    or partition_integer.shape[0] != self.partition_count
                ):
                    raise ValueError("Noise-channel evaluation has the wrong size.")
                partition_shapes = tuple(
                    float(partition[-1].shape_log_likelihood)
                    for partition in partition_evaluations
                )
                phase_channels_valid = bool(
                    np.all(~np.isnan(partition_integer))
                    and np.all(~np.isposinf(partition_integer))
                    and np.all(~np.isnan(integer))
                    and np.all(~np.isposinf(integer))
                )
                if not (
                    np.isfinite(curvature)
                    and np.isfinite(slope)
                    and phase_channels_valid
                    and all(
                        math.isclose(
                            partition_shape,
                            shape,
                            rel_tol=1.0e-10,
                            abs_tol=1.0e-10,
                        )
                        for partition_shape in partition_shapes
                    )
                    and math.isclose(
                        curvature + slope,
                        shape,
                        rel_tol=1.0e-10,
                        abs_tol=1.0e-10,
                    )
                ):
                    raise ValueError("Shape evidence does not split into curvature and slope.")
            except (ArithmeticError, FloatingPointError, RuntimeError, ValueError):
                curvature = -math.inf
                slope = -math.inf
                partition_integer = np.full(
                    (self.partition_count, len(self.ladder)),
                    -math.inf,
                    dtype=float,
                )
                integer = np.full(len(self.ladder), -math.inf, dtype=float)
            self.cache[key] = (curvature, slope, partition_integer, integer)
            self.call_count += 1
        curvature, slope, partition_integer, integer = self.cache[key]
        return (
            float(curvature),
            float(slope),
            partition_integer.copy(),
            integer.copy(),
        )


@dataclass(frozen=True)
class _ResolvedInformationTarget:
    kind: str
    channel_index: int | None = None
    phase_noise: float | None = None
    partition_index: int | None = None


def _resolve_information_target(
    target: GlobalNoiseSMCInformationTarget,
    *,
    ladder: tuple[float, ...],
) -> _ResolvedInformationTarget:
    kind = str(target.kind).strip().lower()
    if kind in {"proposal", "curvature", "shape"}:
        if target.phase_noise is not None:
            raise ValueError(f"The {kind} target cannot declare phase noise.")
        if target.partition_index is not None:
            raise ValueError(f"The {kind} target cannot declare a partition.")
        return _ResolvedInformationTarget(kind=kind)
    if kind not in {"island-phase", "partition-phase", "global-phase"}:
        raise ValueError(f"Unknown global SMC information target: {target.kind}")
    if target.phase_noise is None:
        raise ValueError(f"The {kind} target requires a phase-noise channel.")
    if kind == "partition-phase" and (
        target.partition_index is None or int(target.partition_index) < 0
    ):
        raise ValueError("A partition-phase target requires a non-negative index.")
    if kind != "partition-phase" and target.partition_index is not None:
        raise ValueError(f"The {kind} target cannot declare a partition index.")
    matches = tuple(
        index
        for index, value in enumerate(ladder)
        if math.isclose(
            float(target.phase_noise),
            value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    )
    if len(matches) != 1:
        raise ValueError(
            f"The {kind} target must identify one declared noise channel."
        )
    return _ResolvedInformationTarget(
        kind=kind,
        channel_index=matches[0],
        phase_noise=ladder[matches[0]],
        partition_index=(
            int(target.partition_index)
            if kind == "partition-phase" and target.partition_index is not None
            else None
        ),
    )


def _information_scores(
    curvature: np.ndarray,
    slope: np.ndarray,
    partition_integer: np.ndarray,
    integer: np.ndarray,
    *,
    target: _ResolvedInformationTarget,
    proposal_log_densities: np.ndarray | None = None,
) -> np.ndarray:
    if target.kind == "proposal":
        if proposal_log_densities is None:
            raise ValueError("The proposal target requires proposal log densities.")
        proposal_scores = np.asarray(proposal_log_densities, dtype=float)
        if proposal_scores.shape != curvature.shape:
            raise ValueError("Proposal log densities have the wrong shape.")
        return proposal_scores
    if target.kind == "curvature":
        return curvature
    if target.kind == "shape":
        return curvature + slope
    channel_index = target.channel_index
    if channel_index is None or channel_index < 0 or channel_index >= integer.shape[1]:
        raise ValueError("The global information target channel is out of range.")
    if target.kind in {"island-phase", "partition-phase"}:
        partition_index = (
            partition_integer.shape[1] - 1
            if target.kind == "island-phase"
            else target.partition_index
        )
        if (
            partition_index is None
            or partition_index < 0
            or partition_index >= partition_integer.shape[1]
        ):
            raise ValueError("The phase-island partition target is out of range.")
        phase = partition_integer[:, partition_index, :]
    else:
        phase = integer
    return curvature + slope + phase[:, channel_index]


def _target_scores(
    curvature: np.ndarray,
    slope: np.ndarray,
    partition_integer: np.ndarray,
    integer: np.ndarray,
    *,
    source_target: _ResolvedInformationTarget,
    target_target: _ResolvedInformationTarget,
    bridge_fraction: float,
    proposal_log_densities: np.ndarray | None = None,
) -> np.ndarray:
    fraction = min(max(float(bridge_fraction), 0.0), 1.0)
    source = _information_scores(
        curvature,
        slope,
        partition_integer,
        integer,
        target=source_target,
        proposal_log_densities=proposal_log_densities,
    )
    target = _information_scores(
        curvature,
        slope,
        partition_integer,
        integer,
        target=target_target,
        proposal_log_densities=proposal_log_densities,
    )
    if fraction <= 0.0:
        return source
    if fraction >= 1.0:
        return target
    return (1.0 - fraction) * source + fraction * target


def _next_bridge_fraction(
    *,
    current_fraction: float,
    log_weights: np.ndarray,
    incremental_scores: np.ndarray,
    particles: np.ndarray,
    target_ess_fraction: float,
) -> float:
    remaining = 1.0 - float(current_fraction)
    if remaining <= 1.0e-12:
        return 1.0
    group_indices = np.empty(len(particles), dtype=int)
    grouped: dict[bytes, int] = {}
    for index, point in enumerate(particles):
        key = np.round(np.asarray(point, dtype=np.float64), 13).tobytes()
        if key not in grouped:
            grouped[key] = len(grouped)
        group_indices[index] = grouped[key]
    group_count = len(grouped)
    target_ess = max(float(target_ess_fraction) * group_count, 1.0)

    def mode_ess(values: np.ndarray) -> float:
        if group_count == len(particles):
            return _effective_sample_size(values)
        return _grouped_mode_effective_sample_size(
            values,
            group_indices=group_indices,
            group_count=group_count,
        )

    full_values = log_weights + remaining * incremental_scores
    if not np.any(np.isfinite(full_values)):
        return float(current_fraction)
    full = _normalize_log_weights(full_values)
    if mode_ess(full) >= target_ess:
        return 1.0
    low = 0.0
    high = remaining
    for _ in range(48):
        middle = 0.5 * (low + high)
        trial_values = log_weights + middle * incremental_scores
        if not np.any(np.isfinite(trial_values)):
            high = middle
            continue
        trial = _normalize_log_weights(trial_values)
        if mode_ess(trial) >= target_ess:
            low = middle
        else:
            high = middle
    if low <= 1.0e-12:
        return float(current_fraction)
    return min(float(current_fraction) + low, 1.0)


def _append_archive(
    archive: list[GlobalNoiseSMCArchivePoint],
    *,
    source: str,
    factor_index: int,
    bridge_fraction: float,
    orbit_values: np.ndarray,
    curvature: float,
    slope: float,
    partition_integer: np.ndarray,
    integer: np.ndarray,
) -> None:
    if not (
        np.isfinite(curvature)
        and np.isfinite(slope)
        and np.all(~np.isnan(partition_integer))
        and np.all(~np.isposinf(partition_integer))
        and np.all(~np.isnan(integer))
        and np.all(~np.isposinf(integer))
    ):
        return
    archive.append(
        GlobalNoiseSMCArchivePoint(
            point_id=f"global_smc_point_{len(archive):08d}",
            source=str(source),
            factor_index=int(factor_index),
            bridge_fraction=float(bridge_fraction),
            orbit_normalized_values=np.asarray(orbit_values, dtype=float).copy(),
            curvature_log_likelihood=float(curvature),
            slope_log_likelihood=float(slope),
            shape_log_likelihood=float(curvature + slope),
            partition_integer_log_likelihoods=np.asarray(
                partition_integer,
                dtype=float,
            ).copy(),
            island_integer_log_likelihoods=np.asarray(
                partition_integer[-1],
                dtype=float,
            ).copy(),
            integer_log_likelihoods=np.asarray(integer, dtype=float).copy(),
        )
    )


def _mutate_particles(
    *,
    unit_particles: np.ndarray,
    proposal_group_indices: np.ndarray,
    proposal_log_densities: np.ndarray,
    curvature: np.ndarray,
    slope: np.ndarray,
    partition_integer: np.ndarray,
    integer: np.ndarray,
    factor_index: int,
    source_target: _ResolvedInformationTarget,
    target_target: _ResolvedInformationTarget,
    bridge_fraction: float,
    proposal: _BoundedOrbitProposal,
    evaluator: _GlobalEvaluator,
    archive: list[GlobalNoiseSMCArchivePoint],
    config: GlobalNoiseSMCConfig,
    rng: np.random.Generator,
    proposal_covariances: np.ndarray | None = None,
    protected_particle_indices: Sequence[int] = (),
) -> tuple[int, int]:
    count, dimension = unit_particles.shape
    mutation_count = min(
        count,
        max(int(round(count * float(config.mutation_fraction))), 1),
    )
    if proposal_covariances is None:
        local_covariances = np.asarray(
            [np.cov(unit_particles, rowvar=False)],
            dtype=float,
        )
    else:
        local_covariances = np.asarray(proposal_covariances, dtype=float)
        if local_covariances.ndim == 2:
            local_covariances = local_covariances[None, :, :]
    if dimension == 1:
        local_covariances = local_covariances.reshape(-1, 1, 1)
    if local_covariances.shape[1:] != (dimension, dimension):
        raise ValueError("Global SMC local proposal covariance has the wrong shape.")
    floor = max(
        float(config.covariance_eigenvalue_floor),
        np.finfo(float).eps,
    )
    local_roots = []
    for covariance in local_covariances:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        local_roots.append(
            eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, floor)))
        )
    local_roots = np.asarray(local_roots, dtype=float)
    local_scale = max(float(config.local_proposal_scale), 1.0e-8) * 2.38 / math.sqrt(
        dimension
    )
    local_scale_factors = tuple(
        float(value)
        for value in config.local_proposal_scale_factors
        if np.isfinite(float(value)) and float(value) > 0.0
    )
    if not local_scale_factors:
        raise ValueError("Global SMC local proposal scales must be positive.")
    slice_probability = min(
        max(float(config.directional_slice_probability), 0.0),
        1.0,
    )
    independence_probability = min(
        max(float(config.independence_proposal_probability), 0.0),
        1.0 - slice_probability,
    )
    differential_probability = min(
        max(float(config.differential_evolution_probability), 0.0),
        1.0 - slice_probability - independence_probability,
    )
    slice_axis_probability = min(
        max(float(config.directional_slice_axis_probability), 0.0),
        1.0,
    )
    maximum_slice_steps = max(int(config.maximum_slice_shrink_steps), 1)
    minimum_slice_interval = max(
        float(config.minimum_slice_interval),
        np.finfo(float).eps,
    )
    jitter = max(float(config.proposal_jitter), 0.0)
    attempts = 0
    accepts = 0
    protected = {
        int(index)
        for index in protected_particle_indices
        if 0 <= int(index) < count
    }
    for mutation_step in range(max(int(config.mutation_steps), 0)):
        by_point: dict[bytes, list[int]] = {}
        for index, point in enumerate(unit_particles):
            key = np.round(np.asarray(point, dtype=np.float64), 13).tobytes()
            by_point.setdefault(key, []).append(index)
        duplicate_indices = [
            index
            for values in by_point.values()
            for index in values[1:]
            if index not in protected
        ]
        target_count = max(mutation_count, len(duplicate_indices))
        selected = set(duplicate_indices)
        remaining_indices = np.asarray(
            [
                index
                for index in range(count)
                if index not in selected and index not in protected
            ],
            dtype=int,
        )
        additional_count = min(target_count - len(selected), remaining_indices.size)
        if additional_count > 0:
            selected.update(
                int(value)
                for value in rng.choice(
                    remaining_indices,
                    size=additional_count,
                    replace=False,
                )
            )
        indices = np.asarray(sorted(selected), dtype=int)
        snapshot = unit_particles.copy()
        for index in indices:
            proposal_group = int(proposal_group_indices[index])
            covariance_group = min(
                max(proposal_group, 0),
                local_roots.shape[0] - 1,
            )
            local_root = local_roots[covariance_group]
            current_target = _target_scores(
                curvature[index : index + 1],
                slope[index : index + 1],
                partition_integer[index : index + 1],
                integer[index : index + 1],
                source_target=source_target,
                target_target=target_target,
                bridge_fraction=bridge_fraction,
                proposal_log_densities=proposal_log_densities[index : index + 1],
            )[0]
            draw = rng.random()
            proposal_kind = "local"
            independence_move = False
            if draw < slice_probability and np.isfinite(current_target):
                if rng.random() < slice_axis_probability:
                    direction = np.zeros(dimension, dtype=float)
                    direction[int(rng.integers(dimension))] = (
                        -1.0 if rng.random() < 0.5 else 1.0
                    )
                else:
                    direction = local_root @ rng.normal(size=dimension)
                    direction_norm = float(np.linalg.norm(direction))
                    if not np.isfinite(direction_norm) or direction_norm <= 0.0:
                        direction = rng.normal(size=dimension)
                        direction_norm = float(np.linalg.norm(direction))
                    direction /= direction_norm
                lower_step = -math.inf
                upper_step = math.inf
                for coordinate, component in zip(
                    unit_particles[index],
                    direction,
                ):
                    if abs(float(component)) <= np.finfo(float).eps:
                        continue
                    first = -float(coordinate) / float(component)
                    second = (1.0 - float(coordinate)) / float(component)
                    lower_step = max(lower_step, min(first, second))
                    upper_step = min(upper_step, max(first, second))
                log_slice_height = float(
                    current_target
                    + math.log(max(rng.random(), np.finfo(float).tiny))
                )
                for _ in range(maximum_slice_steps):
                    if upper_step - lower_step <= minimum_slice_interval:
                        break
                    attempts += 1
                    step = float(rng.uniform(lower_step, upper_step))
                    proposed_unit = np.clip(
                        unit_particles[index] + step * direction,
                        0.0,
                        1.0,
                    )
                    proposed_log_density = proposal.log_density_unit(proposed_unit)
                    orbit = proposal.to_orbit(proposed_unit)
                    (
                        proposed_curvature,
                        proposed_slope,
                        proposed_partition_integer,
                        proposed_integer,
                    ) = evaluator.evaluate(orbit)
                    _append_archive(
                        archive,
                        source="mutation-directional-slice",
                        factor_index=factor_index,
                        bridge_fraction=bridge_fraction,
                        orbit_values=orbit,
                        curvature=proposed_curvature,
                        slope=proposed_slope,
                        partition_integer=proposed_partition_integer,
                        integer=proposed_integer,
                    )
                    proposed_target = _target_scores(
                        np.asarray([proposed_curvature]),
                        np.asarray([proposed_slope]),
                        proposed_partition_integer[None, :],
                        proposed_integer[None, :],
                        source_target=source_target,
                        target_target=target_target,
                        bridge_fraction=bridge_fraction,
                        proposal_log_densities=np.asarray(
                            [proposed_log_density]
                        ),
                    )[0]
                    if np.isfinite(proposed_target) and proposed_target >= log_slice_height:
                        unit_particles[index] = proposed_unit
                        proposal_log_densities[index] = proposed_log_density
                        curvature[index] = proposed_curvature
                        slope[index] = proposed_slope
                        partition_integer[index] = proposed_partition_integer
                        integer[index] = proposed_integer
                        accepts += 1
                        break
                    if step < 0.0:
                        lower_step = step
                    else:
                        upper_step = step
                continue
            attempts += 1
            if draw < slice_probability + independence_probability:
                proposed_unit, proposal_source = proposal.draw_group_unit(
                    proposal_group,
                    rng=rng,
                )
                proposal_kind = f"independence-{proposal_source}"
                independence_move = True
            elif (
                draw
                < slice_probability
                + independence_probability
                + differential_probability
                and count >= 3
            ):
                choices = np.delete(np.arange(count), index)
                left, right = rng.choice(choices, size=2, replace=False)
                gamma = 2.38 / math.sqrt(2.0 * dimension)
                proposed_unit = _reflect_unit(
                    unit_particles[index]
                    + gamma * (snapshot[left] - snapshot[right])
                    + rng.normal(0.0, jitter, dimension)
                )
                proposal_kind = "differential-evolution"
            else:
                scale_factor = local_scale_factors[
                    int(rng.integers(len(local_scale_factors)))
                ]
                proposed_unit = _reflect_unit(
                    unit_particles[index]
                    + scale_factor
                    * local_scale
                    * local_root
                    @ rng.normal(size=dimension)
                )
            proposed_log_density = proposal.log_density_unit(proposed_unit)
            proposal_log_ratio = (
                float(
                    proposal.log_density_group_unit(
                        unit_particles[index],
                        int(proposal_group_indices[index]),
                    )
                    - proposal.log_density_group_unit(
                        proposed_unit,
                        int(proposal_group_indices[index]),
                    )
                )
                if independence_move
                else 0.0
            )
            orbit = proposal.to_orbit(proposed_unit)
            (
                proposed_curvature,
                proposed_slope,
                proposed_partition_integer,
                proposed_integer,
            ) = evaluator.evaluate(orbit)
            _append_archive(
                archive,
                source=f"mutation-{proposal_kind}",
                factor_index=factor_index,
                bridge_fraction=bridge_fraction,
                orbit_values=orbit,
                curvature=proposed_curvature,
                slope=proposed_slope,
                partition_integer=proposed_partition_integer,
                integer=proposed_integer,
            )
            proposed_target = _target_scores(
                np.asarray([proposed_curvature]),
                np.asarray([proposed_slope]),
                proposed_partition_integer[None, :],
                proposed_integer[None, :],
                source_target=source_target,
                target_target=target_target,
                bridge_fraction=bridge_fraction,
                proposal_log_densities=np.asarray([proposed_log_density]),
            )[0]
            if np.isnan(proposed_target) or np.isposinf(proposed_target):
                continue
            if np.isneginf(proposed_target):
                continue
            if np.isneginf(current_target):
                log_acceptance = math.inf
            elif not np.isfinite(current_target):
                continue
            else:
                log_acceptance = float(
                    proposed_target - current_target + proposal_log_ratio
                )
            if math.log(max(rng.random(), np.finfo(float).tiny)) < log_acceptance:
                unit_particles[index] = proposed_unit
                proposal_log_densities[index] = proposed_log_density
                curvature[index] = proposed_curvature
                slope[index] = proposed_slope
                partition_integer[index] = proposed_partition_integer
                integer[index] = proposed_integer
                accepts += 1
    return attempts, accepts


def _archive_rankings(
    archive: Sequence[GlobalNoiseSMCArchivePoint],
) -> tuple[tuple[GlobalNoiseSMCArchivePoint, ...], ...]:
    rankings = [
        tuple(sorted(archive, key=lambda point: -point.curvature_log_likelihood)),
        tuple(sorted(archive, key=lambda point: -point.shape_log_likelihood))
    ]
    channel_count = int(archive[0].integer_log_likelihoods.size)
    partition_count = int(
        archive[0].partition_integer_log_likelihoods.shape[0]
    )
    for partition_index in range(partition_count):
        for channel_index in range(channel_count):
            rankings.append(
                tuple(
                    sorted(
                        archive,
                        key=lambda point, partition=partition_index, index=channel_index: -(
                            point.shape_log_likelihood
                            + float(
                                point.partition_integer_log_likelihoods[
                                    partition,
                                    index,
                                ]
                            )
                        ),
                    )
                )
            )
    for channel_index in range(channel_count):
        rankings.append(
            tuple(
                sorted(
                    archive,
                    key=lambda point, index=channel_index: -(
                        point.shape_log_likelihood
                        + float(point.integer_log_likelihoods[index])
                    ),
                )
            )
        )
        rankings.append(
            tuple(
                sorted(
                    archive,
                    key=lambda point, index=channel_index: -float(
                        point.integer_log_likelihoods[index]
                    ),
                )
            )
        )
    return tuple(rankings)


def _local_mode_covariance(
    point: GlobalNoiseSMCArchivePoint,
    *,
    archive_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    floor: float,
) -> np.ndarray:
    width = upper - lower
    unit = (archive_values - lower) / width
    center = (point.orbit_normalized_values - lower) / width
    distances = np.linalg.norm(unit - center, axis=1)
    neighbour_count = min(max(2 * center.size + 1, 16), archive_values.shape[0])
    indices = np.argsort(distances)[:neighbour_count]
    if neighbour_count <= 1:
        covariance = np.diag(np.square(0.05 * width))
    else:
        covariance = np.cov(archive_values[indices], rowvar=False)
        if center.size == 1:
            covariance = np.asarray([[float(covariance)]], dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    absolute_floor = np.square(width) * max(float(floor), np.finfo(float).eps)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, absolute_floor)) @ eigenvectors.T


def select_global_noise_smc_modes(
    archive: Sequence[GlobalNoiseSMCArchivePoint],
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    maximum_modes: int,
    minimum_separation: float,
    covariance_floor: float,
) -> tuple[GlobalNoiseSMCMode, ...]:
    """Select a cross-channel search portfolio without scalarizing evidence."""

    if not archive:
        return ()
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    width = upper_values - lower_values
    target = max(int(maximum_modes), 1)
    rankings = _archive_rankings(archive)
    endpoint_joint_ranking = rankings[-2]
    endpoint_integer_ranking = rankings[-1]
    support_rankings = rankings[:-2]

    def interleave(
        values_by_ranking: Sequence[Sequence[GlobalNoiseSMCArchivePoint]],
    ) -> list[GlobalNoiseSMCArchivePoint]:
        points = []
        for rank in range(max(len(values) for values in values_by_ranking)):
            for values in values_by_ranking:
                if rank < len(values):
                    points.append(values[rank])
        return points

    support_interleaved = interleave(support_rankings)
    all_interleaved = []
    for rank in range(max(len(values) for values in rankings)):
        for values in rankings:
            if rank < len(values):
                all_interleaved.append(values[rank])
    selected: list[GlobalNoiseSMCArchivePoint] = []
    selected_unit: list[np.ndarray] = []
    seen: set[bytes] = set()

    def add(point: GlobalNoiseSMCArchivePoint, *, enforce_separation: bool) -> bool:
        key = np.round(point.orbit_normalized_values, 13).tobytes()
        if key in seen:
            return False
        unit = (point.orbit_normalized_values - lower_values) / width
        if enforce_separation and selected_unit:
            distance = min(float(np.linalg.norm(unit - value)) for value in selected_unit)
            if distance < max(float(minimum_separation), 0.0):
                return False
        seen.add(key)
        selected.append(point)
        selected_unit.append(unit)
        return True

    endpoint_target = max(1, int(math.ceil(0.5 * target)))
    add(endpoint_joint_ranking[0], enforce_separation=False)
    local_endpoint_target = min(endpoint_target, 2)
    for point in endpoint_joint_ranking:
        if len(selected) >= local_endpoint_target:
            break
        add(point, enforce_separation=False)
    for point in endpoint_joint_ranking:
        if len(selected) >= endpoint_target:
            break
        add(point, enforce_separation=True)
    for point in endpoint_joint_ranking:
        if len(selected) >= endpoint_target:
            break
        add(point, enforce_separation=False)

    if len(selected) < target:
        add(rankings[0][0], enforce_separation=False)
    for ranking in (endpoint_integer_ranking, *support_rankings[1:]):
        if len(selected) >= target:
            break
        add(ranking[0], enforce_separation=True)
    for point in support_interleaved:
        if len(selected) >= target:
            break
        add(point, enforce_separation=True)
    for point in all_interleaved:
        if len(selected) >= target:
            break
        add(point, enforce_separation=False)

    archive_values = np.asarray(
        [point.orbit_normalized_values for point in archive],
        dtype=float,
    )
    modes = []
    for index, point in enumerate(selected):
        modes.append(
            GlobalNoiseSMCMode(
                mode_id=f"global_noise_smc_mode_{index:05d}",
                source_point_id=point.point_id,
                source=point.source,
                factor_index=point.factor_index,
                bridge_fraction=point.bridge_fraction,
                orbit_normalized_values=point.orbit_normalized_values.copy(),
                orbit_covariance=_local_mode_covariance(
                    point,
                    archive_values=archive_values,
                    lower=lower_values,
                    upper=upper_values,
                    floor=covariance_floor,
                ),
                curvature_log_likelihood=point.curvature_log_likelihood,
                slope_log_likelihood=point.slope_log_likelihood,
                shape_log_likelihood=point.shape_log_likelihood,
                partition_integer_log_likelihoods=(
                    point.partition_integer_log_likelihoods.copy()
                ),
                island_integer_log_likelihoods=(
                    point.island_integer_log_likelihoods.copy()
                ),
                integer_log_likelihoods=point.integer_log_likelihoods.copy(),
            )
        )
    return tuple(modes)


def run_global_noise_smc(
    *,
    likelihood: Any,
    seeds: Sequence[NoiseAnnealedOrbitSeed] = (),
    config: GlobalNoiseSMCConfig | None = None,
    information_path: GlobalNoiseSMCInformationPath | None = None,
    initial_population: GlobalNoiseSMCPosteriorPopulation | None = None,
    progress_callback: Callable[[GlobalNoiseSMCStage], None] | None = None,
) -> GlobalNoiseSMCResult:
    """Transport the all-session continuous posterior before choosing a path."""

    active = config or GlobalNoiseSMCConfig()
    ladder = _validated_noise_ladder(active.phase_noise_ladder)
    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape or not np.all(lower < upper):
        raise ValueError("Global orbit support must contain finite ordered vectors.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("Global orbit support must be finite.")
    if initial_population is None:
        count = max(int(active.particle_count), 2 * lower.size + 1, len(seeds) + 1)
    else:
        initial_particles = np.asarray(initial_population.particles, dtype=float)
        initial_log_weights = np.asarray(initial_population.log_weights, dtype=float)
        if (
            initial_particles.ndim != 2
            or initial_particles.shape[1:] != lower.shape
            or initial_log_weights.shape != (initial_particles.shape[0],)
        ):
            raise ValueError("The initial posterior population has incompatible shapes.")
        if initial_particles.shape[0] < 2 * lower.size + 1:
            raise ValueError("The initial posterior population is too small.")
        if (
            not np.all(np.isfinite(initial_particles))
            or np.any(initial_particles < lower)
            or np.any(initial_particles > upper)
        ):
            raise ValueError("The initial posterior population lies outside support.")
        if np.any(np.isnan(initial_log_weights)) or np.all(
            np.isneginf(initial_log_weights)
        ):
            raise ValueError("The initial posterior population has invalid weights.")
        if active.retain_seed_centers_for_mode_discovery:
            raise ValueError(
                "Retained seed centers cannot replace posterior population particles."
            )
        count = int(initial_particles.shape[0])
    rng = np.random.default_rng(max(int(active.random_seed), 0))
    proposal = _BoundedOrbitProposal(
        lower=lower,
        upper=upper,
        seeds=seeds,
        broad_weight=active.broad_proposal_weight,
        chart_scale_factors=active.chart_scale_factors,
        covariance_floor=active.covariance_eigenvalue_floor,
    )
    evaluator = _GlobalEvaluator(likelihood, ladder)
    unit_particles = np.empty((count, lower.size), dtype=float)
    proposal_log_densities = np.empty(count, dtype=float)
    curvature = np.empty(count, dtype=float)
    slope = np.empty(count, dtype=float)
    partition_integer = np.empty(
        (count, evaluator.partition_count, len(ladder)),
        dtype=float,
    )
    integer = np.empty((count, len(ladder)), dtype=float)
    archive: list[GlobalNoiseSMCArchivePoint] = []
    margin = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
    for seed in seeds:
        orbit = np.clip(
            np.asarray(seed.orbit_normalized_values, dtype=float),
            lower + margin,
            upper - margin,
        )
        (
            curvature_value,
            slope_value,
            partition_integer_values,
            integer_values,
        ) = evaluator.evaluate(orbit)
        _append_archive(
            archive,
            source=f"declared-chart-center-{seed.source}",
            factor_index=0,
            bridge_fraction=0.0,
            orbit_values=orbit,
            curvature=curvature_value,
            slope=slope_value,
            partition_integer=partition_integer_values,
            integer=integer_values,
        )
    if initial_population is None:
        initial_groups = proposal.stratified_group_indices(count, rng=rng)
        unit_particles = np.empty((count, lower.size), dtype=float)
        log_weights = np.full(count, -math.log(count), dtype=float)
    else:
        margin = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
        orbit_particles = np.clip(initial_particles, lower + margin, upper - margin)
        unit_particles = np.asarray(
            [proposal.to_unit(values) for values in orbit_particles],
            dtype=float,
        )
        initial_groups = np.asarray(
            [proposal.most_likely_group(values) for values in unit_particles],
            dtype=int,
        )
        log_weights = _normalize_log_weights(initial_log_weights)
    proposal_group_indices = initial_groups.copy()
    for index, group_index in enumerate(initial_groups):
        if initial_population is None:
            unit, source = proposal.draw_group_unit(group_index, rng=rng)
            orbit = proposal.to_orbit(unit)
        else:
            unit = unit_particles[index]
            orbit = proposal.to_orbit(unit)
            source = "posterior-population"
        (
            curvature_value,
            slope_value,
            partition_integer_values,
            integer_values,
        ) = evaluator.evaluate(orbit)
        unit_particles[index] = unit
        proposal_log_densities[index] = proposal.log_density_unit(unit)
        curvature[index] = curvature_value
        slope[index] = slope_value
        partition_integer[index] = partition_integer_values
        integer[index] = integer_values
        _append_archive(
            archive,
            source=f"initial-{source}",
            factor_index=0,
            bridge_fraction=0.0,
            orbit_values=orbit,
            curvature=curvature_value,
            slope=slope_value,
            partition_integer=partition_integer_values,
            integer=integer_values,
        )
    protected_particle_indices: list[int] = []
    if active.retain_seed_centers_for_mode_discovery:
        margin = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
        for family_index, seed in enumerate(seeds):
            candidates = np.flatnonzero(initial_groups == family_index + 1)
            if candidates.size == 0:
                continue
            index = int(candidates[0])
            protected_particle_indices.append(index)
            orbit = np.clip(
                np.asarray(seed.orbit_normalized_values, dtype=float),
                lower + margin,
                upper - margin,
            )
            unit = (orbit - lower) / (upper - lower)
            (
                curvature_value,
                slope_value,
                partition_integer_values,
                integer_values,
            ) = evaluator.evaluate(orbit)
            unit_particles[index] = unit
            proposal_group_indices[index] = family_index + 1
            proposal_log_densities[index] = proposal.log_density_unit(unit)
            curvature[index] = curvature_value
            slope[index] = slope_value
            partition_integer[index] = partition_integer_values
            integer[index] = integer_values
            _append_archive(
                archive,
                source=f"initial-retained-center-{seed.source}",
                factor_index=0,
                bridge_fraction=0.0,
                orbit_values=orbit,
                curvature=curvature_value,
                slope=slope_value,
                partition_integer=partition_integer_values,
                integer=integer_values,
            )
    target_ess_fraction = min(
        max(float(active.effective_sample_size_target_fraction), 0.05),
        0.999,
    )
    resample_threshold_fraction = min(
        max(float(active.resample_threshold_fraction), 0.05),
        1.0,
    )
    history: list[GlobalNoiseSMCStage] = []
    step_index = 0
    if information_path is not None:
        if not str(information_path.path_id).strip() or not information_path.targets:
            raise ValueError("A global SMC information path must have an id and targets.")
        declared_targets = tuple(information_path.targets)
    elif active.initial_phase_noise is None:
        declared_targets = (
            GlobalNoiseSMCInformationTarget("curvature"),
            GlobalNoiseSMCInformationTarget("shape"),
            *(
                GlobalNoiseSMCInformationTarget("global-phase", value)
                for value in ladder
            ),
        )
    else:
        requested_noise = float(active.initial_phase_noise)
        matching_indices = tuple(
            index
            for index, value in enumerate(ladder)
            if math.isclose(value, requested_noise, rel_tol=1.0e-12, abs_tol=1.0e-15)
        )
        if len(matching_indices) != 1:
            raise ValueError(
                "The initial phase noise must identify one noise-ladder channel."
            )
        declared_targets = tuple(
            GlobalNoiseSMCInformationTarget("global-phase", ladder[index])
            for index in range(matching_indices[0], len(ladder))
        )
    resolved_path = tuple(
        _resolve_information_target(target, ladder=ladder)
        for target in declared_targets
    )
    initial_source_target = (
        _resolve_information_target(
            initial_population.source_target,
            ladder=ladder,
        )
        if initial_population is not None
        else None
    )
    if initial_source_target is not None and initial_source_target.kind == "proposal":
        raise ValueError("An initial posterior population needs a posterior target.")
    if any(target.kind == "proposal" for target in resolved_path):
        raise ValueError("The proposal distribution is implicit and cannot be a path target.")
    reached_endpoint = True

    for factor_index, target_target in enumerate(resolved_path):
        source_target = (
            (
                initial_source_target
                if initial_source_target is not None
                else _ResolvedInformationTarget("proposal")
            )
            if factor_index == 0
            else resolved_path[factor_index - 1]
        )
        bridge_fraction = 0.0
        incremental_scores = _information_scores(
            curvature,
            slope,
            partition_integer,
            integer,
            target=target_target,
            proposal_log_densities=proposal_log_densities,
        ) - _information_scores(
            curvature,
            slope,
            partition_integer,
            integer,
            target=source_target,
            proposal_log_densities=proposal_log_densities,
        )
        factor_steps = 0
        while bridge_fraction < 1.0 - 1.0e-12:
            factor_steps += 1
            if factor_steps > max(int(active.maximum_bridge_steps), 1):
                reached_endpoint = False
                break
            unique_count = _unique_particle_count(unit_particles)
            if (
                _effective_sample_size(log_weights)
                <= resample_threshold_fraction * count
                or _mode_effective_sample_size(log_weights, unit_particles)
                <= target_ess_fraction * unique_count
            ):
                proposal_covariances = _regularized_group_covariances(
                    unit_particles,
                    log_weights,
                    group_indices=proposal_group_indices,
                    group_count=proposal.group_count,
                )
                indices = _systematic_resample(log_weights, rng=rng)
                if protected_particle_indices:
                    protected_count = len(protected_particle_indices)
                    indices[:protected_count] = np.asarray(
                        protected_particle_indices,
                        dtype=int,
                    )
                    protected_particle_indices = list(range(protected_count))
                unit_particles = unit_particles[indices].copy()
                proposal_group_indices = proposal_group_indices[indices].copy()
                proposal_log_densities = proposal_log_densities[indices].copy()
                curvature = curvature[indices].copy()
                slope = slope[indices].copy()
                partition_integer = partition_integer[indices].copy()
                integer = integer[indices].copy()
                log_weights = np.full(count, -math.log(count), dtype=float)
                _mutate_particles(
                    unit_particles=unit_particles,
                    proposal_group_indices=proposal_group_indices,
                    proposal_log_densities=proposal_log_densities,
                    curvature=curvature,
                    slope=slope,
                    partition_integer=partition_integer,
                    integer=integer,
                    factor_index=factor_index,
                    source_target=source_target,
                    target_target=target_target,
                    bridge_fraction=bridge_fraction,
                    proposal=proposal,
                    evaluator=evaluator,
                    archive=archive,
                    config=active,
                    rng=rng,
                    proposal_covariances=proposal_covariances,
                    protected_particle_indices=protected_particle_indices,
                )
                incremental_scores = _information_scores(
                    curvature,
                    slope,
                    partition_integer,
                    integer,
                    target=target_target,
                    proposal_log_densities=proposal_log_densities,
                ) - _information_scores(
                    curvature,
                    slope,
                    partition_integer,
                    integer,
                    target=source_target,
                    proposal_log_densities=proposal_log_densities,
                )
            next_fraction = _next_bridge_fraction(
                current_fraction=bridge_fraction,
                log_weights=log_weights,
                incremental_scores=incremental_scores,
                particles=unit_particles,
                target_ess_fraction=target_ess_fraction,
            )
            delta = next_fraction - bridge_fraction
            if delta <= 1.0e-12:
                reached_endpoint = False
                break
            log_weights = _normalize_log_weights(
                log_weights + delta * incremental_scores
            )
            bridge_fraction = next_fraction
            ess = _effective_sample_size(log_weights)
            mode_ess = _mode_effective_sample_size(log_weights, unit_particles)
            unique_count = _unique_particle_count(unit_particles)
            resampled = bool(
                ess <= resample_threshold_fraction * count
                or mode_ess <= resample_threshold_fraction * unique_count
                or bridge_fraction < 1.0 - 1.0e-12
            )
            attempts = 0
            accepts = 0
            rejuvenation_rounds = 0
            proposal_covariances = _regularized_group_covariances(
                unit_particles,
                log_weights,
                group_indices=proposal_group_indices,
                group_count=proposal.group_count,
            )
            if resampled:
                indices = _systematic_resample(log_weights, rng=rng)
                if protected_particle_indices:
                    protected_count = len(protected_particle_indices)
                    indices[:protected_count] = np.asarray(
                        protected_particle_indices,
                        dtype=int,
                    )
                    protected_particle_indices = list(range(protected_count))
                unit_particles = unit_particles[indices].copy()
                proposal_group_indices = proposal_group_indices[indices].copy()
                proposal_log_densities = proposal_log_densities[indices].copy()
                curvature = curvature[indices].copy()
                slope = slope[indices].copy()
                partition_integer = partition_integer[indices].copy()
                integer = integer[indices].copy()
                log_weights = np.full(count, -math.log(count), dtype=float)
                ess = float(count)
            move_attempts, move_accepts = _mutate_particles(
                unit_particles=unit_particles,
                proposal_group_indices=proposal_group_indices,
                proposal_log_densities=proposal_log_densities,
                curvature=curvature,
                slope=slope,
                partition_integer=partition_integer,
                integer=integer,
                factor_index=factor_index,
                source_target=source_target,
                target_target=target_target,
                bridge_fraction=bridge_fraction,
                proposal=proposal,
                evaluator=evaluator,
                archive=archive,
                config=active,
                rng=rng,
                proposal_covariances=proposal_covariances,
                protected_particle_indices=protected_particle_indices,
            )
            attempts += move_attempts
            accepts += move_accepts
            if move_attempts:
                rejuvenation_rounds += 1
            minimum_unique = min(
                count,
                max(
                    int(
                        math.ceil(
                            min(
                                max(float(active.minimum_unique_particle_fraction), 0.0),
                                1.0,
                            )
                            * count
                        )
                    ),
                    1,
                ),
            )
            unique_count = _unique_particle_count(unit_particles)
            mode_ess = _mode_effective_sample_size(log_weights, unit_particles)
            while (
                (
                    unique_count < minimum_unique
                    or mode_ess < target_ess_fraction * unique_count
                )
                and rejuvenation_rounds
                < max(int(active.maximum_rejuvenation_rounds), 1)
                and int(active.mutation_steps) > 0
            ):
                move_attempts, move_accepts = _mutate_particles(
                    unit_particles=unit_particles,
                    proposal_group_indices=proposal_group_indices,
                    proposal_log_densities=proposal_log_densities,
                    curvature=curvature,
                    slope=slope,
                    partition_integer=partition_integer,
                    integer=integer,
                    factor_index=factor_index,
                    source_target=source_target,
                    target_target=target_target,
                    bridge_fraction=bridge_fraction,
                    proposal=proposal,
                    evaluator=evaluator,
                    archive=archive,
                    config=active,
                    rng=rng,
                    proposal_covariances=proposal_covariances,
                    protected_particle_indices=protected_particle_indices,
                )
                attempts += move_attempts
                accepts += move_accepts
                rejuvenation_rounds += 1
                unique_count = _unique_particle_count(unit_particles)
                mode_ess = _mode_effective_sample_size(
                    log_weights,
                    unit_particles,
                )
            incremental_scores = _information_scores(
                curvature,
                slope,
                partition_integer,
                integer,
                target=target_target,
                proposal_log_densities=proposal_log_densities,
            ) - _information_scores(
                curvature,
                slope,
                partition_integer,
                integer,
                target=source_target,
                proposal_log_densities=proposal_log_densities,
            )
            mode_ess = _mode_effective_sample_size(log_weights, unit_particles)
            unique_count = _unique_particle_count(unit_particles)
            diversity_satisfied = bool(
                (
                    unique_count >= minimum_unique
                    and mode_ess >= target_ess_fraction * unique_count
                )
                or int(active.mutation_steps) <= 0
            )
            orbit_particles = lower + unit_particles * (upper - lower)
            for index in range(count):
                _append_archive(
                    archive,
                    source="smc-particle",
                    factor_index=factor_index,
                    bridge_fraction=bridge_fraction,
                    orbit_values=orbit_particles[index],
                    curvature=curvature[index],
                    slope=slope[index],
                    partition_integer=partition_integer[index],
                    integer=integer[index],
                )
            target_scores = _target_scores(
                curvature,
                slope,
                partition_integer,
                integer,
                source_target=source_target,
                target_target=target_target,
                bridge_fraction=bridge_fraction,
                proposal_log_densities=proposal_log_densities,
            )
            exact_scores = curvature + slope + integer[:, -1]
            stage = GlobalNoiseSMCStage(
                step_index=step_index,
                factor_index=factor_index,
                factor=(
                    "slope"
                    if target_target.kind == "shape"
                    else (
                        "phase-noise"
                        if target_target.kind == "global-phase"
                        else target_target.kind
                    )
                ),
                source_phase_noise=source_target.phase_noise,
                target_phase_noise=target_target.phase_noise,
                bridge_fraction=bridge_fraction,
                delta_bridge_fraction=delta,
                effective_sample_size=float(ess),
                mode_effective_sample_size=float(mode_ess),
                unique_particle_count=unique_count,
                diversity_satisfied=diversity_satisfied,
                resampled=resampled,
                rejuvenation_round_count=rejuvenation_rounds,
                mutation_attempt_count=attempts,
                mutation_accept_count=accepts,
                mutation_acceptance_fraction=(
                    float(accepts / attempts) if attempts else 0.0
                ),
                best_target_log_likelihood=float(np.max(target_scores)),
                best_exact_log_likelihood=float(np.max(exact_scores)),
                finite_particle_count=int(np.sum(np.isfinite(target_scores))),
            )
            history.append(stage)
            if progress_callback is not None:
                progress_callback(stage)
            step_index += 1
            if not diversity_satisfied:
                reached_endpoint = False
                break
            if len(archive) > max(int(active.maximum_archive_size), count):
                rankings = _archive_rankings(archive)
                kept = []
                seen_ids = set()
                for rank in range(max(len(values) for values in rankings)):
                    for values in rankings:
                        if rank >= len(values):
                            continue
                        point = values[rank]
                        if point.point_id in seen_ids:
                            continue
                        seen_ids.add(point.point_id)
                        kept.append(point)
                        if len(kept) >= int(active.maximum_archive_size):
                            break
                    if len(kept) >= int(active.maximum_archive_size):
                        break
                archive = kept
        if not reached_endpoint:
            break

    orbit_particles = lower + unit_particles * (upper - lower)
    portfolio = select_global_noise_smc_modes(
        archive,
        lower=lower,
        upper=upper,
        maximum_modes=active.portfolio_mode_count,
        minimum_separation=active.minimum_mode_separation,
        covariance_floor=active.covariance_eigenvalue_floor,
    )
    return GlobalNoiseSMCResult(
        phase_noise_ladder=ladder,
        particles=orbit_particles,
        log_weights=_normalize_log_weights(log_weights),
        curvature_log_likelihoods=curvature.copy(),
        slope_log_likelihoods=slope.copy(),
        shape_log_likelihoods=(curvature + slope).copy(),
        partition_integer_log_likelihoods=partition_integer.copy(),
        island_integer_log_likelihoods=partition_integer[:, -1, :].copy(),
        integer_log_likelihoods=integer.copy(),
        portfolio_modes=portfolio,
        archive=tuple(archive),
        stage_history=tuple(history),
        likelihood_call_count=evaluator.call_count,
        reached_exact_endpoint=bool(
            reached_endpoint
            and len(history) > 0
            and history[-1].factor_index == len(resolved_path) - 1
            and history[-1].bridge_fraction >= 1.0 - 1.0e-12
        ),
    )


def run_global_noise_smc_path_portfolio(
    *,
    likelihood: Any,
    path_specs: Sequence[GlobalNoiseSMCInformationPath],
    seeds: Sequence[NoiseAnnealedOrbitSeed] = (),
    config: GlobalNoiseSMCConfig | None = None,
    progress_callback: Callable[[GlobalNoiseSMCPortfolioStage], None] | None = None,
) -> GlobalNoiseSMCPortfolioResult:
    """Run explicit posterior factor orderings without sharing resampling state."""

    active = config or GlobalNoiseSMCConfig()
    ladder = _validated_noise_ladder(active.phase_noise_ladder)
    declared_specs = tuple(path_specs)
    if not declared_specs:
        raise ValueError("A global SMC path portfolio needs at least one path.")
    path_ids = tuple(str(item.path_id).strip() for item in declared_specs)
    if any(not value for value in path_ids) or len(set(path_ids)) != len(path_ids):
        raise ValueError("Global SMC path ids must be non-empty and unique.")
    for path in declared_specs:
        if not path.targets:
            raise ValueError("Every global SMC path must declare a target.")
        for target in path.targets:
            _resolve_information_target(target, ladder=ladder)

    path_results = []
    merged_archive = []
    for path_index, path_spec in enumerate(declared_specs):
        path_id = path_spec.path_id
        initial_noise = next(
            (
                float(target.phase_noise)
                for target in path_spec.targets
                if target.kind in {
                    "island-phase",
                    "partition-phase",
                    "global-phase",
                }
                and target.phase_noise is not None
            ),
            None,
        )
        path_config = replace(
            active,
            initial_phase_noise=None,
            random_seed=max(
                int(active.random_seed) + (path_index + 1) * 104_729,
                0,
            ),
        )

        def report(stage: GlobalNoiseSMCStage) -> None:
            if progress_callback is not None:
                progress_callback(
                    GlobalNoiseSMCPortfolioStage(
                        path_index=path_index,
                        path_id=path_id,
                        initial_phase_noise=initial_noise,
                        stage=stage,
                    )
                )

        result = run_global_noise_smc(
            likelihood=likelihood,
            seeds=seeds,
            config=path_config,
            information_path=path_spec,
            progress_callback=report,
        )
        path_results.append(
            GlobalNoiseSMCPathResult(
                path_index=path_index,
                path_id=path_id,
                initial_phase_noise=initial_noise,
                information_path=path_spec,
                result=result,
            )
        )
        for point in result.archive:
            merged_archive.append(
                GlobalNoiseSMCArchivePoint(
                    point_id=f"global_smc_{path_id}_{point.point_id}",
                    source=f"{path_id}:{point.source}",
                    factor_index=point.factor_index,
                    bridge_fraction=point.bridge_fraction,
                    orbit_normalized_values=point.orbit_normalized_values.copy(),
                    curvature_log_likelihood=point.curvature_log_likelihood,
                    slope_log_likelihood=point.slope_log_likelihood,
                    shape_log_likelihood=point.shape_log_likelihood,
                    partition_integer_log_likelihoods=(
                        point.partition_integer_log_likelihoods.copy()
                    ),
                    island_integer_log_likelihoods=(
                        point.island_integer_log_likelihoods.copy()
                    ),
                    integer_log_likelihoods=point.integer_log_likelihoods.copy(),
                )
            )

    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    portfolio = select_global_noise_smc_modes(
        merged_archive,
        lower=lower,
        upper=upper,
        maximum_modes=active.portfolio_mode_count,
        minimum_separation=active.minimum_mode_separation,
        covariance_floor=active.covariance_eigenvalue_floor,
    )
    return GlobalNoiseSMCPortfolioResult(
        information_paths=tuple(),
        path_specs=declared_specs,
        path_results=tuple(path_results),
        portfolio_modes=portfolio,
        archive=tuple(merged_archive),
        likelihood_call_count=sum(
            item.result.likelihood_call_count for item in path_results
        ),
        reached_exact_endpoint=all(
            item.result.reached_exact_endpoint for item in path_results
        ),
    )


def run_global_noise_smc_portfolio(
    *,
    likelihood: Any,
    information_paths: Sequence[float | None],
    seeds: Sequence[NoiseAnnealedOrbitSeed] = (),
    config: GlobalNoiseSMCConfig | None = None,
    progress_callback: Callable[[GlobalNoiseSMCPortfolioStage], None] | None = None,
) -> GlobalNoiseSMCPortfolioResult:
    """Run independent complete information paths and union only search modes."""

    active = config or GlobalNoiseSMCConfig()
    ladder = _validated_noise_ladder(active.phase_noise_ladder)
    declared_paths = tuple(
        None if value is None else float(value)
        for value in information_paths
    )
    if not declared_paths:
        raise ValueError("A global SMC portfolio needs at least one information path.")
    path_keys: set[str] = set()
    for value in declared_paths:
        if value is None:
            key = "shape"
        else:
            matches = tuple(
                index
                for index, noise in enumerate(ladder)
                if math.isclose(value, noise, rel_tol=1.0e-12, abs_tol=1.0e-15)
            )
            if len(matches) != 1:
                raise ValueError(
                    "Every global SMC information path must start at a declared "
                    "noise-ladder channel or at shape."
                )
            key = f"noise-{matches[0]:04d}"
        if key in path_keys:
            raise ValueError("Global SMC information paths must be unique.")
        path_keys.add(key)

    path_results = []
    merged_archive = []
    path_specs = []
    for initial_noise in declared_paths:
        if initial_noise is None:
            targets = (
                GlobalNoiseSMCInformationTarget("curvature"),
                GlobalNoiseSMCInformationTarget("shape"),
                *(
                    GlobalNoiseSMCInformationTarget("global-phase", noise)
                    for noise in ladder
                ),
            )
            path_id = "shape-first"
        else:
            ladder_index = next(
                index
                for index, noise in enumerate(ladder)
                if math.isclose(
                    initial_noise,
                    noise,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            )
            path_id = f"noise-{initial_noise:.12g}"
            targets = tuple(
                GlobalNoiseSMCInformationTarget("global-phase", ladder[index])
                for index in range(ladder_index, len(ladder))
            )
        path_specs.append(
            GlobalNoiseSMCInformationPath(
                path_id=path_id,
                targets=targets,
            )
        )
    for path_index, (initial_noise, path_spec) in enumerate(
        zip(declared_paths, path_specs, strict=True)
    ):
        path_id = path_spec.path_id
        if initial_noise is None:
            seed_offset = 0
        else:
            ladder_index = next(
                index
                for index, noise in enumerate(ladder)
                if math.isclose(
                    initial_noise,
                    noise,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                )
            )
            seed_offset = (ladder_index + 1) * 104_729
        path_config = replace(
            active,
            initial_phase_noise=initial_noise,
            random_seed=max(int(active.random_seed) + seed_offset, 0),
        )

        def report(stage: GlobalNoiseSMCStage) -> None:
            if progress_callback is not None:
                progress_callback(
                    GlobalNoiseSMCPortfolioStage(
                        path_index=path_index,
                        path_id=path_id,
                        initial_phase_noise=initial_noise,
                        stage=stage,
                    )
                )

        result = run_global_noise_smc(
            likelihood=likelihood,
            seeds=seeds,
            config=path_config,
            information_path=path_spec,
            progress_callback=report,
        )
        path_results.append(
            GlobalNoiseSMCPathResult(
                path_index=path_index,
                path_id=path_id,
                initial_phase_noise=initial_noise,
                information_path=path_spec,
                result=result,
            )
        )
        for point in result.archive:
            merged_archive.append(
                GlobalNoiseSMCArchivePoint(
                    point_id=f"global_smc_{path_id}_{point.point_id}",
                    source=f"{path_id}:{point.source}",
                    factor_index=point.factor_index,
                    bridge_fraction=point.bridge_fraction,
                    orbit_normalized_values=point.orbit_normalized_values.copy(),
                    curvature_log_likelihood=point.curvature_log_likelihood,
                    slope_log_likelihood=point.slope_log_likelihood,
                    shape_log_likelihood=point.shape_log_likelihood,
                    partition_integer_log_likelihoods=(
                        point.partition_integer_log_likelihoods.copy()
                    ),
                    island_integer_log_likelihoods=(
                        point.island_integer_log_likelihoods.copy()
                    ),
                    integer_log_likelihoods=point.integer_log_likelihoods.copy(),
                )
            )

    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    portfolio = select_global_noise_smc_modes(
        merged_archive,
        lower=lower,
        upper=upper,
        maximum_modes=active.portfolio_mode_count,
        minimum_separation=active.minimum_mode_separation,
        covariance_floor=active.covariance_eigenvalue_floor,
    )
    return GlobalNoiseSMCPortfolioResult(
        information_paths=declared_paths,
        path_specs=tuple(path_specs),
        path_results=tuple(path_results),
        portfolio_modes=portfolio,
        archive=tuple(merged_archive),
        likelihood_call_count=sum(
            item.result.likelihood_call_count for item in path_results
        ),
        reached_exact_endpoint=all(
            item.result.reached_exact_endpoint for item in path_results
        ),
    )


__all__ = [
    "GlobalNoiseSMCArchivePoint",
    "GlobalNoiseSMCConfig",
    "GlobalNoiseSMCInformationPath",
    "GlobalNoiseSMCInformationTarget",
    "GlobalNoiseSMCMode",
    "GlobalNoiseSMCPathResult",
    "GlobalNoiseSMCPosteriorPopulation",
    "GlobalNoiseSMCPortfolioResult",
    "GlobalNoiseSMCPortfolioStage",
    "GlobalNoiseSMCResult",
    "GlobalNoiseSMCStage",
    "run_global_noise_smc",
    "run_global_noise_smc_path_portfolio",
    "run_global_noise_smc_portfolio",
    "select_global_noise_smc_modes",
]
