from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import qmc

from apollo_timing.inference.bayesian_conditioned_path_acquisition import (
    ConditionedPathAcquisitionResult,
    ConditionedPathDirectionConfig,
)
from apollo_timing.inference.bayesian_noise_annealed_orbit_transport import (
    NoiseAnnealedOrbitMode,
)


@dataclass(frozen=True)
class NoiseAnnealedIntegerPathConfig:
    phase_noise_ladder: tuple[float, ...] = (
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0,
    )
    paths_per_orbit_mode: int = 8
    round_count: int = 1
    modes_per_channel: int = 24
    maximum_modes_per_path: int = 2
    chart_scale_factors: tuple[float, ...] = (0.3, 3.0, 30.0)
    chart_draws_per_scale: int = 1
    optimizer_starts_per_transport: int = 1
    optimizer_max_evaluations: int = 48
    minimum_mode_separation: float = 1.0e-3
    covariance_eigenvalue_floor: float = 1.0e-12
    conditioned_orbit_modes: int = 8
    conditioned_direction_config: ConditionedPathDirectionConfig | None = field(
        default_factory=ConditionedPathDirectionConfig
    )
    random_seed: int = 20260725


@dataclass(frozen=True)
class NoiseAnnealedIntegerPathMode:
    mode_id: str
    root_seed_id: str
    parent_orbit_mode_id: str
    parent_mode_id: str | None
    source: str
    source_channel_index: int | None
    optimized_channel_index: int | None
    round_index: int
    integer_offsets: tuple[int, ...]
    orbit_normalized_values: np.ndarray
    orbit_covariance: np.ndarray
    channel_log_likelihoods: np.ndarray
    channel_path_log_likelihoods: np.ndarray
    shape_log_likelihood: float
    exact_log_likelihood: float
    exact_path_log_likelihood: float
    exact_spin_posterior_mean: np.ndarray
    exact_spin_posterior_covariance: np.ndarray
    optimizer_evaluations: int
    optimizer_success: bool


@dataclass(frozen=True)
class NoiseAnnealedIntegerPathSummary:
    round_index: int
    channel_index: int
    phase_noise_sigma: float
    candidate_count: int
    retained_count: int
    retained_path_count: int
    retained_root_count: int
    best_channel_log_likelihood: float
    best_exact_log_likelihood: float
    best_shape_log_likelihood: float


@dataclass(frozen=True)
class ConditionedPathAcquisitionRecord:
    root_seed_id: str
    parent_orbit_mode_id: str
    parent_orbit_normalized_values: np.ndarray
    result: ConditionedPathAcquisitionResult


@dataclass(frozen=True)
class NoiseAnnealedIntegerPathResult:
    phase_noise_ladder: tuple[float, ...]
    channel_modes: tuple[tuple[NoiseAnnealedIntegerPathMode, ...], ...]
    portfolio_modes: tuple[NoiseAnnealedIntegerPathMode, ...]
    mode_archive: tuple[NoiseAnnealedIntegerPathMode, ...]
    conditioned_path_acquisitions: tuple[ConditionedPathAcquisitionRecord, ...]
    round_history: tuple[NoiseAnnealedIntegerPathSummary, ...]
    completed_rounds: int
    likelihood_call_count: int


def _validated_ladder(values: Sequence[float]) -> tuple[float, ...]:
    ladder = tuple(float(value) for value in values)
    if (
        not ladder
        or ladder[-1] != 0.0
        or any(left <= right for left, right in zip(ladder, ladder[1:]))
        or not all(np.isfinite(value) and value >= 0.0 for value in ladder)
    ):
        raise ValueError(
            "The fixed-path phase-noise ladder must decrease to exact zero."
        )
    return ladder


def _regularized_covariance(
    covariance: Sequence[Sequence[float]],
    *,
    dimension: int,
    floor: float,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("Fixed-path orbit covariance has the wrong geometry.")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ np.diag(
        np.maximum(eigenvalues, max(float(floor), np.finfo(float).eps))
    ) @ eigenvectors.T


def _inside(values: Sequence[float], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    margin = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
    return np.clip(np.asarray(values, dtype=float), lower + margin, upper - margin)


class _FixedPathEvaluator:
    def __init__(self, likelihood: Any, ladder: tuple[float, ...]) -> None:
        self.likelihood = likelihood
        self.ladder = ladder
        self.scalar_cache: dict[tuple[int, tuple[int, ...], bytes], float] = {}
        self.score_cache: dict[tuple[tuple[int, ...], bytes], tuple[Any, ...]] = {}
        self.call_count = 0

    @staticmethod
    def _point_key(values: Sequence[float]) -> bytes:
        return np.round(np.asarray(values, dtype=np.float64), 13).tobytes()

    def scalar(
        self,
        channel_index: int,
        integer_offsets: tuple[int, ...],
        values: Sequence[float],
    ) -> float:
        point_key = self._point_key(values)
        key = (int(channel_index), integer_offsets, point_key)
        if key not in self.scalar_cache:
            try:
                result = self.likelihood.evaluate_fixed_path(
                    values,
                    integer_offsets,
                    phase_noise_sigma=self.ladder[int(channel_index)],
                )
                value = float(result.log_likelihood)
            except (ArithmeticError, RuntimeError, ValueError):
                value = -math.inf
            self.scalar_cache[key] = value
            self.call_count += 1
        return self.scalar_cache[key]

    def all_scores(
        self,
        integer_offsets: tuple[int, ...],
        values: Sequence[float],
    ) -> tuple[Any, ...]:
        point_key = self._point_key(values)
        key = (integer_offsets, point_key)
        if key not in self.score_cache:
            self.score_cache[key] = self.likelihood.evaluate_fixed_path_noise_channels(
                values,
                integer_offsets,
                self.ladder,
            )
            self.call_count += 1
            for index, evaluation in enumerate(self.score_cache[key]):
                self.scalar_cache[(index, integer_offsets, point_key)] = float(
                    evaluation.log_likelihood
                )
        return self.score_cache[key]


def _make_mode(
    *,
    mode_id: str,
    root_seed_id: str,
    parent_orbit_mode_id: str,
    parent_mode_id: str | None,
    source: str,
    source_channel_index: int | None,
    optimized_channel_index: int | None,
    round_index: int,
    integer_offsets: tuple[int, ...],
    orbit_values: np.ndarray,
    covariance: np.ndarray,
    evaluations: Sequence[Any],
    optimizer_evaluations: int,
    optimizer_success: bool,
) -> NoiseAnnealedIntegerPathMode:
    exact = evaluations[-1]
    return NoiseAnnealedIntegerPathMode(
        mode_id=mode_id,
        root_seed_id=root_seed_id,
        parent_orbit_mode_id=parent_orbit_mode_id,
        parent_mode_id=parent_mode_id,
        source=source,
        source_channel_index=source_channel_index,
        optimized_channel_index=optimized_channel_index,
        round_index=int(round_index),
        integer_offsets=integer_offsets,
        orbit_normalized_values=np.asarray(orbit_values, dtype=float),
        orbit_covariance=np.asarray(covariance, dtype=float),
        channel_log_likelihoods=np.asarray(
            [item.log_likelihood for item in evaluations], dtype=float
        ),
        channel_path_log_likelihoods=np.asarray(
            [item.path_log_likelihood for item in evaluations], dtype=float
        ),
        shape_log_likelihood=float(exact.shape_log_likelihood),
        exact_log_likelihood=float(exact.log_likelihood),
        exact_path_log_likelihood=float(exact.path_log_likelihood),
        exact_spin_posterior_mean=np.asarray(
            exact.spin_posterior_mean, dtype=float
        ),
        exact_spin_posterior_covariance=np.asarray(
            exact.spin_posterior_covariance, dtype=float
        ),
        optimizer_evaluations=max(int(optimizer_evaluations), 0),
        optimizer_success=bool(optimizer_success),
    )


def _conditioned_orbit_parents(
    orbit_channel_modes: Sequence[Sequence[NoiseAnnealedOrbitMode]],
    maximum_modes: int,
) -> tuple[NoiseAnnealedOrbitMode, ...]:
    unique = {}
    for modes in orbit_channel_modes:
        for mode in modes:
            key = (
                mode.root_seed_id,
                np.round(mode.orbit_normalized_values, 12).tobytes(),
            )
            unique.setdefault(key, mode)
    modes = tuple(unique.values())
    if not modes:
        return ()
    rankings = [
        tuple(sorted(modes, key=lambda mode: -mode.exact_log_likelihood)),
        tuple(sorted(modes, key=lambda mode: -mode.shape_log_likelihood)),
        tuple(sorted(modes, key=lambda mode: -mode.exact_integer_log_likelihood)),
    ]
    channel_count = int(modes[0].channel_log_likelihoods.size)
    rankings.extend(
        tuple(
            sorted(
                modes,
                key=lambda mode, target=index: -float(
                    mode.channel_log_likelihoods[target]
                ),
            )
        )
        for index in range(channel_count)
    )
    selected = []
    selected_ids = set()
    target = max(int(maximum_modes), 1)
    for rank in range(max(len(ranking) for ranking in rankings)):
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            mode = ranking[rank]
            key = (
                mode.root_seed_id,
                np.round(mode.orbit_normalized_values, 12).tobytes(),
            )
            if key in selected_ids:
                continue
            selected.append(mode)
            selected_ids.add(key)
            if len(selected) >= target:
                return tuple(selected)
    return tuple(selected)


def _rankings(
    modes: Sequence[NoiseAnnealedIntegerPathMode],
    channel_index: int,
) -> tuple[tuple[NoiseAnnealedIntegerPathMode, ...], ...]:
    rankings = [
        tuple(
            sorted(
                modes,
                key=lambda mode: -float(mode.channel_log_likelihoods[channel_index]),
            )
        ),
        tuple(sorted(modes, key=lambda mode: -mode.exact_log_likelihood)),
        tuple(sorted(modes, key=lambda mode: -mode.shape_log_likelihood)),
        tuple(sorted(modes, key=lambda mode: -mode.exact_path_log_likelihood)),
    ]
    channel_count = int(modes[0].channel_log_likelihoods.size)
    for index in range(channel_count):
        if index in (channel_index, channel_count - 1):
            continue
        rankings.append(
            tuple(
                sorted(
                    modes,
                    key=lambda mode, target=index: -float(
                        mode.channel_log_likelihoods[target]
                    ),
                )
            )
        )
    return tuple(rankings)


def select_noise_annealed_integer_path_modes(
    modes: Sequence[NoiseAnnealedIntegerPathMode],
    *,
    channel_index: int,
    maximum_modes: int,
    maximum_modes_per_path: int,
    lower: Sequence[float],
    upper: Sequence[float],
    minimum_separation: float,
) -> tuple[NoiseAnnealedIntegerPathMode, ...]:
    if not modes:
        return ()
    target = max(int(maximum_modes), 1)
    path_cap = max(int(maximum_modes_per_path), 1)
    lower_values = np.asarray(lower, dtype=float)
    width = np.asarray(upper, dtype=float) - lower_values
    separation = max(float(minimum_separation), 0.0)
    rankings = _rankings(modes, int(channel_index))
    selected: list[NoiseAnnealedIntegerPathMode] = []
    path_counts: dict[tuple[int, ...], int] = {}

    def add(candidate: NoiseAnnealedIntegerPathMode, enforce_separation: bool) -> bool:
        path = candidate.integer_offsets
        if path_counts.get(path, 0) >= path_cap:
            return False
        point = (candidate.orbit_normalized_values - lower_values) / width
        for existing in selected:
            existing_point = (
                existing.orbit_normalized_values - lower_values
            ) / width
            distance = float(np.linalg.norm(point - existing_point))
            if path == existing.integer_offsets and distance == 0.0:
                return False
            if enforce_separation and distance < separation:
                return False
        selected.append(candidate)
        path_counts[path] = path_counts.get(path, 0) + 1
        return True

    for ranking in rankings[:4]:
        for candidate in ranking:
            if add(candidate, False):
                break
        if len(selected) >= target:
            return tuple(selected)
    interleaved = []
    for rank in range(max(len(ranking) for ranking in rankings)):
        for ranking in rankings:
            if rank < len(ranking):
                interleaved.append(ranking[rank])
    for enforce in (True, False):
        for candidate in interleaved:
            add(candidate, enforce)
            if len(selected) >= target:
                return tuple(selected)
    return tuple(selected)


def _chart_starts(
    mode: NoiseAnnealedIntegerPathMode,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    config: NoiseAnnealedIntegerPathConfig,
    round_index: int,
    source_channel_index: int,
    destination_channel_index: int,
) -> tuple[np.ndarray, ...]:
    starts = [_inside(mode.orbit_normalized_values, lower, upper)]
    scales = tuple(
        float(value)
        for value in config.chart_scale_factors
        if np.isfinite(float(value)) and float(value) > 0.0
    )
    draws_per_scale = max(int(config.chart_draws_per_scale), 0)
    if not scales or draws_per_scale == 0:
        return tuple(starts)
    dimension = int(lower.size)
    covariance = _regularized_covariance(
        mode.orbit_covariance,
        dimension=dimension,
        floor=config.covariance_eigenvalue_floor,
    )
    root = np.linalg.cholesky(covariance)
    count = len(scales) * draws_per_scale
    exponent = int(math.ceil(math.log2(max(count, 1))))
    seed = (
        max(int(config.random_seed), 0)
        + 1_000_003 * int(round_index)
        + 10_007 * int(source_channel_index)
        + 101 * int(destination_channel_index)
        + sum(ord(value) for value in mode.mode_id)
    )
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)
    normal = ndtri(np.clip(unit[:count], 1.0e-12, 1.0 - 1.0e-12))
    draw_index = 0
    for scale in scales:
        for _ in range(draws_per_scale):
            starts.append(
                _inside(
                    mode.orbit_normalized_values
                    + float(scale) * (root @ normal[draw_index]),
                    lower,
                    upper,
                )
            )
            draw_index += 1
    return tuple(starts)


def _transport_mode(
    mode: NoiseAnnealedIntegerPathMode,
    *,
    source_channel_index: int,
    destination_channel_index: int,
    round_index: int,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluator: _FixedPathEvaluator,
    config: NoiseAnnealedIntegerPathConfig,
    counter: list[int],
) -> tuple[NoiseAnnealedIntegerPathMode, ...]:
    chart_modes = []
    starts_by_id = {}
    for start_index, start in enumerate(
        _chart_starts(
            mode,
            lower=lower,
            upper=upper,
            config=config,
            round_index=round_index,
            source_channel_index=source_channel_index,
            destination_channel_index=destination_channel_index,
        )
    ):
        try:
            evaluations = evaluator.all_scores(mode.integer_offsets, start)
        except (ArithmeticError, RuntimeError, ValueError):
            continue
        displacement = start - mode.orbit_normalized_values
        covariance = mode.orbit_covariance + np.outer(displacement, displacement)
        mode_id = f"noise_path_{counter[0]:08d}"
        counter[0] += 1
        child = _make_mode(
            mode_id=mode_id,
            root_seed_id=mode.root_seed_id,
            parent_orbit_mode_id=mode.parent_orbit_mode_id,
            parent_mode_id=mode.mode_id,
            source=f"fixed-path-chart-screen-{start_index:03d}",
            source_channel_index=source_channel_index,
            optimized_channel_index=destination_channel_index,
            round_index=round_index,
            integer_offsets=mode.integer_offsets,
            orbit_values=start,
            covariance=covariance,
            evaluations=evaluations,
            optimizer_evaluations=0,
            optimizer_success=True,
        )
        chart_modes.append(child)
        starts_by_id[mode_id] = start
    if not chart_modes:
        return ()
    optimization_seeds = select_noise_annealed_integer_path_modes(
        chart_modes,
        channel_index=destination_channel_index,
        maximum_modes=max(int(config.optimizer_starts_per_transport), 1),
        maximum_modes_per_path=max(int(config.optimizer_starts_per_transport), 1),
        lower=lower,
        upper=upper,
        minimum_separation=0.0,
    )
    transported = list(chart_modes)
    for start_index, chart_mode in enumerate(optimization_seeds):
        optimization = minimize(
            lambda values: -evaluator.scalar(
                destination_channel_index,
                mode.integer_offsets,
                values,
            ),
            starts_by_id[chart_mode.mode_id],
            method="Powell",
            bounds=tuple(zip(lower, upper, strict=True)),
            options={
                "maxfev": max(int(config.optimizer_max_evaluations), 1),
                "xtol": 1.0e-6,
                "ftol": 1.0e-7,
            },
        )
        point = _inside(optimization.x, lower, upper)
        try:
            evaluations = evaluator.all_scores(mode.integer_offsets, point)
        except (ArithmeticError, RuntimeError, ValueError):
            continue
        mode_id = f"noise_path_{counter[0]:08d}"
        counter[0] += 1
        transported.append(
            _make_mode(
                mode_id=mode_id,
                root_seed_id=mode.root_seed_id,
                parent_orbit_mode_id=mode.parent_orbit_mode_id,
                parent_mode_id=mode.mode_id,
                source=f"fixed-path-chart-optimize-{start_index:03d}",
                source_channel_index=source_channel_index,
                optimized_channel_index=destination_channel_index,
                round_index=round_index,
                integer_offsets=mode.integer_offsets,
                orbit_values=point,
                covariance=chart_mode.orbit_covariance,
                evaluations=evaluations,
                optimizer_evaluations=int(getattr(optimization, "nfev", 0)),
                optimizer_success=bool(optimization.success),
            )
        )
    return tuple(transported)


def run_noise_annealed_integer_path_transport(
    *,
    likelihood: Any,
    orbit_channel_modes: Sequence[Sequence[NoiseAnnealedOrbitMode]],
    config: NoiseAnnealedIntegerPathConfig | None = None,
    checkpoint_callback: Callable[
        [
            int,
            tuple[tuple[NoiseAnnealedIntegerPathMode, ...], ...],
            tuple[NoiseAnnealedIntegerPathSummary, ...],
        ],
        None,
    ]
    | None = None,
) -> NoiseAnnealedIntegerPathResult:
    """Decompose orbit mixtures and transport complete path components."""

    active = config or NoiseAnnealedIntegerPathConfig()
    ladder = _validated_ladder(active.phase_noise_ladder)
    if len(orbit_channel_modes) != len(ladder):
        raise ValueError("Every noise channel requires an orbit-mode portfolio.")
    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    evaluator = _FixedPathEvaluator(likelihood, ladder)
    archive = []
    conditioned_acquisitions = []
    counter = [0]
    for channel_index, orbit_modes in enumerate(orbit_channel_modes):
        channel_likelihood = likelihood.with_phase_noise(ladder[channel_index])
        for orbit_mode in orbit_modes:
            marginal = channel_likelihood.evaluate(
                orbit_mode.orbit_normalized_values
            )
            for path_mode in marginal.path_modes[
                : max(int(active.paths_per_orbit_mode), 1)
            ]:
                path = tuple(int(value) for value in path_mode.integer_offsets)
                evaluations = evaluator.all_scores(
                    path,
                    orbit_mode.orbit_normalized_values,
                )
                archive.append(
                    _make_mode(
                        mode_id=f"noise_path_{counter[0]:08d}",
                        root_seed_id=orbit_mode.root_seed_id,
                        parent_orbit_mode_id=orbit_mode.mode_id,
                        parent_mode_id=None,
                        source=f"marginal-channel-{channel_index:02d}",
                        source_channel_index=channel_index,
                        optimized_channel_index=None,
                        round_index=-1,
                        integer_offsets=path,
                        orbit_values=orbit_mode.orbit_normalized_values,
                        covariance=orbit_mode.orbit_covariance,
                        evaluations=evaluations,
                        optimizer_evaluations=0,
                        optimizer_success=True,
                    )
                )
                counter[0] += 1
    direction_config = active.conditioned_direction_config
    supports_conditioned_acquisition = all(
        hasattr(likelihood, name)
        for name in (
            "acquire_conditioned_paths",
            "orbit_values_from_full_normalized",
            "orbit_covariance_from_full",
        )
    )
    if direction_config is not None and supports_conditioned_acquisition:
        parents = _conditioned_orbit_parents(
            orbit_channel_modes,
            active.conditioned_orbit_modes,
        )
        for orbit_mode in parents:
            try:
                marginal = likelihood.evaluate(
                    orbit_mode.orbit_normalized_values
                )
                seed_paths = tuple(
                    path_mode.integer_offsets
                    for path_mode in marginal.path_modes[
                        : max(int(active.paths_per_orbit_mode), 1)
                    ]
                )
                acquisition = likelihood.acquire_conditioned_paths(
                    orbit_mode.orbit_normalized_values,
                    orbit_mode.orbit_covariance,
                    seed_paths=seed_paths,
                    config=direction_config,
                )
            except (ArithmeticError, RuntimeError, ValueError):
                continue
            conditioned_acquisitions.append(
                ConditionedPathAcquisitionRecord(
                    root_seed_id=orbit_mode.root_seed_id,
                    parent_orbit_mode_id=orbit_mode.mode_id,
                    parent_orbit_normalized_values=np.asarray(
                        orbit_mode.orbit_normalized_values,
                        dtype=float,
                    ),
                    result=acquisition,
                )
            )
            for proposal in acquisition.proposals:
                path = tuple(int(value) for value in proposal.integer_offsets)
                try:
                    orbit_values = likelihood.orbit_values_from_full_normalized(
                        proposal.conditioned_normalized_values
                    )
                    orbit_values = _inside(orbit_values, lower, upper)
                    covariance = likelihood.orbit_covariance_from_full(
                        proposal.conditioned_covariance
                    )
                    covariance = _regularized_covariance(
                        covariance,
                        dimension=lower.size,
                        floor=active.covariance_eigenvalue_floor,
                    )
                    evaluations = evaluator.all_scores(path, orbit_values)
                except (ArithmeticError, RuntimeError, ValueError):
                    continue
                archive.append(
                    _make_mode(
                        mode_id=f"noise_path_{counter[0]:08d}",
                        root_seed_id=orbit_mode.root_seed_id,
                        parent_orbit_mode_id=orbit_mode.mode_id,
                        parent_mode_id=None,
                        source=(
                            "conditioned-joint-direction-"
                            + proposal.conditioning_channel
                        ),
                        source_channel_index=None,
                        optimized_channel_index=None,
                        round_index=-1,
                        integer_offsets=path,
                        orbit_values=orbit_values,
                        covariance=covariance,
                        evaluations=evaluations,
                        optimizer_evaluations=0,
                        optimizer_success=True,
                    )
                )
                counter[0] += 1
    if not archive:
        raise RuntimeError("No complete integer path component was recovered.")
    channel_modes = tuple(
        select_noise_annealed_integer_path_modes(
            archive,
            channel_index=index,
            maximum_modes=active.modes_per_channel,
            maximum_modes_per_path=active.maximum_modes_per_path,
            lower=lower,
            upper=upper,
            minimum_separation=active.minimum_mode_separation,
        )
        for index in range(len(ladder))
    )
    history = []
    for round_index in range(max(int(active.round_count), 0)):
        additions = [[] for _ in ladder]
        for source_index, modes in enumerate(channel_modes):
            destinations = []
            if source_index > 0:
                destinations.append(source_index - 1)
            if source_index + 1 < len(ladder):
                destinations.append(source_index + 1)
            for destination_index in destinations:
                for mode in modes:
                    additions[destination_index].extend(
                        _transport_mode(
                            mode,
                            source_channel_index=source_index,
                            destination_channel_index=destination_index,
                            round_index=round_index,
                            lower=lower,
                            upper=upper,
                            evaluator=evaluator,
                            config=active,
                            counter=counter,
                        )
                    )
        next_channels = []
        summaries = []
        for channel_index, (current, children) in enumerate(
            zip(channel_modes, additions, strict=True)
        ):
            pool = tuple(current) + tuple(children)
            selected = select_noise_annealed_integer_path_modes(
                pool,
                channel_index=channel_index,
                maximum_modes=active.modes_per_channel,
                maximum_modes_per_path=active.maximum_modes_per_path,
                lower=lower,
                upper=upper,
                minimum_separation=active.minimum_mode_separation,
            )
            next_channels.append(selected)
            archive.extend(children)
            summary = NoiseAnnealedIntegerPathSummary(
                round_index=round_index,
                channel_index=channel_index,
                phase_noise_sigma=ladder[channel_index],
                candidate_count=len(pool),
                retained_count=len(selected),
                retained_path_count=len({mode.integer_offsets for mode in selected}),
                retained_root_count=len({mode.root_seed_id for mode in selected}),
                best_channel_log_likelihood=max(
                    float(mode.channel_log_likelihoods[channel_index])
                    for mode in selected
                ),
                best_exact_log_likelihood=max(
                    mode.exact_log_likelihood for mode in selected
                ),
                best_shape_log_likelihood=max(
                    mode.shape_log_likelihood for mode in selected
                ),
            )
            history.append(summary)
            summaries.append(summary)
        channel_modes = tuple(next_channels)
        if checkpoint_callback is not None:
            checkpoint_callback(round_index, channel_modes, tuple(summaries))
    union = tuple(mode for modes in channel_modes for mode in modes)
    portfolio = select_noise_annealed_integer_path_modes(
        union,
        channel_index=len(ladder) - 1,
        maximum_modes=max(int(active.modes_per_channel), 1) * len(ladder),
        maximum_modes_per_path=active.maximum_modes_per_path,
        lower=lower,
        upper=upper,
        minimum_separation=active.minimum_mode_separation,
    )
    return NoiseAnnealedIntegerPathResult(
        phase_noise_ladder=ladder,
        channel_modes=channel_modes,
        portfolio_modes=portfolio,
        mode_archive=tuple(archive),
        conditioned_path_acquisitions=tuple(conditioned_acquisitions),
        round_history=tuple(history),
        completed_rounds=max(int(active.round_count), 0),
        likelihood_call_count=evaluator.call_count,
    )


__all__ = [
    "ConditionedPathAcquisitionRecord",
    "NoiseAnnealedIntegerPathConfig",
    "NoiseAnnealedIntegerPathMode",
    "NoiseAnnealedIntegerPathResult",
    "NoiseAnnealedIntegerPathSummary",
    "run_noise_annealed_integer_path_transport",
    "select_noise_annealed_integer_path_modes",
]
