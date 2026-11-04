from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import qmc


@dataclass(frozen=True)
class NoiseAnnealedOrbitConfig:
    """Controls for non-greedy transport across parallel phase-noise channels."""

    phase_noise_ladder: tuple[float, ...] = (
        0.03,
        0.01,
        0.003,
        0.001,
        0.0003,
        0.0,
    )
    round_count: int = 2
    modes_per_channel: int = 12
    chart_scale_factors: tuple[float, ...] = (
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
    )
    chart_draws_per_scale: int = 1
    optimizer_starts_per_transport: int = 2
    optimizer_max_evaluations: int = 96
    minimum_mode_separation: float = 2.0e-3
    covariance_eigenvalue_floor: float = 1.0e-12
    random_seed: int = 20260725


@dataclass(frozen=True)
class NoiseAnnealedOrbitSeed:
    seed_id: str
    orbit_normalized_values: np.ndarray
    orbit_covariance: np.ndarray
    source: str


@dataclass(frozen=True)
class NoiseAnnealedOrbitMode:
    mode_id: str
    root_seed_id: str
    parent_mode_id: str | None
    source: str
    source_channel_index: int | None
    optimized_channel_index: int | None
    round_index: int
    orbit_normalized_values: np.ndarray
    orbit_covariance: np.ndarray
    channel_log_likelihoods: np.ndarray
    channel_integer_log_likelihoods: np.ndarray
    shape_log_likelihood: float
    exact_log_likelihood: float
    exact_integer_log_likelihood: float
    optimizer_evaluations: int
    optimizer_success: bool


@dataclass(frozen=True)
class NoiseAnnealedChannelSummary:
    round_index: int
    channel_index: int
    phase_noise_sigma: float
    candidate_count: int
    retained_count: int
    retained_root_count: int
    best_channel_log_likelihood: float
    best_exact_log_likelihood: float
    best_shape_log_likelihood: float
    best_exact_integer_log_likelihood: float


@dataclass(frozen=True)
class NoiseAnnealedOrbitResult:
    phase_noise_ladder: tuple[float, ...]
    channel_modes: tuple[tuple[NoiseAnnealedOrbitMode, ...], ...]
    portfolio_modes: tuple[NoiseAnnealedOrbitMode, ...]
    mode_archive: tuple[NoiseAnnealedOrbitMode, ...]
    round_history: tuple[NoiseAnnealedChannelSummary, ...]
    completed_rounds: int
    likelihood_call_count: int


def _validated_noise_ladder(values: Sequence[float]) -> tuple[float, ...]:
    ladder = tuple(float(value) for value in values)
    if not ladder or not all(
        np.isfinite(value) and value >= 0.0 for value in ladder
    ):
        raise ValueError("The phase-noise ladder must be finite and non-negative.")
    if len(set(ladder)) != len(ladder):
        raise ValueError("The phase-noise ladder cannot contain duplicates.")
    if any(left <= right for left, right in zip(ladder, ladder[1:])):
        raise ValueError("The phase-noise ladder must decrease toward zero.")
    if ladder[-1] != 0.0:
        raise ValueError("The phase-noise ladder must end at the exact zero-noise target.")
    return ladder


def _regularized_covariance(
    covariance: Sequence[Sequence[float]],
    *,
    dimension: int,
    eigenvalue_floor: float,
) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError("Every orbit seed requires a finite square covariance.")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    floor = max(float(eigenvalue_floor), np.finfo(float).eps)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _inside_bounds(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    margin = np.maximum((upper - lower) * 1.0e-10, 1.0e-12)
    return np.clip(values, lower + margin, upper - margin)


class _ChannelEvaluator:
    def __init__(
        self,
        likelihood: Any,
        phase_noise_ladder: tuple[float, ...],
    ) -> None:
        self.base_likelihood = likelihood
        self.phase_noise_ladder = phase_noise_ladder
        self.channel_likelihoods = tuple(
            likelihood.with_phase_noise(sigma) for sigma in phase_noise_ladder
        )
        self.scalar_cache: dict[tuple[int, bytes], float] = {}
        self.score_cache: dict[bytes, tuple[Any, ...]] = {}
        self.call_count = 0

    @staticmethod
    def _key(values: Sequence[float]) -> bytes:
        point = np.asarray(values, dtype=np.float64)
        return np.round(point, decimals=13).tobytes()

    def scalar(self, channel_index: int, values: Sequence[float]) -> float:
        key = (int(channel_index), self._key(values))
        if key not in self.scalar_cache:
            self.scalar_cache[key] = float(
                self.channel_likelihoods[int(channel_index)](values)
            )
            self.call_count += 1
        return self.scalar_cache[key]

    def all_scores(self, values: Sequence[float]) -> tuple[Any, ...]:
        key = self._key(values)
        if key not in self.score_cache:
            self.score_cache[key] = tuple(
                self.base_likelihood.evaluate_noise_channels(
                    values,
                    self.phase_noise_ladder,
                )
            )
            self.call_count += 1
            for index, evaluation in enumerate(self.score_cache[key]):
                self.scalar_cache[(index, key)] = float(evaluation.log_likelihood)
        return self.score_cache[key]


def _make_mode(
    *,
    mode_id: str,
    root_seed_id: str,
    parent_mode_id: str | None,
    source: str,
    source_channel_index: int | None,
    optimized_channel_index: int | None,
    round_index: int,
    orbit_values: np.ndarray,
    covariance: np.ndarray,
    evaluations: Sequence[Any],
    optimizer_evaluations: int,
    optimizer_success: bool,
) -> NoiseAnnealedOrbitMode:
    channel_log_likelihoods = np.asarray(
        [item.log_likelihood for item in evaluations],
        dtype=float,
    )
    integer_log_likelihoods = np.asarray(
        [item.integer_log_likelihood for item in evaluations],
        dtype=float,
    )
    exact = evaluations[-1]
    return NoiseAnnealedOrbitMode(
        mode_id=mode_id,
        root_seed_id=root_seed_id,
        parent_mode_id=parent_mode_id,
        source=source,
        source_channel_index=source_channel_index,
        optimized_channel_index=optimized_channel_index,
        round_index=int(round_index),
        orbit_normalized_values=np.asarray(orbit_values, dtype=float),
        orbit_covariance=np.asarray(covariance, dtype=float),
        channel_log_likelihoods=channel_log_likelihoods,
        channel_integer_log_likelihoods=integer_log_likelihoods,
        shape_log_likelihood=float(exact.shape_log_likelihood),
        exact_log_likelihood=float(exact.log_likelihood),
        exact_integer_log_likelihood=float(exact.integer_log_likelihood),
        optimizer_evaluations=max(int(optimizer_evaluations), 0),
        optimizer_success=bool(optimizer_success),
    )


def _mode_rankings(
    modes: Sequence[NoiseAnnealedOrbitMode],
    channel_index: int,
) -> tuple[tuple[NoiseAnnealedOrbitMode, ...], ...]:
    rankings = [
        tuple(
            sorted(
                modes,
                key=lambda mode: -float(mode.channel_log_likelihoods[channel_index]),
            )
        ),
        tuple(sorted(modes, key=lambda mode: -mode.exact_log_likelihood)),
        tuple(sorted(modes, key=lambda mode: -mode.shape_log_likelihood)),
        tuple(
            sorted(
                modes,
                key=lambda mode: -mode.exact_integer_log_likelihood,
            )
        ),
    ]
    channel_count = int(modes[0].channel_log_likelihoods.size)
    for index in range(channel_count):
        if index in (int(channel_index), channel_count - 1):
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
    unique = []
    seen_orders: set[tuple[str, ...]] = set()
    for ranking in rankings:
        order = tuple(mode.mode_id for mode in ranking)
        if order in seen_orders:
            continue
        seen_orders.add(order)
        unique.append(ranking)
    return tuple(unique)


def select_noise_annealed_orbit_modes(
    modes: Sequence[NoiseAnnealedOrbitMode],
    *,
    channel_index: int,
    maximum_modes: int,
    lower: Sequence[float],
    upper: Sequence[float],
    minimum_separation: float,
) -> tuple[NoiseAnnealedOrbitMode, ...]:
    """Retain independent evidence channels before filling spatial coverage."""

    if not modes:
        return ()
    target = max(int(maximum_modes), 1)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    width = upper_values - lower_values
    separation = max(float(minimum_separation), 0.0)
    rankings = _mode_rankings(modes, int(channel_index))
    selected: list[NoiseAnnealedOrbitMode] = []
    seen_points: list[np.ndarray] = []

    def distance(candidate: NoiseAnnealedOrbitMode) -> float:
        point = (candidate.orbit_normalized_values - lower_values) / width
        if not seen_points:
            return math.inf
        return min(float(np.linalg.norm(point - existing)) for existing in seen_points)

    def add(candidate: NoiseAnnealedOrbitMode, *, enforce_separation: bool) -> bool:
        point = (candidate.orbit_normalized_values - lower_values) / width
        if any(np.array_equal(point, existing) for existing in seen_points):
            return False
        if enforce_separation and distance(candidate) < separation:
            return False
        selected.append(candidate)
        seen_points.append(point)
        return True

    # These four insertions guarantee independent acquisition without
    # combining incompatible evidence scales into a weighted score.
    for ranking in rankings:
        add(ranking[0], enforce_separation=False)
        if len(selected) >= target:
            return tuple(selected)

    interleaved: list[NoiseAnnealedOrbitMode] = []
    for rank in range(max(len(ranking) for ranking in rankings)):
        for ranking in rankings:
            if rank < len(ranking):
                interleaved.append(ranking[rank])
    for candidate in interleaved:
        add(candidate, enforce_separation=True)
        if len(selected) >= target:
            return tuple(selected)
    # A very narrow posterior can place several evidence modes inside the
    # nominal diversity radius. Fill remaining finite-budget slots by rank,
    # without replacing them with distant points supported by weaker evidence.
    for candidate in interleaved:
        add(candidate, enforce_separation=False)
        if len(selected) >= target:
            break
    return tuple(selected)


def _chart_starts(
    mode: NoiseAnnealedOrbitMode,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    config: NoiseAnnealedOrbitConfig,
    round_index: int,
    source_channel_index: int,
    destination_channel_index: int,
) -> tuple[np.ndarray, ...]:
    starts = [
        _inside_bounds(mode.orbit_normalized_values, lower, upper)
    ]
    draws_per_scale = max(int(config.chart_draws_per_scale), 0)
    scales = tuple(
        float(value)
        for value in config.chart_scale_factors
        if np.isfinite(float(value)) and float(value) > 0.0
    )
    if draws_per_scale == 0 or not scales:
        return tuple(starts)
    dimension = int(lower.size)
    covariance = _regularized_covariance(
        mode.orbit_covariance,
        dimension=dimension,
        eigenvalue_floor=config.covariance_eigenvalue_floor,
    )
    root = np.linalg.cholesky(covariance)
    draw_count = draws_per_scale * len(scales)
    exponent = int(math.ceil(math.log2(max(draw_count, 1))))
    seed = (
        max(int(config.random_seed), 0)
        + 1_000_003 * int(round_index)
        + 10_007 * int(source_channel_index)
        + 101 * int(destination_channel_index)
        + sum(ord(value) for value in mode.mode_id)
    )
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)
    normal = ndtri(np.clip(unit[:draw_count], 1.0e-12, 1.0 - 1.0e-12))
    draw_index = 0
    for scale in scales:
        for _ in range(draws_per_scale):
            proposal = mode.orbit_normalized_values + float(scale) * (
                root @ normal[draw_index]
            )
            starts.append(_inside_bounds(proposal, lower, upper))
            draw_index += 1
    return tuple(starts)


def _transport_mode(
    mode: NoiseAnnealedOrbitMode,
    *,
    source_channel_index: int,
    destination_channel_index: int,
    round_index: int,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluator: _ChannelEvaluator,
    config: NoiseAnnealedOrbitConfig,
    mode_counter: list[int],
) -> tuple[NoiseAnnealedOrbitMode, ...]:
    transported = []
    starts = _chart_starts(
        mode,
        lower=lower,
        upper=upper,
        config=config,
        round_index=round_index,
        source_channel_index=source_channel_index,
        destination_channel_index=destination_channel_index,
    )
    chart_modes = []
    chart_mode_starts: dict[str, np.ndarray] = {}
    for start_index, start in enumerate(starts):
        try:
            evaluations = evaluator.all_scores(start)
        except (ArithmeticError, RuntimeError, ValueError):
            continue
        displacement = np.asarray(start, dtype=float) - mode.orbit_normalized_values
        chart_covariance = mode.orbit_covariance + np.outer(
            displacement,
            displacement,
        )
        mode_index = mode_counter[0]
        mode_counter[0] += 1
        chart_mode = _make_mode(
            mode_id=f"noise_orbit_{mode_index:07d}",
            root_seed_id=mode.root_seed_id,
            parent_mode_id=mode.mode_id,
            source=f"adjacent-chart-screen-{start_index:03d}",
            source_channel_index=source_channel_index,
            optimized_channel_index=destination_channel_index,
            round_index=round_index,
            orbit_values=start,
            covariance=chart_covariance,
            evaluations=evaluations,
            optimizer_evaluations=0,
            optimizer_success=True,
        )
        chart_modes.append(chart_mode)
        chart_mode_starts[chart_mode.mode_id] = start
    if not chart_modes:
        return ()
    optimization_seeds = select_noise_annealed_orbit_modes(
        chart_modes,
        channel_index=destination_channel_index,
        maximum_modes=max(int(config.optimizer_starts_per_transport), 1),
        lower=lower,
        upper=upper,
        minimum_separation=0.0,
    )
    transported.extend(chart_modes)
    for start_index, chart_mode in enumerate(optimization_seeds):
        start = chart_mode_starts[chart_mode.mode_id]
        optimization = minimize(
            lambda values: -evaluator.scalar(destination_channel_index, values),
            start,
            method="Powell",
            bounds=tuple(zip(lower, upper, strict=True)),
            options={
                "maxfev": max(int(config.optimizer_max_evaluations), 1),
                "xtol": 1.0e-6,
                "ftol": 1.0e-7,
            },
        )
        point = _inside_bounds(np.asarray(optimization.x, dtype=float), lower, upper)
        try:
            evaluations = evaluator.all_scores(point)
        except (ArithmeticError, RuntimeError, ValueError):
            continue
        mode_index = mode_counter[0]
        mode_counter[0] += 1
        transported.append(
            _make_mode(
                mode_id=f"noise_orbit_{mode_index:07d}",
                root_seed_id=mode.root_seed_id,
                parent_mode_id=mode.mode_id,
                source=f"adjacent-chart-optimize-{start_index:03d}",
                source_channel_index=source_channel_index,
                optimized_channel_index=destination_channel_index,
                round_index=round_index,
                orbit_values=point,
                covariance=chart_mode.orbit_covariance,
                evaluations=evaluations,
                optimizer_evaluations=int(getattr(optimization, "nfev", 0)),
                optimizer_success=bool(optimization.success),
            )
        )
    return tuple(transported)


def _portfolio_union(
    channel_modes: Sequence[Sequence[NoiseAnnealedOrbitMode]],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    maximum_modes: int,
    minimum_separation: float,
) -> tuple[NoiseAnnealedOrbitMode, ...]:
    all_modes = tuple(mode for modes in channel_modes for mode in modes)
    exact_index = len(channel_modes) - 1
    return select_noise_annealed_orbit_modes(
        all_modes,
        channel_index=exact_index,
        maximum_modes=maximum_modes,
        lower=lower,
        upper=upper,
        minimum_separation=minimum_separation,
    )


def run_noise_annealed_orbit_transport(
    *,
    likelihood: Any,
    seeds: Sequence[NoiseAnnealedOrbitSeed] = (),
    initial_modes: Sequence[NoiseAnnealedOrbitMode] = (),
    config: NoiseAnnealedOrbitConfig | None = None,
    start_round: int = 0,
    checkpoint_callback: Callable[
        [
            int,
            tuple[tuple[NoiseAnnealedOrbitMode, ...], ...],
            tuple[NoiseAnnealedChannelSummary, ...],
        ],
        None,
    ]
    | None = None,
) -> NoiseAnnealedOrbitResult:
    """Transport a multimodal orbit portfolio without a greedy noise path."""

    active = config or NoiseAnnealedOrbitConfig()
    ladder = _validated_noise_ladder(active.phase_noise_ladder)
    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape or not np.all(lower < upper):
        raise ValueError("Orbit support must contain matching ordered bounds.")
    dimension = int(lower.size)
    evaluator = _ChannelEvaluator(likelihood, ladder)
    archive: list[NoiseAnnealedOrbitMode] = list(initial_modes)
    existing_indices = []
    for mode in archive:
        try:
            existing_indices.append(int(mode.mode_id.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    mode_counter = [max(existing_indices, default=-1) + 1]

    for seed in seeds:
        point = _inside_bounds(
            np.asarray(seed.orbit_normalized_values, dtype=float),
            lower,
            upper,
        )
        if point.shape != (dimension,):
            raise ValueError("Orbit seed coordinates have the wrong dimension.")
        covariance = _regularized_covariance(
            seed.orbit_covariance,
            dimension=dimension,
            eigenvalue_floor=active.covariance_eigenvalue_floor,
        )
        evaluations = evaluator.all_scores(point)
        archive.append(
            _make_mode(
                mode_id=f"noise_orbit_{mode_counter[0]:07d}",
                root_seed_id=str(seed.seed_id),
                parent_mode_id=None,
                source=str(seed.source),
                source_channel_index=None,
                optimized_channel_index=None,
                round_index=max(int(start_round) - 1, -1),
                orbit_values=point,
                covariance=covariance,
                evaluations=evaluations,
                optimizer_evaluations=0,
                optimizer_success=True,
            )
        )
        mode_counter[0] += 1
    if not archive:
        raise ValueError("Noise-annealed orbit transport requires at least one seed mode.")

    channel_modes = tuple(
        select_noise_annealed_orbit_modes(
            archive,
            channel_index=index,
            maximum_modes=active.modes_per_channel,
            lower=lower,
            upper=upper,
            minimum_separation=active.minimum_mode_separation,
        )
        for index in range(len(ladder))
    )
    history: list[NoiseAnnealedChannelSummary] = []

    for round_index in range(
        max(int(start_round), 0),
        max(int(start_round), 0) + max(int(active.round_count), 0),
    ):
        additions: list[list[NoiseAnnealedOrbitMode]] = [
            [] for _ in ladder
        ]
        for source_index, source_modes in enumerate(channel_modes):
            destinations = []
            if source_index > 0:
                destinations.append(source_index - 1)
            if source_index + 1 < len(ladder):
                destinations.append(source_index + 1)
            for destination_index in destinations:
                for mode in source_modes:
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
                            mode_counter=mode_counter,
                        )
                    )

        next_channels = []
        round_summaries = []
        for channel_index, (current, candidates) in enumerate(
            zip(channel_modes, additions, strict=True)
        ):
            pool = tuple(current) + tuple(candidates)
            selected = select_noise_annealed_orbit_modes(
                pool,
                channel_index=channel_index,
                maximum_modes=active.modes_per_channel,
                lower=lower,
                upper=upper,
                minimum_separation=active.minimum_mode_separation,
            )
            if not selected:
                raise RuntimeError(
                    f"Phase-noise channel {channel_index} lost every orbit mode."
                )
            next_channels.append(selected)
            archive.extend(candidates)
            summary = NoiseAnnealedChannelSummary(
                round_index=round_index,
                channel_index=channel_index,
                phase_noise_sigma=ladder[channel_index],
                candidate_count=len(pool),
                retained_count=len(selected),
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
                best_exact_integer_log_likelihood=max(
                    mode.exact_integer_log_likelihood for mode in selected
                ),
            )
            history.append(summary)
            round_summaries.append(summary)
        channel_modes = tuple(next_channels)
        if checkpoint_callback is not None:
            checkpoint_callback(round_index, channel_modes, tuple(round_summaries))

    portfolio = _portfolio_union(
        channel_modes,
        lower=lower,
        upper=upper,
        maximum_modes=max(int(active.modes_per_channel), 1) * len(ladder),
        minimum_separation=active.minimum_mode_separation,
    )
    return NoiseAnnealedOrbitResult(
        phase_noise_ladder=ladder,
        channel_modes=channel_modes,
        portfolio_modes=portfolio,
        mode_archive=tuple(archive),
        round_history=tuple(history),
        completed_rounds=max(int(active.round_count), 0),
        likelihood_call_count=evaluator.call_count,
    )


__all__ = [
    "NoiseAnnealedChannelSummary",
    "NoiseAnnealedOrbitConfig",
    "NoiseAnnealedOrbitMode",
    "NoiseAnnealedOrbitResult",
    "NoiseAnnealedOrbitSeed",
    "run_noise_annealed_orbit_transport",
    "select_noise_annealed_orbit_modes",
]
