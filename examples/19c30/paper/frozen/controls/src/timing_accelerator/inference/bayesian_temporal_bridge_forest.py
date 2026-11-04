from __future__ import annotations

"""Mode-preserving Bayesian transport across nested temporal phase islands."""

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import logsumexp

from timing_accelerator.inference.bayesian_global_noise_smc import (
    GlobalNoiseSMCConfig,
    GlobalNoiseSMCInformationPath,
    GlobalNoiseSMCInformationTarget,
    GlobalNoiseSMCMode,
    GlobalNoiseSMCPosteriorPopulation,
    GlobalNoiseSMCResult,
    run_global_noise_smc,
    select_global_noise_smc_modes,
)
from timing_accelerator.inference.bayesian_noise_annealed_orbit_transport import (
    NoiseAnnealedOrbitSeed,
)
from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerBridgeMode,
    group_temporal_partition_bridge_modes,
    temporal_partition_bridge_matrix,
)
from timing_accelerator.inference.integer_least_squares import (
    solve_integer_least_squares,
)


Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TemporalBridgeForestConfig:
    """Controls for branch expansion and conditional family transport."""

    smc_config: GlobalNoiseSMCConfig = field(default_factory=GlobalNoiseSMCConfig)
    maximum_bridge_families: int = 16
    maximum_source_modes: int = 16
    modes_per_bridge_family: int = 2
    minimum_bridge_posterior_weight: float = 0.0
    predictive_lattice_candidates_per_mode: int = 32
    predictive_maximum_lattice_nodes: int = 2_000_000
    orbit_jacobian_sigma_fraction: float = 0.1
    bridge_inference: str = "marginal-portfolio"


@dataclass(frozen=True)
class TemporalBridgeFamilyDiscovery:
    bridge_integers: tuple[int, ...]
    discovery_log_score: float
    best_posterior_weight: float
    source_mode_id: str
    log_predictive_density: float
    proposal_orbit_normalized_values: np.ndarray
    proposal_orbit_covariance: np.ndarray


@dataclass(frozen=True)
class TemporalBridgeFamilyResult:
    family_id: str
    level_index: int
    source_partition: Partition
    target_partition: Partition
    discovery: TemporalBridgeFamilyDiscovery
    transport: GlobalNoiseSMCResult


@dataclass(frozen=True)
class TemporalBridgeMarginalResult:
    transport_id: str
    level_index: int
    source_partition: Partition
    target_partition: Partition
    transport: GlobalNoiseSMCResult


@dataclass(frozen=True)
class TemporalBridgeForestStage:
    level_index: int
    source_partition: Partition
    target_partition: Partition
    discovered_families: tuple[TemporalBridgeFamilyDiscovery, ...]
    family_results: tuple[TemporalBridgeFamilyResult, ...]
    retained_mode_count: int
    marginal_results: tuple[TemporalBridgeMarginalResult, ...] = ()


@dataclass(frozen=True)
class TemporalBridgeForestResult:
    initial_transport: GlobalNoiseSMCResult | None
    stage_history: tuple[TemporalBridgeForestStage, ...]
    final_modes: tuple[GlobalNoiseSMCMode, ...]
    likelihood_call_count: int
    reached_all_sessions: bool
    resumed_from_level_index: int | None = None


@dataclass(frozen=True)
class _BridgeChannelEvaluation:
    log_likelihood: float
    shape_log_likelihood: float
    curvature_log_likelihood: float
    slope_log_likelihood: float
    integer_log_likelihood: float


def _partition(value: Sequence[Sequence[int]]) -> Partition:
    return tuple(tuple(int(session) for session in block) for block in value)


def _bridge_component(
    evaluation: Any,
    *,
    session_count: int,
    source_partition: Partition,
    target_partition: Partition,
    bridge_integers: tuple[int, ...],
) -> SpinIntegerBridgeMode | None:
    for mode in group_temporal_partition_bridge_modes(
        evaluation,
        session_count=session_count,
        source_partitions=source_partition,
        target_partitions=target_partition,
    ):
        if mode.bridge_integers == bridge_integers:
            return mode
    return None


class _ConditionalTemporalBridgeLikelihood:
    """Expose one bridge component as a source-to-target SMC factor."""

    def __init__(
        self,
        likelihood: Any,
        *,
        session_count: int,
        source_partition: Partition,
        target_partition: Partition,
        bridge_integers: tuple[int, ...],
    ) -> None:
        self.likelihood = likelihood
        self.session_count = int(session_count)
        self.source_partition = source_partition
        self.target_partition = target_partition
        self.bridge_integers = tuple(int(value) for value in bridge_integers)
        self.orbit_lower = np.asarray(likelihood.orbit_lower, dtype=float)
        self.orbit_upper = np.asarray(likelihood.orbit_upper, dtype=float)

    def evaluate_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[_BridgeChannelEvaluation, ...]:
        family_evaluator = getattr(
            self.likelihood,
            "evaluate_temporal_bridge_family_noise_channels",
            None,
        )
        if family_evaluator is not None:
            source, families = family_evaluator(
                orbit_normalized_values,
                phase_noise_sigmas,
                source_partitions=self.source_partition,
                target_partitions=self.target_partition,
                bridge_integers=self.bridge_integers,
            )
            source_score = float(source.log_likelihood)
            return tuple(
                _BridgeChannelEvaluation(
                    log_likelihood=float(family.log_likelihood),
                    shape_log_likelihood=source_score,
                    curvature_log_likelihood=source_score,
                    slope_log_likelihood=0.0,
                    integer_log_likelihood=float(
                        family.log_likelihood - source_score
                    ),
                )
                for family in families
            )
        if hasattr(self.likelihood, "evaluate_temporal_bridge_noise_channels"):
            source, targets = (
                self.likelihood.evaluate_temporal_bridge_noise_channels(
                    orbit_normalized_values,
                    phase_noise_sigmas,
                    source_partitions=self.source_partition,
                    target_partitions=self.target_partition,
                )
            )
        else:
            source = self.likelihood.evaluate_partition_noise_channels(
                orbit_normalized_values,
                (0.0,),
                session_partitions=self.source_partition,
            )[0]
            targets = self.likelihood.evaluate_partition_noise_channels(
                orbit_normalized_values,
                phase_noise_sigmas,
                session_partitions=self.target_partition,
            )
        source_score = float(source.log_likelihood)
        channels = []
        for target in targets:
            component = _bridge_component(
                target,
                session_count=self.session_count,
                source_partition=self.source_partition,
                target_partition=self.target_partition,
                bridge_integers=self.bridge_integers,
            )
            target_score = (
                float(target.shape_log_likelihood + component.integer_log_likelihood)
                if component is not None
                else -math.inf
            )
            channels.append(
                _BridgeChannelEvaluation(
                    log_likelihood=target_score,
                    shape_log_likelihood=source_score,
                    curvature_log_likelihood=source_score,
                    slope_log_likelihood=0.0,
                    integer_log_likelihood=float(target_score - source_score),
                )
            )
        return tuple(channels)


class _MarginalTemporalBridgeLikelihood:
    """Bridge partitions while marginalizing all newly introduced integers."""

    def __init__(
        self,
        likelihood: Any,
        *,
        source_partition: Partition,
        target_partition: Partition,
    ) -> None:
        self.likelihood = likelihood
        self.source_partition = source_partition
        self.target_partition = target_partition
        self.orbit_lower = np.asarray(likelihood.orbit_lower, dtype=float)
        self.orbit_upper = np.asarray(likelihood.orbit_upper, dtype=float)

    def evaluate_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[_BridgeChannelEvaluation, ...]:
        if hasattr(self.likelihood, "evaluate_temporal_bridge_noise_channels"):
            source, targets = (
                self.likelihood.evaluate_temporal_bridge_noise_channels(
                    orbit_normalized_values,
                    phase_noise_sigmas,
                    source_partitions=self.source_partition,
                    target_partitions=self.target_partition,
                )
            )
        else:
            source = self.likelihood.evaluate_partition_noise_channels(
                orbit_normalized_values,
                (0.0,),
                session_partitions=self.source_partition,
            )[0]
            targets = self.likelihood.evaluate_partition_noise_channels(
                orbit_normalized_values,
                phase_noise_sigmas,
                session_partitions=self.target_partition,
            )
        source_score = float(source.log_likelihood)
        return tuple(
            _BridgeChannelEvaluation(
                log_likelihood=float(target.log_likelihood),
                shape_log_likelihood=source_score,
                curvature_log_likelihood=source_score,
                slope_log_likelihood=0.0,
                integer_log_likelihood=float(
                    target.log_likelihood - source_score
                ),
            )
            for target in targets
        )


def discover_temporal_bridge_families(
    *,
    likelihood: Any,
    source_modes: Sequence[GlobalNoiseSMCMode],
    source_partition: Sequence[Sequence[int]],
    target_partition: Sequence[Sequence[int]],
    maximum_families: int,
    minimum_posterior_weight: float = 0.0,
    predictive_candidate_count: int = 32,
    maximum_lattice_nodes: int = 2_000_000,
    orbit_jacobian_sigma_fraction: float = 0.1,
) -> tuple[TemporalBridgeFamilyDiscovery, ...]:
    """Enumerate bridge aliases under the source orbit posterior predictive."""

    source = _partition(source_partition)
    target = _partition(target_partition)
    session_count = sum(len(block) for block in source)
    bridge_matrix = temporal_partition_bridge_matrix(
        session_count=session_count,
        source_partitions=source,
        target_partitions=target,
    )
    if bridge_matrix.shape[0] == 0:
        raise ValueError("A temporal bridge transition must introduce an integer gauge.")
    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    width = upper - lower
    records: dict[tuple[int, ...], TemporalBridgeFamilyDiscovery] = {}

    def stable_covariance(values: np.ndarray, *, scale: np.ndarray) -> np.ndarray:
        covariance = np.asarray(values, dtype=float)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = np.maximum(np.square(np.asarray(scale, dtype=float)) * 1.0e-12, 1.0e-14)
        scalar_floor = float(np.min(floor))
        return (
            eigenvectors
            @ np.diag(np.maximum(eigenvalues, scalar_floor))
            @ eigenvectors.T
        )

    def retain(discovery: TemporalBridgeFamilyDiscovery) -> None:
        previous = records.get(discovery.bridge_integers)
        if previous is None or discovery.discovery_log_score > previous.discovery_log_score:
            records[discovery.bridge_integers] = discovery

    for source_mode in source_modes:
        orbit_center = np.asarray(
            source_mode.orbit_normalized_values,
            dtype=float,
        )
        orbit_covariance = stable_covariance(
            np.asarray(source_mode.orbit_covariance, dtype=float),
            scale=width,
        )
        bridge_evaluator = getattr(
            likelihood,
            "evaluate_temporal_bridge_noise_channels",
            None,
        )
        if bridge_evaluator is None:
            source_evaluation = likelihood.evaluate_partition_noise_channels(
                orbit_center,
                (0.0,),
                session_partitions=source,
            )[0]
            evaluation = likelihood.evaluate_partition_noise_channels(
                orbit_center,
                (0.0,),
                session_partitions=target,
            )[0]
        else:
            source_evaluation, target_evaluations = bridge_evaluator(
                orbit_center,
                (0.0,),
                source_partitions=source,
                target_partitions=target,
            )
            evaluation = target_evaluations[0]
        source_log_likelihood = float(source_evaluation.log_likelihood)
        for bridge_mode in group_temporal_partition_bridge_modes(
            evaluation,
            session_count=session_count,
            source_partitions=source,
            target_partitions=target,
        ):
            score = float(
                evaluation.shape_log_likelihood
                + bridge_mode.integer_log_likelihood
            )
            retain(
                TemporalBridgeFamilyDiscovery(
                    bridge_integers=bridge_mode.bridge_integers,
                    discovery_log_score=score,
                    best_posterior_weight=float(bridge_mode.posterior_weight),
                    source_mode_id=source_mode.mode_id,
                    log_predictive_density=float(
                        bridge_mode.log_posterior_weight
                    ),
                    proposal_orbit_normalized_values=orbit_center.copy(),
                    proposal_orbit_covariance=orbit_covariance.copy(),
                )
            )

        bridge_center = bridge_matrix @ np.asarray(
            evaluation.integer_center,
            dtype=float,
        )
        fixed_bridge_covariance = stable_covariance(
            bridge_matrix
            @ np.asarray(evaluation.integer_covariance, dtype=float)
            @ bridge_matrix.T,
            scale=np.ones(bridge_matrix.shape[0], dtype=float),
        )
        sigma_fraction = max(float(orbit_jacobian_sigma_fraction), 1.0e-4)
        finite_steps = np.maximum(
            sigma_fraction
            * np.sqrt(np.maximum(np.diag(orbit_covariance), 0.0)),
            width * 1.0e-6,
        )
        bridge_jacobian = np.zeros(
            (bridge_matrix.shape[0], orbit_center.size),
            dtype=float,
        )
        for coordinate, step in enumerate(finite_steps):
            plus = orbit_center.copy()
            minus = orbit_center.copy()
            plus[coordinate] = min(plus[coordinate] + step, upper[coordinate])
            minus[coordinate] = max(minus[coordinate] - step, lower[coordinate])
            denominator = float(plus[coordinate] - minus[coordinate])
            if denominator <= 0.0:
                continue
            plus_evaluation = likelihood.evaluate_partition_noise_channels(
                plus,
                (0.0,),
                session_partitions=target,
            )[0]
            minus_evaluation = likelihood.evaluate_partition_noise_channels(
                minus,
                (0.0,),
                session_partitions=target,
            )[0]
            bridge_jacobian[:, coordinate] = (
                bridge_matrix
                @ (
                    np.asarray(plus_evaluation.integer_center, dtype=float)
                    - np.asarray(minus_evaluation.integer_center, dtype=float)
                )
                / denominator
            )
        predictive_covariance = stable_covariance(
            fixed_bridge_covariance
            + bridge_jacobian @ orbit_covariance @ bridge_jacobian.T,
            scale=np.ones(bridge_matrix.shape[0], dtype=float),
        )
        try:
            lattice = solve_integer_least_squares(
                bridge_center,
                predictive_covariance,
                candidate_count=max(int(predictive_candidate_count), 1),
                max_visited_nodes=max(int(maximum_lattice_nodes), 1),
            )
        except RuntimeError:
            continue
        raw_log_weights = -0.5 * np.asarray(
            [candidate.squared_mahalanobis for candidate in lattice.candidates],
            dtype=float,
        )
        normalized_log_weights = raw_log_weights - float(logsumexp(raw_log_weights))
        sign, logdet = np.linalg.slogdet(predictive_covariance)
        predictive_log_normalizer = -0.5 * logdet if sign > 0.0 else -math.inf
        gain = (
            orbit_covariance
            @ bridge_jacobian.T
            @ np.linalg.inv(predictive_covariance)
        )
        conditional_covariance = stable_covariance(
            orbit_covariance
            - gain @ bridge_jacobian @ orbit_covariance,
            scale=width,
        )
        for candidate, log_weight in zip(
            lattice.candidates,
            normalized_log_weights,
            strict=True,
        ):
            signature = tuple(int(value) for value in candidate.integer_vector)
            displacement = np.asarray(signature, dtype=float) - bridge_center
            proposal_center = np.clip(
                orbit_center + gain @ displacement,
                lower,
                upper,
            )
            predictive_log_density = float(
                predictive_log_normalizer
                - 0.5 * float(candidate.squared_mahalanobis)
            )
            retain(
                TemporalBridgeFamilyDiscovery(
                    bridge_integers=signature,
                    discovery_log_score=float(
                        source_log_likelihood + log_weight
                    ),
                    best_posterior_weight=float(math.exp(float(log_weight))),
                    source_mode_id=source_mode.mode_id,
                    log_predictive_density=predictive_log_density,
                    proposal_orbit_normalized_values=proposal_center,
                    proposal_orbit_covariance=conditional_covariance.copy(),
                )
            )

    ordered = sorted(
        records.values(),
        key=lambda item: (
            -item.discovery_log_score,
            -item.best_posterior_weight,
            item.bridge_integers,
        ),
    )
    floor = min(max(float(minimum_posterior_weight), 0.0), 1.0)
    best_score = ordered[0].discovery_log_score if ordered else -math.inf
    log_floor = math.log(floor) if floor > 0.0 else -math.inf
    retained = tuple(
        item
        for index, item in enumerate(ordered)
        if index == 0
        or item.discovery_log_score - best_score >= log_floor
    )
    return retained[: max(int(maximum_families), 1)]


def _mode_seeds(
    modes: Sequence[GlobalNoiseSMCMode],
    *,
    maximum_modes: int,
    source: str,
) -> tuple[NoiseAnnealedOrbitSeed, ...]:
    ranked = tuple(modes)[: max(int(maximum_modes), 1)]
    return tuple(
        NoiseAnnealedOrbitSeed(
            seed_id=f"{source}-{index:04d}-{mode.mode_id}",
            orbit_normalized_values=np.asarray(
                mode.orbit_normalized_values,
                dtype=float,
            ).copy(),
            orbit_covariance=np.asarray(mode.orbit_covariance, dtype=float).copy(),
            source=source,
        )
        for index, mode in enumerate(ranked)
    )


def _retain_endpoint_transport_modes(
    result: GlobalNoiseSMCResult,
    *,
    final_factor_index: int,
    lower: Sequence[float],
    upper: Sequence[float],
    config: GlobalNoiseSMCConfig,
) -> GlobalNoiseSMCResult:
    """Exclude annealing intermediates from a completed posterior portfolio."""

    if not result.reached_exact_endpoint:
        return result
    endpoint_archive = tuple(
        point
        for point in result.archive
        if point.factor_index == int(final_factor_index)
        and point.bridge_fraction >= 1.0 - 1.0e-12
    )
    endpoint_modes = select_global_noise_smc_modes(
        endpoint_archive,
        lower=lower,
        upper=upper,
        maximum_modes=config.portfolio_mode_count,
        minimum_separation=config.minimum_mode_separation,
        covariance_floor=config.covariance_eigenvalue_floor,
    )
    return replace(result, portfolio_modes=endpoint_modes)


def _partition_mode_log_likelihood(
    mode: GlobalNoiseSMCMode,
    *,
    partition_index: int,
) -> float:
    channels = np.asarray(
        mode.partition_integer_log_likelihoods,
        dtype=float,
    )
    if channels.ndim != 2 or channels.shape[0] == 0 or channels.shape[1] == 0:
        return -math.inf
    index = min(max(int(partition_index), 0), channels.shape[0] - 1)
    return float(mode.shape_log_likelihood + channels[index, -1])


def run_bayesian_temporal_bridge_forest(
    *,
    likelihood: Any,
    seeds: Sequence[NoiseAnnealedOrbitSeed] = (),
    config: TemporalBridgeForestConfig | None = None,
    progress_callback: Callable[[TemporalBridgeForestStage], None] | None = None,
    family_callback: Callable[[TemporalBridgeFamilyResult], None] | None = None,
    marginal_callback: Callable[[TemporalBridgeMarginalResult], None] | None = None,
    root_callback: Callable[[GlobalNoiseSMCResult], None] | None = None,
    smc_progress_callback: Callable[[int, int, Any], None] | None = None,
    resume_modes: Sequence[GlobalNoiseSMCMode] = (),
    resume_population: GlobalNoiseSMCPosteriorPopulation | None = None,
    resume_level_index: int | None = None,
) -> TemporalBridgeForestResult:
    """Expand bridge integers first, then transport every retained family alone."""

    active = config or TemporalBridgeForestConfig()
    bridge_inference = str(active.bridge_inference).strip().lower()
    if bridge_inference not in {
        "marginalize",
        "marginal-direct",
        "marginal-portfolio",
        "fixed-family",
    }:
        raise ValueError(
            "Temporal bridge inference must be 'marginalize', "
            "'marginal-direct', 'marginal-portfolio', or 'fixed-family'."
        )
    hierarchy = tuple(
        _partition(partition)
        for partition in getattr(likelihood, "session_partition_hierarchy", ())
    )
    if not hierarchy:
        raise ValueError("Temporal bridge forest requires a partition hierarchy.")
    session_count = sum(len(block) for block in hierarchy[0])
    if any(sum(len(block) for block in partition) != session_count for partition in hierarchy):
        raise ValueError("Every temporal partition must cover the same sessions.")

    smc = active.smc_config
    ladder = tuple(float(value) for value in smc.phase_noise_ladder)
    initial_targets = [
        GlobalNoiseSMCInformationTarget("curvature"),
        GlobalNoiseSMCInformationTarget("shape"),
    ]
    if ladder[0] > 0.0:
        initial_targets.append(
            GlobalNoiseSMCInformationTarget("partition-phase", ladder[0], 0)
        )
    initial_targets.append(
        GlobalNoiseSMCInformationTarget("partition-phase", 0.0, 0)
    )
    bridge_smc = replace(
        smc,
        phase_noise_ladder=(0.0,),
        initial_phase_noise=None,
        retain_seed_centers_for_mode_discovery=True,
    )
    bridge_path = GlobalNoiseSMCInformationPath(
        path_id="conditional-temporal-bridge",
        targets=(
            GlobalNoiseSMCInformationTarget("curvature"),
            GlobalNoiseSMCInformationTarget("global-phase", 0.0),
        ),
    )

    resumed = tuple(resume_modes)
    if resumed:
        if resume_level_index is None:
            raise ValueError("Forest resume modes require a resume level index.")
        checkpoint_level = int(resume_level_index)
        if checkpoint_level < 0 or checkpoint_level >= len(hierarchy):
            raise ValueError("The forest resume level lies outside the hierarchy.")
        orbit_dimension = np.asarray(likelihood.orbit_lower, dtype=float).size
        for mode in resumed:
            if float(mode.bridge_fraction) < 1.0 - 1.0e-12:
                raise ValueError(
                    "Forest resume accepts only completed endpoint posterior modes."
                )
            if np.asarray(mode.orbit_normalized_values).shape != (orbit_dimension,):
                raise ValueError("A resumed forest mode has the wrong orbit dimension.")
            if np.asarray(mode.orbit_covariance).shape != (
                orbit_dimension,
                orbit_dimension,
            ):
                raise ValueError("A resumed forest mode has the wrong covariance shape.")
        if resume_population is not None:
            population_particles = np.asarray(
                resume_population.particles,
                dtype=float,
            )
            population_weights = np.asarray(
                resume_population.log_weights,
                dtype=float,
            )
            if (
                population_particles.ndim != 2
                or population_particles.shape[1:] != (orbit_dimension,)
                or population_weights.shape != (population_particles.shape[0],)
            ):
                raise ValueError("A resumed forest population has incompatible shapes.")
        initial = None
        current_modes = resumed
        current_population = resume_population
        start_level_index = checkpoint_level + 1
        total_calls = 0
    else:
        if resume_level_index is not None:
            raise ValueError("A forest resume level requires endpoint modes.")
        initial = run_global_noise_smc(
            likelihood=likelihood,
            seeds=seeds,
            config=smc,
            information_path=GlobalNoiseSMCInformationPath(
                path_id="temporal-bridge-root",
                targets=tuple(initial_targets),
            ),
            progress_callback=(
                (lambda stage: smc_progress_callback(0, -1, stage))
                if smc_progress_callback is not None
                else None
            ),
        )
        initial = _retain_endpoint_transport_modes(
            initial,
            final_factor_index=len(initial_targets) - 1,
            lower=likelihood.orbit_lower,
            upper=likelihood.orbit_upper,
            config=smc,
        )
        if root_callback is not None:
            root_callback(initial)
        current_modes = tuple(initial.portfolio_modes)
        current_population = GlobalNoiseSMCPosteriorPopulation(
            particles=np.asarray(initial.particles, dtype=float).copy(),
            log_weights=np.asarray(initial.log_weights, dtype=float).copy(),
            source_target=GlobalNoiseSMCInformationTarget("curvature"),
        )
        start_level_index = 1
        total_calls = int(initial.likelihood_call_count)
    stages = []

    for level_index in range(start_level_index, len(hierarchy)):
        source_partition = hierarchy[level_index - 1]
        target_partition = hierarchy[level_index]
        source_modes = tuple(
            sorted(
                current_modes,
                key=lambda mode: (
                    -_partition_mode_log_likelihood(
                        mode,
                        partition_index=level_index - 1,
                    ),
                    -mode.shape_log_likelihood,
                ),
            )[: max(int(active.maximum_source_modes), 1)]
        )
        discoveries = discover_temporal_bridge_families(
            likelihood=likelihood,
            source_modes=source_modes,
            source_partition=source_partition,
            target_partition=target_partition,
            maximum_families=active.maximum_bridge_families,
            minimum_posterior_weight=active.minimum_bridge_posterior_weight,
            predictive_candidate_count=(
                active.predictive_lattice_candidates_per_mode
            ),
            maximum_lattice_nodes=active.predictive_maximum_lattice_nodes,
            orbit_jacobian_sigma_fraction=(
                active.orbit_jacobian_sigma_fraction
            ),
        )
        source_seeds = _mode_seeds(
            source_modes,
            maximum_modes=active.maximum_source_modes,
            source=f"temporal-level-{level_index - 1:02d}",
        )
        if bridge_inference in {
            "marginalize",
            "marginal-direct",
            "marginal-portfolio",
        }:
            marginal = _MarginalTemporalBridgeLikelihood(
                likelihood,
                source_partition=source_partition,
                target_partition=target_partition,
            )
            predictive_seeds = tuple(
                NoiseAnnealedOrbitSeed(
                    seed_id=(
                        f"temporal-marginal-predictive-l{level_index:02d}-"
                        f"f{family_index:04d}"
                    ),
                    orbit_normalized_values=np.asarray(
                        discovery.proposal_orbit_normalized_values,
                        dtype=float,
                    ).copy(),
                    orbit_covariance=np.asarray(
                        discovery.proposal_orbit_covariance,
                        dtype=float,
                    ).copy(),
                    source="temporal-bridge-posterior-predictive",
                )
                for family_index, discovery in enumerate(discoveries)
            )
            path_specs = []
            if bridge_inference != "marginal-direct":
                path_specs.append(("continuation", bridge_path, -1))
            if bridge_inference in {"marginal-direct", "marginal-portfolio"}:
                path_specs.append(
                    (
                        "direct",
                        GlobalNoiseSMCInformationPath(
                            path_id="direct-temporal-marginal",
                            targets=(
                                GlobalNoiseSMCInformationTarget(
                                    "global-phase", 0.0
                                ),
                            ),
                        ),
                        -2,
                    )
                )
            marginal_results = []
            endpoint_archive = []
            for path_index, (path_name, path_spec, progress_index) in enumerate(
                path_specs
            ):
                population_transport = bool(
                    path_name == "continuation" and current_population is not None
                )
                active_path_spec = (
                    GlobalNoiseSMCInformationPath(
                        path_id="posterior-population-temporal-bridge",
                        targets=(
                            GlobalNoiseSMCInformationTarget("global-phase", 0.0),
                        ),
                    )
                    if population_transport
                    else path_spec
                )
                path_seeds = (
                    (*predictive_seeds, *source_seeds, *seeds)
                    if path_name == "direct"
                    else (*predictive_seeds, *source_seeds)
                )
                marginal_config = replace(
                    bridge_smc,
                    retain_seed_centers_for_mode_discovery=(
                        not population_transport
                    ),
                    random_seed=(
                        int(bridge_smc.random_seed)
                        + 1009 * level_index
                        + 9176 * path_index
                    ),
                )
                transport = run_global_noise_smc(
                    likelihood=marginal,
                    seeds=path_seeds,
                    config=marginal_config,
                    information_path=active_path_spec,
                    initial_population=(
                        current_population if population_transport else None
                    ),
                    progress_callback=(
                        (
                            lambda stage, level=level_index, marker=progress_index: (
                                smc_progress_callback(level, marker, stage)
                            )
                        )
                        if smc_progress_callback is not None
                        else None
                    ),
                )
                final_factor_index = len(active_path_spec.targets) - 1
                transport = _retain_endpoint_transport_modes(
                    transport,
                    final_factor_index=final_factor_index,
                    lower=marginal.orbit_lower,
                    upper=marginal.orbit_upper,
                    config=marginal_config,
                )
                total_calls += int(transport.likelihood_call_count)
                marginal_result = TemporalBridgeMarginalResult(
                    transport_id=(
                        f"temporal-marginal-l{level_index:02d}-{path_name}"
                    ),
                    level_index=level_index,
                    source_partition=source_partition,
                    target_partition=target_partition,
                    transport=transport,
                )
                marginal_results.append(marginal_result)
                if marginal_callback is not None:
                    marginal_callback(marginal_result)
                if transport.reached_exact_endpoint:
                    endpoint_archive.extend(
                        point
                        for point in transport.archive
                        if point.factor_index == final_factor_index
                        and point.bridge_fraction >= 1.0 - 1.0e-12
                    )
            current_modes = select_global_noise_smc_modes(
                endpoint_archive,
                lower=marginal.orbit_lower,
                upper=marginal.orbit_upper,
                maximum_modes=max(int(active.maximum_source_modes), 1),
                minimum_separation=smc.minimum_mode_separation,
                covariance_floor=smc.covariance_eigenvalue_floor,
            )
            population_candidates = tuple(
                item.transport
                for item in marginal_results
                if item.transport.reached_exact_endpoint
            )
            continuation_population = next(
                (
                    item.transport
                    for item in marginal_results
                    if item.transport.reached_exact_endpoint
                    and item.transport_id.endswith("-continuation")
                ),
                None,
            )
            population_source = continuation_population or (
                population_candidates[0] if population_candidates else None
            )
            current_population = (
                GlobalNoiseSMCPosteriorPopulation(
                    particles=np.asarray(population_source.particles, dtype=float).copy(),
                    log_weights=np.asarray(
                        population_source.log_weights,
                        dtype=float,
                    ).copy(),
                    source_target=GlobalNoiseSMCInformationTarget("curvature"),
                )
                if population_source is not None
                else None
            )
            stage = TemporalBridgeForestStage(
                level_index=level_index,
                source_partition=source_partition,
                target_partition=target_partition,
                discovered_families=discoveries,
                family_results=(),
                retained_mode_count=len(current_modes),
                marginal_results=tuple(marginal_results),
            )
            stages.append(stage)
            if progress_callback is not None:
                progress_callback(stage)
            if not current_modes:
                break
            continue
        family_results = []
        retained_modes = []
        for family_index, discovery in enumerate(discoveries):
            conditional = _ConditionalTemporalBridgeLikelihood(
                likelihood,
                session_count=session_count,
                source_partition=source_partition,
                target_partition=target_partition,
                bridge_integers=discovery.bridge_integers,
            )
            family_config = replace(
                bridge_smc,
                random_seed=(
                    int(bridge_smc.random_seed)
                    + 1009 * level_index
                    + 9176 * family_index
                ),
            )
            family_seeds = (
                NoiseAnnealedOrbitSeed(
                    seed_id=(
                        f"temporal-predictive-l{level_index:02d}-"
                        f"f{family_index:04d}"
                    ),
                    orbit_normalized_values=np.asarray(
                        discovery.proposal_orbit_normalized_values,
                        dtype=float,
                    ).copy(),
                    orbit_covariance=np.asarray(
                        discovery.proposal_orbit_covariance,
                        dtype=float,
                    ).copy(),
                    source="temporal-bridge-posterior-predictive",
                ),
                *source_seeds,
            )
            transport = run_global_noise_smc(
                likelihood=conditional,
                seeds=family_seeds,
                config=family_config,
                information_path=bridge_path,
                progress_callback=(
                    (
                        lambda stage, level=level_index, family=family_index: (
                            smc_progress_callback(level, family, stage)
                        )
                    )
                    if smc_progress_callback is not None
                    else None
                ),
            )
            transport = _retain_endpoint_transport_modes(
                transport,
                final_factor_index=len(bridge_path.targets) - 1,
                lower=conditional.orbit_lower,
                upper=conditional.orbit_upper,
                config=family_config,
            )
            total_calls += int(transport.likelihood_call_count)
            family_id = (
                f"temporal-bridge-l{level_index:02d}-"
                f"f{family_index:04d}"
            )
            family_result = TemporalBridgeFamilyResult(
                family_id=family_id,
                level_index=level_index,
                source_partition=source_partition,
                target_partition=target_partition,
                discovery=discovery,
                transport=transport,
            )
            family_results.append(family_result)
            if family_callback is not None:
                family_callback(family_result)
            ranked = sorted(
                transport.portfolio_modes,
                key=lambda mode: (
                    -mode.exact_log_likelihood,
                    -mode.shape_log_likelihood,
                ),
            )
            if transport.reached_exact_endpoint:
                retained_modes.extend(
                    ranked[: max(int(active.modes_per_bridge_family), 1)]
                )
        current_modes = tuple(retained_modes)
        stage = TemporalBridgeForestStage(
            level_index=level_index,
            source_partition=source_partition,
            target_partition=target_partition,
            discovered_families=discoveries,
            family_results=tuple(family_results),
            retained_mode_count=len(current_modes),
        )
        stages.append(stage)
        if progress_callback is not None:
            progress_callback(stage)
        if not current_modes:
            break

    return TemporalBridgeForestResult(
        initial_transport=initial,
        stage_history=tuple(stages),
        final_modes=current_modes,
        likelihood_call_count=total_calls,
        reached_all_sessions=bool(
            current_modes
            and (
                (int(resume_level_index) if resumed else 0) + len(stages)
                == len(hierarchy) - 1
            )
            and len(hierarchy[-1]) == 1
        ),
        resumed_from_level_index=(
            int(resume_level_index) if resumed else None
        ),
    )


__all__ = [
    "TemporalBridgeFamilyDiscovery",
    "TemporalBridgeFamilyResult",
    "TemporalBridgeForestConfig",
    "TemporalBridgeForestResult",
    "TemporalBridgeForestStage",
    "TemporalBridgeMarginalResult",
    "discover_temporal_bridge_families",
    "run_bayesian_temporal_bridge_forest",
]
