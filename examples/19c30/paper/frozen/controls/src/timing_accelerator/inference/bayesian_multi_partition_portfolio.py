from __future__ import annotations

"""Mode-preserving global inference over a portfolio of phase partitions."""

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np

from timing_accelerator.inference.bayesian_circular_island_closure import (
    CircularIslandClosureConfig,
    CircularIslandClosureMode,
    _acquisition_residual,
    _exact_path_refinement,
    run_bayesian_circular_island_closure,
)
from timing_accelerator.inference.bayesian_collapsed_partition_kernel import (
    CollapsedPartitionKernelConfig,
    run_bayesian_collapsed_partition_kernel,
)
from timing_accelerator.inference.bayesian_global_predictive_lattice import (
    GlobalPredictiveLatticeConfig,
    run_bayesian_global_predictive_lattice,
)
from timing_accelerator.inference.bayesian_phase_partition_transport import (
    PhasePartitionMode,
    _map_partition_integer_vector,
    _normalized_bounds,
    _physical_values,
    _stable_covariance,
    _whitened_partition_phase_residual,
    adjacent_phase_partition_merges,
)


Partition = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MultiPartitionPortfolioConfig:
    maximum_partitions: int = 12
    maximum_original_partitions: int = 6
    parents_per_partition: int = 1
    expand_adjacent_merges: bool = True
    maximum_survivors: int = 32
    protected_per_partition: int = 1
    parameter_mode_separation: float = 2.0
    transport_rounds: int = 1
    transport_seed_count: int = 6
    transitions_per_seed: int = 2
    maximum_lattice_nodes: int = 2_000_000
    random_seed: int = 20260724


@dataclass(frozen=True)
class PartitionHypothesis:
    hypothesis_id: str
    partition: Partition
    source_parent_id: str
    source_level: int
    original_partition: bool
    block_count: int
    acquisition_chi2: float
    full_shape_chi2: float
    full_phase_chi2: float
    full_objective_chi2: float
    parent: PhasePartitionMode
    trusted_integer_offsets: tuple[int, ...] | None = None


@dataclass(frozen=True)
class MultiPartitionCandidate:
    candidate_id: str
    round_index: int
    hypothesis_id: str
    partition: Partition
    source_parent_id: str
    proposal_source: str
    mode: CircularIslandClosureMode
    log_mode_mass: float
    posterior_mass: float


@dataclass(frozen=True)
class MultiPartitionRoundSummary:
    round_index: int
    kernel_count: int
    proposed_mode_count: int
    retained_mode_count: int
    represented_partition_count: int
    unique_path_count: int
    best_phase_chi2: float
    best_objective_chi2: float


@dataclass(frozen=True)
class MultiPartitionPortfolioResult:
    hypotheses: tuple[PartitionHypothesis, ...]
    candidates: tuple[MultiPartitionCandidate, ...]
    round_history: tuple[MultiPartitionRoundSummary, ...]
    total_kernel_evaluations: int
    total_kernel_attempts: int


def _partition_key(partition: Sequence[Sequence[int]]) -> Partition:
    return tuple(tuple(int(session) for session in block) for block in partition)


def _reproject_covariance(
    covariance: np.ndarray,
    *,
    coordinates: Any,
    source_search_scales: Sequence[float] | None = None,
) -> np.ndarray:
    """Express a normalized covariance in the active coordinate chart."""

    result = np.asarray(covariance, dtype=float)
    if source_search_scales is None:
        return _stable_covariance(result)
    source = np.asarray(source_search_scales, dtype=float)
    target = np.asarray(coordinates.search_scales, dtype=float)
    if source.shape != target.shape or result.shape != (target.size, target.size):
        raise ValueError("Source coordinate scales must align with the mode covariance.")
    if not np.all(np.isfinite(source) & (source > 0.0)):
        raise ValueError("Source coordinate scales must be finite and positive.")
    ratio = source / target
    return _stable_covariance(ratio[:, None] * result * ratio[None, :])


def reproject_phase_partition_mode(
    mode: PhasePartitionMode,
    *,
    coordinates: Any,
    source_search_scales: Sequence[float] | None = None,
) -> PhasePartitionMode:
    """Restore a partition mode from its portable physical parameter values."""

    normalized = coordinates.normalize(coordinates.to_search(mode.parameter_values))
    return replace(
        mode,
        normalized_values=np.asarray(normalized, dtype=float),
        covariance=_reproject_covariance(
            mode.covariance,
            coordinates=coordinates,
            source_search_scales=source_search_scales,
        ),
    )


def reproject_multi_partition_candidate(
    candidate: MultiPartitionCandidate,
    *,
    coordinates: Any,
    source_search_scales: Sequence[float] | None = None,
) -> MultiPartitionCandidate:
    """Restore a complete-path candidate in the active coordinate chart."""

    mode = candidate.mode
    normalized = coordinates.normalize(coordinates.to_search(mode.parameter_values))
    return replace(
        candidate,
        mode=replace(
            mode,
            normalized_values=np.asarray(normalized, dtype=float),
            covariance=_reproject_covariance(
                mode.covariance,
                coordinates=coordinates,
                source_search_scales=source_search_scales,
            ),
        ),
    )


def _synthetic_parent(
    source: PhasePartitionMode,
    *,
    partition: Partition,
    mode_id: str,
    normalized_values: np.ndarray | None = None,
    parameter_values: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
) -> PhasePartitionMode:
    return PhasePartitionMode(
        mode_id=mode_id,
        parent_mode_id=source.mode_id,
        proposal_source="multi-partition-transport",
        level_index=source.level_index,
        partition=partition,
        parameter_values=np.asarray(
            source.parameter_values if parameter_values is None else parameter_values,
            dtype=float,
        ),
        normalized_values=np.asarray(
            source.normalized_values if normalized_values is None else normalized_values,
            dtype=float,
        ),
        covariance=_stable_covariance(
            np.asarray(source.covariance if covariance is None else covariance, dtype=float)
        ),
        shape_chi2=float(source.shape_chi2),
        collapsed_phase_deviance=float(source.collapsed_phase_deviance),
        prior_chi2=float(source.prior_chi2),
        objective_chi2=float(source.objective_chi2),
        log_mass=float(source.log_mass),
        posterior_mass=float(source.posterior_mass),
        nfev=0,
        phase_beta=1.0,
        success=True,
    )


def _full_snapshot_score(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    maximum_lattice_nodes: int,
) -> tuple[float, float, np.ndarray]:
    profile = evaluator.evaluate_values(
        _physical_values(coordinates, normalized_values)
    )
    full_partition = (tuple(range(int(profile.session_count))),)
    path = _map_partition_integer_vector(
        profile,
        full_partition,
        maximum_lattice_nodes=maximum_lattice_nodes,
    )
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    phase = _whitened_partition_phase_residual(profile, full_partition, path)
    return float(shape @ shape), float(phase @ phase), path


def _partition_acquisition_chi2(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_values: Sequence[float],
    partition: Partition,
    prior_precision: float,
    maximum_lattice_nodes: int,
) -> float:
    profile = evaluator.evaluate_values(
        _physical_values(coordinates, normalized_values)
    )
    internal = _map_partition_integer_vector(
        profile,
        partition,
        maximum_lattice_nodes=maximum_lattice_nodes,
    )
    residual = _acquisition_residual(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=normalized_values,
        partition=partition,
        internal_integers=internal,
        prior_precision=prior_precision,
    )
    return float(residual @ residual) / max(int(residual.size), 1)


def build_partition_hypotheses(
    *,
    evaluator: Any,
    coordinates: Any,
    parent_modes: Sequence[PhasePartitionMode],
    circular_config: CircularIslandClosureConfig,
    config: MultiPartitionPortfolioConfig,
) -> tuple[PartitionHypothesis, ...]:
    """Build a bounded bank of original and previously discarded merges."""

    if not parent_modes:
        raise ValueError("At least one completed partition mode is required.")
    records: list[PartitionHypothesis] = []
    seen_parent_partition: set[tuple[str, Partition]] = set()
    score_cache: dict[tuple[float, ...], tuple[float, float]] = {}
    for parent in parent_modes:
        if abs(float(parent.phase_beta) - 1.0) > 1.0e-12:
            continue
        original = _partition_key(parent.partition)
        partitions = [original]
        if config.expand_adjacent_merges:
            partitions.extend(adjacent_phase_partition_merges(original))
        state_key = tuple(np.asarray(parent.normalized_values, dtype=float).tolist())
        if state_key not in score_cache:
            shape_chi2, phase_chi2, _ = _full_snapshot_score(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=parent.normalized_values,
                maximum_lattice_nodes=config.maximum_lattice_nodes,
            )
            score_cache[state_key] = shape_chi2, phase_chi2
        shape_chi2, phase_chi2 = score_cache[state_key]
        for partition_index, partition in enumerate(partitions):
            key = (parent.mode_id, partition)
            if key in seen_parent_partition or len(partition) <= 1:
                continue
            seen_parent_partition.add(key)
            acquisition = _partition_acquisition_chi2(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=parent.normalized_values,
                partition=partition,
                prior_precision=circular_config.prior_precision,
                maximum_lattice_nodes=config.maximum_lattice_nodes,
            )
            records.append(
                PartitionHypothesis(
                    hypothesis_id=(
                        f"partition_l{int(parent.level_index):02d}_"
                        f"{len(partition):02d}_{partition_index:02d}_{parent.mode_id}"
                    ),
                    partition=partition,
                    source_parent_id=parent.mode_id,
                    source_level=int(parent.level_index),
                    original_partition=partition == original,
                    block_count=len(partition),
                    acquisition_chi2=acquisition,
                    full_shape_chi2=shape_chi2,
                    full_phase_chi2=phase_chi2,
                    full_objective_chi2=shape_chi2 + phase_chi2,
                    parent=_synthetic_parent(
                        parent,
                        partition=partition,
                        mode_id=f"hypothesis:{parent.mode_id}:{partition_index}",
                    ),
                )
            )
    if not records:
        raise ValueError("No completed multi-island partition hypotheses were built.")
    return select_partition_hypotheses(records, config=config)


def select_partition_hypotheses(
    hypotheses: Sequence[PartitionHypothesis],
    *,
    config: MultiPartitionPortfolioConfig,
) -> tuple[PartitionHypothesis, ...]:
    """Retain evidence-ranked partitions with resolution and merge diversity."""

    grouped: dict[Partition, list[PartitionHypothesis]] = {}
    for hypothesis in hypotheses:
        grouped.setdefault(hypothesis.partition, []).append(hypothesis)
    ordered_groups: dict[Partition, tuple[PartitionHypothesis, ...]] = {}
    for records in grouped.values():
        ordered = sorted(
            records,
            key=lambda item: (
                item.full_objective_chi2,
                item.acquisition_chi2,
                -float(item.parent.posterior_mass),
            ),
        )
        ordered_groups[ordered[0].partition] = tuple(
            ordered[: max(int(config.parents_per_partition), 1)]
        )

    representatives = tuple(records[0] for records in ordered_groups.values())
    limit = max(int(config.maximum_partitions), 1)
    selected: list[PartitionHypothesis] = []
    represented_partitions: set[Partition] = set()
    represented_block_counts: set[int] = set()

    originals = sorted(
        (item for item in representatives if item.original_partition),
        key=lambda item: (item.full_objective_chi2, item.acquisition_chi2),
    )
    original_limit = min(max(int(config.maximum_original_partitions), 0), limit)
    for item in originals:
        if item.partition in represented_partitions:
            continue
        selected.append(item)
        represented_partitions.add(item.partition)
        represented_block_counts.add(item.block_count)
        if len(selected) >= original_limit:
            break

    alternatives = sorted(
        (
            item
            for item in representatives
            if item.partition not in represented_partitions
        ),
        key=lambda item: (
            item.block_count in represented_block_counts,
            item.full_objective_chi2,
            item.acquisition_chi2,
        ),
    )
    for item in alternatives:
        if item.partition in represented_partitions:
            continue
        selected.append(item)
        represented_partitions.add(item.partition)
        represented_block_counts.add(item.block_count)
        if len(selected) >= limit:
            break
    return tuple(
        record
        for representative in selected
        for record in ordered_groups[representative.partition]
    )


def _mode_log_mass(mode: CircularIslandClosureMode) -> float:
    covariance = _stable_covariance(np.asarray(mode.covariance, dtype=float))
    sign, logdet = np.linalg.slogdet(covariance)
    volume = float(logdet) if sign > 0 and np.isfinite(logdet) else -math.inf
    return -0.5 * float(mode.objective_chi2) + 0.5 * volume


def _with_normalized_masses(
    candidates: Sequence[MultiPartitionCandidate],
) -> tuple[MultiPartitionCandidate, ...]:
    if not candidates:
        return ()
    values = np.asarray([candidate.log_mode_mass for candidate in candidates])
    finite = np.isfinite(values)
    if not np.any(finite):
        masses = np.full(values.size, 1.0 / values.size)
    else:
        shifted = np.where(finite, values - np.max(values[finite]), -math.inf)
        weights = np.exp(shifted)
        masses = weights / np.sum(weights)
    return tuple(
        replace(candidate, posterior_mass=float(mass))
        for candidate, mass in zip(candidates, masses, strict=True)
    )


def _parameter_distance(
    left: CircularIslandClosureMode,
    right: CircularIslandClosureMode,
) -> float:
    delta = np.asarray(left.normalized_values) - np.asarray(right.normalized_values)
    covariance = _stable_covariance(
        0.5 * (np.asarray(left.covariance) + np.asarray(right.covariance)),
        floor=1.0e-8,
    )
    try:
        return float(np.sqrt(max(delta @ np.linalg.solve(covariance, delta), 0.0)))
    except np.linalg.LinAlgError:
        return math.inf


def _covariance_log_volume_gain(base: np.ndarray, proposal: np.ndarray) -> float:
    base_covariance = _stable_covariance(np.asarray(base, dtype=float), floor=1.0e-8)
    combined = _stable_covariance(
        base_covariance + np.asarray(proposal, dtype=float), floor=1.0e-8
    )
    base_sign, base_logdet = np.linalg.slogdet(base_covariance)
    combined_sign, combined_logdet = np.linalg.slogdet(combined)
    if base_sign <= 0 or combined_sign <= 0:
        return -math.inf
    return float(combined_logdet - base_logdet)


def select_global_candidates(
    candidates: Sequence[MultiPartitionCandidate],
    *,
    config: MultiPartitionPortfolioConfig,
) -> tuple[MultiPartitionCandidate, ...]:
    """Preserve global fit, partition support, and distinct integer paths."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.log_mode_mass,
            item.mode.objective_chi2,
            item.mode.phase_chi2,
        ),
    )
    if not ordered:
        return ()
    limit = max(int(config.maximum_survivors), 1)
    selected: list[MultiPartitionCandidate] = []

    def add(candidate: MultiPartitionCandidate) -> bool:
        for previous in selected:
            same_path = candidate.mode.integer_offsets == previous.mode.integer_offsets
            if same_path and _parameter_distance(candidate.mode, previous.mode) < max(
                float(config.parameter_mode_separation), 0.0
            ):
                return False
        selected.append(candidate)
        return True

    best_objective = min(ordered, key=lambda item: item.mode.objective_chi2)
    add(best_objective)
    protected = max(int(config.protected_per_partition), 0)
    if protected:
        counts: dict[Partition, int] = {}
        for candidate in ordered:
            if counts.get(candidate.partition, 0) >= protected:
                continue
            if add(candidate):
                counts[candidate.partition] = counts.get(candidate.partition, 0) + 1
            if len(selected) >= limit:
                return _with_normalized_masses(selected)
    for candidate in ordered:
        add(candidate)
        if len(selected) >= limit:
            break
    return _with_normalized_masses(selected)


def _baseline_candidate(
    *,
    evaluator: Any,
    coordinates: Any,
    hypothesis: PartitionHypothesis,
    circular_config: CircularIslandClosureConfig,
    candidate_index: int,
) -> CircularIslandClosureMode | None:
    if hypothesis.trusted_integer_offsets is None:
        _, _, path = _full_snapshot_score(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_values=hypothesis.parent.normalized_values,
            maximum_lattice_nodes=circular_config.maximum_lattice_nodes,
        )
    else:
        path = np.asarray(hypothesis.trusted_integer_offsets, dtype=float)
    return _exact_path_refinement(
        evaluator=evaluator,
        coordinates=coordinates,
        start=np.asarray(hypothesis.parent.normalized_values, dtype=float),
        integer_offsets=path,
        acquisition_chi2=hypothesis.acquisition_chi2,
        internal_phase_chi2=math.nan,
        source=f"baseline:{hypothesis.hypothesis_id}",
        mode_index=candidate_index,
        config=circular_config,
    )


def _candidate(
    *,
    mode: CircularIslandClosureMode,
    hypothesis: PartitionHypothesis,
    round_index: int,
    candidate_index: int,
) -> MultiPartitionCandidate:
    return MultiPartitionCandidate(
        candidate_id=f"portfolio_r{round_index:02d}_{candidate_index:05d}",
        round_index=round_index,
        hypothesis_id=hypothesis.hypothesis_id,
        partition=hypothesis.partition,
        source_parent_id=hypothesis.source_parent_id,
        proposal_source=mode.source,
        mode=mode,
        log_mode_mass=_mode_log_mass(mode),
        posterior_mass=0.0,
    )


def _round_summary(
    *,
    round_index: int,
    kernel_count: int,
    proposed_count: int,
    retained: Sequence[MultiPartitionCandidate],
) -> MultiPartitionRoundSummary:
    return MultiPartitionRoundSummary(
        round_index=round_index,
        kernel_count=kernel_count,
        proposed_mode_count=proposed_count,
        retained_mode_count=len(retained),
        represented_partition_count=len({item.partition for item in retained}),
        unique_path_count=len({item.mode.integer_offsets for item in retained}),
        best_phase_chi2=min(
            (item.mode.phase_chi2 for item in retained), default=math.inf
        ),
        best_objective_chi2=min(
            (item.mode.objective_chi2 for item in retained), default=math.inf
        ),
    )


def run_bayesian_multi_partition_portfolio(
    *,
    evaluator: Any,
    coordinates: Any,
    parent_modes: Sequence[PhasePartitionMode],
    circular_config: CircularIslandClosureConfig | None = None,
    transport_circular_config: CircularIslandClosureConfig | None = None,
    collapsed_config: CollapsedPartitionKernelConfig | None = None,
    transport_collapsed_config: CollapsedPartitionKernelConfig | None = None,
    global_predictive_config: GlobalPredictiveLatticeConfig | None = None,
    config: MultiPartitionPortfolioConfig | None = None,
    resume_candidates: Sequence[MultiPartitionCandidate] = (),
    external_proposal_states: Sequence[Sequence[float]] = (),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    checkpoint_callback: Callable[
        [tuple[MultiPartitionCandidate, ...], tuple[MultiPartitionRoundSummary, ...]],
        None,
    ]
    | None = None,
) -> MultiPartitionPortfolioResult:
    """Infer complete paths without committing to one partition or merge order."""

    active = config or MultiPartitionPortfolioConfig()
    kernel = circular_config or CircularIslandClosureConfig()
    transport_kernel = transport_circular_config or kernel
    collapsed_kernel = collapsed_config
    transport_collapsed_kernel = transport_collapsed_config or collapsed_kernel
    lower, upper = _normalized_bounds(coordinates)
    parent_modes = tuple(
        reproject_phase_partition_mode(mode, coordinates=coordinates)
        for mode in parent_modes
    )
    resume_candidates = tuple(
        reproject_multi_partition_candidate(candidate, coordinates=coordinates)
        for candidate in resume_candidates
    )
    hypotheses = build_partition_hypotheses(
        evaluator=evaluator,
        coordinates=coordinates,
        parent_modes=parent_modes,
        circular_config=kernel,
        config=active,
    )
    all_candidates: list[MultiPartitionCandidate] = list(resume_candidates)
    history: list[MultiPartitionRoundSummary] = []
    evaluations = 0
    attempts = 0
    candidate_serial = len(all_candidates)

    def next_candidate_index() -> int:
        nonlocal candidate_serial
        value = candidate_serial
        candidate_serial += 1
        return value

    def run_kernel(
        hypothesis: PartitionHypothesis,
        *,
        round_index: int,
        active_kernel: CircularIslandClosureConfig,
        active_collapsed_kernel: CollapsedPartitionKernelConfig | None,
        proposal_states: Sequence[Sequence[float]] = (),
    ) -> list[MultiPartitionCandidate]:
        nonlocal evaluations, attempts
        proposed: list[MultiPartitionCandidate] = []
        baseline = _baseline_candidate(
            evaluator=evaluator,
            coordinates=coordinates,
            hypothesis=hypothesis,
            circular_config=active_kernel,
            candidate_index=candidate_serial,
        )
        if baseline is not None:
            candidate_index = next_candidate_index()
            proposed.append(
                _candidate(
                    mode=baseline,
                    hypothesis=hypothesis,
                    round_index=round_index,
                    candidate_index=candidate_index,
                )
            )

        def kernel_progress(record: dict[str, Any]) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        **record,
                        "stage": f"kernel-{record.get('stage', 'unknown')}",
                        "round": round_index,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "block_count": hypothesis.block_count,
                    }
                )

        if active_collapsed_kernel is None:
            result = run_bayesian_circular_island_closure(
                evaluator=evaluator,
                coordinates=coordinates,
                trusted_parent=hypothesis.parent,
                trusted_complete_integer_offsets=hypothesis.trusted_integer_offsets,
                config=active_kernel,
                progress_callback=kernel_progress,
            )
        else:
            result = run_bayesian_collapsed_partition_kernel(
                evaluator=evaluator,
                coordinates=coordinates,
                trusted_parent=hypothesis.parent,
                proposal_states=proposal_states,
                config=active_collapsed_kernel,
                progress_callback=kernel_progress,
            )
        evaluations += int(result.global_evaluations)
        attempts += int(result.local_attempts)
        for mode in result.modes:
            candidate_index = next_candidate_index()
            proposed.append(
                _candidate(
                    mode=mode,
                    hypothesis=hypothesis,
                    round_index=round_index,
                    candidate_index=candidate_index,
                )
            )
        return proposed

    initial_count = len(all_candidates)
    if not all_candidates:
        initial_proposal_states = tuple(
            np.asarray(hypothesis.parent.normalized_values, dtype=float)
            for hypothesis in hypotheses
        )
        for index, hypothesis in enumerate(hypotheses):
            seeded_kernel = replace(
                kernel,
                random_seed=int(active.random_seed) + 1009 * index,
            )
            seeded_collapsed_kernel = (
                replace(
                    collapsed_kernel,
                    random_seed=int(active.random_seed) + 1009 * index,
                )
                if collapsed_kernel is not None
                else None
            )
            proposed = run_kernel(
                hypothesis,
                round_index=0,
                active_kernel=seeded_kernel,
                active_collapsed_kernel=seeded_collapsed_kernel,
                proposal_states=initial_proposal_states,
            )
            all_candidates.extend(proposed)
            initial_count += len(proposed)
            if checkpoint_callback is not None:
                checkpoint_callback(tuple(all_candidates), tuple(history))
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "partition-complete",
                        "round": 0,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "proposed_modes": len(proposed),
                        "total_candidates": len(all_candidates),
                    }
                )
    elif progress_callback is not None:
        progress_callback(
            {
                "stage": "resume-candidates",
                "round": 0,
                "candidate_count": len(all_candidates),
            }
        )
    survivors = select_global_candidates(all_candidates, config=active)
    history.append(
        _round_summary(
            round_index=0,
            kernel_count=0 if resume_candidates else len(hypotheses),
            proposed_count=initial_count,
            retained=survivors,
        )
    )

    for round_index in range(1, max(int(active.transport_rounds), 0) + 1):
        transported: list[MultiPartitionCandidate] = []
        proposal_states = tuple(
            np.asarray(candidate.mode.normalized_values, dtype=float)
            for candidate in survivors
        ) + tuple(
            np.asarray(hypothesis.parent.normalized_values, dtype=float)
            for hypothesis in hypotheses
        ) + tuple(
            np.asarray(state, dtype=float) for state in external_proposal_states
        )
        kernels_run = 0
        if global_predictive_config is not None:
            def predictive_progress(record: dict[str, Any]) -> None:
                if progress_callback is not None:
                    progress_callback({**record, "portfolio_round": round_index})

            predictive = run_bayesian_global_predictive_lattice(
                evaluator=evaluator,
                coordinates=coordinates,
                seed_states=proposal_states,
                config=global_predictive_config,
                progress_callback=predictive_progress,
            )
            evaluations += int(predictive.total_profile_evaluations)
            attempts += int(predictive.total_refinements)
            target = hypotheses[0]
            for mode in predictive.modes:
                candidate_index = next_candidate_index()
                transported.append(
                    _candidate(
                        mode=mode,
                        hypothesis=target,
                        round_index=round_index,
                        candidate_index=candidate_index,
                    )
                )
            kernels_run += 1
        round_seed_pool = select_global_candidates(
            (*all_candidates, *transported),
            config=active,
        )
        seeds = tuple(
            round_seed_pool[: max(int(active.transport_seed_count), 0)]
        )
        for seed_index, seed in enumerate(seeds):
            ranked_targets: list[tuple[float, PartitionHypothesis]] = []
            for hypothesis in hypotheses:
                if hypothesis.partition == seed.partition:
                    continue
                if (
                    transport_collapsed_kernel is not None
                    and transport_collapsed_kernel.use_full_sequence_likelihood
                ):
                    score = -_covariance_log_volume_gain(
                        np.asarray(seed.mode.covariance, dtype=float),
                        np.asarray(hypothesis.parent.covariance, dtype=float),
                    )
                else:
                    score = _partition_acquisition_chi2(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        normalized_values=seed.mode.normalized_values,
                        partition=hypothesis.partition,
                        prior_precision=transport_kernel.prior_precision,
                        maximum_lattice_nodes=active.maximum_lattice_nodes,
                    )
                ranked_targets.append((score, hypothesis))
            ranked_targets.sort(
                key=lambda item: (
                    item[0],
                    abs(item[1].block_count - len(seed.partition)),
                )
            )
            for target_index, (_, target) in enumerate(
                ranked_targets[: max(int(active.transitions_per_seed), 0)]
            ):
                transported_parent = _synthetic_parent(
                    target.parent,
                    partition=target.partition,
                    mode_id=f"transport:{seed.candidate_id}:{target.hypothesis_id}",
                    normalized_values=np.clip(
                        np.asarray(seed.mode.normalized_values, dtype=float), lower, upper
                    ),
                    parameter_values=np.asarray(seed.mode.parameter_values, dtype=float),
                    covariance=_stable_covariance(
                        np.asarray(seed.mode.covariance, dtype=float)
                        + np.asarray(target.parent.covariance, dtype=float),
                        floor=1.0e-8,
                    ),
                )
                transported_hypothesis = replace(
                    target,
                    hypothesis_id=(
                        f"transport_r{round_index:02d}_{seed_index:03d}_"
                        f"{target_index:02d}_{target.hypothesis_id}"
                    ),
                    source_parent_id=seed.candidate_id,
                    parent=transported_parent,
                    trusted_integer_offsets=seed.mode.integer_offsets,
                )
                seeded_kernel = replace(
                    transport_kernel,
                    random_seed=(
                        int(active.random_seed)
                        + 100_003 * round_index
                        + 1009 * seed_index
                        + target_index
                    ),
                )
                seeded_collapsed_kernel = (
                    replace(
                        transport_collapsed_kernel,
                        random_seed=(
                            int(active.random_seed)
                            + 100_003 * round_index
                            + 1009 * seed_index
                            + target_index
                        ),
                    )
                    if transport_collapsed_kernel is not None
                    else None
                )
                proposed = run_kernel(
                    transported_hypothesis,
                    round_index=round_index,
                    active_kernel=seeded_kernel,
                    active_collapsed_kernel=seeded_collapsed_kernel,
                    proposal_states=proposal_states,
                )
                transported.extend(proposed)
                kernels_run += 1
        all_candidates.extend(transported)
        survivors = select_global_candidates(all_candidates, config=active)
        history.append(
            _round_summary(
                round_index=round_index,
                kernel_count=kernels_run,
                proposed_count=len(transported),
                retained=survivors,
            )
        )
        if checkpoint_callback is not None:
            checkpoint_callback(tuple(survivors), tuple(history))
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "transport-round-complete",
                    "round": round_index,
                    "kernel_count": kernels_run,
                    "proposed_modes": len(transported),
                    "best_phase_chi2": history[-1].best_phase_chi2,
                    "best_objective_chi2": history[-1].best_objective_chi2,
                }
            )

    return MultiPartitionPortfolioResult(
        hypotheses=hypotheses,
        candidates=survivors,
        round_history=tuple(history),
        total_kernel_evaluations=evaluations,
        total_kernel_attempts=attempts,
    )
