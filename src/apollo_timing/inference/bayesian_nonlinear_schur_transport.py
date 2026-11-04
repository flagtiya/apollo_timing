from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Sequence

import numpy as np

from apollo_timing.inference.bayesian_conditioned_path_acquisition import (
    ConditionedPathAcquisitionResult,
    ConditionedPathDirectionConfig,
)
from apollo_timing.inference.bayesian_global_lattice_mode_tracker import (
    GlobalLatticeMode,
    GlobalLatticeTrackerConfig,
    LatticeProposal,
    refit_fixed_path_proposal,
)


@dataclass(frozen=True)
class NonlinearSchurTransportConfig:
    round_count: int = 1
    round_index_offset: int = 0
    parent_modes_per_round: int = 6
    exact_refits_per_round: int = 32
    retained_modes: int = 32
    maximum_modes_per_path: int = 2
    maximum_refits_per_path: int = 2
    maximum_modes_per_root: int = 8
    mode_separation: float = 1.0e-5
    minimum_productive_reduction: float = 1.0
    retain_only_productive_children: bool = False
    fixed_path_max_nfev: int = 640
    phase_tempering_ladder: tuple[float, ...] = (
        0.000001,
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.1,
        0.3,
        1.0,
    )
    acquisition_config: ConditionedPathDirectionConfig = field(
        default_factory=ConditionedPathDirectionConfig
    )


@dataclass(frozen=True)
class NonlinearSchurAcquisitionRecord:
    round_index: int
    parent_mode_id: str
    parent_root_mode_id: str
    result: ConditionedPathAcquisitionResult


@dataclass(frozen=True)
class NonlinearSchurRefitRecord:
    round_index: int
    proposal_index: int
    parent_mode_id: str
    root_mode_id: str
    integer_offsets: tuple[int, ...]
    proposal_class: str
    conditioning_channel: str
    predictive_closure_chi2: float
    parent_objective_chi2: float
    predicted_reduction: float
    success: bool
    objective_chi2: float
    actual_reduction: float | None
    agreement_ratio: float | None
    error: str | None


@dataclass(frozen=True)
class NonlinearSchurRoundSummary:
    round_index: int
    parent_count: int
    generated_unique_path_count: int
    proposed_count: int
    same_path_proposed_count: int
    new_path_proposed_count: int
    refitted_count: int
    improved_child_count: int
    productive_parent_count: int
    retained_count: int
    retained_path_count: int
    retained_root_count: int
    best_objective_chi2: float
    best_shape_chi2: float
    best_phase_chi2: float
    median_positive_reduction_agreement: float | None


@dataclass(frozen=True)
class NonlinearSchurTransportResult:
    final_modes: tuple[GlobalLatticeMode, ...]
    mode_archive: tuple[GlobalLatticeMode, ...]
    round_portfolios: tuple[tuple[GlobalLatticeMode, ...], ...]
    acquisitions: tuple[NonlinearSchurAcquisitionRecord, ...]
    refit_history: tuple[NonlinearSchurRefitRecord, ...]
    round_history: tuple[NonlinearSchurRoundSummary, ...]
    completed_rounds: int


@dataclass(frozen=True)
class _AcquiredCandidate:
    parent: GlobalLatticeMode
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    conditioning_channel: str
    phase_precision_factor: float
    selected_objective_chi2: float
    selected_closure_chi2: float
    minimum_closure_chi2: float
    selected_exact_closure_chi2: float
    minimum_exact_closure_chi2: float
    source_groups: tuple[str, ...]
    nonlinear_closure_chi2: float | None = None
    changes_integer_path: bool = True


def _candidate_screen_chi2(candidate: _AcquiredCandidate) -> float:
    score = candidate.nonlinear_closure_chi2
    if score is not None and np.isfinite(score):
        return float(score)
    return float(candidate.minimum_exact_closure_chi2)


def _normalized_distance(left: GlobalLatticeMode, right: GlobalLatticeMode) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left.normalized_values, dtype=float)
            - np.asarray(right.normalized_values, dtype=float)
        )
    )


def _shape_phase_pareto_modes(
    modes: Sequence[GlobalLatticeMode],
) -> tuple[GlobalLatticeMode, ...]:
    """Return the nondominated shape/phase envelope in crowding order."""

    frontier = []
    for mode in modes:
        dominated = any(
            other.shape_chi2 <= mode.shape_chi2
            and other.phase_chi2 <= mode.phase_chi2
            and (
                other.shape_chi2 < mode.shape_chi2
                or other.phase_chi2 < mode.phase_chi2
            )
            for other in modes
        )
        if not dominated:
            frontier.append(mode)
    if len(frontier) <= 2:
        return tuple(sorted(frontier, key=lambda mode: mode.objective_chi2))

    crowding = {mode.mode_id: 0.0 for mode in frontier}
    for attribute in ("shape_chi2", "phase_chi2"):
        ordered = sorted(frontier, key=lambda mode: getattr(mode, attribute))
        lower = float(getattr(ordered[0], attribute))
        upper = float(getattr(ordered[-1], attribute))
        crowding[ordered[0].mode_id] = math.inf
        crowding[ordered[-1].mode_id] = math.inf
        span = max(upper - lower, np.finfo(float).eps)
        for index in range(1, len(ordered) - 1):
            if math.isinf(crowding[ordered[index].mode_id]):
                continue
            previous_value = float(getattr(ordered[index - 1], attribute))
            next_value = float(getattr(ordered[index + 1], attribute))
            crowding[ordered[index].mode_id] += (
                next_value - previous_value
            ) / span
    return tuple(
        sorted(
            frontier,
            key=lambda mode: (
                -crowding[mode.mode_id],
                mode.objective_chi2,
            ),
        )
    )


def select_nonlinear_schur_modes(
    modes: Sequence[GlobalLatticeMode],
    *,
    maximum_modes: int,
    maximum_modes_per_path: int,
    maximum_modes_per_root: int,
    minimum_separation: float,
    preferred_mode_ids: Sequence[str] = (),
) -> tuple[GlobalLatticeMode, ...]:
    """Select an evidence envelope without committing to one scalar ranking."""

    if not modes:
        return ()
    target = max(int(maximum_modes), 1)
    path_cap = max(int(maximum_modes_per_path), 1)
    root_cap = max(int(maximum_modes_per_root), 1)
    separation = max(float(minimum_separation), 0.0)
    rankings = [
        tuple(sorted(modes, key=lambda mode: mode.objective_chi2)),
        tuple(sorted(modes, key=lambda mode: mode.shape_chi2)),
        tuple(sorted(modes, key=lambda mode: mode.phase_chi2)),
        tuple(sorted(modes, key=lambda mode: -mode.log_mass)),
    ]
    preferred_ids = {str(value) for value in preferred_mode_ids}
    preferred = tuple(
        sorted(
            (mode for mode in modes if mode.mode_id in preferred_ids),
            key=lambda mode: mode.objective_chi2,
        )
    )
    pareto = _shape_phase_pareto_modes(modes)
    roots = {}
    for mode in modes:
        roots.setdefault(mode.root_mode_id, []).append(mode)
    rankings.extend(
        tuple(sorted(values, key=lambda mode: mode.objective_chi2))
        for values in roots.values()
    )
    selected = []
    selected_ids = set()
    path_counts = {}
    root_counts = {}

    def add(mode: GlobalLatticeMode, enforce_separation: bool) -> bool:
        if mode.mode_id in selected_ids:
            return False
        path = mode.integer_offsets
        root = mode.root_mode_id
        if path_counts.get(path, 0) >= path_cap or root_counts.get(root, 0) >= root_cap:
            return False
        for existing in selected:
            if (
                existing.integer_offsets == path
                and _normalized_distance(existing, mode) <= 1.0e-12
            ):
                return False
            if enforce_separation and _normalized_distance(existing, mode) < separation:
                return False
        selected.append(mode)
        selected_ids.add(mode.mode_id)
        path_counts[path] = path_counts.get(path, 0) + 1
        root_counts[root] = root_counts.get(root, 0) + 1
        return True

    # Exact closure and the two evidence extremes define the mandatory envelope.
    for ranking in rankings[:3]:
        if ranking:
            add(ranking[0], False)
    preferred_quota = max(target // 3, 1)
    preferred_added = 0
    for mode in preferred:
        if add(mode, False):
            preferred_added += 1
        if len(selected) >= target:
            return tuple(selected[:target])
        if preferred_added >= preferred_quota:
            break
    pareto_quota = max(target // 3, 1)
    pareto_added = 0
    for mode in pareto:
        if add(mode, False):
            pareto_added += 1
        if len(selected) >= target:
            return tuple(selected[:target])
        if pareto_added >= pareto_quota:
            break
    for values in roots.values():
        add(min(values, key=lambda mode: mode.objective_chi2), False)
        if len(selected) >= target:
            return tuple(selected[:target])
    rank_length = max(len(ranking) for ranking in rankings)
    for enforce in (True, False):
        for rank in range(rank_length):
            for ranking in rankings:
                if rank < len(ranking):
                    add(ranking[rank], enforce)
                    if len(selected) >= target:
                        return tuple(selected)
    return tuple(selected)


def _acquired_candidates(
    parent: GlobalLatticeMode,
    acquisition: ConditionedPathAcquisitionResult,
) -> tuple[_AcquiredCandidate, ...]:
    candidates = []
    for proposal in acquisition.proposals:
        source_groups = tuple(str(value) for value in proposal.source_groups)
        channel_index = int(proposal.selected_channel_index)
        closure = np.asarray(proposal.channel_closure_chi2, dtype=float)
        exact_closure = np.asarray(
            proposal.channel_exact_closure_chi2,
            dtype=float,
        )
        objective = np.asarray(proposal.channel_objective_chi2, dtype=float)
        candidates.append(
            _AcquiredCandidate(
                parent=parent,
                integer_offsets=tuple(
                    int(value) for value in proposal.integer_offsets
                ),
                normalized_values=np.asarray(
                    proposal.conditioned_normalized_values,
                    dtype=float,
                ),
                conditioning_channel=proposal.conditioning_channel,
                phase_precision_factor=float(
                    proposal.phase_precision_factor
                ),
                selected_objective_chi2=float(objective[channel_index]),
                selected_closure_chi2=float(closure[channel_index]),
                minimum_closure_chi2=float(np.min(closure)),
                selected_exact_closure_chi2=float(
                    exact_closure[channel_index]
                ),
                minimum_exact_closure_chi2=float(np.min(exact_closure)),
                source_groups=source_groups,
                changes_integer_path=(
                    tuple(int(value) for value in proposal.integer_offsets)
                    != parent.integer_offsets
                ),
            )
        )
    return tuple(candidates)


def _screen_nonlinear_closure_candidates(
    likelihood: Any,
    candidates: Sequence[_AcquiredCandidate],
) -> tuple[_AcquiredCandidate, ...]:
    """Evaluate proposal points with the exact nonlinear timing model when available."""

    scorer = getattr(likelihood, "exact_candidate_closure_chi2", None)
    if not callable(scorer):
        return tuple(candidates)
    screened = []
    for candidate in candidates:
        try:
            score = float(
                scorer(
                    candidate.normalized_values,
                    candidate.integer_offsets,
                )
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            score = math.inf
        screened.append(
            replace(candidate, nonlinear_closure_chi2=score)
        )
    return tuple(screened)


def _select_acquired_candidates(
    candidates: Sequence[_AcquiredCandidate],
    *,
    maximum_candidates: int,
    attempted: set[tuple[tuple[int, ...], bytes]],
    attempted_path_counts: dict[tuple[int, ...], int],
    maximum_refits_per_path: int,
) -> tuple[_AcquiredCandidate, ...]:
    if not candidates:
        return ()
    target = max(int(maximum_candidates), 1)
    rankings = [
        tuple(sorted(candidates, key=_candidate_screen_chi2)),
        tuple(sorted(candidates, key=lambda item: item.minimum_exact_closure_chi2)),
        tuple(sorted(candidates, key=lambda item: item.selected_exact_closure_chi2)),
        tuple(sorted(candidates, key=lambda item: item.minimum_closure_chi2)),
        tuple(sorted(candidates, key=lambda item: item.selected_closure_chi2)),
        tuple(sorted(candidates, key=lambda item: item.selected_objective_chi2)),
    ]
    by_root = {}
    by_channel = {}
    by_parent = {}
    by_path = {}
    for item in candidates:
        by_root.setdefault(item.parent.root_mode_id, []).append(item)
        by_channel.setdefault(item.conditioning_channel, []).append(item)
        by_parent.setdefault(item.parent.mode_id, []).append(item)
        by_path.setdefault(item.integer_offsets, []).append(item)
    rankings.extend(
        tuple(sorted(values, key=_candidate_screen_chi2))
        for values in by_root.values()
    )
    rankings.extend(
        tuple(sorted(values, key=_candidate_screen_chi2))
        for values in by_channel.values()
    )
    rankings.extend(
        tuple(sorted(values, key=_candidate_screen_chi2))
        for values in by_path.values()
    )
    selected = []
    selected_keys = set()
    path_counts = {}
    path_cap = max(int(maximum_refits_per_path), 1)

    def add(item: _AcquiredCandidate) -> bool:
        key = (
            item.integer_offsets,
            np.round(item.normalized_values, 11).tobytes(),
        )
        if key in selected_keys or key in attempted:
            return False
        if (
            attempted_path_counts.get(item.integer_offsets, 0)
            + path_counts.get(item.integer_offsets, 0)
            >= path_cap
        ):
            return False
        selected.append(item)
        selected_keys.add(key)
        path_counts[item.integer_offsets] = (
            path_counts.get(item.integer_offsets, 0) + 1
        )
        return True

    # Keep both posterior transport on the current path and structurally new
    # integer paths in the exact-refit portfolio. Neither class is allowed to
    # suppress the other solely through a local linearized score.
    for changes_path in (False, True):
        values = [
            item for item in candidates
            if item.changes_integer_path is changes_path
        ]
        if values:
            add(min(values, key=_candidate_screen_chi2))
        if len(selected) >= target:
            return tuple(selected)

    # Reserve exact-correction budget for every posterior parent and every
    # phase-temperature stratum before global predictive rankings compete.
    parent_quota = max(target // max(len(by_parent), 1), 1)
    ordered_parents = sorted(
        by_parent.values(),
        key=lambda values: min(item.parent.objective_chi2 for item in values),
    )
    for values in ordered_parents:
        parent_added = 0
        for changes_path in (False, True):
            class_values = [
                item for item in values
                if item.changes_integer_path is changes_path
            ]
            if class_values and add(min(class_values, key=_candidate_screen_chi2)):
                parent_added += 1
            if parent_added >= parent_quota or len(selected) >= target:
                break
        if parent_added >= parent_quota or len(selected) >= target:
            if len(selected) >= target:
                return tuple(selected)
            continue
        by_phase_factor = {}
        for item in values:
            by_phase_factor.setdefault(item.phase_precision_factor, []).append(item)
        for phase_factor in sorted(by_phase_factor, reverse=True):
            item = min(
                by_phase_factor[phase_factor],
                key=_candidate_screen_chi2,
            )
            if add(item):
                parent_added += 1
            if parent_added >= parent_quota or len(selected) >= target:
                break
        if parent_added < parent_quota and len(selected) < target:
            local_rankings = (
                tuple(
                    sorted(
                        values,
                        key=lambda item: item.minimum_exact_closure_chi2,
                    )
                ),
                tuple(
                    sorted(
                        values,
                        key=lambda item: item.selected_objective_chi2,
                    )
                ),
                tuple(
                    sorted(
                        values,
                        key=lambda item: item.minimum_closure_chi2,
                    )
                ),
            )
            local_rank_length = max(len(ranking) for ranking in local_rankings)
            for rank in range(local_rank_length):
                for ranking in local_rankings:
                    if rank < len(ranking) and add(ranking[rank]):
                        parent_added += 1
                    if parent_added >= parent_quota or len(selected) >= target:
                        break
                if parent_added >= parent_quota or len(selected) >= target:
                    break
        if len(selected) >= target:
            return tuple(selected)

    rank_length = max(len(ranking) for ranking in rankings)
    for rank in range(rank_length):
        for ranking in rankings:
            if rank < len(ranking):
                add(ranking[rank])
                if len(selected) >= target:
                    return tuple(selected)
    return tuple(selected)


def run_nonlinear_schur_transport(
    *,
    likelihood: Any,
    evaluator: Any,
    coordinates: Any,
    initial_modes: Sequence[GlobalLatticeMode],
    config: NonlinearSchurTransportConfig | None = None,
    mode_callback: Callable[[int, int, GlobalLatticeMode], None] | None = None,
    checkpoint_callback: Callable[
        [
            int,
            tuple[GlobalLatticeMode, ...],
            NonlinearSchurRoundSummary,
        ],
        None,
    ]
    | None = None,
    refit_callback: Callable[[NonlinearSchurRefitRecord], None] | None = None,
    refit_function: Callable[..., GlobalLatticeMode] = refit_fixed_path_proposal,
) -> NonlinearSchurTransportResult:
    """Iterate complete-path Schur acquisition over a multimode exact portfolio."""

    active = config or NonlinearSchurTransportConfig()
    if not initial_modes:
        raise ValueError("Nonlinear Schur transport requires exact seed modes.")
    archive = list(initial_modes)
    acquisitions = []
    refit_history = []
    round_history = []
    round_portfolios = []
    attempted = {
        (
            mode.integer_offsets,
            np.round(np.asarray(mode.normalized_values, dtype=float), 11).tobytes(),
        )
        for mode in initial_modes
    }
    # Seed modes describe the starting posterior portfolio and do not count as exact
    # refit attempts. Counting them against the per-path refit budget can
    # prevent a multiply represented path from receiving any nonlinear
    # closure.
    attempted_path_counts: dict[tuple[int, ...], int] = {}
    modes_by_id = {mode.mode_id: mode for mode in initial_modes}
    productive_threshold = max(float(active.minimum_productive_reduction), 0.0)
    initial_pareto_ids = {
        mode.mode_id for mode in _shape_phase_pareto_modes(initial_modes)
    }
    preferred_mode_ids = tuple(
        mode.mode_id
        for mode in initial_modes
        if mode.mode_id in initial_pareto_ids
        and mode.parent_mode_id in modes_by_id
        and mode.objective_chi2
        < modes_by_id[mode.parent_mode_id].objective_chi2 - productive_threshold
    )
    portfolio = select_nonlinear_schur_modes(
        initial_modes,
        maximum_modes=active.retained_modes,
        maximum_modes_per_path=active.maximum_modes_per_path,
        maximum_modes_per_root=active.maximum_modes_per_root,
        minimum_separation=active.mode_separation,
        preferred_mode_ids=preferred_mode_ids,
    )
    refit_config = GlobalLatticeTrackerConfig(
        prior_precision=0.0,
        fixed_path_max_nfev=max(int(active.fixed_path_max_nfev), 1),
        phase_tempering_ladder=tuple(
            float(value) for value in active.phase_tempering_ladder
        ),
    )
    for local_round_index in range(max(int(active.round_count), 0)):
        round_index = max(int(active.round_index_offset), 0) + local_round_index
        parents = select_nonlinear_schur_modes(
            portfolio,
            maximum_modes=active.parent_modes_per_round,
            maximum_modes_per_path=active.maximum_modes_per_path,
            maximum_modes_per_root=active.maximum_modes_per_root,
            minimum_separation=active.mode_separation,
            preferred_mode_ids=preferred_mode_ids,
        )
        candidates = []
        generated_count = 0
        excluded_paths = tuple(
            path
            for path, count in attempted_path_counts.items()
            if count >= max(int(active.maximum_refits_per_path), 1)
        )
        for parent in parents:
            try:
                acquisition = likelihood.acquire_conditioned_paths_from_full_chart(
                    parent.normalized_values,
                    parent.covariance,
                    seed_paths=(parent.integer_offsets,),
                    excluded_paths=excluded_paths,
                    config=active.acquisition_config,
                )
            except (ArithmeticError, RuntimeError, ValueError):
                continue
            acquisitions.append(
                NonlinearSchurAcquisitionRecord(
                    round_index=round_index,
                    parent_mode_id=parent.mode_id,
                    parent_root_mode_id=parent.root_mode_id,
                    result=acquisition,
                )
            )
            generated_count += acquisition.generated_unique_path_count
            candidates.extend(_acquired_candidates(parent, acquisition))
        candidates = list(_screen_nonlinear_closure_candidates(likelihood, candidates))
        selected_candidates = _select_acquired_candidates(
            candidates,
            maximum_candidates=active.exact_refits_per_round,
            attempted=attempted,
            attempted_path_counts=attempted_path_counts,
            maximum_refits_per_path=active.maximum_refits_per_path,
        )
        children = []
        improved_children = []
        productive_parent_ids = set()
        positive_agreements = []
        for proposal_index, candidate in enumerate(selected_candidates):
            key = (
                candidate.integer_offsets,
                np.round(candidate.normalized_values, 11).tobytes(),
            )
            attempted.add(key)
            attempted_path_counts[candidate.integer_offsets] = (
                attempted_path_counts.get(candidate.integer_offsets, 0) + 1
            )
            parameter_values = np.asarray(
                coordinates.to_physical(
                    coordinates.denormalize(candidate.normalized_values)
                ),
                dtype=float,
            )
            proposal = LatticeProposal(
                root_mode_id=candidate.parent.root_mode_id,
                parent_mode_id=candidate.parent.mode_id,
                source=(
                    (
                        "nonlinear-schur-new-path-transport|"
                        if candidate.changes_integer_path
                        else "nonlinear-schur-continuous-transport|"
                    )
                    + "|".join(candidate.source_groups)
                ),
                active_sessions=tuple(
                    range(1, len(candidate.integer_offsets) + 1)
                ),
                integer_offsets=candidate.integer_offsets,
                initial_normalized_values=candidate.normalized_values,
                squared_mahalanobis=candidate.selected_objective_chi2,
                log_predictive_density=-0.5
                * _candidate_screen_chi2(candidate),
                kappa=1.0,
                initial_parameter_values=parameter_values,
            )
            candidate_refit_config = (
                refit_config
                if candidate.changes_integer_path
                else replace(refit_config, phase_tempering_ladder=(1.0,))
            )
            try:
                child = refit_function(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    anchor_session=0,
                    proposal=proposal,
                    config=candidate_refit_config,
                    mode_id=(
                        f"nonlinear_schur_r{round_index:03d}_"
                        f"{proposal_index:05d}"
                    ),
                )
                archive.append(child)
                parent_objective = float(candidate.parent.objective_chi2)
                predicted_reduction = (
                    parent_objective - _candidate_screen_chi2(candidate)
                )
                actual_reduction = parent_objective - float(child.objective_chi2)
                agreement_ratio = (
                    actual_reduction / predicted_reduction
                    if predicted_reduction > 1.0e-12
                    else None
                )
                if actual_reduction > productive_threshold:
                    improved_children.append((actual_reduction, child))
                    productive_parent_ids.add(candidate.parent.mode_id)
                    if agreement_ratio is not None and np.isfinite(agreement_ratio):
                        positive_agreements.append(float(agreement_ratio))
                if (
                    not active.retain_only_productive_children
                    or actual_reduction > productive_threshold
                ):
                    children.append(child)
                refit_record = NonlinearSchurRefitRecord(
                        round_index=round_index,
                        proposal_index=proposal_index,
                        parent_mode_id=candidate.parent.mode_id,
                        root_mode_id=candidate.parent.root_mode_id,
                        integer_offsets=candidate.integer_offsets,
                        proposal_class=(
                            "new-integer-path"
                            if candidate.changes_integer_path
                            else "same-path-continuous"
                        ),
                        conditioning_channel=candidate.conditioning_channel,
                        predictive_closure_chi2=(
                            _candidate_screen_chi2(candidate)
                        ),
                        parent_objective_chi2=parent_objective,
                        predicted_reduction=predicted_reduction,
                        success=True,
                        objective_chi2=float(child.objective_chi2),
                        actual_reduction=actual_reduction,
                        agreement_ratio=agreement_ratio,
                        error=None,
                    )
                refit_history.append(refit_record)
                if refit_callback is not None:
                    refit_callback(refit_record)
                if mode_callback is not None:
                    mode_callback(round_index, proposal_index, child)
            except Exception as exc:
                refit_record = NonlinearSchurRefitRecord(
                        round_index=round_index,
                        proposal_index=proposal_index,
                        parent_mode_id=candidate.parent.mode_id,
                        root_mode_id=candidate.parent.root_mode_id,
                        integer_offsets=candidate.integer_offsets,
                        proposal_class=(
                            "new-integer-path"
                            if candidate.changes_integer_path
                            else "same-path-continuous"
                        ),
                        conditioning_channel=candidate.conditioning_channel,
                        predictive_closure_chi2=(
                            _candidate_screen_chi2(candidate)
                        ),
                        parent_objective_chi2=float(
                            candidate.parent.objective_chi2
                        ),
                        predicted_reduction=float(
                            candidate.parent.objective_chi2
                            - _candidate_screen_chi2(candidate)
                        ),
                        success=False,
                        objective_chi2=math.inf,
                        actual_reduction=None,
                        agreement_ratio=None,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                refit_history.append(refit_record)
                if refit_callback is not None:
                    refit_callback(refit_record)
        combined_modes = tuple(portfolio) + tuple(children)
        pareto_ids = {
            mode.mode_id for mode in _shape_phase_pareto_modes(combined_modes)
        }
        preferred_mode_ids = tuple(
            mode.mode_id
            for _, mode in sorted(
                improved_children,
                key=lambda item: (-item[0], item[1].objective_chi2),
            )
            if mode.mode_id in pareto_ids
        )
        portfolio = select_nonlinear_schur_modes(
            combined_modes,
            maximum_modes=active.retained_modes,
            maximum_modes_per_path=active.maximum_modes_per_path,
            maximum_modes_per_root=active.maximum_modes_per_root,
            minimum_separation=active.mode_separation,
            preferred_mode_ids=preferred_mode_ids,
        )
        summary = NonlinearSchurRoundSummary(
            round_index=round_index,
            parent_count=len(parents),
            generated_unique_path_count=generated_count,
            proposed_count=len(selected_candidates),
            same_path_proposed_count=sum(
                not candidate.changes_integer_path
                for candidate in selected_candidates
            ),
            new_path_proposed_count=sum(
                candidate.changes_integer_path
                for candidate in selected_candidates
            ),
            refitted_count=len(children),
            improved_child_count=len(improved_children),
            productive_parent_count=len(productive_parent_ids),
            retained_count=len(portfolio),
            retained_path_count=len({mode.integer_offsets for mode in portfolio}),
            retained_root_count=len({mode.root_mode_id for mode in portfolio}),
            best_objective_chi2=min(mode.objective_chi2 for mode in portfolio),
            best_shape_chi2=min(mode.shape_chi2 for mode in portfolio),
            best_phase_chi2=min(mode.phase_chi2 for mode in portfolio),
            median_positive_reduction_agreement=(
                float(np.median(positive_agreements))
                if positive_agreements
                else None
            ),
        )
        round_history.append(summary)
        round_portfolios.append(portfolio)
        if checkpoint_callback is not None:
            checkpoint_callback(round_index, portfolio, summary)
    return NonlinearSchurTransportResult(
        final_modes=portfolio,
        mode_archive=tuple(archive),
        round_portfolios=tuple(round_portfolios),
        acquisitions=tuple(acquisitions),
        refit_history=tuple(refit_history),
        round_history=tuple(round_history),
        completed_rounds=max(int(active.round_count), 0),
    )


__all__ = [
    "NonlinearSchurAcquisitionRecord",
    "NonlinearSchurRefitRecord",
    "NonlinearSchurRoundSummary",
    "NonlinearSchurTransportConfig",
    "NonlinearSchurTransportResult",
    "run_nonlinear_schur_transport",
    "select_nonlinear_schur_modes",
]
