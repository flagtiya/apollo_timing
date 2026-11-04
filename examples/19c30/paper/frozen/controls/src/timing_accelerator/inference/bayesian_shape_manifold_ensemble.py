from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import emcee
import numpy as np

from timing_accelerator.inference.bayesian_global_lattice_mode_tracker import (
    GlobalLatticeMode,
    GlobalLatticeTrackerConfig,
    InvariantMode,
    LatticeProposal,
    _physical_values,
    refit_fixed_path_proposal,
    select_diverse_lattice_modes,
)
from timing_accelerator.inference.bayesian_joint_lattice_population import (
    JointLatticePopulationConfig,
    generate_joint_lattice_proposals,
)
from timing_accelerator.inference.collapsed_integer_posterior import (
    conditional_integer_paths,
    exact_relative_chi2,
)
from timing_accelerator.inference.integer_least_squares import (
    solve_integer_least_squares,
)


@dataclass(frozen=True)
class ShapeManifoldEnsembleConfig:
    walker_count: int = 64
    step_count: int = 500
    checkpoint_interval: int = 25
    prior_precision: float = 1.0
    initialization_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    de_move_weight: float = 0.50
    snooker_move_weight: float = 0.20
    stretch_move_weight: float = 0.30
    random_seed: int = 20260724


@dataclass(frozen=True)
class ShapeManifoldEnsembleProgress:
    iteration: int
    mean_acceptance_fraction: float
    maximum_log_posterior: float
    median_log_posterior: float


@dataclass(frozen=True)
class ShapeManifoldEnsembleResult:
    iteration: int
    walker_count: int
    mean_acceptance_fraction: float
    final_normalized_values: np.ndarray
    final_log_posterior: np.ndarray
    backend_path: str | None


@dataclass(frozen=True)
class ShapeIntegerBasinCandidate:
    candidate_id: str
    sample_index: int
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    shape_chi2: float
    phase_chi2: float
    prior_chi2: float
    objective_chi2: float
    log_shape_posterior: float


@dataclass(frozen=True)
class ShapeManifoldJointProposalResult:
    """Complete-path proposals built from multiple nonlinear shape charts."""

    selected_charts: tuple[ShapeIntegerBasinCandidate, ...]
    chart_modes: tuple[GlobalLatticeMode, ...]
    proposals: tuple[LatticeProposal, ...]


@dataclass(frozen=True)
class NonlinearPhaseMixtureResult:
    """Complete paths from a joint continuous/phase posterior mixture."""

    evaluated_sample_count: int
    chart_sample_indices: tuple[int, ...]
    neighbor_count: int
    proposals: tuple[LatticeProposal, ...]
    trend_parameter_indices: tuple[int, ...] = ()
    manifold_parameter_indices: tuple[int, ...] = ()
    component_diagnostics: tuple[dict[str, Any], ...] = ()


def shape_log_posterior(
    normalized_values: Sequence[float],
    *,
    evaluator: Any,
    coordinates: Any,
    prior_precision: float,
) -> float:
    values = np.asarray(normalized_values, dtype=float)
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if (
        values.shape != lower.shape
        or not np.all(np.isfinite(values))
        or np.any(values <= lower)
        or np.any(values >= upper)
    ):
        return -math.inf
    try:
        profile = evaluator.evaluate_values(_physical_values(coordinates, values))
        shape_chi2 = float(profile.chi2)
    except Exception:
        return -math.inf
    prior_chi2 = max(float(prior_precision), 0.0) * float(values @ values)
    objective = shape_chi2 + prior_chi2
    return -0.5 * objective if np.isfinite(objective) else -math.inf


def _covariance_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (
        np.asarray(covariance, dtype=float)
        + np.asarray(covariance, dtype=float).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = max(1.0e-12 * scale, np.finfo(float).eps)
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, floor))) @ eigenvectors.T


def initialize_shape_manifold_walkers(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: ShapeManifoldEnsembleConfig | None = None,
) -> np.ndarray:
    active = config or ShapeManifoldEnsembleConfig()
    if not invariant_modes:
        raise ValueError("Shape-manifold sampling requires invariant seed modes.")
    dimension = len(np.asarray(invariant_modes[0].normalized_values, dtype=float))
    walker_count = max(int(active.walker_count), 2 * dimension + 2)
    if walker_count % 2:
        walker_count += 1
    scales = tuple(
        float(value)
        for value in active.initialization_scales
        if np.isfinite(float(value)) and float(value) > 0.0
    ) or (1.0,)
    roots = tuple(_covariance_root(mode.covariance) for mode in invariant_modes)
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    rng = np.random.default_rng(max(int(active.random_seed), 0))
    walkers: list[np.ndarray] = []
    maximum_attempts = max(10_000, 200 * walker_count)
    for attempt in range(maximum_attempts):
        mode_index = attempt % len(invariant_modes)
        scale = scales[(attempt // len(invariant_modes)) % len(scales)]
        center = np.asarray(
            invariant_modes[mode_index].normalized_values,
            dtype=float,
        )
        candidate = center + scale * roots[mode_index] @ rng.normal(size=dimension)
        if np.any(candidate <= lower) or np.any(candidate >= upper):
            continue
        log_posterior = shape_log_posterior(
            candidate,
            evaluator=evaluator,
            coordinates=coordinates,
            prior_precision=active.prior_precision,
        )
        if not np.isfinite(log_posterior):
            continue
        walkers.append(candidate)
        if len(walkers) >= walker_count:
            break
    if len(walkers) != walker_count:
        raise RuntimeError(
            f"Only initialized {len(walkers)} of {walker_count} shape walkers."
        )
    return np.vstack(walkers)


def run_shape_manifold_ensemble(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: ShapeManifoldEnsembleConfig | None = None,
    backend_path: Path | None = None,
    resume: bool = False,
    progress_callback: Callable[[ShapeManifoldEnsembleProgress], None] | None = None,
) -> ShapeManifoldEnsembleResult:
    """Sample the exact bounded shape posterior with mature ensemble kernels."""

    active = config or ShapeManifoldEnsembleConfig()
    dimension = len(np.asarray(invariant_modes[0].normalized_values, dtype=float))
    walker_count = max(int(active.walker_count), 2 * dimension + 2)
    if walker_count % 2:
        walker_count += 1
    backend = None
    initial_state: np.ndarray | None
    if backend_path is not None:
        resolved = Path(backend_path).resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        backend = emcee.backends.HDFBackend(str(resolved))
        if resume:
            if backend.iteration <= 0:
                raise ValueError("The requested shape-chain checkpoint is empty.")
            if backend.shape != (walker_count, dimension):
                raise ValueError("The shape-chain checkpoint has incompatible dimensions.")
            initial_state = None
        else:
            backend.reset(walker_count, dimension)
            initial_state = initialize_shape_manifold_walkers(
                evaluator=evaluator,
                coordinates=coordinates,
                invariant_modes=invariant_modes,
                config=active,
            )
    else:
        if resume:
            raise ValueError("Shape-chain resume requires an HDF backend path.")
        initial_state = initialize_shape_manifold_walkers(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            config=active,
        )

    weights = np.asarray(
        [
            max(float(active.de_move_weight), 0.0),
            max(float(active.snooker_move_weight), 0.0),
            max(float(active.stretch_move_weight), 0.0),
        ],
        dtype=float,
    )
    if float(np.sum(weights)) <= 0.0:
        raise ValueError("At least one shape-manifold ensemble move must be active.")
    moves = [
        (emcee.moves.DEMove(), float(weights[0])),
        (emcee.moves.DESnookerMove(), float(weights[1])),
        (emcee.moves.StretchMove(), float(weights[2])),
    ]
    moves = [(move, weight) for move, weight in moves if weight > 0.0]
    np.random.seed(max(int(active.random_seed), 0))
    sampler = emcee.EnsembleSampler(
        walker_count,
        dimension,
        lambda values: shape_log_posterior(
            values,
            evaluator=evaluator,
            coordinates=coordinates,
            prior_precision=active.prior_precision,
        ),
        moves=moves,
        backend=backend,
    )
    remaining = max(int(active.step_count), 0)
    interval = max(int(active.checkpoint_interval), 1)
    state: Any = initial_state
    first_chunk = True
    while remaining > 0:
        chunk = min(interval, remaining)
        state = sampler.run_mcmc(
            state,
            chunk,
            progress=False,
            skip_initial_state_check=first_chunk,
        )
        first_chunk = False
        remaining -= chunk
        log_probability = np.asarray(state.log_prob, dtype=float)
        summary = ShapeManifoldEnsembleProgress(
            iteration=int(sampler.iteration),
            mean_acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
            maximum_log_posterior=float(np.max(log_probability)),
            median_log_posterior=float(np.median(log_probability)),
        )
        if progress_callback is not None:
            progress_callback(summary)
    last = sampler.get_last_sample()
    return ShapeManifoldEnsembleResult(
        iteration=int(sampler.iteration),
        walker_count=walker_count,
        mean_acceptance_fraction=float(np.mean(sampler.acceptance_fraction)),
        final_normalized_values=np.asarray(last.coords, dtype=float),
        final_log_posterior=np.asarray(last.log_prob, dtype=float),
        backend_path=str(Path(backend_path).resolve()) if backend_path else None,
    )


def score_shape_samples_on_integer_lattice(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_samples: Sequence[Sequence[float]],
    log_shape_posterior_values: Sequence[float] | None = None,
    prior_precision: float = 1.0,
    integer_candidates_per_sample: int = 1,
    maximum_candidates: int = 512,
    maximum_modes_per_path: int = 2,
    minimum_normalized_separation: float = 0.005,
) -> tuple[ShapeIntegerBasinCandidate, ...]:
    """Condition complete paths on samples from the nonlinear shape manifold."""

    samples = np.asarray(normalized_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("Shape-manifold samples must be a two-dimensional array.")
    if log_shape_posterior_values is None:
        log_values = np.full(samples.shape[0], np.nan, dtype=float)
    else:
        log_values = np.asarray(log_shape_posterior_values, dtype=float)
        if log_values.shape != (samples.shape[0],):
            raise ValueError("Shape sample log posterior values are misaligned.")
    raw: list[ShapeIntegerBasinCandidate] = []
    for sample_index, normalized in enumerate(samples):
        try:
            parameters = _physical_values(coordinates, normalized)
            profile = evaluator.evaluate_values(parameters)
            paths = conditional_integer_paths(
                profile,
                candidate_count=max(int(integer_candidates_per_sample), 1),
            )
        except Exception:
            continue
        shape_chi2 = float(profile.chi2)
        prior_chi2 = max(float(prior_precision), 0.0) * float(
            normalized @ normalized
        )
        for path_index, path in enumerate(paths):
            phase_chi2 = exact_relative_chi2(profile, path.integer_offsets)
            objective = shape_chi2 + phase_chi2 + prior_chi2
            raw.append(
                ShapeIntegerBasinCandidate(
                    candidate_id=f"shape_sample_{sample_index:07d}_{path_index:02d}",
                    sample_index=sample_index,
                    integer_offsets=tuple(path.integer_offsets),
                    normalized_values=np.asarray(normalized, dtype=float).copy(),
                    parameter_values=np.asarray(parameters, dtype=float).copy(),
                    shape_chi2=shape_chi2,
                    phase_chi2=phase_chi2,
                    prior_chi2=prior_chi2,
                    objective_chi2=objective,
                    log_shape_posterior=float(log_values[sample_index]),
                )
            )
    selected: list[ShapeIntegerBasinCandidate] = []
    path_counts: dict[tuple[int, ...], int] = {}
    for candidate in sorted(raw, key=lambda item: item.objective_chi2):
        if path_counts.get(candidate.integer_offsets, 0) >= max(
            int(maximum_modes_per_path), 1
        ):
            continue
        if any(
            candidate.integer_offsets == existing.integer_offsets
            and np.linalg.norm(
                candidate.normalized_values - existing.normalized_values
            ) < max(float(minimum_normalized_separation), 0.0)
            for existing in selected
        ):
            continue
        selected.append(candidate)
        path_counts[candidate.integer_offsets] = (
            path_counts.get(candidate.integer_offsets, 0) + 1
        )
        if len(selected) >= max(int(maximum_candidates), 1):
            break
    return tuple(selected)


def select_diverse_shape_manifold_charts(
    candidates: Sequence[ShapeIntegerBasinCandidate],
    *,
    maximum_charts: int = 32,
    minimum_normalized_separation: float = 0.01,
    maximum_shape_delta: float = 50.0,
) -> tuple[ShapeIntegerBasinCandidate, ...]:
    """Select posterior-supported charts by deterministic maximin coverage.

    Conditional integer score is used only to choose the first chart. The
    remaining allocation covers the nonlinear shape manifold in robustly
    standardized coordinates, preventing one lucky local integer rounding
    from consuming the complete-path proposal budget.
    """

    finite = tuple(
        candidate
        for candidate in candidates
        if np.isfinite(candidate.shape_chi2)
        and np.isfinite(candidate.objective_chi2)
        and np.all(np.isfinite(candidate.normalized_values))
    )
    if not finite:
        return ()
    target = min(max(int(maximum_charts), 1), len(finite))
    best_shape = min(candidate.shape_chi2 for candidate in finite)
    supported = tuple(
        candidate
        for candidate in finite
        if candidate.shape_chi2
        <= best_shape + max(float(maximum_shape_delta), 0.0)
    ) or finite
    values = np.vstack(
        [np.asarray(candidate.normalized_values, dtype=float) for candidate in supported]
    )
    lower = np.quantile(values, 0.05, axis=0)
    upper = np.quantile(values, 0.95, axis=0)
    scales = np.maximum(upper - lower, 1.0e-8)
    standardized = (values - np.median(values, axis=0)) / scales

    first = min(
        range(len(supported)),
        key=lambda index: (
            supported[index].objective_chi2,
            supported[index].shape_chi2,
            supported[index].candidate_id,
        ),
    )
    selected_indices = [first]
    selected_ids = {first}
    separation = max(float(minimum_normalized_separation), 0.0)
    while len(selected_indices) < target:
        remaining = [index for index in range(len(supported)) if index not in selected_ids]
        if not remaining:
            break
        distances = np.asarray(
            [
                min(
                    float(np.linalg.norm(standardized[index] - standardized[chosen]))
                    for chosen in selected_indices
                )
                for index in remaining
            ],
            dtype=float,
        )
        order = sorted(
            range(len(remaining)),
            key=lambda local: (
                -distances[local],
                supported[remaining[local]].objective_chi2,
                supported[remaining[local]].candidate_id,
            ),
        )
        chosen_local = order[0]
        if distances[chosen_local] < separation:
            break
        chosen = remaining[chosen_local]
        selected_indices.append(chosen)
        selected_ids.add(chosen)
    return tuple(supported[index] for index in selected_indices)


def shape_manifold_candidates_to_joint_modes(
    candidates: Sequence[ShapeIntegerBasinCandidate],
    *,
    evaluator: Any,
    coordinates: Any,
    prior_precision: float = 1.0,
) -> tuple[GlobalLatticeMode, ...]:
    """Adapt shape charts to the all-session joint-lattice interface."""

    modes: list[GlobalLatticeMode] = []
    active_prior = max(float(prior_precision), 0.0)
    for index, candidate in enumerate(candidates):
        normalized = np.asarray(candidate.normalized_values, dtype=float).copy()
        parameter_values = _physical_values(coordinates, normalized)
        profile = evaluator.evaluate_values(parameter_values)
        path = tuple(int(value) for value in candidate.integer_offsets)
        if len(path) + 1 != int(profile.session_count):
            raise ValueError(
                "A shape chart integer path must cover every source session."
            )
        phase_chi2 = exact_relative_chi2(profile, path)
        shape_chi2 = float(profile.chi2)
        prior_chi2 = active_prior * float(normalized @ normalized)
        objective = shape_chi2 + phase_chi2 + prior_chi2
        modes.append(
            GlobalLatticeMode(
                mode_id=f"shape_joint_chart_{index:04d}",
                root_mode_id=str(candidate.candidate_id),
                parent_mode_id=str(candidate.candidate_id),
                proposal_source="shape-manifold-chart",
                anchor_session=0,
                active_sessions=tuple(range(1, int(profile.session_count))),
                integer_offsets=path,
                parameter_values=parameter_values,
                normalized_values=normalized,
                covariance=np.eye(normalized.size, dtype=float),
                shape_chi2=shape_chi2,
                phase_chi2=phase_chi2,
                prior_chi2=prior_chi2,
                objective_chi2=objective,
                log_mass=-0.5 * objective,
                posterior_mass=0.0,
                squared_mahalanobis=phase_chi2,
                nfev=1,
                success=True,
            )
        )
    return tuple(modes)


def generate_shape_manifold_joint_lattice_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    candidates: Sequence[ShapeIntegerBasinCandidate],
    maximum_charts: int = 32,
    minimum_chart_separation: float = 0.01,
    maximum_shape_delta: float = 50.0,
    config: JointLatticePopulationConfig | None = None,
) -> ShapeManifoldJointProposalResult:
    """Build complete-path Schur proposals from a nonlinear chart ensemble."""

    active = config or JointLatticePopulationConfig()
    charts = select_diverse_shape_manifold_charts(
        candidates,
        maximum_charts=maximum_charts,
        minimum_normalized_separation=minimum_chart_separation,
        maximum_shape_delta=maximum_shape_delta,
    )
    modes = shape_manifold_candidates_to_joint_modes(
        charts,
        evaluator=evaluator,
        coordinates=coordinates,
        prior_precision=active.prior_precision,
    )
    proposals = generate_joint_lattice_proposals(
        evaluator=evaluator,
        coordinates=coordinates,
        parent_modes=modes,
        config=active,
    )
    return ShapeManifoldJointProposalResult(
        selected_charts=charts,
        chart_modes=modes,
        proposals=proposals,
    )


def generate_nonlinear_phase_mixture_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    normalized_samples: Sequence[Sequence[float]],
    maximum_charts: int = 32,
    neighbors_per_chart: int = 64,
    covariance_scales: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    lattice_candidates_per_chart: int = 2,
    maximum_proposals: int = 256,
    maximum_lattice_nodes: int = 1_000_000,
    prior_precision: float = 0.0,
) -> NonlinearPhaseMixtureResult:
    """Infer complete paths from joint nonlinear shape-posterior components.

    The spin trend can move the phase sequence by thousands of turns while the
    nonlinear orbit remains in one small shape-supported region. Components are
    therefore localized only in the non-spin manifold coordinates. Within each
    component APOLLO estimates the joint Gaussian of continuous coordinates and
    the complete relative-session phase sequence. Integer least squares
    marginalizes the continuous coordinates, and Gaussian conditioning then
    computes the continuous start associated with each proposed integer path.

    This is a proposal distribution, not calibrated evidence. Every returned
    state still requires correction by the exact nonlinear fixed-path target.
    """

    samples = np.asarray(normalized_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("Nonlinear phase-mixture samples must be a nonempty matrix.")
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if samples.shape[1:] != lower.shape:
        raise ValueError("Nonlinear phase-mixture samples have the wrong dimension.")

    valid_samples: list[np.ndarray] = []
    profiles: list[Any] = []
    shape_chi2: list[float] = []
    relative_offsets: list[np.ndarray] = []
    session_variances: list[np.ndarray] = []
    for sample in samples:
        if (
            not np.all(np.isfinite(sample))
            or np.any(sample <= lower)
            or np.any(sample >= upper)
        ):
            continue
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, sample))
            offsets = np.asarray(profile.session_offsets, dtype=float)
            variances = np.asarray(profile.session_offset_variances, dtype=float)
        except Exception:
            continue
        if (
            offsets.ndim != 1
            or offsets.size <= 1
            or variances.shape != offsets.shape
            or not np.all(np.isfinite(offsets))
            or not np.all(np.isfinite(variances) & (variances > 0.0))
        ):
            continue
        valid_samples.append(np.asarray(sample, dtype=float).copy())
        profiles.append(profile)
        shape_chi2.append(float(profile.chi2))
        relative_offsets.append(offsets[1:] - offsets[0])
        session_variances.append(variances)
    if not valid_samples:
        return NonlinearPhaseMixtureResult(0, (), 0, ())

    values = np.vstack(valid_samples)
    phase_values = np.vstack(relative_offsets)
    shape_values = np.asarray(shape_chi2, dtype=float)
    variance_values = np.vstack(session_variances)
    parameter_names = tuple(
        str(name).upper()
        for name in getattr(coordinates, "parameter_names", ())
    )
    if len(parameter_names) == values.shape[1]:
        trend_indices = tuple(
            index
            for index, name in enumerate(parameter_names)
            if name in {"F0", "F1"}
        )
    else:
        trend_indices = ()
    manifold_indices = tuple(
        index for index in range(values.shape[1]) if index not in trend_indices
    )
    if not manifold_indices:
        manifold_indices = tuple(range(values.shape[1]))

    manifold_values = values[:, manifold_indices]
    robust_lower = np.quantile(manifold_values, 0.05, axis=0)
    robust_upper = np.quantile(manifold_values, 0.95, axis=0)
    robust_scale = np.maximum(robust_upper - robust_lower, 1.0e-8)
    standardized = (
        manifold_values - np.median(manifold_values, axis=0)
    ) / robust_scale

    chart_target = min(max(int(maximum_charts), 1), values.shape[0])
    chart_indices = [int(np.argmin(shape_values))]
    while len(chart_indices) < chart_target:
        distances = np.asarray(
            [
                min(
                    float(np.linalg.norm(standardized[index] - standardized[chosen]))
                    for chosen in chart_indices
                )
                for index in range(values.shape[0])
            ],
            dtype=float,
        )
        distances[np.asarray(chart_indices, dtype=int)] = -np.inf
        selected = int(np.argmax(distances))
        if not np.isfinite(distances[selected]) or distances[selected] <= 0.0:
            break
        chart_indices.append(selected)

    scales = tuple(
        dict.fromkeys(
            float(value)
            for value in covariance_scales
            if np.isfinite(float(value)) and float(value) > 0.0
        )
    ) or (1.0,)
    neighbor_count = min(max(int(neighbors_per_chart), 2), values.shape[0])
    active_sessions = tuple(range(1, phase_values.shape[1] + 1))
    records: list[tuple[float, int, LatticeProposal]] = []
    family_records: dict[int, list[tuple[float, LatticeProposal]]] = {}
    component_diagnostics: list[dict[str, Any]] = []
    active_prior = max(float(prior_precision), 0.0)
    normalized_lower = np.asarray(coordinates.normalized_lower, dtype=float)
    normalized_upper = np.asarray(coordinates.normalized_upper, dtype=float)

    for chart_order, chart_index in enumerate(chart_indices):
        distances = np.linalg.norm(
            standardized - standardized[chart_index],
            axis=1,
        )
        neighbors = np.argsort(distances)[:neighbor_count]
        local_values = values[neighbors]
        local_phase = phase_values[neighbors]
        value_center = np.mean(local_values, axis=0)
        phase_center = np.mean(local_phase, axis=0)
        centered_values = local_values - value_center
        centered_phase = local_phase - phase_center
        if local_phase.shape[0] > 1:
            denominator = float(local_phase.shape[0] - 1)
            empirical_covariance = centered_phase.T @ centered_phase / denominator
            cross_covariance = centered_values.T @ centered_phase / denominator
        else:
            empirical_covariance = np.zeros(
                (phase_values.shape[1], phase_values.shape[1]),
                dtype=float,
            )
            cross_covariance = np.zeros(
                (values.shape[1], phase_values.shape[1]),
                dtype=float,
            )
        empirical_covariance = np.atleast_2d(
            np.asarray(empirical_covariance, dtype=float)
        )
        cross_covariance = np.asarray(cross_covariance, dtype=float)
        median_variance = np.median(variance_values[neighbors], axis=0)
        measurement_covariance = (
            np.diag(median_variance[1:]) + median_variance[0]
        )
        component_paths: set[tuple[int, ...]] = set()
        raw_candidate_count = 0
        best_start_objective = math.inf

        for covariance_scale in scales:
            scale_squared = covariance_scale**2
            predictive_covariance = (
                measurement_covariance
                + scale_squared * empirical_covariance
            )
            predictive_covariance = 0.5 * (
                predictive_covariance + predictive_covariance.T
            )
            eigenvalues, eigenvectors = np.linalg.eigh(predictive_covariance)
            floor = max(
                float(np.max(eigenvalues)) * 1.0e-13,
                float(np.min(np.diag(measurement_covariance))) * 1.0e-6,
                np.finfo(float).tiny,
            )
            predictive_covariance = (
                eigenvectors * np.maximum(eigenvalues, floor)
            ) @ eigenvectors.T
            lattice = solve_integer_least_squares(
                phase_center,
                predictive_covariance,
                candidate_count=max(int(lattice_candidates_per_chart), 1),
                max_visited_nodes=max(int(maximum_lattice_nodes), 1),
            )
            for candidate in lattice.candidates:
                raw_candidate_count += 1
                path = tuple(int(value) for value in candidate.integer_vector)
                component_paths.add(path)
                innovation = np.asarray(path, dtype=float) - phase_center
                conditioned_start = value_center + (
                    scale_squared
                    * cross_covariance
                    @ np.linalg.solve(predictive_covariance, innovation)
                )
                conditioned_start = np.clip(
                    conditioned_start,
                    normalized_lower,
                    normalized_upper,
                )
                try:
                    start_profile = evaluator.evaluate_values(
                        _physical_values(coordinates, conditioned_start)
                    )
                    start_objective = float(
                        start_profile.chi2
                        + exact_relative_chi2(start_profile, path)
                        + active_prior
                        * float(conditioned_start @ conditioned_start)
                    )
                except Exception:
                    start_objective = math.inf
                best_start_objective = min(
                    best_start_objective,
                    start_objective,
                )
                local_distances = np.linalg.norm(
                    local_values - conditioned_start,
                    axis=1,
                )
                sample_index = int(neighbors[int(np.argmin(local_distances))])
                proposal = LatticeProposal(
                    root_mode_id=f"nonlinear-shape-sample-{sample_index:07d}",
                    parent_mode_id=f"nonlinear-mixture-chart-{chart_order:04d}",
                    source="nonlinear-shape-posterior-mixture",
                    active_sessions=active_sessions,
                    integer_offsets=path,
                    initial_normalized_values=conditioned_start,
                    squared_mahalanobis=float(candidate.squared_mahalanobis),
                    log_predictive_density=float(
                        -0.5 * candidate.squared_mahalanobis
                    ),
                    kappa=float(covariance_scale),
                    initial_parameter_values=_physical_values(
                        coordinates,
                        conditioned_start,
                    ),
                )
                records.append((start_objective, chart_order, proposal))
                family_records.setdefault(chart_order, []).append(
                    (start_objective, proposal)
                )

        eigenvalues = np.linalg.eigvalsh(
            0.5 * (empirical_covariance + empirical_covariance.T)
        )
        eigenvalue_scale = max(
            float(np.max(np.abs(eigenvalues))),
            np.finfo(float).tiny,
        )
        component_diagnostics.append(
            {
                "chart_order": chart_order,
                "chart_sample_index": int(chart_index),
                "neighbor_count": int(neighbors.size),
                "raw_candidate_count": raw_candidate_count,
                "unique_path_count": len(component_paths),
                "best_start_objective_chi2": best_start_objective,
                "phase_covariance_effective_rank": int(
                    np.count_nonzero(eigenvalues > 1.0e-10 * eigenvalue_scale)
                ),
                "phase_covariance_top_eigenvalues": tuple(
                    float(value) for value in eigenvalues[-5:][::-1]
                ),
            }
        )

    selected_proposals: list[LatticeProposal] = []
    selected_states: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def add(proposal: LatticeProposal) -> bool:
        continuous_key = tuple(
            int(value)
            for value in np.rint(
                np.asarray(proposal.initial_normalized_values, dtype=float)
                * 1.0e6
            ).astype(np.int64)
        )
        key = proposal.integer_offsets, continuous_key
        if key in selected_states:
            return False
        selected_states.add(key)
        selected_proposals.append(proposal)
        return True

    family_winners = sorted(
        (
            min(family, key=lambda item: item[0])
            for family in family_records.values()
            if family
        ),
        key=lambda item: item[0],
    )
    for _, proposal in family_winners:
        add(proposal)
        if len(selected_proposals) >= max(int(maximum_proposals), 1):
            break
    if len(selected_proposals) < max(int(maximum_proposals), 1):
        for _, _, proposal in sorted(records, key=lambda item: item[0]):
            add(proposal)
            if len(selected_proposals) >= max(int(maximum_proposals), 1):
                break

    return NonlinearPhaseMixtureResult(
        evaluated_sample_count=values.shape[0],
        chart_sample_indices=tuple(int(value) for value in chart_indices),
        neighbor_count=neighbor_count,
        proposals=tuple(selected_proposals),
        trend_parameter_indices=trend_indices,
        manifold_parameter_indices=manifold_indices,
        component_diagnostics=tuple(component_diagnostics),
    )


def recover_shape_ensemble_exact_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    candidates: Sequence[ShapeIntegerBasinCandidate],
    prior_precision: float = 1.0,
    maximum_proposals: int = 64,
    maximum_modes: int = 32,
    maximum_modes_per_path: int = 2,
    exact_refit_max_nfev: int = 320,
    exact_phase_tempering_ladder: tuple[float, ...] = (
        0.001,
        0.01,
        0.10,
        0.30,
        1.0,
    ),
) -> tuple[GlobalLatticeMode, ...]:
    if not candidates:
        return ()
    session_count = len(candidates[0].integer_offsets) + 1
    active_sessions = tuple(range(1, session_count))
    config = GlobalLatticeTrackerConfig(
        prior_precision=max(float(prior_precision), 0.0),
        fixed_path_max_nfev=max(int(exact_refit_max_nfev), 1),
        phase_tempering_ladder=tuple(exact_phase_tempering_ladder),
    )
    modes: list[GlobalLatticeMode] = []
    for index, candidate in enumerate(
        sorted(candidates, key=lambda item: item.objective_chi2)[
            : max(int(maximum_proposals), 1)
        ]
    ):
        proposal = LatticeProposal(
            root_mode_id="shape-manifold-ensemble",
            parent_mode_id=candidate.candidate_id,
            source="shape-manifold-conditional-path",
            active_sessions=active_sessions,
            integer_offsets=candidate.integer_offsets,
            initial_normalized_values=candidate.normalized_values.copy(),
            squared_mahalanobis=float(candidate.phase_chi2),
            log_predictive_density=-0.5 * float(candidate.objective_chi2),
            kappa=1.0,
            initial_parameter_values=candidate.parameter_values.copy(),
        )
        try:
            modes.append(
                refit_fixed_path_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    anchor_session=0,
                    proposal=proposal,
                    config=config,
                    mode_id=f"shape_ensemble_exact_{index:05d}",
                )
            )
        except Exception:
            continue
    return select_diverse_lattice_modes(
        modes,
        maximum_modes=max(int(maximum_modes), 1),
        maximum_modes_per_path=max(int(maximum_modes_per_path), 1),
        maximum_modes_per_root=max(int(maximum_modes), 1),
        minimum_separation=0.005,
    )


__all__ = [
    "NonlinearPhaseMixtureResult",
    "ShapeIntegerBasinCandidate",
    "ShapeManifoldJointProposalResult",
    "ShapeManifoldEnsembleConfig",
    "ShapeManifoldEnsembleProgress",
    "ShapeManifoldEnsembleResult",
    "initialize_shape_manifold_walkers",
    "generate_nonlinear_phase_mixture_proposals",
    "generate_shape_manifold_joint_lattice_proposals",
    "recover_shape_ensemble_exact_modes",
    "run_shape_manifold_ensemble",
    "score_shape_samples_on_integer_lattice",
    "select_diverse_shape_manifold_charts",
    "shape_manifold_candidates_to_joint_modes",
    "shape_log_posterior",
]
