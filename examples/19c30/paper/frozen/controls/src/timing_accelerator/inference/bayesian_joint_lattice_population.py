from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtri
from scipy.stats import qmc

from timing_accelerator.inference.bayesian_global_lattice_mode_tracker import (
    GlobalLatticeMode,
    GlobalLatticeTrackerConfig,
    InvariantMode,
    LatticeProposal,
    ProfileLinearization,
    _conditional_gaussian,
    _laplace_log_mass,
    _normalize_mode_masses,
    _normalized_bounds,
    _phase_geometry,
    _physical_values,
    _profile_shape_residuals,
    _stable_covariance,
    _stable_inverse,
    exact_fixed_path_chi2,
    finite_difference_profile,
    fixed_path_residual_vector,
    propose_complete_integer_paths,
    refit_fixed_path_proposal,
    select_diverse_lattice_modes,
)
from timing_accelerator.inference.integer_least_squares import (
    build_marginalized_integer_covariance,
    solve_integer_least_squares,
)


@dataclass(frozen=True)
class JointLatticePopulationConfig:
    """Controls for all-session joint continuous/integer mode exploration."""

    generation_count: int = 4
    parents_per_generation: int = 32
    finite_difference_step: float = 2.0e-3
    prior_precision: float = 1.0
    covariance_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    lattice_candidates_per_chart: int = 4
    posterior_draws_per_chart: int = 0
    posterior_draw_seed: int = 20260725
    nonlinear_posterior_transport: bool = False
    require_feasible_conditioned_start: bool = False
    maximum_generation_proposals: int = 128
    maximum_lattice_nodes: int = 2_000_000
    exact_refit_max_nfev: int = 320
    exact_phase_tempering_ladder: tuple[float, ...] = (
        0.001,
        0.01,
        0.10,
        0.30,
        1.0,
    )
    maximum_modes: int = 32
    maximum_modes_per_path: int = 2
    maximum_modes_per_root: int = 32
    minimum_mode_separation: float = 0.005


@dataclass(frozen=True)
class JointLatticePathGeometry:
    """Shape-marginalized Gaussian chart for one complete integer path."""

    anchor_session: int
    active_sessions: tuple[int, ...]
    conditional_mean: np.ndarray
    conditional_covariance: np.ndarray
    integer_center: np.ndarray
    integer_covariance: np.ndarray
    phase_temperature: float


@dataclass(frozen=True)
class _JointLatticeChart:
    root_mode_id: str
    parent_mode_id: str | None
    source_prefix: str
    log_mass: float
    normalized_values: np.ndarray
    posterior_mean: np.ndarray | None = None
    posterior_covariance: np.ndarray | None = None


@dataclass(frozen=True)
class JointLatticeGenerationSummary:
    generation_index: int
    parent_count: int
    proposal_count: int
    unique_path_count: int
    refitted_count: int
    selected_count: int
    selected_path_count: int
    best_shape_chi2: float
    best_phase_chi2: float
    best_objective_chi2: float
    improved: bool


@dataclass(frozen=True)
class JointLatticePopulationResult:
    final_modes: tuple[GlobalLatticeMode, ...]
    generation_history: tuple[JointLatticeGenerationSummary, ...]
    source_session_count: int
    completed_generations: int


@dataclass(frozen=True)
class TemperedJointLatticeConfig:
    """Controls for joint path updates while exact phase information enters."""

    phase_temperature_ladder: tuple[float, ...] = (
        0.001,
        0.003,
        0.01,
        0.03,
        0.10,
        0.30,
        1.0,
    )
    mutations_per_temperature: int = 1
    parents_per_mutation: int = 24
    finite_difference_step: float = 2.0e-3
    prior_precision: float = 1.0
    covariance_scales: tuple[float, ...] = (1.0, 0.5, 2.0, 0.25)
    lattice_candidates_per_chart: int = 4
    maximum_mutation_proposals: int = 64
    maximum_lattice_nodes: int = 2_000_000
    refit_max_nfev: int = 120
    maximum_modes: int = 24
    maximum_modes_per_path: int = 2
    maximum_modes_per_root: int = 24
    minimum_mode_separation: float = 0.005


@dataclass(frozen=True)
class TemperedJointLatticeStageSummary:
    temperature_index: int
    mutation_index: int
    phase_temperature: float
    parent_count: int
    control_count: int
    proposal_count: int
    unique_path_count: int
    refitted_count: int
    selected_count: int
    selected_path_count: int
    best_shape_chi2: float
    best_phase_chi2: float
    best_tempered_objective_chi2: float


@dataclass(frozen=True)
class TemperedJointLatticeResult:
    final_modes: tuple[GlobalLatticeMode, ...]
    stage_history: tuple[TemperedJointLatticeStageSummary, ...]
    source_session_count: int
    reached_exact_temperature: bool


@dataclass(frozen=True)
class ReplicaJointLatticeConfig:
    """Controls for parallel-tempered complete-path mode populations."""

    phase_temperatures: tuple[float, ...] = (0.003, 0.03, 0.30, 1.0)
    round_count: int = 3
    parents_per_replica: int = 16
    finite_difference_step: float = 2.0e-3
    prior_precision: float = 1.0
    covariance_scales: tuple[float, ...] = (1.0, 0.5, 2.0, 0.25)
    lattice_candidates_per_chart: int = 6
    maximum_proposals_per_replica: int = 64
    maximum_lattice_nodes: int = 2_000_000
    refit_max_nfev: int = 120
    modes_per_replica: int = 16
    maximum_modes_per_path: int = 2
    maximum_modes_per_root: int = 16
    minimum_mode_separation: float = 0.005
    exchange_modes_per_pair: int = 4


@dataclass(frozen=True)
class ReplicaJointLatticeRoundSummary:
    round_index: int
    phase_temperature: float
    parent_count: int
    proposal_count: int
    unique_proposed_paths: int
    refitted_count: int
    exchange_refitted_count: int
    selected_count: int
    selected_path_count: int
    best_shape_chi2: float
    best_phase_chi2: float
    best_tempered_objective_chi2: float
    best_exact_objective_chi2: float


@dataclass(frozen=True)
class ReplicaJointLatticeResult:
    exact_modes: tuple[GlobalLatticeMode, ...]
    replica_modes: tuple[tuple[GlobalLatticeMode, ...], ...]
    round_history: tuple[ReplicaJointLatticeRoundSummary, ...]
    phase_temperatures: tuple[float, ...]
    source_session_count: int


def joint_lattice_path_geometry(
    linearization: ProfileLinearization,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    prior_precision: float,
    covariance_scale: float = 1.0,
    phase_temperature: float = 1.0,
    posterior_mean: Sequence[float] | None = None,
    posterior_covariance: Sequence[Sequence[float]] | None = None,
) -> JointLatticePathGeometry:
    """Marginalize continuous parameters before proposing a complete path.

    The shape likelihood and declared parameter prior define a local Gaussian
    posterior for the continuous coordinates. Projecting that posterior through
    the complete relative-phase Jacobian gives the Schur-complement covariance
    on the integer lattice. No existing integer coordinate is conditioned on.
    """

    sessions = tuple(int(value) for value in active_sessions)
    scale = float(covariance_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The joint-lattice covariance scale must be positive.")
    temperature = float(phase_temperature)
    if not np.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError("The joint-lattice phase temperature must be in (0, 1].")
    if (posterior_mean is None) != (posterior_covariance is None):
        raise ValueError(
            "Posterior mean and covariance must be supplied together."
        )
    if posterior_mean is None:
        mean, covariance = _conditional_gaussian(
            linearization,
            anchor_session=int(anchor_session),
            active_sessions=(),
            integer_offsets=(),
            prior_precision=max(float(prior_precision), 0.0),
        )
    else:
        mean = np.asarray(posterior_mean, dtype=float)
        covariance = np.asarray(posterior_covariance, dtype=float)
        dimension = np.asarray(
            linearization.normalized_values,
            dtype=float,
        ).size
        if mean.shape != (dimension,) or not np.all(np.isfinite(mean)):
            raise ValueError("The posterior mean has the wrong shape.")
        if covariance.shape != (dimension, dimension) or not np.all(
            np.isfinite(covariance)
        ):
            raise ValueError("The posterior covariance has the wrong shape.")
        covariance = _stable_covariance(covariance)
    relative, jacobian, noise = _phase_geometry(
        linearization,
        anchor_session=int(anchor_session),
        active_sessions=sessions,
    )
    displacement = mean - np.asarray(linearization.normalized_values, dtype=float)
    center = relative + jacobian @ displacement
    effective_covariance = (scale * scale) * np.asarray(covariance, dtype=float)
    integer_covariance = (
        noise / temperature
        + jacobian @ effective_covariance @ jacobian.T
    )
    integer_covariance = 0.5 * (
        integer_covariance + integer_covariance.T
    )
    return JointLatticePathGeometry(
        anchor_session=int(anchor_session),
        active_sessions=sessions,
        conditional_mean=np.asarray(mean, dtype=float),
        conditional_covariance=effective_covariance,
        integer_center=np.asarray(center, dtype=float),
        integer_covariance=np.asarray(integer_covariance, dtype=float),
        phase_temperature=temperature,
    )


def _conditioned_joint_state(
    linearization: ProfileLinearization,
    geometry: JointLatticePathGeometry,
    integer_offsets: Sequence[int],
) -> np.ndarray:
    _, jacobian, noise = _phase_geometry(
        linearization,
        anchor_session=geometry.anchor_session,
        active_sessions=geometry.active_sessions,
    )
    innovation = (
        np.asarray(integer_offsets, dtype=float) - geometry.integer_center
    )
    effective_noise = noise / float(geometry.phase_temperature)
    noise_precision = _stable_inverse(effective_noise)
    conditional_precision = _stable_inverse(
        geometry.conditional_covariance
    )
    posterior_covariance = _stable_inverse(
        conditional_precision
        + jacobian.T @ noise_precision @ jacobian
    )
    conditioned = geometry.conditional_mean + (
        posterior_covariance
        @ jacobian.T
        @ noise_precision
        @ innovation
    )
    return np.asarray(conditioned, dtype=float)


def _conditioned_joint_start(
    linearization: ProfileLinearization,
    geometry: JointLatticePathGeometry,
    integer_offsets: Sequence[int],
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    return np.clip(
        _conditioned_joint_state(
            linearization,
            geometry,
            integer_offsets,
        ),
        lower,
        upper,
    )


def _decimal_joint_lattice_covariance(
    linearization: ProfileLinearization,
    *,
    anchor_session: int,
    active_sessions: Sequence[int],
    prior_precision: float,
    covariance_scale: float,
    phase_temperature: float,
):
    _, phase_jacobian, noise = _phase_geometry(
        linearization,
        anchor_session=anchor_session,
        active_sessions=active_sessions,
    )
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    weight = max(float(prior_precision), 0.0)
    if weight > 0.0:
        shape_jacobian = np.vstack(
            (
                shape_jacobian,
                math.sqrt(weight) * np.eye(shape_jacobian.shape[1]),
            )
        )
    return build_marginalized_integer_covariance(
        shape_jacobian,
        float(covariance_scale) * phase_jacobian,
        noise / float(phase_temperature),
    )


def _joint_lattice_precision(
    linearization: ProfileLinearization,
    geometry: JointLatticePathGeometry,
) -> np.ndarray:
    _, jacobian, noise = _phase_geometry(
        linearization,
        anchor_session=geometry.anchor_session,
        active_sessions=geometry.active_sessions,
    )
    noise_precision = _stable_inverse(
        noise / float(geometry.phase_temperature)
    )
    conditional_precision = _stable_inverse(
        geometry.conditional_covariance
    )
    posterior_covariance = _stable_inverse(
        conditional_precision
        + jacobian.T @ noise_precision @ jacobian
    )
    precision = noise_precision - (
        noise_precision
        @ jacobian
        @ posterior_covariance
        @ jacobian.T
        @ noise_precision
    )
    return 0.5 * (precision + precision.T)


def _ranked_parent_modes(
    modes: Sequence[GlobalLatticeMode],
    maximum_count: int,
) -> tuple[GlobalLatticeMode, ...]:
    """Allocate proposal charts across complementary evidence channels.

    A low-mass joint mode may still contain a strong wrap-invariant continuous
    solution paired with the wrong integer path.  Such a chart is valuable for
    proposing a replacement path even though it must not be accepted as a
    solution.  Interleaving rank orders keeps joint-posterior, shape, and phase
    evidence separate until a fresh all-session lattice has been evaluated.
    """

    if not modes:
        return ()
    target = max(int(maximum_count), 1)
    rankings = (
        sorted(modes, key=lambda mode: mode.objective_chi2),
        sorted(
            modes,
            key=lambda mode: (mode.shape_chi2 + mode.prior_chi2, mode.phase_chi2),
        ),
        sorted(
            modes,
            key=lambda mode: (mode.phase_chi2 + mode.prior_chi2, mode.shape_chi2),
        ),
        sorted(modes, key=lambda mode: (-mode.log_mass, mode.objective_chi2)),
    )
    selected: list[GlobalLatticeMode] = []
    seen: set[str] = set()
    for rank in range(len(modes)):
        for ranking in rankings:
            candidate = ranking[rank]
            if candidate.mode_id in seen:
                continue
            seen.add(candidate.mode_id)
            selected.append(candidate)
            if len(selected) >= target:
                return tuple(selected)
    return tuple(selected)


def _select_joint_lattice_modes(
    modes: Sequence[GlobalLatticeMode],
    *,
    maximum_modes: int,
    maximum_modes_per_path: int,
    maximum_modes_per_root: int,
    minimum_separation: float,
) -> tuple[GlobalLatticeMode, ...]:
    """Retain proposal evidence diversity without weakening final scoring."""

    target = max(int(maximum_modes), 1)
    path_cap = max(int(maximum_modes_per_path), 1)
    root_cap = max(int(maximum_modes_per_root), 1)
    separation = max(float(minimum_separation), 0.0)
    ranked = _ranked_parent_modes(modes, len(modes))
    selected: list[GlobalLatticeMode] = []
    path_counts: dict[tuple[int, ...], int] = {}
    root_counts: dict[str, int] = {}

    def add(candidate: GlobalLatticeMode) -> bool:
        path = candidate.integer_offsets
        if path_counts.get(path, 0) >= path_cap:
            return False
        if root_counts.get(candidate.root_mode_id, 0) >= root_cap:
            return False
        if any(
            path == existing.integer_offsets
            and np.linalg.norm(
                candidate.normalized_values - existing.normalized_values
            )
            < separation
            for existing in selected
        ):
            return False
        selected.append(candidate)
        path_counts[path] = path_counts.get(path, 0) + 1
        root_counts[candidate.root_mode_id] = (
            root_counts.get(candidate.root_mode_id, 0) + 1
        )
        return True

    best_by_root: dict[str, GlobalLatticeMode] = {}
    for candidate in ranked:
        best_by_root.setdefault(candidate.root_mode_id, candidate)
    for candidate in best_by_root.values():
        add(candidate)
        if len(selected) >= target:
            return _normalize_mode_masses(selected)
    for candidate in ranked:
        add(candidate)
        if len(selected) >= target:
            break
    return _normalize_mode_masses(selected)


def _ranked_invariant_modes(
    modes: Sequence[InvariantMode],
    maximum_count: int,
) -> tuple[InvariantMode, ...]:
    if not modes:
        return ()
    target = max(int(maximum_count), 1)
    best_shape = min(modes, key=lambda mode: mode.shape_chi2)
    ranked = sorted(modes, key=lambda mode: (-mode.log_mass, mode.shape_chi2))
    selected = [best_shape]
    selected.extend(mode for mode in ranked if mode.mode_id != best_shape.mode_id)
    return tuple(selected[:target])


def _generate_joint_lattice_proposals_from_charts(
    *,
    evaluator: Any,
    coordinates: Any,
    charts: Sequence[_JointLatticeChart],
    anchor_session: int,
    active_sessions: Sequence[int],
    config: JointLatticePopulationConfig,
    phase_temperature: float,
) -> tuple[LatticeProposal, ...]:
    if not charts:
        return ()
    sessions = tuple(int(value) for value in active_sessions)
    lower, upper = _normalized_bounds(coordinates)
    scales = tuple(
        dict.fromkeys(
            float(value)
            for value in config.covariance_scales
            if np.isfinite(float(value)) and float(value) > 0.0
        )
    )
    if not scales:
        raise ValueError("At least one positive joint-lattice scale is required.")

    chart_buckets: list[list[LatticeProposal]] = []
    for chart_index, chart in enumerate(charts):
        try:
            linearization = finite_difference_profile(
                evaluator=evaluator,
                coordinates=coordinates,
                normalized_values=chart.normalized_values,
                step=max(float(config.finite_difference_step), 1.0e-7),
            )
        except Exception:
            continue
        scale_buckets: list[list[LatticeProposal]] = []
        for scale_index, scale in enumerate(scales):
            try:
                geometry = joint_lattice_path_geometry(
                    linearization,
                    anchor_session=anchor_session,
                    active_sessions=sessions,
                    prior_precision=config.prior_precision,
                    covariance_scale=scale,
                    phase_temperature=phase_temperature,
                    posterior_mean=chart.posterior_mean,
                    posterior_covariance=chart.posterior_covariance,
                )
                sign, logdet = np.linalg.slogdet(geometry.integer_covariance)
                log_normalization = (
                    0.5 * len(sessions) * math.log(2.0 * math.pi)
                    + 0.5 * float(logdet)
                    if sign > 0 and np.isfinite(logdet)
                    else math.inf
                )
                proposals = []
                try:
                    if chart.posterior_covariance is None:
                        lattice_covariance = _decimal_joint_lattice_covariance(
                            linearization,
                            anchor_session=anchor_session,
                            active_sessions=sessions,
                            prior_precision=config.prior_precision,
                            covariance_scale=scale,
                            phase_temperature=phase_temperature,
                        )
                    else:
                        lattice_covariance = geometry.integer_covariance
                    lattice = solve_integer_least_squares(
                        geometry.integer_center,
                        lattice_covariance,
                        candidate_count=max(
                            int(config.lattice_candidates_per_chart), 1
                        ),
                        max_visited_nodes=max(
                            int(config.maximum_lattice_nodes), 1
                        ),
                    )
                    for candidate in lattice.candidates:
                        path = tuple(
                            int(value) for value in candidate.integer_vector
                        )
                        conditioned = _conditioned_joint_state(
                            linearization,
                            geometry,
                            path,
                        )
                        if bool(config.require_feasible_conditioned_start) and (
                            np.any(conditioned < lower)
                            or np.any(conditioned > upper)
                        ):
                            continue
                        initial = np.clip(conditioned, lower, upper)
                        proposals.append(
                            LatticeProposal(
                                root_mode_id=chart.root_mode_id,
                                parent_mode_id=chart.parent_mode_id,
                                source=(
                                    f"{chart.source_prefix}-ils-scale-"
                                    f"{scale:.6g}"
                                ),
                                active_sessions=sessions,
                                integer_offsets=path,
                                initial_normalized_values=initial,
                                squared_mahalanobis=float(
                                    candidate.squared_mahalanobis
                                ),
                                log_predictive_density=float(
                                    chart.log_mass
                                    - 0.5 * candidate.squared_mahalanobis
                                    - log_normalization
                                ),
                                kappa=scale,
                            )
                        )
                except Exception:
                    pass

                draw_count = max(int(config.posterior_draws_per_chart), 0)
                if draw_count > 0:
                    dimension = geometry.conditional_mean.size
                    exponent = int(math.ceil(math.log2(draw_count)))
                    unit = qmc.Sobol(
                        d=dimension,
                        scramble=True,
                        seed=(
                            max(int(config.posterior_draw_seed), 0)
                            + 104729 * chart_index
                            + 1009 * scale_index
                        ),
                    ).random_base2(exponent)[:draw_count]
                    normal = ndtri(np.clip(unit, 1.0e-12, 1.0 - 1.0e-12))
                    root = np.linalg.cholesky(
                        _stable_covariance(geometry.conditional_covariance)
                    )
                    draws = geometry.conditional_mean + normal @ root.T
                    draws = np.clip(draws, lower, upper)
                    relative, phase_jacobian, _ = _phase_geometry(
                        linearization,
                        anchor_session=anchor_session,
                        active_sessions=sessions,
                    )
                    predicted = relative + (
                        draws - linearization.normalized_values
                    ) @ phase_jacobian.T
                    precision = _joint_lattice_precision(
                        linearization,
                        geometry,
                    )
                    for draw, linear_row in zip(draws, predicted, strict=True):
                        row = linear_row
                        initial_from_draw = False
                        if bool(config.nonlinear_posterior_transport):
                            try:
                                draw_profile = evaluator.evaluate_values(
                                    _physical_values(coordinates, draw)
                                )
                                offsets = np.asarray(
                                    draw_profile.session_offsets,
                                    dtype=float,
                                )
                                row = offsets[list(sessions)] - offsets[
                                    int(anchor_session)
                                ]
                                initial_from_draw = True
                            except Exception:
                                row = linear_row
                        integer_vector = np.rint(row).astype(np.int64)
                        innovation = integer_vector.astype(float) - (
                            geometry.integer_center
                        )
                        squared_mahalanobis = float(
                            innovation @ precision @ innovation
                        )
                        path = tuple(int(value) for value in integer_vector)
                        initial = (
                            np.asarray(draw, dtype=float)
                            if initial_from_draw
                            else _conditioned_joint_start(
                                linearization,
                                geometry,
                                path,
                                lower=lower,
                                upper=upper,
                            )
                        )
                        proposals.append(
                            LatticeProposal(
                                root_mode_id=chart.root_mode_id,
                                parent_mode_id=chart.parent_mode_id,
                                source=(
                                    f"{chart.source_prefix}-"
                                    f"{'nonlinear-qmc' if initial_from_draw else 'qmc'}-scale-"
                                    f"{scale:.6g}"
                                ),
                                active_sessions=sessions,
                                integer_offsets=path,
                                initial_normalized_values=initial,
                                squared_mahalanobis=squared_mahalanobis,
                                log_predictive_density=float(
                                    chart.log_mass
                                    - 0.5 * squared_mahalanobis
                                    - log_normalization
                                ),
                                kappa=scale,
                            )
                        )
                scale_buckets.append(
                    sorted(
                        proposals,
                        key=lambda proposal: (
                            proposal.squared_mahalanobis,
                            proposal.integer_offsets,
                        ),
                    )
                )
            except Exception:
                continue

        chart_bucket: list[LatticeProposal] = []
        maximum_scale_bucket = max(
            (len(bucket) for bucket in scale_buckets),
            default=0,
        )
        for rank in range(maximum_scale_bucket):
            for bucket in scale_buckets:
                if rank < len(bucket):
                    chart_bucket.append(bucket[rank])
        if chart_bucket:
            chart_buckets.append(chart_bucket)

    target = max(int(config.maximum_generation_proposals), 1)
    ordered: list[LatticeProposal] = []
    maximum_bucket = max((len(bucket) for bucket in chart_buckets), default=0)
    for rank in range(maximum_bucket):
        for bucket in chart_buckets:
            if rank < len(bucket):
                ordered.append(bucket[rank])

    selected: list[LatticeProposal] = []
    seen_states: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    seen_paths: set[tuple[int, ...]] = set()

    def add(proposal: LatticeProposal, *, require_new_path: bool) -> bool:
        if require_new_path and proposal.integer_offsets in seen_paths:
            return False
        state_key = tuple(
            int(value)
            for value in np.rint(
                np.asarray(proposal.initial_normalized_values) * 1000.0
            ).astype(np.int64)
        )
        key = proposal.integer_offsets, state_key
        if key in seen_states:
            return False
        seen_states.add(key)
        seen_paths.add(proposal.integer_offsets)
        selected.append(proposal)
        return True

    for proposal in ordered:
        add(proposal, require_new_path=True)
        if len(selected) >= target:
            return tuple(selected)
    for proposal in ordered:
        add(proposal, require_new_path=False)
        if len(selected) >= target:
            return tuple(selected)
    return tuple(selected)


def generate_joint_lattice_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    parent_modes: Sequence[GlobalLatticeMode],
    config: JointLatticePopulationConfig | None = None,
    phase_temperature: float = 1.0,
) -> tuple[LatticeProposal, ...]:
    """Propose complete paths from every retained nonlinear posterior chart."""

    active = config or JointLatticePopulationConfig()
    parents = _ranked_parent_modes(parent_modes, active.parents_per_generation)
    if not parents:
        return ()
    anchor_session = int(parents[0].anchor_session)
    profile = evaluator.evaluate_values(parents[0].parameter_values)
    sessions = tuple(
        session for session in range(int(profile.session_count))
        if session != anchor_session
    )
    if any(
        mode.anchor_session != anchor_session
        or tuple(mode.active_sessions) != sessions
        for mode in parents
    ):
        raise ValueError(
            "Joint lattice parents must use one anchor and every source session."
        )
    charts = tuple(
        _JointLatticeChart(
            root_mode_id=parent.root_mode_id,
            parent_mode_id=parent.mode_id,
            source_prefix="joint-schur",
            log_mass=float(parent.log_mass),
            normalized_values=np.asarray(parent.normalized_values, dtype=float),
        )
        for parent in parents
    )
    return _generate_joint_lattice_proposals_from_charts(
        evaluator=evaluator,
        coordinates=coordinates,
        charts=charts,
        anchor_session=anchor_session,
        active_sessions=sessions,
        config=active,
        phase_temperature=phase_temperature,
    )


def generate_invariant_joint_lattice_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    anchor_session: int,
    config: JointLatticePopulationConfig | None = None,
    phase_temperature: float = 1.0,
) -> tuple[LatticeProposal, ...]:
    """Generate complete paths directly from wrap-invariant posterior modes.

    Every session enters one correlated integer lattice. No existing path,
    prefix, partition, or continuation stage is used as a condition. The full
    continuous posterior covariance of each invariant mode is transported into
    the lattice before paths are enumerated.
    """

    active = config or JointLatticePopulationConfig()
    roots = _ranked_invariant_modes(
        invariant_modes,
        active.parents_per_generation,
    )
    if not roots:
        return ()
    profile = evaluator.evaluate_values(roots[0].parameter_values)
    session_count = int(profile.session_count)
    anchor = int(anchor_session)
    if not 0 <= anchor < session_count:
        raise ValueError("The joint-lattice anchor session is out of range.")
    sessions = tuple(session for session in range(session_count) if session != anchor)
    charts = tuple(
        _JointLatticeChart(
            root_mode_id=root.mode_id,
            parent_mode_id=None,
            source_prefix="invariant-joint-schur",
            log_mass=float(root.log_mass),
            normalized_values=np.asarray(root.normalized_values, dtype=float),
        )
        for root in roots
    )
    return _generate_joint_lattice_proposals_from_charts(
        evaluator=evaluator,
        coordinates=coordinates,
        charts=charts,
        anchor_session=anchor,
        active_sessions=sessions,
        config=active,
        phase_temperature=phase_temperature,
    )


def run_joint_lattice_population(
    *,
    evaluator: Any,
    coordinates: Any,
    seed_modes: Sequence[GlobalLatticeMode],
    config: JointLatticePopulationConfig | None = None,
    start_generation: int = 0,
    checkpoint_callback: Callable[
        [JointLatticeGenerationSummary, tuple[GlobalLatticeMode, ...]], None
    ]
    | None = None,
) -> JointLatticePopulationResult:
    """Explore the exact all-session posterior over `(theta, integer path)`.

    Every generation releases the complete integer vector, marginalizes the
    local continuous chart to construct correlated lattice proposals, and then
    reevaluates each proposal with the exact nonlinear fixed-path likelihood.
    Parent paths are retained as controls, so a proposal generation is never an
    irreversible merge or prefix decision.
    """

    active = config or JointLatticePopulationConfig()
    if not seed_modes:
        raise ValueError("Joint lattice population requires exact seed modes.")
    current = _select_joint_lattice_modes(
        seed_modes,
        maximum_modes=max(int(active.maximum_modes), 1),
        maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
        minimum_separation=max(float(active.minimum_mode_separation), 0.0),
    )
    if not current:
        raise ValueError("Joint lattice population retained no exact seed mode.")
    anchor_session = int(current[0].anchor_session)
    source_session_count = len(current[0].active_sessions) + 1
    exact_config = GlobalLatticeTrackerConfig(
        finite_difference_step=max(float(active.finite_difference_step), 1.0e-7),
        prior_precision=max(float(active.prior_precision), 0.0),
        fixed_path_max_nfev=max(int(active.exact_refit_max_nfev), 1),
        phase_tempering_ladder=tuple(active.exact_phase_tempering_ladder),
        maximum_lattice_nodes=max(int(active.maximum_lattice_nodes), 1),
    )
    history: list[JointLatticeGenerationSummary] = []
    generation_total = max(int(active.generation_count), 0)
    for generation_offset in range(generation_total):
        generation_index = int(start_generation) + generation_offset
        previous_best = min(mode.objective_chi2 for mode in current)
        parents = _ranked_parent_modes(current, active.parents_per_generation)
        proposals = generate_joint_lattice_proposals(
            evaluator=evaluator,
            coordinates=coordinates,
            parent_modes=parents,
            config=active,
        )
        refitted: list[GlobalLatticeMode] = []
        for proposal_index, proposal in enumerate(proposals):
            try:
                refitted.append(
                    refit_fixed_path_proposal(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        anchor_session=anchor_session,
                        proposal=proposal,
                        config=exact_config,
                        mode_id=(
                            f"joint_g{generation_index:03d}_"
                            f"{proposal_index:05d}"
                        ),
                    )
                )
            except Exception:
                continue
        population = tuple(current) + tuple(refitted)
        current = _select_joint_lattice_modes(
            population,
            maximum_modes=max(int(active.maximum_modes), 1),
            maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
            maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
            minimum_separation=max(float(active.minimum_mode_separation), 0.0),
        )
        current_best = min(mode.objective_chi2 for mode in current)
        summary = JointLatticeGenerationSummary(
            generation_index=generation_index,
            parent_count=len(parents),
            proposal_count=len(proposals),
            unique_path_count=len(
                {proposal.integer_offsets for proposal in proposals}
            ),
            refitted_count=len(refitted),
            selected_count=len(current),
            selected_path_count=len({mode.integer_offsets for mode in current}),
            best_shape_chi2=min(mode.shape_chi2 for mode in current),
            best_phase_chi2=min(mode.phase_chi2 for mode in current),
            best_objective_chi2=current_best,
            improved=bool(current_best < previous_best - 1.0e-8),
        )
        history.append(summary)
        if checkpoint_callback is not None:
            checkpoint_callback(summary, current)
        if not proposals or not refitted:
            break
    return JointLatticePopulationResult(
        final_modes=_normalize_mode_masses(current),
        generation_history=tuple(history),
        source_session_count=source_session_count,
        completed_generations=len(history),
    )


def refit_tempered_fixed_path_proposal(
    *,
    evaluator: Any,
    coordinates: Any,
    anchor_session: int,
    proposal: LatticeProposal,
    phase_temperature: float,
    config: TemperedJointLatticeConfig | None = None,
    mode_id: str = "tempered-mode",
) -> GlobalLatticeMode:
    """Refit one complete path on a declared all-session tempered target."""

    active = config or TemperedJointLatticeConfig()
    temperature = float(phase_temperature)
    if not np.isfinite(temperature) or not 0.0 < temperature <= 1.0:
        raise ValueError("The phase temperature must be in (0, 1].")
    lower, upper = _normalized_bounds(coordinates)
    start = np.clip(
        np.asarray(proposal.initial_normalized_values, dtype=float),
        lower,
        upper,
    )
    start_profile = evaluator.evaluate_values(_physical_values(coordinates, start))
    shape_size = _profile_shape_residuals(start_profile).size
    phase_size = len(proposal.active_sessions)
    prior_precision = max(float(active.prior_precision), 0.0)
    failure_size = shape_size + phase_size + (
        start.size if prior_precision > 0.0 else 0
    )
    failure = np.full(failure_size, 1.0e12, dtype=float)

    def residual_vector(values: np.ndarray) -> np.ndarray:
        try:
            profile = evaluator.evaluate_values(_physical_values(coordinates, values))
            result = fixed_path_residual_vector(
                profile,
                anchor_session=anchor_session,
                active_sessions=proposal.active_sessions,
                integer_offsets=proposal.integer_offsets,
                normalized_values=values,
                prior_precision=prior_precision,
            ).copy()
            result[shape_size : shape_size + phase_size] *= math.sqrt(temperature)
            return (
                result
                if result.shape == failure.shape and np.all(np.isfinite(result))
                else failure
            )
        except Exception:
            return failure

    optimization = least_squares(
        residual_vector,
        start,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        diff_step=max(float(active.finite_difference_step), 1.0e-7),
        max_nfev=max(int(active.refit_max_nfev), 1),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    normalized = np.asarray(optimization.x, dtype=float)
    profile = evaluator.evaluate_values(_physical_values(coordinates, normalized))
    shape_chi2, phase_chi2, _ = exact_fixed_path_chi2(
        profile,
        anchor_session=anchor_session,
        active_sessions=proposal.active_sessions,
        integer_offsets=proposal.integer_offsets,
    )
    prior_chi2 = prior_precision * float(normalized @ normalized)
    objective = shape_chi2 + temperature * phase_chi2 + prior_chi2
    jacobian = np.asarray(optimization.jac, dtype=float)
    covariance = _stable_inverse(jacobian.T @ jacobian)
    return GlobalLatticeMode(
        mode_id=mode_id,
        root_mode_id=proposal.root_mode_id,
        parent_mode_id=proposal.parent_mode_id,
        proposal_source=proposal.source,
        anchor_session=int(anchor_session),
        active_sessions=proposal.active_sessions,
        integer_offsets=proposal.integer_offsets,
        parameter_values=_physical_values(coordinates, normalized),
        normalized_values=normalized,
        covariance=covariance,
        shape_chi2=shape_chi2,
        phase_chi2=phase_chi2,
        prior_chi2=prior_chi2,
        objective_chi2=objective,
        log_mass=_laplace_log_mass(objective, covariance),
        posterior_mass=0.0,
        squared_mahalanobis=float(proposal.squared_mahalanobis),
        nfev=int(optimization.nfev),
        success=bool(optimization.success),
    )


def _control_proposal(mode: GlobalLatticeMode) -> LatticeProposal:
    return LatticeProposal(
        root_mode_id=mode.root_mode_id,
        parent_mode_id=mode.mode_id,
        source="joint-tempered-control",
        active_sessions=mode.active_sessions,
        integer_offsets=mode.integer_offsets,
        initial_normalized_values=np.asarray(mode.normalized_values, dtype=float),
        squared_mahalanobis=0.0,
        log_predictive_density=float(mode.log_mass),
        kappa=1.0,
    )


def run_tempered_joint_lattice_population(
    *,
    evaluator: Any,
    coordinates: Any,
    seed_modes: Sequence[GlobalLatticeMode],
    config: TemperedJointLatticeConfig | None = None,
    checkpoint_callback: Callable[
        [TemperedJointLatticeStageSummary, tuple[GlobalLatticeMode, ...]], None
    ]
    | None = None,
) -> TemperedJointLatticeResult:
    """Alternate complete-path and continuous updates at every temperature."""

    active = config or TemperedJointLatticeConfig()
    if not seed_modes:
        raise ValueError("Tempered joint lattice inference requires exact seed modes.")
    ladder = tuple(
        sorted(
            {
                float(value)
                for value in active.phase_temperature_ladder
                if np.isfinite(float(value)) and 0.0 < float(value) <= 1.0
            }
            | {1.0}
        )
    )
    if not ladder:
        raise ValueError("At least one valid phase temperature is required.")
    current = select_diverse_lattice_modes(
        seed_modes,
        maximum_modes=max(int(active.maximum_modes), 1),
        maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
        minimum_separation=max(float(active.minimum_mode_separation), 0.0),
    )
    anchor_session = int(current[0].anchor_session)
    source_session_count = len(current[0].active_sessions) + 1
    proposal_config = JointLatticePopulationConfig(
        generation_count=1,
        parents_per_generation=max(int(active.parents_per_mutation), 1),
        finite_difference_step=max(float(active.finite_difference_step), 1.0e-7),
        prior_precision=max(float(active.prior_precision), 0.0),
        covariance_scales=tuple(active.covariance_scales),
        lattice_candidates_per_chart=max(
            int(active.lattice_candidates_per_chart), 1
        ),
        maximum_generation_proposals=max(
            int(active.maximum_mutation_proposals), 1
        ),
        maximum_lattice_nodes=max(int(active.maximum_lattice_nodes), 1),
        maximum_modes=max(int(active.maximum_modes), 1),
        maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
        minimum_mode_separation=max(float(active.minimum_mode_separation), 0.0),
    )
    history: list[TemperedJointLatticeStageSummary] = []
    for temperature_index, temperature in enumerate(ladder):
        for mutation_index in range(max(int(active.mutations_per_temperature), 1)):
            parents = _ranked_parent_modes(current, active.parents_per_mutation)
            controls: list[GlobalLatticeMode] = []
            for parent_index, parent in enumerate(parents):
                try:
                    controls.append(
                        refit_tempered_fixed_path_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            anchor_session=anchor_session,
                            proposal=_control_proposal(parent),
                            phase_temperature=temperature,
                            config=active,
                            mode_id=(
                                f"tempered_t{temperature_index:02d}_"
                                f"m{mutation_index:02d}_control_{parent_index:04d}"
                            ),
                        )
                    )
                except Exception:
                    continue
            if not controls:
                raise RuntimeError(
                    f"No control mode survived phase temperature {temperature}."
                )
            proposals = generate_joint_lattice_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                parent_modes=controls,
                config=proposal_config,
                phase_temperature=temperature,
            )
            refitted: list[GlobalLatticeMode] = []
            for proposal_index, proposal in enumerate(proposals):
                try:
                    refitted.append(
                        refit_tempered_fixed_path_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            anchor_session=anchor_session,
                            proposal=proposal,
                            phase_temperature=temperature,
                            config=active,
                            mode_id=(
                                f"tempered_t{temperature_index:02d}_"
                                f"m{mutation_index:02d}_{proposal_index:05d}"
                            ),
                        )
                    )
                except Exception:
                    continue
            current = select_diverse_lattice_modes(
                tuple(controls) + tuple(refitted),
                maximum_modes=max(int(active.maximum_modes), 1),
                maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
                maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
                minimum_separation=max(float(active.minimum_mode_separation), 0.0),
            )
            summary = TemperedJointLatticeStageSummary(
                temperature_index=temperature_index,
                mutation_index=mutation_index,
                phase_temperature=temperature,
                parent_count=len(parents),
                control_count=len(controls),
                proposal_count=len(proposals),
                unique_path_count=len(
                    {proposal.integer_offsets for proposal in proposals}
                ),
                refitted_count=len(refitted),
                selected_count=len(current),
                selected_path_count=len({mode.integer_offsets for mode in current}),
                best_shape_chi2=min(mode.shape_chi2 for mode in current),
                best_phase_chi2=min(mode.phase_chi2 for mode in current),
                best_tempered_objective_chi2=min(
                    mode.objective_chi2 for mode in current
                ),
            )
            history.append(summary)
            if checkpoint_callback is not None:
                checkpoint_callback(summary, current)
    return TemperedJointLatticeResult(
        final_modes=_normalize_mode_masses(current),
        stage_history=tuple(history),
        source_session_count=source_session_count,
        reached_exact_temperature=bool(ladder[-1] == 1.0),
    )


def _exact_objective(mode: GlobalLatticeMode) -> float:
    return float(mode.shape_chi2 + mode.phase_chi2 + mode.prior_chi2)


def _exchange_mode_candidates(
    modes: Sequence[GlobalLatticeMode],
    maximum_count: int,
) -> tuple[GlobalLatticeMode, ...]:
    target = max(int(maximum_count), 0)
    if target == 0:
        return ()
    rankings = (
        sorted(modes, key=lambda mode: mode.objective_chi2),
        sorted(modes, key=_exact_objective),
        sorted(modes, key=lambda mode: mode.shape_chi2),
    )
    selected: list[GlobalLatticeMode] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    maximum_rank = max((len(items) for items in rankings), default=0)
    for rank in range(maximum_rank):
        for items in rankings:
            if rank >= len(items):
                continue
            mode = items[rank]
            key = mode.mode_id, mode.integer_offsets
            if key in seen:
                continue
            seen.add(key)
            selected.append(mode)
            if len(selected) >= target:
                return tuple(selected)
    return tuple(selected)


def run_replica_joint_lattice_population(
    *,
    evaluator: Any,
    coordinates: Any,
    seed_modes: Sequence[GlobalLatticeMode],
    config: ReplicaJointLatticeConfig | None = None,
    checkpoint_callback: Callable[
        [ReplicaJointLatticeRoundSummary, tuple[GlobalLatticeMode, ...]], None
    ]
    | None = None,
) -> ReplicaJointLatticeResult:
    """Run bidirectionally coupled all-session phase-temperature replicas."""

    active = config or ReplicaJointLatticeConfig()
    if not seed_modes:
        raise ValueError("Replica joint lattice inference requires exact seed modes.")
    temperatures = tuple(
        sorted(
            {
                float(value)
                for value in active.phase_temperatures
                if np.isfinite(float(value)) and 0.0 < float(value) <= 1.0
            }
            | {1.0}
        )
    )
    anchor_session = int(seed_modes[0].anchor_session)
    source_session_count = len(seed_modes[0].active_sessions) + 1
    if any(
        mode.anchor_session != anchor_session
        or len(mode.active_sessions) + 1 != source_session_count
        for mode in seed_modes
    ):
        raise ValueError("Replica seeds must share one complete all-session path chart.")
    tempered_config = TemperedJointLatticeConfig(
        phase_temperature_ladder=temperatures,
        mutations_per_temperature=1,
        parents_per_mutation=max(int(active.parents_per_replica), 1),
        finite_difference_step=max(float(active.finite_difference_step), 1.0e-7),
        prior_precision=max(float(active.prior_precision), 0.0),
        covariance_scales=tuple(active.covariance_scales),
        lattice_candidates_per_chart=max(
            int(active.lattice_candidates_per_chart), 1
        ),
        maximum_mutation_proposals=max(
            int(active.maximum_proposals_per_replica), 1
        ),
        maximum_lattice_nodes=max(int(active.maximum_lattice_nodes), 1),
        refit_max_nfev=max(int(active.refit_max_nfev), 1),
        maximum_modes=max(int(active.modes_per_replica), 1),
        maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
        minimum_mode_separation=max(float(active.minimum_mode_separation), 0.0),
    )
    proposal_config = JointLatticePopulationConfig(
        generation_count=1,
        parents_per_generation=max(int(active.parents_per_replica), 1),
        finite_difference_step=max(float(active.finite_difference_step), 1.0e-7),
        prior_precision=max(float(active.prior_precision), 0.0),
        covariance_scales=tuple(active.covariance_scales),
        lattice_candidates_per_chart=max(
            int(active.lattice_candidates_per_chart), 1
        ),
        maximum_generation_proposals=max(
            int(active.maximum_proposals_per_replica), 1
        ),
        maximum_lattice_nodes=max(int(active.maximum_lattice_nodes), 1),
        maximum_modes=max(int(active.modes_per_replica), 1),
        maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
        minimum_mode_separation=max(float(active.minimum_mode_separation), 0.0),
    )

    seed_parents = _ranked_parent_modes(seed_modes, active.parents_per_replica)
    replicas: list[tuple[GlobalLatticeMode, ...]] = []
    for temperature_index, temperature in enumerate(temperatures):
        if temperature == 1.0:
            initialized = tuple(seed_parents)
        else:
            initialized_modes: list[GlobalLatticeMode] = []
            for parent_index, parent in enumerate(seed_parents):
                try:
                    initialized_modes.append(
                        refit_tempered_fixed_path_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            anchor_session=anchor_session,
                            proposal=_control_proposal(parent),
                            phase_temperature=temperature,
                            config=tempered_config,
                            mode_id=(
                                f"replica_init_t{temperature_index:02d}_"
                                f"{parent_index:04d}"
                            ),
                        )
                    )
                except Exception:
                    continue
            initialized = tuple(initialized_modes)
        if not initialized:
            raise RuntimeError(
                f"Replica initialization failed at phase temperature {temperature}."
            )
        replicas.append(
            select_diverse_lattice_modes(
                initialized,
                maximum_modes=max(int(active.modes_per_replica), 1),
                maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
                maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
                minimum_separation=max(float(active.minimum_mode_separation), 0.0),
            )
        )

    history: list[ReplicaJointLatticeRoundSummary] = []
    for round_index in range(max(int(active.round_count), 1)):
        mutation_records: list[tuple[int, int, int, int]] = []
        mutated_replicas: list[tuple[GlobalLatticeMode, ...]] = []
        for temperature_index, (temperature, current) in enumerate(
            zip(temperatures, replicas, strict=True)
        ):
            parents = _ranked_parent_modes(current, active.parents_per_replica)
            proposals = generate_joint_lattice_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                parent_modes=parents,
                config=proposal_config,
                phase_temperature=temperature,
            )
            refitted: list[GlobalLatticeMode] = []
            for proposal_index, proposal in enumerate(proposals):
                try:
                    refitted.append(
                        refit_tempered_fixed_path_proposal(
                            evaluator=evaluator,
                            coordinates=coordinates,
                            anchor_session=anchor_session,
                            proposal=proposal,
                            phase_temperature=temperature,
                            config=tempered_config,
                            mode_id=(
                                f"replica_r{round_index:02d}_"
                                f"t{temperature_index:02d}_{proposal_index:05d}"
                            ),
                        )
                    )
                except Exception:
                    continue
            mutated_replicas.append(
                select_diverse_lattice_modes(
                    tuple(current) + tuple(refitted),
                    maximum_modes=max(int(active.modes_per_replica), 1),
                    maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
                    maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
                    minimum_separation=max(float(active.minimum_mode_separation), 0.0),
                )
            )
            mutation_records.append(
                (
                    len(parents),
                    len(proposals),
                    len({proposal.integer_offsets for proposal in proposals}),
                    len(refitted),
                )
            )

        exchange_additions: list[list[GlobalLatticeMode]] = [
            [] for _ in temperatures
        ]
        for left_index in range(len(temperatures) - 1):
            right_index = left_index + 1
            for source_index, destination_index in (
                (left_index, right_index),
                (right_index, left_index),
            ):
                candidates = _exchange_mode_candidates(
                    mutated_replicas[source_index],
                    active.exchange_modes_per_pair,
                )
                for candidate_index, candidate in enumerate(candidates):
                    try:
                        exchange_additions[destination_index].append(
                            refit_tempered_fixed_path_proposal(
                                evaluator=evaluator,
                                coordinates=coordinates,
                                anchor_session=anchor_session,
                                proposal=_control_proposal(candidate),
                                phase_temperature=temperatures[destination_index],
                                config=tempered_config,
                                mode_id=(
                                    f"replica_exchange_r{round_index:02d}_"
                                    f"{source_index:02d}_{destination_index:02d}_"
                                    f"{candidate_index:04d}"
                                ),
                            )
                        )
                    except Exception:
                        continue

        replicas = []
        for temperature_index, (temperature, mutated, additions) in enumerate(
            zip(
                temperatures,
                mutated_replicas,
                exchange_additions,
                strict=True,
            )
        ):
            selected = select_diverse_lattice_modes(
                tuple(mutated) + tuple(additions),
                maximum_modes=max(int(active.modes_per_replica), 1),
                maximum_modes_per_path=max(int(active.maximum_modes_per_path), 1),
                maximum_modes_per_root=max(int(active.maximum_modes_per_root), 1),
                minimum_separation=max(float(active.minimum_mode_separation), 0.0),
            )
            replicas.append(selected)
            (
                parent_count,
                proposal_count,
                unique_proposed_paths,
                refitted_count,
            ) = mutation_records[
                temperature_index
            ]
            summary = ReplicaJointLatticeRoundSummary(
                round_index=round_index,
                phase_temperature=temperature,
                parent_count=parent_count,
                proposal_count=proposal_count,
                unique_proposed_paths=unique_proposed_paths,
                refitted_count=refitted_count,
                exchange_refitted_count=len(additions),
                selected_count=len(selected),
                selected_path_count=len({mode.integer_offsets for mode in selected}),
                best_shape_chi2=min(mode.shape_chi2 for mode in selected),
                best_phase_chi2=min(mode.phase_chi2 for mode in selected),
                best_tempered_objective_chi2=min(
                    mode.objective_chi2 for mode in selected
                ),
                best_exact_objective_chi2=min(
                    _exact_objective(mode) for mode in selected
                ),
            )
            history.append(summary)
            if checkpoint_callback is not None:
                checkpoint_callback(summary, selected)

    exact_index = temperatures.index(1.0)
    return ReplicaJointLatticeResult(
        exact_modes=_normalize_mode_masses(replicas[exact_index]),
        replica_modes=tuple(_normalize_mode_masses(modes) for modes in replicas),
        round_history=tuple(history),
        phase_temperatures=temperatures,
        source_session_count=source_session_count,
    )


__all__ = [
    "JointLatticeGenerationSummary",
    "JointLatticePathGeometry",
    "JointLatticePopulationConfig",
    "JointLatticePopulationResult",
    "TemperedJointLatticeConfig",
    "TemperedJointLatticeResult",
    "TemperedJointLatticeStageSummary",
    "ReplicaJointLatticeConfig",
    "ReplicaJointLatticeResult",
    "ReplicaJointLatticeRoundSummary",
    "generate_joint_lattice_proposals",
    "generate_invariant_joint_lattice_proposals",
    "joint_lattice_path_geometry",
    "refit_tempered_fixed_path_proposal",
    "run_joint_lattice_population",
    "run_replica_joint_lattice_population",
    "run_tempered_joint_lattice_population",
]
