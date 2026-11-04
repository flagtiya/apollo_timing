from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp

from timing_accelerator.inference.bayesian_full_sequence_circular_transport import (
    FullSequenceCircularMode,
    wrapped_gaussian_phase_deviance,
)
from timing_accelerator.inference.bayesian_global_lattice_mode_tracker import (
    GlobalLatticeMode,
    GlobalLatticeTrackerConfig,
    InvariantMode,
    LatticeProposal,
    exact_fixed_path_chi2,
    finite_difference_profile,
    refit_fixed_path_proposal,
    select_diverse_lattice_modes,
)
from timing_accelerator.inference.collapsed_integer_posterior import (
    conditional_integer_paths,
)


@dataclass(frozen=True)
class FullSequenceParticleConfig:
    """Controls for all-session wrapped-Gaussian population transport."""

    particle_count: int = 192
    initial_phase_sigma: float = 0.50
    final_phase_sigma: float = 0.02
    initial_proposal_scales: tuple[float, ...] = (
        0.15,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    )
    initial_root_modes: int = 0
    minimum_root_particles: int = 4
    minimum_root_fraction: float = 0.10
    preserve_root_lineages: bool = True
    prior_precision: float = 1.0
    ess_target_fraction: float = 0.72
    resample_threshold_fraction: float = 0.78
    maximum_temperature_steps: int = 32
    maximum_sigma_ratio: float = 0.70
    minimum_relative_sigma_step: float = 1.0e-3
    maximum_exact_temperature_refinements: int = 4
    move_steps_per_temperature: int = 2
    independent_proposal_probability: float = 0.15
    differential_evolution_probability: float = 0.55
    global_differential_evolution_probability: float = 0.20
    differential_evolution_scale: float = 1.0
    random_walk_scale: float = 0.70
    proposal_scale_multipliers: tuple[float, ...] = (0.25, 0.50, 1.0)
    proposal_jitter: float = 1.0e-5
    minimum_cluster_particles: int = 2
    maximum_clusters: int = 24
    cluster_parameter_separation: float = 0.18
    cluster_phase_separation: float = 0.08
    maximum_output_modes: int = 24
    path_parent_particles: int = 48
    minimum_path_parents_per_root: int = 4
    integer_candidates_per_particle: int = 8
    integer_recovery_phase_sigmas: tuple[float, ...] = (0.0, 0.02, 0.05)
    low_rank_path_directions: int = 6
    low_rank_path_starts: int = 32
    low_rank_path_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    low_rank_path_phase_sigmas: tuple[float, ...] = (0.50, 0.20, 0.08, 0.03)
    low_rank_path_max_nfev: int = 80
    maximum_path_proposals: int = 32
    minimum_path_proposals_per_root: int = 4
    minimum_path_proposals_per_family: int = 2
    maximum_exact_modes: int = 16
    maximum_exact_modes_per_path: int = 2
    maximum_exact_modes_per_root: int = 8
    exact_mode_separation: float = 0.01
    exact_refit_max_nfev: int = 240
    exact_phase_tempering_ladder: tuple[float, ...] = (
        0.003,
        0.01,
        0.03,
        0.10,
        0.30,
        1.0,
    )
    wrapped_gaussian_log_tolerance: float = -27.631021115928547
    wrapped_gaussian_max_grid_points: int = 16_384
    random_seed: int = 20260724


@dataclass(frozen=True)
class FullSequenceParticleSummary:
    particle_id: str
    root_mode_id: str
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    shape_chi2: float
    phase_deviance: float
    prior_chi2: float
    objective_chi2: float
    phase_coherence: float
    phase_signature: tuple[float, ...]
    log_weight: float
    accepted_moves: int


@dataclass(frozen=True)
class ParticleTemperatureSummary:
    temperature_index: int
    phase_sigma: float
    effective_sample_size: float
    cluster_count: int
    root_count: int
    minimum_root_particle_count: int
    resampled: bool
    accepted_moves: int
    evaluated_proposals: int
    best_shape_chi2: float
    best_phase_deviance: float
    best_objective_chi2: float
    maximum_phase_coherence: float


@dataclass(frozen=True)
class FullSequenceParticleResult:
    particles: tuple[FullSequenceParticleSummary, ...]
    circular_modes: tuple[FullSequenceCircularMode, ...]
    temperature_history: tuple[ParticleTemperatureSummary, ...]
    reached_final_temperature: bool
    source_session_count: int
    exact_modes: tuple[GlobalLatticeMode, ...] = tuple()
    path_proposal_count: int = 0


@dataclass(frozen=True)
class ParticleIntegerBasinDiagnostic:
    particle_id: str
    root_mode_id: str
    integer_offsets: tuple[int, ...]
    shape_chi2: float
    phase_chi2: float
    objective_chi2: float
    log_conditional_mass: float
    particle_log_weight: float
    phase_coherence: float


@dataclass
class _Particle:
    particle_id: str
    root_mode_id: str
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: Any
    shape_chi2: float
    phase_deviance: float
    prior_chi2: float
    phase_coherence: float
    phase_signature: tuple[float, ...]
    log_weight: float = 0.0
    accepted_moves: int = 0

    @property
    def objective_chi2(self) -> float:
        return float(self.shape_chi2 + self.phase_deviance + self.prior_chi2)


@dataclass(frozen=True)
class _GaussianProposalComponent:
    root_mode_id: str
    center: np.ndarray
    covariance_root: np.ndarray
    log_abs_root_determinant: float
    scale: float
    log_mixture_weight: float


def _normalized_bounds(coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("Normalized coordinate bounds must be aligned.")
    return lower, upper


def _physical_values(coordinates: Any, normalized_values: Sequence[float]) -> np.ndarray:
    search_values = coordinates.denormalize(
        np.asarray(normalized_values, dtype=float)
    )
    return np.asarray(coordinates.to_physical(search_values), dtype=float)


def _normalized_values(coordinates: Any, parameter_values: Sequence[float]) -> np.ndarray:
    search_values = coordinates.to_search(np.asarray(parameter_values, dtype=float))
    return np.asarray(coordinates.normalize(search_values), dtype=float)


def _covariance_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (
        np.asarray(covariance, dtype=float) + np.asarray(covariance, dtype=float).T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, scale * 1.0e-12)
    return eigenvectors @ np.diag(np.sqrt(clipped))


def _proposal_components(
    *,
    invariant_modes: Sequence[InvariantMode],
    coordinates: Any,
    config: FullSequenceParticleConfig,
) -> tuple[_GaussianProposalComponent, ...]:
    declared_roots = tuple(invariant_modes)
    root_limit = int(config.initial_root_modes)
    roots = (
        declared_roots[:root_limit]
        if root_limit > 0
        else declared_roots
    )
    if not roots:
        raise RuntimeError("Particle transport requires at least one declared root.")
    scales = tuple(
        float(value)
        for value in config.initial_proposal_scales
        if np.isfinite(float(value)) and float(value) > 0.0
    ) or (0.15, 1.0)
    components: list[_GaussianProposalComponent] = []
    log_mixture_weight = -math.log(len(roots)) - math.log(len(scales))
    for mode in roots:
        covariance_root = _covariance_root(np.asarray(mode.covariance, dtype=float))
        sign, log_abs_root_determinant = np.linalg.slogdet(covariance_root)
        if sign == 0.0 or not np.isfinite(log_abs_root_determinant):
            continue
        center = _normalized_values(coordinates, mode.parameter_values)
        for scale in scales:
            components.append(
                _GaussianProposalComponent(
                    root_mode_id=str(mode.mode_id),
                    center=center.copy(),
                    covariance_root=covariance_root.copy(),
                    log_abs_root_determinant=float(log_abs_root_determinant),
                    scale=scale,
                    log_mixture_weight=log_mixture_weight,
                )
            )
    if not components:
        raise RuntimeError("No valid Gaussian components for particle initialization.")
    return tuple(components)


def _sample_proposal_component(
    components: Sequence[_GaussianProposalComponent],
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    component = components[int(rng.integers(0, len(components)))]
    values = component.center + component.scale * (
        component.covariance_root @ rng.normal(size=component.center.size)
    )
    return values, component.root_mode_id


def _mixture_log_proposal_density(
    values: np.ndarray,
    components: Sequence[_GaussianProposalComponent],
) -> float:
    point = np.asarray(values, dtype=float)
    dimension = point.size
    log_terms: list[float] = []
    for component in components:
        whitened = np.linalg.solve(
            component.covariance_root,
            point - component.center,
        ) / component.scale
        log_terms.append(
            component.log_mixture_weight
            - 0.5 * dimension * math.log(2.0 * math.pi)
            - component.log_abs_root_determinant
            - dimension * math.log(component.scale)
            - 0.5 * float(whitened @ whitened)
        )
    return float(logsumexp(log_terms))


def _regularized_covariance(
    values: np.ndarray,
    *,
    fallback: np.ndarray,
    jitter: float,
) -> np.ndarray:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2:
        raise ValueError("Particle coordinates must be a matrix.")
    if points.shape[0] >= 2:
        covariance = np.atleast_2d(np.cov(points, rowvar=False, ddof=1))
    else:
        covariance = np.asarray(fallback, dtype=float).copy()
    if covariance.shape != fallback.shape or not np.all(np.isfinite(covariance)):
        covariance = np.asarray(fallback, dtype=float).copy()
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    reference = np.linalg.eigvalsh(0.5 * (fallback + fallback.T))
    floor = np.maximum(
        np.maximum(reference * 1.0e-3, max(float(jitter), 1.0e-8) ** 2),
        1.0e-14,
    )
    eigenvalues = np.maximum(eigenvalues, np.min(floor))
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _normalise_log_weights(particles: Sequence[_Particle]) -> None:
    values = np.asarray([particle.log_weight for particle in particles], dtype=float)
    normalizer = float(logsumexp(values))
    for particle in particles:
        particle.log_weight -= normalizer


def _effective_sample_size(log_weights: Sequence[float]) -> float:
    values = np.asarray(log_weights, dtype=float)
    normalized = values - logsumexp(values)
    weights = np.exp(normalized)
    return float(1.0 / np.sum(np.square(weights)))


def _wrapped_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (np.asarray(left) - np.asarray(right) + 0.5) % 1.0 - 0.5


def _cluster_particles(
    particles: Sequence[_Particle],
    *,
    config: FullSequenceParticleConfig,
) -> tuple[tuple[int, ...], ...]:
    order = sorted(
        range(len(particles)),
        key=lambda index: particles[index].log_weight,
        reverse=True,
    )
    clusters: list[list[int]] = []
    centers: list[int] = []
    maximum_clusters = max(int(config.maximum_clusters), 1)
    for index in order:
        candidate = particles[index]
        best_cluster = -1
        best_distance = math.inf
        for cluster_index, center_index in enumerate(centers):
            center = particles[center_index]
            if (
                config.preserve_root_lineages
                and candidate.root_mode_id != center.root_mode_id
            ):
                continue
            parameter_distance = float(
                np.linalg.norm(candidate.normalized_values - center.normalized_values)
            )
            phase_distance = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            _wrapped_difference(
                                np.asarray(candidate.phase_signature, dtype=float),
                                np.asarray(center.phase_signature, dtype=float),
                            )
                        )
                    )
                )
            )
            scaled_distance = max(
                parameter_distance
                / max(float(config.cluster_parameter_separation), 1.0e-12),
                phase_distance / max(float(config.cluster_phase_separation), 1.0e-12),
            )
            if scaled_distance < best_distance:
                best_distance = scaled_distance
                best_cluster = cluster_index
        if best_cluster >= 0 and best_distance <= 1.0:
            clusters[best_cluster].append(index)
        elif len(clusters) < maximum_clusters:
            centers.append(index)
            clusters.append([index])
        elif best_cluster >= 0:
            clusters[best_cluster].append(index)
    return tuple(tuple(cluster) for cluster in clusters if cluster)


def _systematic_indices(
    probabilities: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    active_count = max(int(count), 1)
    normalized = np.array(probabilities, dtype=float, copy=True)
    total = float(np.sum(normalized))
    if normalized.ndim != 1 or normalized.size == 0 or not np.isfinite(total) or total <= 0.0:
        raise ValueError("Systematic resampling requires positive finite probabilities.")
    normalized /= total
    positions = (rng.random() + np.arange(active_count, dtype=float)) / active_count
    return np.searchsorted(np.cumsum(normalized), positions, side="right")


def _probabilities_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Log weights must be a non-empty finite vector.")
    return np.exp(values - float(logsumexp(values)))


def _clone_particle(particle: _Particle, particle_id: str) -> _Particle:
    return _Particle(
        particle_id=particle_id,
        root_mode_id=particle.root_mode_id,
        normalized_values=particle.normalized_values.copy(),
        parameter_values=particle.parameter_values.copy(),
        profile=particle.profile,
        shape_chi2=float(particle.shape_chi2),
        phase_deviance=float(particle.phase_deviance),
        prior_chi2=float(particle.prior_chi2),
        phase_coherence=float(particle.phase_coherence),
        phase_signature=tuple(particle.phase_signature),
        log_weight=float(particle.log_weight),
        accepted_moves=int(particle.accepted_moves),
    )


def _root_particle_indices(
    particles: Sequence[_Particle],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    grouped: dict[str, list[int]] = {}
    for index, particle in enumerate(particles):
        grouped.setdefault(str(particle.root_mode_id), []).append(index)
    return tuple(
        (root_mode_id, tuple(indices))
        for root_mode_id, indices in grouped.items()
    )


def _protected_quotas(
    masses: np.ndarray,
    *,
    target: int,
    minimum_count: int,
    minimum_fraction: float,
) -> np.ndarray:
    probabilities = np.asarray(masses, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("Protected quotas require at least one population.")
    probabilities = probabilities / float(np.sum(probabilities))
    active_target = max(int(target), probabilities.size)
    requested_floor = max(
        int(minimum_count),
        int(math.ceil(max(float(minimum_fraction), 0.0) * active_target)),
        1,
    )
    floor = min(requested_floor, active_target // probabilities.size)
    quotas = np.full(probabilities.size, floor, dtype=int)
    remaining = active_target - int(np.sum(quotas))
    if remaining <= 0:
        return quotas
    expected = remaining * probabilities
    additions = np.floor(expected).astype(int)
    quotas += additions
    leftover = active_target - int(np.sum(quotas))
    if leftover > 0:
        order = np.argsort(expected - additions)[::-1]
        quotas[order[:leftover]] += 1
    return quotas


def _root_preserving_resample(
    particles: Sequence[_Particle],
    *,
    config: FullSequenceParticleConfig,
    rng: np.random.Generator,
    serial_start: int,
) -> tuple[list[_Particle], int]:
    roots = _root_particle_indices(particles)
    if not roots:
        raise RuntimeError("Cannot resample an empty particle population.")
    log_weights = np.asarray([particle.log_weight for particle in particles])
    root_log_masses = np.asarray(
        [
            float(logsumexp(log_weights[np.asarray(indices, dtype=int)]))
            for _, indices in roots
        ],
        dtype=float,
    )
    root_masses = _probabilities_from_log_weights(root_log_masses)
    quotas = _protected_quotas(
        root_masses,
        target=max(int(config.particle_count), len(roots)),
        minimum_count=max(int(config.minimum_root_particles), 1),
        minimum_fraction=max(float(config.minimum_root_fraction), 0.0),
    )
    output: list[_Particle] = []
    serial = int(serial_start)
    for (_, indices), root_mass, quota in zip(
        roots, root_masses, quotas, strict=True
    ):
        source_indices = np.asarray(indices, dtype=int)
        conditional = _probabilities_from_log_weights(log_weights[source_indices])
        selected = _systematic_indices(conditional, int(quota), rng)
        child_log_weight = math.log(max(float(root_mass), 1.0e-300)) - math.log(
            int(quota)
        )
        for local_index in selected:
            child = _clone_particle(
                particles[int(source_indices[int(local_index)])],
                f"particle_{serial:06d}",
            )
            serial += 1
            child.log_weight = child_log_weight
            output.append(child)
    _normalise_log_weights(output)
    return output, serial


def _stratified_mode_resample(
    particles: Sequence[_Particle],
    clusters: Sequence[Sequence[int]],
    *,
    config: FullSequenceParticleConfig,
    rng: np.random.Generator,
    serial_start: int,
) -> tuple[list[_Particle], int]:
    target = max(int(config.particle_count), len(clusters))
    log_weights = np.asarray([particle.log_weight for particle in particles], dtype=float)
    cluster_log_masses = np.asarray(
        [
            float(logsumexp(log_weights[np.asarray(cluster, dtype=int)]))
            for cluster in clusters
        ],
        dtype=float,
    )
    cluster_masses = _probabilities_from_log_weights(cluster_log_masses)
    minimum = max(int(config.minimum_cluster_particles), 1)
    if minimum * len(clusters) > target:
        minimum = 1
    quotas = np.full(len(clusters), minimum, dtype=int)
    remaining = target - int(np.sum(quotas))
    if remaining > 0:
        expected = remaining * cluster_masses
        additions = np.floor(expected).astype(int)
        quotas += additions
        leftover = target - int(np.sum(quotas))
        if leftover > 0:
            remainder_order = np.argsort(expected - additions)[::-1]
            quotas[remainder_order[:leftover]] += 1

    output: list[_Particle] = []
    serial = int(serial_start)
    for cluster, cluster_mass, quota in zip(
        clusters, cluster_masses, quotas, strict=True
    ):
        indices = np.asarray(cluster, dtype=int)
        conditional = _probabilities_from_log_weights(log_weights[indices])
        selected = _systematic_indices(conditional, int(quota), rng)
        child_log_weight = math.log(max(float(cluster_mass), 1.0e-300)) - math.log(
            int(quota)
        )
        for local_index in selected:
            child = _clone_particle(
                particles[int(indices[int(local_index)])],
                f"particle_{serial:06d}",
            )
            serial += 1
            child.log_weight = child_log_weight
            output.append(child)
    _normalise_log_weights(output)
    return output, serial


def _phase_metrics(
    particle: _Particle,
    *,
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceParticleConfig,
) -> tuple[float, float, tuple[float, ...]]:
    return wrapped_gaussian_phase_deviance(
        particle.profile.session_offsets,
        reference_variances,
        phase_sigma=phase_sigma,
        log_tolerance=config.wrapped_gaussian_log_tolerance,
        maximum_grid_points=config.wrapped_gaussian_max_grid_points,
    )


def _evaluate_particle(
    normalized_values: np.ndarray,
    *,
    particle_id: str,
    root_mode_id: str,
    evaluator: Any,
    coordinates: Any,
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceParticleConfig,
) -> _Particle | None:
    lower, upper = _normalized_bounds(coordinates)
    normalized = np.asarray(normalized_values, dtype=float)
    if normalized.shape != lower.shape or np.any(normalized <= lower) or np.any(
        normalized >= upper
    ):
        return None
    try:
        parameter_values = _physical_values(coordinates, normalized)
        profile = evaluator.evaluate_values(parameter_values)
        deviance, coherence, signature = wrapped_gaussian_phase_deviance(
            profile.session_offsets,
            reference_variances,
            phase_sigma=phase_sigma,
            log_tolerance=config.wrapped_gaussian_log_tolerance,
            maximum_grid_points=config.wrapped_gaussian_max_grid_points,
        )
    except (
        ArithmeticError,
        FloatingPointError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ):
        return None
    shape = float(profile.chi2)
    prior = max(float(config.prior_precision), 0.0) * float(
        normalized @ normalized
    )
    if not np.isfinite(shape + deviance + prior):
        return None
    return _Particle(
        particle_id=particle_id,
        root_mode_id=root_mode_id,
        normalized_values=normalized.copy(),
        parameter_values=parameter_values,
        profile=profile,
        shape_chi2=shape,
        phase_deviance=float(deviance),
        prior_chi2=prior,
        phase_coherence=float(coherence),
        phase_signature=signature,
    )


def _initial_population(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    reference_variances: np.ndarray,
    config: FullSequenceParticleConfig,
    rng: np.random.Generator,
) -> tuple[list[_Particle], int]:
    components = _proposal_components(
        invariant_modes=invariant_modes,
        coordinates=coordinates,
        config=config,
    )
    root_components: dict[str, tuple[_GaussianProposalComponent, ...]] = {}
    for root_mode_id in dict.fromkeys(
        component.root_mode_id for component in components
    ):
        root_components[root_mode_id] = tuple(
            component
            for component in components
            if component.root_mode_id == root_mode_id
        )
    requested = max(int(config.particle_count), 16, len(root_components))
    equal_root_masses = np.full(len(root_components), 1.0 / len(root_components))
    root_quotas = _protected_quotas(
        equal_root_masses,
        target=requested,
        minimum_count=max(int(config.minimum_root_particles), 1),
        minimum_fraction=max(float(config.minimum_root_fraction), 0.0),
    )
    particles: list[_Particle] = []
    serial = 0
    for (root_mode_id, local_components), quota in zip(
        root_components.items(), root_quotas, strict=True
    ):
        accepted = 0
        attempts = 0
        maximum_attempts = max(int(quota) * 40, 40)
        while accepted < int(quota) and attempts < maximum_attempts:
            if accepted == 0:
                candidate = local_components[0].center.copy()
            else:
                candidate, _ = _sample_proposal_component(local_components, rng)
            particle = _evaluate_particle(
                candidate,
                particle_id=f"particle_{serial:06d}",
                root_mode_id=root_mode_id,
                evaluator=evaluator,
                coordinates=coordinates,
                reference_variances=reference_variances,
                phase_sigma=config.initial_phase_sigma,
                config=config,
            )
            attempts += 1
            if particle is None:
                continue
            particle.log_weight = (
                -0.5 * particle.objective_chi2
                - _mixture_log_proposal_density(candidate, components)
            )
            serial += 1
            accepted += 1
            particles.append(particle)
        if accepted == 0:
            raise RuntimeError(
                f"Declared root {root_mode_id!r} has no valid initial particle."
            )
    if len(particles) < max(16, requested // 2):
        raise RuntimeError("Could not initialize enough valid full-sequence particles.")
    _normalise_log_weights(particles)
    if config.preserve_root_lineages:
        particles, serial = _root_preserving_resample(
            particles, config=config, rng=rng, serial_start=serial
        )
    else:
        clusters = _cluster_particles(particles, config=config)
        particles, serial = _stratified_mode_resample(
            particles,
            clusters,
            config=config,
            rng=rng,
            serial_start=serial,
        )
    return particles, serial


def _resume_population(
    summaries: Sequence[FullSequenceParticleSummary],
    *,
    evaluator: Any,
    coordinates: Any,
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceParticleConfig,
) -> tuple[list[_Particle], int]:
    particles: list[_Particle] = []
    serial = 0
    for summary in summaries:
        particle = _evaluate_particle(
            np.asarray(summary.normalized_values, dtype=float),
            particle_id=str(summary.particle_id),
            root_mode_id=str(summary.root_mode_id),
            evaluator=evaluator,
            coordinates=coordinates,
            reference_variances=reference_variances,
            phase_sigma=phase_sigma,
            config=config,
        )
        if particle is None:
            continue
        particle.log_weight = float(summary.log_weight)
        particle.accepted_moves = int(summary.accepted_moves)
        particles.append(particle)
        try:
            serial = max(serial, int(str(summary.particle_id).rsplit("_", 1)[-1]) + 1)
        except ValueError:
            serial += 1
    if len(particles) < max(16, len(summaries) // 2):
        raise RuntimeError("Could not restore enough valid full-sequence particles.")
    _normalise_log_weights(particles)
    return particles, serial


def _candidate_metrics(
    particles: Sequence[_Particle],
    *,
    reference_variances: np.ndarray,
    phase_sigma: float,
    config: FullSequenceParticleConfig,
) -> tuple[tuple[float, float, tuple[float, ...]], ...]:
    return tuple(
        _phase_metrics(
            particle,
            reference_variances=reference_variances,
            phase_sigma=phase_sigma,
            config=config,
        )
        for particle in particles
    )


def _local_wrapped_quadratic_deviance(
    particle: _Particle,
    *,
    reference_variances: np.ndarray,
    phase_sigma: float,
) -> float:
    offsets = np.asarray(particle.profile.session_offsets, dtype=float)
    relative = _wrapped_difference(offsets, np.full(offsets.shape, offsets[0]))
    inverse_variance = 1.0 / (
        np.asarray(reference_variances, dtype=float) + float(phase_sigma) ** 2
    )
    resultant = np.sum(
        inverse_variance * np.exp(2.0j * math.pi * relative)
    )
    gauge = float(np.angle(resultant) / (2.0 * math.pi))
    residual = _wrapped_difference(relative, np.full(relative.shape, gauge))
    return float(np.sum(inverse_variance * np.square(residual)))


def _surrogate_candidate_metrics(
    particles: Sequence[_Particle],
    *,
    current_sigma: float,
    candidate_sigma: float,
    reference_variances: np.ndarray,
) -> tuple[tuple[float, float, tuple[float, ...]], ...]:
    result: list[tuple[float, float, tuple[float, ...]]] = []
    for particle in particles:
        current = _local_wrapped_quadratic_deviance(
            particle,
            reference_variances=reference_variances,
            phase_sigma=current_sigma,
        )
        candidate = _local_wrapped_quadratic_deviance(
            particle,
            reference_variances=reference_variances,
            phase_sigma=candidate_sigma,
        )
        result.append(
            (
                float(particle.phase_deviance + candidate - current),
                float(particle.phase_coherence),
                tuple(particle.phase_signature),
            )
        )
    return tuple(result)


def _reweighted_ess(
    particles: Sequence[_Particle],
    candidate_metrics: Sequence[tuple[float, float, tuple[float, ...]]],
) -> float:
    values = np.asarray(
        [
            particle.log_weight
            - 0.5 * (float(metrics[0]) - particle.phase_deviance)
            for particle, metrics in zip(particles, candidate_metrics, strict=True)
        ],
        dtype=float,
    )
    return _effective_sample_size(values)


def _choose_next_sigma(
    particles: Sequence[_Particle],
    *,
    current_sigma: float,
    reference_variances: np.ndarray,
    config: FullSequenceParticleConfig,
) -> tuple[float, tuple[tuple[float, float, tuple[float, ...]], ...]]:
    final_sigma = float(config.final_phase_sigma)
    lower_sigma = max(
        final_sigma,
        current_sigma * min(max(float(config.maximum_sigma_ratio), 0.05), 0.999),
    )
    current_ess = _effective_sample_size(
        [particle.log_weight for particle in particles]
    )
    target_ess = max(float(config.ess_target_fraction), 0.1) * current_ess
    exact_cache: dict[float, tuple[tuple[float, float, tuple[float, ...]], ...]] = {}

    def exact_metrics_at(sigma: float):
        key = float(sigma)
        if key not in exact_cache:
            exact_cache[key] = _candidate_metrics(
                particles,
                reference_variances=reference_variances,
                phase_sigma=key,
                config=config,
            )
        return exact_cache[key]

    def surrogate_ess(sigma: float) -> float:
        metrics = _surrogate_candidate_metrics(
            particles,
            current_sigma=current_sigma,
            candidate_sigma=sigma,
            reference_variances=reference_variances,
        )
        return _reweighted_ess(particles, metrics)

    if surrogate_ess(lower_sigma) >= target_ess:
        proposed_sigma = lower_sigma
    else:
        low = math.log(lower_sigma)
        high = math.log(current_sigma)
        for _ in range(24):
            middle = 0.5 * (low + high)
            sigma = math.exp(middle)
            if surrogate_ess(sigma) >= target_ess:
                high = middle
            else:
                low = middle
        proposed_sigma = math.exp(high)

    minimum_change = max(float(config.minimum_relative_sigma_step), 1.0e-6)
    if proposed_sigma >= current_sigma * (1.0 - minimum_change):
        proposed_sigma = max(final_sigma, current_sigma * (1.0 - minimum_change))

    proposed_metrics = exact_metrics_at(proposed_sigma)
    if _reweighted_ess(particles, proposed_metrics) >= target_ess:
        return proposed_sigma, proposed_metrics

    low = math.log(proposed_sigma)
    high = math.log(current_sigma)
    best_sigma = current_sigma
    best_metrics = tuple(
        (
            float(particle.phase_deviance),
            float(particle.phase_coherence),
            tuple(particle.phase_signature),
        )
        for particle in particles
    )
    for _ in range(max(int(config.maximum_exact_temperature_refinements), 1)):
        middle = 0.5 * (low + high)
        sigma = math.exp(middle)
        metrics = exact_metrics_at(sigma)
        if _reweighted_ess(particles, metrics) >= target_ess:
            high = middle
            best_sigma = sigma
            best_metrics = metrics
        else:
            low = middle
    if best_sigma >= current_sigma * (1.0 - minimum_change):
        best_sigma = max(final_sigma, current_sigma * (1.0 - minimum_change))
        best_metrics = exact_metrics_at(best_sigma)
    return best_sigma, best_metrics


def _apply_temperature(
    particles: Sequence[_Particle],
    metrics: Sequence[tuple[float, float, tuple[float, ...]]],
) -> None:
    for particle, (deviance, coherence, signature) in zip(
        particles, metrics, strict=True
    ):
        particle.log_weight -= 0.5 * (float(deviance) - particle.phase_deviance)
        particle.phase_deviance = float(deviance)
        particle.phase_coherence = float(coherence)
        particle.phase_signature = tuple(signature)
    _normalise_log_weights(particles)


def _rejuvenate_population(
    particles: list[_Particle],
    *,
    evaluator: Any,
    coordinates: Any,
    reference_variances: np.ndarray,
    phase_sigma: float,
    fallback_covariance: np.ndarray,
    proposal_components: Sequence[_GaussianProposalComponent],
    config: FullSequenceParticleConfig,
    rng: np.random.Generator,
    serial_start: int,
) -> tuple[int, int, int]:
    accepted = 0
    evaluated = 0
    serial = int(serial_start)
    dimension = particles[0].normalized_values.size
    scale_multipliers = tuple(
        float(value)
        for value in config.proposal_scale_multipliers
        if np.isfinite(float(value)) and float(value) > 0.0
    ) or (1.0,)
    for _ in range(max(int(config.move_steps_per_temperature), 0)):
        clusters = _cluster_particles(particles, config=config)
        cluster_by_particle: dict[int, tuple[int, ...]] = {}
        roots: dict[tuple[int, ...], np.ndarray] = {}
        for cluster in clusters:
            values = np.asarray(
                [particles[index].normalized_values for index in cluster], dtype=float
            )
            covariance = _regularized_covariance(
                values,
                fallback=fallback_covariance,
                jitter=config.proposal_jitter,
            )
            roots[tuple(cluster)] = _covariance_root(covariance)
            for index in cluster:
                cluster_by_particle[index] = tuple(cluster)
        snapshot = tuple(particles)
        for index, particle in enumerate(particles):
            scale_multiplier = float(rng.choice(scale_multipliers))
            cluster = cluster_by_particle[index]
            pool = cluster if len(cluster) >= 3 else tuple(range(len(snapshot)))
            independent = rng.random() < np.clip(
                float(config.independent_proposal_probability), 0.0, 1.0
            )
            proposal = None
            proposal_values = particle.normalized_values
            proposal_root_mode_id = particle.root_mode_id
            if independent:
                lineage_components = tuple(
                    component
                    for component in proposal_components
                    if component.root_mode_id == particle.root_mode_id
                ) or tuple(proposal_components)
                for _ in range(16):
                    proposal_values, proposal_root_mode_id = _sample_proposal_component(
                        lineage_components,
                        rng,
                    )
                    proposal = _evaluate_particle(
                        proposal_values,
                        particle_id=f"particle_{serial:06d}",
                        root_mode_id=proposal_root_mode_id,
                        evaluator=evaluator,
                        coordinates=coordinates,
                        reference_variances=reference_variances,
                        phase_sigma=phase_sigma,
                        config=config,
                    )
                    serial += 1
                    if proposal is not None:
                        break
            elif (
                len(pool) >= 3
                and rng.random()
                < np.clip(float(config.differential_evolution_probability), 0.0, 1.0)
            ):
                if rng.random() < np.clip(
                    float(config.global_differential_evolution_probability), 0.0, 1.0
                ):
                    pool = tuple(range(len(snapshot)))
                choices = [value for value in pool if value != index]
                first, second = rng.choice(choices, size=2, replace=False)
                gamma = (
                    max(float(config.differential_evolution_scale), 0.0)
                    * scale_multiplier
                    * 2.38
                    / math.sqrt(2.0 * dimension)
                )
                proposal_values = (
                    particle.normalized_values
                    + gamma
                    * (
                        snapshot[int(first)].normalized_values
                        - snapshot[int(second)].normalized_values
                    )
                    + rng.normal(
                        0.0,
                        max(float(config.proposal_jitter), 1.0e-8),
                        size=dimension,
                    )
                )
            else:
                scale = (
                    max(float(config.random_walk_scale), 0.0)
                    * scale_multiplier
                    * 2.38
                    / math.sqrt(dimension)
                )
                proposal_values = particle.normalized_values + scale * (
                    roots[cluster] @ rng.normal(size=dimension)
                )
            if proposal is None:
                proposal = _evaluate_particle(
                    proposal_values,
                    particle_id=f"particle_{serial:06d}",
                    root_mode_id=proposal_root_mode_id,
                    evaluator=evaluator,
                    coordinates=coordinates,
                    reference_variances=reference_variances,
                    phase_sigma=phase_sigma,
                    config=config,
                )
                serial += 1
            if proposal is None:
                continue
            evaluated += 1
            log_alpha = -0.5 * (proposal.objective_chi2 - particle.objective_chi2)
            if independent:
                log_alpha += _mixture_log_proposal_density(
                    particle.normalized_values,
                    lineage_components,
                ) - _mixture_log_proposal_density(
                    proposal.normalized_values,
                    lineage_components,
                )
            if math.log(rng.random()) < min(0.0, log_alpha):
                proposal.log_weight = particle.log_weight
                proposal.accepted_moves = particle.accepted_moves + 1
                particles[index] = proposal
                accepted += 1
    return accepted, evaluated, serial


def _particle_summaries(
    particles: Sequence[_Particle],
) -> tuple[FullSequenceParticleSummary, ...]:
    return tuple(
        FullSequenceParticleSummary(
            particle_id=particle.particle_id,
            root_mode_id=particle.root_mode_id,
            normalized_values=particle.normalized_values.copy(),
            parameter_values=particle.parameter_values.copy(),
            shape_chi2=float(particle.shape_chi2),
            phase_deviance=float(particle.phase_deviance),
            prior_chi2=float(particle.prior_chi2),
            objective_chi2=float(particle.objective_chi2),
            phase_coherence=float(particle.phase_coherence),
            phase_signature=tuple(particle.phase_signature),
            log_weight=float(particle.log_weight),
            accepted_moves=int(particle.accepted_moves),
        )
        for particle in particles
    )


def _population_modes(
    particles: Sequence[_Particle],
    *,
    phase_sigma: float,
    fallback_covariance: np.ndarray,
    config: FullSequenceParticleConfig,
) -> tuple[FullSequenceCircularMode, ...]:
    clusters = _cluster_particles(particles, config=config)
    records: list[FullSequenceCircularMode] = []
    weights = np.exp(np.asarray([particle.log_weight for particle in particles]))
    for mode_index, cluster in enumerate(clusters):
        cluster_indices = np.asarray(cluster, dtype=int)
        cluster_mass = float(np.sum(weights[cluster_indices]))
        representative_index = min(
            cluster,
            key=lambda index: particles[index].objective_chi2,
        )
        representative = particles[representative_index]
        values = np.asarray(
            [particles[index].normalized_values for index in cluster], dtype=float
        )
        covariance = _regularized_covariance(
            values,
            fallback=fallback_covariance,
            jitter=config.proposal_jitter,
        )
        records.append(
            FullSequenceCircularMode(
                mode_id=f"particle_mode_{mode_index:04d}",
                root_mode_id=representative.root_mode_id,
                parent_mode_id=None,
                proposal_source="full-sequence-particle-posterior",
                parameter_values=representative.parameter_values.copy(),
                normalized_values=representative.normalized_values.copy(),
                covariance=covariance,
                phase_sigma=float(phase_sigma),
                shape_chi2=float(representative.shape_chi2),
                circular_deviance=float(representative.phase_deviance),
                prior_chi2=float(representative.prior_chi2),
                objective_chi2=float(representative.objective_chi2),
                log_mass=math.log(max(cluster_mass, 1.0e-300)),
                posterior_mass=cluster_mass,
                phase_coherence=float(representative.phase_coherence),
                phase_signature=tuple(representative.phase_signature),
                nfev=0,
                success=True,
            )
        )
    records.sort(key=lambda mode: (-mode.posterior_mass, mode.objective_chi2))
    return tuple(records[: max(int(config.maximum_output_modes), 1)])


def transport_full_sequence_particle_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: FullSequenceParticleConfig | None = None,
    resume_particles: Sequence[FullSequenceParticleSummary] | None = None,
    resume_phase_sigma: float | None = None,
    progress_callback: Callable[
        [ParticleTemperatureSummary, tuple[FullSequenceParticleSummary, ...]], None
    ]
    | None = None,
) -> FullSequenceParticleResult:
    """Transport the complete wrapped-phase posterior as a weighted population."""

    active = config or FullSequenceParticleConfig()
    if not invariant_modes:
        raise ValueError("Particle transport requires invariant posterior modes.")
    if not 0.0 < float(active.final_phase_sigma) < float(
        active.initial_phase_sigma
    ):
        raise ValueError("Particle phase sigmas must be positive and decreasing.")
    rng = np.random.default_rng(int(active.random_seed))
    reference_profile = evaluator.evaluate_values(
        np.asarray(invariant_modes[0].parameter_values, dtype=float)
    )
    reference_variances = np.asarray(
        reference_profile.session_offset_variances, dtype=float
    )
    if resume_particles:
        if resume_phase_sigma is None:
            raise ValueError("A resumed particle population requires its phase sigma.")
        current_sigma = float(resume_phase_sigma)
        if not float(active.final_phase_sigma) < current_sigma <= float(
            active.initial_phase_sigma
        ):
            raise ValueError(
                "The resumed particle sigma must lie above the final sigma and "
                "at or below the initial sigma."
            )
        particles, serial = _resume_population(
            resume_particles,
            evaluator=evaluator,
            coordinates=coordinates,
            reference_variances=reference_variances,
            phase_sigma=current_sigma,
            config=active,
        )
    else:
        particles, serial = _initial_population(
            evaluator=evaluator,
            coordinates=coordinates,
            invariant_modes=invariant_modes,
            reference_variances=reference_variances,
            config=active,
            rng=rng,
        )
        current_sigma = float(active.initial_phase_sigma)
    fallback_covariance = np.asarray(invariant_modes[0].covariance, dtype=float)
    proposal_components = _proposal_components(
        invariant_modes=invariant_modes,
        coordinates=coordinates,
        config=active,
    )
    history: list[ParticleTemperatureSummary] = []

    for temperature_index in range(max(int(active.maximum_temperature_steps), 1)):
        if current_sigma <= float(active.final_phase_sigma) * (1.0 + 1.0e-12):
            break
        next_sigma, metrics = _choose_next_sigma(
            particles,
            current_sigma=current_sigma,
            reference_variances=reference_variances,
            config=active,
        )
        _apply_temperature(particles, metrics)
        ess = _effective_sample_size(
            [particle.log_weight for particle in particles]
        )
        clusters = _cluster_particles(particles, config=active)
        resampled = bool(
            ess
            <= max(float(active.resample_threshold_fraction), 0.0)
            * len(particles)
        )
        if resampled:
            if active.preserve_root_lineages:
                particles, serial = _root_preserving_resample(
                    particles,
                    config=active,
                    rng=rng,
                    serial_start=serial,
                )
            else:
                particles, serial = _stratified_mode_resample(
                    particles,
                    clusters,
                    config=active,
                    rng=rng,
                    serial_start=serial,
                )
            clusters = _cluster_particles(particles, config=active)
        accepted, evaluated, serial = _rejuvenate_population(
            particles,
            evaluator=evaluator,
            coordinates=coordinates,
            reference_variances=reference_variances,
            phase_sigma=next_sigma,
            fallback_covariance=fallback_covariance,
            proposal_components=proposal_components,
            config=active,
            rng=rng,
            serial_start=serial,
        )
        current_sigma = float(next_sigma)
        root_counts = tuple(
            len(indices) for _, indices in _root_particle_indices(particles)
        )
        summary = ParticleTemperatureSummary(
            temperature_index=temperature_index,
            phase_sigma=current_sigma,
            effective_sample_size=float(ess),
            cluster_count=len(clusters),
            root_count=len(root_counts),
            minimum_root_particle_count=min(root_counts),
            resampled=resampled,
            accepted_moves=int(accepted),
            evaluated_proposals=int(evaluated),
            best_shape_chi2=min(particle.shape_chi2 for particle in particles),
            best_phase_deviance=min(
                particle.phase_deviance for particle in particles
            ),
            best_objective_chi2=min(
                particle.objective_chi2 for particle in particles
            ),
            maximum_phase_coherence=max(
                particle.phase_coherence for particle in particles
            ),
        )
        history.append(summary)
        if progress_callback is not None:
            progress_callback(summary, _particle_summaries(particles))

    particle_summaries = _particle_summaries(particles)
    modes = _population_modes(
        particles,
        phase_sigma=current_sigma,
        fallback_covariance=fallback_covariance,
        config=active,
    )
    return FullSequenceParticleResult(
        particles=particle_summaries,
        circular_modes=modes,
        temperature_history=tuple(history),
        reached_final_temperature=bool(
            current_sigma <= float(active.final_phase_sigma) * (1.0 + 1.0e-10)
        ),
        source_session_count=int(reference_profile.session_count),
    )


def _select_path_parent_particles(
    particles: Sequence[FullSequenceParticleSummary],
    *,
    config: FullSequenceParticleConfig,
) -> tuple[FullSequenceParticleSummary, ...]:
    grouped: dict[str, list[FullSequenceParticleSummary]] = {}
    for particle in particles:
        grouped.setdefault(str(particle.root_mode_id), []).append(particle)
    protected: list[FullSequenceParticleSummary] = []
    minimum = max(int(config.minimum_path_parents_per_root), 1)
    for records in grouped.values():
        records.sort(key=lambda item: item.log_weight, reverse=True)
        protected.extend(records[:minimum])
    target = max(int(config.path_parent_particles), len(protected), 1)
    selected_ids = {item.particle_id for item in protected}
    remaining = sorted(particles, key=lambda item: item.log_weight, reverse=True)
    for particle in remaining:
        if len(protected) >= target:
            break
        if particle.particle_id in selected_ids:
            continue
        protected.append(particle)
        selected_ids.add(particle.particle_id)
    return tuple(protected)


def _select_root_stratified_path_proposals(
    proposals: Sequence[LatticeProposal],
    *,
    config: FullSequenceParticleConfig,
) -> tuple[LatticeProposal, ...]:
    grouped: dict[str, list[LatticeProposal]] = {}
    families: dict[tuple[str, str], list[LatticeProposal]] = {}
    for proposal in proposals:
        grouped.setdefault(str(proposal.root_mode_id), []).append(proposal)
        family = _path_proposal_family(proposal)
        families.setdefault((str(proposal.root_mode_id), family), []).append(proposal)
    protected: list[LatticeProposal] = []
    selected_keys: set[tuple[str, tuple[int, ...], str | None, str]] = set()

    def add(proposal: LatticeProposal) -> None:
        key = _path_proposal_identity(proposal)
        if key not in selected_keys:
            protected.append(proposal)
            selected_keys.add(key)

    family_minimum = max(int(config.minimum_path_proposals_per_family), 1)
    for records in families.values():
        records.sort(key=lambda item: item.log_predictive_density, reverse=True)
        for proposal in records[:family_minimum]:
            add(proposal)
    root_minimum = max(int(config.minimum_path_proposals_per_root), 1)
    for records in grouped.values():
        records.sort(key=lambda item: item.log_predictive_density, reverse=True)
        root_count = sum(
            item.root_mode_id == records[0].root_mode_id for item in protected
        )
        for proposal in records:
            if root_count >= root_minimum:
                break
            before = len(protected)
            add(proposal)
            root_count += int(len(protected) > before)
    target = max(int(config.maximum_path_proposals), len(protected), 1)
    for proposal in sorted(
        proposals,
        key=lambda item: item.log_predictive_density,
        reverse=True,
    ):
        if len(protected) >= target:
            break
        key = _path_proposal_identity(proposal)
        if key in selected_keys:
            continue
        add(proposal)
    return tuple(protected)


def _path_proposal_family(proposal: LatticeProposal) -> str:
    source = str(proposal.source)
    if "low-rank" in source:
        return "low-rank"
    if "conditional" in source:
        return "conditional"
    return source


def _path_proposal_identity(
    proposal: LatticeProposal,
) -> tuple[str, tuple[int, ...], str | None, str]:
    return (
        str(proposal.root_mode_id),
        tuple(proposal.integer_offsets),
        proposal.parent_mode_id,
        _path_proposal_family(proposal),
    )


def _low_rank_correlated_path_proposals(
    *,
    evaluator: Any,
    coordinates: Any,
    particle: FullSequenceParticleSummary,
    config: FullSequenceParticleConfig,
) -> tuple[LatticeProposal, ...]:
    """Transport a global path in parameter space, not integer space.

    The session ambiguity covariance is a diagonal noise term plus a
    low-rank timing-parameter term. Searching the full integer dimension is
    therefore unnecessary: anneal the linearized wrapped likelihood along
    the phase-informative parameter directions, then recover all session
    integers conditionally from each transported continuous mode.
    """

    center = np.asarray(particle.normalized_values, dtype=float)
    lower, upper = _normalized_bounds(coordinates)
    linearization = finite_difference_profile(
        evaluator=evaluator,
        coordinates=coordinates,
        normalized_values=center,
        step=max(min(float(config.proposal_jitter) * 2.0, 2.0e-3), 1.0e-7),
    )
    profile = linearization.profile
    if int(profile.session_count) <= 1:
        return tuple()
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    relative = offsets[1:] - offsets[0]
    phase_jacobian = (
        np.asarray(linearization.offset_jacobian[1:], dtype=float)
        - np.asarray(linearization.offset_jacobian[0], dtype=float)
    )
    noise = np.diag(variances[1:]) + float(variances[0])

    dimension = center.size
    prior_precision = max(float(config.prior_precision), 0.0)
    try:
        _left, singular_values, right = np.linalg.svd(
            shape_jacobian,
            full_matrices=True,
        )
    except np.linalg.LinAlgError:
        return tuple()
    precision = np.full(dimension, max(prior_precision, 1.0e-12), dtype=float)
    represented = min(int(singular_values.size), dimension)
    precision[:represented] += np.square(singular_values[:represented])
    covariance = (right.T * (1.0 / precision)[np.newaxis, :]) @ right
    covariance = 0.5 * (covariance + covariance.T)
    gradient = shape_jacobian.T @ shape + prior_precision * center
    conditional_mean = np.clip(center - covariance @ gradient, lower, upper)
    covariance_root = _covariance_root(covariance)

    try:
        _phase_left, _phase_values, phase_right = np.linalg.svd(
            phase_jacobian @ covariance_root,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return tuple()
    direction_count = min(
        max(int(config.low_rank_path_directions), 1),
        phase_right.shape[0],
    )
    directions = covariance_root @ phase_right[:direction_count].T
    starts: list[np.ndarray] = [center.copy(), conditional_mean.copy()]
    for scale in config.low_rank_path_scales:
        if not np.isfinite(float(scale)) or float(scale) <= 0.0:
            continue
        for direction in directions.T:
            for sign in (-1.0, 1.0):
                candidate = np.clip(
                    conditional_mean + sign * float(scale) * direction,
                    lower,
                    upper,
                )
                if any(np.max(np.abs(candidate - prior)) <= 1.0e-10 for prior in starts):
                    continue
                starts.append(candidate)
                if len(starts) >= max(int(config.low_rank_path_starts), 1):
                    break
            if len(starts) >= max(int(config.low_rank_path_starts), 1):
                break
        if len(starts) >= max(int(config.low_rank_path_starts), 1):
            break

    ladder = tuple(
        sorted(
            {
                float(value)
                for value in config.low_rank_path_phase_sigmas
                if np.isfinite(float(value)) and float(value) > 0.0
            },
            reverse=True,
        )
    ) or (0.5, 0.1, 0.03)
    per_stage_nfev = max(
        8,
        int(math.ceil(max(int(config.low_rank_path_max_nfev), 1) / len(ladder))),
    )

    def predicted(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        displacement = np.asarray(state, dtype=float) - center
        return (
            shape + shape_jacobian @ displacement,
            relative + phase_jacobian @ displacement,
        )

    proposals: dict[tuple[int, ...], LatticeProposal] = {}
    for start_index, start in enumerate(starts):
        state = np.asarray(start, dtype=float).copy()
        for phase_sigma in ladder:
            annealed_noise = noise + float(phase_sigma) ** 2 * (
                np.eye(relative.size, dtype=float)
                + np.ones((relative.size, relative.size), dtype=float)
            )
            cholesky = np.linalg.cholesky(annealed_noise)

            def residual_vector(values: np.ndarray) -> np.ndarray:
                local_shape, local_phase = predicted(values)
                wrapped = (local_phase + 0.5) % 1.0 - 0.5
                parts = [local_shape, np.linalg.solve(cholesky, wrapped)]
                if prior_precision > 0.0:
                    parts.append(math.sqrt(prior_precision) * values)
                return np.concatenate(parts)

            fit = least_squares(
                residual_vector,
                state,
                bounds=(lower, upper),
                method="trf",
                jac="2-point",
                max_nfev=per_stage_nfev,
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
            )
            state = np.asarray(fit.x, dtype=float)
        local_shape, local_phase = predicted(state)
        integer_offsets = tuple(int(value) for value in np.rint(local_phase))
        phase_delta = local_phase - np.asarray(integer_offsets, dtype=float)
        phase_chi2 = float(phase_delta @ np.linalg.solve(noise, phase_delta))
        objective = float(local_shape @ local_shape + phase_chi2)
        if prior_precision > 0.0:
            objective += float(prior_precision * (state @ state))
        proposal = LatticeProposal(
            root_mode_id=str(particle.root_mode_id),
            parent_mode_id=str(particle.particle_id),
            source="particle-low-rank-correlated-transport",
            active_sessions=tuple(range(1, int(profile.session_count))),
            integer_offsets=integer_offsets,
            initial_normalized_values=state.copy(),
            initial_parameter_values=_physical_values(coordinates, state),
            squared_mahalanobis=max(objective, 0.0),
            log_predictive_density=float(particle.log_weight - 0.5 * objective),
            kappa=float(ladder[-1]),
        )
        previous = proposals.get(integer_offsets)
        if previous is None or proposal.log_predictive_density > previous.log_predictive_density:
            proposals[integer_offsets] = proposal
    return tuple(
        sorted(
            proposals.values(),
            key=lambda item: item.log_predictive_density,
            reverse=True,
        )
    )


def recover_particle_integer_modes(
    *,
    evaluator: Any,
    coordinates: Any,
    particles: Sequence[FullSequenceParticleSummary],
    config: FullSequenceParticleConfig | None = None,
) -> tuple[tuple[GlobalLatticeMode, ...], int]:
    """Recover complete paths from every protected continuous root lineage."""

    active = config or FullSequenceParticleConfig()
    if not particles:
        return tuple(), 0
    phase_sigmas = tuple(
        sorted(
            {
                float(value)
                for value in active.integer_recovery_phase_sigmas
                if np.isfinite(float(value)) and float(value) >= 0.0
            }
        )
    ) or (0.0,)
    parent_particles = _select_path_parent_particles(particles, config=active)
    proposals_by_key: dict[
        tuple[str, tuple[int, ...], str | None, str], LatticeProposal
    ] = {}

    for particle in parent_particles:
        try:
            correlated = _low_rank_correlated_path_proposals(
                evaluator=evaluator,
                coordinates=coordinates,
                particle=particle,
                config=active,
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            correlated = tuple()
        for proposal in correlated:
            key = _path_proposal_identity(proposal)
            previous = proposals_by_key.get(key)
            if (
                previous is None
                or proposal.log_predictive_density > previous.log_predictive_density
            ):
                proposals_by_key[key] = proposal

    for particle in parent_particles:
        profile = evaluator.evaluate_values(
            np.asarray(particle.parameter_values, dtype=float)
        )
        active_sessions = tuple(range(1, int(profile.session_count)))
        for phase_sigma in phase_sigmas:
            candidates = conditional_integer_paths(
                profile,
                candidate_count=max(int(active.integer_candidates_per_particle), 1),
                added_session_phase_variance=phase_sigma * phase_sigma,
            )
            for candidate in candidates:
                proposal = LatticeProposal(
                    root_mode_id=str(particle.root_mode_id),
                    parent_mode_id=str(particle.particle_id),
                    source=(
                        "full-sequence-particle-conditional-"
                        f"sigma-{phase_sigma:.6g}"
                    ),
                    active_sessions=active_sessions,
                    integer_offsets=tuple(candidate.integer_offsets),
                    initial_normalized_values=np.asarray(
                        particle.normalized_values, dtype=float
                    ).copy(),
                    initial_parameter_values=np.asarray(
                        particle.parameter_values, dtype=float
                    ).copy(),
                    squared_mahalanobis=max(
                        -2.0 * float(candidate.log_conditional_mass), 0.0
                    ),
                    log_predictive_density=float(
                        particle.log_weight + candidate.log_conditional_mass
                    ),
                    kappa=max(phase_sigma, 1.0e-12),
                )
                key = _path_proposal_identity(proposal)
                previous = proposals_by_key.get(key)
                if (
                    previous is None
                    or proposal.log_predictive_density
                    > previous.log_predictive_density
                ):
                    proposals_by_key[key] = proposal
    proposals = _select_root_stratified_path_proposals(
        tuple(proposals_by_key.values()), config=active
    )
    exact_config = GlobalLatticeTrackerConfig(
        prior_precision=max(float(active.prior_precision), 0.0),
        fixed_path_max_nfev=max(int(active.exact_refit_max_nfev), 1),
        phase_tempering_ladder=tuple(active.exact_phase_tempering_ladder),
    )
    exact_modes: list[GlobalLatticeMode] = []
    for index, proposal in enumerate(proposals):
        try:
            exact_modes.append(
                refit_fixed_path_proposal(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    anchor_session=0,
                    proposal=proposal,
                    config=exact_config,
                    mode_id=f"particle_exact_{index:04d}",
                )
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    selected = select_diverse_lattice_modes(
        exact_modes,
        maximum_modes=max(int(active.maximum_exact_modes), 1),
        maximum_modes_per_path=max(int(active.maximum_exact_modes_per_path), 1),
        maximum_modes_per_root=max(int(active.maximum_exact_modes_per_root), 1),
        minimum_separation=max(float(active.exact_mode_separation), 0.0),
    )
    return selected, len(proposals)


def run_full_sequence_particle_transport(
    *,
    evaluator: Any,
    coordinates: Any,
    invariant_modes: Sequence[InvariantMode],
    config: FullSequenceParticleConfig | None = None,
    resume_particles: Sequence[FullSequenceParticleSummary] | None = None,
    resume_phase_sigma: float | None = None,
    progress_callback: Callable[
        [ParticleTemperatureSummary, tuple[FullSequenceParticleSummary, ...]], None
    ]
    | None = None,
) -> FullSequenceParticleResult:
    """Transport the circular posterior, then conditionally recover integers."""

    active = config or FullSequenceParticleConfig()
    result = transport_full_sequence_particle_modes(
        evaluator=evaluator,
        coordinates=coordinates,
        invariant_modes=invariant_modes,
        config=active,
        resume_particles=resume_particles,
        resume_phase_sigma=resume_phase_sigma,
        progress_callback=progress_callback,
    )
    exact_modes, proposal_count = recover_particle_integer_modes(
        evaluator=evaluator,
        coordinates=coordinates,
        particles=result.particles,
        config=active,
    )
    return replace(
        result,
        exact_modes=exact_modes,
        path_proposal_count=proposal_count,
    )


def diagnose_particle_integer_basins(
    *,
    evaluator: Any,
    particles: Sequence[FullSequenceParticleSummary],
    integer_candidates_per_particle: int = 8,
) -> tuple[ParticleIntegerBasinDiagnostic, ...]:
    """Measure exact fixed-path phase chi-square for every posterior particle."""

    diagnostics: list[ParticleIntegerBasinDiagnostic] = []
    candidate_count = max(int(integer_candidates_per_particle), 1)
    for particle in particles:
        profile = evaluator.evaluate_values(
            np.asarray(particle.parameter_values, dtype=float)
        )
        active_sessions = tuple(range(1, int(profile.session_count)))
        candidates = conditional_integer_paths(
            profile,
            candidate_count=candidate_count,
        )
        scored: list[tuple[float, float, float, Any]] = []
        for candidate in candidates:
            shape_chi2, phase_chi2, objective_chi2 = exact_fixed_path_chi2(
                profile,
                anchor_session=0,
                active_sessions=active_sessions,
                integer_offsets=candidate.integer_offsets,
            )
            scored.append((objective_chi2, shape_chi2, phase_chi2, candidate))
        if not scored:
            continue
        objective_chi2, shape_chi2, phase_chi2, candidate = min(
            scored,
            key=lambda item: item[0],
        )
        diagnostics.append(
            ParticleIntegerBasinDiagnostic(
                particle_id=particle.particle_id,
                root_mode_id=particle.root_mode_id,
                integer_offsets=tuple(int(value) for value in candidate.integer_offsets),
                shape_chi2=float(shape_chi2),
                phase_chi2=float(phase_chi2),
                objective_chi2=float(objective_chi2),
                log_conditional_mass=float(candidate.log_conditional_mass),
                particle_log_weight=float(particle.log_weight),
                phase_coherence=float(particle.phase_coherence),
            )
        )
    return tuple(sorted(diagnostics, key=lambda item: item.objective_chi2))


__all__ = [
    "FullSequenceParticleConfig",
    "FullSequenceParticleResult",
    "FullSequenceParticleSummary",
    "ParticleIntegerBasinDiagnostic",
    "ParticleTemperatureSummary",
    "diagnose_particle_integer_basins",
    "recover_particle_integer_modes",
    "run_full_sequence_particle_transport",
    "transport_full_sequence_particle_modes",
]
