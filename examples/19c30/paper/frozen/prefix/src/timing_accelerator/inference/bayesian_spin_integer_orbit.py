from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field, replace
from typing import Any, Sequence

import numpy as np
from scipy.special import expit, logit, logsumexp, ndtri

from timing_accelerator.inference.bayesian_conditioned_path_acquisition import (
    ConditionedPathAcquisitionResult,
    ConditionedPathDirectionConfig,
    JointContinuousIntegerLinearization,
    acquire_conditioned_path_directions,
)
from timing_accelerator.inference.bayesian_global_lattice_mode_tracker import (
    _physical_values,
)
from timing_accelerator.inference.integer_least_squares import (
    IntegerLeastSquaresResult,
    build_predictive_integer_covariance,
    solve_integer_least_squares,
)
from timing_accelerator.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
)


@dataclass(frozen=True)
class SpinIntegerMarginalConfig:
    """Numerical controls for the joint spin and integer marginalization."""

    initial_lattice_candidates: int = 64
    maximum_lattice_candidates: int = 512
    lattice_log_mass_tolerance: float = 1.0e-5
    maximum_lattice_nodes: int = 2_000_000
    added_session_phase_sigma_cycles: float = 0.0


@dataclass(frozen=True)
class SpinOrbitLinearization:
    """Spin response at one fixed nonlinear orbit state.

    The two Jacobians use the same dimensionless spin coordinates. Shape
    residuals have already had one intercept projected from every session.
    Relative offsets are measured against session zero.
    """

    profile: SessionShapeProfile
    spin_reference: np.ndarray
    shape_jacobian: np.ndarray
    relative_session_offsets: np.ndarray
    relative_session_jacobian: np.ndarray
    relative_offset_covariance: np.ndarray
    curvature_jacobian: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros((0, 0), dtype=float)
    )
    standardized_slope_jacobian: np.ndarray = dataclass_field(
        default_factory=lambda: np.zeros((0, 0), dtype=float)
    )


@dataclass(frozen=True)
class SpinIntegerPathMode:
    integer_offsets: tuple[int, ...]
    squared_mahalanobis: float
    log_posterior_weight: float
    posterior_weight: float
    spin_posterior_mean: np.ndarray
    spin_posterior_covariance: np.ndarray


@dataclass(frozen=True)
class SpinIntegerBridgeMode:
    """Posterior mass carried by one newly introduced island bridge."""

    bridge_integers: tuple[int, ...]
    log_posterior_weight: float
    posterior_weight: float
    integer_log_likelihood: float
    path_modes: tuple[SpinIntegerPathMode, ...]


@dataclass(frozen=True)
class SpinIntegerBridgeFamilyEvaluation:
    """Exact Gaussian-lattice mass for one fixed temporal bridge family."""

    log_likelihood: float
    shape_log_likelihood: float
    curvature_log_likelihood: float
    slope_log_likelihood: float
    integer_log_likelihood: float
    bridge_integers: tuple[int, ...]
    conditional_integer_center: np.ndarray
    conditional_integer_covariance: np.ndarray
    retained_lattice_candidates: int
    lattice_log_mass_increment: float
    lattice_mass_converged: bool
    lattice_visited_nodes: int


@dataclass(frozen=True)
class SpinIntegerBridgeFamilyPathMode:
    """One complete target path conditioned on a temporal bridge family."""

    bridge_integers: tuple[int, ...]
    source_integer_offsets: tuple[int, ...]
    target_integer_offsets: tuple[int, ...]
    conditional_squared_mahalanobis: float
    squared_mahalanobis: float
    log_posterior_weight: float
    posterior_weight: float


@dataclass(frozen=True)
class SpinIntegerMarginalEvaluation:
    log_likelihood: float
    shape_log_likelihood: float
    curvature_log_likelihood: float
    slope_log_likelihood: float
    integer_log_likelihood: float
    shape_quadratic: float
    shape_log_determinant: float
    spin_shape_posterior_mean: np.ndarray
    spin_shape_posterior_covariance: np.ndarray
    integer_center: np.ndarray
    integer_covariance: np.ndarray
    path_modes: tuple[SpinIntegerPathMode, ...]
    retained_lattice_candidates: int
    lattice_log_mass_increment: float
    lattice_mass_converged: bool
    lattice_visited_nodes: int


@dataclass(frozen=True)
class SpinFixedIntegerPathEvaluation:
    log_likelihood: float
    shape_log_likelihood: float
    curvature_log_likelihood: float
    slope_log_likelihood: float
    path_log_likelihood: float
    squared_mahalanobis: float
    spin_posterior_mean: np.ndarray
    spin_posterior_covariance: np.ndarray


def _positive_definite_matrix(
    values: Sequence[Sequence[float]],
    *,
    dimension: int,
    label: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{label} must have shape ({dimension}, {dimension}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be finite.")
    matrix = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} must be positive definite.") from exc
    return matrix


def _adaptive_integer_lattice(
    center: np.ndarray,
    covariance: Sequence[Sequence[float]],
    *,
    config: SpinIntegerMarginalConfig,
) -> tuple[IntegerLeastSquaresResult, float, bool]:
    requested = max(int(config.initial_lattice_candidates), 1)
    maximum = max(int(config.maximum_lattice_candidates), requested)
    tolerance = max(float(config.lattice_log_mass_tolerance), 0.0)
    previous_log_mass: float | None = None
    latest: IntegerLeastSquaresResult | None = None
    latest_increment = math.inf
    converged = False

    while True:
        try:
            result = solve_integer_least_squares(
                center,
                covariance,
                candidate_count=requested,
                max_visited_nodes=max(int(config.maximum_lattice_nodes), 1),
            )
        except RuntimeError:
            if latest is None:
                raise
            break
        latest = result
        distances = np.asarray(
            [candidate.squared_mahalanobis for candidate in result.candidates],
            dtype=float,
        )
        current_log_mass = float(logsumexp(-0.5 * distances))
        if previous_log_mass is not None:
            latest_increment = max(current_log_mass - previous_log_mass, 0.0)
            if latest_increment <= tolerance:
                converged = True
                break
        if requested >= maximum or len(result.candidates) < requested:
            break
        previous_log_mass = current_log_mass
        requested = min(2 * requested, maximum)

    if latest is None:
        raise RuntimeError("Integer lattice enumeration produced no result.")
    return latest, float(latest_increment), bool(converged)


def _predictive_covariance_log_determinant(
    *,
    noise_covariance: np.ndarray,
    ambiguity_jacobian: np.ndarray,
    parameter_covariance: np.ndarray,
) -> float:
    """Evaluate ``log det(N + B C B.T)`` through small stable systems."""

    sign_noise, logdet_noise = np.linalg.slogdet(noise_covariance)
    sign_parameter, logdet_parameter = np.linalg.slogdet(parameter_covariance)
    if sign_noise <= 0.0 or sign_parameter <= 0.0:
        raise ValueError("Predictive covariance components must be positive definite.")
    parameter_precision = np.linalg.inv(parameter_covariance)
    noise_weighted_jacobian = np.linalg.solve(
        noise_covariance,
        ambiguity_jacobian,
    )
    core = (
        parameter_precision
        + ambiguity_jacobian.T @ noise_weighted_jacobian
    )
    sign_core, logdet_core = np.linalg.slogdet(core)
    if sign_core <= 0.0:
        raise ValueError("Predictive covariance determinant core is not positive.")
    return float(logdet_noise + logdet_parameter + logdet_core)


def _shape_information_blocks(
    linearization: SpinOrbitLinearization,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return orthogonal curvature and slope observables for one orbit state.

    Older synthetic callers only provide the combined intercept-free shape
    residual. They remain a valid single-block Gaussian model. Real PINT
    linearizations provide both orthogonal blocks explicitly.
    """

    combined = np.asarray(linearization.profile.normalized_residuals, dtype=float)
    combined_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    if combined.ndim != 1 or combined_jacobian.ndim != 2:
        raise ValueError("Shape residuals and Jacobian must be vectors and matrices.")
    spin_dimension = int(combined_jacobian.shape[1])
    if combined_jacobian.shape[0] != combined.size:
        raise ValueError("Shape Jacobian must align with shape residuals.")

    curvature = np.asarray(
        linearization.profile.normalized_orthogonal_residuals,
        dtype=float,
    )
    curvature_jacobian = np.asarray(linearization.curvature_jacobian, dtype=float)
    slopes = np.asarray(
        linearization.profile.standardized_session_slopes,
        dtype=float,
    )
    slope_jacobian = np.asarray(
        linearization.standardized_slope_jacobian,
        dtype=float,
    )
    explicit_blocks = (
        curvature.shape == combined.shape
        and curvature_jacobian.shape == (curvature.size, spin_dimension)
        and slope_jacobian.shape == (slopes.size, spin_dimension)
    )
    if not explicit_blocks:
        return (
            combined,
            combined_jacobian,
            np.zeros(0, dtype=float),
            np.zeros((0, spin_dimension), dtype=float),
        )
    if not all(
        np.all(np.isfinite(values))
        for values in (curvature, curvature_jacobian, slopes, slope_jacobian)
    ):
        raise ValueError("Curvature and slope information blocks must be finite.")
    return curvature, curvature_jacobian, slopes, slope_jacobian


def _gaussian_linear_update(
    residuals: np.ndarray,
    jacobian: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Condition a Gaussian nuisance vector on one whitened linear block."""

    if residuals.ndim != 1 or jacobian.shape != (
        residuals.size,
        prior_mean.size,
    ):
        raise ValueError("A Gaussian information block has inconsistent dimensions.")
    if residuals.size == 0:
        return (
            0.0,
            0.0,
            0.0,
            prior_mean.copy(),
            prior_covariance.copy(),
        )
    prior_precision = np.linalg.inv(prior_covariance)
    posterior_precision = prior_precision + jacobian.T @ jacobian
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_covariance = 0.5 * (
        posterior_covariance + posterior_covariance.T
    )
    centered = residuals + jacobian @ prior_mean
    gradient = jacobian.T @ centered
    posterior_mean = prior_mean - posterior_covariance @ gradient
    quadratic = float(
        centered @ centered
        - gradient @ posterior_covariance @ gradient
    )
    prior_sign, prior_logdet = np.linalg.slogdet(prior_covariance)
    precision_sign, precision_logdet = np.linalg.slogdet(posterior_precision)
    if prior_sign <= 0.0 or precision_sign <= 0.0:
        raise ValueError("Gaussian information update produced an invalid determinant.")
    log_determinant = float(prior_logdet + precision_logdet)
    log_likelihood = float(-0.5 * (quadratic + log_determinant))
    return (
        log_likelihood,
        max(quadratic, 0.0),
        log_determinant,
        posterior_mean,
        posterior_covariance,
    )


def _condition_spin_on_shape_information(
    linearization: SpinOrbitLinearization,
    *,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    curvature, curvature_jacobian, slopes, slope_jacobian = (
        _shape_information_blocks(linearization)
    )
    (
        curvature_log_likelihood,
        curvature_quadratic,
        curvature_log_determinant,
        curvature_mean,
        curvature_covariance,
    ) = _gaussian_linear_update(
        curvature,
        curvature_jacobian,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )
    (
        slope_log_likelihood,
        slope_quadratic,
        slope_log_determinant,
        shape_mean,
        shape_covariance,
    ) = _gaussian_linear_update(
        slopes,
        slope_jacobian,
        prior_mean=curvature_mean,
        prior_covariance=curvature_covariance,
    )
    return (
        curvature_log_likelihood,
        slope_log_likelihood,
        curvature_quadratic + slope_quadratic,
        curvature_log_determinant + slope_log_determinant,
        shape_mean,
        shape_covariance,
    )


def infer_temporal_phase_islands(
    session_epochs: Sequence[float],
    *,
    orbital_period_days: float,
    cadence_gap_factor: float = 4.0,
    orbital_cycle_factor: float = 8.0,
) -> tuple[tuple[int, ...], ...]:
    """Split ordered sessions only at cadence-scale temporal discontinuities."""

    epochs = np.asarray(session_epochs, dtype=float)
    if epochs.ndim != 1 or epochs.size < 2 or not np.all(np.isfinite(epochs)):
        raise ValueError("Temporal phase-island epochs must be a finite vector.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError("Temporal phase-island epochs must be strictly increasing.")
    period = float(orbital_period_days)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("The orbital period used for phase islands must be positive.")
    cadence_factor = max(float(cadence_gap_factor), 1.0)
    cycle_factor = max(float(orbital_cycle_factor), 1.0)
    gaps = np.diff(epochs)
    typical_gap = float(np.median(gaps))
    threshold = max(cadence_factor * typical_gap, cycle_factor * period)
    split_after = set(int(index) for index in np.flatnonzero(gaps > threshold))
    if not split_after:
        return (tuple(range(epochs.size)),)

    partitions = []
    start = 0
    for gap_index in sorted(split_after):
        stop = gap_index + 1
        partitions.append(tuple(range(start, stop)))
        start = stop
    partitions.append(tuple(range(start, epochs.size)))
    return tuple(partitions)


def infer_temporal_phase_island_hierarchy(
    session_epochs: Sequence[float],
    *,
    orbital_period_days: float,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Build nested phase islands by adding gap edges on dyadic orbit scales."""

    epochs = np.asarray(session_epochs, dtype=float)
    if epochs.ndim != 1 or epochs.size < 2 or not np.all(np.isfinite(epochs)):
        raise ValueError("Temporal phase-island epochs must be a finite vector.")
    gaps = np.diff(epochs)
    if np.any(gaps <= 0.0):
        raise ValueError("Temporal phase-island epochs must be strictly increasing.")
    period = float(orbital_period_days)
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("The orbital period used for phase islands must be positive.")

    gap_levels = np.ceil(np.log2(gaps / period)).astype(int)
    hierarchy = []
    for level in sorted(set(int(value) for value in gap_levels)):
        partitions = []
        start = 0
        for gap_index, gap_level in enumerate(gap_levels):
            if int(gap_level) > level:
                stop = gap_index + 1
                partitions.append(tuple(range(start, stop)))
                start = stop
        partitions.append(tuple(range(start, epochs.size)))
        hierarchy.append(tuple(partitions))
    return tuple(hierarchy)


def temporal_phase_island_incidence(
    session_count: int,
    session_partitions: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return an integer basis for phase differences internal to each island."""

    count = int(session_count)
    if count < 2:
        raise ValueError("Phase-island inference needs at least two sessions.")
    partitions = tuple(tuple(int(value) for value in block) for block in session_partitions)
    if not partitions or any(not block for block in partitions):
        raise ValueError("Every temporal phase island must contain a session.")
    flattened = tuple(value for block in partitions for value in block)
    if sorted(flattened) != list(range(count)) or len(set(flattened)) != count:
        raise ValueError("Temporal phase islands must partition every session exactly once.")
    if any(tuple(sorted(block)) != block for block in partitions):
        raise ValueError("Sessions inside a temporal phase island must be ordered.")

    absolute_to_relative = np.zeros((count, count - 1), dtype=float)
    absolute_to_relative[1:, :] = np.eye(count - 1, dtype=float)
    rows = []
    for block in partitions:
        anchor = block[0]
        for session in block[1:]:
            rows.append(
                absolute_to_relative[session] - absolute_to_relative[anchor]
            )
    return np.asarray(rows, dtype=float).reshape(-1, count - 1)


def temporal_partition_bridge_signature(
    *,
    session_count: int,
    source_partitions: Sequence[Sequence[int]],
    target_partitions: Sequence[Sequence[int]],
    target_integer_offsets: Sequence[int],
) -> tuple[int, ...]:
    """Project one target path onto only the gauges added by a refinement merge.

    ``source_partitions`` must refine ``target_partitions``. Integer offsets use
    the same anchor-relative ordering as :func:`temporal_phase_island_incidence`.
    A target block containing ``r`` source islands contributes ``r - 1`` gauge
    integers, all measured from the first source-island anchor. The result is
    therefore independent of integer choices internal to the source islands.
    """

    count = int(session_count)
    source = tuple(tuple(int(value) for value in block) for block in source_partitions)
    target = tuple(tuple(int(value) for value in block) for block in target_partitions)
    for label, partitions in (("source", source), ("target", target)):
        flattened = tuple(value for block in partitions for value in block)
        if (
            not partitions
            or any(not block for block in partitions)
            or sorted(flattened) != list(range(count))
            or len(set(flattened)) != count
            or any(tuple(sorted(block)) != block for block in partitions)
        ):
            raise ValueError(
                f"The {label} temporal partition must cover every session in order."
            )

    target_dimension = count - len(target)
    integers = np.asarray(target_integer_offsets, dtype=np.int64)
    if integers.shape != (target_dimension,):
        raise ValueError("Target integer offsets do not match the target partition.")

    local_offsets = np.zeros(count, dtype=np.int64)
    cursor = 0
    for block in target:
        width = len(block) - 1
        if width:
            local_offsets[np.asarray(block[1:], dtype=int)] = integers[
                cursor : cursor + width
            ]
        cursor += width

    source_by_session: dict[int, int] = {}
    for source_index, block in enumerate(source):
        for session in block:
            source_by_session[session] = source_index

    signature: list[int] = []
    consumed_source_blocks: set[int] = set()
    for target_block in target:
        source_indices: list[int] = []
        for session in target_block:
            source_index = source_by_session[session]
            if not source_indices or source_indices[-1] != source_index:
                source_indices.append(source_index)
        reconstructed = tuple(
            session
            for source_index in source_indices
            for session in source[source_index]
        )
        if reconstructed != target_block:
            raise ValueError("The source temporal partition must refine the target.")
        if any(source_index in consumed_source_blocks for source_index in source_indices):
            raise ValueError("A source island cannot contribute to two target islands.")
        consumed_source_blocks.update(source_indices)
        reference_anchor = source[source_indices[0]][0]
        for source_index in source_indices[1:]:
            anchor = source[source_index][0]
            signature.append(
                int(local_offsets[anchor] - local_offsets[reference_anchor])
            )
    if len(consumed_source_blocks) != len(source):
        raise ValueError("The source temporal partition must refine the target.")
    if len(signature) != len(source) - len(target):
        raise RuntimeError("Temporal bridge signature has the wrong dimension.")
    return tuple(signature)


def temporal_partition_bridge_matrix(
    *,
    session_count: int,
    source_partitions: Sequence[Sequence[int]],
    target_partitions: Sequence[Sequence[int]],
) -> np.ndarray:
    """Return the integer linear map from target paths to bridge gauges."""

    target_dimension = int(session_count) - len(tuple(target_partitions))
    zero = np.zeros(target_dimension, dtype=np.int64)
    zero_signature = temporal_partition_bridge_signature(
        session_count=session_count,
        source_partitions=source_partitions,
        target_partitions=target_partitions,
        target_integer_offsets=zero,
    )
    matrix = np.zeros((len(zero_signature), target_dimension), dtype=float)
    for column in range(target_dimension):
        basis = zero.copy()
        basis[column] = 1
        matrix[:, column] = np.asarray(
            temporal_partition_bridge_signature(
                session_count=session_count,
                source_partitions=source_partitions,
                target_partitions=target_partitions,
                target_integer_offsets=basis,
            ),
            dtype=float,
        )
    return matrix


def temporal_partition_integer_transform(
    *,
    session_count: int,
    source_partitions: Sequence[Sequence[int]],
    target_partitions: Sequence[Sequence[int]],
) -> np.ndarray:
    """Map target integers to source-internal integers followed by bridges."""

    source_incidence = temporal_phase_island_incidence(
        session_count,
        source_partitions,
    )
    target_incidence = temporal_phase_island_incidence(
        session_count,
        target_partitions,
    )
    source_from_target = (
        source_incidence
        @ target_incidence.T
        @ np.linalg.inv(target_incidence @ target_incidence.T)
    )
    rounded_source = np.rint(source_from_target)
    if not np.allclose(source_from_target, rounded_source, atol=1.0e-10):
        raise ValueError("Temporal source integers do not embed in the target lattice.")
    if not np.allclose(
        rounded_source @ target_incidence,
        source_incidence,
        atol=1.0e-10,
    ):
        raise ValueError("Temporal integer coordinate transform is inconsistent.")
    bridge = temporal_partition_bridge_matrix(
        session_count=session_count,
        source_partitions=source_partitions,
        target_partitions=target_partitions,
    )
    transform = np.vstack((rounded_source, bridge))
    if transform.shape[0] != transform.shape[1]:
        raise ValueError("Temporal integer coordinate transform must be square.")
    determinant = float(np.linalg.det(transform))
    if not np.isclose(abs(determinant), 1.0, atol=1.0e-8):
        raise ValueError("Temporal integer coordinate transform must be unimodular.")
    return np.asarray(np.rint(transform), dtype=np.int64)


def group_temporal_partition_bridge_modes(
    evaluation: SpinIntegerMarginalEvaluation,
    *,
    session_count: int,
    source_partitions: Sequence[Sequence[int]],
    target_partitions: Sequence[Sequence[int]],
) -> tuple[SpinIntegerBridgeMode, ...]:
    """Decompose a target marginal likelihood by newly added bridge integers."""

    grouped: dict[tuple[int, ...], list[SpinIntegerPathMode]] = {}
    for path_mode in evaluation.path_modes:
        signature = temporal_partition_bridge_signature(
            session_count=session_count,
            source_partitions=source_partitions,
            target_partitions=target_partitions,
            target_integer_offsets=path_mode.integer_offsets,
        )
        grouped.setdefault(signature, []).append(path_mode)
    if not grouped:
        return ()

    modes = []
    for signature, path_modes in grouped.items():
        log_weight = float(
            logsumexp(
                np.asarray(
                    [mode.log_posterior_weight for mode in path_modes],
                    dtype=float,
                )
            )
        )
        modes.append(
            SpinIntegerBridgeMode(
                bridge_integers=signature,
                log_posterior_weight=log_weight,
                posterior_weight=float(math.exp(log_weight)),
                integer_log_likelihood=float(
                    evaluation.integer_log_likelihood + log_weight
                ),
                path_modes=tuple(path_modes),
            )
        )
    modes.sort(
        key=lambda mode: (
            -mode.log_posterior_weight,
            mode.bridge_integers,
        )
    )
    return tuple(modes)


def restrict_linearization_to_phase_islands(
    linearization: SpinOrbitLinearization,
    session_partitions: Sequence[Sequence[int]],
) -> SpinOrbitLinearization:
    """Remove only island-to-island integer phase constraints."""

    relative = np.asarray(linearization.relative_session_offsets, dtype=float)
    relative_jacobian = np.asarray(
        linearization.relative_session_jacobian,
        dtype=float,
    )
    relative_covariance = np.asarray(
        linearization.relative_offset_covariance,
        dtype=float,
    )
    session_count = relative.size + 1
    incidence = temporal_phase_island_incidence(
        session_count,
        session_partitions,
    )
    if incidence.shape[0] == 0:
        raise ValueError("A phase-island partition must retain an internal phase edge.")
    return replace(
        linearization,
        relative_session_offsets=incidence @ relative,
        relative_session_jacobian=incidence @ relative_jacobian,
        relative_offset_covariance=(
            incidence @ relative_covariance @ incidence.T
        ),
    )


def marginalize_spin_and_integer_paths(
    linearization: SpinOrbitLinearization,
    *,
    spin_prior_mean: Sequence[float],
    spin_prior_covariance: Sequence[Sequence[float]],
    config: SpinIntegerMarginalConfig | None = None,
) -> SpinIntegerMarginalEvaluation:
    """Integrate a linear spin trend and sum retained complete integer paths.

    For a fixed orbit, ``z`` denotes the spin displacement from the
    linearization point. The model is

        y + A z = epsilon_shape
        d + B z - n = epsilon_phase,  n in Z^(S-1).

    A proper Gaussian prior on ``z`` is integrated analytically. The remaining
    correlated integer Gaussian is enumerated without selecting a path first.
    """

    active = config or SpinIntegerMarginalConfig()
    shape = np.asarray(linearization.profile.normalized_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    relative = np.asarray(linearization.relative_session_offsets, dtype=float)
    relative_jacobian = np.asarray(
        linearization.relative_session_jacobian,
        dtype=float,
    )
    if shape.ndim != 1 or not np.all(np.isfinite(shape)):
        raise ValueError("Shape residuals must be a finite vector.")
    if shape_jacobian.ndim != 2 or shape_jacobian.shape[0] != shape.size:
        raise ValueError("Shape Jacobian must align with shape residuals.")
    spin_dimension = int(shape_jacobian.shape[1])
    if spin_dimension == 0:
        raise ValueError("At least one spin nuisance coordinate is required.")
    if relative.ndim != 1 or relative_jacobian.shape != (
        relative.size,
        spin_dimension,
    ):
        raise ValueError("Relative phase data and spin Jacobian must align.")
    if not np.all(np.isfinite(shape_jacobian)) or not np.all(
        np.isfinite(relative) & np.all(np.isfinite(relative_jacobian), axis=1)
    ):
        raise ValueError("Spin linearization arrays must be finite.")

    prior_mean = np.asarray(spin_prior_mean, dtype=float)
    if prior_mean.shape != (spin_dimension,) or not np.all(np.isfinite(prior_mean)):
        raise ValueError("Spin prior mean must match the nuisance dimension.")
    prior_covariance = _positive_definite_matrix(
        spin_prior_covariance,
        dimension=spin_dimension,
        label="Spin prior covariance",
    )
    phase_covariance = _positive_definite_matrix(
        linearization.relative_offset_covariance,
        dimension=relative.size,
        label="Relative phase covariance",
    )
    added_sigma = float(active.added_session_phase_sigma_cycles)
    if not np.isfinite(added_sigma) or added_sigma < 0.0:
        raise ValueError("Added session phase sigma must be finite and non-negative.")
    if added_sigma > 0.0:
        phase_covariance = phase_covariance + added_sigma * added_sigma * (
            np.eye(relative.size, dtype=float)
            + np.ones((relative.size, relative.size), dtype=float)
        )

    (
        curvature_log_likelihood,
        slope_log_likelihood,
        shape_quadratic,
        shape_log_determinant,
        spin_shape_posterior_mean,
        shape_posterior_covariance,
    ) = _condition_spin_on_shape_information(
        linearization,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )
    shape_log_likelihood = float(
        curvature_log_likelihood + slope_log_likelihood
    )

    integer_center = (
        relative + relative_jacobian @ spin_shape_posterior_mean
    )
    integer_covariance = (
        phase_covariance
        + relative_jacobian
        @ shape_posterior_covariance
        @ relative_jacobian.T
    )
    integer_covariance = 0.5 * (integer_covariance + integer_covariance.T)
    integer_enumeration_covariance = build_predictive_integer_covariance(
        relative_jacobian,
        shape_posterior_covariance,
        phase_covariance,
    )
    lattice, log_mass_increment, mass_converged = _adaptive_integer_lattice(
        integer_center,
        integer_enumeration_covariance,
        config=active,
    )
    distances = np.asarray(
        [candidate.squared_mahalanobis for candidate in lattice.candidates],
        dtype=float,
    )
    raw_log_weights = -0.5 * distances
    lattice_log_mass = float(logsumexp(raw_log_weights))
    normalized_log_weights = raw_log_weights - lattice_log_mass
    covariance_logdet = _predictive_covariance_log_determinant(
        noise_covariance=phase_covariance,
        ambiguity_jacobian=relative_jacobian,
        parameter_covariance=shape_posterior_covariance,
    )
    integer_log_likelihood = float(
        lattice_log_mass - 0.5 * covariance_logdet
    )

    phase_precision = np.linalg.inv(phase_covariance)
    shape_posterior_precision = np.linalg.inv(shape_posterior_covariance)
    conditioned_spin_precision = (
        shape_posterior_precision
        + relative_jacobian.T @ phase_precision @ relative_jacobian
    )
    conditioned_spin_covariance = np.linalg.inv(conditioned_spin_precision)
    conditioned_spin_covariance = 0.5 * (
        conditioned_spin_covariance + conditioned_spin_covariance.T
    )
    conditioned_base_gradient = (
        shape_posterior_precision @ spin_shape_posterior_mean
        - relative_jacobian.T @ phase_precision @ relative
    )
    path_modes = []
    for candidate, log_weight in zip(
        lattice.candidates,
        normalized_log_weights,
        strict=True,
    ):
        integers = np.asarray(candidate.integer_vector, dtype=float)
        spin_mean = conditioned_spin_covariance @ (
            conditioned_base_gradient
            + relative_jacobian.T @ phase_precision @ integers
        )
        path_modes.append(
            SpinIntegerPathMode(
                integer_offsets=tuple(
                    int(value) for value in candidate.integer_vector
                ),
                squared_mahalanobis=float(candidate.squared_mahalanobis),
                log_posterior_weight=float(log_weight),
                posterior_weight=float(math.exp(float(log_weight))),
                spin_posterior_mean=spin_mean,
                spin_posterior_covariance=conditioned_spin_covariance.copy(),
            )
        )

    return SpinIntegerMarginalEvaluation(
        log_likelihood=float(shape_log_likelihood + integer_log_likelihood),
        shape_log_likelihood=shape_log_likelihood,
        curvature_log_likelihood=curvature_log_likelihood,
        slope_log_likelihood=slope_log_likelihood,
        integer_log_likelihood=integer_log_likelihood,
        shape_quadratic=max(shape_quadratic, 0.0),
        shape_log_determinant=shape_log_determinant,
        spin_shape_posterior_mean=spin_shape_posterior_mean,
        spin_shape_posterior_covariance=shape_posterior_covariance,
        integer_center=integer_center,
        integer_covariance=integer_covariance,
        path_modes=tuple(path_modes),
        retained_lattice_candidates=len(path_modes),
        lattice_log_mass_increment=log_mass_increment,
        lattice_mass_converged=mass_converged,
        lattice_visited_nodes=int(lattice.visited_nodes),
    )


def marginalize_spin_for_fixed_integer_path(
    linearization: SpinOrbitLinearization,
    integer_offsets: Sequence[int],
    *,
    spin_prior_mean: Sequence[float],
    spin_prior_covariance: Sequence[Sequence[float]],
    added_session_phase_sigma_cycles: float = 0.0,
) -> SpinFixedIntegerPathEvaluation:
    """Integrate spin while retaining one complete all-session path component."""

    shape = np.asarray(linearization.profile.normalized_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    relative = np.asarray(linearization.relative_session_offsets, dtype=float)
    relative_jacobian = np.asarray(
        linearization.relative_session_jacobian,
        dtype=float,
    )
    spin_dimension = int(shape_jacobian.shape[1])
    if shape_jacobian.shape[0] != shape.size or relative_jacobian.shape != (
        relative.size,
        spin_dimension,
    ):
        raise ValueError("Fixed-path spin arrays do not share one geometry.")
    path = np.asarray(integer_offsets, dtype=float)
    if path.shape != relative.shape or not np.all(np.isfinite(path)):
        raise ValueError("The fixed integer path must match every relative session.")
    prior_mean = np.asarray(spin_prior_mean, dtype=float)
    if prior_mean.shape != (spin_dimension,) or not np.all(np.isfinite(prior_mean)):
        raise ValueError("Spin prior mean must match the nuisance dimension.")
    prior_covariance = _positive_definite_matrix(
        spin_prior_covariance,
        dimension=spin_dimension,
        label="Spin prior covariance",
    )
    phase_covariance = _positive_definite_matrix(
        linearization.relative_offset_covariance,
        dimension=relative.size,
        label="Relative phase covariance",
    )
    added_sigma = float(added_session_phase_sigma_cycles)
    if not np.isfinite(added_sigma) or added_sigma < 0.0:
        raise ValueError("Added session phase sigma must be finite and non-negative.")
    if added_sigma > 0.0:
        phase_covariance = phase_covariance + added_sigma * added_sigma * (
            np.eye(relative.size, dtype=float)
            + np.ones((relative.size, relative.size), dtype=float)
        )

    (
        curvature_log_likelihood,
        slope_log_likelihood,
        _,
        _,
        spin_shape_posterior_mean,
        shape_posterior_covariance,
    ) = _condition_spin_on_shape_information(
        linearization,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )
    shape_log_likelihood = float(
        curvature_log_likelihood + slope_log_likelihood
    )

    integer_center = relative + relative_jacobian @ spin_shape_posterior_mean
    integer_covariance = (
        phase_covariance
        + relative_jacobian
        @ shape_posterior_covariance
        @ relative_jacobian.T
    )
    integer_covariance = 0.5 * (integer_covariance + integer_covariance.T)
    sign, logdet = np.linalg.slogdet(integer_covariance)
    if sign <= 0.0:
        raise ValueError("Fixed-path predictive covariance must be positive definite.")
    innovation = path - integer_center
    squared_mahalanobis = float(
        innovation @ np.linalg.solve(integer_covariance, innovation)
    )
    path_log_likelihood = float(-0.5 * (squared_mahalanobis + logdet))

    phase_precision = np.linalg.inv(phase_covariance)
    shape_posterior_precision = np.linalg.inv(shape_posterior_covariance)
    conditioned_spin_precision = (
        shape_posterior_precision
        + relative_jacobian.T @ phase_precision @ relative_jacobian
    )
    conditioned_spin_covariance = np.linalg.inv(conditioned_spin_precision)
    conditioned_spin_covariance = 0.5 * (
        conditioned_spin_covariance + conditioned_spin_covariance.T
    )
    conditioned_base_gradient = (
        shape_posterior_precision @ spin_shape_posterior_mean
        - relative_jacobian.T @ phase_precision @ relative
    )
    spin_posterior_mean = conditioned_spin_covariance @ (
        conditioned_base_gradient
        + relative_jacobian.T @ phase_precision @ path
    )
    return SpinFixedIntegerPathEvaluation(
        log_likelihood=float(shape_log_likelihood + path_log_likelihood),
        shape_log_likelihood=shape_log_likelihood,
        curvature_log_likelihood=curvature_log_likelihood,
        slope_log_likelihood=slope_log_likelihood,
        path_log_likelihood=path_log_likelihood,
        squared_mahalanobis=max(squared_mahalanobis, 0.0),
        spin_posterior_mean=spin_posterior_mean,
        spin_posterior_covariance=conditioned_spin_covariance,
    )


def marginalize_temporal_bridge_family(
    evaluation: SpinIntegerMarginalEvaluation,
    *,
    session_count: int,
    source_partitions: Sequence[Sequence[int]],
    target_partitions: Sequence[Sequence[int]],
    bridge_integers: Sequence[int],
    config: SpinIntegerMarginalConfig | None = None,
) -> SpinIntegerBridgeFamilyEvaluation:
    """Marginalize source-internal paths with one bridge signature fixed.

    The target integer Gaussian is transformed into a unimodular coordinate
    system ``[source internal integers, new bridge integers]``. Conditioning
    the bridge block leaves a lower-dimensional integer lattice that can be
    summed directly, even when this family is absent from a truncated list of
    target path modes.
    """

    active = config or SpinIntegerMarginalConfig()
    transform = temporal_partition_integer_transform(
        session_count=session_count,
        source_partitions=source_partitions,
        target_partitions=target_partitions,
    ).astype(float)
    center = transform @ np.asarray(evaluation.integer_center, dtype=float)
    covariance = (
        transform
        @ np.asarray(evaluation.integer_covariance, dtype=float)
        @ transform.T
    )
    covariance = 0.5 * (covariance + covariance.T)
    signature = np.asarray(bridge_integers, dtype=float)
    bridge_dimension = len(tuple(bridge_integers))
    if bridge_dimension <= 0 or signature.shape != (bridge_dimension,):
        raise ValueError("A fixed temporal bridge family requires an integer vector.")
    if bridge_dimension >= center.size + 1:
        raise ValueError("Temporal bridge dimension exceeds the target lattice.")
    source_dimension = center.size - bridge_dimension
    bridge_center = center[source_dimension:]
    bridge_covariance = covariance[source_dimension:, source_dimension:]
    bridge_covariance = _positive_definite_matrix(
        bridge_covariance,
        dimension=bridge_dimension,
        label="Temporal bridge covariance",
    )
    bridge_innovation = signature - bridge_center
    bridge_sign, bridge_logdet = np.linalg.slogdet(bridge_covariance)
    if bridge_sign <= 0.0:
        raise ValueError("Temporal bridge covariance must be positive definite.")
    bridge_quadratic = float(
        bridge_innovation
        @ np.linalg.solve(bridge_covariance, bridge_innovation)
    )
    bridge_log_likelihood = float(
        -0.5 * (bridge_quadratic + bridge_logdet)
    )

    if source_dimension == 0:
        conditional_center = np.zeros(0, dtype=float)
        conditional_covariance = np.zeros((0, 0), dtype=float)
        internal_log_likelihood = 0.0
        retained_candidates = 1
        log_mass_increment = 0.0
        mass_converged = True
        visited_nodes = 1
    else:
        source_center = center[:source_dimension]
        source_covariance = covariance[:source_dimension, :source_dimension]
        cross_covariance = covariance[:source_dimension, source_dimension:]
        gain = cross_covariance @ np.linalg.inv(bridge_covariance)
        conditional_center = source_center + gain @ bridge_innovation
        conditional_covariance = (
            source_covariance - gain @ cross_covariance.T
        )
        conditional_covariance = _positive_definite_matrix(
            conditional_covariance,
            dimension=source_dimension,
            label="Conditional source-integer covariance",
        )
        lattice, log_mass_increment, mass_converged = _adaptive_integer_lattice(
            conditional_center,
            conditional_covariance,
            config=active,
        )
        distances = np.asarray(
            [candidate.squared_mahalanobis for candidate in lattice.candidates],
            dtype=float,
        )
        conditional_sign, conditional_logdet = np.linalg.slogdet(
            conditional_covariance
        )
        if conditional_sign <= 0.0:
            raise ValueError(
                "Conditional source-integer covariance must be positive definite."
            )
        internal_log_likelihood = float(
            logsumexp(-0.5 * distances) - 0.5 * conditional_logdet
        )
        retained_candidates = len(lattice.candidates)
        visited_nodes = int(lattice.visited_nodes)

    integer_log_likelihood = float(
        bridge_log_likelihood + internal_log_likelihood
    )
    return SpinIntegerBridgeFamilyEvaluation(
        log_likelihood=float(
            evaluation.shape_log_likelihood + integer_log_likelihood
        ),
        shape_log_likelihood=float(evaluation.shape_log_likelihood),
        curvature_log_likelihood=float(evaluation.curvature_log_likelihood),
        slope_log_likelihood=float(evaluation.slope_log_likelihood),
        integer_log_likelihood=integer_log_likelihood,
        bridge_integers=tuple(int(value) for value in signature),
        conditional_integer_center=conditional_center,
        conditional_integer_covariance=conditional_covariance,
        retained_lattice_candidates=retained_candidates,
        lattice_log_mass_increment=float(log_mass_increment),
        lattice_mass_converged=bool(mass_converged),
        lattice_visited_nodes=visited_nodes,
    )


def enumerate_temporal_bridge_family_paths(
    evaluation: SpinIntegerMarginalEvaluation,
    *,
    session_count: int,
    source_partitions: Sequence[Sequence[int]],
    target_partitions: Sequence[Sequence[int]],
    bridge_integers: Sequence[int],
    config: SpinIntegerMarginalConfig | None = None,
) -> tuple[SpinIntegerBridgeFamilyPathMode, ...]:
    """Recover complete target paths while fixing only new bridge gauges.

    Unlike grouping ``evaluation.path_modes``, this function enumerates the
    source-internal lattice after conditioning on the requested bridge. A
    bridge family therefore remains recoverable even when no member appeared
    in a truncated enumeration of the complete target lattice.
    """

    active = config or SpinIntegerMarginalConfig()
    transform_integer = temporal_partition_integer_transform(
        session_count=session_count,
        source_partitions=source_partitions,
        target_partitions=target_partitions,
    )
    transform = transform_integer.astype(float)
    target_center = np.asarray(evaluation.integer_center, dtype=float)
    target_covariance = np.asarray(evaluation.integer_covariance, dtype=float)
    center = transform @ target_center
    covariance = transform @ target_covariance @ transform.T
    covariance = 0.5 * (covariance + covariance.T)

    signature = np.asarray(bridge_integers, dtype=float)
    bridge_dimension = signature.size
    if bridge_dimension <= 0:
        raise ValueError("A temporal bridge family requires at least one integer.")
    if bridge_dimension >= center.size + 1:
        raise ValueError("Temporal bridge dimension exceeds the target lattice.")
    source_dimension = center.size - bridge_dimension
    bridge_center = center[source_dimension:]
    bridge_covariance = _positive_definite_matrix(
        covariance[source_dimension:, source_dimension:],
        dimension=bridge_dimension,
        label="Temporal bridge covariance",
    )
    bridge_innovation = signature - bridge_center
    bridge_quadratic = float(
        bridge_innovation
        @ np.linalg.solve(bridge_covariance, bridge_innovation)
    )

    if source_dimension == 0:
        source_candidates = (((), 0.0),)
    else:
        source_center = center[:source_dimension]
        source_covariance = covariance[:source_dimension, :source_dimension]
        cross_covariance = covariance[:source_dimension, source_dimension:]
        gain = cross_covariance @ np.linalg.inv(bridge_covariance)
        conditional_center = source_center + gain @ bridge_innovation
        conditional_covariance = _positive_definite_matrix(
            source_covariance - gain @ cross_covariance.T,
            dimension=source_dimension,
            label="Conditional source-integer covariance",
        )
        lattice, _, _ = _adaptive_integer_lattice(
            conditional_center,
            conditional_covariance,
            config=active,
        )
        source_candidates = tuple(
            (
                tuple(int(value) for value in candidate.integer_vector),
                float(candidate.squared_mahalanobis),
            )
            for candidate in lattice.candidates
        )

    inverse_float = np.linalg.inv(transform)
    inverse_integer = np.rint(inverse_float).astype(np.int64)
    if not np.allclose(inverse_float, inverse_integer, atol=1.0e-10):
        raise RuntimeError("Temporal integer transform has no integer inverse.")
    raw_log_weights = np.asarray(
        [-0.5 * distance for _, distance in source_candidates],
        dtype=float,
    )
    normalized_log_weights = raw_log_weights - logsumexp(raw_log_weights)
    modes = []
    signature_integer = tuple(int(value) for value in signature)
    for (source_offsets, conditional_distance), log_weight in zip(
        source_candidates,
        normalized_log_weights,
        strict=True,
    ):
        transformed_offsets = np.asarray(
            source_offsets + signature_integer,
            dtype=np.int64,
        )
        target_offsets = inverse_integer @ transformed_offsets
        if not np.array_equal(
            transform_integer @ target_offsets,
            transformed_offsets,
        ):
            raise RuntimeError("A conditioned bridge path failed integer recovery.")
        modes.append(
            SpinIntegerBridgeFamilyPathMode(
                bridge_integers=signature_integer,
                source_integer_offsets=source_offsets,
                target_integer_offsets=tuple(int(value) for value in target_offsets),
                conditional_squared_mahalanobis=conditional_distance,
                squared_mahalanobis=bridge_quadratic + conditional_distance,
                log_posterior_weight=float(log_weight),
                posterior_weight=float(math.exp(float(log_weight))),
            )
        )
    return tuple(modes)


@dataclass(frozen=True)
class SpinIntegerMarginalizedOrbitLikelihood:
    """All-session target over nonlinear orbit coordinates only."""

    evaluator: Any
    coordinates: Any
    orbit_parameter_names: tuple[str, ...]
    spin_parameter_names: tuple[str, ...]
    reference_normalized_values: np.ndarray
    spin_prior_mean: np.ndarray
    spin_prior_covariance: np.ndarray
    spin_finite_difference_steps: np.ndarray
    config: SpinIntegerMarginalConfig
    orbit_support_lower_values: np.ndarray | None = None
    orbit_support_upper_values: np.ndarray | None = None
    session_partitions: tuple[tuple[int, ...], ...] | None = None
    session_partition_hierarchy: (
        tuple[tuple[tuple[int, ...], ...], ...] | None
    ) = None

    def _indices(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        names = tuple(str(name).upper() for name in self.coordinates.parameter_names)
        orbit = tuple(names.index(str(name).upper()) for name in self.orbit_parameter_names)
        spin = tuple(names.index(str(name).upper()) for name in self.spin_parameter_names)
        if set(orbit) & set(spin) or len(orbit) + len(spin) != len(names):
            raise ValueError("Orbit and spin coordinates must form a disjoint full partition.")
        return orbit, spin

    @property
    def orbit_lower(self) -> np.ndarray:
        orbit, _ = self._indices()
        global_lower = np.asarray(
            self.coordinates.normalized_lower,
            dtype=float,
        )[list(orbit)]
        if self.orbit_support_lower_values is None:
            return global_lower
        local_lower = np.asarray(self.orbit_support_lower_values, dtype=float)
        if local_lower.shape != global_lower.shape:
            raise ValueError("Local orbit lower support has the wrong dimension.")
        return np.maximum(local_lower, global_lower)

    @property
    def orbit_upper(self) -> np.ndarray:
        orbit, _ = self._indices()
        global_upper = np.asarray(
            self.coordinates.normalized_upper,
            dtype=float,
        )[list(orbit)]
        if self.orbit_support_upper_values is None:
            return global_upper
        local_upper = np.asarray(self.orbit_support_upper_values, dtype=float)
        if local_upper.shape != global_upper.shape:
            raise ValueError("Local orbit upper support has the wrong dimension.")
        return np.minimum(local_upper, global_upper)

    def linearize(self, orbit_normalized_values: Sequence[float]) -> SpinOrbitLinearization:
        orbit_values = np.asarray(orbit_normalized_values, dtype=float)
        orbit_indices, spin_indices = self._indices()
        if orbit_values.shape != (len(orbit_indices),):
            raise ValueError("Orbit values do not match the orbit dimension.")
        if not np.all(np.isfinite(orbit_values)):
            raise ValueError("Orbit values must be finite.")
        if np.any(orbit_values <= self.orbit_lower) or np.any(
            orbit_values >= self.orbit_upper
        ):
            raise ValueError("Orbit values lie outside the declared support.")

        full = np.asarray(self.reference_normalized_values, dtype=float).copy()
        if full.shape != (len(self.coordinates.parameter_names),):
            raise ValueError("Reference values do not match the full coordinate dimension.")
        full[list(orbit_indices)] = orbit_values
        profile = self.evaluator.evaluate_values(_physical_values(self.coordinates, full))
        steps = np.asarray(self.spin_finite_difference_steps, dtype=float)
        if steps.shape != (len(spin_indices),) or not np.all(
            np.isfinite(steps) & (steps > 0.0)
        ):
            raise ValueError("Spin finite-difference steps must be finite and positive.")

        shape_jacobian = np.empty(
            (profile.normalized_residuals.size, len(spin_indices)),
            dtype=float,
        )
        curvature_jacobian = np.empty(
            (profile.normalized_orthogonal_residuals.size, len(spin_indices)),
            dtype=float,
        )
        observable_slopes = np.asarray(
            profile.session_slope_observable,
            dtype=bool,
        )
        standardized_slope_jacobian = np.empty(
            (int(np.count_nonzero(observable_slopes)), len(spin_indices)),
            dtype=float,
        )
        offset_jacobian = np.empty(
            (profile.session_count, len(spin_indices)),
            dtype=float,
        )

        def standardized_slopes(candidate: SessionShapeProfile) -> np.ndarray:
            candidate_observable = np.asarray(
                candidate.session_slope_observable,
                dtype=bool,
            )
            if not np.array_equal(candidate_observable, observable_slopes):
                raise ValueError("Spin probes changed session slope observability.")
            slopes = np.asarray(candidate.session_slopes, dtype=float)[
                observable_slopes
            ]
            variances = np.asarray(candidate.session_slope_variances, dtype=float)[
                observable_slopes
            ]
            if not np.all(
                np.isfinite(slopes)
                & np.isfinite(variances)
                & (variances > 0.0)
            ):
                raise ValueError("Spin probes produced invalid session slopes.")
            return slopes / np.sqrt(variances)

        for column, (index, step) in enumerate(
            zip(spin_indices, steps, strict=True)
        ):
            plus_values = full.copy()
            minus_values = full.copy()
            plus_values[index] += step
            minus_values[index] -= step
            plus = self.evaluator.evaluate_values(
                _physical_values(self.coordinates, plus_values)
            )
            minus = self.evaluator.evaluate_values(
                _physical_values(self.coordinates, minus_values)
            )
            shape_jacobian[:, column] = (
                plus.normalized_residuals - minus.normalized_residuals
            ) / (2.0 * step)
            curvature_jacobian[:, column] = (
                plus.normalized_orthogonal_residuals
                - minus.normalized_orthogonal_residuals
            ) / (2.0 * step)
            standardized_slope_jacobian[:, column] = (
                standardized_slopes(plus) - standardized_slopes(minus)
            ) / (2.0 * step)
            offset_jacobian[:, column] = (
                plus.session_offsets - minus.session_offsets
            ) / (2.0 * step)

        relative_offsets = profile.session_offsets[1:] - profile.session_offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        relative_covariance = np.diag(profile.session_offset_variances[1:])
        relative_covariance += profile.session_offset_variances[0]
        return SpinOrbitLinearization(
            profile=profile,
            spin_reference=full[list(spin_indices)].copy(),
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative_offsets,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
            curvature_jacobian=curvature_jacobian,
            standardized_slope_jacobian=standardized_slope_jacobian,
        )

    def full_joint_linearize(
        self,
        normalized_values: Sequence[float],
        prior_covariance: Sequence[Sequence[float]],
        *,
        finite_difference_step: float,
        prior_mean_displacement: Sequence[float] | None = None,
    ) -> JointContinuousIntegerLinearization:
        """Linearize shape and all-session phase in every allowed coordinate."""

        full = np.asarray(normalized_values, dtype=float)
        lower = np.asarray(self.coordinates.normalized_lower, dtype=float)
        upper = np.asarray(self.coordinates.normalized_upper, dtype=float)
        if full.shape != lower.shape or upper.shape != lower.shape:
            raise ValueError("Full normalized coordinates have invalid bounds.")
        if np.any(full < lower) or np.any(full > upper):
            raise ValueError("A joint linearization point lies outside support.")

        profile = self.evaluator.evaluate_values(
            _physical_values(self.coordinates, full)
        )
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        offsets = np.asarray(profile.session_offsets, dtype=float)
        shape_jacobian = np.empty((shape.size, full.size), dtype=float)
        offset_jacobian = np.empty((offsets.size, full.size), dtype=float)
        step = max(float(finite_difference_step), 1.0e-7)
        for index in range(full.size):
            plus_values = full.copy()
            minus_values = full.copy()
            plus_values[index] = min(full[index] + step, upper[index])
            minus_values[index] = max(full[index] - step, lower[index])
            denominator = float(plus_values[index] - minus_values[index])
            if denominator <= 0.0:
                raise ValueError("A joint finite-difference coordinate is fixed.")
            plus = self.evaluator.evaluate_values(
                _physical_values(self.coordinates, plus_values)
            )
            minus = self.evaluator.evaluate_values(
                _physical_values(self.coordinates, minus_values)
            )
            plus_shape = np.asarray(plus.normalized_residuals, dtype=float)
            minus_shape = np.asarray(minus.normalized_residuals, dtype=float)
            plus_offsets = np.asarray(plus.session_offsets, dtype=float)
            minus_offsets = np.asarray(minus.session_offsets, dtype=float)
            if plus_shape.shape != shape.shape or minus_shape.shape != shape.shape:
                raise ValueError("Joint finite differences changed shape dimension.")
            if plus_offsets.shape != offsets.shape or minus_offsets.shape != offsets.shape:
                raise ValueError("Joint finite differences changed session dimension.")
            shape_jacobian[:, index] = (
                plus_shape - minus_shape
            ) / denominator
            offset_jacobian[:, index] = (
                plus_offsets - minus_offsets
            ) / denominator

        relative_offsets = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        relative_covariance = np.diag(
            np.asarray(profile.session_offset_variances[1:], dtype=float)
        )
        relative_covariance += float(profile.session_offset_variances[0])
        local_covariance = np.asarray(prior_covariance, dtype=float)
        if local_covariance.shape != (full.size, full.size):
            raise ValueError("A full joint covariance has the wrong dimension.")
        prior_mean = (
            np.zeros(full.size, dtype=float)
            if prior_mean_displacement is None
            else np.asarray(prior_mean_displacement, dtype=float)
        )
        if prior_mean.shape != (full.size,):
            raise ValueError("A full joint prior mean has the wrong dimension.")
        return JointContinuousIntegerLinearization(
            parameter_names=tuple(
                str(name).upper() for name in self.coordinates.parameter_names
            ),
            base_normalized_values=full,
            shape_residuals=shape,
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative_offsets,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
            prior_mean_displacement=prior_mean,
            local_prior_covariance=local_covariance,
            lower_displacement=lower - full,
            upper_displacement=upper - full,
        )

    def joint_linearize(
        self,
        orbit_normalized_values: Sequence[float],
        orbit_covariance: Sequence[Sequence[float]],
        *,
        finite_difference_step: float,
    ) -> JointContinuousIntegerLinearization:
        orbit_values = np.asarray(orbit_normalized_values, dtype=float)
        orbit_indices, spin_indices = self._indices()
        if orbit_values.shape != (len(orbit_indices),):
            raise ValueError("Orbit values do not match the orbit dimension.")
        full = np.asarray(self.reference_normalized_values, dtype=float).copy()
        full[list(orbit_indices)] = orbit_values
        local_covariance = np.zeros((full.size, full.size), dtype=float)
        local_covariance[np.ix_(orbit_indices, orbit_indices)] = np.asarray(
            orbit_covariance,
            dtype=float,
        )
        local_covariance[np.ix_(spin_indices, spin_indices)] = np.asarray(
            self.spin_prior_covariance,
            dtype=float,
        )
        prior_mean = np.zeros(full.size, dtype=float)
        prior_mean[list(spin_indices)] = np.asarray(
            self.spin_prior_mean,
            dtype=float,
        )
        return self.full_joint_linearize(
            full,
            local_covariance,
            finite_difference_step=finite_difference_step,
            prior_mean_displacement=prior_mean,
        )

    def acquire_conditioned_paths(
        self,
        orbit_normalized_values: Sequence[float],
        orbit_covariance: Sequence[Sequence[float]],
        *,
        seed_paths: Sequence[Sequence[int]] = (),
        excluded_paths: Sequence[Sequence[int]] = (),
        config: ConditionedPathDirectionConfig | None = None,
    ) -> ConditionedPathAcquisitionResult:
        active = config or ConditionedPathDirectionConfig()
        linearization = self.joint_linearize(
            orbit_normalized_values,
            orbit_covariance,
            finite_difference_step=active.finite_difference_step,
        )
        return acquire_conditioned_path_directions(
            linearization,
            seed_paths=seed_paths,
            excluded_paths=excluded_paths,
            config=active,
        )

    def acquire_conditioned_paths_from_full_chart(
        self,
        normalized_values: Sequence[float],
        covariance: Sequence[Sequence[float]],
        *,
        seed_paths: Sequence[Sequence[int]] = (),
        excluded_paths: Sequence[Sequence[int]] = (),
        config: ConditionedPathDirectionConfig | None = None,
    ) -> ConditionedPathAcquisitionResult:
        active = config or ConditionedPathDirectionConfig()
        linearization = self.full_joint_linearize(
            normalized_values,
            covariance,
            finite_difference_step=active.finite_difference_step,
        )
        return acquire_conditioned_path_directions(
            linearization,
            seed_paths=seed_paths,
            excluded_paths=excluded_paths,
            config=active,
        )

    def orbit_values_from_full_normalized(
        self,
        normalized_values: Sequence[float],
    ) -> np.ndarray:
        orbit_indices, _ = self._indices()
        values = np.asarray(normalized_values, dtype=float)
        if values.shape != (len(self.coordinates.parameter_names),):
            raise ValueError("Full values do not match the coordinate dimension.")
        return values[list(orbit_indices)].copy()

    def orbit_covariance_from_full(
        self,
        covariance: Sequence[Sequence[float]],
    ) -> np.ndarray:
        orbit_indices, _ = self._indices()
        matrix = np.asarray(covariance, dtype=float)
        dimension = len(self.coordinates.parameter_names)
        if matrix.shape != (dimension, dimension):
            raise ValueError("Full covariance does not match the coordinate dimension.")
        return matrix[np.ix_(orbit_indices, orbit_indices)].copy()

    def evaluate(
        self,
        orbit_normalized_values: Sequence[float],
    ) -> SpinIntegerMarginalEvaluation:
        return marginalize_spin_and_integer_paths(
            self.linearize(orbit_normalized_values),
            spin_prior_mean=self.spin_prior_mean,
            spin_prior_covariance=self.spin_prior_covariance,
            config=self.config,
        )

    def evaluate_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[SpinIntegerMarginalEvaluation, ...]:
        """Evaluate several phase-noise posteriors from one orbit linearization."""

        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = self.linearize(orbit_normalized_values)
        return tuple(
            marginalize_spin_and_integer_paths(
                linearization,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                config=replace(
                    self.config,
                    added_session_phase_sigma_cycles=sigma,
                ),
            )
            for sigma in sigmas
        )

    def evaluate_global_and_island_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[
        tuple[SpinIntegerMarginalEvaluation, ...],
        tuple[SpinIntegerMarginalEvaluation, ...],
    ]:
        """Evaluate global and temporal-island phase targets at one orbit."""

        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = self.linearize(orbit_normalized_values)
        global_channels = tuple(
            marginalize_spin_and_integer_paths(
                linearization,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                config=replace(
                    self.config,
                    added_session_phase_sigma_cycles=sigma,
                ),
            )
            for sigma in sigmas
        )
        if self.session_partitions is None or len(self.session_partitions) <= 1:
            return global_channels, global_channels
        island_linearization = restrict_linearization_to_phase_islands(
            linearization,
            self.session_partitions,
        )
        island_channels = tuple(
            marginalize_spin_and_integer_paths(
                island_linearization,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                config=replace(
                    self.config,
                    added_session_phase_sigma_cycles=sigma,
                ),
            )
            for sigma in sigmas
        )
        return global_channels, island_channels

    def evaluate_global_and_partition_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[
        tuple[SpinIntegerMarginalEvaluation, ...],
        tuple[tuple[SpinIntegerMarginalEvaluation, ...], ...],
    ]:
        """Evaluate every nested temporal partition from one PINT linearization."""

        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = self.linearize(orbit_normalized_values)

        def evaluate_channels(
            active_linearization: SpinOrbitLinearization,
        ) -> tuple[SpinIntegerMarginalEvaluation, ...]:
            return tuple(
                marginalize_spin_and_integer_paths(
                    active_linearization,
                    spin_prior_mean=self.spin_prior_mean,
                    spin_prior_covariance=self.spin_prior_covariance,
                    config=replace(
                        self.config,
                        added_session_phase_sigma_cycles=sigma,
                    ),
                )
                for sigma in sigmas
            )

        global_channels = evaluate_channels(linearization)
        hierarchy = self.session_partition_hierarchy
        if hierarchy is None:
            if self.session_partitions is None:
                return global_channels, (global_channels,)
            hierarchy = (self.session_partitions,)
        partition_channels = tuple(
            evaluate_channels(
                restrict_linearization_to_phase_islands(
                    linearization,
                    partitions,
                )
            )
            for partitions in hierarchy
        )
        return global_channels, partition_channels

    def evaluate_island_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[SpinIntegerMarginalEvaluation, ...]:
        """Evaluate phase likelihood after marginalizing island bridge integers."""

        _, island_channels = self.evaluate_global_and_island_noise_channels(
            orbit_normalized_values,
            phase_noise_sigmas,
        )
        return island_channels

    def evaluate_partition_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
        *,
        session_partitions: Sequence[Sequence[int]],
    ) -> tuple[SpinIntegerMarginalEvaluation, ...]:
        """Evaluate one explicit temporal-island partition."""

        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = restrict_linearization_to_phase_islands(
            self.linearize(orbit_normalized_values),
            session_partitions,
        )
        return tuple(
            marginalize_spin_and_integer_paths(
                linearization,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                config=replace(
                    self.config,
                    added_session_phase_sigma_cycles=sigma,
                ),
            )
            for sigma in sigmas
        )

    def evaluate_temporal_bridge_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
        *,
        source_partitions: Sequence[Sequence[int]],
        target_partitions: Sequence[Sequence[int]],
    ) -> tuple[
        SpinIntegerMarginalEvaluation,
        tuple[SpinIntegerMarginalEvaluation, ...],
    ]:
        """Evaluate both sides of one temporal bridge from one linearization."""

        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = self.linearize(orbit_normalized_values)
        source_linearization = restrict_linearization_to_phase_islands(
            linearization,
            source_partitions,
        )
        target_linearization = restrict_linearization_to_phase_islands(
            linearization,
            target_partitions,
        )
        source = marginalize_spin_and_integer_paths(
            source_linearization,
            spin_prior_mean=self.spin_prior_mean,
            spin_prior_covariance=self.spin_prior_covariance,
            config=replace(
                self.config,
                added_session_phase_sigma_cycles=0.0,
            ),
        )
        targets = tuple(
            marginalize_spin_and_integer_paths(
                target_linearization,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                config=replace(
                    self.config,
                    added_session_phase_sigma_cycles=sigma,
                ),
            )
            for sigma in sigmas
        )
        return source, targets

    def evaluate_temporal_bridge_family_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        phase_noise_sigmas: Sequence[float],
        *,
        source_partitions: Sequence[Sequence[int]],
        target_partitions: Sequence[Sequence[int]],
        bridge_integers: Sequence[int],
    ) -> tuple[
        SpinIntegerMarginalEvaluation,
        tuple[SpinIntegerBridgeFamilyEvaluation, ...],
    ]:
        """Evaluate one fixed bridge family without top-K path truncation."""

        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = self.linearize(orbit_normalized_values)
        source_linearization = restrict_linearization_to_phase_islands(
            linearization,
            source_partitions,
        )
        target_linearization = restrict_linearization_to_phase_islands(
            linearization,
            target_partitions,
        )
        source = marginalize_spin_and_integer_paths(
            source_linearization,
            spin_prior_mean=self.spin_prior_mean,
            spin_prior_covariance=self.spin_prior_covariance,
            config=replace(
                self.config,
                added_session_phase_sigma_cycles=0.0,
            ),
        )
        session_count = sum(len(tuple(block)) for block in source_partitions)
        families = []
        for sigma in sigmas:
            channel_config = replace(
                self.config,
                added_session_phase_sigma_cycles=sigma,
            )
            target = marginalize_spin_and_integer_paths(
                target_linearization,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                config=channel_config,
            )
            families.append(
                marginalize_temporal_bridge_family(
                    target,
                    session_count=session_count,
                    source_partitions=source_partitions,
                    target_partitions=target_partitions,
                    bridge_integers=bridge_integers,
                    config=channel_config,
                )
            )
        return source, tuple(families)

    def with_phase_noise(
        self,
        phase_noise_sigma: float,
    ) -> "SpinIntegerMarginalizedOrbitLikelihood":
        """Return the same exact orbit target at another phase-noise channel."""

        sigma = float(phase_noise_sigma)
        if not np.isfinite(sigma) or sigma < 0.0:
            raise ValueError("Phase noise must be finite and non-negative.")
        return replace(
            self,
            config=replace(
                self.config,
                added_session_phase_sigma_cycles=sigma,
            ),
        )

    def evaluate_fixed_path(
        self,
        orbit_normalized_values: Sequence[float],
        integer_offsets: Sequence[int],
        *,
        phase_noise_sigma: float | None = None,
    ) -> SpinFixedIntegerPathEvaluation:
        sigma = (
            float(self.config.added_session_phase_sigma_cycles)
            if phase_noise_sigma is None
            else float(phase_noise_sigma)
        )
        return marginalize_spin_for_fixed_integer_path(
            self.linearize(orbit_normalized_values),
            integer_offsets,
            spin_prior_mean=self.spin_prior_mean,
            spin_prior_covariance=self.spin_prior_covariance,
            added_session_phase_sigma_cycles=sigma,
        )

    def evaluate_fixed_path_noise_channels(
        self,
        orbit_normalized_values: Sequence[float],
        integer_offsets: Sequence[int],
        phase_noise_sigmas: Sequence[float],
    ) -> tuple[SpinFixedIntegerPathEvaluation, ...]:
        sigmas = tuple(float(value) for value in phase_noise_sigmas)
        if not sigmas or not all(
            np.isfinite(value) and value >= 0.0 for value in sigmas
        ):
            raise ValueError("Phase-noise channels must be finite and non-negative.")
        linearization = self.linearize(orbit_normalized_values)
        return tuple(
            marginalize_spin_for_fixed_integer_path(
                linearization,
                integer_offsets,
                spin_prior_mean=self.spin_prior_mean,
                spin_prior_covariance=self.spin_prior_covariance,
                added_session_phase_sigma_cycles=sigma,
            )
            for sigma in sigmas
        )

    def physical_fixed_path_values(
        self,
        orbit_normalized_values: Sequence[float],
        evaluation: SpinFixedIntegerPathEvaluation,
    ) -> np.ndarray:
        orbit_indices, spin_indices = self._indices()
        full = np.asarray(self.reference_normalized_values, dtype=float).copy()
        full[list(orbit_indices)] = np.asarray(orbit_normalized_values, dtype=float)
        full[list(spin_indices)] += np.asarray(
            evaluation.spin_posterior_mean,
            dtype=float,
        )
        return _physical_values(self.coordinates, full)

    def __call__(self, orbit_normalized_values: Sequence[float]) -> float:
        try:
            value = self.evaluate(orbit_normalized_values).log_likelihood
        except (ArithmeticError, RuntimeError, ValueError):
            return -math.inf
        return float(value) if np.isfinite(value) else -math.inf

    def physical_path_values(
        self,
        orbit_normalized_values: Sequence[float],
        path_mode: SpinIntegerPathMode,
    ) -> np.ndarray:
        orbit_indices, spin_indices = self._indices()
        full = np.asarray(self.reference_normalized_values, dtype=float).copy()
        full[list(orbit_indices)] = np.asarray(orbit_normalized_values, dtype=float)
        full[list(spin_indices)] += np.asarray(
            path_mode.spin_posterior_mean,
            dtype=float,
        )
        return _physical_values(self.coordinates, full)


@dataclass(frozen=True)
class OrbitBoxPriorTransform:
    lower: np.ndarray
    upper: np.ndarray

    def __call__(self, unit_values: Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=float)
        if unit.shape != self.lower.shape:
            raise ValueError("Orbit prior coordinates do not match the support.")
        return self.lower + unit * (self.upper - self.lower)


@dataclass(frozen=True)
class OrbitChartMixturePriorTransform:
    """Bounded broad-plus-logistic-normal orbit chart mixture."""

    lower: np.ndarray
    upper: np.ndarray
    logit_centers: np.ndarray
    logit_covariance_roots: np.ndarray
    broad_component_weight: float

    def __call__(self, unit_values: Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=float)
        if unit.shape != self.lower.shape:
            raise ValueError("Orbit mixture coordinates do not match support.")
        if not np.all(np.isfinite(unit) & (unit >= 0.0) & (unit <= 1.0)):
            raise ValueError("Orbit mixture coordinates must lie in [0, 1].")
        broad_weight = float(self.broad_component_weight)
        if unit[0] < broad_weight:
            broad_unit = unit.copy()
            broad_unit[0] /= broad_weight
            return self.lower + broad_unit * (self.upper - self.lower)

        component_coordinate = (
            (unit[0] - broad_weight)
            / (1.0 - broad_weight)
            * self.logit_centers.shape[0]
        )
        component_index = min(
            int(component_coordinate),
            self.logit_centers.shape[0] - 1,
        )
        local_unit = unit.copy()
        local_unit[0] = component_coordinate - component_index
        probabilities = np.clip(
            local_unit,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
        latent = self.logit_centers[component_index] + (
            self.logit_covariance_roots[component_index]
            @ ndtri(probabilities)
        )
        return self.lower + (self.upper - self.lower) * expit(latent)


def build_orbit_chart_mixture_prior(
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    centers: Sequence[Sequence[float]],
    covariances: Sequence[Sequence[Sequence[float]]],
    scale_factors: Sequence[float],
    broad_component_weight: float = 0.1,
) -> OrbitChartMixturePriorTransform:
    """Build equal-mass multiscale charts in a bounded orbit support."""

    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    center_values = np.asarray(centers, dtype=float)
    covariance_values = np.asarray(covariances, dtype=float)
    if lower_values.ndim != 1 or upper_values.shape != lower_values.shape:
        raise ValueError("Orbit mixture bounds must be matching vectors.")
    dimension = int(lower_values.size)
    if not np.all(
        np.isfinite(lower_values)
        & np.isfinite(upper_values)
        & (lower_values < upper_values)
    ):
        raise ValueError("Orbit mixture bounds must be finite and ordered.")
    if center_values.ndim != 2 or center_values.shape[1] != dimension:
        raise ValueError("Orbit chart centers must match the support dimension.")
    if covariance_values.shape != (
        center_values.shape[0],
        dimension,
        dimension,
    ):
        raise ValueError("Every orbit chart requires a matching covariance.")
    if np.any((center_values <= lower_values) | (center_values >= upper_values)):
        raise ValueError("Orbit chart centers must lie strictly inside support.")
    scales = np.asarray(scale_factors, dtype=float)
    if scales.ndim != 1 or scales.size == 0 or not np.all(
        np.isfinite(scales) & (scales > 0.0)
    ):
        raise ValueError("Orbit chart scales must be finite and positive.")
    broad_weight = float(broad_component_weight)
    if not np.isfinite(broad_weight) or not 0.0 < broad_weight < 1.0:
        raise ValueError("Broad orbit mixture weight must lie between zero and one.")

    width = upper_values - lower_values
    chart_logit_centers = []
    chart_roots = []
    for center, covariance in zip(
        center_values,
        covariance_values,
        strict=True,
    ):
        covariance = _positive_definite_matrix(
            covariance,
            dimension=dimension,
            label="Orbit chart covariance",
        )
        center_probability = (center - lower_values) / width
        center_logit = logit(center_probability)
        logit_jacobian = np.diag(
            1.0 / (width * center_probability * (1.0 - center_probability))
        )
        logit_covariance = logit_jacobian @ covariance @ logit_jacobian.T
        logit_covariance = 0.5 * (
            logit_covariance + logit_covariance.T
        )
        root = np.linalg.cholesky(logit_covariance)
        for scale in scales:
            chart_logit_centers.append(center_logit.copy())
            chart_roots.append(float(scale) * root)
    return OrbitChartMixturePriorTransform(
        lower=lower_values,
        upper=upper_values,
        logit_centers=np.asarray(chart_logit_centers, dtype=float),
        logit_covariance_roots=np.asarray(chart_roots, dtype=float),
        broad_component_weight=broad_weight,
    )


__all__ = [
    "OrbitBoxPriorTransform",
    "OrbitChartMixturePriorTransform",
    "SpinIntegerMarginalConfig",
    "SpinIntegerBridgeMode",
    "SpinIntegerBridgeFamilyEvaluation",
    "SpinIntegerMarginalEvaluation",
    "SpinIntegerMarginalizedOrbitLikelihood",
    "SpinIntegerPathMode",
    "SpinFixedIntegerPathEvaluation",
    "SpinOrbitLinearization",
    "build_orbit_chart_mixture_prior",
    "infer_temporal_phase_island_hierarchy",
    "infer_temporal_phase_islands",
    "group_temporal_partition_bridge_modes",
    "marginalize_temporal_bridge_family",
    "marginalize_spin_and_integer_paths",
    "marginalize_spin_for_fixed_integer_path",
    "restrict_linearization_to_phase_islands",
    "temporal_partition_bridge_signature",
    "temporal_partition_bridge_matrix",
    "temporal_partition_integer_transform",
    "temporal_phase_island_incidence",
]
