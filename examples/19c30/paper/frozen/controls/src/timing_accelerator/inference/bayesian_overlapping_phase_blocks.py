from __future__ import annotations

"""Bayesian mixture messages for overlapping temporal phase blocks.

The module uses every session in the shape likelihood while temporarily
marginalizing integer phase paths inside overlapping temporal blocks.  Block
modes are joined only when their integer differences agree on every shared
session.  The resulting full paths are proposals for exact all-session PINT
closure; block scores never replace the exact timing likelihood.
"""

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from timing_accelerator.inference.bayesian_dual_lattice_enumerator import (
    DualLatticeCellMode,
    DualLatticeCoordinateSystem,
    DualLatticeSupport,
    OrbitalCycleCell,
    PintDualLatticePhaseEvaluator,
    _evaluate_fixed_normalized,
    _normalized_bounds_for_cell,
    _normalized_bounds_for_mode,
    _project_to_connected_physical_segment,
    linearize_dual_lattice_cell_mode,
)
from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
    restrict_linearization_to_phase_islands,
    temporal_phase_island_incidence,
)


@dataclass(frozen=True)
class OverlappingPhaseBlockConfig:
    path_count: int = 64
    maximum_closure_evaluations: int = 320
    finite_difference_step: float = 2.0e-4
    prior_fraction_of_chart: float = 1.0
    lattice: SpinIntegerMarginalConfig = SpinIntegerMarginalConfig(
        initial_lattice_candidates=64,
        maximum_lattice_candidates=128,
        lattice_log_mass_tolerance=1.0e-5,
        maximum_lattice_nodes=2_000_000,
    )


@dataclass(frozen=True)
class PhaseBlockMode:
    mode_id: str
    block: tuple[int, ...]
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    normalized_covariance: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    block_phase_chi2: float
    block_total_chi2: float
    log_laplace_weight: float
    lattice_squared_mahalanobis: float
    lattice_posterior_weight: float
    function_evaluations: int
    converged: bool


@dataclass(frozen=True)
class PhaseBlockMessage:
    block: tuple[int, ...]
    modes: tuple[PhaseBlockMode, ...]


@dataclass(frozen=True)
class GlobalPhasePathMode:
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    normalized_covariance: np.ndarray
    source_mode_ids: tuple[str, ...]
    source_normalized_values: tuple[np.ndarray, ...]
    local_integer_offsets: tuple[tuple[int, ...], ...]
    log_message_weight: float
    log_continuous_compatibility: float


@dataclass(frozen=True)
class GlobalPhaseClosureMode:
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    source_mode_ids: tuple[str, ...]
    start_source: str
    function_evaluations: int
    converged: bool


@dataclass(frozen=True)
class ConditionedComplementaryPhaseMode:
    """One full path produced by a bidirectional phase-island message."""

    source_mode_id: str
    source_block: tuple[int, ...]
    complement_block: tuple[int, ...]
    source_integer_offsets: tuple[int, ...]
    complement_integer_offsets: tuple[int, ...]
    bridge_integer: int
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    normalized_covariance: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    log_message_weight: float
    lattice_squared_mahalanobis: float
    bridge_squared_mahalanobis: float
    approximate_shape_chi2: float
    approximate_phase_chi2: float


def _complementary_block(
    session_count: int,
    source_block: Sequence[int],
) -> tuple[int, ...]:
    source = tuple(int(value) for value in source_block)
    if (
        len(source) < 2
        or tuple(sorted(source)) != source
        or source != tuple(range(source[0], source[-1] + 1))
    ):
        raise ValueError("The source phase island must be a contiguous block.")
    complement = tuple(index for index in range(int(session_count)) if index not in source)
    if (
        len(complement) < 2
        or complement != tuple(range(complement[0], complement[-1] + 1))
    ):
        raise ValueError(
            "A complementary phase message requires two contiguous islands."
        )
    if source[0] != 0 and complement[0] != 0:
        raise ValueError("One complementary phase island must contain session zero.")
    return complement


def condition_linearization_on_phase_block(
    linearization: SpinOrbitLinearization,
    *,
    source_block: Sequence[int],
    source_integer_offsets: Sequence[int],
) -> tuple[SpinOrbitLinearization, tuple[int, ...]]:
    """Fix one island path and leave the complementary island marginalized.

    The fixed island factor is appended to the continuous Gaussian information
    block. The returned integer lattice contains only gauge-free differences
    internal to the other island, so no independently optimized middle block
    can remove a globally supported path.
    """

    relative = np.asarray(linearization.relative_session_offsets, dtype=float)
    jacobian = np.asarray(linearization.relative_session_jacobian, dtype=float)
    covariance = np.asarray(linearization.relative_offset_covariance, dtype=float)
    session_count = int(relative.size + 1)
    source = tuple(int(value) for value in source_block)
    complement = _complementary_block(session_count, source)
    source_integers = np.asarray(source_integer_offsets, dtype=float)
    if source_integers.shape != (len(source) - 1,):
        raise ValueError("Source integers must omit exactly the island anchor.")

    source_incidence = temporal_phase_island_incidence(
        session_count,
        phase_block_partitions(session_count, source),
    )
    complement_incidence = temporal_phase_island_incidence(
        session_count,
        phase_block_partitions(session_count, complement),
    )
    source_covariance = source_incidence @ covariance @ source_incidence.T
    source_cholesky = np.linalg.cholesky(source_covariance)
    fixed_residuals = np.linalg.solve(
        source_cholesky,
        source_incidence @ relative - source_integers,
    )
    fixed_jacobian = np.linalg.solve(
        source_cholesky,
        source_incidence @ jacobian,
    )

    shape = np.asarray(linearization.profile.normalized_residuals, dtype=float)
    shape_jacobian = np.asarray(linearization.shape_jacobian, dtype=float)
    conditioned_shape = np.concatenate((shape, fixed_residuals))
    conditioned_jacobian = np.vstack((shape_jacobian, fixed_jacobian))
    conditioned_profile = replace(
        linearization.profile,
        projected_phase_residuals=conditioned_shape.copy(),
        normalized_residuals=conditioned_shape,
        chi2=float(conditioned_shape @ conditioned_shape),
        dof=max(
            int(conditioned_shape.size - conditioned_jacobian.shape[1]),
            0,
        ),
        normalized_orthogonal_residuals=np.zeros(0, dtype=float),
    )
    return (
        replace(
            linearization,
            profile=conditioned_profile,
            shape_jacobian=conditioned_jacobian,
            relative_session_offsets=complement_incidence @ relative,
            relative_session_jacobian=complement_incidence @ jacobian,
            relative_offset_covariance=(
                complement_incidence @ covariance @ complement_incidence.T
            ),
            curvature_jacobian=np.zeros((0, conditioned_jacobian.shape[1])),
            standardized_slope_jacobian=np.zeros(
                (0, conditioned_jacobian.shape[1])
            ),
        ),
        complement,
    )


def combine_complementary_phase_paths(
    *,
    session_count: int,
    source_block: Sequence[int],
    source_integer_offsets: Sequence[int],
    complement_block: Sequence[int],
    complement_integer_offsets: Sequence[int],
    bridge_integer: int,
) -> tuple[int, ...]:
    """Join two gauge-free island paths with one inferred integer bridge."""

    count = int(session_count)
    source = tuple(int(value) for value in source_block)
    complement = tuple(int(value) for value in complement_block)
    if sorted(source + complement) != list(range(count)) or set(source) & set(
        complement
    ):
        raise ValueError("Complementary phase islands must partition every session.")
    source_values = (0,) + tuple(int(value) for value in source_integer_offsets)
    complement_values = (0,) + tuple(
        int(value) for value in complement_integer_offsets
    )
    if len(source_values) != len(source) or len(complement_values) != len(complement):
        raise ValueError("Each island path must omit exactly its anchor.")

    assignments: dict[int, int] = {}
    for block, values in ((source, source_values), (complement, complement_values)):
        gauge = 0 if 0 in block else int(bridge_integer)
        assignments.update(
            (session, gauge + value)
            for session, value in zip(block, values, strict=True)
        )
    origin = assignments[0]
    return tuple(assignments[index] - origin for index in range(1, count))


def _bridge_posterior(
    *,
    relative_offsets: np.ndarray,
    relative_covariance: np.ndarray,
    base_path: np.ndarray,
    bridge_indicator: np.ndarray,
) -> tuple[float, float]:
    precision_residual = np.linalg.solve(
        relative_covariance,
        relative_offsets - base_path,
    )
    precision_bridge = np.linalg.solve(
        relative_covariance,
        bridge_indicator,
    )
    information = float(bridge_indicator @ precision_bridge)
    if not np.isfinite(information) or information <= 0.0:
        raise ValueError("The inter-island bridge has no finite information.")
    center = float(bridge_indicator @ precision_residual / information)
    return center, float(1.0 / information)


def infer_conditioned_complementary_phase_modes(
    *,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    source_mode: PhaseBlockMode,
    normalized_lower: Sequence[float],
    normalized_upper: Sequence[float],
    config: OverlappingPhaseBlockConfig | None = None,
    bridge_count: int = 5,
) -> tuple[ConditionedComplementaryPhaseMode, ...]:
    """Run one exact Gaussian sum-product update between complementary islands."""

    active = config or OverlappingPhaseBlockConfig()
    lower = np.asarray(normalized_lower, dtype=float)
    upper = np.asarray(normalized_upper, dtype=float)
    reference = np.asarray(source_mode.normalized_values, dtype=float)
    if lower.shape != (8,) or upper.shape != (8,) or np.any(lower >= upper):
        raise ValueError("Conditional message bounds must be valid 8-vectors.")
    if np.any(reference < lower) or np.any(reference > upper):
        raise ValueError("The source phase mode lies outside its message chart.")
    slow, fast = support.denormalize(reference)
    reference_mode = DualLatticeCellMode(
        cell_id=cell.cell_id,
        start_index=0,
        normalized_values=reference,
        slow_values=slow,
        fast_values=fast,
        parameter_values=np.asarray(source_mode.parameter_values, dtype=float),
        local_integer_offsets=source_mode.local_integer_offsets,
        shape_chi2=float(source_mode.shape_chi2),
        function_evaluations=0,
        converged=bool(source_mode.converged),
        status=0,
        message="conditioned complementary phase-island message",
        on_cell_boundary=False,
        normalized_lower=lower,
        normalized_upper=upper,
    )
    full_linearization = linearize_dual_lattice_cell_mode(
        evaluator=evaluator,
        coordinates=coordinates,
        support=support,
        cell=cell,
        mode=reference_mode,
        finite_difference_step=max(float(active.finite_difference_step), 1.0e-7),
    )
    conditioned, complement = condition_linearization_on_phase_block(
        full_linearization,
        source_block=source_mode.block,
        source_integer_offsets=source_mode.integer_offsets,
    )
    half_width = 0.5 * (upper - lower)
    prior_sigma = np.maximum(
        half_width * max(float(active.prior_fraction_of_chart), 1.0e-3),
        1.0e-5,
    )
    requested = max(int(active.path_count), 1)
    lattice_config = SpinIntegerMarginalConfig(
        initial_lattice_candidates=max(
            int(active.lattice.initial_lattice_candidates), requested
        ),
        maximum_lattice_candidates=max(
            int(active.lattice.maximum_lattice_candidates), requested
        ),
        lattice_log_mass_tolerance=float(active.lattice.lattice_log_mass_tolerance),
        maximum_lattice_nodes=max(int(active.lattice.maximum_lattice_nodes), 1),
        added_session_phase_sigma_cycles=float(
            active.lattice.added_session_phase_sigma_cycles
        ),
    )
    marginal = marginalize_spin_and_integer_paths(
        conditioned,
        spin_prior_mean=np.zeros(8, dtype=float),
        spin_prior_covariance=np.diag(np.square(prior_sigma)),
        config=lattice_config,
    )

    count = len(evaluator.sessions)
    other_island = (
        complement if 0 not in complement else tuple(source_mode.block)
    )
    indicator = np.asarray(
        [float(session in other_island) for session in range(1, count)],
        dtype=float,
    )
    results: list[ConditionedComplementaryPhaseMode] = []
    for lattice_mode in marginal.path_modes[:requested]:
        target = np.clip(
            reference + np.asarray(lattice_mode.spin_posterior_mean, dtype=float),
            lower,
            upper,
        )
        target = _project_to_connected_physical_segment(
            reference=reference,
            target=target,
            coordinates=coordinates,
            support=support,
        )
        try:
            _, _, profile = _evaluate_fixed_normalized(
                normalized_values=target,
                evaluator=evaluator,
                coordinates=coordinates,
                support=support,
                local_integer_offsets=source_mode.local_integer_offsets,
            )
        except (ArithmeticError, RuntimeError, ValueError):
            continue
        complement_integers = tuple(
            int(value) for value in lattice_mode.integer_offsets
        )
        base = np.asarray(
            combine_complementary_phase_paths(
                session_count=count,
                source_block=source_mode.block,
                source_integer_offsets=source_mode.integer_offsets,
                complement_block=complement,
                complement_integer_offsets=complement_integers,
                bridge_integer=0,
            ),
            dtype=float,
        )
        relative = np.asarray(
            profile.session_intercepts[1:] - profile.session_intercepts[0],
            dtype=float,
        )
        covariance = np.diag(profile.session_intercept_variances[1:])
        covariance += float(profile.session_intercept_variances[0])
        bridge_center, bridge_variance = _bridge_posterior(
            relative_offsets=relative,
            relative_covariance=covariance,
            base_path=base,
            bridge_indicator=indicator,
        )
        nearest = int(np.rint(bridge_center))
        radius = max(int(math.ceil((max(int(bridge_count), 1) - 1) / 2)), 0)
        bridges = range(nearest - radius, nearest + radius + 1)
        for bridge in bridges:
            path = combine_complementary_phase_paths(
                session_count=count,
                source_block=source_mode.block,
                source_integer_offsets=source_mode.integer_offsets,
                complement_block=complement,
                complement_integer_offsets=complement_integers,
                bridge_integer=bridge,
            )
            phase_delta = relative - np.asarray(path, dtype=float)
            phase_chi2 = float(phase_delta @ np.linalg.solve(covariance, phase_delta))
            bridge_d2 = float((bridge - bridge_center) ** 2 / bridge_variance)
            results.append(
                ConditionedComplementaryPhaseMode(
                    source_mode_id=source_mode.mode_id,
                    source_block=tuple(source_mode.block),
                    complement_block=complement,
                    source_integer_offsets=tuple(source_mode.integer_offsets),
                    complement_integer_offsets=complement_integers,
                    bridge_integer=int(bridge),
                    integer_offsets=path,
                    normalized_values=target.copy(),
                    normalized_covariance=np.asarray(
                        lattice_mode.spin_posterior_covariance,
                        dtype=float,
                    ).copy(),
                    local_integer_offsets=source_mode.local_integer_offsets,
                    log_message_weight=float(
                        marginal.log_likelihood
                        + lattice_mode.log_posterior_weight
                        - 0.5 * bridge_d2
                    ),
                    lattice_squared_mahalanobis=float(
                        lattice_mode.squared_mahalanobis
                    ),
                    bridge_squared_mahalanobis=bridge_d2,
                    approximate_shape_chi2=float(profile.local_chi2),
                    approximate_phase_chi2=phase_chi2,
                )
            )
    results.sort(
        key=lambda item: (
            item.approximate_shape_chi2 + item.approximate_phase_chi2,
            -item.log_message_weight,
            item.integer_offsets,
        )
    )
    return tuple(results)


def infer_overlapping_phase_blocks(
    session_epochs: Sequence[float],
    *,
    block_size: int | None = None,
    minimum_overlap: int = 3,
) -> tuple[tuple[int, ...], ...]:
    """Cover an ordered session sequence with a minimal-separator block chain.

    The default block contains about half the sequence.  Consecutive windows
    advance by the largest stride that preserves the requested overlap, and a
    final endpoint-anchored window completes the cover.  This running-
    intersection construction avoids redundant high-dimensional separators.
    """

    epochs = np.asarray(session_epochs, dtype=float)
    if epochs.ndim != 1 or epochs.size < 3 or not np.all(np.isfinite(epochs)):
        raise ValueError("Phase-block epochs must be a finite vector of length >= 3.")
    if np.any(np.diff(epochs) <= 0.0):
        raise ValueError("Phase-block epochs must be strictly increasing.")
    count = int(epochs.size)
    width = (
        max(6, min(12, int(math.ceil(0.5 * count))))
        if block_size is None
        else int(block_size)
    )
    width = min(max(width, 3), count)
    overlap = min(max(int(minimum_overlap), 2), width - 1)
    if width == count:
        return (tuple(range(count)),)

    nominal_step = max(width - overlap, 1)
    starts = [0]
    final_start = count - width
    while starts[-1] < final_start:
        next_start = min(starts[-1] + nominal_step, final_start)
        if next_start <= starts[-1]:
            next_start = min(starts[-1] + nominal_step, final_start)
        starts.append(next_start)
    if starts[-1] != final_start:
        starts.append(final_start)
    starts = list(dict.fromkeys(starts))
    return tuple(tuple(range(start, start + width)) for start in starts)


def phase_block_partitions(
    session_count: int,
    block: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Represent one contiguous block and singleton unconstrained sessions."""

    count = int(session_count)
    sessions = tuple(int(value) for value in block)
    if (
        len(sessions) < 2
        or tuple(sorted(sessions)) != sessions
        or sessions != tuple(range(sessions[0], sessions[-1] + 1))
        or sessions[0] < 0
        or sessions[-1] >= count
    ):
        raise ValueError("A phase block must be a valid contiguous session interval.")
    return (
        tuple((index,) for index in range(sessions[0]))
        + (sessions,)
        + tuple((index,) for index in range(sessions[-1] + 1, count))
    )


def overlap_integer_signature(
    block: Sequence[int],
    integer_offsets: Sequence[int],
    overlap: Sequence[int],
) -> tuple[int, ...]:
    """Return gauge-free integer differences on a shared session interval."""

    sessions = tuple(int(value) for value in block)
    shared = tuple(int(value) for value in overlap)
    integers = np.asarray(integer_offsets, dtype=np.int64)
    if integers.shape != (len(sessions) - 1,):
        raise ValueError("Block integer offsets must omit only the block anchor.")
    if not shared or any(session not in sessions for session in shared):
        raise ValueError("The overlap must be a non-empty subset of the block.")
    values = np.concatenate((np.zeros(1, dtype=np.int64), integers))
    by_session = {session: int(values[index]) for index, session in enumerate(sessions)}
    anchor = by_session[shared[0]]
    return tuple(by_session[session] - anchor for session in shared)


def _stable_covariance(jacobian: np.ndarray) -> np.ndarray:
    information = np.asarray(jacobian, dtype=float).T @ np.asarray(jacobian, dtype=float)
    information = 0.5 * (information + information.T)
    values, vectors = np.linalg.eigh(information)
    largest = max(float(np.max(values)), 1.0)
    values = np.maximum(values, largest * 1.0e-12)
    covariance = (vectors * (1.0 / values)) @ vectors.T
    return 0.5 * (covariance + covariance.T)


def _log_laplace_weight(chi2: float, covariance: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(np.asarray(covariance, dtype=float))
    if sign <= 0.0 or not np.isfinite(logdet):
        return -math.inf
    dimension = int(np.asarray(covariance).shape[0])
    return float(-0.5 * chi2 + 0.5 * (logdet + dimension * math.log(2.0 * math.pi)))


def solve_overlapping_phase_block_message(
    *,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    reference_mode: DualLatticeCellMode,
    block: Sequence[int],
    config: OverlappingPhaseBlockConfig | None = None,
    message_prefix: str = "block",
) -> PhaseBlockMessage:
    """Marginalize and exact-close every retained path inside one block."""

    active = config or OverlappingPhaseBlockConfig()
    sessions = tuple(int(value) for value in block)
    session_count = len(evaluator.sessions)
    partitions = phase_block_partitions(session_count, sessions)
    incidence = temporal_phase_island_incidence(session_count, partitions)
    linearization = linearize_dual_lattice_cell_mode(
        evaluator=evaluator,
        coordinates=coordinates,
        support=support,
        cell=cell,
        mode=reference_mode,
        finite_difference_step=max(float(active.finite_difference_step), 1.0e-7),
    )
    block_linearization = restrict_linearization_to_phase_islands(
        linearization,
        partitions,
    )
    lower, upper = _normalized_bounds_for_mode(support, cell, reference_mode)
    half_width = 0.5 * (upper - lower)
    prior_sigma = np.maximum(
        half_width * max(float(active.prior_fraction_of_chart), 1.0e-3),
        1.0e-5,
    )
    requested = max(int(active.path_count), 1)
    lattice_config = SpinIntegerMarginalConfig(
        initial_lattice_candidates=max(
            int(active.lattice.initial_lattice_candidates), requested
        ),
        maximum_lattice_candidates=max(
            int(active.lattice.maximum_lattice_candidates), requested
        ),
        lattice_log_mass_tolerance=float(active.lattice.lattice_log_mass_tolerance),
        maximum_lattice_nodes=max(int(active.lattice.maximum_lattice_nodes), 1),
        added_session_phase_sigma_cycles=float(
            active.lattice.added_session_phase_sigma_cycles
        ),
    )
    marginal = marginalize_spin_and_integer_paths(
        block_linearization,
        spin_prior_mean=np.zeros(8, dtype=float),
        spin_prior_covariance=np.diag(np.square(prior_sigma)),
        config=lattice_config,
    )
    reference = np.asarray(reference_mode.normalized_values, dtype=float)

    def evaluate(normalized: np.ndarray):
        return _evaluate_fixed_normalized(
            normalized_values=normalized,
            evaluator=evaluator,
            coordinates=coordinates,
            support=support,
            local_integer_offsets=reference_mode.local_integer_offsets,
        )

    modes: list[PhaseBlockMode] = []
    for rank, lattice_mode in enumerate(marginal.path_modes[:requested]):
        path = np.asarray(lattice_mode.integer_offsets, dtype=float)
        target = np.clip(
            reference + np.asarray(lattice_mode.spin_posterior_mean, dtype=float),
            lower,
            upper,
        )
        start = _project_to_connected_physical_segment(
            reference=reference,
            target=target,
            coordinates=coordinates,
            support=support,
        )

        def residuals(normalized: np.ndarray) -> np.ndarray:
            try:
                _, _, profile = evaluate(normalized)
                relative = np.asarray(
                    profile.session_intercepts[1:] - profile.session_intercepts[0],
                    dtype=float,
                )
                variances = np.asarray(profile.session_intercept_variances, dtype=float)
                covariance = np.diag(variances[1:]) + float(variances[0])
                block_covariance = incidence @ covariance @ incidence.T
                phase = np.linalg.solve(
                    np.linalg.cholesky(block_covariance),
                    incidence @ relative - path,
                )
                return np.concatenate((profile.normalized_local_residuals, phase))
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                return np.full(len(evaluator.toas.table) + path.size, 1.0e8, dtype=float)

        try:
            fit = least_squares(
                residuals,
                start,
                bounds=(lower, upper),
                method="trf",
                jac="2-point",
                x_scale="jac",
                ftol=1.0e-10,
                xtol=1.0e-10,
                gtol=1.0e-10,
                max_nfev=max(int(active.maximum_closure_evaluations), 8),
            )
            physical, _, profile = evaluate(np.asarray(fit.x, dtype=float))
            exact = residuals(np.asarray(fit.x, dtype=float))
            if not np.all(np.isfinite(exact)):
                continue
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        shape_size = len(evaluator.toas.table)
        shape_chi2 = float(exact[:shape_size] @ exact[:shape_size])
        phase_chi2 = float(exact[shape_size:] @ exact[shape_size:])
        covariance = _stable_covariance(np.asarray(fit.jac, dtype=float))
        total = shape_chi2 + phase_chi2
        modes.append(
            PhaseBlockMode(
                mode_id=f"{message_prefix}:p{rank:05d}",
                block=sessions,
                integer_offsets=tuple(int(value) for value in lattice_mode.integer_offsets),
                normalized_values=np.asarray(fit.x, dtype=float).copy(),
                normalized_covariance=covariance,
                parameter_values=np.asarray(physical, dtype=float).copy(),
                local_integer_offsets=profile.local_integer_offsets,
                shape_chi2=shape_chi2,
                block_phase_chi2=phase_chi2,
                block_total_chi2=total,
                log_laplace_weight=_log_laplace_weight(total, covariance),
                lattice_squared_mahalanobis=float(lattice_mode.squared_mahalanobis),
                lattice_posterior_weight=float(lattice_mode.posterior_weight),
                function_evaluations=int(fit.nfev),
                converged=bool(fit.success),
            )
        )
    modes.sort(
        key=lambda item: (
            item.block_total_chi2,
            item.lattice_squared_mahalanobis,
            item.integer_offsets,
        )
    )
    return PhaseBlockMessage(block=sessions, modes=tuple(modes))


def _gaussian_product(
    left_mean: np.ndarray,
    left_covariance: np.ndarray,
    right_mean: np.ndarray,
    right_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    left_precision = np.linalg.inv(np.asarray(left_covariance, dtype=float))
    right_precision = np.linalg.inv(np.asarray(right_covariance, dtype=float))
    precision = left_precision + right_precision
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    mean = covariance @ (
        left_precision @ np.asarray(left_mean, dtype=float)
        + right_precision @ np.asarray(right_mean, dtype=float)
    )
    delta = np.asarray(left_mean, dtype=float) - np.asarray(right_mean, dtype=float)
    summed = np.asarray(left_covariance, dtype=float) + np.asarray(
        right_covariance, dtype=float
    )
    sign, logdet = np.linalg.slogdet(summed)
    if sign <= 0.0:
        compatibility = -math.inf
    else:
        compatibility = float(
            -0.5
            * (
                delta @ np.linalg.solve(summed, delta)
                + logdet
                + delta.size * math.log(2.0 * math.pi)
            )
        )
    return mean, covariance, compatibility


@dataclass(frozen=True)
class _PartialPath:
    assignments: tuple[tuple[int, int], ...]
    normalized_values: np.ndarray
    normalized_covariance: np.ndarray
    source_mode_ids: tuple[str, ...]
    source_normalized_values: tuple[np.ndarray, ...]
    local_integer_offsets: tuple[tuple[int, ...], ...]
    log_message_weight: float
    log_continuous_compatibility: float


def _mode_assignments(mode: PhaseBlockMode) -> dict[int, int]:
    values = (0,) + tuple(int(value) for value in mode.integer_offsets)
    return dict(zip(mode.block, values, strict=True))


def join_overlapping_phase_block_messages(
    messages: Sequence[PhaseBlockMessage],
    *,
    session_count: int,
) -> tuple[GlobalPhasePathMode, ...]:
    """Join all compatible block-mixture components without choosing a winner."""

    ordered = tuple(sorted(messages, key=lambda item: (item.block[0], item.block[-1])))
    if not ordered or any(not message.modes for message in ordered):
        return ()
    if ordered[0].block[0] != 0 or ordered[-1].block[-1] != int(session_count) - 1:
        raise ValueError("Phase-block messages must cover both sequence endpoints.")

    partials = []
    for mode in ordered[0].modes:
        partials.append(
            _PartialPath(
                assignments=tuple(sorted(_mode_assignments(mode).items())),
                normalized_values=np.asarray(mode.normalized_values, dtype=float),
                normalized_covariance=np.asarray(
                    mode.normalized_covariance, dtype=float
                ),
                source_mode_ids=(mode.mode_id,),
                source_normalized_values=(
                    np.asarray(mode.normalized_values, dtype=float),
                ),
                local_integer_offsets=mode.local_integer_offsets,
                log_message_weight=float(mode.log_laplace_weight),
                log_continuous_compatibility=0.0,
            )
        )

    for message in ordered[1:]:
        by_signature: dict[tuple[tuple[int, ...], tuple[int, ...]], list[PhaseBlockMode]] = {}
        for mode in message.modes:
            assignments = _mode_assignments(mode)
            for partial in partials[:1]:
                shared = tuple(sorted(set(dict(partial.assignments)) & set(assignments)))
                break
            else:
                shared = ()
            if not shared:
                raise ValueError("Consecutive phase blocks must overlap.")
            signature = overlap_integer_signature(
                mode.block,
                mode.integer_offsets,
                shared,
            )
            by_signature.setdefault((shared, signature), []).append(mode)

        joined: list[_PartialPath] = []
        for partial in partials:
            existing = dict(partial.assignments)
            shared = tuple(sorted(set(existing) & set(message.block)))
            if not shared:
                raise ValueError("Consecutive phase blocks must overlap.")
            existing_anchor = existing[shared[0]]
            existing_signature = tuple(
                existing[session] - existing_anchor for session in shared
            )
            for mode in by_signature.get((shared, existing_signature), ()):
                local = _mode_assignments(mode)
                gauge = existing_anchor - local[shared[0]]
                aligned = {session: value + gauge for session, value in local.items()}
                if any(existing[session] != aligned[session] for session in shared):
                    continue
                combined = dict(existing)
                combined.update(aligned)
                mean, covariance, compatibility = _gaussian_product(
                    partial.normalized_values,
                    partial.normalized_covariance,
                    mode.normalized_values,
                    mode.normalized_covariance,
                )
                joined.append(
                    _PartialPath(
                        assignments=tuple(sorted(combined.items())),
                        normalized_values=mean,
                        normalized_covariance=covariance,
                        source_mode_ids=partial.source_mode_ids + (mode.mode_id,),
                        source_normalized_values=(
                            partial.source_normalized_values
                            + (np.asarray(mode.normalized_values, dtype=float),)
                        ),
                        local_integer_offsets=partial.local_integer_offsets,
                        log_message_weight=(
                            partial.log_message_weight
                            + float(mode.log_laplace_weight)
                            + compatibility
                        ),
                        log_continuous_compatibility=(
                            partial.log_continuous_compatibility + compatibility
                        ),
                    )
                )
        partials = joined
        if not partials:
            return ()

    result = []
    for partial in partials:
        assignments = dict(partial.assignments)
        if sorted(assignments) != list(range(int(session_count))):
            continue
        gauge = assignments[0]
        path = tuple(assignments[index] - gauge for index in range(1, int(session_count)))
        result.append(
            GlobalPhasePathMode(
                integer_offsets=path,
                normalized_values=partial.normalized_values,
                normalized_covariance=partial.normalized_covariance,
                source_mode_ids=partial.source_mode_ids,
                source_normalized_values=partial.source_normalized_values,
                local_integer_offsets=partial.local_integer_offsets,
                log_message_weight=partial.log_message_weight,
                log_continuous_compatibility=partial.log_continuous_compatibility,
            )
        )
    result.sort(
        key=lambda item: (
            -item.log_message_weight,
            item.integer_offsets,
            item.source_mode_ids,
        )
    )
    return tuple(result)


def close_global_phase_path_mode(
    *,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    mode: GlobalPhasePathMode,
    maximum_function_evaluations: int = 640,
) -> tuple[GlobalPhaseClosureMode, ...]:
    """Refine one joined path on the exact complete-session PINT target."""

    lower, upper = _normalized_bounds_for_cell(support, cell)
    path = np.asarray(mode.integer_offsets, dtype=float)

    def evaluate(normalized: np.ndarray):
        return _evaluate_fixed_normalized(
            normalized_values=normalized,
            evaluator=evaluator,
            coordinates=coordinates,
            support=support,
            local_integer_offsets=mode.local_integer_offsets,
        )

    def residuals(normalized: np.ndarray) -> np.ndarray:
        try:
            _, _, profile = evaluate(normalized)
            relative = np.asarray(
                profile.session_intercepts[1:] - profile.session_intercepts[0],
                dtype=float,
            )
            variances = np.asarray(profile.session_intercept_variances, dtype=float)
            covariance = np.diag(variances[1:]) + float(variances[0])
            phase = np.linalg.solve(
                np.linalg.cholesky(covariance),
                relative - path,
            )
            return np.concatenate((profile.normalized_local_residuals, phase))
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return np.full(len(evaluator.toas.table) + path.size, 1.0e8, dtype=float)

    starts = [("gaussian-product", np.asarray(mode.normalized_values, dtype=float))]
    starts.extend(
        (f"source:{index}", np.asarray(values, dtype=float))
        for index, values in enumerate(mode.source_normalized_values)
    )
    closures: list[GlobalPhaseClosureMode] = []
    seen_starts: list[np.ndarray] = []
    for source, proposal in starts:
        clipped = np.clip(proposal, lower, upper)
        reference = np.clip(mode.source_normalized_values[0], lower, upper)
        try:
            start = _project_to_connected_physical_segment(
                reference=reference,
                target=clipped,
                coordinates=coordinates,
                support=support,
            )
        except ValueError:
            continue
        if any(np.linalg.norm(start - previous) <= 1.0e-10 for previous in seen_starts):
            continue
        seen_starts.append(start)
        try:
            fit = least_squares(
                residuals,
                start,
                bounds=(lower, upper),
                method="trf",
                jac="2-point",
                x_scale="jac",
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-11,
                max_nfev=max(int(maximum_function_evaluations), 8),
            )
            physical, _, profile = evaluate(np.asarray(fit.x, dtype=float))
            exact = residuals(np.asarray(fit.x, dtype=float))
            if not np.all(np.isfinite(exact)):
                continue
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        shape_size = len(evaluator.toas.table)
        shape_chi2 = float(exact[:shape_size] @ exact[:shape_size])
        phase_chi2 = float(exact[shape_size:] @ exact[shape_size:])
        closures.append(
            GlobalPhaseClosureMode(
                integer_offsets=mode.integer_offsets,
                normalized_values=np.asarray(fit.x, dtype=float).copy(),
                parameter_values=np.asarray(physical, dtype=float).copy(),
                local_integer_offsets=profile.local_integer_offsets,
                shape_chi2=shape_chi2,
                phase_chi2=phase_chi2,
                total_chi2=shape_chi2 + phase_chi2,
                source_mode_ids=mode.source_mode_ids,
                start_source=source,
                function_evaluations=int(fit.nfev),
                converged=bool(fit.success),
            )
        )
    closures.sort(key=lambda item: (item.total_chi2, item.start_source))
    return tuple(closures)


__all__ = [
    "ConditionedComplementaryPhaseMode",
    "GlobalPhaseClosureMode",
    "GlobalPhasePathMode",
    "OverlappingPhaseBlockConfig",
    "PhaseBlockMessage",
    "PhaseBlockMode",
    "close_global_phase_path_mode",
    "combine_complementary_phase_paths",
    "condition_linearization_on_phase_block",
    "infer_conditioned_complementary_phase_modes",
    "infer_overlapping_phase_blocks",
    "join_overlapping_phase_block_messages",
    "overlap_integer_signature",
    "phase_block_partitions",
    "solve_overlapping_phase_block_message",
]
