from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass, field
from itertools import combinations
import json
import math
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp

from timing_accelerator.backend.pint_backend import (
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.bayesian_robust_phase_likelihood import (
    RobustPhaseNoiseConfig,
    RobustPhaseNoiseModel,
)
from timing_accelerator.inference.bayesian_short_baseline_update import (
    _linear_gaussian_update,
    _previous_parameter_covariance,
    _session_phase_basis,
)
from timing_accelerator.inference.integer_least_squares import (
    solve_integer_least_squares,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.reports.residual_plot import (
    write_residual_series_plot,
)


ISOLATED_SESSION_UPDATE_ARCHITECTURE = (
    "apollo_full_sequence_isolated_session_posterior"
)
ISOLATED_SESSION_PARAMETER_NAMES = ("RAJ", "DECJ", "F0", "F1")
ISOLATED_SESSION_COORDINATE_NAMES = (
    "SKY_EAST_ARCSEC",
    "SKY_NORTH_ARCSEC",
    "DELTA_F0_X1E9_HZ",
    "DELTA_F1_X1E16_HZ_S",
)


@dataclass(frozen=True)
class IsolatedSessionUpdateConfig:
    """Full-sequence spin-sky and integer-path posterior controls."""

    maximum_contaminated_sessions: int = 2
    maximum_candidate_sessions: int = 16
    maximum_contamination_subsets: int = 256
    maximum_integer_candidates: int = 8
    maximum_integer_nodes: int = 1_000_000
    maximum_output_modes: int = 32
    minimum_inlier_sessions: int = 6
    contamination_probability: float = 0.05
    historical_posterior_probability: float = 0.98
    broad_reconstruction_probability: float = 0.02
    previous_posterior_sigma_scale: float = 4.0
    coordinate_prior_sigmas: tuple[float, float, float, float] = (
        600.0,
        600.0,
        200.0,
        200.0,
    )
    phase_connection_probability_threshold: float = 0.95
    maximum_reduced_session_chi2: float = 5.0
    rank_relative_threshold: float = 1.0e-8
    maximum_design_condition_number: float = 1.0e6
    minimum_pm_span_days: float = 730.5
    minimum_annual_phase_coverage: float = 0.70
    enable_blockwise_long_gap_inference: bool = True
    minimum_long_gap_days: float = 365.25
    long_gap_factor: float = 8.0
    maximum_bridge_integer_candidates: int = 16
    maximum_prior_mahalanobis_distance: float = 8.0
    enable_automatic_session_marginalization: bool = True
    session_marginalization_probability_threshold: float = 0.99
    contamination_subset_probability_threshold: float = 0.95
    minimum_marginalized_standardized_residual: float = 5.0
    maximum_marginalized_session_fraction: float = 0.25
    phase_noise: RobustPhaseNoiseConfig = field(
        default_factory=RobustPhaseNoiseConfig
    )

    def __post_init__(self) -> None:
        nonnegative = (self.maximum_contaminated_sessions,)
        if any(int(value) < 0 for value in nonnegative):
            raise ValueError("The contaminated-session budget cannot be negative.")
        positive_integers = (
            self.maximum_candidate_sessions,
            self.maximum_contamination_subsets,
            self.maximum_integer_candidates,
            self.maximum_integer_nodes,
            self.maximum_output_modes,
            self.minimum_inlier_sessions,
            self.maximum_bridge_integer_candidates,
        )
        if any(int(value) <= 0 for value in positive_integers):
            raise ValueError("Isolated session-update budgets must be positive.")
        probabilities = (
            self.contamination_probability,
            self.historical_posterior_probability,
            self.broad_reconstruction_probability,
            self.session_marginalization_probability_threshold,
            self.contamination_subset_probability_threshold,
        )
        if any(not 0.0 < float(value) < 1.0 for value in probabilities):
            raise ValueError("Session-update probabilities must lie in (0, 1).")
        if not np.isclose(
            self.historical_posterior_probability
            + self.broad_reconstruction_probability,
            1.0,
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("Continuous-prior component probabilities must sum to one.")
        positive = (
            self.previous_posterior_sigma_scale,
            self.rank_relative_threshold,
            self.maximum_design_condition_number,
            self.minimum_pm_span_days,
            self.maximum_reduced_session_chi2,
            self.minimum_long_gap_days,
            self.long_gap_factor,
            self.maximum_prior_mahalanobis_distance,
            self.minimum_marginalized_standardized_residual,
            *self.coordinate_prior_sigmas,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Session-update scales must be positive and finite.")
        if not 0.5 < self.phase_connection_probability_threshold < 1.0:
            raise ValueError("The phase-connection threshold must lie in (0.5, 1).")
        if not 0.0 < self.minimum_annual_phase_coverage <= 1.0:
            raise ValueError("Annual phase coverage must lie in (0, 1].")
        if not 0.0 < self.maximum_marginalized_session_fraction < 1.0:
            raise ValueError(
                "The maximum marginalized-session fraction must lie in (0, 1)."
            )


@dataclass(frozen=True)
class IsolatedSessionUpdateMode:
    prior_component_id: str
    contaminated_session_indices: tuple[int, ...]
    relative_integer_path: tuple[int | None, ...]
    coordinate_names: tuple[str, ...]
    coordinate_mean: np.ndarray
    coordinate_covariance: np.ndarray
    native_parameter_names: tuple[str, ...]
    native_parameter_values: np.ndarray
    native_parameter_covariance: np.ndarray
    global_phase_cycles: float
    global_phase_sigma_cycles: float
    inlier_session_chi2: float
    inlier_session_dof: int
    inlier_weighted_rms_us: float
    log_weight: float
    posterior_weight: float
    block_gauge_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        coordinate = np.asarray(self.coordinate_mean, dtype=float)
        coordinate_covariance = np.asarray(self.coordinate_covariance, dtype=float)
        native = np.asarray(self.native_parameter_values, dtype=float)
        native_covariance = np.asarray(self.native_parameter_covariance, dtype=float)
        if coordinate.shape != (len(self.coordinate_names),):
            raise ValueError("Session-update coordinates do not match their names.")
        if coordinate_covariance.shape != (coordinate.size, coordinate.size):
            raise ValueError("Session-update coordinate covariance has the wrong shape.")
        if native.shape != (len(self.native_parameter_names),):
            raise ValueError("Native timing values do not match their names.")
        if native_covariance.shape != (native.size, native.size):
            raise ValueError("Native timing covariance has the wrong shape.")
        arrays = (coordinate, coordinate_covariance, native, native_covariance)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("Session-update modes must contain finite arrays.")
        object.__setattr__(self, "coordinate_mean", coordinate.copy())
        object.__setattr__(
            self,
            "coordinate_covariance",
            coordinate_covariance.copy(),
        )
        object.__setattr__(self, "native_parameter_values", native.copy())
        object.__setattr__(
            self,
            "native_parameter_covariance",
            native_covariance.copy(),
        )

    @property
    def reduced_inlier_session_chi2(self) -> float:
        if self.inlier_session_dof <= 0:
            return math.inf
        return float(self.inlier_session_chi2 / self.inlier_session_dof)


@dataclass(frozen=True)
class IsolatedSessionUpdateResult:
    modes: tuple[IsolatedSessionUpdateMode, ...]
    status: str
    unique_solution: bool
    dominant_joint_mode_probability: float
    retained_probability: float
    log_evidence: float
    session_epochs_mjd: np.ndarray
    session_intercepts_cycles: np.ndarray
    session_sigmas_cycles: np.ndarray
    session_contamination_probabilities: np.ndarray
    session_standardized_residuals: np.ndarray
    contamination_subset_probabilities: tuple[tuple[tuple[int, ...], float], ...]
    phase_path_probabilities: tuple[
        tuple[tuple[int | None, ...], float], ...
    ]
    prior_component_probabilities: tuple[tuple[str, float], ...]
    local_integer_offsets: tuple[tuple[int, ...], ...]
    inferred_jitter_cycles: float
    observation_span_days: float
    annual_phase_coverage: float
    weighted_design_rank: int
    weighted_design_singular_values: np.ndarray
    weighted_design_condition_number: float
    posterior_correlation: np.ndarray
    proper_motion_eligible: bool
    proper_motion_reason: str
    integer_search_budget_exhausted: bool
    candidate_session_indices: tuple[int, ...]
    searched_contamination_subsets: int
    phase_blocks: tuple[tuple[int, ...], ...] = ()
    long_gap_days: tuple[float, ...] = ()
    blockwise_long_gap_inference_used: bool = False

    @property
    def best_mode(self) -> IsolatedSessionUpdateMode:
        if not self.modes:
            raise ValueError("The isolated session update retained no modes.")
        return self.modes[0]


@dataclass(frozen=True)
class IsolatedSessionMarginalizationDecision:
    """Evidence gate for removing candidate outlier sessions before refitting."""

    session_indices: tuple[int, ...]
    toa_indices: tuple[int, ...]
    session_epochs_mjd: tuple[float, ...]
    session_probabilities: tuple[float, ...]
    standardized_residuals: tuple[float, ...]
    subset_probability: float
    eligible_for_refit: bool
    reason: str


@dataclass(frozen=True)
class IsolatedSessionUpdateArtifacts:
    output_directory: Path
    manifest_path: Path
    certification_path: Path
    candidate_par_paths: tuple[Path, ...]
    preferred_par_path: Path
    solution_tim_path: Path
    residuals_csv_path: Path
    residuals_png_path: Path


@dataclass(frozen=True)
class _RawSessionMode:
    prior_component_id: str
    contaminated_session_indices: tuple[int, ...]
    relative_integer_path: tuple[int | None, ...]
    coordinate_mean: np.ndarray
    coordinate_covariance: np.ndarray
    log_weight: float
    block_gauge_offsets: tuple[int, ...] = ()


def _difference_matrix(size: int) -> np.ndarray:
    if int(size) < 2:
        raise ValueError("Relative phase inference needs at least two inlier sessions.")
    result = np.zeros((size - 1, size), dtype=float)
    result[:, 0] = -1.0
    result[np.arange(size - 1), np.arange(1, size)] = 1.0
    return result


def _infer_phase_blocks(
    session_epochs_mjd: np.ndarray,
    config: IsolatedSessionUpdateConfig,
) -> tuple[tuple[int, ...], ...]:
    """Split sessions only where a gap is exceptional on absolute and local scales."""

    epochs = np.asarray(session_epochs_mjd, dtype=float)
    if epochs.ndim != 1 or epochs.size == 0 or not np.all(np.isfinite(epochs)):
        raise ValueError("Session epochs must be a finite nonempty vector.")
    if epochs.size == 1 or not config.enable_blockwise_long_gap_inference:
        return (tuple(range(epochs.size)),)
    gaps = np.diff(epochs)
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return (tuple(range(epochs.size)),)
    local_scale = float(np.median(positive))
    threshold = max(
        float(config.minimum_long_gap_days),
        float(config.long_gap_factor) * local_scale,
    )
    boundaries = tuple(int(value + 1) for value in np.flatnonzero(gaps >= threshold))
    if not boundaries:
        return (tuple(range(epochs.size)),)
    starts = (0, *boundaries)
    stops = (*boundaries, int(epochs.size))
    return tuple(tuple(range(start, stop)) for start, stop in zip(starts, stops))


def _enumerate_subset_modes_blockwise(
    *,
    subset: tuple[int, ...],
    candidate_session_count: int,
    intercepts: np.ndarray,
    variances: np.ndarray,
    basis: np.ndarray,
    prior_components: Sequence[tuple[str, np.ndarray, float]],
    phase_blocks: tuple[tuple[int, ...], ...],
    config: IsolatedSessionUpdateConfig,
    enforce_prior_support: bool = True,
) -> tuple[list[_RawSessionMode], bool]:
    """Propose block-internal paths, then rank complete bridge paths exactly.

    Long gaps introduce one integer gauge per disconnected phase block.  The
    block contrasts are used only to build a compact proposal portfolio.  Every
    completed path is rescored with the full-sequence likelihood, so no local
    block winner is promoted without global evidence.
    """

    session_count = int(intercepts.size)
    excluded = set(int(value) for value in subset)
    blocks = tuple(
        tuple(index for index in block if index not in excluded)
        for block in phase_blocks
    )
    blocks = tuple(block for block in blocks if block)
    inliers = np.asarray(sorted(index for block in blocks for index in block), dtype=int)
    if inliers.size < int(config.minimum_inlier_sessions) or len(blocks) < 2:
        return [], False

    internal_rows: list[np.ndarray] = []
    for block in blocks:
        anchor = int(block[0])
        for session_index in block[1:]:
            row = np.zeros(session_count, dtype=float)
            row[anchor] = -1.0
            row[int(session_index)] = 1.0
            internal_rows.append(row)
    internal_transform = np.vstack(internal_rows)
    internal_center = internal_transform @ intercepts
    internal_design = internal_transform @ basis
    session_noise = np.diag(variances)
    internal_noise = internal_transform @ session_noise @ internal_transform.T

    bridge_rows = []
    reference_anchor = int(blocks[0][0])
    for block in blocks[1:]:
        row = np.zeros(session_count, dtype=float)
        row[reference_anchor] = -1.0
        row[int(block[0])] = 1.0
        bridge_rows.append(row)
    bridge_transform = np.vstack(bridge_rows)
    bridge_center = bridge_transform @ intercepts
    bridge_design = bridge_transform @ basis
    bridge_noise = bridge_transform @ session_noise @ bridge_transform.T

    contamination_count = len(subset)
    contamination_log_prior = (
        contamination_count * math.log(float(config.contamination_probability))
        + (candidate_session_count - contamination_count)
        * math.log1p(-float(config.contamination_probability))
    )
    modes: dict[tuple[str, tuple[int | None, ...]], _RawSessionMode] = {}
    budget_exhausted = False
    for component_id, prior_covariance, component_log_prior in prior_components:
        internal_predictive = (
            internal_noise
            + internal_design @ prior_covariance @ internal_design.T
        )
        internal_result = solve_integer_least_squares(
            internal_center,
            internal_predictive,
            candidate_count=int(config.maximum_integer_candidates),
            max_visited_nodes=int(config.maximum_integer_nodes),
            return_partial_on_budget=True,
        )
        budget_exhausted = budget_exhausted or (
            internal_result.visited_nodes >= int(config.maximum_integer_nodes)
        )
        for internal_candidate in internal_result.candidates:
            internal_integer = np.asarray(
                internal_candidate.integer_vector,
                dtype=float,
            )
            proposal_mean, proposal_covariance, _ = _linear_gaussian_update(
                prior_mean=np.zeros(4, dtype=float),
                prior_covariance=prior_covariance,
                design=internal_design,
                target=internal_integer - internal_center,
                noise_covariance=internal_noise,
            )
            bridge_predictive = (
                bridge_noise
                + bridge_design @ proposal_covariance @ bridge_design.T
            )
            bridge_prediction = bridge_center + bridge_design @ proposal_mean
            bridge_result = solve_integer_least_squares(
                bridge_prediction,
                bridge_predictive,
                candidate_count=int(config.maximum_bridge_integer_candidates),
                max_visited_nodes=int(config.maximum_integer_nodes),
                return_partial_on_budget=True,
            )
            budget_exhausted = budget_exhausted or (
                bridge_result.visited_nodes >= int(config.maximum_integer_nodes)
            )
            for bridge_candidate in bridge_result.candidates:
                gauges = (
                    0,
                    *tuple(
                        int(value) for value in bridge_candidate.integer_vector
                    ),
                )
                path: list[int | None] = [None] * session_count
                cursor = 0
                for block_index, block in enumerate(blocks):
                    gauge = int(gauges[block_index])
                    path[int(block[0])] = gauge
                    for session_index in block[1:]:
                        path[int(session_index)] = gauge + int(internal_integer[cursor])
                        cursor += 1

                full_difference = _difference_matrix(inliers.size)
                full_center = full_difference @ intercepts[inliers]
                full_design = full_difference @ basis[inliers]
                full_noise = (
                    full_difference
                    @ np.diag(variances[inliers])
                    @ full_difference.T
                )
                full_integer = full_difference @ np.asarray(
                    [path[int(index)] for index in inliers],
                    dtype=float,
                )
                mean, covariance, log_evidence = _linear_gaussian_update(
                    prior_mean=np.zeros(4, dtype=float),
                    prior_covariance=prior_covariance,
                    design=full_design,
                    target=full_integer - full_center,
                    noise_covariance=full_noise,
                )
                if enforce_prior_support and not _inside_prior_support(
                    mean,
                    prior_covariance,
                    float(config.maximum_prior_mahalanobis_distance),
                ):
                    continue
                mode = _RawSessionMode(
                    prior_component_id=component_id,
                    contaminated_session_indices=subset,
                    relative_integer_path=tuple(path),
                    coordinate_mean=mean,
                    coordinate_covariance=covariance,
                    log_weight=float(
                        component_log_prior
                        + contamination_log_prior
                        + log_evidence
                    ),
                    block_gauge_offsets=tuple(gauges),
                )
                key = (component_id, mode.relative_integer_path)
                previous = modes.get(key)
                if previous is None or mode.log_weight > previous.log_weight:
                    modes[key] = mode
    return list(modes.values()), budget_exhausted


def _annual_phase_coverage(session_epochs_mjd: np.ndarray) -> float:
    phase = np.sort(np.mod(np.asarray(session_epochs_mjd, dtype=float), 365.25) / 365.25)
    if phase.size <= 1:
        return 0.0
    gaps = np.diff(np.concatenate((phase, phase[:1] + 1.0)))
    return float(max(0.0, min(1.0, 1.0 - float(np.max(gaps)))))


def _inside_prior_support(
    mean: np.ndarray,
    prior_covariance: np.ndarray,
    maximum_distance: float,
) -> bool:
    """Apply the finite support promised by APOLLO's reconstruction prior."""

    cholesky = np.linalg.cholesky(np.asarray(prior_covariance, dtype=float))
    standardized = np.linalg.solve(cholesky, np.asarray(mean, dtype=float))
    return bool(float(standardized @ standardized) <= maximum_distance**2)


def _native_solution(
    *,
    evaluator: Any,
    coordinate_mean: np.ndarray,
    coordinate_covariance: np.ndarray,
    native_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    names = tuple(str(name).upper() for name in evaluator.parameter_names)
    indices = np.asarray(
        tuple(names.index(name) for name in ISOLATED_SESSION_PARAMETER_NAMES),
        dtype=int,
    )
    base = np.asarray(evaluator.base_parameter_values, dtype=float)[indices]
    values = base + np.asarray(coordinate_mean, dtype=float) * native_steps
    jacobian = np.diag(native_steps)
    covariance = jacobian @ coordinate_covariance @ jacobian.T
    return values, 0.5 * (covariance + covariance.T)


def _mode_diagnostics(
    *,
    intercepts: np.ndarray,
    sigmas: np.ndarray,
    basis: np.ndarray,
    mode: _RawSessionMode,
    f0_hz: float,
) -> tuple[float, float, float, int]:
    inliers = np.asarray(
        [
            index
            for index, value in enumerate(mode.relative_integer_path)
            if value is not None
        ],
        dtype=int,
    )
    integers = np.asarray(
        [mode.relative_integer_path[index] for index in inliers],
        dtype=float,
    )
    residual = (
        intercepts[inliers]
        + basis[inliers] @ mode.coordinate_mean
        - integers
    )
    weights = 1.0 / np.square(sigmas[inliers])
    global_phase = float(np.sum(weights * residual) / np.sum(weights))
    centered = residual - global_phase
    chi2 = float(np.sum(np.square(centered / sigmas[inliers])))
    difference = _difference_matrix(inliers.size)
    design = difference @ basis[inliers]
    noise = difference @ np.diag(np.square(sigmas[inliers])) @ difference.T
    whitened = np.linalg.solve(np.linalg.cholesky(noise), design)
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    rank = int(
        np.sum(
            singular_values
            > max(float(singular_values[0]), 1.0)
            * np.finfo(float).eps
            * max(whitened.shape)
        )
    )
    dof = int(inliers.size - 1 - rank)
    weighted_rms_cycles = math.sqrt(float(np.sum(weights * centered**2) / np.sum(weights)))
    weighted_rms_us = float(weighted_rms_cycles / abs(f0_hz) * 1.0e6)
    return global_phase, chi2, weighted_rms_us, dof


def _session_standardized_residuals(
    *,
    intercepts: np.ndarray,
    sigmas: np.ndarray,
    basis: np.ndarray,
    mode: IsolatedSessionUpdateMode,
) -> np.ndarray:
    """Score every session against the selected global timing state.

    Marginalized sessions are compared with their nearest integer phase. This
    prevents a missing pulse wrap from being mislabeled as data contamination.
    """

    predicted = (
        np.asarray(intercepts, dtype=float)
        + np.asarray(basis, dtype=float) @ mode.coordinate_mean
        - float(mode.global_phase_cycles)
    )
    integer_path = np.empty(predicted.size, dtype=float)
    for index, value in enumerate(mode.relative_integer_path):
        integer_path[index] = (
            float(np.rint(predicted[index])) if value is None else float(value)
        )
    return (predicted - integer_path) / np.asarray(sigmas, dtype=float)


def _global_phase_sigma(
    *,
    sigmas: np.ndarray,
    basis: np.ndarray,
    inliers: np.ndarray,
    coordinate_covariance: np.ndarray,
) -> float:
    weights = 1.0 / np.square(sigmas[inliers])
    normalized = weights / np.sum(weights)
    mean_basis = normalized @ basis[inliers]
    variance = float(1.0 / np.sum(weights))
    variance += float(mean_basis @ coordinate_covariance @ mean_basis)
    return math.sqrt(max(variance, 0.0))


def _enumerate_subset_modes(
    *,
    subset: tuple[int, ...],
    candidate_session_count: int,
    intercepts: np.ndarray,
    variances: np.ndarray,
    basis: np.ndarray,
    prior_components: Sequence[tuple[str, np.ndarray, float]],
    config: IsolatedSessionUpdateConfig,
    enforce_prior_support: bool = True,
) -> tuple[list[_RawSessionMode], bool]:
    session_count = intercepts.size
    inliers = np.asarray(
        [index for index in range(session_count) if index not in subset],
        dtype=int,
    )
    if inliers.size < int(config.minimum_inlier_sessions):
        return [], False
    difference = _difference_matrix(inliers.size)
    inlier_intercepts = intercepts[inliers]
    inlier_basis = basis[inliers]
    inlier_noise = np.diag(variances[inliers])
    center = difference @ inlier_intercepts
    design = difference @ inlier_basis
    noise = difference @ inlier_noise @ difference.T
    contamination_count = len(subset)
    contamination_log_prior = (
        contamination_count * math.log(float(config.contamination_probability))
        + (candidate_session_count - contamination_count)
        * math.log1p(-float(config.contamination_probability))
    )
    modes: list[_RawSessionMode] = []
    budget_exhausted = False
    for component_id, prior_covariance, component_log_prior in prior_components:
        predictive_covariance = noise + design @ prior_covariance @ design.T
        integer_result = solve_integer_least_squares(
            center,
            predictive_covariance,
            candidate_count=int(config.maximum_integer_candidates),
            max_visited_nodes=int(config.maximum_integer_nodes),
            return_partial_on_budget=True,
        )
        budget_exhausted = budget_exhausted or (
            integer_result.visited_nodes >= int(config.maximum_integer_nodes)
        )
        for candidate in integer_result.candidates:
            relative = np.asarray(candidate.integer_vector, dtype=float)
            mean, covariance, log_evidence = _linear_gaussian_update(
                prior_mean=np.zeros(4, dtype=float),
                prior_covariance=prior_covariance,
                design=design,
                target=relative - center,
                noise_covariance=noise,
            )
            if enforce_prior_support and not _inside_prior_support(
                mean,
                prior_covariance,
                float(config.maximum_prior_mahalanobis_distance),
            ):
                continue
            path: list[int | None] = [None] * session_count
            path[int(inliers[0])] = 0
            for session_index, integer in zip(
                inliers[1:], candidate.integer_vector, strict=True
            ):
                path[int(session_index)] = int(integer)
            modes.append(
                _RawSessionMode(
                    prior_component_id=component_id,
                    contaminated_session_indices=subset,
                    relative_integer_path=tuple(path),
                    coordinate_mean=mean,
                    coordinate_covariance=covariance,
                    log_weight=float(
                        component_log_prior
                        + contamination_log_prior
                        + log_evidence
                    ),
                )
            )
    return modes, budget_exhausted


def _candidate_sessions_from_clean_mode(
    *,
    mode: _RawSessionMode,
    intercepts: np.ndarray,
    sigmas: np.ndarray,
    basis: np.ndarray,
    maximum_candidates: int,
) -> tuple[int, ...]:
    integers = np.asarray(mode.relative_integer_path, dtype=float)
    residual = intercepts + basis @ mode.coordinate_mean - integers
    weights = 1.0 / np.square(sigmas)
    residual -= float(np.sum(weights * residual) / np.sum(weights))
    order = np.argsort(np.abs(residual / sigmas))[::-1]
    return tuple(sorted(int(value) for value in order[:maximum_candidates]))


def solve_isolated_session_update(
    *,
    evaluator: Any,
    config: IsolatedSessionUpdateConfig | None = None,
) -> IsolatedSessionUpdateResult:
    """Infer a full isolated timing update without per-session JUMPs.

    The global phase is analytically removed by an integer-preserving difference
    basis. Sparse whole-session contamination is marginalized jointly with the
    relative pulse path and the RAJ/DECJ/F0/F1 posterior.
    """

    active = config or IsolatedSessionUpdateConfig()
    names = tuple(str(name).upper() for name in evaluator.parameter_names)
    if not set(ISOLATED_SESSION_PARAMETER_NAMES).issubset(names):
        raise ValueError(
            "Full isolated session updates require free RAJ, DECJ, F0, and F1."
        )
    phase_values = np.asarray(
        evaluator.raw_phase_residuals(evaluator.base_parameter_values),
        dtype=float,
    )
    noise_model = RobustPhaseNoiseModel.infer(
        phase_values=phase_values,
        reported_sigmas=evaluator.reported_phase_sigmas(),
        epochs_mjd=evaluator.observation_epochs_mjd,
        sessions=evaluator.sessions,
        config=active.phase_noise,
    )
    intercepts, session_sigmas, basis, native_steps, local_offsets = (
        _session_phase_basis(evaluator, noise_model)
    )
    session_count = intercepts.size
    if session_count < int(active.minimum_inlier_sessions):
        raise ValueError(
            "The isolated session update has fewer sessions than its minimum "
            "identifiability requirement."
        )
    epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
    session_epochs = np.asarray(
        [
            float(np.mean(epochs[np.asarray(session.toa_indices, dtype=int)]))
            for session in evaluator.sessions
        ],
        dtype=float,
    )
    phase_blocks = _infer_phase_blocks(session_epochs, active)
    blockwise_long_gap = len(phase_blocks) > 1
    historical_covariance = _previous_parameter_covariance(
        evaluator=evaluator,
        native_steps=native_steps,
        config=active,
        sigma_scale=float(active.previous_posterior_sigma_scale),
    )
    broad_covariance = np.diag(
        np.square(np.asarray(active.coordinate_prior_sigmas, dtype=float))
    )
    prior_components = (
        (
            "historical_posterior",
            historical_covariance,
            math.log(float(active.historical_posterior_probability)),
        ),
        (
            "broad_reconstruction",
            broad_covariance,
            math.log(float(active.broad_reconstruction_probability)),
        ),
    )
    variances = np.square(session_sigmas)
    enumerate_modes = (
        _enumerate_subset_modes_blockwise
        if blockwise_long_gap
        else _enumerate_subset_modes
    )
    clean_kwargs: dict[str, object] = {}
    if blockwise_long_gap:
        clean_kwargs["phase_blocks"] = phase_blocks
    clean_modes, budget_exhausted = enumerate_modes(
        subset=(),
        candidate_session_count=min(
            session_count, int(active.maximum_candidate_sessions)
        ),
        intercepts=intercepts,
        variances=variances,
        basis=basis,
        prior_components=prior_components,
        config=active,
        enforce_prior_support=False,
        **clean_kwargs,
    )
    if not clean_modes:
        raise RuntimeError("The clean isolated session model produced no phase path.")
    clean_best = max(clean_modes, key=lambda item: item.log_weight)
    if session_count <= int(active.maximum_candidate_sessions):
        candidate_sessions = tuple(range(session_count))
    else:
        candidate_sessions = _candidate_sessions_from_clean_mode(
            mode=clean_best,
            intercepts=intercepts,
            sigmas=session_sigmas,
            basis=basis,
            maximum_candidates=int(active.maximum_candidate_sessions),
        )

    subsets: list[tuple[int, ...]] = [()]
    maximum_contaminated = min(
        int(active.maximum_contaminated_sessions),
        len(candidate_sessions),
    )
    for size in range(1, maximum_contaminated + 1):
        subsets.extend(combinations(candidate_sessions, size))
    if len(subsets) > int(active.maximum_contamination_subsets):
        subsets = subsets[: int(active.maximum_contamination_subsets)]

    covariance_by_component = {
        component_id: covariance
        for component_id, covariance, _log_prior in prior_components
    }
    raw_modes = [
        mode
        for mode in clean_modes
        if _inside_prior_support(
            mode.coordinate_mean,
            covariance_by_component[mode.prior_component_id],
            float(active.maximum_prior_mahalanobis_distance),
        )
    ]
    for subset in subsets[1:]:
        subset_modes, subset_exhausted = enumerate_modes(
            subset=tuple(int(value) for value in subset),
            candidate_session_count=len(candidate_sessions),
            intercepts=intercepts,
            variances=variances,
            basis=basis,
            prior_components=prior_components,
            config=active,
            **clean_kwargs,
        )
        raw_modes.extend(subset_modes)
        budget_exhausted = budget_exhausted or subset_exhausted
    if not raw_modes:
        raise RuntimeError("The isolated session posterior produced no finite mode.")

    log_weights = np.asarray([mode.log_weight for mode in raw_modes], dtype=float)
    log_evidence = float(logsumexp(log_weights))
    weights = np.exp(log_weights - log_evidence)
    weights /= float(np.sum(weights))
    order = np.argsort(weights)[::-1]
    subset_weights: dict[tuple[int, ...], float] = {}
    path_weights: dict[tuple[int | None, ...], float] = {}
    component_weights: dict[str, float] = {}
    joint_weights: dict[tuple[tuple[int, ...], tuple[int | None, ...]], float] = {}
    session_contamination = np.zeros(session_count, dtype=float)
    for mode, weight in zip(raw_modes, weights, strict=True):
        probability = float(weight)
        subset_weights[mode.contaminated_session_indices] = (
            subset_weights.get(mode.contaminated_session_indices, 0.0)
            + probability
        )
        path_weights[mode.relative_integer_path] = (
            path_weights.get(mode.relative_integer_path, 0.0) + probability
        )
        component_weights[mode.prior_component_id] = (
            component_weights.get(mode.prior_component_id, 0.0) + probability
        )
        joint_key = (
            mode.contaminated_session_indices,
            mode.relative_integer_path,
        )
        joint_weights[joint_key] = joint_weights.get(joint_key, 0.0) + probability
        for session_index in mode.contaminated_session_indices:
            session_contamination[session_index] += probability

    retained_indices = [
        int(value) for value in order[: int(active.maximum_output_modes)]
    ]
    retained_probability = float(np.sum(weights[retained_indices]))
    f0_index = names.index("F0")
    f0_hz = float(evaluator.base_parameter_values[f0_index])
    modes = []
    for index in retained_indices:
        raw = raw_modes[index]
        native_values, native_covariance = _native_solution(
            evaluator=evaluator,
            coordinate_mean=raw.coordinate_mean,
            coordinate_covariance=raw.coordinate_covariance,
            native_steps=native_steps,
        )
        global_phase, chi2, weighted_rms_us, dof = _mode_diagnostics(
            intercepts=intercepts,
            sigmas=session_sigmas,
            basis=basis,
            mode=raw,
            f0_hz=f0_hz,
        )
        inliers = np.asarray(
            [
                session_index
                for session_index, value in enumerate(raw.relative_integer_path)
                if value is not None
            ],
            dtype=int,
        )
        modes.append(
            IsolatedSessionUpdateMode(
                prior_component_id=raw.prior_component_id,
                contaminated_session_indices=raw.contaminated_session_indices,
                relative_integer_path=raw.relative_integer_path,
                coordinate_names=ISOLATED_SESSION_COORDINATE_NAMES,
                coordinate_mean=raw.coordinate_mean,
                coordinate_covariance=raw.coordinate_covariance,
                native_parameter_names=ISOLATED_SESSION_PARAMETER_NAMES,
                native_parameter_values=native_values,
                native_parameter_covariance=native_covariance,
                global_phase_cycles=global_phase,
                global_phase_sigma_cycles=_global_phase_sigma(
                    sigmas=session_sigmas,
                    basis=basis,
                    inliers=inliers,
                    coordinate_covariance=raw.coordinate_covariance,
                ),
                inlier_session_chi2=chi2,
                inlier_session_dof=dof,
                inlier_weighted_rms_us=weighted_rms_us,
                log_weight=raw.log_weight,
                posterior_weight=float(weights[index]),
                block_gauge_offsets=raw.block_gauge_offsets,
            )
        )

    best = modes[0]
    best_inliers = np.asarray(
        [
            index
            for index, value in enumerate(best.relative_integer_path)
            if value is not None
        ],
        dtype=int,
    )
    difference = _difference_matrix(best_inliers.size)
    design = difference @ basis[best_inliers]
    noise = (
        difference
        @ np.diag(np.square(session_sigmas[best_inliers]))
        @ difference.T
    )
    whitened_design = np.linalg.solve(np.linalg.cholesky(noise), design)
    singular_values = np.linalg.svd(whitened_design, compute_uv=False)
    singular_floor = (
        float(active.rank_relative_threshold) * float(singular_values[0])
        if singular_values.size
        else math.inf
    )
    rank = int(np.sum(singular_values > singular_floor))
    condition = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank > 0
        else math.inf
    )
    native_sigma = np.sqrt(
        np.maximum(np.diag(best.native_parameter_covariance), 0.0)
    )
    posterior_correlation = np.divide(
        best.native_parameter_covariance,
        np.outer(native_sigma, native_sigma),
        out=np.zeros_like(best.native_parameter_covariance),
        where=np.outer(native_sigma, native_sigma) > 0.0,
    )
    span_days = float(np.max(session_epochs) - np.min(session_epochs))
    annual_coverage = _annual_phase_coverage(session_epochs)
    proper_motion_eligible = bool(
        span_days >= float(active.minimum_pm_span_days)
        and annual_coverage >= float(active.minimum_annual_phase_coverage)
        and rank == 4
        and condition <= float(active.maximum_design_condition_number)
    )
    if span_days < float(active.minimum_pm_span_days):
        proper_motion_reason = "observation_span_too_short"
    elif annual_coverage < float(active.minimum_annual_phase_coverage):
        proper_motion_reason = "annual_phase_coverage_too_sparse"
    elif rank < 4 or condition > float(active.maximum_design_condition_number):
        proper_motion_reason = "spin_sky_design_not_identifiable"
    else:
        proper_motion_reason = "eligible_for_separate_model_comparison"

    dominant_joint_probability = min(float(max(joint_weights.values())), 1.0)
    identifiable = bool(
        rank == 4
        and condition <= float(active.maximum_design_condition_number)
        and best.inlier_session_dof > 0
    )
    unique = bool(
        identifiable
        and not budget_exhausted
        and dominant_joint_probability
        >= float(active.phase_connection_probability_threshold)
        and best.reduced_inlier_session_chi2
        <= float(active.maximum_reduced_session_chi2)
    )
    if unique and best.prior_component_id == "historical_posterior":
        status = "unique_update_candidate"
    elif unique:
        status = "unique_reconstruction_candidate"
    else:
        status = "data_insufficient"
    standardized_residuals = _session_standardized_residuals(
        intercepts=intercepts,
        sigmas=session_sigmas,
        basis=basis,
        mode=best,
    )

    return IsolatedSessionUpdateResult(
        modes=tuple(modes),
        status=status,
        unique_solution=unique,
        dominant_joint_mode_probability=dominant_joint_probability,
        retained_probability=retained_probability,
        log_evidence=log_evidence,
        session_epochs_mjd=session_epochs,
        session_intercepts_cycles=intercepts,
        session_sigmas_cycles=session_sigmas,
        session_contamination_probabilities=session_contamination,
        session_standardized_residuals=standardized_residuals,
        contamination_subset_probabilities=tuple(
            sorted(subset_weights.items(), key=lambda item: (-item[1], item[0]))
        ),
        phase_path_probabilities=tuple(
            sorted(path_weights.items(), key=lambda item: -item[1])
        ),
        prior_component_probabilities=tuple(
            sorted(component_weights.items(), key=lambda item: (-item[1], item[0]))
        ),
        local_integer_offsets=local_offsets,
        inferred_jitter_cycles=float(noise_model.jitter_cycles),
        observation_span_days=span_days,
        annual_phase_coverage=annual_coverage,
        weighted_design_rank=rank,
        weighted_design_singular_values=singular_values,
        weighted_design_condition_number=condition,
        posterior_correlation=posterior_correlation,
        proper_motion_eligible=proper_motion_eligible,
        proper_motion_reason=proper_motion_reason,
        integer_search_budget_exhausted=budget_exhausted,
        candidate_session_indices=candidate_sessions,
        searched_contamination_subsets=len(subsets),
        phase_blocks=phase_blocks,
        long_gap_days=tuple(
            float(session_epochs[phase_blocks[index][0]] - session_epochs[phase_blocks[index - 1][-1]])
            for index in range(1, len(phase_blocks))
        ),
        blockwise_long_gap_inference_used=blockwise_long_gap,
    )


def decide_isolated_session_marginalization(
    *,
    evaluator: Any,
    result: IsolatedSessionUpdateResult,
    config: IsolatedSessionUpdateConfig | None = None,
) -> IsolatedSessionMarginalizationDecision:
    """Decide whether posterior outlier sessions may be refit without them.

    The gate combines joint subset probability, per-session contamination
    probability, nearest-integer standardized residual, search convergence, and
    a maximum removable fraction. A rejected decision remains useful as an
    audit record but must not change the scientific input data.
    """

    active = config or IsolatedSessionUpdateConfig()
    best_subset = tuple(
        sorted(
            int(value)
            for value in getattr(
                result.best_mode,
                "contaminated_session_indices",
                (),
            )
        )
    )
    session_count = len(getattr(evaluator, "sessions", ()))
    probabilities = np.asarray(
        getattr(result, "session_contamination_probabilities", ()),
        dtype=float,
    )
    standardized = np.asarray(
        getattr(result, "session_standardized_residuals", ()),
        dtype=float,
    )
    epochs = np.asarray(
        getattr(result, "session_epochs_mjd", ()),
        dtype=float,
    )
    subset_probabilities = dict(
        getattr(result, "contamination_subset_probabilities", ())
    )
    subset_probability = float(subset_probabilities.get(best_subset, 0.0))
    invalid_indices = tuple(
        index
        for index in best_subset
        if index < 0 or index >= session_count
    )

    def decision(reason: str, eligible: bool = False):
        valid_indices = tuple(
            index for index in best_subset if 0 <= index < session_count
        )
        toa_indices = tuple(
            sorted(
                int(index)
                for session_index in valid_indices
                for index in evaluator.sessions[session_index].toa_indices
            )
        )
        return IsolatedSessionMarginalizationDecision(
            session_indices=valid_indices,
            toa_indices=toa_indices,
            session_epochs_mjd=tuple(float(epochs[index]) for index in valid_indices),
            session_probabilities=tuple(
                float(probabilities[index]) for index in valid_indices
            ),
            standardized_residuals=tuple(
                float(standardized[index]) for index in valid_indices
            ),
            subset_probability=subset_probability,
            eligible_for_refit=eligible,
            reason=reason,
        )

    if not active.enable_automatic_session_marginalization:
        return decision("automatic_session_marginalization_disabled")
    if not best_subset:
        return decision("posterior_prefers_no_session_marginalization")
    if session_count <= 0 or probabilities.shape != (session_count,):
        return decision("session_probability_diagnostics_unavailable")
    if standardized.shape != (session_count,) or epochs.shape != (session_count,):
        return decision("session_residual_diagnostics_unavailable")
    if invalid_indices:
        return decision("posterior_session_index_out_of_range")
    if bool(getattr(result, "integer_search_budget_exhausted", True)):
        return decision("integer_search_budget_exhausted")
    if subset_probability < float(
        active.contamination_subset_probability_threshold
    ):
        return decision("contamination_subset_probability_below_threshold")
    if any(
        probabilities[index]
        < float(active.session_marginalization_probability_threshold)
        for index in best_subset
    ):
        return decision("session_contamination_probability_below_threshold")
    if any(
        abs(float(standardized[index]))
        < float(active.minimum_marginalized_standardized_residual)
        for index in best_subset
    ):
        return decision("session_residual_not_decisively_outlying")
    if len(best_subset) / session_count > float(
        active.maximum_marginalized_session_fraction
    ):
        return decision("marginalized_session_fraction_exceeds_limit")
    if session_count - len(best_subset) < int(active.minimum_inlier_sessions):
        return decision("too_few_sessions_remain_for_refit")
    return decision("posterior_outlier_evidence_passed", eligible=True)


def write_isolated_marginalized_tim(
    *,
    evaluator: Any,
    decision: IsolatedSessionMarginalizationDecision,
    path: Path,
) -> Path:
    """Write a filtered TIM copy while preserving the in-memory source TOAs."""

    if not decision.eligible_for_refit or not decision.toa_indices:
        raise ValueError("No evidence-qualified sessions are available to marginalize.")
    toa_count = len(evaluator.toas.table)
    retained = np.ones(toa_count, dtype=bool)
    retained[np.asarray(decision.toa_indices, dtype=int)] = False
    if not np.any(retained):
        raise ValueError("Session marginalization would remove every TOA.")
    filtered_toas = evaluator.toas[retained]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    filtered_toas.write_TOA_file(
        str(temporary),
        format="tempo2",
        include_pn=False,
        include_info=True,
        comment=(
            "APOLLO evidence-qualified session marginalization; original TIM "
            "is preserved separately"
        ),
    )
    temporary.replace(target)
    return target


def _write_residual_artifacts(
    *,
    evaluator: Any,
    result: IsolatedSessionUpdateResult,
    csv_path: Path,
    png_path: Path,
) -> None:
    mode = result.best_mode
    raw = np.asarray(
        evaluator.raw_phase_residuals(mode.native_parameter_values),
        dtype=float,
    )
    residual = raw.copy()
    inlier_toa = np.ones(raw.size, dtype=bool)
    toa_session = np.empty(raw.size, dtype=int)
    for session_index, session in enumerate(evaluator.sessions):
        indices = np.asarray(session.toa_indices, dtype=int)
        local = np.asarray(result.local_integer_offsets[session_index], dtype=float)
        residual[indices] -= local
        relative_integer = mode.relative_integer_path[session_index]
        if relative_integer is None:
            relative_integer = int(
                np.rint(float(np.median(residual[indices])) - mode.global_phase_cycles)
            )
            inlier_toa[indices] = False
        residual[indices] -= int(relative_integer)
        toa_session[indices] = session_index
    residual -= mode.global_phase_cycles
    epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
    f0_hz = float(mode.native_parameter_values[2])
    time_residual_us = residual / f0_hz * 1.0e6
    error_us = (
        np.asarray(evaluator.reported_phase_sigmas(), dtype=float)
        / abs(f0_hz)
        * 1.0e6
    )
    with Path(csv_path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "mjd",
                "phase_residual_cycles",
                "time_residual_us",
                "error_us",
                "session_index",
                "session_contamination_probability",
                "inlier_session",
            )
        )
        for index in range(raw.size):
            session_index = int(toa_session[index])
            writer.writerow(
                (
                    f"{epochs[index]:.15f}",
                    f"{residual[index]:.15g}",
                    f"{time_residual_us[index]:.15g}",
                    f"{error_us[index]:.15g}",
                    session_index,
                    f"{result.session_contamination_probabilities[session_index]:.15g}",
                    int(inlier_toa[index]),
                )
            )

    psr_parameter = next(
        (
            name
            for name in ("PSR", "PSRJ")
            if name in evaluator.model.params
        ),
        None,
    )
    psr = (
        str(evaluator.model[psr_parameter].value)
        if psr_parameter is not None
        else "Pulsar"
    )
    write_residual_series_plot(
        png_path,
        mjd=epochs,
        residual_us=time_residual_us,
        error_us=error_us,
        spin_frequency_hz=f0_hz,
        pulsar_name=psr,
        weighted_rms_us=mode.inlier_weighted_rms_us,
        reduced_chi2=mode.reduced_inlier_session_chi2,
        inlier_mask=inlier_toa,
        weighted_rms_label="Inlier WRMS",
        reduced_chi2_label="Session reduced chi2",
    )


def write_isolated_session_update_artifacts(
    *,
    evaluator: Any,
    result: IsolatedSessionUpdateResult,
    output_directory: Path,
    tim_path: Path,
    maximum_candidates: int = 16,
    marginalization_decision: IsolatedSessionMarginalizationDecision | None = None,
    source_tim_path: Path | None = None,
) -> IsolatedSessionUpdateArtifacts:
    """Write an auditable candidate without changing the original TIM data."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    candidate_dir = output / "session_update_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    selected_modes = result.modes[: max(1, int(maximum_candidates))]
    candidate_paths: list[Path] = []
    candidate_records: list[dict[str, object]] = []
    epochs = np.asarray(evaluator.observation_epochs_mjd, dtype=float)
    for rank, mode in enumerate(selected_modes):
        model = deepcopy(evaluator.model)
        uncertainties = {}
        marginal_sigmas = np.sqrt(
            np.maximum(np.diag(mode.native_parameter_covariance), 0.0)
        )
        for name, value, sigma in zip(
            mode.native_parameter_names,
            mode.native_parameter_values,
            marginal_sigmas,
            strict=True,
        ):
            _set_numeric_model_parameter_value(model, name, float(value))
            if hasattr(model[name], "frozen"):
                model[name].frozen = False
            units = getattr(model[name], "units", None)
            if units is not None and np.isfinite(sigma) and sigma > 0.0:
                uncertainties[name] = float(sigma) * units
        if uncertainties:
            model.set_param_uncertainties(uncertainties)
        for name, value in (
            ("START", float(np.min(epochs))),
            ("FINISH", float(np.max(epochs))),
            ("NTOA", int(epochs.size)),
        ):
            if name in model.params:
                model[name].value = value
        for stale in ("CHI2", "CHI2R", "TRES"):
            if stale in model.params:
                model.remove_param(stale)
        path = candidate_dir / f"candidate_{rank:03d}.par"
        write_pint_model_parfile_roundtrip_safe(model=model, path=path)
        candidate_paths.append(path)
        candidate_records.append(
            {
                "rank": rank,
                "par_file": str(path),
                "posterior_probability": mode.posterior_weight,
                "prior_component_id": mode.prior_component_id,
                "contaminated_session_indices": list(
                    mode.contaminated_session_indices
                ),
                "relative_integer_path": list(mode.relative_integer_path),
                "block_gauge_offsets": list(mode.block_gauge_offsets),
                "global_phase_cycles": mode.global_phase_cycles,
                "inlier_session_chi2": mode.inlier_session_chi2,
                "inlier_session_dof": mode.inlier_session_dof,
                "reduced_inlier_session_chi2": (
                    mode.reduced_inlier_session_chi2
                ),
                "parameters": {
                    name: {"value": float(value), "sigma": float(sigma)}
                    for name, value, sigma in zip(
                        mode.native_parameter_names,
                        mode.native_parameter_values,
                        marginal_sigmas,
                        strict=True,
                    )
                },
                "native_parameter_covariance": (
                    mode.native_parameter_covariance.tolist()
                ),
            }
        )

    preferred = output / (
        "solution.par" if result.unique_solution else "best_candidate.par"
    )
    shutil.copy2(candidate_paths[0], preferred)
    solution_tim = output / "solution.tim"
    shutil.copy2(Path(tim_path), solution_tim)
    residuals_csv = output / "residuals.csv"
    residuals_png = output / "residuals.png"
    _write_residual_artifacts(
        evaluator=evaluator,
        result=result,
        csv_path=residuals_csv,
        png_path=residuals_png,
    )
    marginalization_payload = (
        None
        if marginalization_decision is None
        else {
            "session_indices": list(marginalization_decision.session_indices),
            "toa_indices": list(marginalization_decision.toa_indices),
            "session_epochs_mjd": list(
                marginalization_decision.session_epochs_mjd
            ),
            "session_probabilities": list(
                marginalization_decision.session_probabilities
            ),
            "standardized_residuals": list(
                marginalization_decision.standardized_residuals
            ),
            "subset_probability": marginalization_decision.subset_probability,
            "eligible_for_refit": marginalization_decision.eligible_for_refit,
            "reason": marginalization_decision.reason,
        }
    )

    payload = {
        "schema_version": 1,
        "architecture": ISOLATED_SESSION_UPDATE_ARCHITECTURE,
        "status": result.status,
        "unique_solution_within_session_contamination_model": (
            result.unique_solution
        ),
        "all_toa_pint_certified": False,
        "dominant_joint_mode_probability": (
            result.dominant_joint_mode_probability
        ),
        "retained_probability": result.retained_probability,
        "preferred_par_file": str(preferred),
        "source_tim_file": str(source_tim_path or tim_path),
        "fit_tim_file": str(tim_path),
        "input_tim_was_modified": False,
        "marginalized_tim_copy_used": marginalization_decision is not None,
        "session_marginalization": marginalization_payload,
        "per_session_jumps_written": False,
        "phase_blocks": [list(block) for block in result.phase_blocks],
        "long_gap_days": list(result.long_gap_days),
        "blockwise_long_gap_inference_used": (
            result.blockwise_long_gap_inference_used
        ),
        "session_epochs_mjd": result.session_epochs_mjd.tolist(),
        "session_intercepts_cycles": result.session_intercepts_cycles.tolist(),
        "session_sigmas_cycles": result.session_sigmas_cycles.tolist(),
        "session_contamination_probabilities": (
            result.session_contamination_probabilities.tolist()
        ),
        "session_standardized_residuals": (
            result.session_standardized_residuals.tolist()
        ),
        "contamination_subset_probabilities": [
            {"session_indices": list(indices), "probability": probability}
            for indices, probability in result.contamination_subset_probabilities
        ],
        "phase_path_probabilities": [
            {"relative_integer_path": list(path), "probability": probability}
            for path, probability in result.phase_path_probabilities
        ],
        "prior_component_probabilities": [
            {"component_id": name, "probability": probability}
            for name, probability in result.prior_component_probabilities
        ],
        "diagnostics": {
            "inferred_jitter_cycles": result.inferred_jitter_cycles,
            "observation_span_days": result.observation_span_days,
            "annual_phase_coverage": result.annual_phase_coverage,
            "weighted_design_rank": result.weighted_design_rank,
            "weighted_design_singular_values": (
                result.weighted_design_singular_values.tolist()
            ),
            "weighted_design_condition_number": (
                result.weighted_design_condition_number
            ),
            "posterior_correlation": result.posterior_correlation.tolist(),
            "proper_motion_eligible": result.proper_motion_eligible,
            "proper_motion_reason": result.proper_motion_reason,
            "integer_search_budget_exhausted": (
                result.integer_search_budget_exhausted
            ),
            "candidate_session_indices": list(result.candidate_session_indices),
                "searched_contamination_subsets": (
                    result.searched_contamination_subsets
                ),
                "phase_blocks": [list(block) for block in result.phase_blocks],
                "long_gap_days": list(result.long_gap_days),
                "blockwise_long_gap_inference_used": (
                    result.blockwise_long_gap_inference_used
                ),
        },
        "candidates": candidate_records,
    }
    update_manifest = output / "isolated_session_update.json"
    update_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    certification = output / "certification.json"
    certification.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": ISOLATED_SESSION_UPDATE_ARCHITECTURE,
                "accepted": result.unique_solution,
                "pint_certified": False,
                "certification_scope": "session_posterior_update_candidate",
                "status": result.status,
                "solution_claim": {
                    "status": result.status,
                    "interpretation": (
                        "The basic RAJ/DECJ/F0/F1 update is unique on the retained "
                        "TOAs after evidence-qualified whole-session marginalization."
                        if result.unique_solution
                        and marginalization_decision is not None
                        else "The basic RAJ/DECJ/F0/F1 update is unique within the "
                        "reported sparse whole-session contamination model. "
                        "Contaminated sessions remain in the TIM file and prevent "
                        "an all-TOA ordinary PINT chi-square claim."
                        if result.unique_solution
                        else "The retained posterior does not support a unique "
                        "basic isolated timing update."
                    ),
                },
                "chi2": result.best_mode.inlier_session_chi2,
                "reduced_chi2": result.best_mode.reduced_inlier_session_chi2,
                "weighted_rms_us": result.best_mode.inlier_weighted_rms_us,
                "artifacts": {
                    "par": str(preferred),
                    "tim": str(solution_tim),
                    "residuals_csv": str(residuals_csv),
                    "residuals_png": str(residuals_png),
                    "isolated_session_update": str(update_manifest),
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    audit = output / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": ISOLATED_SESSION_UPDATE_ARCHITECTURE,
                "toa_count": int(epochs.size),
                "session_count": len(evaluator.sessions),
                "fit_parameters": list(ISOLATED_SESSION_PARAMETER_NAMES),
                "parameter_expansion": "disabled",
                "proper_motion_released": False,
                "high_order_spin_released": False,
                "glitch_released": False,
                "per_session_jumps_written": False,
                "input_tim_was_modified": False,
                "source_tim_file": str(source_tim_path or tim_path),
                "fit_tim_file": str(tim_path),
                "marginalized_tim_copy_used": marginalization_decision is not None,
                "session_marginalization": marginalization_payload,
                "blockwise_long_gap_inference_used": (
                    result.blockwise_long_gap_inference_used
                ),
                "phase_blocks": [list(block) for block in result.phase_blocks],
                "long_gap_days": list(result.long_gap_days),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    artifact_manifest = output / "artifact_manifest.json"
    fit_tim = Path(tim_path).resolve()
    extra_fit_tim = (
        (fit_tim,)
        if fit_tim.parent == output.resolve() and fit_tim != solution_tim.resolve()
        else ()
    )
    artifact_paths = (
        preferred,
        solution_tim,
        residuals_csv,
        residuals_png,
        update_manifest,
        certification,
        audit,
        *extra_fit_tim,
        *candidate_paths,
    )
    import hashlib

    artifact_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "architecture": ISOLATED_SESSION_UPDATE_ARCHITECTURE,
                "artifacts": {
                    path.name: {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in artifact_paths
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return IsolatedSessionUpdateArtifacts(
        output_directory=output,
        manifest_path=update_manifest,
        certification_path=certification,
        candidate_par_paths=tuple(candidate_paths),
        preferred_par_path=preferred,
        solution_tim_path=solution_tim,
        residuals_csv_path=residuals_csv,
        residuals_png_path=residuals_png,
    )


__all__ = [
    "ISOLATED_SESSION_UPDATE_ARCHITECTURE",
    "IsolatedSessionUpdateArtifacts",
    "IsolatedSessionUpdateConfig",
    "IsolatedSessionMarginalizationDecision",
    "IsolatedSessionUpdateMode",
    "IsolatedSessionUpdateResult",
    "decide_isolated_session_marginalization",
    "solve_isolated_session_update",
    "write_isolated_marginalized_tim",
    "write_isolated_session_update_artifacts",
]
