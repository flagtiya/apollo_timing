from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import qmc

from apollo_timing.inference.active_timing_contracts import ActiveTimingBranch
from apollo_timing.inference.integer_least_squares import solve_integer_least_squares
from apollo_timing.inference.multimode_session_lattice import (
    MultimodeSessionLatticeConfig,
    generate_global_phase_path_refresh_branches,
    score_session_integer_path,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    integer_session_offsets_to_gap_wraps,
)


DD_REQUIRED_PARAMETERS = ("F0", "F1", "A1", "ECC", "T0", "PB", "OM", "OMDOT")


def _wrap_angle_near(angle: float, reference: float) -> float:
    return float(reference + (angle - reference + math.pi) % (2.0 * math.pi) - math.pi)


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1.0e-12, 1.0 - 1.0e-12))
    return float(math.log(clipped / (1.0 - clipped)))


def _expit(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return float(1.0 / (1.0 + exponential))
    exponential = math.exp(value)
    return float(exponential / (1.0 + exponential))


@dataclass(frozen=True)
class DDSearchCoordinateSystem:
    """A bounded DD search chart centered at a physically meaningful epoch."""

    parameter_names: tuple[str, ...]
    center_values: np.ndarray
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    physical_scales: np.ndarray
    search_center: np.ndarray
    search_lower: np.ndarray
    search_upper: np.ndarray
    search_scales: np.ndarray
    pepoch_mjd: float
    reference_mjd: float
    orbital_cycle_index: int

    def _index(self, name: str) -> int:
        return self.parameter_names.index(name)

    def to_search(self, parameter_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(parameter_values, dtype=float)
        if values.shape != self.center_values.shape or not np.all(np.isfinite(values)):
            raise ValueError("DD parameter values must be finite and dimensionally aligned.")

        result = values.copy()
        f0_index = self._index("F0")
        f1_index = self._index("F1")
        a1_index = self._index("A1")
        ecc_index = self._index("ECC")
        t0_index = self._index("T0")
        pb_index = self._index("PB")
        om_index = self._index("OM")
        omdot_index = self._index("OMDOT")

        pb = float(values[pb_index])
        a1 = float(values[a1_index])
        eccentricity = float(values[ecc_index])
        if pb <= 0.0 or a1 <= 0.0 or not 0.0 < eccentricity < 1.0:
            raise ValueError("DD search coordinates require PB > 0, A1 > 0, and 0 < ECC < 1.")

        spin_delta_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        result[f0_index] = values[f0_index] + values[f1_index] * spin_delta_seconds
        result[a1_index] = math.log(a1)
        result[ecc_index] = _logit(eccentricity)
        orbit_angle = 2.0 * math.pi * (self.reference_mjd - values[t0_index]) / pb
        result[t0_index] = _wrap_angle_near(
            orbit_angle,
            float(self.search_center[t0_index]),
        )
        result[pb_index] = math.log(pb)
        om_reference = math.radians(
            values[om_index]
            + values[omdot_index]
            * (self.reference_mjd - values[t0_index])
            / 365.25
        )
        result[om_index] = _wrap_angle_near(
            om_reference,
            float(self.search_center[om_index]),
        )
        return result

    def to_physical(self, search_values: Sequence[float]) -> np.ndarray:
        search = np.asarray(search_values, dtype=float)
        if search.shape != self.search_center.shape or not np.all(np.isfinite(search)):
            raise ValueError("DD search values must be finite and dimensionally aligned.")

        result = search.copy()
        f0_index = self._index("F0")
        f1_index = self._index("F1")
        a1_index = self._index("A1")
        ecc_index = self._index("ECC")
        t0_index = self._index("T0")
        pb_index = self._index("PB")
        om_index = self._index("OM")
        omdot_index = self._index("OMDOT")

        pb = math.exp(float(search[pb_index]))
        t0 = self.reference_mjd - (
            self.orbital_cycle_index + float(search[t0_index]) / (2.0 * math.pi)
        ) * pb
        spin_delta_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        result[f0_index] = search[f0_index] - search[f1_index] * spin_delta_seconds
        result[a1_index] = math.exp(float(search[a1_index]))
        result[ecc_index] = _expit(float(search[ecc_index]))
        result[t0_index] = t0
        result[pb_index] = pb
        result[om_index] = math.degrees(float(search[om_index])) - (
            search[omdot_index] * (self.reference_mjd - t0) / 365.25
        )
        return result

    def normalize(self, search_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(search_values, dtype=float)
        return (values - self.search_center) / self.search_scales

    def denormalize(self, normalized_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(normalized_values, dtype=float)
        return self.search_center + self.search_scales * values

    @property
    def normalized_lower(self) -> np.ndarray:
        return self.normalize(self.search_lower)

    @property
    def normalized_upper(self) -> np.ndarray:
        return self.normalize(self.search_upper)

    def sample_prior(self, count: int, *, seed: int) -> np.ndarray:
        requested = max(int(count), 0)
        if requested == 0:
            return np.zeros((0, len(self.parameter_names)), dtype=float)
        exponent = int(math.ceil(math.log2(requested)))
        unit = qmc.Sobol(
            d=len(self.parameter_names),
            scramble=True,
            seed=int(seed),
        ).random_base2(exponent)[:requested]
        physical = self.physical_lower + unit * (self.physical_upper - self.physical_lower)
        return np.vstack([self.to_search(row) for row in physical])


@dataclass(frozen=True)
class SpinSupportAudit:
    """Record whether the spin box is closed at the search reference epoch."""

    maximum_time_lever_arm_seconds: float
    f0_half_width: float
    initial_f1_half_width: float
    minimum_f1_half_width: float
    final_f1_half_width: float
    expanded: bool
    configured_minimum_f1_half_width: float = 0.0
    rule: str = "maximum_of_phase_reach_and_configured_recoverability_floor"


def expand_reference_epoch_spin_support(
    *,
    parameter_names: Sequence[str],
    center_values: Sequence[float],
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    observation_epochs_mjd: Sequence[float],
    reference_mjd: float,
    recoverability_f1_half_width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, SpinSupportAudit]:
    """Make F0/F1 support consistent with phase reach over the data span.

    A reference-frequency half-width ``W_F0`` spans ``W_F0 * L`` cycles over
    a lever arm ``L``. Matching that reach with the quadratic spin term needs
    ``0.5 * W_F1 * L**2 >= W_F0 * L``, hence ``W_F1 >= 2 W_F0 / L``.
    A caller may declare a broader symmetric recoverability floor. Both rules
    use only the input support and observation epochs, never a source class or
    fitted solution.
    """

    names = tuple(str(name).upper() for name in parameter_names)
    if "F0" not in names or "F1" not in names:
        raise ValueError("Reference-epoch spin support requires F0 and F1.")
    center = np.asarray(center_values, dtype=float)
    lower = np.asarray(lower_values, dtype=float).copy()
    upper = np.asarray(upper_values, dtype=float).copy()
    epochs = np.asarray(observation_epochs_mjd, dtype=float)
    expected_shape = (len(names),)
    if (
        center.shape != expected_shape
        or lower.shape != expected_shape
        or upper.shape != expected_shape
    ):
        raise ValueError("Spin support arrays must align with parameter names.")
    if (
        epochs.ndim != 1
        or epochs.size == 0
        or not np.all(np.isfinite(epochs))
        or not np.isfinite(float(reference_mjd))
    ):
        raise ValueError("Observation epochs and reference epoch must be finite.")
    if not np.all(np.isfinite(center) & np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("Spin support arrays must be finite.")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Each spin support interval must contain its center.")

    maximum_lever_arm_seconds = float(
        np.max(np.abs(epochs - float(reference_mjd))) * 86400.0
    )
    if maximum_lever_arm_seconds <= 0.0:
        raise ValueError("Spin support requires a positive observation time lever arm.")

    f0_index = names.index("F0")
    f1_index = names.index("F1")
    f0_half_width = max(
        float(center[f0_index] - lower[f0_index]),
        float(upper[f0_index] - center[f0_index]),
    )
    initial_f1_half_width = max(
        float(center[f1_index] - lower[f1_index]),
        float(upper[f1_index] - center[f1_index]),
    )
    configured_minimum = float(recoverability_f1_half_width)
    if not np.isfinite(configured_minimum) or configured_minimum < 0.0:
        raise ValueError("The recoverability F1 half-width must be finite and non-negative.")
    phase_reach_half_width = 2.0 * f0_half_width / maximum_lever_arm_seconds
    minimum_f1_half_width = max(phase_reach_half_width, configured_minimum)
    final_f1_half_width = max(initial_f1_half_width, minimum_f1_half_width)
    tolerance = 16.0 * np.finfo(float).eps * max(final_f1_half_width, 1.0)
    expanded = bool(final_f1_half_width > initial_f1_half_width + tolerance)
    if expanded:
        lower[f1_index] = center[f1_index] - final_f1_half_width
        upper[f1_index] = center[f1_index] + final_f1_half_width

    return (
        lower,
        upper,
        SpinSupportAudit(
            maximum_time_lever_arm_seconds=maximum_lever_arm_seconds,
            f0_half_width=f0_half_width,
            initial_f1_half_width=initial_f1_half_width,
            minimum_f1_half_width=minimum_f1_half_width,
            final_f1_half_width=final_f1_half_width,
            expanded=expanded,
            configured_minimum_f1_half_width=configured_minimum,
        ),
    )


def build_dd_search_coordinate_system(
    *,
    parameter_names: Sequence[str],
    center_values: Sequence[float],
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    pepoch_mjd: float,
    reference_mjd: float,
    bound_sample_count: int = 4096,
    seed: int = 0,
) -> DDSearchCoordinateSystem:
    names = tuple(str(name).upper() for name in parameter_names)
    if set(names) != set(DD_REQUIRED_PARAMETERS) or len(names) != len(DD_REQUIRED_PARAMETERS):
        raise ValueError(
            "The information-ordered DD chart requires exactly: "
            + ", ".join(DD_REQUIRED_PARAMETERS)
        )
    center = np.asarray(center_values, dtype=float)
    lower = np.asarray(lower_values, dtype=float)
    upper = np.asarray(upper_values, dtype=float)
    if center.shape != lower.shape or center.shape != upper.shape or center.shape != (len(names),):
        raise ValueError("DD prior arrays must align with the parameter names.")
    if not np.all(np.isfinite(center) & np.isfinite(lower) & np.isfinite(upper)):
        raise ValueError("DD prior arrays must be finite.")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Each DD prior must contain its center and have positive width.")

    t0_index = names.index("T0")
    pb_index = names.index("PB")
    om_index = names.index("OM")
    center_pb = float(center[pb_index])
    center_orbit_cycles = (float(reference_mjd) - float(center[t0_index])) / center_pb
    center_orbit_angle = float(
        (2.0 * math.pi * center_orbit_cycles + math.pi) % (2.0 * math.pi) - math.pi
    )
    orbital_cycle_index = int(round(center_orbit_cycles - center_orbit_angle / (2.0 * math.pi)))
    center_om_reference = math.radians(
        center[om_index]
        + center[names.index("OMDOT")]
        * (float(reference_mjd) - center[t0_index])
        / 365.25
    )

    placeholder = DDSearchCoordinateSystem(
        parameter_names=names,
        center_values=center.copy(),
        physical_lower=lower.copy(),
        physical_upper=upper.copy(),
        physical_scales=(upper - lower) / 2.0,
        search_center=center.copy(),
        search_lower=np.zeros_like(center),
        search_upper=np.ones_like(center),
        search_scales=np.ones_like(center),
        pepoch_mjd=float(pepoch_mjd),
        reference_mjd=float(reference_mjd),
        orbital_cycle_index=orbital_cycle_index,
    )
    search_center = center.copy()
    search_center[names.index("F0")] = center[names.index("F0")] + center[
        names.index("F1")
    ] * (float(reference_mjd) - float(pepoch_mjd)) * 86400.0
    search_center[names.index("A1")] = math.log(center[names.index("A1")])
    search_center[names.index("ECC")] = _logit(center[names.index("ECC")])
    search_center[t0_index] = center_orbit_angle
    search_center[pb_index] = math.log(center_pb)
    search_center[om_index] = center_om_reference
    placeholder = replace(placeholder, search_center=search_center)

    requested = max(int(bound_sample_count), 256)
    exponent = int(math.ceil(math.log2(requested)))
    unit = qmc.Sobol(d=len(names), scramble=True, seed=int(seed)).random_base2(exponent)[
        :requested
    ]
    physical_samples = lower + unit * (upper - lower)
    transformed = np.vstack(
        [placeholder.to_search(row) for row in np.vstack((physical_samples, center))]
    )
    search_lower = np.min(transformed, axis=0)
    search_upper = np.max(transformed, axis=0)

    for periodic_index in (t0_index, om_index):
        span = search_upper[periodic_index] - search_lower[periodic_index]
        if span > 1.8 * math.pi:
            search_lower[periodic_index] = search_center[periodic_index] - math.pi
            search_upper[periodic_index] = search_center[periodic_index] + math.pi
    search_scales = np.maximum(
        np.maximum(search_center - search_lower, search_upper - search_center),
        np.finfo(float).eps,
    )
    return replace(
        placeholder,
        search_lower=search_lower,
        search_upper=search_upper,
        search_scales=search_scales,
    )


@dataclass(frozen=True)
class SlopeCalibration:
    log_scale_map: float
    scale_map: float
    log_marginal_likelihood: float


def conditional_slope_log_likelihood(
    slope_chi2: float,
    observation_count: int,
    *,
    scale: float,
    log_scale_prior_sigma: float,
) -> float:
    chi2 = max(float(slope_chi2), 0.0)
    count = max(int(observation_count), 0)
    active_scale = max(float(scale), 1.0e-12)
    sigma = max(float(log_scale_prior_sigma), 1.0e-6)
    log_scale = math.log(active_scale)
    return float(
        -count * log_scale
        - 0.5 * chi2 / (active_scale * active_scale)
        - 0.5 * (log_scale / sigma) ** 2
        - math.log(sigma * math.sqrt(2.0 * math.pi))
    )


def calibrated_slope_log_likelihood(
    slope_chi2: float,
    observation_count: int,
    *,
    log_scale_prior_sigma: float,
) -> SlopeCalibration:
    chi2 = max(float(slope_chi2), 0.0)
    count = max(int(observation_count), 0)
    sigma = max(float(log_scale_prior_sigma), 1.0e-6)
    if count == 0:
        return SlopeCalibration(0.0, 1.0, 0.0)

    def score(log_scale: float) -> float:
        return conditional_slope_log_likelihood(
            chi2,
            count,
            scale=math.exp(log_scale),
            log_scale_prior_sigma=sigma,
        )

    def derivative(log_scale: float) -> float:
        return -count + chi2 * math.exp(-2.0 * log_scale) - log_scale / (sigma * sigma)

    lower = -20.0
    upper = 20.0
    derivative_lower = derivative(lower)
    derivative_upper = derivative(upper)
    if derivative_lower <= 0.0:
        log_scale_map = lower
    elif derivative_upper >= 0.0:
        log_scale_map = upper
    else:
        log_scale_map = float(brentq(derivative, lower, upper))
    curvature = (
        2.0 * chi2 * math.exp(-2.0 * log_scale_map) + 1.0 / (sigma * sigma)
    )
    log_marginal = score(log_scale_map) + 0.5 * math.log(2.0 * math.pi / curvature)
    return SlopeCalibration(
        log_scale_map=log_scale_map,
        scale_map=float(math.exp(log_scale_map)),
        log_marginal_likelihood=float(log_marginal),
    )


@dataclass(frozen=True)
class SlopeInformationAudit:
    observation_count: int
    singular_values: np.ndarray
    numerical_rank: int
    null_dimension: int
    condition_number: float
    prior_to_posterior_variance_ratios: np.ndarray


@dataclass(frozen=True)
class OrbitPosteriorMode:
    parameter_values: np.ndarray
    search_values: np.ndarray
    normalized_search_values: np.ndarray
    scaled_covariance: np.ndarray
    slope_chi2: float
    calibration_scale: float
    log_posterior: float
    laplace_log_evidence: float
    posterior_mass: float
    information_audit: SlopeInformationAudit
    source_start_index: int
    nfev: int
    success: bool


@dataclass(frozen=True)
class OrbitPosteriorResult:
    modes: tuple[OrbitPosteriorMode, ...]
    evaluated_prior_count: int
    failed_prior_count: int
    optimized_start_count: int
    observable_slope_count: int
    orbital_alias_count: int = 0
    structured_prior_count: int = 0
    protected_alias_start_count: int = 0


@dataclass(frozen=True)
class InformationOrderedBinarySearchConfig:
    prior_sample_count: int = 256
    optimization_start_count: int = 24
    retained_mode_count: int = 12
    exploitation_fraction: float = 0.6
    minimum_mode_separation: float = 0.08
    max_nfev: int = 60
    finite_difference_fraction: float = 1.0e-3
    log_slope_scale_prior_sigma: float = 0.7
    random_seed: int = 0
    max_prior_expansion_rounds: int = 3
    prior_expansion_factor: float = 4.0
    prior_boundary_fraction: float = 0.9
    slope_reduced_chi2_target: float = 4.0
    structured_orbit_alias_enabled: bool = True
    orbital_alias_oversampling: float = 1.0
    orbital_phase_sample_count: int = 8
    maximum_orbital_alias_count: int = 48
    protected_starts_per_orbital_alias: int = 1
    focused_orbital_alias_count: int = 4
    focused_starts_per_orbital_alias: int = 8
    maximum_structured_anchor_bases: int = 4
    constrain_orbital_alias_optimization: bool = True


@dataclass(frozen=True)
class AdaptiveOrbitPosteriorResult:
    coordinates: DDSearchCoordinateSystem
    posterior: OrbitPosteriorResult
    expansion_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BayesianContinuationConfig:
    max_particles: int = 16
    protected_particles_per_mode: int = 1
    ess_target_fraction: float = 0.7
    minimum_beta_step: float = 1.0e-5
    maximum_beta_steps: int = 24
    trust_radius: float = 1.0
    max_nfev_per_move: int = 24
    integer_candidates_per_particle: int = 4
    reoptimized_integer_candidates_per_particle: int = 2
    finite_difference_fraction: float = 1.0e-4
    log_slope_scale_prior_sigma: float = 0.7
    maximum_orbit_slope_chi2_delta: float = 100.0
    maximum_beta_increment: float = 0.1
    maximum_beta_growth_factor: float = 2.0
    beta_one_refinement_steps: int = 3
    maximum_particles_per_mode: int = 4
    ess_representatives_per_mode: int = 2


@dataclass(frozen=True)
class PhasePathParticle:
    parameter_values: np.ndarray
    integer_offsets: tuple[int, ...]
    source_branch_id: str
    source_mode_index: int
    calibration_scale: float
    log_weight: float
    slope_chi2: float
    orthogonal_chi2: float
    ambiguity_chi2: float
    prior_chi2: float
    path_update_count: int
    nfev: int
    success: bool
    path_proposal_source: str


@dataclass(frozen=True)
class PosteriorPredictivePathProposal:
    integer_offsets: tuple[int, ...]
    normalized_start: np.ndarray
    squared_mahalanobis: float
    source: str


@dataclass(frozen=True)
class BayesianContinuationResult:
    particles: tuple[PhasePathParticle, ...]
    branches: tuple[ActiveTimingBranch, ...]
    beta_history: tuple[dict[str, Any], ...]
    reached_beta_one: bool


@dataclass(frozen=True)
class DelayedIntegerPhaseConfig:
    """Budget for entropy-ordered phase assimilation before integer commitment."""

    initial_relative_session_count: int = 3
    session_batch_size: int = 3
    integer_candidates_per_mode: int = 6
    max_modes: int = 24
    protected_modes_per_parent: int = 1
    maximum_modes_per_integer_path: int = 2
    trust_radius: float = 1.0
    max_nfev_per_candidate: int = 32
    finite_difference_fraction: float = 1.0e-4
    minimum_mode_separation: float = 0.02
    max_integer_visited_nodes: int = 2_000_000
    integer_decimal_precision: int = 64
    maximum_relative_session_count: int | None = None
    wrapped_phase_assimilation_enabled: bool = True
    wrapped_phase_temperature_schedule: tuple[float, ...] = (
        0.35,
        0.20,
        0.10,
        0.05,
        0.025,
        0.0125,
        0.00625,
        0.003,
    )
    wrapped_phase_trust_radius: float = 1.5
    wrapped_phase_max_nfev: int = 32
    wrapped_phase_sigma_points_per_mode: int = 9
    wrapped_phase_sigma_point_radius: float = 2.0
    wrapped_phase_orbital_alias_count: int = 16
    wrapped_phase_orbital_phase_points: int = 2


@dataclass(frozen=True)
class DelayedIntegerPhaseMode:
    parameter_values: np.ndarray
    normalized_search_values: np.ndarray
    integer_offsets_from_anchor: tuple[int, ...]
    source_orbit_mode_index: int
    parent_mode_index: int
    objective_chi2: float
    shape_chi2: float
    ambiguity_chi2: float
    prior_chi2: float
    laplace_log_evidence: float
    nfev: int
    success: bool
    wrapped_phase_chi2: float = 0.0


@dataclass(frozen=True)
class DelayedIntegerPhaseResult:
    modes: tuple[DelayedIntegerPhaseMode, ...]
    branches: tuple[ActiveTimingBranch, ...]
    anchor_session_index: int
    relative_session_order: tuple[int, ...]
    stage_history: tuple[dict[str, Any], ...]
    reached_all_sessions: bool
    wrapped_phase_history: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class PhaseInformationEdge:
    parent_session_index: int
    child_session_index: int
    predictive_variance: float


def _profile_standardized_slopes(profile: Any) -> np.ndarray:
    slopes = np.asarray(profile.standardized_session_slopes, dtype=float)
    if slopes.ndim != 1 or not np.all(np.isfinite(slopes)):
        raise ValueError("Observable standardized session slopes must be finite.")
    if slopes.size == 0:
        raise ValueError("No source session has observable signed slope information.")
    return slopes


def _effective_sample_size(log_weights: Sequence[float]) -> float:
    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    shifted = np.where(finite, values - np.max(values[finite]), -math.inf)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    squared = float(weights @ weights)
    return total * total / squared if total > 0.0 and squared > 0.0 else 0.0


def select_adaptive_beta(
    *,
    current_beta: float,
    remaining_chi2: Sequence[float],
    log_weights: Sequence[float] | None = None,
    ess_target_fraction: float = 0.7,
    minimum_step: float = 1.0e-5,
) -> float:
    """Select the next likelihood temperature from an incremental ESS target."""

    beta = min(max(float(current_beta), 0.0), 1.0)
    if beta >= 1.0:
        return 1.0
    chi2 = np.asarray(remaining_chi2, dtype=float)
    if chi2.ndim != 1 or chi2.size == 0 or not np.all(np.isfinite(chi2)):
        raise ValueError("Continuation chi-square values must be finite and non-empty.")
    base_weights = (
        np.zeros(chi2.size, dtype=float)
        if log_weights is None
        else np.asarray(log_weights, dtype=float)
    )
    if base_weights.shape != chi2.shape:
        raise ValueError("Continuation weights must match the particle count.")
    maximum_delta = 1.0 - beta
    target = min(max(float(ess_target_fraction), 0.0), 1.0) * chi2.size

    def ess(delta: float) -> float:
        return _effective_sample_size(base_weights - 0.5 * float(delta) * chi2)

    if ess(maximum_delta) >= target:
        return 1.0
    floor_delta = min(max(float(minimum_step), 0.0), maximum_delta)
    if ess(floor_delta) <= target:
        return float(beta + floor_delta)
    lower = floor_delta
    upper = maximum_delta
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if ess(midpoint) < target:
            upper = midpoint
        else:
            lower = midpoint
    return float(min(beta + lower, 1.0))


def select_mode_aware_beta(
    *,
    current_beta: float,
    remaining_chi2: Sequence[float],
    mode_labels: Sequence[int],
    ess_target_fraction: float = 0.7,
    minimum_step: float = 1.0e-5,
    maximum_increment: float = 0.1,
    maximum_growth_factor: float | None = None,
    representatives_per_mode: int = 1,
) -> float:
    """Select beta from within-mode ESS while preserving between-mode evidence."""

    chi2 = np.asarray(remaining_chi2, dtype=float)
    labels = np.asarray(mode_labels, dtype=int)
    if labels.shape != chi2.shape:
        raise ValueError("Continuation mode labels must match the particle count.")
    proposals = []
    for label in np.unique(labels):
        selected = np.sort(chi2[labels == label])[
            : max(int(representatives_per_mode), 1)
        ]
        proposals.append(
            select_adaptive_beta(
                current_beta=current_beta,
                remaining_chi2=selected,
                log_weights=np.zeros(selected.size, dtype=float),
                ess_target_fraction=ess_target_fraction,
                minimum_step=minimum_step,
            )
        )
    cap = min(float(current_beta) + max(float(maximum_increment), 0.0), 1.0)
    if maximum_growth_factor is not None:
        growth = max(float(maximum_growth_factor), 1.0)
        growth_cap = (
            max(float(minimum_step), 0.0)
            if float(current_beta) <= 0.0
            else max(
                float(current_beta) + max(float(minimum_step), 0.0),
                float(current_beta) * growth,
            )
        )
        cap = min(cap, growth_cap)
    return float(min((*proposals, cap)))


def _relative_phase_covariance(profile: Any) -> np.ndarray:
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    if variances.ndim != 1 or variances.size != int(profile.session_count):
        raise ValueError("Session offset variances must match the session count.")
    if not np.all(np.isfinite(variances) & (variances > 0.0)):
        raise ValueError("Session offset variances must be finite and positive.")
    if variances.size <= 1:
        return np.zeros((0, 0), dtype=float)
    covariance = np.diag(variances[1:]) + float(variances[0])
    jitter = max(float(np.trace(covariance)) * 1.0e-14, 1.0e-24)
    return covariance + np.eye(covariance.shape[0]) * jitter


def _continuation_components(
    *,
    profile: Any,
    integer_offsets: Sequence[int],
    normalized_search_values: np.ndarray,
    calibration_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    slope = _profile_standardized_slopes(profile) / max(
        float(calibration_scale), 1.0e-12
    )
    orthogonal = np.asarray(profile.normalized_orthogonal_residuals, dtype=float)
    if orthogonal.ndim != 1 or not np.all(np.isfinite(orthogonal)):
        raise ValueError("Orthogonal session residuals must be finite.")
    relative = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0],
        dtype=float,
    )
    integers = np.asarray(integer_offsets, dtype=float)
    if integers.shape != relative.shape:
        raise ValueError("Continuation integer path must match the session count.")
    covariance = _relative_phase_covariance(profile)
    phase = (
        np.linalg.solve(np.linalg.cholesky(covariance), relative - integers)
        if covariance.size
        else np.zeros(0, dtype=float)
    )
    prior = np.asarray(normalized_search_values, dtype=float)
    return slope, orthogonal, phase, prior


def _candidate_integer_paths(
    profile: Any,
    current_path: Sequence[int],
    *,
    candidate_count: int,
) -> tuple[tuple[int, ...], ...]:
    relative = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0],
        dtype=float,
    )
    if relative.size == 0:
        return ((),)
    covariance = _relative_phase_covariance(profile)
    result = solve_integer_least_squares(
        relative,
        covariance,
        candidate_count=max(int(candidate_count), 1),
    )
    candidates = {tuple(int(value) for value in current_path)}
    candidates.update(
        tuple(int(value) for value in candidate.integer_vector)
        for candidate in result.candidates
    )
    return tuple(
        sorted(
        candidates,
        key=lambda path: score_session_integer_path(profile, path)[2],
        )[: max(int(candidate_count), 1)]
    )


def posterior_predictive_integer_path_proposals(
    *,
    profile: Any,
    current_path: Sequence[int],
    normalized_center: Sequence[float],
    residual_jacobian: Sequence[Sequence[float]],
    residual_at_center: Sequence[float] | None = None,
    slope_residual_count: int,
    orthogonal_residual_count: int,
    phase_residual_count: int,
    beta: float,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    candidate_count: int,
) -> tuple[PosteriorPredictivePathProposal, ...]:
    """Enumerate integer paths under the current continuous posterior.

    Phase covariance combines tempered measurement covariance with phase
    displacement from uncertainty in the non-phase posterior. If the residual
    at the linearization center is supplied, the predictive phase mean also
    includes displacement from the non-phase posterior mean. The lattice can
    then propose coherent distant paths reachable by continuous parameter moves,
    beyond the nearest integers at a single point.
    """

    center = np.asarray(normalized_center, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    jacobian = np.asarray(residual_jacobian, dtype=float)
    center_residual = (
        np.zeros(jacobian.shape[0], dtype=float)
        if residual_at_center is None
        else np.asarray(residual_at_center, dtype=float)
    )
    path = tuple(int(value) for value in current_path)
    phase_count = max(int(phase_residual_count), 0)
    slope_count = max(int(slope_residual_count), 0)
    orthogonal_count = max(int(orthogonal_residual_count), 0)
    target_count = max(int(candidate_count), 1)
    parameter_count = center.size
    if (
        center.ndim != 1
        or lower.shape != center.shape
        or upper.shape != center.shape
        or jacobian.ndim != 2
        or jacobian.shape[1] != parameter_count
        or center_residual.shape != (jacobian.shape[0],)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(center_residual))
    ):
        raise ValueError("Posterior-predictive lattice inputs are not aligned.")
    if len(path) != phase_count:
        raise ValueError("Integer path must match the relative phase dimension.")
    if phase_count == 0:
        return (
            PosteriorPredictivePathProposal(
                integer_offsets=(),
                normalized_start=np.clip(center, lower, upper),
                squared_mahalanobis=0.0,
                source="current-path",
            ),
        )
    active_beta = max(float(beta), 1.0e-12)
    phase_start = slope_count + orthogonal_count
    phase_stop = phase_start + phase_count
    if phase_stop > jacobian.shape[0]:
        raise ValueError("Residual Jacobian does not contain the phase block.")

    phase_jacobian_white = jacobian[phase_start:phase_stop] / math.sqrt(active_beta)
    non_phase_jacobian = np.concatenate(
        (jacobian[:phase_start], jacobian[phase_stop:]),
        axis=0,
    )
    non_phase_residual = np.concatenate(
        (center_residual[:phase_start], center_residual[phase_stop:]),
        axis=0,
    )
    precision = non_phase_jacobian.T @ non_phase_jacobian
    precision = 0.5 * (precision + precision.T)
    eigenvalues, eigenvectors = np.linalg.eigh(precision)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    covariance = (
        eigenvectors
        @ np.diag(1.0 / np.clip(eigenvalues, scale * 1.0e-12, None))
        @ eigenvectors.T
    )
    mean_shift = -covariance @ non_phase_jacobian.T @ non_phase_residual
    feasible_mean = np.clip(center + mean_shift, lower, upper)
    mean_shift = feasible_mean - center

    phase_noise = _relative_phase_covariance(profile)
    phase_cholesky = np.linalg.cholesky(phase_noise)
    raw_phase_jacobian = phase_cholesky @ phase_jacobian_white
    predictive_covariance = (
        phase_noise / active_beta
        + raw_phase_jacobian @ covariance @ raw_phase_jacobian.T
    )
    predictive_covariance = 0.5 * (
        predictive_covariance + predictive_covariance.T
    )
    jitter = max(
        float(np.trace(predictive_covariance)) * 1.0e-12,
        1.0e-18,
    )
    predictive_covariance += np.eye(phase_count) * jitter
    relative = np.asarray(
        profile.session_offsets[1:] - profile.session_offsets[0],
        dtype=float,
    )
    predictive_mean = relative + raw_phase_jacobian @ mean_shift
    integer_result = solve_integer_least_squares(
        predictive_mean,
        predictive_covariance,
        candidate_count=target_count,
    )
    predictive_precision = np.linalg.pinv(predictive_covariance, hermitian=True)
    gain = covariance @ raw_phase_jacobian.T @ predictive_precision

    records: dict[tuple[int, ...], PosteriorPredictivePathProposal] = {}

    def add(candidate_path: tuple[int, ...], distance: float, source: str) -> None:
        innovation = np.asarray(candidate_path, dtype=float) - predictive_mean
        conditional_center = np.clip(
            feasible_mean + gain @ innovation,
            lower,
            upper,
        )
        records[candidate_path] = PosteriorPredictivePathProposal(
            integer_offsets=candidate_path,
            normalized_start=conditional_center,
            squared_mahalanobis=float(distance),
            source=source,
        )

    current_innovation = predictive_mean - np.asarray(path, dtype=float)
    current_distance = float(
        current_innovation @ predictive_precision @ current_innovation
    )
    add(path, current_distance, "current-path")
    for candidate in integer_result.candidates:
        candidate_path = tuple(int(value) for value in candidate.integer_vector)
        add(
            candidate_path,
            float(candidate.squared_mahalanobis),
            "posterior-predictive-lattice",
        )

    ranked = sorted(
        records.values(),
        key=lambda item: (
            item.integer_offsets != path,
            item.squared_mahalanobis,
            item.integer_offsets,
        ),
    )
    return tuple(ranked[:target_count])


def _select_particle_portfolio(
    particles: Sequence[PhasePathParticle],
    *,
    maximum_count: int,
    protected_per_mode: int,
    maximum_per_mode: int,
) -> list[PhasePathParticle]:
    ranked = sorted(
        particles,
        key=lambda item: (-item.log_weight, item.source_mode_index, item.integer_offsets),
    )
    target = min(max(int(maximum_count), 1), len(ranked))
    groups: dict[int, list[PhasePathParticle]] = {}
    for particle in ranked:
        groups.setdefault(particle.source_mode_index, []).append(particle)
    selected: list[PhasePathParticle] = []
    selected_keys: set[tuple[int, tuple[int, ...]]] = set()
    mode_counts: dict[int, int] = {}

    def take(particle: PhasePathParticle) -> None:
        key = (particle.source_mode_index, particle.integer_offsets)
        if (
            key in selected_keys
            or len(selected) >= target
            or mode_counts.get(particle.source_mode_index, 0)
            >= max(int(maximum_per_mode), 1)
        ):
            return
        selected.append(particle)
        selected_keys.add(key)
        mode_counts[particle.source_mode_index] = (
            mode_counts.get(particle.source_mode_index, 0) + 1
        )

    quota = max(int(protected_per_mode), 0)
    for mode_index in sorted(
        groups,
        key=lambda index: (-groups[index][0].log_weight, index),
    ):
        for particle in groups[mode_index][:quota]:
            take(particle)
    for particle in ranked:
        take(particle)
    return selected


def _select_continuation_branches(
    branches: Sequence[ActiveTimingBranch],
    *,
    maximum_count: int,
    protected_per_mode: int,
    maximum_orbit_slope_chi2_delta: float,
) -> list[ActiveTimingBranch]:
    pool = list(branches)
    if not pool:
        return []
    slope_scores = [
        float((branch.diagnostics or {}).get("orbit_slope_chi2", math.nan))
        for branch in pool
    ]
    finite_scores = [value for value in slope_scores if np.isfinite(value)]
    if finite_scores:
        threshold = min(finite_scores) + max(
            float(maximum_orbit_slope_chi2_delta),
            0.0,
        )
        supported = [
            branch
            for branch, slope_score in zip(pool, slope_scores, strict=True)
            if np.isfinite(slope_score) and slope_score <= threshold
        ]
        if supported:
            pool = supported
    target = min(max(int(maximum_count), 1), len(pool))
    def rank_key(branch: ActiveTimingBranch) -> tuple[float, float, str]:
        exact_objective = float(
            (branch.diagnostics or {}).get("initial_objective_chi2", math.inf)
        )
        return (exact_objective, -float(branch.log_prior), branch.branch_id)

    ranked = sorted(pool, key=rank_key)
    groups: dict[int, list[ActiveTimingBranch]] = {}
    for branch in ranked:
        mode_index = int((branch.diagnostics or {}).get("mode_index", -1))
        groups.setdefault(mode_index, []).append(branch)
    protected: list[ActiveTimingBranch] = []
    quota = max(int(protected_per_mode), 0)
    for mode_index in sorted(
        groups,
        key=lambda index: (*rank_key(groups[index][0])[:2], index),
    ):
        protected.extend(groups[mode_index][:quota])
        if len(protected) >= target:
            break
    selected = protected[:target]
    if len(selected) >= target:
        return selected
    selected_ids = {branch.branch_id for branch in selected}
    for branch in ranked:
        if branch.branch_id in selected_ids:
            continue
        selected.append(branch)
        selected_ids.add(branch.branch_id)
        if len(selected) >= target:
            break
    return selected


def continue_phase_paths_adaptively(
    *,
    evaluator: Any,
    initial_branches: Sequence[ActiveTimingBranch],
    coordinates: DDSearchCoordinateSystem,
    base_parameter_values: Sequence[float],
    structure_id: str,
    binary_model: str | None,
    config: BayesianContinuationConfig | None = None,
) -> BayesianContinuationResult:
    """Continue slope modes into the exact factorized phase-path target."""

    active = config or BayesianContinuationConfig()
    base = np.asarray(base_parameter_values, dtype=float)
    selected = _select_continuation_branches(
        initial_branches,
        maximum_count=int(active.max_particles),
        protected_per_mode=int(active.protected_particles_per_mode),
        maximum_orbit_slope_chi2_delta=float(
            active.maximum_orbit_slope_chi2_delta
        ),
    )
    particles: list[PhasePathParticle] = []
    for branch in selected:
        diagnostics = branch.diagnostics or {}
        full_path = diagnostics.get("integer_session_offsets")
        if full_path is None:
            continue
        path = tuple(int(value) for value in full_path)[1:]
        continuation_values = diagnostics.get("posterior_mode_parameter_values")
        values = (
            np.asarray(continuation_values, dtype=float)
            if continuation_values is not None
            else base + np.asarray(branch.theta_offset, dtype=float)
        )
        if values.shape != base.shape or not np.all(np.isfinite(values)):
            continue
        try:
            search = coordinates.to_search(values)
            normalized = coordinates.normalize(search)
            profile = evaluator.evaluate_values(values)
            calibration = float(diagnostics.get("slope_calibration_scale", 1.0))
            slope, orthogonal, phase, prior = _continuation_components(
                profile=profile,
                integer_offsets=path,
                normalized_search_values=normalized,
                calibration_scale=calibration,
            )
        except Exception:
            continue
        particles.append(
            PhasePathParticle(
                parameter_values=values,
                integer_offsets=path,
                source_branch_id=branch.branch_id,
                source_mode_index=int(diagnostics.get("mode_index", -1)),
                calibration_scale=calibration,
                log_weight=float(branch.log_prior),
                slope_chi2=float(slope @ slope),
                orthogonal_chi2=float(orthogonal @ orthogonal),
                ambiguity_chi2=float(phase @ phase),
                prior_chi2=float(prior @ prior),
                path_update_count=0,
                nfev=0,
                success=True,
                path_proposal_source="initial-lattice",
            )
        )
    if not particles:
        raise RuntimeError("Bayesian continuation received no valid phase-path particle.")

    beta = 0.0
    beta_one_steps = 0
    history: list[dict[str, Any]] = []
    for step_index in range(max(int(active.maximum_beta_steps), 1)):
        remaining = np.asarray(
            [item.orthogonal_chi2 + item.ambiguity_chi2 for item in particles],
            dtype=float,
        )
        pre_resample_weights = np.asarray(
            [item.log_weight for item in particles],
            dtype=float,
        )
        next_beta = select_mode_aware_beta(
            current_beta=beta,
            remaining_chi2=remaining,
            mode_labels=[item.source_mode_index for item in particles],
            ess_target_fraction=float(active.ess_target_fraction),
            minimum_step=float(active.minimum_beta_step),
            maximum_increment=float(active.maximum_beta_increment),
            maximum_growth_factor=float(active.maximum_beta_growth_factor),
            representatives_per_mode=int(active.ess_representatives_per_mode),
        )
        moved: list[PhasePathParticle] = []
        for particle in particles:
            start_search = coordinates.to_search(particle.parameter_values)
            start_normalized = coordinates.normalize(start_search)
            radius = max(float(active.trust_radius), 1.0e-6)
            global_lower = coordinates.normalized_lower
            global_upper = coordinates.normalized_upper
            interior = np.maximum(
                (global_upper - global_lower) * 1.0e-12,
                np.finfo(float).eps,
            )
            start = np.clip(
                start_normalized,
                global_lower + interior,
                global_upper - interior,
            )
            lower = np.maximum(global_lower, start - radius)
            upper = np.minimum(global_upper, start + radius)
            initial_profile = evaluator.evaluate_values(particle.parameter_values)
            slope0, orthogonal0, phase0, prior0 = _continuation_components(
                profile=initial_profile,
                integer_offsets=particle.integer_offsets,
                normalized_search_values=start_normalized,
                calibration_scale=particle.calibration_scale,
            )
            residual_size = slope0.size + orthogonal0.size + phase0.size + prior0.size

            def residual_vector_for_path(
                integer_path: tuple[int, ...],
            ) -> Any:
                def residual_vector(normalized_values: np.ndarray) -> np.ndarray:
                    try:
                        values = coordinates.to_physical(
                            coordinates.denormalize(normalized_values)
                        )
                        profile = evaluator.evaluate_values(values)
                        slope, orthogonal, phase, prior = _continuation_components(
                            profile=profile,
                            integer_offsets=integer_path,
                            normalized_search_values=normalized_values,
                            calibration_scale=particle.calibration_scale,
                        )
                        return np.concatenate(
                            (
                                slope,
                                math.sqrt(next_beta) * orthogonal,
                                math.sqrt(next_beta) * phase,
                                prior,
                            )
                        )
                    except Exception:
                        return np.full(residual_size, 1.0e12, dtype=float)

                return residual_vector

            optimization = least_squares(
                residual_vector_for_path(particle.integer_offsets),
                start,
                bounds=(lower, upper),
                method="trf",
                jac="2-point",
                diff_step=max(float(active.finite_difference_fraction), 1.0e-7),
                max_nfev=max(int(active.max_nfev_per_move), 1),
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
            )
            final_normalized = np.asarray(optimization.x, dtype=float)
            final_values = coordinates.to_physical(
                coordinates.denormalize(final_normalized)
            )
            final_profile = evaluator.evaluate_values(final_values)
            try:
                path_proposals = posterior_predictive_integer_path_proposals(
                    profile=final_profile,
                    current_path=particle.integer_offsets,
                    normalized_center=final_normalized,
                    residual_jacobian=np.asarray(optimization.jac, dtype=float),
                    residual_at_center=np.asarray(optimization.fun, dtype=float),
                    slope_residual_count=slope0.size,
                    orthogonal_residual_count=orthogonal0.size,
                    phase_residual_count=phase0.size,
                    beta=next_beta,
                    lower_bounds=lower,
                    upper_bounds=upper,
                    candidate_count=int(active.integer_candidates_per_particle),
                )
            except Exception:
                fallback_paths = _candidate_integer_paths(
                    final_profile,
                    particle.integer_offsets,
                    candidate_count=int(active.integer_candidates_per_particle),
                )
                path_proposals = tuple(
                    PosteriorPredictivePathProposal(
                        integer_offsets=candidate_path,
                        normalized_start=final_normalized.copy(),
                        squared_mahalanobis=float(index),
                        source=(
                            "current-path"
                            if candidate_path == particle.integer_offsets
                            else "measurement-lattice-fallback"
                        ),
                    )
                    for index, candidate_path in enumerate(fallback_paths)
                )
            path_proposals = path_proposals[
                : max(
                    int(active.reoptimized_integer_candidates_per_particle),
                    1,
                )
            ]
            for proposal in path_proposals:
                candidate_path = proposal.integer_offsets
                candidate_optimization = optimization
                candidate_normalized = final_normalized
                candidate_values = final_values
                candidate_profile = final_profile
                if candidate_path != particle.integer_offsets:
                    candidate_start = np.clip(
                        proposal.normalized_start,
                        lower + interior,
                        upper - interior,
                    )
                    candidate_optimization = least_squares(
                        residual_vector_for_path(candidate_path),
                        candidate_start,
                        bounds=(lower, upper),
                        method="trf",
                        jac="2-point",
                        diff_step=max(
                            float(active.finite_difference_fraction),
                            1.0e-7,
                        ),
                        max_nfev=max(int(active.max_nfev_per_move), 1),
                        ftol=1.0e-8,
                        xtol=1.0e-8,
                        gtol=1.0e-8,
                    )
                    candidate_normalized = np.asarray(
                        candidate_optimization.x,
                        dtype=float,
                    )
                    candidate_values = coordinates.to_physical(
                        coordinates.denormalize(candidate_normalized)
                    )
                    candidate_profile = evaluator.evaluate_values(candidate_values)
                path_updates = particle.path_update_count + int(
                    candidate_path != particle.integer_offsets
                )
                slope, orthogonal, phase, prior = _continuation_components(
                    profile=candidate_profile,
                    integer_offsets=candidate_path,
                    normalized_search_values=candidate_normalized,
                    calibration_scale=particle.calibration_scale,
                )
                log_scale = math.log(max(particle.calibration_scale, 1.0e-12))
                calibration_penalty = (
                    2.0 * slope.size * log_scale
                    + (
                        log_scale
                        / max(float(active.log_slope_scale_prior_sigma), 1.0e-6)
                    )
                    ** 2
                )
                posterior_chi2 = (
                    float(slope @ slope)
                    + float(prior @ prior)
                    + calibration_penalty
                    + next_beta
                    * (float(orthogonal @ orthogonal) + float(phase @ phase))
                )
                moved.append(
                    PhasePathParticle(
                        parameter_values=candidate_values,
                        integer_offsets=candidate_path,
                        source_branch_id=particle.source_branch_id,
                        source_mode_index=particle.source_mode_index,
                        calibration_scale=particle.calibration_scale,
                        log_weight=float(-0.5 * posterior_chi2),
                        slope_chi2=float(slope @ slope),
                        orthogonal_chi2=float(orthogonal @ orthogonal),
                        ambiguity_chi2=float(phase @ phase),
                        prior_chi2=float(prior @ prior),
                        path_update_count=path_updates,
                        nfev=(
                            particle.nfev
                            + int(optimization.nfev)
                            + (
                                int(candidate_optimization.nfev)
                                if candidate_optimization is not optimization
                                else 0
                            )
                        ),
                        success=(
                            particle.success
                            and bool(optimization.success)
                            and bool(candidate_optimization.success)
                        ),
                        path_proposal_source=proposal.source,
                    )
                )
        particles = _select_particle_portfolio(
            moved,
            maximum_count=int(active.max_particles),
            protected_per_mode=int(active.protected_particles_per_mode),
            maximum_per_mode=int(active.maximum_particles_per_mode),
        )
        updated_weights = [item.log_weight for item in particles]
        history.append(
            {
                "step_index": step_index,
                "beta_start": beta,
                "beta_end": next_beta,
                "pre_resample_effective_sample_size": _effective_sample_size(
                    pre_resample_weights
                ),
                "effective_sample_size": _effective_sample_size(updated_weights),
                "best_slope_chi2": min(item.slope_chi2 for item in particles),
                "best_orthogonal_chi2": min(item.orthogonal_chi2 for item in particles),
                "best_ambiguity_chi2": min(item.ambiguity_chi2 for item in particles),
                "unique_path_count": len({item.integer_offsets for item in particles}),
                "path_update_count": sum(item.path_update_count for item in particles),
            }
        )
        beta = next_beta
        if beta >= 1.0 - 1.0e-12:
            beta = 1.0
            beta_one_steps += 1
            if beta_one_steps >= max(int(active.beta_one_refinement_steps), 1):
                break
    particles.sort(key=lambda item: (-item.log_weight, item.source_branch_id))
    branches = tuple(
        ActiveTimingBranch(
            branch_id=f"information_ordered_continuation_{index:04d}",
            structure_id=str(structure_id),
            binary_model=(str(binary_model).upper() if binary_model else None),
            wraps=integer_session_offsets_to_gap_wraps(particle.integer_offsets),
            theta_names=coordinates.parameter_names,
            theta_offset=np.asarray(particle.parameter_values - base, dtype=float),
            log_prior=float(particle.log_weight),
            proposal_source="apollo-bayesian-information-ordered-continuation",
            diagnostics={
                "proposal_family": "adaptive-phase-orbit-continuation",
                "source_branch_id": particle.source_branch_id,
                "mode_index": particle.source_mode_index,
                "slope_calibration_scale": particle.calibration_scale,
                "integer_session_offsets": (0, *particle.integer_offsets),
                "continuation_beta": beta,
                "slope_chi2": particle.slope_chi2,
                "orthogonal_chi2": particle.orthogonal_chi2,
                "ambiguity_chi2": particle.ambiguity_chi2,
                "prior_chi2": particle.prior_chi2,
                "path_proposal_source": particle.path_proposal_source,
                "certification_theta_offset": np.asarray(
                    particle.parameter_values - base,
                    dtype=float,
                ),
            },
        )
        for index, particle in enumerate(particles)
    )
    return BayesianContinuationResult(
        particles=tuple(particles),
        branches=branches,
        beta_history=tuple(history),
        reached_beta_one=beta >= 1.0,
    )


def _select_independent_starts(
    records: Sequence[tuple[float, int, np.ndarray, np.ndarray]],
    *,
    count: int,
    exploitation_fraction: float,
) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    ranked = sorted(records, key=lambda item: (-item[0], item[1]))
    target = min(max(int(count), 1), len(ranked))
    exploit_count = min(
        max(int(round(target * min(max(float(exploitation_fraction), 0.0), 1.0))), 1),
        target,
    )
    selected = list(ranked[:exploit_count])
    remaining = list(ranked[exploit_count:])
    while len(selected) < target and remaining:
        distances = [
            min(np.linalg.norm(candidate[3] - chosen[3]) for chosen in selected)
            for candidate in remaining
        ]
        index = int(np.argmax(distances))
        selected.append(remaining.pop(index))
    return selected


def estimate_orbital_alias_count(
    *,
    observation_span_days: float,
    pb_lower_days: float,
    pb_upper_days: float,
    oversampling: float = 1.0,
    maximum_count: int = 48,
) -> int:
    """Count resolvable orbital-frequency aliases across a bounded PB prior."""

    span = max(float(observation_span_days), 0.0)
    lower = float(pb_lower_days)
    upper = float(pb_upper_days)
    if not np.isfinite(span) or not np.isfinite(lower) or not np.isfinite(upper):
        return 0
    if span <= 0.0 or lower <= 0.0 or upper <= lower:
        return 0
    cycle_span = span * (1.0 / lower - 1.0 / upper)
    requested = max(int(math.ceil(cycle_span * max(float(oversampling), 1.0))), 1)
    return min(requested, max(int(maximum_count), 1))


def _structured_orbit_alias_samples(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: InformationOrderedBinarySearchConfig,
    anchor_search_values: Sequence[Sequence[float]] = (),
) -> tuple[np.ndarray, np.ndarray, int]:
    dimension = len(coordinates.parameter_names)
    if not bool(config.structured_orbit_alias_enabled):
        return np.zeros((0, dimension), dtype=float), np.zeros(0, dtype=int), 0
    try:
        profile = evaluator.evaluate_values(coordinates.center_values)
    except Exception:
        return np.zeros((0, dimension), dtype=float), np.zeros(0, dtype=int), 0
    epochs = np.asarray(getattr(profile, "session_epochs", ()), dtype=float)
    epochs = epochs[np.isfinite(epochs)]
    if epochs.size < 2:
        return np.zeros((0, dimension), dtype=float), np.zeros(0, dtype=int), 0

    pb_index = coordinates.parameter_names.index("PB")
    phase_index = coordinates.parameter_names.index("T0")
    alias_count = estimate_orbital_alias_count(
        observation_span_days=float(np.max(epochs) - np.min(epochs)),
        pb_lower_days=float(coordinates.physical_lower[pb_index]),
        pb_upper_days=float(coordinates.physical_upper[pb_index]),
        oversampling=float(config.orbital_alias_oversampling),
        maximum_count=int(config.maximum_orbital_alias_count),
    )
    phase_count = max(int(config.orbital_phase_sample_count), 1)
    if alias_count == 0:
        return np.zeros((0, dimension), dtype=float), np.zeros(0, dtype=int), 0

    bases = [coordinates.search_center.copy()]
    maximum_bases = max(int(config.maximum_structured_anchor_bases), 1)
    for values_like in anchor_search_values:
        values = np.asarray(values_like, dtype=float)
        if (
            values.shape != coordinates.search_center.shape
            or not np.all(np.isfinite(values))
            or np.any(values < coordinates.search_lower)
            or np.any(values > coordinates.search_upper)
            or any(np.allclose(values, base, rtol=0.0, atol=1.0e-12) for base in bases)
        ):
            continue
        bases.append(values.copy())
        if len(bases) >= maximum_bases:
            break
    total = alias_count * phase_count * len(bases)
    samples = np.empty((total, dimension), dtype=float)
    frequency_lower = 1.0 / float(coordinates.physical_upper[pb_index])
    frequency_upper = 1.0 / float(coordinates.physical_lower[pb_index])
    frequencies = np.linspace(
        frequency_lower,
        frequency_upper,
        alias_count,
        endpoint=True,
    )
    phases = np.linspace(
        float(coordinates.search_lower[phase_index]),
        float(coordinates.search_upper[phase_index]),
        phase_count,
        endpoint=False,
    )
    alias_labels = np.empty(total, dtype=int)
    row_index = 0
    for alias_index, frequency in enumerate(frequencies):
        for base in bases:
            for phase in phases:
                samples[row_index] = base
                samples[row_index, pb_index] = math.log(1.0 / float(frequency))
                samples[row_index, phase_index] = float(phase)
                alias_labels[row_index] = alias_index
                row_index += 1
    return samples, alias_labels, alias_count


def _protect_orbital_alias_starts(
    records: Sequence[tuple[float, int, np.ndarray, np.ndarray]],
    *,
    alias_by_sample_index: dict[int, int],
    starts_per_alias: int,
    focused_alias_count: int = 0,
    focused_starts_per_alias: int = 0,
) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    quota = max(int(starts_per_alias), 0)
    if quota == 0:
        return []
    by_alias: dict[int, list[tuple[float, int, np.ndarray, np.ndarray]]] = {}
    for record in records:
        alias_index = alias_by_sample_index.get(int(record[1]))
        if alias_index is not None:
            by_alias.setdefault(alias_index, []).append(record)
    ranked_aliases = sorted(
        by_alias,
        key=lambda alias_index: (
            -max(record[0] for record in by_alias[alias_index]),
            alias_index,
        ),
    )
    focused = set(ranked_aliases[: max(int(focused_alias_count), 0)])
    focused_quota = max(int(focused_starts_per_alias), quota)
    protected: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    for alias_index in sorted(by_alias):
        active_quota = focused_quota if alias_index in focused else quota
        protected.extend(
            sorted(by_alias[alias_index], key=lambda item: (-item[0], item[1]))[
                :active_quota
            ]
        )
    return protected


def _orbital_alias_normalized_bounds(
    coordinates: DDSearchCoordinateSystem,
    *,
    alias_index: int,
    alias_count: int,
) -> tuple[float, float]:
    """Return normalized PB bounds for one resolvable frequency cell."""

    count = max(int(alias_count), 1)
    index = min(max(int(alias_index), 0), count - 1)
    pb_index = coordinates.parameter_names.index("PB")
    frequency_lower = 1.0 / float(coordinates.physical_upper[pb_index])
    frequency_upper = 1.0 / float(coordinates.physical_lower[pb_index])
    if count == 1:
        cell_lower = frequency_lower
        cell_upper = frequency_upper
    else:
        centers = np.linspace(frequency_lower, frequency_upper, count, endpoint=True)
        cell_lower = (
            frequency_lower
            if index == 0
            else 0.5 * float(centers[index - 1] + centers[index])
        )
        cell_upper = (
            frequency_upper
            if index == count - 1
            else 0.5 * float(centers[index] + centers[index + 1])
        )
    physical_lower = coordinates.center_values.copy()
    physical_upper = coordinates.center_values.copy()
    physical_lower[pb_index] = 1.0 / cell_upper
    physical_upper[pb_index] = 1.0 / cell_lower
    normalized_lower = coordinates.normalize(coordinates.to_search(physical_lower))
    normalized_upper = coordinates.normalize(coordinates.to_search(physical_upper))
    cell_normalized_lower = float(
        min(normalized_lower[pb_index], normalized_upper[pb_index])
    )
    cell_normalized_upper = float(
        max(normalized_lower[pb_index], normalized_upper[pb_index])
    )
    return (
        max(cell_normalized_lower, float(coordinates.normalized_lower[pb_index])),
        min(cell_normalized_upper, float(coordinates.normalized_upper[pb_index])),
    )


def _slope_information_audit(
    standardized_slope_jacobian: np.ndarray,
    *,
    calibration_scale: float,
) -> tuple[SlopeInformationAudit, np.ndarray]:
    jacobian = np.asarray(standardized_slope_jacobian, dtype=float) / max(
        float(calibration_scale), 1.0e-12
    )
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    threshold = (
        max(jacobian.shape) * np.finfo(float).eps * singular_values[0]
        if singular_values.size and singular_values[0] > 0.0
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    information = jacobian.T @ jacobian
    posterior_covariance = np.linalg.inv(information + np.eye(information.shape[0]))
    eigenvalues = np.linalg.eigvalsh(information)
    condition_number = (
        float(singular_values[0] / singular_values[rank - 1])
        if rank > 0
        else float("inf")
    )
    audit = SlopeInformationAudit(
        observation_count=int(jacobian.shape[0]),
        singular_values=singular_values,
        numerical_rank=rank,
        null_dimension=int(jacobian.shape[1] - rank),
        condition_number=condition_number,
        prior_to_posterior_variance_ratios=1.0 + np.maximum(eigenvalues, 0.0),
    )
    return audit, posterior_covariance


def discover_wrap_invariant_orbit_modes(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    config: InformationOrderedBinarySearchConfig | None = None,
    anchor_parameter_values: Sequence[Sequence[float]] = (),
) -> OrbitPosteriorResult:
    active = config or InformationOrderedBinarySearchConfig()
    prior_samples = coordinates.sample_prior(
        max(int(active.prior_sample_count), 1),
        seed=int(active.random_seed),
    )
    anchor_search: list[np.ndarray] = []
    for values in anchor_parameter_values:
        try:
            search = coordinates.to_search(values)
        except (ValueError, OverflowError):
            continue
        if np.all(search >= coordinates.search_lower) and np.all(
            search <= coordinates.search_upper
        ):
            anchor_search.append(search)
    structured_samples, structured_aliases, orbital_alias_count = (
        _structured_orbit_alias_samples(
            evaluator=evaluator,
            coordinates=coordinates,
            config=active,
            anchor_search_values=anchor_search,
        )
    )
    sample_blocks = [coordinates.search_center[None, :], prior_samples]
    structured_start_index = 1 + len(prior_samples)
    if structured_samples.size:
        sample_blocks.append(structured_samples)
    if anchor_search:
        sample_blocks.append(np.vstack(anchor_search))
    samples = np.vstack(sample_blocks)
    alias_by_sample_index = {
        structured_start_index + offset: int(alias_index)
        for offset, alias_index in enumerate(structured_aliases)
    }
    records: list[tuple[float, int, np.ndarray, np.ndarray]] = []
    failed_prior_count = 0
    observable_count: int | None = None
    for sample_index, search_values in enumerate(samples):
        normalized = coordinates.normalize(search_values)
        try:
            profile = evaluator.evaluate_values(coordinates.to_physical(search_values))
            slopes = _profile_standardized_slopes(profile)
        except Exception:
            failed_prior_count += 1
            continue
        if observable_count is None:
            observable_count = int(slopes.size)
        elif slopes.size != observable_count:
            failed_prior_count += 1
            continue
        calibration = calibrated_slope_log_likelihood(
            float(slopes @ slopes),
            slopes.size,
            log_scale_prior_sigma=float(active.log_slope_scale_prior_sigma),
        )
        log_posterior = calibration.log_marginal_likelihood - 0.5 * float(
            normalized @ normalized
        )
        records.append((log_posterior, sample_index, search_values.copy(), normalized))
    if not records or observable_count is None:
        raise RuntimeError("Slope-only prior exploration produced no valid DD state.")

    independent_starts = _select_independent_starts(
        records,
        count=max(int(active.optimization_start_count), 1),
        exploitation_fraction=float(active.exploitation_fraction),
    )
    protected_starts = _protect_orbital_alias_starts(
        records,
        alias_by_sample_index=alias_by_sample_index,
        starts_per_alias=int(active.protected_starts_per_orbital_alias),
        focused_alias_count=int(active.focused_orbital_alias_count),
        focused_starts_per_alias=int(active.focused_starts_per_orbital_alias),
    )
    target_start_count = max(
        int(active.optimization_start_count),
        len(protected_starts),
        1,
    )
    starts = list(protected_starts)
    selected_source_indices = {int(item[1]) for item in starts}
    for record in independent_starts:
        if int(record[1]) in selected_source_indices:
            continue
        starts.append(record)
        selected_source_indices.add(int(record[1]))
        if len(starts) >= target_start_count:
            break
    normalized_lower = coordinates.normalized_lower
    normalized_upper = coordinates.normalized_upper
    optimized: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, int, bool]] = []
    for _, source_index, _, start_normalized in starts:
        local_lower = normalized_lower.copy()
        local_upper = normalized_upper.copy()
        alias_index = alias_by_sample_index.get(int(source_index))
        if (
            bool(active.constrain_orbital_alias_optimization)
            and alias_index is not None
            and orbital_alias_count > 0
        ):
            pb_index = coordinates.parameter_names.index("PB")
            pb_lower, pb_upper = _orbital_alias_normalized_bounds(
                coordinates,
                alias_index=alias_index,
                alias_count=orbital_alias_count,
            )
            local_lower[pb_index] = max(local_lower[pb_index], pb_lower)
            local_upper[pb_index] = min(local_upper[pb_index], pb_upper)
        bounded_start = np.clip(start_normalized, local_lower, local_upper)
        start_profile = evaluator.evaluate_values(
            coordinates.to_physical(coordinates.denormalize(bounded_start))
        )
        start_slopes = _profile_standardized_slopes(start_profile)
        numerical_scale = max(
            float(np.sqrt(np.mean(np.square(start_slopes)))),
            1.0,
        )

        def residual_vector(normalized_values: np.ndarray) -> np.ndarray:
            try:
                physical = coordinates.to_physical(
                    coordinates.denormalize(normalized_values)
                )
                slopes = _profile_standardized_slopes(evaluator.evaluate_values(physical))
            except Exception:
                return np.full(observable_count, 1.0e12, dtype=float)
            if slopes.shape != (observable_count,):
                return np.full(observable_count, 1.0e12, dtype=float)
            return slopes / numerical_scale

        optimization = least_squares(
            residual_vector,
            bounded_start,
            bounds=(local_lower, local_upper),
            method="trf",
            jac="2-point",
            diff_step=max(float(active.finite_difference_fraction), 1.0e-6),
            max_nfev=max(int(active.max_nfev), 1),
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        final_normalized = np.asarray(optimization.x, dtype=float)
        final_search = coordinates.denormalize(final_normalized)
        final_physical = coordinates.to_physical(final_search)
        final_slopes = _profile_standardized_slopes(
            evaluator.evaluate_values(final_physical)
        )
        slope_chi2 = float(final_slopes @ final_slopes)
        calibration = calibrated_slope_log_likelihood(
            slope_chi2,
            final_slopes.size,
            log_scale_prior_sigma=float(active.log_slope_scale_prior_sigma),
        )
        log_posterior = calibration.log_marginal_likelihood - 0.5 * float(
            final_normalized @ final_normalized
        )
        optimized.append(
            (
                log_posterior,
                source_index,
                final_search,
                final_physical,
                final_normalized,
                int(optimization.nfev),
                bool(optimization.success),
            )
        )

    selected: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, int, bool]] = []
    separation = max(float(active.minimum_mode_separation), 0.0)
    for candidate in sorted(optimized, key=lambda item: (-item[0], item[1])):
        if any(
            np.linalg.norm(candidate[4] - existing[4]) < separation
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max(int(active.retained_mode_count), 1):
            break
    if not selected:
        raise RuntimeError("Slope-only optimization retained no finite orbit mode.")

    finite_difference_fraction = max(float(active.finite_difference_fraction), 1.0e-6)
    parameter_steps = np.maximum(
        coordinates.physical_scales * finite_difference_fraction,
        np.finfo(float).tiny,
    )
    provisional: list[OrbitPosteriorMode] = []
    for (
        log_posterior,
        source_index,
        search_values,
        parameter_values,
        normalized_values,
        nfev,
        success,
    ) in selected:
        profile = evaluator.evaluate_values(parameter_values)
        slopes = _profile_standardized_slopes(profile)
        slope_chi2 = float(slopes @ slopes)
        calibration = calibrated_slope_log_likelihood(
            slope_chi2,
            slopes.size,
            log_scale_prior_sigma=float(active.log_slope_scale_prior_sigma),
        )
        linearization = evaluator.linearize_values(
            parameter_values,
            parameter_steps=parameter_steps,
        )
        slope_jacobian = (
            np.asarray(linearization.standardized_slope_jacobian, dtype=float)
            / finite_difference_fraction
        )
        audit, covariance = _slope_information_audit(
            slope_jacobian,
            calibration_scale=calibration.scale_map,
        )
        sign, logdet = np.linalg.slogdet(covariance)
        laplace_log_evidence = (
            log_posterior
            + 0.5 * float(logdet if sign > 0 else -math.inf)
            + 0.5 * covariance.shape[0] * math.log(2.0 * math.pi)
        )
        provisional.append(
            OrbitPosteriorMode(
                parameter_values=np.asarray(parameter_values, dtype=float),
                search_values=np.asarray(search_values, dtype=float),
                normalized_search_values=np.asarray(normalized_values, dtype=float),
                scaled_covariance=covariance,
                slope_chi2=slope_chi2,
                calibration_scale=calibration.scale_map,
                log_posterior=float(log_posterior),
                laplace_log_evidence=float(laplace_log_evidence),
                posterior_mass=0.0,
                information_audit=audit,
                source_start_index=int(source_index),
                nfev=nfev,
                success=success,
            )
        )

    log_evidence = np.asarray([mode.laplace_log_evidence for mode in provisional])
    finite = np.isfinite(log_evidence)
    if not np.any(finite):
        masses = np.full(len(provisional), 1.0 / len(provisional), dtype=float)
    else:
        shifted = np.where(finite, log_evidence - np.max(log_evidence[finite]), -math.inf)
        weights = np.exp(shifted)
        masses = weights / np.sum(weights)
    modes = tuple(
        replace(mode, posterior_mass=float(mass))
        for mode, mass in zip(provisional, masses, strict=True)
    )
    return OrbitPosteriorResult(
        modes=modes,
        evaluated_prior_count=len(records),
        failed_prior_count=failed_prior_count,
        optimized_start_count=len(starts),
        observable_slope_count=observable_count,
        orbital_alias_count=orbital_alias_count,
        structured_prior_count=len(structured_samples),
        protected_alias_start_count=len(protected_starts),
    )


def posterior_boundary_parameters(
    posterior: OrbitPosteriorResult,
    coordinates: DDSearchCoordinateSystem,
    *,
    boundary_fraction: float,
    mass_floor: float = 0.01,
) -> tuple[str, ...]:
    threshold = min(max(float(boundary_fraction), 0.0), 1.0)
    active_modes = [
        mode for mode in posterior.modes if mode.posterior_mass >= max(float(mass_floor), 0.0)
    ]
    if not active_modes and posterior.modes:
        active_modes = [max(posterior.modes, key=lambda mode: mode.posterior_mass)]
    boundary_names: set[str] = set()
    for mode in active_modes:
        normalized = np.asarray(mode.normalized_search_values, dtype=float)
        for index, name in enumerate(coordinates.parameter_names):
            if abs(float(normalized[index])) >= threshold:
                boundary_names.add(name)
    return tuple(name for name in coordinates.parameter_names if name in boundary_names)


def _expanded_physical_bounds(
    coordinates: DDSearchCoordinateSystem,
    parameter_names: Sequence[str],
    *,
    factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = coordinates.physical_lower.copy()
    upper = coordinates.physical_upper.copy()
    expansion = max(float(factor), 1.0)
    for name in parameter_names:
        index = coordinates.parameter_names.index(str(name).upper())
        half_width = 0.5 * (upper[index] - lower[index]) * expansion
        lower[index] = coordinates.center_values[index] - half_width
        upper[index] = coordinates.center_values[index] + half_width
        if name in {"F0", "A1", "PB"}:
            lower[index] = max(lower[index], np.finfo(float).tiny)
        elif name == "ECC":
            lower[index] = max(lower[index], 1.0e-9)
            upper[index] = min(upper[index], 1.0 - 1.0e-9)
    return lower, upper


def adaptive_discover_wrap_invariant_orbit_modes(
    *,
    evaluator: Any,
    parameter_names: Sequence[str],
    center_values: Sequence[float],
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    pepoch_mjd: float,
    reference_mjd: float,
    config: InformationOrderedBinarySearchConfig | None = None,
) -> AdaptiveOrbitPosteriorResult:
    active = config or InformationOrderedBinarySearchConfig()
    coordinates = build_dd_search_coordinate_system(
        parameter_names=parameter_names,
        center_values=center_values,
        lower_values=lower_values,
        upper_values=upper_values,
        pepoch_mjd=pepoch_mjd,
        reference_mjd=reference_mjd,
        seed=int(active.random_seed),
    )
    history: list[dict[str, Any]] = []
    anchors: tuple[np.ndarray, ...] = ()
    posterior: OrbitPosteriorResult | None = None
    maximum_round = max(int(active.max_prior_expansion_rounds), 0)
    for round_index in range(maximum_round + 1):
        round_config = replace(active, random_seed=int(active.random_seed) + round_index)
        posterior = discover_wrap_invariant_orbit_modes(
            evaluator=evaluator,
            coordinates=coordinates,
            config=round_config,
            anchor_parameter_values=anchors,
        )
        best_mode = max(posterior.modes, key=lambda mode: mode.log_posterior)
        boundary = posterior_boundary_parameters(
            posterior,
            coordinates,
            boundary_fraction=float(active.prior_boundary_fraction),
        )
        reduced_slope_chi2 = best_mode.slope_chi2 / max(
            posterior.observable_slope_count, 1
        )
        should_expand = bool(
            boundary
            and reduced_slope_chi2 > max(float(active.slope_reduced_chi2_target), 0.0)
            and round_index < maximum_round
        )
        history.append(
            {
                "round_index": round_index,
                "best_slope_chi2": best_mode.slope_chi2,
                "best_reduced_slope_chi2": reduced_slope_chi2,
                "boundary_parameters": boundary,
                "expanded": should_expand,
                "physical_lower": coordinates.physical_lower.copy(),
                "physical_upper": coordinates.physical_upper.copy(),
            }
        )
        if not should_expand:
            break
        anchors = tuple(mode.parameter_values for mode in posterior.modes)
        expanded_lower, expanded_upper = _expanded_physical_bounds(
            coordinates,
            boundary,
            factor=float(active.prior_expansion_factor),
        )
        coordinates = build_dd_search_coordinate_system(
            parameter_names=parameter_names,
            center_values=center_values,
            lower_values=expanded_lower,
            upper_values=expanded_upper,
            pepoch_mjd=pepoch_mjd,
            reference_mjd=reference_mjd,
            seed=int(active.random_seed) + round_index + 1,
        )
    if posterior is None:
        raise RuntimeError("Adaptive slope-only search did not execute a posterior round.")
    return AdaptiveOrbitPosteriorResult(
        coordinates=coordinates,
        posterior=posterior,
        expansion_history=tuple(history),
    )


def expand_orbit_modes_over_slope_calibration(
    *,
    evaluator: Any,
    posterior: OrbitPosteriorResult,
    coordinates: DDSearchCoordinateSystem,
    calibration_scales: Sequence[float],
    log_scale_prior_sigma: float,
    finite_difference_fraction: float = 1.0e-3,
    minimum_relative_mass: float = 1.0e-12,
) -> OrbitPosteriorResult:
    """Retain proper-prior kappa hypotheses for phase-conditioned updating."""

    scales = tuple(
        sorted(
            {
                float(value)
                for value in calibration_scales
                if np.isfinite(float(value)) and float(value) > 0.0
            }
        )
    )
    if not scales:
        raise ValueError("At least one positive slope-calibration scale is required.")
    fraction = max(float(finite_difference_fraction), 1.0e-6)
    steps = np.maximum(
        coordinates.physical_scales * fraction,
        np.finfo(float).tiny,
    )
    expanded: list[OrbitPosteriorMode] = []
    for mode in posterior.modes:
        linearization = evaluator.linearize_values(
            mode.parameter_values,
            parameter_steps=steps,
        )
        slope_jacobian = (
            np.asarray(linearization.standardized_slope_jacobian, dtype=float)
            / fraction
        )
        prior_log_density = -0.5 * float(
            mode.normalized_search_values @ mode.normalized_search_values
        )
        for calibration_scale in scales:
            audit, covariance = _slope_information_audit(
                slope_jacobian,
                calibration_scale=calibration_scale,
            )
            conditional_log_posterior = prior_log_density + conditional_slope_log_likelihood(
                mode.slope_chi2,
                posterior.observable_slope_count,
                scale=calibration_scale,
                log_scale_prior_sigma=log_scale_prior_sigma,
            )
            sign, logdet = np.linalg.slogdet(covariance)
            laplace_log_evidence = (
                conditional_log_posterior
                + 0.5 * float(logdet if sign > 0 else -math.inf)
                + 0.5 * covariance.shape[0] * math.log(2.0 * math.pi)
            )
            expanded.append(
                replace(
                    mode,
                    scaled_covariance=covariance,
                    calibration_scale=calibration_scale,
                    log_posterior=float(conditional_log_posterior),
                    laplace_log_evidence=float(laplace_log_evidence),
                    posterior_mass=0.0,
                    information_audit=audit,
                )
            )
    if not expanded:
        raise RuntimeError("Slope-calibration expansion received no orbit modes.")

    log_evidence = np.asarray([mode.laplace_log_evidence for mode in expanded])
    finite = np.isfinite(log_evidence)
    shifted = np.where(finite, log_evidence - np.max(log_evidence[finite]), -math.inf)
    weights = np.exp(shifted)
    masses = weights / np.sum(weights)
    mass_floor = max(float(minimum_relative_mass), 0.0)
    retained = [
        replace(mode, posterior_mass=float(mass))
        for mode, mass in zip(expanded, masses, strict=True)
        if mass >= mass_floor
    ]
    if not retained:
        best_index = int(np.argmax(masses))
        retained = [replace(expanded[best_index], posterior_mass=1.0)]
    else:
        total_mass = sum(mode.posterior_mass for mode in retained)
        retained = [
            replace(mode, posterior_mass=mode.posterior_mass / total_mass)
            for mode in retained
        ]
    retained.sort(key=lambda mode: (-mode.posterior_mass, mode.calibration_scale))
    return replace(posterior, modes=tuple(retained))


def generate_orbit_conditioned_phase_branches(
    *,
    evaluator: Any,
    posterior: OrbitPosteriorResult,
    coordinates: DDSearchCoordinateSystem,
    base_parameter_values: Sequence[float],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    config: MultimodeSessionLatticeConfig | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    branches, diagnostics = generate_global_phase_path_refresh_branches(
        evaluator=evaluator,
        initial_modes=posterior.modes,
        parameter_scales=coordinates.physical_scales,
        base_parameter_values=base_parameter_values,
        parameter_names=coordinates.parameter_names,
        structure_id=structure_id,
        binary_model=binary_model,
        stage_index=stage_index,
        mode_count=len(posterior.modes),
        config=config,
    )
    return branches, {
        **diagnostics,
        "information_order": (
            "signed-slope",
            "orbit-posterior",
            "conditional-phase-lattice",
        ),
        "orbit_mode_count": len(posterior.modes),
        "orbit_mode_masses": [mode.posterior_mass for mode in posterior.modes],
    }


def _linearize_profile_in_search_coordinates(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    normalized_search_values: Sequence[float],
    finite_difference_fraction: float,
) -> tuple[Any, np.ndarray, np.ndarray]:
    center = np.asarray(normalized_search_values, dtype=float)
    lower = coordinates.normalized_lower
    upper = coordinates.normalized_upper
    if center.shape != lower.shape or not np.all(np.isfinite(center)):
        raise ValueError("Normalized phase-search coordinates must be finite and aligned.")
    profile = evaluator.evaluate_values(
        coordinates.to_physical(coordinates.denormalize(center))
    )
    shape = np.asarray(profile.normalized_residuals, dtype=float)
    offsets = np.asarray(profile.session_offsets, dtype=float)
    shape_jacobian = np.empty((shape.size, center.size), dtype=float)
    offset_jacobian = np.empty((offsets.size, center.size), dtype=float)
    nominal_step = max(float(finite_difference_fraction), 1.0e-7)

    for parameter_index in range(center.size):
        plus_step = min(nominal_step, float(upper[parameter_index] - center[parameter_index]))
        minus_step = min(nominal_step, float(center[parameter_index] - lower[parameter_index]))
        plus_profile = None
        minus_profile = None
        if plus_step > 0.0:
            plus = center.copy()
            plus[parameter_index] += plus_step
            plus_profile = evaluator.evaluate_values(
                coordinates.to_physical(coordinates.denormalize(plus))
            )
        if minus_step > 0.0:
            minus = center.copy()
            minus[parameter_index] -= minus_step
            minus_profile = evaluator.evaluate_values(
                coordinates.to_physical(coordinates.denormalize(minus))
            )
        if plus_profile is not None and minus_profile is not None:
            denominator = plus_step + minus_step
            shape_jacobian[:, parameter_index] = (
                np.asarray(plus_profile.normalized_residuals, dtype=float)
                - np.asarray(minus_profile.normalized_residuals, dtype=float)
            ) / denominator
            offset_jacobian[:, parameter_index] = (
                np.asarray(plus_profile.session_offsets, dtype=float)
                - np.asarray(minus_profile.session_offsets, dtype=float)
            ) / denominator
        elif plus_profile is not None:
            shape_jacobian[:, parameter_index] = (
                np.asarray(plus_profile.normalized_residuals, dtype=float) - shape
            ) / plus_step
            offset_jacobian[:, parameter_index] = (
                np.asarray(plus_profile.session_offsets, dtype=float) - offsets
            ) / plus_step
        elif minus_profile is not None:
            shape_jacobian[:, parameter_index] = (
                shape - np.asarray(minus_profile.normalized_residuals, dtype=float)
            ) / minus_step
            offset_jacobian[:, parameter_index] = (
                offsets - np.asarray(minus_profile.session_offsets, dtype=float)
            ) / minus_step
        else:
            raise ValueError("A phase-search coordinate has no finite-difference support.")
    if not np.all(np.isfinite(shape_jacobian)) or not np.all(
        np.isfinite(offset_jacobian)
    ):
        raise ValueError("Phase-search finite differences produced non-finite geometry.")
    return profile, shape_jacobian, offset_jacobian


def select_phase_information_order(
    *,
    profile: Any,
    session_offset_jacobian: Sequence[Sequence[float]],
    parameter_covariance: Sequence[Sequence[float]],
) -> tuple[int, tuple[int, ...], np.ndarray]:
    """Choose an anchor and session order by posterior-predictive phase entropy."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    jacobian = np.asarray(session_offset_jacobian, dtype=float)
    covariance = np.asarray(parameter_covariance, dtype=float)
    session_count = offsets.size
    if (
        session_count < 2
        or variances.shape != (session_count,)
        or jacobian.ndim != 2
        or jacobian.shape[0] != session_count
        or covariance.shape != (jacobian.shape[1], jacobian.shape[1])
        or not np.all(np.isfinite(variances) & (variances > 0.0))
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("Posterior-predictive session geometry is not aligned.")
    predictive_variances = np.empty((session_count, session_count), dtype=float)
    for anchor in range(session_count):
        differences = jacobian - jacobian[anchor]
        propagated = np.einsum(
            "ij,jk,ik->i",
            differences,
            covariance,
            differences,
        )
        predictive_variances[anchor] = np.maximum(
            variances + variances[anchor] + propagated,
            np.finfo(float).tiny,
        )
        predictive_variances[anchor, anchor] = np.finfo(float).tiny

    # The median log variance is robust to one very remote session.  The
    # maximum term breaks ties in favor of an anchor that also controls the
    # longest baseline.
    anchor_scores = []
    for anchor in range(session_count):
        active = np.delete(predictive_variances[anchor], anchor)
        anchor_scores.append(
            (
                float(np.median(np.log(active))),
                float(np.max(np.log(active))),
                anchor,
            )
        )
    anchor = min(anchor_scores)[2]
    order = tuple(
        sorted(
            (index for index in range(session_count) if index != anchor),
            key=lambda index: (predictive_variances[anchor, index], index),
        )
    )
    return anchor, order, predictive_variances[anchor].copy()


def select_phase_information_tree(
    *,
    profile: Any,
    session_offset_jacobian: Sequence[Sequence[float]],
    parameter_covariance: Sequence[Sequence[float]],
) -> tuple[int, tuple[PhaseInformationEdge, ...]]:
    """Build the minimum-predictive-entropy session linking tree."""

    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    jacobian = np.asarray(session_offset_jacobian, dtype=float)
    covariance = np.asarray(parameter_covariance, dtype=float)
    session_count = offsets.size
    if (
        session_count < 2
        or variances.shape != (session_count,)
        or jacobian.ndim != 2
        or jacobian.shape[0] != session_count
        or covariance.shape != (jacobian.shape[1], jacobian.shape[1])
        or not np.all(np.isfinite(variances) & (variances > 0.0))
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("Posterior-predictive session geometry is not aligned.")

    pairwise = np.full((session_count, session_count), np.inf, dtype=float)
    for left in range(session_count):
        differences = jacobian - jacobian[left]
        propagated = np.einsum(
            "ij,jk,ik->i",
            differences,
            covariance,
            differences,
        )
        pairwise[left] = np.maximum(
            variances + variances[left] + propagated,
            np.finfo(float).tiny,
        )
        pairwise[left, left] = np.inf
    pairwise = 0.5 * (pairwise + pairwise.T)
    root_scores = []
    for root in range(session_count):
        row = pairwise[root, np.isfinite(pairwise[root])]
        root_scores.append(
            (
                float(np.median(np.log(row))),
                float(np.max(np.log(row))),
                root,
            )
        )
    root = min(root_scores)[2]
    connected = {root}
    edges: list[PhaseInformationEdge] = []
    while len(connected) < session_count:
        variance, parent, child = min(
            (
                float(pairwise[parent, child]),
                int(parent),
                int(child),
            )
            for parent in connected
            for child in range(session_count)
            if child not in connected
        )
        edges.append(
            PhaseInformationEdge(
                parent_session_index=parent,
                child_session_index=child,
                predictive_variance=variance,
            )
        )
        connected.add(child)
    return root, tuple(edges)




def _tree_phase_geometry(
    profile: Any,
    session_offset_jacobian: np.ndarray,
    *,
    edges: Sequence[PhaseInformationEdge],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(profile.session_offsets, dtype=float)
    variances = np.asarray(profile.session_offset_variances, dtype=float)
    active_edges = tuple(edges)
    incidence = np.zeros((len(active_edges), offsets.size), dtype=float)
    for edge_index, edge in enumerate(active_edges):
        incidence[edge_index, edge.child_session_index] = 1.0
        incidence[edge_index, edge.parent_session_index] = -1.0
    relative = incidence @ offsets
    relative_jacobian = incidence @ session_offset_jacobian
    noise = incidence @ np.diag(variances) @ incidence.T
    jitter = max(float(np.trace(noise)) * 1.0e-14, 1.0e-24)
    noise += np.eye(len(active_edges), dtype=float) * jitter
    return relative, relative_jacobian, noise


def conditional_integer_path_proposals(
    *,
    relative_phase_center: Sequence[float],
    predictive_covariance: Sequence[Sequence[float]],
    active_session_indices: Sequence[int],
    previous_session_indices: Sequence[int],
    parent_offsets_from_anchor: Sequence[int],
    candidate_count: int,
    max_visited_nodes: int = 2_000_000,
    decimal_precision: int = 64,
) -> tuple[tuple[int, ...], ...]:
    """Extend one partial integer mode under its conditional Gaussian mass."""

    center = np.asarray(relative_phase_center, dtype=float)
    covariance = np.asarray(predictive_covariance, dtype=float)
    active = tuple(int(index) for index in active_session_indices)
    previous = tuple(int(index) for index in previous_session_indices)
    parent = tuple(int(value) for value in parent_offsets_from_anchor)
    if center.shape != (len(active),) or covariance.shape != (
        len(active),
        len(active),
    ):
        raise ValueError("Conditional integer geometry must match active sessions.")
    if not set(previous).issubset(active):
        raise ValueError("Previous sessions must be a subset of active sessions.")
    new_sessions = tuple(index for index in active if index not in set(previous))
    if not new_sessions:
        return (tuple(parent[index] for index in active),)
    active_positions = {session: position for position, session in enumerate(active)}
    old_positions = np.asarray(
        [active_positions[index] for index in previous], dtype=int
    )
    new_positions = np.asarray(
        [active_positions[index] for index in new_sessions], dtype=int
    )
    conditional_center = center[new_positions].copy()
    conditional_covariance = covariance[np.ix_(new_positions, new_positions)].copy()
    if old_positions.size:
        old_covariance = covariance[np.ix_(old_positions, old_positions)]
        cross = covariance[np.ix_(new_positions, old_positions)]
        old_precision = np.linalg.pinv(old_covariance, hermitian=True)
        old_integers = np.asarray([parent[index] for index in previous], dtype=float)
        conditional_center += cross @ old_precision @ (
            old_integers - center[old_positions]
        )
        conditional_covariance -= cross @ old_precision @ cross.T
    conditional_covariance = 0.5 * (
        conditional_covariance + conditional_covariance.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(conditional_covariance)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    conditional_covariance = (
        eigenvectors
        @ np.diag(np.clip(eigenvalues, scale * 1.0e-12, None))
        @ eigenvectors.T
    )
    result = solve_integer_least_squares(
        conditional_center,
        conditional_covariance,
        candidate_count=max(int(candidate_count), 1),
        max_visited_nodes=max(int(max_visited_nodes), 1),
        decimal_precision=max(int(decimal_precision), 32),
    )
    proposals = []
    for candidate in result.candidates:
        new_values = dict(zip(new_sessions, candidate.integer_vector, strict=True))
        proposals.append(
            tuple(
                parent[session]
                if session in set(previous)
                else int(new_values[session])
                for session in active
            )
        )
    return tuple(proposals)


def _select_delayed_integer_modes(
    modes: Sequence[DelayedIntegerPhaseMode],
    *,
    active_session_indices: Sequence[int],
    maximum_count: int,
    protected_per_parent: int,
    maximum_per_path: int,
    minimum_separation: float,
) -> list[DelayedIntegerPhaseMode]:
    active = tuple(int(index) for index in active_session_indices)
    ranked = sorted(
        modes,
        key=lambda mode: (
            -mode.laplace_log_evidence,
            mode.objective_chi2,
            mode.source_orbit_mode_index,
            mode.parent_mode_index,
        ),
    )
    target = min(max(int(maximum_count), 1), len(ranked))
    selected: list[DelayedIntegerPhaseMode] = []
    path_counts: dict[tuple[int, ...], int] = {}

    def path_key(mode: DelayedIntegerPhaseMode) -> tuple[int, ...]:
        return tuple(mode.integer_offsets_from_anchor[index] for index in active)

    def take(mode: DelayedIntegerPhaseMode) -> bool:
        path = path_key(mode)
        if path_counts.get(path, 0) >= max(int(maximum_per_path), 1):
            return False
        if any(
            path_key(existing) == path
            and np.linalg.norm(
                mode.normalized_search_values - existing.normalized_search_values
            ) < max(float(minimum_separation), 0.0)
            for existing in selected
        ):
            return False
        selected.append(mode)
        path_counts[path] = path_counts.get(path, 0) + 1
        return True

    quota = max(int(protected_per_parent), 0)
    by_parent: dict[int, list[DelayedIntegerPhaseMode]] = {}
    for mode in ranked:
        by_parent.setdefault(mode.parent_mode_index, []).append(mode)
    for parent_index in sorted(
        by_parent,
        key=lambda index: (
            -by_parent[index][0].laplace_log_evidence,
            by_parent[index][0].objective_chi2,
            index,
        ),
    ):
        kept = 0
        for mode in by_parent[parent_index]:
            if take(mode):
                kept += 1
            if kept >= quota or len(selected) >= target:
                break
        if len(selected) >= target:
            return selected
    for mode in ranked:
        if len(selected) >= target:
            break
        take(mode)
    return selected


def _wrapped_phase_residuals(
    relative_phase: Sequence[float],
    edge_variances: Sequence[float],
    temperature: float,
) -> np.ndarray:
    """Return a one-periodic, locally Gaussian bridge residual.

    This residual defines a proposal likelihood. Near each integer, its square
    is the usual local phase chi-square. Periodicity leaves the long-gap pulse
    count unresolved until the continuous orbit reaches a compatible basin.
    Subsequent evaluation uses the exact integer-conditioned likelihood.
    """

    phase = np.asarray(relative_phase, dtype=float)
    variances = np.asarray(edge_variances, dtype=float)
    if phase.ndim != 1 or variances.shape != phase.shape:
        raise ValueError("Wrapped phase geometry must be one-dimensional and aligned.")
    if not np.all(np.isfinite(phase)) or not np.all(
        np.isfinite(variances) & (variances > 0.0)
    ):
        raise ValueError("Wrapped phase geometry must be finite and positive.")
    broadening = max(float(temperature), 0.0)
    scale = np.sqrt(variances + broadening * broadening)
    return np.sin(math.pi * phase) / (math.pi * scale)


def _wrapped_phase_sigma_point_starts(
    *,
    center: Sequence[float],
    covariance: Sequence[Sequence[float]],
    count: int,
    radius: float,
    lower: Sequence[float],
    upper: Sequence[float],
    interior: Sequence[float],
) -> tuple[np.ndarray, ...]:
    """Generate deterministic posterior sigma starts in normalized coordinates."""

    mean = np.asarray(center, dtype=float)
    covariance_array = np.asarray(covariance, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    interior_array = np.asarray(interior, dtype=float)
    if (
        mean.ndim != 1
        or covariance_array.shape != (mean.size, mean.size)
        or lower_array.shape != mean.shape
        or upper_array.shape != mean.shape
        or interior_array.shape != mean.shape
    ):
        raise ValueError("Wrapped sigma-point geometry is not aligned.")
    symmetric = 0.5 * (covariance_array + covariance_array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1]
    scale_radius = max(float(radius), 0.0)
    starts: list[np.ndarray] = [
        np.clip(mean, lower_array + interior_array, upper_array - interior_array)
    ]
    pair_count = max((max(int(count), 1) - 1) // 2, 0)
    for pair_index in range(pair_count):
        eigen_index = int(order[pair_index % mean.size])
        variance = max(float(eigenvalues[eigen_index]), 0.0)
        direction = (
            scale_radius
            * math.sqrt(variance)
            * eigenvectors[:, eigen_index]
        )
        starts.append(
            np.clip(
                mean + direction,
                lower_array + interior_array,
                upper_array - interior_array,
            )
        )
        if len(starts) >= max(int(count), 1):
            break
        starts.append(
            np.clip(
                mean - direction,
                lower_array + interior_array,
                upper_array - interior_array,
            )
        )
        if len(starts) >= max(int(count), 1):
            break
    unique: list[np.ndarray] = []
    for start in starts:
        if not any(np.linalg.norm(start - existing) < 1.0e-10 for existing in unique):
            unique.append(start)
    return tuple(unique[: max(int(count), 1)])


def _wrapped_phase_orbit_alias_starts(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    center: Sequence[float],
    maximum_alias_count: int,
    orbital_phase_points: int,
    lower: Sequence[float],
    upper: Sequence[float],
    interior: Sequence[float],
) -> tuple[np.ndarray, ...]:
    """Generate DD orbital-frequency aliases from the observation span."""

    normalized_center = np.asarray(center, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    interior_array = np.asarray(interior, dtype=float)
    if (
        normalized_center.shape != coordinates.search_center.shape
        or lower_array.shape != normalized_center.shape
        or upper_array.shape != normalized_center.shape
        or interior_array.shape != normalized_center.shape
    ):
        raise ValueError("Wrapped orbital-alias geometry is not aligned.")
    try:
        profile = evaluator.evaluate_values(
            coordinates.to_physical(coordinates.denormalize(normalized_center))
        )
    except (ArithmeticError, RuntimeError, ValueError):
        return ()
    epochs = np.asarray(getattr(profile, "session_epochs", ()), dtype=float)
    epochs = epochs[np.isfinite(epochs)]
    if epochs.size < 2:
        return ()
    pb_index = coordinates.parameter_names.index("PB")
    phase_index = coordinates.parameter_names.index("T0")
    alias_count = estimate_orbital_alias_count(
        observation_span_days=float(np.max(epochs) - np.min(epochs)),
        pb_lower_days=float(coordinates.physical_lower[pb_index]),
        pb_upper_days=float(coordinates.physical_upper[pb_index]),
        maximum_count=max(int(maximum_alias_count), 1),
    )
    if alias_count <= 0:
        return ()
    frequencies = np.linspace(
        1.0 / float(coordinates.physical_upper[pb_index]),
        1.0 / float(coordinates.physical_lower[pb_index]),
        alias_count,
        endpoint=True,
    )
    phase_count = max(int(orbital_phase_points), 1)
    phases = np.linspace(
        float(coordinates.search_lower[phase_index]),
        float(coordinates.search_upper[phase_index]),
        phase_count,
        endpoint=False,
    )
    starts: list[np.ndarray] = []
    search_center = coordinates.denormalize(normalized_center)
    for frequency in frequencies:
        for orbital_phase in phases:
            search = search_center.copy()
            search[pb_index] = math.log(1.0 / float(frequency))
            search[phase_index] = float(orbital_phase)
            starts.append(
                np.clip(
                    coordinates.normalize(search),
                    lower_array + interior_array,
                    upper_array - interior_array,
                )
            )
    return tuple(starts)


def refine_wrapped_phase_modes(
    *,
    evaluator: Any,
    coordinates: DDSearchCoordinateSystem,
    initial_records: Sequence[tuple[int, OrbitPosteriorMode, np.ndarray]],
    information_tree: Sequence[PhaseInformationEdge],
    session_count: int,
    config: DelayedIntegerPhaseConfig,
    lower: np.ndarray,
    upper: np.ndarray,
    interior: np.ndarray,
) -> tuple[list[DelayedIntegerPhaseMode], tuple[dict[str, Any], ...]]:
    """Assimilate modulo-one phase bridges before integer path inference.

    This annealed multimode proposal uses the entropy tree to order incoming
    information. It assigns no integer to an edge; a separate exact lattice
    stage decodes integers afterwards.
    """

    records = list(initial_records)
    if not records:
        raise ValueError("Wrapped phase refinement requires at least one orbit mode.")
    if len(information_tree) != session_count - 1:
        raise ValueError("The wrapped phase tree must span all source sessions.")
    schedule = tuple(
        max(float(value), 0.0)
        for value in config.wrapped_phase_temperature_schedule
    )
    if not schedule:
        schedule = (0.025,)
    initial_count = min(
        max(int(config.initial_relative_session_count), 1),
        session_count - 1,
    )
    maximum_count = session_count - 1
    if config.maximum_relative_session_count is not None:
        maximum_count = min(
            max(int(config.maximum_relative_session_count), initial_count),
            session_count - 1,
        )
    batch_size = max(int(config.session_batch_size), 1)
    stage_counts = list(range(initial_count, maximum_count + 1, batch_size))
    if not stage_counts or stage_counts[-1] != maximum_count:
        stage_counts.append(maximum_count)

    current: list[DelayedIntegerPhaseMode] = []
    zero_path = tuple(0 for _ in range(session_count))
    for mode_index, mode, normalized in records:
        current.append(
            DelayedIntegerPhaseMode(
                parameter_values=np.asarray(mode.parameter_values, dtype=float).copy(),
                normalized_search_values=np.asarray(normalized, dtype=float).copy(),
                integer_offsets_from_anchor=zero_path,
                source_orbit_mode_index=int(mode_index),
                parent_mode_index=int(mode_index),
                objective_chi2=float(mode.slope_chi2),
                shape_chi2=float(mode.slope_chi2),
                ambiguity_chi2=0.0,
                prior_chi2=float(np.asarray(normalized) @ np.asarray(normalized)),
                laplace_log_evidence=float(mode.laplace_log_evidence),
                nfev=0,
                success=bool(mode.success),
            )
        )

    assimilation_steps = [(count, schedule[0]) for count in stage_counts]
    assimilation_steps.extend(
        (maximum_count, temperature) for temperature in schedule[1:]
    )

    history: list[dict[str, Any]] = []
    full_edge_snapshots: list[tuple[DelayedIntegerPhaseMode, ...]] = []
    covariance_by_source = {
        int(mode_index): np.asarray(mode.scaled_covariance, dtype=float)
        for mode_index, mode, _ in records
    }
    for stage_index, (active_count, temperature) in enumerate(assimilation_steps):
        active_edges = tuple(information_tree[:active_count])
        proposals: list[DelayedIntegerPhaseMode] = []
        failed_count = 0
        stage_parents: Sequence[DelayedIntegerPhaseMode] = current
        if active_count == maximum_count and not full_edge_snapshots:
            expanded_parents: list[DelayedIntegerPhaseMode] = []
            for parent in current:
                covariance = covariance_by_source.get(
                    int(parent.source_orbit_mode_index)
                )
                if covariance is None:
                    expanded_parents.append(parent)
                    continue
                for start in _wrapped_phase_sigma_point_starts(
                    center=parent.normalized_search_values,
                    covariance=covariance,
                    count=int(config.wrapped_phase_sigma_points_per_mode),
                    radius=float(config.wrapped_phase_sigma_point_radius),
                    lower=lower,
                    upper=upper,
                    interior=interior,
                ):
                    expanded_parents.append(
                        replace(
                            parent,
                            parameter_values=coordinates.to_physical(
                                coordinates.denormalize(start)
                            ),
                            normalized_search_values=start,
                        )
                    )
            if current:
                alias_seed = current[0]
                alias_starts = _wrapped_phase_orbit_alias_starts(
                    evaluator=evaluator,
                    coordinates=coordinates,
                    center=records[0][2],
                    maximum_alias_count=int(
                        config.wrapped_phase_orbital_alias_count
                    ),
                    orbital_phase_points=int(
                        config.wrapped_phase_orbital_phase_points
                    ),
                    lower=lower,
                    upper=upper,
                    interior=interior,
                )
                for start in alias_starts:
                    if any(
                        np.linalg.norm(
                            start - parent.normalized_search_values
                        )
                        < 1.0e-8
                        for parent in expanded_parents
                    ):
                        continue
                    expanded_parents.append(
                        replace(
                            alias_seed,
                            parameter_values=coordinates.to_physical(
                                coordinates.denormalize(start)
                            ),
                            normalized_search_values=start,
                        )
                    )
            stage_parents = expanded_parents
        for parent_index, parent in enumerate(stage_parents):
            try:
                normalized_start = np.asarray(
                    parent.normalized_search_values, dtype=float
                )
                profile = evaluator.evaluate_values(
                    coordinates.to_physical(coordinates.denormalize(normalized_start))
                )
                zero_jacobian = np.zeros(
                    (int(profile.session_count), normalized_start.size),
                    dtype=float,
                )
                _, _, phase_noise = _tree_phase_geometry(
                    profile,
                    zero_jacobian,
                    edges=active_edges,
                )
                edge_variances = np.maximum(
                    np.diag(phase_noise), np.finfo(float).tiny
                )
                shape_size = int(np.asarray(profile.normalized_residuals).size)
                residual_size = shape_size + active_count + lower.size
                failure = np.full(residual_size, 1.0e12, dtype=float)

                def residual_vector(values: np.ndarray) -> np.ndarray:
                    try:
                        physical = coordinates.to_physical(
                            coordinates.denormalize(values)
                        )
                        candidate = evaluator.evaluate_values(physical)
                        offsets = np.asarray(candidate.session_offsets, dtype=float)
                        relative = np.asarray(
                            [
                                offsets[edge.child_session_index]
                                - offsets[edge.parent_session_index]
                                for edge in active_edges
                            ],
                            dtype=float,
                        )
                        wrapped = _wrapped_phase_residuals(
                            relative,
                            edge_variances,
                            temperature,
                        )
                        result = np.concatenate(
                            (
                                np.asarray(candidate.normalized_residuals, dtype=float),
                                wrapped,
                                values,
                            )
                        )
                        return result if np.all(np.isfinite(result)) else failure
                    except (ArithmeticError, RuntimeError, ValueError):
                        return failure

                radius = max(float(config.wrapped_phase_trust_radius), 1.0e-6)
                local_lower = np.maximum(lower, normalized_start - radius)
                local_upper = np.minimum(upper, normalized_start + radius)
                start = np.clip(
                    normalized_start,
                    local_lower + interior,
                    local_upper - interior,
                )
                optimization = least_squares(
                    residual_vector,
                    start,
                    bounds=(local_lower, local_upper),
                    method="trf",
                    jac="2-point",
                    diff_step=max(
                        float(config.finite_difference_fraction), 1.0e-7
                    ),
                    max_nfev=max(int(config.wrapped_phase_max_nfev), 1),
                    ftol=1.0e-8,
                    xtol=1.0e-8,
                    gtol=1.0e-8,
                )
                final_normalized = np.asarray(optimization.x, dtype=float)
                final_values = coordinates.to_physical(
                    coordinates.denormalize(final_normalized)
                )
                final_profile = evaluator.evaluate_values(final_values)
                final_offsets = np.asarray(final_profile.session_offsets, dtype=float)
                final_relative = np.asarray(
                    [
                        final_offsets[edge.child_session_index]
                        - final_offsets[edge.parent_session_index]
                        for edge in active_edges
                    ],
                    dtype=float,
                )
                final_wrapped = _wrapped_phase_residuals(
                    final_relative,
                    edge_variances,
                    temperature,
                )
                final_shape = np.asarray(
                    final_profile.normalized_residuals, dtype=float
                )
                shape_chi2 = float(final_shape @ final_shape)
                wrapped_chi2 = float(final_wrapped @ final_wrapped)
                prior_chi2 = float(final_normalized @ final_normalized)
                objective_chi2 = shape_chi2 + wrapped_chi2 + prior_chi2
                laplace = float(
                    parent.laplace_log_evidence
                    - 0.5 * (wrapped_chi2 + max(shape_chi2 - parent.shape_chi2, 0.0))
                )
                proposals.append(
                    DelayedIntegerPhaseMode(
                        parameter_values=final_values,
                        normalized_search_values=final_normalized,
                        integer_offsets_from_anchor=zero_path,
                        source_orbit_mode_index=parent.source_orbit_mode_index,
                        parent_mode_index=parent_index,
                        objective_chi2=objective_chi2,
                        shape_chi2=shape_chi2,
                        ambiguity_chi2=0.0,
                        prior_chi2=prior_chi2,
                        laplace_log_evidence=laplace,
                        nfev=parent.nfev + int(optimization.nfev),
                        success=parent.success and bool(optimization.success),
                        wrapped_phase_chi2=wrapped_chi2,
                    )
                )
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                failed_count += 1

        if not proposals:
            raise RuntimeError(
                f"Wrapped phase assimilation failed at stage {stage_index}."
            )
        ranked = sorted(
            proposals,
            key=lambda mode: (mode.objective_chi2, mode.source_orbit_mode_index),
        )
        selected: list[DelayedIntegerPhaseMode] = []

        def take_stage_mode(candidate: DelayedIntegerPhaseMode) -> None:
            if len(selected) >= max(int(config.max_modes), 1):
                return
            if all(
                np.linalg.norm(
                    candidate.normalized_search_values
                    - existing.normalized_search_values
                )
                >= max(float(config.minimum_mode_separation), 0.0)
                for existing in selected
            ):
                selected.append(candidate)

        by_source: dict[int, list[DelayedIntegerPhaseMode]] = {}
        for candidate in ranked:
            by_source.setdefault(candidate.source_orbit_mode_index, []).append(candidate)
        for source_index in sorted(by_source):
            take_stage_mode(by_source[source_index][0])
        for candidate in ranked:
            take_stage_mode(candidate)
            if len(selected) >= max(int(config.max_modes), 1):
                break
        if not selected:
            selected.append(ranked[0])
        current = selected
        if active_count == maximum_count:
            full_edge_snapshots.append(tuple(current))
        history.append(
            {
                "assimilation_index": stage_index,
                "active_relative_session_count": active_count,
                "active_phase_edges": tuple(
                    (
                        edge.parent_session_index,
                        edge.child_session_index,
                        edge.predictive_variance,
                    )
                    for edge in active_edges
                ),
                "temperature": temperature,
                "input_mode_count": len(proposals) + failed_count,
                "proposal_count": len(proposals),
                "failed_mode_count": failed_count,
                "selected_mode_count": len(current),
                "best_objective_chi2": min(mode.objective_chi2 for mode in current),
                "best_shape_chi2": min(mode.shape_chi2 for mode in current),
                "best_wrapped_phase_chi2": min(
                    mode.wrapped_phase_chi2 for mode in current
                ),
            }
        )

    portfolio: list[DelayedIntegerPhaseMode] = []
    target_count = max(int(config.max_modes), 1)

    def take(candidate: DelayedIntegerPhaseMode) -> None:
        if len(portfolio) >= target_count:
            return
        if any(
            np.linalg.norm(
                candidate.normalized_search_values - existing.normalized_search_values
            )
            < 1.0e-6
            for existing in portfolio
        ):
            return
        portfolio.append(candidate)

    # One protected representative per bridge temperature prevents a narrow
    # late-temperature basin from erasing a broader compatible orbit mode.
    for snapshot in full_edge_snapshots:
        take(min(snapshot, key=lambda mode: mode.objective_chi2))
    for candidate in sorted(
        (mode for snapshot in full_edge_snapshots for mode in snapshot),
        key=lambda mode: (mode.shape_chi2, mode.objective_chi2),
    ):
        take(candidate)
    if not portfolio:
        portfolio = current[:target_count]
    return portfolio, tuple(history)


def infer_delayed_integer_phase_modes(
    *,
    evaluator: Any,
    posterior: OrbitPosteriorResult,
    coordinates: DDSearchCoordinateSystem,
    base_parameter_values: Sequence[float],
    structure_id: str,
    binary_model: str | None,
    stage_index: int,
    config: DelayedIntegerPhaseConfig | None = None,
    resume_modes: Sequence[DelayedIntegerPhaseMode] | None = None,
    resume_relative_session_count: int = 0,
) -> DelayedIntegerPhaseResult:
    """Assimilate low-entropy phase factors before committing a global path.

    Each stage relinearizes the joint spin-orbit geometry and can revise every
    previously represented integer. Integer labels identify posterior islands
    without committing to a chronological bridge choice.
    """

    active = config or DelayedIntegerPhaseConfig()
    base = np.asarray(base_parameter_values, dtype=float)
    if base.shape != coordinates.center_values.shape:
        raise ValueError("Base parameter values must match the DD search coordinates.")
    if not posterior.modes:
        raise ValueError("Delayed integer inference requires orbit posterior modes.")
    lower = coordinates.normalized_lower
    upper = coordinates.normalized_upper
    interior = np.minimum(np.maximum((upper - lower) * 1.0e-12, 1.0e-12), 1.0e-6)

    initial_records: list[tuple[int, OrbitPosteriorMode, np.ndarray]] = []
    for mode_index, mode in enumerate(posterior.modes):
        normalized = np.asarray(mode.normalized_search_values, dtype=float)
        if normalized.shape != lower.shape or not np.all(np.isfinite(normalized)):
            continue
        if any(
            np.linalg.norm(normalized - existing[2])
            < max(float(active.minimum_mode_separation), 1.0e-6)
            for existing in initial_records
        ):
            continue
        initial_records.append((mode_index, mode, normalized.copy()))
    if not initial_records:
        raise RuntimeError("Orbit posterior contains no valid independent phase starts.")

    reference_index, reference_mode, reference_normalized = initial_records[0]
    reference_profile, _, reference_offset_jacobian = (
        _linearize_profile_in_search_coordinates(
            evaluator=evaluator,
            coordinates=coordinates,
            normalized_search_values=reference_normalized,
            finite_difference_fraction=float(active.finite_difference_fraction),
        )
    )
    anchor, information_tree = select_phase_information_tree(
        profile=reference_profile,
        session_offset_jacobian=reference_offset_jacobian,
        parameter_covariance=np.asarray(reference_mode.scaled_covariance, dtype=float),
    )
    session_order = tuple(edge.child_session_index for edge in information_tree)
    session_count = int(reference_profile.session_count)
    initial_count = min(
        max(int(active.initial_relative_session_count), 1),
        session_count - 1,
    )
    batch_size = max(int(active.session_batch_size), 1)
    resume_count = max(int(resume_relative_session_count), 0)
    if resume_count >= session_count:
        raise ValueError("Resume session count exceeds the phase-tree dimension.")
    first_count = (
        initial_count
        if resume_count == 0
        else min(resume_count + batch_size, session_count - 1)
    )
    stage_counts = list(range(first_count, session_count - 1, batch_size))
    if not stage_counts or stage_counts[-1] != session_count - 1:
        stage_counts.append(session_count - 1)
    if active.maximum_relative_session_count is not None:
        maximum_count = min(
            max(int(active.maximum_relative_session_count), first_count),
            session_count - 1,
        )
        stage_counts = [count for count in stage_counts if count <= maximum_count]
        if not stage_counts or stage_counts[-1] != maximum_count:
            stage_counts.append(maximum_count)
    if (
        resume_count == 0
        and active.wrapped_phase_assimilation_enabled
        and stage_counts
    ):
        stage_counts = [stage_counts[-1]]

    modes: list[DelayedIntegerPhaseMode] = []
    wrapped_phase_history: tuple[dict[str, Any], ...] = ()
    if resume_modes is not None:
        for mode in resume_modes:
            if (
                len(mode.integer_offsets_from_anchor) != session_count
                or mode.normalized_search_values.shape != lower.shape
            ):
                raise ValueError("Resume modes do not match the current phase tree.")
            modes.append(mode)
        if not modes:
            raise ValueError("At least one resume mode is required.")
    else:
        if resume_count != 0:
            raise ValueError("A nonzero resume count requires resume modes.")
        if active.wrapped_phase_assimilation_enabled:
            modes, wrapped_phase_history = refine_wrapped_phase_modes(
                evaluator=evaluator,
                coordinates=coordinates,
                initial_records=initial_records,
                information_tree=information_tree,
                session_count=session_count,
                config=active,
                lower=lower,
                upper=upper,
                interior=interior,
            )
        else:
            for mode_index, mode, normalized in initial_records:
                full_path = tuple(0 for _ in range(session_count))
                modes.append(
                    DelayedIntegerPhaseMode(
                        parameter_values=np.asarray(
                            mode.parameter_values, dtype=float
                        ).copy(),
                        normalized_search_values=normalized,
                        integer_offsets_from_anchor=full_path,
                        source_orbit_mode_index=mode_index,
                        parent_mode_index=mode_index,
                        objective_chi2=math.inf,
                        shape_chi2=float(mode.slope_chi2),
                        ambiguity_chi2=math.inf,
                        prior_chi2=float(normalized @ normalized),
                        laplace_log_evidence=float(mode.laplace_log_evidence),
                        nfev=0,
                        success=bool(mode.success),
                    )
                )

    history: list[dict[str, Any]] = []
    for assimilation_index, active_count in enumerate(stage_counts):
        active_sessions = session_order[:active_count]
        active_edges = information_tree[:active_count]
        proposals: list[DelayedIntegerPhaseMode] = []
        failed_count = 0
        for parent_index, parent in enumerate(modes):
            try:
                profile, shape_jacobian, offset_jacobian = (
                    _linearize_profile_in_search_coordinates(
                        evaluator=evaluator,
                        coordinates=coordinates,
                        normalized_search_values=parent.normalized_search_values,
                        finite_difference_fraction=float(
                            active.finite_difference_fraction
                        ),
                    )
                )
                shape_precision = shape_jacobian.T @ shape_jacobian + np.eye(
                    shape_jacobian.shape[1], dtype=float
                )
                shape_covariance = np.linalg.pinv(
                    0.5 * (shape_precision + shape_precision.T),
                    hermitian=True,
                )
                relative, relative_jacobian, phase_noise = _tree_phase_geometry(
                    profile,
                    offset_jacobian,
                    edges=active_edges,
                )
                predictive_covariance = (
                    phase_noise
                    + relative_jacobian @ shape_covariance @ relative_jacobian.T
                )
                predictive_covariance = 0.5 * (
                    predictive_covariance + predictive_covariance.T
                )
                predictive_covariance += np.eye(active_count) * max(
                    float(np.trace(predictive_covariance)) * 1.0e-12,
                    1.0e-18,
                )
                candidate_target = max(
                    int(active.integer_candidates_per_mode), 1
                )
                previous_count = (
                    stage_counts[assimilation_index - 1]
                    if assimilation_index > 0
                    else resume_count
                )
                previous_sessions = session_order[:previous_count]
                parent_edge_offsets = tuple(
                    parent.integer_offsets_from_anchor[edge.child_session_index]
                    - parent.integer_offsets_from_anchor[edge.parent_session_index]
                    for edge in information_tree
                )
                candidate_paths: list[tuple[int, ...]] = []
                try:
                    candidate_paths.extend(
                        conditional_integer_path_proposals(
                            relative_phase_center=relative,
                            predictive_covariance=predictive_covariance,
                            active_session_indices=tuple(range(active_count)),
                            previous_session_indices=tuple(range(previous_count)),
                            parent_offsets_from_anchor=parent_edge_offsets,
                            candidate_count=candidate_target,
                            max_visited_nodes=int(
                                active.max_integer_visited_nodes
                            ),
                            decimal_precision=int(
                                active.integer_decimal_precision
                            ),
                        )
                    )
                except (ArithmeticError, RuntimeError, ValueError):
                    pass

                # A small global refresh lets a parent revise an earlier
                # integer family without allowing refresh proposals to erase
                # every conditionally extended descendant.
                if previous_sessions:
                    try:
                        integer_result = solve_integer_least_squares(
                            relative,
                            predictive_covariance,
                            candidate_count=max(candidate_target // 3, 1),
                            max_visited_nodes=max(
                                int(active.max_integer_visited_nodes), 1
                            ),
                            decimal_precision=max(
                                int(active.integer_decimal_precision), 32
                            ),
                        )
                        candidate_paths.extend(
                            tuple(
                                int(value)
                                for value in candidate.integer_vector
                            )
                            for candidate in integer_result.candidates
                        )
                    except (ArithmeticError, RuntimeError, ValueError):
                        pass
                if not candidate_paths:
                    candidate_paths.append(
                        tuple(
                            int(value)
                            for value in np.rint(relative).astype(np.int64)
                        )
                    )
                inherited_path = tuple(
                    parent_edge_offsets[index]
                    if index < previous_count
                    else int(np.rint(relative[index]))
                    for index in range(active_count)
                )
                if inherited_path not in candidate_paths:
                    candidate_paths.insert(0, inherited_path)
                candidate_paths = list(dict.fromkeys(candidate_paths))

                predictive_precision = np.linalg.pinv(
                    predictive_covariance,
                    hermitian=True,
                )
                gain = shape_covariance @ relative_jacobian.T @ predictive_precision
                phase_cholesky = np.linalg.cholesky(phase_noise)
                shape_size = int(np.asarray(profile.normalized_residuals).size)
                residual_size = shape_size + active_count + lower.size
                failure = np.full(residual_size, 1.0e12, dtype=float)

                for candidate_path in candidate_paths:
                    integers = np.asarray(candidate_path, dtype=float)
                    conditional = np.clip(
                        parent.normalized_search_values
                        + gain @ (integers - relative),
                        lower + interior,
                        upper - interior,
                    )
                    starts = [("gaussian-conditional", conditional)]
                    if candidate_path == inherited_path:
                        starts.append(
                            (
                                "parent-posterior-center",
                                np.clip(
                                    parent.normalized_search_values,
                                    lower + interior,
                                    upper - interior,
                                ),
                            )
                        )

                    def residual_vector(normalized_values: np.ndarray) -> np.ndarray:
                        try:
                            values = coordinates.to_physical(
                                coordinates.denormalize(normalized_values)
                            )
                            candidate_profile = evaluator.evaluate_values(values)
                            shape_residual = np.asarray(
                                candidate_profile.normalized_residuals,
                                dtype=float,
                            )
                            candidate_offsets = np.asarray(
                                candidate_profile.session_offsets,
                                dtype=float,
                            )
                            candidate_relative = np.asarray(
                                [
                                    candidate_offsets[edge.child_session_index]
                                    - candidate_offsets[edge.parent_session_index]
                                    for edge in active_edges
                                ],
                                dtype=float,
                            )
                            phase_residual = np.linalg.solve(
                                phase_cholesky,
                                candidate_relative - integers,
                            )
                            result = np.concatenate(
                                (shape_residual, phase_residual, normalized_values)
                            )
                            return result if np.all(np.isfinite(result)) else failure
                        except (ArithmeticError, RuntimeError, ValueError):
                            return failure

                    for start_source, start_center in starts:
                        radius = max(float(active.trust_radius), 1.0e-6)
                        local_lower = np.maximum(lower, start_center - radius)
                        local_upper = np.minimum(upper, start_center + radius)
                        start = np.clip(
                            start_center,
                            local_lower + interior,
                            local_upper - interior,
                        )
                        optimization = least_squares(
                            residual_vector,
                            start,
                            bounds=(local_lower, local_upper),
                            method="trf",
                            jac="2-point",
                            diff_step=max(
                                float(active.finite_difference_fraction), 1.0e-7
                            ),
                            max_nfev=max(int(active.max_nfev_per_candidate), 1),
                            ftol=1.0e-8,
                            xtol=1.0e-8,
                            gtol=1.0e-8,
                        )
                        final_normalized = np.asarray(optimization.x, dtype=float)
                        final_values = coordinates.to_physical(
                            coordinates.denormalize(final_normalized)
                        )
                        final_profile = evaluator.evaluate_values(final_values)
                        final_shape = np.asarray(
                            final_profile.normalized_residuals, dtype=float
                        )
                        final_offsets = np.asarray(
                            final_profile.session_offsets,
                            dtype=float,
                        )
                        final_relative = np.asarray(
                            [
                                final_offsets[edge.child_session_index]
                                - final_offsets[edge.parent_session_index]
                                for edge in active_edges
                            ],
                            dtype=float,
                        )
                        final_phase = np.linalg.solve(
                            phase_cholesky, final_relative - integers
                        )
                        shape_chi2 = float(final_shape @ final_shape)
                        ambiguity_chi2 = float(final_phase @ final_phase)
                        prior_chi2 = float(final_normalized @ final_normalized)
                        objective_chi2 = shape_chi2 + ambiguity_chi2 + prior_chi2
                        precision = (
                            np.asarray(optimization.jac, dtype=float).T
                            @ np.asarray(optimization.jac, dtype=float)
                        )
                        sign, logdet = np.linalg.slogdet(
                            0.5 * (precision + precision.T)
                        )
                        laplace = (
                            -0.5 * objective_chi2
                            - 0.5 * float(logdet)
                            + 0.5 * lower.size * math.log(2.0 * math.pi)
                            if sign > 0 and np.isfinite(logdet)
                            else -0.5 * objective_chi2
                        )
                        full_path = [0 for _ in range(session_count)]
                        full_path[anchor] = 0
                        for edge, integer in zip(
                            active_edges, candidate_path, strict=True
                        ):
                            full_path[edge.child_session_index] = (
                                full_path[edge.parent_session_index] + int(integer)
                            )
                        proposals.append(
                            DelayedIntegerPhaseMode(
                                parameter_values=final_values,
                                normalized_search_values=final_normalized,
                                integer_offsets_from_anchor=tuple(full_path),
                                source_orbit_mode_index=parent.source_orbit_mode_index,
                                parent_mode_index=parent_index,
                                objective_chi2=objective_chi2,
                                shape_chi2=shape_chi2,
                                ambiguity_chi2=ambiguity_chi2,
                                prior_chi2=prior_chi2,
                                laplace_log_evidence=float(laplace),
                                nfev=parent.nfev + int(optimization.nfev),
                                success=parent.success and bool(optimization.success),
                            )
                        )
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                failed_count += 1

        if not proposals:
            raise RuntimeError(
                f"Delayed integer phase assimilation failed at stage {assimilation_index}."
            )
        modes = _select_delayed_integer_modes(
            proposals,
            active_session_indices=active_sessions,
            maximum_count=int(active.max_modes),
            protected_per_parent=int(active.protected_modes_per_parent),
            maximum_per_path=int(active.maximum_modes_per_integer_path),
            minimum_separation=float(active.minimum_mode_separation),
        )
        history.append(
            {
                "assimilation_index": assimilation_index,
                "active_relative_session_count": active_count,
                "active_session_indices": active_sessions,
                "active_phase_edges": tuple(
                    (
                        edge.parent_session_index,
                        edge.child_session_index,
                        edge.predictive_variance,
                    )
                    for edge in active_edges
                ),
                "input_mode_count": len(proposals) + failed_count,
                "proposal_count": len(proposals),
                "failed_parent_count": failed_count,
                "selected_mode_count": len(modes),
                "unique_active_path_count": len(
                    {
                        tuple(
                            mode.integer_offsets_from_anchor[index]
                            for index in active_sessions
                        )
                        for mode in modes
                    }
                ),
                "best_objective_chi2": min(mode.objective_chi2 for mode in modes),
                "best_shape_chi2": min(mode.shape_chi2 for mode in modes),
                "best_ambiguity_chi2": min(mode.ambiguity_chi2 for mode in modes),
                "modes": tuple(
                    {
                        "parameter_values": mode.parameter_values,
                        "normalized_search_values": mode.normalized_search_values,
                        "integer_offsets_from_anchor": (
                            mode.integer_offsets_from_anchor
                        ),
                        "source_orbit_mode_index": (
                            mode.source_orbit_mode_index
                        ),
                        "objective_chi2": mode.objective_chi2,
                        "shape_chi2": mode.shape_chi2,
                        "ambiguity_chi2": mode.ambiguity_chi2,
                        "prior_chi2": mode.prior_chi2,
                        "laplace_log_evidence": mode.laplace_log_evidence,
                        "nfev": mode.nfev,
                        "success": mode.success,
                        "wrapped_phase_chi2": mode.wrapped_phase_chi2,
                    }
                    for mode in modes
                ),
            }
        )

    modes.sort(key=lambda mode: (-mode.laplace_log_evidence, mode.objective_chi2))
    branches = []
    for mode_index, mode in enumerate(modes):
        offsets = np.asarray(mode.integer_offsets_from_anchor, dtype=np.int64)
        session_zero_offsets = offsets - offsets[0]
        relative_to_zero = tuple(int(value) for value in session_zero_offsets[1:])
        branches.append(
            ActiveTimingBranch(
                branch_id=(
                    f"stage_{int(stage_index):02d}_delayed_integer_phase_"
                    f"{mode_index:04d}"
                ),
                structure_id=str(structure_id),
                binary_model=(str(binary_model).upper() if binary_model else None),
                wraps=integer_session_offsets_to_gap_wraps(relative_to_zero),
                theta_names=coordinates.parameter_names,
                theta_offset=np.asarray(mode.parameter_values - base, dtype=float),
                log_prior=float(mode.laplace_log_evidence),
                proposal_source="apollo-delayed-integer-phase-posterior",
                diagnostics={
                    "proposal_family": "entropy-ordered-delayed-integer-phase",
                    "mode_index": mode_index,
                    "source_orbit_mode_index": mode.source_orbit_mode_index,
                    "anchor_session_index": anchor,
                    "integer_session_offsets": (0, *relative_to_zero),
                    "objective_chi2": mode.objective_chi2,
                    "shape_chi2": mode.shape_chi2,
                    "ambiguity_chi2": mode.ambiguity_chi2,
                    "prior_chi2": mode.prior_chi2,
                    "laplace_log_evidence": mode.laplace_log_evidence,
                    "certification_theta_offset": np.asarray(
                        mode.parameter_values - base, dtype=float
                    ),
                },
            )
        )
    return DelayedIntegerPhaseResult(
        modes=tuple(modes),
        branches=tuple(branches),
        anchor_session_index=anchor,
        relative_session_order=session_order,
        stage_history=tuple(history),
        reached_all_sessions=bool(stage_counts[-1] == session_count - 1),
        wrapped_phase_history=wrapped_phase_history,
    )
