from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, lsq_linear

from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    DualLatticeCellMode,
    DualLatticeCoordinateSystem,
    DualLatticeSupport,
    OrbitalCycleCell,
    PintDualLatticePhaseEvaluator,
    TwoLevelPulseProfile,
    _session_shape_profile,
    profile_fixed_local_pulse_component,
    profile_two_level_pulse_lattice,
)
from apollo_timing.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinIntegerMarginalEvaluation,
    SpinIntegerPathMode,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
    restrict_linearization_to_phase_islands,
    temporal_phase_island_incidence,
)


def dd_binary_delay_seconds(
    barycentric_mjd: Sequence[float],
    *,
    a1_light_seconds: float,
    eccentricity: float,
    t0_mjd: float,
    orbital_period_days: float,
    longitude_periastron_degrees: float,
    longitude_periastron_rate_degrees_per_year: float,
) -> np.ndarray:
    """Evaluate the parameter subset of PINT's DD inverse binary delay.

    The APOLLO scientific contract fixes unlisted DD parameters at their
    input-model values. The Einstein, Shapiro, aberration, and secular-orbit
    terms used by this conditional projector are zero.
    """

    times = np.asarray(barycentric_mjd, dtype=np.longdouble)
    a1 = np.longdouble(a1_light_seconds)
    eccentricity = np.longdouble(eccentricity)
    t0 = np.longdouble(t0_mjd)
    period_days = np.longdouble(orbital_period_days)
    omega0 = np.longdouble(math.radians(longitude_periastron_degrees))
    omdot = np.longdouble(
        math.radians(longitude_periastron_rate_degrees_per_year)
    ) / np.longdouble(365.25 * 86400.0)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
        or not np.isfinite(a1)
        or a1 <= 0.0
        or not np.isfinite(eccentricity)
        or eccentricity <= 0.0
        or eccentricity >= 1.0
        or not np.isfinite(t0)
        or not np.isfinite(period_days)
        or period_days <= 0.0
        or not np.isfinite(omega0)
        or not np.isfinite(omdot)
    ):
        raise ValueError("DD binary-delay inputs lie outside their physical domain.")

    orbital_cycles = (times - t0) / period_days
    mean_anomaly = (
        orbital_cycles - np.floor(orbital_cycles)
    ) * np.longdouble(2.0 * math.pi)
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(32):
        equation = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly
        )
        update = equation / (
            np.longdouble(1.0)
            - eccentricity * np.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= update
        if float(np.max(np.abs(update))) <= 5.0e-15:
            break

    sine_half = np.sin(eccentric_anomaly / np.longdouble(2.0))
    cosine_half = np.cos(eccentric_anomaly / np.longdouble(2.0))
    true_anomaly_local = np.longdouble(2.0) * np.arctan2(
        np.sqrt(np.longdouble(1.0) + eccentricity) * sine_half,
        np.sqrt(np.longdouble(1.0) - eccentricity) * cosine_half,
    )
    true_anomaly_local = np.remainder(
        true_anomaly_local,
        np.longdouble(2.0 * math.pi),
    )
    true_anomaly = (
        np.longdouble(2.0 * math.pi) * orbital_cycles
        + true_anomaly_local
        - mean_anomaly
    )
    period_seconds = period_days * np.longdouble(86400.0)
    periastron_advance = omdot / (
        np.longdouble(2.0 * math.pi) / period_seconds
    )
    omega = omega0 + true_anomaly * periastron_advance

    sine_eccentric = np.sin(eccentric_anomaly)
    cosine_eccentric = np.cos(eccentric_anomaly)
    alpha = a1 * np.sin(omega)
    beta = (
        a1
        * np.sqrt(np.longdouble(1.0) - eccentricity * eccentricity)
        * np.cos(omega)
    )
    roemer_einstein = (
        alpha * (cosine_eccentric - eccentricity)
        + beta * sine_eccentric
    )
    first_derivative = (
        -alpha * sine_eccentric + beta * cosine_eccentric
    )
    second_derivative = (
        -alpha * cosine_eccentric - beta * sine_eccentric
    )
    anomaly_rate = (
        np.longdouble(2.0 * math.pi)
        / period_seconds
        / (np.longdouble(1.0) - eccentricity * cosine_eccentric)
    )
    rate_times_derivative = anomaly_rate * first_derivative
    inverse_delay = roemer_einstein * (
        np.longdouble(1.0)
        - rate_times_derivative
        + rate_times_derivative * rate_times_derivative
        + np.longdouble(0.5)
        * anomaly_rate
        * anomaly_rate
        * roemer_einstein
        * second_derivative
        - np.longdouble(0.5)
        * eccentricity
        * sine_eccentric
        / (np.longdouble(1.0) - eccentricity * cosine_eccentric)
        * anomaly_rate
        * anomaly_rate
        * roemer_einstein
        * first_derivative
    )
    return np.asarray(inverse_delay, dtype=np.longdouble)


@dataclass(frozen=True)
class ConditionalFastProfile:
    slow_normalized_values: np.ndarray
    fast_normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: TwoLevelPulseProfile
    function_evaluations: int
    converged: bool
    status: int
    message: str


@dataclass(frozen=True)
class ConditionalPartitionPathProfile:
    slow_normalized_values: np.ndarray
    fast_normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: TwoLevelPulseProfile
    integer_offsets: tuple[int, ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    function_evaluations: int
    converged: bool
    status: int
    message: str


@dataclass(frozen=True)
class ConditionalSlowCellSearchConfig:
    phi_subdivisions: int = 4
    eccentricity_subdivisions: int = 4
    periastron_rate_subdivisions: int = 4
    fast_projection_evaluations: int = 32
    maximum_function_evaluations: int = 80


@dataclass(frozen=True)
class ConditionalSlowCellMode:
    cell_id: str
    subcell_index: int
    grid_index: tuple[int, int, int]
    normalized_lower: np.ndarray
    normalized_upper: np.ndarray
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    function_evaluations: int
    converged: bool
    status: int
    message: str
    on_subcell_boundary: bool


@dataclass(frozen=True)
class ConditionalSlowPartitionConfig:
    """Controls for a slow-orbit posterior at one temporal partition."""

    fast_shape_max_nfev: int = 32
    fast_finite_difference_step: float = 2.0e-4
    fast_prior_sigma: float = 0.5
    lattice: SpinIntegerMarginalConfig = SpinIntegerMarginalConfig()


@dataclass(frozen=True)
class ConditionalSlowPartitionEvaluation:
    slow_normalized_values: np.ndarray
    fast_shape_normalized_values: np.ndarray
    parameter_values: np.ndarray
    shape_chi2: float
    marginal: SpinIntegerMarginalEvaluation

    @property
    def strict_score(self) -> float:
        if not self.marginal.path_modes:
            return math.inf
        return float(
            self.marginal.shape_quadratic
            + self.marginal.path_modes[0].squared_mahalanobis
        )


@dataclass(frozen=True)
class ConditionalSlowPartitionLikelihood:
    """Marginal likelihood over four slow DD coordinates.

    At every slow point the near-linear block is centered on the exact local
    shape and integrated jointly with all integer constraints internal to the
    requested temporal partition.
    """

    projector: "PintEquivalentDDPhaseProjector"
    reference_fast_normalized_values: np.ndarray
    slow_normalized_lower: np.ndarray
    slow_normalized_upper: np.ndarray
    session_partitions: tuple[tuple[int, ...], ...]
    local_integer_offsets: tuple[tuple[int, ...], ...] | None = None
    config: ConditionalSlowPartitionConfig = ConditionalSlowPartitionConfig()

    @property
    def orbit_lower(self) -> np.ndarray:
        return np.asarray(self.slow_normalized_lower, dtype=float).copy()

    @property
    def orbit_upper(self) -> np.ndarray:
        return np.asarray(self.slow_normalized_upper, dtype=float).copy()

    def _validated_slow(self, values: Sequence[float]) -> np.ndarray:
        slow = np.asarray(values, dtype=float)
        lower = self.orbit_lower
        upper = self.orbit_upper
        if (
            slow.shape != (4,)
            or lower.shape != (4,)
            or upper.shape != (4,)
            or not np.all(np.isfinite(slow))
            or np.any(lower >= upper)
            or np.any(slow < lower)
            or np.any(slow > upper)
        ):
            raise ValueError("Slow partition coordinates lie outside support.")
        return slow

    def evaluate(
        self,
        slow_normalized_values: Sequence[float],
    ) -> ConditionalSlowPartitionEvaluation:
        slow = self._validated_slow(slow_normalized_values)
        conditional = self.projector.profile_fast_shape(
            slow,
            fast_start_normalized_values=self.reference_fast_normalized_values,
            local_integer_offsets=self.local_integer_offsets,
            maximum_function_evaluations=max(
                int(self.config.fast_shape_max_nfev),
                8,
            ),
        )
        linearization = self.projector.linearize_fast(
            conditional,
            finite_difference_step=max(
                float(self.config.fast_finite_difference_step),
                1.0e-7,
            ),
        )
        linearization = restrict_linearization_to_phase_islands(
            linearization,
            self.session_partitions,
        )
        sigma = max(float(self.config.fast_prior_sigma), 1.0e-6)
        marginal = marginalize_spin_and_integer_paths(
            linearization,
            spin_prior_mean=np.zeros(4, dtype=float),
            spin_prior_covariance=np.diag(np.full(4, sigma * sigma)),
            config=self.config.lattice,
        )
        return ConditionalSlowPartitionEvaluation(
            slow_normalized_values=slow.copy(),
            fast_shape_normalized_values=np.asarray(
                conditional.fast_normalized_values,
                dtype=float,
            ).copy(),
            parameter_values=np.asarray(conditional.parameter_values, dtype=float).copy(),
            shape_chi2=float(conditional.profile.local_chi2),
            marginal=marginal,
        )

    def __call__(self, slow_normalized_values: Sequence[float]) -> float:
        try:
            value = self.evaluate(slow_normalized_values).marginal.log_likelihood
        except (ArithmeticError, RuntimeError, ValueError):
            return -math.inf
        return float(value) if np.isfinite(value) else -math.inf

    @staticmethod
    def full_normalized_values(
        evaluation: ConditionalSlowPartitionEvaluation,
        path_mode: SpinIntegerPathMode,
    ) -> np.ndarray:
        fast = evaluation.fast_shape_normalized_values + np.asarray(
            path_mode.spin_posterior_mean,
            dtype=float,
        )
        return np.concatenate((evaluation.slow_normalized_values, fast))


@dataclass(frozen=True)
class ConditionalJointIntegerScore:
    """Zero-temperature joint score after eliminating the fast block.

    The slow orbital coordinates stay fixed. The four fast coordinates are
    integrated in the local Gaussian model and the complete session-integer
    lattice is solved before any path is selected for nonlinear closure.
    """

    slow_normalized_values: np.ndarray
    fast_reference_normalized_values: np.ndarray
    fast_posterior_delta: np.ndarray
    fast_posterior_normalized_values: np.ndarray
    shape_chi2: float
    shape_quadratic: float
    squared_integer_mahalanobis: float
    strict_score: float
    integer_offsets: tuple[int, ...]
    posterior_in_fast_support: bool
    lattice_visited_nodes: int


@dataclass(frozen=True)
class FullWrappedPosteriorScore:
    """Local chart of the joint continuous and complete integer posterior.

    All eight DD coordinates and the complete session-integer lattice enter one
    Gaussian chart, with no preceding shape-only fast optimization. A portfolio
    of these charts can schedule nonlinear closure by wrapped posterior mass
    instead of local shape chi-square.
    """

    reference_normalized_values: np.ndarray
    posterior_delta: np.ndarray
    posterior_normalized_values: np.ndarray
    log_likelihood: float
    shape_quadratic: float
    squared_integer_mahalanobis: float
    strict_score: float
    integer_offsets: tuple[int, ...]
    posterior_in_support: bool
    retained_lattice_candidates: int
    lattice_mass_converged: bool
    lattice_visited_nodes: int


def score_full_wrapped_posterior_mode(
    *,
    projector: PintEquivalentDDPhaseProjector,
    normalized_values: Sequence[float],
    normalized_lower: Sequence[float],
    normalized_upper: Sequence[float],
    local_integer_offsets: Sequence[Sequence[int]] | None = None,
    coordinate_prior_sigma: float | Sequence[float] = 0.25,
    finite_difference_step: float = 2.0e-4,
    lattice_config: SpinIntegerMarginalConfig | None = None,
) -> FullWrappedPosteriorScore:
    """Score one full DD chart after marginalizing the integer lattice.

    The returned posterior displacement is not clipped. When
    ``posterior_in_support`` is false, the caller must reject the displacement
    or open a neighboring protected chart. Clipping would create a false
    posterior mode at the chart boundary.
    """

    reference = np.asarray(normalized_values, dtype=float)
    lower = np.asarray(normalized_lower, dtype=float)
    upper = np.asarray(normalized_upper, dtype=float)
    if (
        reference.shape != (8,)
        or lower.shape != (8,)
        or upper.shape != (8,)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(reference < lower)
        or np.any(reference > upper)
    ):
        raise ValueError("A full wrapped chart needs valid bounded DD coordinates.")

    sigma = np.asarray(coordinate_prior_sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(8, float(sigma), dtype=float)
    if sigma.shape != (8,) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Coordinate prior sigma must contain eight positive values.")

    linearization = projector.linearize_full(
        reference,
        normalized_lower=lower,
        normalized_upper=upper,
        local_integer_offsets=local_integer_offsets,
        finite_difference_step=max(float(finite_difference_step), 1.0e-7),
    )
    marginal = marginalize_spin_and_integer_paths(
        linearization,
        spin_prior_mean=np.zeros(8, dtype=float),
        spin_prior_covariance=np.diag(np.square(sigma)),
        config=lattice_config or SpinIntegerMarginalConfig(),
    )
    if not marginal.path_modes:
        raise RuntimeError("The full wrapped posterior returned no integer path.")
    path = marginal.path_modes[0]
    delta = np.asarray(path.spin_posterior_mean, dtype=float)
    posterior = reference + delta
    shape_quadratic = float(marginal.shape_quadratic)
    integer_quadratic = float(path.squared_mahalanobis)
    return FullWrappedPosteriorScore(
        reference_normalized_values=reference.copy(),
        posterior_delta=delta,
        posterior_normalized_values=posterior,
        log_likelihood=float(marginal.log_likelihood),
        shape_quadratic=shape_quadratic,
        squared_integer_mahalanobis=integer_quadratic,
        strict_score=shape_quadratic + integer_quadratic,
        integer_offsets=tuple(int(value) for value in path.integer_offsets),
        posterior_in_support=bool(
            np.all(posterior >= lower) and np.all(posterior <= upper)
        ),
        retained_lattice_candidates=int(marginal.retained_lattice_candidates),
        lattice_mass_converged=bool(marginal.lattice_mass_converged),
        lattice_visited_nodes=int(marginal.lattice_visited_nodes),
    )


def score_conditional_joint_integer_mode(
    *,
    projector: PintEquivalentDDPhaseProjector,
    slow_normalized_values: Sequence[float],
    fast_normalized_values: Sequence[float],
    local_integer_offsets: Sequence[Sequence[int]] | None = None,
    session_partitions: Sequence[Sequence[int]] | None = None,
    fast_prior_sigma: float | Sequence[float] = 0.5,
    finite_difference_step: float = 2.0e-4,
    maximum_lattice_nodes: int = 2_000_000,
) -> ConditionalJointIntegerScore:
    """Score one slow state using the nearest complete integer path.

    This is the zero-temperature limit of the APOLLO fast-continuous and
    session-integer posterior. It is intended for global mode scheduling and
    must not be treated as an exact nonlinear PINT chi-square.
    """

    slow = np.asarray(slow_normalized_values, dtype=float)
    fast = np.asarray(fast_normalized_values, dtype=float)
    if slow.shape != (4,) or fast.shape != (4,):
        raise ValueError("A conditional joint score needs four slow and four fast values.")
    if not np.all(np.isfinite(slow)) or not np.all(np.isfinite(fast)):
        raise ValueError("Conditional joint coordinates must be finite.")
    if np.any(slow < -1.0) or np.any(slow > 1.0):
        raise ValueError("Conditional slow coordinates lie outside support.")
    if np.any(fast < -1.0) or np.any(fast > 1.0):
        raise ValueError("Conditional fast coordinates lie outside support.")

    sigma = np.asarray(fast_prior_sigma, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(4, float(sigma), dtype=float)
    if sigma.shape != (4,) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("Fast prior sigma must contain four positive finite values.")

    physical = projector._physical_from_normalized(slow, fast)
    profile = projector.profile_physical(
        physical,
        local_integer_offsets=local_integer_offsets,
    )
    conditional = ConditionalFastProfile(
        slow_normalized_values=slow.copy(),
        fast_normalized_values=fast.copy(),
        parameter_values=physical,
        profile=profile,
        function_evaluations=0,
        converged=True,
        status=0,
        message="fixed joint-score reference",
    )
    linearization = projector.linearize_fast(
        conditional,
        finite_difference_step=finite_difference_step,
    )
    if session_partitions is not None:
        linearization = restrict_linearization_to_phase_islands(
            linearization,
            session_partitions,
        )
    marginal = marginalize_spin_and_integer_paths(
        linearization,
        spin_prior_mean=np.zeros(4, dtype=float),
        spin_prior_covariance=np.diag(np.square(sigma)),
        config=SpinIntegerMarginalConfig(
            initial_lattice_candidates=1,
            maximum_lattice_candidates=1,
            lattice_log_mass_tolerance=0.0,
            maximum_lattice_nodes=max(int(maximum_lattice_nodes), 1),
        ),
    )
    if not marginal.path_modes:
        raise RuntimeError("Conditional integer lattice returned no path.")
    path = marginal.path_modes[0]
    posterior_delta = np.asarray(path.spin_posterior_mean, dtype=float)
    posterior = fast + posterior_delta
    shape_quadratic = float(marginal.shape_quadratic)
    integer_mahalanobis = float(path.squared_mahalanobis)
    return ConditionalJointIntegerScore(
        slow_normalized_values=slow.copy(),
        fast_reference_normalized_values=fast.copy(),
        fast_posterior_delta=posterior_delta,
        fast_posterior_normalized_values=posterior,
        shape_chi2=float(profile.local_chi2),
        shape_quadratic=shape_quadratic,
        squared_integer_mahalanobis=integer_mahalanobis,
        strict_score=shape_quadratic + integer_mahalanobis,
        integer_offsets=tuple(int(value) for value in path.integer_offsets),
        posterior_in_fast_support=bool(
            np.all(posterior >= -1.0) and np.all(posterior <= 1.0)
        ),
        lattice_visited_nodes=int(marginal.lattice_visited_nodes),
    )


def _bounded_conditional_mode(
    *,
    projector: PintEquivalentDDPhaseProjector,
    cell_id: str,
    subcell_index: int,
    grid_index: tuple[int, ...],
    normalized_lower: Sequence[float],
    normalized_upper: Sequence[float],
    local_integer_offsets: Sequence[Sequence[int]] | None,
    fast_projection_evaluations: int,
    maximum_function_evaluations: int,
    normalized_start: Sequence[float] | None = None,
) -> ConditionalSlowCellMode:
    """Optimize one protected slow chart while conditionally eliminating fast terms."""

    lower = np.asarray(normalized_lower, dtype=float)
    upper = np.asarray(normalized_upper, dtype=float)
    if (
        lower.shape != (8,)
        or upper.shape != (8,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(lower < -1.0)
        or np.any(upper > 1.0)
    ):
        raise ValueError("A protected conditional chart needs valid normalized bounds.")
    component = (
        None
        if local_integer_offsets is None
        else tuple(
            tuple(int(value) for value in offsets)
            for offsets in local_integer_offsets
        )
    )
    residual_count = len(projector.evaluator.toas.table)
    failure = np.full(residual_count, 1.0e8, dtype=float)
    supplied_start = (
        None
        if normalized_start is None
        else np.asarray(normalized_start, dtype=float)
    )
    if supplied_start is not None and (
        supplied_start.shape != (8,)
        or not np.all(np.isfinite(supplied_start))
    ):
        raise ValueError("A protected conditional start must be a finite 8-vector.")
    if supplied_start is not None and (
        np.any(supplied_start < lower) or np.any(supplied_start > upper)
    ):
        raise ValueError("A protected conditional start lies outside chart bounds.")
    slow_start = (
        0.5 * (lower[:4] + upper[:4])
        if supplied_start is None
        else supplied_start[:4].copy()
    )
    try:
        conditional = projector.profile_fast_shape(
            slow_start,
            fast_start_normalized_values=(
                None if supplied_start is None else supplied_start[4:]
            ),
            local_integer_offsets=component,
            maximum_function_evaluations=max(
                int(fast_projection_evaluations),
                8,
            ),
        )
        start = np.concatenate(
            (slow_start, conditional.fast_normalized_values)
        )
    except (ArithmeticError, RuntimeError, ValueError):
        start = 0.5 * (lower + upper)
        start[4:] = projector.support.normalize(
            projector.support.slow_center,
            projector.support.fast_center,
        )[4:]
    start = np.clip(start, lower, upper)

    def residuals(normalized: np.ndarray) -> np.ndarray:
        try:
            physical = projector._physical_from_normalized(
                normalized[:4],
                normalized[4:],
            )
            profile = projector.profile_physical(
                physical,
                local_integer_offsets=component,
            )
            values = np.asarray(profile.normalized_local_residuals, dtype=float)
            return values if values.shape == failure.shape else failure
        except (ArithmeticError, RuntimeError, ValueError):
            return failure

    result = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=max(int(maximum_function_evaluations), 8),
    )
    normalized = np.asarray(result.x, dtype=float)
    try:
        physical = projector._physical_from_normalized(
            normalized[:4],
            normalized[4:],
        )
        profile = projector.profile_physical(
            physical,
            local_integer_offsets=component,
        )
        shape_chi2 = float(profile.local_chi2)
        fitted_component = profile.local_integer_offsets
    except (ArithmeticError, RuntimeError, ValueError):
        physical = np.full(8, math.nan, dtype=float)
        shape_chi2 = math.inf
        fitted_component = tuple()
    slow_boundary_distance = np.minimum(
        normalized[:4] - lower[:4],
        upper[:4] - normalized[:4],
    )
    slow_width = upper[:4] - lower[:4]
    boundary_tolerance = np.maximum(1.0e-7 * slow_width, 1.0e-10)
    return ConditionalSlowCellMode(
        cell_id=str(cell_id),
        subcell_index=int(subcell_index),
        grid_index=tuple(int(value) for value in grid_index),
        normalized_lower=lower.copy(),
        normalized_upper=upper.copy(),
        normalized_values=normalized,
        parameter_values=physical,
        local_integer_offsets=fitted_component,
        shape_chi2=shape_chi2,
        function_evaluations=int(result.nfev),
        converged=bool(result.success),
        status=int(result.status),
        message=str(result.message),
        on_subcell_boundary=bool(
            np.any(slow_boundary_distance <= boundary_tolerance)
        ),
    )


def fit_conditional_slow_chart(
    *,
    projector: PintEquivalentDDPhaseProjector,
    cell_id: str,
    chart_index: int,
    grid_index: Sequence[int],
    normalized_lower: Sequence[float],
    normalized_upper: Sequence[float],
    normalized_start: Sequence[float] | None = None,
    local_integer_offsets: Sequence[Sequence[int]] | None = None,
    fast_projection_evaluations: int = 32,
    maximum_function_evaluations: int = 100,
) -> ConditionalSlowCellMode:
    """Fit one independently protected chart from an explicit prior point.

    Global island discovery uses this fit with fixed chart bounds. Each prior
    stratum can then be checkpointed and audited independently, without
    inheriting a previously selected local mode.
    """

    return _bounded_conditional_mode(
        projector=projector,
        cell_id=str(cell_id),
        subcell_index=int(chart_index),
        grid_index=tuple(int(value) for value in grid_index),
        normalized_lower=normalized_lower,
        normalized_upper=normalized_upper,
        normalized_start=normalized_start,
        local_integer_offsets=local_integer_offsets,
        fast_projection_evaluations=max(int(fast_projection_evaluations), 8),
        maximum_function_evaluations=max(int(maximum_function_evaluations), 8),
    )


def refine_conditional_slow_mode(
    *,
    projector: PintEquivalentDDPhaseProjector,
    mode: ConditionalSlowCellMode,
    subdivisions: int | Sequence[int] = 2,
    fast_projection_evaluations: int = 32,
    maximum_function_evaluations: int = 100,
) -> tuple[ConditionalSlowCellMode, ...]:
    """Evaluate every child in a four-dimensional protected slow chart.

    Return all children so callers can checkpoint unresolved siblings before
    scheduling a deeper level. No top-K filter is applied. The fast block is
    conditionally projected inside each child.
    """

    if isinstance(subdivisions, int):
        counts = (max(int(subdivisions), 1),) * 4
    else:
        counts = tuple(max(int(value), 1) for value in subdivisions)
        if len(counts) != 4:
            raise ValueError("Slow refinement needs four subdivision counts.")
    parent_lower = np.asarray(mode.normalized_lower, dtype=float)
    parent_upper = np.asarray(mode.normalized_upper, dtype=float)
    if (
        parent_lower.shape != (8,)
        or parent_upper.shape != (8,)
        or np.any(parent_lower >= parent_upper)
    ):
        raise ValueError("The parent conditional mode has invalid chart bounds.")
    edges = tuple(
        np.linspace(parent_lower[index], parent_upper[index], count + 1)
        for index, count in enumerate(counts)
    )
    epsilon = 32.0 * np.finfo(float).eps
    children: list[ConditionalSlowCellMode] = []
    for child_index, grid_index in enumerate(
        np.ndindex(*counts)
    ):
        lower = parent_lower.copy()
        upper = parent_upper.copy()
        for axis, edge_index in enumerate(grid_index):
            lower[axis] = edges[axis][edge_index]
            upper[axis] = edges[axis][edge_index + 1]
        lower[:4] += epsilon
        upper[:4] -= epsilon
        children.append(
            _bounded_conditional_mode(
                projector=projector,
                cell_id=mode.cell_id,
                subcell_index=child_index,
                grid_index=tuple(int(value) for value in grid_index),
                normalized_lower=lower,
                normalized_upper=upper,
                local_integer_offsets=mode.local_integer_offsets,
                fast_projection_evaluations=fast_projection_evaluations,
                maximum_function_evaluations=maximum_function_evaluations,
            )
        )
    children.sort(key=lambda item: (item.shape_chi2, item.subcell_index))
    return tuple(children)


def conditional_mode_to_cell_mode(
    *,
    projector: PintEquivalentDDPhaseProjector,
    mode: ConditionalSlowCellMode,
) -> DualLatticeCellMode:
    """Adapt one refined conditional mode to the coherent lattice interface."""

    slow, fast = projector.support.denormalize(mode.normalized_values)
    return DualLatticeCellMode(
        cell_id=mode.cell_id,
        start_index=mode.subcell_index,
        normalized_values=np.asarray(mode.normalized_values, dtype=float).copy(),
        slow_values=slow,
        fast_values=fast,
        parameter_values=np.asarray(mode.parameter_values, dtype=float).copy(),
        local_integer_offsets=mode.local_integer_offsets,
        shape_chi2=float(mode.shape_chi2),
        function_evaluations=int(mode.function_evaluations),
        converged=bool(mode.converged),
        status=int(mode.status),
        message=str(mode.message),
        on_cell_boundary=bool(mode.on_subcell_boundary),
        normalized_lower=np.asarray(mode.normalized_lower, dtype=float).copy(),
        normalized_upper=np.asarray(mode.normalized_upper, dtype=float).copy(),
    )


class PintEquivalentDDPhaseProjector:
    """Fast exact-DD phase statistics for fixed-slow dual-lattice cells.

    Solar-system, clock, astrometric, and dispersion delays are cached from
    PINT. Only the eight APOLLO fit parameters are reevaluated. Absolute TZR
    phase is omitted: session projection and the relative-anchor lattice both
    remove that single global gauge.
    """

    def __init__(
        self,
        *,
        evaluator: PintDualLatticePhaseEvaluator,
        coordinates: DualLatticeCoordinateSystem,
        support: DualLatticeSupport,
    ) -> None:
        from astropy import units as u

        self.evaluator = evaluator
        self.coordinates = coordinates
        self.support = support
        model = evaluator.model
        toas = evaluator.toas
        binary_components = [
            component
            for name, component in model.components.items()
            if str(name).startswith("Binary")
        ]
        if len(binary_components) != 1:
            raise ValueError("The DD projector requires exactly one binary component.")
        binary = binary_components[0]
        model.delay(toas)
        binary.binarymodel_delay(toas)
        barycentric = np.asarray(
            binary.binary_instance.t.to_value(u.day),
            dtype=np.longdouble,
        )
        tdb = np.asarray(toas.table["tdbld"], dtype=np.longdouble)
        if barycentric.shape != tdb.shape:
            raise RuntimeError("PINT binary barycentric epochs do not align with TOAs.")
        self.barycentric_mjd = barycentric
        self.fixed_nonbinary_delay_seconds = (
            tdb - barycentric
        ) * np.longdouble(86400.0)
        self.spin_epoch_mjd = np.longdouble(model["PEPOCH"].value)
        self.tdb_mjd = tdb
        self.phase_sigmas = np.asarray(evaluator._phase_sigmas(), dtype=float)
        self.base_pulse_numbers = np.asarray(
            evaluator.base_pulse_numbers,
            dtype=np.longdouble,
        )
        self.sessions = tuple(evaluator.sessions)
        self._zero_local_component = tuple(
            tuple(0 for _ in session.toa_indices) for session in self.sessions
        )

    def raw_phase(self, parameter_values: Sequence[float]) -> np.ndarray:
        values = np.asarray(parameter_values, dtype=float)
        if values.shape != (8,) or not np.all(np.isfinite(values)):
            raise ValueError("The DD phase projector requires eight finite parameters.")
        by_name = dict(
            zip(self.evaluator.parameter_names, values, strict=True)
        )
        delay = dd_binary_delay_seconds(
            self.barycentric_mjd,
            a1_light_seconds=float(by_name["A1"]),
            eccentricity=float(by_name["ECC"]),
            t0_mjd=float(by_name["T0"]),
            orbital_period_days=float(by_name["PB"]),
            longitude_periastron_degrees=float(by_name["OM"]),
            longitude_periastron_rate_degrees_per_year=float(by_name["OMDOT"]),
        )
        elapsed = (
            (self.tdb_mjd - self.spin_epoch_mjd) * np.longdouble(86400.0)
            - self.fixed_nonbinary_delay_seconds
            - delay
        )
        f0 = np.longdouble(by_name["F0"])
        f1 = np.longdouble(by_name["F1"])
        return f0 * elapsed + np.longdouble(0.5) * f1 * elapsed * elapsed

    def profile_physical(
        self,
        parameter_values: Sequence[float],
        *,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
        local_integer_radius: int = 0,
    ) -> TwoLevelPulseProfile:
        residuals = np.asarray(
            self.raw_phase(parameter_values) - self.base_pulse_numbers,
            dtype=float,
        )
        if local_integer_offsets is None and int(local_integer_radius) > 0:
            return profile_two_level_pulse_lattice(
                residuals,
                self.phase_sigmas,
                self.sessions,
                integer_radius=max(int(local_integer_radius), 0),
            )
        component = (
            self._zero_local_component
            if local_integer_offsets is None
            else tuple(
                tuple(int(value) for value in session)
                for session in local_integer_offsets
            )
        )
        return profile_fixed_local_pulse_component(
            residuals,
            self.phase_sigmas,
            self.sessions,
            component,
        )

    def _physical_from_normalized(
        self,
        slow_normalized: np.ndarray,
        fast_normalized: np.ndarray,
    ) -> np.ndarray:
        full = np.concatenate((slow_normalized, fast_normalized))
        slow, fast = self.support.denormalize(full)
        physical = self.coordinates.combine_physical(slow, fast)
        if not self.support.contains_physical(physical):
            raise ValueError("A conditional DD point lies outside physical support.")
        return physical

    def profile_normalized(
        self,
        normalized_values: Sequence[float],
        *,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
    ) -> tuple[np.ndarray, TwoLevelPulseProfile]:
        """Evaluate one complete normalized DD coordinate vector."""

        normalized = np.asarray(normalized_values, dtype=float)
        if normalized.shape != (8,) or not np.all(np.isfinite(normalized)):
            raise ValueError("A normalized DD state must be a finite 8-vector.")
        if np.any(normalized < -1.0) or np.any(normalized > 1.0):
            raise ValueError("A normalized DD state lies outside support.")
        physical = self._physical_from_normalized(
            normalized[:4],
            normalized[4:],
        )
        return physical, self.profile_physical(
            physical,
            local_integer_offsets=local_integer_offsets,
        )

    def linearize_full(
        self,
        normalized_values: Sequence[float],
        *,
        normalized_lower: Sequence[float] | None = None,
        normalized_upper: Sequence[float] | None = None,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
        finite_difference_step: float = 2.0e-4,
    ) -> SpinOrbitLinearization:
        """Linearize shape and session anchors in all eight DD coordinates."""

        reference = np.asarray(normalized_values, dtype=float)
        lower = (
            np.full(8, -1.0, dtype=float)
            if normalized_lower is None
            else np.asarray(normalized_lower, dtype=float)
        )
        upper = (
            np.full(8, 1.0, dtype=float)
            if normalized_upper is None
            else np.asarray(normalized_upper, dtype=float)
        )
        if (
            reference.shape != (8,)
            or lower.shape != (8,)
            or upper.shape != (8,)
            or not np.all(np.isfinite(reference))
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower >= upper)
            or np.any(reference < lower)
            or np.any(reference > upper)
        ):
            raise ValueError("Full DD linearization needs valid normalized bounds.")
        _, profile = self.profile_normalized(
            reference,
            local_integer_offsets=local_integer_offsets,
        )
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        offsets = np.asarray(profile.session_intercepts, dtype=float)
        shape_jacobian = np.empty((shape.size, 8), dtype=float)
        offset_jacobian = np.empty((offsets.size, 8), dtype=float)
        requested_step = max(float(finite_difference_step), 1.0e-7)

        def valid_probe(
            index: int,
            direction: float,
        ) -> tuple[float, TwoLevelPulseProfile] | None:
            available = (
                float(upper[index] - reference[index])
                if direction > 0.0
                else float(reference[index] - lower[index])
            )
            probe_step = min(requested_step, available)
            minimum_step = max(
                64.0
                * np.finfo(float).eps
                * max(abs(float(reference[index])), 1.0),
                requested_step * 2.0**-20,
            )
            while probe_step >= minimum_step:
                probe = reference.copy()
                probe[index] += direction * probe_step
                try:
                    _, probe_profile = self.profile_normalized(
                        probe,
                        local_integer_offsets=local_integer_offsets,
                    )
                    return direction * probe_step, probe_profile
                except (ArithmeticError, RuntimeError, ValueError):
                    probe_step *= 0.5
            return None

        for column in range(8):
            plus = valid_probe(column, 1.0)
            minus = valid_probe(column, -1.0)
            if plus is not None and minus is not None:
                plus_step, plus_profile = plus
                minus_step, minus_profile = minus
                denominator = plus_step - minus_step
                shape_delta = (
                    np.asarray(plus_profile.normalized_local_residuals, dtype=float)
                    - np.asarray(minus_profile.normalized_local_residuals, dtype=float)
                )
                offset_delta = (
                    np.asarray(plus_profile.session_intercepts, dtype=float)
                    - np.asarray(minus_profile.session_intercepts, dtype=float)
                )
            elif plus is not None:
                plus_step, plus_profile = plus
                denominator = plus_step
                shape_delta = (
                    np.asarray(plus_profile.normalized_local_residuals, dtype=float)
                    - shape
                )
                offset_delta = (
                    np.asarray(plus_profile.session_intercepts, dtype=float)
                    - offsets
                )
            elif minus is not None:
                minus_step, minus_profile = minus
                denominator = -minus_step
                shape_delta = (
                    shape
                    - np.asarray(minus_profile.normalized_local_residuals, dtype=float)
                )
                offset_delta = (
                    offsets
                    - np.asarray(minus_profile.session_intercepts, dtype=float)
                )
            else:
                raise ValueError(
                    f"No physical full-coordinate probe exists for column {column}."
                )
            shape_jacobian[:, column] = shape_delta / denominator
            offset_jacobian[:, column] = offset_delta / denominator

        relative = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        variances = np.asarray(profile.session_intercept_variances, dtype=float)
        relative_covariance = np.diag(variances[1:]) + float(variances[0])
        return SpinOrbitLinearization(
            profile=_session_shape_profile(profile),
            spin_reference=reference.copy(),
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
        )

    def profile_fast_shape(
        self,
        slow_normalized_values: Sequence[float],
        *,
        fast_start_normalized_values: Sequence[float] | None = None,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
        maximum_function_evaluations: int = 80,
    ) -> ConditionalFastProfile:
        slow = np.asarray(slow_normalized_values, dtype=float)
        if slow.shape != (4,) or not np.all(np.isfinite(slow)):
            raise ValueError("Conditional slow coordinates must be a finite 4-vector.")
        if np.any(slow < -1.0) or np.any(slow > 1.0):
            raise ValueError("Conditional slow coordinates lie outside support.")
        start = (
            self.support.normalize(
                self.support.slow_center,
                self.support.fast_center,
            )[4:]
            if fast_start_normalized_values is None
            else np.asarray(fast_start_normalized_values, dtype=float)
        )
        if start.shape != (4,) or not np.all(np.isfinite(start)):
            raise ValueError("Conditional fast start must be a finite 4-vector.")
        start = np.clip(start, -1.0, 1.0)
        residual_count = len(self.evaluator.toas.table)
        failure = np.full(residual_count, 1.0e8, dtype=float)

        def residuals(fast: np.ndarray) -> np.ndarray:
            try:
                physical = self._physical_from_normalized(slow, fast)
                profile = self.profile_physical(
                    physical,
                    local_integer_offsets=local_integer_offsets,
                )
                value = np.asarray(profile.normalized_local_residuals, dtype=float)
                return value if value.shape == failure.shape else failure
            except (ArithmeticError, RuntimeError, ValueError):
                return failure

        # The four fast coordinates are nearly linear at fixed slow orbit.
        # Eliminate their large dynamic range before nonlinear refinement;
        # otherwise trust-region scaling can stop on a high-shape plateau.
        for _ in range(2):
            center_residuals = residuals(start)
            if np.array_equal(center_residuals, failure):
                break
            jacobian = np.empty((residual_count, 4), dtype=float)
            requested_step = 1.0e-3
            valid_linearization = True
            for column in range(4):
                plus = start.copy()
                minus = start.copy()
                plus[column] = min(plus[column] + requested_step, 1.0)
                minus[column] = max(minus[column] - requested_step, -1.0)
                plus_residuals = residuals(plus)
                minus_residuals = residuals(minus)
                plus_valid = not np.array_equal(plus_residuals, failure)
                minus_valid = not np.array_equal(minus_residuals, failure)
                if plus_valid and minus_valid:
                    denominator = float(plus[column] - minus[column])
                    jacobian[:, column] = (
                        plus_residuals - minus_residuals
                    ) / denominator
                elif plus_valid and plus[column] > start[column]:
                    jacobian[:, column] = (
                        plus_residuals - center_residuals
                    ) / float(plus[column] - start[column])
                elif minus_valid and minus[column] < start[column]:
                    jacobian[:, column] = (
                        center_residuals - minus_residuals
                    ) / float(start[column] - minus[column])
                else:
                    valid_linearization = False
                    break
            if not valid_linearization:
                break
            projected = lsq_linear(
                jacobian,
                -center_residuals,
                bounds=(-1.0 - start, 1.0 - start),
                lsmr_tol="auto",
            )
            candidate = np.clip(start + projected.x, -1.0, 1.0)
            candidate_residuals = residuals(candidate)
            if (
                np.array_equal(candidate_residuals, failure)
                or float(candidate_residuals @ candidate_residuals)
                >= float(center_residuals @ center_residuals)
            ):
                break
            start = candidate

        result = least_squares(
            residuals,
            start,
            bounds=(np.full(4, -1.0), np.full(4, 1.0)),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=max(int(maximum_function_evaluations), 8),
        )
        fast = np.asarray(result.x, dtype=float)
        physical = self._physical_from_normalized(slow, fast)
        profile = self.profile_physical(
            physical,
            local_integer_offsets=local_integer_offsets,
        )
        return ConditionalFastProfile(
            slow_normalized_values=slow.copy(),
            fast_normalized_values=fast,
            parameter_values=physical,
            profile=profile,
            function_evaluations=int(result.nfev),
            converged=bool(result.success),
            status=int(result.status),
            message=str(result.message),
        )

    def profile_fast_partition_path(
        self,
        slow_normalized_values: Sequence[float],
        *,
        session_partitions: Sequence[Sequence[int]],
        integer_offsets: Sequence[int],
        fast_start_normalized_values: Sequence[float] | None = None,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
        maximum_function_evaluations: int = 80,
    ) -> ConditionalPartitionPathProfile:
        """Exactly eliminate the fast block for one partition integer path."""

        slow = np.asarray(slow_normalized_values, dtype=float)
        if slow.shape != (4,) or not np.all(np.isfinite(slow)):
            raise ValueError("Conditional slow coordinates must be a finite 4-vector.")
        if np.any(slow < -1.0) or np.any(slow > 1.0):
            raise ValueError("Conditional slow coordinates lie outside support.")
        start = (
            self.support.normalize(
                self.support.slow_center,
                self.support.fast_center,
            )[4:]
            if fast_start_normalized_values is None
            else np.asarray(fast_start_normalized_values, dtype=float)
        )
        if start.shape != (4,) or not np.all(np.isfinite(start)):
            raise ValueError("Conditional fast start must be a finite 4-vector.")
        start = np.clip(start, -1.0, 1.0)
        incidence = temporal_phase_island_incidence(
            len(self.evaluator.sessions),
            session_partitions,
        )
        path = np.asarray(integer_offsets, dtype=float)
        if path.shape != (incidence.shape[0],) or not np.all(np.isfinite(path)):
            raise ValueError("A partition path must match its internal phase dimension.")
        residual_count = len(self.evaluator.toas.table) + path.size
        failure = np.full(residual_count, 1.0e8, dtype=float)

        def residuals(fast: np.ndarray) -> np.ndarray:
            try:
                _, profile = self.profile_normalized(
                    np.concatenate((slow, fast)),
                    local_integer_offsets=local_integer_offsets,
                )
                relative = (
                    profile.session_intercepts[1:]
                    - profile.session_intercepts[0]
                )
                covariance = np.diag(profile.session_intercept_variances[1:])
                covariance += float(profile.session_intercept_variances[0])
                partition_covariance = incidence @ covariance @ incidence.T
                phase = np.linalg.solve(
                    np.linalg.cholesky(partition_covariance),
                    incidence @ relative - path,
                )
                return np.concatenate((profile.normalized_local_residuals, phase))
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                return failure

        fit = least_squares(
            residuals,
            start,
            bounds=(np.full(4, -1.0), np.full(4, 1.0)),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
            max_nfev=max(int(maximum_function_evaluations), 8),
        )
        fast = np.asarray(fit.x, dtype=float)
        physical, profile = self.profile_normalized(
            np.concatenate((slow, fast)),
            local_integer_offsets=local_integer_offsets,
        )
        exact = residuals(fast)
        if np.array_equal(exact, failure):
            raise RuntimeError("The fast partition profile ended outside support.")
        shape_chi2 = float(profile.local_chi2)
        phase_chi2 = float(exact[-path.size :] @ exact[-path.size :])
        return ConditionalPartitionPathProfile(
            slow_normalized_values=slow.copy(),
            fast_normalized_values=fast,
            parameter_values=physical,
            profile=profile,
            integer_offsets=tuple(int(value) for value in path),
            shape_chi2=shape_chi2,
            phase_chi2=phase_chi2,
            total_chi2=shape_chi2 + phase_chi2,
            function_evaluations=int(fit.nfev),
            converged=bool(fit.success),
            status=int(fit.status),
            message=str(fit.message),
        )

    def linearize_fast(
        self,
        conditional: ConditionalFastProfile,
        *,
        finite_difference_step: float = 2.0e-4,
    ) -> SpinOrbitLinearization:
        slow = np.asarray(conditional.slow_normalized_values, dtype=float)
        fast = np.asarray(conditional.fast_normalized_values, dtype=float)
        profile = conditional.profile
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        offsets = np.asarray(profile.session_intercepts, dtype=float)
        shape_jacobian = np.empty((shape.size, 4), dtype=float)
        offset_jacobian = np.empty((offsets.size, 4), dtype=float)
        step = max(float(finite_difference_step), 1.0e-7)

        def valid_probe(
            index: int,
            direction: float,
        ) -> tuple[float, TwoLevelPulseProfile] | None:
            available = (
                1.0 - float(fast[index])
                if direction > 0.0
                else float(fast[index]) + 1.0
            )
            probe_step = min(step, available)
            minimum_step = max(
                64.0
                * np.finfo(float).eps
                * max(abs(float(fast[index])), 1.0),
                step * 2.0**-20,
            )
            while probe_step >= minimum_step:
                probe = fast.copy()
                probe[index] += direction * probe_step
                try:
                    physical = self._physical_from_normalized(slow, probe)
                    probe_profile = self.profile_physical(
                        physical,
                        local_integer_offsets=profile.local_integer_offsets,
                    )
                    return direction * probe_step, probe_profile
                except (ArithmeticError, RuntimeError, ValueError):
                    probe_step *= 0.5
            return None

        for column in range(4):
            plus_probe = valid_probe(column, 1.0)
            minus_probe = valid_probe(column, -1.0)
            if plus_probe is not None and minus_probe is not None:
                plus_step, plus_profile = plus_probe
                minus_step, minus_profile = minus_probe
                denominator = plus_step - minus_step
                shape_delta = (
                    np.asarray(
                        plus_profile.normalized_local_residuals,
                        dtype=float,
                    )
                    - np.asarray(
                        minus_profile.normalized_local_residuals,
                        dtype=float,
                    )
                )
                offset_delta = (
                    np.asarray(plus_profile.session_intercepts, dtype=float)
                    - np.asarray(minus_profile.session_intercepts, dtype=float)
                )
            elif plus_probe is not None:
                plus_step, plus_profile = plus_probe
                denominator = plus_step
                shape_delta = (
                    np.asarray(
                        plus_profile.normalized_local_residuals,
                        dtype=float,
                    )
                    - shape
                )
                offset_delta = (
                    np.asarray(plus_profile.session_intercepts, dtype=float)
                    - offsets
                )
            elif minus_probe is not None:
                minus_step, minus_profile = minus_probe
                denominator = -minus_step
                shape_delta = (
                    shape
                    - np.asarray(
                        minus_profile.normalized_local_residuals,
                        dtype=float,
                    )
                )
                offset_delta = (
                    offsets
                    - np.asarray(minus_profile.session_intercepts, dtype=float)
                )
            else:
                raise ValueError(
                    f"No physical fast-coordinate probe exists for column {column}."
                )
            shape_jacobian[:, column] = shape_delta / denominator
            offset_jacobian[:, column] = offset_delta / denominator
        relative = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        variances = np.asarray(profile.session_intercept_variances, dtype=float)
        relative_covariance = np.diag(variances[1:])
        relative_covariance += float(variances[0])
        return SpinOrbitLinearization(
            profile=_session_shape_profile(profile),
            spin_reference=fast.copy(),
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
        )


def search_conditional_slow_cell(
    *,
    projector: PintEquivalentDDPhaseProjector,
    cell: OrbitalCycleCell,
    config: ConditionalSlowCellSearchConfig | None = None,
) -> tuple[ConditionalSlowCellMode, ...]:
    """Cover one orbital-frequency cell with protected slow subcells."""

    active = config or ConditionalSlowCellSearchConfig()
    subdivisions = (
        max(int(active.phi_subdivisions), 1),
        max(int(active.eccentricity_subdivisions), 1),
        max(int(active.periastron_rate_subdivisions), 1),
    )
    support = projector.support
    slow_midpoint = 0.5 * (support.slow_lower + support.slow_upper)
    normalized_c_lower = (
        float(cell.lower) - float(slow_midpoint[0])
    ) / float(support.slow_scale[0])
    normalized_c_upper = (
        float(cell.upper) - float(slow_midpoint[0])
    ) / float(support.slow_scale[0])
    edges = tuple(
        np.linspace(-1.0, 1.0, count + 1, dtype=float)
        for count in subdivisions
    )
    residual_count = len(projector.evaluator.toas.table)
    failure = np.full(residual_count, 1.0e8, dtype=float)
    epsilon = 32.0 * np.finfo(float).eps
    modes: list[ConditionalSlowCellMode] = []
    subcell_index = 0
    for phi_index in range(subdivisions[0]):
        for eccentricity_index in range(subdivisions[1]):
            for periastron_rate_index in range(subdivisions[2]):
                lower = np.asarray(
                    (
                        normalized_c_lower,
                        edges[0][phi_index],
                        edges[1][eccentricity_index],
                        edges[2][periastron_rate_index],
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                    ),
                    dtype=float,
                ) + epsilon
                upper = np.asarray(
                    (
                        normalized_c_upper,
                        edges[0][phi_index + 1],
                        edges[1][eccentricity_index + 1],
                        edges[2][periastron_rate_index + 1],
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ),
                    dtype=float,
                ) - epsilon
                slow_start = 0.5 * (lower[:4] + upper[:4])
                try:
                    conditional = projector.profile_fast_shape(
                        slow_start,
                        maximum_function_evaluations=max(
                            int(active.fast_projection_evaluations),
                            8,
                        ),
                    )
                    start = np.concatenate(
                        (slow_start, conditional.fast_normalized_values)
                    )
                except (ArithmeticError, RuntimeError, ValueError):
                    start = 0.5 * (lower + upper)
                    start[4:] = support.normalize(
                        support.slow_center,
                        support.fast_center,
                    )[4:]
                    start = np.clip(start, lower, upper)

                def residuals(normalized: np.ndarray) -> np.ndarray:
                    try:
                        physical = projector._physical_from_normalized(
                            normalized[:4],
                            normalized[4:],
                        )
                        profile = projector.profile_physical(physical)
                        values = np.asarray(
                            profile.normalized_local_residuals,
                            dtype=float,
                        )
                        return values if values.shape == failure.shape else failure
                    except (ArithmeticError, RuntimeError, ValueError):
                        return failure

                result = least_squares(
                    residuals,
                    start,
                    bounds=(lower, upper),
                    method="trf",
                    jac="2-point",
                    x_scale="jac",
                    ftol=1.0e-9,
                    xtol=1.0e-9,
                    gtol=1.0e-9,
                    max_nfev=max(int(active.maximum_function_evaluations), 8),
                )
                normalized = np.asarray(result.x, dtype=float)
                try:
                    physical = projector._physical_from_normalized(
                        normalized[:4],
                        normalized[4:],
                    )
                    profile = projector.profile_physical(physical)
                    shape_chi2 = float(profile.local_chi2)
                    local_offsets = profile.local_integer_offsets
                except (ArithmeticError, RuntimeError, ValueError):
                    physical = np.full(8, math.nan, dtype=float)
                    shape_chi2 = math.inf
                    local_offsets = tuple()
                slow_boundary_distance = np.minimum(
                    normalized[:4] - lower[:4],
                    upper[:4] - normalized[:4],
                )
                modes.append(
                    ConditionalSlowCellMode(
                        cell_id=cell.cell_id,
                        subcell_index=subcell_index,
                        grid_index=(
                            phi_index,
                            eccentricity_index,
                            periastron_rate_index,
                        ),
                        normalized_lower=lower,
                        normalized_upper=upper,
                        normalized_values=normalized,
                        parameter_values=physical,
                        local_integer_offsets=local_offsets,
                        shape_chi2=shape_chi2,
                        function_evaluations=int(result.nfev),
                        converged=bool(result.success),
                        status=int(result.status),
                        message=str(result.message),
                        on_subcell_boundary=bool(
                            np.min(slow_boundary_distance) <= 1.0e-7
                        ),
                    )
                )
                subcell_index += 1
    modes.sort(key=lambda item: (item.shape_chi2, item.subcell_index))
    return tuple(modes)


__all__ = [
    "ConditionalFastProfile",
    "ConditionalJointIntegerScore",
    "FullWrappedPosteriorScore",
    "ConditionalPartitionPathProfile",
    "ConditionalSlowPartitionConfig",
    "ConditionalSlowPartitionEvaluation",
    "ConditionalSlowPartitionLikelihood",
    "ConditionalSlowCellMode",
    "ConditionalSlowCellSearchConfig",
    "PintEquivalentDDPhaseProjector",
    "conditional_mode_to_cell_mode",
    "dd_binary_delay_seconds",
    "fit_conditional_slow_chart",
    "refine_conditional_slow_mode",
    "score_conditional_joint_integer_mode",
    "score_full_wrapped_posterior_mode",
    "search_conditional_slow_cell",
]
