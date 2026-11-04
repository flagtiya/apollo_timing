from __future__ import annotations

from apollo_timing.inference._shared_helpers import (
    _bounded_profile_probe,
    _finite_vector,
    _project_to_feasible_segment,
    _screen_exact_path_proposals,
    _select_path_diverse_particles,
)

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from apollo_timing.inference.timing_constraints import SkyPositionConstraint, resolve_parameter_bounds
from scipy.optimize import least_squares, lsq_linear

from apollo_timing.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.bayesian_binary_model_adapter import (
    build_binary_coordinate_adapter,
    build_binary_lattice_support,
    build_binary_physical_support,
    detect_binary_model_name,
)
from apollo_timing.inference.bayesian_isolated_model_adapter import (
    build_isolated_coordinate_adapter,
    build_isolated_lattice_support,
    build_isolated_physical_support,
)
from apollo_timing.inference.bayesian_isolated_cold_start import (
    IsolatedColdStartConfig,
    IsolatedColdStartMode,
    search_isolated_cold_start_modes,
)
from apollo_timing.inference.bayesian_robust_phase_likelihood import (
    RobustPhaseNoiseModel,
)
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    OrbitalCycleCell,
    SourceSession,
    TwoLevelPulseProfile,
    _initialize_session_local_pulse_numbers,
    infer_contiguous_source_sessions,
    profile_fixed_local_pulse_component,
    profile_two_level_pulse_lattice,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    session_shape_profile_from_two_level,
)
from apollo_timing.inference.bayesian_spin_integer_orbit import (
    SpinFixedIntegerPathEvaluation,
    SpinIntegerMarginalConfig,
    SpinIntegerMarginalEvaluation,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
    marginalize_spin_for_fixed_integer_path,
)




def detect_phase_model_name(model: object) -> str:
    """Return the APOLLO model label without requiring a binary component."""

    if "BINARY" not in model.params or model["BINARY"].value is None:
        return "ISOLATED"
    return detect_binary_model_name(model)


class PintPhaseEvaluator:
    """Exact PINT phase evaluator shared by isolated and binary models."""

    def __init__(
        self,
        *,
        model: object,
        toas: object,
        parameter_names: Sequence[str],
        sessions: Sequence[SourceSession],
        base_pulse_numbers: Sequence[float],
    ) -> None:
        names = tuple(str(name).upper() for name in parameter_names)
        if len(set(names)) != len(names):
            raise ValueError("PINT binary evaluation needs unique free parameters.")
        missing = [name for name in names if name not in model.params]
        if missing:
            raise ValueError("PINT model is missing: " + ", ".join(missing))
        pulse_numbers = np.asarray(base_pulse_numbers, dtype=np.longdouble)
        if pulse_numbers.shape != (len(toas.table),) or not np.all(np.isfinite(pulse_numbers)):
            raise ValueError("Base pulse numbers must align with every TOA.")
        values = []
        for name in names:
            try:
                value = float(model[name].value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Free PINT parameter '{name}' is not numeric.") from exc
            if not np.isfinite(value):
                raise ValueError(f"Free PINT parameter '{name}' is not finite.")
            values.append(value)
        self.model = model
        self.toas = toas
        self.model_name = detect_phase_model_name(model)
        self.parameter_names = names
        self.sessions = tuple(sessions)
        self.base_pulse_numbers = pulse_numbers
        self.base_parameter_values = np.asarray(values, dtype=float)
        self.robust_phase_noise_model: RobustPhaseNoiseModel | None = None

    @classmethod
    def from_files(
        cls,
        *,
        par_path: Path,
        tim_path: Path,
        clock_override_path: Path | None = None,
    ) -> "PintPhaseEvaluator":
        model = _load_pint_model(Path(par_path))
        free = tuple(str(name).upper() for name in model.free_params)
        if not free:
            raise ValueError("The input par has no free parameters for APOLLO to search.")
        toas = _load_pint_toas(
            Path(tim_path),
            seed_model=model,
            clock_override_path=clock_override_path,
        )
        sessions = infer_contiguous_source_sessions(toas)
        if len(sessions) < 2:
            raise ValueError("Bayesian integer-path search requires at least two sessions.")
        base_pulse_numbers = _initialize_session_local_pulse_numbers(toas, model, sessions)
        return cls(
            model=model,
            toas=toas,
            parameter_names=free,
            sessions=sessions,
            base_pulse_numbers=base_pulse_numbers,
        )

    @property
    def observation_epochs_mjd(self) -> np.ndarray:
        from astropy import units as u

        return np.asarray(self.toas.get_mjds().to_value(u.day), dtype=float)

    def reported_phase_sigmas(self) -> np.ndarray:
        from astropy import units as u

        try:
            uncertainties = self.model.scaled_toa_uncertainty(self.toas)
        except (AttributeError, TypeError, ValueError):
            uncertainties = self.toas.get_errors()
        sigma_seconds = np.asarray(uncertainties.to_value(u.s), dtype=float)
        return np.maximum(
            sigma_seconds * abs(float(self.model["F0"].value)),
            1.0e-18,
        )

    def phase_sigmas(self) -> np.ndarray:
        if self.robust_phase_noise_model is not None:
            return np.asarray(
                self.robust_phase_noise_model.core_sigmas,
                dtype=float,
            )
        return self.reported_phase_sigmas()

    def _set_parameters(self, parameter_values: Sequence[float]) -> None:
        values = _finite_vector(
            parameter_values,
            len(self.parameter_names),
            "PINT binary parameters",
        )
        for name, value in zip(self.parameter_names, values, strict=True):
            _set_numeric_model_parameter_value(self.model, name, float(value))

    def raw_phase_residuals(self, parameter_values: Sequence[float]) -> np.ndarray:
        self._set_parameters(parameter_values)
        phase = self.model.phase(self.toas, abs_phase=True)
        values = np.asarray(phase.int.value, dtype=np.longdouble)
        values += np.asarray(phase.frac.value, dtype=np.longdouble)
        return np.asarray(values - self.base_pulse_numbers, dtype=float)

    def evaluate_values(
        self,
        parameter_values: Sequence[float],
        *,
        local_integer_radius: int = 0,
        maximum_retained_local_modes: int | None = None,
    ) -> TwoLevelPulseProfile:
        residuals = self.raw_phase_residuals(parameter_values)
        if self.robust_phase_noise_model is not None:
            robust = self.robust_phase_noise_model.profile(
                residuals,
                self.sessions,
            )
            return TwoLevelPulseProfile(
                normalized_local_residuals=robust.normalized_local_residuals,
                session_intercepts=robust.session_intercepts,
                session_intercept_variances=robust.session_intercept_variances,
                local_integer_offsets=robust.local_integer_offsets,
                local_chi2=robust.local_deviance,
                local_log_likelihood=robust.local_log_likelihood,
            )
        return profile_two_level_pulse_lattice(
            residuals,
            self.phase_sigmas(),
            self.sessions,
            integer_radius=max(int(local_integer_radius), 0),
            maximum_retained_modes=maximum_retained_local_modes,
        )

    def evaluate_fixed_local_component(
        self,
        parameter_values: Sequence[float],
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
    ) -> TwoLevelPulseProfile:
        component = (
            tuple(tuple(0 for _ in session.toa_indices) for session in self.sessions)
            if local_integer_offsets is None
            else tuple(tuple(int(value) for value in offsets) for offsets in local_integer_offsets)
        )
        residuals = self.raw_phase_residuals(parameter_values)
        if self.robust_phase_noise_model is not None:
            robust = self.robust_phase_noise_model.profile(
                residuals,
                self.sessions,
                fixed_local_integer_offsets=component,
            )
            return TwoLevelPulseProfile(
                normalized_local_residuals=robust.normalized_local_residuals,
                session_intercepts=robust.session_intercepts,
                session_intercept_variances=robust.session_intercept_variances,
                local_integer_offsets=robust.local_integer_offsets,
                local_chi2=robust.local_deviance,
                local_log_likelihood=robust.local_log_likelihood,
            )
        return profile_fixed_local_pulse_component(
            residuals,
            self.phase_sigmas(),
            self.sessions,
            component,
        )


@dataclass(frozen=True)
class ContinuousPriorComponent:
    """One continuous-parameter component in the phase-path posterior."""

    component_id: str
    normalized_mean: np.ndarray
    normalized_covariance: np.ndarray
    log_weight: float = 0.0
    fixed_integer_offsets: tuple[tuple[int, int], ...] = tuple()

    def __post_init__(self) -> None:
        mean = np.asarray(self.normalized_mean, dtype=float)
        covariance = np.asarray(self.normalized_covariance, dtype=float)
        if mean.ndim != 1 or not np.all(np.isfinite(mean)):
            raise ValueError("A continuous-prior mean must be a finite vector.")
        if covariance.shape != (mean.size, mean.size) or not np.all(
            np.isfinite(covariance)
        ):
            raise ValueError("A continuous-prior covariance must match its mean.")
        covariance = 0.5 * (covariance + covariance.T)
        if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
            raise ValueError("A continuous-prior covariance must be positive definite.")
        if not self.component_id or not np.isfinite(float(self.log_weight)):
            raise ValueError("A continuous-prior component needs an id and finite weight.")
        fixed = tuple((int(index), int(value)) for index, value in self.fixed_integer_offsets)
        if len({index for index, _value in fixed}) != len(fixed):
            raise ValueError("Fixed integer offsets must use unique path indices.")
        object.__setattr__(self, "normalized_mean", mean.copy())
        object.__setattr__(self, "normalized_covariance", covariance)
        object.__setattr__(self, "fixed_integer_offsets", fixed)


@dataclass(frozen=True)
class BinaryLatticeSearchContext:
    evaluator: PintPhaseEvaluator
    adapter: Any
    support: Any
    cells: tuple[OrbitalCycleCell, ...]
    continuous_prior_components: tuple[ContinuousPriorComponent, ...] = tuple()
    isolated_cold_start_modes: tuple[IsolatedColdStartMode, ...] = tuple()


def _tile_search_cycle_support(
    lower: float,
    upper: float,
    *,
    maximum_cell_width: float,
) -> tuple[OrbitalCycleCell, ...]:
    width = float(maximum_cell_width)
    if not all(np.isfinite(value) for value in (lower, upper, width)):
        raise ValueError("Orbital-cycle support must be finite.")
    if upper <= lower or width <= 0.0:
        raise ValueError("Orbital-cycle support must have positive width.")
    count = max(int(math.ceil((upper - lower) / width)), 1)
    edges = np.linspace(lower, upper, count + 1, dtype=float)
    digits = max(6, len(str(count)))
    return tuple(
        OrbitalCycleCell(
            cell_id=f"C{index:0{digits}d}",
            lower=float(edges[index]),
            upper=float(edges[index + 1]),
        )
        for index in range(count)
    )


def build_phase_lattice_search_context(
    *,
    par_path: Path,
    tim_path: Path,
    maximum_cell_width: float = 0.25,
    support_width_scale: float = 1.0,
    astrometric_bound_arcsec: float = 300.0,
    isolated_cold_start_config: IsolatedColdStartConfig | None = None,
    clock_override_path: Path | None = None,
    custom_bounds: dict[str, float | tuple[float, float]] | None = None,
    physical_parameter_bounds: Sequence[tuple[str, float, float]] = (),
    sky_constraint: SkyPositionConstraint | None = None,
    optional_parameters: Sequence[str] = (),
) -> BinaryLatticeSearchContext:
    evaluator = PintPhaseEvaluator.from_files(
        par_path=Path(par_path),
        tim_path=Path(tim_path),
        clock_override_path=clock_override_path,
    )
    physical_parameter_bounds = resolve_parameter_bounds(
        evaluator.model, physical_parameter_bounds, custom_bounds,
        optional_parameters=optional_parameters,
    )
    intervals = {name: (lo, hi) for name, lo, hi in physical_parameter_bounds}
    # Some zero-valued terms have zero automatic width. Supply a preliminary
    # enclosing width so the adapter can be built before applying exact limits.
    custom_bounds = {name: max(float(evaluator.model[name].value) - lo,
                               hi - float(evaluator.model[name].value))
                     for name, lo, hi in physical_parameter_bounds if name in evaluator.parameter_names} or None
    if sky_constraint is not None and not sky_constraint.report(evaluator.model)["within_bounds"]:
        raise ValueError("Prepared seed lies outside the original sky position constraint.")
    epochs = evaluator.observation_epochs_mjd
    span_days = float(np.ptp(epochs))
    if not np.isfinite(span_days) or span_days <= 0.0:
        raise ValueError("Binary lattice search requires distinct TOA epochs.")
    reference_mjd = float(np.median(epochs))
    cold_start_modes: tuple[IsolatedColdStartMode, ...] = tuple()
    continuous_prior_components: tuple[ContinuousPriorComponent, ...] = tuple()
    def apply_bounds(center: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
        if sky_constraint is not None:
            sky_intervals = sky_constraint.bounding_intervals(sky_constraint.model_values(evaluator.model))
            # Replace automatic coordinate widths with the cap's enclosing box.
            # Explicit user intervals/half-widths still intersect this box.
            for name, (lo, hi) in sky_intervals.items():
                if name not in evaluator.parameter_names:
                    continue
                index = evaluator.parameter_names.index(name)
                if name in intervals:
                    lo, hi = max(lo, intervals[name][0]), min(hi, intervals[name][1])
                if not lo < hi or not lo <= center[index] <= hi:
                    raise ValueError(f"Requested sky region conflicts with the initial {name} or its interval.")
                lower[index], upper[index] = lo, hi
        for name, lo, hi in physical_parameter_bounds:
            name = name.upper()
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"Invalid finite physical support for {name}.")
            if name in evaluator.parameter_names:
                index = evaluator.parameter_names.index(name)
                if not lo <= center[index] <= hi:
                    raise ValueError(f"The initial {name} is outside the requested physical support.")
                if sky_constraint is not None and name in {sky_constraint.longitude_name, sky_constraint.latitude_name}:
                    lower[index], upper[index] = max(lower[index], lo), min(upper[index], hi)
                else:
                    lower[index], upper[index] = lo, hi
            elif name in evaluator.model.params and evaluator.model[name].value is not None:
                if not lo <= float(evaluator.model[name].value) <= hi:
                    raise ValueError(f"Fixed {name} conflicts with its physical support.")
    if evaluator.model_name == "ISOLATED":
        cold_start = isolated_cold_start_config
        cold_enabled = bool(cold_start is not None and cold_start.enabled)
        isolated_astrometric_bound = (
            max(float(astrometric_bound_arcsec), cold_start.astrometric_bound_arcsec)
            if cold_enabled
            else float(astrometric_bound_arcsec)
        )
        adapter = build_isolated_coordinate_adapter(
            model=evaluator.model,
            parameter_names=evaluator.parameter_names,
            observation_span_days=span_days,
            reference_mjd=reference_mjd,
        )
        center, lower, upper = build_isolated_physical_support(
            model=evaluator.model,
            parameter_names=evaluator.parameter_names,
            observation_span_days=span_days,
            width_scale=support_width_scale,
            astrometric_bound_arcsec=isolated_astrometric_bound,
            proper_motion_half_width_mas_per_year=(
                cold_start.proper_motion_half_width_mas_per_year
                if cold_enabled
                else 0.0
            ),
            minimum_frequency_half_width_hz=(
                cold_start.frequency_half_width_hz if cold_enabled else 0.0
            ),
            minimum_frequency_derivative_half_width_hz_s=(
                cold_start.frequency_derivative_half_width_hz_s
                if cold_enabled
                else 0.0
            ),
            custom_bounds=custom_bounds,
        )
        apply_bounds(center, lower, upper)
        support = build_isolated_lattice_support(
            adapter=adapter,
            physical_center=center,
            physical_lower=lower,
            physical_upper=upper,
        )
        if sky_constraint is not None:
            support = replace(support, sky_constraint=sky_constraint)
        if cold_enabled:
            evaluator.robust_phase_noise_model = RobustPhaseNoiseModel.infer(
                phase_values=evaluator.raw_phase_residuals(
                    evaluator.base_parameter_values
                ),
                reported_sigmas=evaluator.reported_phase_sigmas(),
                epochs_mjd=epochs,
                sessions=evaluator.sessions,
                config=cold_start.phase_noise,
            )
            cold_start_modes = search_isolated_cold_start_modes(
                evaluator=evaluator,
                support=support,
                config=cold_start,
            )
            covariance = np.diag(
                np.full(
                    support.dimension,
                    float(cold_start.component_prior_sigma) ** 2,
                    dtype=float,
                )
            )
            continuous_prior_components = tuple(
                ContinuousPriorComponent(
                    component_id=mode.mode_id,
                    normalized_mean=mode.normalized_values,
                    normalized_covariance=covariance,
                    log_weight=0.0,
                )
                for mode in cold_start_modes
            )
    else:
        adapter = build_binary_coordinate_adapter(
            model=evaluator.model,
            parameter_names=evaluator.parameter_names,
            observation_span_days=span_days,
            reference_mjd=reference_mjd,
        )
        center, lower, upper = build_binary_physical_support(
            model=evaluator.model,
            parameter_names=evaluator.parameter_names,
            observation_span_days=span_days,
            width_scale=support_width_scale,
            astrometric_bound_arcsec=astrometric_bound_arcsec,
            custom_bounds=custom_bounds,
        )
        apply_bounds(center, lower, upper)
        support = build_binary_lattice_support(
            adapter=adapter,
            physical_center=center,
            physical_lower=lower,
            physical_upper=upper,
        )
        if sky_constraint is not None:
            support = replace(support, sky_constraint=sky_constraint)
    orbital_index = adapter.orbital_coordinate_index
    if orbital_index is None:
        cells = (
            OrbitalCycleCell(cell_id="C_FIXED", lower=0.0, upper=0.0),
        )
    else:
        cells = _tile_search_cycle_support(
            float(support.search_lower[orbital_index]),
            float(support.search_upper[orbital_index]),
            maximum_cell_width=float(maximum_cell_width),
        )
    return BinaryLatticeSearchContext(
        evaluator=evaluator,
        adapter=adapter,
        support=support,
        cells=cells,
        continuous_prior_components=continuous_prior_components,
        isolated_cold_start_modes=cold_start_modes,
    )


def build_binary_lattice_search_context(
    *,
    par_path: Path,
    tim_path: Path,
    maximum_cell_width: float = 0.25,
    support_width_scale: float = 1.0,
    astrometric_bound_arcsec: float = 300.0,
    clock_override_path: Path | None = None,
    custom_bounds: dict[str, float] | None = None,
) -> BinaryLatticeSearchContext:
    """Compatibility wrapper that still requires a declared binary model."""

    context = build_phase_lattice_search_context(
        par_path=par_path,
        tim_path=tim_path,
        maximum_cell_width=maximum_cell_width,
        support_width_scale=support_width_scale,
        astrometric_bound_arcsec=astrometric_bound_arcsec,
        clock_override_path=clock_override_path,
        custom_bounds=custom_bounds,
    )
    if context.evaluator.model_name == "ISOLATED":
        raise ValueError("Binary lattice context requires a declared PINT binary model.")
    return context


def normalized_bounds_for_binary_cell(
    context: BinaryLatticeSearchContext,
    cell: OrbitalCycleCell,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = context.support.dimension
    lower = np.full(dimension, -1.0, dtype=float)
    upper = np.full(dimension, 1.0, dtype=float)
    index = context.adapter.orbital_coordinate_index
    if index is not None:
        midpoint = context.support.search_midpoint[index]
        scale = context.support.search_scale[index]
        lower[index] = (cell.lower - midpoint) / scale
        upper[index] = (cell.upper - midpoint) / scale
    epsilon = 32.0 * np.finfo(float).eps
    return lower + epsilon, upper - epsilon


@dataclass(frozen=True)
class BinaryConditionalProfile:
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: TwoLevelPulseProfile
    function_evaluations: int
    converged: bool


@dataclass(frozen=True)
class BinaryFixedPathEvaluation:
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: TwoLevelPulseProfile
    local_integer_offsets: tuple[tuple[int, ...], ...]
    integer_offsets: tuple[int, ...]
    shape_residuals: np.ndarray
    phase_residuals: np.ndarray
    shape_chi2: float
    phase_chi2: float
    total_chi2: float


@dataclass(frozen=True)
class BinaryConditionalMode:
    model_name: str
    cell_id: str
    grid_index: tuple[int, int]
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    strict_score: float
    log_likelihood: float
    integer_offsets: tuple[int, ...]
    posterior_weight: float
    lattice_mass_converged: bool
    lattice_log_mass_increment: float
    lattice_visited_nodes: int
    posterior_in_support: bool
    prior_component_id: str = "local_broad"
    prior_log_weight: float = 0.0
    source_cell_id: str | None = None
    posterior_transport_fraction: float = 1.0


@dataclass(frozen=True)
class BinaryCellSearchConfig:
    orbital_samples_per_cell: int = 2
    phase_samples: int = 12
    modes_per_cell: int = 32
    paths_per_component: int = 8
    chart_maximum_evaluations: int = 60
    finite_difference_step: float = 2.0e-5
    coordinate_prior_sigma: float = 0.75
    local_integer_radius: int = 0
    lattice: SpinIntegerMarginalConfig = field(
        default_factory=lambda: SpinIntegerMarginalConfig(
            initial_lattice_candidates=32,
            maximum_lattice_candidates=128,
            lattice_log_mass_tolerance=1.0e-5,
            maximum_lattice_nodes=500_000,
            added_session_phase_sigma_cycles=0.05,
        )
    )


@dataclass(frozen=True)
class BinaryAnnealingConfig:
    phase_sigma_schedule: tuple[float, ...] = (
        0.10,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.001,
        0.0005,
        0.0002,
        0.0001,
        0.0,
    )
    coordinate_prior_sigma: float = 0.20
    paths_per_state: int = 3
    integer_path_transport: bool = False
    integer_path_proposal: str = "block"
    screen_lattice_paths: bool = False
    beam_size: int = 4
    finite_difference_step: float = 2.0e-5
    maximum_closure_evaluations: int = 300
    lattice: SpinIntegerMarginalConfig = field(
        default_factory=lambda: SpinIntegerMarginalConfig(
            initial_lattice_candidates=32,
            maximum_lattice_candidates=128,
            lattice_log_mass_tolerance=1.0e-6,
            maximum_lattice_nodes=1_000_000,
        )
    )

    def __post_init__(self) -> None:
        if self.integer_path_proposal not in {"block", "lattice"}:
            raise ValueError("Integer path proposal must be block or lattice.")
        schedule = tuple(float(value) for value in self.phase_sigma_schedule)
        if not schedule or any(not np.isfinite(value) or value < 0.0 for value in schedule):
            raise ValueError("Phase-noise schedule must be finite and nonnegative.")
        if any(right > left for left, right in zip(schedule, schedule[1:])):
            raise ValueError("Phase-noise schedule must be nonincreasing.")
        if schedule[-1] != 0.0:
            raise ValueError("Phase-noise schedule must end at zero.")


@dataclass(frozen=True)
class BinaryAnnealingStep:
    phase_sigma_cycles: float
    integer_offsets: tuple[int, ...]
    normalized_values: np.ndarray
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    lattice_strict_score: float
    lattice_log_likelihood: float
    lattice_mass_converged: bool
    lattice_log_mass_increment: float
    lattice_visited_nodes: int
    function_evaluations: int
    converged: bool
    proposed_path_count: int = 1
    evaluated_path_count: int = 1
    paths_selected_for_closure: int = 1
    path_proposal_scope: str = "fixed_path"


@dataclass(frozen=True)
class BinaryAnnealedMode:
    model_name: str
    cell_id: str
    source_grid_index: tuple[int, int]
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    integer_offsets: tuple[int, ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    converged: bool
    trace: tuple[BinaryAnnealingStep, ...]
    prior_component_id: str = "local_broad"
    prior_log_weight: float = 0.0
    source_cell_id: str | None = None


class PintBinaryLatticeProjector:
    """Dimension-independent exact-PINT projector in adapted coordinates."""

    def __init__(self, context: BinaryLatticeSearchContext) -> None:
        self.context = context
        self.evaluator = context.evaluator
        self.adapter = context.adapter
        self.support = context.support
        self.sessions = tuple(context.evaluator.sessions)
        self.phase_sigmas = context.evaluator.phase_sigmas()

    @property
    def dimension(self) -> int:
        return self.support.dimension

    def profile_normalized(
        self,
        normalized_values: Sequence[float],
        *,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
    ) -> tuple[np.ndarray, TwoLevelPulseProfile]:
        normalized = _finite_vector(
            normalized_values, self.dimension, "Normalized binary state"
        )
        if np.any(normalized < -1.0) or np.any(normalized > 1.0):
            raise ValueError("Normalized binary state lies outside support.")
        physical = self.support.denormalize_physical(normalized)
        if not self.support.contains_physical(physical):
            raise ValueError("Binary state lies outside physical support.")
        profile = self.evaluator.evaluate_fixed_local_component(
            physical,
            local_integer_offsets,
        )
        return physical, profile

    def evaluate_fixed_path(
        self,
        normalized_values: Sequence[float],
        *,
        local_integer_offsets: Sequence[Sequence[int]],
        integer_offsets: Sequence[int],
        phase_sigma_cycles: float = 0.0,
    ) -> BinaryFixedPathEvaluation:
        """Evaluate one continuous state and one pulse path without linearization."""

        normalized = _finite_vector(
            normalized_values,
            self.dimension,
            "Normalized fixed-path state",
        )
        path = tuple(int(value) for value in integer_offsets)
        expected_path_size = max(len(self.sessions) - 1, 0)
        if len(path) != expected_path_size:
            raise ValueError(
                "Fixed integer path length does not match the session lattice: "
                f"expected {expected_path_size}, got {len(path)}."
            )
        sigma = float(phase_sigma_cycles)
        if not math.isfinite(sigma) or sigma < 0.0:
            raise ValueError("Fixed-path phase sigma must be finite and nonnegative.")

        physical, profile = self.profile_normalized(
            normalized,
            local_integer_offsets=local_integer_offsets,
        )
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        anchors = np.asarray(profile.session_intercepts, dtype=float)
        variances = np.asarray(profile.session_intercept_variances, dtype=float)
        if anchors.shape != (len(self.sessions),) or variances.shape != anchors.shape:
            raise ValueError("Session profile does not match the fixed-path lattice.")
        if not (
            np.all(np.isfinite(shape))
            and np.all(np.isfinite(anchors))
            and np.all(np.isfinite(variances))
        ):
            raise ValueError("Fixed-path profile contains non-finite values.")

        if path:
            covariance = np.diag(variances[1:]) + float(variances[0])
            if sigma > 0.0:
                covariance += sigma * sigma * (
                    np.eye(len(path), dtype=float)
                    + np.ones((len(path), len(path)), dtype=float)
                )
            phase = np.linalg.solve(
                np.linalg.cholesky(covariance),
                anchors[1:] - anchors[0] - np.asarray(path, dtype=float),
            )
        else:
            phase = np.empty(0, dtype=float)
        shape_chi2 = float(shape @ shape)
        phase_chi2 = float(phase @ phase)
        return BinaryFixedPathEvaluation(
            normalized_values=normalized.copy(),
            parameter_values=np.asarray(physical, dtype=float),
            profile=profile,
            local_integer_offsets=tuple(
                tuple(int(value) for value in offsets)
                for offsets in profile.local_integer_offsets
            ),
            integer_offsets=path,
            shape_residuals=shape.copy(),
            phase_residuals=phase.copy(),
            shape_chi2=shape_chi2,
            phase_chi2=phase_chi2,
            total_chi2=float(shape_chi2 + phase_chi2),
        )

    def profile_full_shape(
        self,
        normalized_start: Sequence[float],
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        local_integer_offsets: Sequence[Sequence[int]],
        maximum_function_evaluations: int,
    ) -> BinaryConditionalProfile:
        start = _finite_vector(normalized_start, self.dimension, "Binary chart start")
        lower = _finite_vector(normalized_lower, self.dimension, "Binary lower bounds")
        upper = _finite_vector(normalized_upper, self.dimension, "Binary upper bounds")
        if np.any(lower >= upper):
            raise ValueError("Binary normalized bounds must have positive width.")
        start = np.clip(start, lower, upper)
        failure = np.full(len(self.evaluator.toas.table), 1.0e10, dtype=float)

        def residual(state: np.ndarray) -> np.ndarray:
            try:
                _, profile = self.profile_normalized(
                    state,
                    local_integer_offsets=local_integer_offsets,
                )
                values = np.asarray(profile.normalized_local_residuals, dtype=float)
                if values.shape == failure.shape and np.all(np.isfinite(values)):
                    return values
            except (ArithmeticError, RuntimeError, ValueError):
                pass
            return failure

        fit = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=max(int(maximum_function_evaluations), 8),
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        state = np.asarray(fit.x, dtype=float)
        physical, profile = self.profile_normalized(
            state,
            local_integer_offsets=local_integer_offsets,
        )
        return BinaryConditionalProfile(
            normalized_values=state,
            parameter_values=physical,
            profile=profile,
            function_evaluations=int(fit.nfev),
            converged=bool(fit.success),
        )

    def linearize_full(
        self,
        conditional: BinaryConditionalProfile,
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        finite_difference_step: float,
    ) -> SpinOrbitLinearization:
        """Linearize every coordinate in the normalized search chart."""

        return self.linearize_coordinates(
            conditional,
            normalized_lower=normalized_lower,
            normalized_upper=normalized_upper,
            coordinate_indices=tuple(range(self.dimension)),
            finite_difference_step=finite_difference_step,
        )

    def linearize_coordinates(
        self,
        conditional: BinaryConditionalProfile,
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        coordinate_indices: Sequence[int],
        finite_difference_step: float,
    ) -> SpinOrbitLinearization:
        """Linearize only selected coordinates while evaluating the full model."""

        reference = np.asarray(conditional.normalized_values, dtype=float)
        lower = _finite_vector(normalized_lower, self.dimension, "Binary lower bounds")
        upper = _finite_vector(normalized_upper, self.dimension, "Binary upper bounds")
        if reference.shape != (self.dimension,) or np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("Binary linearization reference lies outside its cell.")
        indices = tuple(int(index) for index in coordinate_indices)
        if not indices:
            raise ValueError("Binary linearization requires at least one coordinate.")
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= self.dimension for index in indices
        ):
            raise ValueError("Binary linearization coordinate indices are invalid.")
        profile = conditional.profile
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        offsets = np.asarray(profile.session_intercepts, dtype=float)
        shape_jacobian = np.empty((shape.size, len(indices)), dtype=float)
        offset_jacobian = np.empty((offsets.size, len(indices)), dtype=float)
        requested_step = max(float(finite_difference_step), 1.0e-7)

        def probe(index: int, direction: float):
            return _bounded_profile_probe(
                reference, index=index, direction=direction, lower=lower, upper=upper,
                requested_step=requested_step,
                evaluate=lambda state: self.profile_normalized(
                    state, local_integer_offsets=profile.local_integer_offsets,
                )[1],
            )

        for column, index in enumerate(indices):
            plus = probe(index, 1.0)
            minus = probe(index, -1.0)
            if plus is not None and minus is not None:
                plus_step, plus_profile = plus
                minus_step, minus_profile = minus
                denominator = plus_step - minus_step
                shape_delta = np.asarray(plus_profile.normalized_local_residuals) - np.asarray(
                    minus_profile.normalized_local_residuals
                )
                offset_delta = np.asarray(plus_profile.session_intercepts) - np.asarray(
                    minus_profile.session_intercepts
                )
            elif plus is not None:
                plus_step, plus_profile = plus
                denominator = plus_step
                shape_delta = np.asarray(plus_profile.normalized_local_residuals) - shape
                offset_delta = np.asarray(plus_profile.session_intercepts) - offsets
            elif minus is not None:
                minus_step, minus_profile = minus
                denominator = -minus_step
                shape_delta = shape - np.asarray(minus_profile.normalized_local_residuals)
                offset_delta = offsets - np.asarray(minus_profile.session_intercepts)
            else:
                raise ValueError(f"No physical probe exists for binary coordinate {index}.")
            shape_jacobian[:, column] = shape_delta / denominator
            offset_jacobian[:, column] = offset_delta / denominator
        relative = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        variances = np.asarray(profile.session_intercept_variances, dtype=float)
        covariance = np.diag(variances[1:]) + float(variances[0])
        return SpinOrbitLinearization(
            profile=session_shape_profile_from_two_level(profile),
            spin_reference=reference[list(indices)].copy(),
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=covariance,
        )


def _local_component_at_state(
    projector: PintBinaryLatticeProjector,
    state: np.ndarray,
    *,
    radius: int,
) -> tuple[tuple[int, ...], ...]:
    physical = projector.support.denormalize_physical(state)
    profile = projector.evaluator.evaluate_values(
        physical,
        local_integer_radius=max(int(radius), 0),
        maximum_retained_local_modes=1,
    )
    return profile.local_integer_offsets


def _continuous_prior_components(
    context: BinaryLatticeSearchContext,
    *,
    reference: np.ndarray,
    default_sigma: float,
) -> tuple[ContinuousPriorComponent, ...]:
    if context.continuous_prior_components:
        return context.continuous_prior_components
    dimension = context.support.dimension
    sigma = max(float(default_sigma), 1.0e-6)
    return (
        ContinuousPriorComponent(
            component_id="local_broad",
            normalized_mean=np.asarray(reference, dtype=float),
            normalized_covariance=np.diag(
                np.full(dimension, sigma * sigma, dtype=float)
            ),
        ),
    )


def _path_matches_prior_component(
    integer_offsets: Sequence[int],
    component: ContinuousPriorComponent,
) -> bool:
    path = tuple(int(value) for value in integer_offsets)
    return all(
        0 <= index < len(path) and path[index] == required
        for index, required in component.fixed_integer_offsets
    )


def _prior_component_by_id(
    context: BinaryLatticeSearchContext,
    component_id: str,
    *,
    reference: np.ndarray,
    default_sigma: float,
) -> ContinuousPriorComponent:
    components = _continuous_prior_components(
        context,
        reference=reference,
        default_sigma=default_sigma,
    )
    for component in components:
        if component.component_id == component_id:
            return component
    raise ValueError(f"Unknown continuous-prior component '{component_id}'.")


def _select_coherent_fixed_path_state(
    *,
    projector: PintBinaryLatticeProjector,
    normalized_states: Sequence[Sequence[float]],
    local_integer_offsets: Sequence[Sequence[int]],
    integer_offsets: Sequence[int],
) -> BinaryFixedPathEvaluation:
    exact_states: list[BinaryFixedPathEvaluation] = []
    for state in normalized_states:
        normalized = np.asarray(state, dtype=float)
        if any(
            np.allclose(
                normalized,
                exact.normalized_values,
                rtol=0.0,
                atol=1.0e-12,
            )
            for exact in exact_states
        ):
            continue
        try:
            exact_states.append(
                projector.evaluate_fixed_path(
                    normalized,
                    local_integer_offsets=local_integer_offsets,
                    integer_offsets=integer_offsets,
                )
            )
        except (
            ArithmeticError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            continue
    if not exact_states:
        raise ValueError("No coherent fixed-path state could be evaluated.")
    return min(exact_states, key=lambda evaluation: evaluation.total_chi2)


def search_binary_orbital_cycle_cell(
    *,
    context: BinaryLatticeSearchContext,
    cell: OrbitalCycleCell,
    config: BinaryCellSearchConfig | None = None,
) -> tuple[BinaryConditionalMode, ...]:
    active = config or BinaryCellSearchConfig()
    projector = PintBinaryLatticeProjector(context)
    lower, upper = normalized_bounds_for_binary_cell(context, cell)
    initial = context.support.normalize_search(context.support.search_center)
    orbital_index = context.adapter.orbital_coordinate_index
    phase_index = context.adapter.phase_coordinate_index
    if orbital_index is None:
        orbital_values = (None,)
    else:
        orbital_values = tuple(
            np.linspace(
                lower[orbital_index],
                upper[orbital_index],
                max(int(active.orbital_samples_per_cell), 1) + 2,
                dtype=float,
            )[1:-1]
        )
    if phase_index is None:
        phase_values = (None,)
    else:
        phase_values = tuple(
            np.linspace(
                lower[phase_index],
                upper[phase_index],
                max(int(active.phase_samples), 2),
                endpoint=False,
                dtype=float,
            )
        )
    starts: list[tuple[tuple[int, int], np.ndarray, str | None]] = []
    for component_index, component in enumerate(context.continuous_prior_components):
        state = np.asarray(component.normalized_mean, dtype=float)
        if np.all(state >= lower) and np.all(state <= upper):
            starts.append(
                ((-2, component_index), state.copy(), component.component_id)
            )
    for orbital_grid, orbital_value in enumerate(orbital_values):
        for phase_grid, phase_value in enumerate(phase_values):
            state = initial.copy()
            if orbital_index is not None:
                state[orbital_index] = float(orbital_value)
            if phase_index is not None:
                state[phase_index] = float(phase_value)
            starts.append(((orbital_grid, phase_grid), state, None))
    if np.all(initial >= lower) and np.all(initial <= upper):
        starts.append(((-1, -1), np.clip(initial, lower, upper), None))

    unique_starts: list[tuple[tuple[int, int], np.ndarray, str | None]] = []
    for grid_index, state, component_id in starts:
        if any(
            np.allclose(state, prior, rtol=0.0, atol=1.0e-10)
            and (
                component_id is None
                or prior_component_id is None
                or component_id == prior_component_id
            )
            for _grid, prior, prior_component_id in unique_starts
        ):
            continue
        unique_starts.append((grid_index, state, component_id))

    modes: list[BinaryConditionalMode] = []
    for grid_index, reference, start_component_id in unique_starts:
        try:
            local_component = _local_component_at_state(
                projector,
                reference,
                radius=active.local_integer_radius,
            )
            conditional = projector.profile_full_shape(
                reference,
                normalized_lower=lower,
                normalized_upper=upper,
                local_integer_offsets=local_component,
                maximum_function_evaluations=active.chart_maximum_evaluations,
            )
            linearization = projector.linearize_full(
                conditional,
                normalized_lower=lower,
                normalized_upper=upper,
                finite_difference_step=active.finite_difference_step,
            )
            components = _continuous_prior_components(
                context,
                reference=conditional.normalized_values,
                default_sigma=active.coordinate_prior_sigma,
            )
            if start_component_id is not None:
                components = tuple(
                    component
                    for component in components
                    if component.component_id == start_component_id
                )
            for component in components:
                marginal: SpinIntegerMarginalEvaluation = marginalize_spin_and_integer_paths(
                    linearization,
                    spin_prior_mean=(
                        component.normalized_mean - conditional.normalized_values
                    ),
                    spin_prior_covariance=component.normalized_covariance,
                    config=active.lattice,
                )
                paths = tuple(
                    path
                    for path in marginal.path_modes
                    if _path_matches_prior_component(path.integer_offsets, component)
                )
                if not paths:
                    continue
                for path in paths[: max(int(active.paths_per_component), 1)]:
                    unconstrained_posterior = (
                        conditional.normalized_values
                        + np.asarray(path.spin_posterior_mean, dtype=float)
                    )
                    try:
                        posterior = _project_to_support_segment(
                            reference=conditional.normalized_values,
                            target=unconstrained_posterior,
                            projector=projector,
                            lower=lower,
                            upper=upper,
                            local_integer_offsets=local_component,
                        )
                    except (
                        ArithmeticError,
                        RuntimeError,
                        ValueError,
                        np.linalg.LinAlgError,
                    ):
                        continue
                    try:
                        exact = _select_coherent_fixed_path_state(
                            projector=projector,
                            normalized_states=(
                                conditional.normalized_values,
                                posterior,
                            ),
                            local_integer_offsets=local_component,
                            integer_offsets=path.integer_offsets,
                        )
                    except ValueError:
                        continue
                    in_support = bool(
                        context.support.contains_physical(exact.parameter_values)
                    )
                    modes.append(
                        BinaryConditionalMode(
                            model_name=context.evaluator.model_name,
                            cell_id=cell.cell_id,
                            grid_index=grid_index,
                            normalized_values=exact.normalized_values.copy(),
                            parameter_values=exact.parameter_values.copy(),
                            local_integer_offsets=exact.local_integer_offsets,
                            shape_chi2=exact.shape_chi2,
                            phase_chi2=exact.phase_chi2,
                            total_chi2=exact.total_chi2,
                            strict_score=exact.total_chi2,
                            log_likelihood=float(
                                marginal.log_likelihood + component.log_weight
                            ),
                            integer_offsets=tuple(
                                int(value) for value in path.integer_offsets
                            ),
                            posterior_weight=float(
                                path.posterior_weight
                                * math.exp(component.log_weight)
                            ),
                            lattice_mass_converged=bool(
                                marginal.lattice_mass_converged
                            ),
                            lattice_log_mass_increment=float(
                                marginal.lattice_log_mass_increment
                            ),
                            lattice_visited_nodes=int(
                                marginal.lattice_visited_nodes
                            ),
                            posterior_in_support=in_support,
                            prior_component_id=component.component_id,
                            prior_log_weight=float(component.log_weight),
                        )
                    )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    modes.sort(
        key=lambda item: (
            not item.posterior_in_support,
            -item.log_likelihood,
            item.strict_score,
            item.grid_index,
        )
    )
    limit = max(int(active.modes_per_cell), 1)
    if not context.continuous_prior_components:
        return tuple(modes[:limit])
    selected: list[BinaryConditionalMode] = []
    selected_ids: set[int] = set()
    for component in context.continuous_prior_components:
        best = next(
            (
                mode
                for mode in modes
                if mode.prior_component_id == component.component_id
            ),
            None,
        )
        if best is not None and len(selected) < limit:
            selected.append(best)
            selected_ids.add(id(best))
    for mode in modes:
        if len(selected) >= limit:
            break
        if id(mode) not in selected_ids:
            selected.append(mode)
    selected.sort(
        key=lambda item: (
            not item.posterior_in_support,
            -item.log_likelihood,
            item.strict_score,
            item.grid_index,
        )
    )
    return tuple(selected)


@dataclass(frozen=True)
class _AnnealingParticle:
    normalized_values: np.ndarray
    integer_offsets: tuple[int, ...]
    objective: float
    trace: tuple[BinaryAnnealingStep, ...]


def _project_to_support_segment(
    *,
    reference: np.ndarray,
    target: np.ndarray,
    projector: PintBinaryLatticeProjector,
    lower: np.ndarray,
    upper: np.ndarray,
    local_integer_offsets: Sequence[Sequence[int]],
    reference_profile_validated: bool = False,
) -> np.ndarray:
    return _project_to_feasible_segment(
        reference=reference, target=target, lower=lower, upper=upper,
        contains=lambda state: projector.support.contains_physical(
            projector.support.denormalize_physical(state),
        ),
        evaluate=lambda state: projector.profile_normalized(
            state, local_integer_offsets=local_integer_offsets,
        ),
        reference_validated=reference_profile_validated,
    )


def _close_softened_path(
    *,
    projector: PintBinaryLatticeProjector,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    local_integer_offsets: Sequence[Sequence[int]],
    integer_offsets: Sequence[int],
    phase_sigma_cycles: float,
    maximum_function_evaluations: int,
) -> tuple[np.ndarray, TwoLevelPulseProfile, float, float, int, bool]:
    path = tuple(int(value) for value in integer_offsets)
    sigma = float(phase_sigma_cycles)
    _, reference_profile = projector.profile_normalized(
        start,
        local_integer_offsets=local_integer_offsets,
    )
    shape_size = np.asarray(reference_profile.normalized_local_residuals).size
    failure = np.full(shape_size + len(path), 1.0e12, dtype=float)

    def residual(state: np.ndarray) -> np.ndarray:
        try:
            evaluation = projector.evaluate_fixed_path(
                state,
                local_integer_offsets=local_integer_offsets,
                integer_offsets=path,
                phase_sigma_cycles=sigma,
            )
            combined = np.concatenate(
                (evaluation.shape_residuals, evaluation.phase_residuals)
            )
            if combined.shape == failure.shape and np.all(np.isfinite(combined)):
                return combined
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        return failure

    fit = least_squares(
        residual,
        np.asarray(start, dtype=float),
        bounds=(lower, upper),
        method="trf",
        jac="2-point",
        x_scale="jac",
        max_nfev=max(int(maximum_function_evaluations), 8),
        ftol=1.0e-10,
        xtol=1.0e-10,
        gtol=1.0e-10,
    )
    state = np.asarray(fit.x, dtype=float)
    evaluation = projector.evaluate_fixed_path(
        state,
        local_integer_offsets=local_integer_offsets,
        integer_offsets=path,
        phase_sigma_cycles=sigma,
    )
    return (
        state,
        evaluation.profile,
        evaluation.shape_chi2,
        evaluation.phase_chi2,
        int(fit.nfev),
        bool(fit.success),
    )


def _propose_integer_path_transport_moves(
    *,
    linearization: SpinOrbitLinearization,
    current_path: Sequence[int],
    fixed_path: SpinFixedIntegerPathEvaluation,
    phase_sigma_cycles: float,
    maximum_paths: int,
) -> tuple[tuple[int, ...], ...]:
    """Generate a small Bayesian bridge-move beam around one complete path.

    A full session lattice grows exponentially with the number of sessions.
    Ambiguity in a phase-connected sequence usually takes the form of a block
    offset: changing one session edge shifts one anchor or the whole suffix
    after a gap. The conditional Gaussian predictive phase posterior scores
    these two move families. Exact PINT closure selects among the retained
    moves. The bounded block transport kernel uses no source-specific wrap
    rule.
    """

    path = np.asarray(tuple(int(value) for value in current_path), dtype=float)
    relative = np.asarray(linearization.relative_session_offsets, dtype=float)
    jacobian = np.asarray(linearization.relative_session_jacobian, dtype=float)
    if path.shape != relative.shape:
        raise ValueError("Integer transport path must match relative phases.")
    phase_covariance = np.asarray(
        linearization.relative_offset_covariance,
        dtype=float,
    )
    sigma = float(phase_sigma_cycles)
    if sigma > 0.0:
        phase_covariance = phase_covariance + sigma * sigma * (
            np.eye(path.size, dtype=float) + np.ones((path.size, path.size), dtype=float)
        )
    predictive = phase_covariance + jacobian @ np.asarray(
        fixed_path.spin_posterior_covariance,
        dtype=float,
    ) @ jacobian.T
    predictive = 0.5 * (predictive + predictive.T)
    predicted_path = relative + jacobian @ np.asarray(
        fixed_path.spin_posterior_mean,
        dtype=float,
    )
    try:
        precision = np.linalg.inv(predictive)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Integer transport predictive covariance is singular.") from exc

    candidates: dict[tuple[int, ...], float] = {}

    def add_candidate(proposed: Sequence[float]) -> None:
        integer_path = tuple(int(value) for value in proposed)
        innovation = np.asarray(integer_path, dtype=float) - predicted_path
        score = float(innovation @ precision @ innovation)
        previous = candidates.get(integer_path)
        if previous is None or score < previous:
            candidates[integer_path] = score

    def add_move(move: np.ndarray) -> None:
        add_candidate(path + move)

    add_candidate(path)
    add_candidate(np.rint(predicted_path))
    add_candidate(np.rint(relative))

    for index in range(path.size):
        for delta in (-1.0, 1.0):
            point_move = np.zeros(path.size, dtype=float)
            point_move[index] = delta
            add_move(point_move)
            suffix_move = np.zeros(path.size, dtype=float)
            suffix_move[index:] = delta
            add_move(suffix_move)

    ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
    limit = max(int(maximum_paths), 1)
    selected = [path for path, _score in ordered[:limit]]
    incumbent = tuple(int(value) for value in path)
    if incumbent not in selected:
        selected[-1] = incumbent
    return tuple(selected)


def _rank_lattice_transport_paths(
    *,
    projector: PintBinaryLatticeProjector,
    marginal: SpinIntegerMarginalEvaluation,
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    local_integer_offsets: Sequence[Sequence[int]],
    current_path: tuple[int, ...],
    component: ContinuousPriorComponent,
    phase_sigma_cycles: float,
    maximum_paths: int,
    additional_paths: Sequence[tuple[int, ...]] = (),
    screen_reference_states: bool = False,
) -> tuple[dict[tuple[int, ...], np.ndarray], int]:
    """Rank finite lattice proposals by exact objective inside physical support.

    The unbounded Gaussian defines the proposal distribution. Before exact
    evaluation, its mode is constrained in the posterior metric to preserve
    the strong spin/sky covariance that coordinate clipping would discard.
    This screens bounded proposals without integrating a truncated posterior.
    """
    proposals: list[tuple[tuple[int, ...], np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for mode in marginal.path_modes:
        path = tuple(int(value) for value in mode.integer_offsets)
        if path in seen or not _path_matches_prior_component(path, component):
            continue
        seen.add(path)
        try:
            mean = np.asarray(mode.spin_posterior_mean, dtype=float)
            target = reference + mean
            if np.any(target < lower) or np.any(target > upper):
                covariance = np.asarray(mode.spin_posterior_covariance, dtype=float)
                whitening = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(mean.size))
                bounded = lsq_linear(
                    whitening, whitening @ mean,
                    bounds=(lower - reference, upper - reference),
                    method="bvls", tol=1.0e-10,
                )
                if not bounded.success:
                    continue
                target = reference + bounded.x
            proposals.append((path, target))
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    incumbent_allowed = _path_matches_prior_component(current_path, component)
    if screen_reference_states and incumbent_allowed:
        proposals.append((current_path, reference.copy()))
    proposals.extend(
        (path, reference.copy()) for path in additional_paths
        if _path_matches_prior_component(path, component)
    )
    reference_validated = False

    def project(target: np.ndarray) -> np.ndarray:
        nonlocal reference_validated
        state = _project_to_support_segment(
            reference=reference, target=target, projector=projector,
            lower=lower, upper=upper, local_integer_offsets=local_integer_offsets,
            reference_profile_validated=reference_validated,
        )
        reference_validated = True
        return state

    return _screen_exact_path_proposals(
        proposals=proposals, reference=reference, maximum_paths=maximum_paths,
        project=project,
        reference_competes=screen_reference_states,
        reserved_path=current_path if incumbent_allowed and not screen_reference_states else None,
        score=lambda state, path: projector.evaluate_fixed_path(
            state, local_integer_offsets=local_integer_offsets,
            integer_offsets=path, phase_sigma_cycles=phase_sigma_cycles,
        ).total_chi2,
    )


def anneal_binary_conditional_mode(
    *,
    context: BinaryLatticeSearchContext,
    mode: BinaryConditionalMode,
    config: BinaryAnnealingConfig | None = None,
) -> tuple[BinaryAnnealedMode, ...]:
    active = config or BinaryAnnealingConfig()
    cell_by_id = {cell.cell_id: cell for cell in context.cells}
    if mode.cell_id not in cell_by_id:
        raise ValueError("The binary mode cell is outside the context tiling.")
    lower, upper = normalized_bounds_for_binary_cell(context, cell_by_id[mode.cell_id])
    projector = PintBinaryLatticeProjector(context)
    local_component = mode.local_integer_offsets
    particles = [
        _AnnealingParticle(
            normalized_values=np.asarray(mode.normalized_values, dtype=float).copy(),
            integer_offsets=mode.integer_offsets,
            objective=float(mode.strict_score),
            trace=tuple(),
        )
    ]
    component = _prior_component_by_id(
        context,
        mode.prior_component_id,
        reference=np.asarray(mode.normalized_values, dtype=float),
        default_sigma=active.coordinate_prior_sigma,
    )
    for phase_sigma in active.phase_sigma_schedule:
        candidates: list[_AnnealingParticle] = []
        for particle in particles:
            try:
                physical, profile = projector.profile_normalized(
                    particle.normalized_values,
                    local_integer_offsets=local_component,
                )
                conditional = BinaryConditionalProfile(
                    normalized_values=particle.normalized_values.copy(),
                    parameter_values=physical,
                    profile=profile,
                    function_evaluations=0,
                    converged=True,
                )
                linearization = projector.linearize_full(
                    conditional,
                    normalized_lower=lower,
                    normalized_upper=upper,
                    finite_difference_step=active.finite_difference_step,
                )
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
            prior_mean = component.normalized_mean - particle.normalized_values
            proposal_states: dict[tuple[int, ...], np.ndarray] = {}
            evaluated_path_count = 1
            proposed_path_count = 1
            path_proposal_scope = "fixed_path"
            if active.screen_lattice_paths or (
                active.integer_path_transport and active.integer_path_proposal == "lattice"
            ):
                path_proposal_scope = "finite_gaussian_lattice_exact_support_screen"
                try:
                    marginal = marginalize_spin_and_integer_paths(
                        linearization, spin_prior_mean=prior_mean,
                        spin_prior_covariance=component.normalized_covariance,
                        config=replace(active.lattice, added_session_phase_sigma_cycles=float(phase_sigma)),
                    )
                    proposal_states, evaluated_path_count = _rank_lattice_transport_paths(
                        projector=projector, marginal=marginal,
                        reference=particle.normalized_values, lower=lower, upper=upper,
                        local_integer_offsets=local_component,
                        current_path=particle.integer_offsets, component=component,
                        phase_sigma_cycles=float(phase_sigma), maximum_paths=active.paths_per_state,
                        additional_paths=(
                            (tuple(int(value) for value in np.rint(
                                np.asarray(profile.session_intercepts)[1:]
                                - float(profile.session_intercepts[0]),
                            )),) if active.screen_lattice_paths else ()
                        ),
                        screen_reference_states=active.screen_lattice_paths,
                    )
                    path_modes = list(proposal_states)
                    proposed_path_count = len(marginal.path_modes) + (
                        int(_path_matches_prior_component(particle.integer_offsets, component)) + 1
                        if active.screen_lattice_paths else 0
                    )
                    lattice_mass_converged = marginal.lattice_mass_converged
                    lattice_log_mass_increment = marginal.lattice_log_mass_increment
                    lattice_visited_nodes = marginal.lattice_visited_nodes
                except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                    path_modes = [particle.integer_offsets]
                    proposed_path_count = evaluated_path_count = 0
                    lattice_mass_converged = False
                    lattice_log_mass_increment = 0.0
                    lattice_visited_nodes = 0
            elif active.integer_path_transport:
                path_proposal_scope = "bounded_block_transport"
                try:
                    incumbent_path = marginalize_spin_for_fixed_integer_path(
                        linearization,
                        particle.integer_offsets,
                        spin_prior_mean=prior_mean,
                        spin_prior_covariance=component.normalized_covariance,
                        added_session_phase_sigma_cycles=float(phase_sigma),
                    )
                    proposed_paths = _propose_integer_path_transport_moves(
                        linearization=linearization,
                        current_path=particle.integer_offsets,
                        fixed_path=incumbent_path,
                        phase_sigma_cycles=float(phase_sigma),
                        maximum_paths=max(int(active.paths_per_state), 1),
                    )
                    path_modes = [
                        path
                        for path in proposed_paths
                        if _path_matches_prior_component(path, component)
                    ]
                except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                    path_modes = []
                if not path_modes and _path_matches_prior_component(
                    particle.integer_offsets,
                    component,
                ):
                    path_modes = [particle.integer_offsets]
                lattice_mass_converged = False
                lattice_log_mass_increment = 0.0
                lattice_visited_nodes = len(path_modes)
                proposed_path_count = evaluated_path_count = len(path_modes)
            else:
                path_modes = [None]
                lattice_mass_converged = True
                lattice_log_mass_increment = 0.0
                lattice_visited_nodes = 1

            for path_mode in path_modes:
                path = (
                    particle.integer_offsets
                    if path_mode is None
                    else path_mode
                )
                try:
                    fixed_path = marginalize_spin_for_fixed_integer_path(
                        linearization,
                        path,
                        spin_prior_mean=prior_mean,
                        spin_prior_covariance=component.normalized_covariance,
                        added_session_phase_sigma_cycles=float(phase_sigma),
                    )
                    proposal = proposal_states.get(path)
                    if proposal is None:
                        proposal = particle.normalized_values + np.asarray(
                            fixed_path.spin_posterior_mean, dtype=float
                        )
                    proposal = _project_to_support_segment(
                        reference=particle.normalized_values,
                        target=proposal,
                        projector=projector,
                        lower=lower,
                        upper=upper,
                        local_integer_offsets=local_component,
                    )
                    state, _, shape_chi2, phase_chi2, evaluations, converged = (
                        _close_softened_path(
                            projector=projector,
                            start=proposal,
                            lower=lower,
                            upper=upper,
                            local_integer_offsets=local_component,
                            integer_offsets=path,
                            phase_sigma_cycles=float(phase_sigma),
                            maximum_function_evaluations=(
                                active.maximum_closure_evaluations
                            ),
                        )
                    )
                except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                    continue
                objective = float(shape_chi2 + phase_chi2)
                log_likelihood = float(fixed_path.log_likelihood + component.log_weight)
                step = BinaryAnnealingStep(
                    phase_sigma_cycles=float(phase_sigma),
                    integer_offsets=path,
                    normalized_values=state.copy(),
                    shape_chi2=shape_chi2,
                    phase_chi2=phase_chi2,
                    total_chi2=objective,
                    lattice_strict_score=float(-2.0 * log_likelihood),
                    lattice_log_likelihood=log_likelihood,
                    lattice_mass_converged=lattice_mass_converged,
                    lattice_log_mass_increment=lattice_log_mass_increment,
                    lattice_visited_nodes=lattice_visited_nodes,
                    function_evaluations=evaluations,
                    converged=converged,
                    proposed_path_count=proposed_path_count,
                    evaluated_path_count=evaluated_path_count,
                    paths_selected_for_closure=len(path_modes),
                    path_proposal_scope=path_proposal_scope,
                )
                candidates.append(
                    _AnnealingParticle(
                        normalized_values=state,
                        integer_offsets=path,
                        objective=objective,
                        trace=(*particle.trace, step),
                    )
                )
        if not candidates:
            return tuple()
        particles = _select_path_diverse_particles(candidates, active.beam_size)

    results: list[BinaryAnnealedMode] = []
    for particle in particles:
        physical, profile = projector.profile_normalized(
            particle.normalized_values,
            local_integer_offsets=local_component,
        )
        final = particle.trace[-1]
        results.append(
            BinaryAnnealedMode(
                model_name=context.evaluator.model_name,
                cell_id=mode.cell_id,
                source_grid_index=mode.grid_index,
                normalized_values=particle.normalized_values.copy(),
                parameter_values=physical,
                local_integer_offsets=profile.local_integer_offsets,
                integer_offsets=particle.integer_offsets,
                shape_chi2=float(final.shape_chi2),
                phase_chi2=float(final.phase_chi2),
                total_chi2=float(final.total_chi2),
                converged=bool(final.converged),
                trace=particle.trace,
                prior_component_id=mode.prior_component_id,
                prior_log_weight=mode.prior_log_weight,
            )
        )
    results.sort(key=lambda item: (item.total_chi2, item.integer_offsets))
    return tuple(results)


# Public model-agnostic names. The binary names remain compatibility aliases.
PhaseLatticeSearchContext = BinaryLatticeSearchContext
PintPhaseLatticeProjector = PintBinaryLatticeProjector
PintBinaryPhaseEvaluator = PintPhaseEvaluator
search_phase_lattice_cell = search_binary_orbital_cycle_cell
anneal_phase_lattice_mode = anneal_binary_conditional_mode


__all__ = [
    "BinaryAnnealedMode",
    "BinaryAnnealingConfig",
    "BinaryAnnealingStep",
    "BinaryCellSearchConfig",
    "BinaryConditionalMode",
    "BinaryConditionalProfile",
    "BinaryFixedPathEvaluation",
    "ContinuousPriorComponent",
    "BinaryLatticeSearchContext",
    "IsolatedColdStartConfig",
    "PhaseLatticeSearchContext",
    "PintBinaryLatticeProjector",
    "PintBinaryPhaseEvaluator",
    "PintPhaseEvaluator",
    "PintPhaseLatticeProjector",
    "anneal_phase_lattice_mode",
    "anneal_binary_conditional_mode",
    "build_binary_lattice_search_context",
    "build_phase_lattice_search_context",
    "detect_phase_model_name",
    "normalized_bounds_for_binary_cell",
    "search_phase_lattice_cell",
    "search_binary_orbital_cycle_cell",
]
