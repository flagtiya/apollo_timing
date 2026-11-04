from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.bayesian_dual_lattice_enumerator import (
    OrbitalCycleCell,
    SourceSession,
    TwoLevelPulseProfile,
    _initialize_session_local_pulse_numbers,
    _session_shape_profile,
    infer_contiguous_source_sessions,
    profile_fixed_local_pulse_component,
    profile_two_level_pulse_lattice,
    tile_orbital_cycle_support,
)
from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinIntegerMarginalEvaluation,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
)


ELL1_CORE_PARAMETERS = ("F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2")
ELL1_FIT_PARAMETERS = (
    "RAJ",
    "DECJ",
    "F0",
    "F1",
    "PB",
    "A1",
    "TASC",
    "EPS1",
    "EPS2",
)
ELL1_SLOW_NAMES = ("C_ORB", "PHI_REF")
ELL1_FAST_NAMES = (
    "F0_REF",
    "F1",
    "F0_A1",
    "F0_A1_EPS1",
    "F0_A1_EPS2",
    "RAJ",
    "DECJ",
)


def _finite_vector(
    values: Sequence[float],
    *,
    dimension: int,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector of length {dimension}.")
    return result


def _wrap_radians(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class ELL1DualLatticeCoordinateSystem:
    """ELL1 coordinates exposing orbital-cycle and pulse-number aliases.

    The slow block contains only the nonlinear orbital frequency and phase.
    At fixed slow coordinates the remaining coordinates are close to linear
    in pulse phase, including the two astrometric coordinates from the input
    fit contract.
    """

    parameter_names: tuple[str, ...]
    observation_span_days: float
    reference_mjd: float
    pepoch_mjd: float
    initial_tasc_mjd: float

    def __post_init__(self) -> None:
        names = tuple(str(name).upper() for name in self.parameter_names)
        if set(names) != set(ELL1_FIT_PARAMETERS) or len(names) != 9:
            raise ValueError(
                "ELL1 dual-lattice coordinates require exactly: "
                + ", ".join(ELL1_FIT_PARAMETERS)
            )
        if not np.isfinite(self.observation_span_days) or self.observation_span_days <= 0:
            raise ValueError("The observation span must be finite and positive.")
        if not all(
            np.isfinite(value)
            for value in (self.reference_mjd, self.pepoch_mjd, self.initial_tasc_mjd)
        ):
            raise ValueError("ELL1 reference epochs must be finite.")

    def _index(self, name: str) -> int:
        return self.parameter_names.index(name)

    def split_physical(
        self,
        parameter_values: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        physical = _finite_vector(
            parameter_values,
            dimension=9,
            label="ELL1 physical parameters",
        )
        by_name = {
            name: float(physical[self._index(name)]) for name in self.parameter_names
        }
        if by_name["F0"] <= 0.0 or by_name["A1"] <= 0.0 or by_name["PB"] <= 0.0:
            raise ValueError("F0, A1, and PB must be positive.")
        spin_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        f0_reference = by_name["F0"] + by_name["F1"] * spin_seconds
        if f0_reference <= 0.0:
            raise ValueError("The reference-epoch spin frequency must be positive.")
        orbital_cycles = (self.reference_mjd - by_name["TASC"]) / by_name["PB"]
        phase_reference = _wrap_radians(2.0 * math.pi * orbital_cycles)
        phase_amplitude = f0_reference * by_name["A1"]
        slow = np.asarray(
            (self.observation_span_days / by_name["PB"], phase_reference),
            dtype=float,
        )
        fast = np.asarray(
            (
                f0_reference,
                by_name["F1"],
                phase_amplitude,
                phase_amplitude * by_name["EPS1"],
                phase_amplitude * by_name["EPS2"],
                by_name["RAJ"],
                by_name["DECJ"],
            ),
            dtype=float,
        )
        return slow, fast

    def combine_physical(
        self,
        slow_values: Sequence[float],
        fast_values: Sequence[float],
    ) -> np.ndarray:
        slow = _finite_vector(
            slow_values,
            dimension=2,
            label="ELL1 slow coordinates",
        )
        fast = _finite_vector(
            fast_values,
            dimension=7,
            label="ELL1 fast coordinates",
        )
        orbital_cycles, phase_reference = map(float, slow)
        (
            f0_reference,
            f1,
            phase_amplitude,
            eps1_amplitude,
            eps2_amplitude,
            raj_hours,
            decj_degrees,
        ) = map(float, fast)
        if orbital_cycles <= 0.0 or f0_reference <= 0.0 or phase_amplitude <= 0.0:
            raise ValueError("C_orb, F0_ref, and F0*A1 must be positive.")

        pb_days = self.observation_span_days / orbital_cycles
        fractional_cycle = _wrap_radians(phase_reference) / (2.0 * math.pi)
        initial_cycles = (self.reference_mjd - self.initial_tasc_mjd) / pb_days
        cycle_index = int(round(initial_cycles - fractional_cycle))
        tasc_mjd = self.reference_mjd - pb_days * (
            cycle_index + fractional_cycle
        )
        spin_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        f0 = f0_reference - f1 * spin_seconds
        a1 = phase_amplitude / f0_reference
        eps1 = eps1_amplitude / phase_amplitude
        eps2 = eps2_amplitude / phase_amplitude
        by_name = {
            "RAJ": raj_hours,
            "DECJ": decj_degrees,
            "F0": f0,
            "F1": f1,
            "PB": pb_days,
            "A1": a1,
            "TASC": tasc_mjd,
            "EPS1": eps1,
            "EPS2": eps2,
        }
        return np.asarray([by_name[name] for name in self.parameter_names], dtype=float)


@dataclass(frozen=True)
class ELL1DualLatticeSupport:
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    slow_lower: np.ndarray
    slow_upper: np.ndarray
    fast_lower: np.ndarray
    fast_upper: np.ndarray
    slow_center: np.ndarray
    fast_center: np.ndarray

    @property
    def slow_scale(self) -> np.ndarray:
        return 0.5 * (self.slow_upper - self.slow_lower)

    @property
    def fast_scale(self) -> np.ndarray:
        return 0.5 * (self.fast_upper - self.fast_lower)

    @property
    def slow_midpoint(self) -> np.ndarray:
        return 0.5 * (self.slow_lower + self.slow_upper)

    @property
    def fast_midpoint(self) -> np.ndarray:
        return 0.5 * (self.fast_lower + self.fast_upper)

    def normalize(
        self,
        slow_values: Sequence[float],
        fast_values: Sequence[float],
    ) -> np.ndarray:
        slow = _finite_vector(slow_values, dimension=2, label="ELL1 slow coordinates")
        fast = _finite_vector(fast_values, dimension=7, label="ELL1 fast coordinates")
        return np.concatenate(
            (
                (slow - self.slow_midpoint) / self.slow_scale,
                (fast - self.fast_midpoint) / self.fast_scale,
            )
        )

    def denormalize(
        self,
        normalized_values: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        normalized = _finite_vector(
            normalized_values,
            dimension=9,
            label="Normalized ELL1 dual-lattice coordinates",
        )
        slow = self.slow_midpoint + self.slow_scale * normalized[:2]
        fast = self.fast_midpoint + self.fast_scale * normalized[2:]
        return slow, fast

    def contains_physical(self, parameter_values: Sequence[float]) -> bool:
        physical = np.asarray(parameter_values, dtype=float)
        if physical.shape != self.physical_lower.shape or not np.all(np.isfinite(physical)):
            return False
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(
            np.maximum(np.abs(self.physical_lower), np.abs(self.physical_upper)),
            1.0,
        )
        return bool(
            np.all(physical >= self.physical_lower - tolerance)
            and np.all(physical <= self.physical_upper + tolerance)
        )


def build_generic_ell1_physical_support(
    *,
    parameter_names: Sequence[str],
    center_values: Sequence[float],
    observation_span_days: float,
    width_scale: float = 1.0,
    minimum_f1_half_width: float = 1.0e-10,
    minimum_eccentricity_half_width: float = 5.0e-3,
    astrometric_bound_arcsec: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a source-class-independent ELL1 recoverability support."""

    names = tuple(str(name).upper() for name in parameter_names)
    if set(names) != set(ELL1_FIT_PARAMETERS) or len(names) != 9:
        raise ValueError("Generic ELL1 support requires the input nine-parameter fit.")
    center = _finite_vector(center_values, dimension=9, label="ELL1 support center")
    span_days = float(observation_span_days)
    scale = float(width_scale)
    f1_floor = float(minimum_f1_half_width)
    eccentricity_floor = float(minimum_eccentricity_half_width)
    sky_bound = float(astrometric_bound_arcsec)
    if not all(
        np.isfinite(value) and value > 0.0
        for value in (span_days, scale, f1_floor, eccentricity_floor, sky_bound)
    ):
        raise ValueError("ELL1 support scales must be finite and positive.")
    index = {name: names.index(name) for name in names}
    pb = float(center[index["PB"]])
    widths = {
        "RAJ": sky_bound / 15.0 / 3600.0,
        "DECJ": sky_bound / 3600.0,
        "F0": max(abs(float(center[index["F0"]])) * 1.0e-5, 5.0 / (span_days * 86400.0)),
        "F1": max(abs(float(center[index["F1"]])) * 20.0, f1_floor),
        "PB": max(abs(pb) * 0.03, 5.0e-4),
        "A1": max(abs(float(center[index["A1"]])) * 0.08, 0.1),
        "TASC": 0.55 * pb,
        "EPS1": max(abs(float(center[index["EPS1"]])) * 5.0, 5.0e-3),
        "EPS2": max(abs(float(center[index["EPS2"]])) * 5.0, 5.0e-3),
    }
    lower = center.copy()
    upper = center.copy()
    for name, width in widths.items():
        parameter_index = index[name]
        half_width = scale * width
        if name in {"EPS1", "EPS2"}:
            half_width = max(half_width, eccentricity_floor)
        lower[parameter_index] -= half_width
        upper[parameter_index] += half_width
    lower[index["F0"]] = max(lower[index["F0"]], np.finfo(float).tiny)
    lower[index["F1"]] = min(lower[index["F1"]], -f1_floor)
    upper[index["F1"]] = max(upper[index["F1"]], f1_floor)
    lower[index["PB"]] = max(lower[index["PB"]], 1.0e-6)
    lower[index["A1"]] = max(lower[index["A1"]], 1.0e-6)
    lower[index["DECJ"]] = max(lower[index["DECJ"]], -90.0)
    upper[index["DECJ"]] = min(upper[index["DECJ"]], 90.0)
    return lower, upper


def build_ell1_dual_lattice_support(
    *,
    coordinates: ELL1DualLatticeCoordinateSystem,
    center_values: Sequence[float],
    physical_lower: Sequence[float],
    physical_upper: Sequence[float],
) -> ELL1DualLatticeSupport:
    center = _finite_vector(center_values, dimension=9, label="ELL1 support center")
    lower = _finite_vector(physical_lower, dimension=9, label="ELL1 lower support")
    upper = _finite_vector(physical_upper, dimension=9, label="ELL1 upper support")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Every ELL1 support interval must contain its center.")
    index = {name: coordinates.parameter_names.index(name) for name in coordinates.parameter_names}
    slow_center, fast_center = coordinates.split_physical(center)
    slow_lower = np.asarray(
        (
            coordinates.observation_span_days / upper[index["PB"]],
            -math.pi,
        ),
        dtype=float,
    )
    slow_upper = np.asarray(
        (
            coordinates.observation_span_days / lower[index["PB"]],
            math.pi,
        ),
        dtype=float,
    )
    spin_seconds = (coordinates.reference_mjd - coordinates.pepoch_mjd) * 86400.0
    f0_reference_corners = np.asarray(
        [
            f0 + f1 * spin_seconds
            for f0 in (lower[index["F0"]], upper[index["F0"]])
            for f1 in (lower[index["F1"]], upper[index["F1"]])
        ],
        dtype=float,
    )
    f0_reference_lower = max(float(np.min(f0_reference_corners)), np.finfo(float).tiny)
    f0_reference_upper = float(np.max(f0_reference_corners))
    amplitude_lower = f0_reference_lower * float(lower[index["A1"]])
    amplitude_upper = f0_reference_upper * float(upper[index["A1"]])
    eps1_limit = max(abs(float(lower[index["EPS1"]])), abs(float(upper[index["EPS1"]])))
    eps2_limit = max(abs(float(lower[index["EPS2"]])), abs(float(upper[index["EPS2"]])))
    fast_lower = np.asarray(
        (
            f0_reference_lower,
            lower[index["F1"]],
            amplitude_lower,
            -amplitude_upper * eps1_limit,
            -amplitude_upper * eps2_limit,
            lower[index["RAJ"]],
            lower[index["DECJ"]],
        ),
        dtype=float,
    )
    fast_upper = np.asarray(
        (
            f0_reference_upper,
            upper[index["F1"]],
            amplitude_upper,
            amplitude_upper * eps1_limit,
            amplitude_upper * eps2_limit,
            upper[index["RAJ"]],
            upper[index["DECJ"]],
        ),
        dtype=float,
    )
    return ELL1DualLatticeSupport(
        physical_lower=lower.copy(),
        physical_upper=upper.copy(),
        slow_lower=slow_lower,
        slow_upper=slow_upper,
        fast_lower=fast_lower,
        fast_upper=fast_upper,
        slow_center=slow_center,
        fast_center=fast_center,
    )


class PintELL1PhaseEvaluator:
    """Mutable exact-PINT evaluator for the nine-parameter ELL1 contract."""

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
        if set(names) != set(ELL1_FIT_PARAMETERS) or len(names) != 9:
            raise ValueError("PINT ELL1 evaluation requires the input nine parameters.")
        missing = [name for name in names if name not in model.params]
        if missing:
            raise ValueError("PINT model is missing: " + ", ".join(missing))
        pulse_numbers = np.asarray(base_pulse_numbers, dtype=np.longdouble)
        if pulse_numbers.shape != (len(toas.table),) or not np.all(np.isfinite(pulse_numbers)):
            raise ValueError("Base pulse numbers must align with every TOA.")
        self.model = model
        self.toas = toas
        self.parameter_names = names
        self.sessions = tuple(sessions)
        self.base_pulse_numbers = pulse_numbers
        self.base_parameter_values = np.asarray(
            [float(model[name].value) for name in names],
            dtype=float,
        )

    @classmethod
    def from_files(
        cls,
        *,
        par_path: Path,
        tim_path: Path,
        clock_override_path: Path | None = None,
    ) -> "PintELL1PhaseEvaluator":
        model = _load_pint_model(Path(par_path))
        if "BINARY" not in model.params or str(model["BINARY"].value).upper() != "ELL1":
            raise ValueError("The ELL1 dual-lattice solver requires BINARY ELL1.")
        free = tuple(str(name).upper() for name in model.free_params)
        if set(free) != set(ELL1_FIT_PARAMETERS) or len(free) != 9:
            raise ValueError(
                "The input ELL1 par must fit exactly: "
                + ", ".join(ELL1_FIT_PARAMETERS)
            )
        toas = _load_pint_toas(
            Path(tim_path),
            seed_model=model,
            clock_override_path=clock_override_path,
        )
        sessions = infer_contiguous_source_sessions(toas)
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

    def phase_sigmas(self) -> np.ndarray:
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

    def _set_parameters(self, parameter_values: Sequence[float]) -> None:
        values = _finite_vector(
            parameter_values,
            dimension=9,
            label="PINT ELL1 parameters",
        )
        by_name = dict(zip(self.parameter_names, values, strict=True))
        if by_name["F0"] <= 0.0 or by_name["PB"] <= 0.0 or by_name["A1"] <= 0.0:
            raise ValueError("PINT ELL1 parameters lie outside their physical domain.")
        for name, value in by_name.items():
            _set_numeric_model_parameter_value(self.model, name, float(value))

    def raw_phase_residuals(self, parameter_values: Sequence[float]) -> np.ndarray:
        self._set_parameters(parameter_values)
        phase = self.model.phase(self.toas, abs_phase=True)
        phase_values = np.asarray(phase.int.value, dtype=np.longdouble)
        phase_values += np.asarray(phase.frac.value, dtype=np.longdouble)
        return np.asarray(phase_values - self.base_pulse_numbers, dtype=float)

    def evaluate_values(
        self,
        parameter_values: Sequence[float],
        *,
        local_integer_radius: int = 0,
        maximum_retained_local_modes: int | None = None,
    ) -> TwoLevelPulseProfile:
        return profile_two_level_pulse_lattice(
            self.raw_phase_residuals(parameter_values),
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
            else tuple(
                tuple(int(value) for value in offsets)
                for offsets in local_integer_offsets
            )
        )
        return profile_fixed_local_pulse_component(
            self.raw_phase_residuals(parameter_values),
            self.phase_sigmas(),
            self.sessions,
            component,
        )


@dataclass(frozen=True)
class ELL1DualLatticeSearchContext:
    evaluator: PintELL1PhaseEvaluator
    coordinates: ELL1DualLatticeCoordinateSystem
    support: ELL1DualLatticeSupport
    cells: tuple[OrbitalCycleCell, ...]


def build_ell1_dual_lattice_search_context(
    *,
    par_path: Path,
    tim_path: Path,
    maximum_cell_width: float = 0.25,
    support_width_scale: float = 1.0,
    minimum_f1_half_width: float = 1.0e-10,
    minimum_eccentricity_half_width: float = 5.0e-3,
    astrometric_bound_arcsec: float = 300.0,
    clock_override_path: Path | None = None,
) -> ELL1DualLatticeSearchContext:
    evaluator = PintELL1PhaseEvaluator.from_files(
        par_path=par_path,
        tim_path=tim_path,
        clock_override_path=clock_override_path,
    )
    epochs = evaluator.observation_epochs_mjd
    span_days = float(np.ptp(epochs))
    if not np.isfinite(span_days) or span_days <= 0.0:
        raise ValueError("ELL1 dual-lattice search requires distinct TOA epochs.")
    reference_mjd = float(np.median(epochs))
    model = evaluator.model
    coordinates = ELL1DualLatticeCoordinateSystem(
        parameter_names=evaluator.parameter_names,
        observation_span_days=span_days,
        reference_mjd=reference_mjd,
        pepoch_mjd=float(model["PEPOCH"].value),
        initial_tasc_mjd=float(model["TASC"].value),
    )
    lower, upper = build_generic_ell1_physical_support(
        parameter_names=evaluator.parameter_names,
        center_values=evaluator.base_parameter_values,
        observation_span_days=span_days,
        width_scale=support_width_scale,
        minimum_f1_half_width=minimum_f1_half_width,
        minimum_eccentricity_half_width=minimum_eccentricity_half_width,
        astrometric_bound_arcsec=astrometric_bound_arcsec,
    )
    support = build_ell1_dual_lattice_support(
        coordinates=coordinates,
        center_values=evaluator.base_parameter_values,
        physical_lower=lower,
        physical_upper=upper,
    )
    pb_index = evaluator.parameter_names.index("PB")
    cells = tile_orbital_cycle_support(
        pb_lower_days=float(lower[pb_index]),
        pb_upper_days=float(upper[pb_index]),
        observation_span_days=span_days,
        maximum_cell_width=maximum_cell_width,
    )
    return ELL1DualLatticeSearchContext(
        evaluator=evaluator,
        coordinates=coordinates,
        support=support,
        cells=cells,
    )


def normalized_bounds_for_ell1_cell(
    support: ELL1DualLatticeSupport,
    cell: OrbitalCycleCell,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(9, -1.0, dtype=float)
    upper = np.full(9, 1.0, dtype=float)
    lower[0] = (cell.lower - support.slow_midpoint[0]) / support.slow_scale[0]
    upper[0] = (cell.upper - support.slow_midpoint[0]) / support.slow_scale[0]
    epsilon = 32.0 * np.finfo(float).eps
    return lower + epsilon, upper - epsilon


@dataclass(frozen=True)
class ELL1ConditionalProfile:
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    profile: TwoLevelPulseProfile
    function_evaluations: int
    converged: bool


@dataclass(frozen=True)
class ELL1ConditionalMode:
    cell_id: str
    grid_index: tuple[int, int]
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    strict_score: float
    log_likelihood: float
    integer_offsets: tuple[int, ...]
    posterior_weight: float
    lattice_mass_converged: bool
    lattice_log_mass_increment: float
    lattice_visited_nodes: int
    posterior_in_support: bool
    source_cell_id: str | None = None
    posterior_transport_fraction: float = 1.0


@dataclass(frozen=True)
class ELL1PosteriorCellLocation:
    cell: OrbitalCycleCell
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    transport_fraction: float = 1.0


@dataclass(frozen=True)
class ELL1CellSearchConfig:
    orbital_samples_per_cell: int = 2
    phase_samples: int = 12
    modes_per_cell: int = 4
    fast_shape_maximum_evaluations: int = 60
    finite_difference_step: float = 2.0e-5
    fast_prior_sigma: float = 0.75
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
class ELL1AnnealingConfig:
    phase_sigma_schedule: tuple[float, ...] = (
        0.10,
        0.05,
        0.02,
        0.01,
        0.005,
        0.002,
        0.0,
    )
    coordinate_prior_sigma: float = 0.20
    paths_per_state: int = 3
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
        schedule = tuple(float(value) for value in self.phase_sigma_schedule)
        if not schedule or any(not np.isfinite(value) or value < 0.0 for value in schedule):
            raise ValueError("The ELL1 phase-noise schedule must be finite and nonnegative.")
        if any(right > left for left, right in zip(schedule, schedule[1:])):
            raise ValueError("The ELL1 phase-noise schedule must be nonincreasing.")
        if schedule[-1] != 0.0:
            raise ValueError("The ELL1 phase-noise schedule must end at zero.")
        positive = (
            self.coordinate_prior_sigma,
            self.finite_difference_step,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("ELL1 annealing scales must be finite and positive.")
        if any(
            int(value) <= 0
            for value in (
                self.paths_per_state,
                self.beam_size,
                self.maximum_closure_evaluations,
            )
        ):
            raise ValueError("ELL1 annealing budgets must be positive integers.")


@dataclass(frozen=True)
class ELL1AnnealingStep:
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


@dataclass(frozen=True)
class ELL1AnnealedMode:
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
    trace: tuple[ELL1AnnealingStep, ...]
    source_cell_id: str | None = None


@dataclass(frozen=True)
class ELL1AnnealingStageDiagnostics:
    phase_sigma_cycles: float
    input_particle_count: int
    profile_lattice_failure_count: int
    empty_path_posterior_count: int
    attempted_path_count: int
    closure_failure_count: int
    closure_nonconverged_count: int
    nonfinite_objective_count: int
    candidate_count: int
    retained_particle_count: int
    failure_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ELL1AnnealingResult:
    modes: tuple[ELL1AnnealedMode, ...]
    stages: tuple[ELL1AnnealingStageDiagnostics, ...]
    completed_schedule: bool
    failure_stage_index: int | None = None
    failure_reason: str | None = None


class PintELL1DualLatticeProjector:
    """Exact PINT phase statistics in normalized ELL1 dual coordinates."""

    def __init__(self, context: ELL1DualLatticeSearchContext) -> None:
        self.context = context
        self.evaluator = context.evaluator
        self.coordinates = context.coordinates
        self.support = context.support
        self.sessions = tuple(context.evaluator.sessions)
        self.phase_sigmas = context.evaluator.phase_sigmas()

    def profile_normalized(
        self,
        normalized_values: Sequence[float],
        *,
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
    ) -> tuple[np.ndarray, TwoLevelPulseProfile]:
        normalized = _finite_vector(
            normalized_values,
            dimension=9,
            label="Normalized ELL1 state",
        )
        if np.any(normalized < -1.0) or np.any(normalized > 1.0):
            raise ValueError("Normalized ELL1 state lies outside support.")
        slow, fast = self.support.denormalize(normalized)
        physical = self.coordinates.combine_physical(slow, fast)
        if not self.support.contains_physical(physical):
            raise ValueError("ELL1 state lies outside physical support.")
        profile = self.evaluator.evaluate_fixed_local_component(
            physical,
            local_integer_offsets,
        )
        return physical, profile

    def profile_fast_shape(
        self,
        slow_normalized_values: Sequence[float],
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        fast_start_normalized_values: Sequence[float],
        local_integer_offsets: Sequence[Sequence[int]],
        maximum_function_evaluations: int,
    ) -> ELL1ConditionalProfile:
        slow = _finite_vector(
            slow_normalized_values,
            dimension=2,
            label="Normalized ELL1 slow state",
        )
        lower = _finite_vector(normalized_lower, dimension=9, label="ELL1 lower bounds")
        upper = _finite_vector(normalized_upper, dimension=9, label="ELL1 upper bounds")
        fast_start = _finite_vector(
            fast_start_normalized_values,
            dimension=7,
            label="Normalized ELL1 fast start",
        )
        if np.any(lower >= upper):
            raise ValueError("ELL1 normalized bounds must have positive width.")
        if np.any(slow < lower[:2]) or np.any(slow > upper[:2]):
            raise ValueError("ELL1 slow state lies outside its cell.")
        fast_start = np.clip(fast_start, lower[2:], upper[2:])
        failure = np.full(len(self.evaluator.toas.table), 1.0e10, dtype=float)

        def residual(fast: np.ndarray) -> np.ndarray:
            state = np.concatenate((slow, fast))
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
            fast_start,
            bounds=(lower[2:], upper[2:]),
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=max(int(maximum_function_evaluations), 8),
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
        )
        state = np.concatenate((slow, np.asarray(fit.x, dtype=float)))
        physical, profile = self.profile_normalized(
            state,
            local_integer_offsets=local_integer_offsets,
        )
        return ELL1ConditionalProfile(
            normalized_values=state,
            parameter_values=physical,
            profile=profile,
            function_evaluations=int(fit.nfev),
            converged=bool(fit.success),
        )

    def profile_full_shape(
        self,
        normalized_start: Sequence[float],
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        local_integer_offsets: Sequence[Sequence[int]],
        maximum_function_evaluations: int,
    ) -> ELL1ConditionalProfile:
        """Use a grid point only as a start for a complete cell-local chart."""

        start = _finite_vector(
            normalized_start,
            dimension=9,
            label="Normalized ELL1 full start",
        )
        lower = _finite_vector(normalized_lower, dimension=9, label="ELL1 lower bounds")
        upper = _finite_vector(normalized_upper, dimension=9, label="ELL1 upper bounds")
        if np.any(lower >= upper):
            raise ValueError("ELL1 normalized bounds must have positive width.")
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
        return ELL1ConditionalProfile(
            normalized_values=state,
            parameter_values=physical,
            profile=profile,
            function_evaluations=int(fit.nfev),
            converged=bool(fit.success),
        )

    def linearize_full(
        self,
        conditional: ELL1ConditionalProfile,
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        finite_difference_step: float,
    ) -> SpinOrbitLinearization:
        reference = np.asarray(conditional.normalized_values, dtype=float)
        lower = _finite_vector(normalized_lower, dimension=9, label="ELL1 lower bounds")
        upper = _finite_vector(normalized_upper, dimension=9, label="ELL1 upper bounds")
        if reference.shape != (9,) or np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("ELL1 full reference lies outside its cell.")
        profile = conditional.profile
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        offsets = np.asarray(profile.session_intercepts, dtype=float)
        shape_jacobian = np.empty((shape.size, 9), dtype=float)
        offset_jacobian = np.empty((offsets.size, 9), dtype=float)
        requested_step = max(float(finite_difference_step), 1.0e-7)

        for index in range(9):
            plus_step = min(requested_step, float(upper[index] - reference[index]))
            minus_step = min(requested_step, float(reference[index] - lower[index]))
            plus_profile = None
            minus_profile = None
            if plus_step > 0.0:
                plus = reference.copy()
                plus[index] += plus_step
                try:
                    _, plus_profile = self.profile_normalized(
                        plus,
                        local_integer_offsets=profile.local_integer_offsets,
                    )
                except (ArithmeticError, RuntimeError, ValueError):
                    plus_profile = None
            if minus_step > 0.0:
                minus = reference.copy()
                minus[index] -= minus_step
                try:
                    _, minus_profile = self.profile_normalized(
                        minus,
                        local_integer_offsets=profile.local_integer_offsets,
                    )
                except (ArithmeticError, RuntimeError, ValueError):
                    minus_profile = None
            if plus_profile is not None and minus_profile is not None:
                denominator = plus_step + minus_step
                shape_delta = (
                    np.asarray(plus_profile.normalized_local_residuals, dtype=float)
                    - np.asarray(minus_profile.normalized_local_residuals, dtype=float)
                )
                offset_delta = (
                    np.asarray(plus_profile.session_intercepts, dtype=float)
                    - np.asarray(minus_profile.session_intercepts, dtype=float)
                )
            elif plus_profile is not None:
                denominator = plus_step
                shape_delta = np.asarray(plus_profile.normalized_local_residuals) - shape
                offset_delta = np.asarray(plus_profile.session_intercepts) - offsets
            elif minus_profile is not None:
                denominator = minus_step
                shape_delta = shape - np.asarray(minus_profile.normalized_local_residuals)
                offset_delta = offsets - np.asarray(minus_profile.session_intercepts)
            else:
                raise ValueError(f"No physical ELL1 full probe exists for coordinate {index}.")
            shape_jacobian[:, index] = shape_delta / denominator
            offset_jacobian[:, index] = offset_delta / denominator

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

    def linearize_fast(
        self,
        conditional: ELL1ConditionalProfile,
        *,
        normalized_lower: Sequence[float],
        normalized_upper: Sequence[float],
        finite_difference_step: float,
    ) -> SpinOrbitLinearization:
        reference = np.asarray(conditional.normalized_values, dtype=float)
        lower = _finite_vector(normalized_lower, dimension=9, label="ELL1 lower bounds")
        upper = _finite_vector(normalized_upper, dimension=9, label="ELL1 upper bounds")
        if reference.shape != (9,) or np.any(reference < lower) or np.any(reference > upper):
            raise ValueError("ELL1 conditional reference lies outside its cell.")
        profile = conditional.profile
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        offsets = np.asarray(profile.session_intercepts, dtype=float)
        shape_jacobian = np.empty((shape.size, 7), dtype=float)
        offset_jacobian = np.empty((offsets.size, 7), dtype=float)
        requested_step = max(float(finite_difference_step), 1.0e-7)

        for fast_column in range(7):
            index = fast_column + 2
            plus_step = min(requested_step, float(upper[index] - reference[index]))
            minus_step = min(requested_step, float(reference[index] - lower[index]))
            plus_profile = None
            minus_profile = None
            if plus_step > 0.0:
                plus = reference.copy()
                plus[index] += plus_step
                try:
                    _, plus_profile = self.profile_normalized(
                        plus,
                        local_integer_offsets=profile.local_integer_offsets,
                    )
                except (ArithmeticError, RuntimeError, ValueError):
                    plus_profile = None
            if minus_step > 0.0:
                minus = reference.copy()
                minus[index] -= minus_step
                try:
                    _, minus_profile = self.profile_normalized(
                        minus,
                        local_integer_offsets=profile.local_integer_offsets,
                    )
                except (ArithmeticError, RuntimeError, ValueError):
                    minus_profile = None
            if plus_profile is not None and minus_profile is not None:
                denominator = plus_step + minus_step
                shape_delta = (
                    np.asarray(plus_profile.normalized_local_residuals, dtype=float)
                    - np.asarray(minus_profile.normalized_local_residuals, dtype=float)
                )
                offset_delta = (
                    np.asarray(plus_profile.session_intercepts, dtype=float)
                    - np.asarray(minus_profile.session_intercepts, dtype=float)
                )
            elif plus_profile is not None:
                denominator = plus_step
                shape_delta = np.asarray(plus_profile.normalized_local_residuals) - shape
                offset_delta = np.asarray(plus_profile.session_intercepts) - offsets
            elif minus_profile is not None:
                denominator = minus_step
                shape_delta = shape - np.asarray(minus_profile.normalized_local_residuals)
                offset_delta = offsets - np.asarray(minus_profile.session_intercepts)
            else:
                raise ValueError(f"No physical ELL1 fast probe exists for column {fast_column}.")
            shape_jacobian[:, fast_column] = shape_delta / denominator
            offset_jacobian[:, fast_column] = offset_delta / denominator

        relative = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        variances = np.asarray(profile.session_intercept_variances, dtype=float)
        relative_covariance = np.diag(variances[1:]) + float(variances[0])
        return SpinOrbitLinearization(
            profile=_session_shape_profile(profile),
            spin_reference=reference[2:].copy(),
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
        )


def _local_component_at_state(
    projector: PintELL1DualLatticeProjector,
    state: np.ndarray,
    *,
    radius: int,
) -> tuple[tuple[int, ...], ...]:
    slow, fast = projector.support.denormalize(state)
    physical = projector.coordinates.combine_physical(slow, fast)
    profile = projector.evaluator.evaluate_values(
        physical,
        local_integer_radius=max(int(radius), 0),
        maximum_retained_local_modes=1,
    )
    return profile.local_integer_offsets


def locate_ell1_posterior_cell(
    *,
    context: ELL1DualLatticeSearchContext,
    normalized_values: Sequence[float],
) -> ELL1PosteriorCellLocation | None:
    """Locate a posterior mode in the global support and orbital tiling.

    Orbital cells partition the search integral; they are not posterior prior
    boundaries. A linearized update may therefore move a mode into an adjacent
    cell while remaining inside the declared physical support.
    """

    normalized = _finite_vector(
        normalized_values,
        dimension=9,
        label="Normalized ELL1 posterior coordinates",
    )
    normalized_tolerance = 128.0 * np.finfo(float).eps
    if np.any(normalized < -1.0 - normalized_tolerance) or np.any(
        normalized > 1.0 + normalized_tolerance
    ):
        return None
    normalized = np.clip(normalized, -1.0, 1.0)
    slow, fast = context.support.denormalize(normalized)
    physical = context.coordinates.combine_physical(slow, fast)
    if not context.support.contains_physical(physical):
        return None
    cells = tuple(context.cells)
    if not cells:
        return None
    orbital_cycles = float(slow[0])
    scale = max(abs(float(cells[0].lower)), abs(float(cells[-1].upper)), 1.0)
    cell_tolerance = 128.0 * np.finfo(float).eps * scale
    if (
        orbital_cycles < float(cells[0].lower) - cell_tolerance
        or orbital_cycles > float(cells[-1].upper) + cell_tolerance
    ):
        return None
    orbital_cycles = float(
        np.clip(orbital_cycles, cells[0].lower, cells[-1].upper)
    )
    upper_edges = np.asarray([cell.upper for cell in cells], dtype=float)
    destination_index = int(np.searchsorted(upper_edges, orbital_cycles, side="right"))
    destination_index = min(destination_index, len(cells) - 1)
    destination = cells[destination_index]
    if (
        orbital_cycles < destination.lower - cell_tolerance
        or orbital_cycles > destination.upper + cell_tolerance
    ):
        return None
    return ELL1PosteriorCellLocation(
        cell=destination,
        normalized_values=normalized,
        parameter_values=np.asarray(physical, dtype=float),
    )


def transport_ell1_posterior_to_support(
    *,
    context: ELL1DualLatticeSearchContext,
    reference_values: Sequence[float],
    posterior_values: Sequence[float],
) -> ELL1PosteriorCellLocation | None:
    """Transport a local Gaussian update through the nonlinear support map.

    The conditional reference is an exact, physically valid PINT state. The
    Gaussian update is only a local proposal, so when its endpoint leaves the
    nonlinear physical support this routine retains the largest valid segment
    of that update. Exact posterior annealing then relinearizes from that state.
    """

    reference = _finite_vector(
        reference_values,
        dimension=9,
        label="Normalized ELL1 transport reference",
    )
    posterior = _finite_vector(
        posterior_values,
        dimension=9,
        label="Normalized ELL1 transport posterior",
    )
    reference_location = locate_ell1_posterior_cell(
        context=context,
        normalized_values=reference,
    )
    if reference_location is None:
        return None
    posterior_location = locate_ell1_posterior_cell(
        context=context,
        normalized_values=posterior,
    )
    if posterior_location is not None:
        return posterior_location

    valid_fraction = 0.0
    invalid_fraction = 1.0
    valid_location = reference_location
    for _ in range(64):
        fraction = 0.5 * (valid_fraction + invalid_fraction)
        state = reference + fraction * (posterior - reference)
        location = locate_ell1_posterior_cell(
            context=context,
            normalized_values=state,
        )
        if location is None:
            invalid_fraction = fraction
        else:
            valid_fraction = fraction
            valid_location = location
    retained_fraction = max(valid_fraction - 1.0e-8, 0.0)
    retained_state = reference + retained_fraction * (posterior - reference)
    retained_location = locate_ell1_posterior_cell(
        context=context,
        normalized_values=retained_state,
    )
    if retained_location is None:
        retained_location = valid_location
        retained_fraction = valid_fraction
    return ELL1PosteriorCellLocation(
        cell=retained_location.cell,
        normalized_values=retained_location.normalized_values,
        parameter_values=retained_location.parameter_values,
        transport_fraction=float(retained_fraction),
    )


def ell1_physical_prior_covariance(
    *,
    context: ELL1DualLatticeSearchContext,
    normalized_reference: Sequence[float],
    prior_sigma: float,
) -> np.ndarray:
    """Map the independent physical prior into the efficient ELL1 chart.

    The chart deliberately couples physical parameters, for example F0_REF
    with F1 and orbital-cycle count with reference orbital phase. Treating its
    coordinates as independent creates posterior mass outside the declared
    physical prior and can move the integer lattice toward impossible paths.
    """

    reference = _finite_vector(
        normalized_reference,
        dimension=9,
        label="Normalized ELL1 prior reference",
    )
    sigma = float(prior_sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("The ELL1 physical-prior sigma must be positive.")
    slow, fast = context.support.denormalize(reference)
    physical = context.coordinates.combine_physical(slow, fast)
    if not context.support.contains_physical(physical):
        raise ValueError("The ELL1 prior reference lies outside physical support.")
    widths = 0.5 * (
        context.support.physical_upper - context.support.physical_lower
    )
    jacobian = np.empty((9, 9), dtype=float)

    def transformed(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed_slow, transformed_fast = context.coordinates.split_physical(values)
        return (
            context.support.normalize(transformed_slow, transformed_fast),
            transformed_slow,
        )

    reference_normalized, reference_slow = transformed(physical)

    def coordinate_delta(
        left_normalized: np.ndarray,
        left_slow: np.ndarray,
        right_normalized: np.ndarray,
        right_slow: np.ndarray,
    ) -> np.ndarray:
        delta = left_normalized - right_normalized
        delta[1] = _wrap_radians(float(left_slow[1] - right_slow[1])) / float(
            context.support.slow_scale[1]
        )
        return delta

    square_root_epsilon = math.sqrt(np.finfo(float).eps)
    for parameter_index, width in enumerate(widths):
        target_step = max(
            float(width) * 1.0e-5,
            square_root_epsilon * max(abs(float(physical[parameter_index])), 1.0),
        )
        plus_step = min(
            target_step,
            float(
                context.support.physical_upper[parameter_index]
                - physical[parameter_index]
            ),
        )
        minus_step = min(
            target_step,
            float(
                physical[parameter_index]
                - context.support.physical_lower[parameter_index]
            ),
        )
        if plus_step > 0.0 and minus_step > 0.0:
            plus = physical.copy()
            minus = physical.copy()
            plus[parameter_index] += plus_step
            minus[parameter_index] -= minus_step
            plus_normalized, plus_slow = transformed(plus)
            minus_normalized, minus_slow = transformed(minus)
            difference = coordinate_delta(
                plus_normalized,
                plus_slow,
                minus_normalized,
                minus_slow,
            )
            denominator = plus_step + minus_step
        elif plus_step > 0.0:
            plus = physical.copy()
            plus[parameter_index] += plus_step
            plus_normalized, plus_slow = transformed(plus)
            difference = coordinate_delta(
                plus_normalized,
                plus_slow,
                reference_normalized,
                reference_slow,
            )
            denominator = plus_step
        elif minus_step > 0.0:
            minus = physical.copy()
            minus[parameter_index] -= minus_step
            minus_normalized, minus_slow = transformed(minus)
            difference = coordinate_delta(
                reference_normalized,
                reference_slow,
                minus_normalized,
                minus_slow,
            )
            denominator = minus_step
        else:
            raise ValueError("The ELL1 physical prior has a zero-width parameter.")
        jacobian[:, parameter_index] = difference / denominator

    physical_standard_deviations = sigma * widths
    covariance = jacobian @ np.diag(
        np.square(physical_standard_deviations)
    ) @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalue_floor = max(float(np.max(eigenvalues)) * 1.0e-12, 1.0e-12)
    covariance = (
        eigenvectors * np.maximum(eigenvalues, eigenvalue_floor)
    ) @ eigenvectors.T
    return 0.5 * (covariance + covariance.T)


def search_ell1_orbital_cycle_cell(
    *,
    context: ELL1DualLatticeSearchContext,
    cell: OrbitalCycleCell,
    config: ELL1CellSearchConfig | None = None,
) -> tuple[ELL1ConditionalMode, ...]:
    """Evaluate a complete all-session posterior portfolio in one ELL1 cell."""

    active = config or ELL1CellSearchConfig()
    projector = PintELL1DualLatticeProjector(context)
    lower, upper = normalized_bounds_for_ell1_cell(context.support, cell)
    initial = context.support.normalize(
        context.support.slow_center,
        context.support.fast_center,
    )
    orbital_count = max(int(active.orbital_samples_per_cell), 1)
    phase_count = max(int(active.phase_samples), 2)
    orbital_values = np.linspace(
        lower[0],
        upper[0],
        orbital_count + 2,
        dtype=float,
    )[1:-1]
    phase_values = np.linspace(
        lower[1],
        upper[1],
        phase_count,
        endpoint=False,
        dtype=float,
    )
    modes: list[ELL1ConditionalMode] = []
    sigma = max(float(active.fast_prior_sigma), 1.0e-6)

    starts = [
        ((orbital_index, phase_index), np.concatenate(((orbital_value, phase_value), initial[2:])))
        for orbital_index, orbital_value in enumerate(orbital_values)
        for phase_index, phase_value in enumerate(phase_values)
    ]
    if lower[0] <= initial[0] <= upper[0]:
        starts.append(((-1, -1), np.clip(initial, lower, upper)))

    for grid_index, reference in starts:
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
                maximum_function_evaluations=active.fast_shape_maximum_evaluations,
            )
            linearization = projector.linearize_full(
                conditional,
                normalized_lower=lower,
                normalized_upper=upper,
                finite_difference_step=active.finite_difference_step,
            )
            prior_covariance = ell1_physical_prior_covariance(
                context=context,
                normalized_reference=conditional.normalized_values,
                prior_sigma=sigma,
            )
            marginal: SpinIntegerMarginalEvaluation = marginalize_spin_and_integer_paths(
                linearization,
                spin_prior_mean=np.zeros(9, dtype=float),
                spin_prior_covariance=prior_covariance,
                config=active.lattice,
            )
            if not marginal.path_modes:
                continue
            path = marginal.path_modes[0]
            posterior = conditional.normalized_values.copy()
            posterior += np.asarray(path.spin_posterior_mean, dtype=float)
            location = transport_ell1_posterior_to_support(
                context=context,
                reference_values=conditional.normalized_values,
                posterior_values=posterior,
            )
            in_support = location is not None
            physical = conditional.parameter_values
            if in_support:
                posterior = location.normalized_values
                physical = location.parameter_values
            strict_score = float(
                marginal.shape_quadratic + path.squared_mahalanobis
            )
            modes.append(
                ELL1ConditionalMode(
                    cell_id=(location.cell.cell_id if location is not None else cell.cell_id),
                    grid_index=grid_index,
                    normalized_values=posterior,
                    parameter_values=np.asarray(physical, dtype=float),
                    local_integer_offsets=conditional.profile.local_integer_offsets,
                    shape_chi2=float(conditional.profile.local_chi2),
                    strict_score=strict_score,
                    log_likelihood=float(marginal.log_likelihood),
                    integer_offsets=tuple(int(value) for value in path.integer_offsets),
                    posterior_weight=float(path.posterior_weight),
                    lattice_mass_converged=bool(marginal.lattice_mass_converged),
                    lattice_log_mass_increment=float(marginal.lattice_log_mass_increment),
                    lattice_visited_nodes=int(marginal.lattice_visited_nodes),
                    posterior_in_support=in_support,
                    source_cell_id=cell.cell_id,
                    posterior_transport_fraction=(
                        location.transport_fraction if location is not None else 0.0
                    ),
                )
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue

    modes.sort(
        key=lambda item: (
            not item.posterior_in_support,
            item.strict_score,
            -item.log_likelihood,
            item.grid_index,
        )
    )
    return tuple(modes[: max(int(active.modes_per_cell), 1)])


@dataclass(frozen=True)
class _ELL1AnnealingParticle:
    normalized_values: np.ndarray
    integer_offsets: tuple[int, ...]
    objective: float
    trace: tuple[ELL1AnnealingStep, ...]


def _project_to_ell1_support_segment(
    *,
    reference: np.ndarray,
    target: np.ndarray,
    projector: PintELL1DualLatticeProjector,
    lower: np.ndarray,
    upper: np.ndarray,
    local_integer_offsets: Sequence[Sequence[int]],
) -> np.ndarray:
    origin = np.asarray(reference, dtype=float)
    proposal = np.asarray(target, dtype=float)

    def valid(alpha: float) -> bool:
        state = origin + float(alpha) * (proposal - origin)
        if np.any(state < lower) or np.any(state > upper):
            return False
        try:
            projector.profile_normalized(
                state,
                local_integer_offsets=local_integer_offsets,
            )
        except (ArithmeticError, RuntimeError, ValueError):
            return False
        return True

    if not valid(0.0):
        raise ValueError("The ELL1 annealing reference lies outside support.")
    if valid(1.0):
        return proposal.copy()
    valid_alpha = 0.0
    invalid_alpha = 1.0
    for _ in range(64):
        midpoint = 0.5 * (valid_alpha + invalid_alpha)
        if valid(midpoint):
            valid_alpha = midpoint
        else:
            invalid_alpha = midpoint
    return origin + max(valid_alpha - 1.0e-8, 0.0) * (proposal - origin)


def _close_softened_ell1_path(
    *,
    projector: PintELL1DualLatticeProjector,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    local_integer_offsets: Sequence[Sequence[int]],
    integer_offsets: Sequence[int],
    phase_sigma_cycles: float,
    maximum_function_evaluations: int,
) -> tuple[np.ndarray, TwoLevelPulseProfile, float, float, int, bool]:
    path = np.asarray(integer_offsets, dtype=float)
    sigma = float(phase_sigma_cycles)
    _, reference_profile = projector.profile_normalized(
        start,
        local_integer_offsets=local_integer_offsets,
    )
    shape_size = np.asarray(reference_profile.normalized_local_residuals).size
    failure = np.full(shape_size + path.size, 1.0e12, dtype=float)

    def residual(state: np.ndarray) -> np.ndarray:
        try:
            _, profile = projector.profile_normalized(
                state,
                local_integer_offsets=local_integer_offsets,
            )
            shape = np.asarray(profile.normalized_local_residuals, dtype=float)
            anchors = np.asarray(profile.session_intercepts, dtype=float)
            variances = np.asarray(profile.session_intercept_variances, dtype=float)
            covariance = np.diag(variances[1:]) + float(variances[0])
            if sigma > 0.0:
                covariance += sigma * sigma * (
                    np.eye(path.size, dtype=float)
                    + np.ones((path.size, path.size), dtype=float)
                )
            phase = np.linalg.solve(
                np.linalg.cholesky(covariance),
                anchors[1:] - anchors[0] - path,
            )
            combined = np.concatenate((shape, phase))
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
    _, profile = projector.profile_normalized(
        state,
        local_integer_offsets=local_integer_offsets,
    )
    vector = residual(state)
    shape = np.asarray(profile.normalized_local_residuals, dtype=float)
    shape_chi2 = float(shape @ shape)
    phase_chi2 = float(vector[shape.size:] @ vector[shape.size:])
    return state, profile, shape_chi2, phase_chi2, int(fit.nfev), bool(fit.success)


def _record_annealing_failure(
    reasons: dict[str, int],
    error: BaseException,
) -> None:
    message = " ".join(str(error).split())
    label = type(error).__name__ + (f": {message[:240]}" if message else "")
    reasons[label] = reasons.get(label, 0) + 1


def _empty_annealing_stage_reason(
    *,
    input_particles: int,
    profile_lattice_failures: int,
    empty_path_posteriors: int,
    attempted_paths: int,
    closure_failures: int,
    nonfinite_objectives: int,
) -> str:
    if profile_lattice_failures >= input_particles:
        return "all particles failed PINT profile, linearization, or lattice marginalization"
    if attempted_paths == 0 and empty_path_posteriors > 0:
        return "integer-lattice marginalization returned no posterior paths"
    if attempted_paths > 0 and closure_failures >= attempted_paths:
        return "all posterior paths failed bounded nonlinear closure"
    if attempted_paths > 0 and nonfinite_objectives >= attempted_paths - closure_failures:
        return "all closed paths produced non-finite objectives"
    return "annealing stage retained no finite posterior candidate"


def anneal_ell1_conditional_mode_diagnosed(
    *,
    context: ELL1DualLatticeSearchContext,
    mode: ELL1ConditionalMode,
    config: ELL1AnnealingConfig | None = None,
) -> ELL1AnnealingResult:
    """Transport a mode portfolio through one complete phase-noise posterior.

    Every stage re-marginalizes the complete session-integer lattice before
    branching into exact softened closures. The retained beam is an explicit
    finite-budget approximation, not a claim that discarded mass is zero.
    """

    active = config or ELL1AnnealingConfig()
    cell_by_id = {cell.cell_id: cell for cell in context.cells}
    if mode.cell_id not in cell_by_id:
        raise ValueError("The ELL1 mode cell is outside the context tiling.")
    cell = cell_by_id[mode.cell_id]
    lower, upper = normalized_bounds_for_ell1_cell(context.support, cell)
    projector = PintELL1DualLatticeProjector(context)
    local_component = mode.local_integer_offsets
    particles = [
        _ELL1AnnealingParticle(
            normalized_values=np.asarray(mode.normalized_values, dtype=float).copy(),
            integer_offsets=mode.integer_offsets,
            objective=float(mode.strict_score),
            trace=tuple(),
        )
    ]
    prior_sigma = max(float(active.coordinate_prior_sigma), 1.0e-6)
    stage_diagnostics: list[ELL1AnnealingStageDiagnostics] = []

    for stage_index, phase_sigma in enumerate(active.phase_sigma_schedule):
        input_particle_count = len(particles)
        profile_lattice_failure_count = 0
        empty_path_posterior_count = 0
        attempted_path_count = 0
        closure_failure_count = 0
        closure_nonconverged_count = 0
        nonfinite_objective_count = 0
        failure_reasons: dict[str, int] = {}
        candidates: list[_ELL1AnnealingParticle] = []
        lattice_config = replace(
            active.lattice,
            added_session_phase_sigma_cycles=float(phase_sigma),
        )
        for particle in particles:
            try:
                physical, profile = projector.profile_normalized(
                    particle.normalized_values,
                    local_integer_offsets=local_component,
                )
                conditional = ELL1ConditionalProfile(
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
                prior_covariance = ell1_physical_prior_covariance(
                    context=context,
                    normalized_reference=particle.normalized_values,
                    prior_sigma=prior_sigma,
                )
                marginal = marginalize_spin_and_integer_paths(
                    linearization,
                    spin_prior_mean=np.zeros(9, dtype=float),
                    spin_prior_covariance=prior_covariance,
                    config=lattice_config,
                )
            except (
                ArithmeticError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                profile_lattice_failure_count += 1
                _record_annealing_failure(failure_reasons, error)
                continue
            path_modes = marginal.path_modes[: max(int(active.paths_per_state), 1)]
            if not path_modes:
                empty_path_posterior_count += 1
            for path_mode in path_modes:
                attempted_path_count += 1
                path = tuple(int(value) for value in path_mode.integer_offsets)
                proposal = particle.normalized_values + np.asarray(
                    path_mode.spin_posterior_mean,
                    dtype=float,
                )
                try:
                    proposal = _project_to_ell1_support_segment(
                        reference=particle.normalized_values,
                        target=proposal,
                        projector=projector,
                        lower=lower,
                        upper=upper,
                        local_integer_offsets=local_component,
                    )
                    (
                        state,
                        _,
                        shape_chi2,
                        phase_chi2,
                        function_evaluations,
                        converged,
                    ) = _close_softened_ell1_path(
                        projector=projector,
                        start=proposal,
                        lower=lower,
                        upper=upper,
                        local_integer_offsets=local_component,
                        integer_offsets=path,
                        phase_sigma_cycles=float(phase_sigma),
                        maximum_function_evaluations=active.maximum_closure_evaluations,
                    )
                except (
                    ArithmeticError,
                    RuntimeError,
                    ValueError,
                    np.linalg.LinAlgError,
                ) as error:
                    closure_failure_count += 1
                    _record_annealing_failure(failure_reasons, error)
                    continue
                objective = float(shape_chi2 + phase_chi2)
                if not np.isfinite(objective):
                    nonfinite_objective_count += 1
                    continue
                if not converged:
                    closure_nonconverged_count += 1
                strict_score = float(
                    marginal.shape_quadratic + path_mode.squared_mahalanobis
                )
                step = ELL1AnnealingStep(
                    phase_sigma_cycles=float(phase_sigma),
                    integer_offsets=path,
                    normalized_values=state.copy(),
                    shape_chi2=shape_chi2,
                    phase_chi2=phase_chi2,
                    total_chi2=objective,
                    lattice_strict_score=strict_score,
                    lattice_log_likelihood=float(marginal.log_likelihood),
                    lattice_mass_converged=bool(marginal.lattice_mass_converged),
                    lattice_log_mass_increment=float(marginal.lattice_log_mass_increment),
                    lattice_visited_nodes=int(marginal.lattice_visited_nodes),
                    function_evaluations=function_evaluations,
                    converged=converged,
                )
                candidates.append(
                    _ELL1AnnealingParticle(
                        normalized_values=state,
                        integer_offsets=path,
                        objective=objective,
                        trace=(*particle.trace, step),
                    )
                )
        deduplicated: list[_ELL1AnnealingParticle] = []
        if candidates:
            candidates.sort(key=lambda item: (item.objective, item.integer_offsets))
            seen: set[tuple[tuple[int, ...], tuple[float, ...]]] = set()
            for candidate in candidates:
                key = (
                    candidate.integer_offsets,
                    tuple(np.round(candidate.normalized_values, decimals=8)),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduplicated.append(candidate)
                if len(deduplicated) >= max(int(active.beam_size), 1):
                    break
        stage_diagnostics.append(
            ELL1AnnealingStageDiagnostics(
                phase_sigma_cycles=float(phase_sigma),
                input_particle_count=input_particle_count,
                profile_lattice_failure_count=profile_lattice_failure_count,
                empty_path_posterior_count=empty_path_posterior_count,
                attempted_path_count=attempted_path_count,
                closure_failure_count=closure_failure_count,
                closure_nonconverged_count=closure_nonconverged_count,
                nonfinite_objective_count=nonfinite_objective_count,
                candidate_count=len(candidates),
                retained_particle_count=len(deduplicated),
                failure_reasons=tuple(
                    sorted(
                        failure_reasons.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:8]
                ),
            )
        )
        if not deduplicated:
            return ELL1AnnealingResult(
                modes=tuple(),
                stages=tuple(stage_diagnostics),
                completed_schedule=False,
                failure_stage_index=stage_index,
                failure_reason=_empty_annealing_stage_reason(
                    input_particles=input_particle_count,
                    profile_lattice_failures=profile_lattice_failure_count,
                    empty_path_posteriors=empty_path_posterior_count,
                    attempted_paths=attempted_path_count,
                    closure_failures=closure_failure_count,
                    nonfinite_objectives=nonfinite_objective_count,
                ),
            )
        particles = deduplicated

    results: list[ELL1AnnealedMode] = []
    for particle in particles:
        physical, profile = projector.profile_normalized(
            particle.normalized_values,
            local_integer_offsets=local_component,
        )
        final_step = particle.trace[-1]
        results.append(
            ELL1AnnealedMode(
                cell_id=mode.cell_id,
                source_grid_index=mode.grid_index,
                normalized_values=particle.normalized_values.copy(),
                parameter_values=physical,
                local_integer_offsets=profile.local_integer_offsets,
                integer_offsets=particle.integer_offsets,
                shape_chi2=float(final_step.shape_chi2),
                phase_chi2=float(final_step.phase_chi2),
                total_chi2=float(final_step.total_chi2),
                converged=bool(final_step.converged),
                trace=particle.trace,
                source_cell_id=mode.source_cell_id or mode.cell_id,
            )
        )
    results.sort(key=lambda item: (item.total_chi2, item.integer_offsets))
    return ELL1AnnealingResult(
        modes=tuple(results),
        stages=tuple(stage_diagnostics),
        completed_schedule=True,
    )


def anneal_ell1_conditional_mode(
    *,
    context: ELL1DualLatticeSearchContext,
    mode: ELL1ConditionalMode,
    config: ELL1AnnealingConfig | None = None,
) -> tuple[ELL1AnnealedMode, ...]:
    """Transport a mode while preserving the historical tuple return API."""

    return anneal_ell1_conditional_mode_diagnosed(
        context=context,
        mode=mode,
        config=config,
    ).modes


__all__ = [
    "ELL1_CORE_PARAMETERS",
    "ELL1_FAST_NAMES",
    "ELL1_FIT_PARAMETERS",
    "ELL1_SLOW_NAMES",
    "ELL1CellSearchConfig",
    "ELL1AnnealedMode",
    "ELL1AnnealingConfig",
    "ELL1AnnealingResult",
    "ELL1AnnealingStageDiagnostics",
    "ELL1AnnealingStep",
    "ELL1ConditionalMode",
    "ELL1PosteriorCellLocation",
    "ELL1DualLatticeCoordinateSystem",
    "ELL1DualLatticeSearchContext",
    "ELL1DualLatticeSupport",
    "PintELL1DualLatticeProjector",
    "PintELL1PhaseEvaluator",
    "build_ell1_dual_lattice_search_context",
    "build_ell1_dual_lattice_support",
    "build_generic_ell1_physical_support",
    "anneal_ell1_conditional_mode",
    "anneal_ell1_conditional_mode_diagnosed",
    "normalized_bounds_for_ell1_cell",
    "locate_ell1_posterior_cell",
    "ell1_physical_prior_covariance",
    "transport_ell1_posterior_to_support",
    "search_ell1_orbital_cycle_cell",
]
