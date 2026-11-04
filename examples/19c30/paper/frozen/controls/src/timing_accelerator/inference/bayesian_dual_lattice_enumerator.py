from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import least_squares

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
)
from timing_accelerator.inference.bayesian_semilinear_lattice import (
    build_generic_dd_physical_support,
)
from timing_accelerator.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
)


DUAL_LATTICE_FIT_PARAMETERS = (
    "F0",
    "F1",
    "A1",
    "ECC",
    "T0",
    "PB",
    "OM",
    "OMDOT",
)
DUAL_LATTICE_SLOW_NAMES = ("C_ORB", "PHI_REF", "LOGIT_ECC", "DOMEGA")
DUAL_LATTICE_FAST_NAMES = ("F0_REF", "F1", "U", "V")


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


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("Eccentricity must lie strictly between zero and one.")
    return float(math.log(value) - math.log1p(-value))


def _wrap_radians(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _wrap_degrees_near(value: float, center: float) -> float:
    return float(center + (value - center + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class DualLatticeCoordinateSystem:
    """DD coordinates that expose orbital-cycle and pulse-phase aliases."""

    parameter_names: tuple[str, ...]
    observation_span_days: float
    reference_mjd: float
    pepoch_mjd: float
    initial_t0_mjd: float
    initial_om_degrees: float

    def __post_init__(self) -> None:
        names = tuple(str(name).upper() for name in self.parameter_names)
        if set(names) != set(DUAL_LATTICE_FIT_PARAMETERS) or len(names) != 8:
            raise ValueError(
                "Dual-lattice DD coordinates require exactly: "
                + ", ".join(DUAL_LATTICE_FIT_PARAMETERS)
            )
        if not np.isfinite(self.observation_span_days) or self.observation_span_days <= 0.0:
            raise ValueError("The observation span must be finite and positive.")
        if not all(
            np.isfinite(value)
            for value in (
                self.reference_mjd,
                self.pepoch_mjd,
                self.initial_t0_mjd,
                self.initial_om_degrees,
            )
        ):
            raise ValueError("Dual-lattice reference values must be finite.")

    def _index(self, name: str) -> int:
        return self.parameter_names.index(name)

    def split_physical(
        self,
        parameter_values: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        physical = _finite_vector(
            parameter_values,
            dimension=8,
            label="DD physical parameters",
        )
        f0 = float(physical[self._index("F0")])
        f1 = float(physical[self._index("F1")])
        a1 = float(physical[self._index("A1")])
        ecc = float(physical[self._index("ECC")])
        t0 = float(physical[self._index("T0")])
        pb = float(physical[self._index("PB")])
        om = float(physical[self._index("OM")])
        omdot = float(physical[self._index("OMDOT")])
        if f0 <= 0.0 or a1 <= 0.0 or pb <= 0.0:
            raise ValueError("F0, A1, and PB must be positive.")

        spin_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        f0_reference = f0 + f1 * spin_seconds
        orbital_cycles = (self.reference_mjd - t0) / pb
        phi_reference = _wrap_radians(2.0 * math.pi * orbital_cycles)
        omega_reference = math.radians(
            om + omdot * (self.reference_mjd - t0) / 365.25
        )
        slow = np.asarray(
            (
                self.observation_span_days / pb,
                phi_reference,
                _logit(ecc),
                omdot * self.observation_span_days / 365.25,
            ),
            dtype=float,
        )
        fast = np.asarray(
            (
                f0_reference,
                f1,
                f0_reference * a1 * math.sin(omega_reference),
                f0_reference * a1 * math.cos(omega_reference),
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
            dimension=4,
            label="Dual-lattice slow coordinates",
        )
        fast = _finite_vector(
            fast_values,
            dimension=4,
            label="Dual-lattice fast coordinates",
        )
        c_orb, phi_reference, logit_ecc, domega = map(float, slow)
        f0_reference, f1, u_value, v_value = map(float, fast)
        if c_orb <= 0.0 or f0_reference <= 0.0:
            raise ValueError("C_orb and F0_ref must be positive.")
        pb = self.observation_span_days / c_orb
        ecc = float(expit(logit_ecc))
        omdot = domega * 365.25 / self.observation_span_days

        fractional_cycle = _wrap_radians(phi_reference) / (2.0 * math.pi)
        initial_cycles = (self.reference_mjd - self.initial_t0_mjd) / pb
        cycle_index = int(round(initial_cycles - fractional_cycle))
        t0 = self.reference_mjd - pb * (cycle_index + fractional_cycle)

        a1_phase = math.hypot(u_value, v_value)
        if a1_phase <= 0.0:
            raise ValueError("The phase-domain A1 vector must have positive norm.")
        a1 = a1_phase / f0_reference
        omega_reference_degrees = math.degrees(math.atan2(u_value, v_value))
        om = omega_reference_degrees - omdot * (self.reference_mjd - t0) / 365.25
        om = _wrap_degrees_near(om, self.initial_om_degrees)
        f0 = f0_reference - f1 * (
            self.reference_mjd - self.pepoch_mjd
        ) * 86400.0

        by_name = {
            "F0": f0,
            "F1": f1,
            "A1": a1,
            "ECC": ecc,
            "T0": t0,
            "PB": pb,
            "OM": om,
            "OMDOT": omdot,
        }
        return np.asarray([by_name[name] for name in self.parameter_names], dtype=float)


@dataclass(frozen=True)
class OrbitalCycleCell:
    cell_id: str
    lower: float
    upper: float

    @property
    def center(self) -> float:
        return float(0.5 * (self.lower + self.upper))

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)


@dataclass(frozen=True)
class DualLatticeSupport:
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

    def normalize(
        self,
        slow_values: Sequence[float],
        fast_values: Sequence[float],
    ) -> np.ndarray:
        slow = _finite_vector(slow_values, dimension=4, label="Slow coordinates")
        fast = _finite_vector(fast_values, dimension=4, label="Fast coordinates")
        slow_midpoint = 0.5 * (self.slow_lower + self.slow_upper)
        fast_midpoint = 0.5 * (self.fast_lower + self.fast_upper)
        return np.concatenate(
            (
                (slow - slow_midpoint) / self.slow_scale,
                (fast - fast_midpoint) / self.fast_scale,
            )
        )

    def denormalize(self, normalized_values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        normalized = _finite_vector(
            normalized_values,
            dimension=8,
            label="Normalized dual-lattice coordinates",
        )
        slow_midpoint = 0.5 * (self.slow_lower + self.slow_upper)
        fast_midpoint = 0.5 * (self.fast_lower + self.fast_upper)
        slow = slow_midpoint + self.slow_scale * normalized[:4]
        fast = fast_midpoint + self.fast_scale * normalized[4:]
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


@dataclass(frozen=True)
class DualLatticeSearchContext:
    """PINT data, coordinate support, and complete orbital-cycle tiling."""

    evaluator: "PintDualLatticePhaseEvaluator"
    coordinates: DualLatticeCoordinateSystem
    support: DualLatticeSupport
    cells: tuple[OrbitalCycleCell, ...]


def build_dual_lattice_support(
    *,
    coordinates: DualLatticeCoordinateSystem,
    center_values: Sequence[float],
    physical_lower: Sequence[float],
    physical_upper: Sequence[float],
) -> DualLatticeSupport:
    center = _finite_vector(center_values, dimension=8, label="DD support center")
    lower = _finite_vector(physical_lower, dimension=8, label="DD support lower")
    upper = _finite_vector(physical_upper, dimension=8, label="DD support upper")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Every DD support interval must contain its center.")
    index = {name: coordinates.parameter_names.index(name) for name in coordinates.parameter_names}
    center_slow, center_fast = coordinates.split_physical(center)
    c_lower = coordinates.observation_span_days / upper[index["PB"]]
    c_upper = coordinates.observation_span_days / lower[index["PB"]]
    slow_lower = np.asarray(
        (
            c_lower,
            -math.pi,
            _logit(float(lower[index["ECC"]])),
            float(lower[index["OMDOT"]])
            * coordinates.observation_span_days
            / 365.25,
        ),
        dtype=float,
    )
    slow_upper = np.asarray(
        (
            c_upper,
            math.pi,
            _logit(float(upper[index["ECC"]])),
            float(upper[index["OMDOT"]])
            * coordinates.observation_span_days
            / 365.25,
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
    f0_reference_lower = float(np.min(f0_reference_corners))
    f0_reference_upper = float(np.max(f0_reference_corners))
    maximum_phase_amplitude = max(
        abs(f0_reference_lower),
        abs(f0_reference_upper),
    ) * float(upper[index["A1"]])
    fast_lower = np.asarray(
        (
            f0_reference_lower,
            lower[index["F1"]],
            -maximum_phase_amplitude,
            -maximum_phase_amplitude,
        ),
        dtype=float,
    )
    fast_upper = np.asarray(
        (
            f0_reference_upper,
            upper[index["F1"]],
            maximum_phase_amplitude,
            maximum_phase_amplitude,
        ),
        dtype=float,
    )
    return DualLatticeSupport(
        physical_lower=lower.copy(),
        physical_upper=upper.copy(),
        slow_lower=slow_lower,
        slow_upper=slow_upper,
        fast_lower=fast_lower,
        fast_upper=fast_upper,
        slow_center=center_slow,
        fast_center=center_fast,
    )


def build_dual_lattice_search_context(
    *,
    par_path: Path,
    tim_path: Path,
    maximum_cell_width: float = 0.25,
    support_width_scale: float = 1.0,
    minimum_f1_half_width: float = 1.0e-10,
    clock_override_path: Path | None = None,
) -> DualLatticeSearchContext:
    """Build the source-agnostic DD search domain used by APOLLO.

    The support is derived only from the input timing model and observation
    span. It does not use a source class, a historical solution, or manually
    supplied pulse wraps.
    """

    evaluator = PintDualLatticePhaseEvaluator.from_files(
        par_path=Path(par_path),
        tim_path=Path(tim_path),
        parameter_names=DUAL_LATTICE_FIT_PARAMETERS,
        clock_override_path=clock_override_path,
    )
    epochs = evaluator.observation_epochs_mjd
    span_days = float(np.ptp(epochs))
    if not np.isfinite(span_days) or span_days <= 0.0:
        raise ValueError("Dual-lattice search requires TOAs at distinct epochs.")
    reference_mjd = float(np.median(epochs))
    model = evaluator.model
    coordinates = DualLatticeCoordinateSystem(
        parameter_names=evaluator.parameter_names,
        observation_span_days=span_days,
        reference_mjd=reference_mjd,
        pepoch_mjd=float(model["PEPOCH"].value),
        initial_t0_mjd=float(model["T0"].value),
        initial_om_degrees=float(model["OM"].value),
    )
    lower, upper = build_generic_dd_physical_support(
        parameter_names=evaluator.parameter_names,
        center_values=evaluator.base_parameter_values,
        observation_span_days=span_days,
        width_scale=float(support_width_scale),
    )
    f1_index = evaluator.parameter_names.index("F1")
    f1_half_width = float(minimum_f1_half_width)
    if not np.isfinite(f1_half_width) or f1_half_width <= 0.0:
        raise ValueError("The minimum F1 half-width must be positive and finite.")
    lower[f1_index] = min(float(lower[f1_index]), -f1_half_width)
    upper[f1_index] = max(float(upper[f1_index]), f1_half_width)
    support = build_dual_lattice_support(
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
        maximum_cell_width=float(maximum_cell_width),
    )
    return DualLatticeSearchContext(
        evaluator=evaluator,
        coordinates=coordinates,
        support=support,
        cells=cells,
    )


def tile_orbital_cycle_support(
    *,
    pb_lower_days: float,
    pb_upper_days: float,
    observation_span_days: float,
    maximum_cell_width: float = 0.25,
) -> tuple[OrbitalCycleCell, ...]:
    """Tile every orbital-frequency alias without a count cap."""

    pb_lower = float(pb_lower_days)
    pb_upper = float(pb_upper_days)
    span = float(observation_span_days)
    maximum_width = float(maximum_cell_width)
    if not all(np.isfinite(value) for value in (pb_lower, pb_upper, span, maximum_width)):
        raise ValueError("Orbital support values must be finite.")
    if pb_lower <= 0.0 or pb_upper <= pb_lower or span <= 0.0 or maximum_width <= 0.0:
        raise ValueError("Orbital support requires 0 < PB_lower < PB_upper and positive widths.")

    c_lower = span / pb_upper
    c_upper = span / pb_lower
    cell_count = max(int(math.ceil((c_upper - c_lower) / maximum_width)), 1)
    edges = np.linspace(c_lower, c_upper, cell_count + 1, dtype=float)
    width_digits = max(6, len(str(cell_count)))
    return tuple(
        OrbitalCycleCell(
            cell_id=f"C{index:0{width_digits}d}",
            lower=float(edges[index]),
            upper=float(edges[index + 1]),
        )
        for index in range(cell_count)
    )


@dataclass(frozen=True)
class SourceSession:
    session_id: str
    toa_indices: tuple[int, ...]
    epoch_mjd: float


SOURCE_SESSION_GAP_DAYS = 0.5


def infer_contiguous_source_sessions(toas: object) -> tuple[SourceSession, ...]:
    """Infer observing sessions from contiguous TOA epochs.

    The Tempo2 name field identifies an individual TOA input and commonly
    changes between integrations in the same observing session. It is useful
    as a human-readable session label, but it is not a session boundary.
    """

    flags = list(toas.table["flags"])
    mjds = np.asarray(toas.get_mjds().value, dtype=float)
    if len(flags) != mjds.size or mjds.size == 0:
        raise ValueError("TOA flags and epochs must be non-empty and aligned.")
    if np.any(np.diff(mjds) < 0.0):
        raise ValueError("TOA epochs must be ordered before session inference.")
    raw_names = [item.get("name") for item in flags]
    names = [str(name) if name is not None else None for name in raw_names]
    sessions: list[SourceSession] = []
    id_counts: dict[str, int] = {}
    start = 0
    for stop in range(1, len(names) + 1):
        if stop < len(names):
            time_gap = float(mjds[stop] - mjds[stop - 1]) > SOURCE_SESSION_GAP_DAYS
            if not time_gap:
                continue
        indices = tuple(range(start, stop))
        base_id = names[start] if names[start] is not None else "session"
        occurrence = id_counts.get(base_id, 0)
        id_counts[base_id] = occurrence + 1
        session_id = base_id if occurrence == 0 else f"{base_id}#{occurrence + 1}"
        sessions.append(
            SourceSession(
                session_id=session_id,
                toa_indices=indices,
                epoch_mjd=float(np.mean(mjds[list(indices)])),
            )
        )
        start = stop
    covered = tuple(index for session in sessions for index in session.toa_indices)
    if covered != tuple(range(mjds.size)):
        raise RuntimeError("Inferred source sessions do not cover each TOA exactly once.")
    return tuple(sessions)


@dataclass(frozen=True)
class LocalPulseCountMode:
    relative_integer_offsets: tuple[int, ...]
    intercept: float
    chi2: float
    log_weight: float


@dataclass(frozen=True)
class LocalPulseCountMixture:
    log_likelihood: float
    modes: tuple[LocalPulseCountMode, ...]

    @property
    def best(self) -> LocalPulseCountMode:
        if not self.modes:
            raise RuntimeError("A local pulse-count mixture has no modes.")
        return min(self.modes, key=lambda item: item.chi2)


def enumerate_local_pulse_count_mixture(
    phase_residuals: Sequence[float],
    phase_sigmas: Sequence[float],
    *,
    integer_radius: int = 1,
    maximum_retained_modes: int | None = None,
) -> LocalPulseCountMixture:
    """Marginalize one session's relative pulse numbers with its anchor fixed."""

    residuals = np.asarray(phase_residuals, dtype=float)
    sigmas = np.asarray(phase_sigmas, dtype=float)
    if residuals.ndim != 1 or sigmas.shape != residuals.shape or residuals.size == 0:
        raise ValueError("Local phase residuals and sigmas must be aligned vectors.")
    if not np.all(np.isfinite(residuals)) or not np.all(
        np.isfinite(sigmas) & (sigmas > 0.0)
    ):
        raise ValueError("Local pulse-count inputs must be finite with positive sigmas.")
    radius = int(integer_radius)
    if radius < 0:
        raise ValueError("The local integer radius cannot be negative.")

    if residuals.size == 1:
        return LocalPulseCountMixture(
            log_likelihood=0.0,
            modes=(
                LocalPulseCountMode(
                    relative_integer_offsets=(0,),
                    intercept=float(residuals[0]),
                    chi2=0.0,
                    log_weight=0.0,
                ),
            ),
        )

    anchor = float(residuals[0])
    integer_centers = np.rint(residuals[1:] - anchor).astype(int)
    alternatives = tuple(range(-radius, radius + 1))
    weights = 1.0 / np.square(sigmas)
    modes: list[LocalPulseCountMode] = []
    for deltas in product(alternatives, repeat=residuals.size - 1):
        integers = np.concatenate(
            (
                np.zeros(1, dtype=int),
                integer_centers + np.asarray(deltas, dtype=int),
            )
        )
        shifted = residuals - integers
        intercept = float(np.sum(weights * shifted) / np.sum(weights))
        normalized = (shifted - intercept) / sigmas
        chi2 = float(normalized @ normalized)
        modes.append(
            LocalPulseCountMode(
                relative_integer_offsets=tuple(int(value) for value in integers),
                intercept=intercept,
                chi2=chi2,
                log_weight=-0.5 * chi2,
            )
        )
    modes.sort(key=lambda item: (item.chi2, item.relative_integer_offsets))
    log_likelihood = float(logsumexp([item.log_weight for item in modes]))
    if maximum_retained_modes is not None:
        retained = max(int(maximum_retained_modes), 1)
        modes = modes[:retained]
    return LocalPulseCountMixture(
        log_likelihood=log_likelihood,
        modes=tuple(modes),
    )


@dataclass(frozen=True)
class TwoLevelPulseProfile:
    normalized_local_residuals: np.ndarray
    session_intercepts: np.ndarray
    session_intercept_variances: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    local_chi2: float
    local_log_likelihood: float


def circular_session_anchor_residuals(
    profile: TwoLevelPulseProfile,
    phase_sigma_cycles: float,
) -> np.ndarray:
    """Embed every relative session anchor in one smooth periodic likelihood."""

    sigma = float(phase_sigma_cycles)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("The circular phase sigma must be finite and positive.")
    offsets = np.asarray(profile.session_intercepts, dtype=float)
    if offsets.ndim != 1 or offsets.size < 2 or not np.all(np.isfinite(offsets)):
        raise ValueError("Circular phase inference needs at least two session anchors.")
    angles = 2.0 * math.pi * (offsets[1:] - offsets[0])
    scale = 2.0 * math.pi * sigma
    return np.concatenate(
        (
            np.sin(angles) / scale,
            (1.0 - np.cos(angles)) / scale,
        )
    )


def _initialize_session_local_pulse_numbers(
    toas: object,
    model: object,
    sessions: Sequence[SourceSession],
) -> np.ndarray:
    """Create an initial local reference without connecting separate sessions."""

    from astropy import units as u
    from pint.residuals import Residuals

    if "pulse_number" in toas.table.colnames:
        supplied = np.ma.asarray(toas.table["pulse_number"])
        supplied_mask = np.ma.getmaskarray(supplied)
        supplied_values = np.asarray(supplied.data, dtype=float)
        if (
            supplied_values.shape == (len(toas.table),)
            and not np.any(supplied_mask)
            and np.all(np.isfinite(supplied_values))
        ):
            return supplied_values.copy()

    if not hasattr(toas, "compute_pulse_numbers"):
        raise RuntimeError("This PINT version cannot compute pulse numbers.")
    toas.compute_pulse_numbers(model)
    pulse_numbers = np.ma.asarray(toas.table["pulse_number"])
    if np.any(np.ma.getmaskarray(pulse_numbers)):
        raise RuntimeError("PINT did not compute every initial pulse number.")
    base = np.asarray(pulse_numbers.data, dtype=float)
    residuals = Residuals(
        toas,
        model,
        subtract_mean=False,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    phase_residuals = np.asarray(
        residuals.phase_resids.to_value(u.dimensionless_unscaled),
        dtype=float,
    )
    adjusted = base.copy()
    for session in sessions:
        indices = np.asarray(session.toa_indices, dtype=int)
        local = phase_residuals[indices]
        unwrapped = np.unwrap(local, period=1.0)
        island_center = int(np.rint(np.median(unwrapped)))
        centered = unwrapped - island_center
        raw_adjustments = local - centered
        integer_adjustments = np.rint(raw_adjustments).astype(int)
        if not np.allclose(raw_adjustments, integer_adjustments, atol=1.0e-8, rtol=0.0):
            raise RuntimeError(
                f"Session '{session.session_id}' did not map to integer pulses."
            )
        adjusted[indices] += integer_adjustments
    return adjusted


class PintDualLatticePhaseEvaluator:
    """Mutable, single-process PINT evaluator for dual-lattice cell searches."""

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
        if set(names) != set(DUAL_LATTICE_FIT_PARAMETERS) or len(names) != 8:
            raise ValueError("PINT dual-lattice evaluation requires the eight DD parameters.")
        missing = [name for name in names if name not in model.params]
        if missing:
            raise ValueError("PINT model is missing: " + ", ".join(missing))
        pulse_numbers = np.asarray(base_pulse_numbers, dtype=np.longdouble)
        if pulse_numbers.shape != (len(toas.table),) or not np.all(
            np.isfinite(pulse_numbers)
        ):
            raise ValueError("Base pulse numbers must align with all PINT TOAs.")
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
        parameter_names: Sequence[str] = DUAL_LATTICE_FIT_PARAMETERS,
        clock_override_path: Path | None = None,
    ) -> "PintDualLatticePhaseEvaluator":
        model = _load_pint_model(Path(par_path))
        free = tuple(str(name).upper() for name in model.free_params)
        if set(free) != set(DUAL_LATTICE_FIT_PARAMETERS) or len(free) != 8:
            raise ValueError(
                "The input par must fit exactly: "
                + ", ".join(DUAL_LATTICE_FIT_PARAMETERS)
            )
        toas = _load_pint_toas(
            Path(tim_path),
            seed_model=model,
            clock_override_path=clock_override_path,
        )
        sessions = infer_contiguous_source_sessions(toas)
        base_pulse_numbers = _initialize_session_local_pulse_numbers(
            toas,
            model,
            sessions,
        )
        return cls(
            model=model,
            toas=toas,
            parameter_names=parameter_names,
            sessions=sessions,
            base_pulse_numbers=base_pulse_numbers,
        )

    @property
    def observation_epochs_mjd(self) -> np.ndarray:
        from astropy import units as u

        return np.asarray(self.toas.get_mjds().to_value(u.day), dtype=float)

    def _phase_sigmas(self) -> np.ndarray:
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

    def evaluate_values(
        self,
        parameter_values: Sequence[float],
        *,
        local_integer_radius: int = 0,
        maximum_retained_local_modes: int | None = None,
    ) -> TwoLevelPulseProfile:
        values = _finite_vector(
            parameter_values,
            dimension=8,
            label="PINT DD parameters",
        )
        by_name = dict(zip(self.parameter_names, values, strict=True))
        if (
            by_name["F0"] <= 0.0
            or by_name["A1"] <= 0.0
            or by_name["PB"] <= 0.0
            or not 0.0 < by_name["ECC"] < 1.0
        ):
            raise ValueError("PINT DD parameters are outside their physical domain.")
        for name, value in by_name.items():
            _set_numeric_model_parameter_value(self.model, name, float(value))
        phase = self.model.phase(self.toas, abs_phase=True)
        phase_values = np.asarray(phase.int.value, dtype=np.longdouble)
        phase_values += np.asarray(phase.frac.value, dtype=np.longdouble)
        residuals = np.asarray(phase_values - self.base_pulse_numbers, dtype=float)
        return profile_two_level_pulse_lattice(
            residuals,
            self._phase_sigmas(),
            self.sessions,
            integer_radius=local_integer_radius,
            maximum_retained_modes=maximum_retained_local_modes,
        )

    def evaluate_fixed_local_component(
        self,
        parameter_values: Sequence[float],
        local_integer_offsets: Sequence[Sequence[int]] | None = None,
    ) -> TwoLevelPulseProfile:
        """Evaluate one local-count posterior component without path switching."""

        values = _finite_vector(
            parameter_values,
            dimension=8,
            label="PINT DD parameters",
        )
        by_name = dict(zip(self.parameter_names, values, strict=True))
        if (
            by_name["F0"] <= 0.0
            or by_name["A1"] <= 0.0
            or by_name["PB"] <= 0.0
            or not 0.0 < by_name["ECC"] < 1.0
        ):
            raise ValueError("PINT DD parameters are outside their physical domain.")
        for name, value in by_name.items():
            _set_numeric_model_parameter_value(self.model, name, float(value))
        phase = self.model.phase(self.toas, abs_phase=True)
        phase_values = np.asarray(phase.int.value, dtype=np.longdouble)
        phase_values += np.asarray(phase.frac.value, dtype=np.longdouble)
        residuals = np.asarray(phase_values - self.base_pulse_numbers, dtype=float)
        component = (
            tuple(tuple(0 for _ in session.toa_indices) for session in self.sessions)
            if local_integer_offsets is None
            else tuple(
                tuple(int(value) for value in offsets)
                for offsets in local_integer_offsets
            )
        )
        return profile_fixed_local_pulse_component(
            residuals,
            self._phase_sigmas(),
            self.sessions,
            component,
        )


@dataclass(frozen=True)
class DualLatticeCellSearchConfig:
    starts_per_cell: int = 4
    maximum_function_evaluations: int = 160
    local_integer_radius: int = 0
    local_count_policy: str = "reference_component"
    parameter_tolerance: float = 1.0e-9
    objective_tolerance: float = 1.0e-9


@dataclass(frozen=True)
class DualLatticeCellMode:
    cell_id: str
    start_index: int
    normalized_values: np.ndarray
    slow_values: np.ndarray
    fast_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    function_evaluations: int
    converged: bool
    status: int
    message: str
    on_cell_boundary: bool
    normalized_lower: np.ndarray | None = None
    normalized_upper: np.ndarray | None = None


@dataclass(frozen=True)
class DualLatticeCoherentConfig:
    finite_difference_step: float = 2.0e-4
    path_count: int = 64
    maximum_closure_evaluations: int = 240
    prior_fraction_of_cell: float = 0.5
    lattice: SpinIntegerMarginalConfig = SpinIntegerMarginalConfig(
        initial_lattice_candidates=64,
        maximum_lattice_candidates=512,
        lattice_log_mass_tolerance=1.0e-5,
        maximum_lattice_nodes=2_000_000,
    )


@dataclass(frozen=True)
class DualLatticeCoherentMode:
    cell_id: str
    integer_offsets: tuple[int, ...]
    squared_mahalanobis: float
    posterior_weight: float
    linear_start_normalized_values: np.ndarray
    normalized_values: np.ndarray
    parameter_values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    shape_chi2: float
    phase_chi2: float
    total_chi2: float
    function_evaluations: int
    converged: bool
    status: int
    message: str
    on_cell_boundary: bool


def search_orbital_cycle_cell(
    *,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    config: DualLatticeCellSearchConfig | None = None,
    seed: int = 0,
) -> tuple[DualLatticeCellMode, ...]:
    """Find complete-data local-rate modes without ranking away the cell."""

    active = config or DualLatticeCellSearchConfig()
    starts_count = max(int(active.starts_per_cell), 1)
    maximum_evaluations = max(int(active.maximum_function_evaluations), 8)
    normalized_lower = np.full(8, -1.0, dtype=float)
    normalized_upper = np.full(8, 1.0, dtype=float)
    slow_midpoint = 0.5 * (support.slow_lower + support.slow_upper)
    normalized_lower[0] = (cell.lower - slow_midpoint[0]) / support.slow_scale[0]
    normalized_upper[0] = (cell.upper - slow_midpoint[0]) / support.slow_scale[0]
    epsilon = 32.0 * np.finfo(float).eps
    normalized_lower += epsilon
    normalized_upper -= epsilon

    center = support.normalize(support.slow_center, support.fast_center)
    center[0] = 0.5 * (normalized_lower[0] + normalized_upper[0])
    center = np.clip(center, normalized_lower, normalized_upper)
    starts = [center]
    rng = np.random.default_rng(max(int(seed), 0))
    index = {name: evaluator.parameter_names.index(name) for name in evaluator.parameter_names}
    for _ in range(1, starts_count):
        physical = rng.uniform(support.physical_lower, support.physical_upper)
        sampled_c_orb = rng.uniform(cell.lower, cell.upper)
        physical[index["PB"]] = coordinates.observation_span_days / sampled_c_orb
        sampled_slow, sampled_fast = coordinates.split_physical(physical)
        sample = support.normalize(sampled_slow, sampled_fast)
        starts.append(np.clip(sample, normalized_lower, normalized_upper))

    modes: list[DualLatticeCellMode] = []
    for start_index, start in enumerate(starts):
        fixed_local_component: tuple[tuple[int, ...], ...] | None = None
        if active.local_count_policy == "initial_map_component":
            start_slow, start_fast = support.denormalize(start)
            start_physical = coordinates.combine_physical(start_slow, start_fast)
            start_profile = evaluator.evaluate_values(
                start_physical,
                local_integer_radius=max(int(active.local_integer_radius), 0),
                maximum_retained_local_modes=1,
            )
            fixed_local_component = start_profile.local_integer_offsets

        def residual_vector(normalized: np.ndarray) -> np.ndarray:
            try:
                slow, fast = support.denormalize(normalized)
                physical = coordinates.combine_physical(slow, fast)
                if not support.contains_physical(physical):
                    raise ValueError("The trial lies outside audited physical support.")
                if active.local_count_policy == "reference_component":
                    profile = evaluator.evaluate_fixed_local_component(physical)
                elif active.local_count_policy == "initial_map_component":
                    assert fixed_local_component is not None
                    profile = evaluator.evaluate_fixed_local_component(
                        physical,
                        fixed_local_component,
                    )
                elif active.local_count_policy == "dynamic_map":
                    profile = evaluator.evaluate_values(
                        physical,
                        local_integer_radius=max(int(active.local_integer_radius), 0),
                        maximum_retained_local_modes=1,
                    )
                else:
                    raise ValueError("Unknown local-count policy.")
                residuals = np.asarray(profile.normalized_local_residuals, dtype=float)
                if residuals.shape != (len(evaluator.toas.table),) or not np.all(
                    np.isfinite(residuals)
                ):
                    raise ValueError("PINT returned an invalid local residual vector.")
                return residuals
            except (ArithmeticError, RuntimeError, ValueError):
                return np.full(len(evaluator.toas.table), 1.0e8, dtype=float)

        result = least_squares(
            residual_vector,
            start,
            bounds=(normalized_lower, normalized_upper),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=max(float(active.objective_tolerance), np.finfo(float).eps),
            xtol=max(float(active.parameter_tolerance), np.finfo(float).eps),
            gtol=max(float(active.objective_tolerance), np.finfo(float).eps),
            max_nfev=maximum_evaluations,
        )
        slow, fast = support.denormalize(result.x)
        physical = coordinates.combine_physical(slow, fast)
        if support.contains_physical(physical):
            if active.local_count_policy == "reference_component":
                profile = evaluator.evaluate_fixed_local_component(physical)
            elif active.local_count_policy == "initial_map_component":
                assert fixed_local_component is not None
                profile = evaluator.evaluate_fixed_local_component(
                    physical,
                    fixed_local_component,
                )
            else:
                profile = evaluator.evaluate_values(
                    physical,
                    local_integer_radius=max(int(active.local_integer_radius), 0),
                    maximum_retained_local_modes=1,
                )
            shape_chi2 = float(profile.local_chi2)
            local_integer_offsets = profile.local_integer_offsets
        else:
            shape_chi2 = math.inf
            local_integer_offsets = tuple()
        boundary_tolerance = max(1.0e-6 * cell.width, 1.0e-10)
        modes.append(
            DualLatticeCellMode(
                cell_id=cell.cell_id,
                start_index=start_index,
                normalized_values=np.asarray(result.x, dtype=float).copy(),
                slow_values=slow,
                fast_values=fast,
                parameter_values=physical,
                local_integer_offsets=local_integer_offsets,
                shape_chi2=shape_chi2,
                function_evaluations=int(result.nfev),
                converged=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                on_cell_boundary=bool(
                    slow[0] - cell.lower <= boundary_tolerance
                    or cell.upper - slow[0] <= boundary_tolerance
                ),
                normalized_lower=normalized_lower.copy(),
                normalized_upper=normalized_upper.copy(),
            )
        )
    modes.sort(key=lambda item: (item.shape_chi2, item.start_index))
    return tuple(modes)


def normalized_bounds_for_cell(
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(8, -1.0, dtype=float)
    upper = np.full(8, 1.0, dtype=float)
    slow_midpoint = 0.5 * (support.slow_lower + support.slow_upper)
    lower[0] = (cell.lower - slow_midpoint[0]) / support.slow_scale[0]
    upper[0] = (cell.upper - slow_midpoint[0]) / support.slow_scale[0]
    epsilon = 32.0 * np.finfo(float).eps
    return lower + epsilon, upper - epsilon


def _normalized_bounds_for_cell(
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility alias for pre-library research runners."""

    return normalized_bounds_for_cell(support, cell)


def _normalized_bounds_for_mode(
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    mode: DualLatticeCellMode,
) -> tuple[np.ndarray, np.ndarray]:
    cell_lower, cell_upper = _normalized_bounds_for_cell(support, cell)
    if mode.normalized_lower is None and mode.normalized_upper is None:
        return cell_lower, cell_upper
    if mode.normalized_lower is None or mode.normalized_upper is None:
        raise ValueError("A mode must provide both normalized chart bounds.")
    lower = _finite_vector(
        mode.normalized_lower,
        dimension=8,
        label="Mode normalized lower bounds",
    )
    upper = _finite_vector(
        mode.normalized_upper,
        dimension=8,
        label="Mode normalized upper bounds",
    )
    tolerance = 1.0e-12
    if np.any(lower >= upper):
        raise ValueError("Mode normalized bounds must have positive width.")
    if np.any(lower < cell_lower - tolerance) or np.any(
        upper > cell_upper + tolerance
    ):
        raise ValueError("Mode normalized bounds must lie inside its orbital cell.")
    reference = np.asarray(mode.normalized_values, dtype=float)
    if np.any(reference < lower - tolerance) or np.any(reference > upper + tolerance):
        raise ValueError("Mode reference lies outside its normalized chart bounds.")
    return lower.copy(), upper.copy()


def _session_shape_profile(profile: TwoLevelPulseProfile) -> SessionShapeProfile:
    normalized = np.asarray(profile.normalized_local_residuals, dtype=float)
    return SessionShapeProfile(
        projected_phase_residuals=normalized.copy(),
        normalized_residuals=normalized,
        session_offsets=np.asarray(profile.session_intercepts, dtype=float),
        session_offset_variances=np.asarray(
            profile.session_intercept_variances,
            dtype=float,
        ),
        chi2=float(profile.local_chi2),
        dof=max(int(normalized.size - profile.session_intercepts.size), 0),
        session_count=int(profile.session_intercepts.size),
    )


def _evaluate_fixed_normalized(
    *,
    normalized_values: np.ndarray,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    local_integer_offsets: tuple[tuple[int, ...], ...],
) -> tuple[np.ndarray, np.ndarray, TwoLevelPulseProfile]:
    slow, fast = support.denormalize(normalized_values)
    physical = coordinates.combine_physical(slow, fast)
    if not support.contains_physical(physical):
        raise ValueError("Coherent trial lies outside audited physical support.")
    profile = evaluator.evaluate_fixed_local_component(
        physical,
        local_integer_offsets,
    )
    return physical, slow, profile


def _project_to_connected_physical_segment(
    *,
    reference: np.ndarray,
    target: np.ndarray,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
) -> np.ndarray:
    """Keep a posterior proposal in the reference's physical support component."""

    origin = _finite_vector(reference, dimension=8, label="Segment reference")
    proposal = _finite_vector(target, dimension=8, label="Segment target")

    def is_physical(alpha: float) -> bool:
        normalized = origin + float(alpha) * (proposal - origin)
        try:
            slow, fast = support.denormalize(normalized)
            physical = coordinates.combine_physical(slow, fast)
        except (ArithmeticError, RuntimeError, ValueError):
            return False
        return support.contains_physical(physical)

    if not is_physical(0.0):
        raise ValueError("The coherent reference is outside physical support.")
    if is_physical(1.0):
        return proposal.copy()

    valid_alpha = 0.0
    invalid_alpha = 1.0
    for alpha in np.linspace(0.0, 1.0, 65, dtype=float)[1:]:
        if not is_physical(float(alpha)):
            invalid_alpha = float(alpha)
            break
        valid_alpha = float(alpha)
    for _ in range(60):
        midpoint = 0.5 * (valid_alpha + invalid_alpha)
        if is_physical(midpoint):
            valid_alpha = midpoint
        else:
            invalid_alpha = midpoint
    interior_alpha = max(valid_alpha - 1.0e-6 * max(valid_alpha, 1.0), 0.0)
    projected = origin + interior_alpha * (proposal - origin)
    if not is_physical(interior_alpha):
        return origin.copy()
    return projected


def linearize_dual_lattice_cell_mode(
    *,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    mode: DualLatticeCellMode,
    finite_difference_step: float = 2.0e-4,
) -> SpinOrbitLinearization:
    """Linearize local shape and global session anchors in all eight coordinates."""

    reference = np.asarray(mode.normalized_values, dtype=float)
    lower, upper = _normalized_bounds_for_mode(support, cell, mode)
    _, _, profile = _evaluate_fixed_normalized(
        normalized_values=reference,
        evaluator=evaluator,
        coordinates=coordinates,
        support=support,
        local_integer_offsets=mode.local_integer_offsets,
    )
    shape = np.asarray(profile.normalized_local_residuals, dtype=float)
    offsets = np.asarray(profile.session_intercepts, dtype=float)
    shape_jacobian = np.empty((shape.size, 8), dtype=float)
    offset_jacobian = np.empty((offsets.size, 8), dtype=float)
    step = max(float(finite_difference_step), 1.0e-7)

    def valid_probe(index: int, direction: float) -> tuple[float, TwoLevelPulseProfile] | None:
        available = (
            float(upper[index] - reference[index])
            if direction > 0.0
            else float(reference[index] - lower[index])
        )
        probe_step = min(step, available)
        minimum_step = max(
            64.0 * np.finfo(float).eps * max(abs(float(reference[index])), 1.0),
            step * 2.0**-20,
        )
        while probe_step >= minimum_step:
            probe = reference.copy()
            probe[index] += direction * probe_step
            try:
                _, _, probe_profile = _evaluate_fixed_normalized(
                    normalized_values=probe,
                    evaluator=evaluator,
                    coordinates=coordinates,
                    support=support,
                    local_integer_offsets=mode.local_integer_offsets,
                )
                return direction * probe_step, probe_profile
            except (ArithmeticError, RuntimeError, ValueError):
                probe_step *= 0.5
        return None

    for index in range(8):
        plus_probe = valid_probe(index, 1.0)
        minus_probe = valid_probe(index, -1.0)
        if plus_probe is not None and minus_probe is not None:
            plus_step, plus_profile = plus_probe
            minus_step, minus_profile = minus_probe
            denominator = plus_step - minus_step
            shape_delta = (
                plus_profile.normalized_local_residuals
                - minus_profile.normalized_local_residuals
            )
            offset_delta = (
                plus_profile.session_intercepts - minus_profile.session_intercepts
            )
        elif plus_probe is not None:
            plus_step, plus_profile = plus_probe
            denominator = plus_step
            shape_delta = plus_profile.normalized_local_residuals - shape
            offset_delta = plus_profile.session_intercepts - offsets
        elif minus_probe is not None:
            minus_step, minus_profile = minus_probe
            denominator = -minus_step
            shape_delta = shape - minus_profile.normalized_local_residuals
            offset_delta = offsets - minus_profile.session_intercepts
        else:
            raise ValueError(
                f"No physical finite-difference probe exists for coordinate {index}."
            )
        shape_jacobian[:, index] = shape_delta / denominator
        offset_jacobian[:, index] = offset_delta / denominator

    relative_offsets = offsets[1:] - offsets[0]
    relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
    variances = np.asarray(profile.session_intercept_variances, dtype=float)
    relative_covariance = np.diag(variances[1:])
    relative_covariance += float(variances[0])
    return SpinOrbitLinearization(
        profile=_session_shape_profile(profile),
        spin_reference=reference.copy(),
        shape_jacobian=shape_jacobian,
        relative_session_offsets=relative_offsets,
        relative_session_jacobian=relative_jacobian,
        relative_offset_covariance=relative_covariance,
    )


def solve_coherent_paths_in_cell(
    *,
    evaluator: PintDualLatticePhaseEvaluator,
    coordinates: DualLatticeCoordinateSystem,
    support: DualLatticeSupport,
    cell: OrbitalCycleCell,
    mode: DualLatticeCellMode,
    config: DualLatticeCoherentConfig | None = None,
) -> tuple[DualLatticeCoherentMode, ...]:
    """Marginalize the full pulse lattice, then exact-close each retained path."""

    active = config or DualLatticeCoherentConfig()
    linearization = linearize_dual_lattice_cell_mode(
        evaluator=evaluator,
        coordinates=coordinates,
        support=support,
        cell=cell,
        mode=mode,
        finite_difference_step=float(active.finite_difference_step),
    )
    lower, upper = _normalized_bounds_for_mode(support, cell, mode)
    reference = np.asarray(mode.normalized_values, dtype=float)
    half_width = 0.5 * (upper - lower)
    prior_sigma = np.maximum(
        half_width * max(float(active.prior_fraction_of_cell), 1.0e-3),
        1.0e-5,
    )
    marginal = marginalize_spin_and_integer_paths(
        linearization,
        spin_prior_mean=np.zeros(8, dtype=float),
        spin_prior_covariance=np.diag(np.square(prior_sigma)),
        config=active.lattice,
    )
    retained = marginal.path_modes[: max(int(active.path_count), 1)]
    exact_modes: list[DualLatticeCoherentMode] = []

    for lattice_mode in retained:
        path = tuple(int(value) for value in lattice_mode.integer_offsets)
        posterior_target = reference + np.asarray(
            lattice_mode.spin_posterior_mean,
            dtype=float,
        )
        posterior_target = np.clip(posterior_target, lower, upper)
        start = _project_to_connected_physical_segment(
            reference=reference,
            target=posterior_target,
            coordinates=coordinates,
            support=support,
        )

        def exact_residuals(normalized: np.ndarray) -> np.ndarray:
            try:
                _, _, profile = _evaluate_fixed_normalized(
                    normalized_values=normalized,
                    evaluator=evaluator,
                    coordinates=coordinates,
                    support=support,
                    local_integer_offsets=mode.local_integer_offsets,
                )
                relative = profile.session_intercepts[1:] - profile.session_intercepts[0]
                covariance = np.diag(profile.session_intercept_variances[1:])
                covariance += float(profile.session_intercept_variances[0])
                phase = np.linalg.solve(
                    np.linalg.cholesky(covariance),
                    relative - np.asarray(path, dtype=float),
                )
                return np.concatenate((profile.normalized_local_residuals, phase))
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                return np.full(len(evaluator.toas.table) + len(path), 1.0e8)

        try:
            result = least_squares(
                exact_residuals,
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
            physical, slow, profile = _evaluate_fixed_normalized(
                normalized_values=np.asarray(result.x, dtype=float),
                evaluator=evaluator,
                coordinates=coordinates,
                support=support,
                local_integer_offsets=mode.local_integer_offsets,
            )
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
        relative = profile.session_intercepts[1:] - profile.session_intercepts[0]
        covariance = np.diag(profile.session_intercept_variances[1:])
        covariance += float(profile.session_intercept_variances[0])
        delta = relative - np.asarray(path, dtype=float)
        phase_chi2 = float(delta @ np.linalg.solve(covariance, delta))
        boundary_tolerance = max(1.0e-6 * cell.width, 1.0e-10)
        exact_modes.append(
            DualLatticeCoherentMode(
                cell_id=cell.cell_id,
                integer_offsets=path,
                squared_mahalanobis=float(lattice_mode.squared_mahalanobis),
                posterior_weight=float(lattice_mode.posterior_weight),
                linear_start_normalized_values=start,
                normalized_values=np.asarray(result.x, dtype=float).copy(),
                parameter_values=physical,
                local_integer_offsets=profile.local_integer_offsets,
                shape_chi2=float(profile.local_chi2),
                phase_chi2=phase_chi2,
                total_chi2=float(profile.local_chi2 + phase_chi2),
                function_evaluations=int(result.nfev),
                converged=bool(result.success),
                status=int(result.status),
                message=str(result.message),
                on_cell_boundary=bool(
                    slow[0] - cell.lower <= boundary_tolerance
                    or cell.upper - slow[0] <= boundary_tolerance
                ),
            )
        )
    exact_modes.sort(key=lambda item: (item.total_chi2, item.squared_mahalanobis))
    return tuple(exact_modes)


def profile_two_level_pulse_lattice(
    phase_residuals: Sequence[float],
    phase_sigmas: Sequence[float],
    sessions: Sequence[SourceSession],
    *,
    integer_radius: int = 0,
    maximum_retained_modes: int | None = None,
) -> TwoLevelPulseProfile:
    """Profile local integers while leaving session-anchor integers unresolved."""

    residuals = np.asarray(phase_residuals, dtype=float)
    sigmas = np.asarray(phase_sigmas, dtype=float)
    if residuals.ndim != 1 or sigmas.shape != residuals.shape:
        raise ValueError("Global phase residuals and sigmas must be aligned vectors.")
    session_tuple = tuple(sessions)
    covered = tuple(index for session in session_tuple for index in session.toa_indices)
    if covered != tuple(range(residuals.size)):
        raise ValueError("Source sessions must cover the phase vector in TOA order.")

    normalized = np.empty_like(residuals)
    intercepts: list[float] = []
    variances: list[float] = []
    integer_modes: list[tuple[int, ...]] = []
    local_chi2 = 0.0
    local_log_likelihood = 0.0
    for session in session_tuple:
        indices = np.asarray(session.toa_indices, dtype=int)
        mixture = enumerate_local_pulse_count_mixture(
            residuals[indices],
            sigmas[indices],
            integer_radius=integer_radius,
            maximum_retained_modes=maximum_retained_modes,
        )
        best = mixture.best
        integers = np.asarray(best.relative_integer_offsets, dtype=float)
        shifted = residuals[indices] - integers
        normalized[indices] = (shifted - best.intercept) / sigmas[indices]
        weights = 1.0 / np.square(sigmas[indices])
        intercepts.append(best.intercept)
        variances.append(float(1.0 / np.sum(weights)))
        integer_modes.append(best.relative_integer_offsets)
        local_chi2 += best.chi2
        local_log_likelihood += mixture.log_likelihood
    return TwoLevelPulseProfile(
        normalized_local_residuals=normalized,
        session_intercepts=np.asarray(intercepts, dtype=float),
        session_intercept_variances=np.asarray(variances, dtype=float),
        local_integer_offsets=tuple(integer_modes),
        local_chi2=float(local_chi2),
        local_log_likelihood=float(local_log_likelihood),
    )


def profile_fixed_local_pulse_component(
    phase_residuals: Sequence[float],
    phase_sigmas: Sequence[float],
    sessions: Sequence[SourceSession],
    local_integer_offsets: Sequence[Sequence[int]],
) -> TwoLevelPulseProfile:
    """Profile session anchors for one fixed set of within-session integers."""

    residuals = np.asarray(phase_residuals, dtype=float)
    sigmas = np.asarray(phase_sigmas, dtype=float)
    session_tuple = tuple(sessions)
    integer_tuple = tuple(
        tuple(int(value) for value in offsets) for offsets in local_integer_offsets
    )
    if residuals.ndim != 1 or sigmas.shape != residuals.shape:
        raise ValueError("Fixed-component residuals and sigmas must be aligned vectors.")
    if len(integer_tuple) != len(session_tuple):
        raise ValueError("A fixed local integer component is required for every session.")
    covered = tuple(index for session in session_tuple for index in session.toa_indices)
    if covered != tuple(range(residuals.size)):
        raise ValueError("Source sessions must cover every fixed-component residual.")

    normalized = np.empty_like(residuals)
    intercepts: list[float] = []
    variances: list[float] = []
    chi2 = 0.0
    for session, offsets in zip(session_tuple, integer_tuple, strict=True):
        indices = np.asarray(session.toa_indices, dtype=int)
        integers = np.asarray(offsets, dtype=float)
        if integers.shape != indices.shape or int(integers[0]) != 0:
            raise ValueError(
                "Each local component must align with its session and fix its anchor to zero."
            )
        weights = 1.0 / np.square(sigmas[indices])
        shifted = residuals[indices] - integers
        intercept = float(np.sum(weights * shifted) / np.sum(weights))
        local = (shifted - intercept) / sigmas[indices]
        normalized[indices] = local
        chi2 += float(local @ local)
        intercepts.append(intercept)
        variances.append(float(1.0 / np.sum(weights)))
    return TwoLevelPulseProfile(
        normalized_local_residuals=normalized,
        session_intercepts=np.asarray(intercepts, dtype=float),
        session_intercept_variances=np.asarray(variances, dtype=float),
        local_integer_offsets=integer_tuple,
        local_chi2=float(chi2),
        local_log_likelihood=float(-0.5 * chi2),
    )


__all__ = [
    "DUAL_LATTICE_FAST_NAMES",
    "DUAL_LATTICE_FIT_PARAMETERS",
    "DUAL_LATTICE_SLOW_NAMES",
    "DualLatticeCoordinateSystem",
    "DualLatticeSearchContext",
    "DualLatticeCellMode",
    "DualLatticeCellSearchConfig",
    "DualLatticeCoherentConfig",
    "DualLatticeCoherentMode",
    "DualLatticeSupport",
    "LocalPulseCountMixture",
    "LocalPulseCountMode",
    "OrbitalCycleCell",
    "PintDualLatticePhaseEvaluator",
    "SourceSession",
    "TwoLevelPulseProfile",
    "circular_session_anchor_residuals",
    "enumerate_local_pulse_count_mixture",
    "build_dual_lattice_support",
    "build_dual_lattice_search_context",
    "infer_contiguous_source_sessions",
    "profile_two_level_pulse_lattice",
    "profile_fixed_local_pulse_component",
    "linearize_dual_lattice_cell_mode",
    "normalized_bounds_for_cell",
    "search_orbital_cycle_cell",
    "solve_coherent_paths_in_cell",
    "tile_orbital_cycle_support",
]
