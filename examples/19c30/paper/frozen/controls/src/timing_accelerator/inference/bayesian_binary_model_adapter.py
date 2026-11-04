from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import qmc


ELL1_BINARY_MODELS = ("ELL1", "ELL1H", "ELL1K")
ECCENTRIC_BINARY_MODELS = ("BT", "DD", "DDS", "DDGR", "DDH", "DDK")
PIECEWISE_BINARY_MODELS = ("BT_PIECEWISE",)
SUPPORTED_BINARY_MODELS = (
    *ELL1_BINARY_MODELS,
    *ECCENTRIC_BINARY_MODELS,
    *PIECEWISE_BINARY_MODELS,
)


@dataclass(frozen=True)
class BinaryModelCapability:
    model_name: str
    family: str
    epoch_parameter: str
    efficient_coordinates: tuple[str, ...]
    exact_pint_likelihood: bool = True
    integer_path_marginalization: bool = True
    phase_noise_annealing: bool = True


def _capability(
    model_name: str,
    *,
    family: str,
    epoch_parameter: str,
    efficient_coordinates: Sequence[str],
) -> BinaryModelCapability:
    return BinaryModelCapability(
        model_name=model_name,
        family=family,
        epoch_parameter=epoch_parameter,
        efficient_coordinates=tuple(efficient_coordinates),
    )


BINARY_MODEL_CAPABILITIES: Mapping[str, BinaryModelCapability] = {
    name: _capability(
        name,
        family="ell1",
        epoch_parameter="TASC",
        efficient_coordinates=(
            "C_ORB",
            "PHI_REF",
            "F0_REF",
            "PHASE_A1",
            "PHASE_EPS1",
            "PHASE_EPS2",
            "SENSITIVITY_SCALED_EXTRAS",
        ),
    )
    for name in ELL1_BINARY_MODELS
}
BINARY_MODEL_CAPABILITIES = {
    **BINARY_MODEL_CAPABILITIES,
    **{
        name: _capability(
            name,
            family="eccentric",
            epoch_parameter="T0",
            efficient_coordinates=(
                "C_ORB",
                "PHI_REF",
                "F0_REF",
                "ECC_ANGLE",
                "DOMEGA",
                "ORBIT_U",
                "ORBIT_V",
                "SENSITIVITY_SCALED_EXTRAS",
            ),
        )
        for name in ECCENTRIC_BINARY_MODELS
    },
    "BT_PIECEWISE": _capability(
        "BT_PIECEWISE",
        family="bt_piecewise",
        epoch_parameter="T0",
        efficient_coordinates=(
            "C_ORB",
            "PHI_REF",
            "F0_REF",
            "ECC_ANGLE",
            "DOMEGA",
            "ORBIT_U",
            "ORBIT_V",
            "PIECEWISE_ORBIT_OFFSETS",
            "SENSITIVITY_SCALED_EXTRAS",
        ),
    ),
}


def canonical_binary_model_name(value: object) -> str:
    name = str(value).strip().upper()
    aliases = {
        "BTPIECEWISE": "BT_PIECEWISE",
        "BT-PIECEWISE": "BT_PIECEWISE",
        "ELL1K": "ELL1K",
    }
    return aliases.get(name, name)


def detect_binary_model_name(model: object) -> str:
    if "BINARY" not in model.params:
        raise ValueError("APOLLO binary-lattice search requires a BINARY model.")
    name = canonical_binary_model_name(model["BINARY"].value)
    if name not in BINARY_MODEL_CAPABILITIES:
        supported = ", ".join(SUPPORTED_BINARY_MODELS)
        raise ValueError(f"Unsupported PINT binary model '{name}'. Supported: {supported}.")
    return name


def _wrap_radians(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _wrap_degrees_near(value: float, center: float) -> float:
    return float(center + (value - center + 180.0) % 360.0 - 180.0)


def _finite_vector(values: Sequence[float], dimension: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector of length {dimension}.")
    return result


def _numeric_parameter_value(model: object, name: str) -> float | None:
    if name not in model.params:
        return None
    value = model[name].value
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _period_days(values: Mapping[str, float]) -> float:
    pb = values.get("PB")
    if pb is not None and np.isfinite(pb) and pb > 0.0:
        return float(pb)
    fb0 = values.get("FB0")
    if fb0 is not None and np.isfinite(fb0) and fb0 > 0.0:
        return float(1.0 / (fb0 * 86400.0))
    raise ValueError("The binary model must provide a positive PB or FB0.")


def _parameter_half_width(
    name: str,
    center: float,
    *,
    span_days: float,
    pb_days: float,
    uncertainty: float | None,
    astrometric_bound_arcsec: float,
) -> float:
    span_seconds = span_days * 86400.0
    known = {
        "RAJ": astrometric_bound_arcsec / 15.0 / 3600.0,
        "DECJ": astrometric_bound_arcsec / 3600.0,
        "F0": max(abs(center) * 1.0e-5, 5.0 / span_seconds),
        "F1": max(abs(center) * 20.0, 1.0e-10),
        "F2": max(abs(center) * 20.0, 60.0 / span_seconds**3),
        "PB": max(abs(pb_days) * 0.03, 5.0e-4),
        "FB0": max(abs(center) * 0.03, 5.0e-12),
        "A1": max(abs(center) * 0.08, 0.1),
        "T0": 0.55 * pb_days,
        "TASC": 0.55 * pb_days,
        "ECC": max(abs(center) * 5.0, 0.05),
        "EPS1": max(abs(center) * 5.0, 5.0e-3),
        "EPS2": max(abs(center) * 5.0, 5.0e-3),
        "OM": 180.0,
        "OMDOT": max(abs(center) * 20.0, 10.0),
        "PBDOT": max(abs(center) * 20.0, 1.0e-10),
        "A1DOT": max(abs(center) * 20.0, 1.0e-10),
        "EDOT": max(abs(center) * 20.0, 1.0e-12),
        "EPS1DOT": max(abs(center) * 20.0, 1.0e-12),
        "EPS2DOT": max(abs(center) * 20.0, 1.0e-12),
        "LNEDOT": max(abs(center) * 10.0, 1.0e-3),
        "M2": max(abs(center) * 0.5, 0.5),
        "MTOT": max(abs(center) * 0.5, 1.0),
        "SINI": 0.5,
        "STIGMA": max(abs(center) * 0.5, 0.1),
        "SHAPMAX": max(abs(center) * 0.5, 0.1),
        "H3": max(abs(center) * 5.0, 1.0e-5),
        "H4": max(abs(center) * 5.0, 1.0e-5),
        "GAMMA": max(abs(center) * 5.0, 1.0e-2),
        "KIN": 90.0,
        "KOM": 180.0,
    }
    width = known.get(name, max(abs(center) * 0.5, 1.0e-8))
    if uncertainty is not None and np.isfinite(uncertainty) and uncertainty > 0.0:
        width = max(width, 10.0 * abs(float(uncertainty)))
    return float(width)


def build_binary_physical_support(
    *,
    model: object,
    parameter_names: Sequence[str],
    observation_span_days: float,
    width_scale: float = 1.0,
    astrometric_bound_arcsec: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(str(name).upper() for name in parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Binary search parameters must be unique and nonempty.")
    span = float(observation_span_days)
    scale = float(width_scale)
    if not np.isfinite(span) or span <= 0.0 or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Binary support span and width scale must be positive.")
    base_values = {
        name: value
        for name in model.params
        if (value := _numeric_parameter_value(model, name)) is not None
    }
    pb_days = _period_days(base_values)
    center = np.asarray(
        [
            base_values[name]
            if name in base_values
            else (_raise_missing_numeric(name))
            for name in names
        ],
        dtype=float,
    )
    lower = center.copy()
    upper = center.copy()
    for index, name in enumerate(names):
        parameter = model[name]
        uncertainty = getattr(parameter, "uncertainty_value", None)
        width = scale * _parameter_half_width(
            name,
            float(center[index]),
            span_days=span,
            pb_days=pb_days,
            uncertainty=float(uncertainty) if uncertainty is not None else None,
            astrometric_bound_arcsec=float(astrometric_bound_arcsec),
        )
        lower[index] -= width
        upper[index] += width

    by_name = {name: index for index, name in enumerate(names)}
    for name in ("F0", "PB", "FB0", "A1", "M2", "MTOT", "H3", "STIGMA", "SHAPMAX"):
        if name in by_name:
            lower[by_name[name]] = max(lower[by_name[name]], np.finfo(float).tiny)
    if "ECC" in by_name:
        lower[by_name["ECC"]] = max(lower[by_name["ECC"]], 0.0)
        upper[by_name["ECC"]] = min(upper[by_name["ECC"]], 1.0 - 1.0e-12)
    if "SINI" in by_name:
        lower[by_name["SINI"]] = max(lower[by_name["SINI"]], 0.0)
        upper[by_name["SINI"]] = min(upper[by_name["SINI"]], 1.0)
    if "DECJ" in by_name:
        lower[by_name["DECJ"]] = max(lower[by_name["DECJ"]], -90.0)
        upper[by_name["DECJ"]] = min(upper[by_name["DECJ"]], 90.0)
    if "KIN" in by_name:
        lower[by_name["KIN"]] = max(lower[by_name["KIN"]], 0.0)
        upper[by_name["KIN"]] = min(upper[by_name["KIN"]], 180.0)
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("The generated binary support does not contain its center.")
    return center, lower, upper


def _raise_missing_numeric(name: str) -> float:
    raise ValueError(f"Free PINT parameter '{name}' has no finite numeric value.")


@dataclass(frozen=True)
class BinaryCoordinateAdapter:
    """Invertible phase-adapted chart for one PINT binary-model family.

    Every input free parameter owns exactly one search coordinate. Core timing
    parameters are replaced by phase-adapted coordinates; model-specific free
    parameters remain in the same joint state and are normalized by support.
    """

    capability: BinaryModelCapability
    parameter_names: tuple[str, ...]
    base_values: Mapping[str, float]
    observation_span_days: float
    reference_mjd: float
    pepoch_mjd: float
    initial_epoch_mjd: float
    initial_om_degrees: float

    def __post_init__(self) -> None:
        names = tuple(str(name).upper() for name in self.parameter_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Binary coordinate parameters must be unique and nonempty.")
        if self.capability.model_name not in BINARY_MODEL_CAPABILITIES:
            raise ValueError("The binary coordinate adapter has an unknown capability.")
        required = (self.observation_span_days, self.reference_mjd, self.pepoch_mjd)
        if not all(np.isfinite(value) for value in required) or self.observation_span_days <= 0:
            raise ValueError("Binary coordinate reference values must be finite.")
        _period_days(self.base_values)
        if self.capability.epoch_parameter not in self.base_values:
            raise ValueError(
                f"{self.capability.model_name} requires {self.capability.epoch_parameter}."
            )

    @property
    def family(self) -> str:
        return self.capability.family

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        active = set(self.parameter_names)
        names: list[str] = []
        eccentric_uv = self.family != "ell1" and {"A1", "OM"}.issubset(active)
        for name in self.parameter_names:
            if name in {"PB", "FB0"}:
                names.append("C_ORB")
            elif name == self.capability.epoch_parameter:
                names.append("PHI_REF")
            elif name == "F0":
                names.append("F0_REF")
            elif name == "A1":
                names.append("ORBIT_U" if eccentric_uv else "PHASE_A1")
            elif name == "OM" and eccentric_uv:
                names.append("ORBIT_V")
            elif name == "OM":
                names.append("OMEGA_REF")
            elif name == "ECC":
                names.append("ECC_ANGLE")
            elif name == "OMDOT":
                names.append("DOMEGA")
            elif name in {"EPS1", "EPS2"}:
                names.append(f"PHASE_{name}")
            else:
                names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("The selected fit flags produce duplicate search coordinates.")
        return tuple(names)

    @property
    def orbital_coordinate_index(self) -> int | None:
        try:
            return self.coordinate_names.index("C_ORB")
        except ValueError:
            return None

    @property
    def phase_coordinate_index(self) -> int | None:
        try:
            return self.coordinate_names.index("PHI_REF")
        except ValueError:
            return None

    def _values(self, physical_values: Sequence[float]) -> dict[str, float]:
        physical = _finite_vector(physical_values, self.dimension, "Binary physical state")
        values = dict(self.base_values)
        values.update(dict(zip(self.parameter_names, physical, strict=True)))
        return values

    def to_search(self, physical_values: Sequence[float]) -> np.ndarray:
        values = self._values(physical_values)
        pb = _period_days(values)
        dt_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        f0 = values.get("F0", 1.0)
        f1 = values.get("F1", 0.0)
        f0_ref = f0 + f1 * dt_seconds
        if f0_ref <= 0.0:
            raise ValueError("Reference-epoch spin frequency must be positive.")
        epoch = values[self.capability.epoch_parameter]
        phi_ref = _wrap_radians(2.0 * math.pi * (self.reference_mjd - epoch) / pb)
        omdot = values.get("OMDOT", 0.0)
        omega_ref = math.radians(
            values.get("OM", self.initial_om_degrees)
            + omdot * (self.reference_mjd - epoch) / 365.25
        )
        phase_a1 = f0_ref * values.get("A1", 1.0)
        active = set(self.parameter_names)
        eccentric_uv = self.family != "ell1" and {"A1", "OM"}.issubset(active)
        search: list[float] = []
        for name in self.parameter_names:
            if name == "PB":
                search.append(self.observation_span_days / pb)
            elif name == "FB0":
                search.append(self.observation_span_days / pb)
            elif name == self.capability.epoch_parameter:
                search.append(phi_ref)
            elif name == "F0":
                search.append(f0_ref)
            elif name == "A1" and eccentric_uv:
                search.append(phase_a1 * math.sin(omega_ref))
            elif name == "OM" and eccentric_uv:
                search.append(phase_a1 * math.cos(omega_ref))
            elif name == "A1":
                search.append(phase_a1)
            elif name == "OM":
                search.append(_wrap_radians(omega_ref))
            elif name == "ECC":
                eccentricity = float(values[name])
                if not 0.0 <= eccentricity < 1.0:
                    raise ValueError("Eccentricity must lie in [0, 1).")
                search.append(math.asin(math.sqrt(eccentricity)))
            elif name == "OMDOT":
                search.append(float(values[name]) * self.observation_span_days / 365.25)
            elif name in {"EPS1", "EPS2"}:
                search.append(phase_a1 * float(values[name]))
            else:
                search.append(float(values[name]))
        return np.asarray(search, dtype=float)

    def from_search(self, search_values: Sequence[float]) -> np.ndarray:
        search = _finite_vector(search_values, self.dimension, "Binary search state")
        by_coordinate = dict(zip(self.coordinate_names, search, strict=True))
        values = dict(self.base_values)
        special = {
            "C_ORB",
            "PHI_REF",
            "F0_REF",
            "PHASE_A1",
            "PHASE_EPS1",
            "PHASE_EPS2",
            "ORBIT_U",
            "ORBIT_V",
            "OMEGA_REF",
            "ECC_ANGLE",
            "DOMEGA",
        }
        for parameter, coordinate in zip(
            self.parameter_names, self.coordinate_names, strict=True
        ):
            if coordinate not in special:
                values[parameter] = float(by_coordinate[coordinate])

        if "C_ORB" in by_coordinate:
            cycles = float(by_coordinate["C_ORB"])
            if cycles <= 0.0:
                raise ValueError("C_ORB must be positive.")
            pb = self.observation_span_days / cycles
            if "PB" in self.parameter_names:
                values["PB"] = pb
            else:
                values["FB0"] = 1.0 / (pb * 86400.0)
        else:
            pb = _period_days(values)

        epoch_name = self.capability.epoch_parameter
        if "PHI_REF" in by_coordinate:
            fraction = _wrap_radians(by_coordinate["PHI_REF"]) / (2.0 * math.pi)
            initial_cycles = (self.reference_mjd - self.initial_epoch_mjd) / pb
            cycle_index = int(round(initial_cycles - fraction))
            values[epoch_name] = self.reference_mjd - pb * (cycle_index + fraction)
        epoch = values[epoch_name]

        if "DOMEGA" in by_coordinate:
            values["OMDOT"] = (
                float(by_coordinate["DOMEGA"]) * 365.25 / self.observation_span_days
            )
        omdot = values.get("OMDOT", 0.0)
        if "ECC_ANGLE" in by_coordinate:
            angle = float(by_coordinate["ECC_ANGLE"])
            if not 0.0 <= angle < 0.5 * math.pi:
                raise ValueError("ECC_ANGLE lies outside its physical domain.")
            values["ECC"] = math.sin(angle) ** 2

        dt_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        f1 = values.get("F1", 0.0)
        if "F0_REF" in by_coordinate:
            f0_ref = float(by_coordinate["F0_REF"])
            values["F0"] = f0_ref - f1 * dt_seconds
        else:
            f0_ref = values.get("F0", 1.0) + f1 * dt_seconds
        if f0_ref <= 0.0:
            raise ValueError("Reference-epoch spin frequency must be positive.")

        if "ORBIT_U" in by_coordinate and "ORBIT_V" in by_coordinate:
            u_value = float(by_coordinate["ORBIT_U"])
            v_value = float(by_coordinate["ORBIT_V"])
            phase_a1 = math.hypot(u_value, v_value)
            if phase_a1 <= 0.0:
                raise ValueError("The orbital phase-amplitude vector must be nonzero.")
            values["A1"] = phase_a1 / f0_ref
            omega_ref = math.degrees(math.atan2(u_value, v_value))
            om = omega_ref - omdot * (self.reference_mjd - epoch) / 365.25
            values["OM"] = _wrap_degrees_near(om, self.initial_om_degrees)
        else:
            phase_a1 = float(by_coordinate.get("PHASE_A1", f0_ref * values.get("A1", 1.0)))
            if "PHASE_A1" in by_coordinate:
                if phase_a1 <= 0.0:
                    raise ValueError("PHASE_A1 must be positive.")
                values["A1"] = phase_a1 / f0_ref
            if "OMEGA_REF" in by_coordinate:
                omega_ref = math.degrees(_wrap_radians(by_coordinate["OMEGA_REF"]))
                om = omega_ref - omdot * (self.reference_mjd - epoch) / 365.25
                values["OM"] = _wrap_degrees_near(om, self.initial_om_degrees)

        for name in ("EPS1", "EPS2"):
            coordinate = f"PHASE_{name}"
            if coordinate in by_coordinate:
                if phase_a1 <= 0.0:
                    raise ValueError("ELL1 phase amplitude must be positive.")
                values[name] = float(by_coordinate[coordinate]) / phase_a1
        return np.asarray([values[name] for name in self.parameter_names], dtype=float)


@dataclass(frozen=True)
class BinaryLatticeSupport:
    adapter: BinaryCoordinateAdapter
    physical_center: np.ndarray
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    search_center: np.ndarray
    search_lower: np.ndarray
    search_upper: np.ndarray

    @property
    def dimension(self) -> int:
        return self.adapter.dimension

    @property
    def search_midpoint(self) -> np.ndarray:
        return 0.5 * (self.search_lower + self.search_upper)

    @property
    def search_scale(self) -> np.ndarray:
        return 0.5 * (self.search_upper - self.search_lower)

    def normalize_search(self, values: Sequence[float]) -> np.ndarray:
        search = _finite_vector(values, self.dimension, "Binary search coordinates")
        return (search - self.search_midpoint) / self.search_scale

    def denormalize_search(self, values: Sequence[float]) -> np.ndarray:
        normalized = _finite_vector(values, self.dimension, "Normalized binary state")
        return self.search_midpoint + self.search_scale * normalized

    def normalize_physical(self, values: Sequence[float]) -> np.ndarray:
        return self.normalize_search(self.adapter.to_search(values))

    def denormalize_physical(self, values: Sequence[float]) -> np.ndarray:
        return self.adapter.from_search(self.denormalize_search(values))

    def contains_physical(self, values: Sequence[float]) -> bool:
        physical = np.asarray(values, dtype=float)
        if physical.shape != self.physical_center.shape or not np.all(np.isfinite(physical)):
            return False
        tolerance = 64.0 * np.finfo(float).eps * np.maximum(
            np.maximum(np.abs(self.physical_lower), np.abs(self.physical_upper)), 1.0
        )
        return bool(
            np.all(physical >= self.physical_lower - tolerance)
            and np.all(physical <= self.physical_upper + tolerance)
        )


@dataclass(frozen=True)
class LatticeSupportCoordinateSystem:
    """Expose any APOLLO lattice support to generic posterior samplers.

    The represented prior is uniform in the declared physical parameter box,
    while proposals and posterior geometry use the model-adapted search chart.
    The Jacobian is exposed in normalized coordinates so generic samplers can
    preserve that physical prior after the nonlinear coordinate transform.
    """

    support: BinaryLatticeSupport

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.support.adapter.parameter_names)

    @property
    def search_center(self) -> np.ndarray:
        return self.support.search_center.copy()

    @property
    def normalized_lower(self) -> np.ndarray:
        return np.full(self.support.dimension, -1.0, dtype=float)

    @property
    def normalized_upper(self) -> np.ndarray:
        return np.full(self.support.dimension, 1.0, dtype=float)

    def normalize(self, search_values: Sequence[float]) -> np.ndarray:
        return self.support.normalize_search(search_values)

    def denormalize(self, normalized_values: Sequence[float]) -> np.ndarray:
        return self.support.denormalize_search(normalized_values)

    def to_search(self, physical_values: Sequence[float]) -> np.ndarray:
        return self.support.adapter.to_search(physical_values)

    def to_physical(self, search_values: Sequence[float]) -> np.ndarray:
        physical = self.support.adapter.from_search(search_values)
        if not self.support.contains_physical(physical):
            raise ValueError(
                "Adapted search state maps outside the declared physical support."
            )
        return physical

    def from_search(self, search_values: Sequence[float]) -> np.ndarray:
        return self.to_physical(search_values)

    def sample_prior(self, count: int, *, seed: int) -> np.ndarray:
        requested = max(int(count), 0)
        if requested == 0:
            return np.zeros((0, self.support.dimension), dtype=float)
        exponent = int(math.ceil(math.log2(requested)))
        unit = qmc.Sobol(
            d=self.support.dimension,
            scramble=True,
            seed=int(seed),
        ).random_base2(exponent)[:requested]
        physical = self.support.physical_lower + unit * (
            self.support.physical_upper - self.support.physical_lower
        )
        return np.vstack(
            [self.support.adapter.to_search(row) for row in physical]
        )

    def sample_normalized_prior(self, count: int, *, seed: int) -> np.ndarray:
        return np.vstack(
            [self.normalize(row) for row in self.sample_prior(count, seed=seed)]
        )

    def log_normalized_prior_density(
        self,
        normalized_values: Sequence[float],
    ) -> float:
        normalized = np.asarray(normalized_values, dtype=float)
        if (
            normalized.shape != (self.support.dimension,)
            or not np.all(np.isfinite(normalized))
            or np.any(normalized <= -1.0)
            or np.any(normalized >= 1.0)
        ):
            return float("-inf")
        search = self.denormalize(normalized)
        try:
            physical = self.to_physical(search)
        except ValueError:
            return float("-inf")

        step = 1.0e-5
        jacobian = np.empty((self.support.dimension, self.support.dimension))
        for index in range(self.support.dimension):
            left_step = min(step, 0.45 * (normalized[index] + 1.0))
            right_step = min(step, 0.45 * (1.0 - normalized[index]))
            if left_step <= 0.0 or right_step <= 0.0:
                return float("-inf")
            left = normalized.copy()
            right = normalized.copy()
            left[index] -= left_step
            right[index] += right_step
            left_physical = self.support.adapter.from_search(
                self.denormalize(left)
            )
            right_physical = self.support.adapter.from_search(
                self.denormalize(right)
            )
            jacobian[:, index] = (
                right_physical - left_physical
            ) / (right_step + left_step)
        sign, log_determinant = np.linalg.slogdet(jacobian)
        if sign == 0.0 or not np.isfinite(log_determinant):
            return float("-inf")
        log_physical_volume = float(
            np.sum(
                np.log(
                    self.support.physical_upper - self.support.physical_lower
                )
            )
        )
        return float(log_determinant - log_physical_volume)


def build_binary_lattice_support(
    *,
    adapter: BinaryCoordinateAdapter,
    physical_center: Sequence[float],
    physical_lower: Sequence[float],
    physical_upper: Sequence[float],
) -> BinaryLatticeSupport:
    dimension = adapter.dimension
    center = _finite_vector(physical_center, dimension, "Binary support center")
    lower = _finite_vector(physical_lower, dimension, "Binary support lower")
    upper = _finite_vector(physical_upper, dimension, "Binary support upper")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Every binary support interval must contain its center.")
    search_center = adapter.to_search(center)
    search_lower = np.empty(dimension, dtype=float)
    search_upper = np.empty(dimension, dtype=float)
    names = adapter.coordinate_names
    physical_index = {name: index for index, name in enumerate(adapter.parameter_names)}
    values_lower = dict(adapter.base_values)
    values_upper = dict(adapter.base_values)
    values_lower.update(dict(zip(adapter.parameter_names, lower, strict=True)))
    values_upper.update(dict(zip(adapter.parameter_names, upper, strict=True)))
    pb_lower = _period_days(values_lower)
    pb_upper = _period_days(values_upper)
    pb_min, pb_max = sorted((pb_lower, pb_upper))
    dt_seconds = (adapter.reference_mjd - adapter.pepoch_mjd) * 86400.0
    f0_bounds = (
        (lower[physical_index["F0"]], upper[physical_index["F0"]])
        if "F0" in physical_index
        else (adapter.base_values.get("F0", 1.0),) * 2
    )
    f1_bounds = (
        (lower[physical_index["F1"]], upper[physical_index["F1"]])
        if "F1" in physical_index
        else (adapter.base_values.get("F1", 0.0),) * 2
    )
    f0_ref_corners = [f0 + f1 * dt_seconds for f0 in f0_bounds for f1 in f1_bounds]
    f0_ref_lower = max(min(f0_ref_corners), np.finfo(float).tiny)
    f0_ref_upper = max(f0_ref_corners)
    a1_bounds = (
        (lower[physical_index["A1"]], upper[physical_index["A1"]])
        if "A1" in physical_index
        else (adapter.base_values.get("A1", 1.0),) * 2
    )
    phase_a1_lower = f0_ref_lower * min(a1_bounds)
    phase_a1_upper = f0_ref_upper * max(a1_bounds)

    for index, coordinate in enumerate(names):
        parameter = adapter.parameter_names[index]
        if coordinate == "C_ORB":
            search_lower[index] = adapter.observation_span_days / pb_max
            search_upper[index] = adapter.observation_span_days / pb_min
        elif coordinate == "PHI_REF":
            search_lower[index], search_upper[index] = -math.pi, math.pi
        elif coordinate == "F0_REF":
            search_lower[index], search_upper[index] = f0_ref_lower, f0_ref_upper
        elif coordinate == "PHASE_A1":
            search_lower[index], search_upper[index] = phase_a1_lower, phase_a1_upper
        elif coordinate in {"PHASE_EPS1", "PHASE_EPS2"}:
            eps_limit = max(abs(lower[index]), abs(upper[index]))
            limit = phase_a1_upper * eps_limit
            search_lower[index], search_upper[index] = -limit, limit
        elif coordinate in {"ORBIT_U", "ORBIT_V"}:
            search_lower[index], search_upper[index] = -phase_a1_upper, phase_a1_upper
        elif coordinate == "OMEGA_REF":
            search_lower[index], search_upper[index] = -math.pi, math.pi
        elif coordinate == "ECC_ANGLE":
            search_lower[index] = math.asin(math.sqrt(max(lower[index], 0.0)))
            search_upper[index] = math.asin(math.sqrt(min(upper[index], 1.0 - 1.0e-12)))
        elif coordinate == "DOMEGA":
            factor = adapter.observation_span_days / 365.25
            search_lower[index], search_upper[index] = lower[index] * factor, upper[index] * factor
        else:
            search_lower[index], search_upper[index] = lower[index], upper[index]
    if np.any(search_lower >= search_upper):
        raise ValueError("The phase-adapted search support has a nonpositive interval.")
    return BinaryLatticeSupport(
        adapter=adapter,
        physical_center=center.copy(),
        physical_lower=lower.copy(),
        physical_upper=upper.copy(),
        search_center=search_center,
        search_lower=search_lower,
        search_upper=search_upper,
    )


def build_binary_coordinate_adapter(
    *,
    model: object,
    parameter_names: Sequence[str],
    observation_span_days: float,
    reference_mjd: float,
) -> BinaryCoordinateAdapter:
    model_name = detect_binary_model_name(model)
    capability = BINARY_MODEL_CAPABILITIES[model_name]
    base_values = {
        name: value
        for name in model.params
        if (value := _numeric_parameter_value(model, name)) is not None
    }
    epoch = capability.epoch_parameter
    if epoch not in base_values:
        raise ValueError(f"PINT {model_name} has no finite {epoch} value.")
    pepoch = base_values.get("PEPOCH", reference_mjd)
    return BinaryCoordinateAdapter(
        capability=capability,
        parameter_names=tuple(str(name).upper() for name in parameter_names),
        base_values=base_values,
        observation_span_days=float(observation_span_days),
        reference_mjd=float(reference_mjd),
        pepoch_mjd=float(pepoch),
        initial_epoch_mjd=float(base_values[epoch]),
        initial_om_degrees=float(base_values.get("OM", 0.0)),
    )


__all__ = [
    "BINARY_MODEL_CAPABILITIES",
    "ECCENTRIC_BINARY_MODELS",
    "ELL1_BINARY_MODELS",
    "PIECEWISE_BINARY_MODELS",
    "SUPPORTED_BINARY_MODELS",
    "BinaryCoordinateAdapter",
    "BinaryLatticeSupport",
    "BinaryModelCapability",
    "build_binary_coordinate_adapter",
    "build_binary_lattice_support",
    "build_binary_physical_support",
    "canonical_binary_model_name",
    "detect_binary_model_name",
]
