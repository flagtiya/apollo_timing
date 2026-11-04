from __future__ import annotations

from apollo_timing.inference._shared_helpers import (
    _finite_vector,
    _wrap_degrees_near,
)

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from apollo_timing.inference.timing_constraints import SkyPositionConstraint, interval_roundoff


LIGHT_TRAVEL_TIME_AU_SECONDS = 499.004783836




def _numeric_parameter_value(model: object, name: str) -> float | None:
    if name not in model.params or model[name].value is None:
        return None
    try:
        value = float(model[name].value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _wrap_hours_near(value: float, center: float) -> float:
    return float(center + (value - center + 12.0) % 24.0 - 12.0)




@dataclass(frozen=True)
class IsolatedCoordinateAdapter:
    """Invertible phase-adapted chart for an isolated PINT timing model."""

    parameter_names: tuple[str, ...]
    base_values: Mapping[str, float]
    observation_span_days: float
    reference_mjd: float
    pepoch_mjd: float

    def __post_init__(self) -> None:
        names = tuple(str(name).upper() for name in self.parameter_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Isolated coordinate parameters must be unique and nonempty.")
        required = (self.observation_span_days, self.reference_mjd, self.pepoch_mjd)
        if not all(np.isfinite(value) for value in required):
            raise ValueError("Isolated coordinate reference values must be finite.")
        if self.observation_span_days <= 0.0:
            raise ValueError("Isolated coordinate span must be positive.")
        if "F0" not in self.base_values or float(self.base_values["F0"]) <= 0.0:
            raise ValueError("An isolated phase chart requires a positive F0.")

    @property
    def family(self) -> str:
        return "isolated"

    @property
    def dimension(self) -> int:
        return len(self.parameter_names)

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        names = []
        for name in self.parameter_names:
            if name == "F0":
                names.append("F0_REF")
            elif name == "F1":
                names.append("SPIN_CURVATURE")
            elif name == "RAJ":
                names.append("SKY_EAST_PHASE")
            elif name == "DECJ":
                names.append("SKY_NORTH_PHASE")
            elif name == "ELONG":
                names.append("ECLIPTIC_LONGITUDE_PHASE")
            elif name == "ELAT":
                names.append("ECLIPTIC_LATITUDE_PHASE")
            else:
                names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("The selected isolated fit flags duplicate search coordinates.")
        return tuple(names)

    @property
    def orbital_coordinate_index(self) -> None:
        return None

    @property
    def phase_coordinate_index(self) -> None:
        return None

    def _values(self, physical_values: Sequence[float]) -> dict[str, float]:
        physical = _finite_vector(
            physical_values, self.dimension, "Isolated physical state"
        )
        values = dict(self.base_values)
        values.update(dict(zip(self.parameter_names, physical, strict=True)))
        return values

    def _reference_frequency(self, values: Mapping[str, float]) -> float:
        dt_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        result = float(values["F0"]) + float(values.get("F1", 0.0)) * dt_seconds
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError("Reference-epoch spin frequency must be positive.")
        return result

    def to_search(self, physical_values: Sequence[float]) -> np.ndarray:
        values = self._values(physical_values)
        f0_ref = self._reference_frequency(values)
        span_seconds = self.observation_span_days * 86400.0
        phase_scale = f0_ref * LIGHT_TRAVEL_TIME_AU_SECONDS
        base_ra = float(self.base_values.get("RAJ", 0.0))
        base_dec = float(self.base_values.get("DECJ", 0.0))
        base_elong = float(self.base_values.get("ELONG", 0.0))
        base_elat = float(self.base_values.get("ELAT", 0.0))
        result = []
        for name in self.parameter_names:
            if name == "F0":
                result.append(f0_ref)
            elif name == "F1":
                result.append(0.5 * float(values[name]) * span_seconds**2)
            elif name == "RAJ":
                delta = math.radians(15.0 * (float(values[name]) - base_ra))
                result.append(phase_scale * math.cos(math.radians(base_dec)) * delta)
            elif name == "DECJ":
                delta = math.radians(float(values[name]) - base_dec)
                result.append(phase_scale * delta)
            elif name == "ELONG":
                delta = math.radians(float(values[name]) - base_elong)
                result.append(phase_scale * math.cos(math.radians(base_elat)) * delta)
            elif name == "ELAT":
                delta = math.radians(float(values[name]) - base_elat)
                result.append(phase_scale * delta)
            else:
                result.append(float(values[name]))
        return np.asarray(result, dtype=float)

    def from_search(self, search_values: Sequence[float]) -> np.ndarray:
        search = _finite_vector(search_values, self.dimension, "Isolated search state")
        coordinates = dict(zip(self.coordinate_names, search, strict=True))
        values = dict(self.base_values)
        span_seconds = self.observation_span_days * 86400.0
        if "SPIN_CURVATURE" in coordinates:
            values["F1"] = 2.0 * float(coordinates["SPIN_CURVATURE"]) / span_seconds**2
        dt_seconds = (self.reference_mjd - self.pepoch_mjd) * 86400.0
        if "F0_REF" in coordinates:
            values["F0"] = float(coordinates["F0_REF"]) - float(
                values.get("F1", 0.0)
            ) * dt_seconds
        f0_ref = self._reference_frequency(values)
        phase_scale = f0_ref * LIGHT_TRAVEL_TIME_AU_SECONDS

        for parameter, coordinate in zip(
            self.parameter_names, self.coordinate_names, strict=True
        ):
            value = float(coordinates[coordinate])
            if coordinate in {"F0_REF", "SPIN_CURVATURE"}:
                continue
            if coordinate == "SKY_EAST_PHASE":
                base_ra = float(self.base_values["RAJ"])
                base_dec = float(self.base_values.get("DECJ", 0.0))
                denominator = phase_scale * math.cos(math.radians(base_dec))
                values[parameter] = _wrap_hours_near(
                    base_ra + math.degrees(value / denominator) / 15.0,
                    base_ra,
                )
            elif coordinate == "SKY_NORTH_PHASE":
                base_dec = float(self.base_values["DECJ"])
                values[parameter] = base_dec + math.degrees(value / phase_scale)
            elif coordinate == "ECLIPTIC_LONGITUDE_PHASE":
                base_elong = float(self.base_values["ELONG"])
                base_elat = float(self.base_values.get("ELAT", 0.0))
                denominator = phase_scale * math.cos(math.radians(base_elat))
                values[parameter] = _wrap_degrees_near(
                    base_elong + math.degrees(value / denominator),
                    base_elong,
                )
            elif coordinate == "ECLIPTIC_LATITUDE_PHASE":
                base_elat = float(self.base_values["ELAT"])
                values[parameter] = base_elat + math.degrees(value / phase_scale)
            else:
                values[parameter] = value
        return np.asarray([values[name] for name in self.parameter_names], dtype=float)


@dataclass(frozen=True)
class IsolatedLatticeSupport:
    adapter: IsolatedCoordinateAdapter
    physical_center: np.ndarray
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    search_center: np.ndarray
    search_lower: np.ndarray
    search_upper: np.ndarray
    sky_constraint: SkyPositionConstraint | None = None

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
        search = _finite_vector(values, self.dimension, "Isolated search coordinates")
        return (search - self.search_midpoint) / self.search_scale

    def denormalize_search(self, values: Sequence[float]) -> np.ndarray:
        normalized = _finite_vector(values, self.dimension, "Normalized isolated state")
        return self.search_midpoint + self.search_scale * normalized

    def normalize_physical(self, values: Sequence[float]) -> np.ndarray:
        return self.normalize_search(self.adapter.to_search(values))

    def denormalize_physical(self, values: Sequence[float]) -> np.ndarray:
        return self.adapter.from_search(self.denormalize_search(values))

    def contains_physical(self, values: Sequence[float]) -> bool:
        physical = np.asarray(values, dtype=float)
        if physical.shape != self.physical_center.shape or not np.all(np.isfinite(physical)):
            return False
        tolerance = interval_roundoff(self.physical_lower, self.physical_upper)
        return bool(
            np.all(physical >= self.physical_lower - tolerance)
            and np.all(physical <= self.physical_upper + tolerance)
            and (self.sky_constraint is None or self.sky_constraint.contains_values(
                self.adapter.parameter_names, physical, self.adapter.base_values))
        )


def build_isolated_coordinate_adapter(
    *,
    model: object,
    parameter_names: Sequence[str],
    observation_span_days: float,
    reference_mjd: float,
) -> IsolatedCoordinateAdapter:
    base_values = {
        name: value
        for name in model.params
        if (value := _numeric_parameter_value(model, name)) is not None
    }
    return IsolatedCoordinateAdapter(
        parameter_names=tuple(str(name).upper() for name in parameter_names),
        base_values=base_values,
        observation_span_days=float(observation_span_days),
        reference_mjd=float(reference_mjd),
        pepoch_mjd=float(base_values.get("PEPOCH", reference_mjd)),
    )


def build_isolated_physical_support(
    *,
    model: object,
    parameter_names: Sequence[str],
    observation_span_days: float,
    width_scale: float = 1.0,
    astrometric_bound_arcsec: float = 300.0,
    proper_motion_half_width_mas_per_year: float = 0.0,
    minimum_frequency_half_width_hz: float = 0.0,
    minimum_frequency_derivative_half_width_hz_s: float = 0.0,
    custom_bounds: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(str(name).upper() for name in parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Isolated search parameters must be unique and nonempty.")
    span_days = float(observation_span_days)
    scale = float(width_scale)
    minimum_f0 = float(minimum_frequency_half_width_hz)
    minimum_f1 = float(minimum_frequency_derivative_half_width_hz_s)
    proper_motion_width = float(proper_motion_half_width_mas_per_year)
    if span_days <= 0.0 or scale <= 0.0:
        raise ValueError("Isolated support span and width scale must be positive.")
    if minimum_f0 < 0.0 or minimum_f1 < 0.0 or proper_motion_width < 0.0:
        raise ValueError(
            "Minimum isolated spin and proper-motion half-widths must be nonnegative."
        )
    base_values = {
        name: value
        for name in model.params
        if (value := _numeric_parameter_value(model, name)) is not None
    }
    missing = [name for name in names if name not in base_values]
    if missing:
        raise ValueError("Free isolated parameters have no numeric value: " + ", ".join(missing))
    center = np.asarray([base_values[name] for name in names], dtype=float)
    lower = center.copy()
    upper = center.copy()
    span_seconds = span_days * 86400.0
    for index, name in enumerate(names):
        value = float(center[index])
        uncertainty = getattr(model[name], "uncertainty_value", None)
        uncertainty_width = (
            10.0 * abs(float(uncertainty))
            if uncertainty is not None and np.isfinite(float(uncertainty))
            else 0.0
        )
        if name == "RAJ":
            width = astrometric_bound_arcsec / 15.0 / 3600.0
        elif name in {"DECJ", "ELONG", "ELAT"}:
            width = astrometric_bound_arcsec / 3600.0
        elif name == "F0":
            width = max(abs(value) * 1.0e-5, 5.0 / span_seconds, minimum_f0)
        elif name == "F1":
            width = max(abs(value) * 20.0, 1.0e-10, minimum_f1)
        elif name.startswith("F") and name[1:].isdigit() and 2 <= int(name[1:]) <= 12:
            order = int(name[1:])
            width = max(abs(value) * 20.0, 10.0 * math.factorial(order + 1) / span_seconds ** (order + 1))
        elif name in {"PMRA", "PMDEC"}:
            width = max(abs(value) * 2.0, proper_motion_width)
        elif name == "DM":
            width = max(abs(value) * 0.1, 1.0)
        else:
            width = max(abs(value) * 0.5, 1.0e-8)
        width = scale * max(width, uncertainty_width)
        lower[index] -= width
        upper[index] += width

    # Apply custom bounds if provided
    if custom_bounds:
        for index, name in enumerate(names):
            if name in custom_bounds:
                custom_width = float(custom_bounds[name])
                if custom_width <= 0.0:
                    raise ValueError(f"Custom bound for {name} must be positive, got {custom_width}")
                lower[index] = center[index] - custom_width
                upper[index] = center[index] + custom_width

    by_name = {name: index for index, name in enumerate(names)}
    if "F0" in by_name:
        lower[by_name["F0"]] = max(lower[by_name["F0"]], np.finfo(float).tiny)
    for name in ("DECJ", "ELAT"):
        if name in by_name:
            lower[by_name[name]] = max(lower[by_name[name]], -90.0)
            upper[by_name[name]] = min(upper[by_name[name]], 90.0)
    if np.any(lower >= upper):
        raise ValueError("Every isolated support interval must have positive width.")
    return center, lower, upper


def build_isolated_lattice_support(
    *,
    adapter: IsolatedCoordinateAdapter,
    physical_center: Sequence[float],
    physical_lower: Sequence[float],
    physical_upper: Sequence[float],
) -> IsolatedLatticeSupport:
    dimension = adapter.dimension
    center = _finite_vector(physical_center, dimension, "Isolated support center")
    lower = _finite_vector(physical_lower, dimension, "Isolated support lower")
    upper = _finite_vector(physical_upper, dimension, "Isolated support upper")
    if np.any(lower >= upper) or np.any(center < lower) or np.any(center > upper):
        raise ValueError("Every isolated support interval must contain its center.")
    search_center = adapter.to_search(center)
    search_lower = np.empty(dimension, dtype=float)
    search_upper = np.empty(dimension, dtype=float)
    physical_index = {name: index for index, name in enumerate(adapter.parameter_names)}
    dt_seconds = (adapter.reference_mjd - adapter.pepoch_mjd) * 86400.0
    f0_bounds = (
        (lower[physical_index["F0"]], upper[physical_index["F0"]])
        if "F0" in physical_index
        else (adapter.base_values["F0"],) * 2
    )
    f1_bounds = (
        (lower[physical_index["F1"]], upper[physical_index["F1"]])
        if "F1" in physical_index
        else (adapter.base_values.get("F1", 0.0),) * 2
    )
    f0_ref_corners = [f0 + f1 * dt_seconds for f0 in f0_bounds for f1 in f1_bounds]
    f0_ref_lower = max(min(f0_ref_corners), np.finfo(float).tiny)
    f0_ref_upper = max(f0_ref_corners)
    span_seconds = adapter.observation_span_days * 86400.0

    for index, (parameter, coordinate) in enumerate(
        zip(adapter.parameter_names, adapter.coordinate_names, strict=True)
    ):
        if coordinate == "F0_REF":
            bounds = (f0_ref_lower, f0_ref_upper)
        elif coordinate == "SPIN_CURVATURE":
            bounds = (
                0.5 * lower[index] * span_seconds**2,
                0.5 * upper[index] * span_seconds**2,
            )
        elif coordinate in {
            "SKY_EAST_PHASE",
            "SKY_NORTH_PHASE",
            "ECLIPTIC_LONGITUDE_PHASE",
            "ECLIPTIC_LATITUDE_PHASE",
        }:
            if parameter == "RAJ":
                angular = [
                    math.radians(15.0 * (value - adapter.base_values["RAJ"]))
                    for value in (lower[index], upper[index])
                ]
                angular = [
                    value * math.cos(math.radians(adapter.base_values.get("DECJ", 0.0)))
                    for value in angular
                ]
            elif parameter == "ELONG":
                angular = [
                    math.radians(value - adapter.base_values["ELONG"])
                    * math.cos(math.radians(adapter.base_values.get("ELAT", 0.0)))
                    for value in (lower[index], upper[index])
                ]
            else:
                angular = [
                    math.radians(value - adapter.base_values[parameter])
                    for value in (lower[index], upper[index])
                ]
            products = [
                frequency * LIGHT_TRAVEL_TIME_AU_SECONDS * angle
                for frequency in (f0_ref_lower, f0_ref_upper)
                for angle in angular
            ]
            bounds = (min(products), max(products))
        else:
            bounds = (lower[index], upper[index])
        search_lower[index], search_upper[index] = bounds
    if np.any(search_lower >= search_upper):
        raise ValueError("The isolated phase-adapted support has a nonpositive interval.")
    return IsolatedLatticeSupport(
        adapter=adapter,
        physical_center=center.copy(),
        physical_lower=lower.copy(),
        physical_upper=upper.copy(),
        search_center=search_center,
        search_lower=search_lower,
        search_upper=search_upper,
    )


__all__ = [
    "IsolatedCoordinateAdapter",
    "IsolatedLatticeSupport",
    "build_isolated_coordinate_adapter",
    "build_isolated_lattice_support",
    "build_isolated_physical_support",
]
