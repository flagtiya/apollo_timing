"""Physical parameter limits, independent of model family and fit authorization."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle

from apollo_timing.io.par_compatibility import _active_parameter_line


def interval_roundoff(lower, upper):
    """Allow floating-point roundoff in native units, not a unit-sized floor."""
    return 64.0 * np.maximum(np.abs(np.spacing(lower)), np.abs(np.spacing(upper)))


def resolve_parameter_bounds(model, bounds=(), custom_bounds=None, *, optional_parameters=(), require_input_within=True):
    """Merge public absolute intervals and legacy half-widths without losing limits."""
    intervals = {}
    optional = {name.upper() for name in optional_parameters}
    for raw_name, lo, hi in bounds:
        name = raw_name.upper()
        if name in intervals:
            raise ValueError(f"Repeated physical parameter bounds for {name}.")
        intervals[name] = (float(lo), float(hi))
    for raw_name, value in (custom_bounds or {}).items():
        name = raw_name.upper()
        if np.isscalar(value):
            if name not in model.free_params:
                raise ValueError("Custom half-widths may only constrain existing free parameters.")
            width = float(value)
            if not math.isfinite(width) or width <= 0:
                raise ValueError(f"Custom half-width for {name} must be finite and positive.")
            center = float(model[name].value)
            interval = (center - width, center + width)
        else:
            if len(value) != 2:
                raise ValueError(f"Custom physical bounds for {name} need [lower, upper].")
            interval = tuple(float(v) for v in value)
        if name in intervals and intervals[name] != interval:
            raise ValueError(f"Conflicting physical bounds for {name}.")
        intervals[name] = interval
    positive = {"F0", "PB", "FB0", "A1", "M2", "MTOT", "H3", "STIGMA", "SHAPMAX"}
    domains = {"ECC": (0., 1. - 1e-12), "SINI": (0., 1.),
               "DECJ": (-90., 90.), "ELAT": (-90., 90.), "KIN": (0., 180.)}
    resolved = []
    for name, (lo, hi) in intervals.items():
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"Invalid finite physical support for {name}.")
        # User intervals may replace automatic widths, never basic physical domains.
        domain = (np.finfo(float).tiny, math.inf) if name in positive else domains.get(name)
        if domain is not None:
            lo, hi = max(lo, domain[0]), min(hi, domain[1])
            if not lo < hi:
                raise ValueError(f"Bounds for {name} have no overlap with its physical domain.")
        value = model[name].value if name in model.params else None
        if value is None:
            if name not in optional:
                raise ValueError(f"Bounded {name} is absent; explicitly authorize its exploration or correct the name.")
        else:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Bounded {name} must be a numeric timing parameter.") from error
            if not math.isfinite(number) or (require_input_within and not lo <= number <= hi):
                raise ValueError(f"Input {name} conflicts with its declared physical bounds.")
        resolved.append((name, lo, hi))
    return tuple(resolved)


def bind_parameter_bounds(config, model):
    """Resolve half-widths once against the input, before fitting or conversion."""
    if not config.custom_bounds and not config.parameter_exploration.bounds:
        return config
    bounds = resolve_parameter_bounds(
        model, config.parameter_exploration.bounds, config.custom_bounds,
        optional_parameters=config.parameter_exploration.parameters,
    )
    return replace(config, custom_bounds=None,
                   parameter_exploration=replace(config.parameter_exploration, bounds=bounds))


def parameter_bounds_report(model, bounds, *, optional_absent=()):
    """Check live or independently reloaded output, including fixed parameters."""
    records = {}
    for raw_name, lo, hi in bounds:
        name = raw_name.upper()
        value = model[name].value if name in model.params else None
        if value is None:
            records[name] = dict(lower=lo, upper=hi, value=None, active_bound=False,
                                 within_bounds=name in optional_absent, status="absent")
            continue
        value = float(value)
        tolerance = float(interval_roundoff(lo, hi))
        active = min(abs(value - lo), abs(value - hi)) <= max(tolerance, (hi - lo) * 1e-7)
        records[name] = dict(lower=float(lo), upper=float(hi), value=value, active_bound=bool(active),
                             within_bounds=bool(math.isfinite(value) and lo - tolerance <= value <= hi + tolerance),
                             status="checked")
    return dict(within_bounds=all(r["within_bounds"] for r in records.values()), parameters=records)


@dataclass(frozen=True)
class SkyPositionConstraint:
    """A spherical cap about the original par position, at its position epoch.

    RAJ is in PINT hours; other position coordinates are in degrees. This is
    a hard search boundary, not an uncertainty ellipse or a proper-motion cap.
    """

    longitude_name: str
    latitude_name: str
    longitude_deg: float
    latitude_deg: float
    radius_arcmin: float
    epoch_mjd: float | None = None
    ecliptic_obliquity: str = "IERS2010"

    def __post_init__(self) -> None:
        if (self.longitude_name, self.latitude_name) not in {
            ("RAJ", "DECJ"), ("ELONG", "ELAT")
        }:
            raise ValueError("Sky support requires one equatorial or ecliptic coordinate pair.")
        if not all(math.isfinite(v) for v in (
            self.longitude_deg, self.latitude_deg, self.radius_arcmin
        )) or not -90 <= self.latitude_deg <= 90:
            raise ValueError("Sky support needs finite coordinates and latitude in [-90, 90].")
        if not 0 < self.radius_arcmin < 10800:
            raise ValueError("Sky radius must be finite and between 0 and 10800 arcmin (exclusive).")
        if self.epoch_mjd is not None and not math.isfinite(self.epoch_mjd):
            raise ValueError("Sky reference epoch must be finite.")

    @classmethod
    def from_par(cls, path: Path, radius_arcmin: float) -> "SkyPositionConstraint":
        # Inspect text so declaration-only binary inputs need not be PINT-loadable.
        entries = {}
        aliases = {"RA": "RAJ", "DEC": "DECJ", "LAMBDA": "ELONG", "BETA": "ELAT"}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            item = _active_parameter_line(line)
            if item is None:
                continue
            key, parts = item
            key = aliases.get(key, key)
            if key in {"RAJ", "DECJ", "ELONG", "ELAT", "POSEPOCH", "PEPOCH", "ECL"}:
                if key in entries or len(parts) < 2:
                    raise ValueError(f"Ambiguous or incomplete sky reference parameter: {key}.")
                entries[key] = parts[1]
        pairs = [p for p in (("RAJ", "DECJ"), ("ELONG", "ELAT")) if all(n in entries for n in p)]
        supplied = set(entries) & {"RAJ", "DECJ", "ELONG", "ELAT"}
        if len(pairs) != 1 or supplied != set(pairs[0]):
            raise ValueError("A sky radius requires exactly one complete input position pair.")
        lon, lat = pairs[0]
        epoch = entries.get("POSEPOCH", entries.get("PEPOCH"))
        return cls(lon, lat, float(Angle(entries[lon], unit=u.hourangle if lon == "RAJ" else u.deg).degree),
                   float(Angle(entries[lat], unit=u.deg).degree), float(radius_arcmin),
                   None if epoch is None else float(epoch.replace("D", "E")), entries.get("ECL", "IERS2010"))

    @property
    def radius_radians(self) -> float:
        return math.radians(self.radius_arcmin / 60)

    @classmethod
    def from_report(cls, report: Mapping) -> "SkyPositionConstraint":
        lon, lat = report["coordinate_names"]
        return cls(lon, lat, float(report["center_longitude_deg"]), float(report["center_latitude_deg"]),
                   float(report["radius_arcmin"]), report.get("reference_epoch_mjd"),
                   report.get("ecliptic_obliquity") or "IERS2010")

    def separation_arcmin(self, values: Mapping[str, float]) -> float:
        lon = float(values[self.longitude_name]) * (15 if self.longitude_name == "RAJ" else 1)
        lat = float(values[self.latitude_name])
        if not (math.isfinite(lon) and math.isfinite(lat) and -90 <= lat <= 90):
            return math.inf
        # atan2 is stable both for tiny separations and near the antipode.
        a, b = math.radians(self.latitude_deg), math.radians(lat)
        d = math.radians((lon - self.longitude_deg + 180) % 360 - 180)
        y = math.hypot(math.cos(b) * math.sin(d),
                       math.cos(a) * math.sin(b) - math.sin(a) * math.cos(b) * math.cos(d))
        x = math.sin(a) * math.sin(b) + math.cos(a) * math.cos(b) * math.cos(d)
        return math.degrees(math.atan2(y, x)) * 60

    def contains_values(self, names: Sequence[str], values: Sequence[float],
                        base_values: Mapping[str, float]) -> bool:
        merged = dict(base_values)
        merged.update(zip(names, values, strict=True))
        return self.separation_arcmin(merged) <= self.radius_arcmin + 1e-8 / 60

    def model_values(self, model: object) -> dict[str, float]:
        names = (self.longitude_name, self.latitude_name)
        if any(n not in model.params or model[n].value is None for n in names):
            raise ValueError("Model changed the sky coordinate frame of the declared constraint.")
        epoch = model["POSEPOCH"].value if "POSEPOCH" in model.params else None
        if epoch is None and "PEPOCH" in model.params:
            epoch = model["PEPOCH"].value
        if self.epoch_mjd is not None and (epoch is None or abs(float(epoch) - self.epoch_mjd) > 1e-9):
            raise ValueError("Model changed the reference epoch of the sky constraint.")
        if self.longitude_name == "ELONG" and "ECL" in model.params and model["ECL"].value != self.ecliptic_obliquity:
            raise ValueError("Model changed the ecliptic reference frame of the sky constraint.")
        return {n: float(model[n].value) for n in names}

    def report(self, model: object) -> dict:
        distance = self.separation_arcmin(self.model_values(model))
        return dict(kind="spherical_cap", coordinate_names=(self.longitude_name, self.latitude_name),
                    center_longitude_deg=self.longitude_deg, center_latitude_deg=self.latitude_deg,
                    reference_epoch_mjd=self.epoch_mjd, radius_arcmin=self.radius_arcmin,
                    ecliptic_obliquity=self.ecliptic_obliquity if self.longitude_name == "ELONG" else None,
                    separation_arcmin=distance, within_bounds=distance <= self.radius_arcmin + 1e-8 / 60,
                    active_bound=abs(distance - self.radius_arcmin) <= max(1e-6 * self.radius_arcmin, 1e-8 / 60),
                    interpretation="hard_position_limit_at_reference_epoch_not_confidence_region")

    def bounding_intervals(self, values: Mapping[str, float]) -> dict[str, tuple[float, float]]:
        """Enclose the whole cap, unwrapping longitude about the live seed."""
        radius = self.radius_arcmin / 60
        lat_lo, lat_hi = max(-90., self.latitude_deg - radius), min(90., self.latitude_deg + radius)
        width = 180. if abs(self.latitude_deg) + radius >= 90 else math.degrees(
            math.asin(min(1., math.sin(self.radius_radians) / math.cos(math.radians(self.latitude_deg)))))
        factor = 15 if self.longitude_name == "RAJ" else 1
        seed = values[self.longitude_name] * factor
        center = seed + (self.longitude_deg - seed + 180) % 360 - 180
        return {self.longitude_name: ((center - width) / factor, (center + width) / factor),
                self.latitude_name: (lat_lo, lat_hi)}


def bind_sky_constraint(config, par_path: Path):
    """Resolve once before discovery/preparation; dataclass replacement retains it."""
    from dataclasses import replace

    radius = config.sky_radius_arcmin
    constraint = config.sky_position_constraint
    if radius is not None:
        if constraint is None:
            constraint = SkyPositionConstraint.from_par(par_path, radius)
        elif radius != constraint.radius_arcmin:
            raise ValueError("Sky radius conflicts with the bound reference region.")
    return replace(config, sky_position_constraint=constraint) if constraint is not None else config
