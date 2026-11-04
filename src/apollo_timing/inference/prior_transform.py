from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from astropy.coordinates import Angle
from astropy import units as u


@dataclass
class ProductSpaceParameterPrior:
    name: str
    kind: str
    center: float
    lower: float
    upper: float


_FAST_PARAMETER_WIDTHS: dict[str, float] = {
    "RAJ": 4.0 / 3600.0,
    "DECJ": 1.0 / 60.0,
    "F0": 1.0e-5,
    "F1": 1.0e-13,
    "F2": 1.0e-22,
    "F3": 1.0e-30,
    "PB": 5.0e-3,
    "PBDOT": 1.0e-8,
    "A1": 5.0e-3,
    "A1DOT": 1.0e-11,
    "T0": 5.0e-2,
    "TASC": 5.0e-2,
    "EPS1": 5.0e-3,
    "EPS2": 5.0e-3,
    "EPS1DOT": 1.0,
    "EPS2DOT": 1.0,
    "DM": 2.0,
}


def _omdot_prior_width(values: dict[str, float]) -> float:
    pb_days = abs(float(values.get("PB", float("nan"))))
    eccentricity = abs(float(values.get("ECC", float("nan"))))
    if not math.isfinite(pb_days) or pb_days <= 0.0:
        return 10.0
    if not math.isfinite(eccentricity) or eccentricity >= 1.0:
        eccentricity = 0.0

    # Use a broad 10-solar-mass total-mass envelope and a
    # three-sigma allowance for the GR periastron-advance scale.
    tsun_seconds = 4.925490947e-6
    orbital_seconds = pb_days * 86400.0
    gr_rate_rad_per_second = (
        3.0
        * (tsun_seconds * 10.0) ** (2.0 / 3.0)
        * (2.0 * math.pi / orbital_seconds) ** (5.0 / 3.0)
        / max(1.0 - eccentricity * eccentricity, 1.0e-12)
    )
    gr_rate_deg_per_year = (
        gr_rate_rad_per_second
        * 180.0
        / math.pi
        * 365.25
        * 86400.0
    )
    return max(0.5, 3.0 * gr_rate_deg_per_year)


def extract_fit_flagged_parameters(par_path: Path) -> list[str]:
    fit_params: list[str] = []
    for raw_line in par_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[2] != "1":
            continue
        fit_params.append(parts[0].upper())
    return fit_params


def _parse_parameter_center(name: str, token: str) -> tuple[str, float]:
    if name == "RAJ":
        return "raj_hourangle", float(Angle(token, unit=u.hourangle).hour)
    if name == "DECJ":
        return "dec_degree", float(Angle(token, unit=u.deg).degree)
    return "numeric", float(token)


def build_parameter_prior_specs(
    par_path: Path,
    allowed_params: set[str] | None = None,
) -> list[ProductSpaceParameterPrior]:
    specs: list[ProductSpaceParameterPrior] = []
    fit_params = set(extract_fit_flagged_parameters(par_path))
    if allowed_params is not None:
        fit_params &= {name.upper() for name in allowed_params}
    numeric_values: dict[str, float] = {}
    for raw_line in par_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 2:
            continue
        try:
            numeric_values[parts[0].upper()] = float(parts[1])
        except ValueError:
            continue
    for raw_line in par_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0].upper()
        if name not in fit_params:
            continue
        kind, center = _parse_parameter_center(name, parts[1])
        width = (
            _omdot_prior_width(numeric_values)
            if name == "OMDOT"
            else _FAST_PARAMETER_WIDTHS.get(name, max(abs(center) * 0.1, 0.1))
        )
        specs.append(
            ProductSpaceParameterPrior(
                name=name,
                kind=kind,
                center=center,
                lower=center - width,
                upper=center + width,
            )
        )
    return specs
