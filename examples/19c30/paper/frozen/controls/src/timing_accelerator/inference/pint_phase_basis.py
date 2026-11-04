from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from timing_accelerator.backend.pint_backend import (
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.bayesian_dual_lattice_conditional import (
    PintEquivalentDDPhaseProjector,
    dd_binary_delay_seconds,
)
from timing_accelerator.inference.bayesian_dual_lattice_enumerator import (
    DUAL_LATTICE_FIT_PARAMETERS,
)
from timing_accelerator.inference.bayesian_phase_basis_release import (
    ProjectedPhaseBasis,
    project_raw_phase_basis,
)


F2_BASIS_NAME = "F2"
ASTROMETRIC_BASIS_NAMES = ("RAJ", "DECJ")
DEFAULT_PHASE_BASIS_MODELS = (
    (F2_BASIS_NAME,),
    ASTROMETRIC_BASIS_NAMES,
    (F2_BASIS_NAME, *ASTROMETRIC_BASIS_NAMES),
)
PROHIBITED_AUTOMATIC_PARAMETERS = ("PBDOT", "EPS1DOT", "EPS2DOT")


@dataclass(frozen=True)
class PintPhaseBasisConfig:
    """Physical coordinates allowed to compete with the eight-parameter DD model."""

    candidate_models: tuple[tuple[str, ...], ...] = DEFAULT_PHASE_BASIS_MODELS
    f2_coordinate_scale_hz_s2: float = 1.0e-20
    astrometric_coordinate_scale_arcsec: float = 10.0
    f2_bound_hz_s2: float = 5.0e-19
    astrometric_bound_arcsec: float = 300.0

    def __post_init__(self) -> None:
        available = {F2_BASIS_NAME, *ASTROMETRIC_BASIS_NAMES}
        for model in self.candidate_models:
            names = tuple(str(name).upper() for name in model)
            if not names or len(set(names)) != len(names):
                raise ValueError("Every phase-basis model must be nonempty and unique.")
            if not set(names).issubset(available):
                raise ValueError("Only F2 and the RAJ/DECJ block may be released.")
            has_raj = "RAJ" in names
            has_decj = "DECJ" in names
            if has_raj != has_decj:
                raise ValueError("RAJ and DECJ must be released as one astrometric block.")
        positive = (
            self.f2_coordinate_scale_hz_s2,
            self.astrometric_coordinate_scale_arcsec,
            self.f2_bound_hz_s2,
            self.astrometric_bound_arcsec,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Phase-basis scales and bounds must be positive and finite.")


@dataclass(frozen=True)
class PintPhaseBasisPortfolio:
    """Projected bases and the native PINT coordinate transformation."""

    bases: Mapping[str, ProjectedPhaseBasis]
    candidate_models: tuple[tuple[str, ...], ...]
    reference_values: Mapping[str, float]
    coordinate_steps: Mapping[str, float]
    normalized_bounds: Mapping[str, float]


def pint_absolute_phase_values(model: object, toas: object) -> np.ndarray:
    """Return PINT absolute model phase without losing integer precision early."""

    phase = model.phase(toas, abs_phase=True)
    values = np.asarray(phase.int.value, dtype=np.longdouble)
    values += np.asarray(phase.frac.value, dtype=np.longdouble)
    return values


def set_dd_base_parameters(
    model: object,
    parameter_values: Sequence[float],
    *,
    parameter_names: Sequence[str] = DUAL_LATTICE_FIT_PARAMETERS,
) -> None:
    values = np.asarray(parameter_values, dtype=float)
    names = tuple(str(name).upper() for name in parameter_names)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise ValueError("DD parameter values must be finite and align with their names.")
    for name, value in zip(names, values, strict=True):
        _set_numeric_model_parameter_value(model, name, float(value))


def ensure_f2_parameter(model: object) -> None:
    """Add a zero-valued F2 to a PINT spindown component when it is selected."""

    if "F2" in model.params:
        return
    model.components["Spindown"].add_param(model["F1"].new_param(2), setup=True)
    _set_numeric_model_parameter_value(model, "F2", 0.0)


def apply_phase_basis_coordinates(
    model: object,
    *,
    basis_names: Sequence[str],
    normalized_values: Sequence[float],
    reference_values: Mapping[str, float],
    coordinate_steps: Mapping[str, float],
) -> None:
    names = tuple(str(name).upper() for name in basis_names)
    values = np.asarray(normalized_values, dtype=float)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise ValueError("Phase-basis coordinates must align and be finite.")
    for name, value in zip(names, values, strict=True):
        if name not in reference_values or name not in coordinate_steps:
            raise ValueError(f"Missing coordinate metadata for phase basis {name}.")
        if name == F2_BASIS_NAME:
            ensure_f2_parameter(model)
        _set_numeric_model_parameter_value(
            model,
            name,
            float(reference_values[name]) + float(value) * float(coordinate_steps[name]),
        )


def _astrometric_bases(
    *,
    projector: PintEquivalentDDPhaseProjector,
    parameter_values: np.ndarray,
    coordinate_scale_arcsec: float,
) -> tuple[ProjectedPhaseBasis, ProjectedPhaseBasis, float, float, float, float]:
    model = projector.evaluator.model
    toas = projector.evaluator.toas
    set_dd_base_parameters(model, parameter_values)
    raj_reference_hours = float(model["RAJ"].value)
    decj_reference_degrees = float(model["DECJ"].value)
    raj_step_hours = float(coordinate_scale_arcsec) / 15.0 / 3600.0
    decj_step_degrees = float(coordinate_scale_arcsec) / 3600.0

    _set_numeric_model_parameter_value(model, "RAJ", raj_reference_hours + raj_step_hours)
    raj_plus = pint_absolute_phase_values(model, toas)
    _set_numeric_model_parameter_value(model, "RAJ", raj_reference_hours - raj_step_hours)
    raj_minus = pint_absolute_phase_values(model, toas)
    _set_numeric_model_parameter_value(model, "RAJ", raj_reference_hours)

    _set_numeric_model_parameter_value(
        model,
        "DECJ",
        decj_reference_degrees + decj_step_degrees,
    )
    decj_plus = pint_absolute_phase_values(model, toas)
    _set_numeric_model_parameter_value(
        model,
        "DECJ",
        decj_reference_degrees - decj_step_degrees,
    )
    decj_minus = pint_absolute_phase_values(model, toas)
    _set_numeric_model_parameter_value(model, "DECJ", decj_reference_degrees)

    common = {
        "phase_sigmas": projector.phase_sigmas,
        "sessions": projector.sessions,
        "coordinate_scale": float(coordinate_scale_arcsec),
        "coordinate_unit": "arcsec",
    }
    raj = project_raw_phase_basis(
        np.asarray((raj_plus - raj_minus) / 2.0, dtype=float),
        name="RAJ",
        **common,
    )
    decj = project_raw_phase_basis(
        np.asarray((decj_plus - decj_minus) / 2.0, dtype=float),
        name="DECJ",
        **common,
    )
    return (
        raj,
        decj,
        raj_reference_hours,
        decj_reference_degrees,
        raj_step_hours,
        decj_step_degrees,
    )


def _f2_basis(
    *,
    projector: PintEquivalentDDPhaseProjector,
    parameter_values: np.ndarray,
    coordinate_scale_hz_s2: float,
) -> ProjectedPhaseBasis:
    by_name = dict(
        zip(DUAL_LATTICE_FIT_PARAMETERS, parameter_values, strict=True)
    )
    binary_delay = dd_binary_delay_seconds(
        projector.barycentric_mjd,
        a1_light_seconds=float(by_name["A1"]),
        eccentricity=float(by_name["ECC"]),
        t0_mjd=float(by_name["T0"]),
        orbital_period_days=float(by_name["PB"]),
        longitude_periastron_degrees=float(by_name["OM"]),
        longitude_periastron_rate_degrees_per_year=float(by_name["OMDOT"]),
    )
    elapsed_seconds = (
        (projector.tdb_mjd - projector.spin_epoch_mjd) * np.longdouble(86400.0)
        - projector.fixed_nonbinary_delay_seconds
        - np.asarray(binary_delay, dtype=np.longdouble)
    )
    raw = np.asarray(
        np.longdouble(coordinate_scale_hz_s2)
        * elapsed_seconds**3
        / np.longdouble(6.0),
        dtype=float,
    )
    return project_raw_phase_basis(
        raw,
        projector.phase_sigmas,
        projector.sessions,
        name=F2_BASIS_NAME,
        coordinate_scale=float(coordinate_scale_hz_s2),
        coordinate_unit="Hz/s^2",
    )


def build_pint_phase_basis_portfolio(
    *,
    projector: PintEquivalentDDPhaseProjector,
    parameter_values: Sequence[float],
    config: PintPhaseBasisConfig | None = None,
) -> PintPhaseBasisPortfolio:
    """Build source-agnostic F2 and astrometric alternatives at one mode."""

    active = config or PintPhaseBasisConfig()
    values = np.asarray(parameter_values, dtype=float)
    if values.shape != (len(DUAL_LATTICE_FIT_PARAMETERS),) or not np.all(
        np.isfinite(values)
    ):
        raise ValueError("Phase-basis construction requires eight finite DD values.")
    f2 = _f2_basis(
        projector=projector,
        parameter_values=values,
        coordinate_scale_hz_s2=active.f2_coordinate_scale_hz_s2,
    )
    raj, decj, raj_reference, decj_reference, raj_step, decj_step = (
        _astrometric_bases(
            projector=projector,
            parameter_values=values,
            coordinate_scale_arcsec=active.astrometric_coordinate_scale_arcsec,
        )
    )
    model = projector.evaluator.model
    f2_reference = float(model["F2"].value) if "F2" in model.params else 0.0
    bases = {basis.name: basis for basis in (f2, raj, decj)}
    return PintPhaseBasisPortfolio(
        bases=bases,
        candidate_models=tuple(
            tuple(str(name).upper() for name in model_names)
            for model_names in active.candidate_models
        ),
        reference_values={
            "F2": f2_reference,
            "RAJ": raj_reference,
            "DECJ": decj_reference,
        },
        coordinate_steps={
            "F2": float(active.f2_coordinate_scale_hz_s2),
            "RAJ": raj_step,
            "DECJ": decj_step,
        },
        normalized_bounds={
            "F2": float(active.f2_bound_hz_s2)
            / float(active.f2_coordinate_scale_hz_s2),
            "RAJ": float(active.astrometric_bound_arcsec)
            / float(active.astrometric_coordinate_scale_arcsec),
            "DECJ": float(active.astrometric_bound_arcsec)
            / float(active.astrometric_coordinate_scale_arcsec),
        },
    )


__all__ = [
    "ASTROMETRIC_BASIS_NAMES",
    "DEFAULT_PHASE_BASIS_MODELS",
    "F2_BASIS_NAME",
    "PROHIBITED_AUTOMATIC_PARAMETERS",
    "PintPhaseBasisConfig",
    "PintPhaseBasisPortfolio",
    "apply_phase_basis_coordinates",
    "build_pint_phase_basis_portfolio",
    "ensure_f2_parameter",
    "pint_absolute_phase_values",
    "set_dd_base_parameters",
]
