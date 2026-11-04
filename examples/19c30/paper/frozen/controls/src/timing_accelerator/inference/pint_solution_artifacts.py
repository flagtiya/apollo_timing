from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from pint.models.parameter import floatParameter
from pint.residuals import Residuals
from scipy.optimize import least_squares

from timing_accelerator.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_model,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.active_timing_contracts import BranchCertification


@dataclass(frozen=True)
class BoundedPintFitResult:
    model: object
    residuals: object
    covariance: np.ndarray
    function_evaluations: int
    active_bounds: tuple[str, ...]
    termination_status: int
    termination_message: str
    optimality: float


def fit_pint_model_with_physical_bounds(
    *,
    model: object,
    toas: object,
    parameter_names: Sequence[str],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    initial_values: Sequence[float] | None = None,
    maximum_evaluations: int = 500,
) -> BoundedPintFitResult:
    """Polish one fixed pulse path without leaving its Bayesian support."""

    names = tuple(str(name).upper() for name in parameter_names)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if not names or lower.shape != (len(names),) or upper.shape != lower.shape:
        raise ValueError("Bounded PINT fit parameters and supports must align.")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Bounded PINT supports must be finite.")
    if np.any(lower >= upper):
        raise ValueError("Every bounded PINT support must have positive width.")
    if initial_values is None:
        initial = np.asarray([float(model[name].value) for name in names], dtype=float)
    else:
        initial = np.asarray(initial_values, dtype=float)
    if initial.shape != lower.shape or not np.all(np.isfinite(initial)):
        raise ValueError("Bounded PINT initial values must be finite and aligned.")

    midpoint = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    normalized_initial = np.clip((initial - midpoint) / scale, -1.0, 1.0)

    def set_normalized(values: np.ndarray) -> np.ndarray:
        physical = midpoint + scale * np.asarray(values, dtype=float)
        for name, value in zip(names, physical, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        return physical

    def normalized_residuals(values: np.ndarray) -> np.ndarray:
        set_normalized(values)
        residuals = Residuals(
            toas,
            model,
            subtract_mean=True,
            use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        try:
            uncertainties = model.scaled_toa_uncertainty(toas)
        except (AttributeError, TypeError, ValueError):
            uncertainties = toas.get_errors()
        normalized = np.asarray(
            (residuals.time_resids / uncertainties).decompose().value,
            dtype=float,
        )
        if not np.all(np.isfinite(normalized)):
            raise ValueError("PINT produced non-finite bounded-fit residuals.")
        return normalized

    fit = least_squares(
        normalized_residuals,
        normalized_initial,
        bounds=(-np.ones(len(names)), np.ones(len(names))),
        x_scale="jac",
        diff_step=1.0e-5,
        max_nfev=max(int(maximum_evaluations), 1),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )
    if not fit.success:
        raise RuntimeError(
            "Bounded PINT least-squares did not converge "
            f"(status={fit.status}, nfev={fit.nfev}): {fit.message}"
        )
    physical = set_normalized(np.asarray(fit.x, dtype=float))
    residuals = Residuals(
        toas,
        model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    jacobian = np.asarray(fit.jac, dtype=float)
    covariance_normalized = np.linalg.pinv(jacobian.T @ jacobian)
    covariance = scale[:, None] * covariance_normalized * scale[None, :]
    degrees_of_freedom = max(len(toas.table) - len(names), 1)
    covariance *= max(float(residuals.chi2) / degrees_of_freedom, 1.0)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Bounded PINT fit produced an invalid covariance matrix.")

    uncertainty_updates = {}
    for index, name in enumerate(names):
        diagonal = float(covariance[index, index])
        units = getattr(model[name], "units", None)
        if diagonal > 0.0 and np.isfinite(diagonal) and units is not None:
            uncertainty_updates[name] = math.sqrt(diagonal) * units
    if uncertainty_updates:
        model.set_param_uncertainties(uncertainty_updates)
    width_tolerance = 1.0e-7 * np.abs(upper - lower)
    floating_tolerance = 64.0 * np.maximum(
        np.abs(np.spacing(lower)), np.abs(np.spacing(upper))
    )
    tolerance = np.maximum(width_tolerance, floating_tolerance)
    active_bounds = tuple(
        name
        for name, value, low, high, tol in zip(
            names, physical, lower, upper, tolerance, strict=True
        )
        if abs(value - low) <= tol or abs(value - high) <= tol
    )
    return BoundedPintFitResult(
        model=model,
        residuals=residuals,
        covariance=covariance,
        function_evaluations=int(fit.nfev),
        active_bounds=active_bounds,
        termination_status=int(fit.status),
        termination_message=str(fit.message),
        optimality=float(fit.optimality),
    )


def extract_fitter_parameter_covariance(
    fitter: object,
    parameter_names: Sequence[str],
) -> np.ndarray:
    """Extract a finite PINT covariance in native par-file parameter units."""

    names = tuple(str(name).upper() for name in parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Covariance parameter names must be unique and nonempty.")
    covariance = fitter.parameter_covariance_matrix.get_label_matrix(list(names))
    matrix = np.asarray(covariance.matrix, dtype=float)
    if matrix.shape != (len(names), len(names)) or not np.all(np.isfinite(matrix)):
        raise ValueError("PINT returned an invalid fitted-parameter covariance matrix.")
    return 0.5 * (matrix + matrix.T)


def write_pint_model_parfile_roundtrip_safe(*, model, path: Path) -> Path:
    serialized_model = deepcopy(model)
    for name in serialized_model.params:
        parameter = serialized_model[name]
        if type(parameter) is not floatParameter or not parameter.unit_scale:
            continue
        factor = float(parameter.scale_factor)
        threshold = abs(float(parameter.scale_threshold))
        value = parameter.value
        if value is not None and np.isfinite(value) and abs(value) > threshold:
            parameter.quantity = (value / factor) * parameter.units
        uncertainty = parameter.uncertainty_value
        if (
            uncertainty is not None
            and np.isfinite(uncertainty)
            and abs(uncertainty) > threshold
        ):
            parameter.uncertainty = (uncertainty / factor) * parameter.units

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_model.write_parfile(path, format="pint")
    return path


def _set_model_uncertainties_from_covariance(
    model,
    certification: BranchCertification,
    *,
    parameter_names: Sequence[str] | None = None,
) -> None:
    covariance = np.asarray(certification.covariance, dtype=float)
    if covariance.ndim != 2:
        return
    covariance_parameter_names = list(parameter_names or certification.fitted_theta)
    uncertainty_updates: dict[str, u.Quantity] = {}
    for index, name in enumerate(covariance_parameter_names):
        if name not in model.params:
            continue
        if index >= covariance.shape[0] or index >= covariance.shape[1]:
            continue
        diagonal_entry = float(covariance[index, index])
        if not np.isfinite(diagonal_entry) or diagonal_entry <= 0.0:
            continue
        uncertainty_updates[name] = np.sqrt(diagonal_entry) * model[name].units
    if uncertainty_updates:
        model.set_param_uncertainties(uncertainty_updates)


def write_certified_solution_parfile(
    *,
    path: Path,
    seed_par_path: Path,
    certification: BranchCertification,
    fit_parameter_names: Sequence[str] | None = None,
) -> Path:
    model = _load_pint_model(seed_par_path)
    if fit_parameter_names is not None:
        _freeze_all_parameters(model)
        for name in fit_parameter_names:
            if name in model.params:
                model[name].frozen = False
    for name, value in certification.fitted_theta.items():
        if name not in model.params:
            continue
        _set_numeric_model_parameter_value(model, name, float(value))
    _set_model_uncertainties_from_covariance(
        model,
        certification,
        parameter_names=fit_parameter_names,
    )
    return write_pint_model_parfile_roundtrip_safe(model=model, path=path)


__all__ = [
    "BoundedPintFitResult",
    "extract_fitter_parameter_covariance",
    "fit_pint_model_with_physical_bounds",
    "write_certified_solution_parfile",
    "write_pint_model_parfile_roundtrip_safe",
]
