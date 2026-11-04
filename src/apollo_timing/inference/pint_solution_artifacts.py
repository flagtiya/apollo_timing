from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from pint.models.parameter import AngleParameter, floatParameter
from pint.residuals import Residuals
from scipy.optimize import least_squares, minimize

from apollo_timing.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_model,
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.active_timing_contracts import BranchCertification
from apollo_timing.inference.timing_constraints import SkyPositionConstraint


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


class BoundedPintFitError(RuntimeError):
    """An unfinished fit, with diagnostic values that are not a certified result."""

    def __init__(self, message: str, *, diagnostics: dict[str, object]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def fit_pint_model_with_physical_bounds(
    *,
    model: object,
    toas: object,
    parameter_names: Sequence[str],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    initial_values: Sequence[float] | None = None,
    maximum_evaluations: int = 500,
    sky_constraint: SkyPositionConstraint | None = None,
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
    # A nonzero support center prevents a nearly-null seed from collapsing
    # SciPy's initial trust radius before weak additional parameters can move.
    normalized_initial = np.clip(1.0 + (initial - midpoint) / scale, 0.0, 2.0)
    sky_base = sky_constraint.model_values(model) if sky_constraint else {}
    initial_chi2: float | None = None

    def active_parameter_bounds(physical: np.ndarray) -> tuple[str, ...]:
        tolerance = np.maximum(
            1.0e-7 * np.abs(upper - lower),
            64.0 * np.maximum(np.abs(np.spacing(lower)), np.abs(np.spacing(upper))),
        )
        return tuple(
            name for name, value, low, high, tol in zip(
                names, physical, lower, upper, tolerance, strict=True
            ) if abs(value - low) <= tol or abs(value - high) <= tol
        )

    def sky_slack(values: np.ndarray) -> float:
        physical = midpoint + scale * (np.asarray(values) - 1.0)
        merged = {**sky_base, **dict(zip(names, physical, strict=True))}
        return 1.0 - (sky_constraint.separation_arcmin(merged) / sky_constraint.radius_arcmin) ** 2

    if sky_constraint is not None and sky_slack(normalized_initial) < -1e-10:
        raise ValueError("Bounded PINT initial position is outside the declared sky region.")

    def set_normalized(values: np.ndarray) -> np.ndarray:
        physical = midpoint + scale * (np.asarray(values, dtype=float) - 1.0)
        for name, value in zip(names, physical, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        return physical

    def normalized_residuals(values: np.ndarray) -> np.ndarray:
        nonlocal initial_chi2
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
        if initial_chi2 is None:
            initial_chi2 = float(normalized @ normalized)
        return normalized

    def normalized_jacobian(values: np.ndarray) -> np.ndarray:
        # Coordinates are dimensionless support fractions. A fixed step avoids
        # cancellation when SciPy's relative step vanishes near a null value.
        step = 1.0e-4
        columns = []
        for index in range(len(names)):
            lo, hi = values.copy(), values.copy()
            lo[index] = max(0.0, values[index] - step)
            hi[index] = min(2.0, values[index] + step)
            columns.append((normalized_residuals(hi) - normalized_residuals(lo))
                           / (hi[index] - lo[index]))
        set_normalized(values)
        return np.column_stack(columns)

    fit = least_squares(
        normalized_residuals,
        normalized_initial,
        bounds=(np.zeros(len(names)), 2 * np.ones(len(names))),
        x_scale="jac",
        jac=normalized_jacobian,
        max_nfev=max(int(maximum_evaluations), 1),
        xtol=1.0e-11,
        ftol=1.0e-11,
        gtol=1.0e-11,
    )
    constrained = False
    if sky_constraint is not None and sky_slack(fit.x) < 0:
        # A nonlinear inequality is necessary: independent coordinate boxes
        # admit points outside the spherical cap, including their corners.
        objective_scale = max(float(np.sum(normalized_residuals(normalized_initial) ** 2)), 1.)

        def objective(values):
            residual = normalized_residuals(values)
            return float(residual @ residual) / (2 * objective_scale)

        def gradient(values):
            residual = normalized_residuals(values)
            return normalized_jacobian(values).T @ residual / objective_scale

        fit = minimize(objective, normalized_initial, jac=gradient, method="SLSQP",
                       bounds=[(0., 2.)] * len(names),
                       constraints=[{"type": "ineq", "fun": sky_slack}],
                       options={"maxiter": max(int(maximum_evaluations), 1), "ftol": 1e-13})
        constrained = True
        if not sky_constraint.contains_values(names, set_normalized(fit.x), sky_base):
            raise RuntimeError("Spherical constrained PINT fit left the declared sky region.")
    if not fit.success:
        last_values = np.asarray(getattr(fit, "x", normalized_initial), dtype=float)
        physical = midpoint + scale * (last_values - 1.0)
        last_residual = np.asarray(getattr(fit, "fun", []), dtype=float)
        last_chi2 = (
            float(last_residual @ last_residual)
            if not constrained and last_residual.size else None
        )
        raise BoundedPintFitError(
            "Bounded PINT least-squares did not converge "
            f"(status={fit.status}, nfev={fit.nfev}): {fit.message}",
            diagnostics={
                "converged": False,
                "optimizer": "slsqp" if constrained else "trf_least_squares",
                "termination_status": int(fit.status),
                "termination_message": str(fit.message),
                "function_evaluations": int(fit.nfev),
                "maximum_evaluations": int(maximum_evaluations),
                "optimality": float(fit.optimality) if hasattr(fit, "optimality") else None,
                "initial_chi2": initial_chi2,
                "last_evaluated_chi2": last_chi2,
                "parameter_names": names,
                "initial_values": tuple(float(value) for value in initial),
                "parameter_values": tuple(float(value) for value in physical),
                "lower_bounds": tuple(float(value) for value in lower),
                "upper_bounds": tuple(float(value) for value in upper),
                "active_bounds": active_parameter_bounds(physical),
            },
        )
    physical = set_normalized(np.asarray(fit.x, dtype=float))
    residuals = Residuals(
        toas,
        model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    jacobian = normalized_jacobian(fit.x) if constrained else np.asarray(fit.jac, dtype=float)
    # A pseudoinverse assigns zero variance to unconstrained directions. That
    # must not become spurious evidence for an added physical parameter.
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    norms = np.linalg.norm(jacobian, axis=0)
    correlation_singular = np.linalg.svd(
        jacobian / np.where(norms > 0, norms, 1), compute_uv=False,
    )
    if (len(singular) != len(names) or not np.all(np.isfinite(singular))
            or singular[-1] <= singular[0] * 1.0e-12 or singular[0] <= 0
            or correlation_singular[-1] <= correlation_singular[0] * 1.0e-6):
        raise ValueError("Bounded PINT fit has a rank-deficient parameter response.")
    if callable(getattr(model, "designmatrix", None)):
        design, design_names, _ = model.designmatrix(toas, incfrozen=False, incoffset=False)
        design = np.asarray(design[:, [design_names.index(n) for n in names]], dtype=float)
        uncertainties = np.asarray(model.scaled_toa_uncertainty(toas).to_value(u.s))
        design -= np.average(design, axis=0, weights=1 / uncertainties**2)
        design /= uncertainties[:, None]
        norms = np.linalg.norm(design, axis=0)
        response = np.linalg.svd(design / np.where(norms > 0, norms, 1), compute_uv=False)
        if response[-1] <= response[0] * 1e-6 or response[0] <= 0:
            raise ValueError("Bounded PINT fit has a rank-deficient analytic parameter response.")
    covariance_normalized = (right.T / singular**2) @ right
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
    active_bounds = active_parameter_bounds(physical)
    if sky_constraint is not None and sky_constraint.report(model)["active_bound"]:
        active_bounds = (*active_bounds, "SKY_POSITION")
    return BoundedPintFitResult(
        model=model,
        residuals=residuals,
        covariance=covariance,
        function_evaluations=int(fit.nfev),
        active_bounds=active_bounds,
        termination_status=int(fit.status),
        termination_message=str(fit.message),
        optimality=float(np.linalg.norm(fit.jac, ord=np.inf)) if constrained else float(fit.optimality),
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
    text = serialized_model.as_parfile(format="pint")
    # PINT's default eight sexagesimal decimals can change a high-chi2
    # boundary fit, or move a position just beyond its hard sky cap on reload.
    for name in serialized_model.params:
        parameter = serialized_model[name]
        if isinstance(parameter, AngleParameter) and parameter.quantity is not None:
            value = (parameter.quantity.to_string(unit=parameter.units, sep=":", precision=16)
                     if name in {"RAJ", "DECJ"} else
                     parameter.quantity.to_string(unit=parameter.units, decimal=True, precision=22))
            text = re.sub(rf"^(\s*{re.escape(name)}\s+)\S+", lambda match: match[1] + value,
                          text, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")
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
