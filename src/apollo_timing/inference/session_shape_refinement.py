"""Shared fixed-local-integer shape refinement; callers own physical support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from apollo_timing.backend.pint_backend import _set_numeric_model_parameter_value
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    SourceSession,
    profile_fixed_local_pulse_component,
    profile_two_level_pulse_lattice,
)


class LocalIntegerComponentChangedError(RuntimeError):
    """A continuous refinement leaves the declared within-session component."""


@dataclass(frozen=True)
class SessionShapeRefinement:
    chi2: float
    values: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]
    converged: bool
    evaluations: int


def relative_session_phases(cycles: np.ndarray, sessions: Sequence[SourceSession]) -> np.ndarray:
    """Subtract session anchors before float conversion to retain phase precision."""
    covered = tuple(index for session in sessions for index in session.toa_indices)
    if covered != tuple(range(len(cycles))) or any(not session.toa_indices for session in sessions):
        raise ValueError("Source sessions must cover every phase exactly once in order.")
    relative = np.empty(len(cycles), dtype=float)
    for session in sessions:
        indices = np.asarray(session.toa_indices, dtype=int)
        relative[indices] = cycles[indices] - cycles[indices[0]]
    return relative


def refine_fixed_session_shape(
    *,
    model: object,
    phase_values: Callable[[], np.ndarray],
    phase_sigmas: np.ndarray,
    sessions: Sequence[SourceSession],
    parameter_names: Sequence[str],
    scales: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> SessionShapeRefinement:
    names = tuple(parameter_names)
    base = np.asarray([float(model[name].value) for name in names])
    scales = np.asarray(scales, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if not names or any(value.shape != base.shape for value in (scales, lower, upper)):
        raise ValueError("Refinement names, scales and bounds must be aligned vectors.")
    if (
        not np.all(np.isfinite(base))
        or not np.all(np.isfinite(scales) & (scales > 0))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(lower > 0)
        or np.any(upper < 0)
    ):
        raise ValueError("Refinement scales and bounds must be finite and contain the seed.")
    sigmas = np.asarray(phase_sigmas, dtype=float)
    initial = profile_two_level_pulse_lattice(
        phase_values(),
        sigmas,
        sessions,
        integer_radius=0,
    )
    component = initial.local_integer_offsets

    def residuals(offsets: np.ndarray) -> np.ndarray:
        physical = base + np.asarray(offsets, dtype=float) * scales
        for name, value in zip(names, physical, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        profile = profile_fixed_local_pulse_component(
            phase_values(),
            sigmas,
            sessions,
            component,
        )
        return profile.normalized_local_residuals

    fit = least_squares(
        residuals,
        np.zeros(len(names)),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=1.0e-3,
        max_nfev=3_000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    final = residuals(fit.x)
    decoded = profile_two_level_pulse_lattice(
        phase_values(),
        sigmas,
        sessions,
        integer_radius=0,
    )
    if decoded.local_integer_offsets != component:
        raise LocalIntegerComponentChangedError(
            "All-session ELL1 shape refinement changed its local integer component."
        )
    return SessionShapeRefinement(
        chi2=float(final @ final),
        values=base + fit.x * scales,
        local_integer_offsets=component,
        converged=bool(fit.success),
        evaluations=int(fit.nfev),
    )
