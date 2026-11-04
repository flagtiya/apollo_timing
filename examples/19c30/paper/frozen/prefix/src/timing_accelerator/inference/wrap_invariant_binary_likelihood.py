from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy import units as u
from pint.residuals import Residuals

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.cluster_manifest import (
    load_cluster_manifest,
    resolve_source_session_manifest,
)
from timing_accelerator.inference.pulse_number_reference import (
    ensure_local_island_pulse_numbers,
)
from timing_accelerator.inference.integer_least_squares import (
    build_marginalized_integer_covariance,
)


def _parameter_domain(name: str) -> tuple[float | None, float | None]:
    """Return the model-domain bounds needed before evaluating a DD model."""
    normalized = str(name).upper()
    if normalized in {"F0", "A1", "PB"}:
        return (0.0, None)
    if normalized == "ECC":
        return (0.0, 1.0)
    return (None, None)


def _values_in_model_domain(
    parameter_names: Sequence[str],
    values: Sequence[float],
) -> bool:
    candidate = np.asarray(values, dtype=float)
    if candidate.ndim != 1 or candidate.size != len(parameter_names):
        return False
    if not np.all(np.isfinite(candidate)):
        return False
    for name, value in zip(parameter_names, candidate, strict=True):
        lower, upper = _parameter_domain(name)
        if lower is not None and value < lower:
            return False
        if upper is not None and value >= upper:
            return False
    return True


@dataclass(frozen=True)
class SessionShapeProfile:
    projected_phase_residuals: np.ndarray
    normalized_residuals: np.ndarray
    session_offsets: np.ndarray
    session_offset_variances: np.ndarray
    chi2: float
    dof: int
    session_count: int
    slope_chi2: float = 0.0
    curvature_chi2: float = 0.0
    session_epochs: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    session_slopes: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    session_slope_variances: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    session_intercept_slope_covariances: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    session_time_leverages: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    session_slope_observable: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    orthogonal_phase_residuals: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )
    normalized_orthogonal_residuals: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=float)
    )

    @property
    def reduced_chi2(self) -> float:
        return float(self.chi2 / self.dof) if self.dof > 0 else float("nan")

    @property
    def relative_log_likelihood(self) -> float:
        return float(-0.5 * self.chi2)

    @property
    def standardized_session_slopes(self) -> np.ndarray:
        slopes = np.asarray(self.session_slopes, dtype=float)
        variances = np.asarray(self.session_slope_variances, dtype=float)
        observable = np.asarray(self.session_slope_observable, dtype=bool)
        if slopes.shape != variances.shape or slopes.shape != observable.shape:
            raise ValueError("Session slope arrays must have matching shapes.")
        if not np.any(observable):
            return np.zeros(0, dtype=float)
        return slopes[observable] / np.sqrt(variances[observable])


def session_shape_profile_from_two_level(
    profile: Any,
) -> SessionShapeProfile:
    """Convert an APOLLO two-level pulse profile to the common shape API.

    The conversion intentionally depends on the profile contract rather than
    on a concrete binary model. It is shared by the binary phase-lattice evaluators.
    """

    normalized = np.asarray(profile.normalized_local_residuals, dtype=float)
    offsets = np.asarray(profile.session_intercepts, dtype=float)
    variances = np.asarray(profile.session_intercept_variances, dtype=float)
    if normalized.ndim != 1:
        raise ValueError("Two-level normalized residuals must be one-dimensional.")
    if offsets.ndim != 1 or variances.shape != offsets.shape:
        raise ValueError("Two-level session intercepts and variances must align.")
    if offsets.size == 0:
        raise ValueError("A two-level profile must contain at least one session.")
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Two-level normalized residuals must be finite.")
    if not np.all(np.isfinite(offsets)) or not np.all(
        np.isfinite(variances) & (variances > 0.0)
    ):
        raise ValueError("Two-level session statistics must be finite and positive.")
    return SessionShapeProfile(
        projected_phase_residuals=normalized.copy(),
        normalized_residuals=normalized,
        session_offsets=offsets,
        session_offset_variances=variances,
        chi2=float(profile.local_chi2),
        dof=max(int(normalized.size - offsets.size), 0),
        session_count=int(offsets.size),
    )


def profile_session_phase_offsets(
    phase_residuals: Sequence[float],
    phase_sigmas: Sequence[float],
    session_toa_indices: Sequence[Sequence[int]],
    session_time_coordinates: Sequence[float] | None = None,
) -> SessionShapeProfile:
    residuals = np.asarray(phase_residuals, dtype=float)
    sigmas = np.asarray(phase_sigmas, dtype=float)
    if residuals.ndim != 1 or sigmas.shape != residuals.shape:
        raise ValueError("Phase residuals and sigmas must be matching one-dimensional arrays.")
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Phase residuals must be finite.")
    if not np.all(np.isfinite(sigmas) & (sigmas > 0.0)):
        raise ValueError("Phase sigmas must be finite and positive.")
    if session_time_coordinates is None:
        time_coordinates = np.arange(residuals.size, dtype=float)
    else:
        time_coordinates = np.asarray(session_time_coordinates, dtype=float)
        if time_coordinates.shape != residuals.shape or not np.all(
            np.isfinite(time_coordinates)
        ):
            raise ValueError(
                "Session time coordinates must match the residual shape and be finite."
            )

    sessions = [np.asarray(indices, dtype=int) for indices in session_toa_indices]
    if not sessions or any(indices.ndim != 1 or indices.size == 0 for indices in sessions):
        raise ValueError("At least one non-empty source session is required.")
    covered = np.concatenate(sessions)
    if np.any((covered < 0) | (covered >= residuals.size)):
        raise ValueError("Source-session TOA indices are out of bounds.")
    if covered.size != residuals.size or np.unique(covered).size != residuals.size:
        raise ValueError("Source sessions must cover every TOA exactly once.")

    projected = np.empty_like(residuals)
    normalized = np.empty_like(residuals)
    offsets: list[float] = []
    offset_variances: list[float] = []
    session_epochs: list[float] = []
    session_slopes: list[float] = []
    session_slope_variances: list[float] = []
    intercept_slope_covariances: list[float] = []
    time_leverages: list[float] = []
    slope_observable: list[bool] = []
    orthogonal = np.empty_like(residuals)
    normalized_orthogonal = np.empty_like(residuals)
    dof = 0
    slope_chi2 = 0.0
    for indices in sessions:
        weights = 1.0 / np.square(sigmas[indices])
        offset = float(np.sum(weights * residuals[indices]) / np.sum(weights))
        local_projected = residuals[indices] - offset
        projected[indices] = local_projected
        normalized[indices] = local_projected / sigmas[indices]
        offsets.append(offset)
        offset_variances.append(float(1.0 / np.sum(weights)))
        session_epoch = float(
            np.sum(weights * time_coordinates[indices]) / np.sum(weights)
        )
        session_epochs.append(session_epoch)
        centered_time = time_coordinates[indices] - session_epoch
        slope_information = float(np.sum(weights * centered_time * centered_time))
        time_scale = max(float(np.max(np.abs(time_coordinates[indices]))), 1.0)
        minimum_time_span = 64.0 * np.finfo(float).eps * time_scale
        observable = bool(
            np.isfinite(slope_information)
            and slope_information > 0.0
            and float(np.ptp(time_coordinates[indices])) > minimum_time_span
        )
        if observable:
            slope_score = float(
                np.sum(weights * centered_time * local_projected)
            )
            slope = slope_score / slope_information
            slope_variance = 1.0 / slope_information
            local_orthogonal = local_projected - slope * centered_time
            local_slope_chi2 = slope * slope / slope_variance
            slope_chi2 += local_slope_chi2
        else:
            slope = float("nan")
            slope_variance = float("inf")
            local_orthogonal = local_projected
        orthogonal[indices] = local_orthogonal
        normalized_orthogonal[indices] = local_orthogonal / sigmas[indices]
        session_slopes.append(float(slope))
        session_slope_variances.append(float(slope_variance))
        intercept_slope_covariances.append(0.0)
        time_leverages.append(float(slope_information))
        slope_observable.append(observable)
        dof += max(int(indices.size) - 1, 0)

    chi2 = float(np.dot(normalized, normalized))
    curvature_chi2 = float(np.dot(normalized_orthogonal, normalized_orthogonal))

    return SessionShapeProfile(
        projected_phase_residuals=projected,
        normalized_residuals=normalized,
        session_offsets=np.asarray(offsets, dtype=float),
        session_offset_variances=np.asarray(offset_variances, dtype=float),
        chi2=chi2,
        dof=int(dof),
        session_count=len(sessions),
        slope_chi2=float(max(slope_chi2, 0.0)),
        curvature_chi2=float(max(curvature_chi2, 0.0)),
        session_epochs=np.asarray(session_epochs, dtype=float),
        session_slopes=np.asarray(session_slopes, dtype=float),
        session_slope_variances=np.asarray(session_slope_variances, dtype=float),
        session_intercept_slope_covariances=np.asarray(
            intercept_slope_covariances, dtype=float
        ),
        session_time_leverages=np.asarray(time_leverages, dtype=float),
        session_slope_observable=np.asarray(slope_observable, dtype=bool),
        orthogonal_phase_residuals=orthogonal,
        normalized_orthogonal_residuals=normalized_orthogonal,
    )


@dataclass(frozen=True)
class ConditionalSessionMode:
    integer_offsets: tuple[int, ...]
    scaled_parameter_offsets: np.ndarray
    parameter_values: np.ndarray
    linearized_chi2: float


@dataclass(frozen=True)
class SessionShapeLinearization:
    parameter_values: np.ndarray
    parameter_steps: np.ndarray
    profile: SessionShapeProfile
    shape_jacobian: np.ndarray
    session_offset_jacobian: np.ndarray
    relative_session_offsets: np.ndarray
    relative_session_jacobian: np.ndarray
    relative_offset_covariance: np.ndarray
    marginalized_integer_covariance: tuple[tuple[object, ...], ...]
    standardized_slope_jacobian: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=float)
    )

    @property
    def marginalized_integer_center(self) -> np.ndarray:
        shape_precision = self.shape_jacobian.T @ self.shape_jacobian
        shape_gradient = self.shape_jacobian.T @ self.profile.normalized_residuals
        shape_mode = np.linalg.solve(shape_precision, -shape_gradient)
        return (
            self.relative_session_offsets
            + self.relative_session_jacobian @ shape_mode
        )

    def condition_on_integer_offsets(
        self,
        integer_offsets: Sequence[int],
    ) -> ConditionalSessionMode:
        integers = np.asarray(integer_offsets, dtype=float)
        if integers.shape != self.relative_session_offsets.shape:
            raise ValueError(
                "Integer offsets must match the relative source-session dimension."
            )
        noise_precision = np.linalg.inv(self.relative_offset_covariance)
        shape_precision = self.shape_jacobian.T @ self.shape_jacobian
        shape_gradient = self.shape_jacobian.T @ self.profile.normalized_residuals
        conditional_precision = (
            shape_precision
            + self.relative_session_jacobian.T
            @ noise_precision
            @ self.relative_session_jacobian
        )
        conditional_gradient = (
            shape_gradient
            + self.relative_session_jacobian.T
            @ noise_precision
            @ (self.relative_session_offsets - integers)
        )
        scaled_offsets = np.linalg.solve(
            conditional_precision,
            -conditional_gradient,
        )
        shape_residuals = (
            self.profile.normalized_residuals
            + self.shape_jacobian @ scaled_offsets
        )
        ambiguity_residuals = (
            self.relative_session_offsets
            + self.relative_session_jacobian @ scaled_offsets
            - integers
        )
        linearized_chi2 = float(
            shape_residuals @ shape_residuals
            + ambiguity_residuals @ noise_precision @ ambiguity_residuals
        )
        return ConditionalSessionMode(
            integer_offsets=tuple(int(value) for value in integer_offsets),
            scaled_parameter_offsets=scaled_offsets,
            parameter_values=(
                self.parameter_values + self.parameter_steps * scaled_offsets
            ),
            linearized_chi2=linearized_chi2,
        )


def integer_session_offsets_to_gap_wraps(
    integer_offsets: Sequence[int],
) -> tuple[int, ...]:
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.asarray(integer_offsets, dtype=np.int64))
    )
    return tuple(int(value) for value in np.diff(cumulative))


class PintSessionShapeLikelihood:
    def __init__(
        self,
        *,
        model: object,
        toas: object,
        parameter_names: Sequence[str],
        session_toa_indices: Sequence[Sequence[int]],
    ) -> None:
        self._model = model
        self._toas = toas
        self.parameter_names = tuple(str(name).upper() for name in parameter_names)
        self.session_toa_indices = tuple(
            tuple(int(index) for index in indices)
            for indices in session_toa_indices
        )
        missing = [name for name in self.parameter_names if name not in model.params]
        if missing:
            raise ValueError(f"PINT model is missing parameters: {', '.join(missing)}")
        self.base_parameter_values = np.asarray(
            [float(model[name].value) for name in self.parameter_names],
            dtype=float,
        )

    @classmethod
    def from_files(
        cls,
        *,
        par_path: Path,
        tim_path: Path,
        cluster_manifest_path: Path,
        parameter_names: Sequence[str],
        clock_override_path: Path | None = None,
    ) -> "PintSessionShapeLikelihood":
        par_path = Path(par_path)
        tim_path = Path(tim_path)
        model = _load_pint_model(par_path)
        toas = _load_pint_toas(
            tim_path,
            seed_model=model,
            clock_override_path=clock_override_path,
        )
        manifest = resolve_source_session_manifest(
            load_cluster_manifest(Path(cluster_manifest_path)),
            tim_path,
        )
        ensure_local_island_pulse_numbers(toas, model, manifest)
        return cls(
            model=model,
            toas=toas,
            parameter_names=parameter_names,
            session_toa_indices=[cluster.toa_indices for cluster in manifest.clusters],
        )

    def evaluate_offsets(self, parameter_offsets: Sequence[float]) -> SessionShapeProfile:
        offsets = np.asarray(parameter_offsets, dtype=float)
        if offsets.shape != self.base_parameter_values.shape:
            raise ValueError(
                "Parameter offsets must match the configured parameter dimension."
            )
        return self.evaluate_values(self.base_parameter_values + offsets)

    def evaluate_values(self, parameter_values: Sequence[float]) -> SessionShapeProfile:
        values = np.asarray(parameter_values, dtype=float)
        if values.shape != self.base_parameter_values.shape:
            raise ValueError(
                "Parameter values must match the configured parameter dimension."
            )
        if not _values_in_model_domain(self.parameter_names, values):
            raise ValueError("Parameter values are outside the supported DD model domain.")
        candidate_model = deepcopy(self._model)
        for name, value in zip(self.parameter_names, values, strict=True):
            _set_numeric_model_parameter_value(candidate_model, name, float(value))
        residuals = Residuals(
            self._toas,
            candidate_model,
            subtract_mean=False,
            use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        phase_residuals = np.asarray(
            residuals.phase_resids.to_value(u.dimensionless_unscaled),
            dtype=float,
        )
        try:
            uncertainties = candidate_model.scaled_toa_uncertainty(self._toas)
        except (AttributeError, TypeError, ValueError):
            uncertainties = self._toas.get_errors()
        sigma_seconds = np.asarray(uncertainties.to_value(u.s), dtype=float)
        spin_hz = abs(float(candidate_model["F0"].value))
        phase_sigmas = np.maximum(sigma_seconds * spin_hz, 1.0e-18)
        return profile_session_phase_offsets(
            phase_residuals,
            phase_sigmas,
            self.session_toa_indices,
            session_time_coordinates=np.asarray(
                self._toas.get_mjds().to_value(u.day), dtype=float
            ),
        )

    def linearize_values(
        self,
        parameter_values: Sequence[float],
        *,
        parameter_steps: Sequence[float],
    ) -> SessionShapeLinearization:
        values = np.asarray(parameter_values, dtype=float)
        steps = np.asarray(parameter_steps, dtype=float)
        if values.shape != self.base_parameter_values.shape:
            raise ValueError(
                "Parameter values must match the configured parameter dimension."
            )
        if steps.shape != values.shape or not np.all(np.isfinite(steps) & (steps > 0.0)):
            raise ValueError("Parameter steps must be finite, positive, and dimensionally matched.")

        profile = self.evaluate_values(values)
        shape_jacobian = np.empty(
            (profile.normalized_residuals.size, values.size),
            dtype=float,
        )
        offset_jacobian = np.empty(
            (profile.session_count, values.size),
            dtype=float,
        )
        observable_slope = np.asarray(profile.session_slope_observable, dtype=bool)
        standardized_slope_jacobian = np.empty(
            (int(np.count_nonzero(observable_slope)), values.size),
            dtype=float,
        )

        def standardized_slopes(candidate: SessionShapeProfile) -> np.ndarray:
            slopes = np.asarray(candidate.session_slopes, dtype=float)
            variances = np.asarray(candidate.session_slope_variances, dtype=float)
            if slopes.shape != observable_slope.shape or variances.shape != observable_slope.shape:
                raise ValueError("Finite-difference slope profiles changed session dimension.")
            selected_slopes = slopes[observable_slope]
            selected_variances = variances[observable_slope]
            if not np.all(
                np.isfinite(selected_slopes)
                & np.isfinite(selected_variances)
                & (selected_variances > 0.0)
            ):
                raise ValueError("Finite-difference probes changed slope observability.")
            return selected_slopes / np.sqrt(selected_variances)

        profile_standardized_slopes = standardized_slopes(profile)
        for parameter_index, step in enumerate(steps):
            displacement = np.zeros_like(values)
            displacement[parameter_index] = step

            # DD has hard physical boundaries.  A central difference is used
            # whenever possible; near a boundary, use the valid one-sided
            # difference instead of allowing one bad probe to abort a search.
            plus = None
            minus = None
            plus_values = values + displacement
            minus_values = values - displacement
            if _values_in_model_domain(self.parameter_names, plus_values):
                try:
                    plus = self.evaluate_values(plus_values)
                except (ArithmeticError, ValueError, RuntimeError):
                    plus = None
            if _values_in_model_domain(self.parameter_names, minus_values):
                try:
                    minus = self.evaluate_values(minus_values)
                except (ArithmeticError, ValueError, RuntimeError):
                    minus = None

            if plus is not None and minus is not None:
                shape_jacobian[:, parameter_index] = (
                    plus.normalized_residuals - minus.normalized_residuals
                ) / 2.0
                offset_jacobian[:, parameter_index] = (
                    plus.session_offsets - minus.session_offsets
                ) / 2.0
                standardized_slope_jacobian[:, parameter_index] = (
                    standardized_slopes(plus)
                    - standardized_slopes(minus)
                ) / 2.0
            elif plus is not None:
                shape_jacobian[:, parameter_index] = (
                    plus.normalized_residuals - profile.normalized_residuals
                )
                offset_jacobian[:, parameter_index] = (
                    plus.session_offsets - profile.session_offsets
                )
                standardized_slope_jacobian[:, parameter_index] = (
                    standardized_slopes(plus)
                    - profile_standardized_slopes
                )
            elif minus is not None:
                shape_jacobian[:, parameter_index] = (
                    profile.normalized_residuals - minus.normalized_residuals
                )
                offset_jacobian[:, parameter_index] = (
                    profile.session_offsets - minus.session_offsets
                )
                standardized_slope_jacobian[:, parameter_index] = (
                    profile_standardized_slopes
                    - standardized_slopes(minus)
                )
            else:
                raise ValueError(
                    f"No valid finite-difference probe for parameter "
                    f"{self.parameter_names[parameter_index]}."
                )

        relative_offsets = profile.session_offsets[1:] - profile.session_offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        relative_covariance = np.diag(profile.session_offset_variances[1:])
        relative_covariance += profile.session_offset_variances[0]
        marginalized_covariance = build_marginalized_integer_covariance(
            shape_jacobian,
            relative_jacobian,
            relative_covariance,
        )
        return SessionShapeLinearization(
            parameter_values=values.copy(),
            parameter_steps=steps.copy(),
            profile=profile,
            shape_jacobian=shape_jacobian,
            session_offset_jacobian=offset_jacobian,
            relative_session_offsets=relative_offsets,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
            marginalized_integer_covariance=marginalized_covariance,
            standardized_slope_jacobian=standardized_slope_jacobian,
        )
