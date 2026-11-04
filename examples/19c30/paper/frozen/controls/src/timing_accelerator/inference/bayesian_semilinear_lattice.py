from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
from scipy.special import logsumexp

from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    SpinFixedIntegerPathEvaluation,
    SpinIntegerMarginalConfig,
    SpinIntegerMarginalEvaluation,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
    marginalize_spin_for_fixed_integer_path,
)

from timing_accelerator.inference.information_ordered_binary_search import (
    DD_REQUIRED_PARAMETERS,
    DDSearchCoordinateSystem,
    SpinSupportAudit,
    build_dd_search_coordinate_system,
    expand_reference_epoch_spin_support,
)


SEMILINEAR_SLOW_NAMES = ("LOG_PB", "PHI_REF", "LOGIT_ECC", "OMDOT")
SEMILINEAR_FAST_NAMES = ("F0_REF", "F1", "A1_SIN_OMEGA", "A1_COS_OMEGA")


def _wrap_near(value: float, center: float) -> float:
    return float(center + (value - center + math.pi) % (2.0 * math.pi) - math.pi)


def _validate_vector(
    values: Sequence[float],
    *,
    dimension: int,
    label: str,
) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector with dimension {dimension}.")
    return result


@dataclass(frozen=True)
class SemilinearSupportConfig:
    """Generic recoverability support for the hard-source solver."""

    minimum_f1_half_width: float = 1.0e-10


@dataclass(frozen=True)
class SemilinearSupportAudit:
    physical_lower: np.ndarray
    physical_upper: np.ndarray
    slow_lower: np.ndarray
    slow_upper: np.ndarray
    fast_lower: np.ndarray
    fast_upper: np.ndarray
    spin: SpinSupportAudit
    bound_sources: tuple[str, ...]


@dataclass(frozen=True)
class SemilinearMarginalConfig:
    """Controls for one fixed-slow conditional lattice evaluation."""

    finite_difference_step: float = 2.0e-4
    fast_prior_standard_deviation: float = 1.0
    closure_path_count: int = 16
    maximum_relinearization_steps: int = 4
    maximum_shape_error: float = 0.25
    maximum_phase_error: float = 0.25
    maximum_delta_chi2: float = 1.0
    lattice: SpinIntegerMarginalConfig = field(
        default_factory=SpinIntegerMarginalConfig
    )


@dataclass(frozen=True)
class SemilinearPathClosure:
    integer_offsets: tuple[int, ...]
    fast_normalized_values: np.ndarray
    parameter_values: np.ndarray | None
    predicted_chi2: float
    exact_shape_chi2: float
    exact_phase_chi2: float
    exact_chi2: float
    shape_error: float
    phase_error: float
    delta_chi2: float
    relinearization_steps: int
    converged: bool
    failure_reason: str | None
    log_laplace_weight: float


@dataclass(frozen=True)
class SemilinearMarginalEvaluation:
    slow_normalized_values: np.ndarray
    surrogate_log_likelihood: float
    exact_corrected_log_likelihood: float
    lattice_mass_converged: bool
    lattice_log_mass_increment: float
    lattice_visited_nodes: int
    retained_lattice_candidates: int
    trusted_path_count: int
    path_closures: tuple[SemilinearPathClosure, ...]


@dataclass(frozen=True)
class _FastLinearization:
    slow_normalized_values: np.ndarray
    fast_reference_normalized_values: np.ndarray
    spin: SpinOrbitLinearization


@dataclass(frozen=True)
class SemilinearDDCoordinateSystem:
    """Split a DD chart into four nonlinear and four conditional coordinates."""

    full: DDSearchCoordinateSystem
    slow_center: np.ndarray
    slow_lower: np.ndarray
    slow_upper: np.ndarray
    slow_scales: np.ndarray
    fast_center: np.ndarray
    fast_lower: np.ndarray
    fast_upper: np.ndarray
    fast_scales: np.ndarray

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.full.parameter_names)

    def _full_index(self, name: str) -> int:
        return self.full.parameter_names.index(name)

    def split_physical(
        self,
        parameter_values: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        physical = _validate_vector(
            parameter_values,
            dimension=len(self.full.parameter_names),
            label="DD physical parameters",
        )
        search = self.full.to_search(physical)
        a1 = math.exp(float(search[self._full_index("A1")]))
        omega = float(search[self._full_index("OM")])
        slow = np.asarray(
            (
                search[self._full_index("PB")],
                search[self._full_index("T0")],
                search[self._full_index("ECC")],
                search[self._full_index("OMDOT")],
            ),
            dtype=float,
        )
        fast = np.asarray(
            (
                search[self._full_index("F0")],
                search[self._full_index("F1")],
                a1 * math.sin(omega),
                a1 * math.cos(omega),
            ),
            dtype=float,
        )
        return slow, fast

    def combine_physical(
        self,
        slow_values: Sequence[float],
        fast_values: Sequence[float],
        *,
        require_support: bool = True,
    ) -> np.ndarray:
        slow = _validate_vector(
            slow_values,
            dimension=len(SEMILINEAR_SLOW_NAMES),
            label="DD slow coordinates",
        )
        fast = _validate_vector(
            fast_values,
            dimension=len(SEMILINEAR_FAST_NAMES),
            label="DD fast coordinates",
        )
        if require_support:
            if np.any(slow < self.slow_lower) or np.any(slow > self.slow_upper):
                raise ValueError("DD slow coordinates lie outside support.")
            if np.any(fast < self.fast_lower) or np.any(fast > self.fast_upper):
                raise ValueError("DD fast coordinates lie outside support.")

        a1 = math.hypot(float(fast[2]), float(fast[3]))
        if not np.isfinite(a1) or a1 <= 0.0:
            raise ValueError("The semilinear A1 components must have positive norm.")
        omega = _wrap_near(
            math.atan2(float(fast[2]), float(fast[3])),
            float(self.full.search_center[self._full_index("OM")]),
        )
        search = np.asarray(self.full.search_center, dtype=float).copy()
        search[self._full_index("PB")] = slow[0]
        search[self._full_index("T0")] = slow[1]
        search[self._full_index("ECC")] = slow[2]
        search[self._full_index("OMDOT")] = slow[3]
        search[self._full_index("F0")] = fast[0]
        search[self._full_index("F1")] = fast[1]
        search[self._full_index("A1")] = math.log(a1)
        search[self._full_index("OM")] = omega
        physical = self.full.to_physical(search)

        if require_support:
            tolerance = 64.0 * np.finfo(float).eps * np.maximum(
                np.maximum(np.abs(self.full.physical_lower), np.abs(self.full.physical_upper)),
                1.0,
            )
            if np.any(physical < self.full.physical_lower - tolerance) or np.any(
                physical > self.full.physical_upper + tolerance
            ):
                raise ValueError("Combined DD parameters lie outside physical support.")
        return physical

    def normalize_slow(self, values: Sequence[float]) -> np.ndarray:
        slow = _validate_vector(
            values,
            dimension=len(SEMILINEAR_SLOW_NAMES),
            label="DD slow coordinates",
        )
        return (slow - self.slow_center) / self.slow_scales

    def denormalize_slow(self, values: Sequence[float]) -> np.ndarray:
        normalized = _validate_vector(
            values,
            dimension=len(SEMILINEAR_SLOW_NAMES),
            label="Normalized DD slow coordinates",
        )
        return self.slow_center + self.slow_scales * normalized

    def normalize_fast(self, values: Sequence[float]) -> np.ndarray:
        fast = _validate_vector(
            values,
            dimension=len(SEMILINEAR_FAST_NAMES),
            label="DD fast coordinates",
        )
        return (fast - self.fast_center) / self.fast_scales

    def denormalize_fast(self, values: Sequence[float]) -> np.ndarray:
        normalized = _validate_vector(
            values,
            dimension=len(SEMILINEAR_FAST_NAMES),
            label="Normalized DD fast coordinates",
        )
        return self.fast_center + self.fast_scales * normalized

    @property
    def normalized_slow_lower(self) -> np.ndarray:
        return self.normalize_slow(self.slow_lower)

    @property
    def normalized_slow_upper(self) -> np.ndarray:
        return self.normalize_slow(self.slow_upper)

    @property
    def normalized_fast_lower(self) -> np.ndarray:
        return self.normalize_fast(self.fast_lower)

    @property
    def normalized_fast_upper(self) -> np.ndarray:
        return self.normalize_fast(self.fast_upper)


@dataclass(frozen=True)
class SemilinearConditionalLatticeLikelihood:
    """All-session fast/integer marginal likelihood at a fixed slow state."""

    evaluator: Any
    coordinates: SemilinearDDCoordinateSystem
    config: SemilinearMarginalConfig = field(
        default_factory=SemilinearMarginalConfig
    )

    def _profile(self, slow_normalized: np.ndarray, fast_normalized: np.ndarray):
        slow = self.coordinates.denormalize_slow(slow_normalized)
        fast = self.coordinates.denormalize_fast(fast_normalized)
        physical = self.coordinates.combine_physical(slow, fast)
        return self.evaluator.evaluate_values(physical)

    def linearize(
        self,
        slow_normalized_values: Sequence[float],
        fast_reference_normalized_values: Sequence[float] | None = None,
    ) -> _FastLinearization:
        slow = _validate_vector(
            slow_normalized_values,
            dimension=4,
            label="Normalized DD slow coordinates",
        )
        slow_lower = self.coordinates.normalized_slow_lower
        slow_upper = self.coordinates.normalized_slow_upper
        if np.any(slow < slow_lower) or np.any(slow > slow_upper):
            raise ValueError("Normalized DD slow coordinates lie outside support.")
        fast = (
            np.zeros(4, dtype=float)
            if fast_reference_normalized_values is None
            else _validate_vector(
                fast_reference_normalized_values,
                dimension=4,
                label="Normalized DD fast reference",
            )
        )
        fast_lower = self.coordinates.normalized_fast_lower
        fast_upper = self.coordinates.normalized_fast_upper
        if np.any(fast < fast_lower) or np.any(fast > fast_upper):
            raise ValueError("Normalized DD fast reference lies outside support.")

        profile = self._profile(slow, fast)
        shape = np.asarray(profile.normalized_residuals, dtype=float)
        offsets = np.asarray(profile.session_offsets, dtype=float)
        if shape.ndim != 1 or offsets.ndim != 1 or offsets.size < 2:
            raise ValueError("A semilinear profile requires shape data and two sessions.")
        shape_jacobian = np.empty((shape.size, fast.size), dtype=float)
        offset_jacobian = np.empty((offsets.size, fast.size), dtype=float)
        step = max(float(self.config.finite_difference_step), 1.0e-7)
        for index in range(fast.size):
            plus = fast.copy()
            minus = fast.copy()
            plus[index] = min(plus[index] + step, fast_upper[index])
            minus[index] = max(minus[index] - step, fast_lower[index])
            denominator = float(plus[index] - minus[index])
            if denominator <= 0.0:
                raise ValueError("A conditional-fast coordinate is fixed at support.")
            plus_profile = self._profile(slow, plus)
            minus_profile = self._profile(slow, minus)
            plus_shape = np.asarray(plus_profile.normalized_residuals, dtype=float)
            minus_shape = np.asarray(minus_profile.normalized_residuals, dtype=float)
            plus_offsets = np.asarray(plus_profile.session_offsets, dtype=float)
            minus_offsets = np.asarray(minus_profile.session_offsets, dtype=float)
            if plus_shape.shape != shape.shape or minus_shape.shape != shape.shape:
                raise ValueError("Fast probes changed the shape residual dimension.")
            if plus_offsets.shape != offsets.shape or minus_offsets.shape != offsets.shape:
                raise ValueError("Fast probes changed the session dimension.")
            shape_jacobian[:, index] = (plus_shape - minus_shape) / denominator
            offset_jacobian[:, index] = (plus_offsets - minus_offsets) / denominator

        relative = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        variances = np.asarray(profile.session_offset_variances, dtype=float)
        if variances.shape != offsets.shape or not np.all(
            np.isfinite(variances) & (variances > 0.0)
        ):
            raise ValueError("Session offset variances must be finite and positive.")
        relative_covariance = np.diag(variances[1:])
        relative_covariance += float(variances[0])
        return _FastLinearization(
            slow_normalized_values=slow.copy(),
            fast_reference_normalized_values=fast.copy(),
            spin=SpinOrbitLinearization(
                profile=profile,
                spin_reference=fast.copy(),
                shape_jacobian=shape_jacobian,
                relative_session_offsets=relative,
                relative_session_jacobian=relative_jacobian,
                relative_offset_covariance=relative_covariance,
            ),
        )

    def _prior(
        self,
        fast_reference_normalized_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        standard_deviation = max(
            float(self.config.fast_prior_standard_deviation),
            1.0e-6,
        )
        return (
            -np.asarray(fast_reference_normalized_values, dtype=float),
            np.eye(4, dtype=float) * standard_deviation * standard_deviation,
        )

    def _fixed_path_prediction(
        self,
        linearization: _FastLinearization,
        path: tuple[int, ...],
        fixed: SpinFixedIntegerPathEvaluation,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        displacement = np.asarray(fixed.spin_posterior_mean, dtype=float)
        candidate = linearization.fast_reference_normalized_values + displacement
        shape = (
            np.asarray(linearization.spin.profile.normalized_residuals, dtype=float)
            + np.asarray(linearization.spin.shape_jacobian, dtype=float) @ displacement
        )
        relative = (
            np.asarray(linearization.spin.relative_session_offsets, dtype=float)
            + np.asarray(linearization.spin.relative_session_jacobian, dtype=float)
            @ displacement
        )
        covariance = np.asarray(
            linearization.spin.relative_offset_covariance,
            dtype=float,
        )
        phase_delta = relative - np.asarray(path, dtype=float)
        predicted_chi2 = float(
            shape @ shape + phase_delta @ np.linalg.solve(covariance, phase_delta)
        )
        return candidate, predicted_chi2, shape, relative

    def _exact_closure(
        self,
        *,
        linearization: _FastLinearization,
        path: tuple[int, ...],
        fixed: SpinFixedIntegerPathEvaluation,
        relinearization_steps: int,
    ) -> SemilinearPathClosure:
        candidate, predicted_chi2, predicted_shape, predicted_relative = (
            self._fixed_path_prediction(linearization, path, fixed)
        )
        fast_lower = self.coordinates.normalized_fast_lower
        fast_upper = self.coordinates.normalized_fast_upper
        if np.any(candidate < fast_lower) or np.any(candidate > fast_upper):
            return SemilinearPathClosure(
                integer_offsets=path,
                fast_normalized_values=candidate,
                parameter_values=None,
                predicted_chi2=predicted_chi2,
                exact_shape_chi2=math.inf,
                exact_phase_chi2=math.inf,
                exact_chi2=math.inf,
                shape_error=math.inf,
                phase_error=math.inf,
                delta_chi2=math.inf,
                relinearization_steps=relinearization_steps,
                converged=False,
                failure_reason="fast_support",
                log_laplace_weight=-math.inf,
            )
        try:
            slow = self.coordinates.denormalize_slow(
                linearization.slow_normalized_values
            )
            fast = self.coordinates.denormalize_fast(candidate)
            physical = self.coordinates.combine_physical(slow, fast)
            exact = self.evaluator.evaluate_values(physical)
        except (ArithmeticError, RuntimeError, ValueError):
            return SemilinearPathClosure(
                integer_offsets=path,
                fast_normalized_values=candidate,
                parameter_values=None,
                predicted_chi2=predicted_chi2,
                exact_shape_chi2=math.inf,
                exact_phase_chi2=math.inf,
                exact_chi2=math.inf,
                shape_error=math.inf,
                phase_error=math.inf,
                delta_chi2=math.inf,
                relinearization_steps=relinearization_steps,
                converged=False,
                failure_reason="pint_evaluation",
                log_laplace_weight=-math.inf,
            )

        exact_shape = np.asarray(exact.normalized_residuals, dtype=float)
        exact_offsets = np.asarray(exact.session_offsets, dtype=float)
        exact_variances = np.asarray(exact.session_offset_variances, dtype=float)
        exact_relative = exact_offsets[1:] - exact_offsets[0]
        exact_covariance = np.diag(exact_variances[1:])
        exact_covariance += float(exact_variances[0])
        phase_delta = exact_relative - np.asarray(path, dtype=float)
        exact_shape_chi2 = float(exact_shape @ exact_shape)
        exact_phase_chi2 = float(
            phase_delta @ np.linalg.solve(exact_covariance, phase_delta)
        )
        exact_chi2 = exact_shape_chi2 + exact_phase_chi2
        shape_error = float(np.linalg.norm(exact_shape - predicted_shape))
        phase_error_delta = exact_relative - predicted_relative
        phase_error = float(
            math.sqrt(
                max(
                    phase_error_delta
                    @ np.linalg.solve(exact_covariance, phase_error_delta),
                    0.0,
                )
            )
        )
        delta_chi2 = abs(exact_chi2 - predicted_chi2)
        converged = bool(
            shape_error <= float(self.config.maximum_shape_error)
            and phase_error <= float(self.config.maximum_phase_error)
            and delta_chi2 <= float(self.config.maximum_delta_chi2)
        )

        posterior_covariance = np.asarray(
            fixed.spin_posterior_covariance,
            dtype=float,
        )
        sign, logdet = np.linalg.slogdet(posterior_covariance)
        prior_sigma = max(float(self.config.fast_prior_standard_deviation), 1.0e-6)
        prior_quadratic = float(candidate @ candidate) / (prior_sigma * prior_sigma)
        log_prior = float(
            -0.5 * prior_quadratic - 4.0 * math.log(prior_sigma * math.sqrt(2.0 * math.pi))
        )
        log_weight = (
            float(
                -0.5 * exact_chi2
                + log_prior
                + 2.0 * math.log(2.0 * math.pi)
                + 0.5 * logdet
            )
            if converged and sign > 0.0 and np.isfinite(logdet)
            else -math.inf
        )
        return SemilinearPathClosure(
            integer_offsets=path,
            fast_normalized_values=candidate,
            parameter_values=np.asarray(physical, dtype=float),
            predicted_chi2=predicted_chi2,
            exact_shape_chi2=exact_shape_chi2,
            exact_phase_chi2=exact_phase_chi2,
            exact_chi2=exact_chi2,
            shape_error=shape_error,
            phase_error=phase_error,
            delta_chi2=delta_chi2,
            relinearization_steps=relinearization_steps,
            converged=converged,
            failure_reason=None if converged else "nonlinear_closure",
            log_laplace_weight=log_weight,
        )

    def refine_path(
        self,
        slow_normalized_values: Sequence[float],
        integer_offsets: Sequence[int],
        *,
        fast_start_normalized_values: Sequence[float] | None = None,
    ) -> SemilinearPathClosure:
        path = tuple(int(value) for value in integer_offsets)
        fast = (
            np.zeros(4, dtype=float)
            if fast_start_normalized_values is None
            else _validate_vector(
                fast_start_normalized_values,
                dimension=4,
                label="Normalized DD fast path start",
            )
        )
        latest: SemilinearPathClosure | None = None
        maximum_steps = max(int(self.config.maximum_relinearization_steps), 1)
        for step_index in range(1, maximum_steps + 1):
            try:
                linearization = self.linearize(slow_normalized_values, fast)
                prior_mean, prior_covariance = self._prior(fast)
                fixed = marginalize_spin_for_fixed_integer_path(
                    linearization.spin,
                    path,
                    spin_prior_mean=prior_mean,
                    spin_prior_covariance=prior_covariance,
                    added_session_phase_sigma_cycles=float(
                        self.config.lattice.added_session_phase_sigma_cycles
                    ),
                )
                latest = self._exact_closure(
                    linearization=linearization,
                    path=path,
                    fixed=fixed,
                    relinearization_steps=step_index,
                )
            except (ArithmeticError, RuntimeError, ValueError):
                break
            if latest.converged or latest.failure_reason != "nonlinear_closure":
                return latest
            fast = latest.fast_normalized_values
        if latest is not None:
            return latest
        return SemilinearPathClosure(
            integer_offsets=path,
            fast_normalized_values=fast,
            parameter_values=None,
            predicted_chi2=math.inf,
            exact_shape_chi2=math.inf,
            exact_phase_chi2=math.inf,
            exact_chi2=math.inf,
            shape_error=math.inf,
            phase_error=math.inf,
            delta_chi2=math.inf,
            relinearization_steps=0,
            converged=False,
            failure_reason="linearization",
            log_laplace_weight=-math.inf,
        )

    def evaluate(
        self,
        slow_normalized_values: Sequence[float],
    ) -> SemilinearMarginalEvaluation:
        linearization = self.linearize(slow_normalized_values)
        prior_mean, prior_covariance = self._prior(
            linearization.fast_reference_normalized_values
        )
        surrogate: SpinIntegerMarginalEvaluation = marginalize_spin_and_integer_paths(
            linearization.spin,
            spin_prior_mean=prior_mean,
            spin_prior_covariance=prior_covariance,
            config=self.config.lattice,
        )
        path_count = min(
            max(int(self.config.closure_path_count), 0),
            len(surrogate.path_modes),
        )
        closures = tuple(
            self.refine_path(
                slow_normalized_values,
                mode.integer_offsets,
                fast_start_normalized_values=(
                    linearization.fast_reference_normalized_values
                    + np.asarray(mode.spin_posterior_mean, dtype=float)
                ),
            )
            for mode in surrogate.path_modes[:path_count]
        )
        trusted_weights = np.asarray(
            [item.log_laplace_weight for item in closures if item.converged],
            dtype=float,
        )
        corrected = (
            float(logsumexp(trusted_weights))
            if trusted_weights.size > 0
            else -math.inf
        )
        return SemilinearMarginalEvaluation(
            slow_normalized_values=np.asarray(slow_normalized_values, dtype=float).copy(),
            surrogate_log_likelihood=float(surrogate.log_likelihood),
            exact_corrected_log_likelihood=corrected,
            lattice_mass_converged=bool(surrogate.lattice_mass_converged),
            lattice_log_mass_increment=float(surrogate.lattice_log_mass_increment),
            lattice_visited_nodes=int(surrogate.lattice_visited_nodes),
            retained_lattice_candidates=int(surrogate.retained_lattice_candidates),
            trusted_path_count=sum(item.converged for item in closures),
            path_closures=closures,
        )


def build_semilinear_coordinate_system(
    full: DDSearchCoordinateSystem,
) -> SemilinearDDCoordinateSystem:
    names = tuple(str(name).upper() for name in full.parameter_names)
    if set(names) != set(DD_REQUIRED_PARAMETERS) or len(names) != len(
        DD_REQUIRED_PARAMETERS
    ):
        raise ValueError("Semilinear DD coordinates require exactly the eight DD parameters.")
    index = {name: names.index(name) for name in names}
    placeholder = SemilinearDDCoordinateSystem(
        full=full,
        slow_center=np.zeros(4, dtype=float),
        slow_lower=np.zeros(4, dtype=float),
        slow_upper=np.ones(4, dtype=float),
        slow_scales=np.ones(4, dtype=float),
        fast_center=np.zeros(4, dtype=float),
        fast_lower=np.zeros(4, dtype=float),
        fast_upper=np.ones(4, dtype=float),
        fast_scales=np.ones(4, dtype=float),
    )
    slow_center, fast_center = placeholder.split_physical(full.center_values)
    slow_indices = (index["PB"], index["T0"], index["ECC"], index["OMDOT"])
    slow_lower = np.asarray(full.search_lower, dtype=float)[list(slow_indices)]
    slow_upper = np.asarray(full.search_upper, dtype=float)[list(slow_indices)]

    maximum_a1 = float(full.physical_upper[index["A1"]])
    fast_lower = np.asarray(
        (
            full.search_lower[index["F0"]],
            full.search_lower[index["F1"]],
            -maximum_a1,
            -maximum_a1,
        ),
        dtype=float,
    )
    fast_upper = np.asarray(
        (
            full.search_upper[index["F0"]],
            full.search_upper[index["F1"]],
            maximum_a1,
            maximum_a1,
        ),
        dtype=float,
    )
    slow_scales = np.maximum(
        np.maximum(slow_center - slow_lower, slow_upper - slow_center),
        np.finfo(float).eps,
    )
    fast_scales = np.maximum(
        np.maximum(fast_center - fast_lower, fast_upper - fast_center),
        np.finfo(float).eps,
    )
    return SemilinearDDCoordinateSystem(
        full=full,
        slow_center=slow_center,
        slow_lower=slow_lower,
        slow_upper=slow_upper,
        slow_scales=slow_scales,
        fast_center=fast_center,
        fast_lower=fast_lower,
        fast_upper=fast_upper,
        fast_scales=fast_scales,
    )


def build_generic_dd_physical_support(
    *,
    parameter_names: Sequence[str],
    center_values: Sequence[float],
    observation_span_days: float,
    width_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build broad DD model-domain bounds without source-class information."""

    names = tuple(str(name).upper() for name in parameter_names)
    if set(names) != set(DD_REQUIRED_PARAMETERS) or len(names) != 8:
        raise ValueError("Generic DD support requires exactly the eight DD parameters.")
    center = _validate_vector(
        center_values,
        dimension=8,
        label="DD support center",
    )
    span_days = float(observation_span_days)
    scale = float(width_scale)
    if not np.isfinite(span_days) or span_days <= 0.0:
        raise ValueError("DD support requires a positive observation span.")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("DD support width scale must be positive.")
    index = {name: names.index(name) for name in names}
    pb = float(center[index["PB"]])
    widths = {
        "F0": max(abs(float(center[index["F0"]])) * 1.0e-5, 5.0 / (span_days * 86400.0)),
        "F1": 3.0e-12,
        "A1": max(abs(float(center[index["A1"]])) * 0.08, 0.1),
        "ECC": 0.20,
        "T0": 0.55 * pb,
        "PB": max(abs(pb) * 0.03, 5.0e-4),
        "OM": 180.0,
        "OMDOT": 15.0,
    }
    lower = center.copy()
    upper = center.copy()
    for name, width in widths.items():
        parameter_index = index[name]
        lower[parameter_index] -= scale * width
        upper[parameter_index] += scale * width
    lower[index["F0"]] = max(lower[index["F0"]], np.finfo(float).tiny)
    lower[index["A1"]] = max(lower[index["A1"]], 1.0e-6)
    lower[index["PB"]] = max(lower[index["PB"]], 1.0e-6)
    lower[index["ECC"]] = max(lower[index["ECC"]], 1.0e-6)
    upper[index["ECC"]] = min(upper[index["ECC"]], 1.0 - 1.0e-6)
    return lower, upper


def build_audited_semilinear_coordinate_system(
    *,
    parameter_names: Sequence[str],
    center_values: Sequence[float],
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    observation_epochs_mjd: Sequence[float],
    pepoch_mjd: float,
    reference_mjd: float,
    config: SemilinearSupportConfig | None = None,
    bound_sample_count: int = 4096,
    seed: int = 0,
) -> tuple[SemilinearDDCoordinateSystem, SemilinearSupportAudit]:
    active = config or SemilinearSupportConfig()
    lower, upper, spin_audit = expand_reference_epoch_spin_support(
        parameter_names=parameter_names,
        center_values=center_values,
        lower_values=lower_values,
        upper_values=upper_values,
        observation_epochs_mjd=observation_epochs_mjd,
        reference_mjd=reference_mjd,
        recoverability_f1_half_width=float(active.minimum_f1_half_width),
    )
    full = build_dd_search_coordinate_system(
        parameter_names=parameter_names,
        center_values=center_values,
        lower_values=lower,
        upper_values=upper,
        pepoch_mjd=pepoch_mjd,
        reference_mjd=reference_mjd,
        bound_sample_count=bound_sample_count,
        seed=seed,
    )
    coordinates = build_semilinear_coordinate_system(full)
    audit = SemilinearSupportAudit(
        physical_lower=np.asarray(lower, dtype=float).copy(),
        physical_upper=np.asarray(upper, dtype=float).copy(),
        slow_lower=coordinates.slow_lower.copy(),
        slow_upper=coordinates.slow_upper.copy(),
        fast_lower=coordinates.fast_lower.copy(),
        fast_upper=coordinates.fast_upper.copy(),
        spin=spin_audit,
        bound_sources=(
            "initial_par_center",
            "tim_observation_epochs",
            "dd_model_domain",
            "generic_symmetric_f1_recoverability_floor",
        ),
    )
    return coordinates, audit


__all__ = [
    "SEMILINEAR_FAST_NAMES",
    "SEMILINEAR_SLOW_NAMES",
    "SemilinearDDCoordinateSystem",
    "SemilinearConditionalLatticeLikelihood",
    "SemilinearMarginalConfig",
    "SemilinearMarginalEvaluation",
    "SemilinearPathClosure",
    "SemilinearSupportAudit",
    "SemilinearSupportConfig",
    "build_audited_semilinear_coordinate_system",
    "build_generic_dd_physical_support",
    "build_semilinear_coordinate_system",
]
