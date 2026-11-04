from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    DualLatticeCoordinateSystem,
    DualLatticeSupport,
    OrbitalCycleCell,
    PintDualLatticePhaseEvaluator,
    _session_shape_profile,
)
from apollo_timing.inference.bayesian_spin_integer_orbit import (
    SpinIntegerMarginalConfig,
    SpinIntegerMarginalEvaluation,
    SpinIntegerPathMode,
    SpinOrbitLinearization,
    marginalize_spin_and_integer_paths,
    restrict_linearization_to_phase_islands,
)
from apollo_timing.inference.wrap_invariant_binary_likelihood import (
    SessionShapeProfile,
)


@dataclass(frozen=True)
class DualLatticeSlowPosteriorConfig:
    """Controls for a slow-orbit posterior with fast coordinates integrated."""

    fast_shape_max_nfev: int = 32
    fast_finite_difference_step: float = 2.0e-4
    fast_prior_sigma: float = 0.5
    lattice: SpinIntegerMarginalConfig = SpinIntegerMarginalConfig()


@dataclass(frozen=True)
class DualLatticeSlowEvaluation:
    slow_normalized_values: np.ndarray
    fast_shape_values: np.ndarray
    shape_chi2: float
    marginal: SpinIntegerMarginalEvaluation


@dataclass(frozen=True)
class DualLatticeSlowMarginalLikelihood:
    """Bayesian target over four DD slow coordinates.

    At every slow point, the four near-linear fast coordinates are first
    centered on the exact session-shape likelihood and then integrated jointly
    with the complete session-anchor integer lattice.
    """

    evaluator: PintDualLatticePhaseEvaluator
    coordinates: DualLatticeCoordinateSystem
    support: DualLatticeSupport
    cell: OrbitalCycleCell
    local_integer_offsets: tuple[tuple[int, ...], ...]
    reference_fast_values: np.ndarray
    slow_support_lower: np.ndarray
    slow_support_upper: np.ndarray
    config: DualLatticeSlowPosteriorConfig = DualLatticeSlowPosteriorConfig()
    session_partitions: tuple[tuple[int, ...], ...] | None = None

    @property
    def orbit_lower(self) -> np.ndarray:
        return np.asarray(self.slow_support_lower, dtype=float).copy()

    @property
    def orbit_upper(self) -> np.ndarray:
        return np.asarray(self.slow_support_upper, dtype=float).copy()

    def _validated_slow(self, values: Sequence[float]) -> np.ndarray:
        slow = np.asarray(values, dtype=float)
        lower = self.orbit_lower
        upper = self.orbit_upper
        if slow.shape != (4,) or lower.shape != (4,) or upper.shape != (4,):
            raise ValueError("Dual-lattice slow coordinates must have dimension four.")
        if not np.all(np.isfinite(slow)) or np.any(slow < lower) or np.any(
            slow > upper
        ):
            raise ValueError("Slow coordinates lie outside declared support.")
        return slow

    def _profile(
        self,
        slow_normalized: np.ndarray,
        fast_normalized: np.ndarray,
    ):
        full = np.concatenate((slow_normalized, fast_normalized))
        slow, fast = self.support.denormalize(full)
        physical = self.coordinates.combine_physical(slow, fast)
        if not self.support.contains_physical(physical):
            raise ValueError("A slow-posterior point lies outside physical support.")
        return self.evaluator.evaluate_fixed_local_component(
            physical,
            self.local_integer_offsets,
        )

    def optimize_fast_shape(
        self,
        slow_normalized_values: Sequence[float],
    ) -> tuple[np.ndarray, float]:
        slow = self._validated_slow(slow_normalized_values)
        start = np.clip(
            np.asarray(self.reference_fast_values, dtype=float),
            -1.0,
            1.0,
        )
        if start.shape != (4,):
            raise ValueError("The reference fast coordinate must have dimension four.")
        try:
            residual_size = int(
                self._profile(slow, start).normalized_local_residuals.size
            )
        except Exception:
            residual_size = len(self.evaluator.toas.table)
        failure = np.full(residual_size, 1.0e8, dtype=float)

        def residuals(fast: np.ndarray) -> np.ndarray:
            try:
                value = np.asarray(
                    self._profile(slow, fast).normalized_local_residuals,
                    dtype=float,
                )
                return (
                    value
                    if value.shape == failure.shape and np.all(np.isfinite(value))
                    else failure
                )
            except (ArithmeticError, RuntimeError, ValueError):
                return failure

        optimization = least_squares(
            residuals,
            start,
            bounds=(np.full(4, -1.0), np.full(4, 1.0)),
            method="trf",
            jac="2-point",
            x_scale="jac",
            ftol=1.0e-9,
            xtol=1.0e-9,
            gtol=1.0e-9,
            max_nfev=max(int(self.config.fast_shape_max_nfev), 8),
        )
        fast = np.asarray(optimization.x, dtype=float)
        profile = self._profile(slow, fast)
        return fast, float(profile.local_chi2)

    def linearize(
        self,
        slow_normalized_values: Sequence[float],
    ) -> SpinOrbitLinearization:
        slow_normalized = self._validated_slow(slow_normalized_values)
        fast_normalized, _ = self.optimize_fast_shape(slow_normalized)
        profile = self._profile(slow_normalized, fast_normalized)
        shape = np.asarray(profile.normalized_local_residuals, dtype=float)
        offsets = np.asarray(profile.session_intercepts, dtype=float)
        shape_jacobian = np.empty((shape.size, 4), dtype=float)
        offset_jacobian = np.empty((offsets.size, 4), dtype=float)
        requested_step = max(
            float(self.config.fast_finite_difference_step),
            1.0e-7,
        )
        for column in range(4):
            plus = fast_normalized.copy()
            minus = fast_normalized.copy()
            plus[column] = min(plus[column] + requested_step, 1.0)
            minus[column] = max(minus[column] - requested_step, -1.0)
            denominator = float(plus[column] - minus[column])
            if denominator <= 0.0:
                raise ValueError("A fast coordinate is fixed at its support boundary.")
            plus_profile = self._profile(slow_normalized, plus)
            minus_profile = self._profile(slow_normalized, minus)
            shape_jacobian[:, column] = (
                np.asarray(plus_profile.normalized_local_residuals, dtype=float)
                - np.asarray(minus_profile.normalized_local_residuals, dtype=float)
            ) / denominator
            offset_jacobian[:, column] = (
                np.asarray(plus_profile.session_intercepts, dtype=float)
                - np.asarray(minus_profile.session_intercepts, dtype=float)
            ) / denominator
        relative_offsets = offsets[1:] - offsets[0]
        relative_jacobian = offset_jacobian[1:] - offset_jacobian[0]
        variances = np.asarray(profile.session_intercept_variances, dtype=float)
        relative_covariance = np.diag(variances[1:])
        relative_covariance += float(variances[0])
        return SpinOrbitLinearization(
            profile=_session_shape_profile(profile),
            spin_reference=fast_normalized.copy(),
            shape_jacobian=shape_jacobian,
            relative_session_offsets=relative_offsets,
            relative_session_jacobian=relative_jacobian,
            relative_offset_covariance=relative_covariance,
        )

    def evaluate(
        self,
        slow_normalized_values: Sequence[float],
    ) -> DualLatticeSlowEvaluation:
        slow = self._validated_slow(slow_normalized_values)
        linearization = self.linearize(slow)
        if self.session_partitions is not None:
            linearization = restrict_linearization_to_phase_islands(
                linearization,
                self.session_partitions,
            )
        sigma = max(float(self.config.fast_prior_sigma), 1.0e-6)
        marginal = marginalize_spin_and_integer_paths(
            linearization,
            spin_prior_mean=np.zeros(4, dtype=float),
            spin_prior_covariance=np.diag(np.full(4, sigma * sigma)),
            config=self.config.lattice,
        )
        residuals = np.asarray(
            linearization.profile.normalized_residuals,
            dtype=float,
        )
        return DualLatticeSlowEvaluation(
            slow_normalized_values=slow.copy(),
            fast_shape_values=np.asarray(
                linearization.spin_reference,
                dtype=float,
            ).copy(),
            shape_chi2=float(residuals @ residuals),
            marginal=marginal,
        )

    def __call__(self, slow_normalized_values: Sequence[float]) -> float:
        try:
            value = self.evaluate(slow_normalized_values).marginal.log_likelihood
        except (ArithmeticError, RuntimeError, ValueError):
            return -math.inf
        return float(value) if np.isfinite(value) else -math.inf

    def full_normalized_values(
        self,
        evaluation: DualLatticeSlowEvaluation,
        path_mode: SpinIntegerPathMode,
    ) -> np.ndarray:
        fast = evaluation.fast_shape_values + np.asarray(
            path_mode.spin_posterior_mean,
            dtype=float,
        )
        return np.concatenate((evaluation.slow_normalized_values, fast))

    def exact_profile_at_full_values(
        self,
        normalized_values: Sequence[float],
    ) -> SessionShapeProfile:
        full = np.asarray(normalized_values, dtype=float)
        if full.shape != (8,) or np.any(full < -1.0) or np.any(full > 1.0):
            raise ValueError("Full dual-lattice values lie outside support.")
        profile = self._profile(full[:4], full[4:])
        return _session_shape_profile(profile)


__all__ = [
    "DualLatticeSlowEvaluation",
    "DualLatticeSlowMarginalLikelihood",
    "DualLatticeSlowPosteriorConfig",
]
