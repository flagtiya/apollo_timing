"""Gaussian phase noise with independent and observation-common scatter.

The two-level factorization integrates one common phase origin, preserving
the noise-dependent normalization needed when comparing covariance scales.
Observation offsets are Gaussian random effects, not free fitted jumps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import solve_triangular

from apollo_timing.inference.bayesian_spin_integer_orbit import SpinOrbitLinearization
from apollo_timing.inference.wrap_invariant_binary_likelihood import SessionShapeProfile


@dataclass(frozen=True)
class ObservationPhaseNoise:
    reported_sigmas: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    independent_sigma: float = 0.0
    common_sigma: float = 0.0

    def __post_init__(self) -> None:
        sigmas = np.asarray(self.reported_sigmas, dtype=float)
        groups = tuple(tuple(int(i) for i in group) for group in self.groups)
        if sigmas.ndim != 1 or not np.all(np.isfinite(sigmas) & (sigmas > 0)):
            raise ValueError("Reported phase errors must be a positive finite vector.")
        if not groups or any(not group for group in groups):
            raise ValueError("At least one nonempty observation is required.")
        if sorted(i for group in groups for i in group) != list(range(sigmas.size)):
            raise ValueError("Observations must partition every TOA exactly once.")
        scales = np.array([self.independent_sigma, self.common_sigma], dtype=float)
        if not np.all(np.isfinite(scales) & (scales >= 0)):
            raise ValueError("Noise scales must be finite and nonnegative.")
        object.__setattr__(self, "reported_sigmas", sigmas.copy())
        object.__setattr__(self, "groups", groups)

    @property
    def diagonal_variances(self) -> np.ndarray:
        return self.reported_sigmas**2 + float(self.independent_sigma)**2

    @property
    def anchor_variances(self) -> np.ndarray:
        inverse = 1 / self.diagonal_variances
        return np.array([1 / inverse[list(group)].sum() for group in self.groups])

    @property
    def relative_covariance(self) -> np.ndarray:
        variances = self.anchor_variances + float(self.common_sigma)**2
        return np.diag(variances[1:]) + variances[0]

    @property
    def shape_log_normalization(self) -> float:
        # Jacobian of weighted local contrasts and relative observation means.
        return float(-.5 * (
            np.log(self.diagonal_variances).sum()
            - np.log(self.anchor_variances).sum()
            + (self.reported_sigmas.size - 1) * np.log(2 * np.pi)
        ))

    def _profile(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        if values.ndim not in (1, 2) or values.shape[0] != self.reported_sigmas.size:
            raise ValueError("Phase data or derivatives must align with TOAs.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Phase data and derivatives must be finite.")
        diagonal = self.diagonal_variances
        anchors = []
        centered = np.empty_like(values)
        for group in self.groups:
            indices = list(group)
            mean = np.average(values[indices], weights=1 / diagonal[indices], axis=0)
            anchors.append(mean)
            centered[indices] = values[indices] - mean
        divisor = np.sqrt(diagonal)
        return centered / (divisor if values.ndim == 1 else divisor[:, None]), np.asarray(anchors)

    def linearize(
        self, phases: Sequence[float], jacobian: np.ndarray, reference: Sequence[float],
    ) -> SpinOrbitLinearization:
        shape, anchors = self._profile(np.asarray(phases, dtype=float))
        shape_jacobian, anchor_jacobian = self._profile(jacobian)
        reference = np.asarray(reference, dtype=float)
        if reference.ndim != 1 or shape_jacobian.shape != (shape.size, reference.size):
            raise ValueError("Timing reference and phase derivatives must align.")
        return SpinOrbitLinearization(
            profile=SessionShapeProfile(
                projected_phase_residuals=shape * np.sqrt(self.diagonal_variances),
                normalized_residuals=shape, session_offsets=anchors,
                session_offset_variances=self.anchor_variances,
                chi2=float(shape @ shape), dof=shape.size-len(self.groups),
                session_count=len(self.groups),
            ),
            spin_reference=reference, shape_jacobian=shape_jacobian,
            relative_session_offsets=anchors[1:]-anchors[0],
            relative_session_jacobian=anchor_jacobian[1:]-anchor_jacobian[0],
            relative_offset_covariance=self.relative_covariance,
        )

    def whitened_residuals(self, phases: Sequence[float]) -> np.ndarray:
        shape, anchors = self._profile(np.asarray(phases, dtype=float))
        relative = solve_triangular(
            np.linalg.cholesky(self.relative_covariance), anchors[1:]-anchors[0], lower=True,
        )
        return np.r_[shape, relative]

    def log_likelihood(self, phases: Sequence[float]) -> float:
        residual = self.whitened_residuals(phases)
        _, logdet = np.linalg.slogdet(self.relative_covariance)
        return float(self.shape_log_normalization - .5*(residual @ residual + logdet))

    def phase_origin(self, phases: Sequence[float]) -> float:
        _, anchors = self._profile(np.asarray(phases, dtype=float))
        return float(np.average(anchors, weights=1/(self.anchor_variances+self.common_sigma**2)))

    def profile_log_likelihood(self, phases: Sequence[float]) -> float:
        """Full normalized Gaussian likelihood at its fitted common origin.

        This profiles the origin; ``log_likelihood`` integrates it. Use the profiled
        likelihood for maximum-likelihood criteria such as BIC and count the origin
        as one parameter. Random observation offsets are integrated through the
        covariance, with no free fitted jumps.
        """
        precision = np.sum(1/(self.anchor_variances+self.common_sigma**2))
        return float(self.log_likelihood(phases)+.5*np.log(precision/(2*np.pi)))

    def covariance_matrix(self) -> np.ndarray:
        matrix = np.diag(self.diagonal_variances)
        for group in self.groups:
            matrix[np.ix_(group, group)] += float(self.common_sigma)**2
        return matrix
