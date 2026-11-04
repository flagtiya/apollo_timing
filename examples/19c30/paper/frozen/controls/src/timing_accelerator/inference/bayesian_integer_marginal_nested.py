from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import dynesty
import numpy as np
from scipy.special import ndtr, ndtri

from timing_accelerator.inference.bayesian_global_lattice_mode_tracker import (
    _physical_values,
)
from timing_accelerator.inference.collapsed_integer_posterior import (
    collapsed_wrapped_log_likelihood,
)


@dataclass(frozen=True)
class IntegerMarginalNestedConfig:
    live_point_count: int = 512
    evidence_tolerance: float = 0.25
    maximum_calls: int | None = None
    phase_noise_sigma_cycles: float = 0.03
    prior_precision: float = 0.0
    bound: str = "multi"
    sample: str = "rslice"
    random_seed: int = 20260725
    finalize_live_points: bool = False
    first_update_min_calls: int | None = None
    first_update_min_efficiency: float | None = None


@dataclass(frozen=True)
class ShapeIslandSupport:
    lower: np.ndarray
    upper: np.ndarray
    sample_lower_quantile: float
    sample_upper_quantile: float
    expansion_factor: float
    support_strategy: str = "shape_island_axis_box"
    reachability_sample_count: int = 0

    def transform(self, unit_values: Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=float)
        if unit.shape != self.lower.shape:
            raise ValueError("Nested unit coordinates do not match the support.")
        return self.lower + unit * (self.upper - self.lower)


@dataclass(frozen=True)
class ShapeIslandPriorTransform:
    """Pickleable prior transform used by dynesty checkpoints."""

    support: ShapeIslandSupport

    def __call__(self, unit_values: Sequence[float]) -> np.ndarray:
        return self.support.transform(unit_values)


@dataclass(frozen=True)
class PhaseReachableMixturePriorTransform:
    """Mixture prior over a broad component and all reachable local charts."""

    support: ShapeIslandSupport
    centers: np.ndarray
    scales: np.ndarray
    broad_component_weight: float

    def __call__(self, unit_values: Sequence[float]) -> np.ndarray:
        unit = np.asarray(unit_values, dtype=float)
        if unit.shape != self.support.lower.shape:
            raise ValueError("Mixture unit coordinates do not match support.")
        if not np.all(np.isfinite(unit) & (unit >= 0.0) & (unit <= 1.0)):
            raise ValueError("Mixture unit coordinates must lie in [0, 1].")
        broad_weight = float(self.broad_component_weight)
        if unit[0] < broad_weight:
            broad_unit = unit.copy()
            broad_unit[0] /= broad_weight
            return self.support.transform(broad_unit)

        chart_coordinate = (
            (unit[0] - broad_weight)
            / (1.0 - broad_weight)
            * self.centers.shape[0]
        )
        chart_index = min(int(chart_coordinate), self.centers.shape[0] - 1)
        local_unit = unit.copy()
        local_unit[0] = chart_coordinate - chart_index
        center = self.centers[chart_index]
        scale = self.scales[chart_index]
        standardized_lower = (self.support.lower - center) / scale
        standardized_upper = (self.support.upper - center) / scale
        probability_lower = ndtr(standardized_lower)
        probability_width = ndtr(standardized_upper) - probability_lower
        probabilities = probability_lower + local_unit * probability_width
        probabilities = np.clip(
            probabilities,
            np.finfo(float).eps,
            1.0 - np.finfo(float).eps,
        )
        return center + scale * ndtri(probabilities)


@dataclass(frozen=True)
class IntegerMarginalLogLikelihood:
    """Pickleable all-session target used by dynesty checkpoints."""

    evaluator: Any
    coordinates: Any
    phase_noise_sigma_cycles: float
    prior_precision: float

    def __call__(self, normalized_values: Sequence[float]) -> float:
        return integer_marginal_log_posterior(
            normalized_values,
            evaluator=self.evaluator,
            coordinates=self.coordinates,
            phase_noise_sigma_cycles=self.phase_noise_sigma_cycles,
            prior_precision=self.prior_precision,
        )


@dataclass(frozen=True)
class IntegerMarginalNestedResult:
    samples: np.ndarray
    log_likelihood: np.ndarray
    log_weights: np.ndarray
    posterior_weights: np.ndarray
    log_evidence: float
    log_evidence_error: float
    call_count: int
    iteration_count: int
    best_normalized_values: np.ndarray
    best_log_likelihood: float
    effective_sample_size: float
    reached_call_limit: bool
    live_samples: np.ndarray
    live_log_likelihood: np.ndarray
    invocation_call_count: int
    resumed_from_checkpoint: bool
    finalized_live_points: bool


def build_shape_island_support(
    normalized_samples: Sequence[Sequence[float]],
    *,
    global_lower: Sequence[float],
    global_upper: Sequence[float],
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
    expansion_factor: float = 4.0,
    minimum_global_width_fraction: float = 1.0e-4,
) -> ShapeIslandSupport:
    """Build a declared local support from wrap-invariant posterior samples."""

    samples = np.asarray(normalized_samples, dtype=float)
    lower = np.asarray(global_lower, dtype=float)
    upper = np.asarray(global_upper, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Shape-island support requires at least two samples.")
    if samples.shape[1:] != lower.shape or upper.shape != lower.shape:
        raise ValueError("Shape-island samples and global bounds must align.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Shape-island samples must be finite.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
        raise ValueError("Shape-island global bounds must be finite and ordered.")
    q_lower = float(lower_quantile)
    q_upper = float(upper_quantile)
    expansion = float(expansion_factor)
    if not 0.0 <= q_lower < q_upper <= 1.0:
        raise ValueError("Shape-island quantiles must be ordered in [0, 1].")
    if not np.isfinite(expansion) or expansion < 1.0:
        raise ValueError("Shape-island expansion must be at least one.")

    sample_lower = np.quantile(samples, q_lower, axis=0)
    sample_upper = np.quantile(samples, q_upper, axis=0)
    center = 0.5 * (sample_lower + sample_upper)
    global_width = upper - lower
    minimum_half_width = (
        max(float(minimum_global_width_fraction), 0.0) * global_width / 2.0
    )
    half_width = np.maximum(
        0.5 * expansion * (sample_upper - sample_lower),
        minimum_half_width,
    )
    support_lower = np.maximum(lower, center - half_width)
    support_upper = np.minimum(upper, center + half_width)
    if np.any(support_lower >= support_upper):
        raise ValueError("Shape-island support collapsed after applying bounds.")
    return ShapeIslandSupport(
        lower=support_lower,
        upper=support_upper,
        sample_lower_quantile=q_lower,
        sample_upper_quantile=q_upper,
        expansion_factor=expansion,
    )


def expand_support_with_phase_reachability(
    support: ShapeIslandSupport,
    normalized_samples: Sequence[Sequence[float]],
    *,
    global_lower: Sequence[float],
    global_upper: Sequence[float],
    lower_quantile: float = 0.0,
    upper_quantile: float = 1.0,
    expansion_factor: float = 1.1,
) -> ShapeIslandSupport:
    """Union shape support with the continuous phase-reachable manifold.

    Reachability samples may come from many integer paths. They are used only
    to declare continuous support; no path identity or path score enters the
    integer-marginalized target.
    """

    samples = np.asarray(normalized_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Phase reachability requires at least two samples.")
    if samples.shape[1:] != support.lower.shape:
        raise ValueError("Phase-reachable samples must match support dimension.")
    reachable = build_shape_island_support(
        samples,
        global_lower=global_lower,
        global_upper=global_upper,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        expansion_factor=expansion_factor,
    )
    return ShapeIslandSupport(
        lower=np.minimum(support.lower, reachable.lower),
        upper=np.maximum(support.upper, reachable.upper),
        sample_lower_quantile=support.sample_lower_quantile,
        sample_upper_quantile=support.sample_upper_quantile,
        expansion_factor=support.expansion_factor,
        support_strategy="shape_plus_phase_reachable_union",
        reachability_sample_count=int(samples.shape[0]),
    )


def build_phase_reachable_mixture_prior(
    support: ShapeIslandSupport,
    normalized_samples: Sequence[Sequence[float]],
    *,
    broad_component_weight: float = 0.1,
    neighbor_count: int = 8,
    local_scale_factor: float = 0.25,
    minimum_support_width_fraction: float = 1.0e-8,
    maximum_support_width_fraction: float = 0.05,
) -> PhaseReachableMixturePriorTransform:
    """Build a proper multimodal prior without selecting one reachable path."""

    centers = np.asarray(normalized_samples, dtype=float)
    if centers.ndim != 2 or centers.shape[0] < 2:
        raise ValueError("A reachable mixture prior requires at least two charts.")
    if centers.shape[1:] != support.lower.shape:
        raise ValueError("Reachable chart centers must match support dimension.")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Reachable chart centers must be finite.")
    if np.any((centers < support.lower) | (centers > support.upper)):
        raise ValueError("Reachable chart centers must lie inside support.")
    broad_weight = float(broad_component_weight)
    if not np.isfinite(broad_weight) or not 0.0 < broad_weight < 1.0:
        raise ValueError("Broad mixture weight must lie strictly between zero and one.")
    scale_factor = float(local_scale_factor)
    if not np.isfinite(scale_factor) or scale_factor <= 0.0:
        raise ValueError("Local mixture scale factor must be positive.")
    minimum_fraction = max(float(minimum_support_width_fraction), 0.0)
    maximum_fraction = float(maximum_support_width_fraction)
    if not np.isfinite(maximum_fraction) or maximum_fraction <= minimum_fraction:
        raise ValueError("Mixture scale fractions must be finite and ordered.")

    width = support.upper - support.lower
    standardized = centers / width
    local_count = min(max(int(neighbor_count), 1), centers.shape[0] - 1)
    scales = np.empty_like(centers)
    for index, center in enumerate(centers):
        distances = np.linalg.norm(standardized - standardized[index], axis=1)
        neighbors = np.argsort(distances)[1 : local_count + 1]
        local_rms = np.sqrt(
            np.mean(np.square(centers[neighbors] - center), axis=0)
        )
        scales[index] = np.clip(
            scale_factor * local_rms,
            minimum_fraction * width,
            maximum_fraction * width,
        )
    return PhaseReachableMixturePriorTransform(
        support=support,
        centers=centers.copy(),
        scales=scales,
        broad_component_weight=broad_weight,
    )


def integer_marginal_log_posterior(
    normalized_values: Sequence[float],
    *,
    evaluator: Any,
    coordinates: Any,
    phase_noise_sigma_cycles: float,
    prior_precision: float,
) -> float:
    """Evaluate the all-session posterior after summing every integer path."""

    values = np.asarray(normalized_values, dtype=float)
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if (
        values.shape != lower.shape
        or not np.all(np.isfinite(values))
        or np.any(values <= lower)
        or np.any(values >= upper)
    ):
        return -math.inf
    sigma = float(phase_noise_sigma_cycles)
    precision = max(float(prior_precision), 0.0)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("Phase-noise sigma must be finite and non-negative.")
    try:
        profile = evaluator.evaluate_values(_physical_values(coordinates, values))
        wrapped, _ = collapsed_wrapped_log_likelihood(
            profile,
            added_session_phase_variance=sigma * sigma,
        )
    except Exception:
        return -math.inf
    log_target = (
        -0.5 * float(profile.chi2)
        + float(wrapped)
        - 0.5 * precision * float(values @ values)
    )
    return float(log_target) if np.isfinite(log_target) else -math.inf


def run_integer_marginal_nested(
    *,
    evaluator: Any,
    coordinates: Any,
    support: ShapeIslandSupport,
    config: IntegerMarginalNestedConfig | None = None,
    checkpoint_path: Path | None = None,
    prior_transform: Any | None = None,
    resume: bool = False,
    print_progress: bool = True,
) -> IntegerMarginalNestedResult:
    """Explore the multimodal integer-marginalized continuous posterior."""

    active = config or IntegerMarginalNestedConfig()
    dimension = int(support.lower.size)
    if support.upper.shape != support.lower.shape:
        raise ValueError("Nested support bounds must align.")

    checkpoint = (
        Path(checkpoint_path).resolve() if checkpoint_path is not None else None
    )
    resumed = bool(resume)
    if resumed:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(
                "Nested resume requires an existing checkpoint file."
            )
        sampler = dynesty.NestedSampler.restore(str(checkpoint))
        if int(sampler.ndim) != dimension:
            raise ValueError("Nested checkpoint dimension does not match support.")
        if bool(sampler.added_live):
            raise ValueError(
                "Nested checkpoint was already finalized with live points and "
                "cannot accept more search budget."
            )
    else:
        first_update: dict[str, float | int] = {}
        if active.first_update_min_calls is not None:
            first_update["min_ncall"] = max(
                int(active.first_update_min_calls),
                1,
            )
        if active.first_update_min_efficiency is not None:
            first_update["min_eff"] = max(
                float(active.first_update_min_efficiency),
                0.0,
            )
        active_prior_transform = (
            prior_transform
            if prior_transform is not None
            else ShapeIslandPriorTransform(support=support)
        )
        sampler = dynesty.NestedSampler(
            IntegerMarginalLogLikelihood(
                evaluator=evaluator,
                coordinates=coordinates,
                phase_noise_sigma_cycles=float(
                    active.phase_noise_sigma_cycles
                ),
                prior_precision=float(active.prior_precision),
            ),
            active_prior_transform,
            dimension,
            nlive=max(int(active.live_point_count), 2 * dimension + 1),
            bound=str(active.bound),
            sample=str(active.sample),
            first_update=first_update or None,
            rstate=np.random.default_rng(max(int(active.random_seed), 0)),
        )
    maximum_calls = (
        max(int(active.maximum_calls), 1)
        if active.maximum_calls is not None
        else None
    )
    initial_call_count = int(sampler.ncall)
    sampler.run_nested(
        maxcall=maximum_calls,
        dlogz=max(float(active.evidence_tolerance), 0.0),
        add_live=bool(active.finalize_live_points),
        print_progress=bool(print_progress),
        checkpoint_file=str(checkpoint) if checkpoint is not None else None,
        resume=resumed,
    )
    nested = sampler.results
    samples = np.asarray(nested.samples, dtype=float)
    log_likelihood = np.asarray(nested.logl, dtype=float)
    log_weights = np.asarray(nested.logwt, dtype=float)
    log_evidence = float(np.asarray(nested.logz, dtype=float)[-1])
    weights = np.exp(log_weights - log_evidence)
    weights /= np.sum(weights)
    live_samples = (
        np.empty((0, dimension), dtype=float)
        if bool(active.finalize_live_points)
        else np.asarray(sampler.live_v, dtype=float).copy()
    )
    live_log_likelihood = (
        np.empty(0, dtype=float)
        if bool(active.finalize_live_points)
        else np.asarray(sampler.live_logl, dtype=float).copy()
    )
    if live_log_likelihood.size > 0 and (
        log_likelihood.size == 0
        or float(np.max(live_log_likelihood)) > float(np.max(log_likelihood))
    ):
        live_best_index = int(np.argmax(live_log_likelihood))
        best_normalized_values = live_samples[live_best_index].copy()
        best_log_likelihood = float(live_log_likelihood[live_best_index])
    else:
        best_index = int(np.argmax(log_likelihood))
        best_normalized_values = samples[best_index].copy()
        best_log_likelihood = float(log_likelihood[best_index])
    effective_sample_size = float(1.0 / np.sum(np.square(weights)))
    call_count = int(sampler.ncall)
    invocation_call_count = max(int(sampler.ncall) - initial_call_count, 0)
    return IntegerMarginalNestedResult(
        samples=samples,
        log_likelihood=log_likelihood,
        log_weights=log_weights,
        posterior_weights=weights,
        log_evidence=log_evidence,
        log_evidence_error=float(np.asarray(nested.logzerr, dtype=float)[-1]),
        call_count=call_count,
        iteration_count=int(samples.shape[0]),
        best_normalized_values=best_normalized_values,
        best_log_likelihood=best_log_likelihood,
        effective_sample_size=effective_sample_size,
        reached_call_limit=bool(
            maximum_calls is not None and invocation_call_count >= maximum_calls
        ),
        live_samples=live_samples,
        live_log_likelihood=live_log_likelihood,
        invocation_call_count=invocation_call_count,
        resumed_from_checkpoint=resumed,
        finalized_live_points=bool(active.finalize_live_points),
    )


__all__ = [
    "IntegerMarginalNestedConfig",
    "IntegerMarginalLogLikelihood",
    "IntegerMarginalNestedResult",
    "PhaseReachableMixturePriorTransform",
    "ShapeIslandSupport",
    "ShapeIslandPriorTransform",
    "build_shape_island_support",
    "build_phase_reachable_mixture_prior",
    "expand_support_with_phase_reachability",
    "integer_marginal_log_posterior",
    "run_integer_marginal_nested",
]
