from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dynesty
import numpy as np

from timing_accelerator.inference.bayesian_spin_integer_orbit import (
    OrbitBoxPriorTransform,
    SpinIntegerMarginalizedOrbitLikelihood,
)


@dataclass(frozen=True)
class SpinIntegerOrbitNestedConfig:
    live_point_count: int = 512
    evidence_tolerance: float = 0.25
    maximum_calls: int | None = None
    bound: str = "multi"
    sample: str = "rslice"
    random_seed: int = 20260725
    finalize_live_points: bool = False
    first_update_min_calls: int | None = None
    first_update_min_efficiency: float | None = None


@dataclass(frozen=True)
class SpinIntegerOrbitNestedResult:
    samples: np.ndarray
    log_likelihood: np.ndarray
    log_weights: np.ndarray
    posterior_weights: np.ndarray
    log_evidence: float
    log_evidence_error: float
    call_count: int
    iteration_count: int
    best_orbit_normalized_values: np.ndarray
    best_log_likelihood: float
    effective_sample_size: float
    reached_call_limit: bool
    live_samples: np.ndarray
    live_log_likelihood: np.ndarray
    invocation_call_count: int
    resumed_from_checkpoint: bool
    finalized_live_points: bool


def run_spin_integer_orbit_nested(
    *,
    likelihood: SpinIntegerMarginalizedOrbitLikelihood,
    config: SpinIntegerOrbitNestedConfig | None = None,
    checkpoint_path: Path | None = None,
    prior_transform: Any | None = None,
    resume: bool = False,
    print_progress: bool = True,
) -> SpinIntegerOrbitNestedResult:
    """Explore the orbit posterior with spin and integer paths integrated."""

    active = config or SpinIntegerOrbitNestedConfig()
    lower = np.asarray(likelihood.orbit_lower, dtype=float)
    upper = np.asarray(likelihood.orbit_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("Orbit support bounds must be matching vectors.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
        raise ValueError("Orbit support bounds must be finite and ordered.")
    dimension = int(lower.size)
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
            raise ValueError("Nested checkpoint dimension does not match orbit support.")
        if bool(sampler.added_live):
            raise ValueError(
                "Nested checkpoint was finalized with live points and cannot resume."
            )
    else:
        first_update: dict[str, float | int] = {}
        if active.first_update_min_calls is not None:
            first_update["min_ncall"] = max(int(active.first_update_min_calls), 1)
        if active.first_update_min_efficiency is not None:
            first_update["min_eff"] = max(
                float(active.first_update_min_efficiency),
                0.0,
            )
        sampler = dynesty.NestedSampler(
            likelihood,
            (
                prior_transform
                if prior_transform is not None
                else OrbitBoxPriorTransform(lower=lower, upper=upper)
            ),
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
    posterior_weights = np.exp(log_weights - log_evidence)
    posterior_weights /= np.sum(posterior_weights)
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
        best_index = int(np.argmax(live_log_likelihood))
        best_values = live_samples[best_index].copy()
        best_log_likelihood = float(live_log_likelihood[best_index])
    else:
        best_index = int(np.argmax(log_likelihood))
        best_values = samples[best_index].copy()
        best_log_likelihood = float(log_likelihood[best_index])
    call_count = int(sampler.ncall)
    invocation_call_count = max(call_count - initial_call_count, 0)
    return SpinIntegerOrbitNestedResult(
        samples=samples,
        log_likelihood=log_likelihood,
        log_weights=log_weights,
        posterior_weights=posterior_weights,
        log_evidence=log_evidence,
        log_evidence_error=float(np.asarray(nested.logzerr, dtype=float)[-1]),
        call_count=call_count,
        iteration_count=int(samples.shape[0]),
        best_orbit_normalized_values=best_values,
        best_log_likelihood=best_log_likelihood,
        effective_sample_size=float(
            1.0 / np.sum(np.square(posterior_weights))
        ),
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
    "SpinIntegerOrbitNestedConfig",
    "SpinIntegerOrbitNestedResult",
    "run_spin_integer_orbit_nested",
]
