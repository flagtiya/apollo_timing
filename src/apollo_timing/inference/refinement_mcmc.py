from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable

import numpy as np

from apollo_timing.models.contracts import NestedEngineConfig, NestedState


@dataclass
class WhiteningTransform:
    center: np.ndarray
    factor: np.ndarray
    inverse_factor: np.ndarray


def build_whitening_transform(center: np.ndarray, covariance: np.ndarray) -> WhiteningTransform:
    center = np.asarray(center, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    if covariance.shape[0] != center.shape[0]:
        raise ValueError("center and covariance dimensions do not match.")

    symmetrized = 0.5 * (covariance + covariance.T)
    jitter = 1.0e-18
    factor = None
    for _ in range(12):
        try:
            factor = np.linalg.cholesky(symmetrized + np.eye(symmetrized.shape[0]) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    if factor is None:
        raise ValueError("Unable to build a positive-definite whitening transform.")
    inverse_factor = np.linalg.inv(factor)
    return WhiteningTransform(center=center, factor=factor, inverse_factor=inverse_factor)


def whiten_parameters(parameters: np.ndarray, transform: WhiteningTransform) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    return transform.inverse_factor @ (parameters - transform.center)


def unwhiten_parameters(parameters: np.ndarray, transform: WhiteningTransform) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    return transform.center + transform.factor @ parameters


def initialize_walker_positions(
    center: np.ndarray,
    covariance: np.ndarray,
    walker_count: int,
    seed: int,
    whitened: bool,
) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    rng = np.random.default_rng(seed)
    ndim = center.shape[0]
    walker_count = max(int(walker_count), max(4, 2 * ndim + 2))
    if whitened:
        return rng.normal(loc=0.0, scale=1.0e-3, size=(walker_count, ndim))
    transform = build_whitening_transform(center, covariance)
    whitened_positions = rng.normal(loc=0.0, scale=1.0e-3, size=(walker_count, ndim))
    return np.array([unwhiten_parameters(item, transform) for item in whitened_positions], dtype=float)


def _build_emcee_sampler(walker_count: int, ndim: int, log_prob_fn: Callable):
    emcee = importlib.import_module("emcee")
    return emcee.EnsembleSampler(walker_count, ndim, log_prob_fn)


def _build_state_from_vector(best_state: NestedState, active_names: list[str], values: np.ndarray) -> NestedState:
    theta = dict(best_state.theta)
    active_theta = dict(best_state.active_theta or {})
    for name, value in zip(active_names, values, strict=True):
        theta[name] = float(value)
        active_theta[name] = float(value)
    return NestedState(
        structure_id=best_state.structure_id,
        wraps=list(best_state.wraps),
        theta=theta,
        active_theta=active_theta,
        inactive_theta=dict(best_state.inactive_theta or {}),
    )


def _fallback_covariance(active_names: list[str], prior_map: dict[str, object]) -> np.ndarray:
    widths = []
    for name in active_names:
        prior = prior_map[name]
        width = max(float(prior.upper) - float(prior.lower), 1.0e-18)
        widths.append((0.05 * width) ** 2)
    return np.diag(widths)


def _covariance_from_posterior_states(
    posterior_states: list[NestedState],
    best_state: NestedState,
    active_names: list[str],
    prior_map: dict[str, object],
) -> tuple[np.ndarray, str]:
    vectors: list[list[float]] = []
    for state in posterior_states:
        if state.structure_id != best_state.structure_id:
            continue
        active_theta = state.active_theta or {}
        if not all(name in active_theta for name in active_names):
            continue
        vectors.append([float(active_theta[name]) for name in active_names])
    if len(vectors) < 2:
        return _fallback_covariance(active_names, prior_map), "prior-width-fallback"
    matrix = np.asarray(vectors, dtype=float)
    if matrix.shape[0] <= 1:
        return _fallback_covariance(active_names, prior_map), "prior-width-fallback"
    covariance = np.cov(matrix, rowvar=False)
    covariance = np.atleast_2d(covariance)
    if covariance.shape != (len(active_names), len(active_names)) or not np.all(np.isfinite(covariance)):
        return _fallback_covariance(active_names, prior_map), "prior-width-fallback"
    covariance = covariance + np.eye(len(active_names)) * 1.0e-18
    return covariance, "posterior-states"


def run_conditional_mcmc_refinement(
    likelihood,
    best_state: NestedState,
    parameter_priors: list,
    config: NestedEngineConfig,
    posterior_states: list[NestedState],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": "skipped",
        "refinement_mode": config.refinement_mode,
        "reason": "Refinement disabled.",
    }
    if config.refinement_mode == "none":
        return {
            "summary": summary,
            "refined_state": best_state,
            "chain_payload": None,
        }
    if config.refinement_mode != "emcee":
        raise ValueError(f"Unsupported refinement_mode: {config.refinement_mode}")
    if config.refinement_target != "top1":
        raise ValueError(f"Unsupported refinement_target: {config.refinement_target}")

    active_theta = dict(best_state.active_theta or {})
    active_names = list(active_theta)
    if not active_names:
        return {
            "summary": {
                "status": "skipped",
                "refinement_mode": config.refinement_mode,
                "reason": "Best state has no active continuous parameters.",
            },
            "refined_state": best_state,
            "chain_payload": None,
        }

    prior_map = {item.name: item for item in parameter_priors if item.name in active_names}
    center = np.array([float(active_theta[name]) for name in active_names], dtype=float)
    covariance, covariance_source = _covariance_from_posterior_states(
        posterior_states=posterior_states,
        best_state=best_state,
        active_names=active_names,
        prior_map=prior_map,
    )
    transform = build_whitening_transform(center, covariance)
    walker_count = max(int(config.refinement_walkers), max(4, 2 * len(active_names) + 2))

    def log_prob_fn(sample: np.ndarray) -> float:
        values = (
            unwhiten_parameters(sample, transform)
            if config.refinement_whitened
            else np.asarray(sample, dtype=float)
        )
        for name, value in zip(active_names, values, strict=True):
            prior = prior_map[name]
            if float(value) < float(prior.lower) or float(value) > float(prior.upper):
                return float("-inf")
        refined_state = _build_state_from_vector(best_state, active_names, values)
        return float(likelihood.evaluate_state(refined_state).log_likelihood)

    initial_positions = initialize_walker_positions(
        center=center,
        covariance=covariance,
        walker_count=walker_count,
        seed=int(config.refinement_seed),
        whitened=bool(config.refinement_whitened),
    )
    sampler = _build_emcee_sampler(walker_count, len(active_names), log_prob_fn)
    sampler.run_mcmc(initial_positions, int(config.refinement_steps), progress=False)

    chain = np.asarray(sampler.get_chain(discard=int(config.refinement_burnin), flat=False), dtype=float)
    flat_chain = np.asarray(sampler.get_chain(discard=int(config.refinement_burnin), flat=True), dtype=float)
    log_prob = np.asarray(sampler.get_log_prob(discard=int(config.refinement_burnin), flat=False), dtype=float)
    flat_log_prob = np.asarray(sampler.get_log_prob(discard=int(config.refinement_burnin), flat=True), dtype=float)
    if flat_chain.size == 0 or flat_log_prob.size == 0:
        return {
            "summary": {
                "status": "failed",
                "refinement_mode": config.refinement_mode,
                "reason": "No post-burnin refinement samples were produced.",
            },
            "refined_state": best_state,
            "chain_payload": None,
        }

    if config.refinement_whitened:
        physical_chain = np.array(
            [[unwhiten_parameters(sample, transform) for sample in step] for step in chain],
            dtype=float,
        )
        flat_samples = np.array([unwhiten_parameters(sample, transform) for sample in flat_chain], dtype=float)
    else:
        physical_chain = chain
        flat_samples = flat_chain

    best_index = int(np.argmax(flat_log_prob))
    refined_values = flat_samples[best_index]
    refined_state = _build_state_from_vector(best_state, active_names, refined_values)
    refined_evaluation = likelihood.evaluate_state(refined_state)
    summary = {
        "status": "success",
        "refinement_mode": config.refinement_mode,
        "parameter_names": active_names,
        "walkers": walker_count,
        "steps": int(config.refinement_steps),
        "burnin": int(config.refinement_burnin),
        "whitened": bool(config.refinement_whitened),
        "covariance_source": covariance_source,
        "sample_count": int(flat_samples.shape[0]),
        "mean_acceptance_fraction": float(np.mean(np.asarray(sampler.acceptance_fraction, dtype=float))),
        "best_log_likelihood": float(refined_evaluation.log_likelihood),
        "best_reduced_chi2": refined_evaluation.reduced_chi2,
        "best_weighted_rms_us": refined_evaluation.weighted_rms_us,
    }
    chain_payload = {
        "chain": physical_chain,
        "flat_samples": flat_samples,
        "log_prob": log_prob,
        "flat_log_prob": flat_log_prob,
        "parameter_names": np.array(active_names, dtype=object),
    }
    return {
        "summary": summary,
        "refined_state": refined_state,
        "chain_payload": chain_payload,
    }
