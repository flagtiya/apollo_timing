"""Finite ELL1 period-cell and phase-trajectory coverage, not posterior weights."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import qmc

from apollo_timing.backend.pint_backend import (
    _load_pint_model,
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    SourceSession,
    profile_fixed_local_pulse_component,
    profile_two_level_pulse_lattice,
)
from apollo_timing.inference.bayesian_partial_ell1_bootstrap import (
    PartialELL1BootstrapMode,
    _phase_cycles,
    _phase_sigmas,
)
from apollo_timing.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.inference.session_shape_refinement import (
    relative_session_phases,
)
from apollo_timing.models.foldbank import (
    EvaluatedMode,
    FoldbankConfig,
    OrbitalPeriodAlias,
    SearchObservation,
    SessionPhaseScore,
)

_POSTERIOR_PARAMETERS = ("PB", "TASC", "A1", "EPS1", "EPS2")
_POSTERIOR_DIFFERENCE_STEPS = np.asarray((1e-8, 1e-7, 1e-5, 1e-6, 1e-6))


@dataclass(frozen=True)
class SessionShapeData:
    trajectory: np.ndarray
    normalized_residuals: np.ndarray
    local_integer_offsets: tuple[tuple[int, ...], ...]

    @property
    def chi2(self) -> float:
        return float(self.normalized_residuals @ self.normalized_residuals)


def session_shape_data(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
    local_integer_offsets: Sequence[Sequence[int]] | None = None,
) -> SessionShapeData:
    relative = relative_session_phases(_phase_cycles(model, toas), sessions)
    sigmas = _phase_sigmas(model, toas)
    if local_integer_offsets is None:
        profile = profile_two_level_pulse_lattice(relative, sigmas, sessions, integer_radius=0)
    else:
        profile = profile_fixed_local_pulse_component(
            relative, sigmas, sessions, local_integer_offsets
        )
    residuals = profile.normalized_local_residuals
    return SessionShapeData(residuals * sigmas, residuals, profile.local_integer_offsets)


def _evaluate_mode(
    *,
    mode: PartialELL1BootstrapMode,
    toas: object,
    matches: Sequence[tuple[SourceSession, SearchObservation]],
    config: FoldbankConfig,
) -> EvaluatedMode:
    model = _load_pint_model(mode.par_path)
    shape = session_shape_data(model=model, toas=toas, sessions=tuple(item[0] for item in matches))
    scores, reasons = [], []
    for session, observation in matches:
        indices = np.asarray(session.toa_indices, dtype=int)
        centered = shape.trajectory[indices]
        residuals = shape.normalized_residuals[indices]
        span = float(np.ptp(centered))
        loss = float(1 - abs(np.mean(np.exp(2j * math.pi * centered))))
        scores.append(
            SessionPhaseScore(
                observation.observation_id,
                len(indices),
                span,
                float(np.sqrt(np.mean(centered**2))),
                loss,
                float(residuals @ residuals),
            )
        )
        if span > config.maximum_phase_span_cycles:
            reasons.append(
                f"{observation.observation_id}:phase_span>{config.maximum_phase_span_cycles:g}"
            )
        if loss > config.maximum_coherent_loss:
            reasons.append(
                f"{observation.observation_id}:coherent_loss>{config.maximum_coherent_loss:g}"
            )
    scored_mode = replace(mode, all_session_shape_chi2=shape.chi2)
    return EvaluatedMode(scored_mode, tuple(scores), shape.trajectory, not reasons, tuple(reasons))


class LocalCoverageUnavailableError(RuntimeError):
    """No finite local box can be constructed; discrete seeds remain usable."""


def _enumerate_orbital_period_aliases(
    *,
    reference_pb_days: float,
    baseline_days: float,
    minimum_pb_days: float | None,
    maximum_pb_days: float | None,
    maximum_aliases: int,
) -> tuple[OrbitalPeriodAlias, ...]:
    """Tile a PB prior into cells no wider than one inter-epoch orbit cycle."""

    reference = float(reference_pb_days)
    baseline = float(baseline_days)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("The reference PB must be finite and positive.")
    if not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("The two observations need a positive epoch baseline.")
    if minimum_pb_days is None or maximum_pb_days is None:
        frequency = 1.0 / reference
        return (
            OrbitalPeriodAlias(
                alias_id="alias_0000",
                center_days=reference,
                minimum_days=reference,
                maximum_days=reference,
                minimum_frequency_per_day=frequency,
                maximum_frequency_per_day=frequency,
                cycles_across_baseline=baseline * frequency,
            ),
        )

    pb_min = float(minimum_pb_days)
    pb_max = float(maximum_pb_days)
    if not pb_min <= reference <= pb_max:
        raise ValueError("The parfile PB must lie inside --pb-min/--pb-max.")
    frequency_min = 1.0 / pb_max
    frequency_max = 1.0 / pb_min
    frequency_width = frequency_max - frequency_min
    alias_count = max(1, int(math.ceil(frequency_width * baseline)))
    if alias_count > int(maximum_aliases):
        raise ValueError(
            f"The PB prior spans {alias_count} orbital-cycle aliases, above "
            f"--max-pb-aliases={maximum_aliases}. Narrow the prior or raise the budget."
        )
    edges = np.linspace(frequency_min, frequency_max, alias_count + 1)
    aliases = []
    for index, (lower_frequency, upper_frequency) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        center_frequency = 0.5 * (float(lower_frequency) + float(upper_frequency))
        aliases.append(
            OrbitalPeriodAlias(
                alias_id=f"alias_{index:04d}",
                center_days=1.0 / center_frequency,
                minimum_days=1.0 / float(upper_frequency),
                maximum_days=1.0 / float(lower_frequency),
                minimum_frequency_per_day=float(lower_frequency),
                maximum_frequency_per_day=float(upper_frequency),
                cycles_across_baseline=baseline * center_frequency,
            )
        )
    return tuple(aliases)


def _select_posterior_phase_coverage(
    accepted: Sequence[tuple[float, np.ndarray, np.ndarray]],
    *,
    maximum_retained: int,
) -> tuple[tuple[float, np.ndarray, np.ndarray], ...]:
    """Retain the best draw, then greedily cover distinct phase trajectories."""

    if maximum_retained <= 0:
        raise ValueError("The posterior retention budget must be positive.")
    ranked = sorted(accepted, key=lambda item: item[0])
    if len(ranked) <= maximum_retained:
        return tuple(ranked)

    trajectories = np.stack([item[2] for item in ranked])
    selected_indices = [0]
    delta = trajectories - trajectories[0]
    delta -= np.rint(delta)
    minimum_distances = np.sqrt(np.mean(np.square(delta), axis=1))
    minimum_distances[0] = -math.inf
    while len(selected_indices) < maximum_retained:
        index = int(np.argmax(minimum_distances))
        selected_indices.append(index)
        delta = trajectories - trajectories[index]
        delta -= np.rint(delta)
        distances = np.sqrt(np.mean(np.square(delta), axis=1))
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected_indices] = -math.inf
    return tuple(ranked[index] for index in selected_indices)


def _posterior_sampling_box(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
    period_alias: OrbitalPeriodAlias,
    maximum_eccentricity: float,
    box_sigma: float,
    maximum_a1_seconds: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center = np.asarray(
        [float(model[name].value) for name in _POSTERIOR_PARAMETERS],
        dtype=float,
    )
    initial = session_shape_data(model=model, toas=toas, sessions=sessions)

    def residuals(values: np.ndarray) -> np.ndarray:
        for name, value in zip(_POSTERIOR_PARAMETERS, values, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        return session_shape_data(
            model=model,
            toas=toas,
            sessions=sessions,
            local_integer_offsets=initial.local_integer_offsets,
        ).normalized_residuals

    center_residuals = residuals(center)
    columns = []
    for index, step in enumerate(_POSTERIOR_DIFFERENCE_STEPS):
        upper = center.copy()
        lower = center.copy()
        upper[index] += step
        lower[index] -= step
        columns.append((residuals(upper) - residuals(lower)) / (2.0 * step))
    residuals(center)
    jacobian = np.column_stack(columns)
    active_columns = np.arange(len(_POSTERIOR_PARAMETERS))
    if period_alias.minimum_days == period_alias.maximum_days:
        active_columns = active_columns[1:]
    active_jacobian = jacobian[:, active_columns]
    covariance = np.linalg.pinv(active_jacobian.T @ active_jacobian, rcond=1.0e-12)
    standard_deviation = np.zeros(len(_POSTERIOR_PARAMETERS))
    standard_deviation[active_columns] = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    half_width = float(box_sigma) * standard_deviation
    lower = center - half_width
    upper = center + half_width
    lower[0] = max(lower[0], period_alias.minimum_days)
    upper[0] = min(upper[0], period_alias.maximum_days)
    lower[2] = max(lower[2], 1.0e-6)
    upper[2] = min(upper[2], maximum_a1_seconds)
    lower[3:] = np.maximum(lower[3:], -float(maximum_eccentricity))
    upper[3:] = np.minimum(upper[3:], float(maximum_eccentricity))
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise LocalCoverageUnavailableError("The local coverage box is not finite.")
    if np.any(upper < lower) or np.any(upper[1:] <= lower[1:]):
        raise LocalCoverageUnavailableError("The local coverage box has zero width.")
    return center, lower, upper, float(center_residuals @ center_residuals)


def _sample_one_continuous_seed(
    job: tuple[
        int,
        PartialELL1BootstrapMode,
        OrbitalPeriodAlias,
        object,
        tuple[SourceSession, ...],
        Path,
        FoldbankConfig,
        float,
        float,
    ],
) -> tuple[tuple[PartialELL1BootstrapMode, ...], dict[str, object]]:
    (
        seed_index,
        seed,
        period_alias,
        toas,
        sessions,
        destination,
        config,
        maximum_eccentricity,
        maximum_a1_seconds,
    ) = job
    model = _load_pint_model(seed.par_path)
    try:
        center, lower, upper, center_chi2 = _posterior_sampling_box(
            model=model,
            toas=toas,
            sessions=sessions,
            period_alias=period_alias,
            maximum_eccentricity=maximum_eccentricity,
            box_sigma=config.posterior_box_sigma,
            maximum_a1_seconds=maximum_a1_seconds,
        )
    except LocalCoverageUnavailableError as error:
        return (), {
            "seed_mode_id": seed.mode_id,
            "orbital_period_alias_id": period_alias.alias_id,
            "status": "skipped",
            "reason": str(error),
            "draws": config.posterior_draws,
            "accepted": 0,
            "retained": 0,
            "heldout_data_used": False,
        }
    initial = session_shape_data(model=model, toas=toas, sessions=sessions)
    draw_power = int(math.ceil(math.log2(max(config.posterior_draws, 1))))
    unit = qmc.Sobol(
        d=len(_POSTERIOR_PARAMETERS),
        scramble=True,
        seed=41004 + seed_index,
    ).random_base2(draw_power)
    # A singleton PB prior is a fixed coordinate, not an invalid sampling box.
    draws = lower + unit[: config.posterior_draws] * (upper - lower)
    accepted = []
    threshold = center_chi2 + config.maximum_candidate_delta_chi2
    for values in draws:
        for name, value in zip(_POSTERIOR_PARAMETERS, values, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        data = session_shape_data(
            model=model,
            toas=toas,
            sessions=sessions,
        )
        chi2 = data.chi2
        if data.local_integer_offsets == initial.local_integer_offsets and chi2 <= threshold:
            accepted.append(
                (
                    chi2,
                    np.asarray(values, dtype=float).copy(),
                    data.trajectory,
                )
            )
    retained = _select_posterior_phase_coverage(
        accepted,
        maximum_retained=config.posterior_retained_per_mode,
    )
    seed_dir = destination / f"seed_{seed_index:04d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for sample_index, (chi2, values, _) in enumerate(retained):
        for name, value in zip(_POSTERIOR_PARAMETERS, values, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        mode_id = f"{seed.mode_id}_posterior_{sample_index:04d}"
        path = seed_dir / f"sample_{sample_index:04d}.par"
        write_pint_model_parfile_roundtrip_safe(model=model, path=path)
        parameters = dict(seed.parameter_values)
        parameters.update(
            {name: float(value) for name, value in zip(_POSTERIOR_PARAMETERS, values, strict=True)}
        )
        generated.append(
            PartialELL1BootstrapMode(
                mode_id=mode_id,
                par_path=path,
                informative_session_chi2=seed.informative_session_chi2,
                all_session_shape_chi2=chi2,
                parameter_values=parameters,
            )
        )
    audit = {
        "status": "completed",
        "seed_mode_id": seed.mode_id,
        "orbital_period_alias_id": period_alias.alias_id,
        "draws": config.posterior_draws,
        "accepted": len(accepted),
        "retained": len(retained),
        "retention_policy": "best_then_greedy_phase_trajectory_coverage",
        "chi2_center": center_chi2,
        "chi2_threshold": threshold,
        "parameter_names": list(_POSTERIOR_PARAMETERS),
        "center": center.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "sobol_seed": 41004 + seed_index,
        "heldout_data_used": False,
    }
    return tuple(generated), audit


def _sample_continuous_foldbank_modes(
    *,
    seed_modes: Sequence[PartialELL1BootstrapMode],
    mode_alias_by_id: dict[str, OrbitalPeriodAlias],
    toas: object,
    sessions: Sequence[SourceSession],
    output_dir: Path,
    config: FoldbankConfig,
    maximum_eccentricity: float,
    maximum_a1_seconds: float,
) -> tuple[tuple[PartialELL1BootstrapMode, ...], tuple[dict[str, object], ...]]:
    if config.posterior_draws == 0:
        return (), ()
    selected_seeds = _select_posterior_seeds_by_alias(
        seed_modes,
        mode_alias_by_id=mode_alias_by_id,
        seeds_per_alias=config.posterior_seed_modes,
    )
    generated = []
    audits = []
    destination = Path(output_dir) / "bootstrap" / "posterior_samples"
    destination.mkdir(parents=True, exist_ok=True)
    session_tuple = tuple(sessions)
    jobs = tuple(
        (
            seed_index,
            seed,
            mode_alias_by_id[seed.mode_id],
            toas,
            session_tuple,
            destination,
            config,
            maximum_eccentricity,
            maximum_a1_seconds,
        )
        for seed_index, seed in enumerate(selected_seeds)
    )
    if config.posterior_workers == 1:
        results = tuple(map(_sample_one_continuous_seed, jobs))
    else:
        with ProcessPoolExecutor(max_workers=config.posterior_workers) as executor:
            results = tuple(executor.map(_sample_one_continuous_seed, jobs))
    for seed, (seed_modes, audit) in zip(selected_seeds, results, strict=True):
        period_alias = mode_alias_by_id[seed.mode_id]
        generated.extend(seed_modes)
        audits.append(audit)
        for mode in seed_modes:
            mode_alias_by_id[mode.mode_id] = period_alias
    return tuple(generated), tuple(audits)


def _select_posterior_seeds_by_alias(
    seed_modes: Sequence[PartialELL1BootstrapMode],
    *,
    mode_alias_by_id: dict[str, OrbitalPeriodAlias],
    seeds_per_alias: int,
) -> tuple[PartialELL1BootstrapMode, ...]:
    """Keep high-probability continuous seeds inside every solved PB alias."""

    grouped: dict[str, list[PartialELL1BootstrapMode]] = {}
    for mode in seed_modes:
        alias = mode_alias_by_id[mode.mode_id]
        grouped.setdefault(alias.alias_id, []).append(mode)
    selected = []
    for alias_id in sorted(grouped):
        ranked = sorted(
            grouped[alias_id],
            key=lambda item: (
                item.all_session_shape_chi2,
                item.informative_session_chi2,
                item.mode_id,
            ),
        )
        selected.extend(ranked[: int(seeds_per_alias)])
    return tuple(selected)


def _trajectory_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    delta -= np.rint(delta)
    return float(np.sqrt(np.mean(np.square(delta))))


def _select_coverage_bank(
    evaluated: Sequence[EvaluatedMode],
    *,
    config: FoldbankConfig,
) -> tuple[EvaluatedMode, ...]:
    eligible = sorted(
        (item for item in evaluated if item.eligible),
        key=lambda item: (
            item.mode.all_session_shape_chi2,
            item.mode.informative_session_chi2,
        ),
    )
    retained: list[EvaluatedMode] = []
    for item in eligible:
        if retained and any(
            _trajectory_distance(item.trajectory, other.trajectory)
            < config.minimum_trajectory_separation_cycles
            for other in retained
        ):
            continue
        retained.append(item)
        if len(retained) >= config.maximum_candidates:
            break
    return tuple(retained)


def _select_alias_stratified_coverage_bank(
    evaluated: Sequence[EvaluatedMode],
    *,
    mode_alias_by_id: dict[str, OrbitalPeriodAlias],
    config: FoldbankConfig,
) -> tuple[EvaluatedMode, ...]:
    """Round-robin diverse trajectories so no solved PB alias crowds out another."""

    grouped: dict[str, list[EvaluatedMode]] = {}
    for item in evaluated:
        alias = mode_alias_by_id[item.mode.mode_id]
        grouped.setdefault(alias.alias_id, []).append(item)
    pools = {
        alias_id: list(_select_coverage_bank(items, config=config))
        for alias_id, items in grouped.items()
    }
    pools = {alias_id: items for alias_id, items in pools.items() if items}
    if len(pools) > config.maximum_candidates:
        raise ValueError(
            f"The fold-bank has {len(pools)} eligible orbital aliases but only "
            f"--max-candidates={config.maximum_candidates}; completeness requires "
            "at least one candidate per alias."
        )
    selected = []
    alias_ids = sorted(pools)
    depth = 0
    while len(selected) < config.maximum_candidates:
        added = False
        for alias_id in alias_ids:
            pool = pools[alias_id]
            if depth >= len(pool):
                continue
            selected.append(pool[depth])
            added = True
            if len(selected) >= config.maximum_candidates:
                break
        if not added:
            break
        depth += 1
    return tuple(selected)


def _screen_relative_chi2(
    evaluated: Sequence[EvaluatedMode],
    *,
    maximum_delta_chi2: float,
) -> tuple[EvaluatedMode, ...]:
    if not evaluated:
        return ()
    finite = [
        item.mode.all_session_shape_chi2
        for item in evaluated
        if item.eligible and np.isfinite(item.mode.all_session_shape_chi2)
    ]
    best = min(finite, default=float("inf"))
    screened = []
    for item in evaluated:
        delta = item.mode.all_session_shape_chi2 - best
        if delta <= float(maximum_delta_chi2):
            screened.append(item)
            continue
        screened.append(
            replace(
                item,
                eligible=False,
                rejection_reasons=(
                    *item.rejection_reasons,
                    f"all_session_delta_chi2>{maximum_delta_chi2:g}",
                ),
            )
        )
    return tuple(screened)


def _screen_relative_chi2_by_alias(
    evaluated: Sequence[EvaluatedMode],
    *,
    mode_alias_by_id: dict[str, OrbitalPeriodAlias],
    maximum_delta_chi2: float,
) -> tuple[EvaluatedMode, ...]:
    """Apply the likelihood window locally within each discrete PB alias."""

    grouped: dict[str, list[EvaluatedMode]] = {}
    for item in evaluated:
        alias_id = mode_alias_by_id[item.mode.mode_id].alias_id
        grouped.setdefault(alias_id, []).append(item)
    screened_by_mode = {}
    for alias_id in sorted(grouped):
        for item in _screen_relative_chi2(
            grouped[alias_id],
            maximum_delta_chi2=maximum_delta_chi2,
        ):
            screened_by_mode[item.mode.mode_id] = item
    return tuple(screened_by_mode[item.mode.mode_id] for item in evaluated)
