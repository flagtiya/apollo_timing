"""Finite, explicitly authorized parameter hypotheses on one certified path."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from io import StringIO
import math
import re
from typing import Any, Mapping

import numpy as np
from pint.models import get_model
from apollo_timing.inference.timing_constraints import SkyPositionConstraint, resolve_parameter_bounds

from apollo_timing.inference.pint_solution_artifacts import (
    fit_pint_model_with_physical_bounds,
)
from apollo_timing.inference.toa_white_noise_certification import (
    evaluate_fixed_path_normalized_likelihood,
)


@dataclass(frozen=True)
class ParameterExplorationConfig:
    parameters: tuple[str, ...] = ()
    bounds: tuple[tuple[str, float, float], ...] = ()
    maximum_hypotheses: int = 16
    maximum_evaluations: int = 500
    minimum_delta_bic: float = 10.0
    minimum_sigma: float = 5.0
    search_bounds: tuple[tuple[str, float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.maximum_hypotheses < 1 or self.maximum_evaluations < 1:
            raise ValueError("Parameter exploration budgets must be positive.")
        if not all(
            math.isfinite(x) and x > 0
            for x in (self.minimum_delta_bic, self.minimum_sigma)
        ):
            raise ValueError(
                "Parameter exploration thresholds must be positive and finite."
            )
        if self.parameters and self.bounds:
            raise ValueError("Hard parameter bounds cannot accompany parameter exploration; use search_bounds / --exploration-bound for finite candidate search domains.")
        if self.search_bounds and not self.parameters:
            raise ValueError("Exploration search bounds require explicit parameter exploration.")
        if any(name.upper() not in {p.upper() for p in self.parameters}
               for name, _, _ in self.search_bounds):
            raise ValueError("Exploration bounds must name explicitly authorized parameters.")
        names = [name.upper() for name, _, _ in (*self.bounds, *self.search_bounds)]
        if len(set(names)) != len(names):
            raise ValueError("Repeated physical parameter bounds are ambiguous.")
        for name, lower, upper in (*self.bounds, *self.search_bounds):
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or not (
                math.isfinite(lower) and math.isfinite(upper) and lower < upper
            ):
                raise ValueError("Parameter bounds require NAME, finite LOWER < UPPER.")
        if any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) for name in self.parameters
        ):
            raise ValueError("Invalid parameter name in exploration authorization.")


def validate_parameter_bounds(model: Any, config: ParameterExplorationConfig) -> None:
    """Reject ignored or conflicting bounds before an expensive search starts."""
    resolve_parameter_bounds(model, config.bounds, optional_parameters=config.parameters)
    resolve_parameter_bounds(model, config.search_bounds, optional_parameters=config.parameters, require_input_within=False)
    if any(name.upper() in model.free_params for name, _, _ in config.search_bounds):
        raise ValueError("Exploration bounds cannot redefine an already-free baseline parameter; use fixed-model hard limits in a separate run.")


def validate_constraint_scope(config: Any, *, model_selection: bool = False) -> None:
    """Keep user hard limits separate from changes of model or parameter set."""
    hard_limits = bool(config.custom_bounds or config.parameter_exploration.bounds
                       or config.sky_radius_arcmin is not None
                       or config.sky_position_constraint is not None)
    if hard_limits and (config.parameter_exploration.parameters or model_selection):
        raise ValueError("Hard parameter/sky constraints require a fixed physical model and parameter set; run exploration separately with --exploration-bound, then constrain the selected model in a new run.")


def explore_parameters_on_fixed_path(
    *,
    model: Any,
    toas: Any,
    config: ParameterExplorationConfig,
    input_bounds: Mapping[str, tuple[float, float]],
    sky_constraint: SkyPositionConstraint | None = None,
) -> tuple[Any, Any, dict[str, Any], np.ndarray | None]:
    """Compare singles and ordered cumulative blocks, not an unbounded powerset.

    Existing fixed values remain the null hypothesis. Only explicitly named
    parameters can be released; non-spin additions need explicit physical bounds.
    """
    from pint.residuals import Residuals

    if sky_constraint is not None:
        raise ValueError("Parameter exploration cannot accompany a hard sky constraint.")
    validate_parameter_bounds(model, config)

    baseline = evaluate_fixed_path_normalized_likelihood(model=model, toas=toas)
    residuals = Residuals(
        toas,
        model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    already_free = tuple(
        name.upper() for name in config.parameters if name.upper() in model.free_params
    )
    requested = tuple(
        dict.fromkeys(
            name.upper()
            for name in config.parameters
            if name.upper() not in model.free_params
        )
    )
    groups = list(
        dict.fromkeys(
            [(name,) for name in requested]
            + [requested[:n] for n in range(2, len(requested) + 1)]
        )
    )
    records: list[dict[str, Any]] = []
    best_model, best_residuals, best_cov = model, residuals, None
    best_bic = baseline.bic
    winner = "baseline"
    bounds = {name: (lo, hi) for name, lo, hi in resolve_parameter_bounds(
        model, config.search_bounds, optional_parameters=config.parameters, require_input_within=False,
    )}
    span = max(
        float(np.ptp(np.asarray(toas.get_mjds().value, dtype=float))) * 86400.0, 1.0
    )
    for index, group in enumerate(groups[: config.maximum_hypotheses]):
        candidate_id = f"parameter_{index:03d}"
        record: dict[str, Any] = dict(
            candidate_id=candidate_id, parameters=group, status="failed", selected=False
        )
        try:
            free = tuple(model.free_params)
            text = model.as_parfile()
            initial_null: dict[str, float] = {}
            for name in group:
                if name in free:
                    raise ValueError(
                        f"{name} is already free; no additional authorization is needed."
                    )
                original = model[name].value if name in model.params else None
                initial_null[name] = float(original) if original is not None else 0.0
                if name in model.params:
                    text = (
                        "\n".join(
                            line
                            for line in text.splitlines()
                            if not line.split() or line.split()[0].upper() != name
                        )
                        + "\n"
                    )
                text += f"{name} {initial_null[name]:.18g} 1\n"
            candidate = get_model(StringIO(text), allow_tcb=True)
            # Parsing PINT's legacy unit-scaled derivatives can rescale values.
            # The live baseline, not its text representation, owns the null.
            for name in model.params:
                if (
                    name in candidate.params
                    and model[name].value is not None
                    and type(model[name]).__name__ != "funcParameter"
                    and getattr(model[name], "unit_scale", False)
                ):
                    candidate[name].quantity = deepcopy(model[name].quantity)
                    if getattr(model[name], "uncertainty", None) is not None:
                        candidate[name].uncertainty = deepcopy(model[name].uncertainty)
            names = (*free, *group)
            candidate.free_params = list(names)
            if set(candidate.free_params) != set(names):
                raise ValueError(
                    "The requested physical parameter is not independently fit-able in this model."
                )
            lower, upper = [], []
            for name in names:
                center = float(candidate[name].value)
                interval = bounds.get(name, input_bounds.get(name))
                if (
                    interval is None
                    and re.fullmatch(r"F\d+", name)
                    and 2 <= int(name[1:]) <= 12
                ):
                    order = int(name[1:])
                    width = 20.0 * math.factorial(order + 1) / span ** (order + 1)
                    interval = (center - width, center + width)
                if interval is None:
                    raise ValueError(
                        f"{name} requires an explicit finite --exploration-bound."
                    )
                if name in group and not interval[0] <= center <= interval[1]:
                    # Candidate domains need not contain the fixed null value.
                    candidate[name].value = interval[0] + 0.5 * (interval[1] - interval[0])
                    center = float(candidate[name].value)
                if not interval[0] <= center <= interval[1]:
                    raise ValueError(
                        f"The initial value of {name} is outside its declared bounds."
                    )
                lower.append(interval[0])
                upper.append(interval[1])
            if len(toas.table) - len(names) < 5:
                raise ValueError("Fewer than five residual degrees of freedom remain.")
            fit = fit_pint_model_with_physical_bounds(
                model=candidate,
                toas=toas,
                parameter_names=names,
                lower_bounds=lower,
                upper_bounds=upper,
                maximum_evaluations=config.maximum_evaluations,
                sky_constraint=sky_constraint,
            )
            metrics = evaluate_fixed_path_normalized_likelihood(
                model=fit.model, toas=toas
            )
            indices = [names.index(name) for name in group]
            covariance = fit.covariance[np.ix_(indices, indices)]
            scales = np.asarray([upper[i] - lower[i] for i in indices])
            scaled = covariance / np.outer(scales, scales)
            eig = np.linalg.eigvalsh(scaled)
            identifiable = bool(
                np.all(np.isfinite(eig))
                and np.min(eig) > 0
                and np.max(eig) / np.min(eig) < 1e12
            )
            delta = np.asarray(
                [float(fit.model[name].value) - initial_null[name] for name in group]
            )
            sigma = (
                float(np.sqrt(max(delta @ np.linalg.solve(covariance, delta), 0)))
                if identifiable
                else 0.0
            )
            eligible = (
                identifiable
                and not fit.active_bounds
                and sigma >= config.minimum_sigma
                and baseline.bic - metrics.bic >= config.minimum_delta_bic
            )
            record.update(
                status="evaluated",
                **metrics.to_payload(),
                delta_bic=baseline.bic - metrics.bic,
                block_sigma=sigma,
                identifiable=identifiable,
                active_bounds=fit.active_bounds,
                sky_position_constraint=sky_constraint.report(fit.model) if sky_constraint else None,
                eligible=eligible,
                physical_bounds={n: (lo, hi) for n, lo, hi in zip(names, lower, upper)},
                fitted_parameters={n: float(fit.model[n].value) for n in names},
            )
            if eligible and metrics.bic < best_bic:
                best_model, best_residuals, best_cov = (
                    fit.model,
                    fit.residuals,
                    fit.covariance,
                )
                best_bic, winner = metrics.bic, candidate_id
        except (
            ArithmeticError,
            RuntimeError,
            ValueError,
            KeyError,
            AttributeError,
            np.linalg.LinAlgError,
        ) as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
    for record in records:
        record["selected"] = record["candidate_id"] == winner
    report = dict(
        status=(
            "disabled"
            if not requested
            else (
                "incomplete"
                if len(groups) > config.maximum_hypotheses
                or any(r["status"] == "failed" for r in records)
                else "evaluated"
            )
        ),
        graph="singles_and_ordered_prefixes",
        scope="conditional_on_this_pulse_path",
        search_bounds=config.search_bounds,
        interpretation="Model preference within finite search domains is not a physical measurement or a uniqueness proof.",
        requested_parameters=requested,
        already_free_parameters=already_free,
        total_hypotheses=len(groups),
        evaluated_hypotheses=len(records),
        selected_candidate_id=winner,
        baseline_bic=baseline.bic,
        selected_bic=best_bic,
        portfolio=records,
        uniqueness_claimed=False,
    )
    return best_model, best_residuals, report, best_cov
