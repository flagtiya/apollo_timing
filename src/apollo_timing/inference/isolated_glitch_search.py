"""Explicit, fixed-pulse-path profiling of one phase-continuous spin glitch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import StringIO
import math
from typing import Any, Callable, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.models.model_builder import ModelBuilder
from scipy.optimize import minimize_scalar

from apollo_timing.inference.isolated_spin_candidates import _freeze_to_contract


@dataclass(frozen=True)
class IsolatedGlitchSearchConfig:
    """Finite epoch search; amplitudes use native PINT Hz and Hz/s units."""

    enabled: bool = False
    epoch_bounds_mjd: tuple[float, float] | None = None
    fixed_epoch_mjd: float | None = None
    epoch_grid_size: int = 24
    maximum_frequency_order: int = 2
    maximum_epoch_refinements: int = 2
    epoch_tolerance_days: float = 1.0e-6
    frequency_half_width_hz: float = 5.0e-7
    frequency_derivative_half_width_hz_s: float = 5.0e-14
    smooth_spin_order: int = 2
    minimum_side_toas: int = 3

    def __post_init__(self) -> None:
        if self.fixed_epoch_mjd is not None and self.epoch_bounds_mjd is not None:
            raise ValueError("Choose a fixed glitch epoch or epoch bounds, not both.")
        if self.fixed_epoch_mjd is not None and not math.isfinite(self.fixed_epoch_mjd):
            raise ValueError("The fixed glitch epoch must be finite.")
        if self.epoch_bounds_mjd is not None:
            if (len(self.epoch_bounds_mjd) != 2
                    or not all(math.isfinite(x) for x in self.epoch_bounds_mjd)
                    or self.epoch_bounds_mjd[0] >= self.epoch_bounds_mjd[1]):
                raise ValueError("Glitch epoch bounds must be finite and ordered.")
        if self.maximum_frequency_order not in (1, 2):
            raise ValueError("Glitch frequency order must be 1 or 2.")
        if not 2 <= self.smooth_spin_order <= 5:
            raise ValueError("The smooth glitch control must allow F2 through at most F5.")
        for name in ("epoch_grid_size", "minimum_side_toas"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 3:
                raise ValueError(f"{name} must be an integer of at least three.")
        if (not isinstance(self.maximum_epoch_refinements, int)
                or self.maximum_epoch_refinements < 0):
            raise ValueError("The epoch refinement budget must be nonnegative.")
        for name in ("epoch_tolerance_days", "frequency_half_width_hz",
                     "frequency_derivative_half_width_hz_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")


@dataclass(frozen=True)
class FixedPathGlitchSearchResult:
    fitter: object
    residuals: object
    selected_candidate_id: str
    selected: bool
    portfolio: tuple[dict[str, Any], ...]
    complete: bool
    status: str


def _glitch_model(
    baseline_model: object, *, epoch: float | None = None,
    frequency_order: int = 0, smooth_order: int | None = None,
) -> object:
    """Add only the explicitly requested terms, retaining all nuisance parameters."""

    text = baseline_model.as_parfile()
    active = list(baseline_model.free_params)
    if smooth_order is not None:
        for order in range(2, smooth_order + 1):
            name = f"F{order}"
            if name not in baseline_model.params:
                text += f"\n{name} 0 1\n"
                active.append(name)
    if epoch is not None:
        text += (f"\nGLEP_1 {epoch:.15f}\nGLPH_1 0 0\nGLF0_1 0 1\n"
                 f"GLF1_1 0 {int(frequency_order == 2)}\n")
        active.append("GLF0_1")
        if frequency_order == 2:
            active.append("GLF1_1")
    model = ModelBuilder()(StringIO(text))
    _freeze_to_contract(model, active)
    return model


def search_glitch_on_fixed_path(
    *, baseline_fitter: object, toas: object, fit_parameters: Sequence[str],
    config: IsolatedGlitchSearchConfig, fit_record: Callable[..., tuple[dict, object]],
    fit_iterations: int, minimum_delta_bic: float, minimum_sigma: float,
    minimum_dof: int, strict_reduced_chi2_threshold: float,
) -> FixedPathGlitchSearchResult:
    """Fit null/smooth/glitch controls on every supplied integer pulse path.

    A failed null fit does not prevent profiling a glitch. This function neither
    discovers new pulse paths nor interprets a conditional improvement as a
    physical detection. GLPH, recovery, and additional events are not released.
    """

    model = baseline_fitter.model
    names = tuple(str(name).upper() for name in fit_parameters)
    if set(names) != set(model.free_params):
        raise ValueError("The glitch baseline must preserve the input fit-flag contract.")
    if "Glitch" in model.components:
        raise ValueError("New-glitch search requires a glitch-free baseline; fit an existing PAR glitch directly.")
    if "pulse_number" not in toas.table.colnames:
        raise ValueError("Glitch profiling requires an explicit retained pulse-number path.")
    if getattr(model, "has_correlated_errors", False):
        raise ValueError("Explicit glitch profiling currently requires uncorrelated TOA errors.")

    # PINT applies GLEP to the emission-time TDB coordinate, not UTC site MJDs.
    epochs = np.sort(np.asarray(toas.table["tdbld"], dtype=np.longdouble)
                     - model.delay(toas).to_value(u.day))
    if len(epochs) < 2 * config.minimum_side_toas:
        raise ValueError("Glitch fitting needs enough TOAs on both sides of the event.")
    searched_epoch = config.fixed_epoch_mjd is None
    bounds = config.epoch_bounds_mjd or (float(epochs[0]), float(epochs[-1]))
    if config.fixed_epoch_mjd is not None:
        trial_epochs = [config.fixed_epoch_mjd]
    else:
        lower, upper = max(bounds[0], float(epochs[0])), min(bounds[1], float(epochs[-1]))
        if lower >= upper:
            raise ValueError("Glitch epoch support does not overlap the observations.")
        bounds = (lower, upper)
        trial_epochs = list(np.linspace(lower, upper, config.epoch_grid_size))

    def covered(epoch: float) -> bool:
        before = int(np.searchsorted(epochs, epoch, side="right"))
        return min(before, len(epochs) - before) >= config.minimum_side_toas

    trial_epochs = [float(epoch) for epoch in trial_epochs if covered(epoch)]
    if not trial_epochs:
        raise ValueError("No trial glitch epoch has the required pre/post-event TOA coverage.")

    records: list[dict[str, Any]] = []
    fits: dict[str, tuple[object, object]] = {}
    cache: dict[tuple[int, float], dict[str, Any]] = {}

    def evaluate(candidate_id: str, kind: str, *, epoch: float | None = None,
                 order: int = 0, smooth_order: int | None = None,
                 stage: str = "grid", fitter: object | None = None) -> dict[str, Any]:
        metadata = dict(candidate_id=candidate_id, model_kind=kind,
                        metric_scope="fixed_retained_pulse_path_reported_errors",
                        glitch_frequency_order=order, changepoint_epoch_mjd=epoch,
                        epoch_is_searched=bool(epoch is not None and searched_epoch),
                        search_stage=stage, input_fit_parameters=names)
        try:
            if fitter is None:
                candidate = _glitch_model(model, epoch=epoch, frequency_order=order,
                                          smooth_order=smooth_order)
                fitter = WLSFitter(toas, candidate, track_mode="use_pulse_numbers")
                fitter.fit_toas(maxiter=fit_iterations)
            added = tuple(n for n in fitter.model.free_params if n not in names)
            record, residuals = fit_record(
                candidate_id=candidate_id, fitter=fitter, toas=toas,
                model_kind=kind, spin_order=smooth_order or 1,
                changepoint_epoch_mjd=epoch if searched_epoch else None,
                added_parameter_blocks={"EXTENSION": added} if added else {},
            )
            # The epoch costs a parameter even though each profile fit freezes it.
            epoch_count = int(epoch is not None and searched_epoch)
            count = len(fitter.model.free_params) + 1 + epoch_count
            dof = len(toas.table) - count
            sigma = fitter.model.scaled_toa_uncertainty(toas).to_value(u.s)
            record.update(metadata, status="evaluated", parameter_count_including_phase=count,
                          residual_degrees_of_freedom=dof,
                          reduced_chi2=float(record["chi2"]) / max(dof, 1),
                          bic=float(record["chi2"]) + float(np.sum(np.log(2 * np.pi * sigma**2)))
                          + count * math.log(len(toas.table)))
            limits = {"GLF0_1": config.frequency_half_width_hz,
                      "GLF1_1": config.frequency_derivative_half_width_hz_s}
            outside = tuple(n for n in added if n in limits
                            and abs(float(fitter.model[n].value)) >= limits[n])
            at_epoch_bound = bool(epoch is not None and searched_epoch and any(
                abs(epoch - edge) <= config.epoch_tolerance_days for edge in bounds))
            record.update(active_bounds=(*outside, *(("GLEP_1",) if at_epoch_bound else ())),
                          within_amplitude_support=not outside,
                          identifiable=bool(record["covariance_full_rank"] and dof >= minimum_dof),
                          fitted_parameter_errors={n: float(fitter.model[n].uncertainty_value)
                                                   for n in fitter.model.free_params})
            fits[candidate_id] = (fitter, residuals)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            record = dict(metadata, status="failed", error=f"{type(error).__name__}: {error}")
        records.append(record)
        return record

    baseline = evaluate("glitch_null", "input_no_glitch", stage="control", fitter=baseline_fitter)
    if "bic" not in baseline:
        raise RuntimeError("The retained glitch baseline could not be evaluated: " + baseline["error"])
    smooth_records = []
    for order in range(2, config.smooth_spin_order + 1):
        smooth_records.append(evaluate(f"glitch_smooth_f{order}", "smooth_spin_control",
                                       smooth_order=order, stage="control"))

    def profile(order: int, epoch: float, stage: str) -> dict[str, Any]:
        key = (order, round(float(epoch), 12))
        if key not in cache:
            cache[key] = evaluate(f"glitch_f{order}_{len(cache):04d}", "continuous_spin_glitch",
                                  epoch=float(epoch), order=order, stage=stage)
        return cache[key]

    refinement_failures = []
    for order in range(1, config.maximum_frequency_order + 1):
        grid = [profile(order, epoch, "grid") for epoch in trial_epochs]
        if not searched_epoch:
            continue
        seeds = sorted((r for r in grid if "bic" in r), key=lambda r: r["bic"])
        seen_intervals = set()
        for seed in seeds:
            if len(seen_intervals) >= config.maximum_epoch_refinements:
                break
            epoch = seed["changepoint_epoch_mjd"]
            index = int(np.searchsorted(epochs, epoch, side="right"))
            lo, hi = max(bounds[0], float(epochs[index - 1])), min(bounds[1], float(epochs[index]))
            interval = (lo, hi)
            if interval in seen_intervals or hi - lo <= 2 * config.epoch_tolerance_days:
                continue
            seen_intervals.add(interval)

            def objective(x: float) -> float:
                result = profile(order, lo + x * (hi - lo), "refinement")
                return float(result.get("chi2", math.inf))

            fit = minimize_scalar(objective, bounds=(0., 1.), method="bounded",
                                  options={"xatol": config.epoch_tolerance_days / (hi - lo),
                                           "maxiter": 64})
            if not fit.success:
                refinement_failures.append(dict(frequency_order=order, interval_mjd=interval,
                                                error=str(fit.message)))

    finite = [r for r in records if "bic" in r]
    valid = [r for r in finite if r["identifiable"] and not r["active_bounds"]]
    controls = [r for r in smooth_records if "bic" in r]
    best_smooth_bic = min((r["bic"] for r in controls), default=math.inf)
    complete = (all(r["status"] != "failed" and r.get("within_amplitude_support", False)
                    for r in records) and not refinement_failures)
    qualified = []
    for record in finite:
        delta = baseline["bic"] - record["bic"]
        sigma = record.get("minimum_added_parameter_significance")
        record.update(delta_bic_vs_baseline=delta, evidence_passed=False)
        if record["model_kind"] == "continuous_spin_glitch":
            record["delta_bic_vs_smooth"] = best_smooth_bic - record["bic"]
        if (record is baseline or record not in valid or delta < minimum_delta_bic
                or sigma is None or sigma < minimum_sigma):
            continue
        if (record["model_kind"] == "continuous_spin_glitch"
                and (not controls or record["delta_bic_vs_smooth"] < minimum_delta_bic)):
            continue
        # A useful diagnostic fit need not have an acceptable absolute chi2.
        record["evidence_passed"] = bool(complete and record["reduced_chi2"] < strict_reduced_chi2_threshold)
        qualified.append(record)
    selected = min(qualified, key=lambda r: r["bic"], default=baseline)
    best_glitches = []
    for order in range(1, config.maximum_frequency_order + 1):
        family = [r for r in finite if r["glitch_frequency_order"] == order]
        if family:
            best = min(family, key=lambda r: r["bic"])
            best["diagnostic_model_par_text"] = fits[best["candidate_id"]][0].model.as_parfile()
            best_glitches.append(best["candidate_id"])
    summary = dict(config=asdict(config), epoch_coordinate="PINT emission-time TDB MJD",
                   effective_epoch_bounds_mjd=bounds if searched_epoch else None,
                   retained_grid_epoch_count=len(trial_epochs), evaluated_trial_count=len(records),
                   best_glitch_candidate_ids=best_glitches, refinement_failures=refinement_failures,
                   comparison_complete=complete, phase_jump_released=False,
                   recovery_released=False, new_pulse_paths_searched=False,
                   physical_glitch_detection_claimed=False)
    baseline["glitch_search"] = summary
    for record in records:
        record["selected"] = record["candidate_id"] == selected["candidate_id"]
    selected_fitter, selected_residuals = fits[selected["candidate_id"]]
    return FixedPathGlitchSearchResult(
        fitter=selected_fitter, residuals=selected_residuals,
        selected_candidate_id=selected["candidate_id"], selected=selected is not baseline,
        portfolio=tuple(records), complete=complete,
        status="explicit_glitch_search_complete" if complete else "explicit_glitch_search_incomplete",
    )
