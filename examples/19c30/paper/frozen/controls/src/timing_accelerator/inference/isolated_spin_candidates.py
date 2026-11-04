from __future__ import annotations

from io import StringIO
import math
import re
from typing import Any, Callable, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.models.model_builder import ModelBuilder, convert_tcb_tdb
from pint.residuals import Residuals

from timing_accelerator.backend.pint_backend import _set_numeric_model_parameter_value


def _freeze_to_contract(model: object, fit_parameters: Sequence[str]) -> None:
    active = {str(name).upper() for name in fit_parameters}
    for name in model.params:
        parameter = model[name]
        if hasattr(parameter, "frozen"):
            parameter.frozen = str(name).upper() not in active


def _candidate_model(
    *,
    source_par_text: str,
    shared_model: object,
    shared_fit_parameters: Sequence[str],
    spin_order: int,
    changepoint_epoch_mjd: float | None,
    glitch_frequency_order: int = 2,
) -> object:
    if glitch_frequency_order not in (1, 2):
        raise ValueError("Glitch frequency order must be 1 or 2.")
    if spin_order == 1 and changepoint_epoch_mjd is not None and glitch_frequency_order == 1:
        # Existing high derivatives and glitches are not part of the separate
        # hypothesis consisting of a single continuous frequency step.
        source_par_text = "\n".join(
            line for line in source_par_text.splitlines()
            if not (
                (fields := line.strip().split())
                and (
                    re.fullmatch(r"(?:GLEP|GLPH|GLF\d+(?:D\d*)?|GLTD\d*)_\d+", fields[0].upper())
                    or (
                        fields[0].upper().startswith("F")
                        and fields[0][1:].isdigit()
                        and int(fields[0][1:]) >= 2
                    )
                )
            )
        )
    source_parameter_names = {
        fields[0].upper()
        for raw_line in source_par_text.splitlines()
        if len(fields := raw_line.strip().split()) >= 2
        and not raw_line.lstrip().startswith(("#", "C "))
    }
    additions = [
        f"F{order} 0 1"
        for order in range(2, int(spin_order) + 1)
        if f"F{order}" not in source_parameter_names
    ]
    if changepoint_epoch_mjd is not None:
        glitch_additions = (
            ("GLEP_1", f"GLEP_1 {float(changepoint_epoch_mjd):.12f}"),
            ("GLPH_1", "GLPH_1 0"),
            ("GLF0_1", "GLF0_1 0 1"),
            ("GLF1_1", f"GLF1_1 0 {int(glitch_frequency_order == 2)}"),
        )
        additions.extend(
            line for name, line in glitch_additions if name not in source_parameter_names
        )
    model = ModelBuilder()(
        StringIO(source_par_text + "\n" + "\n".join(additions) + "\n"),
        allow_tcb=True,
    )
    if "UNITS" in model.params and model["UNITS"].value == "TCB":
        convert_tcb_tdb(model)
    for name in shared_fit_parameters:
        _set_numeric_model_parameter_value(model, name, float(shared_model[name].value))
    active_parameters = [str(name).upper() for name in shared_fit_parameters]
    for order in range(2, int(spin_order) + 1):
        model[f"F{order}"].frozen = False
        active_parameters.append(f"F{order}")
    if changepoint_epoch_mjd is not None:
        _set_numeric_model_parameter_value(model, "GLEP_1", changepoint_epoch_mjd)
        model["GLEP_1"].frozen = True
        model["GLF0_1"].frozen = False
        if glitch_frequency_order == 1:
            model["GLPH_1"].value = 0.0
            model["GLF1_1"].value = 0.0
            active_parameters.append("GLF0_1")
        else:
            active_parameters.extend(("GLF0_1", "GLF1_1"))
    _freeze_to_contract(model, active_parameters)
    if changepoint_epoch_mjd is not None:
        model["GLEP_1"].frozen = True
    return model


def _fit_candidate(
    *,
    source_par_text: str,
    shared_model: object,
    shared_fit_parameters: Sequence[str],
    toas: object,
    spin_order: int,
    changepoint_epoch_mjd: float | None,
    fit_iterations: int,
    glitch_frequency_order: int = 2,
) -> tuple[object, object]:
    model = _candidate_model(
        source_par_text=source_par_text,
        shared_model=shared_model,
        shared_fit_parameters=shared_fit_parameters,
        spin_order=spin_order,
        changepoint_epoch_mjd=changepoint_epoch_mjd,
        glitch_frequency_order=glitch_frequency_order,
    )
    fitter = WLSFitter(toas, model, track_mode="use_pulse_numbers")
    fitter.fit_toas(maxiter=int(fit_iterations))
    residuals = Residuals(
        toas,
        fitter.model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    return fitter, residuals


def _session_changepoint_intervals(
    *,
    toas: object,
    residuals: object,
    maximum_intervals: int,
    session_gap_days: float = 0.5,
) -> tuple[tuple[float, float, float], ...]:
    epochs = np.asarray(toas.get_mjds().value, dtype=float)
    values = np.asarray(residuals.time_resids.to_value(u.s), dtype=float)
    errors = np.asarray(toas.get_errors().to_value(u.s), dtype=float)
    starts = [0]
    starts.extend(int(index) for index in np.flatnonzero(np.diff(epochs) > session_gap_days) + 1)
    stops = [*starts[1:], len(epochs)]
    session_epochs = []
    session_values = []
    session_variances = []
    for start, stop in zip(starts, stops, strict=True):
        weights = 1.0 / np.square(errors[start:stop])
        session_epochs.append(float(np.mean(epochs[start:stop])))
        session_values.append(float(np.sum(weights * values[start:stop]) / np.sum(weights)))
        session_variances.append(float(1.0 / np.sum(weights)))
    ranked = []
    for index in range(len(session_epochs) - 1):
        lower = session_epochs[index]
        upper = session_epochs[index + 1]
        difference = session_values[index + 1] - session_values[index]
        variance = session_variances[index] + session_variances[index + 1]
        score = abs(difference) / math.sqrt(max(variance, np.finfo(float).tiny))
        ranked.append((float(score), float(lower), float(upper)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return tuple(ranked[: max(int(maximum_intervals), 1)])


def fit_frequency_step_family(
    *, source_par_text: str, shared_model: object,
    shared_fit_parameters: Sequence[str], toas: object,
    intervals: Sequence[tuple[float, float, float]],
    epoch_fractions: Sequence[float], fit_iterations: int,
    fit_candidate: Callable[..., tuple[object, object]],
    fit_record: Callable[..., tuple[dict[str, Any], object]],
) -> tuple[list[dict[str, Any]], dict[str, tuple[object, object]]]:
    """Evaluate the minimal step on a fixed path; preserve every failed epoch."""
    records = []
    fits = {}
    fit_parameters = tuple(dict.fromkeys(("F0", "F1", *shared_fit_parameters)))
    for rank, (score, lower, upper) in enumerate(intervals, start=1):
        for fraction in epoch_fractions:
            epoch = lower + float(fraction) * (upper - lower)
            candidate_id = f"spin_f1_frequency_step_{rank}_{float(fraction):.3f}"
            metadata = {
                "candidate_id": candidate_id, "model_kind": "continuous_frequency_step",
                "spin_frequency_order": 1, "glitch_frequency_order": 1,
                "changepoint_epoch_mjd": epoch, "changepoint_interval_mjd": (lower, upper),
                "changepoint_standardized_jump": score,
                "metric_scope": "fixed_retained_pulse_path",
            }
            try:
                fitter, _ = fit_candidate(
                    source_par_text=source_par_text, shared_model=shared_model,
                    shared_fit_parameters=fit_parameters, toas=toas,
                    spin_order=1, changepoint_epoch_mjd=epoch,
                    glitch_frequency_order=1, fit_iterations=fit_iterations,
                )
                record, residuals = fit_record(
                    candidate_id=candidate_id, fitter=fitter, toas=toas,
                    model_kind=metadata["model_kind"], spin_order=1,
                    changepoint_epoch_mjd=epoch,
                )
                fits[candidate_id] = (fitter, residuals)
                record.update(metadata, status="evaluated")
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                record = dict(metadata, status="failed", error=f"{type(error).__name__}: {error}")
            records.append(record)
    return records, fits
