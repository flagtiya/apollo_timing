from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from io import StringIO
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint.fitter import WLSFitter
from pint.models.model_builder import ModelBuilder, convert_tcb_tdb
from pint.residuals import Residuals

from timing_accelerator.backend.pint_backend import _set_numeric_model_parameter_value
from timing_accelerator.inference.pint_solution_artifacts import (
    extract_fitter_parameter_covariance,
)
from timing_accelerator.inference.isolated_spin_candidates import (
    _candidate_model,
    _fit_candidate,
    _freeze_to_contract,
    _session_changepoint_intervals,
    fit_frequency_step_family,
)
from timing_accelerator.inference.toa_white_noise_certification import (
    ToaWhiteNoiseCertificationConfig,
    ToaWhiteNoiseFit,
    fit_global_equad_on_fixed_path,
    normalized_likelihood_bic,
)


@dataclass(frozen=True)
class IsolatedUpdateModelSelectionConfig:
    """Evidence and complexity limits for extending an isolated update model."""

    enabled: bool = True
    maximum_spin_frequency_order: int = 5
    minimum_delta_bic: float = 10.0
    maximum_changepoint_intervals: int = 3
    changepoint_epoch_fractions: tuple[float, ...] = (0.01, 0.5, 0.99)
    minimum_added_parameter_sigma: float = 5.0
    minimum_residual_degrees_of_freedom: int = 5
    fit_iterations: int = 30
    toa_white_noise: ToaWhiteNoiseCertificationConfig = field(
        default_factory=ToaWhiteNoiseCertificationConfig
    )

    def __post_init__(self) -> None:
        if int(self.maximum_spin_frequency_order) < 2:
            raise ValueError("Isolated update model selection must allow at least F2.")
        if not np.isfinite(self.minimum_delta_bic) or self.minimum_delta_bic <= 0.0:
            raise ValueError("Isolated update minimum delta BIC must be positive.")
        if (
            not np.isfinite(self.minimum_added_parameter_sigma)
            or self.minimum_added_parameter_sigma <= 0.0
        ):
            raise ValueError("Added-parameter significance must be positive.")
        if int(self.maximum_changepoint_intervals) <= 0 or int(self.fit_iterations) <= 0:
            raise ValueError("Isolated update model-selection budgets must be positive.")
        if int(self.minimum_residual_degrees_of_freedom) <= 0:
            raise ValueError("Minimum residual degrees of freedom must be positive.")
        fractions = tuple(float(value) for value in self.changepoint_epoch_fractions)
        if not fractions or any(
            not np.isfinite(value) or not 0.0 < value < 1.0 for value in fractions
        ):
            raise ValueError("Changepoint epoch fractions must lie strictly in (0, 1).")


@dataclass(frozen=True)
class IsolatedUpdateModelSelectionResult:
    model: object
    fitter: object
    residuals: object
    fit_parameters: tuple[str, ...]
    selected_candidate_id: str
    selected: bool
    portfolio: tuple[dict[str, Any], ...]
    basic_portfolio_complete: bool
    high_order_portfolio_complete: bool
    model_selection_complete: bool
    high_order_portfolio_status: str
    toa_white_noise_selection: "IsolatedToaWhiteNoiseSelection | None" = None


@dataclass(frozen=True)
class IsolatedToaWhiteNoiseSelection:
    model: object
    fitter: object
    residuals: object
    fit_parameters: tuple[str, ...]
    selected_candidate_id: str
    portfolio: tuple[dict[str, Any], ...]
    model_selection_consistent: bool
    model_selection_complete: bool
    evidence_passed: bool
    status: str
    fit: ToaWhiteNoiseFit
    effective_parameter_count: int


def _bic(*, chi2: float, parameter_count: int, toa_count: int) -> float:
    return float(chi2 + int(parameter_count) * math.log(max(int(toa_count), 1)))


def _position_block(model: object) -> tuple[str, str]:
    parameter_names = {str(name).upper() for name in model.params}
    if {"RAJ", "DECJ"}.issubset(parameter_names):
        return ("RAJ", "DECJ")
    if {"ELONG", "ELAT"}.issubset(parameter_names):
        return ("ELONG", "ELAT")
    raise ValueError(
        "Automatic isolated model selection requires RAJ/DECJ or ELONG/ELAT."
    )


def _block_significance(
    *,
    model: object,
    covariance: np.ndarray,
    fit_parameters: Sequence[str],
    block_parameters: Sequence[str],
    reference_values: Mapping[str, float],
) -> float:
    names = tuple(str(name).upper() for name in fit_parameters)
    block = tuple(str(name).upper() for name in block_parameters)
    if not block or any(name not in names for name in block):
        return 0.0
    indices = [names.index(name) for name in block]
    offsets = np.asarray(
        [float(model[name].value) - float(reference_values[name]) for name in block],
        dtype=float,
    )
    block_covariance = covariance[np.ix_(indices, indices)]
    if not np.all(np.isfinite(block_covariance)):
        return 0.0
    try:
        wald = float(
            offsets
            @ np.linalg.pinv(block_covariance, hermitian=True)
            @ offsets
        )
    except np.linalg.LinAlgError:
        return 0.0
    return math.sqrt(max(wald, 0.0)) if np.isfinite(wald) else 0.0


def _scale_invariant_covariance_rank(covariance: np.ndarray) -> int:
    """Rank a covariance after removing native-unit scale differences."""

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return 0
    diagonal = np.diag(matrix)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        return 0
    scale = np.sqrt(diagonal)
    correlation = matrix / np.outer(scale, scale)
    correlation = 0.5 * (correlation + correlation.T)
    if not np.all(np.isfinite(correlation)):
        return 0
    return int(np.linalg.matrix_rank(correlation, tol=1.0e-8))


def _fit_record(
    *,
    candidate_id: str,
    fitter: object,
    toas: object,
    model_kind: str,
    spin_order: int,
    changepoint_epoch_mjd: float | None,
    added_parameter_blocks: Mapping[str, Sequence[str]] | None = None,
    reference_values: Mapping[str, float] | None = None,
    timing_fit_parameters: Sequence[str] | None = None,
    effective_parameter_count: int | None = None,
    bic: float | None = None,
) -> tuple[dict[str, Any], object]:
    residuals = Residuals(
        toas,
        fitter.model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    fit_parameters = tuple(
        str(name).upper()
        for name in (
            fitter.model.free_params
            if timing_fit_parameters is None
            else timing_fit_parameters
        )
    )
    if added_parameter_blocks is None:
        added_parameters = tuple(
            name
            for name in fit_parameters
            if (
                name.startswith("F")
                and name[1:].isdigit()
                and int(name[1:]) >= 2
            )
            or name in {"GLF0_1", "GLF1_1"}
        )
        blocks = {name: (name,) for name in added_parameters}
    else:
        blocks = {
            str(block_name).upper(): tuple(str(name).upper() for name in names)
            for block_name, names in added_parameter_blocks.items()
        }
        added_parameters = tuple(
            dict.fromkeys(name for names in blocks.values() for name in names)
        )
    references = {
        name: 0.0 for name in added_parameters
    }
    references.update(
        {
            str(name).upper(): float(value)
            for name, value in (reference_values or {}).items()
        }
    )
    covariance_error = None
    try:
        covariance = extract_fitter_parameter_covariance(fitter, fit_parameters)
        covariance_rank = _scale_invariant_covariance_rank(covariance)
    except (
        ArithmeticError,
        AttributeError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        covariance = np.full(
            (len(fit_parameters), len(fit_parameters)),
            np.nan,
            dtype=float,
        )
        covariance_rank = 0
        covariance_error = f"{type(error).__name__}: {error}"
    covariance_full_rank = bool(covariance_rank == len(fit_parameters))
    uncertainties = {}
    significances = {}
    for name in added_parameters:
        uncertainty = getattr(fitter.model[name], "uncertainty_value", None)
        uncertainty_value = float(uncertainty) if uncertainty is not None else np.nan
        value = float(fitter.model[name].value)
        uncertainties[name] = uncertainty_value
        significances[name] = (
            abs(value / uncertainty_value)
            if np.isfinite(uncertainty_value) and uncertainty_value > 0.0
            else 0.0
        )
    block_significances = {
        block_name: _block_significance(
            model=fitter.model,
            covariance=covariance,
            fit_parameters=fit_parameters,
            block_parameters=names,
            reference_values=references,
        )
        for block_name, names in blocks.items()
    }
    chi2 = float(residuals.chi2)
    active_parameter_count = (
        len(fit_parameters)
        + int(changepoint_epoch_mjd is not None and "GLEP_1" not in fit_parameters)
        if effective_parameter_count is None
        else int(effective_parameter_count)
    )
    residual_degrees_of_freedom = int(
        getattr(
            residuals,
            "dof",
            max(len(toas.table) - active_parameter_count - 1, 0),
        )
    )
    record = {
        "candidate_id": candidate_id,
        "model_kind": model_kind,
        "spin_frequency_order": int(spin_order),
        "changepoint_epoch_mjd": changepoint_epoch_mjd,
        "fit_parameters": fit_parameters,
        "parameter_count": len(fit_parameters),
        "effective_parameter_count": active_parameter_count,
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "added_parameters": added_parameters,
        "added_parameter_blocks": blocks,
        "added_parameter_uncertainties": uncertainties,
        "added_parameter_significances": significances,
        "minimum_added_parameter_significance": (
            min(block_significances.values()) if block_significances else None
        ),
        "added_block_significances": block_significances,
        "covariance_rank": covariance_rank,
        "covariance_dimension": len(fit_parameters),
        "covariance_full_rank": covariance_full_rank,
        "covariance_error": covariance_error,
        "chi2": chi2,
        "reduced_chi2": float(residuals.reduced_chi2),
        "weighted_rms_us": float(residuals.rms_weighted().to_value(u.us)),
        "bic": (
            float(bic)
            if bic is not None
            else _bic(
                chi2=chi2,
                parameter_count=active_parameter_count,
                toa_count=len(toas.table),
            )
        ),
        "fitted_parameters": {
            name: float(fitter.model[name].value) for name in fit_parameters
        },
    }
    return record, residuals




def _fit_basic_candidate(
    *,
    source_model: object,
    shared_model: object,
    fit_parameters: tuple[str, ...],
    toas: object,
    fit_iterations: int,
) -> tuple[object, object]:
    """Fit one nested F0/F1/position hypothesis on the fixed pulse path."""

    model = deepcopy(source_model)
    for name in fit_parameters:
        _set_numeric_model_parameter_value(model, name, float(shared_model[name].value))
    _freeze_to_contract(model, fit_parameters)
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




def select_isolated_update_model_on_fixed_path(
    *,
    seed_par_path: Path,
    baseline_fitter: object,
    certified_toas: object,
    baseline_fit_parameters: Sequence[str],
    strict_reduced_chi2_threshold: float,
    config: IsolatedUpdateModelSelectionConfig | None = None,
) -> IsolatedUpdateModelSelectionResult:
    """Select a basic F0/F1/position model, then any supported high-order extension."""

    active = config or IsolatedUpdateModelSelectionConfig()
    source = Path(seed_par_path).read_text(encoding="utf-8")
    source_model = ModelBuilder()(StringIO(source), allow_tcb=True)
    if "UNITS" in source_model.params and source_model["UNITS"].value == "TCB":
        convert_tcb_tdb(source_model)
    position_names = _position_block(source_model)
    reference_values = {
        name: float(source_model[name].value)
        for name in ("F0", "F1", *position_names)
    }
    basic_specs = (
        ("basic_f0", ("F0",), {}),
        ("basic_f0_f1", ("F0", "F1"), {"F1": ("F1",)}),
        (
            "basic_f0_position",
            ("F0", *position_names),
            {"POSITION": position_names},
        ),
        (
            "basic_f0_f1_position",
            ("F0", "F1", *position_names),
            {"F1": ("F1",), "POSITION": position_names},
        ),
    )
    records: list[dict[str, Any]] = []
    fits: dict[str, tuple[object, object]] = {}
    basic_record_by_id: dict[str, dict[str, Any]] = {}
    full_id = "basic_f0_f1_position"

    for candidate_id, fit_parameters, blocks in basic_specs:
        try:
            if candidate_id == full_id:
                fitter = baseline_fitter
                residuals = Residuals(
                    certified_toas,
                    fitter.model,
                    subtract_mean=True,
                    use_abs_phase=True,
                    track_mode="use_pulse_numbers",
                )
            else:
                fitter, residuals = _fit_basic_candidate(
                    source_model=source_model,
                    shared_model=baseline_fitter.model,
                    fit_parameters=fit_parameters,
                    toas=certified_toas,
                    fit_iterations=active.fit_iterations,
                )
            record, residuals = _fit_record(
                candidate_id=candidate_id,
                fitter=fitter,
                toas=certified_toas,
                model_kind="basic_spin_astrometry_block",
                spin_order=1,
                changepoint_epoch_mjd=None,
                added_parameter_blocks=blocks,
                reference_values=reference_values,
            )
            record["basic_model"] = True
            record["requested_fit_parameters"] = fit_parameters
            record.setdefault("added_parameter_blocks", blocks)
            record.setdefault("added_block_significances", {})
            records.append(record)
            basic_record_by_id[candidate_id] = record
            fits[candidate_id] = (fitter, residuals)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            record = {
                "candidate_id": candidate_id,
                "model_kind": "basic_spin_astrometry_block",
                "basic_model": True,
                "requested_fit_parameters": fit_parameters,
                "added_parameter_blocks": blocks,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            records.append(record)
            basic_record_by_id[candidate_id] = record

    parent_by_block = {
        "basic_f0_f1": {"F1": "basic_f0"},
        "basic_f0_position": {"POSITION": "basic_f0"},
        "basic_f0_f1_position": {
            "F1": "basic_f0_position",
            "POSITION": "basic_f0_f1",
        },
    }

    def is_identifiable(record: Mapping[str, Any]) -> bool:
        minimum_significance = record.get("minimum_added_parameter_significance")
        return bool(
            int(record.get("residual_degrees_of_freedom", 0))
            >= active.minimum_residual_degrees_of_freedom
            and bool(record.get("covariance_full_rank", True))
            and (
                minimum_significance is None
                or float(minimum_significance) >= active.minimum_added_parameter_sigma
            )
        )

    def consider_frequency_steps(
        portfolio: Sequence[dict[str, Any]], baseline: Mapping[str, Any], current_id: str,
    ) -> str:
        """Apply the same selection gates to the separate minimal step family."""
        by_id = {str(item["candidate_id"]): item for item in portfolio if "bic" in item}
        qualified = []
        for record in by_id.values():
            if record.get("model_kind") != "continuous_frequency_step":
                continue
            record["identifiable"] = is_identifiable(record)
            record["delta_bic_vs_baseline"] = float(baseline["bic"]) - float(record["bic"])
            record["evidence_passed"] = bool(
                record["delta_bic_vs_baseline"] >= active.minimum_delta_bic
                and record["reduced_chi2"] < strict_reduced_chi2_threshold
                and record["identifiable"]
            )
            if record["evidence_passed"]:
                qualified.append(record)
        if not qualified:
            return current_id
        best = min(qualified, key=lambda item: (float(item["bic"]), item["candidate_id"]))
        return (str(best["candidate_id"]) if float(best["bic"]) < float(by_id[current_id]["bic"])
                else current_id)

    for candidate_id, record in basic_record_by_id.items():
        if "bic" not in record:
            record["identifiable"] = False
            record["evidence_passed"] = False
            continue
        record["identifiable"] = is_identifiable(record)
        deltas: dict[str, float] = {}
        for block_name, parent_id in parent_by_block.get(candidate_id, {}).items():
            parent = basic_record_by_id.get(parent_id, {})
            if "bic" in parent:
                deltas[block_name] = float(parent["bic"]) - float(record["bic"])
        record["delta_bic_by_block"] = deltas
        block_significances = record.get("added_block_significances", {})
        record["evidence_passed"] = bool(
            record["identifiable"]
            and all(
                delta >= active.minimum_delta_bic for delta in deltas.values()
            )
            and all(
                float(block_significances.get(block_name, 0.0))
                >= active.minimum_added_parameter_sigma
                for block_name in deltas
            )
        )
        record["status"] = "evaluated"

    basic_portfolio_complete = all("bic" in record for record in records)
    qualified_basic = [record for record in records if record.get("evidence_passed")]
    finite_basic = [record for record in records if "bic" in record]
    selected_basic_record = min(
        qualified_basic or finite_basic,
        key=lambda item: (float(item["bic"]), int(item.get("parameter_count", 0))),
        default=basic_record_by_id[full_id],
    )
    selected_basic_id = str(selected_basic_record["candidate_id"])
    if selected_basic_id not in fits:
        selected_basic_id = full_id
        selected_basic_record = basic_record_by_id[full_id]
    selected_basic_fitter, selected_basic_residuals = fits[selected_basic_id]
    selected_basic_fit_parameters = tuple(
        str(name).upper() for name in selected_basic_fitter.model.free_params
    )

    if not active.enabled:
        for record in records:
            record["selected"] = record.get("candidate_id") == selected_basic_id
        return IsolatedUpdateModelSelectionResult(
            model=selected_basic_fitter.model,
            fitter=selected_basic_fitter,
            residuals=selected_basic_residuals,
            fit_parameters=selected_basic_fit_parameters,
            selected_candidate_id=selected_basic_id,
            selected=selected_basic_id != full_id,
            portfolio=tuple(records),
            basic_portfolio_complete=basic_portfolio_complete,
            high_order_portfolio_complete=False,
            model_selection_complete=False,
            high_order_portfolio_status="disabled",
        )

    selected_basic_record["selected_basic"] = True
    high_order_complete = True
    high_order_status = "evaluated_to_configured_limit"
    selected_id = selected_basic_id
    baseline_bic = float(selected_basic_record.get("bic", math.inf))

    if float(selected_basic_record.get("reduced_chi2", math.inf)) < strict_reduced_chi2_threshold:
        high_order_status = "not_required_basic_model_adequate"
        high_order_complete = True
    else:
        step_records, step_fits = fit_frequency_step_family(
            source_par_text=source, shared_model=selected_basic_fitter.model,
            shared_fit_parameters=selected_basic_fit_parameters, toas=certified_toas,
            intervals=_session_changepoint_intervals(
                toas=certified_toas, residuals=selected_basic_residuals,
                maximum_intervals=active.maximum_changepoint_intervals,
            ),
            epoch_fractions=active.changepoint_epoch_fractions,
            fit_iterations=active.fit_iterations,
            fit_candidate=_fit_candidate, fit_record=_fit_record,
        )
        records.extend(step_records)
        fits.update(step_fits)
        high_order_complete = bool(step_records) and all("bic" in record for record in step_records)
        for spin_order in range(2, int(active.maximum_spin_frequency_order) + 1):
            smooth_id = f"spin_f{spin_order}"
            try:
                smooth_fitter, smooth_residuals = _fit_candidate(
                    source_par_text=source,
                    shared_model=selected_basic_fitter.model,
                    shared_fit_parameters=selected_basic_fit_parameters,
                    toas=certified_toas,
                    spin_order=spin_order,
                    changepoint_epoch_mjd=None,
                    fit_iterations=active.fit_iterations,
                )
                smooth_record, smooth_residuals = _fit_record(
                    candidate_id=smooth_id,
                    fitter=smooth_fitter,
                    toas=certified_toas,
                    model_kind="smooth_spin_derivative_block",
                    spin_order=spin_order,
                    changepoint_epoch_mjd=None,
                )
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                high_order_complete = False
                records.append(
                    {
                        "candidate_id": smooth_id,
                        "model_kind": "smooth_spin_derivative_block",
                        "spin_frequency_order": spin_order,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            records.append(smooth_record)
            fits[smooth_id] = (smooth_fitter, smooth_residuals)
            smooth_record["identifiable"] = is_identifiable(smooth_record)
            smooth_improvement = baseline_bic - float(smooth_record["bic"])
            smooth_record["delta_bic_vs_baseline"] = smooth_improvement
            if (
                smooth_improvement >= active.minimum_delta_bic
                and smooth_record["reduced_chi2"] < strict_reduced_chi2_threshold
                and is_identifiable(smooth_record)
            ):
                selected_id = smooth_id
                high_order_status = "evidence_selected_smooth_spin"
                break

            intervals = _session_changepoint_intervals(
                toas=certified_toas,
                residuals=smooth_residuals,
                maximum_intervals=active.maximum_changepoint_intervals,
            )
            accepted_extensions = []
            for interval_rank, (score, lower, upper) in enumerate(intervals, start=1):
                for fraction in active.changepoint_epoch_fractions:
                    epoch = lower + float(fraction) * (upper - lower)
                    candidate_id = (
                        f"spin_f{spin_order}_changepoint_{interval_rank}_"
                        f"{float(fraction):.3f}"
                    )
                    try:
                        fitter, residuals = _fit_candidate(
                            source_par_text=source,
                            shared_model=selected_basic_fitter.model,
                            shared_fit_parameters=selected_basic_fit_parameters,
                            toas=certified_toas,
                            spin_order=spin_order,
                            changepoint_epoch_mjd=epoch,
                            fit_iterations=active.fit_iterations,
                        )
                        record, residuals = _fit_record(
                            candidate_id=candidate_id,
                            fitter=fitter,
                            toas=certified_toas,
                            model_kind="spin_derivative_plus_glitch_changepoint",
                            spin_order=spin_order,
                            changepoint_epoch_mjd=epoch,
                        )
                        record["changepoint_interval_mjd"] = (lower, upper)
                        record["changepoint_standardized_jump"] = score
                        record["delta_bic_vs_baseline"] = baseline_bic - float(record["bic"])
                        record["delta_bic_vs_smooth"] = float(smooth_record["bic"]) - float(
                            record["bic"]
                        )
                    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                        high_order_complete = False
                        records.append(
                            {
                                "candidate_id": candidate_id,
                                "model_kind": "spin_derivative_plus_glitch_changepoint",
                                "spin_frequency_order": spin_order,
                                "changepoint_epoch_mjd": epoch,
                                "status": "failed",
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                        continue
                    records.append(record)
                    fits[candidate_id] = (fitter, residuals)
                    record["identifiable"] = is_identifiable(record)
                    if (
                        record["delta_bic_vs_baseline"] >= active.minimum_delta_bic
                        and record["delta_bic_vs_smooth"] >= active.minimum_delta_bic
                        and record["reduced_chi2"] < strict_reduced_chi2_threshold
                        and is_identifiable(record)
                    ):
                        accepted_extensions.append(record)
            if accepted_extensions:
                accepted_extensions.sort(
                    key=lambda record: (record["bic"], record["candidate_id"])
                )
                selected_id = str(accepted_extensions[0]["candidate_id"])
                high_order_status = "evidence_selected_spin_glitch"
                break

    selected_id = consider_frequency_steps(records, selected_basic_record, selected_id)
    if selected_id.startswith("spin_f1_frequency_step_"):
        high_order_status = "evidence_selected_continuous_frequency_step"

    model_selection_complete = basic_portfolio_complete and high_order_complete
    toa_white_noise_selection: IsolatedToaWhiteNoiseSelection | None = None
    if active.toa_white_noise.enabled:
        # Preserve raw failures while evaluating available models diagnostically.
        # Incomplete portfolios remain ineligible for scientific certification.
        conditional_records: list[dict[str, Any]] = [
            dict(record, noise_fit_status="not_evaluated_timing_fit_failed")
            for record in records if "bic" not in record
        ]
        conditional_fits: dict[str, tuple[ToaWhiteNoiseFit, dict[str, Any]]] = {}
        raw_by_id = {
            str(record.get("candidate_id")): record
            for record in records
            if "bic" in record
        }
        for candidate_id, raw_record in raw_by_id.items():
            raw_fit = fits.get(candidate_id)
            if raw_fit is None:
                continue
            try:
                white_fit = fit_global_equad_on_fixed_path(
                    model=raw_fit[0].model,
                    toas=certified_toas,
                    config=active.toa_white_noise,
                )
                timing_parameters = tuple(
                    str(name).upper()
                    for name in raw_record.get(
                        "fit_parameters",
                        tuple(
                            name
                            for name in raw_fit[0].model.free_params
                            if not str(name).upper().startswith(
                                ("EFAC", "EQUAD", "ECORR")
                            )
                        ),
                    )
                )
                # A grid-searched epoch is frozen in PINT, but is still a
                # parameter selected from these data. Carry its penalty in
                # both likelihoods and in the fit used across pulse paths.
                effective_timing_parameter_count = int(
                    raw_record["effective_parameter_count"]
                )
                effective_parameter_count = (
                    effective_timing_parameter_count + white_fit.noise_parameter_count
                )
                reported_bic = normalized_likelihood_bic(
                    log_likelihood=white_fit.reported_error_log_likelihood,
                    parameter_count=effective_timing_parameter_count,
                    toa_count=white_fit.toa_count,
                )
                conditioned_bic = normalized_likelihood_bic(
                    log_likelihood=white_fit.noise_conditioned_log_likelihood,
                    parameter_count=effective_parameter_count,
                    toa_count=white_fit.toa_count,
                )
                white_fit = replace(
                    white_fit,
                    reported_error_bic=reported_bic,
                    noise_conditioned_bic=conditioned_bic,
                    delta_bic=reported_bic - conditioned_bic,
                )
                conditional_record, conditional_residuals = _fit_record(
                    candidate_id=candidate_id,
                    fitter=white_fit.fitter,
                    toas=certified_toas,
                    model_kind=str(raw_record.get("model_kind", "isolated")),
                    spin_order=int(raw_record.get("spin_frequency_order", 1)),
                    changepoint_epoch_mjd=raw_record.get("changepoint_epoch_mjd"),
                    added_parameter_blocks=raw_record.get("added_parameter_blocks"),
                    reference_values=(
                        reference_values if raw_record.get("basic_model") else None
                    ),
                    timing_fit_parameters=timing_parameters,
                    effective_parameter_count=effective_parameter_count,
                    bic=white_fit.noise_conditioned_bic,
                )
                for key in (
                    "basic_model",
                    "requested_fit_parameters",
                    "changepoint_interval_mjd",
                    "changepoint_standardized_jump",
                    "glitch_frequency_order",
                    "metric_scope",
                ):
                    if key in raw_record:
                        conditional_record[key] = raw_record[key]
                conditional_record["reported_error_bic"] = float(
                    white_fit.reported_error_bic
                )
                conditional_record["noise_conditioned_bic"] = float(
                    white_fit.noise_conditioned_bic
                )
                conditional_record["noise_conditioned_delta_bic"] = float(
                    white_fit.delta_bic
                )
                conditional_record["toa_white_noise"] = white_fit.to_payload()
                conditional_record["status"] = "evaluated"
                conditional_records.append(conditional_record)
                conditional_fits[candidate_id] = (white_fit, conditional_record)
            except (
                ArithmeticError,
                RuntimeError,
                ValueError,
                np.linalg.LinAlgError,
            ) as error:
                conditional_records.append(
                    {
                        "candidate_id": candidate_id,
                        "model_kind": raw_record.get("model_kind", "isolated"),
                        "basic_model": bool(raw_record.get("basic_model", False)),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

        conditional_by_id = {
            str(record.get("candidate_id")): record
            for record in conditional_records
            if "bic" in record
        }
        for candidate_id, record in conditional_by_id.items():
            if not record.get("basic_model"):
                continue
            record["identifiable"] = is_identifiable(record)
            deltas: dict[str, float] = {}
            for block_name, parent_id in parent_by_block.get(candidate_id, {}).items():
                parent = conditional_by_id.get(parent_id, {})
                if "bic" in parent:
                    deltas[block_name] = float(parent["bic"]) - float(record["bic"])
            record["delta_bic_by_block"] = deltas
            block_significances = record.get("added_block_significances", {})
            record["evidence_passed"] = bool(
                record["identifiable"]
                and all(
                    delta >= active.minimum_delta_bic for delta in deltas.values()
                )
                and all(
                    float(block_significances.get(block_name, 0.0))
                    >= active.minimum_added_parameter_sigma
                    for block_name in deltas
                )
            )

        conditional_basic_records = [
            record
            for record in conditional_by_id.values()
            if record.get("basic_model")
        ]
        conditional_basic_complete = all(
            candidate_id in conditional_by_id
            for candidate_id, raw_record in raw_by_id.items()
            if raw_record.get("basic_model")
        )
        conditional_qualified_basic = [
            record for record in conditional_basic_records if record.get("evidence_passed")
        ]
        conditional_selected_basic = min(
            conditional_qualified_basic or conditional_basic_records,
            key=lambda item: (float(item["bic"]), int(item.get("parameter_count", 0))),
            default=conditional_by_id.get(selected_basic_id),
        )
        conditional_selected_basic_id = (
            None
            if conditional_selected_basic is None
            else str(conditional_selected_basic["candidate_id"])
        )
        conditional_selected_id = conditional_selected_basic_id
        conditional_high_order_complete = bool(high_order_complete) and all(
            candidate_id in conditional_by_id for candidate_id, raw_record in raw_by_id.items()
            if not raw_record.get("basic_model")
        )
        conditional_high_order_status = "evaluated_to_configured_limit"
        if conditional_selected_basic is None:
            conditional_high_order_complete = False
        elif float(conditional_selected_basic.get("reduced_chi2", math.inf)) < strict_reduced_chi2_threshold:
            conditional_high_order_status = "not_required_basic_model_adequate"
        else:
            conditional_baseline_bic = float(conditional_selected_basic["bic"])
            for spin_order in range(2, int(active.maximum_spin_frequency_order) + 1):
                smooth_id = f"spin_f{spin_order}"
                smooth_record = conditional_by_id.get(smooth_id)
                if smooth_record is None:
                    conditional_high_order_complete = False
                    break
                smooth_record["identifiable"] = is_identifiable(smooth_record)
                smooth_record["delta_bic_vs_baseline"] = (
                    conditional_baseline_bic - float(smooth_record["bic"])
                )
                if (
                    smooth_record["delta_bic_vs_baseline"] >= active.minimum_delta_bic
                    and float(smooth_record.get("reduced_chi2", math.inf))
                    < strict_reduced_chi2_threshold
                    and smooth_record["identifiable"]
                ):
                    conditional_selected_id = smooth_id
                    conditional_high_order_status = "evidence_selected_smooth_spin"
                    break
                accepted_extensions = []
                for candidate_id, record in conditional_by_id.items():
                    if not candidate_id.startswith(f"spin_f{spin_order}_changepoint_"):
                        continue
                    record["identifiable"] = is_identifiable(record)
                    record["delta_bic_vs_baseline"] = (
                        conditional_baseline_bic - float(record["bic"])
                    )
                    record["delta_bic_vs_smooth"] = float(smooth_record["bic"]) - float(
                        record["bic"]
                    )
                    if (
                        record["delta_bic_vs_baseline"] >= active.minimum_delta_bic
                        and record["delta_bic_vs_smooth"] >= active.minimum_delta_bic
                        and float(record.get("reduced_chi2", math.inf))
                        < strict_reduced_chi2_threshold
                        and record["identifiable"]
                    ):
                        accepted_extensions.append(record)
                if accepted_extensions:
                    accepted_extensions.sort(
                        key=lambda record: (record["bic"], record["candidate_id"])
                    )
                    conditional_selected_id = str(accepted_extensions[0]["candidate_id"])
                    conditional_high_order_status = "evidence_selected_spin_glitch"
                    break

        if conditional_selected_basic is not None and conditional_selected_id is not None:
            conditional_selected_id = consider_frequency_steps(
                conditional_records, conditional_selected_basic, conditional_selected_id,
            )

        conditional_model_selection_complete = bool(
            model_selection_complete and conditional_basic_complete and conditional_high_order_complete
        )
        model_selection_consistent = bool(
            conditional_selected_basic_id == selected_basic_id
            and conditional_selected_id == selected_id
        )
        selected_conditional_fit = (
            None
            if conditional_selected_id is None
            else conditional_fits.get(conditional_selected_id)
        )
        evidence_passed = bool(
            conditional_model_selection_complete
            and model_selection_consistent
            and selected_conditional_fit is not None
            and selected_conditional_fit[0].delta_bic
            >= active.toa_white_noise.minimum_noise_delta_bic
            and selected_conditional_fit[0].boundary_valid
        )
        status = (
            "noise_conditioned_diagnostic_incomplete_portfolio"
            if not conditional_model_selection_complete
            else "noise_conditioned_unique_within_fixed_path"
            if evidence_passed
            else "noise_conditioned_model_or_path_not_unique"
        )
        if selected_conditional_fit is None:
            fallback_fit = conditional_fits.get(selected_id)
            if fallback_fit is not None:
                selected_conditional_fit = fallback_fit
        if selected_conditional_fit is not None:
            selected_white_fit, selected_white_record = selected_conditional_fit
            for record in conditional_records:
                record["selected"] = record.get("candidate_id") == conditional_selected_id
            toa_white_noise_selection = IsolatedToaWhiteNoiseSelection(
                model=selected_white_fit.model,
                fitter=selected_white_fit.fitter,
                residuals=selected_white_fit.residuals,
                fit_parameters=selected_white_fit.timing_fit_parameters,
                selected_candidate_id=(
                    conditional_selected_id or selected_id
                ),
                portfolio=tuple(conditional_records),
                model_selection_consistent=model_selection_consistent,
                model_selection_complete=conditional_model_selection_complete,
                evidence_passed=evidence_passed,
                status=status,
                fit=selected_white_fit,
                effective_parameter_count=int(
                    selected_white_record["effective_parameter_count"]
                ),
            )

    selected_fitter, selected_residuals = fits[selected_id]
    for record in records:
        record["selected"] = record.get("candidate_id") == selected_id
    return IsolatedUpdateModelSelectionResult(
        model=selected_fitter.model,
        fitter=selected_fitter,
        residuals=selected_residuals,
        fit_parameters=tuple(str(name).upper() for name in selected_fitter.model.free_params),
        selected_candidate_id=selected_id,
        selected=selected_id != full_id,
        portfolio=tuple(records),
        basic_portfolio_complete=basic_portfolio_complete,
        high_order_portfolio_complete=high_order_complete,
        model_selection_complete=model_selection_complete,
        high_order_portfolio_status=high_order_status,
        toa_white_noise_selection=toa_white_noise_selection,
    )


__all__ = [
    "IsolatedUpdateModelSelectionConfig",
    "IsolatedUpdateModelSelectionResult",
    "IsolatedToaWhiteNoiseSelection",
    "select_isolated_update_model_on_fixed_path",
]
