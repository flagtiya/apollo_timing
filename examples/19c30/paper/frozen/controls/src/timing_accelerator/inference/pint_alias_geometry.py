from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from astropy import units as u
from pint.residuals import Residuals
from pint.simulation import make_fake_toas_uniform

from timing_accelerator.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _prime_local_ephemeris,
    _temporary_clock_override,
)
from timing_accelerator.inference.alias_manifold_solver import (
    AliasGeometryPayload,
    LocalPhaseSlopeEvidence,
)
from timing_accelerator.inference.cluster_manifest import (
    load_cluster_manifest,
    resolve_cluster_manifest,
    resolve_source_session_manifest,
)
from timing_accelerator.inference.pulse_number_reference import ensure_pulse_numbers


def _with_pulse_numbers(toas, model):
    tracked_toas = deepcopy(toas)
    return ensure_pulse_numbers(tracked_toas, model)


def _extract_phase_residuals(model, toas) -> np.ndarray:
    tracked_toas = _with_pulse_numbers(toas, model)
    residuals = Residuals(
        tracked_toas,
        model,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    return np.asarray(residuals.phase_resids.to_value(u.dimensionless_unscaled), dtype=float)


def _cluster_index_map(manifest) -> dict[str, object]:
    return {cluster.cluster_id: cluster for cluster in manifest.clusters}


def _compute_gap_phase_residuals(phase_residuals: np.ndarray, manifest) -> np.ndarray:
    if not manifest.gaps:
        return np.zeros(0, dtype=float)
    cluster_map = _cluster_index_map(manifest)
    gap_values = []
    for gap in manifest.gaps:
        left_cluster = cluster_map[gap.left_cluster]
        right_cluster = cluster_map[gap.right_cluster]
        if not left_cluster.toa_indices or not right_cluster.toa_indices:
            raise ValueError(f"Gap '{gap.gap_id}' references unresolved cluster indices.")
        left_anchor = left_cluster.toa_indices[-1]
        right_anchor = right_cluster.toa_indices[0]
        gap_values.append(float(phase_residuals[right_anchor] - phase_residuals[left_anchor]))
    return np.asarray(gap_values, dtype=float)


def _require_parameter(model, parameter_name: str) -> None:
    if parameter_name not in model.params:
        raise ValueError(f"Parameter '{parameter_name}' is missing from the timing model.")


def _parameter_phase_derivative(model, toas, parameter_name: str) -> np.ndarray:
    _require_parameter(model, parameter_name)
    derivative = model.d_phase_d_param(toas, param=parameter_name, delay=None)
    if hasattr(derivative, "to_value"):
        values = derivative.to_value(derivative.unit)
    else:
        values = derivative
    return np.asarray(values, dtype=float).reshape(-1)


def _gap_derivative_column(model, toas, manifest, parameter_name: str) -> np.ndarray:
    if not manifest.gaps:
        return np.zeros(0, dtype=float)
    phase_derivative = _parameter_phase_derivative(model, toas, parameter_name)
    return _compute_gap_phase_residuals(phase_derivative, manifest)


def _reference_epoch_mjd(model, toas) -> float:
    if "PEPOCH" in model.params and model["PEPOCH"].value is not None:
        return float(model["PEPOCH"].value)
    mjds = toas.get_mjds().to_value(u.day)
    return float(np.mean(mjds))


def _analytic_phase_design_value(parameter_name: str, delta_t_seconds: float) -> float | None:
    name = parameter_name.upper()
    if len(name) >= 2 and name.startswith("F") and name[1:].isdigit():
        order = int(name[1:])
        return float(delta_t_seconds ** (order + 1) / math.factorial(order + 1))
    return None


def _representative_followup_frequency(toas) -> u.Quantity:
    frequencies = np.asarray(toas.table["freq"].quantity.to_value(u.MHz), dtype=float)
    finite = frequencies[np.isfinite(frequencies) & (frequencies > 0.0)]
    if finite.size == 0:
        return 1400.0 * u.MHz
    return float(np.median(finite)) * u.MHz


def _representative_followup_error(toas) -> u.Quantity:
    sigma_seconds = np.asarray(toas.get_errors().to_value(u.s), dtype=float)
    finite = sigma_seconds[np.isfinite(sigma_seconds) & (sigma_seconds > 0.0)]
    if finite.size == 0:
        return 1.0e-6 * u.s
    return float(np.median(finite)) * u.s


def _representative_followup_observatory(toas) -> str:
    observatories = [str(item) for item in toas.get_obss()]
    if not observatories:
        return "@"  # solar-system barycenter fallback accepted by PINT
    return Counter(observatories).most_common(1)[0][0]


def _build_followup_probe_toas(model, toas, epoch_mjd: float, clock_override_path: Path | None):
    ephem = model["EPHEM"].value if "EPHEM" in model.params else "DE421"
    _prime_local_ephemeris(ephem)
    with _temporary_clock_override(clock_override_path):
        return make_fake_toas_uniform(
            epoch_mjd,
            epoch_mjd,
            1,
            model=model,
            error=_representative_followup_error(toas),
            freq=_representative_followup_frequency(toas),
            obs=_representative_followup_observatory(toas),
            add_noise=False,
            include_bipm=False,
        )


def _build_phase_design_row(
    model,
    toas,
    parameter_names: Sequence[str],
    epoch_mjd: float,
    clock_override_path: Path | None = None,
) -> np.ndarray:
    reference_epoch = _reference_epoch_mjd(model, toas)
    delta_t_seconds = float(epoch_mjd - reference_epoch) * 86400.0
    row = []
    probe_toas = None
    for parameter_name in parameter_names:
        analytic = _analytic_phase_design_value(parameter_name, delta_t_seconds)
        if analytic is not None:
            row.append(analytic)
            continue
        if probe_toas is None:
            probe_toas = _build_followup_probe_toas(
                model=model,
                toas=toas,
                epoch_mjd=float(epoch_mjd),
                clock_override_path=clock_override_path,
            )
        row.append(float(_parameter_phase_derivative(model, probe_toas, parameter_name)[0]))
    return np.asarray(row, dtype=float)


def _estimate_toa_error_phase(model, toas) -> float:
    sigma_seconds = np.asarray(toas.get_errors().to(u.s).value, dtype=float)
    representative_sigma = float(np.median(sigma_seconds)) if sigma_seconds.size else 1.0e-6
    spin_hz = abs(float(model["F0"].value)) if "F0" in model.params else 1.0
    return max(representative_sigma * spin_hz, 1.0e-12)


def _build_local_phase_slope_evidence(
    *,
    phase_residuals: np.ndarray,
    parameter_phase_derivatives: np.ndarray,
    mjds: np.ndarray,
    phase_uncertainties: np.ndarray,
    manifest,
    min_toas: int = 2,
) -> LocalPhaseSlopeEvidence:
    residuals = np.asarray(phase_residuals, dtype=float)
    derivatives = np.asarray(parameter_phase_derivatives, dtype=float)
    epochs = np.asarray(mjds, dtype=float)
    uncertainties = np.asarray(phase_uncertainties, dtype=float)
    if derivatives.ndim != 2 or derivatives.shape[0] != residuals.size:
        raise ValueError("parameter_phase_derivatives must have one row per phase residual.")
    if epochs.shape != residuals.shape or uncertainties.shape != residuals.shape:
        raise ValueError("MJD and phase uncertainty arrays must match the phase residuals.")

    cluster_ids: list[str] = []
    cluster_mjd: list[float] = []
    spans: list[float] = []
    phase_changes: list[float] = []
    jacobian_rows: list[np.ndarray] = []
    phase_change_sigmas: list[float] = []
    required_toas = max(int(min_toas), 2)

    for cluster in manifest.clusters:
        indices = np.asarray(cluster.toa_indices, dtype=int)
        if indices.size < required_toas:
            continue
        cluster_epochs = epochs[indices]
        cluster_residuals = residuals[indices]
        cluster_derivatives = derivatives[indices]
        cluster_uncertainties = uncertainties[indices]
        valid = (
            np.isfinite(cluster_epochs)
            & np.isfinite(cluster_residuals)
            & np.isfinite(cluster_uncertainties)
            & (cluster_uncertainties > 0.0)
            & np.all(np.isfinite(cluster_derivatives), axis=1)
        )
        if int(np.count_nonzero(valid)) < required_toas:
            continue
        active_epochs = cluster_epochs[valid]
        active_residuals = cluster_residuals[valid]
        active_derivatives = cluster_derivatives[valid]
        active_uncertainties = cluster_uncertainties[valid]
        epoch_order = np.argsort(active_epochs, kind="stable")
        active_epochs = active_epochs[epoch_order]
        active_residuals = np.unwrap(active_residuals[epoch_order], period=1.0)
        active_derivatives = active_derivatives[epoch_order]
        active_uncertainties = active_uncertainties[epoch_order]
        span_seconds = float(np.ptp(active_epochs) * 86400.0)
        if not np.isfinite(span_seconds) or span_seconds <= 0.0:
            continue

        weights = 1.0 / np.square(active_uncertainties)
        centered_seconds = (
            active_epochs - float(np.sum(weights * active_epochs) / np.sum(weights))
        ) * 86400.0
        slope_precision = float(np.sum(weights * np.square(centered_seconds)))
        if not np.isfinite(slope_precision) or slope_precision <= 0.0:
            continue
        residual_slope = float(
            np.sum(weights * centered_seconds * active_residuals) / slope_precision
        )
        derivative_slopes = (
            np.sum(
                weights[:, None] * centered_seconds[:, None] * active_derivatives,
                axis=0,
            )
            / slope_precision
        )

        cluster_ids.append(str(cluster.cluster_id))
        cluster_mjd.append(float(np.sum(weights * active_epochs) / np.sum(weights)))
        spans.append(span_seconds)
        phase_changes.append(residual_slope * span_seconds)
        jacobian_rows.append(np.asarray(derivative_slopes * span_seconds, dtype=float))
        phase_change_sigmas.append(span_seconds / math.sqrt(slope_precision))

    parameter_count = int(derivatives.shape[1])
    return LocalPhaseSlopeEvidence(
        cluster_ids=tuple(cluster_ids),
        cluster_mjd=np.asarray(cluster_mjd, dtype=float),
        session_span_seconds=np.asarray(spans, dtype=float),
        reference_phase_change=np.asarray(phase_changes, dtype=float),
        phase_change_jacobian=(
            np.vstack(jacobian_rows)
            if jacobian_rows
            else np.zeros((0, parameter_count), dtype=float)
        ),
        phase_change_sigma=np.asarray(phase_change_sigmas, dtype=float),
    )


def _toa_phase_uncertainties(model, toas) -> np.ndarray:
    try:
        uncertainties = model.scaled_toa_uncertainty(toas)
    except (AttributeError, TypeError, ValueError):
        uncertainties = toas.get_errors()
    sigma_seconds = np.asarray(uncertainties.to_value(u.s), dtype=float)
    spin_hz = abs(float(model["F0"].value)) if "F0" in model.params else 1.0
    return np.maximum(sigma_seconds * spin_hz, 1.0e-18)


def build_pint_alias_geometry(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    parameter_names: Sequence[str],
    candidate_followup_epochs: Sequence[float] | None = None,
    clock_override_path: Path | None = None,
) -> AliasGeometryPayload:
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    raw_manifest = load_cluster_manifest(Path(cluster_manifest_path))
    manifest = resolve_cluster_manifest(raw_manifest, tim_path)
    slope_manifest = resolve_source_session_manifest(raw_manifest, tim_path)
    model = _load_pint_model(par_path)
    toas = _load_pint_toas(
        tim_path,
        seed_model=model,
        clock_override_path=clock_override_path,
    )

    phase_residuals = _extract_phase_residuals(model, toas)
    reference_gap_residuals = _compute_gap_phase_residuals(phase_residuals, manifest)
    gap_jacobian = np.zeros((len(manifest.gaps), len(parameter_names)), dtype=float)
    for column_index, parameter_name in enumerate(parameter_names):
        gap_jacobian[:, column_index] = _gap_derivative_column(
            model=model,
            toas=toas,
            manifest=manifest,
            parameter_name=parameter_name,
        )
    parameter_phase_derivatives = (
        np.column_stack(
            [
                _parameter_phase_derivative(model, toas, parameter_name)
                for parameter_name in parameter_names
            ]
        )
        if parameter_names
        else np.zeros((len(phase_residuals), 0), dtype=float)
    )
    local_phase_slope_evidence = _build_local_phase_slope_evidence(
        phase_residuals=phase_residuals,
        parameter_phase_derivatives=parameter_phase_derivatives,
        mjds=np.asarray(toas.get_mjds().to_value(u.day), dtype=float),
        phase_uncertainties=_toa_phase_uncertainties(model, toas),
        manifest=slope_manifest,
    )

    design_by_epoch = {
        float(epoch): _build_phase_design_row(
            model=model,
            toas=toas,
            parameter_names=parameter_names,
            epoch_mjd=float(epoch),
            clock_override_path=clock_override_path,
        )
        for epoch in (candidate_followup_epochs or [])
    }
    return AliasGeometryPayload(
        gap_ids=[gap.gap_id for gap in manifest.gaps],
        parameter_names=[str(name) for name in parameter_names],
        gap_jacobian=gap_jacobian,
        reference_phase_residuals=reference_gap_residuals,
        design_by_epoch=design_by_epoch,
        toa_error_phase=_estimate_toa_error_phase(model, toas),
        local_phase_slope_evidence=local_phase_slope_evidence,
    )
