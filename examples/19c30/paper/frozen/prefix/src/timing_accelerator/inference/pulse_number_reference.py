from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from pint.residuals import Residuals

from timing_accelerator.backend.pint_backend import _load_pint_model, _load_pint_toas
from timing_accelerator.inference.cluster_manifest import (
    load_cluster_manifest,
    resolve_cluster_manifest,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.inference.wrap_state import apply_wraps_to_toas


@dataclass(frozen=True)
class StablePulseNumberReference:
    path: Path
    tim_path: Path
    source_par_path: Path
    full_tim_path: Path
    tzr_mjd: str
    tzr_site: str
    tzr_freq: str
    local_island_corrections: tuple[dict[str, Any], ...]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "tim_path": str(self.tim_path),
            "source_par_path": str(self.source_par_path),
            "full_tim_path": str(self.full_tim_path),
            "tzr_mjd": self.tzr_mjd,
            "tzr_site": self.tzr_site,
            "tzr_freq": self.tzr_freq,
            "local_island_corrections": list(self.local_island_corrections),
        }


def _required_tzr_value(model, name: str) -> str:
    if name not in model.params or model[name].value is None:
        raise RuntimeError(f"PINT did not create the required {name} pulse-number reference.")
    return str(model[name].value)


def has_complete_pulse_numbers(toas) -> bool:
    if "pulse_number" not in toas.table.colnames:
        return False
    pulse_numbers = np.ma.asarray(toas.table["pulse_number"])
    if np.any(np.ma.getmaskarray(pulse_numbers)):
        return False
    try:
        values = np.asarray(pulse_numbers.data, dtype=float)
    except (TypeError, ValueError):
        return False
    return values.size == len(toas.table) and bool(np.all(np.isfinite(values)))


def ensure_pulse_numbers(toas, model):
    if has_complete_pulse_numbers(toas):
        return toas
    if not hasattr(toas, "compute_pulse_numbers"):
        raise ValueError("TOAs object cannot provide the required pulse numbers.")
    toas.compute_pulse_numbers(model)
    if not has_complete_pulse_numbers(toas):
        raise ValueError("PINT did not compute pulse numbers for every TOA.")
    return toas


def build_local_island_pulse_numbers(
    *,
    pulse_numbers: np.ndarray,
    phase_residuals: np.ndarray,
    mjds: np.ndarray,
    manifest,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    base_pulse_numbers = np.asarray(pulse_numbers, dtype=float)
    residuals = np.asarray(phase_residuals, dtype=float)
    epochs = np.asarray(mjds, dtype=float)
    if residuals.shape != base_pulse_numbers.shape or epochs.shape != base_pulse_numbers.shape:
        raise ValueError("Pulse numbers, phase residuals, and MJDs must have matching shapes.")
    if not np.all(np.isfinite(base_pulse_numbers)) or not np.all(np.isfinite(residuals)):
        raise ValueError("Local-island pulse-number inputs must be finite.")

    adjusted = np.array(base_pulse_numbers, copy=True)
    correction_metadata: list[dict[str, Any]] = []
    for cluster in manifest.clusters:
        indices = np.asarray(cluster.toa_indices, dtype=int)
        if indices.size == 0:
            raise ValueError(f"Cluster '{cluster.cluster_id}' has no resolved TOAs.")
        order = np.argsort(epochs[indices], kind="stable")
        ordered_indices = indices[order]
        local_residuals = residuals[ordered_indices]
        unwrapped = np.unwrap(local_residuals, period=1.0)
        island_center = int(np.rint(np.median(unwrapped)))
        centered_unwrapped = unwrapped - island_center
        raw_adjustments = local_residuals - centered_unwrapped
        integer_adjustments = np.rint(raw_adjustments).astype(int)
        if not np.allclose(raw_adjustments, integer_adjustments, atol=1.0e-8, rtol=0.0):
            raise RuntimeError(
                f"Cluster '{cluster.cluster_id}' local phase island did not map to integer pulses."
            )
        adjusted[ordered_indices] += integer_adjustments
        correction_metadata.append(
            {
                "cluster_id": str(cluster.cluster_id),
                "toa_count": int(indices.size),
                "island_center": island_center,
                "pulse_number_adjustments": sorted(
                    int(value) for value in np.unique(integer_adjustments)
                ),
                "adjusted_toa_count": int(np.count_nonzero(integer_adjustments)),
            }
        )
    return adjusted, tuple(correction_metadata)


def ensure_local_island_pulse_numbers(
    toas,
    model,
    manifest,
    *,
    preserve_existing: bool = True,
) -> tuple[object, tuple[dict[str, Any], ...]]:
    """Initialize pulse numbers by unwrapping each observed session locally."""
    if preserve_existing and has_complete_pulse_numbers(toas):
        return toas, tuple()
    ensure_pulse_numbers(toas, model)
    residuals = Residuals(
        toas,
        model,
        subtract_mean=False,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    adjusted, corrections = build_local_island_pulse_numbers(
        pulse_numbers=np.asarray(toas.table["pulse_number"], dtype=float),
        phase_residuals=np.asarray(
            residuals.phase_resids.to_value(u.dimensionless_unscaled),
            dtype=float,
        ),
        mjds=np.asarray(toas.get_mjds().to_value(u.day), dtype=float),
        manifest=manifest,
    )
    toas.table["pulse_number"] = adjusted
    toas.table["pulse_number"].unit = u.dimensionless_unscaled
    return toas, corrections


def prepare_stable_pulse_number_reference(
    *,
    par_path: Path,
    full_tim_path: Path,
    cluster_manifest_path: Path,
    output_path: Path,
    output_tim_path: Path,
    clock_override_path: Path | None = None,
) -> StablePulseNumberReference:
    par_path = Path(par_path)
    full_tim_path = Path(full_tim_path)
    cluster_manifest_path = Path(cluster_manifest_path)
    output_path = Path(output_path)
    output_tim_path = Path(output_tim_path)
    model = _load_pint_model(par_path)
    toas = _load_pint_toas(
        full_tim_path,
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    if not hasattr(toas, "compute_pulse_numbers"):
        raise RuntimeError("This PINT version cannot compute absolute pulse numbers.")
    toas.compute_pulse_numbers(model)
    ensure_pulse_numbers(toas, model)
    manifest = resolve_cluster_manifest(
        load_cluster_manifest(cluster_manifest_path),
        full_tim_path,
    )
    residuals = Residuals(
        toas,
        model,
        subtract_mean=False,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    adjusted_pulse_numbers, local_island_corrections = build_local_island_pulse_numbers(
        pulse_numbers=np.asarray(toas.table["pulse_number"], dtype=float),
        phase_residuals=np.asarray(
            residuals.phase_resids.to_value(u.dimensionless_unscaled),
            dtype=float,
        ),
        mjds=np.asarray(toas.get_mjds().to_value(u.day), dtype=float),
        manifest=manifest,
    )
    toas.table["pulse_number"] = adjusted_pulse_numbers
    toas.table["pulse_number"].unit = u.dimensionless_unscaled

    reference = StablePulseNumberReference(
        path=output_path,
        tim_path=output_tim_path,
        source_par_path=par_path,
        full_tim_path=full_tim_path,
        tzr_mjd=_required_tzr_value(model, "TZRMJD"),
        tzr_site=_required_tzr_value(model, "TZRSITE"),
        tzr_freq=_required_tzr_value(model, "TZRFRQ"),
        local_island_corrections=local_island_corrections,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    write_pint_model_parfile_roundtrip_safe(model=model, path=temporary_path)
    temporary_path.replace(output_path)
    output_tim_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_tim_path = output_tim_path.with_suffix(output_tim_path.suffix + ".tmp")
    toas.write_TOA_file(
        str(temporary_tim_path),
        format="tempo2",
        include_pn=True,
        include_info=True,
        comment="APOLLO local-island pulse-number reference",
    )
    temporary_tim_path.replace(output_tim_path)
    return reference


def write_branch_pulse_number_tim(
    *,
    source_tim_path: Path,
    par_path: Path,
    cluster_manifest_path: Path,
    wraps: tuple[int, ...] | list[int],
    output_tim_path: Path,
    clock_override_path: Path | None = None,
) -> Path:
    """Write the timing input bound to one APOLLO integer phase path.

    A fitted parfile contains continuous parameters only.  The selected
    session-to-session integer connection is stored in the TOA pulse numbers,
    so a certified solution is reproducible only as a par/tim pair.
    """
    source_tim_path = Path(source_tim_path)
    par_path = Path(par_path)
    cluster_manifest_path = Path(cluster_manifest_path)
    output_tim_path = Path(output_tim_path)
    model = _load_pint_model(par_path)
    toas = _load_pint_toas(
        source_tim_path,
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    manifest = resolve_cluster_manifest(
        load_cluster_manifest(cluster_manifest_path),
        source_tim_path,
    )
    # Stable reference files already contain local-island pulse numbers.  For
    # an ordinary input tim, create that reference before applying the branch.
    ensure_local_island_pulse_numbers(toas, model, manifest)
    wrapped_toas = apply_wraps_to_toas(toas, manifest, [int(item) for item in wraps])
    output_tim_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_tim_path.with_suffix(output_tim_path.suffix + ".tmp")
    wrapped_toas.write_TOA_file(
        str(temporary_path),
        format="tempo2",
        include_pn=True,
        include_info=True,
        comment="APOLLO certified integer phase path",
    )
    temporary_path.replace(output_tim_path)
    return output_tim_path
