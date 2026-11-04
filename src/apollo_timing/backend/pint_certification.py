"""Fixed-pulse PINT evaluation shared by the timing solvers.

Each solver defines its own acceptance and model-selection criteria.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint.residuals import Residuals

from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas


def attach_session_pulse_numbers(
    *,
    toas: object,
    base_pulse_numbers: Sequence[float],
    sessions: Sequence[object],
    relative_integer_offsets: Sequence[int],
) -> object:
    session_offsets = np.concatenate(
        (np.zeros(1, dtype=int), np.asarray(relative_integer_offsets, dtype=int))
    )
    if len(session_offsets) != len(sessions):
        raise ValueError("The integer path must contain one offset per session gap.")
    pulse_numbers = np.asarray(base_pulse_numbers, dtype=float).copy()
    for session, offset in zip(sessions, session_offsets, strict=True):
        pulse_numbers[np.asarray(session.toa_indices, dtype=int)] += int(offset)
    certified_toas = deepcopy(toas)
    certified_toas.table["pulse_number"] = pulse_numbers
    certified_toas.table["pulse_number"].unit = u.dimensionless_unscaled
    return certified_toas



def disk_reload_metrics(
    par_path: Path,
    tim_path: Path,
    *,
    clock_override_path: Path | None = None,
) -> tuple[float, float, float]:
    model = _load_pint_model(par_path)
    toas = _load_pint_toas(
        tim_path,
        seed_model=model,
        clock_override_path=clock_override_path,
    )
    residuals = Residuals(
        toas,
        model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    return (
        float(residuals.chi2),
        float(residuals.reduced_chi2),
        float(residuals.rms_weighted().to_value(u.us)),
    )



def candidate_certification_rank(record: Mapping[str, Any]) -> tuple[bool, bool, float]:
    """Rank certifications by acceptance, support membership, then chi-square."""

    chi2 = float(record.get("chi2", np.inf))
    if not np.isfinite(chi2):
        chi2 = np.inf
    return (
        not bool(record.get("accepted", False)),
        not bool(record.get("support_valid", False)),
        chi2,
    )



