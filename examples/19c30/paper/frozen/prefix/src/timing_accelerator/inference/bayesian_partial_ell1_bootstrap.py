from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from timing_accelerator.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.bayesian_dual_lattice_enumerator import (
    SourceSession,
    infer_contiguous_source_sessions,
    profile_fixed_local_pulse_component,
    profile_two_level_pulse_lattice,
)
from timing_accelerator.inference.bayesian_ell1_dual_lattice import (
    ELL1_FIT_PARAMETERS,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)
from timing_accelerator.io.timing_update import normalize_tempo2_tim_file


PARTIAL_ELL1_REQUIRED_PARAMETERS = (
    "RAJ",
    "DECJ",
    "F0",
    "PEPOCH",
    "PB",
)
_BINARY_PARAMETERS = {
    "BINARY",
    "PB",
    "FB0",
    "A1",
    "TASC",
    "EPS1",
    "EPS2",
}
_NON_MODEL_LINES = {"MODE", "NITS"}


@dataclass(frozen=True)
class PartialELL1BootstrapConfig:
    projected_semimajor_axis_stages_seconds: tuple[float, ...] = (
        2.5,
        5.0,
        10.0,
        20.0,
    )
    retained_lattice_modes: int = 256
    retained_exact_modes: int = 4
    exact_mode_chi2_window: float = 2_000.0
    minimum_informative_session_toas: int = 8
    maximum_eccentricity: float = 0.05
    informative_session_reduced_chi2_threshold: float = 5.0
    pair_enumeration_chunk_rows: int = 128
    full_solver_support_width_scale: float = 0.01

    def __post_init__(self) -> None:
        stages = tuple(float(value) for value in self.projected_semimajor_axis_stages_seconds)
        if not stages or any(not np.isfinite(value) or value <= 0.0 for value in stages):
            raise ValueError("A1 bootstrap stages must be finite and positive.")
        if any(right <= left for left, right in zip(stages, stages[1:], strict=False)):
            raise ValueError("A1 bootstrap stages must be strictly increasing.")
        if min(
            self.retained_lattice_modes,
            self.retained_exact_modes,
            self.minimum_informative_session_toas,
            self.pair_enumeration_chunk_rows,
        ) <= 0:
            raise ValueError("Partial ELL1 bootstrap budgets must be positive.")
        positive = (
            self.exact_mode_chi2_window,
            self.maximum_eccentricity,
            self.informative_session_reduced_chi2_threshold,
            self.full_solver_support_width_scale,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Partial ELL1 bootstrap scales must be positive.")


@dataclass(frozen=True)
class PartialELL1HarmonicMode:
    phase_u: float
    phase_v: float
    phase_sin2: float
    phase_cos2: float
    local_spin_offset_hz: float
    chi2: float
    projected_semimajor_axis_seconds: float
    amplitude_stage_seconds: float


@dataclass(frozen=True)
class PartialELL1BootstrapMode:
    mode_id: str
    par_path: Path
    informative_session_chi2: float
    all_session_shape_chi2: float
    parameter_values: dict[str, float]


@dataclass(frozen=True)
class PartialELL1BootstrapResult:
    normalized_tim_path: Path
    audit_path: Path
    modes: tuple[PartialELL1BootstrapMode, ...]
    informative_session_id: str
    informative_session_toas: int
    recommended_support_width_scale: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready summary of the generated seed portfolio."""

        return {
            "normalized_tim_path": str(self.normalized_tim_path),
            "audit_path": str(self.audit_path),
            "modes": [
                {
                    "mode_id": mode.mode_id,
                    "par_path": str(mode.par_path),
                    "informative_session_chi2": mode.informative_session_chi2,
                    "all_session_shape_chi2": mode.all_session_shape_chi2,
                    "parameter_values": mode.parameter_values,
                }
                for mode in self.modes
            ],
            "informative_session_id": self.informative_session_id,
            "informative_session_toas": self.informative_session_toas,
            "recommended_support_width_scale": (
                self.recommended_support_width_scale
            ),
        }


def _active_par_lines(path: Path) -> tuple[str, ...]:
    result = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("C "):
            continue
        result.append(stripped)
    return tuple(result)


def _par_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _active_par_lines(path):
        fields = line.split()
        if len(fields) >= 2:
            values[fields[0].upper()] = fields[1]
    return values


def is_incomplete_ell1_par(path: Path) -> bool:
    values = _par_values(Path(path))
    model_name = values.get("BINARY", "").upper().replace("_", "")
    if model_name not in {"ELL1", "ELL1H", "ELL1K"}:
        return False
    return any(name not in values for name in ("A1", "TASC", "EPS1", "EPS2"))


def _isolated_model_text(path: Path) -> str:
    retained = []
    for line in _active_par_lines(path):
        key = line.split()[0].upper()
        if key in _BINARY_PARAMETERS or key in _NON_MODEL_LINES:
            continue
        retained.append(line)
    if not any(line.split()[0].upper() == "F1" for line in retained):
        retained.append("F1 0")
    return "\n".join(retained) + "\n"


def _complete_model_text(
    path: Path,
    *,
    pb_days: float,
    a1_seconds: float,
    tasc_mjd: float,
    eps1: float = 0.0,
    eps2: float = 0.0,
) -> str:
    binary_token = _par_values(path).get("BINARY", "ELL1")
    retained = []
    for line in _active_par_lines(path):
        key = line.split()[0].upper()
        if key in _BINARY_PARAMETERS or key in _NON_MODEL_LINES:
            continue
        retained.append(line)
    if not any(line.split()[0].upper() == "F1" for line in retained):
        retained.append("F1 0")
    retained.extend(
        (
            f"BINARY {binary_token}",
            f"PB {pb_days:.18g}",
            f"A1 {a1_seconds:.18g}",
            f"TASC {tasc_mjd:.18g}",
            f"EPS1 {eps1:.18g}",
            f"EPS2 {eps2:.18g}",
        )
    )
    return "\n".join(retained) + "\n"


def _pint_model_from_text(text: str) -> object:
    from pint.models import get_model

    return get_model(StringIO(text), allow_tcb=True)


def _phase_cycles(model: object, toas: object) -> np.ndarray:
    phase = model.phase(toas, abs_phase=False)
    values = np.asarray(phase.int.value, dtype=np.longdouble)
    values += np.asarray(phase.frac.value, dtype=np.longdouble)
    return values


def _phase_sigmas(model: object, toas: object) -> np.ndarray:
    from astropy import units as u

    try:
        uncertainties = model.scaled_toa_uncertainty(toas)
    except (AttributeError, TypeError, ValueError):
        uncertainties = toas.get_errors()
    seconds = np.asarray(uncertainties.to_value(u.s), dtype=float)
    return np.maximum(seconds * abs(float(model["F0"].value)), 1.0e-18)


def _informative_session(
    sessions: Sequence[SourceSession],
    minimum_toas: int,
) -> SourceSession:
    eligible = [session for session in sessions if len(session.toa_indices) >= minimum_toas]
    if not eligible:
        raise RuntimeError(
            "Incomplete ELL1 bootstrap needs one session with at least "
            f"{minimum_toas} active TOAs."
        )
    return max(eligible, key=lambda item: len(item.toa_indices))


def _session_harmonic_data(
    *,
    model: object,
    toas: object,
    session: SourceSession,
    pb_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(session.toa_indices, dtype=int)
    mjds = np.asarray(toas.get_mjds().value, dtype=float)[indices]
    cycles = _phase_cycles(model, toas)[indices]
    differences = cycles - cycles[0]
    wrapped = np.asarray(differences - np.rint(differences), dtype=float)
    angle = 2.0 * math.pi * (mjds - mjds[0]) / float(pb_days)
    design = np.column_stack(
        (
            np.sin(angle) - np.sin(angle[0]),
            np.cos(angle) - np.cos(angle[0]),
        )
    )
    sigmas = _phase_sigmas(model, toas)[indices]
    relative_sigmas = np.sqrt(np.square(sigmas) + sigmas[0] ** 2)
    relative_sigmas[0] = sigmas[0]
    return mjds, wrapped, design, relative_sigmas


def _best_integer_basis_pair(design: np.ndarray, sigmas: np.ndarray) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for left in range(1, design.shape[0]):
        for right in range(left + 1, design.shape[0]):
            determinant = abs(float(np.linalg.det(design[[left, right]])))
            score = determinant / (float(sigmas[left]) * float(sigmas[right]))
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None or best[0] <= 0.0:
        raise RuntimeError("The informative session has no rank-two orbital phase basis.")
    return best[1], best[2]


def _retain_best(
    existing_x: list[np.ndarray],
    existing_score: list[float],
    *,
    maximum: int,
) -> tuple[list[np.ndarray], list[float]]:
    if len(existing_score) <= maximum:
        return existing_x, existing_score
    order = np.argsort(existing_score)[:maximum]
    return [existing_x[index] for index in order], [existing_score[index] for index in order]


def _enumerate_circular_lattice_modes(
    *,
    wrapped: np.ndarray,
    design: np.ndarray,
    sigmas: np.ndarray,
    phase_amplitude_bound: float,
    maximum_modes: int,
    chunk_rows: int,
) -> tuple[tuple[float, float, float], ...]:
    left, right = _best_integer_basis_pair(design, sigmas)
    pair_design = design[[left, right]]
    pair_wrapped = wrapped[[left, right]]
    inverse = np.linalg.inv(pair_design)
    bound = float(phase_amplitude_bound)
    corners = np.asarray(
        [(u, v) for u in (-bound, bound) for v in (-bound, bound)],
        dtype=float,
    )
    integer_ranges = []
    for row, observed in zip(pair_design, pair_wrapped, strict=True):
        values = float(observed) + corners @ row
        integer_ranges.append(
            np.arange(
                int(np.floor(np.min(values))) - 1,
                int(np.ceil(np.max(values))) + 2,
                dtype=int,
            )
        )

    retained_x: list[np.ndarray] = []
    retained_score: list[float] = []
    second_range = integer_ranges[1]
    for start in range(0, len(integer_ranges[0]), int(chunk_rows)):
        first = integer_ranges[0][start : start + int(chunk_rows)]
        grid_left, grid_right = np.meshgrid(first, second_range, indexing="ij")
        rhs = np.stack(
            (
                grid_left.ravel() - pair_wrapped[0],
                grid_right.ravel() - pair_wrapped[1],
            ),
            axis=1,
        )
        coordinates = rhs @ inverse.T
        coordinates = coordinates[np.linalg.norm(coordinates, axis=1) <= bound]
        if coordinates.size == 0:
            continue
        residuals = wrapped[None, :] + coordinates @ design.T
        residuals -= np.rint(residuals)
        scores = np.sum(np.square(residuals / sigmas[None, :]), axis=1)
        local_count = min(int(maximum_modes), scores.size)
        local = np.argpartition(scores, local_count - 1)[:local_count]
        retained_x.extend(coordinates[local])
        retained_score.extend(scores[local].tolist())
        retained_x, retained_score = _retain_best(
            retained_x,
            retained_score,
            maximum=8 * int(maximum_modes),
        )

    retained_x, retained_score = _retain_best(
        retained_x,
        retained_score,
        maximum=int(maximum_modes),
    )

    def wrapped_residuals(values: np.ndarray) -> np.ndarray:
        residuals = wrapped + design @ values
        residuals -= np.rint(residuals)
        return residuals / sigmas

    refined = []
    for seed in retained_x:
        result = least_squares(
            wrapped_residuals,
            seed,
            bounds=(-bound, bound),
            max_nfev=300,
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
        )
        residuals = wrapped_residuals(result.x)
        refined.append((float(residuals @ residuals), float(result.x[0]), float(result.x[1])))
    refined.sort()
    unique: list[tuple[float, float, float]] = []
    for mode in refined:
        if all(math.hypot(mode[1] - other[1], mode[2] - other[2]) > 1.0e-3 for other in unique):
            unique.append(mode)
        if len(unique) >= int(maximum_modes):
            break
    return tuple(unique)


def _refine_harmonic_modes(
    *,
    mjds: np.ndarray,
    wrapped: np.ndarray,
    circular_design: np.ndarray,
    sigmas: np.ndarray,
    circular_modes: Sequence[tuple[float, float, float]],
    pb_days: float,
    f0_hz: float,
    amplitude_stage_seconds: float,
) -> tuple[PartialELL1HarmonicMode, ...]:
    angle = 2.0 * math.pi * (mjds - mjds[0]) / float(pb_days)
    delta_seconds = (mjds - mjds[0]) * 86400.0
    full_design = np.column_stack(
        (
            circular_design,
            np.sin(2.0 * angle) - np.sin(2.0 * angle[0]),
            np.cos(2.0 * angle) - np.cos(2.0 * angle[0]),
            delta_seconds,
        )
    )
    weighted_design = full_design / sigmas[:, None]
    modes = []
    for _, phase_u, phase_v in circular_modes:
        values = np.asarray((phase_u, phase_v, 0.0, 0.0, 0.0), dtype=float)
        integers = np.rint(wrapped + full_design @ values)
        for _ in range(4):
            target = (integers - wrapped) / sigmas
            values = np.linalg.lstsq(weighted_design, target, rcond=None)[0]
            updated = np.rint(wrapped + full_design @ values)
            if np.array_equal(updated, integers):
                break
            integers = updated
        residuals = (wrapped + full_design @ values - integers) / sigmas
        amplitude = math.hypot(float(values[0]), float(values[1]))
        modes.append(
            PartialELL1HarmonicMode(
                phase_u=float(values[0]),
                phase_v=float(values[1]),
                phase_sin2=float(values[2]),
                phase_cos2=float(values[3]),
                local_spin_offset_hz=float(values[4]),
                chi2=float(residuals @ residuals),
                projected_semimajor_axis_seconds=amplitude / abs(float(f0_hz)),
                amplitude_stage_seconds=float(amplitude_stage_seconds),
            )
        )
    modes.sort(key=lambda item: item.chi2)
    unique: list[PartialELL1HarmonicMode] = []
    for mode in modes:
        if all(
            math.hypot(mode.phase_u - other.phase_u, mode.phase_v - other.phase_v)
            > 1.0e-3
            for other in unique
        ):
            unique.append(mode)
    return tuple(unique)


def _discover_harmonic_modes(
    *,
    model: object,
    toas: object,
    session: SourceSession,
    pb_days: float,
    config: PartialELL1BootstrapConfig,
) -> tuple[PartialELL1HarmonicMode, ...]:
    mjds, wrapped, design, sigmas = _session_harmonic_data(
        model=model,
        toas=toas,
        session=session,
        pb_days=pb_days,
    )
    f0 = abs(float(model["F0"].value))
    best_modes: tuple[PartialELL1HarmonicMode, ...] = ()
    for stage_seconds in config.projected_semimajor_axis_stages_seconds:
        circular = _enumerate_circular_lattice_modes(
            wrapped=wrapped,
            design=design,
            sigmas=sigmas,
            phase_amplitude_bound=f0 * float(stage_seconds),
            maximum_modes=config.retained_lattice_modes,
            chunk_rows=config.pair_enumeration_chunk_rows,
        )
        harmonic = _refine_harmonic_modes(
            mjds=mjds,
            wrapped=wrapped,
            circular_design=design,
            sigmas=sigmas,
            circular_modes=circular,
            pb_days=pb_days,
            f0_hz=f0,
            amplitude_stage_seconds=stage_seconds,
        )
        if not harmonic:
            continue
        best_modes = harmonic
        degrees = max(len(session.toa_indices) - 5, 1)
        best = harmonic[0]
        if (
            best.chi2 <= config.informative_session_reduced_chi2_threshold * degrees
            and best.projected_semimajor_axis_seconds < 0.9 * float(stage_seconds)
        ):
            break
    if not best_modes:
        raise RuntimeError("No incomplete-ELL1 harmonic posterior mode was found.")
    minimum = best_modes[0].chi2
    return tuple(
        mode
        for mode in best_modes
        if mode.chi2 <= minimum + config.exact_mode_chi2_window
    )[: config.retained_exact_modes]


def _fit_unit_orbit_coordinates(
    *,
    binary_model: object,
    isolated_cycles: np.ndarray,
    toas: object,
    session_indices: np.ndarray,
    harmonic_design: np.ndarray,
    tasc_mjd: float,
) -> np.ndarray:
    _set_numeric_model_parameter_value(binary_model, "TASC", float(tasc_mjd))
    correction = _phase_cycles(binary_model, toas) - isolated_cycles
    relative = np.asarray(
        correction[session_indices] - correction[session_indices[0]],
        dtype=float,
    )
    return np.linalg.lstsq(harmonic_design, relative, rcond=None)[0][:2]


def _map_harmonic_mode_to_pint(
    *,
    partial_par_path: Path,
    isolated_model: object,
    isolated_cycles: np.ndarray,
    toas: object,
    session: SourceSession,
    pb_days: float,
    mode: PartialELL1HarmonicMode,
    config: PartialELL1BootstrapConfig,
) -> tuple[object, float]:
    reference_epoch = float(isolated_model["PEPOCH"].value)
    binary_model = _pint_model_from_text(
        _complete_model_text(
            partial_par_path,
            pb_days=pb_days,
            a1_seconds=1.0,
            tasc_mjd=reference_epoch,
        )
    )
    indices = np.asarray(session.toa_indices, dtype=int)
    mjds = np.asarray(toas.get_mjds().value, dtype=float)[indices]
    angle = 2.0 * math.pi * (mjds - mjds[0]) / float(pb_days)
    delta_seconds = (mjds - mjds[0]) * 86400.0
    design = np.column_stack(
        (
            np.sin(angle) - np.sin(angle[0]),
            np.cos(angle) - np.cos(angle[0]),
            delta_seconds,
        )
    )
    target = np.asarray((mode.phase_u, mode.phase_v), dtype=float)

    def unit_coordinates(phase_offset: float) -> np.ndarray:
        return _fit_unit_orbit_coordinates(
            binary_model=binary_model,
            isolated_cycles=isolated_cycles,
            toas=toas,
            session_indices=indices,
            harmonic_design=design,
            tasc_mjd=reference_epoch + float(phase_offset) * pb_days,
        )

    def mismatch(phase_offset: float) -> float:
        coordinates = unit_coordinates(phase_offset)
        scale = float(target @ coordinates / (coordinates @ coordinates))
        return float(np.sum(np.square(target - scale * coordinates)))

    grid = np.linspace(-0.5, 0.5, 65, endpoint=False)
    scores = np.asarray([mismatch(value) for value in grid], dtype=float)
    center = float(grid[int(np.argmin(scores))])
    half_width = 1.5 / len(grid)
    optimum = minimize_scalar(
        mismatch,
        bounds=(center - half_width, center + half_width),
        method="bounded",
        options={"xatol": 1.0e-13, "maxiter": 300},
    )
    phase_offset = float(optimum.x)
    tasc = reference_epoch + phase_offset * pb_days
    unit = unit_coordinates(phase_offset)
    a1 = float(target @ unit / (unit @ unit))
    if a1 < 0.0:
        tasc += 0.5 * pb_days
        a1 = -a1
    _set_numeric_model_parameter_value(binary_model, "A1", a1)
    _set_numeric_model_parameter_value(binary_model, "TASC", tasc)
    _set_numeric_model_parameter_value(
        binary_model,
        "F0",
        float(isolated_model["F0"].value) + float(mode.local_spin_offset_hz),
    )

    initial_cycles = _phase_cycles(binary_model, toas)
    relative = np.asarray(
        initial_cycles[indices] - initial_cycles[indices[0]],
        dtype=float,
    )
    integer_path = np.rint(relative)
    sigmas = _phase_sigmas(binary_model, toas)[indices]
    relative_sigmas = np.sqrt(np.square(sigmas) + sigmas[0] ** 2)
    relative_sigmas[0] = sigmas[0]
    start = np.asarray((a1, 0.0, 0.0, 0.0), dtype=float)
    lower = np.asarray((1.0e-6, -0.2, -config.maximum_eccentricity, -config.maximum_eccentricity))
    upper = np.asarray(
        (
            config.projected_semimajor_axis_stages_seconds[-1],
            0.2,
            config.maximum_eccentricity,
            config.maximum_eccentricity,
        )
    )

    def residuals(values: np.ndarray) -> np.ndarray:
        physical = (
            float(values[0]),
            tasc + float(values[1]) * pb_days / (2.0 * math.pi),
            float(values[2]),
            float(values[3]),
        )
        for name, value in zip(("A1", "TASC", "EPS1", "EPS2"), physical, strict=True):
            _set_numeric_model_parameter_value(binary_model, name, value)
        cycles = _phase_cycles(binary_model, toas)
        local = np.asarray(cycles[indices] - cycles[indices[0]], dtype=float)
        return (local - integer_path) / relative_sigmas

    fit = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        x_scale=np.asarray((1.0, 0.01, 1.0e-3, 1.0e-3)),
        max_nfev=2_000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    final_residuals = residuals(fit.x)
    return binary_model, float(final_residuals @ final_residuals)


def _refine_all_session_shape(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
) -> float:
    names = ("PB", "TASC", "A1", "EPS1", "EPS2", "F0", "F1", "RAJ", "DECJ")
    base = np.asarray([float(model[name].value) for name in names], dtype=float)
    scales = np.asarray(
        (1.0e-7, 1.0e-5, 1.0e-3, 1.0e-5, 1.0e-5, 1.0e-8, 1.0e-14, 1.0 / 15.0 / 3600.0, 1.0 / 3600.0),
        dtype=float,
    )
    sigmas = _phase_sigmas(model, toas)
    initial = profile_two_level_pulse_lattice(
        np.asarray(_phase_cycles(model, toas), dtype=float),
        sigmas,
        sessions,
        integer_radius=0,
    )
    component = initial.local_integer_offsets

    def residuals(values: np.ndarray) -> np.ndarray:
        physical = base + np.asarray(values, dtype=float) * scales
        for name, value in zip(names, physical, strict=True):
            _set_numeric_model_parameter_value(model, name, float(value))
        profile = profile_fixed_local_pulse_component(
            np.asarray(_phase_cycles(model, toas), dtype=float),
            sigmas,
            sessions,
            component,
        )
        return profile.normalized_local_residuals

    lower = np.asarray((-1000, -300, -200, -500, -500, -1000, -10000, -300, -300), dtype=float)
    upper = -lower
    fit = least_squares(
        residuals,
        np.zeros(len(names), dtype=float),
        bounds=(lower, upper),
        x_scale="jac",
        diff_step=1.0e-3,
        max_nfev=3_000,
        xtol=1.0e-13,
        ftol=1.0e-13,
        gtol=1.0e-13,
    )
    final = residuals(fit.x)
    decoded = profile_two_level_pulse_lattice(
        np.asarray(_phase_cycles(model, toas), dtype=float),
        sigmas,
        sessions,
        integer_radius=0,
    )
    if decoded.local_integer_offsets != component:
        raise RuntimeError("All-session ELL1 shape refinement changed its local integer component.")
    return float(final @ final)


def _write_bootstrap_audit(path: Path, payload: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _write_or_validate_bootstrap_mode(
    *,
    model: object,
    path: Path,
    resume: bool,
) -> None:
    if not resume or not path.is_file():
        write_pint_model_parfile_roundtrip_safe(model=model, path=path)
        return
    existing = _load_pint_model(path)
    if set(existing.free_params) != set(ELL1_FIT_PARAMETERS):
        raise RuntimeError(
            f"Bootstrap resume fit flags changed in {path}; use a new output directory."
        )
    for name in ELL1_FIT_PARAMETERS:
        expected = float(model[name].value)
        retained = float(existing[name].value)
        if not np.isclose(retained, expected, rtol=1.0e-11, atol=1.0e-18):
            raise RuntimeError(
                f"Bootstrap resume changed {name} in {path}; use a new output directory."
            )


def bootstrap_partial_ell1(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: PartialELL1BootstrapConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> PartialELL1BootstrapResult:
    """Infer missing ELL1 coordinates before the complete APOLLO lattice search."""

    active = config or PartialELL1BootstrapConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    if not is_incomplete_ell1_par(par):
        raise ValueError("Partial ELL1 bootstrap requires an incomplete BINARY ELL1 par.")
    values = _par_values(par)
    missing = [name for name in PARTIAL_ELL1_REQUIRED_PARAMETERS if name not in values]
    if missing:
        raise ValueError("Partial ELL1 par is missing: " + ", ".join(missing))
    pb_days = float(values["PB"])
    if not np.isfinite(pb_days) or pb_days <= 0.0:
        raise ValueError("Partial ELL1 bootstrap requires a positive PB.")

    normalized_tim = normalize_tempo2_tim_file(
        input_path=tim,
        output_path=output / "input.normalized.tim",
    )
    isolated_model = _pint_model_from_text(_isolated_model_text(par))
    toas = _load_pint_toas(
        normalized_tim,
        seed_model=isolated_model,
        clock_override_path=clock_override_path,
    )
    sessions = infer_contiguous_source_sessions(toas)
    informative = _informative_session(sessions, active.minimum_informative_session_toas)
    isolated_cycles = _phase_cycles(isolated_model, toas)
    harmonic_modes = _discover_harmonic_modes(
        model=isolated_model,
        toas=toas,
        session=informative,
        pb_days=pb_days,
        config=active,
    )

    exact: list[tuple[float, object, PartialELL1HarmonicMode]] = []
    for harmonic_mode in harmonic_modes:
        model, local_chi2 = _map_harmonic_mode_to_pint(
            partial_par_path=par,
            isolated_model=isolated_model,
            isolated_cycles=isolated_cycles,
            toas=toas,
            session=informative,
            pb_days=pb_days,
            mode=harmonic_mode,
            config=active,
        )
        exact.append((local_chi2, deepcopy(model), harmonic_mode))
    exact.sort(key=lambda item: item[0])
    best_exact = exact[0][0]
    exact = [
        item for item in exact if item[0] <= best_exact + active.exact_mode_chi2_window
    ][: active.retained_exact_modes]

    output_modes: list[PartialELL1BootstrapMode] = []
    audit_modes = []
    for index, (local_chi2, model, harmonic_mode) in enumerate(exact):
        shape_chi2 = _refine_all_session_shape(model=model, toas=toas, sessions=sessions)
        _freeze_all_parameters(model)
        for name in ELL1_FIT_PARAMETERS:
            model[name].frozen = False
        mode_id = f"mode_{index:04d}"
        mode_path = output / "modes" / f"{mode_id}.par"
        _write_or_validate_bootstrap_mode(
            model=model,
            path=mode_path,
            resume=resume,
        )
        parameters = {name: float(model[name].value) for name in ELL1_FIT_PARAMETERS}
        output_modes.append(
            PartialELL1BootstrapMode(
                mode_id=mode_id,
                par_path=mode_path,
                informative_session_chi2=float(local_chi2),
                all_session_shape_chi2=float(shape_chi2),
                parameter_values=parameters,
            )
        )
        audit_modes.append(
            {
                "mode_id": mode_id,
                "par_path": str(mode_path),
                "harmonic": {
                    "phase_u": harmonic_mode.phase_u,
                    "phase_v": harmonic_mode.phase_v,
                    "phase_sin2": harmonic_mode.phase_sin2,
                    "phase_cos2": harmonic_mode.phase_cos2,
                    "local_spin_offset_hz": harmonic_mode.local_spin_offset_hz,
                    "chi2": harmonic_mode.chi2,
                    "projected_semimajor_axis_seconds": harmonic_mode.projected_semimajor_axis_seconds,
                    "amplitude_stage_seconds": harmonic_mode.amplitude_stage_seconds,
                },
                "informative_session_chi2": local_chi2,
                "all_session_shape_chi2": shape_chi2,
                "parameter_values": parameters,
            }
        )
    output_modes.sort(key=lambda item: (item.all_session_shape_chi2, item.informative_session_chi2))
    if not output_modes:
        raise RuntimeError("No exact partial ELL1 bootstrap mode survived.")

    audit_path = _write_bootstrap_audit(
        output / "audit.json",
        {
            "schema_version": 1,
            "architecture": "apollo_bayesian_partial_ell1_harmonic_integer_bootstrap",
            "inputs": {"par": str(par), "tim": str(tim)},
            "normalized_tim": str(normalized_tim),
            "session_count": len(sessions),
            "informative_session": {
                "session_id": informative.session_id,
                "toa_count": len(informative.toa_indices),
                "epoch_mjd": informative.epoch_mjd,
            },
            "scientific_contract": {
                "historical_solution_seed": "forbidden",
                "source_class_prior": "forbidden",
                "manual_integer_wrap": "forbidden",
                "automatic_high_order_parameters": "forbidden",
                "fit_parameters": ELL1_FIT_PARAMETERS,
            },
            "config": {
                "projected_semimajor_axis_stages_seconds": active.projected_semimajor_axis_stages_seconds,
                "retained_lattice_modes": active.retained_lattice_modes,
                "retained_exact_modes": active.retained_exact_modes,
                "exact_mode_chi2_window": active.exact_mode_chi2_window,
                "full_solver_support_width_scale": active.full_solver_support_width_scale,
            },
            "modes": audit_modes,
        },
    )
    return PartialELL1BootstrapResult(
        normalized_tim_path=normalized_tim,
        audit_path=audit_path,
        modes=tuple(output_modes),
        informative_session_id=informative.session_id,
        informative_session_toas=len(informative.toa_indices),
        recommended_support_width_scale=active.full_solver_support_width_scale,
    )


def bootstrap_coarse_ell1(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: PartialELL1BootstrapConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> PartialELL1BootstrapResult:
    """Build complete ELL1 seeds from coarse position, spin, and PB inputs.

    The input par must provide ``RAJ``, ``DECJ``, ``F0``, ``PEPOCH``, and
    ``PB`` while omitting at least one of ``A1``, ``TASC``, ``EPS1``, and
    ``EPS2``. ``F1`` may be absent and is then inferred from a zero-valued
    seed. The bootstrap infers the missing orbital coordinates without a
    historical timing solution, source-class prior, or manual pulse wraps.
    """

    return bootstrap_partial_ell1(
        par_path=par_path,
        tim_path=tim_path,
        output_dir=output_dir,
        config=config,
        clock_override_path=clock_override_path,
        resume=resume,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = PartialELL1BootstrapConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Infer complete ELL1 seed modes from a coarse position, F0, "
            "optional F1, and PB model."
        )
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--a1-stage",
        dest="a1_stages",
        action="append",
        type=float,
        help=(
            "Projected semimajor-axis bound in light-seconds; repeat for a "
            "strictly increasing staged search."
        ),
    )
    parser.add_argument(
        "--retained-lattice-modes",
        type=int,
        default=defaults.retained_lattice_modes,
    )
    parser.add_argument(
        "--retained-exact-modes",
        type=int,
        default=defaults.retained_exact_modes,
    )
    parser.add_argument(
        "--exact-mode-chi2-window",
        type=float,
        default=defaults.exact_mode_chi2_window,
    )
    parser.add_argument(
        "--minimum-informative-session-toas",
        type=int,
        default=defaults.minimum_informative_session_toas,
    )
    parser.add_argument(
        "--maximum-eccentricity",
        type=float,
        default=defaults.maximum_eccentricity,
    )
    parser.add_argument(
        "--support-width-scale",
        type=float,
        default=defaults.full_solver_support_width_scale,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    defaults = PartialELL1BootstrapConfig()
    config = PartialELL1BootstrapConfig(
        projected_semimajor_axis_stages_seconds=(
            tuple(args.a1_stages)
            if args.a1_stages is not None
            else defaults.projected_semimajor_axis_stages_seconds
        ),
        retained_lattice_modes=args.retained_lattice_modes,
        retained_exact_modes=args.retained_exact_modes,
        exact_mode_chi2_window=args.exact_mode_chi2_window,
        minimum_informative_session_toas=(
            args.minimum_informative_session_toas
        ),
        maximum_eccentricity=args.maximum_eccentricity,
        informative_session_reduced_chi2_threshold=(
            defaults.informative_session_reduced_chi2_threshold
        ),
        pair_enumeration_chunk_rows=defaults.pair_enumeration_chunk_rows,
        full_solver_support_width_scale=args.support_width_scale,
    )
    result = bootstrap_coarse_ell1(
        par_path=args.par,
        tim_path=args.tim,
        output_dir=args.output,
        config=config,
        clock_override_path=args.clock_override,
        resume=args.resume,
    )
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PARTIAL_ELL1_REQUIRED_PARAMETERS",
    "PartialELL1BootstrapConfig",
    "PartialELL1BootstrapMode",
    "PartialELL1BootstrapResult",
    "PartialELL1HarmonicMode",
    "bootstrap_coarse_ell1",
    "bootstrap_partial_ell1",
    "build_arg_parser",
    "is_incomplete_ell1_par",
    "main",
    "normalize_tempo2_tim_file",
]
