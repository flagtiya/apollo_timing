"""Multi-session ELL1 seed discovery using APOLLO's common lattice and refiner."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from apollo_timing.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
    SourceSession,
    infer_contiguous_source_sessions,
    profile_two_level_pulse_lattice,
)
from apollo_timing.inference.bayesian_ell1_dual_lattice import ELL1_FIT_PARAMETERS
from apollo_timing.inference.bayesian_partial_ell1_bootstrap import (
    PARTIAL_ELL1_REQUIRED_PARAMETERS,
    PartialELL1BootstrapConfig,
    PartialELL1BootstrapMode,
    PartialELL1BootstrapResult,
    PartialELL1HarmonicMode,
    _complete_model_text,
    _enumerate_circular_lattice_modes,
    _informative_session,
    _isolated_model_text,
    _par_values,
    _phase_cycles,
    _phase_sigmas,
    _pint_model_from_text,
    _write_bootstrap_audit,
    _write_or_validate_bootstrap_mode,
    is_incomplete_ell1_par,
)
from apollo_timing.inference.session_shape_refinement import (
    LocalIntegerComponentChangedError,
    refine_fixed_session_shape,
    relative_session_phases,
)
from apollo_timing.io.foldbank_artifacts import write_alias_par
from apollo_timing.io.timing_update import normalize_tempo2_tim_file
from apollo_timing.models.foldbank import (
    BootstrapEntry,
    BootstrapPortfolio,
    OrbitalPeriodAlias,
)


@dataclass(frozen=True)
class FoldbankBootstrapConfig(PartialELL1BootstrapConfig):
    maximum_spin_frequency_offset_hz: float = 1.0e-5
    orbital_period_min_days: float | None = None
    orbital_period_max_days: float | None = None
    refine_astrometry: bool = False
    refine_spin_frequency: bool = False
    refine_spin_down: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in (
            "retained_lattice_modes",
            "retained_exact_modes",
            "minimum_informative_session_toas",
            "pair_enumeration_chunk_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Foldbank bootstrap budgets must be integers.")
            object.__setattr__(self, name, int(value))
        object.__setattr__(
            self,
            "projected_semimajor_axis_stages_seconds",
            tuple(float(value) for value in self.projected_semimajor_axis_stages_seconds),
        )
        if (
            not np.isfinite(self.maximum_spin_frequency_offset_hz)
            or self.maximum_spin_frequency_offset_hz <= 0.0
        ):
            raise ValueError("The foldbank F0 offset bound must be finite and positive.")
        lower, upper = self.orbital_period_min_days, self.orbital_period_max_days
        if (lower is None) != (upper is None):
            raise ValueError("Both orbital-period bounds must be supplied together.")
        if lower is not None:
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
                raise ValueError("Orbital-period bounds must be finite and positive.")
            if upper <= lower:
                raise ValueError("The upper orbital-period bound must exceed the lower bound.")
        if self.refine_astrometry or self.refine_spin_down:
            raise ValueError("Foldbank does not refine astrometry or spin-down.")


@dataclass(frozen=True)
class FoldbankTrainingData:
    isolated_model: object
    toas: object
    sessions: tuple[SourceSession, ...]
    normalized_tim_path: Path


def prepare_foldbank_training(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    clock_override_path: Path | None,
) -> FoldbankTrainingData:
    values = _par_values(par_path)
    missing = [name for name in PARTIAL_ELL1_REQUIRED_PARAMETERS if name not in values]
    if values.get("BINARY", "").upper() != "ELL1" or missing:
        raise ValueError(
            "Foldbank needs BINARY ELL1 with " + ", ".join(PARTIAL_ELL1_REQUIRED_PARAMETERS)
        )
    normalized = normalize_tempo2_tim_file(
        input_path=tim_path,
        output_path=output_dir / "input.normalized.tim",
    )
    isolated_model = _pint_model_from_text(_isolated_model_text(par_path))
    toas = _load_pint_toas(
        normalized,
        seed_model=isolated_model,
        clock_override_path=clock_override_path,
    )
    sessions = infer_contiguous_source_sessions(toas)
    if len(sessions) < 2:
        raise ValueError("Foldbank discovery needs at least two independent sessions.")
    return FoldbankTrainingData(isolated_model, toas, sessions, normalized)


def _joint_session_harmonic_data(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
    pb_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build within-session phase equations with shared orbital coordinates."""

    all_mjds = np.asarray(toas.get_mjds().value, dtype=float)
    all_cycles = _phase_cycles(model, toas)
    all_sigmas = _phase_sigmas(model, toas)
    reference_epoch = float(model["PEPOCH"].value)
    wrapped_blocks = []
    circular_blocks = []
    elapsed_blocks = []
    sigma_blocks = []
    for session in sessions:
        indices = np.asarray(session.toa_indices, dtype=int)
        mjds = all_mjds[indices]
        differences = all_cycles[indices] - all_cycles[indices[0]]
        wrapped = np.asarray(differences - np.rint(differences), dtype=float)
        angle = 2.0 * math.pi * (mjds - reference_epoch) / float(pb_days)
        circular = np.column_stack(
            (
                np.sin(angle) - np.sin(angle[0]),
                np.cos(angle) - np.cos(angle[0]),
            )
        )
        sigmas = np.sqrt(np.square(all_sigmas[indices]) + all_sigmas[indices[0]] ** 2)
        sigmas[0] = all_sigmas[indices[0]]
        wrapped_blocks.append(wrapped)
        circular_blocks.append(circular)
        elapsed_blocks.append((mjds - mjds[0]) * 86400.0)
        sigma_blocks.append(sigmas)
    return (
        np.concatenate(wrapped_blocks),
        np.vstack(circular_blocks),
        np.concatenate(elapsed_blocks),
        np.concatenate(sigma_blocks),
    )


def _discover_joint_harmonic_modes(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
    pb_days: float,
    config: FoldbankBootstrapConfig,
) -> tuple[PartialELL1HarmonicMode, ...]:
    """Enumerate orbital modes from multiple short sessions jointly."""

    wrapped, design, _, sigmas = _joint_session_harmonic_data(
        model=model,
        toas=toas,
        sessions=sessions,
        pb_days=pb_days,
    )
    f0 = abs(float(model["F0"].value))
    modes: list[PartialELL1HarmonicMode] = []
    for stage_seconds in config.projected_semimajor_axis_stages_seconds:
        circular = _enumerate_circular_lattice_modes(
            wrapped=wrapped,
            design=design,
            sigmas=sigmas,
            phase_amplitude_bound=f0 * float(stage_seconds),
            maximum_modes=config.retained_lattice_modes,
            chunk_rows=config.pair_enumeration_chunk_rows,
        )
        for chi2, phase_u, phase_v in circular:
            candidate = PartialELL1HarmonicMode(
                phase_u=float(phase_u),
                phase_v=float(phase_v),
                phase_sin2=0.0,
                phase_cos2=0.0,
                local_spin_offset_hz=0.0,
                chi2=float(chi2),
                projected_semimajor_axis_seconds=(math.hypot(float(phase_u), float(phase_v)) / f0),
                amplitude_stage_seconds=float(stage_seconds),
            )
            if all(
                math.hypot(
                    candidate.phase_u - existing.phase_u,
                    candidate.phase_v - existing.phase_v,
                )
                > 1.0e-3
                for existing in modes
            ):
                modes.append(candidate)
    if not modes:
        raise RuntimeError("No joint-session incomplete-ELL1 mode was found.")
    modes.sort(key=lambda item: item.chi2)
    minimum = modes[0].chi2
    return tuple(mode for mode in modes if mode.chi2 <= minimum + config.exact_mode_chi2_window)[
        : config.retained_exact_modes
    ]


def _map_joint_harmonic_mode_to_pint(
    *,
    partial_par_path: Path,
    isolated_model: object,
    isolated_cycles: np.ndarray,
    toas: object,
    sessions: Sequence[SourceSession],
    pb_days: float,
    mode: PartialELL1HarmonicMode,
    config: FoldbankBootstrapConfig,
) -> tuple[object, float]:
    """Map a shared multi-session harmonic mode to an exact PINT ELL1 model."""

    reference_epoch = float(isolated_model["PEPOCH"].value)
    binary_model = _pint_model_from_text(
        _complete_model_text(
            partial_par_path,
            pb_days=pb_days,
            a1_seconds=1.0,
            tasc_mjd=reference_epoch,
        )
    )
    _, circular_design, elapsed_seconds, _ = _joint_session_harmonic_data(
        model=isolated_model,
        toas=toas,
        sessions=sessions,
        pb_days=pb_days,
    )
    mapping_design = np.column_stack((circular_design, elapsed_seconds))
    target = np.asarray((mode.phase_u, mode.phase_v), dtype=float)

    def relative_blocks(cycles: np.ndarray) -> np.ndarray:
        blocks = []
        for session in sessions:
            indices = np.asarray(session.toa_indices, dtype=int)
            blocks.append(np.asarray(cycles[indices] - cycles[indices[0]], dtype=float))
        return np.concatenate(blocks)

    def unit_coordinates(phase_offset: float) -> np.ndarray:
        _set_numeric_model_parameter_value(
            binary_model,
            "TASC",
            reference_epoch + float(phase_offset) * pb_days,
        )
        correction = _phase_cycles(binary_model, toas) - isolated_cycles
        relative = relative_blocks(correction)
        return np.linalg.lstsq(mapping_design, relative, rcond=None)[0][:2]

    def mismatch(phase_offset: float) -> float:
        coordinates = unit_coordinates(phase_offset)
        denominator = float(coordinates @ coordinates)
        if denominator <= 0.0:
            return math.inf
        scale = float(target @ coordinates / denominator)
        return float(np.sum(np.square(target - scale * coordinates)))

    grid = np.linspace(-0.5, 0.5, 129, endpoint=False)
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

    initial_cycles = _phase_cycles(binary_model, toas)
    integer_paths = []
    sigma_blocks = []
    all_sigmas = _phase_sigmas(binary_model, toas)
    for session in sessions:
        indices = np.asarray(session.toa_indices, dtype=int)
        relative = np.asarray(initial_cycles[indices] - initial_cycles[indices[0]], dtype=float)
        integer_paths.append(np.rint(relative))
        sigmas = np.sqrt(np.square(all_sigmas[indices]) + all_sigmas[indices[0]] ** 2)
        sigmas[0] = all_sigmas[indices[0]]
        sigma_blocks.append(sigmas)

    start_values = [a1, 0.0, 0.0, 0.0]
    lower_values = [
        1.0e-6,
        -0.2,
        -config.maximum_eccentricity,
        -config.maximum_eccentricity,
    ]
    upper_values = [
        config.projected_semimajor_axis_stages_seconds[-1],
        0.2,
        config.maximum_eccentricity,
        config.maximum_eccentricity,
    ]
    scale_values = [1.0, 0.01, 1.0e-3, 1.0e-3]
    if config.refine_spin_frequency:
        start_values.append(0.0)
        lower_values.append(-config.maximum_spin_frequency_offset_hz)
        upper_values.append(config.maximum_spin_frequency_offset_hz)
        scale_values.append(1.0e-8)
    start = np.asarray(start_values, dtype=float)
    lower = np.asarray(lower_values, dtype=float)
    upper = np.asarray(upper_values, dtype=float)

    def residuals(values: np.ndarray) -> np.ndarray:
        physical = (
            float(values[0]),
            tasc + float(values[1]) * pb_days / (2.0 * math.pi),
            float(values[2]),
            float(values[3]),
        )
        for name, value in zip(("A1", "TASC", "EPS1", "EPS2"), physical, strict=True):
            _set_numeric_model_parameter_value(binary_model, name, value)
        if config.refine_spin_frequency:
            _set_numeric_model_parameter_value(
                binary_model,
                "F0",
                float(isolated_model["F0"].value) + float(values[4]),
            )
        cycles = _phase_cycles(binary_model, toas)
        blocks = []
        for session, integer_path, sigmas in zip(
            sessions, integer_paths, sigma_blocks, strict=True
        ):
            indices = np.asarray(session.toa_indices, dtype=int)
            relative = np.asarray(cycles[indices] - cycles[indices[0]], dtype=float)
            blocks.append((relative - integer_path) / sigmas)
        return np.concatenate(blocks)

    fit = least_squares(
        residuals,
        np.minimum(np.maximum(start, lower + 1.0e-12), upper - 1.0e-12),
        bounds=(lower, upper),
        x_scale=np.asarray(scale_values, dtype=float),
        max_nfev=3_000,
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
    orbital_period_bounds_days: tuple[float, float] | None = None,
    maximum_eccentricity: float = 0.05,
    maximum_spin_frequency_offset_hz: float = 1.0e-5,
    spin_frequency_reference_hz: float | None = None,
    maximum_a1_seconds: float = 20.0,
    refine_spin_frequency: bool = False,
) -> float:
    names = ["PB", "TASC", "A1", "EPS1", "EPS2"]
    if orbital_period_bounds_days is None:
        names.remove("PB")
    if refine_spin_frequency:
        names.append("F0")
    names = tuple(names)
    base = np.asarray([float(model[name].value) for name in names], dtype=float)
    scale_by_name = {
        "PB": 1.0e-7,
        "TASC": 1.0e-5,
        "A1": 1.0e-3,
        "EPS1": 1.0e-5,
        "EPS2": 1.0e-5,
        "F0": 1.0e-8,
    }
    bound_by_name = {
        "PB": 1000.0,
        "TASC": 300.0,
        "A1": 200.0,
        "EPS1": 500.0,
        "EPS2": 500.0,
        "F0": 1000.0,
    }
    scales = np.asarray([scale_by_name[name] for name in names], dtype=float)
    upper = np.asarray([bound_by_name[name] for name in names], dtype=float)
    lower = -upper
    if "F0" in names:
        index = names.index("F0")
        reference = (
            base[index] if spin_frequency_reference_hz is None else spin_frequency_reference_hz
        )
        lower[index] = (reference - maximum_spin_frequency_offset_hz - base[index]) / scales[index]
        upper[index] = (reference + maximum_spin_frequency_offset_hz - base[index]) / scales[index]
    a1_index = names.index("A1")
    lower[a1_index] = (1.0e-6 - base[a1_index]) / scales[a1_index]
    upper[a1_index] = (maximum_a1_seconds - base[a1_index]) / scales[a1_index]
    for name in ("EPS1", "EPS2"):
        index = names.index(name)
        lower[index] = (-float(maximum_eccentricity) - base[index]) / scales[index]
        upper[index] = (float(maximum_eccentricity) - base[index]) / scales[index]
    if orbital_period_bounds_days is not None:
        pb_lower, pb_upper = (float(value) for value in orbital_period_bounds_days)
        if not pb_lower <= base[0] <= pb_upper:
            raise ValueError("The bootstrap PB seed is outside its orbital-alias cell.")
        pb_index = names.index("PB")
        scales[pb_index] = 0.5 * (pb_upper - pb_lower)
        lower[pb_index] = (pb_lower - base[pb_index]) / scales[pb_index]
        upper[pb_index] = (pb_upper - base[pb_index]) / scales[pb_index]
    result = refine_fixed_session_shape(
        model=model,
        phase_values=lambda: relative_session_phases(_phase_cycles(model, toas), sessions),
        phase_sigmas=_phase_sigmas(model, toas),
        sessions=sessions,
        parameter_names=names,
        scales=scales,
        lower=lower,
        upper=upper,
    )
    return result.chi2


def bootstrap_foldbank_ell1(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: FoldbankBootstrapConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
    training_data: FoldbankTrainingData | None = None,
) -> PartialELL1BootstrapResult:
    """Infer folding-only candidates from shared session shapes."""

    active = config or FoldbankBootstrapConfig()
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
    orbital_period_bounds = None
    if active.orbital_period_min_days is not None:
        orbital_period_bounds = (
            float(active.orbital_period_min_days),
            float(active.orbital_period_max_days),
        )
        if not orbital_period_bounds[0] <= pb_days <= orbital_period_bounds[1]:
            raise ValueError("The input PB is outside the configured orbital-alias cell.")

    training = training_data or prepare_foldbank_training(
        par_path=par,
        tim_path=tim,
        output_dir=output,
        clock_override_path=clock_override_path,
    )
    normalized_tim = training.normalized_tim_path
    isolated_model, toas, sessions = training.isolated_model, training.toas, training.sessions
    informative = _informative_session(
        sessions,
        active.minimum_informative_session_toas,
    )
    isolated_cycles = _phase_cycles(isolated_model, toas)
    harmonic_modes = _discover_joint_harmonic_modes(
        model=isolated_model,
        toas=toas,
        sessions=sessions,
        pb_days=pb_days,
        config=active,
    )

    exact: list[tuple[float, object, PartialELL1HarmonicMode]] = []
    for harmonic_mode in harmonic_modes:
        model, _ = _map_joint_harmonic_mode_to_pint(
            partial_par_path=par,
            isolated_model=isolated_model,
            isolated_cycles=isolated_cycles,
            toas=toas,
            sessions=sessions,
            pb_days=pb_days,
            mode=harmonic_mode,
            config=active,
        )
        local_chi2 = profile_two_level_pulse_lattice(
            relative_session_phases(_phase_cycles(model, toas), sessions),
            _phase_sigmas(model, toas),
            sessions,
            integer_radius=0,
        ).local_chi2
        exact.append((local_chi2, deepcopy(model), harmonic_mode))
    exact.sort(key=lambda item: item[0])
    if not exact:
        raise RuntimeError("No harmonic integer mode mapped to an exact PINT ELL1 model.")
    best_exact = exact[0][0]
    exact = [item for item in exact if item[0] <= best_exact + active.exact_mode_chi2_window][
        : active.retained_exact_modes
    ]

    output_modes: list[PartialELL1BootstrapMode] = []
    audit_modes = []
    for index, (local_chi2, model, harmonic_mode) in enumerate(exact):
        refinement_status = "refined_fixed_integer_component"
        refined_model = deepcopy(model)
        try:
            shape_chi2 = _refine_all_session_shape(
                model=refined_model,
                toas=toas,
                sessions=sessions,
                orbital_period_bounds_days=orbital_period_bounds,
                maximum_eccentricity=active.maximum_eccentricity,
                maximum_spin_frequency_offset_hz=(active.maximum_spin_frequency_offset_hz),
                spin_frequency_reference_hz=float(isolated_model["F0"].value),
                maximum_a1_seconds=active.projected_semimajor_axis_stages_seconds[-1],
                refine_spin_frequency=active.refine_spin_frequency,
            )
            model = refined_model
        except LocalIntegerComponentChangedError:
            # The mapped model is already an exact discrete mode.  For a fold
            # bank, retaining that mode is preferable to dropping the entire
            # PB alias merely because continuous refinement crossed its cell.
            shape_chi2 = float(local_chi2)
            refinement_status = "unrefined_integer_preserved"
        _freeze_all_parameters(model)
        fit_parameters = ["A1", "TASC", "EPS1", "EPS2"]
        if orbital_period_bounds is not None:
            fit_parameters.append("PB")
        if active.refine_spin_frequency:
            fit_parameters.append("F0")
        for name in fit_parameters:
            model[name].frozen = False
        mode_id = f"mode_{index:04d}"
        mode_path = output / "modes" / f"{mode_id}.par"
        _write_or_validate_bootstrap_mode(
            model=model,
            path=mode_path,
            resume=resume,
            expected_free_parameters=fit_parameters,
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
                "refinement_status": refinement_status,
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
            "architecture": "apollo_foldbank_joint_session_integer_bootstrap",
            "inputs": {"par": str(par), "tim": str(tim)},
            "normalized_tim": str(normalized_tim),
            "session_count": len(sessions),
            "discovery_sessions": [session.session_id for session in sessions],
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
                "fit_parameters": fit_parameters,
            },
            "config": asdict(active),
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


@dataclass(frozen=True)
class _SeedJob:
    alias: OrbitalPeriodAlias
    seed_id: str
    par_path: Path
    tim_path: Path
    output_dir: Path
    config: FoldbankBootstrapConfig
    training: FoldbankTrainingData
    clock_override_path: Path | None
    resume: bool


def _run_seed(job: _SeedJob) -> tuple[BootstrapEntry | None, dict[str, object]]:
    record = {"seed_id": job.seed_id, "pb_days": float(_par_values(job.par_path)["PB"])}
    try:
        result = bootstrap_foldbank_ell1(
            par_path=job.par_path,
            tim_path=job.tim_path,
            output_dir=job.output_dir,
            config=job.config,
            training_data=job.training,
            clock_override_path=job.clock_override_path,
            resume=job.resume,
        )
    except RuntimeError as error:
        return None, {
            **record,
            "status": "failed",
            "failure_reason": str(error),
            "audit_path": None,
            "mode_count": 0,
        }
    modes = tuple(
        replace(mode, mode_id=f"{job.alias.alias_id}_{job.seed_id}_{mode.mode_id}")
        for mode in result.modes
    )
    result = replace(result, modes=modes)
    return BootstrapEntry(job.alias.alias_id, job.seed_id, result), {
        **record,
        "status": "completed",
        "failure_reason": None,
        "audit_path": str(result.audit_path),
        "mode_count": len(modes),
    }


def build_bootstrap_portfolio(
    *,
    aliases: Sequence[OrbitalPeriodAlias],
    reference_pb_days: float,
    period_range_requested: bool,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: FoldbankBootstrapConfig,
    training: FoldbankTrainingData,
    clock_override_path: Path | None,
    resume: bool,
    workers: int,
) -> BootstrapPortfolio:
    jobs = []
    reference_cell = next(
        (
            alias.alias_id
            for alias in aliases
            if alias.minimum_days <= reference_pb_days <= alias.maximum_days
        ),
        None,
    )
    for alias in aliases:
        seeds = [("center", alias.center_days)]
        if (
            period_range_requested
            and alias.alias_id == reference_cell
            and not math.isclose(alias.center_days, reference_pb_days, abs_tol=1e-15, rel_tol=0)
        ):
            # The reference is another seed in the same cell, never another physical alias.
            seeds.append(("reference", reference_pb_days))
        for seed_id, pb_days in seeds:
            alias_par = write_alias_par(
                source=par_path,
                destination=output_dir / "bootstrap/inputs" / f"{alias.alias_id}_{seed_id}.par",
                pb_days=pb_days,
            )
            alias_config = replace(
                config,
                orbital_period_min_days=alias.minimum_days if period_range_requested else None,
                orbital_period_max_days=alias.maximum_days if period_range_requested else None,
            )
            jobs.append(
                _SeedJob(
                    alias,
                    seed_id,
                    alias_par,
                    tim_path,
                    output_dir / "bootstrap" / alias.alias_id / seed_id,
                    alias_config,
                    training,
                    clock_override_path,
                    resume,
                )
            )
    if workers == 1:
        completed = tuple(map(_run_seed, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            completed = tuple(executor.map(_run_seed, jobs))
    entries = []
    seeds_by_alias = {alias.alias_id: [] for alias in aliases}
    for job, (entry, record) in zip(jobs, completed, strict=True):
        seeds_by_alias[job.alias.alias_id].append(record)
        if entry is not None:
            entries.append(entry)
    records = []
    for alias in aliases:
        seeds = seeds_by_alias[alias.alias_id]
        count = sum(seed["status"] == "completed" for seed in seeds)
        failures = [seed["failure_reason"] for seed in seeds if seed["failure_reason"]]
        records.append(
            {
                **alias.to_dict(),
                "seeds": seeds,
                "status": "completed" if count == len(seeds) else "partial" if count else "failed",
                "failure_reason": "; ".join(failures) if failures else None,
                "completed_seed_count": count,
                "seed_count": len(seeds),
                "mode_count": sum(seed["mode_count"] for seed in seeds),
            }
        )
    return BootstrapPortfolio(tuple(aliases), tuple(entries), tuple(records))
