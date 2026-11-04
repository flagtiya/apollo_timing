from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from timing_accelerator.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_model,
    _load_pint_toas,
    _set_numeric_model_parameter_value,
)
from timing_accelerator.inference.bayesian_binary_model_adapter import (
    BINARY_MODEL_CAPABILITIES,
    ELL1_BINARY_MODELS,
    canonical_binary_model_name,
)
from timing_accelerator.inference.bayesian_dual_lattice_enumerator import (
    SourceSession,
    infer_contiguous_source_sessions,
    profile_two_level_pulse_lattice,
)
from timing_accelerator.inference.bayesian_partial_ell1_bootstrap import (
    PartialELL1BootstrapConfig,
    PartialELL1BootstrapResult,
    PartialELL1HarmonicMode,
    _active_par_lines,
    _discover_harmonic_modes,
    _informative_session,
    _par_values,
    _phase_cycles,
    _phase_sigmas,
    _session_harmonic_data,
    bootstrap_partial_ell1,
    is_incomplete_ell1_par,
    normalize_tempo2_tim_file,
)
from timing_accelerator.inference.pint_solution_artifacts import (
    write_pint_model_parfile_roundtrip_safe,
)


COARSE_BINARY_COMMON_REQUIRED_PARAMETERS = ("RAJ", "DECJ", "F0", "PEPOCH")
_ELL1_CORE = ("A1", "TASC", "EPS1", "EPS2")
_ECCENTRIC_CORE = ("A1", "T0", "ECC", "OM")
_NON_MODEL_LINES = {"MODE", "NITS"}


@dataclass(frozen=True)
class CoarseBinaryBootstrapConfig:
    """Budgets and physical support for incomplete binary startup."""

    projected_semimajor_axis_stages_seconds: tuple[float, ...] = (
        2.5,
        5.0,
        10.0,
        20.0,
        40.0,
    )
    retained_lattice_modes: int = 256
    retained_exact_modes: int = 4
    exact_mode_chi2_window: float = 2_000.0
    minimum_informative_session_toas: int = 8
    informative_session_reduced_chi2_threshold: float = 5.0
    pair_enumeration_chunk_rows: int = 128
    full_solver_support_width_scale: float = 1.0
    legacy_ell1_support_width_scale: float = 0.01
    maximum_ell1_eccentricity: float = 0.05
    maximum_eccentricity: float = 0.95
    eccentricity_seed_grid: tuple[float, ...] = (
        0.0,
        0.03,
        0.1,
        0.25,
        0.45,
        0.7,
        0.88,
    )
    orbital_phase_seed_count: int = 24
    longitude_seed_count: int = 12
    retained_keplerian_seeds_per_harmonic_mode: int = 12
    exact_refinement_max_nfev: int = 2_000
    atlas_eccentricities: tuple[float, ...] = (0.05, 0.35, 0.75)

    def __post_init__(self) -> None:
        stages = tuple(float(value) for value in self.projected_semimajor_axis_stages_seconds)
        if not stages or any(not np.isfinite(value) or value <= 0.0 for value in stages):
            raise ValueError("A1 bootstrap stages must be finite and positive.")
        if any(right <= left for left, right in zip(stages, stages[1:], strict=False)):
            raise ValueError("A1 bootstrap stages must be strictly increasing.")
        positive_integers = (
            self.retained_lattice_modes,
            self.retained_exact_modes,
            self.minimum_informative_session_toas,
            self.pair_enumeration_chunk_rows,
            self.orbital_phase_seed_count,
            self.longitude_seed_count,
            self.retained_keplerian_seeds_per_harmonic_mode,
            self.exact_refinement_max_nfev,
        )
        if any(int(value) <= 0 for value in positive_integers):
            raise ValueError("Coarse binary bootstrap budgets must be positive.")
        positive_scales = (
            self.exact_mode_chi2_window,
            self.informative_session_reduced_chi2_threshold,
            self.full_solver_support_width_scale,
            self.legacy_ell1_support_width_scale,
            self.maximum_ell1_eccentricity,
            self.maximum_eccentricity,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_scales):
            raise ValueError("Coarse binary bootstrap scales must be positive.")
        if self.maximum_ell1_eccentricity >= 1.0 or self.maximum_eccentricity >= 1.0:
            raise ValueError("Bootstrap eccentricity bounds must be below one.")
        eccentricities = tuple(float(value) for value in self.eccentricity_seed_grid)
        if not eccentricities or any(
            not np.isfinite(value) or value < 0.0 or value > self.maximum_eccentricity
            for value in eccentricities
        ):
            raise ValueError("Eccentricity seeds must lie inside the physical support.")
        atlas = tuple(float(value) for value in self.atlas_eccentricities)
        if not atlas or any(
            not np.isfinite(value) or value < 0.0 or value > self.maximum_eccentricity
            for value in atlas
        ):
            raise ValueError("Atlas eccentricities must lie inside the physical support.")

    def ell1_config(self) -> PartialELL1BootstrapConfig:
        return PartialELL1BootstrapConfig(
            projected_semimajor_axis_stages_seconds=(
                self.projected_semimajor_axis_stages_seconds
            ),
            retained_lattice_modes=self.retained_lattice_modes,
            retained_exact_modes=self.retained_exact_modes,
            exact_mode_chi2_window=self.exact_mode_chi2_window,
            minimum_informative_session_toas=self.minimum_informative_session_toas,
            maximum_eccentricity=self.maximum_ell1_eccentricity,
            informative_session_reduced_chi2_threshold=(
                self.informative_session_reduced_chi2_threshold
            ),
            pair_enumeration_chunk_rows=self.pair_enumeration_chunk_rows,
            full_solver_support_width_scale=self.legacy_ell1_support_width_scale,
        )

    def with_semimajor_axis_envelope(
        self,
        maximum_a1_seconds: float,
        *,
        source: str,
    ) -> tuple[CoarseBinaryBootstrapConfig, dict[str, Any] | None]:
        """Grow the staged A1 support so a declared A1 stays inside the search.

        The staged support is the only place where the coarse bootstrap can
        represent a projected semimajor axis. When prior information (for
        example a historical par file) places A1 above the largest default
        stage, every seed would be generated outside its own support. This
        widens the envelopes automatically and returns an auditable record so
        the adjustment is never silent.
        """

        requested = float(maximum_a1_seconds)
        if not np.isfinite(requested) or requested <= 0.0:
            raise ValueError("A1 support envelopes must be finite and positive.")
        stages = tuple(float(value) for value in self.projected_semimajor_axis_stages_seconds)
        previous_top = stages[-1]
        if requested <= previous_top:
            return self, None
        added: list[float] = []
        grown = list(stages)
        while grown[-1] < requested:
            grown.append(2.0 * grown[-1])
            added.append(grown[-1])
        adjusted = replace(
            self,
            projected_semimajor_axis_stages_seconds=tuple(grown),
        )
        record = {
            "adjusted": True,
            "source": str(source),
            "requested_a1_seconds": requested,
            "default_top_stage_seconds": previous_top,
            "effective_top_stage_seconds": grown[-1],
            "added_stages_seconds": added,
            "stages_seconds": list(grown),
        }
        return adjusted, record


@dataclass(frozen=True)
class CoarseBinaryModelSpec:
    model_name: str
    model_token: str
    model_family: str
    period_parameter: str
    period_days: float
    core_parameters: tuple[str, ...]
    missing_core_parameters: tuple[str, ...]
    input_fit_parameters: tuple[str, ...]


@dataclass(frozen=True)
class CoarseBinaryBootstrapMode:
    mode_id: str
    par_path: Path
    informative_session_chi2: float
    all_session_shape_chi2: float
    parameter_values: dict[str, float]


@dataclass(frozen=True)
class CoarseBinaryBootstrapResult:
    model_name: str
    model_family: str
    normalized_tim_path: Path
    audit_path: Path
    modes: tuple[CoarseBinaryBootstrapMode, ...]
    informative_session_id: str
    informative_session_toas: int
    recommended_support_width_scale: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
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
            "recommended_support_width_scale": self.recommended_support_width_scale,
        }


@dataclass(frozen=True)
class _KeplerianSeed:
    a1_seconds: float
    epoch_phase_cycles: float
    eccentricity: float
    omega_radians: float
    local_spin_offset_hz: float
    surrogate_chi2: float


def _input_fit_parameters(path: Path) -> tuple[str, ...]:
    fit = []
    for line in _active_par_lines(path):
        fields = line.split()
        if len(fields) >= 3 and fields[2] == "1":
            fit.append(fields[0].upper())
    return tuple(fit)


def inspect_coarse_binary_par(path: Path) -> CoarseBinaryModelSpec:
    values = _par_values(Path(path))
    if "BINARY" not in values:
        raise ValueError(
            "Coarse binary bootstrap requires an explicit BINARY model; APOLLO "
            "does not infer the physical model family from PB alone."
        )
    token = values["BINARY"]
    model_name = canonical_binary_model_name(token)
    if model_name not in BINARY_MODEL_CAPABILITIES:
        supported = ", ".join(BINARY_MODEL_CAPABILITIES)
        raise ValueError(f"Unsupported PINT binary model '{token}'. Supported: {supported}.")
    capability = BINARY_MODEL_CAPABILITIES[model_name]
    if "PB" in values:
        period_parameter = "PB"
        period_days = float(values["PB"])
    elif "FB0" in values:
        period_parameter = "FB0"
        fb0_hz = float(values["FB0"])
        period_days = 1.0 / (fb0_hz * 86400.0)
    else:
        raise ValueError("Coarse binary bootstrap requires a positive PB or FB0.")
    if not np.isfinite(period_days) or period_days <= 0.0:
        raise ValueError("Coarse binary bootstrap requires a positive PB or FB0.")
    core = _ELL1_CORE if model_name in ELL1_BINARY_MODELS else _ECCENTRIC_CORE
    return CoarseBinaryModelSpec(
        model_name=model_name,
        model_token=token,
        model_family=capability.family,
        period_parameter=period_parameter,
        period_days=float(period_days),
        core_parameters=core,
        missing_core_parameters=tuple(name for name in core if name not in values),
        input_fit_parameters=_input_fit_parameters(Path(path)),
    )


def declared_projected_semimajor_axis_seconds(path: Path) -> float | None:
    """Return the projected semimajor axis declared in a par file, if any.

    Prior information about A1 is the only input that can tell the staged
    bootstrap whether its default support is wide enough, so it is read
    directly from the par text instead of requiring a caller-supplied override.
    """

    values = _par_values(Path(path))
    for name in ("A1", "A0"):
        token = values.get(name)
        if token is None:
            continue
        try:
            value = abs(float(str(token).split()[0]))
        except (IndexError, TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value
    return None


def is_incomplete_binary_par(path: Path) -> bool:
    values = _par_values(Path(path))
    if "BINARY" not in values:
        return False
    model_name = canonical_binary_model_name(values["BINARY"])
    if model_name not in BINARY_MODEL_CAPABILITIES:
        return False
    core = _ELL1_CORE if model_name in ELL1_BINARY_MODELS else _ECCENTRIC_CORE
    period_missing = "PB" not in values and "FB0" not in values
    return period_missing or any(name not in values for name in core)


def _binary_parameter_names() -> set[str]:
    from pint.models.model_builder import AllComponents

    names = {"BINARY"}
    for component_name, component in AllComponents().components.items():
        if component_name.startswith("Binary"):
            names.update(str(name).upper() for name in component.params)
    return names


def _isolated_model_text(path: Path) -> str:
    binary_parameters = _binary_parameter_names()
    retained = []
    for line in _active_par_lines(path):
        key = line.split()[0].upper()
        if key in binary_parameters or key in _NON_MODEL_LINES:
            continue
        retained.append(line)
    if not any(line.split()[0].upper() == "F1" for line in retained):
        retained.append("F1 0")
    return "\n".join(retained) + "\n"


def _complete_model_text(
    path: Path,
    *,
    spec: CoarseBinaryModelSpec,
    parameter_values: dict[str, float],
) -> str:
    replaced = {"BINARY", *spec.core_parameters}
    retained = []
    for line in _active_par_lines(path):
        key = line.split()[0].upper()
        if key in replaced or key in _NON_MODEL_LINES:
            continue
        retained.append(line)
    if not any(line.split()[0].upper() == "F1" for line in retained):
        retained.append("F1 0")
    retained.append(f"BINARY {spec.model_token}")
    for name in spec.core_parameters:
        retained.append(f"{name} {float(parameter_values[name]):.18g}")
    return "\n".join(retained) + "\n"


def _write_canonical_pb_par(
    *,
    path: Path,
    output_path: Path,
    spec: CoarseBinaryModelSpec,
) -> Path:
    """Convert an FB0 coarse declaration to the PB coordinate used by ELL1."""

    retained = []
    fb0_fit_flag = False
    for line in _active_par_lines(path):
        fields = line.split()
        key = fields[0].upper()
        if key == "FB0":
            fb0_fit_flag = len(fields) >= 3 and fields[2] == "1"
            continue
        retained.append(line)
    pb_line = f"PB {spec.period_days:.18g}"
    if fb0_fit_flag:
        pb_line += " 1"
    retained.append(pb_line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(retained) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def _pint_model_from_text(text: str, *, model_name: str) -> object:
    from pint.models import get_model

    try:
        return get_model(StringIO(text), allow_tcb=True)
    except Exception as error:
        raise ValueError(
            f"Cannot materialize the declared {model_name} model after filling its "
            "Keplerian core. A required model-specific input may still be absent: "
            f"{type(error).__name__}: {error}"
        ) from error


def _bootstrap_fit_parameters(
    *,
    model: object,
    spec: CoarseBinaryModelSpec,
) -> tuple[str, ...]:
    requested = ["RAJ", "DECJ", "F0", "F1", spec.period_parameter]
    requested.extend(spec.core_parameters)
    requested.extend(spec.input_fit_parameters)
    result = []
    for name in requested:
        if name in model.params and name not in result:
            result.append(name)
    return tuple(result)


def _freeze_for_bootstrap(model: object, fit_parameters: Sequence[str]) -> None:
    _freeze_all_parameters(model)
    for name in fit_parameters:
        model[name].frozen = False


def _harmonic_target(
    *,
    mjds: np.ndarray,
    circular_design: np.ndarray,
    mode: PartialELL1HarmonicMode,
    pb_days: float,
) -> np.ndarray:
    angle = 2.0 * math.pi * (mjds - mjds[0]) / float(pb_days)
    delta_seconds = (mjds - mjds[0]) * 86400.0
    return (
        circular_design @ np.asarray((mode.phase_u, mode.phase_v), dtype=float)
        + mode.phase_sin2 * (np.sin(2.0 * angle) - np.sin(2.0 * angle[0]))
        + mode.phase_cos2 * (np.cos(2.0 * angle) - np.cos(2.0 * angle[0]))
        + mode.local_spin_offset_hz * delta_seconds
    )


def _solve_kepler_equation(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    mean = np.asarray(mean_anomaly, dtype=float)
    eccentric = mean.copy()
    if eccentricity >= 0.8:
        eccentric.fill(math.pi)
        eccentric += mean - np.mod(mean, 2.0 * math.pi)
    for _ in range(24):
        residual = eccentric - eccentricity * np.sin(eccentric) - mean
        step = residual / np.maximum(1.0 - eccentricity * np.cos(eccentric), 1.0e-10)
        eccentric -= step
        if float(np.max(np.abs(step))) < 1.0e-13:
            break
    return eccentric


def _keplerian_delay_seconds(
    *,
    mjds: np.ndarray,
    pb_days: float,
    a1_seconds: float,
    epoch_phase_cycles: float,
    eccentricity: float,
    omega_radians: float,
    reference_mjd: float,
) -> np.ndarray:
    t0 = reference_mjd + float(epoch_phase_cycles) * float(pb_days)
    mean = 2.0 * math.pi * (np.asarray(mjds, dtype=float) - t0) / float(pb_days)
    eccentric_anomaly = _solve_kepler_equation(mean, float(eccentricity))
    root = math.sqrt(max(1.0 - float(eccentricity) ** 2, 1.0e-12))
    return float(a1_seconds) * (
        (np.cos(eccentric_anomaly) - float(eccentricity)) * math.sin(omega_radians)
        + root * np.sin(eccentric_anomaly) * math.cos(omega_radians)
    )


def _keplerian_surrogate_residuals(
    values: np.ndarray,
    *,
    mjds: np.ndarray,
    target: np.ndarray,
    sigmas: np.ndarray,
    pb_days: float,
    f0_hz: float,
    reference_mjd: float,
) -> np.ndarray:
    a1, phase, eccentricity, omega, spin_offset = np.asarray(values, dtype=float)
    delay = _keplerian_delay_seconds(
        mjds=mjds,
        pb_days=pb_days,
        a1_seconds=a1,
        epoch_phase_cycles=phase,
        eccentricity=eccentricity,
        omega_radians=omega,
        reference_mjd=reference_mjd,
    )
    correction = float(f0_hz) * (delay - delay[0])
    correction += spin_offset * (mjds - mjds[0]) * 86400.0
    return (correction - target) / sigmas


def _discover_keplerian_seeds(
    *,
    mjds: np.ndarray,
    target: np.ndarray,
    sigmas: np.ndarray,
    pb_days: float,
    f0_hz: float,
    reference_mjd: float,
    mode: PartialELL1HarmonicMode,
    config: CoarseBinaryBootstrapConfig,
) -> tuple[_KeplerianSeed, ...]:
    phases = np.linspace(-0.5, 0.5, config.orbital_phase_seed_count, endpoint=False)
    longitudes = np.linspace(-math.pi, math.pi, config.longitude_seed_count, endpoint=False)
    maximum_a1 = config.projected_semimajor_axis_stages_seconds[-1]
    candidates: list[tuple[float, np.ndarray]] = []
    weights = 1.0 / np.square(sigmas)
    delta_seconds = (mjds - mjds[0]) * 86400.0
    target_without_spin = target - mode.local_spin_offset_hz * delta_seconds
    for eccentricity in config.eccentricity_seed_grid:
        for phase in phases:
            for omega in longitudes:
                unit_delay = _keplerian_delay_seconds(
                    mjds=mjds,
                    pb_days=pb_days,
                    a1_seconds=1.0,
                    epoch_phase_cycles=float(phase),
                    eccentricity=float(eccentricity),
                    omega_radians=float(omega),
                    reference_mjd=reference_mjd,
                )
                unit = float(f0_hz) * (unit_delay - unit_delay[0])
                denominator = float(np.sum(weights * np.square(unit)))
                if denominator <= 0.0:
                    continue
                a1 = float(np.sum(weights * target_without_spin * unit) / denominator)
                if not 1.0e-6 <= a1 <= maximum_a1:
                    continue
                values = np.asarray(
                    (a1, phase, eccentricity, omega, mode.local_spin_offset_hz),
                    dtype=float,
                )
                residuals = _keplerian_surrogate_residuals(
                    values,
                    mjds=mjds,
                    target=target,
                    sigmas=sigmas,
                    pb_days=pb_days,
                    f0_hz=f0_hz,
                    reference_mjd=reference_mjd,
                )
                candidates.append((float(residuals @ residuals), values))
    candidates.sort(key=lambda item: item[0])
    retained = candidates[: config.retained_keplerian_seeds_per_harmonic_mode]
    if not retained:
        raise RuntimeError("The canonical Keplerian posterior produced no physical seed.")
    span_seconds = max(float(np.ptp(mjds)) * 86400.0, 1.0)
    spin_bound = max(abs(float(mode.local_spin_offset_hz)) * 10.0, 5.0 / span_seconds)
    lower = np.asarray((1.0e-6, -0.5, 0.0, -math.pi, -spin_bound), dtype=float)
    upper = np.asarray(
        (maximum_a1, 0.5, config.maximum_eccentricity, math.pi, spin_bound),
        dtype=float,
    )
    refined: list[_KeplerianSeed] = []
    for _, seed in retained:
        fit = least_squares(
            _keplerian_surrogate_residuals,
            seed,
            bounds=(lower, upper),
            args=(),
            kwargs={
                "mjds": mjds,
                "target": target,
                "sigmas": sigmas,
                "pb_days": pb_days,
                "f0_hz": f0_hz,
                "reference_mjd": reference_mjd,
            },
            x_scale=np.asarray((1.0, 0.1, 0.1, 1.0, spin_bound), dtype=float),
            max_nfev=1_000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        residuals = _keplerian_surrogate_residuals(
            fit.x,
            mjds=mjds,
            target=target,
            sigmas=sigmas,
            pb_days=pb_days,
            f0_hz=f0_hz,
            reference_mjd=reference_mjd,
        )
        refined.append(
            _KeplerianSeed(
                a1_seconds=float(fit.x[0]),
                epoch_phase_cycles=float(fit.x[1]),
                eccentricity=float(fit.x[2]),
                omega_radians=float(fit.x[3]),
                local_spin_offset_hz=float(fit.x[4]),
                surrogate_chi2=float(residuals @ residuals),
            )
        )
    refined.sort(key=lambda item: item.surrogate_chi2)
    unique: list[_KeplerianSeed] = []
    for seed in refined:
        coordinate = np.asarray(
            (
                seed.a1_seconds,
                seed.epoch_phase_cycles,
                seed.eccentricity,
                math.sin(seed.omega_radians),
                math.cos(seed.omega_radians),
            )
        )
        if all(
            np.linalg.norm(
                coordinate
                - np.asarray(
                    (
                        other.a1_seconds,
                        other.epoch_phase_cycles,
                        other.eccentricity,
                        math.sin(other.omega_radians),
                        math.cos(other.omega_radians),
                    )
                )
            )
            > 1.0e-4
            for other in unique
        ):
            unique.append(seed)
    return tuple(unique[: config.retained_exact_modes])


def _best_rectangular_integer_basis(
    design: np.ndarray,
    sigmas: np.ndarray,
) -> tuple[int, int]:
    best: tuple[float, int, int] | None = None
    for left in range(1, design.shape[0]):
        for right in range(left + 1, design.shape[0]):
            determinant = abs(float(np.linalg.det(design[[left, right]])))
            score = determinant / (float(sigmas[left]) * float(sigmas[right]))
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None or best[0] <= 0.0:
        raise RuntimeError("The informative session has no rank-two Keplerian basis.")
    return best[1], best[2]


def _retain_rectangular_modes(
    modes: list[tuple[float, np.ndarray]],
    maximum: int,
) -> list[tuple[float, np.ndarray]]:
    modes.sort(key=lambda item: item[0])
    return modes[: int(maximum)]


def _enumerate_rectangular_lattice_modes(
    *,
    wrapped: np.ndarray,
    design: np.ndarray,
    sigmas: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    maximum_modes: int,
    chunk_rows: int,
) -> tuple[tuple[float, np.ndarray], ...]:
    """Enumerate the integer lattice for two bounded semilinear coordinates."""

    left, right = _best_rectangular_integer_basis(design, sigmas)
    pair_design = design[[left, right]]
    inverse = np.linalg.inv(pair_design)
    corners = np.asarray(
        [
            (x0, x1)
            for x0 in (float(lower[0]), float(upper[0]))
            for x1 in (float(lower[1]), float(upper[1]))
        ],
        dtype=float,
    )
    integer_ranges = []
    for row, observed in zip(pair_design, wrapped[[left, right]], strict=True):
        projected = float(observed) + corners @ row
        integer_ranges.append(
            np.arange(
                int(np.floor(np.min(projected))) - 1,
                int(np.ceil(np.max(projected))) + 2,
                dtype=int,
            )
        )
    retained: list[tuple[float, np.ndarray]] = []
    second_range = integer_ranges[1]
    for start in range(0, len(integer_ranges[0]), int(chunk_rows)):
        first = integer_ranges[0][start : start + int(chunk_rows)]
        grid_left, grid_right = np.meshgrid(first, second_range, indexing="ij")
        rhs = np.stack(
            (
                grid_left.ravel() - wrapped[left],
                grid_right.ravel() - wrapped[right],
            ),
            axis=1,
        )
        coordinates = rhs @ inverse.T
        inside = np.all(coordinates >= lower, axis=1) & np.all(coordinates <= upper, axis=1)
        coordinates = coordinates[inside]
        if coordinates.size == 0:
            continue
        residuals = wrapped[None, :] + coordinates @ design.T
        residuals -= np.rint(residuals)
        scores = np.sum(np.square(residuals / sigmas[None, :]), axis=1)
        local_count = min(int(maximum_modes), scores.size)
        local = np.argpartition(scores, local_count - 1)[:local_count]
        retained.extend((float(scores[index]), coordinates[index]) for index in local)
        retained = _retain_rectangular_modes(retained, 8 * int(maximum_modes))
    retained = _retain_rectangular_modes(retained, int(maximum_modes))
    refined: list[tuple[float, np.ndarray]] = []
    weighted_design = design / sigmas[:, None]
    for _, coordinates in retained:
        integers = np.rint(wrapped + design @ coordinates)
        target = (integers - wrapped) / sigmas
        fitted = np.linalg.lstsq(weighted_design, target, rcond=None)[0]
        fitted = np.minimum(np.maximum(fitted, lower), upper)
        residuals = (wrapped + design @ fitted - integers) / sigmas
        refined.append((float(residuals @ residuals), fitted))
    return tuple(_retain_rectangular_modes(refined, int(maximum_modes)))


def _direct_keplerian_fixed_path_residuals(
    values: np.ndarray,
    *,
    mjds: np.ndarray,
    wrapped: np.ndarray,
    integers: np.ndarray,
    sigmas: np.ndarray,
    pb_days: float,
    f0_hz: float,
    reference_mjd: float,
) -> np.ndarray:
    a1, phase, eccentricity, omega, spin_offset = np.asarray(values, dtype=float)
    delay = _keplerian_delay_seconds(
        mjds=mjds,
        pb_days=pb_days,
        a1_seconds=a1,
        epoch_phase_cycles=phase,
        eccentricity=eccentricity,
        omega_radians=omega,
        reference_mjd=reference_mjd,
    )
    correction = float(f0_hz) * (delay - delay[0])
    correction += spin_offset * (mjds - mjds[0]) * 86400.0
    return (wrapped + correction - integers) / sigmas


def _discover_direct_keplerian_seeds(
    *,
    mjds: np.ndarray,
    wrapped: np.ndarray,
    sigmas: np.ndarray,
    pb_days: float,
    f0_hz: float,
    reference_mjd: float,
    config: CoarseBinaryBootstrapConfig,
) -> tuple[_KeplerianSeed, ...]:
    """Build the eccentric posterior directly, without a low-e harmonic proxy."""

    phases = np.linspace(-0.5, 0.5, config.orbital_phase_seed_count, endpoint=False)
    maximum_a1 = config.projected_semimajor_axis_stages_seconds[-1]
    span_seconds = max(float(np.ptp(mjds)) * 86400.0, 1.0)
    spin_bound = max(abs(float(f0_hz)) * 1.0e-5, 5.0 / span_seconds)
    semilinear_lower = np.asarray((-maximum_a1, -maximum_a1), dtype=float)
    semilinear_upper = np.asarray((maximum_a1, maximum_a1), dtype=float)
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    modes_per_shape = max(min(config.retained_exact_modes, 2), 1)
    for eccentricity in config.eccentricity_seed_grid:
        for phase in phases:
            t0 = reference_mjd + float(phase) * float(pb_days)
            mean = 2.0 * math.pi * (np.asarray(mjds) - t0) / float(pb_days)
            eccentric_anomaly = _solve_kepler_equation(mean, float(eccentricity))
            basis_u = np.cos(eccentric_anomaly) - float(eccentricity)
            basis_v = math.sqrt(max(1.0 - float(eccentricity) ** 2, 1.0e-12)) * np.sin(
                eccentric_anomaly
            )
            design = float(f0_hz) * np.column_stack(
                (basis_u - basis_u[0], basis_v - basis_v[0])
            )
            try:
                lattice_modes = _enumerate_rectangular_lattice_modes(
                    wrapped=wrapped,
                    design=design,
                    sigmas=sigmas,
                    lower=semilinear_lower,
                    upper=semilinear_upper,
                    maximum_modes=modes_per_shape,
                    chunk_rows=config.pair_enumeration_chunk_rows,
                )
            except (RuntimeError, np.linalg.LinAlgError):
                continue
            for score, fast in lattice_modes:
                orbit_u = float(fast[0])
                orbit_v = float(fast[1])
                a1 = math.hypot(orbit_u, orbit_v)
                if not 1.0e-6 <= a1 <= maximum_a1:
                    continue
                omega = math.atan2(orbit_u, orbit_v)
                state = np.asarray(
                    (a1, phase, eccentricity, omega, 0.0),
                    dtype=float,
                )
                integers = np.rint(wrapped + design @ fast)
                candidates.append((score, state, integers))
        candidates = sorted(candidates, key=lambda item: item[0])[
            : 8 * config.retained_lattice_modes
        ]
    candidates.sort(key=lambda item: item[0])
    retained = candidates[: config.retained_lattice_modes]
    if not retained:
        raise RuntimeError("The direct Keplerian integer lattice produced no mode.")
    lower = np.asarray((1.0e-6, -0.5, 0.0, -math.pi, -spin_bound), dtype=float)
    upper = np.asarray(
        (maximum_a1, 0.5, config.maximum_eccentricity, math.pi, spin_bound),
        dtype=float,
    )
    refined: list[_KeplerianSeed] = []
    for _, state, integers in retained:
        state = np.minimum(np.maximum(state, lower + 1.0e-12), upper - 1.0e-12)
        fit = least_squares(
            _direct_keplerian_fixed_path_residuals,
            state,
            bounds=(lower, upper),
            kwargs={
                "mjds": mjds,
                "wrapped": wrapped,
                "integers": integers,
                "sigmas": sigmas,
                "pb_days": pb_days,
                "f0_hz": f0_hz,
                "reference_mjd": reference_mjd,
            },
            x_scale=np.asarray((1.0, 0.1, 0.1, 1.0, spin_bound), dtype=float),
            max_nfev=1_000,
            xtol=1.0e-12,
            ftol=1.0e-12,
            gtol=1.0e-12,
        )
        residuals = _direct_keplerian_fixed_path_residuals(
            fit.x,
            mjds=mjds,
            wrapped=wrapped,
            integers=integers,
            sigmas=sigmas,
            pb_days=pb_days,
            f0_hz=f0_hz,
            reference_mjd=reference_mjd,
        )
        refined.append(
            _KeplerianSeed(
                a1_seconds=float(fit.x[0]),
                epoch_phase_cycles=float(fit.x[1]),
                eccentricity=float(fit.x[2]),
                omega_radians=float(fit.x[3]),
                local_spin_offset_hz=float(fit.x[4]),
                surrogate_chi2=float(residuals @ residuals),
            )
        )
    refined.sort(key=lambda item: item.surrogate_chi2)
    unique: list[_KeplerianSeed] = []
    for seed in refined:
        if all(
            abs(seed.a1_seconds - other.a1_seconds) > 1.0e-4
            or abs(seed.eccentricity - other.eccentricity) > 1.0e-5
            or abs(seed.epoch_phase_cycles - other.epoch_phase_cycles) > 1.0e-5
            for other in unique
        ):
            unique.append(seed)
        if len(unique) >= config.retained_exact_modes:
            break
    return tuple(unique)


def _keplerian_seed_target(
    *,
    seed: _KeplerianSeed,
    mjds: np.ndarray,
    pb_days: float,
    f0_hz: float,
    reference_mjd: float,
) -> np.ndarray:
    delay = _keplerian_delay_seconds(
        mjds=mjds,
        pb_days=pb_days,
        a1_seconds=seed.a1_seconds,
        epoch_phase_cycles=seed.epoch_phase_cycles,
        eccentricity=seed.eccentricity,
        omega_radians=seed.omega_radians,
        reference_mjd=reference_mjd,
    )
    target = float(f0_hz) * (delay - delay[0])
    target += seed.local_spin_offset_hz * (mjds - mjds[0]) * 86400.0
    return target


def _absolute_integer_path(
    *,
    isolated_cycles: np.ndarray,
    indices: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    relative = np.asarray(
        isolated_cycles[indices] - isolated_cycles[indices[0]], dtype=float
    )
    baseline = np.rint(relative)
    wrapped = relative - baseline
    return baseline + np.rint(wrapped + np.asarray(target, dtype=float))


def _exact_refine_eccentric_model(
    *,
    par_path: Path,
    spec: CoarseBinaryModelSpec,
    isolated_model: object,
    isolated_cycles: np.ndarray,
    toas: object,
    session: SourceSession,
    target: np.ndarray,
    seed: _KeplerianSeed,
    config: CoarseBinaryBootstrapConfig,
) -> tuple[object, float]:
    reference_mjd = float(isolated_model["PEPOCH"].value)
    values = {
        "A1": seed.a1_seconds,
        "T0": reference_mjd + seed.epoch_phase_cycles * spec.period_days,
        "ECC": seed.eccentricity,
        "OM": math.degrees(seed.omega_radians),
    }
    model = _pint_model_from_text(
        _complete_model_text(par_path, spec=spec, parameter_values=values),
        model_name=spec.model_name,
    )
    _set_numeric_model_parameter_value(
        model,
        "F0",
        float(isolated_model["F0"].value) + seed.local_spin_offset_hz,
    )
    indices = np.asarray(session.toa_indices, dtype=int)
    integer_path = _absolute_integer_path(
        isolated_cycles=isolated_cycles,
        indices=indices,
        target=target,
    )
    sigmas = _phase_sigmas(model, toas)[indices]
    relative_sigmas = np.sqrt(np.square(sigmas) + sigmas[0] ** 2)
    relative_sigmas[0] = sigmas[0]
    span_seconds = max(float(np.ptp(np.asarray(toas.get_mjds().value)[indices])) * 86400.0, 1.0)
    spin_bound = max(abs(seed.local_spin_offset_hz) * 10.0, 5.0 / span_seconds)
    start = np.asarray(
        (
            seed.a1_seconds,
            seed.epoch_phase_cycles,
            seed.eccentricity,
            seed.omega_radians,
            seed.local_spin_offset_hz,
        ),
        dtype=float,
    )
    lower = np.asarray((1.0e-6, -0.5, 0.0, -math.pi, -spin_bound), dtype=float)
    upper = np.asarray(
        (
            config.projected_semimajor_axis_stages_seconds[-1],
            0.5,
            config.maximum_eccentricity,
            math.pi,
            spin_bound,
        ),
        dtype=float,
    )
    initial_f0 = float(isolated_model["F0"].value)

    def residuals(state: np.ndarray) -> np.ndarray:
        a1, phase, eccentricity, omega, spin_offset = state
        _set_numeric_model_parameter_value(model, "A1", float(a1))
        _set_numeric_model_parameter_value(
            model, "T0", reference_mjd + float(phase) * spec.period_days
        )
        _set_numeric_model_parameter_value(model, "ECC", float(eccentricity))
        _set_numeric_model_parameter_value(model, "OM", math.degrees(float(omega)))
        _set_numeric_model_parameter_value(model, "F0", initial_f0 + float(spin_offset))
        cycles = _phase_cycles(model, toas)
        relative = np.asarray(cycles[indices] - cycles[indices[0]], dtype=float)
        return (relative - integer_path) / relative_sigmas

    fit = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        x_scale=np.asarray((1.0, 0.1, 0.1, 1.0, spin_bound), dtype=float),
        max_nfev=config.exact_refinement_max_nfev,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    final = residuals(fit.x)
    return deepcopy(model), float(final @ final)


def _materialize_keplerian_atlas(
    *,
    par_path: Path,
    spec: CoarseBinaryModelSpec,
    isolated_model: object,
    toas: object,
    sessions: Sequence[SourceSession],
    config: CoarseBinaryBootstrapConfig,
) -> tuple[tuple[float, float, object, _KeplerianSeed, str], ...]:
    """Retain prior-support anchors so local coarse scores cannot prune truth."""

    reference_mjd = float(isolated_model["PEPOCH"].value)
    anchors = []
    for a1_seconds in config.projected_semimajor_axis_stages_seconds:
        for eccentricity in config.atlas_eccentricities:
            seed = _KeplerianSeed(
                a1_seconds=float(a1_seconds),
                epoch_phase_cycles=0.0,
                eccentricity=float(eccentricity),
                omega_radians=0.0,
                local_spin_offset_hz=0.0,
                surrogate_chi2=math.inf,
            )
            model = _pint_model_from_text(
                _complete_model_text(
                    par_path,
                    spec=spec,
                    parameter_values={
                        "A1": seed.a1_seconds,
                        "T0": reference_mjd,
                        "ECC": seed.eccentricity,
                        "OM": 0.0,
                    },
                ),
                model_name=spec.model_name,
            )
            shape_chi2 = _all_session_shape_chi2(
                model=model,
                toas=toas,
                sessions=sessions,
            )
            anchors.append((shape_chi2, shape_chi2, model, seed, "support_atlas"))
    return tuple(anchors)


def _all_session_shape_chi2(
    *,
    model: object,
    toas: object,
    sessions: Sequence[SourceSession],
) -> float:
    profile = profile_two_level_pulse_lattice(
        np.asarray(_phase_cycles(model, toas), dtype=float),
        _phase_sigmas(model, toas),
        sessions,
        integer_radius=0,
    )
    return float(profile.local_chi2)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _write_or_validate_mode(
    *,
    model: object,
    path: Path,
    fit_parameters: Sequence[str],
    resume: bool,
) -> None:
    if not resume or not path.is_file():
        write_pint_model_parfile_roundtrip_safe(model=model, path=path)
        return
    existing = _load_pint_model(path)
    expected_fit = tuple(str(name).upper() for name in fit_parameters)
    retained_fit = tuple(str(name).upper() for name in existing.free_params)
    if set(expected_fit) != set(retained_fit) or len(expected_fit) != len(retained_fit):
        raise RuntimeError(
            f"Bootstrap resume fit flags changed in {path}; use a new output directory."
        )
    for name in expected_fit:
        expected = float(model[name].value)
        retained = float(existing[name].value)
        if not np.isclose(retained, expected, rtol=1.0e-11, atol=1.0e-18):
            raise RuntimeError(
                f"Bootstrap resume changed {name} in {path}; use a new output directory."
            )


def _convert_legacy_ell1_result(
    result: PartialELL1BootstrapResult,
    *,
    spec: CoarseBinaryModelSpec,
) -> CoarseBinaryBootstrapResult:
    modes = tuple(
        CoarseBinaryBootstrapMode(
            mode_id=mode.mode_id,
            par_path=mode.par_path,
            informative_session_chi2=mode.informative_session_chi2,
            all_session_shape_chi2=mode.all_session_shape_chi2,
            parameter_values=mode.parameter_values,
        )
        for mode in result.modes
    )
    return CoarseBinaryBootstrapResult(
        model_name=spec.model_name,
        model_family=spec.model_family,
        normalized_tim_path=result.normalized_tim_path,
        audit_path=result.audit_path,
        modes=modes,
        informative_session_id=result.informative_session_id,
        informative_session_toas=result.informative_session_toas,
        recommended_support_width_scale=result.recommended_support_width_scale,
    )


def bootstrap_coarse_binary(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: CoarseBinaryBootstrapConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> CoarseBinaryBootstrapResult:
    """Infer a missing Keplerian core for any declared PINT binary family."""

    active = config or CoarseBinaryBootstrapConfig()
    par = Path(par_path).resolve()
    tim = Path(tim_path).resolve()
    output = Path(output_dir).resolve()
    spec = inspect_coarse_binary_par(par)
    if not spec.missing_core_parameters:
        raise ValueError(
            f"The declared {spec.model_name} model already has a complete Keplerian core."
        )
    values = _par_values(par)
    missing_common = [
        name for name in COARSE_BINARY_COMMON_REQUIRED_PARAMETERS if name not in values
    ]
    if missing_common:
        raise ValueError("Coarse binary par is missing: " + ", ".join(missing_common))

    # Keep the regression-proven ELL1 implementation as one family adapter.
    if spec.model_name in ELL1_BINARY_MODELS and is_incomplete_ell1_par(par):
        ell1_par = par
        if spec.period_parameter == "FB0":
            ell1_par = _write_canonical_pb_par(
                path=par,
                output_path=output / "input.canonical_pb.par",
                spec=spec,
            )
        legacy = bootstrap_partial_ell1(
            par_path=ell1_par,
            tim_path=tim,
            output_dir=output,
            config=active.ell1_config(),
            clock_override_path=clock_override_path,
            resume=resume,
        )
        return _convert_legacy_ell1_result(legacy, spec=spec)

    normalized_tim = normalize_tempo2_tim_file(
        input_path=tim,
        output_path=output / "input.normalized.tim",
    )
    isolated_model = _pint_model_from_text(
        _isolated_model_text(par), model_name=spec.model_name
    )
    toas = _load_pint_toas(
        normalized_tim,
        seed_model=isolated_model,
        clock_override_path=clock_override_path,
    )
    sessions = infer_contiguous_source_sessions(toas)
    informative = _informative_session(sessions, active.minimum_informative_session_toas)
    isolated_cycles = _phase_cycles(isolated_model, toas)
    mjds, wrapped, _, sigmas = _session_harmonic_data(
        model=isolated_model,
        toas=toas,
        session=informative,
        pb_days=spec.period_days,
    )
    indices = np.asarray(informative.toa_indices, dtype=int)
    f0_hz = abs(float(isolated_model["F0"].value))
    reference_mjd = float(isolated_model["PEPOCH"].value)
    keplerian_seeds = _discover_direct_keplerian_seeds(
        mjds=mjds,
        wrapped=wrapped,
        sigmas=sigmas,
        pb_days=spec.period_days,
        f0_hz=f0_hz,
        reference_mjd=reference_mjd,
        config=active,
    )
    exact: list[tuple[float, float, object, _KeplerianSeed, str]] = []
    for seed in keplerian_seeds:
        target = _keplerian_seed_target(
            seed=seed,
            mjds=mjds,
            pb_days=spec.period_days,
            f0_hz=f0_hz,
            reference_mjd=reference_mjd,
        )
        model, local_chi2 = _exact_refine_eccentric_model(
            par_path=par,
            spec=spec,
            isolated_model=isolated_model,
            isolated_cycles=isolated_cycles,
            toas=toas,
            session=informative,
            target=target,
            seed=seed,
            config=active,
        )
        shape_chi2 = _all_session_shape_chi2(
            model=model, toas=toas, sessions=sessions
        )
        exact.append((shape_chi2, local_chi2, model, seed, "data_mode"))
    if not exact:
        raise RuntimeError(
            f"The {spec.model_name} canonical Keplerian posterior produced no exact PINT mode."
        )
    exact.sort(key=lambda item: (item[0], item[1]))
    best_shape = exact[0][0]
    data_modes = [
        item for item in exact if item[0] <= best_shape + active.exact_mode_chi2_window
    ][: active.retained_exact_modes]
    exact = data_modes + list(
        _materialize_keplerian_atlas(
            par_path=par,
            spec=spec,
            isolated_model=isolated_model,
            toas=toas,
            sessions=sessions,
            config=active,
        )
    )

    output_modes = []
    audit_modes = []
    for index, (shape_chi2, local_chi2, model, seed, seed_role) in enumerate(exact):
        fit_parameters = _bootstrap_fit_parameters(model=model, spec=spec)
        _freeze_for_bootstrap(model, fit_parameters)
        mode_id = f"mode_{index:04d}"
        mode_path = output / "modes" / f"{mode_id}.par"
        _write_or_validate_mode(
            model=model,
            path=mode_path,
            fit_parameters=fit_parameters,
            resume=resume,
        )
        parameters = {name: float(model[name].value) for name in fit_parameters}
        output_modes.append(
            CoarseBinaryBootstrapMode(
                mode_id=mode_id,
                par_path=mode_path,
                informative_session_chi2=local_chi2,
                all_session_shape_chi2=shape_chi2,
                parameter_values=parameters,
            )
        )
        audit_modes.append(
            {
                "mode_id": mode_id,
                "par_path": str(mode_path),
                "canonical_keplerian_seed": {
                    "a1_seconds": seed.a1_seconds,
                    "epoch_phase_cycles": seed.epoch_phase_cycles,
                    "eccentricity": seed.eccentricity,
                    "omega_radians": seed.omega_radians,
                    "local_spin_offset_hz": seed.local_spin_offset_hz,
                    "surrogate_chi2": (
                        seed.surrogate_chi2
                        if np.isfinite(seed.surrogate_chi2)
                        else None
                    ),
                },
                "seed_role": seed_role,
                "informative_session_chi2": local_chi2,
                "all_session_shape_chi2": shape_chi2,
                "fit_parameters": fit_parameters,
                "parameter_values": parameters,
            }
        )
    audit_path = _write_json(
        output / "audit.json",
        {
            "schema_version": 1,
            "architecture": "apollo_bayesian_canonical_keplerian_bootstrap",
            "inputs": {"par": str(par), "tim": str(tim)},
            "declared_model": {
                "name": spec.model_name,
                "token": spec.model_token,
                "family": spec.model_family,
                "period_parameter": spec.period_parameter,
                "missing_core_parameters": spec.missing_core_parameters,
            },
            "normalized_tim": str(normalized_tim),
            "session_count": len(sessions),
            "informative_session": {
                "session_id": informative.session_id,
                "toa_count": len(indices),
                "epoch_mjd": informative.epoch_mjd,
            },
            "scientific_contract": {
                "model_family_guessing": "forbidden",
                "historical_solution_seed": "forbidden",
                "source_class_prior": "forbidden",
                "manual_integer_wrap": "forbidden",
                "automatic_high_order_parameters": "forbidden",
                "posterior_discovery": "canonical_keplerian_integer_lattice",
                "exact_refinement": "declared_model_pint_fixed_integer_path",
                "support_atlas": "retained_independently_of_local_chi2",
            },
            "config": {
                "projected_semimajor_axis_stages_seconds": (
                    active.projected_semimajor_axis_stages_seconds
                ),
                "eccentricity_seed_grid": active.eccentricity_seed_grid,
                "maximum_eccentricity": active.maximum_eccentricity,
                "atlas_eccentricities": active.atlas_eccentricities,
                "retained_lattice_modes": active.retained_lattice_modes,
                "retained_exact_modes": active.retained_exact_modes,
                "full_solver_support_width_scale": (
                    active.full_solver_support_width_scale
                ),
            },
            "modes": audit_modes,
        },
    )
    return CoarseBinaryBootstrapResult(
        model_name=spec.model_name,
        model_family=spec.model_family,
        normalized_tim_path=normalized_tim,
        audit_path=audit_path,
        modes=tuple(output_modes),
        informative_session_id=informative.session_id,
        informative_session_toas=len(indices),
        recommended_support_width_scale=active.full_solver_support_width_scale,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = CoarseBinaryBootstrapConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Infer a complete Keplerian seed portfolio for an explicitly "
            "declared incomplete PINT binary model."
        )
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--a1-stage", dest="a1_stages", action="append", type=float)
    parser.add_argument(
        "--retained-exact-modes", type=int, default=defaults.retained_exact_modes
    )
    parser.add_argument(
        "--maximum-eccentricity", type=float, default=defaults.maximum_eccentricity
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = CoarseBinaryBootstrapConfig(
        projected_semimajor_axis_stages_seconds=(
            tuple(args.a1_stages)
            if args.a1_stages
            else CoarseBinaryBootstrapConfig().projected_semimajor_axis_stages_seconds
        ),
        retained_exact_modes=args.retained_exact_modes,
        maximum_eccentricity=args.maximum_eccentricity,
        eccentricity_seed_grid=tuple(
            value
            for value in CoarseBinaryBootstrapConfig().eccentricity_seed_grid
            if value <= args.maximum_eccentricity
        ),
    )
    result = bootstrap_coarse_binary(
        par_path=args.par,
        tim_path=args.tim,
        output_dir=args.output,
        config=config,
        clock_override_path=args.clock_override,
        resume=args.resume,
    )
    print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
    return 0


__all__ = [
    "COARSE_BINARY_COMMON_REQUIRED_PARAMETERS",
    "CoarseBinaryBootstrapConfig",
    "CoarseBinaryBootstrapMode",
    "CoarseBinaryBootstrapResult",
    "CoarseBinaryModelSpec",
    "bootstrap_coarse_binary",
    "inspect_coarse_binary_par",
    "is_incomplete_binary_par",
]


if __name__ == "__main__":
    raise SystemExit(main())
