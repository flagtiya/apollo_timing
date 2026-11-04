from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
import math
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from pint import Tsun
from pint.binaryconvert import convert_binary
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.residuals import Residuals

from apollo_timing.backend.pint_backend import (
    _set_numeric_model_parameter_value,
)
from apollo_timing.inference.bayesian_binary_model_adapter import (
    BINARY_MODEL_CAPABILITIES,
    SUPPORTED_BINARY_MODELS,
    canonical_binary_model_name,
)
from apollo_timing.inference.pint_phase_basis import (
    PROHIBITED_AUTOMATIC_PARAMETERS,
)
from apollo_timing.inference.pint_solution_artifacts import (
    extract_fitter_parameter_covariance,
)
from apollo_timing.inference.timing_constraints import parameter_bounds_report


MODEL_SELECTION_ARCHITECTURE = "apollo_fixed_path_binary_model_portfolio"

_ELL1_CORE = ("PB", "A1", "TASC", "EPS1", "EPS2")
_ECCENTRIC_CORE = ("PB", "A1", "ECC", "T0", "OM")
_ORBIT_CORE_NAMES = frozenset((*_ELL1_CORE, *_ECCENTRIC_CORE, "FB0"))
_CONVERTIBLE_MODELS = frozenset(
    ("BT", "DD", "DDS", "DDH", "DDK", "ELL1", "ELL1H", "ELL1K")
)


@dataclass(frozen=True)
class BinaryModelHypothesis:
    candidate_id: str
    model_name: str
    hypothesis: str
    extension_parameters: tuple[str, ...] = ()
    requires_ell1_validity: bool = False
    minimum_eccentricity: float = 0.0
    minimum_span_days: float = 0.0


BINARY_MODEL_HYPOTHESES = (
    BinaryModelHypothesis("geometry_bt", "BT", "keplerian_geometry"),
    BinaryModelHypothesis("geometry_dd", "DD", "keplerian_geometry"),
    BinaryModelHypothesis(
        "geometry_ell1",
        "ELL1",
        "keplerian_geometry",
        requires_ell1_validity=True,
    ),
    BinaryModelHypothesis(
        "apsidal_dd",
        "DD",
        "apsidal_motion",
        ("OMDOT",),
        minimum_eccentricity=0.01,
        minimum_span_days=365.0,
    ),
    BinaryModelHypothesis(
        "shapiro_dds",
        "DDS",
        "shapiro_delay",
        ("M2", "SHAPMAX"),
    ),
    BinaryModelHypothesis(
        "shapiro_ddh",
        "DDH",
        "shapiro_delay",
        ("H3", "STIGMA"),
    ),
    BinaryModelHypothesis(
        "gr_ddgr",
        "DDGR",
        "gr_constrained_pk",
        ("MTOT", "M2"),
        minimum_eccentricity=0.01,
        minimum_span_days=365.0,
    ),
    BinaryModelHypothesis(
        "kinematic_ddk",
        "DDK",
        "kopeikin_geometry",
        ("KIN", "KOM"),
        minimum_span_days=365.0,
    ),
    BinaryModelHypothesis(
        "shapiro_ell1h",
        "ELL1H",
        "shapiro_delay",
        ("H3", "STIGMA"),
        requires_ell1_validity=True,
    ),
    BinaryModelHypothesis(
        "secular_ell1k",
        "ELL1K",
        "low_eccentricity_secular",
        ("OMDOT", "LNEDOT"),
        requires_ell1_validity=True,
        minimum_eccentricity=np.finfo(float).tiny,
        minimum_span_days=365.0,
    ),
    BinaryModelHypothesis(
        "change_point_bt_piecewise",
        "BT_PIECEWISE",
        "piecewise_orbit",
        ("T0X_0000", "A1X_0000"),
        minimum_span_days=30.0,
    ),
)


@dataclass(frozen=True)
class BinaryModelSelectionConfig:
    """Criteria for comparing models on one fixed integer path."""

    enabled: bool = True
    enabled_models: tuple[str, ...] = SUPPORTED_BINARY_MODELS
    minimum_delta_bic: float = 10.0
    minimum_block_significance_sigma: float = 5.0
    extension_trigger_reduced_chi2: float = 1.5
    ell1_truncation_fraction_of_toa_error: float = 0.25
    representation_equivalence_delta_chi2: float = 2.0
    minimum_toas_for_extension: int = 24
    minimum_orbital_phase_bins_for_shapiro: int = 6
    orbital_phase_bins: int = 8
    maximum_multistarts_per_hypothesis: int = 8
    minimum_component_mass_solar: float = 0.05
    maximum_component_mass_solar: float = 5.0

    def __post_init__(self) -> None:
        canonical = tuple(canonical_binary_model_name(name) for name in self.enabled_models)
        unknown = sorted(set(canonical) - set(SUPPORTED_BINARY_MODELS))
        if unknown:
            raise ValueError(f"Unknown binary models in selection config: {unknown}.")
        positive = (
            self.minimum_delta_bic,
            self.minimum_block_significance_sigma,
            self.extension_trigger_reduced_chi2,
            self.ell1_truncation_fraction_of_toa_error,
            self.representation_equivalence_delta_chi2,
            self.minimum_component_mass_solar,
            self.maximum_component_mass_solar,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Binary model-selection thresholds must be positive.")
        if self.minimum_component_mass_solar >= self.maximum_component_mass_solar:
            raise ValueError("Binary component-mass bounds are inverted.")
        positive_integers = (
            self.minimum_toas_for_extension,
            self.minimum_orbital_phase_bins_for_shapiro,
            self.orbital_phase_bins,
            self.maximum_multistarts_per_hypothesis,
        )
        if any(int(value) <= 0 for value in positive_integers):
            raise ValueError("Binary model-selection budgets must be positive integers.")
        if self.minimum_orbital_phase_bins_for_shapiro > self.orbital_phase_bins:
            raise ValueError("Required phase-bin coverage exceeds the available bins.")


# Backward-compatible name used by earlier public solver configs.
PhysicalModelSelectionConfig = BinaryModelSelectionConfig


@dataclass(frozen=True)
class BinaryModelSelectionResult:
    fitter: Any
    residuals: Any
    fit_parameters: tuple[str, ...]
    selected_model_name: str
    selected_model_family: str
    selected_candidate_id: str
    portfolio: tuple[dict[str, Any], ...]
    hypothesis_weights: Mapping[str, float]

    @property
    def selected(self) -> bool:
        return self.selected_candidate_id != "baseline"


def registered_model_names() -> tuple[str, ...]:
    return tuple(SUPPORTED_BINARY_MODELS)


def _binary_name(model: Any) -> str:
    return canonical_binary_model_name(model["BINARY"].value)


def _numeric(model: Any, name: str, default: float | None = None) -> float | None:
    if name not in model.params or model[name].value is None:
        return default
    try:
        value = float(model[name].value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _model_family(model_name: str) -> str:
    canonical = canonical_binary_model_name(model_name)
    return BINARY_MODEL_CAPABILITIES[canonical].family


def _eccentricity(model: Any) -> float:
    value = _numeric(model, "ECC", 0.0)
    return max(float(value or 0.0), 0.0)


def _toa_mjds(toas: Any) -> np.ndarray:
    return np.asarray(toas.get_mjds().value, dtype=float)


def _toa_count(toas: Any) -> int:
    return int(len(toas.table))


def _median_toa_error_seconds(toas: Any) -> float:
    errors = np.asarray(toas.get_errors().to_value(u.s), dtype=float)
    finite = errors[np.isfinite(errors) & (errors > 0.0)]
    return float(np.median(finite)) if finite.size else 1.0e-6


def ell1_approximation_is_resolved(
    model: Any,
    toas: Any,
    config: BinaryModelSelectionConfig,
) -> tuple[bool, float, float]:
    """Return whether neglected O(x e^2) structure is below TOA precision."""

    eccentricity = _eccentricity(model)
    a1_seconds = abs(float(_numeric(model, "A1", 0.0) or 0.0))
    truncation_seconds = a1_seconds * eccentricity**2
    tolerance_seconds = (
        float(config.ell1_truncation_fraction_of_toa_error)
        * _median_toa_error_seconds(toas)
    )
    return truncation_seconds <= tolerance_seconds, truncation_seconds, tolerance_seconds


def _orbital_phase_bin_count(model: Any, toas: Any, bins: int) -> int:
    pb = _numeric(model, "PB")
    epoch = _numeric(model, "T0")
    if epoch is None:
        epoch = _numeric(model, "TASC")
    if pb is None or pb <= 0.0 or epoch is None:
        return 0
    phases = np.mod((_toa_mjds(toas) - epoch) / pb, 1.0)
    occupied = np.unique(np.minimum((phases * bins).astype(int), bins - 1))
    return int(len(occupied))


def _kinematic_prerequisites(model: Any) -> tuple[bool, str]:
    px = float(_numeric(model, "PX", 0.0) or 0.0)
    pmra = float(_numeric(model, "PMRA", 0.0) or 0.0)
    pmdec = float(_numeric(model, "PMDEC", 0.0) or 0.0)
    if px <= 0.0:
        return False, "DDK requires a positive measured PX."
    if math.hypot(pmra, pmdec) <= 0.0:
        return False, "DDK requires non-zero proper motion."
    return True, "eligible"


def _hypothesis_eligibility(
    hypothesis: BinaryModelHypothesis,
    *,
    baseline_model: Any,
    baseline_reduced_chi2: float,
    toas: Any,
    observation_span_days: float,
    ell1_valid: bool,
    config: BinaryModelSelectionConfig,
) -> tuple[bool, str]:
    enabled = {canonical_binary_model_name(name) for name in config.enabled_models}
    if hypothesis.model_name not in enabled:
        return False, "model disabled by configuration"
    if hypothesis.requires_ell1_validity and not ell1_valid:
        return False, "ELL1 truncation is resolved by the TOA precision"
    if _eccentricity(baseline_model) < hypothesis.minimum_eccentricity:
        return False, "eccentricity is below the physical sensitivity threshold"
    if observation_span_days < hypothesis.minimum_span_days:
        return False, "observation span is below the physical sensitivity threshold"
    if hypothesis.extension_parameters:
        if _toa_count(toas) < int(config.minimum_toas_for_extension):
            return False, "too few TOAs for an additional physical block"
        if baseline_reduced_chi2 < float(config.extension_trigger_reduced_chi2):
            return False, "baseline fit does not justify physical expansion"
    if hypothesis.hypothesis == "shapiro_delay":
        occupied = _orbital_phase_bin_count(
            baseline_model,
            toas,
            int(config.orbital_phase_bins),
        )
        if occupied < int(config.minimum_orbital_phase_bins_for_shapiro):
            return False, "orbital phase coverage is insufficient for Shapiro delay"
    if hypothesis.model_name == "DDK":
        return _kinematic_prerequisites(baseline_model)
    if hypothesis.model_name == "BT_PIECEWISE":
        mjds = np.sort(_toa_mjds(toas))
        if mjds.size < 2 or not np.any(np.diff(mjds) > 0.0):
            return False, "no data-supported orbital change point exists"
    return True, "eligible"


def _canonical_dd_model(model: Any) -> Any:
    name = _binary_name(model)
    if name == "DD":
        return deepcopy(model)
    if name not in _CONVERTIBLE_MODELS:
        raise ValueError(f"Cannot convert {name} to the canonical DD geometry.")
    if name in {"ELL1", "ELL1H", "ELL1K"}:
        eps1 = float(_numeric(model, "EPS1", 0.0) or 0.0)
        eps2 = float(_numeric(model, "EPS2", 0.0) or 0.0)
        if eps1 == 0.0 and eps2 == 0.0:
            pb, _ = model.pb()
            lines = _non_binary_par_lines(model)
            lines.extend(
                [
                    "BINARY DD",
                    f"PB {float(pb.to_value(u.d)):.18g}",
                    f"A1 {_numeric(model, 'A1'):.18g}",
                    "ECC 0",
                    f"T0 {_numeric(model, 'TASC'):.18g}",
                    "OM 0",
                ]
            )
            return get_model(StringIO("\n".join(lines) + "\n"), allow_tcb=True)
    return convert_binary(deepcopy(model), "DD")


def _is_exactly_circular_ell1(model: Any) -> bool:
    return bool(
        _binary_name(model) in {"ELL1", "ELL1H", "ELL1K"}
        and float(_numeric(model, "EPS1", 0.0) or 0.0) == 0.0
        and float(_numeric(model, "EPS2", 0.0) or 0.0) == 0.0
    )


def _build_circular_eccentric_model(model: Any, target: str) -> Any:
    if target not in {"BT", "DD", "DDS", "DDH", "DDK"}:
        raise ValueError(f"Unsupported circular eccentric-model target: {target}.")
    lines = _non_binary_par_lines(model)
    lines.extend([f"BINARY {target}", *_eccentric_core_lines(model)])
    if target == "DDS":
        lines.extend(["M2 0.3", f"SHAPMAX {-math.log(0.2):.18g}"])
    elif target == "DDH":
        lines.extend(["H3 1e-6", "STIGMA 0.5"])
    elif target == "DDK":
        lines.extend(["KIN 60", "KOM 0"])
    return get_model(StringIO("\n".join(lines) + "\n"), allow_tcb=True)


def _canonical_ell1_model(model: Any) -> Any:
    name = _binary_name(model)
    if name == "ELL1":
        return deepcopy(model)
    if name not in _CONVERTIBLE_MODELS:
        raise ValueError(f"Cannot convert {name} to the canonical ELL1 geometry.")
    return convert_binary(deepcopy(model), "ELL1")


def _non_binary_par_lines(model: Any) -> list[str]:
    binary_keys = {"BINARY"}
    for component_name, component in model.components.items():
        if component_name.startswith("Binary"):
            binary_keys.update(str(name).upper() for name in component.params)
    retained = []
    for raw_line in model.as_parfile().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split()[0].upper()
        if key not in binary_keys:
            retained.append(line)
    return retained


def _eccentric_core_lines(model: Any) -> list[str]:
    dd = _canonical_dd_model(model)
    return [
        f"PB {_numeric(dd, 'PB'):.18g}",
        f"A1 {_numeric(dd, 'A1'):.18g}",
        f"ECC {_numeric(dd, 'ECC'):.18g}",
        f"T0 {_numeric(dd, 'T0'):.18g}",
        f"OM {_numeric(dd, 'OM'):.18g}",
    ]


def _minimum_companion_mass(mass_function: float, pulsar_mass: float = 1.4) -> float:
    if not np.isfinite(mass_function) or mass_function <= 0.0:
        return 0.2
    low, high = 1.0e-3, 10.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        predicted = middle**3 / (pulsar_mass + middle) ** 2
        if predicted < mass_function:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def _mass_function_solar(model: Any) -> float:
    pb_days = float(_numeric(model, "PB", 0.0) or 0.0)
    a1_seconds = float(_numeric(model, "A1", 0.0) or 0.0)
    if pb_days <= 0.0 or a1_seconds <= 0.0:
        return 0.0
    from astropy.constants import G, M_sun, c

    value = (
        4.0
        * math.pi**2
        * (a1_seconds * c.value) ** 3
        / (G.value * (pb_days * 86400.0) ** 2)
        / M_sun.value
    )
    return float(value)


def _build_ddgr_model(model: Any) -> tuple[Any, list[dict[str, float]]]:
    minimum_m2 = _minimum_companion_mass(_mass_function_solar(model))
    m2_values = sorted({max(minimum_m2 * scale, minimum_m2 + 0.02) for scale in (1.05, 1.3, 1.8)})
    seeds = []
    for m2 in m2_values:
        for pulsar_mass in (1.2, 1.4, 2.0):
            seeds.append({"M2": m2, "MTOT": m2 + pulsar_mass})
    initial = seeds[0]
    lines = _non_binary_par_lines(model)
    lines.extend(
        [
            "BINARY DDGR",
            *_eccentric_core_lines(model),
            f"MTOT {initial['MTOT']:.18g}",
            f"M2 {initial['M2']:.18g}",
        ]
    )
    return get_model(StringIO("\n".join(lines) + "\n"), allow_tcb=True), seeds


def _piecewise_change_interval(toas: Any) -> tuple[float, float]:
    mjds = np.sort(_toa_mjds(toas))
    lower_index = max(int(0.2 * len(mjds)), 1)
    upper_index = min(int(0.8 * len(mjds)), len(mjds) - 1)
    eligible_gaps = np.diff(mjds)[lower_index - 1 : upper_index]
    if eligible_gaps.size == 0:
        raise ValueError("No interior TOA gap supports BT_piecewise selection.")
    split_index = lower_index - 1 + int(np.argmax(eligible_gaps))
    split = 0.5 * (mjds[split_index] + mjds[split_index + 1])
    return float(split), float(mjds[-1] + 1.0e-6)


def _build_bt_piecewise_model(model: Any, toas: Any) -> tuple[Any, list[dict[str, float]]]:
    start, finish = _piecewise_change_interval(toas)
    eccentric_model = _canonical_dd_model(model)
    t0 = float(_numeric(eccentric_model, "T0", 0.0) or 0.0)
    a1 = float(_numeric(eccentric_model, "A1", 0.0) or 0.0)
    lines = _non_binary_par_lines(model)
    lines.extend(
        [
            "BINARY BT_piecewise",
            *_eccentric_core_lines(eccentric_model),
            f"XR1_0000 {start:.18g}",
            f"XR2_0000 {finish:.18g}",
            f"T0X_0000 {t0:.18g}",
            f"A1X_0000 {a1:.18g}",
        ]
    )
    return get_model(StringIO("\n".join(lines) + "\n"), allow_tcb=True), [{}]


def _prepare_hypothesis_model(
    baseline_model: Any,
    hypothesis: BinaryModelHypothesis,
    toas: Any,
) -> tuple[Any, list[dict[str, float]]]:
    target = hypothesis.model_name
    if target == "DDGR":
        return _build_ddgr_model(baseline_model)
    if target == "BT_PIECEWISE":
        return _build_bt_piecewise_model(baseline_model, toas)
    if target in {"ELL1", "ELL1H", "ELL1K"}:
        model = _canonical_ell1_model(baseline_model)
    elif _is_exactly_circular_ell1(baseline_model):
        model = _build_circular_eccentric_model(baseline_model, target)
    else:
        model = _canonical_dd_model(baseline_model)

    if target in {"DDS", "DDH", "ELL1H"} and _binary_name(model) != target:
        if "M2" in model.params:
            model["M2"].value = 0.3
        if "SINI" in model.params:
            model["SINI"].value = 0.8
    current = _binary_name(model)
    if current != target:
        conversion_target = "ELL1k" if target == "ELL1K" else target
        if target == "ELL1H":
            model = convert_binary(model, conversion_target, useSTIGMA=True)
        else:
            model = convert_binary(model, conversion_target)
        model.setup()

    if hypothesis.candidate_id == "apsidal_dd":
        return model, [
            {"OMDOT": value} for value in (0.0, -0.01, 0.01, -0.1, 0.1)
        ]
    if target == "DDS":
        return model, [
            {"M2": m2, "SHAPMAX": shapmax}
            for m2 in (0.1, 0.5, 1.4)
            for shapmax in (0.7, 1.6, 3.0)
        ]
    if target in {"DDH", "ELL1H"}:
        return model, [
            {"H3": h3, "STIGMA": stigma}
            for h3 in (1.0e-7, 1.0e-6, 5.0e-6)
            for stigma in (0.2, 0.5, 0.8)
        ]
    if target == "DDK":
        return model, [
            {"KIN": kin, "KOM": kom}
            for kin in (30.0, 60.0, 90.0, 120.0, 150.0)
            for kom in (0.0, 90.0, 180.0, 270.0)
        ]
    if target == "ELL1K":
        return model, [
            {"OMDOT": omdot, "LNEDOT": 0.0}
            for omdot in (0.0, -0.01, 0.01, -0.1, 0.1)
        ]
    return model, [{}]


def _target_fit_parameters(
    baseline_fit_parameters: Sequence[str],
    baseline_model: Any,
    target_model: Any,
    extension_parameters: Sequence[str],
) -> tuple[str, ...]:
    source_names = tuple(str(name).upper() for name in baseline_fit_parameters)
    source_family = _model_family(_binary_name(baseline_model))
    target_family = _model_family(_binary_name(target_model))
    retained = [
        name
        for name in source_names
        if name not in _ORBIT_CORE_NAMES and name in target_model.params
    ]
    if source_family == target_family:
        retained.extend(
            name for name in source_names if name in _ORBIT_CORE_NAMES and name in target_model.params
        )
    else:
        source_core = _ELL1_CORE if source_family == "ell1" else _ECCENTRIC_CORE
        active_source_core = set(source_core).intersection(source_names)
        if active_source_core and not set(source_core).issubset(source_names):
            raise ValueError("Cross-family model selection requires the complete Keplerian core.")
        if active_source_core:
            target_core = _ELL1_CORE if target_family == "ell1" else _ECCENTRIC_CORE
            retained.extend(name for name in target_core if name in target_model.params)
    retained.extend(str(name).upper() for name in extension_parameters)
    prohibited = {str(name).upper() for name in PROHIBITED_AUTOMATIC_PARAMETERS}
    result = tuple(dict.fromkeys(name for name in retained if name in source_names or name not in prohibited))
    if any(name not in target_model.params for name in result):
        missing = [name for name in result if name not in target_model.params]
        raise ValueError(f"Candidate model lacks fit parameters: {missing}.")
    return result


def _freeze_to_contract(model: Any, fit_parameters: Sequence[str]) -> None:
    active = {str(name).upper() for name in fit_parameters}
    for name in model.params:
        parameter = model[name]
        if hasattr(parameter, "frozen"):
            parameter.frozen = name not in active


def _fit_candidate(
    *,
    base_model: Any,
    seeds: Sequence[Mapping[str, float]],
    toas: Any,
    fit_parameters: tuple[str, ...],
    extension_parameters: tuple[str, ...],
    fit_iterations: int,
    maximum_multistarts: int,
) -> tuple[Any, Any, np.ndarray, dict[str, Any]]:
    best: tuple[float, Any, Any, np.ndarray, dict[str, Any]] | None = None
    best_profiled: tuple[float, Any, Any, np.ndarray, dict[str, Any]] | None = None
    failures: list[str] = []
    for seed in tuple(seeds)[: int(maximum_multistarts)]:
        model = deepcopy(base_model)
        _freeze_to_contract(model, fit_parameters)
        try:
            for name, value in seed.items():
                _set_numeric_model_parameter_value(model, name, float(value))
            fitter = WLSFitter(toas, model, track_mode="use_pulse_numbers")
            fitter.fit_toas(maxiter=int(fit_iterations))
            residuals = Residuals(
                toas,
                fitter.model,
                subtract_mean=True,
                use_abs_phase=True,
                track_mode="use_pulse_numbers",
            )
            chi2 = float(residuals.chi2)
            if not np.isfinite(chi2):
                continue
            try:
                covariance = extract_fitter_parameter_covariance(fitter, fit_parameters)
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                covariance = np.full(
                    (len(fit_parameters), len(fit_parameters)),
                    np.nan,
                    dtype=float,
                )
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
            if not extension_parameters:
                continue
            profiled_model = deepcopy(base_model)
            shared_parameters = tuple(
                name for name in fit_parameters if name not in extension_parameters
            )
            _freeze_to_contract(profiled_model, shared_parameters)
            try:
                for name, value in seed.items():
                    _set_numeric_model_parameter_value(
                        profiled_model,
                        name,
                        float(value),
                    )
                profiled_fitter = WLSFitter(
                    toas,
                    profiled_model,
                    track_mode="use_pulse_numbers",
                )
                profiled_fitter.fit_toas(maxiter=int(fit_iterations))
                profiled_residuals = Residuals(
                    toas,
                    profiled_fitter.model,
                    subtract_mean=True,
                    use_abs_phase=True,
                    track_mode="use_pulse_numbers",
                )
                profiled_chi2 = float(profiled_residuals.chi2)
                if not np.isfinite(profiled_chi2):
                    continue
                profiled_covariance = np.full(
                    (len(fit_parameters), len(fit_parameters)),
                    np.nan,
                    dtype=float,
                )
                diagnostics = {
                    "optimizer_mode": "conditional_seed_profile",
                    "joint_parameters_identifiable": False,
                    "joint_fit_error": failures[-1],
                    "profiled_extension_seed": {
                        str(name).upper(): float(value)
                        for name, value in seed.items()
                    },
                }
                if best_profiled is None or profiled_chi2 < best_profiled[0]:
                    best_profiled = (
                        profiled_chi2,
                        profiled_fitter,
                        profiled_residuals,
                        profiled_covariance,
                        diagnostics,
                    )
            except Exception as profile_error:
                failures.append(
                    "conditional profile "
                    f"{type(profile_error).__name__}: {profile_error}"
                )
            continue
        diagnostics = {
            "optimizer_mode": "joint_wls",
            "joint_parameters_identifiable": bool(np.all(np.isfinite(covariance))),
        }
        if not diagnostics["joint_parameters_identifiable"]:
            diagnostics["identifiability_reason"] = (
                "joint fit converged but its covariance was singular or non-finite"
            )
        if best is None or chi2 < best[0]:
            best = (chi2, fitter, residuals, covariance, diagnostics)
    if best is None:
        if best_profiled is not None:
            return (
                best_profiled[1],
                best_profiled[2],
                best_profiled[3],
                best_profiled[4],
            )
        detail = failures[-1] if failures else "no finite chi-square was produced"
        raise RuntimeError(f"All deterministic multistarts failed; last error: {detail}")
    return best[1], best[2], best[3], best[4]


def _block_significance(
    model: Any,
    covariance: np.ndarray,
    fit_parameters: Sequence[str],
    extension_parameters: Sequence[str],
) -> float:
    if not extension_parameters:
        return math.inf
    names = tuple(str(name).upper() for name in fit_parameters)
    indices = [names.index(str(name).upper()) for name in extension_parameters]
    values = np.asarray(
        [float(model[str(name).upper()].value) for name in extension_parameters],
        dtype=float,
    )
    block = covariance[np.ix_(indices, indices)]
    if not np.all(np.isfinite(block)):
        return 0.0
    try:
        wald = float(values @ np.linalg.pinv(block, hermitian=True) @ values)
    except np.linalg.LinAlgError:
        return 0.0
    return math.sqrt(max(wald, 0.0)) if np.isfinite(wald) else 0.0


def _physical_parameters_valid(
    model: Any,
    hypothesis: BinaryModelHypothesis,
    config: BinaryModelSelectionConfig,
) -> tuple[bool, str]:
    minimum_mass = float(config.minimum_component_mass_solar)
    maximum_mass = float(config.maximum_component_mass_solar)
    m2 = _numeric(model, "M2")
    if "M2" in hypothesis.extension_parameters:
        if m2 is None or not minimum_mass <= m2 <= maximum_mass:
            return False, "companion mass lies outside the configured physical support"
    if hypothesis.model_name == "DDGR":
        mtot = _numeric(model, "MTOT")
        if mtot is None or m2 is None or mtot <= m2:
            return False, "DDGR requires positive pulsar mass"
        if not minimum_mass <= mtot - m2 <= maximum_mass:
            return False, "DDGR pulsar mass lies outside the configured support"
    sini = _numeric(model, "SINI")
    if sini is not None and not 0.0 <= sini <= 1.0:
        return False, "SINI lies outside [0, 1]"
    stigma = _numeric(model, "STIGMA")
    if stigma is not None and not 0.0 < stigma < 1.0:
        return False, "STIGMA lies outside (0, 1)"
    if {"H3", "STIGMA"}.issubset(hypothesis.extension_parameters):
        h3_seconds = _numeric(model, "H3")
        if h3_seconds is None or stigma is None:
            return False, "orthometric Shapiro parameters are incomplete"
        orthometric_m2 = h3_seconds / (Tsun.to_value(u.s) * stigma**3)
        if not minimum_mass <= orthometric_m2 <= maximum_mass:
            return False, (
                "orthometric companion mass lies outside the configured physical support"
            )
    shapmax = _numeric(model, "SHAPMAX")
    if shapmax is not None and shapmax < 0.0:
        return False, "SHAPMAX is negative"
    return True, "valid"


def _hypothesis_weights(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    best_by_hypothesis: dict[str, float] = {}
    for record in records:
        if record.get("status") not in {"baseline", "evaluated"}:
            continue
        if record.get("physical_support_valid") is False:
            continue
        bic = float(record["bic"])
        hypothesis = str(record["hypothesis"])
        best_by_hypothesis[hypothesis] = min(
            bic,
            best_by_hypothesis.get(hypothesis, math.inf),
        )
    if not best_by_hypothesis:
        return {}
    minimum = min(best_by_hypothesis.values())
    raw = {
        name: math.exp(-0.5 * min(value - minimum, 1_000.0))
        for name, value in best_by_hypothesis.items()
    }
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def select_binary_model_on_fixed_path(
    *,
    baseline_fitter: Any,
    certified_toas: Any,
    baseline_fit_parameters: Sequence[str],
    observation_span_days: float,
    fit_iterations: int,
    config: BinaryModelSelectionConfig | None = None,
    sky_constraint=None,
    parameter_bounds: Sequence[tuple[str, float, float]] = (),
    optional_absent: Sequence[str] = (),
) -> BinaryModelSelectionResult:
    """Compare every supported PINT binary-model hypothesis on one pulse path."""

    active = config or BinaryModelSelectionConfig()
    baseline_model = baseline_fitter.model
    if not parameter_bounds_report(baseline_model, parameter_bounds, optional_absent=optional_absent)["within_bounds"]:
        raise ValueError("Model comparison baseline is outside the declared parameter bounds.")
    if sky_constraint is not None and not sky_constraint.report(baseline_model)["within_bounds"]:
        raise ValueError("Model comparison baseline is outside the declared sky region.")
    baseline_residuals = Residuals(
        certified_toas,
        baseline_model,
        subtract_mean=True,
        use_abs_phase=True,
        track_mode="use_pulse_numbers",
    )
    baseline_fit = tuple(str(name).upper() for name in baseline_fit_parameters)
    baseline_chi2 = float(baseline_residuals.chi2)
    baseline_reduced = float(baseline_residuals.reduced_chi2)
    toa_count = max(_toa_count(certified_toas), 2)
    baseline_bic = baseline_chi2 + len(baseline_fit) * math.log(toa_count)
    baseline_name = _binary_name(baseline_model)
    baseline_record: dict[str, Any] = {
        "candidate_id": "baseline",
        "model_name": baseline_name,
        "model_family": _model_family(baseline_name),
        "hypothesis": "keplerian_geometry",
        "status": "baseline",
        "selected": True,
        "fit_parameters": baseline_fit,
        "extension_parameters": (),
        "chi2": baseline_chi2,
        "reduced_chi2": baseline_reduced,
        "bic": baseline_bic,
        "delta_bic": 0.0,
        "approximate_log_bayes_factor": 0.0,
        "block_significance_sigma": None,
    }
    records: list[dict[str, Any]] = [baseline_record]
    if not active.enabled:
        return BinaryModelSelectionResult(
            fitter=baseline_fitter,
            residuals=baseline_residuals,
            fit_parameters=baseline_fit,
            selected_model_name=baseline_name,
            selected_model_family=_model_family(baseline_name),
            selected_candidate_id="baseline",
            portfolio=tuple(records),
            hypothesis_weights={"keplerian_geometry": 1.0},
        )

    ell1_valid, ell1_error, ell1_tolerance = ell1_approximation_is_resolved(
        baseline_model,
        certified_toas,
        active,
    )
    fitted_candidates: dict[str, tuple[Any, Any, tuple[str, ...]]] = {}
    for hypothesis in BINARY_MODEL_HYPOTHESES:
        if hypothesis.model_name == baseline_name and not hypothesis.extension_parameters:
            continue
        eligible, reason = _hypothesis_eligibility(
            hypothesis,
            baseline_model=baseline_model,
            baseline_reduced_chi2=baseline_reduced,
            toas=certified_toas,
            observation_span_days=float(observation_span_days),
            ell1_valid=ell1_valid,
            config=active,
        )
        record: dict[str, Any] = {
            "candidate_id": hypothesis.candidate_id,
            "model_name": hypothesis.model_name,
            "model_family": _model_family(hypothesis.model_name),
            "hypothesis": hypothesis.hypothesis,
            "extension_parameters": hypothesis.extension_parameters,
            "status": "ineligible",
            "selected": False,
            "eligibility_reason": reason,
            "ell1_truncation_seconds": ell1_error,
            "ell1_tolerance_seconds": ell1_tolerance,
        }
        if not eligible:
            records.append(record)
            continue
        try:
            candidate_model, seeds = _prepare_hypothesis_model(
                baseline_model,
                hypothesis,
                certified_toas,
            )
            fit_parameters = _target_fit_parameters(
                baseline_fit,
                baseline_model,
                candidate_model,
                hypothesis.extension_parameters,
            )
            missing = [name for name, _, _ in parameter_bounds
                       if name not in optional_absent and (name not in candidate_model.params
                           or candidate_model[name].value is None)]
            if missing:
                record.update(status="constraint_rejected", physical_support_valid=False,
                              physical_support_reason=f"model conversion cannot preserve bounded parameters: {missing}")
                records.append(record)
                continue
            fitter, residuals, covariance, optimizer_diagnostics = _fit_candidate(
                base_model=candidate_model,
                seeds=seeds,
                toas=certified_toas,
                fit_parameters=fit_parameters,
                extension_parameters=hypothesis.extension_parameters,
                fit_iterations=int(fit_iterations),
                maximum_multistarts=int(active.maximum_multistarts_per_hypothesis),
            )
            chi2 = float(residuals.chi2)
            reduced_chi2 = float(residuals.reduced_chi2)
            bic = chi2 + len(fit_parameters) * math.log(toa_count)
            delta_bic = float(baseline_bic - bic)
            significance = _block_significance(
                fitter.model,
                covariance,
                fit_parameters,
                hypothesis.extension_parameters,
            )
            physical_valid, physical_reason = _physical_parameters_valid(
                fitter.model,
                hypothesis,
                active,
            )
            sky_report = sky_constraint.report(fitter.model) if sky_constraint is not None else None
            bound_report = parameter_bounds_report(fitter.model, parameter_bounds, optional_absent=optional_absent)
            if not bound_report["within_bounds"]:
                record.update(status="constraint_rejected", parameter_bounds=bound_report,
                              physical_support_valid=False,
                              physical_support_reason="unconstrained model fit left the requested parameter intervals; no constrained optimum established")
                records.append(record)
                continue
            if sky_report is not None and not sky_report["within_bounds"]:
                record.update(status="constraint_rejected", sky_position_constraint=sky_report,
                              physical_support_valid=False,
                              physical_support_reason="unconstrained model fit left the requested sky region; no constrained optimum established")
                records.append(record)
                continue
            evidence_passed = bool(
                delta_bic >= float(active.minimum_delta_bic)
                and physical_valid
                and (
                    not hypothesis.extension_parameters
                    or significance >= float(active.minimum_block_significance_sigma)
                )
            )
            record.update(
                {
                    "status": "evaluated",
                    "fit_parameters": fit_parameters,
                    "chi2": chi2,
                    "reduced_chi2": reduced_chi2,
                    "bic": bic,
                    "delta_bic": delta_bic,
                    "approximate_log_bayes_factor": 0.5 * delta_bic,
                    "block_significance_sigma": (
                        significance if hypothesis.extension_parameters else None
                    ),
                    "physical_support_valid": physical_valid,
                    "sky_position_constraint": sky_report,
                    "parameter_bounds": bound_report,
                    "physical_support_reason": physical_reason,
                    "evidence_passed": evidence_passed,
                    **optimizer_diagnostics,
                    "fitted_parameters": {
                        name: float(fitter.model[name].value) for name in fit_parameters
                    },
                }
            )
            fitted_candidates[hypothesis.candidate_id] = (
                fitter,
                residuals,
                fit_parameters,
            )
        except Exception as error:
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        records.append(record)

    qualified = [record for record in records if record.get("evidence_passed")]
    selected_record = min(qualified, key=lambda item: float(item["bic"])) if qualified else baseline_record
    if selected_record is baseline_record and ell1_valid:
        ell1_records = [
            record
            for record in records
            if record.get("candidate_id") == "geometry_ell1"
            and record.get("status") == "evaluated"
            and record.get("physical_support_valid", True)
            and float(record["chi2"])
            <= baseline_chi2 + float(active.representation_equivalence_delta_chi2)
        ]
        if ell1_records:
            selected_record = ell1_records[0]
            selected_record["selection_basis"] = "ELL1 approximation valid and statistically equivalent"

    for record in records:
        record["selected"] = record is selected_record
    selected_id = str(selected_record["candidate_id"])
    if selected_id == "baseline":
        selected_fitter = baseline_fitter
        selected_residuals = baseline_residuals
        selected_fit = baseline_fit
    else:
        selected_fitter, selected_residuals, selected_fit = fitted_candidates[selected_id]
    weights = _hypothesis_weights(records)
    return BinaryModelSelectionResult(
        fitter=selected_fitter,
        residuals=selected_residuals,
        fit_parameters=selected_fit,
        selected_model_name=_binary_name(selected_fitter.model),
        selected_model_family=_model_family(_binary_name(selected_fitter.model)),
        selected_candidate_id=selected_id,
        portfolio=tuple(records),
        hypothesis_weights=weights,
    )


__all__ = [
    "BINARY_MODEL_HYPOTHESES",
    "MODEL_SELECTION_ARCHITECTURE",
    "BinaryModelHypothesis",
    "BinaryModelSelectionConfig",
    "BinaryModelSelectionResult",
    "PhysicalModelSelectionConfig",
    "ell1_approximation_is_resolved",
    "registered_model_names",
    "select_binary_model_on_fixed_path",
]
