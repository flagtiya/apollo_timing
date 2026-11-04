from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from astropy import units as u
from pint.simulation import make_fake_toas_fromMJDs

from apollo_timing.backend.pint_backend import (
    _load_pint_model,
    _load_pint_toas,
    _prime_local_ephemeris,
    _set_numeric_model_parameter_value,
    _temporary_clock_override,
)
from apollo_timing.inference.active_timing_contracts import (
    ObservationWindow,
    to_json_ready,
)
from apollo_timing.inference.observation_utility import (
    BayesianObservationDesignConfig,
    PredictiveObservationMode,
    PredictiveObservationScore,
    gaussian_parameter_information,
    score_multi_wrapped_predictive_observation,
    session_contrast_matrix,
    session_wrapped_phase_matrix,
)
from apollo_timing.inference.pint_phase_basis import pint_absolute_phase_values
from apollo_timing.inference.timing_constraints import SkyPositionConstraint, parameter_bounds_report


@dataclass(frozen=True)
class PintPosteriorModeSummary:
    mode_id: str
    par_path: Path
    probability: float
    profile_score: float
    parameter_names: tuple[str, ...]
    covariance_source: str
    sky_position_constraint: dict | None = None
    parameter_bounds: dict | None = None


@dataclass(frozen=True)
class ObservationNeedDiagnosis:
    status: str
    observation_recommended: bool
    current_solution_accepted: bool | None
    fit_accepted: bool | None
    target_reached: bool | None
    best_reduced_chi2: float | None
    posterior_mode_count: int
    significant_mode_count: int
    branch_entropy_nats: float
    effective_branch_count: float
    dominant_branch_probability: float
    retained_portfolio_unique: bool
    portfolio_coverage_certified: bool
    unique_solution: bool
    joint_sequence_evaluated: bool
    joint_resolution_target_reached: bool
    minimum_sessions_to_resolution: int | None
    maximum_expected_branch_resolution_probability: float
    maximum_parameter_information_nats: float
    rationale: str


@dataclass(frozen=True)
class BayesianObservationRecommendation:
    rank: int
    window: ObservationWindow
    session_epochs_mjd: tuple[float, ...]
    net_information_nats: float
    score: PredictiveObservationScore


@dataclass(frozen=True)
class BayesianObservationSequenceRecommendation:
    session_count: int
    windows: tuple[ObservationWindow, ...]
    session_epochs_mjd: tuple[tuple[float, ...], ...]
    score: PredictiveObservationScore
    reaches_resolution_target: bool
    recommendation_status: str
    minimum_sessions_to_resolution: int | None


@dataclass(frozen=True)
class BayesianObservationPlanResult:
    run_output_dir: Path
    output_dir: Path
    input_tim_path: Path
    posterior_modes: tuple[PintPosteriorModeSummary, ...]
    recommendations: tuple[BayesianObservationRecommendation, ...]
    recommended_sequence: BayesianObservationSequenceRecommendation | None
    evaluated_window_count: int
    evaluated_sequence_count: int
    inferred_observatory: str
    inferred_frequency_mhz: float
    inferred_toa_error_us: float
    portfolio_weight_method: str
    diagnosis: ObservationNeedDiagnosis
    qualification_claim: str
    artifacts: Mapping[str, Path]


@dataclass(frozen=True)
class PintObservationPlannerConfig:
    horizon_days: float = 180.0
    baseline_count: int = 10
    orbital_phase_samples: int = 8
    annual_phase_samples: int = 8
    duration_min: float = 20.0
    toas_per_session: int = 8
    maximum_modes: int = 8
    maximum_recommendations: int = 20
    enable_joint_sequence: bool = True
    maximum_joint_sessions: int = 2
    joint_candidate_pool: int = 8
    significant_mode_probability: float = 0.05
    dominant_mode_probability: float = 0.9
    maximum_unique_entropy_nats: float = 0.35
    minimum_parameter_information_nats: float = 1.0
    design: BayesianObservationDesignConfig = field(
        default_factory=BayesianObservationDesignConfig
    )

    def __post_init__(self) -> None:
        positive_floats = (self.horizon_days, self.duration_min)
        positive_integers = (
            self.baseline_count,
            self.orbital_phase_samples,
            self.annual_phase_samples,
            self.toas_per_session,
            self.maximum_modes,
            self.maximum_recommendations,
            self.maximum_joint_sessions,
            self.joint_candidate_pool,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("Observation-planner horizons and durations must be positive.")
        if not all(int(value) > 0 for value in positive_integers):
            raise ValueError("Observation-planner counts must be positive integers.")
        if self.toas_per_session < 2:
            raise ValueError("A recommended session requires at least two TOAs.")
        if not 2 <= self.maximum_joint_sessions <= 3:
            raise ValueError("Joint observation design supports two or three sessions.")
        if self.joint_candidate_pool < self.maximum_joint_sessions:
            raise ValueError("The joint candidate pool is smaller than the sequence length.")
        if not 0.0 < self.significant_mode_probability < 0.5:
            raise ValueError("The significant-mode probability must lie between 0 and 0.5.")
        if not 0.5 < self.dominant_mode_probability < 1.0:
            raise ValueError("The dominant-mode probability must lie between 0.5 and 1.")
        if (
            not np.isfinite(self.maximum_unique_entropy_nats)
            or self.maximum_unique_entropy_nats < 0.0
        ):
            raise ValueError("The unique-solution entropy threshold must be nonnegative.")
        if (
            not np.isfinite(self.minimum_parameter_information_nats)
            or self.minimum_parameter_information_nats < 0.0
        ):
            raise ValueError("The parameter-information threshold must be nonnegative.")


@dataclass
class _LoadedPintMode:
    mode_id: str
    par_path: Path
    model: object
    probability: float
    profile_score: float
    parameter_names: tuple[str, ...]
    covariance: np.ndarray
    covariance_source: str
    sky_position_constraint: dict | None = None
    parameter_bounds: dict | None = None


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _resolve_candidate_path(
    value: Any,
    *,
    search_output: Path,
    candidate_id: str,
) -> Path | None:
    if value:
        path = Path(str(value)).expanduser()
        resolved_values = (path,) if path.is_absolute() else (search_output / path, path)
        for resolved in resolved_values:
            if resolved.is_file():
                return resolved.resolve()
    candidates = (
        search_output / "candidates" / candidate_id / "solution.par",
        search_output / "solution.par",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _collect_candidate_records(
    output: Path,
    *,
    visited: set[Path] | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    root = Path(output).resolve()
    seen = visited if visited is not None else set()
    if root in seen:
        return []
    seen.add(root)
    records: list[tuple[dict[str, Any], Path]] = []
    portfolio_path = root / "candidate_portfolio.json"
    if portfolio_path.is_file():
        payload = _json(portfolio_path)
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("artifacts"), dict) and "chi2" in item:
                    records.append((item, root))
                    continue
                child_value = item.get("search_output")
                if child_value:
                    child = Path(str(child_value)).expanduser()
                    if not child.is_absolute():
                        child = root / child
                    if not child.is_dir() and item.get("mode_id"):
                        child = root / "bootstrap_mode_solutions" / str(item["mode_id"])
                    if child.is_dir():
                        records.extend(_collect_candidate_records(child, visited=seen))
    if records:
        return records
    for certification_path in sorted((root / "candidates").glob("*/certification.json")):
        record = _json(certification_path)
        if isinstance(record, dict) and "chi2" in record and isinstance(
            record.get("artifacts"), dict
        ):
            records.append((record, root))
    if records:
        return records
    certification_path = root / "certification.json"
    if certification_path.is_file() and (root / "solution.par").is_file():
        record = _json(certification_path)
        if isinstance(record, dict) and "chi2" in record:
            record = dict(record)
            record["candidate_id"] = str(record.get("candidate_id", "solution"))
            record["artifacts"] = {
                **dict(record.get("artifacts") or {}),
                "par": str(root / "solution.par"),
            }
            records.append((record, root))
    return records


def _record_covariance(
    record: Mapping[str, Any],
    model: object,
) -> tuple[tuple[str, ...], np.ndarray, str]:
    stored_names = tuple(
        str(name).upper() for name in (record.get("covariance_parameter_names") or ())
    )
    stored = np.asarray(record.get("parameter_covariance", ()), dtype=float)
    if (
        stored_names
        and len(set(stored_names)) == len(stored_names)
        and all(name in model.params for name in stored_names)
        and stored.shape == (len(stored_names), len(stored_names))
        and np.all(np.isfinite(stored))
    ):
        return stored_names, 0.5 * (stored + stored.T), "full_pint_covariance"
    names = tuple(
        str(name).upper()
        for name in (record.get("fit_parameters") or tuple(model.free_params))
        if str(name).upper() in model.params
    )
    retained_names: list[str] = []
    variances: list[float] = []
    for name in names:
        uncertainty = _finite_float(getattr(model[name], "uncertainty_value", None))
        if uncertainty is None or uncertainty <= 0.0:
            continue
        retained_names.append(name)
        variances.append(uncertainty * uncertainty)
    if not retained_names:
        raise RuntimeError(
            "Candidate has neither a stored covariance nor finite par-file uncertainties."
        )
    return tuple(retained_names), np.diag(variances), "parfile_diagonal_uncertainties"


def _profile_probabilities(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Candidate profile scores must be one finite vector.")
    log_weights = -0.5 * (values - float(np.min(values)))
    log_weights -= float(np.max(log_weights))
    weights = np.exp(np.maximum(log_weights, -745.0))
    return weights / float(weights.sum())


def _candidate_tim_identity(path: Path) -> tuple[Any, ...]:
    """Read actual saved observations and full PNs without loading clock files."""
    observations, pulses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0] in {"C", "FORMAT", "MODE"} or fields[0].startswith("#"):
            continue
        try:
            frequency, mjd, error = map(Decimal, fields[1:4])
            pulse = Decimal(fields[fields.index("-pn", 5) + 1])
            if (not all(v.is_finite() for v in (frequency, mjd, error, pulse))
                    or error <= 0 or pulse != pulse.to_integral_value()):
                raise ValueError("Nonfinite observation or noninteger pulse number")
            flags = fields[5:]
            if len(flags) % 2:
                raise ValueError("Malformed saved TOA flags")
            physical_flags = tuple(sorted((k, v) for k, v in zip(flags[::2], flags[1::2], strict=True)
                                          if k != "-pn"))
            observations.append((fields[0], frequency, mjd, error, fields[4].lower(), physical_flags))
            pulses.append(int(pulse))
        except (ValueError, InvalidOperation, IndexError) as exc:
            raise ValueError(f"Invalid candidate TIM for observation planning: {path}") from exc
    if not observations:
        raise ValueError(f"Empty candidate TIM for observation planning: {path}")
    return tuple(observations), tuple(pulse - pulses[0] for pulse in pulses)


def load_apollo_posterior_modes(
    run_output_dir: Path,
    *,
    maximum_modes: int = 8,
) -> tuple[_LoadedPintMode, ...]:
    """Load a finite APOLLO candidate portfolio as a posterior approximation."""

    records = _collect_candidate_records(Path(run_output_dir))
    loaded: list[tuple[dict[str, Any], Path, Path, object, float]] = []
    used_paths: set[Path] = set()
    for record, search_output in records:
        candidate_id = str(record.get("candidate_id", "candidate"))
        par_path = _resolve_candidate_path(
            (record.get("artifacts") or {}).get("par"),
            search_output=search_output,
            candidate_id=candidate_id,
        )
        score = _finite_float(record.get("annealed_total_chi2"))
        if score is None:
            score = _finite_float(record.get("chi2"))
        if par_path is None or score is None or par_path in used_paths:
            continue
        try:
            model = _load_pint_model(par_path)
            _record_covariance(record, model)
        except (OSError, RuntimeError, ValueError):
            continue
        if record.get("sky_position_constraint") is not None:
            region = SkyPositionConstraint.from_report(record["sky_position_constraint"])
            sky_report = region.report(model)
            if not sky_report["within_bounds"]:
                raise ValueError("Observation-planning candidate is outside its original sky constraint.")
            record = {**record, "sky_position_constraint": sky_report}
        if record.get("parameter_bounds") is not None:
            entries = record["parameter_bounds"]["parameters"]
            bound_report = parameter_bounds_report(model, tuple((n, r["lower"], r["upper"]) for n, r in entries.items()),
                optional_absent=tuple(n for n, r in entries.items() if r["status"] == "absent" and r["within_bounds"]))
            if not bound_report["within_bounds"]:
                raise ValueError("Observation-planning candidate is outside its original parameter bounds.")
            record = {**record, "parameter_bounds": bound_report}
        used_paths.add(par_path)
        loaded.append((record, search_output, par_path, model, score))
    if not loaded:
        raise RuntimeError(
            f"No PINT candidate with covariance information was found in {run_output_dir}."
        )
    loaded.sort(key=lambda item: (item[4], str(item[2])))
    distinct = []
    identities = set()
    observed_data = None
    for item in loaded:
        record, search_output, par_path, model, _ = item
        signature = record.get("pulse_path_signature")
        identity = ("unknown_path", str(par_path))
        tim_value = (record.get("artifacts") or {}).get("tim")
        candidate_tim = Path(str(tim_value)) if tim_value else None
        if candidate_tim is not None and not candidate_tim.is_absolute():
            candidate_tim = search_output / candidate_tim
        data_identity = None
        if candidate_tim is not None and candidate_tim.is_file():
            data_identity, signature = _candidate_tim_identity(candidate_tim)
            if observed_data is not None and data_identity != observed_data:
                raise ValueError("Observation-planning candidates do not share the same TOAs and reported errors.")
            observed_data = data_identity
        if isinstance(signature, (list, tuple)) and signature:
            # Compare full TOA paths, not session offsets or candidate file names.
            path = tuple(int(value) for value in signature)
            if any(value != integer for value, integer in zip(signature, path)):
                raise ValueError("Candidate pulse-path signature must contain integers.")
            path = tuple(value - path[0] for value in path)
            tim = search_output / "input.normalized.tim"
            if data_identity is not None or tim.is_file():
                from pint.models.parameter import funcParameter

                # Derive fixed physics from the actual model even if an older
                # candidate record omits the hypothesis metadata.
                fixed = {
                    name: str(model[name].quantity) for name in model.params
                    if name not in model.free_params
                    and name not in {"TZRMJD", "TZRFRQ", "TZRSITE", "START", "FINISH", "NTOA", "CHI2", "CHI2R", "TRES"}
                    and not isinstance(model[name], funcParameter)
                    and model[name].value is not None
                }
                identity = (
                    str(getattr(getattr(model, "BINARY", None), "value", "isolated")),
                    tuple(sorted(record.get("fit_parameters") or model.free_params)),
                    json.dumps(fixed, sort_keys=True),
                    data_identity if data_identity is not None else hashlib.sha256(tim.read_bytes()).hexdigest(), path,
                )
        if identity not in identities:
            identities.add(identity)
            distinct.append(item)
    loaded = distinct
    loaded = loaded[: max(int(maximum_modes), 1)]
    probabilities = _profile_probabilities([item[4] for item in loaded])
    result: list[_LoadedPintMode] = []
    for index, ((record, search_output, par_path, model, score), probability) in enumerate(
        zip(loaded, probabilities, strict=True)
    ):
        names, covariance, covariance_source = _record_covariance(record, model)
        identifier = f"{search_output.name}:{record.get('candidate_id', index)}"
        result.append(
            _LoadedPintMode(
                mode_id=identifier,
                par_path=par_path,
                model=model,
                probability=float(probability),
                profile_score=float(score),
                parameter_names=names,
                covariance=covariance,
                covariance_source=covariance_source,
                sky_position_constraint=record.get("sky_position_constraint"),
                parameter_bounds=record.get("parameter_bounds"),
            )
        )
    return tuple(result)


def _mode_summaries(modes: Sequence[_LoadedPintMode]) -> tuple[PintPosteriorModeSummary, ...]:
    return tuple(
        PintPosteriorModeSummary(
            mode_id=mode.mode_id,
            par_path=mode.par_path,
            probability=mode.probability,
            profile_score=mode.profile_score,
            parameter_names=mode.parameter_names,
            covariance_source=mode.covariance_source,
            sky_position_constraint=mode.sky_position_constraint,
            parameter_bounds=mode.parameter_bounds,
        )
        for mode in modes
    )


def _qualification_state(
    run_output: Path,
) -> tuple[bool | None, bool | None, float | None]:
    certification_path = Path(run_output) / "certification.json"
    records: list[Mapping[str, Any]] = []
    if certification_path.is_file():
        payload = _json(certification_path)
        if isinstance(payload, dict):
            records.append(payload)
    if not records:
        records.extend(record for record, _ in _collect_candidate_records(run_output))
    accepted_values = [record.get("accepted") for record in records if "accepted" in record]
    target_values = [
        record.get("target_reached") for record in records if "target_reached" in record
    ]
    reduced_values = [
        value
        for value in (_finite_float(record.get("reduced_chi2")) for record in records)
        if value is not None
    ]
    accepted = bool(any(accepted_values)) if accepted_values else None
    target_reached = bool(any(target_values)) if target_values else None
    best_reduced_chi2 = min(reduced_values) if reduced_values else None
    return accepted, target_reached, best_reduced_chi2


def _posterior_uniqueness_metrics(
    modes: Sequence[_LoadedPintMode],
    config: PintObservationPlannerConfig,
) -> tuple[float, float, int, bool]:
    probabilities = np.asarray([mode.probability for mode in modes], dtype=float)
    probabilities /= float(probabilities.sum())
    positive = probabilities > 0.0
    entropy = float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
    dominant = float(np.max(probabilities))
    significant_count = int(
        np.sum(probabilities >= config.significant_mode_probability)
    )
    retained_unique = bool(
        dominant >= config.dominant_mode_probability
        and entropy <= config.maximum_unique_entropy_nats
        and significant_count <= 1
    )
    return entropy, dominant, significant_count, retained_unique


def _portfolio_coverage_certified(run_output: Path) -> bool:
    """Return only explicit APOLLO search/posterior coverage certification."""

    keys = {
        "portfolio_coverage_certified",
        "posterior_coverage_certified",
        "search_coverage_certified",
    }

    def contains_explicit_certification(value: Any) -> bool:
        if isinstance(value, Mapping):
            if any(value.get(key) is True for key in keys):
                return True
            return any(contains_explicit_certification(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_explicit_certification(item) for item in value)
        return False

    root = Path(run_output)
    candidates = (
        root / "posterior_coverage.json",
        root / "candidate_portfolio_metadata.json",
        root / "certification.json",
        root / "audit.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = _json(path)
        if contains_explicit_certification(payload):
            return True
    return False


def _diagnose_observation_need(
    *,
    modes: Sequence[_LoadedPintMode],
    recommendations: Sequence[BayesianObservationRecommendation],
    config: PintObservationPlannerConfig,
    current_solution_accepted: bool | None,
    target_reached: bool | None,
    best_reduced_chi2: float | None,
    portfolio_coverage_certified: bool = False,
    recommended_sequence: BayesianObservationSequenceRecommendation | None = None,
) -> ObservationNeedDiagnosis:
    entropy, dominant, significant_count, retained_unique = (
        _posterior_uniqueness_metrics(modes, config)
    )
    maximum_parameter_information = max(
        (item.score.parameter_information_nats for item in recommendations),
        default=0.0,
    )
    maximum_single_resolution = max(
        (
            item.score.expected_branch_resolution_probability
            for item in recommendations
        ),
        default=0.0,
    )
    sequence_resolution = (
        recommended_sequence.score.expected_branch_resolution_probability
        if recommended_sequence is not None
        else 0.0
    )
    maximum_resolution = max(maximum_single_resolution, sequence_resolution)
    unique_solution = bool(
        current_solution_accepted is True
        and target_reached is not False
        and retained_unique
        and portfolio_coverage_certified
    )

    if not retained_unique:
        status = "branch_ambiguity"
        recommended = True
        rationale = (
            "Multiple retained phase-orbit modes have material posterior weight. Individual "
            "PINT fit acceptance does not establish a unique timing solution; future sessions "
            "are ranked by their expected wrapped-phase branch discrimination."
        )
        if recommended_sequence is not None:
            if recommended_sequence.reaches_resolution_target:
                rationale += (
                    f" The shortest evaluated sequence expected to meet the branch-resolution "
                    f"target contains {recommended_sequence.session_count} sessions."
                )
            else:
                rationale += (
                    f" No sequence up to {recommended_sequence.session_count} sessions met the "
                    "configured resolution target; the reported sequence is the most informative "
                    "candidate within the evaluated budget, not a guarantee of uniqueness."
                )
    elif unique_solution:
        status = "qualified_unique_solution_refinement"
        recommended = False
        rationale = (
            "The fit passes certification, the retained posterior meets the uniqueness "
            "criteria, and APOLLO explicitly certifies portfolio coverage. Ranked "
            "windows provide optional forecasts for parameter refinement."
        )
    elif current_solution_accepted is True:
        status = "qualified_fit_portfolio_limited"
        recommended = False
        rationale = (
            "The best PINT fit passes certification and one mode dominates the retained "
            "portfolio. Global uniqueness remains unconfirmed because search/posterior "
            "coverage has not been explicitly certified. Establish portfolio coverage "
            "before treating further observations as the primary remedy."
        )
    elif maximum_parameter_information >= config.minimum_parameter_information_nats:
        status = "sparse_sampling_candidate"
        recommended = True
        rationale = (
            "The retained portfolio is dominated by one mode, but a future session has "
            "substantial expected local parameter information. Sparse sampling is therefore "
            "a plausible limitation; search incompleteness or model mismatch is not ruled out."
        )
    else:
        status = "solver_or_model_limitation_not_excluded"
        recommended = False
        rationale = (
            "The retained posterior shows neither material branch ambiguity nor a strong "
            "forecast gain in parameter information. Check search coverage and model "
            "adequacy first; current evidence does not justify further observations as "
            "the primary remedy."
        )
    return ObservationNeedDiagnosis(
        status=status,
        observation_recommended=recommended,
        current_solution_accepted=current_solution_accepted,
        fit_accepted=current_solution_accepted,
        target_reached=target_reached,
        best_reduced_chi2=best_reduced_chi2,
        posterior_mode_count=len(modes),
        significant_mode_count=significant_count,
        branch_entropy_nats=entropy,
        effective_branch_count=float(math.exp(entropy)),
        dominant_branch_probability=dominant,
        retained_portfolio_unique=retained_unique,
        portfolio_coverage_certified=bool(portfolio_coverage_certified),
        unique_solution=unique_solution,
        joint_sequence_evaluated=recommended_sequence is not None,
        joint_resolution_target_reached=bool(
            recommended_sequence is not None
            and recommended_sequence.reaches_resolution_target
        ),
        minimum_sessions_to_resolution=(
            recommended_sequence.session_count
            if recommended_sequence is not None
            and recommended_sequence.reaches_resolution_target
            else None
        ),
        maximum_expected_branch_resolution_probability=float(maximum_resolution),
        maximum_parameter_information_nats=float(maximum_parameter_information),
        rationale=rationale,
    )


def _dominant_value(values: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        raise ValueError("Cannot infer an observatory from an empty TOA collection.")
    return min(counts, key=lambda item: (-counts[item], item))


def _infer_observation_setup(toas: object) -> tuple[str, float, float, float]:
    epochs = np.asarray(toas.get_mjds().value, dtype=float)
    if len(epochs) == 0 or not np.all(np.isfinite(epochs)):
        raise ValueError("The APOLLO timing file contains no finite TOA epochs.")
    observatory = _dominant_value([str(value) for value in toas.table["obs"]])
    frequencies = np.asarray(toas.table["freq"].quantity.to_value(u.MHz), dtype=float)
    errors = np.asarray(toas.table["error"].quantity.to_value(u.us), dtype=float)
    return (
        observatory,
        float(np.median(frequencies)),
        float(np.median(errors)),
        float(np.max(epochs)),
    )


def _resolve_solution_tim(run_output: Path) -> Path:
    direct = Path(run_output) / "solution.tim"
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(Path(run_output).glob("**/solution.tim"))
    if not candidates:
        raise RuntimeError(f"No APOLLO solution.tim was found in {run_output}.")
    return candidates[0].resolve()


def _orbital_period_days(model: object) -> float:
    if "PB" in model.params and _finite_float(model["PB"].value) not in (None, 0.0):
        return float(model["PB"].value)
    if "FB0" in model.params and _finite_float(model["FB0"].value) not in (None, 0.0):
        return float(1.0 / (float(model["FB0"].value) * 86400.0))
    raise ValueError("Observation design requires PB or FB0 in the timing model.")


def generate_adaptive_observation_windows(
    *,
    last_observation_mjd: float,
    orbital_period_days: float,
    config: PintObservationPlannerConfig,
    observatory: str,
    frequency_mhz: float,
    toa_error_us: float,
) -> tuple[ObservationWindow, ...]:
    """Generate multi-baseline windows while resolving binary orbital phase."""

    last = float(last_observation_mjd)
    period = float(orbital_period_days)
    if not np.isfinite(last) or not np.isfinite(period) or period <= 0.0:
        raise ValueError("Window generation requires a finite epoch and positive period.")
    minimum_baseline = min(max(min(period, 1.0), 1.0e-6), config.horizon_days)
    baselines = np.geomspace(
        minimum_baseline,
        config.horizon_days,
        num=max(int(config.baseline_count), 1),
    )
    epochs = {
        round(
            last + float(baseline) + period * phase_index / config.orbital_phase_samples,
            12,
        )
        for baseline in baselines
        for phase_index in range(config.orbital_phase_samples)
        if baseline + period * phase_index / config.orbital_phase_samples
        <= config.horizon_days + 1.0e-12
    }
    return tuple(
        ObservationWindow(
            epoch_mjd=epoch,
            duration_min=float(config.duration_min),
            toa_count=int(config.toas_per_session),
            obs=observatory,
            freq_mhz=float(frequency_mhz),
            toa_error_us=float(toa_error_us),
        )
        for epoch in sorted(epochs)
    )


def generate_isolated_observation_windows(
    *,
    last_observation_mjd: float,
    config: PintObservationPlannerConfig,
    observatory: str,
    frequency_mhz: float,
    toa_error_us: float,
) -> tuple[ObservationWindow, ...]:
    """Generate isolated-source windows across spin and annual timescales."""

    last = float(last_observation_mjd)
    if not np.isfinite(last):
        raise ValueError("Window generation requires a finite epoch.")
    horizon = float(config.horizon_days)
    minimum_baseline = min(1.0, horizon)
    logarithmic_offsets = np.geomspace(
        minimum_baseline,
        horizon,
        num=max(int(config.baseline_count), 1),
    )
    tropical_year_days = 365.2425
    annual_spacing = tropical_year_days / int(config.annual_phase_samples)
    annual_offsets = np.arange(
        annual_spacing,
        horizon + 1.0e-12,
        annual_spacing,
    )
    offsets = {
        round(float(offset), 12)
        for offset in np.concatenate((logarithmic_offsets, annual_offsets))
        if 0.0 < float(offset) <= horizon + 1.0e-12
    }
    return tuple(
        ObservationWindow(
            epoch_mjd=last + offset,
            duration_min=float(config.duration_min),
            toa_count=int(config.toas_per_session),
            obs=observatory,
            freq_mhz=float(frequency_mhz),
            toa_error_us=float(toa_error_us),
        )
        for offset in sorted(offsets)
    )


def generate_model_adapted_observation_windows(
    *,
    model: object,
    last_observation_mjd: float,
    config: PintObservationPlannerConfig,
    observatory: str,
    frequency_mhz: float,
    toa_error_us: float,
) -> tuple[ObservationWindow, ...]:
    """Generate default windows on the physical timescales of the PINT model."""

    parameters = set(getattr(model, "params", ()))
    binary_declared = False
    if "BINARY" in parameters:
        binary_name = getattr(model["BINARY"], "value", None)
        binary_declared = (
            bool(str(binary_name).strip()) if binary_name is not None else False
        )
    has_orbital_frequency = any(name in parameters for name in ("PB", "FB0"))
    if binary_declared or has_orbital_frequency:
        return generate_adaptive_observation_windows(
            last_observation_mjd=last_observation_mjd,
            orbital_period_days=_orbital_period_days(model),
            config=config,
            observatory=observatory,
            frequency_mhz=frequency_mhz,
            toa_error_us=toa_error_us,
        )
    return generate_isolated_observation_windows(
        last_observation_mjd=last_observation_mjd,
        config=config,
        observatory=observatory,
        frequency_mhz=frequency_mhz,
        toa_error_us=toa_error_us,
    )


def generate_uniform_observation_windows(
    *,
    start_mjd: float,
    end_mjd: float,
    cadence_days: float,
    duration_min: float,
    toas_per_session: int,
    observatory: str,
    frequency_mhz: float,
    toa_error_us: float,
) -> tuple[ObservationWindow, ...]:
    start = float(start_mjd)
    end = float(end_mjd)
    cadence = float(cadence_days)
    if not all(np.isfinite(value) for value in (start, end, cadence)):
        raise ValueError("Explicit observation-window limits must be finite.")
    if end < start or cadence <= 0.0:
        raise ValueError("Explicit windows require end >= start and positive cadence.")
    step_ratio = (end - start) / cadence
    nearest_step = int(round(step_ratio))
    if math.isclose(step_ratio, nearest_step, rel_tol=0.0, abs_tol=1.0e-8):
        maximum_step = nearest_step
    else:
        maximum_step = int(math.floor(step_ratio))
    return tuple(
        ObservationWindow(
            epoch_mjd=float(start + index * cadence),
            duration_min=float(duration_min),
            toa_count=int(toas_per_session),
            obs=observatory,
            freq_mhz=float(frequency_mhz),
            toa_error_us=float(toa_error_us),
        )
        for index in range(maximum_step + 1)
    )


def _window_epochs(window: ObservationWindow) -> np.ndarray:
    duration_days = float(window.duration_min) / 1440.0
    return np.linspace(
        float(window.epoch_mjd) - 0.5 * duration_days,
        float(window.epoch_mjd) + 0.5 * duration_days,
        num=int(window.toa_count),
    )


def _phase_jacobian(
    model: object,
    toas: object,
    parameter_names: Sequence[str],
    covariance: np.ndarray,
) -> np.ndarray:
    names = tuple(parameter_names)
    result = np.empty((len(toas.table), len(names)), dtype=float)
    for index, name in enumerate(names):
        sigma = math.sqrt(max(float(covariance[index, index]), 0.0))
        value = float(model[name].value)
        spacing = max(abs(float(np.spacing(value))) * 1024.0, np.finfo(float).tiny)
        step = max(0.25 * sigma, spacing)
        plus_model = deepcopy(model)
        minus_model = deepcopy(model)
        _set_numeric_model_parameter_value(plus_model, name, value + step)
        _set_numeric_model_parameter_value(minus_model, name, value - step)
        plus = pint_absolute_phase_values(plus_model, toas)
        minus = pint_absolute_phase_values(minus_model, toas)
        result[:, index] = np.asarray((plus - minus) / (2.0 * step), dtype=float)
    return result


def _predictive_mode_for_window(
    *,
    mode: _LoadedPintMode,
    template_phase: np.ndarray,
    toas: object,
    wrapped_transform: np.ndarray,
    contrast: np.ndarray,
) -> PredictiveObservationMode:
    if mode.sky_position_constraint and mode.sky_position_constraint["active_bound"]:
        raise ValueError("A sky-boundary mode needs a truncated predictive distribution; local Gaussian planning is not valid.")
    bounds = getattr(mode, "parameter_bounds", None)
    if bounds and any(r["active_bound"] for r in bounds["parameters"].values()):
        raise ValueError("A parameter-boundary mode needs a truncated predictive distribution; local Gaussian planning is not valid.")
    phase = pint_absolute_phase_values(mode.model, toas)
    mean = wrapped_transform @ np.asarray(phase - template_phase, dtype=float)
    raw_design = _phase_jacobian(
        mode.model,
        toas,
        mode.parameter_names,
        mode.covariance,
    )
    scaled_errors = mode.model.scaled_toa_uncertainty(toas).to_value(u.s)
    spin_frequency = float(mode.model["F0"].value)
    phase_variances = np.square(np.asarray(scaled_errors, dtype=float) * spin_frequency)
    raw_measurement = np.diag(phase_variances)
    contrast_design = contrast @ raw_design
    contrast_measurement = contrast @ raw_measurement @ contrast.T
    parameter_information = gaussian_parameter_information(
        mode.covariance,
        contrast_design,
        contrast_measurement,
    )
    wrapped_design = wrapped_transform @ raw_design
    wrapped_measurement = wrapped_transform @ raw_measurement @ wrapped_transform.T
    predictive = (
        wrapped_measurement
        + wrapped_design @ mode.covariance @ wrapped_design.T
    )
    return PredictiveObservationMode(
        mode_id=mode.mode_id,
        probability=mode.probability,
        mean=mean,
        covariance=predictive,
        parameter_information_nats=parameter_information,
    )


def _block_diagonal(matrices: Sequence[np.ndarray]) -> np.ndarray:
    rows = sum(matrix.shape[0] for matrix in matrices)
    columns = sum(matrix.shape[1] for matrix in matrices)
    result = np.zeros((rows, columns), dtype=float)
    row_offset = 0
    column_offset = 0
    for matrix in matrices:
        row_stop = row_offset + matrix.shape[0]
        column_stop = column_offset + matrix.shape[1]
        result[row_offset:row_stop, column_offset:column_stop] = matrix
        row_offset = row_stop
        column_offset = column_stop
    return result


def _score_window_sequence(
    *,
    modes: Sequence[_LoadedPintMode],
    windows: Sequence[ObservationWindow],
    design_config: BayesianObservationDesignConfig,
    clock_override_path: Path | None,
) -> tuple[PredictiveObservationScore, tuple[tuple[float, ...], ...]]:
    sequence = tuple(sorted(windows, key=lambda item: item.epoch_mjd))
    if not sequence:
        raise ValueError("Observation sequence scoring requires at least one session.")
    observatories = {str(window.obs) for window in sequence}
    if len(observatories) != 1:
        raise ValueError("Joint observation sequences require one observatory.")
    epoch_groups = tuple(_window_epochs(window) for window in sequence)
    all_epochs = np.concatenate(epoch_groups)
    frequencies = np.concatenate(
        [np.full(len(epochs), float(window.freq_mhz)) for window, epochs in zip(sequence, epoch_groups, strict=True)]
    )
    errors = np.concatenate(
        [np.full(len(epochs), float(window.toa_error_us)) for window, epochs in zip(sequence, epoch_groups, strict=True)]
    )
    template = modes[0].model
    ephem = template["EPHEM"].value if "EPHEM" in template.params else "DE421"
    _prime_local_ephemeris(ephem)
    with _temporary_clock_override(clock_override_path):
        toas = make_fake_toas_fromMJDs(
            all_epochs,
            model=template,
            freq=frequencies * u.MHz,
            obs=next(iter(observatories)),
            error=errors * u.us,
            add_noise=False,
            include_bipm=False,
            subtract_mean=False,
        )
    wrapped_blocks = tuple(
        session_wrapped_phase_matrix(len(epochs)) for epochs in epoch_groups
    )
    contrast_blocks = tuple(
        session_contrast_matrix(len(epochs)) for epochs in epoch_groups
    )
    wrapped_transform = _block_diagonal(wrapped_blocks)
    contrast = _block_diagonal(contrast_blocks)
    wrapped_indices: list[int] = []
    row_offset = 0
    for block in wrapped_blocks:
        wrapped_indices.append(row_offset)
        row_offset += block.shape[0]
    template_phase = pint_absolute_phase_values(template, toas)
    predictive_modes = tuple(
        _predictive_mode_for_window(
            mode=mode,
            template_phase=template_phase,
            toas=toas,
            wrapped_transform=wrapped_transform,
            contrast=contrast,
        )
        for mode in modes
    )
    score = score_multi_wrapped_predictive_observation(
        predictive_modes,
        wrapped_indices,
        design_config,
    )
    return score, tuple(
        tuple(float(value) for value in epochs) for epochs in epoch_groups
    )


def _score_window(
    *,
    modes: Sequence[_LoadedPintMode],
    window: ObservationWindow,
    design_config: BayesianObservationDesignConfig,
    clock_override_path: Path | None,
) -> tuple[PredictiveObservationScore, tuple[float, ...]]:
    score, epoch_groups = _score_window_sequence(
        modes=modes,
        windows=(window,),
        design_config=design_config,
        clock_override_path=clock_override_path,
    )
    return score, epoch_groups[0]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(to_json_ready(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _write_recommendation_csv(
    path: Path,
    recommendations: Sequence[BayesianObservationRecommendation],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "rank",
                "epoch_mjd",
                "duration_min",
                "toa_count",
                "observatory",
                "frequency_mhz",
                "toa_error_us",
                "net_information_nats",
                "total_information_nats",
                "branch_information_nats",
                "parameter_information_nats",
                "expected_branch_resolution_probability",
                "expected_max_branch_probability",
                "expected_branch_entropy_nats",
            ),
        )
        writer.writeheader()
        for item in recommendations:
            writer.writerow(
                {
                    "rank": item.rank,
                    "epoch_mjd": item.window.epoch_mjd,
                    "duration_min": item.window.duration_min,
                    "toa_count": item.window.toa_count,
                    "observatory": item.window.obs,
                    "frequency_mhz": item.window.freq_mhz,
                    "toa_error_us": item.window.toa_error_us,
                    "net_information_nats": item.net_information_nats,
                    "total_information_nats": item.score.total_information_nats,
                    "branch_information_nats": item.score.branch_information_nats,
                    "parameter_information_nats": item.score.parameter_information_nats,
                    "expected_branch_resolution_probability": (
                        item.score.expected_branch_resolution_probability
                    ),
                    "expected_max_branch_probability": (
                        item.score.expected_max_branch_probability
                    ),
                    "expected_branch_entropy_nats": item.score.expected_branch_entropy_nats,
                }
            )
    return path


def _write_sequence_csv(
    path: Path,
    recommendation: BayesianObservationSequenceRecommendation | None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "session_index",
                "epoch_mjd",
                "duration_min",
                "toa_count",
                "observatory",
                "frequency_mhz",
                "toa_error_us",
                "joint_branch_information_nats",
                "joint_resolution_probability",
                "joint_expected_max_branch_probability",
                "reaches_resolution_target",
                "recommendation_status",
                "minimum_sessions_to_resolution",
            ),
        )
        writer.writeheader()
        if recommendation is not None:
            for index, window in enumerate(recommendation.windows, start=1):
                writer.writerow(
                    {
                        "session_index": index,
                        "epoch_mjd": window.epoch_mjd,
                        "duration_min": window.duration_min,
                        "toa_count": window.toa_count,
                        "observatory": window.obs,
                        "frequency_mhz": window.freq_mhz,
                        "toa_error_us": window.toa_error_us,
                        "joint_branch_information_nats": (
                            recommendation.score.branch_information_nats
                        ),
                        "joint_resolution_probability": (
                            recommendation.score.expected_branch_resolution_probability
                        ),
                        "joint_expected_max_branch_probability": (
                            recommendation.score.expected_max_branch_probability
                        ),
                        "reaches_resolution_target": (
                            recommendation.reaches_resolution_target
                        ),
                        "recommendation_status": recommendation.recommendation_status,
                        "minimum_sessions_to_resolution": (
                            recommendation.minimum_sessions_to_resolution
                        ),
                    }
                )
    return path


def _write_recommendation_plot(
    path: Path,
    recommendations: Sequence[BayesianObservationRecommendation],
) -> Path:
    ordered = sorted(recommendations, key=lambda item: item.window.epoch_mjd)
    epochs = np.asarray([item.window.epoch_mjd for item in ordered], dtype=float)
    branch_ambiguous = any(
        item.score.effective_branch_count_before > 1.0 + 1.0e-6
        for item in ordered
    )
    information = np.asarray(
        [
            (
                item.score.branch_information_nats
                if branch_ambiguous
                else item.score.total_information_nats
            )
            for item in ordered
        ]
    )
    resolution = np.asarray(
        [item.score.expected_branch_resolution_probability for item in ordered]
    )
    figure = Figure(figsize=(10.5, 5.5))
    FigureCanvasAgg(figure)
    information_axis = figure.subplots()
    information_axis.scatter(epochs, information, color="#176B87", s=22)
    information_axis.set_xlabel("Candidate session center (MJD)")
    information_axis.set_ylabel(
        (
            "Expected branch information (nats)"
            if branch_ambiguous
            else "Expected information (nats)"
        ),
        color="#176B87",
    )
    information_axis.tick_params(axis="y", labelcolor="#176B87")
    information_axis.grid(True, alpha=0.25)
    if branch_ambiguous:
        resolution_axis = information_axis.twinx()
        resolution_axis.scatter(epochs, resolution, marker="x", color="#B33A3A", s=26)
        resolution_axis.set_ylabel("Branch-resolution probability", color="#B33A3A")
        resolution_axis.tick_params(axis="y", labelcolor="#B33A3A")
        resolution_axis.set_ylim(-0.02, 1.02)
    else:
        information_axis.set_title(
            "No retained branch ambiguity; information gain is parameter-only"
        )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    figure.clear()
    return path


def _write_summary(
    path: Path,
    result: BayesianObservationPlanResult,
) -> Path:
    lines = [
        "# APOLLO Bayesian Observation Plan",
        "",
        f"- Posterior modes: {len(result.posterior_modes)}",
        f"- Evaluated windows: {result.evaluated_window_count}",
        f"- Evaluated joint sequences: {result.evaluated_sequence_count}",
        f"- Observatory: `{result.inferred_observatory}`",
        f"- Frequency: {result.inferred_frequency_mhz:.3f} MHz",
        f"- Assumed TOA error: {result.inferred_toa_error_us:.3f} us",
        f"- Diagnostic status: `{result.diagnosis.status}`",
        f"- PINT fit accepted: `{str(result.diagnosis.fit_accepted).lower()}`",
        f"- Retained portfolio unique: `{str(result.diagnosis.retained_portfolio_unique).lower()}`",
        f"- Portfolio coverage certified: `{str(result.diagnosis.portfolio_coverage_certified).lower()}`",
        f"- Unique solution: `{str(result.diagnosis.unique_solution).lower()}`",
        f"- Minimum sessions to expected resolution: {result.diagnosis.minimum_sessions_to_resolution}",
        f"- Maximum expected branch-resolution probability: {result.diagnosis.maximum_expected_branch_resolution_probability:.6g}",
        f"- Observation recommended: `{str(result.diagnosis.observation_recommended).lower()}`",
        f"- Effective retained branch count: {result.diagnosis.effective_branch_count:.4g}",
        f"- Diagnostic rationale: {result.diagnosis.rationale}",
        "- Forecast: wrapped-phase branch resolution and local parameter information, "
        "conditional on the retained finite portfolio. Rerun APOLLO/PINT after the "
        "observation.",
        "",
        "| Rank | MJD | Total info (nats) | Branch info | Parameter info | Branch resolution |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result.recommendations[:10]:
        lines.append(
            f"| {item.rank} | {item.window.epoch_mjd:.9f} | "
            f"{item.score.total_information_nats:.6g} | "
            f"{item.score.branch_information_nats:.6g} | "
            f"{item.score.parameter_information_nats:.6g} | "
            f"{item.score.expected_branch_resolution_probability:.3f} |"
        )
    if result.recommended_sequence is not None:
        sequence = result.recommended_sequence
        lines.extend(
            [
                "",
                "## Joint session sequence",
                "",
                f"- Candidate sequence length: {sequence.session_count}",
                "- Candidate MJDs: "
                + ", ".join(f"{window.epoch_mjd:.9f}" for window in sequence.windows),
                f"- Expected branch information: {sequence.score.branch_information_nats:.6g} nats",
                f"- Expected branch-resolution probability: {sequence.score.expected_branch_resolution_probability:.6g}",
                f"- Reaches configured resolution target: `{str(sequence.reaches_resolution_target).lower()}`",
                f"- Status: `{sequence.recommendation_status}`",
                f"- Minimum sessions to expected resolution: {sequence.minimum_sessions_to_resolution}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _observation_sort_key(
    item: tuple[ObservationWindow, PredictiveObservationScore, tuple[float, ...]],
    *,
    branch_ambiguous: bool,
) -> tuple[float, ...]:
    window, score, _ = item
    if branch_ambiguous:
        return (
            -score.expected_branch_resolution_probability,
            -score.branch_information_nats,
            -score.expected_max_branch_probability,
            float(window.cost),
            -score.parameter_information_nats,
            float(window.epoch_mjd),
        )
    return (
        -(score.total_information_nats - float(window.cost)),
        -score.expected_branch_resolution_probability,
        float(window.epoch_mjd),
    )


def _sequence_sort_key(
    item: BayesianObservationSequenceRecommendation,
) -> tuple[float, ...]:
    return (
        -item.score.expected_branch_resolution_probability,
        -item.score.branch_information_nats,
        -item.score.expected_max_branch_probability,
        sum(float(window.cost) for window in item.windows),
        item.session_count,
        *(float(window.epoch_mjd) for window in item.windows),
    )


def _recommend_joint_sequence(
    *,
    modes: Sequence[_LoadedPintMode],
    scored_windows: Sequence[
        tuple[ObservationWindow, PredictiveObservationScore, tuple[float, ...]]
    ],
    config: PintObservationPlannerConfig,
    clock_override_path: Path | None,
) -> tuple[BayesianObservationSequenceRecommendation | None, int]:
    pool = tuple(
        item[0] for item in scored_windows[: int(config.joint_candidate_pool)]
    )
    evaluated = 0
    best: BayesianObservationSequenceRecommendation | None = None
    for session_count in range(2, int(config.maximum_joint_sessions) + 1):
        stage: list[BayesianObservationSequenceRecommendation] = []
        for windows in combinations(pool, session_count):
            if len({str(window.obs) for window in windows}) != 1:
                continue
            score, epoch_groups = _score_window_sequence(
                modes=modes,
                windows=windows,
                design_config=config.design,
                clock_override_path=clock_override_path,
            )
            evaluated += 1
            ordered_windows = tuple(sorted(windows, key=lambda item: item.epoch_mjd))
            reaches_target = bool(
                score.expected_branch_resolution_probability
                >= config.design.branch_resolution_probability
            )
            stage.append(
                BayesianObservationSequenceRecommendation(
                    session_count=session_count,
                    windows=ordered_windows,
                    session_epochs_mjd=epoch_groups,
                    score=score,
                    reaches_resolution_target=reaches_target,
                    recommendation_status=(
                        "resolution_target_reached"
                        if reaches_target
                        else "resolution_target_not_reached_within_budget"
                    ),
                    minimum_sessions_to_resolution=(
                        session_count if reaches_target else None
                    ),
                )
            )
        if not stage:
            continue
        stage.sort(key=_sequence_sort_key)
        best = stage[0] if best is None else min(best, stage[0], key=_sequence_sort_key)
        if stage[0].reaches_resolution_target:
            return stage[0], evaluated
    return best, evaluated


def plan_apollo_observations(
    *,
    run_output_dir: Path,
    output_dir: Path | None = None,
    windows: Sequence[ObservationWindow] | None = None,
    config: PintObservationPlannerConfig | None = None,
    observatory: str | None = None,
    frequency_mhz: float | None = None,
    toa_error_us: float | None = None,
    clock_override_path: Path | None = None,
) -> BayesianObservationPlanResult:
    """Rank future observing sessions from an APOLLO candidate portfolio."""

    active = config or PintObservationPlannerConfig()
    run_output = Path(run_output_dir).resolve()
    destination = Path(output_dir or run_output / "observation_plan").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    modes = load_apollo_posterior_modes(
        run_output,
        maximum_modes=active.maximum_modes,
    )
    _, _, _, retained_portfolio_unique = _posterior_uniqueness_metrics(
        modes,
        active,
    )
    branch_ambiguous = not retained_portfolio_unique
    tim_path = _resolve_solution_tim(run_output)
    current_toas = _load_pint_toas(
        tim_path,
        seed_model=modes[0].model,
        clock_override_path=clock_override_path,
    )
    inferred_obs, inferred_frequency, inferred_error, last_mjd = _infer_observation_setup(
        current_toas
    )
    selected_obs = str(observatory or inferred_obs)
    selected_frequency = float(frequency_mhz or inferred_frequency)
    selected_error = float(toa_error_us or inferred_error)
    candidate_windows = tuple(windows or ())
    if not candidate_windows:
        candidate_windows = generate_model_adapted_observation_windows(
            model=modes[0].model,
            last_observation_mjd=last_mjd,
            config=active,
            observatory=selected_obs,
            frequency_mhz=selected_frequency,
            toa_error_us=selected_error,
        )
    normalized_windows = tuple(
        ObservationWindow(
            epoch_mjd=float(window.epoch_mjd),
            duration_min=float(window.duration_min),
            toa_count=int(window.toa_count),
            obs=str(window.obs if window.obs is not None else selected_obs),
            freq_mhz=float(
                window.freq_mhz if window.freq_mhz is not None else selected_frequency
            ),
            toa_error_us=float(
                window.toa_error_us if window.toa_error_us is not None else selected_error
            ),
            cost=float(window.cost),
        )
        for window in candidate_windows
    )
    if not normalized_windows:
        raise ValueError("Observation planning generated no candidate session windows.")
    for window in normalized_windows:
        values = (
            window.epoch_mjd,
            window.duration_min,
            window.freq_mhz,
            window.toa_error_us,
            window.cost,
        )
        if not all(value is not None and np.isfinite(value) for value in values):
            raise ValueError("Every candidate observation window must be finite.")
        if (
            window.duration_min <= 0.0
            or window.toa_count < 2
            or window.freq_mhz <= 0.0
            or window.toa_error_us <= 0.0
            or window.cost < 0.0
        ):
            raise ValueError("Candidate observation-window scales are invalid.")
    scored: list[tuple[ObservationWindow, PredictiveObservationScore, tuple[float, ...]]] = []
    for window in normalized_windows:
        score, epochs = _score_window(
            modes=modes,
            window=window,
            design_config=active.design,
            clock_override_path=clock_override_path,
        )
        scored.append((window, score, epochs))
    scored.sort(
        key=lambda item: _observation_sort_key(
            item,
            branch_ambiguous=branch_ambiguous,
        )
    )
    recommended_sequence = None
    evaluated_sequence_count = 0
    maximum_single_resolution = max(
        (item[1].expected_branch_resolution_probability for item in scored),
        default=0.0,
    )
    if (
        branch_ambiguous
        and active.enable_joint_sequence
        and maximum_single_resolution < active.design.branch_resolution_probability
    ):
        recommended_sequence, evaluated_sequence_count = _recommend_joint_sequence(
            modes=modes,
            scored_windows=scored,
            config=active,
            clock_override_path=clock_override_path,
        )
    recommendations = tuple(
        BayesianObservationRecommendation(
            rank=rank,
            window=window,
            session_epochs_mjd=epochs,
            net_information_nats=float(score.total_information_nats - window.cost),
            score=score,
        )
        for rank, (window, score, epochs) in enumerate(
            scored[: active.maximum_recommendations],
            start=1,
        )
    )
    accepted, target_reached, best_reduced_chi2 = _qualification_state(run_output)
    coverage_certified = _portfolio_coverage_certified(run_output)
    diagnosis = _diagnose_observation_need(
        modes=modes,
        recommendations=recommendations,
        config=active,
        current_solution_accepted=accepted,
        target_reached=target_reached,
        best_reduced_chi2=best_reduced_chi2,
        portfolio_coverage_certified=coverage_certified,
        recommended_sequence=recommended_sequence,
    )
    json_path = destination / "observation_recommendations.json"
    csv_path = destination / "observation_recommendations.csv"
    sequence_csv_path = destination / "observation_sequence.csv"
    png_path = destination / "observation_recommendations.png"
    summary_path = destination / "README.md"
    artifacts = {
        "json": json_path,
        "csv": csv_path,
        "sequence_csv": sequence_csv_path,
        "plot": png_path,
        "summary": summary_path,
    }
    result = BayesianObservationPlanResult(
        run_output_dir=run_output,
        output_dir=destination,
        input_tim_path=tim_path,
        posterior_modes=_mode_summaries(modes),
        recommendations=recommendations,
        recommended_sequence=recommended_sequence,
        evaluated_window_count=len(scored),
        evaluated_sequence_count=evaluated_sequence_count,
        inferred_observatory=selected_obs,
        inferred_frequency_mhz=selected_frequency,
        inferred_toa_error_us=selected_error,
        portfolio_weight_method=(
            "normalized exp(-0.5 * delta profile objective); finite-profile "
            "approximation, not marginal evidence"
        ),
        diagnosis=diagnosis,
        qualification_claim=(
            "Forecasts of expected wrapped-phase branch resolution and local "
            "continuous-parameter information are conditional on the retained finite "
            "posterior portfolio. PINT fit acceptance alone does not establish a unique "
            "solution; global uniqueness also requires explicit search/posterior "
            "coverage certification."
        ),
        artifacts=artifacts,
    )
    _write_json(json_path, result)
    _write_recommendation_csv(csv_path, recommendations)
    _write_sequence_csv(sequence_csv_path, recommended_sequence)
    _write_recommendation_plot(png_path, recommendations)
    _write_summary(summary_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank future sessions by APOLLO posterior-predictive information gain."
    )
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--clock-override", type=Path, default=None)
    parser.add_argument("--obs", default=None)
    parser.add_argument("--freq-mhz", type=float, default=None)
    parser.add_argument("--toa-error-us", type=float, default=None)
    parser.add_argument("--duration-min", type=float, default=20.0)
    parser.add_argument("--toas-per-session", type=int, default=8)
    parser.add_argument("--horizon-days", type=float, default=180.0)
    parser.add_argument("--baseline-count", type=int, default=10)
    parser.add_argument("--orbital-phase-samples", type=int, default=8)
    parser.add_argument("--annual-phase-samples", type=int, default=8)
    parser.add_argument("--maximum-modes", type=int, default=8)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--posterior-samples", type=int, default=256)
    parser.add_argument("--maximum-joint-sessions", type=int, default=2)
    parser.add_argument("--joint-candidate-pool", type=int, default=8)
    parser.add_argument("--disable-joint-sequence", action="store_true")
    parser.add_argument("--start-mjd", type=float, default=None)
    parser.add_argument("--end-mjd", type=float, default=None)
    parser.add_argument("--cadence-days", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = PintObservationPlannerConfig(
        horizon_days=args.horizon_days,
        baseline_count=args.baseline_count,
        orbital_phase_samples=args.orbital_phase_samples,
        annual_phase_samples=args.annual_phase_samples,
        duration_min=args.duration_min,
        toas_per_session=args.toas_per_session,
        maximum_modes=args.maximum_modes,
        maximum_recommendations=args.top,
        enable_joint_sequence=not args.disable_joint_sequence,
        maximum_joint_sessions=args.maximum_joint_sessions,
        joint_candidate_pool=args.joint_candidate_pool,
        design=BayesianObservationDesignConfig(
            posterior_samples_per_mode=args.posterior_samples
        ),
    )
    explicit = any(
        value is not None for value in (args.start_mjd, args.end_mjd, args.cadence_days)
    )
    windows = None
    if explicit:
        if any(
            value is None for value in (args.start_mjd, args.end_mjd, args.cadence_days)
        ):
            raise ValueError(
                "Explicit scheduling requires --start-mjd, --end-mjd, and --cadence-days."
            )
        if args.obs is None or args.freq_mhz is None or args.toa_error_us is None:
            modes = load_apollo_posterior_modes(
                args.run_output,
                maximum_modes=args.maximum_modes,
            )
            tim_path = _resolve_solution_tim(Path(args.run_output))
            toas = _load_pint_toas(
                tim_path,
                seed_model=modes[0].model,
                clock_override_path=args.clock_override,
            )
            inferred_obs, inferred_frequency, inferred_error, _ = _infer_observation_setup(toas)
        else:
            inferred_obs, inferred_frequency, inferred_error = (
                args.obs,
                args.freq_mhz,
                args.toa_error_us,
            )
        windows = generate_uniform_observation_windows(
            start_mjd=args.start_mjd,
            end_mjd=args.end_mjd,
            cadence_days=args.cadence_days,
            duration_min=args.duration_min,
            toas_per_session=args.toas_per_session,
            observatory=str(args.obs or inferred_obs),
            frequency_mhz=float(args.freq_mhz or inferred_frequency),
            toa_error_us=float(args.toa_error_us or inferred_error),
        )
    result = plan_apollo_observations(
        run_output_dir=args.run_output,
        output_dir=args.output,
        windows=windows,
        config=config,
        observatory=args.obs,
        frequency_mhz=args.freq_mhz,
        toa_error_us=args.toa_error_us,
        clock_override_path=args.clock_override,
    )
    print(json.dumps(to_json_ready(result), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BayesianObservationPlanResult",
    "BayesianObservationRecommendation",
    "BayesianObservationSequenceRecommendation",
    "ObservationNeedDiagnosis",
    "PintObservationPlannerConfig",
    "PintPosteriorModeSummary",
    "build_arg_parser",
    "generate_adaptive_observation_windows",
    "generate_isolated_observation_windows",
    "generate_model_adapted_observation_windows",
    "generate_uniform_observation_windows",
    "load_apollo_posterior_modes",
    "main",
    "plan_apollo_observations",
]
