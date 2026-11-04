from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActiveTimingBranch:
    branch_id: str
    structure_id: str
    binary_model: str | None
    wraps: tuple[int, ...]
    theta_names: tuple[str, ...]
    theta_offset: np.ndarray
    log_prior: float
    proposal_source: str
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class BranchCertification:
    branch_id: str
    log_likelihood: float
    fitted_theta: dict[str, float]
    covariance: np.ndarray
    reduced_chi2: float | None
    weighted_rms_us: float | None
    backend: str
    log_evidence: float | None = None
    chi2: float | None = None
    notes: list[str] | None = None


@dataclass(frozen=True)
class ActiveTimingPosterior:
    branch_probabilities: dict[str, float]
    log_evidences: dict[str, float]
    entropy: float
    best_branch_id: str | None
    log_odds_best_second: float | None
    certifiable: bool
    ambiguity_pairs: list[dict[str, float | str]]
    quality_gates: dict[str, dict[str, bool | float | None]]


@dataclass(frozen=True)
class ObservationWindow:
    epoch_mjd: float
    duration_min: float = 20.0
    toa_count: int = 8
    obs: str | None = None
    freq_mhz: float | None = None
    toa_error_us: float | None = None
    cost: float = 0.0


@dataclass(frozen=True)
class ObservationRecommendation:
    rank: int
    window: ObservationWindow
    utility: float
    expected_entropy_drop: float
    expected_log_bayes_gain: float
    phase_separation_sigma: float
    top_discriminated_pairs: list[dict[str, float | str]]
    phase_predictions: dict[str, float]


@dataclass(frozen=True)
class ActiveTimingRunResult:
    output_dir: Path
    branches: list[ActiveTimingBranch]
    certifications: list[BranchCertification]
    posterior: ActiveTimingPosterior
    recommendations: list[ObservationRecommendation]


def to_json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return to_json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_json_ready(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.floating):
        number = value.item()
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, tuple):
        return [to_json_ready(item) for item in value]
    if isinstance(value, list):
        return [to_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    return value
