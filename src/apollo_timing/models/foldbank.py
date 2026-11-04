"""Value contracts for finite folding coverage, independent of execution backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from apollo_timing.inference.bayesian_partial_ell1_bootstrap import (
        PartialELL1BootstrapMode,
        PartialELL1BootstrapResult,
    )


@dataclass(frozen=True)
class FoldbankConfig:
    minimum_toas_per_session: int = 8
    maximum_toas_per_session: int = 16
    maximum_phase_span_cycles: float = 0.35
    maximum_coherent_loss: float = 0.25
    maximum_candidate_delta_chi2: float = 25.0
    minimum_trajectory_separation_cycles: float = 0.05
    maximum_candidates: int = 256
    orbital_period_min_days: float | None = None
    orbital_period_max_days: float | None = None
    maximum_orbital_aliases: int = 256
    profile_bins: int = 64
    subbands: int = 32
    validation_parts: int = 16
    detection_sigma_threshold: float = 8.0
    posterior_draws: int = 0
    posterior_retained_per_mode: int = 256
    posterior_seed_modes: int = 1
    posterior_box_sigma: float = 3.0
    bootstrap_workers: int = 1
    posterior_workers: int = 1
    presto_workers: int = 1

    def __post_init__(self) -> None:
        counts = (
            self.minimum_toas_per_session,
            self.maximum_toas_per_session,
            self.maximum_candidates,
            self.maximum_orbital_aliases,
            self.profile_bins,
            self.subbands,
            self.validation_parts,
            self.posterior_draws,
            self.posterior_retained_per_mode,
            self.posterior_seed_modes,
            self.bootstrap_workers,
            self.posterior_workers,
            self.presto_workers,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in counts):
            raise ValueError("Fold-bank counts and worker budgets must be integers.")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type == "int":
                object.__setattr__(self, field.name, int(value))
            elif value is not None:
                object.__setattr__(self, field.name, float(value))
        if self.minimum_toas_per_session < 2:
            raise ValueError("Fold-bank sessions need at least two TOAs.")
        if self.maximum_toas_per_session < self.minimum_toas_per_session:
            raise ValueError("The maximum session TOA count is below the minimum.")
        if self.maximum_candidates <= 0 or self.maximum_orbital_aliases <= 0:
            raise ValueError("Fold-bank candidate and orbital-alias budgets must be positive.")
        if self.posterior_draws < 0:
            raise ValueError("The posterior draw count cannot be negative.")
        if self.posterior_retained_per_mode <= 0 or self.posterior_seed_modes <= 0:
            raise ValueError("Posterior fold-bank budgets must be positive.")
        if min(self.bootstrap_workers, self.posterior_workers, self.presto_workers) <= 0:
            raise ValueError("The bootstrap, posterior, and PRESTO worker counts must be positive.")
        if not np.isfinite(self.posterior_box_sigma) or self.posterior_box_sigma <= 0.0:
            raise ValueError("The posterior sampling box scale must be positive.")
        if self.profile_bins <= 0 or self.subbands <= 0 or self.validation_parts <= 0:
            raise ValueError("PRESTO profile and subband counts must be positive.")
        fractions = (
            self.maximum_phase_span_cycles,
            self.maximum_coherent_loss,
            self.minimum_trajectory_separation_cycles,
            self.maximum_candidate_delta_chi2,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in fractions):
            raise ValueError("Fold-bank phase thresholds must be finite and positive.")
        if self.maximum_coherent_loss > 1.0:
            raise ValueError("The maximum coherent loss cannot exceed one.")
        if not np.isfinite(self.detection_sigma_threshold) or self.detection_sigma_threshold <= 0.0:
            raise ValueError("The PRESTO detection threshold must be finite and positive.")
        period_bounds = (
            self.orbital_period_min_days,
            self.orbital_period_max_days,
        )
        if (period_bounds[0] is None) != (period_bounds[1] is None):
            raise ValueError("Both --pb-min and --pb-max must be supplied together.")
        if period_bounds[0] is not None:
            lower, upper = (float(value) for value in period_bounds)
            if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0:
                raise ValueError("The fold-bank PB prior must be finite and positive.")
            if upper <= lower:
                raise ValueError("--pb-max must exceed --pb-min.")


@dataclass(frozen=True)
class SearchObservation:
    observation_id: str
    path: Path
    source: str
    telescope: str
    start_mjd: float
    duration_seconds: float
    center_frequency_mhz: float
    channels: int

    @property
    def end_mjd(self) -> float:
        return self.start_mjd + self.duration_seconds / 86400.0

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "path": str(self.path),
            "source": self.source,
            "telescope": self.telescope,
            "start_mjd": self.start_mjd,
            "end_mjd": self.end_mjd,
            "duration_seconds": self.duration_seconds,
            "center_frequency_mhz": self.center_frequency_mhz,
            "channels": self.channels,
        }


@dataclass(frozen=True)
class OrbitalPeriodAlias:
    alias_id: str
    center_days: float
    minimum_days: float
    maximum_days: float
    minimum_frequency_per_day: float
    maximum_frequency_per_day: float
    cycles_across_baseline: float

    def to_dict(self) -> dict[str, object]:
        return {
            "alias_id": self.alias_id,
            "center_days": self.center_days,
            "minimum_days": self.minimum_days,
            "maximum_days": self.maximum_days,
            "minimum_frequency_per_day": self.minimum_frequency_per_day,
            "maximum_frequency_per_day": self.maximum_frequency_per_day,
            "cycles_across_baseline": self.cycles_across_baseline,
        }


@dataclass(frozen=True)
class SessionPhaseScore:
    observation_id: str
    toa_count: int
    phase_span_cycles: float
    phase_rms_cycles: float
    coherent_loss: float
    chi2: float

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "toa_count": self.toa_count,
            "phase_span_cycles": self.phase_span_cycles,
            "phase_rms_cycles": self.phase_rms_cycles,
            "coherent_loss": self.coherent_loss,
            "chi2": self.chi2,
        }


@dataclass(frozen=True)
class EvaluatedMode:
    mode: PartialELL1BootstrapMode
    session_scores: tuple[SessionPhaseScore, ...]
    trajectory: np.ndarray
    eligible: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FoldbankResult:
    output_dir: Path
    manifest_path: Path
    command_script_path: Path
    candidate_paths: tuple[Path, ...]
    fold_results: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "command_script_path": str(self.command_script_path),
            "candidate_paths": [str(path) for path in self.candidate_paths],
            "fold_results": list(self.fold_results),
        }


@dataclass(frozen=True)
class BootstrapEntry:
    alias_id: str
    seed_id: str
    result: PartialELL1BootstrapResult

    def to_dict(self) -> dict[str, object]:
        return {"alias_id": self.alias_id, "seed_id": self.seed_id, "result": self.result.to_dict()}


@dataclass(frozen=True)
class BootstrapPortfolio:
    aliases: tuple[OrbitalPeriodAlias, ...]
    entries: tuple[BootstrapEntry, ...]
    alias_records: tuple[dict[str, object], ...]

    @property
    def results(self) -> tuple[PartialELL1BootstrapResult, ...]:
        return tuple(entry.result for entry in self.entries)

    @property
    def modes(self) -> tuple[PartialELL1BootstrapMode, ...]:
        return tuple(mode for result in self.results for mode in result.modes)

    @property
    def mode_alias_by_id(self) -> dict[str, OrbitalPeriodAlias]:
        aliases = {alias.alias_id: alias for alias in self.aliases}
        return {
            mode.mode_id: aliases[entry.alias_id]
            for entry in self.entries
            for mode in entry.result.modes
        }


@dataclass(frozen=True)
class FoldTask:
    candidate_id: str
    par_path: Path
    observation: SearchObservation
    parts: int

    def __post_init__(self) -> None:
        for value in (self.candidate_id, self.observation.observation_id):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
                raise ValueError("Fold task IDs must be safe single path components.")
        if isinstance(self.parts, bool) or not isinstance(self.parts, Integral) or self.parts <= 0:
            raise ValueError("Fold task parts must be a positive integer.")
