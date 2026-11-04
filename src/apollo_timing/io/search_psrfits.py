"""Search-mode observation inspection and training/held-out input validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from astropy.io import fits

from apollo_timing.models.foldbank import FoldbankConfig, SearchObservation

if TYPE_CHECKING:
    from apollo_timing.inference.bayesian_dual_lattice_enumerator import (
        SourceSession,
    )


def load_search_observations(
    paths: Sequence[Path], *, prefix: str
) -> tuple[SearchObservation, ...]:
    observations = sorted(
        (inspect_search_psrfits(path, observation_id="pending") for path in paths),
        key=lambda observation: observation.start_mjd,
    )
    return tuple(
        replace(item, observation_id=f"{prefix}_{index}")
        for index, item in enumerate(observations, start=1)
    )


def inspect_search_psrfits(path: Path, *, observation_id: str) -> SearchObservation:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    primary = fits.getheader(source, 0)
    subint = fits.getheader(source, "SUBINT")
    if str(primary.get("OBS_MODE", "")).strip().upper() != "SEARCH":
        raise ValueError(f"{source} is not a search-mode PSRFITS file.")
    start_mjd = (
        float(primary["STT_IMJD"])
        + (float(primary.get("STT_SMJD", 0.0)) + float(primary.get("STT_OFFS", 0.0))) / 86400.0
    )
    channels = int(subint["NCHAN"])
    frequency = float(primary["OBSFREQ"])
    if not np.isfinite(start_mjd) or not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(f"{source} has an invalid start epoch or frequency.")
    if channels <= 0 or not str(primary.get("SRC_NAME", "")).strip():
        raise ValueError(f"{source} needs a source name and positive channel count.")
    duration = float(subint["TBIN"]) * int(subint["NSBLK"]) * int(subint["NAXIS2"])
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{source} has a non-positive search duration.")
    return SearchObservation(
        observation_id=observation_id,
        path=source,
        source=str(primary.get("SRC_NAME", "")).strip(),
        telescope=str(primary.get("TELESCOP", "")).strip(),
        start_mjd=start_mjd,
        duration_seconds=duration,
        center_frequency_mhz=frequency,
        channels=channels,
    )


def _validate_two_session_contract(
    sessions: Sequence[SourceSession],
    *,
    config: FoldbankConfig,
) -> None:
    if len(sessions) < 2:
        raise ValueError(
            "apollo-foldbank requires at least two independent observing sessions; "
            f"found {len(sessions)}."
        )
    for index, session in enumerate(sessions, start=1):
        count = len(session.toa_indices)
        if not config.minimum_toas_per_session <= count <= config.maximum_toas_per_session:
            raise ValueError(
                f"Fold-bank session {index} has {count} TOAs; expected "
                f"{config.minimum_toas_per_session}-{config.maximum_toas_per_session}."
            )


def _match_sessions_to_observations(
    *,
    sessions: Sequence[SourceSession],
    observations: Sequence[SearchObservation],
) -> tuple[tuple[SourceSession, SearchObservation], ...]:
    ordered_sessions = sorted(sessions, key=lambda item: item.epoch_mjd)
    ordered_observations = sorted(observations, key=lambda item: item.start_mjd)
    if len(ordered_sessions) != len(ordered_observations):
        raise ValueError("TOA sessions and search observations have different counts.")
    if any(
        left.end_mjd >= right.start_mjd
        for left, right in zip(ordered_observations[:-1], ordered_observations[1:], strict=True)
    ):
        raise ValueError("Search observations overlap and are not independent epochs.")
    matches = []
    for session, observation in zip(ordered_sessions, ordered_observations, strict=True):
        tolerance_days = max(observation.duration_seconds / 86400.0, 1.0e-4)
        if not (
            observation.start_mjd - tolerance_days
            <= session.epoch_mjd
            <= observation.end_mjd + tolerance_days
        ):
            raise ValueError(
                f"TOA session at MJD {session.epoch_mjd:.12f} does not match {observation.path}."
            )
        matches.append((session, observation))
    return tuple(matches)


def _validate_observation_sets(
    training: Sequence[SearchObservation],
    heldout: Sequence[SearchObservation],
) -> None:
    observations = sorted((*training, *heldout), key=lambda item: item.start_mjd)
    if len({item.path for item in observations}) != len(observations):
        raise ValueError("Training and validation observations must have distinct paths.")
    if any(
        left.end_mjd >= right.start_mjd
        for left, right in zip(observations[:-1], observations[1:], strict=True)
    ):
        raise ValueError(
            "Training and validation observations overlap; epochs must be independent."
        )
