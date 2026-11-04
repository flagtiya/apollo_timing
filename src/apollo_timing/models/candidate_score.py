from __future__ import annotations

from apollo_timing.models.contracts import Candidate


def _squared_distance(lhs: dict[str, float], rhs: dict[str, float]) -> float:
    total = 0.0
    for name, value in lhs.items():
        if name in rhs:
            total += (value - rhs[name]) ** 2
    return total


def approximate_candidate_score(
    candidate: Candidate,
    initial_parameters: dict[str, float],
    truth: dict[str, float] | None,
    penalty: float,
) -> float:
    initial_projection = {
        name: initial_parameters[name]
        for name in candidate.hard_parameters
        if name in initial_parameters
    }
    drift_distance = _squared_distance(candidate.hard_parameters, initial_projection)
    if not truth:
        return -drift_distance
    truth_distance = _squared_distance(candidate.hard_parameters, truth)
    return -(truth_distance + penalty * drift_distance)


def exact_candidate_score(candidate: Candidate, truth: dict[str, float]) -> float:
    return -_squared_distance(candidate.hard_parameters, truth)
