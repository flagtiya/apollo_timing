from __future__ import annotations

from timing_accelerator.models.candidate_score import approximate_candidate_score
from timing_accelerator.models.contracts import BootstrapRequest, Candidate, PosteriorSummary


def _posterior_guided_score(
    candidate: Candidate,
    center_parameters: dict[str, float],
    posterior_summary: PosteriorSummary | None,
) -> float | None:
    if posterior_summary is None or posterior_summary.status != "usable":
        return None
    if not posterior_summary.parameter_sigmas:
        return None

    squared_distance = 0.0
    used_parameters = 0
    for name, value in candidate.hard_parameters.items():
        sigma = posterior_summary.parameter_sigmas.get(name)
        center_value = center_parameters.get(name)
        if sigma is None or center_value is None:
            continue
        stabilized_sigma = max(abs(sigma), 1.0e-30)
        squared_distance += ((value - center_value) / stabilized_sigma) ** 2
        used_parameters += 1
    if used_parameters == 0:
        return None
    return -0.5 * squared_distance


def _stage_shaping_score(
    candidate: Candidate,
    center_parameters: dict[str, float],
    basin_shaping_rank_weights: dict[str, float] | None,
) -> float | None:
    if not basin_shaping_rank_weights:
        return None

    squared_distance = 0.0
    used_parameters = 0
    for name, weight in basin_shaping_rank_weights.items():
        if name not in candidate.hard_parameters or name not in center_parameters:
            continue
        squared_distance += max(weight, 0.0) * (
            candidate.hard_parameters[name] - center_parameters[name]
        ) ** 2
        used_parameters += 1
    if used_parameters == 0:
        return None
    return -squared_distance


def rank_candidates(
    request: BootstrapRequest,
    candidates: list[Candidate],
    center_parameters: dict[str, float] | None = None,
    posterior_summary: PosteriorSummary | None = None,
    basin_shaping_rank_weights: dict[str, float] | None = None,
) -> list[Candidate]:
    ranked: list[Candidate] = []
    truth = request.simulation.truth if request.simulation.certification_backend == "mock" else None
    ranking_center = center_parameters or request.initial_parameters
    for candidate in candidates:
        posterior_score = _posterior_guided_score(candidate, ranking_center, posterior_summary)
        stage_shaping_score = _stage_shaping_score(
            candidate,
            ranking_center,
            basin_shaping_rank_weights,
        )
        if truth is None and posterior_score is not None:
            base_score = stage_shaping_score if stage_shaping_score is not None else approximate_candidate_score(
                candidate=candidate,
                initial_parameters=ranking_center,
                truth=None,
                penalty=request.simulation.approximation_penalty,
            )
            candidate.approximate_score = posterior_score + base_score
        else:
            candidate.approximate_score = (
                stage_shaping_score
                if truth is None and stage_shaping_score is not None
                else approximate_candidate_score(
                candidate=candidate,
                initial_parameters=ranking_center,
                truth=truth,
                penalty=request.simulation.approximation_penalty,
            )
            )
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: item.approximate_score
        if item.approximate_score is not None
        else float("-inf"),
        reverse=True,
    )
    return ranked


def retain_top_k(request: BootstrapRequest, ranked_candidates: list[Candidate]) -> list[Candidate]:
    return ranked_candidates[: request.simulation.top_k]
