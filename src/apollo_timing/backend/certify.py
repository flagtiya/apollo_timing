from __future__ import annotations

from apollo_timing.backend.pint_backend import certify_with_pint
from apollo_timing.models.candidate_score import exact_candidate_score
from apollo_timing.models.contracts import BootstrapRequest, Candidate, CertificationResult


def certify_candidates(
    request: BootstrapRequest,
    candidates: list[Candidate],
) -> CertificationResult:
    if request.simulation.certification_backend == "pint":
        return certify_with_pint(request, candidates)

    certified: list[Candidate] = []
    for candidate in candidates:
        candidate.exact_score = exact_candidate_score(candidate, request.simulation.truth)
        certified.append(candidate)
    best_candidate = max(
        certified,
        key=lambda item: item.exact_score if item.exact_score is not None else float("-inf"),
    )
    return CertificationResult(candidates=certified, best_candidate=best_candidate, metrics={})
