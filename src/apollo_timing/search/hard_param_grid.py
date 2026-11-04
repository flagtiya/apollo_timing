from __future__ import annotations

from itertools import product

from apollo_timing.models.contracts import BootstrapRequest, Candidate


def generate_candidates(
    request: BootstrapRequest,
    hard_params: list[str] | None = None,
    center_parameters: dict[str, float] | None = None,
    grid_offsets: dict[str, list[float]] | None = None,
    candidate_prefix: str = "cand",
) -> list[Candidate]:
    hard_params = hard_params or request.simulation.hard_params
    center_parameters = center_parameters or request.initial_parameters
    offset_lookup = grid_offsets or request.simulation.grid_offsets
    candidate_axes: list[list[float]] = []
    for name in hard_params:
        base = center_parameters[name]
        offsets = offset_lookup.get(name, [0.0])
        candidate_axes.append([base + offset for offset in offsets])

    generated: list[Candidate] = []
    for index, combination in enumerate(product(*candidate_axes), start=1):
        hard_parameters = {
            name: value
            for name, value in zip(hard_params, combination, strict=True)
        }
        generated.append(
            Candidate(
                candidate_id=f"{candidate_prefix}-{index:03d}",
                hard_parameters=hard_parameters,
            )
        )
    return generated
