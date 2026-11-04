from __future__ import annotations

from apollo_timing.models.contracts import BootstrapRequest, PriorModel


def build_prior(request: BootstrapRequest) -> PriorModel:
    widths: dict[str, float] = {}
    for name, value in request.initial_parameters.items():
        if name in request.simulation.grid_offsets:
            widths[name] = max(abs(offset) for offset in request.simulation.grid_offsets[name]) or 0.01
        else:
            widths[name] = max(abs(value) * 0.05, 0.01)
    return PriorModel(center=dict(request.initial_parameters), widths=widths)
