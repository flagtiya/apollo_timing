from __future__ import annotations

from timing_accelerator.models.contracts import BootstrapRequest, ParameterPartition


def build_parameter_partition(request: BootstrapRequest) -> ParameterPartition:
    return ParameterPartition(
        hard_params=list(request.simulation.hard_params),
        soft_params=list(request.simulation.soft_params),
    )
