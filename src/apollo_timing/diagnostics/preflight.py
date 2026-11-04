from __future__ import annotations

from apollo_timing.models.contracts import BootstrapRequest, PreflightResult


def run_preflight(request: BootstrapRequest) -> PreflightResult:
    offset_widths = [
        abs(offset)
        for offsets in request.simulation.grid_offsets.values()
        for offset in offsets
    ]
    max_offset = max(offset_widths) if offset_widths else 0.0
    return PreflightResult(
        ok_for_local_accelerator=request.toa_count > 0,
        phase_wrap_risk="low" if max_offset <= 0.5 else "moderate",
        max_phase_sigma_cycles=round(max_offset, 3),
        linearization_quality="acceptable",
        conditioning="acceptable",
        notes=[
            f"TOA count: {request.toa_count}",
            (
                "Real-data full-fit mode enabled"
                if request.simulation.regime == "full-fit"
                else "Synthetic bootstrap mode enabled"
            ),
        ],
    )
