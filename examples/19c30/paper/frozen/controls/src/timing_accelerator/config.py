from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelinePaths:
    par_path: Path
    tim_path: Path
    simulation_config_path: Path
    output_dir: Path

    @classmethod
    def from_strings(
        cls,
        par_path: str,
        tim_path: str,
        simulation_config_path: str,
        output_dir: str,
    ) -> "PipelinePaths":
        return cls(
            par_path=Path(par_path),
            tim_path=Path(tim_path),
            simulation_config_path=Path(simulation_config_path),
            output_dir=Path(output_dir),
        )
