"""Write a certified pulse-number TIM and promote candidate artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Mapping


def write_solution_tim(
    *, toas: object, path: Path,
    comment: str = "APOLLO PINT-certified integer pulse path",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    toas.write_TOA_file(
        str(temporary),
        format="tempo2",
        include_pn=True,
        include_info=True,
        comment=comment,
    )
    temporary.replace(path)
    return path



def copy_candidate_to_solution(
    *,
    candidate: Mapping[str, Any],
    output: Path,
    incumbent_index: int,
) -> None:
    artifacts = candidate["artifacts"]
    destinations = {
        "par": output / "solution.par",
        "tim": output / "solution.tim",
        "residuals_csv": output / "residuals.csv",
        "residuals_png": output / "residuals.png",
    }
    for name, destination in destinations.items():
        shutil.copyfile(Path(artifacts[name]), destination)
    stem = f"{incumbent_index:03d}_chi2_{float(candidate['chi2']):.9f}"
    incumbent = output / "incumbents"
    incumbent.mkdir(parents=True, exist_ok=True)
    for name, suffix in (
        ("par", ".par"),
        ("tim", ".tim"),
        ("residuals_csv", "_residuals.csv"),
        ("residuals_png", "_residuals.png"),
    ):
        shutil.copyfile(Path(artifacts[name]), incumbent / f"{stem}{suffix}")

