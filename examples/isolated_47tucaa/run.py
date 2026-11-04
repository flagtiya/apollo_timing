"""Finite-path recovery audit for the regularly sampled synthetic fixture."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from loguru import logger
import numpy as np
from pint.residuals import Residuals

from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
from apollo_timing.pipeline import solve_apollo_timing
from apollo_timing.pipeline.automatic_timing import _same_complete_toas
from apollo_timing.pipeline.isolated_pulsar_solver import IsolatedPulsarSolverConfig


HERE = Path(__file__).resolve().parent


def run(output: Path, retained_paths: int) -> dict:
    output = output.expanduser().resolve()
    if output.is_relative_to(HERE.parents[1]) or output.exists():
        raise ValueError("Use a fresh output directory outside the source tree.")
    if not 1 <= retained_paths <= 128:
        raise ValueError("The retained-path budget must be between 1 and 128.")
    output.mkdir(parents=True)
    lines = []
    for line in (HERE / "data.tim").read_text().splitlines():
        fields = line.split()
        if "-pn" in fields:
            index = fields.index("-pn")
            del fields[index:index + 2]
            line = " ".join(fields)
        lines.append(line)
    unconnected = output / "unconnected.tim"
    unconnected.write_text("\n".join(lines) + "\n")
    config = IsolatedPulsarSolverConfig(plan_observations_on_failure=False)
    certification_budget = retained_paths * config.annealing.beam_size
    config = replace(
        config,
        cell_search=replace(config.cell_search, paths_per_component=retained_paths),
        annealing=replace(config.annealing, phase_sigma_schedule=(0.0,)),
        maximum_modes_to_anneal=retained_paths,
        maximum_modes_to_certify=certification_budget,
    )
    started = time.perf_counter()
    result = solve_apollo_timing(
        par_path=HERE / "initial.par", tim_path=unconnected,
        output_dir=output / "timing", source_type="single", isolated_config=config,
    )
    wall = time.perf_counter() - started
    # Reference labels are accessed only after the unconnected search completes.
    model = _load_pint_model(result.solution_par)
    reference = _load_pint_toas(HERE / "data.tim", model)
    numbers = np.asarray(reference.table["pulse_number"], dtype=np.longdouble)
    reference_path = tuple(numbers - numbers[0])
    portfolio = json.loads((output / "timing/candidate_portfolio.json").read_text())
    paths = set()
    checked = 0
    for candidate in portfolio:
        if not candidate.get("accepted"):
            continue
        fitted = _load_pint_model(candidate["artifacts"]["par"])
        toas = _load_pint_toas(candidate["artifacts"]["tim"], fitted)
        if set(fitted.free_params) != {"F0", "F1"}:
            raise ValueError("A saved candidate changed the authorized fit parameters.")
        if not _same_complete_toas(toas, reference):
            raise ValueError("A saved candidate changed observations or reported errors.")
        residuals = Residuals(
            toas, fitted, subtract_mean=True, use_abs_phase=True,
            track_mode="use_pulse_numbers",
        )
        chi2 = float(residuals.chi2)
        if not np.isfinite(chi2) or abs(chi2 - candidate["disk_reload_chi2"]) >= 0.01:
            raise ValueError("A saved candidate failed independent residual reloading.")
        pulse_numbers = np.asarray(toas.table["pulse_number"], dtype=np.longdouble)
        paths.add(tuple(pulse_numbers - pulse_numbers[0]))
        checked += 1
    report = dict(
        all_passed=bool(result.accepted and checked and reference_path in paths),
        scope="finite_zero_added_phase_noise_search_no_input_pulse_numbers_fixed_sky",
        retained_path_budget=retained_paths,
        certification_budget=certification_budget,
        toa_count=len(reference.table), wall_seconds=wall,
        accepted_records_checked=checked, distinct_accepted_paths=len(paths),
        supplied_reference_path_retained=reference_path in paths,
        preferred_chi2=result.chi2, uniqueness_proven=False,
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_passed"]:
        raise RuntimeError("The finite search did not retain the reference path.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retained-paths", type=int, default=32)
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    print(json.dumps(run(args.output, args.retained_paths), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
