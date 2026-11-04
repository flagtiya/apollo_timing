"""Sparse isolated data: retain frequency aliases and plan discriminating TOAs."""
from __future__ import annotations

import argparse
from io import StringIO
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from astropy import units as u
from pint.models import get_model
from pint.simulation import make_fake_toas_fromMJDs
from loguru import logger
from apollo_timing.backend.pint_backend import _prime_local_ephemeris
from apollo_timing.inference.bayesian_observation_planner import plan_apollo_observations
from apollo_timing.pipeline import solve_apollo_timing
from apollo_timing.pipeline.isolated_pulsar_solver import IsolatedPulsarSolverConfig
from apollo_timing.inference.bayesian_isolated_cold_start import IsolatedColdStartConfig
from run import fresh_output, write_json


def run(output):
    _prime_local_ephemeris("DE421")
    truth = get_model(StringIO("PSRJ SIM099\nRAJ 12:00:00\nDECJ 10:00:00\nF0 100 1\nF1 0\n"
                               "PEPOCH 59000\nDM 10\nEPHEM DE421\nCLOCK TT(TAI)\nUNITS TDB\n"))
    np.random.seed(1618)
    epochs = 59000 + np.repeat([0, 10, 20], 2) + np.tile([0, 120/86400], 3)
    toas = make_fake_toas_fromMJDs(epochs, truth, obs="@", freq=1400*u.MHz,
                                  error=50*u.us, add_noise=True, include_bipm=False)
    toas.compute_pulse_numbers(truth)
    truth.write_parfile(output / "truth.par")
    toas.write_TOA_file(str(output / "truth.tim"), format="Tempo2", include_pn=True)
    toas.write_TOA_file(str(output / "unconnected.tim"), format="Tempo2", include_pn=False)
    initial = get_model(output / "truth.par")
    initial.F0.value = 100 + 2e-7
    initial.write_parfile(output / "initial.par")
    config = IsolatedPulsarSolverConfig(custom_bounds={"F0": (100-4e-6, 100+4e-6)},
                                        isolated_cold_start=IsolatedColdStartConfig(enabled=False),
                                        plan_observations_on_failure=False)
    result = solve_apollo_timing(par_path=output / "initial.par", tim_path=output / "unconnected.tim",
                                output_dir=output / "timing", source_type="single", isolated_config=config)
    assert result.accepted
    plan = plan_apollo_observations(run_output_dir=output / "timing", output_dir=output / "plan")
    assert len(plan.posterior_modes) >= 2, "The finite timing portfolio did not retain the intended aliases"
    assert plan.recommendations and plan.evaluated_window_count > 0
    assert all(r.window.epoch_mjd > max(epochs) and np.isfinite(r.net_information_nats)
               for r in plan.recommendations)
    assert plan.diagnosis.significant_mode_count >= 2
    report = dict(all_passed=True, source_name="SIM099", seed=1618, toa_count=len(epochs),
                  scope="finite_blind_frequency_alias_portfolio_and_conditional_observation_design",
                  retained_planner_modes=len(plan.posterior_modes),
                  significant_modes=plan.diagnosis.significant_mode_count,
                  evaluated_windows=plan.evaluated_window_count, recommendations=len(plan.recommendations),
                  best_future_mjd=plan.recommendations[0].window.epoch_mjd,
                  forecast_relative_to="last input observation, not today's calendar", uniqueness_proven=False)
    write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = fresh_output(args.output)
    logger.remove()
    logger.add(os.sys.stderr, level="WARNING")
    try:
        report = run(output)
    except Exception as error:
        write_json(output / "verification.json", dict(all_passed=False, error=f"{type(error).__name__}: {error}"))
        raise
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
