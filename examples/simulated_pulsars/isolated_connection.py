"""A bounded blind isolated connection with evaluation/resume/update checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from io import StringIO
import json
import os
from pathlib import Path
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from astropy import units as u
from pint.models import get_model
from pint.simulation import make_fake_toas_fromMJDs
from loguru import logger

from apollo_timing.backend.pint_backend import _load_pint_toas, _prime_local_ephemeris
from apollo_timing.pipeline import solve_apollo_timing
from apollo_timing.pipeline.isolated_pulsar_solver import IsolatedPulsarSolverConfig
from apollo_timing.pipeline.automatic_timing import _same_complete_toas
from apollo_timing.inference.bayesian_isolated_cold_start import IsolatedColdStartConfig

from run import fresh_output, write_json


def run(output):
    _prime_local_ephemeris("DE421")
    model = get_model(StringIO(
        "PSRJ SIM096\nRAJ 12:00:00\nDECJ 10:00:00\nF0 100 1\nF1 -1e-15 1\n"
        "PEPOCH 59000\nDM 10\nEPHEM DE421\nCLOCK TT(TAI)\nUNITS TDB\n"
    ))
    # Twelve nonuniform sessions with genuine within-session frequency coverage.
    days = np.array([0, 1, 3, 8, 17, 29, 43, 61, 78, 96, 111, 130])
    offsets = np.array([0, 1/86400, 2/86400, .01, .01+1/86400, .01+2/86400])
    times = 59000 + np.repeat(days, 6) + np.tile(offsets, len(days))
    state = np.random.get_state()
    try:
        np.random.seed(2718)
        toas = make_fake_toas_fromMJDs(times, model, obs="geocenter",
                                      freq=np.tile([700, 1400, 2100], len(times)//3) * u.MHz,
                                      error=5 * u.us, add_noise=True, include_bipm=False)
    finally:
        np.random.set_state(state)
    toas.compute_pulse_numbers(model)
    model.write_parfile(output / "truth.par")
    toas.write_TOA_file(str(output / "truth.tim"), format="Tempo2", include_pn=True)
    tim = output / "unconnected.tim"
    toas.write_TOA_file(str(tim), format="Tempo2", include_pn=False)
    model.F0.value = 100 + 2e-8
    model.F1.value = 0
    par = output / "initial.par"
    model.write_parfile(par)
    # This is an explicit measurement-scale search box, not a reference PAR seed.
    config = IsolatedPulsarSolverConfig(custom_bounds={"F0": (100-2e-7, 100+2e-7),
                                                       "F1": (-1e-13, 1e-13)},
                                        isolated_cold_start=IsolatedColdStartConfig(enabled=False),
                                        plan_observations_on_failure=False)
    write_json(output / "protocol.json", dict(seed=2718, config=asdict(config),
               scope="blind_integer_connection_with_declared_finite_continuous_support",
               observation_count=len(times), numerical_threads=1))
    started = time.perf_counter()
    cpu_started = time.process_time()
    result = solve_apollo_timing(par_path=par, tim_path=tim, output_dir=output / "timing",
                                source_type="single", isolated_config=config)
    wall, cpu = time.perf_counter()-started, time.process_time()-cpu_started
    write_json(output / "solve_result.json", dict(accepted=result.accepted, chi2=result.chi2,
               reduced_chi2=result.reduced_chi2, wall_seconds=wall, cpu_seconds=cpu,
               timing_scope="May overlap other experiments; not an isolated machine benchmark"))
    fitted = get_model(result.solution_par)
    actual = _load_pint_toas(result.solution_tim, seed_model=fitted, clock_override_path=None)
    reference = _load_pint_toas(output / "truth.tim", seed_model=fitted, clock_override_path=None)
    assert result.accepted and .5 < result.reduced_chi2 < 1.5
    assert _same_complete_toas(actual, reference)
    a, b = actual.table["pulse_number"], reference.table["pulse_number"]
    np.testing.assert_array_equal(a-a[0], b-b[0])
    for name, value in (("F0", 100), ("F1", -1e-15)):
        assert abs(float(fitted[name].value)-value) < 4 * float(fitted[name].uncertainty_value)
    resumed = solve_apollo_timing(par_path=par, tim_path=tim, output_dir=output / "timing",
                                 source_type="single", isolated_config=config, resume=True)
    assert resumed.resumed and resumed.accepted and resumed.chi2 == result.chi2
    evaluated = solve_apollo_timing(par_path=result.solution_par, tim_path=result.solution_tim,
                                   output_dir=output / "evaluate", source_type="single", task="evaluate")
    assert evaluated.accepted and abs(evaluated.chi2-result.disk_reload_chi2) < .01
    # Whole-session historical/new split. No observations are dropped or rescaled.
    reference[:48].write_TOA_file(str(output / "historical.tim"), format="Tempo2", include_pn=True)
    reference[48:].write_TOA_file(str(output / "new.tim"), format="Tempo2", include_pn=False)
    updated = solve_apollo_timing(par_path=result.solution_par, tim_path=output / "historical.tim",
                                 update_tim_paths=(output / "new.tim",), output_dir=output / "update",
                                 source_type="single", isolated_config=config, task="update")
    assert updated.accepted
    updated_toas = _load_pint_toas(updated.solution_tim, seed_model=get_model(updated.solution_par), clock_override_path=None)
    assert _same_complete_toas(updated_toas, reference)
    pn = updated_toas.table["pulse_number"]
    np.testing.assert_array_equal(pn-pn[0], b-b[0])
    report = dict(all_passed=True, scope="finite_blind_connection_then_conditional_evaluate_resume_update",
                  source_name="SIM096", toa_count=len(actual.table), relative_pulse_path_matches=True,
                  solve_wall_seconds=wall, solve_cpu_seconds=cpu, reduced_chi2=result.reduced_chi2,
                  resume_passed=True, evaluate_passed=True, update_passed=True, uniqueness_proven=False)
    write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    output = fresh_output(args.output)
    logger.remove()
    logger.add(os.sys.stderr, level="WARNING")
    try:
        if args.profile:
            import cProfile
            profiler = cProfile.Profile()
            try:
                report = profiler.runcall(run, output)
            finally:
                profiler.dump_stats(output / "profile.pstats")
        else:
            report = run(output)
    except Exception as error:
        write_json(output / "verification.json", dict(all_passed=False, error=f"{type(error).__name__}: {error}"))
        raise
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
