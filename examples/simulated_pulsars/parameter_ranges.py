"""Noisy inside/outside interval checks using the finite parameter recipe matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("parameter_recipe_runner", HERE / "run.py")
recipes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipes)


def available_cases():
    return {**recipes.available_cases(), **{p.stem: p for p in sorted((HERE / "parameter-ranges/scenarios").glob("*.json"))}}


def run_case(output: Path, case: str) -> dict:
    import numpy as np
    from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
    from apollo_timing.inference.bayesian_binary_lattice import build_phase_lattice_search_context
    from apollo_timing.inference.bayesian_isolated_cold_start import IsolatedColdStartConfig
    from apollo_timing.inference.pint_solution_artifacts import (
        fit_pint_model_with_physical_bounds, write_pint_model_parfile_roundtrip_safe,
    )
    from apollo_timing.inference.timing_constraints import parameter_bounds_report
    from apollo_timing.inference.toa_white_noise_certification import evaluate_fixed_path_normalized_likelihood
    from apollo_timing.reports.residual_plot import write_residual_plot

    output.mkdir(parents=True)
    scenario = recipes.read_scenario(available_cases()[case])
    recipes.write_json(output / "scenario.json", scenario)
    inside = recipes.parameter_demo(scenario, output / "inside", True)
    directory = output / "outside"
    directory.mkdir()
    initial = _load_pint_model(output / "inside/initial.par")
    truth = _load_pint_model(output / "inside/truth.par")
    generated = _load_pint_toas(output / "inside/observations.tim", seed_model=initial, clock_override_path=None)
    ordered = generated[np.argsort(generated.get_mjds(high_precision=True), kind="stable")]
    tim = directory / "observations.tim"
    ordered.write_TOA_file(str(tim), format="Tempo2", include_pn=True, order_by_index=False)
    names = tuple(dict.fromkeys((*initial.free_params, *(s["parameter"] for s in scenario["signals"]))))
    initial.free_params = list(names)
    par = write_pint_model_parfile_roundtrip_safe(model=initial, path=directory / "initial.par")
    intervals = dict(scenario["nuisance_bounds"])
    for signal in scenario["signals"]:
        center, width = signal["null"], abs(signal["increment"]) / 10
        intervals[signal["parameter"]] = (center - width, center + width)
    context = build_phase_lattice_search_context(
        par_path=par, tim_path=tim,
        physical_parameter_bounds=tuple((name, *interval) for name, interval in intervals.items()),
        isolated_cold_start_config=IsolatedColdStartConfig(enabled=False),
    )
    evaluator, support = context.evaluator, context.support
    assert support.contains_physical(support.physical_center)
    for signal in scenario["signals"]:
        outside = support.physical_center.copy()
        outside[evaluator.parameter_names.index(signal["parameter"])] = float(truth[signal["parameter"]].value)
        assert not support.contains_physical(outside), signal["parameter"]
    toas = _load_pint_toas(tim, seed_model=initial, clock_override_path=None)
    recipes.compare_toas(ordered, toas)
    fit = fit_pint_model_with_physical_bounds(
        model=deepcopy(initial), toas=toas, parameter_names=evaluator.parameter_names,
        lower_bounds=support.physical_lower, upper_bounds=support.physical_upper,
    )
    solution = write_pint_model_parfile_roundtrip_safe(model=fit.model, path=directory / "solution.par")
    reloaded = _load_pint_model(solution)
    reloaded_toas = _load_pint_toas(tim, seed_model=reloaded, clock_override_path=None)
    recipes.compare_toas(toas, reloaded_toas)
    assert set(reloaded.free_params) == set(names)
    bounds = tuple(zip(evaluator.parameter_names, support.physical_lower, support.physical_upper))
    report = parameter_bounds_report(reloaded, bounds)
    assert report["within_bounds"], report
    metrics = evaluate_fixed_path_normalized_likelihood(model=reloaded, toas=reloaded_toas)
    assert abs(metrics.chi2 - float(fit.residuals.chi2)) < .01
    assert metrics.chi2 - inside["independent_reload"]["chi2"] > 10
    for source, stem in ((par, "before"), (solution, "after")):
        write_residual_plot(source, tim, directory, clock_override_path=None, artifact_stem=stem)
    result = dict(status="passed", case=case, source_name=scenario["source_name"],
                  scope="noisy_known_path_interval_enforcement_not_blind_connection_or_uniqueness",
                  inside=inside, outside=dict(parameter_bounds=report, active_bounds=fit.active_bounds,
                      optimizer_termination_status=fit.termination_status, optimizer_optimality=fit.optimality,
                      constrained_optimum_proven=False,
                      independent_reload=metrics.to_payload(), parameter_covariance=fit.covariance.tolist(),
                      covariance_parameter_names=evaluator.parameter_names,
                      covariance_scope="local_curvature_not_truncated_boundary_confidence_region"),
                  observations_errors_and_pulse_numbers_preserved=True, uniqueness_proven=False)
    recipes.write_json(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(available_cases()), action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("Use 1--32 independent one-thread workers")
    output = recipes.fresh_output(args.output)
    cases = args.case or list(available_cases())
    if len(cases) == 1:
        run_case(output / cases[0], cases[0])
        return 0

    def worker(case):
        with (output / f"{case}.log").open("w") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--case", case,
                                      "--output", str(output / case)], stdout=log, stderr=subprocess.STDOUT)
        path = output / case / case / "verification.json"
        return dict(case=case, passed=process.returncode == 0 and path.is_file(), report=str(path))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(worker, cases))
    passed = all(r["passed"] for r in results)
    recipes.write_json(output / "verification.json", dict(all_passed=passed, cases=len(results), results=results))
    print(json.dumps(dict(all_passed=passed, cases=len(results), output=str(output))))
    return 0 if passed else 1


if __name__ == "__main__":
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    raise SystemExit(main())
