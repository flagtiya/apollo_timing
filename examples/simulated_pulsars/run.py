"""Reproducible noisy feature demonstrations, conditional on a known pulse path."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Parallelism belongs to independent experiments, not nested BLAS pools.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def available_cases():
    return {p.parent.name: p for p in sorted(HERE.glob("*/scenario.json"))}


def read_scenario(path):
    value = json.loads(path.read_text())
    if value["schema"] != 1 or not value["signals"]:
        raise ValueError("Expected a version-1 finite parameter demonstration")
    names = [s["parameter"] for s in value["signals"]]
    if len(names) != len(set(names)) or set(names) & set(value["nuisance_bounds"]):
        raise ValueError("Signal and nuisance parameter names must be disjoint")
    return value


def fresh_output(path):
    path = path.expanduser().resolve()
    if path.is_relative_to(ROOT) or (path.exists() and any(path.iterdir())):
        raise ValueError("Use a fresh output directory outside the source tree")
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_toas(reference, candidate):
    import numpy as np
    from apollo_timing.pipeline.automatic_timing import _same_complete_toas

    if not _same_complete_toas(reference, candidate):
        raise AssertionError("TOAs, reported errors, or frequencies changed")
    np.testing.assert_array_equal(reference.table["pulse_number"], candidate.table["pulse_number"])


def parameter_demo(scenario, output, injected):
    import numpy as np
    from astropy import units as u
    from pint.models import get_model
    from pint.simulation import make_fake_toas_fromMJDs
    from apollo_timing.backend.pint_backend import _load_pint_toas, _prime_local_ephemeris
    from apollo_timing.inference.pint_solution_artifacts import (
        fit_pint_model_with_physical_bounds, write_pint_model_parfile_roundtrip_safe,
    )
    from apollo_timing.inference.timing_parameter_exploration import (
        ParameterExplorationConfig, explore_parameters_on_fixed_path,
    )
    from apollo_timing.inference.toa_white_noise_certification import (
        evaluate_fixed_path_normalized_likelihood,
    )

    output.mkdir()
    _prime_local_ephemeris("DE421")
    baseline = get_model(StringIO(scenario["input_par"]))
    baseline.PSR.value = scenario["source_name"]
    for signal in scenario["signals"]:
        baseline[signal["parameter"]].quantity = signal["null"] * baseline[signal["parameter"]].units
    nuisance = scenario["nuisance_bounds"]
    baseline.free_params = list(nuisance)
    truth = deepcopy(baseline)
    truth.free_params = []
    for signal in scenario["signals"]:
        parameter = truth[signal["parameter"]]
        parameter.quantity = (signal["null"] + (signal["increment"] if injected else 0)) * parameter.units
    # Cadence and noise are declared before fitting and shared by null/injection.
    cadence = scenario["cadence"]
    epochs = np.linspace(-cadence["half_span_days"], cadence["half_span_days"], cadence["epochs"])
    frequencies = np.asarray(cadence["frequencies_mhz"])
    times = 59000 + np.repeat(epochs, len(frequencies))
    rng_state = np.random.get_state()
    try:
        np.random.seed(scenario["seed"])
        toas = make_fake_toas_fromMJDs(
            times, truth, obs="geocenter", freq=np.tile(frequencies, len(epochs)) * u.MHz,
            error=cadence["error_us"] * u.us, add_noise=True, include_bipm=False,
        )
    finally:
        np.random.set_state(rng_state)
    toas.compute_pulse_numbers(truth)
    baseline.phase(toas, abs_phase=True)

    def save_model(model, name):
        path = output / name
        write_pint_model_parfile_roundtrip_safe(model=model, path=path)
        return path

    save_model(truth, "truth.par")
    initial = save_model(baseline, "initial.par")
    tim = output / "observations.tim"
    toas.write_TOA_file(str(tim), name=scenario["source_name"], format="Tempo2", include_pn=True)
    # No PN file is also supplied for users wishing to run a separate connection search.
    ordered = toas[np.argsort(toas.get_mjds(high_precision=True), kind="stable")]
    ordered.write_TOA_file(str(output / "unconnected.tim"), name=scenario["source_name"],
                           format="Tempo2", include_pn=False, order_by_index=False)
    # Fit the actual distributed-format data, not the generator's cached TDBs.
    serialized_toas = _load_pint_toas(tim, seed_model=baseline, clock_override_path=None)
    compare_toas(toas, serialized_toas)
    toas = serialized_toas
    original_toas = deepcopy(toas)
    original_parameters = {p: str(baseline[p].quantity) for p in baseline.params}
    initial_metrics = evaluate_fixed_path_normalized_likelihood(model=baseline, toas=toas)
    null_fit = fit_pint_model_with_physical_bounds(
        model=deepcopy(baseline), toas=toas, parameter_names=tuple(nuisance),
        lower_bounds=[b[0] for b in nuisance.values()], upper_bounds=[b[1] for b in nuisance.values()],
    )
    if null_fit.active_bounds:
        raise AssertionError(f"The baseline nuisance fit hit bounds: {null_fit.active_bounds}")
    baseline_metrics = evaluate_fixed_path_normalized_likelihood(model=null_fit.model, toas=toas)
    bounds = tuple((s["parameter"], *s["bounds"]) for s in scenario["signals"])
    fitted, residuals, comparison, covariance = explore_parameters_on_fixed_path(
        model=null_fit.model, toas=toas,
        config=ParameterExplorationConfig(parameters=tuple(s["parameter"] for s in scenario["signals"]),
                                          search_bounds=bounds),
        input_bounds=nuisance,
    )
    save_model(null_fit.model, "baseline_fitted.par")
    solution = save_model(fitted, "solution.par")
    write_json(output / "comparison.json", comparison)
    if comparison["status"] != "evaluated":
        raise AssertionError(f"Parameter exploration incomplete: {comparison}")
    selected = comparison["selected_candidate_id"] != "baseline"
    if selected != injected:
        raise AssertionError(f"Incorrect null/injection selection: {comparison}")
    if original_parameters != {p: str(baseline[p].quantity) for p in baseline.params}:
        raise AssertionError("Input model changed")
    compare_toas(original_toas, toas)
    recovery = {}
    if injected:
        if covariance is None or not np.all(np.isfinite(covariance)):
            raise AssertionError("Missing finite covariance")
        standard_deviations = np.sqrt(np.diag(covariance))
        correlation = covariance / standard_deviations[:, None] / standard_deviations[None, :]
        if not np.all(np.isfinite(correlation)) or np.min(np.linalg.eigvalsh(correlation)) <= 0:
            raise AssertionError("Covariance is not positive definite")
        for signal in scenario["signals"]:
            name = signal["parameter"]
            covariance_names = [*null_fit.model.free_params, *(s["parameter"] for s in scenario["signals"])]
            i = covariance_names.index(name)
            expected = float(truth[name].value)
            error = abs(float(fitted[name].value) - expected)
            sigma = float(np.sqrt(covariance[i, i]))
            recovery[name] = dict(expected=expected, recovered=float(fitted[name].value),
                                  uncertainty=sigma, error_sigma=error / sigma,
                                  fraction_of_injection=error / abs(signal["increment"]))
            # Fixed family-wide checks; never adjusted from a failed realization.
            if error >= 4 * sigma or error >= .05 * abs(signal["increment"]):
                raise AssertionError(f"Injection recovery failed: {recovery[name]}")
    metrics = evaluate_fixed_path_normalized_likelihood(model=fitted, toas=toas)
    if not .5 <= metrics.reduced_chi2 <= 1.5:
        raise AssertionError(f"Final residuals do not match declared Gaussian noise: {metrics}")
    reloaded_model = get_model(solution)
    reloaded_toas = _load_pint_toas(tim, seed_model=reloaded_model, clock_override_path=None)
    compare_toas(toas, reloaded_toas)
    reloaded = evaluate_fixed_path_normalized_likelihood(model=reloaded_model, toas=reloaded_toas)
    if abs(reloaded.chi2 - metrics.chi2) > .01:
        raise AssertionError("Independent saved-file reload changed chi-square")
    truth_reload = get_model(output / "truth.par")
    for signal in scenario["signals"]:
        name = signal["parameter"]
        if not np.isclose(float(truth_reload[name].value), float(truth[name].value), rtol=1e-12, atol=0):
            raise AssertionError(f"Truth PAR does not round-trip for {name}")
    rows = np.column_stack((toas.get_mjds().value, toas.get_freqs().value, toas.get_errors().to_value(u.us),
                           null_fit.residuals.time_resids.to_value(u.us), residuals.time_resids.to_value(u.us)))
    np.savetxt(output / "residuals.csv", rows, delimiter=",", header="mjd,frequency_mhz,error_us,baseline_us,selected_us", comments="")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    for ax, column, label in zip(axes, (3, 4), ("Baseline (nuisance refitted)", "Authorized parameter selection")):
        for freq in frequencies:
            mask = rows[:, 1] == freq
            ax.errorbar(rows[mask, 0], rows[mask, column], rows[mask, 2], fmt=".", ms=3,
                        label=f"{freq:g} MHz", alpha=.75)
        ax.set_ylabel("Residual (us)")
        ax.set_title(label, loc="left", fontsize=10)
        ax.axhline(0, color="0.5", lw=.5)
    axes[0].legend(ncol=len(frequencies), fontsize=8)
    axes[1].set_xlabel("MJD")
    figure.suptitle(scenario["source_name"] + (" / injected" if injected else " / null control"))
    figure.savefig(output / "residuals.png", dpi=140)
    plt.close(figure)
    command = ["apollo-timing", "--source-type", scenario["source_type"], "--task", "evaluate",
               "--par", str(initial), "--tim", str(tim), "--output", str(output / "separate-evaluation")]
    write_json(output / "commands.json", dict(
        evaluate_initial=command,
        connection_search_note="Use unconnected.tim and an explicitly chosen search budget; this demonstration does not execute or certify a blind connection.",
    ))
    return dict(status="passed", injected=injected, selected=selected, recovery=recovery,
                initial=initial_metrics.to_payload(), baseline=baseline_metrics.to_payload(),
                selected_metrics=metrics.to_payload(), independent_reload=reloaded.to_payload(),
                delta_bic=baseline_metrics.bic - metrics.bic,
                toa_count=len(toas.table), pulse_path_unchanged=True, input_contract_unchanged=True)


def run_one(path, output):
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    scenario = read_scenario(path)
    (output / "scenario.json").write_bytes(path.read_bytes())
    started = time.perf_counter()
    results = []
    for injected in (False, True):
        try:
            result = parameter_demo(scenario, output / ("injected" if injected else "null"), injected)
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = dict(status="failed", injected=injected, error=f"{type(error).__name__}: {error}")
        results.append(result)
    report = dict(case=path.parent.name, source_name=scenario["source_name"],
                  scope="noisy_known_pulse_path_parameter_demonstration_not_blind_timing",
                  all_passed=all(r["status"] == "passed" for r in results), results=results,
                  seed=scenario["seed"], wall_seconds=time.perf_counter() - started,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output / "verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[], help="Repeat to select several independent sources")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=1, help="Independent one-thread processes (1-32)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    cases = available_cases()
    if args.list:
        for name, path in cases.items():
            scenario = read_scenario(path)
            print(name, scenario["source_name"], ",".join(s["parameter"] for s in scenario["signals"]))
        return 0
    if not args.output or not 1 <= args.workers <= 32 or set(args.case) - cases.keys():
        parser.error("Specify --output, valid --case names, and --workers 1-32; see --list")
    output = fresh_output(args.output)
    selected = {n: p for n, p in cases.items() if not args.case or n in args.case}
    if len(selected) == 1:
        report = run_one(next(iter(selected.values())), output)
        print(json.dumps(dict(case=report["case"], all_passed=report["all_passed"])), flush=True)
        return 0 if report["all_passed"] else 1

    def worker(item):
        name, _ = item
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--case", name,
                                      "--output", str(output / name)], stdout=log, stderr=subprocess.STDOUT)
        path = output / name / "verification.json"
        record = json.loads(path.read_text()) if path.is_file() else dict(case=name, all_passed=False)
        record["return_code"] = process.returncode
        print(name, "passed" if record["all_passed"] else "failed", flush=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        from concurrent.futures import as_completed
        futures = [pool.submit(worker, item) for item in selected.items()]
        for future in as_completed(futures):
            records.append(future.result())
            write_json(output / "campaign.json", dict(expected=len(selected), completed=len(records),
                       all_passed=len(records) == len(selected) and all(r["all_passed"] and r["return_code"] == 0 for r in records),
                       timing_scope="Concurrent throughput, not isolated single-run performance", results=records))
    return 0 if all(r["all_passed"] and r["return_code"] == 0 for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
