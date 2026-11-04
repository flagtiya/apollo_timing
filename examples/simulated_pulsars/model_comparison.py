"""Noisy known-path examples for eccentric geometry and Shapiro model upgrades."""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from astropy import units as u
from pint.binaryconvert import convert_binary
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.simulation import make_fake_toas_fromMJDs
from loguru import logger

from apollo_timing.backend.pint_backend import _load_pint_toas, _prime_local_ephemeris
from apollo_timing.inference.bayesian_binary_model_selection import (
    BinaryModelSelectionConfig, select_binary_model_on_fixed_path,
)
from apollo_timing.inference.pint_solution_artifacts import (
    extract_fitter_parameter_covariance, write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.inference.toa_white_noise_certification import evaluate_fixed_path_normalized_likelihood
from run import compare_toas, fresh_output, write_json


def run_case(kind, injected, output):
    output.mkdir()
    base = ("PSRJ SIM097\nRAJ 12:00:00\nDECJ 10:00:00\nF0 100\nF1 0\nPEPOCH 59000\n"
            "DM 10\nEPHEM DE421\nCLOCK TT(TAI)\nUNITS TDB\n")
    if kind == "eccentric-model":
        truth = get_model(StringIO(base + ("BINARY DD\nPB 3\nA1 1\nT0 59000\nECC .1\nOM 40\n" if injected else
                                         "BINARY ELL1\nPB 3\nA1 1\nTASC 59000\nEPS1 .0001\nEPS2 .0002\n")))
        initial = convert_binary(truth, "ELL1") if injected else deepcopy(truth)
        models, expected = ("ELL1", "DD"), "DD" if injected else "ELL1"
    else:
        truth = get_model(StringIO(base.replace("SIM097", "SIM098") +
                    "BINARY ELL1H\nPB 3\nA1 1\nTASC 59000\nEPS1 .0001\nEPS2 .0002\nH3 .000006\nSTIGMA .7\n")) if injected else get_model(StringIO(
                    base.replace("SIM097", "SIM098") + "BINARY ELL1\nPB 3\nA1 1\nTASC 59000\nEPS1 .0001\nEPS2 .0002\n"))
        initial = convert_binary(truth, "ELL1") if injected else deepcopy(truth)
        # Conversion can preserve Shapiro terms; this null explicitly excludes them.
        for name in ("M2", "SINI"):
            if name in initial.params:
                initial[name].value = 0
        models, expected = ("ELL1", "ELL1H"), "ELL1H" if injected else "ELL1"
    initial.free_params = ["F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2"]
    truth.free_params = []
    _prime_local_ephemeris("DE421")
    np.random.seed(31415)
    epochs = 59000 + np.repeat(np.linspace(-730, 730, 80), 3)
    toas = make_fake_toas_fromMJDs(epochs, truth, obs="geocenter", freq=np.tile([700, 1400, 2100], 80) * u.MHz,
                                  error=1 * u.us, add_noise=True, include_bipm=False)
    toas.compute_pulse_numbers(truth)
    initial.phase(toas, abs_phase=True)
    write_pint_model_parfile_roundtrip_safe(model=truth, path=output / "truth.par")
    write_pint_model_parfile_roundtrip_safe(model=initial, path=output / "initial.par")
    tim = output / "observations.tim"
    toas.write_TOA_file(str(tim), format="Tempo2", include_pn=True)
    reloaded = _load_pint_toas(tim, seed_model=initial, clock_override_path=None)
    compare_toas(toas, reloaded)
    toas = reloaded
    original = deepcopy(toas)
    fitter = WLSFitter(toas, initial, track_mode="use_pulse_numbers")
    fitter.fit_toas(maxiter=10)
    baseline = evaluate_fixed_path_normalized_likelihood(model=fitter.model, toas=toas)
    write_pint_model_parfile_roundtrip_safe(model=fitter.model, path=output / "baseline_fitted.par")
    result = select_binary_model_on_fixed_path(
        baseline_fitter=fitter, certified_toas=toas, baseline_fit_parameters=fitter.model.free_params,
        observation_span_days=1460, fit_iterations=10, config=BinaryModelSelectionConfig(enabled_models=models),
    )
    # Persist failed comparisons as evidence, not only successful selections.
    write_json(output / "comparison.json", dict(selected_model=result.selected_model_name, portfolio=list(result.portfolio)))
    assert not any(r["status"] == "failed" for r in result.portfolio), result.portfolio
    assert result.selected_model_name == expected, result.portfolio
    selected = evaluate_fixed_path_normalized_likelihood(model=result.fitter.model, toas=toas)
    assert .5 < selected.reduced_chi2 < 1.5
    covariance = extract_fitter_parameter_covariance(result.fitter, result.fit_parameters)
    correlation = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    assert np.min(np.linalg.eigvalsh(correlation)) > 0
    recovery = {}
    for name in result.fit_parameters:
        if name not in truth.params or truth[name].value is None:
            raise AssertionError(f"The selected parameter {name} has no declared truth")
        delta = abs(float(result.fitter.model[name].value) - float(truth[name].value))
        sigma = float(result.fitter.model[name].uncertainty_value)
        recovery[name] = dict(expected=float(truth[name].value),
                              recovered=float(result.fitter.model[name].value),
                              uncertainty=sigma, error_sigma=delta / sigma)
        assert delta < 4 * sigma, recovery[name]
        if injected and name in ("ECC", "H3", "STIGMA"):
            assert delta < .05 * abs(float(truth[name].value)), recovery[name]
    solution = output / "solution.par"
    write_pint_model_parfile_roundtrip_safe(model=result.fitter.model, path=solution)
    restored = get_model(solution)
    restored_toas = _load_pint_toas(tim, seed_model=restored, clock_override_path=None)
    compare_toas(original, restored_toas)
    final = evaluate_fixed_path_normalized_likelihood(model=restored, toas=restored_toas)
    assert abs(final.chi2-selected.chi2) < .01
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    for ax, residuals, label in zip(axes, (fitter.resids, result.residuals), ("ELL1 baseline", f"Selected {expected}")):
        ax.errorbar(toas.get_mjds().value, residuals.time_resids.to_value(u.us), yerr=toas.get_errors().to_value(u.us), fmt=".", ms=3)
        ax.set_ylabel("Residual (us)")
        ax.set_title(label, loc="left", fontsize=10)
    axes[1].set_xlabel("MJD")
    figure.savefig(output / "residuals.png", dpi=140)
    plt.close(figure)
    return dict(passed=True, injected=injected, expected_model=expected, selected_model=result.selected_model_name,
                baseline=baseline.to_payload(), selected=selected.to_payload(), independent_reload=final.to_payload(),
                delta_bic=baseline.bic-selected.bic, toa_count=len(toas.table),
                recovery=recovery, pulse_path_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("eccentric-model", "shapiro-model"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = fresh_output(args.output)
    logger.remove()
    logger.add(os.sys.stderr, level="WARNING")
    results = []
    for injected in (False, True):
        try:
            result = run_case(args.case, injected, output / ("injected" if injected else "null"))
        except Exception as error:
            import traceback
            traceback.print_exc()
            result = dict(passed=False, injected=injected, error=f"{type(error).__name__}: {error}")
        results.append(result)
    report = dict(all_passed=all(r["passed"] for r in results), case=args.case, seed=31415,
                  scope="known_pulse_path_noisy_physical_model_comparison_not_blind_timing", results=results)
    write_json(output / "verification.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
