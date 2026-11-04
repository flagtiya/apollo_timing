"""Noisy known-path demonstrations of a coupled 1.5-arcmin position cap."""

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
from astropy.coordinates import SkyCoord
from pint.models import get_model
from pint.simulation import make_fake_toas_fromMJDs

from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas, _prime_local_ephemeris
from apollo_timing.inference.pint_solution_artifacts import (
    fit_pint_model_with_physical_bounds, write_pint_model_parfile_roundtrip_safe,
)
from apollo_timing.inference.timing_constraints import SkyPositionConstraint
from apollo_timing.inference.toa_white_noise_certification import evaluate_fixed_path_normalized_likelihood
from apollo_timing.pipeline.automatic_timing import _same_complete_toas
from apollo_timing.reports.residual_plot import write_residual_plot


CASES = {
    "single-equatorial": ("SIM100", False, False, 120., 30.),
    "binary-equatorial": ("SIM101", True, False, 240., -40.),
    "single-wrap-highlat": ("SIM102", False, False, 359.998, 80.),
    "binary-ecliptic": ("SIM103", True, True, 45., 45.),
}


def run_case(output: Path, case: str, outside: bool) -> dict:
    output.mkdir(parents=True)
    source, binary, ecliptic, lon, lat = CASES[case]
    _prime_local_ephemeris("DE421")
    lon_name, lat_name = ("ELONG", "ELAT") if ecliptic else ("RAJ", "DECJ")
    position = (f"ELONG {lon} 1\nELAT {lat} 1\n" if ecliptic else
                f"RAJ {lon / 15} 1\nDECJ {lat} 1\n")
    orbit = "BINARY ELL1\nPB 3.7\nA1 1.2\nTASC 59000\nEPS1 0.0001\nEPS2 0.0002\n" if binary else ""
    initial = get_model(StringIO(f"PSRJ {source}\nF0 100 1\nF1 -1e-15 1\n{position}{orbit}"
                                "DM 10\nPEPOCH 59000\nPOSEPOCH 59000\nEPHEM DE421\nCLOCK TT(TAI)\nUNITS TDB\n"))
    initial_par = write_pint_model_parfile_roundtrip_safe(model=initial, path=output / "initial.par")
    region = SkyPositionConstraint.from_par(initial_par, 1.5)
    truth = deepcopy(initial)
    offset = SkyCoord(lon*u.deg, lat*u.deg).directional_offset_by(45*u.deg, (1.8 if outside else 1.2)*u.arcmin)
    truth[lon_name].quantity = (offset.ra.degree / (1 if ecliptic else 15)) * truth[lon_name].units
    truth[lat_name].quantity = offset.dec.degree * truth[lat_name].units
    truth.free_params = []
    write_pint_model_parfile_roundtrip_safe(model=truth, path=output / "truth.par")
    old_rng = np.random.get_state()
    try:
        np.random.seed(1729)
        generated = make_fake_toas_fromMJDs(np.linspace(58270, 59730, 96), truth, obs="geocenter",
                                           freq=1400*u.MHz, error=2*u.us, add_noise=True, include_bipm=False)
    finally:
        np.random.set_state(old_rng)
    generated.compute_pulse_numbers(truth)
    initial.phase(generated, abs_phase=True)
    tim = output / "observations.tim"
    generated.write_TOA_file(str(tim), format="Tempo2", include_pn=True)
    toas = _load_pint_toas(tim, seed_model=initial, clock_override_path=None)
    if not _same_complete_toas(generated, toas):
        raise AssertionError("Serialized simulation changed TOAs or errors")
    np.testing.assert_array_equal(generated.table["pulse_number"], toas.table["pulse_number"])
    names = tuple(initial.free_params)
    cap_intervals = region.bounding_intervals(region.model_values(initial))
    # Both fits use the same explicit finite box, wider than the cap.
    intervals = {"F0": (100-1e-6, 100+1e-6), "F1": (-1e-13, 1e-13)}
    for name, (low, high) in cap_intervals.items():
        center = float(initial[name].value)
        intervals[name] = (center + 2*(low-center), center + 2*(high-center))
    kwargs = dict(toas=toas, parameter_names=names,
                  lower_bounds=[intervals[n][0] for n in names], upper_bounds=[intervals[n][1] for n in names])
    unconstrained = fit_pint_model_with_physical_bounds(model=deepcopy(initial), **kwargs)
    constrained = fit_pint_model_with_physical_bounds(model=deepcopy(initial), sky_constraint=region, **kwargs)
    for fit, filename in ((unconstrained, "unconstrained.par"), (constrained, "solution.par")):
        write_pint_model_parfile_roundtrip_safe(model=fit.model, path=output / filename)
    reloaded = _load_pint_model(output / "solution.par")
    reloaded_toas = _load_pint_toas(tim, seed_model=reloaded, clock_override_path=None)
    if not _same_complete_toas(toas, reloaded_toas):
        raise AssertionError("Independent reload changed observations or errors")
    np.testing.assert_array_equal(toas.table["pulse_number"], reloaded_toas.table["pulse_number"])
    metrics = evaluate_fixed_path_normalized_likelihood(model=reloaded, toas=reloaded_toas)
    report = region.report(reloaded)
    assert report["within_bounds"]
    assert abs(metrics.chi2 - float(constrained.residuals.chi2)) < .01
    assert set(reloaded.free_params) == set(initial.free_params)
    assert .5 < float(unconstrained.residuals.reduced_chi2) < 1.5
    recovery = {}
    for index, name in enumerate(names):
        difference = float(unconstrained.model[name].value) - float(truth[name].value)
        if name == lon_name:
            period = 360 if ecliptic else 24
            difference = (difference + period/2) % period - period/2
        sigma = abs(difference) / np.sqrt(unconstrained.covariance[index, index])
        recovery[name] = sigma
        assert sigma < 4, (case, name, sigma)
    if outside:
        assert region.report(unconstrained.model)["separation_arcmin"] > 1.5
        assert report["active_bound"] and "SKY_POSITION" in constrained.active_bounds
        assert metrics.reduced_chi2 > 20
    else:
        assert not report["active_bound"] and not constrained.active_bounds
        assert abs(float(constrained.residuals.chi2) - float(unconstrained.residuals.chi2)) < .01
        assert .5 < metrics.reduced_chi2 < 1.5
    for filename, stem in (("initial.par", "before"), ("solution.par", "after")):
        write_residual_plot(output / filename, tim, output, clock_override_path=None, artifact_stem=stem)
    result = dict(status="passed", case=case, source_name=source, outside_truth=outside,
                  scope="known_pulse_path_joint_continuous_fit_not_blind_connection_or_uniqueness",
                  seed=1729, toa_count=len(toas.table), error_us=2, parameters=names,
                  input_bounds=intervals, sky_position_constraint=report,
                  active_bounds=constrained.active_bounds, covariance_parameter_names=names,
                  parameter_covariance=constrained.covariance.tolist(),
                  covariance_scope="local_unconstrained_curvature_not_boundary_confidence_region",
                  unconstrained_chi2=float(unconstrained.residuals.chi2),
                  independent_reload=metrics.to_payload(), recovery_sigma=recovery,
                  observations_errors_and_pulse_numbers_preserved=True,
                  fit_acceptable_within_requested_region=not outside, uniqueness_proven=False)
    (output / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), action="append")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    root = Path(__file__).resolve().parents[2]
    if output.is_relative_to(root) or (output.exists() and any(output.iterdir())):
        parser.error("Use a fresh output directory outside the source tree")
    output.mkdir(parents=True, exist_ok=True)
    results = [run_case(output / case / label, case, outside)
               for case in (args.case or CASES) for label, outside in (("inside", False), ("outside", True))]
    (output / "verification.json").write_text(json.dumps(dict(status="passed", checks=len(results), results=results), indent=2) + "\n")
    print(json.dumps(dict(status="passed", checks=len(results), output=str(output))))


if __name__ == "__main__":
    main()
