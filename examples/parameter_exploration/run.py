"""Real-PINT injection/null checks, conditional on known pulse numbers."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from io import StringIO
import json
import math
from pathlib import Path

import numpy as np
from astropy import units as u
from pint.models import get_model
from pint.simulation import make_fake_toas_fromMJDs

from apollo_timing.backend.pint_backend import _prime_local_ephemeris
from apollo_timing.inference.timing_parameter_exploration import (
    ParameterExplorationConfig, explore_parameters_on_fixed_path,
)

BASE = (
    "PSRJ J1200+1000\nRAJ 12:00:00\nDECJ 10:00:00\n"
    "F0 100\nF1 0\nPEPOCH 59000\nPOSEPOCH 59000\n"
    "DM 10\nDMEPOCH 59000\nEPHEM DE421\nCLOCK TT(TAI)\nUNITS TDB\n"
)
ELL1 = "PB 3\nA1 1\nTASC 59000\nEPS1 .001\nEPS2 .002\n"
DD = "PB 3\nA1 1\nT0 59000\nECC .1\nOM 40\n"
ORBITS = {
    "single": "",
    "ecliptic": "",
    "ELL1": "BINARY ELL1\n" + ELL1,
    "ELL1_FB": "BINARY ELL1\n" + ELL1.replace("PB 3", "FB0 0.000003858024691358025"),
    "ELL1k": "BINARY ELL1k\n" + ELL1,
    "BT": "BINARY BT\n" + DD,
    "DD": "BINARY DD\n" + DD + "M2 .5\nSINI .7\n",
    "DDS": "BINARY DDS\n" + DD + "M2 .5\nSHAPMAX 1\n",
    "DDH": "BINARY DDH\n" + DD + "H3 0.000002\nSTIGMA .5\n",
    "ELL1H": "BINARY ELL1H\n" + ELL1 + "H3 0.000002\nSTIGMA .5\n",
    "ELL1H_H4": "BINARY ELL1H\n" + ELL1 + "H3 0.000002\nH4 0.000001\n",
    "DDGR": "BINARY DDGR\n" + DD + "M2 .5\nMTOT 1.9\n",
    "DDK": "BINARY DDK\n" + DD + "M2 .5\nKIN 60\nKOM 30\nPX 10\nPMRA 20\nPMDEC 10\n",
    "frequency": "FD1 0\nFD2 0\nFD3 0\nFD4 0\n",
    "chromatic": "CM 0\nCM1 0\nCM2 0\nCM3 0\nCM4 0\nCMEPOCH 59000\nTNCHROMIDX 4\n",
    "solarwind": "NE_SW 4\nNE_SW1 0\nNE_SW2 0\nSWEPOCH 59000\nSWM 0\n",
    "glitch": "GLEP_1 59000\nGLPH_1 0\nGLF0_1 0\nGLF1_1 0\nGLF2_1 0\nGLF0D_1 0\nGLTD_1 100\n",
    "dmx": "DMX_0001 0\nDMXR1_0001 58700\nDMXR2_0001 59300\n",
    "piecewise": "PWSTART_1 58700\nPWSTOP_1 59300\nPWEP_1 59000\nPWPH_1 0\nPWF0_1 0\nPWF1_1 0\nPWF2_1 0\n",
}


@dataclass(frozen=True)
class Case:
    model: str
    parameter: str
    increment: float

    @property
    def name(self):
        return f"{self.model}-{self.parameter}"


def cases() -> tuple[Case, ...]:
    result = []
    # A fixed finite test range, not a claim that all indexed orders exist.
    for order in range(2, 13):
        amplitude = .02 * math.factorial(order + 1) / (730 * 86400.) ** (order + 1)
        result.append(Case("single", f"F{order}", amplitude))
    result += [Case("single", f"DM{order}", .03 * math.factorial(order) / 2.0**order) for order in range(1, 13)]
    result += [Case("single", p, v) for p, v in (
        ("RAJ", 1e-5), ("DECJ", .0001), ("PMRA", 20), ("PMDEC", 20), ("PX", 10),
    )]
    result += [Case("ecliptic", p, v) for p, v in (
        ("ELONG", .0001), ("ELAT", .0001), ("PMELONG", 20), ("PMELAT", 20),
    )]
    for model in ("ELL1", "BT", "DD"):
        result += [Case(model, "PBDOT", 1e-8), Case(model, "A1DOT", 1e-12)]
    result += [Case("ELL1", p, .1) for p in ("EPS1DOT", "EPS2DOT")]
    result += [Case("DD", p, v) for p, v in (
        ("EDOT", 1e-12), ("OMDOT", .01), ("GAMMA", .0001),
        ("DR", .1), ("DTH", .0001), ("M2", .1), ("SINI", .05),
    )]
    result += [Case("BT", "EDOT", 1e-12), Case("BT", "OMDOT", .01), Case("BT", "GAMMA", .0001)]
    for model in ("ELL1H", "DDH"):
        result += [Case(model, "H3", 1e-6), Case(model, "STIGMA", .05)]
    result += [Case("ELL1H_H4", "H4", 8e-7), Case("DDS", "M2", .1), Case("DDS", "SHAPMAX", .1)]
    result += [Case("DDGR", "M2", .1), Case("DDGR", "MTOT", .1), Case("DDGR", "XPBDOT", 1e-8)]
    result += [Case("DDK", "KIN", 5), Case("DDK", "KOM", 40)]
    result += [Case("ELL1k", "LNEDOT", .001), Case("ELL1k", "OMDOT", .01)]
    for order in range(1, 5):
        result.append(Case("ELL1_FB", f"FB{order}", .001 * math.factorial(order + 1) / (730 * 86400.) ** (order + 1)))
    result += [Case("frequency", f"FD{i}", 1e-4) for i in range(1, 5)]
    # PINT CM has two additional inverse-MHz powers relative to DM.
    result += [Case("chromatic", p, .03 * 1400**2) for p in ("CM", "CM1", "CM2", "CM3", "CM4")]
    result += [Case("solarwind", p, 10) for p in ("NE_SW", "NE_SW1", "NE_SW2")]
    result += [Case("glitch", p, v) for p, v in (
        ("GLF0_1", 1e-8), ("GLF1_1", 1e-16), ("GLF2_1", 1e-23), ("GLF0D_1", 1e-8),
    )]
    result += [Case("dmx", "DMX_0001", .03)]
    result += [Case("piecewise", p, v) for p, v in (
        ("PWPH_1", .01), ("PWF0_1", 1e-8), ("PWF1_1", 1e-15), ("PWF2_1", 1e-22),
    )]
    return tuple(result)


def make_null(case: Case):
    base = BASE
    if case.model == "ecliptic":
        base = base.replace("RAJ 12:00:00\nDECJ 10:00:00", "ELONG 180\nELAT 20")
    text = base + ORBITS[case.model]
    # PINT requires consecutive indexed derivatives for these components.
    for prefix in ("F", "DM", "FB"):
        suffix = case.parameter.removeprefix(prefix)
        if case.parameter.startswith(prefix) and suffix.isdigit():
            start = 2 if prefix == "F" else 1
            text += "".join(f"{prefix}{i} 0\n" for i in range(start, int(suffix) + 1))
            break
    model = get_model(StringIO(text))
    if case.parameter not in model.params:
        raise ValueError(f"{case.name} is not implemented by the installed PINT model")
    if model[case.parameter].value is None:
        model[case.parameter].value = 0
    model.free_params = []
    return model


def run_case(case: Case, injected: bool) -> dict:
    _prime_local_ephemeris("DE421")
    null = make_null(case)
    truth = deepcopy(null)
    center = float(null[case.parameter].value)
    value = center + (case.increment if injected else 0)
    truth[case.parameter].quantity = value * truth[case.parameter].units
    # Noncommensurate cadence, several years and simultaneous frequency coverage.
    times = 59000 + np.repeat(np.linspace(-730, 730, 48), 3)
    frequencies = np.tile([700., 1400., 2100.], 48)
    toas = make_fake_toas_fromMJDs(
        times, truth, obs="geocenter", freq=frequencies * u.MHz,
        error=.1 * u.us, add_noise=False, include_bipm=False,
    )
    toas.compute_pulse_numbers(truth)
    # Initialize PINT's lazy absolute-phase reference before checking immutability.
    null.phase(toas, abs_phase=True)
    original_pn = np.asarray(toas.table["pulse_number"]).copy()
    original = {name: str(null[name].quantity) for name in null.params}
    radius = 4 * abs(case.increment)
    lower, upper = center - radius, center + radius
    # Physical domains are part of the declared test case, never inferred from fit quality.
    if case.parameter in ("SINI", "STIGMA"):
        lower, upper = max(.05, lower), min(.95, upper)
    if case.parameter in ("M2", "MTOT", "H3", "H4", "SHAPMAX", "KIN"):
        lower = max(lower, center * .1)
    fitted, _, report, covariance = explore_parameters_on_fixed_path(
        model=null, toas=toas,
        config=ParameterExplorationConfig(
            parameters=(case.parameter,), search_bounds=((case.parameter, lower, upper),),
            maximum_evaluations=500,
        ), input_bounds={},
    )
    assert report["status"] == "evaluated", report
    selected = report["selected_candidate_id"] != "baseline"
    assert selected == injected, report
    assert original == {name: str(null[name].quantity) for name in null.params}
    np.testing.assert_array_equal(toas.table["pulse_number"], original_pn)
    recovery_sigma = None
    if injected:
        assert covariance is not None and covariance[0, 0] > 0
        recovered = float(fitted[case.parameter].value)
        recovery_sigma = abs(recovered - value) / math.sqrt(covariance[0, 0])
        assert recovery_sigma < 3, (case.name, recovered, value, recovery_sigma)
        assert abs(recovered - value) < .02 * abs(case.increment), (case.name, recovered, value)
    return dict(case=asdict(case), injected=injected, status="passed",
                recovery_sigma=recovery_sigma, comparison=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    output = args.output.expanduser().resolve()
    if output.is_relative_to(source):
        parser.error("Outputs must be outside the source tree")
    selected = [c for c in cases() if not args.case or c.name in args.case]
    if not selected or set(args.case) - {c.name for c in cases()}:
        parser.error("Unknown case; available cases: " + ", ".join(c.name for c in cases()))
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory; preserve earlier failures")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for case in selected:
        for injected in (False, True):
            try:
                record = run_case(case, injected)
            except Exception as error:
                record = dict(case=asdict(case), injected=injected, status="failed",
                              error=f"{type(error).__name__}: {error}")
            records.append(record)
            print(case.name, "injected" if injected else "null", record["status"], flush=True)
            (output / "results.json").write_text(json.dumps(dict(
                scope="fixed_path_real_pint_injection_and_null_not_paper_reproduction",
                all_passed=all(r["status"] == "passed" for r in records),
                completed=len(records), expected=2*len(selected), results=records,
            ), indent=2, default=str) + "\n")
    return 0 if all(r["status"] == "passed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
