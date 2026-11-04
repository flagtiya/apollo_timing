"""Run full96 timing workflows with input, path and saved-file checks."""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FIT = {"RAJ", "DECJ", "F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2"}
SEARCH = ["--discovery-preset", "thorough", "--orbital-samples-per-cell", "2",
          "--phase-samples", "12", "--modes-per-cell", "32", "--chart-evaluations", "60",
          "--maximum-modes-to-anneal", "24", "--maximum-modes-to-certify", "16",
          "--blind-workers", "1", "--phase-sigma-schedule", "0.1", "0.05", "0.02",
          "0.01", "0.005", "0.002", "0.001", "0.0005", "0.0002", "0.0001", "0",
          "--no-observation-plan"]
WORKFLOWS = ("solve", "evaluate", "resume", "auto-discovery", "update",
             "model-comparison", "observation-plan")


def rows(path):
    data = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] in {"FORMAT", "C", "#"} or line.startswith("#"):
            continue
        if len(fields) < 5:
            raise ValueError("Unexpected TIM control or malformed observation")
        freq, mjd, error = map(Decimal, fields[1:4])
        if not all(v.is_finite() for v in (freq, mjd, error)) or error <= 0:
            raise ValueError("Nonfinite TOA or nonpositive error")
        flags = fields[5:]
        pn = Decimal(flags[flags.index("-pn") + 1]) if "-pn" in flags else None
        if pn is not None and (not pn.is_finite() or pn != pn.to_integral_value()):
            raise ValueError("Invalid pulse number")
        data.append(dict(name=fields[0], freq=freq, mjd=mjd, error=error,
                         obs=fields[4].lower(), pn=pn, line=line))
    data.sort(key=lambda r: (r["mjd"], r["freq"], r["name"]))
    keys = [(r["mjd"], r["freq"], r["obs"]) for r in data]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate physical TOA")
    return data


def signature(data):
    if not data or any(r["pn"] is None for r in data):
        raise ValueError("Complete integer pulse numbers are required")
    return tuple(int(r["pn"] - data[0]["pn"]) for r in data)


def assert_same_toas(data, reference):
    if len(data) != len(reference):
        raise ValueError("TOA count changed")
    for a, b in zip(data, reference, strict=True):
        if any(a[k] != b[k] for k in ("name", "freq", "error", "obs")):
            raise ValueError("Observation identity or reported error changed")
        # PINT's Decimal -> long-double -> decimal roundtrip changes the last
        # digits. This sub-nanosecond tolerance is not a residual-quality gate.
        if abs(a["mjd"] - b["mjd"]) > Decimal("1e-14"):
            raise ValueError("Observation epoch changed")


def validate_inputs():
    provenance = json.loads((HERE / "provenance.json").read_text())
    for name, record in provenance["files"].items():
        p = HERE / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Reference bundle hash mismatch: {name}")
    data, reference = rows(HERE / "test.tim"), rows(HERE / "reference_solution.tim")
    expected = json.loads((HERE / "expected_metrics.json").read_text())
    sessions = 1 + sum(b["mjd"] - a["mjd"] > Decimal(".5") for a, b in zip(data, data[1:]))
    if len(data) != 96 or sessions != 23:
        raise ValueError("The full96/23-session experiment is required")
    if abs(float(data[-1]["mjd"] - data[0]["mjd"]) - expected["time_span_days"]) > 1e-9:
        raise ValueError("Observation span changed")
    assert_same_toas(data, reference)
    signature(reference)
    from apollo_timing.backend.pint_backend import _load_pint_model
    from apollo_timing.io.timing_input import inspect_timing_system
    declaration = inspect_timing_system(HERE / "clean.par")
    if declaration.binary_model or declaration.orphan_orbit_parameters:
        raise ValueError("Blind inputs must not contain orbital seeds")
    model = _load_pint_model(HERE / "clean.par")
    if float(model.PEPOCH.value) != 59368 or set(model.free_params) != {"RAJ", "DECJ", "F0", "F1"}:
        raise ValueError("Archived baseline parameter contract changed")
    return data, reference, expected


def blind_input_text(pepoch):
    text = (HERE / "clean.par").read_text()
    if pepoch == 59368:
        return text
    return "\n".join("PEPOCH 60604" if line.split()[:1] == ["PEPOCH"] else line
                     for line in text.splitlines()) + "\n"


def check_blind_request(output, pepoch):
    request = json.loads((Path(output) / "request.json").read_text())
    expected_par = hashlib.sha256(blind_input_text(pepoch).encode()).hexdigest()
    expected_tim = hashlib.sha256((HERE / "test.tim").read_bytes()).hexdigest()
    if (request.get("task") != "solve" or request.get("updates")
            or request.get("source_type") not in {"auto", "binary"}
            or request.get("discovery_authorized") is not True
            or request.get("par_sha256") != expected_par
            or request.get("tim_sha256") != expected_tim):
        raise ValueError("Saved request is not the declared minimal-input full96 blind protocol")
    selection = Path(output) / "model_selection.json"
    if not selection.is_file() or json.loads(selection.read_text()).get("execution") != "completed":
        raise ValueError("All emitted system/model attempts must complete before accepting a blind run")
    return dict(request_sha256=hashlib.sha256((Path(output) / "request.json").read_bytes()).hexdigest(),
                source_version=request.get("version"), pepoch_mjd=pepoch,
                minimal_input_hash_verified=True, all_emitted_attempts_complete=True)


def check_result(output, *, fitted=True, quality=True):
    _, reference, expected = validate_inputs()
    cert = json.loads((Path(output) / "certification.json").read_text())
    for key in ("chi2", "disk_reload_chi2", "reduced_chi2", "weighted_rms_us"):
        if not math.isfinite(float(cert[key])):
            raise ValueError(f"Nonfinite {key}")
    data = rows(Path(output) / "solution.tim")
    assert_same_toas(data, reference)
    if signature(data) != signature(reference):
        raise ValueError("Wrong relative pulse-number path")
    model_name = cert.get("model_name") or cert.get("model_identity", {}).get("output")
    if model_name != "ELL1":
        raise ValueError(f"Expected ELL1, got {model_name}")
    if fitted and set(cert.get("fit_parameters", ())) != FIT:
        raise ValueError("Unexpected fitted parameters")
    if quality:
        if cert.get("accepted") is not True:
            raise ValueError("The candidate did not pass certification")
        for key, baseline_key in (("chi2", "pint_chi2"), ("reduced_chi2", "reduced_chi2"),
                                  ("weighted_rms_us", "weighted_rms_us")):
            if not fitted and key == "reduced_chi2":
                # Fixed evaluation has 95 dof, not the 86 dof of a nine-parameter refit.
                continue
            if abs(float(cert[key]) - expected[baseline_key]) > .005:
                raise ValueError(f"{key} differs from the archived result")
    if abs(float(cert["chi2"]) - float(cert["disk_reload_chi2"])) > .01:
        raise ValueError("Disk reload changes the fit")
    return cert


def independently_reload(output):
    from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
    from pint.residuals import Residuals

    output = Path(output)
    model = _load_pint_model(output / "solution.par")
    toas = _load_pint_toas(output / "solution.tim", model)
    residuals = Residuals(toas, model, subtract_mean=True, use_abs_phase=True,
                          track_mode="use_pulse_numbers")
    _, _, expected = validate_inputs()
    chi2 = float(residuals.chi2)
    wrms = float(residuals.rms_weighted().to_value("us"))
    if (not math.isfinite(chi2) or not math.isfinite(wrms)
            or abs(chi2 - expected["disk_reload_chi2"]) > .005
            or abs(wrms - expected["weighted_rms_us"]) > .005):
        raise ValueError("Independent PINT reload disagrees with the full96 reference quality")
    return dict(chi2=chi2, weighted_rms_us=wrms, toa_count=len(toas.table))


def split_update(data):
    starts = [0] + [i for i in range(1, len(data))
                    if data[i]["mjd"] - data[i-1]["mjd"] > Decimal(".5")]
    cut = starts[-3]
    return data[:cut], data[cut:]


def audit_portfolio(output):
    """Count physical paths only after inspecting every saved candidate TIM."""
    _, reference, _ = validate_inputs()
    accepted, records = set(), []
    for portfolio in sorted(Path(output).rglob("candidate_portfolio.json")):
        payload = json.loads(portfolio.read_text())
        if not isinstance(payload, list):
            continue
        for record in payload:
            path = (record.get("artifacts") or {}).get("tim")
            if not path:
                continue
            path = Path(path)
            if not path.is_absolute():
                path = portfolio.parent / path
            data = rows(path)
            assert_same_toas(data, reference)
            pulse_path = signature(data)
            matches = pulse_path == signature(reference)
            is_accepted = record.get("accepted") is True
            if is_accepted:
                if (float(record["reduced_chi2"]) >= 20
                        or abs(float(record["chi2"]) - float(record["disk_reload_chi2"])) > .01):
                    raise ValueError("A saved accepted candidate violates certification gates")
                accepted.add(pulse_path)
            records.append(dict(candidate_id=record.get("candidate_id"),
                                model=record.get("model_name"), accepted=is_accepted,
                                reference_path=matches, chi2=record.get("chi2"),
                                artifact=str(path)))
    return dict(accepted_distinct_paths=len(accepted),
                accepted_reference_paths=int(signature(reference) in accepted),
                uniqueness_proven=False, scope="saved_finite_candidate_portfolio",
                candidates=records)


def check_table1_parameters(path):
    from apollo_timing.backend.pint_backend import _load_pint_model

    model = _load_pint_model(path)
    if abs(float(model.PEPOCH.value) - 60604) > 1e-9:
        raise ValueError("This parameter check requires the independently rerun PEPOCH 60604 model")
    # Three printed marginal standard errors, with no error rescaling.
    expected = {
        "RAJ": ((19*3600 + 12*60 + 34.49733) / 3600, .00002 / 3600),
        "DECJ": (-(12*3600 + 15*60 + 21.980) / 3600, .002 / 3600),
        "F0": (189.047239445402, 8e-12), "F1": (-3.024e-16, 2e-19),
        "PB": (17.526125680, 7e-9), "A1": (23.4476454, 6e-7),
        "EPS1": (-2.558e-5, 6e-8), "EPS2": (6.76e-6, 6e-8),
    }
    for name, (value, sigma) in expected.items():
        if abs(float(model[name].value) - value) > 3*sigma:
            raise ValueError(f"{name} differs from the epoch-60604 reference by more than three standard errors")
    cycles = round((60606.87637480 - float(model.TASC.value)) / float(model.PB.value))
    tasc = float(model.TASC.value) + cycles * float(model.PB.value)
    propagated_sigma = (1.4e-7**2 + (cycles * 7e-9)**2)**.5
    if abs(tasc - 60606.87637480) > 3*propagated_sigma:
        raise ValueError("Orbital-cycle-equivalent TASC disagrees with the epoch-60604 reference")
    if float(model.DM.value) != 39.48 or not model.DM.frozen:
        raise ValueError("The epoch-60604 timing fit requires fixed DM 39.48")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workflow", choices=WORKFLOWS, default="solve")
    parser.add_argument("--check-run", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--pepoch", type=int, choices=(59368, 60604), default=59368,
                        help="Original epoch 59368, or a separate epoch-60604 check")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(HERE.parents[1]):
        parser.error("Write outputs outside the source tree")
    if output.exists() and any(output.iterdir()):
        parser.error("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="failed", workflow=args.workflow, reference_solution_checked=False,
                  epoch_parameter_check_passed=False, pepoch_mjd=args.pepoch, uniqueness_proven=False)
    try:
        data, reference, _ = validate_inputs()
        if args.verify_only:
            report.update(status="inputs_verified_not_run", toa_count=96, session_count=23)
            return 0
        if args.check_run:
            report["blind_protocol"] = check_blind_request(args.check_run, args.pepoch)
            cert = check_result(args.check_run)
            if args.pepoch == 60604:
                check_table1_parameters(args.check_run / "solution.par")
            report.update(status="passed", reference_solution_checked=True, chi2=cert["chi2"],
                          epoch_parameter_check_passed=args.pepoch == 60604,
                          independent_reload=independently_reload(args.check_run),
                          pulse_path_matches_all_96=True, portfolio=audit_portfolio(args.check_run))
            return 0

        input_par = HERE / "clean.par"
        if args.pepoch == 60604:
            if args.workflow in {"update", "evaluate"}:
                raise ValueError("These reference-seeded mechanics use PEPOCH 59368, not 60604")
            input_par = output / "blind_pepoch60604.par"
            input_par.write_text(blind_input_text(args.pepoch))
            report["initial_epoch_policy"] = "change_PEPOCH_only_keep_archived_minimal_values_no_reference_seed"

        def timing(par, tim, dest, *options, source="binary"):
            command = [sys.executable, "-m", "apollo_timing.pipeline.apollo_timing",
                       "--par", str(par), "--tim", str(tim), "--output", str(dest),
                       "--source-type", source, *SEARCH, *options]
            with (output / "commands.jsonl").open("a") as stream:
                stream.write(json.dumps(command) + "\n")
            subprocess.run(command, check=True)

        run = output / "timing"
        if args.workflow == "evaluate":
            timing(HERE / "reference_solution.par", HERE / "reference_solution.tim", run, "--task", "evaluate")
            cert = check_result(run, fitted=False)
            if cert.get("fit_parameters"):
                raise ValueError("Evaluate must not fit parameters")
            report["scope"] = "fixed_reference_model_and_pulse_path_evaluation"
        elif args.workflow == "update":
            baseline, added = split_update(reference)
            for path, subset in ((output / "baseline.tim", baseline), (output / "added.tim", added)):
                path.write_text("FORMAT 1\n" + "\n".join(r["line"] for r in subset) + "\n")
            timing(HERE / "reference_solution.par", output / "baseline.tim", run,
                   "--task", "update", "--update-tim", str(output / "added.tim"))
            check_result(run)
            report["scope"] = "reference_seeded_whole_session_update_not_blind_discovery"
        else:
            options = ["--binary-model-selection", "--binary-models", "ELL1", "DD"] if args.workflow == "model-comparison" else []
            timing(input_par, HERE / "test.tim", run, *options,
                   source="auto" if args.workflow == "auto-discovery" else "binary")
            report["blind_protocol"] = check_blind_request(run, args.pepoch)
            check_result(run)
            report["reference_solution_checked"] = True
            report["portfolio"] = audit_portfolio(run)
            if args.pepoch == 60604:
                check_table1_parameters(run / "solution.par")
            report["epoch_parameter_check_passed"] = args.pepoch == 60604
            report["scope"] = "full96_blind_solve_epoch60604" if args.pepoch == 60604 else "full96_blind_solve_epoch59368"
            if args.workflow == "resume":
                timing(input_par, HERE / "test.tim", run, "--resume")
                check_result(run)
            if args.workflow == "auto-discovery":
                selection = json.loads((run / "model_selection.json").read_text())
                if selection["selected_family"] != "binary":
                    raise ValueError("Automatic system comparison did not select binary")
            if args.workflow == "observation-plan":
                from apollo_timing.inference.bayesian_observation_planner import plan_apollo_observations
                result = plan_apollo_observations(run_output_dir=run, output_dir=output / "plan")
                if not result.recommendations:
                    raise ValueError("No observation recommendations")
        report["independent_reload"] = independently_reload(run)
        report["reference_solution_checked"] = True
        report["status"] = "passed"
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
        return 1
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
