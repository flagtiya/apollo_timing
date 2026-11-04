"""Test model comparison or observation planning from an existing candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Completed full96 candidate directory, not a reference PAR")
    parser.add_argument("--workflow", choices=("model-comparison", "observation-plan"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.run.resolve(), args.output.expanduser().resolve()
    if output.is_relative_to(HERE.parents[1]) or output.is_relative_to(source) or output.exists():
        parser.error("Use a fresh external output directory, separate from the input candidate")
    output.mkdir(parents=True)
    h = runpy.run_path(str(HERE / "run.py"))
    report = dict(status="failed", workflow=args.workflow,
                  uniqueness_proven=False, input_run=str(source),
                  scope="Conditional postprocessing of a saved candidate, not a new blind search")
    try:
        h["check_result"](source)
        report["input_reload"] = h["independently_reload"](source)
        files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source.rglob("*") if p.is_file()}
        report["input_sha256"] = files
        if args.workflow == "observation-plan":
            from apollo_timing.inference.bayesian_observation_planner import plan_apollo_observations
            from apollo_timing.models.serialization import to_json_ready
            result = plan_apollo_observations(run_output_dir=source, output_dir=output / "plan")
            if not result.recommendations or not result.posterior_modes:
                raise ValueError("No finite candidate-based observation recommendations")
            scores = [r.net_information_nats for r in result.recommendations]
            if not all(math.isfinite(score) for score in scores):
                raise ValueError("Nonfinite observation score")
            last = float(h["rows"](source / "solution.tim")[-1]["mjd"])
            if any(r.window.epoch_mjd <= last for r in result.recommendations):
                raise ValueError("A recommended epoch does not follow the input observations")
            report.update(posterior_mode_count=len(result.posterior_modes),
                          evaluated_window_count=result.evaluated_window_count,
                          recommendation_count=len(result.recommendations),
                          planning_epoch_policy="After last input TOA, reproducible historical forecast, not calendar-current scheduling",
                          qualification=result.qualification_claim)
            (output / "result.json").write_text(json.dumps(to_json_ready(result), indent=2, allow_nan=False) + "\n")
        else:
            from apollo_timing.backend.pint_backend import _load_pint_model, _load_pint_toas
            from apollo_timing.inference.bayesian_binary_model_selection import (
                BinaryModelSelectionConfig, select_binary_model_on_fixed_path,
            )
            from apollo_timing.inference.pint_solution_artifacts import write_pint_model_parfile_roundtrip_safe
            from pint.fitter import WLSFitter
            from pint.residuals import Residuals
            model = _load_pint_model(source / "solution.par")
            toas = _load_pint_toas(source / "solution.tim", model)
            data = h["rows"](source / "solution.tim")
            result = select_binary_model_on_fixed_path(
                baseline_fitter=WLSFitter(toas, model), certified_toas=toas,
                baseline_fit_parameters=tuple(model.free_params),
                observation_span_days=float(data[-1]["mjd"] - data[0]["mjd"]), fit_iterations=10,
                config=BinaryModelSelectionConfig(enabled_models=("ELL1", "DD", "BT")),
            )
            portfolio = list(result.portfolio)
            geometries = {record["model_name"] for record in portfolio
                          if record["status"] in {"baseline", "evaluated"}
                          and not record.get("extension_parameters")}
            # Inspect actual records rather than treating a returned object as success.
            if not {"ELL1", "DD", "BT"}.issubset(geometries):
                raise ValueError("Model comparison omitted a requested representation")
            if any(r["status"] == "failed" for r in portfolio if not r.get("extension_parameters")):
                raise ValueError("A requested geometry comparison failed")
            selected = output / "selected.par"
            write_pint_model_parfile_roundtrip_safe(model=result.fitter.model, path=selected)
            reloaded = _load_pint_model(selected)
            residuals = Residuals(toas, reloaded, subtract_mean=True, use_abs_phase=True,
                                  track_mode="use_pulse_numbers")
            chi2 = float(residuals.chi2)
            if not math.isfinite(chi2) or abs(chi2 - float(result.residuals.chi2)) > .01:
                raise ValueError("Selected model fails independent PAR reload")
            report.update(selected_model=result.selected_model_name, selected_candidate=result.selected_candidate_id,
                          reload_chi2=chi2, evaluated_geometries=sorted(geometries),
                          criterion="BIC on one fixed pulse path, not a model probability")
            (output / "portfolio.json").write_text(json.dumps(portfolio, indent=2, allow_nan=False) + "\n")
        if any(not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
               for path, digest in files.items()):
            raise ValueError("Postprocessing changed the input candidate artifacts")
        report["status"] = "passed"
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return 1
    finally:
        (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
