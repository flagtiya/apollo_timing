from __future__ import annotations

import argparse
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from timing_accelerator.inference.run_nested_engine import run_nested_engine
from timing_accelerator.io.save_artifacts import ensure_output_dir, write_json, write_text


def _resolve_repo_root() -> Path:
    explicit = os.environ.get("TIMING_ACCELERATOR_REPO_ROOT")
    if explicit:
        path = Path(explicit).resolve()
        if (path / "examples").exists():
            return path

    preferred = Path("/opt/timing-accelerator")
    if (preferred / "examples").exists():
        return preferred

    here = Path(__file__).resolve()
    for candidate in [here.parents[3], here.parents[2], Path.cwd()]:
        if (candidate / "examples").exists():
            return candidate

    raise FileNotFoundError("Unable to resolve timing-accelerator repository root.")


REPO_ROOT = _resolve_repo_root()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _load_numeric_parameter(path: Path, name: str) -> float:
    target = name.upper()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if parts[0].upper() != target:
            continue
        return float(parts[1])
    raise KeyError(f"Parameter '{name}' not found in '{path}'.")


def _validate_isolated_run(result) -> dict[str, object]:
    issues: list[str] = []
    if result.best_state is None:
        issues.append("best_state is missing")
    elif result.best_state.structure_id != "isolated_f0_f1":
        issues.append(f"unexpected structure_id: {result.best_state.structure_id}")
    if not (result.uniqueness_gate_report or {}).get("passed"):
        issues.append("uniqueness gate did not pass")
    if result.unique_solution_par_path is None or not result.unique_solution_par_path.exists():
        issues.append("best_unique_solution.par is missing")

    truth_par = REPO_ROOT / "examples/literature_47tucaa/truth.par"
    if result.refined_best_state is not None:
        refined = result.refined_best_state.active_theta or {}
        f0 = refined.get("F0")
        f1 = refined.get("F1")
        if f0 is None or abs(float(f0) - _load_numeric_parameter(truth_par, "F0")) > 1.0e-8:
            issues.append("F0 recovery exceeded tolerance")
        if f1 is None or abs(float(f1) - _load_numeric_parameter(truth_par, "F1")) > 1.0e-15:
            issues.append("F1 recovery exceeded tolerance")
    else:
        issues.append("refined_best_state is missing")

    return {"status": "success" if not issues else "failure", "issues": issues}


def _validate_fast_run(result) -> dict[str, object]:
    issues: list[str] = []
    if result.best_state is None:
        issues.append("best_state is missing")
    if not (result.uniqueness_gate_report or {}).get("passed"):
        issues.append("uniqueness gate did not pass")
    if result.unique_solution_par_path is None or not result.unique_solution_par_path.exists():
        issues.append("best_unique_solution.par is missing")
    else:
        input_dm = _load_numeric_parameter(REPO_ROOT / "examples/19C101ell1.par", "DM")
        output_dm = _load_numeric_parameter(result.unique_solution_par_path, "DM")
        if abs(output_dm - input_dm) > 1.0e-12:
            issues.append("fixed DM changed in output par")

    best_payload_path = result.output_dir / "best_hypothesis.json"
    if best_payload_path.exists():
        payload = json.loads(best_payload_path.read_text(encoding="utf-8"))
        comparison = payload.get("reference_comparison")
        if comparison is None:
            issues.append("reference comparison is missing")
        else:
            if float(comparison["reduced_chi2_ratio_to_reference"]) > 1.2:
                issues.append("reduced chi2 ratio exceeded threshold")
            if float(comparison["weighted_rms_ratio_to_reference"]) > 1.2:
                issues.append("weighted RMS ratio exceeded threshold")
    else:
        issues.append("best_hypothesis.json is missing")

    return {"status": "success" if not issues else "failure", "issues": issues}


def _run_case(
    label: str,
    output_root: Path,
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    structure_library_path: Path,
    engine_config_path: Path,
    validator,
) -> dict[str, object]:
    case_output = ensure_output_dir(output_root / label)
    status = "failure"
    failure_message = None
    validation = {"status": "failure", "issues": ["run did not complete"]}
    try:
        result = run_nested_engine(
            par_path=par_path,
            tim_path=tim_path,
            cluster_manifest_path=cluster_manifest_path,
            structure_library_path=structure_library_path,
            engine_config_path=engine_config_path,
            output_dir=case_output,
        )
        validation = validator(result)
        status = validation["status"]
    except Exception as exc:  # pragma: no cover - exercised in real server runs
        failure_message = str(exc)
        result = None
    return {
        "label": label,
        "status": status,
        "failure_message": failure_message,
        "validation": validation,
        "output_dir": str(case_output),
    }


def _write_bundle_summary(run_dir: Path, cases: list[dict[str, object]], bundle_tarball: Path) -> None:
    lines = [
        "# Four-Layer Bayesian Timing Regression Bundle",
        "",
        f"- created_at_utc: {_utc_stamp()}",
        f"- bundle_tarball: {bundle_tarball.name}",
        "",
        "## Cases",
    ]
    for case in cases:
        lines.append(f"- {case['label']}: status={case['status']}")
        issues = case["validation"].get("issues", [])
        if issues:
            lines.append(f"  issues={issues}")
        if case["failure_message"]:
            lines.append(f"  failure_message={case['failure_message']}")
    write_text(run_dir / "SUMMARY.md", "\n".join(lines) + "\n")


def _pack_run_directory(run_dir: Path, bundle_tarball: Path) -> None:
    with tarfile.open(bundle_tarball, "w:gz") as archive:
        archive.add(run_dir, arcname=run_dir.name)


def run_server_regression_bundle(
    output_dir: str | Path,
    stamp: str | None = None,
) -> dict[str, object]:
    output_dir = ensure_output_dir(Path(output_dir))
    stamp = stamp or _utc_stamp()
    run_dir = ensure_output_dir(output_dir / f"four-layer-regression-{stamp}")

    cases = [
        _run_case(
            label="isolated_47tucaa",
            output_root=run_dir,
            par_path=REPO_ROOT / "examples/literature_47tucaa/isolated_nested_initial.par",
            tim_path=REPO_ROOT / "examples/literature_47tucaa/data.tim",
            cluster_manifest_path=REPO_ROOT / "examples/literature_47tucaa/isolated_nested_clusters.yaml",
            structure_library_path=REPO_ROOT / "examples/literature_47tucaa/isolated_nested_structure_library.yaml",
            engine_config_path=REPO_ROOT / "examples/literature_47tucaa/isolated_nested_engine_refinement.yaml",
            validator=_validate_isolated_run,
        ),
        _run_case(
            label="fast_binary",
            output_root=run_dir,
            par_path=REPO_ROOT / "examples/19C101ell1.par",
            tim_path=REPO_ROOT / "examples/N4.tim",
            cluster_manifest_path=REPO_ROOT / "examples/real_fast_j1602_clusters.yaml",
            structure_library_path=REPO_ROOT / "examples/real_fast_j1602_structure_library.yaml",
            engine_config_path=REPO_ROOT / "examples/real_fast_j1602_nested_engine_refinement.yaml",
            validator=_validate_fast_run,
        ),
    ]

    overall_status = "success" if all(case["status"] == "success" for case in cases) else "failure"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "cases": cases,
    }
    write_json(run_dir / "manifest.json", manifest)

    bundle_tarball = output_dir / f"timing-accelerator-four-layer-results-{stamp}.tar.gz"
    _write_bundle_summary(run_dir, cases, bundle_tarball)
    _pack_run_directory(run_dir, bundle_tarball)

    return {
        "overall_status": overall_status,
        "run_dir": str(run_dir),
        "bundle_tarball": str(bundle_tarball),
        "manifest_path": str(run_dir / "manifest.json"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the four-layer server regression bundle.")
    parser.add_argument(
        "--output-dir",
        default="/work/output",
        help="Directory where raw outputs and the final tar.gz bundle will be written.",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Optional deterministic run stamp. Defaults to current UTC timestamp.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_server_regression_bundle(output_dir=args.output_dir, stamp=args.stamp)
    print(f"Overall status: {summary['overall_status']}")
    print(f"Run directory: {summary['run_dir']}")
    print(f"Bundle tarball: {summary['bundle_tarball']}")
    return 0 if summary["overall_status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
