from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from timing_accelerator.io.save_artifacts import ensure_output_dir, write_json, write_text
from timing_accelerator.pipeline.block_wise_marginal_evidence_experiment import _resolve_embedded_paths
from timing_accelerator.pipeline.run_pipeline import run_pipeline


def _apply_local_solver_profile(
    payload: dict,
    *,
    mode: str,
    max_iterations: int,
    initial_damping: float,
    damping_multiplier: float,
    min_fractional_improvement: float,
) -> dict:
    updated = deepcopy(payload)
    updated["execution_layer"] = "experimental-strategy"
    stages = updated.get("recovery_stages")
    if not stages:
        fit_params = list(updated.get("post_certification_fit_params") or [])
        updated["recovery_stages"] = [
            {
                "stage_id": "default-coarse-search",
                "kind": "coarse-search",
                "hard_params": list(updated.get("hard_params") or []),
                "top_k": int(updated.get("top_k", 5)),
            },
            {
                "stage_id": "local-fit",
                "kind": "local-fit",
                "fit_params": fit_params,
                "full_solution_stage": True,
            },
        ]
        stages = updated["recovery_stages"]
    for stage in stages:
        if str(stage.get("kind", "")).lower() != "local-fit":
            continue
        stage["local_solver_mode"] = mode
        stage["local_solver_max_iterations"] = max_iterations
        stage["local_solver_initial_damping"] = initial_damping
        stage["local_solver_damping_multiplier"] = damping_multiplier
        stage["local_solver_min_fractional_improvement"] = min_fractional_improvement
    return updated


def _build_linearized_payload(base_payload: dict) -> dict:
    return _apply_local_solver_profile(
        base_payload,
        mode="linearized",
        max_iterations=5,
        initial_damping=0.0,
        damping_multiplier=1.0,
        min_fractional_improvement=0.0,
    )


def _build_robust_payload(base_payload: dict) -> dict:
    return _apply_local_solver_profile(
        base_payload,
        mode="robust-lm",
        max_iterations=10,
        initial_damping=1.0,
        damping_multiplier=8.0,
        min_fractional_improvement=1.0e-6,
    )


def _variant_score(result_payload: dict) -> tuple[int, float, float]:
    official_status = result_payload.get("official_status")
    calibration = result_payload.get("calibration") or {}
    certified_candidate = result_payload.get("certified_candidate") or {}
    debug_candidate = result_payload.get("debug_candidate") or {}
    metrics = result_payload.get("metrics") or {}
    success_rank = 0 if official_status == "success" else 1
    reduced = calibration.get("recovered_fit_reduced_chi2")
    if reduced is None:
        reduced = certified_candidate.get("fit_reduced_chi2")
    if reduced is None:
        reduced = debug_candidate.get("fit_reduced_chi2", float("inf"))
    rms = calibration.get("recovered_weighted_rms_us")
    if rms is None:
        rms = metrics.get("stage_local-fit_weighted_rms_us")
    if rms is None:
        rms = float("inf")
    return success_rank, reduced, rms


def _summarize(results_by_variant: dict[str, dict]) -> tuple[dict, str]:
    ranked = sorted(results_by_variant.items(), key=lambda item: _variant_score(item[1]))
    recommended_variant = ranked[0][0]
    payload = {"recommended_variant": recommended_variant, "variants": []}
    lines = [
        "# Robust Full-Solution Local Solver Experiment",
        "",
        f"Recommended variant: `{recommended_variant}`",
        "",
        "## Variant Summary",
    ]
    for name, result in ranked:
        calibration = result.get("calibration") or {}
        certified_candidate = result.get("certified_candidate") or {}
        debug_candidate = result.get("debug_candidate") or {}
        metrics = result.get("metrics") or {}
        stages = result.get("debug_stages") or []
        final_local_stage = None
        for stage in reversed(stages):
            if stage.get("kind") == "local-fit":
                final_local_stage = stage
                break
        reduced = calibration.get("recovered_fit_reduced_chi2")
        if reduced is None:
            reduced = certified_candidate.get("fit_reduced_chi2")
        if reduced is None:
            reduced = debug_candidate.get("fit_reduced_chi2")
        rms = calibration.get("recovered_weighted_rms_us")
        if rms is None:
            rms = metrics.get("stage_local-fit_weighted_rms_us")
        variant = {
            "variant": name,
            "official_status": result.get("official_status"),
            "recovered_fit_reduced_chi2": reduced,
            "recovered_weighted_rms_us": rms,
            "local_solver_mode": final_local_stage.get("local_solver_mode") if final_local_stage else None,
            "local_solver_diagnostics": (
                final_local_stage.get("local_solver_diagnostics") if final_local_stage else None
            ),
        }
        payload["variants"].append(variant)
        lines.append(
            f"- `{name}`: official_status={variant['official_status']}, "
            f"reduced_chi2={variant['recovered_fit_reduced_chi2']}, "
            f"weighted_rms_us={variant['recovered_weighted_rms_us']}, "
            f"solver={variant['local_solver_mode']}"
        )
    lines.append("")
    return payload, "\n".join(lines)


def run_robust_full_solution_local_solver_experiment(
    par_path: str | Path,
    tim_path: str | Path,
    base_config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    base_config_path = Path(base_config_path)
    output_dir = ensure_output_dir(Path(output_dir))
    configs_dir = ensure_output_dir(output_dir / "configs")
    base_payload = _resolve_embedded_paths(
        json.loads(base_config_path.read_text(encoding="utf-8")),
        base_config_path.parent,
    )

    builders = {
        "legacy_linearized": _build_linearized_payload,
        "robust_lm": _build_robust_payload,
    }
    results_by_variant: dict[str, dict] = {}
    for variant_name, builder in builders.items():
        payload = builder(base_payload)
        config_path = configs_dir / f"{variant_name}.json"
        variant_output_dir = output_dir / variant_name
        write_json(config_path, payload)
        run_pipeline(
            par_path=par_path,
            tim_path=tim_path,
            simulation_config_path=config_path,
            output_dir=variant_output_dir,
        )
        results_by_variant[variant_name] = json.loads(
            (variant_output_dir / "result.json").read_text(encoding="utf-8")
        )

    comparison_payload, comparison_markdown = _summarize(results_by_variant)
    write_json(output_dir / "comparison.json", comparison_payload)
    write_text(output_dir / "comparison.md", comparison_markdown)
    return comparison_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare legacy and robust low-level local solvers.")
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_robust_full_solution_local_solver_experiment(
        par_path=args.par,
        tim_path=args.tim,
        base_config_path=args.base_config,
        output_dir=args.output_dir,
    )
    print(f"Recommended variant: {comparison['recommended_variant']}")
    print(f"Artifacts written to: {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
