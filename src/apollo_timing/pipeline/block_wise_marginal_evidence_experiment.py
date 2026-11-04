from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from apollo_timing.io.save_artifacts import ensure_output_dir, write_json, write_text
from apollo_timing.pipeline.run_pipeline import run_pipeline


def _resolve_embedded_paths(payload: dict, base_dir: Path) -> dict:
    resolved = deepcopy(payload)
    for key in ("truth_par_path",):
        path_value = resolved.get(key)
        if path_value and not Path(path_value).is_absolute():
            resolved[key] = str((base_dir / path_value).resolve())
    fake_toas = resolved.get("generate_fake_toas")
    if fake_toas:
        tim_path = fake_toas.get("tim_path")
        if tim_path and not Path(tim_path).is_absolute():
            fake_toas["tim_path"] = str((base_dir / tim_path).resolve())
    reference_target = resolved.get("reference_target")
    if reference_target:
        for key in ("par_path", "pint_clock_override_path"):
            path_value = reference_target.get(key)
            if path_value and not Path(path_value).is_absolute():
                reference_target[key] = str((base_dir / path_value).resolve())
    return resolved


def _variant_score(result_payload: dict) -> tuple[int, float, float]:
    official_status = result_payload.get("official_status")
    calibration = result_payload.get("calibration") or {}
    debug_candidate = result_payload.get("debug_candidate") or {}
    success_rank = 0 if official_status == "success" else 1
    reduced = calibration.get("recovered_fit_reduced_chi2")
    if reduced is None:
        reduced = debug_candidate.get("fit_reduced_chi2", float("inf"))
    rms = calibration.get("recovered_weighted_rms_us")
    if rms is None:
        rms = float("inf")
    return success_rank, reduced, rms


def _summarize(results_by_variant: dict[str, dict]) -> tuple[dict, str]:
    ranked = sorted(results_by_variant.items(), key=lambda item: _variant_score(item[1]))
    recommended_variant = ranked[0][0]
    payload = {"recommended_variant": recommended_variant, "variants": []}
    lines = [
        "# Block-wise marginal-evidence experiment",
        "",
        f"Recommended variant: `{recommended_variant}`",
        "",
        "## Variant summary",
    ]
    for name, result in ranked:
        calibration = result.get("calibration") or {}
        variant = {
            "variant": name,
            "official_status": result.get("official_status"),
            "posterior_unlock_state": result.get("posterior_unlock_state"),
            "recovered_fit_reduced_chi2": calibration.get("recovered_fit_reduced_chi2"),
            "recovered_weighted_rms_us": calibration.get("recovered_weighted_rms_us"),
        }
        payload["variants"].append(variant)
        lines.append(
            f"- `{name}`: official_status={variant['official_status']}, "
            f"posterior_unlock={variant['posterior_unlock_state']}, "
            f"reduced_chi2={variant['recovered_fit_reduced_chi2']}, "
            f"weighted_rms_us={variant['recovered_weighted_rms_us']}"
        )
    lines.append("")
    return payload, "\n".join(lines)


def _build_fast_fixed_payload(base_payload: dict) -> dict:
    payload = deepcopy(base_payload)
    payload["execution_layer"] = "experimental-strategy"
    return payload


def _build_fast_blockwise_payload(base_payload: dict) -> dict:
    payload = deepcopy(base_payload)
    payload["execution_layer"] = "experimental-strategy"
    stages = list(payload["recovery_stages"])
    final_stage = stages.pop()
    if final_stage["stage_id"] != "local-fit-final":
        raise ValueError("Expected FAST config to end with local-fit-final.")
    stages.extend(
        [
            {
                "stage_id": "block-model-compare-1",
                "kind": "local-fit",
                "fit_params": ["F0", "F1", "PB", "A1", "TASC", "EPS2"],
                "bayesian_release_blocks": [
                    ["RAJ", "DECJ"],
                    ["RAJ", "DECJ", "EPS1"],
                    ["RAJ"],
                    ["DECJ"],
                    ["EPS1"],
                ],
                "bayesian_release_score_threshold": 0.0,
                "bayesian_release_complexity_penalty_scale": 3.0,
                "bayesian_release_fallback_order": [["RAJ", "DECJ", "EPS1"]],
                "basin_score_threshold": -2500000.0,
                "quality_gate_max_reduced_chi2": 15000.0,
                "quality_gate_max_weighted_rms_us": 700.0
            },
            final_stage,
        ]
    )
    payload["recovery_stages"] = stages
    return payload


def _build_47tuc_fixed_payload(base_payload: dict) -> dict:
    payload = deepcopy(base_payload)
    payload["execution_layer"] = "experimental-strategy"
    payload["especially_sparse_benchmark"] = True
    payload["reference_target"] = {"par_path": payload["truth_par_path"]}
    payload["recovery_stages"] = [
        {
            "stage_id": "coarse-f0-f1",
            "kind": "coarse-search",
            "hard_params": ["F0", "F1"],
            "top_k": int(payload.get("top_k", 30)),
            "grid_offsets": payload["grid_offsets"],
        },
        {
            "stage_id": "local-fit-final",
            "kind": "local-fit",
            "fit_params": ["F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC"],
            "full_solution_stage": True,
            "quality_gate_max_reduced_chi2": 100.0,
            "quality_gate_max_weighted_rms_us": 500.0,
        },
    ]
    return payload


def _build_47tuc_blockwise_payload(base_payload: dict) -> dict:
    payload = _build_47tuc_fixed_payload(base_payload)
    payload["execution_layer"] = "experimental-strategy"
    final_stage = payload["recovery_stages"].pop()
    payload["recovery_stages"].extend(
        [
            {
                "stage_id": "block-model-compare-1",
                "kind": "local-fit",
                "fit_params": ["F0", "F1"],
                "bayesian_release_blocks": [
                    ["RAJ", "PMRA"],
                    ["DECJ", "PMDEC"],
                    ["RAJ", "PMRA", "DECJ", "PMDEC"],
                ],
                "bayesian_release_score_threshold": 0.0,
                "bayesian_release_complexity_penalty_scale": 3.0,
                "bayesian_release_fallback_order": [["RAJ", "PMRA", "DECJ", "PMDEC"]],
                "quality_gate_max_reduced_chi2": 100.0,
                "quality_gate_max_weighted_rms_us": 500.0
            },
            final_stage,
        ]
    )
    return payload


def run_block_wise_marginal_evidence_experiment(
    benchmark_name: str,
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

    if benchmark_name == "fast":
        builders = {
            "fixed_schedule": _build_fast_fixed_payload,
            "blockwise_optimizer": _build_fast_blockwise_payload,
        }
    elif benchmark_name == "47tuc":
        builders = {
            "fixed_schedule": _build_47tuc_fixed_payload,
            "blockwise_optimizer": _build_47tuc_blockwise_payload,
        }
    else:
        raise ValueError(f"Unsupported benchmark name: {benchmark_name}")

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
    parser = argparse.ArgumentParser(description="Run block-wise marginal evidence experiment on a benchmark.")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_block_wise_marginal_evidence_experiment(
        benchmark_name=args.benchmark_name,
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
