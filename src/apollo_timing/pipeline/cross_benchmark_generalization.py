from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from apollo_timing.io.save_artifacts import ensure_output_dir, write_json, write_text
from apollo_timing.pipeline.release_order_ablation import build_release_order_comparison
from apollo_timing.pipeline.run_pipeline import run_pipeline
from apollo_timing.pipeline.block_wise_marginal_evidence_experiment import (
    _resolve_embedded_paths,

)


VARIANTS = [
    "baseline_direct_all",
    "raj_then_decj",
    "decj_then_raj",
]


def build_47tuc_full_variant_payload(base_payload: dict, variant_name: str) -> dict:
    payload = deepcopy(base_payload)
    payload["execution_layer"] = "experimental-strategy"
    payload["especially_sparse_benchmark"] = True

    final_stage = {
        "stage_id": "local-fit-final",
        "kind": "local-fit",
        "fit_params": ["F0", "F1", "RAJ", "DECJ", "PMRA", "PMDEC"],
        "full_solution_stage": True,
        "quality_gate_max_reduced_chi2": 100.0,
        "quality_gate_max_weighted_rms_us": 500.0,
    }
    coarse_stage = {
        "stage_id": "coarse-f0-f1",
        "kind": "coarse-search",
        "hard_params": ["F0", "F1"],
        "top_k": int(payload.get("top_k", 30)),
        "grid_offsets": payload.get("grid_offsets", {}),
    }

    if variant_name == "baseline_direct_all":
        payload["recovery_stages"] = [coarse_stage, final_stage]
        return payload

    if variant_name == "raj_then_decj":
        payload["recovery_stages"] = [
            coarse_stage,
            {
                "stage_id": "local-fit-raj-precondition",
                "kind": "local-fit",
                "fit_params": ["F0", "F1", "RAJ", "PMRA"],
                "quality_gate_max_reduced_chi2": 100.0,
                "quality_gate_max_weighted_rms_us": 500.0,
            },
            {
                "stage_id": "local-fit-decj-precondition",
                "kind": "local-fit",
                "fit_params": ["DECJ", "PMDEC"],
                "quality_gate_max_reduced_chi2": 100.0,
                "quality_gate_max_weighted_rms_us": 500.0,
            },
            final_stage,
        ]
        return payload

    if variant_name == "decj_then_raj":
        payload["recovery_stages"] = [
            coarse_stage,
            {
                "stage_id": "local-fit-decj-precondition",
                "kind": "local-fit",
                "fit_params": ["F0", "F1", "DECJ", "PMDEC"],
                "quality_gate_max_reduced_chi2": 100.0,
                "quality_gate_max_weighted_rms_us": 500.0,
            },
            {
                "stage_id": "local-fit-raj-precondition",
                "kind": "local-fit",
                "fit_params": ["RAJ", "PMRA"],
                "quality_gate_max_reduced_chi2": 100.0,
                "quality_gate_max_weighted_rms_us": 500.0,
            },
            final_stage,
        ]
        return payload

    raise ValueError(f"Unsupported 47 Tuc aa variant: {variant_name}")




def run_cross_benchmark_generalization(
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
    results_by_variant: dict[str, dict] = {}

    for variant_name in VARIANTS:
        variant_payload = build_47tuc_full_variant_payload(base_payload, variant_name)
        variant_config_path = configs_dir / f"{variant_name}.json"
        variant_output_dir = output_dir / variant_name
        write_json(variant_config_path, variant_payload)
        run_pipeline(
            par_path=par_path,
            tim_path=tim_path,
            simulation_config_path=variant_config_path,
            output_dir=variant_output_dir,
        )
        results_by_variant[variant_name] = json.loads(
            (variant_output_dir / "result.json").read_text(encoding="utf-8")
        )

    comparison_payload, comparison_markdown = build_release_order_comparison(results_by_variant)
    write_json(output_dir / "comparison.json", comparison_payload)
    write_text(output_dir / "comparison.md", comparison_markdown)
    return comparison_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cross-benchmark generalization on 47 Tuc aa full-fit.")
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_cross_benchmark_generalization(
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
