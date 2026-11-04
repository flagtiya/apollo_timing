from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from apollo_timing.io.save_artifacts import ensure_output_dir, write_json, write_text
from apollo_timing.pipeline.block_wise_marginal_evidence_experiment import _resolve_embedded_paths
from apollo_timing.pipeline.final_block_marginalized_bayesian_experiment import (
    _build_fast_marginalized_payload,
    _build_47tuc_marginalized_payload,
    _variant_score,
)
from apollo_timing.pipeline.run_pipeline import run_pipeline


def _build_fast_automated_payload(base_payload: dict) -> dict:
    payload = deepcopy(_build_fast_marginalized_payload(base_payload))
    payload["execution_layer"] = "experimental-strategy"
    for stage in payload["recovery_stages"]:
        if stage.get("stage_id") != "local-fit-final":
            continue
        stage.pop("final_block_core_blocks", None)
        stage.pop("final_block_marginalized_params", None)
        stage.pop("final_block_fallback_block", None)
        stage["automatic_final_structures"] = [
            {
                "structure_id": "eps1-core",
                "core_block": ["RAJ", "DECJ", "EPS1"],
                "marginalized_params": ["F0", "F1", "PB", "A1", "TASC", "EPS2"],
            },
            {
                "structure_id": "eps1-core-staged",
                "core_block": ["RAJ", "DECJ", "EPS1"],
                "marginalized_params": ["F0", "F1", "PB", "A1", "TASC", "EPS2"],
                "nuisance_groups": [],
            },
            {
                "structure_id": "eps1-late-orbital-first",
                "core_block": ["RAJ", "DECJ"],
                "marginalized_params": ["F0", "F1"],
                "expansion_blocks": [["PB", "A1", "TASC", "EPS2"], ["EPS1"]],
            },
            {
                "structure_id": "eps1-early-orbital-late",
                "core_block": ["RAJ", "DECJ", "EPS1"],
                "marginalized_params": ["F0", "F1"],
                "expansion_blocks": [["PB", "A1", "TASC", "EPS2"]],
            },
        ]
    return payload


def _build_47tuc_automated_payload(base_payload: dict) -> dict:
    payload = deepcopy(_build_47tuc_marginalized_payload(base_payload))
    payload["execution_layer"] = "experimental-strategy"
    for stage in payload["recovery_stages"]:
        if stage.get("kind") != "local-fit":
            continue
        stage.pop("final_block_core_blocks", None)
        stage.pop("final_block_marginalized_params", None)
        stage.pop("final_block_fallback_block", None)
        stage["automatic_final_structures"] = [
            {
                "structure_id": "sky-core-only",
                "core_block": ["RAJ", "DECJ"],
                "marginalized_params": ["F0", "F1", "PMRA", "PMDEC"],
            },
            {
                "structure_id": "sky-then-proper-motion",
                "core_block": ["RAJ", "DECJ"],
                "marginalized_params": ["F0", "F1"],
                "expansion_blocks": [["PMRA", "PMDEC"]],
            },
        ]
    return payload


def _summarize(results_by_variant: dict[str, dict]) -> tuple[dict, str]:
    ranked = sorted(results_by_variant.items(), key=lambda item: _variant_score(item[1]))
    recommended_variant = ranked[0][0]
    payload = {"recommended_variant": recommended_variant, "variants": []}
    lines = [
        "# Automatic final-structure inference experiment",
        "",
        f"Recommended variant: `{recommended_variant}`",
        "",
        "## Variant summary",
    ]
    for name, result in ranked:
        calibration = result.get("calibration") or {}
        certified_candidate = result.get("certified_candidate") or {}
        debug_candidate = result.get("debug_candidate") or {}
        metrics = result.get("metrics") or {}
        stages = result.get("debug_stages") or []
        final_stage = None
        for stage in reversed(stages):
            if stage.get("kind") == "local-fit":
                final_stage = stage
                break
        reduced = calibration.get("recovered_fit_reduced_chi2")
        if reduced is None:
            reduced = certified_candidate.get("fit_reduced_chi2")
        if reduced is None:
            reduced = debug_candidate.get("fit_reduced_chi2")
        rms = calibration.get("recovered_weighted_rms_us")
        if rms is None:
            rms = metrics.get("stage_local-fit-final_weighted_rms_us")
        if rms is None:
            rms = metrics.get("stage_local-fit_weighted_rms_us")
        variant = {
            "variant": name,
            "official_status": result.get("official_status"),
            "recovered_fit_reduced_chi2": reduced,
            "recovered_weighted_rms_us": rms,
            "automated_final_structure_selected_id": (
                final_stage.get("automated_final_structure_selected_id") if final_stage else None
            ),
            "final_block_selected_core_block": (
                final_stage.get("final_block_selected_core_block") if final_stage else None
            ),
        }
        payload["variants"].append(variant)
        lines.append(
            f"- `{name}`: official_status={variant['official_status']}, "
            f"reduced_chi2={variant['recovered_fit_reduced_chi2']}, "
            f"weighted_rms_us={variant['recovered_weighted_rms_us']}, "
            f"selected_structure={variant['automated_final_structure_selected_id']}, "
            f"selected_core_block={variant['final_block_selected_core_block']}"
        )
    lines.append("")
    return payload, "\n".join(lines)


def run_automated_final_structure_inference_experiment(
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
            "manual_marginalized_baseline": _build_fast_marginalized_payload,
            "automated_structure_inference": _build_fast_automated_payload,
        }
    elif benchmark_name == "47tuc":
        builders = {
            "manual_marginalized_baseline": _build_47tuc_marginalized_payload,
            "automated_structure_inference": _build_47tuc_automated_payload,
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
    parser = argparse.ArgumentParser(description="Compare manual final-block strategy against automated structure inference.")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_automated_final_structure_inference_experiment(
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
