from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from apollo_timing.io.save_artifacts import ensure_output_dir, write_json, write_text
from apollo_timing.pipeline.run_pipeline import run_pipeline


VARIANT_ORDER = [
    "baseline_binary_then_all",
    "sky_then_eps1",
    "eps1_then_sky",
]

SPLIT_SKY_VARIANT_ORDER = [
    "eps1_then_raj_then_decj",
    "eps1_then_decj_then_raj",
]


def build_release_order_variant_payload(base_payload: dict, variant_name: str) -> dict:
    payload = deepcopy(base_payload)
    payload["execution_layer"] = "experimental-strategy"
    stages = list(payload.get("recovery_stages", []))

    final_stage = stages.pop()
    if final_stage.get("stage_id") != "local-fit-final":
        raise ValueError("Expected the last stage in the FAST benchmark config to be 'local-fit-final'.")

    if variant_name == "baseline_binary_then_all":
        stages.append(final_stage)
        payload["recovery_stages"] = stages
        return payload

    if variant_name == "sky_then_eps1":
        stages.extend(
            [
                {
                    "stage_id": "local-fit-sky-precondition",
                    "kind": "local-fit",
                    "fit_params": ["RAJ", "DECJ"],
                    "basin_score_threshold": -2500000.0,
                    "quality_gate_max_reduced_chi2": 12000.0,
                    "quality_gate_max_weighted_rms_us": 600.0,
                },
                {
                    "stage_id": "local-fit-eps1-precondition",
                    "kind": "local-fit",
                    "fit_params": ["EPS1"],
                    "basin_score_threshold": -2500000.0,
                    "quality_gate_max_reduced_chi2": 8000.0,
                    "quality_gate_max_weighted_rms_us": 500.0,
                },
            ]
        )
        stages.append(final_stage)
        payload["recovery_stages"] = stages
        return payload

    if variant_name == "eps1_then_sky":
        stages.extend(
            [
                {
                    "stage_id": "local-fit-eps1-precondition",
                    "kind": "local-fit",
                    "fit_params": ["EPS1"],
                    "basin_score_threshold": -2500000.0,
                    "quality_gate_max_reduced_chi2": 12000.0,
                    "quality_gate_max_weighted_rms_us": 600.0,
                },
                {
                    "stage_id": "local-fit-sky-precondition",
                    "kind": "local-fit",
                    "fit_params": ["RAJ", "DECJ"],
                    "basin_score_threshold": -2500000.0,
                    "quality_gate_max_reduced_chi2": 8000.0,
                    "quality_gate_max_weighted_rms_us": 500.0,
                },
            ]
        )
        stages.append(final_stage)
        payload["recovery_stages"] = stages
        return payload

    raise ValueError(f"Unsupported release-order variant: {variant_name}")


def build_split_sky_variant_payload(base_payload: dict, variant_name: str) -> dict:
    payload = build_release_order_variant_payload(base_payload, "eps1_then_sky")
    stages = list(payload.get("recovery_stages", []))

    sky_stage_index = next(
        index
        for index, stage in enumerate(stages)
        if stage.get("stage_id") == "local-fit-sky-precondition"
    )
    stages.pop(sky_stage_index)

    split_stages = {
        "eps1_then_raj_then_decj": [
            {
                "stage_id": "local-fit-raj-precondition",
                "kind": "local-fit",
                "fit_params": ["RAJ"],
                "basin_score_threshold": -2500000.0,
                "quality_gate_max_reduced_chi2": 7500.0,
                "quality_gate_max_weighted_rms_us": 520.0,
            },
            {
                "stage_id": "local-fit-decj-precondition",
                "kind": "local-fit",
                "fit_params": ["DECJ"],
                "basin_score_threshold": -2500000.0,
                "quality_gate_max_reduced_chi2": 7000.0,
                "quality_gate_max_weighted_rms_us": 500.0,
            },
        ],
        "eps1_then_decj_then_raj": [
            {
                "stage_id": "local-fit-decj-precondition",
                "kind": "local-fit",
                "fit_params": ["DECJ"],
                "basin_score_threshold": -2500000.0,
                "quality_gate_max_reduced_chi2": 7500.0,
                "quality_gate_max_weighted_rms_us": 520.0,
            },
            {
                "stage_id": "local-fit-raj-precondition",
                "kind": "local-fit",
                "fit_params": ["RAJ"],
                "basin_score_threshold": -2500000.0,
                "quality_gate_max_reduced_chi2": 7000.0,
                "quality_gate_max_weighted_rms_us": 500.0,
            },
        ],
    }
    if variant_name not in split_stages:
        raise ValueError(f"Unsupported split-sky variant: {variant_name}")

    stages[sky_stage_index:sky_stage_index] = split_stages[variant_name]
    payload["recovery_stages"] = stages
    return payload


def _resolve_embedded_paths(payload: dict, base_dir: Path) -> dict:
    resolved = deepcopy(payload)
    reference_target = resolved.get("reference_target")
    if reference_target:
        for key in ("par_path", "pint_clock_override_path"):
            path_value = reference_target.get(key)
            if path_value and not Path(path_value).is_absolute():
                reference_target[key] = str((base_dir / path_value).resolve())
    return resolved


def _variant_score(result_payload: dict) -> tuple[int, float, float]:
    official_status = result_payload.get("official_status")
    debug_candidate = result_payload.get("debug_candidate") or {}
    calibration = result_payload.get("calibration") or {}
    success_rank = 0 if official_status == "success" else 1
    debug_reduced_chi2 = calibration.get("recovered_fit_reduced_chi2")
    if debug_reduced_chi2 is None:
        debug_reduced_chi2 = debug_candidate.get("fit_reduced_chi2", float("inf"))
    weighted_rms = calibration.get("recovered_weighted_rms_us")
    if weighted_rms is None:
        weighted_rms = float("inf")
    return (success_rank, debug_reduced_chi2, weighted_rms)


def build_release_order_comparison(results_by_variant: dict[str, dict]) -> tuple[dict, str]:
    ranked = sorted(
        results_by_variant.items(),
        key=lambda item: _variant_score(item[1]),
    )
    recommended_variant = ranked[0][0]

    summary_payload = {
        "recommended_variant": recommended_variant,
        "variants": [],
    }
    lines = [
        "# Release-order ablation report",
        "",
        f"Recommended variant: `{recommended_variant}`",
        "",
        "## Variant summary",
    ]

    for variant_name, result_payload in ranked:
        calibration = result_payload.get("calibration") or {}
        variant_summary = {
            "variant": variant_name,
            "official_status": result_payload.get("official_status"),
            "posterior_unlock_state": result_payload.get("posterior_unlock_state"),
            "quality_ratio_to_reference": calibration.get("quality_ratio_to_reference"),
            "recovered_fit_reduced_chi2": calibration.get("recovered_fit_reduced_chi2"),
            "recovered_weighted_rms_us": calibration.get("recovered_weighted_rms_us"),
            "likely_error_points": [
                item.get("code")
                for item in calibration.get("likely_error_points", [])
            ],
        }
        summary_payload["variants"].append(variant_summary)

        lines.append(
            f"- `{variant_name}`: official_status={variant_summary['official_status']}, "
            f"posterior_unlock={variant_summary['posterior_unlock_state']}, "
            f"reduced_chi2={variant_summary['recovered_fit_reduced_chi2']}, "
            f"weighted_rms_us={variant_summary['recovered_weighted_rms_us']}, "
            f"quality_ratio={variant_summary['quality_ratio_to_reference']}"
        )
        if variant_summary["likely_error_points"]:
            lines.append(
                f"- `{variant_name}` likely_error_points={variant_summary['likely_error_points']}"
            )

    lines.append("")
    return summary_payload, "\n".join(lines)


def run_release_order_ablation(
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
    for variant_name in VARIANT_ORDER:
        variant_payload = build_release_order_variant_payload(base_payload, variant_name)
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


def run_split_sky_pair_ablation(
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
    for variant_name in SPLIT_SKY_VARIANT_ORDER:
        variant_payload = build_split_sky_variant_payload(base_payload, variant_name)
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
    parser = argparse.ArgumentParser(description="Run release-order ablation on the FAST benchmark.")
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_release_order_ablation(
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
