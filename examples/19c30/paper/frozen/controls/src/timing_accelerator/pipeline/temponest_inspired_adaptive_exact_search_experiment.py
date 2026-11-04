from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from timing_accelerator.io.save_artifacts import ensure_output_dir, write_json, write_text
from timing_accelerator.pipeline.block_wise_marginal_evidence_experiment import _resolve_embedded_paths
from timing_accelerator.pipeline.full_fit_regime_reset_experiment import (
    _build_full_fit_payload,
    _variant_score,
)
from timing_accelerator.pipeline.run_pipeline import run_pipeline


def _ensure_recovery_stages(payload: dict) -> dict:
    updated = deepcopy(payload)
    if updated.get("recovery_stages"):
        return updated
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
            "fit_params": list(updated.get("post_certification_fit_params") or []),
            "full_solution_stage": True,
        },
    ]
    return updated


def _build_adaptive_exact_payload(base_payload: dict) -> dict:
    payload = _ensure_recovery_stages(_build_full_fit_payload(base_payload))
    payload["execution_layer"] = "full-fit-core"
    for stage in payload["recovery_stages"]:
        if stage.get("kind") != "coarse-search":
            continue
        stage_id = str(stage.get("stage_id", ""))
        if stage_id == "coarse-f0-f1":
            stage["adaptive_exact_search_rounds"] = 2
            stage["adaptive_exact_search_branching"] = 2
        elif stage_id == "coarse-pb-a1-tasc-eps2":
            stage["adaptive_exact_search_rounds"] = 1
            stage["adaptive_exact_search_branching"] = 2
        else:
            stage["adaptive_exact_search_rounds"] = 1
            stage["adaptive_exact_search_branching"] = 1
        stage["adaptive_exact_search_shrink_factor"] = 0.5
        stage["adaptive_exact_search_expand_edge_factor"] = 2.0
    return payload


def _summarize(results_by_variant: dict[str, dict]) -> tuple[dict, str]:
    ranked = sorted(results_by_variant.items(), key=lambda item: _variant_score(item[1]))
    recommended_variant = ranked[0][0]
    payload = {"recommended_variant": recommended_variant, "variants": []}
    lines = [
        "# TempoNest-Inspired Adaptive Exact Search Experiment",
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
        exact_round_count = sum(len(stage.get("exact_search_rounds") or []) for stage in stages)
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
            "exact_search_round_count": exact_round_count,
        }
        payload["variants"].append(variant)
        lines.append(
            f"- `{name}`: official_status={variant['official_status']}, "
            f"reduced_chi2={variant['recovered_fit_reduced_chi2']}, "
            f"weighted_rms_us={variant['recovered_weighted_rms_us']}, "
            f"exact_search_round_count={variant['exact_search_round_count']}"
        )
    lines.append("")
    return payload, "\n".join(lines)


def run_temponest_inspired_adaptive_exact_search_experiment(
    benchmark_name: str,
    par_path: str | Path,
    tim_path: str | Path,
    base_config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    if benchmark_name not in {"fast", "47tuc"}:
        raise ValueError(f"Unsupported benchmark name: {benchmark_name}")
    base_config_path = Path(base_config_path)
    output_dir = ensure_output_dir(Path(output_dir))
    configs_dir = ensure_output_dir(output_dir / "configs")
    base_payload = _resolve_embedded_paths(
        json.loads(base_config_path.read_text(encoding="utf-8")),
        base_config_path.parent,
    )

    builders = {
        "full_fit_reset_baseline": _build_full_fit_payload,
        "adaptive_exact_search": _build_adaptive_exact_payload,
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
    parser = argparse.ArgumentParser(description="Compare full-fit reset against TempoNest-inspired adaptive exact search.")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    comparison = run_temponest_inspired_adaptive_exact_search_experiment(
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
