from __future__ import annotations

from pathlib import Path

from timing_accelerator.models.contracts import SimulationConfig

BOOTSTRAP_SUPPORT_LAYER = "bootstrap-support"
REFERENCE_BASELINE_LAYER = "reference-baseline"
FULL_FIT_CORE_LAYER = "full-fit-core"
EXPERIMENTAL_STRATEGY_LAYER = "experimental-strategy"


RUNNER_LAYER_REGISTRY: dict[str, str] = {
    "architecture_audit": BOOTSTRAP_SUPPORT_LAYER,
    "run_pipeline": BOOTSTRAP_SUPPORT_LAYER,
    "cli": BOOTSTRAP_SUPPORT_LAYER,
    "release_order_ablation": EXPERIMENTAL_STRATEGY_LAYER,
    "split_sky_pair_ablation": EXPERIMENTAL_STRATEGY_LAYER,
    "cross_benchmark_generalization": EXPERIMENTAL_STRATEGY_LAYER,
    "bayesian_release_policy_experiment": EXPERIMENTAL_STRATEGY_LAYER,
    "block_wise_marginal_evidence_experiment": EXPERIMENTAL_STRATEGY_LAYER,
    "robust_full_solution_local_solver_experiment": EXPERIMENTAL_STRATEGY_LAYER,
    "final_block_marginalized_bayesian_experiment": EXPERIMENTAL_STRATEGY_LAYER,
    "nuisance_elimination_staged_final_expansion_experiment": EXPERIMENTAL_STRATEGY_LAYER,
    "automated_final_structure_inference_experiment": EXPERIMENTAL_STRATEGY_LAYER,
    "full_fit_regime_reset_experiment": FULL_FIT_CORE_LAYER,
    "staged_recovery": FULL_FIT_CORE_LAYER,
    "temponest_inspired_adaptive_exact_search_experiment": FULL_FIT_CORE_LAYER,
}


def resolve_execution_layer(simulation: SimulationConfig) -> str:
    if simulation.execution_layer:
        return simulation.execution_layer
    if simulation.regime == "full-fit":
        return FULL_FIT_CORE_LAYER
    return BOOTSTRAP_SUPPORT_LAYER


def classify_runner_layer(module_stem: str) -> str:
    try:
        return RUNNER_LAYER_REGISTRY[module_stem]
    except KeyError as exc:
        raise ValueError(f"Runner layer is not registered for module: {module_stem}") from exc


def architecture_inventory(pipeline_dir: Path) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    audited_paths = sorted(pipeline_dir.glob("*.py"))
    cli_path = pipeline_dir.parent / "cli.py"
    if cli_path.exists():
        audited_paths.append(cli_path)
    for path in audited_paths:
        if path.name.startswith("_") or path.stem == "__init__":
            continue
        inventory.append(
            {
                "module": path.stem,
                "layer": classify_runner_layer(path.stem),
                "path": str(path),
            }
        )
    return inventory
