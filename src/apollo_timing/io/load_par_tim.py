from __future__ import annotations

import json
from pathlib import Path

from apollo_timing.backend.pint_backend import maybe_generate_fake_toas
from apollo_timing.models.contracts import (
    AutomaticFinalStructureConfig,
    BootstrapRequest,
    CalibrationTargetConfig,
    FakeToaGenerationConfig,
    RecoveryStageConfig,
    SimulationConfig,
)


def _resolve_optional_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def load_par_file(par_path: Path) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for raw_line in par_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pieces = line.split()
        if len(pieces) < 2:
            continue
        try:
            parameters[pieces[0].upper()] = float(pieces[1])
        except ValueError:
            continue
    return parameters


def count_toas(tim_path: Path) -> int:
    count = 0
    for raw_line in tim_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("#", "C ")):
            continue
        if line.upper().startswith(("FORMAT", "MODE")):
            continue
        if line:
            count += 1
    return count


def load_simulation_config(simulation_config_path: Path) -> SimulationConfig:
    payload = json.loads(simulation_config_path.read_text(encoding="utf-8"))
    base_dir = simulation_config_path.parent
    hard_params = [name.upper() for name in payload.get("hard_params", [])]
    soft_params = [name.upper() for name in payload.get("soft_params", [])]
    truth = {name.upper(): float(value) for name, value in payload.get("truth", {}).items()}
    grid_offsets = {
        name.upper(): [float(item) for item in offsets]
        for name, offsets in payload.get("grid_offsets", {}).items()
    }
    fake_toa_payload = payload.get("generate_fake_toas")
    fake_toa_config = None
    if fake_toa_payload:
        fake_toa_config = FakeToaGenerationConfig(
            tim_path=(base_dir / fake_toa_payload.get("tim_path", "")).resolve()
            if not Path(fake_toa_payload.get("tim_path", "")).is_absolute()
            else Path(fake_toa_payload.get("tim_path", "")),
            start_mjd=float(fake_toa_payload["start_mjd"]),
            end_mjd=float(fake_toa_payload["end_mjd"]),
            ntoas=int(fake_toa_payload["ntoas"]),
            error_us=float(fake_toa_payload.get("error_us", 1.0)),
            obs=str(fake_toa_payload.get("obs", "@")),
            freq_mhz=float(fake_toa_payload.get("freq_mhz", 1400.0)),
            add_noise=bool(fake_toa_payload.get("add_noise", False)),
        )
    reference_payload = payload.get("reference_target")
    reference_target = None
    if reference_payload:
        reference_par_path = _resolve_optional_path(reference_payload.get("par_path"), base_dir)
        if reference_par_path is None:
            raise ValueError("reference_target.par_path is required when reference_target is provided.")
        reference_target = CalibrationTargetConfig(
            par_path=reference_par_path,
            pint_clock_override_path=_resolve_optional_path(
                reference_payload.get("pint_clock_override_path"),
                base_dir,
            ),
        )
    recovery_stages_payload = payload.get("recovery_stages")
    recovery_stages = None
    if recovery_stages_payload:
        recovery_stages = []
        for index, stage_payload in enumerate(recovery_stages_payload, start=1):
            stage_grid_offsets = stage_payload.get("grid_offsets")
            recovery_stages.append(
                RecoveryStageConfig(
                    stage_id=str(stage_payload.get("stage_id", f"stage-{index}")),
                    kind=str(stage_payload.get("kind", "coarse-search")).lower(),
                    hard_params=[name.upper() for name in stage_payload.get("hard_params", [])],
                    fit_params=[name.upper() for name in stage_payload.get("fit_params", [])],
                    local_solver_mode=str(stage_payload.get("local_solver_mode", "robust-lm")).lower(),
                    local_solver_max_iterations=int(
                        stage_payload.get("local_solver_max_iterations", 8)
                    ),
                    local_solver_initial_damping=float(
                        stage_payload.get("local_solver_initial_damping", 1.0)
                    ),
                    local_solver_damping_multiplier=float(
                        stage_payload.get("local_solver_damping_multiplier", 8.0)
                    ),
                    local_solver_min_fractional_improvement=float(
                        stage_payload.get("local_solver_min_fractional_improvement", 1.0e-6)
                    ),
                    final_block_core_blocks=[
                        [name.upper() for name in block]
                        for block in stage_payload.get("final_block_core_blocks", [])
                    ]
                    or None,
                    final_block_marginalized_params=[
                        name.upper()
                        for name in stage_payload.get("final_block_marginalized_params", [])
                    ]
                    or None,
                    final_block_nuisance_groups=[
                        [name.upper() for name in block]
                        for block in stage_payload.get("final_block_nuisance_groups", [])
                    ]
                    or None,
                    final_block_expansion_blocks=[
                        [name.upper() for name in block]
                        for block in stage_payload.get("final_block_expansion_blocks", [])
                    ]
                    or None,
                    final_block_score_threshold=float(
                        stage_payload.get("final_block_score_threshold", 0.0)
                    ),
                    final_block_fallback_block=[
                        name.upper()
                        for name in stage_payload.get("final_block_fallback_block", [])
                    ]
                    or None,
                    automatic_final_structures=[
                        AutomaticFinalStructureConfig(
                            structure_id=str(
                                structure_payload.get("structure_id", f"structure-{structure_index}")
                            ),
                            core_block=[
                                name.upper() for name in structure_payload.get("core_block", [])
                            ],
                            marginalized_params=[
                                name.upper()
                                for name in structure_payload.get("marginalized_params", [])
                            ],
                            nuisance_groups=[
                                [name.upper() for name in block]
                                for block in structure_payload.get("nuisance_groups", [])
                            ]
                            or None,
                            expansion_blocks=[
                                [name.upper() for name in block]
                                for block in structure_payload.get("expansion_blocks", [])
                            ]
                            or None,
                        )
                        for structure_index, structure_payload in enumerate(
                            stage_payload.get("automatic_final_structures", []),
                            start=1,
                        )
                    ]
                    or None,
                    top_k=int(stage_payload["top_k"]) if "top_k" in stage_payload else None,
                    grid_offsets={
                        name.upper(): [float(item) for item in offsets]
                        for name, offsets in (stage_grid_offsets or {}).items()
                    }
                    or None,
                    adaptive_exact_search_rounds=int(
                        stage_payload.get("adaptive_exact_search_rounds", 0)
                    ),
                    adaptive_exact_search_branching=int(
                        stage_payload.get("adaptive_exact_search_branching", 1)
                    ),
                    adaptive_exact_search_shrink_factor=float(
                        stage_payload.get("adaptive_exact_search_shrink_factor", 0.5)
                    ),
                    adaptive_exact_search_expand_edge_factor=float(
                        stage_payload.get("adaptive_exact_search_expand_edge_factor", 2.0)
                    ),
                    use_posterior_guidance=bool(stage_payload.get("use_posterior_guidance", False)),
                    rewrite_hard_params_from_posterior=bool(
                        stage_payload.get("rewrite_hard_params_from_posterior", False)
                    ),
                    rewrite_fit_params_from_posterior=bool(
                        stage_payload.get("rewrite_fit_params_from_posterior", False)
                    ),
                    planning_confidence_threshold=float(
                        stage_payload.get("planning_confidence_threshold", 0.7)
                    ),
                    max_posterior_group_size=(
                        int(stage_payload["max_posterior_group_size"])
                        if "max_posterior_group_size" in stage_payload
                        else None
                    ),
                    basin_shaping_center_offsets={
                        name.upper(): float(value)
                        for name, value in stage_payload.get(
                            "basin_shaping_center_offsets",
                            {},
                        ).items()
                    }
                    or None,
                    basin_shaping_rank_weights={
                        name.upper(): float(value)
                        for name, value in stage_payload.get(
                            "basin_shaping_rank_weights",
                            {},
                        ).items()
                    }
                    or None,
                    bayesian_release_blocks=[
                        [name.upper() for name in block]
                        for block in stage_payload.get("bayesian_release_blocks", [])
                    ]
                    or None,
                    bayesian_release_score_threshold=float(
                        stage_payload.get("bayesian_release_score_threshold", 0.0)
                    ),
                    bayesian_release_complexity_penalty_scale=float(
                        stage_payload.get("bayesian_release_complexity_penalty_scale", 1.0)
                    ),
                    bayesian_release_fallback_order=[
                        [name.upper() for name in block]
                        for block in stage_payload.get("bayesian_release_fallback_order", [])
                    ]
                    or None,
                    basin_score_threshold=(
                        float(stage_payload["basin_score_threshold"])
                        if "basin_score_threshold" in stage_payload
                        else None
                    ),
                    full_solution_stage=bool(stage_payload.get("full_solution_stage", False)),
                    require_trusted_posterior_unlock=bool(
                        stage_payload.get("require_trusted_posterior_unlock", False)
                    ),
                    posterior_unlock_max_condition_number=(
                        float(stage_payload["posterior_unlock_max_condition_number"])
                        if "posterior_unlock_max_condition_number" in stage_payload
                        else None
                    ),
                    quality_gate_max_reduced_chi2=(
                        float(stage_payload["quality_gate_max_reduced_chi2"])
                        if "quality_gate_max_reduced_chi2" in stage_payload
                        else None
                    ),
                    quality_gate_max_weighted_rms_us=(
                        float(stage_payload["quality_gate_max_weighted_rms_us"])
                        if "quality_gate_max_weighted_rms_us" in stage_payload
                        else None
                    ),
                )
            )

    return SimulationConfig(
        regime=str(payload.get("regime", "bootstrap")).lower(),
        execution_layer=(
            str(payload["execution_layer"]).lower()
            if "execution_layer" in payload and payload["execution_layer"] is not None
            else None
        ),
        hard_params=hard_params,
        soft_params=soft_params,
        truth=truth,
        grid_offsets=grid_offsets,
        especially_sparse_benchmark=bool(payload.get("especially_sparse_benchmark", False)),
        top_k=int(payload.get("top_k", 5)),
        approximation_penalty=float(payload.get("approximation_penalty", 0.1)),
        certification_backend=str(payload.get("certification_backend", "mock")).lower(),
        truth_par_path=_resolve_optional_path(payload.get("truth_par_path"), base_dir),
        generate_fake_toas=fake_toa_config,
        certification_fit_params=[
            name.upper() for name in payload.get("certification_fit_params", [])
        ]
        or None,
        post_certification_fit_params=[
            name.upper() for name in payload.get("post_certification_fit_params", [])
        ]
        or None,
        reference_target=reference_target,
        recovery_stages=recovery_stages,
    )


def build_bootstrap_request(
    par_path: Path,
    tim_path: Path,
    simulation_config_path: Path,
) -> BootstrapRequest:
    simulation = load_simulation_config(simulation_config_path)
    if simulation.generate_fake_toas is not None:
        simulation.generate_fake_toas.tim_path = tim_path
    maybe_generate_fake_toas(simulation)
    return BootstrapRequest(
        par_path=par_path,
        tim_path=tim_path,
        simulation_config_path=simulation_config_path,
        initial_parameters=load_par_file(par_path),
        toa_count=count_toas(tim_path),
        simulation=simulation,
    )
