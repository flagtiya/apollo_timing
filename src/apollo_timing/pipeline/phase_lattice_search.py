"""Ordinary phase-lattice search, retained-mode selection, and annealing.

This workflow writes search checkpoints and calls inference for numerical
evaluation. The solver uses the retained portfolio for PINT certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from apollo_timing.inference.bayesian_binary_lattice import (
    BinaryAnnealedMode,
    BinaryAnnealingStep,
    BinaryConditionalMode,
    BinaryLatticeSearchContext,
    anneal_phase_lattice_mode,
    search_phase_lattice_cell,
)
from apollo_timing.io.artifacts import write_json_atomic
from apollo_timing.io.checkpoints import load_checkpoint_items
from apollo_timing.models.serialization import (
    lattice_log_mass_increment_from_json as _log_mass_increment_from_payload,
)

if TYPE_CHECKING:
    from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
        BayesianBinaryLatticeSolverConfig,
    )


@dataclass(frozen=True)
class PhaseLatticeSearchResult:
    conditional_modes: tuple[BinaryConditionalMode, ...]
    selected_conditional_modes: tuple[BinaryConditionalMode, ...]
    annealed_modes: tuple[BinaryAnnealedMode, ...]
    cell_summaries: tuple[dict[str, Any], ...]
    annealing_summaries: tuple[dict[str, Any], ...]


def _conditional_mode_from_payload(payload: Mapping[str, Any]) -> BinaryConditionalMode:
    return BinaryConditionalMode(
        model_name=str(payload["model_name"]),
        cell_id=str(payload["cell_id"]),
        grid_index=tuple(int(value) for value in payload["grid_index"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        local_integer_offsets=tuple(
            tuple(int(value) for value in component)
            for component in payload["local_integer_offsets"]
        ),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        strict_score=float(payload["strict_score"]),
        log_likelihood=float(payload["log_likelihood"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        posterior_weight=float(payload["posterior_weight"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=_log_mass_increment_from_payload(
            payload["lattice_log_mass_increment"],
        ),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        posterior_in_support=bool(payload["posterior_in_support"]),
        prior_component_id=str(payload.get("prior_component_id", "local_broad")),
        prior_log_weight=float(payload.get("prior_log_weight", 0.0)),
    )



def _annealing_step_from_payload(payload: Mapping[str, Any]) -> BinaryAnnealingStep:
    return BinaryAnnealingStep(
        phase_sigma_cycles=float(payload["phase_sigma_cycles"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        lattice_strict_score=float(payload["lattice_strict_score"]),
        lattice_log_likelihood=float(payload["lattice_log_likelihood"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=_log_mass_increment_from_payload(
            payload["lattice_log_mass_increment"],
        ),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        function_evaluations=int(payload["function_evaluations"]),
        converged=bool(payload["converged"]),
        proposed_path_count=int(payload.get("proposed_path_count", 1)),
        evaluated_path_count=int(payload.get("evaluated_path_count", 1)),
        paths_selected_for_closure=int(payload.get("paths_selected_for_closure", payload.get("retained_path_count", 1))),
        path_proposal_scope=str(payload.get("path_proposal_scope", "fixed_path")),
    )



def _annealed_mode_from_payload(payload: Mapping[str, Any]) -> BinaryAnnealedMode:
    return BinaryAnnealedMode(
        model_name=str(payload["model_name"]),
        cell_id=str(payload["cell_id"]),
        source_grid_index=tuple(int(value) for value in payload["source_grid_index"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        local_integer_offsets=tuple(
            tuple(int(value) for value in component)
            for component in payload["local_integer_offsets"]
        ),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        converged=bool(payload["converged"]),
        trace=tuple(_annealing_step_from_payload(step) for step in payload["trace"]),
        prior_component_id=str(payload.get("prior_component_id", "local_broad")),
        prior_log_weight=float(payload.get("prior_log_weight", 0.0)),
    )



def _select_path_diverse_modes(ranked: Sequence[Any], limit: int) -> tuple[Any, ...]:
    """Retain distinct integer paths first, then distinct continuous-parameter basins."""
    selected: list[Any] = []
    seen_paths: set[tuple[Any, ...]] = set()
    seen_modes: set[tuple[Any, ...]] = set()
    for require_new_path in (True, False):
        for mode in ranked:
            path = (mode.integer_offsets, mode.local_integer_offsets)
            basin = (mode.cell_id, path, tuple(np.round(mode.normalized_values, 8)))
            if basin in seen_modes or (require_new_path and path in seen_paths):
                continue
            selected.append(mode)
            seen_paths.add(path)
            seen_modes.add(basin)
            if len(selected) >= limit:
                return tuple(selected)
    return tuple(selected)



def _select_conditional_modes(
    modes: Sequence[BinaryConditionalMode],
    config: BayesianBinaryLatticeSolverConfig,
) -> tuple[BinaryConditionalMode, ...]:
    ranked = sorted(
        (
            mode
            for mode in modes
            if mode.posterior_in_support
            and np.isfinite(mode.log_likelihood)
            and np.isfinite(mode.total_chi2)
        ),
        key=lambda mode: (-mode.log_likelihood, mode.total_chi2, mode.cell_id),
    )
    if not ranked:
        return tuple()
    log_likelihood_floor = (
        ranked[0].log_likelihood - config.conditional_score_window
    )
    return _select_path_diverse_modes(
        [mode for mode in ranked if mode.log_likelihood >= log_likelihood_floor],
        config.maximum_modes_to_anneal,
    )



def _select_annealed_modes(
    modes: Sequence[BinaryAnnealedMode],
    config: BayesianBinaryLatticeSolverConfig,
) -> tuple[BinaryAnnealedMode, ...]:
    ranked = sorted(
        (mode for mode in modes if np.isfinite(mode.total_chi2)),
        key=lambda mode: (mode.total_chi2, mode.cell_id, mode.integer_offsets),
    )
    if not ranked:
        return tuple()
    ceiling = ranked[0].total_chi2 + config.annealed_chi2_window
    return _select_path_diverse_modes(
        [mode for mode in ranked if mode.total_chi2 <= ceiling],
        config.maximum_modes_to_certify,
    )



def run_phase_lattice_search(
    *,
    context: BinaryLatticeSearchContext,
    output: Path,
    config: BayesianBinaryLatticeSolverConfig,
    run_fingerprint: str,
    model_name: str,
    schema_version: int,
    resume: bool = False,
    backend: Any = None,
) -> PhaseLatticeSearchResult:
    """Search every cell, retain distinct modes, then anneal their integer paths."""
    cell_dir = output / "checkpoints" / "cells"
    all_conditional: list[BinaryConditionalMode] = []
    cell_summaries: list[dict[str, Any]] = []
    for cell in context.cells:
        started = time.perf_counter()
        cpu_started = time.process_time()
        checkpoint = cell_dir / f"{cell.cell_id}.json"
        if resume and checkpoint.is_file():
            modes = load_checkpoint_items(
                checkpoint,
                expected_schema_version=schema_version,
                run_fingerprint=run_fingerprint,
                item_name="modes",
                loader=_conditional_mode_from_payload,
            )
            resumed_cell = True
        else:
            search = backend.search_cell if backend is not None else search_phase_lattice_cell
            modes = search(
                context=context,
                cell=cell,
                config=config.cell_search,
            )
            write_json_atomic(
                checkpoint,
                {
                    "schema_version": schema_version,
                    "run_fingerprint": run_fingerprint,
                    "model_name": model_name,
                    "cell_id": cell.cell_id,
                    "cell_lower": cell.lower,
                    "cell_upper": cell.upper,
                    "modes": modes,
                },
            )
            resumed_cell = False
        all_conditional.extend(modes)
        cell_summaries.append(
            {
                "cell_id": cell.cell_id,
                "mode_count": len(modes),
                "resumed": resumed_cell,
                "current_call_wall_seconds": time.perf_counter() - started,
                "current_process_cpu_seconds": time.process_time() - cpu_started,
                "best_strict_score": min((mode.strict_score for mode in modes), default=None),
                "lattice_mass_converged": all(mode.lattice_mass_converged for mode in modes),
            }
        )

    selected_conditional = _select_conditional_modes(all_conditional, config)
    if not selected_conditional:
        raise RuntimeError("No in-support posterior mode survived phase-lattice search.")
    all_annealed, anneal_summaries = anneal_retained_phase_modes(
        context=context, output=output, config=config,
        run_fingerprint=run_fingerprint, schema_version=schema_version,
        selected_conditional=selected_conditional, resume=resume, backend=backend,
    )
    return PhaseLatticeSearchResult(
        conditional_modes=tuple(all_conditional),
        selected_conditional_modes=tuple(selected_conditional),
        annealed_modes=tuple(all_annealed),
        cell_summaries=tuple(cell_summaries),
        annealing_summaries=tuple(anneal_summaries),
    )


def anneal_retained_phase_modes(
    *,
    context: BinaryLatticeSearchContext,
    output: Path,
    config: BayesianBinaryLatticeSolverConfig,
    run_fingerprint: str,
    schema_version: int,
    selected_conditional: Sequence[BinaryConditionalMode],
    resume: bool = False,
    backend: Any = None,
) -> tuple[tuple[BinaryAnnealedMode, ...], tuple[dict[str, Any], ...]]:
    """Anneal a retained portfolio without repeating discovery or cell search."""
    anneal_dir = output / "checkpoints" / "annealing"
    all_annealed: list[BinaryAnnealedMode] = []
    anneal_summaries: list[dict[str, Any]] = []
    for rank, mode in enumerate(selected_conditional):
        started = time.perf_counter()
        cpu_started = time.process_time()
        mode_id = f"{rank:04d}_{mode.cell_id}_{mode.grid_index[0]}_{mode.grid_index[1]}"
        checkpoint = anneal_dir / f"{mode_id}.json"
        if resume and checkpoint.is_file():
            annealed = load_checkpoint_items(
                checkpoint,
                expected_schema_version=schema_version,
                run_fingerprint=run_fingerprint,
                item_name="modes",
                loader=_annealed_mode_from_payload,
            )
            resumed_mode = True
        else:
            anneal = backend.anneal_mode if backend is not None else anneal_phase_lattice_mode
            annealed = anneal(
                context=context,
                mode=mode,
                config=config.annealing,
            )
            write_json_atomic(
                checkpoint,
                {
                    "schema_version": schema_version,
                    "run_fingerprint": run_fingerprint,
                    "mode_id": mode_id,
                    "source_mode": mode,
                    "modes": annealed,
                    "diagnostics": backend.last_annealing_diagnostics if backend is not None else None,
                },
            )
            resumed_mode = False
        import json
        diagnostics = json.loads(checkpoint.read_text()).get("diagnostics")
        all_annealed.extend(annealed)
        anneal_summaries.append(
            {
                "mode_id": mode_id,
                "source_strict_score": mode.strict_score,
                "output_mode_count": len(annealed),
                "diagnostics": diagnostics,
                "best_total_chi2": min((item.total_chi2 for item in annealed), default=None),
                "resumed": resumed_mode,
                "current_call_wall_seconds": time.perf_counter() - started,
                "current_process_cpu_seconds": time.process_time() - cpu_started,
            }
        )

    return tuple(all_annealed), tuple(anneal_summaries)
