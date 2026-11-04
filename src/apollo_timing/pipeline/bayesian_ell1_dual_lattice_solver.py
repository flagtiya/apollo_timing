from __future__ import annotations

from apollo_timing.models.serialization import lattice_log_mass_increment_from_json

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from apollo_timing.backend.pint_backend import _load_pint_model
from apollo_timing.inference.active_timing_contracts import to_json_ready
from apollo_timing.inference.bayesian_ell1_dual_lattice import (
    ELL1AnnealedMode,
    ELL1AnnealingConfig,
    ELL1AnnealingStep,
    ELL1CellSearchConfig,
    ELL1ConditionalMode,
    ELL1_FIT_PARAMETERS,
)
# Keep historical private imports working without retaining their implementation here.
from apollo_timing.io.timing_solver_artifacts import (
    _atomic_json as _atomic_json, _attach_session_pulse_numbers as _attach_session_pulse_numbers, _candidate_certification_rank as _candidate_certification_rank,
    _format_annealing_failures as _format_annealing_failures,
    _copy_candidate_to_solution as _copy_candidate_to_solution, _disk_reload_metrics as _disk_reload_metrics, _json_digest as _json_digest, _sha256 as _sha256,
    _write_solution_tim as _write_solution_tim,
)


SOLVER_ARCHITECTURE = "apollo_bayesian_ell1_dual_lattice_annealed_solver"
SOLVER_SCHEMA_VERSION = 8


@dataclass(frozen=True)
class BayesianELL1DualLatticeSolverConfig:
    maximum_cell_width: float = 0.25
    support_width_scale: float = 1.0
    minimum_f1_half_width: float = 1.0e-10
    minimum_eccentricity_half_width: float = 5.0e-3
    astrometric_bound_arcsec: float = 300.0
    cell_search: ELL1CellSearchConfig = field(default_factory=ELL1CellSearchConfig)
    annealing: ELL1AnnealingConfig = field(default_factory=ELL1AnnealingConfig)
    maximum_modes_to_anneal: int = 8
    conditional_score_window: float = 2_000.0
    maximum_modes_to_certify: int = 8
    annealed_chi2_window: float = 2_000.0
    fit_iterations: int = 10
    strict_reduced_chi2_threshold: float = 20.0
    target_reduced_chi2_threshold: float = 10.0
    disk_reload_chi2_tolerance: float = 1.0e-2

    def __post_init__(self) -> None:
        positive_floats = (
            self.maximum_cell_width,
            self.support_width_scale,
            self.minimum_f1_half_width,
            self.minimum_eccentricity_half_width,
            self.astrometric_bound_arcsec,
            self.conditional_score_window,
            self.annealed_chi2_window,
            self.strict_reduced_chi2_threshold,
            self.target_reduced_chi2_threshold,
            self.disk_reload_chi2_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive_floats):
            raise ValueError("ELL1 solver scales and thresholds must be positive.")
        positive_integers = (
            self.cell_search.orbital_samples_per_cell,
            self.cell_search.phase_samples,
            self.cell_search.modes_per_cell,
            self.cell_search.fast_shape_maximum_evaluations,
            self.maximum_modes_to_anneal,
            self.maximum_modes_to_certify,
            self.fit_iterations,
        )
        if not all(int(value) > 0 for value in positive_integers):
            raise ValueError("ELL1 solver budgets must be positive integers.")
        if self.target_reduced_chi2_threshold > self.strict_reduced_chi2_threshold:
            raise ValueError("The target reduced chi-square cannot exceed acceptance.")


@dataclass(frozen=True)
class BayesianELL1DualLatticeSolverResult:
    output_dir: Path
    solution_par: Path
    solution_tim: Path
    residuals_csv: Path
    residuals_png: Path
    fit_parameters: tuple[str, ...]
    chi2: float
    disk_reload_chi2: float
    reduced_chi2: float
    weighted_rms_us: float
    accepted: bool
    target_reached: bool
    searched_cells: int
    annealed_modes: int
    certified_modes: int
    resumed: bool = False


def _conditional_mode_from_payload(payload: Mapping[str, Any]) -> ELL1ConditionalMode:
    return ELL1ConditionalMode(
        cell_id=str(payload["cell_id"]),
        grid_index=tuple(int(value) for value in payload["grid_index"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        parameter_values=np.asarray(payload["parameter_values"], dtype=float),
        local_integer_offsets=tuple(
            tuple(int(value) for value in component)
            for component in payload["local_integer_offsets"]
        ),
        shape_chi2=float(payload["shape_chi2"]),
        strict_score=float(payload["strict_score"]),
        log_likelihood=float(payload["log_likelihood"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        posterior_weight=float(payload["posterior_weight"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=lattice_log_mass_increment_from_json(payload["lattice_log_mass_increment"]),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        posterior_in_support=bool(payload["posterior_in_support"]),
        source_cell_id=str(payload.get("source_cell_id", payload["cell_id"])),
        posterior_transport_fraction=float(
            payload.get("posterior_transport_fraction", 1.0)
        ),
    )


def _annealing_step_from_payload(payload: Mapping[str, Any]) -> ELL1AnnealingStep:
    return ELL1AnnealingStep(
        phase_sigma_cycles=float(payload["phase_sigma_cycles"]),
        integer_offsets=tuple(int(value) for value in payload["integer_offsets"]),
        normalized_values=np.asarray(payload["normalized_values"], dtype=float),
        shape_chi2=float(payload["shape_chi2"]),
        phase_chi2=float(payload["phase_chi2"]),
        total_chi2=float(payload["total_chi2"]),
        lattice_strict_score=float(payload["lattice_strict_score"]),
        lattice_log_likelihood=float(payload["lattice_log_likelihood"]),
        lattice_mass_converged=bool(payload["lattice_mass_converged"]),
        lattice_log_mass_increment=lattice_log_mass_increment_from_json(payload["lattice_log_mass_increment"]),
        lattice_visited_nodes=int(payload["lattice_visited_nodes"]),
        function_evaluations=int(payload["function_evaluations"]),
        converged=bool(payload["converged"]),
        proposed_path_count=int(payload.get("proposed_path_count", 1)),
        evaluated_path_count=int(payload.get("evaluated_path_count", 1)),
        paths_selected_for_closure=int(payload.get("paths_selected_for_closure", 1)),
        path_proposal_scope=str(payload.get("path_proposal_scope", "legacy_gaussian_proposals")),
    )


def _annealed_mode_from_payload(payload: Mapping[str, Any]) -> ELL1AnnealedMode:
    return ELL1AnnealedMode(
        cell_id=str(payload["cell_id"]),
        source_grid_index=tuple(
            int(value) for value in payload["source_grid_index"]
        ),
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
        trace=tuple(
            _annealing_step_from_payload(step) for step in payload["trace"]
        ),
        source_cell_id=str(payload.get("source_cell_id", payload["cell_id"])),
    )


def _select_conditional_modes(
    modes: Sequence[ELL1ConditionalMode],
    config: BayesianELL1DualLatticeSolverConfig,
) -> tuple[ELL1ConditionalMode, ...]:
    ranked = sorted(
        (mode for mode in modes if mode.posterior_in_support),
        key=lambda mode: (mode.strict_score, -mode.log_likelihood, mode.cell_id),
    )
    if not ranked:
        return tuple()
    ceiling = ranked[0].strict_score + config.conditional_score_window
    selected: list[ELL1ConditionalMode] = []
    seen: set[tuple[str, tuple[int, ...], tuple[float, ...]]] = set()
    for mode in ranked:
        if mode.strict_score > ceiling:
            break
        key = (
            mode.cell_id,
            mode.integer_offsets,
            tuple(np.round(mode.normalized_values, decimals=8)),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(mode)
        if len(selected) >= config.maximum_modes_to_anneal:
            break
    return tuple(selected)


def _select_annealed_modes(
    modes: Sequence[ELL1AnnealedMode],
    config: BayesianELL1DualLatticeSolverConfig,
) -> tuple[ELL1AnnealedMode, ...]:
    ranked = sorted(
        (mode for mode in modes if np.isfinite(mode.total_chi2)),
        key=lambda mode: (mode.total_chi2, mode.cell_id, mode.integer_offsets),
    )
    if not ranked:
        return tuple()
    ceiling = ranked[0].total_chi2 + config.annealed_chi2_window
    selected: list[ELL1AnnealedMode] = []
    seen: set[tuple[int, ...]] = set()
    for mode in ranked:
        if mode.total_chi2 > ceiling or mode.integer_offsets in seen:
            continue
        seen.add(mode.integer_offsets)
        selected.append(mode)
        if len(selected) >= config.maximum_modes_to_certify:
            break
    return tuple(selected)




def solve_bayesian_ell1_dual_lattice(
    *,
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    config: BayesianELL1DualLatticeSolverConfig | None = None,
    clock_override_path: Path | None = None,
    resume: bool = False,
) -> BayesianELL1DualLatticeSolverResult:
    """Compatibility entry: use mature numerics inside the common binary lifecycle."""

    from apollo_timing.pipeline.bayesian_binary_lattice_solver import (
        BayesianBinaryLatticeSolverConfig,
        solve_bayesian_binary_lattice,
    )
    from apollo_timing.inference.bayesian_binary_lattice import (
        BinaryCellSearchConfig,
        BinaryAnnealingConfig,
    )
    from apollo_timing.inference.bayesian_binary_model_selection import (
        BinaryModelSelectionConfig,
    )

    active = config or BayesianELL1DualLatticeSolverConfig()
    model = _load_pint_model(Path(par_path))
    if (
        "BINARY" not in model.params
        or str(model["BINARY"].value).upper() != "ELL1"
        or set(model.free_params) != set(ELL1_FIT_PARAMETERS)
    ):
        raise ValueError("The compatibility ELL1 entry requires BINARY ELL1 and its nine fit parameters.")
    core = BayesianBinaryLatticeSolverConfig(
        maximum_cell_width=active.maximum_cell_width,
        support_width_scale=active.support_width_scale,
        minimum_f1_half_width=active.minimum_f1_half_width,
        minimum_eccentricity_half_width=active.minimum_eccentricity_half_width,
        astrometric_bound_arcsec=active.astrometric_bound_arcsec,
        cell_search=BinaryCellSearchConfig(
            orbital_samples_per_cell=active.cell_search.orbital_samples_per_cell,
            phase_samples=active.cell_search.phase_samples,
            modes_per_cell=active.cell_search.modes_per_cell,
            chart_maximum_evaluations=active.cell_search.fast_shape_maximum_evaluations,
            finite_difference_step=active.cell_search.finite_difference_step,
            coordinate_prior_sigma=active.cell_search.fast_prior_sigma,
            local_integer_radius=active.cell_search.local_integer_radius,
            lattice=active.cell_search.lattice,
        ),
        annealing=BinaryAnnealingConfig(**{
            name: getattr(active.annealing, name)
            for name in active.annealing.__dataclass_fields__
        }),
        maximum_modes_to_anneal=active.maximum_modes_to_anneal,
        conditional_score_window=active.conditional_score_window,
        maximum_modes_to_certify=active.maximum_modes_to_certify,
        annealed_chi2_window=active.annealed_chi2_window,
        fit_iterations=active.fit_iterations,
        strict_reduced_chi2_threshold=active.strict_reduced_chi2_threshold,
        target_reduced_chi2_threshold=active.target_reduced_chi2_threshold,
        disk_reload_chi2_tolerance=active.disk_reload_chi2_tolerance,
        physical_model_selection=BinaryModelSelectionConfig(enabled=False),
        plan_observations_on_failure=False,
    )
    result = solve_bayesian_binary_lattice(
        par_path=par_path, tim_path=tim_path, output_dir=output_dir,
        config=core, clock_override_path=clock_override_path, resume=resume,
    )
    return BayesianELL1DualLatticeSolverResult(**{
        name: getattr(result, name)
        for name in BayesianELL1DualLatticeSolverResult.__dataclass_fields__
    })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run APOLLO all-cell Bayesian ELL1 dual-lattice search, posterior "
            "annealing, and PINT certification."
        )
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clock-override", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orbital-samples-per-cell", type=int, default=2)
    parser.add_argument("--phase-samples", type=int, default=12)
    parser.add_argument("--modes-per-cell", type=int, default=4)
    parser.add_argument("--chart-evaluations", type=int, default=60)
    parser.add_argument("--maximum-modes-to-anneal", type=int, default=8)
    parser.add_argument("--maximum-modes-to-certify", type=int, default=8)
    parser.add_argument("--fit-iterations", type=int, default=10)
    parser.add_argument("--strict-reduced-chi2", type=float, default=20.0)
    parser.add_argument("--target-reduced-chi2", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BayesianELL1DualLatticeSolverConfig(
        cell_search=ELL1CellSearchConfig(
            orbital_samples_per_cell=args.orbital_samples_per_cell,
            phase_samples=args.phase_samples,
            modes_per_cell=args.modes_per_cell,
            fast_shape_maximum_evaluations=args.chart_evaluations,
        ),
        maximum_modes_to_anneal=args.maximum_modes_to_anneal,
        maximum_modes_to_certify=args.maximum_modes_to_certify,
        fit_iterations=args.fit_iterations,
        strict_reduced_chi2_threshold=args.strict_reduced_chi2,
        target_reduced_chi2_threshold=args.target_reduced_chi2,
    )
    result = solve_bayesian_ell1_dual_lattice(
        par_path=args.par,
        tim_path=args.tim,
        output_dir=args.output,
        config=config,
        clock_override_path=args.clock_override,
        resume=args.resume,
    )
    print(json.dumps(to_json_ready(result), indent=2, allow_nan=False))
    return 0 if result.accepted else 2




__all__ = [
    "BayesianELL1DualLatticeSolverConfig",
    "BayesianELL1DualLatticeSolverResult",
    "build_arg_parser",
    "main",
    "solve_bayesian_ell1_dual_lattice",
]


def _require_current_schema(payload, *, label):
    from apollo_timing.io.checkpoints import require_schema
    require_schema(payload, expected_schema_version=SOLVER_SCHEMA_VERSION, label=label)


def _load_checkpoint_items(path, *, run_fingerprint, item_name, loader):
    from apollo_timing.io.checkpoints import load_checkpoint_items
    return load_checkpoint_items(path, expected_schema_version=SOLVER_SCHEMA_VERSION,
                                 run_fingerprint=run_fingerprint, item_name=item_name, loader=loader)


if __name__ == "__main__":
    raise SystemExit(main())

