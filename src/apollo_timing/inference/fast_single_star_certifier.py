from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from astropy import units as u

from apollo_timing.backend.pint_backend import (
    _freeze_all_parameters,
    _load_pint_model,
    _load_pint_toas,
    _run_linearized_timing_fit,
)
from apollo_timing.inference.cluster_manifest import (
    _iter_tim_rows,
    load_cluster_manifest,
    resolve_cluster_manifest,
)
from apollo_timing.inference.likelihood import ExactPintLikelihood, LikelihoodEvaluation
from apollo_timing.inference.prior_transform import (
    build_parameter_prior_specs,
    extract_fit_flagged_parameters,
)
from apollo_timing.inference.structure_library import load_structure_library
from apollo_timing.models.contracts import (
    ClusterDefinition,
    ClusterManifest,
    NestedState,
    StructureModelConfig,
)


@dataclass(frozen=True)
class FastSingleStarSearchConfig:
    f0_half_width: float = 1.0e-6
    f1_half_width: float = 1.0e-14
    coarse_grid_size: int = 7
    refinement_rounds: int = 3
    shrink_factor: float = 0.25
    structure_id: str = "isolated_f0_f1"
    linearized_fit_iterations: int = 5
    use_scipy_refinement: bool = True
    fit_parameter_names: Sequence[str] | None = None
    grid_parameter_names: Sequence[str] = ("F0", "F1")


@dataclass(frozen=True)
class FastSingleStarSearchResult:
    best_state: NestedState
    best_evaluation: LikelihoodEvaluation
    evaluation_count: int
    final_f0_half_width: float
    final_f1_half_width: float
    fit_parameter_names: tuple[str, ...]
    final_half_widths: dict[str, float]


def _load_numeric_parameter(path: Path, name: str) -> float:
    target = name.upper()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) >= 2 and parts[0].upper() == target:
            return float(parts[1])
    raise KeyError(f"Parameter '{name}' not found in '{path}'.")


def _model_parameter_as_float(model, name: str) -> float:
    parameter = model[name]
    if name == "RAJ":
        return float(parameter.quantity.to(u.hourangle).value)
    if name == "DECJ":
        return float(parameter.quantity.to(u.deg).value)
    return float(parameter.value)


def _extract_model_theta(model, parameter_names: Sequence[str]) -> dict[str, float]:
    return {
        name: _model_parameter_as_float(model, name)
        for name in parameter_names
        if name in model.params
    }


def _resolve_fit_parameter_names(
    par_path: Path,
    config: FastSingleStarSearchConfig,
    model,
) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(extract_fit_flagged_parameters(par_path)))
    if config.fit_parameter_names is not None:
        requested = tuple(dict.fromkeys(str(name).upper() for name in config.fit_parameter_names))
        if set(requested) != set(names):
            raise ValueError(
                "Fast single-star fitting must follow the .par fit flags exactly. "
                f"Requested {list(requested)}, but '{par_path}' marks {list(names)}."
            )
    missing = [name for name in names if name not in model.params]
    if missing:
        raise ValueError(
            f"Fit-flagged parameter(s) are absent from the PINT model: {missing}. "
            "Check the .par file and structure library."
        )
    if not names:
        raise ValueError(f"No fit-flagged parameters found in '{par_path}'.")
    return names


def _build_state(structure_id: str, theta: Mapping[str, float]) -> NestedState:
    active_theta = {str(name).upper(): float(value) for name, value in theta.items()}
    return NestedState(
        structure_id=structure_id,
        wraps=[],
        theta=dict(active_theta),
        active_theta=dict(active_theta),
        inactive_theta={},
    )


def _build_internal_cluster_manifest(tim_path: Path) -> ClusterManifest:
    rows = _iter_tim_rows(Path(tim_path))
    if not rows:
        raise ValueError(f"No TOA rows were found in '{tim_path}'.")
    source_filenames = tuple(
        dict.fromkeys(str(row["source_filename"]) for row in rows)
    )
    return resolve_cluster_manifest(
        ClusterManifest(
            clusters=[
                ClusterDefinition(
                    cluster_id="isolated_all_toas",
                    source_filenames=list(source_filenames),
                )
            ],
            gaps=[],
        ),
        Path(tim_path),
    )


def _build_internal_structure(
    *,
    structure_id: str,
    fit_parameter_names: Sequence[str],
) -> StructureModelConfig:
    names = [str(name).upper() for name in fit_parameter_names]
    return StructureModelConfig(
        structure_id=str(structure_id),
        binary_model=None,
        active_fit_params=names,
        core_params=names,
    )


def run_fast_single_star_search(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path | None = None,
    structure_library_path: Path | None = None,
    config: FastSingleStarSearchConfig | None = None,
    clock_override_path: Path | None = None,
) -> FastSingleStarSearchResult:
    active_config = config or FastSingleStarSearchConfig()
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    fit_model = _load_pint_model(par_path)
    fit_parameter_names = _resolve_fit_parameter_names(par_path, active_config, fit_model)
    if cluster_manifest_path is None:
        manifest = _build_internal_cluster_manifest(tim_path)
    else:
        manifest = resolve_cluster_manifest(
            load_cluster_manifest(Path(cluster_manifest_path)),
            tim_path,
        )
    if structure_library_path is None:
        structures = [
            _build_internal_structure(
                structure_id=active_config.structure_id,
                fit_parameter_names=fit_parameter_names,
            )
        ]
    else:
        structures = load_structure_library(Path(structure_library_path))
    priors = build_parameter_prior_specs(par_path)
    likelihood = ExactPintLikelihood(
        par_path,
        tim_path,
        manifest,
        structures,
        priors,
        clock_override_path=clock_override_path,
    )

    _freeze_all_parameters(fit_model)
    for name in fit_parameter_names:
        fit_model[name].frozen = False
    fit_toas = _load_pint_toas(
        tim_path,
        seed_model=fit_model,
        clock_override_path=clock_override_path,
    )
    fitted_model, _, _, fit_diagnostics = _run_linearized_timing_fit(
        fit_toas,
        fit_model,
        maxiter=max(int(active_config.linearized_fit_iterations), 1),
    )
    fitted_theta = _extract_model_theta(fitted_model, fit_parameter_names)
    fitted_state = _build_state(active_config.structure_id, fitted_theta)
    fitted_evaluation = likelihood.evaluate_state(fitted_state)

    initial_model = _load_pint_model(par_path)
    initial_theta = _extract_model_theta(initial_model, fit_parameter_names)
    center_theta = dict(fitted_theta)
    f0_half_width = float(active_config.f0_half_width)
    f1_half_width = float(active_config.f1_half_width)
    final_half_widths = {
        name: 0.0
        for name in fit_parameter_names
    }
    if "F0" in final_half_widths:
        final_half_widths["F0"] = f0_half_width
    if "F1" in final_half_widths:
        final_half_widths["F1"] = f1_half_width
    covariance_matrix = fit_diagnostics.get("covariance_matrix")
    design_parameter_names = fit_diagnostics.get("design_parameter_names") or []
    if covariance_matrix is not None:
        covariance_array = np.asarray(covariance_matrix, dtype=float)
        for index, name in enumerate(design_parameter_names):
            parameter_name = str(name).upper()
            if parameter_name in final_half_widths and index < covariance_array.shape[0]:
                diagonal = float(covariance_array[index, index])
                if np.isfinite(diagonal) and diagonal > 0.0:
                    final_half_widths[parameter_name] = float(np.sqrt(diagonal))
    grid_size = max(int(active_config.coarse_grid_size), 3)
    if grid_size % 2 == 0:
        grid_size += 1

    best_state = fitted_state
    best_evaluation = fitted_evaluation
    evaluation_count = 2

    grid_parameter_names = tuple(
        name
        for name in (str(item).upper() for item in active_config.grid_parameter_names)
        if name in fit_parameter_names and name in {"F0", "F1"}
    )
    if grid_parameter_names == ("F0", "F1"):
        center_f0 = float(initial_theta["F0"])
        center_f1 = float(initial_theta["F1"])
        for _ in range(max(int(active_config.refinement_rounds), 0) + 1):
            f0_values = np.linspace(center_f0 - f0_half_width, center_f0 + f0_half_width, grid_size)
            f1_values = np.linspace(center_f1 - f1_half_width, center_f1 + f1_half_width, grid_size)
            for f0 in f0_values:
                for f1 in f1_values:
                    theta = dict(center_theta)
                    theta["F0"] = float(f0)
                    theta["F1"] = float(f1)
                    state = _build_state(active_config.structure_id, theta)
                    evaluation = likelihood.evaluate_state(state)
                    evaluation_count += 1
                    if evaluation.log_likelihood > best_evaluation.log_likelihood:
                        best_state = state
                        best_evaluation = evaluation
            center_f0 = float(best_state.active_theta["F0"])
            center_f1 = float(best_state.active_theta["F1"])
            center_theta.update(best_state.active_theta or {})
            f0_half_width *= float(active_config.shrink_factor)
            f1_half_width *= float(active_config.shrink_factor)
            final_half_widths["F0"] = f0_half_width
            final_half_widths["F1"] = f1_half_width

    if active_config.use_scipy_refinement and grid_parameter_names == ("F0", "F1"):
        try:
            from scipy.optimize import minimize
        except Exception:
            minimize = None
        if minimize is not None:
            scale = np.array(
                [
                    max(abs(float(active_config.f0_half_width)), 1.0e-20),
                    max(abs(float(active_config.f1_half_width)), 1.0e-30),
                ],
                dtype=float,
            )
            origin = np.array([center_f0, center_f1], dtype=float)

            def objective(unit_vector: np.ndarray) -> float:
                physical = origin + np.asarray(unit_vector, dtype=float) * scale
                theta = dict(center_theta)
                theta["F0"] = float(physical[0])
                theta["F1"] = float(physical[1])
                state = _build_state(active_config.structure_id, theta)
                evaluation = likelihood.evaluate_state(state)
                return -float(evaluation.log_likelihood)

            optimization = minimize(
                objective,
                x0=np.zeros(2, dtype=float),
                method="Nelder-Mead",
                options={
                    "maxiter": 80,
                    "xatol": 1.0e-8,
                    "fatol": 1.0e-6,
                    "disp": False,
                },
            )
            physical_best = origin + np.asarray(optimization.x, dtype=float) * scale
            theta = dict(center_theta)
            theta["F0"] = float(physical_best[0])
            theta["F1"] = float(physical_best[1])
            scipy_state = _build_state(active_config.structure_id, theta)
            scipy_evaluation = likelihood.evaluate_state(scipy_state)
            evaluation_count += int(getattr(optimization, "nfev", 0)) + 1
            if scipy_evaluation.log_likelihood > best_evaluation.log_likelihood:
                best_state = scipy_state
                best_evaluation = scipy_evaluation
                center_theta.update(scipy_state.active_theta or {})

    return FastSingleStarSearchResult(
        best_state=best_state,
        best_evaluation=best_evaluation,
        evaluation_count=evaluation_count,
        final_f0_half_width=f0_half_width,
        final_f1_half_width=f1_half_width,
        fit_parameter_names=fit_parameter_names,
        final_half_widths=final_half_widths,
    )
