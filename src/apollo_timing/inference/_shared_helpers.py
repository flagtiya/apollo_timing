"""Shared numerical and serialization helpers for inference modules."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from apollo_timing.models.serialization import to_json_ready


def _finite_vector(values: Sequence[float], dimension: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (dimension,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector of length {dimension}.")
    return result


def _bounded_profile_probe(
    reference: np.ndarray,
    *,
    index: int,
    direction: float,
    lower: np.ndarray,
    upper: np.ndarray,
    requested_step: float,
    evaluate: Callable[[np.ndarray], Any],
) -> tuple[float, Any] | None:
    """Shrink a chart step until it also satisfies the native physical domain."""
    available = (
        float(upper[index] - reference[index])
        if direction > 0.0 else float(reference[index] - lower[index])
    )
    step = min(requested_step, available)
    minimum = max(requested_step * 2.0**-20, 64.0 * np.finfo(float).eps)
    while step >= minimum:
        state = reference.copy()
        state[index] += direction * step
        displacement = float(state[index] - reference[index])
        if displacement == 0.0:
            return None
        try:
            return displacement, evaluate(state)
        except (ArithmeticError, RuntimeError, ValueError):
            step *= 0.5
    return None


def _select_path_diverse_particles(particles: Sequence[Any], limit: int) -> list[Any]:
    """Keep distinct integer paths before extra continuous basins of one path."""
    ranked = sorted(particles, key=lambda item: (item.objective, item.integer_offsets))
    selected: list[Any] = []
    seen_paths: set[tuple[int, ...]] = set()
    seen_basins: set[tuple[Any, ...]] = set()
    for require_new_path in (True, False):
        for particle in ranked:
            path = particle.integer_offsets
            basin = (path, tuple(np.round(particle.normalized_values, decimals=8)))
            if basin in seen_basins or (require_new_path and path in seen_paths):
                continue
            selected.append(particle)
            seen_paths.add(path)
            seen_basins.add(basin)
            if len(selected) >= max(int(limit), 1):
                return selected
    return selected


def _project_to_feasible_segment(
    *,
    reference: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    contains: Callable[[np.ndarray], bool],
    evaluate: Callable[[np.ndarray], Any],
    reference_validated: bool = False,
) -> np.ndarray:
    """Shrink a proposal without evaluating timing at every geometric probe."""
    origin = np.asarray(reference, dtype=float)
    proposal = np.asarray(target, dtype=float)

    def valid(alpha: float, *, evaluate_model: bool = True) -> bool:
        state = origin + float(alpha) * (proposal - origin)
        if not np.all(np.isfinite(state)) or np.any(state < lower) or np.any(state > upper):
            return False
        try:
            if not contains(state):
                return False
            if evaluate_model:
                evaluate(state)
        except (ArithmeticError, RuntimeError, ValueError):
            return False
        return True

    if not reference_validated and not valid(0.0):
        raise ValueError("Annealing reference lies outside physical support.")
    if valid(1.0):
        return proposal.copy()
    evaluate_model = valid(1.0, evaluate_model=False)
    valid_alpha, invalid_alpha = 0.0, 1.0
    for _ in range(64):
        midpoint = 0.5 * (valid_alpha + invalid_alpha)
        if valid(midpoint, evaluate_model=evaluate_model):
            valid_alpha = midpoint
        else:
            invalid_alpha = midpoint
    alpha = max(valid_alpha - 1.0e-8, 0.0)
    if not valid(alpha):
        raise ValueError("Support-projected timing state cannot be evaluated.")
    return origin + alpha * (proposal - origin)


def _screen_exact_path_proposals(
    *,
    proposals: Sequence[tuple[tuple[int, ...], np.ndarray]],
    reference: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    score: Callable[[np.ndarray, tuple[int, ...]], float],
    maximum_paths: int,
    reference_competes: bool = True,
    reserved_path: tuple[int, ...] | None = None,
) -> tuple[dict[tuple[int, ...], np.ndarray], int]:
    """Spend the closure budget on distinct paths ranked at feasible states.

    A Gaussian proposal can lose its advantage after support projection. The
    unchanged reference competes by default. The ordinary transport kernel can
    retain its existing incumbent reservation without changing that policy as
    a side effect of a retry. These scores are not truncated posterior weights.
    """
    ranked: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    reference_scores: dict[tuple[int, ...], float] = {}
    for path, target in proposals:
        states = [np.asarray(reference, dtype=float)] if reference_competes else []
        try:
            states.insert(0, project(target))
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        best = None
        for state in states:
            try:
                at_reference = np.array_equal(state, reference)
                if at_reference and path in reference_scores:
                    value = reference_scores[path]
                else:
                    value = float(score(state, path))
                    if at_reference:
                        reference_scores[path] = value
                if np.isfinite(value) and (best is None or value < best[0]):
                    best = value, path, state.copy()
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError):
                continue
        if best is not None:
            ranked.append(best)
    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: dict[tuple[int, ...], np.ndarray] = {}
    for _value, path, state in ranked:
        if path not in selected:
            selected[path] = state
        if len(selected) >= max(int(maximum_paths), 1):
            break
    if reserved_path is not None and reserved_path not in selected:
        if len(selected) >= max(int(maximum_paths), 1):
            selected.pop(next(reversed(selected)))
        selected[reserved_path] = np.asarray(reference, dtype=float).copy()
    return selected, len(ranked)


def _stable_inverse(matrix: np.ndarray, *, floor: float = 1.0e-10) -> np.ndarray:
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    clipped = np.maximum(eigenvalues, max(float(floor) * scale, np.finfo(float).eps))
    return (eigenvectors / clipped) @ eigenvectors.T


def _normalized_bounds(coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(coordinates.normalized_lower, dtype=float)
    upper = np.asarray(coordinates.normalized_upper, dtype=float)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("Normalized coordinate bounds must have matching dimensions.")
    if not np.all(np.isfinite(lower) & np.isfinite(upper) & (lower < upper)):
        raise ValueError("Normalized coordinate bounds must be finite and ordered.")
    return lower, upper


def _physical_values(coordinates: Any, normalized_values: Sequence[float]) -> np.ndarray:
    search = coordinates.denormalize(np.asarray(normalized_values, dtype=float))
    return np.asarray(coordinates.to_physical(search), dtype=float)


def _normalized_values(coordinates: Any, parameter_values: Sequence[float]) -> np.ndarray:
    search_values = coordinates.to_search(np.asarray(parameter_values, dtype=float))
    return np.asarray(coordinates.normalize(search_values), dtype=float)


def effective_sample_size(log_weights: Sequence[float]) -> float:
    """Return ESS for unnormalized log weights."""

    values = np.asarray(log_weights, dtype=float)
    finite = np.isfinite(values)
    if values.ndim != 1 or values.size == 0 or not np.any(finite):
        return 0.0
    shifted = np.where(finite, values - np.max(values[finite]), -math.inf)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    squared = float(weights @ weights)
    return total * total / squared if total > 0.0 and squared > 0.0 else 0.0


def _resolve_relative_path(base_dir: Path, value: Any | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _branch_id(index: int) -> str:
    return f"branch_{index:04d}"


def _wrap_radians(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _wrap_degrees_near(value: float, center: float) -> float:
    return float(center + (value - center + 180.0) % 360.0 - 180.0)


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "_load_json",
    "_wrap_radians",
    "_wrap_degrees_near",
    "_branch_id",
    "_finite_vector",
    "_stable_inverse",
    "_normalized_bounds",
    "_physical_values",
    "_normalized_values",
    "effective_sample_size",
    "_resolve_relative_path",
    "_write_json",
]
