from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from pathlib import Path
from typing import Callable

from apollo_timing.io.save_artifacts import ensure_output_dir, write_json
from apollo_timing.models.contracts import NestedRunResult


@dataclass(frozen=True)
class BinaryTimingCandidate:
    candidate_id: str
    binary_model: str
    parameter_offsets: dict[str, float]
    wrap_penalty: float = 0.0
    log_prior: float = 0.0
    surrogate_score: float | None = None


@dataclass(frozen=True)
class LayeredBinarySearchConfig:
    top_k_to_certify: int = 3
    offset_scale: float = 1.0
    wrap_penalty_scale: float = 1.0


@dataclass(frozen=True)
class LayeredBinarySearchSummary:
    output_dir: Path
    ranking: list[BinaryTimingCandidate]
    certified_candidate_ids: list[str]
    certification_results: list[NestedRunResult]
    best_candidate_id: str | None
    best_result: NestedRunResult | None


BinaryCandidateCertifier = Callable[[BinaryTimingCandidate, Path], NestedRunResult]
RunNestedEngineFn = Callable[..., NestedRunResult]


def _offset_token(name: str, value: float) -> str:
    sign = "p" if float(value) >= 0.0 else "m"
    magnitude = f"{abs(float(value)):g}".replace(".", "p").replace("-", "m")
    return f"{name.lower()}_{sign}{magnitude}"


def generate_binary_timing_candidates(
    binary_models: list[str],
    orbital_offsets: dict[str, list[float]],
    wrap_offsets: list[int] | None = None,
    model_log_priors: dict[str, float] | None = None,
) -> list[BinaryTimingCandidate]:
    wrap_values = wrap_offsets if wrap_offsets is not None else [0]
    parameter_names = list(orbital_offsets)
    offset_grids = [orbital_offsets[name] for name in parameter_names]
    log_priors = {name.upper(): value for name, value in (model_log_priors or {}).items()}
    candidates: list[BinaryTimingCandidate] = []

    for binary_model in binary_models:
        normalized_model = binary_model.upper()
        for offset_values in product(*offset_grids):
            parameter_offsets = {
                name.upper(): float(value)
                for name, value in zip(parameter_names, offset_values, strict=True)
            }
            offset_id = "_".join(
                _offset_token(name, parameter_offsets[name.upper()])
                for name in parameter_names
            )
            for wrap_offset in wrap_values:
                wrap_token = _offset_token("wrap", float(wrap_offset))
                candidate_id = f"{normalized_model.lower()}_{offset_id}_{wrap_token}"
                candidates.append(
                    BinaryTimingCandidate(
                        candidate_id=candidate_id,
                        binary_model=normalized_model,
                        parameter_offsets=parameter_offsets,
                        wrap_penalty=abs(int(wrap_offset)),
                        log_prior=float(log_priors.get(normalized_model, 0.0)),
                    )
                )
    return candidates


def _format_float(value: float) -> str:
    return f"{float(value):.15g}"


def _shift_par_line(raw_line: str, candidate: BinaryTimingCandidate) -> str:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return raw_line
    parts = raw_line.split()
    name = parts[0].upper()
    if name == "BINARY":
        parts[1] = candidate.binary_model.upper()
        return " ".join(parts)
    if name not in candidate.parameter_offsets or len(parts) < 2:
        return raw_line
    try:
        shifted_value = float(parts[1]) + float(candidate.parameter_offsets[name])
    except ValueError:
        return raw_line
    parts[1] = _format_float(shifted_value)
    return " ".join(parts)


def write_candidate_local_par(
    seed_par_path: Path,
    candidate: BinaryTimingCandidate,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shifted_lines = [
        _shift_par_line(raw_line, candidate)
        for raw_line in Path(seed_par_path).read_text(encoding="utf-8").splitlines()
    ]
    output_path.write_text("\n".join(shifted_lines) + "\n", encoding="utf-8")
    return output_path


def write_candidate_structure_library(
    candidate: BinaryTimingCandidate,
    output_path: Path,
    active_fit_params: list[str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = active_fit_params or ["F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2"]
    params_text = ", ".join(name.upper() for name in params)
    output_path.write_text(
        "\n".join(
            [
                "structures:",
                f"  - structure_id: {candidate.binary_model.lower()}_local",
                f"    binary_model: {candidate.binary_model.upper()}",
                f"    active_fit_params: [{params_text}]",
                f"    core_params: [{params_text}]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return output_path


def build_nested_engine_certifier(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    engine_config_path: Path,
    active_fit_params: list[str] | None = None,
    run_nested_engine_fn: RunNestedEngineFn | None = None,
) -> BinaryCandidateCertifier:
    def certify(candidate: BinaryTimingCandidate, output_dir: Path) -> NestedRunResult:
        from apollo_timing.inference.run_nested_engine import run_nested_engine

        runner = run_nested_engine_fn or run_nested_engine
        output_dir = ensure_output_dir(Path(output_dir))
        local_par_path = write_candidate_local_par(
            Path(par_path),
            candidate,
            output_dir / "local_basin.par",
        )
        local_structure_path = write_candidate_structure_library(
            candidate,
            output_dir / "local_structure_library.yaml",
            active_fit_params=active_fit_params,
        )
        return runner(
            par_path=local_par_path,
            tim_path=Path(tim_path),
            cluster_manifest_path=Path(cluster_manifest_path),
            structure_library_path=local_structure_path,
            engine_config_path=Path(engine_config_path),
            output_dir=output_dir,
        )

    return certify


def _surrogate_score(candidate: BinaryTimingCandidate, config: LayeredBinarySearchConfig) -> float:
    offset_scale = max(float(config.offset_scale), 1.0e-12)
    wrap_penalty_scale = max(float(config.wrap_penalty_scale), 1.0e-12)
    offset_penalty = sum((float(value) / offset_scale) ** 2 for value in candidate.parameter_offsets.values())
    wrap_penalty = abs(float(candidate.wrap_penalty)) / wrap_penalty_scale
    return float(candidate.log_prior) - 0.5 * offset_penalty - wrap_penalty


def rank_candidates_by_surrogate(
    candidates: list[BinaryTimingCandidate],
    config: LayeredBinarySearchConfig | None = None,
) -> list[BinaryTimingCandidate]:
    active_config = config or LayeredBinarySearchConfig()
    scored = [
        replace(candidate, surrogate_score=_surrogate_score(candidate, active_config))
        for candidate in candidates
    ]
    return sorted(
        scored,
        key=lambda candidate: (
            -float(candidate.surrogate_score if candidate.surrogate_score is not None else -math.inf),
            candidate.candidate_id,
        ),
    )


def run_layered_binary_timing_search(
    candidates: list[BinaryTimingCandidate],
    config: LayeredBinarySearchConfig,
    output_dir: Path,
    certifier: BinaryCandidateCertifier,
) -> LayeredBinarySearchSummary:
    output_dir = ensure_output_dir(Path(output_dir))
    ranking = rank_candidates_by_surrogate(candidates, config)
    top_k = max(int(config.top_k_to_certify), 0)
    selected = ranking[:top_k]

    certification_results: list[NestedRunResult] = []
    for rank, candidate in enumerate(selected, start=1):
        candidate_dir = output_dir / f"{rank:02d}_{candidate.candidate_id}"
        certification_results.append(certifier(candidate, candidate_dir))

    best_index: int | None = None
    best_log_evidence = float("-inf")
    for index, result in enumerate(certification_results):
        if result.log_evidence is None:
            continue
        if float(result.log_evidence) > best_log_evidence:
            best_log_evidence = float(result.log_evidence)
            best_index = index

    best_result = certification_results[best_index] if best_index is not None else None
    best_candidate_id = selected[best_index].candidate_id if best_index is not None else None
    summary = LayeredBinarySearchSummary(
        output_dir=output_dir,
        ranking=ranking,
        certified_candidate_ids=[candidate.candidate_id for candidate in selected],
        certification_results=certification_results,
        best_candidate_id=best_candidate_id,
        best_result=best_result,
    )
    write_json(
        output_dir / "layered_search_summary.json",
        {
            "ranking": ranking,
            "certified_candidate_ids": summary.certified_candidate_ids,
            "best_candidate_id": summary.best_candidate_id,
            "best_log_evidence": best_result.log_evidence if best_result is not None else None,
        },
    )
    return summary
