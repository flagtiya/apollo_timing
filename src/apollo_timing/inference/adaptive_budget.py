from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BudgetDecisionConfig:
    min_rms_improvement_us: float = 1.0
    max_top_k: int = 256
    top_k_growth: float = 4.0
    max_beam_size: int = 20000
    beam_growth: float = 4.0
    max_radius_sigma: float = 9.0
    radius_growth: float = 1.6
    max_enum_per_gap: int = 201
    enum_growth: float = 1.6


@dataclass(frozen=True)
class BudgetRunSnapshot:
    label: str
    output_dir: str
    top_k: int
    beam_size: int
    radius_sigma: float
    max_enum_per_gap: int
    candidate_count: int
    kept_count: int
    certification_count: int
    posterior_best_branch_id: str | None
    best_branch_id_by_rms: str | None
    best_weighted_rms_us: float | None
    best_reduced_chi2: float | None
    certifiable: bool
    proposal: dict[str, Any]

    @property
    def beam_saturated(self) -> bool:
        return self.candidate_count >= self.beam_size


@dataclass(frozen=True)
class BudgetDecision:
    action: str
    reason: str
    stop: bool
    parameter_policy: str
    next_proposal: dict[str, Any] | None = None


@dataclass(frozen=True)
class SeedPerturbation:
    label: str
    parameter_deltas: dict[str, float]
    diagnostics: dict[str, Any]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _parse_float_token(value: str) -> float | None:
    try:
        return float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def estimate_tim_mjd_span_days(tim_path: Path) -> float:
    mjds: list[float] = []
    for raw_line in Path(tim_path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.split():
            value = _parse_float_token(token)
            if value is None:
                continue
            if 30000.0 <= value <= 90000.0:
                mjds.append(value)
                break
    if len(mjds) < 2:
        raise ValueError(f"Cannot estimate an MJD span from {tim_path}.")
    return max(mjds) - min(mjds)


def _read_par_parameter_float(par_path: Path, parameter_name: str) -> float:
    target = parameter_name.upper()
    for raw_line in Path(par_path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = raw_line.split()
        if len(parts) >= 2 and parts[0].upper() == target:
            value = _parse_float_token(parts[1])
            if value is None:
                break
            return value
    raise ValueError(f"Cannot find numeric {parameter_name} in {par_path}.")


def _label_number(value: float) -> str:
    text = f"{abs(float(value)):g}".replace(".", "p")
    return f"m{text}" if value < 0 else f"p{text}"


def generate_single_axis_seed_perturbations(
    par_path: Path,
    tim_path: Path,
    *,
    phase_turns: list[float] | tuple[float, ...] = (1.0,),
    pb_phase_turns: list[float] | tuple[float, ...] | None = None,
    tasc_fractions: list[float] | tuple[float, ...] = (0.25,),
    include_base: bool = True,
) -> list[SeedPerturbation]:
    span_days = estimate_tim_mjd_span_days(tim_path)
    span_seconds = span_days * 86400.0
    if span_seconds <= 0.0:
        raise ValueError(f"Non-positive TIM span for seed perturbations: {span_days} days.")

    variants: list[SeedPerturbation] = []
    if include_base:
        variants.append(
            SeedPerturbation(
                label="base",
                parameter_deltas={},
                diagnostics={"kind": "base", "span_days": span_days},
            )
        )

    for turns in phase_turns:
        turn_count = abs(float(turns))
        if turn_count == 0.0:
            continue
        f0_delta = turn_count / span_seconds
        f1_delta = 2.0 * turn_count / span_seconds**2
        for sign in (-1.0, 1.0):
            signed_turns = sign * turn_count
            variants.append(
                SeedPerturbation(
                    label=f"f0_phase_{_label_number(signed_turns)}",
                    parameter_deltas={"F0": sign * f0_delta},
                    diagnostics={
                        "kind": "f0_phase",
                        "phase_turns_over_span": signed_turns,
                        "span_days": span_days,
                    },
                )
            )
            variants.append(
                SeedPerturbation(
                    label=f"f1_phase_{_label_number(signed_turns)}",
                    parameter_deltas={"F1": sign * f1_delta},
                    diagnostics={
                        "kind": "f1_phase",
                        "phase_turns_over_span": signed_turns,
                        "span_days": span_days,
                    },
                )
            )

    resolved_pb_phase_turns = phase_turns if pb_phase_turns is None else pb_phase_turns
    pb_days = _read_par_parameter_float(par_path, "PB")
    for turns in resolved_pb_phase_turns:
        turn_count = abs(float(turns))
        if turn_count == 0.0:
            continue
        pb_delta = turn_count * pb_days**2 / span_days
        for sign in (-1.0, 1.0):
            signed_turns = sign * turn_count
            variants.append(
                SeedPerturbation(
                    label=f"pb_phase_{_label_number(signed_turns)}",
                    parameter_deltas={"PB": sign * pb_delta},
                    diagnostics={
                        "kind": "pb_orbital_phase_drift",
                        "orbital_phase_turns_over_span": signed_turns,
                        "pb_days": pb_days,
                        "span_days": span_days,
                    },
                )
            )

    if tasc_fractions:
        for fraction in tasc_fractions:
            fraction_abs = abs(float(fraction))
            if fraction_abs == 0.0:
                continue
            for sign in (-1.0, 1.0):
                signed_fraction = sign * fraction_abs
                variants.append(
                    SeedPerturbation(
                        label=f"tasc_pb_{_label_number(signed_fraction)}",
                        parameter_deltas={"TASC": signed_fraction * pb_days},
                        diagnostics={
                            "kind": "tasc_orbital_phase",
                            "pb_fraction": signed_fraction,
                            "pb_days": pb_days,
                            "span_days": span_days,
                        },
                    )
                )
    return variants


def write_par_parameter_delta_variant(
    source_par_path: Path,
    target_par_path: Path,
    parameter_deltas: dict[str, float],
) -> Path:
    source_par_path = Path(source_par_path)
    target_par_path = Path(target_par_path)
    normalized_deltas = {name.upper(): float(delta) for name, delta in parameter_deltas.items()}
    text = source_par_path.read_text(encoding="utf-8")
    if not normalized_deltas:
        target_par_path.parent.mkdir(parents=True, exist_ok=True)
        target_par_path.write_text(text, encoding="utf-8")
        return target_par_path

    found: set[str] = set()
    lines: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw_line)
            continue
        parts = raw_line.split()
        parameter_name = parts[0].upper() if parts else ""
        if len(parts) >= 2 and parameter_name in normalized_deltas:
            seed_value = _parse_float_token(parts[1])
            if seed_value is None:
                raise ValueError(f"Cannot parse {parameter_name} value in {source_par_path}.")
            parts[1] = _format_par_float(seed_value + normalized_deltas[parameter_name])
            raw_line = " ".join(parts)
            found.add(parameter_name)
        lines.append(raw_line)

    missing = sorted(set(normalized_deltas) - found)
    if missing:
        raise ValueError(
            f"Cannot apply seed perturbation because {', '.join(missing)} "
            f"is absent from {source_par_path}."
        )
    target_par_path.parent.mkdir(parents=True, exist_ok=True)
    target_par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_par_path


def resolve_f1_sign_variants(mode: str) -> list[str]:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"original", "seed", "same"}:
        return ["original"]
    if normalized in {"flipped", "flip", "opposite"}:
        return ["flipped"]
    if normalized in {"positive", "plus", "pos"}:
        return ["positive"]
    if normalized in {"negative", "minus", "neg"}:
        return ["negative"]
    if normalized in {"both", "positive_and_negative", "pos_neg", "signs"}:
        return ["positive", "negative"]
    if normalized in {"original_and_flipped", "seed_and_flipped"}:
        return ["original", "flipped"]
    raise ValueError(
        "f1_sign_mode must be one of: original, flipped, positive, negative, both, "
        "or original-and-flipped."
    )


def _format_par_float(value: float) -> str:
    return f"{float(value):.16g}"


def write_f1_sign_variant_par(source_par_path: Path, target_par_path: Path, variant: str) -> Path:
    source_par_path = Path(source_par_path)
    target_par_path = Path(target_par_path)
    normalized_variant = resolve_f1_sign_variants(variant)
    if len(normalized_variant) != 1:
        raise ValueError("write_f1_sign_variant_par requires a single F1 sign variant.")
    selected_variant = normalized_variant[0]
    text = source_par_path.read_text(encoding="utf-8")
    if selected_variant == "original":
        target_par_path.parent.mkdir(parents=True, exist_ok=True)
        target_par_path.write_text(text, encoding="utf-8")
        return target_par_path

    lines: list[str] = []
    found_f1 = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw_line)
            continue
        parts = raw_line.split()
        if len(parts) >= 2 and parts[0].upper() == "F1":
            seed_value = float(parts[1])
            if selected_variant == "flipped":
                value = -seed_value
            elif selected_variant == "positive":
                value = abs(seed_value)
            elif selected_variant == "negative":
                value = -abs(seed_value)
            else:
                raise ValueError(f"Unsupported F1 sign variant: {selected_variant}")
            if value == 0.0:
                value = 0.0
            parts[1] = _format_par_float(value)
            raw_line = " ".join(parts)
            found_f1 = True
        lines.append(raw_line)
    if not found_f1:
        raise ValueError(f"Cannot apply F1 sign variant: F1 is absent from {source_par_path}.")
    target_par_path.parent.mkdir(parents=True, exist_ok=True)
    target_par_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_par_path


def summarize_run_directory(output_dir: Path, *, label: str | None = None) -> BudgetRunSnapshot:
    output_dir = Path(output_dir)
    diagnostics = _load_json(output_dir / "proposal_diagnostics.json")
    posterior = _load_json(output_dir / "active_posterior.json")
    certifications = _load_json(output_dir / "branch_certifications.json")
    config_path = output_dir / "active_binary_timing.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    proposal = dict((payload or {}).get("proposal") or {})

    finite_certifications = []
    for item in certifications:
        rms = _finite_float(item.get("weighted_rms_us"))
        if rms is None:
            continue
        finite_certifications.append((rms, item))
    finite_certifications.sort(key=lambda pair: pair[0])

    best_item = finite_certifications[0][1] if finite_certifications else None
    best_rms = finite_certifications[0][0] if finite_certifications else None

    kept_count = int(diagnostics.get("kept_count", len(certifications)))
    top_k = int(proposal.get("top_k", kept_count))
    beam_size = int(diagnostics.get("beam_size", proposal.get("beam_size", 0)))
    radius_sigma = float(diagnostics.get("radius_sigma", proposal.get("radius_sigma", 0.0)))
    max_enum = int(
        diagnostics.get(
            "max_enum_per_gap",
            proposal.get("max_enum_per_gap", 0),
        )
    )

    return BudgetRunSnapshot(
        label=label or output_dir.name,
        output_dir=str(output_dir),
        top_k=top_k,
        beam_size=beam_size,
        radius_sigma=radius_sigma,
        max_enum_per_gap=max_enum,
        candidate_count=int(diagnostics.get("candidate_count", 0)),
        kept_count=kept_count,
        certification_count=len(certifications),
        posterior_best_branch_id=posterior.get("best_branch_id"),
        best_branch_id_by_rms=(best_item.get("branch_id") if best_item else None),
        best_weighted_rms_us=best_rms,
        best_reduced_chi2=(
            _finite_float(best_item.get("reduced_chi2")) if best_item else None
        ),
        certifiable=bool(posterior.get("certifiable", False)),
        proposal=proposal,
    )


def _copy_proposal(snapshot: BudgetRunSnapshot) -> dict[str, Any]:
    proposal = dict(snapshot.proposal)
    proposal.setdefault("top_k", snapshot.top_k)
    proposal.setdefault("beam_size", snapshot.beam_size)
    proposal.setdefault("radius_sigma", snapshot.radius_sigma)
    proposal.setdefault("max_enum_per_gap", snapshot.max_enum_per_gap)
    return proposal


def _grow_int(value: int, growth: float, maximum: int) -> int:
    if value >= maximum:
        return value
    return min(maximum, max(value + 1, int(round(value * growth))))


def _grow_float(value: float, growth: float, maximum: float) -> float:
    if value >= maximum:
        return value
    return min(maximum, max(value + 1.0e-12, value * growth))


def _best_rms_improved(
    previous: BudgetRunSnapshot,
    latest: BudgetRunSnapshot,
    config: BudgetDecisionConfig,
) -> bool:
    if previous.best_weighted_rms_us is None or latest.best_weighted_rms_us is None:
        return False
    improvement = previous.best_weighted_rms_us - latest.best_weighted_rms_us
    return improvement >= float(config.min_rms_improvement_us)


def decide_next_budget(
    snapshots: list[BudgetRunSnapshot],
    config: BudgetDecisionConfig | None = None,
) -> BudgetDecision:
    active_config = config or BudgetDecisionConfig()
    parameter_policy = "initial_par_only"
    if not snapshots:
        return BudgetDecision(
            action="run_initial",
            reason="No completed APOLLO run is available yet.",
            stop=False,
            parameter_policy=parameter_policy,
            next_proposal=None,
        )

    latest = snapshots[-1]
    if latest.certifiable:
        return BudgetDecision(
            action="stop_certified",
            reason="The posterior quality gate is already certifiable.",
            stop=True,
            parameter_policy=parameter_policy,
        )

    previous = snapshots[-2] if len(snapshots) >= 2 else None
    improved = bool(previous and _best_rms_improved(previous, latest, active_config))
    next_proposal = _copy_proposal(latest)

    if latest.top_k < int(active_config.max_top_k) and (previous is None or improved):
        next_proposal["top_k"] = _grow_int(
            latest.top_k,
            active_config.top_k_growth,
            int(active_config.max_top_k),
        )
        return BudgetDecision(
            action="increase_certification_top_k",
            reason=(
                "The current run is not certifiable and the certified tail has not "
                "been exhausted under the initial par fit parameters."
            ),
            stop=False,
            parameter_policy=parameter_policy,
            next_proposal=next_proposal,
        )

    if latest.beam_saturated and latest.beam_size < int(active_config.max_beam_size):
        next_proposal["beam_size"] = _grow_int(
            latest.beam_size,
            active_config.beam_growth,
            int(active_config.max_beam_size),
        )
        return BudgetDecision(
            action="increase_beam_size",
            reason=(
                "Increasing top_k did not improve the best certified RMS, while the "
                "Bayesian lattice beam is saturated."
            ),
            stop=False,
            parameter_policy=parameter_policy,
            next_proposal=next_proposal,
        )

    if latest.radius_sigma < float(active_config.max_radius_sigma):
        next_proposal["radius_sigma"] = _grow_float(
            latest.radius_sigma,
            active_config.radius_growth,
            float(active_config.max_radius_sigma),
        )
        return BudgetDecision(
            action="widen_wrap_radius",
            reason=(
                "The best branch is stable after top_k/beam expansion, so the next "
                "initial-par-only test widens the per-gap wrap search radius."
            ),
            stop=False,
            parameter_policy=parameter_policy,
            next_proposal=next_proposal,
        )

    if latest.max_enum_per_gap < int(active_config.max_enum_per_gap):
        next_proposal["max_enum_per_gap"] = _grow_int(
            latest.max_enum_per_gap,
            active_config.enum_growth,
            int(active_config.max_enum_per_gap),
        )
        return BudgetDecision(
            action="increase_gap_enum",
            reason=(
                "The radius is at its configured ceiling, but per-gap integer "
                "enumeration can still be enlarged without adding model parameters."
            ),
            stop=False,
            parameter_policy=parameter_policy,
            next_proposal=next_proposal,
        )

    return BudgetDecision(
        action="stop_initial_par_fit_limit",
        reason=(
            "The configured top_k, beam, radius, and integer-enumeration budgets are "
            "exhausted without a significant RMS improvement. The search has plateaued "
            "with the current initial par parameterization."
        ),
        stop=True,
        parameter_policy=parameter_policy,
    )


def _resolve_config_path(base_dir: Path, value: Any | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _find_payload_par_path(payload: dict[str, Any]) -> Path:
    for section_name in ("backend", "geometry_from_pint", "nested_escalation"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get("par_path"):
            return Path(section["par_path"])
    raise ValueError("F1 sign mode requires a par_path in backend or geometry_from_pint.")


def _find_payload_tim_path(payload: dict[str, Any]) -> Path:
    for section_name in ("backend", "geometry_from_pint", "nested_escalation"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get("tim_path"):
            return Path(section["tim_path"])
    raise ValueError("Seed probe mode requires a tim_path in backend or geometry_from_pint.")


def _set_payload_par_path(payload: dict[str, Any], par_path: Path) -> dict[str, Any]:
    updated = deepcopy(payload)
    for section_name in ("backend", "geometry_from_pint", "nested_escalation"):
        section = updated.get(section_name)
        if isinstance(section, dict) and section.get("par_path"):
            section["par_path"] = str(Path(par_path))
    return updated


def _absolutize_path_payload(payload: dict[str, Any], source_base_dir: Path) -> dict[str, Any]:
    resolved = deepcopy(payload)
    path_keys = {"par_path", "tim_path", "cluster_manifest_path", "clock_override_path", "engine_config_path"}
    for section_name in ("geometry_from_pint", "backend", "nested_escalation"):
        section = resolved.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in path_keys:
            if key in section:
                section[key] = _resolve_config_path(source_base_dir, section[key])
    return resolved


def _payload_for_f1_sign_variant(
    payload: dict[str, Any],
    output_dir: Path,
    variant: str,
) -> dict[str, Any]:
    selected_variant = resolve_f1_sign_variants(variant)[0]
    if selected_variant == "original":
        return deepcopy(payload)
    source_par_path = _find_payload_par_path(payload)
    variant_par_path = Path(output_dir) / f"f1_{selected_variant}.par"
    write_f1_sign_variant_par(source_par_path, variant_par_path, selected_variant)
    return _set_payload_par_path(payload, variant_par_path.resolve())


def _payload_for_seed_perturbation(
    payload: dict[str, Any],
    output_dir: Path,
    seed_perturbation: SeedPerturbation | None,
) -> dict[str, Any]:
    if seed_perturbation is None:
        return deepcopy(payload)
    if not seed_perturbation.parameter_deltas:
        return deepcopy(payload)
    source_par_path = _find_payload_par_path(payload)
    variant_par_path = Path(output_dir) / f"seed_{seed_perturbation.label}.par"
    write_par_parameter_delta_variant(
        source_par_path,
        variant_par_path,
        seed_perturbation.parameter_deltas,
    )
    return _set_payload_par_path(payload, variant_par_path.resolve())


def _f1_sign_mode_arg(value: str) -> str:
    try:
        resolve_f1_sign_variants(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _write_trace(
    output_dir: Path,
    snapshots: list[BudgetRunSnapshot],
    decisions: list[BudgetDecision],
    *,
    f1_sign_mode: str = "original",
    f1_sign_variant: str = "original",
    seed_perturbation: SeedPerturbation | None = None,
) -> None:
    payload = {
        "parameter_policy": "initial_par_only",
        "f1_sign_mode": f1_sign_mode,
        "f1_sign_variant": f1_sign_variant,
        "seed_variant": seed_perturbation.label if seed_perturbation else None,
        "seed_diagnostics": seed_perturbation.diagnostics if seed_perturbation else None,
        "snapshots": [asdict(item) for item in snapshots],
        "decisions": [asdict(item) for item in decisions],
    }
    (output_dir / "adaptive_budget_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Adaptive Budget Trace",
        "",
        "- parameter_policy: initial_par_only",
        f"- f1_sign_mode: {f1_sign_mode}",
        f"- f1_sign_variant: {f1_sign_variant}",
        f"- seed_variant: {seed_perturbation.label if seed_perturbation else None}",
    ]
    for index, snapshot in enumerate(snapshots):
        lines.extend(
            [
                "",
                f"## Round {index:02d}",
                "",
                f"- output_dir: {snapshot.output_dir}",
                f"- top_k: {snapshot.top_k}",
                f"- beam_size: {snapshot.beam_size}",
                f"- radius_sigma: {snapshot.radius_sigma:.6g}",
                f"- max_enum_per_gap: {snapshot.max_enum_per_gap}",
                f"- best_branch_id_by_rms: {snapshot.best_branch_id_by_rms}",
                f"- best_weighted_rms_us: {snapshot.best_weighted_rms_us}",
                f"- certifiable: {snapshot.certifiable}",
            ]
        )
        if index < len(decisions):
            decision = decisions[index]
            lines.extend(
                [
                    f"- decision: {decision.action}",
                    f"- reason: {decision.reason}",
                ]
            )
    (output_dir / "adaptive_budget_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_adaptive_budget_variant(
    base_payload: dict[str, Any],
    output_dir: Path,
    *,
    max_rounds: int = 4,
    decision_config: BudgetDecisionConfig | None = None,
    resume: bool = True,
    f1_sign_mode: str = "original",
    f1_sign_variant: str = "original",
    seed_perturbation: SeedPerturbation | None = None,
) -> list[BudgetRunSnapshot]:
    from apollo_timing.inference.run_active_binary_timing import run_active_binary_timing

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    round_payload = _payload_for_f1_sign_variant(
        base_payload,
        output_dir=output_dir,
        variant=f1_sign_variant,
    )
    round_payload = _payload_for_seed_perturbation(
        round_payload,
        output_dir=output_dir,
        seed_perturbation=seed_perturbation,
    )
    proposal = dict(round_payload.get("proposal") or {})
    snapshots: list[BudgetRunSnapshot] = []
    decisions: list[BudgetDecision] = []

    for round_index in range(max(int(max_rounds), 1)):
        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_payload["proposal"] = dict(proposal)
        round_config_path = round_dir / "active_binary_timing.yaml"
        round_config_path.write_text(
            yaml.safe_dump(round_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if not resume or not (round_dir / "branch_certifications.json").exists():
            run_active_binary_timing(round_config_path, output_dir=round_dir)
        snapshot = summarize_run_directory(round_dir, label=f"round_{round_index:02d}")
        snapshots.append(snapshot)
        decision = decide_next_budget(snapshots, decision_config)
        decisions.append(decision)
        _write_trace(
            output_dir,
            snapshots,
            decisions,
            f1_sign_mode=f1_sign_mode,
            f1_sign_variant=f1_sign_variant,
            seed_perturbation=seed_perturbation,
        )
        if decision.stop or decision.next_proposal is None:
            break
        proposal = dict(decision.next_proposal)

    return snapshots


def _write_multi_variant_trace(
    output_dir: Path,
    records: list[dict[str, Any]],
    *,
    f1_sign_mode: str,
) -> None:
    payload = {
        "parameter_policy": "initial_par_only",
        "f1_sign_mode": f1_sign_mode,
        "variants": records,
    }
    (output_dir / "adaptive_budget_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Adaptive Budget Trace",
        "",
        "- parameter_policy: initial_par_only",
        f"- f1_sign_mode: {f1_sign_mode}",
    ]
    for record in records:
        snapshots = record["snapshots"]
        last = snapshots[-1] if snapshots else {}
        lines.extend(
            [
                "",
                f"## F1 {record['f1_sign_variant']}",
                "",
                f"- output_dir: {record['output_dir']}",
                f"- seed_variant: {record.get('seed_variant')}",
                f"- rounds: {len(snapshots)}",
                f"- best_weighted_rms_us: {last.get('best_weighted_rms_us')}",
                f"- best_branch_id_by_rms: {last.get('best_branch_id_by_rms')}",
                f"- final_decision: {record.get('final_decision', {}).get('action')}",
            ]
        )
    (output_dir / "adaptive_budget_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_adaptive_budget(
    config_path: Path,
    output_dir: Path,
    *,
    max_rounds: int = 4,
    decision_config: BudgetDecisionConfig | None = None,
    resume: bool = True,
    f1_sign_mode: str = "original",
    seed_probe_mode: str = "none",
    seed_phase_turns: list[float] | tuple[float, ...] = (1.0,),
    seed_pb_phase_turns: list[float] | tuple[float, ...] | None = None,
    seed_tasc_fractions: list[float] | tuple[float, ...] = (0.25,),
) -> list[BudgetRunSnapshot]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    absolute_payload = _absolutize_path_payload(base_payload, config_path.parent)
    variants = resolve_f1_sign_variants(f1_sign_mode)
    normalized_seed_probe_mode = str(seed_probe_mode).strip().lower().replace("-", "_")
    if normalized_seed_probe_mode == "none":
        seed_perturbations: list[SeedPerturbation | None] = [None]
    elif normalized_seed_probe_mode in {"single_axis", "single"}:
        seed_perturbations = generate_single_axis_seed_perturbations(
            _find_payload_par_path(absolute_payload),
            _find_payload_tim_path(absolute_payload),
            phase_turns=seed_phase_turns,
            pb_phase_turns=seed_pb_phase_turns,
            tasc_fractions=seed_tasc_fractions,
        )
    else:
        raise ValueError("seed_probe_mode must be one of: none, single-axis.")

    if len(variants) == 1 and seed_perturbations == [None]:
        return _run_adaptive_budget_variant(
            absolute_payload,
            output_dir,
            max_rounds=max_rounds,
            decision_config=decision_config,
            resume=resume,
            f1_sign_mode=f1_sign_mode,
            f1_sign_variant=variants[0],
        )

    all_snapshots: list[BudgetRunSnapshot] = []
    records: list[dict[str, Any]] = []
    for variant in variants:
        for seed_perturbation in seed_perturbations:
            path_parts: list[str] = []
            if len(variants) > 1 or variant != "original":
                path_parts.append(f"f1_{variant}")
            if seed_perturbation is not None:
                path_parts.append(f"seed_{seed_perturbation.label}")
            variant_dir = output_dir.joinpath(*path_parts) if path_parts else output_dir
            snapshots = _run_adaptive_budget_variant(
                absolute_payload,
                variant_dir,
                max_rounds=max_rounds,
                decision_config=decision_config,
                resume=resume,
                f1_sign_mode=f1_sign_mode,
                f1_sign_variant=variant,
                seed_perturbation=seed_perturbation,
            )
            all_snapshots.extend(snapshots)
            trace_path = variant_dir / "adaptive_budget_trace.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
            records.append(
                {
                    "f1_sign_variant": variant,
                    "seed_variant": seed_perturbation.label if seed_perturbation else None,
                    "seed_diagnostics": (
                        seed_perturbation.diagnostics if seed_perturbation else None
                    ),
                    "output_dir": str(variant_dir),
                    "snapshots": trace.get("snapshots", [asdict(item) for item in snapshots]),
                    "decisions": trace.get("decisions", []),
                    "final_decision": (trace.get("decisions") or [{}])[-1],
                }
            )
    _write_multi_variant_trace(output_dir, records, f1_sign_mode=f1_sign_mode)
    return all_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run APOLLO with a diagnosis-driven initial-par-only budget schedule."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-top-k", type=int, default=256)
    parser.add_argument("--max-beam-size", type=int, default=20000)
    parser.add_argument("--max-radius-sigma", type=float, default=9.0)
    parser.add_argument("--max-enum-per-gap", type=int, default=201)
    parser.add_argument("--min-rms-improvement-us", type=float, default=1.0)
    parser.add_argument(
        "--f1-sign-mode",
        type=_f1_sign_mode_arg,
        default="original",
        metavar="MODE",
        help=(
            "F1 sign hypothesis: original, flipped, positive, negative, both "
            "(positive+negative), or original-and-flipped."
        ),
    )
    parser.add_argument(
        "--seed-probe-mode",
        choices=["none", "single-axis"],
        default="none",
        help=(
            "Optional seed-basin diagnostic. single-axis probes F0, F1, and TASC "
            "one parameter axis at a time."
        ),
    )
    parser.add_argument(
        "--seed-phase-turns",
        type=float,
        nargs="*",
        default=[1.0],
        help="Phase turns over the TIM span used for F0/F1 seed perturbations.",
    )
    parser.add_argument(
        "--seed-pb-phase-turns",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Orbital phase turns over the TIM span used for PB seed perturbations. "
            "Defaults to --seed-phase-turns when omitted."
        ),
    )
    parser.add_argument(
        "--seed-tasc-fractions",
        type=float,
        nargs="*",
        default=[0.25],
        help="PB fractions used for TASC seed perturbations.",
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_adaptive_budget(
        config_path=args.config,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        decision_config=BudgetDecisionConfig(
            min_rms_improvement_us=args.min_rms_improvement_us,
            max_top_k=args.max_top_k,
            max_beam_size=args.max_beam_size,
            max_radius_sigma=args.max_radius_sigma,
            max_enum_per_gap=args.max_enum_per_gap,
        ),
        resume=not args.no_resume,
        f1_sign_mode=args.f1_sign_mode,
        seed_probe_mode=args.seed_probe_mode,
        seed_phase_turns=args.seed_phase_turns,
        seed_pb_phase_turns=args.seed_pb_phase_turns,
        seed_tasc_fractions=args.seed_tasc_fractions,
    )


if __name__ == "__main__":
    main()
