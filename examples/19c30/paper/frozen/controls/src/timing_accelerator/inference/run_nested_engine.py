from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from timing_accelerator.inference.cluster_manifest import load_cluster_manifest, resolve_cluster_manifest
from timing_accelerator.inference.hypermodel import ProductSpaceHyperModel, summarize_posterior_samples
from timing_accelerator.inference.likelihood import ExactPintLikelihood
from timing_accelerator.inference.nested_backend import PyMultiNestBackend
from timing_accelerator.inference.prior_transform import (
    build_parameter_prior_specs,
    extract_fit_flagged_parameters,
)
from timing_accelerator.inference.refinement_mcmc import run_conditional_mcmc_refinement
from timing_accelerator.inference.structure_library import load_structure_library
from timing_accelerator.io.save_artifacts import ensure_output_dir, write_json, write_text
from timing_accelerator.models.contracts import NestedEngineConfig, NestedRunResult, NestedState


def _load_engine_config(path: Path) -> NestedEngineConfig:
    payload = _load_structured_payload(path)
    base_dir = path.parent
    reference_par_path = payload.get("reference_par_path")
    clock_override_path = payload.get("pint_clock_override_path")
    return NestedEngineConfig(
        backend=str(payload.get("backend", "pymultinest")).lower(),
        live_points=int(payload.get("live_points", 400)),
        evidence_tolerance=float(payload.get("evidence_tolerance", 0.5)),
        sampling_efficiency=float(payload.get("sampling_efficiency", 0.3)),
        seed=int(payload.get("seed", 1)),
        resume=bool(payload.get("resume", False)),
        reference_par_path=(
            (base_dir / reference_par_path).resolve()
            if reference_par_path and not Path(reference_par_path).is_absolute()
            else (Path(reference_par_path) if reference_par_path else None)
        ),
        pint_clock_override_path=(
            (base_dir / clock_override_path).resolve()
            if clock_override_path and not Path(clock_override_path).is_absolute()
            else (Path(clock_override_path) if clock_override_path else None)
        ),
        fixed_parameters=[str(name).upper() for name in payload.get("fixed_parameters", ["DM"])],
        uniqueness_delta_logz_min=float(payload.get("uniqueness_delta_logz_min", 5.0)),
        uniqueness_structure_mass_min=float(payload.get("uniqueness_structure_mass_min", 0.95)),
        uniqueness_wrap_mass_min=float(payload.get("uniqueness_wrap_mass_min", 0.90)),
        uniqueness_wrap_entropy_max=float(payload.get("uniqueness_wrap_entropy_max", 0.15)),
        uniqueness_quality_ratio_max=float(payload.get("uniqueness_quality_ratio_max", 1.2)),
        uniqueness_max_reduced_chi2=(
            float(payload["uniqueness_max_reduced_chi2"])
            if "uniqueness_max_reduced_chi2" in payload
            else None
        ),
        uniqueness_max_weighted_rms_us=(
            float(payload["uniqueness_max_weighted_rms_us"])
            if "uniqueness_max_weighted_rms_us" in payload
            else None
        ),
        refinement_mode=str(payload.get("refinement_mode", "none")).lower(),
        refinement_walkers=int(payload.get("refinement_walkers", 32)),
        refinement_steps=int(payload.get("refinement_steps", 1000)),
        refinement_burnin=int(payload.get("refinement_burnin", 200)),
        refinement_seed=int(payload.get("refinement_seed", 1)),
        refinement_whitened=bool(payload.get("refinement_whitened", True)),
        refinement_target=str(payload.get("refinement_target", "top1")).lower(),
    )


def _load_structured_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def _extract_global_log_evidence(stats: dict) -> tuple[float | None, float | None]:
    log_evidence = None
    log_evidence_error = None
    for key, value in stats.items():
        lowered = str(key).lower()
        if "global evidence error" in lowered or "global log-evidence error" in lowered:
            log_evidence_error = float(value)
            continue
        if "global evidence" in lowered or "global log-evidence" in lowered:
            log_evidence = float(value)
    return log_evidence, log_evidence_error


def _extract_sample_payload(equal_weighted: np.ndarray, ndim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if equal_weighted.ndim == 1:
        equal_weighted = equal_weighted.reshape(1, -1)
    if equal_weighted.shape[1] < ndim + 1:
        raise ValueError("Equal-weighted posterior must contain ndim parameters plus a log-likelihood column.")
    samples = equal_weighted[:, :ndim]
    log_likelihoods = equal_weighted[:, ndim]
    weights = np.full(samples.shape[0], 1.0 / max(samples.shape[0], 1))
    return samples, log_likelihoods, weights


def _extract_parameter_values(par_path: Path, parameter_names: set[str]) -> dict[str, float | str]:
    names = {name.upper() for name in parameter_names}
    extracted: dict[str, float | str] = {}
    for raw_line in par_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].upper()
        if name not in names:
            continue
        token = parts[1]
        try:
            extracted[name] = float(token)
        except ValueError:
            extracted[name] = token
    return extracted


def _approximate_delta_logz_from_structure_posteriors(structure_posteriors: list) -> float:
    if not structure_posteriors:
        return float("-inf")
    if len(structure_posteriors) == 1:
        return float("inf")
    top1_mass = float(structure_posteriors[0].posterior_mass)
    top2_mass = float(structure_posteriors[1].posterior_mass)
    if top1_mass <= 0.0:
        return float("-inf")
    if top2_mass <= 0.0:
        return float("inf")
    return float(math.log(top1_mass / top2_mass))


def _build_uniqueness_gate_report(
    structure_posteriors: list,
    wrap_posteriors: list,
    reference_comparison: dict | None,
    best_state_reduced_chi2: float | None,
    best_state_weighted_rms_us: float | None,
    engine_config: NestedEngineConfig,
) -> dict[str, object]:
    top_structure = structure_posteriors[0] if structure_posteriors else None
    delta_logz = _approximate_delta_logz_from_structure_posteriors(structure_posteriors)
    top_structure_mass = float(top_structure.posterior_mass) if top_structure else 0.0
    evidence_gate = delta_logz >= engine_config.uniqueness_delta_logz_min
    structure_gate = top_structure_mass >= engine_config.uniqueness_structure_mass_min

    wrap_items: list[dict[str, object]] = []
    wrap_gate = True
    for item in wrap_posteriors:
        map_mass = float(item.posterior_masses.get(item.map_wrap, 0.0))
        map_mass_ok = map_mass >= engine_config.uniqueness_wrap_mass_min
        entropy_ok = float(item.posterior_entropy) <= engine_config.uniqueness_wrap_entropy_max
        wrap_gate = wrap_gate and map_mass_ok and entropy_ok
        wrap_items.append(
            {
                "gap_id": item.gap_id,
                "map_wrap": int(item.map_wrap),
                "map_wrap_mass": map_mass,
                "posterior_entropy": float(item.posterior_entropy),
                "pass_map_mass": map_mass_ok,
                "pass_entropy": entropy_ok,
            }
        )

    quality_payload: dict[str, object] = {
        "mode": "none",
        "pass": True,
    }
    if reference_comparison is not None:
        reduced_ratio = reference_comparison.get("reduced_chi2_ratio_to_reference")
        weighted_ratio = reference_comparison.get("weighted_rms_ratio_to_reference")
        reduced_ok = isinstance(reduced_ratio, (int, float)) and reduced_ratio <= engine_config.uniqueness_quality_ratio_max
        weighted_ok = isinstance(weighted_ratio, (int, float)) and weighted_ratio <= engine_config.uniqueness_quality_ratio_max
        quality_payload = {
            "mode": "reference-ratio",
            "reduced_chi2_ratio_to_reference": reduced_ratio,
            "weighted_rms_ratio_to_reference": weighted_ratio,
            "ratio_threshold": engine_config.uniqueness_quality_ratio_max,
            "pass_reduced_ratio": reduced_ok,
            "pass_weighted_ratio": weighted_ok,
            "pass": bool(reduced_ok and weighted_ok),
        }
    elif (
        engine_config.uniqueness_max_reduced_chi2 is not None
        or engine_config.uniqueness_max_weighted_rms_us is not None
    ):
        reduced_limit = engine_config.uniqueness_max_reduced_chi2
        weighted_limit = engine_config.uniqueness_max_weighted_rms_us
        reduced_ok = (
            True
            if reduced_limit is None
            else (
                isinstance(best_state_reduced_chi2, (int, float))
                and best_state_reduced_chi2 <= reduced_limit
            )
        )
        weighted_ok = (
            True
            if weighted_limit is None
            else (
                isinstance(best_state_weighted_rms_us, (int, float))
                and best_state_weighted_rms_us <= weighted_limit
            )
        )
        quality_payload = {
            "mode": "absolute",
            "best_state_reduced_chi2": best_state_reduced_chi2,
            "best_state_weighted_rms_us": best_state_weighted_rms_us,
            "max_reduced_chi2": reduced_limit,
            "max_weighted_rms_us": weighted_limit,
            "pass_reduced": reduced_ok,
            "pass_weighted": weighted_ok,
            "pass": bool(reduced_ok and weighted_ok),
        }

    overall_pass = bool(evidence_gate and structure_gate and wrap_gate and quality_payload["pass"])
    return {
        "passed": overall_pass,
        "thresholds": {
            "delta_logz_min": engine_config.uniqueness_delta_logz_min,
            "structure_mass_min": engine_config.uniqueness_structure_mass_min,
            "wrap_mass_min": engine_config.uniqueness_wrap_mass_min,
            "wrap_entropy_max": engine_config.uniqueness_wrap_entropy_max,
            "quality_ratio_max": engine_config.uniqueness_quality_ratio_max,
            "max_reduced_chi2": engine_config.uniqueness_max_reduced_chi2,
            "max_weighted_rms_us": engine_config.uniqueness_max_weighted_rms_us,
        },
        "observed": {
            "delta_logz_top1_top2": delta_logz,
            "top_structure_id": top_structure.structure_id if top_structure else None,
            "top_structure_mass": top_structure_mass,
        },
        "evidence_gate_pass": evidence_gate,
        "structure_gate_pass": structure_gate,
        "wrap_gate_pass": wrap_gate,
        "quality_gate": quality_payload,
        "wraps": wrap_items,
    }


def _best_state_from_outputs(
    hypermodel: ProductSpaceHyperModel,
    equal_weighted_samples: np.ndarray,
    log_likelihoods: np.ndarray,
    best_fit: dict | None,
) -> tuple[NestedState | None, float | None, float | None]:
    if best_fit and isinstance(best_fit, dict) and "parameters" in best_fit:
        best_state = hypermodel.decode_physical_vector(list(best_fit["parameters"]))
        return (
            best_state,
            float(best_fit.get("log_likelihood", best_fit.get("loglike", float("nan")))),
            None,
        )
    if equal_weighted_samples.size == 0:
        return None, None, None
    best_index = int(np.argmax(log_likelihoods))
    best_state = hypermodel.decode_physical_vector(equal_weighted_samples[best_index].tolist())
    return best_state, float(log_likelihoods[best_index]), float(1.0 / max(len(log_likelihoods), 1))


def _build_diagnostics_markdown(
    run_result: NestedRunResult,
    reference_comparison: dict | None,
    fixed_parameter_values: dict[str, float | str],
) -> str:
    lines = [
        "# FAST Nested Engine Diagnostics",
        "",
        f"- posterior_sample_count: {run_result.posterior_sample_count}",
        f"- log_evidence: {run_result.log_evidence}",
        f"- unique_solution_par_path: {run_result.unique_solution_par_path}",
        "",
        "## Fixed Parameters",
    ]
    for name, value in sorted(fixed_parameter_values.items()):
        lines.append(f"- `{name}`: {value}")
    lines.extend(
        [
            "",
        "## Structure Posterior",
        ]
    )
    for item in run_result.structure_posteriors:
        lines.append(
            f"- `{item.structure_id}`: posterior_mass={item.posterior_mass}, sample_count={item.sample_count}, rank={item.rank}"
        )
    lines.append("")
    lines.append("## Wrap Posterior")
    for item in run_result.wrap_posteriors:
        lines.append(
            f"- `{item.gap_id}`: map_wrap={item.map_wrap}, posterior_entropy={item.posterior_entropy}"
        )
    if run_result.best_state is not None:
        lines.append("")
        lines.append("## Best Hypothesis")
        lines.append(f"- structure_id: {run_result.best_state.structure_id}")
        lines.append(f"- wraps: {run_result.best_state.wraps}")
        lines.append(f"- active_theta: {run_result.best_state.active_theta}")
        lines.append(f"- best_log_likelihood: {run_result.best_log_likelihood}")
    if run_result.refined_best_state is not None:
        lines.append("")
        lines.append("## Refined Hypothesis")
        lines.append(f"- structure_id: {run_result.refined_best_state.structure_id}")
        lines.append(f"- wraps: {run_result.refined_best_state.wraps}")
        lines.append(f"- active_theta: {run_result.refined_best_state.active_theta}")
    if run_result.refinement_summary:
        lines.append("")
        lines.append("## Refinement")
        lines.append(f"- refinement_summary: {run_result.refinement_summary}")
    if reference_comparison:
        lines.append("")
        lines.append("## Truth Comparison")
        for key in [
            "reference_fit_reduced_chi2",
            "reference_weighted_rms_us",
            "recovered_fit_reduced_chi2",
            "recovered_weighted_rms_us",
            "reduced_chi2_ratio_to_reference",
            "weighted_rms_ratio_to_reference",
            "parameter_deltas",
        ]:
            if key in reference_comparison:
                lines.append(f"- {key}: {reference_comparison[key]}")
    if run_result.uniqueness_gate_report:
        lines.append("")
        lines.append("## Uniqueness Gate")
        lines.append(f"- passed: {run_result.uniqueness_gate_report.get('passed')}")
        observed = run_result.uniqueness_gate_report.get("observed", {})
        lines.append(f"- delta_logz_top1_top2: {observed.get('delta_logz_top1_top2')}")
        lines.append(f"- top_structure_id: {observed.get('top_structure_id')}")
        lines.append(f"- top_structure_mass: {observed.get('top_structure_mass')}")
        lines.append(f"- evidence_gate_pass: {run_result.uniqueness_gate_report.get('evidence_gate_pass')}")
        lines.append(f"- structure_gate_pass: {run_result.uniqueness_gate_report.get('structure_gate_pass')}")
        lines.append(f"- wrap_gate_pass: {run_result.uniqueness_gate_report.get('wrap_gate_pass')}")
        quality_gate = run_result.uniqueness_gate_report.get("quality_gate", {})
        lines.append(f"- quality_gate: {quality_gate}")
    lines.append("")
    return "\n".join(lines)


def run_nested_engine(
    par_path: str | Path,
    tim_path: str | Path,
    cluster_manifest_path: str | Path,
    structure_library_path: str | Path,
    engine_config_path: str | Path,
    output_dir: str | Path,
) -> NestedRunResult:
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    cluster_manifest_path = Path(cluster_manifest_path)
    structure_library_path = Path(structure_library_path)
    engine_config_path = Path(engine_config_path)
    output_dir = ensure_output_dir(Path(output_dir))

    engine_config = _load_engine_config(engine_config_path)
    fit_flagged_params = {name.upper() for name in extract_fit_flagged_parameters(par_path)}
    fixed_parameters = {name.upper() for name in engine_config.fixed_parameters}
    effective_fit_params = fit_flagged_params - fixed_parameters
    fixed_parameter_values = _extract_parameter_values(par_path, fixed_parameters)
    manifest = resolve_cluster_manifest(load_cluster_manifest(cluster_manifest_path), tim_path)
    structures = load_structure_library(
        structure_library_path,
        allowed_fit_params=effective_fit_params,
        forbidden_fit_params=fixed_parameters,
    )
    parameter_priors = build_parameter_prior_specs(par_path, allowed_params=effective_fit_params)
    leaked_fixed_parameters = sorted({prior.name for prior in parameter_priors} & fixed_parameters)
    if leaked_fixed_parameters:
        raise ValueError(
            f"Fixed parameters leaked into nested priors: {leaked_fixed_parameters}. "
            "Check fit-parameter filtering."
        )
    hypermodel = ProductSpaceHyperModel(
        cluster_manifest=manifest,
        structures=structures,
        parameter_priors=parameter_priors,
    )
    likelihood = ExactPintLikelihood(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest=manifest,
        structures=structures,
        parameter_priors=parameter_priors,
        clock_override_path=engine_config.pint_clock_override_path,
    )

    def loglike_callback(cube, ndim: int, nparams: int) -> float:
        state = hypermodel.decode_physical_vector([cube[index] for index in range(ndim)])
        return likelihood.evaluate_state(state).log_likelihood

    backend = PyMultiNestBackend(
        ndim=hypermodel.ndim,
        prior_callback=hypermodel.prior_callback,
        loglike_callback=loglike_callback,
        config=engine_config,
        output_dir=output_dir,
    )
    backend_result = backend.run()
    samples, log_likelihoods, weights = _extract_sample_payload(
        backend_result["equal_weighted_posterior"],
        hypermodel.ndim,
    )
    states = [hypermodel.decode_physical_vector(sample.tolist()) for sample in samples]
    posterior_summary = summarize_posterior_samples(
        states=states,
        log_likelihoods=log_likelihoods.tolist(),
        sample_weights=weights.tolist(),
        gap_ids=[gap.gap_id for gap in manifest.gaps],
    )
    log_evidence, log_evidence_error = _extract_global_log_evidence(backend_result["stats"])
    if log_evidence is None:
        log_evidence = float(posterior_summary["log_evidence"])
    best_state, best_log_likelihood, posterior_weight = _best_state_from_outputs(
        hypermodel=hypermodel,
        equal_weighted_samples=samples,
        log_likelihoods=log_likelihoods,
        best_fit=backend_result.get("best_fit"),
    )
    refinement_result = None
    refinement_summary = {
        "status": "skipped",
        "refinement_mode": engine_config.refinement_mode,
        "reason": "No best state available for refinement.",
    }
    refined_best_state = best_state
    if best_state is not None:
        refinement_result = run_conditional_mcmc_refinement(
            likelihood=likelihood,
            best_state=best_state,
            parameter_priors=parameter_priors,
            config=engine_config,
            posterior_states=states,
        )
        refinement_summary = refinement_result["summary"]
        refined_best_state = refinement_result["refined_state"]
    reference_comparison = None
    best_state_reduced_chi2 = None
    best_state_weighted_rms_us = None
    best_state_failure = None
    certification_state = refined_best_state or best_state
    if certification_state is not None:
        best_eval = likelihood.evaluate_state(certification_state)
        best_state_reduced_chi2 = best_eval.reduced_chi2
        best_state_weighted_rms_us = best_eval.weighted_rms_us
        if best_eval.log_likelihood == float("-inf"):
            best_state_failure = {
                "failure_code": best_eval.failure_code,
                "failure_message": best_eval.failure_message,
            }
    if certification_state is not None and engine_config.reference_par_path is not None:
        reference_comparison = likelihood.build_reference_comparison(
            certification_state,
            engine_config.reference_par_path,
        )
    uniqueness_gate_report = _build_uniqueness_gate_report(
        structure_posteriors=posterior_summary["structure_posteriors"],
        wrap_posteriors=posterior_summary["wrap_posteriors"],
        reference_comparison=reference_comparison,
        best_state_reduced_chi2=best_state_reduced_chi2,
        best_state_weighted_rms_us=best_state_weighted_rms_us,
        engine_config=engine_config,
    )
    unique_solution_par_path = None
    if uniqueness_gate_report["passed"] and certification_state is not None and best_state_failure is None:
        unique_solution_par_path = output_dir / "best_unique_solution.par"
        best_model = likelihood.build_model_from_state(certification_state)
        write_text(unique_solution_par_path, best_model.as_parfile())

    np.savez(
        output_dir / "posterior_samples.npz",
        samples=samples,
        log_likelihoods=log_likelihoods,
        weights=weights,
        dimension_labels=np.array(hypermodel.dimension_labels, dtype=object),
    )
    write_json(
        output_dir / "run_config.json",
        {
            "par_path": par_path.resolve(),
            "tim_path": tim_path.resolve(),
            "cluster_manifest_path": cluster_manifest_path.resolve(),
            "structure_library_path": structure_library_path.resolve(),
            "engine_config_path": engine_config_path.resolve(),
            "engine_config": engine_config,
            "fit_flagged_parameters": sorted(fit_flagged_params),
            "fixed_parameters": sorted(fixed_parameters),
            "effective_fit_params": sorted(effective_fit_params),
            "fixed_parameter_values": fixed_parameter_values,
            "refinement_summary": refinement_summary,
            "uniqueness_gate_report": uniqueness_gate_report,
        },
    )
    write_json(
        output_dir / "log_evidence.json",
        {
            "log_evidence": log_evidence,
            "log_evidence_error": log_evidence_error,
            "backend_stats": backend_result["stats"],
        },
    )
    write_json(output_dir / "structure_posterior.json", posterior_summary["structure_posteriors"])
    write_json(output_dir / "wrap_posterior.json", posterior_summary["wrap_posteriors"])
    write_json(output_dir / "refinement_summary.json", refinement_summary)
    write_json(output_dir / "uniqueness_gate_report.json", uniqueness_gate_report)
    write_json(
        output_dir / "refined_best_hypothesis.json",
        {
            "refined_best_state": refined_best_state,
            "refinement_summary": refinement_summary,
        },
    )
    if refinement_result is not None and refinement_result.get("chain_payload") is not None:
        np.savez(output_dir / "mcmc_chain.npz", **refinement_result["chain_payload"])
    write_json(
        output_dir / "best_hypothesis.json",
        {
            "best_state": best_state,
            "refined_best_state": refined_best_state,
            "best_log_likelihood": best_log_likelihood,
            "posterior_weight": posterior_weight,
            "reference_comparison": reference_comparison,
            "best_state_reduced_chi2": best_state_reduced_chi2,
            "best_state_weighted_rms_us": best_state_weighted_rms_us,
            "best_state_failure": best_state_failure,
            "refinement_summary": refinement_summary,
            "uniqueness_gate_report": uniqueness_gate_report,
            "unique_solution_par_path": unique_solution_par_path,
        },
    )

    result = NestedRunResult(
        output_dir=output_dir,
        log_evidence=log_evidence,
        structure_posteriors=posterior_summary["structure_posteriors"],
        wrap_posteriors=posterior_summary["wrap_posteriors"],
        best_state=best_state,
        refined_best_state=refined_best_state,
        best_log_likelihood=best_log_likelihood,
        posterior_sample_count=int(samples.shape[0]),
        diagnostics_path=output_dir / "diagnostics.md",
        uniqueness_gate_report=uniqueness_gate_report,
        unique_solution_par_path=unique_solution_par_path,
        refinement_summary=refinement_summary,
    )
    diagnostics = _build_diagnostics_markdown(result, reference_comparison, fixed_parameter_values)
    write_text(result.diagnostics_path, diagnostics)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the FAST mixed discrete-continuous nested engine.")
    parser.add_argument("--par", required=True)
    parser.add_argument("--tim", required=True)
    parser.add_argument("--cluster-manifest", required=True)
    parser.add_argument("--structure-library", required=True)
    parser.add_argument("--engine-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_nested_engine(
        par_path=args.par,
        tim_path=args.tim,
        cluster_manifest_path=args.cluster_manifest,
        structure_library_path=args.structure_library,
        engine_config_path=args.engine_config,
        output_dir=args.output_dir,
    )
    print(f"Posterior samples: {result.posterior_sample_count}")
    print(f"Diagnostics written to: {result.diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
