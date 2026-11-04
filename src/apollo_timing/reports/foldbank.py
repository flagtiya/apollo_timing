"""Recovery diagnostics; diagnostic images never define scientific success."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from apollo_timing.models.foldbank import SearchObservation


def _scientific_validation(
    *,
    fold_results: Sequence[dict[str, object]],
    candidate_ids: Sequence[str],
    observation_ids: Sequence[str],
    detection_sigma_threshold: float,
    toa_origin: str = "unknown",
    evaluation_scope: str = "training_epochs",
) -> dict[str, object]:
    threshold = float(detection_sigma_threshold)
    result_by_pair = {
        (str(item["candidate_id"]), str(item["observation_id"])): item for item in fold_results
    }
    recovered = []
    comparisons = []
    for observation_id in observation_ids:
        baseline = result_by_pair.get(("baseline_discovery", observation_id), {})
        baseline_sigma = baseline.get("sigma")
        if baseline_sigma is not None and not np.isfinite(float(baseline_sigma)):
            baseline_sigma = None
        detections = []
        for candidate_id in candidate_ids:
            result = result_by_pair.get((candidate_id, observation_id), {})
            sigma = result.get("sigma")
            if sigma is not None and np.isfinite(float(sigma)) and float(sigma) >= threshold:
                detections.append({"candidate_id": candidate_id, "sigma": float(sigma)})
        baseline_detected = baseline_sigma is not None and float(baseline_sigma) >= threshold
        baseline_available = baseline_sigma is not None and bool(np.isfinite(float(baseline_sigma)))
        recovered_here = baseline_available and not baseline_detected and bool(detections)
        if recovered_here:
            recovered.append(observation_id)
        comparisons.append(
            {
                "observation_id": observation_id,
                "baseline_sigma": (float(baseline_sigma) if baseline_sigma is not None else None),
                "baseline_detected": baseline_detected,
                "baseline_available": baseline_available,
                "bank_detections": detections,
                "recovered_by_foldbank": recovered_here,
            }
        )
    truth_independent_toas = toa_origin in {
        "blind_local_search",
        "external_discovery",
    }
    fold_recovery_success = bool(recovered)
    return {
        "criterion": (
            "truth_independent_multi_session_toas_and_"
            f"{evaluation_scope}_baseline_below_threshold_and_bank_at_or_above_threshold"
        ),
        "evaluation_scope": evaluation_scope,
        "detection_sigma_threshold": threshold,
        "evaluated": bool(fold_results),
        "toa_origin": toa_origin,
        "truth_independent_toas": truth_independent_toas,
        "fold_recovery_success": fold_recovery_success,
        "success": truth_independent_toas and fold_recovery_success,
        "recovered_observations": recovered,
        "comparisons": comparisons,
    }


def _result_png(result: dict[str, object]) -> Path | None:
    products = [Path(str(item)) for item in result.get("products", [])]
    matches = [path for path in products if path.suffix.lower() == ".png" and path.is_file()]
    if len(matches) != 1:
        return None
    return matches[0]


def _write_comparison_png(
    *,
    baseline_png: Path,
    candidate_png: Path,
    baseline_sigma: float,
    candidate_sigma: float,
    title: str,
    destination: Path,
) -> Path:
    import matplotlib.pyplot as plt

    left = plt.imread(baseline_png)
    right = plt.imread(candidate_png)
    figure, axes = plt.subplots(1, 2, figsize=(24, 9.8), dpi=180)
    labels = (
        f"Discovery par: non-detection ({baseline_sigma:.1f} sigma)",
        f"Apollo fold-bank: recovered ({candidate_sigma:.1f} sigma)",
    )
    for axis, image, label in zip(axes, (left, right), labels, strict=True):
        axis.imshow(image, interpolation="none")
        axis.set_title(label, fontsize=17, pad=12)
        axis.set_axis_off()
    figure.suptitle(title, fontsize=20, y=0.985)
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.93, wspace=0.015)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        figure.savefig(temporary, format="png", facecolor="white")
        temporary.replace(destination)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return destination


def _write_scientific_comparisons(
    *,
    fold_results: Sequence[dict[str, object]],
    validation: dict[str, object],
    observations: Sequence[SearchObservation],
    output_dir: Path,
) -> tuple[dict[str, object], ...]:
    by_pair = {
        (str(item["candidate_id"]), str(item["observation_id"])): item for item in fold_results
    }
    observation_by_id = {item.observation_id: item for item in observations}
    records = []
    for comparison in validation["comparisons"]:
        if not comparison["recovered_by_foldbank"]:
            continue
        observation_id = str(comparison["observation_id"])
        detection = max(
            comparison["bank_detections"],
            key=lambda item: float(item["sigma"]),
        )
        candidate_id = str(detection["candidate_id"])
        baseline = by_pair[("baseline_discovery", observation_id)]
        candidate = by_pair[(candidate_id, observation_id)]
        observation = observation_by_id[observation_id]
        destination = (
            Path(output_dir) / "comparisons" / f"{observation_id}_discovery_vs_foldbank.png"
        )
        baseline_png, candidate_png = _result_png(baseline), _result_png(candidate)
        reason = None
        if baseline_png is None or candidate_png is None:
            reason = "PRESTO did not supply one readable PNG per compared fold."
        else:
            try:
                _write_comparison_png(
                    baseline_png=baseline_png,
                    candidate_png=candidate_png,
                    baseline_sigma=float(baseline["sigma"]),
                    candidate_sigma=float(candidate["sigma"]),
                    title=f"PSR {observation.source}, {observation.duration_seconds:.3f} s, PRESTO -nosearch",
                    destination=destination,
                )
            except (OSError, ValueError) as error:
                reason = f"Comparison rendering unavailable: {error}"
        records.append(
            {
                "observation_id": observation_id,
                "baseline_candidate_id": "baseline_discovery",
                "foldbank_candidate_id": candidate_id,
                "baseline_sigma": float(baseline["sigma"]),
                "foldbank_sigma": float(candidate["sigma"]),
                "png": str(destination) if reason is None else None,
                "status": "rendered" if reason is None else "unavailable",
                "reason": reason,
            }
        )
    return tuple(records)
