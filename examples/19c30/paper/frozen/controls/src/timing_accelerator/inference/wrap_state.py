from __future__ import annotations

from copy import deepcopy

import numpy as np

from timing_accelerator.models.contracts import ClusterManifest


def cumulative_cluster_wrap_offsets(wraps: list[int], cluster_count: int) -> list[int]:
    if max(0, cluster_count - 1) != len(wraps):
        raise ValueError(
            f"Wrap vector length {len(wraps)} does not match cluster_count - 1 = {max(0, cluster_count - 1)}."
        )
    offsets = [0]
    cumulative = 0
    for wrap in wraps:
        cumulative += int(wrap)
        offsets.append(cumulative)
    return offsets


def apply_wraps_to_toas(toas, cluster_manifest: ClusterManifest, wraps: list[int]):
    adjusted = deepcopy(toas)
    offsets = cumulative_cluster_wrap_offsets(wraps, len(cluster_manifest.clusters))
    if "pulse_number" not in adjusted.table.colnames:
        raise ValueError("TOAs object does not expose a pulse_number column after pulse-number computation.")
    pulse_numbers = np.array(adjusted.table["pulse_number"], copy=True)
    for cluster_index, cluster in enumerate(cluster_manifest.clusters):
        if cluster.toa_indices is None:
            raise ValueError(f"Cluster '{cluster.cluster_id}' has no resolved TOA indices.")
        if offsets[cluster_index] == 0:
            continue
        pulse_numbers[cluster.toa_indices] = pulse_numbers[cluster.toa_indices] + offsets[cluster_index]
    adjusted.table["pulse_number"] = pulse_numbers
    return adjusted
