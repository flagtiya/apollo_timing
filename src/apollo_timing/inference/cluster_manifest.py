from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from apollo_timing.models.contracts import (
    ClusterDefinition,
    ClusterManifest,
    GapDefinition,
    GapWrapPrior,
)


@dataclass(frozen=True)
class SessionPathPlan:
    cluster_ids: tuple[str, ...]
    anchor_cluster_count: int
    chronological_cluster_ids: tuple[str, ...]
    session_midpoints_mjd: tuple[float, ...]
    largest_gap_days: float
    traversal: str


def _iter_tim_rows(tim_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in tim_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "C ")):
            continue
        upper = line.upper()
        if upper.startswith(("FORMAT", "MODE")):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append(
            {
                "source_filename": parts[0],
                "mjd": float(parts[2]),
            }
        )
    return rows


def load_cluster_manifest(path: Path) -> ClusterManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    clusters = [
        ClusterDefinition(
            cluster_id=str(item["cluster_id"]),
            source_filenames=[str(name) for name in item.get("source_filenames", [])],
            parent_cluster_id=(
                str(item["parent_cluster_id"])
                if item.get("parent_cluster_id") is not None
                else None
            ),
        )
        for item in payload.get("clusters", [])
    ]
    gaps = [
        GapDefinition(
            gap_id=str(item["gap_id"]),
            left_cluster=str(item["left_cluster"]),
            right_cluster=str(item["right_cluster"]),
            wrap_prior=GapWrapPrior(
                family=str(item.get("wrap_prior", {}).get("family", "discrete_laplace")),
                min_wrap=int(item.get("wrap_prior", {}).get("min", -16)),
                max_wrap=int(item.get("wrap_prior", {}).get("max", 16)),
                scale=float(item.get("wrap_prior", {}).get("scale", 1.0)),
            ),
        )
        for item in payload.get("gaps", [])
    ]
    return ClusterManifest(clusters=clusters, gaps=gaps)


def write_cluster_manifest(manifest: ClusterManifest, path: Path) -> Path:
    path = Path(path)
    clusters = []
    for cluster in manifest.clusters:
        item = {
            "cluster_id": cluster.cluster_id,
            "source_filenames": list(cluster.source_filenames),
        }
        if cluster.parent_cluster_id is not None:
            item["parent_cluster_id"] = cluster.parent_cluster_id
        clusters.append(item)
    payload = {
        "clusters": clusters,
        "gaps": [
            {
                "gap_id": gap.gap_id,
                "left_cluster": gap.left_cluster,
                "right_cluster": gap.right_cluster,
                "wrap_prior": {
                    "family": gap.wrap_prior.family,
                    "min": gap.wrap_prior.min_wrap,
                    "max": gap.wrap_prior.max_wrap,
                    "scale": gap.wrap_prior.scale,
                },
            }
            for gap in manifest.gaps
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def plan_largest_gap_anchor_path(
    manifest: ClusterManifest,
    tim_path: Path,
) -> SessionPathPlan:
    """Choose a traversal that postpones the largest temporal bridge."""
    resolved = resolve_source_session_manifest(manifest, tim_path)
    rows = _iter_tim_rows(Path(tim_path))
    mjds = [float(row["mjd"]) for row in rows]
    sessions: list[tuple[float, ClusterDefinition]] = []
    for cluster in resolved.clusters:
        indices = list(cluster.toa_indices or [])
        if not indices:
            raise ValueError(
                f"Session '{cluster.cluster_id}' does not contain any TOAs."
            )
        midpoint = sum(mjds[index] for index in indices) / len(indices)
        sessions.append((float(midpoint), cluster))
    sessions.sort(key=lambda item: (item[0], item[1].cluster_id))
    if not sessions:
        raise ValueError("At least one source session is required.")

    chronological_ids = tuple(cluster.cluster_id for _, cluster in sessions)
    midpoints = tuple(midpoint for midpoint, _ in sessions)
    if len(sessions) == 1:
        return SessionPathPlan(
            cluster_ids=chronological_ids,
            anchor_cluster_count=1,
            chronological_cluster_ids=chronological_ids,
            session_midpoints_mjd=midpoints,
            largest_gap_days=0.0,
            traversal="chronological",
        )

    gap_days = [
        midpoints[index + 1] - midpoints[index]
        for index in range(len(midpoints) - 1)
    ]
    split_index = max(range(len(gap_days)), key=lambda index: gap_days[index])
    left_count = split_index + 1
    right_count = len(sessions) - left_count
    left_span = midpoints[split_index] - midpoints[0]
    right_span = midpoints[-1] - midpoints[split_index + 1]
    use_right_anchor = (right_count, right_span) > (left_count, left_span)
    if use_right_anchor:
        ordered = tuple(reversed(chronological_ids))
        anchor_count = right_count
        traversal = "reverse-chronological"
    else:
        ordered = chronological_ids
        anchor_count = left_count
        traversal = "chronological"
    return SessionPathPlan(
        cluster_ids=ordered,
        anchor_cluster_count=anchor_count,
        chronological_cluster_ids=chronological_ids,
        session_midpoints_mjd=midpoints,
        largest_gap_days=float(gap_days[split_index]),
        traversal=traversal,
    )


def write_session_path_manifest(
    *,
    tim_path: Path,
    cluster_manifest_path: Path,
    output_path: Path,
) -> SessionPathPlan:
    """Write an atomic session manifest in the planned bridge order."""
    tim_path = Path(tim_path)
    source_manifest = load_cluster_manifest(Path(cluster_manifest_path))
    resolved = resolve_source_session_manifest(source_manifest, tim_path)
    plan = plan_largest_gap_anchor_path(resolved, tim_path)
    cluster_by_id = {cluster.cluster_id: cluster for cluster in resolved.clusters}
    ordered_clusters = [
        replace(cluster_by_id[cluster_id], toa_indices=None)
        for cluster_id in plan.cluster_ids
    ]
    gap_by_edge = {
        (gap.left_cluster, gap.right_cluster): gap
        for gap in resolved.gaps
    }
    ordered_gaps: list[GapDefinition] = []
    for left_id, right_id in zip(plan.cluster_ids, plan.cluster_ids[1:]):
        gap = gap_by_edge.get((left_id, right_id))
        if gap is not None:
            ordered_gaps.append(gap)
            continue
        reverse_gap = gap_by_edge.get((right_id, left_id))
        if reverse_gap is None:
            raise ValueError(
                f"Session path edge '{left_id}' -> '{right_id}' is not adjacent."
            )
        ordered_gaps.append(
            replace(
                reverse_gap,
                gap_id=f"{reverse_gap.gap_id}:REV",
                left_cluster=left_id,
                right_cluster=right_id,
            )
        )
    write_cluster_manifest(
        ClusterManifest(clusters=ordered_clusters, gaps=ordered_gaps),
        Path(output_path),
    )
    return plan


def resolve_cluster_manifest(manifest: ClusterManifest, tim_path: Path) -> ClusterManifest:
    rows = _iter_tim_rows(tim_path)
    source_names = [str(item["source_filename"]) for item in rows]
    resolved_clusters: list[ClusterDefinition] = []
    cluster_index = {cluster.cluster_id: index for index, cluster in enumerate(manifest.clusters)}

    for cluster in manifest.clusters:
        indices = [
            index
            for index, source_name in enumerate(source_names)
            if source_name in set(cluster.source_filenames)
        ]
        if not indices:
            raise ValueError(f"Cluster '{cluster.cluster_id}' did not match any TOAs in '{tim_path}'.")
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"Cluster '{cluster.cluster_id}' is not contiguous in '{tim_path}'.")
        resolved_clusters.append(replace(cluster, toa_indices=indices))

    if manifest.gaps:
        if len(manifest.gaps) != max(0, len(manifest.clusters) - 1):
            raise ValueError("Gap count must be exactly cluster_count - 1 for the FAST first version.")
        for gap in manifest.gaps:
            if gap.left_cluster not in cluster_index or gap.right_cluster not in cluster_index:
                raise ValueError(f"Gap '{gap.gap_id}' references an unknown cluster.")
            left_idx = cluster_index[gap.left_cluster]
            right_idx = cluster_index[gap.right_cluster]
            if right_idx - left_idx != 1:
                raise ValueError(
                    f"Gap '{gap.gap_id}' must connect adjacent clusters in the manifest ordering."
                )

    return ClusterManifest(clusters=resolved_clusters, gaps=list(manifest.gaps))


def resolve_source_session_manifest(
    manifest: ClusterManifest,
    tim_path: Path,
) -> ClusterManifest:
    resolved = resolve_cluster_manifest(manifest, tim_path)
    if resolved.clusters and all(
        cluster.parent_cluster_id is not None and len(cluster.source_filenames) == 1
        for cluster in resolved.clusters
    ):
        return resolved

    rows = _iter_tim_rows(tim_path)
    source_names = [str(item["source_filename"]) for item in rows]
    mjds = [float(item["mjd"]) for item in rows]
    ordered_sessions: list[tuple[float, int, int, int, ClusterDefinition]] = []
    for cluster_index, cluster in enumerate(resolved.clusters):
        cluster_indices = set(cluster.toa_indices or [])
        for source_index, source_name in enumerate(cluster.source_filenames):
            indices = [
                index
                for index, row_source_name in enumerate(source_names)
                if index in cluster_indices and row_source_name == source_name
            ]
            if not indices:
                continue
            session_id = (
                cluster.cluster_id
                if len(cluster.source_filenames) == 1
                else f"{cluster.cluster_id}:S{source_index:03d}"
            )
            session = ClusterDefinition(
                cluster_id=session_id,
                source_filenames=[source_name],
                toa_indices=indices,
                parent_cluster_id=cluster.parent_cluster_id or cluster.cluster_id,
            )
            ordered_sessions.append(
                (
                    min(mjds[index] for index in indices),
                    min(indices),
                    cluster_index,
                    source_index,
                    session,
                )
            )

    ordered_sessions.sort(key=lambda item: item[:4])
    sessions = [item[-1] for item in ordered_sessions]
    coarse_gaps = {
        (gap.left_cluster, gap.right_cluster): gap
        for gap in resolved.gaps
    }
    internal_gap_counts: dict[str, int] = {}
    gaps: list[GapDefinition] = []
    for left, right in zip(sessions, sessions[1:]):
        left_parent = str(left.parent_cluster_id)
        right_parent = str(right.parent_cluster_id)
        coarse_gap = coarse_gaps.get((left_parent, right_parent))
        if coarse_gap is not None:
            gaps.append(
                replace(
                    coarse_gap,
                    left_cluster=left.cluster_id,
                    right_cluster=right.cluster_id,
                )
            )
            continue

        if left_parent == right_parent:
            gap_index = internal_gap_counts.get(left_parent, 0)
            internal_gap_counts[left_parent] = gap_index + 1
            gap_id = f"{left_parent}:SG{gap_index:03d}"
        else:
            gap_id = f"{left_parent}:{right_parent}:SG000"
        gaps.append(
            GapDefinition(
                gap_id=gap_id,
                left_cluster=left.cluster_id,
                right_cluster=right.cluster_id,
                wrap_prior=GapWrapPrior(
                    family="discrete_laplace",
                    min_wrap=-16,
                    max_wrap=16,
                    scale=1.0,
                ),
            )
        )
    return ClusterManifest(clusters=sessions, gaps=gaps)
