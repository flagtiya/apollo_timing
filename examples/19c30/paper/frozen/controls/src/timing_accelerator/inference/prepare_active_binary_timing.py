from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from timing_accelerator.inference.cluster_manifest import _iter_tim_rows


def _relative_path(path: Path, base_dir: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base_dir.resolve())
    except ValueError:
        return str(path)


def _read_par_value(par_path: Path, name: str) -> str | None:
    target = name.upper()
    for raw_line in Path(par_path).read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) >= 2 and parts[0].upper() == target:
            return str(parts[1])
    return None


def _default_structure_id(par_path: Path) -> str:
    psrj = _read_par_value(par_path, "PSRJ") or _read_par_value(par_path, "PSR")
    if psrj:
        return f"{psrj.replace('+', 'p').replace('-', 'm')}_local"
    return f"{par_path.stem}_local"


def _cluster_payload_from_tim_rows(
    rows: list[dict[str, object]],
    *,
    max_gap_days: float,
    wrap_min: int,
    wrap_max: int,
    wrap_scale: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No TOA rows were found in the tim file.")

    clusters: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    previous_mjd: float | None = None
    for row in rows:
        mjd = float(row["mjd"])
        if current and previous_mjd is not None and (mjd - previous_mjd) > float(max_gap_days):
            clusters.append(current)
            current = []
        current.append(row)
        previous_mjd = mjd
    if current:
        clusters.append(current)

    cluster_entries = []
    for index, cluster_rows in enumerate(clusters):
        source_filenames = []
        seen = set()
        for row in cluster_rows:
            source_name = str(row["source_filename"])
            if source_name in seen:
                continue
            source_filenames.append(source_name)
            seen.add(source_name)
        cluster_entries.append(
            {
                "cluster_id": f"C{index:03d}",
                "source_filenames": source_filenames,
            }
        )

    gap_entries = []
    for index in range(max(0, len(cluster_entries) - 1)):
        gap_entries.append(
            {
                "gap_id": f"G{index:03d}",
                "left_cluster": cluster_entries[index]["cluster_id"],
                "right_cluster": cluster_entries[index + 1]["cluster_id"],
                "wrap_prior": {
                    "family": "discrete_laplace",
                    "min": int(wrap_min),
                    "max": int(wrap_max),
                    "scale": float(wrap_scale),
                },
            }
        )

    return {
        "clusters": cluster_entries,
        "gaps": gap_entries,
    }


def build_cluster_manifest_payload(
    tim_path: Path,
    *,
    max_gap_days: float = 0.5,
    wrap_min: int = -16,
    wrap_max: int = 16,
    wrap_scale: float = 2.0,
) -> dict[str, Any]:
    rows = _iter_tim_rows(Path(tim_path))
    return _cluster_payload_from_tim_rows(
        rows,
        max_gap_days=max_gap_days,
        wrap_min=wrap_min,
        wrap_max=wrap_max,
        wrap_scale=wrap_scale,
    )


def write_cluster_manifest(
    tim_path: Path,
    output_path: Path,
    *,
    max_gap_days: float = 0.5,
    wrap_min: int = -16,
    wrap_max: int = 16,
    wrap_scale: float = 2.0,
) -> Path:
    payload = build_cluster_manifest_payload(
        tim_path,
        max_gap_days=max_gap_days,
        wrap_min=wrap_min,
        wrap_max=wrap_max,
        wrap_scale=wrap_scale,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def build_active_binary_config_payload(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    *,
    output_dir: Path,
    binary_model: str | None = None,
    clock_override_path: Path | None = None,
    structure_id: str | None = None,
) -> dict[str, Any]:
    par_path = Path(par_path)
    tim_path = Path(tim_path)
    cluster_manifest_path = Path(cluster_manifest_path)
    output_dir = Path(output_dir)
    resolved_binary_model = binary_model or _read_par_value(par_path, "BINARY") or "ELL1"

    geometry_payload = {
        "par_path": _relative_path(par_path, output_dir),
        "tim_path": _relative_path(tim_path, output_dir),
        "cluster_manifest_path": _relative_path(cluster_manifest_path, output_dir),
    }
    backend_payload = {
        "mode": "local-laplace",
        "par_path": _relative_path(par_path, output_dir),
        "tim_path": _relative_path(tim_path, output_dir),
        "cluster_manifest_path": _relative_path(cluster_manifest_path, output_dir),
        "fit_parameter_names": "fit_flags",
        "maxiter": 20,
    }
    if clock_override_path is not None:
        geometry_payload["clock_override_path"] = _relative_path(Path(clock_override_path), output_dir)
        backend_payload["clock_override_path"] = _relative_path(Path(clock_override_path), output_dir)

    return {
        "structure_id": structure_id or _default_structure_id(par_path),
        "binary_model": str(resolved_binary_model).upper(),
        "theta_names": "fit_flags",
        "proposal": {
            "proposal_source": "bayesian-alias-lattice",
            "beam_size": 5000,
            "phase_sigma": 0.08,
            "prior_scale": 1.0,
            "radius_sigma": 5.0,
            "min_radius": 4,
            "max_enum_per_gap": 101,
            "wrap_prior_scale": 200.0,
            "top_k": 16,
        },
        "geometry_from_pint": geometry_payload,
        "backend": backend_payload,
        "posterior": {
            "certifiable_probability": 0.75,
            "certifiable_log_odds": 1.0,
            "max_reduced_chi2": 10.0,
            "max_weighted_rms_us": 50.0,
        },
        "plot_residuals": True,
    }


def write_active_binary_config(
    par_path: Path,
    tim_path: Path,
    cluster_manifest_path: Path,
    output_path: Path,
    *,
    binary_model: str | None = None,
    clock_override_path: Path | None = None,
    structure_id: str | None = None,
) -> Path:
    output_path = Path(output_path)
    payload = build_active_binary_config_payload(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_manifest_path,
        output_dir=output_path.parent,
        binary_model=binary_model,
        clock_override_path=clock_override_path,
        structure_id=structure_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def prepare_active_binary_timing(
    par_path: Path,
    tim_path: Path,
    output_dir: Path,
    *,
    cluster_gap_days: float = 0.5,
    binary_model: str | None = None,
    clock_override_path: Path | None = None,
    structure_id: str | None = None,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    cluster_path = output_dir / "clusters.yaml"
    config_path = output_dir / "active_binary_timing.yaml"
    write_cluster_manifest(
        tim_path=tim_path,
        output_path=cluster_path,
        max_gap_days=cluster_gap_days,
    )
    write_active_binary_config(
        par_path=par_path,
        tim_path=tim_path,
        cluster_manifest_path=cluster_path,
        output_path=config_path,
        binary_model=binary_model,
        clock_override_path=clock_override_path,
        structure_id=structure_id,
    )
    return cluster_path, config_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clusters.yaml and active_binary_timing.yaml from par/tim files."
    )
    parser.add_argument("--par", required=True, type=Path)
    parser.add_argument("--tim", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cluster-gap-days", type=float, default=0.5)
    parser.add_argument("--binary-model", default=None)
    parser.add_argument("--clock-override-path", type=Path, default=None)
    parser.add_argument("--structure-id", default=None)
    args = parser.parse_args()

    cluster_path, config_path = prepare_active_binary_timing(
        par_path=args.par,
        tim_path=args.tim,
        output_dir=args.output_dir,
        cluster_gap_days=args.cluster_gap_days,
        binary_model=args.binary_model,
        clock_override_path=args.clock_override_path,
        structure_id=args.structure_id,
    )
    print(f"Wrote cluster manifest: {cluster_path}")
    print(f"Wrote active timing config: {config_path}")


if __name__ == "__main__":
    main()
