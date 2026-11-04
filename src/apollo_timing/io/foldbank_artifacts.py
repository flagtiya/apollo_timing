"""Auditable Foldbank output and cache contracts."""

from __future__ import annotations

import json
import platform
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Iterator, Sequence

from apollo_timing.inference.bayesian_partial_ell1_bootstrap import (
    PartialELL1BootstrapMode,
    PartialELL1BootstrapResult,
)
from apollo_timing.io.save_artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from apollo_timing.models.foldbank import (
    BootstrapEntry,
    BootstrapPortfolio,
    EvaluatedMode,
    OrbitalPeriodAlias,
)


def _clock_fingerprint(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError("The clock override must be an existing directory.")
    return {
        item.relative_to(root).as_posix(): sha256_file(item)
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def _prepare_run_output(
    output: Path,
    *,
    contract: dict[str, object],
    resume: bool,
) -> None:
    path = output / "run_contract.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not path.is_file():
            raise ValueError("Foldbank output is not empty; use a new output directory.")
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise ValueError("Foldbank resume contract changed; use a new output directory.")
    atomic_write_json(path, contract)


def write_alias_par(*, source: Path, destination: Path, pb_days: float) -> Path:
    from apollo_timing.io.par_compatibility import _active_parameter_line

    lines = []
    replaced = False
    for line in source.read_text(encoding="utf-8").splitlines():
        parsed = _active_parameter_line(line)
        if parsed is not None and parsed[0] == "PB":
            if replaced:
                raise ValueError("Foldbank input must contain one active PB line.")
            _, fields = parsed
            line = " ".join((fields[0], f"{pb_days:.18g}", *fields[2:]))
            replaced = True
        lines.append(line)
    if not replaced:
        raise ValueError("Foldbank input must contain PB.")
    return atomic_write_text(destination, "\n".join(lines) + "\n")


def implementation_fingerprint(module_paths: Sequence[str]) -> dict[str, str]:
    package_root = Path(__file__).parents[1]
    return {name: sha256_file(package_root / name) for name in module_paths}


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{name: version(name) for name in ("numpy", "scipy", "astropy", "pint-pulsar")},
    }


@contextmanager
def foldbank_run_audit(output: Path) -> Iterator[Callable[[str], None]]:
    stage = "training_preflight"
    path = output / "audit.json"

    def advance(name: str) -> None:
        nonlocal stage
        stage = name
        atomic_write_json(path, {"schema_version": 1, "status": "running", "stage": stage})

    advance(stage)
    try:
        yield advance
    except Exception as error:
        try:
            atomic_write_json(
                path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "stage": stage,
                    "exception_type": type(error).__name__,
                    "reason": str(error),
                },
            )
        except OSError:
            pass  # Keep the original scientific/execution exception if storage also fails.
        raise
    else:
        atomic_write_json(path, {"schema_version": 1, "status": "completed", "stage": stage})


def bootstrap_products(portfolio: BootstrapPortfolio) -> list[dict[str, str]]:
    paths = {
        path
        for result in portfolio.results
        for path in (
            result.audit_path,
            result.normalized_tim_path,
            *(mode.par_path for mode in result.modes),
        )
    }
    return [{"path": str(path), "sha256": sha256_file(path)} for path in sorted(paths)]


def load_bootstrap_portfolio(
    *,
    cache_dir: Path,
    aliases: Sequence[OrbitalPeriodAlias],
    contract: dict[str, object],
) -> BootstrapPortfolio:
    manifest = json.loads((Path(cache_dir).resolve() / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 4 or manifest.get("bootstrap_contract") != contract:
        raise ValueError("Bootstrap cache contract does not match; use a new bootstrap.")
    products = manifest.get("bootstrap_products")
    if not products:
        raise ValueError("Bootstrap cache has no product hashes; use a new bootstrap.")
    verified_paths = set()
    for product in products:
        path = Path(product["path"]).resolve()
        if not path.is_file() or sha256_file(path) != product["sha256"]:
            raise ValueError(f"Bootstrap cache product is missing or changed: {path}")
        verified_paths.add(path)
    alias_ids = {item.alias_id for item in aliases}
    records = manifest["orbital_period_prior"]["aliases"]
    if len(records) != len(alias_ids) or {record["alias_id"] for record in records} != alias_ids:
        raise ValueError("Bootstrap cache does not describe every requested orbital cell.")
    registry = {}
    requested = {alias.alias_id: alias.to_dict() for alias in aliases}
    for record in records:
        if any(record.get(name) != value for name, value in requested[record["alias_id"]].items()):
            raise ValueError("Bootstrap cache orbital-cell bounds are inconsistent.")
        if (
            record["seed_count"] != len(record["seeds"])
            or record["completed_seed_count"]
            != sum(seed["status"] == "completed" for seed in record["seeds"])
            or record["mode_count"] != sum(seed["mode_count"] for seed in record["seeds"])
        ):
            raise ValueError("Bootstrap cache orbital-cell counts are inconsistent.")
        for seed in record["seeds"]:
            key = (record["alias_id"], seed["seed_id"])
            if key in registry:
                raise ValueError("Bootstrap cache contains duplicate seed identities.")
            registry[key] = seed
    entries = []
    identities = set()
    mode_ids = set()
    for payload in manifest.get("bootstrap_entries", []):
        if payload["alias_id"] not in alias_ids:
            raise ValueError("Bootstrap cache has a seed outside the requested orbital cells.")
        result = payload["result"]
        identity = (payload["alias_id"], payload["seed_id"])
        if identity in identities or identity not in registry:
            raise ValueError("Bootstrap cache seed registry is inconsistent.")
        identities.add(identity)
        seed = registry[identity]
        if seed["status"] != "completed" or seed["mode_count"] != len(result["modes"]):
            raise ValueError("Bootstrap cache seed status or mode count is inconsistent.")
        modes = tuple(
            PartialELL1BootstrapMode(
                mode_id=mode["mode_id"],
                par_path=Path(mode["par_path"]).resolve(),
                informative_session_chi2=float(mode["informative_session_chi2"]),
                all_session_shape_chi2=float(mode["all_session_shape_chi2"]),
                parameter_values={
                    name: float(value) for name, value in mode["parameter_values"].items()
                },
            )
            for mode in result["modes"]
        )
        normalized = Path(result["normalized_tim_path"]).resolve()
        audit = Path(result["audit_path"]).resolve()
        if audit != Path(seed["audit_path"]).resolve():
            raise ValueError("Bootstrap cache seed audit identity is inconsistent.")
        for mode in modes:
            if mode.mode_id in mode_ids:
                raise ValueError("Bootstrap cache contains duplicate mode identities.")
            mode_ids.add(mode.mode_id)
        if not {normalized, audit, *(mode.par_path for mode in modes)} <= verified_paths:
            raise ValueError("Bootstrap cache references unhashed products.")
        entries.append(
            BootstrapEntry(
                payload["alias_id"],
                payload["seed_id"],
                PartialELL1BootstrapResult(
                    normalized,
                    audit,
                    modes,
                    result["informative_session_id"],
                    int(result["informative_session_toas"]),
                    float(result["recommended_support_width_scale"]),
                ),
            )
        )
    if not entries:
        raise ValueError("Bootstrap cache contains no completed seeds.")
    if identities != {key for key, seed in registry.items() if seed["status"] == "completed"}:
        raise ValueError("Bootstrap cache omits a completed seed.")
    cached_records = tuple({**record, "cache_used": True} for record in records)
    return BootstrapPortfolio(tuple(aliases), tuple(entries), cached_records)


def store_models(
    *,
    output_dir: Path,
    eligible: Sequence[EvaluatedMode],
    selected: Sequence[EvaluatedMode],
) -> tuple[list[tuple[str, Path, EvaluatedMode]], list[tuple[str, Path, EvaluatedMode]]]:
    import shutil

    model_records = []
    paths_by_mode = {}
    directory = output_dir / "models"
    directory.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(eligible):
        model_id = f"model_{index:04d}"
        path = directory / f"{model_id}.par"
        shutil.copy2(item.mode.par_path, path)
        model_records.append((model_id, path, item))
        paths_by_mode[item.mode.mode_id] = path
    candidate_records = [
        (f"candidate_{index:04d}", paths_by_mode[item.mode.mode_id], item)
        for index, item in enumerate(selected)
    ]
    return model_records, candidate_records
