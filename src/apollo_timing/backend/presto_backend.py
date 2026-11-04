"""PRESTO adapter: one task representation for scripts and direct execution."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from apollo_timing.io.save_artifacts import atomic_write_text
from apollo_timing.models.foldbank import FoldbankConfig, FoldTask

_REDUCED_CHI2 = re.compile(r"^# Reduced chi-sqr\s*=\s*([^\s]+)", re.MULTILINE)
_SIGMA = re.compile(r"\(~?\s*([0-9.eE+-]+) sigma\)")


def _prepfold_command(
    *,
    prepfold: str,
    pixi: Path | None,
    pixi_manifest: Path | None,
) -> tuple[str, ...]:
    if pixi_manifest is None:
        if "/" in prepfold:
            prepfold = str(Path(prepfold).resolve())
        return (str(prepfold),)
    executable = str(Path(pixi).resolve()) if pixi is not None else "pixi"
    return (
        executable,
        "run",
        "--manifest-path",
        str(Path(pixi_manifest).resolve()),
        str(prepfold),
    )


def _fold_arguments(
    *,
    par_path: Path,
    search_path: Path,
    output_root: Path,
    parts: int,
    channels: int,
    config: FoldbankConfig,
) -> tuple[str, ...]:
    if channels <= 0 or parts <= 0:
        raise ValueError("Fold tasks need positive channel and time-part counts.")
    subbands = next(
        count for count in range(min(config.subbands, channels), 0, -1) if channels % count == 0
    )
    return (
        "-noxwin",
        "-nosearch",
        "-n",
        str(config.profile_bins),
        "-nsub",
        str(subbands),
        "-npart",
        str(parts),
        "-o",
        str(output_root),
        "-par",
        str(par_path),
        str(search_path),
    )


def _parse_bestprof(path: Path) -> dict[str, float | str]:
    text = Path(path).read_text(encoding="utf-8")
    chi2 = _REDUCED_CHI2.search(text)
    sigma = _SIGMA.search(text)
    result: dict[str, float | str] = {"bestprof": str(Path(path))}
    if chi2 is not None:
        value = float(chi2.group(1))
        if np.isfinite(value):
            result["reduced_chi2"] = value
    if sigma is not None:
        value = float(sigma.group(1))
        if np.isfinite(value):
            result["sigma"] = value
    return result


def _task_arguments(task: FoldTask, scratch: Path, config: FoldbankConfig) -> tuple[str, ...]:
    return _fold_arguments(
        par_path=scratch / f"{task.candidate_id}.par",
        search_path=task.observation.path.resolve(),
        output_root=scratch / task.candidate_id / task.observation.observation_id / "fold",
        parts=task.parts,
        channels=task.observation.channels,
        config=config,
    )


def _validate_tasks(tasks: Sequence[FoldTask], command: Sequence[str]) -> None:
    if not command or not command[0]:
        raise ValueError("Fold tasks require an executable command.")
    seen = set()
    models = {}
    for task in tasks:
        key = (task.candidate_id, task.observation.observation_id)
        if key in seen:
            raise ValueError("Duplicate fold task candidate/observation pair.")
        seen.add(key)
        path = task.par_path.resolve()
        if task.candidate_id in models and models[task.candidate_id] != path:
            raise ValueError("A fold candidate ID references conflicting par paths.")
        models[task.candidate_id] = path


def write_fold_script(
    *,
    path: Path,
    command: Sequence[str],
    tasks: Sequence[FoldTask],
    output_dir: Path,
    config: FoldbankConfig,
) -> Path:
    _validate_tasks(tasks, command)
    output_dir = output_dir.resolve()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'scratch="$(mktemp -d /tmp/apollo-foldbank.XXXXXX)"',
        "trap 'rm -rf \"$scratch\"' EXIT",
        "prepfold_cmd=(" + " ".join(shlex.quote(item) for item in command) + ")",
    ]
    copied = set()
    for task in tasks:
        if task.candidate_id not in copied:
            lines.append(
                f'cp {shlex.quote(str(task.par_path.resolve()))} "$scratch/{task.candidate_id}.par"'
            )
            copied.add(task.candidate_id)
        work = f"$scratch/{task.candidate_id}/{task.observation.observation_id}"
        destination = output_dir / "folds" / task.candidate_id / task.observation.observation_id
        lines.append(f'mkdir -p "{work}" {shlex.quote(str(destination))}')
        arguments = _task_arguments(task, Path("$scratch"), config)
        rendered = [
            f'"{item}"' if item.startswith("$scratch/") else shlex.quote(item) for item in arguments
        ]
        lines.append(f'(cd "{work}"; "${{prepfold_cmd[@]}}" ' + " ".join(rendered) + ")")
        lines.append(f'cp "{work}"/fold* {shlex.quote(str(destination))}/')
    atomic_write_text(path, "\n".join(lines) + "\n")
    path.chmod(0o755)
    return path


def run_fold_tasks(
    *,
    command: Sequence[str],
    tasks: Sequence[FoldTask],
    output_dir: Path,
    config: FoldbankConfig,
) -> tuple[dict[str, object], ...]:
    _validate_tasks(tasks, command)
    output_dir = output_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="apollo-foldbank-", dir="/tmp") as temporary:
        scratch = Path(temporary)
        for task in tasks:
            shutil.copy2(task.par_path, scratch / f"{task.candidate_id}.par")

        def fold_one(task: FoldTask) -> dict[str, object]:
            work = scratch / task.candidate_id / task.observation.observation_id
            work.mkdir(parents=True, exist_ok=True)
            arguments = _task_arguments(task, scratch, config)
            completed = subprocess.run(
                [*command, *arguments], cwd=work, check=False, capture_output=True, text=True
            )
            destination = output_dir / "folds" / task.candidate_id / task.observation.observation_id
            destination.mkdir(parents=True, exist_ok=True)
            log = atomic_write_text(
                destination / "prepfold.log", completed.stdout + completed.stderr
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"PRESTO failed with exit code {completed.returncode}; log: {log}"
                )
            products = []
            for source in sorted(work.glob("fold*")):
                if source.is_file():
                    target = destination / source.name
                    shutil.copy2(source, target)
                    products.append(str(target))
            bestprofs = [Path(item) for item in products if item.endswith(".bestprof")]
            result = {
                "candidate_id": task.candidate_id,
                "observation_id": task.observation.observation_id,
                "products": products,
                "log": str(log),
            }
            if len(bestprofs) == 1:
                result.update(_parse_bestprof(bestprofs[0]))
            return result

        with ThreadPoolExecutor(max_workers=config.presto_workers) as executor:
            return tuple(executor.map(fold_one, tasks))
