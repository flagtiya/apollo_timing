"""Run frozen J1912 stability controls using every emitted blind model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]

def portable_json(value):
    if isinstance(value, dict):
        return {k.replace("${BUNDLE}", str(BUNDLE)): portable_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_json(v) for v in value]
    if isinstance(value, str):
        return value.replace("${BUNDLE}", str(BUNDLE))
    return value

import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback


ROOT = BUNDLE
RESULTS = BUNDLE / "recorded"
OLD_ROOT = RESULTS / "prefix_archive"
OUTPUT = Path.cwd() / "paper-controls"
LIBRARY = BUNDLE / "frozen/controls"
THREADS = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)}
SCHEDULE = (.1, .05, .02, .01, .005, .002, .001, .0005, .0002, .0001, 0.)
ARMS = ("baseline", "phase_grid13", "retained_budget", "f1_zero", "f1_width_broad")


def ephemeris_path() -> Path:
    sys.path.insert(0, str(LIBRARY / 'src'))
    from timing_accelerator.backend.pint_backend import _resolve_local_ephemeris_path

    path = _resolve_local_ephemeris_path('DE421')
    if path is None:
        raise FileNotFoundError('Stability controls require a local DE421 kernel.')
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(output: Path, library: Path, timeout: int) -> None:
    snapshot = output / "source_snapshot"
    if snapshot.exists():
        raise FileExistsError("Source snapshot already exists; do not overwrite a running protocol.")
    original = OLD_ROOT / "inputs" / "blind_f1_initial.par"
    manifest_path = OLD_ROOT / "inputs" / "prefixes" / "sampling_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cases = {int(case['session_count']): case for case in manifest['cases']}
    shutil.copytree(library / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(library / "pyproject.toml", snapshot / "pyproject.toml")
    hashes = {path.relative_to(snapshot).as_posix(): sha256(path) for path in sorted(snapshot.rglob('*')) if path.is_file()}
    source_digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write_json(output / "source_manifest.json", {"source": str(library), "sha256": source_digest, "files": hashes})
    input_dir = output / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, input_dir / "blind_original_f1.par")
    lines = original.read_text().splitlines()
    selected = [index for index, line in enumerate(lines) if line.split() and line.split()[0].upper() == 'F1']
    if len(selected) != 1:
        raise ValueError("Expected one original F1 declaration.")
    fields = lines[selected[0]].split()
    fields[1] = '0'
    lines[selected[0]] = ' '.join(fields)
    (input_dir / 'blind_zero_f1.par').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    tasks = []
    for n in (17, 18, 19):
        case = cases[n]
        source_tim = manifest_path.parent / case['tim_file']
        assert sha256(source_tim) == case['tim_sha256']
        tim = input_dir / source_tim.name
        shutil.copy2(source_tim, tim)
        for arm in ARMS:
            par = input_dir / ('blind_zero_f1.par' if arm == 'f1_zero' else 'blind_original_f1.par')
            task = {
                'case_id': f'n{n}_{arm}', 'session_count': n, 'toa_count': int(case['toa_count']),
                'arm': arm, 'par': str(par), 'par_sha256': sha256(par), 'tim': str(tim), 'tim_sha256': sha256(tim),
                'source_sha256': source_digest, 'source_snapshot': str(snapshot),
                'output': str(output / 'runs' / f'n{n}_{arm}'), 'timeout_seconds': timeout,
                'baseline': str(output / 'runs' / f'n{n}_baseline'),
                'inference_contract': 'no reference solution; no manual orbit or pulse-path seed; all emitted models; no resume',
            }
            task_path = output / 'tasks' / (task['case_id'] + '.json')
            write_json(task_path, task)
            tasks.append(str(task_path))
    write_json(output / 'protocol.json', {
        'date': '2026-10-08', 'protocol': '15 finite-search stability controls',
        'source_sha256': source_digest, 'tasks': tasks, 'single_thread_environment': THREADS,
        'schedule': SCHEDULE, 'timeout_seconds': timeout, 'reference_used_only_after_inference': True,
        'arms': list(ARMS), 'original_par_sha256': sha256(original), 'prefix_manifest_sha256': sha256(manifest_path),
        'runner_sha256': sha256(Path(__file__)),
        'analyzer_sha256': sha256(ROOT / 'scripts/analyze_paths.py'),
        'de421_sha256': sha256(ephemeris_path()),
    })
    print(json.dumps({'prepared_tasks': len(tasks), 'source_sha256': source_digest, 'output': str(output)}, indent=2))


def binary_config(arm):
    from timing_accelerator.pipeline import BayesianBinaryLatticeSolverConfig
    from timing_accelerator.inference.bayesian_binary_lattice import BinaryCellSearchConfig, BinaryAnnealingConfig

    cell = BinaryCellSearchConfig(orbital_samples_per_cell=2, phase_samples=12, modes_per_cell=32, chart_maximum_evaluations=60)
    annealing = BinaryAnnealingConfig(phase_sigma_schedule=SCHEDULE)
    if arm == 'phase_grid13':
        cell = replace(cell, phase_samples=13)
    if arm == 'retained_budget':
        cell = replace(cell, modes_per_cell=64, lattice=replace(cell.lattice, maximum_lattice_candidates=512, maximum_lattice_nodes=2_000_000))
        annealing = replace(
            annealing, paths_per_state=6, beam_size=8, maximum_closure_evaluations=600,
            lattice=replace(annealing.lattice, maximum_lattice_candidates=512, maximum_lattice_nodes=4_000_000),
        )
    return BayesianBinaryLatticeSolverConfig(
        cell_search=cell, annealing=annealing,
        maximum_modes_to_anneal=48 if arm == 'retained_budget' else 24,
        maximum_modes_to_certify=32 if arm == 'retained_budget' else 16,
        plan_observations_on_failure=False,
    )


def worker(task_path: Path, cpu: int) -> int:
    task = json.loads(task_path.read_text())
    os.sched_setaffinity(0, {cpu})
    expected_source = Path(task['source_snapshot']) / 'src'
    sys.path.insert(0, str(expected_source))
    import timing_accelerator
    from loguru import logger
    from timing_accelerator.models.serialization import to_json_ready
    from timing_accelerator.inference.bayesian_blind_orbit_discovery import BlindOrbitDiscoveryConfig, discover_blind_binary_orbits
    from timing_accelerator.pipeline import solve_bayesian_binary_lattice, solve_bayesian_ell1_dual_lattice
    from timing_accelerator.pipeline.bayesian_binary_lattice_solver import _ell1_config
    from timing_accelerator.io.timing_update import normalize_tempo2_tim_file

    assert Path(timing_accelerator.__file__).resolve().is_relative_to(expected_source.resolve())
    assert sha256(Path(task['par'])) == task['par_sha256'] and sha256(Path(task['tim'])) == task['tim_sha256']
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    output = Path(task['output'])
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = binary_config(task['arm'])
    write_json(output / 'started.json', {
        'task_sha256': sha256(task_path), 'source_sha256': task['source_sha256'], 'python': sys.version,
        'package': timing_accelerator.__version__, 'imported_source': timing_accelerator.__file__,
        'config': to_json_ready(config), 'cpu': cpu, 'threads': {name: os.environ.get(name) for name in THREADS},
        'wall_started_unix': time.time(),
    })
    normalized = normalize_tempo2_tim_file(input_path=Path(task['tim']), output_path=output / 'input.normalized.tim')
    if task['arm'] in ('baseline', 'f1_zero'):
        discovery = discover_blind_binary_orbits(
            par_path=Path(task['par']), tim_path=normalized, output_dir=output / 'blind_orbit_discovery',
            config=BlindOrbitDiscoveryConfig(parallel_workers=1), resume=False,
        )
        assert discovery.session_count == task['session_count'] and discovery.toa_count == task['toa_count']
        modes = [dict(mode_id=mode.mode_id, model_name=mode.model_name, par=str(mode.par_path), par_sha256=sha256(mode.par_path), random_seed=mode.random_seed, parameter_values=mode.parameter_values) for mode in discovery.modes]
        write_json(output / 'discovery_portfolio.json', {'modes': modes, 'discovery': to_json_ready(discovery)})
    else:
        baseline = Path(task['baseline']) / 'discovery_portfolio.json'
        original = json.loads(baseline.read_text())
        modes = []
        for mode in original['modes']:
            assert sha256(Path(mode['par'])) == mode['par_sha256']
            copied = output / 'frozen_discovery_inputs' / (mode['mode_id'] + '.par')
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mode['par'], copied)
            modes.append({**mode, 'par': str(copied)})
        write_json(output / 'discovery_portfolio.json', {'modes': modes, 'parent_portfolio': str(baseline), 'parent_sha256': sha256(baseline)})
    attempts = []
    for mode in modes:
        child = output / 'blind_binary_candidates' / mode['mode_id']
        attempt = {'mode': mode, 'output': str(child)}
        try:
            if task['arm'] == 'f1_width_broad':
                if mode['model_name'] != 'ELL1':
                    raise ValueError('The isolated F1-width control applies to ELL1 only.')
                ell1 = replace(_ell1_config(config), minimum_f1_half_width=1.0e-9)
                write_json(child / 'requested_config.json', to_json_ready(ell1))
                result = solve_bayesian_ell1_dual_lattice(par_path=Path(mode['par']), tim_path=normalized, output_dir=child, config=ell1, resume=False)
            else:
                result = solve_bayesian_binary_lattice(par_path=Path(mode['par']), tim_path=normalized, output_dir=child, config=config, resume=False)
            attempt['result'] = to_json_ready(result)
        except Exception as error:
            attempt['error'] = f'{type(error).__name__}: {error}'
            attempt['traceback'] = traceback.format_exc()
            traceback.print_exc()
        attempts.append(attempt)
        write_json(output / 'solver_attempts.json', attempts)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    completed = {
        'case_id': task['case_id'], 'source_sha256': task['source_sha256'], 'mode_count': len(modes),
        'attempts': attempts, 'wall_seconds': time.monotonic() - started,
        'user_cpu_seconds': usage.ru_utime, 'system_cpu_seconds': usage.ru_stime, 'max_rss_kib': usage.ru_maxrss,
        'accepted_child_count': sum(bool(attempt.get('result', {}).get('accepted')) for attempt in attempts),
    }
    write_json(output / 'completed.json', completed)
    print(json.dumps(completed, indent=2))
    return 0 if completed['accepted_child_count'] else 2


def launch(task_path: Path, cpu: int, output_root: Path):
    task = json.loads(task_path.read_text())
    protocol = json.loads((output_root / 'protocol.json').read_text())
    assert sha256(Path(__file__)) == protocol['runner_sha256']
    assert sha256(ROOT / 'scripts/analyze_paths.py') == protocol['analyzer_sha256']
    kernel = ephemeris_path()
    if sha256(kernel) != protocol['de421_sha256']:
        raise ValueError('DE421 changed since reviewer protocol preparation.')
    directory = Path(task['output'])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'launch.json').exists():
        raise FileExistsError(f'Refusing to overwrite existing launch: {directory}')
    environment = {**os.environ, **THREADS, 'PYTHONDONTWRITEBYTECODE': '1', 'MPLBACKEND': 'Agg',
                   'PYTHONPATH': str(Path(task['source_snapshot']) / 'src'),
                   'TIMING_ACCELERATOR_EPHEM_PATH': str(kernel)}
    command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output_root), '--worker', str(task_path), '--cpu', str(cpu)]
    write_json(directory / 'launch.json', {'command': command, 'task_sha256': sha256(task_path), 'source_sha256': task['source_sha256'], 'started_unix': time.time(), 'timeout_seconds': task['timeout_seconds']})
    started = time.monotonic()
    with (directory / 'stdout.log').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=task['timeout_seconds'])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    from analyze_paths import analyze_solver_output
    reference = OLD_ROOT / 'reference/apollo/solution.tim'
    analysis = analyze_solver_output(directory, reference)
    write_json(directory / 'analysis.json', analysis)
    runtime = {'case_id': task['case_id'], 'return_code': code, 'timeout': timed_out, 'wall_seconds': time.monotonic() - started, 'counts': analysis['counts']}
    write_json(directory / 'runtime.json', runtime)
    print(json.dumps(runtime), flush=True)
    return runtime


def run_queue(output: Path, workers: int) -> None:
    protocol = json.loads((output / 'protocol.json').read_text())
    paths = [Path(path) for path in protocol['tasks']]
    pending = sorted(
        [path for path in paths if not (Path(json.loads(path.read_text())['output']) / 'runtime.json').exists()],
        key=lambda path: (ARMS.index(json.loads(path.read_text())['arm']), json.loads(path.read_text())['session_count']),
    )
    cpus = sorted(os.sched_getaffinity(0))[:workers]
    running = {}
    available = list(cpus)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or running:
            for path in list(pending):
                if not available:
                    break
                task = json.loads(path.read_text())
                if task['arm'] not in ('baseline', 'f1_zero'):
                    parent = Path(task['baseline'])
                    if not (parent / 'runtime.json').exists():
                        continue
                    if not (parent / 'discovery_portfolio.json').exists():
                        write_json(Path(task['output']) / 'runtime.json', {'case_id': task['case_id'], 'blocked': 'baseline discovery unavailable'})
                        pending.remove(path)
                        continue
                cpu = available.pop(0)
                running[pool.submit(launch, path, cpu, output)] = (path, cpu)
                pending.remove(path)
                print(f"START {task['case_id']} cpu={cpu}", flush=True)
            if not running and pending:
                raise RuntimeError('No runnable task; inspect parent outcomes.')
            if running:
                finished, _ = wait(running, timeout=30, return_when=FIRST_COMPLETED)
                for future in finished:
                    path, cpu = running.pop(future)
                    available.append(cpu)
                    future.result()
            snapshot = {
                'running': [json.loads(path.read_text())['case_id'] for path, _ in running.values()],
                'pending': [json.loads(path.read_text())['case_id'] for path in pending],
                'completed': [json.loads(path.read_text())['case_id'] for path in paths if (Path(json.loads(path.read_text())['output']) / 'runtime.json').exists()],
                'updated_unix': time.time(),
            }
            write_json(output / 'progress.json', snapshot)
    print('All registered cases have outcomes.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, default=LIBRARY)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(BUNDLE):
        raise ValueError("Use an output directory outside the example bundle")
    if args.worker:
        return worker(args.worker.resolve(), args.cpu)
    if args.prepare:
        prepare(args.output.resolve(), args.library.resolve(), args.timeout)
    if args.run:
        run_queue(args.output.resolve(), args.workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
