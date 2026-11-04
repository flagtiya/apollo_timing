"""Check recorded paper experiments or rerun them with the bundled engines."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

BUNDLE = Path(__file__).resolve().parent
ARCHIVE = BUNDLE / 'recorded/prefix_archive'
CONTROLS = BUNDLE / 'recorded/control_archive'
REFERENCE = ARCHIVE / 'reference/apollo/solution.tim'
ARMS = ('baseline', 'phase_grid13', 'retained_budget', 'f1_zero', 'f1_width_broad')


def load(path):
    return json.loads(Path(path).read_text().replace('${BUNDLE}', str(BUNDLE)))


def prefix_run(n):
    if n == 23:
        return ARCHIVE / 'runs/full/n23'
    return ARCHIVE / f'runs/prefix/n{n:02d}'


def analyzer():
    path = BUNDLE / 'scripts/analyze_paths.py'
    spec = importlib.util.spec_from_file_location('paper_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


def check_recorded(output):
    manifest = load(BUNDLE / 'files.sha256.json')
    present = {p.relative_to(BUNDLE).as_posix() for p in BUNDLE.rglob('*')
               if p.is_file() and p.name != 'files.sha256.json'}
    assert set(manifest) == present, 'Bundle file membership changed'
    for name, expected in manifest.items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected, name
    audit = analyzer()
    prefix_cases, control_cases = [], []
    # Counts in the paper are accepted distinct relative paths, not completed fits.
    expected = {12: (2, 0), 13: (1, 0), 14: (2, 0), 15: (1, 1), 16: (2, 1),
                17: (1, 1), 18: (1, 1), 19: (1, 1), 20: (1, 1), 21: (1, 1),
                22: (1, 1), 23: (1, 1)}
    for n in range(12, 24):
        result = audit.analyze_solver_output(prefix_run(n), REFERENCE)
        counts = result['counts']
        assert (counts['distinct_accepted_paths'], counts['accepted_reference_agreements']) == expected[n], (n, counts)
        assert not counts['unavailable_accepted_paths'], (n, counts)
        benchmark = load(prefix_run(n) / 'benchmark_record.json')
        row = {'session_count': n, 'toa_count': 4 * (n + 1),
               'accepted_paths': counts['distinct_accepted_paths'],
               'reference_paths': counts['accepted_reference_agreements'],
               'wall_seconds': benchmark['wall_seconds']}
        prefix_cases.append(row)
        write(output / f'prefix-{n:02d}.json', result)
    historical = load(CONTROLS / 'reviewer_controls_summary.json')
    for case in historical['cases']:
        n, arm = case['session_count'], case['arm']
        result = audit.analyze_solver_output(CONTROLS / 'runs' / f'n{n}_{arm}', REFERENCE)
        counts = result['counts']
        assert counts['distinct_accepted_paths'] == case['qualified_distinct_accepted_paths'], (n, arm, counts)
        assert not counts['unavailable_accepted_paths'], (n, arm, counts)
        candidates = [c for c in result['candidates'] if c['accepted']]
        best = min(candidates, key=lambda c: c['chi2'])
        assert best['reference_comparison']['agrees_modulo_global_integer'], (n, arm)
        row = {'session_count': n, 'arm': arm, 'accepted_paths': counts['distinct_accepted_paths'],
               'reference_paths': counts['accepted_reference_agreements'], 'best_chi2': best['chi2']}
        control_cases.append(row)
        write(output / f'control-{n}-{arm}.json', result)
    transfer = load(CONTROLS / 'transferred_n16_path_to_n15/report.json')
    assert transfer['diagnostic_strict_quality_gate_passed']
    assert abs(transfer['postfit']['chi2'] - 153.60137) < 1e-4
    epoch = load(BUNDLE / 'recorded/epoch60604/epoch-comparison.json')
    assert epoch['fitted_pepoch'] == 60604 and epoch['reference_verification']['same_pulse_path']
    summary = {'prefixes': prefix_cases, 'controls': control_cases,
               'transferred_path': transfer, 'epoch60604': epoch,
               'scope': 'Fresh file-hash and recorded-path recount only; no new blind search or uniqueness proof'}
    write(output / 'summary.json', summary)
    for name, rows in [('prefixes.csv', prefix_cases), ('controls.csv', control_cases)]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Verified 12 prefix runs, 15 controls, the transferred-path diagnostic and epoch-60604 record.')
    return summary


def plot(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pint.models import get_model
    from pint.residuals import Residuals
    # Load the bundled epoch run, not the candidate used by the prefix table.
    sys.path.insert(0, str(BUNDLE / 'frozen/prefix/src'))
    from timing_accelerator.backend.pint_backend import _load_pint_toas
    model = get_model(BUNDLE / 'recorded/epoch60604/solution.par')
    toas = _load_pint_toas(BUNDLE / 'recorded/epoch60604/solution.tim', seed_model=model,
                           clock_override_path=os.environ.get('PINT_CLOCK_OVERRIDE'))
    residual = Residuals(toas, model, track_mode='use_pulse_numbers')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for ax, x, label in zip(axes, (toas.get_mjds().value, model.orbital_phase(toas, radians=False)),
                             ('MJD', 'Orbital phase'), strict=True):
        ax.errorbar(x, residual.time_resids.to_value('us'),
                    yerr=toas.get_errors().to_value('us'), fmt='.', capsize=0)
        ax.set(xlabel=label, ylabel='Timing residual (us)')
    fig.savefig(output / 'epoch60604-residuals.png', dpi=180)
    fig.savefig(output / 'epoch60604-residuals.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    rows = summary['prefixes']
    axes[0].plot([r['session_count'] for r in rows], [r['accepted_paths'] for r in rows], 'o-', label='Accepted paths')
    axes[0].plot([r['session_count'] for r in rows], [r['reference_paths'] for r in rows], 's--', label='Reference matches')
    axes[0].set(xlabel='Sessions', ylabel='Distinct paths', xticks=range(12, 24), yticks=(0, 1, 2))
    axes[0].legend()
    for index, arm in enumerate(ARMS):
        rows = [r for r in summary['controls'] if r['arm'] == arm]
        axes[1].bar([r['session_count'] + (index - 2) * .14 for r in rows],
                    [r['accepted_paths'] for r in rows], width=.14, label=arm)
    axes[1].set(xlabel='Sessions', ylabel='Accepted paths', xticks=(17, 18, 19))
    axes[1].legend(fontsize=7)
    fig.savefig(output / 'paper-path-counts.png', dpi=180)
    fig.savefig(output / 'paper-path-counts.pdf')
    plt.close(fig)


def environment(engine, ephemeris):
    env = dict(os.environ)
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    env.update(PYTHONPATH=str(BUNDLE / 'frozen' / engine / 'src'), PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    if ephemeris:
        kernel = Path(ephemeris).expanduser().resolve()
        if not kernel.is_file():
            raise FileNotFoundError(kernel)
        env['TIMING_ACCELERATOR_EPHEM_PATH'] = str(kernel)
        env['APOLLO_TIMING_EPHEM_PATH'] = str(kernel)
    return env


def execute(command, env, output, name, timeout):
    with (output / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=BUNDLE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{name} exited with {result.returncode}; inspect the log')


def rerun(args, output):
    if not args.ephem_file:
        raise ValueError('Reruns and residual plotting require --ephem-file for DE421')
    env = environment('controls' if args.task in ('controls', 'transfer') else 'prefix', args.ephem_file)
    if args.task in ('prefixes', 'epoch60604'):
        for n in (args.sessions if args.task == 'prefixes' else [23]):
            command = load(prefix_run(n) / 'benchmark_record.json')['command']
            command[0] = sys.executable
            run_dir = output / (f'n{n:02d}' if args.task == 'prefixes' else 'epoch60604')
            command[command.index('--output') + 1] = str(run_dir)
            if args.task == 'epoch60604':
                text = (ARCHIVE / 'inputs/blind_f1_initial.par').read_text()
                text = '\n'.join('PEPOCH 60604' if line.split()[:1] == ['PEPOCH'] else line for line in text.splitlines()) + '\n'
                par = output / 'initial60604.par'
                par.write_text(text)
                command[command.index('--par') + 1] = str(par)
            execute(command, env, output, run_dir.name, args.timeout)
    elif args.task == 'controls':
        command = [sys.executable, str(BUNDLE / 'scripts/run_controls.py'),
                   '--output', str(output / 'controls'), '--prepare', '--run', '--workers', str(args.workers), '--timeout', str(args.timeout)]
        execute(command, env, output, 'controls', args.timeout * 15)
        for n in (17, 18, 19):
            for arm in ARMS:
                status = load(output / 'controls/runs' / f'n{n}_{arm}' / 'runtime.json')
                if status.get('return_code') != 0 or status.get('timeout'):
                    raise RuntimeError(f'n{n}_{arm} did not complete successfully; inspect its runtime record')
    elif args.task == 'transfer':
        command = [sys.executable, str(BUNDLE / 'scripts/transfer_path.py'),
                   '--old-root', str(ARCHIVE), '--output', str(output / 'transfer'),
                   '--snapshot', str(BUNDLE / 'frozen/controls'), '--ephemeris', str(Path(args.ephem_file).resolve()), '--timeout', str(args.timeout)]
        execute(command, env, output, 'transfer', args.timeout + 30)
        report = load(output / 'transfer/report.json')
        assert report['diagnostic_strict_quality_gate_passed'], report.get('error')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('verify', 'plot', 'prefixes', 'controls', 'transfer', 'epoch60604'), default='verify')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ephem-file', type=Path)
    parser.add_argument('--sessions', nargs='+', type=int, choices=range(12, 24), default=list(range(12, 24)))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        parser.error('--workers and --timeout must be positive')
    output = args.output.expanduser().resolve()
    if output.is_relative_to(BUNDLE.parents[2]) or output.exists():
        parser.error('Use a new output directory outside the repository')
    output.mkdir(parents=True)
    if args.task in ('verify', 'plot'):
        summary = check_recorded(output)
        if args.task == 'plot':
            if not args.ephem_file:
                parser.error('--task plot requires --ephem-file')
            os.environ.update(environment('prefix', args.ephem_file))
            plot(summary, output)
    else:
        rerun(args, output)


if __name__ == '__main__':
    main()
