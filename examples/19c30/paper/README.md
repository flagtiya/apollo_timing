# 19C30 Paper Experiments

English | [中文](README.zh-CN.md)

This directory contains the timing experiments reported for PSR J1912-1215:

| Experiment | Included material |
| --- | --- |
| Chronological prefixes | The 12 original runs using 12–23 sessions, inputs, candidate paths, fit checks and runtimes |
| Search stability | All 15 runs on 17–19 sessions: baseline, 13-point phase grid, larger retained budget, zero initial F1 and broader F1 range |
| Transferred alternative path | The 16-session alternative restricted to 15 sessions, with its fixed-path refit and checks |
| PEPOCH 60604 | Independent full-data run, fitted PAR/TIM and comparison with the original pulse path |

Ready-to-view outputs: [prefix table](tables/prefixes.csv), [control table](tables/controls.csv), [path-count figure](figures/paper-path-counts.png), and [timing residuals](figures/epoch60604-residuals.png). The tables and figures are generated from the recorded experiments, not new blind runs.

`recorded/` contains the historical results. `frozen/prefix/` contains the historical engine retained with the epoch-60604 run. A complete source-file match to the original prefix runs has not been established. `frozen/controls/` contains the engine saved for the stability controls. These are separate from the current `apollo_timing` package. Full numerical identity across dependency versions is not guaranteed; the complete original Python environments were not preserved here.

## Check Recorded Results

From the repository root, after installing the project's dependencies:

```bash
python examples/19c30/paper/run.py --task verify \
  --output "$HOME/apollo-runs/paper-verify"
```

This recounts distinct accepted pulse paths, compares them with the reference and writes summary JSON/CSV. It does not run a new blind search. Input pulse numbers are compared only after inference in the recorded blind experiments.

## Generate Plots

Supply your local DE421 kernel and required observatory clocks:

```bash
export PINT_CLOCK_OVERRIDE=/path/to/clocks

python examples/19c30/paper/run.py --task plot \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-plots"
```

This regenerates prefix/control path-count plots and the epoch-60604 residuals versus MJD and orbital phase as PNG/PDF files. Plot styling is provided by this public script, not a copy of the manuscript layout. Polarization, preliminary newfitorbit fitting and companion-population plots are not APOLLO timing experiments and are not included.

## Rerun The Experiments

Run all 12 chronological prefixes, or select a subset with `--sessions`:

```bash
python examples/19c30/paper/run.py --task prefixes \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-prefixes"

python examples/19c30/paper/run.py --task prefixes --sessions 15 16 \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-prefixes-15-16"
```

Run all 15 stability controls:

```bash
python examples/19c30/paper/run.py --task controls --workers 1 \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-controls"
```

Repeat the 16-to-15 fixed-path diagnostic using the recorded alternative:

```bash
python examples/19c30/paper/run.py --task transfer \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-transfer"
```

Repeat the full-data blind run at PEPOCH 60604:

```bash
python examples/19c30/paper/run.py --task epoch60604 \
  --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/paper-epoch60604"
```

Each invocation requires a new output directory outside the repository. The default timeout is 3600 seconds per blind run; adjust it with `--timeout`. The recorded runtimes are not predictions for your machine.

## Reading The Records

The paper's prefix counts describe accepted distinct paths, not all completed numerical fits. The second 19-session fit was rejected. The transferred 15-session fit is a fixed-path diagnostic and does not change the original blind-search count. The controls did not establish complete search coverage; likelihood ratios are not integrated evidence ratios or path probabilities.

Machine names, user names, environment paths, raw logs and internal review drafts were omitted. `${BUNDLE}` marks paths resolved relative to this directory by the scripts. `${EXTERNAL_RESOURCE}` marks an omitted external resource or obsolete build location. Recorded checkpoints and requests are evidence, not resumable caches after this sanitization.

The full long-running matrix has not been rerun as part of preparing this public bundle. Fresh verification results belong in the external output directory, not among the historical records.
