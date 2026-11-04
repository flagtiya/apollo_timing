# APOLLO-Timing 0.8.3

English | [中文](README.zh-CN.md)

APOLLO-Timing provides isolated and binary pulsar timing through command-line and Python interfaces.

## Features

| Feature | Use |
| --- | --- |
| Phase connection and parameter fitting | Search connections from PAR/TIM files, fit parameters according to input fit flags, and retain alternatives |
| Model evaluation | Check a supplied model and pulse path without refitting |
| Timing updates | Add new TOAs to an existing timing task |
| Orbit initialization and search | Complete incomplete binary inputs or search a specified period range |
| Parameter and model comparison | Compare additional parameters and binary models |
| Parameter limits | Set hard parameter intervals and a joint sky-position radius |
| Single-star glitch fitting | Explicitly search for one permanent frequency step, optionally with a frequency-derivative step |
| Observation planning | Recommend future observing times using retained candidates |
| Foldbank | Generate ELL1 parameter banks and PRESTO folding commands |

Supported binary models: `BT`, `DD`, `DDS`, `DDGR`, `DDH`, `DDK`, `ELL1`, `ELL1H`, `ELL1k`, and `BT_piecewise`.

## Installation

Use Linux or WSL with Python 3.11 or later. Run from the source root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
apollo-timing --version
```

Ephemerides, observatory clocks and PRESTO are obtained separately. Replace the PAR, TIM, FITS and resource paths below with your own files. Use a different output directory for each new run.

## Bundled Examples

Run these commands from the repository root after installation. Set the ephemeris directory to your own resource location. Examples use DE421 unless noted otherwise.

```bash
export APOLLO_TIMING_EPHEM_DIR=/path/to/ephemerides
```

### 19C30: Binary Timing

The PSR J1912-1215 example includes 96 TOAs. Check the inputs, evaluate the supplied reference path, or run a full connection search:

```bash
python examples/19c30/run.py --verify-only \
  --output "$HOME/apollo-runs/19c30-inputs"

python examples/19c30/run.py --workflow evaluate \
  --output "$HOME/apollo-runs/19c30-evaluate"

python examples/19c30/run.py --workflow solve \
  --output "$HOME/apollo-runs/19c30-solve"
```

To try an update from the supplied reference model, or automatic system comparison:

```bash
python examples/19c30/run.py --workflow update \
  --output "$HOME/apollo-runs/19c30-update"

python examples/19c30/run.py --workflow auto-discovery \
  --output "$HOME/apollo-runs/19c30-auto"
```

Input checks and fixed-path evaluation are not blind searches. The solve and discovery commands can take substantially longer.

### 19C30: Paper Experiments

The [paper experiment bundle](examples/19c30/paper/README.md) includes the original 12–23-session prefixes, all 15 stability controls, the 16-to-15 fixed-path diagnostic, the independent PEPOCH-60604 run, and the retained historical engines. Check the recorded results without repeating the long searches:

```bash
python examples/19c30/paper/run.py --task verify \
  --output "$HOME/apollo-runs/paper-verify"
```

Rerun the prefix matrix, stability controls, transferred-path diagnostic or full-data epoch check using a local DE421 kernel:

```bash
python examples/19c30/paper/run.py --task prefixes \
  --ephem-file /path/to/de421.bsp --output "$HOME/apollo-runs/paper-prefixes"

python examples/19c30/paper/run.py --task controls --workers 1 \
  --ephem-file /path/to/de421.bsp --output "$HOME/apollo-runs/paper-controls"

python examples/19c30/paper/run.py --task transfer \
  --ephem-file /path/to/de421.bsp --output "$HOME/apollo-runs/paper-transfer"

python examples/19c30/paper/run.py --task epoch60604 \
  --ephem-file /path/to/de421.bsp --output "$HOME/apollo-runs/paper-epoch60604"
```

Plot commands and resource requirements are in the bundle guide. The historical paper records are separate from runs with the current 0.8.3 solver.

### J1928: Initialize An Incomplete Binary Model

This example requires DE440.

```bash
apollo-bootstrap-ell1 \
  --par examples/j1928_coarse_ell1/coarse.par \
  --tim examples/j1928_coarse_ell1/observations.tim \
  --output "$HOME/apollo-runs/j1928-bootstrap"

apollo-timing \
  --par examples/j1928_coarse_ell1/coarse.par \
  --tim examples/j1928_coarse_ell1/observations.tim \
  --output "$HOME/apollo-runs/j1928-solve"
```

### 47 Tuc AA: Single-Star Fixture

```bash
apollo-timing --source-type single \
  --par examples/isolated_47tucaa/initial.par \
  --tim examples/isolated_47tucaa/data.tim \
  --output "$HOME/apollo-runs/47tucaa" --no-observation-plan

python examples/isolated_47tucaa/run.py --retained-paths 32 \
  --output "$HOME/apollo-runs/47tucaa-path-check"
```

This synthetic fixture checks runtime and connection ambiguity. Inspect the path-check report; a small residual alone does not establish reference-path recovery.

### Simulated Sources: Parameters And Connections

List the 95 parameter cases, run a single null/injection comparison, or run the parameter suite:

```bash
python examples/simulated_pulsars/run.py --list

python examples/simulated_pulsars/run.py --case single-f2 \
  --output "$HOME/apollo-runs/sim-f2"

python examples/simulated_pulsars/run.py --workers 4 \
  --output "$HOME/apollo-runs/sim-parameters"
```

Separate examples cover isolated connection recovery, eccentric and Shapiro model comparisons, and observing plans with ambiguous candidates:

```bash
bash examples/simulated_pulsars/isolated-connection/run.sh \
  "$HOME/apollo-runs/sim-connection"

bash examples/simulated_pulsars/eccentric-model/run.sh \
  "$HOME/apollo-runs/sim-eccentric"

bash examples/simulated_pulsars/shapiro-model/run.sh \
  "$HOME/apollo-runs/sim-shapiro"

bash examples/simulated_pulsars/ambiguity-planning/run.sh \
  "$HOME/apollo-runs/sim-planning"
```

Check parameter intervals and the joint sky-position limit:

```bash
bash examples/simulated_pulsars/parameter-ranges/run.sh \
  "$HOME/apollo-runs/sim-ranges" 4

bash examples/simulated_pulsars/sky-position/run.sh \
  "$HOME/apollo-runs/sim-sky"
```

Parameter and model comparisons use known pulse paths; the isolated-connection example searches without input pulse numbers. All generated results go to the specified external directories.

The separate noiseless parameter checks can be run with:

```bash
python examples/parameter_exploration/run.py --case single-F2 \
  --output "$HOME/apollo-runs/parameter-check"
```

## Commands For Your Own Data

### Automatic Isolated Or Binary Dispatch

```bash
apollo-timing --par source.par --tim observations.tim \
  --output "$HOME/apollo-runs/source"
```

### Isolated And Binary Solving

```bash
apollo-timing --source-type single \
  --par single.par --tim observations.tim \
  --output "$HOME/apollo-runs/single"

apollo-timing --source-type binary \
  --par binary.par --tim observations.tim \
  --output "$HOME/apollo-runs/binary"
```

### Evaluate An Existing Result

```bash
apollo-timing --task evaluate \
  --par solution.par --tim solution.tim \
  --output "$HOME/apollo-runs/evaluate"
```

### Add New Observations

```bash
apollo-timing --task update \
  --par previous.par --tim historical.tim --update-tim new.tim \
  --output "$HOME/apollo-runs/update"
```

### Resume An Unfinished Run

Add `--resume` to the original command only when inputs, configuration and resources are unchanged:

```bash
apollo-timing --par source.par --tim observations.tim \
  --output "$HOME/apollo-runs/source" --resume
```

### Specify Clocks And Ephemerides

```bash
apollo-timing --par source.par --tim observations.tim \
  --clock-dir /path/to/clocks --ephem-file /path/to/de421.bsp \
  --output "$HOME/apollo-runs/local-resources"
```

The ephemeris must match the PAR file's `EPHEM`. Alternatively, use `--ephem-dir /path/to/ephemerides` to specify a kernel directory.

## Orbit And Parameter Examples

### Search For An Unknown Orbit

```bash
apollo-timing --source-type binary \
  --par source.par --tim observations.tim \
  --discovery-period-min-days 1 --discovery-period-max-days 100 \
  --output "$HOME/apollo-runs/orbit-search"
```

### Set Parameter And Position Limits

Adjust the example F0 interval to your input frequency. Setting a bound does not release a fixed parameter.

```bash
apollo-timing --source-type binary \
  --par binary.par --tim observations.tim \
  --parameter-bound F0 99.999999 100.000001 --sky-radius-arcmin 1.5 \
  --output "$HOME/apollo-runs/bounded"
```

### Compare Additional Parameters

```bash
apollo-timing --source-type binary \
  --par binary.par --tim observations.tim \
  --explore-parameter PBDOT --exploration-bound PBDOT -1e-10 1e-10 \
  --output "$HOME/apollo-runs/pbdot"
```

### Compare Binary Models

```bash
apollo-timing --binary-model-selection --binary-models ELL1 DD ELL1H \
  --par binary.par --tim observations.tim \
  --output "$HOME/apollo-runs/model-comparison"
```

### Search For A Single-Star Glitch

Epoch bounds use TDB MJD. Adjust them to your observation span.

```bash
apollo-timing --source-type single --glitch-search \
  --par single.par --tim observations.tim \
  --glitch-epoch-range 58000 59000 --glitch-epoch-samples 32 \
  --output "$HOME/apollo-runs/glitch" --no-observation-plan
```

## Observation Planning

Generate observing recommendations from an existing run directory:

```bash
apollo-observe --run-output "$HOME/apollo-runs/source" \
  --output "$HOME/apollo-runs/observation-plan" --horizon-days 180 --top 10
```

## Foldbank

Supply a coarse ELL1 model, training TOAs and corresponding SEARCH-mode PSRFITS to generate candidate parameter files and folding scripts:

```bash
apollo-foldbank --par coarse-ell1.par --tim training.tim \
  --search epoch1.fits --search epoch2.fits \
  --validation-search heldout.fits \
  --pb-min 6.25 --pb-max 6.37 --toa-origin external_discovery \
  --output "$HOME/apollo-runs/foldbank"
```

Set the period range and TOA origin to match your actual inputs. By default, this generates scripts only. With PRESTO installed, add `--run-presto` to execute folding.

## Outputs

| File | Contents |
| --- | --- |
| `run_summary.json` | Run status and result summary |
| `certification.json` | Candidate acceptance checks and diagnostics |
| `solution.par`, `solution.tim` | Saved model and pulse path |
| `candidate_portfolio.json` | Retained alternative candidates |
| `residuals.csv`, `residuals.png` | Residual data and plot |
| `observation_plan/` | Observing recommendations, when available |

Check the JSON status first. The presence of `solution.par` does not mean that the run passed acceptance checks.

## Python API

```python
from pathlib import Path
from apollo_timing.pipeline import solve_apollo_timing

result = solve_apollo_timing(
    par_path=Path("binary.par"),
    tim_path=Path("observations.tim"),
    output_dir=Path.home() / "apollo-runs" / "python-api",
    source_type="binary",
)
print(result.accepted, result.solution_par)
```

## Command Help

```bash
apollo-timing --help
apollo-bootstrap-binary --help
apollo-bootstrap-ell1 --help
apollo-observe --help
apollo-foldbank --help
```

## License

[MIT](LICENSE)
