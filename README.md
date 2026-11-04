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

## Timing Examples

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
