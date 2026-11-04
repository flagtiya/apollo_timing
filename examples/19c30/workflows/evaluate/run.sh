#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export APOLLO_TIMING_EPHEM_DIR="${APOLLO_TIMING_EPHEM_DIR:-$root/runtime/ephemerides}"
python "$root/examples/19c30/run.py" --workflow evaluate --output "${1:?Supply a fresh output directory outside the source tree}"
