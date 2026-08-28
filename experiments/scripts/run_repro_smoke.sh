#!/usr/bin/env bash
set -euo pipefail

# Reproducibility smoke wrapper. Override PYTHON_BIN for the target environment,
# for example python on the 4090 server.
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" experiments/reproducibility/run_repro_smoke.py \
  --python "$PYTHON_BIN" \
  "$@"

