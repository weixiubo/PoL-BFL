#!/bin/bash
# Run RQ4 Incentive experiment with multiple seeds and save per-seed result files.
# Prefer explicit python from wxb__veryfl_pol.
set -euo pipefail

SEEDS=(42 43 44)

# Determine python binary (env PYTHON > system python)
if [ -n "${PYTHON:-}" ]; then
  PYBIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYBIN="$(command -v python)"
else
  echo "[ERROR] python not found"; exit 1
fi

# Read canonical dirs from experiment_config; fallback to local if import fails
readarray -t DIRS < <("${PYBIN}" - <<'PY'
from pathlib import Path
import sys
try:
    sys.path.insert(0, str((Path(__file__).resolve().parent).resolve()))
    from experiment_config import OUTPUT_CONFIG
    print(OUTPUT_CONFIG['results_dir'])
    print(OUTPUT_CONFIG['log_dir'])
except Exception:
    print(str(Path(__file__).resolve().parent / 'results'))
    print(str(Path(__file__).resolve().parent.parent / 'log'))
PY
)
RESULTS_DIR="${DIRS[0]}"
LOG_ROOT="${DIRS[1]}"

OUTDIR="${RESULTS_DIR}/rq4_incentive"
LOGDIR="${LOG_ROOT}/rq4_multi_seed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR" "$LOGDIR"

echo "Using PYBIN=${PYBIN}"

GPU=${GPU:-1}
export CUDA_VISIBLE_DEVICES=${GPU}


# Ensure deterministic cublas workspace for reproducibility
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for S in "${SEEDS[@]}"; do
  echo "==== Running RQ4 with SEED=${S} on GPU=${GPU} ===="
  export SEED=${S}
  nohup "${PYBIN}" run_rq4_incentive.py > "${LOGDIR}/rq4_seed_${S}.log" 2>&1
  # Save per-seed copy
  if [[ -f "${OUTDIR}/rq4_results.json" ]]; then
    cp -f "${OUTDIR}/rq4_results.json" "${OUTDIR}/rq4_results_seed${S}.json"
    echo "Saved: ${OUTDIR}/rq4_results_seed${S}.json"
  else
    echo "[ERROR] Expected result file not found: ${OUTDIR}/rq4_results.json"
  fi
  sleep 2
done

echo "All seeds finished. Logs at ${LOGDIR}."

