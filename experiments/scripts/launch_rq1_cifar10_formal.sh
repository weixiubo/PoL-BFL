#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SESSION="${SESSION:-polbfl_20260518_rq1_cifar10_formal}"
OUTPUT_DIR="$CODE_ROOT/experiments/results/reproduction/formal/rq1_main_security"
LOG_PATH="$OUTPUT_DIR/cifar10_formal_launcher.log"
GPUS_ARG="${GPUS:-0,1}"
PARALLEL_ARG="${PARALLEL:-2}"
RUN_FILTERS_ARG="${RUN_FILTERS:-cifar10,pol_bfl}"
NUM_WORKERS_ARG="${NUM_WORKERS_OVERRIDE:-0}"
CUBLAS_ARG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
VERIFIER_PORT_BASE_ARG="${VERIFIER_PORT_BASE:-19088}"
PREFLIGHT_ARG="${PREFLIGHT:-1}"
VALIDATE_AFTER_JOB_ARG="${VALIDATE_AFTER_JOB:-1}"
POL_MEMORY_CHECKPOINT_LIMIT_ARG="${POL_MEMORY_CHECKPOINT_LIMIT:-2}"
POL_COMPACT_REMOTE_RESPONSE_ARG="${POL_COMPACT_REMOTE_RESPONSE:-1}"
POL_ENABLE_PARALLEL_CLIENT_TRAINING_ARG="${POL_ENABLE_PARALLEL_CLIENT_TRAINING:-1}"
POL_CLIENT_TRAIN_WORKERS_PER_DEVICE_ARG="${POL_CLIENT_TRAIN_WORKERS_PER_DEVICE:-2}"
POL_CLIENT_TRAIN_MAX_WORKERS_ARG="${POL_CLIENT_TRAIN_MAX_WORKERS:-0}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/_launcher"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists=$SESSION"
  exit 0
fi

COMMAND_FILE="$OUTPUT_DIR/_launcher/${SESSION}_command.sh"
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "CODE_ROOT=%q\n" "$CODE_ROOT"
  printf "PYTHON_BIN=%q\n" "$PYTHON_BIN"
  printf "OUTPUT_DIR=%q\n" "$OUTPUT_DIR"
  printf "SESSION_NAME=%q\n" "$SESSION"
  printf "GPUS_ARG=%q\n" "$GPUS_ARG"
  printf "PARALLEL_ARG=%q\n" "$PARALLEL_ARG"
  printf "RUN_FILTERS_ARG=%q\n" "$RUN_FILTERS_ARG"
  printf "NUM_WORKERS_ARG=%q\n" "$NUM_WORKERS_ARG"
  printf "CUBLAS_ARG=%q\n" "$CUBLAS_ARG"
  printf "VERIFIER_PORT_BASE_ARG=%q\n" "$VERIFIER_PORT_BASE_ARG"
  printf "PREFLIGHT_ARG=%q\n" "$PREFLIGHT_ARG"
  printf "VALIDATE_AFTER_JOB_ARG=%q\n" "$VALIDATE_AFTER_JOB_ARG"
  printf "POL_MEMORY_CHECKPOINT_LIMIT_ARG=%q\n" "$POL_MEMORY_CHECKPOINT_LIMIT_ARG"
  printf "POL_COMPACT_REMOTE_RESPONSE_ARG=%q\n" "$POL_COMPACT_REMOTE_RESPONSE_ARG"
  printf "POL_ENABLE_PARALLEL_CLIENT_TRAINING_ARG=%q\n" "$POL_ENABLE_PARALLEL_CLIENT_TRAINING_ARG"
  printf "POL_CLIENT_TRAIN_WORKERS_PER_DEVICE_ARG=%q\n" "$POL_CLIENT_TRAIN_WORKERS_PER_DEVICE_ARG"
  printf "POL_CLIENT_TRAIN_MAX_WORKERS_ARG=%q\n" "$POL_CLIENT_TRAIN_MAX_WORKERS_ARG"
  cat <<'SCRIPT'

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export NUM_WORKERS_OVERRIDE="$NUM_WORKERS_ARG"
export CUBLAS_WORKSPACE_CONFIG="$CUBLAS_ARG"
export POL_ENABLE_SYBIL_DETECTOR=1
export POL_INTEGRITY=1
export POL_MEMORY_CHECKPOINT_LIMIT="$POL_MEMORY_CHECKPOINT_LIMIT_ARG"
export POL_COMPACT_REMOTE_RESPONSE="$POL_COMPACT_REMOTE_RESPONSE_ARG"
export POL_ENABLE_PARALLEL_CLIENT_TRAINING="$POL_ENABLE_PARALLEL_CLIENT_TRAINING_ARG"
export POL_CLIENT_TRAIN_WORKERS_PER_DEVICE="$POL_CLIENT_TRAIN_WORKERS_PER_DEVICE_ARG"
export POL_CLIENT_TRAIN_MAX_WORKERS="$POL_CLIENT_TRAIN_MAX_WORKERS_ARG"

IFS=',' read -r -a filters <<< "$RUN_FILTERS_ARG"
only_args=()
for filter in "${filters[@]}"; do
  filter="${filter//[[:space:]]/}"
  if [ -n "$filter" ]; then
    only_args+=(--only "$filter")
  fi
done

validation_args=()
if [ "$VALIDATE_AFTER_JOB_ARG" = "1" ]; then
  validation_args+=(
    --validate-after-job
    --validation-results-root "$CODE_ROOT/experiments/results/reproduction/formal"
    --validation-output-root "$CODE_ROOT/experiments/results/reproduction/formal/_validation_gates"
    --validation-tolerance-ma 1.0
    --validation-tolerance-detection 1.0
    --validation-min-rounds-rq1 200
    --validation-min-clients-rq1 50
    --validation-min-clients-per-round-rq1 50
    --validation-min-local-epochs-rq1 5
  )
fi

if [ "$PREFLIGHT_ARG" = "1" ]; then
  "$PYTHON_BIN" -m pytest tests/test_reproducibility_tools.py tests/test_deterministic_replay_data.py
  "$PYTHON_BIN" experiments/reproducibility/audit_reproduction_coverage.py \
    --output-dir "$OUTPUT_DIR/_preflight/audit_${SESSION_NAME}"
  "$PYTHON_BIN" experiments/reproducibility/run_paper_config.py \
    --config-file experiments/reproducibility/configs/paper/rq1_main_security_formal.json \
    --python "$PYTHON_BIN" \
    --gpus "$GPUS_ARG" \
    --parallel "$PARALLEL_ARG" \
    "${only_args[@]}" \
    --resume \
    --dry-run
fi

exec "$PYTHON_BIN" experiments/reproducibility/run_paper_config.py \
  --config-file experiments/reproducibility/configs/paper/rq1_main_security_formal.json \
  --python "$PYTHON_BIN" \
  --gpus "$GPUS_ARG" \
  --parallel "$PARALLEL_ARG" \
  "${only_args[@]}" \
  --resume \
  --start-verifiers \
  --verifier-port-base "$VERIFIER_PORT_BASE_ARG" \
  "${validation_args[@]}"
SCRIPT
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

tmux new-session -d -s "$SESSION" "bash '$COMMAND_FILE' > '$LOG_PATH' 2>&1"

echo "session_started=$SESSION"
echo "log=$LOG_PATH"
echo "command=$COMMAND_FILE"
