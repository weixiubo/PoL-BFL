#!/usr/bin/env bash
set -euo pipefail
TAG=${1:-}
if [ -z "${TAG:-}" ]; then
  echo "Usage: $0 <TAG>" >&2; exit 1
fi

LOGP="experiments/logs/last_${TAG}.logpath"
PIDP="experiments/logs/last_${TAG}.pidpath"
[ -f "$LOGP" ] || { echo "No log pointer: $LOGP" >&2; exit 2; }
[ -f "$PIDP" ] || { echo "No pid pointer: $PIDP" >&2; exit 3; }
LOG=$(cat "$LOGP" || true)
PIDF=$(cat "$PIDP" || true)
PID=$(cat "$PIDF" 2>/dev/null || true)

echo "== POINTERS =="
echo "LOG=$LOG"
echo "PIDF=$PIDF PID=$PID"

if [ -n "$PID" ] && [ -d "/proc/$PID" ]; then
  echo "\n== PROCESS =="
  ps -o pid,pcpu,pmem,etime,cmd -p "$PID" || true
else
  echo "Process not running"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "\n== GPU SUMMARY =="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | sed -n '1,2p'
fi

echo "\n== LOG TAIL (last 50 lines) =="
[ -f "$LOG" ] && tail -n 50 "$LOG" || echo "Log not found: $LOG"

