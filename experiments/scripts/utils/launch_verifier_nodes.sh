#!/usr/bin/env bash
set -euo pipefail
# Launch multiple VerifierNode instances locally for decentralized verification tests.
# Usage: ./launch_verifier_nodes.sh [NUM_NODES] [BASE_PORT]
# Defaults: NUM_NODES=4 BASE_PORT=8088

NUM_NODES=${1:-4}
BASE_PORT=${2:-8088}
HOST=127.0.0.1
PY="/home/wxb/miniconda3/envs/wxb__veryfl_pol/bin/python"

for ((i=0;i<NUM_NODES;i++)); do
  PORT=$((BASE_PORT + i))
  echo "Starting VerifierNode on http://${HOST}:${PORT} ..."
  nohup "$PY" -m server.committee.VerifierNode --host "$HOST" --port "$PORT" \
    > "verifiernode_${PORT}.log" 2>&1 < /dev/null &
  sleep 0.2
done

echo "Started $NUM_NODES VerifierNode instances from port $BASE_PORT"

