#!/usr/bin/env bash
set -euo pipefail

# Build Groth16 artifacts for multiple param_size variants
# Requires: circom (>=2.0), snarkjs (>=0.7), existing ptau at circuits/ptau/pot18_final.ptau

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PTAU="$ROOT_DIR/circuits/ptau/pot18_final.ptau"

if [ ! -f "$PTAU" ]; then
  echo "Missing PTAU at $PTAU" >&2
  exit 1
fi

build_one() {
  local SIZE="$1"
  local CIR="$ROOT_DIR/circuits/param_update_${SIZE}.circom"
  local OUTDIR="$ROOT_DIR/circuits/build_p${SIZE}"
  echo "[build] param_size=${SIZE} -> $OUTDIR"
  mkdir -p "$OUTDIR"
  circom "$CIR" --r1cs --wasm --sym -o "$OUTDIR" -l "$ROOT_DIR/node_modules"
  # Normalize wasm name for prover (expects parameter_update.wasm)
  if [ -f "$OUTDIR/param_update_${SIZE}_js/param_update_${SIZE}.wasm" ]; then
    cp "$OUTDIR/param_update_${SIZE}_js/param_update_${SIZE}.wasm" \
       "$OUTDIR/param_update_${SIZE}_js/parameter_update.wasm"
  fi
  snarkjs groth16 setup \
    "$OUTDIR/param_update_${SIZE}.r1cs" \
    "$PTAU" \
    "$OUTDIR/param_update_${SIZE}_0000.zkey"
  # For benchmarking, use the initial zkey directly (no interactive contribute)
  snarkjs zkey export verificationkey \
    "$OUTDIR/param_update_${SIZE}_0000.zkey" \
    "$OUTDIR/param_update_${SIZE}.vkey.json"
}

build_one 50
build_one 100
build_one 150
build_one 200

echo "All variants built."

