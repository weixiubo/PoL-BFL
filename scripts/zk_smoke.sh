#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CIRCOM_BIN="${CIRCOM_BIN:-circom}"
NODE_BIN="${NODE_BIN:-node}"
SNARKJS_CLI="${SNARKJS_CLI:-$ROOT_DIR/node_modules/snarkjs/cli.js}"
CIRCOMLIB_DIR="${CIRCOMLIB_DIR:-$ROOT_DIR/node_modules}"
BUILD_DIR="${ZK_SMOKE_BUILD_DIR:-$ROOT_DIR/circuits/final/build/smoke}"

mkdir -p "$BUILD_DIR"

"$CIRCOM_BIN" "$ROOT_DIR/circuits/final/sampled_sgd_smoke.circom" \
  -l "$CIRCOMLIB_DIR" --r1cs --wasm --sym -o "$BUILD_DIR"

"$NODE_BIN" "$ROOT_DIR/circuits/final/generate_smoke_input.cjs" "$BUILD_DIR/input.json"

SNARKJS=("$NODE_BIN" "$SNARKJS_CLI")
PTAU_0="$BUILD_DIR/pot15_0000.ptau"
PTAU_1="$BUILD_DIR/pot15_0001.ptau"
PTAU_FINAL="$BUILD_DIR/pot15_final.ptau"

if [[ ! -s "$PTAU_FINAL" ]]; then
  "${SNARKJS[@]}" powersoftau new bn128 15 "$PTAU_0"
  openssl rand -hex 64 | "${SNARKJS[@]}" powersoftau contribute \
    "$PTAU_0" "$PTAU_1" --name="PoL-BFL smoke contribution"
  "${SNARKJS[@]}" powersoftau prepare phase2 "$PTAU_1" "$PTAU_FINAL"
fi

R1CS="$BUILD_DIR/sampled_sgd_smoke.r1cs"
WASM="$BUILD_DIR/sampled_sgd_smoke_js/sampled_sgd_smoke.wasm"
ZKEY_0="$BUILD_DIR/sampled_sgd_smoke_0000.zkey"
ZKEY_FINAL="$BUILD_DIR/sampled_sgd_smoke_final.zkey"
VKEY="$BUILD_DIR/verification_key.json"

"${SNARKJS[@]}" groth16 setup "$R1CS" "$PTAU_FINAL" "$ZKEY_0"
openssl rand -hex 64 | "${SNARKJS[@]}" zkey contribute \
  "$ZKEY_0" "$ZKEY_FINAL" --name="PoL-BFL circuit contribution"
"${SNARKJS[@]}" zkey export verificationkey "$ZKEY_FINAL" "$VKEY"
"${SNARKJS[@]}" groth16 fullprove "$BUILD_DIR/input.json" "$WASM" "$ZKEY_FINAL" \
  "$BUILD_DIR/proof.json" "$BUILD_DIR/public.json"
"${SNARKJS[@]}" groth16 verify "$VKEY" "$BUILD_DIR/public.json" "$BUILD_DIR/proof.json"
"${SNARKJS[@]}" zkey export solidityverifier "$ZKEY_FINAL" "$BUILD_DIR/Groth16Verifier.sol"
"${SNARKJS[@]}" r1cs info "$R1CS"
