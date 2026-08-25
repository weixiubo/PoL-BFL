#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
CIRCOM_BIN="${CIRCOM_BIN:-circom}"
NODE_BIN="${NODE_BIN:-node}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SNARKJS_CLI="${SNARKJS_CLI:-$ROOT_DIR/node_modules/snarkjs/cli.js}"
CIRCOMLIB_DIR="${CIRCOMLIB_DIR:-$ROOT_DIR/node_modules}"
BUILD_DIR="${ZK_REFERENCE_BUILD_DIR:-$ROOT_DIR/circuits/final/build/reference}"
PTAU="${POL_ZK_PTAU:?POL_ZK_PTAU must reference a verified power-21 or larger phase-2 transcript}"

test -s "$PTAU"
mkdir -p "$BUILD_DIR"

"$CIRCOM_BIN" "$ROOT_DIR/circuits/final/sampled_sgd_reference.circom" \
  -l "$CIRCOMLIB_DIR" --r1cs --wasm --sym --c -o "$BUILD_DIR"
"$NODE_BIN" "$ROOT_DIR/circuits/final/generate_reference_input.cjs" "$BUILD_DIR/input.json"

CPP_DIR="$BUILD_DIR/sampled_sgd_reference_cpp"
if [[ "${SKIP_CPP_WITNESS_BUILD:-0}" != "1" ]]; then
  make -C "$CPP_DIR" -j"${ZK_BUILD_JOBS:-2}"
fi

SNARKJS=("$NODE_BIN" "$SNARKJS_CLI")
R1CS="$BUILD_DIR/sampled_sgd_reference.r1cs"
ZKEY_0="$BUILD_DIR/sampled_sgd_reference_0000.zkey"
ZKEY_FINAL="$BUILD_DIR/sampled_sgd_reference_final.zkey"
VKEY="$BUILD_DIR/verification_key.json"

"${SNARKJS[@]}" powersoftau verify "$PTAU" | tee "$BUILD_DIR/powersoftau-verify.log"
"${SNARKJS[@]}" groth16 setup "$R1CS" "$PTAU" "$ZKEY_0"
openssl rand -hex 64 | "${SNARKJS[@]}" zkey contribute \
  "$ZKEY_0" "$ZKEY_FINAL" --name="PoL-BFL reference circuit contribution"
"${SNARKJS[@]}" zkey verify "$R1CS" "$PTAU" "$ZKEY_FINAL" | tee "$BUILD_DIR/zkey-verify.log"
"${SNARKJS[@]}" zkey export verificationkey "$ZKEY_FINAL" "$VKEY"
"${SNARKJS[@]}" zkey export solidityverifier "$ZKEY_FINAL" "$BUILD_DIR/Groth16Verifier.sol"
"${SNARKJS[@]}" r1cs info "$R1CS"

sha256sum "$R1CS" "$ZKEY_FINAL" "$VKEY" > "$BUILD_DIR/artifact-sha256.txt"
"$PYTHON_BIN" -m experiments.final.trust_setup create \
  --build "$BUILD_DIR" \
  --ptau "$PTAU" \
  --powersoftau-verify-log "$BUILD_DIR/powersoftau-verify.log" \
  --zkey-verify-log "$BUILD_DIR/zkey-verify.log" \
  --toolchain "$ROOT_DIR/config/toolchain.lock.json"
