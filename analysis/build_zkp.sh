#!/usr/bin/env bash
set -euo pipefail

# Build Circom/snarkjs artifacts for parameter update ZKP
# Pinned baseline versions (recommend):
#   circom >= 2.1.8
#   snarkjs 0.7.x (tested with 0.7.4)
#   circomlibjs 0.0.8 (already vendored via node_modules)
# Usage:
#   bash scripts/build_zkp.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
CIRCUITS_DIR="$ROOT_DIR/circuits"
BUILD_DIR="$CIRCUITS_DIR/build"
CIRCUIT_NAME="parameter_update"

mkdir -p "$BUILD_DIR"

command -v node >/dev/null 2>&1 || { echo "[ERROR] node is required"; exit 1; }
command -v snarkjs >/dev/null 2>&1 || { echo "[ERROR] snarkjs is required (npm i -g snarkjs@0.7)"; exit 1; }
command -v circom >/dev/null 2>&1 || { echo "[ERROR] circom is required (see https://docs.circom.io)"; exit 1; }

echo "[+] Compiling circuit"
cd "$CIRCUITS_DIR"
circom "$CIRCUIT_NAME.circom" --r1cs --wasm --sym -o "$BUILD_DIR"

# Powers of Tau (phase1)
echo "[+] Powers of Tau"
ptau1="$BUILD_DIR/pot18_0000.ptau"
ptau2="$BUILD_DIR/pot18_0001.ptau"
ptauf="$BUILD_DIR/pot18_final.ptau"
if [ ! -f "$ptauf" ]; then
  snarkjs powersoftau new bn128 18 "$ptau1" -v
  snarkjs powersoftau contribute "$ptau1" "$ptau2" -e "pol-veryfl" -v
  snarkjs powersoftau prepare phase2 "$ptau2" "$ptauf" -v
fi

# Groth16 setup and keys
echo "[+] Groth16 setup"
r1cs="$BUILD_DIR/$CIRCUIT_NAME.r1cs"
zkey0="$BUILD_DIR/${CIRCUIT_NAME}_0000.zkey"
zkey1="$BUILD_DIR/${CIRCUIT_NAME}_0001.zkey"
vkey="$BUILD_DIR/${CIRCUIT_NAME}.vkey.json"

if [ ! -f "$zkey1" ]; then
  snarkjs groth16 setup "$r1cs" "$ptauf" "$zkey0" -v
  snarkjs zkey contribute "$zkey0" "$zkey1" -e "pol-veryfl" -v
fi
snarkjs zkey export verificationkey "$zkey1" "$vkey"

# Optional: export Solidity verifier
echo "[+] Exporting Solidity verifier"
sol="$ROOT_DIR/chainEnv/contracts/Groth16Verifier.sol"
snarkjs zkey export solidityverifier "$zkey1" "$sol"
echo "[+] Done. Artifacts in $BUILD_DIR"

