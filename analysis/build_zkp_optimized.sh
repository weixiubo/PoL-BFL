#!/bin/bash
# Build Optimized ZKP Circuit for PoL-FL
#
# This script compiles the optimized parameter_update_optimized.circom circuit
# and generates the necessary proving/verification keys.
#
# Optimizations:
# 1. Merkle tree hashing (reduces constraints by 33%)
# 2. Optimized L2 distance calculation
# 3. Full security maintained (no sampling)
#
# Expected performance:
# - Constraints: ~600 (vs 900 in original)
# - Proof time: 0.6-1.2s (vs 1-2s in original)
# - Proof size: 256 bytes (same as original)

set -e

echo "=========================================="
echo "Building Optimized ZKP Circuit"
echo "=========================================="

# Configuration
CIRCUIT_NAME="parameter_update_optimized"
CIRCUITS_DIR="circuits"
BUILD_DIR="$CIRCUITS_DIR/build"

# Create build directory
mkdir -p "$BUILD_DIR"

echo "[+] Compiling optimized circuit"
cd "$CIRCUITS_DIR"
circom "$CIRCUIT_NAME.circom" --r1cs --wasm --sym -o "$BUILD_DIR"

# Powers of Tau (phase1) - reuse from original if exists
echo "[+] Powers of Tau"
ptau1="$BUILD_DIR/pot18_0000.ptau"
ptau2="$BUILD_DIR/pot18_0001.ptau"
ptauf="$BUILD_DIR/pot18_final.ptau"

if [ ! -f "$ptauf" ]; then
  echo "  Generating new Powers of Tau..."
  snarkjs powersoftau new bn128 18 "$ptau1" -v
  snarkjs powersoftau contribute "$ptau1" "$ptau2" -e "pol-veryfl-optimized" -v
  snarkjs powersoftau prepare phase2 "$ptau2" "$ptauf" -v
else
  echo "  Reusing existing Powers of Tau: $ptauf"
fi

# Setup (phase2)
echo "[+] Groth16 setup"
r1cs="$BUILD_DIR/${CIRCUIT_NAME}.r1cs"
zkey0="$BUILD_DIR/${CIRCUIT_NAME}_0000.zkey"
zkey1="$BUILD_DIR/${CIRCUIT_NAME}_0001.zkey"

snarkjs groth16 setup "$r1cs" "$ptauf" "$zkey0" -v
snarkjs zkey contribute "$zkey0" "$zkey1" -e "pol-contribution-optimized" -v

# Export verification key
echo "[+] Exporting verification key"
vkey="$BUILD_DIR/verification_key_optimized.json"
snarkjs zkey export verificationkey "$zkey1" "$vkey"

# Export Solidity verifier
echo "[+] Exporting Solidity verifier"
sol_verifier="../chainEnv/contracts/Groth16VerifierOptimized.sol"
snarkjs zkey export solidityverifier "$zkey1" "$sol_verifier"

# Print circuit info
echo ""
echo "=========================================="
echo "Circuit Information"
echo "=========================================="
snarkjs r1cs info "$r1cs"

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo "Circuit: $CIRCUIT_NAME"
echo "R1CS: $r1cs"
echo "Proving key: $zkey1"
echo "Verification key: $vkey"
echo "Solidity verifier: $sol_verifier"
echo ""
echo "Next steps:"
echo "1. Test the circuit: python test_zkp_optimized.py"
echo "2. Benchmark performance: python benchmark_zkp_optimized.py"
echo "3. Deploy Solidity verifier to blockchain"
echo "=========================================="

