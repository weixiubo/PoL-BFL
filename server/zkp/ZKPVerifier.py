"""Groth16 verifier for PoL-BFL parameter-update proofs.

The verifier exposes three explicit backends: deterministic structural checks for
tests, off-chain ``snarkjs`` verification, and the deployed Solidity verifier.
Integrity-mode executions reject the structural-only backend.
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Callable, Dict, Optional

from zkp.hash import fold_weights_from_state, fold_indices

logger = logging.getLogger(__name__)


class ZKPVerifier:
    """
    Zero-Knowledge Proof Verifier for parameter updates

    Verifies ZKP proofs that demonstrate:
    1. Parameter hash consistency
    2. Data usage
    3. Reasonable parameter change

    Without accessing:
    - Actual parameter values
    - Training data
    - Data indices
    """

    def __init__(self, verification_key_path: str = None,
                 use_simulation: bool = True,
                 use_onchain: bool = False,
                 onchain_verifier: Optional[
                     Callable[[Dict, Dict], bool]
                 ] = None):
        """
        Initialize ZKP Verifier

        Args:
            verification_key_path: Path to verification key (.json)
            use_simulation: Run deterministic structural checks for isolated tests
            use_onchain: Verify with the deployed Groth16 Solidity contract
            onchain_verifier: Optional injected contract-call adapter
        """
        if use_simulation and use_onchain:
            raise ValueError("select exactly one ZKP verification backend")
        self.verification_key_path = verification_key_path
        self.use_simulation = use_simulation
        self.use_onchain = use_onchain
        self.onchain_verifier = onchain_verifier

        if os.getenv('POL_INTEGRITY', '0') == '1' and self.use_simulation:
            raise RuntimeError("POL_INTEGRITY=1 forbids ZKP simulation; set use_simulation=False and ensure snarkjs/node are installed.")

        if not self.use_simulation and not self.use_onchain:
            self._verify_dependencies()

        logger.info(f"ZKPVerifier initialized (simulation={self.use_simulation}, onchain={self.use_onchain})")

    def _verify_dependencies(self):
        """Verify that snarkjs and node are installed"""
        try:
            result = subprocess.run(['snarkjs', '--version'], capture_output=True, text=True)
            logger.info(f"snarkjs version: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError("snarkjs is required for ZKP verification")
        try:
            node = subprocess.run(['node', '--version'], capture_output=True, text=True)
            logger.info(f"node version: {node.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError("node is required for Poseidon hash binding")

    def verify_proof(self, proof: Dict, public_signals: Dict) -> bool:
        """
        Verify ZKP proof

        Args:
            proof: ZKP proof (Groth16 format)
            public_signals: Public inputs

        Returns:
            is_valid: True if proof is valid
        """
        if self.use_onchain:
            return self._verify_onchain(proof, public_signals)
        elif self.use_simulation:
            return self._simulate_verification(proof, public_signals)
        else:
            return self._verify_offchain(proof, public_signals)

    def _simulate_verification(self, proof: Dict, public_signals: Dict) -> bool:
        """
        Run deterministic structural checks for isolated unit tests.

        This backend is explicitly rejected when ``POL_INTEGRITY=1``.
        """
        # Check proof structure
        if not self._check_proof_structure(proof):
            logger.error("Invalid proof structure")
            return False

        # Check public signals
        required_fields = ['W_t_hash', 'W_t1_hash', 'data_hash', 'max_distance']
        for field in required_fields:
            if field not in public_signals:
                logger.error(f"Missing public signal: {field}")
                return False

        logger.info("Structural ZKP verification: PASS")
        logger.debug(f"  W_t_hash: {public_signals['W_t_hash'][:16]}...")
        logger.debug(f"  W_t1_hash: {public_signals['W_t1_hash'][:16]}...")

        # Check distance constraint (if available in debug mode)
        if 'actual_distance' in public_signals:
            actual = public_signals['actual_distance']
            max_dist = public_signals['max_distance'] / 1000.0
            if actual > max_dist:
                logger.warning(f"Distance {actual:.2f} exceeds max {max_dist:.2f}")
                return False

        return True

    def _verify_offchain(self, proof: Dict, public_signals: Dict) -> bool:
        """
        Verify proof off-chain using snarkjs

        Workflow:
        1. Write proof and public signals to files
        2. Call snarkjs verify
        3. Read and return result
        """
        with tempfile.TemporaryDirectory(prefix='pol_zkp_verify_') as tmp_dir:
            proof_file = os.path.join(tmp_dir, 'proof.json')
            public_file = os.path.join(tmp_dir, 'public.json')

            with open(proof_file, 'w') as f:
                json.dump(proof, f)

            public_array = [
                public_signals['W_t_hash'],
                public_signals['W_t1_hash'],
                public_signals['data_hash'],
                public_signals['max_distance']
            ]
            with open(public_file, 'w') as f:
                json.dump(public_array, f)

            # Call snarkjs verify
            result = subprocess.run([
                'snarkjs', 'groth16', 'verify',
                self.verification_key_path,
                public_file, proof_file
            ], capture_output=True, text=True)

            # Check result
            is_valid = 'OK' in result.stdout

            if is_valid:
                logger.info("ZKP verification: PASS")
            else:
                logger.warning("ZKP verification: FAIL")
                logger.debug(f"snarkjs output: {result.stdout}")

            return is_valid

    def _verify_onchain(self, proof: Dict, public_signals: Dict) -> bool:
        """Verify a proof through the deployed Groth16 Solidity contract.

        Contract unavailability and malformed inputs fail closed; an on-chain
        request is never silently downgraded to a different backend.
        """
        if not self._check_proof_structure(proof):
            logger.error("Invalid Groth16 proof structure")
            return False
        required = ("W_t_hash", "W_t1_hash", "data_hash", "max_distance")
        if not isinstance(public_signals, dict) or any(
            field not in public_signals for field in required
        ):
            logger.error("Invalid Groth16 public signals")
            return False
        try:
            verifier = self.onchain_verifier
            if verifier is None:
                from chainfl.interact import chain_proxy

                verifier = chain_proxy.verify_zkp_onchain
            verified = bool(verifier(proof, public_signals))
            logger.info("On-chain ZKP verification: %s", "PASS" if verified else "FAIL")
            return verified
        except Exception as exc:
            logger.error("On-chain ZKP verification failed: %s", exc)
            return False

    def _check_proof_structure(self, proof: Dict) -> bool:
        """Check if proof has valid Groth16 structure"""
        required_fields = ['pi_a', 'pi_b', 'pi_c', 'protocol', 'curve']
        for field in required_fields:
            if field not in proof:
                return False
        if proof['protocol'] != 'groth16':
            return False
        if proof['curve'] != 'bn128':
            return False
        return True

    def verify_proof_with_binding(self, current_ckpt: Dict, next_ckpt: Dict,
                                  data_indices: list,
                                  proof: Dict, public_signals: Dict) -> bool:
        """
        Verify proof (off-chain or on-chain) AND bind public signals to raw checkpoints
        via Poseidon fold computed from raw state_dict and indices.
        """
        ok = self.verify_proof(proof, public_signals)
        if not ok:
            return False
        try:
            wt = fold_weights_from_state(current_ckpt.get('data', {}).get('model_state', {}), n=100)
            wt1 = fold_weights_from_state(next_ckpt.get('data', {}).get('model_state', {}), n=100)
            di_hash = fold_indices((data_indices or [])[:32] + [0] * max(0, 32 - len(data_indices or [])))
            if str(public_signals.get('W_t_hash')) != str(wt):
                logger.warning('Binding failed: W_t_hash mismatch')
                return False
            if str(public_signals.get('W_t1_hash')) != str(wt1):
                logger.warning('Binding failed: W_t1_hash mismatch')
                return False
            if str(public_signals.get('data_hash')) != str(di_hash):
                logger.warning('Binding failed: data_hash mismatch')
                return False
            return True
        except Exception as e:
            logger.error(f"Binding verification error: {e}")
            return False

    def estimate_verification_time(self) -> float:
        """
        Estimate verification time in seconds

        Groth16 verification is very fast:
        - Off-chain: ~5-10ms
        - On-chain: ~250,000 Gas (~0.5-1 second block time)
        """
        if self.use_onchain:
            return 0.5  # Block time
        else:
            return 0.01  # 10ms

    def estimate_verification_cost(self, gas_price_gwei: float = 50) -> float:
        """
        Estimate on-chain verification cost in USD

        Args:
            gas_price_gwei: Gas price in Gwei

        Returns:
            cost_usd: Estimated cost in USD
        """
        if not self.use_onchain:
            return 0.0

        # Groth16 verification: ~250,000 Gas
        gas_used = 250000

        # Convert to ETH
        gas_price_eth = gas_price_gwei * 1e-9
        cost_eth = gas_used * gas_price_eth

        # Convert to USD (assume ETH = $2000)
        eth_price_usd = 2000
        cost_usd = cost_eth * eth_price_usd

        return cost_usd


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create verifier (simulation mode)
    verifier = ZKPVerifier(use_simulation=True)

    # Mock proof and public signals
    proof = {
        "pi_a": ["0x123", "0x456"],
        "pi_b": [["0x789", "0xabc"], ["0xdef", "0x012"]],
        "pi_c": ["0x345", "0x678"],
        "protocol": "groth16",
        "curve": "bn128"
    }

    public_signals = {
        "W_t_hash": "a" * 64,
        "W_t1_hash": "b" * 64,
        "data_hash": "c" * 64,
        "max_distance": 1000000,
        "actual_distance": 0.5
    }

    # Verify proof
    is_valid = verifier.verify_proof(proof, public_signals)

    print(f"\nProof verification: {'PASS' if is_valid else 'FAIL'}")
    print(f"Verification time: {verifier.estimate_verification_time():.4f}s")
    print(f"On-chain cost: ${verifier.estimate_verification_cost():.2f}")
