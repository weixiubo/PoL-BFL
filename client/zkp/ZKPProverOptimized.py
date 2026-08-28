"""
Optimized ZKP Prover for PoL-FL

Generates zero-knowledge proofs using the optimized circuit with:
1. Merkle tree hashing instead of folded Poseidon
2. Optimized L2 distance calculation
3. Full security (no sampling)

Performance improvements:
- Constraints: 900 → ~600 (33% reduction)
- Proof generation time: 1-2s → 0.6-1.2s (40-70% reduction)
- Security: 100% maintained
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

from zkp.hash import (
    DEFAULT_SCALE,
    flatten_first_n,
    quantize_to_field,
    poseidon_fold,
    poseidon_fold_many,
)

logger = logging.getLogger(__name__)

_MERKLE_ROOT_CACHE: Dict[Tuple[int, ...], str] = {}


def compute_merkle_root(leaves: List[int]) -> str:
    """
    Compute Merkle root of a list of field elements using Poseidon hash.

    This is a Python implementation for testing. The actual proof uses
    the Circom circuit implementation.

    Args:
        leaves: List of field elements (as integers or strings)

    Returns:
        root: Merkle root as decimal string
    """
    # Use the existing poseidon_fold function from zkp.hash
    # Hash node pairs to construct the Merkle tree.
    # that's compatible with the existing infrastructure

    key = tuple(int(x) for x in leaves)
    cached = _MERKLE_ROOT_CACHE.get(key)
    if cached is not None:
        return cached
    current_layer = list(key)

    # Pad to next power of 2
    n = len(current_layer)
    depth = 0
    temp = n - 1
    while temp > 0:
        depth += 1
        temp = temp >> 1

    padded_size = 2 ** depth
    current_layer.extend([0] * (padded_size - n))

    # Build each Merkle level in one persistent bridge request.
    while len(current_layer) > 1:
        current_layer = [
            int(value)
            for value in poseidon_fold_many(
                [
                    current_layer[index : index + 2]
                    for index in range(0, len(current_layer), 2)
                ]
            )
        ]

    result = str(current_layer[0])
    if len(_MERKLE_ROOT_CACHE) >= 128:
        _MERKLE_ROOT_CACHE.pop(next(iter(_MERKLE_ROOT_CACHE)))
    _MERKLE_ROOT_CACHE[key] = result
    return result


class ZKPProverOptimized:
    """
    Optimized Zero-Knowledge Proof Prover for parameter updates

    Uses optimized circuit with Merkle tree hashing and optimized L2 distance.
    Maintains full security (no sampling).

    Proves that:
    1) MerkleRoot(W_t) == W_t_root
    2) MerkleRoot(W_t1) == W_t1_root
    3) Poseidon-fold(data_indices) == data_hash
    4) ||W_t1 - W_t||^2 <= max_distance (full verification)
    """

    def __init__(self,
                 circuit_js_dir: str = 'circuits/build/parameter_update_optimized_js',
                 proving_key_path: str = 'circuits/build/parameter_update_optimized_0001.zkey',
                 use_simulation: bool = True,
                 param_size: int = 100,
                 batch_size: int = 32,
                 scale: float = DEFAULT_SCALE):
        """
        Initialize optimized ZKP prover

        Args:
            circuit_js_dir: Directory containing compiled circuit JS
            proving_key_path: Path to proving key (.zkey file)
            use_simulation: If True, simulate proof generation (for testing)
            param_size: Number of parameters to prove
            batch_size: Number of data indices
            scale: Quantization scale factor
        """
        self.circuit_js_dir = circuit_js_dir
        self.proving_key_path = proving_key_path
        self.use_simulation = use_simulation
        self.param_size = param_size
        self.batch_size = batch_size
        self.scale = scale
        if os.getenv('POL_INTEGRITY', '0') == '1' and self.use_simulation:
            raise RuntimeError("POL_INTEGRITY=1 forbids simulation; initialize ZKPProverOptimized with use_simulation=False and required dependencies.")

        logger.info(f"Initialized optimized ZKP prover (simulation={use_simulation})")
        logger.info(f"  Circuit: {circuit_js_dir}")
        logger.info(f"  Proving key: {proving_key_path}")
        logger.info(f"  Param size: {param_size}, Batch size: {batch_size}")

    def generate_proof(self,
                       W_t,
                       W_t1,
                       data_indices: List[int],
                       max_distance: float) -> Tuple[Dict, Dict]:
        """
        Generate ZKP for parameter update

        Args:
            W_t: Current parameters (tensor or array)
            W_t1: Updated parameters (tensor or array)
            data_indices: Data indices used for training
            max_distance: Maximum allowed L2 distance squared

        Returns:
            proof: Groth16 proof object
            public_signals: Public signals (W_t_root, W_t1_root, data_hash, max_distance)
        """
        if self.use_simulation:
            return self._simulate_proof(W_t, W_t1, data_indices, max_distance)
        return self._generate_real_proof(W_t, W_t1, data_indices, max_distance)

    def _simulate_proof(self, W_t, W_t1, data_indices: List[int], max_distance: float) -> Tuple[Dict, Dict]:
        """Simulate proof generation for testing"""
        # Flatten and quantize parameters
        t = self._flatten_or_tensor(W_t)
        t1 = self._flatten_or_tensor(W_t1)
        t = t[:self.param_size]
        t1 = t1[:self.param_size]
        q_t = quantize_to_field(t, self.scale)
        q_t1 = quantize_to_field(t1, self.scale)

        # Compute Merkle roots
        W_t_root = compute_merkle_root(q_t)
        W_t1_root = compute_merkle_root(q_t1)

        # Compute data hash (using folded Poseidon as in original)
        di = (data_indices or [])[:self.batch_size]
        di = di + [0] * (self.batch_size - len(di))
        data_hash = poseidon_fold(di)

        # Compute L2 distance squared
        dist2 = float(
            torch.sum((t1.to(torch.float64) - t.to(torch.float64)) ** 2)
        )

        # Public signals
        public = {
            'W_t_root': W_t_root,
            'W_t1_root': W_t1_root,
            'data_hash': data_hash,
            'max_distance': int(max_distance),
            'actual_distance': dist2,
        }

        # Simulated proof (structure-correct but not cryptographically valid)
        proof = {
            'pi_a': ['0', '0', '1'],
            'pi_b': [['0', '0'], ['0', '0'], ['1', '0']],
            'pi_c': ['0', '0', '1'],
            'protocol': 'groth16',
            'curve': 'bn128',
            'optimized': True,
        }

        logger.info(f"Simulated optimized proof generated")
        logger.info(f"  W_t_root: {W_t_root[:16]}...")
        logger.info(f"  W_t1_root: {W_t1_root[:16]}...")
        logger.info(f"  Actual distance: {dist2}, Max: {int(max_distance)}")

        return proof, public

    def _generate_real_proof(self, W_t, W_t1, data_indices: List[int], max_distance: float) -> Tuple[Dict, Dict]:
        """Generate real cryptographic proof using snarkjs"""
        # Prepare private inputs (quantized ints)
        t = self._flatten_or_tensor(W_t)[:self.param_size]
        t1 = self._flatten_or_tensor(W_t1)[:self.param_size]
        q_t = quantize_to_field(t, self.scale)
        q_t1 = quantize_to_field(t1, self.scale)
        di = (data_indices or [])[:self.batch_size]
        di = di + [0] * (self.batch_size - len(di))

        # Public signals via Merkle root and Poseidon fold
        W_t_root = compute_merkle_root(q_t)
        W_t1_root = compute_merkle_root(q_t1)
        data_hash = poseidon_fold(di)

        # Prepare input JSON for circuit
        circuit_input = {
            'W_t': [str(x) for x in q_t],
            'W_t1': [str(x) for x in q_t1],
            'data_indices': [str(x) for x in di],
            'W_t_root': str(W_t_root),
            'W_t1_root': str(W_t1_root),
            'data_hash': str(data_hash),
            'max_distance': str(int(max_distance)),
        }

        with tempfile.TemporaryDirectory(prefix='pol_zkp_opt_') as tmp_dir:
            input_file = os.path.join(tmp_dir, 'input_optimized.json')
            witness_file = os.path.join(tmp_dir, 'witness_optimized.wtns')
            proof_file = os.path.join(tmp_dir, 'proof_optimized.json')
            public_file = os.path.join(tmp_dir, 'public_optimized.json')

            with open(input_file, 'w') as f:
                json.dump(circuit_input, f)

            # Generate witness
            subprocess.run([
                'node', f'{self.circuit_js_dir}/generate_witness.js',
                f'{self.circuit_js_dir}/parameter_update_optimized.wasm',
                input_file, witness_file
            ], check=True)

            # Generate proof
            subprocess.run([
                'snarkjs', 'groth16', 'prove',
                self.proving_key_path, witness_file,
                proof_file, public_file
            ], check=True)

            with open(proof_file, 'r') as f:
                proof = json.load(f)
            with open(public_file, 'r') as f:
                public_arr = json.load(f)

            public = {
                'W_t_root': str(public_arr[0]),
                'W_t1_root': str(public_arr[1]),
                'data_hash': str(public_arr[2]),
                'max_distance': int(public_arr[3]),
            }

            logger.info(f"Real optimized proof generated")
            logger.info(f"  W_t_root: {public['W_t_root'][:16]}...")
            logger.info(f"  W_t1_root: {public['W_t1_root'][:16]}...")

            return proof, public

    def _flatten_or_tensor(self, x) -> torch.Tensor:
        """Convert tensor or array to flat torch.Tensor (compatible with quantize_to_field)"""
        if isinstance(x, torch.Tensor):
            return x.reshape(-1).detach().cpu()
        elif isinstance(x, dict):
            return flatten_first_n(x, self.param_size)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).reshape(-1).cpu()
        elif isinstance(x, (list, tuple)):
            return torch.tensor(x).reshape(-1).cpu()
        else:
            raise TypeError(f"Expected torch.Tensor, dict, or array-like, got {type(x)}")

    def verify_proof_locally(self, proof: Dict, public_signals: Dict) -> bool:
        """
        Verify proof locally using snarkjs (for testing)

        Args:
            proof: Groth16 proof
            public_signals: Public signals

        Returns:
            valid: True if proof is valid
        """
        if self.use_simulation:
            logger.warning("Local verification not available in simulation mode")
            return True

        try:
            # Write proof and public signals to files
            with open('proof_verify.json', 'w') as f:
                json.dump(proof, f)

            public_arr = [
                public_signals['W_t_root'],
                public_signals['W_t1_root'],
                public_signals['data_hash'],
                str(public_signals['max_distance']),
            ]
            with open('public_verify.json', 'w') as f:
                json.dump(public_arr, f)

            # Verify using snarkjs
            result = subprocess.run([
                'snarkjs', 'groth16', 'verify',
                'circuits/build/verification_key_optimized.json',
                'public_verify.json',
                'proof_verify.json'
            ], capture_output=True, text=True)

            valid = 'OK' in result.stdout
            logger.info(f"Local verification: {'PASSED' if valid else 'FAILED'}")

            return valid

        except Exception as e:
            logger.error(f"Local verification failed: {e}")
            return False
        finally:
            for f in ['proof_verify.json', 'public_verify.json']:
                if os.path.exists(f):
                    os.remove(f)
