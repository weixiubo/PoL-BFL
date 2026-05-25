"""
ZKP Prover for PoL-FL

Generates zero-knowledge proofs for parameter updates using Circom/snarkjs.
Implements Poseidon(2) folding and deterministic quantization consistent with the circuit.
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Dict, List, Tuple
import torch
import numpy as np

from zkp.hash import (
    DEFAULT_SCALE,
    flatten_first_n,
    quantize_to_field,
    poseidon_fold,
)

logger = logging.getLogger(__name__)


class ZKPProver:
    """
    Zero-Knowledge Proof Prover for parameter updates

    Proves that:
    1) Poseidon-fold(W_t) == W_t_hash
    2) Poseidon-fold(W_t1) == W_t1_hash
    3) Poseidon-fold(data_indices) == data_hash
    4) ||W_t1 - W_t||^2 <= max_distance (all in Fr with deterministic quantization)
    """

    def __init__(self,
                 circuit_js_dir: str = 'circuits/build/parameter_update_js',
                 proving_key_path: str = 'circuits/build/parameter_update_0001.zkey',
                 param_size: int = 100,
                 batch_size: int = 32,
                 scale: int = DEFAULT_SCALE,
                 use_simulation: bool = False):
        self.circuit_js_dir = circuit_js_dir
        self.proving_key_path = proving_key_path
        self.param_size = param_size
        self.batch_size = batch_size
        self.scale = scale
        self.use_simulation = use_simulation
        if os.getenv('POL_INTEGRITY', '0') == '1' and self.use_simulation:
            raise RuntimeError("POL_INTEGRITY=1 forbids simulation; initialize ZKPProver with use_simulation=False and required dependencies.")
        if not self.use_simulation:
            self._verify_dependencies()
        logger.info(f"ZKPProver initialized (simulation={self.use_simulation})")

    def _verify_dependencies(self):
        try:
            result = subprocess.run(['snarkjs', '--version'], capture_output=True, text=True)
            logger.info(f"snarkjs version: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError("snarkjs is required for ZKP proof generation")

    def _flatten_or_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.reshape(-1).detach().cpu()
        elif isinstance(x, dict):
            return flatten_first_n(x, self.param_size)
        else:
            raise TypeError("Expected torch.Tensor or state_dict dict")

    def generate_proof(self,
                       W_t,
                       W_t1,
                       data_indices: List[int],
                       max_distance: float) -> Tuple[Dict, Dict]:
        if self.use_simulation:
            return self._simulate_proof(W_t, W_t1, data_indices, max_distance)
        return self._generate_real_proof(W_t, W_t1, data_indices, max_distance)

    def _simulate_proof(self, W_t, W_t1, data_indices: List[int], max_distance: float) -> Tuple[Dict, Dict]:
        # Minimal simulation: structure-correct proof and poseidon hashes via helper
        t = self._flatten_or_tensor(W_t)
        t1 = self._flatten_or_tensor(W_t1)
        t = t[:self.param_size]
        t1 = t1[:self.param_size]
        q_t = quantize_to_field(t, self.scale)
        q_t1 = quantize_to_field(t1, self.scale)
        W_t_hash = poseidon_fold(q_t)
        W_t1_hash = poseidon_fold(q_t1)
        di = (data_indices or [])[:self.batch_size]
        di = di + [0] * (self.batch_size - len(di))
        data_hash = poseidon_fold(di)
        # squared distance on quantized ints
        dist2 = int(np.sum((np.array(q_t1, dtype=object) - np.array(q_t, dtype=object))**2))
        public = {
            'W_t_hash': W_t_hash,
            'W_t1_hash': W_t1_hash,
            'data_hash': data_hash,
            'max_distance': (dist2 + 1) if max_distance is None else int(max_distance),
        }
        proof = {
            'pi_a': ['0x0', '0x0'],
            'pi_b': [['0x0', '0x0'], ['0x0', '0x0']],
            'pi_c': ['0x0', '0x0'],
            'protocol': 'groth16',
            'curve': 'bn128',
        }
        return proof, public

    def _generate_real_proof(self, W_t, W_t1, data_indices: List[int], max_distance: float) -> Tuple[Dict, Dict]:
        # Prepare private inputs (quantized ints)
        t = self._flatten_or_tensor(W_t)[:self.param_size]
        t1 = self._flatten_or_tensor(W_t1)[:self.param_size]
        q_t = quantize_to_field(t, self.scale)
        q_t1 = quantize_to_field(t1, self.scale)
        di = (data_indices or [])[:self.batch_size]
        di = di + [0] * (self.batch_size - len(di))

        # Public signals via Poseidon fold (decimal strings)
        W_t_hash = poseidon_fold(q_t)
        W_t1_hash = poseidon_fold(q_t1)
        data_hash = poseidon_fold(di)

        # Use quantized squared L2 distance as max_distance if not provided in field scale
        if max_distance is None:
            dist2 = int(np.sum((np.array(q_t1, dtype=object) - np.array(q_t, dtype=object))**2)) + 1
        else:
            dist2 = int(max_distance)

        input_data = {
            'W_t': [str(int(x)) for x in q_t],
            'W_t1': [str(int(x)) for x in q_t1],
            'data_indices': [str(int(x)) for x in di],
            'W_t_hash': str(int(W_t_hash)),
            'W_t1_hash': str(int(W_t1_hash)),
            'data_hash': str(int(data_hash)),
            'max_distance': str(int(dist2)),
        }

        with tempfile.TemporaryDirectory(prefix='pol_zkp_') as tmp_dir:
            input_file = os.path.join(tmp_dir, 'zkp_input.json')
            witness_file = os.path.join(tmp_dir, 'witness.wtns')
            proof_file = os.path.join(tmp_dir, 'proof.json')
            public_file = os.path.join(tmp_dir, 'public.json')

            with open(input_file, 'w') as f:
                json.dump(input_data, f)

            # Generate witness
            subprocess.run([
                'node', f'{self.circuit_js_dir}/generate_witness.js',
                f'{self.circuit_js_dir}/parameter_update.wasm',
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
                'W_t_hash': str(public_arr[0]),
                'W_t1_hash': str(public_arr[1]),
                'data_hash': str(public_arr[2]),
                'max_distance': int(public_arr[3]),
            }
            return proof, public


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    prover = ZKPProver(use_simulation=True)
    W_t = torch.randn(120)
    W_t1 = W_t + torch.randn(120) * 0.01
    data_indices = list(range(40))
    proof, public = prover.generate_proof(W_t, W_t1, data_indices, max_distance=None)
    print("public:", public)
