# Zero-Knowledge Proof and Gas Measurements

## Scope

This document presents the zero-knowledge proof and smart-contract measurements
supported by the repository.

## ZKP Circuit Optimization

The sampled SGD relation uses BN254 Groth16 with the following configuration:

| Property | Value |
|---|---:|
| Sampled coordinates | 14 |
| Gradient sample ratio | 1 percent |
| SGD steps per challenged interval | 5 |
| R1CS constraints | 1,090,382 |
| Encoded proof size | 192 bytes |

The relation binds the round context, challenge, commitment root, checkpoint
endpoints, batch indices, sampled gradients, auxiliary values, and final model
digest. SHA-256 hash-chain and Merkle membership checks are performed by the
outer verifier.

## Reference ZK benchmark

The benchmark record is available at `evidence/zk_reference_benchmark.json`.

| Metric | Reference measurement | Value reported in the paper |
|---|---:|---:|
| Witness computation | 0.617 s | 1.8 s |
| Proof generation | 2.453 s | 4.2 s |
| Peak prover memory | 922,740 KiB | 2.5 GB |
| Groth16 verification | 4.105 ms | 8.5 ms |
| Proof size | 192 bytes | 192 bytes |

## Gas Cost Optimization

The Solidity protocol stores commitment roots and quorum decisions on chain.
Private training traces and model tensors remain off chain.

| Operation | Repository measurement | Value reported in the paper |
|---|---:|---:|
| Commitment submission | 84,810 gas | 85,000 gas |
| Proof-receipt submission | 111,228 gas | 120,000 gas |
| Reward claim | 44,702 gas | 45,000 gas |
| Slash execution | 54,036 gas | 65,000 gas |
| Reference round | 151,757 gas | approximately 225,000 gas |

The measurements are produced by `scripts/contract_gas_benchmark.py` with an
isolated Ganache instance. The benchmark also verifies that the deployed
runtime remains within the EIP-170 contract-size limit.

## Reproducing the Results

Command-line options for the benchmark programs are available through:

```bash
python scripts/zk_reference_benchmark.py --help
python scripts/zk_production_benchmark.py --help
python scripts/contract_gas_benchmark.py --help
```

## Security Properties

Performance optimizations preserve the bindings between a proof and its round
context, challenge, commitment root, checkpoint endpoints, batch indices,
sampled gradients, and model digest. Verifier decisions remain subject to the
signed committee threshold.
