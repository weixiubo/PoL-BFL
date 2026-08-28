# Paper-to-Code Correspondence

This document maps the principal components of the PoL-BFL paper to their
implementations and tests in the repository. The paper is identified by
DOI `10.1145/3770855.3817739`.

## Protocol components

| Paper component | Implementation | Verification tests |
|---|---|---|
| Canonical hashing and serialization | `polbfl/crypto/canonical.py` | `tests/test_final_protocol_core.py` |
| Learning traces, hash chains, and Merkle commitments | `polbfl/protocol/trace.py`, `polbfl/training/torch_recorder.py` | `tests/test_final_protocol_core.py`, `tests/test_final_torch_recorder.py` |
| Recent and random interval challenges | `polbfl/protocol/challenge.py` | `tests/test_final_protocol_core.py` |
| Strict SGD replay | `polbfl/verification/strict.py`, `polbfl/verification/torch_replay.py` | `tests/test_final_strict_verifier.py`, `tests/test_final_torch_replay.py` |
| Sampled SGD Groth16 relation | `circuits/final/sampled_sgd_transition.circom`, `polbfl/zk/witness.py` | `tests/test_final_zk_recorder.py` |
| Groth16 proving and verification | `polbfl/zk/groth16.py`, `polbfl/zk/icicle_pool.py` | `tests/test_final_groth16_backend.py`, `tests/test_final_icicle_pool.py` |
| Proof encoding | `polbfl/zk/codec.py` | `tests/test_final_groth16_backend.py` |
| Verifier selection | `polbfl/committee/selection.py`, `polbfl/committee/orchestration.py` | `tests/test_final_committee_orchestration.py` |
| Signed quorum receipts | `polbfl/committee/ecdsa.py`, `polbfl/committee/receipts.py` | `tests/test_final_ecdsa_receipts.py` |
| Robust aggregation | `polbfl/aggregation/robust.py` | `tests/test_final_aggregation_and_sybil.py` |
| Sybil screening | `polbfl/sybil/trace_screening.py` | `tests/test_final_aggregation_and_sybil.py` |
| Round orchestration | `polbfl/protocol/round_engine.py` | `tests/test_final_round_engine.py` |
| Rewards, reputation, and stake | `polbfl/incentives` | `tests/test_final_protocol_ledger.py` |
| Solidity settlement protocol | `chainEnv/contracts/PoLBFLProtocol.sol` | `tests/test_final_contract_protocol.py` |
| Content-addressed trace storage | `polbfl/storage/content_addressed.py` | `tests/test_final_content_store.py` |

## Experiment components

| Paper result | Experiment module |
|---|---|
| Table 2: main security comparison | `experiments.final.run_matrix` |
| Table 3: layer contribution | `experiments.final.run_layer_matrix` |
| Table 4: robust-aggregation composability | `experiments.final.run_table4_matrix` |
| Table 5: incentive effectiveness | `experiments.final.run_table5_matrix` |
| Table 6: participant economics | `experiments.final.run_economics` |
| Table 7: system overhead | `experiments.final.run_table7_matrix` |
| Table 8: scalability | `experiments.final.run_scalability_matrix` |
| Table 9: non-IID sensitivity | `experiments.final.run_noniid_matrix` |
| Table 10: adaptive attacks | `experiments.final.run_adaptive_matrix` |
| Table 11: cross-hardware verification | `experiments.final.run_cross_hardware_matrix` |
| Table 12: zero-knowledge proof cost | `experiments.final.aggregate_table12` |
| Table 13: smart-contract gas cost | `scripts/contract_gas_benchmark.py` |
| Figure 2 | `experiments.final.convergence` |
| Figure 3 | `experiments.final.reputation_evolution` |
| Figure 4 | `experiments.final.aggregate_sensitivity` |
| Figure 5 | `experiments.final.gas_price_stress` |
| Figure 6 | `experiments.final.run_sybil_matrix`, `experiments.final.aggregate_figure6` |

Experiment dimensions are defined in
`experiments/final/paper_matrix.json`. Numerical values used for comparison
are stored under `config/` and
`experiments/reproducibility/paper_targets/`.

## Reference proof profile

The reference circuit uses Circom 2.2.2, BN254 Groth16, five SGD steps,
14 sampled coordinates, 48-bit signed values, scale `10^6`, and 1,090,382
R1CS constraints. Proofs use the standard 192-byte BN254 encoding.

The ICICLE-Snark backend produces proofs from Circom witness and proving-key
files. Each proof can also be checked by the Rapidsnark verifier through the
same public-signal interface.
