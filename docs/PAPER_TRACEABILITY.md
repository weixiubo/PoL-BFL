# Paper-to-code traceability

The sole normative source is the final submitted paper identified by SHA-256
`0b013e58d4f99f91470c61a891a4ee89dfd09eff58e131abbe730d1f6f91e6d4`.
`config/paper_targets.json` contains the main scalar gates. Complete multi-method
Tables 2 and 4 and the vector data for Figure 5 are transcribed directly from
that PDF by SHA-gated extractors into dedicated target files. Values not fixed
by the paper are isolated under `implementation_choices` in
`config/paper_protocol.json` or an explicitly labelled inference record.

| Paper requirement | Code owner | Automated evidence |
|---|---|---|
| Canonical private-batch, model, index, context, and object hashing | `polbfl/crypto/canonical.py` | `test_final_protocol_core.py` |
| Trace `tau=(W,I,H,A)`, hash chain, Merkle root, commit-before-challenge | `polbfl/protocol/trace.py`, `polbfl/training/torch_recorder.py` | `test_final_protocol_core.py`, `test_final_torch_recorder.py` |
| Hybrid `Q=K+R` recent/random interval challenge | `polbfl/protocol/challenge.py` | `test_final_protocol_core.py` |
| Exact private SGD replay and endpoint upload binding | `polbfl/verification/strict.py`, `polbfl/verification/torch_replay.py` | `test_final_strict_verifier.py`, `test_final_torch_replay.py`, `test_final_protocol_trainer.py` |
| 1% sampled fixed-point SGD relation, L2 bound, pair/final tolerances | `circuits/final/sampled_sgd_transition.circom`, `polbfl/zk/witness.py` | `test_final_zk_recorder.py`, reference R1CS benchmark |
| Challenge, root, context, interval, data, endpoint and final-model proof binding | `polbfl/zk/prover.py`, `polbfl/zk/bundle.py`, `polbfl/zk/field.py` | `test_final_zk_bundle.py`, `test_final_zk_recorder.py` |
| Real Groth16 proof generation and exact verification | `polbfl/zk/groth16.py`, `polbfl/zk/icicle_pool.py` | `test_final_groth16_backend.py`, `test_final_icicle_pool.py`, `test_final_zk_bundle.py` |
| Production Powers-of-Tau, phase-2 contribution and verification-log provenance | `experiments/final/trust_setup.py`, `scripts/zk_reference_setup.sh` | `test_final_trust_setup.py`, formal preflight |
| Ceremony entropy never exposed through process arguments | all setup/continuation scripts | static ceremony-boundary tests plus verified setup logs |
| 192-byte proof transport | `polbfl/zk/codec.py` | `test_final_groth16_backend.py` |
| Authenticated stake/reputation VRF committee selection and role separation | `polbfl/committee/selection.py`, `polbfl/committee/orchestration.py` | `test_final_committee_orchestration.py` |
| Distinct, timely, exact-field ECDSA 3-of-5 receipts | `polbfl/committee/ecdsa.py`, `polbfl/committee/receipts.py` | `test_final_ecdsa_receipts.py`, `test_final_committee_orchestration.py` |
| Verified-only reputation-weighted Trimmed Mean, Krum and Median | `polbfl/aggregation/robust.py` | `test_final_aggregation_and_sybil.py` |
| Sybil screening using committed trajectory cosine or identical batch indices | `polbfl/sybil/trace_screening.py` | `test_final_aggregation_and_sybil.py`, `test_final_round_engine.py` |
| Reward, EMA reputation, full client/verifier slashing, timeouts and gas-responsive stake | `polbfl/incentives/economics.py`, `polbfl/incentives/ledger.py` | `test_final_protocol_ledger.py` |
| Ordered four-phase round semantics and Layer-1/Layer-2 slashing boundary | `polbfl/protocol/round_engine.py` | `test_final_round_engine.py` |
| Commit-before-VRF, on-chain committee, signed quorum, timeout, reward and slash transitions | `chainEnv/contracts/PoLBFLProtocol.sol` | `test_final_contract_protocol.py` |
| Python/Solidity-identical commitment-bound audit tickets and all-round EVM replay | `polbfl/protocol/audit.py`, `contract_replay.py`, `contract_round_replay.cjs` | protocol, contract, and formal-evidence tests |
| Paper gas gates | optimized `PoLBFLProtocol.sol` | commitment 84,810; receipt 111,228; reward 44,702; slash 54,036 gas in real Ganache transactions |
| Paper experiment settings and every reported acceptance value | `config/paper_protocol.json`, `config/paper_targets.json` | final experiment manifests and validator |
| Full Table 2 baselines and three-seed aggregation | `baseline_algorithms.py`, `run_matrix.py`, `aggregate_table2.py` | source locks and matrix/aggregation tests |
| Executable four-profile Table 3 ablation | `run_layer_matrix.py`, `aggregate_table3.py`, profile-gated `PaperRoundEngine` | layer planner, profile, and aggregate tests |
| Both standalone and PoL-prefilter Table 4 modes | `run_table4_matrix.py`, `aggregate_table4.py` | `test_final_table4_complete.py` |
| Four-method Table 5 incentives with measured marginal attack success | `run_table5_matrix.py`, `aggregate_table5.py` | counterfactual, FedCoin PoSap, and full-matrix tests |
| Four-method Table 7 measured overhead | `run_table7_matrix.py`, `compose_table7_result.py`, controlled Kaizen/Veriblock runners | Table 7 extraction, composition, and matrix tests |
| Real adaptive attacks and cost/profit evidence | `run_adaptive_matrix.py`, `run_adaptive_trial.py`, `aggregate_table10.py` | executable attack and full Table 10 tests |
| Attested cross-hardware trials | `run_cross_hardware_matrix.py`, `run_cross_hardware_trial.py`, `aggregate_table11.py` | hardware planner, attestation, and aggregate tests |
| Production PoL and controlled Kaizen ZK cost | production/bundle benchmarks, controlled ceremony, `aggregate_table12.py` | locked tool/setup and complete Table 12 tests |
| Figures 2-6 from accepted rounds, equations, vector PDF data, and real Sybil identities | figure derivation modules and `run_sybil_matrix.py` | figure-specific tests and PDF digest gates |
| Clean source-bound planning for every formal matrix, including all 108 non-IID cells | all `run_*_matrix.py` entry points | matrix planner tests and formal child manifests |
| Raw predictions, proof sets, 3-of-5 receipts and retained trace hashes | `run_security_cell.py`, `capture_formal_evidence.py` | evidence and scratch-boundary tests |
| Atomic resume without duplicate or uncheckpointed rounds | `recovery.py`, `exclusive_gpu_controller.py` | recovery/controller tests |
| Continuous GPU exclusivity with foreign-PID preemption and uncommitted scratch cleanup | `gpu_idle_supervisor.py`, `supervision.py`, `recovery.py` | supervisor process-group integration and recovery tests |
| Canonical authority/source/content sealing for every final table and figure | `experiments/final/evidence.py`, `experiments/final/audit_measurements.py` | `test_final_evidence_sealing.py`, `test_final_coverage_audit.py` |

## Reference proof profile

- Circom 2.2.2 / BN254 / Groth16.
- Five active or prefix-padded SGD steps, batch capacity 32, fourteen
  deterministic coordinate checks drawn from the committed 1% sample.
- 48-bit signed magnitude and scale `10^6`.
- 1,090,382 R1CS constraints.
- Isolated development-key benchmark: witness median 0.617 seconds (peak 107,264
  KiB), proof median 2.453 seconds (peak 922,740 KiB), native verification
  median 4.105 ms, and canonical proof size 192 bytes. These timings establish
  circuit performance but are not production trust provenance.
- The production key is bound to the officially verified power-21 Hermez
  transcript (55 public contributions plus beacon), an independent
  circuit-specific contribution, and both verification-log hashes in
  `trust_setup.json`.
- The locked CUDA prover emits the same standard proof/public JSON for the same
  Circom witness and proving key; the independent Rapidsnark verifier accepts
  it, while the transport remains the canonical 192-byte encoding.
- A real PyTorch Conv2d/BatchNorm2d/Linear optimizer step has passed the same
  reference proving key, SHA/Merkle bundle verifier, and 3-of-5 ECDSA receipt
  quorum end to end.
- The production dual-4090 benchmark measures witness 0.663 seconds, proof 0.206 seconds, peak prover memory 1.953 GiB, native verification 3.978 ms, and full Merkle-plus-proof verification 5.188 ms.
- The controlled 4,000,000-constraint Kaizen cost baseline measures witness 0.969 seconds, proof 0.606 seconds, peak prover memory 5.516 GiB, and verification 6.783 ms.
- Setup automation requires an explicit verified phase-2 Powers-of-Tau file;
  production artifacts cannot silently fall back to a development CRS.

## Experiment-route coverage

`experiments/final/audit_final_coverage.py` requires an explicit owner for
Tables 2-13 and experimental Figures 2-6: 17 routes in total. Table 2 covers
all six methods and 432 three-seed cells; Table 4 covers both modes and 36
cells. Route coverage deliberately reports `measurement_complete: false`
until accepted evidence exists, so implemented code cannot be confused with
reproduced measurements.

Every final route is sealed with the authoritative PDF digest, one valid
execution-source commit, its complete input hashes, and a canonical JSON
SHA-256 digest. The final measurement audit recomputes that digest and rejects
missing, mixed-source, differently sourced, or content-modified evidence. It
also reopens every declared input and verifies its current streaming SHA-256,
so a sealed aggregate cannot outlive or silently diverge from its raw inputs.
For every route backed by formal training cells, it additionally reruns the
full cell verifier over manifests, predictions, audit selection, proof sets,
3-of-5 receipts, retained traces, and all-round contract replay evidence; a
result JSON alone is never sufficient for final acceptance.
The terminal audit itself additionally requires the authoritative PDF and a
fresh sealed preflight report, verifies the deployed worktree is clean at the
declared commit, recursively follows JSON input manifests, and seals the final
report together with the evidence map, paper, and preflight hashes.
Recursion enters only canonical sealed manifests: their digest must first
recompute exactly, while legacy or third-party component JSON is verified as
one opaque byte-hashed input so human-readable logical labels cannot be
misinterpreted as filesystem paths.
Each sealed route also records `analysis_source`; sealing and terminal audit
both reject an analysis worktree that is dirty or differs from the execution
commit represented by its raw inputs.

After each PoL round, the atomic checkpoint and raw JSONL record are durable;
only audited clients' content-addressed traces remain. Worker result files and
unselected private traces are removed after prediction, proof, receipt and
retained-file hashes have been recorded. This keeps the complete matrix within
disk bounds without weakening the accepted evidence path.

## Acceptance boundary

Protocol and contract tests establish functional and adversarial correctness.
The restoration is accepted only after `scripts/zk_reference_benchmark.py` and
the full 50-client, 200-round experiment matrix emit source-bound manifests and
all comparisons against the PDF-derived target files pass in the required
direction with complete three-seed coverage.
