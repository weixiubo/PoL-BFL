# PoL-BFL

PoL-BFL is a paper-aligned implementation of trustworthy blockchain federated
learning with sampled zero-knowledge Proof-of-Learning, robust aggregation,
Sybil screening, verifiable incentives, and reproducible experiment gates.

The normative specification is the final submitted paper identified by
SHA-256 `0b013e58d4f99f91470c61a891a4ee89dfd09eff58e131abbe730d1f6f91e6d4`.
Machine-readable protocol choices and acceptance values live in
`config/paper_protocol.json`, `config/paper_targets.json`, and
`experiments/final/paper_matrix.json`. SHA-gated PDF extractors generate the
complete Table 2/Table 4 and Figure 5 target files without manual transcription.

## Implemented protocol

- Canonical SHA-256 batch, model, index, checkpoint, hash-chain, and Merkle
  commitments with commit-before-challenge ordering.
- Five-step sampled SGD Groth16 relation with 1% committed gradient sampling,
  fixed-point tolerance, data-index binding, update bounds, and 1,090,382 R1CS
  constraints.
- Dual-GPU ICICLE-Snark proof generation over the locked Circom `.zkey/.wtns`
  artifacts. Every proof is independently checked by the locked Rapidsnark
  verifier and encoded as 192 bytes.
- Authenticated stake/reputation VRF committee selection, disjoint aggregator
  and verifier roles, low-S ECDSA receipts, and strict 3-of-5 quorum handling.
- Reputation-weighted Trimmed Mean, Krum, and Median after proof, Sybil, and
  statistical prefiltering.
- Stake, reward, reputation, timeout, verifier, slashing, and gas-responsive
  economic transitions, including a real Ganache-tested Solidity protocol.
- CIFAR-10/ResNet-18, CIFAR-100/ResNet-34, and FEMNIST/two-layer-CNN paper
  workloads with exact dataset/partition/seed manifests.
- Free-riding, Byzantine, model replacement, ALIE, MinMax, data poisoning, and
  Sybil attack paths, plus checkpoint interpolation, gradient mimicry, partial
  replay, and combined adaptive trajectories.

## Repository layout

```text
polbfl/                 protocol, crypto, ZK, aggregation, committee, economics
client/                 client trainers and private evidence recording
chainEnv/contracts/     Solidity protocol and settlement contracts
circuits/final/         reference Circom relation and native input bridges
experiments/final/      accepted experiment runner, matrix, manifests, validator
config/                 paper protocol, targets, economics, and toolchain locks
scripts/                preflight/build/supervision/benchmark helpers
tests/                  unit, adversarial, real-ZK, contract, and integration tests
docs/                   implementation, dataset, ZK, and reproduction guidance
```

Generated datasets, evidence, checkpoints, result matrices, proving artifacts,
native build outputs, and Node modules are intentionally ignored by Git.

## Reference environment

The locked dual-RTX-4090 reference deployment uses:

- Python 3.13.2;
- PyTorch 2.9.1 with CUDA 12.8 and torchvision 0.24.1;
- Node.js 24.14.0, snarkjs 0.7.5, Circom 2.2.2;
- circomlib 2.0.5 and circomlibjs 0.1.7;
- Rapidsnark at the commit recorded in `config/toolchain.lock.json`;
- ICICLE-Snark 0.1.0 / ICICLE 3.8.0, built for CUDA 12.4 and sm_89;
- Solidity 0.8.20 and Ganache 7.9.2.

Install Python and Node dependencies:

```bash
python -m pip install -r requirements-final.txt
npm ci
```

Build the pinned native helpers:

```bash
cargo build --release --locked --manifest-path tools/poseidon_native/Cargo.toml
install -d -m 0755 .tools/poseidon-native
install -m 0755 \
  tools/poseidon_native/target/release/polbfl-poseidon-native \
  .tools/poseidon-native/polbfl-poseidon-native

bash scripts/build_icicle_snark.sh
```

Reference circuit setup requires an explicit verified phase-2 Powers-of-Tau
file; development CRS artifacts are rejected for formal results. See
`circuits/final/README.md` and `docs/ZKP_AND_BLOCKCHAIN.md`.

## Preflight and formal reproduction

Preflight verifies the paper, datasets, GPUs, deterministic runtime, source,
contracts, circuit, keys, native binaries, shared libraries, and all locked
hashes:

```bash
POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m experiments.final.preflight \
  --paper /absolute/path/to/main.pdf \
  --data-root /absolute/path/to/data \
  --zk-build /absolute/path/to/circuits/final/build/production
```

Dry-run the 432-cell, six-method Table 2 security matrix:

```bash
python -m experiments.final.run_matrix
```

Run selected formal cells on shared dual-GPU hardware:

```bash
python -u -m experiments.final.run_matrix \
  --dataset CIFAR10 \
  --attack FreeRidingNT \
  --execute \
  --supervised \
  --data-root /absolute/path/to/data \
  --method PoLBFL \
  --zk-build /absolute/path/to/circuits/final/build/production
```

The supervisor waits for both GPUs to remain idle and resumes only a
source-compatible atomic checkpoint. A cell can enter an aggregate only when
its result contains `formal_accepted: true`. Shortened, proof-disabled,
PoL-disabled, synthetic, replayed, or busy-GPU runs belong in the diagnostic
namespace and are rejected by the aggregator.

Each accepted PoL round retains raw test predictions and labels, proof-set and
individual proof digests, exact 192-byte proof sizes, three signed receipt
digests, and hashes for every retained audited trace file. Formal evidence
capture recomputes accuracy and all hashes. Non-audited scratch evidence and
worker payloads are removed only after the atomic checkpoint and JSONL record
are durable.

See `docs/REPRODUCING.md` for the complete workflow.

## Tests

Run the final implementation suite without optional live toolchains:

```bash
pytest -q tests/test_final_*.py
```

Real reference proof tests require `POL_ZK_REFERENCE_BUILD`,
`RAPIDSNARK_PROVER`, and `RAPIDSNARK_VERIFIER`. The ICICLE cross-check test also
uses the locked worker/backend/library paths. The Solidity end-to-end test
starts an isolated in-process Ganache provider and does not require publishing
contracts to a network.

## License

This project is released under the MIT License. Third-party native dependencies
retain their upstream licenses.
