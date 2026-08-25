# ZK proof and blockchain components

## Reference Groth16 relation

The accepted circuit sources live under `circuits/final/`. The reference
relation proves a fixed-point sampled SGD transition for one challenged
five-step checkpoint interval:

- 14 deterministic circuit coordinates drawn from the committed 1% gradient
  sample;
- five active or prefix-padded optimizer steps;
- batch capacity 32;
- signed 48-bit values at scale `10^6`;
- pairwise and cumulative tolerance bounds;
- sample-plan, checkpoint-endpoint, gradient, batch-index, auxiliary,
  challenge, context, commitment-root, and final-model bindings;
- 1,090,382 R1CS constraints.

SHA-256 checkpoint/hash-chain/Merkle membership is checked by the outer bundle
verifier over the same public openings. Groth16 verification itself is exact;
numerical tolerance applies only to the encoded trace relation.

## Setup and proving artifacts

Build instructions are in `circuits/final/README.md`. Formal setup requires a
verified phase-2 Powers-of-Tau file of sufficient power. The toolchain lock
records the expected production ceremony checksum and explicitly forbids the
development benchmark CRS for formal results.

Generated `.r1cs`, `.zkey`, `.wasm`, `.wtns`, proof, public-signal, and setup
files are ignored by Git. Each formal manifest nevertheless hashes the exact
R1CS, proving key, verifying key, WASM, native witness generator, prover, and
verifier used by the run.

The reference execution path is:

1. native Circom witness generation;
2. standard BN254 Groth16 proving through the locked dual-GPU ICICLE-Snark
   worker cache;
3. independent locked Rapidsnark verification of every proof;
4. canonical 192-byte proof encoding;
5. independent ECDSA verifier receipts and a strict 3-of-5 quorum.

Build the pinned CUDA prover with:

```bash
bash scripts/build_icicle_snark.sh
```

Run the locked reference benchmark with:

```bash
python scripts/zk_reference_benchmark.py \
  --build circuits/final/build/reference \
  --rapidsnark-prover .tools/rapidsnark/package/bin/prover \
  --rapidsnark-verifier .tools/rapidsnark/package/bin/verifier \
  --output evidence/zk_reference_benchmark.json
```

## Solidity protocol

The formal contract is `chainEnv/contracts/PoLBFLProtocol.sol`. It implements:

- stake-backed client and verifier registration;
- commit-before-VRF round ordering;
- stake/reputation-weighted verifier selection;
- commitment anchoring and audit deadlines;
- ECDSA quorum receipt verification;
- accepted settlement, rejected-audit slashing, timeout slashing, rewards,
  reputation updates, and lock release;
- verifier accountability and equivocation handling;
- authenticated gas-price input for minimum-stake policy.

The end-to-end test compiles Solidity 0.8.20 and starts an isolated in-process
Ganache 7.9.2 provider:

```bash
pytest -q tests/test_final_contract_protocol.py
```

It executes real transactions and enforces the paper gas ceilings for
commitment, proof receipt, reward claim, and slash operations. No network
deployment or GitHub publication is required for the test.
