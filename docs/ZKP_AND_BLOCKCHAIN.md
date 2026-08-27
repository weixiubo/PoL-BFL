# Zero-Knowledge Proof and Blockchain Components

## Sampled SGD relation

The circuit sources are located under `circuits/final/`. The principal
relation proves a fixed-point SGD transition for one challenged five-step
checkpoint interval.

The relation includes:

- 14 coordinates selected from the committed gradient sample;
- five active or prefix-padded optimizer steps;
- a batch capacity of 32;
- signed 48-bit values at scale `10^6`;
- pairwise and cumulative rounding bounds;
- bindings for the sample plan, checkpoint endpoints, gradients, batch indices,
  auxiliary values, challenge, round context, commitment root, and model
  digest;
- 1,090,382 R1CS constraints in the reference profile.

SHA-256 checkpoint chains and Merkle openings are checked by the outer verifier
against the same public commitments. Numerical tolerances apply to the encoded
SGD relation, while Groth16 verification is exact.

## Proving components

The proof path consists of:

1. Circom-compatible witness generation;
2. BN254 Groth16 proof generation through ICICLE-Snark;
3. independent proof verification through Rapidsnark;
4. canonical 192-byte proof encoding;
5. signed verifier receipts and a three-of-five decision threshold.

Build the ICICLE-Snark backend with:

```bash
bash scripts/build_icicle_snark.sh
```

The reference benchmark interface is:

```bash
python scripts/zk_reference_benchmark.py --help
```

Generated circuit, witness, proof, and setup files are excluded from version
control. Experiment manifests record the identities of the circuit, proving
key, verification key, witness generator, prover, and verifier.

Test proving keys are intended for local circuit checks. Deployment keys should
be generated from a verified multi-party Powers-of-Tau transcript and an
independent circuit-specific contribution.

## Solidity protocol

The contract `chainEnv/contracts/PoLBFLProtocol.sol` implements:

- client and verifier registration with stake;
- round initialization and commitment submission;
- stake- and reputation-weighted verifier selection;
- challenge deadlines;
- ECDSA quorum receipt verification;
- reward, reputation, timeout, and slashing transitions;
- verifier accountability;
- gas-responsive minimum-stake updates.

Commitment roots and committee decisions are stored on chain. Training traces,
model tensors, and proof witnesses remain off chain.

The contract test uses Solidity 0.8.20 and an isolated Ganache 7.9.2 instance:

```bash
pytest -q tests/test_final_contract_protocol.py
```

The test covers registration, round transitions, commitment submission,
committee selection, receipt settlement, timeout handling, rewards, slashing,
and gas measurements.
