# Sampled SGD Groth16 Circuit

`sampled_sgd_transition.circom` defines the zero-knowledge relation used for
challenged checkpoint intervals. The circuit verifies sampled gradient
coordinates, a fixed-point SGD transition, ordered batch indices, endpoint
weights, update bounds, and the public values that bind a proof to its protocol
round and challenge.

SHA-256 checkpoint chains and Merkle openings are verified outside the circuit.
A client proof is therefore evaluated together with the corresponding
commitment opening and verifier-committee decision.

## Circuit files

- `sampled_sgd_transition.circom` defines the complete sampled SGD relation.
- `sampled_sgd_reference.circom` provides the reference benchmark profile.
- `poseidon_bridge.cjs` exposes the Circomlib Poseidon implementation to the
  Python trace recorder.
- `generate_reference_input.cjs` generates deterministic reference inputs.

The reference profile contains 1,090,382 R1CS constraints when compiled with
Circom 2.2.2.

## Setup guidance

Deployment key generation uses a verified multi-party Powers-of-Tau transcript
followed by an independent circuit-specific contribution. Proving and
verification keys correspond to the same compiled circuit.
