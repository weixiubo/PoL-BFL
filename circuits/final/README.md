# ZK-PoL sampled SGD circuit

`sampled_sgd_transition.circom` is the protocol circuit for challenged
checkpoint intervals. It binds the public context, sample plan, endpoint
weights, gradients, ordered batch indices, and auxiliary activation/error
terms. It proves the sampled gradient relation, fixed-point SGD transition with
bounded pairwise and cumulative rounding, active-step padding, and an L2 update
bound. Public inputs bind the proof to the round context, pre-challenge Merkle
root, challenge ID, interval index, and ordered private-batch SHA-256
commitment.

The outer verifier must additionally verify the SHA-256 checkpoint hash chain
and Merkle openings that authenticate the circuit's public commitments under
the client commitment root. A proof is accepted only after both checks and an
M-of-N signed verifier decision.

- `sampled_sgd_smoke.circom` is used for fast setup/prove/verify tests.
- `sampled_sgd_reference.circom` is the reference proof profile used by the
  circuit size, proving time, memory, proof size, and verification gates.
- `poseidon_bridge.cjs` is the single Circomlib commitment adapter used by the
  Python recorder; it accepts batched requests on standard input.
- `generate_smoke_input.cjs` and `generate_reference_input.cjs` create valid,
  deterministic test witnesses. They are test fixtures, not experimental
  result generators.

The reference profile has 1,090,382 R1CS constraints with Circom 2.2.2. A
deployment proving key must be produced from a verified multi-party
Powers-of-Tau transcript followed by at least one independent circuit-specific
contribution. The isolated development CRS used by benchmark automation is
never a production trust setup.
