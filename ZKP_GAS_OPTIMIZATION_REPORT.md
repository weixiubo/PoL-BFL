# ZKP Circuit Optimization and Gas Cost Optimization

## ZKP Circuit Optimization

The production PoL relation uses BN254 Groth16, fourteen deterministic
coordinates from the committed one-percent gradient sample, five SGD steps,
fixed-point arithmetic, and explicit context, challenge, Merkle, batch, and
endpoint bindings. The production circuit contains 1,090,382 constraints.

Witness generation uses the compiled native Circom witness generator. Proofs
use the hash-locked dual-GPU ICICLE-Snark backend and are independently checked
by the hash-locked Rapidsnark verifier. The transported proof is the canonical
192-byte BN254 encoding.

## Gas Cost Optimization

The Solidity protocol stores commitment roots and signed quorum decisions
rather than private traces or model tensors. It batches a three-signature
quorum into one receipt transition and separates statistical exclusion from
cryptographic slashing. The optimized runtime remains below EIP-170.

Measured Ganache transactions on the locked contract use at most:

- commitment: 84,810 gas;
- proof receipt: 111,228 gas;
- reward claim: 44,702 gas;
- slash: 54,036 gas;
- honest reference total: 151,757 gas.

## Security

Every accepted proof binds the round context, commitment root, challenge,
checkpoint pair, private-batch commitment, sampled weights, gradients, indices,
auxiliary values, and final model digest. Verifier receipts are distinct,
ECDSA-signed, deadline-bound, and require a three-of-five quorum. Formal
ceremony contribution entropy is supplied through standard input and is never
placed in process arguments.

## Results

On the dual RTX 4090 reference server, a source-bound production benchmark
measured a 0.663 s median witness, 0.206 s median proof, 1.953 GiB peak prover
memory, 3.978 ms Groth16 verification, and 5.188 ms complete Merkle-plus-proof
verification. All values improve on the final-paper acceptance bounds.
