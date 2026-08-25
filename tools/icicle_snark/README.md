# ICICLE-Snark build lock

The formal dual-RTX-4090 path uses the upstream ICICLE-Snark worker to produce
standard BN254 Groth16 proofs from the same Circom `.zkey` and `.wtns` files.
Every proof is independently checked by the locked Rapidsnark verifier before
an M-of-N receipt can be issued.

`Cargo.lock` fixes the dependency graph missing from the upstream repository.
Run `scripts/build_icicle_snark.sh` to check out the pinned upstream commit,
build the CUDA 12.4 Ada backend, and install the runtime under
`.tools/icicle-snark/`.
