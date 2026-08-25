# Native Poseidon helper

This helper evaluates the exact Circomlib BN254 Poseidon parameterization used
by the final ZK-PoL circuit. It preserves the JSON protocol of
`circuits/final/poseidon_bridge.cjs` while removing JavaScript big-integer
overhead from checkpoint construction.

Build the pinned release binary with:

```bash
cargo build --release --locked --manifest-path tools/poseidon_native/Cargo.toml
install -m 0755 tools/poseidon_native/target/release/polbfl-poseidon-native \
  .tools/poseidon-native/polbfl-poseidon-native
```

The formal preflight runs the embedded Circomlib acceptance vector and compares
representative positive, negative, and batched operations against
CircomlibJS. A binary that differs by even one field element is rejected.
