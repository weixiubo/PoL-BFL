# Native Poseidon Helper

The native helper implements the Circomlib BN254 Poseidon parameterization used
by the sampled SGD circuit. It exposes the same JSON request format as
`circuits/final/poseidon_bridge.cjs` and supports batched field operations.

Build and install the helper with:

```bash
cargo build --release --locked --manifest-path tools/poseidon_native/Cargo.toml
install -d -m 0755 .tools/poseidon-native
install -m 0755 tools/poseidon_native/target/release/polbfl-poseidon-native \
  .tools/poseidon-native/polbfl-poseidon-native
```

The helper is tested against CircomlibJS for positive, negative, and batched
inputs. Matching field outputs ensure that native commitments and circuit
commitments use the same parameterization.
