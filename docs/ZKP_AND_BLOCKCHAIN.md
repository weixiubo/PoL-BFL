# ZKP and Blockchain Components

PoL-BFL includes both off-chain verification logic and blockchain-facing components for reproducible experiments.

## ZKP

Circom circuit sources are stored under `circuits/`. Generated artifacts such as `.r1cs`, `.zkey`, `.wasm`, witnesses, proofs, and ptau files are ignored by git.

Build circuits in a local toolchain:

```bash
bash analysis/build_zkp.sh
```

Optimized and scaling measurements are available under `analysis/`.

## Blockchain

Solidity contracts live under `chainEnv/contracts/`. Brownie and Ganache build directories are ignored. Run blockchain tests only after starting a local chain and installing the Brownie-compatible dependencies from `requirements.txt`.

Relevant components:

- `server/committee/VerifierNode.py`
- `server/committee/AggregatorNode.py`
- `server/incentive/`
- `chainfl/interact.py`

The formal FL experiments can run without publishing generated contract artifacts to git.
