# PoL-BFL

PoL-BFL provides the protocol implementation and evaluation software for
*PoL-BFL: Towards Trustworthy Federated Learning with Zero-Knowledge Proofs and
Verifiable Incentives*. The paper was published in the Proceedings of the 32nd
ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026),
DOI [10.1145/3770855.3817739](https://doi.org/10.1145/3770855.3817739).

## Scope

The repository implements the following components:

- canonical Proof-of-Learning traces, hash chains, and Merkle commitments;
- sampled SGD verification with BN254 Groth16 proofs;
- authenticated verifier selection and ECDSA quorum receipts;
- robust aggregation with reputation and Sybil screening;
- stake, reward, reputation, timeout, and slashing transitions;
- Solidity settlement logic for a local Ethereum-compatible test network;
- CIFAR-10, CIFAR-100, and FEMNIST evaluation workloads;
- source-bound experiment manifests and paper-target validation.

## Repository structure

```text
polbfl/                     protocol, cryptography, verification, and economics
client/                     client training and private evidence recording
server/                     aggregation and verifier-node integration
chainEnv/contracts/         Solidity protocol and settlement contracts
circuits/final/             Circom relations and input bridges
experiments/final/          paper experiment runners and aggregators
experiments/reproducibility/  compatibility launch and validation interfaces
experiments/scripts/        diagnostic and analysis utilities
config/                     protocol, target, economics, and toolchain records
scripts/                    build, preflight, supervision, and benchmark tools
docs/                       protocol and reproduction documentation
tests/                      unit, adversarial, proof, contract, and integration tests
```

Generated datasets, checkpoints, proving artifacts, native binaries, and
experiment outputs are excluded from version control.

## Reference environment

The validated reference environment uses:

- Python 3.13.2;
- PyTorch 2.9.1 with CUDA 12.8;
- torchvision 0.24.1;
- Node.js 20.20.2, within the declared `>=18 <21` engine range;
- Circom 2.2.2 and snarkjs 0.7.5;
- ICICLE-Snark 0.1.0 with ICICLE 3.8.0;
- Solidity 0.8.20 and Ganache 7.9.2.

Exact Python, Node, native-tool, and circuit constraints are recorded in
`requirements-final.txt`, `package-lock.json`, and
`config/toolchain.lock.json`.

## Installation

```bash
git clone https://github.com/weixiubo/PoL-BFL.git
cd PoL-BFL

python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-final.txt
npm ci
```

The native Poseidon helper is built with:

```bash
cargo build --release --locked --manifest-path tools/poseidon_native/Cargo.toml
install -d -m 0755 .tools/poseidon-native
install -m 0755 \
  tools/poseidon_native/target/release/polbfl-poseidon-native \
  .tools/poseidon-native/polbfl-poseidon-native
```

The CUDA prover build is generated with:

```bash
bash scripts/build_icicle_snark.sh
```

## Datasets

Dataset archives and FEMNIST writer shards are prepared outside the source
tree. Required checksums, directory layouts, and partition rules are specified
in [docs/DATASETS.md](docs/DATASETS.md).

## Verification

The complete source test suite is executed with:

```bash
pytest -q
```

Formal preflight requires the submitted paper, canonical datasets, and a
production Groth16 build:

```bash
export POLBFL_PAPER_PDF=/path/to/main.pdf
export POLBFL_DATA_ROOT=/path/to/data
export POLBFL_ZK_BUILD=/path/to/circuits/final/build/production

POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m experiments.final.preflight \
  --paper "$POLBFL_PAPER_PDF" \
  --data-root "$POLBFL_DATA_ROOT" \
  --zk-build "$POLBFL_ZK_BUILD"
```

## Experiment reproduction

The paper matrix is defined in `experiments/final/paper_matrix.json`. A dry-run
of the main security matrix is produced with:

```bash
python -m experiments.final.run_matrix
```

Formal execution procedures, evidence requirements, resume behavior, and
target validation are specified in
[docs/REPRODUCING.md](docs/REPRODUCING.md).

## Technical documentation

- [Implementation specification](docs/FINAL_IMPLEMENTATION_SPEC.md)
- [Paper-to-code traceability](docs/PAPER_TRACEABILITY.md)
- [ZK proof and blockchain components](docs/ZKP_AND_BLOCKCHAIN.md)

## Citation

```bibtex
@inproceedings{wei2026polbfl,
  title     = {PoL-BFL: Towards Trustworthy Federated Learning with
               Zero-Knowledge Proofs and Verifiable Incentives},
  author    = {Wei, Xiubo and Chen, Yahong and Hu, Jiahui and Sun, Zhe and
               Wang, Tao and Niu, Ben},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
               Discovery and Data Mining V.2},
  year      = {2026},
  doi       = {10.1145/3770855.3817739}
}
```

## License

PoL-BFL is distributed under the MIT License. Third-party components retain
their respective licenses.
