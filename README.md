# PoL-BFL

PoL-BFL is the reference implementation accompanying
*PoL-BFL: Towards Trustworthy Federated Learning with Zero-Knowledge Proofs and
Verifiable Incentives*. The paper appears in the Proceedings of the 32nd ACM
SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026),
DOI [10.1145/3770855.3817739](https://doi.org/10.1145/3770855.3817739).

The project provides a blockchain-assisted federated learning framework that
combines Proof-of-Learning verification, zero-knowledge proofs, robust
aggregation, Sybil detection, and incentive mechanisms.

## Features

- Proof-of-Learning trace generation and verification
- BN254 Groth16 zero-knowledge proof integration
- Authenticated verifier committees and signed quorum receipts
- Robust aggregation with reputation and Sybil screening
- Stake, reward, reputation, timeout, and slashing mechanisms
- Solidity contracts for protocol settlement
- Evaluation workloads for CIFAR-10, CIFAR-100, and FEMNIST
- Security experiments covering free-riding, Byzantine, poisoning, replacement,
  and Sybil attacks

## Repository structure

```text
polbfl/                     core protocol and cryptographic components
client/                     federated learning clients and evidence generation
server/                     aggregation and verifier-node components
dataset/                    dataset interfaces
model/                      neural network architectures
chainEnv/contracts/         Solidity contracts
circuits/final/             Circom circuits and proof interfaces
experiments/final/          experiment configurations and runners
experiments/reproducibility/  reproduction utilities
analysis/                   analysis and measurement utilities
config/                     protocol and toolchain configuration
scripts/                    build and execution utilities
docs/                       technical documentation
tests/                      automated tests
```

Generated datasets, checkpoints, proof artifacts, native binaries, and
experiment outputs are excluded from version control.

## Requirements

- Python 3.13
- Node.js 18 or 20
- CUDA-enabled PyTorch for GPU execution
- Circom and snarkjs for zero-knowledge proof workflows
- Solidity-compatible tooling and Ganache for blockchain workflows

Exact dependency versions are recorded in `requirements-final.txt`,
`package-lock.json`, and `config/toolchain.lock.json`.

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

## Datasets

The repository includes dataset interfaces and experiment configurations for
CIFAR-10, CIFAR-100, and FEMNIST. Dataset files are not distributed with the
source code. Preparation procedures are provided in
[docs/DATASETS.md](docs/DATASETS.md).

## Verification

The automated test suite is executed with:

```bash
pytest -q
```

## Experiments

Experiment definitions are stored in
`experiments/final/paper_matrix.json`. The configured experiment matrix can be
inspected with:

```bash
python -m experiments.final.run_matrix
```

## Documentation

- [Implementation specification](docs/FINAL_IMPLEMENTATION_SPEC.md)
- [Dataset preparation](docs/DATASETS.md)
- [Experiment reproduction](docs/REPRODUCING.md)
- [Paper-to-code correspondence](docs/PAPER_TRACEABILITY.md)
- [Zero-knowledge proof and blockchain components](docs/ZKP_AND_BLOCKCHAIN.md)

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
