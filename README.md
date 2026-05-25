# PoL-BFL

PoL-BFL is a reference implementation of blockchain-assisted federated learning with verifiable Proof-of-Learning, zero-knowledge proof support, robust aggregation, Sybil detection, and reproducible paper-scale experiment tooling.

The codebase is designed for security-oriented federated learning research: it trains standard FL workloads, records client learning evidence, verifies submitted updates with strict replay or ZKP-backed paths, filters malicious clients before aggregation, and emits validation-ready experiment artifacts.

## Highlights

- Proof-of-Learning verification with checkpoint commitments, Merkle proofs, deterministic replay metadata, and strict verifier nodes.
- Blockchain-facing committee components for verifier receipts, weighted aggregation, anchoring, incentive accounting, and challenge workflows.
- ZKP circuits and snarkjs integration for parameter-update proof experiments.
- Attack implementations for free-riding, Byzantine random noise, model replacement, ALIE, MinMax, data poisoning, and Sybil behaviors.
- Baselines including Vanilla FL, Krum, FoolsGold, ShapleyFL, SDEA, and PoL-BFL.
- Dataset/model support for MNIST, CIFAR-10 with ResNet-18, CIFAR-100 with ResNet-34, and FEMNIST writer partitions.
- Reproduction tooling with manifests, validation reports, protocol checks, and paper-target comparison gates.

## Repository Layout

```text
client/                  FL clients, trainers, PoL managers, ZKP provers
server/                  aggregators, verifier nodes, PoL verification, incentives
dataset/                 dataset adapters
model/                   CNN, ResNet, VGG, and watermark-capable models
experiments/             attacks, runners, formal configs, validation tools
chainEnv/contracts/      Solidity contracts used by the blockchain experiments
circuits/                Circom circuits for ZKP experiments
analysis/                measurement and plotting utilities
tests/                   unit, integration, and smoke tests
```

Datasets, trained checkpoints, experiment outputs, generated ZKP keys, Brownie build artifacts, and node modules are intentionally not included.

## Installation

Use Python 3.9 with a CUDA-enabled PyTorch build for GPU runs.

```bash
git clone https://github.com/weixiubo/PoL-BFL.git
cd PoL-BFL

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

npm install
```

For ZKP experiments, install Circom and snarkjs in the environment used to build the circuits. For blockchain experiments, install Brownie-compatible Solidity tooling and run a local test chain such as Ganache.

## Quick Smoke Test

Run a small dry-run first to confirm the launcher and manifest path:

```bash
python experiments/reproducibility/run_repro_smoke.py \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --dry-run
```

Then run a minimal MNIST smoke:

```bash
bash experiments/scripts/run_repro_smoke.sh \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --gpu 0
```

## Paper-Scale Reproduction

Formal experiment configs live under `experiments/reproducibility/configs/paper/`. The main launcher expands a paper matrix into resumable jobs and writes raw outputs, runner logs, and manifests under `experiments/results/reproduction/`.

```bash
python experiments/reproducibility/run_paper_config.py \
  --config-file experiments/reproducibility/configs/paper/rq1_main_security_formal.json \
  --gpus 0,1 \
  --parallel 2 \
  --only cifar10 \
  --only pol_bfl \
  --resume \
  --start-verifiers \
  --validate-after-job
```

Validate generated outputs against the paper targets:

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/reproduction/formal
```

The validator enforces protocol compatibility before treating a result as a paper-scale claim. Short smoke runs are reported as protocol mismatches, not as successful reproductions.

## Datasets

The repository downloads or prepares datasets at runtime. By default, dataset files are stored under `data/`, which is ignored by git.

FEMNIST uses LEAF-style writer shards. To prepare it from a public parquet mirror:

```bash
python experiments/scripts/tools/prepare_femnist_hf.py \
  --data-root data/FEMNIST
```

See `docs/DATASETS.md` for dataset-specific details.

## Validation Policy

Experiment claims should be tied to raw outputs and a validation manifest. The formal validator compares MA/DR/FPR and related metrics against the paper targets with the configured acceptance gates, and records missing, failing, passing, and protocol-mismatched cells separately.

This repository does not include generated result files. Recreate them with the formal configs above, then publish or archive the resulting manifests alongside your run environment.

## Tests

Focused unit and smoke tests:

```bash
pytest tests/test_reproducibility_tools.py \
       tests/test_deterministic_replay_data.py \
       tests/test_sybil_detector.py
```

ZKP and blockchain tests require the corresponding local toolchains and test-chain services.

## License

This project is released under the MIT License.
