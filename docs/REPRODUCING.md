# Reproducing the Paper Experiments

This guide describes the repository interfaces for running PoL-BFL
experiments. The complete matrix uses 50 clients, 200 communication rounds,
three random seeds, and GPU-based training and proof generation. Individual
experiments can be run independently.

## Installation

Create the Python environment and install the JavaScript dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-final.txt
npm ci
```

## Datasets

Prepare CIFAR-10, CIFAR-100, and FEMNIST under a common dataset directory. The
expected layout and archive checksums are documented in
[DATASETS.md](DATASETS.md).

```bash
export POLBFL_DATA_ROOT=/path/to/data
```

## Native components

Build the native Poseidon helper:

```bash
cargo build --release --locked --manifest-path tools/poseidon_native/Cargo.toml
install -d -m 0755 .tools/poseidon-native
install -m 0755 \
  tools/poseidon_native/target/release/polbfl-poseidon-native \
  .tools/poseidon-native/polbfl-poseidon-native
```

Build the ICICLE-Snark backend:

```bash
bash scripts/build_icicle_snark.sh
```

Set the directory containing the circuit, proving key, verification key,
witness generator, prover, and verifier:

```bash
export POLBFL_ZK_BUILD=/path/to/circuits/final/build
```

## Experiment configuration

The paper matrix is defined in
`experiments/final/paper_matrix.json`. The main security matrix can be
inspected without starting training:

```bash
python -m experiments.final.run_matrix
```

## Running one security experiment

The following command runs the PoL-BFL method on the CIFAR-10 free-riding
configuration for seed 1337:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 \
python -u -m experiments.final.run_security_cell \
  --dataset CIFAR10 \
  --attack FreeRidingNT \
  --method PoLBFL \
  --seed 1337 \
  --run-id cifar10-freeridingnt-polbfl-s1337 \
  --output experiments/results/cifar10-freeridingnt-polbfl-s1337 \
  --data-root "$POLBFL_DATA_ROOT" \
  --zk-build "$POLBFL_ZK_BUILD" \
  --process-training \
  --train-processes-per-gpu 8 \
  --proof-workers 8
```

The number of training processes and proof workers can be adjusted for the
available hardware.

## Experiment modules

| Study | Module |
|---|---|
| Main security comparison | `experiments.final.run_matrix` |
| Layer contribution | `experiments.final.run_layer_matrix` |
| Robust-aggregation composability | `experiments.final.run_table4_matrix` |
| Incentive effectiveness | `experiments.final.run_table5_matrix` |
| Scalability | `experiments.final.run_scalability_matrix` |
| Non-IID sensitivity | `experiments.final.run_noniid_matrix` |
| Adaptive attacks | `experiments.final.run_adaptive_matrix` |
| Cross-hardware verification | `experiments.final.run_cross_hardware_matrix` |
| Sybil scalability | `experiments.final.run_sybil_matrix` |

Each module provides additional command-line options through `--help`.

## Aggregation

Table 2 results from multiple seeds can be aggregated with:

```bash
python -m experiments.final.aggregate_table2 \
  experiments/results/*/result.json \
  --output experiments/results/table2-aggregate.json
```

Other studies provide corresponding `aggregate_*` modules under
`experiments/final/`.

## Result files

A completed experiment directory contains the experiment manifest, per-round
records, final metrics, and the cryptographic and contract records required by
the selected method. Generated results are excluded from version control.

Comparisons across runs should use the same source revision, dependency set,
dataset checksums, partition configuration, protocol parameters, and random
seeds.
