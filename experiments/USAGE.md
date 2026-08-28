# Experiment Usage

## Environment

Set the dataset and zero-knowledge proof directories:

```bash
export POLBFL_DATA_ROOT=/path/to/data
export POLBFL_ZK_BUILD=/path/to/circuits/final/build
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export POL_INTEGRITY=1
```

Dataset preparation is described in
[`docs/DATASETS.md`](../docs/DATASETS.md).

## Inspecting the matrix

The following command prints the main experiment matrix and its resolved
metadata:

```bash
python -m experiments.final.run_matrix
```

## Running one experiment

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

Worker counts can be adjusted for the available hardware. The protocol,
dataset, model, attack, and seed are recorded in the generated manifest.

## Additional interfaces

- `experiments/final/` contains the paper experiment modules and aggregators.
- `experiments/reproducibility/` contains configuration-based launch and
  validation tools.
- `experiments/scripts/` contains analysis, plotting, dataset preparation, and
  supporting experiment utilities.

Additional reproduction details are provided in
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md).
