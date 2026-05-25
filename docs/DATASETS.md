# Datasets

PoL-BFL expects datasets to be prepared locally under `data/` or under the path specified by `POL_DATA_DIR`. Dataset files are not stored in this repository.

## MNIST, FashionMNIST, CIFAR-10, CIFAR-100

The standard torchvision datasets are downloaded by the dataset adapters when they are not already present:

```bash
export POL_DATA_DIR=/path/to/pol-bfl-data
python experiments/reproducibility/run_repro_smoke.py \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --dry-run
```

For multi-GPU servers, keep `POL_DATA_DIR` on a local SSD or a fast shared filesystem.

## FEMNIST

FEMNIST experiments use LEAF-style writer partitions:

```text
data/FEMNIST/train/*.json
data/FEMNIST/test/*.json
```

You can prepare those shards from a public parquet mirror:

```bash
python experiments/scripts/tools/prepare_femnist_hf.py \
  --data-root data/FEMNIST
```

The converter preserves writer IDs and creates deterministic per-writer train/test splits when the source mirror provides a single split.

## Reproducibility

Formal runs record the dataset name, model, partitioning mode, seed, client count, local epochs, and run directory in `run_manifest.json`. Keep that manifest with any published result table.
