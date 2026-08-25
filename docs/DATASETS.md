# Datasets

Formal PoL-BFL runs receive an explicit `--data-root`. Dataset files are not
stored in Git. The run manifest records the resolved dataset root, canonical
archive checksum, partition-index SHA-256, partition mode, and seed.

## CIFAR-10

Expected canonical archive:

```text
data/CIFAR10/cifar-10-python.tar.gz
MD5 c58f30108f718f92721af3b95e74349a
```

The paper workload uses all 50 clients, IID partitioning unless otherwise
specified, deterministic crop/flip augmentation under `POL_INTEGRITY=1`, and
ResNet-18 with 10 output classes.

## CIFAR-100

Expected canonical archive:

```text
data/CIFAR100/cifar-100-python.tar.gz
MD5 eb9058c3a382ffc7106e4002c42a8d85
```

The paper workload uses IID partitioning unless the non-IID study is selected,
deterministic augmentation, and ResNet-34 with 100 output classes.

## FEMNIST

FEMNIST uses LEAF-style writer shards:

```text
data/FEMNIST/train/*.json
data/FEMNIST/test/*.json
```

Formal preflight requires 36 train and 36 test shards. The main experiment
preserves natural writer identities and distributes sorted writers across 50
cross-silo clients; it does not replace the writer partition with an IID split.

When source shards must be prepared from a public parquet mirror:

```bash
python experiments/scripts/tools/prepare_femnist_hf.py \
  --data-root data/FEMNIST
```

## Integrity preflight

Verify all three paper datasets together with the remaining formal runtime:

```bash
POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m experiments.final.preflight \
  --paper /absolute/path/to/main.pdf \
  --data-root /absolute/path/to/data \
  --zk-build /absolute/path/to/circuits/final/build/production
```

The formal runner serializes every client partition index list into a canonical
partition digest. Worker processes reconstruct the same partition under the
same seed; a source or checkpoint mismatch causes a fail-closed run rather than
an inferred result.
