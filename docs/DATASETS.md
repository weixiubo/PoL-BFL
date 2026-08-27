# Datasets

PoL-BFL includes experiment configurations for CIFAR-10, CIFAR-100, and
FEMNIST. Dataset files are stored outside the source tree and supplied to
experiment commands through `--data-root`.

## Directory layout

```text
data/
├── CIFAR10/
│   └── cifar-10-python.tar.gz
├── CIFAR100/
│   └── cifar-100-python.tar.gz
└── FEMNIST/
    ├── train/
    │   └── *.json
    └── test/
        └── *.json
```

## CIFAR-10

Expected archive:

```text
CIFAR10/cifar-10-python.tar.gz
MD5 c58f30108f718f92721af3b95e74349a
```

The main CIFAR-10 configuration uses 50 clients, ResNet-18, and 10 output
classes. IID partitioning is used except in experiments that explicitly select
a Dirichlet partition.

## CIFAR-100

Expected archive:

```text
CIFAR100/cifar-100-python.tar.gz
MD5 eb9058c3a382ffc7106e4002c42a8d85
```

The main CIFAR-100 configuration uses ResNet-34 and 100 output classes.
Non-IID experiments select their Dirichlet concentration parameter in the
experiment configuration.

## FEMNIST

FEMNIST uses LEAF-style writer shards under `FEMNIST/train` and
`FEMNIST/test`. The experiment configuration preserves writer identities when
assigning data to clients.

Writer shards can be prepared from a public parquet source with:

```bash
python experiments/scripts/tools/prepare_femnist_hf.py \
  --data-root data/FEMNIST
```

## Reproducibility information

Experiment manifests record the resolved dataset location, archive checksum,
partition mode, partition-index digest, and random seed. Repeated runs should
use the same dataset version and partition configuration when results are
compared.
