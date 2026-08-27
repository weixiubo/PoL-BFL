# Paper Experiment Suite

This package contains the experiment matrices, runners, aggregators, and result
validation utilities associated with the PoL-BFL evaluation.

## Reference configuration

| Parameter | Value |
|---|---:|
| Clients | 50 |
| Malicious clients | 10 |
| Rounds | 200 |
| Local epochs | 5 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Audit probability | 0.20 |
| Verifier threshold | 3 of 5 |
| Seeds | 1337, 2026, 3817739 |

CIFAR experiments use IID partitions unless a non-IID configuration is
selected. FEMNIST preserves natural writer partitions.

## Entry points

| Study | Module |
|---|---|
| Main security comparison | `experiments.final.run_matrix` |
| Single security experiment | `experiments.final.run_security_cell` |
| Layer contribution | `experiments.final.run_layer_matrix` |
| Robust-aggregation composability | `experiments.final.run_table4_matrix` |
| Incentive effectiveness | `experiments.final.run_table5_matrix` |
| Scalability | `experiments.final.run_scalability_matrix` |
| Non-IID sensitivity | `experiments.final.run_noniid_matrix` |
| Adaptive attacks | `experiments.final.run_adaptive_matrix` |
| Cross-hardware verification | `experiments.final.run_cross_hardware_matrix` |
| Sybil scalability | `experiments.final.run_sybil_matrix` |

Protocol values are defined in `config/paper_protocol.json`, numerical
comparison values are stored under `config/`, and matrix dimensions are
defined in `paper_matrix.json`.

## Single-experiment example

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

## Outputs

A completed run directory contains:

- the resolved experiment configuration;
- dataset, partition, model, attack, method, and seed information;
- per-round metrics and final metrics;
- proof and verifier-receipt information when PoL verification is enabled;
- stake, reputation, communication, storage, and timing measurements;
- contract-transition records when blockchain execution is enabled.

Study-specific `aggregate_*` modules combine results from the seeds and
configurations defined by the matrix. Generated outputs are excluded from
version control.

Additional instructions are provided in
[`docs/REPRODUCING.md`](../../docs/REPRODUCING.md).
