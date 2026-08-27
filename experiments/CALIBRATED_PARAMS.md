# Experiment Parameters

## Paper configuration

The principal experiment settings are defined in
`config/paper_protocol.json` and
`experiments/final/paper_matrix.json`.

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

CIFAR-10 uses ResNet-18, CIFAR-100 uses ResNet-34, and FEMNIST uses a
two-layer convolutional network with natural writer partitions.

## Resource controls

The experiment launchers recognize the following environment variables:

```bash
export NUM_WORKERS_OVERRIDE=0
export POL_MEMORY_CHECKPOINT_LIMIT=2
export POL_COMPACT_REMOTE_RESPONSE=1
export POL_ENABLE_PARALLEL_CLIENT_TRAINING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

These variables control worker scheduling, memory use, response serialization,
parallel training, and deterministic CUDA behavior. Changes to protocol or
model parameters should be recorded in the experiment configuration rather
than encoded as resource controls.
