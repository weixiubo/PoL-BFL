# Calibrated Parameters

These parameters are the default starting points used by the formal PoL-BFL runners. Keep the values fixed when reproducing paper-scale cells unless the experiment explicitly studies a parameter change.

## CIFAR-10

- Model: `ResNet18`
- Clients: `50`
- Rounds: `200`
- Local epochs: `5`
- Malicious ratio: `0.2`
- Verification rate: configured by the paper matrix
- Checkpoint memory limit: `2`
- Parallel client training: enabled

## CIFAR-100

- Model: `ResNet34`
- Clients: `50`
- Rounds: `200`
- Local epochs: `5`
- Malicious ratio: `0.2`

## FEMNIST

- Model: `SimpleCNN`
- Partitioning: `Natural_Writer`
- Data format: LEAF-style writer JSON shards

## Recommended Environment

```bash
export NUM_WORKERS_OVERRIDE=0
export POL_MEMORY_CHECKPOINT_LIMIT=2
export POL_COMPACT_REMOTE_RESPONSE=1
export POL_ENABLE_PARALLEL_CLIENT_TRAINING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
