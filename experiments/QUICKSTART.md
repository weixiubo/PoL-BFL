# Experiment Quickstart

This page lists the shortest commands for checking the PoL-BFL experiment pipeline before running paper-scale jobs.

## Environment

```bash
export POL_DATA_DIR=/path/to/pol-bfl-data
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Smoke Run

```bash
python experiments/reproducibility/run_repro_smoke.py \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --dry-run

bash experiments/scripts/run_repro_smoke.sh \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --gpu 0
```

## CIFAR-10 PoL-BFL Security Matrix

```bash
export NUM_WORKERS_OVERRIDE=0
export POL_MEMORY_CHECKPOINT_LIMIT=2
export POL_COMPACT_REMOTE_RESPONSE=1
export POL_ENABLE_PARALLEL_CLIENT_TRAINING=1

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

## Validation

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/reproduction/formal
```

## Summaries

```bash
python experiments/scripts/summarize_rq1.py
python experiments/scripts/summarize_rq2.py
python experiments/scripts/summarize_rq3.py
python experiments/scripts/summarize_rq4.py
```
