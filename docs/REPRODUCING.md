# Reproducing Experiments

This guide describes the formal reproduction workflow used by PoL-BFL.

## Preflight

Run the lightweight checks first:

```bash
pytest tests/test_reproducibility_tools.py \
       tests/test_deterministic_replay_data.py \
       tests/test_sybil_detector.py

python experiments/reproducibility/audit_reproduction_coverage.py

python experiments/reproducibility/run_paper_config.py \
  --config-file experiments/reproducibility/configs/paper/rq1_main_security_formal.json \
  --only cifar10 \
  --only pol_bfl \
  --dry-run
```

## Formal RQ1 Run

For the main CIFAR-10 PoL-BFL security matrix:

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

Recommended environment for paper-scale PoL-BFL cells:

```bash
export NUM_WORKERS_OVERRIDE=0
export POL_MEMORY_CHECKPOINT_LIMIT=2
export POL_COMPACT_REMOTE_RESPONSE=1
export POL_ENABLE_PARALLEL_CLIENT_TRAINING=1
```

Each cell writes:

```text
run_manifest.json
runner.log
rq1_output/config.json
rq1_output/rq1_results.json
validation_gate_report.md
```

## Validation

Run the validator over a result root:

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/reproduction/formal
```

The validator emits:

```text
validation_manifest.json
validation_report.md
```

MA and DR are accepted when they are within the configured lower tolerance of the paper target. FPR is accepted when it is within the configured upper tolerance. Protocol-incompatible outputs are marked separately and should not be used as paper reproduction claims.

The repository includes the paper target tables needed by the validator under `experiments/reproducibility/paper_targets/`. Pass `--paper-root /path/to/Paper` only when validating against a separate manuscript checkout.
