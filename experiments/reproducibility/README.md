# Configuration-Based Reproduction Utilities

This package provides configuration expansion, reduced-scale launcher checks,
coverage summaries, and result validation.

## Coverage summary

```bash
python experiments/reproducibility/audit_reproduction_coverage.py
```

The command reports the experiment targets represented by the tracked
configuration files and the launcher available for each target.

## Reduced-scale launcher check

```bash
python experiments/reproducibility/run_repro_smoke.py \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --dry-run
```

Removing `--dry-run` executes the configuration and writes its manifest and
results to the configured output directory.

## Configuration expansion

A tracked security configuration can be expanded with:

```bash
python experiments/reproducibility/run_paper_config.py \
  --config-file \
    experiments/reproducibility/configs/rq1_table1_cifar10_free_riding_nt_vanilla_formal.json \
  --gpus 0,1 \
  --dry-run
```

The launcher records the expanded jobs, selected devices, output paths, and
configuration metadata.

## Result validation

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/reproduction
```

The validation report summarizes available, missing, and incompatible result
records for the selected result directory.

The paper experiment matrices under `experiments/final/` provide the
lower-level runners used for the complete evaluation. End-to-end examples are
provided in [`docs/REPRODUCING.md`](../../docs/REPRODUCING.md).
