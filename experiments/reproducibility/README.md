# Paper Experiment Configuration and Execution

This package provides the paper experiment configurations, execution interface,
and result-manifest validation utilities.

## Execute a paper configuration

```bash
python experiments/reproducibility/run_paper_config.py \
  --config-file \
    experiments/reproducibility/configs/paper/rq1_main_security_formal.json \
  --gpus 0,1 \
  --parallel 2 \
  --resume \
  --start-verifiers \
  --validate-after-job
```

The launcher records jobs, selected devices, output paths, configuration
metadata, and run manifests.

## Result validation

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/paper_runs/rq1_main_security
```

The validation report summarizes configuration correspondence, provenance
fields, and metric schemas for records in the selected result directory.

The experiment matrices under `experiments/final/` provide the study runners
and aggregators. End-to-end examples are provided in
[`docs/REPRODUCING.md`](../../docs/REPRODUCING.md).
