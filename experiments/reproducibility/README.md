# Reproducibility Tools

This package contains the formal PoL-BFL reproduction launcher, coverage auditor, smoke runner, and validation tools.

The bundled `paper_targets/tables/` directory contains the table targets used
by the validator. Use `--paper-root` only when you want to validate against a
different local manuscript checkout.

## Coverage Audit

```bash
python experiments/reproducibility/audit_reproduction_coverage.py
```

## Smoke Run

```bash
python experiments/reproducibility/run_repro_smoke.py \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --dry-run
```

## Formal Paper Runs

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

Each job records a `run_manifest.json`, raw result JSON, runner log, and optional validation gate report under `experiments/results/repro_recovery/`.

When extending a partially completed matrix, plan from the latest strict
validation manifest so already substantiated paper cells are not rerun just
because a non-claimed launch parameter changed:

```bash
python experiments/reproducibility/run_paper_config.py \
  --config-file experiments/reproducibility/configs/paper/rq1_main_security_formal.json \
  --only cifar10 \
  --resume \
  --skip-passed-validation-manifest experiments/results/repro_recovery/validation_snapshots/current_strict/validation_manifest.json \
  --dry-run
```

`--resume` also skips cells whose `run_manifest.json` is still marked
`running`. Use `--include-running-manifests` only after confirming that such a
manifest is stale and safe to supersede.

## Validation

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/repro_recovery/formal
```

The validator compares normalized outputs against paper targets and records whether each target is passing, failing, missing, measurement-only, or protocol-incompatible.
