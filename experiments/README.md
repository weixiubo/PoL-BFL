# PoL-BFL Experiments

The `experiments/` package contains attacks, baselines, paper-scale runners, reproducibility configs, and result validators for PoL-BFL.

## Main Entry Points

- `experiments/reproducibility/run_paper_config.py`: expands formal paper matrices into resumable jobs.
- `experiments/reproducibility/validate_reproduction.py`: validates raw outputs against paper targets and protocol gates.
- `experiments/reproducibility/audit_reproduction_coverage.py`: reports which paper targets have runnable configs or require measurement provenance.
- `experiments/scripts/runners/run_rq1_security.py`: main security runner for attacks and baselines.
- `experiments/scripts/runners/run_rq2_layer_contribution.py`: layer contribution ablation runner.
- `experiments/scripts/runners/run_rq5_composability.py`: PoL-BFL plus robust aggregation composability runner.
- `experiments/scripts/runners/run_rq6_noniid.py`: Non-IID sensitivity runner.
- `experiments/scripts/runners/run_rq9_adaptive.py`: adaptive attacker runner.

## Quick Start

```bash
python experiments/reproducibility/run_repro_smoke.py \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --dry-run

bash experiments/scripts/run_repro_smoke.sh \
  --config-file experiments/reproducibility/configs/smoke_mnist.json \
  --gpu 0
```

## Formal Runs

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

Outputs are written under `experiments/results/reproduction/`, which is ignored by git.

## Result Validation

```bash
python experiments/reproducibility/validate_reproduction.py \
  --results-root experiments/results/reproduction/formal
```

The validator keeps raw experiment output separate from claims. It records passing cells, failing cells, missing cells, and protocol mismatches in the validation manifest.
