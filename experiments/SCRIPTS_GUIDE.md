# Experiment Script Guide

## Paper experiment modules

| Purpose | Module |
|---|---|
| Main security matrix | `experiments.final.run_matrix` |
| Single security experiment | `experiments.final.run_security_cell` |
| Layer contribution | `experiments.final.run_layer_matrix` |
| Robust-aggregation composability | `experiments.final.run_table4_matrix` |
| Incentive effectiveness | `experiments.final.run_table5_matrix` |
| Scalability | `experiments.final.run_scalability_matrix` |
| Non-IID sensitivity | `experiments.final.run_noniid_matrix` |
| Adaptive attacks | `experiments.final.run_adaptive_matrix` |
| Cross-hardware verification | `experiments.final.run_cross_hardware_matrix` |
| Sybil scalability | `experiments.final.run_sybil_matrix` |

Each study has a corresponding aggregation or derivation module under
`experiments/final/`.

## Configuration-based utilities

The `experiments/reproducibility/` package provides:

- configuration expansion and execution through `run_paper_config.py`;
- result-manifest validation through `validate_reproduction.py`.

## Supporting utilities

The `experiments/scripts/` directory is organized into:

| Location | Purpose |
|---|---|
| `runners/` | Study-specific launchers |
| `utils/` | Metrics, manifests, plotting, and baseline adapters |
| `tools/` | Dataset and artifact preparation |
| `tests/` | Supporting utility tests |
| directory root | Analysis, reporting, and monitoring programs |

## Usage notes

- Runner options are available through `--help`.
- Experiment comparisons should use consistent datasets, partitions, seeds,
  protocol parameters, and dependency versions.
- Multi-seed tables should include every seed defined by the corresponding
  experiment matrix.

End-to-end examples are provided in
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md).
