# Experiment Package

The `experiments/` directory contains the evaluation software for PoL-BFL.

## Directory structure

| Directory | Purpose |
|---|---|
| `final/` | Paper experiment matrices, single-cell runners, aggregators, and result validation |
| `reproducibility/` | Configuration expansion, execution, and result processing |
| `scripts/` | Analysis, plotting, dataset preparation, monitoring, and supporting runners |

## Main experiment modules

| Study | Module |
|---|---|
| Main security comparison | `experiments.final.run_matrix` |
| Layer contribution | `experiments.final.run_layer_matrix` |
| Robust-aggregation composability | `experiments.final.run_table4_matrix` |
| Incentive effectiveness | `experiments.final.run_table5_matrix` |
| Scalability | `experiments.final.run_scalability_matrix` |
| Non-IID sensitivity | `experiments.final.run_noniid_matrix` |
| Adaptive attacks | `experiments.final.run_adaptive_matrix` |
| Cross-hardware verification | `experiments.final.run_cross_hardware_matrix` |
| Sybil scalability | `experiments.final.run_sybil_matrix` |

Protocol settings are stored in `config/paper_protocol.json`, experiment
dimensions are stored in `experiments/final/paper_matrix.json`, and numerical
comparison values are stored under `config/`.

Run directories record configuration, dataset information, seed, metrics, and
method-specific artifacts.

Execution examples are provided in [`USAGE.md`](USAGE.md) and
[`docs/REPRODUCING.md`](../docs/REPRODUCING.md).
