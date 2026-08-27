# Experiment Support Utilities

This directory contains study-specific launchers, statistical utilities,
plotting programs, monitoring tools, and dataset preparation scripts.

## Directory groups

| Location | Purpose |
|---|---|
| `runners/` | RQ-oriented and study-specific launchers |
| `utils/` | Metrics, manifests, plotting, and baseline adapters |
| `tools/` | Dataset conversion and artifact preparation |
| `tests/` | Tests for supporting utilities |
| directory root | Result analysis, reporting, and process monitoring |

## Statistical utilities

Statistical programs compute summaries from the result files supplied on the
command line. Comparisons across methods should use matching datasets,
partitions, seeds, client counts, round counts, and model configurations.

## Dataset tools

Dataset conversion programs write files beneath the directory supplied through
their command-line options. Dataset files and generated partitions are excluded
from version control.

## Monitoring tools

Monitoring programs report process and result-directory state. They do not
modify unrelated processes. Hardware allocation and process ownership remain
the responsibility of the experiment launcher.

## Related packages

- `experiments/final/` contains the paper experiment matrices and aggregators.
- `experiments/reproducibility/` contains configuration expansion and result
  validation.
- [`experiments/SCRIPTS_GUIDE.md`](../SCRIPTS_GUIDE.md) lists the principal
  command-line interfaces.
