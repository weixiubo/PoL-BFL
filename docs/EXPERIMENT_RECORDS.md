# Experiment Records

The [paper-to-code correspondence](PAPER_TRACEABILITY.md) identifies the study
implementations. The artifact archive preserves the recorded source versions
and experimental files without modifying their numerical contents.

## Record groups

| Archive path | Description |
|---|---|
| `data/VeryFL-2025/` | Recorded 2025 experiments, attack-defense comparisons, ablations and scalability measurements |
| `data/repro_recovery/smoke/` | Recorded development runs, checkpoints, configurations and execution logs |
| `data/repro_recovery/formal/` | Recorded formal-run configurations and execution plans |
| `data/final/polbfl-compact-evidence-cf74e7c/` | Subsequent reproduction records tied to source `cf74e7c` |
| `data/final/polbfl-compact-evidence-778e041-phase0/` | Source-bound analysis, execution coverage and verification records tied to `778e041` |
| `paper/定稿-2026-06-06/` | Formal manuscript sources, figures, bibliography and PDF |

## Subsequent reproduction records

The two compact reproduction bundles report `measurement_complete: false`.
That flag describes the coverage of those source-version-specific runs. Their
individual records distinguish completed cells, equation calculations, measured
contract execution, controlled benchmarks and planned matrices.

| Study | Preserved correspondence in these bundles |
|---|---|
| Table 2 | One CIFAR-10 free-riding PoL-BFL cell for seed 1337, with 200 round records, capture and contract records |
| Tables 3, 4, 5, 8, 9, 10, 11 | Executable study routes and plans; these bundles do not contain every completed matrix cell |
| Table 6 | Recorded economic equation calculation and checks |
| Table 7 | Controlled Veriblock-FL benchmark record; this is one component of the comparison |
| Table 12 | Production and controlled benchmark routes with source-specific requirements |
| Table 13 | Recorded contract transactions, observed gas and formula checks |
| Figures 2 and 3 | Derivation routes tied to accepted run records |
| Figure 5 | Equation reconstruction of the published gas-price curves |
| Figures 4 and 6 | Recorded sensitivity and scalability execution routes |

The broader historical runs remain available separately under `data/` and in
the source history. A run plan represents intended execution. A value extracted
from the published PDF is a comparison target. Neither is relabeled as a newly
measured result.

## Re-execution conditions

The complete matrix uses the dataset versions, seeds, client counts and round
counts specified by the study configuration. GPU training, production proving,
cross-hardware comparisons and optional Brownie integrations require their
documented dependencies. CPU examples and source tests cover their stated
interfaces and are separate from those complete experiments.
