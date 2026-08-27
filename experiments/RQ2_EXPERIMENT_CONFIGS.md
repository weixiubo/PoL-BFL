# Layer Contribution Experiment

## Objective

The layer contribution experiment measures the detection contribution of the
three PoL-BFL defense layers. The implementation corresponds to Table 3 of the
paper and is provided by `experiments.final.run_layer_matrix`.

## Variants

| Variant | Cryptographic verification | Robust aggregation | Sybil and reputation processing | Economic enforcement |
|---|---:|---:|---:|---:|
| `L1` | Enabled | Disabled | Disabled | Disabled |
| `L1L2` | Enabled | Enabled | Disabled | Disabled |
| `L1L3` | Enabled | Disabled | Enabled | Enabled |
| `Full` | Enabled | Enabled | Enabled | Enabled |

Statistical exclusion in Layer 2 is distinct from a cryptographic rejection.
Economic state changes are disabled in variants that omit Layer 3 enforcement.

## Configuration

- datasets: CIFAR-10, CIFAR-100, and FEMNIST;
- attacks: FreeRidingNT, ALIE, and Sybil;
- clients: 50;
- malicious clients: 10;
- rounds: 200;
- local epochs: 5;
- seeds: 1337, 2026, and 3817739.

## Execution

```bash
python -m experiments.final.run_layer_matrix \
  --execute \
  --results-root experiments/results/table3 \
  --data-root "$POLBFL_DATA_ROOT" \
  --zk-build "$POLBFL_ZK_BUILD"
```

Aggregation is provided by
`experiments.final.aggregate_table3`. A complete Table 3 evaluation contains
every dataset, attack, layer variant, and seed combination.
