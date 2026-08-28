# PoL-BFL Protocol and Implementation Specification

This document summarizes the protocol implemented by the repository and
described in *PoL-BFL: Towards Trustworthy Federated Learning with
Zero-Knowledge Proofs and Verifiable Incentives*,
DOI `10.1145/3770855.3817739`.

## System model

PoL-BFL targets cross-silo federated learning with persistent organizational
identities. A deployment contains:

- clients that train local models and commit to learning traces;
- an aggregator that combines eligible model updates;
- a verifier committee that evaluates challenged trace intervals;
- smart contracts for registration, commitments, receipts, stake, reputation,
  rewards, and slashing;
- off-chain storage for learning traces and on-chain commitment roots.

The protocol assumes an authenticated source of gas-price information,
deterministic training for supported hardware profiles, and stake values that
are meaningful to participating organizations.

## Round lifecycle

Each federated learning round consists of four ordered phases.

### Local training and commitment

For each participating client:

1. Train the global model on the client's local data.
2. Record an ordered trace `tau = (W, I, H, A)`, where `W` contains model
   checkpoints, `I` contains batch indices, `H` contains the checkpoint hash
   chain, and `A` contains values required by the proof relation.
3. Commit each private batch as `D_t = SHA256(b_t)`.
4. Bind each checkpoint as `h_t = SHA256(w_t || D_t || I_t || t)` using the
   repository serialization format.
5. Build a Merkle tree over the ordered checkpoint hashes.
6. Submit the model update and commitment root before the challenge is drawn.

Training data and trace contents remain off chain.

### Sampled ZK-PoL verification

After commitments are fixed, the protocol samples clients and checkpoint
intervals for verification. The interval set combines recent checkpoints with
randomly selected checkpoints from the training trajectory.

For each selected interval, the Groth16 relation verifies:

- authentication of the interval endpoints under the commitment root;
- the prescribed fixed-point SGD transition;
- the committed batch indices and sampled gradients;
- the configured update-magnitude bound;
- binding to the client, round, model, optimizer, challenge, and commitment
  root.

Verifier nodes evaluate the proof independently. A decision requires three
distinct, timely, context-matched signed receipts from authorized members of a
five-member committee.

Numerical tolerances apply to the fixed-point trace relation. Groth16 proof
verification remains exact.

### Robust aggregation

Only updates that pass cryptographic verification and Sybil screening enter the
aggregation set. The implementation provides reputation-weighted variants of:

- Trimmed Mean;
- Krum;
- coordinate-wise Median.

Sybil screening uses committed batch-index evidence and checkpoint-trajectory
similarity. Statistical filtering at aggregation and cryptographic proof
verification are recorded as distinct outcomes.

### Incentives and reputation

Clients and verifiers lock stake before participating. The protocol applies:

- rewards for verified participation;
- exponential moving-average reputation updates;
- timeout penalties;
- slashing for signed rejection outcomes and provable verifier misconduct;
- a gas-responsive minimum-stake policy.

The client reward and reputation updates are:

```text
R_i(t) = R_base × (1 + beta_work × Work_i(t) + beta_rep × rho_i(t))
rho_i(t+1) = alpha × rho_i(t) + (1 - alpha) × I_i(t)
```

The minimum stake follows:

```text
S_min(t) = max(
    S_base,
    G(t) × C_ops / (gamma × p_challenge × p_detect) + margin
)
```

## Verifier selection

Verifier selection uses an authenticated round seed with stake and reputation
weights:

```text
score_i =
    VRF(seed_t, address_i)
    × verifier_stake_i / total_verifier_stake
    × verifier_reputation_i
```

Selection is deterministic from the published round inputs and is revealed
after client commitments are fixed. Aggregator and verifier roles are disjoint
within a round.

## Experiment configuration

The paper configuration uses:

| Parameter | Value |
|---|---:|
| Clients | 50 |
| Malicious clients | 10 |
| Rounds | 200 |
| Local epochs | 5 |
| Batch size | 32 |
| Learning rate | 0.01 |
| Audit probability | 0.20 |
| Verifier threshold | 3 of 5 |
| Minimum client stake | 0.05 ETH |
| Slashing ratio | 1.0 |
| Reputation decay | 0.9 |
| Gas-price reference | 1.5 gwei |
| ETH-price reference | 2,500 USD |

CIFAR-10 uses ResNet-18, CIFAR-100 uses ResNet-34, and FEMNIST uses a
two-layer convolutional network with natural writer partitions. IID partitions
are used unless an experiment explicitly studies non-IID data.

The evaluated attack families include free-riding, Byzantine random updates,
model replacement, ALIE, MinMax, data poisoning, Sybil identities, checkpoint
interpolation, gradient mimicry, partial replay, and combined adaptive attacks.

The comparison methods include Vanilla FL, Krum, Trimmed Mean, SDEA,
ShapleyFL, FoolsGold, Veriblock-FL, Kaizen, and FedCoin.

## Proof-system configuration

| Property | Value |
|---|---:|
| Curve and proof system | BN254 Groth16 |
| Sampled gradient ratio | 1 percent |
| SGD steps per challenged interval | 5 |
| Reference constraint count | 1,090,382 |
| Encoded proof size | 192 bytes |
| Pairwise tolerance | `1e-5` |
| Cross-hardware trajectory tolerance | `1e-3` |

SHA-256 checkpoint chains and Merkle openings are verified outside the circuit.
The proving relation binds the challenged SGD transition to its round,
commitment, batch indices, endpoints, and model digest.

## Software organization

The protocol implementation is divided among the following packages:

- `polbfl/protocol` for commitments, challenges, rounds, and storage;
- `polbfl/zk` and `circuits/final` for proof construction and verification;
- `polbfl/committee` for verifier selection and signed receipts;
- `polbfl/aggregation` and `polbfl/sybil` for update filtering;
- `polbfl/incentives` for rewards, reputation, and stake transitions;
- `chainEnv/contracts` for the Solidity implementation;
- `experiments/final` for the paper experiment matrix.
