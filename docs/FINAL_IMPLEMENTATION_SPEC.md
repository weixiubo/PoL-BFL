# PoL-BFL Final Implementation Specification

## Normative authority

The normative system specification is the final KDD 2026 paper:

- Title: *PoL-BFL: Towards Trustworthy Federated Learning with Zero-Knowledge Proofs and Verifiable Incentives*
- DOI: `10.1145/3770855.3817739`
- Canonical PDF SHA-256: `0b013e58d4f99f91470c61a891a4ee89dfd09eff58e131abbe730d1f6f91e6d4`

Every implementation decision must preserve the protocol, security properties,
experimental settings, and directional performance claims below. When the paper
does not fix a wire format or engineering mechanism, the implementation may
choose one, but the choice must be deterministic, testable, and no weaker than
the stated construction.

## System scope

PoL-BFL targets cross-silo blockchain federated learning with persistent,
stake-backed organizational identities. A deployment consists of:

- clients that train locally and commit to PoL traces;
- an elected aggregator that aggregates only eligible updates;
- a randomly selected verifier committee;
- smart contracts that register actors, anchor commitments and receipts, manage
  stake and reputation, and execute rewards and slashing;
- an authenticated gas-price oracle used by the adaptive minimum-stake rule;
- off-chain trace storage with on-chain commitment roots and retrieval
  challenges.

The protocol assumes deterministic training can be configured for supported
hardware profiles and that clients can lock meaningful collateral.

## Protocol state and round lifecycle

For each round `t`, the implementation must execute four ordered phases.

### Phase 1: local training and commitment

For every participating client `i`:

1. Train the broadcast global model `w^(t)` on private data `D_i` for the
   configured local epochs.
2. Construct a PoL trace `tau_i = (W, I, H, A)` where:
   - `W` is the ordered checkpoint sequence saved every `k` optimizer steps;
   - `I` is the exact ordered sequence of batch indices used by training;
   - `H` binds checkpoints into a hash chain;
   - `A` contains the auxiliary gradients and activations required by the proof
     relation.
3. Commit each private batch as `D_t = SHA256(b_t)`.
4. Bind each checkpoint as
   `h_t = SHA256(w_t || D_t || I_t || t)` using a canonical serialization.
5. Build a Merkle tree over ordered checkpoint hashes and submit the 32-byte
   root `CM_i` before the audit challenge is sampled.
6. Submit the client model update and the commitment root to the aggregator.

The trace remains off-chain. The chain stores the commitment and issues random
retrieval challenges; failure to provide committed evidence is penalizable.

### Phase 2: probabilistic ZK-PoL verification

1. Sample the audit set with probability `p_challenge` after commitments are
   fixed.
2. For each audited client, sample `Q = K + R` checkpoint intervals:
   - `K` recent intervals covering the final training stages;
   - `R` unpredictable intervals sampled across the trajectory.
3. Generate a Groth16 proof for every sampled interval. The proof relation must
   establish all of the following:
   - the committed data value is authenticated by the committed trace;
   - both checkpoint endpoints are authenticated under `CM_i`;
   - the transition is consistent with `k` prescribed SGD steps on the private
     committed batches;
   - the sampled update satisfies the configured L2 magnitude bound;
   - the proof is bound to the client, round, model, optimizer configuration,
     challenge, and commitment root.
4. Keep checkpoints, data, gradients, activations, and witnesses private.
5. Verify proofs independently at `N_v` verifier nodes and accept only an
   `M`-of-`N_v` honest-majority result where `M > N_v / 2`.
6. Each receipt must be ECDSA-signed and bind at least the round, client,
   commitment root, challenge, proof digest, decision, verifier identity, and
   protocol version. Duplicate, stale, mismatched, or unauthorized receipts do
   not count.
7. Aggregator and verifier roles are disjoint within a round.

Groth16 verification remains exact. Numerical tolerance applies only to the
fixed-point trace-consistency relation:

- strict pairwise tolerance `delta_pair`;
- relaxed final-trajectory tolerance `delta_final`.

Both conditions must pass.

### Phase 3: robust aggregation

Let `V_t` be clients that passed Layer 1 and Sybil screening. Form
`U_t = {Delta w_i^(t) | i in V_t}` and aggregate only `U_t`.

The implementation must support reputation-weighted robust aggregation with
Trimmed Mean, Krum, and coordinate-wise Median. The one-round reference protocol
uses Trimmed Mean. Sybil screening must use committed-trace evidence, including
duplicate batch-index evidence and checkpoint-trajectory similarity, rather
than gradient similarity alone.

Layer 1 failure affects proof eligibility and slashing. Layer 2 statistical
rejection affects aggregation eligibility and must not by itself be treated as
a cryptographic proof failure.

### Phase 4: rewards, reputation, and slashing

- Every client must stake `S_i >= S_min` before rewarded participation.
- An `M`-of-`N_v` signed rejection or an expired audit response triggers full
  removal from the aggregation pool and slashing `S_i <- S_i - gamma * S_i`.
- Rewards are based on verified participation, normalized work, and reputation:
  `R_i^(t) = R_base * (1 + beta_work * Work_i^(t) + beta_rep * rho_i^(t))`.
- Reputation follows
  `rho_i^(t+1) = alpha * rho_i^(t) + (1-alpha) * I_i^(t)`.
- Verifiers also stake collateral, earn rewards for correct service, and are
  slashable for provable misbehavior.
- The minimum stake adapts to gas price:
  `S_min^(t) = max(S_base, G^(t)*C_ops/(gamma*p_challenge*p_detect) + margin)`.

The implementation must enforce the paper's rationality condition
`p * (R + gamma*S) > Delta c`, with participation additionally requiring
`R - c >= 0`.

## Verifier selection

For every round, select `N_v` verifiers from the registered pool using an
authenticated VRF seed and the paper's stake/reputation weighted score:

`score_i = VRF(seed_t, addr_i) * S_i^verifier / sum_j(S_j^verifier) * Rep_i^verifier`.

Selection must be deterministic from public inputs, auditable after the round,
and unpredictable before commitment.

## Normative experimental configuration

- Hardware reference: Intel Core i7-13700, 64 GB RAM, two RTX 4090 24 GB GPUs.
- Clients: 50.
- Malicious clients: 10 (20%).
- Rounds: 200.
- Local epochs: 5.
- Learning rate: 0.01.
- Batch size: 32.
- Default partition: IID unless explicitly varied.
- CIFAR-10: ResNet-18, 10 classes.
- CIFAR-100: ResNet-34, 100 classes.
- FEMNIST: 2-layer CNN, 62 classes, natural writer non-IID partition.
- Audit probability: 20%.
- Verifier committee: 3-of-5.
- Pairwise tolerance: `1e-5` for the same-architecture reference setting.
- Final tolerance: `1e-3` for the cross-hardware evaluation.
- Minimum client stake: 0.05 ETH.
- Slashing ratio: 1.0.
- Reputation decay: 0.9.
- Gas reference: 1.5 gwei and 2500 USD/ETH.

Required attack families:

- free-riding: no training and lazy training;
- Byzantine random noise and model replacement;
- ALIE and MinMax;
- semantic data poisoning;
- Sybil identities;
- adaptive checkpoint interpolation, gradient mimicry, partial replay, and a
  combined adaptive strategy.

Required baselines:

- Vanilla FL;
- Krum, Trimmed Mean, and SDEA;
- ShapleyFL and FoolsGold;
- Veriblock-FL and Kaizen for proof-system overhead;
- FedCoin for incentive comparison.

## Proof-system acceptance targets

- Groth16 proof size: 192 bytes.
- Proof generation: no more than 4.2 seconds in the reference setup.
- Witness computation: no more than 1.8 seconds.
- Prover memory: no more than 2.5 GB.
- Verification time: no more than 8.5 ms for the Groth16 check.
- Merkle proof size: no more than 1.2 KB.
- Total verification path: no more than 52 ms.
- Circuit size: approximately 1.1 million constraints.
- The proof-cost reduction uses 1% gradient sampling, selective checkpoint
  auditing, and incentive-backed enforcement without weakening binding to the
  challenged SGD relation.

## End-to-end acceptance policy

The final implementation is accepted only when:

1. every normative protocol requirement above has a direct code owner and an
   automated test;
2. unit, property, integration, adversarial, contract, and two-GPU tests pass;
3. generated manifests bind source revision, environment, dataset, seed,
   protocol parameters, raw results, and validation outcome;
4. positive metrics (MA, DR, participation, honest profit) meet or exceed the
   paper values;
5. negative metrics (FPR, ASR, runtime, communication, storage, gas, attacker
   profit) meet or improve upon the paper values;
6. no result is accepted from a protocol-incompatible smoke or simulation run;
7. ZK and blockchain claims are backed by real proofs and executed contract
   transitions, not simulated success values.
