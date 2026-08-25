# Reproducing the final paper experiments

The accepted workflow is rooted at `experiments/final/`. The authoritative
paper digest, protocol configuration, target tables, experiment matrix, source
revision, tool binaries, datasets, partitions, seeds, raw round observations,
and final artifacts are cryptographically bound into each run manifest.

## 1. Build and verify native tools

Build the Circom-compatible Poseidon helper and the pinned ICICLE-Snark CUDA
backend:

```bash
cargo build --release --locked --manifest-path tools/poseidon_native/Cargo.toml
install -d -m 0755 .tools/poseidon-native
install -m 0755 \
  tools/poseidon_native/target/release/polbfl-poseidon-native \
  .tools/poseidon-native/polbfl-poseidon-native

bash scripts/build_icicle_snark.sh
```

Build the reference Circom circuit and Groth16 artifacts using a verified
phase-2 Powers-of-Tau file. Development CRS artifacts are benchmark-only and
cannot populate formal results.

The formal prover consumes the same BN254 `.zkey` and `.wtns` formats as
Rapidsnark. Every CUDA-generated proof is independently checked by the locked
Rapidsnark verifier before committee receipts are issued.

## 2. Preflight

Run preflight with the final submitted PDF and canonical datasets:

```bash
POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m experiments.final.preflight \
  --paper /absolute/path/to/main.pdf \
  --data-root /absolute/path/to/data \
  --zk-build /absolute/path/to/circuits/final/build/production
```

Preflight fails closed on a paper, dataset, source, tool, circuit, proving key,
verifying key, GPU, deterministic-runtime, contract-runtime, or artifact-hash
mismatch. Both RTX 4090 GPUs must be idle for a formal run.

## 3. Formal security cell

The paper configuration uses all 50 clients per round, 10 malicious clients,
200 rounds, five local epochs, batch size 32, learning rate 0.01, a 20% audit
set, and real Groth16 proofs:

```bash
POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 \
python -u -m experiments.final.run_security_cell \
  --dataset CIFAR10 \
  --attack FreeRidingNT \
  --method PoLBFL \
  --seed 1337 \
  --run-id formal-cifar10-freeridingnt-polbfl-s1337 \
  --output experiments/results/final/formal-cifar10-freeridingnt-polbfl-s1337 \
  --data-root /absolute/path/to/data \
  --zk-build /absolute/path/to/circuits/final/build/production \
  --process-training \
  --train-processes-per-gpu 8 \
  --proof-workers 8
```

For a shared server, wrap the identical command with
`scripts/gpu_idle_supervisor.py`. It waits for both GPUs to remain idle,
restarts only retryable resource failures, and appends `--resume` when a
source-compatible checkpoint exists.

Each completed round atomically writes `checkpoint.pt` and appends one raw JSON
record to `rounds.jsonl`. Resume is rejected if the source revision differs.
The raw record contains predictions, labels, update decisions, proof and receipt
digests, and hashes of retained audited traces. Once it and the checkpoint are
durable, worker payloads and unselected traces are pruned. The final
`result.json` reports paper-aligned mean time per round, communication per
round, maximum client storage, MA, DR, FPR, total wall time, and an explicit
formal acceptance gate.

## 4. Result acceptance

Shortened, synthetic, replayed, proof-disabled, PoL-disabled, busy-GPU, or
otherwise protocol-incompatible runs belong under
`experiments/results/diagnostic/`. They cannot be aggregated into a paper cell.

Aggregate three accepted seeds only:

```bash
python -m experiments.final.aggregate_table2 \
  experiments/results/final/*/result.json \
  --output experiments/results/final/table-2-observed.json
```

Validate the observed table in the required direction:

```bash
python -m experiments.final.validate_targets \
  experiments/results/final/table-2-observed.json \
  --targets config/paper_table2_all_methods.json \
  --table table_2_all_methods \
  --output experiments/results/final/table-2-validation.json
```

MA, DR, participation, and honest profit must meet or exceed the paper.
FPR, ASR, runtime, communication, storage, gas, and malicious profit must meet
or improve on the corresponding upper bound. Missing cells or failed formal
gates are rejected rather than imputed.
