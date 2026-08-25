# Final paper experiment suite

This directory is the execution entry point for the final submitted paper's
experiment matrix. Scalar and PDF-extracted multi-method targets live under
`config/paper*_targets.json`; the run matrix lives in `paper_matrix.json`.

Every accepted run must retain:

- the exact source commit and dirty-tree state;
- the authoritative PDF digest and protocol/target configuration digests;
- Python, PyTorch, CUDA, cuDNN, Circom, snarkjs, ICICLE-Snark, rapidsnark,
  Solidity and Ganache versions;
- CPU, RAM, GPU model/UUID/driver, operating system and container information;
- dataset identity, download/source checksum, partition indices and transform
  configuration;
- seed, client population, malicious population and every per-round selected
  client;
- raw predictions, update decisions, proof/receipt digests, timing, memory,
  communication, storage and transaction receipts;
- an explicit validation report against the paper target direction.

Smoke, shortened, synthetic, replayed, calibrated, or protocol-incompatible
runs are written under a separate diagnostic namespace and cannot populate an
accepted paper cell.

The suite uses all 50 clients, 10 fixed malicious clients, 200 rounds, five
local epochs, learning rate 0.01 and batch size 32. CIFAR experiments use IID
partitioning unless the non-IID table is being evaluated. FEMNIST uses the
natural writer partition.

All 50 clients participate in every round, following Algorithm 1's explicit
all-client loop. The audit set contains 20% of those post-commitment clients.

Before a formal cell:

```bash
POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m experiments.final.preflight \
  --paper /absolute/path/to/main.pdf
```

The pinned native Poseidon and ICICLE-Snark helpers are built with
`tools/poseidon_native/Cargo.toml` and `scripts/build_icicle_snark.sh`. Preflight
rejects any helper or shared library whose SHA-256 differs from
`config/toolchain.lock.json`.

The first security cell is launched with:

```bash
POL_INTEGRITY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 \
python -u -m experiments.final.run_security_cell \
  --dataset CIFAR10 --attack FreeRidingNT --method PoLBFL --seed 1337 \
  --run-id formal-cifar10-freeridingnt-polbfl-s1337 \
  --output experiments/results/final/formal-cifar10-freeridingnt-polbfl-s1337 \
  --process-training --train-processes-per-gpu 8 --proof-workers 8
```

Each completed round atomically replaces `checkpoint.pt`. A stopped process is
continued with the identical command plus `--resume`; source-commit mismatch
is rejected before state restoration.

`run_matrix.py` plans all 432 Table 2 cells and prioritizes PoL-BFL before the
five public baselines. `run_table4_matrix.py` plans both Standalone and
PoLBFLPrefilter modes. The non-IID, scalability and spot-check runners use the
same atomic-resume and source-cleanliness gates.

For PoL-BFL, `rounds.jsonl` contains the complete test predictions and labels,
their domain-separated digest, selected audit clients, individual Groth16
proof digests and sizes, proof-set digest, three signed receipt digests, timing,
and hashes of every retained content-addressed evidence file. After the round
checkpoint and JSONL record are durable, worker result files and unselected
private traces are removed. Only selected clients' replayable stores remain.
Formal evidence capture re-hashes those stores and recomputes every round's
accuracy from the raw prediction trace.

Every formal PoL-BFL cell also replays all measured round decisions through
the optimized Solidity protocol on a real local EVM. Python and Solidity use
the same commitment-bound SHA-256 audit ticket, and the replay checks the
selected audit set, 3-of-5 signed decisions, settlement, stake, and reputation
state for every round. Table 3 is executed by `run_layer_matrix.py`; its L1,
L1+L2, L1+L3, and Full profiles explicitly gate robust aggregation,
Sybil/reputation processing, and persistent economic enforcement.
