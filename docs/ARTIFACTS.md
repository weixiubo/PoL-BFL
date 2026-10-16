# Research Artifacts

The [research release](https://github.com/weixiubo/PoL-BFL/releases/tag/research-artifacts-2026-10-16)
contains source versions, paper sources, experimental records, checkpoints,
dataset inputs and public cryptographic parameters associated with PoL-BFL.
The maintained implementation remains in the repository's main branch.

## Download and verification

```bash
git clone https://github.com/weixiubo/PoL-BFL.git
cd PoL-BFL
python3 scripts/artifacts.py restore --root "$HOME/PoL-BFL-artifacts"
python3 scripts/artifacts.py verify --root "$HOME/PoL-BFL-artifacts"
```

Python's standard library and Git are sufficient for downloading and verifying
the artifacts. No GitHub login is required. The release uses parts of at most
1 GiB. The catalog fixes each part's size and SHA-256. The file manifest also
checks individual contents, permissions and relative links.

| Directory | Contents |
|---|---|
| `data/` | Experimental measurements, configurations, checkpoints and recorded runs |
| `paper/` | Formal manuscript, editable figures, poster and paper revisions |
| `source-history/` | Research Git repository and source-version records |
| `source-snapshots/` | Recorded implementation snapshots |
| `datasets/` | Dataset inputs for local preparation and execution |
| `zk-parameters/` | Recorded public proving parameters |
| `reference-implementations/` | Experimental framework and comparison implementations |
| `provenance/` | Source and dataset identity records |

The historical repository is restored as `source-history/research.git`.
Its references and complete object storage are checked independently. Commit
and reference maps retain correspondence with the recorded source versions.

Use a separate destination for each archive edition. Repeated downloads
reuse verified contents. Existing modified files or Git
references cause an error rather than an overwrite. Verification uses no
network. Offline restoration from a previously downloaded set of parts is also
supported:

```bash
python3 scripts/artifacts.py restore --root "$HOME/PoL-BFL-artifacts-offline" \
  --cache-dir /path/to/downloaded-parts --offline
```

The `.artifact-cache` directory contains reusable download data and may be
removed after successful verification. Source exports and their checksums are
also available in the release as `polbfl-source.tar.gz`, `polbfl-source.bundle`,
`ARCHIVE_INDEX.json` and `SHA256SUMS`.

## CPU execution

Python 3.13.2 and Node.js 24.14.0 are the tested runtime versions. Install CPU
dependencies into a project environment:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-cpu.txt
npm ci
python examples/cpu_recovery.py --output /path/to/new-cpu-run
python -m pytest -q
```

The CPU example checks checkpoint continuation and exact SGD transition replay
on synthetic inputs. The MNIST-shaped example also uses synthetic inputs and
demonstrates training and checkpoint generation.

The CUDA dependency set remains in `requirements-final.txt`. GPU experiments,
native proving components and formal circuit setup are described in
[REPRODUCING.md](REPRODUCING.md). Node 24 uses Ganache's JavaScript transport
fallback; performance comparisons require the environment recorded by each run.

## Resource configuration

```bash
. "$HOME/PoL-BFL-artifacts/resources.env"
```

The resource configuration selects the restored dataset directory explicitly.
FEMNIST is supplied as its parquet preparation input. Complete MNIST archives
include the upstream checksums. An incomplete historical MNIST download is
preserved with its associated records; the complete input is under `datasets/`.
CIFAR downloads and FEMNIST conversion are described in [DATASETS.md](DATASETS.md).

The historical `parameter_update` circuit and proving parameters correspond to
that circuit version. `POLBFL_LEGACY_ZK_BUILD` identifies its recorded build.
The sampled-SGD production circuit requires its own build and an explicit
`POLBFL_ZK_BUILD` setting, as documented in [ZKP_AND_BLOCKCHAIN.md](ZKP_AND_BLOCKCHAIN.md).

## Recorded results

The formal paper and subsequent reproduction runs are versioned separately.
[EXPERIMENT_RECORDS.md](EXPERIMENT_RECORDS.md) maps the archived record groups
to the existing paper-to-code documentation. Raw measurements, recorded failures,
run plans, and values extracted from the paper retain their respective meanings.
Downloading and verifying the archive does not rerun the paper's full experiment
matrix.
