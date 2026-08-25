import json
import os
import subprocess
from pathlib import Path

import pytest

from polbfl.zk import IcicleProverPool


def test_icicle_workers_are_recycled_after_bounded_commands(tmp_path):
    worker_binary = tmp_path / "fake-icicle-worker"
    worker_binary.write_text(
        """#!/usr/bin/env python3
import json
import shlex
import sys
from pathlib import Path

for line in sys.stdin:
    arguments = shlex.split(line)
    if not arguments:
        continue
    if arguments[0] == "exit":
        break
    proof = Path(arguments[arguments.index("--proof") + 1])
    public = Path(arguments[arguments.index("--public") + 1])
    proof.write_text(json.dumps({"pi_a": ["1"]}), encoding="utf-8")
    public.write_text(json.dumps(["7"]), encoding="utf-8")
    print("COMMAND_COMPLETED", flush=True)
""",
        encoding="utf-8",
    )
    worker_binary.chmod(0o755)
    backend = tmp_path / "backend"
    libraries = tmp_path / "libraries"
    backend.mkdir()
    libraries.mkdir()
    witness = tmp_path / "witness.wtns"
    proving_key = tmp_path / "proving.zkey"
    witness.write_bytes(b"witness")
    proving_key.write_bytes(b"proving-key")
    pool = IcicleProverPool(
        worker_binary,
        backend_directory=backend,
        library_directories=(libraries,),
        devices=(0,),
        timeout_seconds=10,
        max_commands_per_worker=2,
    )
    initial_pid = pool.worker_pids[0]
    try:
        for index in range(3):
            proof, public = pool.prove(
                witness=witness,
                proving_key=proving_key,
                proof=tmp_path / f"proof-{index}.json",
                public=tmp_path / f"public-{index}.json",
            )
            assert proof == {"pi_a": ["1"]}
            assert public == ("7",)
            if index == 0:
                assert pool.worker_pids == (initial_pid,)
            elif index == 1:
                assert pool.worker_pids[0] != initial_pid
                recycled_pid = pool.worker_pids[0]
            else:
                assert pool.worker_pids == (recycled_pid,)
    finally:
        pool.close()
    assert pool.worker_pids == ()


def test_icicle_worker_proof_is_accepted_by_locked_rapidsnark(tmp_path):
    binary = os.getenv("ICICLE_SNARK_BINARY")
    backend = os.getenv("ICICLE_BACKEND_DIRECTORY")
    libraries = os.getenv("ICICLE_LIBRARY_DIRECTORIES")
    build = os.getenv("POL_ZK_REFERENCE_BUILD")
    verifier = os.getenv("RAPIDSNARK_VERIFIER")
    if not all((binary, backend, libraries, build, verifier)):
        pytest.skip("ICICLE-Snark and reference Groth16 artifacts are required")
    build_path = Path(build)
    pool = IcicleProverPool(
        binary,
        backend_directory=backend,
        library_directories=libraries.split(os.pathsep),
        devices=(int(os.getenv("ICICLE_TEST_DEVICE", "0")),),
        timeout_seconds=300,
        max_commands_per_worker=1,
    )
    assert len(pool.worker_pids) == 1
    initial_pid = pool.worker_pids[0]
    proof_path = tmp_path / "proof.json"
    public_path = tmp_path / "public.json"
    second_proof_path = tmp_path / "proof-2.json"
    second_public_path = tmp_path / "public-2.json"
    try:
        proof, public = pool.prove(
            witness=build_path / "benchmark.wtns",
            proving_key=build_path / "sampled_sgd_reference_final.zkey",
            proof=proof_path,
            public=public_path,
        )
        recycled_pid = pool.worker_pids[0]
        assert recycled_pid != initial_pid
        second_proof, second_public = pool.prove(
            witness=build_path / "benchmark.wtns",
            proving_key=build_path / "sampled_sgd_reference_final.zkey",
            proof=second_proof_path,
            public=second_public_path,
        )
        assert pool.worker_pids[0] != recycled_pid
    finally:
        pool.close()
    assert pool.worker_pids == ()
    assert proof == json.loads(proof_path.read_text(encoding="utf-8"))
    assert public == tuple(
        str(value) for value in json.loads(public_path.read_text(encoding="utf-8"))
    )
    assert second_proof != proof
    assert second_public == public
    verified = subprocess.run(
        [
            verifier,
            str(build_path / "verification_key.json"),
            str(public_path),
            str(proof_path),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert verified.returncode == 0
    assert "Valid proof" in verified.stdout + verified.stderr
    second_verified = subprocess.run(
        [
            verifier,
            str(build_path / "verification_key.json"),
            str(second_public_path),
            str(second_proof_path),
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert second_verified.returncode == 0
    assert "Valid proof" in second_verified.stdout + second_verified.stderr
