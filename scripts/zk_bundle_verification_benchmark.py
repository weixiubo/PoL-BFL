#!/usr/bin/env python3
"""Benchmark Merkle-bound Groth16 verification plus a signed 3-of-5 quorum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from experiments.final.manifest import sha256_file, source_identity, write_manifest_atomic
from experiments.final.trust_setup import validate_trust_setup
from polbfl.committee import (
    ECDSAPublicKeyRegistry,
    ECDSASigner,
    QuorumDecision,
    ReceiptQuorum,
    proof_set_digest,
)
from polbfl.crypto import encode_merkle_proof
from polbfl.protocol import HybridChallengeSampler, RoundContext
from polbfl.storage import ContentAddressedStore
from polbfl.training import TorchPoLRecorder
from polbfl.zk import (
    Groth16Artifacts,
    Groth16Backend,
    PoseidonBridge,
    ZKBundleVerifier,
    ZKCircuitConfig,
    ZKPoLProver,
)


class MiniConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(32)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc = torch.nn.Linear(32 * 4 * 4, 8)

    def forward(self, value):
        value = torch.relu(self.bn(self.conv(value)))
        return self.fc(torch.flatten(self.pool(value), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--icicle-root", type=Path, required=True)
    parser.add_argument("--verifier-binary", type=Path, required=True)
    parser.add_argument("--poseidon-binary", type=Path, required=True)
    parser.add_argument(
        "--toolchain",
        type=Path,
        default=ROOT / "config" / "toolchain.lock.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "config" / "paper_targets.json",
    )
    parser.add_argument("--verify-repeats", type=int, default=10)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.verify_repeats <= 0:
        raise ValueError("verification repeats must be positive")
    build = args.build.resolve()
    icicle_root = args.icicle_root.resolve()
    verifier_binary = args.verifier_binary.resolve()
    poseidon_binary = args.poseidon_binary.resolve()
    source = source_identity(ROOT)
    if not args.allow_dirty and (source["dirty"] or not source["commit"]):
        raise RuntimeError("formal bundle benchmark requires a clean source commit")
    toolchain = json.loads(args.toolchain.read_text(encoding="utf-8"))
    trust = validate_trust_setup(build=build, toolchain=toolchain)
    if not trust["passed"]:
        raise RuntimeError("bundle benchmark requires the production trust setup")
    for label, expected in toolchain["icicle_snark"]["artifacts"].items():
        path = icicle_root / label
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"bundle benchmark ICICLE artifact is not locked: {path}")
    if sha256_file(verifier_binary) != toolchain["rapidsnark"][
        "linux_x86_64_verifier_sha256"
    ]:
        raise RuntimeError("bundle benchmark verifier is not locked")
    if sha256_file(poseidon_binary) != toolchain["native_poseidon"][
        "linux_x86_64_sha256"
    ]:
        raise RuntimeError("bundle benchmark Poseidon binary is not locked")

    backend = Groth16Backend(
        Groth16Artifacts(
            wasm=build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
            proving_key=build / "sampled_sgd_reference_final.zkey",
            verification_key=build / "verification_key.json",
            r1cs=build / "sampled_sgd_reference.r1cs",
        ),
        snarkjs_cli=ROOT / "node_modules" / "snarkjs" / "cli.js",
        witness_binary=build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference",
        verifier_binary=verifier_binary,
        icicle_binary=icicle_root / "bin" / "icicle-snark",
        icicle_backend_directory=icicle_root / "backend",
        icicle_library_directories=(
            icicle_root / "lib",
            icicle_root / "backend" / "cuda",
        ),
        icicle_devices=(0,),
        timeout_seconds=300,
    )
    torch.manual_seed(37)
    torch.use_deterministic_algorithms(True)
    model = MiniConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    context = RoundContext(
        protocol_version="1",
        round_id="production-bundle-benchmark",
        client_id="benchmark-client",
        model_id="mini-conv-bn-linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0x0.0p+0,weight_decay=0x0.0p+0)",
        learning_rate=0.01,
        local_epochs=1,
        batch_size=32,
        checkpoint_interval=5,
        expected_steps=1,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="polbfl-bundle-benchmark-") as directory:
            store = ContentAddressedStore(Path(directory) / "evidence")
            config = ZKCircuitConfig()
            poseidon = PoseidonBridge(native_binary=poseidon_binary, persistent=True)
            try:
                recorder = TorchPoLRecorder(
                    context,
                    store,
                    sampling_seed=b"r" * 32,
                    gradient_sample_rate=0.01,
                    zk_config=config,
                    poseidon_bridge=poseidon,
                )
                recorder.start(model=model, optimizer=optimizer, timestamp_ns=1)
                data = torch.randn(2, 1, 8, 8)
                labels = torch.tensor([2, 5])
                optimizer.zero_grad(set_to_none=True)
                logits = model(data)
                torch.nn.functional.cross_entropy(logits, labels).backward()
                optimizer.step()
                recorder.record_optimizer_step(
                    step=1,
                    epoch=0,
                    model=model,
                    optimizer=optimizer,
                    batch_data=data,
                    batch_labels=labels,
                    batch_indices=(21, 22),
                    activations={"logits": logits.detach()},
                    timestamp_ns=2,
                )
                recorded = recorder.finalize(timestamp_ns=3)
            finally:
                poseidon.close()
            challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
                recorded.trace.commitment,
                vrf_output=b"s" * 32,
                issued_at_ns=4,
                deadline_ns=10_000_000_000,
            )
            bundle = ZKPoLProver(backend, config, store=store).prove_interval(
                recorded=recorded,
                challenge=challenge,
                pair_index=0,
            )
            merkle_bytes = len(encode_merkle_proof(bundle.start.merkle_proof)) + len(
                encode_merkle_proof(bundle.end.merkle_proof)
            )
            proof_digest = proof_set_digest(challenge, [bundle])
            signers = [ECDSASigner.generate(f"verifier-{index}") for index in range(5)]
            registry = ECDSAPublicKeyRegistry(
                {signer.verifier_id: signer.public_pem for signer in signers}
            )
            receipts = [
                signer.receipt(
                    challenge,
                    proof_digest=proof_digest,
                    valid=True,
                    verified_at_ns=100,
                )
                for signer in signers[:3]
            ]
            quorum = ReceiptQuorum(
                committee=[signer.verifier_id for signer in signers],
                threshold=3,
                verify_signature=registry.verify,
            )
            verifier = ZKBundleVerifier(backend)
            total_paths = []
            verification_seconds = []
            for _ in range(args.verify_repeats):
                started = time.perf_counter()
                report = verifier.verify(context, bundle)
                decision = quorum.decide(
                    challenge,
                    receipts,
                    proof_digest=proof_digest,
                )
                total_paths.append(time.perf_counter() - started)
                verification_seconds.append(report.verify_seconds)
                if not report.valid or decision != QuorumDecision.ACCEPT:
                    raise RuntimeError("bundle or signed quorum verification failed")
    finally:
        backend.close()

    metrics = {
        "merkle_proof_kb": merkle_bytes / 1024,
        "total_verification_ms": 1000.0 * statistics.median(total_paths),
        "verification_ms": 1000.0 * statistics.median(verification_seconds),
        "proof_bytes": len(bundle.proof.compact_bytes),
    }
    targets = json.loads(args.targets.read_text(encoding="utf-8"))["table_12_zk"]
    checks = {
        "merkle_proof_kb": metrics["merkle_proof_kb"] <= float(targets["merkle_proof_kb"]),
        "total_verification_ms": metrics["total_verification_ms"]
        <= float(targets["total_verification_ms"]),
        "verification_ms": metrics["verification_ms"] <= float(targets["verification_ms"]),
        "proof_bytes": metrics["proof_bytes"] <= int(targets["proof_bytes"]),
    }
    result = {
        "schema_version": 1,
        "method": "PoLBFL",
        "proof_system": "Groth16",
        "real_benchmark": True,
        "metrics": metrics,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed": sorted(name for name, passed in checks.items() if not passed),
        },
        "source": source,
        "source_commit": source["commit"],
        "trust_setup_record_digest": json.loads(
            (build / "trust_setup.json").read_text(encoding="utf-8")
        )["record_digest"],
        "input_sha256": {
            "trust_setup.json": sha256_file(build / "trust_setup.json"),
            "verification_key.json": sha256_file(build / "verification_key.json"),
            "icicle-snark": sha256_file(icicle_root / "bin" / "icicle-snark"),
            "rapidsnark-verifier": sha256_file(verifier_binary),
            "poseidon-native": sha256_file(poseidon_binary),
        },
        "proof_digest": bundle.proof.proof_digest,
        "proof_set_digest": proof_digest,
        "receipt_digests": [receipt.receipt_digest for receipt in receipts],
    }
    result["formal_accepted"] = bool(
        not args.allow_dirty and result["acceptance"]["passed"]
    )
    body = dict(result)
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_manifest_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["acceptance"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    os.environ.setdefault("POL_INTEGRITY", "1")
    main()
