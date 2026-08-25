import hashlib
import os
from pathlib import Path
import statistics
import time

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cryptography")

from polbfl.committee import ECDSAPublicKeyRegistry, ECDSASigner, QuorumDecision, ReceiptQuorum, proof_set_digest
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


def _backend():
    raw = os.getenv("POL_ZK_REFERENCE_BUILD")
    prover = os.getenv("RAPIDSNARK_PROVER")
    verifier = os.getenv("RAPIDSNARK_VERIFIER")
    if not raw or not prover or not verifier:
        pytest.skip("reference ZK artifacts and native binaries are not configured")
    build = Path(raw)
    return Groth16Backend(
        Groth16Artifacts(
            wasm=build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
            proving_key=build / "sampled_sgd_reference_final.zkey",
            verification_key=build / "verification_key.json",
            r1cs=build / "sampled_sgd_reference.r1cs",
        ),
        snarkjs_cli=os.getenv("SNARKJS_CLI", "node_modules/snarkjs/cli.js"),
        witness_binary=build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference",
        prover_binary=prover,
        verifier_binary=verifier,
    )


def test_real_training_trace_reference_proof_bundle_and_three_of_five_receipts(tmp_path):
    torch.manual_seed(37)
    model = MiniConvNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    context = RoundContext(
        protocol_version="1",
        round_id="reference-e2e",
        client_id="reference-client",
        model_id="mini-conv-bn-linear",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD(momentum=0x0.0p+0,weight_decay=0x0.0p+0)",
        learning_rate=0.01,
        local_epochs=1,
        batch_size=32,
        checkpoint_interval=5,
        expected_steps=1,
    )
    store = ContentAddressedStore(tmp_path / "evidence")
    config = ZKCircuitConfig()
    recorder = TorchPoLRecorder(
        context,
        store,
        sampling_seed=b"r" * 32,
        gradient_sample_rate=1.0,
        zk_config=config,
        poseidon_bridge=PoseidonBridge(),
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
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
        recorded.trace.commitment,
        vrf_output=b"s" * 32,
        issued_at_ns=4,
        deadline_ns=10_000_000_000,
    )
    backend = _backend()
    bundle = ZKPoLProver(backend, config, store=store).prove_interval(
        recorded=recorded,
        challenge=challenge,
        pair_index=0,
    )
    assert len(bundle.proof.compact_bytes) == 192
    merkle_bytes = len(encode_merkle_proof(bundle.start.merkle_proof)) + len(
        encode_merkle_proof(bundle.end.merkle_proof)
    )
    assert merkle_bytes <= int(1.2 * 1024)

    digest = proof_set_digest(challenge, [bundle])
    signers = [ECDSASigner.generate(f"verifier-{index}") for index in range(5)]
    registry = ECDSAPublicKeyRegistry({signer.verifier_id: signer.public_pem for signer in signers})
    receipts = [
        signer.receipt(challenge, proof_digest=digest, valid=True, verified_at_ns=100)
        for signer in signers[:3]
    ]
    quorum = ReceiptQuorum(
        committee=[signer.verifier_id for signer in signers],
        threshold=3,
        verify_signature=registry.verify,
    )
    reports = []
    decisions = []
    total_paths = []
    verifier = ZKBundleVerifier(backend)
    for _ in range(5):
        started = time.perf_counter()
        reports.append(verifier.verify(context, bundle))
        decisions.append(quorum.decide(challenge, receipts, proof_digest=digest))
        total_paths.append(time.perf_counter() - started)
    assert all(report.valid for report in reports), [report.reasons for report in reports]
    assert all(decision == QuorumDecision.ACCEPT for decision in decisions)
    assert statistics.median(report.verify_seconds for report in reports) <= 0.0085
    assert statistics.median(total_paths) <= 0.052
