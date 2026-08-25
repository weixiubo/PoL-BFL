import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from polbfl.protocol import HybridChallengeSampler, PoLTraceBuilder, RoundContext
from polbfl.crypto import hash_batch
from polbfl.zk import (
    PUBLIC_SIGNAL_NAMES,
    Groth16Artifacts,
    Groth16Backend,
    ZKBundleVerifier,
    ZKCheckpointOpening,
    ZKIntervalBundle,
    digest_to_field,
    interval_batch_commitment,
    protocol_binding_field,
)


def _backend_input_trace_and_challenge(tmp_path):
    build_raw = os.getenv("POL_ZK_SMOKE_BUILD")
    if not build_raw:
        pytest.skip("POL_ZK_SMOKE_BUILD is not configured")
    build = Path(build_raw)
    backend = Groth16Backend(
        Groth16Artifacts(
            wasm=build / "sampled_sgd_smoke_js" / "sampled_sgd_smoke.wasm",
            proving_key=build / "sampled_sgd_smoke_final.zkey",
            verification_key=build / "verification_key.json",
            r1cs=build / "sampled_sgd_smoke.r1cs",
        ),
        snarkjs_cli=os.getenv("SNARKJS_CLI", "node_modules/snarkjs/cli.js"),
    )
    context = RoundContext(
        protocol_version="1",
        round_id="zk-round",
        client_id="zk-client",
        model_id="zk-smoke",
        global_model_digest=hashlib.sha256(b"global").hexdigest(),
        optimizer="SGD",
        learning_rate=0.1,
        local_epochs=1,
        batch_size=2,
        checkpoint_interval=2,
    )
    input_path = tmp_path / "input.json"
    generator = Path(__file__).parents[1] / "circuits" / "final" / "generate_smoke_input.cjs"
    subprocess.run(
        [
            os.getenv("NODE_BINARY", "node"),
            str(generator),
            str(input_path),
            str(digest_to_field(context.digest)),
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    circuit_input = json.loads(input_path.read_text(encoding="utf-8"))
    step_evidence = [
        {"step": 1, "batch_digest": hashlib.sha256(b"private-step-1").hexdigest()},
        {"step": 2, "batch_digest": hash_batch([[1, 2], [3, 4]], [0, 1])},
    ]
    batch_commitment = interval_batch_commitment(step_evidence)
    circuit_input["batchCommitmentHash"] = str(digest_to_field(batch_commitment))
    start_aux = {
        "zk": {
            "context_hash": circuit_input["contextHash"],
            "sample_plan_hash": circuit_input["samplePlanHash"],
            "sampled_weights_hash": circuit_input["startWeightsHash"],
        }
    }
    end_aux = {
        "step_evidence": step_evidence,
        "zk": {
            "context_hash": circuit_input["contextHash"],
            "sample_plan_hash": circuit_input["samplePlanHash"],
            "sampled_weights_hash": circuit_input["endWeightsHash"],
            "interval": {
                "gradients_hash": circuit_input["gradientsHash"],
                "data_indices_hash": circuit_input["dataIndicesHash"],
                "auxiliary_hash": circuit_input["auxiliaryHash"],
                "scale": circuit_input["scale"],
                "learning_rate": circuit_input["learningRate"],
                "max_distance_squared": circuit_input["maxDistanceSquared"],
                "max_rounding_error": circuit_input["maxRoundingError"],
                "max_cumulative_rounding_error_squared": circuit_input[
                    "maxCumulativeRoundingErrorSquared"
                ],
                "batch_commitment_hash": circuit_input["batchCommitmentHash"],
                "active_step_count": circuit_input["activeStepCount"],
            },
        }
    }
    builder = PoLTraceBuilder(context)
    builder.append_checkpoint(
        step=0,
        epoch=0,
        timestamp_ns=1,
        model_state={"w": [10, 20, 30, 40]},
        batch_data=[],
        batch_labels=[],
        batch_indices=[],
        auxiliary=start_aux,
    )
    builder.append_checkpoint(
        step=2,
        epoch=0,
        timestamp_ns=2,
        model_state={"w": [7, 17, 25, 35]},
        batch_data=[[1, 2], [3, 4]],
        batch_labels=[0, 1],
        batch_indices=[1, 2, 3, 4],
        auxiliary=end_aux,
    )
    trace = builder.finalize()
    challenge = HybridChallengeSampler(recent_pairs=1, random_pairs=0).sample(
        trace.commitment,
        vrf_output=b"u" * 32,
        issued_at_ns=10,
        deadline_ns=20,
        proof_mode="zk",
    )
    circuit_input["commitmentRootHash"] = str(digest_to_field(trace.commitment.merkle_root))
    circuit_input["challengeHash"] = str(digest_to_field(challenge.challenge_id))
    circuit_input["pairIndex"] = "0"
    circuit_input["protocolBindingHash"] = str(
        protocol_binding_field(
            context_digest=context.digest,
            commitment_root=trace.commitment.merkle_root,
            challenge_id=challenge.challenge_id,
            pair_index=0,
            batch_commitment=batch_commitment,
        )
    )
    return backend, circuit_input, trace, challenge, start_aux, end_aux


def test_real_zk_bundle_composes_groth16_with_trace_membership(tmp_path):
    backend, circuit_input, trace, challenge, start_aux, end_aux = _backend_input_trace_and_challenge(tmp_path)
    proof = backend.prove(circuit_input)
    assert dict(zip(PUBLIC_SIGNAL_NAMES, proof.public_signals))["contextHash"] == circuit_input["contextHash"]
    bundle = ZKIntervalBundle(
        challenge=challenge,
        commitment=trace.commitment,
        pair_index=0,
        start=ZKCheckpointOpening(0, trace.checkpoints[0], trace.checkpoint_proof(0), start_aux),
        end=ZKCheckpointOpening(1, trace.checkpoints[1], trace.checkpoint_proof(1), end_aux),
        proof=proof,
        uploaded_final_model_digest=trace.commitment.final_model_digest,
    )
    report = ZKBundleVerifier(backend).verify(trace.context, bundle)
    assert report.valid, report.reasons

    tampered_end = {**end_aux, "zk": {**end_aux["zk"], "sample_plan_hash": "1"}}
    tampered = ZKIntervalBundle(
        challenge=bundle.challenge,
        commitment=bundle.commitment,
        pair_index=bundle.pair_index,
        start=bundle.start,
        end=ZKCheckpointOpening(1, trace.checkpoints[1], trace.checkpoint_proof(1), tampered_end),
        proof=bundle.proof,
        uploaded_final_model_digest=bundle.uploaded_final_model_digest,
    )
    report = ZKBundleVerifier(backend).verify(trace.context, tampered)
    assert not report.valid
    assert "end_checkpoint_not_bound" in report.reasons
