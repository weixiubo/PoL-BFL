#!/usr/bin/env python3
"""Execute one attested real-trace Table 11 hardware-pair trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "scripts" / "utils"))

import torch

from experiments.final.adaptive_trial_support import (
    forge_bundle,
    production_backend,
    train_reference,
)
from experiments.final.cross_hardware_profiles import (
    KAIZEN_CONFIG,
    evaluate_numerical_probe,
    profile_for_pair,
)
from experiments.final.hardware_attestation import (
    cross_device_numerical_probe,
    gpu_attestation,
    require_gpu_name,
)
from experiments.final.manifest import (
    create_run_manifest,
    source_identity,
    write_manifest_atomic,
)
from experiments.final.preflight import md5_file
from experiments.final.target_provenance import (
    CROSS_HARDWARE_TARGET_FILES,
    load_merged_targets,
    target_paths,
)
from experiments.final.trust_setup import validate_trust_setup
from experiments.scripts.utils.models import create_model
from polbfl.crypto import domain_hash
from polbfl.protocol import HybridChallengeSampler
from polbfl.storage import ContentAddressedStore
from polbfl.zk import ZKBundleVerifier, ZKCircuitConfig, ZKPoLProver


def _idle(indices: set[int]) -> None:
    for index in indices:
        row = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        utilization, memory = [
            int(value.strip()) for value in row.split(",")
        ]
        if utilization > 10 or memory > 1024:
            raise RuntimeError(
                "formal cross-hardware trial requires idle devices"
            )


def _artifact_files(output: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for directory in (
                output / "worker-results",
                output / "traces",
            )
            if directory.exists()
            for path in directory.rglob("*")
            if path.is_file()
        )
    )


def run_trial(
    args: argparse.Namespace,
    *,
    required_method: str = "PoLBFL",
) -> dict:
    profile = profile_for_pair(ROOT, args.hardware_pair)
    if profile.method != required_method:
        raise ValueError(
            "Table 11 verification profile must use its dedicated runner"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity(ROOT)
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal cross-hardware trials require a clean source"
        )
    trainer_attestation = gpu_attestation(args.trainer_device)
    verifier_attestation = gpu_attestation(args.verifier_device)
    require_gpu_name(trainer_attestation, args.expected_trainer)
    require_gpu_name(verifier_attestation, args.expected_verifier)
    _idle({args.trainer_device, args.verifier_device})
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    build = args.zk_build.resolve()
    toolchain = json.loads(
        (ROOT / "config" / "toolchain.lock.json").read_text(
            encoding="utf-8"
        )
    )
    trust = validate_trust_setup(build=build, toolchain=toolchain)
    if not trust["passed"]:
        raise RuntimeError("cross-hardware trial lacks production trust")
    trust_record = json.loads(
        (build / "trust_setup.json").read_text(encoding="utf-8")
    )
    model = create_model("ResNet18", num_classes=10, input_channels=3)
    global_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    state_path = output / "global-state.pt"
    torch.save({"global_state": global_state}, state_path)
    trained = train_reference(
        output=output,
        global_state_path=state_path,
        data_root=args.data_root.resolve(),
        poseidon_binary=args.poseidon_binary.resolve(),
        seed=args.seed,
        client_index=0,
        local_epochs=5,
        device="cuda:" + str(args.trainer_device),
        pair_tolerance=profile.pair_tolerance,
        final_tolerance=profile.final_tolerance,
    )
    probe = cross_device_numerical_probe(
        trained["fingerprint"].checkpoint_vectors,
        trainer_device=args.trainer_device,
        verifier_device=args.verifier_device,
    )
    numerical_decision = evaluate_numerical_probe(profile, probe)
    if not numerical_decision["passed"]:
        raise RuntimeError("cross-device numerical tolerance was exceeded")

    backend = production_backend(
        root=ROOT,
        build=build,
        icicle_root=args.icicle_root.resolve(),
        rapidsnark_prover=args.rapidsnark_prover.resolve(),
        rapidsnark_verifier=args.rapidsnark_verifier.resolve(),
    )
    try:
        recorded = trained["recorded"]
        issued = time.time_ns()
        challenge = HybridChallengeSampler(
            recent_pairs=2,
            random_pairs=3,
        ).sample(
            trained["commitment"],
            vrf_output=hashlib.sha256(
                (
                    "cross-hardware:"
                    + args.hardware_pair
                    + ":"
                    + str(args.seed)
                ).encode()
            ).digest(),
            issued_at_ns=issued,
            deadline_ns=issued + 600_000_000_000,
        )
        circuit_config = ZKCircuitConfig(
            pair_tolerance=profile.pair_tolerance,
            final_tolerance=profile.final_tolerance,
        )
        bundles = ZKPoLProver(
            backend,
            circuit_config,
            store=ContentAddressedStore(trained["store_root"]),
        ).prove_challenge(
            recorded=recorded,
            challenge=challenge,
        )
        bundle = max(bundles, key=lambda item: item.pair_index)
        verifier = ZKBundleVerifier(backend)
        honest_report = verifier.verify(
            recorded.trace.context,
            bundle,
        )
        malicious_bundle = forge_bundle(
            bundle,
            variant="CombinedAdaptive",
            material_digest=domain_hash(
                "POLBFL_CROSS_HARDWARE_MALICIOUS_V1",
                args.hardware_pair,
                args.seed,
                trained["commitment"].trace_digest,
            ),
        )
        malicious_report = verifier.verify(
            recorded.trace.context,
            malicious_bundle,
        )
    finally:
        backend.close()
    if (
        not honest_report.valid
        or not honest_report.proof_valid
        or malicious_report.valid
        or not malicious_report.proof_valid
        or len(bundle.proof.compact_bytes) != 192
    ):
        raise RuntimeError("cross-hardware proof controls failed")

    honest_accepted = bool(
        honest_report.valid and numerical_decision["passed"]
    )
    malicious_accepted = bool(
        malicious_report.valid and numerical_decision["passed"]
    )
    observations = [
        {
            "hardware_pair": args.hardware_pair,
            "method": profile.method,
            "verification_profile_id": profile.profile_id,
            "client_id": "honest-s" + str(args.seed),
            "behavior": "honest",
            "accepted": honest_accepted,
            "proof_digest": bundle.proof.proof_digest,
            "trainer_attestation": trainer_attestation,
            "verifier_attestation": verifier_attestation,
            "real_trace": True,
            "real_groth16": True,
            "cross_device_probe_digest": probe["probe_digest"],
        },
        {
            "hardware_pair": args.hardware_pair,
            "method": profile.method,
            "verification_profile_id": profile.profile_id,
            "client_id": "malicious-s" + str(args.seed),
            "behavior": "malicious",
            "accepted": malicious_accepted,
            "proof_digest": bundle.proof.proof_digest,
            "trainer_attestation": trainer_attestation,
            "verifier_attestation": verifier_attestation,
            "real_trace": True,
            "real_groth16": True,
            "cross_device_probe_digest": probe["probe_digest"],
        },
    ]
    targets = load_merged_targets(
        ROOT, CROSS_HARDWARE_TARGET_FILES
    )["table_11_cross_hardware"][args.hardware_pair]
    observed = {
        "FPR": 100.0 * float(not honest_accepted),
        "honest_pass_rate": 100.0 * float(honest_accepted),
        "DR": 100.0 * float(not malicious_accepted),
        "block_rate": 100.0 * float(not malicious_accepted),
    }
    checks = {
        "FPR": observed["FPR"] <= float(targets["FPR"]),
        "honest_pass_rate": observed["honest_pass_rate"]
        >= float(targets["honest_pass_rate"]),
        "DR": observed["DR"] >= float(targets["DR"]),
        "block_rate": observed["block_rate"]
        >= float(targets["block_rate"]),
        "numerical_profile": numerical_decision["passed"],
        "proof_size": len(bundle.proof.compact_bytes) == 192,
    }
    evidence = {
        "schema_version": 1,
        "hardware_pair": args.hardware_pair,
        "method": profile.method,
        "verification_profile": profile.to_dict(),
        "seed": args.seed,
        "source_commit": source["commit"],
        "trainer_attestation": trainer_attestation,
        "verifier_attestation": verifier_attestation,
        "cross_device_probe": probe,
        "numerical_decision": numerical_decision,
        "groth16_relation_tolerances": {
            "pair_tolerance": circuit_config.pair_tolerance,
            "final_tolerance": circuit_config.final_tolerance,
        },
        "observed_metrics": observed,
        "trace_digest": trained["commitment"].trace_digest,
        "proof": bundle.proof.to_dict(),
        "proof_bytes": len(bundle.proof.compact_bytes),
        "honest_report": {
            "valid": honest_report.valid,
            "proof_valid": honest_report.proof_valid,
            "reasons": list(honest_report.reasons),
            "verify_seconds": honest_report.verify_seconds,
        },
        "malicious_report": {
            "valid": malicious_report.valid,
            "proof_valid": malicious_report.proof_valid,
            "reasons": list(malicious_report.reasons),
            "verify_seconds": malicious_report.verify_seconds,
        },
        "observations": observations,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "failed": sorted(
                name for name, passed in checks.items() if not passed
            ),
        },
    }
    body = dict(evidence)
    evidence["evidence_digest"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    evidence_path = output / "hardware-evidence.json"
    write_manifest_atomic(evidence_path, evidence)
    result = {
        "study": "cross_hardware",
        "hardware_pair": args.hardware_pair,
        "method": profile.method,
        "verification_profile_id": profile.profile_id,
        "seed": args.seed,
        "source_commit": source["commit"],
        "observations": observations,
        "evidence_digest": evidence["evidence_digest"],
        "formal_accepted": bool(evidence["acceptance"]["passed"]),
    }
    result["result_digest"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    result_path = output / "result.json"
    write_manifest_atomic(result_path, result)
    profile_configuration_files = (
        (ROOT / KAIZEN_CONFIG,)
        if profile.method == "Kaizen"
        else ()
    )
    manifest = create_run_manifest(
        root=ROOT,
        run_id=(
            "formal-table11-"
            + args.hardware_pair.lower()
            + "-s"
            + str(args.seed)
        ),
        seed=args.seed,
        configuration_files=(
            ROOT / "config" / "paper_protocol.json",
            ROOT / "config" / "paper_targets.json",
            *target_paths(ROOT, CROSS_HARDWARE_TARGET_FILES),
            ROOT / "config" / "toolchain.lock.json",
            ROOT / "config" / "baseline_sources.lock.json",
            ROOT / "experiments" / "final" / "paper_matrix.json",
            *profile_configuration_files,
        ),
        dataset={
            "name": "CIFAR10",
            "model": "ResNet18",
            "partition": "IID/50",
            "archive_md5": md5_file(
                args.data_root.resolve()
                / "CIFAR10"
                / "cifar-10-python.tar.gz"
            ),
        },
        artifacts=(
            result_path,
            evidence_path,
            state_path,
            *_artifact_files(output),
        ),
        runtime_artifacts=(
            build / "sampled_sgd_reference.r1cs",
            build / "sampled_sgd_reference_final.zkey",
            build / "verification_key.json",
            build
            / "sampled_sgd_reference_cpp"
            / "sampled_sgd_reference",
            build / "trust_setup.json",
            args.poseidon_binary.resolve(),
            args.icicle_root.resolve() / "bin" / "icicle-snark",
            args.rapidsnark_verifier.resolve(),
        ),
        run_parameters={
            "hardware_pair": args.hardware_pair,
            "method": profile.method,
            "verification_profile": profile.to_dict(),
            "trainer_device": args.trainer_device,
            "verifier_device": args.verifier_device,
            "groth16_relation_pair_tolerance": circuit_config.pair_tolerance,
            "groth16_relation_final_tolerance": circuit_config.final_tolerance,
            "trust_setup_record_digest": trust_record["record_digest"],
        },
        state="completed",
    )
    write_manifest_atomic(output / "manifest.json", manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-pair", required=True)
    parser.add_argument("--trainer-device", type=int, required=True)
    parser.add_argument("--verifier-device", type=int, required=True)
    parser.add_argument("--expected-trainer", required=True)
    parser.add_argument("--expected-verifier", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--zk-build",
        type=Path,
        default=ROOT
        / "circuits"
        / "final"
        / "build"
        / "production",
    )
    parser.add_argument(
        "--poseidon-binary",
        type=Path,
        default=ROOT
        / ".tools"
        / "poseidon-native"
        / "polbfl-poseidon-native",
    )
    parser.add_argument(
        "--icicle-root",
        type=Path,
        default=ROOT / ".tools" / "icicle-snark",
    )
    parser.add_argument(
        "--rapidsnark-prover",
        type=Path,
        default=ROOT
        / ".tools"
        / "rapidsnark"
        / "package"
        / "bin"
        / "prover",
    )
    parser.add_argument(
        "--rapidsnark-verifier",
        type=Path,
        default=ROOT
        / ".tools"
        / "rapidsnark"
        / "package"
        / "bin"
        / "verifier",
    )
    return parser.parse_args()


if __name__ == "__main__":
    completed = run_trial(parse_args())
    print(json.dumps(completed, indent=2, sort_keys=True))
    if completed["formal_accepted"] is not True:
        raise SystemExit(1)
