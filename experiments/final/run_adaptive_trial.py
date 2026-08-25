#!/usr/bin/env python3
"""Run one real-trace, real-Groth16 final-paper adaptive trial."""

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
    REFERENCE_TRAJECTORIES,
    VARIANTS,
    adaptive_profit,
    build_adaptive_material,
    forge_bundle,
    production_backend,
    train_reference,
)
from experiments.final.manifest import (
    create_run_manifest,
    source_identity,
    write_manifest_atomic,
)
from experiments.final.preflight import md5_file
from experiments.final.trust_setup import validate_trust_setup
from experiments.scripts.utils.models import create_model
from polbfl.protocol import HybridChallengeSampler
from polbfl.storage import ContentAddressedStore
from polbfl.zk import ZKBundleVerifier, ZKCircuitConfig, ZKPoLProver


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _require_reference_hardware() -> None:
    if torch.cuda.device_count() != 2 or any(
        "4090" not in torch.cuda.get_device_name(index)
        for index in range(2)
    ):
        raise RuntimeError("adaptive trials require exactly two RTX 4090 GPUs")
    usage = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    ).stdout.splitlines()
    parsed = [
        tuple(int(value.strip()) for value in line.split(","))
        for line in usage
    ]
    if any(utilization > 10 or memory > 1024 for utilization, memory in parsed):
        raise RuntimeError(
            "adaptive trials require idle reference GPUs: " + str(parsed)
        )


def _reference(
    args: argparse.Namespace,
    *,
    output: Path,
    state_path: Path,
    client_index: int,
    local_epochs: int = 5,
):
    return train_reference(
        output=output,
        global_state_path=state_path,
        data_root=args.data_root.resolve(),
        poseidon_binary=args.poseidon_binary.resolve(),
        seed=args.seed,
        client_index=client_index,
        local_epochs=local_epochs,
        device="cuda:" + str(client_index % 2),
    )


def _proof_evidence(
    *,
    args: argparse.Namespace,
    honest,
    material_digest: str,
):
    build = args.zk_build.resolve()
    backend = production_backend(
        root=ROOT,
        build=build,
        icicle_root=args.icicle_root.resolve(),
        rapidsnark_prover=args.rapidsnark_prover.resolve(),
        rapidsnark_verifier=args.rapidsnark_verifier.resolve(),
    )
    try:
        recorded = honest["recorded"]
        issued = time.time_ns()
        challenge = HybridChallengeSampler(
            recent_pairs=2,
            random_pairs=3,
        ).sample(
            honest["commitment"],
            vrf_output=hashlib.sha256(
                ("adaptive-challenge:" + str(args.seed)).encode()
            ).digest(),
            issued_at_ns=issued,
            deadline_ns=issued + 600_000_000_000,
        )
        bundles = ZKPoLProver(
            backend,
            ZKCircuitConfig(),
            store=ContentAddressedStore(honest["store_root"]),
        ).prove_challenge(
            recorded=recorded,
            challenge=challenge,
        )
        final_bundle = max(bundles, key=lambda bundle: bundle.pair_index)
        verifier = ZKBundleVerifier(backend)
        honest_report = verifier.verify(
            recorded.trace.context,
            final_bundle,
        )
        if not honest_report.valid or not honest_report.proof_valid:
            raise RuntimeError("honest adaptive control proof was rejected")
        forged = forge_bundle(
            final_bundle,
            variant=args.variant,
            material_digest=material_digest,
        )
        malicious_report = verifier.verify(
            recorded.trace.context,
            forged,
        )
        if not malicious_report.proof_valid:
            raise RuntimeError(
                "adaptive binding test did not execute real Groth16 verification"
            )
        return bundles, final_bundle, honest_report, malicious_report
    finally:
        backend.close()


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


def run_trial(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = source_identity(ROOT)
    if source["dirty"] or not source["commit"]:
        raise RuntimeError(
            "formal adaptive trials require a clean, identified source"
        )
    _require_reference_hardware()
    _seed(args.seed)
    build = args.zk_build.resolve()
    toolchain = json.loads(
        (ROOT / "config" / "toolchain.lock.json").read_text(
            encoding="utf-8"
        )
    )
    trust = validate_trust_setup(build=build, toolchain=toolchain)
    if not trust["passed"]:
        raise RuntimeError("adaptive trial lacks production trust setup")
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
    honest = _reference(
        args,
        output=output,
        state_path=state_path,
        client_index=0,
    )
    honest_training_seconds = (
        float(honest["timings"]["local_training_seconds"])
        + float(honest["timings"]["trace_finalize_seconds"])
    )

    forge_started = time.perf_counter()
    references = []
    reference_count = REFERENCE_TRAJECTORIES[args.variant]
    for offset in range(reference_count):
        local_epochs = (
            2
            if args.variant == "PartialReplay"
            and offset == reference_count - 1
            else 5
        )
        references.append(
            _reference(
                args,
                output=output,
                state_path=state_path,
                client_index=offset + 1,
                local_epochs=local_epochs,
            )
        )
    measured_attack, material_digest = build_adaptive_material(
        args.variant,
        honest=honest,
        references=references,
        global_state=global_state,
        seed=args.seed,
    )
    (
        bundles,
        final_bundle,
        honest_report,
        malicious_report,
    ) = _proof_evidence(
        args=args,
        honest=honest,
        material_digest=material_digest,
    )
    forge_seconds = time.perf_counter() - forge_started
    proof_seconds = sum(
        float(bundle.proof.prove_seconds)
        + float(bundle.proof.witness_seconds)
        for bundle in bundles
    ) + float(honest_report.verify_seconds)
    honest_reference_seconds = honest_training_seconds + proof_seconds
    if forge_seconds <= 0 or honest_reference_seconds <= 0:
        raise RuntimeError("adaptive timing evidence is non-positive")
    ratio = forge_seconds / honest_reference_seconds

    targets = json.loads(
        (ROOT / "config" / "paper_targets.json").read_text(
            encoding="utf-8"
        )
    )
    economics = targets["table_6_profit_usd"]
    malicious_economics = economics["MaliciousNT"]
    malicious_profit = adaptive_profit(
        reward=float(malicious_economics["reward"]),
        base_cost=abs(float(malicious_economics["cost"])),
        slash=abs(float(malicious_economics["slash"])),
        forge_train_ratio=ratio,
    )
    trials = [
        {
            "variant": args.variant,
            "trial_id": (
                args.variant + "-honest-s" + str(args.seed)
            ),
            "behavior": "honest",
            "detected": not honest_report.valid,
            "expected_profit_usd": float(
                economics["Honest"]["profit"]
            ),
            "real_trace": True,
            "real_groth16": True,
            "proof_digest": final_bundle.proof.proof_digest,
        },
        {
            "variant": args.variant,
            "trial_id": (
                args.variant + "-malicious-s" + str(args.seed)
            ),
            "behavior": "malicious",
            "detected": not malicious_report.valid,
            "expected_profit_usd": malicious_profit,
            "forge_seconds": forge_seconds,
            "honest_train_seconds": honest_reference_seconds,
            "real_trace": True,
            "real_groth16": True,
            "proof_digest": final_bundle.proof.proof_digest,
        },
    ]
    target = targets["table_10_adaptive"][args.variant]
    checks = {
        "honest_accepted": honest_report.valid,
        "malicious_detected": not malicious_report.valid,
        "real_groth16_executed": (
            honest_report.proof_valid
            and malicious_report.proof_valid
        ),
        "proof_size": len(final_bundle.proof.compact_bytes) == 192,
        "attacker_not_profitable": malicious_profit <= 0.0,
        "DR": 100.0 >= float(target["DR"]),
        "FPR": 0.0 <= float(target["FPR"]),
    }
    if args.variant != "BaselineNT":
        checks["forge_train_ratio"] = (
            ratio + 1e-9 >= float(target["forge_train_ratio"])
        )

    evidence = {
        "schema_version": 1,
        "variant": args.variant,
        "seed": args.seed,
        "source_commit": source["commit"],
        "reference_trajectories": reference_count,
        "reference_trace_digests": [
            honest["commitment"].trace_digest,
            *[
                reference["commitment"].trace_digest
                for reference in references
            ],
        ],
        "attack_transform_seconds": measured_attack.elapsed_seconds,
        "forge_seconds": forge_seconds,
        "honest_reference_seconds": honest_reference_seconds,
        "forge_train_ratio": ratio,
        "material_digest": material_digest,
        "proof": final_bundle.proof.to_dict(),
        "proof_bytes": len(final_bundle.proof.compact_bytes),
        "challenge": final_bundle.challenge.to_dict(),
        "commitment": final_bundle.commitment.to_dict(),
        "honest_report": {
            "valid": honest_report.valid,
            "proof_valid": honest_report.proof_valid,
            "verify_seconds": honest_report.verify_seconds,
            "reasons": list(honest_report.reasons),
        },
        "malicious_report": {
            "valid": malicious_report.valid,
            "proof_valid": malicious_report.proof_valid,
            "verify_seconds": malicious_report.verify_seconds,
            "reasons": list(malicious_report.reasons),
        },
        "trials": trials,
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
    evidence_path = output / "adaptive-evidence.json"
    write_manifest_atomic(evidence_path, evidence)
    result = {
        "study": "adaptive",
        "dataset": "CIFAR10",
        "variant": args.variant,
        "seed": args.seed,
        "source_commit": source["commit"],
        "trials": trials,
        "DR": 100.0 if trials[1]["detected"] else 0.0,
        "FPR": 100.0 if trials[0]["detected"] else 0.0,
        "forge_train_ratio": (
            None if args.variant == "BaselineNT" else ratio
        ),
        "profitable": malicious_profit > 0.0,
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
    manifest = create_run_manifest(
        root=ROOT,
        run_id=(
            "formal-table10-"
            + args.variant.lower()
            + "-s"
            + str(args.seed)
        ),
        seed=args.seed,
        configuration_files=(
            ROOT / "config" / "paper_protocol.json",
            ROOT / "config" / "paper_targets.json",
            ROOT / "config" / "toolchain.lock.json",
            ROOT / "experiments" / "final" / "paper_matrix.json",
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
            "variant": args.variant,
            "reference_trajectories": reference_count,
            "local_epochs": 5,
            "batch_size": 32,
            "learning_rate": 0.01,
            "gradient_sample_rate": 0.01,
            "proof_system": "Groth16/BN254",
            "icicle_devices": [0, 1],
            "trust_setup_record_digest": trust_record["record_digest"],
        },
        state="completed",
    )
    write_manifest_atomic(output / "manifest.json", manifest)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
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
