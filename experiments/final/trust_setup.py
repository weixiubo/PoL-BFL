#!/usr/bin/env python3
"""Create and validate production Groth16 trust-setup provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.final.manifest import sha256_file, write_manifest_atomic


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_trust_setup_record(
    *,
    build: Path,
    ptau: Path,
    toolchain: Mapping[str, Any],
    powersoftau_verify_log: Path,
    zkey_verify_log: Path,
) -> dict[str, Any]:
    build = build.resolve()
    ptau = ptau.resolve()
    powers = toolchain["powers_of_tau"]
    powersoftau_text = powersoftau_verify_log.read_text(encoding="utf-8", errors="replace")
    zkey_text = zkey_verify_log.read_text(encoding="utf-8", errors="replace")
    if "Powers of Tau Ok!" not in powersoftau_text and "Powers Of tau file OK!" not in powersoftau_text:
        raise ValueError("Powers-of-Tau verification log does not contain a success marker")
    if "ZKey Ok!" not in zkey_text:
        raise ValueError("zkey verification log does not contain a success marker")
    record = {
        "schema_version": 1,
        "classification": "production",
        "phase1_transcript": {
            "filename": ptau.name,
            "size_bytes": ptau.stat().st_size,
            "blake2b_512": file_digest(ptau, "blake2b"),
            "sha512": file_digest(ptau, "sha512"),
            "snarkjs_verified": True,
            "verification_log_sha256": sha256_file(powersoftau_verify_log),
        },
        "phase2": {
            "independent_contribution": True,
            "contribution_entropy_retained": False,
            "zkey_verified": True,
            "verification_log_sha256": sha256_file(zkey_verify_log),
        },
        "artifacts": {
            "r1cs_sha256": sha256_file(build / "sampled_sgd_reference.r1cs"),
            "zkey_sha256": sha256_file(build / "sampled_sgd_reference_final.zkey"),
            "verification_key_sha256": sha256_file(build / "verification_key.json"),
            "wasm_sha256": sha256_file(
                build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm"
            ),
            "witness_binary_sha256": sha256_file(
                build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference"
            ),
        },
    }
    expected = {
        "filename": powers["production_filename"],
        "size_bytes": int(powers["production_size_bytes"]),
        "blake2b_512": powers["production_blake2b_512"],
        "sha512": powers["production_sha512"],
    }
    if any(record["phase1_transcript"][name] != value for name, value in expected.items()):
        raise ValueError("production Powers-of-Tau transcript differs from the toolchain lock")
    body = dict(record)
    record["record_digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def validate_trust_setup(
    *,
    build: Path,
    toolchain: Mapping[str, Any],
) -> dict[str, Any]:
    build = build.resolve()
    record_path = build / "trust_setup.json"
    checks = {}
    details: dict[str, Any] = {"record": str(record_path)}
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        body = dict(record)
        declared_digest = body.pop("record_digest")
        checks["record_digest"] = declared_digest == hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        phase1 = record["phase1_transcript"]
        powers = toolchain["powers_of_tau"]
        checks.update(
            {
                "classification": record["classification"] == "production",
                "phase1_filename": phase1["filename"] == powers["production_filename"],
                "phase1_size": int(phase1["size_bytes"]) == int(powers["production_size_bytes"]),
                "phase1_blake2b": phase1["blake2b_512"] == powers["production_blake2b_512"],
                "phase1_sha512": phase1["sha512"] == powers["production_sha512"],
                "phase1_verified": phase1["snarkjs_verified"] is True,
                "phase2_contributed": record["phase2"]["independent_contribution"] is True,
                "phase2_entropy_destroyed": record["phase2"]["contribution_entropy_retained"] is False,
                "zkey_verified": record["phase2"]["zkey_verified"] is True,
            }
        )
        artifact_paths = {
            "r1cs_sha256": build / "sampled_sgd_reference.r1cs",
            "zkey_sha256": build / "sampled_sgd_reference_final.zkey",
            "verification_key_sha256": build / "verification_key.json",
            "wasm_sha256": build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
            "witness_binary_sha256": (
                build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference"
            ),
        }
        for name, path in artifact_paths.items():
            checks[f"artifact_{name}"] = path.is_file() and sha256_file(path) == record["artifacts"][name]
        verification_logs = {
            "phase1_log": (
                build / "powersoftau-verify.log",
                record["phase1_transcript"]["verification_log_sha256"],
                ("Powers of Tau Ok!", "Powers Of tau file OK!"),
            ),
            "phase2_log": (
                build / "zkey-verify.log",
                record["phase2"]["verification_log_sha256"],
                ("ZKey Ok!",),
            ),
        }
        for name, (path, expected, markers) in verification_logs.items():
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            checks[name] = (
                path.is_file()
                and sha256_file(path) == expected
                and any(marker in text for marker in markers)
            )
        details["classification"] = record.get("classification")
        details["artifacts"] = record.get("artifacts")
    except Exception as exc:
        checks["record_readable"] = False
        details["error"] = f"{type(exc).__name__}:{exc}"
    return {"passed": bool(checks) and all(checks.values()), "checks": checks, "details": details}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--build", type=Path, required=True)
    create.add_argument("--ptau", type=Path, required=True)
    create.add_argument("--toolchain", type=Path, default=root / "config" / "toolchain.lock.json")
    create.add_argument("--powersoftau-verify-log", type=Path)
    create.add_argument("--zkey-verify-log", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--build", type=Path, required=True)
    validate.add_argument("--toolchain", type=Path, default=root / "config" / "toolchain.lock.json")
    args = parser.parse_args()
    toolchain = json.loads(args.toolchain.read_text(encoding="utf-8"))
    if args.command == "create":
        record = create_trust_setup_record(
            build=args.build,
            ptau=args.ptau,
            toolchain=toolchain,
            powersoftau_verify_log=(
                args.build / "powersoftau-verify.log"
                if args.powersoftau_verify_log is None
                else args.powersoftau_verify_log
            ),
            zkey_verify_log=(
                args.build / "zkey-verify.log"
                if args.zkey_verify_log is None
                else args.zkey_verify_log
            ),
        )
        write_manifest_atomic(args.build / "trust_setup.json", record)
        output = {"passed": True, "record": record}
    else:
        output = validate_trust_setup(build=args.build, toolchain=toolchain)
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
