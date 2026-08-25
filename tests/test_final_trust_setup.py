import hashlib
import json

from experiments.final.trust_setup import validate_trust_setup


def test_trust_setup_validation_binds_production_transcript_and_artifacts(tmp_path):
    build = tmp_path / "build"
    (build / "sampled_sgd_reference_js").mkdir(parents=True)
    artifacts = {
        "r1cs_sha256": build / "sampled_sgd_reference.r1cs",
        "zkey_sha256": build / "sampled_sgd_reference_final.zkey",
        "verification_key_sha256": build / "verification_key.json",
        "wasm_sha256": build / "sampled_sgd_reference_js" / "sampled_sgd_reference.wasm",
        "witness_binary_sha256": (
            build / "sampled_sgd_reference_cpp" / "sampled_sgd_reference"
        ),
    }
    artifact_hashes = {}
    for name, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        artifact_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    toolchain = {
        "powers_of_tau": {
            "production_filename": "production.ptau",
            "production_size_bytes": 10,
            "production_blake2b_512": "b" * 128,
            "production_sha512": "c" * 128,
        }
    }
    powers_log = build / "powersoftau-verify.log"
    powers_log.write_text("Powers of Tau Ok!\n", encoding="utf-8")
    zkey_log = build / "zkey-verify.log"
    zkey_log.write_text("ZKey Ok!\n", encoding="utf-8")
    record = {
        "schema_version": 1,
        "classification": "production",
        "phase1_transcript": {
            "filename": "production.ptau",
            "size_bytes": 10,
            "blake2b_512": "b" * 128,
            "sha512": "c" * 128,
            "snarkjs_verified": True,
            "verification_log_sha256": hashlib.sha256(powers_log.read_bytes()).hexdigest(),
        },
        "phase2": {
            "independent_contribution": True,
            "contribution_entropy_retained": False,
            "zkey_verified": True,
            "verification_log_sha256": hashlib.sha256(zkey_log.read_bytes()).hexdigest(),
        },
        "artifacts": artifact_hashes,
    }
    record["record_digest"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (build / "trust_setup.json").write_text(json.dumps(record), encoding="utf-8")
    assert validate_trust_setup(build=build, toolchain=toolchain)["passed"]
    record["classification"] = "development"
    (build / "trust_setup.json").write_text(json.dumps(record), encoding="utf-8")
    assert not validate_trust_setup(build=build, toolchain=toolchain)["passed"]
