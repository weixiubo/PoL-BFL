import hashlib
import json
from pathlib import Path

import pytest

from scripts.capture_formal_evidence import verify_standalone_trial_evidence


def _seal(payload, field):
    body = dict(payload)
    body.pop(field, None)
    payload[field] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def _trial(tmp_path: Path, study: str):
    commit = "a" * 40
    result_path = tmp_path / "result.json"
    evidence_name = (
        "adaptive-evidence.json"
        if study == "adaptive"
        else "hardware-evidence.json"
    )
    shared = {
        "seed": 1337,
        "source_commit": commit,
        "proof_bytes": 192,
        "honest_report": {"valid": True, "proof_valid": True},
        "malicious_report": {"valid": False, "proof_valid": True},
        "acceptance": {"passed": True},
    }
    if study == "adaptive":
        rows = [{"behavior": "honest"}, {"behavior": "malicious"}]
        evidence = {**shared, "variant": "BaselineNT", "trials": rows}
        result = {
            "study": study,
            "variant": "BaselineNT",
            "seed": 1337,
            "source_commit": commit,
            "trials": rows,
            "formal_accepted": True,
        }
    else:
        rows = [{"behavior": "honest"}, {"behavior": "malicious"}]
        evidence = {
            **shared,
            "hardware_pair": "RTX4090_RTX4090",
            "observations": rows,
            "trainer_attestation": {"name": "RTX 4090"},
            "verifier_attestation": {"name": "RTX 4090"},
            "numerical_decision": {"passed": True},
        }
        result = {
            "study": study,
            "hardware_pair": "RTX4090_RTX4090",
            "seed": 1337,
            "source_commit": commit,
            "observations": rows,
            "formal_accepted": True,
        }
    evidence = _seal(evidence, "evidence_digest")
    result["evidence_digest"] = evidence["evidence_digest"]
    result = _seal(result, "result_digest")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / evidence_name).write_text(json.dumps(evidence), encoding="utf-8")
    manifest = {
        "source": {"commit": commit},
        "artifact_sha256": {
            str(result_path): "0" * 64,
            str(tmp_path / evidence_name): "1" * 64,
        },
    }
    return result_path, result, manifest


@pytest.mark.parametrize("study", ["adaptive", "cross_hardware"])
def test_standalone_trial_evidence_is_digest_and_source_bound(tmp_path, study):
    result_path, result, manifest = _trial(tmp_path, study)
    assert verify_standalone_trial_evidence(result_path, result, manifest) == 2
    result["seed"] = 2026
    with pytest.raises(ValueError):
        verify_standalone_trial_evidence(result_path, result, manifest)
