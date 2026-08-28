import hashlib
import json
from pathlib import Path

from experiments.final.audit_final_coverage import (
    COVERAGE,
    audit_coverage,
    coverage_input_paths,
)
from experiments.final.audit_measurements import (
    AUTHORITY_PDF_SHA256,
    FORMAL_CELL_ROUTES,
    _verify_input_hashes,
    audit_measurements,
)
from experiments.final.evidence import seal_evidence


ROOT = Path(__file__).parents[1]


def _formal_binding(tmp_path, study, index):
    if study not in FORMAL_CELL_ROUTES:
        return {}
    path = tmp_path / f"formal-result-{index}.json"
    path.write_text("{}", encoding="utf-8")
    return {"formal_result_paths": [str(path)]}


def _formal_verifier(path, *, root):
    assert path.is_file()
    assert root.is_dir()
    return {"source_commit": "a" * 40}


def _seal(payload):
    return seal_evidence(
        payload,
        analysis_source={"commit": "a" * 40, "dirty": False},
    )


def test_final_coverage_audit_routes_every_study():
    matrix = json.loads(
        (ROOT / "experiments" / "final" / "paper_matrix.json").read_text(encoding="utf-8")
    )
    report = audit_coverage(matrix, root=ROOT)
    assert report["passed"]
    assert set(report["routes"]) == set(COVERAGE) == set(matrix["studies"])
    assert report["scope"] == "implementation_routes"
    assert {
        "table_10_adaptive",
        "table_11_cross_hardware",
    } <= FORMAL_CELL_ROUTES


def test_final_coverage_provenance_hashes_matrix_and_every_unique_owner():
    matrix_path = ROOT / "experiments" / "final" / "paper_matrix.json"
    inputs = coverage_input_paths(matrix_path, root=ROOT)
    assert inputs[0] == matrix_path.resolve()
    assert len(inputs) == 1 + len(COVERAGE) == 18
    assert len(set(inputs)) == len(inputs)
    assert all(path.is_file() for path in inputs)


def test_measurement_audit_requires_accepted_same_source_evidence_for_all_routes(tmp_path):
    commit = "a" * 40
    evidence = {}
    for index, study in enumerate(sorted(COVERAGE)):
        path = tmp_path / f"{index}.json"
        input_path = tmp_path / f"input-{index}.bin"
        input_path.write_bytes(f"input-{index}".encode())
        payload = _seal(
            {
                "acceptance": {"passed": True},
                "source_commit": commit,
                "observed_value": index,
                "input_sha256": {
                    str(input_path): hashlib.sha256(input_path.read_bytes()).hexdigest()
                },
                **_formal_binding(tmp_path, study, index),
            }
        )
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        evidence[study] = str(path)
    report = audit_measurements(
        {
            "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
            "source_commit": commit,
            "evidence": evidence,
        },
        root=tmp_path,
        formal_cell_verifier=_formal_verifier,
    )
    assert report["measurement_complete"]
    assert report["passed"]


def test_measurement_audit_rejects_missing_evidence_digest(tmp_path):
    commit = "a" * 40
    evidence = {}
    for index, study in enumerate(sorted(COVERAGE)):
        path = tmp_path / f"{index}.json"
        input_path = tmp_path / f"missing-digest-input-{index}.bin"
        input_path.write_bytes(f"input-{index}".encode())
        payload = _seal(
            {
                "acceptance": {"passed": True},
                "source_commit": commit,
                "observed_value": index,
                "input_sha256": {
                    str(input_path): hashlib.sha256(input_path.read_bytes()).hexdigest()
                },
                **_formal_binding(tmp_path, study, index),
            }
        )
        if index == 0:
            payload.pop("evidence_digest")
        path.write_text(json.dumps(payload), encoding="utf-8")
        evidence[study] = str(path)
    report = audit_measurements(
        {
            "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
            "source_commit": commit,
            "evidence": evidence,
        },
        root=tmp_path,
        formal_cell_verifier=_formal_verifier,
    )
    assert not report["measurement_complete"]
    assert not report["passed"]


def test_measurement_audit_recomputes_canonical_evidence_digest(tmp_path):
    commit = "a" * 40
    evidence = {}
    for index, study in enumerate(sorted(COVERAGE)):
        path = tmp_path / f"canonical-{index}.json"
        input_path = tmp_path / f"canonical-input-{index}.bin"
        input_path.write_bytes(f"input-{index}".encode())
        payload = _seal(
            {
                "acceptance": {"passed": True},
                "source_commit": commit,
                "observed_value": index,
                "input_sha256": {
                    str(input_path): hashlib.sha256(input_path.read_bytes()).hexdigest()
                },
                **_formal_binding(tmp_path, study, index),
            }
        )
        if index == 0:
            payload["observed_value"] = -1
        path.write_text(json.dumps(payload), encoding="utf-8")
        evidence[study] = str(path)
    report = audit_measurements(
        {
            "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
            "source_commit": commit,
            "evidence": evidence,
        },
        root=tmp_path,
        formal_cell_verifier=_formal_verifier,
    )
    assert not report["measurement_complete"]
    assert any(
        name.endswith(".evidence_content") and not passed
        for name, passed in report["checks"].items()
    )


def test_measurement_audit_rehashes_declared_input_files(tmp_path):
    commit = "a" * 40
    evidence = {}
    first_input = None
    for index, study in enumerate(sorted(COVERAGE)):
        path = tmp_path / f"input-bound-{index}.json"
        input_path = tmp_path / f"bound-input-{index}.bin"
        input_path.write_bytes(f"input-{index}".encode())
        if first_input is None:
            first_input = input_path
        payload = _seal(
            {
                "acceptance": {"passed": True},
                "source_commit": commit,
                "input_sha256": {
                    str(input_path): hashlib.sha256(input_path.read_bytes()).hexdigest()
                },
                **_formal_binding(tmp_path, study, index),
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        evidence[study] = str(path)
    first_input.write_bytes(b"tampered")
    report = audit_measurements(
        {
            "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
            "source_commit": commit,
            "evidence": evidence,
        },
        root=tmp_path,
        formal_cell_verifier=_formal_verifier,
    )
    assert not report["measurement_complete"]
    assert any(
        name.endswith(".input_hashes") and not passed
        for name, passed in report["checks"].items()
    )


def test_input_hash_verifier_recurses_into_json_manifests(tmp_path):
    leaf = tmp_path / "leaf.bin"
    leaf.write_bytes(b"leaf")
    nested = tmp_path / "nested.json"
    nested_payload = _seal(
        {
            "source_commit": "a" * 40,
            "input_sha256": {
                str(leaf): hashlib.sha256(leaf.read_bytes()).hexdigest()
            },
        }
    )
    nested.write_text(
        json.dumps(nested_payload),
        encoding="utf-8",
    )
    root_inputs = {str(nested): hashlib.sha256(nested.read_bytes()).hexdigest()}
    assert _verify_input_hashes(root_inputs, root=tmp_path) == (True, True, True, 2)
    leaf.write_bytes(b"changed")
    assert _verify_input_hashes(root_inputs, root=tmp_path) == (True, True, False, 2)


def test_input_hash_verifier_treats_unsealed_json_as_an_opaque_hashed_input(tmp_path):
    component = tmp_path / "component.json"
    component.write_text(
        json.dumps({"input_sha256": {"logical-tool-label": "a" * 64}}),
        encoding="utf-8",
    )
    inputs = {
        str(component): hashlib.sha256(component.read_bytes()).hexdigest()
    }
    assert _verify_input_hashes(inputs, root=tmp_path) == (True, True, True, 1)


def test_input_hash_verifier_rejects_a_tampered_sealed_json(tmp_path):
    component = tmp_path / "sealed.json"
    payload = _seal(
        {
            "source_commit": "a" * 40,
            "input_sha256": {str(tmp_path / "leaf.bin"): "b" * 64},
        }
    )
    payload["source_commit"] = "b" * 40
    component.write_text(json.dumps(payload), encoding="utf-8")
    inputs = {
        str(component): hashlib.sha256(component.read_bytes()).hexdigest()
    }
    assert _verify_input_hashes(inputs, root=tmp_path) == (True, True, False, 1)


def test_measurement_audit_requires_formal_cell_receipts(tmp_path):
    commit = "a" * 40
    evidence = {}
    omitted = next(iter(FORMAL_CELL_ROUTES))
    for index, study in enumerate(sorted(COVERAGE)):
        path = tmp_path / f"formal-required-{index}.json"
        input_path = tmp_path / f"formal-required-input-{index}.bin"
        input_path.write_bytes(f"input-{index}".encode())
        formal = {} if study == omitted else _formal_binding(tmp_path, study, index)
        payload = _seal(
            {
                "acceptance": {"passed": True},
                "source_commit": commit,
                "input_sha256": {
                    str(input_path): hashlib.sha256(input_path.read_bytes()).hexdigest()
                },
                **formal,
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
        evidence[study] = str(path)
    report = audit_measurements(
        {
            "authority_pdf_sha256": AUTHORITY_PDF_SHA256,
            "source_commit": commit,
            "evidence": evidence,
        },
        root=tmp_path,
        formal_cell_verifier=_formal_verifier,
    )
    assert not report["measurement_complete"]
    assert report["checks"][omitted + ".formal_results_declared"] is False
