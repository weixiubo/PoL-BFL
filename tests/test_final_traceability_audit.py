import copy
import json
from pathlib import Path

from experiments.final.audit_traceability import (
    REQUIRED_CATEGORIES,
    audit_traceability,
    traceability_input_paths,
)
from experiments.final.evidence import AUTHORITY_PDF_SHA256


ROOT = Path(__file__).parents[1]
MANIFEST_PATH = ROOT / "config" / "paper_traceability.json"


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_traceability_manifest_covers_all_requirements_and_categories():
    report = audit_traceability(
        _manifest(), root=ROOT, paper_sha256=AUTHORITY_PDF_SHA256
    )
    assert report["passed"]
    assert report["requirement_count"] == 43
    assert report["documentation_requirement_count"] == 32
    assert report["auxiliary_test_reference_count"] == 1
    assert set(report["categories"]) == REQUIRED_CATEGORIES
    assert all(record["passed"] for record in report["records"])
    ids = {record["id"] for record in report["records"]}
    assert {
        "storage.random_retrieval_challenges",
        "threat_model.scope_assumptions",
    } <= ids


def test_traceability_inputs_are_exact_existing_files():
    paths = traceability_input_paths(MANIFEST_PATH, _manifest(), root=ROOT)
    assert MANIFEST_PATH.resolve() in paths
    assert (ROOT / "docs" / "PAPER_TRACEABILITY.md").resolve() in paths
    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)


def test_traceability_rejects_missing_owner_and_duplicate_id():
    manifest = copy.deepcopy(_manifest())
    manifest["requirements"][0]["owners"] = ["missing-owner.py"]
    manifest["requirements"][1]["id"] = manifest["requirements"][0]["id"]
    report = audit_traceability(
        manifest, root=ROOT, paper_sha256=AUTHORITY_PDF_SHA256
    )
    assert not report["passed"]
    assert not report["checks"]["requirement_ids_unique"]
    assert any(
        name.endswith(":owners_exist") and passed is False
        for name, passed in report["checks"].items()
    )


def test_traceability_rejects_non_authoritative_paper():
    report = audit_traceability(
        _manifest(), root=ROOT, paper_sha256="0" * 64
    )
    assert not report["passed"]
    assert not report["checks"]["authority_pdf"]


def test_auxiliary_contract_test_is_driven_by_collected_python_test():
    report = audit_traceability(
        _manifest(), root=ROOT, paper_sha256=AUTHORITY_PDF_SHA256
    )
    record = next(
        row for row in report["records"] if row["id"] == "contract.audit_replay"
    )
    assert record["checks"]["auxiliary_tests_driven_by_python"]
