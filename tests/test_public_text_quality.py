import json
from pathlib import Path

from scripts.audit_public_text import audit_public_text


ROOT = Path(__file__).parents[1]


def test_all_tracked_public_text_meets_repository_standard():
    report = audit_public_text(ROOT)
    assert report["passed"], json.dumps(report["failures"], indent=2)
    assert report["tracked_file_count"] >= 650
    assert report["text_file_count"] >= 640
    assert report["documentation"]["passed"]
    assert report["documentation"]["document_count"] == 32
