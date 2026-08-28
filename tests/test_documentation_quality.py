import json
from pathlib import Path

from scripts.audit_documentation import audit_repository


ROOT = Path(__file__).parents[1]


def test_all_tracked_documentation_meets_publication_standard():
    report = audit_repository(ROOT)
    assert report["passed"], json.dumps(report["failures"], indent=2)
    assert report["document_count"] >= 32
    assert report["checks"]["required_public_paths"]
    assert report["checks"]["readme_repository_structure"]
    assert report["checks"]["paper_table_provenance"]
    assert report["checks"]["paper_gas_reference"]


def test_readme_runtime_and_installation_are_consistent():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    assert package["engines"]["node"] == ">=18 <21"
    assert "Node.js 18–20" in readme
    assert "Node.js 24" not in readme
    assert "pip install -r requirements-final.txt" in readme
    assert "npm ci" in readme
    assert "docs/REPRODUCING.md" in readme
    assert "docs/assets/polbfl-overview.png" in readme
