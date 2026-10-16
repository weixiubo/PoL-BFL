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
    toolchain = json.loads((ROOT / "config/toolchain.lock.json").read_text())
    major = int(toolchain["node"].split(".")[0])
    assert package["engines"]["node"] == f">={major} <{major + 1}"
    assert f"Node.js {toolchain['node']}" in readme
    assert "pip install -r requirements-final.txt" in readme
    assert "npm ci" in readme
    assert "docs/REPRODUCING.md" in readme
    assert "docs/assets/polbfl-overview.png" in readme
