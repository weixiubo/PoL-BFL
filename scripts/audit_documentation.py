#!/usr/bin/env python3
"""Audit tracked repository documentation for formal publication quality."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT_SUFFIXES = {".md", ".markdown", ".rst", ".tex", ".txt"}
DOCUMENT_BASENAMES = {
    "readme",
    "license",
    "changelog",
    "contributing",
    "code_of_conduct",
}

FORBIDDEN_PATTERNS = (
    ("first_or_second_person", re.compile(
        r"(?<!`)\bI\b(?!`)|(?i:\b(?:we|our|ours|you|your|yours)\b)",
    )),
    ("conversational_contraction", re.compile(r"\b\w+'(?:t|re|ve|ll|d|m)\b", re.IGNORECASE)),
    ("informal_status", re.compile(
        r"\b(?:todo|tbd|fixme|wip|unfinished|intermediate|temporary|currently)\b"
        r"|work in progress|last updated|phase\s+\d+.*complete|new!",
        re.IGNORECASE,
    )),
    ("informal_instruction", re.compile(
        r"quick\s*start|recommended workflow|any questions|make sure|copy-paste"
        r"|\b(?:just|simply|easy|bare-bones)\b",
        re.IGNORECASE,
    )),
    ("promotional_language", re.compile(
        r"publication-ready|game[- ]changing|revolutionary|effortless|seamless(?:ly)?",
        re.IGNORECASE,
    )),
    ("subjective_or_transient_language", re.compile(
        r"\b(?:amazing|awesome|cool|obvious|obviously|good|bad|fast|slow|huge|tiny"
        r"|nice|please|helpful|tip|tips|hack|workaround|old|legacy|earlier|issue"
        r"|problem|warning|attention)\b",
        re.IGNORECASE,
    )),
    ("automated_authorship", re.compile(
        r"\b(?:chatgpt|openai|codex|ai[- ]generated|ai[- ]assisted)\b",
        re.IGNORECASE,
    )),
    ("private_infrastructure", re.compile(
        r"/home/(?:wxb|asus)(?:/|\b)|10\.102\.65\.27|wxb@202502",
        re.IGNORECASE,
    )),
    ("emoji", re.compile(r"[\U0001F300-\U0001FAFF✅❌⚠📋📊📈📝📚📞🎯🎨🔧🔬🚀]")),
    ("exclamation_mark", re.compile(r"!")),
    ("hidden_comment", re.compile(r"<!--")),
)

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
INLINE_CODE = re.compile(r"`[^`]*`")
RAW_TEXT_RULES = {
    "automated_authorship",
    "private_infrastructure",
    "emoji",
    "hidden_comment",
}
REQUIRED_PUBLIC_PATHS = (
    "chainEnv/contracts/PoLBFLProtocol.sol",
    "circuits/final/sampled_sgd_transition.circom",
    "config/paper_protocol.json",
    "config/toolchain.lock.json",
    "docs/DATASETS.md",
    "docs/FINAL_IMPLEMENTATION_SPEC.md",
    "docs/PAPER_TRACEABILITY.md",
    "docs/REPRODUCING.md",
    "docs/ZKP_AND_BLOCKCHAIN.md",
    "experiments/final/paper_matrix.json",
    "experiments/reproducibility/configs/rq1_table1_cifar10_free_riding_nt_vanilla_formal.json",
    "requirements-final.txt",
    "scripts/build_icicle_snark.sh",
    "scripts/gpu_idle_supervisor.py",
)


def tracked_document_paths(root: Path = ROOT) -> tuple[Path, ...]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    paths: list[Path] = []
    for value in raw.decode("utf-8").split("\0"):
        if not value:
            continue
        relative = Path(value)
        basename = relative.name.lower().split(".", 1)[0]
        if relative.suffix.lower() in DOCUMENT_SUFFIXES or basename in DOCUMENT_BASENAMES:
            paths.append((root / relative).resolve())
    return tuple(sorted(paths))


def _relative_link_failures(path: Path, text: str, root: Path) -> list[str]:
    failures: list[str] = []
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group(1).strip().strip("<>").split(maxsplit=1)[0]
        if not target or target.startswith("#") or SCHEME.match(target):
            continue
        target = unquote(target.split("#", 1)[0])
        resolved = (path.parent / target).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            failures.append(target)
    return failures


def audit_document(path: Path, *, root: Path = ROOT) -> list[dict[str, object]]:
    relative = path.relative_to(root.resolve()).as_posix()
    text = path.read_text(encoding="utf-8")
    failures: list[dict[str, object]] = []

    if not text.strip():
        failures.append({"file": relative, "rule": "nonempty", "line": 1})
    if path.suffix.lower() in {".md", ".markdown"}:
        first = next((line for line in text.splitlines() if line.strip()), "")
        if not first.startswith("# "):
            failures.append({"file": relative, "rule": "markdown_title", "line": 1})
    if text and not text.endswith("\n"):
        failures.append({"file": relative, "rule": "terminal_newline", "line": len(text.splitlines())})

    in_code_fence = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.rstrip() != line:
            failures.append({"file": relative, "rule": "trailing_whitespace", "line": number})
        fence = line.lstrip().startswith(("```", "~~~"))
        prose = "" if in_code_fence or fence else INLINE_CODE.sub("", line)
        for rule, pattern in FORBIDDEN_PATTERNS:
            if path.suffix.lower() == ".tex" and rule == "exclamation_mark":
                continue
            candidate = line if rule in RAW_TEXT_RULES else prose
            if pattern.search(candidate):
                failures.append({"file": relative, "rule": rule, "line": number})
        if fence:
            in_code_fence = not in_code_fence

    if in_code_fence:
        failures.append({"file": relative, "rule": "balanced_code_fence", "line": len(text.splitlines())})

    for target in _relative_link_failures(path, text, root):
        failures.append({"file": relative, "rule": "relative_link", "target": target})
    return failures


def audit_repository(root: Path = ROOT) -> dict[str, object]:
    paths = tracked_document_paths(root)
    failures = [
        failure
        for path in paths
        for failure in audit_document(path, root=root)
    ]
    readme = (root / "README.md").read_text(encoding="utf-8")
    package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    paper_tables = tuple(sorted(
        (root / "experiments/reproducibility/paper_targets/tables").glob("*.tex")
    ))
    paper_table_text = {
        path: path.read_text(encoding="utf-8") for path in paper_tables
    }
    gas_table = paper_table_text[
        root / "experiments/reproducibility/paper_targets/tables/table12_gas_breakdown.tex"
    ]
    checks = {
        "documents_present": len(paths) > 0,
        "formal_language": not failures,
        "readme_node_reference": "Node.js 18–20" in readme and "Node.js 24" not in readme,
        "node_engine_range": package.get("engines", {}).get("node") == ">=18 <21",
        "required_public_paths": all((root / path).exists() for path in REQUIRED_PUBLIC_PATHS),
        "readme_repository_structure": all(
            f"{directory}/" in readme
            for directory in (
                "polbfl",
                "client",
                "server",
                "chainEnv/contracts",
                "circuits/final",
                "experiments/final",
                "experiments/reproducibility",
                "dataset",
                "model",
                "analysis",
                "config",
                "scripts",
                "docs",
                "tests",
            )
        ),
        "paper_table_provenance": bool(paper_tables)
        and all(
            "Source: DOI 10.1145/3770855.3817739" in text
            and "result.md" not in text
            for text in paper_table_text.values()
        ),
        "paper_gas_reference": "@1.5 gwei" in gas_table and "@30 gwei" not in gas_table,
    }
    return {
        "passed": all(checks.values()),
        "document_count": len(paths),
        "documents": [path.relative_to(root).as_posix() for path in paths],
        "checks": checks,
        "failures": failures,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_repository(ROOT)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
