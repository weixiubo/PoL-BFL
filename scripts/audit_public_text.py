#!/usr/bin/env python3
"""Audit every tracked text file for public-source publication quality."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_documentation import audit_repository as audit_documentation


BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
LEGAL_PROSE_PATHS = {
    "LICENSE",
    "chainEnv/token/LICENSE",
    "chainEnv/contracts/Groth16Verifier.sol",
}
RULE_DEFINITION_PATHS = {
    "scripts/audit_documentation.py",
    "scripts/audit_public_text.py",
}
EXACT_TOOL_MARKER_PATHS = {
    "experiments/final/trust_setup.py",
    "tests/test_final_trust_setup.py",
}


def _ascii(hexadecimal: str) -> str:
    return bytes.fromhex(hexadecimal).decode("ascii")


INTERNAL_PROCESS_TERMS = (
    _ascii("717569636b"),
    _ascii("7072656c696d696e617279"),
    _ascii("636c656172616e6365"),
    _ascii("6f7665726e69676874"),
    _ascii("706861736535"),
    _ascii("776970"),
    _ascii("746f646f"),
    _ascii("746264"),
    _ascii("6669786d65"),
    _ascii("756e66696e6973686564"),
    _ascii("68616c662d66696e6973686564"),
    _ascii("776f726b20696e2070726f6772657373"),
    _ascii("70726f64756374696f6e2d7265616479"),
    _ascii("6f6e652d636c69636b"),
    _ascii("726561647920746f20757365"),
    _ascii("6e6577206665617475726573"),
)
INFORMAL_NOUNS = (
    _ascii("73747562"),
    _ascii("64756d6d79"),
    _ascii("66616b65"),
)
SUBJECTIVE_TERMS = (
    _ascii("617765736f6d65"),
    _ascii("616d617a696e67"),
    _ascii("657863656c6c656e74"),
    _ascii("70657266656374"),
    _ascii("676f6f64"),
    _ascii("626164"),
    _ascii("6f6276696f75736c79"),
)
FIRST_OR_SECOND_PERSON = (
    _ascii("7765"),
    _ascii("6f7572"),
    _ascii("6f757273"),
    _ascii("796f75"),
    _ascii("796f7572"),
    _ascii("796f757273"),
)
INFORMAL_CHINESE_TERMS = (
    "\u6e05\u969c",
    "\u5feb\u901f",
    "\u6709\u5229",
    "\u4f18\u79c0",
    "\u5b8c\u7f8e",
    "\u4e0d\u9700\u8981\u64cd\u5fc3",
    "\u540e\u7eed\u5b9e\u73b0",
    "\u9700\u8981\u4fee\u6539",
    "\u9700\u8981\u66f4\u65b0",
    "\u5b9e\u9645\u5e94\u8be5",
    "\u7b80\u5316",
)
PROSE_EXCLAMATION = re.compile(r"(?<=[\w\]\)}])!+(?=[\x22\x27\s]|$)")
CONTRACTION = re.compile(r"\b\w+'(?:t|re|ve|ll|d|m)\b", re.IGNORECASE)
ABSOLUTE_USER_HOME = re.compile(r"/home/[A-Za-z0-9._-]+/")
INFORMAL_FILENAME = re.compile(
    r"(?:^|[_./-])(?:"
    + "|".join(
        re.escape(term)
        for term in (
            *INTERNAL_PROCESS_TERMS[:5],
            _ascii("627567"),
            _ascii("666978"),
        )
    )
    + r")(?:[_./-]|$)",
    re.IGNORECASE,
)


def tracked_paths(root: Path = ROOT) -> tuple[Path, ...]:
    raw = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
    )
    return tuple(
        (root / value).resolve()
        for value in raw.decode("utf-8").split("\0")
        if value
    )


def _contains_emoji(text: str) -> bool:
    return any(
        0x1F300 <= ord(character) <= 0x1FAFF
        or 0x2600 <= ord(character) <= 0x27BF
        for character in text
    )


def _word_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(
        r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b",
        re.IGNORECASE,
    )


def audit_public_text(root: Path = ROOT) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    text_paths: list[Path] = []
    binary_paths: list[Path] = []
    process_pattern = _word_pattern(INTERNAL_PROCESS_TERMS)
    informal_noun_pattern = _word_pattern(INFORMAL_NOUNS)
    subjective_pattern = _word_pattern(SUBJECTIVE_TERMS)
    person_pattern = _word_pattern(FIRST_OR_SECOND_PERSON)

    for path in tracked_paths(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in BINARY_SUFFIXES:
            binary_paths.append(path)
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            binary_paths.append(path)
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            failures.append(
                {"file": relative, "rule": "utf8", "offset": error.start}
            )
            continue
        text_paths.append(path)
        if _contains_emoji(text):
            failures.append({"file": relative, "rule": "emoji"})
        if ABSOLUTE_USER_HOME.search(text) and relative not in RULE_DEFINITION_PATHS:
            failures.append({"file": relative, "rule": "absolute_user_home"})
        if relative not in RULE_DEFINITION_PATHS:
            if process_pattern.search(text):
                failures.append({"file": relative, "rule": "internal_process_language"})
            if informal_noun_pattern.search(text):
                failures.append({"file": relative, "rule": "informal_test_noun"})
        if relative not in LEGAL_PROSE_PATHS | RULE_DEFINITION_PATHS:
            if person_pattern.search(text):
                failures.append({"file": relative, "rule": "first_or_second_person"})
            if CONTRACTION.search(text):
                failures.append({"file": relative, "rule": "contraction"})
            if subjective_pattern.search(text):
                failures.append({"file": relative, "rule": "subjective_language"})
        if (
            relative not in EXACT_TOOL_MARKER_PATHS
            and (PROSE_EXCLAMATION.search(text) or "\uff01" in text)
        ):
            failures.append({"file": relative, "rule": "prose_exclamation"})
        if any(term in text for term in INFORMAL_CHINESE_TERMS):
            failures.append({"file": relative, "rule": "informal_chinese_language"})
        if INFORMAL_FILENAME.search(relative):
            failures.append({"file": relative, "rule": "informal_filename"})

    documentation = audit_documentation(root)
    checks = {
        "tracked_files_present": bool(text_paths or binary_paths),
        "all_nonbinary_files_utf8": not any(
            failure["rule"] == "utf8" for failure in failures
        ),
        "formal_public_text": not failures,
        "documentation_passed": documentation["passed"],
    }
    return {
        "passed": all(checks.values()),
        "tracked_file_count": len(text_paths) + len(binary_paths),
        "text_file_count": len(text_paths),
        "binary_file_count": len(binary_paths),
        "checks": checks,
        "documentation": {
            "passed": documentation["passed"],
            "document_count": documentation["document_count"],
            "failure_count": len(documentation["failures"]),
        },
        "failures": failures,
    }


def main() -> int:
    report = audit_public_text(ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
