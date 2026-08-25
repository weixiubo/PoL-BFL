"""Uniform provenance and digest sealing for final-paper aggregate evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


AUTHORITY_PDF_SHA256 = (
    "0b013e58d4f99f91470c61a891a4ee89dfd09eff58e131abbe730d1f6f91e6d4"
)
CANONICAL_DIGEST_SCHEME = "sha256-canonical-json-without-evidence-digest-v1"


def valid_source_commit(value: object) -> bool:
    text = str(value)
    return len(text) == 40 and all(character in "0123456789abcdef" for character in text)


def canonical_digest_matches(payload: Mapping[str, Any]) -> bool:
    """Recompute a canonical sealed-evidence digest without trusting its value."""

    digest = str(payload.get("evidence_digest", ""))
    if (
        payload.get("evidence_digest_scheme") != CANONICAL_DIGEST_SCHEME
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return False
    body = dict(payload)
    body.pop("evidence_digest", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == digest


def require_single_source_commit(
    rows: Iterable[Mapping[str, Any]],
    *,
    context: str,
) -> str:
    """Return the sole valid execution commit represented by aggregate inputs."""

    commits = {str(row.get("source_commit", "")) for row in rows}
    if len(commits) != 1 or not valid_source_commit(next(iter(commits), "")):
        raise ValueError(f"{context} inputs must use one valid source commit")
    return next(iter(commits))


def seal_evidence(
    payload: Mapping[str, Any],
    *,
    source_commit: str | None = None,
    analysis_root: str | Path | None = None,
    analysis_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind aggregate evidence to the paper, execution source, and its own bytes."""

    result = dict(payload)
    declared = source_commit or result.get("source_commit")
    if not valid_source_commit(declared):
        raise ValueError("aggregate evidence requires a valid source commit")
    existing = result.get("source_commit")
    if existing is not None and str(existing) != str(declared):
        raise ValueError("aggregate evidence contains conflicting source commits")
    existing_authority = result.get("authority_pdf_sha256")
    if existing_authority is not None and existing_authority != AUTHORITY_PDF_SHA256:
        raise ValueError("aggregate evidence conflicts with the authority PDF")
    existing_scheme = result.get("evidence_digest_scheme")
    if existing_scheme is not None and existing_scheme != CANONICAL_DIGEST_SCHEME:
        raise ValueError("aggregate evidence uses an incompatible digest scheme")
    if analysis_root is not None and analysis_source is not None:
        raise ValueError("provide either analysis_root or analysis_source, not both")
    existing_analysis = result.get("analysis_source")
    if analysis_source is None and analysis_root is None and existing_analysis is not None:
        if not isinstance(existing_analysis, Mapping):
            raise ValueError("aggregate evidence analysis_source is invalid")
        analysis_source = existing_analysis
    elif analysis_source is not None and existing_analysis is not None:
        if dict(analysis_source) != existing_analysis:
            raise ValueError("aggregate evidence contains conflicting analysis sources")
    if analysis_root is not None:
        from experiments.final.manifest import source_identity

        analysis_source = source_identity(Path(analysis_root))
    if analysis_source is not None:
        analysis_source = dict(analysis_source)
        if (
            analysis_source.get("dirty") is not False
            or analysis_source.get("commit") != str(declared)
        ):
            raise ValueError(
                "aggregate analysis must use the clean declared source commit"
            )
        result["analysis_source"] = analysis_source
    result.pop("evidence_digest", None)
    result.setdefault("schema_version", 1)
    result["authority_pdf_sha256"] = AUTHORITY_PDF_SHA256
    result["source_commit"] = str(declared)
    result["evidence_digest_scheme"] = CANONICAL_DIGEST_SCHEME
    result["evidence_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return result
