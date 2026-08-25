import copy

import pytest

from experiments.final.evidence import (
    AUTHORITY_PDF_SHA256,
    CANONICAL_DIGEST_SCHEME,
    canonical_digest_matches,
    require_single_source_commit,
    seal_evidence,
    valid_source_commit,
)


def test_seal_evidence_binds_authority_source_and_content():
    commit = "a" * 40
    sealed = seal_evidence(
        {"source_commit": commit, "acceptance": {"passed": True}, "value": 7},
        analysis_source={"commit": commit, "dirty": False},
    )
    assert sealed["authority_pdf_sha256"] == AUTHORITY_PDF_SHA256
    assert sealed["source_commit"] == commit
    assert sealed["evidence_digest_scheme"] == CANONICAL_DIGEST_SCHEME
    assert sealed["analysis_source"] == {"commit": commit, "dirty": False}
    assert len(sealed["evidence_digest"]) == 64
    assert canonical_digest_matches(sealed)
    changed = copy.deepcopy(sealed)
    changed["value"] = 8
    assert not canonical_digest_matches(changed)
    resealed = seal_evidence(changed)
    assert resealed["evidence_digest"] != sealed["evidence_digest"]


def test_single_source_commit_rejects_mixed_or_malformed_inputs():
    assert valid_source_commit("b" * 40)
    assert not valid_source_commit("z" * 40)
    assert require_single_source_commit(
        ({"source_commit": "b" * 40}, {"source_commit": "b" * 40}),
        context="test",
    ) == "b" * 40
    with pytest.raises(ValueError, match="one valid source commit"):
        require_single_source_commit(
            ({"source_commit": "b" * 40}, {"source_commit": "c" * 40}),
            context="test",
        )
    with pytest.raises(ValueError, match="one valid source commit"):
        require_single_source_commit(({"source_commit": "short"},), context="test")


def test_seal_evidence_rejects_dirty_or_different_analysis_source():
    commit = "a" * 40
    with pytest.raises(ValueError, match="clean declared source commit"):
        seal_evidence(
            {"source_commit": commit},
            analysis_source={"commit": commit, "dirty": True},
        )
    with pytest.raises(ValueError, match="clean declared source commit"):
        seal_evidence(
            {"source_commit": commit},
            analysis_source={"commit": "b" * 40, "dirty": False},
        )
