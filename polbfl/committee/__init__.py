"""Verifier committee selection and quorum receipts."""

from .receipts import QuorumDecision, ReceiptQuorum, VerificationReceipt
from .selection import CommitteeCandidate, select_committee

try:
    from .ecdsa import ECDSAPublicKeyRegistry, ECDSASigner
except ImportError:  # pragma: no cover - optional cryptography runtime
    ECDSAPublicKeyRegistry = None
    ECDSASigner = None

try:
    from .orchestration import (
        AuthenticatedVRFSeed,
        CommitteeAssignment,
        IndependentVerifier,
        VerifierVote,
        decide_proof_set,
        proof_set_digest,
    )
except ImportError:  # pragma: no cover - optional cryptography runtime
    AuthenticatedVRFSeed = None
    CommitteeAssignment = None
    IndependentVerifier = None
    VerifierVote = None
    decide_proof_set = None
    proof_set_digest = None

__all__ = [
    "CommitteeCandidate",
    "QuorumDecision",
    "ReceiptQuorum",
    "VerificationReceipt",
    "select_committee",
    "ECDSAPublicKeyRegistry",
    "ECDSASigner",
    "AuthenticatedVRFSeed",
    "CommitteeAssignment",
    "IndependentVerifier",
    "VerifierVote",
    "decide_proof_set",
    "proof_set_digest",
]
