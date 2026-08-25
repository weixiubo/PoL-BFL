"""secp256k1 ECDSA identities for signed verifier receipts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature, encode_dss_signature

from polbfl.committee.receipts import VerificationReceipt
from polbfl.protocol import Challenge


SECP256K1_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


@dataclass(frozen=True)
class ECDSASigner:
    verifier_id: str
    _private_key: ec.EllipticCurvePrivateKey

    @classmethod
    def generate(cls, verifier_id: str) -> "ECDSASigner":
        if not verifier_id:
            raise ValueError("verifier ID must be non-empty")
        return cls(verifier_id, ec.generate_private_key(ec.SECP256K1()))

    @classmethod
    def from_private_pem(cls, verifier_id: str, payload: bytes, password: bytes | None = None) -> "ECDSASigner":
        key = serialization.load_pem_private_key(payload, password=password)
        if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256K1):
            raise ValueError("receipt key must be a secp256k1 private key")
        return cls(verifier_id, key)

    @property
    def public_pem(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def private_pem(self, password: bytes | None = None) -> bytes:
        encryption = (
            serialization.NoEncryption()
            if password is None
            else serialization.BestAvailableEncryption(password)
        )
        return self._private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            encryption,
        )

    def sign(self, payload: bytes) -> str:
        raw = self._private_key.sign(payload, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(raw)
        if s > SECP256K1_ORDER // 2:
            s = SECP256K1_ORDER - s
        return encode_dss_signature(r, s).hex()

    def receipt(
        self,
        challenge: Challenge,
        *,
        proof_digest: str,
        valid: bool,
        verified_at_ns: int,
    ) -> VerificationReceipt:
        unsigned = VerificationReceipt(
            protocol_version=challenge.protocol_version,
            challenge_id=challenge.challenge_id,
            round_id=challenge.round_id,
            client_id=challenge.client_id,
            commitment_root=challenge.commitment_root,
            proof_digest=proof_digest,
            verifier_id=self.verifier_id,
            valid=bool(valid),
            verified_at_ns=int(verified_at_ns),
            signature="",
        )
        return VerificationReceipt(
            **unsigned.unsigned_dict(),
            signature=self.sign(unsigned.signing_bytes),
        )


class ECDSAPublicKeyRegistry:
    def __init__(self, public_keys: Mapping[str, bytes]):
        self._keys = {}
        for verifier_id, payload in public_keys.items():
            key = serialization.load_pem_public_key(payload)
            if not isinstance(key, ec.EllipticCurvePublicKey) or not isinstance(key.curve, ec.SECP256K1):
                raise ValueError("receipt public keys must use secp256k1")
            self._keys[str(verifier_id)] = key

    def verify(self, verifier_id: str, payload: bytes, signature: str) -> bool:
        key = self._keys.get(verifier_id)
        if key is None:
            return False
        try:
            raw = bytes.fromhex(signature)
            r, s = decode_dss_signature(raw)
            if not (0 < r < SECP256K1_ORDER and 0 < s <= SECP256K1_ORDER // 2):
                return False
            key.verify(raw, payload, ec.ECDSA(hashes.SHA256()))
            return True
        except (InvalidSignature, ValueError):
            return False
