"""Real Groth16 proof generation and verification."""

from .groth16 import Groth16Artifacts, Groth16Backend, Groth16Proof
from .codec import BN254_BASE_FIELD, decode_groth16_proof, encode_groth16_proof
from .field import (
    BN254_SCALAR_FIELD,
    digest_to_field,
    interval_batch_commitment,
    protocol_binding_field,
)
from .poseidon import PoseidonBridge
from .icicle_pool import IcicleProverPool
from .rapidsnark_pool import RapidsnarkProverPool
from .witness import (
    PADDED_DATA_INDEX,
    ZKCircuitConfig,
    pad_batch_indices,
    quantize,
    require_signed_range,
    signed_components,
)
from .prover import ZKPoLProver
from .bundle import (
    PUBLIC_SIGNAL_NAMES,
    ZKBundleReport,
    ZKBundleVerifier,
    ZKCheckpointOpening,
    ZKIntervalBundle,
)

__all__ = [
    "Groth16Artifacts",
    "Groth16Backend",
    "Groth16Proof",
    "BN254_BASE_FIELD",
    "decode_groth16_proof",
    "encode_groth16_proof",
    "BN254_SCALAR_FIELD",
    "digest_to_field",
    "interval_batch_commitment",
    "protocol_binding_field",
    "PoseidonBridge",
    "IcicleProverPool",
    "RapidsnarkProverPool",
    "PADDED_DATA_INDEX",
    "ZKCircuitConfig",
    "pad_batch_indices",
    "quantize",
    "require_signed_range",
    "signed_components",
    "ZKPoLProver",
    "PUBLIC_SIGNAL_NAMES",
    "ZKBundleReport",
    "ZKBundleVerifier",
    "ZKCheckpointOpening",
    "ZKIntervalBundle",
]
