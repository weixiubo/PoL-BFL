"""PoL trace and model-update verification."""

from .strict import (
    ChallengeResponse,
    CheckpointOpening,
    IntervalWitness,
    StrictTraceVerifier,
    VerificationReport,
    parameter_l2_distance,
)

try:  # Optional until the training runtime is installed.
    from .torch_replay import TorchSGDReplay, TorchSGDReplayConfig
except ImportError:  # pragma: no cover
    TorchSGDReplay = None
    TorchSGDReplayConfig = None

__all__ = [
    "ChallengeResponse",
    "CheckpointOpening",
    "IntervalWitness",
    "StrictTraceVerifier",
    "VerificationReport",
    "parameter_l2_distance",
    "TorchSGDReplay",
    "TorchSGDReplayConfig",
]
