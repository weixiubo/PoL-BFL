"""Training-time PoL trace recording."""

try:
    from .torch_recorder import (
        CheckpointMaterial,
        RecordedTrace,
        StepEvidence,
        TorchPoLRecorder,
    )
except ImportError:  # pragma: no cover - optional training runtime
    CheckpointMaterial = None
    RecordedTrace = None
    StepEvidence = None
    TorchPoLRecorder = None

__all__ = ["CheckpointMaterial", "RecordedTrace", "StepEvidence", "TorchPoLRecorder"]
