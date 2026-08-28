"""Interfaces shared by model-update compression codecs."""

from abc import ABC, abstractmethod
from collections import OrderedDict


class ModelCompression(ABC):
    """Encode and decode a model state dictionary."""

    @abstractmethod
    def encode(self, model_state_dict: OrderedDict) -> OrderedDict:
        """Return a compressed copy of ``model_state_dict``."""

    @abstractmethod
    def decode(self, compressed_state_dict: OrderedDict) -> OrderedDict:
        """Return a floating-point reconstruction of a compressed state."""
