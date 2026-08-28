from .baseCompression import ModelCompression
from .compressionUtil import (
    Int8Quantizer,
    magnitude_prune,
    quantify_decode,
    quantify_encode,
)

__all__ = [
    "Int8Quantizer",
    "ModelCompression",
    "magnitude_prune",
    "quantify_decode",
    "quantify_encode",
]
