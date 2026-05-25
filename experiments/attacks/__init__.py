"""
Attack implementations for security evaluation

This module provides various attack implementations for testing PoL-FL security.
Includes:
- Byzantine attacks (original + Blades framework)
- Free-riding attacks
- Sybil attacks
"""

from .byzantine_attacks import RandomNoiseAttack, LabelFlippingAttack, ModelReplacementAttack
from .free_riding_attacks import NoTrainingAttack, LazyTrainingAttack
from .sybil_attacks import SybilAttack, create_sybil_attack
from .blades_attacks import ALIEAttack, IPMAttack, MinMaxAttack, create_blades_attack

__all__ = [
    # Original Byzantine attacks
    'RandomNoiseAttack',
    'LabelFlippingAttack',
    'ModelReplacementAttack',
    # Free-riding attacks
    'NoTrainingAttack',
    'LazyTrainingAttack',
    # Sybil attacks
    'SybilAttack',
    'create_sybil_attack',
    # Blades framework attacks (NeurIPS 2019, UAI 2020, NDSS 2021)
    'ALIEAttack',
    'IPMAttack',
    'MinMaxAttack',
    'create_blades_attack'
]

