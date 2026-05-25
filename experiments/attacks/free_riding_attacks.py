"""
Free-Riding Attack Implementations

Implements free-riding attacks where clients try to benefit without contributing.
"""

import torch
import logging
from typing import Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class FreeRidingAttack:
    """Base class for free-riding attacks"""
    
    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        logger.info(f"Initialized {attack_name}")
    
    def should_train(self) -> bool:
        """
        Determine if the client should train
        
        Returns:
            should_train: True if client should train
        """
        raise NotImplementedError


class NoTrainingAttack(FreeRidingAttack):
    """
    No Training Attack
    
    Malicious clients don't train at all and submit the global model.
    """
    
    def __init__(self):
        super().__init__("NoTrainingAttack")
    
    def should_train(self) -> bool:
        """Don't train at all"""
        return False
    
    def apply(self, global_model: OrderedDict) -> OrderedDict:
        """
        Return global model without training
        
        Args:
            global_model: Global model state
        
        Returns:
            model: Same as global model (no training)
        """
        logger.debug("No training attack: returning global model")
        return global_model


class LazyTrainingAttack(FreeRidingAttack):
    """
    Lazy Training Attack
    
    Malicious clients train for fewer epochs than required.
    """
    
    def __init__(self, lazy_epochs: int = 1, required_epochs: int = 5):
        super().__init__("LazyTrainingAttack")
        self.lazy_epochs = lazy_epochs
        self.required_epochs = required_epochs
    
    def should_train(self) -> bool:
        """Train but with fewer epochs"""
        return True
    
    def get_training_epochs(self) -> int:
        """
        Get number of epochs to train
        
        Returns:
            epochs: Number of epochs (less than required)
        """
        return self.lazy_epochs
    
    def apply(self, model_state: OrderedDict) -> OrderedDict:
        """
        For lazy training, the model is trained with fewer epochs.
        This method just returns the model.
        """
        logger.debug(f"Lazy training attack: {self.lazy_epochs}/{self.required_epochs} epochs")
        return model_state


class MinimalUpdateAttack(FreeRidingAttack):
    """
    Minimal Update Attack
    
    Malicious clients add minimal noise to global model to appear as if they trained.
    """
    
    def __init__(self, noise_scale: float = 1e-5):
        super().__init__("MinimalUpdateAttack")
        self.noise_scale = noise_scale
    
    def should_train(self) -> bool:
        """Don't train, just add noise"""
        return False
    
    def apply(self, global_model: OrderedDict) -> OrderedDict:
        """
        Add minimal noise to global model
        
        Args:
            global_model: Global model state
        
        Returns:
            model: Global model with minimal noise
        """
        noisy_model = OrderedDict()
        
        for key, param in global_model.items():
            if not (torch.is_tensor(param) and param.is_floating_point()):
                noisy_model[key] = param
                continue
            noise = torch.randn_like(param) * self.noise_scale
            noisy_model[key] = param + noise
        
        logger.debug(f"Minimal update attack: noise_scale={self.noise_scale}")
        return noisy_model


class DataPoisoningAttack(FreeRidingAttack):
    """
    Data Poisoning Attack
    
    Malicious clients use poisoned data to degrade model performance.
    """
    
    def __init__(self, poison_ratio: float = 0.1):
        super().__init__("DataPoisoningAttack")
        self.poison_ratio = poison_ratio
    
    def should_train(self) -> bool:
        """Train with poisoned data"""
        return True
    
    def poison_data(self, data: torch.Tensor, labels: torch.Tensor) -> tuple:
        """
        Poison a portion of the data
        
        Args:
            data: Training data
            labels: Training labels
        
        Returns:
            poisoned_data, poisoned_labels: Poisoned data and labels
        """
        num_samples = len(data)
        num_poison = int(num_samples * self.poison_ratio)
        
        # Randomly select samples to poison
        poison_indices = torch.randperm(num_samples)[:num_poison]
        
        poisoned_data = data.clone()
        poisoned_labels = labels.clone()
        
        # Flip labels for poisoned samples
        num_classes = labels.max().item() + 1
        poisoned_labels[poison_indices] = torch.randint(0, num_classes, (num_poison,))
        
        logger.debug(f"Poisoned {num_poison}/{num_samples} samples")
        return poisoned_data, poisoned_labels
    
    def apply(self, model_state: OrderedDict) -> OrderedDict:
        """
        For data poisoning, the attack happens during training.
        This method just returns the model.
        """
        logger.debug(f"Data poisoning attack: poison_ratio={self.poison_ratio}")
        return model_state


class SelectiveParticipationAttack(FreeRidingAttack):
    """
    Selective Participation Attack
    
    Malicious clients only participate when it's beneficial (e.g., when rewards are high).
    """
    
    def __init__(self, participation_threshold: float = 0.5):
        super().__init__("SelectiveParticipationAttack")
        self.participation_threshold = participation_threshold
    
    def should_participate(self, expected_reward: float, training_cost: float) -> bool:
        """
        Decide whether to participate based on expected utility
        
        Args:
            expected_reward: Expected reward for participation
            training_cost: Cost of training
        
        Returns:
            participate: True if should participate
        """
        utility = expected_reward - training_cost
        return utility > self.participation_threshold
    
    def should_train(self) -> bool:
        """Train if participating"""
        return True
    
    def apply(self, model_state: OrderedDict) -> OrderedDict:
        """
        For selective participation, the decision is made before training.
        This method just returns the model.
        """
        logger.debug("Selective participation attack")
        return model_state


# Factory function for creating attacks
def create_free_riding_attack(attack_type: str, **kwargs) -> FreeRidingAttack:
    """
    Factory function to create free-riding attack instances
    
    Args:
        attack_type: Type of attack
        **kwargs: Attack-specific parameters
    
    Returns:
        attack: Attack instance
    """
    attacks = {
        'no_training': NoTrainingAttack,
        'lazy_training': LazyTrainingAttack,
        'minimal_update': MinimalUpdateAttack,
        'data_poisoning': DataPoisoningAttack,
        'selective_participation': SelectiveParticipationAttack
    }
    
    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    return attacks[attack_type](**kwargs)
