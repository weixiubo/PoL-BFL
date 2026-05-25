"""
Byzantine Attack Implementations

Implements various Byzantine attacks for security evaluation.
"""

import torch
import logging
from typing import Dict, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)


class ByzantineAttack:
    """Base class for Byzantine attacks"""
    
    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        logger.info(f"Initialized {attack_name}")
    
    def apply(self, model_state: OrderedDict, **kwargs) -> OrderedDict:
        """
        Apply attack to model state
        
        Args:
            model_state: Model state dict
            **kwargs: Additional attack parameters
        
        Returns:
            attacked_state: Attacked model state
        """
        raise NotImplementedError


class RandomNoiseAttack(ByzantineAttack):
    """
    Random Noise Attack
    
    Malicious clients send a model perturbed by parameter-scale random noise.
    """
    
    def __init__(self, noise_scale: float = 1.0, min_scale: float = 1e-4):
        super().__init__("RandomNoiseAttack")
        self.noise_scale = noise_scale
        self.min_scale = min_scale
    
    def apply(self, model_state: OrderedDict, **kwargs) -> OrderedDict:
        """
        Perturb model parameters with scale-normalized random noise.

        Args:
            model_state: Original model state
            global_model: Optional reference model state

        Returns:
            attacked_state: Model with random noise
        """
        attacked_state = OrderedDict()
        reference_state = kwargs.get("global_model") or model_state

        for key, param in model_state.items():
            # Only add noise to floating tensors; keep ints as-is (e.g., num_batches_tracked)
            if not (torch.is_tensor(param) and param.is_floating_point()):
                attacked_state[key] = param
                continue

            base = reference_state.get(key, param) if isinstance(reference_state, dict) else param
            base = base.to(device=param.device, dtype=param.dtype)
            scale_source = base.detach().float()
            scale_value = float(scale_source.std(unbiased=False).item()) if scale_source.numel() > 1 else 0.0
            if not torch.isfinite(torch.tensor(scale_value)) or scale_value < self.min_scale:
                mean_abs = float(scale_source.abs().mean().item()) if scale_source.numel() > 0 else 0.0
                scale_value = max(mean_abs, self.min_scale)

            noise = torch.randn_like(param) * (self.noise_scale * scale_value)
            attacked = base + noise
            if key.endswith("running_var"):
                attacked = attacked.clamp_min(self.min_scale)
            attacked_state[key] = attacked

        logger.debug(f"Applied scale-normalized random noise attack (scale={self.noise_scale})")
        return attacked_state


class LabelFlippingAttack(ByzantineAttack):
    """
    Label Flipping Attack
    
    Malicious clients flip labels during training.
    This attack is applied during training, not to the model directly.
    """
    
    def __init__(self, flip_probability: float = 1.0):
        super().__init__("LabelFlippingAttack")
        self.flip_probability = flip_probability
    
    def flip_labels(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Flip labels randomly
        
        Args:
            labels: Original labels
            num_classes: Number of classes
        
        Returns:
            flipped_labels: Flipped labels
        """
        flipped_labels = labels.clone()
        
        # Randomly select labels to flip
        flip_mask = torch.rand(labels.size()) < self.flip_probability
        
        # Flip selected labels to random classes
        random_labels = torch.randint(0, num_classes, labels.size())
        flipped_labels[flip_mask] = random_labels[flip_mask]
        
        return flipped_labels
    
    def apply(self, model_state: OrderedDict, **kwargs) -> OrderedDict:
        """
        For label flipping, we return the model as-is.
        The actual attack happens during training.
        """
        logger.debug("Label flipping attack (applied during training)")
        return model_state


class ModelReplacementAttack(ByzantineAttack):
    """
    Model Replacement Attack
    
    Malicious clients submit a pre-trained wrong model.
    """
    
    def __init__(self, replacement_type: str = 'random'):
        super().__init__("ModelReplacementAttack")
        self.replacement_type = replacement_type
    
    def apply(self, model_state: OrderedDict, **kwargs) -> OrderedDict:
        """
        Replace model with a malicious one
        
        Args:
            model_state: Original model state
            replacement_type: Type of replacement ('random', 'zero', 'previous')
        
        Returns:
            attacked_state: Replaced model
        """
        attacked_state = OrderedDict()
        
        if self.replacement_type == 'random':
            # Random initialization (float tensors get randn; non-float fall back to zeros)
            for key, param in model_state.items():
                if torch.is_tensor(param) and param.is_floating_point():
                    attacked_state[key] = torch.randn_like(param)
                else:
                    attacked_state[key] = torch.zeros_like(param)

        elif self.replacement_type == 'zero':
            # Zero initialization
            for key, param in model_state.items():
                attacked_state[key] = torch.zeros_like(param)
        
        elif self.replacement_type == 'previous':
            # Use previous global model (from kwargs)
            previous_model = kwargs.get('previous_model', model_state)
            attacked_state = previous_model
        
        else:
            raise ValueError(f"Unknown replacement type: {self.replacement_type}")
        
        logger.debug(f"Applied model replacement attack (type={self.replacement_type})")
        return attacked_state


class GradientInversionAttack(ByzantineAttack):
    """
    Gradient Inversion Attack
    
    Malicious clients send inverted gradients to slow down convergence.
    """
    
    def __init__(self, inversion_scale: float = -1.0):
        super().__init__("GradientInversionAttack")
        self.inversion_scale = inversion_scale
    
    def apply(self, model_state: OrderedDict, **kwargs) -> OrderedDict:
        """
        Invert model updates
        
        Args:
            model_state: Updated model state
            inversion_scale: Scale for inversion (negative to invert)
        
        Returns:
            attacked_state: Model with inverted updates
        """
        global_model = kwargs.get('global_model')
        if global_model is None:
            raise ValueError("GradientInversionAttack requires 'global_model' to be provided")
        
        attacked_state = OrderedDict()
        
        for key in model_state.keys():
            a = model_state[key]
            b = global_model[key]
            if not (torch.is_tensor(a) and a.is_floating_point() and torch.is_tensor(b) and b.is_floating_point()):
                attacked_state[key] = a
                continue
            # Compute update: delta = model - global_model
            delta = a - b
            # Invert update
            inverted_delta = delta * self.inversion_scale
            # Apply inverted update
            attacked_state[key] = b + inverted_delta

        logger.debug(f"Applied gradient inversion attack (scale={self.inversion_scale})")
        return attacked_state


class BackdoorAttack(ByzantineAttack):
    """
    Backdoor Attack
    
    Malicious clients inject backdoor triggers into the model.
    This is a more sophisticated attack that requires trigger patterns.
    """
    
    def __init__(self, trigger_pattern: str = 'pixel'):
        super().__init__("BackdoorAttack")
        self.trigger_pattern = trigger_pattern
    
    def apply(self, model_state: OrderedDict, **kwargs) -> OrderedDict:
        """
        For backdoor attack, the model is trained with poisoned data.
        This method returns the backdoored model.
        """
        logger.debug(f"Backdoor attack (trigger={self.trigger_pattern})")
        # The actual backdoor is injected during training
        return model_state


# Factory function for creating attacks
def create_attack(attack_type: str, **kwargs) -> ByzantineAttack:
    """
    Factory function to create attack instances

    Args:
        attack_type: Type of attack
        **kwargs: Attack-specific parameters

    Returns:
        attack: Attack instance
    """
    # Import Blades attacks
    from .blades_attacks import ALIEAttack, IPMAttack, MinMaxAttack

    attacks = {
        # Original Byzantine attacks
        'random_noise': RandomNoiseAttack,
        'label_flipping': LabelFlippingAttack,
        'model_replacement': ModelReplacementAttack,
        'gradient_inversion': GradientInversionAttack,
        'backdoor': BackdoorAttack,
        # Blades framework attacks (A-class conferences)
        'alie': ALIEAttack,  # NeurIPS 2019
        'ipm': IPMAttack,  # UAI 2020
        'minmax': MinMaxAttack,  # NDSS 2021
    }

    if attack_type not in attacks:
        raise ValueError(f"Unknown attack type: {attack_type}")

    return attacks[attack_type](**kwargs)
