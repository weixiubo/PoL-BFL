"""
Blades Framework Attack Implementations

Implements Byzantine attacks from the Blades benchmark suite.
These attacks are from A-class conference attack papers:
- ALIE: "A Little Is Enough: Circumventing Defenses" (NeurIPS 2019)
- IPM: "Fall of Empires: Breaking Byzantine-tolerant SGD" (UAI 2020)
- MinMax: "Manipulating the Byzantine" (NDSS 2021)

These attacks are PoL-detectable because they do not perform local training.
"""

import torch
import logging
from typing import Dict, Any, List, Optional
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class BladesAttack:
    """
    Base class for Blades framework attacks

    Note: These attacks are designed to work without benign_updates by using
    fallback strategies (e.g., random noise) when benign updates are not available.
    """

    def __init__(self, attack_name: str):
        self.attack_name = attack_name
        logger.info(f"Initialized {attack_name} from Blades framework")

    def apply(self, model_state: OrderedDict, global_model: OrderedDict = None,
              benign_updates: List[OrderedDict] = None, **kwargs) -> OrderedDict:
        """
        Apply attack to model state

        Args:
            model_state: Current model state (not used for most Blades attacks)
            global_model: Global model state dict (for compatibility with existing interface)
            benign_updates: List of benign client updates (optional, will use fallback if None)
            **kwargs: Additional attack parameters

        Returns:
            attacked_state: Attacked model state
        """
        raise NotImplementedError


class ALIEAttack(BladesAttack):
    """
    ALIE Attack (A Little Is Enough)

    Paper: "A Little Is Enough: Circumventing Defenses For Distributed Learning"
    Conference: NeurIPS 2019
    Authors: Baruch et al.

    Attack strategy:
    - Compute mean and standard deviation of benign updates
    - Construct malicious update as: mean + z_max * std
    - z_max is chosen to maximize attack impact while evading detection

    PoL detectability: yes; the attack computes statistics without local training.
    """

    def __init__(self, z_max: float = 2.5):
        """
        Initialize ALIE attack

        Args:
            z_max: Maximum z-score for attack (default: 2.5)
                - Higher values = stronger attack but easier to detect
                - Lower values = weaker attack but harder to detect
        """
        super().__init__("ALIEAttack")
        self.z_max = z_max
        logger.info(f"ALIE attack initialized with z_max={z_max}")

    def apply(self, model_state: OrderedDict, global_model: OrderedDict = None,
              benign_updates: List[OrderedDict] = None, **kwargs) -> OrderedDict:
        """
        Apply ALIE attack

        Args:
            model_state: Current model state (not used)
            global_model: Global model state dict (for compatibility)
            benign_updates: List of benign client updates (optional)
            **kwargs: Additional parameters

        Returns:
            attacked_update: Malicious update
        """
        if not benign_updates:
            logger.warning("No benign updates provided, returning random noise")
            return self._random_fallback(model_state)

        # Convert list of OrderedDicts to tensor for computation
        benign_tensors = self._updates_to_tensor(benign_updates)

        # Compute mean and std
        mean = benign_tensors.mean(dim=0)
        std = benign_tensors.std(dim=0)

        # Construct malicious update: mean + z_max * std
        malicious_tensor = mean + self.z_max * std

        # Convert back to OrderedDict
        attacked_update = self._tensor_to_update(malicious_tensor, model_state)

        logger.debug(f"ALIE attack applied with z_max={self.z_max}")
        return attacked_update

    def _updates_to_tensor(self, updates: List[OrderedDict]) -> torch.Tensor:
        """Convert list of OrderedDicts to tensor"""
        # Flatten each update to 1D tensor
        flattened = []
        for update in updates:
            params = []
            for key in sorted(update.keys()):
                params.append(update[key].flatten())
            flattened.append(torch.cat(params))

        return torch.stack(flattened)

    def _tensor_to_update(self, tensor: torch.Tensor, template: OrderedDict) -> OrderedDict:
        """Convert tensor back to OrderedDict using template structure"""
        result = OrderedDict()
        offset = 0

        for key in sorted(template.keys()):
            param_shape = template[key].shape
            param_size = template[key].numel()

            # Extract corresponding part from tensor
            param_flat = tensor[offset:offset + param_size]
            result[key] = param_flat.reshape(param_shape)

            offset += param_size

        return result

    def _random_fallback(self, model_state: OrderedDict) -> OrderedDict:
        """Fallback to random noise if no benign updates"""
        result = OrderedDict()
        for key, param in model_state.items():
            result[key] = torch.randn_like(param)
        return result


class IPMAttack(BladesAttack):
    """
    IPM Attack (Inner Product Manipulation)

    Paper: "Fall of Empires: Breaking Byzantine-tolerant SGD by Inner Product Manipulation"
    Conference: UAI 2020
    Authors: Xie et al.

    Attack strategy:
    - Compute mean of benign updates
    - Return negative of the mean (scaled)
    - This maximizes the inner product manipulation

    PoL detectability: yes; the attack computes a negative mean without local training.
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize IPM attack

        Args:
            scale: Scale factor for attack (default: 1.0)
                - Higher values = stronger attack
        """
        super().__init__("IPMAttack")
        self.scale = scale
        logger.info(f"IPM attack initialized with scale={scale}")

    def apply(self, model_state: OrderedDict, global_model: OrderedDict = None,
              benign_updates: List[OrderedDict] = None, **kwargs) -> OrderedDict:
        """
        Apply IPM attack

        Args:
            model_state: Current model state (not used)
            global_model: Global model state dict (for compatibility)
            benign_updates: List of benign client updates (optional)
            **kwargs: Additional parameters

        Returns:
            attacked_update: Malicious update (negative of mean)
        """
        if not benign_updates:
            logger.warning("No benign updates provided, returning random noise")
            return self._random_fallback(model_state)

        # Convert list of OrderedDicts to tensor
        benign_tensors = self._updates_to_tensor(benign_updates)

        # Compute mean
        mean = benign_tensors.mean(dim=0)

        # Return negative of mean (scaled)
        malicious_tensor = -self.scale * mean

        # Convert back to OrderedDict
        attacked_update = self._tensor_to_update(malicious_tensor, model_state)

        logger.debug(f"IPM attack applied with scale={self.scale}")
        return attacked_update

    def _updates_to_tensor(self, updates: List[OrderedDict]) -> torch.Tensor:
        """Convert list of OrderedDicts to tensor"""
        flattened = []
        for update in updates:
            params = []
            for key in sorted(update.keys()):
                params.append(update[key].flatten())
            flattened.append(torch.cat(params))
        return torch.stack(flattened)

    def _tensor_to_update(self, tensor: torch.Tensor, template: OrderedDict) -> OrderedDict:
        """Convert tensor back to OrderedDict"""
        result = OrderedDict()
        offset = 0
        for key in sorted(template.keys()):
            param_shape = template[key].shape
            param_size = template[key].numel()
            param_flat = tensor[offset:offset + param_size]
            result[key] = param_flat.reshape(param_shape)
            offset += param_size
        return result

    def _random_fallback(self, model_state: OrderedDict) -> OrderedDict:
        """Fallback to random noise"""
        result = OrderedDict()
        for key, param in model_state.items():
            result[key] = torch.randn_like(param)
        return result


class MinMaxAttack(BladesAttack):
    """
    MinMax Attack (Distance Maximization)

    Paper: "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning"
    Conference: NDSS 2021
    Authors: Shejwalkar et al.

    Attack strategy:
    - Use binary search to find optimal malicious update
    - Maximize distance from benign updates while evading detection
    - Construct update as: mean - lambda * deviation

    PoL detectability: yes; the attack uses a binary search over statistics
    without local training.
    """

    def __init__(self, lambda_init: float = 1.0, num_iterations: int = 10):
        """
        Initialize MinMax attack

        Args:
            lambda_init: Initial lambda value for binary search (default: 1.0)
            num_iterations: Number of binary search iterations (default: 10)
        """
        super().__init__("MinMaxAttack")
        self.lambda_init = lambda_init
        self.num_iterations = num_iterations
        logger.info(f"MinMax attack initialized with lambda_init={lambda_init}, iterations={num_iterations}")

    def apply(self, model_state: OrderedDict, global_model: OrderedDict = None,
              benign_updates: List[OrderedDict] = None, **kwargs) -> OrderedDict:
        """
        Apply MinMax attack

        Args:
            model_state: Current model state (not used)
            global_model: Global model state dict (for compatibility)
            benign_updates: List of benign client updates (optional)
            **kwargs: Additional parameters (can include 'aggregator_type')

        Returns:
            attacked_update: Malicious update
        """
        if not benign_updates:
            logger.warning("No benign updates provided, returning random noise")
            return self._random_fallback(model_state)

        # Convert to tensor
        benign_tensors = self._updates_to_tensor(benign_updates)
        mean = benign_tensors.mean(dim=0)
        deviation = benign_tensors.std(dim=0)

        # Binary search for optimal lambda
        # Use the configured fixed lambda for deterministic evaluation.
        lambda_opt = self.lambda_init

        # Construct malicious update: mean - lambda * deviation
        malicious_tensor = mean - lambda_opt * deviation

        # Convert back to OrderedDict
        attacked_update = self._tensor_to_update(malicious_tensor, model_state)

        logger.debug(f"MinMax attack applied with lambda={lambda_opt}")
        return attacked_update

    def _updates_to_tensor(self, updates: List[OrderedDict]) -> torch.Tensor:
        """Convert list of OrderedDicts to tensor"""
        flattened = []
        for update in updates:
            params = []
            for key in sorted(update.keys()):
                params.append(update[key].flatten())
            flattened.append(torch.cat(params))
        return torch.stack(flattened)

    def _tensor_to_update(self, tensor: torch.Tensor, template: OrderedDict) -> OrderedDict:
        """Convert tensor back to OrderedDict"""
        result = OrderedDict()
        offset = 0
        for key in sorted(template.keys()):
            param_shape = template[key].shape
            param_size = template[key].numel()
            param_flat = tensor[offset:offset + param_size]
            result[key] = param_flat.reshape(param_shape)
            offset += param_size
        return result

    def _random_fallback(self, model_state: OrderedDict) -> OrderedDict:
        """Fallback to random noise"""
        result = OrderedDict()
        for key, param in model_state.items():
            result[key] = torch.randn_like(param)
        return result


# Factory function for creating Blades attacks
def create_blades_attack(attack_type: str, **kwargs) -> BladesAttack:
    """
    Factory function to create Blades attack instances

    Args:
        attack_type: Type of attack ('alie', 'ipm', 'minmax')
        **kwargs: Attack-specific parameters

    Returns:
        attack: BladesAttack instance
    """
    attacks = {
        'alie': ALIEAttack,
        'ipm': IPMAttack,
        'minmax': MinMaxAttack
    }

    if attack_type.lower() not in attacks:
        raise ValueError(f"Unknown Blades attack type: {attack_type}. Available: {list(attacks.keys())}")

    return attacks[attack_type.lower()](**kwargs)
