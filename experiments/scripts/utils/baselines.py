"""
Baseline Methods for Comparison

Implements various baseline aggregation methods for comparison with PoL-FL.
"""

import torch
import logging
import numpy as np
import os
from typing import List, Dict, OrderedDict
from collections import OrderedDict as ODict

logger = logging.getLogger(__name__)


def _max_vector_dim() -> int:
    try:
        return max(1, int(os.getenv("ROBUST_VECTOR_MAX_DIM", "50000")))
    except Exception:
        return 50000


def _flatten_model_sampled(model: ODict, max_dim: int = None) -> torch.Tensor:
    """Flatten model state and deterministically subsample for robust scorers."""
    max_dim = _max_vector_dim() if max_dim is None else int(max_dim)
    parts = []
    for key in sorted(model.keys()):
        tensor = model[key]
        if not torch.is_tensor(tensor):
            continue
        vec = tensor.detach().float().cpu().reshape(-1)
        if not torch.isfinite(vec).all():
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        parts.append(vec)
    flat = torch.cat(parts) if parts else torch.empty(0)
    if flat.numel() > max_dim:
        idx = torch.linspace(0, flat.numel() - 1, steps=max_dim).long()
        flat = flat[idx]
    return flat


class BaselineAggregator:
    """Base class for baseline aggregation methods"""
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        logger.info(f"Initialized {method_name}")
    
    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Aggregate models
        
        Args:
            models: List of model state dicts
            weights: List of aggregation weights (default: equal weights)
        
        Returns:
            aggregated_model: Aggregated model state dict
        """
        raise NotImplementedError


class VanillaFLAggregator(BaselineAggregator):
    """
    Vanilla Federated Learning (FedAvg)
    
    Simple weighted average of client models.
    """
    
    def __init__(self):
        super().__init__("Vanilla_FL")
    
    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        FedAvg: Weighted average of models
        
        Args:
            models: List of model state dicts
            weights: List of weights (default: equal weights)
        
        Returns:
            aggregated_model: Averaged model
        """
        if not models:
            raise ValueError("No models to aggregate")
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Initialize aggregated model
        aggregated_model = ODict()
        
        # Weighted average
        for key in models[0].keys():
            aggregated_model[key] = sum(
                w * model[key] for w, model in zip(weights, models)
            )
        
        logger.debug(f"Aggregated {len(models)} models using FedAvg")
        return aggregated_model


class KrumAggregator(BaselineAggregator):
    """
    Krum Aggregation
    
    Byzantine-robust aggregation that selects the model closest to others.
    Reference: Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    """
    
    def __init__(self, num_byzantine: int = 0):
        super().__init__("Krum")
        self.num_byzantine = num_byzantine
    
    def _compute_distance(self, model1: ODict, model2: ODict) -> float:
        """Compute L2 distance between two models"""
        distance = 0.0
        for key in model1.keys():
            distance += torch.sum((model1[key] - model2[key]) ** 2).item()
        return np.sqrt(distance)
    
    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Krum: Select model with smallest sum of distances to closest models
        
        Args:
            models: List of model state dicts
            weights: Not used in Krum
        
        Returns:
            selected_model: Selected model (not averaged)
        """
        if not models:
            raise ValueError("No models to aggregate")
        
        n = len(models)
        f = self.num_byzantine
        
        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_distance(models[i], models[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each model, compute sum of distances to n-f-2 closest models
        scores = []
        for i in range(n):
            # Sort distances for this model
            sorted_distances = np.sort(distances[i])
            # Sum of n-f-2 smallest distances (excluding self)
            score = np.sum(sorted_distances[1:n-f-1])
            scores.append(score)
        
        # Select model with smallest score
        selected_idx = np.argmin(scores)
        
        logger.debug(f"Krum selected model {selected_idx} (score={scores[selected_idx]:.4f})")
        return models[selected_idx]


class TrimmedMeanAggregator(BaselineAggregator):
    """
    Trimmed Mean Aggregation
    
    Byzantine-robust aggregation that removes extreme values before averaging.
    Reference: Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        super().__init__("Trimmed_Mean")
        self.trim_ratio = trim_ratio
    
    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Trimmed Mean: Remove extreme values and average the rest

        Args:
            models: List of model state dicts
            trim_ratio: Ratio of values to trim from each end

        Returns:
            aggregated_model: Trimmed mean model
        """
        if not models:
            raise ValueError("No models to aggregate")

        n = len(models)
        # 修复: 确保至少trim 1个值，避免trim_ratio太小时num_trim=0
        num_trim = max(1, int(n * self.trim_ratio))

        # 确保trim后至少剩下1个模型
        if num_trim >= n // 2:
            num_trim = max(0, n // 2 - 1)

        aggregated_model = ODict()

        for key in models[0].keys():
            # Stack parameters from all models
            params = torch.stack([model[key] for model in models])

            # Sort along the model dimension
            sorted_params, _ = torch.sort(params, dim=0)

            # Trim extreme values
            if num_trim > 0 and num_trim < n // 2:
                trimmed_params = sorted_params[num_trim:-num_trim]
            else:
                trimmed_params = sorted_params

            # Average trimmed parameters
            # Convert to float if needed to avoid dtype issues
            if trimmed_params.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                trimmed_params = trimmed_params.float()
            aggregated_model[key] = torch.mean(trimmed_params, dim=0)

        logger.debug(f"Aggregated {len(models)} models using Trimmed Mean (trim_ratio={self.trim_ratio}, num_trim={num_trim})")
        return aggregated_model


class MedianAggregator(BaselineAggregator):
    """
    Median Aggregation
    
    Byzantine-robust aggregation using coordinate-wise median.
    """
    
    def __init__(self):
        super().__init__("Median")
    
    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Median: Coordinate-wise median of models

        Args:
            models: List of model state dicts

        Returns:
            aggregated_model: Median model
        """
        if not models:
            raise ValueError("No models to aggregate")

        aggregated_model = ODict()

        for key in models[0].keys():
            # Stack parameters from all models
            params = torch.stack([model[key] for model in models])

            # Compute coordinate-wise median
            # Note: torch.median on CUDA doesn't have deterministic implementation
            # Move to CPU for deterministic behavior when needed
            original_device = params.device
            if params.is_cuda and torch.are_deterministic_algorithms_enabled():
                params_cpu = params.cpu()
                median_result = torch.median(params_cpu, dim=0)[0]
                aggregated_model[key] = median_result.to(original_device)
            else:
                aggregated_model[key] = torch.median(params, dim=0)[0]

        logger.debug(f"Aggregated {len(models)} models using Median")
        return aggregated_model


class BulyanAggregator(BaselineAggregator):
    """
    Bulyan Aggregation
    
    Combines Krum with trimmed mean for stronger Byzantine robustness.
    Reference: El Mhamdi et al., "The Hidden Vulnerability of Distributed Learning in Byzantium"
    """
    
    def __init__(self, num_byzantine: int = 0):
        super().__init__("Bulyan")
        self.num_byzantine = num_byzantine
        self.krum = KrumAggregator(num_byzantine)
    
    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        Bulyan: Apply Krum multiple times, then trimmed mean
        
        Args:
            models: List of model state dicts
        
        Returns:
            aggregated_model: Bulyan aggregated model
        """
        if not models:
            raise ValueError("No models to aggregate")
        
        n = len(models)
        f = self.num_byzantine
        
        # Select n-2f models using Krum
        selected_models = []
        remaining_models = models.copy()
        
        for _ in range(n - 2 * f):
            # Run Krum on remaining models
            selected = self.krum.aggregate(remaining_models)
            selected_models.append(selected)
            
            # Remove selected model from remaining
            remaining_models = [m for m in remaining_models if m is not selected]
        
        # Apply trimmed mean on selected models
        trimmed_mean = TrimmedMeanAggregator(trim_ratio=0.1)
        aggregated_model = trimmed_mean.aggregate(selected_models)
        
        logger.debug(f"Aggregated {len(models)} models using Bulyan")
        return aggregated_model


class ShapleyFLAggregator(BaselineAggregator):
    """
    ShapleyFL Aggregator (Simplified for Experiments)

    Uses Shapley values to assess client contributions and filter malicious clients.
    Paper: "ShapleyFL: Robust Federated Learning Based on Shapley Value" (KDD 2023)

    Note: This is a simplified version that doesn't require validation data.
    It uses gradient similarity as a proxy for contribution.
    """

    def __init__(self, threshold_percentile: float = 0.0):
        super().__init__("ShapleyFL")
        self.threshold_percentile = threshold_percentile

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        ShapleyFL: Filter clients based on contribution similarity

        Args:
            models: List of model state dicts
            weights: Not used

        Returns:
            aggregated_model: Filtered and averaged model
        """
        if not models:
            raise ValueError("No models to aggregate")

        # Compute pairwise cosine similarity as proxy for contribution
        from sklearn.metrics.pairwise import cosine_similarity

        # Flatten models to vectors
        model_vectors = []
        for model in models:
            model_vectors.append(_flatten_model_sampled(model).numpy())

        model_matrix = np.array(model_vectors)

        # Compute similarity to mean (proxy for positive contribution)
        mean_vec = np.mean(model_matrix, axis=0, keepdims=True)

        # Check for NaN/Inf in mean vector
        if np.any(np.isnan(mean_vec)) or np.any(np.isinf(mean_vec)):
            logger.warning(f"ShapleyFL: NaN/Inf detected in mean vector, falling back to FedAvg")
            # Fallback to simple averaging
            aggregated_model = ODict()
            for key in models[0].keys():
                aggregated_model[key] = sum(model[key] for model in models) / len(models)
            return aggregated_model

        try:
            similarities = cosine_similarity(model_matrix, mean_vec).flatten()
        except (ValueError, RuntimeError) as e:
            logger.warning(f"ShapleyFL: Failed to compute cosine similarity ({e}), falling back to FedAvg")
            # Fallback to simple averaging
            aggregated_model = ODict()
            for key in models[0].keys():
                aggregated_model[key] = sum(model[key] for model in models) / len(models)
            return aggregated_model

        # Check for NaN/Inf in similarities
        if np.any(np.isnan(similarities)) or np.any(np.isinf(similarities)):
            logger.warning(f"ShapleyFL: NaN/Inf detected in similarities, falling back to FedAvg")
            # Fallback to simple averaging
            aggregated_model = ODict()
            for key in models[0].keys():
                aggregated_model[key] = sum(model[key] for model in models) / len(models)
            return aggregated_model

        # Filter by threshold
        if self.threshold_percentile > 0:
            threshold = np.percentile(similarities, self.threshold_percentile)
        else:
            threshold = 0.0

        selected_indices = [i for i in range(len(models)) if similarities[i] >= threshold]

        if len(selected_indices) == 0:
            logger.warning("ShapleyFL: No clients selected, using all")
            selected_indices = list(range(len(models)))

        # Aggregate selected models with similarity weights
        selected_sims = np.array([similarities[i] for i in selected_indices])
        if np.sum(selected_sims) > 0:
            selected_weights = selected_sims / np.sum(selected_sims)
        else:
            selected_weights = np.ones(len(selected_indices)) / len(selected_indices)

        aggregated_model = ODict()
        for key in models[0].keys():
            aggregated_model[key] = sum(
                selected_weights[i] * models[selected_indices[i]][key]
                for i in range(len(selected_indices))
            )

        logger.debug(f"ShapleyFL: Selected {len(selected_indices)}/{len(models)} clients")
        return aggregated_model


class FoolsGoldAggregator(BaselineAggregator):
    """
    FoolsGold Aggregator (Simplified for Experiments)

    Detects Sybil attacks by analyzing gradient similarity.
    Paper: "Defending Against Sybils in Federated Learning" (RAID 2020)

    Note: This is a simplified version without cumulative history.
    """

    def __init__(self):
        super().__init__("FoolsGold")
        self.summed_deltas = None

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        """
        FoolsGold: Weight clients inversely to gradient similarity

        Args:
            models: List of model state dicts
            weights: Not used

        Returns:
            aggregated_model: FoolsGold weighted model
        """
        if not models:
            raise ValueError("No models to aggregate")

        from sklearn.metrics.pairwise import cosine_similarity

        # Flatten and normalize models
        model_vectors = []
        for model in models:
            vec = _flatten_model_sampled(model).numpy()
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            else:
                # Handle zero-norm case
                vec = np.zeros_like(vec)
            model_vectors.append(vec)

        model_matrix = np.array(model_vectors)

        # Compute FoolsGold weights
        n = len(models)
        epsilon = 1e-5

        # Cosine similarity
        cs = cosine_similarity(model_matrix) - np.eye(n)
        # Check for NaN in cosine similarity
        if not np.isfinite(cs).all():
            logger.warning("FoolsGold: Non-finite values in cosine similarity, replacing with zeros")
            cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)

        # Pardoning mechanism
        maxcs = np.max(cs, axis=1) + epsilon
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

        # Compute weights
        wv = 1 - np.max(cs, axis=1)
        wv = np.clip(wv, 0, 1)

        # Normalize
        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        wv[wv >= 0.99] = 0.99  # Avoid exactly 1.0 to prevent division issues

        # Logit function with numerical stability
        wv_safe = np.clip(wv, epsilon, 1 - epsilon)  # Ensure values are in valid range
        wv = np.log((wv_safe / (1 - wv_safe)) + epsilon) + 0.5
        wv = np.clip(wv, 0, 1)

        # Final NaN check
        if not np.isfinite(wv).all():
            logger.warning("FoolsGold: Non-finite weights detected, falling back to uniform weights")
            wv = np.ones(n) / n
        else:
            # Normalize weights
            if np.sum(wv) > 0:
                wv = wv / np.sum(wv)
            else:
                wv = np.ones(n) / n

        # Aggregate
        aggregated_model = ODict()
        for key in models[0].keys():
            aggregated_model[key] = sum(
                wv[i] * models[i][key]
                for i in range(len(models))
            )

        logger.debug(f"FoolsGold: weights mean={np.mean(wv):.4f}, std={np.std(wv):.4f}")
        return aggregated_model


class SDEAAggregator(BaselineAggregator):
    """
    Clean-room SDEA-style robust aggregation.

    This implementation uses a distribution-estimation filter: estimate the
    central update by coordinate-wise median, score clients by distance to that
    center, drop the highest-scoring f clients, then average the selected set.
    """

    def __init__(self, num_byzantine: int = 0):
        super().__init__("SDEA")
        self.num_byzantine = max(0, int(num_byzantine))
        self.selected_indices = []

    def _flatten_model(self, model: ODict) -> torch.Tensor:
        return _flatten_model_sampled(model)

    def aggregate(self, models: List[ODict], weights: List[float] = None) -> ODict:
        if not models:
            raise ValueError("No models to aggregate")
        if len(models) == 1:
            self.selected_indices = [0]
            return models[0]

        vectors = torch.stack([self._flatten_model(model) for model in models], dim=0)
        center = torch.median(vectors, dim=0).values
        distances = torch.norm(vectors - center.unsqueeze(0), p=2, dim=1)

        n = len(models)
        keep = max(1, n - min(self.num_byzantine, n - 1))
        selected = torch.argsort(distances)[:keep].tolist()
        self.selected_indices = [int(i) for i in selected]

        if weights is not None:
            selected_weights = [float(weights[i]) for i in self.selected_indices]
            total = sum(selected_weights)
            if total <= 0:
                selected_weights = [1.0 / len(self.selected_indices)] * len(self.selected_indices)
            else:
                selected_weights = [w / total for w in selected_weights]
        else:
            selected_weights = [1.0 / len(self.selected_indices)] * len(self.selected_indices)

        aggregated_model = ODict()
        for key in models[0].keys():
            aggregated_model[key] = sum(
                selected_weights[pos] * models[idx][key]
                for pos, idx in enumerate(self.selected_indices)
            )

        logger.debug(f"SDEA: selected {len(self.selected_indices)}/{len(models)} clients")
        return aggregated_model


# Factory function for creating aggregators
def create_aggregator(method: str, **kwargs) -> BaselineAggregator:
    """
    Factory function to create aggregator instances

    Args:
        method: Aggregation method name
        **kwargs: Method-specific parameters

    Returns:
        aggregator: Aggregator instance
    """
    aggregators = {
        'Vanilla_FL': VanillaFLAggregator,
        'Krum': KrumAggregator,
        'Trimmed_Mean': TrimmedMeanAggregator,
        'Median': MedianAggregator,
        'Bulyan': BulyanAggregator,
        'ShapleyFL': ShapleyFLAggregator,
        'FoolsGold': FoolsGoldAggregator,
        'SDEA': SDEAAggregator,
    }

    if method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {method}")

    return aggregators[method](**kwargs)
