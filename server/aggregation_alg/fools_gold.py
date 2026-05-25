"""
FoolsGold Aggregator

Implementation of FoolsGold defense mechanism for Sybil attacks.
Paper: "Defending Against Sybils in Federated Learning" (RAID 2020)

FoolsGold detects Sybil attacks by analyzing gradient similarity:
- Computes pairwise cosine similarity of client gradients
- Assigns lower weights to clients with high similarity (likely Sybils)
- Uses a pardoning mechanism to avoid penalizing honest clients

This is adapted for PoL-BFL framework.
"""

import torch
import numpy as np
import logging
from typing import List, Dict
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from ..base.baseAggregator import ServerAggregator

logger = logging.getLogger(__name__)


class FoolsGoldAggregator(ServerAggregator):
    """
    FoolsGold Aggregator for Sybil attack defense
    
    Key features:
    - Tracks cumulative gradient history for each client
    - Computes cosine similarity between client gradients
    - Assigns weights inversely proportional to similarity (Sybil detection)
    - Uses pardoning mechanism to avoid false positives
    """
    
    def __init__(self, 
                 use_memory: bool = True,
                 learning_rate: float = 1.0,
                 clip_threshold: int = 0):
        """
        Initialize FoolsGold aggregator
        
        Args:
            use_memory: Whether to use cumulative gradient history
            learning_rate: Learning rate for aggregation
            clip_threshold: Number of clients to clip (0 = no clipping)
        """
        super().__init__()
        self.use_memory = use_memory
        self.learning_rate = learning_rate
        self.clip_threshold = clip_threshold
        
        # Track cumulative gradients for each client
        self.summed_deltas = None
        self.num_clients = 0
        self.num_params = 0
        
        # Track weights history
        self.weights_history = []
        
        logger.info(f"Initialized FoolsGold with memory={use_memory}, "
                   f"lr={learning_rate}, clip={clip_threshold}")
    
    def _on_before_aggregation(self, raw_client_model_or_grad_list: List[OrderedDict]) -> List[OrderedDict]:
        """No preprocessing needed"""
        return raw_client_model_or_grad_list
    
    def _on_after_aggregation(self):
        """No postprocessing needed"""
        pass
    
    def test(self):
        """Test method (placeholder)"""
        pass
    
    def _aggregate_alg(self, raw_client_model_or_grad_list: List[OrderedDict] = None) -> OrderedDict:
        """
        Aggregate using FoolsGold
        
        Args:
            raw_client_model_or_grad_list: List of client model updates
        
        Returns:
            aggregated_model: Aggregated model using FoolsGold weighting
        """
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
        
        num_clients = len(raw_client_model_or_grad_list)
        
        if num_clients == 0:
            logger.warning("No client models to aggregate")
            return OrderedDict()
        
        # Convert models to delta matrix (num_clients x num_params)
        delta_matrix = self._models_to_matrix(raw_client_model_or_grad_list)
        
        # Initialize summed_deltas if first round
        if self.summed_deltas is None:
            self.num_clients = num_clients
            self.num_params = delta_matrix.shape[1]
            self.summed_deltas = np.zeros((num_clients, self.num_params))
        
        # Normalize deltas (important for FoolsGold)
        normalized_deltas = self._normalize_deltas(delta_matrix)
        
        # Update cumulative gradients
        if self.use_memory:
            self.summed_deltas += normalized_deltas
            history_deltas = self.summed_deltas
        else:
            history_deltas = normalized_deltas
        
        # Compute FoolsGold weights
        fg_weights = self._compute_foolsgold_weights(history_deltas)
        
        # Store weights for analysis
        self.weights_history.append(fg_weights.copy())
        
        # Aggregate with FoolsGold weights
        aggregated_model = self._weighted_aggregate(
            raw_client_model_or_grad_list,
            fg_weights
        )
        
        logger.info(f"FoolsGold: weights mean={np.mean(fg_weights):.4f}, "
                   f"std={np.std(fg_weights):.4f}, "
                   f"min={np.min(fg_weights):.4f}, max={np.max(fg_weights):.4f}")
        
        return aggregated_model
    
    def _models_to_matrix(self, models: List[OrderedDict]) -> np.ndarray:
        """
        Convert list of model OrderedDicts to numpy matrix
        
        Args:
            models: List of model state dicts
        
        Returns:
            delta_matrix: Matrix of shape (num_clients, num_params)
        """
        num_clients = len(models)
        
        # Flatten each model to 1D array
        flattened_models = []
        for model in models:
            params = []
            for key in sorted(model.keys()):
                if torch.is_tensor(model[key]):
                    params.append(model[key].cpu().numpy().flatten())
                else:
                    params.append(np.array(model[key]).flatten())
            flattened_models.append(np.concatenate(params))
        
        return np.array(flattened_models)
    
    def _normalize_deltas(self, delta_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize each client's delta to unit norm
        
        Args:
            delta_matrix: Matrix of deltas (num_clients x num_params)
        
        Returns:
            normalized_deltas: Normalized delta matrix
        """
        normalized = delta_matrix.copy()
        
        for i in range(len(normalized)):
            norm = np.linalg.norm(normalized[i])
            if norm > 1e-10:  # Avoid division by zero
                normalized[i] = normalized[i] / norm
        
        return normalized
    
    def _compute_foolsgold_weights(self, summed_deltas: np.ndarray) -> np.ndarray:
        """
        Compute FoolsGold weights based on gradient similarity
        
        Args:
            summed_deltas: Cumulative gradients (num_clients x num_params)
        
        Returns:
            weights: FoolsGold weights for each client
        """
        num_clients = len(summed_deltas)
        epsilon = 1e-5
        
        # Compute pairwise cosine similarity
        cs = cosine_similarity(summed_deltas) - np.eye(num_clients)
        
        # Pardoning mechanism: reweight by max similarity
        maxcs = np.max(cs, axis=1) + epsilon
        
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        
        # Compute weights: 1 - max_similarity
        wv = 1 - np.max(cs, axis=1)
        wv = np.clip(wv, 0, 1)
        
        # Rescale so that max value is 1
        if np.max(wv) > 0:
            wv = wv / np.max(wv)
        
        # Avoid weight of exactly 1 (numerical stability)
        wv[wv == 1] = 0.99
        
        # Apply logit function for smoother weighting
        wv = np.log((wv / (1 - wv)) + epsilon) + 0.5
        wv = np.clip(wv, 0, 1)
        
        # Optional: Clip worst clients (Krum-style)
        if self.clip_threshold > 0:
            # Compute Krum scores
            krum_scores = self._get_krum_scores(summed_deltas, num_clients - self.clip_threshold)
            bad_idx = np.argpartition(krum_scores, num_clients - self.clip_threshold)[
                (num_clients - self.clip_threshold):num_clients
            ]
            wv[bad_idx] = 0
        
        return wv
    
    def _get_krum_scores(self, deltas: np.ndarray, groupsize: int) -> np.ndarray:
        """
        Compute Krum scores for clipping
        
        Args:
            deltas: Client deltas (num_clients x num_params)
            groupsize: Number of neighbors to consider
        
        Returns:
            krum_scores: Krum score for each client
        """
        num_clients = len(deltas)
        krum_scores = np.zeros(num_clients)
        
        # Compute pairwise distances
        distances = (
            np.sum(deltas**2, axis=1)[:, None] +
            np.sum(deltas**2, axis=1)[None] -
            2 * np.dot(deltas, deltas.T)
        )
        
        for i in range(num_clients):
            # Sum of distances to (groupsize-1) nearest neighbors
            krum_scores[i] = np.sum(np.sort(distances[i])[1:groupsize])
        
        return krum_scores
    
    def _weighted_aggregate(self, models: List[OrderedDict], weights: np.ndarray) -> OrderedDict:
        """
        Weighted aggregation of models
        
        Args:
            models: List of client models
            weights: FoolsGold weights for each client
        
        Returns:
            aggregated_model: Weighted aggregated model
        """
        aggregated_model = OrderedDict()
        
        # Normalize weights to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            normalized_weights = weights / weight_sum
        else:
            # Fallback to equal weights
            normalized_weights = np.ones(len(models)) / len(models)
        
        # Get first model to determine structure
        first_model = models[0]
        
        for key in first_model.keys():
            # Weighted sum
            weighted_sum = sum(
                normalized_weights[i] * models[i][key]
                for i in range(len(models))
            )
            aggregated_model[key] = weighted_sum
        
        return aggregated_model
    
    def get_weights_history(self) -> List[np.ndarray]:
        """Get history of FoolsGold weights across rounds"""
        return self.weights_history
    
    def reset_memory(self):
        """Reset cumulative gradient memory"""
        self.summed_deltas = None
        self.weights_history = []
        logger.info("FoolsGold memory reset")

