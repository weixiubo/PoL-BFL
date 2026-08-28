"""
ShapleyFL Aggregator

Implementation of ShapleyFL defense mechanism based on Shapley Value.
Paper: "ShapleyFL: Robust Federated Learning Based on Shapley Value" (KDD 2023)

ShapleyFL uses game-theoretic Shapley values to assess client contributions,
which can defend against:
- Byzantine attacks (by identifying low-contribution malicious clients)
- Free-riding attacks (by detecting clients with zero/low contribution)

The adapter evaluates Monte Carlo coalitions against an explicit validation
model or a caller-supplied utility function.
"""

import copy
import torch
import numpy as np
import logging
from typing import Callable, List, Optional, Tuple
from collections import OrderedDict
from ..base.baseAggregator import ServerAggregator

logger = logging.getLogger(__name__)


class ShapleyFLAggregator(ServerAggregator):
    """
    ShapleyFL Aggregator using Monte Carlo Shapley Value estimation
    
    Key features:
    - Computes Shapley values to measure client contributions
    - Filters out clients with negative/low Shapley values
    - Aggregates using weighted average based on Shapley values
    """
    
    def __init__(self, 
                 num_mc_samples: int = 10,
                 threshold_percentile: float = 0.0,
                 validation_data: Tuple = None,
                 device: str = 'cpu',
                 model: Optional[torch.nn.Module] = None,
                 evaluation_fn: Optional[Callable[[OrderedDict], float]] = None,
                 seed: int = 1337):
        """
        Initialize ShapleyFL aggregator
        
        Args:
            num_mc_samples: Number of Monte Carlo samples for Shapley value estimation
            threshold_percentile: Percentile threshold for filtering (0-100)
                - 0: Keep all clients with non-negative Shapley values
                - 50: Keep top 50% clients
            validation_data: Tuple of (val_loader, criterion) for validation
            device: Device for computation ('cpu' or 'cuda')
            model: Model architecture used to evaluate coalition state dictionaries
            evaluation_fn: Optional direct state-dictionary utility function
            seed: Monte Carlo permutation seed
        """
        if num_mc_samples <= 0:
            raise ValueError("num_mc_samples must be positive")
        if not 0.0 <= threshold_percentile <= 100.0:
            raise ValueError("threshold_percentile must be in [0, 100]")
        super().__init__(model=model)
        self.num_mc_samples = num_mc_samples
        self.threshold_percentile = threshold_percentile
        self.validation_data = validation_data
        self.device = device
        self.evaluation_fn = evaluation_fn
        self.rng = np.random.default_rng(seed)
        
        # Track Shapley values over rounds
        self.shapley_history = []
        
        logger.info(f"Initialized ShapleyFL with {num_mc_samples} MC samples, "
                   f"threshold={threshold_percentile}%")
    
    def _on_before_aggregation(self, raw_client_model_or_grad_list: List[OrderedDict]) -> List[OrderedDict]:
        """No preprocessing needed"""
        return raw_client_model_or_grad_list
    
    def _on_after_aggregation(self, aggregated_model):
        """Return the Shapley-weighted model unchanged."""
        return aggregated_model
    
    def test(self, test_data=None, device=None, args=None):
        """Evaluate the configured global model on validation or supplied data."""
        if self.model is None:
            raise RuntimeError("ShapleyFL test requires a model")
        if test_data is None:
            if self.validation_data is None:
                raise RuntimeError("ShapleyFL test requires validation data")
            test_data, criterion = self.validation_data
        else:
            criterion = (args or {}).get('criterion')
        previous_device = self.device
        if device is not None:
            self.device = str(device)
        try:
            return self._evaluate_model(
                self.model.state_dict(), test_data, criterion
            )
        finally:
            self.device = previous_device
    
    def _aggregate_alg(self, raw_client_model_or_grad_list: List[OrderedDict] = None) -> OrderedDict:
        """
        Aggregate using ShapleyFL
        
        Args:
            raw_client_model_or_grad_list: List of client model updates
        
        Returns:
            aggregated_model: Aggregated model using Shapley value weighting
        """
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
        
        num_clients = len(raw_client_model_or_grad_list)
        
        if num_clients == 0:
            logger.warning("No client models to aggregate")
            return OrderedDict()
        
        # If validation data is not available, fall back to FedAvg
        if self.validation_data is None and self.evaluation_fn is None:
            logger.warning("No validation data provided, falling back to FedAvg")
            return self._fedavg_aggregate(raw_client_model_or_grad_list)
        
        # Compute Shapley values
        shapley_values = self._compute_shapley_values(raw_client_model_or_grad_list)
        
        # Store Shapley values for analysis
        self.shapley_history.append(shapley_values.copy())
        
        # Filter clients based on Shapley values
        selected_indices, selected_weights = self._filter_clients(shapley_values)
        
        if len(selected_indices) == 0:
            logger.warning("No clients selected after filtering, using all clients")
            selected_indices = list(range(num_clients))
            selected_weights = np.ones(num_clients) / num_clients
        
        # Aggregate selected clients with Shapley value weights
        aggregated_model = self._weighted_aggregate(
            raw_client_model_or_grad_list,
            selected_indices,
            selected_weights
        )
        
        logger.info(f"ShapleyFL: Selected {len(selected_indices)}/{num_clients} clients, "
                   f"Shapley values: mean={np.mean(shapley_values):.4f}, "
                   f"std={np.std(shapley_values):.4f}")
        
        return aggregated_model
    
    def _compute_shapley_values(self, models: List[OrderedDict]) -> np.ndarray:
        """
        Compute Shapley values using Monte Carlo estimation
        
        Args:
            models: List of client models
        
        Returns:
            shapley_values: Array of Shapley values for each client
        """
        num_clients = len(models)
        shapley_values = np.zeros(num_clients)
        
        # Get validation data
        if self.validation_data is None:
            val_loader, criterion = None, None
        else:
            val_loader, criterion = self.validation_data
        
        # Baseline accuracy (no clients)
        baseline_acc = (
            self._evaluate_model(self.model.state_dict(), val_loader, criterion)
            if self.model is not None
            else 0.0
        )
        
        # Monte Carlo sampling
        for _ in range(self.num_mc_samples):
            # Random permutation of clients
            perm = self.rng.permutation(num_clients)
            
            # Track accuracy as we add clients
            prev_acc = baseline_acc
            
            for j in range(1, num_clients + 1):
                # Get models for first j clients in permutation
                coalition_indices = perm[:j]
                coalition_models = [models[i] for i in coalition_indices]
                
                # Aggregate coalition
                coalition_model = self._fedavg_aggregate(coalition_models)
                
                # Evaluate coalition
                curr_acc = self._evaluate_model(coalition_model, val_loader, criterion)
                
                # Marginal contribution of client perm[j-1]
                marginal_contrib = curr_acc - prev_acc
                shapley_values[perm[j-1]] += marginal_contrib
                
                prev_acc = curr_acc
        
        # Average over MC samples
        shapley_values /= self.num_mc_samples
        
        return shapley_values
    
    def _filter_clients(self, shapley_values: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Filter clients based on Shapley values
        
        Args:
            shapley_values: Array of Shapley values
        
        Returns:
            selected_indices: Indices of selected clients
            selected_weights: Normalized weights for selected clients
        """
        num_clients = len(shapley_values)
        
        # Filter by threshold percentile
        if self.threshold_percentile > 0:
            threshold = np.percentile(shapley_values, self.threshold_percentile)
        else:
            threshold = 0.0  # Keep all non-negative
        
        # Select clients above threshold
        selected_indices = [i for i in range(num_clients) if shapley_values[i] >= threshold]
        
        if len(selected_indices) == 0:
            return [], np.array([])
        
        # Compute weights (normalize positive Shapley values)
        selected_shapley = np.array([shapley_values[i] for i in selected_indices])
        
        # Shift to make all values positive if needed
        if np.min(selected_shapley) < 0:
            selected_shapley = selected_shapley - np.min(selected_shapley)
        
        # Normalize to sum to 1
        if np.sum(selected_shapley) > 0:
            selected_weights = selected_shapley / np.sum(selected_shapley)
        else:
            selected_weights = np.ones(len(selected_indices)) / len(selected_indices)
        
        return selected_indices, selected_weights
    
    def _weighted_aggregate(self, 
                           models: List[OrderedDict], 
                           indices: List[int], 
                           weights: np.ndarray) -> OrderedDict:
        """
        Weighted aggregation of selected models
        
        Args:
            models: All client models
            indices: Indices of selected clients
            weights: Weights for selected clients
        
        Returns:
            aggregated_model: Weighted aggregated model
        """
        aggregated_model = OrderedDict()
        
        # Get first model to determine structure
        first_model = models[indices[0]]
        
        for key in first_model.keys():
            # Weighted sum
            weighted_sum = sum(
                weights[i] * models[indices[i]][key] 
                for i in range(len(indices))
            )
            aggregated_model[key] = weighted_sum
        
        return aggregated_model
    
    def _fedavg_aggregate(self, models: List[OrderedDict]) -> OrderedDict:
        """
        Simple FedAvg aggregation (fallback)
        
        Args:
            models: List of client models
        
        Returns:
            aggregated_model: Averaged model
        """
        if len(models) == 0:
            return OrderedDict()
        
        aggregated_model = OrderedDict()
        num_clients = len(models)
        
        for key in models[0].keys():
            aggregated_model[key] = sum(model[key] for model in models) / num_clients
        
        return aggregated_model
    
    def _evaluate_model(self, model: OrderedDict, val_loader, criterion) -> float:
        """
        Evaluate model on validation data
        
        Args:
            model: Model state dict
            val_loader: Validation data loader
            criterion: Loss criterion
        
        Returns:
            accuracy: Validation accuracy
        """
        if self.evaluation_fn is not None:
            utility = float(self.evaluation_fn(model))
            if not np.isfinite(utility):
                raise ValueError("Shapley utility must be finite")
            return utility
        if self.model is None or val_loader is None:
            raise RuntimeError(
                "Shapley evaluation requires a model and validation loader"
            )

        evaluated = copy.deepcopy(self.model).to(self.device)
        evaluated.load_state_dict(model, strict=True)
        evaluated.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise ValueError("validation batches must contain inputs and labels")
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                logits = evaluated(inputs)
                predictions = logits.argmax(dim=1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
        if total == 0:
            raise ValueError("validation loader is empty")
        return correct / total
    
    def get_shapley_history(self) -> List[np.ndarray]:
        """Get history of Shapley values across rounds"""
        return self.shapley_history
