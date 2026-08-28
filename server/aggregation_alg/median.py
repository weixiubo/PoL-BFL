from ..base.baseAggregator import ServerAggregator
import numpy as np
import torch
from collections import OrderedDict

class medianAggregator(ServerAggregator):
    def __init__(self):
        super().__init__()
        
    def _on_before_aggregation(self, raw_client_model_or_grad_list):
        return raw_client_model_or_grad_list
        
    def _on_after_aggregation(self, aggregated_model_or_grad):
        return aggregated_model_or_grad
        
    def test(self, test_data=None, device=None, args=None):
        return {'pooled_updates': len(self.model_pool)}
        
    def _aggregate_alg(self, raw_client_model_or_grad_list=None):
        if raw_client_model_or_grad_list is None:
            raw_client_model_or_grad_list = self.model_pool
        if not raw_client_model_or_grad_list:
            raise ValueError("coordinate median requires at least one client update")
        if isinstance(raw_client_model_or_grad_list[0], dict):
            keys = raw_client_model_or_grad_list[0].keys()
            if any(tuple(model) != tuple(keys) for model in raw_client_model_or_grad_list):
                raise ValueError("median client updates have incompatible keys")
            aggregated_model = OrderedDict()
            
            for key in keys:
                values = [model[key] for model in raw_client_model_or_grad_list]
                if torch.is_tensor(values[0]):
                    if any(value.shape != values[0].shape for value in values):
                        raise ValueError("median client tensors have incompatible shapes")
                    aggregated_model[key] = torch.median(
                        torch.stack(values), dim=0
                    ).values
                else:
                    aggregated_model[key] = np.median(
                        np.stack([np.asarray(value) for value in values]), axis=0
                    )
                
            return aggregated_model
            
        elif isinstance(raw_client_model_or_grad_list[0], list):
            # For gradient lists
            grads = np.array(raw_client_model_or_grad_list)
            # Compute median along the client axis
            median_grad = np.median(grads, axis=0)
            return median_grad.tolist()
        raise TypeError("median updates must be mappings or gradient lists")
